/*************************************************************************/
/*         Program name: Overlapping Class Generator, OCG (Bitset/OMP)   */
/*                                                                       */
/*  Based on OCG Copyright (C) 2011 Alain Guenoche                       */
/*  Optimized for large biological networks:                             */
/*   - Hash-table vertex indexing during graph loading (O(M) vs O(M*N))  */
/*   - Bitset A_edge / A_co / C_bits (eliminates 533MB-2.2GB B matrix    */
/*     for unweighted graphs; B_flat kept only when -w weighted is used) */
/*   - Symmetric fus_bits bitset for O(p/64) todo discovery & row-best   */
/*   - Compact active cluster list & nz_words of cl1                     */
/*   - Per-node f(u, cl1) caching across overlapping todo clusters       */
/*   - Incremental modularity + merge replay + race-free OpenMP          */
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MaxProt 30000
#define SupCar 25
#define HT_CAP 65536

FILE *FichCar, *FichPlug;

short *Dg, **Cl, *Clas, *Kard, *Q;
int N = 0, Na = 0, NbCliq, NbClIni, NbClas, DgMax, typ = 3,
        verbose = 0, CardMax = 0, ClCh = -1, FCS = 1, cyto = 0, w = 0;
int W_WORDS = 0, F_WORDS = 0;
double **Var;
double alpha = 1.0, SumDg2 = 0., eps = 0.0001, ModMax;

uint64_t *A_edge_bits = NULL, *A_co_bits = NULL, *C_bits = NULL, *fus_bits = NULL;
double *B_flat = NULL; /* Allocated only when w == 1 (weighted graph) */
int **Adj = NULL;      /* Adjacency lists for fast sparse neighbor checks */

char Et[MaxProt][SupCar], FichE[256], FichCyto[256];

int getopt(int argc, char * const argv[], const char * optstring);
extern char *optarg;
extern int optind;

static inline void bit_set(uint64_t *arr, int idx) {
    arr[idx >> 6] |= (1ULL << (idx & 63));
}
static inline void bit_clear(uint64_t *arr, int idx) {
    arr[idx >> 6] &= ~(1ULL << (idx & 63));
}
static inline int bit_test(const uint64_t *arr, int idx) {
    return (int)((arr[idx >> 6] >> (idx & 63)) & 1ULL);
}

int printHelp(char *argv[]) {
    fprintf(stderr, "\nUSAGE: %s [options] <GRAPH FILENAME>\n", argv[0]);
    fprintf(stderr, "\nOPTIONS:\n");
    fprintf(stderr, "\t-h: Print this help and exit.\n");
    fprintf(stderr, "\t-m: Maximal Cliques.\n");
    fprintf(stderr, "\t-e: Edges.\n");
    fprintf(stderr, "\t-c <0|1>: Centered Cliques (default 1 = maximize modularity).\n");
    fprintf(stderr, "\t-s <size>: Maximum allowed class size (0 = unconstrained).\n");
    fprintf(stderr, "\t-n <count>: Minimum number of expected classes (0 = unconstrained).\n");
    fprintf(stderr, "\t-p <file>: Cytoscape partition output file.\n");
    fprintf(stderr, "\t-w: Weighted graph.\n");
    fprintf(stderr, "\t-v: Verbose progress to stderr.\n");
    exit(0);
}

void readargv(int argc, char *argv[]) {
    int c;
    while ((c = getopt(argc, argv, "hmewc:s:n:a:p:v")) != -1) {
        switch (c) {
        case 'h': printHelp(argv); break;
        case 'm': typ = 1; FCS = 0; break;
        case 'e': typ = 2; FCS = 0; break;
        case 'c':
            typ = 3;
            if (strtol(optarg, NULL, 10) == 0)
                FCS = 0;
            break;
        case 's': CardMax = (int)strtol(optarg, NULL, 10); break;
        case 'n': ClCh = (int)strtol(optarg, NULL, 10); break;
        case 'p':
            snprintf(FichCyto, sizeof(FichCyto), "%s", optarg);
            cyto = 1;
            break;
        case 'w': w = 1; break;
        case 'v': verbose = 1; break;
        }
    }
    if (argv[optind] == 0)
        printHelp(argv);
    if (FCS == 0 && CardMax == 0 && ClCh == -1) {
        fprintf(stderr, "Argument error!\n");
        exit(1);
    }
    if (ClCh == -1)
        ClCh = 2;
}

static int compar(const void *e1, const void *e2) {
    return strcmp((const char *)e1, (const char *)e2);
}

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static int intern_vertex(const char *name, int *ht) {
    uint32_t mask = HT_CAP - 1;
    uint32_t idx = fnv1a(name) & mask;
    while (ht[idx] >= 0) {
        if (strcmp(Et[ht[idx]], name) == 0)
            return ht[idx];
        idx = (idx + 1) & mask;
    }
    if (N >= MaxProt) {
        printf("Too many vertices > %d\n", MaxProt);
        exit(1);
    }
    int id = N++;
    strcpy(Et[id], name);
    ht[idx] = id;
    return id;
}

static int find_vertex(const char *name) {
    char *hit = (char *)bsearch(name, Et, (size_t)N, SupCar, compar);
    if (!hit)
        return -1;
    return (int)((hit - (char *)Et) / SupCar);
}

typedef struct {
    char u[SupCar];
    char v[SupCar];
    float w;
} RawEdge;

void lecgraph(int argc, char *argv[]) {
    char Ch1[2 * SupCar], Ch2[2 * SupCar];
    float pond = 1.0f;

    snprintf(FichE, sizeof(FichE), "%s", argv[argc - 1]);
    FichCar = fopen(FichE, "r");
    if (FichCar == NULL) {
        printf("Could not open file: %s\n", FichE);
        exit(1);
    }

    int *ht = malloc(HT_CAP * sizeof(int));
    assert(ht);
    for (int i = 0; i < HT_CAP; i++)
        ht[i] = -1;

    size_t cap = 4096, n_raw = 0;
    RawEdge *raw = malloc(cap * sizeof(RawEdge));
    assert(raw);

    while (fscanf(FichCar, "%49s %49s", Ch1, Ch2) == 2) {
        if (w == 1) {
            if (fscanf(FichCar, "%f", &pond) != 1 && verbose == 1)
                fprintf(stderr, "error while parsing graph file : weight bad format\n");
        }
        if (strlen(Ch1) >= SupCar || strlen(Ch2) >= SupCar) {
            printf("Labels are limited to %d characters\n", SupCar - 1);
            exit(1);
        }
        intern_vertex(Ch1, ht);
        intern_vertex(Ch2, ht);
        if (strcmp(Ch1, Ch2) == 0)
            continue;
        if (n_raw == cap) {
            cap *= 2;
            raw = realloc(raw, cap * sizeof(RawEdge));
            assert(raw);
        }
        strcpy(raw[n_raw].u, Ch1);
        strcpy(raw[n_raw].v, Ch2);
        raw[n_raw].w = pond;
        n_raw++;
    }
    fclose(FichCar);
    free(ht);

    qsort(Et, (size_t)N, SupCar, compar);

    W_WORDS = (N + 63) / 64;
    size_t nw = (size_t)N * (size_t)W_WORDS;
    A_edge_bits = calloc(nw, sizeof(uint64_t));
    A_co_bits = calloc(nw, sizeof(uint64_t));
    Clas = malloc((size_t)N * sizeof(short));
    Dg = calloc((size_t)N, sizeof(short));
    Adj = malloc((size_t)N * sizeof(int *));
    assert(A_edge_bits && A_co_bits && Clas && Dg && Adj);

    double *W_flat = NULL;
    if (w == 1) {
        size_t nn = (size_t)N * (size_t)N;
        W_flat = calloc(nn, sizeof(double));
        B_flat = malloc(nn * sizeof(double));
        assert(W_flat && B_flat);
    }

    for (size_t e = 0; e < n_raw; e++) {
        int ii = find_vertex(raw[e].u);
        int jj = find_vertex(raw[e].v);
        if (ii < 0 || jj < 0)
            continue;
        uint64_t *row_i = A_edge_bits + (size_t)ii * W_WORDS;
        if (!bit_test(row_i, jj)) {
            bit_set(row_i, jj);
            bit_set(A_edge_bits + (size_t)jj * W_WORDS, ii);
            Dg[ii]++;
            Dg[jj]++;
        }
        if (w == 1) {
            W_flat[(size_t)ii * N + jj] = (double)raw[e].w;
            W_flat[(size_t)jj * N + ii] = (double)raw[e].w;
        }
    }
    free(raw);

    DgMax = 0;
    Na = 0;
    for (int i = 0; i < N; i++) {
        short deg = Dg[i];
        if (deg > DgMax) DgMax = deg;
        Na += deg;
        SumDg2 += 1.0 * deg * deg;
        Adj[i] = malloc((size_t)(deg > 0 ? deg : 1) * sizeof(int));
        assert(Adj[i]);
        const uint64_t *row = A_edge_bits + (size_t)i * W_WORDS;
        int pos = 0;
        for (int word = 0; word < W_WORDS; word++) {
            uint64_t m = row[word];
            while (m) {
                int b = __builtin_ctzll(m);
                m &= m - 1;
                Adj[i][pos++] = (word << 6) + b;
            }
        }
    }

    ModMax = 0.0;
    double alpha_na = alpha * (double)Na;
    double inv_alpha = 1.0 / alpha;
    if (w == 1) {
        for (int i = 0; i < N; i++) {
            B_flat[(size_t)i * N + i] = -1.0 * Dg[i] * Dg[i] * inv_alpha;
            for (int j = 0; j < i; j++) {
                size_t idx_ij = (size_t)i * N + j;
                size_t idx_ji = (size_t)j * N + i;
                double val = alpha_na * W_flat[idx_ij] - 1.0 * Dg[i] * Dg[j] * inv_alpha;
                B_flat[idx_ij] = val;
                B_flat[idx_ji] = val;
                if (bit_test(A_edge_bits + (size_t)i * W_WORDS, j))
                    ModMax += val;
            }
        }
        free(W_flat);
    } else {
        for (int i = 0; i < N; i++) {
            double di = (double)Dg[i] * inv_alpha;
            for (int p = 0; p < Dg[i]; p++) {
                int j = Adj[i][p];
                if (j < i)
                    ModMax += alpha_na - di * (double)Dg[j];
            }
        }
    }
    Na /= 2;

    if (ModMax < eps) {
        fprintf(stderr, "Maximum modularity of the graph is negative.\n");
        exit(1);
    }
}

static inline double pair_B(int u, int v) {
    if (w == 1)
        return B_flat[(size_t)u * N + v];
    double edge_uv = (double)bit_test(A_edge_bits + (size_t)u * W_WORDS, v);
    return alpha * (2.0 * Na) * edge_uv - 1.0 * Dg[u] * Dg[v] / alpha;
}

void clasout(int NbCl, double Mod, double MMod) {
    int i, k, kk, ko = 0;
    fflush(stdout);
    printf("#################################################\n");
    printf("# Graph file %s has %d Vertices and %d edges\n", FichE, N, Na);
    printf("# Degre maximum: %d\n", DgMax);
    printf("# Rate of edges: %.4f\n", 1. * Na / N / (N - 1));
    for (i = 0; i < N; i++) {
        const uint64_t *e_row = A_edge_bits + (size_t)i * W_WORDS;
        const uint64_t *c_row = A_co_bits + (size_t)i * W_WORDS;
        for (int word = 0; word < W_WORDS; word++)
            ko += __builtin_popcountll(e_row[word] & c_row[word]);
    }
    ko /= 2;
    printf("# Rate of intraclass edges: %.3f\n#\n", 1. * ko / Na);
    switch (typ) {
    case 1: printf("# Initial classes are: maximal cliques\n"); break;
    case 2: printf("# Initial classes are: edges\n"); break;
    case 3: printf("# Initial classes are: centered cliques\n"); break;
    }
    if (FCS == 1) printf("# Class System is maximizing the modularity\n");
    if (CardMax != 0) printf("# Maximum Class Cardinality: %d\n", CardMax);
    if (ClCh != 2) printf("# Minimum Number of Classes %d\n", ClCh);
    printf("#################################################\n");

    memset(Clas, 0, (size_t)N * sizeof(short));
    for (k = 0; k < NbCl; k++)
        for (i = 0; i < Kard[k]; i++)
            Clas[Cl[k][i]]++;

    k = 0;
    kk = 0;
    for (i = 0; i < N; i++) {
        if (Clas[i] == 0) kk++;
        else if (Clas[i] > 1) k++;
    }
    if (kk) {
        printf("\nUnclustered nodes (%d): ", kk);
        for (i = 0; i < N; i++)
            if (Clas[i] == 0) printf("%s  ", Et[i]);
        printf("\n");
    } else {
        printf("\nUnclustered nodes (%d): None\n", kk);
    }
    printf("Multiclustered nodes (%d):\n", k);
    for (i = 0; i < N; i++)
        if (Clas[i] > 1) printf("%s (%d), ", Et[i], Clas[i]);
    printf("\n\n\nFinal classes (%d), Modularity = %.3f (%.4f):\n", NbCl, Mod, MMod);
    for (k = 0; k < NbCl; k++) {
        printf(">Class %3d  (%d nodes):\n", k + 1, Kard[k]);
        for (i = 0; i < Kard[k]; i++)
            printf("%s ", Et[Cl[k][i]]);
        printf("\n");
    }
    printf("\n");
}

void clasoutcyto(int NbCl) {
    int i, k, kk;
    const char *base = strrchr(FichE, '/');
    const char *bslash = strrchr(FichE, '\\');
    if (bslash && (!base || bslash > base))
        base = bslash;
    base = base ? base + 1 : FichE;

    FichPlug = fopen(FichCyto, "w");
    fprintf(FichPlug, "#ModClust analysis export\n");
    fprintf(FichPlug, "#Algorithm:OCG\n");
    fprintf(FichPlug, "#Network:%s\n", base);
    fprintf(FichPlug, "#Scope:Network\n");
    switch (typ) {
    case 1: fprintf(FichPlug, "#Parameter:Initial_classes=maximal_cliques\n"); break;
    case 2: fprintf(FichPlug, "#Parameter:Initial_classes=edges\n"); break;
    case 3: fprintf(FichPlug, "#Parameter:Initial_classes=centered_cliques\n"); break;
    }
    if (FCS == 1) fprintf(FichPlug, "#Parameter:Class_System=maximizing_modularity\n");
    if (CardMax != 0) fprintf(FichPlug, "#Parameter:Maximum_Cardinality=%d\n", CardMax);
    if (ClCh != 2) fprintf(FichPlug, "#Parameter:Minimum_Number_of_Classes=%d\n", ClCh);
    fprintf(FichPlug, "\n");

    memset(Clas, 0, (size_t)N * sizeof(short));
    for (k = 0; k < NbCl; k++)
        for (i = 0; i < Kard[k]; i++)
            Clas[Cl[k][i]]++;

    for (k = 0; k < NbCl; k++) {
        fprintf(FichPlug, ">ClusterID:%d||\n", k + 1);
        for (i = 0; i < Kard[k]; i++)
            fprintf(FichPlug, "%s\n", Et[Cl[k][i]]);
        fprintf(FichPlug, "\n");
    }
    kk = 0;
    for (i = 0; i < N; i++)
        if (Clas[i] == 0) kk++;
    if (kk) {
        k = NbCl;
        for (i = 0; i < N; i++) {
            if (Clas[i] == 0) {
                fprintf(FichPlug, ">ClusterID:%d||\n", k + 1);
                fprintf(FichPlug, "%s\n\n", Et[i]);
                k++;
            }
        }
    }
    fclose(FichPlug);
}

int Inclus(void) {
    int k, j, flag;
    for (k = 0; k < NbCliq; k++) {
        if (Cl[k][0] < 0) continue;
        flag = 1;
        for (j = 0; j < N; j++) {
            if (Cl[k][j] < Q[j]) {
                flag = 0;
                break;
            }
        }
        if (flag == 1) return 1;
    }
    for (j = 0; j < N; j++)
        Cl[NbCliq][j] = Q[j];
    NbCliq++;
    return 0;
}

int Fermeture(void) {
    int mis = 0;
    memset(A_co_bits, 0, (size_t)N * W_WORDS * sizeof(uint64_t));
    for (int i = 0; i < N; i++)
        bit_set(A_co_bits + (size_t)i * W_WORDS, i);

    for (int c = 0; c < NbClas; c++) {
        for (int j = 1; j < Kard[c]; j++) {
            int jj = Cl[c][j];
            for (int k = 0; k < j; k++) {
                int kk = Cl[c][k];
                bit_set(A_co_bits + (size_t)jj * W_WORDS, kk);
                bit_set(A_co_bits + (size_t)kk * W_WORDS, jj);
            }
        }
    }
    for (int i = 0; i < N; i++) {
        const uint64_t *e_row = A_edge_bits + (size_t)i * W_WORDS;
        const uint64_t *c_row = A_co_bits + (size_t)i * W_WORDS;
        for (int word = 0; word < W_WORDS; word++)
            mis += __builtin_popcountll(e_row[word] & ~c_row[word]);
    }
    return mis / 2;
}

int ClasArete(void) {
    int k = 0;
    Cl = malloc((size_t)Na * sizeof(short *));
    Kard = malloc((size_t)Na * sizeof(short));
    assert(Cl && Kard);
    for (int i = 0; i < Na; i++) {
        Cl[i] = malloc((size_t)CardMax * sizeof(short));
        assert(Cl[i]);
        Kard[i] = 2;
    }
    for (int i = 0; i < N; i++) {
        for (int p = 0; p < Dg[i]; p++) {
            int j = Adj[i][p];
            if (j < i) {
                Cl[k][0] = (short)j;
                Cl[k][1] = (short)i;
                k++;
            }
        }
    }
    return k;
}

int StarCliq(void) {
    short *X = malloc((size_t)N * sizeof(short));
    short *D = malloc((size_t)N * sizeof(short));
    int *stamp = calloc((size_t)N, sizeof(int));
    int cur_stamp = 0;

    Cl = malloc((size_t)N * sizeof(short *));
    Kard = malloc((size_t)N * sizeof(short));
    assert(Cl && Kard && X && D && stamp);
    for (int i = 0; i < N; i++) {
        Cl[i] = malloc((size_t)N * sizeof(short));
        assert(Cl[i]);
    }

    if (verbose == 1)
        fprintf(stderr, "Calculating Initial class System...");
    for (int i = 0; i < N; i++) {
        if (verbose == 1 && i % 200 == 0)
            fprintf(stderr, ".");

        Cl[NbClas][0] = (short)i;
        int card = 1;
        int NbAdj = Dg[i];
        for (int p = 0; p < NbAdj; p++) {
            X[p] = (short)Adj[i][p];
            D[p] = 1;
        }
        for (int k = 1; k < NbAdj; k++) {
            const uint64_t *row = A_edge_bits + (size_t)X[k] * W_WORDS;
            for (int j = 0; j < k; j++) {
                if (bit_test(row, X[j])) {
                    D[k]++;
                    D[j]++;
                }
            }
        }
        for (int step = 0; step < NbAdj; step++) {
            int maxd = 0, kk = -1;
            for (int k = 0; k < NbAdj; k++) {
                if (D[k] >= maxd) {
                    maxd = D[k];
                    kk = k;
                }
            }
            if (maxd < card)
                break;
            int jj = X[kk];
            D[kk] = 0;
            const uint64_t *row_jj = A_edge_bits + (size_t)jj * W_WORDS;
            int links = 0;
            for (int k = 0; k < card; k++) {
                if (bit_test(row_jj, Cl[NbClas][k]))
                    links++;
            }
            if (links >= card)
                Cl[NbClas][card++] = (short)jj;
        }
        Kard[NbClas] = (short)card;

        int flag = 0;
        for (int k = 0; k < NbClas; k++) {
            if (Kard[k] < card)
                continue;
            cur_stamp++;
            for (int j = 0; j < Kard[k]; j++)
                stamp[Cl[k][j]] = cur_stamp;
            flag = 1;
            for (int j = 0; j < card; j++) {
                if (stamp[Cl[NbClas][j]] != cur_stamp) {
                    flag = 0;
                    break;
                }
            }
            if (flag == 1)
                break;
        }
        if (flag == 0)
            NbClas++;
    }
    if (verbose == 1)
        fprintf(stderr, "Done\nNb. of classes %d\n", NbClas);
    free(X);
    free(D);
    free(stamp);
    int mis = Fermeture();
    if (verbose == 1)
        fprintf(stderr, "Nb. of edges not within the classes %d\n", mis);
    return NbClas;
}

int Clique(void) {
    int i, j, k, kk, NbY, Nl, Long, NewLong, cli, flag1, flag2;
    short *X, *Y;
    Q = malloc((size_t)N * sizeof(short));
    X = malloc((size_t)N * sizeof(short));
    Y = malloc((size_t)N * sizeof(short));
    assert(Q && X && Y);

    Nl = 2 * N;
    Long = Nl;
    Cl = malloc((size_t)Nl * sizeof(short *));
    assert(Cl);
    for (i = 0; i < Nl; i++) {
        Cl[i] = malloc((size_t)N * sizeof(short));
        assert(Cl[i]);
    }

    Cl[0][0] = 1;
    for (j = 1; j < N; j++)
        Cl[0][j] = (short)bit_test(A_edge_bits, j);
    NbCliq = 1;

    for (i = 1; i < N - 1; i++) {
        const uint64_t *Ai = A_edge_bits + (size_t)i * W_WORDS;
        for (j = 0; j < i; j++) X[j] = 0;
        X[i] = 1;
        for (j = i + 1; j < N; j++) X[j] = (short)bit_test(Ai, j);

        cli = 0;
        for (k = 0; k < NbCliq; k++)
            if (Cl[k][i] > 0) cli++;
        if (NbCliq + 2 * cli + 1 > Long) {
            NewLong = Long + Nl;
            Cl = realloc(Cl, (size_t)NewLong * sizeof(short *));
            assert(Cl);
            for (k = Long; k < NewLong; k++) {
                Cl[k] = malloc((size_t)N * sizeof(short));
                assert(Cl[k]);
            }
            Long = NewLong;
        }
        for (k = 0; k < NbCliq; k++) {
            if (Cl[k][i] == 0) continue;
            NbY = 0;
            for (j = i + 1; j < N; j++) {
                if (Cl[k][j] == 1 && X[j] == 0)
                    Y[NbY++] = (short)j;
            }
            if (NbY > 0) {
                for (j = 0; j < N; j++) Q[j] = Cl[k][j];
                Cl[k][0] = -1;
                Q[i] = 0;
                flag1 = Inclus();
                Q[i] = 1;
                for (j = 0; j < NbY; j++) Q[Y[j]] = 0;
                flag2 = Inclus();
                if (flag1 + flag2 == 2) continue;
                for (j = 0; j < N; j++) Cl[k][j] = Cl[NbCliq - 1][j];
                NbCliq--;
            }
        }
        for (j = 0; j < N; j++) Q[j] = X[j];
        Inclus();
    }
    Kard = malloc((size_t)NbCliq * sizeof(short));
    assert(Kard);
    k = 0;
    for (i = 0; i < NbCliq; i++) {
        if (Cl[i][0] < 0) continue;
        kk = 0;
        for (j = 0; j < N; j++)
            if (Cl[i][j]) Cl[k][kk++] = (short)j;
        Kard[k++] = (short)kk;
    }
    NbCliq = k;
    free(Q);
    free(X);
    free(Y);
    return NbCliq;
}

double Modularity(void) {
    double Sum = 0.0;
    for (int i = 1; i < N; i++) {
        const uint64_t *c_row = A_co_bits + (size_t)i * W_WORDS;
        for (int word = 0; word < W_WORDS; word++) {
            uint64_t m = c_row[word];
            while (m) {
                int b = __builtin_ctzll(m);
                m &= m - 1;
                int j = (word << 6) + b;
                if (j < i)
                    Sum += pair_B(i, j);
            }
        }
    }
    return Sum;
}

void Effacer(void) {
    for (int k = 0; k < NbClas; k++) {
        memset(Clas, 0, (size_t)N * sizeof(short));
        double deg_sum_k = 0.0;
        if (w == 0) {
            for (int j = 0; j < Kard[k]; j++)
                deg_sum_k += (double)Dg[Cl[k][j]];
        }
        for (int i = 0; i < Kard[k]; i++) {
            int ii = Cl[k][i];
            double Cont = 0.0;
            if (w == 1) {
                const double *Bii = B_flat + (size_t)ii * N;
                for (int j = 0; j < Kard[k]; j++)
                    Cont += Bii[Cl[k][j]];
            } else {
                const uint64_t *e_row = A_edge_bits + (size_t)ii * W_WORDS;
                int e_in = 0;
                for (int j = 0; j < Kard[k]; j++)
                    e_in += bit_test(e_row, Cl[k][j]);
                Cont = alpha * (2.0 * Na) * e_in - ((double)Dg[ii] * deg_sum_k) / alpha;
            }
            if (Cont)
                Clas[ii] = 1;
        }
        int card = 0;
        for (int i = 0; i < N; i++)
            if (Clas[i])
                Cl[k][card++] = (short)i;
        Kard[k] = (short)card;
    }
}

/* Recompute best column jj < ii using fus_bits[ii] so dead/disconnected cols are skipped in O(p/64) */
static void recompute_row_best_bits(int ii, double *row_best_val, int *row_best_j) {
    if (Kard[ii] == 0) {
        row_best_j[ii] = -1;
        return;
    }
    int best_j = -1;
    double best_v = -1.0e300;
    int ki = Kard[ii];
    const double *Var_ii = Var[ii];
    const uint64_t *fb = fus_bits + (size_t)ii * F_WORDS;
    int max_word = (ii + 63) >> 6;
    for (int word = 0; word < max_word; word++) {
        uint64_t m = fb[word];
        while (m) {
            int b = __builtin_ctzll(m);
            m &= m - 1;
            int jj = (word << 6) + b;
            if (jj >= ii)
                break;
            if (Kard[jj] == 0)
                continue;
            double v = Var_ii[jj] / (double)(ki * Kard[jj]);
            if (best_j < 0 || v > best_v) {
                best_v = v;
                best_j = jj;
            }
        }
    }
    row_best_j[ii] = best_j;
    row_best_val[ii] = best_v;
}

/* Compute exact f(u, cl1) using pre-collected non-zero words of cl1_bits */
static inline double compute_node_to_cluster(
    int u, int cl1, int k_cl1, double ds_cl1, const int *nz_words, int n_nz
) {
    const uint64_t *cl1_bits = C_bits + (size_t)cl1 * W_WORDS;
    const uint64_t *co_u = A_co_bits + (size_t)u * W_WORDS;

    if (w == 1) {
        const double *Bu = B_flat + (size_t)u * N;
        double s = 0.0;
        for (int q = 0; q < n_nz; q++) {
            int word = nz_words[q];
            uint64_t m = cl1_bits[word] & ~co_u[word];
            while (m) {
                int b = __builtin_ctzll(m);
                m &= m - 1;
                s += Bu[(word << 6) + b];
            }
        }
        return s;
    }

    int co_cnt = 0;
    for (int q = 0; q < n_nz; q++) {
        int word = nz_words[q];
        co_cnt += __builtin_popcountll(cl1_bits[word] & co_u[word]);
    }
    if (co_cnt == k_cl1)
        return 0.0;

    int edge_cnt = 0;
    if (Dg[u] <= n_nz * 2) {
        const int *adj_u = Adj[u];
        int du = Dg[u];
        for (int p = 0; p < du; p++) {
            int v = adj_u[p];
            if (bit_test(cl1_bits, v) && !bit_test(co_u, v))
                edge_cnt++;
        }
    } else {
        const uint64_t *edge_u = A_edge_bits + (size_t)u * W_WORDS;
        for (int q = 0; q < n_nz; q++) {
            int word = nz_words[q];
            edge_cnt += __builtin_popcountll((cl1_bits[word] & ~co_u[word]) & edge_u[word]);
        }
    }

    double unco_deg_sum;
    if (co_cnt == 0) {
        unco_deg_sum = ds_cl1;
    } else if (co_cnt < (k_cl1 >> 1)) {
        double co_deg_sum = 0.0;
        for (int q = 0; q < n_nz; q++) {
            int word = nz_words[q];
            uint64_t m = cl1_bits[word] & co_u[word];
            while (m) {
                int b = __builtin_ctzll(m);
                m &= m - 1;
                co_deg_sum += (double)Dg[(word << 6) + b];
            }
        }
        unco_deg_sum = ds_cl1 - co_deg_sum;
    } else {
        double s = 0.0;
        for (int q = 0; q < n_nz; q++) {
            int word = nz_words[q];
            uint64_t m = cl1_bits[word] & ~co_u[word];
            while (m) {
                int b = __builtin_ctzll(m);
                m &= m - 1;
                s += (double)Dg[(word << 6) + b];
            }
        }
        unco_deg_sum = s;
    }

    return alpha * (2.0 * Na) * (double)edge_cnt - ((double)Dg[u] * unco_deg_sum) / alpha;
}

int main(int argc, char *argv[]) {
    readargv(argc, argv);
    lecgraph(argc, argv);

    if (CardMax == 0)
        CardMax = N;

    switch (typ) {
    case 1: NbClIni = Clique(); break;
    case 2: NbClIni = ClasArete(); break;
    case 3: NbClIni = StarCliq(); break;
    }
    if (verbose == 1)
        fprintf(stderr, "Number of initial classes %d\n", NbClIni);

    double Mod = Modularity();
    double BestMod = 0.0;
    double MMod = Mod / Na / Na / 2 - SumDg2 / Na / Na / 4;

    if (verbose == 1)
        fprintf(stderr, "Running...\n");

    F_WORDS = (NbClIni + 63) / 64;
    Var = malloc((size_t)NbClIni * sizeof(double *));
    C_bits = calloc((size_t)NbClIni * (size_t)W_WORDS, sizeof(uint64_t));
    uint64_t *C_nbr_bits = calloc((size_t)NbClIni * (size_t)W_WORDS, sizeof(uint64_t));
    fus_bits = calloc((size_t)NbClIni * (size_t)F_WORDS, sizeof(uint64_t));
    double *deg_sum = calloc((size_t)NbClIni, sizeof(double));
    assert(Var && C_bits && C_nbr_bits && fus_bits && deg_sum);

    for (int i = 0; i < NbClIni; i++) {
        Var[i] = malloc((size_t)NbClIni * sizeof(double));
        assert(Var[i]);
        Var[i][i] = -1.0;
        for (int j = 0; j < i; j++) {
            Var[i][j] = -Mod;
            Var[j][i] = 0.0;
        }
        uint64_t *cb = C_bits + (size_t)i * W_WORDS;
        uint64_t *nb = C_nbr_bits + (size_t)i * W_WORDS;
        double ds = 0.0;
        for (int k = 0; k < Kard[i]; k++) {
            int u = Cl[i][k];
            bit_set(cb, u);
            bit_set(nb, u);
            ds += (double)Dg[u];
            const uint64_t *eu = A_edge_bits + (size_t)u * W_WORDS;
            for (int word = 0; word < W_WORDS; word++)
                nb[word] |= eu[word];
        }
        deg_sum[i] = ds;
    }

    short **InitCl = NULL;
    short *InitKard = NULL;
    int *hist_cl1 = NULL, *hist_cl2 = NULL;
    int n_merges = 0, best_step = 0;
    if (FCS == 1) {
        InitCl = malloc((size_t)NbClIni * sizeof(short *));
        InitKard = malloc((size_t)NbClIni * sizeof(short));
        hist_cl1 = malloc((size_t)NbClIni * sizeof(int));
        hist_cl2 = malloc((size_t)NbClIni * sizeof(int));
        assert(InitCl && InitKard && hist_cl1 && hist_cl2);
        for (int i = 0; i < NbClIni; i++) {
            InitKard[i] = Kard[i];
            InitCl[i] = malloc((size_t)Kard[i] * sizeof(short));
            assert(InitCl[i]);
            memcpy(InitCl[i], Cl[i], (size_t)Kard[i] * sizeof(short));
        }
    }

    /* Parallel Var initialization with nz_words of c2 */
    #pragma omp parallel
    {
        int *local_nz = malloc((size_t)W_WORDS * sizeof(int));
        assert(local_nz);

        #pragma omp for schedule(dynamic, 16)
        for (int c1 = 1; c1 < NbClIni; c1++) {
            int k_c1 = Kard[c1];
            const short *C1 = Cl[c1];
            const uint64_t *cb1 = C_bits + (size_t)c1 * W_WORDS;
            for (int c2 = 0; c2 < c1; c2++) {
                const uint64_t *nb2 = C_nbr_bits + (size_t)c2 * W_WORDS;
                int connected = 0;
                for (int word = 0; word < W_WORDS; word++) {
                    if (cb1[word] & nb2[word]) {
                        connected = 1;
                        break;
                    }
                }
                if (!connected) {
                    Var[c2][c1] = 0.0;
                    continue;
                }

                int k_c2 = Kard[c2];
                const uint64_t *cb2 = C_bits + (size_t)c2 * W_WORDS;
                double fus1 = 1.0;

                if (k_c1 + k_c2 > CardMax) {
                    int union_cnt = 0;
                    for (int word = 0; word < W_WORDS; word++)
                        union_cnt += __builtin_popcountll(cb1[word] | cb2[word]);
                    if (union_cnt > CardMax)
                        fus1 = 0.0;
                }

                double VarMod = 0.0;
                if (fus1 > 0.0) {
                    int n_nz = 0;
                    for (int word = 0; word < W_WORDS; word++) {
                        if (cb2[word])
                            local_nz[n_nz++] = word;
                    }
                    for (int a = 0; a < k_c1; a++)
                        VarMod += compute_node_to_cluster(C1[a], c2, k_c2, deg_sum[c2], local_nz, n_nz);
                }
                Var[c1][c2] = VarMod;
                Var[c2][c1] = fus1;
            }
        }
        free(local_nz);
    }
    free(C_nbr_bits);

    /* Populate symmetric fus_bits from upper triangle of Var */
    for (int i = 1; i < NbClIni; i++) {
        for (int j = 0; j < i; j++) {
            if (Var[j][i] >= eps) {
                bit_set(fus_bits + (size_t)i * F_WORDS, j);
                bit_set(fus_bits + (size_t)j * F_WORDS, i);
            }
        }
    }

    double *row_best_val = malloc((size_t)NbClIni * sizeof(double));
    int *row_best_j = malloc((size_t)NbClIni * sizeof(int));
    int *active = malloc((size_t)NbClIni * sizeof(int));
    int *active_pos = malloc((size_t)NbClIni * sizeof(int));
    int *todo = malloc((size_t)NbClIni * sizeof(int));
    int *nz_words = malloc((size_t)W_WORDS * sizeof(int));
    int *node_mark = calloc((size_t)N, sizeof(int));
    int *uniq_nodes = malloc((size_t)N * sizeof(int));
    double *f_val = malloc((size_t)N * sizeof(double));
    int *merge_stamp = calloc((size_t)N, sizeof(int));
    int cur_merge_stamp = 0, mark_stamp = 0;
    assert(row_best_val && row_best_j && active && active_pos && todo && nz_words && node_mark && uniq_nodes && f_val && merge_stamp);

    int n_active = NbClIni;
    for (int i = 0; i < NbClIni; i++) {
        active[i] = i;
        active_pos[i] = i;
    }

    if (NbClIni > 0) {
        row_best_j[0] = -1;
        row_best_val[0] = -1.0e300;
    }

    #pragma omp parallel for schedule(dynamic, 32)
    for (int ii = 1; ii < NbClIni; ii++)
        recompute_row_best_bits(ii, row_best_val, row_best_j);

    /* Main hierarchical merge loop */
    NbClas = NbClIni;
    while (NbClas > ClCh) {
        double VarMax = -1.0 * ModMax;
        int cl1 = -1, cl2 = -1;

        for (int a_idx = 0; a_idx < n_active; a_idx++) {
            int ii = active[a_idx];
            int bj = row_best_j[ii];
            if (bj >= 0) {
                double val = row_best_val[ii];
                if (val > VarMax || (val == VarMax && (ii < cl2 || (ii == cl2 && bj < cl1)))) {
                    VarMax = val;
                    cl1 = bj;
                    cl2 = ii;
                }
            }
        }
        if (cl1 < 0)
            break;

        /* Incremental modularity update + co-clustering bitset update */
        uint64_t *cb1 = C_bits + (size_t)cl1 * W_WORDS;
        uint64_t *cb2 = C_bits + (size_t)cl2 * W_WORDS;
        for (int a = 0; a < Kard[cl1]; a++) {
            int u = Cl[cl1][a];
            uint64_t *co_u = A_co_bits + (size_t)u * W_WORDS;
            for (int word = 0; word < W_WORDS; word++) {
                uint64_t newly = cb2[word] & ~co_u[word];
                if (newly) {
                    co_u[word] |= newly;
                    uint64_t m = newly;
                    while (m) {
                        int b = __builtin_ctzll(m);
                        m &= m - 1;
                        int v = (word << 6) + b;
                        bit_set(A_co_bits + (size_t)v * W_WORDS, u);
                        Mod += pair_B(u, v);
                    }
                }
            }
        }

        /* Merge Cl[cl2] into Cl[cl1] and update C_bits[cl1] & deg_sum[cl1] */
        cur_merge_stamp++;
        int kk = 0;
        double new_ds = 0.0;
        for (int a = 0; a < Kard[cl1]; a++) {
            int u = Cl[cl1][a];
            if (merge_stamp[u] != cur_merge_stamp) {
                merge_stamp[u] = cur_merge_stamp;
                Cl[cl1][kk++] = (short)u;
                new_ds += (double)Dg[u];
            }
        }
        for (int b = 0; b < Kard[cl2]; b++) {
            int u = Cl[cl2][b];
            if (merge_stamp[u] != cur_merge_stamp) {
                merge_stamp[u] = cur_merge_stamp;
                Cl[cl1][kk++] = (short)u;
                new_ds += (double)Dg[u];
            }
        }
        int n_nz = 0;
        for (int word = 0; word < W_WORDS; word++) {
            uint64_t v = cb1[word] | cb2[word];
            cb1[word] = v;
            if (v)
                nz_words[n_nz++] = word;
        }
        Kard[cl1] = (short)kk;
        deg_sum[cl1] = new_ds;
        Kard[cl2] = 0;
        row_best_j[cl2] = -1;

        /* Remove cl2 from compact active list in O(1) */
        int pos2 = active_pos[cl2];
        int last = active[--n_active];
        active[pos2] = last;
        active_pos[last] = pos2;

        if (FCS == 1) {
            hist_cl1[n_merges] = cl1;
            hist_cl2[n_merges] = cl2;
            n_merges++;
            if (Mod >= BestMod) {
                BestMod = Mod;
                best_step = n_merges;
            }
        }

        /* Extract todo candidates in O(p/64) via fus_bits[cl1] | fus_bits[cl2], clearing cl2 links */
        uint64_t *fb1 = fus_bits + (size_t)cl1 * F_WORDS;
        uint64_t *fb2 = fus_bits + (size_t)cl2 * F_WORDS;
        bit_clear(fb1, cl2);
        bit_clear(fb2, cl1);
        int n_todo = 0, n_uniq = 0;
        mark_stamp++;
        for (int word = 0; word < F_WORDS; word++) {
            uint64_t m = fb1[word] | fb2[word];
            while (m) {
                int b = __builtin_ctzll(m);
                m &= m - 1;
                int ii = (word << 6) + b;
                if (Kard[ii] == 0) {
                    bit_clear(fb1, ii);
                    bit_clear(fb2, ii);
                    continue;
                }
                bit_clear(fus_bits + (size_t)ii * F_WORDS, cl2);
                todo[n_todo++] = ii;
                const short *Cii = Cl[ii];
                int k_ii = Kard[ii];
                for (int a = 0; a < k_ii; a++) {
                    int u = Cii[a];
                    if (node_mark[u] != mark_stamp) {
                        node_mark[u] = mark_stamp;
                        uniq_nodes[n_uniq++] = u;
                    }
                }
            }
            fb2[word] = 0;
        }

        /* Compute f_val[u] ONCE per unique active node u */
        int k_cl1 = Kard[cl1];
        double ds_cl1 = deg_sum[cl1];
        #pragma omp parallel for schedule(guided) if(n_uniq >= 128)
        for (int idx = 0; idx < n_uniq; idx++) {
            int u = uniq_nodes[idx];
            f_val[u] = compute_node_to_cluster(u, cl1, k_cl1, ds_cl1, nz_words, n_nz);
        }

        /* Update Var[cl1, ii] and fus_bits for todo rows */
        #pragma omp parallel for schedule(static) if(n_todo >= 256)
        for (int t = 0; t < n_todo; t++) {
            int ii = todo[t];
            const short *Cii = Cl[ii];
            int k_ii = Kard[ii];

            double can_fuse = 1.0;
            if (k_ii + k_cl1 > CardMax) {
                const uint64_t *cb_ii = C_bits + (size_t)ii * W_WORDS;
                int union_cnt = 0;
                for (int word = 0; word < W_WORDS; word++)
                    union_cnt += __builtin_popcountll(cb_ii[word] | cb1[word]);
                if (union_cnt > CardMax)
                    can_fuse = 0.0;
            }

            double VarMod = 0.0;
            if (can_fuse > 0.0) {
                for (int a = 0; a < k_ii; a++)
                    VarMod += f_val[Cii[a]];
            }

            if (ii > cl1) {
                Var[ii][cl1] = VarMod;
                Var[cl1][ii] = can_fuse;
            } else {
                Var[cl1][ii] = VarMod;
                Var[ii][cl1] = can_fuse;
            }
        }

        for (int t = 0; t < n_todo; t++) {
            int ii = todo[t];
            double cf = (ii > cl1) ? Var[cl1][ii] : Var[ii][cl1];
            if (cf >= eps) {
                bit_set(fb1, ii);
                bit_set(fus_bits + (size_t)ii * F_WORDS, cl1);
            } else {
                bit_clear(fb1, ii);
                bit_clear(fus_bits + (size_t)ii * F_WORDS, cl1);
            }
        }

        /* Maintain per-row best cache over compact active list */
        if (cl1 > 0)
            recompute_row_best_bits(cl1, row_best_val, row_best_j);

        #pragma omp parallel for schedule(guided) if(n_active >= 1024)
        for (int a_idx = 0; a_idx < n_active; a_idx++) {
            int ii = active[a_idx];
            if (ii <= cl1)
                continue;
            int bj = row_best_j[ii];
            if (bj == cl1 || bj == cl2) {
                recompute_row_best_bits(ii, row_best_val, row_best_j);
            } else if (Var[cl1][ii] >= eps) {
                double v = Var[ii][cl1] / (double)(Kard[ii] * k_cl1);
                if (bj < 0 || v > row_best_val[ii] || (v == row_best_val[ii] && cl1 < bj)) {
                    row_best_val[ii] = v;
                    row_best_j[ii] = cl1;
                }
            }
        }

        NbClas--;
        if (NbClas % 10 == 0 && verbose == 1) {
            fprintf(stderr, "Remaining classes: %d of %d          \r", NbClas, NbClIni);
            setvbuf(stderr, (char *)NULL, _IONBF, 0);
        }
    }

    free(row_best_val);
    free(row_best_j);
    free(active);
    free(active_pos);
    free(todo);
    free(nz_words);
    free(node_mark);
    free(uniq_nodes);
    free(f_val);
    free(C_bits);
    free(fus_bits);
    free(deg_sum);

    if (verbose == 1)
        fprintf(stderr, "Remaining classes: None                \n\n");

    if (FCS == 1) {
        if (BestMod < eps)
            best_step = 0;
        for (int i = 0; i < NbClIni; i++) {
            Kard[i] = InitKard[i];
            memcpy(Cl[i], InitCl[i], (size_t)InitKard[i] * sizeof(short));
            free(InitCl[i]);
        }
        free(InitCl);
        free(InitKard);
        for (int s = 0; s < best_step; s++) {
            int c1 = hist_cl1[s], c2 = hist_cl2[s];
            cur_merge_stamp++;
            int kk = 0;
            for (int a = 0; a < Kard[c1]; a++) {
                int u = Cl[c1][a];
                if (merge_stamp[u] != cur_merge_stamp) {
                    merge_stamp[u] = cur_merge_stamp;
                    Cl[c1][kk++] = (short)u;
                }
            }
            for (int b = 0; b < Kard[c2]; b++) {
                int u = Cl[c2][b];
                if (merge_stamp[u] != cur_merge_stamp) {
                    merge_stamp[u] = cur_merge_stamp;
                    Cl[c1][kk++] = (short)u;
                }
            }
            Kard[c1] = (short)kk;
            Kard[c2] = 0;
        }
        free(hist_cl1);
        free(hist_cl2);
    }
    free(merge_stamp);

    int compact = 0;
    for (int k = 0; k < NbClIni; k++) {
        if (Kard[k] == 0) continue;
        if (compact != k) {
            memcpy(Cl[compact], Cl[k], (size_t)Kard[k] * sizeof(short));
            Kard[compact] = Kard[k];
        }
        compact++;
    }
    NbClas = compact;

    Effacer();
    Fermeture();

    Mod = Modularity();
    MMod = Mod / Na / Na / 2 - 1. * SumDg2 / Na / Na / 4;
    clasout(NbClas, Mod, MMod);
    if (cyto)
        clasoutcyto(NbClas);

    return 0;
}
