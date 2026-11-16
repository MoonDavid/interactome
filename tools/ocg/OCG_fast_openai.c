/*************************************************************************/
/*         Overlapping Class Generator, OCG (Fast OpenAI / OpenMP)       */
/*                                                                       */
/*  Based on OCG Copyright (C) 2011 Alain Guenoche                       */
/*  Optimized for large biological networks:                             */
/*   - Incremental modularity update (O(|C1|*|C2|) instead of O(N^2))    */
/*   - Per-row maximum cache for O(p^2) merging instead of O(p^3)        */
/*   - Generation-stamped union sizing (eliminates O(N) inner scans)     */
/*   - OpenMP parallelization for Var initialization and updates         */
/*   - Binary search during graph loading and fixed memory allocations   */
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

FILE *FichCar, *FichPlug;

short *Dg, **Cl, *Clas, *Kard, *Q, **BestCl, *BestKard;
int N = 0, Na = 0, NbCliq, NbClIni, NbClas, DgMax, BestNbClas, typ = 3,
        verbose = 0, CardMax = 0, ClCh = -1, FCS = 1, cyto = 0, w = 0;
double **B, **Var;
double alpha = 1.0, SumDg2 = 0., eps = 0.0001, ModMax;
char **A;
char Et[MaxProt][SupCar], FichE[256], FichCyto[256];

int getopt(int argc, char * const argv[], const char * optstring);
extern char *optarg;
extern int optind;

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
        case 'h':
            printHelp(argv);
            break;
        case 'm':
            typ = 1;
            FCS = 0;
            break;
        case 'e':
            typ = 2;
            FCS = 0;
            break;
        case 'c':
            typ = 3;
            if (strtol(optarg, NULL, 10) == 0)
                FCS = 0;
            break;
        case 's':
            CardMax = (int)strtol(optarg, NULL, 10);
            break;
        case 'n':
            ClCh = (int)strtol(optarg, NULL, 10);
            break;
        case 'p':
            snprintf(FichCyto, sizeof(FichCyto), "%s", optarg);
            cyto = 1;
            break;
        case 'w':
            w = 1;
            break;
        case 'v':
            verbose = 1;
            break;
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

int countlines(FILE *fp) {
    int nl = 0;
    char buf[BUFSIZ];
    int has_nl = 1;
    while (fgets(buf, sizeof buf, fp)) {
        has_nl = (strchr(buf, '\n') != NULL);
        if (has_nl)
            ++nl;
    }
    if (!has_nl)
        ++nl;
    return nl;
}

static int compar(const void *e1, const void *e2) {
    return strcmp((const char *)e1, (const char *)e2);
}

static int find_vertex(const char *name) {
    char *hit = (char *)bsearch(name, Et, (size_t)N, SupCar, compar);
    if (!hit)
        return -1;
    return (int)((hit - (char *)Et) / SupCar);
}

/* Labels are sorted after loading; this temporary hash table avoids an
 * O(edges * vertices) linear search while collecting them. */
static void collect_vertex(const char *name, int *slots) {
    uint32_t hash = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++)
        hash = (hash ^ *p) * 16777619u;
    unsigned int slot = hash & 65535u;
    while (slots[slot]) {
        if (strcmp(name, Et[slots[slot] - 1]) == 0) return;
        slot = (slot + 1) & 65535u;
    }
    if (N == MaxProt) {
        fprintf(stderr, "Too many vertices > %d\n", MaxProt);
        exit(1);
    }
    strcpy(Et[N], name);
    slots[slot] = ++N;
}

void lecgraph(int argc, char *argv[]) {
    int NbCar = 0, i, ii, j, jj, NR, rscan;
    char MaxCar = 0, Ch1[2 * SupCar], Ch2[2 * SupCar], OldCh[SupCar] = "";
    float pond = 1.0f;
    int *slots = calloc(65536, sizeof(int));
    assert(slots);

    snprintf(FichE, sizeof(FichE), "%s", argv[argc - 1]);
    FichCar = fopen(FichE, "r");
    if (FichCar == NULL) {
        printf("Could not open file: %s\n", FichE);
        exit(1);
    }
    NR = countlines(FichCar);
    fclose(FichCar);

    /* First pass: collect unique vertex labels */
    FichCar = fopen(FichE, "r");
    for (i = 0; i < NR; i++) {
        rscan = fscanf(FichCar, "%49s %49s", Ch1, Ch2);
        if (rscan != 2)
            break;
        if (w == 1) {
            if (fscanf(FichCar, "%f", &pond) != 1 && verbose == 1)
                fprintf(stderr, "error while parsing graph file : weight bad format\n");
        }
        int l1 = (int)strlen(Ch1), l2 = (int)strlen(Ch2);
        if (l1 > NbCar) NbCar = l1;
        if (l2 > NbCar) NbCar = l2;
        if (NbCar > MaxCar) MaxCar = (char)NbCar;
        if (MaxCar >= SupCar) {
            printf("Labels are limited to %d characters\n", SupCar - 1);
            exit(1);
        }
        collect_vertex(Ch1, slots);
        collect_vertex(Ch2, slots);
    }
    fclose(FichCar);
    free(slots);
    qsort(Et, (size_t)N, SupCar, compar);

    /* Allocate adjacency, modularity, and weight matrices */
    A = malloc((size_t)N * sizeof(char *));
    B = malloc((size_t)N * sizeof(double *));
    Clas = malloc((size_t)N * sizeof(short));
    Dg = malloc((size_t)N * sizeof(short));
    assert(A && B && Clas && Dg);

    for (i = 0; i < N; i++) {
        A[i] = calloc((size_t)N, sizeof(char));
        /* Store weights in B until degrees are known, then transform in place.
         * The zero diagonal is also read by the final contribution filter. */
        B[i] = calloc((size_t)N, sizeof(double));
        assert(A[i] && B[i]);
    }

    /* Second pass: populate symmetric adjacency and weights using bsearch */
    FichCar = fopen(FichE, "r");
    strcpy(OldCh, "");
    ii = -1;
    for (i = 0; i < NR; i++) {
        rscan = fscanf(FichCar, "%49s %49s", Ch1, Ch2);
        if (rscan != 2)
            break;
        if (w == 1) {
            if (fscanf(FichCar, "%f", &pond) != 1 && verbose == 1)
                fprintf(stderr, "error while parsing graph file : weight bad format\n");
        }
        if (strcmp(Ch1, Ch2) == 0)
            continue;
        if (strcmp(Ch1, OldCh) != 0) {
            ii = find_vertex(Ch1);
            strcpy(OldCh, Ch1);
        }
        jj = find_vertex(Ch2);
        if (ii < 0 || jj < 0)
            continue;
        double weight = (w == 1) ? (double)pond : 1.0;
        B[ii][jj] = weight;
        B[jj][ii] = weight;
        A[ii][jj] = 1;
        A[jj][ii] = 1;
    }
    fclose(FichCar);

    DgMax = 0;
    Na = 0;
    for (i = 0; i < N; i++) {
        short deg = 0;
        for (j = 0; j < N; j++)
            deg += (A[i][j] > 0);
        Dg[i] = deg;
        if (deg > DgMax) DgMax = deg;
        Na += deg;
        SumDg2 += 1.0 * deg * deg;
    }

    ModMax = 0.0;
    for (i = 1; i < N; i++) {
        for (j = 0; j < i; j++) {
            double val = alpha * Na * B[i][j] - 1.0 * Dg[i] * Dg[j] / alpha;
            B[i][j] = val;
            B[j][i] = val;
            if (A[i][j])
                ModMax += val;
        }
    }
    Na /= 2;

    if (ModMax < eps) {
        fprintf(stderr, "Maximum modularity of the graph is negative.\n");
        exit(1);
    }

}

void clasout(int NbCl, double Mod, double MMod) {
    int i, j, k, kk, ko = 0;
    fflush(stdout);
    printf("#################################################\n");
    printf("# Graph file %s has %d Vertices and %d edges\n", FichE, N, Na);
    printf("# Degre maximum: %d\n", DgMax);
    printf("# Rate of edges: %.4f\n", 1. * Na / N / (N - 1));
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            if (A[i][j] && A[j][i])
                ko++;
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
    int i, j, jj, k, kk, mis = 0;
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            A[j][i] = 0;
    for (i = 0; i < NbClas; i++) {
        for (j = 1; j < Kard[i]; j++) {
            jj = Cl[i][j];
            for (k = 0; k < j; k++) {
                kk = Cl[i][k];
                if (kk > jj) A[jj][kk] = 1;
                else if (kk < jj) A[kk][jj] = 1;
            }
        }
    }
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            if (A[i][j] == 1 && A[j][i] == 0)
                mis++;
    return mis;
}

int ClasArete(void) {
    int i, j, k = 0;
    Cl = malloc((size_t)Na * sizeof(short *));
    Kard = malloc((size_t)Na * sizeof(short));
    assert(Cl && Kard);
    for (i = 0; i < Na; i++) {
        Cl[i] = malloc((size_t)CardMax * sizeof(short));
        assert(Cl[i]);
        Kard[i] = 2;
    }
    for (i = 1; i < N; i++) {
        for (j = 0; j < i; j++) {
            if (A[i][j]) {
                Cl[k][0] = (short)j;
                Cl[k][1] = (short)i;
                k++;
            }
        }
    }
    return k;
}

int StarCliq(void) {
    int i, j, jj, k, kk, card, maxd, flag, Adj, NbAdj, mis;
    short *X, *D;
    int *stamp = calloc((size_t)N, sizeof(int));
    int cur_stamp = 0;

    Cl = malloc((size_t)N * sizeof(short *));
    Kard = malloc((size_t)N * sizeof(short));
    X = malloc((size_t)N * sizeof(short));
    D = malloc((size_t)N * sizeof(short));
    assert(Cl && Kard && X && D && stamp);
    for (i = 0; i < N; i++) {
        Cl[i] = malloc((size_t)N * sizeof(short));
        assert(Cl[i]);
    }

    if (verbose == 1)
        fprintf(stderr, "Calculating Initial class System...");
    for (i = 0; i < N; i++) {
        if (verbose == 1 && i % 200 == 0)
            fprintf(stderr, ".");

        Cl[NbClas][0] = (short)i;
        card = 1;
        NbAdj = 0;
        for (j = 0; j < N; j++) {
            if (A[i][j] == 1) {
                X[NbAdj] = (short)j;
                D[NbAdj] = 1;
                NbAdj++;
            }
        }
        for (k = 1; k < NbAdj; k++) {
            int xk = X[k];
            for (j = 0; j < k; j++) {
                if (A[xk][X[j]] == 1) {
                    D[k]++;
                    D[j]++;
                }
            }
        }
        kk = -1;
        for (Adj = 0; Adj < NbAdj; Adj++) {
            maxd = 0;
            for (k = 0; k < NbAdj; k++) {
                if (D[k] >= maxd) {
                    maxd = D[k];
                    kk = k;
                }
            }
            if (maxd < card)
                break;
            jj = X[kk];
            D[kk] = 0;
            kk = 0;
            for (k = 0; k < card; k++) {
                if (A[jj][Cl[NbClas][k]] == 1)
                    kk++;
            }
            if (kk >= card) {
                Cl[NbClas][card] = (short)jj;
                card++;
            }
        }
        Kard[NbClas] = (short)card;

        /* Subset deduplication using generation stamp (avoids O(N) zeroing per check) */
        flag = 0;
        for (k = 0; k < NbClas; k++) {
            if (Kard[k] < card)
                continue;
            cur_stamp++;
            for (j = 0; j < Kard[k]; j++)
                stamp[Cl[k][j]] = cur_stamp;
            flag = 1;
            for (j = 0; j < card; j++) {
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
    mis = Fermeture();
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
        Cl[0][j] = A[j][0] ? 1 : 0;
    NbCliq = 1;

    for (i = 1; i < N - 1; i++) {
        for (j = 0; j < i; j++) X[j] = 0;
        X[i] = 1;
        for (j = i + 1; j < N; j++) X[j] = A[j][i] ? 1 : 0;

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
    int i, j;
    double Sum = 0.0;
    for (i = 1; i < N; i++) {
        const char *Ai = A[i];
        const double *Bi = B[i];
        for (j = 0; j < i; j++)
            if (A[j][i])
                Sum += Bi[j];
        (void)Ai;
    }
    return Sum;
}

void Effacer(void) {
    int i, ii, j, jj, k, card;
    for (k = 0; k < NbClas; k++) {
        memset(Clas, 0, (size_t)N * sizeof(short));
        for (i = 0; i < Kard[k]; i++) {
            ii = Cl[k][i];
            double Cont = 0.0;
            for (j = 0; j < Kard[k]; j++) {
                jj = Cl[k][j];
                Cont += B[ii][jj];
            }
            if (Cont)
                Clas[ii] = 1;
        }
        card = 0;
        for (i = 0; i < N; i++)
            if (Clas[i])
                Cl[k][card++] = (short)i;
        Kard[k] = (short)card;
    }
}

/* Recompute the first column jj < ii maximizing Var[ii][jj] / (Kard[ii]*Kard[jj]) */
static void recompute_row_best(int ii, double *row_best_val, int *row_best_j) {
    if (Kard[ii] == 0) {
        row_best_j[ii] = -1;
        return;
    }
    int best_j = -1;
    double best_v = -1.0e300;
    int ki = Kard[ii];
    const double *Var_ii = Var[ii];
    for (int jj = 0; jj < ii; jj++) {
        if (Kard[jj] == 0 || Var[jj][ii] < eps)
            continue;
        double v = Var_ii[jj] / (double)(ki * Kard[jj]);
        if (best_j < 0 || v > best_v) {
            best_v = v;
            best_j = jj;
        }
    }
    row_best_j[ii] = best_j;
    row_best_val[ii] = best_v;
}

int main(int argc, char *argv[]) {
    int i, j, k, ii, jj, kk, cl1, cl2, k1, k2;
    double Mod, BestMod = 0.0, MMod;

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

    Mod = Modularity();
    MMod = Mod / Na / Na / 2 - SumDg2 / Na / Na / 4;

    if (verbose == 1)
        fprintf(stderr, "Running...\n");

    Var = malloc((size_t)NbClIni * sizeof(double *));
    BestCl = malloc((size_t)NbClIni * sizeof(short *));
    BestKard = malloc((size_t)(NbClIni + 1) * sizeof(short));
    assert(Var && BestCl && BestKard);
    for (i = 0; i < NbClIni; i++) {
        Var[i] = malloc((size_t)NbClIni * sizeof(double));
        BestCl[i] = malloc((size_t)CardMax * sizeof(short));
        assert(Var[i] && BestCl[i]);
        Var[i][i] = -1.0;
        for (j = 0; j < i; j++) {
            Var[i][j] = -Mod;
            Var[j][i] = 0.0;
        }
    }
    for (i = 0; i < NbClIni; i++) {
        BestKard[i] = Kard[i];
        for (j = 0; j < Kard[i]; j++)
            BestCl[i][j] = Cl[i][j];
    }

    /* Parallel Var initialization with thread-local generation stamps (no O(N) scans) */
    #pragma omp parallel private(cl2, k1, k2) if(NbClIni >= 256)
    {
        int *local_stamp = calloc((size_t)N, sizeof(int));
        int cur_stamp = 0;
        assert(local_stamp);

        #pragma omp for schedule(dynamic, 16)
        for (cl1 = 1; cl1 < NbClIni; cl1++) {
            int k_c1 = Kard[cl1];
            const short *c1 = Cl[cl1];
            for (cl2 = 0; cl2 < cl1; cl2++) {
                int k_c2 = Kard[cl2];
                const short *c2 = Cl[cl2];
                double VarMod = 0.0;
                double fus1 = 0.0;

                for (k1 = 0; k1 < k_c1; k1++) {
                    int u = c1[k1];
                    const double *Bu = B[u];
                    const char *Au = A[u];
                    for (k2 = 0; k2 < k_c2; k2++) {
                        int v = c2[k2];
                        if (u == v) {
                            fus1 = 1.0;
                        } else if (u < v) {
                            if (A[v][u] == 1) fus1 = 1.0;
                            VarMod += Bu[v] * (1 - Au[v]);
                        } else {
                            if (Au[v] == 1) fus1 = 1.0;
                            VarMod += Bu[v] * (1 - A[v][u]);
                        }
                    }
                }

                /* Only count exact union cardinality when sum exceeds CardMax */
                if (fus1 > 0.0 && k_c1 + k_c2 > CardMax) {
                    int union_cnt = 0;
                    cur_stamp++;
                    for (k1 = 0; k1 < k_c1; k1++) {
                        if (local_stamp[c1[k1]] != cur_stamp) {
                            local_stamp[c1[k1]] = cur_stamp;
                            union_cnt++;
                        }
                    }
                    for (k2 = 0; k2 < k_c2; k2++) {
                        if (local_stamp[c2[k2]] != cur_stamp) {
                            local_stamp[c2[k2]] = cur_stamp;
                            union_cnt++;
                        }
                    }
                    if (union_cnt > CardMax)
                        fus1 = 0.0;
                }

                Var[cl1][cl2] = VarMod;
                Var[cl2][cl1] = fus1;
            }
        }
        free(local_stamp);
    }

    double *row_best_val = malloc((size_t)NbClIni * sizeof(double));
    int *row_best_j = malloc((size_t)NbClIni * sizeof(int));
    assert(row_best_val && row_best_j);

    #pragma omp parallel for schedule(dynamic, 32) if(NbClIni >= 256)
    for (ii = 1; ii < NbClIni; ii++)
        recompute_row_best(ii, row_best_val, row_best_j);

    int *merge_stamp = calloc((size_t)N, sizeof(int));
    int cur_merge_stamp = 0;
    assert(merge_stamp);

    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif
    int *thread_stamps = calloc((size_t)max_threads * (size_t)N, sizeof(int));
    int *thread_cur_stamp = calloc((size_t)max_threads, sizeof(int));
    assert(thread_stamps && thread_cur_stamp);
    int *todo = malloc((size_t)NbClIni * sizeof(int));
    assert(todo);

    /* Main hierarchical merge loop */
    NbClas = NbClIni;
    while (NbClas > ClCh) {
        double VarMax = -1.0 * ModMax;
        cl1 = -1;
        cl2 = -1;

        for (ii = 1; ii < NbClIni; ii++) {
            int bj = row_best_j[ii];
            if (bj >= 0 && row_best_val[ii] > VarMax) {
                VarMax = row_best_val[ii];
                cl1 = bj;
                cl2 = ii;
            }
        }
        if (cl1 < 0)
            break;

        /* Incremental modularity update: only pairs flipping A[min][max] from 0 to 1 */
        for (k1 = 0; k1 < Kard[cl1]; k1++) {
            i = Cl[cl1][k1];
            for (k2 = 0; k2 < Kard[cl2]; k2++) {
                j = Cl[cl2][k2];
                if (i < j) {
                    if (A[i][j] == 0) {
                        A[i][j] = 1;
                        Mod += B[j][i];
                    }
                } else if (i > j) {
                    if (A[j][i] == 0) {
                        A[j][i] = 1;
                        Mod += B[i][j];
                    }
                }
            }
        }

        /* Merge Cl[cl2] into Cl[cl1] using generation stamp (avoids O(N) loop) */
        cur_merge_stamp++;
        kk = 0;
        for (k1 = 0; k1 < Kard[cl1]; k1++) {
            int u = Cl[cl1][k1];
            if (merge_stamp[u] != cur_merge_stamp) {
                merge_stamp[u] = cur_merge_stamp;
                Cl[cl1][kk++] = (short)u;
            }
        }
        for (k2 = 0; k2 < Kard[cl2]; k2++) {
            int u = Cl[cl2][k2];
            if (merge_stamp[u] != cur_merge_stamp) {
                merge_stamp[u] = cur_merge_stamp;
                Cl[cl1][kk++] = (short)u;
            }
        }
        Kard[cl1] = (short)kk;
        Kard[cl2] = 0;
        row_best_j[cl2] = -1;

        if (FCS == 1 && Mod >= BestMod) {
            kk = 0;
            for (k = 0; k < NbClIni; k++) {
                if (Kard[k] == 0) continue;
                memcpy(BestCl[kk], Cl[k], (size_t)Kard[k] * sizeof(short));
                BestKard[kk] = Kard[k];
                kk++;
            }
            BestNbClas = NbClas - 1;
            BestMod = Mod;
        }

        /* Count actual pair work before starting a team. Most classes in a
         * sparse network cannot merge with this class; skip them serially. */
        int n_todo = 0;
        int64_t pair_work = 0;
        for (ii = 0; ii < NbClIni; ii++) {
            if (Kard[ii] == 0 || ii == cl1 || ii == cl2) continue;
            double f1 = ii < cl1 ? Var[ii][cl1] : Var[cl1][ii];
            double f2 = ii < cl2 ? Var[ii][cl2] : Var[cl2][ii];
            if (f1 < eps && f2 < eps) {
                if (ii < cl1) Var[ii][cl1] = 0.0;
                else Var[cl1][ii] = 0.0;
            } else {
                todo[n_todo++] = ii;
                pair_work += (int64_t)Kard[ii] * Kard[cl1];
            }
        }
        #pragma omp parallel private(ii, k, k1) if(n_todo >= 32 && pair_work >= 100000)
        {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            int *local_stamp = thread_stamps + (size_t)tid * (size_t)N;

            #pragma omp for schedule(dynamic, 32)
            for (int task = 0; task < n_todo; task++) {
                ii = todo[task];

                double VarMod = 0.0;
                const short *Cii = Cl[ii];
                const short *Ccl1 = Cl[cl1];
                int k_ii = Kard[ii];
                int k_cl1 = Kard[cl1];

                for (k = 0; k < k_ii; k++) {
                    int u = Cii[k];
                    const double *Bu = B[u];
                    const char *Au = A[u];
                    for (k1 = 0; k1 < k_cl1; k1++) {
                        int v = Ccl1[k1];
                        if (u < v)
                            VarMod += Bu[v] * (1 - Au[v]);
                        else if (u > v)
                            VarMod += Bu[v] * (1 - A[v][u]);
                    }
                }

                double can_fuse = 1.0;
                if (k_ii + k_cl1 > CardMax) {
                    int union_cnt = 0;
                    int l_stamp = ++thread_cur_stamp[tid];
                    for (k = 0; k < k_ii; k++) {
                        if (local_stamp[Cii[k]] != l_stamp) {
                            local_stamp[Cii[k]] = l_stamp;
                            union_cnt++;
                        }
                    }
                    for (k1 = 0; k1 < k_cl1; k1++) {
                        if (local_stamp[Ccl1[k1]] != l_stamp) {
                            local_stamp[Ccl1[k1]] = l_stamp;
                            union_cnt++;
                        }
                    }
                    if (union_cnt > CardMax)
                        can_fuse = 0.0;
                }

                if (ii > cl1) {
                    Var[ii][cl1] = VarMod;
                    Var[cl1][ii] = can_fuse;
                } else {
                    Var[cl1][ii] = VarMod;
                    Var[ii][cl1] = can_fuse;
                }
            }
        }

        /* Maintain per-row best cache */
        if (cl1 > 0)
            recompute_row_best(cl1, row_best_val, row_best_j);

        #pragma omp parallel for schedule(dynamic, 32) if(NbClas >= 2048)
        for (ii = cl1 + 1; ii < NbClIni; ii++) {
            if (Kard[ii] == 0) {
                row_best_j[ii] = -1;
                continue;
            }
            int bj = row_best_j[ii];
            if (bj == cl1 || bj == cl2) {
                recompute_row_best(ii, row_best_val, row_best_j);
            } else if (Var[cl1][ii] >= eps) {
                double v = Var[ii][cl1] / (double)(Kard[ii] * Kard[cl1]);
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
    free(merge_stamp);
    free(thread_stamps);
    free(thread_cur_stamp);
    free(todo);

    if (verbose == 1)
        fprintf(stderr, "Remaining classes: None                \n\n");

    if (FCS == 1) {
        if (BestMod < eps)
            BestNbClas = NbClIni;
        for (k = 0; k < BestNbClas; k++) {
            Kard[k] = BestKard[k];
            memcpy(Cl[k], BestCl[k], (size_t)Kard[k] * sizeof(short));
        }
        NbClas = BestNbClas;
        for (k = NbClas + 1; k < NbClIni; k++)
            Kard[k] = 0;
    } else {
        kk = 0;
        for (k = 0; k < NbClIni; k++) {
            if (Kard[k] == 0) continue;
            memcpy(Cl[kk], Cl[k], (size_t)Kard[k] * sizeof(short));
            Kard[kk++] = Kard[k];
        }
        NbClas = kk;
    }

    Effacer();
    for (i = 0; i < N - 1; i++)
        for (j = i + 1; j < N; j++)
            A[i][j] = 0;
    for (k = 0; k < NbClas; k++) {
        for (i = 0; i < Kard[k] - 1; i++) {
            ii = Cl[k][i];
            for (j = i + 1; j < Kard[k]; j++) {
                jj = Cl[k][j];
                if (ii < jj) A[ii][jj] = 1;
                else A[jj][ii] = 1;
            }
        }
    }

    Mod = Modularity();
    MMod = Mod / Na / Na / 2 - 1. * SumDg2 / Na / Na / 4;
    clasout(NbClas, Mod, MMod);
    if (cyto)
        clasoutcyto(NbClas);

    return 0;
}
