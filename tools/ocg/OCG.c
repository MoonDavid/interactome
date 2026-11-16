/*************************************************************************/
/*         Program name: Overlapping Class Generator, OCG                */
/*                                                                       */
/*                   Copyright (C) 2011 Alain Guenoche                   */
/*                         guenoche@iml.univ-mrs.fr                      */
/*                                                                       */
/*                        with contributions from:                       */
/*                Benoit Robisson and Charles E. Chapple                 */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/* This program is free software: you can redistribute it and/or modify	 */
/* it under the terms of the GNU General Public License as published by	 */
/* the Free Software Foundation, either version 3 of the License, or	 */
/* (at your option) any later version.					 */
/*   									 */
/* This program is distributed in the hope that it will be useful,	 */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of	 */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	 */
/* GNU General Public License for more details.				 */
/*   									 */
/* You should have received a copy of the GNU General Public License	 */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>. */
/*                                                                       */
/*************************************************************************/

/***************************************************************************************************/
/* This program builds an overlapping class system from an unweighted simple graph G=(V,E). 	   */
/* Let |V|=n and |E|=m.										   */
/*  												   */
/* It is essentially a hierarchical ascending algorithm joining two classes at each step.	   */
/* The optimized criterion is the modularity. It can be either the average gain or the total gain. */
/*  												   */
/* The initial overlapping class system can be : 						   */
/* - the set of all maximal cliques (it can take a long time to establish)       		   */
/* - the set of edges (many initial classes (m) implying many steps (O(m))			   */
/* - the set of "centered cliques" (at most n), giving a fast solution for large graphs.	   */
/*  												   */
/* Two class systems can be calculated, the one maximazing the modularity, or the final one.	   */
/* In that case, the expected minimum number of clusters and the maximum caldinality of the 	   */
/* final clusters are required.									   */
/*  												   */
/* Fusion of classes are realized until one of these conditions is fullfiled. 			   */
/* When no more class fusion can be realized the algorithm stops.				   */
/*  												   */
/* Let p be the number of initial classes. The complexity is this algorithm is O(p^3).		   */
/*  												   */
/***************************************************************************************************/

/* READING GUIDE
 * A class is a community stored as a list of vertex indices. Classes can
 * overlap: a protein may belong to several communities. A clique connects
 * every pair of its vertices; maximal means no vertex can be added to it.
 * Centered cliques are greedy seeds, not all maximal cliques.
 *
 * Complexity notation: n=N vertices, m=Na distinct undirected edges,
 * r=input records including duplicates, p=NbClIni initial classes,
 * s_k=Kard[k], S=sum(s_k), c=largest class size encountered (c<=n).
 * Bounds count loop operations for valid inputs and successful allocations;
 * labels have bounded length. They exclude disk/terminal latency.
 *
 * The historical O(p^3) statement above describes pair selection only.
 * This implementation also processes dense matrices and class lists:
 * gain initialization O(p^2*(n+c*n+c^2)); all merges together
 * O(p^3+p*n^2+p^2*(n+c^2)). Seed generation is separate and maximal-clique
 * enumeration can be exponential. Storage after seeding is
 * O(n^2+p*n+p^2); enumeration can require more candidate storage.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define MaxProt 30000		/* Maximum number of vertices (Proteins) */
#define SupCar 25

/* Shared data structures:
 * Et[v]: protein label; Dg[v]: unweighted degree; Na: edge count.
 * Cl[k][t]: vertex at position t in class k; Kard[k]: class size.
 * A zero Kard retires a class during merging without renumbering slots.
 * Clas: reusable vertex mask or membership-count array.
 * BestCl/BestKard: copied snapshot of the best class system encountered.
 * B: scaled modularity contributions for distinct vertex pairs.
 * Var: lower triangle stores gains, upper triangle stores eligibility.
 * A initially stores symmetric graph adjacency. Fermeture() replaces its
 * UPPER triangle (row<column) with co-membership; the LOWER triangle keeps
 * original edges. Edge/maximal-clique seeding skips Fermeture(), so both
 * triangles initially remain adjacency in those modes.
 * FCS=1 selects saved best classes; FCS=0 selects the final constrained ones.
 */
FILE *FichCar, *FichPlug;

short *Dg, **Cl, *Clas, *Kard, *Q, **BestCl, *BestKard;
int N = 0, Na = 0, NbCliq, NbClIni, NbClas, DgMax, BestNbClas, typ = 3,
        verbose = 0, CardMax = 0, ClCh = -1, FCS = 1, cyto = 0, w = 0;
double **B, **Var;
double alpha = 1.0, SumDg2 = 0., eps = 0.0001, ModMax;
char **A;
char Et[MaxProt][SupCar], FichE[120], FichCyto[120];

/*
 This program computes a hierarchy of overlapping classes
 The initial classes can be either the maximal cliques of the graph (i), the edges of the graph (ii) or the centred cliques (iii)
 The fusion of two classes optimizing the modularity of teh resulting class system is performed at each step
 */

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/

/* Print fixed usage text and exit: O(1), apart from output length. */
int printHelp(char *argv[]) {

    fprintf(stderr, "\nUSAGE: %s [options] <GRAPH FILENAME>\n", argv[0]);
    fprintf(stderr, "\nOPTIONS: \n");

    fprintf(stderr, "\t-h: Print this (h)elp and exit.\n\n");
    fprintf(stderr, "\tInitial Class System type. Choices are:\n");
    fprintf(stderr, "\t-m: (M)aximal Cliques.\n");
    fprintf(stderr, "\t-e: (E)dges.\n");
    fprintf(stderr,
            "\t-c: (C)entered Cliques (default), with two possibilities:\n");
    fprintf(stderr, "\t\t0: requires a stop criterion (see -s and -n).\n");
    fprintf(stderr, "\t\t1: maximizing the partition modularity (default).\n");
    fprintf(stderr,
            "\tWarning : with Maximal Cliques and Edges, maximizing the partition modularity is impossible\n");
    fprintf(stderr,
            "\t(it is maximum from the beginning), so it requires a stop criterion (see -s and -n).\n\n");
    fprintf(stderr,
            "\t-s: Maximum allowed class (s)ize. Can take any integer value\n\t\tfrom 0 (default, no constraint) upwards.\n");
    fprintf(stderr,
            "\t-n: Minimum (n)umber of expected classes (0 means no constraint).\n");
    /*	fprintf(stderr,
     "\t-a: (A)lpha parameter of modularity, set granularity of classes.\n");*/
    fprintf(stderr,
            "\t-p: set an output file to import partition in cytoscape (P)lugin.\n");
    fprintf(stderr, "\t-w: (W)eigthed graph.\n");
    fprintf(stderr,
            "\t-v: (V)erbose, print a progress report and some other information to standard error.\n");
    fprintf(stderr,
            "\n\tNote: It is possible to use Centered Cliques with maximizing the modularity (-c 1, default)\n\tand add some constraints with -s and/or -n.\n");
    fprintf(stderr, "\n\n\n");
    exit(0);
}

/* required by getopts */
int getopt(int argc, char * const argv[], const char * optstring);
extern char* optarg;
extern int optind;

/* Parse seed type, stopping limits and export settings.
 * O(total argument text) time, O(1) extra space. Alpha handling is disabled.
 */
void readargv(int argc, char* argv[]) {

    int c;
    /* Reading setup options */

    while ((c = getopt(argc, argv, "hmewc:s:n:a:p:v")) != -1) {
        switch (c) {

        case 'h':
            printHelp(argv);
            exit(0);
            /*******************************************************/
            /* Choice of Initial Class System. Choices are:	       */
            /* 1 : Maximal Cliques          		       */
            /* 2 : Edges					       */
            /* 3 : Centered Cliques	(default)                      */
            /*******************************************************/
        case 'm':
            typ = 1;
            FCS = 0;
            break;
        case 'e':
            typ = 2;
            FCS = 0;
            break;
            /**************************************************************/
            /*   Choice of class system for Centered Cliques. Choices are:*/
            /*   0: Final class system, needs the expected minimum        */
            /*      number of clusters and the maximum caldinality        */
            /*      of the final clusters.    		              */
            /*   1: The class system that maximizes modularity (default)  */
            /**************************************************************/
        case 'c':
            typ = 3;
            if (strtol(optarg, NULL, 10) == 0)
                FCS = 0;
            break;
            /**********************************************************/
            /* Maximum allowed class cardinality. Can take any	  */
            /*  integer value from 0, the default meaning no          */
            /*  constraint upwards                   		  */
            /**********************************************************/
        case 's':
            CardMax = strtol(optarg, NULL, 10);
            break;
            /* Minimum number of expected classes (0 means no constraint) */
        case 'n':
            ClCh = strtol(optarg, NULL, 10);
            break;
            /* Parameter of the modularity alpha */
            /*		case 'a':
             alpha = atof(optarg);
             break;
             set an output file for import partition in cytoscape*/
        case 'p':
            strcat(FichCyto, optarg);
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

    // If no argumernts have been given, print help
    if (argv[optind] == 0) {
        printHelp(argv);
    }
    /* Setup Errors : Incompatible options selected */

    /*
     * If the Final Class system do not maximize the modularity
     * require to limit the class size or set a minimum number of classes
     */
    if (FCS == 0 && CardMax == 0 && ClCh == -1) {
        fprintf(stderr, "Argument error!\n\n");
        fprintf(stderr,
                "If the intial class system is either Maximal Clique (-m) or Edges (-e),\n");
        fprintf(stderr,
                "requires either limiting the maximum class cardinality (-c) \n");
        fprintf(stderr,
                "and/or, setting a minimum number of expected classes (-m)\n\n");
        exit(0);
    }
    /*********************************************************************************/
    /* If no minimum number of classes has been given by the user (-s), use default. */
    /*********************************************************************************/
    if (ClCh == -1)
        ClCh = 2;

    /*	if (alpha < 0.0 && alpha > 1.0) {
     fprintf(stderr, "Argument error!\n\n");
     fprintf(stderr, "The alpha parameter (-a) is a  between 0 and 1 (def=1).\n\n");
     exit(1);
     }*/
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

// Count the number of lines in a file
/* Consume the file to estimate its record count: O(file bytes) time,
 * O(1) buffer space. Assumes nonempty input; the legacy empty-file path
 * examines an uninitialized buffer.
 */
int countlines(FILE *fp) {
    int nl = 0;
    char buf[BUFSIZ];
    while (fgets(buf, sizeof buf, fp))
        if (strchr(buf, '\n'))
            ++nl;
    if (!strchr(buf, '\n'))
        ++nl;
    return nl;
}

/* qsort comparator: O(label length), bounded by fixed label capacity. */
static int compar(const void *e1, const void *e2) {
    return strcmp(e1, e2);
}

/* Load labels, sort them, reread edges, then build degrees and B.
 * Linear label searches cost O(r*n); dense setup costs O(n^2).
 * Sorting usually takes O(n log n) comparisons; qsort has no portable
 * worst-case guarantee. Add O(file bytes) for reading. Space: O(n^2).
 * Duplicate edges overwrite entries; the last weight wins. Degrees count
 * neighbors rather than weighted strengths, including in weighted mode.
 */
void lecgraph(int argc, char* argv[]) {
    int NbCar, i, ii, j, jj, flag, NR, rscan;
    char MaxCar = 0, Ch1[2 * SupCar], Ch2[2 * SupCar], OldCh[SupCar] = "";
    float pond;
    double **W;
    /********************************************************************/
    /* Read an unweighted or weighted graph file, defined by 			      */
    /* the number of records (NR) and NR records containing	      */
    /* Each edge is defined by two labels (corresponding to 	      */
    /* vertices) and optionaly with a weight; the same edge can be duplicated with no effect	      */
    /* Labels are limited to 25 characters			      */
    /********************************************************************/

    /* Get graph filename (1st commandline argument) */
    strcat(FichE, argv[argc - 1]);

    FichCar = fopen(FichE, "r");
    if (FichCar == 0) {
        printf(
                "Could not open file: %s\n\nTry '%s -h' for more information\n\n",
                FichE, argv[0]);
        exit(0);
    }

    /* First pass to count vertices and sort them alphabetically */

    /* Nr is the number of lines (edges) in the file */
    NR = countlines(FichCar);
    fclose(FichCar);
    FichCar = fopen(FichE, "r");
    strcpy(OldCh, "");
    NbCar = 0;
    for (i = 0; i < NR; i++) {
        rscan = fscanf(FichCar, "%s %s", Ch1, Ch2);
        if (rscan != 2 && verbose == 1)
            fprintf(stderr, "error while parsing graph file\n");
        if (w == 1) { // graph pondere?
            rscan = fscanf(FichCar, "%f", &pond);
            if (rscan != 1 && verbose == 1)
                fprintf(stderr,
                        "error while parsing graph file : weight bad format\n");
        }
        if (strlen(Ch1) > NbCar)
            NbCar = strlen(Ch1);
        if (strlen(Ch2) > NbCar)
            NbCar = strlen(Ch2);
        if (NbCar > MaxCar)
            MaxCar = NbCar;
        if (MaxCar >= SupCar) {
            printf("Labels are limited to %d characters\n", (SupCar - 1));
            printf("Non correct edge: %5d   %s, %s\n", i, Ch1, Ch2);
            exit(0);
        }
        if (strcmp(Ch1, OldCh) != 0) /* if it is not the same as before */
        {
            flag = 1;
            for (j = 0; j < N; j++)
                if (strcmp(Ch1, Et[j]) == 0) {
                    flag = 0;
                    strcpy(OldCh, Ch1);
                    break;
                }
            if (flag) {
                strcpy(Et[N], Ch1);
                N++;
                if (N == MaxProt) {
                    printf("To many vertices > %d  ", MaxProt);
                    exit(0);
                }
            }
        }
        flag = 1;
        for (j = 0; j < N; j++)
            if (strcmp(Ch2, Et[j]) == 0) {
                flag = 0;
                break;
            }
        if (flag) {
            strcpy(Et[N], Ch2);
            N++;
            if (N == MaxProt) {
                printf("To many vertices > %d  ", MaxProt);
                exit(0);
            }
        }
    }
    fclose(FichCar);
    qsort(Et, N, SupCar, compar);

    /*	Second read to store the graph  */
    A = malloc((N) * sizeof(int *));
    B = malloc((N) * sizeof(double *));
    W = malloc((N) * sizeof(double *));
    Clas = malloc((N) * sizeof(short));
    assert(A != NULL && B != NULL && Clas != NULL && W != NULL);
    for (i = 0; i < N; i++) {
        A[i] = malloc(N * sizeof(int)); /* A[i][j]=1 if i adjacent to j, A[i][j]=0 otherwise */
        B[i] = malloc(N * sizeof(double));
        W[i] = malloc(N * sizeof(double));
        Dg = malloc((N) * sizeof(short));
        assert(A[i] != NULL && B[i] != NULL && Dg != NULL && W[i] != NULL);
        for (j = 0; j < N; j++) {
            A[i][j] = 0;
            W[i][j] = 0.0;
        }
    }
    /*	A is the to dimentional adjacency table : A[i][j]=A[j][i]=1 iff vertices i and j are linked
     B is the to dimentional table containing the (integer) modularity value of any pair (i,j)
     W is a dimentional table containing the weights of edges.
     Clas is a one dimentional array : Clas[i] is the class number of vertex i
     Dg is a one dimentional array : Dg[i] is the degree (nb. of adjacent vertices) of vertex i
     */

    FichCar = fopen(FichE, "r");

    if (FichCar == 0) {
        printf(
                "Could not open file: %s\n\nTry '%s -h' for more information\n\n",
                FichE, argv[0]);
        exit(0);
    }
    NR = countlines(FichCar);
    fclose(FichCar);
    FichCar = fopen(FichE, "r");

    if (FichCar == 0) {
        printf("Could not open file: %s\n", FichE);
        exit(0);
    }
    strcpy(OldCh, "");
    ii = -1;
    for (i = 0; i < NR; i++) {
        rscan = fscanf(FichCar, "%s %s", Ch1, Ch2);
        if (rscan != 2 && verbose == 1)
            fprintf(stderr, "error while parsing graph file\n");
        if (strcmp(Ch1, Ch2) == 0)
            continue;
        if (w == 1) { // graph pondere?
            rscan = fscanf(FichCar, "%f", &pond);
            if (rscan != 1 && verbose == 1)
                fprintf(stderr,
                        "error while parsing graph file : weight bad format\n");
        }
        if (strcmp(Ch1, OldCh) != 0) /* if it is not the same as before */
        {
            for (j = 0; j < N; j++)
                if (strcmp(Ch1, Et[j]) == 0) {
                    ii = j;
                    strcpy(OldCh, Ch1);
                    break;
                }
        }
        jj = -1;
        for (j = 0; j < N; j++)
            if (strcmp(Ch2, Et[j]) == 0) {
                jj = j;
                break;
            }
        if (ii == -1){
            printf( "ERROR: could not find a protein with name : '%s'", Ch1);
            fflush( stdout);
        }
        if (jj == -1){
            printf( "ERROR: could not find a protein with name : '%s'", Ch2);
            fflush( stdout);
        }
        /* Le graphe est symetrise */
        if (w == 1) {
            W[ii][jj] = pond;
        } else
            W[ii][jj] = 1;
        W[jj][ii] = W[ii][jj];

        A[ii][jj] = 1;
        A[jj][ii] = A[ii][jj];
    }
    fclose(FichCar);

    /* Evaluating the number of edges (Na) the vertex degres, and DgMax its maximum value*/
    DgMax = 0;
    Na = 0;
    for (i = 0; i < N; i++) {
        Dg[i] = 0;
        for (j = 0; j < N; j++)
            if (A[i][j] > 0)
                Dg[i]++;
        if (Dg[i] > DgMax)
            DgMax = Dg[i];
        Na += Dg[i]; // here Na is the double of the number of edges
        SumDg2 = SumDg2 + 1. * Dg[i] * Dg[i];
    }

    /* For distinct u,v: B=alpha*(2m)*weight-degree(u)*degree(v)/alpha.
     * For alpha=1 and unit weights, this is the ordinary modularity pair
     * contribution scaled by 2m. Na still equals 2m in this block.
     * B diagonals are not initialized here; see Effacer() below.
     */
    /* Evaluating the Modularity values and its maximum */
    ModMax = 0.;
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++) {
            B[i][j] = alpha * Na * W[i][j] - 1. * Dg[i] * Dg[j] / alpha;
            B[j][i] = B[i][j];
            ModMax = ModMax + 1. * A[i][j] * B[i][j];
        }
    Na = Na / 2; //True number of edges

    if (ModMax < eps) {
        fprintf(stderr,
                "Maximum modularity of the graph is negative due to edge weights\nPartitionning this graph is impossible with this method.");
        exit(0);
    }
    /*	//Debug Info : print graph
     printf("%.4f\n", ModMax);
     for (i = 1; i < N; i++) {
     for (j = 0; j < i; j++) {
     if (A[i][j] == 1)
     printf("%s\t%s\t%d\t%d\t%.6f\t%.6f\n", Et[i], Et[j], i, j,
     W[i][j], B[i][j]);
     }
     }*/

    //supprime W
    for (i = 0; i < N; i++) {
        free(W[i]);
        W[i] = NULL;
    }
    free(W);
    W = NULL;

    return;
}

/* Report graph statistics and memberships: O(n^2+S) time, reusing Clas.
 * Checking both triangles of A identifies graph edges within classes.
 * Membership counts distinguish unassigned and overlapping proteins.
 */
void clasout(int NbCl, double Mod, double MMod) {
    int i, j, k, kk, ko;
    /* standard output
     *
     * Data struct and variables used in this function:
     * DataStruct : A, Clas, Kard, Cl, Et
     * var : NbCl, Mod, MMod, FichE, N, Na, DgMax, alpha, typ, FCS
     */
    fflush(stdout);
    printf("#################################################\n");
    printf("# Graph file %s has %d Vertices and %d edges\n", FichE, N, Na);
    printf("# Degre maximum: %d\n", DgMax);
    printf("# Rate of edges: %.4f\n", 1. * Na / N / (N - 1));

    ko = 0;
    for (i = 1; i < N; i++) {
        for (j = 0; j < i; j++) {
            if (A[i][j] && A[j][i]) {
                ko++;
            }
        }
    }
    printf("# Rate of intraclass edges: %.3f\n#\n", 1. * ko / Na);
    /*printf("# Parameter modularity alpha: %.1f\n", alpha);*/

    switch (typ) {
    case 1:
        printf("# Initial classes are: maximal cliques\n");
        break;
    case 2:
        printf("# Initial classes are: edges\n");
        break;
    case 3:
        printf("# Initial classes are: centered cliques\n");
        break;
    }

    if (FCS == 1)
        printf("# Class System is maximizing the modularity\n");
    if (CardMax != 0)
        printf("# Maximum Class Cardinality: %d\n", CardMax);
    if (ClCh != 2)
        printf("# Minimum Number of Classes %d\n", ClCh);

    printf("#################################################\n");

    for (i = 0; i < N; i++)
        Clas[i] = 0;
    for (k = 0; k < NbCl; k++) {
        for (i = 0; i < Kard[k]; i++) {
            Clas[Cl[k][i]]++;
        }
    }
    k = 0;
    kk = 0;
    for (i = 0; i < N; i++)
        if (Clas[i] == 0)
            kk++;
        else if (Clas[i] > 1)
            k++;
    if (kk) {
        printf("\nUnclustered nodes (%d): ", kk);
        for (i = 0; i < N; i++)
            if (Clas[i] == 0)
                printf("%s  ", Et[i]);
        printf("\n");
    } else {
        printf("\nUnclustered nodes (%d): None\n", kk);
    }
    printf("Multiclustered nodes (%d):\n", k);
    for (i = 0; i < N; i++) {
        if (Clas[i] > 1) {
            printf("%s (%d), ", Et[i], Clas[i]);
        }
    }
    printf("\n");

    /* Now print the classes */
    printf("\n\nFinal classes (%d), Modularity = %.3f (%.4f):\n", NbCl, Mod,
            MMod);
    for (k = 0; k < NbCl; k++) {
        printf(">Class %3d  (%d nodes):\n", k + 1, Kard[k]);
        for (i = 0; i < Kard[k]; i++) {
            printf("%s ", Et[Cl[k][i]]);
        }
        printf("\n");
    }
    printf("\n");
    return;
}

/* Export overlapping memberships to Cytoscape: O(n+S+filename length).
 * Legacy caveat: missing braces in the unclustered-node block mean its
 * actual singleton output does not match the apparent intended behavior.
 */
void clasoutcyto(int NbCl) {
    int i, k, kk;
    const char *f, *ff;
    /* output for import in the plugin cytoscape
     *
     * Data struct and variables used in this function:
     * DataStruct : Clas, Kard, Cl, Et
     * var : NbCl, FichE, FichCyto, alpha, typ, FCS, N
     */
    f = strtok(FichE, "/");
    while (f != NULL) {
        ff = f;
        f = strtok(NULL, "/");
    }
    FichPlug = fopen(FichCyto, "w");
    fprintf(FichPlug, "#ModClust analysis export\n");
    fprintf(FichPlug, "#Algorithm:OCG\n");
    fprintf(FichPlug, "#Network:%s\n", ff);
    fprintf(FichPlug, "#Scope:Network\n");
    /*fprintf(FichPlug, "#Parameter:alpha=%.1f\n", alpha);*/

    switch (typ) {
    case 1:
        fprintf(FichPlug, "#Parameter:Initial_classes=maximal_cliques\n");
        break;
    case 2:
        fprintf(FichPlug, "#Parameter:Initial_classes=edges\n");
        break;
    case 3:
        fprintf(FichPlug, "#Parameter:Initial_classes=centered_cliques\n");
        break;
    }

    if (FCS == 1)
        fprintf(FichPlug, "#Parameter:Class_System=maximizing_modularity\n");
    if (CardMax != 0)
        fprintf(FichPlug, "#Parameter:Maximum_Cardinality=%d\n", CardMax);
    if (ClCh != 2)
        fprintf(FichPlug, "#Parameter:Minimum_Number_of_Classes=%d\n", ClCh);

    fprintf(FichPlug, "\n");

    for (i = 0; i < N; i++)
        Clas[i] = 0;
    for (k = 0; k < NbCl; k++) {
        for (i = 0; i < Kard[k]; i++) {
            Clas[Cl[k][i]]++;
        }
    }
    /* Now print the classes */
    for (k = 0; k < NbCl; k++) {
        fprintf(FichPlug, ">ClusterID:%d||\n", k + 1);
        for (i = 0; i < Kard[k]; i++) {
            fprintf(FichPlug, "%s\n", Et[Cl[k][i]]);
        }
        fprintf(FichPlug, "\n");
    }
    k = 0;
    kk = 0;
    for (i = 0; i < N; i++)
        if (Clas[i] == 0)
            kk++;
        else if (Clas[i] > 1)
            k++;
    if (kk) { // if unclustered nodes, show as singletons
        k = NbCl;
        for (i = 0; i < N; i++)
            if (Clas[i] == 0)
                fprintf(FichPlug, ">ClusterID:%d||\n", k + 1);
        fprintf(FichPlug, "%s\n\n", Et[i]);
        k++;
    }
    fclose(FichPlug);
    return;
}

/* Check whether candidate mask Q is contained in an active clique mask.
 * Return 1 if contained; otherwise append Q and return 0.
 * With q candidate rows: O(q*n) time, O(1) additional space.
 * Cl rows here are masks, not the index lists used after enumeration.
 */
int Inclus()
/*	To search if array Q is included in one of the cliques in Table Cl*/
/*	If not, add it */

{
    int k, j, flag;

    for (k = 0; k < NbCliq; k++) {
        if (Cl[k][0] < 0)
            continue;
        flag = 1;
        for (j = 0; j < N; j++)
            if (Cl[k][j] < Q[j]) {
                flag = 0;
                break;
            }
        if (flag == 1)
            return 1;
    } // Non inclus
    for (j = 0; j < N; j++)
        Cl[NbCliq][j] = Q[j];
    NbCliq++;
    return 0;
}

/* Reset the upper co-membership mask, mark every within-class pair, then
 * count original edges missing from that mask. Repeated memberships mark
 * the same pair once. O(n^2+sum(s_k^2)) time, O(1) additional space.
 */
int Fermeture() {
    int i, j, jj, k, kk, mis = 0;
    /* This procedure updates the upper right part of Table A (i<j)
     When i < j, A[i][j]=1 iff vertices i and j are joined in at least one class
     When i > j, A[i][j]=1 iff there is an edge between i and j
     */

    /*	//Debug Info : print classes initiales
     printf("\nClasses initiales : cliques centrées\n");
     for (i = 0; i < NbClas; i++) { // pour chaque classe i
     printf("\n> Classe %d (%d) : ", i, Kard[i]);
     for (j = 0; j < Kard[i]; j++) {
     jj = Cl[i][j];
     printf("%s (%d) ", Et[jj], jj);
     }
     }
     printf("\n\n");*/

    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            A[j][i] = 0;
    for (i = 0; i < NbClas; i++) // pour chaque classe i
        for (j = 1; j < Kard[i]; j++) { // pour chaque élément jj dans la classe i
            jj = Cl[i][j];
            for (k = 0; k < j; k++) { // pour chaque paire jj,kk dans la classe i
                kk = Cl[i][k];
                if (kk > jj)
                    A[jj][kk] = 1;
                else if (kk < jj)
                    A[kk][jj] = 1;
            }
        }
    /* Count the edges such the two ends are not joined in any class */
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            if (A[i][j] == 1 && A[j][i] == 0)
                mis++;
    return mis;
}

/* One two-vertex seed per edge: p=m. Dense scan O(n^2+m) loop work;
 * O(m*CardMax) allocated capacity reserves room for later class unions.
 */
int ClasArete() {
    int i, j, k;
    /*  The initial class system is the set of edges
     Cl is a two dimensional table that will contain all the initial classes
     Kard is an array containing the number of elements in each class
     */
    Cl = malloc((Na) * sizeof(short *));
    Kard = malloc((Na) * sizeof(short));
    assert(Cl != NULL && Kard != NULL);
    for (i = 0; i < Na; i++) {
        Cl[i] = malloc(CardMax * sizeof(short));
        assert(Cl[i] != NULL);
        Kard[i] = 2;
    }
    k = 0;
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            if (A[i][j]) {
                Cl[k][0] = j;
                Cl[k][1] = i;
                k++;
            }
    return k;
}

/* Greedy centered-clique seeds: collect each center's neighbors, rank by
 * degree inside that neighborhood, and accept candidates only when linked
 * to every accepted member. Discard seeds contained in an earlier seed.
 * This is order-dependent; it does not enumerate all maximal cliques.
 * Neighborhood work O(n^2+sum(degree(v)^2)); containment checks O(n^3)
 * worst case. Overall O(n^3) time, O(n^2) allocated class storage.
 * Fermeture() then initializes the co-membership mask.
 */
int StarCliq() {
    int i, j, jj, k, kk, card, DgMax, flag = 1, Adj, NbAdj, mis, Somcard = 0;
    short *X, *D;

    /*  Compute the centered cliques as initial class system */
    /*  Cl is a two dimensional table that will contain all the centered cliques as initial classes
     * --> Cl[numéro de la classe][numéro de l'élément] = élément
     * ex : Cl[4][12]=56 == le sommet '56' est le 12ème élément de la classe 4
     Kard is an array containing the number of elements in each clique
     D is a one dimentional array indicating the internal degre of each vertex (nb. of adjacent vertices in its class)
     X is a working vector
     */
    Cl = malloc((N) * sizeof(short *));
    Kard = malloc((N) * sizeof(short));
    X = malloc((N) * sizeof(short));
    D = malloc((N) * sizeof(short));
    assert(Cl != NULL && Kard != NULL && X != NULL && D != NULL);
    for (i = 0; i < N; i++) {
        Cl[i] = malloc((N) * sizeof(short));
        assert(Cl[i] != NULL);
        X[i] = 0;
    }
    if (verbose == 1) {
        fprintf(stderr, "Calculating Initial class System...");
    }
    for (i = 0; i < N; i++) {

        if (verbose == 1 && i % 200 == 0) {
            fprintf(stderr, ".");
        }

        Cl[NbClas][0] = i;
        card = 1;
        NbAdj = 0;
        for (j = 0; j < N; j++)
            if (A[i][j] == 1) {
                X[NbAdj] = j;
                D[NbAdj] = 1;
                NbAdj++;
            }
        /* Calculate the degrees relative to X */
        for (k = 1; k < NbAdj; k++)
            for (j = 0; j < k; j++)
                if (A[X[k]][X[j]] == 1) {
                    D[k]++;
                    D[j]++;
                }
        kk = -1;
        for (Adj = 0; Adj < NbAdj; Adj++) {
            DgMax = 0;
            for (k = 0; k < NbAdj; k++)
                if (D[k] >= DgMax) {
                    DgMax = D[k];
                    kk = k;
                }
            if (DgMax < card)
                break; /* Not enough adjacent vertices */
            jj = X[kk];
            D[kk] = 0;
            kk = 0;
            for (k = 0; k < card; k++) {
                j = Cl[NbClas][k];
                if (A[jj][j] == 1)
                    kk++;
            }
            if (kk - card >= 0) {
                Cl[NbClas][card] = jj;
                card++;
            }
        }
        Kard[NbClas] = card;
        /* Is it a new class ? */
        flag = 0;
        for (k = 0; k < NbClas; k++) {
            if (Kard[k] < card)
                continue;
            for (j = 0; j < N; j++)
                X[j] = 0;
            for (j = 0; j < Kard[k]; j++)
                X[Cl[k][j]] = 1;
            flag = 1;
            for (j = 0; j < Kard[NbClas]; j++)
                if (X[Cl[NbClas][j]] == 0)
                    flag = 0;
            if (flag == 1)
                break; /* The new class exists */
        }
        if (flag == 0) {
            Somcard += card;
            NbClas++;
        }
    }
    if (verbose == 1) {
        fprintf(stderr, "Done\n");
    }
    free(X);
    free(D);
    if (verbose == 1) {
        fprintf(stderr, "Nb. of classes %d\n", NbClas);
    }
    mis = Fermeture();
    if (verbose == 1) {
        fprintf(stderr, "Nb. of edges not within the classes %d\n", mis);
    }

    return NbClas;
}

/* Sequential maximal-clique enumeration using candidate membership masks.
 * Let q be the maximum number of candidate rows held during enumeration.
 * A conservative loop bound is O(n^2*q^2): candidate processing calls
 * Inclus(), which scans q masks of length n. Storage O(n*q+n^2) including
 * graph matrices. q can be exponential; O(p^3) does not cover this stage.
 * Finally convert surviving masks into vertex-index lists and sizes.
 */
int Clique()
/*	Enumeration of all the maximal cliques of a graph
 Barthelemy & Guenoche, Trees and Proximity Representations, J. Wiley, 1991
 */
{
    int i, j, k, kk, NbY, Nl, Long, NewLong, cli, flag1, flag2;
    short *X, *Y;
    Q = malloc((N) * sizeof(short));
    X = malloc((N) * sizeof(short));
    Y = malloc((N) * sizeof(short));
    assert(Q != NULL && X != NULL && Y != NULL);

    /*  Cl is a two dimensional table that will contain all the maximal cliques as initial classes
     Each row corresponds to a clique
     */

    Nl = 2 * N;
    Long = Nl; /* Allocating Nl rows */
    Cl = malloc((Nl) * sizeof(short *));
    assert(Cl != NULL);
    for (i = 0; i < Nl; i++) {
        Cl[i] = malloc(N * sizeof(short));
        assert(Cl[i] != NULL);
    }

    Cl[0][0] = 1;
    for (j = 1; j < N; j++)
        if (A[j][0])
            Cl[0][j] = 1;
        else
            Cl[0][j] = 0;
    NbCliq = 1;

    /* Sequential algorithm described in the book */
    for (i = 1; i < N - 1; i++) {
        for (j = 0; j < i; j++)
            X[j] = 0;
        X[i] = 1;
        for (j = i + 1; j < N; j++)
            if (A[j][i])
                X[j] = 1;
            else
                X[j] = 0;

        /* Allocating again */
        cli = 0;
        for (k = 0; k < NbCliq; k++)
            if (Cl[k][i] > 0)
                cli++;
        if (NbCliq + 2 * cli + 1 > Long) {
            NewLong = Long + Nl;
            Cl = realloc(Cl, NewLong * sizeof(short *));
            assert(Cl != NULL);
            for (k = Long; k < NewLong; k++) {
                Cl[k] = malloc(N * sizeof(short));
                assert(Cl[k] != NULL);
            }
            Long = NewLong;
        }
        for (k = 0; k < NbCliq; k++) {
            if (Cl[k][i] == 0)
                continue;
            NbY = 0;
            for (j = i + 1; j < N; j++)
                if (Cl[k][j] == 1 && X[j] == 0) {
                    Y[NbY] = j;
                    NbY++;
                }
            if (NbY > 0) {
                for (j = 0; j < N; j++)
                    Q[j] = Cl[k][j];
                Cl[k][0] = -1;
                Q[i] = 0;
                flag1 = Inclus();
                Q[i] = 1;
                for (j = 0; j < NbY; j++)
                    Q[Y[j]] = 0;
                flag2 = Inclus();
                if (flag1 + flag2 == 2)
                    continue;
                for (j = 0; j < N; j++)
                    Cl[k][j] = Cl[NbCliq - 1][j];
                NbCliq--;
            }
        }
        for (j = 0; j < N; j++)
            Q[j] = X[j];
        Inclus();
    }
    Kard = malloc((NbCliq) * sizeof(short));
    assert(Kard != NULL);
    /* Eliminating the unused rows and coding cliques as item listes */
    k = 0;
    for (i = 0; i < NbCliq; i++) {
        if (Cl[i][0] < 0)
            continue;
        kk = 0;
        for (j = 0; j < N; j++)
            if (Cl[i][j]) {
                Cl[k][kk] = j;
                kk++;
            }
        Kard[k] = kk;
        k++;
    }
    NbCliq = k;
    free(Q);
    free(X);
    free(Y);
    return NbCliq;
}

/* Sum B for distinct pairs marked together in upper A. Each pair counts
 * once even if shared by several classes. O(n^2) time, O(1) extra space.
 * This scaled objective is normalized later into MMod.
 */
double Modularity()
/*	Computing the integer modularity value */
{
    int i, j;
    double Sum = 0.;
    for (i = 1; i < N; i++)
        for (j = 0; j < i; j++)
            Sum = Sum + 1. * A[j][i] * B[i][j];
    return Sum;
}

/* Sum each member's within-class B contributions and rebuild class lists.
 * O(p*n+sum(s_k^2)) time, reusing Clas as a mask.
 * Actual test is if (Cont): only EXACTLY ZERO sums are removed. Negative
 * sums are retained, and this function does not create singleton classes.
 * Legacy caveat: it reads B diagonals, which lecgraph() never initializes.
 * These comments document the original behavior without fixing it.
 */
void Effacer() {
    int i, ii, j, jj, k, card, NbElEl = 0;
    double Cont;

    /* Calculate the contribution of each element within its class */

    /*	//Debug Info : Kard
     for (k = 0; k < NbClas; k++)
     printf("Kard[%d]=%d\n", k, Kard[k]);*/

    for (k = 0; k < NbClas; k++) {

        for (i = 0; i < N; i++)
            Clas[i] = 0;
        for (i = 0; i < Kard[k]; i++) {
            ii = Cl[k][i];
            Cont = 0.;
            for (j = 0; j < Kard[k]; j++) {
                jj = Cl[k][j];
                Cont = Cont + B[ii][jj];
            }
            if (Cont)
                Clas[ii] = 1;
            else {
                NbElEl++;
            }
        }
        card = 0;
        for (i = 0; i < N; i++)
            if (Clas[i]) {
                Cl[k][card] = i;
                card++;
            }
        Kard[k] = card;

    }
    return;
}

/* Load -> seed -> initialize gains -> merge -> restore selected system
 * -> clean memberships -> rebuild pair mask -> report/export.
 */
int main(int argc, char *argv[]) {
    int i, j, k, ii, jj, kk, cl1, cl2, k1, k2;
    int card = 1;
    double Mod, BestMod = 0., MMod, fus1, fus2, var, VarMod, VarMax;

    readargv(argc, argv);

    /*  Establish a hierarchy of overlapping classes otimizing the modularity criterion  */

    lecgraph(argc, argv);

    if (CardMax == 0)
        CardMax = N;

    switch (typ) {

    case 1:
        NbClIni = Clique();
        break;

    case 2:
        NbClIni = ClasArete();
        break;

    case 3:
        NbClIni = StarCliq();
        break;
    }
    if (verbose == 1) {
        fprintf(stderr, "Number of initial classes %d\n", NbClIni);
    }
    Mod = Modularity();
    MMod = Mod / Na / Na / 2 - SumDg2 / Na / Na / 4;

    //clasout(NbClIni, Mod, MMod);

    /**********************************************************************************************/
    /* Matrix Var contains two types of information, one in the upper right and the other	        */
    /* in the bottom left. 								        */
    /* 											        */
    /* When i > j, Var[i][j] is the modularity variation given by the fusion of classes  i and j. */
    /* When i < j, Var[i][j] = 1 if they can be merged and 0 if not.			        */
    /* Two classes can be merged if they are connected and if their fusion does not pass CardMax. */
    /**********************************************************************************************/

    if (verbose == 1) {
        fprintf(stderr, "Running...");
    }
    Var = malloc((NbClIni) * sizeof(double *));
    BestCl = malloc((NbClIni) * sizeof(short *));
    BestKard = malloc((NbClIni + 1) * sizeof(short));
    assert(Var != NULL && BestCl != NULL && BestKard != NULL);
    for (i = 0; i < NbClIni; i++) {
        Var[i] = malloc(NbClIni * sizeof(double));
        BestCl[i] = malloc(CardMax * sizeof(short));
        assert(Var[i] != NULL && BestCl[i] != NULL);
        Var[i][i] = -1;
        for (j = 0; j < i; j++) {
            Var[i][j] = -Mod;
            Var[j][i] = 0;
        }
    }
    // initialise BestKard et BestCl
    for (i = 0; i < NbClIni; i++) {
        BestKard[i] = Kard[i];
        for (j = 0; j < Kard[i]; j++)
            BestCl[i][j] = Cl[i][j];
    }

    /* Compare all seed pairs. Clas tracks their union; VarMod sums B for
     * cross-pairs not yet marked together. Eligibility requires overlap or
     * a graph connection and a union within CardMax.
     * The union-size scan is INSIDE k1: per pair O(n+s_1*s_2+s_1*n),
     * bounded by O(n+c^2+c*n). Var itself takes O(p^2) storage.
     */
    /* Initialization of the Var table comparing any two initial classes */
    for (cl1 = 1; cl1 < NbClIni; cl1++) {
        if (verbose == 1 && cl1 % 100 == 0) {
            fprintf(stderr, ".");
        }
        for (cl2 = 0; cl2 < cl1; cl2++) {
            VarMod = 0.;
            fus1 = 0.;
            for (i = 0; i < N; i++)
                Clas[i] = 0;
            for (k1 = 0; k1 < Kard[cl1]; k1++) {
                i = Cl[cl1][k1];
                Clas[i] = 1;
                for (k2 = 0; k2 < Kard[cl2]; k2++) {
                    j = Cl[cl2][k2];
                    Clas[j] = 1;
                    if (i == j)
                        fus1 = 1.; /* Classes connexes */
                    else if (i < j && A[j][i] == 1)
                        fus1 = 1.; /* Connected classes  */
                    else if (i > j && A[i][j] == 1)
                        fus1 = 1.; /* Connected classes */
                    if (i < j)
                        VarMod = VarMod + B[i][j] * (1 - A[i][j]);
                    else if (i > j)
                        VarMod = VarMod + B[i][j] * (1 - A[j][i]);
                }
                kk = 0;
                for (k = 0; k < N; k++)
                    if (Clas[k])
                        kk++;
                if (kk > CardMax)
                    fus1 = 0.;
            }
            Var[cl1][cl2] = VarMod;
            Var[cl2][cl1] = fus1;

            /*			//Debug Info: print Var
             if (fus1 == 1. && VarMod > 0.0)
             printf("%d\t%d\t%d\t%d\t%.2f\t%.4f\n", cl1, cl2, Kard[cl1],
             Kard[cl2], fus1, VarMod);*/
        }
    }
    if (verbose == 1)
        fprintf(stderr, ".\n");

    /*		Main loop	*/
    //BestMod = -1000000.0;
    NbClas = NbClIni;
    card = 1;
    /* At most p-1 iterations, retiring one slot per union. Pair selection
     * scans original slots even after retirement: O(p^2) per iteration.
     * Rank by average gain = cached total gain/(size1*size2), including
     * overlapping memberships in the denominator. Strict > retains the
     * first scanned tie; negative scores can exceed the initial -ModMax.
     */
    while (NbClas > ClCh) {
        VarMax = -1. * ModMax;
        cl1 = -1;
        for (ii = 1; ii < NbClIni; ii++) {
            if (Kard[ii] == 0)
                continue; /* old classes are indicated by setting their cardinality to 0 */
            for (jj = 0; jj < ii; jj++) {
                if (Kard[jj] == 0)
                    continue;
                /* Can we fuse them? */
                if (Var[jj][ii] < eps)
                    continue; // no
                card = Kard[ii] * Kard[jj]; // average gain
                var = Var[ii][jj] / card;
                if (var > VarMax) {
                    VarMax = var;
                    cl1 = jj;
                    cl2 = ii;
                }
            }
        }
        if (cl1 < 0)
            break;

        /*		//Debug Info : print fusion
         printf("fusion : %d\t%d\n", cl1, cl2);*/

        for (k1 = 0; k1 < Kard[cl1]; k1++) {
            i = Cl[cl1][k1];
            for (k2 = 0; k2 < Kard[cl2]; k2++) {
                j = Cl[cl2][k2];
                /* Mark UPPER co-membership; preserve LOWER graph edges. */
                if (i < j)
                    A[i][j] = 1;
                else if (i > j)
                    A[j][i] = 1;
            }
        }
        /* Recompute objective O(n^2); use a vertex mask for the set union
         * and write it in vertex order to cl1, retiring cl2. Mask/list work
         * plus the preceding cross-pair marking costs O(n+c^2).
         */
        Mod = Modularity();
        for (k = 0; k < N; k++)
            Clas[k] = 0;
        for (k1 = 0; k1 < Kard[cl1]; k1++)
            Clas[Cl[cl1][k1]] = 1;
        for (k2 = 0; k2 < Kard[cl2]; k2++)
            Clas[Cl[cl2][k2]] = 1;
        kk = 0;
        for (k = 0; k < N; k++)
            if (Clas[k]) {
                Cl[cl1][kk] = k;
                kk++;
            }
        Kard[cl1] = kk;
        Kard[cl2] = 0; /* Cl[cl1][] will contain the new class */

        /* Save all live memberships when the score ties or improves:
         * O(p+S), at most O(p*n). BestMod starts at zero; initial seed
         * lists were copied before the merge loop.
         */
        if (Mod >= BestMod) {
            kk = 0;
            for (k = 0; k < NbClIni; k++) {
                if (Kard[k] == 0)
                    continue;
                for (i = 0; i < Kard[k]; i++)
                    BestCl[kk][i] = Cl[k][i];
                BestKard[kk] = Kard[k];
                kk++;
            }
            BestNbClas = NbClas - 1;
            BestMod = Mod;
            /*			//Debug Info : print best
             printf("BestNbClas=%d, BestMod=%.4f", BestNbClas, BestMod);*/
        }
        /* Update gains involving surviving cl1: O(p*(n+c^2)) per merge.
         * Other pairs retain cached values; this is not a full rebuild.
         */
        /* Updating table Var */
        for (ii = 0; ii < NbClIni; ii++) {
            if (Kard[ii] == 0 || cl1 == ii || cl2 == ii)
                continue;
            fus1 = 0.;
            fus2 = 0.;
            if (ii < cl1)
                fus1 = Var[ii][cl1];
            else
                fus1 = Var[cl1][ii];
            if (ii < cl2)
                fus2 = Var[ii][cl2];
            else
                fus2 = Var[cl2][ii];
            if (fus1 < eps && fus2 < eps) /* Can be fused with neither one nor the other */
            {
                if (ii < cl1)
                    Var[ii][cl1] = 0.;
                else
                    Var[cl1][ii] = 0.;
                continue;
            }
            /* verifying the cardinality condition and computing Var[ii][cl1] */
            VarMod = 0.;
            for (k = 0; k < N; k++)
                Clas[k] = 0;
            for (k = 0; k < Kard[ii]; k++) {
                i = Cl[ii][k];
                Clas[i] = 1;
                for (k1 = 0; k1 < Kard[cl1]; k1++) {
                    j = Cl[cl1][k1];
                    Clas[j] = 1;
                    if (i < j) {
                        VarMod = VarMod + B[i][j] * (1 - A[i][j]);
                    } else if (i > j) {
                        VarMod = VarMod + B[i][j] * (1 - A[j][i]);
                    }
                }
            }
            kk = 0; /* cardinality of the eventual fusion */
            for (k = 0; k < N; k++)
                if (Clas[k])
                    kk++;
            if (kk > CardMax)
                fus1 = 0.;
            else
                fus1 = 1.;

            if (ii > cl1) {
                Var[ii][cl1] = VarMod;
                Var[cl1][ii] = fus1;
            } else {
                Var[cl1][ii] = VarMod;
                Var[ii][cl1] = fus1;
            }
        }
        NbClas--;
        if (NbClas % 10 == 0 && verbose == 1) {
            fprintf(stderr, "Remaining classes: %d of %d          \r", NbClas,
                    NbClIni);
            setvbuf(stderr, (char *) NULL, _IONBF, 0);
        }
    }
    if (verbose == 1)
        fprintf(stderr, "Remaining classes: None                \n\n");
    if (FCS == 1)
    /* Going back to classes making the largest modularity */
    {
        if (BestMod < eps) { // les fusions n'ont pas augmenter la modularité de la partition.
            BestNbClas = NbClIni; // retour aux classes initiales
        }
        for (k = 0; k < BestNbClas; k++) {
            Kard[k] = BestKard[k];
            for (i = 0; i < Kard[k]; i++)
                Cl[k][i] = BestCl[k][i];
        }
        NbClas = BestNbClas;
        for (k = NbClas + 1; k < NbClIni; k++)
            Kard[k] = 0;
    } else {
        kk = 0;
        for (k = 0; k < NbClIni; k++) {
            if (Kard[k] == 0)
                continue;
            for (i = 0; i < Kard[k]; i++)
                Cl[kk][i] = Cl[k][i];
            Kard[kk] = Kard[k];
            kk++;
        }
        NbClas = kk;
    }
    //Debug Info : print NbClas
    //printf("NbClas=%d, NbClIni=%d\n", NbClas, NbClIni);

    /* Apply legacy zero-contribution cleanup (see Effacer caveat), then
     * rebuild upper A from the chosen class lists so the final objective
     * describes those memberships. O(n^2+sum(s_k^2)) for the rebuild.
     */
    Effacer();
    for (i = 0; i < N - 1; i++)
        for (j = i + 1; j < N; j++)
            A[i][j] = 0;
    for (k = 0; k < NbClas; k++)
        for (i = 0; i < Kard[k] - 1; i++) {
            ii = Cl[k][i];
            for (j = i + 1; j < Kard[k]; j++) {
                jj = Cl[k][j];
                if (ii < jj)
                    A[ii][jj] = 1;
                else
                    A[jj][ii] = 1;
            }
        }

    // print final classes
    Mod = Modularity();
    MMod = Mod / Na / Na / 2 - 1. * SumDg2 / Na / Na / 4;
    clasout(NbClas, Mod, MMod);
    if (cyto)
        clasoutcyto(NbClas);

    return (0);
}
