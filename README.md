# Human Interactome Atlas

A new Streamlit app for the local HuRI, merged BioPlex HEK293T and huMAP2 networks.

The active application uses Polars for tabular loading, manipulation, plotting inputs, and CSV exports. NumPy and SciPy handle numerical and sparse-graph calculations. Keep new DataFrame code in Polars; repository guidance is in `AGENTS.md`. The archived application retains its original dependencies.

```bash
.venv/bin/streamlit run app.py
```

**Network Explorer**, the first page, includes local topology statistics, degree distributions, descriptions and paper links, a huMAP2 probability filter, and the preserved BioPlex merge code and pipeline explanation. Statistics and overlaps use the canonical UniProt networks described below.

Data lives in `data/networks/`; `data/allmoon_with_moonlitdb_human.csv` supplies the moonlighting analysis page. The previous application, package, scripts, tests, caches and saved networks are archived in `old_streamlit_app/`. Its original configuration and README are preserved there. Run the legacy app with `cd old_streamlit_app && PYTHONPATH=src ../.venv/bin/streamlit run app.py`.

The standalone BioPlex pipeline is `pipelines/bioplex_merge.py`. It merges the three raw HEK293T release TSVs and writes outputs beside the inputs. See its `--help` for options.

```bash
.venv/bin/python -m pytest tests
```

## Canonical UniProt networks and comparisons

Run `.venv/bin/python networks.py` once with internet access (or use the preparation button). It maps HuRI Ensembl IDs with UniProt's official asynchronous ID Mapping API, resolves existing UniProt accessions to current human primary records, collapses isoforms, and excludes unmapped IDs. HuRI is restricted to reviewed human Swiss-Prot entries; all reviewed matches are retained where genes have more than one. Other networks retain all human UniProtKB matches. Mapping results and jobs persist in `data/mappings/`; canonical Parquet networks and transformation audits persist in `data/canonical/`. Raw networks are unchanged.

The **Shared connections** tab compares nodes or undirected edges for selected base networks and saved versions. Choose an interactive schematic Venn for 2–3 datasets or UpSet for up to 12 datasets. Counts are exclusive intersections; circle areas are not proportional.

The **Confidence & coverage** tab supports huMAP2 and BioPlex. It saves inclusive native-score cutoffs as `data/versions/<network>_<threshold>.parquet`, with source fingerprints, score definitions and provenance JSON. BioPlex uses `pInt`, read from the source `p(Interaction)` field; huMAP2 uses its own confidence score. Scores are not statistical p-values or calibrated across networks. When canonical pairs collapse, the maximum valid score is retained. BioPlex base edges with missing or invalid scores remain in the base graph but are excluded from confidence-filtered versions and threshold counts. Saved versions automatically reappear in selectors after restarting; versions whose canonical parent changed are excluded. HuRI has no confidence score. The chart reports incident nodes and unique canonical edges at 0.05 intervals, highlights 0.05 and 0.95, and exports counts and plots. Only the 0.05 and 0.95 huMAP2 standard versions are kept locally; 0.2 and 0.9 were removed.

HuRI filtering uses `data/mappings/swissprot_human_accessions.tsv` (the UniProt release is recorded in `swissprot_human_source.json`). `HuRI_swissprot_audit.tsv` lists retained and excluded candidates for every HuRI gene; `HuRI_swissprot_summary.json` counts unmatched genes and lost source interactions. The Mapping & provenance tab displays and downloads this audit.


## Moonlighting analysis page

Open **Moonlighting Explorer** from the Streamlit page sidebar. The page uses AllMoon + MoonLitDB source cohorts and resolves accessions through the saved official UniProt mapping (`data/mappings/moonlighting_uniprot.json`). Select a canonical network or saved threshold version as the population; choose all rows or union/intersection of source flags.

Ten independently executed tabs cover mapping/coverage, centralities (with exact Brandes betweenness computed once per network and cached in `data/cache/centralities/`), cohort cohesion with degree-binned permutations, local ego wiring with matched complement controls, multi-paradigm community clustering across 19 cached Classic/Hard, Overlapping, and Fuzzy/Soft algorithms (`data/cache/clusterings/`), random walk with restart and candidate linkers, GO enrichment, node-removal robustness, cross-network rank shifts/partner turnover, and source-set comparisons. Tabs load lazily and each Run button saves only its own result. Changing inputs marks prior results stale; switching tabs retains completed results.

**Compare source sets** opens by default with MoonProt, Multitask and MoonLitDB selected separately. A checked-by-default **Show union of selected sets** option adds a fourth, automatically deduplicated Union cohort to the plots. Hiding Union updates the plots immediately; the analysis always includes Union in its BH correction, keeping other stars unchanged. Union membership and results remain in the saved ZIP. The view displays multiple curated or other source cohorts with stable source colors. Choose box or violin distributions: boxes are the default for discrete metrics such as degree, while violins are the default for continuous metrics. Bars show full-data means with descriptive SEM error bars. Violin shapes sample up to 5,000 values per set; boxes, bar summaries, observed medians and tests use the full data. A background histogram adds dashed vertical lines at each source's full-data observed median; this reference is not a permutation null. Its source selection is independent of the sidebar union/intersection filter; reviewed-entry and publication filters still apply. Shared proteins appear in each applicable source, and the common background excludes the union of selected sets. Each source is tested against that background using two-sided Mann–Whitney U tests, with BH correction across all selected source/metric comparisons plus Union in the run. Stars show adjusted q-values (* < 0.05, ** < 0.01, *** < 0.001, **** < 0.0001); they do not compare sources to one another. Missing tests are unavailable rather than significant. Downloads include plots as HTML and ZIP bundles with source IDs, coverage, overlaps, metrics, tests and provenance. Centrality, local-wiring and community plots use the same box, bar and observed-median views.

Two heatmaps summarize all centrality measures across sets before the individual distributions. **Centrality signatures** show rank-biserial effects (`2U/(n_source × n_background) − 1`): positive means higher values in the source, negative means lower, and zero means no rank tendency. Each cell includes its effect and BH significance label; hover shows medians, median difference/fold, sample sizes and exact test values. Median fold is unavailable when the background median is zero. **Hub enrichment** tests whether a set is overrepresented among proteins at or above each metric's full-network 90th percentile, including ties. The actual hub fraction may exceed 10%; a constant metric makes every finite-valued protein a hub and gives a baseline fold of 1. Hub tests use a separate hypergeometric/BH family over all sets and measures, including Union, with the full finite-valued network as the population. Labels show exact folds; colors saturate at ≥2×. Missing comparisons stay unavailable. Both heatmaps export HTML and CSV and are included in saved analysis tables. Enable advanced centralities for core number, clustering, betweenness, eigenvector and bridging measures, subject to the graph-size guard.

Each network's full node set defines the statistical population. Comparison metrics use the complement of the selected cohort; this is unlabelled background rather than validated negative proteins. Degree bins use `floor(log2 degree)` and empirical tests use +1 corrections. GO uses the local human GOA and ontology (October 2025), excludes NOT qualifiers, propagates is_a/part_of, reports annotation coverage, and corrects over all eligible terms. Do not infer binding exclusivity or validated moonlighting from topology alone.

Successful runs persist ZIPs of CSV tables and JSON parameters, cohort IDs, source provenance and diagnostics in `data/analysis_runs/`. Advanced NetworkX analyses and permutation-heavy diffusion/removal require ≤500,000 edges: select a saved huMAP probability version for these analyses. The unfiltered graph remains available for coverage, basic centrality, diffusion ranking, and GO.

## Cytoscape Bridge page (`py4cytoscape` & Cytoscape.js)

Open **Cytoscape Bridge** (`pages/3_Cytoscape_Bridge.py`, backed by `cytoscape_bridge.py`) from the Streamlit navigation sidebar to connect canonical UniProt networks (`HuRI`, `BioPlex`, `huMAP2`) and moonlighting cohorts (`MoonProt`, `Multitask`, `MoonLitDB`, and `Union`) to a local Cytoscape Desktop instance via **`py4cytoscape`** (CyREST `v1`) and an embedded interactive **Cytoscape.js** viewer.

- **Subnetwork & bridge modes (Polars-native):**
  - *Moonlighting induced + 1-hop linker bridges*: extracts cohort edges and non-cohort proteins that bridge ≥ *k* distinct moonlighting proteins, annotating edges as `moonlighting_internal`, `linker_bridge`, or `background`, and computing Augmented Forman–Ricci curvature ($\kappa_F(u,v) = 4 - k_u - k_v + 3 t_{uv}$).
  - *Ego-network*: extracts 1-hop or 2-hop neighborhoods around selected moonlighting seed proteins.
  - *Cross-network multiplex*: combines edges across 2–3 canonical networks for the same moonlighting seed/bridge node set, tagging each edge with its source network and multiplicity.
- **Cytoscape Desktop automation via `py4cytoscape`:**
  - *Connection & port diagnostics*: probes CyREST (`http://127.0.0.1:1234/v1` by default, configurable host/port in the sidebar) and detects port conflicts (such as LM Studio occupying port `1234`).
  - *Push & visual styling*: sends Polars node/edge tables to Cytoscape via `py4cytoscape`, applies visual mappings (node fill by moonlighting source cohort, diamond shape for multi-source/bridge proteins, border width for linkers, edge color by bridge role or source network), and triggers layouts (`force-directed`, `kamada-kawai`, `circular`, `attribute-circle`, `grid`).
  - *Installed Cytoscape app workflows*: triggers **`clusterMaker2`** (`leiden`, `mcl`, `louvain`, `mcode`, `glay`, `infomap`, `ap`, `fcm`), **`NetworkAnalyzer`**, **`diffusion`** (heat diffusion from selected moonlighting seeds), **`cytoHubba`** (`MCC`, `DMNC`, `MNC`, `Degree`, `EPC`, `BottleNeck`, `EcCentricity`, `Closeness`, `Radiality`, `Betweenness`, `Stress`, `ClusteringCoefficient`), and **`stringApp`** enrichment, pulling updated Cytoscape node tables back into **Polars** DataFrames to compute hypergeometric/BH moonlighting enrichment per cluster and inter-cluster participation coefficients ($P_i = 1 - \sum_c (k_{i,c}/k_i)^2$).
- **Multi-format export bundle:** exports `.cyjs` (Cytoscape.js JSON), `.cx2` (Cytoscape Cyberinfrastructure Exchange v2), `.graphml`, `.sif`, `.cycmd` (Cytoscape command script for `clusterMaker2`/`diffusion`/`analyzer`), and Polars node/edge CSV tables in a single `.zip` bundle.

The full research catalogue in `MOONLIGHTING_TOPOLOGICAL_ANALYSES.md` includes additional exploratory methods beyond this implementation.

## Batch clustering CLI

`cluster_networks.py` provides a Typer/Rich terminal interface with progress,
elapsed times, logs, per-job timeouts and resume. Install the CPU backends and
inspect the available methods before running all discovered networks:

```powershell
uv sync --extra clustering --extra clustering-research
.venv/Scripts/python.exe cluster_networks.py methods
.venv/Scripts/python.exe cluster_networks.py inventory --output data/analysis_runs/clustering_inventory.json
.venv/Scripts/python.exe cluster_networks.py run --dry-run
.venv/Scripts/python.exe cluster_networks.py run
```

Use `.venv/bin/python` on Unix. The executable `interactome-cluster` is available
after installing the project. Restrict a run by repeating `--network` and
`--method`, for example `--network HuRI --method igraph --method leidenalg`.
The default schedules 85 explicit adapters for every prepared network and valid
saved version from `networks.discover_datasets`; missing canonical files are not
reconstructed. `inventory` lists public functions and classes from the requested
namespaces, including utilities which cannot be run as clustering algorithms.

The catalog includes existing Atlas algorithms, NetworkX, igraph, Leiden, CDlib,
scikit-network, graspologic, scikit-learn, NetworKit, standalone Infomap and MCL,
HDBSCAN, Karate Club and cuGraph. Atlas approximations are separate from native
backend algorithms. CDlib algorithms can have further optional dependencies.
Karate Club has legacy NumPy constraints incompatible with this project's NumPy
requirement; use a separately maintained compatible environment for its adapters.
cuGraph requires an appropriate RAPIDS/CUDA environment, usually Linux or WSL2.
Unavailable backends, failed jobs and skipped jobs are reported distinctly.

All runs use unweighted, undirected topology; confidence thresholds determine
which edges are retained. Embedding algorithms use standardized normalized
spectral coordinates, while spectral clustering uses sparse adjacency affinity.
Overlapping memberships are preserved and uncovered nodes have null community
IDs. Soft results include their raw matrix; its backend column order is separate
from normalized community IDs. Incident endpoints are the population; topology
alone does not validate protein function or moonlighting.

Results and provenance are saved under `data/analysis_runs/clustering/`, with
`plan.json`, checkpointed summaries and per-job JSON, logs and Parquet membership
tables. Resume requires matching inputs, audit, parameters, code and dependency
versions; `--no-resume` recomputes. The 500,000-edge NetworkX guard applies.
Costly and dense methods default to 5,000 and 3,000 nodes, respectively; adjust
`--costly-node-limit` or `--dense-node-limit` to attempt larger jobs. Each job has
a 300-second default timeout, adjustable with `--timeout`. Seeds are recorded,
although parallel backends can be nondeterministic. See `run --help` for options.
