# Human Interactome Atlas

A Streamlit app for the local HuRI, merged BioPlex HEK293T, huMAP2 and `TAGC_binary2025` networks.

The active application uses Polars for tabular loading, manipulation, plotting inputs, and CSV exports. NumPy and SciPy handle numerical and sparse-graph calculations. Keep new DataFrame code in Polars; repository guidance is in `AGENTS.md`. The archived application retains its original dependencies.

```bash
.venv/bin/streamlit run app.py
```

**Interactomes Overview**, the first page, includes local topology statistics, degree distributions, descriptions and paper links, a huMAP2 probability filter, and the preserved BioPlex merge code and pipeline explanation. Statistics and overlaps use the canonical UniProt networks described below.

**Topological analysis** includes a shared guide to effects, significance and unavailable results, plus a metric interpretation expander in each tab. These explain metric direction, comparison populations, null models and limitations. Statistical significance is reported with asterisks from adjusted q-values; descriptive curves and native confidence scores are not significance tests.

Under **Advanced centralities → Cytoscape apps**, send full selected networks (up to 500,000 edges each) to Cytoscape Desktop, run NetworkAnalyzer from the page, or calculate cytoHubba/other app scores in Desktop and read their numeric node columns. Select the metric columns and click **Compare imported metrics** for effects, distributions and BH-adjusted significance across all imported network-by-metric tests. Import checks the complete source node set and undirected edges, recomputes membership from the current protein set, and keeps missing/non-finite scores unavailable. Downloads and `data/analysis_runs/` bundles retain imported tables, tests, source fingerprints, app identity, versions and parameters. Changes to sources, protein sets or metric selections invalidate the corresponding results. The Overview also accepts older cached coverage tables and labels them with the current protein set name.

### Ego levels and functional interpretation

Open **Topological analysis → Local wiring → Ego levels**. The protein-set comparison analyzes all eligible selected-set proteins by default, retaining unmatched proteins in descriptive tables. Controls are sampled without replacement in `floor(log2 degree)` bins. Each inferential comparison retains finite matched pairs; one BH family covers all network × level × metric tests and cross-level changes. The combined ZIP preserves the globally adjusted tests, original values, matching, exclusions, memberships and provenance.

- **1.0 — Spokes:** the ego and its direct partners, with only ego–partner links. Partner identities and degree are descriptive; artificial star density/clustering are not tested.
- **1.5 — Connected partners:** the induced ego/direct-partner graph. Metrics include partner density, triangles, effective size and components/fragmentation after ego removal. Clustering and partner density are aliases, not separate tests.
- **2.0 — Two-hop neighborhood:** the full induced graph within two steps. Metrics include second-shell expansion and the fraction of previously disconnected direct-partner pairs reconnected through second-shell proteins. Undefined denominators remain unavailable.

The individual explorer defaults to level 1.5, accepts any network protein, and keeps ambiguous mapped nodes distinct. **Remove ego** changes the diagram while retaining isolated partners; before/after numerical metrics are labelled explicitly. Exact betweenness and efficiency are optional below 500 nodes / 5,000 edges. Analytical neighborhoods are limited to 2,000 nodes / 20,000 edges, diagrams to 300 nodes / 3,000 edges, and source networks to 500,000 edges. Oversized analytical neighborhoods are skipped explicitly, never edge-sampled. Full eligible node/edge tables retain native score columns in the saved explorer bundle.

To enable optional GO ontology views, install the `go` extra:

```powershell
uv sync --extra go
```

GOATOOLS navigates the local `data/GO/go-basic.obo`. The existing human annotation preparation and evidence rules remain in use, including `is_a`/`part_of` propagation; annotations are not propagated a second time. Topology works without the extra or GO files. GO ontology level is independent of ego-network distance: it is the shortest `is_a` distance from the namespace root (level 0). Equal levels do not imply equal biological specificity. Defaults are Biological Process and GO level 3.

For direct partners, all non-ego proteins, the second shell or a direct-partner component, the GO view provides:

- **Strict majority:** each category covering more than 50% of all group nodes, including unannotated nodes in the denominator. Categories can overlap; multiple majority labels or no majority are valid outcomes.
- **Fractional pie:** each node contributes one total unit split equally across its distinct selected-level categories. The largest eight categories are shown, remaining weights become Other, and unannotated/no-selected-level-category slices remain separate. Hover shows weighted equivalents and distinct-node coverage; coverage percentages may overlap. Pie proportions and majority labels are descriptive, not significance tests.
- **Optional component enrichment:** choose up to twenty direct-partner components (the five largest by default). One-sided hypergeometric tests use the full selected network, including unannotated nodes. BH correction includes every selected component × eligible term, including zero hits, within each network. The default eligible-term background size is five. GO level filters displayed enrichment terms only, after correction; changing it or the pie group does not change enrichment q-values. Fold enrichment and adjusted significance stars are plotted separately from composition.

Archives record ontology/package versions, ontology SHA-256, annotation provenance, evidence filters, group choices and all eligible tests. Confidence scores are association scores, not binding strengths; structural brokerage and GO summaries alone do not validate moonlighting, simultaneous binding or binding exclusivity.

Data lives in `data/raw_networks/`; `data/allmoon_with_moonlitdb_human.csv` supplies the moonlighting analysis page. The previous application, package, scripts, tests, caches and saved networks are archived in `old_streamlit_app/`. Its original configuration and README are preserved there. Run the legacy app with `cd old_streamlit_app && PYTHONPATH=src ../.venv/bin/streamlit run app.py`.

The standalone BioPlex pipeline is `pipelines/bioplex_merge.py`. It reads the three HEK293T release TSVs in `data/raw_networks/bioplex/` and writes the merged, deduplicated TSV and Parquet to `data/processed_networks/`. The download script uses the same raw directory. See the merge script's `--help` for options.

```bash
.venv/bin/python -m pytest tests
```

## Canonical UniProt networks and comparisons

Put `HuRI.psi`, the [huMAP2 accession-pair archive](https://zenodo.org/records/15293715/files/humap2_ppis_ACC_20200821.pairsWprob.gz?download=1), and optionally `TAGC_binaryInteractome2025.csv` in `data/raw_networks/`. Run `.venv/bin/python networks.py` with internet access to prepare all networks, or run `.venv/bin/python networks.py --huri-only` for offline HuRI preparation. On Windows, use `.venv/Scripts/python.exe` in these commands. For BioPlex, huMAP2 and TAGC, the app checks the human accession list for exact IDs and queries UniProtKB for secondary accessions, annotated isoforms, and Ensembl and GeneID cross-references. Results persist in `data/mappings/`. Prepared Parquet files and audits live in `data/processed_networks/`.

HuRI uses the UniProtKB IDs in the two PSI-MITAB interactor columns. Rows without two UniProtKB IDs are excluded and counted in the audit. The **Interactome isoform handling** sidebar control applies to HuRI, BioPlex, huMAP2 and TAGC across the app.

- **Collapse to parent accessions** is the default. Strip numeric isoform suffixes, then remove self-loops and deduplicate the resulting pairs.
- **Unsuffixed accessions only** drops source pairs with an explicit isoform suffix on either endpoint, including `-1`, before mapping the remaining IDs.
- **Keep isoforms distinct** preserves the reported HuRI IDs and keeps documented UniProt targets for the mapped networks. It excludes unresolved IDs and parent-only fallbacks. Comparisons use exact IDs. A suffixed node can match the same isoform in another network. Moonlighting cohort IDs are parent accessions.

Selecting an alternate policy prepares its views offline from the source files and the saved UniProt audit. If a source changes, refresh the audit under **Mapping & provenance**. Alternate mapped views and their confidence versions are stored in `data/processed_networks/isoforms/<policy>/`. The app prepares the 0.05 and 0.95 confidence versions and carries over cutoffs saved in the default parent view.

Mapped networks keep **one node per complete UniProt label set**. For example, an original node mapping to `P1` and `P2` becomes one node labelled `P1|P2`, retaining its interactions without expanding them. Source nodes with identical complete label sets merge; `{P1, P2}` stays distinct from `{P1}`. Edges with either endpoint unmapped are removed. Node annotation Parquets in `node_annotations/` retain the accession lists and original source IDs. Network overlap compares complete node label sets. Cohort matching accepts any attached accession but counts each graph node once; such matches retain the mapping ambiguity and do not establish which candidate protein carries the interaction. GO annotations are combined across attached accessions and deduplicated per graph node and term.

The overview's **Mapped UniProt labels** metric counts distinct accessions attached to final graph nodes. A node labelled `P1|P2` contributes one node and two labels; an accession shared by multiple nodes counts once in the label total. Exact isoform labels retain their suffixes in the isoform view. HuRI labels come directly from the supplied PSI IDs.

The expanded **Filtering audit** panel in the overview shows before/after node and edge totals and every filtering step for all available networks and confidence versions. Raw counts include repeated interaction rows. Undirected duplicates are removed **before mapping**, then self-loops and duplicate pairs are checked again after mapping/merging. Scored duplicates retain the maximum valid native score. The audit separates unmapped-endpoint loss, node merging, score validity, and inclusive confidence filtering. HuRI uses supplied PSI UniProt IDs without an external lookup. BioPlex includes the original release-merge stages when their cleaned pair set matches the current merged input; otherwise its trace begins with that merged input. All node counts are incident endpoints, so a mapped protein can disappear when all its partners are filtered out. Missing or stale audits are reported, never reconstructed from final totals. Use **Rebuild networks and filtering audits from saved mapping (offline)** to regenerate outputs and saved cutoffs without contacting UniProt; changed sources require a mapping refresh first.

### Standalone mapping script

[pipelines/map_networks.py](pipelines/map_networks.py) is a commented command-line workflow for mapping and preparation outside Streamlit. It uses the same validated functions as the app, retains raw inputs, rebuilds existing confidence cutoffs, and saves the standard 0.05 and 0.95 cutoffs. Every filtering passage is printed and exported to `data/processed_networks/mapping_filtering_steps.csv`.

```powershell
# Resolve identifiers with UniProt and rebuild mapped networks.
.venv/Scripts/python.exe pipelines/map_networks.py

# Reuse saved mappings offline; changed source fingerprints cause an error.
.venv/Scripts/python.exe pipelines/map_networks.py --offline

# Add a custom cutoff and export all three isoform policies.
.venv/Scripts/python.exe pipelines/map_networks.py --offline --cutoff 0.5 --isoforms collapse canonical_only retain

# Prepare HuRI only, without UniProt access or a saved mapping audit.
.venv/Scripts/python.exe pipelines/map_networks.py --huri-only
```

Use `--data-dir PATH` for another data root. The default is this repository's `data/` directory, regardless of the current working directory. HuRI-only runs export `HuRI_filtering_steps.csv`. BioPlex expects the merged TSV produced by `pipelines/bioplex_merge.py`.

Open **HuRI PSI notebook** in the Interactomes Overview tab bar to read the walkthrough, inspect its Python cells, and download [the executed notebook](notebooks/huri_psi.ipynb). The saved outputs compare all three policies and record the source SHA-256. To rerun, open the notebook from the repository in Jupyter with a Python kernel that has the project dependencies, `ipykernel`, and `matplotlib`. It reads `data/raw_networks/HuRI.psi` and computes the three views in memory.

Open **huMAP2 notebook** in the same tab bar to view and download [the executed huMAP2 notebook](notebooks/humap2.ipynb). It previews the raw archive, walks through the canonical mapping audit, and compares the inclusive 0.05 and 0.95 confidence cutoffs. Rerun it with the same Jupyter setup after preparing huMAP2 to explore a custom cutoff or look up partners by primary accession.

The **Shared connections** tab compares nodes or undirected edges for selected base networks and saved versions. Choose an interactive Venn for two or three datasets, a static Venny4Py diagram for four datasets, or UpSet for up to 12 datasets. Venn labels show exclusive counts and percentages of the selected union. Region areas are fixed.

The **Confidence & coverage** tab supports huMAP2 and BioPlex. It saves inclusive native-score cutoffs in the active policy's `versions/` directory, with source fingerprints, score definitions and provenance JSON. BioPlex uses `pInt`, read from the source `p(Interaction)` field. huMAP2 uses its own confidence score. Each score has a network-specific scale. Duplicate pairs retain the maximum valid score. BioPlex base edges with missing or invalid scores remain in the base graph but are excluded from confidence-filtered versions and threshold counts. Dataset selectors show saved versions for the active policy and check that their parent graph still matches. The chart reports incident nodes and unique edges at 0.05 intervals, highlights 0.05 and 0.95, and exports counts and plots.

The **Original IDs** tab shows pie charts and counts for distinct endpoint identifiers before mapping, isoform collapsing, or confidence filtering. It profiles the primary HuRI PSI interactor columns, huMAP2, TAGC and the original BioPlex HEK293T releases. For BioPlex it counts supplied UniProt accessions and includes NCBI Gene IDs only when no included release provides an accession for that Gene ID. The charts classify identifiers by source format.

Under **Mapping & provenance**, **Unique UniProt mapping results** counts all distinct documented targets per source, including multiple targets from the same ID. It shows exact target IDs, primary accessions and the subset in the official human Swiss-Prot reviewed accession list. These are mapping results before interaction filters. The audit includes HuRI's Ensembl protein IDs and shows the original BioPlex releases separately from the merged graph input.

The **Mapping & provenance** tab lists every source ID, exact isoform matches, parent-accession fallbacks, updated accessions, cross-references and unresolved IDs. Use the network and outcome filters or search by accession. CSV and JSON downloads include every result. Open **Compare edges with isoforms kept distinct** to compare source pairs with modern UniProtKB pairs in any of the four networks. The comparison retains multiple documented targets as one node label set and excludes pairs with unresolved endpoints or parent-only fallbacks. For huMAP2, the selected minimum probability applies before and after mapping. The refresh button reruns the lookup and rebuilds the mapped graphs and their saved confidence cutoffs. Completed accession lookup batches are reusable for 24 hours. The HuRI lookup is an audit alongside the three direct PSI views. The huMAP2 raw source panel previews both ID columns and the confidence score, and counts explicit numeric isoform suffixes in the original file.


## Moonlighting analysis page

Open **Topological Analysis** from the Streamlit sidebar. **Network scope** defaults to **Compare networks** with MoonLitDB selected. The report compares every pair of selected interactomes, using the saved huMAP2 0.05 version by default when available. The analysis tabs group measures by type and use the sidebar's **Network for individual analyses** where a method takes one network. The visualization page uses the overview selection to order its network options and takes its starting network and cohort settings from Topological Analysis. The page uses AllMoon and MoonLitDB source cohorts and resolves accessions through the saved UniProt mapping (`data/mappings/moonlighting_uniprot.json`).

Ten independently executed tabs cover mapping/coverage, centralities (with exact Brandes betweenness computed once per network and cached in `data/cache/centralities/`), cohort cohesion with degree-binned permutations, local ego wiring with matched complement controls, multi-paradigm community clustering across 19 cached Classic/Hard, Overlapping, and Fuzzy/Soft algorithms (`data/cache/clusterings/`), random walk with restart and candidate linkers, GO enrichment, node-removal robustness, cross-network rank shifts/partner turnover, and source-set comparisons. Tabs load lazily and each Run button saves only its own result. Changing inputs marks prior results stale; switching tabs retains completed results.

**Compare source sets** opens by default with MoonProt, Multitask and MoonLitDB selected separately. A checked-by-default **Show union of selected sets** option adds a fourth, automatically deduplicated Union cohort to the plots. Hiding Union updates the plots immediately; the analysis always includes Union in its BH correction, keeping other stars unchanged. Union membership and results remain in the saved ZIP. The view displays multiple curated or other source cohorts with stable source colors. Choose box or violin distributions: boxes are the default for discrete metrics such as degree, while violins are the default for continuous metrics. Bars show full-data means with descriptive SEM error bars. Violin shapes sample up to 5,000 values per set; boxes, bar summaries, observed medians and tests use the full data. A background histogram adds dashed vertical lines at each source's full-data observed median; this reference is not a permutation null. Its source selection is independent of the sidebar union/intersection filter; reviewed-entry and publication filters still apply. Shared proteins appear in each applicable source, and the common background excludes the union of selected sets. Each source is tested against that background using two-sided Mann–Whitney U tests, with BH correction across all selected source/metric comparisons plus Union in the run. Stars show adjusted q-values (* < 0.05, ** < 0.01, *** < 0.001, **** < 0.0001); they do not compare sources to one another. Missing tests are unavailable rather than significant. Downloads include plots as HTML and ZIP bundles with source IDs, coverage, overlaps, metrics, tests and provenance. Centrality, local-wiring and community plots use the same box, bar and observed-median views.

Two heatmaps summarize all centrality measures across sets before the individual distributions. **Centrality signatures** show rank-biserial effects (`2U/(n_source × n_background) − 1`): positive means higher values in the source, negative means lower, and zero means no rank tendency. Each cell includes its effect and BH significance label; hover shows medians, median difference/fold, sample sizes and exact test values. Median fold is unavailable when the background median is zero. **Hub enrichment** tests whether a set is overrepresented among proteins at or above each metric's full-network 90th percentile, including ties. The actual hub fraction may exceed 10%; a constant metric makes every finite-valued protein a hub and gives a baseline fold of 1. Hub tests use a separate hypergeometric/BH family over all sets and measures, including Union, with the full finite-valued network as the population. Labels show exact folds; colors saturate at ≥2×. Missing comparisons stay unavailable. Both heatmaps export HTML and CSV and are included in saved analysis tables. Enable advanced centralities for core number, clustering, betweenness, eigenvector and bridging measures, subject to the graph-size guard.

Each network's full node set defines the statistical population. Comparison metrics use the complement of the selected cohort; this is unlabelled background rather than validated negative proteins. Degree bins use `floor(log2 degree)` and empirical tests use +1 corrections. GO uses the local human GOA and ontology (October 2025), excludes NOT qualifiers, propagates is_a/part_of, reports annotation coverage, and corrects over all eligible terms. Do not infer binding exclusivity or validated moonlighting from topology alone.

Successful runs persist ZIPs of CSV tables and JSON parameters, cohort IDs, source provenance and diagnostics in `data/analysis_runs/`. Advanced NetworkX analyses and permutation-heavy diffusion/removal require ≤500,000 edges: select a saved huMAP probability version for these analyses. The unfiltered graph remains available for coverage, basic centrality, diffusion ranking, and GO.

Use the **Analysis** selector within each of the five tabs.

| Tab | Analyses |
| --- | --- |
| **Sources & coverage** | Side-by-side source centralities and mapping coverage |
| **Topology & wiring** | Basic centrality, path and spectral roles, electrical and walk roles, local wiring, ego geometry, rooted graphlets, edge curvature and L3 links |
| **Communities & function** | Community detection and enrichment, module cartography and brokerage, GO enrichment, GO neighborhood geometry, and protein metadata and partner annotations |
| **Cohesion & resilience** | Cohesion, overlap-preserving source separation, diffusion and linkers, and connectivity and efficiency after removal |
| **Across networks** | Degree and partner turnover, two-layer participation and diffusion, and native-score filtration |

Each method runs on demand and exports its own parameters, tables and provenance.
Sampled neighborhood analyses report degree-bin matching, coverage and computational limits.
The electrical and walk analysis accepts an accession to target a small connected component.
Filtration can use the lowest available validated saved version when a scored base is absent.
See the implementation notes in [the analysis catalogue](MOONLIGHTING_TOPOLOGICAL_ANALYSES.md) for method details.

## Network Visualization page

Open **Network Visualization** from the sidebar and select a canonical network or a saved confidence-filtered version. In **All interactions**, enter a canonical UniProt accession to focus on its direct interactions. **Moonlighting only** shows interactions with both endpoints in the union of the selected moonlighting sets. **Moonlighting + first neighbors** also includes their direct neighbors and any interactions among those neighbors in the selected network. The edge limit defaults to 2,000 and can be raised to 20,000. Above the selected limit, the page shows a repeatable sample and counts only the displayed interactions and their incident proteins.

- **Cytoscape.js** provides accession search, multiple layouts, node and edge inspection, and PNG export. Its browser library loads from a CDN.
- **Apache ECharts** provides circular and force-directed views with pan, zoom, and node sizing by displayed degree.
- **NORMA** opens the same interaction pairs in an embedded iframe and offers a link to open the session in a new browser tab. Clicking **Load in NORMA** sends the displayed accession pairs and view name to the hosted NORMA API, which stores the payload temporarily.

The displayed interactions are available as CSV, with the network's native confidence score when present.

## Cytoscape Bridge page (`py4cytoscape` & Cytoscape.js)

Open **Network Visualization / Cytoscape** in the sidebar and choose **Cytoscape Desktop** to connect canonical UniProt networks (`HuRI`, `BioPlex`, `huMAP2`) and moonlighting cohorts (`MoonProt`, `Multitask`, `MoonLitDB`, and `Union`) to a local Cytoscape Desktop instance via **`py4cytoscape`** (CyREST `v1`) and an embedded interactive **Cytoscape.js** viewer.

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

## Cluster annotation

- **Network Visualization** → **Cluster and annotate this view** runs Louvain or
  an overlapping method on the displayed graph. Enrichment uses all proteins in
  the selected network or saved version. Inspecting a cluster highlights its
  proteins in Cytoscape.js and Apache ECharts.
- **Topological Analysis** → **Communities & function** → **Community map** → **Annotate these communities with
  GO and Reactome** uses the memberships from the current clustering run.
- **Cytoscape Bridge** → **clusterMaker2 & Module Bridging** → **Annotate Cytoscape
  clusters with GO and Reactome** reads the selected Desktop network and
  optionally compares both AutoAnnotate labels.

Each panel includes the core and interface labels, term evidence for the selected
cluster, and a ZIP download. Results are saved under `data/analysis_runs/`.

### Command line

Use `interactome-annotate` after installing the project, or run
`python -m annotate_clusters` from the repository root. For a saved clustering,
pass its `memberships.parquet` and the exact network or threshold version that
produced it. A CSV with one `protein,community` row per membership also works.
Leave the community blank for unassigned proteins.

```powershell
interactome-annotate --memberships path/to/memberships.parquet --network HuRI --output data/analysis_runs/huri_annotations
```

The default inputs are `data/GO/HUMAN-uniprot.gaf.gz`, `data/GO/go-basic.obo`,
and `data/reactome/UniProt2Reactome.txt`. Use `--gaf`, `--obo`, and `--reactome`
to supply other snapshots. GO includes BP, MF, and CC. Reactome uses the human
pathways in the supplied lowest-level mapping.

To use existing Cytoscape clusters, pass the network SUID and the node column
names. A list column preserves overlaps. For text such as `1;2`, add
`--cluster-separator ";"`. The background is every UniProt accession in the
selected Cytoscape network, so select the full network used for clustering.

```powershell
interactome-annotate --cytoscape 123 --uniprot-column accession --cluster-column communities --compare-cytoscape 123 --description-column description --output data/analysis_runs/cytoscape_annotations
```

The optional comparison requires a Cytoscape Desktop network view with
AutoAnnotate and WordCloud installed. It runs WordCloud twice on the same full
clusters, using the selected description column and then per-protein GO and
Reactome names. Use `--cyrest-url` to override the configured local endpoint.
The comparison writes temporary node columns and removes them afterward.
AutoAnnotate may expand collapsed groups in the selected view. You can also
compare a file-based run against a Cytoscape view containing its assigned
proteins.

`cluster_labels.csv` lists the exclusive-core label, shared-interface label,
and full-cluster context when the core has no label. When comparison is requested,
it also includes both WordCloud labels, text coverage, and status. The WordCloud
labels summarize text, while the Python labels use one-sided overrepresentation
against the full network. `enrichment.parquet` retains supporting accessions,
counts, fold enrichment, p-values, and q-values, including evidence for terms
suppressed from label selection. Zero-hit terms have implicit p-values of 1
and count in the single run-wide BH family. `manifest.json` records source
fingerprints, coverage, and any comparison problems.

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
