# Human Interactome Atlas

**Human Interactome Atlas** is a Streamlit and Polars platform for comparing canonical human protein–protein interaction (PPI) networks (**HuRI**, **BioPlex HEK293T**, **huMAP2**, **huMAP3**, and **TAGC_binary2025**) in a common UniProtKB accession space and analyzing moonlighting proteins from **AllMoon** and **MoonLitDB**.

The active application uses **Polars** for all DataFrame loading, filtering, joins, grouping, reshaping, and CSV/Parquet exports, with **NumPy** and **SciPy** for numerical and sparse-matrix calculations. **NetworkX**, **igraph**, **leidenalg**, optional **graph-tool**, and native C/Java/C++ backends power graph metrics, community detection, and layouts. Repository coding and statistical conventions are documented in [`AGENTS.md`](AGENTS.md).

---

## Quick Start & Installation

### Environment & Optional Extras

Install the core package and optional research extras from the repository root:

```powershell
# Core dependencies
uv sync

# Optional feature extras (combine as needed)
uv sync --extra clustering --extra clustering-research --extra embeddings --extra text-annotation --extra go --extra test
```

Or with `pip`:

```bash
pip install -e ".[clustering,clustering-research,embeddings,text-annotation,go,test]"
```

| Extra | Packages | Enables |
| --- | --- | --- |
| `clustering` | `typer`, `rich`, `scikit-learn`, `igraph`, `leidenalg`, `scikit-network`, `markov-clustering` | Batch clustering CLI, `igraph` layouts, Leiden/Infomap/MCL adapters |
| `clustering-research` | `cdlib`, `graspologic`, `networkit`, `infomap`, `hdbscan` | Extended clustering backends in `cluster_networks.py` |
| `embeddings` | `nodevectors`, `umap-learn`, `scikit-learn` | ProNE / GGVec network embeddings and UMAP projections |
| `text-annotation` | `yake` | Optional YAKE keyphrase extraction for cluster text annotation |
| `go` | `goatools` | Auxiliary GO tooling |
| `test` | `pytest` | Test suite execution |

> **Note on optional external tools:**
> - `graph-tool` is optional for SFDP/spring static and Sigma layouts. On Windows it cannot be installed with `pip` and requires running the app inside WSL or Docker ([graph-tool installation guide](https://graph-tool.skewed.de/installation.html)).
> - **Karate Club** (`karateclub`) has legacy NumPy constraints incompatible with NumPy ≥ 1.26 and can be isolated in a dedicated `.venv-karateclub` worker environment.
> - **cuGraph** (`cugraph`) requires a RAPIDS/CUDA environment (typically Linux or WSL2), optionally isolated in `.venv-cugraph`.

### Running the App & Tests

```bash
# Unix / macOS
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest tests
```

```powershell
# Windows PowerShell
.venv/Scripts/streamlit.exe run app.py
.venv/Scripts/python.exe -m pytest tests
```

### Installed CLI Entry Points

Installing the project registers four console commands (also runnable as Python modules):

- `interactome-cluster` (`cluster_networks.py`): batch community detection across 85 explicit adapters.
- `interactome-annotate` (`clusterannotation.py` / `annotate_clusters.py`): GO BP/MF/CC, Reactome, UniProt `FUNCTION`, and GeneRIF cluster annotation.
- `interactome-embed` (`network_embeddings.py`): ProNE / GGVec sparse binary network embeddings.
- `interactome-ocg` (`pipelines/run_ocg_c.py`): native C Overlapping Cluster Generator (`c.ocg`).

---

## Repository & Data Layout

### Application & Core Modules

- `app.py`: Navigation entry point (**Main analysis**: `Interactomes Overview`, `Topological Analysis`; **Other analyses**: `Cluster reports`, `Clustering and communities`, `OCG Exploration`, `Network embeddings`, `Network Visualization / Cytoscape`) and global **Interactome isoform handling** control.
- `pages/1_Interactomes_overview.py` (`networks.py`): Canonical network preparation, official UniProtKB mapping, confidence threshold versions, filtering audits, dataset overlaps, and interactive notebooks.
- `pages/2_Topological_analysis.py` (`moonlighting.py`, `moonlighting_extensions.py`, `topology_display.py`, `topology_connectivity.py`, `topology_jobs.py`, `topology_polling.py`, `saved_monte_carlo.py`, `automatic_monte_carlo.py`, `huri_matched_ui.py`, `plot_exports.py`): Cross-interactome topology comparisons, centrality and hub heatmaps, Monte Carlo shortest-path and cohesion simulations, HuRI-matched sensitivity tests, source-set comparisons, and Cytoscape app metric comparisons.
- `pages/7_Reports.py` (`cluster_reports.py`): Curated MoonLitDB & MoonProt community report explorer for BioPlex and TAGC.
- `pages/5_Precomputed_clusters.py` (`clustering_explorer.py`, `clustering_methods.py`, `cluster_projection.py`, `cluster_annotation_ui.py`, `native_clustering.py`): Unified clustering runner/browser, saved-run enrichment, PCA/UMAP feature projections, functional annotation, and overlapping membership burden testing.
- `pages/OCG_exploration.py` (`ocg_exploration.py`, `ocg_visualization.py`, `ocg_annotation_ui.py`, `ocg_ego_overlap.py`, `ocg_ego_comparison.py`, `network_graph_utils.py`): Dedicated OCG workspace with global overview, 6-view/multi-framework annotation & interactive network exploration, membership histograms, degree-matched permutation tests, cross-interactome conservation, and ego-cluster inspector.
- `pages/6_Network_embeddings.py` (`network_embeddings.py`, `embedding_display.py`): ProNE & GGVec embeddings and 2D/3D UMAP/PCA/t-SNE projections.
- `pages/4_Network_Visualization.py` (`network_view.py`, `network_graph_utils.py`) and `pages/3_Cytoscape_Bridge.py` (`cytoscape_bridge.py`): Multi-framework network viewers and live `py4cytoscape` CyREST integration.
- `pipelines/`: Standalone scripts (`bioplex_merge.py`, `map_networks.py`, `precompute_monte_carlo.py`, `run_ocg_c.py`, `run_clusterone.py`, `study_moonlitdb_communities.py`, `assemble_community_study.py`, `moonlitdb_path_atlas.py`, `start_norma.ps1`).
- `tests/`: Unit and integration test suite (`test_networks.py`, `test_moonlighting.py`, `test_ocg_annotation.py`, `test_ocg_exploration.py`, `test_clusterannotation.py`, `test_cytoscape_bridge.py`, etc.).

### Data Directories

| Directory / File | Contents |
| --- | --- |
| `data/raw_networks/` | Raw source interactome files (never modified in place) |
| `data/allmoon_with_moonlitdb_human.csv` | Curated human moonlighting cohort metadata (`MoonLitDB`, `MoonProt`, `Multitask`, `Curated`, `MetaMoon`, `MoonDB`) |
| `data/mappings/` | Official UniProt mapping caches (`moonlighting_uniprot.json`, `generif_geneid_uniprot.json`, network audits) |
| `data/processed_networks/` | Canonical network Parquets, filtering step CSVs, `node_annotations/`, `versions/` (confidence cutoffs), and `isoforms/` (alternate isoform policies) |
| `data/GO/` | Local human GAF (`HUMAN-uniprot.gaf.gz`) and ontology (`go-basic.obo`, October 2025) |
| `data/reactome/` | Human Reactome lowest-level pathway mappings (`UniProt2Reactome.txt`) |
| `data/annotations/` | Local UniProt/Swiss-Prot `FUNCTION` snapshot (`uniprot_functions.tsv`) and cached GO propagation artifacts |
| `data/generif/` | Optional human `generifs_basic.gz` and offline `abstracts.tsv` (`pmid`, `abstract`) |
| `data/plots/` | Automatically exported static PNG/SVG and standalone HTML figures (`topological_analysis/`, `monte_carlo/`, etc.) |
| `data/analysis_runs/` | Persisted analysis bundles, clustering outputs (`clustering/`, `clustering_raw/`, `clustering_annotations/`), shortest-path distance caches, automatic Monte Carlo runs, and community studies |

---

## Canonical UniProt Networks, Isoform Handling & Provenance

### Preparing Networks & Standalone Mapping

Place `HuRI.psi`, the [huMAP2 accession-pair archive](https://zenodo.org/records/15293715/files/humap2_ppis_ACC_20200821.pairsWprob.gz?download=1), `humap3_all_ppis_202305.pairsWprob.gz`, the BioPlex HEK293T release TSVs (`data/raw_networks/bioplex/`), and optionally `TAGC_binaryInteractome2025.csv` in `data/raw_networks/`:

- Merge BioPlex HEK293T releases:

  ```powershell
  .venv/Scripts/python.exe pipelines/bioplex_merge.py
  ```

- Prepare all canonical networks via the UniProtKB mapping service (or offline for HuRI):

  ```powershell
  .venv/Scripts/python.exe networks.py
  .venv/Scripts/python.exe networks.py --huri-only
  ```

- Or use the standalone mapping CLI [`pipelines/map_networks.py`](pipelines/map_networks.py), which preserves raw inputs, builds standard `0.05` and `0.95` confidence cutoffs, and writes `data/processed_networks/mapping_filtering_steps.csv`:

  ```powershell
  # Resolve identifiers with UniProt and rebuild mapped networks
  .venv/Scripts/python.exe pipelines/map_networks.py

  # Reuse saved mappings offline (errors if source fingerprints changed)
  .venv/Scripts/python.exe pipelines/map_networks.py --offline

  # Add a custom cutoff and export all three isoform policies
  .venv/Scripts/python.exe pipelines/map_networks.py --offline --cutoff 0.5 --isoforms collapse canonical_only retain

  # Prepare HuRI only without UniProt network access
  .venv/Scripts/python.exe pipelines/map_networks.py --huri-only
  ```

### Complete UniProt Label-Set Semantics & Confidence Versions

- **Node identity (`P1|P2`):** Mapped networks retain **one node per complete UniProt label set**. If an original identifier maps to `{P1, P2}`, the node is labeled `P1|P2` without edge expansion; source nodes with identical complete label sets merge, while `{P1, P2}` stays distinct from `{P1}`. Pairs with an unmapped endpoint are removed. Cohort matching accepts any attached accession while counting each graph node once; GO annotations combine across attached accessions and deduplicate per node and term.
- **Mapped UniProt labels:** Counts distinct accessions attached to final graph nodes (a `P1|P2` node contributes 1 node and 2 labels; an accession shared by multiple nodes counts once in the label total).
- **Confidence cutoffs (`versions/`):**
  - **huMAP2 & huMAP3:** Inclusive probability thresholds (`0.05` and `0.95` prepared by default). Native confidence scores are network-specific and not calibrated across releases (see [huMAP3 file definitions](https://humap3.proteincomplexes.org/download)).
  - **huMAP3 (`humap3_all_ppis_202305.pairsWprob.gz`):** Prepared against official UniProt release `2026_04` (25,992,007 source rows; 17,233 distinct source endpoint IDs). Under parent-accession collapse: base contains **16,634 nodes / 25,436,441 edges**; `≥ 0.05` retains **15,506 nodes / 3,757,080 edges**; `≥ 0.95` retains **13,028 nodes / 98,360 edges** (use `huMAP3_0.95` for analyses subject to the 500,000-edge guard).
  - **BioPlex:** Uses `pInt` from `p(Interaction)`; base edges with missing/invalid scores remain in the base graph but are excluded from confidence-filtered versions. Duplicate pairs retain the maximum valid native score.

### Isoform Policies

The sidebar **Interactome isoform handling** control applies globally across the app (storing alternate views under `data/processed_networks/isoforms/<policy>/`):

1. **Collapse to parent accessions (default):** Strips numeric isoform suffixes (`-1`, `-2`, ...), removes self-loops, and deduplicates pairs.
2. **Unsuffixed accessions only:** Drops source pairs carrying any explicit isoform suffix (including `-1`) before mapping.
3. **Keep isoforms distinct:** Preserves reported HuRI PSI IDs and documented UniProt isoform targets for mapped networks while excluding unresolved IDs and parent-only fallbacks. Comparisons use exact IDs; moonlighting cohort IDs remain parent accessions.

---

## Streamlit Workspaces

### 1. Interactomes Overview (`pages/1_Interactomes_overview.py`)

- **Network overview & filtering audit:** Node/edge totals, degree distributions, paper references, and stage-by-stage **Filtering audit** separating unmapped-endpoint loss, node merging, score validity, and confidence filtering. Includes an offline rebuild action from saved mappings.
- **Shared connections:** Compares incident nodes or undirected edges across base networks and saved versions using interactive Venn diagrams (2–3 sets), static **Venny4Py** diagrams (4 sets), or **UpSet** plots (up to 12 sets).
- **Confidence & coverage:** Curves across 0.05 intervals for huMAP2/huMAP3 and BioPlex highlighting `0.05` and `0.95`, with version creation and CSV/plot export.
- **Original IDs & Mapping & provenance:** Profiles raw source ID formats before mapping, inspects exact/parent/cross-reference UniProt outcomes with search/CSV/JSON downloads, compares edges with isoforms retained, and refreshes the UniProt lookup (completed batches cached for 24 hours).
- **Embedded notebooks:** View and download executed [`notebooks/huri_psi.ipynb`](notebooks/huri_psi.ipynb) and [`notebooks/humap2.ipynb`](notebooks/humap2.ipynb).

### 2. Topological Analysis (`pages/2_Topological_analysis.py`)

Each tab runs independently on demand and persists bundles (CSVs, JSON provenance, parameters, and diagnostics) under `data/analysis_runs/`:

- **Interactome stats and [protein set] coverage:** Accession counts and coverage over both the complete interactome and its largest connected component (LCC), using the full mapped protein set as the denominator.
- **Topology metrics enriched:**
  - Four stacked primary centralities (**degree**, **k-coreness**, **clustering coefficient**, and **exact betweenness**) in **Static** (PNG/SVG, `symlog` scale for degree) or **Interactive** mode, with optional Benjamini–Hochberg (BH) adjustment toggle (when enabled, adjusts within each interactome separately for centrality and hub tests).
  - **Other topological metrics:** Normalized unweighted Laplacian centrality, triangle count, effective size (`k − 2T/k`), Burt's neighborhood efficiency, eigenvector centrality, bridging coefficient, and bridging centrality (subject to the 500,000-edge guard for advanced metrics).
  - **Centrality signatures & Hub enrichment heatmaps:** Rank-biserial effects (`2U / (n_source × n_background) − 1`) and 90th-percentile hypergeometric hub enrichment.
  - **Cross-network connectivity comparison:** Protein-by-network degree-percentile heatmap and network-pair-by-protein partner-Jaccard heatmap (descriptive, up to 80 proteins).
  - Automatic cached background computation for missing centralities and automatic plot persistence under `data/plots/topological_analysis/`.
- **Monte Carlo & connectivity:**
  - Automatically loads or prepares **1,000-trial** degree-bin-matched (`floor(log2 degree)`) shortest-path and cohesion comparisons together in the background (`data/analysis_runs/automatic_monte_carlo/`), defaulting to LCC scope (`--scope lcc`) with resumable exact-distance BFS caches in `data/analysis_runs/shortest_path_distances/`.
  - **Connectivity overview & subset betweenness:** Full-network components, 1-hop partners/pairs, reachable/unreachable pair counts, unweighted shortest-path distributions, and `betweenness_centrality_subset` mediation with optional degree-matched Monte Carlo tests.
- **HuRI-matched MoonLitDB:** Anchors MoonLitDB membership to a selected HuRI reference (identical intersection across all selected networks or per-network available members) and runs shortest-path & cohesion Monte Carlo tests.
- **Compare source sets:** Side-by-side box, violin (up to 5,000 sampled shapes; full data for medians/tests), bar (mean ± SEM), and observed-median background histograms across `MoonProt`, `Multitask`, `MoonLitDB`, and deduplicated `Union` against their common union-complement background.
- **Advanced centralities → Cytoscape apps:** Sends networks (≤ 500,000 edges) to Cytoscape Desktop, triggers `NetworkAnalyzer`, or imports numeric node tables (e.g., `cytoHubba`) with strict full-graph validation and BH-corrected comparisons.

### 3. Cluster Reports (`pages/7_Reports.py`)

Reads saved BioPlex and TAGC MoonLitDB/MoonProt community studies (`data/analysis_runs/<protein_set>_small_community_study/reviewed_evidence.json`) without rerunning clustering or applying sidebar isoform policies. Filter enriched clusters by any significant saved GO or Reactome term, inspect full/core/interface hierarchy flows, and view member gene names, protein names, Swiss-Prot `FUNCTION` descriptions, MoonLitDB GO-slim roles, or MoonProt primary/secondary descriptions.

Regenerate the underlying studies with:

```bash
python pipelines/study_moonlitdb_communities.py
python pipelines/assemble_community_study.py
# For MoonProt, pass --protein-set moonprot to both scripts
```

### 4. Clustering and Communities (`pages/5_Precomputed_clusters.py`)

- **Single unified algorithm selector:** Defaults to `c.ocg` when available and lists saved-version counts across Atlas, `cluster_networks.py`, and native backends. Open **Compute a clustering version** to run and save a new partition.
- **One clustering version views:**
  - *Clusters*: Summary counts, protein accession filter, multi-row cluster selection, member CSV download, and on-demand GO/Reactome/text autoannotation.
  - *Cluster enrichment*: One-sided hypergeometric protein-set enrichment over all communities with BH correction, multi-row selection, and bulk autoannotation of significant or selected clusters.
  - *Protein projection*: Cached 2D **PCA** or **UMAP** projections from saved `soft_memberships.npz` or reconstructed 16D standardized spectral topology features (`cluster_projection.py`).
  - *Functional annotation*: Full-run GO/Reactome evidence, GO BP/MF Sankey hierarchy diagrams (with black-on-white contrast and depth range slider), Swiss-Prot `FUNCTION` / GeneRIF phrases, and ZIP download.
  - *Run audit*: Provenance JSON and worker logs.
- **All saved clustering runs:** Ranks communities across all compatible saved runs for the network with one joint BH correction and includes **Annotation overview across clustering methods** (vertical method/library summaries + Markdown/CSV export).
- **Overlapping memberships:** Counts exact distinct cluster IDs per protein (including unassigned proteins as 0), tests increased overlap frequency (`k ≥ 2`, one-sided Fisher exact) and membership burden (one-sided Mann–Whitney U) across all source sets with one joint BH family, and optionally applies an exact user threshold to saved soft matrices (`soft_memberships.npz`) for fuzzy C-means/NMF/RWR/neighborhood.
- **Additional community analyses:** Accesses module cartography/brokerage, algorithm comparisons, and Cytoscape `clusterMaker2` workflows.

### 5. OCG Exploration (`pages/OCG_exploration.py`)

Dedicated workspace for native C Overlapping Cluster Generator (`c.ocg`) results:

- **Global overview:** Interactive cluster-overlap meta-network (cluster size, full-run overlap degree, Jaccard or overlap coefficient similarity) and average-linkage hierarchically ordered similarity matrix across **Plotly 2D + Matrix**, **Cytoscape.js** (cluster overlap & bipartite bridges), **Apache ECharts**, **Sigma.js WebGL**, **PyVis**, **3D Explorer** (`Plotly 3D`), **Static (Matplotlib PNG/SVG/PDF)**, and **UpSet Intersection Plot**, plus full-run Cytoscape GraphML export and CyREST push.
- **Annotation and interactive network (`ocg_annotation_ui.py`):**
  - Automatically reuses matching full-run annotation bundles or runs fast GO/Reactome autoannotation across all clusters containing the selected protein set (`moonlighting_cluster_view`), preserving the complete correction family and protein-set BH `q`-values.
  - Supports **6 complementary views of OCG clustering**:
    1. **Cluster overlap network:** Annotated cluster-to-cluster overlap graph with click-to-inspect nodes and edges in Apache ECharts and protein-set enrichment diamond highlights.
    2. **Bipartite (clusters + bridge proteins):** Two-mode graph linking OCG clusters (hexagons) to multi-cluster bridge proteins (`k ≥ 2`), highlighting protein-set bridges as blue diamonds.
    3. **Selected cluster PPI (core vs. interface):** Intra-cluster physical PPI subnetwork distinguishing exclusive **Core (`k = 1`)** from shared **Interface (`k ≥ 2`)** proteins and separating **Interface–Interface**, **Core–Interface**, and **Core–Core** interactions, with optional single-protein focus.
    4. **Overlapping cluster pair PPI:** Induced PPI network comparing two overlapping clusters (`C_a` vs. `C_b`), separating shared interface (`C_a ∩ C_b`), cluster-exclusive members, and dashed cross-cluster PPI bridges.
    5. **Overlap matrix & hierarchy:** Average-linkage hierarchically ordered similarity matrix labeled with top saved GO/Reactome terms and BH significance stars (available above the 500,000-edge guard).
    6. **UpSet & core/interface composition:** Stacked Core vs. Interface cluster bar chart with BH significance stars, Interface-fraction vs. Fold-enrichment / Internal-density bubble plot, UpSet pair-intersection plot, and composition/quality CSV export (available above the 500,000-edge guard).
  - Every network view renders across **all visualization frameworks**: **Apache ECharts**, **Cytoscape.js**, **Sigma.js**, **PyVis**, **Plotly 2D**, **Static (Matplotlib PNG/SVG/PDF)**, **3D (Plotly 3D)**, and **NORMA** (`normaviz.org`), plus GraphML, node/edge CSV, Cytoscape ZIP bundle downloads, and live **Cytoscape Desktop** CyREST push.
- **Membership counts:** Exact histogram for `k = 0, 1, 2, ...` distinct cluster memberships across all run proteins, the selected protein set, and other proteins.
- **Degree-matched overlap:** Exact and cumulative `k` curves plus a `floor(log2 degree)` permutation test comparing observed multi-cluster fraction (`k ≥ 2`) and mean `k` against degree-matched background draws (`+1` empirical p-values, BH correction across the two summary tests and separately across per-bin Fisher exact tests), with pointwise 95% null intervals.
- **Cross-interactome:** Checks multi-cluster (`k ≥ 2`) conservation across all networks with saved `c.ocg` runs and tests protein-set hypergeometric enrichment at each conservation threshold.
- **Ego-cluster inspector:** Searchable across all proteins in the run using local gene names, protein names, and UniProt accessions. Displays full-run cluster memberships with cached GO/Reactome labels, lazy UniProt `FUNCTION` & MoonLitDB GO-slim descriptions, **Overlapping cluster hulls** (`ocg_ego_overlap.py`) alongside Plotly 2D, Bipartite Star, Cytoscape.js, Apache ECharts, 3D, and Static Matplotlib viewers, cross-interactome ego comparisons (`ocg_ego_comparison.py`), and NORMA3 ZIP/API integration.

### 6. Network Embeddings (`pages/6_Network_embeddings.py`)

Trains **ProNE** (default, seeded) or **GGVec** embeddings from `nodevectors` on sparse undirected binary adjacency without edge sampling (`network_embeddings.py` / `interactome-embed`), saving vectors and provenance under `data/analysis_runs/network_embeddings/`. Computes cached 2D/3D **UMAP**, **PCA**, and **t-SNE** projections with degree coloring, accession search, and non-destructive protein-set or custom-accession ring/diamond outlines.

### 7. Network Visualization & Cytoscape Desktop

- **Network viewers (`pages/4_Network_Visualization.py`):** Visualize `All interactions` (with optional single-protein focus), `Moonlighting only`, or `Moonlighting + first neighbors` up to 100,000 sampled edges across **Cytoscape.js**, **Apache ECharts**, **Sigma.js** (offline Sigma.js 3.0.2 + Graphology 0.26.0 WebGL with cached `igraph` DrL/FR, `graph-tool` SFDP, or circular coordinates), **PyVis**, **Static** (PNG/SVG/PDF), **3D** (Plotly 3D), and **NORMA** (`normaviz.org`). Includes on-demand **Cluster and annotate this view** (Louvain or MCODE with fluff) and full-selection GraphML/CSV/ZIP/CyREST export.
- **Cytoscape Desktop Bridge (`pages/3_Cytoscape_Bridge.py`):** Connects via `py4cytoscape` CyREST (default port auto-detected from `~/CytoscapeConfiguration/cytoscape3.props`, checking for port conflicts such as LM Studio on 1234). Supports Moonlighting + 1-hop linker bridges with Augmented Forman–Ricci curvature ($\kappa_F(u,v) = 4 - k_u - k_v + 3 t_{uv}$), cross-network multiplex subnetworks, `clusterMaker2`, `NetworkAnalyzer`, `diffusion`, `cytoHubba`, `stringApp`, inter-cluster participation coefficients ($P_i = 1 - \sum_c (k_{i,c}/k_i)^2$), and `.cyjs`/`.cx2`/`.graphml`/`.sif`/`.cycmd`/CSV ZIP bundles.

---

## Cluster Annotation & Text Mining

Implemented in [`clusterannotation.py`](clusterannotation.py) and detailed in [`CLUSTER_ANNOTATION_METHODS.md`](CLUSTER_ANNOTATION_METHODS.md):

- **Protein subsets:** Evaluates each cluster's **full** membership, exclusive **core** (`k = 1` in the complete partition), and shared **interface** (`k ≥ 2`).
- **GO BP/MF/CC & Reactome:**
  - *Majority rule (> 50%):* Descriptive coverage over all subset proteins (missing annotations remain in the denominator); GO terms propagate through `is_a`/`part_of` (roots excluded) and report deepest majority terms alongside multi-label depth profiles (`GO_*_hierarchy_labels`).
  - *Hypergeometric enrichment:* One-sided test against the complete original network population over eligible terms (≥ 5 background proteins). One BH family spans all tested clusters, nonempty subsets, libraries, and eligible terms (including zero-hit tests). Filtering views or diagrams never alters saved `q`-values.
  - *GO hierarchy Sankey diagrams:* Interactive depth flows (`2 → 3 → 4` or custom depth range slider) with explicit black-on-white contrast, bold counts/percentages, and downloadable HTML + `go_hierarchy_support.parquet`.
- **UniProt `FUNCTION` & GeneRIF text scoring:**
  - Uses the reviewed human UniProtKB/Swiss-Prot snapshot (`data/annotations/uniprot_functions.tsv`; release `2026_04`: 20,429 entries, 17,361 with `FUNCTION` comments) and optional human (taxon 9606) `data/generif/generifs_basic.gz` + offline `data/generif/abstracts.tsv` (`pmid`, `abstract`).
  - Descriptive ranking methods (minimum 2 supporting proteins; no significance tests/stars): per-protein L2-normalized **TF-IDF**, cluster-level **c-TF-IDF**, **Document lift**, **Dirichlet-smoothed log-odds** (human UniProt prior, $\alpha_0 = 1{,}000$), and optional **YAKE** keyphrases (`1 / (1 + native_score)`).

### Annotation CLI Examples

```powershell
# Annotate a saved clustering membership Parquet
interactome-annotate --memberships path/to/memberships.parquet --network HuRI --output data/analysis_runs/huri_annotations

# Include UniProt FUNCTION download (when TSV absent) and GeneRIF text
python -m clusterannotation --memberships memberships.parquet --network HuRI --uniprot-functions data/annotations/uniprot_functions.tsv --fetch-uniprot --generif data/generif/generifs_basic.gz --output data/analysis_runs/function_labels

# Annotate Cytoscape Desktop clusters and compare with AutoAnnotate/WordCloud
interactome-annotate --cytoscape 123 --uniprot-column accession --cluster-column communities --compare-cytoscape 123 --description-column description --output data/analysis_runs/cytoscape_annotations
```

---

## Command-Line Pipelines & Batch Runners

### Batch Clustering Lab (`cluster_networks.py`)

Runs up to 85 explicit adapters over unweighted, undirected canonical networks with isolated workers, resource telemetry, per-job timeouts, cost/dense node guards (default 5,000 / 3,000), the 500,000-edge NetworkX guard, and cryptographic resume keys under `data/analysis_runs/clustering/`:

```powershell
.venv/Scripts/python.exe cluster_networks.py methods
.venv/Scripts/python.exe cluster_networks.py inventory --output data/analysis_runs/clustering_inventory.json
.venv/Scripts/python.exe cluster_networks.py run --dry-run
.venv/Scripts/python.exe cluster_networks.py run --network HuRI --method igraph --method leidenalg
```

Native external clustering adapters (`native.clusterone` and `native.oslom`) require Java or `OSLOM_EXEC` (`oslom_undir` on PATH). ClusterONE defaults to the local `tools/clusterone/cluster_one-1.0.jar`; `CLUSTERONE_JAR` and `JAVA` can override the JAR and Java executable. The WSL runner in `tools/native/run-native-clustering.sh` uses the local tools directory:

```bash
python cluster_networks.py run --method native.clusterone --method native.oslom
```

### Native OCG and ClusterONE Runners

- **Native C OCG (`pipelines/run_ocg_c.py` / `interactome-ocg`):**

  ```powershell
  python -m pipelines.run_ocg_c --input data/processed_networks/HuRI.parquet
  ```

  Saves runs as `c.ocg` under `data/analysis_runs/clustering_raw/<network>/c.ocg/<hash>/` (30,000-node / 500,000-edge guards).
- **ClusterONE Java CLI with optional Cytoscape import (`pipelines/run_clusterone.py`):**

  ```powershell
  .venv/Scripts/python.exe pipelines/run_clusterone.py data/processed_networks/HuRI.parquet data/processed_networks/BioPlex.parquet data/processed_networks/versions/huMAP2_0.05.parquet --min-size 3 --min-density 0.3 --cytoscape --base-url http://127.0.0.1:1234/v1
  ```

  Pass `--weight-column probability` (huMAP2/huMAP3) or `--weight-column pInt` (BioPlex) for weighted runs on a single network (≤ 500,000 edges).

### Standalone Monte Carlo Precomputation (`pipelines/precompute_monte_carlo.py`)

Precomputes degree-bin-matched shortest-path and cohesion simulations and warms the shared BFS shortest-path cache (`data/analysis_runs/shortest_path_distances/`):

```bash
# Default: 1,000 simulations, seed 42, degree-bin matched, MoonLitDB across 5 default networks
.venv/bin/python pipelines/precompute_monte_carlo.py --dry-run
.venv/bin/python pipelines/precompute_monte_carlo.py --networks data/processed_networks/HuRI.parquet --protein-set moonlitdb --analyses shortest_path cohesion --scope lcc --simulations 1000
# Use --protein-set each for all individual sources or --protein-set all for their union
```

### Local NORMA3 Server (`tools/NORMA/`)

When installed under `tools/NORMA/`, launch the local NORMA3 API server on `http://127.0.0.1:8000` via `start_norma.cmd` or:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File pipelines/start_norma.ps1
```

Rebuild the frontend with `npm.cmd ci && npm.cmd run build` inside `tools/NORMA/frontend` (Node 22 required).

---

## Statistical & Scientific Conventions

- **Background populations:** Comparisons test observed protein sets against the unlabelled network complement (or shared union complement in multi-source comparisons) within the full network or LCC. Unlabelled proteins are not validated negative controls.
- **Degree-bin matching:** Uses `floor(log2(max(1, degree)))` bins computed in the active comparison graph.
- **Empirical tests:** Shortest-path simulations use twice the smaller inclusive tail with `+1` correction; cohesion and degree-matched multi-membership use upper-tail `+1` empirical p-values (`(hits + 1) / (N + 1)`).
- **Multiple-testing correction & stars:** Benjamini–Hochberg (BH) adjustment is applied over each explicit test family prior to display filtering. Significance labels follow `q < 0.05` (`*`), `q < 0.01` (`**`), `q < 0.001` (`***`), `q < 0.0001` (`****`), `ns` for valid non-significant tests, and `n/a` when unavailable. Descriptive counts, majority votes, text scores, overlap matrices, and graph layouts carry no significance tests.
- **Computational guards:** Advanced NetworkX metrics, permutation-heavy diffusion/removal analyses, ClusterONE/OCG runners, and interactive graph layouts enforce a **500,000-edge guard** (`ADVANCED_EDGE_LIMIT`); select confidence-filtered versions (`huMAP2_0.95`, `huMAP3_0.95`) for dense interactomes.
- **Biological caveat:** Network topology, co-clustering, and multi-cluster overlap identify candidate complexes and hubs but do not by themselves establish moonlighting function or binding exclusivity.
