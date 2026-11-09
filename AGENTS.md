# Project guidance

These instructions apply throughout this repository.

## Project purpose and layout

This project is a Streamlit application for comparing human protein interaction networks (HuRI, BioPlex HEK293T, and huMAP2) in a common UniProt accession space, and studying moonlighting proteins from AllMoon and MoonLitDB.

- `app.py`: Human Interactome Atlas navigation and page names.
- `pages/1_Network_Explorer.py`: network overview, topology summaries, dataset overlaps, huMAP2/BioPlex threshold versions, and mapping audits.
- `pages/2_Moonlighting_Analysis.py`: coverage, centrality, cohesion, local wiring, communities, diffusion, GO enrichment, robustness, cross-network comparisons, and source-set comparisons.
- `networks.py`: canonical network preparation, official UniProt mapping, saved versions, provenance, network statistics, and overlap visualizations.
- `moonlighting.py`: moonlighting cohort preparation, topology/diffusion/GO analyses, overlapping source-set comparisons against their shared union-complement background, distribution plots, and centrality/hub heatmaps.
- `pipelines/bioplex_merge.py`: standalone BioPlex release merge pipeline.
- `tests/`: tests for the current application (`test_networks.py` and `test_moonlighting.py`).
- `README.md`: setup and current behavior. `MOONLIGHTING_TOPOLOGICAL_ANALYSES.md`: broader research catalogue, including methods not yet implemented.

## DataFrame manipulation and performance

- **Use Polars for all DataFrame loading, filtering, joins, grouping, aggregation, deduplication, and reshaping in the active application.** Use `import polars as pl`.
- Prefer Polars expressions and vectorized operations over Python row loops, row-wise callbacks, and pandas `apply`.
- For large files, prefer lazy scans such as `pl.scan_parquet(...)`; push filters and column selection before collecting, and collect only the data needed.
- Keep tabular processing and plotting inputs in Polars. Use NumPy arrays at scientific-library boundaries when needed; do not introduce pandas imports or conversions.
- Preserve the archived legacy application unless explicitly asked to migrate it.
- Preserve explicit schemas, accession strings, null handling, join cardinality, and deterministic output ordering. Avoid loading the full unfiltered huMAP2 graph into NetworkX unnecessarily.

## Data and scientific conventions

- Source networks live in `data/networks/`; preserve raw inputs. Store mappings in `data/mappings/`, canonical Parquet networks in `data/canonical/`, threshold versions in `data/versions/`, plots in `data/plots/`, and analysis bundles in `data/analysis_runs/`.
- Preserve source fingerprints, mapping audits, parameters, and provenance when producing derived data. Use existing writers and dataset discovery rather than bypassing their validation.
- Canonical networks use human UniProt primary accessions, collapse isoform suffixes, represent undirected pairs consistently, and exclude self-loops. HuRI uses reviewed human Swiss-Prot mappings; preserve all reviewed matches when a gene maps to several entries.
- For duplicate canonical huMAP2 pairs, retain the maximum probability. Probability filtering is inclusive (`probability >= cutoff`). Standard cutoffs are **0.05 and 0.95**; retain support for custom cutoffs. These are association scores, not statistical p-values.
- BioPlex confidence filtering uses native `pInt` (source `p(Interaction)`), with duplicate maximums and inclusive cutoffs. Preserve base edges with invalid or missing scores as null, audit them, and exclude them from scored versions. Keep native scores distinct; huMAP2 and BioPlex scores are not calibrated across networks.
- Threshold plots should mark both standard cutoffs with dashed lines and highlight exact retained-edge counts. Explain the coverage/stringency tradeoff without claiming the retention curve proves an optimal cutoff.
- Nodes counted from edge lists are incident endpoints; do not imply that isolated proteins are included.
- Moonlighting analyses use the selected network's full node set as the population. The cohort complement is unlabelled background, not validated negative proteins.
- Default to separate MoonProt, Multitask and MoonLitDB comparisons. Automatically compute their deduplicated Union and show it with a checked-by-default visibility checkbox; hiding it must not change the BH correction family or require rerunning the analysis.
- For simultaneous source-set comparisons, keep source memberships separate, report overlap, and exclude the union from the shared background. Correct across all source-by-metric tests in the run; plot significance asterisks from adjusted values only, never from sampled display data or unavailable tests.
- Centrality heatmaps use rank-biserial effects versus the shared background; positive means higher centrality. Hub enrichment instead uses the full finite-valued network and a ties-inclusive 90th-percentile cutoff, with its own hypergeometric/BH test family. Keep these populations and correction families explicit, include Union even when hidden, and never replace unavailable effects or zero-denominator median folds with invented zeros.
- Preserve degree-bin matching (`floor(log2 degree)`), empirical-test +1 corrections, and GO multiple-testing correction over all eligible terms. Topology alone does not validate moonlighting or binding exclusivity.
- Keep analysis tabs lazy and independently runnable, invalidate stale results when inputs change, and preserve the 500,000-edge guard for advanced NetworkX and permutation-heavy diffusion/removal analyses.

## Running and checking changes

- Run the app with `.venv/bin/streamlit run app.py`.
- Run tests with `.venv/bin/python -m pytest tests`; use focused tests during development and checks appropriate to the change.
- Run `.venv/bin/python networks.py` only when network preparation is needed; it uses the online UniProt service and persists mapping results.
- Keep changes focused, update documentation when behavior changes, and visually inspect saved plots for readable labels, correct values, and clipping.
