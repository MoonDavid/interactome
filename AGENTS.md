# Project guidance

These instructions apply throughout this repository.

## Project purpose and layout

This project is a Streamlit application for comparing human protein interaction networks (HuRI, BioPlex HEK293T, and huMAP2) in a common UniProt accession space, and studying moonlighting proteins from AllMoon and MoonLitDB.

- `app.py`: Human Interactome Atlas navigation and page names.
- `pages/1_Interactomes_overview.py`: network overview, topology summaries, dataset overlaps, huMAP2/BioPlex/HuRI threshold versions, and mapping audits.
- `pages/2_Topological_analysis.py`: coverage, centrality, cohesion, local wiring, communities, diffusion, GO enrichment, robustness, cross-network comparisons, and source-set comparisons.
- `networks.py`: canonical network preparation, official UniProt mapping, saved versions, provenance, network statistics, and overlap visualizations.
- `moonlighting.py`: moonlighting protein-set preparation, topology/diffusion/GO analyses, overlapping source-set comparisons against their shared union-complement background, distribution plots, and centrality/hub heatmaps.
- `pipelines/bioplex_merge.py`: standalone BioPlex release merge pipeline.
- `tests/`: tests for the current application (`test_networks.py` and `test_moonlighting.py`).
- `README.md`: setup and current behavior. `MOONLIGHTING_TOPOLOGICAL_ANALYSES.md`: broader research catalogue, including methods not yet implemented.

## DataFrame manipulation and performance

- **Use Polars for all DataFrame loading, filtering, joins, grouping, aggregation, deduplication, and reshaping in the active application.** 

## Data and scientific conventions

- Always annotate statistical significance in plots with asterisks wherever a valid test is available: `*` q < 0.05, `**` q < 0.01, `***` q < 0.001, `****` q < 0.0001. Use the existing `significance_label` helper and adjusted values from the full applicable correction family, never raw p-values or sampled display data. State the test, comparison population, correction family and thresholds in a caption or legend. Use `ns` for a valid non-significant test and `n/a` for an unavailable test. Do not invent tests or stars for descriptive plots, native confidence scores or missing results; explain when no test applies. Preserve effect sizes and directions alongside significance, with readable, unclipped labels.
- Add contextual info boxes or expanders explaining what metrics measure, how to interpret higher/lower values, the comparison background or null model, and limitations. Use "protein set" in user-facing text and identify the selected source by name, such as MoonLitDB.
- Source networks live in `data/raw_networks/`; preserve raw inputs. Store mappings in `data/mappings/`, canonical Parquet networks in `data/processed_networks/`, threshold versions in `data/processed_networks/versions/`, plots in `data/plots/`, and analysis bundles in `data/analysis_runs/`.
- Preserve degree-bin matching (`floor(log2 degree)`), empirical-test +1 corrections, and GO multiple-testing correction over all eligible terms. Topology alone does not validate moonlighting or binding exclusivity.
- Keep analysis tabs lazy and independently runnable, invalidate stale results when inputs change, and preserve the 500,000-edge guard for advanced NetworkX and permutation-heavy diffusion/removal analyses.

## Running and checking changes

- Run the app with `.venv/bin/streamlit run app.py`.
- Run tests with `.venv/bin/python -m pytest tests`; use focused tests during development and checks appropriate to the change.
- Run `.venv/bin/python networks.py` only when network preparation is needed; it uses the online UniProt service and persists mapping results.
- Keep changes focused, update documentation when behavior changes, and visually inspect saved plots for readable labels, correct values, and clipping.
