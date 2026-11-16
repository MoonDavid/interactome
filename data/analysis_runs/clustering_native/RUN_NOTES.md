# Native clustering attempts — 2026-11-16

Original ClusterONE 1.0 JAR and locally compiled OSLOM2 undirected executable were run through the existing native adapters on six prepared, canonical UniProt network views. Raw inputs were preserved.

- ClusterONE: all six views succeeded (HuRI, BioPlex, TAGC_binary2025, huMAP2_0.05, huMAP2_0.95, huMAP3_0.95); native default thresholds.
- OSLOM: huMAP2_0.05 and huMAP2_0.95 succeeded; HuRI, BioPlex, TAGC_binary2025 and huMAP3_0.95 exceeded the native adapter's fixed 300-second subprocess limit. These are recorded as failed with a TimeoutExpired reason, not valid empty clustering results.
- Both methods use undirected, unweighted topology. OSLOM used seed 42, 10 first-level runs, 10 higher-level runs, and the finest hierarchy output (`tp`).
- The original sequential scheduler was replaced by concurrent workers. The interrupted HuRI OSLOM worker was restarted. Job identities and source fingerprints were preserved.
- All eight successful outputs were checked against input hashes, protein populations, edge counts, duplicate memberships, community sizes, coverage and overlap counts. See `validation.json`.
- `summary.csv`, `summary.json` and `summary.parquet` contain all 12 outcomes. Each network/method folder contains its exact job, result and worker log; successful folders also contain memberships and community sizes in Parquet.
- `environment.txt` records the installed Python dependencies. The four native-adapter tests passed. These runs do not provide evidence validating protein moonlighting.

Local rerun entry point: `bash .tools/run-native-clustering.sh` from the project in WSL Ubuntu. It uses the persistent `.venv-native` Python environment and resumes successful results. Increasing the batch `--timeout` does not increase the separate native adapter's fixed 300-second limit.
