#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CLUSTERONE_JAR="${CLUSTERONE_JAR:-$PWD/tools/clusterone/cluster_one-1.0.jar}"
export JAVA="${JAVA:-$PWD/tools/native/java-wsl}"
export OSLOM_EXEC="${OSLOM_EXEC:-$PWD/tools/native/OSLOM2/oslom_undir}"
export TMPDIR="${TMPDIR:-$PWD/tools/native/tmp}"
mkdir -p "$TMPDIR"
exec "${NATIVE_PYTHON:-$PWD/.venv-native/bin/python}" cluster_networks.py run \
    --method native.clusterone --method native.oslom \
    --network HuRI --network BioPlex --network TAGC_binary2025 \
    --network huMAP2_0.05 --network huMAP2_0.95 --network huMAP3_0.95 \
    --output data/analysis_runs/clustering_native --threads 4 "$@"
