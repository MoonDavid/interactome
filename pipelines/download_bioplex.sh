#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p -- data/raw_networks/bioplex || exit 1
cd -- data/raw_networks/bioplex || exit 1
files=(
  "BioPlex_interactionList_v2.tsv|BioPlex_HEK293T_v1.0.tsv"
  "BioPlex_interactionList_v4a.tsv|BioPlex_HEK293T_v2.0.tsv"
  "BioPlex_293T_Network_10K_Dec_2019.tsv|BioPlex_HEK293T_v3.0.tsv"
  "BioPlex_HCT116_Network_5.5K_Dec_2019.tsv|BioPlex_HCT116_v3.0.tsv"
)
status=0
for entry in "${files[@]}"; do
  source_name="${entry%%|*}"
  filename="${entry#*|}"
  if curl --fail --location --retry 3 --connect-timeout 20 --output "${filename}.part" "https://bioplex.hms.harvard.edu/data/${source_name}"; then
    if test -s "${filename}.part"; then
      mv -- "${filename}.part" "$filename"
      ls -lh -- "$filename"
    else
      printf 'Empty download: %s\n' "$filename" >&2
      status=1
    fi
  else
    printf 'Download failed: %s\n' "$filename" >&2
    status=1
  fi
done
exit "$status"
