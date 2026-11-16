"""Reproduce the requested BioPlex/TAGC small-community annotation study."""
from pathlib import Path
import sys
import json
import argparse
from datetime import datetime, timezone

# The project launcher may be unavailable; permit an installed compatible Python
# to reuse pure-Python/abi3 project dependencies after loading its own SciPy.
import numpy as np
import scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import polars as pl
from clustering_explorer import saved_runs, all_saved_cluster_enrichment, annotate_run
from moonlighting import cohort_ids, load_moon_data
from networks import discover_datasets

OUT = ROOT / 'data/analysis_runs/moonlitdb_small_community_study'


def main(source='moonlitdb'):
    OUT = ROOT / f'data/analysis_runs/{source}_small_community_study'
    OUT.mkdir(parents=True, exist_ok=True)
    cohort, _ = cohort_ids(load_moon_data(ROOT / 'data'), [source])
    runs = saved_runs()
    datasets = discover_datasets(ROOT / 'data')
    summary = []
    for network in ('BioPlex', 'TAGC_binary2025'):
        print(f'Ranking {network}', flush=True)
        folder = OUT / network
        folder.mkdir(exist_ok=True)
        all_rows, excluded = all_saved_cluster_enrichment(runs, network, datasets[network], cohort)
        all_rows.write_parquet(folder / 'all_community_tests.parquet')
        all_rows.drop('directory').write_csv(folder / 'all_community_tests.csv')
        excluded.write_csv(folder / 'excluded_runs.csv')
        selected = (all_rows.filter((pl.col('proteins') < 200) & (pl.col('qvalue') < .05))
                    .head(200).with_row_index('rank', offset=1))
        selected.write_csv(folder / 'selected_clusters.csv')
        bundles = []
        for (directory,), group in selected.group_by('directory', maintain_order=True):
            print(f"Annotating {network}: {group['method'][0]} / {group['version'][0]} ({group.height} clusters)", flush=True)
            target = annotate_run(Path(directory), group['community'].to_list())
            bundles.append(dict(directory=directory, annotation_directory=str(target),
                                communities=group['community'].to_list()))
            (folder / 'annotation_bundles.json').write_text(json.dumps(bundles, indent=2), encoding='utf-8')
        summary.append(dict(network=network, tested=all_rows.height,
                            compatible_runs=all_rows.select('directory').n_unique(), excluded_runs=excluded.height,
                            significant_small=all_rows.filter((pl.col('proteins') < 200) & (pl.col('qvalue') < .05)).height,
                            selected=selected.height, annotation_bundles=len(bundles)))
        print(summary[-1], flush=True)
    (OUT / 'study_manifest.json').write_text(json.dumps(dict(
        generated_at=datetime.now(timezone.utc).isoformat(), protein_set={'moonlitdb': 'MoonLitDB', 'moonprot': 'MoonProt'}[source],
        max_proteins_exclusive=200,
        selection='Up to 200 clusters per network; proteins < 200 and global BH q < 0.05; existing global-q/fold/hit ranking',
        networks=summary), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protein-set', choices=['moonlitdb', 'moonprot'], default='moonlitdb')
    main(parser.parse_args().protein_set)
