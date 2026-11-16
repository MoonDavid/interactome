"""Preannotate significant MoonLitDB-enriched c.ocg clusters in saved networks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))

import polars as pl
from clusterannotation import file_input
from clustering_explorer import saved_runs, saved_cluster_enrichment, annotate_run
from moonlighting import load_moon_data, cohort_ids
from networks import discover_datasets, _fingerprint

OUT = Path(__file__).resolve().parent
cohort, _ = cohort_ids(load_moon_data(ROOT / 'data'), ['moonlitdb'])
datasets = discover_datasets(ROOT / 'data')
runs = saved_runs().filter((pl.col('method') == 'c.ocg') & (pl.col('status') == 'success'))
summary = []
for run in runs.sort('network', 'run').iter_rows(named=True):
    record = {k: run[k] for k in ('network', 'method', 'run', 'directory')}
    try:
        directory = Path(run['directory'])
        path = datasets[run['network']]
        audit = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        expected = audit.get('job', {}).get('provenance', {}).get('input', {}).get('sha256')
        if not expected or expected != _fingerprint(path)['sha256']:
            raise ValueError('Saved run fingerprint differs from the current network or is missing')
        frame, population, _ = file_input(directory / 'memberships.parquet', str(path))
        results = saved_cluster_enrichment(frame, population, cohort)
        selection = results.filter((pl.col('qvalue') < .05) & (pl.col('fold_enrichment') > 1))
        results.write_csv(OUT / f"{run['network']}_{run['run']}_enrichment.csv")
        record.update(tested_clusters=results.height, significant_clusters=selection.height)
        print(f"{run['network']} / {run['run']}: {selection.height}/{results.height} significant clusters", flush=True)
        if selection.height:
            target = annotate_run(directory, selection['community'].to_list())
            if not (target / 'manifest.json').is_file():
                raise ValueError('Annotation did not produce a manifest')
            record.update(status='complete', annotation_bundle=str(target))
        else:
            record['status'] = 'no_significant_clusters'
    except Exception as exc:
        record.update(status='failed', reason=str(exc))
    summary.append(record)
    (OUT / 'summary.json').write_text(json.dumps({'protein_set': 'MoonLitDB',
        'selection': 'BH q < 0.05 and fold enrichment > 1; BH across all clusters within each saved version',
        'runs': summary}, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
print('FINISHED', flush=True)
