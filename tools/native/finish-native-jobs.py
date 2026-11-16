"""Finish this saved native batch concurrently, preserving its identities/results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import polars as pl
import psutil

root = Path(__file__).resolve().parents[2]
output = root / 'data/analysis_runs/clustering_native'
jobs = json.loads((output / 'plan.json').read_text())['jobs']
targets = {str(Path(job['target']) / 'job.json') for job in jobs}
active = {}
# Stop only this batch's scheduler. Its already running worker keeps its result.
for process in psutil.process_iter(['pid', 'cmdline']):
    args = process.info['cmdline'] or []
    if len(args) > 2 and Path(args[1]).name == 'cluster_networks.py':
        if args[2] == 'run' and str(output.relative_to(root)) in args:
            process.terminate()
        elif args[2] == '_worker' and len(args) > 3 and args[3] in targets:
            active[args[3]] = process

env = os.environ.copy()
env.update(PYTHONHASHSEED='42', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           MKL_NUM_THREADS='4', POLARS_MAX_THREADS='4')
handles = []
for job in jobs:
    target = Path(job['target'])
    result_file = target / 'result.json'
    if result_file.is_file():
        continue
    job_file = str(target / 'job.json')
    if job_file in active:
        continue
    target.mkdir(parents=True, exist_ok=True)
    Path(job_file).write_text(json.dumps(job, indent=2))
    log = (target / 'worker.log').open('w')
    handles.append(log)
    process = subprocess.Popen([sys.executable, str(root / 'cluster_networks.py'),
                                '_worker', job_file], cwd=root, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    active[job_file] = process
    print(f'Started {job["network"]} / {job["method"]}', flush=True)

previous = -1
while True:
    rows = []
    for job in jobs:
        path = Path(job['target']) / 'result.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        row = {key: result.get(key) for key in (
            'status', 'nodes', 'edges', 'communities', 'covered_nodes',
            'unassigned_nodes', 'overlapping_nodes', 'elapsed_seconds',
            'peak_rss_bytes', 'reason')}
        row.update(network=job['network'], method=job['method'], target=job['target'])
        rows.append(row)
    if len(rows) != previous:
        temporary = output / 'summary.parallel.tmp'
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(output / 'summary.json')
        pl.from_dicts(rows, infer_schema_length=None).write_parquet(output / 'summary.parquet')
        (output / 'live.json').write_text(json.dumps({
            'status': 'complete' if len(rows) == len(jobs) else 'running',
            'completed_jobs': len(rows), 'total_jobs': len(jobs),
            'scheduler': 'parallel native workers',
        }, indent=2))
        print(f'{len(rows)}/{len(jobs)} results saved', flush=True)
        previous = len(rows)
    if len(rows) == len(jobs):
        break
    time.sleep(1)
for log in handles:
    log.close()
print(json.dumps(rows, indent=2), flush=True)
