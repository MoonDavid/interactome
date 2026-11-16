"""Run saved OCG annotation jobs in three isolated processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
LOGS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from clustering_explorer import saved_runs


def run(row):
    name = row['network'] + '_' + row['run']
    print(f"START {name}", flush=True)
    env = dict(os.environ, POLARS_MAX_THREADS='2', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (LOGS / f'{name}.stdout.log').open('w') as out, (LOGS / f'{name}.stderr.log').open('w') as err:
        process = subprocess.Popen([
            sys.executable, '-u', '-c',
            'import sys; from clustering_explorer import annotate_run; '
            'print(annotate_run(sys.argv[1]), flush=True)', row['directory'],
        ], cwd=ROOT, env=env, stdout=out, stderr=err)
        (LOGS / f'{name}.pid').write_text(str(process.pid))
        code = process.wait()
    result = dict(network=row['network'], run=row['run'], pid=process.pid,
                  exit_code=code, finished=datetime.now(timezone.utc).isoformat())
    (LOGS / f'{name}.status.json').write_text(json.dumps(result, indent=2))
    print(f"FINISH {name}: exit {code}", flush=True)
    return result


if __name__ == '__main__':
    rows = [row for row in saved_runs().iter_rows(named=True)
            if row['status'] == 'success' and row['method'] == 'c.ocg']
    print(f'Queued {len(rows)} OCG runs; maximum 3 concurrent workers', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, rows))
    (LOGS / 'parallel_results.json').write_text(json.dumps(results, indent=2))
    sys.exit(int(any(row['exit_code'] for row in results)))
