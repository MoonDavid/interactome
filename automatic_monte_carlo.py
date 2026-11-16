"""One background Monte Carlo writer shared by Streamlit sessions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import traceback
from functools import lru_cache

import psutil

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def network_digest(path, size, mtime):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=128)
def corrected_summary(path, size, mtime):
    import polars as pl
    return 'fdr_bh' in pl.read_csv(path, n_rows=0).columns


def alive(record):
    try:
        process = psutil.Process(record['pid'])
        return process.is_running() and abs(process.create_time() - record['created']) < 1
    except (psutil.Error, KeyError):
        return False


@lru_cache(maxsize=32)
def _legacy_cohort(data, protein_set, reviewed_only, min_publications, stamps):
    from pipelines.precompute_monte_carlo import load_cohort
    return load_cohort(Path(data), protein_set, reviewed_only, min_publications)


def cohort_matches(meta, config, metadata_path):
    """Validate older batch exports that predate the cohort fingerprint.

    Legacy reuse requires unchanged local source/mapping files, the same named
    source and filters, and identical currently resolved membership. Custom sets
    and files changed since the export must use a fingerprinted new run.
    """
    digest = hashlib.sha256(json.dumps(config['cohort']).encode()).hexdigest()
    if meta.get('cohort_sha256') is not None:
        return meta['cohort_sha256'] == digest
    if config['protein_set'] == 'custom' or meta.get('protein_set') != config['protein_set']:
        return False
    data = Path(config['data'])
    try:
        sources = [data / 'allmoon_with_moonlitdb_human.csv', data / 'mappings/moonlighting_uniprot.json']
        stamps = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in sources)
        if any(stamp[1] > Path(metadata_path).stat().st_mtime_ns for stamp in stamps):
            return False
        return _legacy_cohort(str(data), config['protein_set'], config['reviewed_only'],
                              config['min_publications'], stamps) == set(config['cohort'])
    except (OSError, ValueError, KeyError):
        return False


def ensure_precomputed(data, networks, protein_set, cohort, scope, seed=42,
                       reviewed_only=False, min_publications=0):
    """Return readiness or launch one isolated, resumable job without blocking the UI."""
    data = Path(data).resolve()
    config = dict(data=str(data), networks=[str(Path(p).resolve()) for p in networks],
                  protein_set=protein_set, cohort=sorted(cohort), scope=scope, seed=int(seed),
                  simulations=1000, reviewed_only=reviewed_only, min_publications=int(min_publications))
    config['inputs'] = {str(p): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in config['networks']}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    output = data / 'analysis_runs'
    jobs = output / 'automatic_monte_carlo'
    jobs.mkdir(parents=True, exist_ok=True)
    status = jobs / f'{key}.status.json'
    previous = {}
    if status.exists():
        previous = json.loads(status.read_text())
        if previous['status'] == 'failed':
            return previous
    # Accept existing complete batch results with the requested parameters.
    ready_networks = []
    for network in config['networks']:
        if network in previous.get('skipped_networks', []):
            continue
        ready = True
        for analysis in ('shortest_path', 'cohesion'):
            prefix = output / f'{protein_set}_{Path(network).stem}_{analysis}'
            archive = output / 'archive' / f'monte_carlo_{scope}' / prefix.name
            if Path(f'{archive}.json').exists():
                try:
                    if not Path(f'{prefix}.json').exists() or json.loads(Path(f'{prefix}.json').read_text()).get('scope') != scope:
                        prefix = archive
                except ValueError:
                    pass
            try:
                meta = json.loads(Path(f'{prefix}.json').read_text())
                ready &= meta.get('network_sha256') == network_digest(network, *config['inputs'][network])
                ready &= cohort_matches(meta, config, Path(f'{prefix}.json'))
                ready &= not meta.get('uniform', False)
                ready &= all(meta.get(field) == config[field] for field in
                             ('scope', 'seed', 'simulations', 'reviewed_only', 'min_publications'))
                if 'automatic_job_key' in meta:
                    origin = jobs / f"{meta['automatic_job_key']}.status.json"
                    ready &= json.loads(origin.read_text()).get('status') == 'success'
                ready &= Path(f'{prefix}.parquet').is_file() and Path(f'{prefix}_summary.csv').is_file()
                # The final batch summary carries full-family q-values. File
                # timestamps can be reordered by copying/exporting a bundle.
                if 'automatic_job_key' not in meta:
                    summary = Path(f'{prefix}_summary.csv')
                    ready &= corrected_summary(str(summary), summary.stat().st_size, summary.stat().st_mtime_ns)
            except (OSError, ValueError):
                ready = False
        if ready:
            ready_networks.append(network)
    available = {'ready_networks': ready_networks, 'skipped_networks': previous.get('skipped_networks', [])}
    if len(ready_networks) + len(available['skipped_networks']) == len(config['networks']):
        return {'status': 'success', **available}
    lock = jobs / 'writer.json'
    if lock.exists():
        try:
            if alive(json.loads(lock.read_text())):
                return {'status': 'running', **available, 'message': 'Precomputing shortest path and cohesion. Results refresh automatically.'}
        except (ValueError, OSError):
            return {'status': 'running', **available, 'message': 'Starting precomputation…'}
        lock.unlink(missing_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {'status': 'running', **available, 'message': 'Another precompute job is running.'}
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'pid': os.getpid(), 'created': psutil.Process().create_time()}, stream)
    config_path = jobs / f'{key}.json'
    config_path.write_text(json.dumps(config))
    try:
        with (jobs / f'{key}.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(config_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        lock.write_text(json.dumps({'pid': process.pid, 'created': psutil.Process(process.pid).create_time()}))
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    return {'status': 'running', **available, 'message': 'Precomputing shortest path and cohesion. Results refresh automatically.'}


def worker(config_path):
    from pipelines import precompute_monte_carlo as pipeline
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    output = Path(config['data']) / 'analysis_runs'
    staging = config_path.parent / f'{config_path.stem}.results'
    status = config_path.with_suffix('.status.json')
    try:
        args = argparse.Namespace(data=Path(config['data']), networks=list(map(Path, config['networks'])),
            protein_set=config['protein_set'], simulations=config['simulations'], seed=config['seed'],
            analyses=['shortest_path', 'cohesion'], uniform=False, reviewed_only=config['reviewed_only'],
            min_publications=config['min_publications'], scope=config['scope'], dry_run=False,
            verbose=True, no_progress=True)
        pipeline.run_protein_set(args, output_dir=staging, cohort_override=config['cohort'])
        skipped = [path for path in config['networks']
                   if not (staging / f"{config['protein_set']}_{Path(path).stem}_shortest_path.json").exists()]
        for metadata in staging.glob('*.json'):
            record = json.loads(metadata.read_text())
            record['automatic_job_key'] = config_path.stem
            metadata.write_text(json.dumps(record, indent=2))
        # Preserve the alternate scope before publishing a different population.
        for metadata in staging.glob('*.json'):
            current = output / metadata.name
            if current.exists():
                old = json.loads(current.read_text())
                if old.get('scope', 'network') != config['scope']:
                    archive = output / 'archive' / f"monte_carlo_{old.get('scope', 'network')}"
                    archive.mkdir(parents=True, exist_ok=True)
                    for sibling in (current, current.with_suffix('.parquet'), current.with_name(f'{current.stem}_summary.csv')):
                        if sibling.exists():
                            shutil.copy2(sibling, archive / sibling.name)
        for path in staging.glob('*'):
            os.replace(path, output / path.name)
        status.write_text(json.dumps({'status': 'success', 'skipped_networks': skipped}))
    except Exception as exc:
        traceback.print_exc()
        status.write_text(json.dumps({'status': 'failed', 'message': str(exc)}))
    finally:
        (config_path.parent / 'writer.json').unlink(missing_ok=True)


if __name__ == '__main__':
    worker(sys.argv[1])
