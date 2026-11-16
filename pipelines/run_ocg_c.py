"""Compile and run the original C OCG implementation on a canonical Parquet network.

Example:
    python -m pipelines.run_ocg_c --input data/processed_networks/HuRI.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'ocg' / 'bin' / ('ocg_fast.exe' if sys.platform == 'win32' else 'ocg_fast')
METHOD = 'c.ocg'


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), size=path.stat().st_size)


def prepare_edges(path):
    """Read canonical accession-space Parquet without altering the source."""
    edges = pl.read_parquet(path).select(pl.col('a', 'b').cast(pl.String).str.strip_chars())
    edges = edges.drop_nulls().filter((pl.col('a') != '') & (pl.col('b') != '') &
                                    (pl.col('a') != pl.col('b')))
    edges = edges.select(pl.min_horizontal('a', 'b').alias('a'),
                         pl.max_horizontal('a', 'b').alias('b')).unique().sort('a', 'b')
    if edges.is_empty():
        raise ValueError('No non-self edges in the selected network')
    nodes = pl.concat([edges.select(pl.col('a').alias('protein')),
                       edges.select(pl.col('b').alias('protein'))]).unique().sort('protein')
    nodes = nodes.with_row_index('node', offset=1).with_columns(pl.col('node').cast(pl.String))
    indexed = (edges.join(nodes.rename({'protein': 'a', 'node': 'i'}), on='a')
        .join(nodes.rename({'protein': 'b', 'node': 'j'}), on='b').select('i', 'j'))
    return indexed, nodes


def decode_partition(path: Path, nodes: pl.DataFrame) -> pl.DataFrame:
    """Parse OCG's >ClusterID output into a (protein, community) table."""
    rows: list[tuple[str, int]] = []
    cluster: int | None = None

    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('>ClusterID:'):
            # Convert 1-based ">ClusterID:1||" labels to 0-based integer IDs.
            cluster = int(line.removeprefix('>ClusterID:').split('|', 1)[0]) - 1
        elif cluster is not None:
            rows.append((line, cluster))

    if rows:
        members = (
            pl.DataFrame(rows, schema={'node': pl.String, 'community': pl.Int64}, orient='row')
            .unique()
            .join(nodes, on='node')
            .select('protein', 'community')
        )
    else:
        members = pl.DataFrame(schema={'protein': pl.String, 'community': pl.Int64})

    # Keep proteins with no cluster assignment explicitly as community = null.
    assigned_proteins = members.select('protein').unique()
    unassigned = nodes.join(assigned_proteins, on='protein', how='anti').select(
        'protein',
        pl.lit(None, dtype=pl.Int64).alias('community'),
    )
    return pl.concat([members, unassigned]).sort('protein', 'community')


def run(args: argparse.Namespace) -> Path:
    """Prepare the network, execute C OCG, and save standardized outputs."""
    if not BINARY.is_file():
        raise FileNotFoundError(f'Compiled OCG binary not found: {BINARY}')
    threads = getattr(args, 'threads', None)
    if threads is None:
        threads = min(8, os.cpu_count() or 1)
    if threads < 1:
        raise ValueError('threads must be positive')
    edges, nodes = prepare_edges(args.input)

    # OCG.c allocates static arrays for MaxProt=30,000 nodes; also enforce the 500k edge guard.
    if nodes.height > 30_000 or edges.height > 500_000:
        raise ValueError(
            f'Graph exceeds OCG limits ({nodes.height:,} nodes / {edges.height:,} edges; max 30,000 / 500,000)'
        )

    # Hash inputs and parameters so identical runs are reused instead of recomputed.
    params = {
        'init_class_sys': args.init_class_sys,
        'max_class_card': args.max_class_card,
        'cent_class_sys': args.cent_class_sys,
        'min_class': args.min_class,
    }
    job = {
        'network': args.network,
        'method': METHOD,
        'parameters': params,
        'provenance': {'input': fingerprint(args.input), 'c_binary': fingerprint(BINARY)},
    }
    key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    target = args.output.resolve() / args.network / METHOD / key[:16]

    result_file = target / 'result.json'
    if result_file.is_file() and json.loads(result_file.read_text(encoding='utf-8')).get('status') == 'success':
        print(f'Reusing saved C OCG run: {target}')
        return target

    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # Run inside a temp folder using short relative filenames because OCG.c uses
    # fixed 120-character buffers for the input and partition file paths.
    with TemporaryDirectory(prefix='ocg_c_') as tmp:
        work = Path(tmp)
        edges.write_csv(work / 'edges.txt', separator=' ', include_header=False)
        nodes.write_parquet(target / 'node_map.parquet')

        cmd = [str(BINARY)]
        if args.init_class_sys == 1:
            cmd.append('-m')
        elif args.init_class_sys == 2:
            cmd.append('-e')
        else:
            cmd.extend(['-c', str(args.cent_class_sys)])
        if args.max_class_card:
            cmd.extend(['-s', str(args.max_class_card)])
        if args.min_class:
            cmd.extend(['-n', str(args.min_class)])
        cmd.extend(['-v', '-p', 'ocg_partition.txt', 'edges.txt'])

        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE')
        with (target / 'worker.log').open('w', encoding='utf-8') as log:
            proc = subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=args.timeout, env=env)
        if proc.returncode:
            raise ValueError(f'C OCG failed (exit {proc.returncode}); see {target / "worker.log"}')

        members = decode_partition(work / 'ocg_partition.txt', nodes)
        shutil.copy2(work / 'ocg_partition.txt', target / 'ocg_partition.txt')

    # Save memberships, cluster sizes, and summary metadata for the explorer.
    assigned = members.drop_nulls('community')
    sizes = assigned.group_by('community').len().sort('community')
    members.write_parquet(target / 'memberships.parquet')
    sizes.write_parquet(target / 'community_sizes.parquet')

    covered = assigned['protein'].n_unique()
    overlapping = assigned.group_by('protein').len().filter(pl.col('len') > 1).height
    result = {
        'method': METHOD,
        'key': key,
        'status': 'success',
        'job': job,
        'nodes': nodes.height,
        'edges': edges.height,
        'communities': sizes.height,
        'covered_nodes': covered,
        'unassigned_nodes': nodes.height - covered,
        'overlapping_nodes': overlapping,
        'elapsed_seconds': time.monotonic() - started,
        'execution': {'requested_threads': threads},
    }
    result_file.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Saved C OCG run: {target}')
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Canonical Parquet with a/b accession columns')
    parser.add_argument('--network', help='Network label (defaults to input filename stem)')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/analysis_runs/clustering_raw')
    parser.add_argument('--init-class-sys', type=int, choices=[1, 2, 3], default=3,
                        help='1: maximal cliques, 2: edges, 3: centered cliques (default)')
    parser.add_argument('--max-class-card', type=int, default=0, help='Max cluster size (0 = unconstrained)')
    parser.add_argument('--cent-class-sys', type=int, choices=[0, 1], default=1,
                        help='0: stop by constraints, 1: maximize modularity (default)')
    parser.add_argument('--min-class', type=int, default=2, help='Minimum expected clusters (0 = unconstrained)')
    parser.add_argument('--timeout', type=float, default=3600, help='Execution timeout in seconds')
    parser.add_argument('--threads', type=int, default=None,
                        help='OpenMP workers (default: up to 8 CPUs); use 1 for serial execution')
    args = parser.parse_args()
    args.network = args.network or args.input.stem
    run(args)


if __name__ == '__main__':
    main()
