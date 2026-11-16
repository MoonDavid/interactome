"""Batch ClusterONE CLI execution, with optional Cytoscape 3 visualization.

Run from the repository root; see README.md for examples.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_edges(path, weight_column, max_edges):
    import polars as pl

    frame = pl.read_parquet(path).select('a', 'b', *([weight_column] if weight_column else []))
    if frame.height > max_edges:
        raise ValueError(f'{path}: {frame.height:,} edges exceeds {max_edges:,}; select a threshold version.')
    for column in ('a', 'b'):
        if frame[column].null_count() or frame.filter(
            pl.col(column).str.contains(r'\s') | (pl.col(column) == '')
        ).height:
            raise ValueError(f'{path}: empty, null or whitespace-containing node IDs')
    frame = frame.filter(pl.col('a') != pl.col('b')).with_columns(
        pl.min_horizontal('a', 'b').alias('a'), pl.max_horizontal('a', 'b').alias('b')
    )
    if weight_column:
        frame = frame.with_columns(pl.col(weight_column).cast(pl.Float64, strict=True))
        weights = pl.col(weight_column)
        if frame.filter((weights.is_null() | ~weights.is_finite() | (weights < 0) | (weights > 1)).fill_null(True)).height:
            raise ValueError('Weights must all be finite values in [0,1]; use unweighted mode or a filtered version.')
        frame = frame.group_by('a', 'b').agg(weights.max())
    else:
        frame = frame.unique(subset=['a', 'b'])
    if not frame.height:
        raise ValueError(f'{path}: no non-loop edges')
    return frame.sort('a', 'b')


def publish(edges, memberships, title, base_url):
    members = defaultdict(list)
    for protein, cluster in memberships:
        members[protein].append(cluster)
    proteins = sorted(set(edges['a']) | set(edges['b']))
    payload = {'data': {'name': title}, 'elements': {
        'nodes': [{'data': {'id': p, 'name': p,
                           'clusterone_clusters': ';'.join(members[p]),
                           'clusterone_count': len(members[p])}} for p in proteins],
        'edges': [{'data': {'id': f'e{i}', 'source': a, 'target': b}}
                  for i, (a, b) in enumerate(edges.select('a', 'b').iter_rows())]}}
    request = Request(base_url.rstrip('/') + '/networks?collection=ClusterONE',
                      data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('networks', nargs='+', type=Path, help='Canonical Parquets with a/b endpoint columns')
    parser.add_argument('--jar', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'tools' / 'clusterone' / 'cluster_one-1.0.jar',
                        help='ClusterONE executable JAR (default: tools/clusterone/cluster_one-1.0.jar)')
    parser.add_argument('--java', default='java', help='Java executable path')
    parser.add_argument('--min-size', type=int, default=3)
    parser.add_argument('--min-density', type=float, default=0.3)
    parser.add_argument('--weight-column', help='Explicit confidence column; default: unweighted')
    parser.add_argument('--max-edges', type=int, default=500_000, help='Guard; cannot exceed 500000')
    parser.add_argument('--timeout', type=float, default=3600, help='Java timeout per network in seconds')
    parser.add_argument('--output', type=Path, default=Path('data/analysis_runs/clusterone'))
    parser.add_argument('--cytoscape', action='store_true', help='Also import annotated networks into running Cytoscape')
    parser.add_argument('--base-url', default='http://127.0.0.1:1234/v1')
    args = parser.parse_args()
    if not args.jar.is_file():
        parser.error('ClusterONE JAR does not exist')
    if args.min_size < 1 or not 0 <= args.min_density <= 1 or not 1 <= args.max_edges <= 500_000 or args.timeout <= 0:
        parser.error('Invalid size, density, edge guard or timeout')
    import polars as pl

    jar_hash = sha256(args.jar)
    for path in args.networks:
        edges = load_edges(path, args.weight_column, args.max_edges)
        run = args.output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + path.stem)
        run.mkdir(parents=True, exist_ok=False)
        edge_path = run / 'edges.tsv'
        edges.write_csv(edge_path, separator='\t', include_header=False)
        command = [args.java, '-jar', str(args.jar.resolve()),
                   '--output-format', 'plain', '--min-size', str(args.min_size),
                   '--min-density', str(args.min_density), str(edge_path.resolve())]
        manifest = {'source': str(path.resolve()), 'source_sha256': sha256(path),
                    'jar_sha256': jar_hash, 'command': command, 'parameters': {
                        'min_size': args.min_size, 'min_density': args.min_density,
                        'weight_column': args.weight_column}, 'edges': edges.height,
                    'status': 'started'}
        manifest_path = run / 'manifest.json'
        def save_manifest():
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        save_manifest()
        try:
            with (run / 'clusters.txt').open('w', encoding='utf-8') as out, (run / 'stderr.log').open('w', encoding='utf-8') as err:
                subprocess.run(command, stdout=out, stderr=err, check=True, timeout=args.timeout)
            proteins = set(edges['a']) | set(edges['b'])
            memberships = []
            for index, line in enumerate((run / 'clusters.txt').read_text(encoding='utf-8').splitlines(), 1):
                for protein in sorted(set(line.split())):
                    if protein not in proteins:
                        raise ValueError(f'Unexpected ClusterONE output node: {protein}')
                    memberships.append((protein, str(index)))
            pl.DataFrame(memberships, schema={'protein': pl.String, 'cluster': pl.String}, orient='row').write_csv(run / 'memberships.csv')
            manifest['status'] = 'clustered'
            save_manifest()
            if args.cytoscape:
                manifest['cytoscape'] = publish(edges, memberships, path.stem + ' ClusterONE', args.base_url)
            manifest['status'] = 'complete'
            save_manifest()
        except Exception as exc:
            manifest['error'] = str(exc)
            save_manifest()
            raise
        print(f'{path.stem}: saved {run}')


if __name__ == '__main__':
    main()
