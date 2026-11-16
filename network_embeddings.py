"""Sparse nodevectors embeddings and independently cached visualization projections."""
import argparse
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data/analysis_runs/network_embeddings'


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path, method='ProNE', dimensions=32, seed=42):
    if method not in ('ProNE', 'GGVec') or not 2 <= dimensions <= 128:
        raise ValueError('Choose ProNE or GGVec and 2–128 embedding dimensions')
    versions = {}
    for name in ('nodevectors', 'csrgraph', 'numpy', 'scipy', 'scikit-learn', 'gensim', 'numba'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unavailable'
    return dict(source_sha256=fingerprint(path), method=method, dimensions=dimensions,
                seed=seed, versions=versions, code_sha256=fingerprint(__file__),
                topology='undirected binary; no self loops; unique pairs')


def run_directory(path, config, output=OUTPUT):
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
    return Path(output) / Path(path).stem / key


def indexed_adjacency(path):
    edges = pl.read_parquet(path, columns=['a', 'b']).drop_nulls().with_columns(
        pl.col('a').cast(pl.String), pl.col('b').cast(pl.String))
    # Keep loop-only proteins in the population, but remove loops from topology.
    nodes = pl.concat([edges['a'], edges['b']]).unique().sort().to_list()
    edges = edges.filter(pl.col('a') != pl.col('b')).select(
        pl.min_horizontal('a', 'b').alias('a'), pl.max_horizontal('a', 'b').alias('b')).unique()
    index = pl.DataFrame({'protein': nodes, 'index': np.arange(len(nodes))})
    edges = edges.join(index.rename({'protein': 'a', 'index': 'u'}), on='a').join(
        index.rename({'protein': 'b', 'index': 'v'}), on='b')
    u, v = edges['u'].to_numpy(), edges['v'].to_numpy()
    adj = sparse.coo_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])),
                            shape=(len(nodes), len(nodes))).tocsr()
    return nodes, adj, edges.height


def embed(path, method='ProNE', dimensions=32, seed=42, output=OUTPUT):
    config = identity(path, method, dimensions, seed)
    target = run_directory(path, config, output)
    if (target / 'manifest.json').is_file() and (target / 'vectors.parquet').is_file():
        return target
    try:
        import nodevectors
    except ImportError as exc:
        raise RuntimeError('Install the embeddings extra: pip install -e ".[embeddings]". '
                           f'nodevectors import failed: {exc}') from exc
    nodes, adj, edge_count = indexed_adjacency(path)
    if len(nodes) <= dimensions + 1 or edge_count == 0:
        raise ValueError('Network needs edges and more proteins than embedding dimensions + 1')
    # No NetworkX conversion or edge sampling, including networks above 500k edges.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        model = (nodevectors.ProNE(n_components=dimensions, verbose=False) if method == 'ProNE'
                 else nodevectors.GGVec(n_components=dimensions, threads=1, max_epoch=100,
                                        negative_ratio=.05, verbose=False))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            vectors = np.asarray(model.fit_transform(adj))
    finally:
        np.random.set_state(state)
    if vectors.shape != (len(nodes), dimensions) or not np.isfinite(vectors).all():
        raise ValueError('nodevectors returned invalid or non-finite vectors')
    if fingerprint(path) != config['source_sha256']:
        raise ValueError('Source changed during embedding; rerun against the new input')
    target.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({'protein': nodes, 'degree': np.diff(adj.indptr),
                          **{f'e{i}': vectors[:, i] for i in range(dimensions)}})
    frame.write_parquet(target / 'vectors.parquet')
    manifest = dict(config, source=str(Path(path).resolve()), proteins=len(nodes), edges=edge_count,
                    warnings=list(dict.fromkeys(str(item.message) for item in caught)),
                    reproducibility='Seeded ProNE; GGVec uses internal stochastic Numba routines '
                                    'and may vary despite the recorded seed.')
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target


def project(frame, method='UMAP', dimensions=2, seed=42, neighbors=15, min_dist=.1, perplexity=30.):
    values = frame.select(pl.col('^e[0-9]+$')).to_numpy()
    n, width = values.shape
    if dimensions not in (2, 3) or min(n, width) < dimensions or not np.isfinite(values).all():
        raise ValueError('Projection needs finite features and enough proteins and dimensions')
    if method == 'PCA':
        from sklearn.decomposition import PCA
        coordinates = PCA(n_components=dimensions, random_state=seed).fit_transform(values)
    elif method == 'UMAP':
        from umap import UMAP
        coordinates = UMAP(n_components=dimensions, random_state=seed, n_jobs=1,
                           n_neighbors=min(neighbors, n - 1), min_dist=min_dist,
                           init='random').fit_transform(values)
    elif method == 't-SNE':
        from sklearn.manifold import TSNE
        coordinates = TSNE(n_components=dimensions, random_state=seed, init='pca',
                           learning_rate='auto', perplexity=min(perplexity, n - 1)).fit_transform(values)
    else:
        raise ValueError('Unknown projection method')
    if not np.isfinite(coordinates).all():
        raise ValueError('Projection returned non-finite coordinates')
    return frame.select('protein', 'degree').with_columns(
        [pl.Series(axis, coordinates[:, i]) for i, axis in enumerate('xyz'[:dimensions])])


def saved_projection(directory, **parameters):
    directory = Path(directory)
    config = dict(parameters, vectors_sha256=fingerprint(directory / 'vectors.parquet'),
                  code_sha256=fingerprint(__file__),
                  versions={name: importlib.metadata.version(name) for name in ('scikit-learn', 'umap-learn')})
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
    target = directory / f'projection_{key}.parquet'
    if not target.is_file():
        points = project(pl.read_parquet(directory / 'vectors.parquet'), **parameters)
        points.write_parquet(target)
        target.with_suffix('.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return pl.read_parquet(target)


def main():
    from networks import discover_datasets
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ProNE', 'GGVec'], default='ProNE')
    parser.add_argument('--dimensions', type=int, default=32)
    parser.add_argument('--isoforms', choices=['collapse', 'retain', 'canonical_only'], default='collapse')
    parser.add_argument('--networks', nargs='+', help='Omit to embed every discovered network and saved version')
    args = parser.parse_args()
    datasets = discover_datasets(ROOT / 'data', isoforms=args.isoforms)
    if args.networks and set(args.networks) - datasets.keys():
        parser.error(f'Unknown networks: {set(args.networks) - datasets.keys()}')
    failures = []
    for name, path in datasets.items():
        if args.networks and name not in args.networks:
            continue
        print(f'Embedding {name} ({args.isoforms})', flush=True)
        try:
            print(embed(path, args.method, args.dimensions), flush=True)
        except Exception as exc:
            failures.append(name)
            print(f'FAILED {name}: {exc}', flush=True)
    if failures:
        raise SystemExit(f'Failed networks: {failures}')


if __name__ == '__main__':
    main()
