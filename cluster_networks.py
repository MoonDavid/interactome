"""Batch community discovery: explicit adapters, isolated jobs, inspectable results.

Run ``python cluster_networks.py --help``. Optional backends are imported only in
workers; listing methods never imports heavyweight or GPU packages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Annotated

import numpy as np
import polars as pl
from scipy import sparse
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from networks import _fingerprint, _write_json, _write_parquet, discover_datasets, load_network

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Cluster canonical interactomes with native and research backends.")
console = Console()




MODULES = [
    'cdlib.algorithms', 'networkx.algorithms.community', 'igraph', 'leidenalg',
    'sknetwork.clustering', 'sknetwork.hierarchy', 'graspologic.partition',
    'graspologic.cluster', 'karateclub', 'sklearn.cluster', 'networkit.community',
    'cugraph.community', 'infomap', 'markov_clustering', 'hdbscan',
]
PACKAGES = ['polars', 'numpy', 'scipy', 'networkx', 'typer', 'rich', 'igraph',
            'leidenalg', 'scikit-network', 'scikit-learn', 'cdlib', 'graspologic',
            'karateclub', 'networkit', 'cugraph-cu12', 'cudf-cu12', 'infomap',
            'markov-clustering', 'hdbscan']
SCHEMA_VERSION = 1
EDGE_LIMIT = 500_000


@dataclass(frozen=True)
class Method:
    id: str
    backend: str
    name: str
    family: str = 'hard'
    costly: bool = False
    dense: bool = False


def catalog() -> dict[str, Method]:
    from moonlighting import CLUSTERING_CATALOG
    methods = [Method(f'atlas.{name}', 'atlas', name, spec['paradigm'],
                      name not in {'louvain', 'label_propagation', 'k_core_shells'})
               for name, spec in CLUSTERING_CATALOG.items()]

    def add(backend, names, family='hard', costly=False, dense=False):
        methods.extend(Method(f'{backend}.{name}', backend, name, family, costly, dense)
                       for name in names.split())

    add('networkx', 'louvain_communities fast_label_propagation_communities '
        'asyn_lpa_communities label_propagation_communities')
    add('networkx', 'greedy_modularity_communities girvan_newman', 'hierarchical', True)
    add('networkx', 'k_clique_communities', 'overlapping', True)
    add('igraph', 'multilevel leiden infomap label_propagation')
    add('igraph', 'fastgreedy walktrap leading_eigenvector edge_betweenness', 'hierarchical', True)
    add('leidenalg', 'RBConfigurationVertexPartition CPMVertexPartition ModularityVertexPartition')
    add('sknetwork', 'Louvain Leiden PropagationClustering')
    add('sknetwork_hierarchy', 'Paris LouvainHierarchy', 'hierarchical', True)
    add('graspologic_partition', 'leiden hierarchical_leiden')
    add('graspologic_cluster', 'AutoGMMCluster KMeansCluster', 'embedding', True)
    add('sklearn', 'KMeans MiniBatchKMeans Birch DBSCAN OPTICS MeanShift', 'embedding', True)
    add('sklearn', 'AffinityPropagation AgglomerativeClustering', 'embedding', True, True)
    add('sklearn', 'SpectralClustering', 'spectral', True)
    add('cdlib', 'louvain leiden infomap label_propagation')
    add('cdlib', 'demon ego_networks slpa angel kclique', 'overlapping', True)
    add('cdlib', 'walktrap surprise_communities significance_communities', 'hard', True)
    add('karateclub', 'BigClam DANMF MNMF NNSED SymmNMF', 'soft', True)
    add('karateclub', 'EgoNetSplitter', 'overlapping', True)
    add('karateclub', 'EdMot SCD GEMSEC', 'hard', True)
    add('networkit', 'PLM PLP ParallelLeiden')
    add('cugraph', 'louvain leiden ecg')
    add('infomap', 'Infomap')
    add('markov_clustering', 'mcl', 'flow', True)
    add('hdbscan', 'HDBSCAN', 'embedding', True)
    return {m.id: m for m in methods}


def versions() -> dict:
    result = {'python': sys.version, 'platform': sys.platform}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_digest() -> str:
    digest = hashlib.sha256()
    for name in ('cluster_networks.py', 'moonlighting.py', 'networks.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


@app.command()
def inventory(output: Annotated[Path | None, typer.Option(help='Save callable names, signatures and errors as JSON.')] = None,
              timeout: Annotated[float, typer.Option(min=1)] = 60):
    """Inspect every requested module (including unavailable packages), in isolation."""
    results = {}
    for module in MODULES:
        with console.status(f'Inspecting {module} ...'):
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                       '_inspect', module], capture_output=True,
                                      text=True, timeout=timeout)
                result = json.loads(proc.stdout) if proc.returncode == 0 else {
                    'error': proc.stderr[-4000:] or f'exit {proc.returncode}'}
            except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
                result = {'error': str(exc)}
        results[module] = result
        if 'error' in result:
            console.print(f'[yellow]{module}: unavailable[/yellow]', markup=True)
            console.print(result['error'], markup=False)
        else:
            console.rule(f'{module}: {len(result["callables"])} public callables')
            for entry in result['callables']:
                console.print(f'  {entry["name"]}{entry["signature"]}', markup=False)
    if output:
        _write_json(output, results)


@app.command('methods')
def list_methods():
    """List executable adapters; inventory also includes non-clustering utilities."""
    table = Table(title='Clustering adapters (unweighted, undirected graphs)')
    for title in ('ID', 'Family', 'Cost guard'):
        table.add_column(title)
    for method in catalog().values():
        table.add_row(method.id, method.family,
                      'node limit' if method.costly or method.dense else 'timeout')
    console.print(table)


def select_methods(patterns: list[str]) -> list[Method]:
    available = catalog()
    if not patterns or patterns == ['all']:
        return list(available.values())
    selected = {}
    for pattern in patterns:
        matches = [m for m in available.values() if pattern == m.id or pattern == m.backend]
        if not matches:
            raise typer.BadParameter(f'Unknown method/backend: {pattern}. Use methods to list IDs.')
        selected.update({m.id: m for m in matches})
    return list(selected.values())


def indexed_graph(path: Path) -> dict:
    # The canonical loader preserves validated pair semantics and deterministic order.
    edges = load_network(path).edges
    nodes = pl.concat([edges.select(pl.col('a').alias('protein')),
                       edges.select(pl.col('b').alias('protein'))]).unique().sort('protein')
    lookup = nodes.with_row_index('idx')
    indexed = (edges.join(lookup, left_on='a', right_on='protein', validate='m:1')
               .rename({'idx': 'u'})
               .join(lookup, left_on='b', right_on='protein', validate='m:1')
               .rename({'idx': 'v'}).select('u', 'v'))
    u, v = indexed['u'].to_numpy(), indexed['v'].to_numpy()
    n = nodes.height
    adj = sparse.coo_matrix((np.ones(2 * len(u), dtype=float),
                            (np.concatenate([u, v]), np.concatenate([v, u]))),
                           shape=(n, n)).tocsr()
    return {'nodes': nodes['protein'].to_numpy(), 'adj': adj, 'u': u, 'v': v,
            'degree': np.diff(adj.indptr), 'edge_count': edges.height}


def nx_graph(graph):
    if graph['edge_count'] > EDGE_LIMIT:
        raise ValueError('NetworkX guard: more than 500,000 edges')
    import networkx as nx
    return nx.from_scipy_sparse_array(graph['adj'])


def ig_graph(graph):
    import igraph as ig
    return ig.Graph(n=len(graph['nodes']), edges=list(zip(graph['u'], graph['v'])), directed=False)


def embedding(graph, params):
    """Normalized spectral embedding, not raw adjacency treated as observations."""
    from sklearn.manifold import spectral_embedding
    from sklearn.preprocessing import StandardScaler
    n = len(graph['nodes'])
    dimensions = min(params['dimensions'], n - 2)
    if dimensions < 1:
        raise ValueError('Embedding methods require at least three incident nodes')
    values = spectral_embedding(graph['adj'], n_components=dimensions,
                                random_state=params['seed'], drop_first=True)
    return StandardScaler().fit_transform(values)


def groups_from_labels(labels):
    buckets = {}
    for node, label in enumerate(labels):
        if int(label) >= 0:  # -1 is noise/unassigned; never invent memberships.
            buckets.setdefault(int(label), set()).add(node)
    return list(buckets.values())


def execute(method: Method, graph: dict, p: dict):
    """Return communities, optional raw soft memberships, and explicit semantics."""
    n, seed, resolution = len(graph['nodes']), p['seed'], p['resolution']
    k = min(p['clusters'], n)
    extra, soft = {}, None
    backend, name = method.backend, method.name
    if backend == 'atlas':
        from moonlighting import _execute_clustering_algorithm
        communities, soft, extra = _execute_clustering_algorithm(
            graph, nx_graph(graph), algorithm=name, resolution=resolution,
            seed=seed, n_clusters=k, clique_k=p['clique_k'],
            mcl_inflation=p['inflation'], membership_cutoff=p['membership_cutoff'])
        extra['implementation'] = 'existing Atlas implementation; see CLUSTERING_CATALOG descriptions'
    elif backend == 'networkx':
        import networkx as nx
        network = nx_graph(graph)
        kwargs = {}
        if name in {'louvain_communities', 'asyn_lpa_communities', 'fast_label_propagation_communities'}:
            kwargs['seed'] = seed
        if name in {'louvain_communities', 'greedy_modularity_communities'}:
            kwargs['resolution'] = resolution
        if name == 'k_clique_communities':
            kwargs['k'] = p['clique_k']
        result = getattr(nx.community, name)(network, **kwargs)
        communities = list(next(result)) if name == 'girvan_newman' else list(result)
        if name == 'girvan_newman':
            extra['hierarchy_cut'] = 'first split'
    elif backend == 'igraph':
        import igraph as ig
        ig.set_random_number_generator(random.Random(seed))
        kwargs = {'resolution': resolution} if name in {'multilevel', 'leiden'} else {}
        if name == 'leiden':
            kwargs.update(objective_function='modularity', n_iterations=-1)
        result = getattr(ig_graph(graph), f'community_{name}')(**kwargs)
        if hasattr(result, 'as_clustering'):
            result = result.as_clustering()
            extra['hierarchy_cut'] = 'backend optimal modularity cut'
        communities = list(result)
    elif backend == 'leidenalg':
        import leidenalg as la
        kwargs = {} if name == 'ModularityVertexPartition' else {'resolution_parameter': resolution}
        communities = list(la.find_partition(ig_graph(graph), getattr(la, name),
                                             seed=seed, n_iterations=-1, **kwargs))
        extra['objective'] = name
    elif backend in {'sknetwork', 'sknetwork_hierarchy'}:
        mod = importlib.import_module('sknetwork.clustering' if backend == 'sknetwork' else 'sknetwork.hierarchy')
        cls = getattr(mod, name)
        kwargs = {}
        signature = inspect.signature(cls).parameters
        if 'random_state' in signature:
            kwargs['random_state'] = seed
        if 'resolution' in signature:
            kwargs['resolution'] = resolution
        result = cls(**kwargs).fit_predict(graph['adj'])
        if backend == 'sknetwork_hierarchy':
            from sknetwork.hierarchy import cut_straight
            result = cut_straight(result, n_clusters=k)
            extra['hierarchy_cut'] = {'n_clusters': k}
        communities = groups_from_labels(result)
    elif backend == 'graspologic_partition':
        from graspologic import partition
        # Native sparse edge list avoids unnecessary NetworkX conversion.
        edges = [(int(u), int(v), 1.0) for u, v in zip(graph['u'], graph['v'])]
        result = getattr(partition, name)(edges, resolution=resolution, random_seed=seed)
        if name == 'hierarchical_leiden':
            result = result.final_level_hierarchical_clustering()
            extra['hierarchy_cut'] = 'final level'
        communities = groups_from_labels([result.get(i, -1) for i in range(n)])
    elif backend in {'sklearn', 'graspologic_cluster', 'hdbscan'}:
        module = {'sklearn': 'sklearn.cluster', 'graspologic_cluster': 'graspologic.cluster',
                  'hdbscan': 'hdbscan'}[backend]
        cls = getattr(importlib.import_module(module), name)
        if name == 'SpectralClustering':
            result = cls(n_clusters=k, affinity='precomputed', random_state=seed).fit_predict(graph['adj'])
            extra['input'] = 'sparse binary adjacency affinity'
        else:
            features = embedding(graph, p)
            kwargs = {
                'KMeans': {'n_clusters': k, 'random_state': seed, 'n_init': 10},
                'MiniBatchKMeans': {'n_clusters': k, 'random_state': seed, 'n_init': 10},
                'Birch': {'n_clusters': k}, 'AgglomerativeClustering': {'n_clusters': k},
                'AffinityPropagation': {'random_state': seed},
                'AutoGMMCluster': {'min_components': 2, 'max_components': k, 'random_state': seed},
                'KMeansCluster': {'max_clusters': k, 'random_state': seed},
            }.get(name, {})
            result = cls(**kwargs).fit_predict(features)
            extra['input'] = 'standardized normalized spectral embedding'
            extra['embedding_dimensions_actual'] = features.shape[1]
        communities = groups_from_labels(result)
    elif backend == 'cdlib':
        from cdlib import algorithms
        kwargs = {
            'louvain': {'resolution': resolution, 'randomize': seed},
            'demon': {'epsilon': 0.25}, 'slpa': {'t': 100, 'r': 0.1},
            'angel': {'threshold': 0.25}, 'kclique': {'k': p['clique_k']},
        }.get(name, {})
        communities = getattr(algorithms, name)(nx_graph(graph), **kwargs).communities
    elif backend == 'karateclub':
        mod = importlib.import_module('karateclub')
        cls = getattr(mod, name)
        kwargs = {'seed': seed} if 'seed' in inspect.signature(cls).parameters else {}
        if 'dimensions' in inspect.signature(cls).parameters:
            kwargs['dimensions'] = k
        model = cls(**kwargs)
        model.fit(nx_graph(graph))
        if method.family == 'soft':
            soft = np.asarray(model.get_embedding())
            communities = groups_from_labels(soft.argmax(axis=1))
            extra['soft_semantics'] = 'raw nonnegative affiliation strengths; primary groups use argmax'
        else:
            memberships = model.get_memberships()
            buckets = {}
            for node, labels in memberships.items():
                for label in (labels if isinstance(labels, (list, set, tuple)) else [labels]):
                    buckets.setdefault(label, set()).add(int(node))
            communities = list(buckets.values())
    elif backend == 'networkit':
        import networkit as nk
        nk.setSeed(seed, False)
        nk.setNumberOfThreads(p['threads'])
        network = nk.Graph(n, weighted=False, directed=False)
        for u, v in zip(graph['u'], graph['v']):
            network.addEdge(int(u), int(v))
        kwargs = {'gamma': resolution} if name in {'PLM', 'ParallelLeiden'} else {}
        model = getattr(nk.community, name)(network, **kwargs).run()
        communities = groups_from_labels(model.getPartition().getVector())
        extra['determinism'] = 'parallel algorithms may vary despite seed'
    elif backend == 'cugraph':
        import cudf
        import cugraph
        network = cugraph.Graph(directed=False)
        edges = cudf.DataFrame({'src': graph['u'], 'dst': graph['v']})
        network.from_cudf_edgelist(edges, source='src', destination='dst', renumber=False)
        func = getattr(cugraph, name)
        kwargs = {'resolution': resolution} if 'resolution' in inspect.signature(func).parameters else {}
        if 'random_state' in inspect.signature(func).parameters:
            kwargs['random_state'] = seed
        result, _ = func(network, **kwargs)
        # GPU arrays cross the scientific-library boundary without pandas.
        vertices = result['vertex'].to_cupy().get()
        labels = np.full(n, -1, dtype=np.int64)
        labels[vertices] = result['partition'].to_cupy().get()
        communities = groups_from_labels(labels)
    elif backend == 'infomap':
        from infomap import Infomap
        model = Infomap(f'--two-level --silent --seed {max(1, seed)} --num-trials 10')
        for u, v in zip(graph['u'], graph['v']):
            model.add_link(int(u), int(v))
        model.run()
        memberships = model.get_modules()
        communities = groups_from_labels([memberships.get(i, -1) for i in range(n)])
    elif backend == 'markov_clustering':
        import markov_clustering as mc
        result = mc.run_mcl(graph['adj'], inflation=p['inflation'])
        communities = list(mc.get_clusters(result))
    else:
        raise ValueError(f'No adapter for {method.id}')
    return communities, soft, extra


def membership_table(nodes, communities):
    """Long-form cover with null rows for uncovered nodes; no synthetic assignments."""
    n = len(nodes)
    groups = {tuple(sorted({int(i) for i in members})) for members in communities if len(members)}
    groups = sorted(groups, key=lambda members: (-len(members), members))
    rows = []
    covered = set()
    for cluster, members in enumerate(groups):
        if any(i < 0 or i >= n for i in members):
            raise ValueError('Backend returned node indices outside the input population')
        for i in members:
            rows.append((str(nodes[i]), cluster))
            covered.add(i)
    rows.extend((str(nodes[i]), None) for i in range(n) if i not in covered)
    frame = pl.DataFrame(rows, schema={'protein': pl.String, 'community': pl.Int64}, orient='row')
    sizes = frame.filter(pl.col('community').is_not_null()).group_by('community').len().sort('community')
    return frame.sort('protein', 'community'), sizes, len(covered)


def _worker(job: dict) -> int:
    target = Path(job['target'])
    started = time.monotonic()
    result = {'method': job['method'], 'key': job['key'], 'status': 'failed'}
    try:
        p = job['parameters']
        random.seed(p['seed'])
        np.random.seed(p['seed'])
        method = catalog()[job['method']]
        edge_count = pl.scan_parquet(job['path']).select(pl.len()).collect().item()
        if method.backend in {'atlas', 'networkx', 'cdlib', 'karateclub'} and edge_count > EDGE_LIMIT:
            result.update(status='skipped', reason='NetworkX guard: more than 500,000 edges')
        else:
            graph = indexed_graph(Path(job['path']))
            n = len(graph['nodes'])
            limit = p['dense_node_limit'] if method.dense else p['costly_node_limit']
            if not n:
                result.update(status='skipped', reason='Empty graph after canonical cleaning')
            elif (method.costly or method.dense) and n > limit:
                result.update(status='skipped', reason=f'Cost guard: {n:,} nodes exceeds {limit:,}')
            else:
                communities, soft, extra = execute(method, graph, p)
                frame, sizes, covered = membership_table(graph['nodes'], communities)
                _write_parquet(target / 'memberships.parquet', frame)
                _write_parquet(target / 'community_sizes.parquet', sizes)
                if soft is not None:
                    soft = np.asarray(soft)
                    if soft.ndim != 2 or soft.shape[0] != n or not np.isfinite(soft).all():
                        raise ValueError('Invalid soft membership matrix')
                    np.savez_compressed(target / 'soft_memberships.npz',
                                        proteins=graph['nodes'].astype(str), memberships=soft)
                    extra['soft_column_order'] = 'original backend order, independent of normalized community IDs'
                result.update(status='success', nodes=n, edges=graph['edge_count'],
                              communities=sizes.height, covered_nodes=covered,
                              unassigned_nodes=n - covered,
                              overlapping_nodes=frame.filter(pl.col('community').is_not_null())
                              .group_by('protein').len().filter(pl.col('len') > 1).height,
                              details=extra)
    except (ImportError, ModuleNotFoundError) as exc:
        result.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    except Exception as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    result['job'] = job
    _write_json(target / 'result.json', result)
    return 0 if result['status'] in {'success', 'skipped', 'unavailable'} else 1


def reusable(target: Path, key: str) -> dict | None:
    try:
        result = json.loads((target / 'result.json').read_text(encoding='utf-8'))
        if (result['status'] == 'success' and result['key'] == key
                and (target / 'memberships.parquet').is_file()
                and (target / 'community_sizes.parquet').is_file()):
            if result.get('details', {}).get('soft_column_order') and not (target / 'soft_memberships.npz').is_file():
                return None
            return result
    except (OSError, ValueError, KeyError):
        pass
    return None


@app.command()
def run(
    root: Annotated[Path, typer.Option(help='Existing prepared data root.')] = Path('data'),
    output: Annotated[Path, typer.Option(help='Results directory; reuse to resume.')] = Path('data/analysis_runs/clustering'),
    network: Annotated[list[str] | None, typer.Option(help='Repeat dataset names; default: all discovered networks/versions.')] = None,
    method: Annotated[list[str] | None, typer.Option(help='Repeat adapter IDs or backend names; default: all.')] = None,
    seed: Annotated[int, typer.Option(min=1, max=2**32 - 1)] = 42,
    resolution: Annotated[float, typer.Option(min=0.000001)] = 1.0,
    clusters: Annotated[int, typer.Option(min=2)] = 16,
    dimensions: Annotated[int, typer.Option(min=1)] = 16,
    clique_k: Annotated[int, typer.Option(min=2)] = 3,
    inflation: Annotated[float, typer.Option(min=1.01)] = 2.0,
    membership_cutoff: Annotated[float, typer.Option(min=0, max=1)] = 0.2,
    timeout: Annotated[float, typer.Option(min=1, help='Seconds per worker, including startup and IO.')] = 300,
    costly_node_limit: Annotated[int, typer.Option(min=1, help='Raise explicitly for costly methods.')] = 5000,
    dense_node_limit: Annotated[int, typer.Option(min=1)] = 3000,
    threads: Annotated[int, typer.Option(min=1)] = 1,
    resume: Annotated[bool, typer.Option('--resume/--no-resume')] = True,
    dry_run: Annotated[bool, typer.Option(help='Write a plan without executing clustering.')] = False,
):
    """Run all selected methods on all discovered major networks, one isolated job at a time."""
    datasets = discover_datasets(root)
    names = network or list(datasets)
    missing = set(names) - set(datasets)
    if missing:
        raise typer.BadParameter(f'Unknown/unprepared datasets: {sorted(missing)}; available: {list(datasets)}')
    if not names:
        raise typer.BadParameter('No prepared canonical datasets found. Prepare networks first.')
    methods = select_methods(method or ['all'])
    params = dict(seed=seed, resolution=resolution, clusters=clusters, dimensions=dimensions,
                  clique_k=clique_k, inflation=inflation, membership_cutoff=membership_cutoff,
                  costly_node_limit=costly_node_limit, dense_node_limit=dense_node_limit, threads=threads)
    if not all(math.isfinite(v) for v in (resolution, inflation, membership_cutoff, timeout)):
        raise typer.BadParameter('Numeric options must be finite')
    output = output.resolve()
    environment, code_hash = versions(), source_digest()
    jobs = []
    for name in dict.fromkeys(names):
        path = datasets[name].resolve()
        provenance = {'input': _fingerprint(path)}
        if path.with_suffix('.json').exists():
            provenance['audit'] = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        for adapter in methods:
            identity = dict(network=name, method=adapter.id, adapter=asdict(adapter),
                            provenance=provenance, parameters=params, environment=environment,
                            code_sha256=code_hash, schema_version=SCHEMA_VERSION,
                            semantics='undirected binary topology; scores not used as weights')
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            target = output / name / adapter.id / key[:16]
            jobs.append(dict(identity, key=key, path=str(path), target=str(target)))
    _write_json(output / 'plan.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                                     'timeout_seconds': timeout, 'jobs': jobs})
    console.print(Panel(f'[bold cyan]{len(jobs)} jobs[/bold cyan] | {len(set(names))} networks | '
                        f'{len(methods)} methods\nResults: {output}\n'
                        'Unweighted topology | missing backends and guarded jobs are reported explicitly',
                        title='Interactome clustering lab'))
    if dry_run:
        console.print('Plan saved; no clustering executed.')
        return
    results = []
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(threads)
    env['PYTHONHASHSEED'] = str(seed)
    with Progress(SpinnerColumn(), TextColumn('{task.description}'), BarColumn(),
                  TextColumn('{task.completed}/{task.total}'), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task('Starting', total=len(jobs))
        for job in jobs:
            target = Path(job['target'])
            progress.update(task, description=f'{job["network"]} / {job["method"]}')
            result = reusable(target, job['key']) if resume else None
            cached = result is not None
            if not cached:
                _write_json(target / 'job.json', job)
                (target / 'result.json').unlink(missing_ok=True)
                started = time.monotonic()
                try:
                    with (target / 'worker.log').open('w', encoding='utf-8') as log:
                        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                                 '_worker', str(target / 'job.json')],
                                                stdout=log, stderr=subprocess.STDOUT, env=env)
                        try:
                            proc.wait(timeout=timeout)
                        except (subprocess.TimeoutExpired, KeyboardInterrupt):
                            proc.kill()
                            proc.wait()
                            raise
                    result = json.loads((target / 'result.json').read_text(encoding='utf-8'))
                    if result.get('key') != job['key'] or proc.returncode not in (0, 1):
                        raise ValueError(f'Worker exited with code {proc.returncode}')
                except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    result = dict(status='timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'failed',
                                  reason=str(exc), key=job['key'], method=job['method'],
                                  elapsed_seconds=round(time.monotonic() - started, 3), job=job)
                    _write_json(target / 'result.json', result)
            row = {k: result.get(k) for k in ('status', 'nodes', 'edges', 'communities',
                    'covered_nodes', 'unassigned_nodes', 'overlapping_nodes', 'elapsed_seconds', 'reason')}
            row.update(network=job['network'], method=job['method'], resumed=cached, target=str(target))
            results.append(row)
            # Checkpoint after every job, including failures/timeouts.
            _write_json(output / 'summary.json', results)
            _write_parquet(output / 'summary.parquet', pl.from_dicts(results, infer_schema_length=None))
            color = {'success': 'green', 'unavailable': 'yellow', 'skipped': 'yellow'}.get(result['status'], 'red')
            progress.console.print(f'[{color}]{result["status"].upper()}[/{color}] '
                                   f'{job["network"]} / {job["method"]}' + (' (resumed)' if cached else ''))
            if result.get('reason'):
                progress.console.print(result['reason'], markup=False)
            progress.advance(task)
    counts = pl.from_dicts(results, infer_schema_length=None).group_by('status').len().sort('status')
    table = Table(title='Run status')
    table.add_column('Status')
    table.add_column('Jobs', justify='right')
    for status, count in counts.iter_rows():
        table.add_row(status, str(count))
    console.print(table)
    console.print(f'Summary: {output / "summary.parquet"}')
    if any(r['status'] in {'failed', 'timeout'} for r in results):
        raise typer.Exit(1)


def inspect_module(name):
    """Keep backend import output out of the machine-readable inventory channel."""
    from contextlib import redirect_stdout
    try:
        with redirect_stdout(sys.stderr):
            module = importlib.import_module(name)
        entries = []
        for member, obj in inspect.getmembers(module):
            if not member.startswith('_') and (inspect.isfunction(obj) or inspect.isclass(obj)):
                try:
                    signature = str(inspect.signature(obj))
                except (ValueError, TypeError):
                    signature = ' (signature unavailable)'
                entries.append({'name': member, 'signature': signature})
        print(json.dumps({'callables': entries}))
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '_worker':
        return _worker(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')))
    if len(sys.argv) == 3 and sys.argv[1] == '_inspect':
        inspect_module(sys.argv[2])
        return 0
    app()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
