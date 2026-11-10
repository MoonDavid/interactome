"""Moonlighting protein cohort preparation, network topology, diffusion, GO enrichment, comparisons, and plots."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import io
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import zipfile

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import polars as pl
from networks import matching_nodes
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist
from scipy.stats import hypergeom, mannwhitneyu, rankdata, spearmanr

CACHE_ROOT = Path(__file__).resolve().parent / 'data' / 'cache'
_CENTRALITY_MEM_CACHE: dict[str, tuple[pl.DataFrame, dict]] = {}
_CLUSTERING_MEM_CACHE: dict[str, dict] = {}
_METHOD_MEM_CACHE: dict[str, dict] = {}

SOURCES = ['curated', 'moonlitdb', 'moonprot', 'metamoon', 'multitask', 'moondb_curated', 'moondb_predicted']
ADVANCED_EDGE_LIMIT = 500_000
DEFAULT_GAF = Path('/home/davide/data/GO/goa_human.gaf')
DEFAULT_OBO = Path('/home/davide/data/GO/go-basic.obo')
GO_POLICY = 'UniProtKB human primary taxon; exclude NOT; collapse numeric isoform suffix; GO ancestor propagation via is_a and part_of; evidence retained'

GROUP_COLORS = {
    'curated': '#0072B2', 'moonlitdb': '#D55E00', 'moonprot': '#009E73',
    'metamoon': '#CC79A7', 'multitask': '#E69F00',
    'moondb_curated': '#56B4E9', 'moondb_predicted': '#8064A2',
    'Curated': '#0072B2', 'MoonLitDB': '#D55E00', 'MoonProt': '#009E73',
    'MetaMoon': '#CC79A7', 'Multitask': '#E69F00',
    'MoonDB curated': '#56B4E9', 'MoonDB predicted': '#8064A2',
    'Union': '#7B3294', 'All sources': '#7B3294', 'All sources (Union)': '#7B3294',
    'Moonlighting': '#0072B2', 'Background': '#737373',
}
GROUP_LABELS = {
    'curated': 'Curated', 'moonlitdb': 'MoonLitDB', 'moonprot': 'MoonProt',
    'metamoon': 'MetaMoon', 'multitask': 'Multitask',
    'moondb_curated': 'MoonDB curated', 'moondb_predicted': 'MoonDB predicted',
    'Curated': 'Curated', 'MoonLitDB': 'MoonLitDB', 'MoonProt': 'MoonProt',
    'MetaMoon': 'MetaMoon', 'Multitask': 'Multitask',
    'MoonDB curated': 'MoonDB curated', 'MoonDB predicted': 'MoonDB predicted',
    'Union': 'Union', 'All sources': 'All sources', 'All sources (Union)': 'All sources (Union)',
    'Moonlighting': 'Moonlighting', 'Background': 'Background',
}
SIGNIFICANCE_LEGEND = (
    'Asterisks compare each set with Background using BH-adjusted q-values: '
    '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; '
    'ns: q ≥ 0.05; n/a: unavailable. Display sampling does not change the tests.'
)
METRIC_LABELS = {
    'degree': 'Degree', 'strength': 'Weighted<br>strength', 'pagerank': 'PageRank',
    'mean_neighbor_degree': 'Mean neighbor<br>degree', 'core_number': 'Core<br>number',
    'clustering': 'Clustering', 'betweenness_approx': 'Betweenness<br>centrality',
    'betweenness': 'Betweenness<br>centrality',
    'eigenvector': 'Eigenvector<br>centrality', 'bridging_coefficient': 'Bridging<br>coefficient',
    'bridging_centrality': 'Bridging<br>centrality',
    'participation': 'Community<br>participation', 'community_entropy': 'Community<br>entropy',
    'within_module_z': 'Within-module<br>z score',
    'num_communities': 'Overlapping<br>modules',
    'fuzzy_entropy': 'Fuzzy<br>entropy',
    'fuzzy_participation': 'Fuzzy<br>participation',
    'effective_clusters': 'Effective<br>clusters',
    'top2_membership_ratio': 'Top-2<br>ambiguity ratio',
}
COLORSCALE = [[0, '#2166AC'], [.5, '#FFFFFF'], [1, '#B35806']]

CLUSTERING_CATALOG: dict[str, dict[str, str]] = {
    # 1. Classic / Hard Partitioning
    'louvain': {
        'label': 'Louvain Modularity (Classic Hard)',
        'paradigm': 'classic',
        'description': 'Multi-level modularity optimization (Blondel et al. 2008).',
    },
    'greedy_modularity': {
        'label': 'Clauset–Newman–Moore Greedy Modularity (Classic Hard)',
        'paradigm': 'classic',
        'description': 'Agglomerative hierarchical modularity maximization (Clauset et al. 2004).',
    },
    'label_propagation': {
        'label': 'Fast Consensus Label Propagation / LPA (Classic Hard)',
        'paradigm': 'classic',
        'description': 'Parameter-free epidemic neighborhood label consensus (Raghavan et al. 2007).',
    },
    'asyn_fluid': {
        'label': 'Asynchronous Fluid Communities (Classic Hard)',
        'paradigm': 'classic',
        'description': 'Density-driven fluid expansion/contraction on connected components (Parés et al. 2018).',
    },
    'mcl': {
        'label': 'Markov Cluster Algorithm / MCL (Flow Hard)',
        'paradigm': 'classic',
        'description': 'Stochastic flow expansion & inflation on the sparse transition matrix (van Dongen 2000).',
    },
    'walktrap_markov': {
        'label': 'Walktrap Multi-Step Markov Flow (Flow Hard)',
        'paradigm': 'classic',
        'description': 'Short random-walk transition profile similarity + Ward/spectral partitioning (Pons & Latapy 2005).',
    },
    'spectral': {
        'label': 'Normalized Laplacian Spectral Clustering (Spectral Hard)',
        'paradigm': 'classic',
        'description': 'Normalized graph Laplacian eigen-embedding + k-means discretization (Shi & Malik 2000).',
    },
    'ravasz_tom': {
        'label': 'Ravasz Topological Overlap / TOM (Hierarchical Hard)',
        'paradigm': 'classic',
        'description': 'Shared-partner topological overlap matrix (Ravasz et al. 2002) + modularity partitioning.',
    },
    'k_core_shells': {
        'label': 'k-Core Corona Connected Shells (Structural Hard)',
        'paradigm': 'classic',
        'description': 'Connected components of maximal k-core shells.',
    },
    # 2. Overlapping / Multi-Membership Clustering
    'ego_splitting': {
        'label': 'Ego-Splitting Persona Clustering (Overlapping)',
        'paradigm': 'overlapping',
        'description': 'Splits each protein into local ego-net personas, clusters the persona graph, and maps back to overlapping node communities (Epasto et al. 2017).',
    },
    'link_communities': {
        'label': 'Ahn–Bagrow–Lehmann Link / Edge Communities (Overlapping)',
        'paradigm': 'overlapping',
        'description': 'Clusters edges by shared-partner Jaccard similarity and projects edge modules onto endpoint proteins (Ahn et al. 2010).',
    },
    'k_clique_percolation': {
        'label': 'Clique Percolation Method / CFinder k=3 (Overlapping)',
        'paradigm': 'overlapping',
        'description': 'Unions of adjacent k-cliques sharing k-1 vertices (Palla et al. 2005).',
    },
    'demon': {
        'label': 'DEMON Local Ego Consensus (Overlapping)',
        'paradigm': 'overlapping',
        'description': 'Democratic Estimate of Modular Organization via ego-minus-ego label propagation (Coscia et al. 2012).',
    },
    'mcode_complexes': {
        'label': 'MCODE Protein Complex Detection + Fluff (Overlapping)',
        'paradigm': 'overlapping',
        'description': 'Local core-density vertex weighting + seed-and-extend with overlapping periphery (Bader & Hogue 2003).',
    },
    'bigclam_nmf': {
        'label': 'BigCLAM Affiliation Graph Model (Overlapping)',
        'paradigm': 'overlapping',
        'description': 'Continuous non-negative community affiliation strengths thresholded into multi-community memberships (Yang & Leskovec 2013).',
    },
    # 3. Fuzzy / Soft-Membership Clustering
    'fuzzy_c_means': {
        'label': 'Spectral Fuzzy C-Means / FCM (Fuzzy Soft)',
        'paradigm': 'fuzzy',
        'description': 'Bezdek Fuzzy C-Means on the normalized Laplacian spectral embedding (fuzzifier m=2.0).',
    },
    'fuzzy_nmf': {
        'label': 'Symmetric Non-Negative Matrix Factorization / SymNMF (Fuzzy Soft)',
        'paradigm': 'fuzzy',
        'description': 'Sparse multiplicative SymNMF (A ≈ H H^T) row-normalized into soft membership probabilities.',
    },
    'fuzzy_rwr_diffusion': {
        'label': 'Fuzzy Random-Walk / Heat Diffusion Affiliation (Fuzzy Soft)',
        'paradigm': 'fuzzy',
        'description': 'Stationary RWR diffusion affinities toward module attractors normalized into soft probability vectors.',
    },
    'fuzzy_neighborhood': {
        'label': 'Fuzzy Neighborhood Consensus (Fuzzy Soft)',
        'paradigm': 'fuzzy',
        'description': 'Dirichlet-smoothed neighbor-module probability distribution measuring partner dispersion across modules.',
    },
}


# ---------------------------------------------------------------------------
# Cohort Data, Sparse Graph Context, Cross-Network & Reproducible Archive
# ---------------------------------------------------------------------------

def load_moon_data(root=Path('data')):
    root = Path(root)
    data = pl.read_csv(root / 'allmoon_with_moonlitdb_human.csv', infer_schema=False).fill_null('')
    data = data.with_columns(
        pl.col(col).str.to_lowercase().is_in(['true', '1', 'yes']).alias(col)
        for col in SOURCES)
    mapping_path = root / 'mappings/moonlighting_uniprot.json'
    mapping = json.loads(mapping_path.read_text())['mapping'] if mapping_path.exists() else {}
    lookup = pl.DataFrame({'_base': list(mapping),
                           '_targets': [targets or [''] for targets in mapping.values()]},
                          schema={'_base': pl.String, '_targets': pl.List(pl.String)})
    return (data.with_columns(pl.col('uniprot').str.strip_chars().alias('input_accession'))
            .with_columns(pl.col('input_accession').str.split('-').list.first().alias('_base'))
            .join(lookup, on='_base', how='left', maintain_order='left')
            .explode('_targets')
            .with_columns(pl.col('_targets').fill_null('').str.split('-').list.first().alias('canonical_id'))
            .with_columns(pl.when(pl.col('canonical_id') != '').then(pl.lit('mapped'))
                          .otherwise(pl.lit('unmapped')).alias('mapping_status'))
            .drop('_base', '_targets'))


def cohort_ids(data, sources=None, require_all=False, reviewed_only=False, min_publications=0):
    selected = data
    if sources:
        condition = pl.all_horizontal(*sources) if require_all else pl.any_horizontal(*sources)
        selected = selected.filter(condition)
    if reviewed_only:
        selected = selected.filter(pl.col('Reviewed').cast(pl.String).str.to_lowercase() == 'reviewed')
    if min_publications:
        selected = selected.filter(pl.col('moonlitdb_number_of_publications').cast(pl.Float64, strict=False).fill_nan(0).fill_null(0) >= min_publications)
    return set(selected['canonical_id']) - {'', None}, selected.clone()


def graph_from_path(path):
    """Load the entire selected canonical graph. Edge scores are affinities."""
    frame = pl.read_parquet(path)
    frame = frame.filter(pl.col('a').is_not_null() & pl.col('b').is_not_null() & (pl.col('a') != pl.col('b')))
    nodes = pl.concat([frame['a'], frame['b'].rename('a')]).unique().sort()
    lookup = pl.DataFrame({'protein': nodes}).with_row_index('i')
    ia = frame.select('a').join(lookup, left_on='a', right_on='protein', how='left', maintain_order='left')['i'].to_numpy()
    ib = frame.select('b').join(lookup, left_on='b', right_on='protein', how='left', maintain_order='left')['i'].to_numpy()
    size = len(nodes)
    rr = np.concatenate([ia, ib])
    cc = np.concatenate([ib, ia])
    adj = coo_matrix((np.ones(2 * len(frame), dtype=np.uint8), (rr, cc)), shape=(size, size)).tocsr()
    adj.data[:] = 1
    if 'probability' in frame.columns:
        scores = frame['probability'].to_numpy()
        if not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError('Invalid edge affinity in canonical graph')
        weights = coo_matrix((np.concatenate([scores, scores]), (rr, cc)), shape=(size, size)).tocsr()
    else:
        weights = adj.astype(np.float64)
    array = nodes.to_numpy()
    return {'nodes': array, 'index': {str(v): i for i, v in enumerate(array)}, 'adj': adj, 'weights': weights,
            'degree': np.diff(adj.indptr), 'edge_count': adj.nnz // 2, 'weighted_available': 'probability' in frame.columns}


def mapping_coverage(graph, data, selected_ids):
    nodes = set(graph['nodes'])
    mapped = matching_nodes(nodes, selected_ids)
    labels = {label for node in nodes for label in str(node).split('|')}
    matched_ids = labels & selected_ids
    audit = data.select('input_accession', 'canonical_id', 'mapping_status', 'Gene Names', 'Reviewed').with_columns(
        pl.col('canonical_id').is_in(sorted(selected_ids)).alias('in_selected_cohort'),
        pl.col('canonical_id').is_in(sorted(labels)).alias('in_network'))
    summary = {'network_nodes': len(nodes), 'network_edges': graph['edge_count'],
               'cohort_canonical_ids': len(selected_ids), 'cohort_in_network': len(matched_ids),
               'cohort_network_nodes': len(mapped),
               'cohort_absent': len(selected_ids - labels),
               'coverage_fraction': len(matched_ids) / len(selected_ids) if selected_ids else 0,
               'comparison_nodes': len(nodes - mapped)}
    return {'summary': summary, 'audit': audit}


def cross_network_analysis(first, second, cohort):
    common = matching_nodes(set(first['nodes']) & set(second['nodes']), cohort)
    records = []
    percentiles = [rankdata(g['degree'], method='average') / max(1, len(g['nodes'])) for g in (first, second)]
    for protein in sorted(common):
        i, j = first['index'][protein], second['index'][protein]
        a = set(first['nodes'][first['adj'].indices[first['adj'].indptr[i]:first['adj'].indptr[i + 1]]])
        b = set(second['nodes'][second['adj'].indices[second['adj'].indptr[j]:second['adj'].indptr[j + 1]]])
        records.append({
            'protein': protein, 'degree_A': int(first['degree'][i]), 'degree_B': int(second['degree'][j]),
            'degree_percentile_A': float(percentiles[0][i]), 'degree_percentile_B': float(percentiles[1][j]),
            'percentile_shift_B_minus_A': float(percentiles[1][j] - percentiles[0][i]),
            'partner_jaccard': len(a & b) / len(a | b) if a | b else 0,
            'partner_turnover': 1 - len(a & b) / len(a | b) if a | b else 0,
        })
    table = pl.DataFrame(records, schema={
        'protein': pl.String, 'degree_A': pl.Int64, 'degree_B': pl.Int64,
        'degree_percentile_A': pl.Float64, 'degree_percentile_B': pl.Float64,
        'percentile_shift_B_minus_A': pl.Float64, 'partner_jaccard': pl.Float64, 'partner_turnover': pl.Float64,
    })
    rho = pvalue = None
    if len(table) > 2 and table['degree_A'].n_unique() > 1 and table['degree_B'].n_unique() > 1:
        rho, pvalue = map(float, spearmanr(table['degree_A'].to_numpy(), table['degree_B'].to_numpy()))
    return {'table': table, 'summary': {
        'shared_cohort_nodes': len(common), 'spearman_degree_rho': rho, 'spearman_p': pvalue,
        'note': 'Partner Jaccard uses full neighborhood sets. Percentile ranks use all nodes in each network. Assay differences affect capture.',
    }}


def result_archive(result, metadata):
    """Export tables plus JSON parameters; no opaque Python pickle required."""
    tables = {}
    def encode(value, prefix='result'):
        if isinstance(value, pl.DataFrame):
            filename = prefix + '.csv'
            tables[filename] = value.write_csv()
            return {'table': filename, 'rows': value.height}
        if isinstance(value, dict):
            return {str(k): encode(v, prefix + '_' + str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [encode(v, prefix) for v in value]
        if isinstance(value, np.generic):
            return encode(value.item(), prefix)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value
    payload = {'metadata': metadata, 'results': encode(result)}
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(encode(payload), indent=2, allow_nan=False))
        for filename, content in tables.items():
            z.writestr(filename, content)
    return output.getvalue()


def save_analysis(result, metadata, root=Path('data')):
    meta = dict(metadata, created_utc=datetime.now(timezone.utc).isoformat())
    content = result_archive(result, meta)
    key = hashlib.sha256(content).hexdigest()[:12]
    folder = Path(root) / 'analysis_runs'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{metadata['analysis']}_{key}.zip"
    path.write_bytes(content)
    return path, content


def _decode_archive(archive_bytes: bytes) -> dict:
    """Reconstruct a result dictionary (with Polars DataFrames) from a result_archive ZIP."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes), 'r') as z:
        manifest = json.loads(z.read('manifest.json'))
        def decode(item):
            if isinstance(item, dict):
                if set(item.keys()) == {'table', 'rows'} and isinstance(item['table'], str) and item['table'].endswith('.csv'):
                    csv_bytes = z.read(item['table'])
                    return pl.read_csv(io.BytesIO(csv_bytes), try_parse_dates=False)
                return {k: decode(v) for k, v in item.items()}
            if isinstance(item, list):
                return [decode(v) for v in item]
            return item
        return {
            'metadata': manifest.get('metadata', {}),
            'result': decode(manifest.get('results', {})),
        }


def cached_analysis_run(name: str, metadata: dict, compute_fn, root: Path = Path('data'),
                        use_cache: bool = True) -> dict:
    """Load an analysis result from memory/disk cache (`data/cache/methods/` or `data/analysis_runs/`) or compute and cache it."""
    sig_payload = json.dumps(metadata, sort_keys=True, default=str)
    signature = hashlib.sha256(sig_payload.encode('utf-8')).hexdigest()
    short_sig = signature[:16]
    cache_key = f'{name}_{short_sig}'

    if use_cache and cache_key in _METHOD_MEM_CACHE:
        cached = _METHOD_MEM_CACHE[cache_key]
        return {**cached, 'cache_hit': True}

    cache_dir = Path(root) / 'cache' / 'methods'
    zip_cache = cache_dir / f'{cache_key}.zip'
    if use_cache and zip_cache.is_file():
        try:
            archive_bytes = zip_cache.read_bytes()
            decoded = _decode_archive(archive_bytes)
            entry = {
                'signature': signature,
                'result': decoded['result'],
                'file': str(zip_cache),
                'archive': archive_bytes,
                'cache_hit': True,
            }
            _METHOD_MEM_CACHE[cache_key] = entry
            return entry
        except Exception:
            pass

    result = compute_fn()
    saved, archive = save_analysis(result, metadata, root)
    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            zip_cache.write_bytes(archive)
        except Exception:
            pass
    entry = {
        'signature': signature,
        'result': result,
        'file': str(saved),
        'archive': archive,
        'cache_hit': False,
    }
    if use_cache:
        _METHOD_MEM_CACHE[cache_key] = entry
    return entry


# ---------------------------------------------------------------------------
# Centralities, Cohesion, Local Ego Wiring & Louvain Cartography
# ---------------------------------------------------------------------------

def bh_adjust(pvalues):
    """Benjamini-Hochberg correction, retaining NaNs for unavailable tests."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        order = valid[np.argsort(p[valid], kind='stable')]
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return out


def _selection(graph, cohort):
    present = matching_nodes(graph['nodes'], cohort)
    query = np.asarray([str(node) in present for node in graph['nodes']], dtype=bool)
    if not query.any():
        raise ValueError('No selected moonlighting proteins occur in this network.')
    return query


def _advanced_graph(graph):
    if graph['adj'].nnz // 2 > ADVANCED_EDGE_LIMIT:
        raise ValueError('This analysis supports up to 500,000 edges. Select a saved huMAP probability-threshold version for advanced topology.')
    return nx.from_scipy_sparse_array(graph['adj'])


def _comparison_table(frame, metrics, group='Moonlighting'):
    rows = []
    mask = frame['is_moonlighting'].to_numpy().astype(bool)
    for metric in metrics:
        values = frame[metric].to_numpy().astype(float)
        a, b = values[mask], values[~mask]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        p = float(mannwhitneyu(a, b, alternative='two-sided').pvalue) if len(a) and len(b) else np.nan
        rows.append({'group': group, 'metric': metric, 'query_n': len(a), 'background_n': len(b),
                     'query_median': float(np.median(a)) if len(a) else np.nan,
                     'background_median': float(np.median(b)) if len(b) else np.nan,
                     'p_value': p})
    result = pl.DataFrame(rows)
    return result.with_columns(pl.Series('fdr_bh', bh_adjust(result['p_value'])))


def _pagerank(weights, personalization=None, alpha=.85, tolerance=1e-10, max_iterations=1000):
    n = weights.shape[0]
    if not n:
        return np.array([]), {'converged': True, 'iterations': 0, 'l1_residual': 0.0}
    restart = np.full(n, 1 / n) if personalization is None else np.asarray(personalization, dtype=float)
    restart /= restart.sum()
    strengths = np.asarray(weights.sum(axis=1)).ravel()
    inverse = np.divide(1., strengths, out=np.zeros(n), where=strengths > 0)
    score = restart.copy()
    for iteration in range(1, max_iterations + 1):
        following = alpha * (weights.T @ (inverse * score) + score[strengths == 0].sum() * restart) + (1 - alpha) * restart
        residual = float(np.abs(following - score).sum())
        score = following
        if residual < tolerance:
            break
    return score, {'converged': residual < tolerance, 'iterations': iteration, 'l1_residual': residual}


def _graph_fingerprint(graph) -> str:
    """Compute a fast deterministic SHA-256 fingerprint for a graph structure."""
    h = hashlib.sha256()
    nodes = [str(n) for n in graph['nodes']]
    h.update(','.join(nodes).encode('utf-8'))
    adj = graph['adj']
    h.update(np.asarray(adj.indptr, dtype=np.int64).tobytes())
    h.update(np.asarray(adj.indices, dtype=np.int64).tobytes())
    weights = graph.get('weights')
    if weights is not None and weights is not adj:
        h.update(np.asarray(weights.data, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _brandes_chunk_worker(args):
    """Exact Brandes shortest-path accumulation for a chunk of source nodes on CSR lists."""
    n, indptr, indices, sources = args
    cb = [0.0] * n
    for s in sources:
        if indptr[s] == indptr[s + 1]:
            continue
        stack = []
        pred = [[] for _ in range(n)]
        sigma = [0.0] * n
        sigma[s] = 1.0
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        q_idx = 0
        while q_idx < len(queue):
            v = queue[q_idx]
            q_idx += 1
            stack.append(v)
            dv = dist[v] + 1
            sv = sigma[v]
            for pos in range(indptr[v], indptr[v + 1]):
                w = indices[pos]
                dw = dist[w]
                if dw < 0:
                    queue.append(w)
                    dist[w] = dv
                    sigma[w] = sv
                    pred[w].append(v)
                elif dw == dv:
                    sigma[w] += sv
                    pred[w].append(v)
        delta = [0.0] * n
        while stack:
            w = stack.pop()
            coeff = (1.0 + delta[w]) / sigma[w]
            for v in pred[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                cb[w] += delta[w]
    return np.asarray(cb, dtype=np.float64)


def _exact_betweenness_csr(adj: sparse.csr_matrix, normalized: bool = True) -> np.ndarray:
    """Compute exact Brandes betweenness centrality across all nodes (parallelized for large graphs)."""
    n = adj.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=np.float64)
    if n <= 450:
        net = nx.from_scipy_sparse_array(adj)
        bc = nx.betweenness_centrality(net, k=None, normalized=normalized)
        return np.asarray([bc[i] for i in range(n)], dtype=np.float64)

    indptr = adj.indptr.tolist()
    indices = adj.indices.tolist()
    non_isolated = [u for u in range(n) if indptr[u] < indptr[u + 1]]
    workers = min(max(1, (os.cpu_count() or 4) - 1), 12, max(1, len(non_isolated) // 100))
    if workers <= 1:
        total = _brandes_chunk_worker((n, indptr, indices, non_isolated))
    else:
        chunks = [non_isolated[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_brandes_chunk_worker, [(n, indptr, indices, chunk) for chunk in chunks if chunk]))
        total = np.sum(parts, axis=0) if parts else np.zeros(n, dtype=np.float64)

    total *= 0.5  # undirected graph
    if normalized and n > 2:
        total *= 2.0 / ((n - 1) * (n - 2))
    return total


def cached_network_centralities(graph, extra=False, seed=42, landmarks=None,
                                use_cache=True, cache_dir: Path | None = None) -> tuple[pl.DataFrame, dict]:
    """Compute graph-level node centralities (including exact betweenness) once and cache to disk."""
    degree = np.asarray(graph['degree'], dtype=float)
    n = len(degree)
    adjacency = graph['adj']
    weights = graph.get('weights', adjacency)
    exact_mode = landmarks is None or int(landmarks) >= n
    fp = _graph_fingerprint(graph)
    mode_tag = 'exact' if exact_mode else f'k{int(landmarks)}_s{int(seed)}'
    cache_key = f'{fp}_extra{int(bool(extra))}_{mode_tag}'

    if use_cache and cache_key in _CENTRALITY_MEM_CACHE:
        df_cached, notes_cached = _CENTRALITY_MEM_CACHE[cache_key]
        return df_cached.clone(), dict(notes_cached, cache_hit=True)

    cdir = (Path(cache_dir) if cache_dir is not None else CACHE_ROOT / 'centralities')
    parquet_path = cdir / f'centralities_{cache_key}.parquet'
    meta_path = cdir / f'centralities_{cache_key}.json'

    if use_cache and n >= 50 and parquet_path.exists() and meta_path.exists():
        try:
            df_disk = pl.read_parquet(parquet_path)
            notes_disk = json.loads(meta_path.read_text(encoding='utf-8'))
            if df_disk.height == n and df_disk['protein'].to_list() == [str(x) for x in graph['nodes']]:
                notes_disk['cache_hit'] = True
                notes_disk['cache_path'] = str(parquet_path)
                _CENTRALITY_MEM_CACHE[cache_key] = (df_disk, notes_disk)
                return df_disk.clone(), dict(notes_disk)
        except Exception:
            pass

    pagerank, convergence = _pagerank(weights)
    neighbor_degree = np.divide(np.asarray(adjacency @ degree).ravel(), degree,
                                out=np.zeros(n), where=degree > 0)
    frame = pl.DataFrame({
        'protein': [str(x) for x in graph['nodes']],
        'degree': degree,
        'strength': np.asarray(weights.sum(axis=1)).ravel(),
        'pagerank': pagerank,
        'mean_neighbor_degree': neighbor_degree,
    })
    notes = {
        'pagerank': convergence,
        'comparison': 'Two-sided Mann–Whitney U; Benjamini–Hochberg across displayed metrics; background excludes cohort proteins.',
        'cache_hit': False,
    }
    if extra:
        network = _advanced_graph(graph)
        core = nx.core_number(network)
        clustering = nx.clustering(network)
        if exact_mode:
            betweenness_arr = _exact_betweenness_csr(adjacency, normalized=True)
            notes['betweenness_mode'] = 'exact (all nodes, cached)'
            notes['betweenness_landmarks'] = n
        else:
            sample = min(int(landmarks), n)
            betweenness_dict = nx.betweenness_centrality(network, k=sample, seed=int(seed), normalized=True)
            betweenness_arr = np.asarray([betweenness_dict[i] for i in range(n)], dtype=np.float64)
            notes['betweenness_mode'] = f'landmark approximation (k={sample})'
            notes['betweenness_landmarks'] = sample
            notes['betweenness_seed'] = int(seed)

        frame = frame.with_columns(
            pl.Series('core_number', [core[i] for i in range(n)], dtype=pl.Int64),
            pl.Series('clustering', [clustering[i] for i in range(n)], dtype=pl.Float64),
            pl.Series('betweenness_approx', betweenness_arr, dtype=pl.Float64),
        )
        if n > 2:
            _, eigenvectors = eigsh(adjacency.astype(float), k=1, which='LA', tol=1e-8, v0=np.random.default_rng(seed).random(n))
            eigenvector = np.abs(eigenvectors[:, 0])
        else:
            _, eigenvectors = np.linalg.eigh(adjacency.toarray().astype(float))
            eigenvector = np.abs(eigenvectors[:, -1])
        frame = frame.with_columns(pl.Series('eigenvector', eigenvector, dtype=pl.Float64))
        inverse_degree = np.divide(1., degree, out=np.zeros(n), where=degree > 0)
        neighbor_inverse = np.asarray(adjacency @ inverse_degree).ravel()
        frame = frame.with_columns(pl.Series('bridging_coefficient', np.divide(inverse_degree, neighbor_inverse, out=np.zeros(n), where=neighbor_inverse > 0), dtype=pl.Float64))
        frame = frame.with_columns((pl.col('betweenness_approx') * pl.col('bridging_coefficient')).alias('bridging_centrality'))
        notes['eigenvector'] = 'Unweighted leading adjacency eigenvector, L2-normalized; disconnected components with tied largest eigenvalues need not have a unique eigenvector.'

    if use_cache:
        _CENTRALITY_MEM_CACHE[cache_key] = (frame, dict(notes))
        if n >= 50:
            try:
                cdir.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(parquet_path)
                notes['cache_path'] = str(parquet_path)
                meta_path.write_text(json.dumps(notes, indent=2), encoding='utf-8')
            except Exception:
                pass
    return frame.clone(), notes


def centrality_analysis(graph, cohort, extra=False, seed=42, landmarks=None, use_cache=True, set_name='Moonlighting'):
    """Grouped centralities (with exact cached betweenness by default); compares cohort with its network complement."""
    mask = _selection(graph, cohort)
    base_frame, notes = cached_network_centralities(
        graph, extra=extra, seed=seed, landmarks=landmarks, use_cache=use_cache
    )
    frame = base_frame.with_columns(
        pl.Series('is_moonlighting', mask),
        pl.Series('group', np.where(mask, set_name, 'Background')),
    )
    ordered_cols = ['protein', 'is_moonlighting', 'group', 'degree', 'strength', 'pagerank', 'mean_neighbor_degree']
    names = ['degree', 'strength', 'pagerank', 'mean_neighbor_degree']
    if extra:
        extra_cols = ['core_number', 'clustering', 'betweenness_approx', 'eigenvector', 'bridging_coefficient', 'bridging_centrality']
        ordered_cols += extra_cols
        names += extra_cols
    frame = frame.select(ordered_cols)
    return {'metrics': frame, 'tests': _comparison_table(frame, names, group=set_name), 'notes': notes}


def _degree_bins(degree):
    return np.floor(np.log2(np.maximum(1, np.asarray(degree)))).astype(int)


def _sample_matching(degree, query, rng, candidates=None):
    bins = _degree_bins(degree)
    selected = []
    candidate_mask = np.ones(len(degree), dtype=bool) if candidates is None else np.asarray(candidates, dtype=bool)
    for bucket, count in Counter(bins[query]).items():
        pool = np.flatnonzero((bins == bucket) & candidate_mask)
        if len(pool) < count:
            raise ValueError(f'Insufficient background proteins in degree bin {bucket}: need {count}, have {len(pool)}. Reduce the sample size.')
        selected.extend(rng.choice(pool, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def _induced_stats(adjacency, indices):
    induced = adjacency[indices][:, indices]
    edges = int(induced.nnz // 2)
    if len(indices):
        components, labels = connected_components(induced, directed=False)
        lcc = int(np.bincount(labels).max())
    else:
        components = lcc = 0
    return edges, lcc, int(components)


def cohesion_analysis(graph, cohort, permutations=100, seed=42, degree_matched=True):
    """Induced cohesion versus whole-network random protein sets."""
    mask = _selection(graph, cohort)
    if not 1 <= int(permutations) <= 10000:
        raise ValueError('Use between 1 and 10,000 permutations.')
    adjacency, degree = graph['adj'], np.asarray(graph['degree'])
    indices = np.flatnonzero(mask)
    edges, lcc, components = _induced_stats(adjacency, indices)
    volume = float(degree[indices].sum())
    boundary = int(volume - 2 * edges)
    other_volume = float(degree[~mask].sum())
    denominator = min(volume, other_volume)
    conductance = boundary / denominator if denominator > 0 else np.nan
    edge_count = adjacency.nnz // 2
    endpoint_fraction = volume / (volume + other_volume) if edge_count else np.nan
    expected_same_status = endpoint_fraction ** 2 + (1 - endpoint_fraction) ** 2
    observed_same_status = 1 - boundary / edge_count if edge_count else np.nan
    assortativity = ((observed_same_status - expected_same_status) / (1 - expected_same_status)
                     if 0 < endpoint_fraction < 1 else np.nan)
    rng = np.random.default_rng(seed)
    null = []
    for run in range(int(permutations)):
        sampled = _sample_matching(degree, indices, rng) if degree_matched else rng.choice(len(degree), size=len(indices), replace=False)
        null_edges, null_lcc, _ = _induced_stats(adjacency, sampled)
        null.append({'permutation': run + 1, 'internal_edges': null_edges, 'largest_component': null_lcc})
    null = pl.DataFrame(null)
    summary = {
        'query_nodes': len(indices), 'internal_edges': edges,
        'density': 2 * edges / (len(indices) * (len(indices) - 1)) if len(indices) > 1 else 0.,
        'components': components, 'largest_component': lcc, 'boundary_edges': boundary,
        'conductance': conductance, 'boundary_edges_per_cohort_protein': boundary / len(indices),
        'cohort_status_assortativity': assortativity,
        'permutations': int(permutations), 'seed': int(seed),
        'null_model': 'Whole-network sampling matched in floor(log2 degree) bins' if degree_matched else 'Uniform whole-network protein sampling',
        'internal_edges_null_mean': float(null['internal_edges'].mean()),
        'largest_component_null_mean': float(null['largest_component'].mean()),
        'internal_edges_p_value': (1 + int((null['internal_edges'] >= edges).sum())) / (int(permutations) + 1),
        'largest_component_p_value': (1 + int((null['largest_component'] >= lcc).sum())) / (int(permutations) + 1),
    }
    adjusted = bh_adjust([summary['internal_edges_p_value'], summary['largest_component_p_value']])
    summary.update(internal_edges_fdr_bh=float(adjusted[0]), largest_component_fdr_bh=float(adjusted[1]))
    return {'summary': summary, 'null': null}


def wiring_analysis(graph, cohort, max_nodes=100, seed=42, set_name='Moonlighting'):
    """Exact ego statistics for a cohort sample and degree-binned controls."""
    _advanced_graph(graph)
    mask = _selection(graph, cohort)
    rng = np.random.default_rng(seed)
    query = np.flatnonzero(mask)
    if len(query) > int(max_nodes):
        query = np.sort(rng.choice(query, size=int(max_nodes), replace=False))
    if int(max_nodes) < 1:
        raise ValueError('max_nodes must be positive.')
    controls = _sample_matching(np.asarray(graph['degree']), query, rng, candidates=~mask)
    rows = []
    for node in np.concatenate([query, controls]):
        neighbors = graph['adj'].indices[graph['adj'].indptr[node]:graph['adj'].indptr[node + 1]]
        k = len(neighbors)
        ego = graph['adj'][neighbors][:, neighbors]
        triangles = ego.nnz // 2
        components, labels = connected_components(ego, directed=False) if k else (0, np.array([], dtype=int))
        sizes = np.bincount(labels) if k else np.array([])
        entropy = float(-(sizes / k * np.log(sizes / k)).sum()) if k else 0.
        wedges = k * (k - 1) // 2
        rows.append({
            'protein': str(graph['nodes'][node]), 'is_moonlighting': bool(mask[node]),
            'group': set_name if mask[node] else 'Background', 'degree': k,
            'triangles': int(triangles), 'clustering': triangles / wedges if wedges else 0.,
            'open_wedges': int(wedges - triangles), 'ego_components': int(components),
            'ego_component_entropy': entropy,
            'effective_size': k - 2 * triangles / k if k else 0.,
        })
        
    frame = pl.DataFrame(rows)
    columns = ['triangles', 'clustering', 'open_wedges', 'ego_components', 'ego_component_entropy', 'effective_size']
    return {'metrics': frame, 'tests': _comparison_table(frame, columns, group=set_name),
            'notes': {'query_sample': len(query), 'control_sample': len(controls), 'seed': int(seed),
                      'controls': 'Network complement, sampled without replacement in the same floor(log2 degree) bins as the sampled cohort.'}}


def _spectral_embedding(adjacency: sparse.csr_matrix, k: int, seed: int = 42) -> np.ndarray:
    """Compute top-k normalized Laplacian spectral coordinates for a sparse graph."""
    n = adjacency.shape[0]
    k = max(2, min(int(k), max(2, n - 1)))
    deg = np.asarray(adjacency.sum(axis=1)).ravel().astype(float)
    inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros(n), where=deg > 0)
    norm_adj = sparse.diags(inv_sqrt) @ adjacency.astype(float) @ sparse.diags(inv_sqrt)
    if n > k + 2:
        try:
            rng = np.random.default_rng(seed)
            _, vecs = eigsh(norm_adj, k=k, which='LA', tol=1e-6, maxiter=1500, v0=rng.random(n))
            emb = vecs[:, ::-1]
        except Exception:
            _, vecs = np.linalg.eigh(norm_adj.toarray())
            emb = vecs[:, -k:][:, ::-1]
    else:
        _, vecs = np.linalg.eigh(norm_adj.toarray())
        emb = vecs[:, -k:][:, ::-1]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 1e-12)


def _kmeans_numpy(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ in NumPy returning (labels, centroids)."""
    n, d = X.shape
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    centers = np.empty((k, d), dtype=float)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(dist_sq.sum())
        if total > 1e-12:
            probs = dist_sq / total
            idx = int(rng.choice(n, p=probs))
        else:
            idx = int(rng.integers(0, n))
        centers[c] = X[idx]
        dist_sq = np.minimum(dist_sq, np.sum((X - centers[c]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return labels, centers


def _execute_clustering_algorithm(
    graph,
    network: nx.Graph,
    algorithm: str = 'louvain',
    resolution: float = 1.0,
    seed: int = 42,
    n_clusters: int = 0,
    mcl_inflation: float = 2.0,
    clique_k: int = 3,
    fuzzy_m: float = 2.0,
    membership_cutoff: float = 0.2,
) -> tuple[list[set[int]], np.ndarray | None, dict]:
    """Execute any classic, flow, overlapping, or fuzzy clustering algorithm on graph."""
    algo = str(algorithm).lower().strip()
    if algo not in CLUSTERING_CATALOG:
        raise ValueError(f"Unknown clustering algorithm '{algorithm}'. Choose from: {', '.join(CLUSTERING_CATALOG)}")

    adjacency: sparse.csr_matrix = graph['adj']
    n = adjacency.shape[0]
    indptr = adjacency.indptr
    indices = adjacency.indices
    deg = np.asarray(graph['degree'], dtype=float)
    target_k = int(n_clusters) if int(n_clusters) >= 2 else max(2, min(32, int(round(math.sqrt(max(4, n)) / 1.5))))
    extra_meta: dict = {'algorithm': algo, 'paradigm': CLUSTERING_CATALOG[algo]['paradigm']}

    # ---- 1. CLASSIC / HARD PARTITIONING ----
    if algo == 'louvain':
        comms = [set(c) for c in nx.community.louvain_communities(network, seed=int(seed), resolution=float(resolution), weight=None)]
        return comms, None, extra_meta

    if algo == 'greedy_modularity':
        comms = [set(c) for c in nx.community.greedy_modularity_communities(network, weight=None, resolution=float(resolution))]
        return comms, None, extra_meta

    if algo == 'label_propagation':
        comms = [set(c) for c in nx.community.fast_label_propagation_communities(network, seed=int(seed))]
        return comms, None, extra_meta

    if algo == 'asyn_fluid':
        comms = []
        rng_k = max(2, min(target_k, max(2, n // 3)))
        for comp in nx.connected_components(network):
            sub = network.subgraph(comp)
            if len(comp) <= 3:
                comms.append(set(comp))
            else:
                k_sub = max(2, min(rng_k, len(comp) // 2))
                try:
                    for c in nx.community.asyn_fluidc(sub, k=k_sub, max_iter=100, seed=int(seed)):
                        comms.append(set(c))
                except Exception:
                    comms.append(set(comp))
        return comms, None, extra_meta

    if algo == 'mcl':
        inflation = max(1.2, float(mcl_inflation))
        M = (adjacency + sparse.eye(n, format='csr')).astype(float)
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        M = M @ sparse.diags(np.divide(1.0, col_sums, out=np.zeros(n), where=col_sums > 0))
        for _ in range(18):
            M = (M @ M).tocsr()
            M.data = np.power(M.data, inflation)
            col_sums = np.asarray(M.sum(axis=0)).ravel()
            M = M @ sparse.diags(np.divide(1.0, col_sums, out=np.zeros(n), where=col_sums > 0))
            M.data[M.data < 1e-4] = 0.0
            M.eliminate_zeros()
            if M.nnz <= n * 2:
                break
        sym = ((M + M.T) > 1e-3).astype(int)
        n_comp, labels = connected_components(sym, directed=False)
        buckets: dict[int, set[int]] = {}
        for u, c in enumerate(labels):
            buckets.setdefault(int(c), set()).add(u)
        extra_meta['mcl_inflation'] = inflation
        return list(buckets.values()), None, extra_meta

    if algo == 'walktrap_markov':
        inv_deg = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
        P = sparse.diags(inv_deg) @ adjacency.astype(float)
        emb0 = _spectral_embedding(adjacency, k=min(target_k, 16), seed=int(seed))
        walk_feat = (emb0 + P @ emb0 + (P @ (P @ emb0))) / 3.0
        labels, _ = _kmeans_numpy(walk_feat, k=target_k, seed=int(seed))
        buckets = {}
        for u, c in enumerate(labels):
            buckets.setdefault(int(c), set()).add(u)
        return list(buckets.values()), None, extra_meta

    if algo == 'spectral':
        emb = _spectral_embedding(adjacency, k=min(target_k, 20), seed=int(seed))
        labels, _ = _kmeans_numpy(emb, k=target_k, seed=int(seed))
        buckets = {}
        for u, c in enumerate(labels):
            buckets.setdefault(int(c), set()).add(u)
        return list(buckets.values()), None, extra_meta

    if algo == 'ravasz_tom':
        shared = (adjacency @ adjacency).tocoo()
        rows, cols, vals = shared.row, shared.col, shared.data.astype(float)
        mask_edge = rows < cols
        rows, cols, vals = rows[mask_edge], cols[mask_edge], vals[mask_edge]
        a_ij = np.asarray(adjacency[rows, cols]).ravel().astype(float)
        denom = np.minimum(deg[rows], deg[cols]) + 1.0 - a_ij
        tom_val = np.divide(vals + a_ij, denom, out=np.zeros_like(vals), where=denom > 0)
        keep = (a_ij > 0) | (tom_val >= 0.15)
        tom_graph = nx.Graph()
        tom_graph.add_nodes_from(range(n))
        for u, v, w in zip(rows[keep].tolist(), cols[keep].tolist(), tom_val[keep].tolist()):
            tom_graph.add_edge(u, v, weight=float(w))
        comms = [set(c) for c in nx.community.louvain_communities(tom_graph, weight='weight', resolution=float(resolution), seed=int(seed))]
        return comms, None, extra_meta

    if algo == 'k_core_shells':
        cores = nx.core_number(network)
        buckets = {}
        for k_val in sorted(set(cores.values())):
            shell_nodes = [u for u, c in cores.items() if c == k_val]
            for comp in nx.connected_components(network.subgraph(shell_nodes)):
                buckets[len(buckets)] = set(comp)
        return list(buckets.values()), None, extra_meta

    # ---- 2. OVERLAPPING / MULTI-MEMBERSHIP CLUSTERING ----
    if algo == 'ego_splitting':
        persona_graph = nx.Graph()
        node_personas: dict[int, list[int]] = {}
        neighbor_persona_map: dict[tuple[int, int], int] = {}
        p_id = 0
        for u in range(n):
            nbrs = indices[indptr[u]:indptr[u + 1]]
            if len(nbrs) == 0:
                persona_graph.add_node(p_id)
                node_personas[u] = [p_id]
                p_id += 1
                continue
            ego_sub = adjacency[nbrs][:, nbrs]
            n_c, comp_labels = connected_components(ego_sub, directed=False)
            u_pids = []
            for c_idx in range(n_c):
                persona_graph.add_node(p_id)
                u_pids.append(p_id)
                for local_idx in np.flatnonzero(comp_labels == c_idx):
                    v = int(nbrs[local_idx])
                    neighbor_persona_map[(u, v)] = p_id
                p_id += 1
            node_personas[u] = u_pids

        for u, v in network.edges():
            pu = neighbor_persona_map.get((u, v))
            pv = neighbor_persona_map.get((v, u))
            if pu is not None and pv is not None:
                persona_graph.add_edge(pu, pv)

        p_comms = nx.community.louvain_communities(persona_graph, seed=int(seed), resolution=float(resolution), weight=None)
        p_to_comm = {}
        for c_idx, members in enumerate(p_comms):
            for p_node in members:
                p_to_comm[p_node] = c_idx

        comm_sets: dict[int, set[int]] = {}
        for u, pids in node_personas.items():
            for pid in pids:
                c_idx = p_to_comm[pid]
                comm_sets.setdefault(c_idx, set()).add(u)
        comms = [members for members in comm_sets.values() if members]
        return comms, None, extra_meta

    if algo == 'link_communities':
        edges = list(network.edges())
        if not edges:
            return [{u} for u in range(n)], None, extra_meta
        # Partition edges via line-graph / incident-partner Louvain
        if len(edges) <= 25_000:
            lg = nx.line_graph(network)
            edge_comms = nx.community.louvain_communities(lg, seed=int(seed), resolution=float(resolution))
            comms = []
            covered = set()
            for e_set in edge_comms:
                if len(e_set) >= 2:
                    nodes_in_c = {u for e in e_set for u in e}
                    comms.append(nodes_in_c)
                    covered.update(nodes_in_c)
            for u in range(n):
                if u not in covered:
                    comms.append({u})
            return comms, None, extra_meta
        else:
            # Scalable high-edge link clustering via base Louvain + cross-module bridge edge communities
            base_comms = [set(c) for c in nx.community.louvain_communities(network, seed=int(seed), resolution=float(resolution))]
            node_c = np.zeros(n, dtype=int)
            for idx_c, c_set in enumerate(base_comms):
                for u in c_set:
                    node_c[u] = idx_c
            pair_buckets: dict[tuple[int, int], set[int]] = {}
            for u, v in edges:
                cu, cv = int(node_c[u]), int(node_c[v])
                if cu != cv:
                    key = (min(cu, cv), max(cu, cv))
                    pair_buckets.setdefault(key, set()).update((u, v))
            for bridge_nodes in pair_buckets.values():
                if len(bridge_nodes) >= 3:
                    base_comms.append(bridge_nodes)
            return base_comms, None, extra_meta

    if algo == 'k_clique_percolation':
        k_val = max(3, int(clique_k))
        cpm = [set(c) for c in nx.community.k_clique_communities(network, k=k_val)]
        covered = set().union(*cpm) if cpm else set()
        # Fallback connected modules for non-clique nodes so every node has at least 1 community
        rem = [u for u in range(n) if u not in covered]
        if rem:
            sub = network.subgraph(rem)
            for comp in nx.connected_components(sub):
                cpm.append(set(comp))
        extra_meta['clique_k'] = k_val
        return cpm, None, extra_meta

    if algo == 'demon':
        raw_modules: list[set[int]] = []
        for u in range(n):
            nbrs = indices[indptr[u]:indptr[u + 1]]
            if len(nbrs) < 2:
                continue
            ego_sub = network.subgraph(nbrs.tolist())
            for lpa_c in nx.community.fast_label_propagation_communities(ego_sub, seed=int(seed)):
                if len(lpa_c) >= 2:
                    mod = set(lpa_c)
                    mod.add(u)
                    raw_modules.append(mod)
        # Deduplicate and merge modules with Jaccard >= 0.6
        raw_modules.sort(key=len, reverse=True)
        merged: list[set[int]] = []
        for mod in raw_modules[:min(len(raw_modules), 1500)]:
            placed = False
            for existing in merged:
                inter = len(mod & existing)
                if inter / max(1, min(len(mod), len(existing))) >= 0.65:
                    existing.update(mod)
                    placed = True
                    break
            if not placed:
                merged.append(set(mod))
        covered = set().union(*merged) if merged else set()
        for u in range(n):
            if u not in covered:
                merged.append({u})
        return merged, None, extra_meta

    if algo == 'mcode_complexes':
        cores = nx.core_number(network)
        weights = {}
        for u in range(n):
            nbrs = indices[indptr[u]:indptr[u + 1]].tolist()
            if len(nbrs) < 2:
                weights[u] = 0.0
            else:
                ego = adjacency[nbrs][:, nbrs]
                e_cnt = ego.nnz // 2 + len(nbrs)
                n_cnt = len(nbrs) + 1
                density = (2.0 * e_cnt) / (n_cnt * (n_cnt - 1))
                weights[u] = float(cores[u]) * density
        order = sorted(range(n), key=lambda x: weights[x], reverse=True)
        complexes: list[set[int]] = []
        seeded: set[int] = set()
        for seed_u in order:
            if weights[seed_u] <= 0 or seed_u in seeded:
                continue
            comp = {seed_u}
            w_seed = weights[seed_u]
            frontier = set(indices[indptr[seed_u]:indptr[seed_u + 1]].tolist())
            while frontier:
                v = frontier.pop()
                if v not in comp and weights[v] >= 0.2 * w_seed:
                    comp.add(v)
                    frontier.update(indices[indptr[v]:indptr[v + 1]].tolist())
            # Fluff expansion: allow overlapping neighbors with local density >= 0.5
            fluff = set()
            for u in list(comp):
                for v in indices[indptr[u]:indptr[u + 1]].tolist():
                    if v not in comp:
                        v_nbrs = set(indices[indptr[v]:indptr[v + 1]].tolist())
                        if len(v_nbrs & comp) >= max(2, int(0.5 * len(v_nbrs))):
                            fluff.add(v)
            seeded.update(comp)
            complexes.append(comp | fluff)
        covered = set().union(*complexes) if complexes else set()
        for u in range(n):
            if u not in covered:
                complexes.append({u})
        return complexes, None, extra_meta

    if algo == 'bigclam_nmf':
        emb = np.maximum(0.0, _spectral_embedding(adjacency, k=min(target_k, 16), seed=int(seed)))
        F = emb + 0.05
        for _ in range(15):
            AF = adjacency @ F
            denom = F @ (F.T @ F) + 1e-9
            F = np.maximum(0.0, F * (AF / denom))
        row_max = np.maximum(F.max(axis=1, keepdims=True), 1e-9)
        F_norm = F / row_max
        row_sum = np.maximum(F.sum(axis=1, keepdims=True), 1e-9)
        soft_u = F / row_sum
        comms_dict: dict[int, set[int]] = {}
        for u in range(n):
            active = np.flatnonzero(F_norm[u] >= 0.35).tolist()
            if not active:
                active = [int(np.argmax(F[u]))]
            for c_idx in active:
                comms_dict.setdefault(int(c_idx), set()).add(u)
        return list(comms_dict.values()), soft_u, extra_meta

    # ---- 3. FUZZY / SOFT-MEMBERSHIP CLUSTERING ----
    if algo == 'fuzzy_c_means':
        emb = _spectral_embedding(adjacency, k=min(target_k, 16), seed=int(seed))
        _, centers = _kmeans_numpy(emb, k=target_k, seed=int(seed), max_iter=35)
        m_val = max(1.15, float(fuzzy_m))
        power = 2.0 / (m_val - 1.0)
        for _ in range(25):
            dists = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
            dists = np.maximum(dists, 1e-8)
            inv_d = np.power(1.0 / dists, power)
            U = inv_d / np.maximum(inv_d.sum(axis=1, keepdims=True), 1e-12)
            um = np.power(U, m_val)
            centers = (um.T @ emb) / np.maximum(um.sum(axis=0)[:, None], 1e-12)
        comms_dict = {}
        cutoff = max(1.0 / max(2, target_k) * 1.35, float(membership_cutoff))
        for u in range(n):
            active = np.flatnonzero(U[u] >= cutoff).tolist()
            if not active:
                active = [int(np.argmax(U[u]))]
            for c_idx in active:
                comms_dict.setdefault(int(c_idx), set()).add(u)
        extra_meta['fuzzy_m'] = m_val
        return list(comms_dict.values()), U, extra_meta

    if algo == 'fuzzy_nmf':
        rng = np.random.default_rng(seed)
        H = np.abs(_spectral_embedding(adjacency, k=min(target_k, 16), seed=int(seed))) + 0.02 * rng.random((n, min(target_k, 16)))
        for _ in range(20):
            AH = adjacency @ H
            HHtH = H @ (H.T @ H) + 1e-9
            H = H * np.sqrt(np.maximum(0.0, AH / HHtH))
        row_sum = np.maximum(H.sum(axis=1, keepdims=True), 1e-12)
        U = H / row_sum
        comms_dict = {}
        cutoff = max(1.0 / max(2, U.shape[1]) * 1.35, float(membership_cutoff))
        for u in range(n):
            active = np.flatnonzero(U[u] >= cutoff).tolist()
            if not active:
                active = [int(np.argmax(U[u]))]
            for c_idx in active:
                comms_dict.setdefault(int(c_idx), set()).add(u)
        return list(comms_dict.values()), U, extra_meta

    if algo == 'fuzzy_rwr_diffusion':
        base_comms = [set(c) for c in nx.community.louvain_communities(network, seed=int(seed), resolution=float(resolution))]
        # Use top communities (up to 24) as diffusion attractors
        sorted_comms = sorted(base_comms, key=len, reverse=True)[:min(len(base_comms), 24)]
        k_c = len(sorted_comms)
        inv_deg = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
        P_t = adjacency.T.multiply(inv_deg[None, :]).tocsr()
        U = np.zeros((n, k_c), dtype=float)
        restart = 0.25
        for idx_c, members in enumerate(sorted_comms):
            tele = np.zeros(n, dtype=float)
            tele[list(members)] = 1.0 / max(1, len(members))
            vec = tele.copy()
            for _ in range(20):
                vec = restart * tele + (1.0 - restart) * (P_t @ vec)
            U[:, idx_c] = vec
        row_sum = np.maximum(U.sum(axis=1, keepdims=True), 1e-12)
        U = U / row_sum
        comms_dict = {}
        cutoff = max(1.0 / max(2, k_c) * 1.5, float(membership_cutoff))
        for u in range(n):
            active = np.flatnonzero(U[u] >= cutoff).tolist()
            if not active:
                active = [int(np.argmax(U[u]))]
            for c_idx in active:
                comms_dict.setdefault(int(c_idx), set()).add(u)
        return list(comms_dict.values()), U, extra_meta

    if algo == 'fuzzy_neighborhood':
        base_comms = [set(c) for c in nx.community.louvain_communities(network, seed=int(seed), resolution=float(resolution))]
        k_c = len(base_comms)
        primary = np.zeros(n, dtype=int)
        for idx_c, members in enumerate(base_comms):
            for u in members:
                primary[u] = idx_c
        U = np.zeros((n, max(1, k_c)), dtype=float)
        for u in range(n):
            nbrs = indices[indptr[u]:indptr[u + 1]]
            counts = np.bincount(primary[nbrs], minlength=k_c).astype(float)
            counts[primary[u]] += 1.0  # self-prior
            U[u] = counts / max(1.0, counts.sum())
        comms_dict = {}
        cutoff = max(1.0 / max(2, k_c) * 1.5, float(membership_cutoff))
        for u in range(n):
            active = np.flatnonzero(U[u] >= cutoff).tolist()
            if not active:
                active = [int(np.argmax(U[u]))]
            for c_idx in active:
                comms_dict.setdefault(int(c_idx), set()).add(u)
        return list(comms_dict.values()), U, extra_meta

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def run_cached_clustering(
    graph,
    algorithm: str = 'louvain',
    resolution: float = 1.0,
    seed: int = 42,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    **algo_params,
) -> dict:
    """Run any classic, overlapping, or fuzzy clustering algorithm once and cache on disk."""
    algo = str(algorithm).lower().strip()
    if algo not in CLUSTERING_CATALOG:
        raise ValueError(f"Unknown clustering algorithm '{algorithm}'.")

    adjacency: sparse.csr_matrix = graph['adj']
    n = adjacency.shape[0]
    fp = _graph_fingerprint(graph)
    param_sig = json.dumps({'res': round(float(resolution), 4), 'seed': int(seed), **algo_params}, sort_keys=True)
    param_hash = hashlib.sha256(param_sig.encode('utf-8')).hexdigest()[:10]
    cache_key = f'{fp}_{algo}_{param_hash}'

    if use_cache and cache_key in _CLUSTERING_MEM_CACHE:
        cached = _CLUSTERING_MEM_CACHE[cache_key]
        return {
            'node_frame': cached['node_frame'].clone(),
            'communities': [ list(c) for c in cached['communities'] ],
            'meta': dict(cached['meta'], cache_hit=True),
        }

    cdir = Path(cache_dir) if cache_dir is not None else CACHE_ROOT / 'clusterings'
    parquet_path = cdir / f'cluster_{cache_key}.parquet'
    json_path = cdir / f'cluster_{cache_key}.json'

    if use_cache and n >= 50 and parquet_path.exists() and json_path.exists():
        try:
            df_disk = pl.read_parquet(parquet_path)
            payload = json.loads(json_path.read_text(encoding='utf-8'))
            if df_disk.height == n and df_disk['protein'].to_list() == [str(x) for x in graph['nodes']]:
                meta_disk = payload['meta']
                meta_disk['cache_hit'] = True
                meta_disk['cache_path'] = str(parquet_path)
                comms_disk = payload['communities']
                _CLUSTERING_MEM_CACHE[cache_key] = {
                    'node_frame': df_disk,
                    'communities': comms_disk,
                    'meta': meta_disk,
                }
                return {
                    'node_frame': df_disk.clone(),
                    'communities': [list(c) for c in comms_disk],
                    'meta': dict(meta_disk),
                }
        except Exception:
            pass

    network = _advanced_graph(graph)
    raw_communities, soft_u, extra_meta = _execute_clustering_algorithm(
        graph, network, algorithm=algo, resolution=float(resolution), seed=int(seed), **algo_params
    )
    # Normalize communities to non-empty sorted lists of node indices
    communities = [sorted(c) for c in raw_communities if len(c) > 0]
    if not communities:
        communities = [list(range(n))]
    num_comms = len(communities)

    # Track all communities each node belongs to, plus primary community (largest or highest soft membership)
    node_comms: list[list[int]] = [[] for _ in range(n)]
    comm_sizes = [len(c) for c in communities]
    for c_idx, members in enumerate(communities):
        for u in members:
            node_comms[u].append(c_idx)
    for u in range(n):
        if not node_comms[u]:
            node_comms[u].append(0)
            communities[0].append(u)

    assignments = np.zeros(n, dtype=int)
    num_memberships = np.zeros(n, dtype=int)
    comms_str = []
    for u in range(n):
        c_list = node_comms[u]
        num_memberships[u] = len(c_list)
        comms_str.append(';'.join(str(c) for c in c_list))
        if soft_u is not None and soft_u.shape[1] >= num_comms:
            assignments[u] = int(np.argmax(soft_u[u]))
        else:
            assignments[u] = max(c_list, key=lambda c: comm_sizes[c])

    # Guimerà–Amaral participation, community entropy, and within-module z-score
    participation = np.zeros(n, dtype=float)
    entropy = np.zeros(n, dtype=float)
    within = np.zeros(n, dtype=float)
    indptr = adjacency.indptr
    indices = adjacency.indices
    for node in range(n):
        neighbors = indices[indptr[node]:indptr[node + 1]]
        counts = np.bincount(assignments[neighbors], minlength=num_comms)
        degree = len(neighbors)
        fractions = counts[counts > 0] / degree if degree else np.array([])
        participation[node] = 1.0 - float((fractions ** 2).sum()) if degree else 0.0
        entropy[node] = float(-(fractions * np.log(fractions)).sum()) if len(fractions) else 0.0
        within[node] = float(counts[assignments[node]])

    z = np.zeros(n, dtype=float)
    for members in communities:
        idx_arr = np.asarray(members, dtype=int)
        if len(idx_arr) > 1:
            std = within[idx_arr].std()
            if std > 0:
                z[idx_arr] = (within[idx_arr] - within[idx_arr].mean()) / std

    # Fuzzy / soft membership metrics (from soft_u if available, else derived from multi-membership + neighbor affinities)
    max_membership = np.ones(n, dtype=float)
    top2_ratio = np.zeros(n, dtype=float)
    fuzzy_entropy = np.zeros(n, dtype=float)
    fuzzy_part = np.zeros(n, dtype=float)
    effective_clusters = np.ones(n, dtype=float)

    if soft_u is not None and soft_u.shape[0] == n and soft_u.shape[1] >= 1:
        k_soft = soft_u.shape[1]
        log_k = math.log(max(2, k_soft))
        for u in range(n):
            p = np.maximum(0.0, soft_u[u])
            s = float(p.sum())
            p = p / s if s > 0 else np.full(k_soft, 1.0 / k_soft)
            pos = p[p > 1e-12]
            h = float(-(pos * np.log(pos)).sum())
            fuzzy_entropy[u] = h / log_k if k_soft > 1 else 0.0
            fuzzy_part[u] = 1.0 - float((p ** 2).sum())
            effective_clusters[u] = math.exp(h)
            sorted_p = np.sort(p)[::-1]
            max_membership[u] = float(sorted_p[0])
            top2_ratio[u] = float(sorted_p[1] / sorted_p[0]) if len(sorted_p) > 1 and sorted_p[0] > 1e-12 else 0.0
    else:
        log_k = math.log(max(2, num_comms))
        for u in range(n):
            neighbors = indices[indptr[u]:indptr[u + 1]]
            counts = np.bincount(assignments[neighbors], minlength=num_comms).astype(float)
            for c_idx in node_comms[u]:
                counts[c_idx] += 1.0
            total_c = float(counts.sum())
            p = counts / total_c if total_c > 0 else np.array([1.0])
            pos = p[p > 1e-12]
            h = float(-(pos * np.log(pos)).sum())
            fuzzy_entropy[u] = h / log_k if num_comms > 1 else 0.0
            fuzzy_part[u] = 1.0 - float((p ** 2).sum())
            effective_clusters[u] = math.exp(h)
            sorted_p = np.sort(p)[::-1]
            max_membership[u] = float(sorted_p[0])
            top2_ratio[u] = float(sorted_p[1] / sorted_p[0]) if len(sorted_p) > 1 and sorted_p[0] > 1e-12 else 0.0

    # Compute modularity on primary hard partition
    hard_buckets: dict[int, set[int]] = {}
    for u, c_id in enumerate(assignments):
        hard_buckets.setdefault(int(c_id), set()).add(u)
    try:
        mod_val = float(nx.community.modularity(network, list(hard_buckets.values()), weight=None, resolution=float(resolution)))
    except Exception:
        mod_val = float('nan')

    node_frame = pl.DataFrame({
        'protein': [str(x) for x in graph['nodes']],
        'community': assignments,
        'communities_str': comms_str,
        'num_communities': num_memberships,
        'is_overlapping_member': num_memberships > 1,
        'participation': participation,
        'community_entropy': entropy,
        'within_module_z': z,
        'max_membership': max_membership,
        'top2_membership_ratio': top2_ratio,
        'fuzzy_entropy': fuzzy_entropy,
        'fuzzy_participation': fuzzy_part,
        'effective_clusters': effective_clusters,
    })
    meta = {
        'algorithm': algo,
        'algorithm_label': CLUSTERING_CATALOG[algo]['label'],
        'paradigm': CLUSTERING_CATALOG[algo]['paradigm'],
        'communities': num_comms,
        'resolution': float(resolution),
        'seed': int(seed),
        'modularity': mod_val,
        'overlapping_nodes': int((num_memberships > 1).sum()),
        'overlapping_fraction': float((num_memberships > 1).mean()),
        'mean_communities_per_node': float(num_memberships.mean()),
        'cache_hit': False,
        **extra_meta,
    }

    if use_cache:
        _CLUSTERING_MEM_CACHE[cache_key] = {
            'node_frame': node_frame,
            'communities': communities,
            'meta': dict(meta),
        }
        if n >= 50:
            try:
                cdir.mkdir(parents=True, exist_ok=True)
                node_frame.write_parquet(parquet_path)
                meta['cache_path'] = str(parquet_path)
                json_path.write_text(json.dumps({'meta': meta, 'communities': communities}, indent=2), encoding='utf-8')
            except Exception:
                pass

    return {
        'node_frame': node_frame.clone(),
        'communities': [list(c) for c in communities],
        'meta': meta,
    }


def community_analysis(graph, cohort, seed=42, resolution=1.0, algorithm='louvain', use_cache=True, set_name='Moonlighting', **algo_params):
    """Full-network multi-paradigm clustering (Classic, Overlapping, Fuzzy) plus cohort enrichment (cached)."""
    mask = _selection(graph, cohort)
    cached_res = run_cached_clustering(
        graph, algorithm=algorithm, resolution=resolution, seed=seed, use_cache=use_cache, **algo_params
    )
    base_df = cached_res['node_frame']
    communities = cached_res['communities']
    meta = cached_res['meta']

    frame = base_df.with_columns(
        pl.Series('is_moonlighting', mask),
        pl.Series('group', np.where(mask, set_name, 'Background')),
    ).select([
        'protein', 'is_moonlighting', 'group', 'community', 'communities_str',
        'num_communities', 'is_overlapping_member',
        'participation', 'community_entropy', 'within_module_z',
        'fuzzy_entropy', 'fuzzy_participation', 'effective_clusters',
        'top2_membership_ratio', 'max_membership',
    ])

    rows = []
    for community_id, members in enumerate(communities):
        members = list(members)
        count = int(mask[members].sum())
        rows.append({
            'community': community_id,
            'network_proteins': len(members),
            'moonlighting_proteins': count,
            'expected': len(members) * mask.sum() / len(mask),
            'p_value': float(hypergeom.sf(count - 1, len(mask), int(mask.sum()), len(members))),
        })
    enrichment = pl.DataFrame(rows)
    enrichment = enrichment.with_columns(pl.Series('fdr_bh', bh_adjust(enrichment['p_value'])))

    test_metrics = [
        'participation', 'community_entropy', 'within_module_z',
        'num_communities', 'fuzzy_entropy', 'fuzzy_participation',
        'effective_clusters', 'top2_membership_ratio',
    ]
    summary = {
        'algorithm': meta['algorithm'],
        'algorithm_label': meta['algorithm_label'],
        'paradigm': meta['paradigm'],
        'communities': len(communities),
        'resolution': float(resolution),
        'seed': int(seed),
        'modularity': meta['modularity'],
        'overlapping_nodes': meta['overlapping_nodes'],
        'overlapping_fraction': meta['overlapping_fraction'],
        'mean_communities_per_node': meta['mean_communities_per_node'],
        'cache_hit': meta.get('cache_hit', False),
        'cache_path': meta.get('cache_path'),
        'scope': f"Full selected interactome; {meta['algorithm_label']} ({meta['paradigm']}). Cached per network and parameter signature.",
    }
    return {
        'metrics': frame,
        'enrichment': enrichment.sort('fdr_bh', maintain_order=True),
        'tests': _comparison_table(frame, test_metrics),
        'summary': summary,
    }


def compare_clustering_algorithms(
    graph,
    cohort,
    algorithms: list[str] | None = None,
    seed: int = 42,
    resolution: float = 1.0,
    use_cache: bool = True,
    set_name: str = 'Moonlighting',
) -> dict:
    """Run, cache, and compare multiple clustering algorithms (Classic, Overlapping, Fuzzy) on the same network."""
    algos = list(dict.fromkeys(algorithms or list(CLUSTERING_CATALOG.keys())))
    mask = _selection(graph, cohort)
    summary_rows = []
    effect_rows = []
    per_algorithm = {}
    key_metrics = ['participation', 'community_entropy', 'within_module_z', 'num_communities', 'fuzzy_entropy', 'top2_membership_ratio']

    for algo in algos:
        res = community_analysis(graph, cohort, seed=seed, resolution=resolution, algorithm=algo, use_cache=use_cache, set_name=set_name)
        per_algorithm[algo] = res
        s = res['summary']
        df = res['metrics']
        enr = res['enrichment']
        moon_ov = float(df.filter(pl.col('is_moonlighting'))['is_overlapping_member'].mean() or 0.0)
        bg_ov = float(df.filter(~pl.col('is_moonlighting'))['is_overlapping_member'].mean() or 0.0)
        sig_comms = int((enr['fdr_bh'] < 0.05).sum()) if len(enr) else 0

        summary_rows.append({
            'algorithm': algo,
            'label': s['algorithm_label'],
            'paradigm': s['paradigm'],
            'communities': s['communities'],
            'modularity': s['modularity'],
            'overlapping_fraction': s['overlapping_fraction'],
            'moonlighting_overlap_frac': moon_ov,
            f'{set_name.lower().replace(" ", "_")}_overlap_frac': moon_ov,
            'background_overlap_frac': bg_ov,
            'enriched_communities_bh': sig_comms,
            'cache_hit': bool(s.get('cache_hit', False)),
        })

        for metric in key_metrics:
            vals = df[metric].to_numpy().astype(float)
            a, b = vals[mask], vals[~mask]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            avail = len(a) > 0 and len(b) > 0
            u_res = mannwhitneyu(a, b, alternative='two-sided') if avail else None
            rb = float(2.0 * u_res.statistic / (len(a) * len(b)) - 1.0) if avail else np.nan
            effect_rows.append({
                'group': algo,
                'label': s['algorithm_label'],
                'paradigm': s['paradigm'],
                'metric': metric,
                'query_n': len(a),
                'background_n': len(b),
                'query_median': float(np.median(a)) if len(a) else np.nan,
                'background_median': float(np.median(b)) if len(b) else np.nan,
                'rank_biserial': rb,
                'p_value': float(u_res.pvalue) if avail else np.nan,
            })

    effects_df = pl.DataFrame(effect_rows).with_columns(
        pl.Series('fdr_bh', bh_adjust([r['p_value'] for r in effect_rows]))
    )
    summary_df = pl.DataFrame(summary_rows)
    return {
        'summary_table': summary_df,
        'effects': effects_df,
        'per_algorithm': per_algorithm,
        'summary': {
            'algorithms_evaluated': len(algos),
            'paradigms': sorted({CLUSTERING_CATALOG[a]['paradigm'] for a in algos}),
            'resolution': float(resolution),
            'seed': int(seed),
            'correction': 'Benjamini–Hochberg across all algorithm-by-metric comparisons in the suite.',
        },
    }


def clustering_comparison_heatmap(effects: pl.DataFrame) -> go.Figure:
    """Render a rank-biserial effect heatmap across Classic, Overlapping, and Fuzzy clustering algorithms."""
    if effects.is_empty():
        return _base_heatmap('Multi-paradigm clustering comparison')
    algos = effects['group'].unique(maintain_order=True).to_list()
    metrics = effects['metric'].unique(maintain_order=True).to_list()
    labels_map = {r['group']: f"[{r['paradigm'].upper()}] {r['group']}" for r in effects.iter_rows(named=True)}
    lookup = {(r['group'], r['metric']): r for r in effects.iter_rows(named=True)}
    z, text, hover = [], [], []
    for algo in algos:
        zr, tr, hr = [], [], []
        for m in metrics:
            row = lookup.get((algo, m))
            if row is None or not _finite(row.get('rank_biserial')):
                zr.append(None); tr.append('NA'); hr.append('Unavailable')
                continue
            eff = float(row['rank_biserial'])
            stars = significance_label(row.get('fdr_bh'))
            zr.append(eff)
            tr.append(f'{eff:+.2f}<br>{stars}')
            hr.append(
                f"{row.get('label', algo)} · {_metric_label(m).replace('<br>', ' ')}<br>"
                f"Paradigm: {row.get('paradigm')}<br>"
                f"Rank-biserial (Moonlighting vs Background): {eff:+.4f}<br>"
                f"Moonlighting median: {_fmt(row.get('query_median'))}<br>"
                f"Background median: {_fmt(row.get('background_median'))}<br>"
                f"BH-adjusted q: {_fmt(row.get('fdr_bh'), 6)} ({stars})"
            )
        z.append(zr); text.append(tr); hover.append(hr)
    fig = go.Figure(go.Heatmap(
        x=[_metric_label(m) for m in metrics],
        y=[labels_map.get(a, a) for a in algos],
        z=z, zmin=-1, zmax=1, zmid=0, colorscale=COLORSCALE,
        text=text, texttemplate='%{text}', textfont=dict(size=12),
        customdata=hover, hovertemplate='%{customdata}<extra></extra>',
        xgap=3, ygap=3,
        colorbar=dict(title='Rank-biserial<br>effect', tickvals=[-1, 0, 1], ticktext=['−1 · lower', '0 · equal', '+1 · higher']),
    ))
    fig.update_layout(
        template='plotly_white',
        title='Moonlighting topological roles across Classic, Overlapping & Fuzzy clustering algorithms',
        height=max(420, 140 + 48 * len(algos)),
        margin=dict(l=210, r=45, t=90, b=85),
        yaxis=dict(title='Clustering algorithm', autorange='reversed'),
        xaxis=dict(title='Clustering role metric', tickangle=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Sparse Random Walk with Restart (RWR) & Node-Removal Robustness
# ---------------------------------------------------------------------------

def _cohort_indices(graph, cohort):
    indices = np.array(sorted({graph['index'][protein] for protein in matching_nodes(graph['nodes'], cohort)}), dtype=int)
    if not len(indices):
        raise ValueError('No selected cohort proteins are present in this network.')
    return indices


def _sample_matched(degree, query_indices, rng):
    bins = _degree_bins(degree)
    chosen = []
    for label in np.unique(bins[query_indices]):
        count = np.count_nonzero(bins[query_indices] == label)
        chosen.extend(rng.choice(np.flatnonzero(bins == label), size=count, replace=False))
    return np.asarray(chosen, dtype=int)


def _transition(graph, weighted):
    matrix = sparse.csr_matrix(graph['weights'] if weighted else graph['adj'], dtype=float)
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0):
        raise ValueError('Walk affinities must be finite and nonnegative.')
    strengths = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1., strengths, out=np.zeros_like(strengths), where=strengths > 0)
    transition = matrix.T.multiply(inverse[None, :]).tocsr()
    return transition, strengths == 0


def _solve(transition, dangling, teleport, restart, max_iterations=500, tolerance=1e-10):
    scores = teleport.copy()
    residual = float('inf')
    for iteration in range(1, max_iterations + 1):
        propagated = transition @ scores + scores[dangling].sum() * teleport
        updated = restart * teleport + (1 - restart) * propagated
        residual = float(np.abs(updated - scores).sum())
        scores = updated
        if residual <= tolerance:
            break
    return scores, {'converged': residual <= tolerance, 'iterations': iteration,
                    'residual_l1': residual, 'tolerance': tolerance,
                    'maximum_iterations': max_iterations}


def random_walk_analysis(graph, cohort, restart=.3, weighted=False, steps=10,
                         permutations=0, seed=42):
    """Rank all network proteins by personalized random walk with restart."""
    if not np.isfinite(restart) or not 0 < restart <= 1:
        raise ValueError('Restart probability must be greater than 0 and at most 1.')
    if not 1 <= int(steps) <= 200:
        raise ValueError('Walk trace steps must be between 1 and 200.')
    if not 0 <= int(permutations) <= 50:
        raise ValueError('RWR permutations must be between 0 and 50.')
    permutations, steps = int(permutations), int(steps)
    n = len(graph['nodes'])
    query = _cohort_indices(graph, cohort)
    edge_count = graph['adj'].nnz // 2
    if permutations and edge_count > ADVANCED_EDGE_LIMIT:
        raise ValueError('RWR permutation tests require at most 500,000 edges; choose a saved huMAP threshold version.')
    transition, dangling = _transition(graph, weighted)
    teleport = np.zeros(n)
    teleport[query] = 1 / len(query)
    scores, convergence = _solve(transition, dangling, teleport, restart)
    baseline, baseline_convergence = _solve(transition, dangling, np.full(n, 1 / n), restart)
    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[query] = True
    seed_neighbors = np.asarray(graph['adj'] @ seed_mask.astype(float)).ravel().astype(int)
    degree = np.asarray(graph['degree']).astype(int)
    nonseeds = np.flatnonzero(~seed_mask)
    linker_p = np.full(n, np.nan)
    linker_q = np.full(n, np.nan)
    if len(nonseeds):
        linker_p[nonseeds] = hypergeom.sf(seed_neighbors[nonseeds] - 1, n - 1, len(query), degree[nonseeds])
        linker_q[nonseeds] = bh_adjust(linker_p[nonseeds])
    ranked = pl.DataFrame({
        'protein': graph['nodes'], 'is_seed': seed_mask,
        'score': scores, 'baseline_score': baseline,
        'lift': np.divide(scores, baseline, out=np.zeros(n), where=baseline > 0),
        'degree': degree, 'seed_neighbors': seed_neighbors,
        'linker_pvalue': linker_p, 'linker_qvalue': linker_q,
    })
    null_converged = True
    if permutations:
        rng = np.random.default_rng(seed)
        exceedances = np.zeros(len(nonseeds), dtype=int)
        for _ in range(permutations):
            sampled = _sample_matched(degree, query, rng)
            null_teleport = np.zeros(n)
            null_teleport[sampled] = 1 / len(sampled)
            null_scores, null_status = _solve(transition, dangling, null_teleport, restart)
            null_converged = null_converged and null_status['converged']
            exceedances += null_scores[nonseeds] >= scores[nonseeds]
        pvalues, qvalues = np.full(n, np.nan), np.full(n, np.nan)
        pvalues[nonseeds] = (1 + exceedances) / (permutations + 1)
        qvalues[nonseeds] = bh_adjust(pvalues[nonseeds])
        ranked = ranked.with_columns(pl.Series('empirical_pvalue', pvalues), pl.Series('empirical_qvalue', qvalues))
    trace = []
    current = teleport.copy()
    for step in range(steps + 1):
        mass = float(current[query].sum())
        trace.append({'step': step, 'seed_mass': mass, 'nonseed_mass': float(current[~seed_mask].sum())})
        current = restart * teleport + (1 - restart) * (transition @ current + current[dangling].sum() * teleport)
    summary = {
        'network_nodes': n, 'network_edges': edge_count,
        'mapped_seeds': len(query), 'nonseed_candidates': len(nonseeds),
        'restart_probability': restart, 'weighted': bool(weighted),
        'affinity_policy': 'probability weights' if weighted else 'binary edges',
        'score_convergence': convergence, 'baseline_convergence': baseline_convergence,
        'permutations': permutations, 'all_null_walks_converged': null_converged,
        'null_model': 'Seed counts matched within floor(log2 degree) bins; full network eligible, without replacement',
        'seed': seed, 'background': 'All proteins in the selected network',
    }
    return {'ranked': ranked.sort(['score', 'protein'], descending=[True, False]),
            'trace': pl.DataFrame(trace), 'summary': summary}


def _remaining_lcc(adjacency, removed):
    keep = np.ones(adjacency.shape[0], dtype=bool)
    keep[removed] = False
    remaining = int(keep.sum())
    if not remaining:
        return 0
    subgraph = adjacency[keep][:, keep]
    _, labels = connected_components(subgraph, directed=False)
    return int(np.bincount(labels).max())


def robustness_analysis(graph, cohort, permutations=100, seed=42):
    """Remove cohort nodes and compare remaining LCC with degree-matched removals."""
    if not 0 <= int(permutations) <= 100:
        raise ValueError('Removal permutations must be between 0 and 100.')
    permutations = int(permutations)
    query = _cohort_indices(graph, cohort)
    n = len(graph['nodes'])
    edge_count = graph['adj'].nnz // 2
    if permutations and edge_count > ADVANCED_EDGE_LIMIT:
        raise ValueError('Removal permutation tests require at most 500,000 edges; choose a saved huMAP threshold version.')
    original_lcc = _remaining_lcc(graph['adj'], np.array([], dtype=int))
    observed = _remaining_lcc(graph['adj'], query)
    rng = np.random.default_rng(seed)
    values = []
    for permutation in range(permutations):
        sampled = _sample_matched(graph['degree'], query, rng)
        count = _remaining_lcc(graph['adj'], sampled)
        values.append({'permutation': permutation + 1, 'remaining_lcc_nodes': count,
                       'fraction_original_network': count / n})
    pvalue = (1 + sum(row['remaining_lcc_nodes'] <= observed for row in values)) / (permutations + 1) if permutations else None
    summary = {
        'network_nodes': n, 'removed_nodes': len(query),
        'original_lcc_nodes': original_lcc, 'remaining_lcc_nodes': observed,
        'remaining_lcc_fraction_original': observed / n,
        'remaining_lcc_fraction_survivors': observed / (n - len(query)) if n > len(query) else 0.,
        'lcc_node_loss': original_lcc - observed,
        'empirical_pvalue_lower_tail': pvalue, 'permutations': permutations,
        'seed': seed, 'null_model': 'Matched floor(log2 degree) histogram; full graph eligible without replacement',
        'background': 'All proteins in the selected network',
    }
    return {'summary': summary, 'null': pl.DataFrame(values, schema={'permutation': pl.Int64, 'remaining_lcc_nodes': pl.Int64, 'fraction_original_network': pl.Float64})}


# ---------------------------------------------------------------------------
# Gene Ontology (GO) Annotation Propagation & Overrepresentation
# ---------------------------------------------------------------------------

def _signature(path):
    path = Path(path)
    info = path.stat()
    return {'path': str(path.resolve()), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ontology(path):
    terms, alternate, headers = {}, {}, []
    current = None
    def finish():
        if current and current.get('id') and not current.get('obsolete', False):
            terms[current['id']] = current
            for alt in current['alt']:
                alternate[alt] = current['id']
    with Path(path).open(encoding='utf-8') as stream:
        for raw in stream:
            line = raw.rstrip('\n')
            if line == '[Term]':
                finish()
                current = {'parents': [], 'alt': []}
            elif line.startswith('['):
                finish()
                current = None
            elif current is None:
                if line.startswith(('format-version:', 'data-version:', 'date:')):
                    headers.append(line)
            elif line.startswith('id: '):
                current['id'] = line[4:]
            elif line.startswith('name: '):
                current['name'] = line[6:]
            elif line.startswith('namespace: '):
                current['namespace'] = line[11:]
            elif line.startswith('alt_id: '):
                current['alt'].append(line[8:])
            elif line.startswith('is_a: '):
                current['parents'].append(line[6:].split()[0])
            elif line.startswith('relationship: part_of '):
                current['parents'].append(line.split()[2])
            elif line == 'is_obsolete: true':
                current['obsolete'] = True
        finish()
    @lru_cache(None)
    def ancestors(term):
        result = {term}
        for parent in terms[term]['parents']:
            parent = alternate.get(parent, parent)
            if parent in terms:
                result.update(ancestors(parent))
        return frozenset(result)
    return terms, alternate, headers, ancestors


def prepare_go_annotations(output_dir=Path('data/annotations'), gaf_path=DEFAULT_GAF,
                           obo_path=DEFAULT_OBO) -> Path:
    """Build/reuse propagated human GO annotations and their source provenance."""
    output_dir, gaf_path, obo_path = Path(output_dir), Path(gaf_path), Path(obo_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'human_go_propagated.parquet'
    metadata_path = target.with_suffix('.json')
    signatures = {'gaf': _signature(gaf_path), 'obo': _signature(obo_path)}
    if target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('sources') == signatures and metadata.get('policy') == GO_POLICY:
            return target
    terms, alternate, ontology_headers, ancestors = _ontology(obo_path)
    records, headers = [], []
    audit = {'source_rows': 0, 'not_rows_excluded': 0, 'nonhuman_or_nonuniprot_rows_excluded': 0,
             'unknown_or_obsolete_terms_excluded': 0}
    open_gaf = gzip.open if gaf_path.suffix.lower() == '.gz' else open
    with open_gaf(gaf_path, 'rt', encoding='utf-8') as stream:
        for raw in stream:
            if raw.startswith('!'):
                if raw.startswith(('!gaf-version:', '!date-generated:', '!go-version:', '!generated-by:')):
                    headers.append(raw.strip())
                continue
            fields = raw.rstrip('\n').split('\t')
            if len(fields) < 13:
                continue
            audit['source_rows'] += 1
            if 'NOT' in fields[3].split('|'):
                audit['not_rows_excluded'] += 1
                continue
            if fields[0] != 'UniProtKB' or fields[12].split('|')[0] != 'taxon:9606':
                audit['nonhuman_or_nonuniprot_rows_excluded'] += 1
                continue
            term = alternate.get(fields[4], fields[4])
            if term not in terms:
                audit['unknown_or_obsolete_terms_excluded'] += 1
                continue
            protein = re.sub(r'-\d+$', '', fields[1])
            if protein:
                records.append((protein, term, fields[6]))
    direct = pl.DataFrame(records, schema={'protein': pl.String, 'direct_term': pl.String, 'evidence': pl.String}, orient='row').unique()
    del records
    aspect_lookup = {'biological_process': 'P', 'molecular_function': 'F', 'cellular_component': 'C'}
    links = [(term, ancestor, aspect_lookup.get(terms[ancestor].get('namespace'), ''), terms[ancestor].get('name', ancestor))
             for term in direct['direct_term'].unique().to_list() for ancestor in ancestors(term)]
    ancestry = pl.DataFrame(links, schema={'direct_term': pl.String, 'term': pl.String, 'aspect': pl.String, 'name': pl.String}, orient='row')
    propagated = (direct.join(ancestry, on='direct_term').drop('direct_term')
                  .group_by('protein', 'term', 'aspect', 'name')
                  .agg(pl.col('evidence').unique().sort().str.join('|'))
                  .sort('protein', 'term'))
    temporary = target.with_suffix('.tmp.parquet')
    propagated.write_parquet(temporary, compression='zstd')
    temporary.replace(target)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'policy': GO_POLICY,
        'sources': signatures, 'gaf_sha256': _sha256(gaf_path), 'obo_sha256': _sha256(obo_path),
        'gaf_headers': headers, 'ontology_headers': ontology_headers,
        'audit': audit, 'direct_unique_evidence_rows': len(direct),
        'propagated_protein_term_rows': len(propagated),
        'proteins': propagated['protein'].n_unique(), 'terms': propagated['term'].n_unique(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return target


def go_enrichment(annotation_path, universe, cohort, aspect='BP', min_size=5,
                  exclude_iea=False) -> dict:
    """One-sided hypergeometric enrichment; BH includes all eligible terms."""
    if aspect not in {'BP', 'MF', 'CC', 'All'}:
        raise ValueError('aspect must be BP, MF, CC, or All')
    if int(min_size) != min_size or min_size < 1:
        raise ValueError('min_size must be a positive integer')
    population = {str(x) for x in universe}
    query = matching_nodes(population, cohort)
    if not population or not query:
        raise ValueError('The network universe and mapped cohort must both contain proteins')
    node_labels = pl.DataFrame({'node': sorted(population)}).with_columns(
        pl.col('node').str.split('|').alias('protein')).explode('protein')
    annotations = (pl.scan_parquet(annotation_path)
                   .join(node_labels.lazy(), on='protein', how='inner')
                   .drop('protein').rename({'node': 'protein'}))
    if aspect != 'All':
        annotations = annotations.filter(pl.col('aspect') == {'BP': 'P', 'MF': 'F', 'CC': 'C'}[aspect])
    if exclude_iea:
        annotations = annotations.filter(pl.col('evidence').str.split('|').list.eval(pl.element() != 'IEA').list.any())
    data = annotations.select('protein', 'term', 'name').unique().collect()
    annotated = set(data['protein'].to_list())
    counts = (data.group_by('term', 'name').agg(
        pl.col('protein').n_unique().alias('background_hits'),
        pl.col('protein').filter(pl.col('protein').is_in(sorted(query))).unique().sort().alias('query_proteins'))
        .filter(pl.col('background_hits') >= int(min_size)).sort('term'))
    rows = []
    n, N = len(query), len(population)
    for row in counts.iter_rows(named=True):
        hits, background = len(row['query_proteins']), row['background_hits']
        rows.append({'term': row['term'], 'name': row['name'], 'background_hits': background,
                     'query_hits': hits, 'fold': (hits / n) / (background / N),
                     'pvalue': float(hypergeom.sf(hits - 1, N, background, n)),
                     'proteins': '|'.join(row['query_proteins'])})
    columns = ['term', 'name', 'background_hits', 'query_hits', 'fold', 'pvalue', 'qvalue', 'proteins']
    table = pl.DataFrame(rows)
    if len(table):
        table = table.with_columns(pl.Series('qvalue', bh_adjust(table['pvalue'])))
        table = table.select(columns).sort(['qvalue', 'pvalue', 'term'])
    else:
        table = pl.DataFrame(schema={'term': pl.String, 'name': pl.String, 'background_hits': pl.Int64, 'query_hits': pl.Int64, 'fold': pl.Float64, 'pvalue': pl.Float64, 'qvalue': pl.Float64, 'proteins': pl.String})
    summary = {
        'population_size': N, 'query_size': n, 'annotated_population': len(annotated),
        'annotated_query': len(query & annotated), 'unannotated_population': N - len(annotated),
        'unannotated_query': n - len(query & annotated), 'tested_terms': len(table),
        'aspect': aspect, 'min_background_size': int(min_size), 'exclude_iea': bool(exclude_iea),
        'test': 'One-sided hypergeometric overrepresentation; BH over all eligible background terms including zero-hit terms',
        'annotation_path': str(annotation_path),
    }
    provenance = Path(annotation_path).with_suffix('.json')
    if provenance.exists():
        summary['annotation_provenance'] = json.loads(provenance.read_text())
    return {'table': table, 'summary': summary}


# ---------------------------------------------------------------------------
# Multi-Set Source Comparison (Shared Union-Complement Background)
# ---------------------------------------------------------------------------

def compare_source_sets(graph, data: pl.DataFrame, sources: list[str],
                        reviewed_only=False, min_publications=0, extra=False,
                        seed=42, landmarks=None, include_union=False):
    """Compute network metrics once, then test each source against the same control."""
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError('Select at least one moonlighting source set.')
    if {'Background', 'Union'} & set(sources):
        raise ValueError('Background and Union are reserved derived comparison groups.')
    unknown = set(sources) - set(data.columns)
    if unknown:
        raise ValueError('Unknown source sets: ' + ', '.join(sorted(unknown)))
    nodes = set(map(str, graph['nodes']))
    selected = {source: cohort_ids(data, [source], reviewed_only=reviewed_only,
                                   min_publications=min_publications)[0]
                for source in sources}
    present = {source: matching_nodes(nodes, ids) for source, ids in selected.items()}
    union = set().union(*present.values())
    if not union:
        raise ValueError('No selected source-set proteins occur in this network.')

    groups = list(sources)
    if include_union:
        selected['Union'] = set().union(*selected.values())
        present['Union'] = union
        groups.append('Union')

    base = centrality_analysis(graph, union, extra=extra, seed=seed, landmarks=landmarks)
    all_metrics = base['metrics']
    metric_names = base['tests']['metric'].to_list()
    background = all_metrics.filter(~pl.col('is_moonlighting'))
    frames, records, hub_records = [], [], []
    hub_populations = {}
    for metric in metric_names:
        values = all_metrics[metric].to_numpy().astype(float)
        values = values[np.isfinite(values)]
        cutoff = float(np.quantile(values, .9)) if len(values) else np.nan
        hub_populations[metric] = (len(values), int((values >= cutoff).sum()), cutoff)
    for source in groups:
        frame = all_metrics.filter(pl.col('protein').is_in(sorted(present[source])))
        frames.append(frame.with_columns(pl.lit(source).alias('group')))
        for metric in metric_names:
            query_values = frame[metric].to_numpy().astype(float)
            background_values = background[metric].to_numpy().astype(float)
            query_values = query_values[np.isfinite(query_values)]
            background_values = background_values[np.isfinite(background_values)]
            available = len(query_values) > 0 and len(background_values) > 0
            test = mannwhitneyu(query_values, background_values, alternative='two-sided') if available else None
            query_median = float(np.median(query_values)) if len(query_values) else np.nan
            background_median = float(np.median(background_values)) if len(background_values) else np.nan
            records.append({
                'group': source, 'comparator': 'Background', 'metric': metric,
                'query_n': len(query_values), 'background_n': len(background_values),
                'query_median': query_median, 'background_median': background_median,
                'rank_biserial': float(2 * test.statistic / (len(query_values) * len(background_values)) - 1) if available else np.nan,
                'median_difference': query_median - background_median if available else np.nan,
                'median_fold': query_median / background_median if available and background_median != 0 else np.nan,
                'p_value': float(test.pvalue) if available else np.nan,
            })
            population_n, top_n, cutoff = hub_populations[metric]
            hits = int((query_values >= cutoff).sum())
            hub_available = len(query_values) > 0 and population_n > 0 and top_n > 0
            hub_records.append({
                'group': source, 'metric': metric, 'query_n': len(query_values),
                'population_n': population_n, 'top_n': top_n,
                'query_top_n': hits, 'cutoff': cutoff,
                'fold': (hits / len(query_values)) / (top_n / population_n) if hub_available else np.nan,
                'p_value': float(hypergeom.sf(hits - 1, population_n, top_n, len(query_values))) if hub_available else np.nan,
            })
    frames.append(background.with_columns(pl.lit('Background').alias('group')))
    tests = pl.DataFrame(records, schema={
        'group': pl.String, 'comparator': pl.String, 'metric': pl.String,
        'query_n': pl.Int64, 'background_n': pl.Int64,
        'query_median': pl.Float64, 'background_median': pl.Float64,
        'rank_biserial': pl.Float64, 'median_difference': pl.Float64,
        'median_fold': pl.Float64, 'p_value': pl.Float64,
    }).with_columns(pl.Series('fdr_bh', bh_adjust([row['p_value'] for row in records])))
    profiles = tests.select('group', 'metric', 'query_n', 'background_n',
                            'query_median', 'background_median', 'rank_biserial',
                            'median_difference', 'median_fold', 'p_value', 'fdr_bh')
    hub_enrichment = pl.DataFrame(hub_records, schema={
        'group': pl.String, 'metric': pl.String, 'query_n': pl.Int64,
        'population_n': pl.Int64, 'top_n': pl.Int64, 'query_top_n': pl.Int64,
        'cutoff': pl.Float64, 'fold': pl.Float64, 'p_value': pl.Float64,
    }).with_columns(pl.Series('fdr_bh', bh_adjust([row['p_value'] for row in hub_records])))
    coverage = pl.DataFrame({
        'source': groups,
        'selected_ids': [len(selected[source]) for source in groups],
        'network_ids': [len(present[source]) for source in groups],
        'missing_ids': [len(selected[source] - {label for node in nodes for label in str(node).split('|')}) for source in groups],
    })
    overlap = pl.DataFrame([
        {'source_a': first, 'source_b': second,
         'selected_overlap': len(selected[first] & selected[second]),
         'network_overlap': len(present[first] & present[second])}
        for first, second in combinations(sources, 2)
    ], schema={'source_a': pl.String, 'source_b': pl.String,
               'selected_overlap': pl.Int64, 'network_overlap': pl.Int64})
    return {
        'metrics': pl.concat(frames), 'tests': tests,
        'profiles': profiles, 'hub_enrichment': hub_enrichment,
        'coverage': coverage, 'overlap': overlap,
        'source_ids': {source: sorted(ids) for source, ids in selected.items()},
        'summary': {
            'sources': sources, 'displayed_sets': groups, 'include_union': bool(include_union), 'network_nodes': len(nodes),
            'reviewed_only': bool(reviewed_only),
            'min_publications': int(min_publications),
            'union_network_ids': len(union), 'background_nodes': len(nodes - union),
            'test': 'Two-sided Mann–Whitney U against a shared unlabelled background',
            'correction': 'Benjamini–Hochberg across all finite displayed-set-by-metric tests in this run (including Union when shown)',
            'finite_tests': int(np.isfinite(tests['p_value'].to_numpy()).sum()),
            'effects': 'Rank-biserial = 2U/(query_n × background_n) − 1; positive means the source set tends to have higher values than shared background. Median fold is unavailable when background median is zero.',
            'hub_enrichment': {
                'threshold': 'At or above the full-network 90th percentile for each metric; ties are included, so the actual top fraction can exceed 10%.',
                'population': 'All network proteins with finite values for the metric, including all selected source sets; distinct from the shared comparison background.',
                'test': 'One-sided hypergeometric overrepresentation of the top set',
                'fold': '(query_top_n/query_n) / (top_n/population_n)',
                'correction': 'Separate Benjamini–Hochberg family across all finite displayed-set-by-metric hub tests, including Union when shown.',
                'finite_tests': int(np.isfinite(hub_enrichment['p_value'].to_numpy()).sum()),
            },
            'background': 'All network proteins outside the union of the selected source sets',
            'overlap': 'Proteins can appear in multiple source groups; source comparisons are dependent. No source-versus-source tests are performed.',
            'interpretation': 'Descriptive comparisons do not adjust for degree or network dependence; background proteins are not validated negatives.',
            'centrality_notes': base['notes'],
        },
    }


# ---------------------------------------------------------------------------
# Distribution Plots & Centrality / Hub Heatmaps
# ---------------------------------------------------------------------------

def significance_label(q):
    """Map a finite, valid adjusted p-value to strict significance thresholds."""
    try:
        value = float(q)
    except (TypeError, ValueError):
        return 'n/a'
    if not math.isfinite(value) or not 0 <= value <= 1:
        return 'n/a'
    for threshold, stars in ((.0001, '****'), (.001, '***'), (.01, '**'), (.05, '*')):
        if value < threshold:
            return stars
    return 'ns'


def metric_figure(frame: pl.DataFrame, metric: str, tests: pl.DataFrame | None = None,
                  max_per_group: int = 5000, seed: int = 42,
                  plot_kind: str = 'violin') -> go.Figure:
    """Plot supplied full-data tests with boxes, means, or distributions."""
    if plot_kind not in {'violin', 'box', 'bar', 'observed'}:
        raise ValueError('plot_kind must be violin, box, bar, or observed')
    if max_per_group < 1:
        raise ValueError('max_per_group must be positive')
    fig = go.Figure()
    title = metric.replace('_', ' ').title()
    fig.update_layout(template='plotly_white', title=f'{title} by protein set',
                      yaxis_title=title, xaxis_title='Protein set',
                      legend_title_text='Protein set', height=570,
                      margin=dict(l=65, r=30, t=110, b=130), violinmode='group',
                      legend=dict(orientation='h', yanchor='top', y=-.24, x=0))
    if not {'group', metric}.issubset(frame.columns) or not frame.schema[metric].is_numeric():
        fig.add_annotation(text='No numeric distribution available', x=.5, y=.5,
                           xref='paper', yref='paper', showarrow=False)
        return fig
    groups_present = frame['group'].drop_nulls().unique().to_list()
    groups = [group for group in GROUP_COLORS if group in groups_present and group != 'Background']
    groups += sorted(group for group in groups_present if group not in GROUP_COLORS)
    if 'Background' in groups_present:
        groups.append('Background')
    finite_groups = {group: frame.filter((pl.col('group') == group) & pl.col(metric).is_finite()) for group in groups}
    for group in groups:
        rows = finite_groups[group]
        if rows.is_empty():
            continue
        sampled = rows.sample(n=min(rows.height, max_per_group), seed=seed)
        name = GROUP_LABELS.get(group, group)
        color = GROUP_COLORS.get(group, '#555555')
        values = rows[metric]
        if plot_kind == 'observed':
            if group == 'Background':
                fig.add_trace(go.Histogram(x=values.to_list(), name='Background',
                                           marker_color=color, opacity=.45, histnorm='probability density',
                                           hovertemplate=title + ': %{x}<br>Density: %{y}<extra>Background</extra>'))
                continue
            median = float(values.median())
            test_label = ''
            test_hover = ''
            if tests is not None and {'metric', 'fdr_bh'}.issubset(tests.columns):
                candidates = tests.filter(pl.col('metric') == metric)
                if 'group' in candidates.columns:
                    candidates = candidates.filter(pl.col('group') == group)
                elif group == 'Background':
                    candidates = candidates.head(0)
                if 'comparator' in candidates.columns:
                    candidates = candidates.filter(pl.col('comparator') == 'Background')
                if len(candidates) and 'Background' in finite_groups and len(finite_groups['Background']):
                    q = candidates['fdr_bh'][0]
                    test_label = ' (' + significance_label(q) + ')'
                    test_hover = '<br>BH-adjusted q: unavailable' if significance_label(q) == 'n/a' else f'<br>BH-adjusted q = {float(q):.6g}'
            fig.add_trace(go.Scatter(x=[median, median], y=[0, 1], yaxis='y2', mode='lines',
                                     name=f'{name}<br>median {median:.4g}' + test_label,
                                     legendgroup=group, line=dict(color=color, dash='dash', width=2),
                                     hovertemplate=f'{name}<br>Full-data median: {median:.6g}' + test_hover + '<extra></extra>'))
        elif plot_kind == 'bar':
            mean = float(values.mean())
            sem = float(values.std()) / math.sqrt(len(values)) if len(values) > 1 else 0.
            fig.add_trace(go.Bar(x=[name], y=[mean], name=name, legendgroup=group,
                                 marker_color=color, error_y=dict(type='data', array=[sem], visible=True),
                                 customdata=[[sem, len(values)]],
                                 hovertemplate=title + ': %{y}<br>SEM: %{customdata[0]}<br>n: %{customdata[1]}<extra>' + name + '</extra>'))
        elif plot_kind == 'box':
            fig.add_trace(go.Box(x=[name] * rows.height, y=values.to_list(), name=name,
                                 legendgroup=group, marker_color=color, line_color=color,
                                 boxpoints=False, boxmean=True,
                                 hovertemplate=title + ': %{y}<extra>' + name + '</extra>'))
        else:
            proteins = sampled['protein'].to_list() if 'protein' in sampled.columns else [''] * sampled.height
            fig.add_trace(go.Violin(x=[name] * sampled.height, y=sampled[metric].to_list(), name=name,
                                    legendgroup=group, scalegroup=group, scalemode='width',
                                    line_color=color, fillcolor=color, opacity=.75,
                                    box_visible=True, meanline_visible=True, points=False, spanmode='hard',
                                    customdata=proteins,
                                    hovertemplate='%{customdata}<br>' + title + ': %{y}<extra>' + name + '</extra>'))
    if plot_kind == 'bar':
        fig.update_layout(title=f'{title}<br><sup>Full-data mean ± SEM (descriptive)</sup>', yaxis_rangemode='tozero')
    if plot_kind == 'observed':
        fig.update_layout(title=f'{title}<br><sup>Source medians against the background distribution</sup>',
                          xaxis_title=title, yaxis_title='Background probability density',
                          yaxis2=dict(overlaying='y', range=[0, 1], visible=False, fixedrange=True),
                          legend_title_text=None,
                          legend=dict(entrywidthmode='pixels', entrywidth=200))
        fig.add_annotation(x=0, y=1.10, xref='paper', yref='paper', xanchor='left',
                           text='Reference: background protein values (not a permutation null).',
                           showarrow=False, font=dict(size=12, color='#444444'))
    if plot_kind in {'bar', 'box'}:
        fig.update_layout(showlegend=False, margin_b=100)
    if plot_kind != 'observed':
        fig.update_xaxes(categoryorder='array', categoryarray=[GROUP_LABELS.get(g, g) for g in groups],
                         tickangle=-25 if len(groups) > 3 else 0)
    if not fig.data:
        fig.add_annotation(text='No finite values available', x=.5, y=.5,
                           xref='paper', yref='paper', showarrow=False)
        return fig
    if plot_kind != 'observed' and tests is not None and {'metric', 'fdr_bh'}.issubset(tests.columns):
        non_bg_groups = [g for g in groups if g != 'Background']
        fallback_group = non_bg_groups[0] if non_bg_groups else 'Moonlighting'
        for test in tests.filter(pl.col('metric') == metric).iter_rows(named=True):
            group = test.get('group', fallback_group)
            comparator = test.get('comparator', 'Background')
            if group not in groups or comparator != 'Background' or 'Background' not in groups:
                continue
            if frame.filter((pl.col('group') == group) & pl.col(metric).is_finite()).is_empty():
                continue
            if frame.filter((pl.col('group') == 'Background') & pl.col(metric).is_finite()).is_empty():
                continue
            q = test['fdr_bh']
            label = significance_label(q)
            hover = 'BH-adjusted q: unavailable' if label == 'n/a' else f'BH-adjusted q = {float(q):.6g}'
            fig.add_annotation(x=GROUP_LABELS.get(group, group), y=1.04, xref='x', yref='paper',
                               text=f'<b>{label}</b>', hovertext=hover + '<br>Comparison: ' + GROUP_LABELS.get(group, group) + ' vs Background',
                               showarrow=False, font=dict(size=17, color=GROUP_COLORS.get(group, '#555555')))
        fig.add_annotation(x=0, y=1.10, xref='paper', yref='paper', xanchor='left',
                           text='Significance versus Background (BH-adjusted q)',
                           showarrow=False, font=dict(size=12, color='#444444'))
    return fig


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _fmt(value, digits=4):
    return f'{float(value):.{digits}g}' if _finite(value) else 'unavailable'


def _axes(table):
    groups = table['group'].unique(maintain_order=True).to_list()
    metrics = table['metric'].unique(maintain_order=True).to_list()
    return groups, metrics


def _base_heatmap(title, groups=(), metrics=()):
    fig = go.Figure()
    fig.update_layout(template='plotly_white', title=title,
                      height=max(340, 135 + 68 * len(groups)),
                      margin=dict(l=145, r=45, t=105, b=95),
                      paper_bgcolor='white', plot_bgcolor='#E5E7EB',
                      xaxis=dict(title='Centrality metric', side='bottom', tickangle=0),
                      yaxis=dict(title='Protein set', autorange='reversed'))
    if not groups or not metrics:
        fig.add_annotation(x=.5, y=.5, xref='paper', yref='paper',
                           text='No comparison results available', showarrow=False)
    return fig


def _metric_label(metric):
    label = METRIC_LABELS.get(metric)
    if label:
        return label
    words = str(metric).replace('_', ' ').split()
    return ' '.join(words[:2]).capitalize() + ('<br>' + ' '.join(words[2:]) if len(words) > 2 else '')


def _missing_labels(fig, z, metrics, groups):
    for i, row in enumerate(z):
        for j, value in enumerate(row):
            if value is None:
                fig.add_annotation(x=_metric_label(metrics[j]), y=GROUP_LABELS.get(groups[i], groups[i]),
                                   text='NA', showarrow=False, font=dict(color='#555555', size=13))


def centrality_heatmap(profiles: pl.DataFrame, value='rank_biserial') -> go.Figure:
    """Show source-versus-background effects with stars from supplied BH q-values."""
    if value != 'rank_biserial':
        raise ValueError('Supported centrality heatmap value: rank_biserial')
    if profiles.is_empty() or not {'group', 'metric', value}.issubset(profiles.columns):
        return _base_heatmap('Centrality profiles across protein sets')
    groups, metrics = _axes(profiles)
    fig = _base_heatmap('Centrality profiles across protein sets', groups, metrics)
    lookup = {(r['group'], r['metric']): r for r in profiles.iter_rows(named=True)}
    z, text, hover = [], [], []
    for group in groups:
        zr, tr, hr = [], [], []
        for metric in metrics:
            row = lookup.get((group, metric))
            if row is None or not _finite(row.get(value)):
                zr.append(None); tr.append('NA'); hr.append('Comparison unavailable')
                continue
            effect = float(row[value])
            stars = significance_label(row.get('fdr_bh'))
            zr.append(effect); tr.append(f'{effect:+.2f}<br>{stars}')
            hr.append(f'{GROUP_LABELS.get(group, group)} · {_metric_label(metric).replace("<br>", " ")}<br>'
                      f'Rank-biserial effect: {effect:.6g}<br>'
                      f'Source median: {_fmt(row.get("query_median"))}<br>'
                      f'Background median: {_fmt(row.get("background_median"))}<br>'
                      f'Median difference: {_fmt(row.get("median_difference"))}<br>'
                      f'Median fold: {_fmt(row.get("median_fold"))}<br>'
                      f'Source n: {row.get("query_n", "unavailable")}<br>'
                      f'Background n: {row.get("background_n", "unavailable")}<br>'
                      f'Raw p: {_fmt(row.get("p_value"), 6)}<br>'
                      f'BH-adjusted q: {_fmt(row.get("fdr_bh"), 6)} ({stars})')
        z.append(zr); text.append(tr); hover.append(hr)
    fig.add_trace(go.Heatmap(
        x=[_metric_label(m) for m in metrics], y=[GROUP_LABELS.get(g, g) for g in groups], z=z,
        zmin=-1, zmax=1, zmid=0, colorscale=COLORSCALE,
        colorbar=dict(title='Rank-biserial<br>effect', tickvals=[-1, 0, 1],
                      ticktext=['−1 · lower', '0 · equal', '+1 · higher']),
        text=text, texttemplate='%{text}', textfont=dict(size=13),
        customdata=hover, hovertemplate='%{customdata}<extra></extra>',
        hoverongaps=False, xgap=3, ygap=3))
    fig.add_annotation(x=0, y=1.13, xref='paper', yref='paper', xanchor='left', showarrow=False,
                       text='Source vs shared background · positive effects indicate higher centrality · stars: BH-adjusted q',
                       font=dict(size=11, color='#444444'))
    _missing_labels(fig, z, metrics, groups)
    return fig


def hub_enrichment_heatmap(table: pl.DataFrame) -> go.Figure:
    """Show exact hub-enrichment fold; only the color saturates above twofold."""
    if table.is_empty() or not {'group', 'metric', 'fold'}.issubset(table.columns):
        return _base_heatmap('Hub enrichment across protein sets')
    groups, metrics = _axes(table)
    fig = _base_heatmap('Hub enrichment across protein sets', groups, metrics)
    lookup = {(r['group'], r['metric']): r for r in table.iter_rows(named=True)}
    z, text, hover = [], [], []
    for group in groups:
        zr, tr, hr = [], [], []
        for metric in metrics:
            row = lookup.get((group, metric))
            if row is None or not _finite(row.get('fold')) or row['fold'] < 0:
                zr.append(None); tr.append('NA'); hr.append('Comparison unavailable'); continue
            fold = float(row['fold']); stars = significance_label(row.get('fdr_bh'))
            zr.append(fold - 1); tr.append(f'{fold:.2f}×<br>{stars}')
            hr.append(f'{GROUP_LABELS.get(group, group)} · {_metric_label(metric).replace("<br>", " ")}<br>'
                      f'Hub enrichment fold: {fold:.6g}×<br>'
                      f'Source hubs: {row.get("query_top_n", "unavailable")} / {row.get("query_n", "unavailable")}<br>'
                      f'Network hubs: {row.get("top_n", "unavailable")} / {row.get("population_n", "unavailable")}<br>'
                      f'Hub cutoff: {_fmt(row.get("cutoff"))}<br>'
                      f'Raw p: {_fmt(row.get("p_value"), 6)}<br>'
                      f'BH-adjusted q: {_fmt(row.get("fdr_bh"), 6)} ({stars})<br>'
                      'Color saturates at 2×; cell labels report the exact fold.')
        z.append(zr); text.append(tr); hover.append(hr)
    fig.add_trace(go.Heatmap(
        x=[_metric_label(m) for m in metrics], y=[GROUP_LABELS.get(g, g) for g in groups],
        z=z, zmin=-1, zmax=1, zmid=0, colorscale=COLORSCALE,
        colorbar=dict(title='Hub enrichment', tickvals=[-1, 0, 1], ticktext=['0×', '1× · expected', '≥2×']),
        text=text, texttemplate='%{text}', textfont=dict(size=13), customdata=hover,
        hovertemplate='%{customdata}<extra></extra>', hoverongaps=False, xgap=3, ygap=3))
    fig.add_annotation(x=0, y=1.13, xref='paper', yref='paper', xanchor='left', showarrow=False,
                       text='Exact fold enrichment · colors saturate at ≥2× · stars: BH-adjusted q',
                       font=dict(size=11, color='#444444'))
    _missing_labels(fig, z, metrics, groups)
    return fig


def protein_set_network_summary(frame: pl.DataFrame, set_name: str = 'Protein set') -> pl.DataFrame:
    """Label current and cached overview summaries without changing their stored schema."""
    aliases = {
        'Cohort proteins present': f'{set_name} proteins present',
        'Protein set members present': f'{set_name} proteins present',
        'Moonlighting proteins present': f'{set_name} proteins present',
        'Cohort network nodes': f'{set_name} network nodes',
        'Protein set network nodes': f'{set_name} network nodes',
        'Cohort coverage (%)': f'{set_name} coverage (%)',
        'Protein set coverage (%)': f'{set_name} coverage (%)',
        'Moonlighting coverage (%)': f'{set_name} coverage (%)',
    }
    for old, new in aliases.items():
        if old == new or old not in frame.columns:
            continue
        frame = frame.drop(old) if new in frame.columns else frame.rename({old: new})
    return frame


def multi_network_centrality_comparison(
    graphs: Mapping[str, dict],
    cohort: set[str],
    extra: bool = False,
    seed: int = 42,
    landmarks: int | None = None,
    use_cache: bool = True,
    set_name: str = 'Moonlighting',
) -> dict:
    """Compute cached centralities across multiple networks and compare cohort vs background across networks."""
    if not graphs:
        raise ValueError('Provide at least one network to compare.')
    metrics_frames = []
    effect_records = []
    hub_records = []
    network_summaries = []
    notes_by_network = {}

    for net_name, graph in graphs.items():
        nodes = set(map(str, graph['nodes']))
        present = matching_nodes(nodes, cohort)
        mask = np.asarray([str(node) in present for node in graph['nodes']], dtype=bool)
        base_frame, notes = cached_network_centralities(
            graph, extra=extra, seed=seed, landmarks=landmarks, use_cache=use_cache
        )
        notes_by_network[net_name] = notes
        covered_labels = len(cohort & {label for node in nodes for label in str(node).split('|')})
        n_nodes, n_edges = len(nodes), graph['edge_count']
        network_summaries.append({
            'Network': net_name,
            'Nodes': n_nodes,
            'Edges': n_edges,
            'Mean degree': 2 * n_edges / n_nodes if n_nodes else 0.0,
            'Density': 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
            'Cohort proteins present': covered_labels,
            f'{set_name} proteins present': covered_labels,
            'Cohort network nodes': int(mask.sum()),
            'Cohort coverage (%)': 100.0 * covered_labels / len(cohort) if cohort else 0.0,
            f'{set_name} coverage (%)': 100.0 * covered_labels / len(cohort) if cohort else 0.0,
            'Background nodes': int((~mask).sum()),
            'Cached': bool(notes.get('cache_hit', False)),
        })
        metric_cols = ['degree', 'strength', 'pagerank', 'mean_neighbor_degree']
        if extra:
            metric_cols += ['core_number', 'clustering', 'betweenness_approx', 'eigenvector', 'bridging_coefficient', 'bridging_centrality']
        metric_cols = [c for c in metric_cols if c in base_frame.columns]

        net_frame = base_frame.with_columns(
            pl.lit(net_name).alias('network'),
            pl.Series('is_moonlighting', mask),
            pl.Series('group', np.where(mask, set_name, 'Background')),
        ).select(['network', 'protein', 'is_moonlighting', 'group', *metric_cols])
        metrics_frames.append(net_frame)

        for metric in metric_cols:
            vals = base_frame[metric].to_numpy().astype(float)
            a, b = vals[mask], vals[~mask]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            avail = len(a) > 0 and len(b) > 0
            test = mannwhitneyu(a, b, alternative='two-sided') if avail else None
            q_med = float(np.median(a)) if len(a) else np.nan
            bg_med = float(np.median(b)) if len(b) else np.nan
            rb = float(2.0 * test.statistic / (len(a) * len(b)) - 1.0) if avail else np.nan
            effect_records.append({
                'group': net_name,
                'network': net_name,
                'comparator': 'Background',
                'metric': metric,
                'query_n': len(a),
                'background_n': len(b),
                'query_median': q_med,
                'background_median': bg_med,
                'rank_biserial': rb,
                'median_difference': q_med - bg_med if avail else np.nan,
                'median_fold': q_med / bg_med if avail and bg_med != 0 else np.nan,
                'p_value': float(test.pvalue) if avail else np.nan,
            })
            finite_all = vals[np.isfinite(vals)]
            cutoff = float(np.quantile(finite_all, 0.9)) if len(finite_all) else np.nan
            pop_n = len(finite_all)
            top_n = int((finite_all >= cutoff).sum()) if len(finite_all) else 0
            hits = int((a >= cutoff).sum()) if len(a) and np.isfinite(cutoff) else 0
            hub_avail = len(a) > 0 and pop_n > 0 and top_n > 0
            hub_records.append({
                'group': net_name,
                'network': net_name,
                'metric': metric,
                'query_n': len(a),
                'population_n': pop_n,
                'top_n': top_n,
                'query_top_n': hits,
                'cutoff': cutoff,
                'fold': (hits / len(a)) / (top_n / pop_n) if hub_avail else np.nan,
                'p_value': float(hypergeom.sf(hits - 1, pop_n, top_n, len(a))) if hub_avail else np.nan,
            })

    combined_metrics = pl.concat(metrics_frames, how='diagonal_relaxed') if metrics_frames else pl.DataFrame()
    effects_df = pl.DataFrame(effect_records, schema={
        'group': pl.String, 'network': pl.String, 'comparator': pl.String, 'metric': pl.String,
        'query_n': pl.Int64, 'background_n': pl.Int64,
        'query_median': pl.Float64, 'background_median': pl.Float64,
        'rank_biserial': pl.Float64, 'median_difference': pl.Float64,
        'median_fold': pl.Float64, 'p_value': pl.Float64,
    }).with_columns(pl.Series('fdr_bh', bh_adjust([r['p_value'] for r in effect_records])))

    hubs_df = pl.DataFrame(hub_records, schema={
        'group': pl.String, 'network': pl.String, 'metric': pl.String, 'query_n': pl.Int64,
        'population_n': pl.Int64, 'top_n': pl.Int64, 'query_top_n': pl.Int64,
        'cutoff': pl.Float64, 'fold': pl.Float64, 'p_value': pl.Float64,
    }).with_columns(pl.Series('fdr_bh', bh_adjust([r['p_value'] for r in hub_records])))

    return {
        'networks': pl.DataFrame(network_summaries),
        'metrics': combined_metrics,
        'tests': effects_df,
        'profiles': effects_df,
        'hub_enrichment': hubs_df,
        'notes': notes_by_network,
    }


def cross_network_metric_figure(
    frame: pl.DataFrame,
    metric: str,
    tests: pl.DataFrame | None = None,
    max_per_group: int = 5000,
    seed: int = 42,
    plot_kind: str = 'box',
    log_scale: bool = False,
    set_name: str | None = None,
) -> go.Figure:
    """Single unified plot showing how a metric varies for a protein set vs Background across multiple networks."""
    if plot_kind not in {'box', 'violin', 'bar'}:
        raise ValueError('plot_kind must be box, violin, or bar')
    fig = go.Figure()
    title = metric.replace('_', ' ').title()
    target_group = set_name or (
        [g for g in frame['group'].drop_nulls().unique(maintain_order=True).to_list() if g != 'Background']
        if 'group' in frame.columns else []
    )
    target_name = target_group if isinstance(target_group, str) else (target_group[0] if target_group else 'Moonlighting')
    label_name = GROUP_LABELS.get(target_name, target_name)
    fig.update_layout(
        template='plotly_white',
        title=f'{title} across interactomes ({label_name} vs Background)',
        yaxis_title=title + (' (log scale)' if log_scale else ''),
        xaxis_title='Interactome network',
        legend_title_text='Protein group',
        height=520,
        margin=dict(l=65, r=30, t=105, b=95),
        boxmode='group',
        violinmode='group',
        barmode='group',
        legend=dict(orientation='h', yanchor='top', y=-0.18, x=0),
    )
    if log_scale:
        fig.update_yaxes(type='log')
    if frame.is_empty() or not {'network', 'group', metric}.issubset(frame.columns) or not frame.schema[metric].is_numeric():
        fig.add_annotation(text='No numeric distribution available', x=0.5, y=0.5,
                           xref='paper', yref='paper', showarrow=False)
        return fig

    networks = frame['network'].drop_nulls().unique(maintain_order=True).to_list()
    present = frame['group'].drop_nulls().unique(maintain_order=True).to_list()
    groups = [g for g in GROUP_COLORS if g in present and g != 'Background']
    groups += [g for g in present if g not in GROUP_COLORS]
    if 'Background' in present:
        groups.append('Background')
    if len(groups) > 2:
        fig.update_layout(title=f'{title} across interactomes by protein set')
    for group in groups:
        color = GROUP_COLORS.get(group, '#555555')
        label = GROUP_LABELS.get(group, group)
        sub = frame.filter((pl.col('group') == group) & pl.col(metric).is_finite())
        if sub.is_empty():
            continue
        if plot_kind == 'bar':
            means, sems, counts, x_nets = [], [], [], []
            for net in networks:
                net_rows = sub.filter(pl.col('network') == net)
                if net_rows.is_empty():
                    continue
                vals = net_rows[metric]
                x_nets.append(net)
                means.append(float(vals.mean()))
                sems.append(float(vals.std()) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                counts.append(len(vals))
            fig.add_trace(go.Bar(
                x=x_nets, y=means, name=label, legendgroup=group, marker_color=color,
                error_y=dict(type='data', array=sems, visible=True),
                customdata=list(zip(sems, counts)),
                hovertemplate='%{x} · ' + label + '<br>' + title + ' mean: %{y:.4g}<br>SEM: %{customdata[0]:.4g}<br>n: %{customdata[1]}<extra></extra>',
            ))
        elif plot_kind == 'box':
            fig.add_trace(go.Box(
                x=sub['network'].to_list(), y=sub[metric].to_list(), name=label,
                legendgroup=group, marker_color=color, line_color=color,
                boxpoints=False, boxmean=True,
                hovertemplate='%{x} · ' + label + '<br>' + title + ': %{y}<extra></extra>',
            ))
        else:
            sampled_parts = [
                sub.filter(pl.col('network') == net).sample(
                    n=min(sub.filter(pl.col('network') == net).height, max_per_group), seed=seed
                )
                for net in networks if not sub.filter(pl.col('network') == net).is_empty()
            ]
            sampled = pl.concat(sampled_parts) if sampled_parts else sub
            proteins = sampled['protein'].to_list() if 'protein' in sampled.columns else [''] * sampled.height
            fig.add_trace(go.Violin(
                x=sampled['network'].to_list(), y=sampled[metric].to_list(), name=label,
                legendgroup=group, scalegroup=group, scalemode='width',
                line_color=color, fillcolor=color, opacity=0.75,
                box_visible=True, meanline_visible=True, points=False, spanmode='hard',
                customdata=proteins,
                hovertemplate='%{x} · %{customdata}<br>' + title + ': %{y}<extra>' + label + '</extra>',
            ))

    fig.update_xaxes(categoryorder='array', categoryarray=networks)
    if tests is not None and len(groups) <= 2 and {'metric', 'fdr_bh'}.issubset(tests.columns):
        net_col = 'network' if 'network' in tests.columns else 'group'
        for row in tests.filter(pl.col('metric') == metric).iter_rows(named=True):
            net = row.get(net_col)
            if net not in networks:
                continue
            q = row.get('fdr_bh')
            stars = significance_label(q)
            rb = row.get('rank_biserial')
            rb_txt = f' (r={float(rb):+.2f})' if _finite(rb) else ''
            hover = f'{net}: Moonlighting vs Background<br>BH-adjusted q = {_fmt(q, 6)}{rb_txt}'
            fig.add_annotation(
                x=net, y=1.05, xref='x', yref='paper',
                text=f'<b>{stars}</b>{rb_txt}', hovertext=hover,
                showarrow=False, font=dict(size=13, color='#0072B2'),
            )
        fig.add_annotation(
            x=0, y=1.12, xref='paper', yref='paper', xanchor='left',
            text='Stars & rank-biserial r compare Moonlighting vs Background within each network (BH-adjusted q across networks & metrics)',
            showarrow=False, font=dict(size=11, color='#444444'),
        )
    return fig


# ---------------------------------------------------------------------------
# Pooling per-network results into one cross-network comparison
# ---------------------------------------------------------------------------

_POOL_EXCLUDED = {'network', 'protein', 'is_moonlighting', 'group', 'community', 'is_seed'}
_EFFECT_SCHEMA = {
    'network': pl.String, 'group': pl.String, 'metric': pl.String,
    'query_n': pl.Int64, 'background_n': pl.Int64,
    'query_median': pl.Float64, 'background_median': pl.Float64,
    'rank_biserial': pl.Float64, 'median_difference': pl.Float64,
    'median_fold': pl.Float64, 'p_value': pl.Float64,
}


def pool_network_comparisons(results: Mapping[str, dict], metrics_key: str = 'metrics') -> dict:
    """Stack per-network metric tables and test every network-by-metric pair in one BH family.

    Each result must hold a per-protein table with a ``group`` column labelled ``Moonlighting``
    or ``Background``. The cohort is compared with that network's own background (two-sided
    Mann–Whitney U, rank-biserial effect); Benjamini–Hochberg runs across all finite tests from
    all networks together. Degree-matched samples stay as sampled by the original analysis.
    """
    frames, records, metric_order = [], [], []
    for network, result in results.items():
        frame = (result or {}).get(metrics_key)
        if not isinstance(frame, pl.DataFrame) or frame.is_empty() or 'group' not in frame.columns:
            continue
        tested = result.get('tests')
        if isinstance(tested, pl.DataFrame) and 'metric' in tested.columns:
            metrics = [m for m in tested['metric'].to_list() if m in frame.columns and frame.schema[m].is_numeric()]
        else:
            metrics = [c for c, dtype in frame.schema.items() if dtype.is_numeric() and c not in _POOL_EXCLUDED]
        metric_order += [m for m in metrics if m not in metric_order]
        keep = [c for c in ('protein',) if c in frame.columns]
        frames.append(frame.select(
            pl.lit(network).alias('network'), *keep, pl.col('group').cast(pl.String),
            *[pl.col(m).cast(pl.Float64) for m in metrics],
        ))
        mask = (frame['group'].cast(pl.String) != 'Background').to_numpy()
        for metric in metrics:
            values = frame[metric].cast(pl.Float64).to_numpy()
            a, b = values[mask], values[~mask]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            available = len(a) > 0 and len(b) > 0
            test = mannwhitneyu(a, b, alternative='two-sided') if available else None
            q_med = float(np.median(a)) if len(a) else np.nan
            b_med = float(np.median(b)) if len(b) else np.nan
            records.append({
                'network': network, 'group': network, 'metric': metric,
                'query_n': len(a), 'background_n': len(b),
                'query_median': q_med, 'background_median': b_med,
                'rank_biserial': float(2 * test.statistic / (len(a) * len(b)) - 1) if available else np.nan,
                'median_difference': q_med - b_med if available else np.nan,
                'median_fold': q_med / b_med if available and b_med != 0 else np.nan,
                'p_value': float(test.pvalue) if available else np.nan,
            })
    tests = pl.DataFrame(records, schema=_EFFECT_SCHEMA).with_columns(
        pl.Series('fdr_bh', bh_adjust([r['p_value'] for r in records]), dtype=pl.Float64))
    if metric_order and tests.height:
        order = {m: i for i, m in enumerate(metric_order)}
        networks = {n: i for i, n in enumerate(results)}
        tests = tests.sort(pl.col('network').replace_strict(networks, return_dtype=pl.Int64),
                           pl.col('metric').replace_strict(order, return_dtype=pl.Int64))
    metrics = pl.concat(frames, how='diagonal_relaxed') if frames else pl.DataFrame()
    return {
        'metrics': metrics, 'tests': tests, 'metric_order': metric_order,
        'correction': 'Benjamini–Hochberg across all finite network-by-metric tests in this comparison',
    }


def stack_network_tables(tables: Mapping[str, pl.DataFrame | None], p_column: str | None = None,
                         q_column: str = 'fdr_bh') -> pl.DataFrame:
    """Concatenate per-network tables with a leading ``network`` column.

    When ``p_column`` is given, ``q_column`` is recomputed with Benjamini–Hochberg across every
    finite p-value from all networks, so the correction family spans the whole comparison.
    """
    frames = [
        table.with_columns(pl.lit(network).alias('network')).select('network', pl.exclude('network'))
        for network, table in tables.items() if isinstance(table, pl.DataFrame) and table.height
    ]
    if not frames:
        return pl.DataFrame()
    stacked = pl.concat(frames, how='diagonal_relaxed')
    if p_column and p_column in stacked.columns:
        stacked = stacked.with_columns(pl.Series(
            q_column, bh_adjust(stacked[p_column].cast(pl.Float64).to_numpy()), dtype=pl.Float64))
    return stacked
