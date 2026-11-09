"""Moonlighting protein cohort preparation, network topology, diffusion, GO enrichment, comparisons, and plots."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
from itertools import combinations
import json
import math
from pathlib import Path
import re
import zipfile

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import hypergeom, mannwhitneyu, rankdata, spearmanr

SOURCES = ['curated', 'moonlitdb', 'moonprot', 'metamoon', 'multitask', 'moondb_curated', 'moondb_predicted']
ADVANCED_EDGE_LIMIT = 500_000
DEFAULT_GAF = Path('/home/davide/data/GO/goa_human.gaf')
DEFAULT_OBO = Path('/home/davide/data/GO/go-basic.obo')
GO_POLICY = 'UniProtKB human primary taxon; exclude NOT; collapse numeric isoform suffix; GO ancestor propagation via is_a and part_of; evidence retained'

GROUP_COLORS = {
    'curated': '#0072B2', 'moonlitdb': '#D55E00', 'moonprot': '#009E73',
    'metamoon': '#CC79A7', 'multitask': '#E69F00',
    'moondb_curated': '#56B4E9', 'moondb_predicted': '#8064A2',
    'Union': '#7B3294', 'Moonlighting': '#0072B2', 'Background': '#737373',
}
GROUP_LABELS = {
    'curated': 'Curated', 'moonlitdb': 'MoonLitDB', 'moonprot': 'MoonProt',
    'metamoon': 'MetaMoon', 'multitask': 'Multitask',
    'moondb_curated': 'MoonDB curated', 'moondb_predicted': 'MoonDB predicted',
    'Union': 'Union', 'Moonlighting': 'Moonlighting', 'Background': 'Background',
}
SIGNIFICANCE_LEGEND = (
    'Asterisks compare each set with Background using BH-adjusted q-values: '
    '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; '
    'ns: q ≥ 0.05; n/a: unavailable. Display sampling does not change the tests.'
)
METRIC_LABELS = {
    'degree': 'Degree', 'strength': 'Weighted<br>strength', 'pagerank': 'PageRank',
    'mean_neighbor_degree': 'Mean neighbor<br>degree', 'core_number': 'Core<br>number',
    'clustering': 'Clustering', 'betweenness_approx': 'Approximate<br>betweenness',
    'eigenvector': 'Eigenvector<br>centrality', 'bridging_coefficient': 'Bridging<br>coefficient',
    'bridging_centrality': 'Bridging<br>centrality',
    'participation': 'Community<br>participation', 'community_entropy': 'Community<br>entropy',
    'within_module_z': 'Within-module<br>z score',
}
COLORSCALE = [[0, '#2166AC'], [.5, '#FFFFFF'], [1, '#B35806']]


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
    mapped = nodes & selected_ids
    audit = data.select('input_accession', 'canonical_id', 'mapping_status', 'Gene Names', 'Reviewed').with_columns(
        pl.col('canonical_id').is_in(sorted(selected_ids)).alias('in_selected_cohort'),
        pl.col('canonical_id').is_in(sorted(nodes)).alias('in_network'))
    summary = {'network_nodes': len(nodes), 'network_edges': graph['edge_count'],
               'cohort_canonical_ids': len(selected_ids), 'cohort_in_network': len(mapped),
               'cohort_absent': len(selected_ids - nodes),
               'coverage_fraction': len(mapped) / len(selected_ids) if selected_ids else 0,
               'comparison_nodes': len(nodes - mapped)}
    return {'summary': summary, 'audit': audit}


def cross_network_analysis(first, second, cohort):
    common = set(first['nodes']) & set(second['nodes']) & cohort
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
    query = np.asarray([str(node) in cohort for node in graph['nodes']], dtype=bool)
    if not query.any():
        raise ValueError('No selected moonlighting proteins occur in this network.')
    return query


def _advanced_graph(graph):
    if graph['adj'].nnz // 2 > ADVANCED_EDGE_LIMIT:
        raise ValueError('This analysis supports up to 500,000 edges. Select a saved huMAP probability-threshold version for advanced topology.')
    return nx.from_scipy_sparse_array(graph['adj'])


def _comparison_table(frame, metrics):
    rows = []
    mask = frame['is_moonlighting'].to_numpy().astype(bool)
    for metric in metrics:
        values = frame[metric].to_numpy().astype(float)
        a, b = values[mask], values[~mask]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        p = float(mannwhitneyu(a, b, alternative='two-sided').pvalue) if len(a) and len(b) else np.nan
        rows.append({'metric': metric, 'query_n': len(a), 'background_n': len(b),
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


def centrality_analysis(graph, cohort, extra=False, seed=42, landmarks=32):
    """Grouped centralities; tests compare cohort with its network complement."""
    mask = _selection(graph, cohort)
    degree = np.asarray(graph['degree'], dtype=float)
    adjacency = graph['adj']
    weights = graph.get('weights', adjacency)
    pagerank, convergence = _pagerank(weights)
    neighbor_degree = np.divide(np.asarray(adjacency @ degree).ravel(), degree,
                                out=np.zeros(len(degree)), where=degree > 0)
    frame = pl.DataFrame({
        'protein': graph['nodes'], 'is_moonlighting': mask,
        'group': np.where(mask, 'Moonlighting', 'Background'), 'degree': degree,
        'strength': np.asarray(weights.sum(axis=1)).ravel(), 'pagerank': pagerank,
        'mean_neighbor_degree': neighbor_degree,
    })
    names = ['degree', 'strength', 'pagerank', 'mean_neighbor_degree']
    notes = {'pagerank': convergence, 'comparison': 'Two-sided Mann–Whitney U; Benjamini–Hochberg across displayed metrics; background excludes cohort proteins.'}
    if extra:
        network = _advanced_graph(graph)
        n = len(degree)
        core = nx.core_number(network)
        clustering = nx.clustering(network)
        sample = min(int(landmarks), n)
        betweenness = nx.betweenness_centrality(network, k=sample, seed=int(seed), normalized=True)
        frame = frame.with_columns(
            pl.Series('core_number', [core[i] for i in range(n)], dtype=pl.Int64),
            pl.Series('clustering', [clustering[i] for i in range(n)], dtype=pl.Float64),
            pl.Series('betweenness_approx', [betweenness[i] for i in range(n)], dtype=pl.Float64))
        if n > 2:
            _, eigenvectors = eigsh(adjacency.astype(float), k=1, which='LA', tol=1e-8, v0=np.random.default_rng(seed).random(n))
            eigenvector = np.abs(eigenvectors[:, 0])
        else:
            _, eigenvectors = np.linalg.eigh(adjacency.toarray().astype(float))
            eigenvector = np.abs(eigenvectors[:, -1])
        frame = frame.with_columns(pl.Series('eigenvector', eigenvector))
        inverse_degree = np.divide(1., degree, out=np.zeros(n), where=degree > 0)
        neighbor_inverse = np.asarray(adjacency @ inverse_degree).ravel()
        frame = frame.with_columns(pl.Series('bridging_coefficient', np.divide(inverse_degree, neighbor_inverse, out=np.zeros(n), where=neighbor_inverse > 0)))
        frame = frame.with_columns((pl.col('betweenness_approx') * pl.col('bridging_coefficient')).alias('bridging_centrality'))
        names += ['core_number', 'clustering', 'betweenness_approx', 'eigenvector', 'bridging_coefficient', 'bridging_centrality']
        notes['eigenvector'] = 'Unweighted leading adjacency eigenvector, L2-normalized; disconnected components with tied largest eigenvalues need not have a unique eigenvector.'
        notes['betweenness_landmarks'] = sample
        notes['betweenness_seed'] = int(seed)
    return {'metrics': frame, 'tests': _comparison_table(frame, names), 'notes': notes}


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
        'conductance': conductance, 'permutations': int(permutations), 'seed': int(seed),
        'null_model': 'Whole-network sampling matched in floor(log2 degree) bins' if degree_matched else 'Uniform whole-network protein sampling',
        'internal_edges_null_mean': float(null['internal_edges'].mean()),
        'largest_component_null_mean': float(null['largest_component'].mean()),
        'internal_edges_p_value': (1 + int((null['internal_edges'] >= edges).sum())) / (int(permutations) + 1),
        'largest_component_p_value': (1 + int((null['largest_component'] >= lcc).sum())) / (int(permutations) + 1),
    }
    adjusted = bh_adjust([summary['internal_edges_p_value'], summary['largest_component_p_value']])
    summary.update(internal_edges_fdr_bh=float(adjusted[0]), largest_component_fdr_bh=float(adjusted[1]))
    return {'summary': summary, 'null': null}


def wiring_analysis(graph, cohort, max_nodes=100, seed=42):
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
            'group': 'Moonlighting' if mask[node] else 'Background', 'degree': k,
            'triangles': int(triangles), 'clustering': triangles / wedges if wedges else 0.,
            'open_wedges': int(wedges - triangles), 'ego_components': int(components),
            'ego_component_entropy': entropy,
            'effective_size': k - 2 * triangles / k if k else 0.,
        })
    frame = pl.DataFrame(rows)
    columns = ['triangles', 'clustering', 'open_wedges', 'ego_components', 'ego_component_entropy', 'effective_size']
    return {'metrics': frame, 'tests': _comparison_table(frame, columns),
            'notes': {'query_sample': len(query), 'control_sample': len(controls), 'seed': int(seed),
                      'controls': 'Network complement, sampled without replacement in the same floor(log2 degree) bins as the sampled cohort.'}}


def community_analysis(graph, cohort, seed=42, resolution=1.0):
    """Full-network Louvain cartography plus cohort community enrichment."""
    mask = _selection(graph, cohort)
    network = _advanced_graph(graph)
    communities = nx.community.louvain_communities(network, seed=int(seed), resolution=float(resolution), weight=None)
    assignments = np.empty(len(mask), dtype=int)
    for label, members in enumerate(communities):
        assignments[list(members)] = label
    participation, entropy, within = np.zeros(len(mask)), np.zeros(len(mask)), np.zeros(len(mask))
    adjacency = graph['adj']
    for node in range(len(mask)):
        neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
        counts = np.bincount(assignments[neighbors], minlength=len(communities))
        degree = len(neighbors)
        fractions = counts[counts > 0] / degree if degree else np.array([])
        participation[node] = 1 - (fractions ** 2).sum() if degree else 0.
        entropy[node] = -(fractions * np.log(fractions)).sum()
        within[node] = counts[assignments[node]]
    z = np.zeros(len(mask))
    for community in communities:
        indices = np.asarray(list(community), dtype=int)
        std = within[indices].std()
        if std > 0:
            z[indices] = (within[indices] - within[indices].mean()) / std
    frame = pl.DataFrame({
        'protein': graph['nodes'], 'is_moonlighting': mask,
        'group': np.where(mask, 'Moonlighting', 'Background'), 'community': assignments,
        'participation': participation, 'community_entropy': entropy, 'within_module_z': z,
    })
    rows = []
    for community, members in enumerate(communities):
        members = list(members)
        count = int(mask[members].sum())
        rows.append({'community': community, 'network_proteins': len(members), 'moonlighting_proteins': count,
                     'expected': len(members) * mask.sum() / len(mask),
                     'p_value': float(hypergeom.sf(count - 1, len(mask), int(mask.sum()), len(members)))})
    enrichment = pl.DataFrame(rows)
    enrichment = enrichment.with_columns(pl.Series('fdr_bh', bh_adjust(enrichment['p_value'])))
    return {'metrics': frame, 'enrichment': enrichment.sort('fdr_bh', maintain_order=True),
            'tests': _comparison_table(frame, ['participation', 'community_entropy', 'within_module_z']),
            'summary': {'communities': len(communities), 'resolution': float(resolution), 'seed': int(seed),
                        'modularity': nx.community.modularity(network, communities, weight=None, resolution=float(resolution)),
                        'scope': 'Full selected interactome; unweighted hard-partition Louvain. Community indices have no intrinsic biological meaning.'}}


# ---------------------------------------------------------------------------
# Sparse Random Walk with Restart (RWR) & Node-Removal Robustness
# ---------------------------------------------------------------------------

def _cohort_indices(graph, cohort):
    indices = np.array(sorted({graph['index'][protein] for protein in cohort
                               if protein in graph['index']}), dtype=int)
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
    with Path(path).open() as stream:
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
    with gaf_path.open() as stream:
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
    query = {str(x) for x in cohort} & population
    if not population or not query:
        raise ValueError('The network universe and mapped cohort must both contain proteins')
    annotations = pl.scan_parquet(annotation_path).filter(pl.col('protein').is_in(sorted(population)))
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
                        seed=42, landmarks=32, include_union=False):
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
    present = {source: ids & nodes for source, ids in selected.items()}
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
        'missing_ids': [len(selected[source] - nodes) for source in groups],
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
                elif group != 'Moonlighting':
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
        for test in tests.filter(pl.col('metric') == metric).iter_rows(named=True):
            group = test.get('group', 'Moonlighting')
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
