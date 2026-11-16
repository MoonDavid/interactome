"""Additional moonlighting analyses using the canonical sparse interactome."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from functools import lru_cache
from itertools import combinations, permutations
from math import comb
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components, dijkstra, laplacian
from scipy.sparse.linalg import cg, eigs, eigsh
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu, rankdata, spearmanr

from moonlighting import (
    ADVANCED_EDGE_LIMIT, _advanced_graph, _comparison_table, _degree_bins,
    _pagerank, _remaining_lcc, _sample_matching, _selection, bh_adjust,
    go_enrichment, run_cached_clustering, significance_label,
)
from networks import SCORE_COLUMNS, _scored_frame, _threshold, _valid_score, matching_nodes


def _guard(graph):
    if graph['adj'].nnz // 2 > ADVANCED_EDGE_LIMIT:
        raise ValueError('Choose a saved network with at most 500,000 edges for this analysis.')


def _metrics(graph, mask, values, notes, indices=None, set_name='Moonlighting', **tables):
    indices = np.arange(len(mask)) if indices is None else np.asarray(indices)
    frame = pl.DataFrame({
        'protein': graph['nodes'][indices], 'is_moonlighting': mask[indices],
        'group': np.where(mask[indices], set_name, 'Background'), **values,
    })
    return {
        'metrics': frame, 'tests': _comparison_table(frame, list(values), group=set_name),
        'notes': {**notes, 'comparison': 'Two-sided Mann–Whitney U; BH across finite tests in this run. Exploratory because graph observations are dependent.'},
        **tables,
    }


def _matched_nodes(graph, mask, limit, seed):
    """Draw equal cohort/control counts per degree bin, reporting unavailable matches."""
    rng = np.random.default_rng(seed)
    query = np.flatnonzero(mask)
    if len(query) > limit:
        query = rng.choice(query, limit, replace=False)
    bins = _degree_bins(graph['degree'])
    matched = []
    for bucket in np.unique(bins[query]):
        candidates = query[bins[query] == bucket]
        controls = np.flatnonzero((bins == bucket) & ~mask)
        size = min(len(candidates), len(controls))
        matched.extend(rng.choice(candidates, size, replace=False))
        matched.extend(rng.choice(controls, size, replace=False))
    if not matched:
        raise ValueError('No cohort proteins can be matched to unlabelled proteins in their degree bins.')
    return np.sort(matched), {
        'requested_query_sample': len(query), 'matched_per_group': len(matched) // 2,
        'unmatched_query_sample': len(query) - len(matched) // 2,
        'sampling': 'Without replacement in floor(log2 degree) bins; bins with insufficient controls contribute only available pairs.',
        'seed': seed,
    }


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0)


def _neighbor_slice(matrix, node):
    return slice(matrix.indptr[node], matrix.indptr[node + 1])


def path_centralities(graph, cohort, seed=42, set_name='Moonlighting'):
    """Whole-network path metrics and sparse adjacency-based centralities."""
    network = _advanced_graph(graph)
    mask = _selection(graph, cohort)
    adj = graph['adj'].astype(float)
    degree = np.asarray(graph['degree'], dtype=float)
    n = len(degree)
    if n < 2:
        raise ValueError('Path analysis requires at least two proteins.')
    closeness, harmonic = np.zeros(n), np.zeros(n)
    for start in range(0, n, 64):
        indices = np.arange(start, min(start + 64, n))
        distances = dijkstra(adj, directed=True, unweighted=True, indices=indices)
        reachable = np.isfinite(distances) & (distances > 0)
        count = reachable.sum(axis=1)
        total = np.where(reachable, distances, 0).sum(axis=1)
        closeness[indices] = np.nan_to_num(_divide(count * count, total * max(1, n - 1)))
        harmonic[indices] = np.divide(1., distances, out=np.zeros_like(distances), where=reachable).sum(axis=1)
    # A maximum-row-sum bound keeps both linear systems away from singularity.
    beta = 0.5 / max(1., degree.max())
    identity = sparse.eye(n, format='csr')
    def solve(system, rhs):
        result, status = cg(system, rhs, atol=1e-10)
        if status:
            raise ValueError('The attenuated centrality solver did not converge.')
        return result

    onion = nx.onion_layers(network)
    landmark_count = min(64, n)
    scores = nx.betweenness_centrality(network, k=landmark_count, seed=seed)
    betweenness = np.array([scores[i] for i in range(n)])
    residual = betweenness.copy()
    bins = _degree_bins(degree)
    for bucket in np.unique(bins):
        members = bins == bucket
        residual[members] -= np.median(betweenness[members])
    return _metrics(graph, mask, {
        'onion_layer': [onion[i] for i in range(n)], 'closeness': closeness,
        'harmonic': harmonic, 'katz': solve(identity - beta * adj, np.ones(n)),
        'bonacich_negative': solve(identity + beta * adj, degree),
        'betweenness_per_degree': _divide(betweenness, degree),
        'betweenness_degree_bin_residual': residual,
        'articulation_point': np.isin(np.arange(n), list(nx.articulation_points(network))).astype(int),
    }, {'paths': 'Exact unweighted paths; closeness uses the reachable fraction and harmonic sums reciprocal distances.',
        'attenuation_magnitude': beta, 'betweenness_landmarks': landmark_count, 'seed': seed,
        'betweenness_residual': 'Subtracts the median within the full network floor(log2 degree) bin.'},
        set_name=set_name)


def spectral_roles(graph, cohort, seed=42, set_name='Moonlighting'):
    """Bounded sparse spectral summaries without dense interactome matrices."""
    _guard(graph)
    mask = _selection(graph, cohort)
    adj = graph['adj'].astype(float)
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    if n <= 64:
        eigenvalues, vectors = np.linalg.eigh(adj.toarray())
        mode = 'exact'
    else:
        eigenvalues, vectors = eigsh(adj, k=32, which='LA', v0=rng.random(n))
        mode = 'top 32 adjacency modes (truncated)'
    # Compute a scaled exponential to avoid overflow on dense components.
    shift = eigenvalues.max()
    subgraph = (vectors ** 2) @ np.exp(eigenvalues - shift)
    values = {
        'log_subgraph_centrality': np.log(subgraph, out=np.full(n, np.nan), where=subgraph > 0) + shift,
    }
    core = nx.k_core(_advanced_graph(graph), k=2)
    components = sorted((sorted(c) for c in nx.connected_components(core)), key=lambda c: (-len(c), c[0]))
    # Compute the Perron mode in each cyclic component, then keep the dominant
    # modes with equal component norms if their spectral radii tie.
    modes = []
    for members in components:
        sub = adj[members][:, members]
        size = len(members)
        block = sparse.bmat([[None, sparse.diags(np.diff(sub.indptr).astype(float) - 1)],
                             [-sparse.eye(size), sub]], format='csr')
        # For a simple cycle the positive non-backtracking centrality is uniform.
        if np.all(np.diff(sub.indptr) == 2):
            modes.append((1., members, np.full(size, 1 / np.sqrt(size))))
            continue
        ev, evectors = eigs(block, k=1, which='LR', v0=rng.random(2 * size), tol=1e-7)
        mode_vector = np.abs(evectors[size:, 0])
        modes.append((ev[0].real, members, mode_vector / np.linalg.norm(mode_vector)))
    nb = np.full(n, np.nan)
    if modes:
        radius = max(item[0] for item in modes)
        nb[:] = 0
        for value, members, vector in modes:
            if np.isclose(value, radius):
                nb[members] = vector
        nb /= np.linalg.norm(nb)
    values['nonbacktracking_core_centrality'] = nb
    leading = vectors[:, np.isclose(eigenvalues, shift)]
    eigenvector = np.sqrt((leading ** 2).sum(axis=1))
    values['nonbacktracking_minus_eigenvector_rank'] = (rankdata(nb) - rankdata(np.round(eigenvector, 12))) / n

    count, labels = connected_components(adj, directed=False)
    members = np.flatnonzero(labels == np.bincount(labels, minlength=count).argmax())
    fiedler = np.full(n, np.nan)
    roughness = np.full(n, np.nan)
    eigengap = None
    if len(members) >= 2:
        sub = adj[members][:, members]
        normalized_laplacian = laplacian(sub, normed=True).tocsr()
        if len(members) <= 4:
            spectrum, embedding = np.linalg.eigh(normalized_laplacian.toarray())
        else:
            spectrum, embedding = eigsh(normalized_laplacian, k=3, which='SM', v0=rng.random(len(members)))
        order = np.argsort(spectrum)
        spectrum, embedding = spectrum[order], embedding[:, order]
        eigengap = float(spectrum[1])
        # An individual Fiedler axis is not identifiable for a repeated eigenvalue.
        if len(spectrum) == 2 or not np.isclose(spectrum[1], spectrum[2]):
            axis = embedding[:, 1]
            axis *= np.sign(axis[np.argmax(np.abs(axis))])
            fiedler[members] = axis
            degrees = np.diff(sub.indptr)
            roughness[members] = (sub @ (axis ** 2) - 2 * axis * (sub @ axis)) / degrees + axis ** 2
    values.update(fiedler_coordinate=fiedler, fiedler_local_roughness=roughness)
    return _metrics(graph, mask, values, {
        'subgraph_centrality': mode + '; log diagonal of the adjacency exponential.',
        'log_estrada_index_exact_or_truncated': float(np.log(np.exp(eigenvalues - shift).sum()) + shift),
        'nonbacktracking': 'Ihara–Bass Perron modes on the 2-core; zero outside the dominant cyclic cores; unavailable on forests.',
        'fiedler': 'Largest component only, normalized Laplacian; unavailable for tied Fiedler eigenvalues.',
        'algebraic_connectivity_lcc': eigengap, 'lcc_nodes': len(members), 'seed': seed,
    }, set_name=set_name)


def _hodge_participation(network):
    """Diagonal harmonic projector aggregated over incident edges."""
    n, m = len(network), network.number_of_edges()
    if m > 400:
        return np.full(n, np.nan), None
    edges = sorted(tuple(sorted(edge)) for edge in network.edges)
    index = {edge: i for i, edge in enumerate(edges)}
    boundary = np.zeros((n, m))
    for i, (u, v) in enumerate(edges):
        boundary[u, i], boundary[v, i] = -1, 1
    upper = np.zeros((m, m))
    for u in network:
        for v, w in combinations(sorted(node for node in network[u] if node > u), 2):
            if network.has_edge(v, w):
                columns = [index[u, v], index[u, w], index[v, w]]
                upper[np.ix_(columns, columns)] += [[1, -1, 1], [-1, 1, -1], [1, -1, 1]]
    eigenvalues, modes = np.linalg.eigh(boundary.T @ boundary + upper)
    zero = np.abs(eigenvalues) < 1e-8
    return np.abs(boundary) @ (modes[:, zero] ** 2).sum(axis=1), int(zero.sum())


def electrical_and_walk_roles(graph, cohort, protein=None, set_name='Moonlighting'):
    """Exact electrical, heat and quantum metrics on one bounded component."""
    _guard(graph)
    mask = _selection(graph, cohort)
    adj = graph['adj'].astype(float)
    _, labels = connected_components(adj, directed=False)
    if protein is not None and protein not in graph['index']:
        raise ValueError('Choose a protein present in the selected network.')
    component = np.bincount(labels).argmax() if protein is None else labels[graph['index'][protein]]
    nodes = np.flatnonzero(labels == component)
    if len(nodes) > 350:
        raise ValueError(f'This component has {len(nodes):,} proteins. Choose a protein in a component of at most 350, or a stricter network version.')
    if len(nodes) < 2:
        raise ValueError('Choose a connected component with at least two proteins.')
    sub = adj[nodes][:, nodes]
    network = nx.from_scipy_sparse_array(sub)
    eigenvalues, vectors = np.linalg.eigh(laplacian(sub).toarray())
    squares = vectors ** 2
    inverse = (vectors[:, 1:] / eigenvalues[1:]) @ vectors[:, 1:].T
    diag = inverse.diagonal()
    resistance = np.maximum(0, diag[:, None] + diag - 2 * inverse)
    rows, cols = sub.nonzero()
    curvature = 1 - 0.5 * np.bincount(rows, weights=resistance[rows, cols], minlength=len(nodes))
    normalized = np.linalg.eigvalsh(laplacian(sub, normed=True).toarray())
    kemeny = np.sum(1 / normalized[1:])
    # Exact deletion criticality is intentionally bounded more tightly than the one-time eigensolve.
    criticality = np.full(len(nodes), np.nan)
    if len(nodes) <= 80:
        cut_vertices = set(nx.articulation_points(network))
        for i in range(len(nodes)):
            if i not in cut_vertices:
                keep = np.arange(len(nodes)) != i
                spectrum = np.linalg.eigvalsh(laplacian(sub[keep][:, keep], normed=True).toarray())
                criticality[i] = np.sum(1 / spectrum[1:]) - kemeny
    times = np.geomspace(0.1, 10, 9)
    heat = squares @ np.exp(-eigenvalues[:, None] * times)
    quantum = np.abs(squares @ np.exp(-1j * eigenvalues[:, None] * times)) ** 2
    wavelet = squares @ ((eigenvalues[:, None] * times * np.exp(-eigenvalues[:, None] * times)) ** 2)
    hodge, harmonic_dimension = _hodge_participation(network)
    current = nx.current_flow_betweenness_centrality(network, solver='full') if len(nodes) >= 3 else dict.fromkeys(network, 0.)
    return _metrics(graph, mask, {
        'current_flow_betweenness': [current[i] for i in range(len(nodes))],
        'information_centrality': 1 / resistance.sum(axis=1),
        'resistance_curvature': curvature, 'kemeny_deletion_delta': criticality,
        'heat_return_mean': heat.mean(axis=1), 'quantum_return_mean': quantum.mean(axis=1),
        'heat_wavelet_peak_energy': wavelet.max(axis=1), 'hodge_harmonic_participation': hodge,
    }, {'population': 'Selected connected component only; unweighted edges. Tests require finite scores in both groups.',
        'kemeny': float(kemeny), 'kemeny_deletion': 'Only components of up to 80 nodes; unavailable for deletions that disconnect the component.',
        'scales': times.tolist(), 'hodge_harmonic_dimension': harmonic_dimension,
        'hodge': 'Sum of diagonal harmonic edge projector weights incident to each protein; 2-skeleton, at most 400 edges.'}, indices=nodes,
        set_name=set_name,
        edges=pl.DataFrame({'a': graph['nodes'][nodes[rows[rows < cols]]],
                            'b': graph['nodes'][nodes[cols[rows < cols]]],
                            'effective_resistance': resistance[rows[rows < cols], cols[rows < cols]],
                            'resistance_curvature': (curvature[rows] / np.diff(sub.indptr)[rows] +
                                                     curvature[cols] / np.diff(sub.indptr)[cols])[rows < cols]}))

def _edge_mask(neighbors, members):
    return sum(1 << i for i, (u, v) in enumerate(combinations(members, 2)) if v in neighbors[u])


@lru_cache(maxsize=1)
def _five_node_signatures():
    """Rooted isomorphism classes for all labelled graphs on five vertices."""
    pairs = list(combinations(range(5), 2))
    transforms = [[pairs.index((min(order[u], order[v]), max(order[u], order[v]))) for u, v in pairs]
                  for order in ((0, *p) for p in permutations(range(1, 5)))]
    return tuple(min(sum(((mask >> edge) & 1) << i for i, edge in enumerate(order))
                     for order in transforms) for mask in range(1 << len(pairs)))


def _four_node_orbit(edges, degree, root):
    if edges == 3:
        return (6 if root == 1 else 7) if degree == [1, 1, 1, 3] else (4 if root == 1 else 5)
    if edges == 4:
        return 8 if degree == [2, 2, 2, 2] else {1: 9, 2: 10, 3: 11}[root]
    if edges == 5:
        return 12 if root == 2 else 13
    return 14


def _rooted_graphlets(neighbors, node, order, budget=25_000):
    """Enumerate connected induced subsets through a root, once per size."""
    frontier = {(node,)}
    counts = Counter()
    completed = 1
    for size in range(2, order + 1):
        expanded = set()
        for members in frontier:
            for vertex in set().union(*(neighbors[u] for u in members)).difference(members):
                expanded.add((node, *sorted((*members[1:], vertex))))
                if len(expanded) > budget:
                    return counts, completed
        for members in expanded:
            if size == 2:
                counts['orbit_0'] += 1
                continue
            if size == 5:
                signature = _five_node_signatures()[_edge_mask(neighbors, members)]
                counts[f'rooted_5_{signature:03x}'] += 1
                continue
            degree = [sum(v in neighbors[u] for v in members) for u in members]
            edges = sum(degree) // 2
            if size == 3:
                orbit = 3 if edges == 3 else degree[0]
            else:
                orbit = _four_node_orbit(edges, sorted(degree), degree[0])
            counts[f'orbit_{orbit}'] += 1
        completed = size
        frontier = expanded
    return counts, completed


@lru_cache(maxsize=1)
def _connected_five_node_orbits():
    signatures = _five_node_signatures()
    result = set()
    pairs = list(combinations(range(5), 2))
    for mask in set(signatures):
        graph = nx.Graph()
        graph.add_nodes_from(range(5))
        graph.add_edges_from(edge for i, edge in enumerate(pairs) if mask & (1 << i))
        if nx.is_connected(graph):
            result.add(mask)
    return sorted(result)


def graphlet_profiles(graph, cohort, limit=25, seed=42, order=4, set_name='Moonlighting'):
    _guard(graph)
    if order not in (4, 5):
        raise ValueError('Choose graphlet order four or five.')
    mask = _selection(graph, cohort)
    indices, notes = _matched_nodes(graph, mask, limit, seed)
    adj = graph['adj']
    neighbors = [set(map(int, adj.indices[_neighbor_slice(adj, i)])) for i in range(len(mask))]
    profiles = [_rooted_graphlets(neighbors, int(i), order) for i in indices]
    bowtie = _five_node_signatures()[_edge_mask([{1, 2, 3, 4}, {0, 2}, {0, 1}, {0, 4}, {0, 3}], tuple(range(5)))] if order == 5 else None
    columns = [(f'orbit_{i}', 2 if i == 0 else 3 if i < 4 else 4) for i in range(15)]
    if order == 5:
        columns.extend((f'rooted_5_{i:03x}' if i != bowtie else 'rooted_5_bowtie_center', 5)
                       for i in _connected_five_node_orbits())
    values = {}
    for name, size in columns:
        key = f'rooted_5_{bowtie:03x}' if name == 'rooted_5_bowtie_center' else name
        values[name] = [float(counts[key]) if size <= complete else np.nan for counts, complete in profiles]
    completed_orders = np.array([size for _, size in profiles])
    extras = {}
    for projection_order in range(order, 2, -1):
        complete = completed_orders >= projection_order
        if complete.sum() < 2:
            continue
        projection_columns = [values[name] for name, size in columns if size <= projection_order]
        matrix = np.log1p(np.column_stack(projection_columns)[complete])
        centered = matrix - matrix.mean(axis=0)
        _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
        scores = centered @ vectors[:2].T
        extras['projection'] = pl.DataFrame({
            'protein': graph['nodes'][indices[complete]], 'group': np.where(mask[indices[complete]], set_name, 'Background'),
            'PC1': scores[:, 0], 'PC2': scores[:, 1],
        })
        notes['projection_order'] = projection_order
        notes['pca_variance_fraction'] = (singular[:2] ** 2 / (singular @ singular)).tolist() if singular.any() else None
        break
    return _metrics(graph, mask, values, {
        **notes, 'maximum_order': order, 'complete_profiles': int((completed_orders == order).sum()),
        'budget': '25,000 connected induced subsets per root and order; uncompleted orders and larger orders are missing.',
        'orbit_labels': 'Standard 0–14 for 2–4 vertices. Five-vertex rooted classes use canonical hexadecimal edge masks with root first, and a named bowtie center.',
        'pca': 'Centered log1p counts for the highest order of at least three with two or more complete profiles; descriptive.',
    }, indices=indices, set_name=set_name, **extras)

def cohort_separation(graph, left, right, permutations_count=49, seed=42):
    """Closest-member network separation on the largest connected component."""
    _guard(graph)
    adjacency = graph['adj'].astype(float)
    _, labels = connected_components(adjacency, directed=False)
    nodes = np.flatnonzero(labels == np.bincount(labels).argmax())
    adj = adjacency[nodes][:, nodes]
    a = np.isin(graph['nodes'][nodes], sorted(left))
    b = np.isin(graph['nodes'][nodes], sorted(right))
    if not 0 <= permutations_count <= 199:
        raise ValueError('Use between 0 and 199 separation permutations.')
    if min(a.sum(), b.sum()) < 2:
        raise ValueError('Each source set needs at least two proteins in the largest connected component.')
    membership = a.astype(np.uint8) + 2 * b.astype(np.uint8)
    rows, cols = adj.nonzero()

    def distances(flags):
        first, second = [np.flatnonzero(flags & bit) for bit in (1, 2)]
        # Multi-source minima handle the cross-set case, including shared nodes.
        def nearest(indices):
            distance, _, owner = dijkstra(
                adj, directed=True, unweighted=True, indices=indices,
                min_only=True, return_predecessors=True)
            # The nearest other source is reached through a boundary of
            # the source's Voronoi cell, even when nearest-source ties occur.
            different = owner[rows] != owner[cols]
            u, v = rows[different], cols[different]
            closest = np.full(len(nodes), np.inf)
            np.minimum.at(closest, owner[u], distance[u] + 1 + distance[v])
            return distance, float(closest[indices].mean())

        from_a, aa = nearest(first)
        from_b, bb = nearest(second)
        ab = (from_a[second].sum() + from_b[first].sum()) / (len(first) + len(second))
        return aa, bb, float(ab), float(ab - (aa + bb) / 2)

    observed = distances(membership)
    rng = np.random.default_rng(seed)
    bins = _degree_bins(graph['degree'][nodes])
    pools = [np.flatnonzero(bins == bucket) for bucket in np.unique(bins)]
    null = []
    for _ in range(permutations_count):
        shuffled = membership.copy()
        for pool in pools:
            shuffled[pool] = rng.permutation(membership[pool])
        null.append(distances(shuffled)[-1])
    return {'summary': {
        'lcc_nodes': len(nodes), 'network_nodes': len(labels), 'source_A_in_lcc': int(a.sum()),
        'source_B_in_lcc': int(b.sum()), 'overlap_in_lcc': int((a & b).sum()),
        'within_A': observed[0], 'within_B': observed[1], 'between': observed[2], 'separation': observed[3],
        'empirical_p_closer': (1 + sum(v <= observed[3] for v in null)) / (len(null) + 1) if null else None,
        'null': 'Joint membership shuffled within full-network floor(log2 degree) bins on the largest component; preserves overlap and sizes.',
        'distance': 'Symmetric nearest-member distances; within-set excludes self; cross-set shared accessions contribute zero.',
        'seed': seed,
    }, 'null': pl.DataFrame({'separation': null}, schema={'separation': pl.Float64})}


def removal_efficiency(graph, cohort, permutations_count=19, seed=42, landmarks=24):
    """Batch removal curves with matched controls and sampled global efficiency."""
    _guard(graph)
    mask = _selection(graph, cohort)
    if not 0 <= permutations_count <= 99 or not 1 <= landmarks <= 64:
        raise ValueError('Use 0–99 removal permutations and 1–64 efficiency landmarks.')
    adj = graph['adj'].astype(float)
    n = len(mask)
    rng = np.random.default_rng(seed)
    query = rng.permutation(np.flatnonzero(mask))
    ordering = rng.permutation(n)
    removed_counts = sorted({0, len(query), *(int(np.ceil(len(query) * part)) for part in (0.25, .5, .75))})

    def efficiency(removed):
        keep = np.ones(n, dtype=bool)
        keep[removed] = False
        size = keep.sum()
        if size < 2:
            return np.nan
        remap = np.cumsum(keep) - 1
        sources = remap[ordering[keep[ordering]][:landmarks]]
        distances = dijkstra(adj[keep][:, keep], directed=True, unweighted=True, indices=sources)
        inverses = np.divide(1., distances, out=np.zeros_like(distances), where=distances > 0)
        return float(inverses.sum(axis=1).mean() / (size - 1))

    observed, null = [], []
    for count in removed_counts:
        removed = query[:count]
        lcc = _remaining_lcc(adj, removed)
        eff = efficiency(removed)
        lcc_null, eff_null = [], []
        for trial in range(permutations_count if count else 0):
            matched = _sample_matching(graph['degree'], removed, rng)
            lcc_null.append(_remaining_lcc(adj, matched))
            eff_null.append(efficiency(matched))
            null.append((count, trial + 1, lcc_null[-1] / n, eff_null[-1]))
        eligible = [value for value in eff_null if np.isfinite(value)]
        observed.append((count, lcc / n, eff,
                         (1 + sum(v / n <= lcc / n for v in lcc_null)) / (len(lcc_null) + 1) if lcc_null else None,
                         (1 + sum(v <= eff for v in eligible)) / (len(eligible) + 1) if eligible and np.isfinite(eff) else None))
    frame = pl.DataFrame(observed, schema={
        'removed': pl.Int64, 'lcc_fraction_original': pl.Float64, 'global_efficiency_estimate': pl.Float64,
        'lcc_p_lower': pl.Float64, 'efficiency_p_lower': pl.Float64,
    }, orient='row')
    adjusted = bh_adjust(np.concatenate([frame['lcc_p_lower'].to_numpy(), frame['efficiency_p_lower'].to_numpy()]))
    return {'table': frame.with_columns(pl.Series('lcc_fdr_bh', adjusted[:len(frame)]),
                                       pl.Series('efficiency_fdr_bh', adjusted[len(frame):])),
            'null': pl.DataFrame(null, schema={'removed': pl.Int64, 'permutation': pl.Int64,
                                             'lcc_fraction_original': pl.Float64, 'global_efficiency_estimate': pl.Float64}, orient='row'),
            'summary': {'landmarks': landmarks, 'seed': seed, 'permutations': permutations_count,
                        'efficiency': 'Mean reciprocal path distance to all other surviving nodes from uniformly ordered surviving landmarks; disconnected pairs contribute zero.',
                        'null': 'Same size and floor(log2 degree) histogram at each batch size; all network nodes eligible.',
                        'correction': 'BH across all tested batch sizes and both outcomes.'}}


def module_brokerage(graph, cohort, seed=42, set_name='Moonlighting'):
    """Guimerà–Amaral role census and unordered open-wedge brokerage."""
    _guard(graph)
    mask = _selection(graph, cohort)
    network = _advanced_graph(graph)
    frame = run_cached_clustering(graph, algorithm='louvain', seed=seed)['node_frame']
    module = frame['community'].to_numpy()
    z = frame['within_module_z'].to_numpy()
    part = frame['participation'].to_numpy()
    roles = np.where(z < 2.5, np.select([part <= .05, part <= .62, part <= .80], ['R1', 'R2', 'R3'], default='R4'),
                     np.select([part <= .30, part <= .75], ['R5', 'R6'], default='R7'))
    roles_frame = frame.select('protein').with_columns(
        pl.Series('role', roles), pl.Series('group', np.where(mask, set_name, 'Background')))
    role_counts = roles_frame.group_by('role', 'group').len().sort('role', 'group')
    indices, notes = _matched_nodes(graph, mask, 100, seed)
    brokerage = np.full((len(indices), 4), np.nan)
    for i, v in enumerate(indices):
        neighbors = list(network[v])
        if len(neighbors) > 200:
            continue
        brokerage[i] = 0
        center = module[v]
        for u, w in combinations(neighbors, 2):
            if network.has_edge(u, w):
                continue
            a, b = module[u], module[w]
            category = 0 if a == b == center else 1 if center in (a, b) else 2 if a == b else 3
            brokerage[i, category] += 1
    return _metrics(graph, mask, {
        name: brokerage[:, i] for i, name in enumerate(
            ('coordinator', 'gatekeeper_or_representative', 'consultant', 'liaison'))
    }, {**notes, 'partition': 'Cached Louvain at resolution 1.0; role thresholds from Guimerà–Amaral.',
        'brokerage': 'Open wedges; undirected edges merge gatekeeper and representative. Missing for degree above 200.'},
        indices=indices, roles=roles_frame, role_counts=role_counts, set_name=set_name)


def _ollivier_curvature(adj, u, v):
    """Lazy random-walk transport distance for one unweighted edge."""
    left = np.r_[u, adj.indices[_neighbor_slice(adj, u)]]
    right = np.r_[v, adj.indices[_neighbor_slice(adj, v)]]
    a, b = len(left), len(right)
    distance = dijkstra(adj, directed=True, unweighted=True, indices=left, limit=3)[:, right]
    margins = np.r_[.5, np.full(a - 1, .5 / (a - 1)), .5, np.full(b - 1, .5 / (b - 1))]
    positions = np.arange(a * b)
    constraints = sparse.coo_matrix(
        (np.ones(2 * a * b), (np.r_[np.repeat(np.arange(a), b), a + np.tile(np.arange(b), a)],
                              np.tile(positions, 2))), shape=(a + b, a * b)).tocsr()
    result = linprog(distance.ravel(), A_eq=constraints, b_eq=margins, method='highs')
    return 1 - result.fun if result.success else np.nan


def edge_geometry(graph, cohort, seed=42, limit=25, ollivier=False, set_name='Moonlighting'):
    """Sampled node summaries of incident edge geometry and ranked L3 nonedges."""
    network = _advanced_graph(graph)
    mask = _selection(graph, cohort)
    indices, notes = _matched_nodes(graph, mask, limit, seed)
    adj = graph['adj'].astype(float)
    degree = np.asarray(graph['degree'], dtype=float)
    normalized = sparse.diags(np.nan_to_num(_divide(1, np.sqrt(degree))))
    middle = normalized @ adj @ normalized
    selected = nx.edge_betweenness_centrality(network, k=min(32, len(mask)), seed=seed)
    scores, predictions, curvature = [], [], []
    rng = np.random.default_rng(seed)
    measured_edges = {}
    for i in indices:
        neighbors = set(network[i])
        shared = np.array([len(neighbors.intersection(network[j])) for j in sorted(neighbors)])
        nodes = np.array(sorted(neighbors), dtype=int)
        edge_bc = [selected.get((i, j), selected.get((j, i))) for j in nodes]
        scores.append([shared.mean(), (shared / (degree[i] + degree[nodes] - shared)).mean(), np.mean(edge_bc)]
                      if len(nodes) else [np.nan] * 3)
        if ollivier:
            candidates = nodes[(degree[nodes] <= 24)] if degree[i] <= 24 else np.array([], dtype=int)
            edges = rng.choice(candidates, size=min(4, len(candidates)), replace=False)
            estimates = []
            for j in edges:
                key = tuple(sorted((i, j)))
                if key not in measured_edges:
                    measured_edges[key] = _ollivier_curvature(adj, *key)
                if np.isfinite(measured_edges[key]):
                    estimates.append(measured_edges[key])
            curvature.append(np.mean(estimates) if estimates else np.nan)
        paths = (adj[[i], :] @ middle @ adj).toarray().ravel()
        paths[i] = 0
        paths[nodes] = 0
        for j in np.argsort(-paths, kind='stable')[:10]:
            if paths[j] > 0:
                predictions.append((graph['nodes'][i], graph['nodes'][j], float(paths[j])))
    values = dict(zip(('mean_edge_embeddedness', 'mean_edge_jaccard', 'mean_edge_betweenness'), np.asarray(scores).T))
    if ollivier:
        values['sampled_ollivier_curvature'] = curvature
    return _metrics(graph, mask, values,
                    {**notes, 'edge_betweenness_landmarks': min(32, len(mask)),
                     'ollivier': 'When enabled, exact transport with idleness 0.5 on up to four uniformly sampled incident edges with both endpoint degrees ≤24. Missing if no eligible edge.',
                     'l3': 'Ten highest-scoring nonedges per sampled protein. Sum over length-three paths weighted by 1/sqrt(degree_u × degree_v) for internal vertices.'},
                    indices=indices, set_name=set_name, l3_candidates=pl.DataFrame(predictions, schema={'protein': pl.String, 'candidate': pl.String, 'l3': pl.Float64}, orient='row'))


def cohort_filtration(root, network, cohort, custom=None, *, saved_version=None, minimum=0., parent=None, set_name='Moonlighting'):
    """Count incident endpoint retention and within-cohort ties by native score."""
    column = SCORE_COLUMNS[network]
    minimum = _threshold(minimum)
    custom = _threshold(custom) if custom is not None else None
    if custom is not None and custom < minimum:
        raise ValueError(f'The available saved version starts at {minimum:g}.')
    source = (_scored_frame(network, Path(root), parent=parent) if saved_version is None else
              pl.scan_parquet(saved_version).filter(_valid_score(column) & (pl.col(column) >= minimum)))
    frame = source.select('a', 'b', column).collect()
    ids = sorted(matching_nodes(pl.concat([frame['a'], frame['b'].rename('a')]).unique(), cohort))
    a, b = pl.col('a').is_in(ids), pl.col('b').is_in(ids)
    node_max = pl.concat([frame.select(pl.col(key).alias('protein'), column) for key in ('a', 'b')]).group_by('protein').agg(pl.col(column).max())
    series = [np.sort(data[column].to_numpy()) for data in
              (frame, node_max, node_max.filter(pl.col('protein').is_in(ids)), frame.filter(a & b), frame.filter(a ^ b))]
    thresholds = sorted({t for t in np.round(np.linspace(0, 1, 21), 2) if t >= minimum} |
                        {minimum} | ({custom} if custom is not None else set()))
    table = pl.DataFrame({'threshold': thresholds, **{
        name: len(scores) - np.searchsorted(scores, thresholds, side='left')
        for name, scores in zip(('network_edges', 'network_nodes', 'cohort_nodes', 'cohort_internal_edges', 'cohort_boundary_edges'), series)
    }})
    return {'table': table, 'summary': {'native_score': column, 'selected_cohort_ids': len(cohort),
                                      'protein_set': set_name,
                                      'minimum_available_score': minimum,
                                      'threshold': 'Inclusive; unscored edges excluded, and node counts include retained endpoints only.'}}


def multiplex_roles(first, second, cohort, coupling=.25, set_name='Moonlighting'):
    """Two-layer unweighted supra-graph with interlayer ties only for shared accessions."""
    for graph in (first, second):
        _guard(graph)
    if not np.isfinite(coupling) or coupling <= 0:
        raise ValueError('Interlayer coupling must be positive and finite.')
    a, b = first['nodes'], second['nodes']
    cohort = matching_nodes(np.concatenate([a, b]), cohort)
    nodes, ai, bi = np.intersect1d(a, b, return_indices=True)
    if not len(nodes):
        raise ValueError('The two networks need at least one shared protein.')
    coupling_matrix = sparse.csr_matrix((np.full(len(ai), coupling), (ai, bi)), shape=(len(a), len(b)))
    adj = sparse.bmat([[first['adj'], coupling_matrix], [coupling_matrix.T, second['adj']]], format='csr')
    scores, convergence = _pagerank(adj)
    seed_mask = np.isin(np.concatenate([a, b]), sorted(cohort))
    if not seed_mask.any():
        raise ValueError('No selected cohort proteins occur in either layer.')
    rwr, walk_convergence = _pagerank(adj, personalization=seed_mask.astype(float), alpha=.7)
    left, right = first['degree'][ai], second['degree'][bi]
    total = left + right
    participation = 2 * (1 - (left ** 2 + right ** 2) / total ** 2)
    all_nodes = np.union1d(a, b)
    combined = {'nodes': all_nodes}
    locations_a = np.searchsorted(all_nodes, a)
    locations_b = np.searchsorted(all_nodes, b)
    positions = np.searchsorted(all_nodes, nodes)
    layer_participation = np.zeros(len(all_nodes))
    layer_participation[positions] = participation
    supra_pagerank, supra_rwr = np.zeros(len(all_nodes)), np.zeros(len(all_nodes))
    for locations, part in [(locations_a, slice(0, len(a))), (locations_b, slice(len(a), None))]:
        supra_pagerank[locations] += scores[part]
        supra_rwr[locations] += rwr[part]
    return _metrics(combined, np.isin(all_nodes, sorted(cohort)), {
        'multiplex_participation': layer_participation, 'multiplex_pagerank': supra_pagerank,
    }, {'coupling': coupling, 'shared_proteins': len(nodes), 'pagerank': convergence, 'rwr': walk_convergence,
        'population': 'Union of endpoint proteins. Single-layer proteins have zero participation. Native affinity scores are not combined.',
        'rwr_seeding': 'Uniform over cohort layer copies, restart 0.3. RWR scores are reported separately because cohort labels are seeds.'},
        indices=None, set_name=set_name,
        ranked=pl.DataFrame({'protein': all_nodes, 'multiplex_rwr': supra_rwr, 'is_seed': np.isin(all_nodes, sorted(cohort))})
        .sort('multiplex_rwr', descending=True, maintain_order=True))

def _annotation_provenance(annotation_path):
    metadata = Path(annotation_path).with_suffix('.json')
    return json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else None


def go_neighborhood_geometry(graph, cohort, annotation_path, aspect='P', seed=42, limit=100, set_name='Moonlighting'):
    """Compare local GO annotation signals while keeping unknown annotations missing."""
    _guard(graph)
    mask = _selection(graph, cohort)
    indices, notes = _matched_nodes(graph, mask, limit, seed)
    annotations = (pl.scan_parquet(annotation_path)
                   .filter((pl.col('aspect') == aspect) & pl.col('protein').is_in(graph['nodes'].tolist()))
                   .select('protein', 'term').unique().collect())
    counts = annotations.group_by('term').len()
    terms = counts.filter((pl.col('len') >= 2) & (pl.col('len') < annotations['protein'].n_unique())).sort('term')
    proteins = pl.DataFrame({'protein': graph['nodes']}).with_row_index('i')
    coordinates = (annotations.join(proteins, on='protein').join(terms.with_row_index('j'), on='term'))
    if coordinates.is_empty():
        raise ValueError('No informative GO terms for this aspect and network.')
    signal = sparse.csr_matrix((np.ones(len(coordinates)),
                               (coordinates['i'].to_numpy(), coordinates['j'].to_numpy())),
                              shape=(len(mask), len(terms)))
    term_counts = np.diff(signal.indptr)
    adj = graph['adj']
    values = []
    for i in indices:
        neighbors = adj.indices[_neighbor_slice(adj, i)]
        known = neighbors[term_counts[neighbors] > 0]
        row = [len(known) / len(neighbors) if len(neighbors) else np.nan, np.nan, np.nan, np.nan]
        if len(known):
            prevalence = np.asarray(signal[known].mean(axis=0)).ravel()
            row[3] = np.sum(prevalence * (1 - prevalence))
            if term_counts[i]:
                shared = (signal[known] @ signal[[i], :].T).toarray().ravel()
                row[1] = np.mean(1 - shared / (term_counts[i] + term_counts[known] - shared))
                row[2] = np.mean(term_counts[i] + term_counts[known] - 2 * shared)
        values.append(row)
    return _metrics(graph, mask, dict(zip(('go_partner_coverage', 'go_partner_jaccard_distance',
                    'go_local_dirichlet_energy', 'go_neighbor_signal_variance'), np.asarray(values).T)),
                    {**notes, 'aspect': aspect, 'informative_terms': len(terms),
                     'annotation': 'Propagated binary GO signals; terms on at least two but fewer than all annotated proteins. Unknown annotation profiles are excluded.',
                     'annotation_provenance': _annotation_provenance(annotation_path),
                     'dirichlet': 'Mean squared Euclidean distance from the focal binary annotation vector to annotated neighbors.',
                     'variance': 'Sum of population variances across terms in annotated neighbors.'}, indices=indices, set_name=set_name)


def metadata_wiring(graph, cohort, selected):
    """Within-cohort metadata associations and distances to other OMIM-tagged proteins."""
    _guard(graph)
    mask = _selection(graph, cohort)
    adj = graph['adj'].astype(float)
    columns = ['Length', 'moonprot_f1_location', 'moonprot_f2_location', 'moonprot_si_quaternary',
               'moonprot_si_disorder', 'moonprot_si_disprot', 'moonprot_disease_omim',
               'moonlitdb_number_of_publications', 'moonlitdb_number_of_moonlighting_functions']
    network = pl.DataFrame({'canonical_id': graph['nodes'][mask], 'degree': graph['degree'][mask]})
    annotation = selected.select(
        'canonical_id', *[pl.col(c).cast(pl.String) if c in selected.columns else
                          pl.lit(None, dtype=pl.String).alias(c) for c in columns])
    annotation = annotation.with_columns(pl.col(columns).str.strip_chars().replace('', None))
    numeric = {'Length', 'moonlitdb_number_of_publications', 'moonlitdb_number_of_moonlighting_functions'}
    annotation = annotation.with_columns(pl.col(c).cast(pl.Float64, strict=False).fill_nan(None) for c in sorted(numeric))
    annotation = annotation.group_by('canonical_id').agg(
        *[pl.when(pl.col(c).drop_nulls().n_unique() == 1).then(pl.col(c).drop_nulls().first()).alias(c) for c in columns])
    frame = network.join(annotation, on='canonical_id', how='left', maintain_order='left')
    frame = frame.with_columns(
        (pl.col('moonprot_f1_location').str.to_lowercase() != pl.col('moonprot_f2_location').str.to_lowercase()).alias('location_strings_differ'))
    disease = frame.filter(pl.col('moonprot_disease_omim').str.contains(r'\b\d{6}\b'))
    anchors = np.array([graph['index'][p] for p in disease['canonical_id']], dtype=int)
    if len(anchors):
        targets = np.flatnonzero(mask)
        distances = dijkstra(adj, directed=True, unweighted=True, indices=anchors)[:, targets]
        distances[distances == 0] = np.inf
        minimum = distances.min(axis=0)
        frame = frame.with_columns(pl.Series('nearest_other_omim_tagged_protein',
                                            np.where(np.isfinite(minimum), minimum, np.nan)).fill_nan(None))
    tests = []
    for column in sorted(numeric | {'location_strings_differ'}):
        eligible = frame.select('degree', column).drop_nulls()
        rho = p = np.nan
        if len(eligible) >= 3 and min(eligible['degree'].n_unique(), eligible[column].n_unique()) > 1:
            rho, p = spearmanr(eligible['degree'].to_numpy(), eligible[column].to_numpy())
        tests.append((column, len(eligible), float(rho), float(p)))
    tests = pl.DataFrame(tests, schema={'annotation': pl.String, 'n': pl.Int64,
                                      'spearman_rho': pl.Float64, 'p_value': pl.Float64}, orient='row')
    return {'table': frame, 'tests': tests.with_columns(pl.Series('fdr_bh', bh_adjust(tests['p_value']))),
            'summary': {'omim_tagged_proteins': len(anchors),
                        'population': 'Selected annotated cohort only. Missing or conflicting metadata remain unknown.',
                        'locations': 'A lexical contrast of both reported function locations after case normalization.',
                        'omim': 'Nearest reachable other selected protein with a six-digit OMIM reference. Within-cohort descriptive distances; no disease-complete network labels.',
                        'tests': 'Spearman degree associations; BH across finite tests.'}}

def annotated_partner_partition(graph, selected, annotation_path):
    """Match named MoonLitDB function GO slims to propagated GO partner terms."""
    _guard(graph)
    columns = {
        'moonlitdb_goslims_canonical': 'canonical_function',
        'moonlitdb_goslims_moonlighting': 'moonlighting_function',
    }
    if any(column not in selected.columns for column in columns):
        raise ValueError('Partner partitioning requires both MoonLitDB GO-slim profile columns.')
    dtype = pl.List(pl.Struct({'label': pl.String, 'confidence': pl.Float64, 'count': pl.Int64}))
    labels = (
        selected.select('canonical_id', *columns)
        .filter(pl.col('canonical_id').is_in(graph['nodes'].tolist()))
        .with_columns(pl.col(column).replace('', '[]').str.json_decode(dtype).alias(label)
                      for column, label in columns.items())
        .unpivot(index='canonical_id', on=list(columns.values()), variable_name='function', value_name='go')
        .explode('go').unnest('go')
        .select(pl.col('canonical_id').alias('protein'), 'function',
                pl.col('label').str.strip_chars().str.to_lowercase().alias('term_name'))
        .filter(pl.col('term_name').is_not_null() & (pl.col('term_name') != ''))
        .unique()
    )
    eligible = labels.group_by('protein').agg(pl.col('function').n_unique().alias('profiles')).filter(pl.col('profiles') == 2)
    labels = labels.join(eligible.select('protein'), on='protein', how='semi')
    indices = np.array([graph['index'][p] for p in eligible['protein']], dtype=int)
    local, partners = graph['adj'][indices].nonzero()
    edges = pl.DataFrame({'protein': graph['nodes'][indices[local]], 'partner': graph['nodes'][partners]})
    go_terms = (
        pl.scan_parquet(annotation_path)
        .filter(pl.col('protein').is_in(edges['partner'].unique().to_list()))
        .select(pl.col('protein').alias('partner'), pl.col('name').str.strip_chars().str.to_lowercase().alias('term_name'))
        .unique().collect()
    )
    hits = edges.join(labels, on='protein').join(go_terms, on=['partner', 'term_name']).group_by('protein', 'partner').agg(
        (pl.col('function') == 'canonical_function').any().alias('canonical'),
        (pl.col('function') == 'moonlighting_function').any().alias('moonlighting'))
    edges = edges.join(hits, on=['protein', 'partner'], how='left').with_columns(
        pl.col('canonical', 'moonlighting').fill_null(False),
        pl.col('partner').is_in(go_terms['partner'].unique().implode()).alias('annotated'))
    return (edges.group_by('protein').agg(
        pl.len().alias('partners'),
        (~pl.col('annotated')).sum().alias('go_unannotated'),
        (pl.col('canonical') & ~pl.col('moonlighting')).sum().alias('canonical_only'),
        (~pl.col('canonical') & pl.col('moonlighting')).sum().alias('moonlighting_only'),
        (pl.col('canonical') & pl.col('moonlighting')).sum().alias('both'),
        (pl.col('annotated') & ~pl.col('canonical') & ~pl.col('moonlighting')).sum().alias('annotated_without_match'),
    ).sort('protein'))
