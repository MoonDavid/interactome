"""Py4Cytoscape bridge, subnetwork/bridge extraction, Cytoscape app clustering/hub workflows, and exports."""
from __future__ import annotations

from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import networkx as nx
import numpy as np
import polars as pl
import py4cytoscape as p4c
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import hypergeom, mannwhitneyu

from moonlighting import (
    CLUSTERING_CATALOG,
    SOURCES,
    bh_adjust,
    clustering_comparison_heatmap,
    community_analysis,
    compare_clustering_algorithms,
    significance_label,
)


def _detect_cytoscape_rest_port(default: int = 1235) -> int:
    """Read rest.port from ~/CytoscapeConfiguration/cytoscape3.props if present, defaulting to 1235."""
    props_path = Path.home() / 'CytoscapeConfiguration' / 'cytoscape3.props'
    if props_path.exists():
        try:
            for line in props_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                line = line.strip()
                if line.startswith('rest.port='):
                    return int(line.split('=', 1)[1].strip())
        except Exception:
            pass
    return default


DEFAULT_CYREST_PORT = _detect_cytoscape_rest_port(1235)
DEFAULT_BASE_URL = f'http://127.0.0.1:{DEFAULT_CYREST_PORT}/v1'

CLUSTERMAKER_ALGORITHMS: dict[str, dict[str, Any]] = {
    'leiden': {
        'label': 'Leiden Clustering (clusterMaker2)',
        'command': 'cluster leiden',
        'default_params': {'resolution_parameter': 1.0, 'n_iterations': 2},
        'column_candidates': ['__leidenCluster', '__leiden', 'leidenCluster'],
    },
    'mcl': {
        'label': 'MCL — Markov Cluster (clusterMaker2)',
        'command': 'cluster mcl',
        'default_params': {'inflation_parameter': 2.5},
        'column_candidates': ['__mclCluster', 'mclCluster'],
    },
    'louvain': {
        'label': 'Louvain Modularity (clusterMaker2)',
        'command': 'cluster louvain',
        'default_params': {'resolution': 1.0},
        'column_candidates': ['__louvainCluster', 'louvainCluster'],
    },
    'mcode': {
        'label': 'MCODE Complex Detection (clusterMaker2)',
        'command': 'cluster mcode',
        'default_params': {'haircut': 'true', 'fluff': 'false', 'scoreCutoff': 0.2},
        'column_candidates': ['__mcodeCluster', 'MCODE_Cluster', 'mcodeCluster'],
    },
    'glay': {
        'label': 'GLay Community Clustering (clusterMaker2)',
        'command': 'cluster glay',
        'default_params': {},
        'column_candidates': ['__glayCluster', 'glayCluster'],
    },
    'infomap': {
        'label': 'Infomap Flow Clustering (clusterMaker2)',
        'command': 'cluster infomap',
        'default_params': {'trials': 10},
        'column_candidates': ['__infomapCluster', 'infomapCluster'],
    },
    'ap': {
        'label': 'Affinity Propagation (clusterMaker2)',
        'command': 'cluster ap',
        'default_params': {'preference': -0.5},
        'column_candidates': ['__apCluster', 'apCluster'],
    },
    'fcm': {
        'label': 'Fuzzy C-Means Overlapping Clusters (clusterMaker2)',
        'command': 'cluster fcm',
        'default_params': {'findex': 1.5, 'membershipThreshold': 0.2},
        'column_candidates': ['__fcmCluster', 'fcmCluster'],
    },
}

CYTOHUBBA_METRICS: list[str] = [
    'MCC', 'DMNC', 'MNC', 'Degree', 'EPC', 'BottleNeck',
    'EcCentricity', 'Closeness', 'Radiality', 'Betweenness',
    'Stress', 'ClusteringCoefficient',
]

NETWORK_ANALYZER_METRICS: list[str] = [
    'BetweennessCentrality', 'ClosenessCentrality', 'ClusteringCoefficient',
    'NeighborhoodConnectivity', 'TopologicalCoefficient', 'Stress',
    'Radiality', 'AverageShortestPathLength', 'Eccentricity', 'Degree',
]

ROLE_COLORS: dict[str, str] = {
    'Moonlighting': '#0072B2',
    'Linker': '#D55E00',
    'Background': '#7F8C8D',
}


def build_base_url(host: str = '127.0.0.1', port: int = DEFAULT_CYREST_PORT) -> str:
    host = host.strip().removeprefix('http://').removeprefix('https://').rstrip('/')
    return f'http://{host}:{int(port)}/v1'


def check_cytoscape_connection(base_url: str = DEFAULT_BASE_URL, timeout: float = 2.5) -> dict[str, Any]:
    """Check Cytoscape Desktop connection via py4cytoscape and diagnose port conflicts."""
    root_url = base_url.rstrip('/')
    try:
        resp = requests.get(f'{root_url}/version', timeout=timeout)
        try:
            payload = resp.json()
        except ValueError:
            payload = {'raw': resp.text[:300]}
        if isinstance(payload, dict) and 'error' in payload and 'Unexpected endpoint' in str(payload.get('error', '')):
            return {
                'connected': False,
                'base_url': base_url,
                'port_conflict_detected': True,
                'conflicting_service': 'LM Studio / OpenAI-compatible local server',
                'error': str(payload['error']),
                'remediation': (
                    'Port 1234 is currently occupied by LM Studio (or another local LLM server). '
                    'Either stop the local server on port 1234 in LM Studio and restart Cytoscape, '
                    'or set Cytoscape rest.port=1235 (Edit → Preferences → Properties... → rest.port) '
                    'and connect on port 1235.'
                ),
                'installed_apps': [],
                'networks': [],
            }
        version_info = p4c.cytoscape_version_info(base_url=base_url)
        try:
            installed_apps = p4c.get_installed_apps(base_url=base_url)
        except Exception:
            installed_apps = []
        try:
            networks = p4c.get_network_list(base_url=base_url, get_suids=True)
        except Exception:
            networks = []
        return {
            'connected': True,
            'base_url': base_url,
            'port_conflict_detected': False,
            'version_info': version_info,
            'installed_apps': installed_apps,
            'networks': networks,
        }
    except Exception as exc:
        return {
            'connected': False,
            'base_url': base_url,
            'port_conflict_detected': False,
            'error': str(exc),
            'remediation': (
                'Ensure Cytoscape Desktop is running and CyREST is active on the selected port '
                f'({base_url}). Note: if LM Studio was running on port 1234 when Cytoscape started, '
                'CyREST cannot bind to 1234 until LM Studio frees the port or rest.port is changed.'
            ),
            'installed_apps': [],
            'networks': [],
        }


def _protein_metadata_frame(moon_df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate moonlighting metadata by canonical_id using Polars expressions."""
    mapped = moon_df.filter(pl.col('canonical_id').is_not_null() & (pl.col('canonical_id') != ''))
    if not len(mapped):
        return pl.DataFrame(schema={
            'protein': pl.String, 'gene_symbol': pl.String, 'source_sets': pl.String,
            'source_count': pl.Int64, 'f1_location': pl.String, 'f2_location': pl.String,
            'goslims_canonical': pl.String, 'goslims_moonlighting': pl.String,
            'publications': pl.Int64,
        })
    source_exprs = [pl.col(s).any().alias(s) for s in SOURCES if s in mapped.columns]
    extra_cols = {
        'Gene Names': 'gene_raw',
        'moonprot_f1_location': 'f1_location',
        'moonprot_f2_location': 'f2_location',
        'moonlitdb_goslims_canonical': 'goslims_canonical',
        'moonlitdb_goslims_moonlighting': 'goslims_moonlighting',
        'moonlitdb_number_of_publications': 'pubs_raw',
    }
    agg_exprs = list(source_exprs)
    for src_col, alias in extra_cols.items():
        if src_col in mapped.columns:
            agg_exprs.append(pl.col(src_col).cast(pl.String).filter(pl.col(src_col).cast(pl.String) != '').first().fill_null('').alias(alias))
        else:
            agg_exprs.append(pl.lit('').alias(alias))
    grouped = mapped.group_by('canonical_id', maintain_order=True).agg(agg_exprs).rename({'canonical_id': 'protein'})
    active_sources = [s for s in SOURCES if s in grouped.columns]
    grouped = grouped.with_columns(
        pl.col('gene_raw').str.split(' ').list.first().fill_null('').alias('gene_symbol'),
        pl.concat_str(
            [pl.when(pl.col(s)).then(pl.lit(s)).otherwise(pl.lit('')) for s in active_sources],
            separator='|',
        ).str.replace_all(r'\|+', '|').str.strip_chars('|').alias('source_sets'),
        pl.sum_horizontal([pl.col(s).cast(pl.Int64) for s in active_sources]).alias('source_count'),
        pl.col('pubs_raw').cast(pl.Float64, strict=False).fill_nan(0).fill_null(0).cast(pl.Int64).alias('publications'),
    )
    return grouped.select(
        'protein', 'gene_symbol', 'source_sets', 'source_count',
        'f1_location', 'f2_location', 'goslims_canonical', 'goslims_moonlighting', 'publications',
    )


def build_subnetwork_tables(
    graph: dict[str, Any],
    moon_df: pl.DataFrame,
    cohort: set[str],
    *,
    mode: str = 'cohort_linkers',
    seed_proteins: list[str] | None = None,
    hop_radius: int = 1,
    max_linkers: int = 30,
    max_nodes: int = 180,
    include_background_neighbors: bool = True,
) -> dict[str, Any]:
    """Extract a moonlighting/bridge subnetwork in Polars with exact curvature and wedge metrics."""
    nodes_arr = graph['nodes']
    node_to_idx = graph['index']
    adj = graph['adj']
    weights = graph['weights']
    degrees = graph['degree'].astype(np.int64)
    n_total = len(nodes_arr)

    cohort_in_net = sorted(p for p in cohort if p in node_to_idx)
    if not cohort_in_net:
        raise ValueError('No mapped cohort proteins are present in the selected network.')

    cohort_mask = np.zeros(n_total, dtype=bool)
    for p in cohort_in_net:
        cohort_mask[node_to_idx[p]] = True

    # 1-hop moonlighting neighbor counts for every protein in the graph
    moon_neighbors = np.asarray(adj @ cohort_mask.astype(np.int64)).ravel()
    total_moon = int(cohort_mask.sum())

    # Hypergeometric linker enrichment for non-cohort nodes with >= 2 moonlighting neighbors
    linker_pvals = np.full(n_total, np.nan, dtype=float)
    candidate_idx = np.flatnonzero((~cohort_mask) & (moon_neighbors >= 2) & (degrees > 0))
    if len(candidate_idx):
        linker_pvals[candidate_idx] = hypergeom.sf(
            moon_neighbors[candidate_idx] - 1,
            n_total,
            total_moon,
            degrees[candidate_idx],
        )
    linker_qvals = bh_adjust(linker_pvals)
    baseline_frac = total_moon / max(1, n_total)
    linker_fold = np.where(
        degrees > 0,
        (moon_neighbors / np.maximum(1, degrees)) / max(1e-12, baseline_frac),
        0.0,
    )

    # Local bridging coefficient beta_v = (1/k_v) / sum_{u in N(v)} (1/k_u)
    inv_deg = np.where(degrees > 0, 1.0 / np.maximum(1, degrees), 0.0)
    sum_inv_neighbor_deg = np.asarray(adj @ inv_deg).ravel()
    bridging_coeff = np.where(
        (degrees > 1) & (sum_inv_neighbor_deg > 0),
        inv_deg / sum_inv_neighbor_deg,
        0.0,
    )

    # Select subnetwork node indices according to mode
    ego_component_map: dict[int, int] = {}
    if mode == 'ego_bridge':
        seeds = [p for p in (seed_proteins or cohort_in_net[:3]) if p in node_to_idx]
        if not seeds:
            seeds = cohort_in_net[:3]
        seed_idx = [node_to_idx[p] for p in seeds]
        visited = set(seed_idx)
        frontier = set(seed_idx)
        for _ in range(max(1, min(2, int(hop_radius)))):
            next_frontier: set[int] = set()
            for u in frontier:
                nbrs = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
                next_frontier.update(int(v) for v in nbrs)
            visited.update(next_frontier)
            frontier = next_frontier
        # Compute ego-neighborhood connected components when removing the seed(s)
        neighbor_only = sorted(visited - set(seed_idx))
        if neighbor_only:
            sub_nbr = adj[neighbor_only, :][:, neighbor_only]
            _, comp_labels = connected_components(sub_nbr, directed=False)
            for idx_val, comp_id in zip(neighbor_only, comp_labels):
                ego_component_map[int(idx_val)] = int(comp_id) + 1
        for s in seed_idx:
            ego_component_map[int(s)] = 0
        ordered_candidates = seed_idx + sorted(
            visited - set(seed_idx),
            key=lambda i: (not cohort_mask[i], -int(moon_neighbors[i]), -float(bridging_coeff[i]), str(nodes_arr[i])),
        )
        selected_indices = sorted(ordered_candidates[:max_nodes])
    elif mode == 'top_bridges':
        cohort_indices = np.flatnonzero(cohort_mask)
        ranked_cohort = cohort_indices[np.argsort(-bridging_coeff[cohort_indices], kind='stable')]
        core_seeds = [int(i) for i in ranked_cohort[:min(len(ranked_cohort), max(5, max_nodes // 3))]]
        selected_set = set(core_seeds)
        if include_background_neighbors:
            # Add top linkers and immediate bridge partners up to max_nodes
            nbr_pool: set[int] = set()
            for u in core_seeds:
                nbrs = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
                nbr_pool.update(int(v) for v in nbrs)
            ranked_nbrs = sorted(
                nbr_pool - selected_set,
                key=lambda i: (not cohort_mask[i], -int(moon_neighbors[i]), -float(bridging_coeff[i]), str(nodes_arr[i])),
            )
            selected_set.update(ranked_nbrs[:max(0, max_nodes - len(selected_set))])
        selected_indices = sorted(selected_set)
    else:
        # Default: 'cohort_linkers'
        cohort_indices = [int(i) for i in np.flatnonzero(cohort_mask)]
        linker_order = sorted(
            [int(i) for i in candidate_idx],
            key=lambda i: (
                float(linker_qvals[i]) if np.isfinite(linker_qvals[i]) else 1.0,
                -int(moon_neighbors[i]),
                -float(bridging_coeff[i]),
                str(nodes_arr[i]),
            ),
        )[:max(0, int(max_linkers))]
        # Prioritize cohort nodes that have edges inside cohort + linkers
        pool = sorted(set(cohort_indices + linker_order))
        sub_adj_pool = adj[pool, :][:, pool]
        sub_deg_pool = np.diff(sub_adj_pool.indptr)
        connected_pool = [pool[k] for k in range(len(pool)) if sub_deg_pool[k] > 0]
        if not connected_pool:
            connected_pool = pool
        ranked_pool = sorted(
            connected_pool,
            key=lambda i: (
                0 if (not cohort_mask[i] and i in linker_order) else (1 if cohort_mask[i] else 2),
                -int(moon_neighbors[i]),
                -float(bridging_coeff[i]),
                str(nodes_arr[i]),
            ),
        )
        selected_indices = sorted(ranked_pool[:max_nodes])

    sub_adj = adj[selected_indices, :][:, selected_indices].tocsr()
    sub_weights = weights[selected_indices, :][:, selected_indices].tocsr()
    sub_degrees = np.diff(sub_adj.indptr).astype(np.int64)

    # Shared triangles per edge in the full graph and in the subnetwork
    coo = sub_adj.tocoo()
    upper_mask = coo.row < coo.col
    u_local = coo.row[upper_mask]
    v_local = coo.col[upper_mask]

    edge_records: list[dict[str, Any]] = []
    node_non_simmelian_counts = np.zeros(len(selected_indices), dtype=np.int64)
    node_triangles = np.zeros(len(selected_indices), dtype=np.int64)

    linker_set = {
        int(i) for i in candidate_idx
        if (np.isfinite(linker_qvals[i]) and linker_qvals[i] < 0.05) or moon_neighbors[i] >= 3
    }

    for ul, vl in zip(u_local, v_local):
        ug = selected_indices[int(ul)]
        vg = selected_indices[int(vl)]
        u_name = str(nodes_arr[ug])
        v_name = str(nodes_arr[vg])

        # Exact full-graph neighborhood intersection
        nbrs_u = adj.indices[adj.indptr[ug]:adj.indptr[ug + 1]]
        nbrs_v = adj.indices[adj.indptr[vg]:adj.indptr[vg + 1]]
        shared_full = int(len(np.intersect1d(nbrs_u, nbrs_v, assume_unique=True)))
        union_full = int(len(nbrs_u) + len(nbrs_v) - shared_full)
        jaccard_full = float(shared_full / union_full) if union_full > 0 else 0.0

        # Subnetwork shared triangles and Augmented Forman-Ricci curvature
        sub_nbrs_u = sub_adj.indices[sub_adj.indptr[ul]:sub_adj.indptr[ul + 1]]
        sub_nbrs_v = sub_adj.indices[sub_adj.indptr[vl]:sub_adj.indptr[vl + 1]]
        shared_sub = int(len(np.intersect1d(sub_nbrs_u, sub_nbrs_v, assume_unique=True)))
        node_triangles[ul] += shared_sub
        node_triangles[vl] += shared_sub
        if shared_full == 0:
            node_non_simmelian_counts[ul] += 1
            node_non_simmelian_counts[vl] += 1

        ku_sub = int(sub_degrees[ul])
        kv_sub = int(sub_degrees[vl])
        ku_full = int(degrees[ug])
        kv_full = int(degrees[vg])

        # Samal et al. (2018) Augmented Forman-Ricci curvature: 4 - k_u - k_v + 3 * t_uv
        frc_sub = float(4 - ku_sub - kv_sub + 3 * shared_sub)
        frc_full = float(4 - ku_full - kv_full + 3 * shared_full)

        # Edge class
        u_role = 'Moonlighting' if cohort_mask[ug] else ('Linker' if ug in linker_set or moon_neighbors[ug] >= 2 else 'Background')
        v_role = 'Moonlighting' if cohort_mask[vg] else ('Linker' if vg in linker_set or moon_neighbors[vg] >= 2 else 'Background')
        roles = sorted([u_role, v_role])
        if roles == ['Moonlighting', 'Moonlighting']:
            edge_class = 'M-M'
        elif 'Moonlighting' in roles and 'Linker' in roles:
            edge_class = 'M-Linker'
        elif 'Moonlighting' in roles:
            edge_class = 'M-B'
        else:
            edge_class = 'B-B'

        is_bridge = bool(shared_full == 0 and ku_sub >= 2 and kv_sub >= 2)
        w_val = float(sub_weights[ul, vl])

        edge_records.append({
            'source': u_name,
            'target': v_name,
            'interaction': 'pp',
            'name': f'{u_name} (pp) {v_name}',
            'score': w_val,
            'edge_class': edge_class,
            'shared_triangles': shared_full,
            'subnetwork_triangles': shared_sub,
            'jaccard_overlap': jaccard_full,
            'forman_ricci_sub': frc_sub,
            'forman_ricci_full': frc_full,
            'is_bridge_edge': is_bridge,
        })

    edges_df = pl.DataFrame(
        edge_records,
        schema={
            'source': pl.String,
            'target': pl.String,
            'interaction': pl.String,
            'name': pl.String,
            'score': pl.Float64,
            'edge_class': pl.String,
            'shared_triangles': pl.Int64,
            'subnetwork_triangles': pl.Int64,
            'jaccard_overlap': pl.Float64,
            'forman_ricci_sub': pl.Float64,
            'forman_ricci_full': pl.Float64,
            'is_bridge_edge': pl.Boolean,
        },
    )

    node_records: list[dict[str, Any]] = []
    for local_i, global_i in enumerate(selected_indices):
        protein = str(nodes_arr[global_i])
        k_sub = int(sub_degrees[local_i])
        k_full = int(degrees[global_i])
        tri_sub = int(node_triangles[local_i]) // 2
        wedges_sub = max(0, k_sub * (k_sub - 1) // 2 - tri_sub)
        clust_sub = float(2 * tri_sub / (k_sub * (k_sub - 1))) if k_sub > 1 else 0.0
        non_simm_frac = float(node_non_simmelian_counts[local_i] / k_sub) if k_sub > 0 else 0.0
        is_moon = bool(cohort_mask[global_i])
        if is_moon:
            role = 'Moonlighting'
        elif global_i in linker_set or int(moon_neighbors[global_i]) >= 2:
            role = 'Linker'
        else:
            role = 'Background'

        q_val = float(linker_qvals[global_i]) if np.isfinite(linker_qvals[global_i]) else None
        p_val = float(linker_pvals[global_i]) if np.isfinite(linker_pvals[global_i]) else None

        node_records.append({
            'id': protein,
            'name': protein,
            'protein': protein,
            'node_role': role,
            'is_moonlighting': is_moon,
            'full_degree': k_full,
            'subnetwork_degree': k_sub,
            'moonlighting_neighbors': int(moon_neighbors[global_i]),
            'linker_fold': float(linker_fold[global_i]),
            'linker_pvalue': p_val,
            'linker_qvalue': q_val,
            'linker_significance': significance_label(q_val) if q_val is not None else 'n/a',
            'bridging_coefficient': float(bridging_coeff[global_i]),
            'subnetwork_clustering': clust_sub,
            'subnetwork_triangles': tri_sub,
            'subnetwork_open_wedges': int(wedges_sub),
            'non_simmelian_fraction': non_simm_frac,
            'ego_component': int(ego_component_map.get(global_i, 0)),
        })

    base_nodes_df = pl.DataFrame(
        node_records,
        schema={
            'id': pl.String,
            'name': pl.String,
            'protein': pl.String,
            'node_role': pl.String,
            'is_moonlighting': pl.Boolean,
            'full_degree': pl.Int64,
            'subnetwork_degree': pl.Int64,
            'moonlighting_neighbors': pl.Int64,
            'linker_fold': pl.Float64,
            'linker_pvalue': pl.Float64,
            'linker_qvalue': pl.Float64,
            'linker_significance': pl.String,
            'bridging_coefficient': pl.Float64,
            'subnetwork_clustering': pl.Float64,
            'subnetwork_triangles': pl.Int64,
            'subnetwork_open_wedges': pl.Int64,
            'non_simmelian_fraction': pl.Float64,
            'ego_component': pl.Int64,
        },
    )

    meta_df = _protein_metadata_frame(moon_df)
    nodes_df = (
        base_nodes_df
        .join(meta_df, on='protein', how='left', maintain_order='left')
        .with_columns(
            pl.col('gene_symbol').fill_null(''),
            pl.col('source_sets').fill_null(''),
            pl.col('source_count').fill_null(0),
            pl.col('f1_location').fill_null(''),
            pl.col('f2_location').fill_null(''),
            pl.col('goslims_canonical').fill_null(''),
            pl.col('goslims_moonlighting').fill_null(''),
            pl.col('publications').fill_null(0),
        )
        .with_columns(
            pl.when(pl.col('gene_symbol') != '')
            .then(pl.col('gene_symbol'))
            .otherwise(pl.col('protein'))
            .alias('label')
        )
    )

    bridge_edges_count = int(edges_df['is_bridge_edge'].sum()) if len(edges_df) else 0
    neg_curvature_count = int((edges_df['forman_ricci_sub'] < 0).sum()) if len(edges_df) else 0
    summary = {
        'mode': mode,
        'subnetwork_nodes': nodes_df.height,
        'subnetwork_edges': edges_df.height,
        'moonlighting_nodes': int(nodes_df['is_moonlighting'].sum()),
        'linker_nodes': int((nodes_df['node_role'] == 'Linker').sum()),
        'background_nodes': int((nodes_df['node_role'] == 'Background').sum()),
        'bridge_edges_zero_triangles': bridge_edges_count,
        'negative_forman_ricci_edges': neg_curvature_count,
        'ego_components_bridged': int(nodes_df['ego_component'].max() or 0),
    }
    return {'nodes': nodes_df, 'edges': edges_df, 'summary': summary}


def build_multiplex_subnetwork_tables(
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    name_a: str,
    name_b: str,
    moon_df: pl.DataFrame,
    cohort: set[str],
    *,
    seed_proteins: list[str] | None = None,
    max_nodes: int = 120,
) -> dict[str, Any]:
    """Build a cross-network multiplex subnetwork comparing two canonical networks (e.g. HuRI vs BioPlex)."""
    common_cohort = sorted((set(graph_a['nodes']) & set(graph_b['nodes'])) & cohort)
    if not common_cohort:
        raise ValueError(f'No selected cohort proteins are shared between {name_a} and {name_b}.')

    seeds = [p for p in (seed_proteins or common_cohort[:15]) if p in graph_a['index'] and p in graph_b['index']]
    if not seeds:
        seeds = common_cohort[:15]

    node_pool: set[str] = set(seeds)
    turnover_map: dict[str, float] = {}
    for p in seeds:
        ia = graph_a['index'][p]
        ib = graph_b['index'][p]
        nbrs_a = {str(graph_a['nodes'][j]) for j in graph_a['adj'].indices[graph_a['adj'].indptr[ia]:graph_a['adj'].indptr[ia + 1]]}
        nbrs_b = {str(graph_b['nodes'][j]) for j in graph_b['adj'].indices[graph_b['adj'].indptr[ib]:graph_b['adj'].indptr[ib + 1]]}
        union_ab = nbrs_a | nbrs_b
        turnover_map[p] = 1.0 - (len(nbrs_a & nbrs_b) / len(union_ab)) if union_ab else 0.0
        # Prioritize shared partners and moonlighting partners
        prioritized = sorted(
            union_ab,
            key=lambda u: (not (u in nbrs_a and u in nbrs_b), u not in cohort, u),
        )
        for u in prioritized:
            if len(node_pool) >= max_nodes:
                break
            node_pool.add(u)

    selected_nodes = sorted(node_pool)
    idx_map = {u: i for i, u in enumerate(selected_nodes)}

    edges_a: dict[tuple[str, str], float] = {}
    edges_b: dict[tuple[str, str], float] = {}
    for g, target_dict in [(graph_a, edges_a), (graph_b, edges_b)]:
        present = [u for u in selected_nodes if u in g['index']]
        g_indices = [g['index'][u] for u in present]
        sub = g['weights'][g_indices, :][:, g_indices].tocoo()
        for r, c, val in zip(sub.row, sub.col, sub.data):
            if r < c:
                pair = (present[int(r)], present[int(c)])
                target_dict[pair] = float(val)

    all_pairs = sorted(set(edges_a) | set(edges_b))
    edge_rows: list[dict[str, Any]] = []
    for u, v in all_pairs:
        in_a = (u, v) in edges_a
        in_b = (u, v) in edges_b
        if in_a and in_b:
            layer = 'Shared'
        elif in_a:
            layer = f'{name_a}_only'
        else:
            layer = f'{name_b}_only'
        edge_rows.append({
            'source': u,
            'target': v,
            'interaction': layer,
            'name': f'{u} ({layer}) {v}',
            'score': float(max(edges_a.get((u, v), 0.0), edges_b.get((u, v), 0.0))),
            'edge_class': layer,
            'shared_triangles': 0,
            'subnetwork_triangles': 0,
            'jaccard_overlap': 1.0 if in_a and in_b else 0.0,
            'forman_ricci_sub': 0.0,
            'forman_ricci_full': 0.0,
            'is_bridge_edge': bool(in_a != in_b),
        })

    edges_df = pl.DataFrame(
        edge_rows,
        schema={
            'source': pl.String, 'target': pl.String, 'interaction': pl.String, 'name': pl.String,
            'score': pl.Float64, 'edge_class': pl.String, 'shared_triangles': pl.Int64,
            'subnetwork_triangles': pl.Int64, 'jaccard_overlap': pl.Float64,
            'forman_ricci_sub': pl.Float64, 'forman_ricci_full': pl.Float64, 'is_bridge_edge': pl.Boolean,
        },
    )

    deg_sub = np.zeros(len(selected_nodes), dtype=np.int64)
    for u, v in all_pairs:
        deg_sub[idx_map[u]] += 1
        deg_sub[idx_map[v]] += 1

    node_rows: list[dict[str, Any]] = []
    for i, u in enumerate(selected_nodes):
        da = int(graph_a['degree'][graph_a['index'][u]]) if u in graph_a['index'] else 0
        db = int(graph_b['degree'][graph_b['index'][u]]) if u in graph_b['index'] else 0
        is_moon = u in cohort
        node_rows.append({
            'id': u,
            'name': u,
            'protein': u,
            'node_role': 'Moonlighting' if is_moon else ('Linker' if (da > 0 and db > 0) else 'Background'),
            'is_moonlighting': is_moon,
            'full_degree': da + db,
            'subnetwork_degree': int(deg_sub[i]),
            'moonlighting_neighbors': 0,
            'linker_fold': float(turnover_map.get(u, 0.0)),
            'linker_pvalue': None,
            'linker_qvalue': None,
            'linker_significance': f'A:{da} / B:{db}',
            'bridging_coefficient': float(turnover_map.get(u, 0.0)),
            'subnetwork_clustering': 0.0,
            'subnetwork_triangles': 0,
            'subnetwork_open_wedges': 0,
            'non_simmelian_fraction': float(turnover_map.get(u, 0.0)),
            'ego_component': 1 if (da > 0 and db > 0) else (2 if da > 0 else 3),
        })

    base_nodes = pl.DataFrame(node_rows)
    meta_df = _protein_metadata_frame(moon_df)
    nodes_df = (
        base_nodes
        .join(meta_df, on='protein', how='left', maintain_order='left')
        .with_columns(
            pl.col('gene_symbol').fill_null(''),
            pl.col('source_sets').fill_null(''),
            pl.col('source_count').fill_null(0),
            pl.col('f1_location').fill_null(''),
            pl.col('f2_location').fill_null(''),
            pl.col('goslims_canonical').fill_null(''),
            pl.col('goslims_moonlighting').fill_null(''),
            pl.col('publications').fill_null(0),
        )
        .with_columns(
            pl.when(pl.col('gene_symbol') != '')
            .then(pl.col('gene_symbol'))
            .otherwise(pl.col('protein'))
            .alias('label')
        )
    )
    summary = {
        'mode': f'multiplex_{name_a}_vs_{name_b}',
        'subnetwork_nodes': nodes_df.height,
        'subnetwork_edges': edges_df.height,
        'shared_edges': int((edges_df['edge_class'] == 'Shared').sum()) if len(edges_df) else 0,
        f'{name_a}_only_edges': int((edges_df['edge_class'] == f'{name_a}_only').sum()) if len(edges_df) else 0,
        f'{name_b}_only_edges': int((edges_df['edge_class'] == f'{name_b}_only').sum()) if len(edges_df) else 0,
    }
    return {'nodes': nodes_df, 'edges': edges_df, 'summary': summary}


def to_cytoscapejs_json(nodes_df: pl.DataFrame, edges_df: pl.DataFrame, title: str = 'Human Interactome Subnetwork') -> dict[str, Any]:
    """Convert Polars node and edge tables directly to Cytoscape.js / CyREST JSON format."""
    node_elements = []
    for row in nodes_df.to_dicts():
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        clean['id'] = str(clean.get('id') or clean['protein'])
        clean['name'] = str(clean.get('name') or clean['protein'])
        node_elements.append({'data': clean})

    edge_elements = []
    for idx, row in enumerate(edges_df.to_dicts()):
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        clean['id'] = f'e{idx}'
        clean['source'] = str(clean['source'])
        clean['target'] = str(clean['target'])
        edge_elements.append({'data': clean})

    return {
        'data': {'name': title},
        'elements': {'nodes': node_elements, 'edges': edge_elements},
    }


def apply_moonlighting_visual_style(
    network_suid: int,
    style_name: str = 'Interactome_Moonlighting_Style',
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Create and apply a visual style in Cytoscape Desktop using py4cytoscape."""
    defaults = {
        'NODE_FILL_COLOR': '#7F8C8D',
        'NODE_SHAPE': 'ELLIPSE',
        'NODE_SIZE': 35,
        'NODE_BORDER_WIDTH': 2.0,
        'NODE_BORDER_PAINT': '#2C3E50',
        'NODE_LABEL_FONT_SIZE': 11,
        'NODE_LABEL_COLOR': '#1A1A1A',
        'EDGE_WIDTH': 2.0,
        'EDGE_STROKE_UNSELECTED_PAINT': '#95A5A6',
        'EDGE_TRANSPARENCY': 190,
    }
    node_labels = p4c.map_visual_property('node label', 'label', 'p')
    node_color = p4c.map_visual_property(
        'node fill color',
        'node_role',
        'd',
        ['Moonlighting', 'Linker', 'Background'],
        ['#0072B2', '#D55E00', '#7F8C8D'],
    )
    node_shape = p4c.map_visual_property(
        'node shape',
        'node_role',
        'd',
        ['Moonlighting', 'Linker', 'Background'],
        ['DIAMOND', 'HEXAGON', 'ELLIPSE'],
    )
    node_size = p4c.map_visual_property(
        'node size',
        'subnetwork_degree',
        'c',
        [1, 10, 40],
        [28.0, 48.0, 75.0],
    )
    edge_color = p4c.map_visual_property(
        'edge stroke unselected paint',
        'edge_class',
        'd',
        ['M-M', 'M-Linker', 'M-B', 'B-B', 'Shared'],
        ['#0072B2', '#D55E00', '#56B4E9', '#BDC3C7', '#009E73'],
    )
    p4c.create_visual_style(
        style_name,
        defaults=defaults,
        mappings=[node_labels, node_color, node_shape, node_size, edge_color],
        base_url=base_url,
    )
    p4c.set_visual_style(style_name, network=network_suid, base_url=base_url)
    return style_name


def send_to_cytoscape(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    title: str,
    *,
    collection: str = 'Human Interactome Atlas',
    layout: str = 'force-directed',
    apply_style: bool = True,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Push Polars node and edge tables to Cytoscape Desktop via py4cytoscape."""
    cyjs = to_cytoscapejs_json(nodes_df, edges_df, title=title)
    suid = int(p4c.create_network_from_cytoscapejs(cyjs, title=title, collection=collection, base_url=base_url))
    applied_style = None
    if apply_style:
        try:
            applied_style = apply_moonlighting_visual_style(suid, base_url=base_url)
        except Exception:
            applied_style = 'default'
    try:
        p4c.layout_network(layout_name=layout, network=suid, base_url=base_url)
    except Exception:
        p4c.layout_network(network=suid, base_url=base_url)
    try:
        p4c.fit_content(network=suid, base_url=base_url)
    except Exception:
        pass
    return {
        'network_suid': suid,
        'title': title,
        'collection': collection,
        'layout': layout,
        'style': applied_style,
        'nodes_sent': nodes_df.height,
        'edges_sent': edges_df.height,
    }


def fetch_cytoscape_table_polars(
    network: int | str | None = None,
    table: str = 'node',
    base_url: str = DEFAULT_BASE_URL,
) -> pl.DataFrame:
    """Fetch a Cytoscape node, edge, or network table directly into a Polars DataFrame via CyREST."""
    suid = p4c.get_network_suid(network, base_url=base_url)
    table_type = f'default{table.lower()}' if not table.lower().startswith('default') else table.lower()
    rows = p4c.cyrest_get(f'networks/{suid}/tables/{table_type}/rows', base_url=base_url)
    if not rows:
        return pl.DataFrame()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        clean_row: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, list):
                clean_row[k] = '|'.join(str(x) for x in v)
            elif isinstance(v, dict):
                clean_row[k] = json.dumps(v, sort_keys=True)
            else:
                clean_row[k] = v
        normalized_rows.append(clean_row)
    return pl.from_dicts(normalized_rows, infer_schema_length=None)


def compute_cluster_enrichment_and_bridging(
    node_table: pl.DataFrame,
    edge_table: pl.DataFrame,
    cluster_col: str,
    cohort: set[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Given a Cytoscape node table with a cluster column, compute moonlighting enrichment and inter-cluster bridging in Polars."""
    if cluster_col not in node_table.columns:
        raise ValueError(f'Cluster column {cluster_col!r} not found in Cytoscape node table.')

    id_col = 'name' if 'name' in node_table.columns else ('id' if 'id' in node_table.columns else node_table.columns[0])
    df = node_table.with_columns(
        pl.col(id_col).cast(pl.String).alias('_node_id'),
        pl.col(cluster_col).cast(pl.String).fill_null('unassigned').alias('_cluster'),
    )
    if 'is_moonlighting' in df.columns:
        df = df.with_columns(pl.col('is_moonlighting').cast(pl.Boolean).fill_null(False).alias('_is_moon'))
    elif cohort is not None:
        df = df.with_columns(pl.col('_node_id').is_in(sorted(cohort)).alias('_is_moon'))
    else:
        df = df.with_columns(pl.lit(False).alias('_is_moon'))

    total_nodes = df.height
    total_moon = int(df['_is_moon'].sum())

    cluster_stats = (
        df.filter(pl.col('_cluster') != 'unassigned')
        .group_by('_cluster')
        .agg(
            pl.len().alias('cluster_size'),
            pl.col('_is_moon').sum().cast(pl.Int64).alias('moonlighting_hits'),
            pl.col('_node_id').filter(pl.col('_is_moon')).sort().str.join('|').alias('moonlighting_proteins'),
        )
        .sort(['moonlighting_hits', 'cluster_size'], descending=[True, True])
        .rename({'_cluster': 'cluster'})
    )

    if len(cluster_stats) and total_nodes > 0 and total_moon > 0:
        sizes = cluster_stats['cluster_size'].to_numpy()
        hits = cluster_stats['moonlighting_hits'].to_numpy()
        pvals = np.where(
            hits > 0,
            hypergeom.sf(hits - 1, total_nodes, total_moon, sizes),
            1.0,
        )
        qvals = bh_adjust(pvals)
        folds = (hits / np.maximum(1, sizes)) / (total_moon / total_nodes)
        cluster_stats = cluster_stats.with_columns(
            pl.Series('fold_enrichment', folds, dtype=pl.Float64),
            pl.Series('p_value', pvals, dtype=pl.Float64),
            pl.Series('fdr_bh', qvals, dtype=pl.Float64),
            pl.Series('significance', [significance_label(q) for q in qvals], dtype=pl.String),
        )
    else:
        cluster_stats = cluster_stats.with_columns(
            pl.lit(0.0).alias('fold_enrichment'),
            pl.lit(np.nan).alias('p_value'),
            pl.lit(np.nan).alias('fdr_bh'),
            pl.lit('n/a').alias('significance'),
        )

    # Inter-cluster bridging for every node using edge_table
    node_cluster_map = dict(zip(df['_node_id'].to_list(), df['_cluster'].to_list()))
    node_moon_map = dict(zip(df['_node_id'].to_list(), df['_is_moon'].to_list()))
    label_col = 'label' if 'label' in df.columns else id_col
    node_label_map = dict(zip(df['_node_id'].to_list(), df[label_col].cast(pl.String).to_list()))

    src_col = 'source' if 'source' in edge_table.columns else None
    tgt_col = 'target' if 'target' in edge_table.columns else None
    adj_clusters: dict[str, list[str]] = {u: [] for u in node_cluster_map}

    if src_col and tgt_col and len(edge_table):
        for s, t in zip(edge_table[src_col].cast(pl.String).to_list(), edge_table[tgt_col].cast(pl.String).to_list()):
            cs = node_cluster_map.get(s, 'unassigned')
            ct = node_cluster_map.get(t, 'unassigned')
            if s in adj_clusters:
                adj_clusters[s].append(ct)
            if t in adj_clusters:
                adj_clusters[t].append(cs)
    elif 'name' in edge_table.columns and len(edge_table):
        for ename in edge_table['name'].cast(pl.String).to_list():
            parts = ename.split(' (')
            if len(parts) == 2 and ') ' in parts[1]:
                s = parts[0]
                t = parts[1].split(') ', 1)[1]
                cs = node_cluster_map.get(s, 'unassigned')
                ct = node_cluster_map.get(t, 'unassigned')
                if s in adj_clusters:
                    adj_clusters[s].append(ct)
                if t in adj_clusters:
                    adj_clusters[t].append(cs)

    bridge_rows: list[dict[str, Any]] = []
    for u, own_c in node_cluster_map.items():
        nbr_c = [c for c in adj_clusters.get(u, []) if c != 'unassigned']
        k = len(nbr_c)
        if k > 0:
            counts: dict[str, int] = {}
            for c in nbr_c:
                counts[c] = counts.get(c, 0) + 1
            probs = np.array(list(counts.values()), dtype=float) / k
            participation = float(1.0 - np.sum(probs ** 2))
            entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-15))))
            distinct_c = len(counts)
            inter_edges = sum(1 for c in nbr_c if c != own_c)
        else:
            participation = 0.0
            entropy = 0.0
            distinct_c = 0
            inter_edges = 0
        bridge_rows.append({
            'protein': u,
            'label': node_label_map.get(u, u),
            'is_moonlighting': bool(node_moon_map.get(u, False)),
            'assigned_cluster': own_c,
            'clustered_degree': k,
            'distinct_neighbor_clusters': distinct_c,
            'inter_cluster_edges': inter_edges,
            'clustermaker_participation': participation,
            'clustermaker_entropy': entropy,
        })

    bridging_df = pl.DataFrame(bridge_rows).sort(
        ['clustermaker_participation', 'distinct_neighbor_clusters', 'inter_cluster_edges'],
        descending=[True, True, True],
    )
    return {'cluster_enrichment': cluster_stats, 'node_cluster_bridging': bridging_df}


def run_clustermaker2(
    algorithm: str = 'leiden',
    *,
    network: int | str | None = None,
    params: dict[str, Any] | None = None,
    cohort: set[str] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Execute a clusterMaker2 algorithm in Cytoscape Desktop via py4cytoscape and return Polars tables."""
    algo_key = algorithm.lower()
    if algo_key not in CLUSTERMAKER_ALGORITHMS:
        raise ValueError(f'Unsupported clusterMaker2 algorithm: {algorithm}')
    spec = CLUSTERMAKER_ALGORITHMS[algo_key]
    suid = p4c.get_network_suid(network, base_url=base_url)

    merged_params = dict(spec['default_params'])
    if params:
        merged_params.update(params)
    param_tokens = [f'network="SUID:{suid}"']
    for k, v in merged_params.items():
        if v is not None and v != '':
            param_tokens.append(f'{k}={v}')
    cmd = f"{spec['command']} {' '.join(param_tokens)}"
    cmd_response = p4c.commands_post(cmd, base_url=base_url)

    node_df = fetch_cytoscape_table_polars(network=suid, table='node', base_url=base_url)
    edge_df = fetch_cytoscape_table_polars(network=suid, table='edge', base_url=base_url)

    detected_col = next((c for c in spec['column_candidates'] if c in node_df.columns), None)
    if detected_col is None:
        # Look for any column matching *Cluster*
        detected_col = next((c for c in node_df.columns if 'cluster' in c.lower()), None)
    if detected_col is None:
        return {
            'command': cmd,
            'raw_response': cmd_response,
            'cluster_column': None,
            'node_table': node_df,
            'cluster_enrichment': pl.DataFrame(),
            'node_cluster_bridging': pl.DataFrame(),
        }

    analysis = compute_cluster_enrichment_and_bridging(node_df, edge_df, detected_col, cohort=cohort)
    return {
        'command': cmd,
        'raw_response': cmd_response,
        'cluster_column': detected_col,
        'node_table': node_df,
        'cluster_enrichment': analysis['cluster_enrichment'],
        'node_cluster_bridging': analysis['node_cluster_bridging'],
    }


def subnetwork_to_graph_context(nodes_df: pl.DataFrame, edges_df: pl.DataFrame) -> dict[str, Any]:
    """Convert Polars subnetwork node and edge tables into a sparse graph context for cached clustering."""
    nodes = sorted({str(u) for u in nodes_df['id'].to_list()})
    idx = {n: i for i, n in enumerate(nodes)}
    size = len(nodes)
    rows, cols = [], []
    for r in edges_df.select(['source', 'target']).iter_rows(named=True):
        u, v = str(r['source']), str(r['target'])
        if u in idx and v in idx and u != v:
            iu, iv = idx[u], idx[v]
            rows.extend([iu, iv])
            cols.extend([iv, iu])
    adj = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(size, size)).tocsr()
    if adj.nnz:
        adj.data[:] = 1
    weights = adj.astype(np.float64)
    arr = np.asarray(nodes)
    return {
        'nodes': arr,
        'index': idx,
        'adj': adj,
        'weights': weights,
        'degree': np.diff(adj.indptr),
        'edge_count': adj.nnz // 2,
        'weighted_available': False,
    }


def push_clustering_to_cytoscape(
    metrics_df: pl.DataFrame,
    algorithm: str,
    *,
    network: int | str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Push cached Classic, Overlapping, or Fuzzy clustering columns into a Cytoscape Desktop node table via CyREST."""
    suid = p4c.get_network_suid(network, base_url=base_url)
    prefix = f'atlas_{algorithm}'
    rows_payload = []
    for r in metrics_df.iter_rows(named=True):
        rows_payload.append({
            'id': str(r['protein']),
            f'{prefix}_community': int(r['community']),
            f'{prefix}_communities_str': str(r.get('communities_str', r['community'])),
            f'{prefix}_num_communities': int(r.get('num_communities', 1)),
            f'{prefix}_is_overlapping': bool(r.get('is_overlapping_member', False)),
            f'{prefix}_participation': float(r.get('participation', 0.0)),
            f'{prefix}_fuzzy_entropy': float(r.get('fuzzy_entropy', 0.0)),
            f'{prefix}_top2_ratio': float(r.get('top2_membership_ratio', 0.0)),
        })
    p4c.cyrest_put(
        f'networks/{suid}/tables/defaultnode',
        body={'key': 'id', 'dataKey': 'id', 'data': rows_payload},
        base_url=base_url,
    )
    return {'suid': suid, 'prefix': prefix, 'rows_updated': len(rows_payload)}


def _mannwhitney_summary_polars(df: pl.DataFrame, metrics: list[str], cohort: set[str] | None = None) -> pl.DataFrame:
    """Compare Moonlighting vs Background across numeric columns pulled from Cytoscape apps."""
    if not len(df):
        return pl.DataFrame()
    id_col = 'name' if 'name' in df.columns else ('id' if 'id' in df.columns else df.columns[0])
    if 'is_moonlighting' in df.columns:
        mask = df['is_moonlighting'].cast(pl.Boolean).fill_null(False).to_numpy()
    elif cohort is not None:
        mask = df[id_col].cast(pl.String).is_in(sorted(cohort)).to_numpy()
    else:
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    for m in metrics:
        if m not in df.columns:
            continue
        vals = df[m].cast(pl.Float64, strict=False).to_numpy()
        a = vals[mask]
        b = vals[~mask]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) and len(b):
            u_stat, p_val = mannwhitneyu(a, b, alternative='two-sided')
            rbc = float(2.0 * u_stat / (len(a) * len(b)) - 1.0)
            p_float = float(p_val)
        else:
            rbc = np.nan
            p_float = np.nan
        rows.append({
            'metric': m,
            'moonlighting_n': int(len(a)),
            'background_n': int(len(b)),
            'moonlighting_median': float(np.median(a)) if len(a) else np.nan,
            'background_median': float(np.median(b)) if len(b) else np.nan,
            'rank_biserial': rbc,
            'p_value': p_float,
        })
    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows)
    qvals = bh_adjust(out['p_value'].to_numpy())
    return out.with_columns(
        pl.Series('fdr_bh', qvals, dtype=pl.Float64),
        pl.Series('significance', [significance_label(q) for q in qvals], dtype=pl.String),
    )


def run_network_analyzer(
    network: int | str | None = None,
    cohort: set[str] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Run Cytoscape NetworkAnalyzer via py4cytoscape and return Polars node/edge metrics and statistical tests."""
    suid = p4c.get_network_suid(network, base_url=base_url)
    cmd = f'analyzer analyze network="SUID:{suid}" directed=false'
    resp = p4c.commands_post(cmd, base_url=base_url)
    node_df = fetch_cytoscape_table_polars(network=suid, table='node', base_url=base_url)
    edge_df = fetch_cytoscape_table_polars(network=suid, table='edge', base_url=base_url)
    available_metrics = [m for m in NETWORK_ANALYZER_METRICS if m in node_df.columns]
    tests_df = _mannwhitney_summary_polars(node_df, available_metrics, cohort=cohort)
    return {
        'command': cmd,
        'raw_response': resp,
        'available_metrics': available_metrics,
        'node_table': node_df,
        'edge_table': edge_df,
        'tests': tests_df,
    }


def run_cytoscape_diffusion(
    seed_ids: list[str],
    *,
    network: int | str | None = None,
    time_param: float = 0.1,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Select seed nodes in Cytoscape, run Cytoscape heat diffusion via py4cytoscape, and fetch ranked Polars table."""
    suid = p4c.get_network_suid(network, base_url=base_url)
    p4c.clear_selection(type='both', network=suid, base_url=base_url)
    selected = p4c.select_nodes(seed_ids, by_col='name', preserve_current_selection=False, network=suid, base_url=base_url)
    cmd = f'diffusion diffuse time={float(time_param)}'
    resp = p4c.commands_post(cmd, base_url=base_url)
    node_df = fetch_cytoscape_table_polars(network=suid, table='node', base_url=base_url)
    heat_cols = [c for c in node_df.columns if 'diffusion_output_heat' in c]
    if heat_cols:
        latest_heat = sorted(heat_cols)[-1]
        node_df = node_df.sort(latest_heat, descending=True)
    return {
        'command': cmd,
        'selected_seeds': selected,
        'raw_response': resp,
        'node_table': node_df,
    }


def extract_cytohubba_scores(
    network: int | str | None = None,
    cohort: set[str] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Fetch any cytoHubba ranking columns from the active Cytoscape node table into Polars and test enrichment."""
    suid = p4c.get_network_suid(network, base_url=base_url)
    node_df = fetch_cytoscape_table_polars(network=suid, table='node', base_url=base_url)
    found_cols = [c for c in CYTOHUBBA_METRICS if c in node_df.columns]
    tests_df = _mannwhitney_summary_polars(node_df, found_cols, cohort=cohort)
    return {
        'network_suid': suid,
        'cytohubba_columns': found_cols,
        'node_table': node_df,
        'tests': tests_df,
    }


def run_string_enrichment(
    network: int | str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    """Trigger stringApp functional enrichment on the active Cytoscape network and pull results into Polars."""
    suid = p4c.get_network_suid(network, base_url=base_url)
    p4c.set_current_network(network=suid, base_url=base_url)
    resp = p4c.commands_post('string retrieve enrichment allNetSpecies="Homo sapiens"', base_url=base_url)
    return {'command': 'string retrieve enrichment allNetSpecies="Homo sapiens"', 'raw_response': resp}


def render_cytoscapejs_html(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    color_by: str = 'node_role',
    layout_name: str = 'cose',
    highlight_bridges_only: bool = False,
    height: int = 620,
) -> str:
    """Build a self-contained interactive Cytoscape.js viewer HTML widget for Streamlit."""
    cyjs = to_cytoscapejs_json(nodes_df, edges_df)
    elements_json = json.dumps(cyjs['elements'])
    safe_layout = html.escape(layout_name)
    safe_color_by = html.escape(color_by)
    bridge_flag = 'true' if highlight_bridges_only else 'false'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; }}
  #toolbar {{
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 8px 12px; background: #ffffff; border-bottom: 1px solid #e0e0e0; font-size: 13px;
  }}
  #toolbar select, #toolbar input, #toolbar button {{
    padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px;
  }}
  #toolbar button {{ background: #f4f6f8; cursor: pointer; }}
  #toolbar button:hover {{ background: #e8ecef; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
  #container {{ display: flex; height: {height - 46}px; width: 100%; }}
  #cy {{ flex: 1; height: 100%; background: #ffffff; }}
  #inspector {{
    width: 280px; border-left: 1px solid #e0e0e0; background: #fcfcfc;
    padding: 10px 12px; overflow-y: auto; font-size: 12px; line-height: 1.45;
  }}
  #inspector h4 {{ margin: 0 0 6px 0; font-size: 13px; color: #1f2937; }}
  .kv {{ margin-bottom: 4px; word-break: break-word; }}
  .kv b {{ color: #374151; }}
</style>
</head>
<body>
<div id="toolbar">
  <span><b>Layout:</b>
    <select id="layoutSelect">
      <option value="cose" {"selected" if safe_layout == "cose" else ""}>CoSE (Force-Directed)</option>
      <option value="concentric" {"selected" if safe_layout == "concentric" else ""}>Concentric (By Degree)</option>
      <option value="circle" {"selected" if safe_layout == "circle" else ""}>Circle</option>
      <option value="breadthfirst" {"selected" if safe_layout == "breadthfirst" else ""}>Breadthfirst Hierarchy</option>
      <option value="grid" {"selected" if safe_layout == "grid" else ""}>Grid</option>
    </select>
  </span>
  <span><b>Filter:</b>
    <input type="text" id="searchInput" placeholder="Gene / UniProt ID..."/>
  </span>
  <label><input type="checkbox" id="bridgeToggle" {"checked" if bridge_flag == "true" else ""}/> Highlight bridge edges (&kappa;<sub>F</sub> &lt; 0 / 0 triangles)</label>
  <button id="fitBtn">Fit View</button>
  <button id="pngBtn">Export PNG</button>
  <span style="margin-left:auto;">
    <span class="legend-dot" style="background:#0072B2;"></span>Moonlighting
    <span class="legend-dot" style="background:#D55E00;margin-left:8px;"></span>Linker
    <span class="legend-dot" style="background:#7F8C8D;margin-left:8px;"></span>Background
  </span>
</div>
<div id="container">
  <div id="cy"></div>
  <div id="inspector">
    <h4>Node / Edge Inspector</h4>
    <div id="details">Click any node or interaction edge to inspect its topological &amp; moonlighting attributes.</div>
  </div>
</div>
<script>
const rawElements = {elements_json};
const colorPalette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#8064A2', '#2E8B57', '#CD5C5C', '#4682B4'];
const colorByField = "{safe_color_by}";

function nodeColor(ele) {{
  const role = ele.data('node_role');
  if (colorByField === 'node_role' || !ele.data(colorByField)) {{
    if (role === 'Moonlighting') return '#0072B2';
    if (role === 'Linker') return '#D55E00';
    return '#7F8C8D';
  }}
  const val = String(ele.data(colorByField));
  let hash = 0;
  for (let i = 0; i < val.length; i++) hash = (hash * 31 + val.charCodeAt(i)) % colorPalette.length;
  return colorPalette[Math.abs(hash)];
}}

const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: rawElements,
  style: [
    {{
      selector: 'node',
      style: {{
        'label': 'data(label)',
        'background-color': nodeColor,
        'shape': function(ele) {{
          const r = ele.data('node_role');
          return r === 'Moonlighting' ? 'diamond' : (r === 'Linker' ? 'hexagon' : 'ellipse');
        }},
        'width': 'mapData(subnetwork_degree, 1, 30, 22, 54)',
        'height': 'mapData(subnetwork_degree, 1, 30, 22, 54)',
        'font-size': 10,
        'color': '#111827',
        'text-valign': 'top',
        'text-halign': 'center',
        'border-width': 1.8,
        'border-color': '#1f2937'
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 'mapData(score, 0, 1, 1.5, 3.5)',
        'line-color': function(ele) {{
          if (ele.data('is_bridge_edge')) return '#C0392B';
          const c = ele.data('edge_class');
          if (c === 'M-M') return '#0072B2';
          if (c === 'M-Linker') return '#D55E00';
          if (c === 'Shared') return '#009E73';
          return '#95A5A6';
        }},
        'line-style': function(ele) {{
          return ele.data('is_bridge_edge') ? 'dashed' : 'solid';
        }},
        'opacity': 0.8
      }}
    }},
    {{
      selector: ':selected',
      style: {{
        'border-width': 4,
        'border-color': '#F39C12',
        'line-color': '#F39C12'
      }}
    }},
    {{
      selector: '.dimmed',
      style: {{ 'opacity': 0.15 }}
    }}
  ],
  layout: {{
    name: "{safe_layout}",
    animate: false,
    concentric: function(node) {{ return node.data('subnetwork_degree') || 1; }},
    levelWidth: function() {{ return 2; }}
  }}
}});

function applyBridgeFilter(onlyBridges) {{
  cy.edges().forEach(e => {{
    const isBridge = e.data('is_bridge_edge') || (e.data('forman_ricci_sub') < 0 && e.data('shared_triangles') === 0);
    if (onlyBridges && !isBridge) e.addClass('dimmed');
    else e.removeClass('dimmed');
  }});
}}
applyBridgeFilter(document.getElementById('bridgeToggle').checked);

document.getElementById('layoutSelect').addEventListener('change', function(ev) {{
  const name = ev.target.value;
  cy.layout({{
    name: name,
    animate: false,
    concentric: function(node) {{ return node.data('subnetwork_degree') || 1; }},
    levelWidth: function() {{ return 2; }}
  }}).run();
}});

document.getElementById('bridgeToggle').addEventListener('change', function(ev) {{
  applyBridgeFilter(ev.target.checked);
}});

document.getElementById('searchInput').addEventListener('input', function(ev) {{
  const q = ev.target.value.trim().toLowerCase();
  if (!q) {{
    cy.elements().removeClass('dimmed');
    applyBridgeFilter(document.getElementById('bridgeToggle').checked);
    return;
  }}
  cy.elements().addClass('dimmed');
  const matched = cy.nodes().filter(n => {{
    const lbl = String(n.data('label') || '').toLowerCase();
    const id = String(n.data('id') || '').toLowerCase();
    const srcs = String(n.data('source_sets') || '').toLowerCase();
    return lbl.includes(q) || id.includes(q) || srcs.includes(q);
  }});
  matched.closedNeighborhood().removeClass('dimmed');
}});

document.getElementById('fitBtn').addEventListener('click', function() {{
  cy.fit();
}});

document.getElementById('pngBtn').addEventListener('click', function() {{
  const png64 = cy.png({{ full: true, scale: 2, bg: '#ffffff' }});
  const a = document.createElement('a');
  a.href = png64;
  a.download = 'cytoscape_subnetwork.png';
  a.click();
}});

cy.on('tap', 'node', function(evt) {{
  const d = evt.target.data();
  const keys = Object.keys(d).filter(k => d[k] !== null && d[k] !== '');
  document.getElementById('details').innerHTML = keys.map(k => `<div class="kv"><b>${{k}}:</b> ${{d[k]}}</div>`).join('');
}});

cy.on('tap', 'edge', function(evt) {{
  const d = evt.target.data();
  const keys = Object.keys(d).filter(k => d[k] !== null && d[k] !== '');
  document.getElementById('details').innerHTML = keys.map(k => `<div class="kv"><b>${{k}}:</b> ${{d[k]}}</div>`).join('');
}});
</script>
</body>
</html>"""


def export_cx2_json(nodes_df: pl.DataFrame, edges_df: pl.DataFrame, title: str = 'Human Interactome Subnetwork') -> str:
    """Export subnetwork in Cytoscape CX2 JSON format."""
    node_id_map: dict[str, int] = {}
    cx_nodes = []
    for idx, row in enumerate(nodes_df.to_dicts(), start=1):
        uid = str(row.get('id') or row['protein'])
        node_id_map[uid] = idx
        attrs = {k: v for k, v in row.items() if v is not None and not (isinstance(v, float) and not np.isfinite(v))}
        cx_nodes.append({'id': idx, 'v': attrs})

    cx_edges = []
    for idx, row in enumerate(edges_df.to_dicts(), start=1):
        s_id = node_id_map.get(str(row['source']))
        t_id = node_id_map.get(str(row['target']))
        if s_id is None or t_id is None:
            continue
        attrs = {k: v for k, v in row.items() if k not in ('source', 'target') and v is not None and not (isinstance(v, float) and not np.isfinite(v))}
        cx_edges.append({'id': idx, 's': s_id, 't': t_id, 'v': attrs})

    cx2_doc = [
        {'CXVersion': '2.0', 'hasFragments': False},
        {'networkAttributes': [{'name': title, 'generated_by': 'Human Interactome Atlas'}]},
        {'nodes': cx_nodes},
        {'edges': cx_edges},
        {'status': [{'success': True}]},
    ]
    return json.dumps(cx2_doc, indent=2)


def export_sif_string(edges_df: pl.DataFrame) -> str:
    """Export subnetwork in Cytoscape Simple Interaction Format (.sif)."""
    if not len(edges_df):
        return ''
    lines = [
        f"{s}\t{i}\t{t}"
        for s, i, t in zip(
            edges_df['source'].cast(pl.String).to_list(),
            edges_df['interaction'].cast(pl.String).to_list(),
            edges_df['target'].cast(pl.String).to_list(),
        )
    ]
    return '\n'.join(lines) + '\n'


def export_graphml_string(nodes_df: pl.DataFrame, edges_df: pl.DataFrame) -> str:
    """Export subnetwork in GraphML format compatible with Cytoscape Desktop."""
    g = nx.Graph()
    for row in nodes_df.to_dicts():
        uid = str(row.get('id') or row['protein'])
        attrs = {k: ('' if v is None or (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in row.items() if k != 'id'}
        g.add_node(uid, **attrs)
    for row in edges_df.to_dicts():
        s = str(row['source'])
        t = str(row['target'])
        attrs = {k: ('' if v is None or (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in row.items() if k not in ('source', 'target')}
        g.add_edge(s, t, **attrs)
    buf = io.BytesIO()
    nx.write_graphml(g, buf)
    return buf.getvalue().decode('utf-8')


def build_cycmd_script(
    network_name: str = 'Human_Interactome_Subnetwork',
    cluster_algorithm: str = 'leiden',
    layout_name: str = 'force-directed',
) -> str:
    """Generate a Cytoscape Command Script (.cycmd) for Tools -> Execute Command File in Cytoscape Desktop."""
    spec = CLUSTERMAKER_ALGORITHMS.get(cluster_algorithm.lower(), CLUSTERMAKER_ALGORITHMS['leiden'])
    params_str = ' '.join(f'{k}={v}' for k, v in spec['default_params'].items())
    return "\n".join([
        f'# Cytoscape Automation Script for {network_name}',
        '# Run via Cytoscape Desktop: Tools -> Execute Command File...',
        'analyzer analyze directed=false',
        f'{spec["command"]} {params_str}'.strip(),
        f'layout {layout_name}',
        'view fit content',
        '',
    ])


def export_cytoscape_bundle_zip(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    title: str = 'moonlighting_subnetwork',
    extra_tables: dict[str, pl.DataFrame] | None = None,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Package all Cytoscape formats (.cyjs, .cx2, .graphml, .sif, .cycmd) and Polars CSV tables into a ZIP."""
    safe_stem = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in title).strip('_') or 'subnetwork'
    cyjs_str = json.dumps(to_cytoscapejs_json(nodes_df, edges_df, title=title), indent=2)
    cx2_str = export_cx2_json(nodes_df, edges_df, title=title)
    sif_str = export_sif_string(edges_df)
    graphml_str = export_graphml_string(nodes_df, edges_df)
    cycmd_str = build_cycmd_script(network_name=title)

    manifest = {
        'title': title,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'nodes': nodes_df.height,
        'edges': edges_df.height,
        'metadata': metadata or {},
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f'{safe_stem}.cyjs', cyjs_str)
        zf.writestr(f'{safe_stem}.cx2', cx2_str)
        zf.writestr(f'{safe_stem}.graphml', graphml_str)
        zf.writestr(f'{safe_stem}.sif', sif_str)
        zf.writestr(f'{safe_stem}_commands.cycmd', cycmd_str)
        zf.writestr(f'{safe_stem}_nodes.csv', nodes_df.write_csv())
        zf.writestr(f'{safe_stem}_edges.csv', edges_df.write_csv())
        if extra_tables:
            for name, df in extra_tables.items():
                if isinstance(df, pl.DataFrame) and len(df):
                    zf.writestr(f'{safe_stem}_{name}.csv', df.write_csv())
        zf.writestr('manifest.json', json.dumps(manifest, indent=2))
    return buf.getvalue()
