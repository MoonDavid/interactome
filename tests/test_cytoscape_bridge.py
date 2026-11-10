"""Tests for the Py4Cytoscape bridge, Polars subnetwork/bridge extraction, app clustering/hubs, and exports."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import polars as pl
import pytest
from scipy.sparse import coo_matrix

from cytoscape_bridge import (
    build_base_url,
    build_cycmd_script,
    build_multiplex_subnetwork_tables,
    build_subnetwork_tables,
    build_topology_network_tables,
    check_cytoscape_connection,
    compute_cluster_enrichment_and_bridging,
    cytoscape_topology_comparison,
    export_cx2_json,
    export_cytoscape_bundle_zip,
    export_graphml_string,
    export_sif_string,
    extract_cytohubba_scores,
    fetch_cytoscape_table_polars,
    render_cytoscapejs_html,
    run_clustermaker2,
    run_cytoscape_diffusion,
    run_network_analyzer,
    send_to_cytoscape,
    to_cytoscapejs_json,
)


def _topology_import(tmp_path):
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a': ['P1|Q1', 'P2', 'B1'], 'b': ['P2', 'B1', 'B2'],
                  'pInt': [0.9, 0.8, None]}).write_parquet(path)
    reference = build_topology_network_tables(path, {'Q1', 'P2'})
    nodes = pl.DataFrame({'SUID': [10, 11, 12, 13], 'name': ['P1|Q1', 'P2', 'B1', 'B2'],
                          'is_moonlighting': [False, False, True, True],
                          'MCC': [10.0, 9.0, 1.0, float('nan')],
                          'Unavailable': pl.Series([None] * 4, dtype=pl.Float64)})
    edges = pl.DataFrame({'source': [10, 11, 12], 'target': [11, 12, 13]})
    return reference, nodes, edges


def test_full_topology_app_import_uses_current_membership_and_finite_values(tmp_path):
    from moonlighting import bh_adjust, pool_network_comparisons

    reference, nodes, edges = _topology_import(tmp_path)
    assert reference['nodes'].height == 4
    assert reference['edges']['pInt'].null_count() == 1
    assert set(reference['nodes'].filter(pl.col('is_moonlighting'))['protein']) == {'P1|Q1', 'P2'}
    result = cytoscape_topology_comparison(nodes.reverse(), edges, reference, {'Q1', 'P2'},
                                            ['MCC', 'Unavailable'], 'MoonLitDB')
    metrics = result['metrics']
    assert metrics['protein'].to_list() == ['B1', 'B2', 'P1|Q1', 'P2']
    assert set(metrics.filter(pl.col('group') == 'MoonLitDB')['protein']) == {'P1|Q1', 'P2'}
    assert metrics['MCC'].null_count() == 1
    pooled = pool_network_comparisons({'First': result, 'Second': result})
    available = pooled['tests'].filter(pl.col('metric') == 'MCC')
    assert available['query_n'].to_list() == [2, 2]
    assert available['background_n'].to_list() == [1, 1]
    assert (available['rank_biserial'] == 1.0).all()
    missing = pooled['tests'].filter(pl.col('metric') == 'Unavailable')
    assert missing['rank_biserial'].is_nan().all()
    np.testing.assert_allclose(pooled['tests']['fdr_bh'].to_numpy(),
                               bh_adjust(pooled['tests']['p_value'].to_numpy()), equal_nan=True)


@pytest.mark.parametrize('change', ['missing_node', 'duplicate_node', 'changed_edge', 'self_loop', 'parallel_edge'])
def test_topology_app_import_rejects_modified_source_graphs(tmp_path, change):
    reference, nodes, edges = _topology_import(tmp_path)
    if change == 'missing_node':
        nodes = nodes.head(3)
    elif change == 'duplicate_node':
        nodes = pl.concat([nodes.head(3), nodes.head(1)])
    elif change == 'changed_edge':
        edges = pl.DataFrame({'source': [10, 10, 12], 'target': [11, 12, 13]})
    elif change == 'self_loop':
        edges = pl.DataFrame({'source': [10, 11, 12], 'target': [10, 12, 13]})
    else:
        edges = pl.concat([edges, edges.head(1)])
    with pytest.raises(ValueError):
        cytoscape_topology_comparison(nodes, edges, reference, {'Q1'}, ['MCC'])


def test_topology_export_preserves_edge_guard(tmp_path, monkeypatch):
    reference, nodes, edges = _topology_import(tmp_path)
    monkeypatch.setattr('cytoscape_bridge.ADVANCED_EDGE_LIMIT', 2)
    with pytest.raises(ValueError, match='500,000'):
        build_topology_network_tables(tmp_path / 'network.parquet', {'Q1'})


def _toy_graph() -> dict:
    """Create a 7-node graph with two triangles bridged by a moonlighting protein P1 and linker L1."""
    nodes = np.array(['P1', 'P2', 'P3', 'P4', 'P5', 'L1', 'B1'])
    idx = {str(n): i for i, n in enumerate(nodes)}
    # Triangle 1: (P1, P2, P3)
    # Triangle 2: (P4, P5, B1)
    # Bridge edges: (P1, P4) [0 shared triangles!], (L1, P1), (L1, P2), (L1, P4)
    edges = [
        ('P1', 'P2'), ('P2', 'P3'), ('P1', 'P3'),
        ('P4', 'P5'), ('P5', 'B1'), ('P4', 'B1'),
        ('P1', 'P4'),
        ('L1', 'P1'), ('L1', 'P2'), ('L1', 'P4'),
    ]
    rr = [idx[u] for u, v in edges] + [idx[v] for u, v in edges]
    cc = [idx[v] for u, v in edges] + [idx[u] for u, v in edges]
    n = len(nodes)
    adj = coo_matrix((np.ones(len(rr), dtype=np.uint8), (rr, cc)), shape=(n, n)).tocsr()
    weights = adj.astype(np.float64)
    return {
        'nodes': nodes,
        'index': idx,
        'adj': adj,
        'weights': weights,
        'degree': np.diff(adj.indptr),
        'edge_count': len(edges),
        'weighted_available': False,
    }


def _toy_moon_df() -> pl.DataFrame:
    return pl.DataFrame({
        'canonical_id': ['P1', 'P2', 'P4'],
        'Gene Names': ['GAPDH GAPD', 'ENO1', 'PGK1'],
        'curated': [True, True, False],
        'moonlitdb': [True, False, True],
        'moonprot': [True, True, False],
        'metamoon': [True, False, False],
        'multitask': [False, True, True],
        'moondb_curated': [False, False, False],
        'moondb_predicted': [False, False, False],
        'moonprot_f1_location': ['Cytoplasm', 'Cytoplasm', 'Cytoplasm'],
        'moonprot_f2_location': ['Nucleus', 'Cell membrane', 'Nucleus'],
        'moonlitdb_goslims_canonical': ['glycolysis', 'glycolysis', 'glycolysis'],
        'moonlitdb_goslims_moonlighting': ['apoptosis', 'plasminogen activation', 'DNA replication'],
        'moonlitdb_number_of_publications': ['12', '8', '5'],
    })


def test_build_subnetwork_tables_and_forman_ricci():
    g = _toy_graph()
    moon_df = _toy_moon_df()
    cohort = {'P1', 'P2', 'P4'}

    res = build_subnetwork_tables(g, moon_df, cohort, mode='cohort_linkers', max_linkers=10, max_nodes=20)
    nodes_df = res['nodes']
    edges_df = res['edges']

    assert isinstance(nodes_df, pl.DataFrame)
    assert isinstance(edges_df, pl.DataFrame)
    assert {'P1', 'P2', 'P4', 'L1'}.issubset(set(nodes_df['protein'].to_list()))

    # L1 connects to P1, P2, P4 (all 3 moonlighting proteins), so it must be classified as a Linker
    l1_row = nodes_df.filter(pl.col('protein') == 'L1').row(0, named=True)
    assert l1_row['node_role'] == 'Linker'
    assert l1_row['moonlighting_neighbors'] == 3

    # Check P1 metadata join
    p1_row = nodes_df.filter(pl.col('protein') == 'P1').row(0, named=True)
    assert p1_row['label'] == 'GAPDH'
    assert 'moonprot' in p1_row['source_sets']

    # Check ego_bridge mode around P1
    ego_res = build_subnetwork_tables(
        g, moon_df, cohort, mode='ego_bridge', seed_proteins=['P1'], hop_radius=1, max_nodes=20
    )
    ego_edges = ego_res['edges']
    # Edge (P2, P3) has shared triangle P1 in full graph; check Forman-Ricci formula 4 - k_u - k_v + 3*t_uv
    for row in ego_edges.to_dicts():
        u_deg = ego_res['nodes'].filter(pl.col('protein') == row['source'])['subnetwork_degree'][0]
        v_deg = ego_res['nodes'].filter(pl.col('protein') == row['target'])['subnetwork_degree'][0]
        expected_frc = 4 - u_deg - v_deg + 3 * row['subnetwork_triangles']
        assert row['forman_ricci_sub'] == pytest.approx(expected_frc)


def test_multiplex_subnetwork_and_exports():
    g_a = _toy_graph()
    g_b = _toy_graph()
    moon_df = _toy_moon_df()
    cohort = {'P1', 'P2', 'P4'}

    mux = build_multiplex_subnetwork_tables(g_a, g_b, 'HuRI', 'BioPlex', moon_df, cohort, max_nodes=20)
    assert mux['nodes'].height > 0
    assert (mux['edges']['edge_class'] == 'Shared').all()

    cyjs = to_cytoscapejs_json(mux['nodes'], mux['edges'], title='TestMux')
    assert cyjs['data']['name'] == 'TestMux'
    assert len(cyjs['elements']['nodes']) == mux['nodes'].height
    assert len(cyjs['elements']['edges']) == mux['edges'].height

    cx2 = json.loads(export_cx2_json(mux['nodes'], mux['edges'], title='TestMux'))
    assert cx2[0]['CXVersion'] == '2.0'

    sif = export_sif_string(mux['edges'])
    assert 'Shared' in sif

    gml = export_graphml_string(mux['nodes'], mux['edges'])
    assert '<graphml' in gml

    cycmd = build_cycmd_script('TestMux', 'leiden', 'force-directed')
    assert 'cluster leiden' in cycmd

    html_out = render_cytoscapejs_html(mux['nodes'], mux['edges'])
    assert 'cytoscape(' in html_out

    bundle = export_cytoscape_bundle_zip(mux['nodes'], mux['edges'], title='TestMux')
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        assert 'TestMux.cyjs' in names
        assert 'TestMux.cx2' in names
        assert 'TestMux.graphml' in names
        assert 'TestMux.sif' in names
        assert 'TestMux_nodes.csv' in names
        assert 'TestMux_edges.csv' in names


import io


def test_compute_cluster_enrichment_and_bridging():
    node_table = pl.DataFrame({
        'name': ['P1', 'P2', 'P3', 'P4', 'P5', 'B1'],
        'label': ['GAPDH', 'ENO1', 'P3', 'PGK1', 'P5', 'B1'],
        'is_moonlighting': [True, True, False, True, False, False],
        '__leidenCluster': ['1', '1', '1', '2', '2', '2'],
    })
    edge_table = pl.DataFrame({
        'source': ['P1', 'P2', 'P1', 'P4', 'P5', 'P1'],
        'target': ['P2', 'P3', 'P3', 'P5', 'B1', 'P4'],
    })
    out = compute_cluster_enrichment_and_bridging(node_table, edge_table, '__leidenCluster')
    enr = out['cluster_enrichment']
    bridging = out['node_cluster_bridging']

    assert enr.height == 2
    # P1 has 2 edges in cluster 1 (P2, P3) and 1 edge in cluster 2 (P4) -> participation = 1 - ((2/3)^2 + (1/3)^2) = 4/9
    p1_bridge = bridging.filter(pl.col('protein') == 'P1').row(0, named=True)
    assert p1_bridge['distinct_neighbor_clusters'] == 2
    assert p1_bridge['inter_cluster_edges'] == 1
    assert p1_bridge['clustermaker_participation'] == pytest.approx(4.0 / 9.0)


def test_py4cytoscape_bridge_with_mocks(monkeypatch):
    assert build_base_url('127.0.0.1', 1235) == 'http://127.0.0.1:1235/v1'

    # Simulate LM Studio port conflict on 1234
    class DummyResp:
        def json(self):
            return {'error': 'Unexpected endpoint or method. (GET /v1/version)'}

    monkeypatch.setattr('requests.get', lambda url, timeout=2.5: DummyResp())
    conflict = check_cytoscape_connection('http://127.0.0.1:1234/v1')
    assert not conflict['connected']
    assert conflict['port_conflict_detected'] is True

    # Simulate connected Cytoscape + py4cytoscape operations
    class ConnectedResp:
        def json(self):
            return {'apiVersion': 'v1', 'cytoscapeVersion': '3.10.4'}

    monkeypatch.setattr('requests.get', lambda url, timeout=2.5: ConnectedResp())
    monkeypatch.setattr('py4cytoscape.cytoscape_version_info', lambda base_url=None: {'apiVersion': 'v1', 'cytoscapeVersion': '3.10.4'})
    monkeypatch.setattr('py4cytoscape.get_installed_apps', lambda base_url=None: [{'appName': 'clusterMaker2', 'status': 'Installed'}])
    monkeypatch.setattr('py4cytoscape.get_network_list', lambda base_url=None, get_suids=False: [{'name': 'Net1', 'suid': 101}])

    ok = check_cytoscape_connection('http://127.0.0.1:1234/v1')
    assert ok['connected'] is True
    assert ok['installed_apps'][0]['appName'] == 'clusterMaker2'

    # Mock table fetch and clusterMaker2 / NetworkAnalyzer / cytoHubba / diffusion
    monkeypatch.setattr('py4cytoscape.get_network_suid', lambda network=None, base_url=None: 101)
    monkeypatch.setattr('py4cytoscape.set_current_network', lambda network=None, base_url=None: None)
    monkeypatch.setattr('py4cytoscape.commands_post', lambda cmd, base_url=None: {'status': 'ok', 'cmd': cmd})
    monkeypatch.setattr('py4cytoscape.clear_selection', lambda type='both', network=None, base_url=None: None)
    monkeypatch.setattr('py4cytoscape.select_nodes', lambda nodes, by_col='name', preserve_current_selection=False, network=None, base_url=None: nodes)
    monkeypatch.setattr('py4cytoscape.create_network_from_cytoscapejs', lambda cyjs, title=None, collection=None, base_url=None: 101)
    monkeypatch.setattr('py4cytoscape.create_visual_style', lambda *a, **kw: None)
    monkeypatch.setattr('py4cytoscape.set_visual_style', lambda *a, **kw: None)
    monkeypatch.setattr('py4cytoscape.layout_network', lambda *a, **kw: None)
    monkeypatch.setattr('py4cytoscape.fit_content', lambda *a, **kw: None)

    mock_node_rows = [
        {'SUID': 1, 'name': 'P1', 'is_moonlighting': True, '__leidenCluster': 1, 'BetweennessCentrality': 0.5, 'MCC': 12.0, 'diffusion_output_heat': 0.9},
        {'SUID': 2, 'name': 'P2', 'is_moonlighting': True, '__leidenCluster': 1, 'BetweennessCentrality': 0.4, 'MCC': 10.0, 'diffusion_output_heat': 0.8},
        {'SUID': 3, 'name': 'B1', 'is_moonlighting': False, '__leidenCluster': 2, 'BetweennessCentrality': 0.05, 'MCC': 1.0, 'diffusion_output_heat': 0.1},
    ]
    mock_edge_rows = [
        {'SUID': 10, 'source': 'P1', 'target': 'P2', 'name': 'P1 (pp) P2'},
        {'SUID': 11, 'source': 'P1', 'target': 'B1', 'name': 'P1 (pp) B1'},
    ]

    def fake_cyrest_get(op, parameters=None, base_url=None, **kw):
        if 'defaultnode/rows' in op:
            return mock_node_rows
        if 'defaultedge/rows' in op:
            return mock_edge_rows
        return {}

    monkeypatch.setattr('py4cytoscape.cyrest_get', fake_cyrest_get)

    g = _toy_graph()
    sub = build_subnetwork_tables(g, _toy_moon_df(), {'P1', 'P2', 'P4'})
    sent = send_to_cytoscape(sub['nodes'], sub['edges'], 'TestNet')
    assert sent['network_suid'] == 101

    cm = run_clustermaker2('leiden', network=101, cohort={'P1', 'P2'})
    assert cm['cluster_column'] == '__leidenCluster'
    assert cm['cluster_enrichment'].height == 2

    na = run_network_analyzer(network=101, cohort={'P1', 'P2'})
    assert 'BetweennessCentrality' in na['available_metrics']
    assert na['tests'].height == 1

    ch = extract_cytohubba_scores(network=101, cohort={'P1', 'P2'})
    assert 'MCC' in ch['cytohubba_columns']
    assert ch['tests'].height == 1

    diff = run_cytoscape_diffusion(['P1', 'P2'], network=101)
    assert diff['node_table']['name'][0] == 'P1'


def test_exact_cached_betweenness_and_all_19_clustering_algorithms(tmp_path):
    import networkx as nx
    from moonlighting import (
        CLUSTERING_CATALOG,
        cached_network_centralities,
        centrality_analysis,
        clustering_comparison_heatmap,
        compare_clustering_algorithms,
        run_cached_clustering,
    )

    g = _toy_graph()
    nx_g = nx.from_scipy_sparse_array(g['adj'])
    exact_nx = nx.betweenness_centrality(nx_g, k=None, normalized=True)

    res = centrality_analysis(g, {'P1', 'P2', 'P4'}, extra=True, landmarks=None)
    assert res['notes']['betweenness_mode'] == 'exact (all nodes, cached)'
    assert np.allclose(res['metrics']['betweenness_approx'].to_numpy(), [exact_nx[i] for i in range(len(g['nodes']))])

    # Second call hits memory/disk cache
    _, notes2 = cached_network_centralities(g, extra=True, landmarks=None, cache_dir=tmp_path / 'cent')
    assert notes2['cache_hit'] is True

    # Test all 19 clustering algorithms across classic, overlapping, and fuzzy paradigms
    assert len(CLUSTERING_CATALOG) == 19
    suite = compare_clustering_algorithms(g, {'P1', 'P2', 'P4'}, algorithms=list(CLUSTERING_CATALOG.keys()), seed=42, resolution=1.0)
    assert suite['summary_table'].height == 19
    assert set(suite['summary_table']['paradigm'].to_list()) == {'classic', 'overlapping', 'fuzzy'}
    fig = clustering_comparison_heatmap(suite['effects'])
    assert len(fig.data) == 1

    # Verify cache hit on second call
    cached_c = run_cached_clustering(g, algorithm='ego_splitting', resolution=1.0, seed=42)
    assert cached_c['meta']['cache_hit'] is True
