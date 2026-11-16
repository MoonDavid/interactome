"""Unit tests for ocg_exploration analysis helpers."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ocg_exploration import (
    cluster_overlap_graph,
    cluster_quality_metrics,
    cross_network_ocg_comparison,
    degree_matched_multi_membership,
    ego_cluster_subgraph,
    inter_cluster_functional_divergence,
    load_ocg_memberships,
    membership_count_distribution,
)


def _sample_network():
    # Two triangles (A,B,C) and (C,D,E) overlapping at bridge protein C, plus unassigned F.
    edges = pl.DataFrame({
        'a': ['A', 'A', 'B', 'C', 'C', 'D', 'A', 'F'],
        'b': ['B', 'C', 'C', 'D', 'E', 'E', 'F', 'E'],
    })
    memberships = pl.DataFrame({
        'protein': ['A', 'B', 'C', 'C', 'D', 'E', 'F'],
        'community': [0, 0, 0, 1, 1, 1, None],
    })
    return edges, memberships


def test_load_ocg_memberships_counts_overlap_and_nulls(tmp_path: Path):
    _, memberships = _sample_network()
    memberships.write_parquet(tmp_path / 'memberships.parquet')
    loaded, summary = load_ocg_memberships(tmp_path)
    assert loaded.height == 7
    row_c = summary.filter(pl.col('protein') == 'C').row(0, named=True)
    row_f = summary.filter(pl.col('protein') == 'F').row(0, named=True)
    assert row_c['memberships'] == 2
    assert row_c['is_multi'] is True
    assert row_f['memberships'] == 0
    assert row_f['is_assigned'] is False


def test_membership_distribution_exact_counts_and_denominators(tmp_path):
    _, memberships = _sample_network()
    # Duplicate assignments must not inflate k; a gap at k=3 stays explicit.
    memberships = pl.concat([memberships, memberships.head(1), pl.DataFrame({
        'protein': ['G'] * 4, 'community': [0, 1, 2, 3],
    })])
    memberships.write_parquet(tmp_path / 'memberships.parquet')
    _, summary = load_ocg_memberships(tmp_path)
    counts = membership_count_distribution(summary, {'C', 'G', 'absent'})
    all_counts = counts.filter(pl.col('group') == 'All run proteins')
    assert all_counts['proteins'].to_list() == [1, 4, 1, 0, 1]
    assert all_counts['proteins'].sum() == summary.height
    selected = counts.filter(pl.col('group') == 'Selected protein set')
    assert selected['population'].unique().to_list() == [2]
    assert selected['fraction'].sum() == pytest.approx(1)
    empty = membership_count_distribution(summary, set()).filter(pl.col('group') == 'Selected protein set')
    assert empty['fraction'].null_count() == empty.height


def test_global_overview_preserves_isolates_and_unfiltered_matrix():
    from ocg_visualization import global_overlap_figures
    _, members = _sample_network()
    members = pl.concat([members, pl.DataFrame({'protein': ['X'], 'community': [2]})])
    graph, heatmap, nodes, pairs, stats = global_overlap_figures(members, min_shared=2)
    assert stats['total_clusters'] == stats['displayed_clusters'] == 3
    assert stats['isolated_display_nodes'] == 3
    assert pairs.is_empty()
    assert len(graph.data[-1].x) == 3
    # The matrix retains the C0/C1 overlap even when graph thresholds hide it.
    import numpy as np
    matrix = np.array(heatmap.data[0].z)
    assert matrix.max() == 1
    assert np.allclose(matrix, matrix.T)
    assert np.isclose(matrix, 0.2).any()
    _, coef_map, _, _, _ = global_overlap_figures(members, metric='overlap_coef')
    assert any(v == pytest.approx(1 / 3) for row in coef_map.data[0].z for v in row)
    _, _, limited, _, limited_stats = global_overlap_figures(members, max_clusters=1)
    assert limited.height == 1
    assert limited_stats['total_clusters'] == 3


def test_global_overview_without_assigned_clusters():
    from ocg_visualization import global_overlap_figures
    _, _, nodes, _, stats = global_overlap_figures(pl.DataFrame({
        'protein': ['A'], 'community': [None],
    }))
    assert nodes.is_empty()
    assert stats['total_clusters'] == 0


def test_cytoscape_export_round_trip_preserves_attributes_and_isolates():
    import networkx as nx
    from ocg_visualization import cytoscape_overlap_graphml
    _, members = _sample_network()
    members = pl.concat([members, pl.DataFrame({'protein': ['X'], 'community': [2]})])
    graph = nx.parse_graphml(cytoscape_overlap_graphml(members))
    assert set(graph) == {'C0', 'C1', 'C2'}
    assert graph.degree('C2') == 0
    assert graph.nodes['C0']['protein_count'] == 3
    assert graph.nodes['C0']['proteins'] == 'A;B;C'
    assert graph['C0']['C1']['shared_proteins'] == 1
    assert graph['C0']['C1']['jaccard'] == pytest.approx(0.2)
    assert graph['C0']['C1']['overlap_coefficient'] == pytest.approx(1 / 3)
    filtered = nx.parse_graphml(cytoscape_overlap_graphml(members, min_similarity=0.5))
    assert len(filtered) == 3
    assert filtered.number_of_edges() == 0


def test_cluster_quality_metrics_and_overlap_graph():
    edges, memberships = _sample_network()
    quality = cluster_quality_metrics(memberships, edges, cohort={'C', 'D'})
    assert quality.height == 2
    c0 = quality.filter(pl.col('community') == 0).row(0, named=True)
    assert c0['size'] == 3
    assert c0['internal_edges'] == 3
    assert c0['internal_density'] == pytest.approx(1.0)
    assert c0['shared_proteins'] == 1
    assert 'qvalue' in quality.columns

    pairs, bridge = cluster_overlap_graph(memberships, min_shared=1)
    assert pairs.height == 1
    pair = pairs.row(0, named=True)
    assert pair['shared_proteins'] == 1
    assert pair['jaccard'] == pytest.approx(1 / 5)
    assert pair['overlap_coef'] == pytest.approx(1 / 3)
    assert bridge.filter(pl.col('community') == 0)['overlap_neighbors'][0] == 1


def test_degree_matched_multi_membership_and_divergence(tmp_path: Path):
    edges, memberships = _sample_network()
    memberships.write_parquet(tmp_path / 'memberships.parquet')
    _, summary = load_ocg_memberships(tmp_path)
    res, bin_df = degree_matched_multi_membership(summary, edges, cohort={'C', 'A'}, n_perm=50, seed=1)
    assert res['valid'] is True
    assert 0 < res['multi_pvalue'] <= 1.0
    assert not bin_df.is_empty()

    enrichment = pl.DataFrame({
        'community': ['0', '0', '1', '1'],
        'subset': ['full', 'full', 'full', 'full'],
        'library': ['GO_BP', 'GO_BP', 'GO_BP', 'GO_BP'],
        'term_id': ['GO:0001', 'GO:0002', 'GO:0003', 'GO:0004'],
        'term_name': ['t1', 't2', 't3', 't4'],
        'qvalue': [0.001, 0.01, 0.002, 0.02],
    }).with_columns(pl.col('community').cast(pl.Int64))
    div_df, div_sum = inter_cluster_functional_divergence(memberships, enrichment, cohort={'C'})
    assert div_sum['valid'] is True
    assert div_df.height == 1
    assert div_df['divergence_score'][0] == pytest.approx(1.0)


def test_cross_network_and_ego_subgraph(tmp_path: Path):
    edges, memberships = _sample_network()
    r1 = tmp_path / 'run1'
    r2 = tmp_path / 'run2'
    r1.mkdir()
    r2.mkdir()
    memberships.write_parquet(r1 / 'memberships.parquet')
    memberships.write_parquet(r2 / 'memberships.parquet')
    runs = pl.DataFrame({
        'network': ['NetA', 'NetB'],
        'run': ['r1', 'r2'],
        'directory': [str(r1), str(r2)],
    })
    conserved, enrich = cross_network_ocg_comparison(runs, cohort={'C'})
    row_c = conserved.filter(pl.col('protein') == 'C').row(0, named=True)
    assert row_c['networks_multi'] == 2
    assert not enrich.is_empty()

    nodes, sub_edges = ego_cluster_subgraph('C', memberships, edges, max_neighbors=10)
    assert 'C' in nodes['protein'].to_list()
    assert sub_edges.filter(pl.col('shared_clusters') > 0).height >= 4


def test_prepare_ocg_overlap_display_focus_cohort_and_isolates():
    from ocg_visualization import prepare_ocg_overlap_display
    _, members = _sample_network()
    members = pl.concat([members, pl.DataFrame({'protein': ['X'], 'community': [2]})])
    nodes, drawn, within, stats = prepare_ocg_overlap_display(
        members, max_clusters=1, cohort={'C'}, focus_cluster=2,
    )
    assert nodes['community'].to_list() == [2]
    assert drawn.is_empty() and within.is_empty()
    assert stats['isolated_display_nodes'] == 1

    nodes_all, drawn_all, _, stats_all = prepare_ocg_overlap_display(members, cohort={'C', 'D'})
    assert stats_all['total_clusters'] == 3
    assert stats_all['isolated_display_nodes'] == 1
    assert drawn_all.height == 1
    c0 = nodes_all.filter(pl.col('community') == 0).row(0, named=True)
    assert c0['cohort_hits'] == 1
    assert c0['cohort_fraction'] == pytest.approx(1 / 3)


def test_ocg_dynamic_viewers_cytoscape_echarts_sigma_pyvis_and_3d():
    import json
    import re
    from ocg_visualization import (
        ocg_3d_figure,
        ocg_cytoscapejs_html,
        ocg_echarts_option,
        ocg_pyvis_html,
        ocg_sigma_html,
    )
    _, members = _sample_network()
    members = pl.concat([members, pl.DataFrame({'protein': ['X'], 'community': [2]})])

    cy_overlap = ocg_cytoscapejs_html(members, cohort={'C'}, mode='cluster_overlap')
    assert 'OCG cluster overlap network' in cy_overlap
    assert '"id": "C0"' in cy_overlap
    cy_bip = ocg_cytoscapejs_html(members, cohort={'C'}, mode='bipartite_bridges')
    assert 'OCG bipartite cluster-bridge network' in cy_bip
    assert '"id": "C"' in cy_bip

    opt_force, _, drawn_ec, _ = ocg_echarts_option(members, cohort={'C'}, layout='force', include_bridges=False)
    assert opt_force['series'][0]['layout'] == 'force'
    assert len(drawn_ec) == 1
    opt_circ, _, _, _ = ocg_echarts_option(members, cohort={'C'}, layout='circular', include_bridges=True)
    assert opt_circ['series'][0]['layout'] == 'circular'
    assert any(c['name'] == 'Multi-cluster protein (k≥2)' for c in opt_circ['series'][0]['categories'])

    sigma_doc = ocg_sigma_html(members, cohort={'C'}, engine='Circular', title='OCG Sigma Test')
    assert '__NETWORK_DATA__' not in sigma_doc
    assert 'src="https://' not in sigma_doc
    payload = json.loads(re.search(r'const data = (.*);', sigma_doc).group(1))
    assert {n['key'] for n in payload['nodes']} == {'C0', 'C1', 'C2'}

    pytest.importorskip('pyvis')
    pyvis_doc = ocg_pyvis_html(members, cohort={'C'}, physics=False, labels=True)
    assert 'new vis.Network' in pyvis_doc
    assert 'href="https://cdn.jsdelivr.net/npm/bootstrap' not in pyvis_doc

    fig_3d = ocg_3d_figure(members, cohort={'C'}, protein_set_label='MoonLitDB')
    assert len(fig_3d.data) == 3


def test_ocg_static_overlap_and_upset_figures_and_empty_handling():
    from network_view import static_image_bytes
    from ocg_visualization import ocg_static_overlap_figure, ocg_static_upset_figure
    _, members = _sample_network()
    members = pl.concat([members, pl.DataFrame({'protein': ['X'], 'community': [2]})])

    fig_ov = ocg_static_overlap_figure(members, cohort={'C'}, engine='NetworkX', layout='Spring', labels=True)
    assert len(fig_ov.axes) >= 2
    assert static_image_bytes(fig_ov, 'png').startswith(b'\x89PNG')
    assert b'<svg' in static_image_bytes(fig_ov, 'svg')

    fig_up = ocg_static_upset_figure(members, cohort={'C'}, protein_set_label='MoonLitDB')
    assert len(fig_up.axes) == 3
    assert static_image_bytes(fig_up, 'png').startswith(b'\x89PNG')
    assert b'<svg' in static_image_bytes(fig_up, 'svg')

    empty = pl.DataFrame({'protein': ['A'], 'community': [None]})
    assert ocg_static_overlap_figure(empty).axes
    assert ocg_static_upset_figure(empty).axes


def test_ego_cluster_multi_framework_builders():
    from network_view import static_image_bytes
    from ocg_visualization import (
        ego_cluster_3d_figure,
        ego_cluster_bipartite_figure,
        ego_cluster_cytoscapejs_html,
        ego_cluster_echarts_option,
        ego_cluster_static_figure,
    )
    edges, memberships = _sample_network()
    nodes_df, sub_edges_df = ego_cluster_subgraph('C', memberships, edges, max_neighbors=10)

    bip = ego_cluster_bipartite_figure('C', nodes_df, sub_edges_df, memberships, cohort={'C', 'A'})
    assert len(bip.data) >= 4

    cy = ego_cluster_cytoscapejs_html('C', nodes_df, sub_edges_df, memberships, cohort={'C'}, include_clusters=True)
    assert '"id": "C0"' in cy and '"id": "C1"' in cy

    ec = ego_cluster_echarts_option('C', nodes_df, sub_edges_df, memberships, cohort={'C'}, include_clusters=True)
    assert any(d['id'] == 'C0' for d in ec['series'][0]['data'])

    fig3d = ego_cluster_3d_figure('C', nodes_df, sub_edges_df, memberships, cohort={'C'})
    assert len(fig3d.data) >= 2

    fig_static = ego_cluster_static_figure('C', nodes_df, sub_edges_df, memberships, cohort={'C'})
    assert static_image_bytes(fig_static, 'png').startswith(b'\x89PNG')
    assert b'<svg' in static_image_bytes(fig_static, 'svg')


def test_ocg_cluster_ppi_and_pair_ppi_view_builders():
    from ocg_visualization import build_ocg_cluster_ppi_view, build_ocg_pair_ppi_view

    edges, memberships = _sample_network()
    # Add another cross-cluster PPI (A-D) and a second bridge G so C0 has Interface-Interface edges too.
    edges = pl.concat([edges, pl.DataFrame({'a': ['A', 'A'], 'b': ['D', 'G']})])
    memberships = pl.concat([memberships, pl.DataFrame({'protein': ['G', 'G'], 'community': [0, 2]})])

    nodes_c0, edges_c0, stats_c0 = build_ocg_cluster_ppi_view(
        0, memberships, edges, cohort={'C'}, protein_set_label='MoonLitDB',
    )
    assert stats_c0['core_proteins'] == 2  # A, B
    assert stats_c0['interface_proteins'] == 2  # C, G
    assert stats_c0['cohort_proteins'] == 1
    row_a = nodes_c0.filter(pl.col('id') == 'A').row(0, named=True)
    row_c = nodes_c0.filter(pl.col('id') == 'C').row(0, named=True)
    assert row_a['ocg_role'] == 'Core (k=1)'
    assert row_c['ocg_role'] == 'Interface (k≥2)'
    assert row_c['shape'] == 'diamond'
    assert set(edges_c0['interaction'].to_list()) == {'Core–Core', 'Core–Interface'}

    # Focus protein restricts cluster PPI view to focus + internal neighbors
    nodes_focus, _, _ = build_ocg_cluster_ppi_view(0, memberships, edges, focus_protein='G')
    assert set(nodes_focus['id'].to_list()) == {'A', 'G'}

    # Pair PPI view across C0 and C1 should identify C as shared interface and A-D as Cross-cluster PPI
    nodes_pair, edges_pair, stats_pair = build_ocg_pair_ppi_view(
        0, 1, memberships, edges, cohort={'C'}, protein_set_label='MoonLitDB',
    )
    assert stats_pair['shared_proteins'] == 1
    assert stats_pair['only_a_proteins'] == 3  # A, B, G
    assert stats_pair['only_b_proteins'] == 2  # D, E
    assert stats_pair['cross_cluster_edges'] == 1  # A-D
    cross = edges_pair.filter(pl.col('interaction') == 'Cross-cluster PPI').row(0, named=True)
    assert cross['edge_style'] == 'dashed'
    assert set(nodes_pair.filter(pl.col('cluster_selected'))['id'].to_list()) == {'C'}


def test_unified_multi_framework_ocg_renderers():
    import json
    import re
    from network_view import static_image_bytes
    from ocg_visualization import (
        build_ocg_pair_ppi_view,
        render_ocg_view_3d_plotly,
        render_ocg_view_cytoscapejs,
        render_ocg_view_echarts,
        render_ocg_view_plotly2d,
        render_ocg_view_pyvis,
        render_ocg_view_sigma,
        render_ocg_view_static,
    )

    edges, memberships = _sample_network()
    edges = pl.concat([edges, pl.DataFrame({'a': ['A'], 'b': ['D']})])
    nodes_df, edges_df, _ = build_ocg_pair_ppi_view(0, 1, memberships, edges, cohort={'C'})

    fig2d = render_ocg_view_plotly2d(nodes_df, edges_df, title='Pair PPI', engine='NetworkX', layout='Spring')
    assert len(fig2d.data) >= 2

    opt_force = render_ocg_view_echarts(nodes_df, edges_df, layout='force')
    assert opt_force['series'][0]['layout'] == 'force'
    assert len(opt_force['series'][0]['data']) == nodes_df.height
    opt_circ = render_ocg_view_echarts(nodes_df, edges_df, layout='circular')
    assert opt_circ['series'][0]['layout'] == 'circular'

    cy_html = render_ocg_view_cytoscapejs(nodes_df, edges_df, layout_name='cose', title='Pair Cytoscape')
    assert 'Pair Cytoscape' in cy_html
    assert '"id": "C"' in cy_html

    sigma_html, sigma_nodes = render_ocg_view_sigma(nodes_df, edges_df, engine='NetworkX Spring', title='Pair Sigma')
    assert '__NETWORK_DATA__' not in sigma_html
    assert {'x', 'y'}.issubset(sigma_nodes.columns)
    payload = json.loads(re.search(r'const data = (.*);', sigma_html).group(1))
    assert len(payload['nodes']) == nodes_df.height

    fig_static = render_ocg_view_static(nodes_df, edges_df, title='Pair Static', engine='NetworkX', layout='Spring')
    assert static_image_bytes(fig_static, 'png').startswith(b'\x89PNG')
    assert b'<svg' in static_image_bytes(fig_static, 'svg')

    fig3d = render_ocg_view_3d_plotly(nodes_df, edges_df, title='Pair 3D')
    assert len(fig3d.data) >= 2

    pytest.importorskip('pyvis')
    pv_html = render_ocg_view_pyvis(nodes_df, edges_df, physics=False, labels=True, legend_text='Legend')
    assert 'new vis.Network' in pv_html
    assert 'href="https://cdn.jsdelivr.net/npm/bootstrap' not in pv_html

    # Empty view handling across renderers
    empty_n = nodes_df.head(0)
    empty_e = edges_df.head(0)
    assert render_ocg_view_echarts(empty_n, empty_e)['series'] == []
    assert static_image_bytes(render_ocg_view_static(empty_n, empty_e, title='Empty'), 'png').startswith(b'\x89PNG')
