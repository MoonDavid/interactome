"""Exact first-neighbor semantics and visualization/export membership."""
import ast
from pathlib import Path

import polars as pl

from networks import SCORE_COLUMNS, matching_nodes
from network_view import network_3d_figure
from network_view import styled_network_nodes, static_network_figure, static_image_bytes
from cytoscape_bridge import export_graphml_string


def preview_function():
    # Load the pure selection function without executing the Streamlit page.
    tree = ast.parse((Path(__file__).parents[1] / 'pages/4_Network_Visualization.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preview_tables')
    function.decorator_list = []
    namespace = dict(pl=pl, Path=Path, SCORE_COLUMNS=SCORE_COLUMNS, matching_nodes=matching_nodes,
                     ALL='All interactions', MOON_NEIGHBORS='Moonlighting + first neighbors')
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<preview>', 'exec'), namespace)
    return namespace['preview_tables']


def test_first_neighbors_excludes_neighbor_neighbor_edges_and_exports_all(tmp_path):
    path = tmp_path / 'edges.parquet'
    pl.DataFrame({'a': ['M', 'M', 'A', 'X'], 'b': ['A', 'B', 'B', 'Y']}).write_parquet(path)
    preview = preview_function()
    nodes, edges, truncated = preview(path, 0, 0, '', None, 'Moonlighting + first neighbors', ('M',))
    assert set(edges.rows()) == {('M', 'A'), ('M', 'B')}
    assert nodes.filter('is_moonlighting')['protein'].to_list() == ['M']
    assert not truncated
    assert preview(path, 0, 0, '', 1, 'Moonlighting + first neighbors', ('M',))[2]
    assert preview(path, 0, 0, '', None, 'Moonlighting + first neighbors', ())[1].is_empty()
    fig = network_3d_figure(nodes, edges, 'Test')
    assert fig.data[2].marker.color == '#0072B2'
    assert len(fig.data[2].x) == 1


def test_grouped_accession_membership(tmp_path):
    path = tmp_path / 'edges.parquet'
    pl.DataFrame({'a': ['M|Q', 'A'], 'b': ['A', 'B']}).write_parquet(path)
    nodes, edges, _ = preview_function()(path, 0, 0, '', None, 'Moonlighting + first neighbors', ('M',))
    assert edges.height == 1
    assert nodes.filter('is_moonlighting')['protein'].to_list() == ['M|Q']


def test_graphml_retains_highlights_and_static_images(tmp_path):
    import io
    import networkx as nx
    nodes = styled_network_nodes(pl.DataFrame({
        'protein': ['M', 'A', 'B'], 'is_moonlighting': [True, False, False],
        'subnetwork_degree': [2, 1, 1], 'label': ['M', 'A', 'B'],
    }), ['A'], 'MoonLitDB')
    edges = pl.DataFrame({'source': ['M', 'M'], 'target': ['A', 'B']})
    graph = nx.read_graphml(io.BytesIO(export_graphml_string(nodes, edges).encode()))
    assert graph.nodes['M']['is_moonlighting'] is True
    assert graph.nodes['M']['color'] == '#0072B2'
    assert graph.nodes['M']['protein_set'] == 'MoonLitDB'
    assert graph.nodes['A']['cluster_selected'] is True
    assert graph.nodes['A']['border_color'] == '#F39C12'
    assert graph.number_of_edges() == 2
    for layout in ['Spring', 'Circular', 'Kamada–Kawai']:
        fig = static_network_figure(nodes, edges, 'MoonLitDB first neighbors', layout=layout, labels=True)
        assert static_image_bytes(fig, 'png').startswith(b'\x89PNG')
        assert b'<svg' in static_image_bytes(fig, 'svg')
        assert static_image_bytes(fig, 'pdf').startswith(b'%PDF')


def test_igraph_static_layout_if_installed():
    import pytest
    pytest.importorskip('igraph')
    nodes = styled_network_nodes(pl.DataFrame({'protein': ['M', 'A'],
        'is_moonlighting': [True, False], 'subnetwork_degree': [1, 1]}))
    edges = pl.DataFrame({'source': ['M'], 'target': ['A']})
    for layout in ['Spring', 'Circular']:
        assert static_network_figure(nodes, edges, 'igraph', engine='igraph', layout=layout).axes


def test_pyvis_embeds_assets_and_membership_colors():
    import pytest
    pytest.importorskip('pyvis')
    from network_view import pyvis_network_html
    nodes = styled_network_nodes(pl.DataFrame({'protein': ['M', 'A'],
        'is_moonlighting': [True, False], 'subnetwork_degree': [1, 1]}), ['A'], 'MoonLitDB')
    edges = pl.DataFrame({'source': ['M'], 'target': ['A']})
    document = pyvis_network_html(nodes, edges, physics=False, labels=True)
    assert '#0072B2' in document and '#F39C12' in document
    assert 'MoonLitDB' in document and 'Displayed degree' in document
    assert 'new vis.Network' in document
    assert '"enabled": false' in document
    assert 'src="lib/' not in document
    assert 'src="https://' not in document
    assert 'href="https://cdn.jsdelivr.net/npm/bootstrap' not in document


def test_graph_tool_layout_if_installed():
    import pytest
    pytest.importorskip('graph_tool')
    from network_view import graph_tool_positions
    proteins = ['M', 'A', 'B']
    edges = pl.DataFrame({'source': ['M', 'M'], 'target': ['A', 'B']})
    for layout in ['SFDP', 'Spring', 'Circular']:
        positions = graph_tool_positions(proteins, edges, layout)
        assert set(positions) == set(proteins)
        assert all(len(point) == 2 for point in positions.values())


def test_sigma_fixed_layout_and_offline_html():
    import json
    import re
    from network_view import sigma_layout_positions, sigma_network_html
    nodes = styled_network_nodes(pl.DataFrame({'protein': ['M', 'A'],
        'is_moonlighting': [True, False], 'subnetwork_degree': [1, 1]}), ['A'], 'MoonLitDB')
    edges = pl.DataFrame({'source': ['M'], 'target': ['A']})
    positions = sigma_layout_positions(nodes, edges, 'Circular')
    assert positions == sigma_layout_positions(nodes, edges, 'Circular')
    document = sigma_network_html(nodes, edges, positions, '</script> test')
    payload = json.loads(re.search(r'const data = (.*);', document).group(1))
    assert payload['nodes'][0]['attributes']['color'] == '#0072B2'
    assert payload['nodes'][1]['attributes']['cluster_selected'] is True
    assert payload['nodes'][0]['attributes']['x'] == positions['M'][0]
    assert len(payload['edges']) == 1
    assert payload['title'] == '</script> test'
    assert 'src="https://' not in document
    assert '__NETWORK_DATA__' not in document


def test_sigma_igraph_engines_if_installed():
    import pytest
    import math
    pytest.importorskip('igraph')
    from network_view import sigma_layout_positions
    nodes = pl.DataFrame({'protein': ['M', 'A', 'B']})
    edges = pl.DataFrame({'source': ['M', 'M'], 'target': ['A', 'B']})
    for engine in ['igraph FR', 'igraph DrL']:
        positions = sigma_layout_positions(nodes, edges, engine)
        assert set(positions) == set(nodes['protein'])
        assert all(math.isfinite(v) for point in positions.values() for v in point)
