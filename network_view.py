"""Protein-network views; geometry does not use protein-set labels."""
import polars as pl
import plotly.graph_objects as go

from network_graph_utils import layout_3d


def sigma_layout_positions(nodes, edges, engine='Circular'):
    """Fixed Python coordinates for a larger WebGL preview, independent of membership."""
    import math
    proteins = nodes['protein'].to_list()
    if not proteins:
        return {}
    if engine == 'graph-tool SFDP':
        return graph_tool_positions(proteins, edges, 'SFDP')
    initial = [[math.cos(2 * math.pi * i / len(proteins)), math.sin(2 * math.pi * i / len(proteins))]
               for i in range(len(proteins))]
    if engine == 'Circular':
        return dict(zip(proteins, initial))
    import igraph as ig
    index = {p: i for i, p in enumerate(proteins)}
    graph = ig.Graph(n=len(proteins), edges=[(index[a], index[b]) for a, b in edges.select('source', 'target').iter_rows()])
    if engine == 'igraph DrL':
        layout = graph.layout_drl(seed=initial)
    elif engine == 'igraph FR':
        layout = graph.layout_fruchterman_reingold(seed=initial, niter=150, grid='auto')
    else:
        raise ValueError(f'Unknown Sigma layout engine: {engine}')
    coordinates = layout.coords
    # Lay out disconnected components at comparable scales. Otherwise DrL can
    # spread components so far apart that their internal edges become invisible.
    components = sorted(graph.connected_components(), key=lambda c: (-len(c), c))
    if len(components) > 1:
        columns = math.ceil(math.sqrt(len(components)))
        largest = len(components[0])
        for cell, component in enumerate(components):
            xs = [coordinates[i][0] for i in component]
            ys = [coordinates[i][1] for i in component]
            center = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
            radius = max(.25, math.sqrt(len(component) / largest))
            for i in component:
                coordinates[i] = [(coordinates[i][0] - center[0]) / span * radius * 1.8 + (cell % columns) * 2.5,
                                  (coordinates[i][1] - center[1]) / span * radius * 1.8 + (cell // columns) * 2.5]
    return dict(zip(proteins, coordinates))


def sigma_network_html(nodes, edges, positions, title='Protein network'):
    """Offline Sigma.js 3 + Graphology renderer. No browser force simulation."""
    import json
    import math
    from pathlib import Path
    node_data = []
    for row in nodes.iter_rows(named=True):
        point = positions[row['protein']]
        if len(point) != 2 or not all(math.isfinite(float(x)) for x in point):
            raise ValueError('Layout coordinates must be finite 2D points.')
        node_data.append({'key': row['protein'], 'attributes': dict(
            x=float(point[0]), y=float(point[1]), label=row['protein'], color=row['color'],
            size=max(2, float(row['size']) / 3), protein_set=row['protein_set'],
            degree=row['subnetwork_degree'], cluster_selected=row['cluster_selected'],
            labelColor='#F39C12' if row['cluster_selected'] else '#1F2937',
            forceLabel=row['cluster_selected'])})
    payload = json.dumps({'nodes': node_data, 'edges': [
        {'key': str(i), 'source': a, 'target': b, 'attributes': {'color': '#94A3B8', 'size': 1}}
        for i, (a, b) in enumerate(edges.select('source', 'target').iter_rows())], 'title': title}).replace('<', '\\u003c')
    assets = Path(__file__).parent / 'assets/sigma'
    template = (assets / 'viewer.html').read_text(encoding='utf-8')
    return template.replace('__GRAPHOLOGY_JS__', (assets / 'graphology.umd.min.js').read_text(encoding='utf-8')).replace(
        '__SIGMA_JS__', (assets / 'sigma.min.js').read_text(encoding='utf-8')).replace('__NETWORK_DATA__', payload)


def pyvis_network_html(nodes, edges, *, physics=True, labels=False):
    """Standalone HTML with embedded vis-network assets and portable highlight colors."""
    import html
    import json
    import re
    from pyvis.network import Network

    network = Network(height='650px', width='100%', directed=False, cdn_resources='in_line')
    for row in nodes.iter_rows(named=True):
        network.add_node(row['protein'], label=row['protein'] if labels else ' ',
            title=f"{html.escape(row['protein'])}<br>{html.escape(row['protein_set'])}<br>Displayed degree: {row['subnetwork_degree']}",
            shape='diamond' if row['is_moonlighting'] else 'dot', size=float(row['size']),
            color={'background': row['color'], 'border': row['border_color'],
                   'highlight': {'background': row['color'], 'border': '#F39C12'}},
            borderWidth=3 if row['cluster_selected'] else 1)
    for a, b in edges.select('source', 'target').iter_rows():
        network.add_edge(a, b, color='#9CA3AF', width=1)
    network.set_options(json.dumps({
        'layout': {'randomSeed': 42},
        'physics': {'enabled': physics, 'solver': 'barnesHut', 'stabilization': {'iterations': 150}},
        'interaction': {'hover': True, 'navigationButtons': True, 'keyboard': True},
        'edges': {'smooth': False},
    }))
    document = network.generate_html(notebook=False)
    # PyVis embeds vis-network but its default template still fetches Bootstrap.
    # No menus use Bootstrap here, so remove those unused remote resources.
    document = re.sub(r'<link\b[^>]*href="https://cdn\.jsdelivr\.net/npm/bootstrap[^>]*>', '', document)
    document = re.sub(r'<script\b[^>]*src="https://cdn\.jsdelivr\.net/npm/bootstrap[^>]*>\s*</script>', '', document)
    legend = '<p style="font:14px sans-serif">Blue diamonds: selected moonlighting protein set · Gray: other proteins · Orange border: selected cluster</p>'
    return document.replace('<body>', '<body>' + legend, 1)


def graph_tool_positions(proteins, edges, layout='SFDP'):
    """Optional native graph-tool layout; import only when explicitly selected."""
    import graph_tool as gt
    from graph_tool.draw import sfdp_layout, fruchterman_reingold_layout
    import math

    graph = gt.Graph(directed=False)
    graph.add_vertex(len(proteins))
    index = {p: i for i, p in enumerate(proteins)}
    graph.add_edge_list([(index[a], index[b]) for a, b in edges.select('source', 'target').iter_rows()])
    initial = graph.new_vertex_property('vector<double>')
    for i in range(len(proteins)):
        angle = 2 * math.pi * i / max(1, len(proteins))
        initial[graph.vertex(i)] = [math.cos(angle), math.sin(angle)]
    if layout == 'Circular':
        positions = initial
    elif layout == 'SFDP':
        positions = sfdp_layout(graph, pos=initial)
    else:
        positions = fruchterman_reingold_layout(graph, pos=initial, n_iter=100)
    return {p: list(positions[graph.vertex(i)]) for i, p in enumerate(proteins)}


def styled_network_nodes(nodes, highlighted=(), set_name='Selected protein set'):
    """Portable style attributes; GraphML readers may require explicit mappings."""
    return nodes.with_columns(
        pl.col('protein').is_in(list(highlighted)).alias('cluster_selected'),
        pl.when('is_moonlighting').then(pl.lit('#0072B2')).otherwise(pl.lit('#9CA3AF')).alias('color'),
        pl.when('is_moonlighting').then(pl.lit('diamond')).otherwise(pl.lit('ellipse')).alias('shape'),
        pl.when('is_moonlighting').then(pl.lit(set_name)).otherwise(pl.lit('Other proteins')).alias('protein_set'),
        (6 + 3 * pl.col('subnetwork_degree').sqrt()).clip(upper_bound=30).alias('size'),
    ).with_columns(
        pl.when('cluster_selected').then(pl.lit('#F39C12')).otherwise(pl.lit('#374151')).alias('border_color'))


def static_network_figure(nodes, edges, title, *, engine='NetworkX', layout='Spring', labels=False):
    """Publication-friendly static drawing of the displayed graph."""
    import networkx as nx
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from textwrap import fill

    proteins = nodes['protein'].to_list()
    graph = nx.Graph()
    graph.add_nodes_from(proteins)
    graph.add_edges_from(edges.select('source', 'target').iter_rows())
    if engine == 'graph-tool':
        positions = graph_tool_positions(proteins, edges, layout)
    elif engine == 'igraph':
        import igraph as ig
        index = {p: i for i, p in enumerate(proteins)}
        igraph = ig.Graph(n=len(proteins), edges=[(index[a], index[b]) for a, b in graph.edges()])
        initial = igraph.layout_circle()
        positions = initial if layout == 'Circular' else igraph.layout_fruchterman_reingold(seed=initial.coords, niter=150)
        positions = dict(zip(proteins, positions.coords))
    elif layout == 'Circular':
        positions = nx.circular_layout(graph)
    elif layout == 'Kamada–Kawai':
        positions = nx.kamada_kawai_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=42, iterations=80)
    fig = Figure(figsize=(12, 9), layout='constrained')
    ax = fig.subplots()
    ax.add_collection(LineCollection([(positions[a], positions[b]) for a, b in graph.edges()],
                                     colors='#9CA3AF', linewidths=.7, alpha=.6, zorder=1))
    for member, marker in ((False, 'o'), (True, 'D')):
        subset = nodes.filter(pl.col('is_moonlighting') == member)
        if subset.is_empty():
            continue
        points = np.asarray([positions[p] for p in subset['protein']])
        ax.scatter(points[:, 0], points[:, 1], s=[v ** 2 for v in subset['size']],
                   c=subset['color'].to_list(), marker=marker,
                   edgecolors=subset['border_color'].to_list(),
                   linewidths=[2 if v else .4 for v in subset['cluster_selected']], zorder=2)
    if labels:
        for protein, point in positions.items():
            ax.annotate(protein, point, xytext=(3, 3), textcoords='offset points', fontsize=7)
    handles = [Line2D([], [], marker='D', color='#0072B2', linestyle='', label='Selected protein set'),
               Line2D([], [], marker='o', color='#9CA3AF', linestyle='', label='Other proteins')]
    if nodes['cluster_selected'].any():
        handles.append(Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor='#F39C12',
                              linestyle='', label='Selected cluster (orange border)'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.01), frameon=False, ncol=3)
    ax.set_title(f'{fill(title, width=100)}\n{engine} · {layout} · {nodes.height:,} proteins · {edges.height:,} interactions', fontsize=12)
    ax.autoscale()
    ax.margins(.12)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return fig


def static_image_bytes(figure, format):
    import io
    output = io.BytesIO()
    figure.savefig(output, format=format, dpi=200, bbox_inches='tight', facecolor='white')
    return output.getvalue()


def network_3d_figure(nodes, edges, title, highlighted=()):
    positions = layout_3d(nodes['protein'].to_list(),
                          edges.select('source', 'target').with_columns(pl.lit(1).alias('interactions')))
    coordinates = [[], [], []]
    for a, b in edges.select('source', 'target').iter_rows():
        for dim in range(3):
            coordinates[dim].extend([positions[a][dim], positions[b][dim], None])
    fig = go.Figure(go.Scatter3d(x=coordinates[0], y=coordinates[1], z=coordinates[2],
        mode='lines', line=dict(color='#9CA3AF', width=1), opacity=.35,
        hoverinfo='skip', showlegend=False))
    for member, label, color in ((False, 'Other proteins', '#9CA3AF'), (True, 'Selected protein set', '#0072B2')):
        subset = nodes.filter(pl.col('is_moonlighting') == member)
        points = [positions[p] for p in subset['protein']]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in points], y=[p[1] for p in points], z=[p[2] for p in points],
            mode='markers', name=label, customdata=subset.select('protein', 'subnetwork_degree').rows(),
            hovertemplate='%{customdata[0]}<br>Displayed degree: %{customdata[1]}<extra>%{fullData.name}</extra>',
            marker=dict(color=color, size=[min(14, 4 + d ** .5) for d in subset['subnetwork_degree']],
                        symbol='diamond' if member else 'circle')))
    selected = nodes.filter(pl.col('protein').is_in(list(highlighted)))
    if selected.height:
        points = [positions[p] for p in selected['protein']]
        fig.add_trace(go.Scatter3d(x=[p[0] for p in points], y=[p[1] for p in points],
            z=[p[2] for p in points], mode='markers', name='Selected cluster',
            marker=dict(color='#F39C12', size=3), text=selected['protein'].to_list()))
    hidden_axis = dict(visible=False)
    fig.update_layout(title=title, height=650, margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(xaxis=hidden_axis, yaxis=hidden_axis, zaxis=hidden_axis, aspectmode='data'),
        legend=dict(orientation='h'), template='plotly_white')
    return fig
