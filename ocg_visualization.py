"""Readable, bounded community-level and ego-cluster figures for OCG exploration."""
from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import re

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from network_graph_utils import cluster_colors, layout_3d, protein_identity
from moonlighting import significance_label
from ocg_exploration import _deduplicated_edges, cluster_overlap_graph, cluster_quality_metrics


def cytoscape_overlap_graphml(
    memberships: pl.DataFrame, *, min_shared: int = 1,
    min_similarity: float = 0.0, metric: str = 'jaccard',
) -> str:
    """Portable full-run community network; retain isolates and numeric attributes."""
    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    pairs, clusters = cluster_overlap_graph(memberships, max_edges=None)
    graph = nx.Graph()
    graph.graph.update(
        description='OCG community overlap network; each node is a cluster, not a protein',
        minimum_shared_proteins=int(min_shared), minimum_similarity=float(min_similarity),
        similarity_metric=metric, edge_statistics='Descriptive overlap; no significance test',
    )
    assigned = memberships.filter(pl.col('community').is_not_null()).unique()
    member_lists = dict(assigned.group_by('community').agg(
        pl.col('protein').sort().str.join(';').alias('proteins')).iter_rows())
    for row in clusters.iter_rows(named=True):
        cid = row['community']
        graph.add_node(f'C{cid}', name=f'C{cid}', community=int(cid),
            protein_count=int(row['size']), proteins=member_lists[cid],
            full_run_overlap_neighbors=int(row['overlap_neighbors']),
            full_run_shared_links=int(row['total_shared_links']))
    if not pairs.is_empty():
        for row in pairs.filter((pl.col('shared_proteins') >= min_shared)
                               & (pl.col(metric) >= min_similarity)).iter_rows(named=True):
            graph.add_edge(f"C{row['community_a']}", f"C{row['community_b']}",
                shared_proteins=int(row['shared_proteins']), jaccard=float(row['jaccard']),
                overlap_coefficient=float(row['overlap_coef']), weight=float(row[metric]),
                interaction='shares proteins')
    return '\n'.join(nx.generate_graphml(graph))


def global_overlap_figures(
    memberships: pl.DataFrame, *, max_clusters: int = 150,
    min_shared: int = 1, min_similarity: float = 0.0, metric: str = 'jaccard',
) -> tuple[go.Figure, go.Figure, pl.DataFrame, pl.DataFrame, dict]:
    """Keep isolated clusters; calculate overlaps before applying display limits.

    Matrix entries use every overlapping pair among displayed clusters. Edge
    filters and the 2,500-edge drawing cap apply only to the node-link view.
    """
    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    max_clusters = max(1, min(300, int(max_clusters)))
    pairs, clusters = cluster_overlap_graph(memberships, max_edges=None)
    if clusters.is_empty():
        return go.Figure(), go.Figure(), clusters, pairs, {'total_clusters': 0}
    nodes = clusters.sort('size', 'community', descending=[True, False]).head(max_clusters)
    ids = nodes['community'].to_list()
    within = pairs.filter(
        pl.col('community_a').is_in(ids) & pl.col('community_b').is_in(ids)
    ) if not pairs.is_empty() else pairs
    qualifying = within.filter(
        (pl.col('shared_proteins') >= min_shared) & (pl.col(metric) >= min_similarity)
    ) if not within.is_empty() else within
    drawn = qualifying.head(2500)
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for row in drawn.iter_rows(named=True):
        graph.add_edge(row['community_a'], row['community_b'], weight=row[metric])
    pos = nx.spring_layout(graph, seed=42, iterations=70, weight='weight')
    figure = go.Figure()
    # Width bins encode shared-protein counts without a separate trace per edge.
    maximum = int(drawn['shared_proteins'].max()) if not drawn.is_empty() else 1
    for band in range(3):
        xs, ys = [], []
        for row in drawn.iter_rows(named=True):
            strength = math.log1p(row['shared_proteins']) / math.log1p(maximum)
            if min(2, int(strength * 3)) != band:
                continue
            u, v = row['community_a'], row['community_b']
            xs.extend([pos[u][0], pos[v][0], None])
            ys.extend([pos[u][1], pos[v][1], None])
        if xs:
            figure.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                line=dict(width=0.6 + band * 1.2, color='rgba(120,135,155,0.45)'),
                hoverinfo='skip', showlegend=False))
    figure.add_trace(go.Scatter(
        x=[pos[c][0] for c in ids], y=[pos[c][1] for c in ids],
        mode='markers+text', text=[f'C{c}' if i < 15 else '' for i, c in enumerate(ids)],
        textposition='top center', cliponaxis=False,
        customdata=nodes.select('community', 'size', 'overlap_neighbors', 'total_shared_links').rows(),
        hovertemplate='Cluster C%{customdata[0]}<br>Proteins: %{customdata[1]}'
                      '<br>Overlapping clusters (full run): %{customdata[2]}'
                      '<br>Shared links (full run): %{customdata[3]}<extra></extra>',
        marker=dict(size=[max(9, 55 * math.sqrt(s / max(nodes['size']))) for s in nodes['size']],
                    color=nodes['overlap_neighbors'].to_list(), colorscale='Viridis',
                    showscale=True, colorbar=dict(title='Overlapping<br>clusters'),
                    line=dict(width=1, color='white')), showlegend=False,
    ))
    figure.update_layout(title='Community overview · one node per OCG cluster', height=620,
        margin=dict(l=35, r=60, t=70, b=35),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))

    index = {c: i for i, c in enumerate(ids)}
    matrix = np.eye(len(ids))
    for row in within.iter_rows(named=True):
        a, b = index[row['community_a']], index[row['community_b']]
        matrix[a, b] = matrix[b, a] = row[metric]
    order = leaves_list(linkage(squareform(np.clip(1 - matrix, 0, 1), checks=False),
                                method='average')) if len(ids) > 1 else [0]
    labels = [f'C{ids[i]}' for i in order]
    matrix = matrix[np.ix_(order, order)]
    metric_label = 'Jaccard similarity' if metric == 'jaccard' else 'Overlap coefficient'
    heatmap = go.Figure(go.Heatmap(z=matrix.tolist(), x=labels, y=labels, zmin=0, zmax=1,
        colorscale='Blues', colorbar=dict(title='Similarity'),
        hovertemplate='%{y} × %{x}<br>Similarity: %{z:.3f}<extra></extra>'))
    heatmap.update_layout(title=f'Community overlap · {metric_label}', height=650,
        margin=dict(l=70, r=60, t=65, b=80),
        xaxis=dict(title='Cluster (hierarchical order)', showticklabels=len(ids) <= 60, tickangle=-60),
        yaxis=dict(title='Cluster (same order)', autorange='reversed', showticklabels=len(ids) <= 60))
    stats = dict(total_clusters=clusters.height, displayed_clusters=len(ids),
                 qualifying_edges=qualifying.height, drawn_edges=drawn.height,
                 isolated_display_nodes=nx.number_of_isolates(graph))
    return figure, heatmap, nodes, drawn, stats


def prepare_ocg_overlap_display(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    focus_cluster: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Prepare bounded OCG cluster nodes, overlap edges, and optional protein-set hit counts."""
    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    max_clusters = max(1, min(300, int(max_clusters)))
    pairs, clusters = cluster_overlap_graph(memberships, max_edges=None)
    if clusters.is_empty():
        empty_nodes = pl.DataFrame(schema={
            'community': pl.Int64, 'size': pl.Int64, 'overlap_neighbors': pl.Int64,
            'total_shared_links': pl.Int64, 'cohort_hits': pl.Int64,
            'cohort_fraction': pl.Float64, 'member_preview': pl.String,
        })
        return empty_nodes, pairs, pairs, {
            'total_clusters': 0, 'displayed_clusters': 0, 'qualifying_edges': 0,
            'drawn_edges': 0, 'isolated_display_nodes': 0,
        }

    assigned = memberships.filter(pl.col('community').is_not_null()).select('protein', 'community').unique()
    cohort_set = set(cohort) if cohort else set()
    cluster_meta = (
        assigned.with_columns(pl.col('protein').is_in(sorted(cohort_set)).alias('_in_cohort'))
        .sort('community', 'protein')
        .group_by('community')
        .agg(
            pl.col('_in_cohort').sum().cast(pl.Int64).alias('cohort_hits'),
            pl.col('protein').head(8).str.join(', ').alias('member_preview'),
        )
    )
    enriched_clusters = (
        clusters.join(cluster_meta, on='community', how='left')
        .with_columns(
            pl.col('cohort_hits').fill_null(0).cast(pl.Int64),
            pl.col('member_preview').fill_null(''),
        )
        .with_columns(
            pl.when(pl.col('size') > 0)
            .then(pl.col('cohort_hits') / pl.col('size'))
            .otherwise(0.0)
            .alias('cohort_fraction')
        )
        .sort('size', 'community', descending=[True, False])
    )

    if focus_cluster is not None and not enriched_clusters.filter(pl.col('community') == focus_cluster).is_empty():
        focus_row = enriched_clusters.filter(pl.col('community') == focus_cluster)
        others = enriched_clusters.filter(pl.col('community') != focus_cluster).head(max(0, max_clusters - 1))
        nodes = pl.concat([focus_row, others])
    else:
        nodes = enriched_clusters.head(max_clusters)

    ids = nodes['community'].to_list()
    within = pairs.filter(
        pl.col('community_a').is_in(ids) & pl.col('community_b').is_in(ids)
    ) if not pairs.is_empty() else pairs
    qualifying = within.filter(
        (pl.col('shared_proteins') >= min_shared) & (pl.col(metric) >= min_similarity)
    ) if not within.is_empty() else within
    drawn = qualifying.head(2500)

    incident = set()
    if not drawn.is_empty():
        incident.update(drawn['community_a'].to_list())
        incident.update(drawn['community_b'].to_list())
    isolated = sum(1 for cid in ids if cid not in incident)

    stats = {
        'total_clusters': clusters.height,
        'displayed_clusters': len(ids),
        'qualifying_edges': qualifying.height,
        'drawn_edges': drawn.height,
        'isolated_display_nodes': isolated,
    }
    return nodes, drawn, within, stats


def _displayed_bridge_proteins(
    memberships: pl.DataFrame,
    cluster_ids: list[int],
    cohort: set[str] | None = None,
    *,
    max_bridges: int = 200,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Select multi-cluster bridge proteins (k>=2) shared across displayed clusters."""
    if not cluster_ids:
        return pl.DataFrame(), pl.DataFrame()
    cohort_set = set(cohort) if cohort else set()
    links = (
        memberships.filter(pl.col('community').is_in(cluster_ids))
        .select('protein', 'community')
        .unique()
    )
    if links.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    bridges = (
        links.group_by('protein')
        .agg(
            pl.col('community').n_unique().alias('displayed_clusters'),
            pl.col('community').sort().cast(pl.String).str.join(';').alias('clusters_str'),
        )
        .filter(pl.col('displayed_clusters') >= 2)
        .with_columns(pl.col('protein').is_in(sorted(cohort_set)).alias('in_cohort'))
        .sort(['in_cohort', 'displayed_clusters', 'protein'], descending=[True, True, False])
        .head(max_bridges)
    )
    if bridges.is_empty():
        return bridges, pl.DataFrame(schema={'protein': pl.String, 'community': pl.Int64})
    bridge_links = links.join(bridges.select('protein'), on='protein', how='semi')
    return bridges, bridge_links


def _pack_component_coords(
    node_ids: list[str],
    edges: list[tuple[str, str, float]],
    raw_coords: dict[str, tuple[float, float]],
) -> dict[str, list[float]]:
    """Pack disconnected components into a readable 2D grid so isolates do not crush hubs."""
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for u, v, w in edges:
        if u != v:
            graph.add_edge(u, v, weight=float(w))
    components = sorted(nx.connected_components(graph), key=lambda c: (-len(c), sorted(c)))
    if len(components) <= 1:
        return {str(k): [float(raw_coords[k][0]), float(raw_coords[k][1])] for k in node_ids}
    columns = math.ceil(math.sqrt(len(components)))
    largest = max(len(components[0]), 1)
    packed: dict[str, list[float]] = {}
    for cell, comp in enumerate(components):
        xs = [raw_coords[n][0] for n in comp]
        ys = [raw_coords[n][1] for n in comp]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
        radius = max(0.25, math.sqrt(len(comp) / largest))
        col_idx = cell % columns
        row_idx = cell // columns
        for n in comp:
            px = (raw_coords[n][0] - cx) / span * radius * 1.8 + col_idx * 2.6
            py = (raw_coords[n][1] - cy) / span * radius * 1.8 + row_idx * 2.6
            packed[str(n)] = [float(px), float(py)]
    return packed


def _compute_2d_layout(
    node_ids: list[str],
    edges: list[tuple[str, str, float]],
    *,
    engine: str = 'NetworkX',
    layout: str = 'Spring',
) -> dict[str, list[float]]:
    """Compute deterministic 2D coordinates across NetworkX, igraph, and Circular layouts."""
    if not node_ids:
        return {}
    n = len(node_ids)
    circular = {
        str(nid): [math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)]
        for i, nid in enumerate(node_ids)
    }
    mode = f'{engine} {layout}'.strip() if engine not in layout and engine != 'Circular' else layout
    if engine == 'Circular' or layout == 'Circular':
        return circular

    if engine == 'graph-tool' or 'SFDP' in mode:
        from network_view import graph_tool_positions
        edge_df = (
            pl.DataFrame({'source': [str(u) for u, v, _ in edges if u != v],
                          'target': [str(v) for u, v, _ in edges if u != v]})
            if edges else pl.DataFrame(schema={'source': pl.String, 'target': pl.String})
        )
        gt_layout = 'SFDP' if 'SFDP' in mode or layout == 'SFDP' else 'Spring'
        raw_pos = graph_tool_positions([str(nid) for nid in node_ids], edge_df, layout=gt_layout)
        raw = {str(k): (float(v[0]), float(v[1])) for k, v in raw_pos.items()}
        return _pack_component_coords(list(map(str, node_ids)), edges, raw)

    if engine == 'igraph' or mode in {'igraph FR', 'igraph DrL'}:
        import igraph as ig
        idx = {str(nid): i for i, nid in enumerate(node_ids)}
        ig_edges = [(idx[str(u)], idx[str(v)]) for u, v, _ in edges if u != v]
        graph = ig.Graph(n=n, edges=ig_edges, directed=False)
        init = [circular[str(nid)] for nid in node_ids]
        if 'DrL' in mode:
            coords = graph.layout_drl(seed=init).coords
        else:
            coords = graph.layout_fruchterman_reingold(seed=init, niter=180, grid='auto').coords
        raw = {str(nid): (float(coords[i][0]), float(coords[i][1])) for i, nid in enumerate(node_ids)}
        return _pack_component_coords(list(map(str, node_ids)), edges, raw)

    graph = nx.Graph()
    graph.add_nodes_from(map(str, node_ids))
    for u, v, w in edges:
        if u != v:
            graph.add_edge(str(u), str(v), weight=max(float(w), 1e-6))
    if 'Kamada' in mode:
        pos = nx.kamada_kawai_layout(graph, weight='weight')
    else:
        pos = nx.spring_layout(graph, seed=42, iterations=80, weight='weight')
    raw = {str(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    return _pack_component_coords(list(map(str, node_ids)), edges, raw)


def _viridis_hex(value: float, vmax: float) -> str:
    """Map a nonnegative scalar into an RGB hex color on the Viridis scale."""
    from matplotlib import colormaps
    norm = min(1.0, max(0.0, float(value) / max(vmax, 1.0)))
    r, g, b, _ = colormaps['viridis'](norm)
    return f'#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}'


def _cytoscapejs_document(
    node_elements: list[dict],
    edge_elements: list[dict],
    *,
    layout_name: str = 'cose',
    height: int = 640,
    title: str = 'OCG interactive explorer',
    legend_html: str = '',
) -> str:
    """Render a self-contained Cytoscape.js HTML widget with layout, filter, and inspector."""
    payload = json.dumps({'nodes': node_elements, 'edges': edge_elements}).replace('<', r'\u003c')
    safe_layout = escape(layout_name)
    safe_title = escape(title)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; color: #111827; }}
  #toolbar {{
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 8px 12px; background: #ffffff; border-bottom: 1px solid #e5e7eb; font-size: 13px;
  }}
  #toolbar select, #toolbar input, #toolbar button {{
    padding: 4px 8px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 12px;
  }}
  #toolbar button {{ background: #f3f4f6; cursor: pointer; }}
  #toolbar button:hover {{ background: #e5e7eb; }}
  #container {{ display: flex; height: {max(320, height - 48)}px; width: 100%; }}
  #cy {{ flex: 1; height: 100%; background: #ffffff; }}
  #inspector {{
    width: 290px; border-left: 1px solid #e5e7eb; background: #fcfcfc;
    padding: 10px 12px; overflow-y: auto; font-size: 12px; line-height: 1.45;
  }}
  #inspector h4 {{ margin: 0 0 6px 0; font-size: 13px; color: #1f2937; }}
  .kv {{ margin-bottom: 4px; word-break: break-word; }}
  .kv b {{ color: #374151; }}
</style>
</head>
<body>
<div id="toolbar">
  <strong>{safe_title}</strong>
  <span>Layout:
    <select id="layoutSelect">
      <option value="cose" {"selected" if safe_layout == "cose" else ""}>CoSE (Force-Directed)</option>
      <option value="concentric" {"selected" if safe_layout == "concentric" else ""}>Concentric</option>
      <option value="circle" {"selected" if safe_layout == "circle" else ""}>Circle</option>
      <option value="breadthfirst" {"selected" if safe_layout == "breadthfirst" else ""}>Breadthfirst</option>
      <option value="grid" {"selected" if safe_layout == "grid" else ""}>Grid</option>
    </select>
  </span>
  <input type="text" id="searchInput" placeholder="Search cluster or UniProt..."/>
  <label><input type="checkbox" id="cohortToggle"/> Highlight protein-set members</label>
  <button id="fitBtn">Fit View</button>
  <button id="pngBtn">Export PNG</button>
  <span style="margin-left:auto;font-size:12px;color:#374151;">{legend_html}</span>
</div>
<div id="container">
  <div id="cy"></div>
  <div id="inspector">
    <h4>Selection Inspector</h4>
    <div id="details">Click any node or edge to inspect cluster overlap attributes.</div>
  </div>
</div>
<script>
const rawElements = {payload};
if (typeof cytoscape !== 'function') {{
  document.getElementById('cy').textContent = 'Cytoscape.js could not load from CDN. Try Apache ECharts, Sigma.js, or Static.';
  throw new Error('Cytoscape.js could not load');
}}
const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: rawElements,
  style: [
    {{
      selector: 'node',
      style: {{
        'label': 'data(label)',
        'background-color': 'data(color)',
        'shape': 'data(shape)',
        'width': 'data(display_size)',
        'height': 'data(display_size)',
        'font-size': 10,
        'color': '#111827',
        'text-valign': 'top',
        'text-halign': 'center',
        'border-width': 'data(border_width)',
        'border-color': 'data(border_color)'
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 'data(edge_width)',
        'line-color': 'data(edge_color)',
        'line-style': 'data(edge_style)',
        'opacity': 0.72
      }}
    }},
    {{
      selector: ':selected',
      style: {{
        'border-width': 4,
        'border-color': '#D97706',
        'line-color': '#D97706'
      }}
    }},
    {{
      selector: '.dimmed',
      style: {{ 'opacity': 0.14 }}
    }}
  ],
  layout: {{
    name: "{safe_layout}",
    animate: false
  }}
}});

function applyFilters() {{
  const q = document.getElementById('searchInput').value.trim().toLowerCase();
  const onlyCohort = document.getElementById('cohortToggle').checked;
  if (!q && !onlyCohort) {{
    cy.elements().removeClass('dimmed');
    return;
  }}
  cy.elements().addClass('dimmed');
  const matched = cy.nodes().filter(n => {{
    const d = n.data();
    const matchesSearch = !q || [d.id, d.label, d.member_preview, d.clusters_str].some(
      v => v && String(v).toLowerCase().includes(q)
    );
    const matchesCohort = !onlyCohort || Boolean(d.in_cohort || d.has_cohort || d.is_center);
    return matchesSearch && matchesCohort;
  }});
  matched.closedNeighborhood().removeClass('dimmed');
}}

document.getElementById('layoutSelect').addEventListener('change', ev => {{
  cy.layout({{ name: ev.target.value, animate: false }}).run();
}});
document.getElementById('searchInput').addEventListener('input', applyFilters);
document.getElementById('cohortToggle').addEventListener('change', applyFilters);
document.getElementById('fitBtn').addEventListener('click', () => cy.fit());
document.getElementById('pngBtn').addEventListener('click', () => {{
  const a = document.createElement('a');
  a.href = cy.png({{ full: true, scale: 2, bg: '#ffffff' }});
  a.download = 'ocg_network.png';
  a.click();
}});
cy.on('tap', 'node, edge', evt => {{
  const d = evt.target.data();
  const details = document.getElementById('details');
  details.replaceChildren();
  for (const [k, v] of Object.entries(d)) {{
    if (v === null || v === '' || ['color', 'shape', 'display_size', 'border_width', 'border_color', 'edge_width', 'edge_color', 'edge_style'].includes(k)) continue;
    const row = document.createElement('div');
    row.className = 'kv';
    const b = document.createElement('b');
    b.textContent = `${{k}}: `;
    row.append(b, String(v));
    details.append(row);
  }}
}});
</script>
</body>
</html>"""


def ocg_cytoscapejs_html(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    mode: str = 'cluster_overlap',
    layout_name: str = 'cose',
    height: int = 640,
) -> str:
    """Build an interactive Cytoscape.js viewer for OCG cluster overlap or bipartite bridges."""
    if mode not in {'cluster_overlap', 'bipartite_bridges'}:
        raise ValueError('mode must be cluster_overlap or bipartite_bridges')
    nodes, drawn, _, _ = prepare_ocg_overlap_display(
        memberships, max_clusters=max_clusters, min_shared=min_shared,
        min_similarity=min_similarity, metric=metric, cohort=cohort,
    )
    if nodes.is_empty():
        return _cytoscapejs_document([], [], layout_name=layout_name, height=height,
                                     title='No OCG clusters to display')

    vmax = max(1, int(nodes['overlap_neighbors'].max() or 1))
    max_size = max(1, int(nodes['size'].max() or 1))
    node_elements = []
    for row in nodes.iter_rows(named=True):
        has_cohort = row['cohort_hits'] > 0
        node_elements.append({'data': {
            'id': f"C{row['community']}",
            'label': f"C{row['community']}" if nodes.height <= 80 or has_cohort else '',
            'node_kind': 'OCG cluster',
            'community': int(row['community']),
            'size': int(row['size']),
            'overlap_neighbors': int(row['overlap_neighbors']),
            'total_shared_links': int(row['total_shared_links']),
            'cohort_hits': int(row['cohort_hits']),
            'has_cohort': bool(has_cohort),
            'member_preview': row['member_preview'],
            'color': _viridis_hex(row['overlap_neighbors'], vmax),
            'shape': 'hexagon' if mode == 'bipartite_bridges' else ('diamond' if has_cohort else 'ellipse'),
            'display_size': round(18 + 38 * math.sqrt(row['size'] / max_size), 1),
            'border_width': 3.5 if has_cohort else 1.5,
            'border_color': '#F39C12' if has_cohort else '#1F2937',
        }})

    edge_elements = []
    if mode == 'cluster_overlap':
        for idx, row in enumerate(drawn.iter_rows(named=True)):
            edge_elements.append({'data': {
                'id': f'e{idx}',
                'source': f"C{row['community_a']}",
                'target': f"C{row['community_b']}",
                'shared_proteins': int(row['shared_proteins']),
                'jaccard': round(float(row['jaccard']), 4),
                'overlap_coef': round(float(row['overlap_coef']), 4),
                'edge_width': round(min(7.0, 1.0 + math.sqrt(row['shared_proteins'])), 2),
                'edge_color': '#64748B',
                'edge_style': 'solid',
            }})
        legend = 'Color: overlapping neighbors (full run) · Gold border/diamond: contains protein set'
        title = 'OCG cluster overlap network'
    else:
        cluster_ids = nodes['community'].to_list()
        bridges, bridge_links = _displayed_bridge_proteins(memberships, cluster_ids, cohort)
        for row in bridges.iter_rows(named=True):
            in_cohort = bool(row['in_cohort'])
            node_elements.append({'data': {
                'id': row['protein'],
                'label': row['protein'] if in_cohort or bridges.height <= 60 else '',
                'node_kind': 'Bridge protein (k≥2)',
                'displayed_clusters': int(row['displayed_clusters']),
                'clusters_str': row['clusters_str'],
                'in_cohort': in_cohort,
                'color': '#0072B2' if in_cohort else '#94A3B8',
                'shape': 'diamond' if in_cohort else 'ellipse',
                'display_size': round(12 + 4 * min(6, row['displayed_clusters']), 1),
                'border_width': 2.5 if in_cohort else 1.0,
                'border_color': '#F39C12' if in_cohort else '#334155',
            }})
        for idx, row in enumerate(bridge_links.iter_rows(named=True)):
            edge_elements.append({'data': {
                'id': f'b{idx}',
                'source': row['protein'],
                'target': f"C{row['community']}",
                'relationship': 'belongs to OCG cluster',
                'edge_width': 1.2,
                'edge_color': '#94A3B8',
                'edge_style': 'dashed',
            }})
        legend = 'Hexagon: OCG cluster · Diamond/Blue: bridge protein in set · Circle/Gray: other bridge protein'
        title = 'OCG bipartite cluster-bridge network'

    return _cytoscapejs_document(
        node_elements, edge_elements, layout_name=layout_name, height=height,
        title=title, legend_html=legend,
    )


def ocg_echarts_option(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    layout: str = 'force',
    include_bridges: bool = False,
) -> tuple[dict, pl.DataFrame, pl.DataFrame, dict]:
    """Build an Apache ECharts option for force, circular-chord, or bipartite OCG views."""
    if layout not in {'force', 'circular'}:
        raise ValueError('layout must be force or circular')
    nodes, drawn, _, stats = prepare_ocg_overlap_display(
        memberships, max_clusters=max_clusters, min_shared=min_shared,
        min_similarity=min_similarity, metric=metric, cohort=cohort,
    )
    if nodes.is_empty():
        return {'title': {'text': 'No assigned OCG clusters'}, 'series': []}, nodes, drawn, stats

    vmax = max(1, int(nodes['overlap_neighbors'].max() or 1))
    max_size = max(1, int(nodes['size'].max() or 1))
    data = []
    for row in nodes.iter_rows(named=True):
        cid = int(row['community'])
        has_cohort = row['cohort_hits'] > 0
        tip = (
            f"Cluster C{cid}<br>Proteins: {row['size']:,}"
            f"<br>Overlapping clusters (full run): {row['overlap_neighbors']:,}"
            f"<br>Protein-set hits: {row['cohort_hits']:,}"
        )
        if row['member_preview']:
            tip += f"<br>Sample: {escape(row['member_preview'])}"
        item = {
            'id': f'C{cid}',
            'name': f'C{cid}',
            'value': int(row['size']),
            'symbol': 'roundRect' if include_bridges else 'circle',
            'symbolSize': round(12 + 34 * math.sqrt(row['size'] / max_size), 1),
            'itemStyle': {
                'color': _viridis_hex(row['overlap_neighbors'], vmax),
                'borderColor': '#F39C12' if has_cohort else '#1F2937',
                'borderWidth': 3 if has_cohort else 1,
            },
            'label': {'show': nodes.height <= 45 or has_cohort},
            'tooltip': {'formatter': tip},
        }
        if include_bridges:
            item['category'] = 0
        data.append(item)

    links = []
    categories = None
    if include_bridges:
        categories = [
            {'name': 'OCG cluster'},
            {'name': 'Multi-cluster protein (k≥2)'},
        ]
        bridges, bridge_links = _displayed_bridge_proteins(memberships, nodes['community'].to_list(), cohort)
        for row in bridges.iter_rows(named=True):
            in_cohort = bool(row['in_cohort'])
            data.append({
                'id': row['protein'],
                'name': row['protein'],
                'category': 1,
                'value': int(row['displayed_clusters']),
                'symbol': 'diamond' if in_cohort else 'circle',
                'symbolSize': 12 if in_cohort else 8,
                'itemStyle': {
                    'color': '#0072B2' if in_cohort else '#9CA3AF',
                    'borderColor': '#F39C12' if in_cohort else '#4B5563',
                    'borderWidth': 2 if in_cohort else 1,
                },
                'label': {'show': in_cohort or bridges.height <= 30},
                'tooltip': {'formatter': (
                    f"{escape(row['protein'])}<br>Displayed OCG clusters: {row['displayed_clusters']}"
                    f"<br>Clusters: {escape(row['clusters_str'])}"
                )},
            })
        for row in bridge_links.iter_rows(named=True):
            links.append({
                'source': row['protein'],
                'target': f"C{row['community']}",
                'lineStyle': {'width': 1.0, 'color': '#94A3B8', 'curveness': 0.15 if layout == 'circular' else 0.0},
            })
    else:
        for row in drawn.iter_rows(named=True):
            links.append({
                'source': f"C{row['community_a']}",
                'target': f"C{row['community_b']}",
                'value': int(row['shared_proteins']),
                'lineStyle': {
                    'width': round(min(7.0, 0.8 + math.sqrt(row['shared_proteins'])), 2),
                    'curveness': 0.25 if layout == 'circular' else 0.0,
                },
                'tooltip': {'formatter': (
                    f"C{row['community_a']} ↔ C{row['community_b']}"
                    f"<br>Shared proteins: {row['shared_proteins']:,}"
                    f"<br>Jaccard: {row['jaccard']:.3f}"
                    f"<br>Overlap coef: {row['overlap_coef']:.3f}"
                )},
            })

    series = {
        'type': 'graph',
        'layout': layout,
        'roam': True,
        'draggable': True,
        'data': data,
        'links': links,
        'emphasis': {'focus': 'adjacency', 'label': {'show': True}},
        'lineStyle': {'opacity': 0.45},
        'label': {'position': 'right'},
    }
    if categories is not None:
        series['categories'] = categories
    if layout == 'force':
        series['force'] = {'repulsion': 200, 'edgeLength': [55, 165], 'layoutAnimation': False}
    else:
        series['circular'] = {'rotateLabel': True}

    option = {'tooltip': {}, 'series': [series]}
    if categories is not None:
        option['legend'] = {'data': [c['name'] for c in categories], 'top': 4}
    return option, nodes, drawn, stats


def ocg_sigma_html(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    engine: str = 'NetworkX Spring',
    title: str = 'OCG community overlap',
) -> str:
    """Offline Sigma.js 3 + Graphology WebGL viewer with deterministic Python coordinates."""
    nodes, drawn, _, _ = prepare_ocg_overlap_display(
        memberships, max_clusters=max_clusters, min_shared=min_shared,
        min_similarity=min_similarity, metric=metric, cohort=cohort,
    )
    node_ids = [f"C{c}" for c in nodes['community'].to_list()]
    edges = [
        (f"C{r['community_a']}", f"C{r['community_b']}", float(r[metric]))
        for r in drawn.iter_rows(named=True)
    ]
    layout_engine, _, layout_name = engine.partition(' ')
    if engine == 'Circular':
        layout_engine, layout_name = 'Circular', 'Circular'
    coords = _compute_2d_layout(node_ids, edges, engine=layout_engine, layout=layout_name or 'Spring')
    deg_map: dict[str, int] = {nid: 0 for nid in node_ids}
    for u, v, _ in edges:
        deg_map[u] = deg_map.get(u, 0) + 1
        deg_map[v] = deg_map.get(v, 0) + 1

    vmax = max(1, int(nodes['overlap_neighbors'].max() or 1)) if not nodes.is_empty() else 1
    max_size = max(1, int(nodes['size'].max() or 1)) if not nodes.is_empty() else 1
    node_data = []
    for row in nodes.iter_rows(named=True):
        cid = int(row['community'])
        nid = f'C{cid}'
        point = coords[nid]
        has_cohort = row['cohort_hits'] > 0
        node_data.append({
            'key': nid,
            'attributes': {
                'x': float(point[0]),
                'y': float(point[1]),
                'label': nid,
                'color': _viridis_hex(row['overlap_neighbors'], vmax),
                'size': round(4 + 14 * math.sqrt(row['size'] / max_size), 2),
                'protein_set': f"{row['size']} proteins · {row['cohort_hits']} in set · {row['overlap_neighbors']} overlaps",
                'degree': deg_map[nid],
                'cluster_selected': bool(has_cohort),
                'labelColor': '#D97706' if has_cohort else '#1F2937',
                'forceLabel': bool(has_cohort and nodes.height <= 100),
            },
        })
    edge_data = [
        {
            'key': str(i),
            'source': u,
            'target': v,
            'attributes': {'color': '#94A3B8', 'size': round(min(4.0, 0.8 + 2.5 * w), 2)},
        }
        for i, (u, v, w) in enumerate(edges)
    ]
    payload = json.dumps({'nodes': node_data, 'edges': edge_data, 'title': title}).replace('<', r'\u003c')
    assets = Path(__file__).parent / 'assets/sigma'
    template = (assets / 'viewer.html').read_text(encoding='utf-8')
    return (
        template.replace('__GRAPHOLOGY_JS__', (assets / 'graphology.umd.min.js').read_text(encoding='utf-8'))
        .replace('__SIGMA_JS__', (assets / 'sigma.min.js').read_text(encoding='utf-8'))
        .replace('__NETWORK_DATA__', payload)
    )


def ocg_pyvis_html(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    physics: bool = True,
    labels: bool = True,
) -> str:
    """Standalone PyVis (vis-network) interactive HTML viewer for OCG clusters."""
    from pyvis.network import Network

    nodes, drawn, _, _ = prepare_ocg_overlap_display(
        memberships, max_clusters=max_clusters, min_shared=min_shared,
        min_similarity=min_similarity, metric=metric, cohort=cohort,
    )
    net = Network(height='640px', width='100%', directed=False, cdn_resources='in_line')
    vmax = max(1, int(nodes['overlap_neighbors'].max() or 1)) if not nodes.is_empty() else 1
    max_size = max(1, int(nodes['size'].max() or 1)) if not nodes.is_empty() else 1
    for row in nodes.iter_rows(named=True):
        cid = int(row['community'])
        nid = f'C{cid}'
        has_cohort = row['cohort_hits'] > 0
        fill = _viridis_hex(row['overlap_neighbors'], vmax)
        border = '#F39C12' if has_cohort else '#1F2937'
        tip = (
            f"Cluster {nid}<br>Proteins: {row['size']:,}"
            f"<br>Overlapping clusters (full run): {row['overlap_neighbors']:,}"
            f"<br>Protein-set hits: {row['cohort_hits']:,}"
        )
        net.add_node(
            nid,
            label=nid if labels else ' ',
            title=tip,
            shape='diamond' if has_cohort else 'dot',
            size=round(10 + 26 * math.sqrt(row['size'] / max_size), 1),
            color={'background': fill, 'border': border,
                   'highlight': {'background': fill, 'border': '#D97706'}},
            borderWidth=3 if has_cohort else 1,
        )
    for row in drawn.iter_rows(named=True):
        net.add_edge(
            f"C{row['community_a']}",
            f"C{row['community_b']}",
            color='#94A3B8',
            width=round(min(6.0, 1.0 + math.sqrt(row['shared_proteins'])), 2),
            title=f"Shared: {row['shared_proteins']} · Jaccard: {row['jaccard']:.3f} · Overlap coef: {row['overlap_coef']:.3f}",
        )
    net.set_options(json.dumps({
        'layout': {'randomSeed': 42},
        'physics': {'enabled': physics, 'solver': 'barnesHut', 'stabilization': {'iterations': 150}},
        'interaction': {'hover': True, 'navigationButtons': True, 'keyboard': True},
        'edges': {'smooth': False},
    }))
    document = net.generate_html(notebook=False)
    document = re.sub(r'<link\b[^>]*href="https://cdn\.jsdelivr\.net/npm/bootstrap[^>]*>', '', document)
    document = re.sub(r'<script\b[^>]*src="https://cdn\.jsdelivr\.net/npm/bootstrap[^>]*>\s*</script>', '', document)
    legend = (
        '<p style="font:13px sans-serif;margin:8px 12px;">'
        'Diamonds / orange border: OCG clusters containing selected protein-set members · '
        'Fill color: overlapping cluster count (Viridis) · Edge width: shared proteins</p>'
    )
    return document.replace('<body>', '<body>' + legend, 1)


def ocg_3d_figure(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    protein_set_label: str = 'Selected protein set',
) -> go.Figure:
    """Rotatable 3D Plotly view of OCG cluster overlaps with component-packed coordinates."""
    nodes, drawn, _, _ = prepare_ocg_overlap_display(
        memberships, max_clusters=max_clusters, min_shared=min_shared,
        min_similarity=min_similarity, metric=metric, cohort=cohort,
    )
    if nodes.is_empty():
        return go.Figure()
    node_ids = [f"C{c}" for c in nodes['community'].to_list()]
    link_df = drawn.select(
        pl.format('C{}', pl.col('community_a')).alias('source'),
        pl.format('C{}', pl.col('community_b')).alias('target'),
        pl.col('shared_proteins').alias('interactions'),
    ) if not drawn.is_empty() else pl.DataFrame(schema={'source': pl.String, 'target': pl.String, 'interactions': pl.Int64})
    coords = layout_3d(node_ids, link_df)

    ex, ey, ez = [], [], []
    for a, b in link_df.select('source', 'target').iter_rows():
        ex.extend([coords[a][0], coords[b][0], None])
        ey.extend([coords[a][1], coords[b][1], None])
        ez.extend([coords[a][2], coords[b][2], None])
    fig = go.Figure()
    if ex:
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines', line=dict(color='#94A3B8', width=1.4),
            opacity=0.45, hoverinfo='skip', showlegend=False,
        ))

    max_size = max(1, int(nodes['size'].max() or 1))
    xs = [coords[f"C{c}"][0] for c in nodes['community'].to_list()]
    ys = [coords[f"C{c}"][1] for c in nodes['community'].to_list()]
    zs = [coords[f"C{c}"][2] for c in nodes['community'].to_list()]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='markers', name='OCG clusters',
        customdata=nodes.select('community', 'size', 'overlap_neighbors', 'cohort_hits').rows(),
        hovertemplate=(
            'Cluster C%{customdata[0]}<br>Proteins: %{customdata[1]}'
            '<br>Overlapping clusters (full run): %{customdata[2]}'
            f'<br>{escape(protein_set_label)} hits: ' + '%{customdata[3]}<extra></extra>'
        ),
        marker=dict(
            size=[max(5, 22 * math.sqrt(s / max_size)) for s in nodes['size'].to_list()],
            color=nodes['overlap_neighbors'].to_list(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Overlapping<br>clusters'),
            line=dict(width=1.2, color='#1F2937'),
        ),
    ))
    cohort_nodes = nodes.filter(pl.col('cohort_hits') > 0)
    if not cohort_nodes.is_empty():
        c_ids = cohort_nodes['community'].to_list()
        fig.add_trace(go.Scatter3d(
            x=[coords[f'C{c}'][0] for c in c_ids],
            y=[coords[f'C{c}'][1] for c in c_ids],
            z=[coords[f'C{c}'][2] for c in c_ids],
            mode='markers',
            name=f'Contains {protein_set_label}',
            hoverinfo='skip',
            marker=dict(
                size=[max(8, 25 * math.sqrt(s / max_size)) for s in cohort_nodes['size'].to_list()],
                color='rgba(243, 156, 18, 0.25)',
                line=dict(width=2.5, color='#F39C12'),
            ),
        ))
    hidden = dict(visible=False)
    fig.update_layout(
        title='3D OCG community overlap network',
        height=640,
        margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode='data'),
        template='plotly_white',
    )
    return fig




def ocg_static_overlap_figure(
    memberships: pl.DataFrame,
    *,
    max_clusters: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    protein_set_label: str = 'Selected protein set',
    engine: str = 'NetworkX',
    layout: str = 'Spring',
    labels: bool = True,
):
    """Two-panel publication-ready Matplotlib figure: overlap graph + similarity matrix."""
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    nodes, drawn, within, stats = prepare_ocg_overlap_display(
        memberships, max_clusters=max_clusters, min_shared=min_shared,
        min_similarity=min_similarity, metric=metric, cohort=cohort,
    )
    fig = Figure(figsize=(15, 7), layout='constrained')
    if nodes.is_empty():
        ax = fig.subplots()
        ax.text(0.5, 0.5, 'No assigned OCG clusters to display', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig

    ax_graph, ax_mat = fig.subplots(1, 2)
    ids = nodes['community'].to_list()
    node_ids = [f'C{c}' for c in ids]
    edges = [
        (f"C{r['community_a']}", f"C{r['community_b']}", float(r[metric]))
        for r in drawn.iter_rows(named=True)
    ]
    coords = _compute_2d_layout(node_ids, edges, engine=engine, layout=layout)

    if edges:
        max_shared = max(int(drawn['shared_proteins'].max() or 1), 1)
        segs = [(coords[f"C{r['community_a']}"], coords[f"C{r['community_b']}"]) for r in drawn.iter_rows(named=True)]
        widths = [0.5 + 2.5 * math.log1p(r['shared_proteins']) / math.log1p(max_shared) for r in drawn.iter_rows(named=True)]
        ax_graph.add_collection(LineCollection(segs, colors='#64748B', linewidths=widths, alpha=0.45, zorder=1))

    pts = np.asarray([coords[nid] for nid in node_ids])
    max_size = max(1, int(nodes['size'].max() or 1))
    sizes = [max(45.0, 620.0 * (s / max_size)) for s in nodes['size'].to_list()]
    overlaps = nodes['overlap_neighbors'].to_numpy()
    has_cohort = [h > 0 for h in nodes['cohort_hits'].to_list()]
    sc = ax_graph.scatter(
        pts[:, 0], pts[:, 1], s=sizes, c=overlaps, cmap='viridis',
        edgecolors=['#F39C12' if h else '#1F2937' for h in has_cohort],
        linewidths=[2.2 if h else 0.8 for h in has_cohort], zorder=2,
    )
    fig.colorbar(sc, ax=ax_graph, shrink=0.75, label='Overlapping clusters (full run)')
    if labels:
        for idx_row, row in enumerate(nodes.iter_rows(named=True)):
            if idx_row < 20 or row['cohort_hits'] > 0 and nodes.height <= 60:
                nid = f"C{row['community']}"
                ax_graph.annotate(nid, coords[nid], xytext=(3, 3), textcoords='offset points', fontsize=7.5)

    handles = [
        Line2D([], [], marker='o', markerfacecolor='#94A3B8', markeredgecolor='#1F2937',
               linestyle='', markersize=8, label='OCG cluster'),
    ]
    if any(has_cohort):
        handles.append(Line2D(
            [], [], marker='o', markerfacecolor='#94A3B8', markeredgecolor='#F39C12',
            markeredgewidth=2.2, linestyle='', markersize=8, label=f'Contains {protein_set_label}',
        ))
    ax_graph.legend(handles=handles, loc='lower left', frameon=True, fontsize=8.5)
    ax_graph.set_title(
        f"OCG overlap graph ({engine} · {layout})\n"
        f"{stats['displayed_clusters']}/{stats['total_clusters']} clusters · {stats['drawn_edges']} edges",
        fontsize=11,
    )
    ax_graph.autoscale()
    ax_graph.margins(0.12)
    ax_graph.set_aspect('equal')
    ax_graph.set_axis_off()

    index = {c: i for i, c in enumerate(ids)}
    matrix = np.eye(len(ids))
    for row in within.iter_rows(named=True):
        a, b = index[row['community_a']], index[row['community_b']]
        matrix[a, b] = matrix[b, a] = row[metric]
    order = leaves_list(linkage(squareform(np.clip(1 - matrix, 0, 1), checks=False), method='average')) if len(ids) > 1 else [0]
    ordered_labels = [f'C{ids[i]}' for i in order]
    matrix = matrix[np.ix_(order, order)]
    im = ax_mat.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    fig.colorbar(im, ax=ax_mat, shrink=0.75, label='Jaccard similarity' if metric == 'jaccard' else 'Overlap coefficient')
    if len(ids) <= 40:
        ticks = list(range(len(ordered_labels)))
        ax_mat.set_xticks(ticks, ordered_labels, rotation=60, ha='right', fontsize=7.5)
        ax_mat.set_yticks(ticks, ordered_labels, fontsize=7.5)
    else:
        ax_mat.set_xticks([])
        ax_mat.set_yticks([])
    metric_label = 'Jaccard similarity' if metric == 'jaccard' else 'Overlap coefficient'
    ax_mat.set_title(f'Cluster overlap matrix · {metric_label}\n(average-linkage hierarchical order)', fontsize=11)
    return fig


def ocg_static_upset_figure(
    memberships: pl.DataFrame,
    *,
    top_pairs: int = 25,
    metric: str = 'jaccard',
    cohort: set[str] | None = None,
    protein_set_label: str = 'Selected protein set',
):
    """UpSet-style static Matplotlib intersection plot of top overlapping OCG cluster pairs."""
    from matplotlib.figure import Figure

    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    pairs, clusters = cluster_overlap_graph(memberships, min_shared=1, max_edges=None)
    fig = Figure(figsize=(13, 7.5), layout='constrained')
    if pairs.is_empty() or clusters.is_empty():
        ax = fig.subplots()
        ax.text(0.5, 0.5, 'No overlapping OCG cluster pairs to display.', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig

    shown_pairs = pairs.sort(['shared_proteins', metric], descending=[True, True]).head(max(1, min(40, int(top_pairs))))
    cohort_set = set(cohort) if cohort else set()
    assigned = memberships.filter(pl.col('community').is_not_null()).select('protein', 'community').unique()
    if cohort_set:
        cohort_members = assigned.filter(pl.col('protein').is_in(sorted(cohort_set)))
        pair_cohort = (
            cohort_members.rename({'community': 'community_a'})
            .join(cohort_members.rename({'community': 'community_b'}), on='protein', how='inner')
            .filter(pl.col('community_a') < pl.col('community_b'))
            .group_by('community_a', 'community_b')
            .agg(pl.len().alias('shared_cohort_hits'))
        )
        shown_pairs = shown_pairs.join(pair_cohort, on=['community_a', 'community_b'], how='left').with_columns(
            pl.col('shared_cohort_hits').fill_null(0).cast(pl.Int64)
        )
    else:
        shown_pairs = shown_pairs.with_columns(pl.lit(0, dtype=pl.Int64).alias('shared_cohort_hits'))

    part_ids = set(shown_pairs['community_a'].to_list()) | set(shown_pairs['community_b'].to_list())
    part_clusters = clusters.filter(pl.col('community').is_in(sorted(part_ids))).sort('size', 'community', descending=[True, False])
    cid_order = part_clusters['community'].to_list()
    cid_to_y = {cid: idx for idx, cid in enumerate(cid_order)}

    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 3.25], height_ratios=[2.2, 2.1])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_sizes = fig.add_subplot(gs[1, 0])
    ax_matrix = fig.add_subplot(gs[1, 1], sharex=ax_bars, sharey=ax_sizes)

    x_idx = np.arange(shown_pairs.height)
    shared_vals = shown_pairs['shared_proteins'].to_numpy()
    cohort_vals = shown_pairs['shared_cohort_hits'].to_numpy()
    sim_vals = shown_pairs[metric].to_numpy()

    ax_bars.bar(x_idx, shared_vals, color='#3B82F6', alpha=0.85, label='All shared proteins')
    if cohort_set:
        ax_bars.bar(x_idx, cohort_vals, color='#D55E00', alpha=0.95, label=f'Shared {protein_set_label}')
    ymax = max(int(shared_vals.max()), 1)
    for x, total, sim in zip(x_idx, shared_vals, sim_vals):
        ax_bars.text(x, total + ymax * 0.025, f'{total}\n({sim:.2f})', ha='center', va='bottom', fontsize=7.5)
    ax_bars.set_ylim(0, ymax * 1.24)
    ax_bars.set_ylabel('Shared proteins')
    metric_name = 'Jaccard' if metric == 'jaccard' else 'Overlap coef'
    ax_bars.set_title(f'Top {shown_pairs.height} OCG cluster intersections (annotated with count and {metric_name})', fontsize=11)
    ax_bars.legend(loc='upper right', frameon=False, fontsize=8.5)
    ax_bars.spines[['top', 'right']].set_visible(False)
    ax_bars.tick_params(axis='x', labelbottom=False)

    y_idx = np.arange(len(cid_order))
    sizes = part_clusters['size'].to_numpy()
    ax_sizes.barh(y_idx, sizes, color='#64748B', alpha=0.85)
    ax_sizes.invert_xaxis()
    ax_sizes.set_yticks(y_idx, [f'C{c}' for c in cid_order], fontsize=8)
    ax_sizes.yaxis.tick_right()
    ax_sizes.set_xlabel('Cluster size (proteins)')
    ax_sizes.spines[['top', 'left']].set_visible(False)

    # Background grid of inactive dots
    gx, gy = np.meshgrid(x_idx, y_idx)
    ax_matrix.scatter(gx.ravel(), gy.ravel(), s=28, color='#E5E7EB', zorder=1)
    for x, row in enumerate(shown_pairs.iter_rows(named=True)):
        ya = cid_to_y[row['community_a']]
        yb = cid_to_y[row['community_b']]
        ax_matrix.plot([x, x], [ya, yb], color='#1F2937', linewidth=2.0, zorder=2)
        ax_matrix.scatter([x, x], [ya, yb], s=55, color='#1F2937', zorder=3)
    ax_matrix.set_xticks(x_idx, [f"C{r['community_a']}∩C{r['community_b']}" for r in shown_pairs.iter_rows(named=True)], rotation=65, ha='right', fontsize=7.5)
    ax_matrix.tick_params(axis='y', labelleft=False)
    ax_matrix.invert_yaxis()
    ax_matrix.spines[['top', 'right', 'left']].set_visible(False)
    ax_matrix.grid(axis='y', color='#F1F5F9', linewidth=0.6)
    return fig


def _ego_cluster_context(
    protein: str,
    nodes_df: pl.DataFrame,
    memberships: pl.DataFrame | None,
    cohort: set[str] | None,
) -> tuple[dict[str, dict], list[int], dict[int, set[str]]]:
    """Assemble node attributes and per-cluster memberships for an ego neighborhood."""
    cohort_set = set(cohort) if cohort else set()
    node_meta = {}
    for row in nodes_df.iter_rows(named=True):
        p = row['protein']
        node_meta[p] = {
            **row,
            'in_cohort': p in cohort_set,
        }
    center_clusters: list[int] = []
    cluster_members: dict[int, set[str]] = {}
    if memberships is not None and not memberships.is_empty():
        node_names = set(node_meta)
        sub_mem = (
            memberships.filter(pl.col('community').is_not_null() & pl.col('protein').is_in(sorted(node_names)))
            .select('protein', 'community')
            .unique()
        )
        center_clusters = sorted(
            sub_mem.filter(pl.col('protein') == protein)['community'].cast(pl.Int64).to_list()
        )
        for row in sub_mem.iter_rows(named=True):
            cid = int(row['community'])
            cluster_members.setdefault(cid, set()).add(row['protein'])
    return node_meta, center_clusters, cluster_members


def ego_cluster_bipartite_figure(
    protein: str,
    nodes_df: pl.DataFrame,
    sub_edges_df: pl.DataFrame,
    memberships: pl.DataFrame,
    *,
    cohort: set[str] | None = None,
    protein_set_label: str = 'Selected protein set',
) -> go.Figure:
    """Concentric bipartite ego view: center protein -> OCG clusters -> grouped 1-hop neighbors."""
    node_meta, center_clusters, cluster_members = _ego_cluster_context(protein, nodes_df, memberships, cohort)
    fig = go.Figure()
    if not node_meta:
        return fig

    neighbors = [p for p in node_meta if p != protein]
    if not center_clusters:
        # No OCG clusters on center protein: fall back to single-ring star around center.
        pos = {protein: (0.0, 0.0)}
        for idx, nb in enumerate(sorted(neighbors)):
            ang = 2 * math.pi * idx / max(1, len(neighbors))
            pos[nb] = (2.0 * math.cos(ang), 2.0 * math.sin(ang))
    else:
        pos = {protein: (0.0, 0.0)}
        c_colors = cluster_colors(center_clusters)
        cluster_angles = {
            cid: 2 * math.pi * idx / len(center_clusters)
            for idx, cid in enumerate(center_clusters)
        }
        for cid, ang in cluster_angles.items():
            pos[f'C{cid}'] = (1.0 * math.cos(ang), 1.0 * math.sin(ang))

        # Group each neighbor under its first shared center cluster, or 'other'.
        buckets: dict[int | str, list[str]] = {cid: [] for cid in center_clusters}
        buckets['other'] = []
        for nb in sorted(neighbors):
            shared_c = next((cid for cid in center_clusters if nb in cluster_members.get(cid, set())), None)
            buckets[shared_c if shared_c is not None else 'other'].append(nb)

        outer_order: list[str] = []
        for cid in center_clusters:
            outer_order.extend(buckets[cid])
        outer_order.extend(buckets['other'])
        for idx, nb in enumerate(outer_order):
            ang = 2 * math.pi * idx / max(1, len(outer_order))
            pos[nb] = (2.35 * math.cos(ang), 2.35 * math.sin(ang))

        # Protein-cluster membership links
        mx, my = [], []
        for cid in center_clusters:
            cx, cy = pos[f'C{cid}']
            mx.extend([0.0, cx, None])
            my.extend([0.0, cy, None])
            for nb in neighbors:
                if nb in cluster_members.get(cid, set()):
                    nxp, nyp = pos[nb]
                    mx.extend([cx, nxp, None])
                    my.extend([cy, nyp, None])
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode='lines', name='Cluster membership',
            line=dict(width=1.0, color='#CBD5E1', dash='dot'), hoverinfo='skip',
        ))

    # PPI edges
    shared_x, shared_y, cross_x, cross_y = [], [], [], []
    for row in sub_edges_df.iter_rows(named=True):
        a, b = row['a'], row['b']
        if a not in pos or b not in pos:
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        if int(row['shared_clusters']) > 0:
            shared_x.extend([x0, x1, None])
            shared_y.extend([y0, y1, None])
        else:
            cross_x.extend([x0, x1, None])
            cross_y.extend([y0, y1, None])
    if cross_x:
        fig.add_trace(go.Scatter(
            x=cross_x, y=cross_y, mode='lines', name='Interaction (different clusters)',
            line=dict(width=0.9, color='#94A3B8'), opacity=0.45, hoverinfo='skip',
        ))
    if shared_x:
        fig.add_trace(go.Scatter(
            x=shared_x, y=shared_y, mode='lines', name='Interaction (shared OCG cluster)',
            line=dict(width=1.6, color='#3B82F6'), opacity=0.75, hoverinfo='skip',
        ))

    if center_clusters:
        c_colors = cluster_colors(center_clusters)
        fig.add_trace(go.Scatter(
            x=[pos[f'C{cid}'][0] for cid in center_clusters],
            y=[pos[f'C{cid}'][1] for cid in center_clusters],
            mode='markers+text',
            text=[f'C{cid}' for cid in center_clusters],
            textposition='top center',
            name='OCG clusters',
            hovertext=[f'OCG Cluster C{cid} · {len(cluster_members.get(cid, set()))} local members' for cid in center_clusters],
            hoverinfo='text',
            marker=dict(
                size=18, symbol='hexagon',
                color=[c_colors[cid] for cid in center_clusters],
                line=dict(width=1.5, color='#111827'),
            ),
        ))

    # Plot neighbors and center protein
    for is_center, label in ((False, '1-hop neighbor'), (True, f'Center ({protein})')):
        subset = [p for p in node_meta if node_meta[p]['is_center'] == is_center]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[p][0] for p in subset],
            y=[pos[p][1] for p in subset],
            mode='markers+text' if is_center or len(subset) <= 35 else 'markers',
            text=[p if is_center or node_meta[p]['in_cohort'] or node_meta[p]['memberships'] >= 2 else '' for p in subset],
            textposition='top center',
            name=label,
            hovertext=[
                f"{p} | k={node_meta[p]['memberships']} | {protein_set_label}: {'yes' if node_meta[p]['in_cohort'] else 'no'}"
                f" | clusters={node_meta[p]['communities_str']}"
                for p in subset
            ],
            hoverinfo='text',
            marker=dict(
                size=[22 if is_center else (13 if node_meta[p]['memberships'] >= 2 else 9) for p in subset],
                symbol=['star' if is_center else ('diamond' if node_meta[p]['in_cohort'] else 'circle') for p in subset],
                color=['#F39C12' if is_center else ('#0072B2' if node_meta[p]['in_cohort'] else '#64748B') for p in subset],
                line=dict(width=[2.2 if node_meta[p]['memberships'] >= 2 or is_center else 1 for p in subset], color='#111827'),
            ),
        ))

    fig.update_layout(
        title=f'Concentric bipartite ego-cluster map for {protein} ({len(center_clusters)} OCG clusters)',
        height=540,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=20, r=20, t=55, b=20),
        template='plotly_white',
    )
    return fig


def ego_cluster_cytoscapejs_html(
    protein: str,
    nodes_df: pl.DataFrame,
    sub_edges_df: pl.DataFrame,
    memberships: pl.DataFrame,
    *,
    cohort: set[str] | None = None,
    include_clusters: bool = True,
    layout_name: str = 'cose',
    height: int = 560,
) -> str:
    """Interactive Cytoscape.js ego-cluster viewer with optional OCG cluster hub nodes."""
    node_meta, center_clusters, cluster_members = _ego_cluster_context(protein, nodes_df, memberships, cohort)
    node_elements = []
    for p, meta in node_meta.items():
        is_center = bool(meta['is_center'])
        in_cohort = bool(meta['in_cohort'])
        node_elements.append({'data': {
            'id': p,
            'label': p if is_center or in_cohort or meta['memberships'] >= 2 or len(node_meta) <= 40 else '',
            'node_kind': 'Center protein' if is_center else '1-hop neighbor',
            'memberships_k': int(meta['memberships']),
            'clusters_str': meta['communities_str'],
            'in_cohort': in_cohort,
            'is_center': is_center,
            'color': '#F39C12' if is_center else ('#0072B2' if in_cohort else '#64748B'),
            'shape': 'star' if is_center else ('diamond' if in_cohort else 'ellipse'),
            'display_size': 34 if is_center else (24 if meta['memberships'] >= 2 else 18),
            'border_width': 3 if is_center or meta['memberships'] >= 2 else 1.2,
            'border_color': '#D97706' if meta['memberships'] >= 2 else '#1F2937',
        }})

    edge_elements = []
    for idx, row in enumerate(sub_edges_df.iter_rows(named=True)):
        shared = int(row['shared_clusters']) > 0
        edge_elements.append({'data': {
            'id': f'e{idx}',
            'source': row['a'],
            'target': row['b'],
            'shared_clusters': int(row['shared_clusters']),
            'edge_width': 2.2 if shared else 1.1,
            'edge_color': '#3B82F6' if shared else '#CBD5E1',
            'edge_style': 'solid' if shared else 'dotted',
        }})

    if include_clusters and center_clusters:
        c_colors = cluster_colors(center_clusters)
        for cid in center_clusters:
            node_elements.append({'data': {
                'id': f'C{cid}',
                'label': f'C{cid}',
                'node_kind': 'OCG cluster',
                'local_members': len(cluster_members.get(cid, set())),
                'color': c_colors[cid],
                'shape': 'hexagon',
                'display_size': 28,
                'border_width': 2.0,
                'border_color': '#111827',
            }})
            for member in sorted(cluster_members.get(cid, set())):
                edge_elements.append({'data': {
                    'id': f'm_{cid}_{member}',
                    'source': member,
                    'target': f'C{cid}',
                    'relationship': 'OCG membership',
                    'edge_width': 1.0,
                    'edge_color': '#94A3B8',
                    'edge_style': 'dashed',
                }})

    legend = 'Star: center · Blue diamond: in protein set · Blue edge: shared OCG cluster · Hexagon: OCG cluster'
    return _cytoscapejs_document(
        node_elements, edge_elements, layout_name=layout_name, height=height,
        title=f'Ego-cluster view: {protein}', legend_html=legend,
    )


def ego_cluster_echarts_option(
    protein: str,
    nodes_df: pl.DataFrame,
    sub_edges_df: pl.DataFrame,
    memberships: pl.DataFrame,
    *,
    cohort: set[str] | None = None,
    include_clusters: bool = True,
    layout: str = 'force',
) -> dict:
    """Interactive Apache ECharts option for a protein's ego-cluster neighborhood."""
    if layout not in {'force', 'circular'}:
        raise ValueError('layout must be force or circular')
    node_meta, center_clusters, cluster_members = _ego_cluster_context(protein, nodes_df, memberships, cohort)
    categories = [
        {'name': 'Center protein'},
        {'name': 'Shared OCG cluster neighbor'},
        {'name': 'Other 1-hop neighbor'},
    ]
    if include_clusters and center_clusters:
        categories.append({'name': 'OCG cluster'})

    center_set = set(center_clusters)
    data = []
    for p, meta in node_meta.items():
        is_center = bool(meta['is_center'])
        in_cohort = bool(meta['in_cohort'])
        shares = any(p in cluster_members.get(cid, set()) for cid in center_set)
        cat = 0 if is_center else (1 if shares else 2)
        data.append({
            'id': p,
            'name': p,
            'category': cat,
            'value': int(meta['memberships']),
            'symbol': 'pin' if is_center else ('diamond' if in_cohort else 'circle'),
            'symbolSize': 26 if is_center else (14 if meta['memberships'] >= 2 else 10),
            'itemStyle': {
                'color': '#F39C12' if is_center else ('#0072B2' if in_cohort else ('#3B82F6' if shares else '#9CA3AF')),
                'borderColor': '#111827',
                'borderWidth': 2 if is_center or meta['memberships'] >= 2 else 1,
            },
            'label': {'show': is_center or meta['memberships'] >= 2 or len(node_meta) <= 35},
            'tooltip': {'formatter': f"{escape(p)}<br>k={meta['memberships']}<br>Clusters: {escape(meta['communities_str'])}"},
        })

    links = []
    for row in sub_edges_df.iter_rows(named=True):
        shared = int(row['shared_clusters']) > 0
        links.append({
            'source': row['a'],
            'target': row['b'],
            'lineStyle': {
                'width': 1.8 if shared else 0.9,
                'color': '#3B82F6' if shared else '#CBD5E1',
                'type': 'solid' if shared else 'dotted',
                'curveness': 0.2 if layout == 'circular' else 0.0,
            },
        })

    if include_clusters and center_clusters:
        c_colors = cluster_colors(center_clusters)
        for cid in center_clusters:
            data.append({
                'id': f'C{cid}',
                'name': f'C{cid}',
                'category': 3,
                'symbol': 'roundRect',
                'symbolSize': 22,
                'itemStyle': {'color': c_colors[cid], 'borderColor': '#111827', 'borderWidth': 1.5},
                'label': {'show': True},
                'tooltip': {'formatter': f'OCG cluster C{cid}<br>Local members: {len(cluster_members.get(cid, set()))}'},
            })
            for member in sorted(cluster_members.get(cid, set())):
                links.append({
                    'source': member,
                    'target': f'C{cid}',
                    'lineStyle': {'width': 0.9, 'color': '#94A3B8', 'type': 'dashed'},
                })

    series = {
        'type': 'graph',
        'layout': layout,
        'roam': True,
        'draggable': True,
        'categories': categories,
        'data': data,
        'links': links,
        'emphasis': {'focus': 'adjacency', 'label': {'show': True}},
        'label': {'position': 'right'},
    }
    if layout == 'force':
        series['force'] = {'repulsion': 180, 'edgeLength': [45, 130], 'layoutAnimation': False}
    return {'tooltip': {}, 'legend': {'data': [c['name'] for c in categories], 'top': 4}, 'series': [series]}


def ego_cluster_3d_figure(
    protein: str,
    nodes_df: pl.DataFrame,
    sub_edges_df: pl.DataFrame,
    memberships: pl.DataFrame | None = None,
    *,
    cohort: set[str] | None = None,
    protein_set_label: str = 'Selected protein set',
) -> go.Figure:
    """Rotatable 3D Plotly view of a protein's ego-cluster neighborhood."""
    node_meta, _, _ = _ego_cluster_context(protein, nodes_df, memberships, cohort)
    if not node_meta:
        return go.Figure()
    link_df = sub_edges_df.select(
        pl.col('a').alias('source'),
        pl.col('b').alias('target'),
        (pl.col('shared_clusters') + 1).alias('interactions'),
    )
    coords = layout_3d(list(node_meta), link_df)
    fig = go.Figure()
    for is_shared, name, color, width in (
        (False, 'Different OCG clusters', '#CBD5E1', 1.2),
        (True, 'Shared OCG cluster(s)', '#3B82F6', 2.4),
    ):
        ex, ey, ez = [], [], []
        for row in sub_edges_df.iter_rows(named=True):
            if (int(row['shared_clusters']) > 0) != is_shared:
                continue
            a, b = row['a'], row['b']
            ex.extend([coords[a][0], coords[b][0], None])
            ey.extend([coords[a][1], coords[b][1], None])
            ez.extend([coords[a][2], coords[b][2], None])
        if ex:
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez, mode='lines', name=name,
                line=dict(color=color, width=width), opacity=0.65, hoverinfo='skip',
            ))

    plist = list(node_meta)
    fig.add_trace(go.Scatter3d(
        x=[coords[p][0] for p in plist],
        y=[coords[p][1] for p in plist],
        z=[coords[p][2] for p in plist],
        mode='markers+text',
        text=[p if node_meta[p]['is_center'] or node_meta[p]['memberships'] >= 2 else '' for p in plist],
        hovertext=[
            f"{p} | k={node_meta[p]['memberships']} | {protein_set_label}: {'yes' if node_meta[p]['in_cohort'] else 'no'}"
            f" | clusters={node_meta[p]['communities_str']}"
            for p in plist
        ],
        hoverinfo='text',
        name='Proteins',
        marker=dict(
            size=[14 if node_meta[p]['is_center'] else 7 for p in plist],
            symbol=['diamond' if node_meta[p]['in_cohort'] else 'circle' for p in plist],
            color=[node_meta[p]['memberships'] for p in plist],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='Clusters (k)'),
            line=dict(width=1.0, color='#111827'),
        ),
    ))
    if protein in coords:
        fig.add_trace(go.Scatter3d(
            x=[coords[protein][0]], y=[coords[protein][1]], z=[coords[protein][2]],
            mode='markers', name=f'Center ({protein})', hoverinfo='skip',
            marker=dict(size=18, color='rgba(243, 156, 18, 0.22)', line=dict(width=2.5, color='#F39C12')),
        ))
    hidden = dict(visible=False)
    fig.update_layout(
        title=f'3D 1-hop OCG ego network for {protein}',
        height=540,
        margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode='data'),
        template='plotly_white',
    )
    return fig


def ego_cluster_static_figure(
    protein: str,
    nodes_df: pl.DataFrame,
    sub_edges_df: pl.DataFrame,
    memberships: pl.DataFrame | None = None,
    *,
    cohort: set[str] | None = None,
    protein_set_label: str = 'Selected protein set',
):
    """Publication-ready static Matplotlib view of an OCG ego-cluster neighborhood."""
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    node_meta, _, _ = _ego_cluster_context(protein, nodes_df, memberships, cohort)
    fig = Figure(figsize=(10, 8), layout='constrained')
    ax = fig.subplots()
    if not node_meta:
        ax.text(0.5, 0.5, 'No neighborhood nodes to display.', ha='center', va='center')
        ax.set_axis_off()
        return fig

    edges = [(r['a'], r['b'], float(r['shared_clusters'] + 1)) for r in sub_edges_df.iter_rows(named=True)]
    coords = _compute_2d_layout(list(node_meta), edges, engine='NetworkX', layout='Spring')
    cross_segs, shared_segs = [], []
    for r in sub_edges_df.iter_rows(named=True):
        seg = (coords[r['a']], coords[r['b']])
        (shared_segs if int(r['shared_clusters']) > 0 else cross_segs).append(seg)
    if cross_segs:
        ax.add_collection(LineCollection(cross_segs, colors='#CBD5E1', linewidths=0.9, linestyles='dotted', zorder=1))
    if shared_segs:
        ax.add_collection(LineCollection(shared_segs, colors='#3B82F6', linewidths=1.5, alpha=0.8, zorder=1))

    max_k = max([m['memberships'] for m in node_meta.values()], default=1)
    for in_cohort, marker in ((False, 'o'), (True, 'D')):
        subset = [p for p, m in node_meta.items() if m['in_cohort'] == in_cohort]
        if not subset:
            continue
        pts = np.asarray([coords[p] for p in subset])
        sc = ax.scatter(
            pts[:, 0], pts[:, 1],
            s=[340 if node_meta[p]['is_center'] else 110 for p in subset],
            c=[node_meta[p]['memberships'] for p in subset],
            cmap='plasma', vmin=0, vmax=max(1, max_k),
            marker=marker,
            edgecolors=['#F39C12' if node_meta[p]['is_center'] else '#111827' for p in subset],
            linewidths=[2.6 if node_meta[p]['is_center'] else 0.8 for p in subset],
            zorder=2,
        )
    # Colorbar using a scalar mappable so legend & marker splits stay exact.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=max(1, max_k)), cmap='plasma')
    fig.colorbar(sm, ax=ax, shrink=0.72, label='OCG cluster memberships (k)')

    for p, m in node_meta.items():
        if m['is_center'] or m['memberships'] >= 2 or len(node_meta) <= 35:
            ax.annotate(p, coords[p], xytext=(4, 4), textcoords='offset points',
                        fontsize=8 if m['is_center'] else 7,
                        fontweight='bold' if m['is_center'] else 'normal')

    handles = [
        Line2D([], [], color='#3B82F6', linewidth=1.6, label='Shared OCG cluster(s)'),
        Line2D([], [], color='#CBD5E1', linewidth=1.2, linestyle=':', label='No shared OCG cluster'),
        Line2D([], [], marker='D', color='#4B5563', linestyle='', markersize=7, label=protein_set_label),
        Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor='#F39C12',
               markeredgewidth=2.2, linestyle='', markersize=8, label=f'Center ({protein})'),
    ]
    ax.legend(handles=handles, loc='lower left', frameon=True, fontsize=8.5)
    ax.set_title(f'1-hop OCG ego network for {protein} (k={node_meta[protein]["memberships"]})', fontsize=12)
    ax.autoscale()
    ax.margins(0.14)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return fig


def _best_cluster_label_lookup(evidence: pl.DataFrame | None) -> dict[str, dict]:
    """Extract the strongest significant full-cluster GO/Reactome label per cluster."""
    if evidence is None or evidence.is_empty():
        return {}
    needed = {'community', 'subset', 'library', 'term_id', 'term_name', 'qvalue', 'fold_enrichment'}
    if not needed.issubset(evidence.columns):
        return {}
    best = (
        evidence.filter(
            (pl.col('subset') == 'full')
            & (pl.col('qvalue').is_not_null())
            & (pl.col('qvalue') < 0.05)
            & (pl.col('fold_enrichment') > 1.0)
        )
        .sort('qvalue', 'term_id')
        .unique('community', maintain_order=True)
    )
    lookup: dict[str, dict] = {}
    for row in best.iter_rows(named=True):
        qval = float(row['qvalue'])
        sig = significance_label(qval)
        lookup[str(row['community'])] = {
            'term_id': str(row['term_id']),
            'term_name': str(row['term_name']),
            'library': str(row['library']),
            'qvalue': qval,
            'fold_enrichment': float(row['fold_enrichment']),
            'significance': sig,
            'label_text': f"{row['term_name']} {sig}",
        }
    return lookup


def _select_overlap_subgraph(
    clusters: pl.DataFrame,
    pairs: pl.DataFrame,
    *,
    focus: int | str | None = None,
    limit: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """Select up to `limit` clusters (retaining `focus` and its overlaps) and filtered edges."""
    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    limit = max(1, min(300, int(limit)))
    if clusters.is_empty():
        return clusters, pairs.head(0), pairs.head(0), {
            'total_clusters': 0, 'displayed_clusters': 0, 'qualifying_edges': 0,
            'drawn_edges': 0, 'isolated_display_nodes': 0,
        }
    eligible = (
        pairs.filter((pl.col('shared_proteins') >= min_shared) & (pl.col(metric) >= min_similarity))
        if not pairs.is_empty() else pairs
    )
    ordered = clusters.sort(['size', 'community'], descending=[True, False])
    if focus is not None:
        focus_key = str(focus)
        match = ordered.filter(pl.col('community').cast(pl.String) == focus_key)
        if not match.is_empty():
            focus_val = match['community'][0]
            neighbors = (
                eligible.filter((pl.col('community_a') == focus_val) | (pl.col('community_b') == focus_val))
                if not eligible.is_empty() else eligible
            )
            ids = [focus_val]
            for row in neighbors.iter_rows(named=True):
                other = row['community_b'] if row['community_a'] == focus_val else row['community_a']
                if other not in ids:
                    ids.append(other)
            nodes = ordered.filter(pl.col('community').is_in(ids[:limit]))
        else:
            nodes = ordered.head(limit)
    else:
        nodes = ordered.head(limit)

    ids = nodes['community'].to_list()
    within = (
        pairs.filter(pl.col('community_a').is_in(ids) & pl.col('community_b').is_in(ids))
        if not pairs.is_empty() else pairs
    )
    qualifying = (
        eligible.filter(pl.col('community_a').is_in(ids) & pl.col('community_b').is_in(ids))
        if not eligible.is_empty() else eligible
    )
    drawn = qualifying.head(2500)
    incident: set = set()
    if not drawn.is_empty():
        incident.update(drawn['community_a'].to_list())
        incident.update(drawn['community_b'].to_list())
    isolated = sum(1 for cid in ids if cid not in incident)
    stats = {
        'total_clusters': clusters.height,
        'displayed_clusters': len(ids),
        'qualifying_edges': qualifying.height,
        'drawn_edges': drawn.height,
        'isolated_display_nodes': isolated,
    }
    return nodes, drawn, within, stats


def build_annotated_ocg_overlap_view(
    clusters: pl.DataFrame,
    pairs: pl.DataFrame,
    evidence: pl.DataFrame | None = None,
    *,
    focus: int | str | None = None,
    limit: int = 150,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    protein_set_label: str = 'Selected protein set',
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Build standardized cluster-overlap node/edge tables enriched with GO/Reactome & set tests."""
    nodes_raw, drawn, _, stats = _select_overlap_subgraph(
        clusters, pairs, focus=focus, limit=limit,
        min_shared=min_shared, min_similarity=min_similarity, metric=metric,
    )
    if nodes_raw.is_empty():
        return pl.DataFrame(), pl.DataFrame(), stats

    labels = _best_cluster_label_lookup(evidence)
    deg_map: dict[str, int] = {}
    for row in drawn.iter_rows(named=True):
        a = f"C{row['community_a']}"
        b = f"C{row['community_b']}"
        deg_map[a] = deg_map.get(a, 0) + 1
        deg_map[b] = deg_map.get(b, 0) + 1

    vmax = max(1, int(nodes_raw['overlap_neighbors'].max() or 1)) if 'overlap_neighbors' in nodes_raw.columns else 1
    max_size = max(1, int(nodes_raw['size'].max() or 1))
    focus_key = str(focus) if focus is not None else None

    node_rows = []
    for row in nodes_raw.iter_rows(named=True):
        cid = row['community']
        cid_str = str(cid)
        nid = f'C{cid_str}'
        size = int(row['size'])
        overlap_deg = int(row.get('overlap_neighbors') or 0)
        is_focus = focus_key is not None and cid_str == focus_key

        hits = row.get('protein_set_hits')
        fold = row.get('fold_enrichment')
        qval = row.get('qvalue')
        sig = row.get('significance') or (significance_label(qval) if qval is not None else 'n/a')
        is_sig_enriched = bool(qval is not None and math.isfinite(float(qval)) and float(qval) < 0.05 and fold is not None and float(fold) > 1.0)

        ann = labels.get(cid_str)
        top_term = ann['label_text'] if ann else 'No significant saved label'
        top_library = ann['library'] if ann else ''
        tip_lines = [
            f'Cluster {nid} ({size:,} proteins)',
            f'Overlapping clusters (full run): {overlap_deg:,}',
            f'Functional label: {top_term}',
        ]
        if ann:
            tip_lines.append(f"{ann['library']} · q={ann['qvalue']:.3g} {ann['significance']} · fold={ann['fold_enrichment']:.3g}×")
        if hits is not None:
            fold_txt = f'{float(fold):.3g}×' if fold is not None and math.isfinite(float(fold)) else 'n/a'
            q_txt = f'{float(qval):.3g}' if qval is not None and math.isfinite(float(qval)) else 'n/a'
            tip_lines.append(f'{protein_set_label} hits: {int(hits)}/{size} · fold={fold_txt} · q={q_txt} {sig}')

        if is_focus:
            node_role = 'Linker'
            fill = '#db6b35'
            border = '#111827'
            bwidth = 3.5
        elif is_sig_enriched:
            node_role = 'Moonlighting'
            fill = '#0072B2'
            border = '#F39C12'
            bwidth = 3.0
        else:
            node_role = 'Background'
            fill = _viridis_hex(overlap_deg, vmax)
            border = '#1F2937'
            bwidth = 1.4

        node_rows.append({
            'id': nid,
            'protein': nid,
            'name': nid,
            'label': f'{nid} {sig}' if is_sig_enriched else nid,
            'node_kind': 'Focus OCG cluster' if is_focus else (f'{protein_set_label}-enriched cluster' if is_sig_enriched else 'OCG cluster'),
            'node_role': node_role,
            'community': cid_str,
            'size': size,
            'subnetwork_degree': max(1, deg_map.get(nid, 0)),
            'overlap_neighbors': overlap_deg,
            'total_shared_links': int(row.get('total_shared_links') or 0),
            'protein_set_hits': int(hits) if hits is not None else 0,
            'fold_enrichment': float(fold) if fold is not None and math.isfinite(float(fold)) else None,
            'qvalue': float(qval) if qval is not None and math.isfinite(float(qval)) else None,
            'significance': sig,
            'top_term': top_term,
            'top_library': top_library,
            'display_size': round(16 + 34 * math.sqrt(size / max_size), 1),
            'color': fill,
            'border_color': border,
            'border_width': bwidth,
            'shape': 'hexagon' if is_focus else ('diamond' if is_sig_enriched else 'ellipse'),
            'is_moonlighting': is_sig_enriched,
            'cluster_selected': bool(is_focus or is_sig_enriched),
            'has_cohort': bool(hits is not None and int(hits) > 0),
            'protein_set': f'{size:,} proteins · {overlap_deg:,} overlaps · {top_term}',
            'show_label': bool(is_focus or is_sig_enriched or nodes_raw.height <= 45),
            'tooltip': '\n'.join(tip_lines),
        })

    edge_rows = []
    for row in drawn.iter_rows(named=True):
        src = f"C{row['community_a']}"
        tgt = f"C{row['community_b']}"
        shared = int(row['shared_proteins'])
        jac = float(row['jaccard'])
        oc = float(row['overlap_coef'])
        edge_rows.append({
            'source': src,
            'target': tgt,
            'interaction': 'shares_proteins',
            'edge_class': 'Shared',
            'score': float(row[metric]),
            'interactions': max(1, shared),
            'shared_proteins': shared,
            'jaccard': round(jac, 4),
            'overlap_coef': round(oc, 4),
            'edge_width': round(min(7.0, 1.0 + math.sqrt(shared)), 2),
            'edge_color': '#64748B',
            'edge_style': 'solid',
            'tooltip': f'{src} ↔ {tgt}\nShared proteins: {shared:,}\nJaccard: {jac:.3f}\nOverlap coef: {oc:.3f}',
        })

    nodes_df = pl.from_dicts(node_rows, infer_schema_length=None)
    edges_df = (
        pl.from_dicts(edge_rows, infer_schema_length=None)
        if edge_rows else pl.DataFrame(schema={
            'source': pl.String, 'target': pl.String, 'interaction': pl.String,
            'edge_class': pl.String, 'score': pl.Float64, 'interactions': pl.Int64,
            'shared_proteins': pl.Int64, 'jaccard': pl.Float64, 'overlap_coef': pl.Float64,
            'edge_width': pl.Float64, 'edge_color': pl.String, 'edge_style': pl.String, 'tooltip': pl.String,
        })
    )
    return nodes_df, edges_df, stats


def build_annotated_ocg_bipartite_view(
    clusters: pl.DataFrame,
    pairs: pl.DataFrame,
    memberships: pl.DataFrame,
    evidence: pl.DataFrame | None = None,
    *,
    cohort: set[str] | None = None,
    focus: int | str | None = None,
    limit: int = 60,
    max_bridges: int = 200,
    min_shared: int = 1,
    min_similarity: float = 0.0,
    metric: str = 'jaccard',
    metadata: dict | None = None,
    protein_set_label: str = 'Selected protein set',
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Build bipartite OCG network linking clusters to multi-cluster bridge proteins (k≥2)."""
    nodes_raw, _, _, stats = _select_overlap_subgraph(
        clusters, pairs, focus=focus, limit=min(120, max(1, int(limit))),
        min_shared=min_shared, min_similarity=min_similarity, metric=metric,
    )
    if nodes_raw.is_empty():
        return pl.DataFrame(), pl.DataFrame(), {**stats, 'bridge_proteins': 0, 'membership_links': 0}

    cluster_ids = nodes_raw['community'].to_list()
    cluster_strs = [str(c) for c in cluster_ids]
    cohort_set = set(cohort) if cohort else set()
    labels = _best_cluster_label_lookup(evidence)
    focus_key = str(focus) if focus is not None else None

    assigned = (
        memberships.filter(pl.col('community').is_not_null())
        .with_columns(pl.col('community').cast(pl.String).alias('_cstr'))
        .select('protein', '_cstr')
        .unique()
    )
    full_k_map = dict(assigned.group_by('protein').agg(pl.len().alias('k')).iter_rows())
    displayed_links = assigned.filter(pl.col('_cstr').is_in(cluster_strs))

    if displayed_links.is_empty():
        bridges = pl.DataFrame()
        bridge_links = pl.DataFrame(schema={'protein': pl.String, '_cstr': pl.String})
    else:
        bridge_agg = (
            displayed_links.group_by('protein')
            .agg(
                pl.col('_cstr').n_unique().alias('displayed_clusters'),
                pl.col('_cstr').sort().str.join(';').alias('clusters_str'),
                (pl.col('_cstr') == focus_key).any().alias('_touches_focus') if focus_key is not None else pl.lit(False).alias('_touches_focus'),
            )
            .filter(pl.col('displayed_clusters') >= 2)
            .with_columns(pl.col('protein').is_in(sorted(cohort_set)).alias('in_cohort'))
            .sort(['_touches_focus', 'in_cohort', 'displayed_clusters', 'protein'], descending=[True, True, True, False])
            .head(max(1, min(400, int(max_bridges))))
        )
        bridges = bridge_agg
        bridge_links = displayed_links.join(bridges.select('protein'), on='protein', how='semi')

    deg_map: dict[str, int] = {}
    for row in bridge_links.iter_rows(named=True):
        p = row['protein']
        c = f"C{row['_cstr']}"
        deg_map[p] = deg_map.get(p, 0) + 1
        deg_map[c] = deg_map.get(c, 0) + 1

    max_size = max(1, int(nodes_raw['size'].max() or 1))
    node_rows = []
    for row in nodes_raw.iter_rows(named=True):
        cid_str = str(row['community'])
        nid = f'C{cid_str}'
        size = int(row['size'])
        is_focus = focus_key is not None and cid_str == focus_key
        qval = row.get('qvalue')
        fold = row.get('fold_enrichment')
        sig = row.get('significance') or (significance_label(qval) if qval is not None else 'n/a')
        is_sig = bool(qval is not None and math.isfinite(float(qval)) and float(qval) < 0.05 and fold is not None and float(fold) > 1.0)
        ann = labels.get(cid_str)
        top_term = ann['label_text'] if ann else 'No significant saved label'

        tip = f'OCG Cluster {nid} ({size:,} proteins)\nFunctional label: {top_term}'
        if row.get('protein_set_hits') is not None:
            tip += f"\n{protein_set_label}: {row['protein_set_hits']}/{size} ({sig})"
        node_rows.append({
            'id': nid,
            'protein': nid,
            'name': nid,
            'label': nid,
            'node_kind': 'OCG cluster',
            'node_role': 'Linker' if is_focus else ('Moonlighting' if is_sig else 'Background'),
            'size': size,
            'subnetwork_degree': max(1, deg_map.get(nid, 0)),
            'display_size': round(20 + 28 * math.sqrt(size / max_size), 1),
            'color': '#db6b35' if is_focus else '#327DA8',
            'border_color': '#F39C12' if is_focus or is_sig else '#1F2937',
            'border_width': 3.2 if is_focus or is_sig else 1.6,
            'shape': 'hexagon',
            'is_moonlighting': False,
            'cluster_selected': bool(is_focus or is_sig),
            'in_cohort': False,
            'has_cohort': bool((row.get('protein_set_hits') or 0) > 0),
            'clusters_str': cid_str,
            'protein_set': f'Cluster {size:,} proteins · {top_term}',
            'show_label': True,
            'tooltip': tip,
        })

    for row in bridges.iter_rows(named=True):
        p = row['protein']
        in_cohort = bool(row['in_cohort'])
        full_k = int(full_k_map.get(p, row['displayed_clusters']))
        ident = protein_identity(p, metadata) if metadata else {'label': p, 'tooltip': f'UniProt: {p}'}
        short_label = ident['label']
        tip = f"{ident['tooltip']}\nFull-run OCG memberships (k): {full_k}\nDisplayed clusters: {row['clusters_str']}\n{protein_set_label}: {'yes' if in_cohort else 'no'}"
        node_rows.append({
            'id': p,
            'protein': p,
            'name': p,
            'label': short_label,
            'node_kind': f'{protein_set_label} bridge (k≥2)' if in_cohort else 'Bridge protein (k≥2)',
            'node_role': 'Moonlighting' if in_cohort else 'Background',
            'size': max(2, full_k),
            'subnetwork_degree': max(1, deg_map.get(p, 0)),
            'display_size': round(12 + 3.5 * min(6, int(row['displayed_clusters'])), 1),
            'color': '#0072B2' if in_cohort else '#94A3B8',
            'border_color': '#F39C12' if in_cohort else '#334155',
            'border_width': 2.6 if in_cohort else 1.2,
            'shape': 'diamond' if in_cohort else 'ellipse',
            'is_moonlighting': in_cohort,
            'cluster_selected': True,
            'in_cohort': in_cohort,
            'has_cohort': in_cohort,
            'clusters_str': row['clusters_str'],
            'protein_set': f'Bridge k={full_k} ({row["displayed_clusters"]} shown)',
            'show_label': bool(in_cohort or bridges.height <= 40),
            'tooltip': tip,
        })

    cohort_bridge_set = set(bridges.filter(pl.col('in_cohort'))['protein'].to_list()) if not bridges.is_empty() else set()
    edge_rows = []
    for row in bridge_links.iter_rows(named=True):
        p = row['protein']
        c = f"C{row['_cstr']}"
        in_c = p in cohort_bridge_set
        edge_rows.append({
            'source': p,
            'target': c,
            'interaction': 'belongs_to_cluster',
            'edge_class': 'M-Linker' if in_c else 'Shared',
            'score': 1.0,
            'interactions': 1,
            'edge_width': 1.5 if in_c else 1.0,
            'edge_color': '#0072B2' if in_c else '#94A3B8',
            'edge_style': 'dashed',
            'tooltip': f'{p} belongs to {c}',
        })

    nodes_df = pl.from_dicts(node_rows, infer_schema_length=None)
    edges_df = (
        pl.from_dicts(edge_rows, infer_schema_length=None)
        if edge_rows else pl.DataFrame(schema={
            'source': pl.String, 'target': pl.String, 'interaction': pl.String,
            'edge_class': pl.String, 'score': pl.Float64, 'interactions': pl.Int64,
            'edge_width': pl.Float64, 'edge_color': pl.String, 'edge_style': pl.String, 'tooltip': pl.String,
        })
    )
    bipartite_stats = {
        **stats,
        'displayed_clusters': nodes_raw.height,
        'bridge_proteins': bridges.height,
        'membership_links': edges_df.height,
    }
    return nodes_df, edges_df, bipartite_stats


def build_ocg_cluster_ppi_view(
    community: int | str,
    memberships: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    cohort: set[str] | None = None,
    metadata: dict | None = None,
    max_nodes: int = 400,
    max_edges: int = 4000,
    focus_protein: str = '',
    protein_set_label: str = 'Selected protein set',
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Build intra-cluster PPI subnetwork contrasting Core (k=1) vs Interface (k≥2) proteins."""
    cid_str = str(community)
    cohort_set = set(cohort) if cohort else set()
    assigned = (
        memberships.filter(pl.col('community').is_not_null())
        .with_columns(pl.col('community').cast(pl.String).alias('_cstr'))
        .select('protein', '_cstr')
        .unique()
    )
    members = sorted(assigned.filter(pl.col('_cstr') == cid_str)['protein'].unique().to_list())
    if not members:
        return pl.DataFrame(), pl.DataFrame(), {
            'full_nodes': 0, 'full_edges': 0, 'displayed_nodes': 0, 'displayed_edges': 0,
            'core_proteins': 0, 'interface_proteins': 0, 'cohort_proteins': 0,
        }

    k_info = dict(
        assigned.filter(pl.col('protein').is_in(members))
        .group_by('protein')
        .agg(
            pl.len().alias('k'),
            pl.col('_cstr').sort().str.join(';').alias('clusters_str'),
        )
        .select('protein', pl.struct('k', 'clusters_str'))
        .iter_rows()
    )

    dedup = _deduplicated_edges(edges)
    internal = dedup.filter(pl.col('a').is_in(members) & pl.col('b').is_in(members))
    full_nodes = len(members)
    full_edges = internal.height

    focus_clean = focus_protein.strip().upper()
    if focus_clean and focus_clean in members:
        incident = internal.filter((pl.col('a') == focus_clean) | (pl.col('b') == focus_clean))
        keep_set = set(incident['a'].to_list()) | set(incident['b'].to_list()) | {focus_clean}
        members = sorted(keep_set)
        internal = internal.filter(pl.col('a').is_in(members) & pl.col('b').is_in(members))

    deg_series = (
        pl.concat([internal.select(pl.col('a').alias('protein')), internal.select(pl.col('b').alias('protein'))])
        .group_by('protein')
        .len(name='deg')
    ) if not internal.is_empty() else pl.DataFrame(schema={'protein': pl.String, 'deg': pl.Int64})
    deg_map = dict(deg_series.iter_rows())

    ranked = sorted(
        members,
        key=lambda p: (
            p == focus_clean,
            p in cohort_set,
            k_info.get(p, {}).get('k', 1) >= 2,
            deg_map.get(p, 0),
            p,
        ),
        reverse=True,
    )
    shown_members = ranked[:max(1, min(2000, int(max_nodes)))]
    shown_set = set(shown_members)
    shown_edges = (
        internal.filter(pl.col('a').is_in(shown_set) & pl.col('b').is_in(shown_set))
        .sort('a', 'b')
        .head(max(1, min(20000, int(max_edges))))
    )

    shown_deg: dict[str, int] = {}
    for a, b in shown_edges.select('a', 'b').iter_rows():
        shown_deg[a] = shown_deg.get(a, 0) + 1
        shown_deg[b] = shown_deg.get(b, 0) + 1

    core_count = 0
    interface_count = 0
    cohort_count = 0
    node_rows = []
    for p in shown_members:
        meta_k = k_info.get(p, {'k': 1, 'clusters_str': cid_str})
        k_val = int(meta_k['k'])
        c_str = str(meta_k['clusters_str'])
        is_interface = k_val >= 2
        in_cohort = p in cohort_set
        is_focus = p == focus_clean
        if is_interface:
            interface_count += 1
        else:
            core_count += 1
        if in_cohort:
            cohort_count += 1

        ident = protein_identity(p, metadata) if metadata else {'label': p, 'tooltip': f'UniProt: {p}'}
        ocg_role = 'Interface (k≥2)' if is_interface else 'Core (k=1)'
        deg = int(deg_map.get(p, 0))
        tip = (
            f"{ident['tooltip']}\nCluster role in C{cid_str}: {ocg_role}\n"
            f"Full-run OCG memberships (k): {k_val} ({c_str})\n"
            f"Internal cluster PPI degree: {deg}\n{protein_set_label}: {'yes' if in_cohort else 'no'}"
        )
        if in_cohort:
            fill = '#0072B2'
            shape = 'diamond'
            node_role = 'Moonlighting'
        elif is_interface:
            fill = '#D55E00'
            shape = 'hexagon'
            node_role = 'Linker'
        else:
            fill = '#009E73'
            shape = 'ellipse'
            node_role = 'Background'

        node_rows.append({
            'id': p,
            'protein': p,
            'name': p,
            'label': ident['label'],
            'node_kind': f'{protein_set_label} · {ocg_role}' if in_cohort else ocg_role,
            'node_role': node_role,
            'ocg_role': ocg_role,
            'memberships_k': k_val,
            'clusters_str': c_str,
            'size': max(1, deg + 1),
            'subnetwork_degree': max(1, shown_deg.get(p, 0)),
            'display_size': round(15 + 3.5 * math.sqrt(deg) + (5 if is_focus else 0), 1),
            'color': fill,
            'border_color': '#111827' if is_focus else ('#F39C12' if is_interface else '#1F2937'),
            'border_width': 3.5 if is_focus else (2.6 if is_interface or in_cohort else 1.2),
            'shape': 'star' if is_focus else shape,
            'is_moonlighting': in_cohort,
            'cluster_selected': bool(is_interface or is_focus),
            'in_cohort': in_cohort,
            'has_cohort': in_cohort,
            'protein_set': f'{ocg_role} · k={k_val} · deg={deg}',
            'show_label': bool(is_focus or in_cohort or is_interface or len(shown_members) <= 35),
            'tooltip': tip,
        })

    edge_rows = []
    interface_edges = 0
    core_interface_edges = 0
    core_edges = 0
    for a, b in shown_edges.select('a', 'b').iter_rows():
        a_int = int(k_info.get(a, {}).get('k', 1)) >= 2
        b_int = int(k_info.get(b, {}).get('k', 1)) >= 2
        if a_int and b_int:
            etype, eclass, ecolor, ewidth = 'Interface–Interface', 'M-M', '#D55E00', 2.0
            interface_edges += 1
        elif a_int or b_int:
            etype, eclass, ecolor, ewidth = 'Core–Interface', 'M-Linker', '#3B82F6', 1.5
            core_interface_edges += 1
        else:
            etype, eclass, ecolor, ewidth = 'Core–Core', 'B-B', '#94A3B8', 1.1
            core_edges += 1
        edge_rows.append({
            'source': a,
            'target': b,
            'interaction': etype,
            'edge_class': eclass,
            'score': 1.0,
            'interactions': 1,
            'edge_width': ewidth,
            'edge_color': ecolor,
            'edge_style': 'solid',
            'tooltip': f'{a} ↔ {b}\nInternal PPI: {etype}',
        })

    nodes_df = pl.from_dicts(node_rows, infer_schema_length=None)
    edges_df = (
        pl.from_dicts(edge_rows, infer_schema_length=None)
        if edge_rows else pl.DataFrame(schema={
            'source': pl.String, 'target': pl.String, 'interaction': pl.String,
            'edge_class': pl.String, 'score': pl.Float64, 'interactions': pl.Int64,
            'edge_width': pl.Float64, 'edge_color': pl.String, 'edge_style': pl.String, 'tooltip': pl.String,
        })
    )
    stats = {
        'full_nodes': full_nodes,
        'full_edges': full_edges,
        'displayed_nodes': nodes_df.height,
        'displayed_edges': edges_df.height,
        'core_proteins': core_count,
        'interface_proteins': interface_count,
        'cohort_proteins': cohort_count,
        'interface_edges': interface_edges,
        'core_interface_edges': core_interface_edges,
        'core_edges': core_edges,
    }
    return nodes_df, edges_df, stats


def build_ocg_pair_ppi_view(
    community_a: int | str,
    community_b: int | str,
    memberships: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    cohort: set[str] | None = None,
    metadata: dict | None = None,
    max_nodes: int = 500,
    max_edges: int = 5000,
    protein_set_label: str = 'Selected protein set',
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Build induced PPI subnetwork across two overlapping OCG clusters (shared interface vs exclusive)."""
    ca, cb = str(community_a), str(community_b)
    cohort_set = set(cohort) if cohort else set()
    assigned = (
        memberships.filter(pl.col('community').is_not_null())
        .with_columns(pl.col('community').cast(pl.String).alias('_cstr'))
        .select('protein', '_cstr')
        .unique()
    )
    mem_a = set(assigned.filter(pl.col('_cstr') == ca)['protein'].to_list())
    mem_b = set(assigned.filter(pl.col('_cstr') == cb)['protein'].to_list())
    union = sorted(mem_a | mem_b)
    if not union:
        return pl.DataFrame(), pl.DataFrame(), {
            'full_nodes': 0, 'full_edges': 0, 'displayed_nodes': 0, 'displayed_edges': 0,
            'shared_proteins': 0, 'only_a_proteins': 0, 'only_b_proteins': 0,
            'cross_cluster_edges': 0, 'shared_interface_edges': 0,
        }

    shared = mem_a & mem_b
    only_a = mem_a - mem_b
    only_b = mem_b - mem_a

    full_k_map = dict(
        assigned.filter(pl.col('protein').is_in(union))
        .group_by('protein').agg(pl.len().alias('k')).iter_rows()
    )
    dedup = _deduplicated_edges(edges)
    pair_edges = dedup.filter(pl.col('a').is_in(union) & pl.col('b').is_in(union))
    full_nodes = len(union)
    full_edges = pair_edges.height

    deg_series = (
        pl.concat([pair_edges.select(pl.col('a').alias('protein')), pair_edges.select(pl.col('b').alias('protein'))])
        .group_by('protein').len(name='deg')
    ) if not pair_edges.is_empty() else pl.DataFrame(schema={'protein': pl.String, 'deg': pl.Int64})
    deg_map = dict(deg_series.iter_rows())

    ranked = sorted(
        union,
        key=lambda p: (p in shared, p in cohort_set, deg_map.get(p, 0), p),
        reverse=True,
    )
    shown_members = ranked[:max(1, min(2000, int(max_nodes)))]
    shown_set = set(shown_members)
    shown_edges = (
        pair_edges.filter(pl.col('a').is_in(shown_set) & pl.col('b').is_in(shown_set))
        .sort('a', 'b')
        .head(max(1, min(20000, int(max_edges))))
    )
    shown_deg: dict[str, int] = {}
    for a, b in shown_edges.select('a', 'b').iter_rows():
        shown_deg[a] = shown_deg.get(a, 0) + 1
        shown_deg[b] = shown_deg.get(b, 0) + 1

    node_rows = []
    for p in shown_members:
        in_cohort = p in cohort_set
        is_shared = p in shared
        in_a = p in only_a
        deg = int(deg_map.get(p, 0))
        k_val = int(full_k_map.get(p, 1))
        ident = protein_identity(p, metadata) if metadata else {'label': p, 'tooltip': f'UniProt: {p}'}

        if is_shared:
            group_role = f'Shared interface (C{ca}∩C{cb})'
            fill = '#D97706'
            shape = 'diamond' if in_cohort else 'hexagon'
            node_role = 'Moonlighting' if in_cohort else 'Linker'
        elif in_a:
            group_role = f'C{ca} only'
            fill = '#327DA8'
            shape = 'diamond' if in_cohort else 'ellipse'
            node_role = 'Moonlighting' if in_cohort else 'Background'
        else:
            group_role = f'C{cb} only'
            fill = '#009E73'
            shape = 'diamond' if in_cohort else 'ellipse'
            node_role = 'Moonlighting' if in_cohort else 'Background'

        tip = (
            f"{ident['tooltip']}\nPair partition: {group_role}\n"
            f"Full-run OCG memberships (k): {k_val}\nPair-subgraph PPI degree: {deg}\n"
            f"{protein_set_label}: {'yes' if in_cohort else 'no'}"
        )
        node_rows.append({
            'id': p,
            'protein': p,
            'name': p,
            'label': ident['label'],
            'node_kind': f'{protein_set_label} · {group_role}' if in_cohort else group_role,
            'node_role': node_role,
            'ocg_role': group_role,
            'memberships_k': k_val,
            'clusters_str': f'{ca};{cb}' if is_shared else (ca if in_a else cb),
            'size': max(1, deg + 1),
            'subnetwork_degree': max(1, shown_deg.get(p, 0)),
            'display_size': round(15 + 3.5 * math.sqrt(deg) + (5 if is_shared else 0), 1),
            'color': fill,
            'border_color': '#003366' if in_cohort else '#1F2937',
            'border_width': 3.2 if in_cohort or is_shared else 1.2,
            'shape': shape,
            'is_moonlighting': in_cohort,
            'cluster_selected': bool(is_shared),
            'in_cohort': in_cohort,
            'has_cohort': in_cohort,
            'protein_set': f'{group_role} · k={k_val} · deg={deg}',
            'show_label': bool(is_shared or in_cohort or len(shown_members) <= 35),
            'tooltip': tip,
        })

    edge_rows = []
    cross_edges = 0
    shared_int_edges = 0
    for a, b in shown_edges.select('a', 'b').iter_rows():
        if a in shared or b in shared:
            etype, eclass, ecolor, estyle, ewidth = 'Shared interface PPI', 'M-Linker', '#D97706', 'solid', 2.0
            shared_int_edges += 1
        elif a in only_a and b in only_a:
            etype, eclass, ecolor, estyle, ewidth = f'Within C{ca}', 'B-B', '#327DA8', 'solid', 1.3
        elif a in only_b and b in only_b:
            etype, eclass, ecolor, estyle, ewidth = f'Within C{cb}', 'B-B', '#009E73', 'solid', 1.3
        else:
            etype, eclass, ecolor, estyle, ewidth = 'Cross-cluster PPI', 'M-M', '#7C3AED', 'dashed', 1.8
            cross_edges += 1
        edge_rows.append({
            'source': a,
            'target': b,
            'interaction': etype,
            'edge_class': eclass,
            'score': 1.0,
            'interactions': 1,
            'edge_width': ewidth,
            'edge_color': ecolor,
            'edge_style': estyle,
            'tooltip': f'{a} ↔ {b}\n{etype}',
        })

    nodes_df = pl.from_dicts(node_rows, infer_schema_length=None)
    edges_df = (
        pl.from_dicts(edge_rows, infer_schema_length=None)
        if edge_rows else pl.DataFrame(schema={
            'source': pl.String, 'target': pl.String, 'interaction': pl.String,
            'edge_class': pl.String, 'score': pl.Float64, 'interactions': pl.Int64,
            'edge_width': pl.Float64, 'edge_color': pl.String, 'edge_style': pl.String, 'tooltip': pl.String,
        })
    )
    stats = {
        'full_nodes': full_nodes,
        'full_edges': full_edges,
        'displayed_nodes': nodes_df.height,
        'displayed_edges': edges_df.height,
        'shared_proteins': len(shared),
        'only_a_proteins': len(only_a),
        'only_b_proteins': len(only_b),
        'cross_cluster_edges': cross_edges,
        'shared_interface_edges': shared_int_edges,
    }
    return nodes_df, edges_df, stats


def render_ocg_view_plotly2d(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    title: str = 'OCG network view',
    engine: str = 'NetworkX',
    layout: str = 'Spring',
    show_labels: bool = True,
) -> go.Figure:
    """Render any standardized OCG (nodes_df, edges_df) view as a 2D interactive Plotly graph."""
    fig = go.Figure()
    if nodes_df.is_empty():
        fig.update_layout(title='No nodes to display', height=480)
        return fig

    node_ids = nodes_df['id'].to_list()
    edge_tuples = [
        (r['source'], r['target'], float(r.get('score') or 1.0))
        for r in edges_df.iter_rows(named=True)
    ] if not edges_df.is_empty() else []
    coords = _compute_2d_layout(node_ids, edge_tuples, engine=engine, layout=layout)

    if not edges_df.is_empty():
        dash_map = {'solid': 'solid', 'dashed': 'dash', 'dotted': 'dot'}
        for (ecolor, estyle, interaction), group in edges_df.group_by(
            ['edge_color', 'edge_style', 'interaction'], maintain_order=True
        ):
            ex, ey = [], []
            for r in group.iter_rows(named=True):
                if r['source'] in coords and r['target'] in coords:
                    x0, y0 = coords[r['source']]
                    x1, y1 = coords[r['target']]
                    ex.extend([x0, x1, None])
                    ey.extend([y0, y1, None])
            if ex:
                mean_w = float(group['edge_width'].mean() or 1.2)
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode='lines', name=str(interaction),
                    line=dict(width=max(0.8, min(3.5, mean_w)), color=str(ecolor), dash=dash_map.get(str(estyle), 'solid')),
                    opacity=0.65, hoverinfo='skip',
                ))

    sym_map = {'diamond': 'diamond', 'hexagon': 'hexagon', 'star': 'star', 'roundRect': 'square', 'ellipse': 'circle'}
    for kind, group in nodes_df.group_by('node_kind', maintain_order=True):
        ids_g = group['id'].to_list()
        pts = [coords[i] for i in ids_g]
        labels = [
            (escape(str(r['label'])) if (show_labels and r['show_label']) or r['cluster_selected'] else '')
            for r in group.iter_rows(named=True)
        ]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            mode='markers+text' if any(labels) else 'markers',
            text=labels,
            textposition='top center',
            name=str(kind),
            hovertext=[escape(str(r['tooltip'])).replace('\n', '<br>') for r in group.iter_rows(named=True)],
            hoverinfo='text',
            marker=dict(
                size=[max(8.0, float(r['display_size']) * 0.55) for r in group.iter_rows(named=True)],
                color=[str(r['color']) for r in group.iter_rows(named=True)],
                symbol=[sym_map.get(str(r['shape']), 'circle') for r in group.iter_rows(named=True)],
                line=dict(
                    width=[float(r['border_width']) * 0.8 for r in group.iter_rows(named=True)],
                    color=[str(r['border_color']) for r in group.iter_rows(named=True)],
                ),
            ),
        ))

    fig.update_layout(
        title=f'{title} ({engine} · {layout})',
        height=620,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation='h', y=-0.04),
        template='plotly_white',
    )
    return fig


def render_ocg_view_echarts(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    layout: str = 'force',
) -> dict:
    """Render any standardized OCG (nodes_df, edges_df) view as an Apache ECharts graph option."""
    if layout not in {'force', 'circular'}:
        raise ValueError('layout must be force or circular')
    if nodes_df.is_empty():
        return {'title': {'text': 'No nodes to display'}, 'series': []}

    kinds = nodes_df['node_kind'].unique(maintain_order=True).to_list()
    cat_index = {str(k): i for i, k in enumerate(kinds)}
    categories = [{'name': str(k)} for k in kinds]
    sym_map = {'diamond': 'diamond', 'hexagon': 'roundRect', 'star': 'pin', 'roundRect': 'roundRect', 'ellipse': 'circle'}

    data = []
    for row in nodes_df.iter_rows(named=True):
        lbl = str(row['label'])
        data.append({
            'id': str(row['id']),
            'name': str(row['id']),
            'category': cat_index[str(row['node_kind'])],
            'value': int(row['size']),
            'symbol': sym_map.get(str(row['shape']), 'circle'),
            'symbolSize': max(8.0, min(55.0, float(row['display_size']) * 0.75)),
            'itemStyle': {
                'color': str(row['color']),
                'borderColor': str(row['border_color']),
                'borderWidth': float(row['border_width']),
            },
            'label': {
                'show': bool(row['show_label']),
                'formatter': lbl,
            },
            'tooltip': {'formatter': escape(str(row['tooltip'])).replace('\n', '<br>')},
        })

    links = []
    if not edges_df.is_empty():
        for row in edges_df.iter_rows(named=True):
            links.append({
                'source': str(row['source']),
                'target': str(row['target']),
                'value': float(row.get('score') or 1.0),
                'lineStyle': {
                    'width': float(row['edge_width']),
                    'color': str(row['edge_color']),
                    'type': 'dashed' if row['edge_style'] == 'dashed' else ('dotted' if row['edge_style'] == 'dotted' else 'solid'),
                    'curveness': 0.2 if layout == 'circular' else 0.0,
                },
                'tooltip': {'formatter': escape(str(row['tooltip'])).replace('\n', '<br>')},
            })

    series = {
        'type': 'graph',
        'layout': layout,
        'roam': True,
        'draggable': True,
        'categories': categories,
        'data': data,
        'links': links,
        'emphasis': {'focus': 'adjacency', 'label': {'show': True}},
        'lineStyle': {'opacity': 0.55},
        'label': {'position': 'right'},
    }
    if layout == 'force':
        series['force'] = {'repulsion': 200, 'edgeLength': [50, 160], 'layoutAnimation': False}
    else:
        series['circular'] = {'rotateLabel': True}

    return {'tooltip': {}, 'legend': {'data': [c['name'] for c in categories], 'top': 4}, 'series': [series]}


def render_ocg_view_cytoscapejs(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    layout_name: str = 'cose',
    height: int = 620,
    title: str = 'OCG network view',
    legend_html: str = '',
) -> str:
    """Render any standardized OCG (nodes_df, edges_df) view as a self-contained Cytoscape.js widget."""
    if nodes_df.is_empty():
        return _cytoscapejs_document([], [], layout_name=layout_name, height=height, title=title, legend_html=legend_html)

    skip_node = {'protein', 'name', 'show_label', 'tooltip'}
    node_elements = []
    for row in nodes_df.iter_rows(named=True):
        data = {k: v for k, v in row.items() if k not in skip_node and v is not None}
        data['id'] = str(row['id'])
        data['label'] = str(row['label']) if row['show_label'] or row['cluster_selected'] else ''
        node_elements.append({'data': data})

    skip_edge = {'tooltip'}
    edge_elements = []
    if not edges_df.is_empty():
        for idx, row in enumerate(edges_df.iter_rows(named=True)):
            data = {k: v for k, v in row.items() if k not in skip_edge and v is not None}
            data['id'] = f'e{idx}'
            data['source'] = str(row['source'])
            data['target'] = str(row['target'])
            edge_elements.append({'data': data})

    return _cytoscapejs_document(
        node_elements, edge_elements, layout_name=layout_name,
        height=height, title=title, legend_html=legend_html,
    )


def render_ocg_view_sigma(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    engine: str = 'NetworkX Spring',
    title: str = 'OCG network view',
) -> tuple[str, pl.DataFrame]:
    """Render any standardized OCG view in offline Sigma.js 3 WebGL; return (html, nodes_with_xy)."""
    if nodes_df.is_empty():
        empty_xy = nodes_df.with_columns(pl.lit(0.0).alias('x'), pl.lit(0.0).alias('y')) if nodes_df.width else pl.DataFrame()
        payload = json.dumps({'nodes': [], 'edges': [], 'title': title}).replace('<', r'\u003c')
        assets = Path(__file__).parent / 'assets/sigma'
        template = (assets / 'viewer.html').read_text(encoding='utf-8')
        html_doc = (
            template.replace('__GRAPHOLOGY_JS__', (assets / 'graphology.umd.min.js').read_text(encoding='utf-8'))
            .replace('__SIGMA_JS__', (assets / 'sigma.min.js').read_text(encoding='utf-8'))
            .replace('__NETWORK_DATA__', payload)
        )
        return html_doc, empty_xy

    node_ids = nodes_df['id'].to_list()
    raw_edges = [
        (str(r['source']), str(r['target']), float(r.get('score') or 1.0))
        for r in edges_df.iter_rows(named=True)
    ] if not edges_df.is_empty() else []

    if engine == 'Circular':
        layout_engine, layout_name = 'Circular', 'Circular'
    elif engine == 'graph-tool SFDP':
        layout_engine, layout_name = 'graph-tool', 'SFDP'
    else:
        layout_engine, _, layout_name = engine.partition(' ')
        layout_name = layout_name or 'Spring'

    coords = _compute_2d_layout(node_ids, raw_edges, engine=layout_engine, layout=layout_name)

    # Graphology UndirectedGraph forbids duplicate undirected pairs.
    seen_pairs: set[tuple[str, str]] = set()
    edge_data = []
    deg_map: dict[str, int] = {str(nid): 0 for nid in node_ids}
    if not edges_df.is_empty():
        for idx, row in enumerate(edges_df.iter_rows(named=True)):
            u, v = str(row['source']), str(row['target'])
            if u == v:
                continue
            pair = (u, v) if u < v else (v, u)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            deg_map[u] = deg_map.get(u, 0) + 1
            deg_map[v] = deg_map.get(v, 0) + 1
            edge_data.append({
                'key': f'e{idx}',
                'source': u,
                'target': v,
                'attributes': {
                    'color': str(row['edge_color']),
                    'size': round(min(4.0, float(row['edge_width']) * 0.7), 2),
                },
            })

    node_data = []
    xs, ys = [], []
    for row in nodes_df.iter_rows(named=True):
        nid = str(row['id'])
        pt = coords[nid]
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
        node_data.append({
            'key': nid,
            'attributes': {
                'x': float(pt[0]),
                'y': float(pt[1]),
                'label': str(row['label']),
                'color': str(row['color']),
                'size': round(max(3.0, min(18.0, float(row['display_size']) * 0.35)), 2),
                'protein_set': str(row['protein_set']),
                'degree': deg_map.get(nid, 0),
                'cluster_selected': bool(row['cluster_selected']),
                'labelColor': '#D97706' if row['cluster_selected'] else '#1F2937',
                'forceLabel': bool(row['cluster_selected'] and nodes_df.height <= 120),
            },
        })

    payload = json.dumps({'nodes': node_data, 'edges': edge_data, 'title': title}).replace('<', r'\u003c')
    assets = Path(__file__).parent / 'assets/sigma'
    template = (assets / 'viewer.html').read_text(encoding='utf-8')
    html_doc = (
        template.replace('__GRAPHOLOGY_JS__', (assets / 'graphology.umd.min.js').read_text(encoding='utf-8'))
        .replace('__SIGMA_JS__', (assets / 'sigma.min.js').read_text(encoding='utf-8'))
        .replace('__NETWORK_DATA__', payload)
    )
    nodes_with_xy = nodes_df.with_columns(pl.Series('x', xs), pl.Series('y', ys))
    return html_doc, nodes_with_xy


def render_ocg_view_pyvis(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    physics: bool = True,
    labels: bool = True,
    legend_text: str = '',
) -> str:
    """Render any standardized OCG view as an offline PyVis (vis-network) HTML document."""
    from pyvis.network import Network

    net = Network(height='640px', width='100%', directed=False, cdn_resources='in_line')
    if not nodes_df.is_empty():
        sym_map = {'diamond': 'diamond', 'hexagon': 'hexagon', 'star': 'star', 'roundRect': 'box', 'ellipse': 'dot'}
        for row in nodes_df.iter_rows(named=True):
            nid = str(row['id'])
            fill = str(row['color'])
            net.add_node(
                nid,
                label=str(row['label']) if (labels and row['show_label']) or row['cluster_selected'] else ' ',
                title=escape(str(row['tooltip'])).replace('\n', '<br>'),
                shape=sym_map.get(str(row['shape']), 'dot'),
                size=round(max(8.0, min(38.0, float(row['display_size']) * 0.72)), 1),
                color={
                    'background': fill,
                    'border': str(row['border_color']),
                    'highlight': {'background': fill, 'border': '#D97706'},
                },
                borderWidth=float(row['border_width']),
            )
    if not edges_df.is_empty():
        for row in edges_df.iter_rows(named=True):
            net.add_edge(
                str(row['source']),
                str(row['target']),
                color=str(row['edge_color']),
                width=float(row['edge_width']),
                dashes=row['edge_style'] in ('dashed', 'dotted'),
                title=escape(str(row['tooltip'])).replace('\n', '<br>'),
            )
    net.set_options(json.dumps({
        'layout': {'randomSeed': 42},
        'physics': {'enabled': physics, 'solver': 'barnesHut', 'stabilization': {'iterations': 150}},
        'interaction': {'hover': True, 'navigationButtons': True, 'keyboard': True},
        'edges': {'smooth': False},
    }))
    document = net.generate_html(notebook=False)
    document = re.sub(r'<link\b[^>]*href="https://cdn\.jsdelivr\.net/npm/bootstrap[^>]*>', '', document)
    document = re.sub(r'<script\b[^>]*src="https://cdn\.jsdelivr\.net/npm/bootstrap[^>]*>\s*</script>', '', document)
    if legend_text:
        banner = f'<p style="font:13px sans-serif;margin:8px 12px;">{escape(legend_text)}</p>'
        document = document.replace('<body>', '<body>' + banner, 1)
    return document


def render_ocg_view_static(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    title: str = 'OCG network view',
    engine: str = 'NetworkX',
    layout: str = 'Spring',
    labels: bool = True,
):
    """Render any standardized OCG view as a publication-ready Matplotlib Figure."""
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    fig = Figure(figsize=(12, 8.5), layout='constrained')
    ax = fig.subplots()
    if nodes_df.is_empty():
        ax.text(0.5, 0.5, 'No nodes to display.', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig

    node_ids = nodes_df['id'].to_list()
    edge_tuples = [
        (str(r['source']), str(r['target']), float(r.get('score') or 1.0))
        for r in edges_df.iter_rows(named=True)
    ] if not edges_df.is_empty() else []
    coords = _compute_2d_layout(node_ids, edge_tuples, engine=engine, layout=layout)

    edge_handles = []
    if not edges_df.is_empty():
        lstyle_map = {'solid': 'solid', 'dashed': 'dashed', 'dotted': 'dotted'}
        for (ecolor, estyle, interaction), group in edges_df.group_by(
            ['edge_color', 'edge_style', 'interaction'], maintain_order=True
        ):
            segs = [
                (coords[str(r['source'])], coords[str(r['target'])])
                for r in group.iter_rows(named=True)
                if str(r['source']) in coords and str(r['target']) in coords
            ]
            if segs:
                widths = [float(r['edge_width']) * 0.85 for r in group.iter_rows(named=True)]
                ax.add_collection(LineCollection(
                    segs, colors=str(ecolor), linewidths=widths,
                    linestyles=lstyle_map.get(str(estyle), 'solid'), alpha=0.65, zorder=1,
                ))
                edge_handles.append(Line2D(
                    [], [], color=str(ecolor), linewidth=1.6,
                    linestyle=lstyle_map.get(str(estyle), 'solid'), label=str(interaction),
                ))

    marker_map = {'diamond': 'D', 'hexagon': 'h', 'star': '*', 'roundRect': 's', 'ellipse': 'o'}
    node_handles = []
    for (kind, shape, color), group in nodes_df.group_by(['node_kind', 'shape', 'color'], maintain_order=True):
        pts = np.asarray([coords[str(r['id'])] for r in group.iter_rows(named=True)])
        marker = marker_map.get(str(shape), 'o')
        sizes = [max(40.0, (float(r['display_size']) * 0.55) ** 2) for r in group.iter_rows(named=True)]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=sizes, c=[str(color)] * len(pts), marker=marker,
            edgecolors=[str(r['border_color']) for r in group.iter_rows(named=True)],
            linewidths=[float(r['border_width']) * 0.65 for r in group.iter_rows(named=True)],
            zorder=2,
        )
        if len(node_handles) < 6:
            node_handles.append(Line2D(
                [], [], marker=marker, markerfacecolor=str(color), markeredgecolor='#1F2937',
                linestyle='', markersize=7.5, label=str(kind),
            ))

    if labels:
        for row in nodes_df.iter_rows(named=True):
            if row['show_label'] or row['cluster_selected']:
                pt = coords[str(row['id'])]
                ax.annotate(
                    str(row['label']), pt, xytext=(4, 4), textcoords='offset points',
                    fontsize=7.5 if row['cluster_selected'] else 6.8,
                    fontweight='bold' if row['cluster_selected'] else 'normal',
                )

    all_handles = node_handles + edge_handles[:4]
    if all_handles:
        ax.legend(handles=all_handles, loc='upper left', bbox_to_anchor=(0, -0.01), frameon=False, ncol=3, fontsize=8.5)
    ax.set_title(
        f'{title}\n{engine} · {layout} · {nodes_df.height:,} nodes · {edges_df.height:,} links',
        fontsize=11.5,
    )
    ax.autoscale()
    ax.margins(0.13)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return fig


def render_ocg_view_3d_plotly(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    title: str = '3D OCG network view',
) -> go.Figure:
    """Render any standardized OCG (nodes_df, edges_df) view as a rotatable Plotly 3D graph."""
    fig = go.Figure()
    if nodes_df.is_empty():
        fig.update_layout(title='No nodes to display', height=540)
        return fig

    node_ids = nodes_df['id'].to_list()
    link_df = (
        edges_df.select('source', 'target', 'interactions')
        if not edges_df.is_empty()
        else pl.DataFrame(schema={'source': pl.String, 'target': pl.String, 'interactions': pl.Int64})
    )
    coords = layout_3d(node_ids, link_df)

    if not edges_df.is_empty():
        for (ecolor, interaction), group in edges_df.group_by(['edge_color', 'interaction'], maintain_order=True):
            ex, ey, ez = [], [], []
            for r in group.iter_rows(named=True):
                s, t = str(r['source']), str(r['target'])
                if s in coords and t in coords:
                    ex.extend([coords[s][0], coords[t][0], None])
                    ey.extend([coords[s][1], coords[t][1], None])
                    ez.extend([coords[s][2], coords[t][2], None])
            if ex:
                fig.add_trace(go.Scatter3d(
                    x=ex, y=ey, z=ez, mode='lines', name=str(interaction),
                    line=dict(color=str(ecolor), width=max(1.2, float(group['edge_width'].mean() or 1.4))),
                    opacity=0.55, hoverinfo='skip',
                ))

    sym3d = {'diamond': 'diamond', 'hexagon': 'square', 'star': 'diamond', 'roundRect': 'square', 'ellipse': 'circle'}
    for kind, group in nodes_df.group_by('node_kind', maintain_order=True):
        pts = [coords[str(r['id'])] for r in group.iter_rows(named=True)]
        texts = [
            (escape(str(r['label'])) if r['cluster_selected'] or nodes_df.height <= 40 else '')
            for r in group.iter_rows(named=True)
        ]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in pts],
            y=[p[1] for p in pts],
            z=[p[2] for p in pts],
            mode='markers+text' if any(texts) else 'markers',
            text=texts,
            name=str(kind),
            hovertext=[escape(str(r['tooltip'])).replace('\n', '<br>') for r in group.iter_rows(named=True)],
            hoverinfo='text',
            marker=dict(
                size=[max(5.0, float(r['display_size']) * 0.42) for r in group.iter_rows(named=True)],
                color=[str(r['color']) for r in group.iter_rows(named=True)],
                symbol=[sym3d.get(str(r['shape']), 'circle') for r in group.iter_rows(named=True)],
                line=dict(width=1.2, color='#111827'),
            ),
        ))

    hidden = dict(visible=False)
    fig.update_layout(
        title=title,
        height=620,
        margin=dict(l=0, r=0, b=0, t=55),
        scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden, aspectmode='data'),
        legend=dict(orientation='h'),
        template='plotly_white',
    )
    return fig




def create_ocg_norma_session(name: str, edges_df: pl.DataFrame) -> tuple[str, float]:
    """Create a validated temporary visualization session on normaviz.org."""
    import time
    from urllib.parse import parse_qs, urljoin, urlsplit
    import requests

    norma_origin = 'https://www.normaviz.org'
    payload = {
        'name': name,
        'edges': edges_df.select('source', 'target').to_dicts(),
        'settings': {'layout': 'fr'},
    }
    response = requests.post(
        f'{norma_origin}/api/external', json=payload, timeout=30, allow_redirects=False,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get('url'), str):
        raise ValueError('NORMA returned no session link.')
    url = urljoin(f'{norma_origin}/', data['url'])
    parts = urlsplit(url)
    if (
        parts.scheme != 'https'
        or parts.netloc not in ('www.normaviz.org', 'normaviz.org')
        or parts.path != '/norma.html'
        or not parse_qs(parts.query).get('session')
    ):
        raise ValueError('NORMA returned an unexpected session link.')
    hours = float(data.get('expiresInHours', 24))
    return url, time.monotonic() + hours * 3600


def annotated_ocg_overlap_matrix_figure(
    clusters: pl.DataFrame,
    pairs: pl.DataFrame,
    evidence: pl.DataFrame | None = None,
    *,
    limit: int = 80,
    metric: str = 'jaccard',
    protein_set_label: str = 'Selected protein set',
    sort_by: str = 'qvalue',
):
    """Build Plotly and Matplotlib hierarchically ordered cluster-overlap matrices with annotations."""
    from matplotlib.figure import Figure

    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    limit = max(2, min(150, int(limit)))
    if clusters.is_empty():
        empty_fig = Figure(figsize=(8, 5))
        ax = empty_fig.subplots()
        ax.text(0.5, 0.5, 'No OCG clusters available.', ha='center', va='center')
        ax.set_axis_off()
        return go.Figure(), empty_fig, pl.DataFrame()

    if sort_by == 'qvalue' and 'qvalue' in clusters.columns:
        selected = clusters.sort(['qvalue', 'size', 'community'], descending=[False, True, False]).head(limit)
    else:
        selected = clusters.sort(['size', 'community'], descending=[True, False]).head(limit)

    ids = selected['community'].to_list()
    n = len(ids)
    within = (
        pairs.filter(pl.col('community_a').is_in(ids) & pl.col('community_b').is_in(ids))
        if not pairs.is_empty() else pairs
    )

    index = {c: i for i, c in enumerate(ids)}
    matrix = np.eye(n)
    shared_mat = np.zeros((n, n), dtype=int)
    for row in selected.iter_rows(named=True):
        i = index[row['community']]
        shared_mat[i, i] = int(row['size'])
    if not within.is_empty():
        for row in within.iter_rows(named=True):
            a, b = index[row['community_a']], index[row['community_b']]
            matrix[a, b] = matrix[b, a] = float(row[metric])
            shared_mat[a, b] = shared_mat[b, a] = int(row['shared_proteins'])

    order = (
        leaves_list(linkage(squareform(np.clip(1 - matrix, 0, 1), checks=False), method='average'))
        if n > 1 else [0]
    )
    matrix = matrix[np.ix_(order, order)]
    shared_mat = shared_mat[np.ix_(order, order)]

    func_lookup = _best_cluster_label_lookup(evidence)
    ordered_rows = [selected.row(i, named=True) for i in order]
    axis_labels = []
    short_labels = []
    table_rows = []
    for r in ordered_rows:
        cid_str = str(r['community'])
        ann = func_lookup.get(cid_str)
        row_sig = r.get('significance') or (significance_label(r.get('qvalue')) if 'qvalue' in r else 'n/a')
        if ann:
            term = ann['term_name']
            short = term[:24] + ('…' if len(term) > 24 else '')
            axis_lbl = f"C{cid_str} · {short} ({ann['significance']})"
        elif 'significance' in r:
            axis_lbl = f"C{cid_str} ({row_sig})"
        else:
            axis_lbl = f"C{cid_str}"
        axis_labels.append(axis_lbl)
        short_labels.append(f"C{cid_str} ({row_sig})" if 'significance' in r else f"C{cid_str}")
        table_rows.append({
            'community': cid_str,
            'axis_label': axis_lbl,
            'size': int(r['size']),
            'protein_set_hits': int(r['protein_set_hits']) if r.get('protein_set_hits') is not None else None,
            'fold_enrichment': float(r['fold_enrichment']) if r.get('fold_enrichment') is not None else None,
            'qvalue': float(r['qvalue']) if r.get('qvalue') is not None else None,
            'significance': row_sig,
            'top_term': ann['label_text'] if ann else '',
        })

    metric_label = 'Jaccard similarity' if metric == 'jaccard' else 'Overlap coefficient'
    hover = []
    for i in range(n):
        row_tips = []
        for j in range(n):
            row_tips.append(
                f"{escape(axis_labels[i])} × {escape(axis_labels[j])}"
                f"<br>{metric_label}: {matrix[i, j]:.3f}"
                f"<br>Shared proteins: {int(shared_mat[i, j]):,}"
            )
        hover.append(row_tips)

    plotly_fig = go.Figure(go.Heatmap(
        z=matrix.tolist(),
        x=axis_labels,
        y=axis_labels,
        zmin=0,
        zmax=1,
        colorscale='Blues',
        text=hover,
        hoverinfo='text',
        colorbar=dict(title=metric_label),
    ))
    plotly_fig.update_layout(
        title=f'Annotated OCG overlap matrix · {metric_label} (top {n} clusters; {protein_set_label})',
        height=max(560, min(860, 320 + n * 10)),
        margin=dict(l=160 if n <= 45 else 60, r=40, t=70, b=150 if n <= 45 else 60),
        xaxis=dict(title='Cluster (average-linkage hierarchical order)', showticklabels=n <= 45, tickangle=-50),
        yaxis=dict(title='Cluster (same order)', autorange='reversed', showticklabels=n <= 45),
    )

    fig_w = max(9.5, min(16.0, 6.5 + 0.14 * n))
    fig_h = max(8.0, min(14.5, 5.5 + 0.13 * n))
    static_fig = Figure(figsize=(fig_w, fig_h), layout='constrained')
    ax = static_fig.subplots()
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    static_fig.colorbar(im, ax=ax, shrink=0.75, label=metric_label)
    if n <= 40:
        ticks = list(range(n))
        ax.set_xticks(ticks, axis_labels if n <= 25 else short_labels, rotation=55, ha='right', fontsize=7.2)
        ax.set_yticks(ticks, axis_labels if n <= 25 else short_labels, fontsize=7.2)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(
        f'Annotated OCG overlap matrix · {metric_label}\n(top {n} clusters, average-linkage hierarchical order)',
        fontsize=11,
    )
    return plotly_fig, static_fig, pl.from_dicts(table_rows, infer_schema_length=None)


def ocg_core_interface_landscape_figures(
    clusters: pl.DataFrame,
    memberships: pl.DataFrame,
    edges: pl.DataFrame | None = None,
    evidence: pl.DataFrame | None = None,
    *,
    protein_set_label: str = 'Selected protein set',
    top_n: int = 35,
):
    """Build Core (k=1) vs Interface (k≥2) cluster composition & enrichment landscape figures."""
    from matplotlib.figure import Figure

    if clusters.is_empty():
        empty = Figure(figsize=(8, 4.5))
        ax = empty.subplots()
        ax.text(0.5, 0.5, 'No OCG clusters available.', ha='center', va='center')
        ax.set_axis_off()
        return go.Figure(), go.Figure(), empty, pl.DataFrame()

    assigned = (
        memberships.filter(pl.col('community').is_not_null())
        .with_columns(pl.col('community').cast(pl.String).alias('_cstr'))
        .select('protein', '_cstr')
        .unique()
    )
    protein_k = assigned.group_by('protein').agg(pl.len().alias('_k'))
    comp = (
        assigned.join(protein_k, on='protein', how='left')
        .group_by('_cstr')
        .agg(
            pl.len().alias('member_count'),
            (pl.col('_k') == 1).sum().cast(pl.Int64).alias('core_proteins'),
            (pl.col('_k') >= 2).sum().cast(pl.Int64).alias('interface_proteins'),
        )
        .with_columns(
            (pl.col('interface_proteins') / pl.col('member_count')).alias('interface_fraction'),
            (pl.col('core_proteins') / pl.col('member_count')).alias('core_fraction'),
        )
    )

    landscape = (
        clusters.with_columns(pl.col('community').cast(pl.String).alias('_cstr'))
        .join(comp, on='_cstr', how='left')
        .with_columns(
            pl.col('core_proteins').fill_null(0).cast(pl.Int64),
            pl.col('interface_proteins').fill_null(0).cast(pl.Int64),
            pl.col('interface_fraction').fill_null(0.0),
            pl.col('core_fraction').fill_null(0.0),
        )
    )

    if edges is not None and not edges.is_empty():
        qm = (
            cluster_quality_metrics(memberships, edges, cohort=None)
            .with_columns(pl.col('community').cast(pl.String).alias('_cstr'))
            .select('_cstr', 'internal_density', 'conductance', 'internal_edges', 'boundary_edges')
        )
        landscape = landscape.join(qm, on='_cstr', how='left')

    func_lookup = _best_cluster_label_lookup(evidence)
    top_terms = []
    top_libs = []
    top_term_qs = []
    top_term_sigs = []
    for cid_str in landscape['_cstr'].to_list():
        ann = func_lookup.get(cid_str)
        top_terms.append(ann['term_name'] if ann else '')
        top_libs.append(ann['library'] if ann else '')
        top_term_qs.append(ann['qvalue'] if ann else None)
        top_term_sigs.append(ann['significance'] if ann else 'n/a')

    landscape = landscape.with_columns(
        pl.Series('top_term', top_terms),
        pl.Series('top_library', top_libs),
        pl.Series('top_term_qvalue', top_term_qs, dtype=pl.Float64),
        pl.Series('top_term_significance', top_term_sigs),
    )

    if 'qvalue' in landscape.columns:
        landscape = landscape.sort(
            ['qvalue', 'fold_enrichment', 'size', '_cstr'],
            descending=[False, True, True, False],
            nulls_last=True,
        )
    else:
        landscape = landscape.sort(['interface_proteins', 'size', '_cstr'], descending=[True, True, False])

    landscape = landscape.drop('_cstr')
    top = landscape.head(max(1, min(60, int(top_n))))

    x_labels = [f"C{r['community']}" for r in top.iter_rows(named=True)]
    core_vals = top['core_proteins'].to_list()
    int_vals = top['interface_proteins'].to_list()
    totals = [c + i for c, i in zip(core_vals, int_vals)]
    sigs = [
        str(r.get('significance') or (significance_label(r.get('qvalue')) if 'qvalue' in r else 'n/a'))
        for r in top.iter_rows(named=True)
    ]

    fig_bar = go.Figure()
    fig_bar.add_bar(
        x=x_labels, y=core_vals, name='Core (k = 1, exclusive)',
        marker_color='#009E73',
        customdata=list(zip(totals, sigs, top['top_term'].to_list())),
        hovertemplate='%{x}<br>Core (k=1): %{y}<br>Total: %{customdata[0]}<br>Set sig: %{customdata[1]}<br>Label: %{customdata[2]}<extra></extra>',
    )
    fig_bar.add_bar(
        x=x_labels, y=int_vals, name='Interface (k ≥ 2, shared)',
        marker_color='#D55E00',
        customdata=list(zip(totals, sigs, top['top_term'].to_list())),
        hovertemplate='%{x}<br>Interface (k≥2): %{y}<br>Total: %{customdata[0]}<br>Set sig: %{customdata[1]}<br>Label: %{customdata[2]}<extra></extra>',
    )
    max_tot = max(totals, default=1)
    for lbl, tot, sig in zip(x_labels, totals, sigs):
        fig_bar.add_annotation(
            x=lbl, y=tot + max_tot * 0.04, text=sig, showarrow=False, font=dict(size=11),
        )
    fig_bar.update_layout(
        title=f'Core (k=1) vs. Interface (k≥2) proteins across top {top.height} OCG clusters ({protein_set_label})',
        barmode='stack',
        height=480,
        xaxis=dict(title='OCG cluster (ranked by protein-set BH q-value)', tickangle=-45),
        yaxis=dict(title='Proteins in cluster', range=[0, max_tot * 1.18]),
        legend=dict(orientation='h', y=1.08),
        margin=dict(l=60, r=25, t=80, b=85),
    )

    has_fold = 'fold_enrichment' in landscape.columns and landscape['fold_enrichment'].null_count() < landscape.height
    has_q = 'qvalue' in landscape.columns and landscape['qvalue'].null_count() < landscape.height
    y_col = 'fold_enrichment' if has_fold else ('internal_density' if 'internal_density' in landscape.columns else 'size')
    y_title = (
        f'{protein_set_label} fold enrichment' if y_col == 'fold_enrichment'
        else ('Internal PPI density' if y_col == 'internal_density' else 'Cluster size (proteins)')
    )

    plot_df = landscape.filter(pl.col(y_col).is_not_null())
    if has_q:
        neg_log_q = [
            -math.log10(max(float(q), 1e-300)) if q is not None and math.isfinite(float(q)) else 0.0
            for q in plot_df['qvalue'].to_list()
        ]
        cbar_title = '-log10(BH q)'
    else:
        neg_log_q = plot_df['overlap_neighbors'].to_list() if 'overlap_neighbors' in plot_df.columns else [0] * plot_df.height
        cbar_title = 'Overlapping<br>clusters'

    max_sz = max(1, int(plot_df['size'].max() or 1)) if not plot_df.is_empty() else 1
    scatter_sigs = [
        str(r.get('significance') or 'n/a') for r in plot_df.iter_rows(named=True)
    ]
    scatter_texts = [
        f"C{r['community']} {s}" if s in {'*', '**', '***', '****'} else ''
        for r, s in zip(plot_df.iter_rows(named=True), scatter_sigs)
    ]
    fig_scatter = go.Figure()
    if not plot_df.is_empty():
        fig_scatter.add_trace(go.Scatter(
            x=plot_df['interface_fraction'].to_list(),
            y=plot_df[y_col].to_list(),
            mode='markers+text' if any(scatter_texts) else 'markers',
            text=scatter_texts,
            textposition='top center',
            hovertext=[
                f"C{r['community']} ({r['size']} proteins)"
                f"<br>Interface fraction (k≥2): {r['interface_fraction']:.1%}"
                f"<br>{y_title}: {float(r[y_col]):.3g}"
                f"<br>Significance: {s}"
                + (f"<br>Label: {escape(str(r['top_term']))} ({r['top_term_significance']})" if r['top_term'] else '')
                for r, s in zip(plot_df.iter_rows(named=True), scatter_sigs)
            ],
            hoverinfo='text',
            marker=dict(
                size=[max(8.0, 32.0 * math.sqrt(int(r['size']) / max_sz)) for r in plot_df.iter_rows(named=True)],
                color=neg_log_q,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=cbar_title),
                line=dict(width=1.0, color='#111827'),
            ),
            showlegend=False,
        ))
    fig_scatter.update_layout(
        title=f'Cluster interface fraction vs. {y_title} ({protein_set_label})',
        height=480,
        xaxis=dict(title='Interface protein fraction (k ≥ 2 / cluster size)', tickformat='.0%', range=[-0.04, 1.06]),
        yaxis=dict(title=y_title, rangemode='tozero'),
        margin=dict(l=65, r=35, t=65, b=65),
        template='plotly_white',
    )

    static_fig = Figure(figsize=(15.5, 6.8), layout='constrained')
    ax1, ax2 = static_fig.subplots(1, 2)
    idx = np.arange(len(x_labels))
    ax1.bar(idx, core_vals, color='#009E73', label='Core (k = 1)')
    ax1.bar(idx, int_vals, bottom=core_vals, color='#D55E00', label='Interface (k ≥ 2)')
    for i, (tot, sig) in enumerate(zip(totals, sigs)):
        ax1.text(i, tot + max_tot * 0.025, sig, ha='center', va='bottom', fontsize=7.5)
    ax1.set_ylim(0, max_tot * 1.2)
    ax1.set_xticks(idx, x_labels, rotation=55, ha='right', fontsize=7.5)
    ax1.set_ylabel('Proteins in cluster')
    ax1.set_title(f'Core vs. interface proteins (top {top.height} clusters)\nBH stars: {protein_set_label} enrichment', fontsize=10.5)
    ax1.legend(loc='upper right', frameon=False, fontsize=8.5)
    ax1.spines[['top', 'right']].set_visible(False)

    if not plot_df.is_empty():
        sc = ax2.scatter(
            plot_df['interface_fraction'].to_numpy(),
            plot_df[y_col].to_numpy(),
            s=[max(35.0, 420.0 * (int(r['size']) / max_sz)) for r in plot_df.iter_rows(named=True)],
            c=neg_log_q,
            cmap='viridis',
            edgecolors='#111827',
            linewidths=0.7,
            alpha=0.88,
        )
        static_fig.colorbar(sc, ax=ax2, shrink=0.8, label=cbar_title.replace('<br>', ' '))
        labeled_count = 0
        for r, s in zip(plot_df.iter_rows(named=True), scatter_sigs):
            if s in {'*', '**', '***', '****'} and labeled_count < 15:
                ax2.annotate(
                    f"C{r['community']} {s}",
                    (float(r['interface_fraction']), float(r[y_col])),
                    xytext=(4, 4), textcoords='offset points', fontsize=7.5,
                )
                labeled_count += 1
    ax2.set_xlim(-0.05, 1.06)
    ax2.set_xlabel('Interface fraction (k ≥ 2 / cluster size)')
    ax2.set_ylabel(y_title)
    ax2.set_title(f'Interface share vs. {y_title}', fontsize=10.5)
    ax2.spines[['top', 'right']].set_visible(False)
    return fig_bar, fig_scatter, static_fig, landscape
