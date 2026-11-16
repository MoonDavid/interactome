"""Overlapping OCG ego hulls and documented NORMA interchange formats."""
import io
import json
import math
from html import escape
from urllib.parse import urljoin, urlparse
from zipfile import ZipFile, ZIP_DEFLATED

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.spatial import ConvexHull

from network_graph_utils import cluster_colors
from moonlighting import significance_label


def ego_groups(nodes, memberships, center, max_clusters=8, scope='center'):
    """Rank clusters deterministically; never assign overlapping nodes exclusively."""
    local = (memberships.select('protein', pl.col('community').cast(pl.String))
             .drop_nulls().unique().join(nodes.select('protein').unique(), on='protein', how='semi'))
    center_ids = local.filter(pl.col('protein') == center)['community'].to_list()
    if scope == 'center':
        local = local.filter(pl.col('community').is_in(center_ids))
    ranked = local.group_by('community').agg(pl.len().alias('visible_proteins')).with_columns(
        pl.col('community').is_in(center_ids).alias('contains_center')).sort(
            ['contains_center', 'visible_proteins', 'community'], descending=[True, True, False])
    selected = ranked.head(max(1, int(max_clusters)))
    ids = selected['community'].to_list()
    return {c: local.filter(pl.col('community') == c)['protein'].sort().to_list() for c in ids}, ranked


def ego_overlap_figure(nodes, edges, groups, center, *, names=None, labels=None,
                       opacity=.16, show_names=True, group_layout=True, fixed_positions=None):
    names, labels = names or {}, labels or {}
    graph = nx.Graph()
    proteins = nodes['protein'].to_list()
    graph.add_nodes_from(('protein', p) for p in proteins)
    graph.add_edges_from((('protein', a), ('protein', b)) for a, b in edges.select('a', 'b').iter_rows())
    if group_layout:
        for c, members in groups.items():
            graph.add_edges_from((('cluster', c), ('protein', p)) for p in members)
    if fixed_positions is None:
        positions = nx.spring_layout(graph, seed=42, iterations=150, weight=None)
        pos = {p: positions[('protein', p)] for p in proteins}
    else:
        pos = {p: np.asarray(fixed_positions[p]) for p in proteins}
    colors = cluster_colors(groups)
    fig = go.Figure()
    radius = .045
    for c, members in groups.items():
        # Inflate every point before taking a hull: valid for singletons and pairs too.
        cloud = np.array([pos[p] + radius * np.array([math.cos(t), math.sin(t)])
                          for p in members for t in np.linspace(0, 2*math.pi, 16, endpoint=False)])
        boundary = cloud[ConvexHull(cloud).vertices]
        boundary = np.vstack([boundary, boundary[0]])
        color = colors[c]
        rgb = ','.join(str(int(color[i:i+2], 16)) for i in (1, 3, 5))
        title = f'C{c}' + (f' · {labels[c]}' if labels.get(c) else '')
        fig.add_trace(go.Scatter(x=boundary[:, 0], y=boundary[:, 1], mode='lines',
            fill='toself', fillcolor=f'rgba({rgb},{opacity})', line=dict(color=color, width=1.5),
            name=escape(title[:90] + ('…' if len(title) > 90 else '')), legendgroup=c,
            hovertemplate=f'{escape(title)}<br>{len(members)} visible proteins<extra></extra>'))
    x, y = [], []
    for a, b in edges.select('a', 'b').iter_rows():
        x.extend([pos[a][0], pos[b][0], None]); y.extend([pos[a][1], pos[b][1], None])
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='rgba(100,116,139,.4)', width=1),
                             hoverinfo='skip', showlegend=False))
    for p in proteins:
        memberships_here = [c for c, members in groups.items() if p in members]
        for i, c in enumerate(memberships_here):
            angles = np.linspace(2*math.pi*i/len(memberships_here), 2*math.pi*(i+1)/len(memberships_here), 12)
            fig.add_trace(go.Scatter(x=pos[p][0]+.018*np.cos(angles), y=pos[p][1]+.018*np.sin(angles),
                mode='lines', line=dict(color=colors[c], width=4), showlegend=False, hoverinfo='skip', legendgroup=c))
    text = [escape(names.get(p, p)) for p in proteins]
    tips = [f'{escape(names.get(p,p))}<br>{escape(p)}<br>Shown clusters: ' +
            (', '.join('C'+c for c, members in groups.items() if p in members) or 'none') for p in proteins]
    fig.add_trace(go.Scatter(x=[pos[p][0] for p in proteins], y=[pos[p][1] for p in proteins],
        mode='markers+text' if show_names else 'markers', text=text, textposition='top center',
        hovertext=tips, hovertemplate='%{hovertext}<extra></extra>', name='Proteins',
        marker=dict(size=[14 if p == center else 7 for p in proteins],
                    color=['#f97316' if p == center else '#ffffff' for p in proteins],
                    line=dict(color='#334155', width=1)), customdata=proteins))
    fig.update_layout(height=720, template='plotly_white', margin=dict(l=15,r=15,t=40,b=15),
        title=f'Overlapping OCG ego clusters · {names.get(center, center)}',
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
        legend=dict(orientation='h', y=-.03), dragmode='pan')
    return fig


def norma_payload(nodes, edges, groups, center, *, name='OCG ego network', names=None, labels=None):
    names, labels = names or {}, labels or {}
    colors = cluster_colors(groups)
    return dict(name=name, directed=False,
        nodes=[dict(id=p, gene=names.get(p,p), role='center' if p == center else 'neighbor')
               for p in nodes['protein'].to_list()],
        edges=[dict(source=a, target=b) for a,b in edges.select('a','b').iter_rows()],
        groups=[dict(name='OCG '+c, members=members, color=colors[c], description=labels.get(c,''))
                for c,members in groups.items()],
        settings=dict(layout='fr', showGroupHulls=True),
        description='Displayed 1-hop ego network; groups are truncated to visible proteins. OCG membership overlaps are retained.')


def norma_export_zip(payload):
    buffer = io.BytesIO()
    network = pl.DataFrame(payload['edges'], schema={'source':pl.String,'target':pl.String}).rename(
        {'source':'Source','target':'Target'}).write_csv(separator='\t')
    annotation = '\n'.join(g['name']+'\t'+','.join(g['members']) for g in payload['groups'])+'\n'
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('network.tsv', network)
        archive.writestr('annotations.tsv', annotation)
        archive.writestr('norma_api_payload.json', json.dumps(payload, indent=2))
        archive.writestr('README.txt', 'NORMA3: upload network.tsv as network and annotations.tsv as annotation.\n'
            'NORMA current API: POST norma_api_payload.json to /api/external on a NORMA server.\n'
            'API docs: https://www.normaviz.org/\n'
            'Only displayed ego nodes/edges and selected groups are exported; group membership overlaps are preserved.\n'
            'An enclosing hull may cover nonmembers; use membership records, not hull geometry, to infer membership.\n')
    return buffer.getvalue()


def send_to_norma(payload, server):
    import requests
    if urlparse(server).scheme not in ('http','https'):
        raise ValueError('NORMA server must use http or https')
    response = requests.post(server.rstrip('/')+'/api/external', json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    url = urljoin(server.rstrip('/')+'/', result['url'])
    if urlparse(url).scheme not in ('http','https'):
        raise ValueError('NORMA returned an invalid view URL')
    return url


def overlap_ego_panel(nodes, edges, memberships, center, names, summary, network):
    import streamlit as st
    scope = st.selectbox('Clusters to outline', ['Center protein clusters', 'All clusters touching this ego network'],
                         key='ocg_hull_scope')
    max_clusters = st.slider('Maximum clusters to outline', 1, 20, 8, key='ocg_hull_max')
    groups, ranked = ego_groups(nodes, memberships, center, max_clusters,
                               'center' if scope == 'Center protein clusters' else 'all')
    lookup = {}
    if summary is not None:
        for row in summary.iter_rows(named=True):
            terms = []
            for library in ('GO_BP', 'GO_MF', 'GO_CC', 'Reactome'):
                term, q, fold = (row.get(library + suffix) for suffix in ('_enrichment', '_qvalue', '_fold'))
                if term:
                    terms.append(f'{library}: {term} {significance_label(q)}' +
                                 (f' (q={q:.3g}, fold={fold:.3g}×)' if q is not None and fold is not None else ''))
            lookup[str(row['community'])] = ' · '.join(terms)
    chosen = st.multiselect('Visible cluster outlines', list(groups), default=list(groups),
        format_func=lambda c: f'C{c} · {lookup.get(c) or "No saved significant label"}', key='ocg_hull_selected')
    groups = {c:groups[c] for c in chosen}
    a,b,c = st.columns(3)
    opacity = a.slider('Cluster shading opacity', .05, .4, .16, key='ocg_hull_opacity')
    show_names = b.checkbox('Show protein names', value=True, key='ocg_hull_names')
    group_layout = c.checkbox('Use cluster-aware layout', value=True, key='ocg_hull_layout')
    st.caption(f'{len(groups)} of {ranked.height} eligible clusters outlined. Center-containing clusters are prioritized, '
        'then visible member count. The neighbor limit above controls proteins; all displayed proteins and edges remain '
        'when cluster outlines are hidden. Multicolour rings show overlapping memberships among outlined clusters.')
    fig = ego_overlap_figure(nodes, edges, groups, center, names=names, labels=lookup,
                             opacity=opacity, show_names=show_names, group_layout=group_layout)
    st.plotly_chart(fig, width='stretch', key='ocg_hull_plot')
    st.info('Shaded boundaries enclose visible members; they can also enclose nonmembers. Hover over a protein '
            'to inspect its memberships. Distances are layout coordinates, not functional similarity or explained variance. '
            'This descriptive view has no statistical significance test. The complete run memberships are retained.')
    if summary is not None:
        st.caption('Functional labels retain saved full-run, one-sided hypergeometric tests against the original '
                   'network and BH correction over all eligible terms, libraries, clusters and nonempty subsets. '
                   '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable. '
                   'No test is recomputed from displayed nodes.')
    payload = norma_payload(nodes, edges, groups, center, name=f'{network} · {center} OCG ego', names=names, labels=lookup)
    st.download_button('Download NORMA3 files + API payload', norma_export_zip(payload),
                       f'{network}_{center}_norma.zip', 'application/zip', on_click='ignore')
    st.download_button('Download interactive hull plot', fig.to_html(include_plotlyjs=True),
                       f'{network}_{center}_overlap.html', 'text/html', on_click='ignore')
    with st.expander('Send this ego network to a NORMA API server'):
        st.caption('The ZIP contains network/annotation TSV files and an API payload for NORMA3. '
                   'Only displayed nodes, edges and outlined '
                   'cluster memberships are sent. [API documentation](https://www.normaviz.org/).')
        server = st.text_input('NORMA server address', 'http://127.0.0.1:8000', key='ocg_norma_server')
        if st.button('Send displayed ego network to NORMA', key='ocg_norma_send'):
            try:
                url = send_to_norma(payload, server)
                st.link_button('Open exported network in NORMA', url)
            except (ValueError, KeyError) as exc:
                st.error(f'NORMA export failed: {exc}')
            except Exception as exc:
                st.error(f'Could not contact the NORMA API server: {exc}')
