"""Interactive viewers for a shared subset of a canonical interactome."""
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urljoin, urlsplit

import polars as pl
import requests
import streamlit as st

from annotate_clusters import network_population, snapshot_sha256
from cluster_annotation_ui import annotation_panel, atlas_memberships
from cytoscape_bridge import render_cytoscapejs_html, subnetwork_to_graph_context
from moonlighting import GROUP_LABELS, SOURCES, cohort_ids, load_moon_data, run_cached_clustering
from networks import SCORE_COLUMNS, _fingerprint, discover_datasets, matching_nodes

DATA = Path(__file__).resolve().parents[1] / 'data'
NORMA_ORIGIN = 'https://www.normaviz.org'
VIEW_HEIGHT = 650
ALL = 'All interactions'
MOON_ONLY = 'Moonlighting only'
MOON_NEIGHBORS = 'Moonlighting + first neighbors'


@st.cache_data(show_spinner=False, max_entries=4)
def moonlighting_data(csv_mtime: int, mapping_mtime: int) -> pl.DataFrame:
    return load_moon_data(DATA)


@st.cache_data(show_spinner=False, max_entries=4)
def annotation_population(path: Path, mtime_ns: int, size: int) -> pl.DataFrame:
    return network_population(path)


@st.cache_data(show_spinner=False, max_entries=6)
def preview_clustering(nodes: pl.DataFrame, edges: pl.DataFrame, algorithm: str) -> pl.DataFrame:
    graph = subnetwork_to_graph_context(nodes.select(pl.col('protein').alias('id')), edges)
    result = run_cached_clustering(graph, algorithm=algorithm, use_cache=False)
    return atlas_memberships(result['node_frame'])


@st.cache_data(show_spinner=False, max_entries=24)
def preview_tables(
    path: Path, mtime_ns: int, size: int, accession: str, limit: int,
    scope: str, seeds: tuple[str, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, bool]:
    """Select a bounded, repeatable edge sample and its incident proteins."""
    frame = pl.scan_parquet(path)
    score_columns = [name for name in SCORE_COLUMNS.values() if name in frame.collect_schema()]
    def matches(column, proteins):
        return pl.col(column).is_in(proteins) | pl.col(column).str.split('|').list.eval(
            pl.element().is_in(proteins)).list.any()
    if accession:
        frame = frame.filter(matches('a', [accession]) | matches('b', [accession]))
    if scope != ALL:
        proteins = list(seeds)
        if scope == MOON_NEIGHBORS:
            incident = frame.filter(matches('a', proteins) | matches('b', proteins))
            proteins = (
                pl.concat([incident.select(pl.col(endpoint).alias('protein')) for endpoint in ('a', 'b')])
                .unique().collect(engine='streaming')['protein'].to_list()
            )
        if scope == MOON_NEIGHBORS:
            frame = frame.filter(pl.col('a').is_in(proteins) & pl.col('b').is_in(proteins))
        else:
            frame = frame.filter(matches('a', proteins) & matches('b', proteins))
    sampled = (
        frame.select('a', 'b', *score_columns)
        .with_columns(pl.struct('a', 'b').hash(seed=42).alias('_order'))
        .bottom_k(limit + 1, by=['_order', 'a', 'b'])
        .sort('_order', 'a', 'b')
        .drop('_order')
        .collect(engine='streaming')
    )
    edges = sampled.head(limit).rename({'a': 'source', 'b': 'target'})
    nodes = (
        pl.concat([edges['source'], edges['target']]).rename('protein').to_frame()
        .group_by('protein').len(name='subnetwork_degree')
        .with_columns(pl.col('protein').alias('label'))
        .sort('protein')
    )
    return nodes, edges, sampled.height > limit


def create_norma_session(name: str, edges: pl.DataFrame) -> tuple[str, float]:
    payload = {
        'name': name,
        'edges': edges.select('source', 'target').to_dicts(),
        'settings': {'layout': 'fr'},
    }
    response = requests.post(
        f'{NORMA_ORIGIN}/api/external', json=payload, timeout=30, allow_redirects=False,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get('url'), str):
        raise ValueError('NORMA returned no session link.')
    url = urljoin(f'{NORMA_ORIGIN}/', data['url'])
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


st.subheader('Network viewers')
st.write('Explore a canonical interactome, a protein’s direct interactions, or a moonlighting subnetwork in three viewers.')

datasets = discover_datasets(DATA, isoforms=st.session_state.get('isoform_policy', 'collapse'))
if not datasets:
    st.info('Prepare a canonical network on Interactomes Overview first.')
    st.stop()

analysis = st.session_state.get('visualization_analysis_context', {})
preferred = [name for name in st.session_state.get('overview_networks', []) if name in datasets]
network_options = list(dict.fromkeys([*preferred, *datasets]))
default_network = analysis.get('network')
if default_network not in network_options:
    default_network = network_options[0]

with st.container(border=True):
    dataset_col, limit_col = st.columns(2)
    network_name = dataset_col.selectbox('Network or saved version', network_options,
                                         index=network_options.index(default_network), key='viz_network')
    limit = limit_col.selectbox(
        'Edge limit', [250, 1000, 2000, 5000, 10000, 20000], index=2, key='viz_limit',
        help='Larger graphs take longer to lay out and draw in the browser.',
    )
    scope = st.segmented_control(
        'Network scope', [ALL, MOON_ONLY, MOON_NEIGHBORS], default=MOON_NEIGHBORS if analysis else ALL,
        required=True, key='viz_scope',
    )
    accession = ''
    sources = []
    seeds = ()
    if scope == ALL:
        accession = st.text_input(
            'Focus protein (optional)', placeholder='P04406', max_chars=16, key='viz_accession',
            help='Enter the exact UniProt ID, including its suffix if it is a retained isoform.',
        ).strip().upper()
    else:
        sources = st.multiselect(
            'Moonlighting sets (empty = all)', SOURCES, default=analysis.get('sources', ['moonlitdb']),
            format_func=GROUP_LABELS.__getitem__, key='viz_sources', persist_state='page',
            help='Includes proteins in any selected set, with overlapping memberships deduplicated.',
        )
        moon_csv = DATA / 'allmoon_with_moonlitdb_human.csv'
        mapping = DATA / 'mappings/moonlighting_uniprot.json'
        if not moon_csv.is_file() or not mapping.is_file():
            st.info('Prepare the moonlighting data and UniProt mappings on Topological Analysis first.')
            st.stop()
        moon = moonlighting_data(moon_csv.stat().st_mtime_ns, mapping.stat().st_mtime_ns)
        seeds = tuple(sorted(cohort_ids(
            moon, sources, analysis.get('require_all', False),
            analysis.get('reviewed_only', False), analysis.get('min_pubs', 0),
        )[0]))
        if analysis:
            st.caption('Uses the source matching, review status and publication filters from Topological Analysis.')
        if scope == MOON_NEIGHBORS:
            st.caption('Includes interactions between any two proteins in the set or its one-hop neighborhood.')

if accession and not re.fullmatch(r'[A-Z0-9]+(?:-\d+)?', accession):
    st.warning('Enter one UniProt accession, with its isoform suffix if applicable.')
    st.stop()

path = datasets[network_name]
file_info = path.stat()
preview_inputs = (path, file_info.st_mtime_ns, file_info.st_size, accession, limit, scope, seeds)
selection = (*preview_inputs, tuple(sources))
with st.spinner('Selecting interactions…'):
    nodes, edges, truncated = preview_tables(*preview_inputs)

session = st.session_state.get('viz_norma_session')
if session and (session['selection'] != selection or session['expires'] <= time.monotonic()):
    st.session_state.pop('viz_norma_session')
    session = None

if edges.is_empty():
    st.info(f'No interactions match this selection in {network_name}.')
    st.stop()

description = f'{accession} direct interactions' if accession else scope
coverage = f'Repeatable sample · {description}' if truncated else description
st.caption(f'{coverage} · {nodes.height:,} incident proteins · {edges.height:,} interactions')

viewer = st.segmented_control(
    'Visualization tool', ['Cytoscape.js', 'Apache ECharts', 'NORMA'],
    default='Cytoscape.js', required=True, key='viz_tool',
)
title = f'{network_name} · {accession}' if accession else network_name
if sources:
    title = f'{network_name} · {scope} ({", ".join(GROUP_LABELS[source] for source in sources)})'
    displayed_seeds = len(matching_nodes(nodes['protein'], seeds))
    st.caption(f'{displayed_seeds:,} displayed proteins belong to the selected moonlighting sets.')

highlighted = []
annotation = st.expander('Cluster and annotate this view', key='viz_annotation', on_change='rerun')
if annotation.open:
    with annotation:
        st.caption('Clustering uses the displayed interactions. Enrichment uses all proteins in the selected '
                   'network or saved version. A sampled view can produce different clusters from the full network.')
        algorithms = {'louvain': 'Louvain', 'ego_splitting': 'Ego splitting · overlapping',
                      'mcode_complexes': 'MCODE with fluff · overlapping'}
        algorithm = st.selectbox('Clustering algorithm', list(algorithms), format_func=algorithms.__getitem__,
                                 key='viz_annotation_algorithm')

        def preview_memberships():
            source = {
                'network': _fingerprint(path),
                'visualization': {
                    'network': network_name, 'scope': scope, 'focus': accession, 'moonlighting_sets': sources,
                    'edge_limit': limit, 'displayed_edges': edges.height, 'truncated': truncated,
                    'edges_sha256': snapshot_sha256(edges.select('source', 'target').rows()),
                    'algorithm': algorithm, 'resolution': 1.0, 'seed': 42,
                },
            }
            return preview_clustering(nodes, edges, algorithm), annotation_population(
                path, file_info.st_mtime_ns, file_info.st_size,
            ), source

        result, cluster = annotation_panel(
            'viz_cluster_annotation', (selection, algorithm), preview_memberships, DATA,
            button='Cluster and annotate displayed graph',
        )
        if cluster is not None:
            highlighted = (
                result['memberships'].filter(pl.col('community') == cluster)['protein'].to_list()
            )
            if viewer == 'NORMA':
                st.caption('Switch to Cytoscape.js or Apache ECharts to highlight the selected cluster.')
            else:
                st.caption(f'Highlighted cluster {cluster} · {len(highlighted):,} displayed proteins')

if viewer == 'Cytoscape.js':
    st.caption('Search by accession, switch layouts, inspect nodes or edges, and export a PNG.')
    viewer_nodes = nodes
    if highlighted:
        viewer_nodes = nodes.with_columns(
            pl.when(pl.col('protein').is_in(highlighted)).then(pl.lit('Cluster member')).alias('cluster_highlight')
        )
    st.iframe(
        render_cytoscapejs_html(
            viewer_nodes, edges, layout_name='concentric', height=VIEW_HEIGHT - 10, show_bridge_controls=False,
            color_by='cluster_highlight' if highlighted else 'node_role',
        ),
        height=VIEW_HEIGHT, alt=f'{title} in Cytoscape.js',
    )
elif viewer == 'Apache ECharts':
    layout = st.segmented_control(
        'Layout', ['circular', 'force'], default='circular', required=True,
        format_func=str.capitalize, key='viz_echarts_layout',
    )
    chart_nodes = nodes.select(
        pl.col('protein').alias('name'),
        pl.col('subnetwork_degree').alias('value'),
        (6 + 3 * pl.col('subnetwork_degree').sqrt()).clip(upper_bound=30).alias('symbolSize'),
    )
    if highlighted:
        chart_nodes = chart_nodes.with_columns(pl.struct(
            pl.when(pl.col('name').is_in(highlighted)).then(pl.lit('#0072B2'))
            .otherwise(pl.lit('#9CA3AF')).alias('color')
        ).alias('itemStyle'))
    st.caption('Drag to pan, scroll to zoom, and hover over a protein to see its degree in this view.')
    st.echarts_chart(
        {
            'tooltip': {'trigger': 'item', 'renderMode': 'richText'},
            'series': [{
                'type': 'graph', 'layout': layout, 'roam': True, 'animation': False,
                'data': chart_nodes.to_dicts(),
                'links': edges.select('source', 'target').to_dicts(),
                'emphasis': {'focus': 'adjacency', 'label': {'show': True}},
                'lineStyle': {'opacity': 0.45},
            }],
        },
        height=VIEW_HEIGHT, alt=f'{title} network with protein sizes based on displayed degree',
    )
else:
    st.caption('Creating a NORMA session sends the displayed UniProt pairs to normaviz.org for temporary storage.')
    if st.button('Load in NORMA', type='primary', key='viz_load_norma'):
        with st.spinner('Creating the NORMA session…'):
            try:
                url, expires = create_norma_session(title, edges)
            except (requests.RequestException, ValueError, TypeError) as exc:
                st.error(f'Could not create a NORMA session. {exc}')
            else:
                session = {'selection': selection, 'url': url, 'expires': expires}
                st.session_state['viz_norma_session'] = session
    if session:
        st.link_button('Open NORMA in a new tab', session['url'], icon=':material/open_in_new:')
        st.iframe(session['url'], height=780, alt=f'{title} in NORMA')

st.download_button(
    'Download displayed interactions', edges.write_csv(),
    file_name=f'{network_name}{f"_{accession}" if accession else ""}_interactions.csv',
    mime='text/csv', on_click='ignore', icon=':material/download:',
)
