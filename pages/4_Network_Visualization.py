"""Interactive viewers for a shared subset of a canonical interactome."""
from pathlib import Path
from importlib.util import find_spec
import re
import time
from urllib.parse import parse_qs, urljoin, urlsplit

import polars as pl
import requests
import streamlit as st

from clusterannotation import network_population, snapshot_sha256
from cluster_annotation_ui import annotation_panel, atlas_memberships
from cytoscape_bridge import render_cytoscapejs_html, subnetwork_to_graph_context
from cytoscape_bridge import DEFAULT_BASE_URL, export_cytoscape_bundle_zip, export_graphml_string, send_to_cytoscape
from network_view import network_3d_figure, styled_network_nodes, static_network_figure, static_image_bytes, pyvis_network_html
from network_view import sigma_layout_positions, sigma_network_html
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
def sigma_positions(nodes: pl.DataFrame, edges: pl.DataFrame, engine: str):
    """Cache fixed coordinates, including component packing for igraph layouts."""
    return sigma_layout_positions(nodes, edges, engine)


@st.cache_data(show_spinner=False, max_entries=6)
def preview_clustering(nodes: pl.DataFrame, edges: pl.DataFrame, algorithm: str) -> pl.DataFrame:
    graph = subnetwork_to_graph_context(nodes.select(pl.col('protein').alias('id')), edges)
    result = run_cached_clustering(graph, algorithm=algorithm, use_cache=False)
    return atlas_memberships(result['node_frame'])


@st.cache_data(show_spinner=False, max_entries=24)
def preview_tables(
    path: Path, mtime_ns: int, size: int, accession: str, limit: int | None,
    scope: str, seeds: tuple[str, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, bool]:
    """Select a bounded, repeatable edge sample and its incident proteins."""
    frame = pl.scan_parquet(path)
    score_columns = [name for name in dict.fromkeys(SCORE_COLUMNS.values()) if name in frame.collect_schema()]
    def matches(column, proteins):
        return pl.col(column).is_in(proteins) | pl.col(column).str.split('|').list.eval(
            pl.element().is_in(proteins)).list.any()
    if accession:
        frame = frame.filter(matches('a', [accession]) | matches('b', [accession]))
    if scope != ALL:
        proteins = list(seeds)
        if scope == MOON_NEIGHBORS:
            frame = frame.filter(matches('a', proteins) | matches('b', proteins))
        else:
            frame = frame.filter(matches('a', proteins) & matches('b', proteins))
    frame = frame.select('a', 'b', *score_columns)
    sampled = (frame if limit is None else (
        frame
        .with_columns(pl.struct('a', 'b').hash(seed=42).alias('_order'))
        .bottom_k(limit + 1, by=['_order', 'a', 'b'])
        .sort('_order', 'a', 'b')
        .drop('_order')
    )).collect(engine='streaming')
    edges = (sampled if limit is None else sampled.head(limit)).rename({'a': 'source', 'b': 'target'})
    nodes = (
        pl.concat([edges['source'], edges['target']]).rename('protein').to_frame()
        .group_by('protein').len(name='subnetwork_degree')
        .with_columns(pl.col('protein').alias('label'))
        .sort('protein')
    )
    present = list(matching_nodes(nodes['protein'], seeds))
    nodes = nodes.with_columns(pl.col('protein').is_in(present).alias('is_moonlighting')).with_columns(
        pl.when('is_moonlighting').then(pl.lit('Moonlighting')).otherwise(pl.lit('Background')).alias('node_role'))
    return nodes, edges, limit is not None and sampled.height > limit


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
st.write('Explore a canonical interactome, a protein’s direct interactions, or a moonlighting protein set and its direct partners in 2D or 3D.')

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
        'Edge limit', [250, 1000, 2000, 5000, 10000, 20000, 50000, 100000], index=2, key='viz_limit',
        help='Choose a repeatable edge sample. Large selections show performance warnings but remain renderable in every viewer.',
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
            st.caption('Includes every interaction with at least one endpoint in the selected protein set; excludes edges between two neighbors outside the set.')

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
    'Visualization tool', ['Cytoscape.js', 'Apache ECharts', 'Sigma.js', 'PyVis', 'Static', '3D', 'NORMA'],
    default='Cytoscape.js', required=True, key='viz_tool',
)
title = f'{network_name} · {accession}' if accession else network_name
if sources:
    title = f'{network_name} · {scope} ({", ".join(GROUP_LABELS[source] for source in sources)})'
    displayed_seeds = len(matching_nodes(nodes['protein'], seeds))
    st.caption(f'{displayed_seeds:,} displayed proteins belong to the selected moonlighting sets.')

highlighted = []
st.caption('Blue diamonds / blue nodes: selected moonlighting protein set. Gray: other proteins. '
           'Node size reflects degree in the displayed graph. These are descriptive views; no significance test applies. '
           'Spatial proximity is a layout property and does not validate moonlighting.')
annotation = st.expander('Cluster and annotate this view', key='viz_annotation', on_change='rerun')
if annotation.open:
    with annotation:
        st.caption('Clustering uses the displayed interactions. Enrichment uses all proteins in the selected '
                   'network or saved version. A sampled view can produce different clusters from the full network.')
        algorithms = {'louvain': 'Louvain',
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

if edges.height > 20000 and viewer != 'Sigma.js':
    st.warning('More than 20,000 displayed interactions: rendering may be slow or use substantial memory. Sigma.js is an option for larger graphs.')
if viewer == 'Sigma.js':
    engines = (['igraph DrL', 'igraph FR'] if find_spec('igraph') else []) + (['graph-tool SFDP'] if find_spec('graph_tool') else []) + ['Circular']
    sigma_engine = st.selectbox('Python layout engine', engines, key='viz_sigma_engine')
    st.caption('Sigma.js 3 / Graphology use WebGL with fixed Python coordinates; no force simulation runs in the browser. '
               'Layout time and browser speed depend on the graph and hardware. '
               'Hover for neighbors and details, search by accession, and pan or zoom. Orange labels identify the selected cluster.')
    if nodes.height > 20000 or edges.height > 100000:
        st.warning('More than 20,000 displayed proteins or 100,000 interactions: layout and rendering may be slow or use substantial memory.')
    sigma_key = (selection, sigma_engine)
    sigma_state = st.session_state.get('viz_sigma_layout')
    if sigma_state and sigma_state['key'] != sigma_key:
        st.session_state.pop('viz_sigma_layout', None)
        sigma_state = None
    if st.button('Build Python layout', key='viz_sigma_build', type='primary'):
        try:
            with st.spinner('Computing fixed layout in Python…'):
                positions = sigma_positions(nodes.select('protein'), edges.select('source', 'target'), sigma_engine)
            sigma_state = {'key': sigma_key, 'positions': positions}
            st.session_state['viz_sigma_layout'] = sigma_state
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.error(f'Could not build the selected layout. Try Circular or another installed engine. {exc}')
    if sigma_state:
        sigma_nodes = styled_network_nodes(nodes, highlighted).join(pl.DataFrame({
            'protein': list(sigma_state['positions']),
            'x': [p[0] for p in sigma_state['positions'].values()],
            'y': [p[1] for p in sigma_state['positions'].values()],
        }), on='protein', how='left')
        sigma_html = sigma_network_html(sigma_nodes, edges, sigma_state['positions'], f'{title} · {coverage}')
        st.iframe(sigma_html, height=750, alt=f'{title}: fixed-layout Sigma.js WebGL network')
        st.download_button('Download interactive Sigma HTML', sigma_html, file_name='network_sigma.html', mime='text/html', on_click='ignore')
        st.download_button('Download displayed GraphML with coordinates', data=lambda: export_graphml_string(sigma_nodes, edges),
                           file_name='network_sigma.graphml', mime='application/graphml+xml', on_click='ignore')
    else:
        st.info('Build a layout to display this selection. Coordinates are cached and reused until the network selection or layout engine changes.')
elif viewer == 'Cytoscape.js':
    st.caption('Search by accession, switch layouts, inspect nodes or edges, and export a PNG.')
    viewer_nodes = nodes
    if highlighted:
        viewer_nodes = nodes.with_columns(
            pl.col('protein').is_in(highlighted).alias('cluster_selected')
        )
    st.iframe(
        render_cytoscapejs_html(
            viewer_nodes, edges, layout_name='concentric', height=VIEW_HEIGHT - 10, show_bridge_controls=False,
            color_by='node_role',
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
    chart_nodes = chart_nodes.with_columns(pl.struct(
        pl.when(nodes['is_moonlighting']).then(pl.lit('#0072B2'))
        .otherwise(pl.lit('#9CA3AF')).alias('color'),
        pl.when(pl.col('name').is_in(highlighted)).then(pl.lit('#F39C12'))
        .otherwise(pl.lit('#374151')).alias('borderColor'),
        pl.lit(2).alias('borderWidth'),
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
elif viewer == 'PyVis':
    if not find_spec('pyvis'):
        st.info('PyVis is unavailable in this environment. Install the project dependencies to enable it (pyvis>=0.3.2,<0.4).')
    else:
        if nodes.height > 2000:
            st.warning('More than 2,000 displayed proteins: PyVis rendering and force simulation may be slow. Turn off simulation for a fixed layout.')
        physics = st.toggle('Enable force simulation', value=True, key='viz_pyvis_physics')
        show_labels = st.checkbox('Show protein labels', value=nodes.height <= 100, key='viz_pyvis_labels')
        st.caption('Drag proteins, pan and zoom, or hover for accession, protein-set membership and degree. '
                   'Turn off simulation for a fixed layout. The HTML download embeds its JavaScript and highlight colors.')
        pyvis_html = pyvis_network_html(styled_network_nodes(nodes, highlighted), edges,
                                         physics=physics, labels=show_labels)
        st.iframe(pyvis_html, height=VIEW_HEIGHT + 20, alt=f'{title}: interactive PyVis network with protein-set highlighting')
        st.download_button('Download interactive PyVis HTML', pyvis_html,
                           file_name='network_pyvis.html', mime='text/html', on_click='ignore')
elif viewer == 'Static':
    engines = ['NetworkX'] + (['igraph'] if find_spec('igraph') else []) + ['graph-tool']
    engine = st.selectbox('Layout engine', engines, key='viz_static_engine')
    if not find_spec('igraph'):
        st.caption('igraph is optional (the clustering extra). NetworkX layouts work without it.')
    graph_tool_ready = engine != 'graph-tool' or find_spec('graph_tool') is not None
    if not graph_tool_ready:
        st.info('graph-tool is not installed in this Python environment. On Windows it requires running the app inside WSL or Docker; '
                'it cannot be installed into this native Windows environment with pip. NetworkX and igraph remain available.')
        st.link_button('graph-tool installation instructions', 'https://graph-tool.skewed.de/installation.html')
    layouts = ['SFDP', 'Spring', 'Circular'] if engine == 'graph-tool' else ['Spring', 'Circular'] + (['Kamada–Kawai'] if engine == 'NetworkX' else [])
    static_layout = st.selectbox('Static layout', layouts, key='viz_static_layout')
    labels = st.checkbox('Show protein accessions (small graphs)', disabled=nodes.height > 100, key='viz_static_labels')
    if not graph_tool_ready:
        st.caption('Install graph-tool in the same supported environment that runs Streamlit, then select this engine again. Network downloads remain available below.')
    else:
        if nodes.height > 2000:
            st.warning('More than 2,000 displayed proteins: static layout and image export may be slow or use substantial memory. Circular avoids force-layout computation.')
        if static_layout == 'Kamada–Kawai' and nodes.height > 500:
            st.warning('Kamada–Kawai computes all-pairs distances; above 500 proteins it may be slow or use substantial memory. Spring or Circular may be faster.')
        with st.spinner('Drawing static network…'):
            static_fig = static_network_figure(styled_network_nodes(nodes, highlighted), edges,
                f'{title} · {coverage}', engine=engine, layout=static_layout, labels=labels and nodes.height <= 100)
            st.image(static_image_bytes(static_fig, 'png'), alt=f'{title}: static network with protein-set highlighting', width='stretch')
        for fmt, mime in [('png', 'image/png'), ('svg', 'image/svg+xml'), ('pdf', 'application/pdf')]:
            st.download_button(f'Download {fmt.upper()} figure', data=lambda fmt=fmt: static_image_bytes(static_fig, fmt),
                               file_name=f'network.{fmt}', mime=mime, on_click='ignore', key=f'viz_static_{fmt}')
        st.caption('Image downloads contain the displayed graph, including any preview sampling. SVG and PDF retain vector graphics and highlight colors.')
elif viewer == '3D':
    if nodes.height > 2000:
        st.warning('More than 2,000 displayed proteins: the 3D force layout and browser rendering may be slow or use substantial memory.')
    with st.spinner('Computing 3D layout…'):
        st.plotly_chart(network_3d_figure(nodes, edges, title, highlighted), width='stretch',
                        key='viz_3d', alt=f'{title}: rotatable 3D network with highlighted moonlighting proteins')
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

with st.expander('Download network / open in Cytoscape Desktop'):
    st.caption('Import the ZIP’s GraphML via File → Import → Network → File, or send directly to a running Cytoscape Desktop. '
               'Membership and displayed degree are retained as node attributes. Direct sending applies the blue moonlighting style.')
    full_export = st.checkbox('Use all matching interactions (ignore browser edge limit)', value=True, key='viz_full_export')
    base_url = st.text_input('Cytoscape CyREST URL', value=DEFAULT_BASE_URL, key='viz_cyrest')
    def export_tables():
        if full_export:
            export_nodes, export_edges = preview_tables(path, file_info.st_mtime_ns, file_info.st_size, accession, None, scope, seeds)[:2]
        else:
            export_nodes, export_edges = nodes, edges
        return styled_network_nodes(export_nodes, highlighted,
            ', '.join(GROUP_LABELS[s] for s in (sources or SOURCES)) if scope != ALL else 'No protein set selected'), export_edges
    st.caption('GraphML preserves is_moonlighting, protein_set, color, shape, size, cluster_selected and border_color attributes. '
               'GraphML has no universal visual style: map these attributes in your viewer to restore the colors and shapes. '
               'In Cytoscape, map node fill to color and border paint to border_color. Orange borders mark the selected displayed cluster; '
               'full export does not recompute cluster membership for additional proteins.')
    st.download_button('Download GraphML', data=lambda: export_graphml_string(*export_tables()),
                       file_name='network.graphml', mime='application/graphml+xml', on_click='ignore')
    st.download_button('Download node attributes', data=lambda: export_tables()[0].write_csv(),
                       file_name='network_nodes.csv', mime='text/csv', on_click='ignore')
    st.download_button('Download Cytoscape bundle',
                       data=lambda: export_cytoscape_bundle_zip(*export_tables(), title=title,
                           metadata={'network': str(path), 'scope': scope, 'sources': sources,
                                     'full_selection': full_export, 'browser_truncated': truncated}),
                       file_name='network_cytoscape.zip', mime='application/zip', on_click='ignore')
    if st.button('Send to Cytoscape Desktop', key='viz_send_cytoscape'):
        try:
            with st.spinner('Sending network to Cytoscape…'):
                exported_nodes, exported_edges = export_tables()
                sent = send_to_cytoscape(exported_nodes, exported_edges, title=title, base_url=base_url)
            st.success(f'Opened {exported_nodes.height:,} proteins and {exported_edges.height:,} interactions in Cytoscape (network {sent["network_suid"]}).')
        except Exception as exc:
            st.error(f'Could not send to Cytoscape. Start Cytoscape Desktop and check its CyREST URL. {exc}')
