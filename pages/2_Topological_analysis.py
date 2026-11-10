"""Cross-network topological analyses of moonlighting proteins.

Every analysis runs on every selected interactome and is shown side by side. The first tab gives the
overview statistics; later tabs add progressively more specialised topology.
"""
from pathlib import Path
from itertools import combinations
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import json
import hashlib
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cluster_annotation_ui import annotation_panel, atlas_memberships
from cytoscape_bridge import (
    CLUSTERMAKER_ALGORITHMS, CYTOHUBBA_METRICS, NETWORK_ANALYZER_METRICS, DEFAULT_CYREST_PORT,
    build_base_url, build_subnetwork_tables, build_topology_network_tables,
    check_cytoscape_connection, compute_cluster_enrichment_and_bridging, cytoscape_topology_comparison,
    extract_cytohubba_scores, fetch_cytoscape_table_polars,
    push_clustering_to_cytoscape, run_clustermaker2, run_network_analyzer, send_to_cytoscape,
)
from networks import SCORE_COLUMNS, _fingerprint, discover_datasets, map_ids, matching_nodes
from moonlighting_extensions import (
    annotated_partner_partition, cohort_filtration, cohort_separation, edge_geometry, ego_geometry,
    electrical_and_walk_roles, go_neighborhood_geometry, graphlet_profiles,
    metadata_wiring, module_brokerage, multiplex_roles, path_centralities,
    removal_efficiency, spectral_roles,
)
from moonlighting import (
    ADVANCED_EDGE_LIMIT, CLUSTERING_CATALOG, COLORSCALE, DEFAULT_GAF, DEFAULT_OBO, GROUP_COLORS, GROUP_LABELS,
    SIGNIFICANCE_LEGEND, SOURCES, bh_adjust, cached_analysis_run, centrality_heatmap, cohort_ids,
    cohesion_analysis, community_analysis, compare_clustering_algorithms, compare_source_sets,
    cross_network_analysis, cross_network_metric_figure, go_enrichment, graph_from_path,
    hub_enrichment_heatmap, load_moon_data, multi_network_centrality_comparison,
    pool_network_comparisons, prepare_go_annotations, protein_set_network_summary, random_walk_analysis, robustness_analysis,
    significance_label, stack_network_tables, wiring_analysis,
)

DATA = ROOT / 'data'
LOG_METRICS = {'degree', 'strength', 'pagerank', 'betweenness_approx', 'bridging_centrality'}


def setting(widget, *args, key, **kwargs):
    """Remember widget values across page switches."""
    stored = st.session_state.setdefault('moon_analysis_settings', {})
    if key in stored:
        saved = st.session_state.get(key, stored[key])
        positional = list(args)
        if widget == st.selectbox:
            options = positional[1] if len(positional) > 1 else kwargs['options']
            if saved in options:
                kwargs['index'] = list(options).index(saved)
            else:
                st.session_state.pop(key, None)
                stored.pop(key, None)
        else:
            position = 1 if widget == st.checkbox else 3
            if len(positional) > position:
                positional[position] = saved
            else:
                kwargs['value'] = saved
        args = tuple(positional)
    if key in st.session_state:
        kwargs.pop('index', None)
        kwargs.pop('value', None)
        kwargs.pop('default', None)
    value = widget(*args, key=key, **kwargs)
    stored[key] = value
    return value


def pretty(name):
    return str(name).replace('_', ' ').capitalize()


st.set_page_config(page_title='Topological Analysis | Human Interactome Atlas', page_icon=':material/hub:', layout='wide')
st.markdown(
    '<style>.block-container{padding-top:2.2rem}'
    '.stTabs [data-baseweb="tab"] p{font-size:1.02rem;font-weight:600}</style>',
    unsafe_allow_html=True,
)

datasets = discover_datasets(DATA, isoforms=st.session_state.get('isoform_policy', 'collapse'))
if not datasets:
    st.title('Topological analysis')
    st.info('Prepare canonical networks on the overview page first.')
    st.stop()


@st.cache_data(show_spinner=False)
def moon_data(mtime, mapping_mtime):
    return load_moon_data(DATA)


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_graph(path, mtime, size):
    return graph_from_path(Path(path))


@st.cache_data(show_spinner=False)
def cached_edge_count(path, mtime, size):
    return pl.scan_parquet(path).select(pl.len()).collect().item()


@st.cache_data(show_spinner=False)
def cached_provenance(path, mtime):
    return json.loads(Path(path).read_text())


mapping_file = DATA / 'mappings/moonlighting_uniprot.json'
raw_path = DATA / 'allmoon_with_moonlitdb_human.csv'
if not mapping_file.exists():
    st.title('Topological analysis')
    st.info('Resolve the moonlighting protein IDs before running the analyses.')
    if st.button('Map moonlighting accessions with UniProt'):
        with st.spinner('Resolving UniProt IDs…'):
            raw = pl.read_csv(raw_path, schema_overrides={'uniprot': pl.String})
            result = map_ids(raw['uniprot'].drop_nulls().str.split('-').list.first().to_list(), 'UniProtKB_AC-ID', DATA / 'mappings')
            mapping_file.write_text(json.dumps(result, indent=2))
        st.rerun()
    st.stop()

moon = moon_data(raw_path.stat().st_mtime_ns, mapping_file.stat().st_mtime_ns)

default_networks = [
    name for name, path in datasets.items()
    if path.parent.name != 'versions' and (name != 'huMAP2' or 'huMAP2_0.05' not in datasets)
]
default_networks += [v for v in ('huMAP2_0.05', 'huMAP2_0.95') if v in datasets and v not in default_networks]
default_networks = default_networks or list(datasets)[:4]

# ---------------------------------------------------------------------------
# Sidebar: one network selection and one protein set for every tab
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title('Topological analysis')
    active_nets = st.multiselect(
        'Interactomes to compare', list(datasets), default=default_networks,
        key='topo_networks', persist_state='page',
        help='Every analysis runs on all of these networks and shows them side by side.',
    )
    st.subheader('Moonlighting protein set')
    set_options = [
        'moonlitdb', 'curated', 'moonprot', 'metamoon', 'multitask',
        'moondb_curated', 'moondb_predicted', 'all', 'custom',
    ]
    set_labels = {
        'moonlitdb': 'MoonLitDB',
        'curated': 'Curated',
        'moonprot': 'MoonProt',
        'metamoon': 'MetaMoon',
        'multitask': 'Multitask',
        'moondb_curated': 'MoonDB curated',
        'moondb_predicted': 'MoonDB predicted',
        'all': 'All sources (Union)',
        'custom': 'Custom combination…',
    }
    chosen_set = setting(
        st.selectbox, 'Protein set', set_options,
        format_func=lambda s: set_labels.get(s, s),
        key='moon_protein_set',
    )
    if chosen_set == 'custom':
        sources = st.multiselect('Sources (empty = all rows)', SOURCES, default=['moonlitdb'],
                                 format_func=lambda source: GROUP_LABELS.get(source, source),
                                 key='moon_sources', persist_state='page')
        all_sources = setting(st.checkbox, 'Require all selected sources', key='moon_all_sources')
        set_name = (' ∩ '.join([GROUP_LABELS.get(s, s) for s in sources]) if all_sources
                    else ' + '.join([GROUP_LABELS.get(s, s) for s in sources])) if sources else 'All sources'
    elif chosen_set == 'all':
        sources = None
        all_sources = False
        set_name = 'All sources'
    else:
        sources = [chosen_set]
        all_sources = False
        set_name = set_labels[chosen_set]
    reviewed_only = setting(st.checkbox, 'Reviewed source entries only', key='moon_reviewed')
    min_pubs = setting(st.number_input, 'Minimum MoonLitDB publications', 0, 10000, 0, 1, key='moon_min_pubs')
    st.subheader('Computation')
    seed = setting(st.number_input, 'Random seed', 0, 2**31 - 1, 42, 1, key='moon_seed')
    auto_run = st.toggle('Compute automatically', value=True, key='moon_auto_run',
                         help='Loads cached results or computes them when you open an analysis. '
                              'Turn off to compute only with the Recalculate button.')

cohort, selected = cohort_ids(moon, sources, all_sources, reviewed_only, min_pubs)
with st.sidebar:
    st.caption(f"Selected: {set_name} · {selected['input_accession'].n_unique():,} source accessions · {len(cohort):,} resolved UniProt IDs · "
               f"{selected.filter(pl.col('mapping_status') == 'unmapped')['input_accession'].n_unique():,} unmapped. "
               'Source flags may overlap.')

st.caption('HUMAN INTERACTOME ATLAS')
st.title('Topology of moonlighting proteins across interactomes')
st.caption(f'Moonlighting protein set: **{set_name}**.')
st.caption('Each network is compared with its own background: all other proteins in that network, which are unlabelled, '
           'not validated non-moonlighting controls. Results are cached in `data/cache/` and `data/analysis_runs/`.')

if not active_nets:
    st.info('Select at least one interactome in the sidebar.')
    st.stop()
if not cohort:
    st.warning(f'No mapped proteins satisfy the filters for {set_name}. The Source sets tab can still compare individual sources.')

st.session_state['visualization_analysis_context'] = {
    'network': active_nets[0], 'sources': sources, 'require_all': all_sources,
    'reviewed_only': reviewed_only, 'min_pubs': int(min_pubs),
    'protein_set': chosen_set, 'protein_set_name': set_name,
}
base_context = {
    'cohort': sorted(cohort), 'sources': sources,
    'protein_set': chosen_set, 'protein_set_name': set_name,
    'source_mode': 'intersection' if all_sources else 'union',
    'reviewed_source_only': reviewed_only, 'min_moonlitdb_publications': int(min_pubs),
    'random_seed': int(seed),
    'moon_csv_mtime_ns': raw_path.stat().st_mtime_ns,
    'moon_mapping_mtime_ns': mapping_file.stat().st_mtime_ns,
}


def graph_for(name):
    p = datasets[name]
    info = p.stat()
    g = cached_graph(str(p), info.st_mtime_ns, info.st_size)
    if not len(g['nodes']):
        raise ValueError('The network has no retained edges or nodes.')
    return g


def edges_of(name):
    p = datasets[name]
    info = p.stat()
    return cached_edge_count(str(p), info.st_mtime_ns, info.st_size)


def fingerprint(name):
    p = datasets[name]
    info = p.stat()
    record = {'path': str(p), 'mtime_ns': info.st_mtime_ns, 'size': info.st_size}
    meta = p.with_suffix('.json')
    if meta.exists():
        record['provenance'] = cached_provenance(str(meta), meta.stat().st_mtime_ns)
    return record


# ---------------------------------------------------------------------------
# Running analyses across networks
# ---------------------------------------------------------------------------
def run_items(name, label, items, parameters, compute, *, advanced=False):
    """Run ``compute(item)`` for every item (a network or network pair) with disk and memory caching.

    ``items`` maps a display label to the dataset names it uses. Returns ``{label: result}``.
    """
    skipped = {}
    if advanced:
        for item, used in list(items.items()):
            if any(edges_of(n) > ADVANCED_EDGE_LIMIT for n in used):
                skipped[item] = 'more than 500,000 edges; use a huMAP2 threshold version'
        items = {k: v for k, v in items.items() if k not in skipped}
    store = st.session_state.setdefault('topo_results', {})
    left, right = st.columns([1, 4], vertical_alignment='center')
    refresh = left.button('Recalculate', key=f'refresh_{name}', icon=':material/refresh:',
                          help='Ignore caches and recompute for every network.')
    results, entries, pending = {}, {}, []
    for item, used in items.items():
        metadata = dict(base_context, analysis=name, parameters=parameters, item=item,
                        networks={n: fingerprint(n) for n in used})
        signature = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()
        key = f'{name}|{item}'
        previous = store.get(key)
        if previous and previous['signature'] == signature and not refresh:
            results[item], entries[item] = previous['result'], previous
            continue
        if not (auto_run or refresh):
            pending.append(item)
            continue
        try:
            with st.spinner(f'{label} · {item}…'):
                entry = cached_analysis_run(name, metadata, lambda item=item: compute(item), DATA, use_cache=not refresh)
        except Exception as exc:
            skipped[item] = str(exc)
            continue
        store[key] = {'signature': signature, 'result': entry['result'], 'archive': entry['archive'],
                      'file': entry['file']}
        results[item], entries[item] = entry['result'], store[key]
    with right:
        notes = []
        if results:
            notes.append('Computed for ' + ', '.join(results))
        if skipped:
            notes.append('Not available for ' + '; '.join(f'{k} ({v})' for k, v in skipped.items()))
        if notes:
            st.caption(' · '.join(notes))
    if pending:
        st.info('Automatic computation is off. Click Recalculate to run for ' + ', '.join(pending) + '.')
    if entries:
        st.download_button(
            'Download tables, parameters and provenance for all networks (ZIP)',
            lambda: combined_zip(entries), f'{name}_all_networks.zip', 'application/zip',
            key=f'download_{name}', icon=':material/download:',
        )
    return results


def run_networks(name, label, parameters, compute, *, advanced=False):
    return run_items(name, label, {n: [n] for n in active_nets}, parameters, compute, advanced=advanced)


def combined_zip(entries):
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for item, entry in entries.items():
            safe = ''.join(c if c.isalnum() or c in '._-' else '_' for c in item)
            archive.writestr(f'{safe}/{Path(entry["file"]).name}', entry['archive'])
    return output.getvalue()


# ---------------------------------------------------------------------------
# Shared displays
# ---------------------------------------------------------------------------
def table(frame, key, **kwargs):
    st.dataframe(frame, width='stretch', hide_index=True, key=key, **kwargs)


def relabel_heatmap(fig, title, x_title='Metric', y_title='Interactome', note=None):
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    for annotation in fig.layout.annotations:
        if annotation.text and annotation.text.startswith(('Source vs shared', 'Exact fold')) and note:
            annotation.text = note
    columns = len(fig.data[0].x) if fig.data else 0
    if columns > 7:
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(margin_b=150)
    return fig


def effect_heatmap(tests, key, title=None):
    if title is None:
        title = f'{set_name} versus background in each network'
    fig = relabel_heatmap(
        centrality_heatmap(tests), title,
        note=f'Rank-biserial effect · positive = higher in {set_name} · stars: BH-adjusted q across all cells')
    st.plotly_chart(fig, width='stretch', key=key)


def distribution_controls(metrics, key, default_kind='Box'):
    c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment='bottom')
    metric = c1.selectbox('Metric', metrics, format_func=pretty, key=f'{key}_metric')
    kind = c2.segmented_control('Plot', ['Box', 'Violin', 'Bar'], default=default_kind, required=True,
                                key=f'{key}_kind')
    log = c3.toggle('Log scale', value=metric in LOG_METRICS, key=f'{key}_log_{metric}')
    return metric, kind.lower(), log


def show_metric_comparison(results, key, *, degree_matched=False):
    """Effect heatmap plus one distribution chart for every metric-type analysis."""
    pooled = pool_network_comparisons(results)
    if pooled['tests'].is_empty():
        st.info('No comparable values for the selected networks.')
        return pooled
    effect_heatmap(pooled['tests'], f'{key}_heatmap')
    st.caption(f'Each cell tests {set_name} proteins against the background of that network with a two-sided '
               'Mann–Whitney U test. Benjamini–Hochberg runs across every network-by-metric cell shown. '
               + ('Groups are degree-matched samples (floor(log₂ degree) bins), so finite counts can differ. '
                  if degree_matched else '')
               + 'Graph observations are dependent, so treat stars as exploratory.')
    metric, kind, log = distribution_controls(pooled['metric_order'], key)
    fig = cross_network_metric_figure(pooled['metrics'], metric, pooled['tests'], plot_kind=kind, log_scale=log, set_name=set_name)
    st.plotly_chart(fig, width='stretch', key=f'{key}_dist')
    with st.expander('Test statistics and per-protein values'):
        table(pooled['tests'], f'{key}_tests')
        table(pooled['metrics'], f'{key}_values')
    return pooled


def interval_plot(frame, observed, null_mean, low, high, title, y_title, key, stars=None):
    """Observed values against the null mean and 95% interval for every network."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame['network'], y=frame[null_mean], mode='markers', name='Null mean (95% interval)',
        marker=dict(color='#737373', size=11, symbol='line-ew-open', line_width=3),
        error_y=dict(type='data', symmetric=False,
                     array=(frame[high] - frame[null_mean]).to_list(),
                     arrayminus=(frame[null_mean] - frame[low]).to_list(), color='#9CA3AF'),
    ))
    fig.add_trace(go.Scatter(
        x=frame['network'], y=frame[observed], mode='markers+text', name=f'{set_name} protein set',
        marker=dict(color=GROUP_COLORS.get(set_name, GROUP_COLORS['Moonlighting']), size=13),
        text=frame[stars].to_list() if stars else None, textposition='middle right',
    ))
    fig.update_layout(template='plotly_white', title=title, yaxis_title=y_title, xaxis_title='Interactome',
                      height=420, legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig, width='stretch', key=key)


def null_interval(null, column):
    values = null[column].drop_nulls().to_numpy().astype(float) if null is not None and len(null) else np.array([])
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    return float(values.mean()), float(np.quantile(values, .025)), float(np.quantile(values, .975))


def go_paths():
    local_gaf = DATA / 'GO' / 'HUMAN-uniprot.gaf.gz'
    local_obo = DATA / 'GO' / 'go-basic.obo'
    return (local_gaf if local_gaf.exists() else DEFAULT_GAF,
            local_obo if local_obo.exists() else DEFAULT_OBO)


def input_signatures(paths):
    return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths if p.exists()}


def go_annotation_signatures():
    raw = go_paths()
    cache = DATA / 'annotations' / 'human_go_propagated.parquet'
    return input_signatures(raw if all(p.exists() for p in raw) else [cache, cache.with_suffix('.json')])


def annotation_file():
    gaf, obo = go_paths()
    cached = DATA / 'annotations' / 'human_go_propagated.parquet'
    if not (gaf.exists() and obo.exists()) and cached.exists() and cached.with_suffix('.json').exists():
        return cached
    return prepare_go_annotations(DATA / 'annotations', gaf, obo)


def cytoscape_metrics_panel():
    """Explicit full-network app execution/import with source validation and saved results."""
    st.info('Send the complete selected networks to Cytoscape, run NetworkAnalyzer here or calculate metrics '
            'with an installed app in Cytoscape Desktop, then read its numeric node columns. '
            f'{set_name} is compared with all other proteins in each source network.')
    with st.expander('Cytoscape metric definitions and interpretation'):
        st.markdown(
            '**NetworkAnalyzer** provides degree, clustering, shortest-path centralities, stress and neighborhood metrics. '
            '**cytoHubba** provides hub scores such as MCC (maximal-clique centrality), MNC (largest connected neighbor '
            'component), DMNC (density of that component), and bottleneck scores. These measure different structural roles. '
            'Other apps can contribute numeric node columns; consult the app’s definition before interpreting their direction. '
            'Use scores rather than rank columns when higher should mean more central.'
        )
        st.markdown(
            'The imported node set and edge pairs must match the complete source network. '
            'Missing or non-finite metric values stay unavailable. Positive rank-biserial effects mean higher values in '
            f'{set_name}. Two-sided Mann–Whitney tests use BH across every imported network-by-metric comparison in the run. '
            'Network observations are dependent, so interpret stars as exploratory.'
        )
    c1, c2 = st.columns([2, 1])
    host = c1.text_input('Cytoscape host', st.session_state.get('cy_host', '127.0.0.1'), key='topo_metrics_host')
    port = c2.number_input('CyREST port', 1, 65535, int(st.session_state.get('cy_port', DEFAULT_CYREST_PORT)),
                           key='topo_metrics_port')
    base_url = build_base_url(host, int(port))
    status = check_cytoscape_connection(base_url=base_url)
    if status['connected']:
        st.success('Connected to Cytoscape Desktop.')
    else:
        st.warning(status.get('remediation') or f'Cytoscape is not reachable at {base_url}.')
    if st.button('Refresh connection', key='topo_metrics_connect'):
        st.rerun()
    context = dict(base_context, base_url=base_url, networks={n: fingerprint(n) for n in active_nets})
    signature = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
    state = st.session_state.setdefault('topo_cytoscape_metrics', {})
    if state and state.get('signature') != signature:
        state.clear()
        st.info('Inputs changed. Send the current networks again to update Cytoscape results.')
    eligible = [n for n in active_nets if edges_of(n) <= ADVANCED_EDGE_LIMIT]
    skipped = [n for n in active_nets if n not in eligible]
    if skipped:
        st.caption('Over the 500,000-edge limit: ' + ', '.join(skipped) + '. Select a threshold version.')
    if st.button('Send full networks to Cytoscape', key='topo_metrics_send', type='primary',
                 disabled=not status['connected'] or not eligible):
        state.clear()
        state.update(signature=signature, runs={}, errors={})
        for net in eligible:
            try:
                with st.spinner(f'{net}: sending the full network…'):
                    reference = build_topology_network_tables(datasets[net], cohort)
                    sent = send_to_cytoscape(reference['nodes'], reference['edges'],
                                             title=f'{net} · {set_name} topology', collection='Topological metrics',
                                             layout='grid', apply_style=False, base_url=base_url)
                    state['runs'][net] = {'suid': sent['network_suid'], 'reference': reference}
            except Exception as exc:
                state['errors'][net] = str(exc)
    for net, message in state.get('errors', {}).items():
        st.error(f'{net}: {message}')
    runs = state.get('runs', {})
    if not runs:
        return
    table(pl.DataFrame([{'Network': n, 'Cytoscape SUID': r['suid'],
                         'Nodes': r['reference']['nodes'].height, 'Edges': r['reference']['edges'].height}
                        for n, r in runs.items()]), 'topo_metrics_networks')
    source = st.selectbox('Metric source', ['NetworkAnalyzer', 'cytoHubba', 'Other installed app'], key='topo_metrics_source')
    app_name = (st.text_input('App name for provenance', key='topo_metrics_app_name').strip()
                if source == 'Other installed app' else source)
    if source != 'NetworkAnalyzer':
        st.caption('Calculate metrics on each listed network in Cytoscape Desktop, keep its node and edge set unchanged, '
                   'and save the scores to numeric node-table columns. Then read the columns here.')
    action = 'Run NetworkAnalyzer and read metrics' if source == 'NetworkAnalyzer' else 'Read app metric columns'
    if st.button(action, key='topo_metrics_read', disabled=not status['connected'] or not app_name):
        state.update(raw={}, read_errors={}, source=source, app_name=app_name)
        state.pop('comparison', None)
        state.pop('entries', None)
        for net, run in runs.items():
            try:
                with st.spinner(f'{net}: reading {app_name}…'):
                    if source == 'NetworkAnalyzer':
                        imported = run_network_analyzer(network=run['suid'], cohort=cohort, base_url=base_url)
                        nodes, edges = imported['node_table'], imported['edge_table']
                        candidates = imported['available_metrics']
                    else:
                        if source == 'cytoHubba':
                            imported = extract_cytohubba_scores(network=run['suid'], cohort=cohort, base_url=base_url)
                            nodes = imported['node_table']
                            candidates = imported['cytohubba_columns']
                        else:
                            nodes = fetch_cytoscape_table_polars(network=run['suid'], table='node', base_url=base_url)
                            candidates = [c for c, dtype in nodes.schema.items() if dtype.is_numeric()
                                          and c not in {'SUID', 'id', 'name', 'protein', 'selected', 'is_moonlighting'}]
                        edges = fetch_cytoscape_table_polars(network=run['suid'], table='edge', base_url=base_url)
                    if not candidates:
                        raise ValueError('No metric columns found. Calculate scores in the app and store them in the node table first.')
                    state['raw'][net] = {'nodes': nodes, 'edges': edges, 'candidates': candidates,
                                         'command': imported.get('command') if source == 'NetworkAnalyzer' else None}
            except Exception as exc:
                state['read_errors'][net] = str(exc)
    if state.get('source') != source or state.get('app_name') != app_name:
        st.caption('Read the columns for this metric source to continue.')
        return
    for net, message in state.get('read_errors', {}).items():
        st.error(f'{net}: {message}')
    raw = state.get('raw', {})
    available = sorted({m for r in raw.values() for m in r['candidates']})
    if not available:
        return
    columns = st.multiselect('Metric columns to compare', available, default=available,
                             key=f'topo_metrics_columns_{source}_{hashlib.sha256("|".join(available).encode()).hexdigest()[:12]}')
    comparison_signature = (source, app_name, tuple(columns))
    if st.button('Compare imported metrics', key='topo_metrics_compare', type='primary', disabled=not columns):
        results, errors = {}, {}
        for net, imported in raw.items():
            metrics = [m for m in columns if m in imported['candidates']]
            if not metrics:
                continue
            try:
                results[net] = cytoscape_topology_comparison(imported['nodes'], imported['edges'],
                                                           runs[net]['reference'], cohort, metrics, set_name)
            except Exception as exc:
                errors[net] = str(exc)
        pooled = pool_network_comparisons(results)
        entries = {}
        import_hashes = {n: hashlib.sha256(r['nodes'].write_json().encode()).hexdigest() for n, r in raw.items()}
        for net, result in results.items():
            result['tests'] = pooled['tests'].filter(pl.col('network') == net)
            metadata = dict(context, analysis='cytoscape_topology', item=net, app=app_name,
                            cytoscape_suid=runs[net]['suid'], parameters={'metrics': columns},
                            import_hashes=import_hashes, command=raw[net]['command'],
                            cytoscape_version=status.get('version_info'), installed_apps=status.get('installed_apps'),
                            correction=pooled['correction'])
            try:
                entry = cached_analysis_run('cytoscape_topology', metadata, lambda result=result: result, DATA, use_cache=False)
                entries[net] = entry
            except Exception as exc:
                errors[net] = f'Results computed but saving failed: {exc}'
        state.update(comparison=results, comparison_signature=comparison_signature, entries=entries, comparison_errors=errors)
    if state.get('comparison_signature') == comparison_signature:
        for net, message in state.get('comparison_errors', {}).items():
            st.error(f'{net}: {message}')
        if state.get('comparison'):
            st.markdown(f'##### {app_name}: {set_name} versus network background')
            show_metric_comparison(state['comparison'], 'cytoscape_metrics')
            if state.get('entries'):
                st.download_button('Download app metrics, tests and provenance (ZIP)',
                                   lambda: combined_zip(state['entries']), 'cytoscape_topology.zip', 'application/zip',
                                   key='topo_metrics_download')
    elif state.get('comparison'):
        st.info('Metric selection changed. Compare imported metrics again to update the tests and BH family.')


def chooser(label, options, key):
    return st.segmented_control(label, options, default=options[0], required=True, key=key)


METRIC_GUIDES = {
    'Overview': (
        '**Coverage** is the fraction of selected accessions found in a network. Graph nodes and accession counts can differ for nodes with several mapped labels. Absence does not establish absence of biological function.',
        '**Degree** counts partners; **strength** sums edge weights. **PageRank** and **eigenvector centrality** reward connections to other central proteins. **Core number** measures membership in a densely connected core. Higher scores describe connectivity, not necessarily biological importance.',
        '**Clustering** measures connections among neighbors; **betweenness** measures shortest-path traffic; **bridging centrality** combines betweenness with a local bridging factor.',
        '**Rank-biserial effect** ranges from −1 to +1: positive means higher values in the selected protein set than background, negative means lower. Near zero suggests little distributional separation; this is not a median fold change.',
        '**Hub fold enrichment** is observed divided by expected representation above the network’s ties-inclusive 90th percentile. Above 1 means enrichment. Its hypergeometric/BH tests form a separate family from centrality comparisons.',
        '**Spearman correlation** compares degree ranks across shared proteins; **partner Jaccard** is shared neighbors divided by their union. Percentile shifts describe changes in relative rank. Assay coverage and network density can influence all these comparisons.',
    ),
    'Source sets': (
        '**Membership and coverage:** sources can overlap; Union counts each protein once. Coverage describes representation in a network, not source completeness or quality.',
        '**Shared background** contains network proteins outside the union of compared sources. Each source remains a separate group. Overlap makes comparisons dependent; hiding Union does not change the correction family.',
        '**Centrality effects** compare each source with the shared background: positive means higher centrality. **Hub folds** use the full finite-valued network and a ties-inclusive 90th-percentile cutoff, with their own hypergeometric/BH family.',
        'Read effects together with finite sample counts and adjusted significance. Missing effects and undefined median folds are unavailable, not zero. Network enrichment does not establish which source contains more true moonlighting proteins.',
    ),
    'Advanced centralities': (
        '**Path & reach:** closeness summarizes reciprocal mean shortest-path distance, adjusted by the reachable fraction; harmonic centrality sums reciprocal distances, giving unreachable proteins zero contribution. Betweenness measures shortest-path traffic. Articulation points disconnect their component when removed; onion layers describe successive core peeling.',
        '**Katz and Bonacich scores** combine direct and indirect walks with attenuation. Direction depends on the definition and parameters; negative Bonacich values are neither probabilities nor negative interactions.',
        '**Spectral roles:** subgraph centrality emphasizes closed walks; non-backtracking centrality discounts immediate reversals. Fiedler roughness measures local changes in a component-scale spectral signal. Rank shifts compare rankings under different definitions.',
        '**Electrical & walks:** current-flow betweenness spreads flow over multiple routes; information centrality uses resistance-based accessibility. Heat/quantum return measures retention; wavelet energy describes diffusion-scale variation. Hodge harmonic participation measures incident edges’ involvement in harmonic flow under the chosen triangle-filled complex. Higher is not universally better across these metrics.',
        'Component restrictions and exact-computation size limits can make scores unavailable. Compare finite counts and effect directions within a metric rather than raw scales across metrics.',
    ),
    'Local wiring': (
        '**Triangles and clustering** describe connected neighbors; **open wedges** count unconnected neighbor pairs. **Effective size** measures nonredundant contacts. More neighbor components or higher component entropy can indicate more fragmented partner groups.',
        '**Burt constraint** measures contact redundancy: high values suggest tightly interdependent contacts; low values can support brokerage. Simmelian ties are supported by triangles; square clustering and homology describe longer local structures and holes.',
        '**Graphlet orbits** count roles in small induced subgraphs and depend strongly on degree. PCA is fitted independently in each network, so its axes are not shared and visible separation is descriptive. Partial enumeration requires caution.',
        '**Edge geometry:** embeddedness and overlap describe shared partners; edge betweenness measures shortest-path traffic. Curvature depends on its definition; bridge-like geometry does not establish binding exclusivity. Length-three scores rank structural candidate links, not validated interactions.',
        'Controls are sampled within floor(log₂ degree) bins. Matching reduces degree differences but does not remove all biases. Tests use the available matched samples; plotting samples do not redefine the tests.',
    ),
    'Clustering & communities': (
        '**Communities** are groups found from connectivity. Their size and membership depend on algorithm, resolution, graph density and seed. They are not automatically biological complexes.',
        '**Enrichment** asks whether a community has more protein set members than expected. Read observed/expected counts, fold and BH-adjusted q together. Full-network atlas communities use the selected network as reference population.',
        '**Module roles:** within-module connectivity describes prominence inside a group; participation describes how contacts spread across modules. High participation can suggest brokerage; high internal connectivity can suggest a local hub. Roles depend on the partition.',
        '**Algorithm comparison:** more enriched communities is not automatically better; algorithms may split the same region differently. For overlapping or fuzzy partitions, use the reported membership rules.',
        '**Cytoscape subnetworks** are selected around the protein set. Their enrichment is conditional on this selection and cannot be interpreted as unbiased whole-network enrichment.',
    ),
    'Cohesion & resilience': (
        '**Cohesion:** internal edges count within-set links; the largest component counts mutually reachable set members. Values above random-set expectations suggest cohesion. Low conductance means relatively fewer boundary edges; assortativity measures preference for matching membership status.',
        '**Null intervals** contain the central 95% of simulated reference values; they are not confidence intervals for the observed set. Cohesion uses upper-tail empirical tests. With B permutations, the smallest possible p is 1/(B+1), before BH. Degree matching preserves floor(log₂ degree) bin counts.',
        '**Source separation** compares between-set nearest-member distances with within-set distances in the largest component. Negative means mixing, positive means separation. The null preserves degree bins and overlap; significance tests unusually close placement.',
        '**Diffusion:** restart-walk probability describes proximity to seed proteins. Lift divides by a uniform-teleport reference; neither score is a p-value. Linker tests ask whether a background protein has more set neighbors than expected, with BH within each network.',
        '**Removal impact:** lower surviving largest-component fractions mean more fragmentation. Compare with equally sized degree-matched removals. Global efficiency summarizes reciprocal shortest-path accessibility and is estimated where indicated.',
        '**Removal curves** describe cumulative removal batches; dashed null means are descriptive references. A visible gap alone does not establish significance. Graph deletion does not model every consequence of an experimental knockout.',
    ),
    'Function': (
        '**GO enrichment:** query hits count annotated set proteins; background hits count annotated proteins in the network population. Fold is observed divided by expected representation; above 1 means enrichment. Large folds with few hits can be unstable.',
        '**GO q-values** adjust over all eligible tested terms within each network, including terms hidden from the top list. Color is −log₁₀(q); darker means stronger adjusted evidence. Text shows fold and stars. GO terms overlap and annotation coverage is incomplete.',
        '**GO neighborhood geometry:** coverage measures available partner annotations; Jaccard distance measures dissimilarity. Dirichlet energy and signal variance describe local changes in annotation signals. Higher variation suggests heterogeneous annotated surroundings, not proven multiple functions.',
        '**Metadata Spearman ρ** ranges from −1 to +1: positive means degree rises with an annotation, negative means it falls. Missing metadata changes the sample. Publication counts can reflect study attention; correlation does not establish causation.',
        '**Partner function partitioning** compares partner annotations with reported function profiles. Missing/shared annotations prevent clean partitions and do not establish mutually exclusive binding.',
    ),
    'Layers & scores': (
        '**Multiplex participation** describes how contacts spread across layers. Supra-graph PageRank combines within-layer links with links between shared accessions. The interlayer weight controls coupling, so compare results at the same setting.',
        '**Score filtration** retains edges with native score ≥ cutoff, then counts incident set members and within-set edges. A protein disappears when its last retained edge is removed; isolated proteins are not added.',
        'huMAP2 probability, BioPlex pInt and HuRI author scores have distinct, uncalibrated scales. They are not p-values. Dashed 0.05/0.95 references are standard huMAP2 cutoffs. Retention describes the coverage/stringency tradeoff, not an optimal cutoff.',
        'Filtration curves are descriptive and receive no significance stars without a separate valid test. Multiplex centrality effects and adjusted tests use the stated background.',
    ),
}


def metric_guide(section):
    """Show static explanations without triggering any analysis."""
    with st.expander('Metrics and how to interpret these results', expanded=False):
        st.caption(f'Selected protein set: {set_name}. Source comparisons use their individually named sets.')
        for explanation in METRIC_GUIDES[section]:
            st.markdown(explanation)


with st.expander('How to read effects, significance and missing results', expanded=False):
    st.info('Read effect direction and magnitude together with sample size and adjusted significance. '
            'Topology describes network associations; it does not validate moonlighting.')
    st.markdown(
        'Stars use BH-adjusted q-values: `*` q < 0.05; `**` q < 0.01; '
        '`***` q < 0.001; `****` q < 0.0001. `ns` means a valid test did not pass q < 0.05; '
        '`n/a` or an empty cell means the test or value is unavailable. Neither establishes equality. '
        'Captions and test tables identify comparisons and correction families. '
        'Descriptive curves, PCA coordinates and confidence scores receive no invented significance stars.'
    )
    st.markdown(
        'Background proteins are unlabelled, not validated negatives. Source comparisons exclude the union '
        'from their shared background; hub tests use the full finite-valued network. '
        'Differences between interactomes can reflect assay design, coverage and filtering. '
        'Check finite sample counts and full test tables when values are sparse or unavailable.'
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
TABS = ['Overview', 'Source sets', 'Advanced centralities', 'Local wiring',
        'Clustering & communities', 'Cohesion & resilience', 'Function', 'Layers & scores']
tab_containers = dict(zip(TABS, st.tabs(TABS, key='topo_tab', on_change='rerun')))


def tab_open(name):
    return tab_containers[name].open


# ---- 1. Overview ----------------------------------------------------------
if tab_open('Overview'):
    with tab_containers['Overview']:
        st.subheader('Overview statistics')
        metric_guide('Overview')
        st.write(f'Coverage of {set_name} and core centralities (degree, PageRank, core number, clustering, betweenness, '
                 'eigenvector, bridging) for every selected interactome.')
        eligible = [n for n in active_nets if edges_of(n) <= ADVANCED_EDGE_LIMIT]
        too_large = [n for n in active_nets if n not in eligible]
        if too_large:
            st.caption(f"Not shown: {', '.join(too_large)} (more than 500,000 edges). Use huMAP2_0.05 or huMAP2_0.95.")

        def compute_overview(_):
            graphs = {name: graph_for(name) for name in eligible}
            multi = multi_network_centrality_comparison(graphs, cohort, extra=True, seed=int(seed), landmarks=None, use_cache=True, set_name=set_name)
            pairs, proteins = [], []
            for first, second in combinations(eligible, 2):
                pair = cross_network_analysis(graphs[first], graphs[second], cohort)
                pairs.append({
                    'Network A': first, 'Network B': second,
                    f'Shared {set_name} proteins': pair['summary']['shared_cohort_nodes'],
                    'Spearman degree rho': pair['summary']['spearman_degree_rho'],
                    'Spearman p (unadjusted)': pair['summary']['spearman_p'],
                    'Median partner Jaccard': pair['table']['partner_jaccard'].median(),
                    'Median degree percentile shift (B minus A)': pair['table']['percentile_shift_B_minus_A'].median(),
                })
                proteins.append(pair['table'].with_columns(pl.lit(first).alias('Network A'), pl.lit(second).alias('Network B')))
            presence = pl.DataFrame({'protein': sorted(cohort)}, schema={'protein': pl.String}).with_columns(
                pl.col('protein').is_in(sorted({label for node in graphs[name]['nodes'] for label in str(node).split('|')})).alias(name)
                for name in eligible)
            return {**{k: multi[k] for k in ('networks', 'metrics', 'profiles', 'hub_enrichment')},
                    'pairs': pl.DataFrame(pairs) if pairs else pl.DataFrame(),
                    'pair_proteins': pl.concat(proteins) if proteins else pl.DataFrame(),
                    'presence': presence}

        results = run_items('overview', 'Overview statistics', {'All networks': eligible} if eligible and cohort else {},
                            {'networks': eligible, 'extra': True, 'report_format': 4}, compute_overview)
        result = results.get('All networks')
        if result:
            stats = protein_set_network_summary(result['networks'], set_name)
            st.markdown(f'##### 1 · Networks and {set_name} coverage')
            c1, c2 = st.columns([3, 2])
            with c1:
                table(stats.drop('Cached', strict=False), 'overview_stats',
                      column_config={'Density': st.column_config.NumberColumn(format='%.2e'),
                                     'Mean degree': st.column_config.NumberColumn(format='%.2f'),
                                     f'{set_name} coverage (%)': st.column_config.NumberColumn(format='%.1f')})
            with c2:
                fig = px.bar(stats, x='Network', y=f'{set_name} coverage (%)', text=f'{set_name} proteins present',
                             color='Network', title=f'Share of {set_name} present in each network')
                fig.update_layout(template='plotly_white', showlegend=False, height=330, margin=dict(t=50, b=30))
                st.plotly_chart(fig, width='stretch', key='overview_coverage')
            presence = result['presence']
            if presence.height and len(presence.columns) > 1:
                counts = presence.with_columns(pl.sum_horizontal(pl.exclude('protein').cast(pl.Int64)).alias('networks')) \
                    .group_by('networks').len().sort('networks')
                st.caption(f'{set_name} proteins found in 0, 1, … of the compared networks: '
                           + ' · '.join(f"{row['networks']}: {row['len']:,}" for row in counts.iter_rows(named=True))
                           + '. Absence from a network does not establish absence of an interaction or function.')

            st.markdown(f'##### 2 · Are {set_name} proteins more central?')
            effect_heatmap(result['profiles'], 'overview_effects', f'Centrality: {set_name} versus background')
            st.caption('Orange: higher than the network background; blue: lower. Rank-biserial runs from −1 to +1 so '
                       'measures with different units are comparable. BH across all networks and measures.')

            st.markdown('##### 3 · Are they over-represented among hubs?')
            hub_fig = relabel_heatmap(hub_enrichment_heatmap(result['hub_enrichment']), f'Enrichment of {set_name} among top-10% proteins',
                                      note='Exact fold enrichment · color saturates at ≥2× · stars: BH-adjusted q across all cells')
            st.plotly_chart(hub_fig, width='stretch', key='overview_hubs')
            st.caption('Hubs are proteins at or above the 90th percentile of each measure in each network (ties included). '
                       '1× is the network baseline. One-sided hypergeometric tests form a separate BH family; they do not control for degree.')

            st.markdown('##### 4 · Inspect one measure')
            metrics = [c for c in result['metrics'].columns if c not in {'network', 'protein', 'is_moonlighting', 'group'}]
            metric, kind, log = distribution_controls(metrics, 'overview')
            st.plotly_chart(cross_network_metric_figure(result['metrics'], metric, result['profiles'], plot_kind=kind, log_scale=log, set_name=set_name),
                            width='stretch', key='overview_dist')
            st.caption(SIGNIFICANCE_LEGEND)

            if result['pairs'].height:
                st.markdown('##### 5 · Do the same proteins keep their role across networks?')
                table(result['pairs'], 'overview_pairs')
                labels = [f'{a} vs {b}' for a, b in combinations(eligible, 2)]
                chosen = st.selectbox('Protein-level detail', range(len(labels)), format_func=labels.__getitem__, key='overview_pair')
                first, second = list(combinations(eligible, 2))[chosen]
                proteins = result['pair_proteins'].filter((pl.col('Network A') == first) & (pl.col('Network B') == second))
                if proteins.height:
                    fig = px.scatter(proteins, x='degree_percentile_A', y='degree_percentile_B', color='partner_jaccard',
                                     hover_data=['protein', 'degree_A', 'degree_B'], color_continuous_scale='Viridis',
                                     labels={'degree_percentile_A': first + ' degree percentile',
                                             'degree_percentile_B': second + ' degree percentile',
                                             'partner_jaccard': 'Partner Jaccard'})
                    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='#999'))
                    fig.update_layout(template='plotly_white', height=460)
                    st.plotly_chart(fig, width='stretch', key='overview_pair_scatter')
                st.caption('Percentiles use all nodes of each network. Assay differences affect which partners are captured.')
            with st.expander(f'Per-protein values, test tables and {set_name} presence by network'):
                table(result['profiles'], 'overview_profiles')
                table(result['hub_enrichment'], 'overview_hub_table')
                table(result['presence'], 'overview_presence')
                table(result['metrics'], 'overview_metrics')

# ---- 2. Source sets -------------------------------------------------------
if tab_open('Source sets'):
    with tab_containers['Source sets']:
        st.subheader('Compare moonlighting source sets in every network')
        metric_guide('Source sets')
        st.write('Each source is shown separately against the shared background of each network (all proteins outside the '
                 'union of the selected sources). Proteins listed by several sources appear in each of them.')
        c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment='bottom')
        compare_sources = c1.multiselect('Source sets', SOURCES, default=['moonprot', 'multitask', 'moonlitdb'],
                                         format_func=lambda s: GROUP_LABELS.get(s, s), key='compare_sources')
        include_union = c2.checkbox('Show Union', value=True, key='compare_include_union',
                                    help='Hiding Union does not change the correction family.')
        compare_extra = c3.checkbox('Advanced measures', value=True, key='compare_extra',
                                    help='Adds core number, clustering, exact betweenness, eigenvector and bridging (≤500,000 edges).')
        if not compare_sources:
            st.info('Choose at least one source set.')
        else:
            results = run_networks(
                'source_comparison', 'Source sets',
                {'analysis_revision': 'centrality_profiles_hubs_v1', 'sources': compare_sources, 'include_union': True,
                 'advanced': compare_extra, 'landmarks': None},
                lambda net: compare_source_sets(graph_for(net), moon, compare_sources, reviewed_only=reviewed_only,
                                                min_publications=min_pubs, extra=compare_extra, seed=int(seed),
                                                landmarks=None, include_union=True),
                advanced=compare_extra,
            )
            if results:
                def labelled(key):
                    return {net: r[key].with_columns(
                        pl.concat_str(pl.lit(net + ' · '), pl.col('group').replace(GROUP_LABELS)).alias('group'),
                        pl.col('group').alias('source'))
                        for net, r in results.items()}

                profiles = stack_network_tables(labelled('profiles'), p_column='p_value')
                hubs = stack_network_tables(labelled('hub_enrichment'), p_column='p_value')
                if not include_union:
                    profiles = profiles.filter(pl.col('source') != 'Union')
                    hubs = hubs.filter(pl.col('source') != 'Union')
                coverage = stack_network_tables({n: r['coverage'] for n, r in results.items()})
                if not include_union:
                    coverage = coverage.filter(pl.col('source') != 'Union')
                fig = px.bar(coverage.with_columns(pl.col('source').replace(GROUP_LABELS)), x='network', y='network_ids',
                             color='source', barmode='group', title='Source proteins present in each network',
                             labels={'network_ids': 'Proteins in network', 'network': 'Interactome', 'source': 'Source'},
                             color_discrete_map={GROUP_LABELS[k]: v for k, v in GROUP_COLORS.items()})
                fig.update_layout(template='plotly_white', height=360)
                st.plotly_chart(fig, width='stretch', key='source_coverage')

                fig = relabel_heatmap(centrality_heatmap(profiles), 'Centrality: each source versus background',
                                      y_title='Network · source set',
                                      note='Rank-biserial effect · positive = higher than background · stars: BH-adjusted q across all networks, sources and measures')
                st.plotly_chart(fig, width='stretch', key='source_effects')
                fig = relabel_heatmap(hub_enrichment_heatmap(hubs), 'Enrichment among top-10% proteins',
                                      y_title='Network · source set',
                                      note='Exact fold · color saturates at ≥2× · stars: BH across all networks, sources and measures (separate family)')
                st.plotly_chart(fig, width='stretch', key='source_hubs')
                st.caption('Benjamini–Hochberg is recomputed across every network, source (including Union) and measure. '
                           'Overlapping sources are not independent; no source-versus-source tests are performed.')

                frames = [r['metrics'].with_columns(pl.lit(net).alias('network')) for net, r in results.items()]
                values = pl.concat(frames, how='diagonal_relaxed')
                if not include_union:
                    values = values.filter(pl.col('group') != 'Union')
                metric_names = profiles['metric'].unique(maintain_order=True).to_list()
                metric, kind, log = distribution_controls(metric_names, 'source')
                st.plotly_chart(cross_network_metric_figure(values, metric, plot_kind=kind, log_scale=log),
                                width='stretch', key='source_dist')
                with st.expander('Coverage, overlap, test tables and protein set metadata'):
                    table(coverage, 'source_coverage_table')
                    table(stack_network_tables({n: r['overlap'] for n, r in results.items()}), 'source_overlap')
                    table(profiles, 'source_profiles')
                    table(hubs, 'source_hub_table')
                    columns = ['input_accession', 'canonical_id', 'Gene Names', 'Reviewed'] + SOURCES + \
                              ['moonlitdb_number_of_publications', 'moonlitdb_number_of_moonlighting_functions']
                    table(selected.select([c for c in columns if c in selected.columns]), 'selected_metadata')

# ---- 3. Advanced centralities --------------------------------------------
if tab_open('Advanced centralities'):
    with tab_containers['Advanced centralities']:
        st.subheader('Advanced centralities')
        metric_guide('Advanced centralities')
        method = chooser('Analysis', ['Path & reach', 'Spectral', 'Electrical & walks'], 'adv_method')
        if method == 'Path & reach':
            st.caption('Closeness, harmonic, onion, Katz and negative Bonacich scores, plus degree-adjusted landmark betweenness and articulation points.')
            results = run_networks('path_centralities', 'Path centralities', {},
                                   lambda net: path_centralities(graph_for(net), cohort, seed=int(seed), set_name=set_name), advanced=True)
            show_metric_comparison(results, 'path')
        elif method == 'Spectral':
            st.caption('Log subgraph centrality, non-backtracking centrality and rank shifts, and local roughness of the largest component’s Fiedler signal.')
            results = run_networks('spectral_roles', 'Spectral roles', {},
                                   lambda net: spectral_roles(graph_for(net), cohort, seed=int(seed), set_name=set_name), advanced=True)
            show_metric_comparison(results, 'spectral')
        else:
            st.caption('Current-flow betweenness, information centrality, resistance curvature, heat and quantum return, heat-wavelet energy '
                       'and Hodge participation. Exact on components up to 350 proteins; networks whose target component is larger are reported as unavailable.')
            protein = st.text_input('Protein accession in the target component (blank = largest component)',
                                    key='moon_electrical_protein').strip().upper() or None
            results = run_networks('electrical_and_walk_roles', 'Electrical & walk roles', {'protein': protein},
                                   lambda net: electrical_and_walk_roles(graph_for(net), cohort, protein, set_name=set_name), advanced=True)
            show_metric_comparison(results, 'electrical')
            if results:
                with st.expander('Edge-level results'):
                    table(stack_network_tables({n: r.get('edges') for n, r in results.items()}), 'electrical_edges')

# ---- 4. Local wiring -------------------------------------------------------
if tab_open('Local wiring'):
    with tab_containers['Local wiring']:
        st.subheader('Local wiring around each protein')
        metric_guide('Local wiring')
        st.caption(f'{set_name} proteins are compared with background proteins sampled in the same floor(log₂ degree) bins, '
                   'so differences are not simply explained by degree. These patterns suggest structural roles; they do not establish binding exclusivity.')
        method = chooser('Analysis', ['Ego neighbourhood', 'Ego geometry', 'Graphlets', 'Edge geometry'], 'wiring_method')
        if method == 'Ego neighbourhood':
            count = setting(st.number_input, f'{set_name} proteins sampled per network', 10, 500, 100, 10, key='wiring_count')
            st.caption('Exact ego clustering, triangles, open wedges, effective size, component count and component entropy.')
            results = run_networks('local_wiring', 'Ego neighbourhood', {'max_nodes': int(count)},
                                   lambda net: wiring_analysis(graph_for(net), cohort, max_nodes=int(count), seed=int(seed), set_name=set_name), advanced=True)
            show_metric_comparison(results, 'wiring', degree_matched=True)
        elif method == 'Ego geometry':
            c1, c2, c3 = st.columns(3, vertical_alignment='bottom')
            limit = c1.number_input(f'{set_name} proteins sampled per network', 5, 100, 25, 5, key='ego_geometry_limit')
            walks = c2.checkbox('Add split-ego diffusion, hyperbolicity and portraits', key='ego_walk_geometry')
            cutoff = c3.number_input('huMAP2 strong-edge probability', 0., 1., .5, .05, key='ego_wedge_cutoff',
                                     help='Only used for huMAP2 networks, which carry probabilities.')
            st.caption('Burt constraint, square and Barrat clustering, Simmelian ties and brokerage, Forman curvature, walk entropy and vertex-link homology.')
            results = run_networks(
                'ego_geometry', 'Ego geometry', {'max_nodes': int(limit), 'walk_geometry': walks, 'score_cutoff': cutoff},
                lambda net: ego_geometry(graph_for(net), cohort, int(limit), int(seed),
                                         cutoff if net.split('_')[0] == 'huMAP2' else None, walks, set_name=set_name), advanced=True)
            show_metric_comparison(results, 'ego', degree_matched=True)
        elif method == 'Graphlets':
            c1, c2 = st.columns(2)
            order = c1.selectbox('Largest graphlet order', [4, 5], key='graphlet_order')
            limit = c2.number_input(f'{set_name} proteins sampled per network', 5, 50, 20, 5, key='graphlet_limit')
            st.caption('Exact induced orbit profiles up to the chosen order. Dense roots can exceed the enumeration budget and return partial profiles.')
            results = run_networks('graphlet_profiles', 'Graphlet profiles', {'max_nodes': int(limit), 'order': int(order)},
                                   lambda net: graphlet_profiles(graph_for(net), cohort, int(limit), int(seed), int(order), set_name=set_name), advanced=True)
            show_metric_comparison(results, 'graphlet', degree_matched=True)
            projections = stack_network_tables({n: r.get('projection') for n, r in results.items()})
            if projections.height:
                fig = px.scatter(projections, x='PC1', y='PC2', color='group', facet_col='network', facet_col_wrap=3,
                                 hover_name='protein', color_discrete_map=GROUP_COLORS,
                                 title='PCA of graphlet profiles (computed separately in each network)')
                fig.update_xaxes(matches=None)
                fig.update_yaxes(matches=None)
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, width='stretch', key='graphlet_pca')
        else:
            ollivier = st.checkbox('Include sampled Ollivier–Ricci curvature', key='edge_ollivier')
            st.caption('Degree-matched anchors summarise edge betweenness, neighbourhood overlap and embeddedness. '
                       'Length-three path scores rank unobserved pairs as structural candidates.')
            results = run_networks('edge_geometry', 'Edge geometry', {'max_nodes': 25, 'ollivier': ollivier},
                                   lambda net: edge_geometry(graph_for(net), cohort, int(seed), ollivier=ollivier, set_name=set_name), advanced=True)
            show_metric_comparison(results, 'edge', degree_matched=True)
            if results:
                with st.expander('Length-three candidate links'):
                    table(stack_network_tables({n: r.get('l3_candidates') for n, r in results.items()}), 'l3_candidates')


# ---- 5. Clustering & communities -----------------------------------------
def cytoscape_panel():
    st.write(f'Send the {set_name} subnetwork of each selected interactome to Cytoscape Desktop, run a `clusterMaker2` '
             f'algorithm on each, and compare which clusters concentrate {set_name} proteins.')
    c1, c2, c3 = st.columns([2, 1, 3], vertical_alignment='bottom')
    host = c1.text_input('Cytoscape host', st.session_state.get('cy_host', '127.0.0.1'), key='topo_cy_host')
    port = c2.number_input('CyREST port', 1, 65535, int(st.session_state.get('cy_port', DEFAULT_CYREST_PORT)), 1, key='topo_cy_port')
    base_url = build_base_url(host, int(port))
    status = check_cytoscape_connection(base_url=base_url)
    with c3:
        if status['connected']:
            version = status.get('version_info', {}).get('cytoscapeVersion', '')
            apps = {a.get('appName', '').lower() for a in status.get('installed_apps', []) if a.get('status') == 'Installed'}
            st.success(f'Connected to Cytoscape {version}' + ('' if not apps or 'clustermaker2' in apps
                                                            else ' · clusterMaker2 is not listed as installed'))
        else:
            st.warning(status.get('remediation') or f'Cytoscape is not reachable at {base_url}.')
            if st.button('Check again', key='topo_cy_retry'):
                st.rerun()

    c1, c2, c3 = st.columns(3)
    algorithm = c1.selectbox('clusterMaker2 algorithm', list(CLUSTERMAKER_ALGORITHMS),
                             format_func=lambda k: CLUSTERMAKER_ALGORITHMS[k]['label'], key='topo_cm_algo')
    spec = CLUSTERMAKER_ALGORITHMS[algorithm]
    params_text = c2.text_input('Parameters (key=value)', ' '.join(f'{k}={v}' for k, v in spec['default_params'].items()),
                                key=f'topo_cm_params_{algorithm}')
    layout = c3.selectbox('Layout', ['force-directed', 'cose', 'kamada-kawai', 'circular', 'grid'], key='topo_cy_layout')
    c1, c2, c3 = st.columns(3, vertical_alignment='bottom')
    max_nodes = c1.slider('Maximum nodes per subnetwork', 50, 1000, 300, 50, key='topo_cy_nodes')
    max_linkers = c2.slider('Top linker proteins added', 0, 200, 40, 10, key='topo_cy_linkers',
                            help=f'Proteins outside {set_name} with significantly many {set_name} neighbors.')
    push_atlas = c3.checkbox('Also add Atlas community columns', value=True, key='topo_cy_push_atlas',
                             help='Adds the Atlas partition chosen under Atlas communities as atlas_* node columns for side-by-side comparison.')
    params = dict(token.split('=', 1) for token in params_text.split() if '=' in token)
    atlas_algorithm = st.session_state.get('community_algorithm', 'louvain')
    atlas_resolution = float(st.session_state.get('community_resolution', 1.0))

    state = st.session_state.setdefault('topo_cytoscape', {})
    if st.button(f'Send {len(active_nets)} networks to Cytoscape and run {spec["label"]}', type='primary',
                 disabled=not status['connected'] or not cohort, key='topo_cy_run', icon=':material/send:'):
        runs, errors = {}, {}
        for net in active_nets:
            try:
                with st.spinner(f'{net}: building subnetwork and running clusterMaker2…'):
                    sub = build_subnetwork_tables(graph_for(net), moon, cohort, mode='cohort_linkers',
                                                  max_linkers=int(max_linkers), max_nodes=int(max_nodes))
                    sent = send_to_cytoscape(sub['nodes'], sub['edges'], title=f'{net} · {set_name} subnetwork',
                                             collection='Topological analysis', layout=layout, base_url=base_url)
                    note = None
                    if push_atlas:
                        try:
                            communities = community_analysis(graph_for(net), cohort, seed=int(seed),
                                                             resolution=atlas_resolution, algorithm=atlas_algorithm,
                                                             set_name=set_name)
                            members = communities['metrics'].filter(pl.col('protein').is_in(sub['nodes']['protein'].implode()))
                            push_clustering_to_cytoscape(members, atlas_algorithm, network=sent['network_suid'], base_url=base_url)
                        except Exception as exc:
                            note = f'Atlas columns not added: {exc}'
                    cm = run_clustermaker2(algorithm, network=sent['network_suid'], params=params, cohort=cohort, base_url=base_url)
                runs[net] = {'suid': sent['network_suid'], 'nodes': sent['nodes_sent'], 'edges': sent['edges_sent'],
                             'moonlighting': sub['summary']['moonlighting_nodes'], 'column': cm['cluster_column'],
                             'command': cm['command'], 'enrichment': cm['cluster_enrichment'],
                             'bridging': cm['node_cluster_bridging'], 'note': note}
            except Exception as exc:
                errors[net] = str(exc)
        state.clear()
        state.update(runs=runs, errors=errors, base_url=base_url, algorithm=algorithm, cohort=frozenset(cohort))

    runs = state.get('runs')
    if runs and status['connected'] and st.button('Re-read clusters from Cytoscape', key='topo_cy_reread',
                                                   help='Use after editing or re-running clustering in Cytoscape Desktop.'):
        for net, run in runs.items():
            try:
                nodes = fetch_cytoscape_table_polars(network=run['suid'], table='node', base_url=state['base_url'])
                edges = fetch_cytoscape_table_polars(network=run['suid'], table='edge', base_url=state['base_url'])
                column = run['column'] or next((c for c in nodes.columns if 'cluster' in c.lower()), None)
                if column:
                    analysis = compute_cluster_enrichment_and_bridging(nodes, edges, column, cohort=cohort)
                    run.update(column=column, enrichment=analysis['cluster_enrichment'], bridging=analysis['node_cluster_bridging'])
            except Exception as exc:
                state['errors'][net] = f'Re-read failed: {exc}'
    for net, message in state.get('errors', {}).items():
        st.error(f'{net}: {message}')
    if not runs:
        st.caption('For other subnetwork modes, cytoHubba, NetworkAnalyzer and exports, open Network Visualization / Cytoscape.')
        return
    if state.get('cohort') != frozenset(cohort):
        st.info(f'The {set_name} protein set changed since these Cytoscape runs. Run again to update them.')
    enrichment = stack_network_tables({n: r['enrichment'] for n, r in runs.items()}, p_column='p_value')
    if enrichment.height and 'fdr_bh' in enrichment.columns:
        enrichment = enrichment.with_columns(
            pl.col('fdr_bh').map_elements(significance_label, return_dtype=pl.String).alias('significance'))
    bridging = stack_network_tables({n: r['bridging'] for n, r in runs.items()})
    rows = []
    for net, run in runs.items():
        net_enr = enrichment.filter(pl.col('network') == net) if enrichment.height else enrichment
        net_bridge = bridging.filter(pl.col('network') == net) if bridging.height else bridging
        rows.append({
            'Network': net, 'Cytoscape SUID': run['suid'], 'Nodes sent': run['nodes'], 'Edges sent': run['edges'],
            f'{set_name} nodes': run['moonlighting'], 'Cluster column': run['column'] or 'not found',
            'Clusters': net_enr.height,
            'Enriched clusters (q < 0.05)': int((net_enr['fdr_bh'] < .05).sum()) if net_enr.height and 'fdr_bh' in net_enr.columns else 0,
            'Max fold': float(net_enr['fold_enrichment'].max()) if net_enr.height and 'fold_enrichment' in net_enr.columns else None,
            f'{set_name} proteins bridging ≥2 clusters': int(net_bridge.filter(
                pl.col('is_moonlighting') & (pl.col('distinct_neighbor_clusters') >= 2)).height) if net_bridge.height else 0,
        })
        if run.get('note'):
            st.caption(f"{net}: {run['note']}")
    st.markdown(f"##### clusterMaker2 · {CLUSTERMAKER_ALGORITHMS[state['algorithm']]['label']}")
    table(pl.DataFrame(rows), 'topo_cy_summary')
    st.caption(f'Cluster enrichment uses the subnetwork sent to Cytoscape as population, not the full interactome. '
               'BH runs across all clusters from all networks. Each subnetwork is selected around {set_name}, so '
               'enrichment describes concentration within that neighbourhood.')
    if bridging.height:
        fig = px.scatter(bridging, x='clustered_degree', y='clustermaker_participation', color='is_moonlighting',
                         facet_col='network', facet_col_wrap=3,
                         hover_data=['protein', 'label', 'assigned_cluster', 'distinct_neighbor_clusters'],
                         color_discrete_map={True: GROUP_COLORS.get(set_name, GROUP_COLORS['Moonlighting']), False: GROUP_COLORS['Background']},
                         labels={'clustered_degree': 'Subnetwork degree', 'clustermaker_participation': 'Participation across clusters',
                                 'is_moonlighting': set_name},
                         title=f'Proteins linking several clusterMaker2 clusters ({set_name} vs background)')
        fig.update_xaxes(matches=None)
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, width='stretch', key='topo_cy_bridging')
    with st.expander('Cluster enrichment and bridging tables'):
        table(enrichment, 'topo_cy_enrichment')
        table(bridging, 'topo_cy_bridge_table')


if tab_open('Clustering & communities'):
    with tab_containers['Clustering & communities']:
        st.subheader('Clustering and communities')
        metric_guide('Clustering & communities')
        method = chooser('Analysis', ['Atlas communities', 'Module roles', 'Algorithm comparison', 'Cytoscape (clusterMaker2)'],
                         'cluster_method')
        if method == 'Atlas communities':
            c1, c2 = st.columns([3, 1])
            algorithm = setting(st.selectbox, 'Clustering algorithm', list(CLUSTERING_CATALOG.keys()),
                                format_func=lambda a: f"[{CLUSTERING_CATALOG[a]['paradigm'].upper()}] {CLUSTERING_CATALOG[a]['label']}",
                                key='community_algorithm')
            with c2:
                resolution = setting(st.number_input, 'Resolution', min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                     key='community_resolution')
            st.caption(f"{CLUSTERING_CATALOG[algorithm]['description']} Partitions are cached per network in `data/cache/clusterings/`.")
            results = run_networks('communities', 'Communities', {'algorithm': algorithm, 'resolution': float(resolution)},
                                   lambda net: community_analysis(graph_for(net), cohort, seed=int(seed),
                                                                  resolution=float(resolution), algorithm=algorithm,
                                                                  set_name=set_name),
                                   advanced=True)
            if results:
                enrichment = stack_network_tables({n: r['enrichment'] for n, r in results.items()}, p_column='p_value')
                rows = []
                for net, r in results.items():
                    s = r['summary']
                    net_enr = enrichment.filter(pl.col('network') == net)
                    rows.append({
                        'Network': net, 'Communities': s['communities'], 'Modularity': s.get('modularity'),
                        'Overlapping nodes (%)': 100 * float(s.get('overlapping_fraction') or 0),
                        f'Communities enriched in {set_name} (q < 0.05)': int((net_enr['fdr_bh'] < .05).sum()),
                        f'{set_name} proteins in enriched communities': int(net_enr.filter(pl.col('fdr_bh') < .05)['moonlighting_proteins'].sum()),
                    })
                table(pl.DataFrame(rows), 'community_summary',
                      column_config={'Modularity': st.column_config.NumberColumn(format='%.3f'),
                                     'Overlapping nodes (%)': st.column_config.NumberColumn(format='%.1f')})
                st.caption('Community enrichment: one-sided hypergeometric test with each full network as population; '
                           'BH across all communities of all networks.')
                pooled = show_metric_comparison(results, 'community')
                frames = []
                for net, r in results.items():
                    m = r['metrics'].with_columns(pl.lit(net).alias('network'))
                    background = m.filter(pl.col('group') == 'Background')
                    frames += [m.filter(pl.col('group') == set_name),
                               background.sample(n=min(2000, background.height), seed=int(seed))]
                fig = px.scatter(pl.concat(frames, how='diagonal_relaxed'), x='within_module_z', y='participation',
                                 color='group', facet_col='network', facet_col_wrap=3, opacity=.6,
                                 color_discrete_map=GROUP_COLORS, hover_data=['protein', 'community', 'num_communities'],
                                 labels={'within_module_z': 'Within-module degree z', 'participation': 'Participation'},
                                 title=f'Position inside versus across communities ({set_name} vs background)')
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, width='stretch', key='community_cartography')
                st.caption('Up to 2,000 background proteins per network are drawn for display; tests use all proteins.')
                with st.expander('Community enrichment table'):
                    table(enrichment, 'community_enrichment')
                annotation = st.expander('Annotate communities with GO and Reactome', key='moon_cluster_annotation', on_change='rerun')
                if annotation.open:
                    with annotation:
                        target = st.selectbox('Network to annotate', list(results), key='community_annotation_network')
                        r = results[target]
                        p = datasets[target]
                        info = p.stat()
                        clustering = r['summary']
                        members = r['metrics'].with_columns(pl.col('communities_str').cast(pl.String))
                        annotation_panel(
                            f'moon_community_annotation_{target}',
                            (str(p), info.st_mtime_ns, info.st_size, clustering['algorithm'], clustering['resolution'], clustering['seed']),
                            lambda: (atlas_memberships(members), members.select('protein'),
                                     {'network': {'name': target, **_fingerprint(p), 'audit': fingerprint(target).get('provenance')},
                                      'clustering': {k: clustering[k] for k in ('algorithm', 'resolution', 'seed')}}),
                            DATA,
                        )
        elif method == 'Module roles':
            st.caption('Guimerà–Amaral roles from the cached Louvain partition, and Gould–Fernandez brokerage counts for degree-matched samples.')
            results = run_networks('module_brokerage', 'Module roles', {'partition': 'louvain', 'resolution': 1.0},
                                   lambda net: module_brokerage(graph_for(net), cohort, int(seed), set_name=set_name), advanced=True)
            if results:
                counts = stack_network_tables({n: r['role_counts'] for n, r in results.items()})
                counts = counts.with_columns((pl.col('len') / pl.col('len').sum().over('network', 'group')).alias('fraction'))
                fig = px.bar(counts, x='role', y='fraction', color='group', barmode='group', facet_col='network',
                             facet_col_wrap=3, color_discrete_map=GROUP_COLORS,
                             labels={'fraction': 'Share of group', 'role': 'Role'},
                             title='Guimerà–Amaral roles (R1–R4 non-hubs, R5–R7 hubs)')
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, width='stretch', key='module_roles')
                show_metric_comparison(results, 'brokerage', degree_matched=True)
        elif method == 'Algorithm comparison':
            st.caption('Runs several Classic, Overlapping and Fuzzy algorithms on every network and compares community-role effects. '
                       'This can take a while the first time; results are cached.')
            default_suite = ['louvain', 'label_propagation', 'mcl', 'link_communities', 'k_clique_percolation',
                             'fuzzy_c_means', 'fuzzy_rwr_diffusion']
            suite = st.multiselect('Algorithms', list(CLUSTERING_CATALOG), default=[a for a in default_suite if a in CLUSTERING_CATALOG],
                                   format_func=lambda a: f"[{CLUSTERING_CATALOG[a]['paradigm'].upper()}] {a}", key='community_suite_algos')
            resolution = float(st.session_state.get('community_resolution', 1.0))
            if suite:
                results = run_networks('clustering_suite', 'Algorithm comparison', {'algorithms': suite, 'resolution': resolution},
                                       lambda net: {k: v for k, v in compare_clustering_algorithms(
                                           graph_for(net), cohort, algorithms=suite, seed=int(seed), resolution=resolution, set_name=set_name).items()
                                           if k != 'per_algorithm'}, advanced=True)
                if results:
                    summaries = stack_network_tables({n: r['summary_table'] for n, r in results.items()})
                    effects = stack_network_tables({n: r['effects'] for n, r in results.items()}, p_column='p_value')
                    metric = st.selectbox('Community measure', effects['metric'].unique(maintain_order=True).to_list(),
                                          format_func=pretty, key='suite_metric')
                    view = effects.filter(pl.col('metric') == metric)
                    algorithms = view['group'].unique(maintain_order=True).to_list()
                    nets = list(results)
                    lookup = {(r['group'], r['network']): r for r in view.iter_rows(named=True)}
                    z = [[lookup.get((a, n), {}).get('rank_biserial') for n in nets] for a in algorithms]
                    text = [[(f"{lookup[(a, n)]['rank_biserial']:+.2f}<br>{significance_label(lookup[(a, n)]['fdr_bh'])}"
                              if (a, n) in lookup and lookup[(a, n)]['rank_biserial'] is not None else 'NA') for n in nets]
                            for a in algorithms]
                    fig = go.Figure(go.Heatmap(z=z, x=nets, y=algorithms, zmin=-1, zmax=1, zmid=0, colorscale=COLORSCALE,
                                               text=text, texttemplate='%{text}', xgap=3, ygap=3,
                                               colorbar=dict(title='Rank-biserial')))
                    fig.update_layout(template='plotly_white', title=f'{pretty(metric)}: {set_name} versus background',
                                      xaxis_title='Interactome', yaxis_title='Algorithm', yaxis_autorange='reversed',
                                      height=max(360, 120 + 45 * len(algorithms)))
                    st.plotly_chart(fig, width='stretch', key='suite_heatmap')
                    st.caption('BH across every algorithm, measure and network.')
                    table(summaries, 'suite_summary')
                    with st.expander('All algorithm-by-measure tests'):
                        table(effects, 'suite_effects')
        else:
            cytoscape_panel()





# ---- 6. Cohesion & resilience --------------------------------------------
if tab_open('Cohesion & resilience'):
    with tab_containers['Cohesion & resilience']:
        st.subheader('Cohesion and resilience')
        metric_guide('Cohesion & resilience')
        method = chooser('Analysis', ['Protein set cohesion', 'Source separation', 'Diffusion & linkers', 'Removal impact', 'Removal curves'],
                         'cohesion_method')
        if method == 'Protein set cohesion':
            c1, c2 = st.columns(2, vertical_alignment='bottom')
            perms = c1.number_input('Permutations', 10, 1000, 100, 10, key='cohesion_perms')
            matched = c2.checkbox(f'Match the {set_name} degree distribution', value=True, key='cohesion_match')
            results = run_networks('cohesion', 'Cohesion', {'permutations': int(perms), 'degree_matched': matched},
                                   lambda net: cohesion_analysis(graph_for(net), cohort, permutations=int(perms),
                                                                 seed=int(seed), degree_matched=matched))
            if results:
                rows = []
                for net, r in results.items():
                    s = r['summary']
                    for metric, label in (('internal_edges', 'Internal edges'), ('largest_component', 'Largest component')):
                        mean, low, high = null_interval(r['null'], metric)
                        rows.append({'network': net, 'measure': label, 'observed': s[metric], 'null_mean': mean,
                                     'null_low': low, 'null_high': high, 'ratio': s[metric] / mean if mean else None,
                                     'p_value': s[f'{metric}_p_value'], 'cohort_nodes': s['query_nodes'],
                                     'conductance': s['conductance'], 'assortativity': s['cohort_status_assortativity']})
                frame = pl.DataFrame(rows).with_columns(pl.Series('fdr_bh', bh_adjust([r['p_value'] for r in rows])))
                frame = frame.with_columns(pl.col('fdr_bh').map_elements(significance_label, return_dtype=pl.String).alias('significance'))
                c1, c2 = st.columns(2)
                for col, measure in zip((c1, c2), ('Internal edges', 'Largest component')):
                    with col:
                        interval_plot(frame.filter(pl.col('measure') == measure), 'observed', 'null_mean', 'null_low', 'null_high',
                                      f'{measure}: {set_name} versus random sets', measure, f'cohesion_{measure}', 'significance')
                st.caption('Upper-tail empirical p = (extreme + 1)/(permutations + 1); BH across both measures in all networks. '
                           + ('Random sets match floor(log₂ degree) counts.' if matched else 'Random sets are drawn uniformly.'))
                table(frame, 'cohesion_table')
        elif method == 'Source separation':
            c1, c2, c3 = st.columns(3)
            first = c1.selectbox('Source A', SOURCES, format_func=GROUP_LABELS.get, key='separation_source_a')
            second = c2.selectbox('Source B', [s for s in SOURCES if s != first], format_func=GROUP_LABELS.get, key='separation_source_b')
            trials = c3.number_input('Permutations', 0, 199, 49, 1, key='separation_trials')
            a, _ = cohort_ids(moon, [first], reviewed_only=reviewed_only, min_publications=min_pubs)
            b, _ = cohort_ids(moon, [second], reviewed_only=reviewed_only, min_publications=min_pubs)
            st.caption('Nearest-member distances in each largest component. The null shuffles both memberships within degree bins and keeps their overlap. '
                       'Negative separation means the two sources sit closer together than they sit within themselves.')
            results = run_networks('cohort_separation', 'Source separation',
                                   {'source_A': first, 'source_B': second, 'source_A_ids': sorted(a), 'source_B_ids': sorted(b),
                                    'permutations': int(trials)},
                                   lambda net: cohort_separation(graph_for(net), matching_nodes(graph_for(net)['nodes'], a),
                                                                 matching_nodes(graph_for(net)['nodes'], b), int(trials), int(seed)),
                                   advanced=True)
            if results:
                rows = []
                for net, r in results.items():
                    s = r['summary']
                    mean, low, high = null_interval(r['null'], 'separation')
                    rows.append({'network': net, 'separation': s['separation'], 'null_mean': mean, 'null_low': low, 'null_high': high,
                                 'within_A': s['within_A'], 'within_B': s['within_B'], 'between': s['between'],
                                 'A_in_lcc': s['source_A_in_lcc'], 'B_in_lcc': s['source_B_in_lcc'], 'overlap': s['overlap_in_lcc'],
                                 'p_closer': s['empirical_p_closer']})
                frame = pl.DataFrame(rows)
                if frame['p_closer'].drop_nulls().len():
                    frame = frame.with_columns(pl.Series('fdr_bh', bh_adjust(frame['p_closer'].cast(pl.Float64).to_numpy())))
                    frame = frame.with_columns(pl.col('fdr_bh').map_elements(significance_label, return_dtype=pl.String).alias('significance'))
                interval_plot(frame, 'separation', 'null_mean', 'null_low', 'null_high',
                              f'{GROUP_LABELS[first]} – {GROUP_LABELS[second]} separation', 'Separation (shortest-path steps)',
                              'separation_plot', 'significance' if 'significance' in frame.columns else None)
                table(frame, 'separation_table')
        elif method == 'Diffusion & linkers':
            c1, c2, c3 = st.columns(3, vertical_alignment='bottom')
            restart = c1.slider('Restart probability', 0.05, 0.95, 0.3, 0.05, key='walk_restart')
            weighted = c2.checkbox('Use huMAP2 probabilities as weights', key='walk_weighted')
            walk_perms = c3.number_input('Degree-matched permutations (0 = ranking only)', 0, 50, 0, 5, key='walk_perms')
            st.caption(f'Random walks with restart from {set_name}. Scores are probabilities, not p values; lift divides by a uniform-teleport baseline. '
                       f'Linker tests count {set_name} neighbors (hypergeometric, BH within each network). Predictions are exploratory.')
            results = run_networks('random_walk', 'Diffusion', {'restart': float(restart), 'weighted': weighted, 'permutations': int(walk_perms)},
                                   lambda net: random_walk_analysis(graph_for(net), cohort, restart=float(restart),
                                                                    weighted=weighted and graph_for(net)['weighted_available'],
                                                                    permutations=int(walk_perms), seed=int(seed)))
            if results:
                top_n = st.slider('Top candidates per network', 10, 200, 50, 10, key='walk_top')
                tops = {n: r['ranked'].filter(~pl.col('is_seed').cast(pl.Boolean)).head(top_n).with_row_index('rank', offset=1)
                        for n, r in results.items()}
                stacked = stack_network_tables(tops)
                recurring = (stacked.group_by('protein').agg(pl.col('network').sort().alias('networks'), pl.len().alias('n_networks'),
                                                            pl.col('rank').min().alias('best_rank'))
                             .filter(pl.col('n_networks') >= 2).sort(['n_networks', 'best_rank'], descending=[True, False]))
                st.markdown(f'##### Candidates in the top {top_n} of at least two networks ({recurring.height})')
                table(recurring.with_columns(pl.col('networks').list.join(', ')), 'walk_recurring')
                st.markdown('##### Top candidates side by side')
                side = pl.DataFrame({'rank': list(range(1, top_n + 1))})
                for net, frame in tops.items():
                    side = side.join(frame.select('rank', pl.col('protein').alias(net)), on='rank', how='left')
                table(side, 'walk_side_by_side')
                trace = stack_network_tables({n: r['trace'] for n, r in results.items()})
                fig = px.line(trace, x='step', y='seed_mass', color='network', markers=True,
                              labels={'step': 'Restart-walk steps', 'seed_mass': f'Probability mass on {set_name}'},
                              title=f'How much walk probability stays on {set_name}')
                fig.update_layout(template='plotly_white', height=380)
                st.plotly_chart(fig, width='stretch', key='walk_trace')
                with st.expander('Full ranked tables'):
                    table(stacked, 'walk_ranked')
        elif method == 'Removal impact':
            removal_perms = st.number_input('Matched removal permutations (0 = observed only)', 0, 100, 50, 10, key='removal_perms')
            st.caption(f'Remove {set_name} and measure the surviving largest connected component, compared with equal-sized degree-matched removals. '
                       'Lower-tail p tests unusually strong fragmentation; BH across networks.')
            results = run_networks('removal_robustness', 'Removal impact', {'permutations': int(removal_perms)},
                                   lambda net: robustness_analysis(graph_for(net), cohort, permutations=int(removal_perms), seed=int(seed)),
                                   advanced=int(removal_perms) > 0)
            if results:
                rows = []
                for net, r in results.items():
                    s = r['summary']
                    mean, low, high = null_interval(r['null'], 'fraction_original_network')
                    rows.append({'network': net, 'removed_nodes': s['removed_nodes'],
                                 'remaining_lcc_fraction': s['remaining_lcc_fraction_original'],
                                 'null_mean': mean, 'null_low': low, 'null_high': high,
                                 'lcc_node_loss': s['lcc_node_loss'], 'p_lower': s['empirical_pvalue_lower_tail']})
                frame = pl.DataFrame(rows, schema_overrides={'p_lower': pl.Float64})
                frame = frame.with_columns(pl.Series('fdr_bh', bh_adjust(frame['p_lower'].to_numpy())))
                frame = frame.with_columns(pl.col('fdr_bh').map_elements(significance_label, return_dtype=pl.String).alias('significance'))
                interval_plot(frame, 'remaining_lcc_fraction', 'null_mean', 'null_low', 'null_high',
                              f'Largest component left after removing {set_name}', 'Fraction of original nodes',
                              'removal_plot', 'significance')
                table(frame, 'removal_table')
        else:
            trials = st.number_input('Matched permutations per batch (0 = observed only)', 0, 99, 19, 1, key='efficiency_trials')
            results = run_networks('removal_efficiency', 'Removal curves', {'permutations': int(trials), 'landmarks': 24},
                                   lambda net: removal_efficiency(graph_for(net), cohort, int(trials), int(seed)), advanced=True)
            if results:
                c1, c2 = st.columns(2)
                observed = stack_network_tables({n: r['table'] for n, r in results.items()})
                null = stack_network_tables({n: r['null'] for n, r in results.items()})
                for col, (column, title) in zip((c1, c2), [('lcc_fraction_original', 'Largest component / original network'),
                                                           ('global_efficiency_estimate', 'Estimated global efficiency')]):
                    with col:
                        fig = px.line(observed, x='removed', y=column, color='network', markers=True,
                                      labels={'removed': f'{set_name} proteins removed', column: title}, title=title)
                        if null.height:
                            reference = null.group_by('network', 'removed').agg(pl.col(column).mean()).sort('network', 'removed')
                            for trace in px.line(reference, x='removed', y=column, color='network').data:
                                trace.update(line_dash='dash', showlegend=False, hovertemplate='Matched null mean: %{y:.4g}<extra>%{fullData.name}</extra>')
                                fig.add_trace(trace)
                        fig.update_layout(template='plotly_white', height=420)
                        st.plotly_chart(fig, width='stretch', key=f'efficiency_{column}')
                st.caption(f'Solid: {set_name} removal. Dashed: mean of degree-matched removals of the same size.')
                table(observed, 'efficiency_table')

# ---- 7. Function ----------------------------------------------------------
if tab_open('Function'):
    with tab_containers['Function']:
        st.subheader('Function')
        metric_guide('Function')
        method = chooser('Analysis', ['GO enrichment', 'GO neighbourhood geometry', 'Protein metadata'], 'function_method')
        if method == 'GO enrichment':
            c1, c2, c3 = st.columns(3, vertical_alignment='bottom')
            aspect = c1.selectbox('GO aspect', ['BP', 'MF', 'CC', 'All'], key='go_aspect')
            min_size = c2.number_input('Minimum term size in network', 1, 1000, 5, 1, key='go_min_size')
            exclude_iea = c3.checkbox('Exclude IEA-only annotations', key='go_iea')
            st.caption(f'Enrichments compare {set_name} against each network’s background with a one-sided hypergeometric test (unannotated proteins included). '
                       'GO is propagated over is_a and part_of; BH within each network over every eligible term.')
            results = run_networks('go_enrichment', 'GO enrichment',
                                   {'aspect': aspect, 'min_size': int(min_size), 'exclude_iea': exclude_iea,
                                    'annotation_inputs': go_annotation_signatures()},
                                   lambda net: go_enrichment(annotation_file(), set(graph_for(net)['nodes']), cohort, aspect=aspect,
                                                             min_size=int(min_size), exclude_iea=exclude_iea))
            if results:
                top = st.slider('Top terms per network', 5, 30, 10, key='go_top')
                stacked = stack_network_tables({n: r['table'] for n, r in results.items()})
                terms = stacked.filter(pl.col('query_hits') > 0).sort('qvalue').group_by('network', maintain_order=True).head(top)['term'].unique()
                view = stacked.filter(pl.col('term').is_in(terms.implode()))
                if view.height:
                    order = view.group_by('term', 'name').agg(pl.col('qvalue').min()).sort('qvalue')
                    nets = list(results)
                    lookup = {(r['term'], r['network']): r for r in view.iter_rows(named=True)}
                    z, text = [], []
                    for row in order.iter_rows(named=True):
                        zr, tr = [], []
                        for n in nets:
                            hit = lookup.get((row['term'], n))
                            if hit is None or hit['qvalue'] is None:
                                zr.append(None); tr.append('')
                            else:
                                zr.append(-np.log10(max(hit['qvalue'], 1e-300)))
                                tr.append(f"{hit['fold']:.1f}×<br>{significance_label(hit['qvalue'])}")
                        z.append(zr); text.append(tr)
                    labels = [f"{(r['name'] or r['term'])[:60]}" for r in order.iter_rows(named=True)]
                    fig = go.Figure(go.Heatmap(z=z, x=nets, y=labels, colorscale='Blues', text=text, texttemplate='%{text}',
                                               colorbar=dict(title='−log₁₀ q'), xgap=3, ygap=2, hoverongaps=False))
                    fig.update_layout(template='plotly_white', title=f'Top GO terms enriched in {set_name} across networks (fold and BH stars)',
                                      yaxis_autorange='reversed', height=max(420, 120 + 26 * len(labels)),
                                      margin=dict(l=340), xaxis_title='Interactome')
                    st.plotly_chart(fig, width='stretch', key='go_heatmap')
                    st.caption('Empty cells: term not eligible in that network. Cells show fold enrichment and BH-adjusted stars.')
                else:
                    st.info(f'No eligible term has a {set_name} hit under these filters.')
                with st.expander('All tested terms'):
                    table(stacked, 'go_table')
            st.markdown('[GO enrichment background guidance](https://www.geneontology.org/docs/go-enrichment-analysis/)')
        elif method == 'GO neighbourhood geometry':
            aspects = {'P': 'Biological process', 'F': 'Molecular function', 'C': 'Cellular component'}
            aspect = st.selectbox('GO aspect', list(aspects), format_func=aspects.get, key='go_geometry_aspect')
            st.caption('Degree-matched comparison of partner GO coverage, Jaccard distance, local Dirichlet energy and neighbour signal variance.')
            results = run_networks('go_neighborhood_geometry', 'GO neighbourhood geometry',
                                   {'aspect': aspect, 'annotation_inputs': go_annotation_signatures(), 'max_nodes': 100},
                                   lambda net: go_neighborhood_geometry(graph_for(net), cohort, annotation_file(), aspect, int(seed), set_name=set_name),
                                   advanced=True)
            show_metric_comparison(results, 'go_geometry', degree_matched=True)
        else:
            st.caption(f'Within-{set_name} Spearman correlations between degree and sequence length, publication and function counts, and reported location differences.')
            partition = st.checkbox('Match partner GO annotations to both MoonLitDB function profiles', key='metadata_partner_partition')
            inputs = go_annotation_signatures() if partition else {}

            def compute_metadata(net):
                graph = graph_for(net)
                output = metadata_wiring(graph, cohort, selected)
                if partition:
                    annotation = annotation_file()
                    output['partner_partition'] = annotated_partner_partition(graph, selected, annotation)
                return output

            results = run_networks('metadata_wiring', 'Protein metadata', {'partner_partition': partition, 'annotation_inputs': inputs},
                                   compute_metadata, advanced=True)
            if results:
                tests = stack_network_tables({n: r['tests'] for n, r in results.items()}, p_column='p_value')
                fig = px.bar(tests.with_columns(pl.col('fdr_bh').map_elements(significance_label, return_dtype=pl.String).alias('stars')),
                             x='annotation', y='spearman_rho', color='network', barmode='group', text='stars',
                             labels={'spearman_rho': 'Spearman ρ with degree', 'annotation': 'Annotation'},
                             title=f'Do annotations track network degree within {set_name}?')
                fig.update_layout(template='plotly_white', height=420)
                st.plotly_chart(fig, width='stretch', key='metadata_plot')
                st.caption('BH across all annotations and networks. Descriptive; metadata can be missing or conflicting.')
                table(tests, 'metadata_tests')
                with st.expander('Per-protein metadata'):
                    table(stack_network_tables({n: r['table'] for n, r in results.items()}), 'metadata_table')
                    if partition:
                        table(stack_network_tables({n: r.get('partner_partition') for n, r in results.items()}), 'metadata_partners')

# ---- 8. Layers & scores ---------------------------------------------------
if tab_open('Layers & scores'):
    with tab_containers['Layers & scores']:
        st.subheader('Network layers and confidence scores')
        metric_guide('Layers & scores')
        method = chooser('Analysis', ['Multiplex pairs', 'Score filtration'], 'layers_method')
        if method == 'Multiplex pairs':
            pairs = [f'{a} × {b}' for a, b in combinations(active_nets, 2)]
            if not pairs:
                st.info('Select at least two interactomes.')
            else:
                c1, c2 = st.columns([3, 1])
                chosen = c1.multiselect('Network pairs', pairs, default=pairs[:min(3, len(pairs))], key='multiplex_pairs')
                coupling = c2.number_input('Interlayer edge weight', .01, 2., .25, .05, key='multiplex_coupling')
                st.caption('Layer participation and supra-graph PageRank for each two-layer network. Shared accessions couple the layers.')
                lookup = dict(zip(pairs, combinations(active_nets, 2)))
                results = run_items('multiplex_roles', 'Multiplex', {p: list(lookup[p]) for p in chosen}, {'coupling': coupling},
                                    lambda p: multiplex_roles(graph_for(lookup[p][0]), graph_for(lookup[p][1]), cohort, coupling, set_name=set_name),
                                    advanced=True)
                show_metric_comparison(results, 'multiplex')
        else:
            scored = {}
            for family, column in SCORE_COLUMNS.items():
                if family in datasets and column in pl.scan_parquet(datasets[family]).collect_schema():
                    scored[family] = (family, None, 0.)
                    continue
                for label, candidate in datasets.items():
                    if label.startswith(family + '_') and column in pl.scan_parquet(candidate).collect_schema():
                        audit = json.loads(candidate.with_suffix('.json').read_text())
                        minimum = float(audit['threshold'])
                        current = next((k for k, v in scored.items() if v[0] == family), None)
                        if current is None or minimum < scored[current][2]:
                            scored.pop(current, None)
                            scored[label] = (family, audit, minimum)
            if not scored:
                st.info('Prepare a canonical network with native scores to run filtration.')
            else:
                st.caption(f'{set_name} retention as the native confidence cutoff rises (inclusive). huMAP2 probability, BioPlex pInt and '
                           'HuRI author score are separate, uncalibrated scales; compare shapes, not values.')
                results = run_items(
                    'cohort_filtration', 'Score filtration', {label: [label] for label in scored}, {},
                    lambda label: cohort_filtration(
                        DATA, scored[label][0], cohort,
                        saved_version=datasets[label] if scored[label][1] is not None else None,
                        parent=datasets[label] if scored[label][1] is None else None,
                        minimum=scored[label][2], set_name=set_name))
                if results:
                    stacked = stack_network_tables({n: r['table'] for n, r in results.items()})
                    c1, c2 = st.columns(2)
                    for col, (column, label) in zip((c1, c2), [('cohort_nodes', f'{set_name} proteins retained'),
                                                               ('cohort_internal_edges', f'Within-{set_name} edges retained')]):
                        with col:
                            fig = px.line(stacked, x='threshold', y=column, color='network', markers=True, log_y=True,
                                          labels={'threshold': 'Inclusive native score cutoff', column: label}, title=label)
                            for threshold in (.05, .95):
                                fig.add_vline(x=threshold, line_dash='dash', line_color='#999')
                            fig.update_layout(template='plotly_white', height=420)
                            st.plotly_chart(fig, width='stretch', key=f'filtration_{column}')
                    st.caption('Dashed lines mark the standard huMAP2 cutoffs 0.05 and 0.95. The curves show the coverage/stringency '
                               'trade-off; they do not identify an optimal cutoff.')
                    table(stacked, 'filtration_table')

with st.expander('Analysis scope and reproducibility'):
    st.write('Every analysis runs separately on each selected network, with that network’s proteins as population. '
             'Where several networks are shown together, Benjamini–Hochberg is recomputed across all networks in the view '
             '(GO enrichment keeps one family per network because each network is its own universe). '
             'Each run saves CSV tables, parameters, network fingerprints, protein set accessions and the random seed in `data/analysis_runs/`. '
             'Advanced analyses are limited to networks with at most 500,000 edges. Degree bins are a practical adjustment, not exact degree matching.')
    spec = ROOT / 'MOONLIGHTING_TOPOLOGICAL_ANALYSES.md'
    if spec.exists():
        st.download_button('Download analysis catalogue', spec.read_text(encoding='utf-8'),
                           'MOONLIGHTING_TOPOLOGICAL_ANALYSES.md', 'text/markdown')
