"""Py4Cytoscape bridge, Cytoscape app clustering/hubs/diffusion automation, and interactive Cytoscape.js viewer."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import monotonic_ns

import plotly.express as px
import polars as pl
import py4cytoscape as p4c
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotate_clusters import annotation_input, annotation_node_table
from cluster_annotation_ui import annotation_panel
from cytoscape_bridge import (
    CLUSTERMAKER_ALGORITHMS,
    CYTOHUBBA_METRICS,
    DEFAULT_CYREST_PORT,
    build_base_url,
    build_cycmd_script,
    build_multiplex_subnetwork_tables,
    build_subnetwork_tables,
    check_cytoscape_connection,
    compute_cluster_enrichment_and_bridging,
    export_cx2_json,
    export_cytoscape_bundle_zip,
    export_graphml_string,
    export_sif_string,
    extract_cytohubba_scores,
    fetch_cytoscape_table_polars,
    push_clustering_to_cytoscape,
    render_cytoscapejs_html,
    run_clustermaker2,
    run_cytoscape_diffusion,
    run_network_analyzer,
    run_string_enrichment,
    send_to_cytoscape,
    subnetwork_to_graph_context,
    to_cytoscapejs_json,
)
from moonlighting import (
    CLUSTERING_CATALOG,
    GROUP_COLORS,
    GROUP_LABELS,
    SOURCES,
    clustering_comparison_heatmap,
    cohort_ids,
    community_analysis,
    compare_clustering_algorithms,
    graph_from_path,
    load_moon_data,
)
from networks import discover_datasets

DATA = ROOT / 'data'

st.set_page_config(
    page_title='Cytoscape Bridge | Human Interactome Atlas',
    page_icon=':material/share:',
    layout='wide',
)
st.caption('HUMAN INTERACTOME ATLAS')
st.subheader('Cytoscape Desktop')
st.write(
    'Extract moonlighting subnetworks, 1-hop linker bridges, and negative-curvature switch edges in Polars, '
    'explore them in the embedded Cytoscape.js viewer, push them to Cytoscape Desktop via `py4cytoscape`, '
    'and run your installed Cytoscape apps (`clusterMaker2`, `cytoHubba`, `NetworkAnalyzer`, `diffusion`, `stringApp`) '
    'with results pulled back into Polars.'
)

isoforms = st.session_state.get('isoform_policy', 'collapse')
datasets = discover_datasets(DATA, isoforms=isoforms)
if not datasets:
    st.info('Prepare canonical networks on Interactomes Overview first.')
    st.stop()

if st.session_state.get('cy_isoform_policy') != isoforms:
    for result_key in ('cy_subnetwork', 'cm_result', 'mp_single_res', 'mp_suite_res',
                       'na_result', 'diff_result', 'ch_result', 'str_result', 'last_sent_suid'):
        st.session_state.pop(result_key, None)
    st.session_state['cy_isoform_policy'] = isoforms


@st.cache_data(show_spinner=False)
def cached_moon_data(mtime: int, mapping_mtime: int) -> pl.DataFrame:
    return load_moon_data(DATA)


@st.cache_resource(show_spinner=False, max_entries=3)
def cached_graph(path: str, mtime: int, size: int) -> dict:
    return graph_from_path(Path(path))


raw_path = DATA / 'allmoon_with_moonlitdb_human.csv'
mapping_file = DATA / 'mappings/moonlighting_uniprot.json'
if not mapping_file.exists():
    st.warning('Resolve the moonlighting protein IDs on Topological Analysis first.')
    st.stop()

moon = cached_moon_data(raw_path.stat().st_mtime_ns, mapping_file.stat().st_mtime_ns)
analysis = st.session_state.get('visualization_analysis_context', {})
preferred = [name for name in st.session_state.get('overview_networks', []) if name in datasets]
network_options = list(dict.fromkeys([*preferred, *datasets]))
default_network = analysis.get('network')
if default_network not in network_options:
    default_network = network_options[0]

with st.sidebar:
    st.title('Cytoscape Bridge')
    st.subheader('CyREST Connection')
    cy_host = st.text_input('Cytoscape Host', value='127.0.0.1', key='cy_host')
    cy_port = st.number_input('CyREST Port', min_value=1, max_value=65535, value=int(DEFAULT_CYREST_PORT), step=1, key='cy_port')
    base_url = build_base_url(cy_host, int(cy_port))
    st.caption(f'Endpoint: `{base_url}`')

    st.subheader('Interactome & Protein Set')
    network_name = st.selectbox('Primary interactome', network_options,
                                index=network_options.index(default_network), key='cy_network')
    sources = st.multiselect(
        'Moonlighting sources (empty = all)',
        SOURCES,
        default=analysis.get('sources', ['moonlitdb']),
        format_func=lambda s: GROUP_LABELS.get(s, s),
        key='cy_sources',
    )
    require_all = st.checkbox('Require all selected sources', value=analysis.get('require_all', False), key='cy_require_all')
    reviewed_only = st.checkbox('Reviewed source entries only', value=analysis.get('reviewed_only', False), key='cy_reviewed')
    min_pubs = st.number_input('Minimum MoonLitDB publications', 0, 10000, analysis.get('min_pubs', 0), 1, key='cy_min_pubs')

cohort, selected_rows = cohort_ids(moon, sources, require_all, reviewed_only, min_pubs)
network_path = datasets[network_name]
network_stat = network_path.stat()
subnetwork_inputs = (str(network_path), network_stat.st_mtime_ns, network_stat.st_size, frozenset(cohort))
if st.session_state.get('cy_subnetwork_inputs') != subnetwork_inputs:
    st.session_state.pop('cy_subnetwork', None)
    st.session_state['cy_subnetwork_inputs'] = subnetwork_inputs


def get_graph(name: str | None = None) -> dict:
    target = name or network_name
    p = datasets[target]
    info = p.stat()
    return cached_graph(str(p), info.st_mtime_ns, info.st_size)


# Check live connection to Cytoscape Desktop
conn_status = check_cytoscape_connection(base_url=base_url)

with st.container(border=True):
    col_status, col_info, col_actions = st.columns([2, 3, 2])
    with col_status:
        if conn_status['connected']:
            vinfo = conn_status.get('version_info', {})
            st.success(
                f"Connected to Cytoscape {vinfo.get('cytoscapeVersion', '')} "
                f"(CyREST {vinfo.get('apiVersion', 'v1')}, py4cytoscape {vinfo.get('py4cytoscapeVersion', p4c.__version__)})"
            )
        elif conn_status.get('port_conflict_detected'):
            st.error(f"Port {int(cy_port)} conflict: {conn_status.get('conflicting_service')}")
        else:
            st.warning(f"Cytoscape CyREST not reachable at {base_url}")
    with col_info:
        if conn_status['connected']:
            apps = conn_status.get('installed_apps', [])
            app_names = sorted(a.get('appName', '') for a in apps if a.get('status') == 'Installed')
            st.caption(f"**Installed Cytoscape Apps ({len(app_names)}):** " + (', '.join(app_names) if app_names else 'Core apps loaded'))
            nets = conn_status.get('networks', [])
            if nets:
                st.caption('**Active Session Networks:** ' + ', '.join(f"{n['name']} (SUID:{n['suid']})" for n in nets[:8]))
            else:
                st.caption('**Active Session Networks:** None loaded yet in Cytoscape Desktop.')
        else:
            st.caption(conn_status.get('remediation', ''))
            if conn_status.get('port_conflict_detected'):
                st.info(
                    'Tip: LM Studio uses port 1234 for its local LLM server. Either stop the server in LM Studio '
                    'and restart Cytoscape, or change `rest.port` in Cytoscape (`Edit → Preferences → Properties... → rest.port`) '
                    'to `1235` and set CyREST Port in the sidebar to `1235`.'
                )
    with col_actions:
        if st.button('Refresh Cytoscape Connection', width='stretch'):
            st.rerun()

tab_sub, tab_cluster, tab_apps, tab_export = st.tabs([
    '1. Subnetwork & Bridge Builder',
    '2. clusterMaker2 & Module Bridging',
    '3. cytoHubba, NetworkAnalyzer & Diffusion',
    '4. Export Bundle (.cyjs / .cx2 / .graphml / .cycmd)',
])

# ---------------------------------------------------------------------------
# TAB 1: Subnetwork & Bridge Builder + Interactive Cytoscape.js Viewer
# ---------------------------------------------------------------------------
with tab_sub:
    st.subheader(f'Build Moonlighting & Bridge Subnetwork ({network_name})')
    g_primary = get_graph(network_name)
    cohort_in_primary = sorted(p for p in cohort if p in g_primary['index'])

    # Map canonical IDs to gene labels for convenient selection
    gene_lookup_df = (
        moon.filter(pl.col('canonical_id').is_in(cohort_in_primary))
        .group_by('canonical_id')
        .agg(pl.col('Gene Names').str.split(' ').list.first().first().fill_null('').alias('gene'))
    )
    gene_map = dict(zip(gene_lookup_df['canonical_id'].to_list(), gene_lookup_df['gene'].to_list()))

    c1, c2, c3 = st.columns(3)
    mode_options = {
        'cohort_linkers': 'Protein Set + Top 1-Hop Linker Bridges',
        'ego_bridge': 'Protein Ego-Network & Ego-Fragmentation Bridge',
        'top_bridges': 'Top Bridging-Centrality Hubs & Incident Bridges',
        'multiplex': 'Cross-Network Multiplex Comparison (A vs B)',
    }
    sub_mode = c1.selectbox('Subnetwork extraction mode', list(mode_options), format_func=lambda k: mode_options[k], key='cy_sub_mode')
    max_nodes = c2.slider('Maximum nodes in subnetwork', min_value=20, max_value=400, value=120, step=10, key='cy_max_nodes')

    seed_proteins: list[str] = []
    hop_radius = 1
    max_linkers = 25
    second_network = None

    if sub_mode == 'cohort_linkers':
        max_linkers = c3.slider('Top 1-hop linker candidates to include', 0, 100, 25, 5, key='cy_max_linkers')
    elif sub_mode == 'ego_bridge':
        hop_radius = c3.radio('Ego-network radius (hops)', [1, 2], index=0, horizontal=True, key='cy_ego_radius')
        default_seeds = [p for p in ['P04406', 'P06733', 'P60174', 'P00558'] if p in cohort_in_primary] or cohort_in_primary[:2]
        seed_proteins = st.multiselect(
            'Select moonlighting seed protein(s) to inspect local ego-bridging',
            cohort_in_primary,
            default=default_seeds[:2],
            format_func=lambda p: f"{gene_map.get(p) or p} ({p})" if gene_map.get(p) else p,
            key='cy_ego_seeds',
        )
    elif sub_mode == 'top_bridges':
        include_bg = c3.checkbox('Include 1-hop non-moonlighting bridge partners', value=True, key='cy_top_bg')
    elif sub_mode == 'multiplex':
        other_nets = [n for n in datasets if n != network_name]
        second_network = c3.selectbox('Comparison network B', other_nets, key='cy_multiplex_b')
        default_seeds = [p for p in ['P04406', 'P06733', 'P60174', 'P00558'] if p in cohort_in_primary] or cohort_in_primary[:10]
        seed_proteins = st.multiselect(
            'Seed moonlighting proteins (shared across networks)',
            cohort_in_primary,
            default=default_seeds[:8],
            format_func=lambda p: f"{gene_map.get(p) or p} ({p})" if gene_map.get(p) else p,
            key='cy_mux_seeds',
        )

    if st.button('Build Subnetwork & Compute Bridge Curvature', type='primary', key='btn_build_sub'):
        with st.spinner('Extracting Polars subnetwork and computing Forman–Ricci curvature & bridge metrics…'):
            try:
                if sub_mode == 'multiplex' and second_network:
                    sub_res = build_multiplex_subnetwork_tables(
                        g_primary,
                        get_graph(second_network),
                        network_name,
                        second_network,
                        moon,
                        cohort,
                        seed_proteins=seed_proteins,
                        max_nodes=int(max_nodes),
                    )
                    sub_title = f"{network_name}_vs_{second_network}_multiplex"
                else:
                    sub_res = build_subnetwork_tables(
                        g_primary,
                        moon,
                        cohort,
                        mode=sub_mode,
                        seed_proteins=seed_proteins,
                        hop_radius=int(hop_radius),
                        max_linkers=int(max_linkers),
                        max_nodes=int(max_nodes),
                    )
                    sub_title = f"{network_name}_{sub_mode}_{len(sub_res['nodes'])}n"
                st.session_state['cy_subnetwork'] = {
                    'title': sub_title,
                    'nodes': sub_res['nodes'],
                    'edges': sub_res['edges'],
                    'summary': sub_res['summary'],
                }
            except Exception as exc:
                st.error(f'Failed to build subnetwork: {exc}')

    # Auto-build initial subnetwork if none stored yet
    if 'cy_subnetwork' not in st.session_state and cohort_in_primary:
        sub_res = build_subnetwork_tables(
            g_primary,
            moon,
            cohort,
            mode='cohort_linkers',
            max_linkers=25,
            max_nodes=120,
        )
        st.session_state['cy_subnetwork'] = {
            'title': f"{network_name}_cohort_linkers_{len(sub_res['nodes'])}n",
            'nodes': sub_res['nodes'],
            'edges': sub_res['edges'],
            'summary': sub_res['summary'],
        }

    active_sub = st.session_state.get('cy_subnetwork')
    if active_sub:
        nodes_df: pl.DataFrame = active_sub['nodes']
        edges_df: pl.DataFrame = active_sub['edges']
        summ: dict = active_sub['summary']

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Subnetwork Nodes', f"{nodes_df.height:,}")
        m2.metric('Subnetwork Edges', f"{edges_df.height:,}")
        m3.metric('Moonlighting / Linkers', f"{summ.get('moonlighting_nodes', 0)} / {summ.get('linker_nodes', 0)}")
        m4.metric('Bridge Edges (0 Triangles)', f"{summ.get('bridge_edges_zero_triangles', 0):,}")
        m5.metric('Negative Curvature Edges (κF < 0)', f"{summ.get('negative_forman_ricci_edges', 0):,}")

        with st.container(border=True):
            st.markdown('#### Send to Cytoscape Desktop (`py4cytoscape`)')
            sc1, sc2, sc3, sc4 = st.columns([2, 2, 2, 2])
            net_title_input = sc1.text_input('Cytoscape Network Title', value=active_sub['title'], key='cy_send_title')
            collection_input = sc2.text_input('Cytoscape Collection', value='Human Interactome Atlas', key='cy_send_coll')
            cy_layout = sc3.selectbox(
                'Cytoscape Layout',
                ['force-directed', 'kamada-kawai', 'cose', 'attribute-circle', 'circular', 'hierarchical', 'grid'],
                key='cy_send_layout',
            )
            if sc4.button('Send to Cytoscape Desktop', type='primary', width='stretch', disabled=not conn_status['connected']):
                with st.spinner('Sending network, Polars attributes, and visual style via py4cytoscape…'):
                    try:
                        sent = send_to_cytoscape(
                            nodes_df,
                            edges_df,
                            title=net_title_input,
                            collection=collection_input,
                            layout=cy_layout,
                            apply_style=True,
                            base_url=base_url,
                        )
                        st.session_state['last_sent_suid'] = sent['network_suid']
                        st.success(
                            f"Created network **{sent['title']}** in Cytoscape Desktop "
                            f"(SUID: `{sent['network_suid']}`, {sent['nodes_sent']} nodes, {sent['edges_sent']} edges)!"
                        )
                    except Exception as exc:
                        st.error(f'Py4Cytoscape send failed: {exc}')

        st.markdown('#### Interactive Cytoscape.js Network & Bridge Viewer')
        vc1, vc2, vc3 = st.columns([2, 2, 2])
        color_options = [c for c in ['node_role', 'ego_component', 'source_sets'] + [col for col in nodes_df.columns if 'cluster' in col.lower()] if c in nodes_df.columns]
        viewer_color = vc1.selectbox('Color nodes by', color_options, index=0, key='cyjs_color_by')
        viewer_layout = vc2.selectbox('Initial viewer layout', ['cose', 'concentric', 'circle', 'breadthfirst', 'grid'], index=0, key='cyjs_layout')
        only_bridges = vc3.checkbox('Dim non-bridge edges initially', value=False, key='cyjs_only_bridges')

        html_widget = render_cytoscapejs_html(
            nodes_df,
            edges_df,
            color_by=viewer_color,
            layout_name=viewer_layout,
            highlight_bridges_only=only_bridges,
            height=620,
        )
        components.html(html_widget, height=630, scrolling=False)

        col_nodes, col_edges = st.columns(2)
        with col_nodes:
            st.markdown('##### Top Bridge & Linker Proteins in Subnetwork')
            st.dataframe(
                nodes_df.sort(['is_moonlighting', 'moonlighting_neighbors', 'bridging_coefficient'], descending=[True, True, True]),
                width='stretch',
                hide_index=True,
            )
        with col_edges:
            st.markdown('##### Inter-Module Switch & Bridge Edges (Sorted by Forman–Ricci Curvature $\\kappa_F$)')
            st.dataframe(
                edges_df.sort(['is_bridge_edge', 'forman_ricci_sub', 'shared_triangles'], descending=[True, False, False]),
                width='stretch',
                hide_index=True,
            )

# ---------------------------------------------------------------------------
# TAB 2: clusterMaker2 Clustering & Inter-Cluster Bridging
# ---------------------------------------------------------------------------
with tab_cluster:
    st.subheader('Run `clusterMaker2` in Cytoscape Desktop & Analyze Multi-Cluster Bridging')
    st.write(
        'Execute any installed `clusterMaker2` clustering algorithm (`leiden`, `mcl`, `louvain`, `mcode`, `glay`, `infomap`, `ap`, `fcm`) '
        'on the active Cytoscape network via `py4cytoscape`, pull the resulting cluster assignments back into Polars, '
        'and test which clusters are enriched in moonlighting proteins and which proteins bridge multiple `clusterMaker2` complexes.'
    )

    if not conn_status['connected']:
        st.info('Connect to Cytoscape Desktop in the sidebar to run `clusterMaker2` live, or use the `.cycmd` script in Tab 4.')
    else:
        nets = conn_status.get('networks', [])
        if not nets:
            st.info('Send a subnetwork to Cytoscape Desktop in Tab 1 first.')
        else:
            net_choices = {n['suid']: f"{n['name']} (SUID: {n['suid']})" for n in nets}
            cc1, cc2, cc3 = st.columns(3)
            target_suid = cc1.selectbox('Target Cytoscape Network', list(net_choices), format_func=lambda k: net_choices[k], key='cm_net_suid')
            algo_key = cc2.selectbox(
                'clusterMaker2 Algorithm',
                list(CLUSTERMAKER_ALGORITHMS),
                format_func=lambda k: CLUSTERMAKER_ALGORITHMS[k]['label'],
                key='cm_algo',
            )
            spec = CLUSTERMAKER_ALGORITHMS[algo_key]
            custom_param_str = cc3.text_input(
                'Algorithm parameters (key=value)',
                value=' '.join(f'{k}={v}' for k, v in spec['default_params'].items()),
                key=f'cm_params_{algo_key}',
            )

            if st.button(f"Run {spec['label']} via Py4Cytoscape", type='primary', key='btn_run_cm'):
                parsed_params = {}
                for token in custom_param_str.split():
                    if '=' in token:
                        k, v = token.split('=', 1)
                        parsed_params[k.strip()] = v.strip()
                with st.spinner(f"Running `{spec['command']}` in Cytoscape Desktop and fetching tables into Polars…"):
                    try:
                        cm_res = run_clustermaker2(
                            algo_key,
                            network=int(target_suid),
                            params=parsed_params,
                            cohort=cohort,
                            base_url=base_url,
                        )
                        cm_res['annotation_target'] = (base_url, target_suid, monotonic_ns())
                        st.session_state['cm_result'] = cm_res
                        if cm_res['cluster_column'] and 'cy_subnetwork' in st.session_state:
                            # Merge cluster column into active_sub['nodes'] if node IDs match
                            c_col = cm_res['cluster_column']
                            pulled = cm_res['node_table']
                            if 'name' in pulled.columns and c_col in pulled.columns:
                                mapping_df = pulled.select(
                                    pl.col('name').cast(pl.String).alias('protein'),
                                    pl.col(c_col).cast(pl.String).alias(c_col),
                                ).unique(subset=['protein'])
                                current_nodes = st.session_state['cy_subnetwork']['nodes']
                                if c_col in current_nodes.columns:
                                    current_nodes = current_nodes.drop(c_col)
                                st.session_state['cy_subnetwork']['nodes'] = current_nodes.join(
                                    mapping_df, on='protein', how='left', maintain_order='left'
                                )
                    except Exception as exc:
                        st.error(f'clusterMaker2 command failed: {exc}')

            # Also allow analyzing any existing cluster column in the Cytoscape node table
            if st.button('Inspect Existing Cluster Columns in Cytoscape Network', key='btn_inspect_existing_cm'):
                try:
                    ndf = fetch_cytoscape_table_polars(network=int(target_suid), table='node', base_url=base_url)
                    edf = fetch_cytoscape_table_polars(network=int(target_suid), table='edge', base_url=base_url)
                    cluster_cols = [c for c in ndf.columns if 'cluster' in c.lower() or c in ('ego_component', 'community')]
                    if cluster_cols:
                        chosen_col = cluster_cols[0]
                        analysis = compute_cluster_enrichment_and_bridging(ndf, edf, chosen_col, cohort=cohort)
                        st.session_state['cm_result'] = {
                            'command': f'inspect existing column {chosen_col}',
                            'cluster_column': chosen_col,
                            'annotation_target': (base_url, target_suid, monotonic_ns()),
                            'node_table': ndf,
                            'cluster_enrichment': analysis['cluster_enrichment'],
                            'node_cluster_bridging': analysis['node_cluster_bridging'],
                        }
                    else:
                        st.warning(f'No cluster columns found yet in network {target_suid}. Available columns: {", ".join(ndf.columns)}')
                except Exception as exc:
                    st.error(f'Failed to fetch Cytoscape tables: {exc}')

            annotation = st.expander('Annotate Cytoscape clusters with GO and Reactome',
                                     key='cy_cluster_annotation', on_change='rerun')
            
            if annotation.open:
                with annotation:
                    current = st.session_state.get('cm_result', {})
                    target = current.get('annotation_target', ())
                    current_column = current.get('cluster_column') if target[:2] == (base_url, target_suid) else None
                    st.caption('The selected Cytoscape network supplies all background proteins. '
                               'Use the full clustering network and rerun after editing its assignments.')
                    fields = st.columns(3)
                    accession_column = fields[0].text_input('UniProt node column', 'name', key='cy_annotation_protein').strip()
                    cluster_column = fields[1].text_input(
                        'Cluster node column', current_column or 'community',
                        key=f'cy_annotation_column_{target_suid}_{current_column}',
                    ).strip()
                    separator = fields[2].text_input(
                        'Text membership separator (optional)', key='cy_annotation_separator',
                        help='Leave blank for a list column or single cluster IDs.',
                    ) or None
                    compare = st.checkbox('Compare with Cytoscape AutoAnnotate', key='cy_annotation_compare')
                    comparison = None
                    if compare:
                        description_column = st.text_input(
                            'Protein description node column (optional)', key='cy_annotation_description',
                            help='AutoAnnotate also runs a separate pass using GO and Reactome term names.',
                        ).strip() or None
                        comparison = {'network': target_suid, 'uniprot_column': accession_column,
                                      'description_column': description_column, 'base_url': base_url}
                    if accession_column and cluster_column:
                        annotation_panel(
                            'cy_live_annotation', (base_url, target_suid, accession_column, cluster_column,
                                                   separator, target if current_column else None),
                            lambda: annotation_input(
                                annotation_node_table(target_suid, base_url), target_suid,
                                accession_column, cluster_column, separator, base_url,
                            ),
                            DATA, cytoscape=comparison,
                        )
                    else:
                        st.info('Enter the UniProt and cluster node column names.')

    cm_result = st.session_state.get('cm_result')
    if cm_result:
        st.caption(f"Command executed: `{cm_result.get('command')}` · Cluster column: `{cm_result.get('cluster_column')}`")
        enr_df: pl.DataFrame = cm_result.get('cluster_enrichment', pl.DataFrame())
        bridging_df: pl.DataFrame = cm_result.get('node_cluster_bridging', pl.DataFrame())

        if len(enr_df):
            st.markdown('#### `clusterMaker2` Cluster Enrichment (Hypergeometric + BH FDR)')
            st.dataframe(enr_df, width='stretch', hide_index=True)
            st.download_button(
                'Download cluster enrichment (CSV)',
                enr_df.write_csv(),
                'clustermaker2_cluster_enrichment.csv',
                'text/csv',
                key='dl_cm_enr',
            )
        if len(bridging_df):
            st.markdown('#### Inter-Cluster Bridge Proteins Across `clusterMaker2` Modules')
            fig_cm = px.scatter(
                bridging_df,
                x='clustered_degree',
                y='clustermaker_participation',
                color='is_moonlighting',
                hover_data=['protein', 'label', 'assigned_cluster', 'distinct_neighbor_clusters', 'inter_cluster_edges'],
                title='Proteins Bridging Multiple clusterMaker2 Modules (Participation Coefficient vs Degree)',
                labels={
                    'clustered_degree': 'Subnetwork Degree',
                    'clustermaker_participation': 'clusterMaker2 Participation Coefficient (P_i)',
                },
            )
            st.plotly_chart(fig_cm, width='stretch')
            st.dataframe(bridging_df, width='stretch', hide_index=True)
            st.download_button(
                'Download inter-cluster bridging table (CSV)',
                bridging_df.write_csv(),
                'clustermaker2_node_bridging.csv',
                'text/csv',
                key='dl_cm_bridge',
            )

    st.divider()
    st.subheader('Multi-Paradigm Cached Clustering Suite (Classic + Overlapping + Fuzzy)')
    st.write(
        'Run and cache all 19 Classic/Hard (`louvain`, `greedy_modularity`, `label_propagation`, `asyn_fluid`, `mcl`, `walktrap_markov`, `spectral`, `ravasz_tom`, `k_core_shells`), '
        'Overlapping (`ego_splitting`, `link_communities`, `k_clique_percolation`, `demon`, `mcode_complexes`, `bigclam_nmf`), '
        'and Fuzzy/Soft (`fuzzy_c_means`, `fuzzy_nmf`, `fuzzy_rwr_diffusion`, `fuzzy_neighborhood`) clustering algorithms on the extracted subnetwork or full interactome, '
        'and push their node columns (`community`, `communities_str`, `num_communities`, `is_overlapping`, `participation`, `fuzzy_entropy`, `top2_ratio`) directly to Cytoscape Desktop!'
    )
    if active_sub:
        mp_c1, mp_c2, mp_c3 = st.columns([2, 1, 1])
        mp_scope = mp_c1.radio('Clustering Target Graph', ['Extracted Subnetwork', 'Full Primary Interactome (cached)'], horizontal=True, key='mp_scope')
        mp_algo = mp_c2.selectbox(
            'Single Algorithm to Inspect / Push to Cytoscape',
            list(CLUSTERING_CATALOG.keys()),
            format_func=lambda a: f"[{CLUSTERING_CATALOG[a]['paradigm'].upper()}] {a}",
            key='mp_single_algo',
        )
        mp_res_val = mp_c3.number_input('Resolution', 0.1, 5.0, 1.0, 0.1, key='mp_res')

        btn_c1, btn_c2 = st.columns(2)
        if btn_c1.button(f"Run & Cache {CLUSTERING_CATALOG[mp_algo]['label']}", type='primary', key='btn_mp_single'):
            with st.spinner(f"Running & caching {mp_algo}…"):
                try:
                    target_g = subnetwork_to_graph_context(active_sub['nodes'], active_sub['edges']) if mp_scope == 'Extracted Subnetwork' else get_graph(network_name)
                    single_res = community_analysis(target_g, cohort, seed=42, resolution=float(mp_res_val), algorithm=mp_algo, use_cache=True)
                    st.session_state['mp_single_res'] = single_res
                    if conn_status['connected'] and conn_status.get('networks'):
                        push_info = push_clustering_to_cytoscape(single_res['metrics'], mp_algo, network=conn_status['networks'][0]['suid'], base_url=base_url)
                        st.success(f"Cached `{mp_algo}` and pushed `{push_info['prefix']}_*` columns to Cytoscape SUID {push_info['suid']}!")
                except Exception as exc:
                    st.error(f"Clustering failed: {exc}")

        if btn_c2.button('Run & Cache ALL 19 Clustering Algorithms (Classic + Overlapping + Fuzzy)', key='btn_mp_all'):
            with st.spinner('Running & caching all 19 Classic, Overlapping, and Fuzzy clustering algorithms…'):
                try:
                    target_g = subnetwork_to_graph_context(active_sub['nodes'], active_sub['edges']) if mp_scope == 'Extracted Subnetwork' else get_graph(network_name)
                    all_suite = compare_clustering_algorithms(target_g, cohort, algorithms=list(CLUSTERING_CATALOG.keys()), seed=42, resolution=float(mp_res_val), use_cache=True)
                    st.session_state['mp_suite_res'] = all_suite
                except Exception as exc:
                    st.error(f"Suite comparison failed: {exc}")

        if 'mp_single_res' in st.session_state:
            s_res = st.session_state['mp_single_res']
            st.json(s_res['summary'])
            st.dataframe(s_res['tests'], width='stretch', hide_index=True)
            st.dataframe(s_res['metrics'].sort(['num_communities', 'fuzzy_entropy', 'participation'], descending=[True, True, True]), width='stretch', hide_index=True)

        if 'mp_suite_res' in st.session_state:
            suite_res = st.session_state['mp_suite_res']
            st.markdown('#### All 19 Clustering Algorithms — Summary & Moonlighting Role Comparison')
            st.dataframe(suite_res['summary_table'], width='stretch', hide_index=True)
            st.plotly_chart(clustering_comparison_heatmap(suite_res['effects']), width='stretch')

# ---------------------------------------------------------------------------
# TAB 3: cytoHubba, NetworkAnalyzer, Heat Diffusion & stringApp
# ---------------------------------------------------------------------------
with tab_apps:
    st.subheader('Integrate `NetworkAnalyzer`, `diffusion`, `cytoHubba`, `stringApp` & Custom CyREST Commands')

    if not conn_status['connected']:
        st.info('Connect to Cytoscape Desktop in the sidebar to run live app commands.')
    else:
        nets = conn_status.get('networks', [])
        net_choices = {n['suid']: f"{n['name']} (SUID: {n['suid']})" for n in nets} if nets else {}
        selected_app_suid = st.selectbox(
            'Active Cytoscape Network for App Execution',
            list(net_choices) if net_choices else [None],
            format_func=lambda k: net_choices.get(k, 'No network loaded'),
            key='app_target_suid',
        )

        col_na, col_diff, col_ch = st.columns(3)
        with col_na:
            with st.container(border=True):
                st.markdown('#### 1. `NetworkAnalyzer`')
                st.caption('Runs Cytoscape `analyzer analyze` and compares Moonlighting vs Background across all computed centralities.')
                if st.button('Run NetworkAnalyzer', type='primary', width='stretch', disabled=selected_app_suid is None, key='btn_na'):
                    with st.spinner('Running Cytoscape NetworkAnalyzer…'):
                        try:
                            st.session_state['na_result'] = run_network_analyzer(
                                network=int(selected_app_suid),
                                cohort=cohort,
                                base_url=base_url,
                            )
                        except Exception as exc:
                            st.error(f'NetworkAnalyzer failed: {exc}')

        with col_diff:
            with st.container(border=True):
                st.markdown('#### 2. Cytoscape `diffusion`')
                diff_time = st.number_input('Diffusion time parameter', 0.001, 5.0, 0.1, 0.05, key='cy_diff_time')
                if st.button('Run Heat Diffusion from Protein Set Seeds', type='primary', width='stretch', disabled=selected_app_suid is None, key='btn_diff'):
                    with st.spinner('Selecting moonlighting seeds and running Cytoscape heat diffusion…'):
                        try:
                            ndf_curr = fetch_cytoscape_table_polars(network=int(selected_app_suid), table='node', base_url=base_url)
                            name_col = 'name' if 'name' in ndf_curr.columns else 'id'
                            net_nodes = set(ndf_curr[name_col].cast(pl.String).to_list())
                            seeds_in_cy = sorted(net_nodes & cohort)
                            if not seeds_in_cy:
                                st.warning('No mapped moonlighting protein set nodes found in this Cytoscape network.')
                            else:
                                st.session_state['diff_result'] = run_cytoscape_diffusion(
                                    seeds_in_cy,
                                    network=int(selected_app_suid),
                                    time_param=float(diff_time),
                                    base_url=base_url,
                                )
                        except Exception as exc:
                            st.error(f'Cytoscape diffusion failed: {exc}')

        with col_ch:
            with st.container(border=True):
                st.markdown('#### 3. `cytoHubba` & `stringApp`')
                st.caption('Pull `cytoHubba` hub/bottleneck scores (`MCC`, `BottleNeck`, `EPC`, etc.) or trigger `stringApp` enrichment.')
                if st.button('Pull cytoHubba Scores into Polars', width='stretch', disabled=selected_app_suid is None, key='btn_ch'):
                    with st.spinner('Reading node table from Cytoscape…'):
                        try:
                            st.session_state['ch_result'] = extract_cytohubba_scores(
                                network=int(selected_app_suid),
                                cohort=cohort,
                                base_url=base_url,
                            )
                        except Exception as exc:
                            st.error(f'Failed to read cytoHubba columns: {exc}')
                if st.button('Run stringApp Functional Enrichment', width='stretch', disabled=selected_app_suid is None, key='btn_str_enr'):
                    with st.spinner('Running stringApp enrichment…'):
                        try:
                            st.session_state['str_result'] = run_string_enrichment(
                                network=int(selected_app_suid),
                                base_url=base_url,
                            )
                        except Exception as exc:
                            st.error(f'stringApp enrichment note: {exc}')

        if 'na_result' in st.session_state:
            na_res = st.session_state['na_result']
            st.markdown('#### `NetworkAnalyzer` Results (Polars)')
            if len(na_res['tests']):
                st.dataframe(na_res['tests'], width='stretch', hide_index=True)
            with st.expander('Full NetworkAnalyzer Node Table'):
                st.dataframe(na_res['node_table'], width='stretch', hide_index=True)

        if 'diff_result' in st.session_state:
            diff_res = st.session_state['diff_result']
            st.markdown('#### Cytoscape Heat Diffusion Results (Polars)')
            st.dataframe(diff_res['node_table'], width='stretch', hide_index=True)

        if 'ch_result' in st.session_state:
            ch_res = st.session_state['ch_result']
            st.markdown('#### `cytoHubba` Hub & Bottleneck Analysis (Polars)')
            if ch_res['cytohubba_columns']:
                st.success(f"Detected cytoHubba columns: {', '.join(ch_res['cytohubba_columns'])}")
                if len(ch_res['tests']):
                    st.dataframe(ch_res['tests'], width='stretch', hide_index=True)
                st.dataframe(ch_res['node_table'], width='stretch', hide_index=True)
            else:
                st.info(
                    'No `cytoHubba` columns (`MCC`, `BottleNeck`, `EPC`, `DMNC`, `MNC`, etc.) found on this network yet. '
                    'In Cytoscape Desktop, open the **cytoHubba** panel on the left, click **Calculate** (all 12 methods), '
                    'and then click **Pull cytoHubba Scores into Polars** above!'
                )

        st.divider()
        st.markdown('#### Interactive CyREST Command & Namespace Explorer (`py4cytoscape.commands_post` / `commands_help`)')
        cmd_col1, cmd_col2 = st.columns([1, 2])
        namespace = cmd_col1.selectbox(
            'Inspect Command Namespace',
            ['cluster', 'analyzer', 'diffusion', 'string', 'cluego', 'enrichmentmap', 'network', 'layout', 'vizmap', 'session', 'apps'],
            key='cy_ns_select',
        )
        if cmd_col1.button('List Available Commands in Namespace', key='btn_ns_help'):
            try:
                help_items = p4c.commands_help(namespace, base_url=base_url)
                st.session_state['ns_help_output'] = {'namespace': namespace, 'commands': help_items}
            except Exception as exc:
                st.warning(f'Namespace `{namespace}` help returned: {exc}')

        if 'ns_help_output' in st.session_state:
            cmd_col1.json(st.session_state['ns_help_output'])

        raw_cmd = cmd_col2.text_input(
            'Execute arbitrary Cytoscape Command string via `p4c.commands_post`',
            value='apps list installed',
            key='cy_raw_cmd',
        )
        if cmd_col2.button('Run Command via Py4Cytoscape', key='btn_raw_cmd'):
            try:
                out = p4c.commands_post(raw_cmd, base_url=base_url)
                cmd_col2.json(out)
            except Exception as exc:
                cmd_col2.error(f'Command error: {exc}')

# ---------------------------------------------------------------------------
# TAB 4: Export Bundle (.cyjs, .cx2, .graphml, .sif, .cycmd, Polars CSVs)
# ---------------------------------------------------------------------------
with tab_export:
    st.subheader('Export Subnetwork & Cytoscape Automation Scripts')
    active_sub = st.session_state.get('cy_subnetwork')
    if not active_sub:
        st.info('Build a subnetwork in Tab 1 first.')
    else:
        nodes_df = active_sub['nodes']
        edges_df = active_sub['edges']
        title = active_sub['title']

        st.write(
            f'Current subnetwork **{title}** has **{nodes_df.height:,} nodes** and **{edges_df.height:,} edges** '
            'with full moonlighting metadata, 1-hop linker hypergeometric q-values, shared triangle counts, '
            'and Augmented Forman–Ricci curvature.'
        )

        extra_tables = {}
        if 'cm_result' in st.session_state:
            cm_r = st.session_state['cm_result']
            if isinstance(cm_r.get('cluster_enrichment'), pl.DataFrame) and len(cm_r['cluster_enrichment']):
                extra_tables['clustermaker2_enrichment'] = cm_r['cluster_enrichment']
            if isinstance(cm_r.get('node_cluster_bridging'), pl.DataFrame) and len(cm_r['node_cluster_bridging']):
                extra_tables['clustermaker2_bridging'] = cm_r['node_cluster_bridging']

        bundle_bytes = export_cytoscape_bundle_zip(
            nodes_df,
            edges_df,
            title=title,
            extra_tables=extra_tables,
            metadata=active_sub['summary'],
        )
        st.download_button(
            'Download Complete Cytoscape Bundle (.ZIP: .cyjs, .cx2, .graphml, .sif, .cycmd, nodes.csv, edges.csv)',
            bundle_bytes,
            f'{title}_cytoscape_bundle.zip',
            'application/zip',
            type='primary',
            key='dl_cy_zip',
        )

        e1, e2, e3, e4 = st.columns(4)
        e1.download_button(
            'Download Cytoscape.js (.cyjs)',
            json.dumps(to_cytoscapejs_json(nodes_df, edges_df, title=title), indent=2),
            f'{title}.cyjs',
            'application/json',
            key='dl_cyjs',
        )
        e2.download_button(
            'Download Cytoscape CX2 (.cx2)',
            export_cx2_json(nodes_df, edges_df, title=title),
            f'{title}.cx2',
            'application/json',
            key='dl_cx2',
        )
        e3.download_button(
            'Download GraphML (.graphml)',
            export_graphml_string(nodes_df, edges_df),
            f'{title}.graphml',
            'application/xml',
            key='dl_graphml',
        )
        e4.download_button(
            'Download SIF (.sif)',
            export_sif_string(edges_df),
            f'{title}.sif',
            'text/plain',
            key='dl_sif',
        )

        cycmd_script = build_cycmd_script(network_name=title, cluster_algorithm='leiden', layout_name='force-directed')
        with st.expander('View / Download Cytoscape Command File (`.cycmd` for Tools → Execute Command File…)', expanded=True):
            st.code(cycmd_script, language='bash')
            st.download_button(
                'Download .cycmd Script',
                cycmd_script,
                f'{title}_automation.cycmd',
                'text/plain',
                key='dl_cycmd',
            )
