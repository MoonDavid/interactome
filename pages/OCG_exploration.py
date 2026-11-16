"""Dedicated exploration workspace for C OCG overlapping protein-interaction clusters."""
from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from clustering_explorer import ROOT, annotate_run
from cytoscape_bridge import DEFAULT_BASE_URL, send_to_cytoscape
from moonlighting import ADVANCED_EDGE_LIMIT, GROUP_LABELS, SOURCES, cohort_ids, load_moon_data
from network_view import static_image_bytes
from networks import discover_datasets, load_network
from ocg_exploration import (
    cross_network_ocg_comparison,
    degree_matched_multi_membership,
    ego_cluster_subgraph,
    load_ocg_memberships,
    membership_count_distribution,
    saved_ocg_runs,
)
from ocg_visualization import (
    cytoscape_overlap_graphml,
    ego_cluster_3d_figure,
    ego_cluster_bipartite_figure,
    ego_cluster_cytoscapejs_html,
    ego_cluster_echarts_option,
    ego_cluster_static_figure,
    global_overlap_figures,
    ocg_3d_figure,
    ocg_cytoscapejs_html,
    ocg_echarts_option,
    ocg_pyvis_html,
    ocg_sigma_html,
    ocg_static_overlap_figure,
    ocg_static_upset_figure,
    prepare_ocg_overlap_display,
)

st.title('OCG Exploration')
st.caption(
    'Explore Overlapping Cluster Generator (c.ocg) results: global community overview, membership counts, '
    'Cytoscape export, degree-matched '
    'multi-membership, cluster-overlap meta-networks, '
    'cross-interactome conservation, and ego-cluster neighborhoods.'
)

st.info(
    'c.ocg assigns proteins to zero, one, or multiple overlapping clusters based on undirected '
    'binary topology. Because hub proteins naturally participate in more clusters, degree-matched '
    'controls help distinguish overlapping memberships '
    'from degree effects. Topology alone does not validate moonlighting or binding exclusivity.'
)

ocg_runs = saved_ocg_runs()
if ocg_runs.is_empty():
    st.warning('No saved c.ocg runs found. Compute a c.ocg run on the Clustering and communities page first.')
    st.stop()

datasets = discover_datasets(ROOT / 'data')
available_networks = sorted(set(ocg_runs['network'].to_list()) & set(datasets))
if not available_networks:
    st.warning('Saved c.ocg runs exist, but their canonical network Parquet files were not found under data/.')
    st.stop()

c_net, c_run, c_set = st.columns([1.2, 1.4, 1.2])
network = c_net.selectbox('Network', available_networks, key='ocg_exp_network')
net_runs = ocg_runs.filter(pl.col('network') == network).sort('run', descending=True)
run_labels = {
    row['directory']: (
        f"{row['run']} · {row['communities']} clusters · {row['overlapping_nodes']} overlapping "
        f"(init={row['init_class_sys']}, cent={row['cent_class_sys']}, min={row['min_class']}, max={row['max_class_card']})"
    )
    for row in net_runs.iter_rows(named=True)
}
directory = c_run.selectbox(
    'c.ocg run version',
    net_runs['directory'].to_list(),
    format_func=lambda d: run_labels.get(d, d),
    key=f'ocg_exp_run_{network}',
)
protein_set_key = c_set.selectbox(
    'Protein set',
    SOURCES,
    index=SOURCES.index('moonlitdb') if 'moonlitdb' in SOURCES else 0,
    format_func=lambda s: GROUP_LABELS.get(s, s),
    key='ocg_exp_protein_set',
)
protein_set_label = GROUP_LABELS.get(protein_set_key, protein_set_key)

memberships, protein_summary = load_ocg_memberships(directory)
cohort, _ = cohort_ids(load_moon_data(ROOT / 'data'), [protein_set_key])

run_row = net_runs.filter(pl.col('directory') == directory).row(0, named=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric('Clusters', f"{run_row['communities']:,}")
m2.metric('Covered proteins', f"{run_row['covered_nodes']:,}")
m3.metric('Unassigned (k=0)', f"{run_row['unassigned_nodes']:,}")
m4.metric('Overlapping (k≥2)', f"{run_row['overlapping_nodes']:,}")
max_k = int(protein_summary['memberships'].max() or 0)
m5.metric('Max memberships (k)', f'{max_k}')

# File identities invalidate previous results when a run or network is replaced.
input_id = (
    directory, protein_set_key,
    (Path(directory) / 'memberships.parquet').stat().st_mtime_ns,
    Path(datasets[network]).stat().st_mtime_ns,
)


def run_result(key, parameters, label, compute):
    """Separate result slots keep inactive views idle and prevent stale displays."""
    identity = (input_id, parameters)
    if st.button(label, type='primary', key=f'{key}_run'):
        with st.spinner('Preparing this view…'):
            value = compute()
        st.session_state[key] = {'id': identity, 'value': value}
    cached = st.session_state.get(key)
    if cached and cached['id'] == identity:
        return cached['value']
    st.info('Run this view to display results for the current inputs. Other views run independently.')
    return None


st.caption('Each view runs independently. Results stay available when switching views; changed inputs require a new run.')
view = st.segmented_control(
    'Exploration view',
    [
        'Global overview',
        'Annotation and interactive network',
        'Membership counts',
        'Degree-matched overlap',
        'Cross-interactome',
        'Ego-cluster inspector',
    ],
    default='Global overview',
    required=True,
    key='ocg_exp_view',
)

@st.fragment
def functional_cluster_labels():
    """Reuse the persisted full-run GO/Reactome bundle and its input fingerprints."""
    try:
        from ocg_annotation_ui import available_annotation_bundle
        with st.spinner('Preparing cached GO and Reactome labels for all clusters…'):
            bundle, _ = available_annotation_bundle(directory, None)
            if bundle is None:
                bundle = annotate_run(directory, text_annotations=False)
        return pl.read_csv(bundle / 'label_summary.csv', schema_overrides={'community': pl.String})
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        st.warning(f'Functional labels unavailable: {exc}')
        return None


def render_exploration():
    if view == 'Annotation and interactive network':
        from ocg_annotation_ui import annotation_workspace
        annotation_workspace(directory, memberships, input_id, run_row.get('edges') or 0,
                             cohort, protein_set_label, datasets[network])

    elif view == 'Global overview':
        st.subheader(f'Global community overview ({network})')
        if (run_row.get('edges') or 0) > ADVANCED_EDGE_LIMIT:
            st.warning(f'This run exceeds the {ADVANCED_EDGE_LIMIT:,}-edge guard for advanced graph layouts.')
            return
        st.info(
            'Each node represents a whole OCG cluster. Larger nodes contain more proteins; '
            'color shows the number of overlapping clusters in the full run. Wider edges mean more shared proteins. '
            'Jaccard divides shared proteins by the union; the overlap coefficient divides by the smaller cluster '
            'and highlights nested complexes. Layout distance is schematic, not biological distance. '
            'These are descriptive similarities; no significance test or stars apply.'
        )
        a, b = st.columns(2)
        cap = a.slider('Maximum clusters to display (largest first)', 10, 300, 150, step=10,
                       key='ocg_global_cap', persist_state='session')
        metric = b.selectbox('Overlap similarity', ['jaccard', 'overlap_coef'],
            format_func=lambda v: {'jaccard': 'Jaccard', 'overlap_coef': 'Overlap coefficient'}[v],
            key='ocg_global_metric', persist_state='session')
        c, d = st.columns(2)
        minimum = c.number_input('Minimum shared proteins (graph edges)', min_value=1, value=1,
                                 key='ocg_global_shared', persist_state='session')
        similarity = d.slider('Minimum similarity (graph edges)', 0.0, 1.0, 0.1, 0.05,
                              key='ocg_global_similarity', persist_state='session')
        result = run_result('ocg_global_result', (cap, metric, minimum, similarity), 'Run global overview',
            lambda: global_overlap_figures(memberships, max_clusters=cap, metric=metric,
                                          min_shared=int(minimum), min_similarity=similarity))
        if result is None:
            return
        graph, matrix, clusters, pairs, stats = result
        if not stats['total_clusters']:
            st.info('No assigned clusters in this run.')
            return
        st.caption(
            f"Showing {stats['displayed_clusters']:,} of {stats['total_clusters']:,} clusters (largest first), "
            f"including {stats['isolated_display_nodes']:,} with no drawn edges. "
            f"Drawing {stats['drawn_edges']:,} of {stats['qualifying_edges']:,} qualifying edges among shown clusters "
            '(at most 2,500, ranked by shared proteins). Color statistics use the full run before display filtering.'
        )
        framework = st.segmented_control(
            'Visualization framework',
            [
                'Plotly 2D + Matrix',
                'Cytoscape.js',
                'Apache ECharts',
                'Sigma.js WebGL',
                'PyVis',
                '3D Explorer',
                'Static (Matplotlib)',
                'UpSet Intersection Plot',
            ],
            default='Plotly 2D + Matrix',
            required=True,
            key='ocg_global_framework',
        )
        if framework == 'Plotly 2D + Matrix':
            st.plotly_chart(graph, width='stretch', alt='Global overview of overlapping OCG communities')
            st.plotly_chart(matrix, width='stretch', alt='Hierarchically ordered community overlap matrix')
            st.caption('The matrix uses all overlaps among displayed clusters, without graph-edge thresholds. '
                       'Rows and columns share average-linkage ordering of 1 − similarity; diagonal = 1. '
                       'Hover for cluster IDs and exact values. Functional labels are not inferred from topology.')
        elif framework == 'Cytoscape.js':
            c1, c2 = st.columns(2)
            cy_mode = c1.segmented_control(
                'Graph representation',
                ['cluster_overlap', 'bipartite_bridges'],
                default='cluster_overlap',
                format_func=lambda m: 'Cluster overlap network' if m == 'cluster_overlap' else 'Bipartite (clusters + bridge proteins)',
                required=True,
                key='ocg_cy_mode',
            )
            cy_layout = c2.selectbox('Initial Cytoscape.js layout', ['cose', 'concentric', 'circle', 'breadthfirst', 'grid'], key='ocg_cy_layout')
            cy_html = ocg_cytoscapejs_html(
                memberships, max_clusters=cap, min_shared=int(minimum), min_similarity=similarity,
                metric=metric, cohort=cohort, mode=cy_mode, layout_name=cy_layout, height=640,
            )
            st.iframe(cy_html, height=655, alt='Interactive Cytoscape.js OCG cluster viewer')
            st.download_button('Download Cytoscape.js HTML', cy_html, f'{network}_ocg_cytoscapejs.html',
                               'text/html', key='ocg_cy_html_dl', on_click='ignore')
        elif framework == 'Apache ECharts':
            c1, c2 = st.columns(2)
            ec_layout = c1.segmented_control(
                'ECharts layout', ['force', 'circular'], default='force',
                format_func=lambda v: 'Force' if v == 'force' else 'Circular chord',
                required=True, key='ocg_ec_layout',
            )
            ec_bridges = c2.checkbox('Include multi-cluster bridge proteins (bipartite)', value=False, key='ocg_ec_bridges')
            ec_option, _, _, _ = ocg_echarts_option(
                memberships, max_clusters=cap, min_shared=int(minimum), min_similarity=similarity,
                metric=metric, cohort=cohort, layout=ec_layout, include_bridges=ec_bridges,
            )
            st.echarts_chart(ec_option, height=640, alt='Interactive Apache ECharts OCG cluster viewer')
        elif framework == 'Sigma.js WebGL':
            sigma_engines = (
                ['NetworkX Spring', 'NetworkX Kamada–Kawai', 'Circular']
                + (['igraph FR', 'igraph DrL'] if find_spec('igraph') else [])
            )
            sigma_engine = st.selectbox('Fixed Python layout engine', sigma_engines, key='ocg_sigma_engine')
            sigma_html = ocg_sigma_html(
                memberships, max_clusters=cap, min_shared=int(minimum), min_similarity=similarity,
                metric=metric, cohort=cohort, engine=sigma_engine,
                title=f'{network} · OCG clusters ({sigma_engine})',
            )
            st.iframe(sigma_html, height=700, alt='Offline Sigma.js WebGL OCG cluster viewer')
            st.download_button('Download interactive Sigma HTML', sigma_html, f'{network}_ocg_sigma.html',
                               'text/html', key='ocg_sigma_dl', on_click='ignore')
        elif framework == 'PyVis':
            if not find_spec('pyvis'):
                st.info('PyVis is unavailable in this environment. Install pyvis>=0.3.2,<0.4 to enable it.')
            else:
                p1, p2 = st.columns(2)
                pv_physics = p1.toggle('Enable Barnes–Hut physics simulation', value=True, key='ocg_pyvis_physics')
                pv_labels = p2.checkbox('Show cluster labels', value=cap <= 120, key='ocg_pyvis_labels')
                pv_html = ocg_pyvis_html(
                    memberships, max_clusters=cap, min_shared=int(minimum), min_similarity=similarity,
                    metric=metric, cohort=cohort, physics=pv_physics, labels=pv_labels,
                )
                st.iframe(pv_html, height=680, alt='Interactive PyVis OCG cluster viewer')
                st.download_button('Download PyVis HTML', pv_html, f'{network}_ocg_pyvis.html',
                                   'text/html', key='ocg_pyvis_dl', on_click='ignore')
        elif framework == '3D Explorer':
            fig_3d = ocg_3d_figure(
                memberships, max_clusters=cap, min_shared=int(minimum), min_similarity=similarity,
                metric=metric, cohort=cohort, protein_set_label=protein_set_label,
            )
            st.plotly_chart(fig_3d, width='stretch', alt='3D Plotly OCG community overlap network')
        elif framework == 'Static (Matplotlib)':
            s1, s2, s3 = st.columns(3)
            st_engines = ['NetworkX'] + (['igraph'] if find_spec('igraph') else []) + ['Circular']
            st_engine = s1.selectbox('Static layout engine', st_engines, key='ocg_static_engine')
            st_layouts = (['FR', 'DrL', 'Circular'] if st_engine == 'igraph'
                          else (['Circular'] if st_engine == 'Circular' else ['Spring', 'Kamada–Kawai', 'Circular']))
            st_layout = s2.selectbox('Layout algorithm', st_layouts, key='ocg_static_layout')
            st_labels = s3.checkbox('Annotate major clusters', value=True, key='ocg_static_labels')
            with st.spinner('Rendering static OCG overview…'):
                static_fig = ocg_static_overlap_figure(
                    memberships, max_clusters=cap, min_shared=int(minimum), min_similarity=similarity,
                    metric=metric, cohort=cohort, protein_set_label=protein_set_label,
                    engine=st_engine, layout=st_layout, labels=st_labels,
                )
                st.image(static_image_bytes(static_fig, 'png'), width='stretch',
                         alt='Static two-panel OCG overlap network and similarity matrix')
            d1, d2, d3 = st.columns(3)
            for col, fmt, mime in [(d1, 'png', 'image/png'), (d2, 'svg', 'image/svg+xml'), (d3, 'pdf', 'application/pdf')]:
                col.download_button(f'Download {fmt.upper()}', data=lambda fmt=fmt: static_image_bytes(static_fig, fmt),
                                    file_name=f'{network}_ocg_overview.{fmt}', mime=mime,
                                    key=f'ocg_static_dl_{fmt}', on_click='ignore')
        else:
            top_n_pairs = st.slider('Top overlapping cluster pairs', 5, 40, 25, step=5, key='ocg_upset_pairs')
            with st.spinner('Rendering UpSet cluster-intersection plot…'):
                upset_fig = ocg_static_upset_figure(
                    memberships, top_pairs=int(top_n_pairs), metric=metric,
                    cohort=cohort, protein_set_label=protein_set_label,
                )
                st.image(static_image_bytes(upset_fig, 'png'), width='stretch',
                         alt='UpSet-style OCG cluster intersection plot')
            u1, u2, u3 = st.columns(3)
            for col, fmt, mime in [(u1, 'png', 'image/png'), (u2, 'svg', 'image/svg+xml'), (u3, 'pdf', 'application/pdf')]:
                col.download_button(f'Download UpSet {fmt.upper()}', data=lambda fmt=fmt: static_image_bytes(upset_fig, fmt),
                                    file_name=f'{network}_ocg_upset.{fmt}', mime=mime,
                                    key=f'ocg_upset_dl_{fmt}', on_click='ignore')

        st.dataframe(clusters, hide_index=True, width='stretch')
        st.download_button('Download displayed overlap pairs', pairs.write_csv(), 'ocg_overlap_pairs.csv',
                           'text/csv', key='ocg_global_csv')
        st.markdown('**Open the full community network in Cytoscape**')
        st.caption('Export includes every cluster, including isolated clusters, and all overlap edges meeting '
                   'the current shared-protein and similarity thresholds. Browser node and edge caps do not apply. '
                   'Attributes include member UniProt accessions, cluster size and overlap scores.')
        graphml = run_result('ocg_cytoscape_result', (metric, minimum, similarity),
            'Prepare Cytoscape export', lambda: cytoscape_overlap_graphml(
                memberships, metric=metric, min_shared=int(minimum), min_similarity=similarity))
        if graphml is not None:
            st.download_button('Download Cytoscape network (.graphml)', graphml,
                f'{network}_ocg_communities.graphml', 'application/graphml+xml', key='ocg_cytoscape_download')
            st.caption('In Cytoscape, use File → Import → Network from File and choose this GraphML file. '
                       'Apply a layout, map node size to protein_count, node color to full_run_overlap_neighbors, '
                       'and edge width to shared_proteins. Each node represents a cluster; '
                       'its proteins attribute lists the member accessions.')
            with st.expander('Send displayed OCG network to Cytoscape Desktop (CyREST)'):
                cy_url = st.text_input('Cytoscape CyREST URL', value=DEFAULT_BASE_URL, key='ocg_cyrest_url')
                if st.button('Send displayed OCG clusters to Cytoscape Desktop', key='ocg_send_cyrest'):
                    try:
                        with st.spinner('Sending OCG community overlap network to Cytoscape Desktop…'):
                            cy_nodes_raw, cy_edges_raw, _, _ = prepare_ocg_overlap_display(
                                memberships, max_clusters=cap, min_shared=int(minimum),
                                min_similarity=similarity, metric=metric, cohort=cohort,
                            )
                            cy_nodes = cy_nodes_raw.select(
                                pl.format('C{}', pl.col('community')).alias('id'),
                                pl.format('C{}', pl.col('community')).alias('name'),
                                pl.format('C{}', pl.col('community')).alias('protein'),
                                pl.format('C{}', pl.col('community')).alias('label'),
                                pl.when(pl.col('cohort_hits') > 0).then(pl.lit('Moonlighting')).otherwise(pl.lit('Background')).alias('node_role'),
                                pl.col('size').alias('subnetwork_degree'),
                                'community', 'size', 'overlap_neighbors', 'total_shared_links', 'cohort_hits', 'cohort_fraction',
                            )
                            cy_edges = cy_edges_raw.select(
                                pl.format('C{}', pl.col('community_a')).alias('source'),
                                pl.format('C{}', pl.col('community_b')).alias('target'),
                                pl.lit('shares_proteins').alias('interaction'),
                                pl.lit('Shared').alias('edge_class'),
                                pl.col(metric).alias('score'),
                                'shared_proteins', 'jaccard', 'overlap_coef',
                            )
                            sent = send_to_cytoscape(cy_nodes, cy_edges, title=f'{network} OCG clusters', base_url=cy_url)
                        st.success(f"Sent {sent['nodes_sent']} OCG clusters and {sent['edges_sent']} overlap edges (SUID {sent['network_suid']}).")
                    except Exception as exc:
                        st.error(f'Could not send to Cytoscape Desktop: {exc}')

    elif view == 'Membership counts':
        st.subheader(f'Proteins by number of OCG clusters ({network})')
        st.info('k is the exact number of distinct clusters containing a protein: 0 = unassigned, '
                '1 = one cluster, and 2 or more = overlapping membership. Each protein is counted once. '
                f'The selected protein set is {protein_set_label}; counts include only proteins recorded in this run. '
                'These are descriptive counts, so no significance test or stars apply.')
        counts = run_result('ocg_counts_result', (), 'Run membership counts',
                            lambda: membership_count_distribution(protein_summary, cohort))
        if counts is None:
            return
        only_overlap = st.checkbox('Show only overlapping proteins (k ≥ 2)', value=False,
                                  key='ocg_counts_multi', persist_state='session')
        show_groups = st.checkbox(f'Compare {protein_set_label} with other run proteins', value=False,
                                 key='ocg_counts_groups', persist_state='session')
        percent = st.checkbox('Show percentage within each group', value=False,
                             key='ocg_counts_percent', persist_state='session')
        shown = counts.filter(pl.col('memberships') >= 2) if only_overlap else counts
        shown = shown.filter(pl.col('group') != 'All run proteins') if show_groups else shown.filter(
            pl.col('group') == 'All run proteins')
        fig = go.Figure()
        for group in shown['group'].unique(maintain_order=True):
            data = shown.filter(pl.col('group') == group)
            fig.add_bar(x=data['memberships'].to_list(),
                y=(data['fraction'] * 100 if percent else data['proteins']).to_list(),
                name=protein_set_label if group == 'Selected protein set' else group,
                customdata=data.select('proteins', 'population').rows(),
                text=data['proteins'].to_list() if not percent and data.height <= 30 else None,
                textposition='outside', cliponaxis=False,
                hovertemplate='Clusters (k): %{x}<br>Proteins: %{customdata[0]}'
                              '<br>Group population: %{customdata[1]}<br>Value: %{y}<extra>%{fullData.name}</extra>')
        fig.update_layout(title='Exact cluster-membership counts', barmode='group', height=460,
            xaxis=dict(title='Number of clusters per protein (k)', tickmode='linear',
                       dtick=max(1, (max_k + 24) // 25)),
            yaxis=dict(title='Group proteins (%)' if percent else 'Number of proteins', rangemode='tozero'),
            margin=dict(l=65, r=25, t=65, b=65))
        st.plotly_chart(fig, width='stretch', alt='Number of proteins at each exact cluster membership count')
        st.caption('Percentages use the full group population, including k = 0 and k = 1, '
                   'even when only overlapping proteins are shown. Empty groups have unavailable percentages.')
        st.dataframe(shown, hide_index=True, width='stretch')
        st.download_button('Download all membership counts', counts.write_csv(), 'ocg_membership_counts.csv',
                           'text/csv', key='ocg_counts_csv')

    elif view == 'Degree-matched overlap':
        st.subheader(f'Degree-matched multi-membership test ({protein_set_label} vs. {network} background)')
        st.caption(
            f'Compares the fraction of multi-cluster proteins (k ≥ 2) and mean cluster memberships (k) in '
            f'{protein_set_label} against degree-bin-matched (floor(log2 degree)) random draws from the '
            'non-cohort network background using +1 empirical permutation p-values and BH adjustment across '
            'the two summary tests (and separately across per-bin Fisher exact tests). '
            '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable.'
        )
        labels = functional_cluster_labels()
        if labels is not None:
            with st.expander('Cached GO and Reactome labels · all clusters'):
                st.dataframe(labels, hide_index=True, width='stretch')
        distribution = membership_count_distribution(protein_summary, cohort).sort('group', 'memberships')
        distribution = distribution.with_columns(
            pl.col('fraction').cum_sum().over('group').alias('cumulative_fraction'))
        st.caption('Exact and cumulative k distributions compare all run proteins, the selected protein set, '
                   'and other run proteins. These descriptive plots are not degree-matched tests; no test applies.')
        for value, title in [('fraction', 'Exact membership distribution'),
                             ('cumulative_fraction', 'Cumulative membership distribution')]:
            chart = px.line(distribution.to_dict(as_series=False), x='memberships', y=value,
                            color='group', markers=True, title=title)
            st.plotly_chart(chart, width='stretch')
        n_perm = st.slider('Permutations', min_value=200, max_value=2000, value=1000, step=200, key='ocg_deg_perm', persist_state="session")
        if st.button('Run degree-matched permutation test', type='primary', key='ocg_deg_run'):
            loaded = load_network(datasets[network])
            with st.spinner('Running degree-bin-matched permutations…'):
                summary, bin_df = degree_matched_multi_membership(
                    protein_summary, loaded.edges, cohort, n_perm=int(n_perm), seed=42
                )
            st.session_state['ocg_deg_result'] = {
                'id': (input_id, int(n_perm)),
                'summary': summary,
                'bin_df': bin_df,
            }

        cached = st.session_state.get('ocg_deg_result')
        if cached and cached['id'] == (input_id, int(n_perm)):
            summary = cached['summary']
            bin_df = cached['bin_df']
            if not summary.get('valid'):
                st.warning(summary.get('reason', 'Could not compute degree-matched test.'))
            else:
                rows = summary.get('membership_distribution', [])
                if rows:
                    fig = go.Figure()
                    fig.add_scatter(x=[r['k'] for r in rows], y=[r['null_high'] for r in rows],
                                    mode='lines', line=dict(width=0), showlegend=False)
                    fig.add_scatter(x=[r['k'] for r in rows], y=[r['null_low'] for r in rows],
                                    mode='lines', line=dict(width=0), fill='tonexty', name='Null 95% interval')
                    for column, name in [('observed', protein_set_label), ('null_mean', 'Degree-matched null')]:
                        fig.add_scatter(x=[r['k'] for r in rows], y=[r[column] for r in rows],
                                        mode='lines+markers', name=name)
                    fig.update_layout(title='Complete degree-matched membership distribution',
                                      xaxis_title='Cluster memberships (k)', yaxis_title='Protein fraction')
                    st.plotly_chart(fig, width='stretch')
                    st.caption('Pointwise 95% permutation intervals are descriptive, not simultaneous confidence '
                               'bands or per-k significance tests. Observed and null use the same matched degree bins.')
                s1, s2, s3, s4 = st.columns(4)
                s1.metric(
                    f'{protein_set_label} k≥2',
                    f"{summary['obs_multi_fraction']:.1%}",
                    delta=f"Matched null {summary['matched_null_multi_mean']:.1%} ({summary['multi_significance']})",
                )
                s2.metric('Multi-cluster q-value', f"{summary['multi_qvalue']:.4g}")
                s3.metric(
                    f'{protein_set_label} mean k',
                    f"{summary['obs_mean_k']:.2f}",
                    delta=f"Matched null {summary['matched_null_k_mean']:.2f} ({summary['k_significance']})",
                )
                s4.metric('Mean-k q-value', f"{summary['k_qvalue']:.4g}")

                if not bin_df.is_empty():
                    melted = (
                        bin_df.select(
                            'degree_range',
                            'significance',
                            pl.col('cohort_multi_fraction').alias(protein_set_label),
                            pl.col('background_multi_fraction').alias('Network background'),
                        )
                        .unpivot(
                            index=['degree_range', 'significance'],
                            variable_name='Group',
                            value_name='Multi-cluster fraction (k ≥ 2)',
                        )
                    )
                    fig_bin = px.bar(
                        melted.to_pandas(),
                        x='degree_range',
                        y='Multi-cluster fraction (k ≥ 2)',
                        color='Group',
                        barmode='group',
                        title=f'Multi-cluster fraction (k ≥ 2) by degree bin ({network})',
                        labels={'degree_range': 'Degree bin (2^b to 2^(b+1)-1)'},
                    )
                    # Annotate significance stars above each bin.
                    max_y = bin_df.select(
                        pl.max_horizontal('cohort_multi_fraction', 'background_multi_fraction').alias('m')
                    )['m'].to_list()
                    for row, ymax in zip(bin_df.iter_rows(named=True), max_y):
                        fig_bin.add_annotation(
                            x=row['degree_range'],
                            y=float(ymax) + 0.03,
                            text=row['significance'],
                            showarrow=False,
                            font=dict(size=12),
                        )
                    fig_bin.update_layout(height=400, yaxis_range=[0, min(1.12, max(max_y, default=0.5) + 0.14)])
                    st.plotly_chart(fig_bin, width='stretch')
                    st.dataframe(bin_df, hide_index=True, width='stretch')

    elif view == 'Cross-interactome':
        st.subheader(f'Cross-interactome c.ocg conservation ({protein_set_label})')
        st.caption(
            'Checks whether multi-cluster membership (k ≥ 2) is conserved across all networks with saved c.ocg runs. '
            'Enrichment of the selected protein set at each conservation threshold uses one-sided hypergeometric tests '
            'BH-adjusted across thresholds. * q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise.'
        )
        cross_inputs = tuple((r['directory'], (Path(r['directory']) / 'memberships.parquet').stat().st_mtime_ns)
                             for r in ocg_runs.iter_rows(named=True))
        result = run_result('ocg_cross_result', cross_inputs, 'Run cross-interactome comparison',
                            lambda: cross_network_ocg_comparison(ocg_runs, cohort))
        if result is None:
            return
        conserved_df, enrich_df = result
        if conserved_df.is_empty():
            st.info('No c.ocg runs available for cross-interactome comparison.')
        else:
            if not enrich_df.is_empty():
                st.markdown('**Protein set enrichment by number of networks where protein is multi-clustered (k ≥ 2)**')
                st.dataframe(enrich_df, hide_index=True, width='stretch')
            st.markdown('**Most conserved multi-cluster proteins**')
            only_cohort = st.checkbox(f'Show only {protein_set_label} proteins', value=False, key='ocg_cross_only_cohort', persist_state="session")
            display_df = conserved_df.filter(pl.col('in_cohort')) if only_cohort else conserved_df
            st.dataframe(display_df.head(500), hide_index=True, width='stretch')

    elif view == 'Ego-cluster inspector':
        st.subheader(f'Ego-cluster neighborhood inspector ({network})')
        st.caption(
            'Inspect the 1-hop neighborhood of any protein, colored by number of OCG cluster memberships. '
            'Edges connecting partners that share at least one OCG cluster are highlighted.'
        )
        loaded = load_network(datasets[network])
        if loaded.edges.height > ADVANCED_EDGE_LIMIT:
            st.warning(f'{network} exceeds the {ADVANCED_EDGE_LIMIT:,}-edge guard for interactive graph layout.')
        else:
            only_multi_cohort = st.checkbox(
                f'Filter candidate list to {protein_set_label} proteins with k ≥ 2',
                value=False,
                key='ocg_ego_filter', persist_state="session",
            )
            candidates = protein_summary
            if only_multi_cohort:
                candidates = candidates.filter(pl.col('is_multi') & pl.col('protein').is_in(cohort))
            cand_list = candidates['protein'].unique().sort().to_list()
            if not cand_list:
                st.info('No proteins match the candidate filter. Turn it off to inspect all proteins in this run.')
                return
            from network_graph_utils import protein_identity
            from ocg_annotation_ui import local_protein_names, ego_protein_label, ego_protein_description_panel, ego_membership_panel
            from html import escape
            metadata = local_protein_names(ROOT / 'data')
            names = {p: protein_identity(p, metadata) for p in cand_list}
            selected_protein = st.selectbox('Center protein (gene or protein name)', cand_list,
                format_func=lambda p: ego_protein_label(p, names[p], metadata),
                key='ocg_ego_protein', persist_state="session")
            ego_protein_description_panel(selected_protein, ROOT / 'data')
            ego_membership_panel(selected_protein, memberships, directory)
            st.caption('Labels use local gene names, then protein names; UniProt is retained in hover and '
                       'as a fallback when no name is available. Nodes keep their exact accession identities. '
                       f'All {len(cand_list):,} matching proteins are searchable; proteins with one OCG membership '
                       'are included when the candidate filter is off.')
            max_nb = st.slider('Max neighbors to display', 10, 120, 50, step=10, key='ocg_ego_max_nb', persist_state="session")
            from ocg_ego_comparison import cross_ego_panel
            cross_ego_panel(selected_protein, network, directory, ocg_runs, datasets, ROOT / 'data', metadata)

            result = run_result('ocg_ego_result', (selected_protein, int(max_nb), only_multi_cohort),
                'Run ego-cluster inspector', lambda: ego_cluster_subgraph(
                    selected_protein, memberships, loaded.edges, max_neighbors=int(max_nb)))
            if result is None:
                return
            nodes_df, sub_edges_df = result
            labels = functional_cluster_labels()
            local_memberships = memberships.filter(pl.col('protein').is_in(nodes_df['protein'].implode()))
            cluster_ids = local_memberships['community'].unique().sort().to_list()
            label_lookup = {} if labels is None else {
                str(r['community']): ' · '.join(str(v) for k, v in r.items()
                    if k.endswith('_enrichment') and v) for r in labels.iter_rows(named=True)}
            selected_cluster = st.selectbox('Explore neighborhood cluster', cluster_ids,
                format_func=lambda c: f'{c} · {label_lookup.get(str(c)) or "No significant functional label"}',
                key='ocg_ego_cluster')
            if labels is not None:
                st.dataframe(labels.filter(pl.col('community') == str(selected_cluster)),
                             hide_index=True, width='stretch')
            st.dataframe(local_memberships.filter(pl.col('community') == selected_cluster),
                         hide_index=True, width='stretch')
            st.caption('Cluster labels use full-run GO/Reactome enrichment and its complete correction family. '
                       'The member table contains proteins visible in this ego network.')
            node_meta = {r['protein']: r for r in nodes_df.iter_rows(named=True)}
            identities = {p: protein_identity(p, metadata) for p in node_meta}

            ego_viewer = st.segmented_control(
                'Neighborhood viewer',
                [
                    'Overlapping cluster hulls',
                    'Plotly 2D',
                    'Bipartite Star (Proteins + OCG Clusters)',
                    'Cytoscape.js',
                    'Apache ECharts',
                    '3D',
                    'Static (Matplotlib)',
                ],
                default='Overlapping cluster hulls',
                required=True,
                key='ocg_ego_viewer',
            )
            if ego_viewer == 'Overlapping cluster hulls':
                from ocg_ego_overlap import overlap_ego_panel
                overlap_ego_panel(nodes_df, sub_edges_df, memberships, selected_protein,
                                  {p: identities[p]['label'] for p in identities}, labels, network)
            elif ego_viewer == 'Plotly 2D':
                g = nx.Graph()
                for p in node_meta:
                    g.add_node(p)
                for r in sub_edges_df.iter_rows(named=True):
                    g.add_edge(r['a'], r['b'], shared=int(r['shared_clusters']))

                pos = nx.spring_layout(g, seed=42)
                shared_x, shared_y, cross_x, cross_y = [], [], [], []
                for u, v, d in g.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    if d.get('shared', 0) > 0:
                        shared_x.extend([x0, x1, None])
                        shared_y.extend([y0, y1, None])
                    else:
                        cross_x.extend([x0, x1, None])
                        cross_y.extend([y0, y1, None])

                fig_ego = go.Figure()
                if cross_x:
                    fig_ego.add_trace(go.Scatter(x=cross_x, y=cross_y, mode='lines', name='No shared OCG cluster',
                                                 line=dict(width=0.9, color='#cbd5e1', dash='dot'), hoverinfo='none'))
                if shared_x:
                    fig_ego.add_trace(go.Scatter(x=shared_x, y=shared_y, mode='lines', name='Shared OCG cluster(s)',
                                                 line=dict(width=1.5, color='#3b82f6'), hoverinfo='none'))

                fig_ego.add_trace(go.Scatter(
                    x=[pos[p][0] for p in g.nodes()],
                    y=[pos[p][1] for p in g.nodes()],
                    mode='markers+text',
                    text=[escape(identities[p]['label']) for p in g.nodes()],
                    textposition='top center',
                    hovertext=[
                        escape(identities[p]['tooltip']).replace('\n', '<br>') +
                        f"<br>k={node_meta[p]['memberships']} | clusters={node_meta[p]['communities_str']}"
                        for p in g.nodes()
                    ],
                    hoverinfo='text',
                    name='Proteins',
                    marker=dict(
                        size=[20 if node_meta[p]['is_center'] else 11 for p in g.nodes()],
                        color=[node_meta[p]['memberships'] for p in g.nodes()],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(title='Clusters (k)'),
                        line=dict(width=[2.5 if node_meta[p]['is_center'] else 1 for p in g.nodes()], color='#111827'),
                    ),
                ))
                fig_ego.update_layout(
                    title=f'1-hop OCG ego network for {escape(identities[selected_protein]["label"])} (k={node_meta[selected_protein]["memberships"]})',
                    height=480,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=15, r=15, t=45, b=15),
                )
                st.plotly_chart(fig_ego, width='stretch')
            elif ego_viewer == 'Bipartite Star (Proteins + OCG Clusters)':
                fig_bip = ego_cluster_bipartite_figure(
                    selected_protein, nodes_df, sub_edges_df, memberships,
                    cohort=cohort, protein_set_label=protein_set_label,
                )
                st.plotly_chart(fig_bip, width='stretch', alt='Concentric bipartite ego-cluster map')
            elif ego_viewer == 'Cytoscape.js':
                ec1, ec2 = st.columns(2)
                ego_cy_clusters = ec1.checkbox('Include OCG cluster nodes', value=True, key='ocg_ego_cy_clusters')
                ego_cy_layout = ec2.selectbox('Layout', ['cose', 'concentric', 'circle', 'breadthfirst', 'grid'], key='ocg_ego_cy_layout')
                ego_cy_html = ego_cluster_cytoscapejs_html(
                    selected_protein, nodes_df, sub_edges_df, memberships,
                    cohort=cohort, include_clusters=ego_cy_clusters, layout_name=ego_cy_layout, height=560,
                )
                st.iframe(ego_cy_html, height=575, alt='Cytoscape.js ego-cluster neighborhood')
            elif ego_viewer == 'Apache ECharts':
                ee1, ee2 = st.columns(2)
                ego_ec_layout = ee1.segmented_control('Layout', ['force', 'circular'], default='force',
                                                      format_func=str.capitalize, required=True, key='ocg_ego_ec_layout')
                ego_ec_clusters = ee2.checkbox('Include OCG cluster nodes', value=True, key='ocg_ego_ec_clusters')
                st.echarts_chart(
                    ego_cluster_echarts_option(
                        selected_protein, nodes_df, sub_edges_df, memberships,
                        cohort=cohort, include_clusters=ego_ec_clusters, layout=ego_ec_layout,
                    ),
                    height=540,
                    alt='Apache ECharts ego-cluster neighborhood',
                )
            elif ego_viewer == '3D':
                st.plotly_chart(
                    ego_cluster_3d_figure(
                        selected_protein, nodes_df, sub_edges_df, memberships,
                        cohort=cohort, protein_set_label=protein_set_label,
                    ),
                    width='stretch',
                    alt='3D OCG ego-cluster neighborhood',
                )
            else:
                ego_static = ego_cluster_static_figure(
                    selected_protein, nodes_df, sub_edges_df, memberships,
                    cohort=cohort, protein_set_label=protein_set_label,
                )
                st.image(static_image_bytes(ego_static, 'png'), width='stretch',
                         alt='Static Matplotlib OCG ego-cluster neighborhood')
                ed1, ed2, ed3 = st.columns(3)
                for col, fmt, mime in [(ed1, 'png', 'image/png'), (ed2, 'svg', 'image/svg+xml'), (ed3, 'pdf', 'application/pdf')]:
                    col.download_button(f'Download {fmt.upper()}', data=lambda fmt=fmt: static_image_bytes(ego_static, fmt),
                                        file_name=f'{network}_{selected_protein}_ego_ocg.{fmt}', mime=mime,
                                        key=f'ocg_ego_static_dl_{fmt}', on_click='ignore')

            named_nodes = nodes_df.with_columns(
                pl.Series('name', [identities[p]['label'] for p in nodes_df['protein']]))
            st.dataframe(named_nodes.select('name', pl.exclude('name')).sort('is_center', 'memberships', descending=[True, True]),
                         hide_index=True, width='stretch')


render_exploration()
