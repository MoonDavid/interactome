"""GO/Reactome annotation and interactive OCG overlap exploration."""
from __future__ import annotations

import hashlib
from html import escape
from importlib.util import find_spec
from pathlib import Path
import time

import polars as pl
import requests
import streamlit as st

from clusterannotation import DATA as DEFAULT_DATA, DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME, network_population
from clustering_explorer import annotate_run, annotation_target, saved_cluster_enrichment
from cytoscape_bridge import DEFAULT_BASE_URL, export_cytoscape_bundle_zip, export_graphml_string, send_to_cytoscape
from moonlighting import ADVANCED_EDGE_LIMIT, significance_label
from network_view import static_image_bytes
from ocg_exploration import cluster_overlap_graph
from ocg_visualization import (
    annotated_ocg_overlap_matrix_figure,
    build_annotated_ocg_bipartite_view,
    build_annotated_ocg_overlap_view,
    build_ocg_cluster_ppi_view,
    build_ocg_pair_ppi_view,
    create_ocg_norma_session,
    ocg_core_interface_landscape_figures,
    ocg_static_upset_figure,
    render_ocg_view_3d_plotly,
    render_ocg_view_cytoscapejs,
    render_ocg_view_echarts,
    render_ocg_view_plotly2d,
    render_ocg_view_pyvis,
    render_ocg_view_sigma,
    render_ocg_view_static,
)


def available_annotation_bundle(directory, selected_ids):
    """Prefer full-run disk caches, retaining their original correction family."""
    for communities, text in ((None, True), (None, False), (selected_ids, False)):
        target, _, _, _ = annotation_target(directory, communities, text_annotations=text)
        if (target / 'manifest.json').is_file():
            return target, communities is None
    return None, False


def protein_cluster_labels(protein, memberships, summary=None):
    """List the center protein's full-run memberships, independently of neighbors."""
    members = memberships.select('protein', pl.col('community').cast(pl.String)).drop_nulls().unique()
    selected = members.filter(pl.col('protein') == protein).select('community').unique()
    sizes = members.group_by('community').agg(pl.len().alias('cluster_proteins'))
    result = selected.join(sizes, on='community', how='left')
    if summary is not None:
        columns = ['community'] + [c for c in summary.columns if c not in ('community', 'proteins')]
        result = result.join(summary.select(columns).with_columns(pl.col('community').cast(pl.String)),
                             on='community', how='left')
        for column in summary.columns:
            if column.endswith('_qvalue'):
                result = result.with_columns(pl.Series(column.removesuffix('_qvalue') + '_significance',
                    [significance_label(q) for q in result[column]]))
    return result.sort('community')


def ego_membership_panel(protein, memberships, directory):
    st.markdown('**OCG clusters containing the selected protein**')
    ids = memberships.filter(pl.col('protein') == protein)['community'].unique().to_list()
    if not ids:
        st.info('This protein has no assigned OCG cluster in the saved run.')
        return
    try:
        bundle, full_run = available_annotation_bundle(directory, ids)
        summary = (pl.read_csv(bundle / 'label_summary.csv', schema_overrides={'community': pl.String})
                   if bundle else None)
        table = protein_cluster_labels(protein, memberships, summary)
        st.caption(f'{protein} participates in {table.height:,} OCG cluster(s). '
                   'Cluster sizes and memberships use the complete saved run, independently of the displayed ego neighbors.')
        if bundle:
            st.success('Cluster labels loaded automatically from the saved annotation cache.')
        else:
            st.info('Cluster memberships are available; functional labels are not yet cached for this run.')
        st.dataframe(table, hide_index=True, width='stretch')
        if bundle:
            st.caption('GO/Reactome labels retain the saved annotation q-values and correction family '
                       f'({"all clusters in the full run" if full_run else "the saved selected-cluster annotation family"}). '
                       'One-sided hypergeometric tests use the full original network background; BH includes '
                       'all eligible terms, libraries and nonempty full/core/interface subsets in that family. '
                       '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable. '
                       'Hierarchy votes are descriptive. Cluster labels do not establish the selected protein’s function.')
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        st.dataframe(protein_cluster_labels(protein, memberships), hide_index=True)
        st.warning(f'Saved functional labels unavailable: {exc}')


def ego_protein_label(protein, identity, metadata):
    row = metadata.get(protein) or metadata.get(protein.split('-')[0], {})
    name = row.get('protein_name', '')
    parts = list(dict.fromkeys(p for p in (identity['label'], name, protein) if p))
    return ' · '.join(parts)


@st.cache_data(max_entries=32, show_spinner=False)
def protein_function_details(data_root, protein, signatures):
    """Read source descriptions without inferring functional roles from topology."""
    root = Path(data_root)
    parent = protein.split('-')[0]
    result = {'description': '', 'canonical': '', 'moonlighting': '',
              'moonlitdb': False, 'accession': parent}
    functions = root / 'annotations/uniprot_functions.tsv'
    if functions.is_file():
        frame = pl.scan_csv(functions, separator='\t', infer_schema=False).filter(
            pl.col('Entry') == parent).collect()
        if frame.height:
            result['description'] = frame['Function [CC]'][0] or ''
    source = root / 'allmoon_with_moonlitdb_human.csv'
    if source.is_file():
        frame = pl.scan_csv(source, infer_schema=False).filter(pl.col('uniprot') == parent).collect()
        if frame.height:
            row = frame.row(0, named=True)
            for field, column in [('canonical', 'moonlitdb_goslims_canonical'),
                                  ('moonlighting', 'moonlitdb_goslims_moonlighting')]:
                result[field] = row.get(column) or ''
            result['moonlitdb'] = str(row.get('moonlitdb', '')).lower() in ('true', '1', '1.0')
    return result


def ego_protein_description_panel(protein, data_root):
    with st.expander('UniProt description and MoonLitDB GO-slim functions',
                     key='ocg_ego_protein_description', on_change='rerun') as panel:
        if not panel.open:
            return
        root = Path(data_root)
        paths = [root / 'annotations/uniprot_functions.tsv', root / 'allmoon_with_moonlitdb_human.csv']
        signatures = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                           if p.is_file() else (str(p), None, None) for p in paths)
        details = protein_function_details(str(root), protein, signatures)
        st.markdown(f'[UniProt · {protein}](https://www.uniprot.org/uniprotkb/{protein}/entry)')
        if protein != details['accession']:
            st.caption(f"Functional descriptions below refer to parent accession {details['accession']}.")
        st.markdown('**UniProt FUNCTION description**')
        st.write(details['description'] or 'No UniProt FUNCTION description available in the local cache.')
        st.markdown('**MoonLitDB · canonical GO-slim functions**')
        st.write(details['canonical'] or 'No canonical GO-slim annotations available in the local MoonLitDB data.')
        st.markdown('**MoonLitDB · moonlighting GO-slim functions**')
        st.write(details['moonlighting'] or 'No moonlighting GO-slim annotations available in the local MoonLitDB data.')
        st.caption('GO-slim roles are source annotations from MoonLitDB, independent of the selected protein set. '
                   'Missing annotations do not establish absence of a function. These descriptions are not '
                   'cluster enrichment tests; no statistical significance test applies.')


@st.cache_data(max_entries=4, show_spinner=False)
def overlap_tables(memberships):
    return cluster_overlap_graph(memberships, max_edges=None)


@st.cache_data(max_entries=4, show_spinner=False)
def cluster_protein_set_enrichment(memberships, network_path, network_signature, cohort):
    """Correct over the complete run before filtering clusters with protein-set hits."""
    return saved_cluster_enrichment(memberships, network_population(Path(network_path)), set(cohort))


def moonlighting_cluster_view(clusters, pairs, enrichment):
    eligible = enrichment.filter(pl.col('protein_set_hits') > 0).with_columns(
        pl.col('community').cast(clusters.schema['community']))
    clusters = clusters.join(eligible, on='community', how='inner')
    ids = clusters['community']
    if not pairs.is_empty():
        pairs = pairs.filter(pl.col('community_a').is_in(ids.implode()) &
                             pl.col('community_b').is_in(ids.implode()))
    return clusters, pairs


@st.cache_data(max_entries=4, show_spinner=False)
def protein_name_lookup(data_root, signatures):
    from network_graph_utils import protein_metadata
    return {r['accession']: r for r in protein_metadata(data_root).iter_rows(named=True)}


def local_protein_names(data_root):
    paths = [Path(data_root) / 'GO/HUMAN-uniprot.gaf.gz',
             Path(data_root) / 'allmoon_with_moonlitdb_human.csv',
             Path(data_root) / 'raw_networks/bioplex/BioPlex_HEK293T_v3.0.tsv']
    signatures = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.is_file())
    return protein_name_lookup(str(data_root), signatures)


def overlap_option(clusters, pairs, evidence=None, *, focus=None, limit=150,
                   min_shared=1, min_similarity=0.0, metric='jaccard'):
    """Bound the display, retaining a requested cluster even outside largest-N."""
    if metric not in {'jaccard', 'overlap_coef'}:
        raise ValueError('Unknown overlap metric')
    limit = max(1, min(300, int(limit)))
    eligible = pairs.filter((pl.col('shared_proteins') >= min_shared) &
                            (pl.col(metric) >= min_similarity)) if not pairs.is_empty() else pairs
    nodes = clusters.sort(['size', 'community'], descending=[True, False])
    if focus is not None:
        neighbors = eligible.filter((pl.col('community_a') == focus) |
                                    (pl.col('community_b') == focus)) if not eligible.is_empty() else eligible
        ids = [focus]
        for row in neighbors.iter_rows(named=True):
            other = row['community_b'] if row['community_a'] == focus else row['community_a']
            if other not in ids:
                ids.append(other)
        nodes = nodes.filter(pl.col('community').is_in(ids[:limit]))
    else:
        nodes = nodes.head(limit)
    ids = nodes['community'].to_list()
    drawn = (eligible.filter(pl.col('community_a').is_in(ids) & pl.col('community_b').is_in(ids))
             .head(2500)) if not eligible.is_empty() else eligible
    labels = {}
    if evidence is not None and not evidence.is_empty():
        best = (evidence.filter((pl.col('subset') == 'full') & (pl.col('qvalue') < .05) &
                               (pl.col('fold_enrichment') > 1))
                .sort('qvalue', 'term_id').unique('community', maintain_order=True))
        labels = {str(row['community']): row for row in best.iter_rows(named=True)}
    data = []
    for row in nodes.iter_rows(named=True):
        cid, size = row['community'], row['size']
        annotation = labels.get(str(cid))
        term = (f"{annotation['term_name']} {significance_label(annotation['qvalue'])}"
                if annotation else 'No significant saved label')
        tip = (f'C{cid}<br>{size} proteins<br>{row["overlap_neighbors"]} overlapping clusters (full run)'
               f'<br>{escape(term)}')
        if 'protein_set_hits' in row:
            tip += (f'<br>Moonlighting protein-set hits: {row["protein_set_hits"]}'
                    f'<br>Protein-set enrichment: {row["fold_enrichment"]:.3g}×'
                    f' · q={row["qvalue"]:.3g} {significance_label(row["qvalue"])}')
        if annotation:
            tip += (f'<br>{escape(annotation["library"])} · q={annotation["qvalue"]:.3g}'
                    f' · fold={annotation["fold_enrichment"]:.3g}×')
        data.append(dict(id=str(cid), name=f'C{cid}', value=size,
            symbolSize=max(14, min(65, 8 + 3 * size ** .5)),
            itemStyle={'color': '#db6b35' if cid == focus else '#327da8'},
            tooltip={'formatter': tip},
            label={'show': cid == focus or len(ids) <= 40}))
    links = [dict(source=str(r['community_a']), target=str(r['community_b']),
        value=r['shared_proteins'],
        lineStyle={'width': min(8, 1 + r['shared_proteins'] ** .5)},
        tooltip={'formatter': f'C{r["community_a"]} ↔ C{r["community_b"]}'
                 f'<br>Shared proteins: {r["shared_proteins"]}'
                 f'<br>Jaccard: {r["jaccard"]:.3f}<br>Overlap coefficient: {r["overlap_coef"]:.3f}'})
        for r in drawn.iter_rows(named=True)]
    option = {'tooltip': {}, 'series': [dict(type='graph', layout='force', roam=True,
        draggable=True, selectedMode='single', data=data, links=links,
        force={'repulsion': 220, 'edgeLength': [60, 180], 'layoutAnimation': False},
        emphasis={'focus': 'adjacency', 'label': {'show': True}},
        select={'itemStyle': {'borderColor': '#111111', 'borderWidth': 4}},
        lineStyle={'opacity': .45}, label={'position': 'right'})]}
    return option, nodes, drawn


VIEWER_FRAMEWORKS = [
    'Apache ECharts',
    'Cytoscape.js',
    'Sigma.js',
    'PyVis',
    'Plotly 2D',
    'Static',
    '3D',
    'NORMA',
]


@st.cache_data(max_entries=2, show_spinner=False)
def _cached_network_edges(network_path: str, signatures: tuple) -> pl.DataFrame:
    from networks import load_network
    return load_network(Path(network_path)).edges


@st.cache_data(max_entries=6, show_spinner=False)
def _cached_induced_edges(network_path: str, signatures: tuple, proteins: tuple[str, ...]) -> pl.DataFrame:
    path = Path(network_path)
    if not proteins:
        return pl.DataFrame(schema={'a': pl.String, 'b': pl.String})
    if path.suffix == '.parquet':
        return (
            pl.scan_parquet(path)
            .select(
                pl.col('a').cast(pl.String).str.strip_chars(),
                pl.col('b').cast(pl.String).str.strip_chars(),
            )
            .filter(pl.col('a').is_in(proteins) & pl.col('b').is_in(proteins))
            .collect()
        )
    from networks import load_network
    return load_network(path).edges.filter(pl.col('a').is_in(proteins) & pl.col('b').is_in(proteins))


def _cytoscape_export_expander(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    title: str,
    description: str,
    key_prefix: str,
) -> None:
    if nodes_df.is_empty():
        return
    with st.expander('Download network / open in Cytoscape Desktop'):
        st.caption(
            'Export GraphML, node/edge CSVs, or a full Cytoscape bundle (.zip), or push directly to a running '
            'Cytoscape Desktop via CyREST. GraphML retains colors, shapes, sizes, roles and annotation attributes.'
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button(
            'Download GraphML',
            data=lambda: export_graphml_string(nodes_df, edges_df),
            file_name=f'{key_prefix}.graphml',
            mime='application/graphml+xml',
            key=f'{key_prefix}_dl_graphml',
            on_click='ignore',
        )
        c2.download_button(
            'Download nodes (CSV)',
            data=lambda: nodes_df.write_csv(),
            file_name=f'{key_prefix}_nodes.csv',
            mime='text/csv',
            key=f'{key_prefix}_dl_nodes',
            on_click='ignore',
        )
        c3.download_button(
            'Download edges (CSV)',
            data=lambda: edges_df.write_csv(),
            file_name=f'{key_prefix}_edges.csv',
            mime='text/csv',
            key=f'{key_prefix}_dl_edges',
            on_click='ignore',
        )
        c4.download_button(
            'Cytoscape bundle (.zip)',
            data=lambda: export_cytoscape_bundle_zip(
                nodes_df, edges_df, title=title, metadata={'description': description},
            ),
            file_name=f'{key_prefix}_cytoscape.zip',
            mime='application/zip',
            key=f'{key_prefix}_dl_zip',
            on_click='ignore',
        )
        base_url = st.text_input('Cytoscape CyREST URL', value=DEFAULT_BASE_URL, key=f'{key_prefix}_cyrest')
        if st.button('Send to Cytoscape Desktop', key=f'{key_prefix}_send_cy'):
            try:
                with st.spinner('Sending network to Cytoscape Desktop…'):
                    sent = send_to_cytoscape(nodes_df, edges_df, title=title, base_url=base_url)
                st.success(
                    f'Opened {sent["nodes_sent"]:,} nodes and {sent["edges_sent"]:,} links '
                    f'in Cytoscape Desktop (SUID {sent["network_suid"]}).'
                )
            except Exception as exc:
                st.error(f'Could not send to Cytoscape Desktop: {exc}')


def _render_multi_framework_network(
    nodes_df: pl.DataFrame,
    edges_df: pl.DataFrame,
    *,
    title: str,
    description: str,
    key_prefix: str,
    framework: str,
    legend_text: str,
    selected_node: str | None = None,
    highlight_label: str | None = None,
) -> None:
    if nodes_df.is_empty():
        st.info('No nodes match the current display filters.')
        return
    if edges_df.height > 20000 and framework != 'Sigma.js':
        st.warning('More than 20,000 displayed links: rendering may be slow or use substantial memory. Sigma.js is an option for larger graphs.')

    if framework == 'Apache ECharts':
        ec_layout = st.segmented_control(
            'Layout', ['force', 'circular'], default='force', required=True,
            format_func=str.capitalize, key=f'{key_prefix}_ec_layout',
        )
        option = render_ocg_view_echarts(nodes_df, edges_df, layout=ec_layout)
        st.echarts_chart(option, height=620, key=f'{key_prefix}_echarts', alt=title)

    elif framework == 'Cytoscape.js':
        cy_layout = st.selectbox(
            'Cytoscape.js layout', ['cose', 'concentric', 'circle', 'breadthfirst', 'grid'],
            key=f'{key_prefix}_cy_layout',
        )
        cy_html = render_ocg_view_cytoscapejs(
            nodes_df, edges_df, layout_name=cy_layout, height=620, title=title, legend_html=legend_text,
        )
        st.iframe(cy_html, height=635, alt=f'{title} in Cytoscape.js')
        st.download_button(
            'Download Cytoscape.js HTML', cy_html, f'{key_prefix}_cytoscapejs.html',
            'text/html', key=f'{key_prefix}_cy_dl', on_click='ignore',
        )

    elif framework == 'Sigma.js':
        sigma_engines = (
            (['igraph DrL', 'igraph FR'] if find_spec('igraph') else [])
            + (['graph-tool SFDP'] if find_spec('graph_tool') else [])
            + ['NetworkX Spring', 'NetworkX Kamada–Kawai', 'Circular']
        )
        sigma_engine = st.selectbox('Python layout engine', sigma_engines, key=f'{key_prefix}_sigma_engine')
        st.caption('Sigma.js 3 / Graphology use WebGL with fixed Python coordinates; no browser force simulation runs.')
        sigma_html, sigma_nodes = render_ocg_view_sigma(nodes_df, edges_df, engine=sigma_engine, title=title)
        st.iframe(sigma_html, height=700, alt=f'{title} in Sigma.js WebGL')
        s1, s2 = st.columns(2)
        s1.download_button(
            'Download interactive Sigma HTML', sigma_html, f'{key_prefix}_sigma.html',
            'text/html', key=f'{key_prefix}_sigma_dl', on_click='ignore',
        )
        s2.download_button(
            'Download GraphML with coordinates',
            data=lambda: export_graphml_string(sigma_nodes, edges_df),
            file_name=f'{key_prefix}_sigma.graphml',
            mime='application/graphml+xml',
            key=f'{key_prefix}_sigma_gml',
            on_click='ignore',
        )

    elif framework == 'PyVis':
        if not find_spec('pyvis'):
            st.info('PyVis is unavailable in this environment. Install pyvis>=0.3.2,<0.4 to enable it.')
        else:
            p1, p2 = st.columns(2)
            pv_physics = p1.toggle('Enable force simulation', value=True, key=f'{key_prefix}_pv_physics')
            pv_labels = p2.checkbox('Show node labels', value=nodes_df.height <= 120, key=f'{key_prefix}_pv_labels')
            pv_html = render_ocg_view_pyvis(
                nodes_df, edges_df, physics=pv_physics, labels=pv_labels, legend_text=legend_text,
            )
            st.iframe(pv_html, height=660, alt=f'{title} in PyVis')
            st.download_button(
                'Download interactive PyVis HTML', pv_html, f'{key_prefix}_pyvis.html',
                'text/html', key=f'{key_prefix}_pv_dl', on_click='ignore',
            )

    elif framework == 'Plotly 2D':
        p1, p2, p3 = st.columns(3)
        pl_engines = (
            ['NetworkX']
            + (['igraph'] if find_spec('igraph') else [])
            + (['graph-tool'] if find_spec('graph_tool') else [])
            + ['Circular']
        )
        pl_engine = p1.selectbox('Layout engine', pl_engines, key=f'{key_prefix}_pl_engine')
        pl_layouts = (
            ['FR', 'DrL', 'Circular'] if pl_engine == 'igraph'
            else (['SFDP', 'Spring', 'Circular'] if pl_engine == 'graph-tool'
                  else (['Circular'] if pl_engine == 'Circular' else ['Spring', 'Kamada–Kawai', 'Circular']))
        )
        pl_layout = p2.selectbox('Layout algorithm', pl_layouts, key=f'{key_prefix}_pl_layout')
        pl_labels = p3.checkbox('Show labels', value=True, key=f'{key_prefix}_pl_labels')
        with st.spinner('Computing 2D network layout…'):
            fig_pl = render_ocg_view_plotly2d(
                nodes_df, edges_df, title=title, engine=pl_engine, layout=pl_layout, show_labels=pl_labels,
            )
        st.plotly_chart(fig_pl, width='stretch', key=f'{key_prefix}_pl_chart', alt=title)

    elif framework == 'Static':
        s1, s2, s3 = st.columns(3)
        st_engines = ['NetworkX'] + (['igraph'] if find_spec('igraph') else []) + ['graph-tool', 'Circular']
        st_engine = s1.selectbox('Static layout engine', st_engines, key=f'{key_prefix}_st_engine')
        gt_ready = st_engine != 'graph-tool' or find_spec('graph_tool') is not None
        st_layouts = (
            ['SFDP', 'Spring', 'Circular'] if st_engine == 'graph-tool'
            else (['FR', 'DrL', 'Circular'] if st_engine == 'igraph'
                  else (['Circular'] if st_engine == 'Circular' else ['Spring', 'Kamada–Kawai', 'Circular']))
        )
        st_layout = s2.selectbox('Static layout', st_layouts, key=f'{key_prefix}_st_layout')
        st_labels = s3.checkbox('Show node labels', value=nodes_df.height <= 120, key=f'{key_prefix}_st_labels')
        if not gt_ready:
            st.info(
                'graph-tool is not installed in this Python environment. On Windows it requires running inside WSL or Docker. '
                'NetworkX and igraph remain available.'
            )
            st.link_button('graph-tool installation instructions', 'https://graph-tool.skewed.de/installation.html')
        else:
            with st.spinner('Rendering static figure…'):
                fig_st = render_ocg_view_static(
                    nodes_df, edges_df, title=title, engine=st_engine, layout=st_layout, labels=st_labels,
                )
            st.image(static_image_bytes(fig_st, 'png'), width='stretch', alt=title)
            d1, d2, d3 = st.columns(3)
            for col, fmt, mime in [(d1, 'png', 'image/png'), (d2, 'svg', 'image/svg+xml'), (d3, 'pdf', 'application/pdf')]:
                col.download_button(
                    f'Download {fmt.upper()}',
                    data=lambda fmt=fmt: static_image_bytes(fig_st, fmt),
                    file_name=f'{key_prefix}.{fmt}',
                    mime=mime,
                    key=f'{key_prefix}_st_dl_{fmt}',
                    on_click='ignore',
                )

    elif framework == '3D':
        with st.spinner('Computing 3D layout…'):
            fig_3d = render_ocg_view_3d_plotly(nodes_df, edges_df, title=title)
        st.plotly_chart(fig_3d, width='stretch', key=f'{key_prefix}_3d_plotly', alt=title)

    else:  # NORMA
        if edges_df.is_empty():
            st.info('NORMA requires at least one link in the displayed view.')
        else:
            st.caption('Creating a NORMA session sends the displayed source–target pairs to normaviz.org for temporary storage.')
            norma_key = f'{key_prefix}_norma_session'
            edge_sig = (nodes_df.height, edges_df.height, title)
            session = st.session_state.get(norma_key)
            if session and (session.get('sig') != edge_sig or session.get('expires', 0) <= time.monotonic()):
                st.session_state.pop(norma_key, None)
                session = None
            if st.button('Load in NORMA', type='primary', key=f'{key_prefix}_load_norma'):
                with st.spinner('Creating the NORMA session…'):
                    try:
                        url, expires = create_ocg_norma_session(title, edges_df)
                    except (requests.RequestException, ValueError, TypeError) as exc:
                        st.error(f'Could not create a NORMA session. {exc}')
                    else:
                        session = {'sig': edge_sig, 'url': url, 'expires': expires}
                        st.session_state[norma_key] = session
            if session:
                st.link_button('Open NORMA in a new tab', session['url'], icon=':material/open_in_new:')
                st.iframe(session['url'], height=760, alt=f'{title} in NORMA')

    _cytoscape_export_expander(
        nodes_df, edges_df, title=title, description=description, key_prefix=key_prefix,
    )


def annotation_workspace(directory, memberships, input_id, edge_count, cohort, protein_set_label, network_path):
    from cluster_annotation_ui import hierarchy_panel, readable_annotation_sections, bundle_zip

    pairs, clusters = overlap_tables(memberships)
    if clusters.is_empty():
        st.info('No assigned clusters in this run.')
        return
    path = Path(network_path)
    path_sig = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    enrichment = cluster_protein_set_enrichment(memberships, str(path),
        (path.stat().st_mtime_ns, path.stat().st_size), tuple(sorted(cohort)))
    total = clusters.height
    clusters, pairs = moonlighting_cluster_view(clusters, pairs, enrichment)
    selected_ids = clusters.sort('community')['community'].to_list()
    st.subheader(f'GO and Reactome · clusters containing {protein_set_label}')
    st.caption(f'{clusters.height:,} of {total:,} clusters contain at least one {protein_set_label} protein. '
               'All members of these clusters are retained, including proteins outside the selected protein set.')
    if not selected_ids:
        st.info(f'No clusters contain a {protein_set_label} protein in this run.')
        return
    st.info('Annotate clusters containing the selected moonlighting protein set with GO BP/MF/CC and human Reactome. '
        'The fast mode skips UniProt FUNCTION and GeneRIF text analysis and reuses matching disk bundles. '
        'One-sided hypergeometric tests use the full network background; BH correction includes all '
        'eligible terms, libraries, qualifying clusters and nonempty full/core/interface subsets, including hidden results. '
        '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable. '
        'Core/interface membership uses the complete original run. Changing display filters or selecting a cluster preserves this annotation correction family.')
    source_files = [DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME,
                    Path(__file__).with_name('clusterannotation.py'),
                    Path(__file__).with_name('clustering_explorer.py'),
                    Path(__file__).with_name('moonlighting.py')]
    identity = (input_id, tuple(selected_ids), tuple(sorted(cohort)), tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                               if p.is_file() else (str(p), None, None) for p in source_files))
    key = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
    bundle, full_run_cache = available_annotation_bundle(directory, selected_ids)
    if bundle is None and st.button('Autoannotate moonlighting-containing clusters · GO + Reactome', type='primary', key='ocg_annotate_all'):
        try:
            with st.spinner('Annotating qualifying clusters with GO and Reactome; reusing matching saved results…'):
                bundle = annotate_run(directory, community=selected_ids, text_annotations=False)
            st.session_state['ocg_go_reactome_bundle'] = (identity, str(bundle))
        except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
            st.warning(f'Annotation unavailable: {exc}')
    cached = st.session_state.get('ocg_go_reactome_bundle')
    if bundle is None:
        bundle = Path(cached[1]) if cached and cached[0] == identity else None
    evidence = pl.read_parquet(bundle / 'enrichment.parquet') if bundle else None
    if bundle:
        summary = pl.read_csv(bundle / 'label_summary.csv', schema_overrides={'community': pl.String})
        st.success(f'Using saved annotation cache automatically · {summary.height:,} clusters. No annotation computation needed.')
        if full_run_cache:
            st.caption('These annotations use the full saved run: BH q-values include all eligible terms, '
                       'libraries, clusters and nonempty full/core/interface subsets. Displaying only clusters '
                       f'containing {protein_set_label} preserves those q-values. Downloads retain the full bundle.')
        st.download_button('Download all cluster annotations', lambda: bundle_zip(bundle),
            'ocg_go_reactome_annotations.zip', 'application/zip', key=f'ocg_ann_zip_{key}', on_click='ignore')
        overview = st.expander('All cluster labels', on_change='rerun')
        if overview.open:
            st.dataframe(summary.select([c for c in summary.columns if not c.startswith('UniProt_FUNCTION')]),
                         hide_index=True, alt='GO and Reactome labels for every OCG cluster')
    else:
        st.caption('Run autoannotation once to add functional labels and GO Sankey diagrams. '
                   'All OCG visualizations remain independently explorable below.')

    st.markdown(f'**{protein_set_label} enrichment by cluster**')
    st.caption('One-sided hypergeometric tests compare protein-set membership with the full network; '
        'BH correction covers every cluster in the original run, before the ≥1-hit filter. '
        'Fold >1 indicates overrepresentation; having a member does not imply significant enrichment. '
        '* q <0.05; ** q <0.01; *** q <0.001; **** q <0.0001; ns otherwise; n/a unavailable.')
    st.dataframe(clusters.select('community', 'proteins', 'protein_set_hits', 'expected_hits',
        'fold_enrichment', 'qvalue', 'significance').sort('qvalue', 'community'), hide_index=True,
        alt=f'{protein_set_label} enrichment in qualifying OCG clusters')
    all_ids = selected_ids
    selected = st.selectbox('Inspect a single cluster', all_ids,
        format_func=lambda c: f'C{c}', key=f'ocg_inspect_{key}', persist_state='session')
    chosen = memberships.filter(pl.col('community') == selected).select('protein').unique().sort('protein')
    st.caption(f'C{selected}: {chosen.height:,} proteins. Select any cluster here, including those outside the graph display cap.')

    st.markdown('**Multi-view OCG clustering & interactome visualization**')
    ocg_view = st.segmented_control(
        'OCG clustering view',
        [
            'Cluster overlap network',
            'Bipartite (clusters + bridge proteins)',
            'Selected cluster PPI (core vs. interface)',
            'Overlapping cluster pair PPI',
            'Overlap matrix & hierarchy',
            'UpSet & core/interface composition',
        ],
        default='Cluster overlap network',
        required=True,
        key=f'ocg_ann_view_{key}',
    )

    network_views = {
        'Cluster overlap network',
        'Bipartite (clusters + bridge proteins)',
        'Selected cluster PPI (core vs. interface)',
        'Overlapping cluster pair PPI',
    }
    if ocg_view in network_views and edge_count > ADVANCED_EDGE_LIMIT:
        st.warning(f'Interactive graph layout disabled above the {ADVANCED_EDGE_LIMIT:,}-edge guard. '
                   'Overlap matrix, composition charts, annotation and cluster inspection remain available.')
    elif ocg_view == 'Cluster overlap network':
        st.info(
            'Nodes are OCG clusters sized by protein count; edges connect clusters sharing proteins. '
            'In Apache ECharts, click any cluster node to inspect it below or click an edge to list shared proteins. '
            'Switch frameworks to explore with Cytoscape.js, Sigma.js WebGL, PyVis, Plotly 2D, Static Matplotlib, 3D, or NORMA.'
        )
        a, b = st.columns(2)
        focused = a.toggle('Focus on selected cluster and its overlaps', value=False, key='ocg_ann_focus')
        cap = b.slider('Maximum displayed clusters', 10, 300, 150, step=10, key='ocg_ann_cap')
        a, b, c = st.columns(3)
        metric = a.selectbox('Similarity', ['jaccard', 'overlap_coef'], key='ocg_ann_metric')
        minimum = b.number_input('Minimum shared proteins', min_value=1, value=1, key='ocg_ann_min')
        similarity = c.slider('Minimum similarity', 0.0, 1.0, 0.0, .05, key='ocg_ann_similarity')
        framework = st.segmented_control(
            'Visualization framework', VIEWER_FRAMEWORKS, default='Apache ECharts',
            required=True, key=f'ocg_ann_fw_overlap_{key}',
        )
        nodes_std, edges_std, stats = build_annotated_ocg_overlap_view(
            clusters, pairs, evidence,
            focus=selected if focused else None, limit=cap, min_shared=int(minimum),
            min_similarity=similarity, metric=metric, protein_set_label=protein_set_label,
        )
        if framework == 'Apache ECharts':
            option, nodes, drawn = overlap_option(clusters, pairs, evidence,
                focus=selected if focused else None, limit=cap, min_shared=int(minimum),
                min_similarity=similarity, metric=metric)
            chart_key = f'ocg_dynamic_{key}'

            def select_node():
                state = st.session_state.get(chart_key, {})
                for group in state.get('selection', {}).get('selected', []):
                    indices = group.get('data_indices', [])
                    if group.get('data_type') == 'node' and indices:
                        index = indices[-1]
                        if 0 <= index < nodes.height:
                            st.session_state[f'ocg_inspect_{key}'] = nodes['community'][index]

            event = st.echarts_chart(option, height=620, key=chart_key, on_select=select_node,
                alt='Draggable OCG clusters connected by shared proteins')
            for group in event.selection.selected:
                indices = group.get('data_indices', [])
                if group.get('data_type') == 'edge' and indices:
                    index = indices[-1]
                    if 0 <= index < drawn.height:
                        row = drawn.row(index, named=True)
                        left = memberships.filter(pl.col('community') == row['community_a']).select('protein').unique()
                        right = memberships.filter(pl.col('community') == row['community_b']).select('protein').unique()
                        st.markdown(f'**Shared proteins · C{row["community_a"]} ↔ C{row["community_b"]}**')
                        st.dataframe(left.join(right, on='protein', how='inner').sort('protein'), hide_index=True)
            st.caption(f'Showing {nodes.height:,}/{clusters.height:,} clusters and {drawn.height:,} edges '
                '(maximum 2,500 edges, strongest shared-protein counts first). Drag nodes, pan or zoom; '
                'click a node to inspect its annotation or an edge to list shared proteins. '
                'Node size encodes cluster size; edge width encodes shared proteins. '
                'Hover shows similarity and the strongest saved significant label. '
                'Overlap and layout are descriptive: no significance test applies to them.')
            _cytoscape_export_expander(
                nodes_std, edges_std,
                title=f'OCG cluster overlap ({protein_set_label})',
                description='OCG cluster-cluster overlap network with enrichment and functional annotations',
                key_prefix=f'ocg_ov_{key}',
            )
        else:
            st.caption(
                f"Showing {stats['displayed_clusters']:,}/{stats['total_clusters']:,} clusters and "
                f"{stats['drawn_edges']:,} overlap edges. Orange hexagon: focus cluster; blue diamond: significant "
                f"{protein_set_label} enrichment (BH q < 0.05, fold > 1); Viridis fill: full-run overlap degree."
            )
            _render_multi_framework_network(
                nodes_std, edges_std,
                title=f'OCG cluster overlap ({protein_set_label})',
                description='OCG cluster-cluster overlap network with enrichment and functional annotations',
                key_prefix=f'ocg_ov_{key}',
                framework=framework,
                legend_text=f'Hexagon: focus C{selected} · Blue diamond: {protein_set_label}-enriched (q<0.05) · Viridis: overlap neighbors',
                selected_node=f'C{selected}',
                highlight_label=f'{protein_set_label}-enriched (q<0.05)',
            )

    elif ocg_view == 'Bipartite (clusters + bridge proteins)':
        st.info(
            'Two-mode bipartite view linking OCG clusters (hexagons) directly to the individual multi-cluster bridge '
            f'proteins (k ≥ 2) shared across them. Blue diamonds highlight {protein_set_label} bridge proteins; '
            'gray circles are other multi-cluster bridges. Topology alone does not establish moonlighting.'
        )
        b1, b2, b3 = st.columns(3)
        bip_focus = b1.toggle('Focus on selected cluster neighborhood', value=True, key='ocg_bip_focus')
        bip_cap = b2.slider('Maximum clusters', 10, 120, 45, step=5, key='ocg_bip_cap')
        bip_bridges = b3.slider('Maximum bridge proteins', 20, 400, 150, step=10, key='ocg_bip_bridges')
        c1, c2, c3 = st.columns(3)
        bip_metric = c1.selectbox('Cluster selection metric', ['jaccard', 'overlap_coef'], key='ocg_bip_metric')
        bip_min = c2.number_input('Minimum shared proteins', min_value=1, value=1, key='ocg_bip_min')
        bip_sim = c3.slider('Minimum similarity', 0.0, 1.0, 0.0, 0.05, key='ocg_bip_sim')
        framework = st.segmented_control(
            'Visualization framework', VIEWER_FRAMEWORKS, default='Cytoscape.js',
            required=True, key=f'ocg_ann_fw_bip_{key}',
        )
        metadata = local_protein_names(DEFAULT_DATA)
        nodes_bip, edges_bip, b_stats = build_annotated_ocg_bipartite_view(
            clusters, pairs, memberships, evidence,
            cohort=cohort, focus=selected if bip_focus else None,
            limit=bip_cap, max_bridges=bip_bridges,
            min_shared=int(bip_min), min_similarity=bip_sim, metric=bip_metric,
            metadata=metadata, protein_set_label=protein_set_label,
        )
        st.caption(
            f"Showing {b_stats['displayed_clusters']:,} OCG clusters, {b_stats['bridge_proteins']:,} bridge proteins (k ≥ 2), "
            f"and {b_stats['membership_links']:,} membership links. Layout is descriptive; no significance test applies."
        )
        _render_multi_framework_network(
            nodes_bip, edges_bip,
            title=f'OCG bipartite clusters & bridge proteins (C{selected})',
            description='Bipartite graph of OCG clusters and multi-cluster bridge proteins',
            key_prefix=f'ocg_bip_{key}_{selected}',
            framework=framework,
            legend_text=f'Hexagon: OCG cluster · Blue diamond: {protein_set_label} bridge · Gray circle: other bridge (k≥2)',
            selected_node=f'C{selected}',
            highlight_label=f'{protein_set_label} bridge proteins',
        )

    elif ocg_view == 'Selected cluster PPI (core vs. interface)':
        st.info(
            f'Physical protein–protein interaction (PPI) subnetwork inside cluster C{selected}. '
            'Core proteins (k = 1, green circles) belong exclusively to this cluster; '
            'Interface proteins (k ≥ 2, orange hexagons / gold borders) are shared with other OCG clusters; '
            f'blue diamonds mark {protein_set_label} proteins. Edges separate Interface–Interface (orange), '
            'Core–Interface (blue), and Core–Core (gray) interactions.'
        )
        p1, p2, p3 = st.columns(3)
        ppi_focus = p1.text_input(f'Focus protein inside C{selected} (optional)', key=f'ocg_ppi_focus_{selected}').strip().upper()
        ppi_max_n = p2.slider('Max proteins', 25, 1000, 300, step=25, key='ocg_ppi_max_nodes')
        ppi_max_e = p3.slider('Max interactions', 100, 8000, 3000, step=100, key='ocg_ppi_max_edges')
        framework = st.segmented_control(
            'Visualization framework', VIEWER_FRAMEWORKS, default='Cytoscape.js',
            required=True, key=f'ocg_ann_fw_ppi_{key}',
        )
        member_tuple = tuple(chosen['protein'].to_list())
        induced_edges = _cached_induced_edges(str(path), path_sig, member_tuple)
        metadata = local_protein_names(DEFAULT_DATA)
        nodes_ppi, edges_ppi, p_stats = build_ocg_cluster_ppi_view(
            selected, memberships, induced_edges,
            cohort=cohort, metadata=metadata, max_nodes=ppi_max_n, max_edges=ppi_max_e,
            focus_protein=ppi_focus, protein_set_label=protein_set_label,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Core proteins (k=1)', f"{p_stats['core_proteins']:,}")
        m2.metric('Interface proteins (k≥2)', f"{p_stats['interface_proteins']:,}")
        m3.metric('Displayed PPIs', f"{p_stats['displayed_edges']:,}/{p_stats['full_edges']:,}")
        m4.metric('Interface–Interface PPIs', f"{p_stats.get('interface_edges', 0):,}")
        _render_multi_framework_network(
            nodes_ppi, edges_ppi,
            title=f'C{selected} internal PPI network (core vs. interface)',
            description=f'Intra-cluster PPI wiring inside OCG cluster C{selected}',
            key_prefix=f'ocg_ppi_{key}_{selected}',
            framework=framework,
            legend_text=f'Blue diamond: {protein_set_label} · Orange hexagon: Interface (k≥2) · Green circle: Core (k=1)',
            selected_node=ppi_focus or None,
            highlight_label=f'{protein_set_label} / Interface',
        )

    elif ocg_view == 'Overlapping cluster pair PPI':
        st.info(
            f'Compare physical PPIs across C{selected} and one of its overlapping OCG clusters. '
            'Amber nodes sit in the shared interface (C_a ∩ C_b); blue and green nodes are exclusive to each cluster; '
            f'diamonds mark {protein_set_label} proteins. Dashed purple edges indicate direct PPIs bridging '
            'exclusive members across the two clusters.'
        )
        pair_neighbors = pairs.filter((pl.col('community_a') == selected) | (pl.col('community_b') == selected)) if not pairs.is_empty() else pairs
        if pair_neighbors.is_empty():
            st.info(f'Cluster C{selected} has no overlapping qualifying clusters to compare at the PPI level.')
        else:
            partner_ids = sorted({
                r['community_b'] if r['community_a'] == selected else r['community_a']
                for r in pair_neighbors.iter_rows(named=True)
            })
            q1, q2, q3 = st.columns(3)
            partner = q1.selectbox(
                f'Compare C{selected} with', partner_ids,
                format_func=lambda c: f'C{c}', key=f'ocg_pair_partner_{key}_{selected}',
            )
            pair_max_n = q2.slider('Max proteins', 25, 1000, 400, step=25, key='ocg_pair_max_nodes')
            pair_max_e = q3.slider('Max interactions', 100, 8000, 3500, step=100, key='ocg_pair_max_edges')
            framework = st.segmented_control(
                'Visualization framework', VIEWER_FRAMEWORKS, default='Cytoscape.js',
                required=True, key=f'ocg_ann_fw_pair_{key}',
            )
            union_proteins = tuple(
                memberships.filter(pl.col('community').is_in([selected, partner]))['protein']
                .unique().sort().to_list()
            )
            pair_edges_raw = _cached_induced_edges(str(path), path_sig, union_proteins)
            metadata = local_protein_names(DEFAULT_DATA)
            nodes_pair, edges_pair, pair_stats = build_ocg_pair_ppi_view(
                selected, partner, memberships, pair_edges_raw,
                cohort=cohort, metadata=metadata, max_nodes=pair_max_n, max_edges=pair_max_e,
                protein_set_label=protein_set_label,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f'Shared (C{selected}∩C{partner})', f"{pair_stats['shared_proteins']:,}")
            m2.metric(f'C{selected} exclusive', f"{pair_stats['only_a_proteins']:,}")
            m3.metric(f'C{partner} exclusive', f"{pair_stats['only_b_proteins']:,}")
            m4.metric('Cross-cluster PPIs', f"{pair_stats['cross_cluster_edges']:,}")
            _render_multi_framework_network(
                nodes_pair, edges_pair,
                title=f'PPI bridge: C{selected} ↔ C{partner}',
                description=f'Induced PPI subnetwork across overlapping clusters C{selected} and C{partner}',
                key_prefix=f'ocg_pair_{key}_{selected}_{partner}',
                framework=framework,
                legend_text=f'Amber: C{selected}∩C{partner} · Blue: C{selected} · Green: C{partner} · Diamond: {protein_set_label} · Dashed purple: cross-cluster PPI',
                highlight_label=protein_set_label,
            )

    elif ocg_view == 'Overlap matrix & hierarchy':
        st.info(
            'Hierarchically clustered pairwise overlap matrix (average linkage on 1 − similarity). '
            'Row and column labels combine cluster ID, top enriched GO/Reactome annotation (if cached), '
            'and BH-adjusted significance stars (* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns; n/a). '
            'Matrix overlap values are descriptive.'
        )
        m1, m2, m3, m4 = st.columns(4)
        mat_cap = m1.slider('Maximum clusters', 5, 150, 60, step=5, key='ocg_mat_cap')
        mat_metric = m2.selectbox('Similarity metric', ['jaccard', 'overlap_coef'], key='ocg_mat_metric')
        mat_sort = m3.selectbox('Order input clusters by', ['qvalue', 'size'],
                                format_func=lambda v: 'Protein-set BH q-value' if v == 'qvalue' else 'Cluster size',
                                key='ocg_mat_sort')
        mat_mode = m4.segmented_control('Renderer', ['Interactive (Plotly)', 'Static (Matplotlib)'],
                                        default='Interactive (Plotly)', required=True, key='ocg_mat_mode')
        fig_mat, static_mat, mat_df = annotated_ocg_overlap_matrix_figure(
            clusters, pairs, evidence, limit=mat_cap, metric=mat_metric,
            protein_set_label=protein_set_label, sort_by=mat_sort,
        )
        if mat_mode == 'Interactive (Plotly)':
            st.plotly_chart(fig_mat, width='stretch', key=f'ocg_mat_plotly_{key}',
                            alt='Hierarchically ordered annotated OCG overlap matrix')
        else:
            st.image(static_image_bytes(static_mat, 'png'), width='stretch',
                     alt='Static annotated OCG overlap matrix')
            d1, d2, d3 = st.columns(3)
            for col, fmt, mime in [(d1, 'png', 'image/png'), (d2, 'svg', 'image/svg+xml'), (d3, 'pdf', 'application/pdf')]:
                col.download_button(
                    f'Download Matrix {fmt.upper()}',
                    data=lambda fmt=fmt: static_image_bytes(static_mat, fmt),
                    file_name=f'ocg_overlap_matrix.{fmt}', mime=mime,
                    key=f'ocg_mat_dl_{fmt}', on_click='ignore',
                )
        if not mat_df.is_empty():
            st.dataframe(mat_df, hide_index=True, width='stretch')

    else:  # UpSet & core/interface composition
        st.info(
            'Compare each cluster’s Core (k = 1, exclusive) vs. Shared Interface (k ≥ 2) composition alongside '
            f'{protein_set_label} BH-adjusted enrichment (* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns; n/a), '
            'and inspect the top overlapping cluster intersections in an UpSet plot.'
        )
        u1, u2, u3, u4 = st.columns(4)
        top_n_comp = u1.slider('Clusters in Core/Interface chart', 10, 60, 30, step=5, key='ocg_comp_top')
        upset_pairs = u2.slider('Top pairs in UpSet plot', 5, 40, 25, step=5, key='ocg_ann_upset_pairs')
        upset_metric = u3.selectbox('UpSet similarity metric', ['jaccard', 'overlap_coef'], key='ocg_ann_upset_metric')
        comp_mode = u4.segmented_control('Renderer', ['Interactive (Plotly) + UpSet', 'Static (Matplotlib)'],
                                         default='Interactive (Plotly) + UpSet', required=True, key='ocg_comp_mode')
        full_edges = _cached_network_edges(str(path), path_sig) if edge_count <= ADVANCED_EDGE_LIMIT else None
        fig_bar, fig_scatter, static_comp, landscape_df = ocg_core_interface_landscape_figures(
            clusters, memberships, edges=full_edges, evidence=evidence,
            protein_set_label=protein_set_label, top_n=top_n_comp,
        )
        if comp_mode == 'Interactive (Plotly) + UpSet':
            c_left, c_right = st.columns(2)
            c_left.plotly_chart(fig_bar, width='stretch', key=f'ocg_comp_bar_{key}',
                                alt='Core vs interface composition by cluster with BH significance')
            c_right.plotly_chart(fig_scatter, width='stretch', key=f'ocg_comp_scatter_{key}',
                                 alt='Interface fraction vs fold enrichment / density')
        else:
            st.image(static_image_bytes(static_comp, 'png'), width='stretch',
                     alt='Static Core vs Interface composition and enrichment landscape')
            cd1, cd2, cd3 = st.columns(3)
            for col, fmt, mime in [(cd1, 'png', 'image/png'), (cd2, 'svg', 'image/svg+xml'), (cd3, 'pdf', 'application/pdf')]:
                col.download_button(
                    f'Download Composition {fmt.upper()}',
                    data=lambda fmt=fmt: static_image_bytes(static_comp, fmt),
                    file_name=f'ocg_core_interface.{fmt}', mime=mime,
                    key=f'ocg_comp_dl_{fmt}', on_click='ignore',
                )
        qualifying_mem = memberships.filter(pl.col('community').is_in(selected_ids))
        upset_fig = ocg_static_upset_figure(
            qualifying_mem, top_pairs=int(upset_pairs), metric=upset_metric,
            cohort=cohort, protein_set_label=protein_set_label,
        )
        st.image(static_image_bytes(upset_fig, 'png'), width='stretch',
                 alt='UpSet-style OCG cluster intersection plot')
        ud1, ud2, ud3 = st.columns(3)
        for col, fmt, mime in [(ud1, 'png', 'image/png'), (ud2, 'svg', 'image/svg+xml'), (ud3, 'pdf', 'application/pdf')]:
            col.download_button(
                f'Download UpSet {fmt.upper()}',
                data=lambda fmt=fmt: static_image_bytes(upset_fig, fmt),
                file_name=f'ocg_upset.{fmt}', mime=mime,
                key=f'ocg_ann_upset_dl_{fmt}', on_click='ignore',
            )
        if not landscape_df.is_empty():
            st.dataframe(landscape_df, hide_index=True, width='stretch')
            st.download_button(
                'Download cluster composition & quality table (CSV)',
                landscape_df.write_csv(), 'ocg_cluster_composition.csv', 'text/csv',
                key=f'ocg_comp_csv_{key}', on_click='ignore',
            )

    st.subheader(f'Cluster C{selected}')
    selected_enrichment = clusters.filter(pl.col('community') == selected).row(0, named=True)
    a, b, c = st.columns(3)
    a.metric(f'{protein_set_label} proteins', f'{selected_enrichment["protein_set_hits"]}/{selected_enrichment["proteins"]}')
    b.metric('Protein-set fold enrichment', f'{selected_enrichment["fold_enrichment"]:.3g}×')
    c.metric('Protein-set enrichment q', f'{selected_enrichment["qvalue"]:.3g} {selected_enrichment["significance"]}')
    neighbors = pairs.filter((pl.col('community_a') == selected) | (pl.col('community_b') == selected)) if not pairs.is_empty() else pairs
    if neighbors.is_empty():
        st.info('This cluster shares no proteins with other clusters.')
    else:
        st.markdown('**Overlaps with other moonlighting-containing clusters (before graph filters)**')
        st.dataframe(neighbors, hide_index=True, alt='Exact shared-protein counts and cluster similarities')
        other_ids = sorted({r['community_b'] if r['community_a'] == selected else r['community_a']
                            for r in neighbors.iter_rows(named=True)})
        other = st.selectbox('Show proteins shared with cluster', other_ids,
            format_func=lambda c: f'C{c}', key=f'ocg_shared_{key}_{selected}')
        shared = chosen.join(memberships.filter(pl.col('community') == other).select('protein').unique(),
                             on='protein', how='inner').sort('protein')
        st.dataframe(shared, hide_index=True, alt=f'Proteins shared by C{selected} and C{other}')
        st.download_button('Download shared proteins', shared.write_csv(),
            f'ocg_C{selected}_C{other}_shared.csv', 'text/csv', key=f'ocg_shared_csv_{key}_{selected}')
    with st.expander('All member proteins'):
        st.dataframe(chosen, hide_index=True, alt=f'Protein members of C{selected}')
    if bundle:
        selected_evidence = evidence.filter(pl.col('community') == str(selected))
        readable_annotation_sections(selected_evidence)
        hierarchy_panel(bundle, str(selected), f'ocg_hierarchy_{key}_{selected}')
        with st.expander('All GO and Reactome enrichment evidence'):
            st.dataframe(selected_evidence, hide_index=True, alt='Original corrected GO and Reactome enrichment results')
