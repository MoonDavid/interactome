"""Canonical human interaction networks, durable versions, and overlap exploration."""
from pathlib import Path
import importlib
import json
import altair as alt
import polars as pl
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import networks

# Streamlit reruns the entry point while imported modules can remain cached in a
# long-running server. Refresh the loader so existing sessions pick up updates.
importlib.reload(networks)
from networks import (
    BIOPLEX_ORIGINAL, FILES, ISOFORM_POLICIES, NETWORK_MAPPING, SOURCE_ISOFORM, TAGC_NAME,
    available_huri_isoform_policies, bioplex_release_paths, bioplex_source_ids, compare_isoform_mapping,
    degree_distribution, discover_datasets, ensure_swissprot_reference, filtering_audit_table, mapping_target_statistics,
    load_edges, load_network, membership_counts, network_source_path, prepare_huri_psi,
    prepare_mapping, rebuild_mapped_networks, rebuild_networks_from_saved_mapping, save_isoform_mapping_edges, save_scored_version, scan_humap_source,
    scan_huri_interactors, scan_mapping_source, score_column, source_endpoint_profile,
    source_identifier_counts, summarize_network,
    threshold_curve, upset_figure, venn4_figure, venn_figure,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
st.set_page_config(page_title='Interactomes Overview | Human Interactome Atlas', page_icon=':material/hub:', layout='wide')
st.markdown('<style>.block-container{padding-top:2.5rem}</style>', unsafe_allow_html=True)


st.write(f'Compare HuRI, BioPlex HEK293T, huMAP2 and {TAGC_NAME}. Save confidence-filtered networks and revisit them across sessions.')

@st.cache_data(show_spinner=False, max_entries=16)
def analyze(path_string, mtime, size):
    loaded = load_network(Path(path_string))
    statistics = dict(summarize_network(loaded))
    return statistics, degree_distribution(loaded)

@st.cache_data(show_spinner=False, max_entries=16)
def overlap_data(specs, mode):
    return membership_counts({name: load_edges(path) for name, path, mtime, size in specs}, mode)

@st.cache_data(show_spinner=False, max_entries=8)
def curve_data(parent, mtime, size, name):
    cache_dir = DATA / 'cache' / 'threshold_curves'
    cache_file = cache_dir / f'{name}_{mtime}_{size}.parquet'
    if cache_file.is_file():
        return pl.read_parquet(cache_file)
    curve = threshold_curve(DATA, name=name, parent=parent)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        curve.write_parquet(cache_file)
    except OSError:
        pass
    return curve


@st.cache_data(show_spinner=False, max_entries=8)
def cached_filtering_audit(specs):
    return filtering_audit_table({name: Path(path) for name, path, _, _ in specs})


@st.cache_data(show_spinner='Reading the raw huMAP2 columns…', max_entries=2)
def humap_source_view(path, mtime_ns, size):
    source = scan_humap_source(path)
    ids, rows, isoform_rows = source_endpoint_profile(source)
    isoforms = pl.Series('Source identifier', ids, dtype=pl.String).to_frame().filter(
        pl.col('Source identifier').str.contains(SOURCE_ISOFORM))
    return source.head(12).collect(), len(ids), rows, isoform_rows, isoforms


@st.cache_data(show_spinner='Counting distinct source identifiers…', max_entries=12)
def raw_identifier_breakdown(name, source_stamps):
    paths = [Path(path) for path, _, _ in source_stamps]
    if name == 'BioPlex':
        return source_identifier_counts(bioplex_source_ids(paths))
    path = paths[0]
    if name == 'HuRI':
        source = scan_huri_interactors(path)
    else:
        source = scan_mapping_source(name, path)
    ids, _, _ = source_endpoint_profile(source)
    return source_identifier_counts(pl.Series(ids, dtype=pl.String))


@st.cache_data(show_spinner=False, max_entries=2)
def mapping_view(path, mtime_ns, size, reference_stamp):
    audit = json.loads(path.read_text(encoding='utf-8'))
    source_records = audit['sources']
    if 'TAGC' in source_records:
        source_records.setdefault(TAGC_NAME, source_records.pop('TAGC'))
    sources = pl.DataFrame(
        [(name, source['ids']) for name, source in source_records.items()],
        schema={'Network': pl.String, 'Source identifier': pl.List(pl.String)}, orient='row',
    ).explode('Source identifier').drop_nulls()
    entries = pl.DataFrame(
        [{'Source identifier': identifier, **entry} for identifier, entry in audit['entries'].items()],
        schema={'Source identifier': pl.String, 'namespace': pl.String, 'query': pl.String,
                'parent_query': pl.String, 'status': pl.String, 'targets': pl.List(pl.String),
                'primary': pl.List(pl.String)},
    )
    joined = sources.join(entries, on='Source identifier', how='left', validate='m:1')
    reviewed = pl.Series('Entry', [], dtype=pl.String)
    if reference_stamp is not None:
        reviewed = pl.read_csv(reference_stamp[0], separator='\t', schema={'Entry': pl.String})['Entry']
    joined = joined.with_columns(
        pl.col('primary').list.filter(pl.element().is_in(reviewed.implode())).alias('reviewed'))
    statistics = mapping_target_statistics(joined)
    if reference_stamp is None:
        statistics = statistics.with_columns(pl.lit(None).alias('Distinct Swiss-Prot accessions'))
    table = joined.select(
        'Network', 'Source identifier', pl.col('namespace').alias('Namespace'),
        pl.col('query').alias('Exact query'), pl.col('parent_query').alias('Parent query'),
        pl.col('status').alias('Outcome'), pl.col('targets').list.join(', ').alias('UniProt matches'),
        pl.col('primary').list.join(', ').alias('Primary accessions'),
        (pl.col('reviewed').list.join(', ') if reference_stamp else pl.lit(None)).alias('Reviewed Swiss-Prot accessions'),
        pl.col('targets').list.len().alias('Matches'),
    ).sort('Network', 'Source identifier')
    return table, statistics, audit['created_utc'], audit['uniprot_releases'], {
        name: source['source'] for name, source in source_records.items()
    }


@st.cache_data(show_spinner='Comparing source and mapped edges…', max_entries=8)
def isoform_edge_view(name, source, source_stamp, report, report_stamp, cutoff):
    audit = json.loads(report.read_text(encoding='utf-8'))
    entries = audit['entries']
    return compare_isoform_mapping(name, source, entries, cutoff)


isoforms = st.session_state.get('isoform_policy', 'collapse')
datasets = discover_datasets(DATA, isoforms=isoforms)
psi_policies = available_huri_isoform_policies(DATA)
if len(psi_policies) < len(ISOFORM_POLICIES) and network_source_path(DATA, 'HuRI').is_file():
    if st.button('Prepare HuRI from PSI'):
        with st.spinner('Preparing HuRI isoform views…'):
            prepare_huri_psi(DATA)
        st.rerun()
if isoforms == 'collapse' and not ('HuRI' in datasets and 'BioPlex' in datasets and any(name == 'huMAP2' or name.startswith('huMAP2_') for name in datasets)):
    st.info(f'Prepare HuRI from its PSI file and resolve BioPlex, huMAP2 and {TAGC_NAME} identifiers with UniProt. Results are saved on disk.')
    st.code('.venv/bin/python networks.py', language='bash')
    if st.button('Map IDs with UniProt and prepare networks', type='primary'):
        try:
            with st.spinner('Mapping identifiers and preparing canonical networks…'):
                prepare_huri_psi(DATA)
                ensure_swissprot_reference(DATA)
                mapping = prepare_mapping(DATA)
                rebuild_mapped_networks(mapping, DATA)
            st.rerun()
        except Exception as exc:
            st.error(f'Preparation failed: {exc}. Existing source files are unchanged; retry to reuse completed mapping results.')

with st.sidebar:
    st.title('Interactomes Overview')
    previous = st.session_state.get('overview_networks', [name for name in datasets if name != 'huMAP2'])
    active = st.multiselect('Networks to explore', list(datasets), default=[name for name in previous if name in datasets])
    st.session_state['overview_networks'] = active
    st.caption('Dataset selectors include confidence versions saved for the active isoform policy.')

results, degrees = {}, {}
with st.spinner('Calculating topology…'):
    for name in active:
        path = datasets[name]
        info = path.stat()
        results[name], degrees[name] = analyze(str(path), info.st_mtime_ns, info.st_size)
for start in range(0, len(active), 3):
    for col, name in zip(st.columns(3), active[start:start+3]):
        with col:
            with st.container(border=True):
                st.subheader(name)
                a,b,c=st.columns(3)
                a.metric('Nodes', f"{results[name]['nodes']:,}")
                b.metric('Edges', f"{results[name]['edges']:,}")
                c.metric('Mapped UniProt labels', f"{results[name]['mapped_accessions']:,}")
st.caption(f'{ISOFORM_POLICIES[isoforms]}. Counts include proteins incident to at least one edge.')
st.caption('Multiple UniProt matches annotate one node; identical complete label sets merge. '
           'Mapped UniProt labels count distinct accessions attached to the final nodes, preserving isoform suffixes in the isoform view. '
           'Protein set matching uses any attached accession, counting each node once; ambiguous labels remain ambiguous.')

with st.expander('Filtering audit: nodes and edges before and after every step', expanded=True, on_change='rerun') as filtering_panel:
    if filtering_panel.open:
        audit_specs = tuple(
            (name, str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for name, path in datasets.items() if path.is_file()
        )
        filtering, missing_audits = cached_filtering_audit(audit_specs)
        st.write('All available networks and saved confidence versions are included, independently of the network selector. '
                 'Before deduplication, edge counts are interaction rows, including repeated or reversed pairs. '
                 'Nodes are distinct incident endpoint IDs at each step; edges lost with an unmapped endpoint '
                 'can also leave mapped proteins without any remaining edges.')
        if missing_audits:
            st.warning('Detailed filtering audits are missing or stale for: ' + ', '.join(missing_audits))
        if not filtering.is_empty():
            endpoints = filtering.group_by('Network', maintain_order=True).agg(
                pl.col('Nodes').first().alias('Nodes before'),
                pl.col('Edges / rows').first().alias('Interaction rows before'),
                pl.col('Nodes').last().alias('Nodes after'),
                pl.col('Edges / rows').last().alias('Unique edges after'))
            st.dataframe(endpoints, hide_index=True, alt='Network node and edge counts before and after filtering')
            st.dataframe(filtering, hide_index=True, height=520,
                         alt='All filtering passages with remaining nodes, edges, and losses per step')
            st.download_button('Download all filtering steps (CSV)', filtering.write_csv(),
                               'network_filtering_steps.csv', 'text/csv')
        st.caption('HuRI uses supplied PSI UniProt IDs without external lookup. BioPlex includes original release '
                   'merge steps when they reproduce the current merged input; otherwise its audit starts at the merged file. '
                   'Duplicate native scores use the maximum valid value. BioPlex invalid scores stay null in the base graph.')
        if st.button('Rebuild networks and filtering audits from saved mapping (offline)', key='rebuild_filtering_audits'):
            try:
                with st.spinner('Rebuilding networks, node labels, filtering audits and saved confidence versions…'):
                    rebuild_networks_from_saved_mapping(DATA)
                    if isoforms != 'collapse':
                        networks.prepare_isoform_views(DATA, isoforms)
                st.rerun()
            except (OSError, ValueError, KeyError) as exc:
                st.error(f'Could not rebuild filtering audits: {exc}')

overview, overlap, versions, mapping_tab, resources, pipeline = st.tabs(
    ['Network landscape', 'Interactomes overlap', 'Interaction scores & thresholds', 'Mapping & provenance',
     'Interactome descriptions & data sources', 'BioPlex preparation'],
    key='network_explorer_tab', on_change='rerun',
)
def render_original_ids():
    st.subheader('Which identifiers came with each interactome?')
    st.write('Each pie counts distinct IDs in the source interaction endpoints before UniProt lookup, '
             'isoform collapsing or confidence filtering. An isoform suffix counts as a separate ID.')
    st.caption('Identifier types are based on the format in the file. A UniProt accession shown here '
               'may be historical. Counts describe IDs in edges, which can represent the same protein.')
    st.caption('See Mapping & provenance for distinct UniProt and reviewed Swiss-Prot results.')
    bio_sources = bioplex_release_paths(DATA)
    source_views = [
        ('HuRI', 'HuRI · PSI-MITAB', [network_source_path(DATA, 'HuRI')],
         'Primary interactor columns. Ensembl IDs in the alternate-ID columns are additional annotations.'),
        ('huMAP2', 'huMAP2', [network_source_path(DATA, 'huMAP2')],
         'All pairs in the original accession archive.'),
        ('BioPlex', 'BioPlex HEK293T', bio_sources,
         'Distinct supplied accessions, plus NCBI Gene IDs with no accession in any included release.'),
        (TAGC_NAME, TAGC_NAME, [network_source_path(DATA, TAGC_NAME)],
         'The two interactor columns of the supplied CSV.'),
    ]
    id_colors = {
        'UniProt accession': '#287A9F',
        'Ensembl gene ID': '#CF702B',
        'Ensembl protein ID': '#8C62B5',
        'Ensembl transcript ID': '#A34962',
        'NCBI Gene ID': '#3A8A6C',
        'Other ID': '#7C858E',
    }
    for start in range(0, len(source_views), 2):
        for col, (name, title, paths, detail) in zip(st.columns(2), source_views[start:start + 2]):
            with col, st.container(border=True):
                st.subheader(title)
                st.caption(detail)
                stamps = []
                for path in paths:
                    if path.is_file():
                        info = path.stat()
                        stamps.append((str(path), info.st_mtime_ns, info.st_size))
                if not stamps:
                    st.info('Source files are missing: ' + ', '.join(path.name for path in paths))
                    continue
                if len(stamps) < len(paths):
                    st.warning(f'Showing {len(stamps)} of {len(paths)} expected source releases.')
                counts = raw_identifier_breakdown(name, tuple(stamps))
                if counts.is_empty():
                    st.info('The source contains no endpoint identifiers.')
                    continue
                total = counts['Distinct IDs'].sum()
                counts = counts.with_columns((pl.col('Distinct IDs') / total).alias('Share'))
                types = counts['Identifier type'].to_list()
                pie = alt.Chart(counts).mark_arc(stroke='white', strokeWidth=1).encode(
                    theta=alt.Theta('Distinct IDs:Q', stack=True),
                    color=alt.Color(
                        'Identifier type:N', legend=alt.Legend(title=None, orient='bottom', columns=2),
                        scale=alt.Scale(domain=types, range=[id_colors[kind] for kind in types]),
                    ),
                    tooltip=[
                        alt.Tooltip('Identifier type:N'),
                        alt.Tooltip('Distinct IDs:Q', format=',d'),
                        alt.Tooltip('Share:Q', format='.2%'),
                        alt.Tooltip('Example ID:N'),
                    ],
                ).properties(height=172)
                st.altair_chart(pie, alt=f'Original endpoint identifiers by type in {title}')
                st.caption(f'{total:,} distinct source IDs')
                st.dataframe(
                    counts.select('Identifier type', 'Distinct IDs', 'Share', 'Example ID'), hide_index=True,
                    column_config={'Share': st.column_config.NumberColumn(format='percent')},
                    alt=f'Counts and percentages by identifier type in {title}',
                )


if overview.open:
    with overview:
        if results:
            st.subheader('The network landscape at a glance')
            fields = {'nodes':'Nodes (distinct endpoint IDs)','edges':'Unique edges','mapped_accessions':'Mapped UniProt labels (distinct accessions)','mean_degree':'Mean degree','density':'Density','components':'Components','largest_component':'Largest component (nodes)','largest_component_fraction':'Largest component fraction'}
            st.dataframe(pl.DataFrame({'Metric':list(fields.values()), **{name:pl.Series(name,[result[key] for key in fields],dtype=pl.Float64) for name,result in results.items()}}), width='stretch', hide_index=True)
            st.caption('Each distinct node label set in a retained edge counts as one node. '
                       'A node labelled P1|P2 is one node with two mapped UniProt labels. '
                       'Labels shared across nodes count once in the label total. Edges are unique unordered pairs.')
            frames=[]
            for name, frame in degrees.items():
                frames.append(frame.with_columns(pl.lit(name).alias('Network')))
            fig=px.scatter(pl.concat(frames), x='degree', y='count', color='Network', log_x=True, log_y=True, labels={'degree':'Degree (interaction partners)','count':'Nodes at this degree'}, title='How connections are distributed across proteins')
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, width='stretch')
        else:
            st.info('Select datasets in the sidebar to show their statistics.')

if overlap.open:
    with overlap:
        st.subheader('Interactome overlap')
        default_humap = next((name for name in ['huMAP2_0.05', 'huMAP2_0.95'] if name in datasets), None)
        selected=st.multiselect('Datasets to compare', list(datasets), default=[name for name in ['HuRI','BioPlex',default_humap,TAGC_NAME] if name in datasets], max_selections=12)
        c1,c2=st.columns(2)
        mode=c1.radio('Compare', ['Edges','Nodes'], horizontal=True)
        style=c2.radio('Diagram', ['Venn','UpSet'], horizontal=True)
        st.caption('Each node or undirected edge appears in exactly one region. Venn percentages use the union of the selected networks as the denominator. See Exact intersection counts for the datasets in each observed region.')
        if selected:
            if style=='Venn' and len(selected) not in (2,3,4):
                st.info('Venn supports two to four datasets. Choose UpSet for one or more than four.')
            else:
                specs=tuple((name,str(datasets[name]),datasets[name].stat().st_mtime_ns,datasets[name].stat().st_size) for name in selected)
                with st.spinner('Computing exact intersections…'):
                    counts=overlap_data(specs,mode)
                if style == 'Venn' and len(selected) == 4:
                    st.caption(f'{mode} · Schematic Venn with fixed region areas')
                    st.pyplot(venn4_figure(counts, selected), width='content',
                              alt=f'Exclusive {mode.lower()} overlap among {", ".join(selected)}')
                else:
                    fig=venn_figure(counts, selected) if style=='Venn' else upset_figure(counts,selected)
                    fig.update_layout(title=f'{mode}: '+fig.layout.title.text)
                    st.plotly_chart(fig, width='stretch', key='overlap_plot')
                table=counts.with_columns(
                    pl.Series('Present in', [', '.join(n for i,n in enumerate(selected) if int(mask)&(1<<i)) for mask in counts['mask']]),
                    pl.Series('Absent from', [', '.join(n for i,n in enumerate(selected) if not int(mask)&(1<<i)) for mask in counts['mask']]),
                )
                with st.expander('Exact intersection counts'):
                    st.dataframe(table.select('Present in','Absent from','count'),width='stretch')
                    st.download_button('Download overlap counts',table.write_csv(),'overlap_counts.csv','text/csv')
        else:
            st.info('Select datasets to compare.')

if versions.open and any(name in datasets for name in ['huMAP2', 'BioPlex', 'HuRI']):
    with versions:
        st.subheader('Choose your confidence threshold')
        scored_name=st.selectbox('Scored interaction network',[name for name in ['huMAP2','BioPlex','HuRI'] if name in datasets],key='confidence_network')
        score=score_column(scored_name)
        if scored_name == 'huMAP2':
            st.write('The huMAP2 confidence score (0–1) estimates support for a protein-pair association: higher scores indicate greater confidence. Keep pairs whose score is at least the selected cutoff. This is not a statistical p-value.')
        elif scored_name == 'BioPlex':
            st.write('BioPlex pInt, labelled p(Interaction) in the source files, is the model’s interaction confidence score (0–1). Keep protein pairs with pInt at least the selected cutoff. It is not a statistical p-value, and it is not calibrated to the same scale as huMAP2 scores.')
        else:
            st.write('HuRI PSI-MITAB supplies an author score (0–1) in its confidence column for scored interactions, while some interactions have no score (-). The base HuRI network retains all valid pairs (keeping missing scores as null); confidence-filtered versions keep pairs with an author score at least the selected cutoff.')
        st.caption('When source pairs map to the same pair under the active isoform policy, retain the highest confidence score.')
        parent=datasets[scored_name]; info=parent.stat()
        if scored_name == 'HuRI' and score not in pl.read_parquet_schema(parent) and network_source_path(DATA, 'HuRI').is_file():
            with st.spinner('Updating HuRI views with PSI-MITAB author scores…'):
                prepare_huri_psi(DATA)
                datasets = discover_datasets(DATA, isoforms=isoforms)
                parent = datasets[scored_name]
                info = parent.stat()
        if score not in pl.read_parquet_schema(parent):
            st.warning(f'The canonical {scored_name} network does not yet preserve its confidence score. Rebuild this canonical network before filtering.')
        else:
            if scored_name == 'huMAP2':
                standard=st.selectbox('Standard confidence cutoff', [0.05, 0.95, 'Custom'],key='confidence_standard_'+scored_name)
                cutoff=st.number_input('Minimum interaction confidence score',min_value=0.0,max_value=1.0,value=0.05,step=0.05,format='%.4f',key='confidence_custom_'+scored_name) if standard == 'Custom' else float(standard)
            else:
                cutoff=st.number_input('Minimum interaction confidence score',min_value=0.0,max_value=1.0,value=0.5,step=0.05,format='%.4f',key='confidence_custom_'+scored_name)
            version_label=str(float(cutoff)).removesuffix('.0') if cutoff else '0'
            st.caption(f'Saved name: {scored_name}_{version_label}. Filtering includes scores equal to the cutoff.')
            if st.button('Save filtered network to disk',type='primary'):
                with st.spinner('Writing network and provenance…'):
                    output=save_scored_version(scored_name,cutoff,DATA,isoforms=isoforms)
                st.session_state['saved_isoform_version']=(isoforms, output.stem)
                st.rerun()
            saved_policy, saved_name = st.session_state.get('saved_isoform_version', (None, None))
            if saved_policy == isoforms and saved_name in datasets:
                st.success(f'Saved {saved_name} for the active isoform policy.')
            if scored_name == 'huMAP2':
                st.warning('The unfiltered huMAP2 base includes the full scored candidate list, including low-confidence associations.')
            with st.spinner('Calculating threshold sensitivity…'):
                curve=curve_data(parent,info.st_mtime_ns,info.st_size,scored_name)
            log_counts=st.toggle('Logarithmic count axes',value=True)
            fig=make_subplots(rows=1,cols=2,subplot_titles=['Edges retained','Nodes with at least one retained edge'])
            for i, metric in enumerate(['edges','nodes'],1):
                fig.add_trace(go.Scatter(x=curve['threshold'],y=curve[metric],mode='lines+markers',line_color='#555555',name=metric.title(),hovertemplate='Confidence score ≥ %{x:.2f}<br>'+metric+': %{y:,}<extra></extra>'),row=1,col=i)
                if scored_name == 'huMAP2':
                    for threshold, color in [(0.05, '#0072B2'), (0.95, '#B35C00')]:
                        count=int(curve.sort((pl.col('threshold')-threshold).abs())[metric][0])
                        fig.add_vline(x=threshold,line_dash='dash',line_color=color,row=1,col=i)
                        fig.add_trace(go.Scatter(x=[threshold],y=[count],mode='markers+text',marker=dict(size=11,color=color),text=[f'<b>{threshold:.2f}: {count:,}</b>'],textposition='top right' if threshold < .5 else 'top left',showlegend=False),row=1,col=i)
                fig.update_xaxes(title_text='Minimum interaction confidence score (≥)',row=1,col=i)
                fig.update_yaxes(title_text=metric.title()+' (count)',type='log' if log_counts else 'linear',row=1,col=i)
            fig.update_layout(template='plotly_white',title=f'{scored_name}: network size by confidence score',showlegend=False)
            st.plotly_chart(fig,width='stretch')
            st.download_button('Download threshold counts',curve.write_csv(),scored_name+'_threshold_curve.csv','text/csv')
            st.download_button('Download confidence plot (HTML)',fig.to_html(include_plotlyjs='cdn'),scored_name+'_confidence_coverage.html','text/html')
            saved_counts = DATA / 'plots' / f'{scored_name}_threshold_counts.csv'
            if saved_counts.is_file() and pl.read_csv(saved_counts).equals(curve):
                for extension, mime in [('png', 'image/png'), ('svg', 'image/svg+xml'), ('pdf', 'application/pdf')]:
                    plot_path = DATA / 'plots' / f'{scored_name}_threshold_retention.{extension}'
                    if plot_path.exists():
                        st.download_button(f'Download annotated plot ({extension.upper()})',
                                           plot_path.read_bytes(), plot_path.name, mime)
        st.write('Saved versions')
        saved=[{'Dataset':name,'File':str(path.relative_to(ROOT))} for name,path in datasets.items() if path.parent.name=='versions']
        if saved: st.dataframe(pl.DataFrame(saved),width='stretch',hide_index=True)
        else: st.caption('No threshold versions saved yet.')

if versions.open and not any(name in datasets for name in ['huMAP2', 'BioPlex', 'HuRI']):
    versions.info('Prepare a scored HuRI, BioPlex or huMAP2 network to explore confidence thresholds.')

if mapping_tab.open:
    with mapping_tab:
        with st.expander('Original endpoint identifier breakdown', expanded=False):
            render_original_ids()
        st.subheader('Trace each network to its source')
        st.write('The audit checks complete IDs first, including isoform suffixes, and retries unresolved isoforms with the parent accession. Every human UniProtKB match is shown.')
        st.caption(f'BioPlex, huMAP2 and {TAGC_NAME} use the saved UniProt mappings and the active isoform policy. '
                   'HuRI PSI views use the IDs supplied in the interactor columns. HuRI lookup results are shown here for comparison.')
        st.markdown('[UniProt ID Mapping tool](https://www.uniprot.org/mapping) · [Official API documentation](https://www.uniprot.org/help/id_mapping_prog)')
        if st.button('Refresh UniProt mapping and mapped networks', key='refresh_network_mapping'):
            try:
                with st.spinner('Resolving source IDs and rebuilding mapped networks…'):
                    ensure_swissprot_reference(DATA)
                    rebuild_mapped_networks(prepare_mapping(DATA), DATA)
                st.rerun()
            except Exception as exc:
                st.error(f'Mapping refresh failed. {exc}')

        humap_source = network_source_path(DATA, 'huMAP2')
        with st.expander('huMAP2 raw source and isoforms', expanded=True):
            if humap_source.is_file():
                info = humap_source.stat()
                preview, unique_ids, source_rows, isoform_rows, source_isoforms = humap_source_view(
                    humap_source, info.st_mtime_ns, info.st_size)
                st.write('The source is a headerless TSV. The preview labels its fields **a** and **b** for source IDs and **probability** for the native confidence score.')
                st.caption(f'{source_rows:,} source rows · {unique_ids:,} distinct IDs · '
                           f'{source_isoforms.height:,} distinct IDs with an isoform suffix · '
                           f'{isoform_rows:,} rows with at least one suffix')
                st.dataframe(preview, hide_index=True, alt='First 12 rows of the raw huMAP2 source')
                if source_isoforms.height:
                    st.dataframe(source_isoforms, hide_index=True, alt='Source IDs with an explicit isoform suffix')
                else:
                    st.info('Neither endpoint column contains an explicit numeric isoform suffix.')
            else:
                st.info(f'Place {humap_source.name} in data/raw_networks to inspect its raw columns.')

        report = DATA / 'mappings' / NETWORK_MAPPING
        if report.is_file():
            info = report.stat()
            reference = DATA / 'mappings' / 'swissprot_human_accessions.tsv'
            reference_stamp = None
            if reference.is_file():
                ref_info = reference.stat()
                reference_stamp = (reference, ref_info.st_mtime_ns, ref_info.st_size)
            table, statistics, created, releases, source_info = mapping_view(
                report, info.st_mtime_ns, info.st_size, reference_stamp)
            st.caption(f"Audit saved {created[:10]} UTC · UniProtKB release {', '.join(releases) or 'unavailable'}")
            for name in FILES:
                source = network_source_path(DATA, name)
                if source.is_file():
                    stat = source.stat()
                    previous = source_info.get(name, {})
                    if (stat.st_size, stat.st_mtime_ns) != (previous.get('size'), previous.get('mtime_ns')):
                        st.warning(f'The {name} source has changed since this audit. Refresh to include the current file.')
            release_stamps = []
            for path in bioplex_release_paths(DATA):
                if path.is_file():
                    stat = path.stat()
                    release_stamps.append((path.name, stat.st_size, stat.st_mtime_ns))
            audited_stamps = [(Path(item['path']).name, item['size'], item['mtime_ns'])
                              for item in source_info.get(BIOPLEX_ORIGINAL, [])]
            if release_stamps != audited_stamps:
                st.warning('Refresh the audit to include the current BioPlex release files.')
            st.subheader('Unique UniProt mapping results')
            st.caption('Every documented target is retained. Shared targets count once per source, so the number '
                       'of distinct targets can exceed the number of source IDs with a match. These are lookup '
                       'counts before interaction filters. The BioPlex releases and merged input are shown separately.')
            st.dataframe(statistics, hide_index=True, alt='Distinct UniProt and reviewed Swiss-Prot targets per source')
            metadata = reference.with_name('swissprot_human_source.json')
            if reference_stamp and metadata.is_file():
                reviewed_release = json.loads(metadata.read_text(encoding='utf-8')).get('uniprot_release')
                st.caption(f'Swiss-Prot counts use the reviewed human primary accession list '
                           f'from UniProt release {reviewed_release or "unknown"}.')
            elif not reference_stamp:
                st.info('Refresh the mapping to download the reviewed human Swiss-Prot reference.')
            counts = table.group_by('Network', 'Outcome').len().pivot(
                on='Outcome', index='Network', values='len').fill_null(0).sort('Network')
            st.dataframe(counts, hide_index=True, alt='Identifier mapping outcomes by network')
            network_filter, outcome_filter = st.columns(2)
            network_choice = network_filter.selectbox('Mapping network', ['All networks', *sorted(source_info)],
                                                      key='mapping_network')
            outcome = outcome_filter.selectbox('Mapping outcome', ['All outcomes', *sorted(table['Outcome'].unique())],
                                               key='mapping_outcome')
            query = st.text_input('Search source or current accession', key='mapping_query').strip()
            filtered = table
            if network_choice != 'All networks':
                filtered = filtered.filter(pl.col('Network') == network_choice)
            if outcome != 'All outcomes':
                filtered = filtered.filter(pl.col('Outcome') == outcome)
            if query:
                filtered = filtered.filter(pl.any_horizontal(
                    pl.col('Source identifier', 'Exact query', 'Parent query', 'UniProt matches', 'Primary accessions')
                    .str.to_uppercase().str.contains(query.upper(), literal=True)))
            st.caption(f'{filtered.height:,} source IDs across the selected networks. A shared ID has one row per network.')
            st.dataframe(filtered, hide_index=True, height=460, alt='UniProt lookup results for each source identifier')
            st.download_button('Download all mapping results (CSV)', table.write_csv(), 'network_uniprot_mapping.csv',
                               'text/csv', on_click='ignore')
            st.download_button('Download mapping provenance (JSON)', report.read_bytes(), report.name,
                               'application/json', on_click='ignore')
            with st.expander('Compare edges with isoforms kept distinct', on_change='rerun') as comparison:
                if comparison.open:
                    st.caption('Uses the same saved UniProtKB results across all four networks. '
                               'Explicit isoform matches and all documented targets are retained. '
                               'Pairs with an unmapped endpoint or a parent-only fallback are excluded.')
                    source_names = FILES
                    name = st.selectbox('Network for edge comparison', sorted(source_info.keys() & source_names.keys()))
                    source = network_source_path(DATA, name)
                    cutoff = 0.05
                    if name == 'huMAP2':
                        cutoff = st.number_input('Minimum huMAP2 probability', 0.0, 1.0, 0.05, step=0.01,
                                                 help='Applies to both sides of the comparison. Use 0 for all valid scores.')
                    if not source.is_file():
                        st.info(f'The local source file is missing: {source.name}')
                    else:
                        stat = source.stat()
                        previous = source_info[name]
                        stamp = (stat.st_size, stat.st_mtime_ns)
                        if stamp != (previous['size'], previous['mtime_ns']):
                            st.info('Refresh the mapping audit before comparing this source.')
                        else:
                            before, after, removed, added, preview, source_ids, mapped_ids = isoform_edge_view(
                                name, source, stamp, report, (info.st_size, info.st_mtime_ns), cutoff,
                            )
                            metrics = st.columns(4)
                            for col, label, value in zip(metrics, (
                                'Source pairs', 'Mapped pairs', 'Source pairs absent after mapping', 'New pair IDs',
                            ), (before, after, removed, added)):
                                col.metric(label, f'{value:,}')
                            st.caption('First 100 mapped pairs in accession order.')
                            st.dataframe(preview, hide_index=True, alt='Mapped undirected pairs with isoforms retained')
                            st.dataframe(pl.DataFrame({
                                'Metric': ['Distinct endpoint IDs', 'Distinct IDs after removing isoform suffixes'],
                                'Source': list(source_ids), 'Mapped': list(mapped_ids),
                            }), hide_index=True, alt='Source and mapped node and parent accession counts')
                            st.caption('Source counts can include Ensembl and GeneID identifiers. Mapped counts use UniProtKB.')
                            if st.button('Save mapped edge list for later'):
                                with st.spinner('Saving the mapped edges and provenance…'):
                                    saved_path = save_isoform_mapping_edges(name, source, report, cutoff)
                                st.success(f'Saved {saved_path.relative_to(ROOT)} and its provenance JSON.')
        else:
            st.info('Run the UniProt mapping refresh to populate the per-identifier table.')

        st.caption('Mapping results live in data/mappings. Prepared networks and audits live in data/processed_networks. '
                   'Dataset selectors show saved confidence versions that match the active isoform view.')
        for name in FILES:
            dataset = datasets.get(name)
            path = dataset.with_suffix('.json') if dataset is not None else None
            if path and path.exists():
                audit=json.loads(path.read_text())
                with st.expander(name+' mapping audit'):
                    st.json(audit)
                    st.download_button('Download '+name+' audit',json.dumps(audit,indent=2),name+'_mapping_audit.json','application/json')
if resources.open:
    with resources:
        st.subheader('HuRI · a systematic binary map')
        st.write('HuRI uses complementary yeast two-hybrid assays to systematically screen human protein pairs. Its binary design provides a reference for studying direct interactions, with assay-dependent coverage.')
        st.markdown('[Luck et al. (2020). A reference map of the human binary protein interactome. Nature 580, 402–408.](https://doi.org/10.1038/s41586-020-2188-x) · [Official downloads](https://interactome-atlas.org/download)')
        st.subheader('BioPlex · HEK293T protein associations')
        st.write('BioPlex uses affinity purification followed by mass spectrometry to identify proteins associated with tagged baits in human cells. The local file merges the three HEK293T releases; HCT116 is excluded. Co-purification can reflect membership in a shared complex rather than direct contact.')
        st.markdown('''Release papers:
- [Huttlin et al. (2015). The BioPlex Network: A Systematic Exploration of the Human Interactome. Cell.](https://doi.org/10.1016/j.cell.2015.06.043)
- [Huttlin et al. (2017). Architecture of the human interactome defines protein communities and disease networks. Nature.](https://doi.org/10.1038/nature22366)
- [Huttlin et al. (2021). Dual proteome-scale networks reveal cell-specific remodeling of the human interactome. Cell.](https://pubmed.ncbi.nlm.nih.gov/33961781/)

[BioPlex source downloads](https://bioplex.hms.harvard.edu/interactions.php)''')
        st.subheader('huMAP2 · integrated protein-complex evidence')
        st.write('hu.MAP 2.0 integrates more than 15,000 mass-spectrometry experiments using machine learning to infer protein associations and assemblies. This app reads the scored pairwise interaction file, rather than the final complex catalogue. The probability cutoff changes which pairs enter the graph.')
        st.markdown('[Drew et al. (2021). hu.MAP 2.0: integration of over 15,000 proteomic experiments builds a global compendium of human multiprotein assemblies. Molecular Systems Biology 17, e10016.](https://doi.org/10.15252/msb.202010016)')
        st.subheader(TAGC_NAME)
        st.write('The local CSV supplies two interactor identifiers per row. The app resolves them to human UniProtKB accessions and uses unique undirected pairs.')
        st.caption('Paper descriptions explain the resources. All displayed counts are calculated from your canonicalized local files and may differ from publication totals. huMAP2 integrates overlapping proteomics sources, so the networks are not independent evidence sets.')

if pipeline.open:
    with pipeline:
        st.subheader('Building the BioPlex HEK293T network')
        st.caption('The standalone merge script produces the merged TSV. The app then maps accessions according to the active isoform policy.')
        st.markdown('''1. Read HEK293T v1.0, v2.0 and v3.0 as text; align the nine source fields by column position and parse probabilities as numbers.
2. Concatenate releases and optionally filter on pInt (default: no cutoff).
3. Remove missing/empty UniProt endpoints and exact-accession self-interactions. Keep isoform suffixes.
4. Sort the endpoint accessions to create an undirected pair key.
5. Select the original row with the highest pInt per pair; prefer the newest release on score ties. Preserve winning-row orientation and fields.
6. Attach every supporting release, separated by `|`, and export the sorted TSV and Parquet files.
''')
        st.success('Verified: rebuilding with this code and the three local raw releases reproduces every parsed field and row of the copied merged TSV (141,310 interactions; default pInt cutoff).')
        st.code('python pipelines/bioplex_merge.py --bioplex-dir /path/to/bioplex_releases', language='bash')
        st.caption('The command writes merged TSV and Parquet beside the release inputs. Use a working copy of the release directory when rebuilding.')
        source = (ROOT / 'pipelines/bioplex_merge.py').read_text()
        with st.expander('View merge and deduplication code'):
            st.code(source, language='python', line_numbers=True)
        st.download_button('Download merge script', source, file_name='bioplex_merge.py', mime='text/x-python')
        with st.expander('View original source-download script'):
            st.code((ROOT / 'pipelines/download_bioplex.sh').read_text(), language='bash')
