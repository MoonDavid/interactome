"""Canonical human interaction networks, durable versions, and overlap exploration."""
from pathlib import Path
import importlib
import json
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
    build_canonical_networks, degree_distribution, discover_datasets,
    load_edges, load_network, membership_counts, prepare_huri_swissprot,
    prepare_mapping, save_scored_version, score_column, summarize_network,
    threshold_curve, upset_figure, venn_figure,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
st.set_page_config(page_title='Network Explorer | Human Interactome Atlas', page_icon=':material/hub:', layout='wide')
st.markdown('<style>.block-container{padding-top:2.5rem}</style>', unsafe_allow_html=True)
st.caption('HUMAN INTERACTOME ATLAS')
st.title('Explore the human interactome')
st.write('Compare HuRI, BioPlex HEK293T and huMAP2 in a shared UniProt accession space. Save confidence-filtered networks and revisit them across sessions.')

@st.cache_data(show_spinner=False, max_entries=16)
def analyze(path_string, mtime, size):
    loaded = load_network(Path(path_string))
    return summarize_network(loaded), degree_distribution(loaded)

@st.cache_data(show_spinner=False, max_entries=16)
def overlap_data(specs, mode):
    return membership_counts({name: load_edges(path) for name, path, mtime, size in specs}, mode)

@st.cache_data(show_spinner=False, max_entries=2)
def curve_data(mtime, size, name):
    return threshold_curve(DATA, name=name)

datasets = discover_datasets(DATA)
if not ('HuRI' in datasets and 'BioPlex' in datasets and any(name == 'huMAP2' or name.startswith('huMAP2_') for name in datasets)):
    st.info('Prepare the canonical networks using the official UniProt ID Mapping service. Mapping results and network files are cached on disk.')
    st.code('.venv/bin/python networks.py', language='bash')
    if st.button('Map IDs with UniProt and prepare networks', type='primary'):
        try:
            with st.spinner('Mapping identifiers and preparing canonical networks…'):
                mapping = prepare_mapping(DATA)
                build_canonical_networks(mapping, DATA, huri_mapping=prepare_huri_swissprot(mapping, DATA))
            st.rerun()
        except Exception as exc:
            st.error(f'Preparation failed: {exc}. Existing source files are unchanged; retry to reuse completed mapping results.')
    st.stop()

with st.sidebar:
    st.title('Network Explorer')
    st.caption('Canonical UniProt accessions')
    active = st.multiselect('Networks to explore', list(datasets), default=list(datasets))
    st.caption('Saved huMAP2 versions are automatically discovered from disk on every launch.')

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
                a,b=st.columns(2)
                a.metric('Nodes', f"{results[name]['nodes']:,}")
                b.metric('Edges', f"{results[name]['edges']:,}")
                st.caption('Canonical UniProt · undirected pairs')
st.caption('HuRI uses reviewed human UniProtKB/Swiss-Prot matches; other networks use human UniProtKB mappings. Multiple matches are expanded. Isoforms are collapsed to their parent accession. Unmapped IDs, self-loops and duplicate pairs are excluded. AP-MS/co-complex associations and binary assay interactions have different biological meanings.')

huri_audit=json.loads((datasets['HuRI'].with_suffix('.json')).read_text())
if huri_audit.get('one_to_many_id_count',0):
    st.info(f"HuRI retains reviewed Swiss-Prot matches. {huri_audit['one_to_many_id_count']:,} source genes have multiple reviewed matches; expanded pairs are possible assignments, not additional experimentally observed interactions. See the Mapping & provenance for exclusions.")

overview, overlap, versions, mapping_tab, resources, pipeline = st.tabs(['Network landscape','Shared connections','Confidence & coverage','Mapping & provenance','Interactome descriptions & data sources','BioPlex preparation'])
with overview:
    if results:
        st.subheader('The network landscape at a glance')
        fields = {'nodes':'Nodes','edges':'Unique edges','mean_degree':'Mean degree','density':'Density','components':'Components','largest_component':'Largest component (nodes)','largest_component_fraction':'Largest component fraction'}
        st.dataframe(pl.DataFrame({'Metric':list(fields.values()), **{name:pl.Series(name,[result[key] for key in fields],dtype=pl.Float64) for name,result in results.items()}}), width='stretch', hide_index=True)
        st.caption('Nodes are endpoints of retained edges; isolated proteins absent from the edge lists are not counted. Density = 2E / N(N−1); mean degree = 2E/N.')
        frames=[]
        for name, frame in degrees.items():
            frames.append(frame.with_columns(pl.lit(name).alias('Network')))
        fig=px.scatter(pl.concat(frames), x='degree', y='count', color='Network', log_x=True, log_y=True, labels={'degree':'Degree (interaction partners)','count':'Nodes at this degree'}, title='How connections are distributed across proteins')
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, width='stretch')
    else:
        st.info('Select datasets in the sidebar to show their statistics.')

with overlap:
    st.subheader('Interactome overlap')
    default_humap = 'huMAP2' if 'huMAP2' in datasets else next((name for name in ['huMAP2_0.05', 'huMAP2_0.95'] if name in datasets), None)
    selected=st.multiselect('Datasets to compare', list(datasets), default=[name for name in ['HuRI','BioPlex',default_humap] if name in datasets], max_selections=12)
    c1,c2=st.columns(2)
    mode=c1.radio('Compare', ['Edges','Nodes'], horizontal=True)
    style=c2.radio('Diagram', ['Venn','UpSet'], horizontal=True)
    st.caption('Counts describe exclusive regions: each node or undirected edge appears in exactly one combination. Venn percentages use the union of the selected networks as the denominator. Hover over a region or bar for included and excluded datasets.')
    if selected:
        if style=='Venn' and len(selected) not in (2,3):
            st.info('Venn supports two or three datasets. Choose UpSet for one or more than three.')
        else:
            specs=tuple((name,str(datasets[name]),datasets[name].stat().st_mtime_ns,datasets[name].stat().st_size) for name in selected)
            with st.spinner('Computing exact intersections…'):
                counts=overlap_data(specs,mode)
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

with versions:
    st.subheader('Choose your confidence threshold')
    scored_name=st.selectbox('Scored interaction network',[name for name in ['huMAP2','BioPlex'] if name in datasets],key='confidence_network')
    score=score_column(scored_name)
    if scored_name == 'huMAP2':
        st.write('The huMAP2 confidence score (0–1) estimates support for a protein-pair association: higher scores indicate greater confidence. Keep pairs whose score is at least the selected cutoff. This is not a statistical p-value.')
    else:
        st.write('BioPlex pInt, labelled p(Interaction) in the source files, is the model’s interaction confidence score (0–1). Keep protein pairs with pInt at least the selected cutoff. It is not a statistical p-value, and it is not calibrated to the same scale as huMAP2 scores.')
    st.caption('When source pairs map to the same canonical pair, retain the highest confidence score. HuRI has no confidence score.')
    parent=datasets[scored_name]; info=parent.stat()
    if score not in pl.read_parquet_schema(parent):
        st.warning(f'The canonical {scored_name} network does not yet preserve its confidence score. Rebuild this canonical network using the saved UniProt mapping before filtering.')
    else:
        standard=st.selectbox('Standard confidence cutoff', [0.05, 0.95, 'Custom'],key='confidence_standard_'+scored_name)
        cutoff=st.number_input('Minimum interaction confidence score',min_value=0.0,max_value=1.0,value=0.05,step=0.05,format='%.4f',key='confidence_custom_'+scored_name) if standard == 'Custom' else float(standard)
        version_label=str(float(cutoff)).removesuffix('.0') if cutoff else '0'
        st.caption(f'Saved name: {scored_name}_{version_label}. Filtering includes scores equal to the cutoff.')
        if st.button('Save filtered network to disk',type='primary'):
            with st.spinner('Writing network and provenance…'):
                output=save_scored_version(scored_name,cutoff,DATA)
            st.session_state['saved_version']=output.stem
            st.rerun()
        if st.session_state.get('saved_version'):
            st.success(f"Saved {st.session_state['saved_version']}; available in the dataset selectors and on future launches.")
        if scored_name == 'huMAP2':
            st.warning('The unfiltered huMAP2 base includes the full scored candidate list, including low-confidence associations.')
        with st.spinner('Calculating threshold sensitivity…'):
            curve=curve_data(info.st_mtime_ns,info.st_size,scored_name)
        log_counts=st.toggle('Logarithmic count axes',value=True)
        fig=make_subplots(rows=1,cols=2,subplot_titles=['Edges retained','Nodes with at least one retained edge'])
        for i, metric in enumerate(['edges','nodes'],1):
            fig.add_trace(go.Scatter(x=curve['threshold'],y=curve[metric],mode='lines+markers',line_color='#555555',name=metric.title(),hovertemplate='Confidence score ≥ %{x:.2f}<br>'+metric+': %{y:,}<extra></extra>'),row=1,col=i)
            for threshold, color in [(0.05, '#0072B2'), (0.95, '#B35C00')]:
                count=int(curve.sort((pl.col('threshold')-threshold).abs())[metric][0])
                fig.add_vline(x=threshold,line_dash='dash',line_color=color,row=1,col=i)
                fig.add_trace(go.Scatter(x=[threshold],y=[count],mode='markers+text',marker=dict(size=11,color=color),text=[f'<b>{threshold:.2f}: {count:,}</b>'],textposition='top right' if threshold < .5 else 'top left',showlegend=False),row=1,col=i)
            fig.update_xaxes(title_text='Minimum interaction confidence score (≥)',row=1,col=i)
            fig.update_yaxes(title_text=metric.title()+' (count)',type='log' if log_counts else 'linear',row=1,col=i)
        fig.update_layout(template='plotly_white',title=f'{scored_name}: network size by confidence score',showlegend=False)
        st.plotly_chart(fig,width='stretch')
        st.download_button('Download threshold counts',curve.write_csv(),scored_name+'_threshold_curve.csv','text/csv')
        st.download_button('Download confidence plot (HTML)',fig.to_html(include_plotlyjs=True),scored_name+'_confidence_coverage.html','text/html')
        for extension, mime in [('png', 'image/png'), ('svg', 'image/svg+xml'), ('pdf', 'application/pdf')]:
            plot_path=DATA / 'plots' / f'{scored_name}_threshold_retention.{extension}'
            if plot_path.exists():
                st.download_button(f'Download annotated plot ({extension.upper()})',plot_path.read_bytes(),plot_path.name,mime)
    st.write('Saved versions')
    saved=[{'Dataset':name,'File':str(path.relative_to(ROOT))} for name,path in datasets.items() if path.parent.name=='versions']
    if saved: st.dataframe(pl.DataFrame(saved),width='stretch',hide_index=True)
    else: st.caption('No threshold versions saved yet.')

with mapping_tab:
    st.subheader('Trace each network to its source')
    st.markdown('[UniProt ID Mapping tool](https://www.uniprot.org/mapping) · [Official API documentation](https://www.uniprot.org/help/id_mapping_prog)')
    st.write('Ensembl and GeneID → human UniProtKB primary accession; sp|accession|name wrappers are parsed before mapping. HuRI is restricted to reviewed human Swiss-Prot accessions; all reviewed matches are retained when a gene has several. Gene-level interactions may expand into multiple accession pairs. Isoform suffix removal groups at the UniProt entry level; it does not infer which sequence isoform mediated an interaction.')
    swiss_summary = DATA / 'mappings/HuRI_swissprot_summary.json'
    swiss_audit = DATA / 'mappings/HuRI_swissprot_audit.tsv'
    if swiss_summary.exists() and swiss_audit.exists():
        st.subheader('HuRI: reviewed accession coverage')
        st.json(json.loads(swiss_summary.read_text()))
        st.download_button('Download per-gene HuRI mapping audit', swiss_audit.read_bytes(), 'HuRI_swissprot_audit.tsv', 'text/tab-separated-values')
        st.caption('The audit lists retained reviewed accessions and excluded candidate accessions for each source gene. Excluded candidates are not in the reviewed reference list; they may be unreviewed or absent from the current reference.')
        audit_table = pl.read_csv(swiss_audit, separator='\t').with_columns(pl.col(pl.String).fill_null(''))
        audit_view = st.selectbox('Show HuRI genes', ['Without a reviewed match', 'With excluded candidates', 'All genes'])
        if audit_view == 'Without a reviewed match':
            audit_table = audit_table.filter(pl.col('reviewed_count') == 0)
        elif audit_view == 'With excluded candidates':
            audit_table = audit_table.filter(pl.col('excluded_count') > 0)
        st.dataframe(audit_table, width='stretch', hide_index=True)
    st.caption('Mappings and mapping job IDs are saved in data/mappings. Canonical files and per-network audits live in data/canonical. Raw inputs remain in data/networks. Saved versions carry source fingerprints; versions from an older canonical base are excluded from automatic discovery.')
    for name in ['HuRI','BioPlex','huMAP2']:
        path=DATA/'canonical'/f'{name}.json'
        if path.exists():
            audit=json.loads(path.read_text())
            with st.expander(name+' mapping audit'):
                st.json(audit)
                st.download_button('Download '+name+' audit',json.dumps(audit,indent=2),name+'_mapping_audit.json','application/json')
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
    st.caption('Paper descriptions explain the resources. All displayed counts are calculated from your canonicalized local files and may differ from publication totals. huMAP2 integrates overlapping proteomics sources, so the networks are not independent evidence sets.')

with pipeline:
    st.subheader('Building the BioPlex HEK293T network')
    st.caption('Preserved implementation from the previous app, now available as a standalone script. This produces the raw merged TSV. The overview subsequently maps its accessions and collapses isoforms; those additional transformations are documented in the Mapping & provenance.')
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
