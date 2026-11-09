"""Independent, reproducible moonlighting analyses against whole interactomes."""
from pathlib import Path
import json
import hashlib
import sys
import polars as pl
import plotly.express as px
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from networks import discover_datasets, map_ids
from moonlighting import (
    GROUP_COLORS, GROUP_LABELS, SIGNIFICANCE_LEGEND, SOURCES,
    centrality_analysis, centrality_heatmap, cohort_ids, cohesion_analysis,
    community_analysis, compare_source_sets, cross_network_analysis,
    go_enrichment, graph_from_path, hub_enrichment_heatmap, load_moon_data,
    mapping_coverage, metric_figure, prepare_go_annotations,
    random_walk_analysis, robustness_analysis, save_analysis,
    significance_label, wiring_analysis,
)
DATA=ROOT/'data'

def setting(widget,*args,key,**kwargs):
    stored=st.session_state.setdefault('moon_analysis_settings',{})
    if key in stored:
        saved=st.session_state.get(key,stored[key])
        positional=list(args)
        if widget == st.selectbox:
            options=positional[1] if len(positional)>1 else kwargs['options']
            if saved in options: kwargs['index']=list(options).index(saved)
            else:
                st.session_state.pop(key,None)
                stored.pop(key,None)
        else:
            position=1 if widget == st.checkbox else 3
            if len(positional)>position: positional[position]=saved
            else: kwargs['value']=saved
        args=tuple(positional)
    value=widget(*args,key=key,**kwargs)
    stored[key]=value
    return value

st.set_page_config(page_title='Moonlighting Explorer | Human Interactome Atlas',page_icon=':material/hub:',layout='wide')
st.caption('HUMAN INTERACTOME ATLAS')
st.title('Explore the network roles of moonlighting proteins')
st.write('Map AllMoon + MoonLitDB cohorts onto each interactome, compare their topology with the network background, and explore functional enrichment and diffusion.')
st.caption('Each tab runs only when selected. Each analysis has its own Run button and stored results. All network proteins define the background; background labels mean unlabelled proteins, not proven non-moonlighting controls.')
datasets=discover_datasets(DATA)
if not datasets:
    st.info('Prepare canonical networks on the overview page first.'); st.stop()

@st.cache_data(show_spinner=False)
def moon_data(mtime,mapping_mtime):
    return load_moon_data(DATA)

@st.cache_resource(show_spinner=False,max_entries=2)
def cached_graph(path,mtime,size):
    return graph_from_path(Path(path))

mapping_file=DATA/'mappings/moonlighting_uniprot.json'
if not mapping_file.exists():
    st.info('Resolve the moonlighting protein IDs before running the analyses.')
    if st.button('Map moonlighting accessions with UniProt'):
        with st.spinner('Resolving UniProt IDs…'):
            raw=pl.read_csv(DATA/'allmoon_with_moonlitdb_human.csv',schema_overrides={'uniprot':pl.String})
            result=map_ids(raw['uniprot'].drop_nulls().str.split('-').list.first().to_list(),'UniProtKB_AC-ID',DATA/'mappings')
            mapping_file.write_text(json.dumps(result,indent=2))
        st.rerun()
    st.stop()
raw_path=DATA/'allmoon_with_moonlitdb_human.csv'
moon=moon_data(raw_path.stat().st_mtime_ns,mapping_file.stat().st_mtime_ns)
with st.sidebar:
    st.title('Moonlighting Explorer')
    network=setting(st.selectbox,'Interactome',list(datasets),key='moon_network')
    with st.expander('Combined-cohort controls (other analyses)'):
        st.caption('Compare source sets is the default view. These combined-cohort controls apply to the other analysis tabs.')
        sources=st.multiselect('Combined-cohort sources (empty = all rows)',SOURCES,default=['moonprot','multitask','moonlitdb'],format_func=lambda source:GROUP_LABELS.get(source,source),key='moon_sources')
        all_sources=setting(st.checkbox,'Require all selected sources',key='moon_all_sources')
    reviewed_only=setting(st.checkbox,'Reviewed source entries only',key='moon_reviewed')
    min_pubs=setting(st.number_input,'Minimum MoonLitDB publications',0,10000,0,1,key='moon_min_pubs')
    seed=setting(st.number_input,'Random seed',0,2**31-1,42,1,key='moon_seed')
    st.caption('Source flags may overlap. Source review status comes from the input CSV; IDs are resolved against saved UniProt mappings.')
cohort,selected=cohort_ids(moon,sources,all_sources,reviewed_only,min_pubs)
path=datasets[network]
stat=path.stat()
context={'network':network,'network_path':str(path),'network_mtime_ns':stat.st_mtime_ns,
         'network_size':stat.st_size,'cohort':sorted(cohort),'sources':sources,'source_mode':'intersection' if all_sources else 'union',
         'reviewed_source_only':reviewed_only,'min_moonlitdb_publications':int(min_pubs),'random_seed':int(seed),
         'moon_csv_mtime_ns':raw_path.stat().st_mtime_ns,'moon_mapping_mtime_ns':mapping_file.stat().st_mtime_ns}
meta_path=path.with_suffix('.json')
if meta_path.exists(): context['network_provenance']=json.loads(meta_path.read_text())
with st.expander('Combined-cohort counts (other analyses)'):
    cols=st.columns(3)
    cols[0].metric('Selected source accessions',selected['input_accession'].n_unique())
    cols[1].metric('Resolved canonical IDs',len(cohort))
    cols[2].metric('Source rows without mapping',selected.filter(pl.col('mapping_status')=='unmapped')['input_accession'].n_unique())
if not cohort:
    st.warning('No mapped proteins satisfy the combined-cohort filters. You can still choose source sets separately in Compare source sets.')

def graph_for(name=None):
    p=datasets[name or network]; info=p.stat()
    g=cached_graph(str(p),info.st_mtime_ns,info.st_size)
    if not len(g['nodes']): raise ValueError('The selected network has no retained edges or nodes.')
    return g

def run_panel(name,parameters,compute):
    """A result belongs to one analysis, one graph, one cohort, one parameter set."""
    metadata=dict(context,analysis=name,parameters=parameters,
                  correction_scope=parameters.get('correction_scope','Per analysis and selected network; no correction across separately requested runs'))
    signature=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    store=st.session_state.setdefault('moon_analysis_results',{})
    if st.button('Run '+name.replace('_',' '),key='run_'+name,type='primary'):
        try:
            with st.spinner('Running '+name.replace('_',' ')+'…'):
                result=compute()
                saved,archive=save_analysis(result,metadata,DATA)
            store[name]={'signature':signature,'result':result,'file':str(saved),'archive':archive}
        except Exception as exc:
            st.error(f'{name.replace("_"," ").title()} failed: {exc}')
    previous=store.get(name)
    if previous and previous['signature']!=signature:
        st.info('Inputs changed. Run this analysis again to calculate results for the current settings.'); return None
    if previous:
        st.download_button('Download tables, parameters and provenance (ZIP)',previous['archive'],Path(previous['file']).name,'application/zip',key='download_'+name)
        st.caption('Saved run: '+str(Path(previous['file']).relative_to(ROOT)))
        return previous['result']
    st.caption('Ready to run. Other tabs will retain their own results.'); return None

def table(frame,key):
    st.dataframe(frame,width='stretch',hide_index=True,key=key)

def summary(result):
    with st.expander('Methods, counts and convergence',expanded=True):
        st.json(result.get('summary',result.get('notes',{})))

def centrality_overview(result,prefix):
    st.subheader('Centrality signatures across moonlighting sets')
    st.write('Read across a row to see which centrality measures distinguish a set. Orange indicates higher values than the shared background; blue indicates lower values. The rank-biserial effect size runs from −1 to +1, so measures with different units can be compared.')
    profiles=result['profiles']
    fig=centrality_heatmap(profiles)
    st.plotly_chart(fig,width='stretch',key=prefix+'_centrality_heatmap')
    st.download_button('Download centrality heatmap (HTML)',fig.to_html(include_plotlyjs=True),prefix+'_centrality_heatmap.html','text/html',key=prefix+'_centrality_heatmap_download')
    st.download_button('Download centrality effects (CSV)',profiles.write_csv(),prefix+'_centrality_effects.csv','text/csv',key=prefix+'_effects_csv')
    st.caption(SIGNIFICANCE_LEGEND)
    st.subheader('Enrichment among high-centrality proteins')
    st.write('For each measure, hubs are proteins at or above the network’s 90th percentile. Ties are included, so the selected fraction can exceed 10%. Enrichment compares each set’s hub fraction with the full network’s hub fraction: 1× is the network baseline.')
    hubs=result['hub_enrichment']
    hub_fig=hub_enrichment_heatmap(hubs)
    st.plotly_chart(hub_fig,width='stretch',key=prefix+'_hub_heatmap')
    st.download_button('Download hub enrichment heatmap (HTML)',hub_fig.to_html(include_plotlyjs=True),prefix+'_hub_enrichment.html','text/html',key=prefix+'_hub_heatmap_download')
    st.download_button('Download hub enrichment tests (CSV)',hubs.write_csv(),prefix+'_hub_enrichment.csv','text/csv',key=prefix+'_hubs_csv')
    st.caption('Hub stars use one-sided hypergeometric overrepresentation tests, BH-corrected across all sets and measures, including Union even when hidden. This is a separate test family from the centrality-shift tests. Overlapping sets and graph structure limit independent interpretation; these tests do not control for degree.')
    with st.expander('Effect sizes and hub enrichment statistics'):
        table(profiles,prefix+'_profiles')
        table(hubs,prefix+'_hubs')

def metric_plot(result,prefix):
    frame=result['metrics']
    numeric=[c for c,dtype in frame.schema.items() if dtype.is_numeric() and c != 'community']
    if not numeric: return
    metric=setting(st.selectbox,'Metric to inspect',numeric,key=prefix+'_metric')
    discrete=(frame.schema[metric].is_integer() or metric in {'degree','core_number','ego_components','triangles','open_wedges'})
    distribution=st.radio('Distribution style',['Box','Violin'],index=0 if discrete else 1,horizontal=True,key=prefix+'_distribution_'+metric).lower()
    plot_columns=st.columns(2) if frame['group'].n_unique() <= 5 else [st.container(),st.container()]
    for column, kind in zip(plot_columns, [distribution,'bar']):
        with column:
            fig=metric_figure(frame,metric,result.get('tests'),plot_kind=kind)
            st.plotly_chart(fig,width='stretch',key=prefix+'_'+kind)
            st.download_button(f'Download {kind} plot (HTML)',fig.to_html(include_plotlyjs=True),prefix+'_'+metric+'_'+kind+'.html','text/html',key=prefix+'_'+kind+'_download')
    observed=metric_figure(frame,metric,result.get('tests'),plot_kind='observed')
    st.plotly_chart(observed,width='stretch',key=prefix+'_observed')
    st.download_button('Download observed-value plot (HTML)',observed.to_html(include_plotlyjs=True),prefix+'_'+metric+'_observed.html','text/html',key=prefix+'_observed_download')
    st.caption(SIGNIFICANCE_LEGEND)
    st.caption('Violin shapes sample up to 5,000 proteins per set. Boxes, bar means, descriptive SEM error bars and dashed observed-median lines use all finite observations. The histogram shows background protein values, not a permutation null. Comparisons do not control for degree or network dependence.')
    if 'tests' in result: table(result['tests'],prefix+'_tests')
    with st.expander('Per-protein metrics'): table(frame,prefix+'_metrics')

labels=['Cohort & coverage','Network roles','Cohort cohesion','Local connections','Community map','Diffusion & linkers','Functional enrichment','Network resilience','Cross-network comparison','Compare source sets']
ordered_labels=[labels[-1],*labels[:-1]]
containers=st.tabs(ordered_labels,key='moon_analysis_tab',on_change='rerun')
tabs_by_label=dict(zip(ordered_labels,containers))
tabs=[tabs_by_label[label] for label in labels]
if tabs[0].open:
    with tabs[0]:
        st.subheader('Locate your cohort in '+network)
        st.write('Coverage is the intersection of resolved cohort accessions and all endpoint proteins in the chosen network. Absence from a network does not establish absence of an interaction or function.')
        result=run_panel('mapping_coverage',{},lambda:mapping_coverage(graph_for(),selected,cohort))
        if result:
            summary(result); table(result['audit'],'coverage_audit')
        with st.expander('Source cohort counts and metadata'):
            counts=pl.DataFrame({'Source':SOURCES,'Source accessions':[moon.filter(pl.col(source))['input_accession'].n_unique() for source in SOURCES]})
            table(counts,'source_counts')
            columns=['input_accession','canonical_id','Gene Names','Reviewed']+SOURCES+['moonlitdb_number_of_publications','moonlitdb_number_of_moonlighting_functions']
            table(selected.select(columns),'selected_metadata')
        st.caption('UniProt mapping cache: data/mappings/moonlighting_uniprot.json. Unmapped accessions are retained in the audit.')

if tabs[1].open:
    with tabs[1]:
        st.subheader('Identify hubs, bridges and influential proteins')
        extra=setting(st.checkbox,'Include core number, clustering, eigenvector, approximate betweenness and bridging centrality',key='cent_extra')
        landmarks=setting(st.number_input,'Betweenness landmarks (approximation)',8,256,32,8,key='cent_landmarks')
        st.caption('Basic metrics: degree, weighted strength, PageRank and mean neighbor degree. PageRank uses probability affinities when available. Advanced metrics use unweighted topology and require ≤500,000 edges. Comparisons use two-sided Mann–Whitney U and BH across metrics; degree bias and network dependence affect interpretation.')
        result=run_panel('centrality',{'advanced':extra,'landmarks':int(landmarks)},lambda:centrality_analysis(graph_for(),cohort,extra=extra,seed=int(seed),landmarks=int(landmarks)))
        if result: summary(result); metric_plot(result,'cent')

if tabs[2].open:
    with tabs[2]:
        st.subheader('Measure how closely the cohort connects')
        perms=setting(st.number_input,'Cohesion permutations',10,1000,100,10,key='cohesion_perms')
        matched=setting(st.checkbox,'Match the cohort degree distribution',value=True,key='cohesion_match')
        st.caption('The null samples without replacement from the full network, matching cohort size and floor(log₂ degree) counts when enabled. Original cohort nodes remain eligible. Upper-tail empirical p values use (extreme + 1)/(permutations + 1); BH covers internal-edge and LCC tests. This matches degree bins, not individual degrees.')
        result=run_panel('cohesion',{'permutations':int(perms),'degree_matched':matched},lambda:cohesion_analysis(graph_for(),cohort,permutations=int(perms),seed=int(seed),degree_matched=matched))
        if result:
            summary(result)
            for metric in ['internal_edges','largest_component']:
                fig=px.histogram(result['null'],x=metric,nbins=25,title=metric.replace('_',' ').title()+' under the null',labels={metric:metric.replace('_',' ').title()})
                fig.add_vline(x=result['summary'][metric],line_color='#bb6c3c',annotation_text='Observed: '+significance_label(result['summary'].get(metric+'_fdr_bh')))
                st.plotly_chart(fig,width='stretch',key='cohesion_'+metric)
            st.caption('Stars use BH-adjusted empirical p-values across the two cohesion tests: * < 0.05; ** < 0.01; *** < 0.001; **** < 0.0001; ns ≥ 0.05. The reference is the selected random-set null model.')

if tabs[3].open:
    with tabs[3]:
        st.subheader('Explore each protein’s local neighborhood')
        count=setting(st.number_input,'Maximum cohort proteins sampled',10,500,100,10,key='wiring_count')
        st.caption('Exact ego-neighborhood clustering, triangles, open wedges, effective size, component count and component entropy. A same-sized complement control sample matches log₂-degree bins. Missing degree-matched controls are reported explicitly. These patterns suggest structural roles; they do not establish binding exclusivity. ≤500,000 edges.')
        result=run_panel('local_wiring',{'max_nodes':int(count)},lambda:wiring_analysis(graph_for(),cohort,max_nodes=int(count),seed=int(seed)))
        if result: summary(result); metric_plot(result,'wiring')

if tabs[4].open:
    with tabs[4]:
        st.subheader('Map proteins within and across communities')
        resolution=setting(st.number_input,'Louvain resolution',min_value=0.1,max_value=5.0,value=1.0,step=0.1,key='community_resolution')
        st.caption('Louvain partitions the full unweighted graph into non-overlapping modules. Participation and within-module z scores describe cross-module roles. Hypergeometric community enrichment uses all network proteins, with BH across every community. ≤500,000 edges.')
        result=run_panel('communities',{'resolution':float(resolution)},lambda:community_analysis(graph_for(),cohort,seed=int(seed),resolution=float(resolution)))
        if result:
            summary(result)
            fig=px.scatter(result['metrics'],x='within_module_z',y='participation',color='group',color_discrete_map=GROUP_COLORS,hover_data=['protein','community'],title='Within-module position and cross-module participation',labels={'within_module_z':'Within-module degree z score','participation':'Participation coefficient'})
            st.plotly_chart(fig,width='stretch')
            table(result['enrichment'],'community_enrichment'); metric_plot(result,'community')

if tabs[5].open:
    with tabs[5]:
        st.subheader('Follow network diffusion to candidate linkers')
        restart=setting(st.slider,'Restart probability',0.05,0.95,0.3,0.05,key='walk_restart')
        weighted=setting(st.checkbox,'Use huMAP probabilities as transition affinities',key='walk_weighted')
        walk_perms=setting(st.number_input,'Degree-matched score permutations (0 = ranking only)',0,50,0,5,key='walk_perms')
        st.caption('Seeds are mapped cohort proteins. RWR scores are probabilities, not p values; lift divides by a uniform-teleport baseline. Linker hypergeometric tests count seed neighbors and use BH across all nonseed candidates. Optional empirical score p values use matched random seeds and a +1 correction; at most 50 permutations and ≤500,000 edges. Predictions are exploratory and are not validated moonlighting labels.')
        result=run_panel('random_walk',{'restart':float(restart),'weighted':weighted,'permutations':int(walk_perms)},lambda:random_walk_analysis(graph_for(),cohort,restart=float(restart),weighted=weighted,permutations=int(walk_perms),seed=int(seed)))
        if result:
            summary(result)
            candidates=result['ranked'].filter(~pl.col('is_seed'))
            table(candidates.head(200),'walk_candidates')
            st.caption('Top 200 by RWR score shown; the ZIP includes all proteins and seed labels. Linker tests and walk rankings answer different questions.')
            fig=px.line(result['trace'],x='step',y=['seed_mass','nonseed_mass'],markers=True,title='Probability mass during seeded diffusion',labels={'step':'Restart-walk steps','value':'Probability mass','variable':'Group'})
            st.plotly_chart(fig,width='stretch')

if tabs[6].open:
    with tabs[6]:
        st.subheader('Discover enriched biological functions')
        aspect=setting(st.selectbox,'GO aspect',['BP','MF','CC','All'],key='go_aspect')
        min_size=setting(st.number_input,'Minimum term size in network background',1,1000,5,1,key='go_min_size')
        exclude_iea=setting(st.checkbox,'Exclude electronically inferred (IEA-only) annotations',key='go_iea')
        st.caption('One-sided hypergeometric test; full network population includes unannotated proteins. GO annotations propagate over is_a and part_of; NOT annotations are excluded. BH includes every eligible background term, including zero-hit terms. Local GOA/ontology sources date from October 2025; coverage and file provenance are included in every result.')
        def compute_go():
            annotation=prepare_go_annotations(DATA/'annotations')
            return go_enrichment(annotation,set(graph_for()['nodes']),cohort,aspect=aspect,min_size=int(min_size),exclude_iea=exclude_iea)
        go_inputs={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [DATA/'annotations/human_go_propagated.parquet',Path('/home/davide/data/GO/goa_human.gaf'),Path('/home/davide/data/GO/go-basic.obo')] if p.exists()}
        result=run_panel('go_enrichment',{'aspect':aspect,'min_size':int(min_size),'exclude_iea':exclude_iea,'annotation_inputs':go_inputs},compute_go)
        if result:
            summary(result)
            hits=result['table'].filter(pl.col('query_hits') > 0).head(25)
            if len(hits):
                hits=hits.with_columns((-pl.col('qvalue').clip(lower_bound=1e-300).log10()).alias('minus_log10_q'),pl.col('qvalue').map_elements(significance_label,return_dtype=pl.String).alias('significance'))
                fig=px.scatter(hits,x='fold',y='name',size='query_hits',color='minus_log10_q',hover_data=['term','pvalue','qvalue','background_hits'],text='significance',color_continuous_scale='Blues',title='Top GO terms by adjusted p value',labels={'fold':'Fold enrichment','name':'GO term','minus_log10_q':'−log₁₀ BH q'})
                fig.update_traces(textposition='middle right'); fig.update_yaxes(autorange='reversed'); fig.update_layout(height=750,margin=dict(r=100))
                st.plotly_chart(fig,width='stretch')
            else: st.info('No eligible term has a cohort hit under these filters.')
            st.caption('Stars use GO term BH-adjusted q-values: * < 0.05; ** < 0.01; *** < 0.001; **** < 0.0001; ns ≥ 0.05.')
            table(result['table'],'go_table')
        st.markdown('[GO enrichment background guidance](https://www.geneontology.org/docs/go-enrichment-analysis/) · [GO annotation rules](https://www.geneontology.org/docs/go-annotations/)')

if tabs[7].open:
    with tabs[7]:
        st.subheader('Test network resilience to cohort removal')
        removal_perms=setting(st.number_input,'Matched removal permutations (0 = observed only)',0,100,50,10,key='removal_perms')
        st.caption('Remove the mapped cohort and count the surviving largest connected component. Compare against equal-sized removals with matching log₂-degree bins. The lower-tail empirical p value tests unusually strong fragmentation. Fractions use original node count and surviving node count separately. Permutations require ≤500,000 edges.')
        result=run_panel('removal_robustness',{'permutations':int(removal_perms)},lambda:robustness_analysis(graph_for(),cohort,permutations=int(removal_perms),seed=int(seed)))
        if result:
            summary(result)
            if len(result['null']):
                fig=px.histogram(result['null'],x='remaining_lcc_nodes',title='Surviving LCC after degree-matched removals',labels={'remaining_lcc_nodes':'Nodes in surviving largest component'})
                fig.add_vline(x=result['summary']['remaining_lcc_nodes'],line_color='#bb6c3c',annotation_text='Cohort removal')
                st.plotly_chart(fig,width='stretch')

if tabs[8].open:
    with tabs[8]:
        st.subheader('Compare protein connections across networks')
        options=[n for n in datasets if n!=network]
        if not options: st.info('At least two canonical networks are required.')
        else:
            other=setting(st.selectbox,'Compare with network B',options,key='cross_other')
            p=datasets[other]; info=p.stat()
            st.caption('A is the sidebar network. Compare cohort proteins present in both networks. Full-network degree percentile ranks normalize graph size; partner turnover uses full neighborhood sets. Capture bias and differing assay regimes limit biological interpretation.')
            result=run_panel('cross_network',{'network_B':other,'path_B':str(p),'mtime_B':info.st_mtime_ns,'size_B':info.st_size,'network_B_provenance':json.loads(p.with_suffix('.json').read_text()) if p.with_suffix('.json').exists() else {}},lambda:cross_network_analysis(graph_for(),graph_for(other),cohort))
            if result:
                summary(result)
                if len(result['table']):
                    fig=px.scatter(result['table'],x='degree_percentile_A',y='degree_percentile_B',color='partner_turnover',hover_data=['protein','degree_A','degree_B'],title=f'{network} versus {other}: cohort degree ranks',labels={'degree_percentile_A':network+' degree percentile','degree_percentile_B':other+' degree percentile','partner_turnover':'Partner turnover'})
                    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
                    st.plotly_chart(fig,width='stretch'); table(result['table'],'cross_table')
                else: st.info('No selected cohort proteins are present in both networks.')


if tabs[9].open:
    with tabs[9]:
        st.subheader('Compare moonlighting source sets together')
        st.write('Display each source separately on the same axes. Source sets can overlap: a protein shared by two sources appears in both distributions. The shared background excludes the union of all selected sets.')
        compare_sources=st.multiselect('Source sets to compare',SOURCES,default=['moonprot','multitask','moonlitdb'],format_func=lambda source:GROUP_LABELS.get(source,source),key='compare_sources')
        include_union=setting(st.checkbox,'Show union of selected sets',value=True,key='compare_include_union')
        compare_extra=setting(st.checkbox,'Include advanced centrality metrics',key='compare_extra')
        st.caption('Union is created automatically from the selected sets, counting each canonical protein once. Union tests stay in the same BH correction even when its plots are hidden, so the stars remain consistent. This view uses its own source selection and the sidebar reviewed-entry and publication filters. Sidebar union/intersection settings apply to the other tabs. All selected source-versus-background tests and metrics share one Benjamini–Hochberg correction. Overlapping sets are not independent; stars do not test differences between sources.')
        if compare_sources:
            parameters={'analysis_revision':'centrality_profiles_hubs_v1','sources':compare_sources,'include_union':True,'advanced':compare_extra,'landmarks':32,
                        'correction_scope':'Benjamini–Hochberg across all displayed set-versus-background tests and metrics, including the automatically computed Union'}
            result=run_panel('source_comparison',parameters,lambda:compare_source_sets(graph_for(),moon,compare_sources,reviewed_only=reviewed_only,min_publications=min_pubs,extra=compare_extra,seed=int(seed),landmarks=32,include_union=True))
            if result:
                summary(result)
                display_result=dict(result)
                if not include_union:
                    for table_name,column in [('metrics','group'),('tests','group'),('profiles','group'),('hub_enrichment','group'),('coverage','source')]:
                        display_result[table_name]=result[table_name].filter(pl.col(column)!='Union')
                st.write('Source coverage')
                table(display_result['coverage'],'comparison_coverage')
                if len(display_result['metrics']):
                    centrality_overview(display_result,'source_comparison')
                    st.subheader('Inspect individual centrality measures')
                    metric_plot(display_result,'source_comparison')
                else: st.info('None of the selected source sets has mapped proteins in this network.')
                with st.expander('Shared proteins between source sets'):
                    table(result['overlap'],'comparison_overlap')
        else:
            st.info('Choose at least one source set to compare.')

with st.expander('Analysis scope and reproducibility'):
    st.write('Implemented analyses follow the classical topology, wiring, cohort-null-model, community, diffusion, and integrative GO sections of MOONLIGHTING_TOPOLOGICAL_ANALYSES.md. The catalogue also lists research ideas that are not implemented here, including quantum walks, Hodge decompositions, PU learning, and full graphlet spectra.')
    st.write('Every successful run saves CSV tables, parameters, network fingerprints, cohort accessions, random seed and method diagnostics in data/analysis_runs. BH corrections apply within each requested analysis and network; repeated exploratory runs are not jointly corrected. Degree bins are a practical adjustment rather than exact individual-degree matching.')
    spec=ROOT/'MOONLIGHTING_TOPOLOGICAL_ANALYSES.md'
    if spec.exists(): st.download_button('Download analysis catalogue',spec.read_text(),'MOONLIGHTING_TOPOLOGICAL_ANALYSES.md','text/markdown')
