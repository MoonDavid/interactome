"""Compute, reuse, analyze and annotate community partitions in one workspace."""
import json
import subprocess
from pathlib import Path
from runpy import run_path

import polars as pl
import streamlit as st

from clustering_explorer import ROOT, saved_runs, clustering_options, compute_clustering, annotation_target, annotate_run, saved_cluster_enrichment
from cluster_annotation_ui import bundle_zip, hierarchy_panel, saved_cluster_annotation_panel, all_runs_enrichment_panel, overlapping_membership_panel
from networks import discover_datasets
from moonlighting import significance_label, _signature

st.title('Clustering and communities')
st.caption('Choose an algorithm, reuse a saved version or compute one, then analyze and annotate its clusters.')
datasets = discover_datasets(ROOT / 'data')
runs = saved_runs()
networks = sorted(set(datasets) | (set(runs['network']) if not runs.is_empty() else set()))
if not networks:
    st.info('Prepare a canonical network on the overview page first.')
    st.stop()
network = st.selectbox('Network', networks, key='community_network')
scope = st.segmented_control('Analysis scope', ['One clustering version', 'All saved clustering runs', 'Overlapping memberships'],
    default='One clustering version', required=True, key='community_scope')
if scope == 'Overlapping memberships':
    overlapping_membership_panel(network, runs, datasets)
    st.stop()
if scope == 'All saved clustering runs':
    all_runs_enrichment_panel(network, runs, datasets)
    st.stop()
options = clustering_options(runs, network)
labels = dict(options.select('method', 'label').iter_rows())
methods = options['method'].to_list()
method = st.selectbox('Clustering algorithm', methods,
    index=methods.index('c.ocg') if 'c.ocg' in methods else 0,
    format_func=labels.__getitem__, key=f'community_algorithm_{network}')
available = runs.filter((pl.col('network') == network) & (pl.col('method') == method)) if not runs.is_empty() else runs
saved_count = options.filter(pl.col('method') == method)['saved_versions'][0]
st.caption(f'{saved_count} saved version(s) available. Selecting a version does not rerun clustering.'
    if saved_count else 'No saved version available. Compute this algorithm below to create one.')
version_key = f'community_version_{network}_{method}'
with st.expander('Compute a clustering version', expanded=not saved_count):
    c1, c2, c3 = st.columns(3)
    seed = c1.number_input('Random seed', 1, 2**32-1, 42, key='community_compute_seed')
    resolution = c2.number_input('Resolution', min_value=0.000001, value=1.0, step=0.1, key='community_compute_resolution')
    clusters = c3.number_input('Target clusters (where applicable)', min_value=2, value=16, key='community_compute_clusters')
    timeout = st.number_input('Worker timeout (seconds)', min_value=1, value=600, step=60, key='community_compute_timeout')
    st.caption('Optional backends must be installed. The existing 500,000-edge NetworkX and costly/dense '
        'method guards apply. These parameters apply to a new computation; saved versions keep their recorded parameters.')
    if st.button('Compute and save clustering', type='primary', disabled=network not in datasets):
        try:
            with st.spinner(f'Running {method}…'):
                log = compute_clustering(network, method, seed=int(seed), resolution=float(resolution), clusters=int(clusters), timeout=int(timeout))
            runs = saved_runs()
            available = runs.filter((pl.col('network') == network) & (pl.col('method') == method))
            successful = available.filter(pl.col('status') == 'success')
            if not successful.is_empty():
                newest = max(successful.iter_rows(named=True), key=lambda row: (Path(row['directory']) / 'result.json').stat().st_mtime_ns)
                st.session_state[version_key] = newest['run']
            st.code(log)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            st.error(f'Clustering could not complete: {exc}')
if available.is_empty():
    st.info('No saved result for this algorithm yet.')
    st.stop()
ordered = available.sort('run').sort(pl.col('status') != 'success', maintain_order=True)
status = {row['run']: row['status'] for row in ordered.iter_rows(named=True)}
version = st.selectbox('Clustering version', ordered['run'].to_list(), key=version_key,
    format_func=lambda run: f'{run} · saved' if status[run] == 'success' else f'{run} · {status[run]}')
row = available.filter(pl.col('run') == version).row(0, named=True)
directory = Path(row['directory'])
st.info('Communities use the recorded network version and undirected binary topology, independently '
    'of the sidebar isoform setting. Cluster IDs are local to a version. Topology alone does not '
    'validate moonlighting or protein complexes.')
view = st.segmented_control('Analysis', ['Clusters', 'Cluster enrichment', 'Protein projection', 'Functional annotation', 'Run audit'],
    default='Clusters', required=True, key='community_view')
if view == 'Run audit':
    st.dataframe(available, hide_index=True)
    st.json(json.loads((directory / 'result.json').read_text(encoding='utf-8')))
    if (directory / 'worker.log').is_file():
        st.code((directory / 'worker.log').read_text(encoding='utf-8', errors='replace')[-20000:])
elif row['status'] != 'success':
    st.warning(f"Run status: {row['status']}. {row['reason'] or ''}")
elif view == 'Protein projection':
    from cluster_projection import projection_panel
    projection_panel(directory)
elif view == 'Cluster enrichment':
    from clusterannotation import file_input
    from moonlighting import load_moon_data, cohort_ids, GROUP_LABELS, SOURCES
    protein_set = st.selectbox('Protein set source', SOURCES, index=SOURCES.index('moonlitdb'),
        format_func=lambda source: GROUP_LABELS.get(source, source), key='community_protein_set')
    st.caption('One-sided hypergeometric tests compare the named protein set with the original full network; '
        'BH includes every cluster in this version. Fold >1 means overrepresentation. Overlapping clusters '
        'share proteins. * q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable.')
    try:
        path = datasets[network]
        inputs = [ROOT / 'data/allmoon_with_moonlitdb_human.csv', ROOT / 'data/mappings/moonlighting_uniprot.json']
        identity = (str(directory), protein_set, _signature(path), _signature(directory / 'memberships.parquet'), tuple(_signature(p) for p in inputs))
        state_key = 'community_enrichment_result'
        if st.button('Analyze cluster enrichment', type='primary'):
            st.session_state.pop(state_key, None)
            frame, population, source = file_input(directory / 'memberships.parquet', str(path))
            cohort, _ = cohort_ids(load_moon_data(ROOT / 'data'), [protein_set])
            results = saved_cluster_enrichment(frame, population, cohort)
            st.session_state[state_key] = {'identity': identity, 'results': results}
        state = st.session_state.get(state_key)
        if state is not None and state['identity'] == identity:
            results = state['results']
            st.caption('Select one or more rows, then autoannotate the selected clusters below.')
            event = st.dataframe(results, hide_index=True, on_select='rerun', selection_mode='multi-row',
                key=f'enrichment_rows_{network}_{method}_{version}_{protein_set}', alt='Select clusters from enrichment results')
            selected = [results['community'][i] for i in event.selection.rows]
            annotation_scope = st.selectbox('Clusters to autoannotate',
                ['All significant clusters', 'Selected rows', 'All tested clusters'],
                key=f'enrichment_annotation_scope_{network}_{method}_{version}_{protein_set}')
            if annotation_scope == 'All significant clusters':
                selected = results.filter((pl.col('qvalue') < .05) &
                    (pl.col('fold_enrichment') > 1))['community'].to_list()
            elif annotation_scope == 'All tested clusters':
                selected = results['community'].to_list()
            st.caption('Significant clusters have original full-version BH q < 0.05 and fold enrichment > 1. '
                'Functional annotation applies its own correction across all eligible terms, libraries and '
                'subsets of the clusters chosen here; it does not reuse the protein-set enrichment q-values.')
            if selected:
                saved_cluster_annotation_panel(directory, selected, f'enrichment_annotation_{network}_{method}_{version}_{protein_set}')
            else:
                st.info('No clusters match this annotation choice.')
            st.download_button('Download community protein-set enrichment', results.write_csv(), 'community_enrichment.csv', 'text/csv', on_click='ignore')
        elif state is not None:
            st.info('The clustering version, protein set or input files changed. Analyze cluster enrichment again.')
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Cluster enrichment unavailable: {exc}')
elif view == 'Clusters':
    st.dataframe(pl.DataFrame([{k: row[k] for k in ('nodes', 'edges', 'communities', 'covered_nodes', 'unassigned_nodes', 'overlapping_nodes')}]))
    st.caption('Coverage counts proteins assigned at least once; overlap counts proteins in several clusters. Sizes are descriptive; no significance test applies.')
    sizes = pl.read_parquet(directory / 'community_sizes.parquet').sort('len', descending=True)
    protein = st.text_input('Find clusters containing a UniProt accession').strip().upper()
    members = pl.scan_parquet(directory / 'memberships.parquet')
    if protein:
        choices = members.filter(pl.col('protein') == protein).select('community').drop_nulls().unique().collect()
        sizes = sizes.join(choices, on='community', how='semi')
    st.caption('Select one or more cluster rows to inspect proteins and run autoannotation.')
    event = st.dataframe(sizes, hide_index=True, on_select='rerun', selection_mode='multi-row',
        key=f'cluster_rows_{network}_{method}_{version}_{protein}', alt='Select communities')
    selected = [sizes['community'][i] for i in event.selection.rows]
    if selected:
        proteins = members.filter(pl.col('community').is_in(selected)).collect()
        st.dataframe(proteins, hide_index=True)
        st.download_button('Download selected cluster proteins', proteins.write_csv(), 'cluster_members.csv', 'text/csv', on_click='ignore')
    saved_cluster_annotation_panel(directory, selected, f'cluster_annotation_{network}_{method}_{version}')
else:
    st.caption('Functional labels use one-sided hypergeometric tests against the full original network. '
        'One BH family includes all eligible terms, clusters, nonempty full/core/interface subsets and '
        'GO BP/MF/CC and Reactome libraries. * q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; '
        'ns otherwise; n/a unavailable. Automatic labels require biological review.')
    try:
        target, _, _, _ = annotation_target(directory)
        if not (target / 'manifest.json').is_file():
            st.info('No current functional annotation bundle for this version.')
            if st.button('Calculate and save functional annotation', type='primary'):
                with st.spinner('Annotating clusters…'):
                    annotate_run(directory)
                st.rerun()
        else:
            labels = pl.read_csv(target / 'cluster_labels.csv', schema_overrides={'community': pl.String})
            event = st.dataframe(labels, hide_index=True, on_select='rerun', selection_mode='single-row',
                key=f'annotation_rows_{target.name}', alt='Select an annotated cluster')
            if event.selection.rows:
                cluster = labels['community'][event.selection.rows[0]]
                hierarchy_panel(target, cluster, f'all_{target.name}_{cluster}')
                evidence = pl.scan_parquet(target / 'enrichment.parquet').filter(pl.col('community') == cluster).sort('qvalue').collect()
                evidence = evidence.with_columns(pl.Series('significance', [significance_label(q) for q in evidence['qvalue']]))
                st.dataframe(evidence.head(200), hide_index=True)
            else:
                st.info('Select a label row to inspect its hierarchy and evidence.')
            st.download_button('Download full annotation bundle', lambda: bundle_zip(target), 'cluster_annotation.zip', 'application/zip', on_click='ignore')
            with st.expander('Annotation provenance and coverage'):
                st.json(json.loads((target / 'manifest.json').read_text(encoding='utf-8')))
    except (OSError, ValueError) as exc:
        st.warning(f'Annotation unavailable: {exc}')

with st.expander('Additional community analyses', expanded=False, key='community_additional', on_change='rerun') as extra:
    if extra.open:
        analysis = st.segmented_control('Additional analysis', ['Module roles', 'Algorithm comparison', 'Cytoscape (clusterMaker2)'],
            default='Module roles', required=True, key='community_additional_analysis')
        run_path(str(Path(__file__).with_name('2_Topological_analysis.py')),
            init_globals={'CLUSTERING_WORKSPACE': True, 'COMMUNITY_ANALYSIS_MODE': analysis, 'COMMUNITY_DEFAULT_NETWORK': network})
