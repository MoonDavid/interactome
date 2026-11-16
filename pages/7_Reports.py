"""Read saved scientific reports without rerunning their analyses."""
import json
from pathlib import Path

import polars as pl
import streamlit as st
from cluster_reports import theme_index, protein_metadata, cluster_proteins

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(max_entries=4)
def load_evidence(path, modified_ns):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@st.cache_data(max_entries=2)
def load_protein_metadata(root, fingerprints, schema_version):
    # Streamlit does not hash the implementation of imported helper functions.
    # Include both the schema and helper fingerprint in the cache arguments.
    return protein_metadata(root)


st.title('Cluster reports')
st.caption('Saved results; opening a report does not run clustering or enrichment.')
protein_set = st.segmented_control('Protein set', ['MoonLitDB', 'MoonProt'],
                                   default='MoonLitDB', required=True, key='report_protein_set')
STUDY = ROOT / f'data/analysis_runs/{protein_set.lower()}_small_community_study'
evidence_path = STUDY / 'reviewed_evidence.json'
if not evidence_path.is_file():
    st.info('No saved community study is available yet.')
    st.stop()

st.header(f'{protein_set}-enriched communities in BioPlex and TAGC')
try:
    evidence = load_evidence(str(evidence_path), evidence_path.stat().st_mtime_ns)
    manifest = json.loads((STUDY / 'study_manifest.json').read_text(encoding='utf-8'))
except (OSError, ValueError) as exc:
    st.error(f'The saved report could not be read: {exc}')
    st.stop()

st.caption(f"Generated: {manifest['generated_at']} · Protein set: {manifest['protein_set']}")
st.info(manifest['selection'] + '. This saved report uses its original network backgrounds; '
        'the sidebar isoform setting does not recalculate it.')
summary = pl.from_dicts(evidence['summary'])
st.markdown('Across the saved compatible runs, **' +
            str(summary['selected'].sum()) + '** enriched cluster results were selected. '
            'Overlapping clusters and repeated memberships are not independent discoveries.')
st.dataframe(summary.select('network', 'compatible_runs', 'tested', 'significant_small',
                            'selected', 'unique_memberships', 'background',
                            pl.col(protein_set.lower()).alias(f'{protein_set} in network')),
             hide_index=True)
network = st.segmented_control('Interactome', summary['network'].to_list(),
                               default=summary['network'][0], required=True, key='report_network')
clusters = [row for row in evidence['clusters'] if row['network'] == network]
if not clusters:
    st.info('No clusters meet the saved selection criteria in this network.')
    st.stop()
population = next(row for row in evidence['summary'] if row['network'] == network)
st.markdown(f"**{network}** contributes **{population['selected']} selected cluster results**, "
            f"representing **{population['unique_memberships']} distinct protein memberships**. "
            f"The top-ranked cluster has **{population['top_hits']}/{population['top_size']} "
            f"{protein_set} proteins** ({population['top_fold']:.2f}× enrichment), with leading "
            f"functional label **{population['top_label']}**.")
with st.expander('Functional themes across distinct protein memberships'):
    themes = pl.from_dicts([row for row in evidence['themes'] if row['network'] == network])
    st.dataframe(themes.select('term', 'clusters'), hide_index=True)
    st.caption('Descriptive leading GO biological-process label counts after exact membership '
               'deduplication. No significance test applies to these frequencies.')
columns = ['rank', 'method', 'community', 'proteins', 'protein_set_hits',
           'fold_enrichment', 'qvalue', 'significance', 'label']
terms = theme_index(clusters)
themes = terms.select('library', 'term_id', 'term_name').unique().sort('library', 'term_name')
theme_labels = {f"{row['library']} · {row['term_name']} [{row['term_id']}]":
                (row['library'], row['term_id']) for row in themes.iter_rows(named=True)}
theme = st.selectbox('Functional theme', ['All themes', *theme_labels], key=f'report_theme_{protein_set}_{network}')
st.caption('Themes match any significant saved full-cluster GO or Reactome term (annotation q < 0.05), '
           'including terms beyond the leading label. Filtering preserves the original q-values and ranking.')
if theme != 'All themes':
    library, term_id = theme_labels[theme]
    ranks = set(terms.filter((pl.col('library') == library) & (pl.col('term_id') == term_id))['rank'])
    clusters = [row for row in clusters if row['rank'] in ranks]
st.caption(f'{len(clusters)} matching cluster results in {network}.')
table = pl.from_dicts([{key: row[key] for key in columns} for row in clusters])
st.subheader('Selected clusters')
st.dataframe(table, hide_index=True)
st.download_button('Download selected clusters', table.write_csv(),
                   file_name=f'{protein_set}_{network}_enriched_clusters.csv', mime='text/csv')
by_rank = {row['rank']: row for row in clusters}
selection_key = f'report_cluster_{protein_set}_{network}'
if st.session_state.get(selection_key) not in by_rank:
    st.session_state[selection_key] = clusters[0]['rank']
rank = st.selectbox('Inspect cluster', list(by_rank), key=selection_key,
                    format_func=lambda rank: f"{rank} · {by_rank[rank]['method']} · "
                    f"{by_rank[rank]['community']} · {by_rank[rank]['proteins']} proteins")
detail = by_rank[rank]
st.markdown(f"**{detail['protein_set_hits']}/{detail['proteins']} {protein_set} proteins** · "
            f"{detail['fold_enrichment']:.2f}× enrichment · "
            f"global q = {detail['qvalue']:.3g} {detail['significance']}")
for key, label in [('GO_BP_enrichment', 'GO biological process'),
                   ('GO_MF_enrichment', 'GO molecular function'),
                   ('GO_CC_enrichment', 'GO cellular component'),
                   ('Reactome_enrichment', 'Reactome')]:
    st.markdown(f"**{label}:** {detail.get(key) or 'No significant label'}")
st.markdown('**Swiss-Prot FUNCTION keywords:** ' +
            (detail.get('UniProt_FUNCTION_tfidf') or 'Unavailable'))
st.caption('FUNCTION keywords are descriptive; no significance test applies.')
flow = st.expander('GO flow diagrams for this cluster',
                   key=f'report_go_flow_{protein_set}_{network}_{rank}', on_change='rerun')
if flow.open:
    with flow:
        annotation_directory = detail.get('annotation_directory')
        if not annotation_directory or not Path(annotation_directory).is_dir():
            st.info('The saved annotation bundle for this cluster is unavailable.')
        else:
            from cluster_annotation_ui import hierarchy_panel
            try:
                hierarchy_panel(Path(annotation_directory), str(detail['community']),
                                f'report_go_{protein_set}_{network}_{rank}')
            except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
                st.warning(f'The saved GO flow diagram could not be loaded: {exc}')
st.subheader('Proteins in this cluster')
metadata_inputs = [ROOT / 'data' / relative for relative in
                   ('allmoon_with_moonlitdb_human.csv', 'mappings/moonlighting_uniprot.json',
                    'annotations/uniprot_functions.tsv')]
metadata_inputs.append(ROOT / 'cluster_reports.py')
try:
    metadata = load_protein_metadata(str(ROOT / 'data'),
        tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
              for path in metadata_inputs if path.is_file()), schema_version=3)
    proteins = cluster_proteins(detail['members'], metadata, protein_set)
    moonlitdb_count = proteins.filter(pl.col(protein_set)).height
    if moonlitdb_count != detail['protein_set_hits']:
        st.warning(f'Current {protein_set} mapping differs from the saved study hit count. '
                   'Protein membership labels below use the current local mapping; saved tests are unchanged.')
    show_all = st.toggle('Show all cluster proteins', value=False, key='report_all_proteins')
    displayed = proteins if show_all else proteins.filter(pl.col(protein_set))
    st.caption(f'{displayed.height} proteins shown · {moonlitdb_count} {protein_set} members in the cluster. '
               'Names use the local protein-set metadata; functions use the saved Swiss-Prot FUNCTION snapshot. '
               'MoonLitDB canonical and moonlighting functions are the source-provided GO-slim annotations. '
               'Unavailable fields are shown explicitly. Descriptions do not establish a cluster-specific function.')
    st.dataframe(displayed, hide_index=True)
    st.download_button('Download displayed proteins', displayed.write_csv(),
                       file_name=f'{protein_set}_{network}_cluster_{rank}_proteins.csv', mime='text/csv')
    if not displayed.is_empty():
        protein_key = 'report_protein'
        choices = displayed['UniProt'].to_list()
        if st.session_state.get(protein_key) not in choices:
            st.session_state[protein_key] = choices[0]
        labels = {row['UniProt']: f"{row['Gene names']} · {row['Protein name']} · {row['UniProt']}"
                  for row in displayed.iter_rows(named=True)}
        accession = st.selectbox('Read complete protein function', choices,
                                  format_func=labels.__getitem__, key=protein_key)
        protein = displayed.filter(pl.col('UniProt') == accession).row(0, named=True)
        st.markdown(f"**{protein['Gene names']} — {protein['Protein name']}**")
        st.caption(f"{accession} · {protein_set} member: {'yes' if protein[protein_set] else 'no'}")
        if protein_set == 'MoonProt':
            st.markdown('**MoonProt primary function:**')
            st.markdown(protein.get('MoonProt primary function') or 'Unavailable')
            st.markdown('**MoonProt secondary function:**')
            st.markdown(protein.get('MoonProt secondary function') or 'Unavailable')
        st.markdown('**MoonLitDB canonical functions (GO-slim):**')
        st.markdown(protein.get('MoonLitDB canonical functions') or 'Unavailable')
        st.markdown('**MoonLitDB moonlighting functions (GO-slim):**')
        st.markdown(protein.get('MoonLitDB moonlighting functions') or 'Unavailable')
        st.markdown('**Swiss-Prot function description:**')
        st.markdown(protein['Function description'])
except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
    st.warning(f'Protein metadata could not be loaded: {exc}')
    st.text(' · '.join(detail['members']))
if detail['terms']:
    st.dataframe(pl.from_dicts(detail['terms']), hide_index=True)
else:
    st.info('No significant full-cluster functional terms were saved.')
with st.expander('Methods and interpretation'):
    st.markdown(f'One-sided hypergeometric tests compare {protein_set} membership with the full '
                'corresponding interactome. BH correction includes every tested community across '
                'all compatible saved runs within each network, before size or significance filtering. '
                'Higher fold enrichment means greater MoonLitDB representation relative to that background.\n\n'
                'GO and Reactome enrichment use the full network background. Annotation BH correction '
                'spans all eligible terms, libraries and nonempty full/core/interface subsets of selected '
                'clusters within each saved run. The table shows significant full-cluster terms. '
                'Annotation q-values and community q-values have different correction families.\n\n'
                'Significance: * q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001. '
                'ns denotes a valid nonsignificant test; n/a denotes an unavailable test.\n\n'
                'Leading labels summarize functional enrichment; they do not establish distinct '
                'functions or validate moonlighting. Network coverage and overlapping partitions '
                'affect interpretation. These functional annotations are exploratory.')
