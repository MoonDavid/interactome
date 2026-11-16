"""Shared Streamlit controls for functional cluster annotation."""
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import streamlit as st

from clusterannotation import (
    DATA as DEFAULT_DATA, DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME,
    DEFAULT_FUNCTIONS,
    annotate_memberships, write_annotation_bundle,
)
from moonlighting import _signature


def uniprot_function_report(paths, community=None):
    """Render a compact descriptive FUNCTION summary and return its Markdown."""
    frames = []
    for path in paths:
        if Path(path).is_file():
            frame = pl.read_parquet(path)
            if community is not None:
                frame = frame.filter(pl.col('community') == str(community))
            frames.append(frame)
    lines = ['**UniProt / GeneRIF keywords and phrases**', '']
    text = pl.concat(frames, how='diagonal_relaxed') if frames else pl.DataFrame()
    if text.is_empty():
        lines.append('No saved UniProt FUNCTION labels available; rerun annotation with the FUNCTION source to generate them.')
    else:
        names = {'tfidf': 'TF-IDF', 'document_lift': 'Document lift (log₂)'}
        for (subset, method), group in text.group_by('subset', 'method', maintain_order=True):
            top = group.sort(['score', 'label'], descending=[True, False]).unique('label', maintain_order=True).head(5)
            lines += [f'**{subset.capitalize()} · {names.get(method, method)}**', '']
            lines += [f"- **{row['label']}** · score={row['score']:.2g} · "
                f"{row['protein_hits']}/{row['query_size']} proteins "
                f"({row['protein_hits'] / row['query_size']:.0%}) · cluster {row['community']}"
                for row in top.iter_rows(named=True)]
            lines.append('')
    st.markdown('\n'.join(lines))
    note = ('Descriptive phrases from Swiss-Prot FUNCTION comments and available human GeneRIF text; up to five per subset and scoring method. '
        'TF-IDF highlights distinctive text; document lift compares phrase frequency in annotated subset '
        'proteins with the annotated network complement. c-TF-IDF ranks cluster pseudo-documents against network term frequencies. Log-odds uses a human UniProt prior (strength 1,000) and the annotated network complement; higher positive values indicate distinctive terms. YAKE uses lexical statistics, not cosine similarity; full abstracts require a local PMID/abstract TSV. Scores are not significance tests; no stars apply. '
        'Support uses all subset proteins as denominator. Missing text and redundant phrases limit interpretation.')
    st.caption(note)
    from importlib.util import find_spec
    if find_spec('yake') is None:
        st.caption('YAKE keyphrases unavailable: install the optional text-annotation extra to enable them.')
    return '\n'.join([*lines, note])


def reactome_majority(evidence):
    """Rank available pathways supported by strictly more than half of a subset."""
    return (evidence.filter((pl.col('library') == 'Reactome') & (pl.col('query_size') > 0))
        .with_columns((pl.col('query_hits') / pl.col('query_size')).alias('support_fraction'))
        .filter(pl.col('support_fraction') > .5)
        .sort(['support_fraction', 'query_hits', 'term_id'], descending=[True, True, False])
        .unique('term_id', maintain_order=True).head(5))


def readable_annotation_sections(evidence):
    """One entry per term, with support and enrichment tags on the same evidence row."""
    from moonlighting import significance_label
    report = []
    for library in evidence['library'].unique(maintain_order=True):
        heading = f'**{library.replace("GO_", "GO ")}**'
        st.markdown(heading)
        rows = (evidence.filter(pl.col('library') == library)
            .with_columns((pl.col('query_hits') / pl.col('query_size')).alias('support_fraction'))
            .sort('qvalue', 'pvalue', 'term_id'))
        enriched = rows.filter((pl.col('qvalue') < .05) & (pl.col('fold_enrichment') > 1))
        enriched = enriched.unique('term_id', maintain_order=True).head(5 if library == 'Reactome' else 1)
        majority = (rows.filter(pl.col('support_fraction') > .5)
            .sort(['support_fraction', 'query_hits', 'term_id'], descending=[True, True, False])
            .unique('term_id', maintain_order=True).head(5))
        selected = pl.concat([enriched, majority]).unique(maintain_order=True)
        lines = []
        for (term,), group in selected.group_by('term_id', maintain_order=True):
            contexts = []
            for row in group.iter_rows(named=True):
                tags = []
                if row['support_fraction'] > .5:
                    tags.append('Majority')
                if row['qvalue'] < .05 and row['fold_enrichment'] > 1:
                    tags.append('Enriched')
                provenance = (f" · run {row['run']} / bundle {row['bundle']}" if 'bundle' in row else '')
                contexts.append(f"{' + '.join(tags)} · {row['query_hits']}/{row['query_size']} proteins "
                    f"({row['support_fraction']:.0%}) · q={row['qvalue']:.2g} {significance_label(row['qvalue'])} · "
                    f"{row['fold_enrichment']:.2g}× · cluster {row['community']} / {row['subset']}{provenance}")
            lines.append(f"- **{group['term_name'][0]}** ({term}): " + '; '.join(contexts))
        if not lines:
            lines = ['No available terms meet majority or significant-enrichment criteria.']
        st.markdown('\n'.join(lines))
        report += [heading, '', *lines, '']
    note = ('Majority: >50% of all proteins in the indicated subset, including unannotated proteins; '
        'descriptive, with no significance test. Enriched: fold >1 and original BH q <0.05. '
        'One-sided hypergeometric tests use the full network; BH covers all eligible terms, libraries, '
        'clusters and subsets in each bundle, including hidden results. Stars refer only to enrichment: '
        '* q <0.05; ** q <0.01; *** q <0.001; **** q <0.0001; ns otherwise; n/a unavailable. '
        'Lists combine up to five majority terms with the strongest GO term or up to five enriched Reactome '
        'terms. Support is limited to terms retained in saved evidence; correction families can differ between bundles.')
    st.caption(note)
    return '\n'.join([*report, note])

def saved_annotation_overview(network, key):
    """Read existing annotation bundles only; preserve each original test family."""
    from moonlighting import significance_label
    root = DEFAULT_DATA / 'analysis_runs' / 'clustering_annotations' / network
    bundles = sorted(root.glob('*/*/*/enrichment.parquet'))
    if not bundles:
        st.info('No saved functional annotations for this network yet. Autoannotate clusters below first.')
        return
    frames, skipped = [], []
    for path in bundles:
        try:
            frame = pl.read_parquet(path).with_columns(
                pl.lit(path.parents[2].name).alias('method'),
                pl.lit(path.parents[1].name).alias('run'),
                pl.lit(path.parent.name).alias('bundle'))
            frames.append(frame)
        except (OSError, pl.exceptions.PolarsError) as exc:
            skipped.append(f'{path.parent}: {exc}')
    if skipped:
        st.warning(f'{len(skipped)} unreadable annotation bundles were excluded.')
        with st.expander('Excluded annotation files'):
            st.text('\n'.join(skipped))
    if not frames:
        return
    evidence = pl.concat(frames, how='diagonal_relaxed').sort(
        ['qvalue', 'pvalue', 'method', 'term_id'])
    evidence = evidence.with_columns(pl.Series('significance',
        [significance_label(q) for q in evidence['qvalue']]))
    st.info('Lower adjusted q-values indicate stronger enrichment evidence; fold enrichment above 1 '
        'indicates overrepresentation. Each saved bundle uses one-sided hypergeometric tests against '
        'its full network background and BH correction across all eligible terms, libraries, clusters '
        'and subsets in that bundle. Families and selected clusters can differ between bundles; '
        'this overview does not establish which clustering method is best. Repeated annotations '
        'are not independent replications. Saved bundles may describe earlier network snapshots. '
        'GO support and Swiss-Prot FUNCTION labels are descriptive and have no significance test.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Clustering methods', evidence['method'].n_unique())
    c2.metric('Saved annotation bundles', len(frames))
    c3.metric('Significant enrichment rows', evidence.filter(pl.col('qvalue') < .05).height)
    best = pl.concat([
        evidence.filter(pl.col('library') != 'Reactome').group_by('method', 'library', maintain_order=True).first(),
        evidence.filter((pl.col('library') == 'Reactome') & (pl.col('qvalue') < .05) &
                        (pl.col('fold_enrichment') > 1))
            .unique(['method', 'library', 'term_id'], maintain_order=True)
            .group_by('method', 'library', maintain_order=True).head(5),
    ], how='diagonal_relaxed').select(
        'method', 'library', 'term_name', 'term_id', 'community', 'subset', 'query_hits',
        'fold_enrichment', 'qvalue', 'significance', 'run', 'bundle')
    report = [f'# Saved annotation overview · {network}', '',
        'Strongest GO annotation and up to five distinct significant enriched Reactome pathways per clustering method.', '',
        'Tests: one-sided hypergeometric against each original full network; BH correction '
        'across all eligible terms, libraries, clusters and subsets within each saved bundle. '
        'Different correction families and network snapshots limit comparisons between methods. '
        'This ranking does not establish biological superiority.', '',
        '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable.', '']
    st.markdown('**Annotation report by clustering method**')
    for method in evidence['method'].unique(maintain_order=True):
        with st.container(border=True):
            st.markdown(f'### {method}')
            report += ['', f'## {method}', '', readable_annotation_sections(evidence.filter(pl.col('method') == method))]
            function_paths = [path.parent / 'uniprot_function_labels.parquet'
                              for path in bundles if path.parents[2].name == method]
            report += ['', f'## {method} · UniProt FUNCTION', '',
                       uniprot_function_report(function_paths)]
    st.caption('* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns: valid '
        'non-significant test; n/a: unavailable. Ranking retains original full-family q-values.')
    with st.expander('All saved annotation evidence'):
        st.dataframe(evidence, hide_index=True, alt='All saved annotation evidence ranked by adjusted q-value')
    st.download_button('Download annotation overview (CSV)', best.write_csv(),
        f'{network}_annotation_overview.csv', 'text/csv', key=f'{key}_download', on_click='ignore')
    st.download_button('Download readable annotation report', '\n'.join(report),
        f'{network}_annotation_report.md', 'text/markdown', key=f'{key}_report', on_click='ignore')


def overlapping_membership_panel(network, runs, datasets):
    """Inspect saved overlap and test protein-set membership multiplicity on demand."""
    from clustering_explorer import ROOT, all_saved_membership_burden
    from moonlighting import SOURCES, GROUP_LABELS, cohort_ids, load_moon_data
    st.subheader('Do protein-set proteins belong to more clusters?')
    if runs.is_empty() or network not in datasets:
        st.info('A local network and saved clustering runs are required.')
        return
    eligible = runs.filter((pl.col('network') == network) & (pl.col('status') == 'success'))
    if eligible.is_empty():
        st.info('No successful saved runs for this network.')
        return
    st.markdown('**Saved runs with observed overlapping membership**')
    st.dataframe(eligible.filter(pl.col('overlapping_nodes') > 0).select(
        'method', 'run', 'overlapping_nodes', 'covered_nodes'), hide_index=True,
        alt='Saved clustering runs with proteins assigned to multiple clusters')
    st.info('Count distinct cluster IDs per protein within each saved version. Zero means unassigned; '
        'two or more means overlapping membership. Higher counts suggest participation in more detected '
        'communities. Compare each named protein set with proteins outside the union of all source '
        'protein sets in the same network. Missing annotations or memberships can affect interpretation. '
        'This comparison is unadjusted for degree: hubs often have more memberships. Overlap does not '
        'validate moonlighting, and cluster counts are not directly comparable between algorithms. '
        'Ego-network overlap can be largely built into the method.')
    custom_fuzzy = st.checkbox('Choose a fuzzy membership threshold', value=True,
        key=f'membership_custom_fuzzy_{network}')
    fuzzy_threshold, fuzzy_fallback = None, True
    if custom_fuzzy:
        fuzzy_threshold = st.slider('Minimum fuzzy membership score', .01, 1.0, .2, .01,
            key=f'membership_fuzzy_threshold_{network}')
        fuzzy_fallback = st.checkbox('Keep strongest cluster when no score meets threshold', value=True,
            key=f'membership_fuzzy_fallback_{network}')
        st.caption('Applies score ≥ threshold to saved row-normalized scores from fuzzy C-means, '
            'fuzzy NMF, fuzzy RWR and fuzzy neighborhood. Other methods retain their saved assignments. '
            'This uses the exact chosen threshold, without the original algorithm’s adaptive minimum. '
            'Higher thresholds usually reduce overlap. Scores are method-specific affiliation strengths, '
            'not calibrated probabilities. The inventory above describes the original saved assignments; '
            'the results below reflect your threshold. Missing soft matrices are excluded and reported. '
            'Choose a threshold before testing; q-values below cover this threshold only, not repeated exploration.')
    inputs = [datasets[network], ROOT / 'data/allmoon_with_moonlitdb_human.csv',
              ROOT / 'data/mappings/moonlighting_uniprot.json']
    identity = (network, fuzzy_threshold, fuzzy_fallback, tuple(_signature(p) for p in inputs), tuple(
        (run['method'], run['run'], _signature(Path(run['directory']) / 'memberships.parquet'),
         _signature(Path(run['directory']) / 'result.json'),
         _signature(Path(run['directory']) / 'soft_memberships.npz')
         if (Path(run['directory']) / 'soft_memberships.npz').is_file() else None)
        for run in eligible.iter_rows(named=True)), _signature(ROOT / 'clustering_explorer.py'))
    state_key = f'membership_burden_{network}'
    if st.button('Compare cluster memberships for all protein sets', type='primary',
                 key=f'{state_key}_run'):
        st.session_state.pop(state_key, None)
        with st.spinner('Counting memberships in saved runs…'):
            data = load_moon_data(ROOT / 'data')
            cohorts = {source: set(cohort_ids(data, [source])[0]) for source in SOURCES}
            results, diagnostics = all_saved_membership_burden(eligible, network, datasets[network], cohorts,
                fuzzy_threshold=fuzzy_threshold, fuzzy_fallback=fuzzy_fallback)
        st.session_state[state_key] = dict(identity=identity, results=results, diagnostics=diagnostics)
    state = st.session_state.get(state_key)
    if state is None:
        return
    if state['identity'] != identity:
        st.info('Inputs changed. Run the membership comparison again.')
        return
    results = state['results']
    if state['diagnostics'].height:
        with st.expander('Excluded saved runs'):
            st.dataframe(state['diagnostics'], hide_index=True, alt='Incompatible saved clustering runs')
    if results.is_empty():
        st.info('No compatible saved runs could be compared.')
        return
    source = st.selectbox('Protein set', SOURCES, format_func=lambda value: GROUP_LABELS.get(value, value),
        key=f'{state_key}_source')
    display = results.filter(pl.col('source') == source)
    import plotly.graph_objects as go
    fig = go.Figure()
    labels = [f"{row['method']} · {row['version']}" for row in display.iter_rows(named=True)]
    fig.add_bar(name=GROUP_LABELS.get(source, source), x=labels,
        y=display['protein_set_multi_fraction'].to_list(),
        text=display['overlap_significance'].to_list(), textposition='outside', cliponaxis=False)
    fig.add_bar(name='Shared union-complement background', x=labels,
        y=display['background_multi_fraction'].to_list())
    fig.update_layout(barmode='group', yaxis=dict(title='Fraction in ≥2 clusters', tickformat='.0%', range=[0, 1.15]),
        margin=dict(b=200, t=50), height=650, xaxis=dict(tickangle=-55))
    st.plotly_chart(fig, width='stretch', alt='Protein-set and background fractions assigned to multiple clusters')
    st.caption('Stars: one-sided Fisher exact test for increased fraction in ≥2 clusters. '
        'Membership burden: one-sided Mann–Whitney U for larger membership counts (ties corrected). '
        'One BH family covers both tests, every source protein set and every compatible saved run '
        'in this network, including non-overlapping runs and hidden results. '
        '* q <0.05; ** q <0.01; *** q <0.001; **** q <0.0001; ns otherwise; n/a unavailable. '
        'Positive rank-biserial means larger counts in the protein set; odds ratio >1 means greater '
        'overlap odds. All network proteins, including unassigned proteins, enter the comparison.')
    st.dataframe(display, hide_index=True, alt='Cluster membership counts, effect sizes and adjusted tests')
    st.download_button('Download all membership comparisons', results.write_csv(),
        f'{network}_protein_set_membership_comparison.csv', 'text/csv', on_click='ignore', key=f'{state_key}_csv')


def all_runs_enrichment_panel(network, runs, datasets):
    """Rank communities across saved partitions before selecting annotations."""
    from clustering_explorer import ROOT, all_saved_cluster_enrichment
    from moonlighting import SOURCES, GROUP_LABELS, cohort_ids, load_moon_data
    st.subheader(f'Community enrichment across all saved runs · {network}')
    overview = st.expander('Annotation overview across clustering methods', on_change='rerun',
                           key=f'annotation_overview_{network}')
    if overview.open:
        with overview:
            saved_annotation_overview(network, f'annotation_overview_{network}')
    if runs.is_empty() or network not in datasets:
        st.info('A local interactome and successful saved clustering runs are required.')
        return
    eligible = runs.filter((pl.col('network') == network) & (pl.col('status') == 'success'))
    if eligible.is_empty():
        st.info('No successful saved clustering runs for this interactome yet.')
        return
    source = st.selectbox('Protein set source', SOURCES, index=SOURCES.index('moonlitdb'),
        format_func=lambda value: GROUP_LABELS.get(value, value), key='all_runs_protein_set')
    st.caption('One-sided hypergeometric tests use the complete selected interactome as background. '
        'One BH correction includes every community in every valid saved algorithm/version, including '
        'communities hidden by display filters. Rank uses this global q-value, then fold enrichment '
        'and protein-set hits. * q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; '
        'n/a unavailable. Repeated or overlapping partitions are dependent; enrichment does not establish '
        'which algorithm is biologically best. Saved runs from other network snapshots are excluded and reported.')
    try:
        inputs = [datasets[network], ROOT / 'data/allmoon_with_moonlitdb_human.csv',
                  ROOT / 'data/mappings/moonlighting_uniprot.json']
        inventory = []
        for run in eligible.sort('method', 'run').iter_rows(named=True):
            directory = Path(run['directory'])
            inventory.append((run['method'], run['run'], tuple(
                _signature(p) if p.is_file() else str(p)
                for p in [directory / 'result.json', directory / 'memberships.parquet'])))
        identity = (network, source, tuple(_signature(p) for p in inputs), inventory,
                    _signature(ROOT / 'clustering_explorer.py'))
        state_key = 'all_runs_enrichment_result'
        if st.button('Analyze all saved clustering runs', type='primary', key='all_runs_enrichment_run'):
            st.session_state.pop(state_key, None)
            with st.spinner('Comparing communities across saved clustering runs…'):
                cohort, _ = cohort_ids(load_moon_data(ROOT / 'data'), [source])
                results, diagnostics = all_saved_cluster_enrichment(eligible, network, datasets[network], cohort)
            st.session_state[state_key] = dict(identity=identity, results=results, diagnostics=diagnostics)
        state = st.session_state.get(state_key)
        if state is None:
            st.info(f'{eligible.height} successful saved runs available. Analyze them to rank their communities.')
            return
        if state['identity'] != identity:
            st.info('The interactome, saved runs or protein set changed. Analyze all saved runs again.')
            return
        results, diagnostics = state['results'], state['diagnostics']
        if diagnostics.height:
            with st.expander(f'Excluded saved runs ({diagnostics.height})'):
                st.dataframe(diagnostics, hide_index=True)
        if results.is_empty():
            st.warning('No compatible saved communities could be analyzed.')
            return
        summary = results.group_by('method', 'version').agg(
            pl.len().alias('communities'), (pl.col('qvalue') < .05).sum().alias('enriched_communities'),
            pl.col('qvalue').min().alias('best_global_qvalue')).sort('best_global_qvalue', 'method')
        with st.expander('Saved-run comparison summary'):
            st.dataframe(summary, hide_index=True, alt='Community enrichment summary by clustering run')
        c1, c2, c3 = st.columns(3)
        hits = c1.number_input('Minimum protein-set hits', min_value=0, value=2, key='all_runs_min_hits')
        size = c2.number_input('Minimum cluster size', min_value=1, value=2, key='all_runs_min_size')
        significant = c3.checkbox('Only globally significant communities', key='all_runs_significant')
        display = results.filter((pl.col('protein_set_hits') >= hits) & (pl.col('proteins') >= size))
        if significant:
            display = display.filter(pl.col('qvalue') < .05)
        st.caption(f'{display.height:,} displayed / {results.height:,} tested communities. '
                   'Select the most enriched rows below to inspect and autoannotate their clusters.')
        event = st.dataframe(display.drop('directory'), hide_index=True, on_select='rerun',
            selection_mode='multi-row', key=f'all_runs_rows_{network}_{source}_{hits}_{size}_{significant}',
            alt='Ranked community enrichment across all saved clustering runs')
        annotation_scope = st.selectbox('Clusters to autoannotate',
            ['All significant clusters', 'Selected rows', 'All tested clusters'],
            key=f'all_runs_annotation_scope_{network}_{source}')
        if annotation_scope == 'All significant clusters':
            selection = results.filter((pl.col('qvalue') < .05) & (pl.col('fold_enrichment') > 1))
        elif annotation_scope == 'All tested clusters':
            selection = results
        else:
            selection = display[event.selection.rows] if event.selection.rows else display.head(0)
        st.caption('Bulk annotation uses the full tested results, independently of display filters. '
            'Significant means global BH q < 0.05 and fold enrichment > 1. Functional annotation '
            'uses a separate BH family per saved version across all eligible terms, libraries and '
            'subsets of its chosen clusters; it is not corrected jointly across versions.')
        if not selection.is_empty():
            from clustering_explorer import annotate_run
            groups = list(selection.group_by('directory', maintain_order=True))
            st.caption(f'{selection.height:,} cluster(s) across {len(groups):,} saved version(s) chosen for annotation.')
            if st.button('Autoannotate all chosen clusters', type='primary',
                         key=f'all_runs_annotation_batch_{network}_{source}'):
                progress = st.progress(0.0, text='Annotating chosen clusters…')
                for index, ((directory,), group) in enumerate(groups):
                    try:
                        annotate_run(Path(directory), group['community'].to_list())
                    except (OSError, ValueError) as exc:
                        st.warning(f"Annotation failed for {group['method'][0]} / {group['version'][0]}: {exc}")
                    progress.progress((index + 1) / len(groups), text=f'Processed {index + 1}/{len(groups)} saved versions')
                progress.empty()
            for (directory,), group in selection.group_by('directory', maintain_order=True):
                label = f"{group['method'][0]} · {group['version'][0]} · {group.height} chosen cluster(s)"
                with st.expander(label, expanded=False):
                    saved_cluster_annotation_panel(Path(directory), group['community'].to_list(),
                        f"all_runs_annotation_{network}_{source}_{group['method'][0]}_{group['version'][0]}")
        else:
            st.info('No clusters match this annotation choice.')
        st.download_button('Download all community enrichment results', results.drop('directory').write_csv(),
            f'{network}_all_runs_community_enrichment.csv', 'text/csv', on_click='ignore')
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Cross-run enrichment unavailable: {exc}')


def saved_cluster_annotation_panel(directory, communities, key):
    """Annotate table-selected communities with one explicit correction family."""
    import json
    from clustering_explorer import annotation_target, annotate_run, community_selection
    from moonlighting import significance_label
    selected = community_selection(communities)
    if not selected:
        st.info('Select one or more cluster rows to run autoannotation.')
        return
    st.caption(f'{len(selected)} selected cluster(s). GO/Reactome enrichment uses the original full '
        'network background and one BH family over all eligible terms, libraries and nonempty '
        'full/core/interface subsets of these selected clusters. Core/interface membership uses '
        'the complete partition. GO support and Swiss-Prot FUNCTION scores are descriptive. '
        '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable.')
    try:
        target, _, _, _ = annotation_target(directory)
        full_run_cache = (target / 'manifest.json').is_file()
        if not full_run_cache:
            target, _, _, _ = annotation_target(directory, selected)
        if not (target / 'manifest.json').is_file() and st.button('Autoannotate selected clusters', type='primary', key=f'{key}_run'):
            with st.spinner('Annotating selected clusters…'):
                annotate_run(directory, selected)
        if not (target / 'manifest.json').is_file():
            return
        st.success('Using saved annotation cache automatically. No annotation computation needed.')
        if full_run_cache:
            st.caption('Full-run cache: the BH correction family includes all eligible terms, libraries, '
                       'clusters and nonempty full/core/interface subsets in the original run. '
                       'Selecting clusters preserves the original q-values; downloads retain the full bundle.')
        labels = pl.read_csv(target / 'label_summary.csv', schema_overrides={'community': pl.String})
        labels = labels.filter(pl.col('community').is_in(selected))
        event = st.dataframe(labels.select('community', 'proteins'), hide_index=True, on_select='rerun', selection_mode='single-row',
            key=f'{key}_{target.name}_summary', alt='Functional labels for selected clusters')
        cluster = labels['community'][event.selection.rows[0] if event.selection.rows else 0]
        evidence = pl.read_parquet(target / 'enrichment.parquet').filter(
            pl.col('community') == cluster).sort('qvalue')
        evidence = evidence.with_columns(pl.Series('significance',
            [significance_label(q) for q in evidence['qvalue']]))
        st.subheader(f'Annotation report · cluster {cluster}')
        readable_annotation_sections(evidence)
        uniprot_function_report([target / 'uniprot_function_labels.parquet'], cluster)
        hierarchy_panel(target, cluster, f'{key}_{target.name}_{cluster}_hierarchy')
        with st.expander('GO hierarchy support and enrichment evidence'):
            support = pl.read_csv(target / 'go_hierarchy_labels.csv', schema_overrides={'community': pl.String})
            st.dataframe(support.filter(pl.col('community') == cluster), hide_index=True)
            st.dataframe(evidence.head(200), hide_index=True)
            st.caption('Up to 200 enrichment rows; q-values retain the saved annotation correction family described above.')
        text_path = target / 'uniprot_function_labels.parquet'
        if text_path.is_file():
            text = pl.read_parquet(text_path).filter(pl.col('community') == cluster)
            if text.height:
                with st.expander('Swiss-Prot FUNCTION keywords and phrases'):
                    st.dataframe(text, hide_index=True)
        st.download_button('Download selected cluster annotations', lambda: bundle_zip(target),
            'selected_cluster_annotation.zip', 'application/zip', key=f'{key}_{target.name}_download',
            on_click='ignore')
        with st.expander('Annotation provenance and coverage'):
            st.json(json.loads((target / 'manifest.json').read_text(encoding='utf-8')))
    except (OSError, ValueError) as exc:
        st.warning(f'Annotation unavailable: {exc}')


def hierarchy_panel(bundle, cluster, key):
    """Inspect overlapping BP/MF hierarchy evidence without rerunning annotation."""
    import json
    from clusterannotation import hierarchy_sankey
    from moonlighting import _ontology
    bundle = Path(bundle)
    support = bundle / 'go_hierarchy_support.parquet'
    if not support.is_file():
        st.info('Recalculate annotation to enable GO hierarchy diagrams for this older bundle.')
        return
    st.subheader('GO hierarchy')
    library = st.segmented_control('GO hierarchy', ['GO_BP', 'GO_MF'], default='GO_BP',
        required=True, key=f'{key}_library', format_func=lambda value: {
            'GO_BP': 'Biological process (BP)', 'GO_MF': 'Molecular function (MF)'}[value])
    subset = st.segmented_control('Cluster protein subset', ['full', 'core', 'interface'],
        default='full', required=True, key=f'{key}_subset')
    c1, c2 = st.columns(2)
    top = c1.slider('Maximum terms per depth', 2, 20, 8, key=f'{key}_terms')
    minimum = c2.number_input('Minimum supporting proteins', min_value=1, value=2, step=1,
                              key=f'{key}_hits')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    obo = Path(manifest['sources']['go']['sources']['obo']['path'])
    _, _, _, ancestors = _ontology(obo)
    votes = pl.read_parquet(support)
    available = votes.filter((pl.col('community') == str(cluster)) &
        (pl.col('library') == library) & (pl.col('subset') == subset))
    if available.is_empty():
        st.info('No GO hierarchy support is available for this selection.')
        return
    low, high = int(available['depth'].min()), int(available['depth'].max())
    if low < high:
        default_low = min(max(2, low), high)
        default_high = min(max(4, default_low), high)
        depth_min, depth_max = st.slider('GO depth range (minimum–maximum)', low, high,
            (default_low, default_high), key=f'{key}_{library}_{subset}_depth_{low}_{high}')
    else:
        depth_min = depth_max = low
        st.caption(f'Available GO depth: {low}')
    evidence = pl.read_parquet(bundle / 'enrichment.parquet')
    fig, counts, links = hierarchy_sankey(votes, evidence, ancestors, cluster, library, subset,
        depths=tuple(range(depth_min, depth_max + 1)), max_terms=int(top), min_hits=int(minimum))
    if counts.is_empty():
        st.info('No supported terms in the selected depth range meet these display settings.')
        return
    if links.is_empty():
        st.info('No GO ancestor links connect the displayed depth levels. Protein support is listed below.')
    if counts.height:
        st.plotly_chart(fig, width='stretch', theme=None, key=f'{key}_sankey',
                        alt=f'{library} ancestor relationships and protein support for cluster {cluster}')
        st.download_button('Download hierarchy diagram (HTML)', fig.to_html(include_plotlyjs=True),
            f'cluster_{cluster}_{library}_hierarchy.html', 'text/html', key=f'{key}_html')
    st.caption('Depth is minimum distance from a GO root through is_a/part_of. Links connect actual '
        'ancestors at adjacent displayed depths; width counts distinct supporting proteins shared by '
        'both terms. Annotations overlap, so widths are not conserved flows and percentages need not '
        'sum to 100%. Showing top terms by support per depth; the table retains all qualifying terms. '
        'Node stars use the original one-sided hypergeometric tests against the full network and the '
        'bundle’s complete BH family, including hidden terms: * q < 0.05; ** q < 0.01; '
        '*** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable. '
        'Support counts and link widths are descriptive and have no separate significance test.')
    with st.expander('All qualifying hierarchy labels and protein links'):
        st.dataframe(counts, hide_index=True, alt='GO support in the selected depth range')
        if links.height:
            st.dataframe(links, hide_index=True, alt='Ancestor links and their supporting proteins')


def atlas_memberships(node_frame: pl.DataFrame) -> pl.DataFrame:
    """Preserve every membership emitted by Atlas clustering."""
    return (
        node_frame.select('protein', pl.col('communities_str').str.split(';').alias('community'))
        .explode('community')
    )


def bundle_zip(bundle: Path) -> bytes:
    archive = BytesIO()
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
        for name in ('cluster_labels.csv', 'enrichment.parquet', 'manifest.json',
                     'go_hierarchy_labels.csv', 'label_summary.csv',
                     'uniprot_function_labels.parquet', 'go_hierarchy_support.parquet'):
            if not (bundle / name).is_file():
                continue
            zipped.write(bundle / name, name)
    return archive.getvalue()


def annotation_panel(key, identity, load_inputs, data_root: Path, *, cytoscape=None,
                     button='Annotate clusters'):
    """Run on demand and expose the current run and selected cluster to its caller."""
    paths = tuple(data_root / path.relative_to(DEFAULT_DATA) for path in (DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME))
    gaf, obo, reactome = paths
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        st.info(f'Add the local annotation files before running: {", ".join(missing)}.')
        return None, None
    mode = st.segmented_control('Annotation scope', ['All clusters', 'Selected cluster'],
        default='All clusters', required=True, key=f'{key}_scope')
    selected_community, loaded = None, None
    if mode == 'Selected cluster':
        loaded = load_inputs()
        membership = loaded[0].select('community', 'protein').drop_nulls('community').unique()
        sizes = membership.group_by('community').agg(pl.col('protein').n_unique().alias('proteins')).sort('proteins', descending=True)
        event = st.dataframe(sizes, hide_index=True, on_select='rerun', selection_mode='single-row',
            key=f'{key}_choose_cluster_{hash(str(identity))}', alt='Select a cluster to annotate')
        if event.selection.rows:
            selected_community = str(sizes['community'][event.selection.rows[0]])
        else:
            st.info('Select one cluster row, then run annotation.')
    functions = data_root / DEFAULT_FUNCTIONS.relative_to(DEFAULT_DATA)
    functions = functions if functions.is_file() else None
    signature = (identity, tuple(_signature(path) for path in paths), cytoscape,
                 selected_community, mode, _signature(Path(__file__)),
                 _signature(Path(__file__).with_name('clusterannotation.py')), _signature(functions) if functions else None,
                 *[_signature(p) if p.is_file() else None for p in
                   (data_root / 'generif/generifs_basic.gz', data_root / 'generif/abstracts.tsv',
                    data_root / 'mappings/generif_geneid_uniprot.json')])
    state_key = f'{key}_result'
    st.caption('GO and Reactome labels use exclusive cores and shared interfaces determined from all memberships. '
               'Tested clusters, subsets, and libraries share one FDR correction; selecting one cluster '
               'limits that family to its subsets and terms. Background is the full network.')
    if st.button('Autoannotate selected cluster' if mode == 'Selected cluster' else button,
                 key=f'{key}_run', type='primary',
                 disabled=mode == 'Selected cluster' and selected_community is None):
        st.session_state.pop(state_key, None)
        try:
            with st.spinner('Calculating GO and Reactome enrichment…'):
                frame, population, source = loaded if loaded is not None else load_inputs()
                result = annotate_memberships(
                    frame, population, source, gaf=gaf, obo=obo, reactome=reactome, cytoscape=cytoscape,
                    communities=[selected_community] if mode == 'Selected cluster' else None,
                    uniprot_functions=functions,
                    generif=data_root / 'generif/generifs_basic.gz',
                    abstracts=data_root / 'generif/abstracts.tsv',
                    gene_mapping=data_root / 'mappings/generif_geneid_uniprot.json',
                )
                bundle = data_root / 'analysis_runs' / f'cluster_annotation_{uuid4().hex[:16]}'
                write_annotation_bundle(bundle, result)
                result.pop('evidence')
                st.session_state[state_key] = {'signature': signature, 'bundle': bundle, **result}
        except Exception as exc:
            st.error(f'Cluster annotation failed. {exc}')
    result = st.session_state.get(state_key)
    if result is None:
        return None, None
    if result['signature'] != signature:
        st.info('Inputs changed. Run annotation again for the current selection.')
        return None, None

    labels = result['labels']
    manifest = result['manifest']
    st.caption(f'{labels.height:,} clusters · {manifest["coverage"]["background"]:,} background proteins · '
               f'{manifest["tests"]:,} enrichment tests')
    display_columns = ['community', 'full_size', 'primary_label', 'interface_label', 'full_context_label']
    comparison = manifest['cytoscape_comparison']
    if comparison['status'] != 'not_requested':
        display_columns += ['autoannotate_description_label', 'autoannotate_terms_label']
        if comparison['status'] != 'complete':
            st.warning(f'Cytoscape comparison is {comparison["status"]}. Python enrichment is available.')
            with st.expander('Cytoscape comparison details'):
                st.json(comparison)
    selection = st.dataframe(
        labels.select(display_columns), hide_index=True, alt='Functional labels by cluster',
        on_select='rerun', selection_mode='single-row', key=f'{key}_{result["bundle"].name}_labels',
        column_config={
            'community': 'Cluster', 'full_size': 'Proteins', 'primary_label': 'Core identity',
            'interface_label': 'Shared interface', 'full_context_label': 'Full-cluster context',
            'autoannotate_description_label': 'Cytoscape · descriptions',
            'autoannotate_terms_label': 'Cytoscape · GO and Reactome',
        },
    )
    names = {
        row['community']: row['primary_label'] or row['full_context_label'] or row['interface_label']
        for row in labels.iter_rows(named=True)
    }
    cluster = labels['community'][selection.selection.rows[0]] if selection.selection.rows else selected_community
    if cluster is None:
        st.caption('Select a label row to inspect its hierarchy and evidence.')
    if cluster is not None:
        details = labels.filter(pl.col('community') == cluster)
        with st.expander('Selected cluster metadata and label details'):
            st.dataframe(details.transpose(include_header=True, header_name='Field', column_names=['Value']),
                hide_index=True, alt='Selected cluster metadata listed vertically')
        complete_evidence = pl.read_parquet(result['bundle'] / 'enrichment.parquet').filter(
            pl.col('community') == cluster)
        st.subheader(f'Annotation report · cluster {cluster}')
        readable_annotation_sections(complete_evidence)
        uniprot_function_report([result['bundle'] / 'uniprot_function_labels.parquet'], cluster)
        hierarchy_panel(result['bundle'], cluster, f'{key}_{cluster}_hierarchy')
        evidence = (
            pl.scan_parquet(result['bundle'] / 'enrichment.parquet').filter(pl.col('community') == cluster)
            .sort('qvalue', 'pvalue', 'library', 'term_id').head(200).collect()
        )
        if evidence.is_empty():
            st.caption('No eligible terms had protein hits for this cluster.')
        else:
            st.caption('Up to 200 terms with hits, ordered by adjusted p-value. The download contains all evidence.')
            st.dataframe(evidence, hide_index=True, alt='Enrichment evidence for the selected cluster')
    bundle = result['bundle']
    st.download_button(
        'Download annotation bundle', lambda: bundle_zip(bundle), f'{bundle.name}.zip',
        'application/zip', on_click='ignore', key=f'{key}_download',
    )
    return result, cluster
