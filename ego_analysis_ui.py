"""Streamlit views for exact ego levels and optional GO interpretation."""
from __future__ import annotations

import hashlib
import html
import json
import math
import textwrap

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from moonlighting import (
    COLORSCALE, cached_analysis_run, cross_network_metric_figure, significance_label, stack_network_tables,
)
from moonlighting_extensions import (
    ego_component_enrichment, ego_go_composition, ego_go_pie_table, ego_level_analysis,
    ego_level_comparison, ego_neighborhood, go_ontology_levels,
)
from networks import matching_nodes


def category_color(term):
    """Stable across networks, reruns and different category subsets."""
    if term == '__unannotated__':
        return '#B8B8B8'
    if term == '__no_level__':
        return '#737373'
    if term == '__other__':
        return '#D6D6D6'
    hue = int(hashlib.sha256(str(term).encode()).hexdigest()[:8], 16) % 360
    return f'hsl({hue},60%,45%)'


def ego_network_figure(snapshot, remove_ego=False, seed=42):
    nodes = snapshot['nodes'].filter(~pl.col('is_ego')) if remove_ego else snapshot['nodes']
    edges = snapshot['edges'].filter(pl.col('source').is_in(nodes['protein'].implode())
                                     & pl.col('target').is_in(nodes['protein'].implode()))
    if nodes.height > 300 or edges.height > 3000:
        raise ValueError('Diagram unavailable above 300 nodes or 3,000 edges; exact tables are retained.')
    network = nx.Graph()
    network.add_nodes_from(nodes['protein'].to_list())  # Preserve isolates after ego removal.
    network.add_edges_from(edges.select('source', 'target').iter_rows())
    positions = nx.spring_layout(network, seed=int(seed)) if len(network) else {}
    x, y = [], []
    for a, b in network.edges:
        x.extend([float(positions[a][0]), float(positions[b][0]), None])
        y.extend([float(positions[a][1]), float(positions[b][1]), None])
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#C4C4C4', width=1),
                              hoverinfo='skip', showlegend=False))
    node_x, node_y, colors, symbols, sizes, labels = [], [], [], [], [], []
    for row in nodes.iter_rows(named=True):
        position = positions[row['protein']]
        node_x.append(float(position[0])); node_y.append(float(position[1]))
        colors.append('#E69F00' if row['is_ego'] else category_color(row['partner_component'])
                      if row['shell'] == 1 else '#8B8B8B')
        symbols.append('star' if row['is_ego'] else 'circle' if row['shell'] == 1 else 'square')
        sizes.append(21 if row['is_ego'] else 11)
        labels.append(f"{html.escape(row['protein'])}<br>Shell {row['shell']}<br>Partner component: {row['partner_component'] or '—'}")
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', text=labels, hovertemplate='%{text}<extra></extra>',
                             marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=0.5, color='white')),
                             showlegend=False))
    fig.update_layout(template='plotly_white', height=480, margin=dict(l=10, r=10, t=45, b=10),
                      title=f"{snapshot['summary']['protein']} · ego level {snapshot['summary']['level']}"
                            + (' · ego removed' if remove_ego else ''),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'))
    return fig


def ego_go_pie_figure(composition, title):
    view = ego_go_pie_table(composition)
    view = view.with_columns(
        pl.when(pl.col('term').str.starts_with('__')).then(pl.col('name'))
        .otherwise(pl.concat_str('term', pl.lit(' · '), pl.col('name').str.slice(0, 42))).alias('label'))
    fig = px.pie(view, values='weight', names='label', color='term',
                 color_discrete_map={t: category_color(t) for t in view['term']},
                 custom_data=['name', 'term', 'protein_count', 'coverage'], title=title)
    fig.update_traces(textinfo='percent', hovertemplate=(
        '%{customdata[0]}<br>%{customdata[1]}<br>Weighted protein equivalents: %{value:.3g}'
        '<br>%{customdata[2]} distinct proteins; coverage %{customdata[3]:.1%}'
        '<br>Pie share: %{percent}<extra></extra>'))
    fig.update_layout(template='plotly_white', height=500, legend=dict(orientation='h', y=-0.1),
                      margin=dict(t=60, b=160))
    return fig


def ego_effect_figure(tests, set_name):
    """Put long metric labels on rows; encode only finite effects and adjusted tests."""
    metrics = tests['metric'].unique(maintain_order=True).to_list()
    networks = tests['network'].unique(maintain_order=True).to_list()
    labels = []
    for metric in metrics:
        level, _, name = metric.partition(' · ')
        level = '2.0 − 1.5' if level == 'change' else level
        labels.append('<br>'.join(textwrap.wrap(f'{level} · {name.replace("_", " ")}', width=42)))
    lookup = {(row['metric'], row['network']): row for row in tests.iter_rows(named=True)}
    z, text, hover, missing = [], [], [], []
    for metric in metrics:
        zr, tr, hr, mr = [], [], [], []
        for network in networks:
            row = lookup[(metric, network)]
            value = row['rank_biserial']
            finite = value is not None and math.isfinite(value)
            stars = significance_label(row['fdr_bh'])
            zr.append(value if finite else None)
            tr.append(f'{value:+.2f}<br>{stars}' if finite else '')
            hr.append(f"{html.escape(network)}<br>{html.escape(metric)}<br>Set n: {row['query_n']}"
                      f"<br>Background n: {row['background_n']}<br>BH q: {row['fdr_bh']}<br>{stars}")
            mr.append(None if finite else 1)
        z.append(zr); text.append(tr); hover.append(hr); missing.append(mr)
    fig = go.Figure(go.Heatmap(z=missing, x=networks, y=labels, colorscale=[[0, '#E5E7EB'], [1, '#E5E7EB']],
                               showscale=False, hoverinfo='skip', xgap=2, ygap=2))
    fig.add_trace(go.Heatmap(z=z, x=networks, y=labels, text=text, texttemplate='%{text}',
                             hovertext=hover, hovertemplate='%{hovertext}<extra></extra>', hoverongaps=False,
                             colorscale=COLORSCALE, zmin=-1, zmax=1, xgap=2, ygap=2,
                             colorbar=dict(title='Rank-biserial<br>effect', tickvals=[-1, 0, 1])))
    for i, row in enumerate(missing):
        for j, value in enumerate(row):
            if value is not None:
                fig.add_annotation(x=networks[j], y=labels[i], text='n/a', showarrow=False, font=dict(color='#555'))
    fig.update_layout(template='plotly_white', title=f'{set_name}: ego effects versus degree-matched background',
                      xaxis_title='Interactome', yaxis_title='Level · metric', yaxis_autorange='reversed',
                      height=max(450, 150 + 38 * len(metrics)), margin=dict(l=325, r=90, t=75, b=60))
    return fig


def _summary_table(summary):
    return pl.DataFrame({'Metric': list(summary), 'Value': [str(v) if v is not None else 'n/a' for v in summary.values()]})


def _group_members(snapshot, scope):
    nodes = snapshot['nodes']
    if scope == 'Direct partners':
        return nodes.filter(pl.col('shell') == 1)['protein'].to_list()
    if scope == 'Second shell':
        return nodes.filter(pl.col('shell') == 2)['protein'].to_list()
    if scope == 'All non-ego proteins':
        return nodes.filter(~pl.col('is_ego'))['protein'].to_list()
    return nodes.filter(pl.col('partner_component') == scope)['protein'].to_list()


def ego_levels_panel(*, source_paths, graph_for, protein_set, set_name, seed, run_networks,
                     context, data_dir, annotation_file, ontology_path, annotation_signatures):
    st.info('Level 1.0 contains spokes only; 1.5 also includes all direct-partner links; '
            '2.0 includes the full induced two-hop neighborhood. Ego removal preserves isolated partners.')
    with st.expander('What these ego measurements mean'):
        st.markdown(
            '**Partner density/clustering** measures links among direct partners. **Effective size** describes '
            'nonredundant contacts. After ego removal, more components, higher component entropy or higher fragmentation '
            'can suggest brokerage between partner groups. At level 2, **partner-pair reconnection** measures how many '
            'previously disconnected direct-partner pairs become connected through the second shell. '
            'Undefined denominators remain unavailable. Density and clustering are aliases and are not tested twice.'
        )
        st.markdown(
            'Confidence scores are native association scores, not binding strengths. Topology does not establish '
            'separate functions, simultaneous binding or binding exclusivity. Component IDs are local to each ego/network; '
            'matching colors do not establish identical components across interactomes.'
        )
    if not source_paths:
        st.info('Select a source network with at most 500,000 edges; use a saved threshold version for larger networks.')
        return
    c1, c2 = st.columns(2)
    max_nodes = c1.number_input('Maximum nodes per analytical neighborhood', 1, 2000, 2000, key='ego_max_nodes')
    max_edges = c2.number_input('Maximum edges per analytical neighborhood', 1, 20000, 20000, key='ego_max_edges')
    budgets = {'max_nodes': int(max_nodes), 'max_edges': int(max_edges), 'ego_version': 1}
    mode = st.segmented_control('Ego workflow', ['Protein-set comparison', 'Individual ego explorer'],
                                default='Protein-set comparison', required=True, key='ego_workflow')
    if mode == 'Protein-set comparison':
        st.caption(f'All eligible {set_name} proteins are analyzed. Controls are sampled without replacement in '
                   'floor(log₂ degree) bins. Unmatched query proteins remain in descriptive tables; '
                   'each test uses finite matched pairs. Oversized neighborhoods are skipped, never truncated.')
        results = run_networks('ego_levels', 'Ego levels', budgets,
                               lambda net: ego_level_analysis(graph_for(net), protein_set, seed=int(seed),
                                                               set_name=set_name, max_nodes=int(max_nodes), max_edges=int(max_edges)),
                               advanced=True)
        if not results:
            return
        pooled = ego_level_comparison(results)
        metadata = dict(context, analysis='ego_levels_comparison', parameters=budgets,
                        correction=pooled['correction'], matched_results=list(results))
        # The per-network bundles above contain descriptive values. This bundle also preserves
        # the combined correction family, which cannot be computed inside one network's run.
        entry = cached_analysis_run('ego_levels_comparison', metadata, lambda: {'comparison': pooled, 'networks': results}, data_dir)
        st.download_button('Download ego levels and globally adjusted comparisons', entry['archive'],
                           'ego_levels_comparison.zip', 'application/zip', key='ego_levels_pooled_download')
        summary = pl.from_dicts([{'Network': n, **r['notes']} for n, r in results.items()])
        st.dataframe(summary, hide_index=True, width='stretch', key='ego_batch_summary')
        tests = pooled['tests']
        fig = ego_effect_figure(tests, set_name)
        st.plotly_chart(fig, width='stretch', key='ego_batch_heatmap', alt=f'Ego metric effects for {set_name}')
        st.caption('Positive effects mean higher values in the protein set. Two-sided Mann–Whitney tests; '
                   'BH across all finite network × level × metric tests, including cross-level changes. '
                   '`*` q < 0.05; `**` q < 0.01; `***` q < 0.001; `****` q < 0.0001; ns: not significant; n/a: unavailable. '
                   'These are exploratory comparisons between dependent network observations.')
        available = [m for m in tests['metric'].unique(maintain_order=True) if m in pooled['metrics'].columns]
        if available:
            metric = st.selectbox('Ego metric distribution', available, key='ego_batch_metric')
            kind = st.segmented_control('Distribution plot', ['Box', 'Violin'], default='Box', required=True, key='ego_batch_plot')
            fig = cross_network_metric_figure(pooled['metrics'], metric, tests, plot_kind=kind.lower())
            st.plotly_chart(fig, width='stretch', key='ego_batch_distribution', alt='Matched finite ego metric distributions')
        with st.expander('All egos, exclusions, matching, memberships and tests'):
            st.dataframe(tests, hide_index=True, key='ego_batch_tests')
            for name in ('metrics', 'matches', 'memberships'):
                st.dataframe(stack_network_tables({n: r[name] for n, r in results.items()}),
                             hide_index=True, key=f'ego_batch_{name}')
        return

    accession = st.text_input('Ego UniProt accession or exact graph node label', key='ego_accession').strip()
    if not accession:
        st.caption(f'Enter an accession from {set_name}, or another network protein for an individual case study.')
        return
    choices = {}
    for net in source_paths:
        graph = graph_for(net)
        candidates = sorted(matching_nodes(graph['nodes'], {accession}))
        if accession in graph['index']:
            candidates = [accession]
        if len(candidates) > 1:
            choices[net] = st.selectbox(f'{net}: choose the distinct mapped graph node', candidates, key=f'ego_ambiguous_{net}_{accession}')
            st.caption(f'{net}: several graph nodes carry this accession; these nodes are not merged.')
        elif candidates:
            choices[net] = candidates[0]
        else:
            st.info(f'{accession} is absent from {net}.')
    if not choices:
        return
    level = st.segmented_control('Ego-network level', ['1.0', '1.5', '2.0'], default='1.5', required=True, key='ego_level')
    remove = st.toggle('Remove ego from diagram', value=False, key='ego_remove')
    exact = st.checkbox('Exact local betweenness and efficiency (up to 500 nodes / 5,000 edges)', key='ego_exact')
    parameters = dict(budgets, nodes=choices, level=level, exact_centrality=exact)

    def extract(net):
        if net not in choices:
            raise ValueError('Ego absent from this network.')
        return ego_neighborhood(graph_for(net), choices[net], level, int(max_nodes), int(max_edges), exact,
                                source_path=source_paths[net])

    snapshots = run_networks('ego_explorer', 'Individual ego', parameters, extract, advanced=True)
    valid = {}
    for net, snapshot in snapshots.items():
        st.markdown(f'##### {net} · {set_name} context · {snapshot["summary"]["protein"]}')
        if snapshot['summary']['status'] != 'ok':
            st.warning(snapshot['summary']['reason'])
            continue
        valid[net] = snapshot
        st.dataframe(_summary_table(snapshot['summary']), hide_index=True, key=f'ego_summary_{net}')
        try:
            fig = ego_network_figure(snapshot, remove_ego=remove, seed=seed)
            st.plotly_chart(fig, width='stretch', key=f'ego_network_{net}', alt=f'{net} ego network for {accession}')
        except ValueError as exc:
            st.info(str(exc))
        st.caption('Star: ego; circles: direct partners colored by their level-1.5 component; squares: second shell. '
                   'Removing the ego changes the diagram only; the table labels before/after metrics explicitly.')
        with st.expander(f'{net}: exact node and edge tables'):
            st.dataframe(snapshot['nodes'], hide_index=True, key=f'ego_nodes_{net}')
            st.dataframe(snapshot['edges'], hide_index=True, key=f'ego_edges_{net}')
    if not valid or not st.toggle('Add optional GO composition and enrichment', value=False, key='ego_go_enabled'):
        return
    _ego_go_panel(valid=valid, graph_for=graph_for, run_networks=run_networks, parameters=parameters,
                  annotation_file=annotation_file, ontology_path=ontology_path,
                  annotation_signatures=annotation_signatures, set_name=set_name)


def _ego_go_panel(*, valid, graph_for, run_networks, parameters, annotation_file, ontology_path,
                  annotation_signatures, set_name):
    try:
        levels, provenance = go_ontology_levels(ontology_path)
        annotation = annotation_file()
    except (ImportError, OSError, ValueError) as exc:
        st.warning(f'GO views unavailable: {exc}. Topology results remain available.')
        return
    c1, c2 = st.columns(2)
    aspect = c1.selectbox('GO aspect', ['BP', 'MF', 'CC'], key='ego_go_aspect')
    aspect_maximum = levels.filter(pl.col('aspect') == {'BP': 'P', 'MF': 'F', 'CC': 'C'}[aspect])['go_level'].max()
    if aspect_maximum is None:
        st.info(f'The supplied ontology has no {aspect} terms.')
        return
    maximum = int(aspect_maximum)
    go_level = c2.slider('GO ontology level (independent of ego level)', 1, max(3, maximum), 3, key='ego_go_level')
    exclude_iea = st.checkbox('Exclude IEA-only GO annotations', key='ego_go_exclude_iea')
    st.caption('GO level is the shortest is_a distance from the aspect root (root = 0). '
               'Equal levels are not equal biological specificity. Existing is_a/part_of propagation is reused. '
               'The ego is excluded from functional groups.')
    components = sorted({c for r in valid.values() for c in r['nodes']['partner_component'].drop_nulls().unique()})
    scope = st.selectbox('Proteins summarized in GO pie', ['Direct partners', 'All non-ego proteins', 'Second shell', *components],
                         key='ego_go_scope')
    go_inputs = dict(parameters, aspect=aspect, exclude_iea=exclude_iea, annotation_inputs=annotation_signatures(),
                     ontology=provenance, annotation_path=str(annotation))
    annotation_metadata = annotation.with_suffix('.json')
    if annotation_metadata.exists():
        go_inputs['annotation_provenance'] = json.loads(annotation_metadata.read_text(encoding='utf-8'))

    def composition(net):
        if net not in valid:
            raise ValueError('No eligible ego neighborhood in this network.')
        result = ego_go_composition(_group_members(valid[net], scope), annotation, levels, aspect, int(go_level), exclude_iea)
        result['ontology'] = provenance
        return result

    results = run_networks('ego_go_composition', 'Ego GO composition', dict(go_inputs, scope=scope, go_level=int(go_level)), composition, advanced=True)
    for net, result in results.items():
        summary = result['summary']
        st.markdown(f'##### {net} · {scope} · {set_name} context')
        st.caption(f"{summary['group_size']} group proteins; {summary['annotated_proteins']} annotated in {aspect}; "
                   f"{summary['mapped_proteins']} mapped at GO level {go_level}.")
        if summary['group_size']:
            fig = ego_go_pie_figure(result, f'{net}: {scope} · {aspect} level {go_level}')
            st.plotly_chart(fig, width='stretch', key=f'ego_go_pie_{net}', alt=f'{net} fractional GO composition')
        else:
            st.info('This functional group is empty at the selected ego level.')
        majority = result['majority']
        if majority.height:
            st.write('Strict majority categories (>50% of all group proteins):')
            st.dataframe(majority.select('term', 'name', 'protein_count', 'coverage'), hide_index=True, key=f'ego_go_majority_{net}')
        else:
            st.info('No majority category (>50% coverage).')
        with st.expander(f'{net}: full GO composition and node memberships'):
            st.dataframe(result['table'], hide_index=True, key=f'ego_go_full_{net}')
            st.dataframe(result['memberships'], hide_index=True, key=f'ego_go_memberships_{net}')
    st.caption('Each protein contributes one total pie unit split equally across its distinct GO categories. '
               'Coverage counts proteins in each category and can overlap. Majority labels and pie shares are descriptive; '
               'they receive no significance stars and do not establish enrichment or mutually exclusive functions.')
    if not st.toggle('Test GO enrichment of direct-partner components', value=False, key='ego_go_enrich'):
        return
    selections = {}
    for net, snapshot in valid.items():
        counts = snapshot['nodes'].filter(pl.col('partner_component').is_not_null()).group_by('partner_component').len().sort(
            ['len', 'partner_component'], descending=[True, False])
        options = counts['partner_component'].to_list()
        selections[net] = st.multiselect(f'{net}: components for enrichment', options, default=options[:5], max_selections=20,
                                         key=f'ego_go_components_{net}_{snapshot["summary"]["protein"]}')
    minimum = st.number_input('Minimum background proteins per eligible GO term', 1, 10000, 5, key='ego_go_min_size')

    def enrich(net):
        if net not in valid:
            raise ValueError('No eligible ego neighborhood in this network.')
        result = ego_component_enrichment(valid[net], graph_for(net), annotation, selections[net], levels,
                                          aspect, int(minimum), exclude_iea)
        result['ontology'] = provenance
        return result

    # GO display level and pie scope are deliberately absent from the enrichment signature.
    enriched = run_networks('ego_component_enrichment', 'Ego component GO enrichment',
                            dict(go_inputs, components=selections, min_size=int(minimum)), enrich, advanced=True)
    for net, result in enriched.items():
        view = result['table'].filter((pl.col('go_level') == int(go_level)) & (pl.col('query_hits') > 0)).sort('qvalue').head(20)
        if view.height:
            view = view.with_columns(pl.concat_str('component', pl.lit(' · '), 'term', pl.lit(' · '),
                                                   pl.col('name').str.slice(0, 55)).alias('label'))
            fig = px.bar(view, x='fold', y='label', text='significance', orientation='h',
                         hover_data=['query_hits', 'background_hits', 'qvalue'],
                         title=f'{net}: component enrichment · displayed GO level {go_level}')
            fig.update_layout(template='plotly_white', yaxis_autorange='reversed', height=max(400, 28 * view.height + 140),
                              xaxis_title='Fold enrichment', yaxis_title='Partner component · GO term', margin=dict(l=100, r=65))
            fig.update_traces(textposition='outside', cliponaxis=False)
            st.plotly_chart(fig, width='stretch', key=f'ego_go_enrich_plot_{net}', alt=f'{net} adjusted component GO enrichment')
        else:
            st.info(f'{net}: no hit terms at this display level; inspect the full tested-term table.')
        with st.expander(f'{net}: all eligible component GO tests, including zero hits'):
            st.dataframe(result['table'], hide_index=True, key=f'ego_go_enrich_table_{net}')
            st.dataframe(result['summaries'], hide_index=True, key=f'ego_go_enrich_summary_{net}')
    st.caption('One-sided hypergeometric enrichment against the full selected network, including unannotated nodes. '
               'BH spans all selected component × eligible-term tests within each network, including zero hits. '
               'The GO level only filters the display after correction. `*` q < .05; `**` q < .01; `***` q < .001; '
               '`****` q < .0001; ns: not significant; n/a: unavailable. This family is separate from topology tests.')
