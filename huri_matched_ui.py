"""HuRI-anchored MoonLitDB sensitivity analysis, computed only on request."""
import polars as pl
import plotly.graph_objects as go
import streamlit as st

from moonlighting import bh_adjust, cohort_ids, scoped_graph, shortest_path_monte_carlo, significance_label


def accession_labels(graph):
    return {accession for node in graph['nodes'] for accession in str(node).split('|')}


def render_huri_matched(moon, datasets, active_nets, graph_for, edges_of, run_items,
                        table, render_chart, reviewed_only, min_pubs, scope, seed, edge_limit):
    st.header('HuRI-matched MoonLitDB protein set')
    st.info('This sensitivity test holds MoonLitDB membership constant to help distinguish protein coverage '
            'effects from network structure effects. Shortest paths use all proteins in each comparison '
            'population, including proteins outside MoonLitDB. It does not establish a biological cause.')
    references = [name for name in datasets if name.lower().startswith('huri')]
    if not references:
        st.warning('A prepared HuRI network is required for this test.')
        return
    reference = st.selectbox('HuRI reference network', references, key='huri_matched_reference')
    st.caption('Uses MoonLitDB regardless of the sidebar protein source; the reviewed-entry and publication filters still apply.')
    moonlitdb, _ = cohort_ids(moon, ['moonlitdb'], False, reviewed_only, min_pubs)
    anchor = moonlitdb & accession_labels(graph_for(reference))
    eligible = [name for name in dict.fromkeys([reference, *active_nets]) if edges_of(name) <= edge_limit]
    skipped = [name for name in dict.fromkeys([reference, *active_nets]) if name not in eligible]
    if skipped:
        st.caption('Excluded by the 500,000-edge guard: ' + ', '.join(skipped))
    if reference not in eligible:
        st.warning('Select a HuRI reference with at most 500,000 edges.')
        return
    populations = {name: accession_labels(scoped_graph(graph_for(name), scope)) for name in eligible}
    shared = anchor.intersection(*populations.values())
    mode = st.radio('Protein membership', ['Identical proteins in every comparison population',
                    'All HuRI MoonLitDB proteins available in each population'], key='huri_matched_mode')
    fixed = mode.startswith('Identical')
    tested = shared if fixed else anchor
    st.write(f'MoonLitDB mapped proteins: **{len(moonlitdb):,}** · In HuRI: **{len(anchor):,}** · '
             f'Shared across eligible comparison populations: **{len(shared):,}**')
    table(pl.DataFrame([{'Network': name, 'HuRI MoonLitDB accessions': len(anchor),
                        'Available in comparison population': len(anchor & labels),
                        'Missing from comparison population': len(anchor - labels),
                        'Tested accessions': len(tested & labels)} for name, labels in populations.items()]),
          'huri_matched_coverage')
    if not fixed:
        st.warning('Membership varies where HuRI proteins are absent. Use the identical-proteins option for the strict coverage control.')
    st.download_button('Download tested accession set', '\n'.join(sorted(tested)),
                       'huri_matched_moonlitdb.txt', key='huri_matched_accessions')
    with st.expander('Null model and interpretation'):
        st.markdown('Each network uses its own other-protein background, excluding the tested protein set, '
                    'and equally sized random sets matched by floor(log2 degree). Degrees are measured in '
                    'the chosen whole-network or largest-component population. Merged graph nodes count once. '
                    'Mean shortest path includes reachable unordered pairs only; inspect reachability alongside it. '
                    'Negative observed minus null means closer than random, positive means farther apart. '
                    'Network size, topology, ascertainment and degree distributions can still differ. '
                    'These are within-network tests, not a test of the difference between networks.')
    simulations = st.number_input('Monte Carlo simulations', 10, 10000, 1000, 10, key='huri_matched_simulations')
    if len(tested) < 2:
        st.warning('At least two shared accessions are needed; adjust the networks, scope or source filters.')
        return
    parameters = dict(version=1, protein_set='moonlitdb', reviewed_only=bool(reviewed_only),
                      min_publications=int(min_pubs), reference=reference, cohort=sorted(tested), scope=scope,
                      simulations=int(simulations), seed=int(seed), mode=mode)
    results = run_items('huri_matched_moonlitdb', 'HuRI-matched MoonLitDB Monte Carlo',
        {name: list(dict.fromkeys([reference, name])) for name in eligible}, parameters,
        lambda name: shortest_path_monte_carlo(scoped_graph(graph_for(name), scope), tested,
                      permutations=int(simulations), seed=int(seed)), advanced=True, manual=True)
    if not results:
        return
    if len(results) != len(eligible):
        st.info('Waiting for all eligible networks before calculating the complete correction family.')
        return
    rows = []
    for name, result in results.items():
        summary = result['summary']
        for metric, observed, null, effect, p in [
            ('Average shortest path', 'observed', 'null_mean', 'effect_observed_minus_null', 'p_value'),
            ('Reachable pair fraction', 'reachable_pair_fraction', 'reachability_null_mean',
             'reachability_effect_observed_minus_null', 'reachability_p_value')]:
            rows.append(dict(Network=name, Metric=metric, Observed=summary[observed],
                             Null=summary[null], Effect=summary[effect], p=summary[p],
                             Nodes=summary['query_nodes']))
    frame = pl.DataFrame(rows).with_columns(pl.Series('q', bh_adjust([row['p'] for row in rows])))
    frame = frame.with_columns(pl.Series('Significance', [significance_label(q) for q in frame['q']]))
    table(frame, 'huri_matched_summary')
    st.download_button('Download adjusted comparison summary', frame.write_csv(),
                       'huri_matched_moonlitdb_summary.csv', 'text/csv', key='huri_matched_summary_csv')
    for metric, column in [('Average shortest path', 'mean_shortest_path'),
                           ('Reachable pair fraction', 'reachable_pair_fraction')]:
        fig = go.Figure()
        for row in frame.filter(pl.col('Metric') == metric).iter_rows(named=True):
            name = row['Network']
            fig.add_trace(go.Box(y=results[name]['null'][column].to_list(), name=name, boxpoints=False))
            fig.add_trace(go.Scatter(x=[name], y=[row['Observed']], mode='markers+text',
                text=[f"{row['Significance']} · Δ={row['Effect']:+.3g}"], textposition='top center',
                cliponaxis=False, marker=dict(color='firebrick', size=11), name=f'{name} observed'))
        fig.update_layout(title=f'HuRI-matched MoonLitDB · {metric}', template='plotly_white',
                          yaxis_title=metric, margin=dict(t=90), showlegend=False)
        render_chart(fig, width='stretch', key='huri_matched_' + column)
    st.caption('Boxes show Monte Carlo null distributions; red points show observed values. '
               'Two-sided inclusive empirical tests with +1 corrections; BH correction across both metrics '
               'and every eligible network in this run. * q < 0.05; ** q < 0.01; *** q < 0.001; '
               '**** q < 0.0001; ns = non-significant; n/a = unavailable. Any undefined trial makes '
               'the shortest-path test unavailable. Δ = observed minus null mean.')
