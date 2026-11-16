"""Browse persisted Monte Carlo results without running network analyses."""
from pathlib import Path
from io import BytesIO
import json
import math

import polars as pl
import plotly.express as px
import streamlit as st

from moonlighting import bh_adjust, significance_label

SIGNIFICANCE_LEGEND = (
    'Unadjusted empirical p-values: * p < 0.05; ** p < 0.01; *** p < 0.001; **** p < 0.0001; '
    'ns: p ≥ 0.05; n/a: unavailable. No multiple-testing correction. '
    'Tests compare the observed protein set with random protein sets.')
METRICS = {
    'mean_shortest_path': ('Average shortest path length', 'Interaction steps', 'shortest_path'),
    'internal_edges': ('Cohesion: internal edges', 'Within-set edges', 'cohesion'),
    'largest_component': ('Largest component', 'Proteins in the induced largest component', 'cohesion'),
}



def plot_details(metric):
    if metric == 'reachable_pair_fraction':
        return 'Reachable pairs', 'Reachable pair fraction', 'reachability'
    return METRICS[metric]


def number(value, signed=False):
    return ('n/a' if value is None or not math.isfinite(value)
            else format(value, '+.5g' if signed else '.5g'))


def empirical_calculation(values, observed, two_sided=False):
    """Count inclusive tails over every stored trial, retaining undefined-test rules."""
    finite = values.filter(values.is_finite())
    valid_observed = observed is not None and math.isfinite(observed)
    count = len(values)
    available = valid_observed and len(finite) == count and count > 0
    lower_count = int((finite <= observed).sum()) if valid_observed else None
    upper_count = int((finite >= observed).sum()) if valid_observed else None
    ties = int((finite == observed).sum()) if valid_observed else None
    lower = upper = p = None
    if available and lower_count is not None and upper_count is not None:
        lower = (1 + lower_count) / (count + 1)
        upper = (1 + upper_count) / (count + 1)
        p = min(1., 2 * min(lower, upper)) if two_sided else upper
    return dict(simulations=count, finite_simulations=len(finite), invalid_simulations=count-len(finite),
                observed=observed if valid_observed else None, lower_count=lower_count, upper_count=upper_count,
                ties=ties, lower_tail=lower, upper_tail=upper, recalculated_p=p,
                test='Two-sided doubled smaller inclusive tail' if two_sided else 'Upper inclusive tail',
                formula='min(1, 2 × min((1 + n≤)/(B + 1), (1 + n≥)/(B + 1)))' if two_sided
                        else '(1 + n≥)/(B + 1)')


def static_distribution_figure(null, row, network, set_name, metadata):
    """Matplotlib figure with legend and inference; methods live in UI expanders."""
    from matplotlib.figure import Figure
    metric = row['measure']
    title, axis, family = plot_details(metric)
    values = null.filter(pl.col(metric).is_finite())[metric].to_list()
    fig = Figure(figsize=(10, 6.8), dpi=160, facecolor='white')
    ax = fig.add_axes((0.10, 0.20, 0.86, 0.60))
    ax.hist(values, bins=35, color='#527fa8', edgecolor='white', label=f'Random protein sets (finite B={len(values)})')
    if row['observed'] is not None and math.isfinite(row['observed']):
        ax.axvline(row['observed'], color='firebrick', linewidth=2.4, label=f"Observed = {number(row['observed'])}")
    if row['null_mean'] is not None and math.isfinite(row['null_mean']):
        ax.axvline(row['null_mean'], color='#555555', linestyle='--', linewidth=1.8,
                   label=f"Null mean = {number(row['null_mean'])}")
    ax.margins(x=0.08)
    ax.set(xlabel=axis, ylabel='Simulations')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2, fontsize=9)
    fig.text(0.10, 0.95, f'{set_name} · {network}\n{title}', fontsize=15, weight='bold', va='top')
    fig.text(0.10, 0.835, f"Effect (observed − null) = {number(row['effect_observed_minus_null'], True)}   "
             f"Empirical p = {number(row.get('p_value'))}   {significance_label(row.get('p_value'))}", fontsize=11)
    return fig


def render_distribution(null, row, network, set_name, metadata, mode, key, output_folder, basename):
    title = plot_details(row['measure'])[0]
    alt = f'{set_name}: observed {title.lower()} versus random protein sets in {network}'
    if mode == 'Interactive':
        st.plotly_chart(distribution_figure(null, row, network, set_name), width='stretch', key=key, alt=alt)
        return
    fig = static_distribution_figure(null, row, network, set_name, metadata)
    png, svg = BytesIO(), BytesIO()
    fig.savefig(png, format='png', bbox_inches='tight', dpi=180)
    fig.savefig(svg, format='svg', bbox_inches='tight')
    output_folder.mkdir(parents=True, exist_ok=True)
    for extension, content in [('png', png.getvalue()), ('svg', svg.getvalue())]:
        (output_folder / f'{basename}.{extension}').write_bytes(content)
    st.image(png.getvalue(), width='stretch', alt=alt)
    with st.container(horizontal=True):
        st.download_button('Download PNG', png.getvalue(), f'{basename}.png', 'image/png', key=f'{key}_png')
        st.download_button('Download SVG', svg.getvalue(), f'{basename}.svg', 'image/svg+xml', key=f'{key}_svg')


def render_calculations(results, entry, row):
    metric = row['measure']
    panel = st.expander(f'Exact calculations · {plot_details(metric)[0]}',
                        key=f'saved_mc_calculation_{metric}', on_change='rerun')
    if not panel.open:
        return
    with panel:
        calc = empirical_calculation(entry['null'][metric], row['observed'], metric in ('mean_shortest_path', 'reachable_pair_fraction'))
        values = entry['null'].filter(pl.col(metric).is_finite())[metric]
        st.markdown(f"**Observed:** {number(row['observed'])} · **Null average:** {number(row['null_mean'])} · "
                    f"**Effect (observed − null average):** {number(row['effect_observed_minus_null'], True)}")
        if len(values):
            st.code(f'Null average = {float(values.sum())!r} / {len(values)} = {float(values.mean())!r}\n'
                    f"Effect = {row['observed']!r} − {row['null_mean']!r} = {row['effect_observed_minus_null']!r}")
        st.markdown('**B** is every stored trial; **n≤** and **n≥** include ties. '
                    'The +1 correction adds one to the tail count and denominator. '
                    'If any trial or the observed statistic is non-finite, the test is unavailable; '
                    'finite-only display values are never used to rescue a test.')
        if calc['recalculated_p'] is not None:
            st.code(f"Lower tail = (1 + {calc['lower_count']}) / ({calc['simulations']} + 1) = {calc['lower_tail']!r}\n"
                    f"Upper tail = (1 + {calc['upper_count']}) / ({calc['simulations']} + 1) = {calc['upper_tail']!r}\n"
                    f"p = {calc['formula']} = {calc['recalculated_p']!r}")
            if row.get('p_value') is not None and not math.isclose(calc['recalculated_p'], row['p_value'], rel_tol=1e-10, abs_tol=1e-12):
                st.warning('The stored p-value differs from the value reconstructed from these trials. '
                           'Stored inference is preserved; inspect the source files before interpreting it.')
        st.caption(SIGNIFICANCE_LEGEND)
        metadata = entry['metadata']
        st.markdown('**Saved calculation settings and provenance**')
        st.json(metadata)
        st.download_button('Download exact calculations', json.dumps({**calc,
            'saved_p': row.get('p_value'), 'saved_fdr_bh': row.get('fdr_bh'), 'multiple_testing_correction': 'None; displayed significance uses raw empirical p-values',
            'metadata': metadata}, indent=2, default=str),
            file_name=f"{entry['metadata']['protein_set']}_{entry['network']}_{metric}_calculations.json",
            mime='application/json', key=f'saved_mc_calculation_download_{metric}')


def load_saved_result(metadata_path):
    """Load a matched triplet, preserving correction over the full saved family."""
    path = Path(metadata_path)
    metadata = json.loads(path.read_text())
    network = Path(metadata['network_path'].replace('\\', '/')).stem
    analysis = metadata['analysis']
    expected = [metric for metric, (_, _, family) in METRICS.items() if family == analysis]
    if not expected:
        raise ValueError('Unknown saved analysis')
    null = pl.read_parquet(path.with_suffix('.parquet'))
    summary = pl.read_csv(path.with_name(f'{path.stem}_summary.csv'), schema_overrides={
        key: pl.Float64 for key in ('observed', 'null_mean', 'effect_observed_minus_null', 'p_value', 'fdr_bh')})
    if set(summary['measure'].to_list()) != set(expected) or not set(expected) <= set(null.columns):
        raise ValueError('Saved summary and distribution metrics do not match')
    if summary.filter((pl.col('protein_set') != metadata['protein_set']) | (pl.col('network') != network)).height:
        raise ValueError('Saved summary has a different protein set or interactome')
    if 'fdr_bh' not in summary.columns:
        summary = summary.with_columns(pl.lit(None, dtype=pl.Float64).alias('fdr_bh'))
    summary = summary.with_columns(pl.col('p_value').map_elements(
        significance_label, return_dtype=pl.String, skip_nulls=False).alias('significance'))
    return {'metadata': metadata, 'network': network, 'null': null, 'summary': summary}


@st.cache_data(show_spinner=False, max_entries=100)
def _cached_result(path, stamp):
    return load_saved_result(path)


def distribution_figure(null, row, network, set_name):
    metric = row['measure']
    title, axis, _ = plot_details(metric)
    finite = null.filter(pl.col(metric).is_finite())
    fig = px.histogram(finite, x=metric, nbins=35, labels={metric: axis},
                       title=f'{title} · {network}', color_discrete_sequence=['#527fa8'])
    observed, effect, q = row['observed'], row['effect_observed_minus_null'], row.get('p_value')
    label = significance_label(q)
    def fmt(value, signed=False):
        return ('n/a' if value is None or not math.isfinite(value)
                else format(value, '+.4g' if signed else '.4g'))
    if observed is not None and math.isfinite(observed):
        fig.add_vline(x=observed, line_color='firebrick', line_width=3, name='Observed protein set', showlegend=True)
    if row['null_mean'] is not None and math.isfinite(row['null_mean']):
        fig.add_vline(x=row['null_mean'], line_color='#555555', line_dash='dash', name='Null mean', showlegend=True)
    fig.update_traces(name='Random protein sets', showlegend=True)
    bounds = [value for value in (finite[metric].min(), finite[metric].max(), observed, row['null_mean'])
              if value is not None and math.isfinite(value)]
    if bounds:
        lower, upper = min(bounds), max(bounds)
        padding = max((upper - lower) * 0.08, abs(lower) * 0.01, 0.05)
        fig.update_xaxes(range=[lower - padding, upper + padding])
    fig.add_annotation(x=0, y=1.05, xref='paper', yref='paper', showarrow=False,
                       xanchor='left', yanchor='bottom',
                       font=dict(size=14),
                       text=f'<b>Observed {fmt(observed)}</b> · <b>Null average {fmt(row["null_mean"])}</b><br>'
                            f'<b>Effect {fmt(effect, True)}</b> · <b>{label}</b><br>'
                            f"Empirical p = {fmt(q)} · {set_name}")
    fig.update_layout(template='plotly_white', height=500, margin=dict(t=175, b=100),
                      yaxis_title='Simulations', bargap=0.05,
                      legend=dict(orientation='h', y=-0.25, x=0))
    return fig


def saved_reachability(entry):
    """Derive reachability inference from all stored trials, including undefined path means."""
    row = entry['summary'].filter(pl.col('measure') == 'mean_shortest_path').row(0, named=True)
    observed = row.get('reachable_pair_fraction')
    null = entry['null']
    if observed is None or 'reachable_pair_fraction' not in null.columns:
        return None
    values = null['reachable_pair_fraction']
    if not math.isfinite(observed) or values.null_count() or not values.is_finite().all() or not len(values):
        return None
    p = min(1., 2 * min((1 + (values <= observed).sum()) / (len(values) + 1),
                        (1 + (values >= observed).sum()) / (len(values) + 1)))
    return {'observed': observed, 'null_mean': values.mean(), 'p_value': p,
            'effect_observed_minus_null': observed - values.mean()}


def adjust_reachability(results):
    """Correct before any display-network selection, across each saved run."""
    families = {}
    for entry in results:
        if entry['metadata']['analysis'] != 'shortest_path':
            continue
        reach = saved_reachability(entry)
        if reach is None:
            continue
        meta = entry['metadata']
        family = json.dumps({key: meta.get(key) for key in (
            'protein_set', 'scope', 'seed', 'simulations', 'uniform', 'reviewed_only',
            'min_publications', 'networks', 'run_signature')}, sort_keys=True)
        entry['reachability'] = reach
        families.setdefault(family, []).append(entry)
    for entries in families.values():
        expected = {Path(path.replace('\\', '/')).stem for path in entries[0]['metadata'].get('networks', [])}
        complete = bool(expected) and expected <= {entry['network'] for entry in entries}
        adjusted = bh_adjust([entry['reachability']['p_value'] for entry in entries]) if complete else [None] * len(entries)
        for entry, q in zip(entries, adjusted):
            entry['reachability']['fdr_bh'] = q


def render_saved_monte_carlo(folder, protein_set, set_name, reviewed_only=False, min_publications=0, scope='network', selected_networks=None):
    st.header(f'Monte Carlo · {set_name}')
    st.info(
        f'How Monte Carlo works: compare the observed {set_name} protein set with many random '
        'protein sets of the same size from the selected comparison graph. In degree-matched runs, '
        'each random set preserves the number of proteins in each floor(log2(max(1, degree))) bin '
        '(degrees 2–3, 4–7, 8–15, etc.). This controls for broad differences in connectivity, '
        'rather than matching every degree exactly. Proteins are sampled without replacement within '
        'each trial and may appear again in other trials; observed-set proteins remain eligible in '
        'these saved runs.')
    st.caption('Choose an interactome from the saved runs for the sidebar protein set. '
               'These results load from disk; no simulation runs. The saved seed and sampling settings apply.')
    results, problems = [], []
    candidates = list(Path(folder).glob(f'{protein_set}_*.json'))
    archive = Path(folder) / 'archive' / f'monte_carlo_{scope}'
    candidates += [path for path in archive.glob(f'{protein_set}_*.json')
                   if not (Path(folder) / path.name).exists()
                   or json.loads((Path(folder) / path.name).read_text()).get('scope', 'network') != scope]
    for path in sorted(candidates):
        if not path.stem.endswith(('_shortest_path', '_cohesion')):
            continue
        try:
            siblings = [path, path.with_suffix('.parquet'), path.with_name(f'{path.stem}_summary.csv')]
            stamp = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in siblings)
            entry = _cached_result(str(path), stamp)
            meta = entry['metadata']
            if (meta.get('scope', 'network') == scope and meta['protein_set'] == protein_set and bool(meta.get('reviewed_only', False)) == reviewed_only
                    and int(meta.get('min_publications', 0)) == min_publications):
                results.append(entry)
        except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
            problems.append(f'{path.name}: {exc}')
    for problem in problems:
        st.warning(f'Unable to load saved result: {problem}')
    if selected_networks is not None:
        results = [entry for entry in results if entry['network'] in selected_networks]
    if not results:
        st.info('Monte Carlo results are being prepared for this protein set and population.')
        return
    networks = sorted({entry['network'] for entry in results})
    network = st.selectbox('Monte Carlo interactome', networks, key='saved_mc_network')
    mode = st.segmented_control('Plot format', ['Interactive', 'Static'], default='Interactive',
                                required=True, key='saved_mc_plot_format')
    output_folder = Path(folder).parent / 'plots' / 'monte_carlo'
    if mode == 'Static':
        st.caption('Static figures include legends and statistical annotations; methods are in the expanders. '
                   'PNG and SVG copies are saved in data/plots/monte_carlo/ and can be downloaded below each plot.')
    chosen = [entry for entry in results if entry['network'] == network]
    with st.expander('Metrics, null model and statistical interpretation'):
        st.markdown(
            '**Average shortest path length:** mean distance between distinct reachable protein-set pairs. '
            'Paths can use any interactome protein. Lower values mean closer reachable pairs; disconnected '
            'pairs are excluded; this mean describes within-component proximity.\n\n'
            '**Cohesion (internal edges)** counts direct within-set interactions. **Largest component** '
            'counts proteins in the largest connected part of the induced protein-set subgraph. '
            'Both retain present set proteins as isolates; larger values mean more direct grouping.\n\n'
            '**Effect = observed − null mean**, in the metric’s units. Shortest-path tests use twice the '
            'smaller inclusive empirical tail; cohesion uses the upper tail. Both apply the +1 correction. '
            'Asterisks and ns use unadjusted empirical p-values for each individual test. '
            'No multiple-testing correction is applied in this view. Undefined tests show n/a. '
            'Topology alone does not validate moonlighting or binding exclusivity.')
        st.caption(SIGNIFICANCE_LEGEND)
        for entry in chosen:
            meta = entry['metadata']
            st.markdown(f"**{meta['analysis']} saved run:** {meta['simulations']} simulations; seed {meta['seed']}; "
                        f"scope {meta['scope']}; reviewed only {meta.get('reviewed_only', False)}; "
                        f"minimum publications {meta.get('min_publications', 0)}.")
            st.caption(meta['null_model'])
            st.caption('Original requested networks: ' + ', '.join(meta['networks']))
        st.caption('Saved results describe the original source data and mappings; current protein-set membership '
                   'has not been revalidated against those historical inputs.')
    for metric, (title, _, analysis) in METRICS.items():
        entry = next((item for item in chosen if item['metadata']['analysis'] == analysis), None)
        if entry is None:
            st.info(f'No saved {title.lower()} result for this interactome.')
            continue
        row = entry['summary'].filter(pl.col('measure') == metric).row(0, named=True)
        render_distribution(entry['null'], row, network, set_name, entry['metadata'], mode,
                            f'saved_mc_{metric}', output_folder, f'{protein_set}_{network}_{metric}')
        render_calculations(results, entry, row)
        finite_count = entry['null'].filter(pl.col(metric).is_finite()).height
        with st.expander(f'Plot interpretation · {title}'):
            st.caption(f"Red line: observed; dashed line: null mean. Finite simulations: {finite_count}/"
                       f"{entry['null'].height}. Effect = observed − null mean. {row['significance']} uses unadjusted empirical p.")
            st.caption('Saved comparison background: ' + entry['metadata']['null_model'])
            st.caption('Two-sided doubled-smaller inclusive empirical tail for shortest path; '
                       'upper inclusive empirical tail for cohesion. Both use the +1 correction. '
                       + SIGNIFICANCE_LEGEND)
    st.subheader('Results table')
    summary = pl.concat([entry['summary'] for entry in chosen], how='diagonal_relaxed')
    display = summary.select(
        pl.col('measure').replace({k: v[0] for k, v in METRICS.items()}).alias('Metric'),
        pl.col('observed').alias('Observed'), pl.col('null_mean').alias('Null mean'),
        pl.col('effect_observed_minus_null').alias('Effect (observed − null)'),
        pl.col('p_value').alias('Empirical p'),
        pl.col('significance').alias('Significance'), pl.col('query_nodes').alias('Set proteins'),
        pl.col('valid_simulations').alias('Valid simulations'))
    st.dataframe(display, hide_index=True, width='stretch', alt=f'{set_name} Monte Carlo results for {network}')
    with st.expander('Results table interpretation'):
        st.caption(SIGNIFICANCE_LEGEND + ' Missing numeric values indicate unavailable statistics.')
    st.download_button('Download results table', summary.write_csv().encode(),
                       file_name=f'{protein_set}_{network}_monte_carlo_summary.csv', mime='text/csv', key='saved_mc_download')
