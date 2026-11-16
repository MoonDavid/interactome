"""Saved Monte Carlo exports retain full-family inference when browsed."""
import json
from pathlib import Path

import polars as pl
import pytest

from saved_monte_carlo import load_saved_result, distribution_figure, saved_reachability, empirical_calculation


def write_result(folder, q=0.02, p=0.001):
    prefix = folder / 'moonlitdb_HuRI_shortest_path'
    metadata = dict(protein_set='moonlitdb', network_path='/old/data/HuRI.parquet', analysis='shortest_path',
                    simulations=3, seed=42, scope='network', null_model='Degree matched, including query proteins.',
                    correction='BH across five requested networks.', networks=['HuRI', 'BioPlex'],
                    reviewed_only=False, min_publications=0)
    prefix.with_suffix('.json').write_text(json.dumps(metadata))
    pl.DataFrame({'simulation': [1, 2, 3], 'mean_shortest_path': [2., 3., 4.],
                  'reachable_pair_fraction': [1., 1., 1.]}).write_parquet(prefix.with_suffix('.parquet'))
    pl.DataFrame([dict(protein_set='moonlitdb', network='HuRI', measure='mean_shortest_path',
                       observed=1., null_mean=3., effect_observed_minus_null=-2., p_value=p,
                       fdr_bh=q, significance='wrong', query_nodes=10, valid_simulations=3,
                       reachable_pair_fraction=1.)]).write_csv(f'{prefix}_summary.csv')
    return prefix.with_suffix('.json')


@pytest.mark.parametrize('q', [0.02, 0.2, None])
def test_empirical_p_controls_plot_and_table(tmp_path, q):
    label = '**'
    result = load_saved_result(write_result(tmp_path, q=q, p=0.001))
    row = result['summary'].row(0, named=True)
    assert row['p_value'] == 0.001
    assert row['fdr_bh'] == q
    assert row['significance'] == label
    fig = distribution_figure(result['null'], row, 'HuRI', 'MoonLitDB')
    assert f'<b>{label}</b>' in fig.layout.annotations[0].text
    assert '<b>Observed 1</b>' in fig.layout.annotations[0].text
    assert '<b>Null average 3</b>' in fig.layout.annotations[0].text
    assert '<b>Effect -2</b>' in fig.layout.annotations[0].text
    assert [shape.x0 for shape in fig.layout.shapes] == [1., 3.]


def test_rejects_mismatched_summary(tmp_path):
    path = write_result(tmp_path)
    summary = path.with_name(f'{path.stem}_summary.csv')
    pl.read_csv(summary).with_columns(pl.lit('BioPlex').alias('network')).write_csv(summary)
    with pytest.raises(ValueError, match='different protein set or interactome'):
        load_saved_result(path)


def test_saved_panel_selection_and_filters(tmp_path):
    from streamlit.testing.v1 import AppTest
    write_result(tmp_path)
    script = f'''
import streamlit as st
from saved_monte_carlo import render_saved_monte_carlo
source = st.selectbox('Protein set', ['moonlitdb', 'moonprot'], key='source')
reviewed = st.checkbox('Reviewed only', key='reviewed')
render_saved_monte_carlo({str(tmp_path)!r}, source, source, reviewed)
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key='saved_mc_network').value == 'HuRI'
    assert len(app.get('plotly_chart')) == 1
    assert len(app.dataframe) == 1
    labels = [panel.label for panel in app.expander]
    assert 'Exact calculations · Average shortest path length' in labels
    assert 'Exact calculations · Reachable pairs' not in labels
    app.session_state['saved_mc_calculation_mean_shortest_path'] = True
    app.run()
    assert not app.exception
    assert not any(box.label == 'Calculation metric' for box in app.selectbox)
    assert any('Lower tail = (1 + 0) / (3 + 1)' in item.value for item in app.code)
    app.checkbox(key='reviewed').check().run()
    assert not app.exception
    assert len(app.dataframe) == 0
    assert any('results are being prepared' in element.value for element in app.info)
    app.selectbox(key='source').select('moonprot').run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0


def test_saved_reachability_uses_all_trials(tmp_path):
    entry = load_saved_result(write_result(tmp_path))
    entry['summary'] = entry['summary'].with_columns(pl.lit(0.).alias('reachable_pair_fraction'))
    entry['null'] = pl.DataFrame({'mean_shortest_path': [None, 2., 3.],
                                 'reachable_pair_fraction': [0., 1., 1.]})
    row = saved_reachability(entry)
    assert row['p_value'] == 1
    assert row['null_mean'] == pytest.approx(2 / 3)
    assert row['effect_observed_minus_null'] == pytest.approx(-2 / 3)


def test_saved_reachability_complete_family(tmp_path):
    from streamlit.testing.v1 import AppTest
    path = write_result(tmp_path)
    meta = json.loads(path.read_text())
    meta['networks'] = ['data/processed_networks/HuRI.parquet']
    path.write_text(json.dumps(meta))
    app = AppTest.from_string(f'''
from saved_monte_carlo import render_saved_monte_carlo
render_saved_monte_carlo({str(tmp_path)!r}, 'moonlitdb', 'MoonLitDB')
''').run(timeout=30)
    assert not app.exception
    table = app.dataframe[0].value
    assert table['Empirical p'].to_list() == [1., 1.]
    assert table['Significance'].to_list() == ['ns', 'ns']


def test_exact_counts_preserve_ties_and_invalid_trials():
    calc = empirical_calculation(pl.Series([1., 2., 2., 4.]), 2., True)
    assert (calc['lower_count'], calc['upper_count'], calc['ties']) == (3, 3, 2)
    assert calc['lower_tail'] == calc['upper_tail'] == 0.8
    assert calc['recalculated_p'] == 1.
    assert empirical_calculation(pl.Series([1., 2., 2., 4.]), 4.)['recalculated_p'] == 0.4
    assert empirical_calculation(pl.Series([1., None]), 2., True)['recalculated_p'] is None


def test_static_format_and_per_plot_calculations(tmp_path):
    from streamlit.testing.v1 import AppTest
    write_result(tmp_path)
    app = AppTest.from_string(f'''
from saved_monte_carlo import render_saved_monte_carlo
render_saved_monte_carlo({str(tmp_path)!r}, 'moonlitdb', 'MoonLitDB')
''').run(timeout=30)
    app.segmented_control(key='saved_mc_plot_format').set_value('Static').run(timeout=30)
    assert not app.exception
    assert len(app.get('plotly_chart')) == 0
    folder = tmp_path.parent / 'plots' / 'monte_carlo'
    assert (folder / 'moonlitdb_HuRI_mean_shortest_path.png').is_file()
    assert (folder / 'moonlitdb_HuRI_mean_shortest_path.svg').is_file()
    app.session_state['saved_mc_calculation_mean_shortest_path'] = True
    app.run(timeout=30)
    assert not app.exception
    assert any('Lower tail' in code.value for code in app.code)
