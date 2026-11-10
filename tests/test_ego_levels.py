"""Exact ego definitions, matched inference, and non-exclusive GO composition."""
import builtins
import io
import json
from pathlib import Path
import shutil
import zipfile

import networkx as nx
import numpy as np
import polars as pl
import pytest
from scipy.stats import hypergeom

from moonlighting import _decode_archive, bh_adjust, graph_from_path, result_archive
from moonlighting_extensions import (
    _go_level_table, ego_component_enrichment, ego_go_composition, ego_go_pie_table,
    ego_level_analysis, ego_level_comparison, ego_neighborhood, go_ontology_levels,
)
from ego_analysis_ui import category_color, ego_effect_figure, ego_go_pie_figure, ego_network_figure


def graph(tmp_path, edges):
    path = tmp_path / 'graph.parquet'
    pl.DataFrame({'a': [str(a) for a, _ in edges], 'b': [str(b) for _, b in edges]}).write_parquet(path)
    return graph_from_path(path)


def test_spokes_triangle_and_removed_isolates(tmp_path):
    g = graph(tmp_path, [('E', 'A'), ('E', 'B'), ('A', 'B')])
    spoke = ego_neighborhood(g, 'E', '1.0')
    induced = ego_neighborhood(g, 'E', '1.5')
    assert spoke['edges'].height == 2
    assert induced['edges'].height == 3
    assert induced['summary']['partner_density'] == 1
    assert induced['summary']['triangles'] == 1
    assert np.isnan(ego_neighborhood(g, 'E', '2.0')['summary']['partner_pair_reconnection'])
    assert 'partner_density' not in spoke['summary']
    figure = ego_network_figure(spoke, remove_ego=True)
    assert len(figure.data[1].x) == 2
    assert len(figure.data[0].x) == 0


def test_second_shell_reconnects_separate_partner_groups(tmp_path):
    g = graph(tmp_path, [('E', 'A'), ('E', 'B'), ('E', 'C'), ('E', 'D'),
                          ('A', 'B'), ('C', 'D'), ('A', 'X'), ('C', 'X')])
    first = ego_neighborhood(g, 'E')
    second = ego_neighborhood(g, 'E', '2.0', exact_centrality=True)
    assert first['nodes'].height == 5
    assert first['summary']['partner_components'] == 2
    assert first['summary']['fragmentation'] == .5
    assert second['nodes'].height == 6
    assert second['summary']['second_shell_size'] == 1
    assert second['summary']['components_without_ego'] == 1
    assert second['summary']['partner_pair_reconnection'] == 1
    assert second['nodes'].filter(pl.col('shell') == 2)['protein'].to_list() == ['X']
    assert second['summary']['global_efficiency'] > 0
    direct = nx.Graph()
    direct.add_nodes_from(second['nodes']['protein'])
    direct.add_edges_from(second['edges'].iter_rows())
    assert second['summary']['ego_betweenness'] == pytest.approx(nx.betweenness_centrality(direct)['E'])


def test_display_centrality_limits_and_empty_archives(tmp_path):
    g = graph(tmp_path, nx.star_graph(501).edges)
    snapshot = ego_neighborhood(g, '0', exact_centrality=True)
    assert snapshot['summary']['ego_betweenness'] is None
    with pytest.raises(ValueError, match='Diagram unavailable'):
        ego_network_figure(snapshot)
    with pytest.raises(ValueError, match='2,000'):
        ego_neighborhood(g, '0', max_nodes=2001)
    result = ego_level_analysis(g, set())
    assert result['metrics'].height == 0
    decoded = _decode_archive(result_archive(result, {'analysis': 'empty'}))['result']
    assert decoded['memberships'].height == 0
    assert decoded['matches'].height == 0


def test_budgets_never_return_partial_graph_and_keep_native_scores(tmp_path):
    g = graph(tmp_path, [('E', 'A'), ('E', 'B'), ('A', 'X')])
    skipped = ego_neighborhood(g, 'E', '2.0', max_nodes=3)
    assert skipped['summary']['status'] == 'skipped'
    assert skipped['nodes'].is_empty() and skipped['edges'].is_empty()
    assert ego_neighborhood(g, 'E', max_edges=1)['summary']['status'] == 'skipped'
    path = tmp_path / 'scored.parquet'
    pl.DataFrame({'a': ['A', 'B'], 'b': ['E', 'E'], 'pInt': [.95, None]}).write_parquet(path)
    snapshot = ego_neighborhood(graph_from_path(path), 'E', source_path=path)
    assert snapshot['edges']['pInt'].drop_nulls().to_list() == [.95]
    assert snapshot['edges']['pInt'].null_count() == 1
    with pytest.raises(ValueError):
        ego_neighborhood(g, 'not present')
    with pytest.raises(ValueError):
        ego_neighborhood(g, 'E', '3.0')
    g['adj'] = g['adj'].copy()
    # Guard uses the actual sparse adjacency size, not an editable edge_count field.
    import moonlighting_extensions as extensions
    old = extensions.ADVANCED_EDGE_LIMIT
    try:
        extensions.ADVANCED_EDGE_LIMIT = 2
        with pytest.raises(ValueError, match='500,000'):
            ego_neighborhood(g, 'E')
    finally:
        extensions.ADVANCED_EDGE_LIMIT = old


def test_all_query_egos_matching_shortages_and_global_bh(tmp_path):
    g = graph(tmp_path, nx.cycle_graph(4).edges)
    result = ego_level_analysis(g, {'0', '1', '2'}, seed=7, set_name='MoonLitDB')
    again = ego_level_analysis(g, {'0', '1', '2'}, seed=7, set_name='MoonLitDB')
    assert result['notes']['query_proteins'] == 3
    assert result['notes']['matched_pairs'] == 1
    assert result['notes']['unmatched_query_proteins'] == 2
    assert result['matches'].equals(again['matches'])
    query = result['metrics'].filter((pl.col('group') == 'MoonLitDB') & (pl.col('level') == '1.5'))
    assert query.height == 3
    assert query['pair_id'].null_count() == 2
    assert query['protein'].n_unique() == 3
    comparison = ego_level_comparison({'one': result, 'two': result})
    effects = ego_effect_figure(comparison['tests'], 'MoonLitDB')
    assert list(effects.data[1].x) == ['one', 'two']
    assert len(effects.data[1].y) == comparison['tests']['metric'].n_unique()
    assert not comparison['tests']['metric'].str.starts_with('1.0').any()
    assert (comparison['tests']['query_n'] == comparison['tests']['background_n']).all()
    np.testing.assert_allclose(comparison['tests']['fdr_bh'], bh_adjust(comparison['tests']['p_value']), equal_nan=True)
    # A missing control value removes both members from this metric's comparison only.
    result['metrics'] = result['metrics'].with_columns(
        pl.when((pl.col('group') == 'Background') & (pl.col('level') == '1.5'))
        .then(None).otherwise(pl.col('partner_density')).alias('partner_density'))
    tests = ego_level_comparison({'one': result})['tests']
    missing = tests.filter(pl.col('metric') == '1.5 · partner_density').row(0, named=True)
    assert missing['query_n'] == missing['background_n'] == 0
    assert missing['significance'] == 'n/a'
    restored = _decode_archive(result_archive(result, {'analysis': 'ego'}))['result']
    assert ego_level_comparison({'restored': restored})['tests'].height == tests.height


@pytest.fixture
def go_data(tmp_path):
    pytest.importorskip('goatools')
    obo = tmp_path / 'go.obo'
    terms = [
        ('GO:0008150', 'biological process', []), ('GO:0000001', 'Category A', ['GO:0008150']),
        ('GO:0000002', 'Category B', ['GO:0008150']),
        ('GO:0000003', 'Shared child', ['GO:0000001', 'GO:0000002']),
        ('GO:0000004', 'Background function', ['GO:0008150', 'GO:0000001']),
    ]
    obo.write_text('format-version: 1.2\ndata-version: test-fixture\n\n' + '\n\n'.join(
        '[Term]\nid: ' + term + '\nname: ' + name + '\nnamespace: biological_process\n'
        + '\n'.join('is_a: ' + p for p in parents) for term, name, parents in terms) + '\n', encoding='utf-8')
    levels, metadata = go_ontology_levels(obo)
    rows = []
    for protein, ids in [('P1', [0, 1, 2, 3]), ('P2', [0, 1, 2, 3]), ('P3', [0, 1]), ('P4', [0]), ('Q', [0, 4])]:
        for index in ids:
            term, name, _ = terms[index]
            rows.append({'protein': protein, 'term': term, 'name': name, 'aspect': 'P',
                         'evidence': 'IEA' if protein == 'P3' else 'IEA|EXP'})
    annotation = tmp_path / 'annotations.parquet'
    pl.from_dicts(rows + [rows[1]]).write_parquet(annotation)  # Duplicate annotations do not add votes.
    return annotation, levels, metadata


def test_go_fractional_mass_majority_and_missing_categories(go_data):
    annotation, levels, metadata = go_data
    assert levels.filter(pl.col('term') == 'GO:0000004')['go_level'].item() == 1  # shortest path
    result = ego_go_composition(['P1', 'P2', 'P3', 'P4', 'P5'], annotation, levels, go_level=1)
    assert result['summary']['group_size'] == 5
    assert result['summary']['annotated_proteins'] == 4
    assert result['summary']['mapped_proteins'] == 3
    assert result['table']['weight'].sum() == pytest.approx(5)
    assert result['table']['fraction'].sum() == pytest.approx(1)
    assert set(result['table']['term']) == {'GO:0000001', 'GO:0000002', '__no_level__', '__unannotated__'}
    assert result['majority']['term'].to_list() == ['GO:0000001']
    multiple = ego_go_composition(['P1', 'P2', 'P5'], annotation, levels, go_level=1)
    assert set(multiple['majority']['term']) == {'GO:0000001', 'GO:0000002'}
    tied = ego_go_composition(['P1', 'P2', 'P3', 'P4'], annotation, levels, go_level=1)
    assert 'GO:0000002' not in tied['majority']['term'].to_list()
    filtered = ego_go_composition(['P3'], annotation, levels, go_level=1, exclude_iea=True)
    assert filtered['summary']['annotated_proteins'] == 0
    combined = ego_go_composition(['P1|P2', 'P3'], annotation, levels, go_level=1)
    assert combined['table']['weight'].sum() == pytest.approx(2)
    assert combined['summary']['mapped_proteins'] == 2
    pie = ego_go_pie_table(result, top=1)
    assert pie['weight'].sum() == pytest.approx(5)
    assert '__other__' in pie['term'].to_list()
    figure = ego_go_pie_figure(result, 'GO composition')
    assert sum(figure.data[0].values) == pytest.approx(5)
    assert category_color('GO:0000001') == category_color('GO:0000001')
    empty = ego_go_composition([], annotation, levels)
    assert empty['table'].is_empty() and empty['summary']['group_size'] == 0


def test_enrichment_family_includes_zero_hits_and_display_filter_is_independent(tmp_path, go_data):
    annotation, levels, metadata = go_data
    g = graph(tmp_path, [('E', 'P1'), ('E', 'P2'), ('E', 'P3'), ('E', 'P4'), ('P4', 'Q')])
    snapshot = ego_neighborhood(g, 'E')
    result = ego_component_enrichment(snapshot, g, annotation, ['C1', 'C2'], levels, min_size=1)
    tests = result['table']
    np.testing.assert_allclose(tests['qvalue'], bh_adjust(tests['pvalue']), equal_nan=True)
    assert tests.filter(pl.col('query_hits') == 0).height > 0
    assert (tests.filter(pl.col('query_hits') == 0)['pvalue'] == 1).all()
    hit = tests.filter((pl.col('component') == 'C1') & (pl.col('term') == 'GO:0000001')).row(0, named=True)
    assert hit['pvalue'] == pytest.approx(hypergeom.sf(hit['query_hits'] - 1, len(g['nodes']), hit['background_hits'], 1))
    one = tests.filter(pl.col('go_level') == 1)
    two = tests.filter(pl.col('go_level') == 2)
    assert one.height and two.height
    # The saved family includes every level, independent of which view is displayed.
    archive = result_archive(result, {'analysis': 'go'})
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert 'result_table.csv' in z.namelist()
    empty = ego_component_enrichment(snapshot, g, annotation, [], levels)
    assert _decode_archive(result_archive(empty, {'analysis': 'empty_go'}))['result']['table'].height == 0


def test_missing_optional_go_dependency_does_not_affect_topology(tmp_path, monkeypatch):
    import_module = builtins.__import__
    def without_go(name, *args, **kwargs):
        if name == 'goatools' or name.startswith('goatools.'):
            raise ImportError('not installed')
        return import_module(name, *args, **kwargs)
    _go_level_table.cache_clear()
    monkeypatch.setattr(builtins, '__import__', without_go)
    obo = tmp_path / 'empty.obo'
    obo.write_text('format-version: 1.2\n')
    with pytest.raises(ImportError, match='optional GO extra'):
        go_ontology_levels(obo)
    g = graph(tmp_path, [('E', 'A')])
    assert ego_neighborhood(g, 'E')['summary']['status'] == 'ok'


def test_ego_page_lazy_go_controls_and_cache_invalidation(tmp_path, go_data):
    from streamlit.testing.v1 import AppTest
    from moonlighting import SOURCES

    annotation, levels, metadata = go_data
    data = tmp_path / 'data'
    for folder in ('processed_networks', 'mappings', 'annotations', 'GO'):
        (data / folder).mkdir(parents=True, exist_ok=True)
    pl.DataFrame({'a': ['E', 'E', 'E', 'E', 'P4'], 'b': ['P1', 'P2', 'P3', 'P4', 'Q']}).write_parquet(
        data / 'processed_networks/HuRI.parquet')
    pl.from_dicts([{'uniprot': p, 'Reviewed': 'reviewed', 'Gene Names': p,
                    'moonlitdb_number_of_publications': '2',
                    **{s: s == 'moonlitdb' for s in SOURCES}} for p in ('P1', 'P2')]).write_csv(
        data / 'allmoon_with_moonlitdb_human.csv')
    (data / 'mappings/moonlighting_uniprot.json').write_text(json.dumps({'mapping': {'P1': ['P1'], 'P2': ['P2']}}))
    shutil.copyfile(annotation, data / 'annotations/human_go_propagated.parquet')
    (data / 'annotations/human_go_propagated.json').write_text('{}')
    shutil.copyfile(tmp_path / 'go.obo', data / 'GO/go-basic.obo')
    page = Path(__file__).resolve().parents[1] / 'pages/2_Topological_analysis.py'
    code = page.read_text(encoding='utf-8').replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(tmp_path)!r})')
    code = code.replace('def annotation_file():\n', "def annotation_file():\n    st.session_state['ego_test_go_calls'] = st.session_state.get('ego_test_go_calls', 0) + 1\n")
    app = AppTest.from_string(code, default_timeout=45)
    app.session_state['topo_tab'] = 'Local wiring'
    app.run()
    assert not app.exception
    assert app.segmented_control(key='wiring_method').value == 'Ego levels'
    assert 'ego_test_go_calls' not in app.session_state
    assert app.session_state['topo_results']['ego_levels|HuRI']['result']['notes']['query_proteins'] == 2
    app.segmented_control(key='ego_workflow').set_value('Individual ego explorer').run()
    app.text_input(key='ego_accession').set_value('E').run()
    assert not app.exception
    assert app.segmented_control(key='ego_level').value == '1.5'
    assert 'ego_test_go_calls' not in app.session_state
    app.segmented_control(key='ego_level').set_value('2.0').run()
    assert not app.exception
    assert app.session_state['topo_results']['ego_explorer|HuRI']['result']['nodes'].height == 6
    app.toggle(key='ego_go_enabled').set_value(True).run()
    assert not app.exception
    app.slider(key='ego_go_level').set_value(1).run()
    app.toggle(key='ego_go_enrich').set_value(True).run()
    app.number_input(key='ego_go_min_size').set_value(1).run()
    assert not app.exception
    stored = app.session_state['topo_results']['ego_component_enrichment|HuRI']['result']['table']
    original_q = stored.select('component', 'term', 'qvalue').sort('component', 'term')
    count = len(list((data / 'analysis_runs').glob('ego_component_enrichment_*.zip')))
    app.slider(key='ego_go_level').set_value(2).run()
    app.selectbox(key='ego_go_scope').set_value('Second shell').run()
    assert not app.exception
    assert len(list((data / 'analysis_runs').glob('ego_component_enrichment_*.zip'))) == count
    assert original_q.equals(app.session_state['topo_results']['ego_component_enrichment|HuRI']['result']['table']
                             .select('component', 'term', 'qvalue').sort('component', 'term'))
    app.number_input(key='ego_max_nodes').set_value(3).run()
    assert not app.exception
    assert app.session_state['topo_results']['ego_explorer|HuRI']['result']['summary']['status'] == 'skipped'
    assert len(app.get('plotly_chart')) == 0
