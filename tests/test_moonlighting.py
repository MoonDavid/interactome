"""Tests for moonlighting data, topology, diffusion, GO enrichment, comparisons, plots, and page."""
import io
import json
import math
from pathlib import Path
import zipfile

import networkx as nx
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest
from scipy import sparse
from scipy.stats import hypergeom, mannwhitneyu
from streamlit.testing.v1 import AppTest

import moonlighting
from moonlighting import (
    GROUP_COLORS, SOURCES, bh_adjust, centrality_analysis, centrality_heatmap,
    cohort_ids, cohesion_analysis, community_analysis, compare_source_sets,
    cross_network_analysis, go_enrichment, graph_from_path,
    hub_enrichment_heatmap, load_moon_data, mapping_coverage, metric_figure,
    prepare_go_annotations, random_walk_analysis, result_archive,
    robustness_analysis, significance_label, wiring_analysis,
)


# ---------------------------------------------------------------------------
# Cohort Data, Graph Context & Reproducible Exports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cached_aliases', [False, True])
def test_cached_overview_coverage_uses_selected_set_columns(cached_aliases):
    import plotly.express as px
    from moonlighting import protein_set_network_summary

    original = pl.DataFrame({
        'Network': ['HuRI'], 'Nodes': [10], 'Edges': [12],
        'Cohort proteins present': [3], 'Cohort network nodes': [3], 'Cohort coverage (%)': [75.0],
    })
    if cached_aliases:
        original = original.with_columns(pl.lit(3).alias('MoonLitDB proteins present'),
                                         pl.lit(75.0).alias('MoonLitDB coverage (%)'))
    displayed = protein_set_network_summary(original, 'MoonLitDB')
    figure = px.bar(displayed, x='Network', y='MoonLitDB coverage (%)', text='MoonLitDB proteins present')
    assert list(figure.data[0].y) == [75.0]
    assert list(figure.data[0].text) == [3.0]
    assert 'MoonLitDB network nodes' in displayed.columns
    assert not any('Cohort' in column for column in displayed.columns)
    assert 'Cohort coverage (%)' in original.columns


def test_mapping_filters_and_full_network_background(tmp_path):
    (tmp_path / 'mappings').mkdir()
    records = []
    for accession, flags in [('OLD', {'moonprot': True}), ('Q2', {'moonlitdb': True}), ('GONE', {})]:
        records.append(dict(uniprot=accession, Reviewed='reviewed', **{'Gene Names': accession},
                            moonlitdb_number_of_publications=1, **{s: flags.get(s, False) for s in SOURCES}))
    pl.DataFrame(records).write_csv(tmp_path / 'allmoon_with_moonlitdb_human.csv')
    (tmp_path / 'mappings/moonlighting_uniprot.json').write_text(json.dumps({'mapping': {'OLD': ['P1'], 'Q2': ['Q2'], 'GONE': []}}))
    data = load_moon_data(tmp_path)
    ids, cohort = cohort_ids(data, ['moonprot', 'moonlitdb'])
    assert ids == {'P1', 'Q2'}
    assert not cohort_ids(data, ['moonprot', 'moonlitdb'], require_all=True)[0]
    path = tmp_path / 'graph.parquet'
    pl.DataFrame({'a': ['P1', 'Q2'], 'b': ['Q2', 'Z'], 'probability': [.2, .9]}).write_parquet(path)
    graph = graph_from_path(path)
    assert graph['edge_count'] == 2
    assert set(graph['nodes']) == {'P1', 'Q2', 'Z'}
    assert graph['weights'][graph['index']['P1'], graph['index']['Q2']] == .2
    coverage = mapping_coverage(graph, data, ids)
    assert coverage['summary']['network_nodes'] == 3
    assert coverage['summary']['comparison_nodes'] == 1
    assert coverage['audit'].filter(pl.col('input_accession') == 'GONE')['mapping_status'][0] == 'unmapped'


def test_export_tables_and_nonfinite_diagnostics():
    result = {'table': pl.DataFrame({'a': [1, 2]}), 'summary': {'missing': float('nan'), 'value': np.float64(float('inf'))}}
    content = result_archive(result, {'analysis': 'fixture'})
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['results']['summary'] == {'missing': None, 'value': None}
        assert manifest['results']['table']['rows'] == 2
        assert archive.read('result_table.csv').decode() == 'a\n1\n2\n'


def test_multilabel_cohort_and_go_count_graph_node_once(tmp_path):
    path = tmp_path / 'graph.parquet'
    pl.DataFrame({'a': ['P1|P2', 'B'], 'b': ['B', 'C']}).write_parquet(path)
    graph = graph_from_path(path)
    assert moonlighting._selection(graph, {'P1', 'P2'}).sum() == 1
    assert len(moonlighting._cohort_indices(graph, {'P1', 'P2'})) == 1
    result = centrality_analysis(graph, {'P2'}, use_cache=False)
    assert result['metrics']['is_moonlighting'].sum() == 1
    data = pl.DataFrame({'input_accession': ['P1', 'P2'], 'canonical_id': ['P1', 'P2'],
                         'mapping_status': ['unchanged'] * 2, 'Gene Names': ['x', 'y'], 'Reviewed': ['reviewed'] * 2})
    summary = mapping_coverage(graph, data, {'P1', 'P2'})['summary']
    assert summary['cohort_in_network'] == 2
    assert summary['cohort_network_nodes'] == 1
    assert summary['comparison_nodes'] == 2
    annotation = tmp_path / 'annotations.parquet'
    pl.DataFrame({'protein': ['P1', 'P2', 'B'], 'term': ['T1'] * 3,
                  'name': ['one'] * 3, 'aspect': ['P'] * 3, 'evidence': ['EXP'] * 3}).write_parquet(annotation)
    enriched = go_enrichment(annotation, graph['nodes'], {'P1', 'P2'}, min_size=1)
    assert enriched['summary']['query_size'] == 1
    assert enriched['table']['background_hits'][0] == 2
    assert enriched['table']['query_hits'][0] == 1


def test_polars_mapping_expansion_and_publication_filter(tmp_path):
    (tmp_path / 'mappings').mkdir()
    records = []
    for accession, publications in [(' OLD-2 ', '2'), ('Q2', 'NaN'), ('GONE', 'invalid')]:
        records.append(dict(uniprot=accession, Reviewed='reviewed', **{'Gene Names': accession},
                            moonlitdb_number_of_publications=publications,
                            **{source: 'yes' if source == 'moonprot' else '' for source in SOURCES}))
    pl.DataFrame(records).write_csv(tmp_path / 'allmoon_with_moonlitdb_human.csv')
    (tmp_path / 'mappings/moonlighting_uniprot.json').write_text(json.dumps(
        {'mapping': {'OLD': ['P1-2', 'P2'], 'Q2': ['Q2']}}))
    data = load_moon_data(tmp_path)
    assert isinstance(data, pl.DataFrame)
    assert data['canonical_id'].to_list() == ['P1', 'P2', 'Q2', '']
    ids, selected = cohort_ids(data, ['moonprot'], reviewed_only=True, min_publications=1)
    assert ids == {'P1', 'P2'}
    assert selected.height == 2


def test_empty_mapping_and_cross_network_schema(tmp_path):
    pl.DataFrame({'uniprot': ['MISSING'], 'Reviewed': [''], 'Gene Names': [''],
                  'moonlitdb_number_of_publications': [''],
                  **{source: [False] for source in SOURCES}}).write_csv(tmp_path / 'allmoon_with_moonlitdb_human.csv')
    data = load_moon_data(tmp_path)
    assert data['mapping_status'].to_list() == ['unmapped']
    assert cohort_ids(data)[0] == set()
    graph = {'nodes': np.array(['A']), 'degree': np.array([0])}
    result = cross_network_analysis(graph, graph, set())
    assert result['table'].is_empty()
    assert result['table'].schema['degree_A'] == pl.Int64
    assert result['summary']['spearman_degree_rho'] is None


# ---------------------------------------------------------------------------
# Centralities, Cohesion, Local Wiring & Communities
# ---------------------------------------------------------------------------

def graph_context(network):
    nodes = np.array([f'P{x}' for x in network.nodes])
    adjacency = nx.to_scipy_sparse_array(network, nodelist=list(network.nodes), format='csr', dtype=float)
    return {'nodes': nodes, 'adj': adjacency, 'weights': adjacency,
            'degree': np.asarray(adjacency.sum(axis=1)).ravel(),
            'index': {node: i for i, node in enumerate(nodes)}}


def test_centrality_matches_networkx_and_complement():
    network = nx.path_graph(5)
    graph = graph_context(network)
    result = centrality_analysis(graph, {'P1', 'P2'})
    metrics = result['metrics']
    expected = nx.pagerank(network, tol=1e-12, max_iter=1000)
    assert np.allclose(metrics['pagerank'], list(expected.values()), atol=1e-9)
    assert metrics['degree'].to_list() == [1, 2, 2, 2, 1]
    assert result['notes']['pagerank']['converged']
    assert (result['tests']['query_n'] == 2).all()
    assert (result['tests']['background_n'] == 3).all()
    assert np.allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])
    with pytest.raises(ValueError, match='No selected'):
        centrality_analysis(graph, {'MISSING'})


def test_degree_bin_null_and_reproducibility():
    network = nx.disjoint_union(nx.complete_graph(3), nx.path_graph(2))
    graph = graph_context(network)
    result = cohesion_analysis(graph, {'P0', 'P1', 'P2'}, permutations=9, seed=3)
    assert result['summary']['internal_edges'] == 3
    assert result['summary']['largest_component'] == 3
    assert result['summary']['boundary_edges'] == 0
    assert result['summary']['conductance'] == 0
    assert result['summary']['internal_edges_p_value'] == 1
    assert result['null']['internal_edges'].to_list() == [3] * 9
    assert result['null'].equals(cohesion_analysis(graph, {'P0', 'P1', 'P2'}, 9, 3)['null'])


def test_ego_fragmentation_cycle_and_community_accounting():
    graph = graph_context(nx.cycle_graph(6))
    result = wiring_analysis(graph, {'P0', 'P2'}, max_nodes=2)
    assert result['metrics']['ego_components'].to_list() == [2] * 4
    assert result['metrics']['open_wedges'].to_list() == [1] * 4
    assert result['metrics']['effective_size'].to_list() == [2.] * 4
    communities = community_analysis(graph, {'P0', 'P2'})
    assert communities['enrichment']['network_proteins'].sum() == 6
    assert communities['enrichment']['moonlighting_proteins'].sum() == 2
    assert communities['metrics']['participation'].is_between(0, 1).all()


def test_extra_centralities_return_polars_and_match_networkx():
    network = nx.cycle_graph(6)
    result = centrality_analysis(graph_context(network), {'P0'}, extra=True, landmarks=6)
    metrics = result['metrics']
    assert isinstance(metrics, pl.DataFrame)
    assert isinstance(result['tests'], pl.DataFrame)
    assert metrics['core_number'].to_list() == [2] * 6
    assert np.allclose(metrics['clustering'], list(nx.clustering(network).values()))
    assert np.allclose(metrics['betweenness_approx'], list(nx.betweenness_centrality(network).values()))
    assert np.allclose(metrics['bridging_centrality'], metrics['betweenness_approx'] * metrics['bridging_coefficient'])


# ---------------------------------------------------------------------------
# Random Walk with Restart & Robustness
# ---------------------------------------------------------------------------

def edge_graph(edges, nodes=None):
    if nodes is None:
        nodes = sorted({node for edge in edges for node in edge[:2]})
    nodes = np.array(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    rows, cols, values = [], [], []
    for edge in edges:
        a, b = edge[:2]
        value = edge[2] if len(edge) > 2 else 1.
        rows.extend([index[a], index[b]])
        cols.extend([index[b], index[a]])
        values.extend([value, value])
    weights = sparse.csr_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes)))
    adj = weights.copy()
    adj.data = np.ones_like(adj.data)
    return {'nodes': nodes, 'index': index, 'weights': weights, 'adj': adj,
            'degree': np.asarray(adj.sum(axis=1)).ravel()}


def test_random_walk_mass_baseline_and_linker_hypergeometric():
    g = edge_graph([('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'E')])
    result = random_walk_analysis(g, {'A', 'C'}, restart=.3)
    ranked = {row['protein']: row for row in result['ranked'].iter_rows(named=True)}
    assert sum(row['score'] for row in ranked.values()) == pytest.approx(1)
    assert sum(row['baseline_score'] for row in ranked.values()) == pytest.approx(1)
    assert result['summary']['score_convergence']['converged']
    assert result['summary']['baseline_convergence']['converged']
    assert ranked['B']['seed_neighbors'] == 2
    assert ranked['B']['linker_pvalue'] == pytest.approx(hypergeom.sf(1, 4, 2, 3))
    assert np.isnan(ranked['A']['linker_pvalue'])
    assert ranked['E']['linker_pvalue'] == 1
    assert ranked['B']['linker_qvalue'] >= ranked['B']['linker_pvalue']
    assert np.allclose(result['trace']['seed_mass'] + result['trace']['nonseed_mass'], 1.)
    assert result['trace']['seed_mass'][0] == 1


def test_weight_affinities_and_dangling_mass_return():
    g = edge_graph([('A', 'B', .9), ('A', 'C', .1)])
    binary = {row['protein']: row for row in random_walk_analysis(g, {'A'}, weighted=False)['ranked'].iter_rows(named=True)}
    weighted = {row['protein']: row for row in random_walk_analysis(g, {'A'}, weighted=True)['ranked'].iter_rows(named=True)}
    assert binary['B']['score'] == pytest.approx(binary['C']['score'])
    assert weighted['B']['score'] > weighted['C']['score']
    isolate = edge_graph([('A', 'B')], nodes=['A', 'B', 'Z'])
    result = random_walk_analysis(isolate, {'Z'})
    assert result['ranked'].filter(pl.col('protein') == 'Z')['score'][0] == pytest.approx(1)
    assert np.allclose(result['trace']['seed_mass'], 1)


def test_degree_matched_null_reproducible_and_plus_one_corrected():
    g = edge_graph([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'A')])
    first = random_walk_analysis(g, {'A'}, permutations=5, seed=7)
    second = random_walk_analysis(g, {'A'}, permutations=5, seed=7)
    assert_frame_equal(first['ranked'], second['ranked'])
    p = first['ranked'].filter(~pl.col('is_seed'))['empirical_pvalue']
    assert p.is_between(1 / 6, 1).all()
    assert np.allclose(p * 6, np.round(p * 6))


def test_removal_lcc_and_degenerate_full_cohort():
    g = edge_graph([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
    result = robustness_analysis(g, {'C'}, permutations=5, seed=2)
    assert result['summary']['original_lcc_nodes'] == 5
    assert result['summary']['remaining_lcc_nodes'] == 2
    assert result['summary']['remaining_lcc_fraction_original'] == .4
    assert result['summary']['remaining_lcc_fraction_survivors'] == .5
    assert 1 / 6 <= result['summary']['empirical_pvalue_lower_tail'] <= 1
    all_removed = robustness_analysis(g, set(g['nodes']), permutations=0)
    assert all_removed['summary']['remaining_lcc_nodes'] == 0
    assert all_removed['summary']['empirical_pvalue_lower_tail'] is None
    assert all_removed['null'].is_empty()


def test_invalid_walk_inputs_explicit():
    g = edge_graph([('A', 'B')])
    with pytest.raises(ValueError, match='No selected cohort'):
        random_walk_analysis(g, {'X'})
    with pytest.raises(ValueError, match='Restart'):
        random_walk_analysis(g, {'A'}, restart=0)
    with pytest.raises(ValueError, match='between 0 and 50'):
        random_walk_analysis(g, {'A'}, permutations=51)


# ---------------------------------------------------------------------------
# GO Enrichment
# ---------------------------------------------------------------------------

def gaf_row(protein, term, evidence='EXP', qualifier='', database='UniProtKB', taxon='taxon:9606'):
    return '\t'.join([database, protein, 'symbol', qualifier, term, 'PMID:1', evidence, '', 'P', 'name', '', 'protein', taxon, '20250101', 'test', '', '']) + '\n'


def test_gaf_propagation_not_evidence_and_provenance(tmp_path):
    obo = tmp_path / 'go.obo'
    obo.write_text('data-version: fixture\n\n[Term]\nid: GO:1\nname: root\nnamespace: biological_process\n\n[Term]\nid: GO:2\nname: child\nnamespace: biological_process\nis_a: GO:1 ! root\n\n[Term]\nid: GO:3\nname: part\nnamespace: biological_process\nrelationship: part_of GO:2 ! child\n\n[Term]\nid: GO:4\nname: regulation\nnamespace: biological_process\nrelationship: regulates GO:1 ! root\n')
    gaf = tmp_path / 'go.gaf'
    gaf.write_text('!gaf-version: 2.2\n!date-generated: fixture\n' + gaf_row('P1-2', 'GO:3') + gaf_row('P2', 'GO:2', 'IEA') + gaf_row('P3', 'GO:2', qualifier='NOT|enables') + gaf_row('P4', 'GO:1', taxon='taxon:10090') + gaf_row('P5', 'GO:4'))
    path = prepare_go_annotations(tmp_path / 'out', gaf, obo)
    data = pl.read_parquet(path)
    assert set(data.filter(pl.col('protein') == 'P1')['term']) == {'GO:1', 'GO:2', 'GO:3'}
    assert set(data.filter(pl.col('protein') == 'P5')['term']) == {'GO:4'}
    assert not data['protein'].is_in(['P3', 'P4']).any()
    meta = json.loads(path.with_suffix('.json').read_text())
    assert meta['audit']['not_rows_excluded'] == 1
    assert meta['audit']['nonhuman_or_nonuniprot_rows_excluded'] == 1
    assert 'data-version: fixture' in meta['ontology_headers']
    assert prepare_go_annotations(tmp_path / 'out', gaf, obo) == path
    out = go_enrichment(path, {'P1', 'P2', 'P5', 'UNANNOTATED'}, {'P1'}, min_size=1, exclude_iea=True)
    assert out['summary']['population_size'] == 4
    assert out['summary']['annotated_population'] == 2
    assert out['summary']['tested_terms'] == 4


def test_enrichment_full_universe_and_bh_zero_hit_terms(tmp_path):
    path = tmp_path / 'annotations.parquet'
    pl.DataFrame({'protein': ['P1', 'P2', 'P3', 'P4'], 'term': ['T1', 'T1', 'T2', 'T3'], 'name': ['one', 'one', 'two', 'three'], 'aspect': ['P'] * 4, 'evidence': ['EXP'] * 4}).write_parquet(path)
    result = go_enrichment(path, {'P1', 'P2', 'P3', 'P4', 'P5', 'P6'}, {'P1', 'P2', 'OUTSIDE'}, min_size=1)
    table = {row['term']: row for row in result['table'].iter_rows(named=True)}
    assert result['summary']['query_size'] == 2
    assert result['summary']['annotated_population'] == 4
    assert table['T1']['pvalue'] == pytest.approx(hypergeom.sf(1, 6, 2, 2))
    assert table['T1']['qvalue'] == pytest.approx(3 * hypergeom.sf(1, 6, 2, 2))
    assert table['T1']['fold'] == 3
    assert table['T2']['query_hits'] == 0
    assert table['T2']['pvalue'] == 1
    with pytest.raises(ValueError):
        go_enrichment(path, {'P1'}, {'NOT_PRESENT'})


def test_empty_enrichment_preserves_typed_table(tmp_path):
    path = tmp_path / 'annotations.parquet'
    pl.DataFrame({'protein': ['P1'], 'term': ['T1'], 'name': ['one'], 'aspect': ['P'], 'evidence': ['EXP']}).write_parquet(path)
    result = go_enrichment(path, {'P1', 'P2'}, {'P1'}, min_size=2)
    assert result['table'].is_empty()
    assert result['table'].schema['qvalue'] == pl.Float64
    assert result['table'].schema['term'] == pl.String
    assert result['summary']['tested_terms'] == 0


# ---------------------------------------------------------------------------
# Multi-Set Source Comparisons
# ---------------------------------------------------------------------------

def comparison_fixture():
    network = nx.path_graph(6)
    nodes = np.array([f'P{i}' for i in network.nodes])
    adjacency = nx.to_scipy_sparse_array(network, format='csr', dtype=float)
    graph = {'nodes': nodes, 'adj': adjacency, 'weights': adjacency,
             'degree': np.asarray(adjacency.sum(axis=1)).ravel(),
             'index': {node: i for i, node in enumerate(nodes)}}
    data = pl.DataFrame({'canonical_id': ['P0', 'P1', 'P2', 'MISSING'],
                         'alpha': [True, True, False, False],
                         'beta': [False, True, True, True],
                         'empty': [False] * 4,
                         'Reviewed': ['reviewed'] * 4,
                         'moonlitdb_number_of_publications': ['2', '1', '0', '3']})
    return graph, data


def test_overlap_shared_background_and_global_bh(monkeypatch):
    graph, data = comparison_fixture()
    original = moonlighting.centrality_analysis
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(moonlighting, 'centrality_analysis', counted)
    result = compare_source_sets(graph, data, ['alpha', 'beta'])
    assert len(calls) == 1
    frame = result['metrics']
    assert frame.filter(pl.col('protein') == 'P1')['group'].to_list() == ['alpha', 'beta']
    assert frame.filter(pl.col('group') == 'Background')['protein'].to_list() == ['P3', 'P4', 'P5']
    assert result['coverage'].to_dicts()[1] == {'source': 'beta', 'selected_ids': 3, 'network_ids': 2, 'missing_ids': 1}
    assert result['overlap'].row(0, named=True)['network_overlap'] == 1
    tests = result['tests']
    degree = tests.filter((pl.col('group') == 'alpha') & (pl.col('metric') == 'degree')).row(0, named=True)
    assert degree['p_value'] == pytest.approx(mannwhitneyu([1, 2], [2, 2, 1], alternative='two-sided').pvalue)
    assert degree['background_n'] == 3
    assert np.allclose(tests['fdr_bh'], bh_adjust(tests['p_value']))
    assert result['summary']['finite_tests'] == 8


def test_empty_source_unavailable_background_and_filters():
    graph, data = comparison_fixture()
    result = compare_source_sets(graph, data, ['alpha', 'empty'], min_publications=2)
    assert result['coverage']['network_ids'].to_list() == [1, 0]
    empty_tests = result['tests'].filter(pl.col('group') == 'empty')
    assert empty_tests['p_value'].is_nan().all()
    assert empty_tests['fdr_bh'].is_nan().all()
    assert empty_tests['query_n'].to_list() == [0] * 4
    all_data = pl.DataFrame({'canonical_id': graph['nodes'], 'all': [True] * 6})
    no_control = compare_source_sets(graph, all_data, ['all'])
    assert no_control['tests']['background_n'].to_list() == [0] * 4
    assert no_control['tests']['fdr_bh'].is_nan().all()
    assert no_control['overlap'].schema == {'source_a': pl.String, 'source_b': pl.String, 'selected_overlap': pl.Int64, 'network_overlap': pl.Int64}


def test_no_selected_network_proteins_and_invalid_sources():
    graph, data = comparison_fixture()
    with pytest.raises(ValueError, match='Select at least one'):
        compare_source_sets(graph, data, [])
    with pytest.raises(ValueError, match='No selected'):
        compare_source_sets(graph, data, ['empty'])
    with pytest.raises(ValueError, match='Unknown source'):
        compare_source_sets(graph, data, ['unknown'])


def test_advanced_analysis_preserves_edge_limit(monkeypatch):
    graph, data = comparison_fixture()
    monkeypatch.setattr(moonlighting, 'ADVANCED_EDGE_LIMIT', 1)
    with pytest.raises(ValueError, match='500,000 edges'):
        compare_source_sets(graph, data, ['alpha'], extra=True)


def test_advanced_source_comparison_mixed_clustering_values():
    graph, data = comparison_fixture()
    network = nx.path_graph(6)
    network.add_edge(3, 5)
    adjacency = nx.to_scipy_sparse_array(network, format='csr', dtype=float)
    graph.update(adj=adjacency, weights=adjacency,
                 degree=np.asarray(adjacency.sum(axis=1)).ravel())
    result = compare_source_sets(graph, data, ['alpha', 'beta'], extra=True, landmarks=6)
    metrics = result['metrics'].unique(subset='protein').sort('protein')
    assert metrics.schema['clustering'] == pl.Float64
    assert np.allclose(metrics['clustering'], list(nx.clustering(network).values()))
    assert metrics['clustering'].min() == 0
    assert metrics['clustering'].max() == 1
    assert result['tests'].height == 20
    assert result['tests']['fdr_bh'].is_finite().all()


def test_automatic_union_deduplicates_and_joins_correction_family():
    graph, data = comparison_fixture()
    result = compare_source_sets(graph, data, ['alpha', 'beta'], include_union=True)
    union = result['metrics'].filter(pl.col('group') == 'Union')
    assert union['protein'].to_list() == ['P0', 'P1', 'P2']
    assert union['protein'].n_unique() == 3
    assert result['source_ids']['Union'] == ['MISSING', 'P0', 'P1', 'P2']
    assert result['coverage'].filter(pl.col('source') == 'Union').row(0, named=True) == {
        'source': 'Union', 'selected_ids': 4, 'network_ids': 3, 'missing_ids': 1}
    assert result['summary']['background_nodes'] == 3
    assert result['tests'].height == 12
    assert np.allclose(result['tests']['fdr_bh'], bh_adjust(result['tests']['p_value']))


def test_effect_profiles_and_full_network_hub_tests(monkeypatch):
    graph, _ = comparison_fixture()
    data = pl.DataFrame({'canonical_id': graph['nodes'],
                         'low': [True, True, False, False, False, False],
                         'high': [False, False, False, False, True, True],
                         'empty': [False] * 6})
    calls = []
    def fixed_metrics(graph, cohort, **kwargs):
        calls.append(1)
        mask = [node in cohort for node in graph['nodes']]
        return {'metrics': pl.DataFrame({
            'protein': graph['nodes'], 'is_moonlighting': mask,
            'group': ['Moonlighting' if value else 'Background' for value in mask],
            'score': [0., 0., 1., 1., 2., 2.],
            'constant': [2.] * 6,
            'zero_median': [0., 0., 0., 0., 1., 1.],
            'nonfinite': [float('nan')] * 6,
        }), 'tests': pl.DataFrame({'metric': ['score', 'constant', 'zero_median', 'nonfinite']}),
                'notes': {}}
    monkeypatch.setattr(moonlighting, 'centrality_analysis', fixed_metrics)
    result = compare_source_sets(graph, data, ['low', 'high', 'empty'], include_union=True)
    assert len(calls) == 1
    profiles = result['profiles']
    def row(table, group, metric):
        return table.filter((pl.col('group') == group) & (pl.col('metric') == metric)).row(0, named=True)
    low = row(profiles, 'low', 'score')
    high = row(profiles, 'high', 'score')
    assert low['rank_biserial'] == -1
    assert high['rank_biserial'] == 1
    assert high['median_difference'] == 1
    assert high['median_fold'] == 2
    assert row(profiles, 'high', 'constant')['rank_biserial'] == 0
    assert row(profiles, 'high', 'constant')['median_fold'] == 1
    assert np.isnan(row(profiles, 'high', 'zero_median')['median_fold'])
    assert row(profiles, 'high', 'zero_median')['median_difference'] == 1
    assert np.isnan(row(profiles, 'empty', 'score')['rank_biserial'])
    assert profiles.schema['rank_biserial'] == pl.Float64
    assert profiles.schema['median_fold'] == pl.Float64
    assert np.allclose(result['tests']['fdr_bh'], bh_adjust(result['tests']['p_value']), equal_nan=True)

    hubs = result['hub_enrichment']
    high_hub = row(hubs, 'high', 'score')
    assert high_hub['population_n'] == 6
    assert high_hub['query_n'] == 2
    assert high_hub['top_n'] == 2
    assert high_hub['query_top_n'] == 2
    assert high_hub['cutoff'] == 2
    assert high_hub['fold'] == 3
    assert high_hub['p_value'] == pytest.approx(hypergeom.sf(1, 6, 2, 2))
    assert row(hubs, 'low', 'score')['fold'] == 0
    assert row(hubs, 'low', 'score')['p_value'] == 1
    constant = row(hubs, 'high', 'constant')
    assert constant['top_n'] == 6
    assert constant['fold'] == 1
    assert constant['p_value'] == 1
    assert row(hubs, 'Union', 'score')['query_n'] == 4
    assert row(hubs, 'Union', 'score')['fold'] == 1.5
    assert np.isnan(row(hubs, 'empty', 'score')['p_value'])
    assert row(hubs, 'high', 'nonfinite')['population_n'] == 0
    assert np.isnan(row(hubs, 'high', 'nonfinite')['fold'])
    assert np.allclose(hubs['fdr_bh'], bh_adjust(hubs['p_value']), equal_nan=True)
    assert result['summary']['hub_enrichment']['finite_tests'] == 9


# ---------------------------------------------------------------------------
# Distribution Plots & Heatmaps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value,label', [(0, '****'), (.0001, '***'), (.001, '**'), (.01, '*'), (.05, 'ns'), (1, 'ns'), (None, 'n/a'), (math.nan, 'n/a'), (math.inf, 'n/a'), (-.1, 'n/a')])
def test_significance_boundaries(value, label):
    assert significance_label(value) == label


def fixture_frame():
    return pl.DataFrame({'group': ['Background', 'moonprot', 'curated', 'Background', 'moonprot', 'curated'],
                         'protein': ['B', 'M', 'C', 'B2', 'M2', 'C2'], 'degree': [1., 4., 2., 2., 8., 3.]})


def test_stable_colors_order_and_full_data_q_annotations():
    frame = fixture_frame()
    tests = pl.DataFrame({'metric': ['degree', 'degree'], 'group': ['curated', 'moonprot'],
                          'comparator': ['Background', 'Background'], 'p_value': [.000001, .000001], 'fdr_bh': [.2, .009]})
    fig = metric_figure(frame, 'degree', tests, max_per_group=1)
    assert [trace.name for trace in fig.data] == ['Curated', 'MoonProt', 'Background']
    assert [trace.line.color for trace in fig.data] == [GROUP_COLORS[x] for x in ['curated', 'moonprot', 'Background']]
    assert all(len(trace.y) == 1 for trace in fig.data)
    assert [a.text for a in fig.layout.annotations][:2] == ['<b>ns</b>', '<b>**</b>']
    assert '0.009' in fig.layout.annotations[1].hovertext
    assert fig.to_json() == metric_figure(frame, 'degree', tests, max_per_group=1).to_json()


def test_existing_single_cohort_and_missing_tests():
    frame = fixture_frame().with_columns(pl.when(pl.col('group') == 'curated').then(pl.lit('Moonlighting')).otherwise(pl.col('group')).alias('group'))
    tests = pl.DataFrame({'metric': ['degree'], 'fdr_bh': [float('nan')]})
    fig = metric_figure(frame, 'degree', tests)
    assert fig.layout.annotations[0].text == '<b>n/a</b>'
    assert not metric_figure(frame, 'degree').layout.annotations
    raw_only = pl.DataFrame({'metric': ['degree'], 'p_value': [.00001]})
    assert not metric_figure(frame, 'degree', raw_only).layout.annotations


def test_empty_and_nonfinite_distributions():
    frame = pl.DataFrame(schema={'group': pl.String, 'protein': pl.String, 'degree': pl.Float64})
    assert not metric_figure(frame, 'degree').data
    frame = pl.DataFrame({'group': ['curated'], 'degree': [float('nan')]})
    assert not metric_figure(frame, 'degree').data
    with pytest.raises(ValueError):
        metric_figure(frame, 'degree', max_per_group=0)


def test_box_uses_all_finite_values():
    frame = fixture_frame()
    fig = metric_figure(frame, 'degree', plot_kind='box', max_per_group=1)
    assert all(trace.type == 'box' for trace in fig.data)
    assert list(fig.data[0].y) == [2., 3.]


def test_bar_full_data_mean_and_sem_independent_of_sampling():
    frame = fixture_frame()
    fig = metric_figure(frame, 'degree', plot_kind='bar', max_per_group=1)
    assert all(trace.type == 'bar' for trace in fig.data)
    assert fig.data[0].y[0] == pytest.approx(2.5)
    assert fig.data[0].error_y.array[0] == pytest.approx(.5)
    assert 'descriptive' in fig.layout.title.text
    singleton = metric_figure(pl.DataFrame({'group': ['curated'], 'degree': [4.]}), 'degree', plot_kind='bar')
    assert singleton.data[0].error_y.array[0] == 0


def test_observed_lines_use_full_data_medians_and_source_q_values():
    frame = fixture_frame()
    tests = pl.DataFrame({'group': ['moonprot'], 'comparator': ['Background'], 'metric': ['degree'], 'p_value': [.000001], 'fdr_bh': [.1]})
    fig = metric_figure(frame, 'degree', tests, max_per_group=1, plot_kind='observed')
    source = [trace for trace in fig.data if trace.type == 'scatter']
    assert len(source) == 2
    assert list(source[0].x) == [2.5, 2.5]
    assert list(source[1].x) == [6., 6.]
    assert source[1].name.endswith('(ns)')
    assert all(trace.line.dash == 'dash' for trace in source)
    hist = [trace for trace in fig.data if trace.type == 'histogram'][0]
    assert list(hist.x) == [1., 2.]
    assert fig.layout.xaxis.title.text == 'Degree'
    assert all(a.xref != 'x' for a in fig.layout.annotations)


@pytest.mark.parametrize('kind', ['box', 'bar', 'observed'])
def test_all_plot_types_handle_missing_finite_values(kind):
    frame = pl.DataFrame({'group': ['curated', 'Background'], 'degree': [float('nan'), None]})
    assert not metric_figure(frame, 'degree', plot_kind=kind).data
    with pytest.raises(ValueError):
        metric_figure(frame, 'degree', plot_kind='invalid')


def test_effect_mapping_and_q_not_raw_p():
    frame = pl.DataFrame({'group': ['moonprot', 'moonprot', 'moonlitdb'],
                          'metric': ['degree', 'pagerank', 'degree'], 'rank_biserial': [0., .4, -.8],
                          'p_value': [.000001] * 3, 'fdr_bh': [.5, .009, .00001],
                          'query_median': [2., 3., 4.], 'background_median': [2., 2., 6.],
                          'median_difference': [0., 1., -2.], 'median_fold': [1., 1.5, 2 / 3],
                          'query_n': [10] * 3, 'background_n': [100] * 3})
    fig = centrality_heatmap(frame); trace = fig.data[0]
    assert list(trace.y) == ['MoonProt', 'MoonLitDB']
    assert list(trace.x) == ['Degree', 'PageRank']
    assert list(trace.z[0]) == [0., .4]
    assert list(trace.z[1]) == [-.8, None]
    assert trace.text[0][0] == '+0.00<br>ns'
    assert trace.text[0][1] == '+0.40<br>**'
    assert trace.text[1][0] == '-0.80<br>****'
    assert 'Source median: 2' in trace.customdata[0][0]
    assert trace.zmin == -1 and trace.zmax == 1
    assert any(a.text == 'NA' for a in fig.layout.annotations)


def test_hub_zero_fold_and_color_saturation():
    frame = pl.DataFrame({'group': ['moonprot', 'moonprot', 'moonlitdb'],
                          'metric': ['degree', 'pagerank', 'degree'], 'fold': [0., 5., 1.],
                          'p_value': [.000001] * 3, 'fdr_bh': [.5, .009, .01],
                          'query_n': [10] * 3, 'query_top_n': [0, 5, 1], 'population_n': [100] * 3, 'top_n': [10] * 3,
                          'cutoff': [5.] * 3})
    fig = hub_enrichment_heatmap(frame); trace = fig.data[0]
    assert list(trace.z[0]) == [-1., 4.]
    assert trace.zmax == 1
    assert trace.text[0][0] == '0.00×<br>ns'
    assert trace.text[0][1] == '5.00×<br>**'
    assert trace.text[1][0] == '1.00×<br>*'
    assert '≥2×' in trace.colorbar.ticktext
    assert 'Network hubs: 10 / 100' in trace.customdata[0][1]


@pytest.mark.parametrize('function,column', [(centrality_heatmap, 'rank_biserial'), (hub_enrichment_heatmap, 'fold')])
def test_empty_and_nonfinite_heatmaps(function, column):
    assert not function(pl.DataFrame()).data
    frame = pl.DataFrame({'group': ['curated', 'moonprot'], 'metric': ['degree', 'degree'], column: [float('nan'), None], 'fdr_bh': [.00001, float('nan')]})
    fig = function(frame)
    assert list(fig.data[0].z[0]) == [None]
    assert list(fig.data[0].z[1]) == [None]
    assert all('****' not in str(row) for row in fig.data[0].text)


def test_many_metric_labels_and_missing_q_are_readable():
    metrics = ['degree', 'strength', 'pagerank', 'mean_neighbor_degree', 'core_number', 'clustering', 'betweenness_approx', 'eigenvector', 'bridging_centrality']
    frame = pl.DataFrame({'group': ['moonprot'] * len(metrics), 'metric': metrics, 'rank_biserial': [0.] * len(metrics)})
    fig = centrality_heatmap(frame)
    assert len(fig.data[0].x) == len(metrics)
    assert '<br>' in fig.data[0].x[3]
    assert fig.layout.xaxis.tickangle == 0
    assert all(text.endswith('n/a') for text in fig.data[0].text[0])
    with pytest.raises(ValueError):
        centrality_heatmap(frame, value='p_value')


# ---------------------------------------------------------------------------
# Source Comparison Test
# ---------------------------------------------------------------------------

def test_source_comparison_and_stale_results(tmp_path):
    data = tmp_path / 'data'
    (data / 'processed_networks').mkdir(parents=True)
    (data / 'mappings').mkdir()
    pl.DataFrame({'a': [f'P{i}' for i in range(7)],
                  'b': [f'P{i + 1}' for i in range(7)]}).write_parquet(data / 'processed_networks/HuRI.parquet')
    pl.DataFrame({'a': [f'P{i}' for i in range(6)],
                  'b': [f'P{i + 1}' for i in range(6)]}).write_parquet(data / 'processed_networks/BioPlex.parquet')
    rows = []
    for i in range(5):
        rows.append({'uniprot': f'P{i}', 'Gene Names': f'G{i}', 'Reviewed': 'reviewed',
                     'moonlitdb_number_of_publications': 2,
                     'moonlitdb_number_of_moonlighting_functions': 2,
                     **{source: i in ({'curated': [0, 1], 'moonprot': [1, 2],
                                       'moonlitdb': [2, 3], 'multitask': [0, 4], 'moondb_curated': [3, 4]}.get(source, []))
                        for source in SOURCES}})
    pl.DataFrame(rows).write_csv(data / 'allmoon_with_moonlitdb_human.csv')
    (data / 'mappings/moonlighting_uniprot.json').write_text(json.dumps(
        {'mapping': {f'P{i}': [f'P{i}'] for i in range(5)}}))

    moon = load_moon_data(data)
    g_huri = graph_from_path(data / 'processed_networks/HuRI.parquet')
    g_bioplex = graph_from_path(data / 'processed_networks/BioPlex.parquet')
    compare_sources = ['moonprot', 'multitask', 'moonlitdb']

    res_huri = compare_source_sets(g_huri, moon, compare_sources, include_union=True)
    res_bioplex = compare_source_sets(g_bioplex, moon, compare_sources, include_union=True)

    assert res_huri['summary']['background_nodes'] == 3
    assert res_huri['metrics'].filter(pl.col('protein') == 'P1')['group'].to_list() == ['moonprot', 'Union']
    assert res_huri['tests'].height == 16
    assert res_huri['profiles'].height == 16
    assert res_huri['hub_enrichment'].height == 16

    res_no_union = compare_source_sets(g_huri, moon, compare_sources, include_union=False)
    assert 'Union' not in res_no_union['metrics']['group'].to_list()
    assert res_no_union['profiles'].filter(pl.col('group') == 'Union').is_empty()

    res_alt = compare_source_sets(g_huri, moon, ['curated', 'moonprot'], include_union=True)
    assert set(res_alt['coverage']['source'].to_list()) == {'curated', 'moonprot', 'Union'}
