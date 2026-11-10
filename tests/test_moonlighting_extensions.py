"""Regression checks for new catalogue methods with exact small-graph answers."""
import networkx as nx
import numpy as np
import polars as pl
import pytest

from moonlighting import graph_from_path
from moonlighting_extensions import (
    _connected_five_node_orbits, _hodge_participation, _link_homology,
    _rooted_graphlets, cohort_filtration, cohort_separation, removal_efficiency,
)


def _graph(tmp_path, edges):
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a': [str(a) for a, _ in edges], 'b': [str(b) for _, b in edges]}).write_parquet(path)
    return graph_from_path(path)


def test_graphlets_classify_standard_orbits_and_budget():
    graph = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)])
    neighbors = [set(graph[i]) for i in range(5)]
    counts, order = _rooted_graphlets(neighbors, 0, 5)
    assert order == 5
    assert counts['orbit_0'] == 4
    assert counts['orbit_3'] == 2
    assert counts['orbit_2'] == 4
    assert counts['orbit_11'] == 4
    assert len(_connected_five_node_orbits()) == 58
    partial, order = _rooted_graphlets(neighbors, 0, 5, budget=4)
    assert order == 2
    assert partial == {'orbit_0': 4}


def test_filled_triangles_remove_harmonic_cycles():
    cycle = nx.cycle_graph(4)
    triangle = nx.complete_graph(3)
    assert _link_homology(cycle) == (1, 0)
    assert _link_homology(triangle) == (0, 1)
    participation, rank = _hodge_participation(cycle)
    assert rank == 1
    np.testing.assert_allclose(participation, .5)
    _, rank = _hodge_participation(triangle)
    assert rank == 0


def test_separation_handles_tied_voronoi_cells_and_overlap(tmp_path):
    graph = _graph(tmp_path, list(nx.path_graph(7).edges))
    result = cohort_separation(graph, {'0', '2', '6'}, {'2', '4'}, permutations_count=0)['summary']
    assert result['within_A'] == pytest.approx(8 / 3)
    assert result['within_B'] == 2
    assert result['between'] == pytest.approx(6 / 5)
    assert result['overlap_in_lcc'] == 1


def test_removal_efficiency_counts_disconnected_pairs_as_zero(tmp_path):
    graph = _graph(tmp_path, list(nx.path_graph(3).edges))
    result = removal_efficiency(graph, {'1'}, permutations_count=0)['table']
    assert result['global_efficiency_estimate'].to_list() == pytest.approx([5 / 6, 0])
    assert result['lcc_fraction_original'].to_list() == pytest.approx([1, 1 / 3])


def test_filtration_uses_native_score_and_inclusive_saved_floor(tmp_path):
    version = tmp_path / 'version.parquet'
    pl.DataFrame({'a': ['P1', 'P1', 'P2', 'P3'], 'b': ['P2', 'P3', 'P3', 'P4'],
                  'probability': [.05, .95, .1, None]}).write_parquet(version)
    table = cohort_filtration(tmp_path, 'huMAP2', {'P1', 'P2'}, .96, saved_version=version, minimum=.05)['table']
    assert table['threshold'].min() == .05
    assert table.filter(pl.col('threshold') == .05)['cohort_internal_edges'].item() == 1
    assert table.filter(pl.col('threshold') == .95)['cohort_nodes'].item() == 1
    assert table.filter(pl.col('threshold') == .96)['network_edges'].item() == 0
