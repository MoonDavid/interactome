"""Exact fixed-graph distances, matching, and restart behavior."""
import networkx as nx
import numpy as np
import pytest

from pipelines.precompute_monte_carlo import (
    degree_matched_samples, distance_cache, pair_mean, run_analysis,
)


def test_exact_distances_and_cache_reuse(tmp_path, monkeypatch):
    graph = nx.path_graph(['a', 'b', 'c', 'd'])
    nodes = sorted(graph)
    distances, sentinel, _ = distance_cache(graph, nodes, [0], tmp_path)
    assert pair_mean(distances, np.array([0, 3]), sentinel) == (3, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed row must be reused')
    monkeypatch.setattr(nx, 'single_source_shortest_path_length', forbidden)
    distance_cache(graph, nodes, [0], tmp_path)


def test_matching_and_full_graph_paths(tmp_path):
    graph = nx.path_graph(['a', 'b', 'c', 'd', 'e'])
    nodes, query, samples = degree_matched_samples(graph, {'a', 'e'}, 20, 42)
    assert all(set(sample) == set(query) for sample in samples)
    rows, null, _ = run_analysis(graph, {'a', 'e'}, 20, 42, True, ['shortest_path'], tmp_path)
    assert rows[0]['observed'] == 4  # intermediates are outside the protein set
    assert rows[0]['p_value'] == 1
    assert null['mean_shortest_path'].to_list() == [4.] * 20


def test_disconnected_pairs_and_changed_topology(tmp_path):
    graph = nx.Graph([('a', 'b'), ('c', 'd')])
    nodes = sorted(graph)
    distances, sentinel, first = distance_cache(graph, nodes, range(4), tmp_path)
    mean, fraction = pair_mean(distances, np.arange(4), sentinel)
    assert mean == 1
    assert fraction == pytest.approx(1/3)
    graph.add_edge('b', 'c')
    distances, sentinel, second = distance_cache(graph, nodes, range(4), tmp_path)
    assert first != second
    assert pair_mean(distances, np.arange(4), sentinel)[0] == pytest.approx(10/6)


def test_cli_verbose_saves_summary(tmp_path, capsys):
    import json
    import polars as pl
    from pipelines.precompute_monte_carlo import main

    (tmp_path / 'mappings').mkdir()
    (tmp_path / 'mappings/moonlighting_uniprot.json').write_text(
        json.dumps({'mapping': {'a': ['a'], 'd': ['d']}}))
    pl.DataFrame({'uniprot': ['a', 'd'], 'moonlitdb': ['true', 'true']}).write_csv(
        tmp_path / 'allmoon_with_moonlitdb_human.csv')
    path = tmp_path / 'tiny.parquet'
    pl.DataFrame({'a': ['a', 'b', 'c'], 'b': ['b', 'c', 'd']}).write_parquet(path)
    assert main(['--data', str(tmp_path), '--networks', str(path), '--simulations', '3',
                 '--verbose', '--no-progress', '--analyses', 'shortest_path', 'cohesion']) == 0
    output = capsys.readouterr().err
    assert 'Checkpoint saved' in output
    assert 'Monte Carlo results' in output
    assert 'Summary saved' in output
    summary = next((tmp_path / 'analysis_runs').glob('*summary.csv'))
    assert pl.read_csv(summary).height == 3


def test_each_source_uses_independent_arguments(monkeypatch):
    import pipelines.precompute_monte_carlo as pipeline

    calls = []
    monkeypatch.setattr(pipeline, 'run_protein_set', lambda args: calls.append(args))
    assert pipeline.main(['--networks', 'tiny.parquet', '--protein-set', 'each']) == 0
    assert [args.protein_set for args in calls] == pipeline.SOURCES
    assert len({id(args) for args in calls}) == len(pipeline.SOURCES)
