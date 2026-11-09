"""Adapter contracts: node identity, partial covers, sparse graph inputs, isolation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

pytest.importorskip('typer')
pytest.importorskip('rich')
import cluster_networks as cli


@pytest.fixture
def graph_path(tmp_path):
    path = tmp_path / 'canonical' / 'HuRI.parquet'
    path.parent.mkdir()
    pl.DataFrame({'a': ['P1', 'P2', 'P1', 'P3', 'P4', 'P5', 'P4'],
                  'b': ['P2', 'P3', 'P3', 'P4', 'P5', 'P6', 'P6']}).write_parquet(path)
    return path


def parameters():
    return dict(seed=42, clusters=2, dimensions=2, resolution=1.0, clique_k=3,
                inflation=2.0, membership_cutoff=0.2, threads=1,
                costly_node_limit=5000, dense_node_limit=3000)


def test_cover_preserves_overlap_and_unassigned():
    frame, sizes, covered = cli.membership_table(np.array(['A', 'B', 'C', 'D']),
                                                [{0, 1}, {1, 2}, {1, 0}])
    assert covered == 3
    assert sizes['len'].to_list() == [2, 2]
    assert frame.filter(pl.col('protein') == 'B').height == 2
    assert frame.filter(pl.col('protein') == 'D')['community'].to_list() == [None]
    with pytest.raises(ValueError, match='outside'):
        cli.membership_table(['A'], [{7}])


def test_indexed_graph_cleans_and_keeps_accessions(tmp_path):
    path = tmp_path / 'pairs.parquet'
    pl.DataFrame({'a': ['P02', 'P01', 'P03', None],
                  'b': ['P01', 'P02', 'P03', 'P04']}).write_parquet(path)
    graph = cli.indexed_graph(path)
    assert graph['nodes'].tolist() == ['P01', 'P02']
    assert graph['edge_count'] == 1
    np.testing.assert_array_equal(graph['adj'].toarray(), [[0, 1], [1, 0]])


@pytest.mark.parametrize('method,module', [
    ('networkx.louvain_communities', 'networkx'),
    ('networkx.girvan_newman', 'networkx'),
    ('networkx.k_clique_communities', 'networkx'),
    ('atlas.fuzzy_c_means', 'networkx'),
    ('igraph.leiden', 'igraph'),
    ('igraph.walktrap', 'igraph'),
    ('leidenalg.RBConfigurationVertexPartition', 'leidenalg'),
    ('sknetwork.Louvain', 'sknetwork'),
    ('sknetwork_hierarchy.Paris', 'sknetwork'),
    ('sklearn.KMeans', 'sklearn'),
    ('sklearn.SpectralClustering', 'sklearn'),
    ('markov_clustering.mcl', 'markov_clustering'),
])
def test_adapter_population_contract(graph_path, method, module):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f'{module} optional dependency not installed')
    graph = cli.indexed_graph(graph_path)
    groups, soft, _ = cli.execute(cli.catalog()[method], graph, parameters())
    frame, sizes, covered = cli.membership_table(graph['nodes'], groups)
    assert set(frame['protein']) == set(graph['nodes'])
    assert sizes.height >= 1
    assert covered >= 1
    if soft is not None:
        assert soft.shape[0] == len(graph['nodes'])


def test_worker_error_and_resume_identity(graph_path, tmp_path):
    target = tmp_path / 'job'
    job = {'path': str(graph_path), 'target': str(target), 'key': 'expected',
           'method': 'networkx.louvain_communities', 'parameters': parameters()}
    assert cli._worker(job) == 0
    assert cli.reusable(target, 'expected')['status'] == 'success'
    assert cli.reusable(target, 'changed') is None
    (target / 'memberships.parquet').unlink()
    assert cli.reusable(target, 'expected') is None
    job['method'] = 'does.not.exist'
    assert cli._worker(job) == 1
    assert json.loads((target / 'result.json').read_text())['status'] == 'failed'


def test_cli_subprocess_and_resume(graph_path, tmp_path):
    output = tmp_path / 'results'
    args = [sys.executable, str(Path(cli.__file__)), 'run', '--root', str(graph_path.parent.parent),
            '--output', str(output), '--method', 'networkx.louvain_communities', '--timeout', '60']
    first = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0, first.stdout + first.stderr
    assert pl.read_parquet(output / 'summary.parquet')['status'].to_list() == ['success']
    second = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert second.returncode == 0, second.stdout + second.stderr
    assert pl.read_parquet(output / 'summary.parquet')['resumed'].to_list() == [True]


def test_guard_precedes_networkx_allocation(graph_path, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'EDGE_LIMIT', 1)
    monkeypatch.setattr(cli, 'indexed_graph', lambda _: pytest.fail('Graph allocated before edge guard'))
    target = tmp_path / 'guarded'
    assert cli._worker({'path': str(graph_path), 'target': str(target), 'key': 'guard',
                        'method': 'networkx.louvain_communities', 'parameters': parameters()}) == 0
    assert json.loads((target / 'result.json').read_text())['status'] == 'skipped'
