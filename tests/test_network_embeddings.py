import sys
import types

import numpy as np
import polars as pl
import pytest

from network_embeddings import embed, indexed_adjacency, project
from embedding_display import embedding_figure


@pytest.mark.parametrize('dimensions', [2, 3])
@pytest.mark.parametrize('outline', ['circle-open', 'diamond-open'])
def test_protein_set_outlines_preserve_degree_colors_and_coordinates(dimensions, outline):
    points = pl.DataFrame({'protein': ['P1', 'P2', 'P3'], 'degree': [1, 10, 100],
                          'x': [0., 1., 2.], 'y': [2., 0., 1.], 'z': [1., 2., 0.],
                          'in_protein_set': [False, True, False]})
    original = points.clone()
    fig = embedding_figure(points, dimensions, 'PCA', 'Degree (log10 + 1)',
                           'MoonLitDB', outline=outline)
    assert len(fig.data) == 3
    np.testing.assert_allclose(fig.data[0].marker.color, np.log10([2, 11, 101]))
    assert list(fig.data[0].x) == points['x'].to_list()
    assert list(fig.data[-1].x) == [1.]
    assert fig.data[-1].marker.symbol == outline
    assert fig.data[-1].name == 'MoonLitDB · highlighted'
    assert fig.data[-1].showlegend and not fig.data[-2].showlegend
    if dimensions == 3:
        assert list(fig.data[-1].z) == [2.]
    assert points.equals(original)
    plain = embedding_figure(points, dimensions, 'PCA', 'Degree (log10 + 1)',
                             'MoonLitDB', highlight=False)
    assert len(plain.data) == 1


def test_empty_protein_set_and_search_highlighting_are_independent():
    points = pl.DataFrame({'protein': ['P1', 'P2'], 'degree': [1, 2],
                          'x': [0., 1.], 'y': [1., 0.], 'in_protein_set': [False, False]})
    fig = embedding_figure(points, 2, 'UMAP', 'Degree (log10 + 1)', 'MoonLitDB', query='P2')
    assert len(fig.data) == 2
    assert fig.data[-1].name == 'Search match'
    assert fig.data[-1].marker.symbol == 'x'
    assert fig.data[-1].text == ('P2',)


def test_sparse_topology_preserves_order_and_deduplicates(tmp_path):
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a': ['B', 'A', 'A', 'C', 'D'], 'b': ['A', 'B', 'B', 'B', 'D']}).write_parquet(path)
    nodes, adjacency, edges = indexed_adjacency(path)
    assert nodes == ['A', 'B', 'C', 'D']
    assert edges == 2
    np.testing.assert_array_equal(adjacency.toarray(), [[0, 1, 0, 0], [1, 0, 1, 0],
                                                       [0, 1, 0, 0], [0, 0, 0, 0]])


def test_embedding_cache_and_source_invalidation(tmp_path, monkeypatch):
    calls = []

    class Model:
        def __init__(self, n_components, **kwargs):
            self.width = n_components

        def fit_transform(self, adjacency):
            calls.append(adjacency.shape)
            return np.arange(adjacency.shape[0] * self.width).reshape(-1, self.width)

    monkeypatch.setitem(sys.modules, 'nodevectors', types.SimpleNamespace(ProNE=Model))
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a': ['B', 'A', 'C', 'D'], 'b': ['A', 'C', 'D', 'E']}).write_parquet(path)
    first = embed(path, dimensions=2, output=tmp_path / 'runs')
    assert pl.read_parquet(first / 'vectors.parquet')['protein'].to_list() == ['A', 'B', 'C', 'D', 'E']
    assert embed(path, dimensions=2, output=tmp_path / 'runs') == first
    assert len(calls) == 1
    pl.DataFrame({'a': ['B', 'A', 'C', 'D'], 'b': ['A', 'D', 'E', 'E']}).write_parquet(path)
    assert embed(path, dimensions=2, output=tmp_path / 'runs') != first
    assert len(calls) == 2


@pytest.mark.parametrize('dimensions', [2, 3])
def test_pca_projection_keeps_accessions(dimensions):
    pytest.importorskip('sklearn')
    frame = pl.DataFrame({'protein': ['P1', 'P2', 'P3', 'P4'], 'degree': [1, 2, 2, 1],
                          'e0': [1., 0., 2., 3.], 'e1': [0., 1., 2., 4.], 'e2': [4., 1., 0., 2.]})
    points = project(frame, method='PCA', dimensions=dimensions)
    assert points['protein'].to_list() == frame['protein'].to_list()
    assert points.columns == ['protein', 'degree', *'xyz'[:dimensions]]
    assert np.isfinite(points.select(*'xyz'[:dimensions]).to_numpy()).all()


def test_reject_nonfinite_embedding(tmp_path, monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, adj):
            return np.full((adj.shape[0], 2), np.nan)

    monkeypatch.setitem(sys.modules, 'nodevectors', types.SimpleNamespace(ProNE=Model))
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a': ['A', 'B', 'C'], 'b': ['B', 'C', 'D']}).write_parquet(path)
    with pytest.raises(ValueError, match='non-finite'):
        embed(path, dimensions=2, output=tmp_path / 'runs')
