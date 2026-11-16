"""Check full-network ranks, missing cells and coverage-aware partner overlap."""
import numpy as np
import polars as pl
import pytest

from moonlighting import cross_network_analysis, graph_from_path
from topology_connectivity import degree_percentiles, degree_percentile_heatmap, partner_overlap_heatmap


def test_percentiles_use_background_and_missing_cells_remain_missing():
    metrics = pl.DataFrame({
        'network': ['N1'] * 3 + ['N2'] * 3,
        'protein': ['A', 'B', 'C', 'A', 'B', 'D'],
        'degree': [1., 1., 2., 4., 1., 1.],
        'is_moonlighting': [True, False, True, True, False, False],
    })
    table = degree_percentiles(metrics)
    assert table.filter((pl.col('protein') == 'A') & (pl.col('network') == 'N1'))['degree_percentile'][0] == .5
    fig = degree_percentile_heatmap(table, ['A', 'C'], ['N1', 'N2'])
    z = np.asarray(fig.data[1].z, dtype=float)
    assert z[0].tolist() == [.5, 1.]
    assert z[1, 0] == 1.
    assert np.isnan(z[1, 1])
    assert 'Not present' in fig.data[0].text[1][1]
    assert fig.data[1].zmin == 0 and fig.data[1].zmax == 1


def test_partner_heatmap_distinguishes_full_and_shared_population(tmp_path):
    def graph(name, partners):
        path = tmp_path / f'{name}.parquet'
        pl.DataFrame({'a': ['A', 'A', 'B'], 'b': ['B', partners, 'C']}).write_parquet(path)
        return graph_from_path(path)
    result = cross_network_analysis(graph('N1', 'X'), graph('N2', 'Y'), {'A'})
    table = result['table'].with_columns(pl.lit('N1').alias('Network A'), pl.lit('N2').alias('Network B'))
    row = table.row(0, named=True)
    assert row['partner_jaccard'] == pytest.approx(1 / 3)
    assert row['partner_jaccard_shared'] == 1
    assert row['partners_A_in_both'] == row['partners_B_in_both'] == 1
    assert row['shared_partner_union'] == 1
    shared = partner_overlap_heatmap(table, ['A', 'ABSENT'], [('N1', 'N2')])
    full = partner_overlap_heatmap(table, ['A'], [('N1', 'N2')], shared_only=False)
    assert shared.data[1].z[0][0] == 1
    assert np.isnan(shared.data[1].z[0][1])
    assert full.data[1].z[0][0] == pytest.approx(1 / 3)
    assert 'N1=1, N2=1' in shared.data[1].text[0][0]


def test_no_shared_eligible_partners_is_undefined(tmp_path):
    graphs = []
    for other in ('X', 'Y'):
        path = tmp_path / f'{other}.parquet'
        pl.DataFrame({'a': ['A'], 'b': [other]}).write_parquet(path)
        graphs.append(graph_from_path(path))
    row = cross_network_analysis(*graphs, {'A'})['table'].row(0, named=True)
    assert row['shared_partner_union'] == 0
    assert np.isnan(row['partner_jaccard_shared'])
