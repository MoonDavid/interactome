"""Native runner preparation preserves accession identities and overlap."""
import polars as pl
import pytest

from pipelines.run_ocg_c import prepare_edges, decode_partition


def test_preparation_and_partition_preserve_overlap_and_unassigned(tmp_path):
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a': ['A', 'B', 'C', 'D', 'E'],
                  'b': ['B', 'A', 'D', 'E', 'E']}).write_parquet(path)
    edges, nodes = prepare_edges(path)
    assert edges.height == 3
    assert nodes['protein'].to_list() == ['A', 'B', 'C', 'D', 'E']
    output = tmp_path / 'partition.txt'
    output.write_text('>ClusterID:1||\n1\n2\n>ClusterID:2||\n1\n')
    members = decode_partition(output, nodes)
    assert members.filter(pl.col('protein') == 'A').height == 2
    assert members.filter(pl.col('community').is_null())['protein'].to_list() == ['C', 'D', 'E']


def test_empty_network_is_rejected(tmp_path):
    path = tmp_path / 'network.parquet'
    pl.DataFrame({'a':['A'], 'b':['A']}).write_parquet(path)
    with pytest.raises(ValueError, match='No non-self edges'):
        prepare_edges(path)
