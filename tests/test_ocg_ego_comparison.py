import io
from zipfile import ZipFile

import polars as pl

from ocg_ego_comparison import compare_ego_records, member_label_table, comparison_zip


def records():
    return [dict(network='first',run='one',
        edges=pl.DataFrame({'a':['A','A','A','B'],'b':['B','C','D','C']}),
        memberships=pl.DataFrame({'protein':['A','A','B','C','D'], 'community':[1,2,1,2,None]}),
        labels=pl.DataFrame({'community':['1','2'], 'GO_BP_enrichment':['first function','second function'],
                             'GO_BP_qvalue':[.001,.04]})),
        dict(network='second',run='two',
        edges=pl.DataFrame({'a':['A','A','B'],'b':['B','E','E']}),
        memberships=pl.DataFrame({'protein':['A','B','B','E'],'community':[1,1,3,3]}))]


def test_full_neighbor_comparison_is_independent_of_drawing_cap():
    views,summary,table,overlap=compare_ego_records(records(),'A',max_neighbors=1)
    assert summary['full_neighbor_count'].to_list()==[3,2]
    assert summary['displayed_proteins'].to_list()==[2,2]
    assert overlap['shared_neighbors'][0]==1
    assert overlap['neighbor_jaccard'][0]==.25
    assert set(table['protein'])=={'A','B','C','D','E'}
    missing=table.filter((pl.col('network')=='second') & (pl.col('protein')=='D')).row(0,named=True)
    assert not missing['present_in_network'] and missing['cluster_count']==0
    unassigned=table.filter((pl.col('network')=='first') & (pl.col('protein')=='D')).row(0,named=True)
    assert unassigned['present_in_network'] and unassigned['cluster_count']==0
    labels=member_label_table(views['first'],['A'])
    assert labels['community'].to_list()==['1','2']
    assert labels['GO_BP_enrichment'].to_list()==['first function','second function']
    assert labels['GO_BP_significance'].to_list()==['**','*']


def test_reference_neighborhood_absence_and_exports():
    source=records()
    source.append(dict(network='absent',edges=pl.DataFrame({'a':['B'],'b':['Z']}),
                       memberships=pl.DataFrame({'protein':['B','Z'],'community':[9,9]})))
    views,summary,table,overlap=compare_ego_records(source,'A',max_neighbors=3,reference='first')
    assert set(views['second']['nodes']['protein'])=={'A','B'}
    assert set(views['absent']['nodes']['protein'])=={'B'}
    assert not summary.filter(pl.col('network')=='absent')['center_present'][0]
    assert '9' not in views['first']['memberships']['community'].to_list()
    with ZipFile(io.BytesIO(comparison_zip(views,summary,table,overlap,'A'))) as archive:
        assert 'protein_memberships.csv' in archive.namelist()
        assert '1_first/norma.zip' in archive.namelist()
        assert b'first function' in archive.read('1_first/cluster_labels.csv')


def test_absent_center_is_not_created_as_a_network_node():
    source=records()
    views,summary,table,overlap=compare_ego_records(source,'MISSING')
    assert not any(summary['center_present'])
    assert all(v['nodes'].is_empty() for v in views.values())
    assert overlap['neighbor_jaccard'][0] is None
    assert table['protein'].to_list()==['MISSING','MISSING']
