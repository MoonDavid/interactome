"""huMAP3 follows the shared audited mapping and native-score pipeline."""
import json

import polars as pl

import networks as n


def test_humap3_mapping_filtering_and_versions(tmp_path):
    raw = tmp_path / 'raw_networks'
    raw.mkdir()
    source = raw / n.FILES['huMAP3']
    source.write_text('OLD\tQ1\t0.05\nQ1\tOLD\t0.95\nP1-2\tQ1\t0.8\n'
                      'P1\tOLD\t1\nMISSING\tQ1\t0.9\nQ2\tQ1\tbad\nQ3\tQ1\t1.1\n')
    assert n.source_endpoint_profile(n.scan_mapping_source('huMAP3', source))[1] == 7
    mapping = {'OLD': ['P1'], 'P1': ['P1'], 'P1-2': ['P1'],
               'Q1': ['Q1'], 'Q2': ['Q2'], 'Q3': ['Q3']}
    path = n.build_canonical_networks(mapping, tmp_path, names=('huMAP3',))['huMAP3']
    assert pl.read_parquet(path).rows() == [('P1', 'Q1', .95)]
    audit = json.loads(path.with_suffix('.json').read_text())
    assert audit['unmapped_ids'] == ['MISSING']
    assert audit['invalid_score_rows'] == 2
    assert audit['self_loop_rows'] == 1
    assert audit['filtering_stages'][-1]['edges'] == 1
    version = n.save_scored_version('huMAP3', .95, tmp_path)
    assert pl.read_parquet(version).height == 1
    assert 'huMAP3_0.95' in n.discover_datasets(tmp_path)
    assert n.threshold_curve(tmp_path, [0, .95, 1], name='huMAP3')['edges'].to_list() == [1, 1, 0]
    n.save_scored_version('huMAP3', 1, tmp_path)
    n.rebuild_mapped_networks(mapping, tmp_path)
    assert pl.read_parquet(tmp_path / 'processed_networks/versions/huMAP3_1.parquet').is_empty()
