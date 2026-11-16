import json

from clustering_explorer import saved_runs


def test_discovery_uses_portable_paths_and_keeps_failed_runs(tmp_path):
    for key, status in [('old', 'failed'), ('new', 'success')]:
        target = tmp_path / 'HuRI' / 'igraph.infomap' / key
        target.mkdir(parents=True)
        (target / 'result.json').write_text(json.dumps({
            'status': status, 'job': {'target': '/another/computer/run'},
        }))
    broken = tmp_path / 'HuRI' / 'igraph.infomap' / 'broken'
    broken.mkdir()
    (broken / 'result.json').write_text('{')
    frame = saved_runs(tmp_path)
    assert frame.height == 2
    assert set(frame['status']) == {'failed', 'success'}
    assert all(str(tmp_path) in directory for directory in frame['directory'])
    assert set(frame['network']) == {'HuRI'}


def test_saved_partition_enrichment_uses_full_background():
    import polars as pl
    from clustering_explorer import saved_cluster_enrichment
    frame = pl.DataFrame({'protein': ['A', 'B', 'B', 'C'],
                          'community': ['one', 'one', 'two', 'two']})
    population = pl.DataFrame({'protein': ['A', 'B', 'C', 'D']})
    result = saved_cluster_enrichment(frame, population, {'A', 'B'})
    one = result.filter(pl.col('community') == 'one').row(0, named=True)
    assert one['protein_set_hits'] == 2
    assert one['expected_hits'] == 1
    assert one['fold_enrichment'] == 2
    assert abs(one['pvalue'] - 1/6) < 1e-12
    assert abs(one['qvalue'] - 1/3) < 1e-12


def test_unified_selector_includes_saved_and_unsaved_methods(tmp_path):
    import polars as pl
    from clustering_explorer import clustering_options, community_selection
    (tmp_path / 'memberships.parquet').touch()
    (tmp_path / 'community_sizes.parquet').touch()
    runs = pl.DataFrame({'network': ['HuRI', 'HuRI'],
        'method': ['atlas.louvain', 'atlas.label_propagation'],
        'status': ['success', 'failed'], 'directory': [str(tmp_path)] * 2})
    options = clustering_options(runs, 'HuRI')
    saved = options.filter(pl.col('method') == 'atlas.louvain').row(0, named=True)
    assert saved['saved_versions'] == 1
    assert 'saved version' in saved['label']
    missing = options.filter(pl.col('method') == 'atlas.label_propagation').row(0, named=True)
    assert missing['saved_versions'] == 0
    assert 'not saved' in missing['label']
    assert community_selection([2, 1, '2']) == ['1', '2']
    assert community_selection('2') == ['2']
    assert community_selection(None) is None


def test_cross_run_enrichment_corrects_all_communities_and_excludes_stale_runs(tmp_path):
    import polars as pl
    import numpy as np
    from clustering_explorer import all_saved_cluster_enrichment
    from networks import _fingerprint
    from moonlighting import bh_adjust
    network = tmp_path / 'network.parquet'
    pl.DataFrame({'a': ['A', 'C'], 'b': ['B', 'D']}).write_parquet(network)
    fingerprint = _fingerprint(network)['sha256']
    runs = []
    for name, communities, expected in [
        ('first', ['one', 'one', 'two', 'two'], fingerprint),
        ('second', ['one', 'two', 'two', 'two'], fingerprint),
        ('stale', ['one'] * 4, 'wrong-fingerprint')]:
        directory = tmp_path / name
        directory.mkdir()
        pl.DataFrame({'protein': ['A', 'B', 'C', 'D'], 'community': communities}).write_parquet(directory / 'memberships.parquet')
        (directory / 'result.json').write_text(json.dumps({'job': {'provenance': {'input': {'sha256': expected}}}}))
        runs.append(dict(network='HuRI', method=name, run='version', status='success', directory=str(directory)))
    results, diagnostics = all_saved_cluster_enrichment(pl.from_dicts(runs), 'HuRI', network, {'A', 'B'})
    assert results.height == 4
    assert set(results['method']) == {'first', 'second'}
    assert diagnostics['method'].to_list() == ['stale']
    np.testing.assert_allclose(results['qvalue'], bh_adjust(results['pvalue']))
    best = results.row(0, named=True)
    assert best['method'] == 'first' and best['community'] == 'one'
    assert best['protein_set_fraction'] == 1
    assert best['fold_enrichment'] == 2
    assert best['qvalue'] > best['run_qvalue']
    assert results.filter(pl.col('protein_set_hits') == 0)['pvalue'].to_list() == [1.0]
