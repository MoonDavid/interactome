"""Automatic scheduling preserves one cache writer and complete result bundles."""
import json
import hashlib
import os
from pathlib import Path

import polars as pl
import psutil
import pytest

import automatic_monte_carlo as automatic


def setup_data(tmp_path):
    network = tmp_path / 'processed_networks' / 'tiny.parquet'
    network.parent.mkdir()
    pl.DataFrame({'a': ['A', 'B', 'C'], 'b': ['B', 'C', 'D']}).write_parquet(network)
    return network


def test_existing_writer_prevents_duplicate(tmp_path, monkeypatch):
    network = setup_data(tmp_path)
    jobs = tmp_path / 'analysis_runs' / 'automatic_monte_carlo'
    jobs.mkdir(parents=True)
    (jobs / 'writer.json').write_text(json.dumps({'pid': psutil.Process().pid,
                                                'created': psutil.Process().create_time()}))
    monkeypatch.setattr(automatic.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('duplicate writer')))
    result = automatic.ensure_precomputed(tmp_path, [network], 'custom', {'A', 'C'}, 'lcc')
    assert result['status'] == 'running'


def test_schedule_publish_and_reuse_both_analyses(tmp_path, monkeypatch):
    network = setup_data(tmp_path)
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        # Execute the real pipeline synchronously on a tiny fixed graph.
        automatic.worker(command[-1])
        return psutil.Process()

    monkeypatch.setattr(automatic.subprocess, 'Popen', launch)
    args = (tmp_path, [network], 'custom', {'A', 'C'}, 'lcc')
    assert automatic.ensure_precomputed(*args)['status'] == 'running'
    assert automatic.ensure_precomputed(*args)['status'] == 'success'
    assert len(calls) == 1
    for analysis in ('shortest_path', 'cohesion'):
        prefix = tmp_path / 'analysis_runs' / f'custom_tiny_{analysis}'
        assert pl.read_parquet(prefix.with_suffix('.parquet')).height == 1000
        meta = json.loads(prefix.with_suffix('.json').read_text())
        assert meta['scope'] == 'lcc'
        assert meta['automatic_job_key']
        assert Path(f'{prefix}_summary.csv').exists()
    # Changing the population must not reuse the previous result.
    jobs = tmp_path / 'analysis_runs' / 'automatic_monte_carlo'
    (jobs / 'writer.json').unlink(missing_ok=True)
    assert automatic.ensure_precomputed(tmp_path, [network], 'custom', {'A', 'C'}, 'network')['status'] == 'running'
    assert len(calls) == 2
    assert (tmp_path / 'analysis_runs/archive/monte_carlo_lcc/custom_tiny_cohesion.json').exists()


@pytest.mark.parametrize('network_name', ['tiny', 'huMAP2_0.05', 'huMAP2_0.95'])
def test_legacy_lcc_batch_reused_without_launching_simulation(tmp_path, monkeypatch, network_name):
    network = setup_data(tmp_path)
    network = network.rename(network.with_name(f'{network_name}.parquet'))
    source = tmp_path / 'allmoon_with_moonlitdb_human.csv'
    mapping = tmp_path / 'mappings/moonlighting_uniprot.json'
    mapping.parent.mkdir()
    source.write_text('local source')
    mapping.write_text('{}')
    for path in (source, mapping):
        os.utime(path, (100, 100))
    monkeypatch.setattr(automatic, '_legacy_cohort', lambda *args: {'A', 'C'})
    monkeypatch.setattr(automatic.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('unnecessary simulation')))
    folder = tmp_path / 'analysis_runs'
    folder.mkdir()
    for analysis in ('shortest_path', 'cohesion'):
        prefix = folder / f'moonlitdb_{network_name}_{analysis}'
        Path(f'{prefix}.json').write_text(json.dumps(dict(
            protein_set='moonlitdb', scope='lcc', seed=42, simulations=1000,
            reviewed_only=False, min_publications=0,
            network_sha256=hashlib.sha256(network.read_bytes()).hexdigest())))
        Path(f'{prefix}.parquet').touch()
        Path(f'{prefix}_summary.csv').write_text('fdr_bh\n0.2\n')
    result = automatic.ensure_precomputed(tmp_path, [network], 'moonlitdb', {'A', 'C'}, 'lcc')
    assert result['status'] == 'success'
    assert result['ready_networks'] == [str(network.resolve())]
    # Source edits and changed current membership invalidate legacy reuse.
    config = dict(data=str(tmp_path), protein_set='moonlitdb', cohort=['A'],
                  reviewed_only=False, min_publications=0)
    meta_path = folder / f'moonlitdb_{network_name}_shortest_path.json'
    meta = json.loads(meta_path.read_text())
    assert not automatic.cohort_matches(meta, config, meta_path)
    config['cohort'] = ['A', 'C']
    os.utime(source, (meta_path.stat().st_mtime + 10,) * 2)
    assert not automatic.cohort_matches(meta, config, meta_path)
