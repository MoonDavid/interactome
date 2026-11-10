"""Offline end-to-end checks for the standalone mapping command."""
import gzip
import json

import polars as pl
import requests

import networks
from pipelines.map_networks import main


def test_offline_mapping_labels_cutoffs_and_stale_source(tmp_path, monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError('Offline mapping must not contact UniProt')
    monkeypatch.setattr(requests.sessions.Session, 'request', no_network)
    raw = tmp_path / 'raw_networks'
    raw.mkdir()
    (raw / networks.FILES[networks.TAGC_NAME]).write_text('Interactor,InteractionB\nA,B\nB,A\nD,B\nA,X\n')
    with gzip.open(raw / networks.FILES['huMAP2'], 'wt') as handle:
        handle.write('A\tB\t0.05\nB\tA\t0.2\nD\tB\t0.7\nA\tC\t0.95\nA\tX\t0.9\n')
    entries = {source: {'primary': labels, 'targets': labels, 'status': 'Cross-reference'}
               for source, labels in {'A': ['P1', 'P2'], 'D': ['P2', 'P1'],
                                      'B': ['Q1'], 'C': ['Q2'], 'X': []}.items()}
    sources = {name: {'source': networks._fingerprint(networks.network_source_path(tmp_path, name)),
                      'ids': list(entries)} for name in ('huMAP2', networks.TAGC_NAME)}
    mapping_dir = tmp_path / 'mappings'
    mapping_dir.mkdir()
    (mapping_dir / networks.NETWORK_MAPPING).write_text(json.dumps({'entries': entries, 'sources': sources}))

    assert main(['--data-dir', str(tmp_path), '--offline', '--cutoff', '0.7', '--isoforms', 'retain']) == 0
    datasets = networks.discover_datasets(tmp_path)
    assert pl.read_parquet(datasets['huMAP2']).rows() == [('P1|P2', 'Q1', .7), ('P1|P2', 'Q2', .95)]
    assert pl.read_parquet(datasets['huMAP2_0.95']).height == 1
    assert 'huMAP2_0.7' in datasets
    report = pl.read_csv(tmp_path / 'processed_networks/mapping_filtering_steps.csv')
    assert set(report['Isoform policy']) == {'collapse', 'retain'}
    assert 'Deduplicate before UniProt mapping' in set(report['Filtering step'])
    assert 'Deduplicate after UniProt mapping' in set(report['Filtering step'])
    assert 'huMAP2_0.7' in set(report['Network'])

    # Stale lookup/source combinations must fail before rebuilding any graph.
    before = datasets[networks.TAGC_NAME].read_bytes()
    source = raw / networks.FILES[networks.TAGC_NAME]
    source.write_text(source.read_text() + 'B,C\n')
    assert main(['--data-dir', str(tmp_path), '--offline']) == 1
    assert datasets[networks.TAGC_NAME].read_bytes() == before
    assert 'source changed' in capsys.readouterr().err


def test_huri_only_needs_no_mapping_audit(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('HuRI preparation must not contact UniProt')
    monkeypatch.setattr(requests.sessions.Session, 'request', no_network)
    raw = tmp_path / 'raw_networks'
    raw.mkdir()
    (raw / 'HuRI.psi').write_text('uniprotkb:P12345-2\tuniprotkb:Q67890\n'
                                  'uniprotkb:P12345\tuniprotkb:Q67890\n')
    assert main(['--data-dir', str(tmp_path), '--huri-only', '--isoforms', 'retain']) == 0
    report = pl.read_csv(tmp_path / 'processed_networks/HuRI_filtering_steps.csv')
    assert set(report['Network']) == {'HuRI'}
    assert set(report['Isoform policy']) == {'collapse', 'retain'}
    assert not (tmp_path / 'mappings').exists()
