"""Offline enrichment and the optional CyREST comparison."""
import gzip
import json

import numpy as np
import polars as pl
import pytest
import requests

import annotate_clusters as annotate
from moonlighting import bh_adjust, prepare_go_annotations


@pytest.fixture
def inputs(tmp_path):
    proteins = [f'P{number:05}' for number in range(1, 81)]
    network = tmp_path / 'network.parquet'
    pl.DataFrame({'a': proteins[::2], 'b': proteins[1::2]}).write_parquet(network)
    members = tmp_path / 'members.csv'
    groups = [
        ('A', proteins[:10] + [proteins[0] + '-2']),
        ('B', proteins[5:15]), ('C', proteins[15:20]),
        ('D', proteins[20:25]), ('E', proteins[20:25]),
        (None, [proteins[-1]]),
    ]
    pl.DataFrame([(p, c) for c, ids in groups for p in ids],
                 schema=['protein', 'community'], orient='row').write_csv(members)
    obo = tmp_path / 'go-basic.obo'
    obo.write_text(
        'format-version: 1.2\n'
        '\n[Term]\nid: GO:0000001\nname: Parent\nnamespace: biological_process\n'
        '\n[Term]\nid: GO:0000002\nname: Child\nnamespace: biological_process\n'
        'is_a: GO:0000001 ! Parent\n'
        '\n[Term]\nid: GO:0000003\nname: Zero hits\nnamespace: cellular_component\n',
        encoding='utf-8',
    )
    gaf = tmp_path / 'human.gaf'
    go_rows = [(p, 'GO:0000002') for p in proteins[:5]] + [(p, 'GO:0000003') for p in proteins[30:35]]
    gaf.write_text('!gaf-version: 2.2\n' + ''.join(
        f'UniProtKB\t{p}\tG\t\t{term}\tPMID:1\tEXP\t\tP\t\t\tprotein\ttaxon:9606\n'
        for p, term in go_rows
    ), encoding='utf-8')
    compressed = gaf.with_suffix('.gaf.gz')
    compressed.write_bytes(gzip.compress(gaf.read_bytes()))
    reactome = tmp_path / 'UniProt2Reactome.txt'
    reactome.write_text(''.join(
        f'{p}-1\tR-HSA-1\turl\tShared pathway\tTAS\tHomo sapiens\n'
        for p in proteins[5:10]
    ) + f'{proteins[10]}\tR-MMU-2\turl\tMouse pathway\tIEA\tMus musculus\n', encoding='utf-8')
    return network, members, obo, gaf, compressed, reactome


def test_offline_bundle_and_global_correction(inputs, tmp_path):
    network, members, obo, _, compressed, reactome = inputs
    output = tmp_path / 'result'
    annotate.main([
        '--network', str(network), '--memberships', str(members), '--gaf', str(compressed),
        '--obo', str(obo), '--reactome', str(reactome), '--output', str(output),
    ])
    labels = {row['community']: row for row in pl.read_csv(output / 'cluster_labels.csv').to_dicts()}
    assert labels['A']['primary_label'] == 'Child'
    assert labels['A']['primary_library'] == 'GO_BP'
    assert labels['A']['interface_label'] == 'Shared pathway'
    assert labels['A']['interface_library'] == 'Reactome'
    assert labels['A']['core_size'] == 5
    assert labels['A']['full_context_label'] is None
    assert labels['B']['primary_status'] == 'no_qualifying_term'
    assert labels['B']['full_context_label'] == 'Shared pathway'
    assert labels['C']['primary_status'] == 'no_qualifying_term'
    assert labels['C']['interface_status'] == 'empty_subset'
    assert labels['D']['primary_status'] == labels['E']['primary_status'] == 'empty_subset'
    assert labels['D']['full_size'] == labels['D']['interface_size'] == 5
    evidence = pl.read_parquet(output / 'enrichment.parquet')
    assert 'GO:0000001' in evidence['term_id'].to_list()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['coverage']['background'] == 80
    assert manifest['coverage']['unassigned'] == 55
    assert manifest['tests'] == 4 * 12
    assert manifest['coverage']['libraries']['Reactome']['eligible_terms'] == 1
    padding = manifest['tests'] - evidence.height
    dense = bh_adjust(np.r_[evidence['pvalue'].to_numpy(), np.ones(padding)])
    np.testing.assert_allclose(evidence['qvalue'].to_numpy(), dense[:evidence.height])


def test_gaf_compression_preserves_annotations(inputs, tmp_path):
    _, _, obo, gaf, compressed, _ = inputs
    plain = prepare_go_annotations(tmp_path / 'plain', gaf, obo)
    zipped = prepare_go_annotations(tmp_path / 'compressed', compressed, obo)
    assert pl.read_parquet(plain).equals(pl.read_parquet(zipped))


def test_unmatched_membership_is_reported():
    population = pl.DataFrame({'protein': ['P00001']})
    with pytest.raises(ValueError, match='outside the network background: P00002'):
        annotate.membership_queries(
            pl.DataFrame({'protein': ['P00002-2'], 'community': [0]}), population,
        )


def test_cytoscape_memberships_and_comparison(monkeypatch):
    rows = [
        {'SUID': 1, 'accession': 'P00001-1', 'group': [0, 0], 'description': 'First'},
        {'SUID': 2, 'accession': 'P00002', 'group': [0, 1], 'description': 'Shared'},
        {'SUID': 3, 'accession': 'P00003', 'group': '1;2', 'description': 'Third'},
        {'SUID': 4, 'accession': 'P00004', 'group': None},
        {'SUID': 5, 'accession': 'other', 'group': None},
    ]
    pairs, population, source = annotate.annotation_input(rows, 42, 'accession', 'group', ';', 'http://mock')
    assert population.height == 4
    assert source['cytoscape']['rows_without_valid_accession'] == 1
    full = annotate.membership_queries(pairs, population).filter(pl.col('subset') == 'full')
    terms = pl.DataFrame({'protein': ['P00001', 'P00002', 'P00003'], 'term_text': ['one', 'two', 'three']})
    calls = []

    def fake_cyrest(url, method, path, body=None):
        calls.append((method, path, body))
        if path.endswith('/rows'):
            return rows
        if path == annotate.AUTOANNOTATE_COMMAND:
            assert body['network'] == 'SUID:42'
            assert body['returnJsonOnly'] == 'true'
            assert body['useClusterMaker'] == 'false'
            if body['labelColumn'] == 'description':
                return {'data': [{'nodes': [3], 'label': 'Third'},
                                 {'nodes': [3, 2], 'label': 'Shared'},
                                 {'nodes': [2, 1], 'label': 'First'}], 'errors': []}
            return {'data': [{'nodes': [2, 3], 'label': 'two three'},
                             {'nodes': [1, 4], 'label': 'wrong members'},
                             {'nodes': [3], 'label': 'three'}], 'errors': []}

    monkeypatch.setattr(annotate, 'cyrest', fake_cyrest)
    result, summary = annotate.annotation_comparison(full, population, terms, 42, 'accession', 'description', 'http://mock')
    assert result['autoannotate_description_label'].to_list() == ['First', 'Shared', 'Third']
    assert result['autoannotate_terms_label'].to_list() == [None, 'two three', 'three']
    assert result['autoannotate_terms_status'][0] == 'membership_mismatch'
    assert summary['status'] == 'partial'
    assert summary['passes']['terms']['unexpected_member_sets'] == 1
    put = next(body for method, _, body in calls if method == 'PUT')
    column = next(key for key in put['data'][0] if key.endswith('_clusters'))
    assert put['data'][1][column] == ['0', '1']
    assert [method for method, _, _ in calls].count('DELETE') == 2


def test_unavailable_autoannotate_preserves_offline_results(inputs, tmp_path, monkeypatch):
    network, members, obo, gaf, _, reactome = inputs
    output = tmp_path / 'result'

    def unavailable(*args, **kwargs):
        raise requests.ConnectionError('Cytoscape is unavailable')

    monkeypatch.setattr(annotate, 'cyrest', unavailable)
    annotate.main([
        '--network', str(network), '--memberships', str(members), '--gaf', str(gaf), '--obo', str(obo),
        '--reactome', str(reactome), '--output', str(output), '--compare-cytoscape', '42',
        '--uniprot-column', 'accession', '--description-column', 'description',
    ])
    labels = pl.read_csv(output / 'cluster_labels.csv')
    assert labels['primary_label'][0] == 'Child'
    assert labels['autoannotate_terms_status'].unique().to_list() == ['unavailable']
    assert json.loads((output / 'manifest.json').read_text())['cytoscape_comparison']['status'] == 'unavailable'
