"""Tests for canonical network preparation, UniProt mapping, topology summaries, and overlap plots."""
import gzip
import json
import polars as pl
import pytest

import networks
from networks import (
    build_canonical_networks, degree_distribution, discover_datasets,
    load_network, map_ids, membership_counts, normalize_identifier,
    prepare_huri_swissprot, save_humap_version, save_scored_version,
    score_column, summarize_network, threshold_curve, upset_figure,
    venn_figure,
)


# ---------------------------------------------------------------------------
# Network Topology Stats & Loading
# ---------------------------------------------------------------------------

def test_undirected_cleaning_and_components(tmp_path):
    path = tmp_path / 'HuRI.tsv'
    path.write_text('A\tB\nB\tA\nB\tC\nD\tE\nA\tA\n\tZ\n')
    net = load_network(path)
    stats = summarize_network(net)
    assert stats['nodes'] == 5
    assert stats['edges'] == 3
    assert stats['components'] == 2
    assert stats['largest_component'] == 3
    assert stats['mean_degree'] == 1.2
    assert stats['density'] == .3
    assert stats['duplicate_rows'] == 1
    assert stats['self_loop_rows'] == 1
    assert stats['missing_rows'] == 1
    assert degree_distribution(net).to_dicts() == [{'degree': 1, 'count': 4}, {'degree': 2, 'count': 1}]


def test_humap_threshold_inclusive_and_empty(tmp_path):
    path = tmp_path / 'humap.pairsWprob.gz'
    with gzip.open(path, 'wt') as f:
        f.write('A\tB\t0.5\nB\tC\t0.9\nC\tD\t0.1\n')
    net = load_network(path, .5)
    assert net.input_rows == 3
    assert net.retained_rows == 2
    assert summarize_network(net)['edges'] == 2
    empty = summarize_network(load_network(path, 1.0))
    assert empty['nodes'] == empty['edges'] == empty['components'] == 0
    with pytest.raises(ValueError):
        load_network(path, float('nan'))


def test_bioplex_exact_isoforms(tmp_path):
    path = tmp_path / 'BioPlex.tsv'
    path.write_text('Uniprot A\tUniprot B\tSymbol A\nP1\tP1-2\tX\nP1-2\tP2\tY\n')
    stats = summarize_network(load_network(path))
    assert stats['nodes'] == 3
    assert stats['edges'] == 2


def test_canonical_parquet_and_saved_version(tmp_path):
    path = tmp_path / 'huMAP2_0.2.parquet'
    pl.DataFrame({'a': ['P1', 'P1'], 'b': ['P2', 'P3'], 'probability': [0.2, 0.8]}).write_parquet(path)
    stats = summarize_network(load_network(path))
    assert stats['nodes'] == 3
    assert stats['edges'] == 2


# ---------------------------------------------------------------------------
# Canonical UniProt Networks & Confidence Versions
# ---------------------------------------------------------------------------

def fixture(root):
    folder = root / 'networks'
    folder.mkdir()
    (folder / 'HuRI.tsv').write_text('ENSG1\tENSG2\nENSG1\tENSG3\nENSG1\tENSG4\n')
    (folder / 'BioPlex_HEK293T_merged_dedup.tsv').write_text('Uniprot A\tUniprot B\nP1-2\tP2\nP1\tP2\nP1-2\tP1\n')
    with gzip.open(folder / 'humap2_ppis_ACC_20200821.pairsWprob.gz', 'wt') as out:
        out.write('P1-2\tP2\t0.2\nP2\tP1\t0.9\nP2\tP3\t0.5\nP4\tP5\t0.1\nP1-2\tP1\t0.99\n')
    return {'ENSG1': ['P1', 'P3'], 'ENSG2': ['P2'], 'ENSG3': ['P1']}


def test_mapping_canonicalization_and_audit(tmp_path):
    mapping = fixture(tmp_path)
    paths = build_canonical_networks(mapping, tmp_path)
    huri = pl.read_parquet(paths['HuRI'])
    assert set(huri.iter_rows()) == {('P1', 'P2'), ('P2', 'P3'), ('P1', 'P3')}
    assert pl.read_parquet(paths['BioPlex']).rows() == [('P1', 'P2')]
    humap = pl.read_parquet(paths['huMAP2'])
    assert humap.filter((pl.col('a') == 'P1') & (pl.col('b') == 'P2'))['probability'][0] == .9
    audit = json.loads(paths['HuRI'].with_suffix('.json').read_text())
    assert audit['unmapped_ids'] == ['ENSG4']
    assert audit['unmapped_edge_rows'] == 1
    assert audit['self_loop_rows'] == 1


def test_versions_discovery_inclusive_curve_and_stale(tmp_path):
    build_canonical_networks(fixture(tmp_path), tmp_path)
    version = save_humap_version(.5, tmp_path)
    assert version.name == 'huMAP2_0.5.parquet'
    assert pl.read_parquet(version).height == 2
    assert 'huMAP2_0.5' in discover_datasets(tmp_path)
    assert threshold_curve(tmp_path, [0, .5, .9, 1]).to_dicts() == [
        {'threshold': 0., 'nodes': 5, 'edges': 3},
        {'threshold': .5, 'nodes': 3, 'edges': 2},
        {'threshold': .9, 'nodes': 2, 'edges': 1},
        {'threshold': 1., 'nodes': 0, 'edges': 0}]
    meta = tmp_path / 'canonical/huMAP2.json'
    content = json.loads(meta.read_text())
    content['output']['sha256'] = 'changed'
    meta.write_text(json.dumps(content))
    assert 'huMAP2_0.5' not in discover_datasets(tmp_path)
    with pytest.raises(ValueError):
        save_humap_version(float('nan'), tmp_path)


def test_explicit_unmapped_uniprot_drops_instead_of_suffix_fallback(tmp_path):
    mapping = fixture(tmp_path)
    mapping['P2'] = []
    paths = build_canonical_networks(mapping, tmp_path)
    assert pl.read_parquet(paths['BioPlex']).is_empty()
    assert pl.read_parquet(paths['huMAP2']).rows() == [('P4', 'P5', .1)]
    audit = json.loads(paths['huMAP2'].with_suffix('.json').read_text())
    assert audit['unmapped_ids'] == ['P2']
    assert audit['unmapped_edge_rows'] == 3


def test_source_self_loop_removed_before_ambiguous_mapping_expands(tmp_path):
    mapping = fixture(tmp_path)
    (tmp_path / 'networks/HuRI.tsv').write_text('ENSG1\tENSG1\nENSG1\tENSG2\n')
    paths = build_canonical_networks(mapping, tmp_path)
    assert set(pl.read_parquet(paths['HuRI']).iter_rows()) == {('P1', 'P2'), ('P2', 'P3')}
    audit = json.loads(paths['HuRI'].with_suffix('.json').read_text())
    assert audit['source_self_loop_rows'] == 1
    assert audit['self_loop_rows'] == 0


def test_close_threshold_names_do_not_overwrite(tmp_path):
    build_canonical_networks(fixture(tmp_path), tmp_path)
    first = save_humap_version(.20000001, tmp_path)
    second = save_humap_version(.20000002, tmp_path)
    assert first != second
    assert first.exists() and second.exists()
    assert first.name == 'huMAP2_0.20000001.parquet'


@pytest.mark.parametrize('header', ['pInt', 'p(Interaction)'])
def test_bioplex_native_score_max_invalid_base_edges_and_inclusive_versions(tmp_path, header):
    mapping = fixture(tmp_path)
    (tmp_path / 'networks/BioPlex_HEK293T_merged_dedup.tsv').write_text(
        f'Uniprot A\tUniprot B\t{header}\n'
        'P1-2\tP2\t0.5\nP2\tP1\t0.95\nP2\tP3\t0.05\n'
        'P4\tP5\tbad\nP6\tP7\tNaN\nP8\tP9\t1.2\n'
        'P10\tP11\t\nP12\tP13\tinf\n')
    paths = build_canonical_networks(mapping, tmp_path, names=('BioPlex',))
    frame = pl.read_parquet(paths['BioPlex'])
    assert frame.height == 7
    assert frame.schema == {'a': pl.String, 'b': pl.String, 'pInt': pl.Float64}
    assert frame.filter((pl.col('a') == 'P1') & (pl.col('b') == 'P2'))['pInt'][0] == .95
    assert frame['pInt'].null_count() == 5
    audit = json.loads(paths['BioPlex'].with_suffix('.json').read_text())
    assert audit['invalid_score_rows'] == 5
    assert audit['source_score_column'] == header
    assert audit['score_column'] == 'pInt'
    assert threshold_curve(tmp_path, [0, .05, .95, 1], name='BioPlex').to_dicts() == [
        {'threshold': 0., 'nodes': 3, 'edges': 2},
        {'threshold': .05, 'nodes': 3, 'edges': 2},
        {'threshold': .95, 'nodes': 2, 'edges': 1},
        {'threshold': 1., 'nodes': 0, 'edges': 0},
    ]
    version = save_scored_version('BioPlex', .95, tmp_path)
    assert pl.read_parquet(version).rows() == [('P1', 'P2', .95)]
    meta = json.loads(version.with_suffix('.json').read_text())
    assert meta['network'] == 'BioPlex'
    assert meta['score_column'] == 'pInt'
    assert 'BioPlex_0.95' in discover_datasets(tmp_path)
    frame.head(1).write_parquet(paths['BioPlex'])
    assert 'BioPlex_0.95' not in discover_datasets(tmp_path)


def test_unscored_bioplex_remains_supported_with_clear_filter_error(tmp_path):
    build_canonical_networks(fixture(tmp_path), tmp_path)
    assert score_column('BioPlex') == 'pInt'
    with pytest.raises(ValueError, match='no pInt score column'):
        save_scored_version('BioPlex', .5, tmp_path)
    with pytest.raises(ValueError, match='no pInt score column'):
        threshold_curve(tmp_path, name='BioPlex')
    with pytest.raises(ValueError, match='no supported'):
        save_scored_version('HuRI', .5, tmp_path)


def test_bioplex_discovery_requires_correct_native_score_metadata(tmp_path):
    mapping = fixture(tmp_path)
    source = tmp_path / 'networks/BioPlex_HEK293T_merged_dedup.tsv'
    source.write_text('Uniprot A\tUniprot B\tpInt\nP1\tP2\t0.9\n')
    build_canonical_networks(mapping, tmp_path, names=('BioPlex',))
    version = save_scored_version('BioPlex', .5, tmp_path)
    meta_path = version.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    meta['score_column'] = 'probability'
    meta_path.write_text(json.dumps(meta))
    assert 'BioPlex_0.5' not in discover_datasets(tmp_path)


# ---------------------------------------------------------------------------
# Dataset Overlaps (Venn & UpSet)
# ---------------------------------------------------------------------------

def edge_frame(rows):
    return pl.DataFrame(rows, schema=['a', 'b'], orient='row')


def test_exclusive_edges_reverse_duplicates_and_totals():
    data = {'A': edge_frame([('P1', 'P2'), ('P2', 'P1'), ('P2', 'P3')]),
            'B': edge_frame([('P2', 'P1'), ('P3', 'P4')]),
            'C': edge_frame([('P1', 'P2'), ('P4', 'P5')])}
    counts = membership_counts(data)
    assert dict(zip(counts['mask'], counts['count'])) == {1: 1, 2: 1, 4: 1, 7: 1}
    assert counts['count'].sum() == 4
    for bit in range(3):
        assert counts.filter((pl.col('mask') // (1 << bit)) % 2 == 1)['count'].sum() == 2
    assert len(venn_figure(counts, list(data)).data) == 1
    assert len(upset_figure(counts, list(data)).data) > 1


def test_nodes_and_empty_dataset():
    data = {'A': edge_frame([('P1', 'P2'), ('P2', 'P3')]),
            'B': edge_frame([('P2', 'P4')]),
            'Empty': edge_frame([])}
    counts = membership_counts(data, 'Nodes')
    assert dict(zip(counts['mask'], counts['count'])) == {1: 2, 2: 1, 3: 1}
    empty = membership_counts({'Empty': edge_frame([])})
    assert empty.is_empty()
    assert len(upset_figure(empty, ['Empty']).data) == 1


def test_overlap_validation():
    with pytest.raises(ValueError):
        membership_counts({})
    with pytest.raises(ValueError):
        membership_counts({'A': edge_frame([])}, 'invalid')
    with pytest.raises(ValueError):
        venn_figure(membership_counts({'A': edge_frame([])}), ['A'])


def test_venn_percentages_use_union_and_handle_empty_regions():
    counts = pl.DataFrame({'mask': [1, 3, 7], 'count': [2, 3, 5]})
    plot = venn_figure(counts, ['A', 'B', 'C'])
    labels = dict(zip([1, 2, 3, 4, 5, 6, 7], plot.data[0].text))
    assert '20.00%' in labels[1]
    assert '30.00%' in labels[3]
    assert '50.00%' in labels[7]
    assert '0.00%' in labels[2]
    assert 'Union total: 10' in plot.data[0].hovertext[0]
    empty = pl.DataFrame(schema={'mask': pl.Int64, 'count': pl.Int64})
    assert all('0.00%' in label for label in venn_figure(empty, ['A', 'B']).data[0].text)


# ---------------------------------------------------------------------------
# UniProt ID Mapping API & Swiss-Prot Audit
# ---------------------------------------------------------------------------

class Reply:
    def __init__(self, payload, links=None):
        self.payload = payload
        self.links = links or {}
    def json(self): return self.payload
    def raise_for_status(self): pass


class Session:
    def __init__(self): self.calls = []
    def mount(self, *args): pass
    def post(self, url, **kwargs):
        self.calls.append(url)
        return Reply({'jobId': 'test-job'})
    def get(self, url, **kwargs):
        self.calls.append(url)
        if '/status/' in url: return Reply({'results': True})
        if 'page2' in url:
            return Reply({'results': [{'from': 'ENSG1', 'to': {'primaryAccession': 'Q00002', 'organism': {'taxonId': 9606}}}]})
        return Reply({'results': [
            {'from': 'ENSG1', 'to': {'primaryAccession': 'P00001', 'organism': {'taxonId': 9606}}},
            {'from': 'ENSG2', 'to': {'primaryAccession': 'MOUSE1', 'organism': {'taxonId': 10090}}},
        ]}, {'next': {'url': 'https://rest.uniprot.org/page2'}})


def test_mapping_pagination_human_multiplicity_and_cache(tmp_path, monkeypatch):
    session = Session()
    monkeypatch.setattr(networks.requests, 'Session', lambda: session)
    payload = map_ids(['ENSG1', 'ENSG2', 'ENSG1'], 'Ensembl', tmp_path)
    assert payload['mapping'] == {'ENSG1': ['P00001', 'Q00002'], 'ENSG2': []}
    assert payload['one_to_many'] == {'ENSG1': ['P00001', 'Q00002']}
    assert payload['unmapped'] == ['ENSG2']
    calls = len(session.calls)
    assert map_ids(['ENSG2', 'ENSG1'], 'Ensembl', tmp_path) == payload
    assert len(session.calls) == calls


def test_mixed_humap_identifier_normalization():
    assert normalize_identifier('ENSG000001.4') == ('Ensembl', 'ENSG000001')
    assert normalize_identifier('57336') == ('GeneID', '57336')
    assert normalize_identifier('sp|P01891|1A68_HUMAN') == ('UniProtKB_AC-ID', 'P01891')
    assert normalize_identifier('P24530-3') == ('UniProtKB_AC-ID', 'P24530')


def test_huri_reviewed_filter_and_exclusion_audit(tmp_path):
    (tmp_path / 'networks').mkdir()
    (tmp_path / 'mappings').mkdir()
    (tmp_path / 'networks/HuRI.tsv').write_text('G1\tG2\nG1\tG3\nG1\tG4\n')
    (tmp_path / 'mappings/swissprot_human_accessions.tsv').write_text('Entry\nP1\nP2\nP3\n')
    (tmp_path / 'mappings/swissprot_human_source.json').write_text('{"uniprot_release":"test"}')
    mapping = {'G1': ['P1', 'Q1'], 'G2': ['P2', 'P3'], 'G3': ['Q3'], 'G4': []}
    reviewed = prepare_huri_swissprot(mapping, tmp_path)
    assert reviewed == {'G1': ['P1'], 'G2': ['P2', 'P3'], 'G3': [], 'G4': []}
    rows = pl.read_csv(tmp_path / 'mappings/HuRI_swissprot_audit.tsv', separator='\t').to_dicts()
    assert [row['status'] for row in rows] == ['reviewed_match', 'reviewed_match', 'other_matches_only', 'no_uniprot_match']
    summary = json.loads((tmp_path / 'mappings/HuRI_swissprot_summary.json').read_text())
    assert summary['genes_with_multiple_reviewed_matches'] == 1
    assert summary['nonloop_source_rows_lacking_reviewed_endpoint'] == 2
    path = build_canonical_networks(mapping, tmp_path, names=('HuRI',), huri_mapping=reviewed)['HuRI']
    assert set(pl.read_parquet(path).iter_rows()) == {('P1', 'P2'), ('P1', 'P3')}
