"""Tests for canonical network preparation, UniProt mapping, topology summaries, and overlap plots."""
import gzip
import json
import polars as pl
import pytest

import networks
from moonlighting_extensions import cohort_filtration
from networks import (
    build_canonical_networks, degree_distribution, discover_datasets,
    load_network, map_ids, membership_counts, normalize_identifier,
    save_humap_version, save_scored_version,
    score_column, summarize_network, threshold_curve, upset_figure,
    venn_figure,
)


# ---------------------------------------------------------------------------
# Network Topology Stats & Loading
# ---------------------------------------------------------------------------

def test_huri_psi_policies_use_source_accessions(tmp_path):
    directory = tmp_path / 'raw_networks'
    directory.mkdir()
    (directory / 'HuRI.psi').write_text(
        '#ID(s) interactor A\tID(s) interactor B\tAlt IDs\n'
        'uniprotkb:P12345\tuniprotkb:Q67890\t-\n'
        'uniprotkb:Q67890\tuniprotkb:P12345-2\t-\n'
        'uniprotkb:P12345-1\tuniprotkb:P12345-2\t-\n'
        'uniprotkb:Q67890\tuniprotkb:Q67890\t-\n'
        'ensembl:ENSP123.1\tuniprotkb:Q67890\t-\n'
    )
    outputs = build_canonical_networks(
        {'P12345': ['P99999'], 'ENSP123': ['P99999']}, tmp_path, names=('HuRI',),
    )
    expected = {
        'collapse': [('P12345', 'Q67890')],
        'canonical_only': [('P12345', 'Q67890')],
        'retain': [('P12345', 'Q67890'), ('P12345-1', 'P12345-2'), ('P12345-2', 'Q67890')],
    }
    assert set(networks.available_huri_isoform_policies(tmp_path)) == set(expected)
    for policy, pairs in expected.items():
        path = discover_datasets(tmp_path, huri_isoforms=policy)['HuRI']
        assert discover_datasets(tmp_path, isoforms=policy)['HuRI'] == path
        assert pl.read_parquet(path).rows() == pairs
        audit = json.loads(path.with_suffix('.json').read_text())
        assert audit['isoform_policy'] == policy
        assert audit['rows_without_two_uniprot_ids'] == 1
        assert audit['source_self_loop_rows'] == 1
    assert discover_datasets(tmp_path)['HuRI'] == outputs['HuRI']
    audit = json.loads(outputs['HuRI'].with_suffix('.json').read_text())
    assert audit['self_loops_after_isoform_policy'] == audit['duplicate_rows'] == 1
    del audit['isoform_policy']
    outputs['HuRI'].with_suffix('.json').write_text(json.dumps(audit))
    assert 'HuRI' not in discover_datasets(tmp_path)


@pytest.mark.parametrize('name,score', [(k, v) for k, v in networks.SCORE_COLUMNS.items() if k != 'HuRI'])
def test_mapped_isoform_views_keep_native_scores_and_separate_versions(tmp_path, monkeypatch, name, score):
    source = networks.network_source_path(tmp_path, name)
    source.parent.mkdir()
    header = 'Uniprot A\tUniprot B\tp(Interaction)\n' if name == 'BioPlex' else ''
    source.write_text(header + 'P11111-2\tP22222\t0.9\nP22222\tP11111-2\t0.7\n'
                      'P11111\tP22222\t0.5\nP11111-9\tP33333\t0.99\n')
    current = {'P11111', 'P11111-2', 'P22222', 'P33333'}
    monkeypatch.setattr(networks, '_current_uniprot_ids', lambda directory: (current, 'test-release'))
    monkeypatch.setattr(networks, '_search_uniprot', lambda *args, **kwargs: ({'P11111': ['P11111']}, {}, set()))
    mapping = networks.prepare_mapping(tmp_path)
    build_canonical_networks(mapping, tmp_path, names=(name,))
    collapsed = save_scored_version(name, 0.9, tmp_path)

    assert networks.prepare_isoform_views(tmp_path, 'retain') == []
    assert networks.prepare_isoform_views(tmp_path, 'canonical_only') == []
    distinct = discover_datasets(tmp_path, isoforms='retain')
    unsuffixed = discover_datasets(tmp_path, isoforms='canonical_only')
    assert pl.read_parquet(distinct[name]).rows() == [('P11111', 'P22222', .5), ('P11111-2', 'P22222', .9)]
    assert pl.read_parquet(unsuffixed[name]).rows() == [('P11111', 'P22222', .5)]
    label = collapsed.stem
    assert pl.read_parquet(distinct[label]).rows() == [('P11111-2', 'P22222', .9)]
    assert pl.read_parquet(unsuffixed[label]).is_empty()
    assert discover_datasets(tmp_path)[label] == collapsed
    assert threshold_curve(tmp_path, [.9], name, parent=distinct[name])['edges'].item() == 1
    result = cohort_filtration(tmp_path, name, {'P11111-2', 'P22222'}, parent=distinct[name])
    assert result['table'].filter(pl.col('threshold') == .9)['cohort_internal_edges'].item() == 1
    assert json.loads(distinct[label].with_suffix('.json').read_text())['score_column'] == score

    distinct[label].unlink()
    assert networks.prepare_isoform_views(tmp_path, 'retain') == []
    assert distinct[label].is_file()
    source.write_text(source.read_text() + 'P11111\tP33333\t0.75\n')
    assert discover_datasets(tmp_path, isoforms='retain') == {}
    assert networks.prepare_isoform_views(tmp_path, 'retain') == [name]


def test_undirected_cleaning_and_components(tmp_path):
    path = tmp_path / 'pairs.tsv'
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
    folder = root / 'raw_networks'
    folder.mkdir()
    (root / 'processed_networks').mkdir()
    (folder / networks.HURI_PSI).write_text('uniprotkb:P1\tuniprotkb:P2\n')
    (folder / networks.FILES[networks.TAGC_NAME]).write_text(
        'Interactor,InteractionB\nENSG1,ENSG2\nENSG1,ENSG3\nENSG1,ENSG4\n')
    ((root / 'processed_networks') / 'BioPlex_HEK293T_merged_dedup.tsv').write_text('Uniprot A\tUniprot B\nP1-2\tP2\nP1\tP2\nP1-2\tP1\n')
    with gzip.open(folder / 'humap2_ppis_ACC_20200821.pairsWprob.gz', 'wt') as out:
        out.write('P1-2\tP2\t0.2\nP2\tP1\t0.9\nP2\tP3\t0.5\nP4\tP5\t0.1\nP1-2\tP1\t0.99\n')
    return {**{f'P{i}': [f'P{i}'] for i in range(1, 14)}, 'P1-2': ['P1'],
            'ENSG1': ['P1', 'P3'], 'ENSG2': ['P2'], 'ENSG3': ['P1']}


def test_mapping_canonicalization_and_audit(tmp_path):
    mapping = fixture(tmp_path)
    paths = build_canonical_networks(mapping, tmp_path)
    tagc = pl.read_parquet(paths[networks.TAGC_NAME])
    assert set(tagc.iter_rows()) == {('P1|P3', 'P2'), ('P1', 'P1|P3')}
    assert pl.read_parquet(paths['BioPlex']).rows() == [('P1', 'P2')]
    humap = pl.read_parquet(paths['huMAP2'])
    assert humap.filter((pl.col('a') == 'P1') & (pl.col('b') == 'P2'))['probability'][0] == .9
    audit = json.loads(paths[networks.TAGC_NAME].with_suffix('.json').read_text())
    assert audit['unmapped_ids'] == ['ENSG4']
    assert audit['unmapped_edge_rows'] == 1
    assert audit['self_loop_rows'] == 0


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
    meta = tmp_path / 'processed_networks/huMAP2.json'
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


def test_source_self_loop_removed_before_multilabel_mapping(tmp_path):
    mapping = fixture(tmp_path)
    source = networks.network_source_path(tmp_path, networks.TAGC_NAME)
    source.write_text('Interactor,InteractionB\nENSG1,ENSG1\nENSG1,ENSG2\n')
    path = build_canonical_networks(mapping, tmp_path, names=(networks.TAGC_NAME,))[networks.TAGC_NAME]
    assert set(pl.read_parquet(path).iter_rows()) == {('P1|P3', 'P2')}
    audit = json.loads(path.with_suffix('.json').read_text())
    assert audit['source_self_loop_rows'] == 1
    assert audit['self_loop_rows'] == 0


def test_filtering_stages_preserve_multilabel_nodes_and_deduplicate_twice(tmp_path):
    mapping = fixture(tmp_path)
    source = networks.network_source_path(tmp_path, networks.TAGC_NAME)
    source.write_text('Interactor,InteractionB\nA,B\nB,A\nA,C\nD,B\nE,B\nF,B\nD,A\nG,H\nA,A\n,B\n')
    mapping.update({'A': ['P3', 'P1', 'P1'], 'B': ['P2'], 'C': ['P2'],
                    'D': ['P1', 'P3'], 'E': [], 'F': ['P2'], 'G': ['P1'], 'H': ['P3']})
    path = build_canonical_networks(mapping, tmp_path, names=(networks.TAGC_NAME,))[networks.TAGC_NAME]
    assert pl.read_parquet(path).rows() == [('P1', 'P3'), ('P1|P3', 'P2')]
    audit = json.loads(path.with_suffix('.json').read_text())
    stages = audit['filtering_stages']
    assert [(s['nodes'], s['edges']) for s in stages] == [
        (8, 10), (8, 9), (8, 8), (8, 8), (8, 7), (7, 6), (4, 6), (4, 4), (4, 2), (4, 2)]
    assert stages[4]['edges_removed'] == 1
    assert stages[8]['edges_removed'] == 2
    assert stages[6]['nodes_removed'] == 3
    assert all(s['edges_removed'] >= 0 for s in stages)
    annotations = pl.read_parquet(audit['node_annotations']['path'])
    assert annotations.filter(pl.col('node') == 'P1|P3').row(0, named=True) == {
        'node': 'P1|P3', 'source_ids': ['A', 'D'], 'uniprot_accessions': ['P1', 'P3']}
    summary = summarize_network(load_network(path))
    assert summary['nodes'] == 4
    assert summary['mapped_accessions'] == 3
    assert summary['parent_accessions'] == 3
    table, missing = networks.filtering_audit_table({'TAGC': path})
    assert not missing and table.height == len(stages)
    source.write_text(source.read_text() + 'A,H\n')
    assert networks.filtering_audit_table({'TAGC': path})[1] == ['TAGC']


def test_duplicate_score_max_before_and_after_mapping_and_confidence_audit(tmp_path):
    mapping = fixture(tmp_path)
    source = networks.network_source_path(tmp_path, 'BioPlex')
    source.write_text('Uniprot A\tUniprot B\tpInt\nA\tB\tbad\nB\tA\t0.5\nC\tB\t0.95\nZ\tB\t0.99\n')
    mapping.update({'A': ['P1', 'P3'], 'C': ['P3', 'P1'], 'B': ['P2']})
    path = build_canonical_networks(mapping, tmp_path, names=('BioPlex',))['BioPlex']
    assert pl.read_parquet(path).rows() == [('P1|P3', 'P2', .95)]
    audit = json.loads(path.with_suffix('.json').read_text())
    assert audit['unmapped_ids'] == ['Z']
    assert audit['invalid_score_rows'] == 1
    version = save_scored_version('BioPlex', .95, tmp_path)
    stages = json.loads(version.with_suffix('.json').read_text())['filtering_stages']
    assert stages[-1]['stage'] == 'Inclusive pInt >= 0.95'
    assert stages[-1]['nodes'] == 2 and stages[-1]['edges'] == 1
    assert not networks.filtering_audit_table({'base': path, 'version': version})[1]
    source.write_text(source.read_text() + 'B\tC\t0.4\n')
    assert networks.filtering_audit_table({'base': path, 'version': version})[1] == ['base', 'version']


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
    (tmp_path / 'processed_networks/BioPlex_HEK293T_merged_dedup.tsv').write_text(
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
    assert score_column('HuRI') == 'author_score'
    with pytest.raises(ValueError, match='no pInt score column'):
        save_scored_version('BioPlex', .5, tmp_path)
    with pytest.raises(ValueError, match='no pInt score column'):
        threshold_curve(tmp_path, name='BioPlex')
    with pytest.raises(ValueError, match='no author_score score column'):
        save_scored_version('HuRI', .5, tmp_path)
    with pytest.raises(ValueError, match='no supported'):
        save_scored_version(networks.TAGC_NAME, .5, tmp_path)


def test_huri_author_score_versions_and_curve(tmp_path):
    source = networks.network_source_path(tmp_path, 'HuRI')
    source.parent.mkdir(parents=True, exist_ok=True)
    pad = '\t-' * 12
    source.write_text(
        f'uniprotkb:P11111-1\tuniprotkb:P22222{pad}\tauthor score: 0.94\n'
        f'uniprotkb:P22222\tuniprotkb:P11111-2{pad}\tauthor score: 0.72\n'
        f'uniprotkb:P11111\tuniprotkb:P33333{pad}\t-\n'
    )
    collapse_path = networks.prepare_huri_psi(tmp_path)
    collapse_df = pl.read_parquet(collapse_path)
    assert collapse_df.schema == {'a': pl.String, 'b': pl.String, 'author_score': pl.Float64}
    assert collapse_df.rows() == [('P11111', 'P22222', .94), ('P11111', 'P33333', None)]
    assert threshold_curve(tmp_path, [0., .5, .95], name='HuRI').to_dicts() == [
        {'threshold': 0., 'nodes': 2, 'edges': 1},
        {'threshold': .5, 'nodes': 2, 'edges': 1},
        {'threshold': .95, 'nodes': 0, 'edges': 0},
    ]
    version = save_scored_version('HuRI', .9, tmp_path, isoforms='collapse')
    assert pl.read_parquet(version).rows() == [('P11111', 'P22222', .94)]
    assert 'HuRI_0.9' in discover_datasets(tmp_path, isoforms='collapse')
    retain_version = save_scored_version('HuRI', .8, tmp_path, isoforms='retain')
    assert pl.read_parquet(retain_version).rows() == [('P11111-1', 'P22222', .94)]
    assert 'HuRI_0.8' in discover_datasets(tmp_path, isoforms='retain')


def test_bioplex_discovery_requires_correct_native_score_metadata(tmp_path):
    mapping = fixture(tmp_path)
    source = tmp_path / 'processed_networks/BioPlex_HEK293T_merged_dedup.tsv'
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


def test_network_mapping_audits_isoforms_fallback_and_tagc(tmp_path, monkeypatch):
    raw = tmp_path / 'raw_networks'
    raw.mkdir()
    (raw / networks.HURI_PSI).write_text(
        'uniprotkb:P11111-3\tuniprotkb:P22222-9\nensembl:ENSP00001.3\tuniprotkb:P11111-3\n')
    (raw / networks.FILES['TAGC_binary2025']).write_text(
        'Interactor,InteractionB\nP22222-9,P33333\nP33333,P22222-9\nP00000,P33333\n')
    (raw / networks.FILES['huMAP2']).write_text(
        'sp|P11111-3|EXAMPLE_HUMAN\tP22222-9\t0.95\n')
    calls = []

    def search_batch(ids, directory, *, namespace):
        calls.append((ids, namespace))
        matches = {'P22222': ['P55555'], 'P33333': ['P44444', 'P66666'], 'ENSP00001': ['P44444']}
        return {'mapping': {identifier: matches.get(identifier, []) for identifier in ids},
                'isoforms': {}, 'uniprot_releases': ['test-release']}

    monkeypatch.setattr(networks, '_current_uniprot_ids', lambda directory: ({'P11111-3'}, 'test-release'))
    monkeypatch.setattr(networks, '_search_uniprot_batch', search_batch)
    mapping = networks.prepare_mapping(tmp_path)
    assert calls == [(['P00000', 'P22222', 'P33333'], 'UniProtKB_AC-ID'),
                     (['ENSP00001'], 'Ensembl')]
    audit = json.loads((tmp_path / 'mappings' / networks.NETWORK_MAPPING).read_text())
    assert audit['entries']['P11111-3']['targets'] == ['P11111-3']
    assert audit['entries']['P11111-3']['status'] == 'Exact isoform'
    assert audit['entries']['P22222-9']['parent_query'] == 'P22222'
    assert audit['entries']['P22222-9']['status'] == 'Parent fallback'
    assert audit['entries']['P00000']['status'] == 'Unmapped'
    assert audit['sources']['HuRI']['ids'] == ['P11111-3', 'P22222-9', 'ensembl:ENSP00001.3']
    assert audit['entries']['ensembl:ENSP00001.3']['primary'] == ['P44444']
    assert audit['sources']['huMAP2']['isoform_rows'] == 1
    tagc = networks.build_canonical_networks(mapping, tmp_path, names=('TAGC_binary2025',))['TAGC_binary2025']
    assert tagc.name == 'TAGC_binary2025.parquet'
    assert pl.read_parquet(tagc).rows() == [('P44444|P66666', 'P55555')]
    assert networks.discover_datasets(tmp_path) == {'TAGC_binary2025': tagc}
    legacy = tagc.rename(tagc.with_name('TAGC.parquet'))
    assert networks.discover_datasets(tmp_path) == {'TAGC_binary2025': legacy}


def test_mapping_statistics_deduplicate_targets_and_retain_multiple_matches():
    rows = pl.DataFrame({
        'Network': ['A', 'A', 'B'],
        'targets': [['P1-2', 'P2'], ['P2', 'P3', 'P4'], []],
        'primary': [['P1', 'P2'], ['P2', 'P3', 'P4'], []],
        'reviewed': [['P1', 'P2'], ['P2', 'P3'], []],
    })
    assert networks.mapping_target_statistics(rows).rows() == [
        ('A', 2, 2, 2, 4, 4, 3),
        ('B', 1, 0, 0, 0, 0, 0),
    ]
