"""Hierarchy votes must count proteins, preserve ties and include missing coverage."""
import polars as pl

from clusterannotation import majority_labels, label_summary, function_text_labels, hierarchy_sankey


def test_bundle_writer_reuses_existing_cache(tmp_path):
    from clusterannotation import write_annotation_bundle
    output = tmp_path / 'cached'
    output.mkdir()
    manifest = output / 'manifest.json'
    manifest.write_text('{"cached": true}')
    # No result is needed when a published cache is already available.
    write_annotation_bundle(output, {})
    assert manifest.read_text() == '{"cached": true}'


def test_bundle_writer_reuses_concurrently_published_cache(tmp_path, monkeypatch):
    from pathlib import Path
    from clusterannotation import write_annotation_bundle
    output = tmp_path / 'cached'
    frame = pl.DataFrame({'proteins': [['A']], 'label': ['saved']})
    result = dict(labels=frame.drop('proteins'), majority=frame,
                  summary=frame.drop('proteins'), text_labels=frame,
                  evidence=frame, manifest={'new': True})

    def publish_elsewhere_then_fail(source, destination):
        assert (source / 'manifest.json').is_file()
        destination.mkdir()
        (destination / 'manifest.json').write_text('{"winner": true}')
        raise FileExistsError(183, 'Already exists', str(destination))

    monkeypatch.setattr(Path, 'rename', publish_elsewhere_then_fail)
    write_annotation_bundle(output, result)
    assert (output / 'manifest.json').read_text() == '{"winner": true}'
    assert list(tmp_path.iterdir()) == [output]


def test_hierarchy_majority_uses_all_proteins_and_keeps_ties():
    terms = {
        'root': {'parents': []},
        'broad': {'parents': ['root']},
        'child': {'parents': ['broad']},
        'other': {'parents': ['broad']},
    }
    queries = pl.DataFrame({'protein': ['A', 'B', 'C', 'D'],
                            'community': ['1'] * 4, 'subset': ['full'] * 4})
    annotations = pl.DataFrame([
        {'protein': protein, 'library': 'GO_BP', 'term_id': term, 'term_name': term}
        for protein, term in [('A', 'root'), ('A', 'broad'), ('B', 'broad'),
                              ('C', 'broad'), ('A', 'child'), ('B', 'child'),
                              ('A', 'child'), ('B', 'other'), ('C', 'other')]
    ])
    result = majority_labels(queries, annotations, terms)
    assert 'root' not in result['term_id'].to_list()
    broad = result.filter(pl.col('term_id') == 'broad').row(0, named=True)
    assert broad['support_fraction'] == 0.75
    assert broad['is_majority']
    children = result.filter(pl.col('depth') == 2)
    assert children['term_id'].to_list() == ['child', 'other']
    assert children['query_hits'].to_list() == [2, 2]
    assert children['is_majority'].to_list() == [False, False]


def test_no_annotation_produces_typed_empty_votes():
    queries = pl.DataFrame({'protein': ['A'], 'community': ['1'], 'subset': ['full']})
    annotations = pl.DataFrame(schema={c: pl.String for c in
                                      ['protein', 'library', 'term_id', 'term_name']})
    result = majority_labels(queries, annotations, {'root': {'parents': []}})
    assert result.is_empty()
    assert result.schema['support_fraction'] == pl.Float64


def test_unequal_processes_both_appear_in_hierarchy_summary():
    queries = pl.DataFrame({'protein': [str(i) for i in range(20)],
                            'community': ['1'] * 20, 'subset': ['full'] * 20})
    terms = {'root': {'parents': []}, 'b1': {'parents': ['root']},
             'b2': {'parents': ['root']}}
    annotations = pl.DataFrame([
        {'protein': str(i), 'library': 'GO_BP', 'term_id': 'b1' if i < 8 else 'b2',
         'term_name': 'process b1' if i < 8 else 'process b2'} for i in range(20)])
    votes = majority_labels(queries, annotations, terms)
    assert votes['query_hits'].to_list() == [12, 8]
    labels = pl.DataFrame({'community': ['1'], 'full_size': [20]})
    evidence = pl.DataFrame(schema={'community': pl.String, 'subset': pl.String,
        'library': pl.String, 'qvalue': pl.Float64, 'pvalue': pl.Float64,
        'query_hits': pl.UInt32, 'term_id': pl.String})
    summary = label_summary(labels, votes, evidence)
    text = summary['GO_BP_hierarchy_labels'][0]
    assert 'process b1 [b1] (8/20 proteins, 40.0%)' in text
    assert 'process b2 [b2] (12/20 proteins, 60.0%)' in text
    assert summary['GO_BP_majority'][0] == 'process b2'


def test_function_text_labels_use_background_and_remove_references(tmp_path):
    path = tmp_path / 'functions.tsv'
    pl.DataFrame({'Entry': ['A', 'B', 'C', 'D'], 'Function [CC]': [
        'FUNCTION: DNA repair. {ECO:0000269|PubMed:123}',
        'FUNCTION: DNA repair.', 'FUNCTION: Lipid transport.', None,
    ]}).write_csv(path, separator='\t')
    population = pl.DataFrame({'protein': ['A', 'B', 'C', 'D', 'E']})
    queries = pl.DataFrame({'protein': ['A', 'B', 'E'], 'community': ['1'] * 3,
                            'subset': ['full'] * 3})
    evidence, manifest = function_text_labels(queries, population, path)
    assert manifest['annotated_background'] == 3
    assert set(evidence['method']) == {'tfidf', 'document_lift'}
    phrases = evidence.filter(pl.col('label') == 'dna repair')
    assert phrases.height == 2
    assert phrases['protein_hits'].to_list() == [2, 2]
    assert phrases['query_size'].to_list() == [3, 3]
    assert phrases['annotated_query'].to_list() == [2, 2]
    assert all(proteins == ['A', 'B'] for proteins in phrases['proteins'].to_list())
    assert not any('pubmed' in label or 'function' in label for label in evidence['label'])


def test_function_text_not_requested_is_explicit():
    evidence, manifest = function_text_labels(None, None, None)
    assert evidence.is_empty()
    assert manifest['status'] == 'not_requested'


def test_function_download_validates_total_before_writing(tmp_path, monkeypatch):
    import clusterannotation as module
    import pytest

    class Response:
        headers = {'X-Total-Results': '2', 'X-UniProt-Release': 'test'}
        text = 'Entry\tFunction [CC]\nA\tDNA repair\n'
        links = {}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(module.requests.Session, 'get', lambda *a, **k: Response())
    target = tmp_path / 'functions.tsv'
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        module.fetch_uniprot_functions(target)
    assert not target.exists()
    Response.headers['X-Total-Results'] = '1'
    module.fetch_uniprot_functions(target)
    assert pl.read_csv(target, separator='\t')['Entry'].to_list() == ['A']
    assert target.with_suffix('.json').is_file()


def test_sankey_counts_support_only_on_actual_ancestor_links():
    votes = pl.DataFrame([
        dict(community='1', subset='full', library='GO_BP', depth=depth,
             term_id=term, term_name=term, query_hits=len(proteins), query_size=3,
             support_fraction=len(proteins)/3, is_majority=True, proteins=proteins)
        for term, depth, proteins in [('parent', 2, ['A', 'B', 'C']),
            ('child', 3, ['A', 'B']), ('unrelated', 3, ['A', 'B']),
            ('leaf', 4, ['A', 'B'])]])
    evidence = pl.DataFrame({'community': ['1'], 'subset': ['full'], 'library': ['GO_BP'],
        'term_id': ['child'], 'qvalue': [0.001], 'fold_enrichment': [2.0]})
    ancestry = {'parent': {'parent'}, 'child': {'parent', 'child'},
                'unrelated': {'unrelated'}, 'leaf': {'parent', 'child', 'leaf'}}
    figure, counts, links = hierarchy_sankey(votes, evidence, lambda term: ancestry[term], '1')
    assert set(zip(links['source_term'], links['target_term'])) == {('parent', 'child'), ('child', 'leaf')}
    assert links['value'].to_list() == [2, 2]
    assert all(p == ['A', 'B'] for p in links['proteins'].to_list())
    assert counts.height == 4
    assert len(figure.data[0].node.label) == 4
    assert figure.data[0].textfont.color == '#111111'
    assert any('<b>2/3 proteins (67%)</b>' in label for label in figure.data[0].node.label)
    assert figure.layout.paper_bgcolor == '#ffffff'
    assert figure.layout.hoverlabel.font.color == '#111111'
    assert all(annotation.font.color == '#111111' for annotation in figure.layout.annotations)


def test_generif_human_mapping_abstracts_and_pseudodoc_scores(tmp_path):
    import json
    import math
    from clusterannotation import generif_documents
    genes = tmp_path / 'generif.tsv'
    genes.write_text('#Tax ID\tGene ID\tPubMed ID (PMID) list\tlast update timestamp\tGeneRIF text\n'
                     '9606\t1\t11, 12\t2026\tDNA repair complex.\n'
                     '9606\t2\t11\t2026\tDNA repair complex.\n'
                     '10090\t3\t13\t2026\tMouse artifact.\n')
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'taxon': 9606, 'mapping': {'1': ['A'], '2': ['B'], '3': ['C']}}))
    abstracts = tmp_path / 'abstracts.tsv'
    pl.DataFrame({'pmid': ['11'], 'abstract': ['Double strand repair.']}).write_csv(abstracts, separator='\t')
    docs, metadata = generif_documents(genes, mapping, abstracts)
    assert docs.height == 4
    assert set(docs['protein']) == {'A', 'B'}
    assert metadata['abstracts']
    functions = tmp_path / 'functions.tsv'
    pl.DataFrame({'protein': ['A', 'B', 'C'], 'function': ['DNA repair complex.', 'DNA repair complex.', 'Lipid transport.']}).write_csv(functions, separator='\t')
    population = pl.DataFrame({'protein': ['A', 'B', 'C']})
    queries = pl.DataFrame({'protein': ['A', 'B', 'C'], 'community': ['1', '1', '2'], 'subset': ['full'] * 3})
    result, manifest = function_text_labels(queries, population, functions, generif=genes, gene_mapping=mapping)
    phrase = result.filter((pl.col('method') == 'ctfidf') & (pl.col('label') == 'dna repair complex'))
    # Identical source statements are deduplicated per protein; average cluster length = 4.
    assert math.isclose(phrase['score'][0], 2 / 6 * math.log1p(4 / 2))
    assert 'log_odds' in result['method']
    selected, _ = function_text_labels(queries.filter(pl.col('community') == '1'), population, functions,
                                     generif=genes, gene_mapping=mapping, reference_queries=queries)
    assert selected.filter(pl.col('method') == 'ctfidf').equals(result.filter(pl.col('method') == 'ctfidf'))
    assert not any('mouse' in term for term in result['label'])
    assert manifest['prior_strength'] == 1000

