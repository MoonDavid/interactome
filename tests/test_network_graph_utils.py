import numpy as np
import polars as pl
from network_graph_utils import layout_3d, protein_metadata, protein_identity


def test_force_layout_is_seeded_3d_and_keeps_isolated_clusters():
    links = pl.DataFrame({'source': ['1'], 'target': ['2'], 'interactions': [4]})
    first = layout_3d(['1', '2', 'isolated'], links)
    second = layout_3d(['1', '2', 'isolated'], links)
    assert first == second
    assert all(len(value) == 3 and np.isfinite(value).all() for value in first.values())
    larger = layout_3d(list(map(str, range(2001))), links)
    assert len(larger) == 2001
    assert all(len(value) == 3 and np.isfinite(value).all() for value in larger.values())


def test_human_gene_names_and_isoform_metadata_preserve_original_ids(tmp_path):
    import gzip
    (tmp_path / 'GO').mkdir()
    def row(accession, symbol, name, taxon):
        fields = ['UniProtKB', accession, symbol, '', 'GO:1', 'PMID:1', 'IDA', '', 'P',
                  name, '', 'protein', taxon, '20260101', 'UniProt', '', '']
        return '\t'.join(fields) + '\n'
    with gzip.open(tmp_path / 'GO/HUMAN-uniprot.gaf.gz', 'wt', encoding='utf-8') as handle:
        handle.write('!gaf-version: 2.2\n' + row('P1', 'GENE1', 'Protein one', 'taxon:9606') +
                     row('MOUSE', 'MouseGene', 'Mouse protein', 'taxon:10090'))
    table = protein_metadata(tmp_path)
    lookup = {r['accession']: r for r in table.iter_rows(named=True)}
    assert set(lookup) == {'P1'}
    named = protein_identity('P1-2', lookup)
    assert named['label'] == 'GENE1'
    assert 'Protein one' in named['tooltip']
    assert 'UniProt: P1-2' in named['tooltip']
    assert 'parent-accession' in named['tooltip']
    assert protein_identity('UNKNOWN', lookup)['label'] == 'UNKNOWN'

