import polars as pl

from cluster_reports import cluster_proteins, theme_index


def test_theme_index_includes_secondary_terms_and_uses_adjusted_values():
    terms = [{'library': 'GO_BP', 'term_id': 'GO:1', 'term_name': 'secondary', 'qvalue': .01},
             {'library': 'Reactome', 'term_id': 'R:1', 'term_name': 'excluded', 'qvalue': .05}]
    index = theme_index([{'rank': 2, 'terms': terms}, {'rank': 3, 'terms': terms[:1]}])
    assert index.sort('rank')['rank'].to_list() == [2, 3]
    assert index['term_id'].unique().to_list() == ['GO:1']
    assert theme_index([]).is_empty()


def test_membership_join_preserves_unknown_members_and_full_descriptions():
    metadata = pl.DataFrame({'UniProt': ['P1', 'P2'], 'MoonLitDB': [True, False],
                             'Gene names': ['GENE1', 'GENE2'], 'Protein name': ['First', 'Second'],
                             'Function description': ['A full function description.', None],
                             'MoonLitDB canonical functions': ['Translation', ''],
                             'MoonLitDB moonlighting functions': ['Immune response', None]})
    proteins = cluster_proteins(['P1', 'P2', 'P3'], metadata)
    assert proteins.height == 3
    assert proteins.filter(pl.col('MoonLitDB'))['UniProt'].to_list() == ['P1']
    assert proteins.filter(pl.col('UniProt') == 'P1')['Function description'][0] == 'A full function description.'
    unknown = proteins.filter(pl.col('UniProt') == 'P3').row(0, named=True)
    assert unknown['Gene names'] == 'Unavailable'
    assert unknown['Function description'] == 'Unavailable'
    assert not unknown['MoonLitDB']
    assert unknown['MoonLitDB canonical functions'] == 'Unavailable'
    assert proteins.filter(pl.col('UniProt') == 'P1')['MoonLitDB moonlighting functions'][0] == 'Immune response'


def test_old_metadata_without_moonlitdb_function_columns():
    metadata = pl.DataFrame({'UniProt': ['P1'], 'MoonLitDB': [True],
                             'Gene names': ['GENE1'], 'Protein name': ['First'],
                             'Function description': ['Function']})
    row = cluster_proteins(['P1'], metadata).row(0, named=True)
    assert row['MoonLitDB canonical functions'] == 'Unavailable'
    assert row['MoonLitDB moonlighting functions'] == 'Unavailable'


def test_moonprot_membership_is_independent_of_moonlitdb():
    metadata = pl.DataFrame({'UniProt': ['P1', 'P2'], 'MoonLitDB': [True, False],
                             'MoonProt': [False, True], 'Gene names': ['G1', 'G2'],
                             'Protein name': ['First', 'Second'],
                             'Function description': ['Function1', 'Function2']})
    rows = cluster_proteins(['P1', 'P2'], metadata, 'MoonProt')
    assert rows['UniProt'][0] == 'P2'
    assert rows.filter(pl.col('MoonProt'))['UniProt'].to_list() == ['P2']
