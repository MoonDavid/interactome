"""Protein metadata and functional-theme indexes for saved cluster reports."""
from pathlib import Path

import polars as pl

MOONLITDB_FUNCTIONS = {
    'moonlitdb_goslims_canonical': 'MoonLitDB canonical functions',
    'moonlitdb_goslims_moonlighting': 'MoonLitDB moonlighting functions',
}


def theme_index(clusters):
    """Index significant saved full-cluster terms without recomputing any tests."""
    rows = [{'rank': row['rank'], 'library': term['library'], 'term_id': term['term_id'],
             'term_name': term['term_name']}
            for row in clusters for term in row['terms'] if term['qvalue'] < .05]
    return pl.from_dicts(rows, schema={'rank': pl.Int64, 'library': pl.String,
                                     'term_id': pl.String, 'term_name': pl.String}).unique()


def protein_metadata(root):
    """Reuse the application's canonical MoonLitDB mapping and local FUNCTION snapshot."""
    from moonlighting import load_moon_data
    root = Path(root)
    data = load_moon_data(root).filter(pl.col('canonical_id') != '')
    metadata = (data.group_by('canonical_id').agg(
        pl.col('moonlitdb').any().alias('MoonLitDB'),
        pl.col('moonprot').any().alias('MoonProt'),
        pl.col('Gene Names').filter(pl.col('Gene Names') != '').unique().sort().str.join(' | ').alias('Gene names'),
        pl.col('Protein names').filter(pl.col('Protein names') != '').unique().sort().str.join(' | ').alias('Protein name'),
        *(pl.col(source).filter(pl.col('moonlitdb') & (pl.col(source).str.strip_chars() != ''))
          .unique().sort().str.join(' | ').alias(label)
          for source, label in MOONLITDB_FUNCTIONS.items()),
        *(pl.col(source).filter(pl.col('moonprot') & (pl.col(source).str.strip_chars() != ''))
          .unique().sort().str.join(' | ').alias(label)
          for source, label in [('moonprot_f1_desc', 'MoonProt primary function'),
                                ('moonprot_f2_desc', 'MoonProt secondary function')]))
        .rename({'canonical_id': 'UniProt'}))
    path = root / 'annotations/uniprot_functions.tsv'
    if path.is_file():
        functions = (pl.read_csv(path, separator='\t', infer_schema=False)
                     .select(pl.col('Entry').alias('UniProt'),
                             pl.col('Function [CC]').alias('Function description'))
                     .group_by('UniProt').agg(
                         pl.col('Function description').drop_nulls().unique().sort().str.join(' ')))
        metadata = metadata.join(functions, on='UniProt', how='full', coalesce=True)
    else:
        metadata = metadata.with_columns(pl.lit(None, dtype=pl.String).alias('Function description'))
    return metadata


def cluster_proteins(members, metadata, protein_set='MoonLitDB'):
    # Older saved/cached metadata may predate the MoonLitDB annotation columns.
    metadata = metadata.with_columns(
        pl.lit(None, dtype=pl.String).alias(column)
        for column in MOONLITDB_FUNCTIONS.values() if column not in metadata.columns)
    if 'MoonProt' not in metadata.columns:
        metadata = metadata.with_columns(pl.lit(False).alias('MoonProt'))
    return (pl.DataFrame({'UniProt': members}, schema={'UniProt': pl.String})
            .join(metadata, on='UniProt', how='left')
            .with_columns(pl.col('MoonLitDB', 'MoonProt').fill_null(False),
                          pl.col('Gene names', 'Protein name', 'Function description',
                                 *MOONLITDB_FUNCTIONS.values())
                          .fill_null('Unavailable').replace('', 'Unavailable'))
            .sort([protein_set, 'Gene names', 'UniProt'], descending=[True, False, False]))
