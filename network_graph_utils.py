"""Shared graph layouts, colours and local protein identities."""
import colorsys
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl

UNASSIGNED = 'Unassigned'


def layout_3d(nodes, edges, seed=42):
    """A seeded force layout of the displayed topology, independent of protein sets."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from((a, b, float(np.log1p(count))) for a, b, count in edges.select('source', 'target', 'interactions').iter_rows())
    components = sorted(nx.connected_components(graph), key=lambda group: (-len(group), sorted(group)))
    positions = {}
    for index, component in enumerate(components):
        local = nx.spring_layout(graph.subgraph(component), dim=3, seed=int(seed), iterations=80, weight='weight')
        if len(components) == 1:
            center, radius = np.zeros(3), 1.
        elif index == 0 and len(component) > 1:
            center, radius = np.zeros(3), .8
        else:
            # Pack disconnected components separately so isolates cannot shrink
            # the connected interactome into an unreadable speck at the center.
            t = (index + .5) / len(components)
            y = 1 - 2 * t
            angle = index * np.pi * (3 - np.sqrt(5))
            center = 1.15 * np.asarray([np.sqrt(1-y*y)*np.cos(angle), y, np.sqrt(1-y*y)*np.sin(angle)])
            radius = max(.025, .25 * np.sqrt(len(component) / max(1, len(components[0]))))
        for node, point in local.items():
            positions[str(node)] = [float(value) for value in center + radius * point]
    return positions


def cluster_colors(communities):
    colors = {}
    for index, community in enumerate(sorted(communities)):
        rgb = colorsys.hsv_to_rgb((index * .61803398875) % 1, .65, .8)
        colors[community] = '#' + ''.join(f'{round(value * 255):02x}' for value in rgb)
    colors[UNASSIGNED] = '#999999'
    return colors


def protein_metadata(data_root):
    """Local human GAF names, with moonlighting/BioPlex names as fallbacks."""
    root = Path(data_root)
    frames = []
    gaf = root / 'GO' / 'HUMAN-uniprot.gaf.gz'
    if gaf.is_file():
        raw = pl.read_csv(gaf, separator='\t', comment_prefix='!', has_header=False,
                          infer_schema=False, columns=[0, 1, 2, 9, 12])
        raw = raw.rename(dict(zip(raw.columns, ['database', 'accession', 'gene_name', 'protein_name', 'taxon'])))
        frames.append(raw.filter((pl.col('database') == 'UniProtKB') &
                      (pl.col('taxon').str.split('|').list.first() == 'taxon:9606'))
                      .select('accession', 'gene_name', 'protein_name').unique().with_columns(pl.lit(0).alias('priority')))
    moon = root / 'allmoon_with_moonlitdb_human.csv'
    if moon.is_file():
        raw = pl.read_csv(moon, infer_schema=False, columns=['uniprot', 'Gene Names', 'Protein names'])
        frames.append(raw.rename({'uniprot': 'accession', 'Gene Names': 'gene_name', 'Protein names': 'protein_name'})
                      .with_columns(pl.lit(1).alias('priority')))
    bioplex = root / 'raw_networks' / 'bioplex' / 'BioPlex_HEK293T_v3.0.tsv'
    if bioplex.is_file():
        raw = pl.read_csv(bioplex, separator='\t', infer_schema=False, columns=['UniprotA', 'UniprotB', 'SymbolA', 'SymbolB'])
        for endpoint in ('A', 'B'):
            frames.append(raw.select(pl.col(f'Uniprot{endpoint}').alias('accession'),
                                      pl.col(f'Symbol{endpoint}').alias('gene_name'))
                          .with_columns(pl.lit('').alias('protein_name'), pl.lit(2).alias('priority')))
    if not frames:
        return pl.DataFrame(schema={'accession': pl.String, 'gene_name': pl.String, 'protein_name': pl.String})
    return (pl.concat([f.select('accession', 'gene_name', 'protein_name', 'priority') for f in frames])
            .fill_null('').sort('priority').group_by('accession', maintain_order=True)
            .agg(pl.col('gene_name').filter(pl.col('gene_name').str.strip_chars() != '').first(),
                 pl.col('protein_name').filter(pl.col('protein_name').str.strip_chars() != '').first())
            .fill_null(''))


def protein_identity(protein, metadata):
    """Preserve exact node IDs; parent metadata may describe a retained isoform."""
    genes, names, parent_used = [], [], False
    for accession in protein.split('|'):
        row = metadata.get(accession)
        if row is None and '-' in accession:
            row = metadata.get(accession.rsplit('-', 1)[0])
            parent_used = parent_used or row is not None
        if row:
            if row.get('gene_name'):
                genes.append(row['gene_name'])
            if row.get('protein_name'):
                names.append(row['protein_name'])
    gene = ' / '.join(dict.fromkeys(genes))
    name = ' / '.join(dict.fromkeys(names))
    tooltip = f"Gene: {gene or 'unavailable'}\nProtein: {name or 'unavailable'}\nUniProt: {protein}"
    if parent_used:
        tooltip += '\nNames from parent-accession metadata'
    return {'label': gene or name or protein, 'tooltip': tooltip}

