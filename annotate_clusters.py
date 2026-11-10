"""Annotate existing human UniProt clusters with GO and Reactome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import polars as pl
import requests
from scipy.stats import hypergeom

from moonlighting import _ontology, bh_adjust, prepare_go_annotations
from networks import _fingerprint, _write_json, _write_parquet, discover_datasets, load_network


DATA = Path(__file__).resolve().parent / 'data'
DEFAULT_GAF = DATA / 'GO' / 'HUMAN-uniprot.gaf.gz'
DEFAULT_OBO = DATA / 'GO' / 'go-basic.obo'
DEFAULT_REACTOME = DATA / 'reactome' / 'UniProt2Reactome.txt'
LIBRARIES = ('GO_BP', 'GO_MF', 'GO_CC', 'Reactome')
TERM = ['library', 'term_id', 'term_name']
QUERY = ['community', 'subset']
MIN_BACKGROUND_HITS = 5
MIN_LABEL_HITS = 2
MAX_QVALUE = 0.05
ANCESTOR_OVERLAP = 0.70
UNIPROT_ACCESSION = r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$'
AUTOANNOTATE_COMMAND = 'commands/autoannotate/annotate-clusterBoosted'


def protein_id(column: str = 'protein') -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strip_chars().str.to_uppercase().str.replace(r'-\d+$', '')


def membership_queries(memberships: pl.DataFrame, population: pl.DataFrame) -> pl.DataFrame:
    if not {'protein', 'community'} <= set(memberships.columns):
        raise ValueError('Memberships require protein and community columns')
    pairs = (
        memberships.select(protein_id(), pl.col('community').cast(pl.String).str.strip_chars())
        .filter(pl.col('community').is_not_null() & (pl.col('community') != ''))
        .unique()
    )
    if population.is_empty():
        raise ValueError('The network background contains no UniProt proteins')
    if pairs.is_empty():
        raise ValueError('No assigned cluster memberships were found')
    outside = pairs.join(population, on='protein', how='anti')['protein'].unique().sort()
    if len(outside):
        examples = ', '.join(p or '<missing>' for p in outside.head(5))
        raise ValueError(f'{len(outside)} assigned accession(s) outside the network background: {examples}')
    split = pairs.with_columns(
        pl.when(pl.len().over('protein') == 1).then(pl.lit('core'))
        .otherwise(pl.lit('interface')).alias('subset')
    )
    return pl.concat([pairs.with_columns(pl.lit('full').alias('subset')), split])


def load_annotations(go_path: Path, reactome_path: Path, population: pl.DataFrame) -> pl.DataFrame:
    go = pl.scan_parquet(go_path).select(
        'protein', pl.col('term').alias('term_id'), pl.col('name').alias('term_name'),
        pl.col('aspect').replace_strict({'P': 'GO_BP', 'F': 'GO_MF', 'C': 'GO_CC'}, default=None)
        .alias('library'),
    ).filter(pl.col('library').is_not_null())
    reactome = (
        pl.scan_csv(reactome_path, separator='\t', has_header=False, quote_char=None, infer_schema=False,
                    new_columns=['protein', 'term_id', 'url', 'term_name', 'evidence', 'species'])
        .filter((pl.col('species') == 'Homo sapiens') & pl.col('term_id').str.starts_with('R-HSA-'))
        .select(protein_id(), 'term_id', 'term_name', pl.lit('Reactome').alias('library'))
    )
    return (
        pl.concat([go.select('protein', *TERM), reactome.select('protein', *TERM)])
        .join(population.lazy(), on='protein', how='semi')
        .filter(pl.col('term_name').is_not_null() & (pl.col('term_name') != ''))
        # Use one deterministic name per identifier before counting distinct accessions.
        .with_columns(pl.col('term_name').min().over('library', 'term_id'))
        .unique()
        .collect()
    )


def enrichment(queries: pl.DataFrame, annotations: pl.DataFrame, population_size: int):
    sizes = queries.group_by(QUERY).agg(pl.len().alias('query_size'))
    background = annotations.group_by(TERM).agg(pl.len().alias('background_hits')).filter(
        pl.col('background_hits') >= MIN_BACKGROUND_HITS
    )
    hits = (
        queries.join(annotations.join(background, on=TERM, how='semi'), on='protein')
        .group_by([*QUERY, *TERM]).agg(
            pl.len().alias('query_hits'), pl.col('protein').sort().alias('proteins'),
        )
        .join(sizes, on=QUERY, validate='m:1')
        .join(background, on=TERM, validate='m:1')
        .with_columns(
            pl.lit(population_size, dtype=pl.UInt32).alias('population_size'),
            (pl.col('query_hits') / pl.col('query_size') /
             (pl.col('background_hits') / population_size)).alias('fold_enrichment'),
        )
    )
    pvalues = hypergeom.sf(
        hits['query_hits'].to_numpy().astype(np.int64) - 1, population_size,
        hits['background_hits'].to_numpy().astype(np.int64),
        hits['query_size'].to_numpy().astype(np.int64),
    )
    total_tests = sizes.height * background.height
    # Zero-hit tests all have p=1. Include them in the BH denominator without
    # materializing the cluster-by-term cross product.
    adjusted = (np.minimum(1, bh_adjust(pvalues) * (total_tests / len(pvalues)))
                if len(pvalues) else pvalues)
    return (
        hits.with_columns(pl.Series('pvalue', pvalues), pl.Series('qvalue', adjusted))
        .sort([*QUERY, 'qvalue', 'pvalue', 'library', 'term_id']),
        sizes, background, total_tests,
    )


def select_labels(evidence: pl.DataFrame, sizes: pl.DataFrame, ancestors) -> pl.DataFrame:
    candidates = (
        evidence.filter((pl.col('qvalue') <= MAX_QVALUE) & (pl.col('query_hits') >= MIN_LABEL_HITS))
        .with_columns(
            (-pl.col('qvalue').clip(lower_bound=np.finfo(float).tiny).log10()
             * (pl.col('population_size') / pl.col('background_hits')).log()).alias('score')
        )
        .sort(['score', 'qvalue', 'term_id'], descending=[True, False, False])
    )
    winners = {}
    for key, group in candidates.group_by(QUERY):
        rows = group.to_dicts()
        go_hits = {row['term_id']: set(row['proteins']) for row in rows if row['library'] != 'Reactome'}
        suppressed = set()
        for child, child_hits in go_hits.items():
            for parent in (ancestors(child) & go_hits.keys()) - {child}:
                parent_hits = go_hits[parent]
                if len(parent_hits & child_hits) / len(parent_hits | child_hits) > ANCESTOR_OVERLAP:
                    suppressed.add(parent)
        winners[key] = next(row for row in rows if row['term_id'] not in suppressed)

    size_by_query = {(c, s): n for c, s, n in sizes.iter_rows()}
    labels = []
    for community in sorted(sizes['community'].unique()):
        row = {'community': community}
        for subset, prefix in [('core', 'primary'), ('interface', 'interface'), ('full', 'full_context')]:
            size = size_by_query.get((community, subset), 0)
            term = winners.get((community, subset))
            if subset == 'full' and (community, 'core') in winners:
                term, status = None, 'core_labeled'
            else:
                status = 'labeled' if term else 'no_qualifying_term' if size else 'empty_subset'
            row[f'{subset}_size'] = size
            row[f'{prefix}_status'] = status
            for key, suffix in [('term_name', 'label'), ('term_id', 'term_id'), ('library', 'library')]:
                row[f'{prefix}_{suffix}'] = term[key] if term else None
        labels.append(row)
    return pl.from_dicts(labels, schema_overrides={
        f'{prefix}_{suffix}': pl.String
        for prefix in ('primary', 'interface', 'full_context')
        for suffix in ('label', 'term_id', 'library')
    })


def network_population(path: Path) -> pl.DataFrame:
    edges = load_network(path).edges
    return (
        pl.concat([edges.select(protein_id(col).alias('protein')) for col in ('a', 'b')])
        .filter(pl.col('protein') != '').unique().sort('protein')
    )


def file_input(memberships: Path, network: str):
    path = Path(network)
    if not path.is_file():
        path = discover_datasets().get(network)
        if path is None:
            raise ValueError(f'Unknown network {network!r}; use a discovered name or canonical Parquet path')
    if path.suffix.lower() != '.parquet':
        raise ValueError('The background must be a canonical or saved-version Parquet file')
    network_source = _fingerprint(path.resolve())
    source = {'network': network_source, 'memberships': _fingerprint(memberships.resolve())}
    if path.with_suffix('.json').is_file():
        source['network_audit'] = _fingerprint(path.with_suffix('.json').resolve())
    result_path = memberships.with_name('result.json')
    if memberships.name == 'memberships.parquet' and result_path.is_file():
        job = json.loads(result_path.read_text(encoding='utf-8'))
        expected = job.get('job', {}).get('provenance', {}).get('input', {}).get('sha256')
        if job.get('status') != 'success' or (expected and expected != network_source['sha256']):
            raise ValueError('The clustering run did not succeed on the supplied network version')
        source['clustering_result'] = _fingerprint(result_path.resolve())
    population = network_population(path)
    if memberships.suffix.lower() == '.csv':
        frame = pl.read_csv(memberships, columns=['protein', 'community'], infer_schema=False)
    elif memberships.suffix.lower() == '.parquet':
        frame = pl.read_parquet(memberships, columns=['protein', 'community'])
    else:
        raise ValueError('Memberships must be CSV or Parquet')
    return frame, population, source


def cyrest(base_url: str, method: str, path: str, body=None):
    headers = {'Accept': 'application/json'} if path == AUTOANNOTATE_COMMAND else None
    response = requests.request(
        method, f'{base_url.rstrip("/")}/{path}', json=body,
        headers=headers, timeout=(5, 120),
    )
    response.raise_for_status()
    if 'json' in response.headers.get('Content-Type', ''):
        return response.json()
    return response.text


def annotation_node_table(network: int, base_url: str) -> list[dict]:
    # Read the raw rows because the general bridge flattens list-valued cells.
    rows = cyrest(base_url, 'GET', f'networks/{network}/tables/defaultnode/rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Cytoscape network {network} returned no node rows')
    return rows


def cytoscape_proteins(rows: list[dict], column: str) -> pl.DataFrame:
    if not any(column in row for row in rows):
        raise ValueError(f'Cytoscape UniProt column {column!r} was not found')
    return pl.DataFrame({
        'SUID': [row['SUID'] for row in rows],
        'protein': [row.get(column) for row in rows],
    }, schema={'SUID': pl.Int64, 'protein': pl.String}).with_columns(protein_id())


def snapshot_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def annotation_input(rows, network, uniprot_column, cluster_column, separator, base_url):
    if not any(cluster_column in row for row in rows):
        raise ValueError(f'Cytoscape cluster column {cluster_column!r} was not found')
    nodes = cytoscape_proteins(rows, uniprot_column)
    population = nodes.filter(pl.col('protein').str.contains(UNIPROT_ACCESSION)).select('protein').unique().sort('protein')
    clusters = []
    for row in rows:
        value = row.get(cluster_column)
        if isinstance(value, list):
            clusters.append([str(item) for item in value if item is not None])
        elif value is None:
            clusters.append([])
        else:
            clusters.append(str(value).split(separator) if separator else [str(value)])
    pairs = (
        nodes.with_columns(pl.Series('community', clusters, dtype=pl.List(pl.String)))
        .explode('community').select('protein', 'community')
    )
    source = {'cytoscape': {
        'network_suid': network, 'base_url': base_url,
        'uniprot_column': uniprot_column, 'cluster_column': cluster_column,
        'cluster_separator': separator, 'node_rows': nodes.height,
        'rows_without_valid_accession': nodes.filter(
            ~pl.col('protein').str.contains(UNIPROT_ACCESSION).fill_null(False)
        ).height,
        'background_sha256': snapshot_sha256(population['protein'].to_list()),
        'memberships_sha256': snapshot_sha256(pairs.unique().sort('protein', 'community').rows()),
    }}
    return pairs, population, source


def autoannotate_labels(base_url, network, cluster_column, text_column, suid_to_protein, expected):
    reply = cyrest(base_url, 'POST', AUTOANNOTATE_COMMAND, {
        'network': f'SUID:{network}', 'useClusterMaker': 'false', 'clusterIdColumn': cluster_column,
        'labelColumn': text_column, 'returnJsonOnly': 'true', 'createSingletonClusters': 'false',
    })
    if not isinstance(reply, dict) or reply.get('errors') or not isinstance(reply.get('data'), list):
        raise ValueError(f'AutoAnnotate did not return clusters: {reply!r}')
    matched = {}
    mismatches = 0
    for item in reply['data']:
        members = frozenset(suid_to_protein.get(int(node)) for node in item['nodes'])
        if members not in expected:
            mismatches += 1
            continue
        matched.setdefault(members, set()).add(str(item.get('label') or '').strip())
    return matched, mismatches


def has_text(value) -> bool:
    if isinstance(value, list):
        return any(has_text(item) for item in value)
    return isinstance(value, str) and bool(value.strip())


def annotation_comparison(full, population, term_text, network, uniprot_column, description_column, base_url):
    grouped = full.group_by('community').agg(pl.col('protein').sort()).sort('community')
    expected = {community: frozenset(proteins) for community, proteins in grouped.iter_rows()}
    expected_sets = set(expected.values())
    passes = ('description', 'terms')
    compared = {community: {'community': community} for community in expected}
    for row in compared.values():
        for name in passes:
            row[f'autoannotate_{name}_label'] = None
            row[f'autoannotate_{name}_status'] = 'unavailable'
            row[f'autoannotate_{name}_text_proteins'] = None
    summary = {
        'status': 'unavailable', 'network_suid': network, 'base_url': base_url,
        'uniprot_column': uniprot_column, 'description_column': description_column,
        'command': AUTOANNOTATE_COMMAND, 'passes': {},
    }
    table = f'networks/{network}/tables/defaultnode'
    prefix = f'atlas_annotation_{uuid4().hex}'
    cluster_column, term_column = f'{prefix}_clusters', f'{prefix}_terms'
    created = []
    try:
        rows = annotation_node_table(network, base_url)
        nodes = cytoscape_proteins(rows, uniprot_column).join(population, on='protein', how='semi')
        missing = full.select('protein').unique().join(nodes, on='protein', how='anti')
        if missing.height:
            examples = ', '.join(missing['protein'].sort().head(5))
            raise ValueError(f'Cytoscape network is missing {missing.height} assigned accessions: {examples}')
        cyrest(base_url, 'GET', 'commands/autoannotate')
        suid_to_protein = dict(nodes.iter_rows())
        descriptions = {row['SUID']: row.get(description_column) for row in rows}
        summary['input_sha256'] = snapshot_sha256(sorted(
            (suid, protein, descriptions[suid]) for suid, protein in suid_to_protein.items()
        ))
        payload = (
            nodes.join(full.group_by('protein').agg(pl.col('community').sort().alias(cluster_column)),
                       on='protein', how='inner', validate='m:1')
            .join(term_text.rename({'term_text': term_column}), on='protein', how='left', validate='m:1')
            .drop('protein').sort('SUID')
        )
        for column, is_list in [(cluster_column, True), (term_column, False)]:
            cyrest(base_url, 'POST', f'{table}/columns', {'name': column, 'type': 'String', 'list': is_list})
            created.append(column)
        cyrest(base_url, 'PUT', table, {'key': 'SUID', 'dataKey': 'SUID', 'data': payload.to_dicts()})
        described = {protein for suid, protein in suid_to_protein.items() if has_text(descriptions[suid])}
        for name, column, text_proteins in [
            ('description', description_column, described),
            ('terms', term_column, set(term_text['protein'])),
        ]:
            for community, members in expected.items():
                count = len(members & text_proteins)
                compared[community][f'autoannotate_{name}_text_proteins'] = count
                if not count:
                    compared[community][f'autoannotate_{name}_status'] = 'missing_text'
            if not column or not text_proteins.intersection(full['protein']):
                summary['passes'][name] = {'status': 'missing_text'}
                continue
            try:
                matched, mismatches = autoannotate_labels(
                    base_url, network, cluster_column, column, suid_to_protein, expected_sets,
                )
                summary['passes'][name] = {'status': 'returned', 'unexpected_member_sets': mismatches}
                for community, members in expected.items():
                    row = compared[community]
                    if row[f'autoannotate_{name}_status'] == 'missing_text':
                        continue
                    labels = matched.get(members, set())
                    if len(labels) == 1:
                        label = next(iter(labels))
                        row[f'autoannotate_{name}_label'] = label or None
                        row[f'autoannotate_{name}_status'] = 'labeled' if label else 'empty_label'
                    else:
                        row[f'autoannotate_{name}_status'] = 'membership_mismatch'
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                summary['passes'][name] = {'status': 'unavailable', 'error': str(exc)}
        statuses = [row[f'autoannotate_{name}_status'] for row in compared.values() for name in passes]
        unexpected = any(result.get('unexpected_member_sets') for result in summary['passes'].values())
        summary['status'] = ('complete' if all(s == 'labeled' for s in statuses) and not unexpected
                             else 'partial' if 'labeled' in statuses else 'unavailable')
    except (requests.RequestException, KeyError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        summary['error'] = str(exc)
    finally:
        for column in reversed(created):
            try:
                cyrest(base_url, 'DELETE', f'{table}/columns/{column}')
            except (requests.RequestException, ValueError) as exc:
                summary.setdefault('cleanup_errors', []).append(str(exc))
    return pl.from_dicts(list(compared.values()), schema_overrides={
        **{f'autoannotate_{name}_label': pl.String for name in passes},
        **{f'autoannotate_{name}_text_proteins': pl.Int64 for name in passes},
    }), summary


def annotate_memberships(frame, population, source, *, gaf=DEFAULT_GAF, obo=DEFAULT_OBO,
                         reactome=DEFAULT_REACTOME, cytoscape=None):
    """Compute one annotation result for the CLI or the Streamlit app."""
    queries = membership_queries(frame, population)
    full = queries.filter(pl.col('subset') == 'full').select('community', 'protein')
    assigned = full.select('protein').unique()

    with TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        go_path = prepare_go_annotations(scratch, gaf, obo)
        go_provenance = json.loads(go_path.with_suffix('.json').read_text(encoding='utf-8'))
        annotations = load_annotations(go_path, reactome, population)
        evidence, sizes, background, tests = enrichment(queries, annotations, population.height)
        _, _, _, ancestors = _ontology(obo)
        labels = select_labels(evidence, sizes, ancestors)
        comparison = {'status': 'not_requested'}
        if cytoscape:
            terms_by_protein = (
                annotations.group_by('protein')
                .agg(pl.col('term_name').unique().sort().str.join(' | ').alias('term_text'))
            )
            compared, comparison = annotation_comparison(
                full, population, terms_by_protein, **cytoscape,
            )
            labels = labels.join(compared, on='community', validate='1:1', maintain_order='left')
        coverage = {}
        for library in LIBRARIES:
            proteins = annotations.filter(pl.col('library') == library).select('protein').unique()
            coverage[library] = {
                'annotated_background': proteins.height,
                'annotated_assigned': proteins.join(assigned, on='protein', how='semi').height,
                'eligible_terms': background.filter(pl.col('library') == library).height,
            }
        manifest = {
            'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'sources': {**source, 'go': go_provenance, 'reactome': _fingerprint(reactome.resolve())},
            'parameters': {
                'libraries': LIBRARIES, 'min_background_hits': MIN_BACKGROUND_HITS,
                'min_label_hits': MIN_LABEL_HITS, 'max_qvalue': MAX_QVALUE,
                'ancestor_hit_jaccard_strictly_above': ANCESTOR_OVERLAP,
                'test': 'one-sided hypergeometric; one BH family across nonempty subsets and libraries',
                'reactome_policy': 'human R-HSA pathways in the supplied lowest-level mapping',
            },
            'coverage': {
                'background': population.height, 'assigned': assigned.height,
                'unassigned': population.height - assigned.height, 'libraries': coverage,
            },
            'clusters': labels.height, 'tested_subsets': sizes.height,
            'memberships_sha256': snapshot_sha256(full.sort('protein', 'community').rows()),
            'tests': tests, 'zero_hit_tests_omitted_from_table': tests - evidence.height,
            'cytoscape_comparison': comparison,
        }
    return {'labels': labels, 'evidence': evidence, 'manifest': manifest, 'memberships': full}


def write_annotation_bundle(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        result['labels'].write_csv(bundle / 'cluster_labels.csv')
        _write_parquet(bundle / 'enrichment.parquet', result['evidence'])
        _write_json(bundle / 'manifest.json', result['manifest'])
        bundle.rename(output)


def run(args):
    base_url = args.cyrest_url
    if not base_url and (args.cytoscape is not None or args.compare_cytoscape is not None):
        from cytoscape_bridge import DEFAULT_BASE_URL
        base_url = DEFAULT_BASE_URL
    if args.output.exists():
        raise ValueError(f'Output already exists: {args.output}')
    if args.memberships:
        frame, population, source = file_input(args.memberships, args.network)
    else:
        nodes = annotation_node_table(args.cytoscape, base_url)
        frame, population, source = annotation_input(
            nodes, args.cytoscape, args.uniprot_column, args.cluster_column,
            args.cluster_separator, base_url,
        )
    cytoscape = None
    if args.compare_cytoscape is not None:
        cytoscape = {
            'network': args.compare_cytoscape, 'uniprot_column': args.uniprot_column,
            'description_column': args.description_column, 'base_url': base_url,
        }
    result = annotate_memberships(
        frame, population, source, gaf=args.gaf, obo=args.obo, reactome=args.reactome, cytoscape=cytoscape,
    )
    write_annotation_bundle(args.output, result)
    return result['labels'].height, result['manifest']['cytoscape_comparison']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--memberships', type=Path, help='Long-form CSV or Parquet memberships')
    inputs.add_argument('--cytoscape', type=int, metavar='SUID', help='Read clusters from Cytoscape Desktop')
    parser.add_argument('--network', help='Discovered network name or canonical Parquet path for file input')
    parser.add_argument('--uniprot-column', help='Node column containing UniProt accessions')
    parser.add_argument('--cluster-column', help='Node column containing existing cluster assignments')
    parser.add_argument('--cluster-separator', help='Separator for multiple IDs in a scalar cluster column')
    parser.add_argument('--compare-cytoscape', type=int, metavar='SUID', help='Compare with AutoAnnotate on this network')
    parser.add_argument('--description-column', help='Cytoscape text column for the protein-description pass')
    parser.add_argument('--cyrest-url', help='Override the local Cytoscape CyREST URL')
    parser.add_argument('--gaf', type=Path, default=DEFAULT_GAF)
    parser.add_argument('--obo', type=Path, default=DEFAULT_OBO)
    parser.add_argument('--reactome', type=Path, default=DEFAULT_REACTOME)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the result bundle')
    args = parser.parse_args(argv)
    if args.memberships and not args.network:
        parser.error('--memberships requires --network')
    if args.cytoscape is not None and (args.network or not args.cluster_column):
        parser.error('--cytoscape uses its own network background and requires --cluster-column')
    if (args.cytoscape is not None or args.compare_cytoscape is not None) and not args.uniprot_column:
        parser.error('Cytoscape input or comparison requires --uniprot-column')
    if args.cluster_separator == '':
        parser.error('--cluster-separator must be nonempty')
    try:
        count, comparison = run(args)
    except Exception as exc:
        parser.exit(1, f'interactome-annotate: {exc}\n')
    print(f'Annotated {count} clusters. Results saved to {args.output}')
    if comparison['status'] != 'not_requested':
        print(f'Cytoscape comparison: {comparison["status"]}. See manifest.json for details.')


if __name__ == '__main__':
    main()
