"""Canonical UniProt networks, ID mapping, topology summaries, threshold versions, and overlap plots."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
import re
import tempfile
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from urllib3.util.retry import Retry
from pipelines.bioplex_merge import BIOPLEX_HEK293T_RELEASES

if TYPE_CHECKING:
    from matplotlib.figure import Figure

API = 'https://rest.uniprot.org'
POLICY = (
    'UniProt primary accessions; remove terminal numeric isoform suffix; '
    'one node per sorted UniProt label set (pipe-separated); merge identical label sets; '
    'deduplicate undirected source pairs before mapping; remove source loops before '
    'mapping and canonical loops after mapping; duplicate native score=max; '
    'BioPlex invalid scores=null with edges retained'
)
TAGC_NAME = 'TAGC_binary2025'
FILES = {
    'HuRI': 'HuRI.psi',
    'BioPlex': 'BioPlex_HEK293T_merged_dedup.tsv',
    'huMAP2': 'humap2_ppis_ACC_20200821.pairsWprob.gz',
    TAGC_NAME: 'TAGC_binaryInteractome2025.csv',
}
TAGC_COLUMNS = ['Interactor', 'InteractionB']
NETWORK_MAPPING = 'network_uniprot_mapping.json'
SCORE_COLUMNS = {'huMAP2': 'probability', 'BioPlex': 'pInt', 'HuRI': 'author_score'}
HURI_PSI = FILES['HuRI']
BIOPLEX_ORIGINAL = 'BioPlex HEK293T releases'
ISOFORM_POLICIES = {
    'collapse': 'Collapse to parent accessions',
    'canonical_only': 'Unsuffixed accessions only',
    'retain': 'Keep isoforms distinct',
}
HURI_ISOFORM_POLICIES = ISOFORM_POLICIES
ISOFORM_SUFFIX = r'-\d+$'
SOURCE_ISOFORM = r'^(?:[A-Z0-9]+-\d+$|(?:sp|tr)\|[A-Z0-9]+-\d+\|)'
PSI_UNIPROT = r'(?:^|\|)uniprotkb:([A-Z0-9]+(?:-\d+)?)(?:\||$)'
PSI_AUTHOR_SCORE = r'(?:^|\|)author score:\s*([^|]+)'
UNIPROT_ACCESSION = r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})'
SOURCE_ACCESSION = rf'^{UNIPROT_ACCESSION}(?:-\d+)?$'
UNIPROT_CACHE_SECONDS = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Official UniProt ID Mapping & Swiss-Prot Filtering
# ---------------------------------------------------------------------------

def map_ids(ids, source, cache_dir=Path('data/mappings'), existing_jobs=None):
    ids = sorted(set(ids))
    signature = hashlib.sha256(json.dumps([source, ids]).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{source}_{signature[:16]}.json'
    if path.exists():
        return json.loads(path.read_text())
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    mapping = {identifier: [] for identifier in ids}
    jobs = []
    batches = [ids[start:start + 10000] for start in range(0, len(ids), 10000)]
    if existing_jobs:
        batches = [ids for _ in existing_jobs]
    for batch_index, batch in enumerate(batches):
        checkpoint = cache_dir / f'{source}_{signature[:16]}_batch{batch_index}.job.json'
        if existing_jobs:
            job = existing_jobs[batch_index]
        elif checkpoint.exists():
            job = json.loads(checkpoint.read_text())['jobId']
        else:
            response = session.post(
                f'{API}/idmapping/run',
                data={'from': source, 'to': 'UniProtKB', 'ids': ','.join(batch)},
                timeout=60,
            )
            response.raise_for_status()
            job = response.json()['jobId']
            checkpoint.write_text(json.dumps({'jobId': job}))
        jobs.append(job)
        print(f'{source}: submitted {len(batch)} identifiers, job {job}', flush=True)
        deadline = time.monotonic() + 600
        while True:
            response = session.get(f'{API}/idmapping/status/{job}', timeout=60)
            response.raise_for_status()
            status = response.json()
            if status.get('jobStatus') not in ('RUNNING', 'NEW'):
                if status.get('jobStatus'):
                    raise RuntimeError(f'UniProt job failed: {status}')
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f'UniProt job timed out: {job}')
            time.sleep(3)
        url = f'{API}/idmapping/uniprotkb/results/{job}?format=json&size=500&fields=accession,organism_id'
        while url:
            response = session.get(url, timeout=90)
            response.raise_for_status()
            for row in response.json().get('results', []):
                target = row['to']
                if not isinstance(target, dict) or target.get('organism', {}).get('taxonId') != 9606:
                    continue
                accession = target['primaryAccession'].split('-')[0]
                if row['from'] not in mapping:
                    continue
                if accession not in mapping[row['from']]:
                    mapping[row['from']].append(accession)
            url = response.links.get('next', {}).get('url')
        checkpoint.unlink(missing_ok=True)
    payload = {
        'source': source, 'target': 'UniProtKB primary accession', 'taxon': 9606,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
        'input_sha256': signature, 'mapping': mapping,
        'unmapped': [k for k, v in mapping.items() if not v],
        'one_to_many': {k: v for k, v in mapping.items() if len(v) > 1},
    }
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return payload


def normalize_identifier(identifier, *, retain_isoforms=False):
    """Return the mapping namespace and normalized query ID for mixed endpoints."""
    ensembl_id = identifier.removeprefix('ensembl:')
    if ensembl_id.startswith(('ENSG', 'ENSP', 'ENST')):
        return 'Ensembl', ensembl_id.split('.')[0]
    if identifier.isdigit():
        return 'GeneID', identifier
    if '|' in identifier:
        fields = identifier.split('|')
        if len(fields) >= 3 and fields[0] in ('sp', 'tr'):
            identifier = fields[1]
    return 'UniProtKB_AC-ID', identifier if retain_isoforms else re.sub(ISOFORM_SUFFIX, '', identifier)


def scan_humap_source(path: Path) -> pl.LazyFrame:
    """The headerless huMAP2 source has two IDs and a native probability."""
    return pl.scan_csv(path, separator='\t', has_header=False, infer_schema=False,
                       new_columns=['a', 'b', 'probability'])


def source_endpoint_profile(frame: pl.LazyFrame) -> tuple[list[str], int, int]:
    """Collect distinct endpoints and counts without materializing raw edges."""
    a, b, rows, isoform_rows = frame.select(
        pl.col('a', 'b').str.strip_chars().drop_nulls().unique().implode(),
        pl.len(),
        pl.any_horizontal(pl.col('a', 'b').str.strip_chars().str.contains(SOURCE_ISOFORM))
        .fill_null(False).sum().alias('isoform_rows'),
    ).collect(engine='streaming').row(0)
    return sorted((set(a) | set(b)) - {''}), rows, isoform_rows


def source_identifier_counts(ids: pl.Series) -> pl.DataFrame:
    """Count distinct source endpoint identifiers by format, without a UniProt lookup."""
    identifier = pl.col('id')
    accession = identifier.str.replace(r'^uniprotkb:', '').str.replace(r'^(?:sp|tr)\|([^|]+)\|.*$', '${1}')
    kind = (
        pl.when(accession.str.contains(SOURCE_ACCESSION)).then(pl.lit('UniProt accession'))
        .when(identifier.str.contains(r'^(?:ensembl:)?ENSG\d+(?:\.\d+)?$')).then(pl.lit('Ensembl gene ID'))
        .when(identifier.str.contains(r'^(?:ensembl:)?ENSP\d+(?:\.\d+)?$')).then(pl.lit('Ensembl protein ID'))
        .when(identifier.str.contains(r'^(?:ensembl:)?ENST\d+(?:\.\d+)?$')).then(pl.lit('Ensembl transcript ID'))
        .when(identifier.str.contains(r'^\d+$')).then(pl.lit('NCBI Gene ID'))
        .otherwise(pl.lit('Other ID'))
    )
    return (
        ids.alias('id').to_frame()
        .with_columns(identifier.str.strip_chars())
        .filter(pl.col('id').is_not_null() & ~pl.col('id').is_in(['', '-']))
        .unique()
        .group_by(kind.alias('Identifier type')).agg(
            pl.len().alias('Distinct IDs'), pl.col('id').min().alias('Example ID'))
        .sort('Distinct IDs', 'Identifier type', descending=[True, False])
    )


def scan_huri_interactors(path: Path) -> pl.LazyFrame:
    """Keep both primary PSI-MITAB endpoint fields, including non-UniProt IDs."""
    return pl.scan_csv(path, separator='\t', has_header=False, comment_prefix='#',
                       quote_char=None, infer_schema=False).select(
        pl.nth(0).alias('a'), pl.nth(1).alias('b'))


def bioplex_source_ids(paths: Sequence[Path]) -> pl.Series:
    """Use source accessions where supplied, then Gene IDs without an accession in any release."""
    endpoints = []
    for path in paths:
        release = pl.scan_csv(path, separator='\t', infer_schema=False)
        # The release schemas share column order, even where their headers differ.
        for gene, accession in ((0, 2), (1, 3)):
            endpoints.append(release.select(
                pl.nth(gene).str.strip_chars().alias('gene'),
                pl.nth(accession).str.strip_chars().alias('accession'),
            ))
    pairs = pl.concat(endpoints).unique().collect(engine='streaming')
    has_accession = pl.col('accession').str.contains(SOURCE_ACCESSION).fill_null(False)
    supplied = pairs.filter(has_accession)
    fallback = pairs.filter(~has_accession).join(
        supplied.select('gene').unique(), on='gene', how='anti')
    ids = pl.concat([supplied['accession'], fallback['gene']]).drop_nulls().unique().sort()
    return ids.filter(~ids.is_in(['', '-']))


def bioplex_release_paths(root: Path) -> list[Path]:
    return [root / 'raw_networks' / 'bioplex' / filename for _, filename in BIOPLEX_HEK293T_RELEASES]


def _uniprot_session():
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    return session


def _current_uniprot_ids(directory):
    """Download the current human primary and indexed isoform accessions in one request."""
    cache = directory / 'uniprot_search' / 'human_accessions.json'
    if cache.is_file() and time.time() - cache.stat().st_mtime < UNIPROT_CACHE_SECONDS:
        result = json.loads(cache.read_text())
    else:
        with _uniprot_session() as session:
            response = session.get(f'{API}/uniprotkb/stream', timeout=120, params={
                'query': 'organism_id:9606', 'format': 'list', 'includeIsoform': 'true',
            })
            response.raise_for_status()
        ids = response.text.split()
        if not ids or not all(re.fullmatch(UNIPROT_ACCESSION + r'(?:-\d+)?', identifier) for identifier in ids):
            raise ValueError('UniProt returned an invalid human accession list')
        result = {'ids': ids, 'release': response.headers.get('X-UniProt-Release')}
        _write_json(cache, result)
    return set(result['ids']), result['release']


def _search_uniprot_batch(ids, directory, *, namespace='UniProtKB_AC-ID'):
    """Resolve obsolete IDs or cross-references and collect explicit isoform aliases."""
    stage = f'{namespace}-v1'
    digest = hashlib.sha256(json.dumps([stage, ids]).encode()).hexdigest()
    cache = directory / 'uniprot_search' / f'{stage}_{digest[:16]}.json'
    if cache.is_file() and time.time() - cache.stat().st_mtime < UNIPROT_CACHE_SECONDS:
        return json.loads(cache.read_text())

    ids = set(ids)
    fields = None
    if namespace == 'UniProtKB_AC-ID':
        terms = ' OR '.join(sorted(ids))
        clause = f'accession:({terms}) OR sec_acc:({terms})'
    else:
        database = namespace.lower()
        clause = 'xref:(' + ' OR '.join(f'{database}-{identifier}' for identifier in sorted(ids)) + ')'
        fields = f'accession,organism_id,xref_{database}'
    params = {'query': f'({clause}) AND organism_id:9606', 'format': 'json', 'size': 250,
              'includeIsoform': 'false'}
    if fields:
        params['fields'] = fields
    mapping = {identifier: set() for identifier in ids}
    isoforms = {}
    releases = set()
    with _uniprot_session() as session:
        url = f'{API}/uniprotkb/search'
        while url:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            release = response.headers.get('X-UniProt-Release')
            if release:
                releases.add(release)
            for entry in response.json()['results']:
                if entry['organism']['taxonId'] != 9606:
                    continue
                accession = entry['primaryAccession']
                sources = [accession, *entry.get('secondaryAccessions', [])]
                if namespace == 'Ensembl':
                    sources = []
                    for xref in entry.get('uniProtKBCrossReferences', []):
                        if xref['database'] == 'Ensembl':
                            sources.append(xref['id'].split('.')[0])
                            sources.extend(prop['value'].split('.')[0] for prop in xref.get('properties', [])
                                           if prop['key'] in ('GeneId', 'ProteinId'))
                elif namespace == 'GeneID':
                    sources = [xref['id'] for xref in entry.get('uniProtKBCrossReferences', [])
                               if xref['database'] == 'GeneID']
                for source in ids.intersection(sources):
                    mapping[source].add(accession)
                for comment in entry.get('comments', []):
                    for isoform in comment.get('isoforms', []):
                        primary, *aliases = isoform['isoformIds']
                        for alias in [primary, *aliases]:
                            isoforms.setdefault(alias, set()).add(primary)
            url = response.links.get('next', {}).get('url')
            params = None
    result = {
        'mapping': {identifier: sorted(mapping[identifier]) for identifier in sorted(ids)},
        'isoforms': {identifier: sorted(isoforms[identifier]) for identifier in sorted(isoforms)},
        'uniprot_releases': sorted(releases),
    }
    _write_json(cache, result)
    return result


def _search_uniprot(ids, directory, *, namespace='UniProtKB_AC-ID'):
    # UniProt caps queries at 100 conditions. Accession lookups contain two OR lists.
    batch_size = 48 if namespace == 'UniProtKB_AC-ID' else 95
    pattern = {'UniProtKB_AC-ID': UNIPROT_ACCESSION, 'Ensembl': r'ENS[GPT]\d+', 'GeneID': r'\d+'}[namespace]
    ids = {identifier for identifier in ids if re.fullmatch(pattern, identifier)}
    mapping = {}
    aliases = {}
    releases = set()

    def add(result):
        matches = ids.intersection(result['mapping'])
        if not matches:
            return
        mapping.update({identifier: result['mapping'][identifier] for identifier in matches})
        for identifier, targets in result['isoforms'].items():
            aliases.setdefault(identifier, set()).update(targets)
        releases.update(result['uniprot_releases'])

    # Completed lookups can be reused even when a changed source shifts the batch boundaries.
    for path in sorted((directory / 'uniprot_search').glob(f'{namespace}-v1_*.json')):
        if time.time() - path.stat().st_mtime < UNIPROT_CACHE_SECONDS:
            add(json.loads(path.read_text()))
    needed = sorted(ids - mapping.keys())
    batches = [needed[start:start + batch_size] for start in range(0, len(needed), batch_size)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(lambda batch: _search_uniprot_batch(
                batch, directory, namespace=namespace), batches):
            add(result)
    return mapping, aliases, releases


def network_source_path(root: Path, name: str) -> Path:
    if name == 'HuRI':
        return root / 'raw_networks' / HURI_PSI
    directory = 'processed_networks' if name == 'BioPlex' else 'raw_networks'
    return root / directory / FILES[name]


def scan_mapping_source(name: str, path: Path) -> pl.LazyFrame:
    """Read source identifiers at their original isoform resolution."""
    if name == 'HuRI':
        frame = scan_huri_interactors(path).with_columns(pl.col('a', 'b').str.extract(PSI_UNIPROT))
    elif name == 'huMAP2':
        frame = scan_humap_source(path)
    else:
        columns = TAGC_COLUMNS if name == TAGC_NAME else ['Uniprot A', 'Uniprot B']
        frame = pl.scan_csv(path, separator=',' if name == TAGC_NAME else '\t',
                            infer_schema=False).select(pl.col(columns[0]).alias('a'),
                                                       pl.col(columns[1]).alias('b'))
    return frame


def _isoform_mapping_pairs(name: str, path: Path, entries: dict, cutoff: float):
    source = scan_mapping_source(name, path)
    if name == 'huMAP2':
        source = source.with_columns(pl.col('probability').cast(pl.Float64, strict=False))
        source = source.filter(_valid_score('probability') & (pl.col('probability') >= cutoff))
    pairs = source.select(pl.col('a', 'b').str.strip_chars()).drop_nulls().filter(
        (pl.col('a') != '') & (pl.col('b') != '') & (pl.col('a') != pl.col('b'))
    ).select(pl.min_horizontal('a', 'b').alias('a'), pl.max_horizontal('a', 'b').alias('b')).unique()
    lookup = pl.DataFrame(
        [(identifier, '|'.join(sorted(set(entry['targets'])))) for identifier, entry in entries.items()
         if entry['status'] != 'Parent fallback' and entry['targets']],
        schema={'source': pl.String, 'target': pl.String}, orient='row',
    ).lazy()
    mapped = pairs.join(lookup.rename({'target': 'mapped_a'}), left_on='a', right_on='source').join(
        lookup.rename({'target': 'mapped_b'}), left_on='b', right_on='source'
    ).filter(pl.col('mapped_a') != pl.col('mapped_b')).select(
        pl.min_horizontal('mapped_a', 'mapped_b').alias('a'),
        pl.max_horizontal('mapped_a', 'mapped_b').alias('b'),
    ).unique()
    return pairs, mapped


def compare_isoform_mapping(name: str, path: Path, entries: dict, cutoff: float = 0.05):
    """Compare source and strictly mapped undirected pairs at isoform resolution."""
    pairs, mapped = _isoform_mapping_pairs(name, path, entries, cutoff)
    def counts(frame):
        endpoints = pl.concat_list('a', 'b').explode()
        return frame.select(
            pl.len(), endpoints.n_unique().alias('nodes'),
            endpoints.str.split('|').explode().str.replace(ISOFORM_SUFFIX, '').n_unique().alias('parents'),
        )
    source_count, mapped_count, shared_count, preview = pl.collect_all(
        [counts(pairs), counts(mapped),
         pairs.join(mapped, on=['a', 'b'], how='semi').select(pl.len()),
         mapped.sort('a', 'b').head(100)], engine='streaming',
    )
    before, after, shared = source_count.item(0, 0), mapped_count.item(0, 0), shared_count.item()
    return before, after, before - shared, after - shared, preview, source_count.row(0)[1:], mapped_count.row(0)[1:]


def save_isoform_mapping_edges(name: str, source: Path, report: Path, cutoff: float) -> Path:
    """Save the strict mapping edge list alongside its source and audit fingerprints."""
    audit = json.loads(report.read_text(encoding='utf-8'))
    _, mapped = _isoform_mapping_pairs(name, source, audit['entries'], cutoff)
    suffix = f'_{cutoff:g}' if name == 'huMAP2' else ''
    target = report.parent.parent / 'processed_networks' / 'isoform_edges' / f'{name}{suffix}.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        mapped.sink_parquet(temporary, engine='streaming')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    _write_json(target.with_suffix('.json'), {
        'network': name, 'source': _fingerprint(source), 'mapping_audit': _fingerprint(report),
        'uniprot_releases': audit['uniprot_releases'],
        'minimum_probability': cutoff if name == 'huMAP2' else None,
        'policy': 'Keep all documented targets as one node label set. Merge identical sets. Exclude unmapped endpoints and parent-only fallbacks.',
        'output': _fingerprint(target),
    })
    return target


def prepare_mapping(root=Path('data')):
    """Audit source IDs, then return parent accessions for canonical network builds."""
    root = Path(root)
    directory = root / 'mappings'
    sources = {}
    for name in FILES:
        path = network_source_path(root, name)
        if not path.is_file():
            continue
        if name == 'HuRI':
            frame = scan_huri_interactors(path).with_columns(
                pl.coalesce(pl.col(endpoint).str.extract(PSI_UNIPROT), pl.col(endpoint)).alias(endpoint)
                for endpoint in ('a', 'b'))
        else:
            frame = scan_mapping_source(name, path)
        ids, rows, isoform_rows = source_endpoint_profile(frame)
        sources[name] = {'source': _fingerprint(path), 'ids': ids, 'rows': rows, 'isoform_rows': isoform_rows}
    releases = [path for path in bioplex_release_paths(root) if path.is_file()]
    if releases:
        sources[BIOPLEX_ORIGINAL] = {'source': [_fingerprint(path) for path in releases],
                                    'ids': bioplex_source_ids(releases).to_list()}
    if not sources:
        raise FileNotFoundError(f'No network source files found in {root}')

    queries = {original: normalize_identifier(original, retain_isoforms=True)
               for original in sorted({identifier for source in sources.values() for identifier in source['ids']})}
    results = {}
    accession_ids = {query for ns, query in queries.values() if ns == 'UniProtKB_AC-ID'}
    current, release = _current_uniprot_ids(directory)
    missing = accession_ids - current
    # An explicit isoform can also be annotated on its parent but absent from the index.
    parents = {re.sub(ISOFORM_SUFFIX, '', identifier) for identifier in missing}
    accessions, aliases, releases = _search_uniprot(parents, directory)
    if release:
        releases.add(release)
    for namespace in ('Ensembl', 'GeneID'):
        ids = {query for ns, query in queries.values() if ns == namespace}
        results[namespace], _, source_releases = _search_uniprot(ids, directory, namespace=namespace)
        releases.update(source_releases)
    entries = {}
    mapping = {}
    for original, (namespace, query) in queries.items():
        parent_query = None
        if namespace == 'UniProtKB_AC-ID' and re.search(ISOFORM_SUFFIX, query):
            targets = [query] if query in current else sorted(aliases.get(query, set()))
            status = 'Exact isoform'
            if not targets:
                parent_query = re.sub(ISOFORM_SUFFIX, '', query)
                targets = accessions.get(parent_query, [])
                status = 'Parent fallback'
        elif namespace == 'UniProtKB_AC-ID':
            targets = [query] if query in current else accessions.get(query, [])
            status = 'Primary unchanged' if targets == [query] else 'Updated accession'
        else:
            targets = results[namespace].get(query, [])
            status = 'Cross-reference'
        primary = sorted({re.sub(ISOFORM_SUFFIX, '', target) for target in targets})
        mapping[original] = primary
        entries[original] = {'namespace': namespace, 'query': query, 'parent_query': parent_query,
                             'status': status if targets else 'Unmapped', 'targets': targets, 'primary': primary}
    created_utc = datetime.now(timezone.utc).isoformat()
    _write_json(directory / 'combined.json', {
        'mapping': mapping, 'created_utc': created_utc,
        'policy': 'Human UniProtKB primary accessions after exact isoform lookup and parent fallback',
    })
    _write_json(directory / NETWORK_MAPPING, {
        'created_utc': created_utc, 'taxon': 9606, 'sources': sources, 'entries': entries,
        'accession_api': [f'{API}/uniprotkb/stream', f'{API}/uniprotkb/search'],
        'uniprot_releases': sorted(releases),
    })
    return mapping


def ensure_swissprot_reference(root=Path('data')) -> Path:
    """Cache the official reviewed human primary accession list."""
    root = Path(root)
    directory = root / 'mappings'
    directory.mkdir(parents=True, exist_ok=True)
    accessions = directory / 'swissprot_human_accessions.tsv'
    source = directory / 'swissprot_human_source.json'
    if not accessions.exists() or not source.exists():
        response = requests.get(f'{API}/uniprotkb/stream', params={
            'query': '(organism_id:9606) AND (reviewed:true)',
            'format': 'tsv', 'fields': 'accession'}, timeout=120)
        response.raise_for_status()
        lines = response.text.splitlines()
        if not lines or lines[0] != 'Entry' or len(lines) < 2:
            raise ValueError('Invalid or empty UniProt reviewed accession list')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as handle:
            handle.write(response.text)
        temporary = Path(handle.name)
        try:
            temporary.replace(accessions)
        finally:
            temporary.unlink(missing_ok=True)
        _write_json(source, {
            'url': response.url,
            'uniprot_release': response.headers.get('x-uniprot-release'),
            'release_date': response.headers.get('x-uniprot-release-date'),
            'count': len(lines) - 1,
        })
    return accessions


def mapping_target_statistics(rows: pl.DataFrame) -> pl.DataFrame:
    """Summarize per-network coverage and count the union of all mapping targets."""
    return rows.group_by('Network').agg(
        pl.len().alias('Source IDs'),
        (pl.col('primary').list.len() > 0).sum().alias('Source IDs with a match'),
        (pl.col('primary').list.len() > 1).sum().alias('Source IDs with multiple primary matches'),
        pl.col('targets').explode().drop_nulls().n_unique().alias('Distinct UniProt target IDs'),
        pl.col('primary').explode().drop_nulls().n_unique().alias('Distinct UniProt primary accessions'),
        pl.col('reviewed').explode().drop_nulls().n_unique().alias('Distinct Swiss-Prot accessions'),
    ).sort('Network')


# ---------------------------------------------------------------------------
# Canonical Network Preparation & Confidence-Filtered Versions
# ---------------------------------------------------------------------------

def score_column(name: str) -> str:
    """Return the native canonical score column for a supported base network."""
    try:
        return SCORE_COLUMNS[name]
    except KeyError:
        raise ValueError(f'{name} has no supported interaction confidence score') from None


def _valid_score(column: str) -> pl.Expr:
    return pl.col(column).is_finite() & pl.col(column).is_between(0, 1)


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    info = path.stat()
    return {'path': str(path), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'sha256': digest.hexdigest()}


def _matches_fingerprint(path: Path, fingerprint: dict) -> bool:
    info = path.stat()
    return bool(fingerprint.get('sha256')) and info.st_size == fingerprint.get('size') and (
        info.st_mtime_ns == fingerprint.get('mtime_ns')
        or _fingerprint(path)['sha256'] == fingerprint['sha256']
    )


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as file:
        json.dump(value, file, indent=2)
        temporary = Path(file.name)
    temporary.replace(path)


def _write_parquet(path: Path, frame: pl.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        frame.write_parquet(temporary, compression='zstd')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _huri_path(root: Path, isoforms: str) -> Path:
    if isoforms not in HURI_ISOFORM_POLICIES:
        raise ValueError(f'Unknown HuRI isoform policy: {isoforms}')
    directory = root / 'processed_networks'
    if isoforms != 'collapse':
        directory = directory / 'huri_isoforms' / isoforms
    return directory / 'HuRI.parquet'


def _network_directory(root: Path, isoforms: str) -> Path:
    if isoforms not in ISOFORM_POLICIES:
        raise ValueError(f'Unknown isoform policy: {isoforms}')
    directory = root / 'processed_networks'
    return directory if isoforms == 'collapse' else directory / 'isoforms' / isoforms


def read_huri_psi_endpoints(source: Path) -> pl.DataFrame:
    """Read the two raw PSI-MITAB interactor ID columns (and author score when present)."""
    schema = pl.scan_csv(source, separator='\t', has_header=False,
                         comment_prefix='#', quote_char=None, infer_schema=False).collect_schema().names()
    has_score = len(schema) > 14
    cols = [0, 1, 14] if has_score else [0, 1]
    endpoints = pl.read_csv(source, separator='\t', has_header=False,
                            comment_prefix='#', quote_char=None, infer_schema=False,
                            columns=cols)
    endpoints.columns = ['a', 'b', 'author_score'] if has_score else ['a', 'b']
    if has_score:
        parsed = pl.col('author_score').str.extract(PSI_AUTHOR_SCORE).str.strip_chars().cast(pl.Float64, strict=False)
        endpoints = endpoints.with_columns(parsed.alias('author_score')).with_columns(
            pl.when(_valid_score('author_score').fill_null(False)).then(pl.col('author_score')).otherwise(None).alias('author_score')
        )
    return endpoints


def iter_huri_psi_views(endpoints: pl.DataFrame):
    """Yield each policy's undirected edges and filter counts without writing files."""
    initial_stages = []
    _filter_stage(initial_stages, 'Source interaction rows', endpoints)
    score = 'author_score' if 'author_score' in endpoints.columns else None
    select_cols = [pl.col('a', 'b').str.extract(PSI_UNIPROT)] + ([pl.col(score)] if score else [])
    pairs = endpoints.select(*select_cols).filter(pl.col('a').is_not_null() & pl.col('b').is_not_null())
    _filter_stage(initial_stages, 'Keep rows with two supplied UniProt IDs (no external lookup)', pairs)
    missing_rows = endpoints.height - pairs.height
    source_self_loops = pairs.filter(pl.col('a') == pl.col('b')).height
    pairs = pairs.filter(pl.col('a') != pl.col('b'))
    _filter_stage(initial_stages, 'Remove source self-loops', pairs)
    has_isoform = pl.any_horizontal(pl.col('a', 'b').str.contains(ISOFORM_SUFFIX))
    isoform_rows = pairs.filter(has_isoform).height
    pairs = _deduplicate_pairs(pairs, score)
    _filter_stage(initial_stages, 'Deduplicate before accession normalization', pairs)

    for policy in HURI_ISOFORM_POLICIES:
        stages = list(initial_stages)
        frame = pairs
        if policy == 'canonical_only':
            frame = frame.filter(~has_isoform)
        elif policy == 'collapse':
            frame = frame.with_columns(pl.col('a', 'b').str.replace(ISOFORM_SUFFIX, ''))
        _filter_stage(stages, 'Apply isoform policy / merge identical accession labels', frame)
        before_loops = frame.height
        frame = frame.filter(pl.col('a') != pl.col('b')).with_columns(
            pl.min_horizontal('a', 'b').alias('a'),
            pl.max_horizontal('a', 'b').alias('b'),
        )
        self_loops = before_loops - frame.height
        _filter_stage(stages, 'Remove self-loops after normalization', frame)
        before_dedup = frame.height
        frame = _deduplicate_pairs(frame, score)
        _filter_stage(stages, 'Deduplicate after accession normalization', frame)
        yield policy, frame, {
            'input_rows': endpoints.height, 'rows_without_two_uniprot_ids': missing_rows,
            'source_self_loop_rows': source_self_loops,
            'rows_with_isoform_suffixes': isoform_rows,
            'rows_removed_by_isoform_policy': isoform_rows if policy == 'canonical_only' else 0,
            'self_loops_after_isoform_policy': self_loops,
            'duplicate_rows': initial_stages[-1]['edges_removed'] + before_dedup - frame.height,
            'filtering_stages': stages, 'edges': frame.height,
            'nodes': pl.concat([frame['a'], frame['b'].rename('a')]).n_unique(),
        }


def prepare_huri_psi(root: Path = Path('data')) -> Path:
    """Build all HuRI views directly from the PSI-MITAB interactor ID columns."""
    root = Path(root)
    source = network_source_path(root, 'HuRI')
    source_info = _fingerprint(source)
    for policy, frame, counts in iter_huri_psi_views(read_huri_psi_endpoints(source)):
        target = _huri_path(root, policy)
        label = HURI_ISOFORM_POLICIES[policy]
        _write_parquet(target, frame)
        score = 'author_score' if 'author_score' in frame.columns else None
        _write_json(target.with_suffix('.json'), {
            'name': 'HuRI', 'created_at': datetime.now(timezone.utc).isoformat(),
            'normalization_policy': f'Use PSI-MITAB UniProtKB interactor IDs directly; {label}; undirected; remove self-loops and duplicates; duplicate author_score=max; missing scores=null with edges retained',
            'isoform_policy': policy, 'source': source_info, 'output': _fingerprint(target),
            'score_column': score,
            'invalid_score_policy': 'retain base edges with null score; exclude from scored versions' if score else None,
            **counts,
        })
    return _huri_path(root, 'collapse')


def available_huri_isoform_policies(root: Path = Path('data')) -> dict[str, str]:
    """Expose prepared views with the matching PSI provenance."""
    policies = {}
    source_sha = None
    source = network_source_path(root, 'HuRI')
    for policy, label in HURI_ISOFORM_POLICIES.items():
        path = _huri_path(Path(root), policy)
        try:
            audit = json.loads(path.with_suffix('.json').read_text())
            audit_source = audit.get('source', {}).get('sha256')
            if (audit.get('isoform_policy') == policy and audit_source
                    and (policy == 'collapse' or audit_source == source_sha)
                    and (policy != 'collapse' or not source.is_file()
                         or _matches_fingerprint(source, audit['source']))
                    and _matches_fingerprint(path, audit.get('output', {}))):
                policies[policy] = label
                if policy == 'collapse':
                    source_sha = audit_source
        except (OSError, ValueError):
            continue
    return policies


def _map_endpoints(frame: pl.DataFrame, mapping: dict[str, list[str]], required: bool,
                   *, retain_isoforms: bool = False) -> pl.DataFrame:
    """Relabel each endpoint once; multiple accessions annotate a single node."""
    clean = pl.col('a', 'b').str.strip_chars()
    frame = frame.with_columns(clean).filter(
        pl.col('a').is_not_null() & pl.col('b').is_not_null()
        & (pl.col('a') != '') & (pl.col('b') != ''))
    rows = [(source, '|'.join(sorted({target if retain_isoforms else re.sub(ISOFORM_SUFFIX, '', target)
                                     for target in targets})) or None, True)
            for source, targets in mapping.items()]
    lookup = pl.DataFrame(rows, schema={'original': pl.String, 'mapped': pl.String, 'known_mapping': pl.Boolean}, orient='row').unique()
    for endpoint in ['a', 'b']:
        frame = frame.join(lookup, left_on=endpoint, right_on='original', how='left')
        expression = pl.col('mapped') if required else pl.when(pl.col('known_mapping').fill_null(False)).then(pl.col('mapped')).otherwise(pl.col(endpoint).str.replace(r'-\d+$', ''))
        frame = frame.with_columns(expression.alias(endpoint)).drop('mapped', 'known_mapping')
    frame = frame.filter(pl.col('a').is_not_null() & pl.col('b').is_not_null())
    return frame.with_columns(
        pl.min_horizontal('a', 'b').alias('a'),
        pl.max_horizontal('a', 'b').alias('b'))


def _filter_stage(stages: list[dict], label: str, frame: pl.DataFrame) -> None:
    """Record observed endpoint counts, including rows before pair deduplication."""
    nodes = (pl.concat([frame['a'], frame['b'].rename('a')]).to_frame()
             .filter(pl.col('a').is_not_null() & ~pl.col('a').str.strip_chars().is_in(['', '-']))
             .get_column('a').n_unique())
    previous = stages[-1] if stages else None
    stages.append({'stage': label, 'nodes': nodes, 'edges': frame.height,
                   'nodes_removed': previous['nodes'] - nodes if previous else 0,
                   'edges_removed': previous['edges'] - frame.height if previous else 0})


def _undirected_pairs(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.min_horizontal('a', 'b').alias('a'),
                              pl.max_horizontal('a', 'b').alias('b'))


def _deduplicate_pairs(frame: pl.DataFrame, score: str | None = None) -> pl.DataFrame:
    """Deduplicate undirected pairs, preserving the maximum valid native score."""
    frame = _undirected_pairs(frame)
    if score and score in frame.columns:
        return frame.group_by('a', 'b').agg(pl.col(score).max()).sort('a', 'b')
    return frame.unique(subset=['a', 'b']).sort('a', 'b')


def matching_nodes(nodes, accessions) -> set[str]:
    """Resolve accession annotations to graph nodes, counting each node once."""
    selected = set(accessions)
    return {str(node) for node in nodes
            if str(node) in selected or ('|' in str(node) and selected.intersection(str(node).split('|')))}


def _bioplex_release_stages(root: Path, merged: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    """Include original release filtering only when it reproduces the merged pair set."""
    paths = bioplex_release_paths(root)
    if not all(path.is_file() for path in paths):
        return [], []
    source_info = [_fingerprint(path) for path in paths]
    raw = pl.concat([
        pl.scan_csv(path, separator='\t', infer_schema=False).select(
            pl.nth(2).str.strip_chars().alias('a'), pl.nth(3).str.strip_chars().alias('b'))
        for path in paths
    ]).collect(engine='streaming')
    stages = []
    _filter_stage(stages, 'Original BioPlex releases combined', raw)
    raw = raw.filter(pl.all_horizontal(pl.col('a', 'b').is_not_null() & ~pl.col('a', 'b').is_in(['', '-'])))
    _filter_stage(stages, 'Release merge: remove missing endpoint IDs', raw)
    raw = raw.filter(pl.col('a') != pl.col('b'))
    _filter_stage(stages, 'Release merge: remove source self-loops', raw)
    raw = _deduplicate_pairs(raw)
    _filter_stage(stages, 'Release merge: deduplicate undirected pairs across releases', raw)
    expected = _deduplicate_pairs(merged.select('a', 'b'))
    if not raw.equals(expected):
        return [], source_info
    return stages, source_info


def build_canonical_networks(mapping: dict[str, list[str]], root: Path = Path('data'),
                             names: tuple[str, ...] | None = None,
                             *,
                             isoforms: str = 'collapse', mapping_audit: dict | None = None,
                             expected_source: dict | None = None) -> dict[str, Path]:
    """Build PSI HuRI and canonicalize the other networks with the supplied mapping."""
    root = Path(root)
    directory = _network_directory(root, isoforms)
    if isoforms != 'collapse' and mapping_audit is None:
        raise ValueError('Alternate mapped views require a UniProt mapping audit')
    outputs = {}
    for name in FILES:
        if names is not None and name not in names:
            continue
        source = network_source_path(root, name)
        if name == TAGC_NAME and not source.is_file() and names is None:
            continue
        if name == 'HuRI':
            outputs[name] = prepare_huri_psi(root)
            continue
        selected_digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        source_info = _fingerprint(source)
        if expected_source is not None and source_info['sha256'] != expected_source['sha256']:
            raise ValueError(f'{name} source does not match the saved mapping audit')
        target = directory / f'{name}.parquet'
        source_score_column = None
        if name == 'BioPlex':
            columns = pl.scan_csv(source, separator='\t', infer_schema=False).collect_schema().names()
            source_score_column = next((column for column in ('pInt', 'p(Interaction)') if column in columns), None)
            frame = pl.read_csv(source, separator='\t', infer_schema=False,
                                columns=['Uniprot A', 'Uniprot B'] + ([source_score_column] if source_score_column else []))
            frame.columns = ['a', 'b'] + (['pInt'] if source_score_column else [])
        elif name == TAGC_NAME:
            frame = pl.read_csv(source, infer_schema=False, columns=TAGC_COLUMNS)
            frame.columns = ['a', 'b']
        else:
            frame = scan_humap_source(source).collect()
        frame = frame.with_columns(pl.col('a', 'b').str.strip_chars())
        input_rows = frame.height
        stages = []
        release_sources = []
        if name == 'BioPlex':
            stages, release_sources = _bioplex_release_stages(root, frame)
        _filter_stage(stages, 'Source interaction rows', frame)
        frame = frame.filter(pl.all_horizontal(
            pl.col('a', 'b').is_not_null() & ~pl.col('a', 'b').is_in(['', '-'])))
        _filter_stage(stages, 'Remove missing endpoint IDs', frame)
        source_self_loop_rows = frame.filter(pl.col('a') == pl.col('b')).height
        frame = frame.filter(pl.col('a') != pl.col('b'))
        _filter_stage(stages, 'Remove source self-loops', frame)
        removed_isoform_rows = 0
        if isoforms == 'canonical_only':
            before = frame.height
            frame = frame.filter(~pl.any_horizontal(
                pl.col('a', 'b').str.contains(SOURCE_ISOFORM).fill_null(False)))
            removed_isoform_rows = before - frame.height
        _filter_stage(stages, 'Apply source isoform policy', frame)
        score = SCORE_COLUMNS.get(name)
        invalid_scores = 0
        if score and score in frame.columns:
            frame = frame.with_columns(pl.col(score).cast(pl.Float64, strict=False))
            valid = _valid_score(score).fill_null(False)
            invalid_scores = frame.filter(~valid).height
            frame = frame.with_columns(pl.when(valid).then(pl.col(score)).otherwise(None).alias(score))
        frame = _deduplicate_pairs(frame, score)
        _filter_stage(stages, 'Deduplicate before UniProt mapping', frame)
        nodes = set(pl.concat([frame['a'], frame['b'].rename('a')]).drop_nulls().unique().to_list())
        required = True
        unmapped = sorted(node for node in nodes if (required and node not in mapping)
                          or (node in mapping and not mapping[node]))
        unmapped_edge_rows = frame.filter(pl.col('a').is_in(unmapped) | pl.col('b').is_in(unmapped)).height
        multi_mapped = {node: mapping[node] for node in nodes if len(mapping.get(node, [])) > 1}
        frame = frame.filter(~pl.col('a').is_in(unmapped) & ~pl.col('b').is_in(unmapped))
        _filter_stage(stages, 'Remove edges with unmapped endpoints', frame)
        frame = _map_endpoints(frame, mapping, required=required, retain_isoforms=isoforms == 'retain')
        expanded_rows = frame.height
        _filter_stage(stages, 'Relabel nodes / merge identical UniProt label sets', frame)
        loops = frame.filter(pl.col('a') == pl.col('b')).height
        frame = frame.filter(pl.col('a') != pl.col('b'))
        _filter_stage(stages, 'Remove self-loops after mapping', frame)
        frame = _deduplicate_pairs(frame, score)
        _filter_stage(stages, 'Deduplicate after UniProt mapping', frame)
        if score and score in frame.columns:
            if name == 'huMAP2':
                frame = frame.filter(_valid_score(score))
        _filter_stage(stages, 'Apply native score validity policy', frame)
        if not _matches_fingerprint(source, source_info):
            raise ValueError(f'{name} source changed while preparing the network. Retry with the current source.')
        _write_parquet(target, frame)
        node_table = pl.DataFrame(
            [(source_id, '|'.join(sorted({target_id if isoforms == 'retain' else re.sub(ISOFORM_SUFFIX, '', target_id)
                                         for target_id in mapping[source_id]})))
             for source_id in sorted(nodes) if mapping.get(source_id)],
            schema={'source_id': pl.String, 'node': pl.String}, orient='row',
        ).join(pl.concat([frame.select(pl.col('a').alias('node')),
                          frame.select(pl.col('b').alias('node'))]).unique(), on='node', how='semi').group_by('node').agg(
            pl.col('source_id').sort().alias('source_ids')).with_columns(
                pl.col('node').str.split('|').alias('uniprot_accessions')).sort('node')
        node_path = directory / 'node_annotations' / f'{name}.parquet'
        _write_parquet(node_path, node_table)
        audit = {
            'name': name, 'created_at': datetime.now(timezone.utc).isoformat(),
            'normalization_policy': POLICY,
            'mapping_sha256': selected_digest,
            'source': source_info, 'output': _fingerprint(target),
            'input_rows': input_rows, 'expanded_mapped_rows': expanded_rows,
            'node_identity': 'sorted UniProt label set; multiple labels remain one node',
            'filtering_stages': stages,
            'node_annotations': _fingerprint(node_path),
            'original_release_sources': release_sources,
            'original_release_stages_available': bool(release_sources and stages[0]['stage'] == 'Original BioPlex releases combined'),
            'unmapped_ids': unmapped, 'unmapped_edge_rows': unmapped_edge_rows,
            'one_to_many_ids': multi_mapped, 'one_to_many_id_count': len(multi_mapped),
            'source_self_loop_rows': source_self_loop_rows,
            'self_loop_rows': loops, 'invalid_score_rows': invalid_scores,
            'score_column': score if score in frame.columns else None,
            'source_score_column': source_score_column if name == 'BioPlex' else score,
            'invalid_score_policy': 'retain base edges with null score; exclude from scored versions' if name == 'BioPlex' else 'exclude invalid scored rows' if name == 'huMAP2' else None,
            'edges': frame.height,
            'nodes': pl.concat([frame['a'], frame['b'].rename('a')]).n_unique(),
        }
        if isoforms != 'collapse':
            audit.update(
                normalization_policy=(
                    'Map only unsuffixed source endpoints to UniProt primary accessions'
                    if isoforms == 'canonical_only' else
                    'Keep documented UniProt targets including exact isoforms; exclude parent-only fallbacks'
                ) + '; exclude unmapped endpoints; undirected; remove loops; duplicate native score=max',
                isoform_policy=isoforms, mapping_audit=mapping_audit,
                rows_removed_by_isoform_policy=removed_isoform_rows,
            )
        _write_json(target.with_suffix('.json'), audit)
        outputs[name] = target
    return outputs


def discover_datasets(root: Path = Path('data'), *, huri_isoforms: str = 'collapse',
                      isoforms: str = 'collapse') -> dict[str, Path]:
    """Discover base networks and scored versions matching their current parent."""
    root = Path(root)
    directory = _network_directory(root, isoforms)
    datasets = {name: directory / f'{name}.parquet' for name in FILES}
    if isoforms == 'collapse':
        huri = datasets['HuRI']
        # Keep standalone Parquet inputs usable, but require PSI provenance when a
        # local source or preparation audit identifies this as a generated view.
        if ((network_source_path(root, 'HuRI').is_file() or huri.with_suffix('.json').is_file())
                and 'collapse' not in available_huri_isoform_policies(root)):
            datasets.pop('HuRI')
    if isoforms == 'collapse' and not datasets[TAGC_NAME].is_file():
        datasets[TAGC_NAME] = root / 'processed_networks' / 'TAGC.parquet'
    if isoforms != 'collapse':
        datasets = {name: path for name, path in datasets.items()
                    if name != 'HuRI' and _valid_isoform_view(root, name, path, isoforms)}
        if isoforms in available_huri_isoform_policies(root):
            datasets = {'HuRI': _huri_path(root, isoforms), **datasets}
    if huri_isoforms != 'collapse':
        if huri_isoforms not in available_huri_isoform_policies(root):
            raise ValueError(f'HuRI isoform policy is not prepared: {huri_isoforms}')
        datasets['HuRI'] = _huri_path(root, huri_isoforms)
    datasets = {name: path for name, path in datasets.items() if path.exists()}
    for name in SCORE_COLUMNS:
        base = datasets.get(name)
        base_meta = base.with_suffix('.json') if base else (directory / f'{name}.json')
        if isoforms != 'collapse' and not base:
            continue
        if not base and not base_meta.exists():
            continue
        if base_meta.exists():
            try:
                audit_output = json.loads(base_meta.read_text()).get('output', {})
            except (OSError, ValueError):
                continue
            if base and audit_output and not _matches_fingerprint(base, audit_output):
                continue
            fingerprint = audit_output.get('sha256') or (_fingerprint(base)['sha256'] if base else None)
        else:
            fingerprint = _fingerprint(base)['sha256'] if base else None
        if not fingerprint:
            continue
        for path in sorted((directory / 'versions').glob(f'{name}_*.parquet')):
            if _version_matches(path, name, fingerprint, isoforms):
                datasets[path.stem] = path
    return datasets


def _valid_isoform_view(root: Path, name: str, path: Path, policy: str) -> bool:
    try:
        audit = json.loads(path.with_suffix('.json').read_text())
        return (audit.get('isoform_policy') == policy
                and bool(audit.get('node_identity'))
                and _matches_fingerprint(path, audit['output'])
                and _matches_fingerprint(network_source_path(root, name), audit['source'])
                and _matches_fingerprint(root / 'mappings' / NETWORK_MAPPING, audit['mapping_audit']))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def prepare_isoform_views(root: Path, policy: str) -> list[str]:
    """Prepare alternate views offline with the current source files and saved UniProt audit."""
    root = Path(root)
    directory = _network_directory(root, policy)
    if policy == 'collapse':
        return []
    unavailable = []
    if policy not in available_huri_isoform_policies(root):
        if network_source_path(root, 'HuRI').is_file():
            prepare_huri_psi(root)
        elif _huri_path(root, 'collapse').is_file():
            unavailable.append('HuRI')
    if policy in available_huri_isoform_policies(root):
        huri_path = _huri_path(root, policy)
        if score_column('HuRI') in pl.read_parquet_schema(huri_path):
            huri_cutoffs = _saved_cutoffs(root / 'processed_networks', 'HuRI') | _saved_cutoffs(directory, 'HuRI')
            huri_sha = json.loads(huri_path.with_suffix('.json').read_text())['output']['sha256']
            for value in sorted(huri_cutoffs):
                target = _version_path(directory, 'HuRI', value)
                if not _version_matches(target, 'HuRI', huri_sha, policy):
                    save_scored_version('HuRI', value, root, isoforms=policy)
    report = root / 'mappings' / NETWORK_MAPPING
    mapped_names = [name for name in FILES if name != 'HuRI'
                    and (network_source_path(root, name).is_file()
                         or (root / 'processed_networks' / f'{name}.parquet').is_file()
                         or (name == TAGC_NAME and (root / 'processed_networks' / 'TAGC.parquet').is_file()))]
    audit = None
    for name in mapped_names:
        path = directory / f'{name}.parquet'
        if not _valid_isoform_view(root, name, path, policy):
            raw = network_source_path(root, name)
            if not report.is_file() or not raw.is_file():
                unavailable.append(name)
                continue
            if audit is None:
                report_fingerprint = _fingerprint(report)
                audit = json.loads(report.read_text(encoding='utf-8'))
            source = audit['sources'].get(name) or (audit['sources'].get('TAGC') if name == TAGC_NAME else None)
            if not source or not _matches_fingerprint(raw, source['source']):
                unavailable.append(name)
                continue
            entries = {identifier: audit['entries'][identifier] for identifier in source['ids']}
            if policy == 'retain':
                mapping = {identifier: entry['targets'] if entry['status'] != 'Parent fallback' else []
                           for identifier, entry in entries.items()}
            else:
                mapping = {identifier: entry['primary'] for identifier, entry in entries.items()}
            build_canonical_networks(mapping, root, names=(name,), isoforms=policy,
                                     mapping_audit=report_fingerprint, expected_source=source['source'])
        if name in SCORE_COLUMNS and score_column(name) in pl.read_parquet_schema(path):
            default_cutoffs = {0.05, 0.95} if name == 'huMAP2' else set()
            cutoffs = default_cutoffs | _saved_cutoffs(root / 'processed_networks', name) | _saved_cutoffs(directory, name)
            parent_sha = json.loads(path.with_suffix('.json').read_text())['output']['sha256']
            for value in sorted(cutoffs):
                target = _version_path(directory, name, value)
                if not _version_matches(target, name, parent_sha, policy):
                    save_scored_version(name, value, root, isoforms=policy)
    return unavailable


def _saved_cutoffs(directory: Path, name: str) -> set[float]:
    values = set()
    for path in (directory / 'versions').glob(f'{name}_*.json'):
        try:
            meta = json.loads(path.read_text())
            if meta.get('network') == name and path.with_suffix('.parquet').is_file():
                values.add(_threshold(meta['threshold']))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return values


def rebuild_mapped_networks(mapping: dict[str, list[str]], root: Path = Path('data')) -> dict[str, Path]:
    """Refresh the mapped graphs and regenerate their previously saved cutoffs."""
    names = tuple(name for name in ('BioPlex', 'huMAP2', TAGC_NAME)
                  if network_source_path(root, name).is_file())
    cutoffs = {name: _saved_cutoffs(root / 'processed_networks', name) for name in names if name in SCORE_COLUMNS}
    outputs = build_canonical_networks(mapping, root, names=names)
    for name, values in cutoffs.items():
        if score_column(name) in pl.read_parquet_schema(outputs[name]):
            for value in sorted(values):
                save_scored_version(name, value, root)
    return outputs


def rebuild_networks_from_saved_mapping(root: Path = Path('data')) -> dict[str, Path]:
    """Rebuild audited topology offline, refusing stale source/mapping combinations."""
    root = Path(root)
    report = root / 'mappings' / NETWORK_MAPPING
    audit = json.loads(report.read_text(encoding='utf-8'))
    for name in ('BioPlex', 'huMAP2', TAGC_NAME):
        source = network_source_path(root, name)
        if not source.is_file():
            continue
        record = audit['sources'].get(name) or (audit['sources'].get('TAGC') if name == TAGC_NAME else None)
        if not record or not _matches_fingerprint(source, record['source']):
            raise ValueError(f'{name}: source changed; refresh the UniProt mapping first')
    mapping = {identifier: entry['primary'] for identifier, entry in audit['entries'].items()}
    huri_cutoffs = _saved_cutoffs(root / 'processed_networks', 'HuRI')
    outputs = rebuild_mapped_networks(mapping, root)
    if network_source_path(root, 'HuRI').is_file():
        outputs['HuRI'] = prepare_huri_psi(root)
        if score_column('HuRI') in pl.read_parquet_schema(outputs['HuRI']):
            for value in sorted(huri_cutoffs):
                save_scored_version('HuRI', value, root)
    return outputs


def filtering_audit_table(datasets: Mapping[str, Path]) -> tuple[pl.DataFrame, list[str]]:
    """Read verified saved stage counts; never infer absent stages from final counts."""
    def current(path: Path, audit: dict, seen: set[Path]) -> bool:
        resolved = path.resolve()
        if resolved in seen or not _matches_fingerprint(path, audit['output']):
            return False
        seen.add(resolved)
        for source in audit.get('original_release_sources', []):
            if not _matches_fingerprint(Path(source['path']), source):
                return False
        source = audit.get('source')
        if source:
            source_path = Path(source['path'])
            if not _matches_fingerprint(source_path, source):
                return False
            if audit.get('parent_sha256'):
                parent_audit = json.loads(source_path.with_suffix('.json').read_text())
                return current(source_path, parent_audit, seen)
        return True

    records, unavailable = [], []
    for name, path in datasets.items():
        try:
            audit = json.loads(path.with_suffix('.json').read_text())
            if not audit.get('filtering_stages') or not current(path, audit, set()):
                unavailable.append(name)
                continue
            records.extend({'Network': name, 'Step': index, 'Filtering step': stage['stage'],
                            'Nodes': stage['nodes'], 'Edges / rows': stage['edges'],
                            'Nodes removed / merged': stage['nodes_removed'],
                            'Edges / rows removed': stage['edges_removed']}
                           for index, stage in enumerate(audit['filtering_stages'], 1))
        except (OSError, ValueError, KeyError, TypeError):
            unavailable.append(name)
    return pl.DataFrame(records, schema={
        'Network': pl.String, 'Step': pl.Int64, 'Filtering step': pl.String,
        'Nodes': pl.Int64, 'Edges / rows': pl.Int64,
        'Nodes removed / merged': pl.Int64, 'Edges / rows removed': pl.Int64,
    }), unavailable


def _threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Confidence threshold must be finite and between 0 and 1')
    return 0.0 if value == 0 else value


def save_humap_version(threshold: float, root: Path = Path('data')) -> Path:
    """Persist inclusive-score filtered canonical huMAP edges and provenance."""
    return save_scored_version('huMAP2', threshold, root)


def _scored_frame(name: str, root: Path, *, parent: Path | None = None) -> pl.LazyFrame:
    column = score_column(name)
    parent = parent if parent is not None else root / 'processed_networks' / f'{name}.parquet'
    frame = pl.scan_parquet(parent)
    if column not in frame.collect_schema():
        raise ValueError(f'{name} canonical network has no {column} score column. Rebuild this canonical network from a scored source file before applying confidence cutoffs.')
    return frame.with_columns(pl.col(column).cast(pl.Float64, strict=False)).filter(_valid_score(column))


def _version_path(directory: Path, name: str, threshold: float) -> Path:
    label = str(threshold).removesuffix('.0')
    return directory / 'versions' / f'{name}_{label}.parquet'


def _version_matches(path: Path, name: str, fingerprint: str, isoforms: str) -> bool:
    try:
        meta = json.loads(path.with_suffix('.json').read_text())
        column = meta.get('score_column', 'probability' if name == 'huMAP2' else None)
        return (meta['parent_sha256'] == fingerprint and column == score_column(name)
                and meta.get('isoform_policy', 'collapse') == isoforms and path.is_file()
                and (isoforms == 'collapse' or _matches_fingerprint(path, meta['output'])))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def save_scored_version(name: str, threshold: float, root: Path = Path('data'), *,
                        isoforms: str = 'collapse') -> Path:
    """Save valid native-score pairs at an inclusive threshold with provenance."""
    threshold = _threshold(threshold)
    root = Path(root)
    column = score_column(name)
    directory = _network_directory(root, isoforms)
    parent = _huri_path(root, isoforms) if name == 'HuRI' else directory / f'{name}.parquet'
    if isoforms != 'collapse':
        valid = (isoforms in available_huri_isoform_policies(root)) if name == 'HuRI' else _valid_isoform_view(root, name, parent, isoforms)
        if not valid:
            raise ValueError(f'{name} is unavailable for isoform policy {isoforms}')
    target = _version_path(directory, name, threshold)
    parent_info = _fingerprint(parent)
    frame = _scored_frame(name, root, parent=parent).filter(pl.col(column) >= threshold).collect(engine='streaming')
    parent_audit = json.loads(parent.with_suffix('.json').read_text()) if parent.with_suffix('.json').is_file() else {}
    stages = list(parent_audit.get('filtering_stages', []))
    valid_counts = _scored_frame(name, root, parent=parent).select(
        pl.len().alias('edges'), pl.concat_list('a', 'b').explode().n_unique().alias('nodes'),
    ).collect(engine='streaming').row(0, named=True)
    previous = stages[-1] if stages else valid_counts
    stages.append({'stage': 'Keep edges with valid native scores for confidence filtering', **valid_counts,
                   'nodes_removed': previous['nodes'] - valid_counts['nodes'],
                   'edges_removed': previous['edges'] - valid_counts['edges']})
    _filter_stage(stages, f'Inclusive {column} >= {threshold:g}', frame)
    if not _matches_fingerprint(parent, parent_info):
        raise ValueError(f'{name} changed while saving its confidence version. Retry with the current view.')
    _write_parquet(target, frame)
    norm_policy = (
        parent_audit.get('normalization_policy', POLICY)
        if (isoforms != 'collapse' or name == 'HuRI')
        else POLICY
    )
    meta = {
        'name': target.stem, 'threshold': threshold, 'comparison': '>=',
        'network': name, 'score_column': column,
        'filtering_stages': stages,
        'node_identity': parent_audit.get('node_identity'),
        'node_annotations': parent_audit.get('node_annotations'),
        'invalid_score_policy': 'exclude non-finite, missing, and out-of-range native scores',
        'parent_sha256': parent_info['sha256'],
        'isoform_policy': isoforms, 'source': parent_info,
        'normalization_policy': norm_policy,
        'output': _fingerprint(target), 'created_at': datetime.now(timezone.utc).isoformat(),
        'edges': frame.height,
        'nodes': pl.concat([frame['a'], frame['b'].rename('a')]).n_unique(),
    }
    _write_json(target.with_suffix('.json'), meta)
    return target


def load_edges(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=['a', 'b'])


def threshold_curve(root: Path = Path('data'), thresholds=None, name: str = 'huMAP2', *,
                    parent: Path | None = None) -> pl.DataFrame:
    """Count edges and incident nodes at inclusive thresholds without materializing full edge copies."""
    thresholds = [_threshold(t) for t in (thresholds if thresholds is not None else np.round(np.linspace(0, 1, 21), 2))]
    column = score_column(name)
    scored = _scored_frame(name, Path(root), parent=parent)
    node_max_lazy = pl.concat([
        scored.group_by(pl.col('a').alias('node')).agg(pl.col(column).max()),
        scored.group_by(pl.col('b').alias('node')).agg(pl.col(column).max()),
    ]).group_by('node').agg(pl.col(column).max())
    edge_counts_lazy = scored.select([
        (pl.col(column) >= t).sum().alias(f'e_{i}') for i, t in enumerate(thresholds)
    ])
    node_max, edge_counts = pl.collect_all([node_max_lazy, edge_counts_lazy], engine='streaming')
    node_scores = np.sort(node_max[column].to_numpy())
    edges = [int(c or 0) for c in edge_counts.row(0)] if thresholds else []
    return pl.DataFrame({
        'threshold': thresholds,
        'nodes': [len(node_scores) - int(np.searchsorted(node_scores, t, side='left')) for t in thresholds],
        'edges': edges,
    })


# ---------------------------------------------------------------------------
# Network Topology Summary & Degree Distribution
# ---------------------------------------------------------------------------

@dataclass
class LoadedNetwork:
    edges: pl.DataFrame
    input_rows: int
    retained_rows: int
    missing_rows: int
    self_loop_rows: int
    duplicate_rows: int
    threshold: float


def load_network(path: str | Path, threshold: float = 0.0) -> LoadedNetwork:
    """Read a known network file and remove missing IDs, loops and repeated pairs."""
    path = Path(path)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be a finite number between 0 and 1')
    is_humap = 'humap' in path.name.lower() or path.name.endswith('.pairsWprob.gz')
    is_bioplex = 'bioplex' in path.name.lower()
    if path.suffix == '.parquet':
        frame = pl.read_parquet(path).select('a', 'b')
    elif is_bioplex:
        frame = pl.read_csv(path, separator='\t', infer_schema=False,
                            columns=['Uniprot A', 'Uniprot B'])
        frame.columns = ['a', 'b']
    else:
        frame = pl.read_csv(path, separator='\t', has_header=False,
                            infer_schema=False,
                            new_columns=['a', 'b', 'probability'] if is_humap else ['a', 'b'])
    input_rows = frame.height
    if is_humap and path.suffix != '.parquet' and threshold > 0:
        frame = frame.filter(pl.col('probability').cast(pl.Float64, strict=True) >= threshold)
    retained_rows = frame.height
    frame = frame.select(pl.col('a', 'b').str.strip_chars())
    frame = frame.filter(pl.col('a').is_not_null() & pl.col('b').is_not_null()
                         & (pl.col('a') != '') & (pl.col('b') != ''))
    missing_rows = retained_rows - frame.height
    self_loop_rows = frame.filter(pl.col('a') == pl.col('b')).height
    frame = frame.filter(pl.col('a') != pl.col('b')).select(
        pl.min_horizontal('a', 'b').alias('a'),
        pl.max_horizontal('a', 'b').alias('b'))
    before_dedup = frame.height
    frame = frame.unique().sort('a', 'b')
    return LoadedNetwork(frame, input_rows, retained_rows, missing_rows,
                         self_loop_rows, before_dedup - frame.height, threshold)


def _degrees(network: LoadedNetwork) -> pl.DataFrame:
    return pl.concat([network.edges.select(pl.col('a').alias('node')),
                      network.edges.select(pl.col('b').alias('node'))]).group_by('node').len(name='degree')


def degree_distribution(network: LoadedNetwork) -> pl.DataFrame:
    """Number of nodes at each nonzero degree."""
    return _degrees(network).group_by('degree').len(name='count').sort('degree')


def summarize_network(network: LoadedNetwork) -> dict:
    """Measure a loop-free undirected graph, with nodes defined by edge endpoints."""
    edges = network.edges
    degrees = _degrees(network)
    n, m = degrees.height, edges.height
    largest = components = 0
    if n:
        ids = degrees.select('node').with_row_index('index')
        a = edges.join(ids, left_on='a', right_on='node', maintain_order='left').get_column('index').to_numpy()
        b = edges.join(ids, left_on='b', right_on='node', maintain_order='left').get_column('index').to_numpy()
        graph = coo_matrix((np.ones(m, dtype=np.uint8), (a, b)), shape=(n, n)).tocsr()
        components, labels = connected_components(graph, directed=False)
        largest = int(np.bincount(labels).max())
    return {
        'nodes': n,
        'mapped_accessions': degrees['node'].str.split('|').explode().drop_nulls().n_unique(),
        'parent_accessions': degrees['node'].str.split('|').explode().str.replace(ISOFORM_SUFFIX, '').drop_nulls().n_unique(),
        'edges': m, 'components': int(components),
        'largest_component': largest,
        'largest_component_fraction': largest / n if n else 0.0,
        'density': 2 * m / (n * (n - 1)) if n > 1 else 0.0,
        'mean_degree': 2 * m / n if n else 0.0,
        'input_rows': network.input_rows, 'retained_rows': network.retained_rows,
        'missing_rows': network.missing_rows, 'self_loop_rows': network.self_loop_rows,
        'duplicate_rows': network.duplicate_rows, 'threshold': network.threshold,
    }


# ---------------------------------------------------------------------------
# Dataset Overlaps (Schematic Venn & UpSet)
# ---------------------------------------------------------------------------

def membership_counts(datasets: Mapping[str, pl.DataFrame], mode: str = 'Edges') -> pl.DataFrame:
    """Count exclusive intersections without Python objects per edge."""
    if not 1 <= len(datasets) <= 12:
        raise ValueError('Select between 1 and 12 datasets')
    if mode not in {'Edges', 'Nodes'}:
        raise ValueError('mode must be Edges or Nodes')
    tagged = []
    for index, frame in enumerate(datasets.values()):
        clean = frame.select(pl.col('a', 'b').cast(pl.String).str.strip_chars()).filter(
            pl.col('a').is_not_null() & pl.col('b').is_not_null()
            & (pl.col('a') != '') & (pl.col('b') != ''))
        if mode == 'Edges':
            clean = clean.filter(pl.col('a') != pl.col('b')).select(
                pl.min_horizontal('a', 'b').alias('a'),
                pl.max_horizontal('a', 'b').alias('b')).unique()
            keys = ['a', 'b']
        else:
            clean = pl.concat([clean.select(pl.col('a').alias('node')),
                               clean.select(pl.col('b').alias('node'))]).unique()
            keys = ['node']
        tagged.append(clean.with_columns(pl.lit(1 << index, dtype=pl.UInt16).alias('bit')))
    counts = (pl.concat(tagged).group_by(keys).agg(pl.col('bit').sum().alias('mask'))
              .group_by('mask').len(name='count').sort('mask'))
    return counts.cast({'mask': pl.Int64, 'count': pl.Int64})


def _label(mask: int, names: Sequence[str]) -> str:
    return ' ∩ '.join(name for i, name in enumerate(names) if mask & (1 << i))


def _hover(mask: int, count: int, names: Sequence[str]) -> str:
    included = _label(mask, names)
    excluded = ', '.join(name for i, name in enumerate(names) if not mask & (1 << i))
    return f'Present in: {included}<br>Absent from: {excluded or "none"}<br>Exclusive count: {count:,}'


def venn_figure(counts: pl.DataFrame, names: Sequence[str]) -> go.Figure:
    """Schematic 2/3-set Venn; exact region counts, not area proportional."""
    if len(names) not in (2, 3):
        raise ValueError('Venn diagrams support exactly 2 or 3 datasets; use UpSet for more')
    colors = ['rgba(36,124,135,0.20)', 'rgba(187,108,60,0.20)', 'rgba(102,112,144,0.20)']
    centers = [(-.45, 0), (.45, 0)] if len(names) == 2 else [(-.45, .3), (.45, .3), (0, -.48)]
    positions = {1: (-.77, 0), 2: (.77, 0), 3: (0, 0)} if len(names) == 2 else {
        1: (-.76, .53), 2: (.76, .53), 3: (0, .61), 4: (0, -.95),
        5: (-.43, -.26), 6: (.43, -.26), 7: (0, .02)}
    values = dict(zip(counts['mask'], counts['count']))
    total = sum(values.values())
    percentages = {mask: 100 * values.get(mask, 0) / total if total else 0.0 for mask in positions}
    fig = go.Figure()
    for i, (x, y) in enumerate(centers):
        fig.add_shape(type='circle', x0=x - .85, x1=x + .85, y0=y - .85, y1=y + .85,
                      line=dict(color=colors[i].replace('0.20', '0.9'), width=2), fillcolor=colors[i])
    masks = list(positions)
    fig.add_trace(go.Scatter(
        x=[positions[m][0] for m in masks], y=[positions[m][1] for m in masks],
        mode='markers+text', marker=dict(size=32, color='rgba(255,255,255,0.01)'),
        text=[f'{values.get(m, 0):,}<br><span style="font-size:12px">{percentages[m]:.2f}%</span>' for m in masks],
        textfont=dict(size=18),
        hovertext=[_hover(m, values.get(m, 0), names) + f'<br>Share of selected-network union: {percentages[m]:.2f}%<br>Union total: {total:,}' for m in masks],
        hovertemplate='%{hovertext}<extra></extra>',
    ))
    label_positions = [(-.8, 1.02), (.8, 1.02)] if len(names) == 2 else [(-.85, 1.4), (.85, 1.4), (0, -1.55)]
    for name, (x, y) in zip(names, label_positions):
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, font=dict(size=14))
    fig.update_layout(
        title='Interactome overlap · schematic Venn<br><sup>Percentages of the selected-network union; areas not proportional</sup>',
        height=560, showlegend=False, template='plotly_white',
        xaxis=dict(visible=False, range=[-1.65, 1.65]),
        yaxis=dict(visible=False, range=[-1.75, 1.65], scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=90, b=10),
    )
    return fig


def venn4_figure(counts: pl.DataFrame, names: Sequence[str]) -> Figure:
    """Draw four-set Venn regions using the exact, precomputed counts."""
    from matplotlib.figure import Figure
    from venny4py.venny4py import venny4py

    if len(names) != 4:
        raise ValueError('This Venn diagram requires exactly four datasets')

    bits = [1 << i for i in range(4)]
    masks = [sum(group) for size in range(1, 5) for group in combinations(bits, size)]
    # Use one placeholder per region to obtain the layout without expanding the counts.
    sets = {name: {mask for mask in masks if mask & bit} for name, bit in zip(names, bits)}
    fig = Figure(figsize=(7, 6), facecolor='white')
    ax = fig.subplots()
    ax.set_aspect('equal')
    with tempfile.TemporaryDirectory() as directory:
        venny4py(sets=sets, asax=ax, out=directory, font_size=8.5, line_width=1,
                 colors=['#247c87', '#bb6c3c', '#667090', '#779b3e'])

    values = dict(zip(counts['mask'], counts['count']))
    total = sum(values.values())
    # Venny4py places set totals first, followed by combinations in size order.
    for label in ax.texts[:4]:
        label.remove()
    for label, mask in zip(ax.texts, masks, strict=True):
        count = values.get(mask, 0)
        percentage = 100 * count / total if total else 0.0
        label.set_text(f'{count:,}\n{percentage:.2f}%')
        label.set_linespacing(0.95)
    return fig


def upset_figure(counts: pl.DataFrame, names: Sequence[str]) -> go.Figure:
    """Interactive UpSet bars and membership matrix, using exclusive counts."""
    if not 1 <= len(names) <= 12:
        raise ValueError('Select between 1 and 12 datasets')
    data = counts.filter(pl.col('count') > 0).sort(['count', 'mask'], descending=[True, False])
    masks = data['mask'].to_list()
    values = data['count'].to_list()
    xs = list(range(len(masks)))
    hover = [_hover(mask, count, names) for mask, count in zip(masks, values)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.035,
                        row_heights=[.64, .36])
    fig.add_trace(go.Bar(x=xs, y=values, marker_color='#247c87', customdata=hover,
                         hovertemplate='%{customdata}<extra></extra>', name='Exclusive count'), row=1, col=1)
    line_x, line_y = [], []
    dot_x, dot_y, dot_colors, dot_hover = [], [], [], []
    for x, mask, tooltip in zip(xs, masks, hover):
        selected = [i for i in range(len(names)) if mask & (1 << i)]
        if len(selected) > 1:
            line_x.extend([x, x, None])
            line_y.extend([min(selected), max(selected), None])
        for i in range(len(names)):
            dot_x.append(x)
            dot_y.append(i)
            dot_colors.append('#247c87' if mask & (1 << i) else '#dce2e5')
            dot_hover.append(tooltip)
    if line_x:
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                 line=dict(color='#247c87', width=2), hoverinfo='skip', showlegend=False), row=2, col=1)
    if dot_x:
        fig.add_trace(go.Scatter(x=dot_x, y=dot_y, mode='markers',
                                 marker=dict(size=10, color=dot_colors), hovertext=dot_hover,
                                 hovertemplate='%{hovertext}<extra></extra>', showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text='Exclusive intersection count', row=1, col=1)
    fig.update_yaxes(tickvals=list(range(len(names))), ticktext=list(names), autorange='reversed', row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text='Dataset combination (filled dots indicate membership)',
                     tickvals=xs, ticktext=[str(i + 1) for i in xs], row=2, col=1)
    fig.update_layout(title='Exclusive dataset overlap · UpSet', template='plotly_white',
                      height=max(500, 360 + len(names) * 26), showlegend=False, margin=dict(l=15, r=15, t=65, b=30))
    return fig


def plot_threshold_retention(name: str = 'huMAP2', root: Path = Path('data')):
    """Render saved exact huMAP2, BioPlex, or HuRI threshold counts (requires matplotlib)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(root) / 'plots'
    with (out / f'{name}_threshold_counts.csv').open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    total = rows[0]['edges']
    fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')
    ax.plot([r['threshold'] for r in rows], [r['edges'] for r in rows], color='#555555', marker='.', linewidth=1.8)
    if name == 'huMAP2':
        for threshold, color, xpos, ypos in [(0.05, '#0072B2', .22, .80), (.95, '#B35C00', .57, .42)]:
            row = next(r for r in rows if r['threshold'] == threshold)
            edges = int(row['edges'])
            ax.axvline(threshold, color=color, linestyle='--', linewidth=1.5)
            ax.scatter([threshold], [edges], color=color, s=65, zorder=5)
            ax.annotate(f'Score ≥ {threshold:.2f}\n{edges:,} edges\n{100 * edges / total:.3f}% of all candidates',
                        xy=(threshold, edges), xytext=(xpos, ypos), textcoords='axes fraction',
                        fontsize=12, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=.5', fc='white', ec=color),
                        arrowprops=dict(arrowstyle='->', color=color))
    ax.set(yscale='log', xlim=(-.02, 1.02), xlabel='Minimum interaction confidence score (retain score ≥ cutoff)',
           ylabel='Unique canonical edges retained (log scale)', title=f'{name}: edges retained by confidence score')
    xticks = [0, .05, .2, .4, .6, .8, .95, 1] if name == 'huMAP2' else [0, .2, .4, .6, .8, 1]
    ax.set_xticks(xticks, [f'{x:g}' for x in xticks])
    ax.grid(axis='y', alpha=.2)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'{name}_threshold_retention.{extension}', dpi=220)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare canonical UniProt networks or render threshold plots.')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--plot-threshold', choices=['huMAP2', 'BioPlex', 'HuRI'], help='Render static threshold retention plots instead of running UniProt mapping.')
    action.add_argument('--huri-only', action='store_true', help='Prepare the HuRI PSI views offline.')
    args = parser.parse_args()
    if args.plot_threshold:
        plot_threshold_retention(args.plot_threshold)
    elif args.huri_only:
        print(prepare_huri_psi(), flush=True)
    else:
        print(prepare_huri_psi(), flush=True)
        ensure_swissprot_reference()
        mapping = prepare_mapping()
        print(rebuild_mapped_networks(mapping), flush=True)
