"""Canonical UniProt networks, ID mapping, topology summaries, threshold versions, and overlap plots."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
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

API = 'https://rest.uniprot.org'
POLICY = (
    'UniProt primary accessions; remove terminal numeric isoform suffix; '
    'expand all mapped accessions; undirected; remove source loops before '
    'mapping and canonical loops after mapping; duplicate native score=max; '
    'BioPlex invalid scores=null with edges retained'
)
FILES = {
    'HuRI': 'HuRI.tsv',
    'BioPlex': 'BioPlex_HEK293T_merged_dedup.tsv',
    'huMAP2': 'humap2_ppis_ACC_20200821.pairsWprob.gz',
}
SCORE_COLUMNS = {'huMAP2': 'probability', 'BioPlex': 'pInt'}


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


def normalize_identifier(identifier):
    """Return the mapping namespace and normalized query ID for mixed endpoints."""
    if identifier.startswith('ENSG'):
        return 'Ensembl', identifier.split('.')[0]
    if identifier.isdigit():
        return 'GeneID', identifier
    if '|' in identifier:
        fields = identifier.split('|')
        if len(fields) >= 3 and fields[0] in ('sp', 'tr'):
            identifier = fields[1]
    return 'UniProtKB_AC-ID', re.sub(r'-\d+$', '', identifier)


def prepare_mapping(root=Path('data'), recover_jobs=None):
    root = Path(root)
    huri = pl.read_csv(root / 'networks/HuRI.tsv', separator='\t', has_header=False,
                       new_columns=['a', 'b'], infer_schema=False)
    genes = pl.concat([huri['a'], huri['b']]).unique().drop_nulls().to_list()
    ensembl = map_ids(genes, 'Ensembl', root / 'mappings')
    bio = pl.read_csv(root / 'networks/BioPlex_HEK293T_merged_dedup.tsv', separator='\t',
                      columns=['Uniprot A', 'Uniprot B'], infer_schema=False)
    humap = pl.read_csv(root / 'networks/humap2_ppis_ACC_20200821.pairsWprob.gz', separator='\t',
                        has_header=False, new_columns=['a', 'b', 'score'], columns=[0, 1], infer_schema=False)
    originals = pl.concat([bio.to_series(0), bio.to_series(1), humap.to_series(0), humap.to_series(1)]).unique().drop_nulls().to_list()
    queries = {original: normalize_identifier(original) for original in originals}
    tables = {'Ensembl': dict(ensembl['mapping'])}
    for namespace in ('Ensembl', 'GeneID', 'UniProtKB_AC-ID'):
        needed = sorted({query for ns, query in queries.values() if ns == namespace and query not in tables.get(namespace, {})})
        if needed:
            result = map_ids(needed, namespace, root / 'mappings',
                             existing_jobs=recover_jobs if namespace == 'UniProtKB_AC-ID' else None)
            tables.setdefault(namespace, {}).update(result['mapping'])
    mapping = dict(ensembl['mapping'])
    mapping.update({original: tables[ns].get(query, []) for original, (ns, query) in queries.items()})
    (root / 'mappings/combined.json').write_text(json.dumps({
        'mapping': mapping,
        'policy': 'Namespace-aware human UniProtKB primary accessions; isoforms collapsed; all one-to-many mappings retained; unmapped dropped',
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return mapping


def prepare_huri_swissprot(mapping, root=Path('data')):
    """Restrict HuRI to reviewed human accessions and save a per-gene audit."""
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
        accessions.write_text(response.text)
        source.write_text(json.dumps({
            'url': response.url,
            'uniprot_release': response.headers.get('x-uniprot-release'),
            'release_date': response.headers.get('x-uniprot-release-date'),
            'count': len(lines) - 1,
        }, indent=2))
    lines = accessions.read_text().splitlines()
    if not lines or lines[0] != 'Entry' or len(lines) < 2:
        raise ValueError('Invalid or empty cached UniProt reviewed accession list')
    reviewed = set(lines[1:])
    raw = pl.read_csv(root / 'networks/HuRI.tsv', separator='\t', has_header=False,
                      new_columns=['a', 'b'], infer_schema=False)
    genes = sorted(pl.concat([raw['a'], raw['b']]).unique().drop_nulls().to_list())
    filtered = {}
    audit_rows = []
    for gene in genes:
        candidates = sorted(set(mapping.get(gene, [])))
        kept = [value for value in candidates if value in reviewed]
        excluded = [value for value in candidates if value not in reviewed]
        filtered[gene] = kept
        audit_rows.append({
            'ensembl_gene': gene,
            'status': 'reviewed_match' if kept else 'other_matches_only' if candidates else 'no_uniprot_match',
            'reviewed_accessions': '|'.join(kept),
            'excluded_accessions': '|'.join(excluded),
            'reviewed_count': len(kept),
            'excluded_count': len(excluded),
        })
    audit = pl.DataFrame(audit_rows)
    audit.write_csv(directory / 'HuRI_swissprot_audit.tsv', separator='\t')
    nonloops = raw.filter(pl.col('a') != pl.col('b'))
    lost = [gene for gene, targets in filtered.items() if not targets]
    source_metadata = json.loads(source.read_text())
    summary = {
        'policy': 'Human reviewed UniProtKB/Swiss-Prot entries only; retain all reviewed matches; remove source loops before mapping and resulting loops after mapping',
        'reviewed_reference': source_metadata,
        'source_gene_count': len(genes),
        'genes_with_reviewed_match': sum(bool(targets) for targets in filtered.values()),
        'genes_with_multiple_reviewed_matches': sum(len(targets) > 1 for targets in filtered.values()),
        'genes_with_only_other_matches': sum(row['status'] == 'other_matches_only' for row in audit_rows),
        'genes_without_any_uniprot_match': sum(row['status'] == 'no_uniprot_match' for row in audit_rows),
        'excluded_nonreviewed_or_absent_accession_assignments': sum(row['excluded_count'] for row in audit_rows),
        'nonloop_source_rows_lacking_reviewed_endpoint': nonloops.filter(pl.col('a').is_in(lost) | pl.col('b').is_in(lost)).height,
        'audit_file': str(directory / 'HuRI_swissprot_audit.tsv'),
    }
    (directory / 'HuRI_swissprot_summary.json').write_text(json.dumps(summary, indent=2))
    return filtered


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


def _map_endpoints(frame: pl.DataFrame, mapping: dict[str, list[str]], required: bool) -> pl.DataFrame:
    """Expand mappings through joins without iterating interaction rows."""
    clean = pl.col('a', 'b').str.strip_chars()
    frame = frame.with_columns(clean).filter(
        pl.col('a').is_not_null() & pl.col('b').is_not_null()
        & (pl.col('a') != '') & (pl.col('b') != ''))
    rows = [(source, target.split('-')[0] if target else None, True) for source, targets in mapping.items()
            for target in (targets or [None])]
    lookup = pl.DataFrame(rows, schema={'original': pl.String, 'mapped': pl.String, 'known_mapping': pl.Boolean}, orient='row').unique()
    for endpoint in ['a', 'b']:
        frame = frame.join(lookup, left_on=endpoint, right_on='original', how='left')
        expression = pl.col('mapped') if required else pl.when(pl.col('known_mapping').fill_null(False)).then(pl.col('mapped')).otherwise(pl.col(endpoint).str.replace(r'-\d+$', ''))
        frame = frame.with_columns(expression.alias(endpoint)).drop('mapped', 'known_mapping')
    frame = frame.filter(pl.col('a').is_not_null() & pl.col('b').is_not_null())
    return frame.with_columns(
        pl.min_horizontal('a', 'b').alias('a'),
        pl.max_horizontal('a', 'b').alias('b'))


def build_canonical_networks(mapping: dict[str, list[str]], root: Path = Path('data'),
                             names: tuple[str, ...] | None = None,
                             huri_mapping: dict[str, list[str]] | None = None) -> dict[str, Path]:
    """Map HuRI with supplied UniProt mapping; canonicalize other accessions."""
    root = Path(root)
    outputs = {}
    for name, filename in FILES.items():
        if names is not None and name not in names:
            continue
        selected_mapping = huri_mapping if name == 'HuRI' and huri_mapping is not None else mapping
        selected_digest = hashlib.sha256(json.dumps(selected_mapping, sort_keys=True).encode()).hexdigest()
        source = root / 'networks' / filename
        target = root / 'canonical' / f'{name}.parquet'
        source_score_column = None
        if name == 'BioPlex':
            columns = pl.scan_csv(source, separator='\t', infer_schema=False).collect_schema().names()
            source_score_column = next((column for column in ('pInt', 'p(Interaction)') if column in columns), None)
            frame = pl.read_csv(source, separator='\t', infer_schema=False,
                                columns=['Uniprot A', 'Uniprot B'] + ([source_score_column] if source_score_column else []))
            frame.columns = ['a', 'b'] + (['pInt'] if source_score_column else [])
        else:
            frame = pl.read_csv(source, separator='\t', has_header=False,
                                infer_schema=False,
                                new_columns=['a', 'b', 'probability'] if name == 'huMAP2' else ['a', 'b'])
        input_rows = frame.height
        source_self_loop_rows = frame.filter(pl.col('a') == pl.col('b')).height
        frame = frame.filter(pl.col('a') != pl.col('b'))
        nodes = set(pl.concat([frame['a'], frame['b'].rename('a')]).drop_nulls().unique().to_list())
        unmapped = sorted(node for node in nodes if (name == 'HuRI' and not selected_mapping.get(node)) or (node in selected_mapping and not selected_mapping[node]))
        unmapped_edge_rows = frame.filter(pl.col('a').is_in(unmapped) | pl.col('b').is_in(unmapped)).height
        multi_mapped = {node: selected_mapping[node] for node in nodes if len(selected_mapping.get(node, [])) > 1}
        frame = _map_endpoints(frame, selected_mapping, required=name == 'HuRI')
        expanded_rows = frame.height
        loops = frame.filter(pl.col('a') == pl.col('b')).height
        frame = frame.filter(pl.col('a') != pl.col('b'))
        invalid_scores = 0
        score = SCORE_COLUMNS.get(name)
        if score and score in frame.columns:
            frame = frame.with_columns(pl.col(score).cast(pl.Float64, strict=False))
            valid = _valid_score(score)
            invalid_scores = frame.filter(~valid.fill_null(False)).height
            if name == 'huMAP2':
                frame = frame.filter(valid)
            else:
                frame = frame.with_columns(pl.when(valid).then(pl.col(score)).otherwise(None).alias(score))
            frame = frame.group_by('a', 'b').agg(pl.col(score).max()).sort('a', 'b')
        else:
            frame = frame.unique(subset=['a', 'b']).sort('a', 'b')
        _write_parquet(target, frame)
        audit = {
            'name': name, 'created_at': datetime.now(timezone.utc).isoformat(),
            'normalization_policy': POLICY + ('; HuRI restricted to reviewed Swiss-Prot entries' if name == 'HuRI' and huri_mapping is not None else ''),
            'mapping_sha256': selected_digest,
            'source': _fingerprint(source), 'output': _fingerprint(target),
            'input_rows': input_rows, 'expanded_mapped_rows': expanded_rows,
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
        _write_json(target.with_suffix('.json'), audit)
        outputs[name] = target
    return outputs


def discover_datasets(root: Path = Path('data')) -> dict[str, Path]:
    """Discover base networks and scored versions matching their current parent."""
    root = Path(root)
    datasets = {name: root / 'canonical' / f'{name}.parquet' for name in FILES}
    datasets = {name: path for name, path in datasets.items() if path.exists()}
    for name, column in SCORE_COLUMNS.items():
        base = datasets.get(name)
        base_meta = root / 'canonical' / f'{name}.json'
        if not base and not base_meta.exists():
            continue
        fingerprint = _fingerprint(base)['sha256'] if base else None
        try:
            audited_fingerprint = json.loads(base_meta.read_text()).get('output', {}).get('sha256', fingerprint) if base_meta.exists() else fingerprint
        except (OSError, ValueError):
            continue
        if fingerprint is None:
            fingerprint = audited_fingerprint
        if not fingerprint or fingerprint != audited_fingerprint:
            continue
        for path in sorted((root / 'versions').glob(f'{name}_*.parquet')):
            try:
                meta = json.loads(path.with_suffix('.json').read_text())
                version_column = meta.get('score_column', 'probability' if name == 'huMAP2' else None)
                if meta['parent_sha256'] == fingerprint == audited_fingerprint and version_column == column:
                    datasets[path.stem] = path
            except (OSError, ValueError, KeyError):
                continue
    return datasets


def _threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Confidence threshold must be finite and between 0 and 1')
    return 0.0 if value == 0 else value


def save_humap_version(threshold: float, root: Path = Path('data')) -> Path:
    """Persist inclusive-score filtered canonical huMAP edges and provenance."""
    return save_scored_version('huMAP2', threshold, root)


def _scored_frame(name: str, root: Path) -> pl.LazyFrame:
    column = score_column(name)
    parent = root / 'canonical' / f'{name}.parquet'
    frame = pl.scan_parquet(parent)
    if column not in frame.collect_schema():
        raise ValueError(f'{name} canonical network has no {column} score column. Rebuild this canonical network from a scored source file before applying confidence cutoffs.')
    return frame.with_columns(pl.col(column).cast(pl.Float64, strict=False)).filter(_valid_score(column))


def save_scored_version(name: str, threshold: float, root: Path = Path('data')) -> Path:
    """Save valid native-score pairs at an inclusive threshold with provenance."""
    threshold = _threshold(threshold)
    root = Path(root)
    column = score_column(name)
    parent = root / 'canonical' / f'{name}.parquet'
    label = str(threshold).removesuffix('.0')
    target = root / 'versions' / f'{name}_{label}.parquet'
    frame = _scored_frame(name, root).filter(pl.col(column) >= threshold).collect()
    _write_parquet(target, frame)
    meta = {
        'name': target.stem, 'threshold': threshold, 'comparison': '>=',
        'network': name, 'score_column': column,
        'invalid_score_policy': 'exclude non-finite, missing, and out-of-range native scores',
        'parent_sha256': _fingerprint(parent)['sha256'],
        'normalization_policy': POLICY, 'source': _fingerprint(parent),
        'output': _fingerprint(target), 'created_at': datetime.now(timezone.utc).isoformat(),
        'edges': frame.height,
        'nodes': pl.concat([frame['a'], frame['b'].rename('a')]).n_unique(),
    }
    _write_json(target.with_suffix('.json'), meta)
    return target


def load_edges(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=['a', 'b'])


def threshold_curve(root: Path = Path('data'), thresholds=None, name: str = 'huMAP2') -> pl.DataFrame:
    """Count edges and incident nodes at inclusive thresholds without graph copies."""
    thresholds = [_threshold(t) for t in (thresholds if thresholds is not None else np.round(np.linspace(0, 1, 21), 2))]
    column = score_column(name)
    frame = _scored_frame(name, Path(root)).select('a', 'b', column).collect()
    edge_scores = np.sort(frame[column].to_numpy())
    node_max = pl.concat([
        frame.select(pl.col('a').alias('node'), column),
        frame.select(pl.col('b').alias('node'), column),
    ]).group_by('node').agg(pl.col(column).max())
    node_scores = np.sort(node_max[column].to_numpy())
    return pl.DataFrame({
        'threshold': thresholds,
        'nodes': [len(node_scores) - int(np.searchsorted(node_scores, t, side='left')) for t in thresholds],
        'edges': [len(edge_scores) - int(np.searchsorted(edge_scores, t, side='left')) for t in thresholds],
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
        'nodes': n, 'edges': m, 'components': int(components),
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
    """Render saved exact huMAP2 or BioPlex threshold counts (requires matplotlib)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(root) / 'plots'
    with (out / f'{name}_threshold_counts.csv').open() as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    total = rows[0]['edges']
    fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')
    ax.plot([r['threshold'] for r in rows], [r['edges'] for r in rows], color='#555555', marker='.', linewidth=1.8)
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
    ax.set_xticks([0, .05, .2, .4, .6, .8, .95, 1], ['0', '0.05', '0.2', '0.4', '0.6', '0.8', '0.95', '1'])
    ax.grid(axis='y', alpha=.2)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'{name}_threshold_retention.{extension}', dpi=220)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare canonical UniProt networks or render threshold plots.')
    parser.add_argument('--plot-threshold', choices=['huMAP2', 'BioPlex'], help='Render static threshold retention plots instead of running UniProt mapping.')
    args = parser.parse_args()
    if args.plot_threshold:
        plot_threshold_retention(args.plot_threshold)
    else:
        mapping = prepare_mapping()
        print(build_canonical_networks(mapping, huri_mapping=prepare_huri_swissprot(mapping)), flush=True)
