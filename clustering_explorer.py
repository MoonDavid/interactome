"""Portable discovery and reproducible annotation of saved clustering jobs."""
import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/analysis_runs/clustering_raw'
DEFAULT_METHODS = ('cugraph.louvain', 'cugraph.leiden', 'igraph.infomap',
                   'markov_clustering.mcl')


def clustering_options(runs, network):
    """A single method inventory, including runnable methods with no saved result."""
    from clustering_methods import catalog
    available = runs.filter(pl.col('network') == network) if not runs.is_empty() else runs
    methods = set(catalog())
    if not available.is_empty():
        methods.update(available['method'].to_list())
    choices = []
    for method in sorted(methods):
        matches = (available.filter((pl.col('method') == method) & (pl.col('status') == 'success'))
                   if not available.is_empty() else available)
        saved = sum((Path(row['directory']) / 'memberships.parquet').is_file() and
                    (Path(row['directory']) / 'community_sizes.parquet').is_file()
                    for row in matches.iter_rows(named=True))
        choices.append({'method': method, 'saved_versions': saved,
                        'label': f'{method} · {saved} saved version(s)' if saved else f'{method} · not saved'})
    return pl.from_dicts(choices)


def compute_clustering(network, method, *, seed=42, resolution=1.0, clusters=16, timeout=600):
    """Use the existing isolated batch worker, guards, provenance and disk layout."""
    import subprocess
    import sys
    from clustering_methods import catalog
    if method not in catalog():
        raise ValueError(f'No runnable adapter for historical method {method}')
    if method.startswith('atlas.'):
        payload = dict(network=network, method=method, seed=seed, resolution=resolution, clusters=clusters)
        command = [sys.executable, '-c',
            'import json,sys; from clustering_explorer import compute_atlas_version; '
            'print(compute_atlas_version(**json.loads(sys.argv[1])))', json.dumps(payload)]
    else:
        command = [sys.executable, str(ROOT / 'cluster_networks.py'), 'run',
            '--root', str(ROOT / 'data'), '--output', str(RAW), '--network', network,
            '--method', method, '--seed', str(seed), '--resolution', str(resolution),
            '--clusters', str(clusters), '--timeout', str(timeout)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=timeout + 90, check=False)
    if result.returncode:
        raise ValueError((result.stderr or result.stdout)[-6000:])
    return result.stdout[-6000:]


def compute_atlas_version(network, method, seed=42, resolution=1.0, clusters=16):
    """Reuse the Atlas parameter cache and persist the same portable run format."""
    import time
    from tempfile import TemporaryDirectory
    from networks import discover_datasets, _fingerprint
    from moonlighting import graph_from_path, run_cached_clustering, ADVANCED_EDGE_LIMIT
    started = time.monotonic()
    path = discover_datasets(ROOT / 'data')[network]
    if pl.scan_parquet(path).select(pl.len()).collect().item() > ADVANCED_EDGE_LIMIT:
        raise ValueError('NetworkX guard: more than 500,000 edges')
    parameters = dict(seed=seed, resolution=resolution, n_clusters=clusters)
    job = {'network': network, 'method': method, 'parameters': parameters,
           'provenance': {'input': _fingerprint(path)},
           'code_sha256': hashlib.sha256((ROOT / 'moonlighting.py').read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    target = RAW / network / method / key[:16]
    if (target / 'result.json').is_file():
        return target
    graph = graph_from_path(path)
    result = run_cached_clustering(graph, algorithm=method.removeprefix('atlas.'), **parameters)
    members = result['node_frame'].select('protein',
        pl.col('communities_str').str.split(';').alias('community')).explode('community')
    members = members.with_columns(pl.col('community').cast(pl.Int64, strict=False)).unique()
    sizes = members.filter(pl.col('community').is_not_null()).group_by('community').len().sort('community')
    covered = members.filter(pl.col('community').is_not_null()).select('protein').unique().height
    target.parent.mkdir(parents=True, exist_ok=True)
    job.update(key=key, path=str(path), target=str(target))
    with TemporaryDirectory(dir=target.parent) as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        members.write_parquet(bundle / 'memberships.parquet')
        sizes.write_parquet(bundle / 'community_sizes.parquet')
        (bundle / 'job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
        summary = dict(job=job, key=key, method=method, status='success', nodes=len(graph['nodes']),
            edges=graph['edge_count'], communities=sizes.height, covered_nodes=covered,
            unassigned_nodes=len(graph['nodes'])-covered,
            overlapping_nodes=members.filter(pl.col('community').is_not_null()).group_by('protein').len().filter(pl.col('len') > 1).height,
            elapsed_seconds=time.monotonic()-started, details=result['meta'])
        (bundle / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        bundle.rename(target)
    return target


def community_selection(community):
    if community is None:
        return None
    values = community if isinstance(community, (list, tuple, set)) else [community]
    return sorted(set(map(str, values)))


def saved_runs(root=RAW):
    rows = []
    for path in sorted(Path(root).glob('*/*/*/result.json')):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            rows.append(dict(network=path.parents[2].name, method=path.parents[1].name,
                             run=path.parent.name, directory=str(path.parent),
                             **{k: result.get(k) for k in ('status', 'nodes', 'edges', 'communities',
                                'covered_nodes', 'unassigned_nodes', 'overlapping_nodes',
                                'elapsed_seconds', 'reason')}))
        except (OSError, ValueError):
            continue
    return pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()


def annotation_target(directory, community=None, *, text_annotations=True):
    """Reject mismatched source networks; fingerprint all annotation inputs and code."""
    from clusterannotation import file_input, DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME, DEFAULT_FUNCTIONS, DEFAULT_GENERIF, DEFAULT_GENE_MAPPING, DEFAULT_ABSTRACTS
    from networks import discover_datasets
    directory = Path(directory)
    network = directory.parents[1].name
    path = discover_datasets(ROOT / 'data').get(network)
    if path is None:
        raise ValueError(f'Local network unavailable: {network}')
    frame, population, source = file_input(directory / 'memberships.parquet', str(path))
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode())
    digest.update(json.dumps({'communities': community_selection(community)}).encode())
    if not text_annotations:
        digest.update(b'GO_Reactome_only')
    for p in (DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME,
              ROOT / 'annotate_clusters.py', ROOT / 'clusterannotation.py', ROOT / 'moonlighting.py', ROOT / 'clustering_explorer.py'):
        with p.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    for extra in ((DEFAULT_GENERIF, DEFAULT_GENE_MAPPING, DEFAULT_ABSTRACTS) if text_annotations else ()):
        digest.update(str(extra).encode())
        if extra.is_file():
            digest.update(hashlib.sha256(extra.read_bytes()).digest())
    if text_annotations and DEFAULT_FUNCTIONS.is_file():
        with DEFAULT_FUNCTIONS.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    target = ROOT / 'data/analysis_runs/clustering_annotations' / network / directory.parent.name / directory.name / digest.hexdigest()[:16]
    return target, frame, population, source


def annotate_run(directory, community=None, *, text_annotations=True):
    from clusterannotation import annotate_memberships, write_annotation_bundle, DEFAULT_FUNCTIONS, DEFAULT_GENERIF
    target, frame, population, source = annotation_target(directory, community, text_annotations=text_annotations)
    if not (target / 'manifest.json').is_file():
        write_annotation_bundle(target, annotate_memberships(
            frame, population, source, go_cache=ROOT / 'data/annotations',
            communities=community_selection(community), generif=DEFAULT_GENERIF if text_annotations else None,
            uniprot_functions=DEFAULT_FUNCTIONS if text_annotations and DEFAULT_FUNCTIONS.is_file() else None))
    return target


def saved_cluster_enrichment(frame, population, cohort):
    """Test a protein set in a saved partition without rerunning clustering."""
    import numpy as np
    from scipy.stats import hypergeom
    from clusterannotation import membership_queries
    from moonlighting import bh_adjust, significance_label
    queries = membership_queries(frame, population).filter(pl.col('subset') == 'full')
    present = set(population['protein']) & set(cohort)
    counts = (queries.with_columns(pl.col('protein').is_in(present).alias('in_protein_set'))
        .group_by('community').agg(pl.len().alias('proteins'),
                                  pl.col('in_protein_set').sum().alias('protein_set_hits')))
    n, k = population.height, len(present)
    pvalues = hypergeom.sf(counts['protein_set_hits'].to_numpy().astype(np.int64)-1,
                          n, k, counts['proteins'].to_numpy().astype(np.int64))
    qvalues = bh_adjust(pvalues)
    return (counts.with_columns(pl.Series('pvalue', pvalues), pl.Series('qvalue', qvalues),
            pl.Series('significance', [significance_label(q) for q in qvalues]),
            (pl.col('proteins') * k / n).alias('expected_hits'))
        .with_columns(pl.when(pl.col('expected_hits') > 0)
            .then(pl.col('protein_set_hits') / pl.col('expected_hits'))
            .otherwise(None).alias('fold_enrichment'))
        .sort('qvalue', 'community'))


def all_saved_cluster_enrichment(runs, network, path, cohort):
    """Pool every valid saved partition with one BH family across all communities."""
    from clusterannotation import network_population
    from networks import _fingerprint
    from moonlighting import bh_adjust, significance_label
    population = network_population(Path(path))
    fingerprint = _fingerprint(Path(path))['sha256']
    eligible = runs.filter((pl.col('network') == network) & (pl.col('status') == 'success'))
    tables, diagnostics = [], []
    present = len(set(population['protein']) & set(cohort))
    for run in eligible.iter_rows(named=True):
        directory = Path(run['directory'])
        try:
            audit = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            expected = audit.get('job', {}).get('provenance', {}).get('input', {}).get('sha256')
            if not expected or expected != fingerprint:
                raise ValueError('Saved network fingerprint is missing or differs from the selected interactome')
            frame = pl.read_parquet(directory / 'memberships.parquet', columns=['protein', 'community'])
            result = saved_cluster_enrichment(frame, population, cohort).rename({'qvalue': 'run_qvalue'})
            result = result.drop('significance').with_columns(
                pl.lit(network).alias('network'), pl.lit(run['method']).alias('method'),
                pl.lit(run['run']).alias('version'), pl.lit(str(directory)).alias('directory'),
                (pl.col('protein_set_hits') / pl.col('proteins')).alias('protein_set_fraction'),
                pl.lit(population.height).alias('background_proteins'),
                pl.lit(present).alias('protein_set_in_network'))
            tables.append(result)
        except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
            diagnostics.append({'method': run['method'], 'version': run['run'], 'reason': str(exc)})
    if not tables:
        return pl.DataFrame(), pl.from_dicts(diagnostics) if diagnostics else pl.DataFrame()
    combined = pl.concat(tables, how='diagonal_relaxed')
    qvalues = bh_adjust(combined['pvalue'])
    combined = combined.with_columns(pl.Series('qvalue', qvalues),
        pl.Series('significance', [significance_label(q) for q in qvalues]))
    combined = combined.select('network', 'method', 'version', 'community', 'proteins',
        'protein_set_hits', 'protein_set_fraction', 'expected_hits', 'fold_enrichment',
        'pvalue', 'qvalue', 'significance', 'run_qvalue', 'background_proteins',
        'protein_set_in_network', 'directory').sort(
            ['qvalue', 'fold_enrichment', 'protein_set_hits', 'method', 'version', 'community'],
            descending=[False, True, True, False, False, False], nulls_last=True)
    return combined, pl.from_dicts(diagnostics) if diagnostics else pl.DataFrame()


def fuzzy_membership_counts(proteins, memberships, threshold, fallback=True):
    """Threshold normalized fuzzy scores without changing any saved partition."""
    import numpy as np
    scores = np.asarray(memberships, dtype=float)
    if not 0 < threshold <= 1:
        raise ValueError('Fuzzy threshold must be in (0, 1]')
    if (scores.ndim != 2 or scores.shape[0] != len(proteins) or scores.shape[1] == 0
            or not np.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError('Invalid saved fuzzy membership matrix')
    if len(set(proteins)) != len(proteins):
        raise ValueError('Duplicate proteins in saved fuzzy matrix')
    if not np.allclose(scores.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Saved fuzzy scores are not row-normalized; cannot apply probability threshold')
    counts = (scores >= threshold).sum(axis=1)
    if fallback:
        counts = np.maximum(counts, 1)
    return pl.DataFrame({'protein': list(map(str, proteins)), 'memberships': counts})


def membership_burden(frame, population, cohorts, *, membership_counts=None):
    """Compare distinct memberships against the shared source-union complement."""
    from scipy.stats import fisher_exact, mannwhitneyu
    ids = set(population['protein'])
    union = set().union(*cohorts.values()) & ids
    counts = membership_counts if membership_counts is not None else (frame.select('protein', 'community').drop_nulls().unique()
        .join(population, on='protein', how='semi').group_by('protein')
        .agg(pl.col('community').n_unique().alias('memberships')))
    counts = population.join(counts, on='protein', how='left').with_columns(
        pl.col('memberships').fill_null(0))
    background = counts.filter(~pl.col('protein').is_in(union))['memberships']
    rows = []
    for source, cohort in cohorts.items():
        selected = counts.filter(pl.col('protein').is_in(cohort))['memberships']
        n, b = len(selected), len(background)
        multi, bg_multi = int((selected >= 2).sum()), int((background >= 2).sum())
        valid = bool(n and b)
        odds, fisher_p = fisher_exact([[multi, n-multi], [bg_multi, b-bg_multi]],
            alternative='greater') if valid else (None, None)
        u, mw_p = mannwhitneyu(selected.to_numpy(), background.to_numpy(),
            alternative='greater', method='asymptotic') if valid else (None, None)
        rows.append(dict(source=source, protein_set_n=n, background_n=b,
            protein_set_mean=selected.mean(), background_mean=background.mean(),
            protein_set_median=selected.median(), background_median=background.median(),
            protein_set_multi_fraction=multi/n if n else None,
            background_multi_fraction=bg_multi/b if b else None,
            overlap_odds_ratio=odds, overlap_pvalue=fisher_p, burden_pvalue=mw_p,
            rank_biserial=2*float(u)/(n*b)-1 if valid else None,
            covered_proteins=int((counts['memberships'] > 0).sum()),
            overlapping_proteins=int((counts['memberships'] >= 2).sum())))
    return pl.from_dicts(rows, infer_schema_length=None), counts


def all_saved_membership_burden(runs, network, path, cohorts, *, fuzzy_threshold=None, fuzzy_fallback=True):
    """Audit compatible saved runs and correct both tests across every run/source."""
    from clusterannotation import network_population
    from networks import _fingerprint
    from moonlighting import bh_adjust, significance_label
    population = network_population(Path(path))
    fingerprint = _fingerprint(Path(path))['sha256']
    tables, diagnostics = [], []
    eligible = runs.filter((pl.col('network') == network) & (pl.col('status') == 'success'))
    for run in eligible.iter_rows(named=True):
        directory = Path(run['directory'])
        try:
            audit = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            expected = audit.get('job', {}).get('provenance', {}).get('input', {}).get('sha256')
            if expected != fingerprint:
                raise ValueError('Saved network fingerprint missing or different from selected network')
            frame = pl.read_parquet(directory / 'memberships.parquet', columns=['protein', 'community'])
            counts = None
            fuzzy = run['method'] in {'atlas.fuzzy_c_means', 'atlas.fuzzy_nmf',
                'atlas.fuzzy_rwr_diffusion', 'atlas.fuzzy_neighborhood'}
            if fuzzy and fuzzy_threshold is not None:
                import numpy as np
                with np.load(directory / 'soft_memberships.npz', allow_pickle=False) as saved:
                    counts = fuzzy_membership_counts(saved['proteins'], saved['memberships'],
                        fuzzy_threshold, fuzzy_fallback)
                if set(counts['protein']) != set(population['protein']):
                    raise ValueError('Saved fuzzy matrix proteins differ from network population')
            table, _ = membership_burden(frame, population, cohorts, membership_counts=counts)
            tables.append(table.with_columns(pl.lit(run['method']).alias('method'),
                pl.lit(run['run']).alias('version'),
                pl.lit(fuzzy_threshold if fuzzy else None, dtype=pl.Float64).alias('fuzzy_threshold'),
                pl.lit(fuzzy_fallback if fuzzy and fuzzy_threshold is not None else None,
                       dtype=pl.Boolean).alias('fuzzy_argmax_fallback')))
        except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
            diagnostics.append(dict(method=run['method'], version=run['run'], reason=str(exc)))
    if not tables:
        return pl.DataFrame(), pl.from_dicts(diagnostics) if diagnostics else pl.DataFrame()
    combined = pl.concat(tables, how='diagonal_relaxed')
    # One family includes both scientific questions and every valid run/source.
    values = combined['overlap_pvalue'].to_list() + combined['burden_pvalue'].to_list()
    valid = [(i, value) for i, value in enumerate(values) if value is not None]
    adjusted = dict(zip([i for i, _ in valid], bh_adjust([value for _, value in valid])))
    n = combined.height
    for offset, name in [(0, 'overlap'), (n, 'burden')]:
        q = [adjusted.get(i+offset) for i in range(n)]
        combined = combined.with_columns(pl.Series(f'{name}_qvalue', q, dtype=pl.Float64),
            pl.Series(f'{name}_significance', [significance_label(value) for value in q]))
    return combined.sort('overlap_qvalue', nulls_last=True), (
        pl.from_dicts(diagnostics) if diagnostics else pl.DataFrame())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    args = parser.parse_args()
    runs = saved_runs()
    failures = []
    selected = runs.filter((pl.col('status') == 'success') & pl.col('method').is_in(args.methods))
    selected = selected.with_columns(pl.col('method').replace_strict(
        {method: i for i, method in enumerate(args.methods)}, return_dtype=pl.Int64).alias('_priority')).sort('_priority', 'network', 'run')
    for row in selected.iter_rows(named=True):
        print(f"Annotating {row['network']} / {row['method']} / {row['run']}", flush=True)
        try:
            print(annotate_run(row['directory']), flush=True)
        except Exception as exc:
            failures.append(dict(network=row['network'], method=row['method'], run=row['run'], error=str(exc)))
            print(f'ANNOTATION FAILED: {failures[-1]}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} annotation runs failed; see diagnostics above')


if __name__ == '__main__':
    main()
