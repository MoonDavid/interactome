"""Standalone fixed-network Monte Carlo with reusable exact NetworkX distances.

Run from the repository root for all five default prepared interactomes::

    .venv/bin/python pipelines/precompute_monte_carlo.py \
        --networks \
        data/processed_networks/HuRI.parquet \
        data/processed_networks/BioPlex.parquet \
        data/processed_networks/TAGC_binary2025.parquet \
        data/processed_networks/versions/huMAP2_0.05.parquet \
        data/processed_networks/versions/huMAP2_0.95.parquet \
        --protein-set moonlitdb --simulations 1000 \
        --analyses shortest_path cohesion --verbose

Replace --protein-set moonlitdb with --protein-set each to run every source separately.
Append --dry-run to validate inputs without computing or writing results.

Only third-party dependencies: networkx, numpy, polars, rich. No application imports.
Do not run two writers against the same distance cache concurrently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from contextlib import contextmanager

import networkx as nx
import numpy as np
import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table


class Reporter:
    """Progress on terminals; timestamped milestones in redirected logs."""

    def __init__(self, verbose=False, no_progress=False):
        """Configure stderr logging and enable animations only on a terminal.

        ``verbose`` includes diagnostic messages; ``no_progress`` suppresses live
        displays while retaining stage summaries and the final results table.
        """
        self.console = Console(stderr=True, highlight=False, log_path=False)
        self.verbose = verbose
        self.live = self.console.is_terminal and not no_progress

    def log(self, message, detail=False):
        """Write a timestamped message; show diagnostic details only in verbose mode.

        Disable Rich markup so paths and user-supplied text print literally.
        """
        if not detail or self.verbose:
            self.console.log(message, markup=False, _stack_offset=1, log_locals=False)

    @contextmanager
    def stage(self, description):
        """Wrap an operation of unknown length in a spinner and elapsed-time log.

        Exceptions propagate to the caller; a completion message is emitted only
        when the wrapped operation finishes successfully.
        """
        start = time.monotonic()
        self.log(description)
        with self.console.status(description, spinner='dots') if self.live else _idle():
            yield
        self.log(f'{description}: done in {time.monotonic()-start:.1f}s')

    def track(self, items, description):
        """Yield a sized iterable with progress, elapsed time, and estimated time left.

        Advance after the caller processes each item, so the rate measures actual
        work. Verbose mode also emits throughput updates about every ten seconds.
        No animation is written when output is redirected.
        """
        total = len(items)
        start = time.monotonic()
        self.log(f'{description}: {total:,} items')
        with Progress(SpinnerColumn(), TextColumn('{task.description}'), BarColumn(),
                      MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
                      console=self.console, disable=not self.live) as progress:
            task = progress.add_task(description, total=total)
            last_log = start
            for step, item in enumerate(items, 1):
                yield item
                progress.update(task, advance=1)
                now = time.monotonic()
                if self.verbose and now-last_log >= 10:
                    rate = step/max(now-start, 1e-9)
                    self.log(f'{description}: {step:,}/{total:,}; {rate:.1f}/s; ETA {(total-step)/rate:.0f}s')
                    last_log = now
        self.log(f'{description}: done in {time.monotonic()-start:.1f}s')


@contextmanager
def _idle():
    """Provide a no-op context manager when a stage spinner is disabled."""
    yield


REPORT = Reporter()


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['curated', 'moonlitdb', 'moonprot', 'metamoon', 'multitask', 'moondb_curated', 'moondb_predicted']


def load_graph(path):
    """Load a canonical a/b Parquet edge list as an undirected simple graph.

    Polars removes missing endpoints, self-interactions, and duplicate rows;
    NetworkX also collapses reversed edges. Scores are ignored: every interaction
    has unit length. Proteins absent from the edge list are not added as isolates.
    """
    edges = (pl.read_parquet(path).select('a', 'b').drop_nulls()
             .filter(pl.col('a') != pl.col('b')).unique())
    graph = nx.Graph()
    graph.add_edges_from(edges.iter_rows())
    return graph


def load_cohort(data, source, reviewed_only=False, min_publications=0):
    """Return canonical accessions for the filtered, locally mapped protein set.

    ``data`` contains the AllMoon/MoonLitDB CSV and saved UniProt mapping JSON.
    ``source='all'`` selects the union of source memberships. Optional reviewed
    and publication filters apply to rows before mapping. Isoforms collapse to
    base accessions; unmapped inputs are excluded and one-to-many targets retained.
    This function reads local files only and does not contact UniProt.
    """
    frame = pl.read_csv(data / 'allmoon_with_moonlitdb_human.csv', infer_schema=False)
    columns = SOURCES if source == 'all' else [source]
    frame = frame.filter(pl.any_horizontal([
        pl.col(c).str.to_lowercase().is_in(['true', '1', 'yes']) for c in columns]))
    if reviewed_only:
        frame = frame.filter(pl.col('Reviewed').str.to_lowercase() == 'reviewed')
    if min_publications:
        frame = frame.filter(pl.col('moonlitdb_number_of_publications')
                             .cast(pl.Float64, strict=False).fill_null(0).fill_nan(0) >= min_publications)
    mapping = json.loads((data / 'mappings/moonlighting_uniprot.json').read_text())['mapping']
    inputs = frame.select(pl.col('uniprot').str.strip_chars().str.split('-').list.first()).unique()
    return {target.split('-')[0] for accession in inputs['uniprot'].drop_nulls()
            for target in (mapping.get(accession) or []) if target}


def degree_matched_samples(graph, query, runs, seed, matched=True):
    """Return ordered nodes, observed indices, and ``runs`` random index arrays.

    Intersect ``query`` with the graph and require at least two retained proteins.
    Matched trials preserve the exact observed counts in floor(log2 degree) bins,
    rather than matching every degree exactly. With ``matched=False``, sample
    uniformly while preserving size. Sampling is without replacement within a
    trial; different trials may overlap and may include observed proteins.
    ``seed`` makes draws reproducible for the same graph and query.
    """
    nodes = sorted(graph)
    index = {node: i for i, node in enumerate(nodes)}
    query_indices = np.array(sorted(index[n] for n in query if n in index), dtype=int)
    if len(query_indices) < 2:
        raise ValueError('At least two protein-set members must occur in the comparison network.')
    # Degree bins retain broad hub/non-hub composition without requiring exact degrees.
    bins = np.floor(np.log2(np.maximum(1, [graph.degree(n) for n in nodes]))).astype(int)
    buckets, counts = np.unique(bins[query_indices], return_counts=True)
    pools = [(np.flatnonzero(bins == b), count) for b, count in zip(buckets, counts)]
    if not matched:
        pools = [(np.arange(len(nodes)), len(query_indices))]
    rng = np.random.default_rng(seed)
    samples = [np.concatenate([rng.choice(pool, count, replace=False) for pool, count in pools])
               for _ in REPORT.track(range(runs), 'Drawing random protein sets')]
    return nodes, query_indices, samples


def graph_cache_key(graph, nodes):
    """Identify the graph and its node order so changed networks use a new cache."""
    digest = hashlib.sha256(b'networkx-unweighted-undirected-v1')
    digest.update(json.dumps(nodes, separators=(',', ':')).encode())
    node_index = {node: i for i, node in enumerate(nodes)}

    # Sort neighbors and count each undirected edge once for a reproducible key.
    for i, node in enumerate(nodes):
        neighbors = sorted(node_index[v] for v in graph[node] if node_index[v] > i)
        digest.update(np.asarray([i, len(neighbors), *neighbors], dtype='<u8').tobytes())
    return digest.hexdigest()


def distance_cache(graph, nodes, needed, folder):
    """Compute missing distance rows and reuse rows saved by previous runs.

    ``nodes`` gives the order of proteins in the matrix. ``needed`` contains
    the row numbers required for the observed and random protein sets.

    Return the distance matrix, the number meaning "no path", and the cache
    folder. The matrix stays on disk instead of being loaded entirely into RAM.
    Only read rows marked complete. Run one writer per cache folder.
    """
    # 1. Use a separate cache folder for each graph and node order.
    cache_folder = Path(folder) / graph_cache_key(graph, nodes)
    cache_folder.mkdir(parents=True, exist_ok=True)
    distances_path = cache_folder / 'distances.npy'
    completed_path = cache_folder / 'ready.npy'

    # 2. Choose compact storage. Integer arrays cannot store infinity, so reserve
    #    their largest number for "no path": 65535 with uint16.
    node_count = len(nodes)
    dtype = np.dtype('uint16' if node_count < 65535 else 'uint32')
    unreachable = np.iinfo(dtype).max

    if distances_path.exists() and completed_path.exists():
        distances = np.load(distances_path, mmap_mode='r+')
        completed = np.load(completed_path, mmap_mode='r+')
    else:
        distances = np.lib.format.open_memmap(
            distances_path, mode='w+', dtype=dtype, shape=(node_count, node_count))
        completed = np.lib.format.open_memmap(
            completed_path, mode='w+', dtype=bool, shape=(node_count,))
        completed[:] = False
        completed.flush()

    # 3. Skip source proteins whose distances are already saved.
    missing_rows = [int(i) for i in needed if not completed[i]]
    reused_rows = len(needed) - len(missing_rows)
    size_gib = node_count * node_count * dtype.itemsize / 2**30
    REPORT.log(f'Distance cache: {reused_rows:,} rows reused; '
               f'{len(missing_rows):,} BFS rows remaining; matrix {size_gib:.2f} GiB')
    REPORT.log(f'Cache directory: {cache_folder}', detail=True)

    # 4. Search the full graph once for each missing source protein.
    node_index = {node: i for i, node in enumerate(nodes)}
    unsaved_rows = []
    for step, source_row in enumerate(REPORT.track(missing_rows, 'Exact NetworkX BFS'), 1):
        # Begin with "no path" everywhere, then fill in all reachable distances.
        row = np.full(node_count, unreachable, dtype=dtype)
        source_protein = nodes[source_row]
        reachable_distances = nx.single_source_shortest_path_length(graph, source_protein)
        for target_protein, distance in reachable_distances.items():
            row[node_index[target_protein]] = distance
        distances[source_row] = row
        unsaved_rows.append(source_row)

        # Save every 32 rows, plus the last batch. Write distances BEFORE marking
        # rows complete, so an interrupted batch can safely be recomputed.
        if len(unsaved_rows) == 32 or step == len(missing_rows):
            distances.flush()
            completed[unsaved_rows] = True
            completed.flush()
            unsaved_rows.clear()
            REPORT.log(f'Checkpoint saved: {step:,}/{len(missing_rows):,} BFS rows', detail=True)

    return distances, unreachable, str(cache_folder)


def pair_mean(distances, indices, unreachable):
    """Return (finite-pair mean distance, reachable fraction) for a protein set.

    ``indices`` must contain at least two distinct indices with completed cached
    rows. Count each unordered pair once, exclude self-pairs, and omit unreachable
    distances from the mean. The reachable fraction uses all possible pairs as
    its denominator; return NaN for the mean when no pair is reachable.

    Read one upper-triangle row slice at a time to use O(set size) temporary RAM.
    Accumulate with uint64 to avoid overflow in the compact cached distance dtype.
    """
    total, reachable = 0, 0
    for position, source in enumerate(indices[:-1]):
        values = distances[source, indices[position+1:]]
        values = values[values != unreachable]
        total += int(values.sum(dtype=np.uint64))
        reachable += len(values)
    pairs = len(indices) * (len(indices)-1) // 2
    return (total / reachable if reachable else float('nan')), reachable / pairs


def empirical_p(observed, null, two_sided=False):
    """Compute a tie-inclusive Monte Carlo p-value with the +1 correction.

    The default upper tail tests enrichment (larger observed cohesion). For path
    length, ``two_sided=True`` doubles the smaller corrected tail, capped at one,
    to detect either shorter or longer distances. Return None if the observed
    value or any null value is nonfinite; do not silently drop invalid trials.
    """
    if not np.isfinite(observed) or not np.all(np.isfinite(null)):
        return None
    lower = (1 + np.count_nonzero(null <= observed)) / (len(null)+1)
    upper = (1 + np.count_nonzero(null >= observed)) / (len(null)+1)
    return float(min(1, 2*min(lower, upper)) if two_sided else upper)


def bh_adjust(values):
    """Return Benjamini–Hochberg q-values in the original input order.

    Only finite, nonmissing p-values enter the correction family. Missing tests
    remain None. The reverse cumulative minimum makes sorted adjusted values
    monotonic, and initialization at one caps them at one.
    """
    valid = [i for i, p in enumerate(values) if p is not None and np.isfinite(p)]
    order = sorted(valid, key=lambda i: values[i])
    output, previous = [None] * len(values), 1.0
    for rank in range(len(order), 0, -1):
        i = order[rank-1]
        previous = min(previous, values[i] * len(order) / rank)
        output[i] = previous
    return output


def significance_label(q):
    """Translate an adjusted q-value into the project's significance notation.

    Use strict q < 0.05/0.01/0.001/0.0001 thresholds, ``ns`` for a valid
    nonsignificant result, and ``n/a`` for a missing test. Pass BH-adjusted values.
    """
    if q is None:
        return 'n/a'
    return next((label for threshold, label in [(0.0001, '****'), (0.001, '***'), (0.01, '**'), (0.05, '*')]
                 if q < threshold), 'ns')


def run_analysis(graph, query, runs, seed, matched, analyses, cache):
    """Evaluate one fixed graph against reproducible random protein sets.

    ``query`` contains accessions; only members present in ``graph`` are tested.
    Draw controls once and reuse the same sets across requested analyses. Paths
    traverse the whole comparison graph, whereas cohesion uses induced subgraphs.

    Return (summary rows, null-distribution Polars DataFrame, cache/reachability
    metadata). Summary rows contain raw empirical p-values; main() applies BH
    after collecting the full requested network/metric correction families.
    """
    nodes, observed_indices, samples = degree_matched_samples(graph, query, runs, seed, matched)
    sets = [observed_indices, *samples]
    columns, observed, notes = {}, {}, {}
    if 'shortest_path' in analyses:
        # Deduplicate sources across every trial before doing any expensive traversal.
        needed = np.unique(np.concatenate(sets))
        distances, unreachable, location = distance_cache(graph, nodes, needed, cache)
        values = [pair_mean(distances, indices, unreachable) for indices in REPORT.track(sets, 'Shortest-path observed + null means')]
        observed['mean_shortest_path'] = values[0][0]
        columns['mean_shortest_path'] = [v[0] for v in values[1:]]
        columns['reachable_pair_fraction'] = [v[1] for v in values[1:]]
        notes.update(distance_cache=location, reachable_pair_fraction=values[0][1])
    if 'cohesion' in analyses:
        values = []
        for indices in REPORT.track(sets, 'Cohesion observed + null statistics'):
            subgraph = graph.subgraph(nodes[i] for i in indices)
            values.append((subgraph.number_of_edges(), max(map(len, nx.connected_components(subgraph)), default=0)))
        for j, metric in enumerate(['internal_edges', 'largest_component']):
            observed[metric] = values[0][j]
            columns[metric] = [v[j] for v in values[1:]]
    rows = []
    for metric, value in observed.items():
        null = np.asarray(columns[metric], dtype=float)
        finite = null[np.isfinite(null)]
        mean = float(finite.mean()) if len(finite) else None
        rows.append(dict(measure=metric, observed=value if np.isfinite(value) else None,
                         null_mean=mean, effect_observed_minus_null=float(value-mean) if mean is not None and np.isfinite(value) else None,
                         p_value=empirical_p(value, null, metric == 'mean_shortest_path'),
                         query_nodes=len(observed_indices), valid_simulations=len(finite),
                         **{k: v for k, v in notes.items() if k != 'distance_cache'}))
    return rows, pl.DataFrame(columns).with_row_index('simulation', offset=1), notes


def main(argv=None):
    """Run the CLI and save per-network null distributions and a corrected summary.

    ``argv`` is an optional argument list for programmatic use; None reads the
    process arguments. Validate settings, load the protein set, enforce the edge
    guard, and optionally restrict each comparison graph to its largest component.
    Dry runs validate inputs without writing simulations or distance caches.

    Outputs go under data/analysis_runs: per-network Parquet null distributions,
    JSON provenance, and a CSV summary with BH correction across all requested
    networks separately for shortest path and cohesion. Return zero on success;
    failures propagate to the command-line error handler below.
    """
    global REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--networks', nargs='+', type=Path, required=True, help='Canonical Parquet paths with a/b columns.')
    parser.add_argument('--protein-set', choices=[*SOURCES, 'all', 'each'], default='moonlitdb',
                        help='One source, all sources as a union, or each source separately.')
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--analyses', nargs='+', choices=['cohesion', 'shortest_path'], default=['shortest_path'])
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--reviewed-only', action='store_true')
    parser.add_argument('--min-publications', type=int, default=0)
    parser.add_argument('--scope', choices=['network', 'lcc'], default='network')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true', help='Log cache paths, checkpoints, throughput and ETA.')
    parser.add_argument('--no-progress', action='store_true', help='Disable animated progress; retain readable stage logs.')
    args = parser.parse_args(argv)
    if args.protein_set == 'each':
        # Keep each source's correction family independent across its networks.
        for source in SOURCES:
            source_args = argparse.Namespace(**vars(args))
            source_args.protein_set = source
            run_protein_set(source_args)
        return 0
    return run_protein_set(args)


def run_protein_set(args):
    """Run one source across requested networks using the shared distance caches.

    Called once per source for --protein-set each. Source-specific settings enter
    output signatures so results cannot overwrite those of another protein set.
    BH correction is performed separately for each source and analysis family.
    """
    global REPORT
    REPORT = Reporter(args.verbose, args.no_progress)
    REPORT.console.print(Panel('Fixed-network Monte Carlo\nExact NetworkX distances • reusable disk cache', title='Interactome'))
    REPORT.log(f'Simulations: {args.simulations:,}; seed: {args.seed}; protein set: {args.protein_set}; scope: {args.scope}; degree matching: {not args.uniform}')
    if not 1 <= args.simulations <= 10000 or args.min_publications < 0:
        parser.error('Use 1–10,000 simulations and nonnegative minimum publications.')
    with REPORT.stage('Loading and mapping protein set'):
        cohort = load_cohort(args.data, args.protein_set, args.reviewed_only, args.min_publications)
    REPORT.log(f'Mapped protein set: {len(cohort):,} proteins')
    context = {**vars(args), 'distance_definition': 'Exact unweighted mean over finite unordered distinct within-set pairs; paths traverse the fixed full graph. Disconnected pairs excluded and reachable fractions reported.',
               'null_model': 'Independent equal-size sets sampled without replacement per trial from the comparison graph, including query proteins; floor(log2 degree) counts preserved unless --uniform.',
               'test': 'Shortest path: two-sided doubled smaller +1 empirical tail (unavailable if any trial has no reachable pairs). Cohesion: upper +1 empirical tail.',
               'correction': 'BH across all requested networks/metrics separately for shortest path and cohesion. Stars: q<0.05/0.01/0.001/0.0001; ns otherwise; n/a unavailable.',
               'limitation': 'Finite-pair means condition on reachability; compare reachable fractions or use --scope lcc. Topology does not establish moonlighting.'}
    output = args.data / 'analysis_runs'
    signature_context = {k: v for k, v in context.items() if k not in ('verbose', 'no_progress', 'dry_run')}
    signature = hashlib.sha256(json.dumps(signature_context, sort_keys=True, default=str).encode()).hexdigest()[:16]
    rows = []
    for number, path in enumerate(args.networks, 1):
        REPORT.console.rule(f'Network {number}/{len(args.networks)}: {path.stem}')
        with REPORT.stage(f'Loading {path}'):
            graph = load_graph(path)
        if graph.number_of_edges() > 500000:
            raise ValueError(f'{path}: exceeds 500,000-edge guard; choose a threshold version.')
        if args.scope == 'lcc' and graph:
            graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        REPORT.log(f'{path.stem}: {len(graph):,} proteins; {graph.number_of_edges():,} edges; {len(cohort & set(graph)):,} set proteins')
        if args.dry_run:
            degree_matched_samples(graph, cohort, 1, args.seed, not args.uniform)
            continue
        if len(cohort & set(graph)) < 2:
            REPORT.log(f'Skipping {path.stem}/{args.protein_set}: fewer than two mapped set proteins.')
            continue
        results, null, notes = run_analysis(graph, cohort, args.simulations, args.seed, not args.uniform,
                                            args.analyses, args.data / 'analysis_runs/shortest_path_distances')
        output.mkdir(parents=True, exist_ok=True)
        network_id = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        prefix = output / f'monte_carlo_{args.protein_set}_{path.stem}_{network_id}_{signature}'
        REPORT.log(f"Saving null distributions: {Path(f'{prefix}.parquet')}")
        null.write_parquet(Path(f'{prefix}.parquet'))
        Path(f'{prefix}.json').write_text(json.dumps({**context, **notes, 'network_path': str(path.resolve()),
            'network_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}, indent=2, default=str))
        rows.extend(dict(protein_set=args.protein_set, network=path.stem, network_path=str(path.resolve()), **r) for r in results)
    if rows:
        # Correct once across all requested networks, not separately per network.
        for family in [('mean_shortest_path',), ('internal_edges', 'largest_component')]:
            subset = [r for r in rows if r['measure'] in family]
            for row, q in zip(subset, bh_adjust([r['p_value'] for r in subset])):
                row.update(fdr_bh=q, significance=significance_label(q))
        destination = output / f'monte_carlo_{args.protein_set}_{signature}_summary.csv'
        pl.DataFrame(rows).write_csv(destination)
        table = Table(title='Monte Carlo results')
        for column in ['Network', 'Measure', 'Observed', 'Null mean', 'Effect', 'q (BH)', 'Significance']:
            table.add_column(column)
        def formatted(value):
            """Render optional numbers consistently in the terminal results table."""
            return 'n/a' if value is None else f'{value:.5g}'
        for row in rows:
            table.add_row(row['network'], row['measure'], formatted(row['observed']),
                          formatted(row['null_mean']), formatted(row['effect_observed_minus_null']),
                          formatted(row['fdr_bh']), row['significance'])
        REPORT.console.print(table)
        REPORT.log(f'Summary saved: {destination}')
    else:
        REPORT.log('Dry run complete.' if args.dry_run else 'No eligible network comparisons for this protein set.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        REPORT.log('Interrupted. Completed distance checkpoints are preserved; rerun the same command to resume.')
        raise SystemExit(130)
    except (ValueError, OSError, KeyError) as exc:
        if REPORT.verbose:
            REPORT.console.print_exception(show_locals=False)
        else:
            REPORT.log(f'Error: {exc}. Use --verbose for the traceback.')
        raise SystemExit(1)
