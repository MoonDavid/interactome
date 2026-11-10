"""Map the local interactomes to UniProt and export every filtering passage.

Run from a terminal; Streamlit is not needed:

    .venv/Scripts/python.exe pipelines/map_networks.py
    .venv/Scripts/python.exe pipelines/map_networks.py --offline
    .venv/Scripts/python.exe pipelines/map_networks.py --huri-only
    .venv/Scripts/python.exe pipelines/map_networks.py --cutoff 0.5 --isoforms retain

The first command contacts UniProt. ``--offline`` uses the saved mapping audit
and verifies that the source files have not changed. ``--huri-only`` needs no
mapping audit: HuRI uses the UniProt IDs supplied in its PSI-MITAB file directly.

Node identity
-------------
An original ID mapping to [P1, P2] becomes ONE node labelled P1|P2. It does not
become two nodes or create two copies of its interactions. Identical complete
label sets merge, regardless of accession order. P1|P2 remains distinct from P1.
Unmapped endpoints are dropped, along with their incident interactions.

Filtering order for the mapped networks
--------------------------------------
1. Load source endpoint IDs and native confidence scores as supplied.
2. Remove missing endpoints and source self-loops; apply the isoform policy.
3. Deduplicate undirected source pairs BEFORE mapping.
4. Drop interactions with an unmapped endpoint.
5. Relabel each endpoint once and merge identical UniProt label sets.
6. Remove self-loops and deduplicate again AFTER mapping.
7. Apply score validity rules, then save inclusive confidence cutoffs.

Duplicate scored pairs retain the maximum VALID native score. BioPlex base
edges with invalid scores remain with null scores; huMAP2 invalid-score edges
are excluded. Neither network's score is a statistical p-value.

This script orchestrates the tested functions in networks.py, which implement
the steps above in Polars and write atomic Parquet/JSON outputs with source
fingerprints. Keeping one implementation makes the command-line outputs and
the application's filtering audit agree.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile

import polars as pl
import requests

# Support both `python pipelines/map_networks.py` and
# `python -m pipelines.map_networks`, including from outside the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))

from networks import (  # noqa: E402
    FILES, ISOFORM_POLICIES, SCORE_COLUMNS,
    discover_datasets, filtering_audit_table, network_source_path,
    prepare_huri_psi, prepare_isoform_views, prepare_mapping,
    rebuild_mapped_networks, rebuild_networks_from_saved_mapping,
    save_scored_version,
)


def confidence_cutoff(value: str) -> float:
    """Reject invalid CLI cutoffs before any lookup or output mutation."""
    try:
        cutoff = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Cutoff must be a number between 0 and 1') from exc
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise argparse.ArgumentTypeError('Cutoff must be finite and between 0 and 1')
    return cutoff


def export_filtering_audit(root: Path, policies: list[str], *, huri_only: bool = False) -> Path:
    """Export verified counts for all requested policies, including saved versions."""
    tables = []
    for policy in policies:
        datasets = discover_datasets(root, isoforms=policy)
        if huri_only:
            datasets = {name: path for name, path in datasets.items() if name == 'HuRI'}
        table, unavailable = filtering_audit_table(datasets)
        if unavailable:
            raise ValueError(f'{policy}: missing or stale filtering audits: {", ".join(unavailable)}')
        tables.append(table.with_columns(pl.lit(policy).alias('Isoform policy')))

    # Export the SAME saved stage counts that the overview displays. Writing via
    # a temporary sibling prevents a partial CSV if the process is interrupted.
    result = pl.concat(tables).select('Isoform policy', pl.exclude('Isoform policy'))
    destination = root / 'processed_networks' / ('HuRI_filtering_steps.csv' if huri_only else 'mapping_filtering_steps.csv')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        result.write_csv(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    # Print every passage, rather than just the final topology. Before
    # deduplication, "edges" means interaction rows, including repeated pairs.
    for row in result.iter_rows(named=True):
        print(f"[{row['Isoform policy']}/{row['Network']}] {row['Step']:02d}. "
              f"{row['Filtering step']}: {row['Nodes']:,} nodes; "
              f"{row['Edges / rows']:,} edges/rows; "
              f"removed/merged {row['Nodes removed / merged']:,} nodes, "
              f"removed {row['Edges / rows removed']:,} edges/rows", flush=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    """Parse options, run mapping/preparation, and report the saved artifacts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data',
                        help='Data root containing raw_networks/, mappings/ and processed_networks/')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline', action='store_true', help='Reuse the saved UniProt audit; make no network requests')
    mode.add_argument('--huri-only', action='store_true', help='Prepare only HuRI from PSI, without an external lookup')
    parser.add_argument('--cutoff', action='append', type=confidence_cutoff, default=[],
                        help='Additional inclusive native-score cutoff; repeatable (0.05 and 0.95 always included)')
    parser.add_argument('--isoforms', nargs='+', choices=tuple(ISOFORM_POLICIES), default=['collapse'],
                        help='Views to export; parent-accession collapse is always built first')
    args = parser.parse_args(argv)
    root = args.data_dir.resolve()
    policies = list(dict.fromkeys(['collapse', *args.isoforms]))

    try:
        # Step A: resolve identifiers, then rebuild the parent-accession graphs.
        # BioPlex input is the merged release TSV produced by bioplex_merge.py;
        # raw inputs are never overwritten. Existing saved cutoffs are rebuilt.
        if args.huri_only:
            print('Preparing HuRI directly from supplied PSI UniProt IDs...', flush=True)
            prepare_huri_psi(root)  # Writes all three HuRI isoform views.
        elif args.offline:
            print('Checking source fingerprints and reusing saved UniProt mappings...', flush=True)
            rebuild_networks_from_saved_mapping(root)
        else:
            available = [name for name in FILES if network_source_path(root, name).is_file()]
            if not available:
                raise FileNotFoundError(f'No network source files found under {root}')
            print('Resolving identifiers with UniProt (completed lookup caches are reused)...', flush=True)
            mapping = prepare_mapping(root)
            print('Building mapped networks and per-node accession annotations...', flush=True)
            rebuild_mapped_networks(mapping, root)
            if 'HuRI' in available:
                prepare_huri_psi(root)

        # Step B: save the standard confidence versions plus requested cutoffs.
        # Each network uses its own native score; validity and >= filtering are
        # handled by the shared writer. Never load the large huMAP2 graph into
        # NetworkX just to prepare or filter its edge list.
        if not args.huri_only:
            datasets = discover_datasets(root)
            for name, column in SCORE_COLUMNS.items():
                parent = datasets.get(name)
                if parent is None:
                    continue
                if column not in pl.read_parquet_schema(parent):
                    print(f'{name}: no native {column} column; confidence versions skipped.', flush=True)
                    continue
                for cutoff in sorted({0.05, 0.95, *args.cutoff}):
                    print(f'Saving {name}: {column} >= {cutoff:g}...', flush=True)
                    save_scored_version(name, cutoff, root)

            # Step C: optional unsuffixed-only or exact-isoform views use the
            # same saved lookup audit; this step makes no additional API calls.
            for policy in policies:
                if policy == 'collapse':
                    continue
                print(f'Preparing isoform view: {policy}...', flush=True)
                unavailable = prepare_isoform_views(root, policy)
                if unavailable:
                    raise ValueError(f'{policy}: unavailable networks: {", ".join(unavailable)}')

        # Step D: save a readable trace for every network and confidence version.
        report = export_filtering_audit(root, policies, huri_only=args.huri_only)
        print(f'Filtering audit: {report}', flush=True)
        print(f'Network Parquets, provenance JSON and node_annotations/: {root / "processed_networks"}', flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
        print(f'Mapping failed: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
