"""Merge BioPlex HEK293T releases without converting their original columns."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import polars as pl

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_BIOPLEX_DIR = DEFAULT_DATA_DIR / "raw_networks" / "bioplex"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "processed_networks"
BIOPLEX_HEK293T_RELEASES = (
    ("v1.0", "BioPlex_HEK293T_v1.0.tsv"),
    ("v2.0", "BioPlex_HEK293T_v2.0.tsv"),
    ("v3.0", "BioPlex_HEK293T_v3.0.tsv"),
)


def merge_and_deduplicate_bioplex_hek293(
    bioplex_dir: Path = DEFAULT_BIOPLEX_DIR,
    min_pint: float = 0.0,
    write_merged_to_bioplex_dir: bool = True,
    verbose: bool = False,
    *,
    output_dir: Path | None = None,
) -> pl.DataFrame:
    """Return original BioPlex fields plus a single release-provenance column.

    Match undirected pairs by exact UniProt accession, ignoring bait/prey
    direction but retaining isoform suffixes. Keep the highest-pInt original row;
    prefer the newest release only when probabilities tie. The ``source`` column lists
    every supporting release after filtering, separated by ``|``.

    Source headers are matched by position to the first release's headers:
    Entrez A/B, UniProt A/B, symbol A/B, pW, pNI, pInt. Original values and
    endpoint orientation are retained. Rows without usable UniProt IDs and
    exact-accession self-interactions are excluded. No shared-schema snapshot is made.
    """
    started = perf_counter()

    def report(message: str) -> None:
        # Flush progress immediately, including when stdout is redirected to a log.
        if verbose:
            print(message, flush=True)

    bioplex_dir = Path(bioplex_dir)
    report(f"Building BioPlex HEK293T merged network from {bioplex_dir}")
    report("Deduplication: exact UniProt accession pairs; highest pInt wins; newest release breaks ties.")
    releases: list[pl.DataFrame] = []
    columns: list[str] = []

    for rank, (version, filename) in enumerate(BIOPLEX_HEK293T_RELEASES):
        # Read identifiers as text so numeric inference cannot change their values.
        frame = pl.read_csv(bioplex_dir / filename, separator="\t", infer_schema=False)
        # Preserve the first release's headers; later releases use the same positions.
        if not columns:
            columns = frame.columns
        frame.columns = columns

        # Polars uses col() to read existing fields and lit() for constant values.
        # The release source is public; _release_rank is a temporary selection priority.
        frame = frame.with_columns(
            pl.col(columns[6:9]).cast(pl.Float64),  # Make probabilities numeric.
            pl.lit(version).alias("source"),  # Identify this file's release on every row.
            pl.lit(rank).alias("_release_rank"),  # Larger ranks represent newer releases.
        )
        releases.append(frame)
        report(f"  Loaded {version}: {len(frame):,} rows ({filename})")

    # Stack all release observations before applying the optional confidence cutoff.
    merged = pl.concat(releases)
    report(f"Combined input: {len(merged):,} rows")
    before_filter = len(merged)
    if min_pint > 0.0:
        merged = merged.filter(pl.col(columns[8]) >= min_pint)

    report(
        f"Confidence filter (pInt >= {min_pint:g}): "
        f"{len(merged):,} retained; {before_filter - len(merged):,} removed"
        if min_pint > 0.0 else "Confidence filter: disabled"
    )

    # Use the exact original accessions as keys; isoform suffixes remain significant.
    merged = merged.with_columns(
        pl.col(columns[2]).alias("_a"),
        pl.col(columns[3]).alias("_b"),
    )
    before_ids = len(merged)
    # Remove unusable IDs first so their count is separate from self-interactions.
    merged = merged.filter(
        pl.col("_a").is_not_null() & (pl.col("_a") != "")
        & pl.col("_b").is_not_null() & (pl.col("_b") != "")
    )
    report(f"Missing/empty UniProt IDs removed: {before_ids - len(merged):,} rows")
    before_loops = len(merged)
    merged = merged.filter(pl.col("_a") != pl.col("_b"))
    report(f"Exact-accession self-interactions removed: {before_loops - len(merged):,} rows")

    # Sorting the two IDs gives A–B and B–A the same key without flipping source rows.
    merged = merged.with_columns(
        pl.min_horizontal("_a", "_b").alias("_u"),
        pl.max_horizontal("_a", "_b").alias("_v"),
    )

    if verbose:
        for version, _ in BIOPLEX_HEK293T_RELEASES:
            release = merged.filter(pl.col("source") == version)
            unique_pairs = release.select("_u", "_v").unique().height
            report(
                f"  {version} after filtering: {len(release):,} rows; "
                f"{unique_pairs:,} unique pairs; {len(release) - unique_pairs:,} repeated rows"
            )

    # Gather every supporting release before dropping repeated observations.
    sources = merged.group_by("_u", "_v").agg(
        pl.col("source").unique().sort().str.join("|").alias("source")
    )

    # Highest pInt wins across all releases; newer releases break score ties. Stable sorting
    # preserves file order for exact ties; missing probabilities sort last.
    result = (
        merged.sort(
            [columns[8], "_release_rank"],
            descending=True,
            nulls_last=True,
            maintain_order=True,
        )
        .unique(subset=["_u", "_v"], keep="first", maintain_order=True)
        .drop("source")  # Replace the winning row's source with all supporting releases.
        .join(sources, on=["_u", "_v"], how="left")
        .sort("_u", "_v")  # Produce reproducible output order.
        .select(*columns, "source")  # Keep original columns and release provenance only.
    )

    report(f"Deduplicated: {len(result):,} interactions; {len(merged) - len(result):,} duplicate rows collapsed")
    if verbose:
        # Count the accession union in the final output, not A and B independently.
        accession_a = result.get_column(columns[2])
        accession_b = result.get_column(columns[3])
        distinct_accessions = pl.concat([accession_a, accession_b.rename(accession_a.name)]).n_unique()
        report("Final accession statistics (exact IDs, including isoform suffixes):")
        report(f"  Distinct accessions in {columns[2]}: {accession_a.n_unique():,}")
        report(f"  Distinct accessions in {columns[3]}: {accession_b.n_unique():,}")
        report(f"  Total distinct accessions across both columns: {distinct_accessions:,}")
        # Canonicalize only for this summary; keep exact IDs in deduplication/output.
        all_accessions = pl.concat([accession_a, accession_b.rename(accession_a.name)])
        canonical_count = all_accessions.str.split("-").list.first().n_unique()
        report(f"  Total distinct UniProt IDs without isoform suffixes: {canonical_count:,}")
        # Describe winning-row scores, so each final interaction is counted once.
        raw_scores = result.get_column(columns[8])
        score = raw_scores.drop_nulls()
        score = score.filter(score.is_finite())
        report("Final pInt distribution (one score per deduplicated interaction):")
        report(f"  Finite scores: {len(score):,}; missing: {raw_scores.null_count():,}; "
               f"non-finite: {len(raw_scores) - raw_scores.null_count() - len(score):,}")
        if len(score):
            report(f"  Mean: {score.mean():.6g}; population std: {score.std(ddof=0):.6g}")
            for label, quantile in (("Min", 0), ("P5", .05), ("P25", .25),
                                    ("Median", .5), ("P75", .75), ("P95", .95), ("Max", 1)):
                report(f"  {label}: {score.quantile(quantile, interpolation='linear'):.6g}")
            report("  Score ranges (percentages of finite scores):")
            for lower, upper in ((0, .75), (.75, .9), (.9, .95), (.95, .99), (.99, 1)):
                # Include exactly 1 in the final bin; all other upper bounds are exclusive.
                upper_mask = score <= upper if upper == 1 else score < upper
                count = int(((score >= lower) & upper_mask).sum())
                closing = "]" if upper == 1 else ")"
                report(f"    [{lower:g}, {upper:g}{closing}: {count:,} ({count / len(score):.1%})")
            outside = int(((score < 0) | (score > 1)).sum())
            report(f"    Outside [0, 1]: {outside:,} ({outside / len(score):.1%})")

    if write_merged_to_bioplex_dir:
        # Export original fields plus source.
        destination = Path(output_dir) if output_dir is not None else (
            DEFAULT_OUTPUT_DIR if bioplex_dir.resolve() == DEFAULT_BIOPLEX_DIR else bioplex_dir
        )
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "BioPlex_HEK293T_merged_dedup"
        result.write_csv(output.with_suffix(".tsv"), separator="\t")
        result.write_parquet(output.with_suffix(".parquet"), compression="zstd")
        for suffix in (".tsv", ".parquet"):
            path = output.with_suffix(suffix)
            report(f"Wrote {path} ({path.stat().st_size:,} bytes)")

    report(f"Build completed in {perf_counter() - started:.2f} seconds")
    return result


def main_cli() -> None:
    """Merge release TSVs and export the deduplicated table to processed networks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bioplex-dir", type=Path, default=DEFAULT_BIOPLEX_DIR,
                        help="Directory containing the three BioPlex HEK293T release TSVs")
    parser.add_argument("--min-pint", type=float, default=0.0,
                        help="Minimum interaction probability (default: no filtering)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed build statistics")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # The function writes both TSV and Parquet; report the final interaction count.
    merged = merge_and_deduplicate_bioplex_hek293(
        args.bioplex_dir, min_pint=args.min_pint, verbose=not args.quiet, output_dir=args.output_dir,
    )
    print(f"Saved {len(merged):,} unique interactions in {args.output_dir}")


if __name__ == "__main__":
    main_cli()
