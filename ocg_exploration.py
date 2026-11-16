"""Deep exploration utilities for C OCG overlapping protein-interaction clusters."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import fisher_exact, hypergeom, mannwhitneyu

from clustering_explorer import RAW, saved_runs
from moonlighting import bh_adjust, significance_label


def saved_ocg_runs(root: Path = RAW) -> pl.DataFrame:
    """Return successful saved c.ocg runs enriched with their recorded OCG parameters."""
    runs = saved_runs(root)
    if runs.is_empty():
        return pl.DataFrame()
    ocg = runs.filter((pl.col('method') == 'c.ocg') & (pl.col('status') == 'success'))
    if ocg.is_empty():
        return pl.DataFrame()
    param_rows = []
    for row in ocg.iter_rows(named=True):
        directory = Path(row['directory'])
        params = {}
        try:
            audit = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            params = audit.get('job', {}).get('parameters', {}) or {}
        except (OSError, ValueError):
            params = {}
        param_rows.append({
            'directory': row['directory'],
            'init_class_sys': params.get('init_class_sys'),
            'cent_class_sys': params.get('cent_class_sys'),
            'max_class_card': params.get('max_class_card'),
            'min_class': params.get('min_class'),
        })
    return ocg.join(pl.from_dicts(param_rows, infer_schema_length=None), on='directory', how='left')


def load_ocg_memberships(directory: str | Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load (protein, community) memberships and compute per-protein membership counts."""
    path = Path(directory) / 'memberships.parquet'
    memberships = pl.read_parquet(path, columns=['protein', 'community']).unique()
    proteins = memberships.select('protein').unique()
    assigned = (
        memberships.filter(pl.col('community').is_not_null())
        .group_by('protein')
        .agg(
            pl.col('community').n_unique().alias('memberships'),
            pl.col('community').sort().cast(pl.String).str.join(';').alias('communities_str'),
        )
    )
    protein_summary = (
        proteins.join(assigned, on='protein', how='left')
        .with_columns(
            pl.col('memberships').fill_null(0).cast(pl.Int64),
            pl.col('communities_str').fill_null(''),
        )
        .with_columns(
            (pl.col('memberships') > 0).alias('is_assigned'),
            (pl.col('memberships') >= 2).alias('is_multi'),
        )
        .sort('memberships', 'protein', descending=[True, False])
    )
    return memberships, protein_summary


def membership_count_distribution(protein_summary: pl.DataFrame, cohort: set[str]) -> pl.DataFrame:
    """Exact k histogram; each protein contributes once to the run and its disjoint group."""
    summary = protein_summary.select('protein', 'memberships').unique(subset='protein')
    maximum = int(summary['memberships'].max() or 0)
    grid = pl.DataFrame({'memberships': range(maximum + 1)})
    frames = []
    for label, subset in (
        ('All run proteins', summary),
        ('Selected protein set', summary.filter(pl.col('protein').is_in(cohort))),
        ('Other run proteins', summary.filter(~pl.col('protein').is_in(cohort))),
    ):
        counts = subset.group_by('memberships').agg(pl.len().alias('proteins'))
        frames.append(grid.join(counts, on='memberships', how='left').with_columns(
            pl.col('proteins').fill_null(0),
            pl.lit(label).alias('group'),
            pl.lit(subset.height).alias('population'),
        ).with_columns(
            pl.when(pl.col('population') > 0)
            .then(pl.col('proteins') / pl.col('population'))
            .otherwise(None).alias('fraction')
        ))
    return pl.concat(frames)


def _deduplicated_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """Normalize undirected edges (a, b) with a < b and no self-loops."""
    return (
        edges.select('a', 'b')
        .filter((pl.col('a').is_not_null()) & (pl.col('b').is_not_null()) & (pl.col('a') != pl.col('b')))
        .with_columns(
            pl.when(pl.col('a') < pl.col('b')).then(pl.col('a')).otherwise(pl.col('b')).alias('u'),
            pl.when(pl.col('a') < pl.col('b')).then(pl.col('b')).otherwise(pl.col('a')).alias('v'),
        )
        .select(pl.col('u').alias('a'), pl.col('v').alias('b'))
        .unique()
    )


def protein_degrees(edges: pl.DataFrame) -> pl.DataFrame:
    """Compute undirected degree and floor(log2(degree)) bin for each protein."""
    dedup = _deduplicated_edges(edges)
    deg = (
        pl.concat([dedup.select(pl.col('a').alias('protein')), dedup.select(pl.col('b').alias('protein'))])
        .group_by('protein')
        .agg(pl.len().alias('degree'))
        .with_columns(pl.col('degree').log(2).floor().cast(pl.Int64).alias('degree_bin'))
    )
    return deg


def cluster_quality_metrics(
    memberships: pl.DataFrame,
    edges: pl.DataFrame,
    cohort: set[str] | list[str] | None = None,
) -> pl.DataFrame:
    """Compute internal density, conductance, overlap fraction, and optional cohort enrichment per cluster."""
    assigned = memberships.filter(pl.col('community').is_not_null()).select('protein', 'community').unique()
    if assigned.is_empty():
        return pl.DataFrame()

    dedup = _deduplicated_edges(edges)
    protein_counts = assigned.group_by('protein').agg(pl.len().alias('_k'))
    cluster_nodes = assigned.join(protein_counts, on='protein', how='left')

    node_stats = cluster_nodes.group_by('community').agg(
        pl.len().alias('size'),
        (pl.col('_k') >= 2).sum().alias('shared_proteins'),
    )

    # Map edges to endpoints' communities to count internal and boundary edges cleanly.
    deg = protein_degrees(dedup).select('protein', 'degree')
    total_deg = (
        cluster_nodes.join(deg, on='protein', how='left')
        .with_columns(pl.col('degree').fill_null(0))
        .group_by('community')
        .agg(pl.col('degree').sum().alias('degree_sum'))
    )

    internal = (
        dedup.join(assigned.rename({'protein': 'a', 'community': 'ca'}), on='a', how='inner')
        .join(assigned.rename({'protein': 'b', 'community': 'cb'}), on='b', how='inner')
        .filter(pl.col('ca') == pl.col('cb'))
        .group_by(pl.col('ca').alias('community'))
        .agg(pl.len().alias('internal_edges'))
    )

    metrics = (
        node_stats.join(total_deg, on='community', how='left')
        .join(internal, on='community', how='left')
        .with_columns(
            pl.col('internal_edges').fill_null(0).cast(pl.Int64),
            pl.col('degree_sum').fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (pl.col('size') * (pl.col('size') - 1) / 2).alias('possible_edges'),
            (pl.col('degree_sum') - 2 * pl.col('internal_edges')).clip(lower_bound=0).alias('boundary_edges'),
            (pl.col('shared_proteins') / pl.col('size')).alias('shared_fraction'),
        )
        .with_columns(
            pl.when(pl.col('possible_edges') > 0)
            .then(pl.col('internal_edges') / pl.col('possible_edges'))
            .otherwise(None)
            .alias('internal_density'),
            pl.when(pl.col('degree_sum') > 0)
            .then(pl.col('boundary_edges') / pl.col('degree_sum'))
            .otherwise(None)
            .alias('conductance'),
        )
    )

    if cohort is not None:
        all_proteins = memberships.select('protein').unique()
        present = set(all_proteins['protein'].to_list()) & set(cohort)
        n_pop = all_proteins.height
        k_pop = len(present)
        hits_df = (
            assigned.with_columns(pl.col('protein').is_in(present).alias('_hit'))
            .group_by('community')
            .agg(pl.col('_hit').sum().alias('cohort_hits'))
        )
        metrics = metrics.join(hits_df, on='community', how='left').with_columns(
            pl.col('cohort_hits').fill_null(0).cast(pl.Int64)
        )
        if n_pop > 0 and k_pop > 0:
            pvalues = hypergeom.sf(
                metrics['cohort_hits'].to_numpy().astype(np.int64) - 1,
                n_pop,
                k_pop,
                metrics['size'].to_numpy().astype(np.int64),
            )
            qvalues = bh_adjust(pvalues)
            metrics = (
                metrics.with_columns(
                    (pl.col('cohort_hits') / pl.col('size')).alias('cohort_fraction'),
                    (pl.col('size') * k_pop / n_pop).alias('expected_hits'),
                    pl.Series('pvalue', pvalues),
                    pl.Series('qvalue', qvalues),
                    pl.Series('significance', [significance_label(q) for q in qvalues]),
                )
                .with_columns(
                    pl.when(pl.col('expected_hits') > 0)
                    .then(pl.col('cohort_hits') / pl.col('expected_hits'))
                    .otherwise(None)
                    .alias('fold_enrichment')
                )
            )
        else:
            metrics = metrics.with_columns(
                pl.lit(0.0).alias('cohort_fraction'),
                pl.lit(0.0).alias('expected_hits'),
                pl.lit(None, dtype=pl.Float64).alias('fold_enrichment'),
                pl.lit(None, dtype=pl.Float64).alias('pvalue'),
                pl.lit(None, dtype=pl.Float64).alias('qvalue'),
                pl.lit('n/a').alias('significance'),
            )

    return metrics.sort('size', 'community', descending=[True, False])


def degree_matched_multi_membership(
    protein_summary: pl.DataFrame,
    edges: pl.DataFrame,
    cohort: set[str] | list[str],
    *,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[dict, pl.DataFrame]:
    """Compare k>=2 and mean memberships of cohort vs. degree-bin-matched network background."""
    deg = protein_degrees(edges)
    df = (
        protein_summary.join(deg, on='protein', how='left')
        .with_columns(
            pl.col('degree').fill_null(0),
            pl.col('degree_bin').fill_null(-1),
            pl.col('protein').is_in(set(cohort)).alias('in_cohort'),
        )
    )

    cohort_df = df.filter(pl.col('in_cohort'))
    bg_df = df.filter(~pl.col('in_cohort'))
    if cohort_df.is_empty() or bg_df.is_empty():
        return {'valid': False, 'reason': 'Cohort or background has zero proteins in this network.'}, pl.DataFrame()

    # Degree-matched sampling across bins shared by cohort and background.
    bg_by_bin: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row in bg_df.group_by('degree_bin').agg(
        pl.col('is_multi').cast(pl.Float64).alias('multi'),
        pl.col('memberships').cast(pl.Float64).alias('k'),
    ).iter_rows(named=True):
        bg_by_bin[int(row['degree_bin'])] = (
            np.asarray(row['multi'], dtype=float),
            np.asarray(row['k'], dtype=float),
        )

    matched_bins = []
    for row in cohort_df.group_by('degree_bin').agg(pl.len().alias('n_cohort')).iter_rows(named=True):
        b = int(row['degree_bin'])
        if b in bg_by_bin:
            matched_bins.append((b, int(row['n_cohort'])))

    if not matched_bins:
        return {'valid': False, 'reason': 'No overlapping degree bins between cohort and background.'}, pl.DataFrame()

    obs_sub = cohort_df.filter(pl.col('degree_bin').is_in([b for b, _ in matched_bins]))
    obs_multi_frac = float(obs_sub['is_multi'].mean())
    obs_mean_k = float(obs_sub['memberships'].mean())
    n_matched = obs_sub.height

    rng = np.random.default_rng(seed)
    perm_multi = np.zeros(n_perm, dtype=float)
    perm_k = np.zeros(n_perm, dtype=float)

    k_max = int(df['memberships'].max())
    perm_hist = np.zeros((n_perm, k_max + 1), dtype=float)
    for b, n_draw in matched_bins:
        bg_multi, bg_k = bg_by_bin[b]
        replace = n_draw > len(bg_multi)
        for i in range(n_perm):
            idx = rng.choice(len(bg_multi), size=n_draw, replace=replace)
            perm_multi[i] += bg_multi[idx].sum()
            perm_k[i] += bg_k[idx].sum()
            perm_hist[i] += np.bincount(bg_k[idx].astype(int), minlength=k_max + 1)

    perm_hist /= n_matched
    observed_hist = np.bincount(obs_sub['memberships'].to_numpy().astype(int), minlength=k_max + 1) / n_matched
    distribution = [dict(k=k, observed=float(observed_hist[k]),
                         null_mean=float(perm_hist[:, k].mean()),
                         null_low=float(np.quantile(perm_hist[:, k], .025)),
                         null_high=float(np.quantile(perm_hist[:, k], .975)))
                    for k in range(k_max + 1)]
    perm_multi /= n_matched
    perm_k /= n_matched

    emp_p_multi = float((np.sum(perm_multi >= obs_multi_frac) + 1) / (n_perm + 1))
    emp_p_k = float((np.sum(perm_k >= obs_mean_k) + 1) / (n_perm + 1))
    q_multi, q_k = bh_adjust([emp_p_multi, emp_p_k])

    summary = {
        'valid': True,
        'membership_distribution': distribution,
        'n_cohort_total': cohort_df.height,
        'n_cohort_matched': n_matched,
        'n_background_total': bg_df.height,
        'obs_multi_fraction': obs_multi_frac,
        'matched_null_multi_mean': float(perm_multi.mean()),
        'matched_null_multi_sd': float(perm_multi.std(ddof=1)) if n_perm > 1 else 0.0,
        'multi_pvalue': emp_p_multi,
        'multi_qvalue': float(q_multi),
        'multi_significance': significance_label(q_multi),
        'obs_mean_k': obs_mean_k,
        'matched_null_k_mean': float(perm_k.mean()),
        'matched_null_k_sd': float(perm_k.std(ddof=1)) if n_perm > 1 else 0.0,
        'k_pvalue': emp_p_k,
        'k_qvalue': float(q_k),
        'k_significance': significance_label(q_k),
        'n_perm': n_perm,
    }

    # Per-degree-bin Fisher exact tests (cohort k>=2 vs background k>=2 within each bin).
    bin_rows = []
    for b in sorted(df['degree_bin'].unique().to_list()):
        if b < 0:
            continue
        sub = df.filter(pl.col('degree_bin') == b)
        c_sub = sub.filter(pl.col('in_cohort'))
        b_sub = sub.filter(~pl.col('in_cohort'))
        if c_sub.is_empty() or b_sub.is_empty():
            continue
        c_multi = int(c_sub['is_multi'].sum())
        b_multi = int(b_sub['is_multi'].sum())
        _, pval = fisher_exact(
            [[c_multi, c_sub.height - c_multi], [b_multi, b_sub.height - b_multi]],
            alternative='greater',
        )
        bin_rows.append({
            'degree_bin': b,
            'degree_range': f'{2**b}–{2**(b + 1) - 1}',
            'cohort_n': c_sub.height,
            'cohort_multi': c_multi,
            'cohort_multi_fraction': c_multi / c_sub.height,
            'background_n': b_sub.height,
            'background_multi': b_multi,
            'background_multi_fraction': b_multi / b_sub.height,
            'pvalue': float(pval),
        })

    if bin_rows:
        bin_df = pl.from_dicts(bin_rows)
        qvals = bh_adjust(bin_df['pvalue'].to_list())
        bin_df = bin_df.with_columns(
            pl.Series('qvalue', qvals),
            pl.Series('significance', [significance_label(q) for q in qvals]),
        ).sort('degree_bin')
    else:
        bin_df = pl.DataFrame()

    return summary, bin_df


def cluster_overlap_graph(
    memberships: pl.DataFrame,
    *,
    min_shared: int = 1,
    max_edges: int | None = 5000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build cluster-cluster overlap edges (Jaccard & overlap coefficient) and bridge stats."""
    assigned = memberships.filter(pl.col('community').is_not_null()).select('protein', 'community').unique()
    if assigned.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    sizes = assigned.group_by('community').agg(pl.len().alias('size'))
    pairs = (
        assigned.rename({'community': 'community_a'})
        .join(assigned.rename({'community': 'community_b'}), on='protein', how='inner')
        .filter(pl.col('community_a') < pl.col('community_b'))
        .group_by('community_a', 'community_b')
        .agg(pl.len().alias('shared_proteins'))
        .filter(pl.col('shared_proteins') >= min_shared)
    )
    if pairs.is_empty():
        return pl.DataFrame(), sizes.with_columns(
            pl.lit(0, dtype=pl.Int64).alias('overlap_neighbors'),
            pl.lit(0, dtype=pl.Int64).alias('total_shared_links'),
        )

    pairs = (
        pairs.join(sizes.rename({'community': 'community_a', 'size': 'size_a'}), on='community_a', how='left')
        .join(sizes.rename({'community': 'community_b', 'size': 'size_b'}), on='community_b', how='left')
        .with_columns(
            (pl.col('shared_proteins') / (pl.col('size_a') + pl.col('size_b') - pl.col('shared_proteins'))).alias('jaccard'),
            (pl.col('shared_proteins') / pl.min_horizontal('size_a', 'size_b')).alias('overlap_coef'),
        )
        .sort('shared_proteins', 'jaccard', descending=[True, True])
    )

    bridge = (
        pl.concat([
            pairs.select(pl.col('community_a').alias('community'), 'shared_proteins'),
            pairs.select(pl.col('community_b').alias('community'), 'shared_proteins'),
        ])
        .group_by('community')
        .agg(
            pl.len().alias('overlap_neighbors'),
            pl.col('shared_proteins').sum().alias('total_shared_links'),
        )
        .join(sizes, on='community', how='right')
        .with_columns(
            pl.col('overlap_neighbors').fill_null(0).cast(pl.Int64),
            pl.col('total_shared_links').fill_null(0).cast(pl.Int64),
        )
        .sort('overlap_neighbors', 'total_shared_links', descending=[True, True])
    )

    return (pairs.head(max_edges) if max_edges is not None else pairs), bridge


def inter_cluster_functional_divergence(
    memberships: pl.DataFrame,
    enrichment_df: pl.DataFrame,
    cohort: set[str] | list[str] | None = None,
    *,
    q_cutoff: float = 0.05,
    subset: str = 'full',
    libraries: list[str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Measure functional divergence (1 - mean pairwise Jaccard of enriched terms) across each multi-cluster protein's clusters."""
    if enrichment_df.is_empty():
        return pl.DataFrame(), {'valid': False, 'reason': 'No enrichment terms available.'}

    terms = enrichment_df.filter((pl.col('subset') == subset) & (pl.col('qvalue') < q_cutoff))
    if libraries:
        terms = terms.filter(pl.col('library').is_in(libraries))
    if terms.is_empty():
        return pl.DataFrame(), {'valid': False, 'reason': 'No significant terms pass the selected q-value/library filters.'}

    cluster_terms: dict[int, set[str]] = {}
    cluster_top_label: dict[int, str] = {}
    for row in (
        terms.sort('qvalue')
        .group_by('community')
        .agg(
            pl.col('term_id').unique().alias('term_ids'),
            pl.col('term_name').first().alias('top_term'),
            pl.col('library').first().alias('top_library'),
        )
        .iter_rows(named=True)
    ):
        cid = int(row['community'])
        cluster_terms[cid] = set(row['term_ids'])
        cluster_top_label[cid] = f"C{cid}: {row['top_term']} ({row['top_library']})"

    multi_members = (
        memberships.filter(pl.col('community').is_not_null())
        .unique()
        .group_by('protein')
        .agg(pl.col('community').sort().alias('communities'))
        .filter(pl.col('communities').list.len() >= 2)
    )

    cohort_set = set(cohort) if cohort is not None else set()
    rows = []
    for row in multi_members.iter_rows(named=True):
        annotated = [int(c) for c in row['communities'] if int(c) in cluster_terms]
        if len(annotated) < 2:
            continue
        jaccards = []
        union_terms: set[str] = set()
        for i in range(len(annotated)):
            s_i = cluster_terms[annotated[i]]
            union_terms |= s_i
            for j in range(i + 1, len(annotated)):
                s_j = cluster_terms[annotated[j]]
                union = len(s_i | s_j)
                jaccards.append(len(s_i & s_j) / union if union else 0.0)
        mean_jaccard = float(np.mean(jaccards))
        rows.append({
            'protein': row['protein'],
            'in_cohort': row['protein'] in cohort_set,
            'total_clusters': len(row['communities']),
            'annotated_clusters': len(annotated),
            'mean_pairwise_jaccard': mean_jaccard,
            'divergence_score': 1.0 - mean_jaccard,
            'unique_terms_total': len(union_terms),
            'cluster_summaries': ' | '.join(cluster_top_label[c] for c in annotated[:5]),
        })

    if not rows:
        return pl.DataFrame(), {'valid': False, 'reason': 'No multi-cluster proteins had >=2 annotated clusters.'}

    result_df = pl.from_dicts(rows).sort('divergence_score', 'annotated_clusters', descending=[True, True])
    summary: dict = {'valid': True, 'n_proteins': result_df.height}

    if cohort is not None:
        in_c = result_df.filter(pl.col('in_cohort'))['divergence_score'].to_numpy()
        out_c = result_df.filter(~pl.col('in_cohort'))['divergence_score'].to_numpy()
        if len(in_c) > 0 and len(out_c) > 0:
            u_stat, mw_p = mannwhitneyu(in_c, out_c, alternative='greater', method='asymptotic')
            summary.update({
                'cohort_n': int(len(in_c)),
                'background_n': int(len(out_c)),
                'cohort_median_divergence': float(np.median(in_c)),
                'background_median_divergence': float(np.median(out_c)),
                'rank_biserial': float(2 * u_stat / (len(in_c) * len(out_c)) - 1),
                'pvalue': float(mw_p),
                'significance': significance_label(float(mw_p)),
            })
    return result_df, summary


def cross_network_ocg_comparison(
    ocg_runs: pl.DataFrame,
    cohort: set[str] | list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare c.ocg multi-membership (k>=2) conservation across all networks with saved c.ocg runs."""
    if ocg_runs.is_empty():
        return pl.DataFrame(), pl.DataFrame()

    # Keep the newest run per network if multiple exist.
    per_network = ocg_runs.sort('run', descending=True).unique(subset=['network'], keep='first')
    frames = []
    for row in per_network.iter_rows(named=True):
        _, summary = load_ocg_memberships(row['directory'])
        frames.append(
            summary.select(
                'protein',
                pl.lit(row['network']).alias('network'),
                'memberships',
                'is_multi',
            )
        )
    if not frames:
        return pl.DataFrame(), pl.DataFrame()

    long_df = pl.concat(frames)
    cohort_set = set(cohort)
    protein_conserved = (
        long_df.group_by('protein')
        .agg(
            pl.len().alias('networks_present'),
            pl.col('is_multi').sum().alias('networks_multi'),
            pl.col('memberships').mean().alias('mean_memberships'),
            pl.col('network').filter(pl.col('is_multi')).sort().str.join(', ').alias('multi_networks'),
        )
        .with_columns(pl.col('protein').is_in(cohort_set).alias('in_cohort'))
        .sort('networks_multi', 'networks_present', 'mean_memberships', descending=[True, True, True])
    )

    # Enrichment of cohort across conservation thresholds (multi in >= 1, >= 2, >= 3 networks).
    max_net = int(protein_conserved['networks_multi'].max() or 0)
    n_total = protein_conserved.height
    k_total = int(protein_conserved['in_cohort'].sum())
    enrich_rows = []
    for thresh in range(1, max_net + 1):
        sub = protein_conserved.filter(pl.col('networks_multi') >= thresh)
        hits = int(sub['in_cohort'].sum())
        pval = float(hypergeom.sf(hits - 1, n_total, k_total, sub.height)) if (n_total and k_total and sub.height) else 1.0
        expected = sub.height * k_total / n_total if n_total else 0.0
        enrich_rows.append({
            'min_multi_networks': thresh,
            'proteins': sub.height,
            'cohort_hits': hits,
            'cohort_fraction': hits / sub.height if sub.height else 0.0,
            'expected_hits': expected,
            'fold_enrichment': hits / expected if expected > 0 else None,
            'pvalue': pval,
        })

    if enrich_rows:
        enrich_df = pl.from_dicts(enrich_rows)
        qvals = bh_adjust(enrich_df['pvalue'].to_list())
        enrich_df = enrich_df.with_columns(
            pl.Series('qvalue', qvals),
            pl.Series('significance', [significance_label(q) for q in qvals]),
        )
    else:
        enrich_df = pl.DataFrame()

    return protein_conserved, enrich_df


def ego_cluster_subgraph(
    protein: str,
    memberships: pl.DataFrame,
    edges: pl.DataFrame,
    *,
    max_neighbors: int = 80,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract the 1-hop ego neighborhood of a protein annotated with OCG cluster memberships."""
    dedup = _deduplicated_edges(edges)
    neighbors_df = pl.concat([
        dedup.filter(pl.col('a') == protein).select(pl.col('b').alias('protein')),
        dedup.filter(pl.col('b') == protein).select(pl.col('a').alias('protein')),
    ]).unique()

    _, summary = load_ocg_memberships(memberships) if isinstance(memberships, (str, Path)) else (
        memberships,
        memberships.filter(pl.col('community').is_not_null())
        .group_by('protein')
        .agg(
            pl.col('community').n_unique().alias('memberships'),
            pl.col('community').sort().cast(pl.String).str.join(';').alias('communities_str'),
        ),
    )

    # Prioritize multi-cluster / assigned neighbors if capped.
    neighbor_info = (
        neighbors_df.join(summary, on='protein', how='left')
        .with_columns(
            pl.col('memberships').fill_null(0).cast(pl.Int64),
            pl.col('communities_str').fill_null('unassigned'),
        )
        .sort('memberships', 'protein', descending=[True, False])
        .head(max_neighbors)
    )

    center_info = (
        pl.DataFrame({'protein': [protein]})
        .join(summary, on='protein', how='left')
        .with_columns(
            pl.col('memberships').fill_null(0).cast(pl.Int64),
            pl.col('communities_str').fill_null('unassigned'),
        )
    )
    nodes = pl.concat([
        center_info.with_columns(pl.lit(True).alias('is_center')),
        neighbor_info.with_columns(pl.lit(False).alias('is_center')),
    ]).unique(subset=['protein'], keep='first')

    node_set = set(nodes['protein'].to_list())
    sub_edges = dedup.filter((pl.col('a').is_in(node_set)) & (pl.col('b').is_in(node_set)))

    # Annotate whether each edge shares an OCG cluster.
    assigned = memberships.filter(pl.col('community').is_not_null()).select('protein', 'community').unique()
    shared_edge_counts = (
        sub_edges.join(assigned.rename({'protein': 'a', 'community': 'ca'}), on='a', how='inner')
        .join(assigned.rename({'protein': 'b', 'community': 'cb'}), on='b', how='inner')
        .filter(pl.col('ca') == pl.col('cb'))
        .group_by('a', 'b')
        .agg(pl.len().alias('shared_clusters'))
    )
    sub_edges = sub_edges.join(shared_edge_counts, on=['a', 'b'], how='left').with_columns(
        pl.col('shared_clusters').fill_null(0).cast(pl.Int64)
    )
    return nodes, sub_edges
