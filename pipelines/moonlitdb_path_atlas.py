"""Static topology evidence for the saved MoonLitDB shortest-path runs.

Run from the repository root. No raw inputs or existing analyses are changed.
"""
import hashlib
import json
import random
from pathlib import Path
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'moonlighting.py').exists())
sys.path.insert(0, str(REPO))
import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import polars as pl
from scipy import sparse
from moonlighting import significance_label
from ocg_exploration import saved_ocg_runs
from pipelines.precompute_monte_carlo import load_cohort


def _ocg_lcc_membership(ocg_dir: Path, labels: list[str]):
    """Load saved c.ocg memberships aligned to the LCC node order."""
    mem = (
        pl.read_parquet(ocg_dir / 'memberships.parquet', columns=['protein', 'community'])
        .filter(pl.col('community').is_not_null())
        .unique()
    )
    label_to_idx = {name: i for i, name in enumerate(labels)}
    mem = mem.filter(pl.col('protein').is_in(labels))
    c_ids = sorted(mem['community'].unique().to_list())
    c_map = {cid: j for j, cid in enumerate(c_ids)}
    rows = [label_to_idx[p] for p in mem['protein'].to_list()]
    cols = [c_map[c] for c in mem['community'].to_list()]
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int16), (rows, cols)),
        shape=(len(labels), len(c_ids)),
    )
    k_vec = np.asarray(matrix.sum(axis=1)).ravel().astype(int)
    grouped = (
        mem.group_by('protein')
        .agg(pl.col('community').sort().cast(pl.String).str.join(';').alias('communities'))
    )
    comm_map = dict(zip(grouped['protein'].to_list(), grouped['communities'].to_list()))
    comm_list = [comm_map.get(name, '') for name in labels]
    return matrix, k_vec, comm_list, len(c_ids)


def main():
    root = Path('data')
    out = root / 'analysis_runs/moonlitdb_path_atlas'
    plots = root / 'plots/moonlitdb_path_atlas'
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(root, 'moonlitdb')
    ocg_runs = saved_ocg_runs(root / 'analysis_runs/clustering_raw')
    names = ['HuRI', 'BioPlex', 'huMAP2_0.05', 'huMAP2_0.95', 'TAGC_binary2025']
    fig, axes = plt.subplots(len(names), 2, figsize=(15, 23))
    rows, provenance = [], []
    ig.set_random_number_generator(random.Random(42))
    for row, name in enumerate(names):
        prefix = root / 'analysis_runs' / f'moonlitdb_{name}_shortest_path'
        meta = json.loads(Path(str(prefix) + '.json').read_text())
        path = Path(meta['network_path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta['network_sha256'] or meta['scope'] != 'lcc':
            raise ValueError(f'Stale or incompatible saved run: {name}')
        matching_ocg = []
        for ocg_row in ocg_runs.filter(pl.col('network') == name).sort('run', descending=True).iter_rows(named=True):
            ocg_dir = Path(ocg_row['directory'])
            ocg_result = json.loads((ocg_dir / 'result.json').read_text(encoding='utf-8'))
            ocg_sha = ocg_result.get('job', {}).get('provenance', {}).get('input', {}).get('sha256')
            if ocg_sha == digest and (ocg_dir / 'memberships.parquet').is_file():
                matching_ocg.append((ocg_row['run'], ocg_dir))
        if not matching_ocg:
            raise ValueError(f'No matching c.ocg run with SHA-256 {digest} found for {name}')
        ocg_run_id, ocg_dir = matching_ocg[0]
        summary = pl.read_csv(str(prefix) + '_summary.csv').row(0, named=True)
        frame = pl.read_parquet(path).select('a', 'b').filter(pl.col('a') != pl.col('b')).unique()
        graph = ig.Graph.TupleList(frame.iter_rows(), directed=False).simplify()
        graph = graph.connected_components().giant()
        labels = graph.vs['name']
        mask = np.array([bool(set(n.split('|')) & cohort) for n in labels])
        selected = np.flatnonzero(mask)
        assert len(selected) == summary['query_nodes']
        distances = np.array(graph.distances(source=selected, target=selected))
        tri = np.triu_indices(len(selected), 1)
        values = distances[tri]
        assert np.isclose(values.mean(), summary['observed'])
        partition = np.array(graph.community_multilevel().membership)
        same = partition[selected][tri[0]] == partition[selected][tri[1]]
        ocg_matrix, ocg_k, ocg_comms, ocg_clusters_lcc = _ocg_lcc_membership(ocg_dir, labels)
        same_ocg = (ocg_matrix[selected] @ ocg_matrix[selected].T).toarray()[tri] > 0
        # Decompose the saved null using exactly its seeded degree-bin draws.
        order = np.argsort(labels)
        sorted_labels = np.array(labels)[order]
        degree = np.array(graph.degree())[order]
        bins = np.floor(np.log2(np.maximum(degree, 1))).astype(int)
        query_sorted = np.flatnonzero(mask[order])
        buckets, counts = np.unique(bins[query_sorted], return_counts=True)
        pools = [(np.flatnonzero(bins == b), count) for b, count in zip(buckets, counts)]
        cached = np.load(Path(meta['distance_cache']) / 'distances.npy', mmap_mode='r')
        ready = np.load(Path(meta['distance_cache']) / 'ready.npy', mmap_mode='r')
        assert cached.shape == (len(labels), len(labels))
        assert np.isclose(cached[np.ix_(query_sorted, query_sorted)][tri].mean(), values.mean())
        rng = np.random.default_rng(meta['seed'])
        null_parts, null_ocg_parts = [], []
        for _ in range(meta['simulations']):
            sample = np.concatenate([rng.choice(pool, count, replace=False) for pool, count in pools])
            assert ready[sample].all(), 'Incomplete cached distance rows'
            d = cached[np.ix_(sample, sample)][tri]
            sample_orig = order[sample]
            groups = partition[sample_orig]
            within = groups[tri[0]] == groups[tri[1]]
            null_parts.append((d.mean(), within.mean(), d[within].sum()/len(d), d[~within].sum()/len(d)))
            ocg_within = (ocg_matrix[sample_orig] @ ocg_matrix[sample_orig].T).toarray()[tri] > 0
            null_ocg_parts.append((
                ocg_within.mean(),
                d[ocg_within].sum()/len(d),
                d[~ocg_within].sum()/len(d),
                float((ocg_k[sample_orig] >= 2).mean()),
                float(ocg_k[sample_orig].mean()),
            ))
        null_parts = np.array(null_parts)
        null_ocg_parts = np.array(null_ocg_parts)
        assert np.isclose(null_parts[:, 0].mean(), summary['null_mean'])
        ocg_within_contrib = float(values[same_ocg].sum()/len(values) - null_ocg_parts[:, 1].mean())
        ocg_between_contrib = float(values[~same_ocg].sum()/len(values) - null_ocg_parts[:, 2].mean())
        assert np.isclose(ocg_within_contrib + ocg_between_contrib, summary['effect_observed_minus_null'])
        internal = graph.induced_subgraph(selected.tolist())
        positions = np.array(graph.layout_fruchterman_reingold(niter=400, grid='grid').coords)
        edges = np.array(graph.get_edgelist())
        incident = mask[edges].any(axis=1)
        shown = np.unique(edges[incident])
        for col, edge_indices in enumerate([np.ones(len(edges), dtype=bool), incident]):
            ax = axes[row, col]
            ax.add_collection(LineCollection(positions[edges[edge_indices]], colors='#bfc3c7', linewidths=.18, alpha=.20))
            visible = np.arange(graph.vcount()) if col == 0 else shown
            grey = visible[~mask[visible]]
            ax.scatter(*positions[grey].T, s=3, c='#b5b9bd', linewidths=0, alpha=.55)
            ax.scatter(*positions[selected].T, s=3, c='#0072B2', linewidths=0, alpha=1)
            ax.autoscale_view()
            ax.set_aspect('equal')
            ax.axis('off')
            title = f'{name} | ' + ('Entire largest component' if col == 0 else 'Edges touching MoonLitDB')
            ax.set_title(title, fontsize=12, loc='left', weight='bold')
            ax.text(0, -.025, f'{graph.vcount():,} proteins; {graph.ecount():,} edges; {len(selected)} blue proteins' if col == 0
                    else f'Mean path {values.mean():.3f} vs {summary["null_mean"]:.3f} expected; '
                         f'Δ {summary["effect_observed_minus_null"]:+.3f} {significance_label(summary["fdr_bh"])}',
                    transform=ax.transAxes, fontsize=10)
        record = dict(network=name, **{k: summary[k] for k in ['observed', 'null_mean', 'effect_observed_minus_null', 'fdr_bh']},
                      nodes=graph.vcount(), edges=graph.ecount(), moonlitdb_nodes=len(selected),
                      direct_pairs=int((values == 1).sum()), two_hop_pairs=int((values == 2).sum()),
                      fraction_pairs_ge_4=float((values >= 4).mean()),
                      fraction_same_community=float(same.mean()),
                      within_community_mean=float(values[same].mean()), between_community_mean=float(values[~same].mean()),
                      null_fraction_same_community=float(null_parts[:, 1].mean()),
                      within_pair_contribution_to_delta=float(values[same].sum()/len(values)-null_parts[:, 2].mean()),
                      between_pair_contribution_to_delta=float(values[~same].sum()/len(values)-null_parts[:, 3].mean()),
                      ocg_run=ocg_run_id, ocg_clusters_lcc=int(ocg_clusters_lcc),
                      ocg_multi_fraction=float((ocg_k[selected] >= 2).mean()),
                      null_ocg_multi_fraction=float(null_ocg_parts[:, 3].mean()),
                      ocg_mean_memberships=float(ocg_k[selected].mean()),
                      null_ocg_mean_memberships=float(null_ocg_parts[:, 4].mean()),
                      fraction_same_ocg_cluster=float(same_ocg.mean()),
                      within_ocg_mean=float(values[same_ocg].mean()) if same_ocg.any() else float('nan'),
                      between_ocg_mean=float(values[~same_ocg].mean()) if (~same_ocg).any() else float('nan'),
                      null_fraction_same_ocg_cluster=float(null_ocg_parts[:, 0].mean()),
                      ocg_within_pair_contribution_to_delta=ocg_within_contrib,
                      ocg_between_pair_contribution_to_delta=ocg_between_contrib,
                      induced_components=len(internal.connected_components()),
                      largest_induced_component=max(internal.connected_components().sizes()))
        rows.append(record)
        pl.DataFrame({'hops': values.astype(int)}).group_by('hops').len().sort('hops').write_csv(out / f'{name}_hops.csv')
        pl.DataFrame({'protein': labels, 'moonlitdb': mask, 'x': positions[:, 0], 'y': positions[:, 1],
                      'community': partition, 'ocg_memberships': ocg_k, 'ocg_communities': ocg_comms}).write_parquet(out / f'{name}_positions.parquet')
        provenance.append(dict(network=name, network_sha256=digest, metadata=meta,
                               ocg_run=ocg_run_id, ocg_result_path=str(ocg_dir / 'result.json')))
        print(json.dumps(record), flush=True)
    fig.suptitle('Where MoonLitDB proteins sit in the interactomes', fontsize=21, weight='bold', y=.995)
    fig.text(.06, .968, 'Blue = MoonLitDB; grey = other proteins. Every node has the same size. Each row preserves the full-graph layout.', fontsize=11)
    fig.text(.06, .012, 'Paths use all edges in the largest component. Right panels hide grey–grey edges and are not used to calculate distances.\n'
             'Saved tests: 1,000 size/degree-bin matched random sets drawn from the full component (including MoonLitDB); two-sided +1 empirical test.\n'
             'BH correction across all requested shortest-path network/metric tests: * q<.05, ** q<.01, *** q<.001, **** q<.0001; ns = non-significant.\n'
             'Layout and community summaries are descriptive; geometric separation is not a measurement of hop distance.', fontsize=10)
    fig.subplots_adjust(top=.945, bottom=.065, hspace=.25, wspace=.12)
    fig.savefig(plots / 'moonlitdb_topology.png', dpi=180, facecolor='white')
    plt.close(fig)
    pl.DataFrame(rows).write_csv(out / 'topology_evidence.csv')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
