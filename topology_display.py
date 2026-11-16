"""Static topology boxplots using the same full-data tests as the interactive view."""
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from moonlighting import significance_label


METRIC_GUIDES = {
    'laplacian_centrality': ('Laplacian centrality', 'Relative drop in unweighted Laplacian energy '
        'when this protein is removed: [k² + k + 2Σ(neighbor degree)] / Σ(all nodes: degree² + degree). '
        'Higher values indicate more local structural impact; lower values indicate less. Isolates score '
        'zero. This is a degree-dependent structural score, not a simulated biological knockout effect.'),
    'triangles': ('Triangle count', 'Number of connected pairs of neighbors, each forming a triangle '
        'with the protein. Higher values indicate more closed interaction triples; zero means none. '
        'Unlike the clustering coefficient this is a count and strongly depends on degree. Triangles '
        'do not establish simultaneous binding or a protein complex.'),
    'effective_size': ('Effective size', 'Unweighted structural-hole measure: k − 2T/k, where k is '
        'degree and T is the triangle count. Higher values indicate more nonredundant neighbors; lower '
        'values indicate more connections among neighbors. Isolates are undefined and excluded from '
        'tests. This measure depends on degree and does not demonstrate functional diversity.'),
    'neighborhood_efficiency': ('Neighborhood efficiency', 'Effective size divided by degree: '
        '1 − 2T/k². Higher values indicate less redundant neighbors relative to the number of partners; '
        'lower values indicate more redundancy. Isolates are undefined and excluded from tests. '
        'A leaf scores 1 despite having only one partner. This is Burt’s ego-network efficiency, '
        'not shortest-path local efficiency.'),
    'degree': ('Degree', 'Number of distinct interaction partners in the retained, unweighted graph. '
               'Higher values mean more partners; lower values mean fewer.'),
    'strength': ('Strength', 'Sum of retained interaction weights for a protein. Higher values mean '
                 'greater total interaction weight; lower values mean less. When every edge has weight 1, '
                 'strength equals degree. Weight scales depend on the source network and are not directly '
                 'comparable across assays.'),
    'pagerank': ('PageRank', 'Weighted random-walk importance with restart. Higher values indicate '
                 'more probability of visiting a protein through the network; lower values indicate less. '
                 'Scores are relative to each network and depend on its size, weights and connectivity.'),
    'mean_neighbor_degree': ('Mean neighbor degree', 'Average unweighted degree of a protein’s '
                             'interaction partners; zero is assigned to isolated proteins. Higher values '
                             'mean neighbors have more partners; lower values mean neighbors have fewer. '
                             'This describes the neighborhood rather than the protein’s own degree.'),
    'eigenvector': ('Eigenvector centrality', 'Importance from connections to other highly scoring '
                    'proteins, using the leading eigenvector of the unweighted adjacency matrix with '
                    'L2 normalization. Higher values indicate stronger connection to that dominant '
                    'network structure. Disconnected components may receive near-zero scores; tied '
                    'largest eigenvalues can make scores non-unique.'),
    'bridging_coefficient': ('Bridging coefficient', 'Local bridging factor: (1/k) divided by the '
                             'sum of 1/k_neighbor over all interaction partners, using unweighted degrees. '
                             'Zero is assigned when the denominator is zero. Higher values favor proteins '
                             'with few partners whose partners have many connections; lower values indicate '
                             'less of this local pattern. It does not by itself establish that a protein '
                             'connects distinct communities.'),
    'bridging_centrality': ('Bridging centrality', 'Normalized unweighted betweenness multiplied by '
                            'the bridging coefficient. Higher values combine shortest-path mediation '
                            'with a local pattern of few partners connected to highly connected neighbors. '
                            'Low or zero values mean at least one factor is small or zero. In this overview '
                            'betweenness is exact. A high score does not establish a biological bridge '
                            'or validate moonlighting.'),
    'core_number': ('K-coreness', 'Largest k for which a protein belongs to a k-core: a maximal subgraph '
                    'where every node has at least k neighbors inside that subgraph. Higher values mean '
                    'membership in a more densely connected core; lower values indicate peripheral membership.'),
    'clustering': ('Clustering coefficient', 'Fraction of possible pairs of a protein’s neighbors that '
                   'are connected: 2T/[k(k−1)], with zero assigned when k < 2. Higher values indicate '
                   'a more interconnected local neighborhood; lower values a more open neighborhood.'),
    'betweenness': ('Betweenness centrality', 'Exact normalized, unweighted betweenness: the sum '
                           'of fractions of shortest paths between other node pairs passing through a '
                           'protein, normalized by (N−1)(N−2)/2 for an undirected graph. Endpoints are '
                           'excluded; tied shortest paths share credit. Higher values indicate greater '
                           'potential path mediation; low or zero values indicate less mediation.'),
}


def static_topology_boxplot(frame, metric, tests, set_name, significance_column='fdr_bh'):
    """Plot full finite values; brackets identify the within-network comparison."""
    title = METRIC_GUIDES[metric][0]
    networks = frame['network'].unique(maintain_order=True).to_list()
    fig = Figure(figsize=(max(12, len(networks) * 2.7), 6.4), layout='constrained')
    ax = fig.subplots()
    colors = ('#D55E00', '#0072B2')
    for index, network in enumerate(networks):
        for offset, is_query, color in zip((-.19, .19), (True, False), colors):
            values = frame.filter(
                (pl.col('network') == network) &
                (pl.col('is_moonlighting') == is_query) & pl.col(metric).is_finite()
            )[metric].to_numpy()
            if not len(values):
                continue
            boxes = ax.boxplot([values], positions=[index + offset], widths=.30,
                               patch_artist=True, showfliers=False, showmeans=True,
                               medianprops={'color': '#222222'},
                               meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                          'markeredgecolor': '#222222', 'markersize': 4})
            boxes['boxes'][0].set(facecolor=color, alpha=.65)
        rows = tests.filter((pl.col('network') == network) & (pl.col('metric') == metric))
        row = rows.row(0, named=True) if rows.height else {}
        def number(key):
            value = row.get(key)
            return f'{value:.3g}' if value is not None and np.isfinite(value) else 'n/a'
        # Axes-fraction height keeps brackets clear even for zero-valued distributions.
        transform = ax.get_xaxis_transform()
        ax.plot([index-.19, index-.19, index+.19, index+.19], [.87, .90, .90, .87],
                transform=transform, color='#333333', linewidth=1.2, clip_on=False)
        effect = row.get('rank_biserial')
        effect_text = f'{effect:+.2f}' if effect is not None and np.isfinite(effect) else 'n/a'
        basis = 'BH q' if significance_column == 'fdr_bh' else 'raw p'
        q_text = f"q={number('fdr_bh')} · " if significance_column == 'fdr_bh' else ''
        ax.text(index, .92, f"{significance_label(row.get(significance_column))} ({basis})\np={number('p_value')}\n"
                f"{q_text}r={effect_text}",
                transform=transform, ha='center', va='bottom', fontsize=10)
    # Preserve zero degree values on the log-like display.
    if metric == 'degree':
        ax.set_yscale('symlog', linthresh=1)
    low, high = ax.get_ylim()
    if metric == 'degree':
        high = max(high, 1) ** 1.45 + 2
    else:
        high = low + (high - low) * 1.45
    ax.set_ylim(low, high)
    ax.set_xticks(range(len(networks)), networks)
    ax.set_xlim(-.65, max(len(networks)-.35, .65))
    ax.set_ylabel(title + (' (symlog; linear from 0 to 1)' if metric == 'degree' else ''))
    ax.set_title(f'{title}: {set_name} versus background', pad=62)
    ax.grid(axis='y', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Patch(facecolor=c, alpha=.65, label=label)
                        for c, label in zip(colors, (set_name, 'Background'))],
               loc='outside lower center', ncol=2, frameon=False)
    return fig
