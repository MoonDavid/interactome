"""Descriptive cross-network connectivity views with explicit missing cells."""
import numpy as np
import polars as pl
import plotly.graph_objects as go


def degree_percentiles(metrics):
    """Rank against all finite nodes before selecting protein set members."""
    return (metrics.filter(pl.col('degree').is_finite())
            .with_columns((pl.col('degree').rank('average').over('network') /
                           pl.len().over('network')).alias('degree_percentile'))
            .filter(pl.col('is_moonlighting'))
            .select('network', 'protein', 'degree', 'degree_percentile'))


def _heatmap(x, y, z, hover, title, color_title, height):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=x, y=y, z=np.zeros((len(y), len(x))),
        colorscale=[[0, '#dedede'], [1, '#dedede']], showscale=False,
        text=hover, hovertemplate='%{text}<extra></extra>'))
    fig.add_trace(go.Heatmap(x=x, y=y, z=z, zmin=0, zmax=1, colorscale='Viridis',
        colorbar=dict(title=color_title), text=hover, hoverongaps=False,
        hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(title=title, template='plotly_white', height=height,
                      margin=dict(l=180, r=100, t=75, b=150))
    fig.update_yaxes(autorange='reversed', tickmode='array', tickvals=y, ticktext=y)
    fig.update_xaxes(tickmode='array', tickvals=x, ticktext=x, tickangle=-55)
    return fig


def degree_percentile_heatmap(table, proteins, networks):
    lookup = {(r['protein'], r['network']): r for r in table.iter_rows(named=True)}
    z, hover = [], []
    for protein in proteins:
        values, texts = [], []
        for network in networks:
            row = lookup.get((protein, network))
            values.append(row['degree_percentile'] if row else np.nan)
            texts.append(f'{protein} · {network}<br>' +
                (f"Degree: {row['degree']:g}<br>Percentile: {100 * row['degree_percentile']:.1f}%" if row
                 else 'Not present in this network'))
        z.append(values)
        hover.append(texts)
    return _heatmap(networks, proteins, z, hover, 'Degree percentile across networks',
                    'Percentile<br>(0–1)', max(400, 22 * len(proteins) + 200))


def partner_overlap_heatmap(table, proteins, network_pairs, shared_only=True):
    lookup = {(r['protein'], r['Network A'], r['Network B']): r for r in table.iter_rows(named=True)}
    z, hover = [], []
    for a, b in network_pairs:
        values, texts = [], []
        for protein in proteins:
            row = lookup.get((protein, a, b))
            value = row['partner_jaccard_shared' if shared_only else 'partner_jaccard'] if row else np.nan
            values.append(value)
            details = 'Protein absent from one or both networks'
            if row:
                details = (f"Full partners: {a}={row['degree_A']}, {b}={row['degree_B']}<br>"
                    f"Partners present in both networks: {a}={row['partners_A_in_both']}, {b}={row['partners_B_in_both']}<br>"
                    f"Shared-population intersection / union: {row['shared_partner_intersection']} / {row['shared_partner_union']}<br>"
                    f"Proteins present in both networks: {row['shared_population_n']}<br>" +
                    (f'Jaccard: {value:.3f}' if np.isfinite(value) else 'Jaccard unavailable: no eligible partners'))
            texts.append(f'{protein} · {a} vs {b}<br>{details}')
        z.append(values)
        hover.append(texts)
    labels = [f'{a} vs {b}' for a, b in network_pairs]
    title = 'Partner overlap · partners present in both networks' if shared_only else 'Partner overlap · full neighborhoods'
    return _heatmap(proteins, labels, z, hover, title, 'Jaccard<br>(0–1)', max(400, 30 * len(labels) + 220))
