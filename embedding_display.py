"""Embedding display with protein-set outlines independent of point color."""
import numpy as np
import plotly.graph_objects as go
import polars as pl


def embedding_figure(points, dimensions, projection, color_by, set_name,
                     highlight=True, outline='circle-open', marker_size=11,
                     background_opacity=.65, labels=False, query=''):
    trace = go.Scattergl if dimensions == 2 else go.Scatter3d
    fig = go.Figure()
    color_max = max(float(np.log10(points['degree'].max() + 1)), 1e-12)
    groups = [(points, 'Proteins', None)] if color_by.startswith('Degree') else [
        (points.filter(~pl.col('in_protein_set')), 'Network background', '#aaaaaa'),
        (points.filter(pl.col('in_protein_set')), set_name, '#D55E00')]
    def coords(frame):
        return {axis: frame[axis].to_list() for axis in 'xyz'[:dimensions]}
    def hover(frame):
        return dict(customdata=frame.select('protein', 'degree', 'in_protein_set').rows(),
            hovertemplate='%{customdata[0]}<br>Degree: %{customdata[1]}<br>'
                          'In selected protein set: %{customdata[2]}<extra>%{fullData.name}</extra>')
    for frame, name, shade in groups:
        marker = dict(size=4, opacity=background_opacity)
        if shade:
            marker['color'] = shade
        else:
            marker.update(color=np.log10(frame['degree'].to_numpy() + 1), colorscale='Viridis',
                          cmin=0, cmax=color_max, colorbar=dict(title='log10(degree + 1)'))
        fig.add_trace(trace(**coords(frame), name=name, mode='markers', marker=marker, **hover(frame)))
    selected = points.filter(pl.col('in_protein_set'))
    if highlight and selected.height:
        # White halo plus dark open marker stays visible on both ends of Viridis.
        for shade, width, legend in [('#ffffff', 5, False), ('#111111', 2, True)]:
            fig.add_trace(trace(**coords(selected), name=f'{set_name} · highlighted',
                legendgroup='protein_set_outline', showlegend=legend,
                mode='markers+text' if labels and legend else 'markers',
                text=selected['protein'].to_list() if labels and legend else None,
                textposition='top center',
                marker=dict(size=marker_size, symbol=outline, color=shade,
                            opacity=1, line=dict(color=shade, width=width)), **hover(selected)))
    found = points.filter(pl.col('protein').is_in(query)) if isinstance(query, (list, set)) else points.filter(pl.col('protein') == query)
    if found.height:
        fig.add_trace(trace(**coords(found), name='Search match', mode='markers+text',
            text=found['protein'].to_list(), textposition='top center',
            marker=dict(color='#CC79A7', symbol='x', size=13), **hover(found)))
    fig.update_layout(height=700, margin=dict(t=40, b=90), legend=dict(orientation='h'),
        xaxis_title=f'{projection} 1', yaxis_title=f'{projection} 2',
        scene=dict(xaxis_title=f'{projection} 1', yaxis_title=f'{projection} 2', zaxis_title=f'{projection} 3'))
    return fig
