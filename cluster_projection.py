"""Reproducible 2D views of saved clustering features or network topology."""
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh

from clustering_explorer import ROOT


def project_features(proteins, features, method='PCA', seed=42):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or len(proteins) != len(features) or not np.isfinite(features).all():
        raise ValueError('Invalid protein feature matrix')
    if len(proteins) < 3 or features.shape[1] < 1:
        raise ValueError('Projection needs at least three proteins and one feature')
    if method == 'UMAP':
        from umap import UMAP
        coordinates = UMAP(n_components=2, random_state=seed, n_neighbors=min(15, len(proteins)-1),
                           init='random', n_jobs=1).fit_transform(features)
    elif method == 'PCA':
        centered = features - features.mean(axis=0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        coordinates = np.zeros((len(proteins), 2))
        width = min(2, len(s))
        coordinates[:, :width] = u[:, :width] * s[:width]
    else:
        raise ValueError('Unknown projection method')
    return pl.DataFrame({'protein': proteins, 'x': coordinates[:, 0], 'y': coordinates[:, 1]})


def projection(directory, method='PCA'):
    from annotate_clusters import file_input
    from networks import discover_datasets
    directory = Path(directory)
    network = directory.parents[1].name
    path = discover_datasets(ROOT / 'data').get(network)
    if path is None:
        raise ValueError(f'Local network unavailable: {network}')
    # Validate the saved run's network fingerprint before reconstructing any features.
    _, _, source = file_input(directory / 'memberships.parquet', str(path))
    soft = directory / 'soft_memberships.npz'
    identity = {'source': source, 'projection': method, 'seed': 42,
                'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if soft.is_file():
        identity['soft_sha256'] = hashlib.sha256(soft.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    target = ROOT / 'data/analysis_runs/clustering_projections' / network / directory.parent.name / directory.name / key
    if target.with_suffix('.parquet').is_file():
        return pl.read_parquet(target.with_suffix('.parquet')), json.loads(target.with_suffix('.json').read_text())
    if soft.is_file():
        with np.load(soft, allow_pickle=False) as saved:
            proteins, features = saved['proteins'].astype(str), saved['memberships']
        description = 'Saved soft-membership features from this clustering_raw run'
    else:
        from cluster_networks import indexed_graph
        graph = indexed_graph(path)
        n = len(graph['nodes'])
        if n < 4:
            raise ValueError('Topology projection requires at least four proteins')
        degree = np.asarray(graph['adj'].sum(axis=1)).ravel()
        inverse = sparse.diags(1 / np.sqrt(np.maximum(degree, 1)))
        normalized = inverse @ graph['adj'] @ inverse
        # Sparse normalized adjacency; no dense protein-by-protein matrix.
        k = min(17, n - 1)
        values, vectors = eigsh(normalized, k=k, which='LA',
                                v0=np.random.default_rng(42).normal(size=n), tol=1e-4)
        order = np.argsort(values)[::-1][1:]
        features = vectors[:, order]
        features = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-12)
        proteins = graph['nodes']
        description = ('Reconstructed standardized spectral topology features (up to 16 dimensions); '
                       'original clustering coordinates were not saved. This is a new network view, '
                       'not the original algorithm embedding.')
    points = project_features(proteins, features, method)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**identity, 'features': description, 'proteins': points.height}
    points.write_parquet(target.with_suffix('.parquet'))
    target.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    return points, manifest


def projection_panel(directory):
    import importlib.util
    import plotly.graph_objects as go
    import streamlit as st
    from moonlighting import GROUP_LABELS, SOURCES, cohort_ids, load_moon_data
    data = ROOT / 'data'
    if not (data / 'mappings/moonlighting_uniprot.json').is_file():
        st.info('Prepare moonlighting accession mappings on the topological analysis page first.')
        return
    sources = st.multiselect('Highlight protein set', SOURCES, default=['moonlitdb'],
                            format_func=lambda value: GROUP_LABELS.get(value, value))
    require_all = st.checkbox('Require all selected sources')
    selected, _ = cohort_ids(load_moon_data(data), sources=sources, require_all=require_all)
    label = (' ∩ ' if require_all else ' ∪ ').join(GROUP_LABELS.get(s, s) for s in sources) or 'All sources'
    choices = ['PCA', 'UMAP'] if importlib.util.find_spec('umap') else ['PCA']
    method = st.selectbox('2D projection', choices)
    if 'UMAP' not in choices:
        st.caption('PCA is available. Install the optional umap-learn package to enable UMAP.')
    st.info('Each point is one protein. The selected protein set is colored orange over a gray network '
            'background. Proximity summarizes feature similarity; axes have no biological units. '
            'UMAP emphasizes local neighborhoods and can distort global distances. '
            'Disconnected components can create artificial separation. This descriptive projection has '
            'no significance test and does not establish functional differences or moonlighting.')
    identity = (str(directory), method)
    if st.button('Calculate / load projection', type='primary'):
        st.session_state.pop('cluster_projection', None)
        try:
            with st.spinner('Preparing protein projection…'):
                points, manifest = projection(directory, method)
            st.session_state['cluster_projection'] = (identity, points, manifest)
        except Exception as exc:
            st.error(f'Projection unavailable: {exc}')
    state = st.session_state.get('cluster_projection')
    if not state or state[0] != identity:
        return
    _, points, manifest = state
    st.caption(manifest['features'])
    from networks import matching_nodes
    matched = matching_nodes(set(points['protein']), selected)
    members = pl.read_parquet(Path(directory) / 'memberships.parquet').group_by('protein').agg(
        pl.col('community').drop_nulls().cast(pl.String).str.join(';').alias('communities'))
    points = points.join(members, on='protein', how='left').with_columns(
        pl.col('protein').is_in(sorted(matched)).alias('in_protein_set'))
    st.caption(f'{label}: {len(matched):,} of {points.height:,} plotted proteins. '
               'All proteins are displayed; highlighted proteins are drawn last.')
    fig = go.Figure()
    for highlight, name, color, size in [(False, 'Network background', '#b8b8b8', 4),
                                         (True, label, '#D55E00', 8)]:
        frame = points.filter(pl.col('in_protein_set') == highlight)
        fig.add_trace(go.Scattergl(x=frame['x'].to_list(), y=frame['y'].to_list(), mode='markers',
                                  name=name, customdata=frame.select('protein', 'communities').rows(),
                                  hovertemplate='%{customdata[0]}<br>Clusters: %{customdata[1]}<extra>%{fullData.name}</extra>',
                                  marker=dict(color=color, size=size, opacity=.85 if highlight else .35)))
    fig.update_layout(height=650, xaxis_title=f'{method} 1', yaxis_title=f'{method} 2',
                      legend=dict(orientation='h'), margin=dict(t=50, b=50))
    st.plotly_chart(fig, width='stretch', alt=f'{method} protein projection highlighting {label}')
    st.download_button('Download coordinates and protein set membership', points.write_csv(),
                       'protein_projection.csv', 'text/csv')
