"""Explore nodevectors embeddings independently of community detection."""
import json

import polars as pl
import streamlit as st

from network_embeddings import ROOT, embed, identity, run_directory, saved_projection
from embedding_display import embedding_figure
from networks import discover_datasets, matching_nodes

st.title('Network embeddings')
st.caption('Learn protein features with nodevectors, then explore interactive 2D or 3D projections.')
with st.expander('How to interpret this view', expanded=True):
    st.write('Each point is a protein in the complete selected network. ProNE uses sparse matrix '
             'factorization and propagation; GGVec learns local graph relationships. Both use '
             'undirected binary topology, without confidence weights or self loops. Degree counts '
             'unique interaction partners; higher degree means more partners, not greater biological importance.')
    st.write('PCA preserves directions of greatest feature variance. UMAP emphasizes local neighborhoods; '
             't-SNE emphasizes local similarity and can distort sizes and global distances. Axes have no '
             'biological units. Disconnected components and stochastic optimization can create separation. '
             'Independent networks have unaligned coordinates. These are descriptive views: no significance '
             'test or stars apply. Proximity does not validate moonlighting, binding exclusivity or function.')

policy = st.session_state.get('isoform_policy', 'collapse')
datasets = discover_datasets(ROOT / 'data', isoforms=policy)
if not datasets:
    st.info('Prepare canonical networks on the overview page first.')
    st.stop()
network = st.selectbox('Network and saved threshold version', list(datasets), key='embedding_network')
method = st.selectbox('Embedding algorithm', ['ProNE', 'GGVec'])
dimensions = st.select_slider('Embedding dimensions', [8, 16, 32, 64, 128], value=32)
seed = int(st.number_input('Random seed', min_value=0, max_value=2**32 - 1, value=42))
st.caption(f'Isoform policy: {policy}. Every protein and edge is used; no network sampling or NetworkX conversion. '
           'GGVec may vary between runs despite the recorded seed. Large networks can take several minutes.')
config = identity(datasets[network], method, dimensions, seed)
directory = run_directory(datasets[network], config)
if st.button('Calculate / load selected embedding', type='primary'):
    try:
        with st.spinner(f'Embedding {network}…'):
            embed(datasets[network], method, dimensions, seed)
    except Exception as exc:
        st.error(f'Embedding unavailable: {exc}')
if st.button('Calculate / load all networks and saved versions'):
    progress = st.progress(0.)
    failures = []
    for i, (name, path) in enumerate(datasets.items()):
        try:
            with st.spinner(f'Embedding {name}…'):
                embed(path, method, dimensions, seed)
        except Exception as exc:
            failures.append(f'{name}: {exc}')
        progress.progress((i + 1) / len(datasets))
    if failures:
        st.error('\n\n'.join(failures))
    else:
        st.success('All discovered networks and saved versions are ready.')
if not (directory / 'manifest.json').is_file():
    st.info('Calculate the selected embedding or all networks above. Install dependencies with '
            '`pip install -e ".[embeddings]"` if needed.')
    st.stop()
manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
for warning in manifest.get('warnings', []):
    st.warning(f'Embedding diagnostic: {warning}')
st.caption(f"{manifest['proteins']:,} proteins · {manifest['edges']:,} edges · {dimensions} learned features")
projection = st.selectbox('Projection method', ['UMAP', 'PCA', 't-SNE'])
view_dimensions = st.radio('View dimensions', [2, 3], horizontal=True, format_func=lambda n: f'{n}D')
neighbors, min_dist, perplexity = 15, .1, 30.
if projection == 'UMAP':
    neighbors = int(st.number_input('UMAP neighbors', min_value=2, max_value=200, value=15))
    min_dist = st.slider('UMAP minimum distance', 0., .99, .1)
elif projection == 't-SNE':
    perplexity = st.slider('t-SNE perplexity', 2., 100., 30.)
parameters = dict(method=projection, dimensions=view_dimensions, seed=seed,
                  neighbors=neighbors, min_dist=min_dist, perplexity=perplexity)
selection = (str(directory), json.dumps(parameters, sort_keys=True))
if st.button('Calculate / load projection'):
    st.session_state.pop('embedding_projection', None)
    try:
        with st.spinner('Projecting all proteins…'):
            points = saved_projection(directory, **parameters)
        st.session_state['embedding_projection'] = (selection, points)
    except Exception as exc:
        st.error(f'Projection unavailable: {exc}')
state = st.session_state.get('embedding_projection')
if not state or state[0] != selection:
    st.stop()
points = state[1]
color = st.selectbox('Color proteins by', ['Degree (log10 + 1)', 'Protein set'])
highlight_enabled = st.checkbox('Highlight a protein set', value=True, key='embedding_highlight_enabled')
outline_style, outline_size, point_opacity, show_labels = 'circle-open', 11, .65, False
highlight = set()
label = 'Custom protein set'
if highlight_enabled or color == 'Protein set':
    from moonlighting import SOURCES, GROUP_LABELS, cohort_ids, load_moon_data
    source_options = ['Custom accessions', *SOURCES]
    source = st.selectbox('Protein set source', source_options, index=source_options.index('moonlitdb'),
                          format_func=lambda s: GROUP_LABELS.get(s, s), key='embedding_highlight_source')
    if source == 'Custom accessions':
        text = st.text_area('UniProt accessions (spaces, commas or newlines)')
        selected = set(text.upper().replace(',', ' ').split())
    else:
        label = GROUP_LABELS.get(source, source)
        try:
            selected, _ = cohort_ids(load_moon_data(ROOT / 'data'), sources=[source])
        except (OSError, ValueError) as exc:
            st.warning(f'{label} unavailable. Prepare protein set mappings on the topological analysis page. {exc}')
            selected = set()
    highlight = matching_nodes(set(points['protein']), selected)
    st.caption(f'{label}: {len(highlight):,} of {points.height:,} network proteins matched '
               f'from {len(selected):,} selected accessions.')
    if not highlight:
        st.info(f'No {label} proteins matched this network.')
if highlight_enabled:
    style_name = st.selectbox('Protein set highlight style', ['Outline rings', 'Diamond outlines'])
    outline_style = 'circle-open' if style_name == 'Outline rings' else 'diamond-open'
    outline_size = st.slider('Protein set outline size', 6, 20, 11)
    point_opacity = st.slider('Base point opacity', .1, 1., .65, step=.05)
    show_labels = st.checkbox('Label highlighted proteins', value=False)
    st.caption('Outlined markers identify the selected protein set. Their centers retain the selected '
               'color encoding; white halos and dark borders keep them visible. Highlighting only '
               'changes the display and reuses the current embedding and projection.')
query = st.text_input('Find a protein (exact UniProt accession)').strip().upper()
if query and query not in set(points['protein']):
    st.warning(f'{query} is absent from this network.')
points = points.with_columns(pl.col('protein').is_in(sorted(highlight)).alias('in_protein_set'))
fig = embedding_figure(points, view_dimensions, projection, color, label,
                       highlight=highlight_enabled, outline=outline_style, marker_size=outline_size,
                       background_opacity=point_opacity, labels=show_labels, query=query)
st.plotly_chart(fig, width='stretch', alt=f'{network} proteins in a {view_dimensions}D {projection} projection')
st.caption('All proteins shown. Zoom and hover in 2D; drag to rotate in 3D. No significance test applies.')
st.download_button('Download coordinates and protein set membership', points.write_csv(),
                   f'{network}_{projection}_{view_dimensions}d.csv', 'text/csv')
st.download_button('Download learned vectors', lambda: (directory / 'vectors.parquet').read_bytes(),
                   f'{network}_vectors.parquet', 'application/octet-stream')
with st.expander('Embedding provenance'):
    st.json(manifest)
