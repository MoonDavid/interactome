"""Shared Streamlit controls for functional cluster annotation."""
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl
import streamlit as st

from annotate_clusters import (
    DATA as DEFAULT_DATA, DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME,
    annotate_memberships, write_annotation_bundle,
)
from moonlighting import _signature


def atlas_memberships(node_frame: pl.DataFrame) -> pl.DataFrame:
    """Preserve every membership emitted by Atlas clustering."""
    return (
        node_frame.select('protein', pl.col('communities_str').str.split(';').alias('community'))
        .explode('community')
    )


def bundle_zip(bundle: Path) -> bytes:
    archive = BytesIO()
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as zipped:
        for name in ('cluster_labels.csv', 'enrichment.parquet', 'manifest.json'):
            zipped.write(bundle / name, name)
    return archive.getvalue()


def annotation_panel(key, identity, load_inputs, data_root: Path, *, cytoscape=None,
                     button='Annotate clusters'):
    """Run on demand and expose the current run and selected cluster to its caller."""
    paths = tuple(data_root / path.relative_to(DEFAULT_DATA) for path in (DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME))
    gaf, obo, reactome = paths
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        st.info(f'Add the local annotation files before running: {", ".join(missing)}.')
        return None, None
    signature = (identity, tuple(_signature(path) for path in paths), cytoscape)
    state_key = f'{key}_result'
    st.caption('GO and Reactome labels use exclusive cores and shared interfaces. '
               'All tested clusters, subsets, and libraries share one FDR correction.')
    if st.button(button, key=f'{key}_run', type='primary'):
        st.session_state.pop(state_key, None)
        try:
            with st.spinner('Calculating GO and Reactome enrichment…'):
                frame, population, source = load_inputs()
                result = annotate_memberships(
                    frame, population, source, gaf=gaf, obo=obo, reactome=reactome, cytoscape=cytoscape,
                )
                bundle = data_root / 'analysis_runs' / f'cluster_annotation_{uuid4().hex[:16]}'
                write_annotation_bundle(bundle, result)
                result.pop('evidence')
                st.session_state[state_key] = {'signature': signature, 'bundle': bundle, **result}
        except Exception as exc:
            st.error(f'Cluster annotation failed. {exc}')
    result = st.session_state.get(state_key)
    if result is None:
        return None, None
    if result['signature'] != signature:
        st.info('Inputs changed. Run annotation again for the current selection.')
        return None, None

    labels = result['labels']
    manifest = result['manifest']
    st.caption(f'{labels.height:,} clusters · {manifest["coverage"]["background"]:,} background proteins · '
               f'{manifest["tests"]:,} enrichment tests')
    display_columns = ['community', 'full_size', 'primary_label', 'interface_label', 'full_context_label']
    comparison = manifest['cytoscape_comparison']
    if comparison['status'] != 'not_requested':
        display_columns += ['autoannotate_description_label', 'autoannotate_terms_label']
        if comparison['status'] != 'complete':
            st.warning(f'Cytoscape comparison is {comparison["status"]}. Python enrichment is available.')
            with st.expander('Cytoscape comparison details'):
                st.json(comparison)
    st.dataframe(
        labels.select(display_columns), hide_index=True, alt='Functional labels by cluster',
        column_config={
            'community': 'Cluster', 'full_size': 'Proteins', 'primary_label': 'Core identity',
            'interface_label': 'Shared interface', 'full_context_label': 'Full-cluster context',
            'autoannotate_description_label': 'Cytoscape · descriptions',
            'autoannotate_terms_label': 'Cytoscape · GO and Reactome',
        },
    )
    names = {
        row['community']: row['primary_label'] or row['full_context_label'] or row['interface_label']
        for row in labels.iter_rows(named=True)
    }
    cluster = st.selectbox(
        'Inspect cluster', list(names), index=None, placeholder='Select a cluster',
        format_func=lambda item: f'{item} · {names[item]}' if names[item] else item,
        key=f'{key}_{result["bundle"].name}_cluster',
    )
    if cluster is not None:
        details = labels.filter(pl.col('community') == cluster)
        st.dataframe(
            details, hide_index=True, alt='Selected cluster labels, sizes, and availability',
        )
        evidence = (
            pl.scan_parquet(result['bundle'] / 'enrichment.parquet').filter(pl.col('community') == cluster)
            .sort('qvalue', 'pvalue', 'library', 'term_id').head(200).collect()
        )
        if evidence.is_empty():
            st.caption('No eligible terms had protein hits for this cluster.')
        else:
            st.caption('Up to 200 terms with hits, ordered by adjusted p-value. The download contains all evidence.')
            st.dataframe(evidence, hide_index=True, alt='Enrichment evidence for the selected cluster')
    bundle = result['bundle']
    st.download_button(
        'Download annotation bundle', lambda: bundle_zip(bundle), f'{bundle.name}.zip',
        'application/zip', on_click='ignore', key=f'{key}_download',
    )
    return result, cluster
