"""Compare bounded ego neighborhoods with network-local overlapping OCG clusters."""
import io
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import networkx as nx
import polars as pl

from ocg_exploration import ego_cluster_subgraph
from ocg_ego_overlap import ego_groups, ego_overlap_figure, norma_payload, norma_export_zip
from moonlighting import ADVANCED_EDGE_LIMIT, significance_label


def compare_ego_records(records, center, *, max_neighbors=50, reference=None):
    """Use full memberships and full neighbor sets for comparisons; caps affect drawings."""
    views, all_ids = {}, {center}
    reference_ids = None
    if reference is not None:
        record = next(r for r in records if r['network'] == reference)
        reference_nodes, _ = ego_cluster_subgraph(center, record['memberships'], record['edges'],
                                                  max_neighbors=max_neighbors)
        reference_ids = reference_nodes['protein'].to_list()
    summary_rows = []
    for record in records:
        network, edges, members = record['network'], record['edges'], record['memberships']
        population = set(pl.concat([edges['a'], edges['b']]).unique().to_list())
        neighbors = set(pl.concat([
            edges.filter(pl.col('a') == center)['b'], edges.filter(pl.col('b') == center)['a']
        ]).unique().to_list()) - {center}
        assigned = members.select('protein', pl.col('community').cast(pl.String)).drop_nulls().unique()
        own_clusters = assigned.filter(pl.col('protein') == center)['community'].sort().to_list()
        if center in population:
            nodes, local_edges = ego_cluster_subgraph(center, members, edges, max_neighbors=max_neighbors)
        else:
            nodes = pl.DataFrame(schema={'protein':pl.String})
            local_edges = edges.head(0)
        if reference_ids is not None:
            nodes = pl.DataFrame({'protein': sorted(set(reference_ids) & population)}, schema={'protein':pl.String})
            shown = nodes['protein'].to_list()
            local_edges = edges.filter(pl.col('a').is_in(shown) & pl.col('b').is_in(shown))
        all_ids.update(neighbors)  # Full, uncapped union for the membership comparison table.
        views[network] = dict(nodes=nodes, edges=local_edges, memberships=assigned,
            population=population, neighbors=neighbors, labels=record.get('labels'), run=record.get('run',''))
        summary_rows.append(dict(network=network, run=record.get('run',''),
            center_present=center in population, center_clusters='; '.join(own_clusters),
            center_cluster_count=len(own_clusters), full_neighbor_count=len(neighbors),
            displayed_proteins=nodes.height, displayed_edges=local_edges.height))
    rows, overlaps = [], []
    for network, view in views.items():
        ids = pl.DataFrame({'protein':sorted(all_ids)})
        counts = view['memberships'].group_by('protein').agg(
            pl.col('community').sort().alias('cluster_ids'), pl.len().alias('cluster_count'))
        table = ids.join(counts, on='protein', how='left').with_columns(
            pl.col('cluster_count').fill_null(0),
            pl.col('cluster_ids').list.join('; ').fill_null('').alias('cluster_ids'),
            pl.lit(network).alias('network'), pl.lit(view['run']).alias('run'),
            pl.col('protein').is_in(view['population']).alias('present_in_network'),
            pl.col('protein').is_in(view['neighbors']).alias('is_center_neighbor'),
            pl.col('protein').is_in(view['nodes']['protein'].to_list()).alias('displayed'),
            (pl.col('protein') == center).alias('is_center'))
        rows.append(table)
    networks = list(views)
    for i, a in enumerate(networks):
        for b in networks[i+1:]:
            left, right = views[a]['neighbors'], views[b]['neighbors']
            union = left | right
            overlaps.append(dict(network_a=a, network_b=b, shared_neighbors=len(left & right),
                                 union_neighbors=len(union), neighbor_jaccard=len(left & right)/len(union) if union else None,
                                 center_present_both=center in views[a]['population'] and center in views[b]['population']))
    return views, pl.from_dicts(summary_rows), pl.concat(rows), pl.from_dicts(overlaps) if overlaps else pl.DataFrame()


def member_label_table(view, proteins):
    """One row per protein/cluster; labels remain scoped to the original network/run."""
    table = view['memberships'].filter(pl.col('protein').is_in(proteins))
    labels = view.get('labels')
    if labels is not None:
        table = table.join(labels.with_columns(pl.col('community').cast(pl.String)), on='community', how='left')
        for column in labels.columns:
            if column.endswith('_qvalue'):
                table = table.with_columns(pl.Series(column.removesuffix('_qvalue')+'_significance',
                    [significance_label(q) for q in table[column]]))
    return table.sort('protein','community')


def comparison_zip(views, summary, proteins, overlaps, center):
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        for name, frame in [('networks',summary), ('protein_memberships',proteins), ('neighbor_overlaps',overlaps)]:
            archive.writestr(name+'.csv', frame.write_csv())
        for index, (network, view) in enumerate(views.items()):
            prefix = f'{index+1}_{network}/'
            archive.writestr(prefix+'cluster_labels.csv', member_label_table(view, proteins['protein'].unique().to_list()).write_csv())
            groups, _ = ego_groups(view['nodes'],view['memberships'],center, max_clusters=100000,scope='all')
            payload = norma_payload(view['nodes'],view['edges'],groups,center,name=f'{network} · {center}')
            archive.writestr(prefix+'norma.zip', norma_export_zip(payload))
        archive.writestr('README.txt', 'Full neighbor sets are compared before display caps. Cluster IDs are local to each network/run.\n'
            'Membership tables cover the center and union of full neighbor sets, including missing and unassigned proteins.\n'
            'NORMA bundles contain displayed subgraphs and all their overlapping cluster groups, independently of outline caps.\n'
            'Saved enrichment q-values are retained, never recalculated from displayed proteins.\n')
    return buffer.getvalue()


def cross_ego_panel(center, reference_network, reference_directory, runs, datasets, data_root, metadata):
    import streamlit as st
    from clusterannotation import (file_input, DEFAULT_GAF, DEFAULT_OBO, DEFAULT_REACTOME,
        DEFAULT_FUNCTIONS, DEFAULT_GENERIF, DEFAULT_GENE_MAPPING, DEFAULT_ABSTRACTS)
    from network_graph_utils import protein_identity
    from ocg_annotation_ui import available_annotation_bundle

    with st.expander('Compare this ego protein across interactomes', key='ocg_cross_ego_open', on_change='rerun') as panel:
        if not panel.open:
            return
        options = sorted(set(runs['network']) & set(datasets))
        networks = st.multiselect('Interactomes to compare', options, default=options, key='ocg_cross_ego_networks')
        mode = st.radio('Neighborhood comparison', ['Each network’s own ego neighbors', 'Fixed reference neighborhood'],
                        key='ocg_cross_ego_mode')
        st.caption('Each network uses its own neighbors, or the displayed reference neighborhood is held fixed. '
                   'Full neighbor overlap statistics always use uncapped neighbor sets. Cluster IDs and colours are local '
                   'to each network and do not imply matched biological clusters. Missing proteins and unassigned proteins are distinguished.')
        max_neighbors = st.slider('Maximum neighbors per network (or reference)', 10,120,50,step=10,key='ocg_cross_ego_max_neighbors')
        max_clusters = st.slider('Maximum cluster outlines per network',1,20,8,key='ocg_cross_ego_max_clusters')
        show_names = st.checkbox('Show names in comparison plots',True,key='ocg_cross_ego_names')
        selected = []
        for network in networks:
            choices = runs.filter(pl.col('network') == network).sort('run',descending=True)
            dirs = choices['directory'].to_list()
            default = dirs.index(str(reference_directory)) if network == reference_network and str(reference_directory) in dirs else 0
            directory = st.selectbox(f'{network} · OCG run',dirs,index=default,
                format_func=lambda p: Path(p).name,key=f'ocg_cross_ego_run_{network}')
            selected.append((network,directory))
        if len(networks)<2:
            st.info('Select at least two interactomes.')
            return
        if mode == 'Fixed reference neighborhood' and reference_network not in networks:
            st.info(f'Include {reference_network} to use its reference neighborhood.')
            return
        files = [Path(p) for n,d in selected for p in (datasets[n],Path(d)/'memberships.parquet',Path(d)/'result.json')]
        files += [Path(__file__), Path(__file__).with_name('ocg_exploration.py'),
                  Path(__file__).with_name('clustering_explorer.py'), Path(__file__).with_name('clusterannotation.py'),
                  Path(__file__).with_name('moonlighting.py')]
        files += [p for p in (DEFAULT_GAF,DEFAULT_OBO,DEFAULT_REACTOME,DEFAULT_FUNCTIONS,
                             DEFAULT_GENERIF,DEFAULT_GENE_MAPPING,DEFAULT_ABSTRACTS) if p.is_file()]
        # Annotation files and code are included too: cached session results must not hide newly completed labels.
        annotation_root = Path(data_root)/'analysis_runs/clustering_annotations'
        annotation_files = [p for n,d in selected for p in (annotation_root/n/Path(d).parent.name/Path(d).name).glob('*/manifest.json')]
        identity = (center,reference_network,tuple(selected),mode,max_neighbors,
                    tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files+annotation_files))
        state_key='ocg_cross_ego_result'
        if st.button('Compare ego networks',type='primary',key='ocg_cross_ego_run'):
            st.session_state.pop(state_key,None)
            records, failures = [], []
            with st.spinner('Loading selected networks, verifying provenance and reading cached annotations…'):
                for network,directory in selected:
                    try:
                        path = Path(datasets[network])
                        if pl.scan_parquet(path).select(pl.len()).collect().item()>ADVANCED_EDGE_LIMIT:
                            raise ValueError(f'Exceeds the {ADVANCED_EDGE_LIMIT:,}-edge guard')
                        members,_,_ = file_input(Path(directory)/'memberships.parquet',str(path))
                        edges = pl.read_parquet(path,columns=['a','b'])
                        bundle, full = available_annotation_bundle(directory,None)
                        labels = pl.read_csv(bundle/'label_summary.csv',schema_overrides={'community':pl.String}) if bundle else None
                        records.append(dict(network=network,run=Path(directory).name,edges=edges,memberships=members,labels=labels))
                    except (OSError,ValueError,pl.exceptions.PolarsError) as exc:
                        failures.append(f'{network}: {exc}')
                if len(records)>=2 and (mode != 'Fixed reference neighborhood' or any(r['network']==reference_network for r in records)):
                    results = compare_ego_records(records,center,max_neighbors=max_neighbors,
                        reference=reference_network if mode=='Fixed reference neighborhood' else None)
                    st.session_state[state_key] = (identity,results,failures)
                else:
                    st.error('At least two compatible networks, including the fixed reference when requested, are required.')
                    for failure in failures: st.warning(failure)
        state = st.session_state.get(state_key)
        if not state:
            return
        if state[0] != identity:
            st.info('The protein, runs, inputs, annotation caches or comparison settings changed. Compare ego networks again.')
            return
        views,summary,proteins,overlaps = state[1]
        for failure in state[2]: st.warning(failure)
        st.dataframe(summary,hide_index=True,width='stretch')
        st.markdown('**Neighbor conservation · full uncapped sets**')
        st.dataframe(overlaps,hide_index=True,width='stretch')
        st.caption('Jaccard = shared neighbors / union of neighbors. Empty unions are unavailable. '
                   'These are descriptive comparisons; no significance test applies. A missing center has no recorded neighbors.')
        # Align proteins across panels using a single deterministic layout of the union drawing.
        graph=nx.Graph()
        for view in views.values():
            graph.add_nodes_from(view['nodes']['protein'].to_list())
            graph.add_edges_from(view['edges'].select('a','b').iter_rows())
        positions=nx.spring_layout(graph,seed=42,iterations=100)
        xrange = [min((v[0] for v in positions.values()),default=-1)-.1,
                  max((v[0] for v in positions.values()),default=1)+.1]
        yrange = [min((v[1] for v in positions.values()),default=-1)-.1,
                  max((v[1] for v in positions.values()),default=1)+.1]
        names={p:protein_identity(p,metadata)['label'] for p in graph}
        columns=st.columns(2)
        for index,(network,view) in enumerate(views.items()):
            with columns[index%2]:
                st.markdown(f'**{network} · {view["run"]}**')
                if center not in view['population']:
                    st.info('Center protein absent from this network.')
                if view['nodes'].is_empty():
                    continue
                groups,ranked=ego_groups(view['nodes'],view['memberships'],center,max_clusters,scope='all')
                label_lookup={} if view['labels'] is None else {str(r['community']): ' · '.join(
                    f'{v} {significance_label(r.get(k.removesuffix("_enrichment")+"_qvalue"))}'
                    for k,v in r.items() if k.endswith('_enrichment') and v) for r in view['labels'].iter_rows(named=True)}
                fig=ego_overlap_figure(view['nodes'],view['edges'],groups,center,names=names,labels=label_lookup,
                                       show_names=show_names,fixed_positions=positions)
                fig.update_layout(height=520,title=network,showlegend=False)
                fig.update_xaxes(range=xrange)
                fig.update_yaxes(range=yrange)
                st.plotly_chart(fig,width='stretch',key=f'ocg_cross_ego_plot_{network}')
                st.caption(f'{len(groups)}/{ranked.height} clusters outlined. Same protein coordinates across panels; '
                           'outlines may enclose nonmembers. Memberships, not geometry, define clusters.')
        st.markdown('**Protein memberships across networks**')
        st.dataframe(proteins,hide_index=True,width='stretch')
        inspect=st.selectbox('Inspect center or neighbor cluster labels',sorted(proteins['protein'].unique()),
            index=sorted(proteins['protein'].unique()).index(center),
            format_func=lambda p:f'{protein_identity(p,metadata)["label"]} · {p}',key='ocg_cross_ego_inspect')
        for network,view in views.items():
            st.markdown(f'**{network} · {inspect}**')
            if inspect not in view['population']:
                st.info('Protein absent from this network.')
                continue
            table=member_label_table(view,[inspect])
            if table.is_empty(): st.info('Protein present but without an assigned OCG cluster.')
            else: st.dataframe(table,hide_index=True,width='stretch')
            if view['labels'] is None: st.caption('Functional labels are not cached for this run; no annotation job was started.')
        st.caption('Saved functional labels retain one-sided hypergeometric tests against each original network and '
                   'BH over all eligible terms, libraries, clusters and nonempty full/core/interface subsets in that run. '
                   '* q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001; ns otherwise; n/a unavailable. '
                   'No tests are recomputed from displayed neighborhoods. Topology does not validate moonlighting.')
        st.download_button('Download cross-network ego comparison + NORMA bundles',
            lambda: comparison_zip(views,summary,proteins,overlaps,center),
            f'{center}_cross_network_ego.zip','application/zip',on_click='ignore',key='ocg_cross_ego_zip')
