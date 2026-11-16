import polars as pl

from ocg_annotation_ui import overlap_option
from ocg_exploration import cluster_overlap_graph


def test_selected_protein_cluster_labels_include_all_own_memberships():
    from ocg_annotation_ui import protein_cluster_labels
    members = pl.DataFrame({'protein': ['A', 'A', 'A', 'B', 'C', 'D'],
                            'community': [1, 1, 2, 1, 2, 3]})
    summary = pl.DataFrame({'community': ['1', '2', '3'], 'proteins': [2, 2, 1],
                            'GO_BP_enrichment': ['first', 'second', 'neighbor only'],
                            'GO_BP_qvalue': [.009, .04, .001]})
    result = protein_cluster_labels('A', members, summary)
    assert result['community'].to_list() == ['1', '2']
    assert result['cluster_proteins'].to_list() == [2, 2]
    assert result['GO_BP_enrichment'].to_list() == ['first', 'second']
    assert result['GO_BP_qvalue'].to_list() == [.009, .04]
    assert result['GO_BP_significance'].to_list() == ['**', '*']
    assert protein_cluster_labels('D', members)['community'].to_list() == ['3']


def test_ego_selector_names_and_source_function_details(tmp_path):
    from ocg_annotation_ui import ego_protein_label, protein_function_details
    metadata = {'P04406': {'protein_name': 'Glyceraldehyde-3-phosphate dehydrogenase'}}
    label = ego_protein_label('P04406', {'label': 'GAPDH'}, metadata)
    assert label == 'GAPDH · Glyceraldehyde-3-phosphate dehydrogenase · P04406'
    assert ego_protein_label('UNKNOWN', {'label': 'UNKNOWN'}, {}) == 'UNKNOWN'
    (tmp_path / 'annotations').mkdir()
    pl.DataFrame({'Entry': ['P04406'], 'Function [CC]': ['Source description']}).write_csv(
        tmp_path / 'annotations/uniprot_functions.tsv', separator='\t')
    pl.DataFrame({'uniprot': ['P04406'], 'moonlitdb': ['true'],
                  'moonlitdb_goslims_canonical': ['canonical source term'],
                  'moonlitdb_goslims_moonlighting': ['moonlighting source term']}).write_csv(
        tmp_path / 'allmoon_with_moonlitdb_human.csv')
    details = protein_function_details(str(tmp_path), 'P04406-2', ())
    assert details['description'] == 'Source description'
    assert details['canonical'] == 'canonical source term'
    assert details['moonlighting'] == 'moonlighting source term'
    assert details['accession'] == 'P04406'
    missing = protein_function_details(str(tmp_path), 'UNKNOWN', ())
    assert not missing['description'] and not missing['moonlighting']


def test_annotation_cache_automatically_prefers_full_run(tmp_path, monkeypatch):
    import ocg_annotation_ui as ui
    calls = []
    def target(directory, communities, *, text_annotations):
        calls.append((communities, text_annotations))
        path = tmp_path / ('full' if communities is None else 'subset') / str(text_annotations)
        return path, None, None, None
    monkeypatch.setattr(ui, 'annotation_target', target)
    full = tmp_path / 'full' / 'True'
    full.mkdir(parents=True)
    (full / 'manifest.json').write_text('{}')
    assert ui.available_annotation_bundle('run', ['1']) == (full, True)
    assert calls == [(None, True)]
    (full / 'manifest.json').unlink()
    subset = tmp_path / 'subset' / 'False'
    subset.mkdir(parents=True)
    (subset / 'manifest.json').write_text('{}')
    assert ui.available_annotation_bundle('run', ['1']) == (subset, False)
    (subset / 'manifest.json').unlink()
    assert ui.available_annotation_bundle('run', ['1']) == (None, False)


def test_focused_network_retains_small_cluster_and_reports_exact_overlap():
    members = pl.DataFrame({'protein': ['A', 'B', 'C', 'C', 'D', 'X'],
                            'community': [1, 1, 1, 2, 2, 3]})
    pairs, clusters = cluster_overlap_graph(members, max_edges=None)
    option, nodes, drawn = overlap_option(clusters, pairs, focus=2, limit=1)
    assert nodes['community'].to_list() == [2]
    assert drawn.is_empty()
    option, nodes, drawn = overlap_option(clusters, pairs, focus=2)
    assert set(nodes['community']) == {1, 2}
    assert drawn['shared_proteins'].to_list() == [1]
    link = option['series'][0]['links'][0]
    assert link['value'] == 1
    assert 'Jaccard: 0.250' in link['tooltip']['formatter']
    assert option['series'][0]['draggable']
    assert option['series'][0]['selectedMode'] == 'single'
    _, nodes, drawn = overlap_option(clusters, pairs, focus=2, min_shared=2)
    assert nodes['community'].to_list() == [2]
    assert drawn.is_empty()


def test_network_label_uses_original_adjusted_significance_and_escaped_text():
    members = pl.DataFrame({'protein': ['A', 'B'], 'community': [1, 2]})
    pairs, clusters = cluster_overlap_graph(members, max_edges=None)
    evidence = pl.DataFrame({'community': ['1', '1', '2'], 'subset': ['full'] * 3,
        'library': ['GO_BP'] * 3, 'term_id': ['GO:1', 'GO:2', 'GO:3'],
        'term_name': ['<binding>', 'non-significant', 'depleted'],
        'qvalue': [.009, .2, .001], 'fold_enrichment': [2., 4., .8]})
    option, _, _ = overlap_option(clusters, pairs, evidence)
    tips = {n['id']: n['tooltip']['formatter'] for n in option['series'][0]['data']}
    assert '&lt;binding&gt; **' in tips['1']
    assert 'q=0.009' in tips['1']
    assert 'No significant saved label' in tips['2']


def test_go_reactome_mode_skips_text_sources(monkeypatch, tmp_path):
    import clusterannotation
    import clustering_explorer
    calls = []
    target = tmp_path / 'bundle'
    def fake_target(directory, community, *, text_annotations):
        assert not text_annotations
        return target, 'frame', 'population', 'source'
    def annotate(*args, **kwargs):
        calls.append(kwargs)
        return {'result': True}
    monkeypatch.setattr(clustering_explorer, 'annotation_target', fake_target)
    monkeypatch.setattr(clusterannotation, 'annotate_memberships', annotate)
    monkeypatch.setattr(clusterannotation, 'write_annotation_bundle', lambda *args: None)
    clustering_explorer.annotate_run('run', text_annotations=False)
    assert calls[0]['uniprot_functions'] is None
    assert calls[0]['generif'] is None
    assert calls[0]['communities'] is None


def test_moonlighting_filter_retains_complete_clusters_and_original_enrichment_family():
    from clustering_explorer import saved_cluster_enrichment
    from ocg_annotation_ui import moonlighting_cluster_view
    members = pl.DataFrame({'protein': ['A', 'B', 'B', 'C', 'D', 'E'],
                            'community': [1, 1, 2, 2, 3, 3]})
    population = pl.DataFrame({'protein': ['A', 'B', 'C', 'D', 'E', 'F']})
    enrichment = saved_cluster_enrichment(members, population, {'B'})
    pairs, clusters = cluster_overlap_graph(members, max_edges=None)
    filtered, edges = moonlighting_cluster_view(clusters, pairs, enrichment)
    assert set(filtered['community']) == {1, 2}
    assert filtered['size'].to_list() == [2, 2]
    assert filtered['protein_set_hits'].to_list() == [1, 1]
    assert edges.height == 1
    for row in filtered.iter_rows(named=True):
        original = enrichment.filter(pl.col('community') == str(row['community'])).row(0, named=True)
        assert row['qvalue'] == original['qvalue']
    empty, edges = moonlighting_cluster_view(clusters, pairs,
        saved_cluster_enrichment(members, population, set()))
    assert empty.is_empty() and edges.is_empty()


def test_ego_name_prefers_gene_then_protein_then_accession():
    from network_graph_utils import protein_identity
    metadata = {'A': {'gene_name': 'GENE1', 'protein_name': 'Protein one'},
                'B': {'gene_name': '', 'protein_name': 'Protein two'}}
    assert protein_identity('A', metadata)['label'] == 'GENE1'
    assert protein_identity('B', metadata)['label'] == 'Protein two'
    assert protein_identity('C', metadata)['label'] == 'C'
    assert protein_identity('A-2', metadata)['label'] == 'GENE1'
    assert 'UniProt: A-2' in protein_identity('A-2', metadata)['tooltip']


def test_annotated_ocg_overlap_and_bipartite_views_preserve_labels_and_exports():
    import zipfile
    from io import BytesIO
    from clustering_explorer import saved_cluster_enrichment
    from cytoscape_bridge import export_cytoscape_bundle_zip, export_graphml_string
    from ocg_annotation_ui import moonlighting_cluster_view
    from ocg_visualization import build_annotated_ocg_bipartite_view, build_annotated_ocg_overlap_view

    members = pl.DataFrame({'protein': ['A', 'B', 'C', 'B', 'D', 'E', 'X'],
                            'community': [1, 1, 1, 2, 2, 2, 3]})
    population = pl.DataFrame({'protein': ['A', 'B', 'C', 'D', 'E', 'X', 'Y', 'Z']})
    enrichment = saved_cluster_enrichment(members, population, {'B', 'A'})
    pairs, clusters = cluster_overlap_graph(members, max_edges=None)
    filtered, f_pairs = moonlighting_cluster_view(clusters, pairs, enrichment)
    evidence = pl.DataFrame({
        'community': ['1', '2'],
        'subset': ['full', 'full'],
        'library': ['GO_BP', 'Reactome'],
        'term_id': ['GO:0001', 'R-HSA-1'],
        'term_name': ['<signaling>', 'metabolism'],
        'qvalue': [0.004, 0.25],
        'fold_enrichment': [3.5, 2.0],
    })
    nodes_ov, edges_ov, stats_ov = build_annotated_ocg_overlap_view(
        filtered, f_pairs, evidence, focus=1, metric='jaccard', protein_set_label='MoonLitDB',
    )
    assert stats_ov['displayed_clusters'] == 2
    assert edges_ov.height == 1
    c1 = nodes_ov.filter(pl.col('id') == 'C1').row(0, named=True)
    assert c1['node_role'] == 'Linker'
    assert '<signaling> **' in c1['top_term']
    assert 'MoonLitDB hits:' in c1['tooltip']

    metadata = {'B': {'gene_name': 'GENE_B', 'protein_name': 'Bridge B'}}
    nodes_bp, edges_bp, stats_bp = build_annotated_ocg_bipartite_view(
        filtered, f_pairs, members, evidence, cohort={'B'}, focus=1,
        metadata=metadata, protein_set_label='MoonLitDB',
    )
    assert stats_bp['bridge_proteins'] == 1
    assert stats_bp['membership_links'] == 2
    b_row = nodes_bp.filter(pl.col('id') == 'B').row(0, named=True)
    assert b_row['label'] == 'GENE_B'
    assert b_row['shape'] == 'diamond'

    for n_df, e_df in ((nodes_ov, edges_ov), (nodes_bp, edges_bp)):
        gml = export_graphml_string(n_df, e_df)
        assert '<graphml' in gml
        zbytes = export_cytoscape_bundle_zip(n_df, e_df, title='ocg_test')
        with zipfile.ZipFile(BytesIO(zbytes)) as zf:
            names = zf.namelist()
            assert any(name.endswith('.graphml') for name in names)
            assert any(name.endswith('.sif') for name in names)
            assert any(name.endswith('.cx2') for name in names)


def test_annotated_matrix_and_core_interface_landscape_preserve_bh_stars():
    from clustering_explorer import saved_cluster_enrichment
    from network_view import static_image_bytes
    from ocg_annotation_ui import moonlighting_cluster_view
    from ocg_visualization import annotated_ocg_overlap_matrix_figure, ocg_core_interface_landscape_figures

    members = pl.DataFrame({
        'protein': ['A', 'B', 'C', 'B', 'D', 'E', 'C', 'F'],
        'community': [1, 1, 1, 2, 2, 2, 3, 3],
    })
    edges = pl.DataFrame({'a': ['A', 'B', 'C', 'D', 'E'], 'b': ['B', 'C', 'D', 'E', 'F']})
    population = pl.DataFrame({'protein': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']})
    enrichment = saved_cluster_enrichment(members, population, {'A', 'B', 'C'})
    pairs, clusters = cluster_overlap_graph(members, max_edges=None)
    filtered, f_pairs = moonlighting_cluster_view(clusters, pairs, enrichment)
    evidence = pl.DataFrame({
        'community': ['1'], 'subset': ['full'], 'library': ['GO_BP'],
        'term_id': ['GO:0001'], 'term_name': ['glycolysis'],
        'qvalue': [0.0005], 'fold_enrichment': [4.0],
    })

    fig_mat, static_mat, mat_df = annotated_ocg_overlap_matrix_figure(
        filtered, f_pairs, evidence, limit=10, metric='jaccard', protein_set_label='MoonLitDB',
    )
    assert fig_mat.data[0].z is not None
    assert mat_df.height == 3
    assert any('***' in lbl for lbl in mat_df['axis_label'].to_list())
    assert static_image_bytes(static_mat, 'png').startswith(b'\x89PNG')
    assert b'<svg' in static_image_bytes(static_mat, 'svg')

    fig_bar, fig_scatter, static_comp, landscape_df = ocg_core_interface_landscape_figures(
        filtered, members, edges=edges, evidence=evidence, protein_set_label='MoonLitDB',
    )
    assert len(fig_bar.data) == 2
    assert len(fig_scatter.data) == 1
    assert 'internal_density' in landscape_df.columns
    c1 = landscape_df.filter(pl.col('community') == 1).row(0, named=True)
    assert c1['core_proteins'] == 1 and c1['interface_proteins'] == 2
    assert static_image_bytes(static_comp, 'png').startswith(b'\x89PNG')
    assert b'<svg' in static_image_bytes(static_comp, 'svg')
