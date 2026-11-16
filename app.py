"""Navigation entry point for the Human Interactome Atlas."""

if __name__ != '__mp_main__':
    from pathlib import Path
    from runpy import run_path

    import streamlit as st
    from networks import ISOFORM_POLICIES, prepare_isoform_views

    ROOT = Path(__file__).resolve().parent


    def visualization_page():
        st.title('Network Visualization / Cytoscape')
        viewer = st.segmented_control(
            'Workspace', ['Network viewers', 'Cytoscape Desktop'],
            default='Network viewers', required=True, key='visualization_workspace',
        )
        script = {'Network viewers': '4_Network_Visualization.py',
                  'Cytoscape Desktop': '3_Cytoscape_Bridge.py'}[viewer]
        run_path(str(ROOT / 'pages' / script))


    st.set_page_config(page_title='Human Interactome Atlas', page_icon=':material/hub:', layout='wide')
    navigation_pages = {
        'Main analysis': [
        st.Page('pages/1_Interactomes_overview.py', title='Interactomes Overview', icon=':material/hub:', default=True),
        st.Page('pages/2_Topological_analysis.py', title='Topological Analysis', icon=':material/science:'),
        ],
        'Other analyses': [
        st.Page('pages/7_Reports.py', title='Cluster reports', icon=':material/description:',
                help='Explore saved cluster reports by functional theme and inspect MoonLitDB proteins.'),
        st.Page('pages/5_Precomputed_clusters.py', title='Clustering and communities', icon=':material/account_tree:'),
        st.Page('pages/OCG_exploration.py', title='OCG Exploration', icon=':material/workspaces:'),
        st.Page('pages/6_Network_embeddings.py', title='Network embeddings', icon=':material/scatter_plot:'),
        st.Page(visualization_page, title='Network Visualization / Cytoscape', icon=':material/polyline:',
                url_path='network-visualization'),
        ],
    }
    page = st.navigation(navigation_pages, position='hidden')
    with st.sidebar:
        for main_page in navigation_pages['Main analysis']:
            st.page_link(main_page)
        with st.expander('Other analyses', expanded=False):
            for other_page in navigation_pages['Other analyses']:
                st.page_link(other_page)
        policy = st.segmented_control(
            'Interactome isoform handling', tuple(ISOFORM_POLICIES), default='collapse', required=True,
            format_func=ISOFORM_POLICIES.__getitem__, key='isoform_policy', wrap=True,
            help='Applies to every interactome across the app. Unsuffixed only drops source pairs with an explicit isoform suffix, including -1.',
        )
        if policy == 'retain':
            st.caption('Node IDs match exactly. Moonlighting protein set IDs are parent accessions. '
                       'Mapped networks keep documented isoform matches and exclude parent-only fallbacks.')
        if policy != 'collapse':
            try:
                with st.spinner('Checking isoform views…'):
                    unavailable = prepare_isoform_views(ROOT / 'data', policy)
                if unavailable:
                    st.warning(f'Isoform views are unavailable for {", ".join(unavailable)}. '
                               'Check the source files and mapping audit under Mapping & provenance.')
            except (OSError, ValueError, KeyError) as exc:
                st.warning(f'Could not prepare every isoform view. {exc}')
    page.run()



