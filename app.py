"""Navigation entry point for the Human Interactome Atlas."""
import streamlit as st

st.set_page_config(page_title='Human Interactome Atlas', page_icon=':material/hub:', layout='wide')
page = st.navigation([
    st.Page('pages/1_Network_Explorer.py', title='Network Explorer', icon=':material/hub:', default=True),
    st.Page('pages/2_Moonlighting_Analysis.py', title='Moonlighting Explorer', icon=':material/science:'),
    st.Page('pages/3_Cytoscape_Bridge.py', title='Cytoscape Bridge', icon=':material/share:'),
])
page.run()
