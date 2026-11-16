"""One refresh timer per analysis, with an immutable snapshot of pending jobs."""
import streamlit as st


@st.fragment(run_every='2s')
def watch_topology_jobs(jobs):
    """Refresh results when any captured job completes, including failures."""
    if any(job.done() for job in jobs):
        st.rerun(scope='app')
    st.caption('Updating missing topological metrics in the background. '
               'Compatible cached results remain visible; adjusted values update when complete.')
