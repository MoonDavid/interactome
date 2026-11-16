"""Exercise background polling across job completion and full-page reruns."""
from streamlit.testing.v1 import AppTest


def test_polling_tracks_each_pending_job_and_survives_full_reruns():
    app = AppTest.from_string('''
import streamlit as st
from concurrent.futures import Future
from topology_polling import watch_topology_jobs
if 'jobs' not in st.session_state:
    st.session_state.jobs = (Future(), Future())
jobs = st.session_state.jobs
if st.button('Finish first'):
    jobs[0].set_result(None)
if st.button('Finish second'):
    jobs[1].set_result(None)
pending = tuple(job for job in jobs if not job.done())
if pending:
    watch_topology_jobs(pending)
st.write(f'Pending: {len(pending)}')
''')
    app.run()
    assert not app.exception
    assert len(app.caption) == 1
    assert app.markdown[-1].value == 'Pending: 2'
    app.button[0].click().run()
    assert not app.exception
    assert len(app.caption) == 1
    assert app.markdown[-1].value == 'Pending: 1'
    app.button[1].click().run()
    assert not app.exception
    assert len(app.caption) == 0
    assert app.markdown[-1].value == 'Pending: 0'
