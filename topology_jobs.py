"""Background computation without Streamlit calls or session-state access."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix='topology')
_JOBS = {}
_LOCK = Lock()


def submit_once(signature, compute, refresh=False):
    """Share an in-flight job across reruns; refresh replaces only completed work."""
    with _LOCK:
        job = _JOBS.get(signature)
        if job is None or (refresh and job.done()):
            job = _EXECUTOR.submit(compute)
            _JOBS[signature] = job
        # Keep in-flight jobs, but bound completed references across changed inputs.
        completed = [key for key, value in _JOBS.items() if value.done() and key != signature]
        for key in completed[:-8]:
            del _JOBS[key]
        return job
