"""Display the bundled HuRI walkthrough and its saved Jupyter outputs."""
from pathlib import Path

from app_pages.notebook_viewer import show_notebook

show_notebook(
    Path(__file__).resolve().parents[1] / 'notebooks' / 'huri_psi.ipynb',
    'HuRI PSI-MITAB notebook',
    'Saved notebook run covering all three isoform policies. Rerun in Jupyter to explore another accession or a different source file.',
    code_key='huri_notebook_show_code',
)
