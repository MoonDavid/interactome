"""Display saved outputs from the bundled Jupyter notebooks."""
import base64
import json
from pathlib import Path

import streamlit as st


def show_notebook(path: Path, title: str, caption: str, *, code_key: str) -> None:
    st.title(title)
    st.caption(caption)

    notebook_bytes = path.read_bytes()
    st.download_button('Download Jupyter notebook', notebook_bytes, path.name,
                       mime='application/x-ipynb+json', on_click='ignore', icon=':material/download:')
    show_code = st.checkbox('Show Python cells', key=code_key)

    for cell in json.loads(notebook_bytes)['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            if not source.startswith('# '):
                st.markdown(source)
        elif cell['cell_type'] == 'code':
            if show_code:
                st.code(source, language='python')
            for output in cell['outputs']:
                data = output.get('data', {})
                if 'image/png' in data:
                    st.image(base64.b64decode(''.join(data['image/png'])), width=760,
                             alt=output.get('metadata', {}).get('alt'))
                elif 'text/markdown' in data:
                    st.markdown(''.join(data['text/markdown']))
                elif output['output_type'] == 'stream':
                    st.code(''.join(output['text']), language=None)
