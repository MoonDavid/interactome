"""Persist generated analysis figures with descriptive names and provenance."""
import hashlib
import json
from pathlib import Path
import re


def _slug(value, limit):
    return re.sub(r'[^\w.-]+', '_', str(value), flags=re.ASCII).strip('_.')[:limit] or 'plot'


def save_analysis_plot(figure, directory, name, set_name, networks, context):
    """Save Plotly as standalone HTML, or Matplotlib as PNG/SVG, plus metadata.

    The content digest keeps different settings/results from overwriting one another.
    Identical reruns reuse existing files.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    interactive = hasattr(figure, 'to_plotly_json')
    payload = figure.to_json() if interactive else repr([
        (ax.get_title(), [(list(line.get_xdata()), list(line.get_ydata()))
                          for line in ax.lines], [text.get_text() for text in ax.texts])
        for ax in figure.axes
    ])
    metadata = dict(context, plot=name, protein_set_name=set_name, networks=list(networks),
                    format='interactive_html' if interactive else 'static_png_svg')
    digest = hashlib.sha256((payload + json.dumps(metadata, sort_keys=True, default=str)).encode()).hexdigest()[:12]
    stem = '__'.join((_slug(name, 65), _slug(set_name, 22),
                      _slug('_'.join(networks), 52), digest))
    target = directory / stem
    if interactive:
        path = target.with_name(target.name + '.html')
        if not path.exists():
            figure.write_html(path, include_plotlyjs=True, full_html=True)
        paths = [path]
    else:
        paths = [target.with_name(target.name + suffix) for suffix in ('.png', '.svg')]
        for path in paths:
            if not path.exists():
                figure.savefig(path, dpi=180, bbox_inches='tight')
    metadata['files'] = [path.name for path in paths]
    target.with_name(target.name + '.json').write_text(
        json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    return paths
