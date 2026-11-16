import io
import json
from zipfile import ZipFile

import polars as pl

from ocg_ego_overlap import ego_groups, ego_overlap_figure, norma_payload, norma_export_zip, send_to_norma


def sample():
    return (pl.DataFrame({'protein':['A','B','C','D']}),
            pl.DataFrame({'a':['A','A','A'], 'b':['B','C','D']}),
            pl.DataFrame({'protein':['A','B','A','C','D','OUTSIDE'], 'community':[1,1,2,2,3,1]}))


def test_limits_preserve_overlaps_and_full_memberships():
    nodes, edges, members = sample()
    groups, ranked = ego_groups(nodes, members, 'A', 2)
    assert groups == {'1':['A','B'], '2':['A','C']}
    assert ranked.height == 2
    limited, ranked = ego_groups(nodes, members, 'A', 1, 'all')
    assert list(limited) == ['1'] and ranked.height == 3
    payload = norma_payload(nodes, edges, groups, 'A')
    assert len(payload['nodes']) == 4
    assert all('A' in g['members'] for g in payload['groups'])
    assert 'OUTSIDE' not in json.dumps(payload)
    with ZipFile(io.BytesIO(norma_export_zip(payload))) as archive:
        assert archive.read('annotations.tsv').decode() == 'OCG 1\tA,B\nOCG 2\tA,C\n'
        assert archive.read('network.tsv').decode().startswith('Source\tTarget\n')
        assert json.loads(archive.read('norma_api_payload.json')) == payload


def test_hulls_handle_singletons_pairs_and_no_outlines():
    nodes, edges, members = sample()
    groups, _ = ego_groups(nodes, members, 'A', 5, 'all')
    fig = ego_overlap_figure(nodes, edges, groups, 'A')
    assert len([t for t in fig.data if t.fill == 'toself']) == 3
    assert list(fig.data[-1].customdata) == ['A','B','C','D']
    assert 'C1, C2' in fig.data[-1].hovertext[0]
    assert len(ego_overlap_figure(nodes, edges, {}, 'A').data[-1].x) == 4


def test_norma_api_contract(monkeypatch):
    import requests
    calls = []
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'url':'/norma.html?session=test'}
    monkeypatch.setattr(requests, 'post', lambda url, **kwargs: (calls.append((url,kwargs)) or Response()))
    assert send_to_norma({'nodes':[]}, 'http://localhost:8000') == 'http://localhost:8000/norma.html?session=test'
    assert calls[0] == ('http://localhost:8000/api/external', {'json':{'nodes':[]}, 'timeout':30})
