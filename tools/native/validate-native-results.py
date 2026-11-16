import json
from pathlib import Path

import polars as pl

from networks import _fingerprint, load_network

root = Path(__file__).resolve().parents[2]
output = root / 'data/analysis_runs/clustering_native'
rows = json.loads((output / 'summary.json').read_text())
checks = []
for row in rows:
    if row['status'] != 'success':
        continue
    target = Path(row['target'])
    result = json.loads((target / 'result.json').read_text())
    source = Path(result['job']['path'])
    assert _fingerprint(source)['sha256'] == result['job']['provenance']['input']['sha256']
    edges = load_network(source).edges
    proteins = set(edges['a']) | set(edges['b'])
    memberships = pl.read_parquet(target / 'memberships.parquet')
    sizes = pl.read_parquet(target / 'community_sizes.parquet')
    assigned = memberships.filter(pl.col('community').is_not_null())
    actual = assigned.group_by('community').len().sort('community')
    assert actual.equals(sizes)
    assert memberships.unique().height == memberships.height
    assert set(memberships['protein']) == proteins
    assert result['nodes'] == len(proteins)
    assert result['edges'] == edges.height
    assert result['communities'] == sizes.height
    assert result['covered_nodes'] == assigned['protein'].n_unique()
    assert result['unassigned_nodes'] == len(proteins) - assigned['protein'].n_unique()
    assert result['overlapping_nodes'] == assigned.group_by('protein').len().filter(pl.col('len') > 1).height
    checks.append({'network': row['network'], 'method': row['method'], 'validated': True})
pl.from_dicts(rows, infer_schema_length=None).write_csv(output / 'summary.csv')
(output / 'validation.json').write_text(json.dumps(checks, indent=2))
print(f'Validated all {len(checks)} successful outputs against input proteins, edge counts, memberships and community sizes.')
