"""Assemble inspectable report evidence from the completed community study."""
from pathlib import Path
import sys
import json
import hashlib
import argparse
from collections import Counter
sys.path.append(str(Path(__file__).resolve().parents[1] / '.venv/Lib/site-packages'))
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from moonlighting import significance_label
OUT = ROOT / 'data/analysis_runs/moonlitdb_small_community_study'


def main(source='moonlitdb'):
    OUT = ROOT / f'data/analysis_runs/{source}_small_community_study'
    manifest = json.loads((OUT / 'study_manifest.json').read_text())
    clusters, summary, themes, matches = [], [], [], []
    for network in manifest['networks']:
        name = network['network']
        folder = OUT / name
        selected = pl.read_csv(folder / 'selected_clusters.csv', schema_overrides={'community': pl.String})
        annotations = {}
        for bundle in json.loads((folder / 'annotation_bundles.json').read_text()):
            target = Path(bundle['annotation_directory'])
            labels = pl.read_csv(target / 'label_summary.csv', schema_overrides={'community': pl.String})
            evidence = pl.read_parquet(target / 'enrichment.parquet')
            members = pl.read_parquet(Path(bundle['directory']) / 'memberships.parquet').with_columns(pl.col('community').cast(pl.String))
            for row in labels.iter_rows(named=True):
                community = row['community']
                proteins = sorted(members.filter(pl.col('community') == community)['protein'].unique().to_list())
                terms = evidence.filter((pl.col('community') == community) & (pl.col('subset') == 'full') & (pl.col('qvalue') < .05)).sort('qvalue')
                terms = terms.with_columns(pl.col('qvalue').map_elements(significance_label, return_dtype=pl.String).alias('significance'))
                annotations[bundle['directory'], community] = dict(
                    **row, members=proteins, membership_signature=hashlib.sha256('\n'.join(proteins).encode()).hexdigest(),
                    terms=terms.select('library', 'term_id', 'term_name', 'query_hits', 'query_size', 'fold_enrichment', 'pvalue', 'qvalue', 'significance').to_dicts(),
                    annotation_directory=str(target))
        rows = []
        for row in selected.iter_rows(named=True):
            annotation = annotations[row['directory'], row['community']]
            assert len(annotation['members']) == row['proteins']
            rows.append({**annotation, **row, 'label': annotation.get('GO_BP_enrichment') or annotation.get('GO_CC_enrichment') or 'No significant functional label'})
        clusters.extend(rows)
        unique = {row['membership_signature']: row for row in reversed(rows)}
        bp = Counter(row.get('GO_BP_enrichment') or 'No significant GO BP label' for row in unique.values())
        for term, count in bp.most_common():
            themes.append(dict(network=name, term=term, clusters=count))
        proteins = set().union(*(set(row['members']) for row in rows)) if rows else set()
        summary.append(dict(**network, unique_memberships=len(unique), unique_proteins=len(proteins),
                            background=rows[0]['background_proteins'],
                            **{source: rows[0]['protein_set_in_network']},
                            methods=len({row['method'] for row in rows}), top_label=rows[0]['label'],
                            top_hits=rows[0]['protein_set_hits'], top_size=rows[0]['proteins'], top_fold=rows[0]['fold_enrichment']))
    left = [r for r in clusters if r['network'] == 'BioPlex']
    right = [r for r in clusters if r['network'] != 'BioPlex']
    seen = set()
    for a in left:
        aset = set(a['members'])
        for b in right:
            bset = set(b['members'])
            j = len(aset & bset) / len(aset | bset)
            key = a['membership_signature'], b['membership_signature']
            if j >= .5 and key not in seen:
                seen.add(key)
                matches.append(dict(bioplex_rank=a['rank'], tagc_rank=b['rank'], shared=len(aset & bset),
                                    jaccard=j, bioplex_label=a['label'], tagc_label=b['label']))
    matches.sort(key=lambda row: (-row['jaccard'], -row['shared']))
    # Independently verify selected p-values, enrichment ratios and strict size bound.
    import scipy.stats
    for row in clusters:
        assert 0 < row['proteins'] < manifest.get('max_proteins_exclusive', 100) and row['qvalue'] < .05
        p = scipy.stats.hypergeom.sf(row['protein_set_hits'] - 1, row['background_proteins'], row['protein_set_in_network'], row['proteins'])
        assert abs(p - row['pvalue']) <= max(1e-300, p * 1e-10)
        expected = row['proteins'] * row['protein_set_in_network'] / row['background_proteins']
        assert abs(row['fold_enrichment'] - row['protein_set_hits'] / expected) < 1e-9
    data = dict(summary=summary, clusters=clusters, themes=themes, matches=matches)
    (OUT / 'reviewed_evidence.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (OUT / 'validation_summary.json').write_text(json.dumps(dict(
        summary=summary, validated_clusters=len(clusters),
        max_proteins_exclusive=manifest.get('max_proteins_exclusive', 100),
        checks=['Strict protein-count bound and global q < 0.05',
                'Membership counts match saved cluster sizes',
                'Hypergeometric p-values independently recomputed',
                'Fold enrichment independently recomputed']), indent=2), encoding='utf-8')
    snapshot_path = OUT / 'report/src/data.json'
    if not snapshot_path.is_file():
        print(json.dumps(dict(summary=summary, report='Available in Streamlit Cluster reports'), indent=2))
        return
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    snapshot.update(generatedAt=manifest['generated_at'], status='reviewed', buildStatus='complete', filters=[])
    snapshot['queries'] = {key: dict(rows=rows, source=dict(
        label='Local saved clustering and functional annotation study',
        files=[str(OUT / 'study_manifest.json'), str(OUT / 'reviewed_evidence.json')],
        description='MoonLitDB; global community BH within each network before filters; annotation BH across terms/libraries/subsets of selected clusters within each saved run.'))
        for key, rows in data.items()}
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    (OUT / 'report_snapshot.json').write_text(json.dumps(dict(
        title=snapshot.get('title'), status='reviewed', buildStatus='complete',
        generatedAt=manifest['generated_at'], selected_clusters=len(clusters)), indent=2), encoding='utf-8')
    print(json.dumps(dict(summary=summary, themes=themes, matches=matches[:8]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protein-set', choices=['moonlitdb', 'moonprot'], default='moonlitdb')
    main(parser.parse_args().protein_set)
