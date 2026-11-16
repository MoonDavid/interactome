"""Annotate existing human UniProt clusters with GO and Reactome."""
from __future__ import annotations

import argparse
from collections import Counter
from io import StringIO
import math
import re
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import polars as pl
import requests
from scipy.stats import hypergeom

from moonlighting import _ontology, bh_adjust, prepare_go_annotations
from networks import _fingerprint, _write_json, _write_parquet, discover_datasets, load_network


DATA = Path(__file__).resolve().parent / 'data'
DEFAULT_GAF = DATA / 'GO' / 'HUMAN-uniprot.gaf.gz'
DEFAULT_OBO = DATA / 'GO' / 'go-basic.obo'
DEFAULT_REACTOME = DATA / 'reactome' / 'UniProt2Reactome.txt'
LIBRARIES = ('GO_BP', 'GO_MF', 'GO_CC', 'Reactome')
TERM = ['library', 'term_id', 'term_name']
QUERY = ['community', 'subset']
MIN_BACKGROUND_HITS = 5
MIN_LABEL_HITS = 2
MAX_QVALUE = 0.05
ANCESTOR_OVERLAP = 0.70
UNIPROT_ACCESSION = r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$'
AUTOANNOTATE_COMMAND = 'commands/autoannotate/annotate-clusterBoosted'
DEFAULT_GENERIF = DATA / 'generif' / 'generifs_basic.gz'
DEFAULT_GENE_MAPPING = DATA / 'mappings' / 'generif_geneid_uniprot.json'
DEFAULT_ABSTRACTS = DATA / 'generif' / 'abstracts.tsv'
DEFAULT_FUNCTIONS = DATA / 'annotations' / 'uniprot_functions.tsv'
TEXT_STOPWORDS = frozenset('a an the and or of to in on for by with from as at is are be '
    'may can that this which it its also function functions protein proteins involved '
    'plays role required activity similar similarity probably contains belongs family'.split())


def fetch_uniprot_functions(path):
    """Persist reviewed human Swiss-Prot FUNCTION comments without overwriting."""
    path = Path(path)
    if path.exists():
        return
    url = 'https://rest.uniprot.org/uniprotkb/search'
    params = {'query': 'organism_id:9606 AND reviewed:true', 'format': 'tsv',
              'fields': 'accession,cc_function', 'size': 500}
    pages, expected, release = [], None, None
    with requests.Session() as session:
        while url:
            for attempt in range(3):
                try:
                    response = session.get(url, params=params, timeout=(10, 60))
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
            table = pl.read_csv(StringIO(response.text), separator='\t', infer_schema=False)
            if not {'Entry', 'Function [CC]'} <= set(table.columns):
                raise ValueError('UniProt response is missing accession or FUNCTION columns')
            current_release = response.headers.get('X-UniProt-Release')
            if expected is None:
                expected = int(response.headers['X-Total-Results'])
                release = current_release
            elif release != current_release:
                raise ValueError('UniProt release changed during download; retry the snapshot')
            pages.append(table)
            if len(pages) % 10 == 0:
                print(f'UniProt FUNCTION: {sum(p.height for p in pages):,}/{expected:,} entries', flush=True)
            url, params = response.links.get('next', {}).get('url'), None
    table = pl.concat(pages)
    if table.height != expected or table['Entry'].n_unique() != expected:
        raise ValueError('Incomplete or duplicate UniProt snapshot; no TSV written')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tsv.part')
    table.write_csv(temporary, separator='\t')
    temporary.rename(path)
    _write_json(path.with_suffix('.json'), {
        'query': 'organism_id:9606 AND reviewed:true', 'endpoint': 'https://rest.uniprot.org/uniprotkb/search',
        'release': release, 'entries': expected,
        'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'snapshot': _fingerprint(path)})


def generif_documents(path, mapping_path=DEFAULT_GENE_MAPPING, abstracts_path=None):
    """Load human GeneRIF statements and optional local PMID/abstract text, offline."""
    mapping_source = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    if mapping_source.get('taxon') != 9606:
        raise ValueError('GeneRIF requires a human GeneID-to-UniProt mapping')
    pairs = [(gene, protein) for gene, proteins in mapping_source['mapping'].items()
             for protein in proteins]
    mapping = pl.DataFrame(pairs, schema=['gene', 'protein'], orient='row')
    raw = pl.read_csv(path, separator='\t', infer_schema=False, quote_char=None).rename({
        '#Tax ID': 'taxon', 'Gene ID': 'gene', 'GeneRIF text': 'function',
        'PubMed ID (PMID) list': 'pmids'})
    human = raw.filter(pl.col('taxon') == '9606').join(mapping, on='gene', how='inner')
    statements = human.select('protein', 'function').unique()
    if abstracts_path is not None:
        abstracts = pl.read_csv(abstracts_path, separator='\t', infer_schema=False)
        links = (human.select('protein', pl.col('pmids').str.split(',').alias('pmid'))
                 .explode('pmid').with_columns(pl.col('pmid').str.strip_chars()))
        linked = links.join(abstracts.select('pmid', pl.col('abstract').alias('function')),
                            on='pmid', how='inner').select('protein', 'function').unique()
        statements = pl.concat([statements, linked]).unique()
    return statements, {'source': _fingerprint(Path(path).resolve()),
        'mapping': _fingerprint(Path(mapping_path).resolve()),
        'mapped_human_proteins': human['protein'].n_unique(),
        'abstracts': _fingerprint(Path(abstracts_path).resolve()) if abstracts_path else None,
        'limitations': 'GeneRIF statements are not abstracts; mapping coverage may be incomplete; '
                      'one-to-many GeneID mappings retain all candidate accessions.'}


def _text_features(text, max_ngram):
    """Identical tokenization for network text and informative proteome prior."""
    text = re.sub(r'\{[^}]*\}|\([^)]*(?:PubMed|By similarity)[^)]*\)', ' ', text)
    features = Counter()
    for sentence in re.split(r'[.;:!?]', text.lower()):
        tokens = re.findall(r'[a-z][a-z0-9-]*', sentence)
        for size in range(1, max_ngram + 1):
            for i in range(len(tokens) - size + 1):
                phrase = tokens[i:i + size]
                if all(len(w) > 2 and w not in TEXT_STOPWORDS for w in phrase):
                    features[' '.join(phrase)] += 1
    return features


def function_text_labels(queries, population, path, *, generif=None, gene_mapping=DEFAULT_GENE_MAPPING, abstracts=None, reference_queries=None):
    """Rank protein text and cluster pseudo-documents with auditable descriptive scores."""
    schema = {'community': pl.String, 'subset': pl.String, 'method': pl.String,
              'label': pl.String, 'score': pl.Float64, 'protein_hits': pl.UInt32,
              'query_size': pl.UInt32, 'annotated_query': pl.UInt32,
              'background_hits': pl.UInt32, 'proteins': pl.List(pl.String)}
    if path is None and generif is None:
        return pl.DataFrame(schema=schema), {'status': 'not_requested'}
    raw = pl.read_csv(path, separator='\t', infer_schema=False) if path else pl.DataFrame(schema={'protein': pl.String, 'function': pl.String})
    raw = raw.rename({key: value for key, value in
        [('Entry', 'protein'), ('Function [CC]', 'function'), ('cc_function', 'function')]
        if key in raw.columns})
    if not {'protein', 'function'} <= set(raw.columns):
        raise ValueError('FUNCTION TSV requires Entry/Function [CC] or protein/function columns')
    prior_raw = raw.clone()
    gene_manifest = {'status': 'not_requested'}
    if generif is not None:
        gene_text, gene_manifest = generif_documents(generif, gene_mapping, abstracts)
        raw = pl.concat([raw.select('protein', 'function'), gene_text]).unique()
    all_documents = (raw.select(protein_id(), 'function')
        .filter(pl.col('function').is_not_null()).group_by('protein')
        .agg(pl.col('function').unique().str.join('. ')))
    documents = (all_documents.join(population, on='protein', how='semi')
        .filter(pl.col('function').is_not_null()).group_by('protein')
        .agg(pl.col('function').unique().str.join(' ')))
    max_ngram = 4 if generif is not None else 2
    counts = {protein: features for protein, text in documents.iter_rows()
              if (features := _text_features(text, max_ngram))}
    prior = Counter()
    for text in prior_raw['function'].drop_nulls():
        prior.update(_text_features(text, max_ngram))
    background = Counter()
    for document in counts.values():
        background.update(document)
    background_total = sum(background.values())
    reference_queries = queries if reference_queries is None else reference_queries
    full_groups = reference_queries.filter(pl.col('subset') == 'full').group_by('community').agg(pl.col('protein'))
    average_words = sum(sum(v for t, v in counts.get(p, {}).items() if ' ' not in t) for ps in full_groups['protein'] for p in ps) / max(1, full_groups.height)
    prior_denominator = sum(prior.values()) + 0.5 * len(set(prior) | set(background))
    try:
        from yake import KeywordExtractor
        extractor = KeywordExtractor(lan='en', n=4, top=100, dedupLim=0.9) if generif is not None else None
        keyphrase_status = 'available'
    except ImportError:
        extractor = None
        keyphrase_status = 'unavailable: install interactome[text-annotation] for YAKE'
    df = Counter(term for document in counts.values() for term in document)
    n = len(counts)
    rows = []
    for key, group in queries.group_by(QUERY):
        community, subset = key
        proteins = sorted(set(group['protein']))
        annotated = [p for p in proteins if p in counts]
        supporters = {}
        tfidf = Counter()
        for protein in annotated:
            document = counts[protein]
            weights = {term: (1 + math.log(freq)) * (1 + math.log((1+n)/(1+df[term])))
                       for term, freq in document.items()}
            norm = math.sqrt(sum(w*w for w in weights.values()))
            for term, weight in weights.items():
                supporters.setdefault(term, []).append(protein)
                tfidf[term] += weight / norm / len(proteins)
        cluster_counts = Counter()
        for protein in annotated:
            cluster_counts.update(counts[protein])
        cluster_total = sum(cluster_counts.values())
        cluster_words = sum(v for t, v in cluster_counts.items() if ' ' not in t)
        yake_scores = {phrase.lower(): 1 / (1 + score) for phrase, score in extractor.extract_keywords(
            '. '.join(documents.filter(pl.col('protein').is_in(annotated))['function']))
            if ' ' in phrase} if extractor else {}
        methods = ['tfidf', 'document_lift'] + (['ctfidf', 'log_odds'] if generif is not None else []) + (['yake'] if extractor else [])
        for method in methods:
            if method in ('document_lift', 'log_odds') and n == len(annotated):
                continue  # No annotated complement is available for comparison.
            candidates = []
            for term, support in supporters.items():
                if len(support) < MIN_LABEL_HITS:
                    continue
                outside_hits = df[term] - len(support)
                score = (tfidf[term] if method == 'tfidf' else
                    math.log2(((len(support)+0.5)/(len(annotated)+1)) /
                              ((outside_hits+0.5)/(n-len(annotated)+1))))
                if method == 'yake':
                    score = yake_scores.get(term, 0.0)
                elif method == 'ctfidf':
                    score = cluster_counts[term] / max(1, cluster_words) * math.log1p(average_words / background[term])
                elif method == 'log_odds':
                    if not prior:
                        continue
                    # Informative prior: human UniProt corpus; additive vocabulary smoothing.
                    alpha0 = 1000.0
                    alpha = alpha0 * (prior[term] + 0.5) / prior_denominator
                    inside = cluster_counts[term]
                    outside = background[term] - inside
                    outside_total = background_total - cluster_total
                    score = math.log((inside + alpha) / (cluster_total - inside + alpha0 - alpha)) - math.log(
                        (outside + alpha) / (outside_total - outside + alpha0 - alpha))
                if score > 0:
                    candidates.append((term, score, support))
            for term, score, support in sorted(candidates, key=lambda x: (-x[1], x[0]))[:10]:
                rows.append(dict(community=community, subset=subset, method=method,
                    label=term, score=score, protein_hits=len(support), query_size=len(proteins),
                    annotated_query=len(annotated), background_hits=df[term], proteins=support))
    table = pl.from_dicts(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return table, {'status': 'complete', 'source': _fingerprint(Path(path).resolve()) if path else None,
        'generif': gene_manifest, 'keyphrases': keyphrase_status,
        'log_odds_status': 'available' if prior else 'unavailable: no UniProt prior corpus',
        'ctfidf_average_words': average_words,
        'prior_annotated_proteins': prior_raw.filter(pl.col('function').is_not_null())['protein'].n_unique(), 'prior_strength': 1000.0,
        'log_odds_units': 'natural log-odds over the combined n-gram feature vocabulary',
        'annotated_background': n, 'population': population.height,
        'policy': 'UniProt FUNCTION plus optional human GeneRIF/local linked abstracts; ctfidf uses full-cluster average word count and network term frequencies; log_odds uses human UniProt prior and network complement; 1-4 grams with GeneRIF; evidence references removed; lowercase unigrams and bigrams; '
        'fixed stopwords; sublinear TF, smoothed protein-document IDF, L2 normalization, '
        'mean over all subset proteins; document_lift uses annotated subset vs annotated complement '
        'with 0.5/1 smoothing; >=2 supporting proteins; top 10 positive scores per method/subset; '
        'descriptive scores, no p-values or significance; phrases may be redundant'}


def protein_id(column: str = 'protein') -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strip_chars().str.to_uppercase().str.replace(r'-\d+$', '')


def membership_queries(memberships: pl.DataFrame, population: pl.DataFrame) -> pl.DataFrame:
    if not {'protein', 'community'} <= set(memberships.columns):
        raise ValueError('Memberships require protein and community columns')
    pairs = (
        memberships.select(protein_id(), pl.col('community').cast(pl.String).str.strip_chars())
        .filter(pl.col('community').is_not_null() & (pl.col('community') != ''))
        .unique()
    )
    if population.is_empty():
        raise ValueError('The network background contains no UniProt proteins')
    if pairs.is_empty():
        raise ValueError('No assigned cluster memberships were found')
    outside = pairs.join(population, on='protein', how='anti')['protein'].unique().sort()
    if len(outside):
        examples = ', '.join(p or '<missing>' for p in outside.head(5))
        raise ValueError(f'{len(outside)} assigned accession(s) outside the network background: {examples}')
    split = pairs.with_columns(
        pl.when(pl.len().over('protein') == 1).then(pl.lit('core'))
        .otherwise(pl.lit('interface')).alias('subset')
    )
    return pl.concat([pairs.with_columns(pl.lit('full').alias('subset')), split])


def load_annotations(go_path: Path, reactome_path: Path, population: pl.DataFrame) -> pl.DataFrame:
    go = pl.scan_parquet(go_path).select(
        'protein', pl.col('term').alias('term_id'), pl.col('name').alias('term_name'),
        pl.col('aspect').replace_strict({'P': 'GO_BP', 'F': 'GO_MF', 'C': 'GO_CC'}, default=None)
        .alias('library'),
    ).filter(pl.col('library').is_not_null())
    reactome = (
        pl.scan_csv(reactome_path, separator='\t', has_header=False, quote_char=None, infer_schema=False,
                    new_columns=['protein', 'term_id', 'url', 'term_name', 'evidence', 'species'])
        .filter((pl.col('species') == 'Homo sapiens') & pl.col('term_id').str.starts_with('R-HSA-'))
        .select(protein_id(), 'term_id', 'term_name', pl.lit('Reactome').alias('library'))
    )
    return (
        pl.concat([go.select('protein', *TERM), reactome.select('protein', *TERM)])
        .join(population.lazy(), on='protein', how='semi')
        .filter(pl.col('term_name').is_not_null() & (pl.col('term_name') != ''))
        # Use one deterministic name per identifier before counting distinct accessions.
        .with_columns(pl.col('term_name').min().over('library', 'term_id'))
        .unique()
        .collect()
    )


def enrichment(queries: pl.DataFrame, annotations: pl.DataFrame, population_size: int):
    sizes = queries.group_by(QUERY).agg(pl.len().alias('query_size'))
    background = annotations.group_by(TERM).agg(pl.len().alias('background_hits')).filter(
        pl.col('background_hits') >= MIN_BACKGROUND_HITS
    )
    hits = (
        queries.join(annotations.join(background, on=TERM, how='semi'), on='protein')
        .group_by([*QUERY, *TERM]).agg(
            pl.len().alias('query_hits'), pl.col('protein').sort().alias('proteins'),
        )
        .join(sizes, on=QUERY, validate='m:1')
        .join(background, on=TERM, validate='m:1')
        .with_columns(
            pl.lit(population_size, dtype=pl.UInt32).alias('population_size'),
            (pl.col('query_hits') / pl.col('query_size') /
             (pl.col('background_hits') / population_size)).alias('fold_enrichment'),
        )
    )
    pvalues = hypergeom.sf(
        hits['query_hits'].to_numpy().astype(np.int64) - 1, population_size,
        hits['background_hits'].to_numpy().astype(np.int64),
        hits['query_size'].to_numpy().astype(np.int64),
    )
    total_tests = sizes.height * background.height
    # Zero-hit tests all have p=1. Include them in the BH denominator without
    # materializing the cluster-by-term cross product.
    adjusted = (np.minimum(1, bh_adjust(pvalues) * (total_tests / len(pvalues)))
                if len(pvalues) else pvalues)
    return (
        hits.with_columns(pl.Series('pvalue', pvalues), pl.Series('qvalue', adjusted))
        .sort([*QUERY, 'qvalue', 'pvalue', 'library', 'term_id']),
        sizes, background, total_tests,
    )


def select_labels(evidence: pl.DataFrame, sizes: pl.DataFrame, ancestors) -> pl.DataFrame:
    candidates = (
        evidence.filter((pl.col('qvalue') <= MAX_QVALUE) & (pl.col('query_hits') >= MIN_LABEL_HITS))
        .with_columns(
            (-pl.col('qvalue').clip(lower_bound=np.finfo(float).tiny).log10()
             * (pl.col('population_size') / pl.col('background_hits')).log()).alias('score')
        )
        .sort(['score', 'qvalue', 'term_id'], descending=[True, False, False])
    )
    winners = {}
    for key, group in candidates.group_by(QUERY):
        rows = group.to_dicts()
        go_hits = {row['term_id']: set(row['proteins']) for row in rows if row['library'] != 'Reactome'}
        suppressed = set()
        for child, child_hits in go_hits.items():
            for parent in (ancestors(child) & go_hits.keys()) - {child}:
                parent_hits = go_hits[parent]
                if len(parent_hits & child_hits) / len(parent_hits | child_hits) > ANCESTOR_OVERLAP:
                    suppressed.add(parent)
        winners[key] = next(row for row in rows if row['term_id'] not in suppressed)

    size_by_query = {(c, s): n for c, s, n in sizes.iter_rows()}
    labels = []
    for community in sorted(sizes['community'].unique()):
        row = {'community': community}
        for subset, prefix in [('core', 'primary'), ('interface', 'interface'), ('full', 'full_context')]:
            size = size_by_query.get((community, subset), 0)
            term = winners.get((community, subset))
            if subset == 'full' and (community, 'core') in winners:
                term, status = None, 'core_labeled'
            else:
                status = 'labeled' if term else 'no_qualifying_term' if size else 'empty_subset'
            row[f'{subset}_size'] = size
            row[f'{prefix}_status'] = status
            for key, suffix in [('term_name', 'label'), ('term_id', 'term_id'), ('library', 'library')]:
                row[f'{prefix}_{suffix}'] = term[key] if term else None
        labels.append(row)
    return pl.from_dicts(labels, schema_overrides={
        f'{prefix}_{suffix}': pl.String
        for prefix in ('primary', 'interface', 'full_context')
        for suffix in ('label', 'term_id', 'library')
    })


def majority_labels(queries, annotations, terms):
    """Count unique proteins at each GO minimum root distance, excluding roots.

    GO is a DAG, so depths describe minimum distances, not exclusive levels.
    A protein can support multiple terms; missing annotations stay in the denominator.
    """
    @lru_cache(None)
    def depth(term):
        parents = [p for p in terms[term]['parents'] if p in terms]
        return 1 + min(map(depth, parents)) if parents else 0

    schema = {'community': pl.String, 'subset': pl.String, 'library': pl.String,
              'depth': pl.Int64, 'term_id': pl.String, 'term_name': pl.String,
              'query_hits': pl.UInt32, 'query_size': pl.UInt32,
              'support_fraction': pl.Float64, 'is_majority': pl.Boolean,
              'proteins': pl.List(pl.String)}
    go = annotations.filter(pl.col('library').str.starts_with('GO_')).unique(
        subset=['protein', 'library', 'term_id'])
    depths = pl.DataFrame({'term_id': list(terms), 'depth': [depth(t) for t in terms]},
                          schema={'term_id': pl.String, 'depth': pl.Int64})
    sizes = queries.group_by(QUERY).agg(pl.col('protein').n_unique().alias('query_size'))
    votes = (queries.join(go, on='protein').join(depths, on='term_id')
             .filter(pl.col('depth') > 0)
             .group_by([*QUERY, *TERM, 'depth'])
             .agg(pl.col('protein').n_unique().alias('query_hits'),
                  pl.col('protein').unique().sort().alias('proteins'))
             .join(sizes, on=QUERY)
             .with_columns((pl.col('query_hits') / pl.col('query_size')).alias('support_fraction'))
             .with_columns((pl.col('support_fraction') > 0.5).alias('is_majority')))
    if votes.is_empty():
        return pl.DataFrame(schema=schema)
    # Keep every supported process, including unequal counts and minority labels.
    return votes.select(list(schema)).sort(
        [*QUERY, 'library', 'depth', 'query_hits', 'term_id'],
        descending=[False, False, False, False, True, False])


def hierarchy_sankey(votes, evidence, ancestors, community, library='GO_BP', subset='full',
                     depths=(2, 3, 4), max_terms=8, min_hits=2):
    """Show real GO ancestor links weighted by distinct shared supporting proteins.

    This is an overlapping annotation DAG, not a conserved flow of unique proteins.
    Display filtering never recalculates enrichment or its correction family.
    """
    import html
    import textwrap
    import plotly.graph_objects as go
    from moonlighting import significance_label

    selected = votes.filter((pl.col('community') == str(community)) &
        (pl.col('subset') == subset) & (pl.col('library') == library) &
        pl.col('depth').is_in(depths) & (pl.col('query_hits') >= min_hits))
    rows = []
    for depth in sorted(set(depths)):
        rows.extend(selected.filter(pl.col('depth') == depth)
                    .sort(['query_hits', 'term_id'], descending=[True, False]).head(max_terms).to_dicts())
    adjusted = {row['term_id']: row for row in evidence.filter(
        (pl.col('community') == str(community)) & (pl.col('subset') == subset) &
        (pl.col('library') == library)).iter_rows(named=True)}
    links = []
    for i, parent in enumerate(rows):
        for j, child in enumerate(rows):
            if child['depth'] != parent['depth'] + 1 or parent['term_id'] not in ancestors(child['term_id']):
                continue
            support = sorted(set(parent['proteins']) & set(child['proteins']))
            if support:
                links.append({'source': i, 'target': j, 'value': len(support),
                              'source_term': parent['term_id'], 'target_term': child['term_id'],
                              'proteins': support})
    labels, hover = [], []
    for row in rows:
        test = adjusted.get(row['term_id'], {})
        q, fold = test.get('qvalue'), test.get('fold_enrichment')
        star = significance_label(q) if q is not None else 'n/a'
        name = '<br>'.join(html.escape(line) for line in textwrap.wrap(row['term_name'], 32))
        labels.append(f"<b>{name}</b><br><b>{row['query_hits']}/{row['query_size']} proteins "
                      f"({row['support_fraction']:.0%})</b> · {star}")
        hover.append(f"{row['term_id']} · depth {row['depth']}<br>"
                     f"q={q:.3g}, fold={fold:.3g}" if q is not None and fold is not None else
                     f"{row['term_id']} · depth {row['depth']}<br>Enrichment unavailable")
    figure = go.Figure(go.Sankey(arrangement='snap',
        textfont=dict(color='#111111', size=18, family='Arial, sans-serif', shadow='none'),
        node=dict(label=labels, customdata=hover, hovertemplate='%{customdata}<extra></extra>',
                  pad=55, thickness=18,
                  x=[0.03 + 0.74 * (row['depth']-min(depths))/max(1,max(depths)-min(depths)) for row in rows]),
        link=dict(source=[link['source'] for link in links], target=[link['target'] for link in links],
                  value=[link['value'] for link in links], color='rgba(80, 90, 110, 0.14)',
                  hovertemplate='%{value} shared supporting proteins<extra></extra>')))
    figure.update_layout(template='plotly_white',
        title=f'{library.replace("GO_", "GO ")} · cluster {community} · {subset}',
        font=dict(size=14, color='#111111'),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        hoverlabel=dict(bgcolor='#ffffff', bordercolor='#333333', font=dict(color='#111111', size=14)),
        margin=dict(l=35, r=300, t=135, b=85),
        height=max(700, 220 + 145 * max((sum(r['depth'] == d for r in rows) for d in depths), default=0)))
    for depth in depths:
        figure.add_annotation(x=0.03 + 0.74*(depth-min(depths))/max(1,max(depths)-min(depths)),
            y=1.06, xref='paper', yref='paper', text=f'Depth {depth}',
            font=dict(color='#111111', size=16), showarrow=False)
    return figure, selected.drop('proteins'), pl.from_dicts(links) if links else pl.DataFrame()


def label_summary(labels, majority, evidence):
    """One row per cluster with independent descriptive and enrichment labels."""
    rows = []
    for row in labels.iter_rows(named=True):
        community = row['community']
        summary = {'community': community, 'proteins': row['full_size']}
        for library in LIBRARIES:
            hits = evidence.filter((pl.col('community') == community) &
                (pl.col('subset') == 'full') & (pl.col('library') == library) &
                (pl.col('qvalue') < MAX_QVALUE) & (pl.col('query_hits') >= MIN_LABEL_HITS))
            best = hits.sort('qvalue', 'pvalue', 'term_id').head(1).to_dicts()
            summary[f'{library}_enrichment'] = best[0]['term_name'] if best else None
            summary[f'{library}_qvalue'] = best[0]['qvalue'] if best else None
            summary[f'{library}_fold'] = best[0]['fold_enrichment'] if best else None
            if library.startswith('GO_'):
                processes = majority.filter((pl.col('community') == community) &
                    (pl.col('subset') == 'full') & (pl.col('library') == library) &
                    (pl.col('query_hits') >= MIN_LABEL_HITS))
                summary[f'{library}_hierarchy_labels'] = '; '.join(
                    f"depth {term['depth']}: {term['term_name']} [{term['term_id']}] "
                    f"({term['query_hits']}/{term['query_size']} proteins, "
                    f"{term['support_fraction']:.1%})"
                    for term in processes.iter_rows(named=True)) or None
                supported = majority.filter((pl.col('community') == community) &
                    (pl.col('subset') == 'full') & (pl.col('library') == library) & pl.col('is_majority'))
                if supported.height:
                    supported = supported.filter(pl.col('depth') == pl.col('depth').max())
                summary[f'{library}_majority'] = '; '.join(supported['term_name']) or None
        rows.append(summary)
    return pl.from_dicts(rows, infer_schema_length=None)


def network_population(path: Path) -> pl.DataFrame:
    edges = load_network(path).edges
    return (
        pl.concat([edges.select(protein_id(col).alias('protein')) for col in ('a', 'b')])
        .filter(pl.col('protein') != '').unique().sort('protein')
    )


def file_input(memberships: Path, network: str):
    path = Path(network)
    if not path.is_file():
        path = discover_datasets().get(network)
        if path is None:
            raise ValueError(f'Unknown network {network!r}; use a discovered name or canonical Parquet path')
    if path.suffix.lower() != '.parquet':
        raise ValueError('The background must be a canonical or saved-version Parquet file')
    network_source = _fingerprint(path.resolve())
    source = {'network': network_source, 'memberships': _fingerprint(memberships.resolve())}
    if path.with_suffix('.json').is_file():
        source['network_audit'] = _fingerprint(path.with_suffix('.json').resolve())
    result_path = memberships.with_name('result.json')
    if memberships.name == 'memberships.parquet' and result_path.is_file():
        job = json.loads(result_path.read_text(encoding='utf-8'))
        expected = job.get('job', {}).get('provenance', {}).get('input', {}).get('sha256')
        if job.get('status') != 'success' or (expected and expected != network_source['sha256']):
            raise ValueError('The clustering run did not succeed on the supplied network version')
        source['clustering_result'] = _fingerprint(result_path.resolve())
    population = network_population(path)
    if memberships.suffix.lower() == '.csv':
        frame = pl.read_csv(memberships, columns=['protein', 'community'], infer_schema=False)
    elif memberships.suffix.lower() == '.parquet':
        frame = pl.read_parquet(memberships, columns=['protein', 'community'])
    else:
        raise ValueError('Memberships must be CSV or Parquet')
    return frame, population, source


def cyrest(base_url: str, method: str, path: str, body=None):
    headers = {'Accept': 'application/json'} if path == AUTOANNOTATE_COMMAND else None
    response = requests.request(
        method, f'{base_url.rstrip("/")}/{path}', json=body,
        headers=headers, timeout=(5, 120),
    )
    response.raise_for_status()
    if 'json' in response.headers.get('Content-Type', ''):
        return response.json()
    return response.text


def annotation_node_table(network: int, base_url: str) -> list[dict]:
    # Read the raw rows because the general bridge flattens list-valued cells.
    rows = cyrest(base_url, 'GET', f'networks/{network}/tables/defaultnode/rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Cytoscape network {network} returned no node rows')
    return rows


def cytoscape_proteins(rows: list[dict], column: str) -> pl.DataFrame:
    if not any(column in row for row in rows):
        raise ValueError(f'Cytoscape UniProt column {column!r} was not found')
    return pl.DataFrame({
        'SUID': [row['SUID'] for row in rows],
        'protein': [row.get(column) for row in rows],
    }, schema={'SUID': pl.Int64, 'protein': pl.String}).with_columns(protein_id())


def snapshot_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def annotation_input(rows, network, uniprot_column, cluster_column, separator, base_url):
    if not any(cluster_column in row for row in rows):
        raise ValueError(f'Cytoscape cluster column {cluster_column!r} was not found')
    nodes = cytoscape_proteins(rows, uniprot_column)
    population = nodes.filter(pl.col('protein').str.contains(UNIPROT_ACCESSION)).select('protein').unique().sort('protein')
    clusters = []
    for row in rows:
        value = row.get(cluster_column)
        if isinstance(value, list):
            clusters.append([str(item) for item in value if item is not None])
        elif value is None:
            clusters.append([])
        else:
            clusters.append(str(value).split(separator) if separator else [str(value)])
    pairs = (
        nodes.with_columns(pl.Series('community', clusters, dtype=pl.List(pl.String)))
        .explode('community').select('protein', 'community')
    )
    source = {'cytoscape': {
        'network_suid': network, 'base_url': base_url,
        'uniprot_column': uniprot_column, 'cluster_column': cluster_column,
        'cluster_separator': separator, 'node_rows': nodes.height,
        'rows_without_valid_accession': nodes.filter(
            ~pl.col('protein').str.contains(UNIPROT_ACCESSION).fill_null(False)
        ).height,
        'background_sha256': snapshot_sha256(population['protein'].to_list()),
        'memberships_sha256': snapshot_sha256(pairs.unique().sort('protein', 'community').rows()),
    }}
    return pairs, population, source


def autoannotate_labels(base_url, network, cluster_column, text_column, suid_to_protein, expected):
    reply = cyrest(base_url, 'POST', AUTOANNOTATE_COMMAND, {
        'network': f'SUID:{network}', 'useClusterMaker': 'false', 'clusterIdColumn': cluster_column,
        'labelColumn': text_column, 'returnJsonOnly': 'true', 'createSingletonClusters': 'false',
    })
    if not isinstance(reply, dict) or reply.get('errors') or not isinstance(reply.get('data'), list):
        raise ValueError(f'AutoAnnotate did not return clusters: {reply!r}')
    matched = {}
    mismatches = 0
    for item in reply['data']:
        members = frozenset(suid_to_protein.get(int(node)) for node in item['nodes'])
        if members not in expected:
            mismatches += 1
            continue
        matched.setdefault(members, set()).add(str(item.get('label') or '').strip())
    return matched, mismatches


def has_text(value) -> bool:
    if isinstance(value, list):
        return any(has_text(item) for item in value)
    return isinstance(value, str) and bool(value.strip())


def annotation_comparison(full, population, term_text, network, uniprot_column, description_column, base_url):
    grouped = full.group_by('community').agg(pl.col('protein').sort()).sort('community')
    expected = {community: frozenset(proteins) for community, proteins in grouped.iter_rows()}
    expected_sets = set(expected.values())
    passes = ('description', 'terms')
    compared = {community: {'community': community} for community in expected}
    for row in compared.values():
        for name in passes:
            row[f'autoannotate_{name}_label'] = None
            row[f'autoannotate_{name}_status'] = 'unavailable'
            row[f'autoannotate_{name}_text_proteins'] = None
    summary = {
        'status': 'unavailable', 'network_suid': network, 'base_url': base_url,
        'uniprot_column': uniprot_column, 'description_column': description_column,
        'command': AUTOANNOTATE_COMMAND, 'passes': {},
    }
    table = f'networks/{network}/tables/defaultnode'
    prefix = f'atlas_annotation_{uuid4().hex}'
    cluster_column, term_column = f'{prefix}_clusters', f'{prefix}_terms'
    created = []
    try:
        rows = annotation_node_table(network, base_url)
        nodes = cytoscape_proteins(rows, uniprot_column).join(population, on='protein', how='semi')
        missing = full.select('protein').unique().join(nodes, on='protein', how='anti')
        if missing.height:
            examples = ', '.join(missing['protein'].sort().head(5))
            raise ValueError(f'Cytoscape network is missing {missing.height} assigned accessions: {examples}')
        cyrest(base_url, 'GET', 'commands/autoannotate')
        suid_to_protein = dict(nodes.iter_rows())
        descriptions = {row['SUID']: row.get(description_column) for row in rows}
        summary['input_sha256'] = snapshot_sha256(sorted(
            (suid, protein, descriptions[suid]) for suid, protein in suid_to_protein.items()
        ))
        payload = (
            nodes.join(full.group_by('protein').agg(pl.col('community').sort().alias(cluster_column)),
                       on='protein', how='inner', validate='m:1')
            .join(term_text.rename({'term_text': term_column}), on='protein', how='left', validate='m:1')
            .drop('protein').sort('SUID')
        )
        for column, is_list in [(cluster_column, True), (term_column, False)]:
            cyrest(base_url, 'POST', f'{table}/columns', {'name': column, 'type': 'String', 'list': is_list})
            created.append(column)
        cyrest(base_url, 'PUT', table, {'key': 'SUID', 'dataKey': 'SUID', 'data': payload.to_dicts()})
        described = {protein for suid, protein in suid_to_protein.items() if has_text(descriptions[suid])}
        for name, column, text_proteins in [
            ('description', description_column, described),
            ('terms', term_column, set(term_text['protein'])),
        ]:
            for community, members in expected.items():
                count = len(members & text_proteins)
                compared[community][f'autoannotate_{name}_text_proteins'] = count
                if not count:
                    compared[community][f'autoannotate_{name}_status'] = 'missing_text'
            if not column or not text_proteins.intersection(full['protein']):
                summary['passes'][name] = {'status': 'missing_text'}
                continue
            try:
                matched, mismatches = autoannotate_labels(
                    base_url, network, cluster_column, column, suid_to_protein, expected_sets,
                )
                summary['passes'][name] = {'status': 'returned', 'unexpected_member_sets': mismatches}
                for community, members in expected.items():
                    row = compared[community]
                    if row[f'autoannotate_{name}_status'] == 'missing_text':
                        continue
                    labels = matched.get(members, set())
                    if len(labels) == 1:
                        label = next(iter(labels))
                        row[f'autoannotate_{name}_label'] = label or None
                        row[f'autoannotate_{name}_status'] = 'labeled' if label else 'empty_label'
                    else:
                        row[f'autoannotate_{name}_status'] = 'membership_mismatch'
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                summary['passes'][name] = {'status': 'unavailable', 'error': str(exc)}
        statuses = [row[f'autoannotate_{name}_status'] for row in compared.values() for name in passes]
        unexpected = any(result.get('unexpected_member_sets') for result in summary['passes'].values())
        summary['status'] = ('complete' if all(s == 'labeled' for s in statuses) and not unexpected
                             else 'partial' if 'labeled' in statuses else 'unavailable')
    except (requests.RequestException, KeyError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        summary['error'] = str(exc)
    finally:
        for column in reversed(created):
            try:
                cyrest(base_url, 'DELETE', f'{table}/columns/{column}')
            except (requests.RequestException, ValueError) as exc:
                summary.setdefault('cleanup_errors', []).append(str(exc))
    return pl.from_dicts(list(compared.values()), schema_overrides={
        **{f'autoannotate_{name}_label': pl.String for name in passes},
        **{f'autoannotate_{name}_text_proteins': pl.Int64 for name in passes},
    }), summary


def annotate_memberships(frame, population, source, *, gaf=DEFAULT_GAF, obo=DEFAULT_OBO,
                         reactome=DEFAULT_REACTOME, cytoscape=None, go_cache=None,
                         uniprot_functions=None, communities=None, generif=None,
                         gene_mapping=DEFAULT_GENE_MAPPING, abstracts=DEFAULT_ABSTRACTS):
    """Compute one annotation result for the CLI or the Streamlit app."""
    queries = membership_queries(frame, population)
    reference_queries = queries
    if communities is not None:
        requested = sorted(set(map(str, communities)))
        queries = queries.filter(pl.col('community').is_in(requested))
        if queries.is_empty():
            raise ValueError('No memberships found for the selected clusters')
    full = queries.filter(pl.col('subset') == 'full').select('community', 'protein')
    assigned = full.select('protein').unique()

    with TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        go_path = prepare_go_annotations(Path(go_cache) if go_cache else scratch, gaf, obo)
        go_provenance = json.loads(go_path.with_suffix('.json').read_text(encoding='utf-8'))
        annotations = load_annotations(go_path, reactome, population)
        evidence, sizes, background, tests = enrichment(queries, annotations, population.height)
        terms, _, _, ancestors = _ontology(obo)
        labels = select_labels(evidence, sizes, ancestors)
        majority = majority_labels(queries, annotations, terms)
        summary = label_summary(labels, majority, evidence)
        text_labels, text_manifest = function_text_labels(queries, population,
            uniprot_functions or (DEFAULT_FUNCTIONS if generif and DEFAULT_FUNCTIONS.is_file() else None),
            generif=generif if generif and Path(generif).is_file() else None, gene_mapping=gene_mapping,
            abstracts=abstracts if abstracts and Path(abstracts).is_file() else None,
            reference_queries=reference_queries)
        for method in ('tfidf', 'document_lift', 'ctfidf', 'log_odds', 'yake'):
            column = f'UniProt_FUNCTION_{method}'
            ranked = (text_labels.filter((pl.col('subset') == 'full') &
                      (pl.col('method') == method)).group_by('community', maintain_order=True)
                      .agg(pl.col('label').str.join('; ').alias(column)))
            summary = summary.join(ranked, on='community', how='left', validate='1:1')
        comparison = {'status': 'not_requested'}
        if cytoscape:
            terms_by_protein = (
                annotations.group_by('protein')
                .agg(pl.col('term_name').unique().sort().str.join(' | ').alias('term_text'))
            )
            compared, comparison = annotation_comparison(
                full, population, terms_by_protein, **cytoscape,
            )
            labels = labels.join(compared, on='community', validate='1:1', maintain_order='left')
        coverage = {}
        for library in LIBRARIES:
            proteins = annotations.filter(pl.col('library') == library).select('protein').unique()
            coverage[library] = {
                'annotated_background': proteins.height,
                'annotated_assigned': proteins.join(assigned, on='protein', how='semi').height,
                'eligible_terms': background.filter(pl.col('library') == library).height,
            }
        manifest = {
            'schema_version': 5, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'uniprot_function_labels': text_manifest,
            'sources': {**source, 'go': go_provenance, 'reactome': _fingerprint(reactome.resolve())},
            'parameters': {
                'selected_communities': requested if communities is not None else None,
                'libraries': LIBRARIES, 'min_background_hits': MIN_BACKGROUND_HITS,
                'min_label_hits': MIN_LABEL_HITS, 'max_qvalue': MAX_QVALUE,
                'ancestor_hit_jaccard_strictly_above': ANCESTOR_OVERLAP,
                'test': 'one-sided hypergeometric; one BH family across nonempty subsets and libraries',
                'reactome_policy': 'human R-HSA pathways in the supplied lowest-level mapping',
                'majority_policy': 'GO is_a/part_of propagation; minimum root distance; roots excluded; '
                    'unique protein votes / all subset proteins; strict >50%; all terms with hits per depth; '
                    'summary includes every term with >=2 hits at every depth and deepest majority; '
                    'protein supports may overlap; descriptive, no statistical test',
            },
            'coverage': {
                'background': population.height, 'assigned': assigned.height,
                'unassigned': population.height - assigned.height, 'libraries': coverage,
            },
            'clusters': labels.height, 'tested_subsets': sizes.height,
            'memberships_sha256': snapshot_sha256(full.sort('protein', 'community').rows()),
            'tests': tests, 'zero_hit_tests_omitted_from_table': tests - evidence.height,
            'cytoscape_comparison': comparison,
        }
    return {'labels': labels, 'evidence': evidence, 'manifest': manifest, 'memberships': full,
            'majority': majority, 'summary': summary, 'text_labels': text_labels}


def write_annotation_bundle(output: Path, result: dict) -> None:
    """Publish atomically, reusing an already published annotation cache."""
    if (output / 'manifest.json').is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        result['labels'].write_csv(bundle / 'cluster_labels.csv')
        result['majority'].drop('proteins').write_csv(bundle / 'go_hierarchy_labels.csv')
        _write_parquet(bundle / 'go_hierarchy_support.parquet', result['majority'])
        result['summary'].write_csv(bundle / 'label_summary.csv')
        _write_parquet(bundle / 'uniprot_function_labels.parquet', result['text_labels'])
        _write_parquet(bundle / 'enrichment.parquet', result['evidence'])
        _write_json(bundle / 'manifest.json', result['manifest'])
        try:
            bundle.rename(output)
        except OSError:
            # Another session can finish the same cache key while we compute.
            # Only accept a published bundle; do not hide unrelated I/O errors.
            if not (output / 'manifest.json').is_file():
                raise


def run(args):
    base_url = args.cyrest_url
    if not base_url and (args.cytoscape is not None or args.compare_cytoscape is not None):
        from cytoscape_bridge import DEFAULT_BASE_URL
        base_url = DEFAULT_BASE_URL
    if args.output.exists():
        raise ValueError(f'Output already exists: {args.output}')
    if args.memberships:
        frame, population, source = file_input(args.memberships, args.network)
    else:
        nodes = annotation_node_table(args.cytoscape, base_url)
        frame, population, source = annotation_input(
            nodes, args.cytoscape, args.uniprot_column, args.cluster_column,
            args.cluster_separator, base_url,
        )
    cytoscape = None
    if args.compare_cytoscape is not None:
        cytoscape = {
            'network': args.compare_cytoscape, 'uniprot_column': args.uniprot_column,
            'description_column': args.description_column, 'base_url': base_url,
        }
    if args.fetch_uniprot:
        fetch_uniprot_functions(args.uniprot_functions or DEFAULT_FUNCTIONS)
    functions = args.uniprot_functions or (DEFAULT_FUNCTIONS if args.fetch_uniprot else None)
    result = annotate_memberships(
        frame, population, source, gaf=args.gaf, obo=args.obo, reactome=args.reactome, cytoscape=cytoscape,
        uniprot_functions=functions, generif=args.generif, gene_mapping=args.gene_mapping, abstracts=args.abstracts,
    )
    write_annotation_bundle(args.output, result)
    return result['labels'].height, result['manifest']['cytoscape_comparison']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--memberships', type=Path, help='Long-form CSV or Parquet memberships')
    inputs.add_argument('--cytoscape', type=int, metavar='SUID', help='Read clusters from Cytoscape Desktop')
    parser.add_argument('--network', help='Discovered network name or canonical Parquet path for file input')
    parser.add_argument('--uniprot-column', help='Node column containing UniProt accessions')
    parser.add_argument('--cluster-column', help='Node column containing existing cluster assignments')
    parser.add_argument('--cluster-separator', help='Separator for multiple IDs in a scalar cluster column')
    parser.add_argument('--compare-cytoscape', type=int, metavar='SUID', help='Compare with AutoAnnotate on this network')
    parser.add_argument('--description-column', help='Cytoscape text column for the protein-description pass')
    parser.add_argument('--cyrest-url', help='Override the local Cytoscape CyREST URL')
    parser.add_argument('--gaf', type=Path, default=DEFAULT_GAF)
    parser.add_argument('--obo', type=Path, default=DEFAULT_OBO)
    parser.add_argument('--reactome', type=Path, default=DEFAULT_REACTOME)
    parser.add_argument('--uniprot-functions', type=Path,
                        help='Local TSV with Entry/Function [CC] or protein/function columns')
    parser.add_argument('--generif', type=Path, help='GeneRIF basic TSV/gzip (human entries only)')
    parser.add_argument('--gene-mapping', type=Path, default=DEFAULT_GENE_MAPPING)
    parser.add_argument('--abstracts', type=Path, default=DEFAULT_ABSTRACTS, help='Optional local pmid/abstract TSV')
    parser.add_argument('--fetch-uniprot', action='store_true',
                        help='Download reviewed human Swiss-Prot FUNCTION comments when the snapshot does not exist')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the result bundle')
    args = parser.parse_args(argv)
    if args.memberships and not args.network:
        parser.error('--memberships requires --network')
    if args.cytoscape is not None and (args.network or not args.cluster_column):
        parser.error('--cytoscape uses its own network background and requires --cluster-column')
    if (args.cytoscape is not None or args.compare_cytoscape is not None) and not args.uniprot_column:
        parser.error('Cytoscape input or comparison requires --uniprot-column')
    if args.cluster_separator == '':
        parser.error('--cluster-separator must be nonempty')
    try:
        count, comparison = run(args)
    except Exception as exc:
        parser.exit(1, f'interactome-annotate: {exc}\n')
    print(f'Annotated {count} clusters. Results saved to {args.output}')
    if comparison['status'] != 'not_requested':
        print(f'Cytoscape comparison: {comparison["status"]}. See manifest.json for details.')


if __name__ == '__main__':
    main()
