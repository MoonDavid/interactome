# Cluster annotation — notes

## Protein sets

- **Full:** all cluster proteins.
- **Core:** proteins exclusive to the cluster.
- **Interface:** proteins shared with other clusters.
- Enrichment background: all proteins in the selected network.

## GO / Reactome

### Majority rule

- Count distinct proteins supporting each term.
- Support = annotated cluster proteins / all cluster proteins.
- Majority: **support > 50%**; missing annotations remain in the denominator.
- GO: propagate `is_a` / `part_of`; exclude roots; report deepest majority terms.
- Reactome: report pathways supported by more than half of the protein set.
- Descriptive coverage; no statistical test.

### Enrichment

- One-sided hypergeometric test: is the term overrepresented in the cluster?
- N = network size; M = network term hits; n = cluster size; x = cluster term hits.

$$
p=P(X\ge x),\qquad X\sim\mathrm{Hypergeometric}(N,M,n)
$$

$$
\mathrm{fold\ enrichment}=\frac{x/n}{M/N}
$$

- Eligible terms: at least 5 background proteins.
- BH correction across all tested clusters, nonempty subsets and GO/Reactome terms; include zero-hit tests.
- Compact labels: at least 2 supporting proteins, q ≤ 0.05; reduce redundant GO ancestors.
- Stars use adjusted q: `*` < 0.05; `**` < 0.01; `***` < 0.001; `****` < 0.0001. Valid nonsignificant: `ns`; unavailable: `n/a`.
- Majority = broad cluster coverage. Enrichment = excess relative to background. Neither implies the other.

## UniProt FUNCTION / GeneRIF

### Input

- Protein document = UniProt FUNCTION text + human GeneRIF statements + optional linked abstracts.
- GeneRIF: GeneID → UniProt mapping; taxon 9606 only; deduplicate protein/text pairs.
- GeneRIF statements are **not abstracts**. Abstracts require a local `pmid` / `abstract` TSV.
- Cluster pseudo-document = concatenated protein documents.
- Count-based preprocessing: remove evidence references; lowercase; remove short tokens and stopwords; extract adjacent n-grams.
- FUNCTION only: 1–2 grams. With GeneRIF: 1–4 grams.
- Retain terms supported by at least 2 proteins; top 10 positive scores per method/subset.

### TF-IDF

- Score each protein document, then average normalized vectors over all cluster proteins.
- f(t,p) = term count in protein p; D = annotated network documents; d(t) = documents containing t.

$$
w_{t,p}=(1+\log f_{t,p})\left[1+\log\frac{1+D}{1+d_t}\right]
$$

$$
\mathrm{score}(t,C)=\frac{1}{|C|}\sum_{p\in C}\frac{w_{t,p}}{\|w_p\|_2}
$$

- Absent terms contribute zero. Missing protein documents remain in the cluster denominator.
- High score: term shared across cluster proteins and less common in network documents.

### c-TF-IDF

- Score the cluster as one document.

$$
\mathrm{score}(t,C_k)=\frac{f_{t,k}}{N_k}\log\left(1+\frac{A}{f_t}\right)
$$

- f(t,k): cluster term count; N(k): retained cluster word count.
- A: mean full-cluster word count; f(t): term count across network protein documents.
- Full partition defines A, including for selected-cluster annotation.
- High score: frequent within cluster, uncommon across network text.

### Document lift

- Compare term prevalence among annotated cluster proteins and annotated network-complement proteins.
- h/a = cluster proteins containing term / annotated cluster proteins.
- b/B = complement proteins containing term / annotated complement proteins.

$$
\mathrm{score}=\log_2\frac{(h+0.5)/(a+1)}{(b+0.5)/(B+1)}
$$

- Positive: higher prevalence inside cluster.

### Dirichlet-smoothed log-odds

- Compare cluster and network-complement term counts.
- Informative prior: entire supplied human UniProt FUNCTION corpus; strength α₀ = 1,000.
- Prior term weights use additive 0.5 smoothing over the combined n-gram vocabulary.

$$
\delta_t=\log\frac{a_t+\alpha_t}{n_C-a_t+\alpha_0-\alpha_t}
-\log\frac{b_t+\alpha_t}{n_B-b_t+\alpha_0-\alpha_t}
$$

- a, b: term counts inside/outside cluster; n(C), n(B): total feature counts.
- Positive: term favors cluster. Smoothing stabilizes sparse counts, useful for small clusters.
- Requires UniProt prior and nonempty annotated complement. No z-score or significance test.

### YAKE

- Extract multiword phrases from the cluster document.
- Signals: frequency, position, capitalization, neighboring words, sentence spread.
- Keep 2–4 word phrases passing the protein-support filters.
- Native score: lower is better. Displayed score = 1 / (1 + native score).
- Within-document ranking; no proteome comparison, embeddings or cosine similarity.
- Optional dependency: `pip install -e ".[text-annotation]"`.


