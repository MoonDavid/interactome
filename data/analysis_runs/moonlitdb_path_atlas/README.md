# MoonLitDB positions and average shortest paths

Saved 16 November 2026. Open [the static network image](moonlitdb_topology.png).
Blue nodes are MoonLitDB proteins; all other nodes are grey. Every node has the same size.
The left column shows the entire largest connected component. The right column keeps only edges with at least one MoonLitDB endpoint, using the same full-network positions.

## What the results actually show

| Network | MoonLitDB nodes | Observed mean hops | Expected mean hops | Difference | BH q | Significance |
|---|---:|---:|---:|---:|---:|---|
| HuRI | 200 | 4.070 | 4.007 | +0.063 | 0.237762 | ns |
| BioPlex | 394 | 3.727 | 3.671 | +0.057 | 0.005994 | ** |
| huMAP2, threshold 0.05 | 305 | 5.311 | 5.821 | -0.510 | 0.005994 | ** |
| huMAP2, threshold 0.95 | 78 | 6.725 | 7.194 | -0.470 | 0.237762 | ns |
| TAGC binary2025 | 429 | 3.233 | 3.285 | -0.053 | 0.011988 | * |

The direction is higher in HuRI and BioPlex and lower in huMAP2 and TAGC, but the HuRI increase and huMAP2 0.95 decrease are not statistically significant. BioPlex and TAGC effects are small in absolute hops. A shorter-than-expected path in huMAP2 does not mean a shorter absolute path than in BioPlex: each expectation belongs to its own network.

## Topological interpretation

Average shortest path depends on where *all pairs* of MoonLitDB proteins sit, including routes through grey proteins. Many local blue-blue connections can coexist with long distances between blue proteins occupying different regions. We interpret this placement using both a disjoint Louvain partition and the saved overlapping `c.ocg` clusterings (`OCG` allows proteins to belong to multiple overlapping communities simultaneously).

- **HuRI:** only 13 direct blue-blue pairs among 200 proteins. The MoonLitDB-only graph has 187 components and its largest component contains five proteins. The blue set is locally sparse. In Louvain, 15.82% of blue pairs occupy the same community, versus 16.81% for the matched random sets. In the saved `c.ocg` clustering (`baf37da45ad36e01`, 481 active clusters in the LCC), 22.00% of MoonLitDB proteins belong to ≥ 2 overlapping OCG clusters (versus 17.87% in the degree-matched null; mean $k = 1.64$ versus $1.42$), yet only 1.75% of MoonLitDB pairs share at least one OCG cluster versus 2.89% in the matched null (averaging 3.049 hops when sharing an OCG cluster versus 4.088 hops when disjoint). Individual MoonLitDB proteins sit in multiple local OCG modules, while different MoonLitDB pairs occupy separated modules across the network; the small positive shortest-path effect remains compatible with the null.
- **BioPlex:** 250 direct blue-blue pairs coexist with 241 MoonLitDB-only components; the largest contains 83 of 394 proteins. There are connected local groups, but most blue pairs span different communities: in Louvain, 9.66% share a community versus 13.74% in the matched null (observed between-community pairs average 3.767 hops, versus 3.360 within communities). Under `c.ocg` (`bdc40c5d6b02d84b`, 393 LCC clusters), 41.88% of MoonLitDB proteins belong to ≥ 2 clusters (versus 37.84% null; mean $k = 2.51$ versus $2.27$), whereas only 3.00% of blue pairs share at least one OCG cluster versus 3.60% in the matched null (mean 2.822 hops within shared OCG clusters versus 3.755 hops between disjoint OCG clusters). Both disjoint and overlapping community views support local multi-complex participation combined with broad dispersion across modules. Extra local edges alone do not ensure a shorter average over all pairs.
- **huMAP2 0.05:** 404 direct blue-blue pairs and a largest MoonLitDB-only component of 114 of 305 proteins show substantial local connection. Same-community blue pairs are more frequent than in the matched null in both Louvain (8.65% versus 4.79%; within-community 2.979 hops versus 5.532 between) and `c.ocg` (`072ec593edf82537`, 1,841 LCC clusters: 2.92% share ≥ 1 OCG cluster versus 2.02% null; within-OCG 2.262 hops versus 5.402 between), even though individual `c.ocg` multi-membership is lower than matched degree controls (46.89% with $k \ge 2$ versus 55.63% null; mean $k = 3.34$ versus $3.68$). Concentration in connected regions is consistent with the shorter-than-expected mean; the pair decomposition below also shows that the weighted between-community part dominates.
- **huMAP2 0.95:** the sparse, branched full-component picture makes separated groups and their connecting routes visible. Within-community blue pairs average 2.802 hops in Louvain and 1.464 hops in `c.ocg` (`a9964e54dea668eb`, 192 LCC clusters), compared with 7.370 and 7.145 hops between disjoint communities. Blue pairs share communities more often than random sets in both Louvain (14.12% versus 11.76%) and `c.ocg` (7.39% versus 6.53%), alongside elevated multi-cluster participation (61.54% with $k \ge 2$ versus 55.36% null; mean $k = 3.56$ versus $3.45$). However, only 78 MoonLitDB proteins remain in this largest component and the shortest-path reduction is not significant.
- **TAGC:** the largest MoonLitDB-only component contains 225 of 429 proteins, with 382 direct pairs and 11,141 pairs at two hops. Same-community blue pairs are slightly more frequent than matched sets in Louvain (16.11% versus 14.49%; 3.080 within versus 3.262 between). In `c.ocg` (`81c464cbc2a9d356`, 862 LCC clusters), 39.39% of MoonLitDB proteins belong to ≥ 2 overlapping clusters (versus 34.86% null; mean $k = 3.52$ versus $2.95$), while 2.21% of pairs share an OCG cluster versus 2.45% in the matched null (2.312 hops within shared OCG clusters versus 3.253 hops between). Thus MoonLitDB proteins act individually as multi-cluster bridges across compact TAGC modules, while shorter paths between distinct modules accompany the small significant overall path-length reduction.

These community comparisons (both Louvain and `c.ocg`) are descriptive, with no new significance tests in this atlas; formal degree-matched `c.ocg` permutation testing is available in the `OCG Exploration` page. They refer to one Louvain partition and one saved `c.ocg` clustering per network and can change with parameters. The evidence supports a topological interpretation, not a unique causal explanation or biological validation of moonlighting.

## Exact accounting of the mean difference

For each protein set, split the total sum of pair distances into within-community and between-community pairs, dividing both sums by the total number of pairs. Subtract the corresponding means over the 1,000 matched random sets. The two resulting contributions sum to observed minus expected:

### Disjoint Louvain partition

| Network | Within-community contribution (hops) | Between-community contribution (hops) | Total difference |
|---|---:|---:|---:|
| HuRI | -0.0235 | +0.0862 | +0.0626 |
| BioPlex | -0.1388 | +0.1954 | +0.0566 |
| huMAP2 0.05 | +0.0942 | -0.6043 | -0.5101 |
| huMAP2 0.95 | +0.0907 | -0.5602 | -0.4695 |
| TAGC | +0.0605 | -0.1130 | -0.0525 |

### Overlapping `c.ocg` clustering (pairs sharing ≥ 1 OCG cluster vs. 0 shared OCG clusters)

| Network | Within-OCG contribution (hops) | Between-OCG contribution (hops) | Total difference |
|---|---:|---:|---:|
| HuRI | -0.0370 | +0.0996 | +0.0626 |
| BioPlex | -0.0203 | +0.0769 | +0.0566 |
| huMAP2 0.05 | +0.0149 | -0.5250 | -0.5101 |
| huMAP2 0.95 | +0.0088 | -0.4783 | -0.4695 |
| TAGC | -0.0085 | -0.0440 | -0.0525 |

In both the disjoint Louvain partition and the overlapping `c.ocg` decomposition, the weighted between-community contribution determines the sign and dominates the magnitude of the difference in all five cases. This accounting combines changes in the fraction of pairs in each category and their distances; it does **not** prove that a change in between-community distance alone causes the effect. Likewise, more or fewer pairs sharing communities is supporting structural context, not a complete explanation on its own.

## Reading the image correctly

The full-network panels provide context; the filtered panels make blue proteins and their immediate connections easier to see. A filtered panel can appear disconnected even when its blue proteins are connected through grey-grey edges in the full graph. All reported paths use the entire largest component, including those hidden edges. Two-dimensional layout separation is not a calibrated hop-distance scale, and layouts from different networks do not share coordinates. Dense overlaps in the full panels limit visual interpretation; use the measured evidence alongside the image.

## Methods and provenance

Inputs are the exact canonical Parquet networks named in the saved shortest-path metadata. Their SHA-256 hashes were checked against both those saved shortest-path runs and the saved `c.ocg` clustering runs (`data/analysis_runs/clustering_raw/<network>/c.ocg/<run>/result.json`). MoonLitDB membership uses the local AllMoon CSV and saved UniProt mapping, without reviewed-only or publication-count filtering. Node labels containing multiple accessions count as MoonLitDB when any member accession belongs to the protein set.

The comparison population is each network's largest connected component. Distances are exact unweighted shortest paths between unordered distinct MoonLitDB pairs using all component nodes as intermediates. All pairs are reachable in this scope. Independently recalculated igraph means agree with the saved observed means. For `c.ocg`, a pair of LCC proteins is counted as within-OCG when they share at least one saved OCG cluster (`|C(u) ∩ C(v)| ≥ 1`), and per-protein multiplicity `k` counts distinct OCG clusters containing that protein in the LCC.

Expected values reproduce 1,000 seeded random sets, equal in size and in counts of floor(log2 degree) bins. These saved runs sample from the full comparison component, **including MoonLitDB proteins**. This differs from the complement-only null used by some other application functions; the explanation here follows the saved runs, not that alternative. Random sampling is without replacement within a trial; different trials can overlap. Seed: 42.

Significance uses the saved two-sided doubled smaller empirical tail with +1 corrections and BH adjustment across the full originally requested shortest-path network/metric correction family, including the originally requested huMAP3 result even though it is not pictured here. The existing significance_label helper supplies labels: * q<0.05, ** q<0.01, *** q<0.001, **** q<0.0001; ns means a valid non-significant test. No significance is assigned to layouts or descriptive community summaries.

## Saved files and reproduction

- `moonlitdb_topology.png`: visually inspected static image.
- `topology_evidence.csv`: precise network, pair-distance, Louvain, `c.ocg`, and difference-decomposition values.
- `*_hops.csv`: exact MoonLitDB pair counts by shortest-path distance.
- `*_positions.parquet`: protein identities, blue/grey membership, saved coordinates, Louvain community IDs, and OCG membership counts/cluster IDs (`ocg_memberships`, `ocg_communities`).
- `provenance.json`: source hashes, saved-run metadata, and matched `c.ocg` run references.
- `reproduce.py`: copy of the generation script. The maintained source is `pipelines/moonlitdb_path_atlas.py`.

Run `python pipelines/moonlitdb_path_atlas.py` from the repository root using an environment with the project's scientific dependencies. It reads existing distance caches and saved results; it does not contact UniProt or change raw inputs. It writes the generated image under `data/plots/moonlitdb_path_atlas/` and evidence under this directory. The image here is a companion copy of that output. Original data and distance caches remain in their project locations; this directory is not a standalone copy of all source networks. Layout/community coordinates can vary with library/runtime details; the saved coordinates preserve the image's actual placement.
