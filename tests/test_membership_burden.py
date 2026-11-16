"""Protein-level overlap must preserve zeros and a shared union-complement null."""
import polars as pl

from clustering_explorer import membership_burden, fuzzy_membership_counts


def test_fuzzy_threshold_and_optional_argmax_fallback():
    scores = [[.6, .4], [.5, .5], [.9, .1]]
    low = fuzzy_membership_counts(['A', 'B', 'C'], scores, .4, False)
    high = fuzzy_membership_counts(['A', 'B', 'C'], scores, .7, False)
    fallback = fuzzy_membership_counts(['A', 'B', 'C'], scores, .7, True)
    assert low['memberships'].to_list() == [2, 2, 1]
    assert high['memberships'].to_list() == [0, 0, 1]
    assert fallback['memberships'].to_list() == [1, 1, 1]


def test_fuzzy_rejects_unnormalized_affiliations():
    import pytest
    with pytest.raises(ValueError, match='row-normalized'):
        fuzzy_membership_counts(['A'], [[3, 2]], .2)


def test_distinct_memberships_and_shared_background():
    population = pl.DataFrame({'protein': ['A', 'B', 'C', 'D', 'E']})
    memberships = pl.DataFrame({'protein': ['A', 'A', 'A', 'B', 'C', 'D', 'X'],
        'community': ['1', '1', '2', '1', '2', None, '3']})
    result, counts = membership_burden(memberships, population,
        {'first': {'A', 'B'}, 'second': {'B', 'C'}})
    assert dict(counts.select('protein', 'memberships').iter_rows()) == {
        'A': 2, 'B': 1, 'C': 1, 'D': 0, 'E': 0}
    first = result.filter(pl.col('source') == 'first').row(0, named=True)
    assert first['background_n'] == 2  # Exclude the entire A/B/C source union.
    assert first['protein_set_mean'] == 1.5
    assert first['background_mean'] == 0
    assert first['protein_set_multi_fraction'] == .5
    assert first['rank_biserial'] == 1
    assert first['overlapping_proteins'] == 1


def test_missing_source_has_unavailable_tests():
    population = pl.DataFrame({'protein': ['A', 'B']})
    memberships = pl.DataFrame({'protein': ['A'], 'community': ['1']})
    result, _ = membership_burden(memberships, population, {'missing': {'X'}})
    row = result.row(0, named=True)
    assert row['protein_set_n'] == 0
    assert row['overlap_pvalue'] is None
    assert row['burden_pvalue'] is None


def test_disjoint_partition_does_not_create_overlap():
    population = pl.DataFrame({'protein': ['A', 'B']})
    memberships = pl.DataFrame({'protein': ['A', 'B'], 'community': ['1', '2']})
    result, _ = membership_burden(memberships, population, {'source': {'A'}})
    row = result.row(0, named=True)
    assert row['overlapping_proteins'] == 0
    assert row['overlap_pvalue'] == 1
    assert row['rank_biserial'] == 0
