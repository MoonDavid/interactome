"""Verify supplied full-family q-values survive display choices."""
import unittest

import polars as pl
import numpy as np
from unittest.mock import patch
from polars.testing import assert_frame_equal

from moonlighting import centrality_heatmap, cross_network_metric_figure
from moonlighting import multi_network_centrality_comparison, bh_adjust
from topology_display import static_topology_boxplot


class TopologyDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tests = pl.DataFrame({
            'group': ['HuRI'], 'network': ['HuRI'], 'metric': ['degree'],
            'rank_biserial': [.5], 'p_value': [1e-9], 'fdr_bh': [.2],
        })
        self.metrics = pl.DataFrame({
            'network': ['HuRI'] * 4, 'protein': ['A', 'B', 'C', 'D'],
            'group': ['MoonLitDB', 'MoonLitDB', 'Background', 'Background'],
            'degree': [3., 4., 1., 2.],
        })

    def test_heatmap_q_labels_preserve_effect_and_adjusted_test(self):
        q = centrality_heatmap(self.tests, annotation_mode='q').data[0]
        effect = centrality_heatmap(self.tests, annotation_mode='effect').data[0]
        self.assertEqual(q.z[0][0], effect.z[0][0])
        self.assertIn('ns', q.text[0][0])
        self.assertIn('q=0.2', q.text[0][0])
        self.assertIn('+0.50', effect.text[0][0])

    def test_box_label_choice_preserves_full_data_and_source_name(self):
        for mode, expected in [('q', 'q=0.2'), ('effect', 'r=+0.50')]:
            figure = cross_network_metric_figure(
                self.metrics, 'degree', self.tests, set_name='MoonLitDB', annotation_mode=mode)
            annotation = next(a for a in figure.layout.annotations if a.x == 'HuRI')
            self.assertIn('ns', annotation.text)
            self.assertIn(expected, annotation.text)
            self.assertIn('MoonLitDB vs Background', annotation.hovertext)
            self.assertEqual(sum(len(trace.y) for trace in figure.data), 4)

    def test_optional_raw_p_significance(self):
        figure = cross_network_metric_figure(
            self.metrics, 'degree', self.tests, set_name='MoonLitDB', significance_column='p_value')
        annotation = next(a for a in figure.layout.annotations if a.x == 'HuRI')
        self.assertIn('****', annotation.text)
        self.assertIn('p=1e-09', annotation.text)
        self.assertIn('Unadjusted p', annotation.hovertext)
        metrics = self.metrics.with_columns(pl.col('group').eq('MoonLitDB').alias('is_moonlighting'))
        fig = static_topology_boxplot(metrics, 'degree', self.tests, 'MoonLitDB', significance_column='p_value')
        self.assertIn('**** (raw p)', fig.axes[0].texts[0].get_text())
        self.assertIn('p=1e-09', fig.axes[0].texts[0].get_text())
        self.assertNotIn('q=', fig.axes[0].texts[0].get_text())
        self.assertIn('r=+0.50', fig.axes[0].texts[0].get_text())

    def test_static_bracket_uses_adjusted_significance_and_shows_raw_p(self):
        metrics = self.metrics.with_columns(pl.col('group').eq('MoonLitDB').alias('is_moonlighting'))
        fig = static_topology_boxplot(metrics, 'degree', self.tests, 'MoonLitDB')
        label = fig.axes[0].texts[0].get_text()
        self.assertIn('ns', label)
        self.assertIn('p=1e-09', label)
        self.assertIn('q=0.2', label)
        self.assertIn('r=+0.50', label)

    def test_network_correction_independent_of_other_networks(self):
        nodes = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
        graph = {'nodes': nodes, 'edge_count': 5}
        first = pl.DataFrame({'protein': nodes, 'degree': [6., 5., 1., 2., 3., 4.],
                              'strength': [1., 2., 3., 4., 5., 6.]})
        second = first.with_columns(pl.lit(1.).alias('degree'))
        with patch('moonlighting.cached_network_centralities', return_value=(first, {})):
            single = multi_network_centrality_comparison({'A': graph}, {'A', 'B'},
                                                        correction_scope='within_network')
        with patch('moonlighting.cached_network_centralities', side_effect=[(first, {}), (second, {})]):
            combined = multi_network_centrality_comparison({'A': graph, 'B': graph}, {'A', 'B'},
                                                          correction_scope='within_network')
        for key in ('profiles', 'hub_enrichment'):
            assert_frame_equal(single[key], combined[key].filter(pl.col('network') == 'A'))
            for part in combined[key].partition_by('network'):
                np.testing.assert_allclose(part['fdr_bh'].to_numpy(), bh_adjust(part['p_value']))


if __name__ == '__main__':
    unittest.main()
