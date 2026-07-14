#!/usr/bin/env python3
"""Tests for fault transect output tag naming."""

import unittest

from plotdata.fault_transect.naming import (
    extract_fault_tag, format_fault_plot_title, format_dataset_display_label,
    format_figure_suptitle, resolve_output_tag, index_html_stem,
    build_basename, MAP_LABEL, PROFILE_LABEL, multi_period_stem)


class TestExtractFaultTag(unittest.TestCase):

    def test_fault_suffix_before_underscore(self):
        self.assertEqual(extract_fault_tag('FiandacaFault_FA.kmz'), 'Fiandaca')

    def test_underscore_suffix(self):
        self.assertEqual(extract_fault_tag('Fiandaca_FA.kmz'), 'Fiandaca')

    def test_no_fault_or_underscore(self):
        self.assertEqual(extract_fault_tag('FiandacaFA.kmz'), 'FiandacaFA')

    def test_lowercase_fault(self):
        self.assertEqual(extract_fault_tag('/path/Pernicana_fault_system.kmz'), 'Pernicana')

    def test_explicit_tag_wins(self):
        self.assertEqual(resolve_output_tag('FiandacaFault_FA.kmz', 'Custom'), 'Custom')

    def test_auto_when_tag_empty(self):
        self.assertEqual(resolve_output_tag('Fiandaca_FA.kmz', ''), 'Fiandaca')
        self.assertEqual(resolve_output_tag('Fiandaca_FA.kmz', None), 'Fiandaca')

    def test_index_html_stem(self):
        self.assertEqual(index_html_stem('EtnaSenA44', 'Fiandaca'), 'EtnaSenA44_Fiandaca')

    def test_format_fault_plot_title(self):
        self.assertEqual(format_fault_plot_title('Fiandaca'), 'Fiandaca Fault')
        self.assertEqual(format_fault_plot_title(''), '')

    def test_format_dataset_display_label(self):
        self.assertEqual(format_dataset_display_label('ascending'), 'Ascending')
        self.assertEqual(format_dataset_display_label('horizontal'), 'Horizontal')
        self.assertEqual(format_dataset_display_label(''), '')

    def test_format_figure_suptitle(self):
        self.assertEqual(format_figure_suptitle('Fiandaca Fault', 'ascending'),
                         'Fiandaca Fault\nAscending')
        self.assertEqual(format_figure_suptitle('Fiandaca Fault', ''), 'Fiandaca Fault')
        self.assertEqual(format_figure_suptitle('', 'descending'), 'Descending')

    def test_build_basename(self):
        self.assertEqual(
            build_basename('EtnaSenA44', 'Fiandaca', MAP_LABEL, '20141020', '20260626'),
            'EtnaSenA44_Fiandaca_map_20141020_20260626')
        self.assertEqual(
            build_basename('EtnaSenA44', 'Fiandaca', PROFILE_LABEL, '20141020', '20260626'),
            'EtnaSenA44_Fiandaca_profile_20141020_20260626')

    def test_multi_period_stem(self):
        periods = [('20141020', '20181231'), ('20190101', '20260626')]
        self.assertEqual(
            multi_period_stem('EtnaSenA44', 'Fiandaca', PROFILE_LABEL, periods),
            'EtnaSenA44_Fiandaca_profile_20141020-20181231_20190101-20260626')


if __name__ == '__main__':
    unittest.main()
