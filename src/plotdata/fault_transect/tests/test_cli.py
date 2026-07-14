#!/usr/bin/env python3
"""Tests for CLI parsing: periods, segment spec, naming, output dirs."""

import unittest

from plotdata.cli.plot_fault_transect import (
    parse_periods, format_map_title, cmd_line_parse, _side_by_side)
from plotdata.fault_transect.naming import build_basename
from plotdata.fault_transect.load_data import default_output_dir


class TestPeriodParsing(unittest.TestCase):

    def test_space_separated(self):
        periods = parse_periods(['20141020:20181231', '20190101:20260626'])
        self.assertEqual(periods, [('20141020', '20181231'), ('20190101', '20260626')])

    def test_comma_separated(self):
        periods = parse_periods(['20141020:20181231,20190101:20260626'])
        self.assertEqual(periods, [('20141020', '20181231'), ('20190101', '20260626')])

    def test_empty(self):
        self.assertEqual(parse_periods([]), [])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_periods(['2014:2018'])
        with self.assertRaises(ValueError):
            parse_periods(['20141020-20181231'])


class TestNaming(unittest.TestCase):

    def test_with_tag(self):
        self.assertEqual(build_basename('Etna', 'Pernicana', 'map', '20141020', '20260626'),
                         'Etna_Pernicana_map_20141020_20260626')

    def test_without_tag_raises(self):
        with self.assertRaises(ValueError):
            build_basename('Etna', '', 'map', '20141020', '20260626')

    def test_map_title(self):
        self.assertEqual(format_map_title('Pernicana', '20141020', '20260626'),
                         '2014-10-20 - 2026-06-26')
        self.assertEqual(format_map_title('', '20141020', '20260626'),
                         '2014-10-20 - 2026-06-26')

class TestCmdLineParse(unittest.TestCase):

    def test_defaults(self):
        inps = cmd_line_parse(['FiandacaFA.kmz', 'Etna/mintpy'])
        self.assertEqual(inps.tag_string, 'FiandacaFA')
        self.assertEqual(inps.plot_type, 'both')
        self.assertEqual(inps.along_step, 1.0)
        self.assertEqual(inps.perp_width, 0.5)
        self.assertEqual(inps.sample_method, 'mean')
        self.assertEqual(inps.reference_side, 'left')
        self.assertEqual(inps.plot_layout, 'stacked')
        self.assertEqual(inps.save, 'png')
        self.assertEqual(inps.profile_spacing, inps.along_step)  # default follows along-step
        self.assertFalse(inps.show_flag)
        self.assertFalse(inps.upload)
        self.assertEqual(inps.title_position, 'upper-right')

    def test_explicit_tag(self):
        inps = cmd_line_parse(['FiandacaFault_FA.kmz', 'Etna/mintpy', '--tag', 'Fiandaca'])
        self.assertEqual(inps.tag_string, 'Fiandaca')

    def test_auto_tag_from_fault(self):
        inps = cmd_line_parse(['FiandacaFault_FA.kmz', 'Etna/mintpy'])
        self.assertEqual(inps.tag_string, 'Fiandaca')

    def test_map_stack_axis_default(self):
        inps = cmd_line_parse(['FiandacaFA.kmz', 'Etna/mintpy'])
        self.assertEqual(inps.map_stack_axis, 'lat')

    def test_map_stack_axis_lon(self):
        inps = cmd_line_parse(['FiandacaFA.kmz', 'Etna/mintpy', '--map-stack-axis', 'lon'])
        self.assertEqual(inps.map_stack_axis, 'lon')

    def test_data_required(self):
        with self.assertRaises(SystemExit):
            cmd_line_parse(['fault.kmz'])

    def test_too_many_inputs(self):
        with self.assertRaises(SystemExit):
            cmd_line_parse(['fault.kmz', 'a', 'b', 'c', 'd', 'e'])

    def test_3d_layout_rejected(self):
        with self.assertRaises(SystemExit):
            cmd_line_parse(['fault.kmz', 'Etna/mintpy', '--plot-layout', '3d'])


class TestPeriodLayout(unittest.TestCase):

    def _inps(self, **kwargs):
        args = ['fault.kmz', 'Etna/mintpy']
        for key, value in kwargs.items():
            args.extend([f'--{key.replace("_", "-")}', str(value)])
        return cmd_line_parse(args)

    def test_single_period_never_combined(self):
        self.assertFalse(_side_by_side(self._inps(), 1))

    def test_auto_combines_compact_layouts(self):
        self.assertTrue(_side_by_side(self._inps(plot_layout='subplot'), 2))
        self.assertTrue(_side_by_side(self._inps(plot_layout='stacked'), 2))

    def test_auto_separates_fullpage_layout(self):
        inps = self._inps(plot_layout='separate', plot_type='profile')
        self.assertFalse(_side_by_side(inps, 2))

    def test_explicit_override(self):
        inps = self._inps(plot_layout='subplot', period_layout='separate-page')
        self.assertFalse(_side_by_side(inps, 2))


class TestOutputDir(unittest.TestCase):

    def test_mintpy_source(self):
        out = default_output_dir('/scratch/EtnaSenA44/mintpy/S1_x.he5', 'EtnaSenA44', 'mintpy')
        self.assertEqual(out, '/scratch/EtnaSenA44/transects_mintpy')

    def test_miaplpy_source(self):
        out = default_output_dir('/scratch/EtnaSenA44/miaplpy/network_single_reference/S1_x.he5',
                                 'EtnaSenA44', 'miaplpy')
        self.assertEqual(out, '/scratch/EtnaSenA44/transects_miaplpy')


if __name__ == '__main__':
    unittest.main()
