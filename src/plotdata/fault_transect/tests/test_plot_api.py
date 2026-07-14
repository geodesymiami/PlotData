#!/usr/bin/env python3
"""Tests for plot_api helpers."""

import unittest

from plotdata.fault_transect.plot_api import (
    title_coords, stacked_map_lat_offset, stacked_map_lon_offset,
    stacked_map_axis_offset, stacked_map_ytick_pairs, stacked_map_xtick_pairs,
    compute_map_lat_stack_step, compute_map_lon_stack_step, compute_map_stack_step,
    print_offset_summaries, stacked_curve_y_offset)


class TestStackedCurveOffset(unittest.TestCase):

    def test_first_curve_at_baseline(self):
        self.assertEqual(stacked_curve_y_offset(0, 5, 2.0), 0.0)

    def test_later_curves_stack_upward(self):
        self.assertEqual(stacked_curve_y_offset(3, 5, 2.0), 6.0)


class TestTitleCoords(unittest.TestCase):

    def test_upper_right_default(self):
        x, y, ha, va = title_coords('upper-right', 10.0, 20.0, 30.0, 40.0)
        self.assertEqual(ha, 'right')
        self.assertEqual(va, 'top')
        self.assertAlmostEqual(x, 19.8)
        self.assertAlmostEqual(y, 39.8)

    def test_upper_left(self):
        x, y, ha, va = title_coords('upper-left', 10.0, 20.0, 30.0, 40.0)
        self.assertEqual(ha, 'left')
        self.assertEqual(va, 'top')
        self.assertAlmostEqual(x, 10.2)
        self.assertAlmostEqual(y, 39.8)

    def test_lower_right_with_lat_offset(self):
        x, y, ha, va = title_coords('lower-right', 0.0, 10.0, 0.0, 10.0,
                                           lat_offset=5.0)
        self.assertEqual(ha, 'right')
        self.assertEqual(va, 'bottom')
        self.assertAlmostEqual(y, 5.2)

    def test_stacked_lat_offset_first_period_unshifted(self):
        self.assertEqual(stacked_map_lat_offset(0, 0.25), 0.0)
        self.assertEqual(stacked_map_lat_offset(1, 0.25), -0.25)
        self.assertEqual(stacked_map_lat_offset(3, 0.25), -0.75)

    def test_stacked_lon_offset(self):
        self.assertEqual(stacked_map_lon_offset(0, 0.15), 0.0)
        self.assertEqual(stacked_map_lon_offset(2, 0.15), 0.30)

    def test_stacked_axis_offset(self):
        self.assertEqual(stacked_map_axis_offset(1, 0.2, 'lat'), (-0.2, 0.0))
        self.assertEqual(stacked_map_axis_offset(1, 0.2, 'lon'), (0.0, 0.2))

    def test_stacked_ytick_labels_match_single_period(self):
        pairs = stacked_map_ytick_pairs(37.50, 37.70, pad=0.02, lat_step=0.30,
                                        n_periods=3, nbins=3)
        self.assertTrue(pairs)
        for loc, true_lat in pairs:
            self.assertAlmostEqual(loc, true_lat)
            self.assertGreaterEqual(true_lat, 37.50 - 0.02)
            self.assertLessEqual(true_lat, 37.70 + 0.02)

    def test_stacked_xtick_labels_match_single_period(self):
        pairs = stacked_map_xtick_pairs(14.5, 15.0, pad=0.02, lon_step=0.30,
                                          n_periods=3, nbins=3)
        self.assertTrue(pairs)
        for loc, true_lon in pairs:
            self.assertAlmostEqual(loc, true_lon)

    def test_title_offset_nudge(self):
        x, y, ha, va = title_coords('upper-right', 10.0, 20.0, 30.0, 40.0,
                                    title_offset_lon=0.05, title_offset_lat=-0.02)
        self.assertAlmostEqual(x, 19.85)
        self.assertAlmostEqual(y, 39.78)

    def test_compute_map_lat_stack_step_manual(self):
        self.assertAlmostEqual(
            compute_map_lat_stack_step(0.5, 37.5, 37.7, manual=0.4), 0.4)

    def test_compute_map_lat_stack_step_auto(self):
        step = compute_map_lat_stack_step(0.5, 37.5, 37.7)
        self.assertGreater(step, 0.2)

    def test_compute_map_lon_stack_step_manual(self):
        self.assertAlmostEqual(
            compute_map_lon_stack_step(0.5, 37.6, 14.5, 15.0, manual=0.35), 0.35)

    def test_compute_map_stack_step_axis(self):
        self.assertAlmostEqual(
            compute_map_stack_step('lat', 0.5, 14.5, 15.0, 37.5, 37.7, manual=0.4), 0.4)
        self.assertAlmostEqual(
            compute_map_stack_step('lon', 0.5, 14.5, 15.0, 37.5, 37.7, manual=0.4), 0.4)

    def test_print_offset_summaries(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_offset_summaries([{
                'kind': 'map_stack',
                'project': 'Etna',
                'n_periods': 2,
                'axis': 'lat',
                'step_deg': 0.25,
                'source': 'auto',
            }])
        out = buf.getvalue()
        self.assertIn('latitude step 0.2500 deg', out)
        self.assertNotIn('lat shift', out)

    def test_print_offset_summaries_lon_axis(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_offset_summaries([{
                'kind': 'map_stack',
                'project': 'Etna',
                'n_periods': 3,
                'axis': 'lon',
                'step_deg': 0.12,
                'source': 'manual',
            }])
        self.assertIn('longitude step 0.1200 deg', buf.getvalue())

    def test_invalid_position(self):
        with self.assertRaises(ValueError):
            title_coords('center', 0.0, 1.0, 0.0, 1.0)


if __name__ == '__main__':
    unittest.main()
