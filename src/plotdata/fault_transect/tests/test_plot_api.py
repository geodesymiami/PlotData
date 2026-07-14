#!/usr/bin/env python3
"""Tests for plot_api helpers."""

import unittest

from plotdata.fault_transect.plot_api import title_coords, stacked_map_ytick_pairs


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

    def test_stacked_ytick_labels_match_single_period(self):
        pairs = stacked_map_ytick_pairs(37.50, 37.70, pad=0.02, lat_step=0.30,
                                        n_periods=3, nbins=3)
        self.assertTrue(pairs)
        for loc, true_lat in pairs:
            self.assertAlmostEqual(loc, true_lat)
            self.assertGreaterEqual(true_lat, 37.50 - 0.02)
            self.assertLessEqual(true_lat, 37.70 + 0.02)

    def test_invalid_position(self):
        with self.assertRaises(ValueError):
            title_coords('center', 0.0, 1.0, 0.0, 1.0)


if __name__ == '__main__':
    unittest.main()
