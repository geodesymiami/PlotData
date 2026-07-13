#!/usr/bin/env python3
"""Tests for side-box sampling and offset computation on synthetic grids."""

import unittest

import numpy as np

from plotdata.fault_transect.fault_sampling import sample_points
from plotdata.fault_transect.sampling import sample_side_box, grid_latlon_vectors
from plotdata.fault_transect.offset import compute_offset_series


def synthetic_grid(north_value=2.0, south_value=-1.0, fault_lat=37.0):
    """Grid split by an E-W fault: north_value above fault_lat, south below."""
    lats = np.linspace(37.05, 36.95, 101)   # descending, like Y_FIRST + Y_STEP<0
    lons = np.linspace(14.95, 15.55, 121)
    north_mask = np.repeat((lats > fault_lat)[:, None], len(lons), axis=1)
    data = np.where(north_mask, north_value, south_value).astype(float)
    return data, lats, lons


class TestSampling(unittest.TestCase):

    def setUp(self):
        self.data, self.lats, self.lons = synthetic_grid()
        # E-W fault walking east: left = north
        self.points = sample_points([(15.0, 37.0), (15.5, 37.0)], along_step_km=2.0)

    def test_left_box_samples_north(self):
        point = self.points[len(self.points) // 2]
        left = sample_side_box(self.data, self.lats, self.lons, point, 'left',
                               half_along_km=1.0, perp_width_km=2.0, method='mean')
        self.assertAlmostEqual(left.value, 2.0, places=6)
        self.assertGreater(left.count, 0)

    def test_right_box_samples_south(self):
        point = self.points[len(self.points) // 2]
        right = sample_side_box(self.data, self.lats, self.lons, point, 'right',
                                half_along_km=1.0, perp_width_km=2.0, method='median')
        self.assertAlmostEqual(right.value, -1.0, places=6)

    def test_nearest_method(self):
        point = self.points[len(self.points) // 2]
        left = sample_side_box(self.data, self.lats, self.lons, point, 'left',
                               half_along_km=1.0, perp_width_km=2.0, method='nearest')
        self.assertAlmostEqual(left.value, 2.0, places=6)

    def test_all_nan_box(self):
        data = np.full_like(self.data, np.nan)
        point = self.points[0]
        result = sample_side_box(data, self.lats, self.lons, point, 'left',
                                 half_along_km=1.0, perp_width_km=2.0, method='mean')
        self.assertTrue(np.isnan(result.value))
        self.assertEqual(result.count, 0)

    def test_offset_series_sign(self):
        # reference = left = north (2.0); other = south (-1.0) -> offset = 3.0
        series = compute_offset_series(self.data, self.lats, self.lons, self.points,
                                       perp_width_km=2.0, along_step_km=2.0,
                                       sample_method='mean', reference_side='left')
        finite = [v for v in series.offset if np.isfinite(v)]
        self.assertGreater(len(finite), 0)
        for value in finite:
            self.assertAlmostEqual(value, 3.0, places=6)

    def test_offset_series_reference_right(self):
        series = compute_offset_series(self.data, self.lats, self.lons, self.points,
                                       perp_width_km=2.0, along_step_km=2.0,
                                       sample_method='mean', reference_side='right')
        finite = [v for v in series.offset if np.isfinite(v)]
        for value in finite:
            self.assertAlmostEqual(value, -3.0, places=6)

    def test_grid_latlon_vectors(self):
        attr = {'Y_FIRST': '37.05', 'Y_STEP': '-0.001', 'X_FIRST': '14.95',
                'X_STEP': '0.005', 'LENGTH': '101', 'WIDTH': '121'}
        lats, lons = grid_latlon_vectors(attr)
        self.assertEqual(len(lats), 101)
        self.assertEqual(len(lons), 121)
        self.assertAlmostEqual(lats[0], 37.05)
        self.assertAlmostEqual(lons[-1], 14.95 + 0.005 * 120)


if __name__ == '__main__':
    unittest.main()
