#!/usr/bin/env python3
"""Tests for sampling-point geometry along the fault."""

import unittest

import numpy as np

from plotdata.fault_transect.fault_sampling import (
    sample_points, cumulative_distance_km, offset_latlon, local_east_north_km)


class TestFaultSampling(unittest.TestCase):

    def test_east_west_fault_left_is_north(self):
        # walking east: tangent = (1, 0), left normal = (0, 1) = north
        coords = [(15.0, 37.0), (15.5, 37.0)]
        points = sample_points(coords, along_step_km=5.0)
        self.assertGreater(len(points), 2)
        for p in points:
            self.assertAlmostEqual(p.tangent[0], 1.0, places=3)
            self.assertAlmostEqual(p.tangent[1], 0.0, places=3)
            self.assertAlmostEqual(p.left_normal[0], 0.0, places=3)
            self.assertAlmostEqual(p.left_normal[1], 1.0, places=3)

    def test_flipped_fault_left_is_south(self):
        coords = [(15.5, 37.0), (15.0, 37.0)]     # walking west
        points = sample_points(coords, along_step_km=5.0)
        for p in points:
            self.assertAlmostEqual(p.left_normal[1], -1.0, places=3)

    def test_spacing_and_along_range(self):
        coords = [(15.0, 37.0), (15.5, 37.0)]     # ~44 km at lat 37
        points = sample_points(coords, along_step_km=1.0, along_start_km=5.0, along_end_km=10.0)
        along = [p.along_km for p in points]
        self.assertAlmostEqual(along[0], 5.0)
        self.assertAlmostEqual(along[-1], 10.0)
        steps = np.diff(along)
        np.testing.assert_allclose(steps, 1.0, rtol=1e-6)

    def test_along_start_beyond_end_raises(self):
        coords = [(15.0, 37.0), (15.01, 37.0)]    # ~0.9 km
        with self.assertRaises(ValueError):
            sample_points(coords, along_step_km=1.0, along_start_km=5.0)

    def test_cumulative_distance(self):
        coords = [(15.0, 37.0), (15.0, 37.5), (15.0, 38.0)]
        cum = cumulative_distance_km(coords)
        self.assertAlmostEqual(cum[0], 0.0)
        self.assertAlmostEqual(cum[2], 2 * cum[1], places=6)

    def test_offset_latlon_roundtrip(self):
        lat, lon = offset_latlon(37.0, 15.0, east_km=1.0, north_km=2.0)
        east, north = local_east_north_km(37.0, 15.0, [lat], [lon])
        self.assertAlmostEqual(east[0], 1.0, places=4)
        self.assertAlmostEqual(north[0], 2.0, places=4)


if __name__ == '__main__':
    unittest.main()
