#!/usr/bin/env python3
"""Tests for offset colorscale limit resolution."""

import unittest

import numpy as np

from plotdata.fault_transect.offset import OffsetSeries
from plotdata.fault_transect.limits import build_offset_color_limits, global_symmetric_limit


class TestLimits(unittest.TestCase):

    def test_global_symmetric_limit(self):
        s1 = OffsetSeries(offset=[1.0, -3.0, np.nan])
        s2 = OffsetSeries(offset=[2.0])
        self.assertAlmostEqual(global_symmetric_limit([s1, s2]), 3.0)

    def test_build_offset_color_limits_from_vlim(self):
        lims = build_offset_color_limits(2, (-4.0, 4.0), False, [])
        self.assertEqual(lims, [(-4.0, 4.0), (-4.0, 4.0)])

    def test_build_offset_color_limits_auto_colorscale(self):
        s1 = OffsetSeries(offset=[1.0, -2.0])
        s2 = OffsetSeries(offset=[0.5])
        lims = build_offset_color_limits(2, None, True, [s1, s2])
        self.assertEqual(lims, [(-2.0, 2.0), (-2.0, 2.0)])

    def test_build_offset_color_limits_auto_per_period(self):
        lims = build_offset_color_limits(2, None, False, [])
        self.assertEqual(lims, [None, None])


if __name__ == '__main__':
    unittest.main()
