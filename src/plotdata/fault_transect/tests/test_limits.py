#!/usr/bin/env python3
"""Tests for shared value limit resolution."""

import unittest

import numpy as np

from plotdata.fault_transect.offset import OffsetSeries
from plotdata.fault_transect.limits import build_value_limits, global_symmetric_limit


class TestLimits(unittest.TestCase):

    def test_global_symmetric_limit(self):
        s1 = OffsetSeries(offset=[1.0, -3.0, np.nan])
        s2 = OffsetSeries(offset=[2.0])
        self.assertAlmostEqual(global_symmetric_limit([s1, s2]), 3.0)

    def test_build_value_limits_from_ylim(self):
        lims = build_value_limits(2, [(-4.0, 4.0), (-5.0, 5.0)], False, [], None, None)
        self.assertEqual(lims, [(-4.0, 4.0), (-5.0, 5.0)])

    def test_build_value_limits_auto_colorscale(self):
        s1 = OffsetSeries(offset=[1.0, -2.0])
        s2 = OffsetSeries(offset=[0.5])
        lims = build_value_limits(2, None, True, [s1, s2], None, None)
        self.assertEqual(lims, [(-2.0, 2.0), (-2.0, 2.0)])


if __name__ == '__main__':
    unittest.main()
