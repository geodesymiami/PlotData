#!/usr/bin/env python3
"""Tests for timeseries helpers and export."""

import tempfile
import unittest

import numpy as np

from plotdata.fault_transect.export import read_timeseries_txt, write_timeseries_txt
from plotdata.fault_transect.timeseries import (
    LocationTimeseries, TimeseriesBundle, interior_period_dates,
    plotted_interior_period_dates, stacked_timeseries_y, stacked_profile_y)


class TestInteriorPeriodDates(unittest.TestCase):

    def test_single_period_empty(self):
        self.assertEqual(interior_period_dates([('20141001', '20181222')]), [])

    def test_three_periods(self):
        periods = [
            ('20141001', '20181222'),
            ('20181228', '20201225'),
            ('20201225', '20260701'),
        ]
        self.assertEqual(interior_period_dates(periods),
                         ['20181222', '20181228', '20201225'])

    def test_consecutive_touching_dedupes(self):
        periods = [('20141001', '20181222'), ('20181222', '20201225')]
        self.assertEqual(interior_period_dates(periods), ['20181222'])

    def test_plotted_interior_snaps_to_acquisitions(self):
        periods = [
            ('20141001', '20181222'),
            ('20181228', '20201225'),
            ('20201225', '20260701'),
        ]
        plotted = ['20141020', '20181222', '20181228', '20201225', '20260701']
        self.assertEqual(plotted_interior_period_dates(periods, plotted),
                         ['20181222', '20181228', '20201225'])


class TestStackedTimeseriesY(unittest.TestCase):

    def test_anchored_at_first_date(self):
        y0 = stacked_timeseries_y(np.array([1.0, 2.0, 3.0]), 0, 10.0)
        y1 = stacked_timeseries_y(np.array([4.0, 5.0, 6.0]), 1, 10.0)
        self.assertEqual(y0[0], 0.0)
        self.assertEqual(y1[0], 10.0)
        self.assertAlmostEqual(y1[0] - y0[0], 10.0)
        self.assertAlmostEqual(y1[2] - y0[2], 10.0)

    def test_stacked_profile_y(self):
        y = stacked_profile_y(np.array([1.0, 3.0, 5.0]), 4.0)
        self.assertAlmostEqual(float(np.nanmedian(y)), 4.0)


class TestTimeseriesExport(unittest.TestCase):

    def test_roundtrip(self):
        bundle = TimeseriesBundle(
            dates=['20190101', '20190201'],
            unit='cm',
            reference_side='left',
            locations=[
                LocationTimeseries(0, 0.0, 37.5, 14.5, np.array([1.0, 2.0])),
                LocationTimeseries(2, 2.0, 37.51, 14.51, np.array([3.0, np.nan])),
            ])
        bracket = 'test bracket'
        with tempfile.TemporaryDirectory() as tmp:
            path = f'{tmp}/ts.txt'
            write_timeseries_txt(path, bundle, bracket)
            loaded, got_bracket = read_timeseries_txt(path)
        self.assertEqual(got_bracket, bracket)
        self.assertEqual(loaded.dates, bundle.dates)
        self.assertEqual(len(loaded.locations), 2)
        self.assertAlmostEqual(loaded.locations[0].offset[1], 2.0)
        self.assertTrue(np.isnan(loaded.locations[1].offset[1]))


if __name__ == '__main__':
    unittest.main()
