#!/usr/bin/env python3
"""Tests for period parsing/adjustment."""

import unittest

from plotdata.fault_transect.periods import (
    consecutive_start_flags, gap_start_flags, is_consecutive_boundary,
    parse_period_chunks, snap_end_date, snap_first_period_start, snap_gap_start,
    validate_and_adjust_periods)


class TestPeriods(unittest.TestCase):

    def test_valid_gap_periods_unchanged(self):
        periods = [('20141001', '20181224'), ('20181225', '20201224')]
        self.assertEqual(validate_and_adjust_periods(periods), periods)
        self.assertEqual(consecutive_start_flags(periods), [False, False])

    def test_consecutive_boundary_unchanged(self):
        periods = [
            ('20141001', '20181224'),
            ('20181225', '20201224'),
            ('20201224', '20260701'),
        ]
        adjusted = validate_and_adjust_periods(periods)
        self.assertEqual(adjusted[2][0], '20201224')
        self.assertEqual(adjusted[2][1], '20260701')
        self.assertEqual(consecutive_start_flags(adjusted), [False, False, True])

    def test_consecutive_touching_boundary(self):
        periods = [('20141001', '20181224'), ('20181224', '20201224')]
        adjusted = validate_and_adjust_periods(periods)
        self.assertEqual(adjusted[1][0], '20181224')
        self.assertTrue(is_consecutive_boundary('20181224', '20181224'))

    def test_overlap_raises(self):
        with self.assertRaises(ValueError):
            validate_and_adjust_periods([
                ('20141001', '20181224'),
                ('20181224', '20201224'),
                ('20201220', '20260701'),
            ])

    def test_start_after_end_raises(self):
        with self.assertRaises(ValueError):
            validate_and_adjust_periods([('20200101', '20190101')])

    def test_parse_chunks(self):
        periods = parse_period_chunks(['20141001:20181224,20181225:20201224'])
        self.assertEqual(periods, [('20141001', '20181224'), ('20181225', '20201224')])

    def test_gap_start_flags(self):
        periods = [
            ('20141001', '20181224'),
            ('20181225', '20201225'),
            ('20201225', '20260701'),
        ]
        self.assertEqual(gap_start_flags(periods), [False, True, False])
        self.assertEqual(consecutive_start_flags(periods), [False, False, True])

    def test_four_period_mixed_boundaries(self):
        periods = [
            ('20141001', '20181224'),
            ('20181225', '20201225'),
            ('20201225', '20230601'),
            ('20230701', '20260701'),
        ]
        self.assertEqual(gap_start_flags(periods), [False, True, False, True])
        self.assertEqual(consecutive_start_flags(periods), [False, False, True, False])
        for gap, consec in zip(gap_start_flags(periods), consecutive_start_flags(periods)):
            self.assertFalse(gap and consec)

    def test_snap_gap_start_uses_first_acquisition_on_or_after(self):
        dates = ['20181220', '20181222', '20181226', '20190101']
        self.assertEqual(snap_gap_start(dates, '20181225'), '20181226')

    def test_snap_first_period_start_uses_last_acquisition_on_or_before(self):
        dates = ['20141020', '20181222', '20190101']
        self.assertEqual(snap_first_period_start(dates, '20141001'), '20141020')

    def test_snap_end_date_uses_last_acquisition_on_or_before(self):
        dates = ['20141020', '20181222', '20190101']
        self.assertEqual(snap_end_date(dates, '20181224'), '20181222')


if __name__ == '__main__':
    unittest.main()
