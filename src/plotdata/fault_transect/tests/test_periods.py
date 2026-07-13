#!/usr/bin/env python3
"""Tests for period parsing/adjustment and ylim parsing."""

import unittest

from plotdata.fault_transect.periods import (
    consecutive_start_flags, is_consecutive_boundary,
    parse_period_chunks, validate_and_adjust_periods, parse_ylim_tokens)


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


class TestYlimParsing(unittest.TestCase):

    def test_single_pair_all_periods(self):
        pairs = parse_ylim_tokens(['-4', '4'], 3)
        self.assertEqual(pairs, [(-4.0, 4.0)] * 3)

    def test_per_period_pairs(self):
        pairs = parse_ylim_tokens(['-4', '4', '-5', '5'], 2)
        self.assertEqual(pairs, [(-4.0, 4.0), (-5.0, 5.0)])

    def test_mismatched_count_raises(self):
        with self.assertRaises(ValueError):
            parse_ylim_tokens(['-4', '4', '-5', '5'], 3)


if __name__ == '__main__':
    unittest.main()
