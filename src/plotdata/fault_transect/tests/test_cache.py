#!/usr/bin/env python3
"""Tests for cache freshness and txt round-trip."""

import os
import tempfile
import time
import unittest

from plotdata.fault_transect.cache import (
    cache_is_fresh, combined_figure_is_fresh, figure_is_fresh, figure_style_matches,
    map_period_bracket, parse_txt_header, should_write_txt, txt_cache_hit,
    write_figure_style)
from plotdata.fault_transect.export import read_offset_txt, write_offset_txt
from plotdata.fault_transect.offset import OffsetSeries


class _Inps:
    fault_file = '/tmp/fault.kmz'
    fault_segment = 'all'
    fault_segment_by = 'auto'
    flip_fault = False
    along_step = 1.0
    perp_width = 0.5
    perp_offset = 0.5
    sample_method = 'mean'
    reference_side = 'left'
    profile_length = 4.0
    plot_layout = 'stacked'
    cloud_profiles = 0


class _FreshnessInps:
    force = False
    plots_only = False


class TestCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, 'data.txt')
        self.he5 = os.path.join(self.tmp.name, 'input.he5')
        self.kmz = os.path.join(self.tmp.name, 'fault.kmz')
        open(self.he5, 'w').close()
        open(self.kmz, 'w').close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_not_fresh_when_input_newer(self):
        open(self.cache, 'w').close()
        time.sleep(0.02)
        open(self.he5, 'w').close()
        self.assertFalse(cache_is_fresh(self.cache, self.he5, self.kmz))

    def test_cache_fresh_when_newer_than_inputs(self):
        open(self.he5, 'w').close()
        open(self.kmz, 'w').close()
        time.sleep(0.02)
        open(self.cache, 'w').close()
        self.assertTrue(cache_is_fresh(self.cache, self.he5, self.kmz))

    def _sample_series(self):
        series = OffsetSeries(reference_side='left', unit='cm/yr')
        series.along_km = [0.0]
        series.lat = [37.0]
        series.lon = [15.0]
        series.left_val = [1.0]
        series.right_val = [0.5]
        series.offset = [0.5]
        return series

    def test_txt_cache_hit_requires_matching_bracket(self):
        bracket = map_period_bracket(_Inps, '20141001', '20181224')
        write_offset_txt(self.cache, self._sample_series(), bracket)
        self.assertTrue(txt_cache_hit(self.cache, self.he5, bracket, self.kmz))
        self.assertFalse(txt_cache_hit(self.cache, self.he5, bracket + ' x', self.kmz))

    def test_should_write_txt_skips_when_fresh(self):
        bracket = map_period_bracket(_Inps, '20141001', '20181224')
        write_offset_txt(self.cache, self._sample_series(), bracket)
        inps = _FreshnessInps()
        self.assertFalse(should_write_txt(inps, self.cache, self.he5, bracket, self.kmz))

    def test_should_write_txt_when_force(self):
        bracket = map_period_bracket(_Inps, '20141001', '20181224')
        write_offset_txt(self.cache, self._sample_series(), bracket)
        inps = _FreshnessInps()
        inps.force = True
        self.assertTrue(should_write_txt(inps, self.cache, self.he5, bracket, self.kmz))

    def test_figure_fresh_requires_style_sidecar(self):
        txt = os.path.join(self.tmp.name, 'a.txt')
        img = os.path.join(self.tmp.name, 'a.png')
        open(self.he5, 'w').close()
        time.sleep(0.02)
        open(txt, 'w').close()
        time.sleep(0.02)
        open(img, 'w').close()
        style = 'cmap=jet|dpi=300'
        self.assertFalse(figure_is_fresh(img, txt, style, self.he5, self.kmz))
        write_figure_style(img, style)
        self.assertTrue(figure_is_fresh(img, txt, style, self.he5, self.kmz))
        self.assertFalse(figure_is_fresh(img, txt, 'cmap=viridis|dpi=300', self.he5, self.kmz))

    def test_combined_figure_requires_all_txts_fresh(self):
        txt1 = os.path.join(self.tmp.name, 'a.txt')
        txt2 = os.path.join(self.tmp.name, 'b.txt')
        img = os.path.join(self.tmp.name, 'combo.png')
        open(self.he5, 'w').close()
        time.sleep(0.02)
        open(txt1, 'w').close()
        open(txt2, 'w').close()
        time.sleep(0.02)
        open(img, 'w').close()
        style = 'map-style'
        write_figure_style(img, style)
        self.assertTrue(combined_figure_is_fresh(img, [txt1, txt2], style, self.he5, self.kmz))
        time.sleep(0.02)
        open(self.he5, 'w').close()
        self.assertFalse(combined_figure_is_fresh(img, [txt1, txt2], style, self.he5, self.kmz))

    def test_map_bracket_includes_segment_options(self):
        bracket = map_period_bracket(_Inps, '20141001', '20181224')
        self.assertIn('fault-segment=all', bracket)
        self.assertIn('flip-fault=0', bracket)

    def test_offset_txt_roundtrip(self):
        series = OffsetSeries(reference_side='left', unit='cm/yr')
        series.along_km = [0.0, 1.0]
        series.lat = [37.0, 37.1]
        series.lon = [15.0, 15.1]
        series.left_val = [1.0, 2.0]
        series.right_val = [0.5, 1.0]
        series.offset = [0.5, 1.0]
        bracket = map_period_bracket(_Inps, '20141001', '20181224')
        path = os.path.join(self.tmp.name, 'map.txt')
        write_offset_txt(path, series, bracket)
        loaded, loaded_bracket = read_offset_txt(path)
        self.assertEqual(loaded_bracket, bracket)
        self.assertEqual(len(loaded.along_km), 2)
        self.assertAlmostEqual(loaded.offset[1], 1.0)
        _, parsed = parse_txt_header(path)
        self.assertEqual(parsed, bracket)

    def test_figure_style_matches(self):
        img = os.path.join(self.tmp.name, 'fig.png')
        open(img, 'w').close()
        write_figure_style(img, 'style-a')
        self.assertTrue(figure_style_matches(img, 'style-a'))
        self.assertFalse(figure_style_matches(img, 'style-b'))


if __name__ == '__main__':
    unittest.main()
