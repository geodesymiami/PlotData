#!/usr/bin/env python3
"""Tests for horz/vert timeseries cache helpers."""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace

from plotdata.horzvert_cache import (
    build_hv_fingerprint,
    hv_cache_hit,
    hvparams_matches,
    hvparams_path,
    infer_output_subdir,
    processing_dir_span,
    should_recompute_hv,
    longest_output_subdir,
    write_hvparams,
)


class _Inps:
    ref_lalo = [36.4, 25.47]
    mask_vmin = [0.55, 0.55]
    interval_index = 2
    period = []
    start_date = []
    stop_date = []
    horz_az_angle = 90
    window_size = 3
    exclude_dates = []
    lat_step = -0.00014
    no_swap = False
    geom_file = None
    force = False
    overwrite = False


class TestHorzvertCache(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.geo1 = os.path.join(self.tmp.name, 'geo_S1_asc.he5')
        self.geo2 = os.path.join(self.tmp.name, 'geo_S1_desc.he5')
        self.vert = os.path.join(self.tmp.name, 'S1_vert_029_036_miaplpy.he5')
        self.horz = os.path.join(self.tmp.name, 'S1_horz_029_036_miaplpy.he5')
        open(self.geo1, 'w').close()
        open(self.geo2, 'w').close()

    def tearDown(self):
        self.tmp.cleanup()

    def _fingerprint(self):
        return build_hv_fingerprint(_Inps, ['geo_S1_asc.he5', 'geo_S1_desc.he5'])

    def test_cache_miss_when_products_missing(self):
        fingerprint = self._fingerprint()
        self.assertFalse(hv_cache_hit(self.vert, self.horz, self.geo1, self.geo2, fingerprint))

    def test_cache_fresh_when_newer_than_inputs_and_params_match(self):
        fingerprint = self._fingerprint()
        open(self.geo1, 'w').close()
        open(self.geo2, 'w').close()
        time.sleep(0.02)
        open(self.vert, 'w').close()
        open(self.horz, 'w').close()
        write_hvparams(self.vert, fingerprint)
        self.assertTrue(hv_cache_hit(self.vert, self.horz, self.geo1, self.geo2, fingerprint))

    def test_cache_stale_when_input_newer(self):
        fingerprint = self._fingerprint()
        open(self.vert, 'w').close()
        open(self.horz, 'w').close()
        write_hvparams(self.vert, fingerprint)
        time.sleep(0.02)
        open(self.geo1, 'w').close()
        self.assertFalse(hv_cache_hit(self.vert, self.horz, self.geo1, self.geo2, fingerprint))

    def test_cache_miss_when_fingerprint_mismatch(self):
        fingerprint = self._fingerprint()
        open(self.vert, 'w').close()
        open(self.horz, 'w').close()
        write_hvparams(self.vert, fingerprint)
        other_inps = _Inps()
        other_inps.ref_lalo = [36.5, 25.5]
        other = build_hv_fingerprint(other_inps, ['geo_S1_asc.he5', 'geo_S1_desc.he5'])
        self.assertFalse(hvparams_matches(self.vert, other))
        self.assertFalse(hv_cache_hit(self.vert, self.horz, self.geo1, self.geo2, other))

    def test_should_recompute_when_force(self):
        fingerprint = self._fingerprint()
        open(self.vert, 'w').close()
        open(self.horz, 'w').close()
        write_hvparams(self.vert, fingerprint)
        inps = SimpleNamespace(force=True, overwrite=False)
        self.assertTrue(should_recompute_hv(inps, self.vert, self.horz, self.geo1, self.geo2, fingerprint))

    def test_hvparams_path_suffix(self):
        self.assertEqual(hvparams_path(self.vert), f'{self.vert}.hvparams')

    def test_infer_output_subdir_keeps_dated_name(self):
        path = 'EtnaSenA44/miaplpy_202001_202412/network_delaunay_4/S1.he5'
        self.assertEqual(infer_output_subdir(path), 'miaplpy_202001_202412')
        self.assertEqual(infer_output_subdir('ChilesSenD142/mintpy/geo.he5'), 'mintpy')

    def test_longest_output_subdir_picks_longer_period(self):
        a = 'EtnaSenA44/miaplpy_202001_202412/network_delaunay_4/a.he5'
        b = 'EtnaSenD124/miaplpy_202001_202410/network_delaunay_4/b.he5'
        self.assertEqual(longest_output_subdir([a, b]), 'miaplpy_202001_202412')
        self.assertEqual(longest_output_subdir([b, a]), 'miaplpy_202001_202412')
        self.assertGreater(
            processing_dir_span('miaplpy_202001_202412'),
            processing_dir_span('miaplpy_202001_202410'),
        )


if __name__ == '__main__':
    unittest.main()
