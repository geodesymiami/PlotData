#!/usr/bin/env python3
"""Tests for MintPy-style colormap resolution."""

import unittest

try:
    import matplotlib
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from plotdata.fault_transect.colormaps import coherence_colormap, resolve_colormap


@unittest.skipUnless(HAS_MPL, 'matplotlib not installed')
class TestColormaps(unittest.TestCase):

    def test_jet_default(self):
        cmap = resolve_colormap('jet')
        self.assertEqual(cmap(1.0)[:3][0], cmap(1.0)[0])  # smoke
        rgba_high = cmap(1.0)
        rgba_low = cmap(0.0)
        self.assertGreater(rgba_high[0], rgba_low[0])  # red channel higher at top

    def test_coherence_white_at_one(self):
        cmap = coherence_colormap()
        top = cmap(1.0)[:3]
        bottom = cmap(0.0)[:3]
        self.assertGreater(top[0], 0.9)
        self.assertGreater(top[1], 0.9)
        self.assertGreater(top[2], 0.9)
        self.assertLess(bottom[2], 0.5)

    def test_coherence_name(self):
        cmap = resolve_colormap('coherence')
        self.assertEqual(cmap.name, 'coherence')

    def test_reverse_suffix(self):
        cmap = resolve_colormap('jet_r')
        top = cmap(1.0)[:3]
        bottom = cmap(0.0)[:3]
        self.assertGreater(bottom[0], top[0])

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            resolve_colormap('not_a_real_colormap_xyz')


if __name__ == '__main__':
    unittest.main()
