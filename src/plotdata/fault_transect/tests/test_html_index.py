#!/usr/bin/env python3
"""Tests for HTML index generation."""

import os
import tempfile
import unittest

from plotdata.fault_transect.html_index import build_index_html, write_index_html
from plotdata.fault_transect.naming import index_html_stem


class TestHtmlIndex(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = self.tmp.name
        self.img = os.path.join(self.out_dir, 'EtnaSenA44_Fiandaca_map.png')
        self.txt = os.path.join(self.out_dir, 'EtnaSenA44_Fiandaca_map.txt')
        open(self.img, 'w').close()
        open(self.txt, 'w').close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_html_stem(self):
        self.assertEqual(index_html_stem('EtnaSenA44', 'Fiandaca'), 'EtnaSenA44_Fiandaca')
        self.assertEqual(index_html_stem('EtnaSenA44', ''), 'EtnaSenA44')

    def test_write_named_and_index_copy(self):
        entries = [{'group': 'EtnaSenA44 map', 'image': self.img, 'txt': self.txt}]
        named, index = write_index_html(
            self.out_dir, 'plot_fault_transect.py test', entries, 'EtnaSenA44_Fiandaca')
        self.assertEqual(named, os.path.join(self.out_dir, 'EtnaSenA44_Fiandaca.html'))
        self.assertEqual(index, os.path.join(self.out_dir, 'index.html'))
        self.assertTrue(os.path.isfile(named))
        self.assertTrue(os.path.isfile(index))
        with open(named, encoding='utf-8') as handle:
            named_body = handle.read()
        with open(index, encoding='utf-8') as handle:
            index_body = handle.read()
        self.assertEqual(named_body, index_body)
        self.assertIn('EtnaSenA44_Fiandaca', named_body)

    def test_build_index_html_links(self):
        page = build_index_html(
            self.out_dir, 'cmd', [{'group': 'g', 'image': self.img, 'txt': self.txt}],
            title='EtnaSenA44_Fiandaca')
        self.assertIn('EtnaSenA44_Fiandaca_map.png', page)
        self.assertIn('EtnaSenA44_Fiandaca_map.txt', page)


if __name__ == '__main__':
    unittest.main()
