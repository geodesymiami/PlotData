#!/usr/bin/env python3
"""Tests for fault transect input resolution helpers."""

import unittest

from plotdata.fault_transect.load_data import resolve_dataset_label


class TestResolveDatasetLabel(unittest.TestCase):

    def test_ascending_from_project_path(self):
        path = '/scratch/EtnaSenA44/mintpy'
        self.assertEqual(resolve_dataset_label(path), 'ascending')

    def test_descending_from_project_path(self):
        path = '/scratch/EtnaSenDT124/miaplpy/network_single_reference'
        self.assertEqual(resolve_dataset_label(path), 'descending')

    def test_horizontal_from_eos_filename(self):
        eos = ('/scratch/Etna/mintpy/S1_horz_044_124_mintpy_20141020_20260626'
               '_POLYGON.he5')
        self.assertEqual(resolve_dataset_label('/scratch/Etna/mintpy', eos), 'horizontal')

    def test_vertical_from_eos_filename(self):
        eos = ('/scratch/Etna/mintpy/S1_vert_044_124_mintpy_20141020_20260626'
               '_POLYGON.he5')
        self.assertEqual(resolve_dataset_label('/scratch/Etna/mintpy', eos), 'vertical')

    def test_horz_directory(self):
        path = '/scratch/Etna/horz_timeseries'
        self.assertEqual(resolve_dataset_label(path), 'horizontal')


if __name__ == '__main__':
    unittest.main()
