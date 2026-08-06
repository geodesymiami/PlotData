#!/usr/bin/env python3
"""Tests for horz/vert HDFEOS output metadata."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from plotdata.cli.horzvert_timeseries import create_hdfeos_output


class TestHorzvertHdfeosOutput(unittest.TestCase):
    def test_create_hdfeos_output_mintpy_file_type(self):
        """Top-level lat/lon aliases must not make MintPy infer geometry."""
        from mintpy.utils import readfile

        dates = np.array([b"20250104", b"20250116"], dtype="S8")
        cube = np.zeros((2, 3, 4), dtype=np.float32)
        cube[1] = 0.01
        lat = np.linspace(36.35, 36.36, 3, dtype=np.float32)
        lon = np.linspace(25.44, 25.45, 4, dtype=np.float32)
        mask = np.ones((3, 4), dtype=bool)
        meta = {
            "Y_FIRST": "36.36",
            "X_FIRST": "25.44",
            "Y_STEP": "-0.00045",
            "X_STEP": "0.00056",
            "REF_LAT": "36.3573",
            "REF_LON": "25.4461",
            "REF_Y": "1",
            "REF_X": "2",
            "relative_orbit": "109",
            "ORBIT_DIRECTION": "DESCENDING",
        }

        with tempfile.TemporaryDirectory() as tmp:
            he5 = Path(tmp) / "S1_vert_109_029_test.he5"
            create_hdfeos_output(
                cube,
                dates,
                mask,
                np.array([0, 12], dtype=np.uint8),
                np.zeros(2, dtype=np.float32),
                lat,
                lon,
                meta,
                str(he5),
                3,
                4,
            )
            atr = readfile.read_attribute(str(he5))
            self.assertEqual(atr["FILE_TYPE"], "HDFEOS")


if __name__ == "__main__":
    unittest.main()
