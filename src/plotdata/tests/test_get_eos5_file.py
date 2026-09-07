#!/usr/bin/env python3
"""Tests for get_eos5_file path resolution."""

import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from plotdata.helper_functions import get_eos5_file


class TestGetEos5File(unittest.TestCase):
    def test_file_with_trailing_slash(self):
        with tempfile.TemporaryDirectory() as td:
            he5 = os.path.join(td, "S1_desc_009_Del4DS_coh075.he5")
            open(he5, "wb").close()
            with patch("sys.stdout", new_callable=StringIO):
                got = get_eos5_file(he5 + "/", scratch=td)
            self.assertEqual(got, he5)

    def test_directory_picks_newest_he5(self):
        with tempfile.TemporaryDirectory() as td:
            net = os.path.join(td, "network_delaunay_4")
            os.makedirs(net)
            older = os.path.join(net, "S1_old.he5")
            newer = os.path.join(net, "S1_new.he5")
            open(older, "wb").close()
            os.utime(older, (1, 1))
            open(newer, "wb").close()
            os.utime(newer, (100, 100))
            with patch("sys.stdout", new_callable=StringIO):
                got = get_eos5_file(net + "/", scratch=td)
            self.assertEqual(got, newer)


if __name__ == "__main__":
    unittest.main()
