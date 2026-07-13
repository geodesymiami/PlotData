#!/usr/bin/env python3
"""Tests for KMZ reading, segment joining, and joint KMZ writing."""

import os
import tempfile
import unittest
import zipfile

from plotdata.fault_transect.kmz_fault import (
    read_fault_kmz, join_segments, parse_segment_spec,
    write_fault_kmz, joint_output_paths, polyline_length_km)


def make_kmz(path, segments):
    """segments: list of (name, [(lon, lat), ...])"""
    placemarks = []
    for name, coords in segments:
        coord_str = ' '.join(f'{lon},{lat},0' for lon, lat in coords)
        placemarks.append(f"""
    <Placemark><name>{name}</name>
      <LineString><coordinates>{coord_str}</coordinates></LineString>
    </Placemark>""")
    kml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           + ''.join(placemarks) + '</Document></kml>')
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('doc.kml', kml)


class TestKmzFault(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_segments(self):
        path = os.path.join(self.tmp.name, 'fault.kmz')
        make_kmz(path, [('A', [(15.0, 37.0), (15.1, 37.0)]),
                        ('B', [(15.1, 37.0), (15.2, 37.05)])])
        segments = read_fault_kmz(path)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].name, 'A')
        self.assertEqual(len(segments[0].coords), 2)

    def test_join_unordered_and_reversed(self):
        # true chain: P0 -> P1 -> P2 -> P3 going east; stored shuffled + middle reversed
        p0, p1, p2, p3 = (15.0, 37.0), (15.1, 37.0), (15.2, 37.0), (15.3, 37.0)
        path = os.path.join(self.tmp.name, 'fault.kmz')
        make_kmz(path, [('mid_reversed', [p2, p1]),
                        ('east', [p2, p3]),
                        ('west', [p0, p1])])
        segments = read_fault_kmz(path)
        joined, report = join_segments(segments)
        lons = [c[0] for c in joined]
        # accept either direction, but must be monotonic through all 4 points
        if lons[0] > lons[-1]:
            lons = lons[::-1]
        self.assertEqual(lons, sorted(lons))
        self.assertAlmostEqual(lons[0], 15.0)
        self.assertAlmostEqual(lons[-1], 15.3)
        self.assertEqual(len(report.order), 3)
        self.assertFalse(report.warnings)

    def test_join_gap_warning(self):
        path = os.path.join(self.tmp.name, 'fault.kmz')
        make_kmz(path, [('A', [(15.0, 37.0), (15.1, 37.0)]),
                        ('B', [(15.3, 37.0), (15.4, 37.0)])])   # ~17 km gap
        segments = read_fault_kmz(path)
        _, report = join_segments(segments, gap_warn_km=1.0)
        self.assertEqual(len(report.warnings), 1)

    def test_parse_segment_spec(self):
        self.assertEqual(parse_segment_spec('all', 5), [0, 1, 2, 3, 4])
        self.assertEqual(parse_segment_spec('3', 5), [3])
        self.assertEqual(parse_segment_spec('1-3', 5), [1, 2, 3])
        self.assertEqual(parse_segment_spec('0,2,4', 5), [0, 2, 4])
        self.assertEqual(parse_segment_spec('0,2-3', 5), [0, 2, 3])
        with self.assertRaises(ValueError):
            parse_segment_spec('7', 5)

    def test_write_and_reread_joint_kmz(self):
        coords = [(15.0, 37.0), (15.1, 37.01), (15.2, 37.02)]
        path = os.path.join(self.tmp.name, 'fault_joint.kmz')
        write_fault_kmz(path, coords, name='test_joint')
        segments = read_fault_kmz(path)
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0].coords), 3)
        self.assertAlmostEqual(segments[0].coords[1][0], 15.1, places=6)

    def test_joint_output_paths(self):
        kmz, txt = joint_output_paths('/data/PFS_fault_.kmz')
        self.assertEqual(os.path.basename(kmz), 'PFS_fault_joint.kmz')
        self.assertEqual(os.path.basename(txt), 'PFS_fault_joint.txt')
        kmz2, _ = joint_output_paths('/data/fault.kmz', outdir='/tmp/out')
        self.assertEqual(kmz2, '/tmp/out/fault_joint.kmz')

    def test_polyline_length(self):
        # ~111.19 km per degree latitude
        length = polyline_length_km([(15.0, 37.0), (15.0, 38.0)])
        self.assertAlmostEqual(length, 111.19, delta=0.5)


if __name__ == '__main__':
    unittest.main()
