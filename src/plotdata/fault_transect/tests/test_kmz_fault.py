#!/usr/bin/env python3
"""Tests for KMZ reading, segment joining, and joint KMZ writing."""

import os
import tempfile
import unittest
import zipfile

from plotdata.fault_transect.kmz_fault import (
    read_fault_kmz, join_segments, parse_segment_spec, resolve_segment_spec,
    write_fault_kmz, write_fault_kmz_segments, joint_output_paths, polyline_length_km,
    orient_segments_for_sequence, write_segment_qc, closest_point_on_polyline,
    homogenized_polylines, FaultSegment)


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

    def test_resolve_segment_spec_by_label(self):
        segments = [
            FaultSegment('PFS1', [(0, 0), (1, 0)]),
            FaultSegment('PFS10', [(2, 0), (3, 0)]),
            FaultSegment('PFS11', [(4, 0), (5, 0)]),
            FaultSegment('PFS12', [(6, 0), (7, 0)]),
            FaultSegment('PFS2', [(8, 0), (9, 0)]),
            FaultSegment('PFS3', [(10, 0), (11, 0)]),
            FaultSegment('PFS4', [(12, 0), (13, 0)]),
            FaultSegment('PFS5', [(14, 0), (15, 0)]),
        ]
        indices, mode = resolve_segment_spec('1,2,4,5', segments)
        self.assertEqual(indices, [0, 4, 6, 7])  # PFS1, PFS2, PFS4, PFS5 — not PFS3
        self.assertIn('PFS1', mode)
        self.assertNotIn('PFS3', mode.split('->')[-1])
        indices_idx, _ = resolve_segment_spec('1,2,4', segments, segment_by='index')
        self.assertEqual(indices_idx, [1, 2, 4])  # PFS10, PFS11, PFS2 by KMZ index

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
        self.assertEqual(os.path.basename(txt), 'PFS_fault_joint_qc.txt')
        kmz2, _ = joint_output_paths('/data/fault.kmz', outdir='/tmp/out')
        self.assertEqual(kmz2, '/tmp/out/fault_joint.kmz')

    def test_write_segments_kmz_and_qc(self):
        path = os.path.join(self.tmp.name, 'fault.kmz')
        make_kmz(path, [('A', [(15.0, 37.0), (15.1, 37.0)]),
                        ('B', [(15.3, 37.0), (15.4, 37.0)])])
        segments = read_fault_kmz(path)
        oriented, qc, total = orient_segments_for_sequence(segments)
        out_kmz = os.path.join(self.tmp.name, 'out_joint.kmz')
        out_qc = os.path.join(self.tmp.name, 'out_joint_qc.txt')
        write_fault_kmz_segments(out_kmz, oriented, name='out_joint')
        write_segment_qc(out_qc, path, qc, total)
        reread = read_fault_kmz(out_kmz)
        self.assertEqual(len(reread), 2)
        with open(out_qc, encoding='utf-8') as f:
            first = f.readline()
        self.assertIn('order_index', first)
        self.assertIn('gap_to_prev_km', first)

    def test_orient_includes_gaps_in_cumulative_km(self):
        segments = [
            FaultSegment('A', [(15.0, 37.0), (15.1, 37.0)]),
            FaultSegment('B', [(15.3, 37.0), (15.4, 37.0)]),
        ]
        oriented, qc, total = orient_segments_for_sequence(segments)
        self.assertAlmostEqual(qc[0]['cum_start_km'], 0.0)
        self.assertAlmostEqual(qc[1]['cum_start_km'], qc[0]['cum_end_km'])
        self.assertGreater(qc[1]['gap_to_prev_km'], 1.0)
        self.assertAlmostEqual(
            total, qc[1]['cum_end_km'])
        self.assertEqual(len(homogenized_polylines(oriented)), 2)

    def test_polyline_length(self):
        # ~111.19 km per degree latitude
        length = polyline_length_km([(15.0, 37.0), (15.0, 38.0)])
        self.assertAlmostEqual(length, 111.19, delta=0.5)

    def test_trim_at_close_gap_no_overlap(self):
        """When gap <= 300 m, next segment is trimmed at the connection point."""
        prev_end = (15.0, 37.0)
        seg_coords = [(15.1, 37.0), (15.0, 37.0026), (15.2, 37.0)]
        _, conn_lon, conn_lat, _, _ = closest_point_on_polyline(
            prev_end[1], prev_end[0], seg_coords)

        segments = [
            FaultSegment('A', [(14.9, 37.0), prev_end]),
            FaultSegment('B', seg_coords),
        ]
        oriented, qc, _ = orient_segments_for_sequence(segments, connect_gap_km=0.3)
        self.assertTrue(qc[1]['trimmed'])
        self.assertLess(qc[1]['gap_to_prev_km'], 0.35)
        self.assertAlmostEqual(oriented[1].coords[0][0], conn_lon, places=3)
        self.assertAlmostEqual(oriented[1].coords[0][1], conn_lat, places=3)
        self.assertEqual(len(homogenized_polylines(oriented)), 2)

    def test_no_trim_when_gap_large(self):
        prev_end = (15.0, 37.0)
        segments = [
            FaultSegment('A', [(14.9, 37.0), prev_end]),
            FaultSegment('B', [(15.1, 37.0), (15.2, 37.0)]),
        ]
        oriented, qc, _ = orient_segments_for_sequence(segments, connect_gap_km=0.3)
        self.assertFalse(qc[1]['trimmed'])
        self.assertAlmostEqual(oriented[1].coords[0][0], 15.1, places=3)

if __name__ == '__main__':
    unittest.main()
