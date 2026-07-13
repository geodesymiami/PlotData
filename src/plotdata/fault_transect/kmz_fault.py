#!/usr/bin/env python3
"""Read fault traces from KMZ/KML, join multi-segment faults, write joint KMZ."""

import os
import math
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


@dataclass
class FaultSegment:
    name: str
    coords: list          # list of (lon, lat) tuples


@dataclass
class JoinReport:
    order: list = field(default_factory=list)   # dicts: name, index, reversed, gap_km
    total_length_km: float = 0.0
    warnings: list = field(default_factory=list)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def polyline_length_km(coords):
    """Length of a (lon, lat) polyline in km."""
    total = 0.0
    for (lon0, lat0), (lon1, lat1) in zip(coords[:-1], coords[1:]):
        total += haversine_km(lat0, lon0, lat1, lon1)
    return total


def _kml_root(path):
    if path.lower().endswith('.kmz'):
        with zipfile.ZipFile(path) as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith('.kml')]
            if not kml_names:
                raise ValueError(f'No .kml found inside {path}')
            with zf.open(kml_names[0]) as f:
                return ET.parse(f).getroot()
    return ET.parse(path).getroot()


def read_fault_kmz(path):
    """Read all LineString segments from a KMZ or KML file.

    Returns a list of FaultSegment in file order.
    """
    root = _kml_root(path)
    segments = []
    for placemark in root.findall('.//{*}Placemark'):
        name_el = placemark.find('{*}name')
        name = name_el.text.strip() if name_el is not None and name_el.text else ''
        for linestring in placemark.findall('.//{*}LineString'):
            coords_el = linestring.find('{*}coordinates')
            if coords_el is None or not coords_el.text:
                continue
            coords = []
            for token in coords_el.text.split():
                parts = token.split(',')
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
            if len(coords) >= 2:
                seg_name = name or f'segment_{len(segments)}'
                segments.append(FaultSegment(name=seg_name, coords=coords))
    if not segments:
        raise ValueError(f'No LineString segments found in {path}')
    return segments


def parse_segment_spec(spec, num_segments):
    """Parse --fault-segment value: 'all', '3', '2-8', '0,2,5', '0,3-5'."""
    spec = (spec or 'all').strip().lower()
    if spec == 'all':
        return list(range(num_segments))
    indices = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            lo, hi = chunk.split('-', 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(chunk))
    bad = [i for i in indices if i < 0 or i >= num_segments]
    if bad:
        raise ValueError(f'--fault-segment indices {bad} out of range 0-{num_segments - 1}')
    # preserve order, drop duplicates
    seen = set()
    result = []
    for i in indices:
        if i not in seen:
            result.append(i)
            seen.add(i)
    return result


def _endpoint(seg, which):
    """Return (lat, lon) of segment start (0) or end (-1)."""
    lon, lat = seg.coords[0] if which == 0 else seg.coords[-1]
    return lat, lon


def join_segments(segments, gap_warn_km=1.0):
    """Join unordered, possibly reversed segments into one polyline.

    Greedy endpoint-proximity chaining: the seed is the endpoint whose nearest
    endpoint on any other segment is farthest away (a fault terminus); then the
    unused segment with the closest endpoint to the running chain end is
    appended (reversing it if needed) until all segments are used.

    Returns (joined_coords, JoinReport) with joined_coords as (lon, lat) list.
    """
    if len(segments) == 1:
        report = JoinReport(order=[{'index': 0, 'name': segments[0].name, 'reversed': False, 'gap_km': 0.0}],
                            total_length_km=polyline_length_km(segments[0].coords))
        return list(segments[0].coords), report

    # Find the terminus endpoint: max distance to nearest endpoint of another segment
    best = None   # (distance, seg_idx, which)
    for i, seg in enumerate(segments):
        for which in (0, -1):
            lat, lon = _endpoint(seg, which)
            nearest = min(
                haversine_km(lat, lon, *_endpoint(other, w))
                for j, other in enumerate(segments) if j != i
                for w in (0, -1)
            )
            if best is None or nearest > best[0]:
                best = (nearest, i, which)

    _, seed_idx, seed_which = best
    seed = segments[seed_idx]
    seed_coords = list(seed.coords) if seed_which == 0 else list(reversed(seed.coords))

    report = JoinReport()
    report.order.append({'index': seed_idx, 'name': seed.name,
                         'reversed': seed_which == -1, 'gap_km': 0.0})
    joined = seed_coords
    used = {seed_idx}

    while len(used) < len(segments):
        end_lon, end_lat = joined[-1]
        candidate = None    # (gap_km, seg_idx, which)
        for i, seg in enumerate(segments):
            if i in used:
                continue
            for which in (0, -1):
                lat, lon = _endpoint(seg, which)
                gap = haversine_km(end_lat, end_lon, lat, lon)
                if candidate is None or gap < candidate[0]:
                    candidate = (gap, i, which)
        gap, idx, which = candidate
        seg = segments[idx]
        coords = list(seg.coords) if which == 0 else list(reversed(seg.coords))
        if gap > gap_warn_km:
            report.warnings.append(
                f'Gap of {gap:.2f} km between "{report.order[-1]["name"]}" and "{seg.name}"')
        # avoid duplicating a shared vertex
        if gap < 1e-6:
            coords = coords[1:]
        joined.extend(coords)
        report.order.append({'index': idx, 'name': seg.name, 'reversed': which == -1, 'gap_km': gap})
        used.add(idx)

    report.total_length_km = polyline_length_km(joined)
    return joined, report


def write_fault_kmz(path, coords, name='joint_fault'):
    """Write a single-LineString KMZ from (lon, lat) coords."""
    coord_str = ' '.join(f'{lon:.10f},{lat:.10f},0' for lon, lat in coords)
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <Placemark>
      <name>{name}</name>
      <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coord_str}</coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', kml)


def write_join_report(path, report, source_kmz, joined_coords):
    """Write a human-readable txt summary of the segment join."""
    lines = [
        f'Joint fault created from: {source_kmz}',
        f'Segments joined: {len(report.order)}',
        f'Total length: {report.total_length_km:.3f} km',
        f'Vertices: {len(joined_coords)}',
        f'Start (lat, lon): {joined_coords[0][1]:.6f}, {joined_coords[0][0]:.6f}',
        f'End   (lat, lon): {joined_coords[-1][1]:.6f}, {joined_coords[-1][0]:.6f}',
        '',
        'Join order (segment  reversed  gap_to_previous_km):',
    ]
    for entry in report.order:
        lines.append(f'  {entry["index"]:3d}  {entry["name"]:<20s}  '
                     f'{"yes" if entry["reversed"] else "no ":<3s}  {entry["gap_km"]:8.3f}')
    if report.warnings:
        lines.append('')
        lines.append('Warnings:')
        lines.extend(f'  {w}' for w in report.warnings)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def joint_output_paths(kmz_path, outdir=None):
    """Return (joint_kmz_path, joint_txt_path) for a source KMZ."""
    stem = os.path.splitext(os.path.basename(kmz_path))[0].rstrip('_')
    directory = outdir if outdir else os.path.dirname(os.path.abspath(kmz_path))
    return (os.path.join(directory, f'{stem}_joint.kmz'),
            os.path.join(directory, f'{stem}_joint.txt'))
