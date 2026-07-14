#!/usr/bin/env python3
"""Read fault traces from KMZ/KML and write a homogenized (ordered) KMZ.

The homogenized fault contains one LineString per oriented segment (never merged).
Along-strike distance runs to each segment end, includes the gap to the next
segment start, then continues along the next segment.
"""

import os
import math
import re
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


def segment_label_number(name):
    """Trailing integer in a placemark name, e.g. PFS3 -> 3. None if absent."""
    if not name:
        return None
    m = re.search(r'(\d+)\s*$', str(name).strip())
    return int(m.group(1)) if m else None


def _parse_segment_spec_tokens(spec):
    """Parse --fault-segment string into ordered integer tokens (labels or indices)."""
    spec = (spec or 'all').strip().lower()
    if spec == 'all':
        return None
    tokens = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            lo, hi = chunk.split('-', 1)
            tokens.extend(range(int(lo), int(hi) + 1))
        else:
            tokens.append(int(chunk))
    # preserve order, drop duplicates
    seen = set()
    result = []
    for t in tokens:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result


def resolve_segment_spec(spec, segments, segment_by='auto'):
    """Resolve --fault-segment to KMZ segment indices in requested order.

    segment_by:
      auto  - use placemark label numbers (e.g. PFS3 -> 3) when every segment
              has a unique trailing number and every token matches a label;
              otherwise use 0-based KMZ read order indices.
      label - require label-number resolution (PFS-style names).
      index - 0-based index into segments as read from the KMZ file.
    """
    num_segments = len(segments)
    tokens = _parse_segment_spec_tokens(spec)
    if tokens is None:
        return list(range(num_segments)), 'all'

    labels = [segment_label_number(s.name) for s in segments]
    has_labels = all(l is not None for l in labels) and len(set(labels)) == len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)} if has_labels else {}

    use_label = segment_by == 'label'
    if segment_by == 'auto' and has_labels:
        if all(t in label_to_idx for t in tokens) and 0 not in tokens:
            use_label = True
        else:
            use_label = False

    if use_label:
        bad = [t for t in tokens if t not in label_to_idx]
        if bad:
            avail = sorted(label_to_idx)
            raise ValueError(
                f'--fault-segment label(s) {bad} not found; '
                f'available segment labels: {avail}')
        indices = [label_to_idx[t] for t in tokens]
        names = [segments[i].name for i in indices]
        mode = f'labels {tokens} -> {names}'
        return indices, mode

    if segment_by == 'label' and not has_labels:
        raise ValueError(
            '--fault-segment-by label requires unique trailing numbers in segment names')

    bad = [i for i in tokens if i < 0 or i >= num_segments]
    if bad:
        raise ValueError(f'--fault-segment indices {bad} out of range 0-{num_segments - 1}')
    names = [segments[i].name for i in tokens]
    mode = f'KMZ indices {tokens} -> {names}'
    return tokens, mode


def parse_segment_spec(spec, num_segments, segments=None, segment_by='auto'):
    """Parse --fault-segment value: 'all', '3', '2-8', '0,2,5', '0,3-5'.

    When segments is provided, numbers refer to placemark label suffixes (PFS3 -> 3)
    if segment_by is auto/label and names support it; otherwise 0-based KMZ indices.
    """
    if segments is not None:
        indices, _ = resolve_segment_spec(spec, segments, segment_by=segment_by)
        return indices
    tokens = _parse_segment_spec_tokens(spec)
    if tokens is None:
        return list(range(num_segments))
    bad = [i for i in tokens if i < 0 or i >= num_segments]
    if bad:
        raise ValueError(f'--fault-segment indices {bad} out of range 0-{num_segments - 1}')
    return tokens


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


def _coords_to_kml_coordinates(coords):
    return ' '.join(f'{lon:.10f},{lat:.10f},0' for lon, lat in coords)


def write_fault_kmz(path, coords, name='fault'):
    """Write a single-LineString KMZ from (lon, lat) coords."""
    coord_str = _coords_to_kml_coordinates(coords)
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


def write_fault_kmz_segments(path, segments, name='homogenized_fault'):
    """Write a KMZ containing one Placemark per (possibly disconnected) segment.

    segments: list of FaultSegment (coords are lon/lat tuples).
    """
    placemarks = []
    for i, seg in enumerate(segments):
        coord_str = _coords_to_kml_coordinates(seg.coords)
        seg_name = seg.name or f'segment_{i}'
        placemarks.append(f"""
    <Placemark>
      <name>{seg_name}</name>
      <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coord_str}</coordinates>
      </LineString>
    </Placemark>""")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    {''.join(placemarks)}
  </Document>
</kml>
"""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', kml)


CONNECT_GAP_KM_DEFAULT = 0.3


def _local_scale(lat0):
    km_per_deg_lat = math.pi / 180.0 * 6371.0
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(lat0))
    return km_per_deg_lat, km_per_deg_lon


def closest_point_on_polyline(lat, lon, coords):
    """Closest point on a (lon, lat) polyline to the query point.

    Returns (dist_km, conn_lon, conn_lat, edge_i, t) where the closest point lies
    on edge coords[edge_i] -> coords[edge_i + 1] at fraction t in [0, 1].
    """
    if len(coords) == 1:
        lon0, lat0 = coords[0]
        return haversine_km(lat, lon, lat0, lon0), lon0, lat0, 0, 0.0

    best = None
    for i in range(len(coords) - 1):
        lon0, lat0 = coords[i]
        lon1, lat1 = coords[i + 1]
        mid_lat = (lat0 + lat1) / 2.0
        km_lat, km_lon = _local_scale(mid_lat)
        x0 = (lon0 - lon) * km_lon
        y0 = (lat0 - lat) * km_lat
        x1 = (lon1 - lon) * km_lon
        y1 = (lat1 - lat) * km_lat
        dx, dy = x1 - x0, y1 - y0
        len2 = dx * dx + dy * dy
        if len2 == 0:
            t = 0.0
            cx, cy = x0, y0
        else:
            t = max(0.0, min(1.0, -(x0 * dx + y0 * dy) / len2))
            cx = x0 + t * dx
            cy = y0 + t * dy
        conn_lon = lon + cx / km_lon
        conn_lat = lat + cy / km_lat
        dist = haversine_km(lat, lon, conn_lat, conn_lon)
        if best is None or dist < best[0]:
            best = (dist, conn_lon, conn_lat, i, t)
    return best


def trim_polyline_from_connection(coords, conn_lon, conn_lat, edge_i, t, prev_lat, prev_lon):
    """Return polyline from the connection point toward the segment endpoint farther from prev."""
    coords = list(coords)
    conn = (conn_lon, conn_lat)
    if 1e-6 < t < 1.0 - 1e-6:
        coords = coords[:edge_i + 1] + [conn] + coords[edge_i + 1:]
        split_idx = edge_i + 1
    elif t >= 1.0 - 1e-6:
        split_idx = edge_i + 1
    else:
        split_idx = edge_i

    d_start = haversine_km(prev_lat, prev_lon, coords[0][1], coords[0][0])
    d_end = haversine_km(prev_lat, prev_lon, coords[-1][1], coords[-1][0])

    if d_end >= d_start:
        out = [conn]
        for j in range(split_idx + 1, len(coords)):
            if coords[j] != conn:
                out.append(coords[j])
        return out

    out = [conn]
    for j in range(split_idx - 1, -1, -1):
        if coords[j] != conn:
            out.append(coords[j])
    return out


def orient_segments_for_sequence(segments, connect_gap_km=CONNECT_GAP_KM_DEFAULT):
    """Orient segments in the given order.

    Orientation (forward vs reverse) is chosen to minimize the distance from the
    previous segment's end to the closest point anywhere on the next segment.

    If that closest distance is <= connect_gap_km (300 m), the next segment is
    trimmed to begin at that connection point (start may move) so adjacent traces
    do not overlap. Segments are never merged into one LineString.

    Along-strike distance always includes the gap from the previous segment's end
    to the (possibly trimmed) start of the next segment, then the segment length.
    gap_to_prev_km is that end-to-start distance; gap_lat/gap_lon are the closest
    point used for orientation and trimming.
    """
    oriented = []
    qc = []
    cum = 0.0
    prev_end_latlon = None

    for order_index, seg in enumerate(segments):
        coords_fwd = list(seg.coords)
        coords_rev = list(reversed(seg.coords))
        conn_lon = conn_lat = None
        gap_closest = 0.0

        if prev_end_latlon is None:
            chosen = coords_fwd
            reversed_flag = False
            gap = 0.0
        else:
            prev_lat, prev_lon = prev_end_latlon
            best = None
            for coords_cand, rev_flag in ((coords_fwd, False), (coords_rev, True)):
                dist, clon, clat, edge_i, t = closest_point_on_polyline(
                    prev_lat, prev_lon, coords_cand)
                if best is None or dist < best[0]:
                    best = (dist, coords_cand, rev_flag, clon, clat, edge_i, t)
            gap_closest, chosen, reversed_flag, conn_lon, conn_lat, edge_i, t = best

            if gap_closest <= connect_gap_km:
                chosen = trim_polyline_from_connection(
                    chosen, conn_lon, conn_lat, edge_i, t,
                    prev_lat, prev_lon)

            start_lon, start_lat = chosen[0]
            gap = haversine_km(prev_lat, prev_lon, start_lat, start_lon)

        length = polyline_length_km(chosen)
        start_lon, start_lat = chosen[0]
        end_lon, end_lat = chosen[-1]
        cum_end = cum + gap + length
        qc.append({
            'order_index': order_index,
            'segment_index': None,
            'name': seg.name,
            'reversed': reversed_flag,
            'trimmed': (order_index > 0 and gap_closest <= connect_gap_km),
            'gap_lat': conn_lat,
            'gap_lon': conn_lon,
            'start_lat': start_lat,
            'start_lon': start_lon,
            'end_lat': end_lat,
            'end_lon': end_lon,
            'length_km': length,
            'cum_start_km': cum,
            'cum_end_km': cum_end,
            'gap_to_prev_km': gap,
            'n_vertices': len(chosen),
        })
        oriented.append(FaultSegment(name=seg.name, coords=chosen))
        cum = cum_end
        prev_end_latlon = (end_lat, end_lon)

    return oriented, qc, cum


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
    """Return (joint_kmz_path, qc_txt_path) for a source KMZ."""
    kmz_path = os.path.abspath(kmz_path)
    if is_joint_kmz_path(kmz_path):
        return kmz_path, joint_qc_path(kmz_path)
    stem = os.path.splitext(os.path.basename(kmz_path))[0].rstrip('_')
    directory = outdir if outdir else os.path.dirname(kmz_path)
    return (os.path.join(directory, f'{stem}_joint.kmz'),
            os.path.join(directory, f'{stem}_joint_qc.txt'))


def is_joint_kmz_path(path):
    """True when ``path`` basename ends with ``_joint.kmz``."""
    return os.path.basename(path).lower().endswith('_joint.kmz')


def joint_qc_path(joint_kmz):
    """QC txt path paired with a ``*_joint.kmz`` file."""
    joint_kmz = os.path.abspath(joint_kmz)
    if joint_kmz.lower().endswith('_joint.kmz'):
        return joint_kmz[:-len('.kmz')] + '_qc.txt'
    return joint_kmz.replace('.kmz', '_joint_qc.txt')


def read_qc_source_kmz(qc_path):
    """Read ``# source_kmz`` from a joint QC file, or return None."""
    if not qc_path or not os.path.isfile(qc_path):
        return None
    with open(qc_path, encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('# source_kmz '):
                return line.split(None, 2)[2].strip()
    return None


def resolve_joint_paths(input_kmz, outdir=None):
    """Return (source_kmz, joint_kmz, joint_qc_path) for an input fault file."""
    input_kmz = os.path.abspath(input_kmz)
    if is_joint_kmz_path(input_kmz):
        joint_kmz = input_kmz
        qc_path = joint_qc_path(joint_kmz)
        source = read_qc_source_kmz(qc_path) or input_kmz
        return os.path.abspath(source), joint_kmz, qc_path
    source = input_kmz
    joint_kmz, qc_path = joint_output_paths(source, outdir)
    return source, joint_kmz, qc_path


def needs_joint_processing(num_selected_segments):
    """True when more than one segment is selected (orientation/trim required)."""
    return num_selected_segments > 1


def prepare_fault_geometry(inps):
    """Read, orient if needed, optionally write joint KMZ; return polylines + cache paths.

    Returns (polylines, source_kmz, cache_paths) where ``cache_paths`` lists KMZ
    files used for freshness checks on data txt and figures.
    """
    from plotdata.fault_transect.cache import cache_is_fresh

    segments = read_fault_kmz(inps.fault_file)
    indices, seg_mode = resolve_segment_spec(
        inps.fault_segment, segments, segment_by=inps.fault_segment_by)
    selected = [segments[i] for i in indices]
    print(f'Read {len(segments)} segment(s) from {inps.fault_file}; '
          f'using {len(selected)} ({seg_mode})')

    for seg, seg_idx in zip(selected, indices):
        seg.name = seg.name or f'segment_{seg_idx}'

    if inps.flip_fault:
        selected = [type(s)(name=s.name, coords=list(reversed(s.coords)))
                    for s in reversed(selected)]

    source_kmz, joint_kmz, joint_qc = resolve_joint_paths(inps.fault_file, inps.outdir)

    if not needs_joint_processing(len(selected)):
        coords = list(selected[0].coords) if selected else []
        return [coords] if coords else [], source_kmz, (source_kmz,)

    oriented, qc_rows, total = orient_segments_for_sequence(selected)
    for row, seg_idx in zip(qc_rows, indices):
        row['segment_index'] = int(seg_idx)
        if row['gap_to_prev_km'] > 0:
            print(f"segment jump: {row['gap_to_prev_km']:.1f} km -> {row['name']}")

    if inps.outdir:
        os.makedirs(inps.outdir, exist_ok=True)
    joint_dir = os.path.dirname(joint_kmz)
    if joint_dir:
        os.makedirs(joint_dir, exist_ok=True)

    write_joint = True
    if not getattr(inps, 'force', False) and os.path.isfile(joint_kmz) and cache_is_fresh(
            joint_kmz, source_kmz):
        print(f'Using existing joint fault KMZ: {joint_kmz}')
        write_joint = False

    if write_joint:
        stem = os.path.splitext(os.path.basename(joint_kmz))[0]
        write_fault_kmz_segments(joint_kmz, oriented, name=stem)
        write_segment_qc(joint_qc, source_kmz, qc_rows, total)
        print(f'Homogenized fault KMZ: {joint_kmz}')
        print(f'QC report:            {joint_qc}')
        print(f'Total fault length:   {total:.3f} km ({len(oriented)} segments)')

    return homogenized_polylines(oriented), source_kmz, (source_kmz, joint_kmz)


def homogenized_polylines(oriented_segments, qc_rows=None):
    """Return one polyline per oriented segment (never merged).

    qc_rows is accepted for API compatibility but is not used.
    """
    return [list(seg.coords) for seg in oriented_segments]


def write_segment_qc(path, source_kmz, qc_rows, total_length_km):
    """Write a QC table for the homogenized (ordered) segments.

    First line is the column names / explanation, as requested.
    gap_to_prev_km is the end-to-start distance to the (possibly trimmed) next
    segment and is included in cum_start_km / cum_end_km. trimmed=1 when the
    segment start was moved to the closest connection point (gap <= 300 m).
    gap_lat/gap_lon are that closest point.
    """
    header = (
        'order_index segment_index name reversed trimmed n_vertices length_km '
        'cum_start_km cum_end_km gap_to_prev_km gap_lat gap_lon start_lat start_lon end_lat end_lon '
        '[gap_to_prev_km = prev end to next start, included in along-km; trimmed=1 if start moved at <=300 m gap]'
    )
    lines = [header]
    for row in qc_rows:
        gap_lat = row.get('gap_lat')
        gap_lon = row.get('gap_lon')
        gap_lat_s = f'{gap_lat:.6f}' if gap_lat is not None else 'nan'
        gap_lon_s = f'{gap_lon:.6f}' if gap_lon is not None else 'nan'
        lines.append(
            f"{int(row.get('order_index', -1) if row.get('order_index', -1) is not None else -1):d} "
            f"{int(row.get('segment_index', -1) if row.get('segment_index', -1) is not None else -1):d} "
            f"{row.get('name', '')} "
            f"{1 if row.get('reversed') else 0:d} "
            f"{1 if row.get('trimmed') else 0:d} "
            f"{int(row.get('n_vertices', 0) if row.get('n_vertices', 0) is not None else 0):d} "
            f"{row.get('length_km', 0.0):.3f} "
            f"{row.get('cum_start_km', 0.0):.3f} "
            f"{row.get('cum_end_km', 0.0):.3f} "
            f"{row.get('gap_to_prev_km', 0.0):.1f} "
            f"{gap_lat_s} "
            f"{gap_lon_s} "
            f"{row.get('start_lat', float('nan')):.6f} "
            f"{row.get('start_lon', float('nan')):.6f} "
            f"{row.get('end_lat', float('nan')):.6f} "
            f"{row.get('end_lon', float('nan')):.6f}"
        )
    lines.append(f'# total_length_km {total_length_km:.3f}')
    lines.append(f'# source_kmz {source_kmz}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
