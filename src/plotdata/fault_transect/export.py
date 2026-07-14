#!/usr/bin/env python3
"""Data txt export: every figure gets a companion .txt with the same basename."""

import os

import numpy as np

from plotdata.fault_transect.offset import OffsetSeries
from plotdata.fault_transect.profiles import Profile, ProfileBundle
from plotdata.fault_transect.timeseries import TimeseriesBundle


def _parse_float(token):
    if token in ('nan', 'NaN'):
        return float('nan')
    return float(token)


def _write_txt(path, header, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for row in rows:
            f.write(' '.join(row) + '\n')
    print(f'Data saved to {path}')


def _fmt(value, precision=6):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'nan'
    return f'{value:.{precision}f}'


def write_offset_txt(path, series, bracket_info):
    """Map-plot data: one row per sampling point along the fault."""
    header = f'along_km lat lon left_val right_val offset [{bracket_info}]'
    rows = []
    for i in range(len(series.along_km)):
        rows.append([
            _fmt(series.along_km[i], 3),
            _fmt(series.lat[i], 8),
            _fmt(series.lon[i], 8),
            _fmt(series.left_val[i]),
            _fmt(series.right_val[i]),
            _fmt(series.offset[i]),
        ])
    _write_txt(path, header, rows)


def write_profiles_txt(path, bundle, main_indices, cloud_profiles, bracket_info):
    """Profile data: all profiles of a figure; is_main flags the black curves."""
    header = f'profile_index along_km lat lon across_km value is_main [{bracket_info}]'
    main_set = set(main_indices)
    shown = set()
    for idx in main_indices:
        shown.add(idx)
        for delta in range(-cloud_profiles, cloud_profiles + 1):
            shown.add(idx + delta)

    rows = []
    for profile in bundle.profiles:
        if profile.index not in shown:
            continue
        is_main = '1' if profile.index in main_set else '0'
        for k in range(len(profile.value)):
            rows.append([
                str(profile.index),
                _fmt(profile.along_km, 3),
                _fmt(profile.lat[k], 8),
                _fmt(profile.lon[k], 8),
                _fmt(profile.across_km[k], 4),
                _fmt(profile.value[k]),
                is_main,
            ])
    _write_txt(path, header, rows)


def read_offset_txt(path):
    """Load map offset series written by :func:`write_offset_txt`."""
    from plotdata.fault_transect.cache import parse_txt_header

    header, bracket = parse_txt_header(path)
    expected = 'along_km lat lon left_val right_val offset'
    if not header.startswith(expected):
        raise ValueError(f'{path}: unexpected offset header')
    series = OffsetSeries()
    ref_side = 'left'
    for token in bracket.split():
        if token.startswith('reference-side='):
            ref_side = token.split('=', 1)[1]
    series.reference_side = ref_side
    with open(path, encoding='utf-8') as handle:
        handle.readline()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            series.along_km.append(_parse_float(parts[0]))
            series.lat.append(_parse_float(parts[1]))
            series.lon.append(_parse_float(parts[2]))
            series.left_val.append(_parse_float(parts[3]))
            series.right_val.append(_parse_float(parts[4]))
            series.offset.append(_parse_float(parts[5]))
    return series, bracket


def read_profiles_txt(path):
    """Load profile bundle and main indices written by :func:`write_profiles_txt`."""
    from plotdata.fault_transect.cache import parse_txt_header

    header, bracket = parse_txt_header(path)
    expected = 'profile_index along_km lat lon across_km value is_main'
    if not header.startswith(expected):
        raise ValueError(f'{path}: unexpected profile header')
    rows_by_index = {}
    main_indices = set()
    profile_length = 0.0
    with open(path, encoding='utf-8') as handle:
        handle.readline()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx = int(parts[0])
            along_km = _parse_float(parts[1])
            lat = _parse_float(parts[2])
            lon = _parse_float(parts[3])
            across = _parse_float(parts[4])
            value = _parse_float(parts[5])
            is_main = parts[6] == '1'
            rows_by_index.setdefault(idx, []).append(
                (along_km, lat, lon, across, value, is_main))
            if is_main:
                main_indices.add(idx)
            profile_length = max(profile_length, abs(across) * 2.0)

    profiles = []
    for idx in sorted(rows_by_index):
        rows = rows_by_index[idx]
        along_km = rows[0][0]
        center_lat = rows[len(rows) // 2][1]
        center_lon = rows[len(rows) // 2][2]
        across = np.asarray([row[3] for row in rows], dtype=float)
        lats = np.asarray([row[1] for row in rows], dtype=float)
        lons = np.asarray([row[2] for row in rows], dtype=float)
        values = np.asarray([row[4] for row in rows], dtype=float)
        profiles.append(Profile(index=idx, along_km=along_km,
                                center_lat=center_lat, center_lon=center_lon,
                                across_km=across, lat=lats, lon=lons, value=values))

    bundle = ProfileBundle(profiles=profiles, profile_length_km=profile_length, unit='')
    return bundle, sorted(main_indices), bracket


def write_timeseries_txt(path, bundle, bracket_info):
    """Timeseries data: one row per date per along-fault location."""
    header = (f'location_index along_km lat lon date offset '
              f'[{bracket_info}]')
    rows = []
    for loc in bundle.locations:
        for date, value in zip(bundle.dates, loc.offset):
            rows.append([
                str(loc.point_index),
                _fmt(loc.along_km, 3),
                _fmt(loc.lat, 8),
                _fmt(loc.lon, 8),
                str(date),
                _fmt(value),
            ])
    _write_txt(path, header, rows)


def read_timeseries_txt(path):
    """Load timeseries bundle written by :func:`write_timeseries_txt`."""
    from plotdata.fault_transect.cache import parse_txt_header
    from plotdata.fault_transect.timeseries import LocationTimeseries, TimeseriesBundle

    header, bracket = parse_txt_header(path)
    expected = 'location_index along_km lat lon date offset'
    if not header.startswith(expected):
        raise ValueError(f'{path}: unexpected timeseries header')
    ref_side = 'left'
    for token in bracket.split():
        if token.startswith('reference-side='):
            ref_side = token.split('=', 1)[1]
    by_index = {}
    dates = []
    seen_dates = set()
    with open(path, encoding='utf-8') as handle:
        handle.readline()
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx = int(parts[0])
            along_km = _parse_float(parts[1])
            lat = _parse_float(parts[2])
            lon = _parse_float(parts[3])
            date = parts[4]
            value = _parse_float(parts[5])
            if date not in seen_dates:
                dates.append(date)
                seen_dates.add(date)
            by_index.setdefault(idx, {'along_km': along_km, 'lat': lat, 'lon': lon,
                                      'values': []})
            by_index[idx]['values'].append(value)

    locations = []
    for idx in sorted(by_index):
        entry = by_index[idx]
        locations.append(LocationTimeseries(
            point_index=idx,
            along_km=entry['along_km'],
            lat=entry['lat'],
            lon=entry['lon'],
            offset=np.asarray(entry['values'], dtype=float)))
    bundle = TimeseriesBundle(dates=dates, locations=locations,
                              unit='cm', reference_side=ref_side)
    return bundle, bracket
