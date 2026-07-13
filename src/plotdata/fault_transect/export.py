#!/usr/bin/env python3
"""Data txt export: every figure gets a companion .txt with the same basename."""

import os

import numpy as np


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
