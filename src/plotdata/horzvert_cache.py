#!/usr/bin/env python3
"""Cache freshness and parameter sidecars for horz/vert timeseries products."""

import glob
import os
import re
from datetime import datetime

from plotdata.fault_transect.cache import cache_is_fresh

# mintpy / miaplpy, optionally with YYYYMM or YYYYMMDD span (keep full dirname).
_PROC_DIR_RE = re.compile(r'^(mintpy|miaplpy)(?:_(\d{6}|\d{8})_(\d{6}|\d{8}))?$')


def hvparams_path(vert_path):
    """Sidecar path storing processing-parameter fingerprint for a vert HE5."""
    return f'{vert_path}.hvparams'


def _repr_value(value):
    if value is None:
        return 'None'
    if isinstance(value, (tuple, list)):
        return '[' + ','.join(_repr_value(v) for v in value) + ']'
    return str(value)


def build_hv_fingerprint(inps, input_basenames):
    """Build a pipe-joined fingerprint of horz/vert processing options."""
    ref_lalo = getattr(inps, 'ref_lalo', None)
    if ref_lalo is not None and not isinstance(ref_lalo, (list, tuple)):
        ref_lalo = [ref_lalo]
    mask_vmin = getattr(inps, 'mask_vmin', None) or []
    period = getattr(inps, 'period', None) or []
    start_date = getattr(inps, 'start_date', None) or []
    stop_date = getattr(inps, 'stop_date', None) or []
    exclude_dates = getattr(inps, 'exclude_dates', None) or []
    geom_file = getattr(inps, 'geom_file', None) or []
    sorted_inputs = sorted(input_basenames)

    parts = [
        f'ref-lalo={_repr_value(ref_lalo)}',
        f'mask-thresh={_repr_value(mask_vmin)}',
        f'intervals={getattr(inps, "interval_index", None)}',
        f'period={_repr_value(period)}',
        f'start-date={_repr_value(start_date)}',
        f'end-date={_repr_value(stop_date)}',
        f'horz-az-angle={getattr(inps, "horz_az_angle", None)}',
        f'window-size={getattr(inps, "window_size", None)}',
        f'exclude-dates={_repr_value(exclude_dates)}',
        f'lat-step={getattr(inps, "lat_step", None)}',
        f'no-swap={int(bool(getattr(inps, "no_swap", False)))}',
        f'geom-file={_repr_value([os.path.basename(p) for p in geom_file])}',
        f'inputs={_repr_value(sorted_inputs)}',
    ]
    return '|'.join(parts)


def read_hvparams(vert_path):
    """Return stored fingerprint from sidecar, or None if missing/unreadable."""
    path = hvparams_path(vert_path)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as handle:
            return handle.read().strip()
    except OSError:
        return None


def write_hvparams(vert_path, fingerprint):
    """Write processing-parameter fingerprint sidecar next to vert HE5."""
    path = hvparams_path(vert_path)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(fingerprint)


def hvparams_matches(vert_path, fingerprint):
    """True when sidecar exists and matches the expected fingerprint."""
    stored = read_hvparams(vert_path)
    return stored is not None and stored == fingerprint


def predict_geo_input_path(file_path):
    """Return geocoded HE5 path that horzvert would use for this input."""
    if not file_path:
        return file_path
    file_path = os.path.abspath(file_path)
    file_dir, file_base = os.path.split(file_path)
    if file_base.startswith('geo_'):
        return file_path
    return os.path.join(file_dir, f'geo_{file_base}')


def infer_output_subdir(file_path):
    """Return mintpy/miaplpy path component (keep dated suffix when present)."""
    if not file_path:
        return None
    for element in os.path.normpath(file_path).split(os.sep):
        if _PROC_DIR_RE.match(element):
            return element
    return None


def processing_dir_span(name):
    """Comparable period length for a processing-method dir (months or days).

    Bare mintpy/miaplpy (no dates) → 0 so dated names win when picking the longer span.
    """
    match = _PROC_DIR_RE.match(name or '')
    if not match or not match.group(2):
        return 0
    start, end = match.group(2), match.group(3)
    if len(start) == 6:
        start_m = int(start[:4]) * 12 + int(start[4:6])
        end_m = int(end[:4]) * 12 + int(end[4:6])
        return end_m - start_m
    start_d = datetime.strptime(start, '%Y%m%d')
    end_d = datetime.strptime(end, '%Y%m%d')
    return (end_d - start_d).days


def longest_output_subdir(file_paths):
    """Among input paths, keep the mintpy/miaplpy dir covering the longer period."""
    dirs = []
    for path in file_paths or []:
        name = infer_output_subdir(path)
        if name:
            dirs.append(name)
    if not dirs:
        return None
    return max(dirs, key=processing_dir_span)


def locate_hv_outputs(output_dir):
    """Return newest (vert_path, horz_path) under output_dir, or (None, None)."""
    if not output_dir or not os.path.isdir(output_dir):
        return None, None

    vert_candidates = sorted(
        glob.glob(os.path.join(output_dir, '*vert*.he5')),
        key=os.path.getmtime,
        reverse=True,
    )
    horz_candidates = sorted(
        glob.glob(os.path.join(output_dir, '*horz*.he5')),
        key=os.path.getmtime,
        reverse=True,
    )
    vert_path = vert_candidates[0] if vert_candidates else None
    horz_path = horz_candidates[0] if horz_candidates else None
    return vert_path, horz_path


def hv_cache_hit(vert_path, horz_path, geo1, geo2, fingerprint):
    """True when vert/horz exist, are fresh vs geo inputs, and params match."""
    if not vert_path or not horz_path:
        return False
    if not os.path.isfile(vert_path) or not os.path.isfile(horz_path):
        return False
    if not hvparams_matches(vert_path, fingerprint):
        return False
    if not cache_is_fresh(vert_path, geo1, geo2):
        return False
    if not cache_is_fresh(horz_path, geo1, geo2):
        return False
    return True


def should_recompute_hv(inps, vert_path, horz_path, geo1, geo2, fingerprint):
    """True when horz/vert products should be (re)computed."""
    if getattr(inps, 'force', False) or getattr(inps, 'overwrite', False):
        return True
    return not hv_cache_hit(vert_path, horz_path, geo1, geo2, fingerprint)


def geometry_cache_fresh(geometry_file, eos_file):
    """True when geometry file exists and is newer than the source HE5."""
    return cache_is_fresh(geometry_file, eos_file)


def clean_hv_products(output_dir):
    """Remove cached horz/vert HE5 products and sidecars under output_dir."""
    if not output_dir or not os.path.isdir(output_dir):
        return 0
    removed = 0
    patterns = ('*vert*.he5', '*horz*.he5', '*.hvparams')
    for pattern in patterns:
        for path in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed
