#!/usr/bin/env python3
"""Cache freshness and lookup for companion .txt / image products."""

import glob
import os
import re


def parse_txt_header(path):
    """Return (column_header, bracket_info) from the first line of a data txt."""
    with open(path, encoding='utf-8') as handle:
        line = handle.readline().strip()
    if '[' not in line or not line.endswith(']'):
        raise ValueError(f'{path}: expected bracketed metadata in header')
    split = line.rindex('[')
    return line[:split].strip(), line[split + 1:-1].strip()


def cache_is_fresh(cache_path, *input_paths):
    """True when cache exists and no input file is newer than the cache."""
    if not cache_path or not os.path.isfile(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for path in input_paths:
        if path and os.path.isfile(path) and os.path.getmtime(path) > cache_mtime:
            return False
    return True


def figure_is_fresh(img_path, txt_path, update_mode, *input_paths):
    """MintPy-style skip: txt fresh vs inputs and image not older than txt."""
    if not update_mode:
        return False
    if not cache_is_fresh(txt_path, *input_paths):
        return False
    if not img_path or not os.path.isfile(img_path):
        return False
    return os.path.getmtime(img_path) >= os.path.getmtime(txt_path)


def map_period_bracket(inps, start, end):
    source = getattr(inps, 'fault_source_kmz', inps.fault_file)
    flip = 1 if inps.flip_fault else 0
    return (f'fault={os.path.basename(source)} '
            f'fault-segment={inps.fault_segment} fault-segment-by={inps.fault_segment_by} '
            f'flip-fault={flip} '
            f'along-step={inps.along_step} perp-width={inps.perp_width} '
            f'perp-offset={inps.perp_offset} '
            f'sample-method={inps.sample_method} reference-side={inps.reference_side} '
            f'period={start}:{end}')


def profile_period_bracket(inps, start, end):
    return (map_period_bracket(inps, start, end)
            + f' profile-length={inps.profile_length} layout={inps.plot_layout} '
            f'cloud-profiles={inps.cloud_profiles}')


def combined_figure_is_fresh(img_path, txt_paths, update_mode, *input_paths):
    """True when every txt is input-fresh and the image is not older than the newest txt."""
    if not update_mode:
        return False
    if not txt_paths:
        return False
    for txt_path in txt_paths:
        if not cache_is_fresh(txt_path, *input_paths):
            return False
    if not img_path or not os.path.isfile(img_path):
        return False
    newest_txt = max(os.path.getmtime(p) for p in txt_paths)
    return os.path.getmtime(img_path) >= newest_txt


def _basename_candidates(project, tag_string, label, start, end):
    tag_string = (tag_string or '').strip()
    if tag_string:
        yield f'{project}_{tag_string}_{label}_{start}_{end}'
    yield f'{project}_{label}_{start}_{end}'


def find_cached_txt(out_dir, project, tag_string, label, start, end, bracket_info):
    """Locate a txt whose header bracket matches ``bracket_info``."""
    for stem in _basename_candidates(project, tag_string, label, start, end):
        path = os.path.join(out_dir, f'{stem}.txt')
        if not os.path.isfile(path):
            continue
        try:
            _, bracket = parse_txt_header(path)
        except ValueError:
            continue
        if bracket == bracket_info:
            return path

    patterns = [os.path.join(out_dir, f'{project}_{label}_*.txt')]
    tag_string = (tag_string or '').strip()
    if tag_string:
        patterns.insert(0, os.path.join(out_dir, f'{project}_{tag_string}_{label}_*.txt'))
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                _, bracket = parse_txt_header(path)
            except ValueError:
                continue
            if bracket == bracket_info:
                return path
    return None


def discover_map_periods(out_dir, project, tag_string):
    """Return sorted list of (start, end) parsed from per-period map txt filenames."""
    periods = []
    patterns = [os.path.join(out_dir, f'{project}_map_*.txt')]
    tag_string = (tag_string or '').strip()
    if tag_string:
        patterns.insert(0, os.path.join(out_dir, f'{project}_{tag_string}_map_*.txt'))
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            stem = os.path.splitext(os.path.basename(path))[0]
            match = re.search(r'_map_(\d{8})_(\d{8})$', stem)
            if not match:
                continue
            key = (match.group(1), match.group(2))
            if key not in seen:
                seen.add(key)
                periods.append(key)
    return sorted(periods)
