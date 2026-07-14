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


def txt_cache_hit(txt_path, eos_file, bracket_info, *fault_paths):
    """True when txt exists, is newer than data inputs, and bracket metadata matches."""
    if not txt_path or not os.path.isfile(txt_path):
        return False
    if not cache_is_fresh(txt_path, eos_file, *fault_paths):
        return False
    try:
        _, bracket = parse_txt_header(txt_path)
    except ValueError:
        return False
    return bracket == bracket_info


def should_write_txt(inps, txt_path, eos_file, bracket_info, *fault_paths):
    """True when a data txt should be (re)written."""
    if inps.plots_only:
        return False
    if inps.force:
        return True
    return not txt_cache_hit(txt_path, eos_file, bracket_info, *fault_paths)


def figure_style_path(img_path):
    return f'{img_path}.style'


def _style_repr(value):
    if value is None:
        return 'None'
    if isinstance(value, (tuple, list)):
        return '[' + ','.join(_style_repr(v) for v in value) + ']'
    return str(value)


def map_figure_style_key(inps, color_lims, n_periods, dataset_label=''):
    """Fingerprint of map figure styling (not data-processing options)."""
    title_off = inps.title_offset if inps.title_offset is not None else (0.0, 0.0)
    return '|'.join([
        f'cmap={inps.colormap}',
        f'cmap_vlist={_style_repr(inps.cmap_vlist)}',
        f'vlim={_style_repr(inps.vlim)}',
        f'auto_cs={inps.auto_colorscale}',
        f'color_lims={_style_repr(color_lims)}',
        f'font={inps.font_size}',
        f'dpi={inps.dpi}',
        f'save={inps.save}',
        f'title_pos={inps.title_position}',
        f'title_off={_style_repr(title_off)}',
        f'map_stack={inps.map_stack_offset}',
        f'map_stack_axis={inps.map_stack_axis}',
        f'n_periods={n_periods}',
        f'tag={inps.tag_string}',
        f'dataset={dataset_label or ""}',
        f'scatter={inps.scatter_size}',
    ])


def profile_figure_style_key(inps, plot_layout, n_periods, dataset_label=''):
    title_off = inps.title_offset if inps.title_offset is not None else (0.0, 0.0)
    return '|'.join([
        f'layout={plot_layout}',
        f'cmap={inps.colormap}',
        f'font={inps.font_size}',
        f'dpi={inps.dpi}',
        f'save={inps.save}',
        f'stack={inps.stack_offset}',
        f'subplot_cols={inps.subplot_cols}',
        f'profile_lines={inps.profile_lines}',
        f'cloud={inps.cloud_profiles}',
        f'title_pos={inps.title_position}',
        f'title_off={_style_repr(title_off)}',
        f'n_periods={n_periods}',
        f'tag={inps.tag_string}',
        f'plot_step_factor={inps.plot_step_factor}',
        f'dataset={dataset_label or ""}',
        f'scatter={inps.scatter_size}',
    ])


def write_figure_style(img_path, style_key):
    with open(figure_style_path(img_path), 'w', encoding='utf-8') as handle:
        handle.write(style_key)


def figure_style_matches(img_path, style_key):
    style_path = figure_style_path(img_path)
    if not os.path.isfile(style_path):
        return False
    with open(style_path, encoding='utf-8') as handle:
        return handle.read().strip() == style_key


def figure_is_fresh(img_path, txt_path, style_key, *data_input_paths):
    """True when txt is input-fresh, image is not older than txt, and plot style unchanged."""
    if not cache_is_fresh(txt_path, *data_input_paths):
        return False
    if not img_path or not os.path.isfile(img_path):
        return False
    if os.path.getmtime(img_path) < os.path.getmtime(txt_path):
        return False
    return figure_style_matches(img_path, style_key)


def combined_figure_is_fresh(img_path, txt_paths, style_key, *data_input_paths):
    """True when every txt is input-fresh and the image matches style and txt ages."""
    if not txt_paths:
        return False
    for txt_path in txt_paths:
        if not cache_is_fresh(txt_path, *data_input_paths):
            return False
    if not img_path or not os.path.isfile(img_path):
        return False
    newest_txt = max(os.path.getmtime(p) for p in txt_paths)
    if os.path.getmtime(img_path) < newest_txt:
        return False
    return figure_style_matches(img_path, style_key)


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
    along_end = inps.along_end if inps.along_end is not None else 'end'
    return (map_period_bracket(inps, start, end)
            + f' along-start={inps.along_start} along-end={along_end}'
            f' profile-length={inps.profile_length} layout={inps.plot_layout} '
            f'cloud-profiles={inps.cloud_profiles} plot-step-factor={inps.plot_step_factor}')


def timeseries_period_bracket(inps, periods, span_start, span_end):
    """Bracket metadata for a full-span timeseries product."""
    import re
    period_list = ','.join(f'{s}:{e}' for s, e in periods)
    base = map_period_bracket(inps, span_start, span_end)
    base = re.sub(r'period=\S+', f'periods={period_list}', base)
    along_end = inps.along_end if inps.along_end is not None else 'end'
    return (base + f' along-start={inps.along_start} along-end={along_end}'
            f' plot-step-factor={inps.plot_step_factor}')


def timeseries_figure_style_key(inps, dataset_label=''):
    title_off = inps.title_offset if inps.title_offset is not None else (0.0, 0.0)
    return '|'.join([
        f'cmap={inps.colormap}',
        f'font={inps.font_size}',
        f'dpi={inps.dpi}',
        f'save={inps.save}',
        f'stack={inps.stack_offset}',
        f'title_pos={inps.title_position}',
        f'title_off={_style_repr(title_off)}',
        f'vlim={_style_repr(inps.vlim)}',
        f'tag={inps.tag_string}',
        f'plot_step_factor={inps.plot_step_factor}',
        f'ts_map=1',
        f'dataset={dataset_label or ""}',
        f'scatter={inps.scatter_size}',
    ])


def _basename_candidates(project, tag_string, label, start, end):
    tag_string = (tag_string or '').strip()
    if tag_string:
        yield f'{project}_{tag_string}_{label}_{start}_{end}'
    yield f'{project}_{label}_{start}_{end}'


def _cached_txt_glob_patterns(out_dir, project, tag_string, label):
    """Glob patterns for txt lookup (tagged names first, then legacy)."""
    tag_string = (tag_string or '').strip()
    patterns = []
    if tag_string:
        patterns.append(os.path.join(out_dir, f'{project}_{tag_string}_{label}_*.txt'))
    patterns.append(os.path.join(out_dir, f'{project}_{label}_*.txt'))
    if label == 'profile':
        for legacy in ('profiles_stacked', 'profiles_subplot', 'profiles_separate'):
            if tag_string:
                patterns.insert(0, os.path.join(out_dir, f'{project}_{tag_string}_{legacy}_*.txt'))
            patterns.append(os.path.join(out_dir, f'{project}_{legacy}_*.txt'))
    return patterns


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

    for pattern in _cached_txt_glob_patterns(out_dir, project, tag_string, label):
        for path in sorted(glob.glob(pattern)):
            try:
                _, bracket = parse_txt_header(path)
            except ValueError:
                continue
            if bracket == bracket_info:
                return path
    return None


def discover_timeseries_span(out_dir, project, tag_string):
    """Return (start, end) from a cached timeseries txt filename, or None."""
    patterns = [os.path.join(out_dir, f'{project}_timeseries_*.txt')]
    tag_string = (tag_string or '').strip()
    if tag_string:
        patterns.insert(0, os.path.join(out_dir, f'{project}_{tag_string}_timeseries_*.txt'))
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            stem = os.path.splitext(os.path.basename(path))[0]
            match = re.search(r'_timeseries_(\d{8})_(\d{8})$', stem)
            if match:
                return match.group(1), match.group(2)
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
