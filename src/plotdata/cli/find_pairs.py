#!/usr/bin/env python3
"""
Lightweight pairing script: reads date lists from two timeseries files and
produces dates.txt using the same pairing logic as horzvert_timeseries.py
but without computing horizontal/vertical outputs.
"""

import os
import argparse
import re
import numpy as np
from datetime import datetime, timedelta
from plotdata.helper_functions import get_file_names, prepend_scratchdir_if_needed, to_date
from mintpy.utils import readfile
from mintpy.objects import timeseries, HDFEOS


def _track_label(meta):
    direction = meta.get('ORBIT_DIRECTION', '')
    direction_char = direction[0].upper() if direction else ''
    rel = meta.get('relative_orbit') or meta.get('relativeOrbit')
    try:
        rel_num = f"{int(rel):03d}"
    except Exception:
        rel_num = str(rel) if rel is not None else ''
    return f"{direction_char}{rel_num}"


def create_parser(iargs=None):
    parser = argparse.ArgumentParser(
        description='Find date pairs between two timeseries and write dates.txt',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('file', nargs=2, help='Ascending and descending files; both geocoded similarly.')
    parser.add_argument('--start-date', dest='start_date', nargs='*', default=[], metavar='YYYYMMDD', help='Start date of limited period')
    parser.add_argument('--end-date', dest='stop_date', nargs='*', default=[], metavar='YYYYMMDD', help='End date of limited period')
    parser.add_argument('--period', dest='period', nargs='*', default=[], metavar='YYYYMMDD:YYYYMMDD', help='Period of the search')
    parser.add_argument('--search-interval', dest='search_interval', type=int, default=1, help='Number of repeat intervals to search for date pairing (default: %(default)s).')
    return parser.parse_args(iargs)


def parse_periods(inps):
    if inps.period:
        for p in inps.period:
            delimiters = '[,:\\-\\s]'
            dates = re.split(delimiters, p)

            if len(dates) < 2 or len(dates[0]) != 8 or len(dates[1]) != 8:
                raise ValueError('Date format not valid, it must be in the format YYYYMMDD')

            inps.start_date.append(dates[0])
            inps.stop_date.append(dates[1])


def limit_dates(date_list, inps):
    intervals = []
    max_len = max(len(inps.start_date), len(inps.stop_date))
    for i in range(max_len):
        start = inps.start_date[i] if i < len(inps.start_date) else None
        stop = inps.stop_date[i] if i < len(inps.stop_date) else None
        intervals.append((start, stop))

    if not intervals:
        return date_list

    dates = np.array([to_date(d) for d in date_list])
    mask = np.zeros(len(dates), dtype=bool)

    for start, stop in intervals:
        start_d = to_date(start) if start else dates.min()
        stop_d = to_date(stop) if stop else dates.max()
        mask |= (dates >= start_d) & (dates <= stop_d)

    return date_list[mask]


def match_dates(a, b, lower_shift, upper_shift):
    """Match dates allowing shifts between lower_shift and upper_shift (inclusive)."""
    if lower_shift > upper_shift:
        lower_shift, upper_shift = upper_shift, lower_shift

    print("-" * 50)
    print(f"Matching dates with shift range [{lower_shift}, {upper_shift}] days\n")

    a_vals = np.array([to_date(x) for x in a])
    b_vals = np.array([to_date(x) for x in b])

    b_index = {d: idx for idx, d in enumerate(b_vals)}
    shift_candidates = {}
    for shift in range(lower_shift, upper_shift + 1):
        shifted = a_vals + timedelta(days=shift)
        matches = []
        for i, shifted_date in enumerate(shifted):
            if shifted_date in b_index:
                matches.append((a_vals[i], b_vals[b_index[shifted_date]]))
        shift_candidates[shift] = matches

    matched_a = set()
    matched_b = set()
    all_pairs = []
    shifts_ordered = []
    for s in range(0, upper_shift + 1):
        shifts_ordered.append(s)
    for s in range(-1, lower_shift - 1, -1):
        shifts_ordered.append(s)

    for shift_val in shifts_ordered:
        added = 0
        for da, db in shift_candidates[shift_val]:
            if da in matched_a or db in matched_b:
                continue
            matched_a.add(da)
            matched_b.add(db)
            all_pairs.append((da, db))
            added += 1
        shift_str = f"+{shift_val}" if shift_val > 0 else str(shift_val)
        print(f"shift={shift_str} pairs found={added}")

    if not all_pairs:
        return np.empty((0, 2), dtype=object)
    return np.array(all_pairs, dtype=object)


def describe_shift(ts1_dates, ts2_dates, meta1, meta2, limit):
    """Describe the minimal shift (in days) from ts1 to ts2, signed by direction."""
    a_vals = np.array([to_date(x) for x in ts1_dates])
    b_vals = set(to_date(x) for x in ts2_dates)
    shift_val = None
    best_abs = None
    for k in range(0, limit + 1):
        for shift in (k, -k) if k > 0 else (0,):
            shifted = a_vals + timedelta(days=shift)
            if any(d in b_vals for d in shifted):
                if best_abs is None or abs(shift) < best_abs:
                    shift_val = shift
                    best_abs = abs(shift)
                break
        if shift_val is not None:
            break

    if shift_val is None:
        return f"diff {_track_label(meta1)} to {_track_label(meta2)}: none"

    suffix = "day" if abs(shift_val) == 1 else "days"
    sign_str = "+" if shift_val > 0 else ""
    return f"diff {_track_label(meta1)} to {_track_label(meta2)}: {sign_str}{shift_val} {suffix}"


def write_date_table(ts1_dates, ts2_dates, pairs, meta1, meta2, output_path, note=None):
    """Write a table aligning timeseries dates and marking matched pairs."""
    col_width = 8  # YYYYMMDD

    def _fmt_date(val):
        return to_date(val).strftime("%Y%m%d")

    def _date_key(date_str):
        return datetime.strptime(date_str, "%Y%m%d").date()

    ts1_list = [_fmt_date(d) for d in ts1_dates]
    ts2_list = [_fmt_date(d) for d in ts2_dates]
    pair_list = [(_fmt_date(p[0]), _fmt_date(p[1])) for p in pairs]

    matched1 = {d1 for d1, _ in pair_list}
    matched2 = {d2 for _, d2 in pair_list}

    entries = []
    for d1, d2 in pair_list:
        entries.append(('*', d1, d2, min(_date_key(d1), _date_key(d2))))

    for d1 in ts1_list:
        if d1 not in matched1:
            entries.append((' ', d1, '', _date_key(d1)))

    for d2 in ts2_list:
        if d2 not in matched2:
            entries.append((' ', '', d2, _date_key(d2)))

    entries.sort(key=lambda x: x[3])

    header = f" {_track_label(meta1):>{col_width}}  {_track_label(meta2):>{col_width}}"
    lines = [header]
    for marker, d1, d2, _ in entries:
        lines.append(f"{marker}{d1:>{col_width}}  {d2:>{col_width}}")

    summary = f"Totals: {_track_label(meta1)} {len(ts1_dates)}, {_track_label(meta2)} {len(ts2_dates)}, pairs {len(pair_list)}"
    lines.append(summary)
    if note:
        lines.append(note)
    lines.append("")  # trailing newline

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def match_and_filter_dates(ts1_dates, ts2_dates, meta1, meta2, inps):
    """Match dates between two date arrays."""
    def _repeat_interval(meta):
        mission = meta.get('mission', '')
        if 'Sen' in mission:
            return 12
        return 12

    repeat_interval = _repeat_interval(meta1)
    span = max(0, inps.search_interval) * repeat_interval
    if span == 0:
        lower_shift = upper_shift = 0
    else:
        lower_shift = -(span // 2) + (1 if span % 2 == 0 else 0)
        upper_shift = lower_shift + span - 1
    print(f"Shift search window: [{lower_shift}, {upper_shift}] days (span={span})")

    pairs = match_dates(ts1_dates, ts2_dates, lower_shift, upper_shift)

    ts1_filtered = np.array([d for d in ts1_dates if to_date(d) in pairs[:, 0]], dtype=object)
    ts2_filtered = np.array([d for d in ts2_dates if to_date(d) in pairs[:, 1]], dtype=object)

    delta_days_array = np.array([(datetime.strptime(to_date(y).strftime("%Y%m%d"), "%Y%m%d").date() - datetime.strptime(to_date(x).strftime("%Y%m%d"), "%Y%m%d").date()).days for x, y in zip(ts1_filtered, ts2_filtered)])

    return ts1_filtered, ts2_filtered, delta_days_array, pairs, (lower_shift, upper_shift)


def load_dates(file_path, inps):
    work_dir = prepend_scratchdir_if_needed(file_path)
    eos_file, _, _, project_base_dir, _, _ = get_file_names(work_dir)
    attr = readfile.read_attribute(eos_file)

    file_type = attr.get('FILE_TYPE')
    if file_type == 'timeseries':
        obj = timeseries(eos_file)
    elif file_type == 'HDFEOS':
        obj = HDFEOS(eos_file)
    else:
        raise ValueError(f"Unsupported input file type: {file_type}")

    obj.open()
    dates = np.array(obj.dateList)
    obj.close()

    dates = limit_dates(dates, inps)
    meta = {
        'mission': attr.get('mission', ''),
        'relative_orbit': attr.get('relative_orbit') or attr.get('relativeOrbit'),
        'ORBIT_DIRECTION': attr.get('ORBIT_DIRECTION', ''),
        'FILE_PATH': eos_file,
        'project_base_dir': project_base_dir,
    }
    return dates, meta, project_base_dir


def main(iargs=None):
    inps = create_parser(iargs)
    parse_periods(inps)

    dates_meta = []
    project_base_dir = None
    for f in inps.file:
        dates, meta, project_base_dir = load_dates(f, inps)
        dates_meta.append((dates, meta))

    (ts1_dates, meta1), (ts2_dates, meta2) = dates_meta

    ts1_filtered, ts2_filtered, delta_days, pairs, shift_window = match_and_filter_dates(ts1_dates, ts2_dates, meta1, meta2, inps)
    max_shift = max(abs(shift_window[0]), abs(shift_window[1]))
    diff_msg = describe_shift(ts1_filtered, ts2_filtered, meta1, meta2, limit=max_shift)
    print(diff_msg)
    write_date_table(ts1_dates, ts2_dates, pairs, meta1, meta2, os.path.join(project_base_dir, "dates.txt"), note=diff_msg)


if __name__ == "__main__":
    main()
