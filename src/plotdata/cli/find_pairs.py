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
        description='Find date pairs between two timeseries and write image_pairs.txt',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('file', nargs=2, help='Ascending and descending files; both geocoded similarly.')
    parser.add_argument('--start-date', dest='start_date', nargs='*', default=[], metavar='YYYYMMDD', help='Start date of limited period')
    parser.add_argument('--end-date', dest='stop_date', nargs='*', default=[], metavar='YYYYMMDD', help='End date of limited period')
    parser.add_argument('--period', dest='period', nargs='*', default=[], metavar='YYYYMMDD:YYYYMMDD', help='Period of the search')
    parser.add_argument('--search-interval', dest='search_interval', type=int, default=1, help='Number of repeat intervals to search for date pairing (default: %(default)s).')
    parser.add_argument('--exclude-dates', dest='exclude_dates', nargs='*', default=[], metavar='YYYYMMDD[,YYYYMMDD...]', help='Dates to exclude before pairing')
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
    # Normalize exclude dates (split comma-separated entries)
    excludes = []
    for item in inps.exclude_dates:
        excludes.extend([d for d in item.split(',') if d])
    inps.exclude_dates = excludes


def limit_dates(date_list, inps):
    intervals = []
    max_len = max(len(inps.start_date), len(inps.stop_date))
    for i in range(max_len):
        start = inps.start_date[i] if i < len(inps.start_date) else None
        stop = inps.stop_date[i] if i < len(inps.stop_date) else None
        intervals.append((start, stop))

    if not intervals:
        mask = np.ones(len(date_list), dtype=bool)
        dates = np.array([to_date(d) for d in date_list])
    else:
        dates = np.array([to_date(d) for d in date_list])
        mask = np.zeros(len(dates), dtype=bool)

        for start, stop in intervals:
            start_d = to_date(start) if start else dates.min()
            stop_d = to_date(stop) if stop else dates.max()
            mask |= (dates >= start_d) & (dates <= stop_d)

    if inps.exclude_dates:
        exclude_set = {to_date(d) for d in inps.exclude_dates}
        mask &= ~np.isin(dates, list(exclude_set))

    return date_list[mask]


def match_dates(a, b, schedule):
    """Match dates following the provided shift schedule (ordered list of (shift, block))."""
    print("-" * 50)
    print("Matching dates with custom shift schedule\n")

    a_vals = np.array([to_date(x) for x in a])
    b_vals = np.array([to_date(x) for x in b])

    b_index = {d: idx for idx, d in enumerate(b_vals)}

    matched_a = set()
    matched_b = set()
    all_pairs = []
    block_counts = {}
    block_pairs = {}
    block_map = {}
    shift_map = {}
    for shift_num, block_idx in schedule:
        added = 0
        shifted = a_vals + timedelta(days=shift_num)
        for i, shifted_date in enumerate(shifted):
            if shifted_date not in b_index:
                continue
            da = a_vals[i]
            db = b_vals[b_index[shifted_date]]
            if da in matched_a or db in matched_b:
                continue
            matched_a.add(da)
            matched_b.add(db)
            all_pairs.append((da, db))
            block_map[(da, db)] = block_idx
            shift_map[(da, db)] = shift_num
            added += 1
            if shift_num != 0:
                block_counts[block_idx] = block_counts.get(block_idx, 0) + 1
                block_pairs.setdefault(block_idx, []).append(
                    f"{to_date(da).strftime('%Y%m%d')}->{to_date(db).strftime('%Y%m%d')} ({'+' if shift_num>0 else ''}{shift_num})"
                )
        shift_str = f"+{shift_num}" if shift_num > 0 else str(shift_num)
        print(f"shift={shift_str} pairs found={added}")

    if not all_pairs:
        return np.empty((0, 2), dtype=object), block_counts, block_pairs, block_map, shift_map
    return np.array(all_pairs, dtype=object), block_counts, block_pairs, block_map, shift_map


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


def write_date_table(ts1_dates, ts2_dates, pairs, meta1, meta2, output_path, note=None, pair_symbols=None, pair_shifts=None, legend_lines=None):
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
        symbol = '*'
        if pair_symbols:
            symbol = pair_symbols.get((d1, d2), symbol)
        entries.append((symbol, d1, d2, min(_date_key(d1), _date_key(d2))))

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
        shift_txt = ""
        if pair_shifts is not None:
            s_val = pair_shifts.get((d1, d2))
            if s_val is not None:
                sign = "+" if s_val > 0 else ""
                shift_txt = f" ({sign}{s_val})" if s_val != 0 else " (0)"
        lines.append(f"{marker}{d1:>{col_width}}  {d2:>{col_width}}{shift_txt}")

    summary = f"Totals: {_track_label(meta1)} {len(ts1_dates)}, {_track_label(meta2)} {len(ts2_dates)}, pairs {len(pair_list)}"
    lines.append(summary)
    if note:
        lines.append(note)
    if legend_lines:
        lines.append("")
        lines.extend(legend_lines)
    lines.append("")  # trailing newline

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def match_and_filter_dates(ts1_dates, ts2_dates, meta1, meta2, inps):
    """Match dates between two date arrays."""
    def _shift_schedule(search_interval):
        schedule = []
        block_ranges = []
        pos_start, pos_end = 0, 6
        neg_start, neg_end = -1, -5
        for block in range(max(1, search_interval) * 2):
            if block % 2 == 0:
                block_ranges.append((pos_start, pos_end))
                for s in range(pos_start, pos_end + 1):
                    schedule.append((s, block))
                pos_start = pos_end + 1
                if block == 0:
                    pos_end = pos_start + 4  # 7..11
                else:
                    pos_end = pos_start + 5  # 12..17 onward
            else:
                block_ranges.append((neg_start, neg_end))
                for s in range(neg_start, neg_end - 1, -1):
                    schedule.append((s, block))
                neg_start = neg_end - 1
                neg_end = neg_start - 5
        return schedule, block_ranges

    schedule, block_ranges = _shift_schedule(inps.search_interval)
    print(f"Shift schedule blocks: {block_ranges}")

    pairs, block_counts, block_pairs, block_map, shift_map = match_dates(ts1_dates, ts2_dates, schedule)

    ts1_filtered = np.array([d for d in ts1_dates if to_date(d) in pairs[:, 0]], dtype=object)
    ts2_filtered = np.array([d for d in ts2_dates if to_date(d) in pairs[:, 1]], dtype=object)

    delta_days_array = np.array([(datetime.strptime(to_date(y).strftime("%Y%m%d"), "%Y%m%d").date() - datetime.strptime(to_date(x).strftime("%Y%m%d"), "%Y%m%d").date()).days for x, y in zip(ts1_filtered, ts2_filtered)])

    return ts1_filtered, ts2_filtered, delta_days_array, pairs, block_ranges, block_counts, block_pairs, block_map, shift_map


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

    ts1_filtered, ts2_filtered, delta_days, pairs, block_ranges, block_counts, block_pairs, block_map, shift_map = match_and_filter_dates(ts1_dates, ts2_dates, meta1, meta2, inps)
    max_shift = max((max(abs(r[0]), abs(r[1])) for r in block_ranges), default=0)
    diff_msg = describe_shift(ts1_filtered, ts2_filtered, meta1, meta2, limit=max_shift)
    print(diff_msg)
    interval_lines = []
    symbols = list("*+-:!@#$%^&():\";'<>,.?/")
    for idx, rng in enumerate(block_ranges):
        rng_str = f"{rng[0]}..{rng[1]}"
        count = block_counts.get(idx, 0)
        interval_lines.append("")
        interval_lines.append(f"Interval {idx+1} [{rng_str}]: {count} pairs")
        for (da, db), bidx in block_map.items():
            if bidx != idx:
                continue
            d1 = to_date(da).strftime("%Y%m%d")
            d2 = to_date(db).strftime("%Y%m%d")
            s_val = shift_map.get((da, db), 0)
            sign = "+" if s_val > 0 else ""
            sym = symbols[idx % len(symbols)]
            interval_lines.append(f"{sym}{d1}  {d2} ({sign}{s_val})")
    symbol_map = {}
    shift_display = {}
    for (da, db), block_idx in block_map.items():
        d1 = to_date(da).strftime("%Y%m%d")
        d2 = to_date(db).strftime("%Y%m%d")
        s_val = shift_map.get((da, db), 0)
        symbol = symbols[block_idx % len(symbols)]
        symbol_map[(d1, d2)] = symbol
        shift_display[(d1, d2)] = s_val
    extra = diff_msg + ("\n" + "\n".join(interval_lines) if interval_lines else "")
    legend_lines = []
    for s in sorted(set(symbol_map.values()), key=lambda x: symbols.index(x)):
        shifts = sorted({shift_display[k] for k, v in symbol_map.items() if v == s}, key=abs)
        if shifts:
            shift_txt = ", ".join([f"{'+' if sv > 0 else ''}{sv} days" for sv in shifts])
            legend_lines.append(f"{s} {shift_txt}")
    write_date_table(ts1_dates, ts2_dates, pairs, meta1, meta2, os.path.join(project_base_dir, "image_pairs.txt"), note=extra, pair_symbols=symbol_map, pair_shifts=shift_display, legend_lines=legend_lines)


if __name__ == "__main__":
    main()
