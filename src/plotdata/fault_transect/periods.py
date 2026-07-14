#!/usr/bin/env python3
"""Parse and validate --period date ranges."""

import re


def format_date_display(date_yyyymmdd):
    """Format YYYYMMDD for display as YYYY-MM-DD."""
    text = str(date_yyyymmdd)
    if len(text) != 8 or not text.isdigit():
        return text
    return f'{text[:4]}-{text[4:6]}-{text[6:8]}'


def format_period_display(start_date, end_date):
    """Format a period range for plot labels (no tag/project prefix)."""
    return f'{format_date_display(start_date)} - {format_date_display(end_date)}'


def parse_period_chunks(tokens):
    """Parse --period tokens into list of (start, end) YYYYMMDD strings."""
    periods = []
    for token in tokens:
        for chunk in token.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            match = re.fullmatch(r'(\d{8}):(\d{8})', chunk)
            if not match:
                raise ValueError(f'Invalid --period "{chunk}": expected YYYYMMDD:YYYYMMDD')
            periods.append((match.group(1), match.group(2)))
    return periods


def is_consecutive_boundary(prev_end, start):
    """True when period N+1 starts on the same day period N ends (back-to-back)."""
    return start == prev_end


def consecutive_start_flags(periods):
    """For each period, True if its start equals the previous period's end."""
    if not periods:
        return []
    flags = [False]
    for i in range(1, len(periods)):
        flags.append(is_consecutive_boundary(periods[i - 1][1], periods[i][0]))
    return flags


def gap_start_flags(periods):
    """For each period, True if its start is after the previous period's end (a gap)."""
    if not periods:
        return []
    flags = [False]
    for i in range(1, len(periods)):
        flags.append(int(periods[i][0]) > int(periods[i - 1][1]))
    return flags


def _eos_date_list(eos_file):
    from mintpy.objects import HDFEOS
    return [d.decode() if isinstance(d, bytes) else str(d)
            for d in HDFEOS(eos_file).get_date_list()]


def snap_first_period_start(date_list, start_date):
    """Nearest acquisition on or before ``start_date`` (first period)."""
    if int(start_date) < int(date_list[0]):
        print(f'WARNING: no acquisition on or before {start_date}; using {date_list[0]}')
        return date_list[0]
    for date in reversed(date_list):
        if int(date) <= int(start_date):
            return date
    return date_list[0]


def snap_gap_start(date_list, start_date):
    """First acquisition on or after ``start_date`` (gap between periods)."""
    if int(start_date) > int(date_list[-1]):
        print(f'WARNING: no acquisition on or after {start_date}; using {date_list[-1]}')
        return date_list[-1]
    for date in date_list:
        if int(date) >= int(start_date):
            if date != start_date:
                print(f'WARNING: gap period start {start_date} is not an '
                      f'acquisition; using {date}')
            return date
    return date_list[-1]


def snap_end_date(date_list, end_date):
    """Nearest acquisition on or before ``end_date``."""
    if int(end_date) > int(date_list[-1]):
        print(f'WARNING: no acquisition on or before {end_date}; using {date_list[-1]}')
        return date_list[-1]
    for date in reversed(date_list):
        if int(date) <= int(end_date):
            return date
    return date_list[-1]


def snap_period_dates(eos_file, start_date, end_date, *, consecutive_start=False, gap_start=False):
    """Snap requested period bounds to available acquisitions.

    * First period: start on or before requested start; end on or before requested end.
    * Gap (``start > prev_end``): start on or **after** requested start.
    * Consecutive (``start == prev_end``): start on or before boundary (shared date).
    """
    date_list = _eos_date_list(eos_file)
    req_start, req_end = start_date, end_date
    if consecutive_start:
        start_date = snap_consecutive_start(eos_file, start_date)
    elif gap_start:
        start_date = snap_gap_start(date_list, start_date)
    else:
        start_date = snap_first_period_start(date_list, start_date)
    end_date = snap_end_date(date_list, end_date)

    print('###############################################')
    print(' Period of data:  ', format_date_display(date_list[0]),
          format_date_display(date_list[-1]))
    print(' Period requested:', format_date_display(req_start), format_date_display(req_end))
    print(' Period used:     ', format_date_display(start_date), format_date_display(end_date))
    print('###############################################')
    return start_date, end_date


def snap_consecutive_start(eos_file, boundary_date):
    """Use ``boundary_date`` if it is an acquisition; else nearest earlier acquisition.

    Only for consecutive periods where period N+1 starts on period N's end date.
    """
    date_list = _eos_date_list(eos_file)
    if boundary_date in date_list:
        return boundary_date
    for date in reversed(date_list):
        if int(date) <= int(boundary_date):
            if date != boundary_date:
                print(f'WARNING: consecutive period start {boundary_date} is not an '
                      f'acquisition; using {date}')
            return date
    print(f'WARNING: no acquisition on or before {boundary_date}; using {date_list[0]}')
    return date_list[0]


def validate_and_adjust_periods(periods, adjust=True):
    """Validate period ordering and boundary semantics.

    * ``start > prev_end``: gap between periods (e.g. 20181224 then 20181225).
    * ``start == prev_end``: consecutive periods sharing one boundary date.
    * ``start < prev_end``: overlap — always rejected.
    """
    if not periods:
        return []

    adjusted = []
    for i, (start, end) in enumerate(periods):
        if start > end:
            raise ValueError(
                f'--period {start}:{end}: start date must be on or before end date')

        if i > 0:
            prev_end = adjusted[-1][1]
            if start < prev_end:
                raise ValueError(
                    f'--period {start}:{end} overlaps the previous period '
                    f'(ends {prev_end})')
            if start == prev_end:
                pass  # consecutive — keep boundary date as-is
            elif start > prev_end:
                pass  # gap — keep explicit start date

        adjusted.append((start, end))
    return adjusted
