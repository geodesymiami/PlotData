#!/usr/bin/env python3
"""Parse and validate --period date ranges."""

import re

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


def snap_consecutive_start(eos_file, boundary_date):
    """Use ``boundary_date`` if it is an acquisition; else nearest earlier acquisition.

    Only for consecutive periods where period N+1 starts on period N's end date.
    """
    from mintpy.objects import HDFEOS

    date_list = [d.decode() if isinstance(d, bytes) else str(d)
                 for d in HDFEOS(eos_file).get_date_list()]
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


def parse_ylim_tokens(tokens, num_periods):
    """Parse --ylim values into one (ymin, ymax) tuple per period.

    One pair applies to all periods; N pairs apply to N periods in order.
    """
    if not tokens:
        return None
    if len(tokens) % 2 != 0:
        raise ValueError('--ylim requires pairs of values: YMIN YMAX [YMIN2 YMAX2 ...]')
    pairs = [(float(tokens[i]), float(tokens[i + 1])) for i in range(0, len(tokens), 2)]
    for ymin, ymax in pairs:
        if ymin >= ymax:
            raise ValueError(f'--ylim pair ({ymin}, {ymax}): ymin must be less than ymax')
    if len(pairs) == 1:
        return [pairs[0]] * num_periods
    if len(pairs) != num_periods:
        raise ValueError(
            f'--ylim provides {len(pairs)} period limit(s) but {num_periods} period(s) '
            f'were requested')
    return pairs
