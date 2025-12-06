#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import re
import numpy as np
import math
from typing import Any
from types import SimpleNamespace
from datetime import datetime, timedelta
from mintpy.objects.resample import resample
from mintpy.objects import timeseries, HDFEOS
from mintpy.utils import readfile, utils as ut, writefile
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert, get_overlap_lalo
from plotdata.helper_functions import (
    get_file_names, prepend_scratchdir_if_needed, extract_window, detect_cores,
    find_reference_points_from_subsets, create_geometry_file, find_longitude_degree, to_date, get_output_filename
)
from concurrent.futures import ProcessPoolExecutor

SCRATCHDIR = os.getenv('SCRATCHDIR')
EXAMPLE = """
Example usage:
    horzvert_timeseries.py ChilesSenD142/mintpy ChilesSenA120/mintpy --ref-lalo 0.84969 -77.86430
    horzvert_timeseries.py ChilesSenD142/mintpy ChilesSenA120/mintpy --ref-lalo 0.84969 -77.86430 --dry-run
    horzvert_timeseries.py ChilesSenD142/mintpy ChilesSenA120/mintpy --ref-lalo 0.84969 -77.86430 --intervals 6
    horzvert_timeseries.py hvGalapagosSenD128/mintpy hvGalapagosSenA106/mintpy --ref-lalo -0.81 -91.190
    horzvert_timeseries.py hvGalapagosSenD128/miaplpy/network_single_reference hvGalapagosSenA106/network_single_reference --ref-lalo -0.81 -91.190
    horzvert_timeseries.py FernandinaSenD128/mintpy FernandinaSenA106/mintpy --ref-lalo -0.453 -91.390
    horzvert_timeseries.py FernandinaSenD128/miaplpy/network_delaunay_4 FernandinaSenA106/miaplpy/network_delaunay_4 --ref-lalo -0.453 -91.390
"""


def create_parser(iargs=None, namespace=None):
    """
    Creates command line argument parser object.

    Args:
        iargs (list): List of command line arguments (default: None)
        namespace (argparse.Namespace): Namespace object to store parsed arguments (default: None)

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate vertical and horizontal timeseries',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)

    parser.add_argument('file', nargs=2, help='Ascending and descending files\n' 'If geocoded in needs to be same posting.')
    parser.add_argument('-g','--geom-file', dest='geom_file', nargs=2, help='Geometry files for the input data files.')
    parser.add_argument('--mask-thresh', dest='mask_vmin', nargs='+', type=float, default=[None, None], help='coherence threshold for masking (default: %(default)s).')
    parser.add_argument('--ref-lalo', nargs='*', metavar=('LATITUDE,LONGITUDE or LATITUDE LONGITUDE'), default=None, type=str, help='reference point (default: existing reference point)')
    parser.add_argument('--lat-step', dest='lat_step', type=float, default=-0.00014, help='latitude step for geocoding (lon step same, from find_longitude_degree) (default: %(default)s).')
    parser.add_argument('--horz-az-angle', dest='horz_az_angle', type=float, default=90, help='Horizontal azimuth angle (default: %(default)s).')
    parser.add_argument('--window-size', dest='window_size', type=int, default=3, help='window size (square side in number of pixels) for reference point look up (default: %(default)s).')
    parser.add_argument('-ow', '--overwrite', dest='overwrite', action='store_true', help='Overwrite all previously generated files')
    parser.add_argument('-ts', '--timeseries', dest='timeseries', action='store_true', help='Output timeseries file in addition to HDFEOS format')
    parser.add_argument('--intervals', dest='interval_index', type=int, default=2,
            help=('Interval block index [0..repeat_interval/2] to search (1=first positive block, 2=first negative, etc.). '
                  '\n>3: pairs larger than repeat_interval/2 can be formed. (Default: 2, immediate pairs only).'))
    parser.add_argument('--start-date', dest='start_date', nargs='*', default=[], metavar='YYYYMMDD', help='Start date of limited period')
    parser.add_argument('--end-date', dest='stop_date', nargs='*', default=[], metavar='YYYYMMDD', help='End date of limited period')
    parser.add_argument('--period', dest='period', nargs='*', default=[], metavar='YYYYMMDD:YYYYMMDD', help='Period of the search')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', help='Write image_pairs.txt only (no horz/vert processing)')
    parser.add_argument('--exclude-dates', dest='exclude_dates', nargs='*', default=[], metavar='YYYYMMDD[,YYYYMMDD...]', help='Dates to exclude before pairing (for debugging)')
    parser.add_argument('--no-swap', dest='no_swap', action='store_true', help='Do not swap datasets (for debugging)')

    inps = parser.parse_args(iargs, namespace)

    if inps.ref_lalo:
        inps.ref_lalo = parse_lalo(inps.ref_lalo)

    if len(inps.mask_vmin) > len(inps.file):
        inps.mask_vmin = inps.mask_vmin[:len(inps.file)]
    elif len(inps.mask_vmin) < len(inps.file):
        inps.mask_vmin = (inps.mask_vmin * (len(inps.file)))[:len(inps.file)]


    return inps


def parse_lalo(str_lalo):
    """Parse the lat/lon input from the command line.

    Args:
        str_lalo (str or list): The lat/lon input as a string or list.

    Returns:
        list or float: Parsed lat/lon coordinates.
    """
    if ',' in str_lalo[0]:
        lalo = [[float(coord) for coord in pair.split(',')] for pair in str_lalo]
    else:
        lalo = [[float(str_lalo[i]), float(str_lalo[i+1])] for i in range(0, len(str_lalo), 2)]

    if len(lalo) == 1:  # if given as one string containing ','
        lalo = lalo[0]
    return lalo


def parse_periods(inps):
    """Expand --period entries into start/stop lists and normalize excludes."""
    if inps.period:
        for p in inps.period:
            delimiters = '[,:\\-\\s]'
            dates = re.split(delimiters, p)

            if len(dates) < 2 or len(dates[0]) != 8 or len(dates[1]) != 8:
                raise ValueError('Date format not valid, it must be in the format YYYYMMDD')

            inps.start_date.append(dates[0])
            inps.stop_date.append(dates[1])

    excludes = []
    for item in inps.exclude_dates:
        excludes.extend([d for d in item.split(',') if d])
    inps.exclude_dates = excludes


def configure_logging(directory=None):
    """
    Configure logging so all INFO entries are written to:
      - Quiet noisy third-party loggers.
      - <project_base_dir>/log      (if provided)
      - $SCRATCHDIR/log             (if SCRATCHDIR exists)

    Logging handlers are added only once, even if called multiple times.
    """

    noisy = ('ipykernel', 'ipykernel.comm', 'jupyter_client', 'zmq', 'tornado', 'asyncio', 'matplotlib')
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # If logger has no handlers, configure them
    if not logger.handlers:
        paths = [
            os.path.join(directory, 'log') if directory else None,
            os.path.join(os.getenv('SCRATCHDIR'), 'log') if os.getenv('SCRATCHDIR') else None,
        ]

        for path in paths:
            if not path:
                continue

            # Create parent directory if missing
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            handler = logging.FileHandler(path)
            formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    # Log the command-line call (script name + args)
    cmd_args = [os.path.basename(sys.argv[0])] + sys.argv[1:]
    cmd_command = ' '.join(cmd_args)
    logger.info(cmd_command)


def _infer_project_base_dir(input_paths):
    """Best-effort guess of the project base directory from input paths."""
    scratchdir = os.getenv('SCRATCHDIR')
    keywords = ('SenD', 'SenA', 'SenDT', 'SenAT', 'CskAT', 'CskDT')
    dir = scratchdir

    for path in input_paths:
        for element in os.path.normpath(path).split(os.sep):
            for keyword in keywords:
                if keyword in element:
                    base = element.split(keyword)[0]
                    dir = os.path.join(scratchdir, base)
                    break

    return dir


# Globals used by worker
_G_DATA = None
_G_INC = None
_G_AZ = None
_G_HORZ_AZ_ANGLE = None

def _init_worker(data, los_inc_angle, los_az_angle, horz_az_angle):
    """Initializer: sets global variables in each worker process."""
    global _G_DATA, _G_INC, _G_AZ, _G_HORZ_AZ_ANGLE
    _G_DATA = data
    _G_INC = los_inc_angle
    _G_AZ = los_az_angle
    _G_HORZ_AZ_ANGLE = horz_az_angle

def _asc_desc_worker(i):
    """Worker for a single time slice index i."""
    # Use globals set by _init_worker
    slice_data = _G_DATA[:, i]    # shape (2, length, width)
    hvert, dvert = asc_desc2horz_vert(
        slice_data,
        _G_INC,
        _G_AZ,
        _G_HORZ_AZ_ANGLE,
    )
    return i, dvert, hvert

def _strip_marker_lines(lines, symbols):
    """Remove leading marker+space when present to print summaries cleanly."""
    sym_set = set(symbols)
    cleaned = []
    for ln in lines:
        if ln and ln[0] in sym_set:
            stripped = ln[1:]
            if stripped.startswith(" "):
                stripped = stripped[1:]
            cleaned.append(stripped)
        else:
            cleaned.append(ln)
    return cleaned

def get_repeat_interval(meta1, meta2):
    """Return repeat interval (days) based on mission/project naming."""
    names = [meta1.get('mission', ''), meta2.get('mission', '')]
    for n in names:
        low = (n or '').lower()
        if 'csk' in low or 'cosmo' in low:
            return 1
        if 'tsx' in low or 'terr' in low:
            return 11
        if 'sen' in low:
            return 12
    return 12  # sensible default if unknown

def match_dates(a, b, schedule):
    """Match dates following the provided shift schedule (ordered list of (shift, block))."""
    print("-"*50)
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


def match_and_filter_pairs(ts1_dates, ts2_dates, meta1, meta2, inps):
    """Match dates between two date arrays (used by fast/dry-run paths)."""
    def _shift_schedule(interval_index):
        schedule = []
        block_ranges = []
        repeat = get_repeat_interval(meta1, meta2)
        step = math.ceil(repeat / 2)
        max_blocks = max(1, interval_index)
        k = 0
        while len(block_ranges) < max_blocks:
            # positive block for this interval
            pos_start = k * step if k == 0 else k * step + 1  # avoid duplicate boundaries
            pos_end = (k + 1) * step
            block_idx_pos = len(block_ranges)
            block_ranges.append((pos_start, pos_end))
            for s in range(pos_start, pos_end + 1):
                schedule.append((s, block_idx_pos))
            if len(block_ranges) >= max_blocks:
                break
            # negative block for this interval
            neg_start, neg_end = -(k * step + 1), -(k + 1) * step
            block_idx_neg = len(block_ranges)
            block_ranges.append((neg_start, neg_end))
            for s in range(neg_start, neg_end - 1, -1):
                schedule.append((s, block_idx_neg))
            k += 1
        return schedule, block_ranges

    schedule, block_ranges = _shift_schedule(inps.interval_index)
    print(f"Shift schedule blocks: {block_ranges}")

    pairs, block_counts, block_pairs, block_map, shift_map = match_dates(ts1_dates, ts2_dates, schedule)

    ts1_filtered = np.array([d for d in ts1_dates if to_date(d) in pairs[:, 0]], dtype=object)
    ts2_filtered = np.array([d for d in ts2_dates if to_date(d) in pairs[:, 1]], dtype=object)

    delta_days_array = np.array([
        (datetime.strptime(to_date(y).strftime("%Y%m%d"), "%Y%m%d").date() -
         datetime.strptime(to_date(x).strftime("%Y%m%d"), "%Y%m%d").date()).days
        for x, y in zip(ts1_filtered, ts2_filtered)
    ])

    return ts1_filtered, ts2_filtered, delta_days_array, pairs, block_ranges, block_counts, block_pairs, block_map, shift_map


def load_timeseries_file(file_path, geometry_file_input, mask_vmin, inps):
    """Load and process a timeseries file.

    Args:
        file_path: Path to the timeseries file
        geometry_file_input: Path to geometry file (from command line, or None)
        inps: Parsed command line arguments

    Returns:
        tuple: (timeseries_object, los_inc_angle, los_az_angle, mask, metadata, project_base_dir, geometry_file)
    """
    work_dir = prepend_scratchdir_if_needed(file_path)
    eos_file, _, geometry_file, project_base_dir, _, _ = get_file_names(work_dir)

    metadata = readfile.read_attribute(eos_file)

    # Use provided geometry file if available
    if geometry_file_input:
        geometry_file = prepend_scratchdir_if_needed(geometry_file_input)

    # Get reference point from metadata if not provided
    if not inps.ref_lalo:
        inps.ref_lalo = parse_lalo([metadata['REF_LAT'], metadata['REF_LON']])

    # Create geometry file if needed
    overwrite_flag = getattr(inps, 'overwrite', False)
    if not os.path.exists(geometry_file) or overwrite_flag:
        os.makedirs(os.path.dirname(geometry_file), exist_ok=True)
        create_geometry_file(eos_file, os.path.dirname(geometry_file))

    # Load file
    file_type = metadata['FILE_TYPE']

    if file_type == 'timeseries':
        obj = timeseries(eos_file)
    elif file_type == 'HDFEOS':
        obj = HDFEOS(eos_file)
    else:
        raise ValueError(f'Unsupported input file type: {file_type}')

    obj.open()

    # Handle bperp for HDFEOS
    if hasattr(obj, 'datasetGroupNameDict'):
        if obj.datasetGroupNameDict.get('bperp') == 'geometry':
            obj.datasetGroupNameDict['bperp'] = 'observation'

    data = obj.read()                 # full timeseries array (may be large)
    dateList = np.array(obj.dateList) # convert to numpy or keep list
    metadata = dict(obj.metadata)     # copy metadata dict

    try:
        bperp = obj.read('bperp')
    except Exception:
        bperp = None

    # Close file-backed object
    obj.close()

    # Handle bperp location in HDFEOS files
    if hasattr(obj, 'datasetGroupNameDict') and obj.datasetGroupNameDict.get('bperp') == 'geometry':
        obj.datasetGroupNameDict['bperp'] = 'observation'

    # Read mask or Temporal Coherence
    mask = None
    if mask_vmin or not (hasattr(obj, 'datasetGroupNameDict') and 'mask' in obj.datasetGroupNameDict):
        if hasattr(obj, 'datasetGroupNameDict') and 'temporalCoherence' in obj.datasetGroupNameDict:
            tc_data = obj.read('temporalCoherence')
            mask_vmin = mask_vmin if mask_vmin else 0.65
            mask = np.where(tc_data >= mask_vmin, True, False)
        else:
            raise ValueError(f"Temporal Coherence dataset not found in {eos_file} for masking.")
    else:
        if hasattr(obj, 'datasetGroupNameDict') and 'mask' in obj.datasetGroupNameDict:
            mask = readfile.read(eos_file, datasetName='mask')[0]
        elif os.path.exists(geometry_file):
            mask = readfile.read(geometry_file, datasetName='mask')[0]
        else:
            print(f"Mask dataset not found in {eos_file} or {geometry_file}, proceeding without mask.")

    # Try to read los angles from geometry file first, fallback to metadata calculation
    if os.path.exists(geometry_file):
        los_inc_angle = readfile.read(geometry_file, datasetName='incidenceAngle')[0]
        los_az_angle = readfile.read(geometry_file, datasetName='azimuthAngle')[0]
    else:
        raise FileNotFoundError("Geometry file not found")

    ts_obj = SimpleNamespace()
    ts_obj.data = data
    ts_obj.dateList = dateList
    ts_obj.metadata = metadata
    ts_obj.bperp = bperp

    # Ensure consistent metadata
    ts_obj.metadata['FILE_TYPE'] = 'timeseries'
    ts_obj.metadata['FILE_PATH'] = eos_file

    ts_obj = limit_timeseries(ts_obj, inps)

    return ts_obj, los_inc_angle, los_az_angle, mask, project_base_dir, geometry_file


def geocode_timeseries(obj, los_inc_angle, los_az_angle, mask, geometry_file, inps):
    """Geocode timeseries data if needed.

    Args:
        obj: Timeseries object
        los_inc_angle: Line-of-sight incidence angle
        los_az_angle: Line-of-sight azimuth angle
        mask: Mask array
        geometry_file: Path to geometry file
        inps: Parsed command line arguments

    Returns:
        tuple: (geocoded_obj, geocoded_los_inc, geocoded_los_az, geocoded_mask, y_step, x_step)
    """

    if 'Y_STEP' not in obj.metadata:
        lon_step = find_longitude_degree(inps.ref_lalo[0], inps.lat_step)
        resampling_obj = resample(geometry_file, lalo_step=[inps.lat_step, lon_step])
        resampling_obj.open()
        resampling_obj.src_meta = obj.metadata
        resampling_obj.prepare()

        obj.data = resampling_obj.run_resample(src_data=obj.data)
        mask = resampling_obj.run_resample(src_data=mask)
        los_inc_angle = resampling_obj.run_resample(src_data=los_inc_angle)
        los_az_angle = resampling_obj.run_resample(src_data=los_az_angle)

        obj.metadata['LENGTH'], obj.metadata['WIDTH'] = los_az_angle.shape
        obj.metadata['Y_STEP'], obj.metadata['X_STEP'] = inps.lat_step, lon_step
        obj.metadata['Y_FIRST'], obj.metadata['X_FIRST'] = (
            np.nanmax(readfile.read(geometry_file, datasetName='latitude')[0]),
            np.nanmin(readfile.read(geometry_file, datasetName='longitude')[0])
        )

        y_step = inps.lat_step
        x_step = lon_step
    else:
        y_step = obj.metadata['Y_STEP']
        x_step = obj.metadata['X_STEP']

    return obj, los_inc_angle, los_az_angle, mask, y_step, x_step


def match_and_filter_dates(ts1, ts2, inps):
    """Match dates between two timeseries and filter by date difference.

    Args:
        ts1: First timeseries object
        ts2: Second timeseries object
        thresh_method: Method for threshold calculation ('min' or 'percentile')

    Returns:
        tuple: (filtered_ts1, filtered_ts2, delta_days, bperp, date_list, pairs)
    """
    def _shift_schedule(interval_index):
        schedule = []
        block_ranges = []
        repeat = get_repeat_interval(ts1.metadata, ts2.metadata)
        step = math.ceil(repeat / 2)
        max_blocks = max(1, interval_index)
        k = 0
        while len(block_ranges) < max_blocks:
            # positive block for this interval
            pos_start = k * step if k == 0 else k * step + 1  # avoid duplicate boundaries
            pos_end = (k + 1) * step
            block_idx_pos = len(block_ranges)
            block_ranges.append((pos_start, pos_end))
            for s in range(pos_start, pos_end + 1):
                schedule.append((s, block_idx_pos))
            if len(block_ranges) >= max_blocks:
                break
            # negative block for this interval
            neg_start, neg_end = -(k * step + 1), -(k + 1) * step
            block_idx_neg = len(block_ranges)
            block_ranges.append((neg_start, neg_end))
            for s in range(neg_start, neg_end - 1, -1):
                schedule.append((s, block_idx_neg))
            k += 1
        return schedule, block_ranges

    schedule, block_ranges = _shift_schedule(inps.interval_index)
    print(f"Shift schedule blocks: {block_ranges}")

    # Match dates and get dropped indices
    pairs, block_counts, block_pairs, block_map, shift_map = match_dates(ts1.dateList, ts2.dateList, schedule)
    index1 = [i for i, d in enumerate(ts1.dateList) if to_date(d) in pairs[:, 0]]
    index2 = [i for i, d in enumerate(ts2.dateList) if to_date(d) in pairs[:, 1]]

    # Remove non-matching dates
    for i, ts in zip([index1, index2], [ts1, ts2]):
        ts.dateList = ts.dateList[i]
        ts.data = ts.data[i]
        # ts.bperp = ts.bperp[i]

    print('-' * 50)
    print(f'New date list length: {len(ts1.dateList)}\n')

    bperp = ts1.bperp[index1]
    date_list = ts1.dateList

    # Calculate delta (date differences for valid indexes)
    # delta = np.array([(datetime.strptime(y, "%Y%m%d").date() - datetime.strptime(x, "%Y%m%d").date()).days for x, y in zip(ts1.dateList[valid_indexes], ts2.dateList[valid_indexes])])
    delta_days_array = np.array([(datetime.strptime(y, "%Y%m%d").date() - datetime.strptime(x, "%Y%m%d").date()).days for x, y in zip(ts1.dateList, ts2.dateList)])

    return ts1, ts2, delta_days_array, bperp, date_list, pairs, (block_ranges, block_counts, block_pairs, block_map, shift_map)


def describe_shift(ts1_dates, ts2_dates, meta1, meta2, limit=12):
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


def limit_timeseries(ts_obj, inps):
    """Limit a timeseries to the requested date windows."""
    intervals = []
    max_len = max(len(inps.start_date), len(inps.stop_date))
    for i in range(max_len):
        start = inps.start_date[i] if i < len(inps.start_date) else None
        stop = inps.stop_date[i] if i < len(inps.stop_date) else None
        intervals.append((start, stop))

    if not intervals:
        dates_mask = np.ones(len(ts_obj.dateList), dtype=bool)
    else:
        dates = np.array([to_date(d) for d in ts_obj.dateList])
        dates_mask = np.zeros(len(dates), dtype=bool)

        for start, stop in intervals:
            start_d = to_date(start) if start else dates.min()
            stop_d = to_date(stop) if stop else dates.max()
            dates_mask |= (dates >= start_d) & (dates <= stop_d)

    if inps.exclude_dates:
        exclude_set = {to_date(d) for d in inps.exclude_dates}
        dates = np.array([to_date(d) for d in ts_obj.dateList])
        dates_mask &= ~np.isin(dates, list(exclude_set))

    ts_obj.dateList = ts_obj.dateList[dates_mask]
    ts_obj.data = ts_obj.data[dates_mask]
    if getattr(ts_obj, 'bperp', None) is not None:
        ts_obj.bperp = ts_obj.bperp[dates_mask]

    return ts_obj


def is_delta_days_max_occurrence_negative(delta_days_array):
    """Return True if the most frequent delta is negative."""
    if delta_days_array.size == 0:
        return False
    vals, counts = np.unique(delta_days_array, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx] < 0


def delta_days_max_occurrence(delta_days_array):
    """Return (value, count) of the most frequent delta_days entry."""
    # Only consider positive deltas when selecting the mode.
    pos_vals = delta_days_array[delta_days_array > 0]
    if pos_vals.size == 0:
        return None, 0
    vals, counts = np.unique(pos_vals, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx], counts[idx]


def _track_label(meta):
    direction = meta.get('ORBIT_DIRECTION', '')
    direction_char = direction[0].upper() if direction else ''
    rel = meta.get('relative_orbit') or meta.get('relativeOrbit')
    try:
        rel_num = f"{int(rel):03d}"
    except Exception:
        rel_num = str(rel) if rel is not None else ''
    return f"{direction_char}{rel_num}"


def write_date_table(ts1_dates, ts2_dates, pairs, meta1, meta2, output_path, note=None, extra_lines=None, pair_symbols=None, pair_shifts=None, legend_lines=None):
    """Write a table aligning timeseries dates and marking matched pairs."""
    col_width = 8  # YYYYMMDD
    SYMBOLS = list("*+-:!@#$%^&():\";'<>,.?/")

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
    if extra_lines:
        lines.extend(extra_lines)
    if legend_lines:
        lines.append("")
        lines.extend(legend_lines)
    lines.append("")  # ensure trailing newline when joined

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def create_timeseries_output(ts_data, date_list, mask, delta_days, bperp, latitude, longitude,
                             metadata, output_path, file_type='timeseries'):
    """Create and write a timeseries output file.

    Args:
        ts_data: Timeseries data array
        date_list: List of dates
        mask: Mask array
        delta_days: Date difference array
        bperp: Perpendicular baseline array
        latitude: Latitude array
        longitude: Longitude array
        metadata: Metadata dictionary
        output_path: Output file path
        file_type: Type of timeseries ('timeseries' or 'mask')
    """
    # Prepare metadata
    output_metadata = metadata.copy()
    output_metadata['WIDTH'], output_metadata['LENGTH'] = mask.shape[1], mask.shape[0]
    output_metadata['xmax'], output_metadata['ymax'] = mask.shape[1], mask.shape[0]
    output_metadata['FILE_PATH'] = output_path
    output_metadata['FILE_TYPE'] = 'timeseries'
    output_metadata['PROCESSOR'] = 'mintpy'
    output_metadata['PROJECT_NAME'] = os.path.basename(os.path.dirname(output_path))
    output_metadata['REF_DATE'] = str(date_list[0])

    mask_dtype = 'bool' if file_type == 'timeseries' else 'uint8'
    ts_dict = {
        'timeseries': ts_data.astype('float32'),
        'date': date_list.astype('S8'),
        'mask': mask.astype(mask_dtype),
        'delta': delta_days.astype('uint8'),
        'bperp': bperp.astype('float32'),
        'latitude': latitude.astype('float32'),
        'longitude': longitude.astype('float32')
    }

    writefile.write(ts_dict, output_path, metadata=output_metadata)


def create_hdfeos_output(ts_data, date_list, mask, delta_days, bperp, latitude, longitude,
                         metadata, output_path, length, width):
    """Create and write an HDFEOS output file with proper structure.

    Args:
        ts_data: Timeseries data array (n_time, length, width)
        date_list: List of dates
        mask: Mask array (length, width)
        delta_days: Date difference array
        bperp: Perpendicular baseline array
        latitude: Latitude array (length,)
        longitude: Longitude array (width,)
        metadata: Metadata dictionary
        output_path: Output file path (should end with .he5)
        length: Number of rows
        width: Number of columns
    """
    # Ensure output path has .he5 extension
    if not output_path.endswith('.he5'):
        output_path = output_path.replace('.h5', '.he5')

    if latitude.ndim == 1:
        lat_grid = np.tile(latitude[:, np.newaxis], (1, width))
    else:
        lat_grid = latitude
    if longitude.ndim == 1:
        lon_grid = np.tile(longitude, (length, 1))
    else:
        lon_grid = longitude

    # Structure data dictionary with HDFEOS paths
    hdfeos_dict = {
        # Geometry datasets (using NaN placeholders where None is requested)
        'HDFEOS/GRIDS/timeseries/geometry/azimuthAngle': np.full((length, width), np.nan, dtype='float32'),
        'HDFEOS/GRIDS/timeseries/geometry/height': np.full((length, width), np.nan, dtype='float32'),
        'HDFEOS/GRIDS/timeseries/geometry/incidenceAngle': np.full((length, width), np.nan, dtype='float32'),
        'HDFEOS/GRIDS/timeseries/geometry/latitude': lat_grid.astype('float32'),
        'HDFEOS/GRIDS/timeseries/geometry/longitude': lon_grid.astype('float32'),
        'HDFEOS/GRIDS/timeseries/geometry/shadowMask': np.zeros((length, width), dtype='uint8'),
        'HDFEOS/GRIDS/timeseries/geometry/slantRangeDistance': np.full((length, width), np.nan, dtype='float32'),
        # Observation datasets
        'HDFEOS/GRIDS/timeseries/observation/bperp': bperp.astype('float32'),
        'HDFEOS/GRIDS/timeseries/observation/date': date_list.astype('S8'),
        'HDFEOS/GRIDS/timeseries/observation/displacement': ts_data.astype('float32'),
        'HDFEOS/GRIDS/timeseries/observation/delta': delta_days.astype('uint8'),
        # Quality datasets (using NaN placeholders where None is requested)
        'HDFEOS/GRIDS/timeseries/quality/avgSpatialCoherence': np.full((length, width), np.nan, dtype='float32'),
        'HDFEOS/GRIDS/timeseries/quality/mask': mask.astype('bool'),
        'HDFEOS/GRIDS/timeseries/quality/temporalCoherence': np.full((length, width), np.nan, dtype='float32'),
    }

    # Update metadata for HDFEOS format
    hdfeos_metadata = metadata.copy()
    hdfeos_metadata['FILE_TYPE'] = 'HDFEOS'
    hdfeos_metadata['FILE_PATH'] = output_path
    hdfeos_metadata['WIDTH'] = str(width)
    hdfeos_metadata['LENGTH'] = str(length)
    hdfeos_metadata['FILE_TYPE'] = 'HDFEOS'
    hdfeos_metadata['PROCESSOR'] = 'mintpy'
    hdfeos_metadata['PROJECT_NAME'] = os.path.basename(os.path.dirname(output_path))
    hdfeos_metadata['REF_DATE'] = str(date_list[0])
    hdfeos_metadata['diplacement_type'] = 'vertical' if 'vert' in output_path else 'horizontal'

    # Write using writefile.write
    writefile.write(hdfeos_dict, out_file=output_path, metadata=hdfeos_metadata)
    print(f'HDFEOS file created: {output_path}')


def process_reference_points(ts1, ts2, inps):
    """Process reference points for both timeseries.

    Args:
        ts1: First timeseries object
        ts2: Second timeseries object
        window: Dictionary containing window data
        inps: Parsed command line arguments
    """
    # Find reference points from the subsets
    refs_lalo = find_reference_points_from_subsets(ts1.window, ts2.window, inps.window_size)

    # List of timeseries objects and their corresponding reference coordinates
    timeseries_list = [(ts1, refs_lalo[0]), (ts2, refs_lalo[1])]

    for ts, refs in timeseries_list:
        # Compute coordinates and update metadata
        coords = ut.coordinate(ts.metadata).lalo2yx(*refs)
        ts.metadata['REF_LAT'], ts.metadata['REF_LON'], ts.metadata['REF_Y'], ts.metadata['REF_X'] = *refs, *coords

        # Create an empty array for modified data
        modified_data = np.empty_like(ts.data)

        # Process each slice in the timeseries data
        total_slices = len(ts.data)
        print(f'Reference point processing for {ts.metadata["FILE_PATH"]}')
        for i, slice_data in enumerate(ts.data):
            # Print progress
            print(f'\rProcessing slice {i+1}/{total_slices}', end='', flush=True)

            # Extract the reference value for the current slice
            ref_value = slice_data[ts.metadata['REF_Y'], ts.metadata['REF_X']]

            if np.isnan(ref_value):
                raise ValueError(f'Reference value at slice {i} is nan')

            # Subtract the reference value from the entire slice
            modified_data[i] = slice_data - ref_value

        print('')
        # Update the timeseries data with the modified data
        ts.data = modified_data

    return ts1, ts2


def compute_horzvert_timeseries(ts1, ts2, date_list, inps):
    """Compute horizontal and vertical timeseries from ascending/descending data.

    Args:
        ts1: First timeseries object
        ts2: Second timeseries object
        inps: Parsed command line arguments

    Returns:
        tuple: (vertical_timeseries, horizontal_timeseries, mask, latitude, longitude)
    """
    # Calculate the overlapping area in lat/lon
    atr_list = [ts1.metadata, ts2.metadata]
    S, N, W, E = get_overlap_lalo(atr_list)
    lat_step = float(atr_list[0]['Y_STEP'])
    lon_step = float(atr_list[0]['X_STEP'])
    length = int(round((S - N) / lat_step))
    width = int(round((E - W) / lon_step))

    latitude = np.linspace(N, S, length)
    longitude = np.linspace(W, E, width)

    los_inc_angle = np.zeros((2, length, width), dtype=np.float32)
    los_az_angle = np.zeros((2, length, width), dtype=np.float32)
    mask = np.zeros((2, length, width), dtype=np.float32)
    data = np.zeros((2, len(date_list), length, width), dtype=np.float32)

    # Extract overlapping area
    for i, ts in enumerate([ts1, ts2]):
        y0, x0 = ut.coordinate(ts.metadata).lalo2yx(N, W)
        los_inc_angle[i] = ts.los_inc_angle[y0:y0 + length, x0:x0 + width]
        los_az_angle[i] = ts.los_az_angle[y0:y0 + length, x0:x0 + width]
        mask[i] = ts.mask[y0:y0 + length, x0:x0 + width]
        data[i] = np.stack([d[y0:y0 + length, x0:x0 + width] for d in ts.data])

    mask = np.logical_and(mask[0], mask[1])

    # Compute horizontal and vertical components (serial) (old code)
    # vertical_list = []
    # horizontal_list = []
    # for i in range(data.shape[1]):
    #     hvert, dvert = asc_desc2horz_vert(data[:, i], los_inc_angle, los_az_angle, inps.horz_az_angle)
    #     vertical_list.append(dvert)
    #     horizontal_list.append(hvert)

    # vertical_timeseries = np.stack(vertical_list, axis=0)
    # horizontal_timeseries = np.stack(horizontal_list, axis=0)

    # Compute horizontal and vertical components in parallel
    n_times = data.shape[1]
    ncores = detect_cores()
    print(f"Using {ncores} cores for horz/vert decomposition")

    vertical_list = [None] * n_times
    horizontal_list = [None] * n_times

    with ProcessPoolExecutor(
        max_workers=ncores,
        initializer=_init_worker,
        initargs=(data, los_inc_angle, los_az_angle, inps.horz_az_angle),
    ) as pool:
        for i, dvert, hvert in pool.map(_asc_desc_worker, range(n_times)):
            vertical_list[i] = dvert
            horizontal_list[i] = hvert

    vertical_timeseries = np.stack(vertical_list, axis=0)
    horizontal_timeseries = np.stack(horizontal_list, axis=0)

    return vertical_timeseries, horizontal_timeseries, mask, latitude, longitude



def main(iargs=None, namespace=None):
    """Main function to generate vertical and horizontal timeseries."""
    inps = create_parser(iargs, namespace)
    configure_logging(_infer_project_base_dir(inps.file))
    image_pairs_written = False
    symbol_map = {}
    shift_display = {}

    # Always write image_pairs.txt using the fast pairing logic
    parse_periods(inps)

    def run_fast(files):
        dates_meta_local = []
        project_base = None
        for f in files:
            dates, meta, project_base = load_dates(f, inps)
            dates_meta_local.append((dates, meta))
        (d1, m1), (d2, m2) = dates_meta_local
        ts1_f, ts2_f, delta_fast, pairs_fast, block_ranges, block_counts, block_pairs, block_map, shift_map = match_and_filter_pairs(d1, d2, m1, m2, inps)
        mode_val, mode_count = delta_days_max_occurrence(delta_fast)
        return {
            "d1": d1, "d2": d2, "m1": m1, "m2": m2,
            "ts1_f": ts1_f, "ts2_f": ts2_f,
            "delta": delta_fast, "pairs": pairs_fast,
            "block_ranges": block_ranges, "block_counts": block_counts,
            "block_pairs": block_pairs, "block_map": block_map, "shift_map": shift_map,
            "mode_val": mode_val, "mode_count": mode_count,
            "base": project_base,
        }

    res_orig = run_fast(inps.file)
    res_swap = res_orig
    if not inps.no_swap:
        print()
        print("Testing swapped input files:")
        res_swap = run_fast(list(reversed(inps.file)))

    chosen = res_swap if (not inps.no_swap and res_swap["mode_count"] > res_orig["mode_count"]) else res_orig
    if chosen is res_swap and not inps.no_swap:
        inps.file = list(reversed(inps.file))
        if inps.geom_file:
            inps.geom_file = list(reversed(inps.geom_file))

    d1 = chosen["d1"]
    d2 = chosen["d2"]
    m1 = chosen["m1"]
    m2 = chosen["m2"]
    ts1_f = chosen["ts1_f"]
    ts2_f = chosen["ts2_f"]
    delta_fast = chosen["delta"]
    pairs_fast = chosen["pairs"]
    block_ranges = chosen["block_ranges"]
    block_counts = chosen["block_counts"]
    block_pairs = chosen["block_pairs"]
    block_map = chosen["block_map"]
    shift_map = chosen["shift_map"]
    project_base_dir = chosen["base"]

    max_shift = max((max(abs(r[0]), abs(r[1])) for r in block_ranges), default=0)
    diff_msg = describe_shift(ts1_f, ts2_f, m1, m2, limit=max_shift)
    symbols = list("*+-:!@#$%^&():\";'<>,.?/")
    # Assign symbols by signed shift: positives first, then zero, then negatives.
    def _sign_order(val):
        if val > 0:
            return (abs(val), 0)
        if val == 0:
            return (abs(val), 1)
        return (abs(val), 2)

    shifts_sorted = sorted(set(shift_map.values()), key=_sign_order)
    shift_symbol = {s: symbols[i % len(symbols)] for i, s in enumerate(shifts_sorted)}

    symbol_map = {}
    shift_display = {}
    for (da, db), _ in block_map.items():
        d1s = to_date(da).strftime("%Y%m%d")
        d2s = to_date(db).strftime("%Y%m%d")
        s_val = shift_map.get((da, db), 0)
        symbol = shift_symbol.get(s_val, symbols[len(shift_symbol) % len(symbols)])
        symbol_map[(d1s, d2s)] = symbol
        shift_display[(d1s, d2s)] = s_val

    interval_lines = []
    for idx, rng in enumerate(block_ranges):
        rng_str = f"{rng[0]}..{rng[1]}"
        count = block_counts.get(idx, 0)
        interval_lines.append("")
        interval_lines.append(f"Interval {idx+1} [{rng_str}]: {count} pairs")
        for (da, db), bidx in block_map.items():
            if bidx != idx:
                continue
            d1s = to_date(da).strftime("%Y%m%d")
            d2s = to_date(db).strftime("%Y%m%d")
            s_val = shift_display.get((d1s, d2s), 0)
            sign = "+" if s_val > 0 else ""
            sym = symbol_map.get((d1s, d2s), symbols[idx % len(symbols)])
            interval_lines.append(f"{sym}{d1s}  {d2s} ({sign}{s_val})")

    # Summary with counts per signed shift (positive first, then zero, then negative).
    shift_counts = {}
    shift_symbol = {}
    for pair, shift_val in shift_display.items():
        shift_counts[shift_val] = shift_counts.get(shift_val, 0) + 1
        if shift_val not in shift_symbol:
            shift_symbol[shift_val] = symbol_map[pair]
    legend_lines = ["Summary:", f"{_track_label(m1)}: {len(d1)} images, {_track_label(m2)}: {len(d2)} images"]
    total_pairs = 0
    for shift_val in sorted(shift_counts.keys(), key=_sign_order):
        sym = shift_symbol.get(shift_val, symbols[shift_val % len(symbols)] if shift_counts else symbols[0])
        sign = "+" if shift_val > 0 else ""
        count = shift_counts[shift_val]
        total_pairs += count
        pair_txt = "pair" if count == 1 else "pairs"
        legend_lines.append(f"{sym} {sign}{shift_val} days  {count} {pair_txt}")
    legend_lines.append(f"Total: {total_pairs} pairs")

    note_text = diff_msg
    if interval_lines:
        note_text += "\n" + "\n".join(interval_lines)

    print("Writing image_pairs.txt .....")
    for line in _strip_marker_lines(legend_lines, symbols):
        print(line)
    write_date_table(
        d1,
        d2,
        pairs_fast,
        m1,
        m2,
        os.path.join(project_base_dir, "image_pairs.txt"),
        note=note_text,
        pair_symbols=symbol_map,
        pair_shifts=shift_display,
        legend_lines=legend_lines,
    )
    image_pairs_written = True

    if inps.dry_run:
        return

    os.chdir(SCRATCHDIR)

    # Load and process both timeseries files
    project_base_dir = None

    y_step = None
    x_step = None

    timseries = []

    for idx, f in enumerate(inps.file):
        geometry_file_input = inps.geom_file[idx] if inps.geom_file and idx < len(inps.geom_file) else None
        obj, los_inc_angle, los_az_angle, mask, project_base_dir, geometry_file = load_timeseries_file(f, geometry_file_input, inps.mask_vmin[idx], inps)

        # Check step consistency from previous iteration
        if 'Y_STEP' in obj.metadata:
            if y_step is not None and x_step is not None:
                if (y_step != obj.metadata['Y_STEP']) or (x_step != obj.metadata['X_STEP']):
                    print('-' * 50)
                    raise ValueError('Files have different steps size for Geocoding')

        # Geocode if needed
        obj, los_inc_angle, los_az_angle, mask, y_step, x_step = geocode_timeseries(obj, los_inc_angle, los_az_angle, mask, geometry_file, inps)

        # Prepare data for reference point selection
        slicedata = obj.data[0]
        stack = np.nanmean(obj.data, axis=0)
        mask2 = np.multiply(~np.isnan(stack), stack != 0.)
        combined_mask = np.logical_and(mask, mask2)
        masked_data = np.where(combined_mask, slicedata, np.nan)
        extracted_data = extract_window(masked_data, obj.metadata, inps.ref_lalo[0], inps.ref_lalo[1], inps.window_size)

        # window[obj.metadata['FILE_PATH']] = {'obj': obj, 'data': extracted_data}
        obj.window = extracted_data
        obj.los_inc_angle = los_inc_angle
        obj.los_az_angle = los_az_angle
        obj.mask = mask

        timseries.append(obj)

    # Process reference points
    ts1, ts2 = process_reference_points(*timseries, inps)
    original_ts1_dates = list(ts1.dateList)
    original_ts2_dates = list(ts2.dateList)

    # Match and filter dates
    ts1, ts2, delta_days, bperp, date_list, pairs, interval_summary = match_and_filter_dates(ts1, ts2, inps)
    ts1.metadata['REF_DATELIST_FILE'] = ts1.metadata['FILE_PATH']
    block_ranges, block_counts, block_pairs, block_map, shift_map = interval_summary
    max_shift = max((max(abs(r[0]), abs(r[1])) for r in block_ranges), default=12)
    diff_msg = describe_shift(ts1.dateList, ts2.dateList, ts1.metadata, ts2.metadata, limit=max_shift)
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
            interval_lines.append(f"{sign if sign else ''}{d1}  {d2} ({sign}{s_val})")
    # Build symbol/shift map per pair using interval index
    symbol_map = {}
    shift_display = {}
    for (da, db), block_idx in block_map.items():
        d1 = to_date(da).strftime("%Y%m%d")
        d2 = to_date(db).strftime("%Y%m%d")
        symbol_map[(d1, d2)] = symbols[block_idx % len(symbols)]
        shift_display[(d1, d2)] = shift_map.get((da, db), 0)

    # Legend for symbols
    # Only write image_pairs.txt once (already written in fast path)
    if not image_pairs_written:
        # Summary with counts per signed shift (positive first, then zero, then negative).
        def _sign_order(val):
            if val > 0:
                return (abs(val), 0)
            if val == 0:
                return (abs(val), 1)
            return (abs(val), 2)

        shift_counts = {}
        shift_symbol = {}
        for pair, shift_val in shift_display.items():
            shift_counts[shift_val] = shift_counts.get(shift_val, 0) + 1
            if shift_val not in shift_symbol:
                shift_symbol[shift_val] = symbol_map[pair]

        legend_lines = ["Summary:", f"{_track_label(ts1.metadata)}: {len(original_ts1_dates)} images, {_track_label(ts2.metadata)}: {len(original_ts2_dates)} images"]
        total_pairs = 0
        for shift_val in sorted(shift_counts.keys(), key=_sign_order):
            sym = shift_symbol.get(shift_val, symbols[shift_val % len(symbols)] if shift_counts else symbols[0])
            sign = "+" if shift_val > 0 else ""
            count = shift_counts[shift_val]
            total_pairs += count
            pair_txt = "pair" if count == 1 else "pairs"
            legend_lines.append(f"{sym} {sign}{shift_val} days  {count} {pair_txt}")
        legend_lines.append(f"Total: {total_pairs} pairs")

        print("Writing image_pairs.txt .....")
        for line in _strip_marker_lines(legend_lines, symbols):
            print(line)
        write_date_table(original_ts1_dates, original_ts2_dates, pairs, ts1.metadata, ts2.metadata, os.path.join(project_base_dir, "image_pairs.txt"), note=diff_msg, extra_lines=interval_lines, pair_symbols=symbol_map, pair_shifts=shift_display, legend_lines=legend_lines)

    # Compute horizontal and vertical timeseries
    vertical_timeseries, horizontal_timeseries, mask, latitude, longitude = compute_horzvert_timeseries(ts1, ts2, date_list, inps)
    ts1.metadata['relative_orbit_second'] = ts2.metadata['relative_orbit']
    ts1.metadata['ORBIT_DIRECTION_SECOND'] = ts2.metadata['ORBIT_DIRECTION']
    # Derive start/end dates from the paired date_list (horz/vert range).
    date_objs = [to_date(d) for d in date_list]
    ts1.metadata['first_date'] = min(date_objs).strftime('%Y-%m-%d')
    ts1.metadata['last_date'] = max(date_objs).strftime('%Y-%m-%d')

    # Create output files

    vertical_path = os.path.join(project_base_dir, get_output_filename(ts1.metadata, None, direction='vert'))
    horizontal_path = os.path.join(project_base_dir, get_output_filename(ts1.metadata, None, direction='horz'))

    mask_path = os.path.join(project_base_dir, 'maskTempCoh.h5')

    if inps.timeseries:
        # create_timeseries_output(vertical_timeseries, date_list, mask, delta_days, bperp, latitude, longitude, ts1.metadata, vertical_path.replace('.he5', '.h5'), 'timeseries')
        # create_timeseries_output(horizontal_timeseries, date_list, mask, delta_days, bperp, latitude, longitude, ts1.metadata, horizontal_path.replace('.he5', '.h5'), 'timeseries')
        create_timeseries_output(vertical_timeseries, date_list, mask, delta_days, bperp, latitude, longitude, ts1.metadata, os.path.join(project_base_dir, 'vert_timeseries.h5'), 'timeseries')
        create_timeseries_output(horizontal_timeseries, date_list, mask, delta_days, bperp, latitude, longitude, ts1.metadata, os.path.join(project_base_dir, 'horz_timeseries.h5'), 'timeseries')

    create_hdfeos_output(vertical_timeseries, date_list, mask, delta_days, bperp, latitude, longitude,
                     ts1.metadata, vertical_path.replace('.h5', '.he5'), mask.shape[0], mask.shape[1])
    create_hdfeos_output(horizontal_timeseries, date_list, mask, delta_days, bperp, latitude, longitude,
                     ts1.metadata, horizontal_path.replace('.h5', '.he5'), mask.shape[0], mask.shape[1])

    # Write mask file
    mask_meta = {
        'FILE_TYPE': 'mask',
        'LENGTH': str(mask.shape[0]),
        'WIDTH': str(mask.shape[1])
    }

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    if not os.path.exists(mask_path) or inps.overwrite:
        writefile.write({'mask': mask.astype('bool')}, out_file=mask_path, metadata=mask_meta)


if __name__ == "__main__":
    main()
