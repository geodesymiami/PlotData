#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
from typing import Any
from scipy.stats import skew
from datetime import datetime
from types import SimpleNamespace
from mintpy.objects.resample import resample
from mintpy.objects import timeseries, HDFEOS
from scipy.optimize import linear_sum_assignment
from mintpy.save_hdfeos5 import get_output_filename
from mintpy.utils import readfile, utils as ut, writefile
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert, get_overlap_lalo
from plotdata.helper_functions import (
    get_file_names, prepend_scratchdir_if_needed, extract_window,
    find_reference_points_from_subsets, create_geometry_file, find_longitude_degree
)


SCRATCHDIR = os.getenv('SCRATCHDIR')
EXAMPLE = """
Example usage:
    horzvert_timeseries.py ChilesSenAT120/mintpy ChilesSenDT142/mintpy --ref-lalo 0.84969 -77.86430
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

    parser.add_argument('file', nargs=2, help='Ascending and descending files\n' 'Both files need to be geocoded in the same spatial resolution.')
    parser.add_argument('-g','--geom-file', dest='geom_file', nargs=2, help='Geometry files for the input data files.')
    parser.add_argument('--mask-thresh', dest='mask_vmin', type=float, default=0.55, help='coherence threshold for masking (default: %(default)s).')
    parser.add_argument('--ref-lalo', nargs='*', metavar=('LATITUDE,LONGITUDE or LATITUDE LONGITUDE'), default=None, type=str, help='reference point (default: existing reference point)')
    parser.add_argument('--lat-step', dest='lat_step', type=float, default=-0.0002, help='latitude step for geocoding (default: %(default)s).')
    parser.add_argument('--horz-az-angle', dest='horz_az_angle', type=float, default=90, help='Horizontal azimuth angle (default: %(default)s).')
    parser.add_argument('--window-size', dest='window_size', type=int, default=3, help='window size (square side in number of pixels) for reference point look up (default: %(default)s).')
    parser.add_argument('--date-filtering', dest='date_thresh_method', type=str, default='min', choices=['min', 'percentile'], help='Method for date difference threshold: "min" uses minimum difference, "percentile" uses percentile-based threshold (default: %(default)s).')
    parser.add_argument('-ow', '--overwrite', dest='overwrite', action='store_true', help='Overwrite all previously generated files')
    parser.add_argument('-ts', '--timeseries', dest='timeseries', action='store_true', help='Output timeseries file in addition to HDFEOS format')

    inps = parser.parse_args(iargs, namespace)

    if inps.ref_lalo:
        inps.ref_lalo = parse_lalo(inps.ref_lalo)

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


def match_and_get_dropped_indices(a, b):
    """Match two date lists and return indices of unmatched dates.

    Args:
        a: First list of dates (as integers)
        b: Second list of dates (as integers)

    Returns:
        tuple: (dropped_indices_a, dropped_indices_b)
    """
    a = np.array(a)
    b = np.array(b)

    # Compute pairwise absolute differences
    cost = np.abs(a[:, None] - b[None, :])

    # Find optimal assignment
    i, j = linear_sum_assignment(cost)

    # Handle different lengths: only match the min length
    n = min(len(a), len(b))
    matched_i = i[:n]
    matched_j = j[:n]

    # Compute dropped indices
    dropped_a = sorted(set(range(len(a))) - set(matched_i))
    dropped_b = sorted(set(range(len(b))) - set(matched_j))

    return dropped_a, dropped_b


def load_timeseries_file(file_path, geometry_file_input, inps):
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
    if not os.path.exists(geometry_file) or inps.overwrite:
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

    # Read mask (same for both file types)
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


def match_and_filter_dates(ts1, ts2, thresh_method='min'):
    """Match dates between two timeseries and filter by date difference.

    Args:
        ts1: First timeseries object
        ts2: Second timeseries object
        thresh_method: Method for threshold calculation ('min' or 'percentile')

    Returns:
        tuple: (filtered_ts1, filtered_ts2, delta, bperp, date_list)
    """
    # Match dates and get dropped indices
    date_list1 = [int(x) for x in ts1.dateList]
    date_list2 = [int(x) for x in ts2.dateList]
    index1, index2 = match_and_get_dropped_indices(date_list1, date_list2)

    # Remove non-matching dates
    for i, ts in zip([index1, index2], [ts1, ts2]):
        ts.dateList = np.delete(ts.dateList, i)
        mask = np.ones(ts.data.shape[0], dtype=bool)
        mask[i] = False
        ts.data = ts.data[mask]

    print('-' * 50)
    print(f'New date list length: {len(ts1.dateList)}\n')

    # Calculate date differences
    diff = [(datetime.strptime(x, "%Y%m%d").date() - datetime.strptime(y, "%Y%m%d").date()).days for x, y in zip(ts1.dateList, ts2.dateList)]

    # Calculate threshold based on selected method
    if thresh_method == 'min':
        # Use minimum difference as threshold
        dynamic_threshold = min(np.abs(np.array(diff)))
        print('-' * 50)
        print(f"Minimum threshold value: {dynamic_threshold}\n")
    elif thresh_method == 'percentile':
        # Find indexes where the absolute difference is less than 30 (initial filter)
        valid_indexes = [i for i, dif in enumerate(diff) if abs(dif) < 30]

        # Convert differences to absolute values
        differences = [abs(diff[i]) for i in valid_indexes]

        if differences:
            data_skewness = skew(differences)

            # Dynamically determine the percentile threshold
            if data_skewness > 1:  # Highly skewed data
                percentile_threshold = 90  # Use a stricter threshold
            elif data_skewness < -1:  # Left-skewed data (unlikely for absolute differences)
                percentile_threshold = 99
            else:  # Symmetric or moderately skewed data
                percentile_threshold = 95

            dynamic_threshold = np.percentile(differences, percentile_threshold)
            print('-' * 50)
            print(f"Dynamic Threshold for date difference (value at {percentile_threshold}th percentile): {dynamic_threshold}\n")
        else:
            # Fallback to minimum if no valid differences found
            dynamic_threshold = min(np.abs(np.array(diff)))
            print('-' * 50)
            print(f"No valid differences found, using minimum threshold: {dynamic_threshold}\n")

    # Filter by threshold
    valid_indexes = [i for i, dif in enumerate(diff) if abs(dif) <= dynamic_threshold]
    print('-' * 50)
    print(f'New date list length after drop: {len(valid_indexes)}\n')

    # Apply filtering
    ts1.data = ts1.data[valid_indexes, :]
    ts2.data = ts2.data[valid_indexes, :]
    bperp = ts1.bperp[valid_indexes]
    date_list = ts1.dateList[valid_indexes]

    # Calculate delta (date differences for valid indexes)
    delta = np.array([
        (datetime.strptime(y, "%Y%m%d").date() - datetime.strptime(x, "%Y%m%d").date()).days
        for x, y in zip(ts1.dateList[valid_indexes], ts2.dateList[valid_indexes])
    ])

    return ts1, ts2, delta, bperp, date_list


def create_timeseries_output(ts_data, date_list, mask, delta, bperp, latitude, longitude,
                             metadata, output_path, file_type='timeseries'):
    """Create and write a timeseries output file.

    Args:
        ts_data: Timeseries data array
        date_list: List of dates
        mask: Mask array
        delta: Date difference array
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
        'delta': delta.astype('uint8'),
        'bperp': bperp.astype('float32'),
        'latitude': latitude.astype('float32'),
        'longitude': longitude.astype('float32')
    }

    writefile.write(ts_dict, output_path, metadata=output_metadata)


def create_hdfeos_output(ts_data, date_list, mask, delta, bperp, latitude, longitude,
                         metadata, output_path, length, width):
    """Create and write an HDFEOS output file with proper structure.

    Args:
        ts_data: Timeseries data array (n_time, length, width)
        date_list: List of dates
        mask: Mask array (length, width)
        delta: Date difference array
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
        'HDFEOS/GRIDS/timeseries/observation/delta': delta.astype('uint8'),
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

    # Compute horizontal and vertical components
    vertical_list = []
    horizontal_list = []
    for i in range(data.shape[1]):
        hvert, dvert = asc_desc2horz_vert(data[:, i], los_inc_angle, los_az_angle, inps.horz_az_angle)
        vertical_list.append(dvert)
        horizontal_list.append(hvert)

    vertical_timeseries = np.stack(vertical_list, axis=0)
    horizontal_timeseries = np.stack(horizontal_list, axis=0)

    return vertical_timeseries, horizontal_timeseries, mask, latitude, longitude


def main(iargs=None, namespace=None):
    """Main function to generate vertical and horizontal timeseries."""
    inps = create_parser(iargs, namespace)
    os.chdir(SCRATCHDIR)

    # Load and process both timeseries files
    project_base_dir = None

    y_step = None
    x_step = None

    timseries = []

    for idx, f in enumerate(inps.file):
        geometry_file_input = inps.geom_file[idx] if inps.geom_file and idx < len(inps.geom_file) else None
        obj, los_inc_angle, los_az_angle, mask, project_base_dir, geometry_file = load_timeseries_file(f, geometry_file_input, inps)

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
        combined_mask = np.logical_and(mask > inps.mask_vmin, mask2)
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

    # Match and filter dates
    ts1, ts2, delta, bperp, date_list = match_and_filter_dates(ts1, ts2, inps.date_thresh_method)
    ts1.metadata['REF_DATELIST_FILE'] = ts1.metadata['FILE_PATH']

    # Compute horizontal and vertical timeseries
    vertical_timeseries, horizontal_timeseries, mask, latitude, longitude = compute_horzvert_timeseries(ts1, ts2, date_list, inps)
    ts1.metadata['relative_orbit_second'] = ts2.metadata['relative_orbit']
    ts1.metadata['ORBIT_DIRECTION_SECOND'] = ts2.metadata['ORBIT_DIRECTION']

    # Create output files
    vertical_path = os.path.join(project_base_dir, get_output_filename(ts1.metadata, None, suffix='vert'))
    horizontal_path = os.path.join(project_base_dir, get_output_filename(ts1.metadata, None, suffix='horz'))
    mask_path = os.path.join(project_base_dir, 'maskTempCoh.h5')

    if inps.timeseries:
        create_timeseries_output(vertical_timeseries, date_list, mask, delta, bperp, latitude, longitude, ts1.metadata, vertical_path.replace('.he5', '.h5'), 'timeseries')

        create_timeseries_output(horizontal_timeseries, date_list, mask, delta, bperp, latitude, longitude, ts1.metadata, horizontal_path.replace('.he5', '.h5'), 'timeseries')

    create_hdfeos_output(vertical_timeseries, date_list, mask, delta, bperp, latitude, longitude,
                         ts1.metadata, vertical_path.replace('.h5', '.he5'), mask.shape[0], mask.shape[1])

    create_hdfeos_output(horizontal_timeseries, date_list, mask, delta, bperp, latitude, longitude,
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

    for path in [os.path.join(project_base_dir, 'log') if project_base_dir else None, os.path.join(os.getenv('SCRATCHDIR'), 'log') if os.getenv('SCRATCHDIR') else None]:
        if not path:
            continue
        # Ensure parent directory exists so logging can create the file
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        logging.basicConfig(filename=path, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d')

    # Log the command-line command
    cmd_command = ' '.join(sys.argv)
    logging.info(cmd_command)


if __name__ == "__main__":
    main()