#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from scipy.stats import norm, skew
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from mintpy.utils import readfile, utils as ut, writefile
from mintpy.objects.resample import resample
from mintpy.objects import timeseries, HDFEOS
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert, get_overlap_lalo
from plotdata.helper_functions import (get_file_names, prepend_scratchdir_if_needed, extract_window, find_reference_points_from_subsets, select_reference_point, create_geometry_file, find_longitude_degree)


SCRATCHDIR = os.getenv('SCRATCHDIR')
EXAMPLE = f"""
SOMEt
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

    if len(lalo) == 1:     # if given as one string containing ','
        lalo = lalo[0]
    return lalo


def match_dates(list1, list2):
    # Convert dates to integers for easier comparison
    indexed_list1 = list(enumerate(map(int, list1)))
    indexed_list2 = list(enumerate(map(int, list2)))

    matched_indexes1 = set()
    matched_indexes2 = set()

    # Match dates
    for index1, date1 in indexed_list1:
        closest_index2, closest_date2 = min(
            indexed_list2,
            key=lambda x: abs(x[1] - date1)
        )
        matched_indexes1.add(index1)
        matched_indexes2.add(closest_index2)
        indexed_list2.remove((closest_index2, closest_date2))  # Remove matched element

    # Find removed indexes
    removed_indexes1 = [index for index, _ in indexed_list1 if index not in matched_indexes1]
    removed_indexes2 = [index for index, _ in indexed_list2 if index not in matched_indexes2]

    return removed_indexes1, removed_indexes2


def match_and_get_dropped_indices(a, b):
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

    # Return matched lists and dropped indices
    return dropped_a, dropped_b


def main(iargs=None, namespace=None):

    inps = create_parser()
    window = {}
    y_step = None
    x_step = None

    for f in inps.file:
        work_dir = prepend_scratchdir_if_needed(f)
        eos_file, _, geometry_file, project_base_dir, out_vel_file, inputs_folder = get_file_names(work_dir)

        metadata = readfile.read_attribute(eos_file)

        if not inps.ref_lalo:
            inps.ref_lalo = parse_lalo([metadata['REF_LAT'], metadata['REF_LON']])

        # TODO add overwrite option
        if not os.path.exists(geometry_file) or True:
            create_geometry_file(eos_file, os.path.dirname(geometry_file))

        # Identify file type and open it
        if metadata['FILE_TYPE'] == 'timeseries':
            obj = timeseries(eos_file)
            los_inc_angle = ut.incidence_angle(metadata, dimension=0, print_msg=False)
            los_az_angle = ut.heading2azimuth_angle(float(metadata['HEADING']), look_direction='right')
        elif metadata['FILE_TYPE'] == 'HDFEOS':
            hdf_obj = HDFEOS(eos_file)
            hdf_obj.open()
            obj = timeseries()
            obj.data = hdf_obj.read()
            obj.dateList = hdf_obj.dateList
            obj.metadata = hdf_obj.metadata
            los_inc_angle = readfile.read(geometry_file, datasetName='incidenceAngle')[0]
            los_az_angle  = readfile.read(geometry_file, datasetName='azimuthAngle')[0]
            mask = readfile.read(eos_file, datasetName='mask')[0]
            hdf_obj.close()
        else:
            raise ValueError(f'Input file is {metadata["FILE_TYPE"]}, not timeseries.')

        obj.metadata['FILE_TYPE'] = 'timeseries'

        # GEOCODING
        if 'Y_STEP' not in obj.metadata:
            # coord.coordinate(atr, geometry_file).radar2geo(data)
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
            obj.metadata['Y_FIRST'], obj.metadata['X_FIRST'] = np.nanmax(readfile.read(geometry_file, datasetName='latitude')[0]), np.nanmin(readfile.read(geometry_file, datasetName='longitude')[0])
        else:
            y_step = y_step if y_step else obj.metadata['Y_STEP']
            x_step = x_step if x_step else obj.metadata['X_STEP']

            if (y_step != obj.metadata['Y_STEP']) or (x_step != obj.metadata['X_STEP']):
                print('-' * 50)
                raise ValueError('Files have different steps size for Geocoding')

        slicedata = obj.data[0]

    # ------------------- for REFERENCING -------------------- #
        # Compute the second mask based on the temporal average
        stack = np.nanmean(obj.data, axis=0)
        mask2 = np.multiply(~np.isnan(stack), stack != 0.)

        # TODO removed temporalCOherence for masking, using actual mask in the file
        # combined_mask = np.logical_and(mask > inps.mask_vmin, mask2)
        combined_mask = np.logical_and(mask > inps.mask_vmin, mask2)

        # Apply the combined mask to the data
        masked_data = np.where(combined_mask, slicedata, np.nan)
        extracted_data = extract_window(masked_data, obj.metadata, inps.ref_lalo[0], inps.ref_lalo[1], inps.window_size)

        window[obj] = {}
        window[obj]['data'] = extracted_data
        obj.los_inc_angle = los_inc_angle
        obj.los_az_angle = los_az_angle
        obj.mask = mask

# ------------------- REFERENCING -------------------- #

    # Extract keys from the window dictionary
    ts1, ts2 = window.keys()

    # Find reference points from the subsets
    refs_lalo = find_reference_points_from_subsets(window[ts1]['data'], window[ts2]['data'], inps.window_size)

    # List of timeseries objects and their corresponding reference coordinates
    timeseries_list = [(ts1, refs_lalo[0]), (ts2, refs_lalo[1])]

    for ts, refs in timeseries_list:
        # Compute coordinates and update metadata
        coords = ut.coordinate(ts.metadata).lalo2yx(*refs)
        ts.metadata['REF_LAT'], ts.metadata['REF_LON'], ts.metadata['REF_Y'], ts.metadata['REF_X'] = *refs, *coords

        # Create an empty array for modified data
        modified_data = np.empty_like(ts.data)

        # Process each slice in the timeseries data
        for i, slice in enumerate(ts.data):
            # Extract the reference value for the current slice
            ref_value = slice[ts.metadata['REF_Y'], ts.metadata['REF_X']]

            if np.isnan(ref_value):
                raise ValueError(f'Reference value at slice {i} is nan')

            # Subtract the reference value from the entire slice
            modified_data[i] = slice - ref_value

        # Update the timeseries data with the modified data
        ts.data = modified_data

# ----------------------------------------------------- #

# ------------------- DATE MATCHING -------------------- #
    ## Find index for non-matching periods
    # index1, index2 = match_dates(ts1.dateList, ts2.dateList)
    index1, index2 = match_and_get_dropped_indices(list(map(lambda x: int(x), ts1.dateList)), list(map(lambda x: int(x), ts2.dateList)))

    ##Remove non-matching dates
    for i, ts in zip([index1, index2], [ts1, ts2]):
        ts.dateList = np.delete(ts.dateList, i)
        mask = np.ones(ts.data.shape[0], dtype=bool)
        mask[i] = False
        ts.data = ts.data[mask]

    print('-' * 50)
    print(f'New date list length: {len(ts1.dateList)}\n')

# ------------------- DATE FILTERING -------------------- #

    # Calculate differences and find indexes where the absolute difference is less than 30
    diff = [(datetime.strptime(x, "%Y%m%d").date() - datetime.strptime(y, "%Y%m%d").date()).days for x, y in zip(ts1.dateList, ts2.dateList)]

    # Find indexes where the absolute difference is less than 30
    valid_indexes = [i for i, dif in enumerate(diff) if abs(dif) < 30]

    # Convert differences to absolute values
    differences = list(map(lambda x: abs(diff[x]), valid_indexes))

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

    # Find indexes where the absolute difference is less than 30
    valid_indexes = [i for i, dif in enumerate(diff) if abs(dif) <= dynamic_threshold]

    print('-' * 50)
    print(f'New date list length after drop: {len(valid_indexes)}\n')

    ts1.data = ts1.data[valid_indexes, :]
    ts2.data = ts2.data[valid_indexes, :]

    delta = np.array([(datetime.strptime(y, "%Y%m%d").date() - datetime.strptime(x, "%Y%m%d").date()).days for x, y in zip(ts1.dateList[valid_indexes], ts2.dateList[valid_indexes])])
    ts.dateList = ts1.dateList[valid_indexes]

# ----------------------------------------------------- #

# ------------------- VERTICAL -------------------- #
    ## 1. calculate the overlapping area in lat/lon
    atr_list = [ts1.metadata, ts2.metadata]
    S, N, W, E = get_overlap_lalo(atr_list)
    lat_step = float(atr_list[0]['Y_STEP'])
    lon_step = float(atr_list[0]['X_STEP'])
    length = int(round((S - N) / lat_step))
    width  = int(round((E - W) / lon_step))

    los_inc_angle = np.zeros((2, length, width), dtype=np.float32)
    los_az_angle  = np.zeros((2, length, width), dtype=np.float32)
    mask  = np.zeros((2, length, width), dtype=np.float32)
    data = np.zeros((2, len(ts1.dateList), length, width), dtype=np.float32)

    # Extact overlapping
    for i, ts in enumerate([ts1, ts2]):
        y0, x0 = ut.coordinate(ts.metadata).lalo2yx(N, W)
        los_inc_angle[i] = ts.los_inc_angle[y0:y0 + length, x0:x0 + width]
        los_az_angle[i] = ts.los_az_angle[y0:y0 + length, x0:x0 + width]
        mask[i] = ts.mask[y0:y0 + length, x0:x0 + width]
        data[i] = np.stack([d[y0:y0 + length, x0:x0 + width] for d in ts.data])

    mask = np.logical_and(mask[0], mask[1])

    vertical_list = []
    horizontal_list = []
    for i in range(data.shape[1]):
        hvert, dvert = asc_desc2horz_vert(data[:, i], los_inc_angle, los_az_angle, inps.horz_az_angle)
        vertical_list.append(dvert)
        horizontal_list.append(hvert)

    vertical_timeseries = np.stack(vertical_list, axis=0)
    horizontal_timeseries = np.stack(horizontal_list, axis=0)

# ------------------- MAKE FILE -------------------- #

    vts = timeseries()
    vts.dateList = ts1.dateList
    vts.metadata = ts1.metadata
    vts.metadata['WIDTH'], vts.metadata['LENGTH'] = width, length
    vts.metadata['xmax'], vts.metadata['ymax'] = width, length
    vts.metadata['FILE_PATH'] = os.path.join(project_base_dir, 'up_timeseries.h5')
    vts.metadata['FILE_TYPE'] = 'timeseries'
    vts.metadata['PROCESSOR'] = 'mintpy'
    vts.metadata['maskFile'] = vts.metadata['FILE_PATH']
    vts.metadata['PROJECT_NAME'] = os.path.basename(project_base_dir)
    vts.metadata['REF_DATE'] = str(ts1.dateList[0])
    vts.metadata.pop('ORBIT_DIRECTION')


    ts_dict = {
        'timeseries': vertical_timeseries.astype('float32'),
        'date': ts.dateList.astype('S8'),
        'mask': mask.astype('uint8'), 
        'delta': delta.astype('float32'),
    }


    writefile.write(ts_dict, vts.metadata['FILE_PATH'], metadata = vts.metadata)

    hts = timeseries()
    hts.dateList = ts1.dateList
    hts.metadata = ts1.metadata
    hts.metadata['WIDTH'], hts.metadata['LENGTH'] = width, length
    hts.metadata['xmax'], hts.metadata['ymax'] = width, length
    hts.metadata['FILE_PATH'] = os.path.join(project_base_dir, 'hz_timeseries.h5')
    hts.metadata['FILE_TYPE'] = 'timeseries'
    hts.metadata['PROCESSOR'] = 'mintpy'
    hts.metadata['maskFile'] = hts.metadata['FILE_PATH']
    hts.metadata['PROJECT_NAME'] = os.path.basename(project_base_dir)
    hts.metadata['REF_DATE'] = str(ts1.dateList[0])

    ts_dict = {
        'timeseries': horizontal_timeseries.astype('float32'),
        'date': ts.dateList.astype('S8'),
        'mask': mask.astype('uint8'),
        'delta': delta.astype('float32'),
    }

    writefile.write(ts_dict, hts.metadata['FILE_PATH'], metadata = hts.metadata)

if __name__ == "__main__":
    main()