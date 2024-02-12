import os
import h5py
import numpy as np
import geopandas as gpd
import contextily as ctx
import georaster
from pathlib import Path
import matplotlib.pyplot as plt
import simplekml
import zipfile
from PIL import Image
from shapely.geometry import box
from mintpy.utils import readfile, utils as ut
from mintpy import save_kmz

def change_reference_point(data, attr, ref_lalo, file_type):
    """Change reference point of data to ref_lalo"""
    ref_lat = ref_lalo[0]
    ref_lon = ref_lalo[1]
    point_lalo = np.array([ref_lat, ref_lon])
    if file_type == 'HDFEOS':             # for data in radar coordinates (different for SARPROZ, Andreas)
        coord = ut.coordinate(attr, lookup_file='inputs/geometryRadar.h5')   # radar coord
        ref_y, ref_x = coord.geo2radar(point_lalo[0], point_lalo[1])[:2]
        if data.ndim == 2:
            data -= data[ref_y, ref_x]
        elif data.ndim == 3:
            for i in range(data.shape[0]):
                data[i] -= data[i, ref_y, ref_x]
    return data

def extract_data_at_point(timeseries, attr, lalo, file_type):
    """Extract timeseries at point"""
    lat = lalo[0]
    lon = lalo[1]
    point_lalo = np.array([lat, lon])
    if file_type == 'HDFEOS':             # for data in radar coordinates (different for SARPROZ, Andreas)
        coord = ut.coordinate(attr, lookup_file='inputs/geometryRadar.h5')   # radar coord
        y, x = coord.geo2radar(point_lalo[0], point_lalo[1])[:2]
        timeseries_at_point = timeseries[:, y, x]
    return timeseries_at_point

def extract_subset_from_data(inps, plot_box_dict):
    """Extract subset from data"""
    mask_extract = np.ones(inps.displacement.shape, dtype=np.float32)
    mask_extract[inps.lat < plot_box_dict['lat1']] = 0
    mask_extract[inps.lat > plot_box_dict['lat2']] = 0
    mask_extract[inps.lon < plot_box_dict['lon1']] = 0
    mask_extract[inps.lon > plot_box_dict['lon2']] = 0

    inps.displacement = np.array(inps.displacement[mask_extract == 1])
    inps.velocity = np.array(inps.velocity[mask_extract == 1])
    inps.dem_error = np.array(inps.dem_error[mask_extract == 1])
    inps.elevation = np.array(inps.elevation[mask_extract == 1]) 
    inps.height = np.array(inps.height[mask_extract == 1])
    inps.lat = np.array(inps.lat[mask_extract == 1])
    inps.lon = np.array(inps.lon[mask_extract == 1])
    inps.inc_angle = np.array(inps.inc_angle[mask_extract == 1])
    inps.data = np.array(inps.data[mask_extract == 1])

    return

def correct_geolocation(inps):
    """Correct the geolocation using DEM error"""
    print('Run geolocation correction ...')

    latitude = inps.lat
    longitude = inps.lon
    dem_error = inps.dem_error
    inc_angle = np.deg2rad(inps.inc_angle)
    # az_angle = np.deg2rad(float(inps.HEADING))    # this is what MiaplPy is using
    az_angle = np.deg2rad( -inps.az_angle - 270)    # FA 2/2024:  This works for ascending but not sure whether OK for descending orbits.

    rad_latitude = np.deg2rad(latitude)

    one_degree_latitude = 111132.92 - 559.82 * np.cos(2*rad_latitude) + \
                            1.175 * np.cos(4 * rad_latitude) - 0.0023 * np.cos(6 * rad_latitude)

    one_degree_longitude = 111412.84 * np.cos(rad_latitude) - \
                            93.5 * np.cos(3 * rad_latitude) + 0.118 * np.cos(5 * rad_latitude)

    print('one_degree_latitude, one_degree_longitude:', np.mean(one_degree_latitude), np.mean(one_degree_longitude))

    dx = np.divide((dem_error) * (1 / np.tan(inc_angle)) * np.cos(az_angle), one_degree_longitude)  # converted to degree
    dy = np.divide((dem_error) * (1 / np.tan(inc_angle)) * np.sin(az_angle), one_degree_latitude)  # converted to degree

    sign = np.sign(latitude)
    latitude += sign * dy

    sign = np.sign(longitude)
    longitude += sign * dx

    inps.lat = latitude
    inps.lon = longitude    
    return 

def calculate_mean_amplitude(slcStack, out_amplitude):
    """
    Calculate the mean amplitude from the SLC stack and save it to a file.

    Args:
        slcStack (str): Path to the SLC stack file.
        out_amplitude (str): Path to the output amplitude file.

    Returns:
        None
    """

    with h5py.File(slcStack, 'r') as f:
        slcs = f['slc']
        s_shape = slcs.shape
        mean_amplitude = np.zeros((s_shape[1], s_shape[2]), dtype='float32')
        lines = np.arange(0, s_shape[1], 100)

        for t in lines:
            last = t + 100
            if t == lines[-1]:
                last = s_shape[1]  # Adjust the last index for the final block

            # Calculate mean amplitude for the current block
            mean_amplitude[t:last, :] = np.mean(np.abs(f['slc'][:, t:last, :]), axis=0)

        # Save the calculated mean amplitude to the output file
        np.save(out_amplitude, mean_amplitude)

def default_backscatter_file():
    options = ['mean_amplitude.npy', '../inputs/slcStack.h5']
    for option in options:
        if os.path.exists(option):
            print(f'Using {option} for backscatter.')
            return option
    raise FileNotFoundError(f'USER ERROR: No backscatter file found {options}.')