#! /usr/bin/env python3
import os
import math
import subprocess
import glob
from mintpy.utils import readfile
from mintpy.objects import HDFEOS
from scipy.interpolate import interp1d
import numpy as np
from pathlib import Path

EXAMPLE = """example:
  plot_data.py  MaunaLoaSenDT87 MaunaLoaSenAT124
  plot_data.py  MaunaLoaSenDT87
"""


def get_file_names(path):
    """gets the youngest eos5 file. Path can be:
    MaunaLoaSenAT124
    MaunaLoaSenAT124/mintpy/S1_qq.he5
    ~/onedrive/scratch/MaunaLoaSenAT124/mintpy/S1_qq.he5'
    """
    scratch = os.getenv('SCRATCHDIR')
    if os.path.isfile(glob.glob(path)[0]):
        eos_file = glob.glob(path)[0]

    elif os.path.isfile(os.path.join(scratch, path)):
        eos_file = scratch + '/' + path

    else:
        if 'mintpy' in path or 'network' in path :
            files = glob.glob(path + '/*.he5')

        else:
            files = glob.glob( path + '/mintpy/*.he5' )

        if len(files) == 0:
            raise Exception('USER ERROR: No HDF5EOS files found in ' + path)

        eos_file = max(files, key=os.path.getctime)

    print('HDF5EOS file used:', eos_file)

    metadata = readfile.read(eos_file)[1]
    velocity_file = 'geo/geo_velocity.h5'
    geometryRadar_file = 'geo/geo_geometryRadar.h5'

    # Check if geocoded
    if 'Y_STEP' not in metadata:
        velocity_file = (velocity_file.split(os.sep)[-1]).replace('geo_', '')
        geometryRadar_file = geometryRadar_file.split(os.sep)[-1].replace('geo_', '')

    keywords = ['SenD','SenA','SenDT', 'SenAT', 'CskAT', 'CskDT']
    elements = path.split(os.sep)
    project_dir = None
    for element in elements:
        for keyword in keywords:
            if keyword in element:
                project_dir = element
                project_base_dir = element.split(keyword)[0]
                track_dir = keyword + element.split(keyword)[1]
                break

    project_base_dir = os.path.join(scratch, project_base_dir)
    vel_file = os.path.join(eos_file.rsplit('/', 1)[0], velocity_file)
    geometry_file = os.path.join(eos_file.rsplit('/', 1)[0], geometryRadar_file)

    inputs_folder = os.path.join(scratch, project_dir)
    out_vel_file = os.path.join(project_base_dir, track_dir, velocity_file.split(os.sep)[-1])

    return eos_file, vel_file, geometry_file, project_base_dir, out_vel_file, inputs_folder


def prepend_scratchdir_if_needed(path):
    """ Prepends $SCRATCHDIR if not in path """

    path_obj = Path(path)
    scratch_dir_obj = Path(os.getenv('SCRATCHDIR'))

    if str(scratch_dir_obj) not in str(path_obj):
        path = os.path.join(scratch_dir_obj, path_obj)

    return path


def save_gbis_plotdata(eos_file, geo_vel_file, start_date_mod, end_date_mod):
    timeseries_file = eos_file.rsplit('/', 1)[0] + '/timeseries_tropHgt_demErr.h5'
    vel_file = geo_vel_file.replace('geo_','')
    geom_file = vel_file.replace('velocity','inputs/geometryRadar')
    print('eos_file', eos_file)

    cmd = f'save_gbis.py {vel_file} -g {os.path.dirname(eos_file)}/inputs/geometryRadar.h5'
    print('save_gbis command:',cmd.split())
    output = subprocess.check_output(cmd.split())


def remove_directory_containing_mintpy_from_path(path):
    mintpy_dir = None
    dirs = path.split('/')
    for i in range(len(dirs) - 1, -1, -1):
        dir = dirs[i]
        if 'mintpy' in dir:
            mintpy_dir = dir
            # Remove the directory and all subsequent directories
            dirs = dirs[:i]
            break
    cleaned_path = '/'.join(dirs)
    return cleaned_path,  mintpy_dir


def find_nearest_start_end_date(fname, start_date, end_date):
    ''' Find nearest dates to start and end dates given as YYYYMMDD '''

    dateList = HDFEOS(fname).get_date_list()
    if start_date and end_date:

        if int(start_date) < int(dateList[0]):
            raise Exception("USER ERROR: No date found earlier than ", start_date )
        if int(end_date) > int(dateList[-1]):
            raise Exception("USER ERROR:  No date found later than ", end_date )

        for date in reversed(dateList):
            if int(date) <= int(start_date):
                # print("Date just before start date:", date)
                mod_start_date = date
                break
        for date in reversed(dateList):
            if int(date) <= int(end_date):
                # print("Date just before end date:", date)
                mod_end_date = date
                break
    else:
        mod_start_date = start_date if start_date else dateList[0]
        mod_end_date = end_date if end_date else dateList[-1]

    print('###############################################')
    print(' Period of data:  ', dateList[0], dateList[-1])
    if start_date and end_date:
        print(' Period requested:', start_date, end_date)
    else:
        print(' Period requested:', start_date, end_date)
    print(' Period used:     ', mod_start_date, mod_end_date)
    print('###############################################')

    return mod_start_date, mod_end_date


def get_data_type(file):
    dir = os.path.dirname(file)
    while 'Sen' not in os.path.basename(dir) and 'Csk' not in os.path.basename(dir):
        dir = os.path.dirname(dir)
        if dir == os.path.dirname(dir):  # Check if we have reached the root directory
            break
    if 'Sen' in os.path.basename(dir) or 'Csk' in os.path.basename(dir):
        #print("Directory containing 'Sen' or 'Csk':", dir)
        tmp = dir.split('Sen')[1][0] if 'Sen' in os.path.basename(dir) else dir.split('Csk')[1][0]
        direction = tmp[0]
        if direction == 'A':
            type = 'Asc'
        elif direction == 'D':
            type = 'Desc'
        else:
            raise Exception('USER ERROR: direction is not A or D -- exiting ')
    else:
        #print("File does not contain 'Sen' or 'Csk':", file)
        if 'up.h5' in file:
            type = 'Up'
        elif 'hz.h5' in file:
            type = 'Horz'
        else:
            type = 'Dem'
            #raise Exception('ERROR: file not up.h5 or horz.h5 -- exiting: ' + file)

    return type


def get_dem_extent(atr_dem):
    # get the extent which is required for plotting
    # [-156.0, -154.99, 18.99, 20.00]
    dem_extent = [float(atr_dem['X_FIRST']), float(atr_dem['X_FIRST']) + int(atr_dem['WIDTH'])*float(atr_dem['X_STEP']),
        float(atr_dem['Y_FIRST']) + int(atr_dem['FILE_LENGTH'])*float(atr_dem['Y_STEP']), float(atr_dem['Y_FIRST'])]
    return(dem_extent)


def extract_window(vel_file, lat, lon, window_size=3):
    data, metadata = readfile.read(vel_file)

    length = int(metadata['LENGTH'])
    width = int(metadata['WIDTH'])

    latitude, longitude = get_bounding_box(metadata)

    # Define the latitude and longitude edges
    lat_edges = np.linspace(min(latitude), max(latitude), length)
    lon_edges = np.linspace(min(longitude), max(longitude), width)

    # Check if the reference point is within the data coverage
    if lat < min(lat_edges) or lat > max(lat_edges) or lon < min(lon_edges) or lon > max(lon_edges):
        raise ValueError('input reference point is OUT of data coverage on file: ' + vel_file)

    # Find the indices of the specified point
    lat_idx = np.searchsorted(lat_edges, lat)
    lon_idx = np.searchsorted(lon_edges, lon)

    # Extract the subarray
    lat_start = max(lat_idx - window_size, 0)
    lat_end = min(lat_idx + window_size + 1, len(lat_edges))
    lon_start = max(lon_idx - window_size, 0)
    lon_end = min(lon_idx + window_size + 1, len(lon_edges))

    # Check if the window outfit the data coverage
    if lat_start<0 or lat_end>length:
        raise ValueError('Latitude range is too large for the data coverage on file: ' + vel_file)

    if lon_start<0 or lon_end>width:
        raise ValueError('Longitude range is too large for the data coverage on file: ' + vel_file)

    subarray = data[lat_start:lat_end, lon_start:lon_end]
    sublat = lat_edges[lat_start:lat_end]
    sublon = lon_edges[lon_start:lon_end]

    return ~np.isnan(subarray) ,sublat, sublon


def find_longitude_degree(ref_lat, lat_step):
    # Find the longitude step in degrees that covers the same distance as the latitude step
    return float(lat_step) / math.cos(math.radians(int(ref_lat)))


def select_reference_point(out_mskd_file, window_size, ref_lalo):
    """
    Selects a reference point from one or two masked velocity files.

    If only one file is provided, the function selects the nearest valid point.
    If two files are provided, it finds the closest overlapping valid point.

    Parameters:
    - out_mskd_file: list of one or two velocity file paths
    - window_size: integer defining the search window size
    - ref_lalo: list [lat, lon] defining the initial reference point
    """
    num_files = len(out_mskd_file)

    if num_files not in [1, 2]:
        raise ValueError('Function supports either one or two data directories.')

    # Extract the subarray for each dataset (handling one or two cases)
    extracted_data = [extract_window(velocity, ref_lalo[0], ref_lalo[1], window_size) for velocity in out_mskd_file]

    subdata1, sublat1, sublon1 = extracted_data[0]  # First dataset

    if num_files == 2:
        subdata2, _, _ = extracted_data[1]  # Second dataset
        paired = list(zip(subdata1, subdata2))
    else:
        paired = [(i, None) for i in subdata1]  # Only one dataset available

    valid_indices = []

    # Find valid indices where data is available
    for ind, (i, j) in enumerate(paired):
        if num_files == 2 and np.logical_and(i, j).any():
            valid_indices.append((ind, np.where(np.logical_and(i, j))))
        elif num_files == 1 and np.any(i):
            valid_indices.append((ind, np.where(i)))

    if not valid_indices:
        raise ValueError("No valid reference points found in the selected window.")

    # Initialize shortest distance tracker
    shortest_distance = window_size * 2 + 1  

    # Find the closest valid data point to the center of the window
    for ind, indices in valid_indices:
        distances = np.sqrt((ind - window_size) ** 2 + (indices[0] - window_size) ** 2)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        if min_distance < shortest_distance:
            shortest_distance = min_distance
            ref_lalo = [sublat1[ind], sublon1[indices[0][min_distance_index]]]

    print('-' * 50)
    print(f"Reference point selected: {ref_lalo[0]:.4f}, {ref_lalo[1]:.4f}")
    print('-' * 50)

    return ref_lalo 


def draw_box(central_lat, central_lon, distance_km = 20, distance_deg = None):
    if not distance_deg:
        # Offsets in degrees conversion
        lat_offset = distance_km / 111  # Approximate conversion from km to degrees latitude
        lon_offset = distance_km / 111  # Approximate conversion from km to degrees longitude

    else:
        lat_offset = distance_deg
        lon_offset = distance_deg

    # Calculate min and max coordinates
    min_lat = central_lat - lat_offset
    max_lat = central_lat + lat_offset
    min_lon = central_lon - lon_offset
    max_lon = central_lon + lon_offset

    region = [min_lon, max_lon, min_lat, max_lat]

    print(f"Min Latitude: {min_lat}, Max Latitude: {max_lat}")
    print(f"Min Longitude: {min_lon}, Max Longitude: {max_lon}\n")

    return region


def calculate_distance(lat_1, lon_1, lat_2, lon_2):
    """
    Calculate the distance between two points on Earth using their longitude and latitude coordinates from degrees.

    Parameters:
    lat_1 (float): Latitude of the first point.
    lon_1 (float): Longitude of the first point.
    lat_2 (float): Latitude of the second point.
    lon_2 (float): Longitude of the second point.

    Returns:
    float: The distance between the two points in kilometers.
    """
    return (((lat_1 - lat_2)*111)**2 + ((lon_1 - lon_2)*111)**2)**0.5


def parse_polygon(polygon):
    """
    Parses a polygon string retreive from ASF vertex tool and extracts the latitude and longitude coordinates.

    Args:
        polygon (str): The polygon string in the format "POLYGON((lon1 lat1, lon2 lat2, ...))".

    Returns:
        tuple: A tuple containing the latitude and longitude coordinates as lists.
               The latitude list contains the minimum and maximum latitude values.
               The longitude list contains the minimum and maximum longitude values.
    """
    latitude = []
    longitude = []
    pol = polygon.replace("POLYGON((", "").replace("))", "")

    # Split the string into a list of coordinates
    for word in pol.split(','):
        if (float(word.split(' ')[1])) not in latitude:
            latitude.append(float(word.split(' ')[1]))
        if (float(word.split(' ')[0])) not in longitude:
            longitude.append(float(word.split(' ')[0]))

    longitude = [round(min(longitude),2), round(max(longitude),2)]
    latitude = [round(min(latitude),2), round(max(latitude),2)]
    region = [longitude[0], longitude[1], latitude[0], latitude[1]]

    return region


def get_bounding_box(metadata):
    """
    Calculate the bounding box coordinates based on the given metadata.

    Args:
        metadata (dict): A dictionary containing the metadata information.

    Returns:
        tuple: A tuple containing two lists, the first list represents the latitude range and the second list represents the longitude range.
    """
    lat_out = []
    lon_out = []

    length = int(metadata['LENGTH'])
    width = int(metadata['WIDTH'])

    for y_i, x_i in zip([0, length], [0, width]):
        lat_i = None if y_i is None else (y_i + 0.5) * float(metadata['Y_STEP']) + float(metadata['Y_FIRST'])
        lon_i = None if x_i is None else (x_i + 0.5) * float(metadata['X_STEP']) + float(metadata['X_FIRST'])
        lat_out.append(lat_i)
        lon_out.append(lon_i)

    return [min(lat_out), max(lat_out)], [min(lon_out), max(lon_out)]


def draw_vectors(elevation, vertical, horizontal, line):
    v = interpolate(elevation, vertical)
    h = interpolate(elevation, horizontal)

    length = np.sqrt(v**2 + h**2)

    #Normalization
    nv = [1 if val > 0 else -1 if val < 0 else 0 for val in v]
    nh = [1 if val > 0 else -1 if val < 0 else 0 for val in h]

    v1 = abs(v)
    h1 = abs(h)

    m = max(v1) if max(v1) > max(h1) else max(h1)

    tv = (v1 - 0) / (m - 0)
    th = (h1 - 0) / (m - 0)

    # Matrix times normalized data
    v = nv * tv
    h = nh * th

    x_coords = np.linspace(0, calculate_distance(line[0][0], line[1][0], line[0][1], line[1][1])*1000, len(elevation))

    return x_coords, v, h


def interpolate(x, y):
    len_x = len(x)
    len_y = len(y)

    # Interpolate to match lengths
    if len_x > len_y:
        x_old = np.linspace(0, 1, len_y)
        x_new = np.linspace(0, 1, len_x)
        y_values_interpolated = interp1d(x_old, y, kind='linear')(x_new)
        y = y_values_interpolated

    return y


def unpack_file(file):
    if isinstance(file, (list, tuple)):
        for element in file:
            result = unpack_file(element)
            if result is not None:
                return result
    else:
        return file
    return None