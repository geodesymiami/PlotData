#!/usr/bin/env python3
# Authors: Farzaneh Aziz Zanjani & Falk Amelung
# This script plots velocity, DEM error, and estimated elevation on the backscatter.
############################################################
import argparse
import os
import sys

import numpy as np
import georaster
import h5py

import matplotlib.pyplot as plt
from mintpy.utils import readfile, arg_utils, utils as ut
from plotdata.helper_functions_PS import *

def plot_scatter(ax, inps, marker='o', colorbar=True):
    
    if  inps.background == 'open_street_map' or inps.background == 'geotiff':
        im = ax.scatter(inps.lon, inps.lat, c=inps.data, s=inps.point_size, cmap=inps.colormap, marker=marker)
    elif  inps.background == 'backscatter':
        # Create a boolean mask for the condition
        mask = (inps.yv < inps.amplitude.shape[0]) & (inps.xv < inps.amplitude.shape[1])
        xv_filtered = inps.xv[mask]
        yv_filtered = inps.yv[mask]
        data_filtered = inps.data[mask]
        
        im = ax.scatter(xv_filtered, yv_filtered, c=data_filtered, s=inps.point_size, cmap=inps.colormap, marker=marker)
        # im = ax.scatter(inps.xv, inps.yv, c=inps.data, s=inps.point_size, cmap=inps.colormap, marker=marker)
   
    if colorbar:
        cbar = plt.colorbar(im,
                            ax=ax,
                            shrink=1,
                            orientation='horizontal',
                            pad=0.1)
        cbar.set_label(inps.cbar_label)
        if inps.vlim is not None:
            clim=(inps.vlim[0], inps.vlim[1])
            im.set_clim(clim[0], clim[1])

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def update_input_namespace(inps):
    """
    Extract relevant data based on specified coordinates and masks.
    """
    # parse subset_lalo, update namespace, add a dictionary of subset latlon
    keys = ['lat1', 'lat2', 'lon1', 'lon2']
    lat1, lat2, lon1, lon2 = [float(val) for val in inps.subset_lalo.replace(':', ',').split(',')]
    inps.coords = {
        key: val for (key, val) in zip(keys, [lat1, lat2, lon1, lon2])
    }
    # read latitude, longitude, height
    latitude = readfile.read(inps.geometry_file, datasetName='latitude')[0]
    longitude = readfile.read(inps.geometry_file, datasetName='longitude')[0]
    inc_angle = readfile.read(inps.geometry_file, datasetName='incidenceAngle')[0]
    height = readfile.read(inps.geometry_file, datasetName='height')[0]

    # read data, convert velocty to cm/yr and convert demError to estimated elevation
    dataset_names = readfile.get_dataset_list(inps.data_file)
    data, atr = readfile.read(inps.data_file, datasetName=dataset_names[0])
    if dataset_names[0] == 'velocity':
        data = data * 100             # convert to cm/yr
        cbar_label = 'Velocity [cm/yr]'
    elif dataset_names[0] == 'dem':        #  for demErr.h5
        data = data
        cbar_label = f"Dem error {atr['UNIT']}"
        if inps.estimated_elevation_flag:
            data = data + height + inps.dem_offset
            cbar_label = f"Estimated elevation {atr['UNIT']}"
            print(f"Added offset to the dem error and height: {inps.dem_offset} meters")

  # read demErr file (even if given as data file) and calculate estimated elevation
    try:
        demErr, atr = readfile.read('demErr.h5')
        estimated_elevation = height + demErr + inps.dem_offset
    except:
        raise FileNotFoundError(f'USER ERROR: file demErr.h5 not found.')

    # change reference point if given
    if dataset_names[0] == 'velocity' and inps.ref_lalo:   
        # Need function:  inps.data = change_reference_point(inps.data, inps.ref_lalo)
        ref_lat = inps.ref_lalo[0]
        ref_lon = inps.ref_lalo[1]
        points_lalo = np.array([ref_lat, ref_lon])
        ref_y, ref_x = coord.geo2radar(points_lalo[0], points_lalo[1])[:2]
        inps.data -= inps.data[ref_y, ref_x]

    inps.lat = latitude
    inps.lon = longitude
    inps.inc_angle = inc_angle
    inps.data = data
    inps.height = height
    inps.demErr = demErr
    inps.estimated_elevation = estimated_elevation
    inps.HEADING = float(atr['HEADING'])

    if inps.correct_geo:
       correct_geolocation(inps)

    # Fari: This should be a separate function
    mask = np.ones(data.shape, dtype=np.float32)
    mask[latitude<lat1] = 0
    mask[latitude>lat2] = 0
    mask[longitude<lon1] = 0
    mask[longitude>lon2] = 0
    
    if inps.mask:
        mask_ps = readfile.read(inps.mask, datasetName='mask')[0]
        mask *= mask_ps  # Apply mask_p within the specified ymin, ymax, xmin, xmax

    inps.lat = np.array(inps.lat[mask == 1])
    inps.lon = np.array(inps.lon[mask == 1])
    inps.data = np.array(inps.data[mask == 1])
    inps.height = np.array(inps.height[mask == 1])
    inps.demErr = np.array(inps.demErr[mask == 1])
    inps.estimated_elevation = np.array(inps.estimated_elevation[mask == 1])
    
    inps.cbar_label = cbar_label
    inps.HEADING = float(atr['HEADING'])

    if inps.background =='backscatter':
        # Fari: Here it should call one function
        coord = ut.coordinate(atr, inps.geometry_file)
        yg1, xg1 = coord.geo2radar(lat1, lon1)[:2]
        yg2, xg2 = coord.geo2radar(lat2, lon2)[:2]
        yg3, xg3 = coord.geo2radar(lat1, lon2)[:2]
        yg4, xg4 = coord.geo2radar(lat2, lon2)[:2]
        print("Lat, Lon, y, x: ", lat1, lon1, yg1, xg1)
        print("Lat, Lon, y, x: ", lat2, lon2, yg2, xg2)
        print("Lat, Lon, y, x: ", lat1, lon2, yg3, xg3)
        print("Lat, Lon, y, x: ", lat2, lon2, yg4, xg4)
        ymin = min(yg1, yg2, yg3, yg4)
        ymax = max(yg1, yg2, yg3, yg4)
        xmin = min(xg1, xg2, xg3, xg4)
        xmax = max(xg1, xg2, xg3, xg4)
        x = np.linspace(0, data.shape[1] - 1, data.shape[1])
        y = np.linspace(0, data.shape[0] - 1, data.shape[0])
        x, y = np.meshgrid(x, y)
        inps.xv = xmax - np.array(x[mask == 1])
        inps.yv = np.array(y[mask == 1]) - ymin
        backscatter_file = default_backscatter_file() 
        if not os.path.exists(inps.out_amplitude):
            calculate_mean_amplitude(backscatter_file, inps.out_amplitude)
        inps.amplitude = np.fliplr(np.load(inps.out_amplitude)[ymin:ymax, xmin:xmax])

def configure_plot_settings(inps):
    """
    Configure plot settings based on command-line arguments.

    inps:
        inps (argparse.Namespace): Parsed command-line arguments.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot,
        matplotlib.colors.Colormap, matplotlib.colors.Normalize: Figure, Axes,
        colormap, and normalization for color scale.
    """
    plt.rcParams['font.size'] = inps.fontsize
    plt.rcParams['axes.labelsize'] = inps.fontsize
    plt.rcParams['xtick.labelsize'] = inps.fontsize
    plt.rcParams['ytick.labelsize'] = inps.fontsize
    plt.rcParams['axes.titlesize'] = inps.fontsize

    if inps.colormap:
        inps.colormap = plt.get_cmap(inps.colormap)
    else:
        inps.colormap = plt.get_cmap('jet')

    fig, ax = plt.subplots(figsize=inps.figsize)
    return fig, ax

def persistent_scatterers(inps):
    update_input_namespace(inps)

    fig, ax = configure_plot_settings(inps)

    # Adding background image
    if inps.background == 'open_street_map':
        add_open_street_map_image(ax, inps.coords)
    elif inps.background == 'backscatter':
        add_backscatter_image(ax, inps.amplitude)
    elif inps.background == 'satellite':
        add_satellite_image(ax)
    elif inps.background == 'geotiff':
        add_geotiff_image(ax, inps.geotiff, inps.coords)
    else:
        raise Exception("USER ERROR: background option not supported:", inps.background )
        
    plot_scatter(ax=ax, inps=inps)
    fig.tight_layout()
    # save figure
    if inps.save_fig:
        print(f'save figure to {inps.outfile} with dpi={inps.fig_dpi}')
        if not inps.disp_whitespace:
            fig.savefig(inps.outfile, transparent=True, dpi=inps.fig_dpi, pad_inches=0.0)
        else:
            fig.savefig(inps.outfile, transparent=True, dpi=inps.fig_dpi, bbox_inches='tight')
    
    plt.show(block=False)

    if inps.kml_3d:
        # create kml-3d file
        print('create kml 3D file')
        create_kml_3D_file(inps)