#! /usr/bin/env python3
import os
from mintpy.utils import readfile, writefile
from matplotlib.colors import LinearSegmentedColormap, LightSource
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from plotdata.helper_functions import get_dem_extent

def modify_colormap(cmap_name = "plasma_r", exclude_beginning = 0.15, exclude_end = 0.25, show = False):
    """ modify a colormap by excluding percentages at the beginning and end """

    cmap = plt.cm.get_cmap(cmap_name)
    cmap

    num_colors_to_exclude_beginning = int(len(cmap.colors) * exclude_beginning)
    num_colors_to_exclude_end = int(len(cmap.colors) * exclude_end)
    
    # Create a custom colormap by excluding the specified percentages of colors
    colors = cmap.colors[num_colors_to_exclude_beginning:-num_colors_to_exclude_end]
    cmap_custom = LinearSegmentedColormap.from_list('cmap_custom', colors, N=256)
    
    if show is True: 
        # Create a plot with the custom colormap
        data = [[0, 1, 2, 4], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
        plt.imshow(data, cmap=cmap_custom)
        plt.colorbar()
        plt.show()

    return cmap_custom
    
def add_colorbar(ax, cmap, start_date="", end_date=""):
    # Convert date strings to datetime objects
    start_time_date = datetime.strptime(start_date, "%Y%m%d")
    end_time_date = datetime.strptime(end_date, "%Y%m%d")
    
    # Create a separate colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []  # Hack to avoid normalization
    cbar = plt.colorbar(sm, ax = ax, shrink=0.5)  # Adjust the shrink value as needed
    
    # Set custom tick locations and labels on the colorbar
    ticks = np.linspace(0, 1, 5)  # You can adjust the number of ticks as needed
    tick_labels = [start_time_date + (end_time_date - start_time_date) * t for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([label.strftime("%Y-%m-%d") for label in tick_labels])    
    cbar.set_label("Time")
    
def get_ticks(extent, step_size=0.2):
    # Calculate the start values for lat and long and generate sequences with step size of 0.2
    lat_start = np.ceil(extent[2] / step_size) * step_size
    lon_start = np.ceil(extent[0] / step_size) * step_size
    lats = np.arange(lat_start, extent[3], step_size)
    lons = np.arange(lon_start, extent[1], step_size)
    return lons, lats

def get_step_size(plot_extent):
    # calculate step size so that there are around 5 longitude labels
    
    step_size = (plot_extent[1]-plot_extent[0]) / 5
    if step_size <= 0.15:
        rounded_step_size = 0.1
    elif step_size <= 0.35:
        rounded_step_size = 0.2
    else:
        rounded_step_size = 0.5
    return rounded_step_size

def get_basemap(dem_file):
    dem, atr_dem = readfile.read(dem_file)
    dem_extent = get_dem_extent(atr_dem)
    ls = LightSource(azdeg=315, altdeg=45)
    dem_shade = ls.shade(dem, vert_exag=1.0, cmap=plt.cm.gray, vmin=-20000, vmax=np.nanmax(dem)+2500)
    return dem_shade,dem_extent

def plot_shaded_relief(ax, dem_file, plot_box = []):
    
    factory_default_figsize = plt.rcParamsDefault['figure.figsize']
    current_default_figsize = plt.rcParams['figure.figsize']
    increase_factor = 1.5
    if current_default_figsize == factory_default_figsize:
        new_default_figsize = [size * increase_factor for size in current_default_figsize]
        plt.rcParams['figure.figsize'] = new_default_figsize

    #plot DEM as background
    dem_shade, dem_extent = get_basemap(dem_file);

    ax.imshow(dem_shade, origin='upper', cmap=plt.cm.gray, extent=dem_extent)
    # ax.add_feature(cfeature.COASTLINE)
    
    if len(plot_box) == 0:
       plot_extent = dem_extent          # x_min, x_max, y_min, y_max
    else:
       plot_extent = [plot_box[2], plot_box[3], plot_box[0], plot_box[1]]   
     
    # Add latitude and longitude labels to the plot
    step_size = get_step_size(plot_extent)
    lons, lats = get_ticks(plot_extent, step_size = step_size)
    print(lons)
    print(lats)

    ax.set_xticks(lons)
    ax.set_yticks(lats)

    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.1, 0.5)


    return ax

