#!/usr/bin/env python3
# Authors: Farzaneh Aziz Zanjani & Falk Amelung
# This script plots displacement, velocity, estimated elevation or DEM error on open street map or backscatter.
############################################################
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from mintpy.utils import readfile, ptime, utils as ut
from mintpy.objects import HDFEOS
from mintpy.utils import plot as pp
from plotdata.helper_functions_PS import *
from plotdata.plot_functions_PS import *

def update_input_namespace(inps):
    """ Extract relevant data based on specified coordinates and masks.  """

    # determne file type (HDFEOS, SARPROZ or ANDREAS)
    files = glob.glob(inps.data_file)
    if not files:
        raise FileNotFoundError(f'USER ERROR: file {inps.data_file} not found.') 
    
    try:
        metadata = readfile.read_attribute(files[0])
        metadata['FILE_TYPE']
        inps.file_type = readfile.read_attribute(files[0])['FILE_TYPE']
    except:
        # Fari, here we just need a function to figure out which file_type
        inps.file_type = "SARPROZ"
        inps.file_type = "Andreas"
        pass
    
    if inps.file_type == 'HDFEOS':
        # read HDFEOS data, convert velocty to cm/yr,  convert dem_error to estimated elevation
        latitude, _ = readfile.read(files[0], datasetName='HDFEOS/GRIDS/timeseries/geometry/latitude')
        longitude, _ = readfile.read(files[0], datasetName='HDFEOS/GRIDS/timeseries/geometry/longitude')
        height, _  = readfile.read(files[0], datasetName='HDFEOS/GRIDS/timeseries/geometry/height')
        inc_angle, _  = readfile.read(files[0], datasetName='HDFEOS/GRIDS/timeseries/geometry/incidenceAngle')
        az_angle, _  = readfile.read(files[0], datasetName='HDFEOS/GRIDS/timeseries/geometry/azimuthAngle')

        date_list = HDFEOS(files[0]).get_date_list()
        dataset_first = f'HDFEOS/GRIDS/timeseries/observation/displacement-{date_list[0]}'
        dataset_last = f'HDFEOS/GRIDS/timeseries/observation/displacement-{date_list[-1]}'
        displacement_first, attr = readfile.read(files[0], datasetName=dataset_first)
        displacement_last, attr  = readfile.read(files[0], datasetName=dataset_last)
        displacement = (displacement_last - displacement_first) * 100    # total displacement
        
        label_dict = dict()
        label_dict['displacement'] = {'str': 'Total displacement', 'unit': 'cm' }
        label_dict['height'] = {'str': 'Dem height', 'unit': 'm' }
        label_dict['dem_error'] = {'str': 'Dem error', 'unit': 'm' }
        label_dict['elevation'] = {'str': 'Estimated elevation', 'unit': 'm' }

        if inps.lalo:
            print('reading full timeseries data ....')
            timeseries, _ = readfile.read(files[0], datasetName=date_list)
            timeseries *= 100   # convert to cm
            
        # legacy/compatibility code: read demErr.h5 because missing in S1*he5 file, 
        #                            read velocity.h5 (should be created on the fly)
        try:
            dem_error, attr = readfile.read('demErr.h5', datasetName='dem')
            elevation = height + dem_error + inps.dem_offset
        except:
            raise FileNotFoundError(f'USER ERROR: file demErr.h5 not found.')

        try:
            velocity, attr = readfile.read('velocity.h5', datasetName='velocity')
            velocity = velocity * 100             # convert to cm/yr
            label_dict['velocity'] = {'str': 'Velocity', 'unit': 'cm/yr' }
        except:
            raise FileNotFoundError(f'USER ERROR: file velocity.h5 not found.')

        # change reference point (function uses geometryRadar.h5. Need a function that works for HDFEOS-radar and other file types)
        if inps.ref_lalo:  
            displacement = change_reference_point(displacement, attr, inps.ref_lalo, inps.file_type) 
            velocity = change_reference_point(velocity, attr, inps.ref_lalo, inps.file_type) 
            if inps.lalo:
                timeseries = change_reference_point(timeseries, attr, inps.ref_lalo, inps.file_type) 

        # mask = np.ones(displacement.shape, dtype=np.float32)
        mask = readfile.read(inps.mask, datasetName='mask')[0]
        
        inps.displacement = np.array(displacement[mask == 1])
        inps.velocity = np.array(velocity[mask == 1])
        inps.dem_error = np.array(dem_error[mask == 1])
        inps.elevation = np.array(elevation[mask == 1]) 
        inps.height = np.array(height[mask == 1])
        inps.lat = np.array(latitude[mask == 1])
        inps.lon = np.array(longitude[mask == 1])
        inps.inc_angle = np.array(inc_angle[mask == 1])
        inps.az_angle = np.array(az_angle[mask == 1])
        inps.HEADING = float(attr['HEADING'])
        
        if inps.lalo:
            inps.timeseries = timeseries
            inps.date_list = date_list
            inps.num_date = len(inps.date_list)
            inps.dates, inps.yearList = ptime.date_list2vector(inps.date_list)

        # from mintpy.tsview import read_exclude_date
        # (inps.ex_date_list, inps.ex_dates, inps.ex_flag) = read_exclude_date(inps.ex_date_list, inps.date_list)

        # get data for time series plot
        if inps.lalo:
            inps.timeseries_at_point = extract_data_at_point(timeseries, attr, inps.lalo, inps.file_type)

    elif inps.file_type == 'SARPROZ':
        # read_data_SARPROZ(inps)
        pass
    elif inps.file_type == 'ANDREAS':
        # read_data_ANDREAS(inps)
        pass
    
    # assign the dataset of interest
    inps.data = getattr(inps, inps.dataset)
    inps.label_dict = label_dict[inps.dataset]
    
    if not inps.vlim: 
        inps.vlim = [np.nanmin(inps.data), np.nanmax(inps.data)]
                  
    # parse subset_lalo or get from data,  create coords dictionary 
    if inps.subset_lalo:
        lat1, lat2, lon1, lon2 = [float(val) for val in inps.subset_lalo.replace(':', ',').split(',')]
    else:
        lat1, lat2 = np.min(latitude), np.max(latitude)
        lon1, lon2 = np.min(longitude), np.max(longitude)  
    keys = ['lat1', 'lat2', 'lon1', 'lon2']
    inps.coords = {   key: val for (key, val) in zip(keys, [lat1, lat2, lon1, lon2])  }
 
    if inps.subset_lalo:
       extract_subset_from_data(inps, inps.coords)

    # correct  geolocation using DEM error
    if inps.correct_geo:
       correct_geolocation(inps)

    if inps.background =='backscatter':
        # Fari: This should go into a function
        mask[latitude<lat1] = 0
        mask[latitude>lat2] = 0
        mask[longitude<lon1] = 0
        mask[longitude>lon2] = 0
        
        coord = ut.coordinate(attr, inps.geometry_file)
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
        x = np.linspace(0, displacement.shape[1] - 1, displacement.shape[1])
        y = np.linspace(0, displacement.shape[0] - 1, displacement.shape[0])
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


    figs, axs = [], []
    fig, ax = plt.subplots(figsize=inps.figsize)
    figs.append(fig), axs.append(ax)

    inps.figsize_ts = [12,5]
    if inps.lalo:
        for i in range(len(inps.lalo)):
            fig, ax = plt.subplots(figsize=inps.figsize_ts)
            figs.append(fig), axs.append(ax)

    return figs, axs

def persistent_scatterers(inps):
    # create kml file, display figure to screen, or save figure

    update_input_namespace(inps)

    # create 2d or 3d kml file and exit
    if inps.kml_2d or inps.kml_3d:
        create_kml_file(inps)
        return
    
    # Configure plot and start ax
    figs, axs = configure_plot_settings(inps)

    # Add background image 
    if inps.background == 'open_street_map' or inps.background == 'satellite':
        add_open_street_map_image(axs[0], inps.coords, inps.background)
    elif inps.background == 'backscatter':
        add_backscatter_image(axs[0], inps.amplitude)
    elif inps.background == 'geotiff':
        add_geotiff_image(axs[0], inps.geotiff, inps.coords)
    else:
        raise Exception("USER ERROR: background option not supported:", inps.background )

    # plot data    
    plot_scatter(ax=axs[0], inps=inps)
    figs[0].tight_layout()
    
    if inps.lalo:
        # create time series plots
        for ax in axs[1:]: 
            plot_timeseries(ax=ax, inps=inps)
       
    # save figure
    if not inps.save_fig:
        plt.show()
        # plt.show(block=False)
    else:
        print(f'save figure to {inps.outfile} with dpi={inps.fig_dpi}')
        if not inps.disp_whitespace:
            figs[0].savefig(inps.outfile, transparent=True, dpi=inps.fig_dpi, pad_inches=0.0)
        else:
            figs[0].savefig(inps.outfile, transparent=True, dpi=inps.fig_dpi, bbox_inches='tight')
    
        if inps.lalo:
            i=0
            for fig in figs[1:]:
                i += 1
                outfile = f'timeseries{i}.png'
                print(f'save figure to {outfile} with dpi={inps.fig_dpi}')
                if not inps.disp_whitespace:
                    fig.savefig(outfile, transparent=True, dpi=inps.fig_dpi, pad_inches=0.0)
                else:
                    fig.savefig(outfile, transparent=True, dpi=inps.fig_dpi, bbox_inches='tight')
