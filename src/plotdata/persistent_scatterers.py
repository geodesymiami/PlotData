#!/usr/bin/env python3
# Authors: Farzaneh Aziz Zanjani & Falk Amelung
# This script plots displacement, velocity, estimated elevation or DEM error on open street map or backscatter.
############################################################
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from mintpy.utils import readfile, utils as ut
from mintpy.objects import HDFEOS
from plotdata.helper_functions_PS import *

def update_input_namespace(inps):
    """ Extract relevant data based on specified coordinates and masks.  """

    # read data, convert velocty to cm/yr,  convert dem_error to estimated elevation
    files = glob.glob(inps.data_file)
    if not files:
        raise FileNotFoundError(f'USER ERROR: file {inps.data_file} not found.') 
    
    try:
        metadata = readfile.read_attribute(files[0])
        metadata['FILE_TYPE']
        inps.file_type = readfile.read_attribute(files[0])['FILE_TYPE']
    except:
        # Fari, here we just need something to figure out which file_type
        inps.file_type = "SARPROZ"
        inps.file_type = "Andreas"
        pass
    
    if inps.file_type == 'HDFEOS':
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

    # change reference point. Need function: inps.data = change_reference_point(data, inps.ref_lalo, file_type)
    if inps.ref_lalo:  
        displacement = change_reference_point(displacement, inps.ref_lalo, inps.file_type) 
        velocity = change_reference_point(velocity, inps.ref_lalo, inps.file_type) 
       # FA: REF_LAT/LON is not available. Need to calculate and add to inps for plotting
    
    # parse subset_lalo or get from data,  create coords dictionary 
    if inps.subset_lalo:
        lat1, lat2, lon1, lon2 = [float(val) for val in inps.subset_lalo.replace(':', ',').split(',')]
    else:
        lat1, lat2 = np.min(latitude), np.max(latitude)
        lon1, lon2 = np.min(longitude), np.max(longitude)  
    keys = ['lat1', 'lat2', 'lon1', 'lon2']
    inps.coords = {   key: val for (key, val) in zip(keys, [lat1, lat2, lon1, lon2])  }
       
    # Fari: Why  is this 
    mask = np.ones(displacement.shape, dtype=np.float32)
    mask[latitude<lat1] = 0
    mask[latitude>lat2] = 0
    mask[longitude<lon1] = 0
    mask[longitude>lon2] = 0
    
    if inps.mask:
        mask_ps = readfile.read(inps.mask, datasetName='mask')[0]
        mask *= mask_ps  # Apply mask_p within the specified ymin, ymax, xmin, xmax
  
    inps.displacement = np.array(displacement[mask == 1])
    inps.velocity = np.array(velocity[mask == 1])
    inps.dem_error = np.array(dem_error[mask == 1])
    inps.elevation = np.array(elevation[mask == 1]) 
    inps.height = np.array(height[mask == 1])
    inps.lat = np.array(latitude[mask == 1])
    inps.lon = np.array(longitude[mask == 1])
    inps.inc_angle = np.array(inc_angle[mask == 1])
    inps.az_angle = np.array(az_angle[mask == 1])
    
    # correct the geolocation if option is given
    if inps.correct_geo:
       correct_geolocation(inps)
    
    # assign the dataset of interest
    inps.data = getattr(inps, inps.dataset)
    inps.label_dict = label_dict[inps.dataset]

    if not inps.vlim: 
        inps.vlim = [np.nanmin(inps.data), np.nanmax(inps.data)]
     
    if inps.background =='backscatter':
        # Fari: Here it should call one function
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

    fig, ax = plt.subplots(figsize=inps.figsize)
    return fig, ax

def persistent_scatterers(inps):
    # create kml file, display figure to screen, or save figure

    update_input_namespace(inps)

    fig, ax = configure_plot_settings(inps)

    # create 2d or 3d kml file and exit
    if inps.kml_2d or inps.kml_3d:
        create_kml_file(inps)
        return

    # Add background image and plot
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
    if not inps.save_fig:
        plt.show()
        # plt.show(block=False)
    else:
        print(f'save figure to {inps.outfile} with dpi={inps.fig_dpi}')
        if not inps.disp_whitespace:
            fig.savefig(inps.outfile, transparent=True, dpi=inps.fig_dpi, pad_inches=0.0)
        else:
            fig.savefig(inps.outfile, transparent=True, dpi=inps.fig_dpi, bbox_inches='tight')
    
def plot_scatter(ax, inps, marker='o', colorbar=True):
    
    if  inps.background == 'open_street_map' or inps.background == 'geotiff':
        im1 = ax.scatter(inps.lon, inps.lat, c=inps.data, s=inps.point_size, cmap=inps.colormap, marker=marker)
        if inps.ref_lalo:
            ax.scatter(inps.ref_lalo[1], inps.ref_lalo[0], color='black', s=inps.point_size*1.2, marker='s')

    elif  inps.background == 'backscatter':
        # Create a boolean mask for the condition
        mask = (inps.yv < inps.amplitude.shape[0]) & (inps.xv < inps.amplitude.shape[1])
        xv_filtered = inps.xv[mask]
        yv_filtered = inps.yv[mask]
        data_filtered = inps.data[mask]
        
        im1 = ax.scatter(xv_filtered, yv_filtered, c=data_filtered, s=inps.point_size, cmap=inps.colormap, marker=marker)
        # im = ax.scatter(inps.xv, inps.yv, c=inps.data, s=inps.point_size, cmap=inps.colormap, marker=marker)
   
    if colorbar:
        cbar = plt.colorbar(im1,
                            ax=ax,
                            shrink=1,
                            orientation='horizontal',
                            pad=0.02)
        cbar.set_label(inps.label_dict['str'] + ' [' + inps.label_dict['unit'] + ']' )
        if inps.vlim is not None:
            clim=(inps.vlim[0], inps.vlim[1])
            im1.set_clim(clim[0], clim[1])

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

