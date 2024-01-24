#!/usr/bin/env python
# coding: utf-8

# ## Plot InSAR, GPS and seismicity data
# Assumes that the data are located in  `$SCRATCHDIR` (e.g.  `$SCRATCHDIR/MaunaLoaSenDT87/mintpy`).
# Output is  written into  `$SCRATCHDIR/MaunaLoa/SenDT87` and `$SCRATCHDIR/MaunaLoa/SenAT124`

import os
import matplotlib.pyplot as plt
import scipy.io as sio
from mintpy.utils import readfile, writefile
from mintpy.defaults.plot import *
from mintpy.view import prep_slice, plot_slice
from mintpy.cli import reference_point, asc_desc2horz_vert, save_gdal, mask
from helper_functions import get_file_names, get_data_type, get_plot_box
from helper_functions import prepend_scratchdir_if_needed, find_nearest_start_end_date
from helper_functions import  save_gbis_plotdata
from plot_functions import plot_shaded_relief
from plot_functions import modify_colormap, add_colorbar
from seismicity import get_earthquakes, normalize_earthquake_times
from gps import get_gps
from insar import generate_view_velocity_cmd, generate_view_ifgram_cmd
import subprocess

def run_prepare(inps):
    # Prepare data for plotting
    # Hardwired: move to argparse
    inps.depth_range="0 10"
    inps.cmap_name = "plasma_r"; inps.exclude_beginning = 0.2; inps.exclude_end = 0.2
    
    # Hardwired for Hawaii
    if 'GPSDIR' in os.environ:
        inps.gps_dir = os.getenv('GPSDIR') + '/data'
    else:
        inps.gps_dir = os.getenv('SCRATCHDIR') + '/MaunaLoa/MLtry/data'
    
    print('run_prepare: inps.gps_dir:' , inps.gps_dir)
    inps.gps_list_file = inps.gps_dir + '/GPS_BenBrooks_03-05full.txt'

    data_dir = inps.data_dir
    dem_file =  inps.dem_file
    if inps.dem_file:
        dem_file =  inps.dem_file
    else:
        dem_file =  inps.data_dir[0] + '/geo/geo_geometryRadar.h5'

    plot_type = inps.plot_type
    reference_lalo = inps.reference_lalo
    mask_vmin = inps.mask_vmin
    flag_save_gbis =  inps.flag_save_gbis
    if inps.period:
        inps.period = [val for val in inps.period.split('-')]      # converts to period=['20220101', '20221101']
        start_date = inps.period[0]
        end_date = inps.period[1]

    # calculate velocities for periods of interest
    data_dict = {}
    if plot_type == 'velocity' or plot_type == 'horzvert':
        for dir in data_dir:
            work_dir = prepend_scratchdir_if_needed(dir)
            eos_file, q, q, q, out_geo_vel_file = get_file_names(work_dir)
            temp_coh_file=out_geo_vel_file.replace('velocity.h5','temporalCoherence.tif')
            start_date, end_date = find_nearest_start_end_date(eos_file, inps.period)
            # get masked geo_velocity.h5 with MintPy
            # cmd = f'{eos_file} --start-date {start_date_mod} --end-date {end_date_mod} --output {out_geo_vel_file}'
            # timeseries2velocity.main( cmd.split() )
            cmd = f'{eos_file} --start-date {start_date} --end-date {end_date} --output {out_geo_vel_file}'
            cmd =['timeseries2velocity.py'] + cmd.split()
            output = subprocess.check_output(cmd)
            #print(output.decode())
            cmd = f'{eos_file} --dset temporalCoherence --output {temp_coh_file}'
            save_gdal.main( cmd.split() )
            cmd = f'{out_geo_vel_file} --mask {temp_coh_file} --mask-vmin { mask_vmin} --outfile {out_geo_vel_file}'
            mask.main( cmd.split() )
            if reference_lalo:
                cmd = f'{out_geo_vel_file} --lat {reference_lalo[0]} --lon {reference_lalo[1]}'
                reference_point.main( cmd.split() )
            if flag_save_gbis:
                save_gbis_plotdata(eos_file, out_geo_vel_file, start_date, end_date)
            data_dict[out_geo_vel_file] = {
            'start_date': start_date,
            'end_date': end_date
            }
    elif plot_type == 'step':
        for dir in data_dir:
            work_dir = prepend_scratchdir_if_needed(dir)
            eos_file, geo_vel_file, geo_geometry_file, out_dir, out_geo_vel_file = get_file_names(work_dir)
            geo_step, atr = readfile.read(geo_vel_file, datasetName='step20210306')
            out_geo_step_file = out_geo_vel_file.replace('velocity','step')
            writefile.write(geo_step, out_file=out_geo_step_file, metadata=atr)
            if reference_lalo:
                cmd = f'{out_geo_step_file} --lat {reference_lalo[0]} --lon {reference_lalo[1]}'
                reference_point.main( cmd.split() )
            data_dict[out_geo_step_file] = {
            'start_date': atr['mintpy.timeFunc.stepDate'],
            'end_date': atr['mintpy.timeFunc.stepDate']
            }
    elif plot_type == 'shaded-relief':
        data_dict[dem_file] = {
        'start_date': start_date,
        'end_date': end_date
        }
 
    # calculate horizontal and vertical
    if  plot_type == 'horzvert':
        data_dict = {}
        q, q, q, q, out_geo_vel_file0 = get_file_names( prepend_scratchdir_if_needed(data_dir[0]) )
        q, q, q, q, out_geo_vel_file1 = get_file_names( prepend_scratchdir_if_needed(data_dir[1]) )
        
        cmd = f'{out_geo_vel_file0} {out_geo_vel_file1}'
        asc_desc2horz_vert.main( cmd.split() )
        data_dict['up.h5'] = {'start_date': start_date, 'end_date': end_date}
        data_dict['hz.h5'] = {'start_date': start_date, 'end_date': end_date}
    
    if inps.plot_box is None:
        inps.plot_box = get_plot_box(data_dict)
    
    return data_dict

def run_plot(data_dict, inps):

    gps_dir = inps.gps_dir
    gps_list_file = inps.gps_list_file
    plot_box = inps.plot_box
    flag_seismicity = inps.flag_seismicity
    flag_gps = inps.flag_gps
    plot_type = inps.plot_type
    line_file = inps.line_file
    gps_scale_fac = inps.gps_scale_fac
    gps_key_length = inps.gps_key_length
    gps_unit = inps.gps_unit
    font_size = inps.font_size
    if inps.period:
        # period = [val for val in inps.period.split('-')]      # converts to period=['20220101', '20221101']
        start_date = inps.period[0]
        end_date = inps.period[1]
    else:
        start_date = data_dict[next(iter(data_dict))]['start_date']
        end_date = data_dict[next(iter(data_dict))]['end_date']
    cmap_name = inps.cmap_name
    exclude_beginning = inps.exclude_beginning
    exclude_end = inps.exclude_end

    # initialize plot
    if len(data_dict) == 2:
        fig, axes = plt.subplots(1, 2, figsize=[12, 5] )
        inps.font_size = int(inps.font_size*0.7)  
    else:
        fig, axes = plt.subplots(figsize=[12, 5] )
        axes = [axes] 
        
    for i, (file, dict) in enumerate(data_dict.items()):
        
        if plot_type == 'velocity' or plot_type == 'horzvert' or plot_type == 'ifgram' or plot_type == 'step':
            if plot_type == 'velocity' or plot_type == 'horzvert' or plot_type == 'step':
                cmd = generate_view_velocity_cmd(file, inps) 
            elif plot_type == 'ifgram':
                cmd = generate_view_ifgram_cmd(work_dir, date12, inps)
            data, atr, tmp_inps = prep_slice(cmd)
            q0, q1, q2, q3 = plot_slice(axes[i], data, atr, tmp_inps)
        elif plot_type == 'shaded-relief':
            plot_shaded_relief(axes[i], file, plot_box = plot_box)
     # plot title
        data_type = get_data_type(file)
        axes[i].set_title(data_type + ': ' + dict['start_date'] + ' - ' + dict['end_date']);
     
        # plot fault lines
        if line_file:
            lines=sio.loadmat(line_file,squeeze_me=True);
            axes[i].plot(lines['Lllh'][:,0],lines['Lllh'][:,1],color='black', linestyle='dashed',linewidth=2)
     
        # plot events
        if flag_seismicity:
            events_df = get_earthquakes(start_date, end_date, plot_box)
            norm_times = normalize_earthquake_times(events_df, start_date, end_date)
            cmap = modify_colormap( cmap_name = cmap_name, exclude_beginning = exclude_beginning, exclude_end = exclude_end, show = False)
            if not events_df.shape[0] == 0:
                axes[i].scatter(events_df["Longitude"],events_df["Latitude"],s=2*events_df["Magnitude"] ** 3, c=norm_times,cmap=cmap,alpha=0.8)
            # plot scale only if there is only one plot (if there are no two plots)
            if  not ( len(data_dict.items()) == 2):
                add_colorbar(ax = axes[i], cmap = cmap, start_date = start_date, end_date = end_date)
        
        if flag_gps:
            gps,lon,lat,U,V,Z,quiver_label = get_gps(gps_dir, gps_list_file, plot_box, start_date, end_date, gps_unit, inps.gps_key_length)
            (gps_dir, gps_list_file, plot_box, start_date, end_date, gps_unit, gps_key_length)
            quiv=axes[i].quiver(lon, lat, U, V, scale = gps_scale_fac, color='blue')
            axes[i].quiverkey(quiv, -155.50, 19.57, gps_key_length*10 , quiver_label, labelpos='N',coordinates='data',
                              color='blue',fontproperties={'size': font_size}) 
    plt.show()
