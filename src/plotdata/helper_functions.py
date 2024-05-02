#! /usr/bin/env python3
import os
import argparse
import subprocess
import glob
from mintpy.utils import readfile, writefile
from mintpy.objects import HDFEOS
from mintpy.utils.arg_utils import create_argument_parser
import numpy as np
from pathlib import Path

EXAMPLE = """example:
  plot_data.py  MaunaLoaSenDT87 MaunaLoaSenAT124 
  plot_data.py  MaunaLoaSenDT87       
"""
     
def something(iargs=None):
    print('QQ: Falk_test')
    return

def print_string(string=None):
    """print a string"""
    print('QQ-print_string:', string)
    print('QQ-print_again:', string)
    return

def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    print('cmd_line_parse: iargs:',iargs)
    print('cmd_line_parse: parser:',parser)
    
    #import pdb; pdb.set_trace()
    args = parser.parse_args(args=iargs)
    print('cmd_line_parse: args:',args)
    print('cmd_line_parse: args.plot_box:',args.plot_box)

    if len(args.data_dir) < 1 or len(args.data_dir) > 2:
        parser.error('ERROR: You must provide 1 or 2 directory paths.')
        
    inps = args
    print('cmd_line_parse: inps.plot_box:',inps.plot_box)
    inps.plot_box = [float(val) for val in args.plot_box.replace(':', ',').split(',')]  # converts to plot_box=[19.3, 19.6, -155.8, -155.4]
    if inps.ref_lalo:
        ref_lalo = args.ref_lalo
        inps.ref_lalo = [float(val) for val in ref_lalo.split(',')]         # converts to reference_point=[19.3, -155.8]
    if inps.period:
        period = args.period
        inps.period = [val for val in period.split('-')]                                # converts to period=['20220101', '20221101']

    return inps

def is_jupyter():
    jn = True
    try:
        get_ipython()
    except:
        jn = False
    return jn

def get_file_names(path):
    """gets the youngest eos5 file. Path can be: 
    MaunaLoaSenAT124
    MaunaLoaSenAT124/mintpy/S1_qq.he5
    ~/onedrive/scratch/MaunaLoaSenAT124/mintpy/S1_qq.he5'
    """
    if os.path.isfile(path):
        eos_file = path
    elif os.path.isfile(os.getenv('SCRATCHDIR') + '/' + path):
        eos_file = os.getenv('SCRATCHDIR') + '/' + path
    else:
        if 'mintpy' in path or 'network' in path :
            files = glob.glob( path + '/*.he5' )
        else:
            files = glob.glob( path + '/mintpy/*.he5' )
        if len(files) == 0:
            raise Exception('USER ERROR: No HDF5EOS files found in ' + path)
        eos_file = max(files, key=os.path.getctime)
    print('HDF5EOS file used:', eos_file)

    keywords = ['SenDT', 'SenAT', 'CskAT', 'CskDT']
    elements = path.split(os.sep)   
    project_dir = None
    for element in elements:
        for keyword in keywords:
            if keyword in element:
                project_dir = element
                project_base_dir = element.split(keyword)[0]
                track_dir = keyword + element.split(keyword)[1]
                break
    
    geo_vel_file = eos_file.rsplit('/', 1)[0] + '/geo/geo_velocity.h5'
    geo_geometry_file = eos_file.rsplit('/', 1)[0] + '/geo/geo_geometryRadar.h5'

    out_geo_vel_file = project_base_dir + '/' + track_dir + '/geo_velocity.h5'

    return eos_file, geo_vel_file, geo_geometry_file, project_base_dir, out_geo_vel_file    

def prepend_scratchdir_if_needed(path):
    """ Prepends $SCRATCHDIR if not in path """

    path_obj = Path(path)
    scratch_dir_obj = Path(os.getenv('SCRATCHDIR'))

    if str(scratch_dir_obj) not in str(path_obj):
        path = str(scratch_dir_obj / path_obj)

    return path

def save_gbis_plotdata(eos_file, geo_vel_file, start_date_mod, end_date_mod):
    timeseries_file = eos_file.rsplit('/', 1)[0] + '/timeseries_tropHgt_demErr.h5'
    vel_file = geo_vel_file.replace('geo_','')
    geom_file = vel_file.replace('velocity','inputs/geometryRadar')
    print('eos_file', eos_file)

    cmd = f'timeseries2velocity.py {timeseries_file} --start-date {start_date_mod} --end-date {end_date_mod} --output {vel_file}' 
    cmd1 = f'save_gbis.py {vel_file} -g {os.path.dirname(eos_file)}/inputs/geometryRadar.h5' 
    print('timeseries2velocity command:',cmd)
    output = subprocess.check_output(cmd.split())
    print('save_gbis command:',cmd1.split())
    output = subprocess.check_output(cmd1.split())

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
  
def find_nearest_start_end_date(fname, period):
    ''' Find nearest dates to start and end dates given as YYYYMMDD '''
    
    dateList = HDFEOS(fname).get_date_list()
    
    if period:
        # period = [val for val in period.split('-')]         # converts to period=['20220101', '20221101']
        start_date = period[0]
        end_date = period[1]

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
        mod_start_date = dateList[0]
        mod_end_date = dateList[-1]

    print('###############################################')
    print(' Period of data:  ', dateList[0], dateList[-1])
    if period:
        print(' Period requested:', start_date, end_date) 
    else:
        print(' Period requested:', period)
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
        if file == 'up.h5':
            type = 'Up'
        elif file == 'hz.h5':
            type = 'Horz'
        else:
            type = 'Dem'
            #raise Exception('ERROR: file not up.h5 or horz.h5 -- exiting: ' + file)  
           
    return type

def get_plot_box(data_dict):
    ''' get plot_box from data_dict '''
    plot_box = []
    file = next(iter(data_dict))        # get first key
    atr = readfile.read_attribute(file)
    plot_box = [float(atr['Y_FIRST']) + int(atr['FILE_LENGTH'])*float(atr['Y_STEP']), float(atr['Y_FIRST']), 
        float(atr['X_FIRST']), float(atr['X_FIRST']) + int(atr['WIDTH'])*float(atr['X_STEP'])] 
    return plot_box

def get_dem_extent(atr_dem):
    # get the extent which is required for plotting
    # [-156.0, -154.99, 18.99, 20.00]
    dem_extent = [float(atr_dem['X_FIRST']), float(atr_dem['X_FIRST']) + int(atr_dem['WIDTH'])*float(atr_dem['X_STEP']), 
        float(atr_dem['Y_FIRST']) + int(atr_dem['FILE_LENGTH'])*float(atr_dem['Y_STEP']), float(atr_dem['Y_FIRST'])] 
    return(dem_extent)     

