#!/usr/bin/env python3
############################################################
# Program is part of PlotData                              #
# Author: Falk Amelung, Giacomo Di Silvestro Dec 2023      #
############################################################

import os
import re
import sys

# The asgeo import breaks when called by readfile.py unless I do the following
from osgeo import gdal, osr

sys.path.insert(0, '/Users/giacomo/code/Playground/Plot_data2/src')
import argparse
from datetime import datetime
from plotdata.utils.argument_parsers import add_date_arguments, add_location_arguments, add_plot_parameters_arguments, add_map_parameters_arguments, add_save_arguments,add_gps_arguments

############################################################
EXAMPLE = """example:
        plot_data.py GalapagosSenDT128
        plot_data.py GalapagosSenDT128/mintpy
        plot_data.py GalapagosSenDT128/mintpy/S1_IW12_128_0593_0597_20181005_XXXXXXXX.he5
        plot_data.py GalapagosSenDT128/mintpy  --plot-type=velocity --subset-lalo=-0.52:-0.28,-91.7:-91.4 --period=20200131-20231231 --gps --seismicity
        plot_data.py GalapagosSenDT128/mintpy GalapagosSenAT106/mintpy_orig  --plot-type=horzvert --subset-lalo=-1.0:-0.75,-91.55:-91.25 --period=20220101-20230831 --vlim -5 5
        plot_data.py MaunaLoaSenDT87/mintpy/
        plot_data.py MaunaLoaSenDT87 --plot-type ifgram --seismicity --gps
        plot_data.py MaunaLoaSenDT87 --plot-type shaded_relief --seismicity --gps
        plot_data.py MaunaLoaSenDT87 --plot-type velocity --seismicity --gps
        plot_data.py MaunaLoaSenAT124 MaunaLoaSenDT87 --ref-lalo 19.55,-155.45
        plot_data.py MaunaLoaSenDT87/mintpy_5_20  --plot-type velocity
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type velocity --ref-lalo 19.495,-155.555  --period 20181001-20221122 --subset-lalo 19.43:19.5,-155.62:-155.55 --vlim -5 5
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type horzvert --ref-lalo 19.495,-155.555  --period 20181001-20221122 --subset-lalo 19.43:19.5,-155.62:-155.55 --vlim -5 5
        plot_data.py MaunaLoaSenDT87/mintpy_5_20  --plot-type shaded-relief --gps --period 20181001-20221122 --dem-file $SCRATCHDIR/MaunaLoa/MLtry/data/demGeo.h5
        plot_data.py MaunaLoaSenDT87/mintpy_5_20  --plot-type shaded-relief --gps --gps-scale-fac 200 --gps-key-length 1
        plot_data.py MaunaLoaSenDT87/mintpy_5_20  --plot-type shaded-relief --gps --seismicity
        plot_data.py MaunaLoaSenDT87/mintpy_5_20  --plot-type shaded-relief --subset-lalo 19.43:19.5,-155.62:-155.55  --seismicity
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type velocity --ref-lalo 19.55,-155.45 --period 20220801-20221127 --vlim -20 20 --save-gbis --gps --seismicity --fontsize 14
        plot_data.py GalapagosSenDT128/mintpy  --plot-type=velocity --subset-lalo=-0.52:-0.28,-91.7:-91.4 --period=20200131-20221231 --gps --seismicity
        plot_data.py GalapagosSenDT128/mintpy --plot-type=velocity --subset-lalo=-0.52:-0.28,-91.7:-91.4 --period=20200131-20220430
        plot_data.py GalapagosSenDT128/mintpy --plot-type=velocity --period=20200131-20220430
        plot_data.py GalapagosSenDT128/mintpy --plot-type=velocity --subset-lalo=-0.52:-0.28,-91.7:-91.4
        plot_data.py GalapagosSenDT128/mintpy --subset-lalo=-0.86:-0.77:-91.19:-91.07 --ref-lalo=-0.771,-91.19
"""

def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('data_dir', nargs='*', help='Directory(s) with InSAR data.\n')
    parser.add_argument('--seismicity', dest='flag_seismicity', action='store_true', default=False, help='flag to add seismicity')
    parser.add_argument('--dem', dest='dem_file', default=None, help='external DEM file (Default: geo/geo_geometryRadar.h5)')
    parser.add_argument('--lines', dest='line_file', default=None, help='fault file (Default: None, but plotdata/data/hawaii_lines_new.mat for Hawaii)')
    parser.add_argument('--unit', dest='unit', default="cm", help='InSAR units (Default: cm)')
    parser.add_argument('--mask-thresh', dest='mask_vmin', type=float, default=0.7, help='coherence threshold for masking (Default: 0.7)')
    parser.add_argument("--noreference", dest="show_reference_point",  action='store_false', default=True, help="hide reference point (default: False)" )
    parser.add_argument("--section", dest="line", nargs=4, metavar="LON1 LON2 LAT1 LAT2", type=float, default=None, help="Section coordinates for deformation vectors")
    parser.add_argument("--resample-vector", dest="resample_vector", type=int, default=1, help="resample factor for deformation vectors (default: %(default)s).")
    # parser.add_argument('--window_size', dest='window_size', type=int, default=3, help='window size (square side in number of pixels) for reference point look up (default: %(default)s).')
    # parser.add_argument('--lat-step', dest='lat_step', type=float, default=None, help='latitude step for geocoding (default: %(default)s).')
    # parser.add_argument('--subset-lalo',  nargs='?', dest='plot_box', type=str, default=None, help='geographic area plotted')
    # parser.add_argument('--gps', dest='flag_gps', action='store_true', default=False, help='flag to add GPS vectors')
    # parser.add_argument('--gps-scale-fac', dest='gps_scale_fac', default=500, type=int, help='GPS scale factor (Default: 500)')
    # parser.add_argument('--gps-key-length', dest='gps_key_length', default=4, type=int, help='GPS key length (Default: 4)')
    # parser.add_argument('--gps-units', dest='gps_unit', default="cm", help='GPS units (Default: cm)')
    # parser.add_argument('--fontsize', dest='font_size', default=12, type=int, help='fontsize for view.py (Default: 12)')
    # parser.add_argument('--ref-lalo', nargs='*',  metavar=('LAT,LON or LAT LON'), type=str, default=None, help='reference point (default:  existing reference point)')
    # parser.add_argument("--lalo", nargs='*',  metavar=('LAT,LON or LAT LON or LAT1,LON1  LAT2,LON2'), type=str, default=None, help="lat/lon coords of  pixel for timeseries  (default: ?)")
    # parser.add_argument('--vlim', dest='vlim', nargs=2, metavar=('VMIN', 'VMAX'), type=float, help='colorlimit')
    # parser.add_argument('--save-gbis', dest='flag_save_gbis', action='store_true', default=False, help='save GBIS files')
    # parser.add_argument('--style', dest='style', choices={'image', 'scatter'}, default='image', help='Plot data as image or scatter (default: %(default)s).')
    # parser.add_argument('--scatter-size', dest='scatter_marker_size', type=float, metavar='SIZE', default=10, help='Scatter marker size in points**2 (default: %(default)s).')
    # parser.add_argument('--shade-exag ',dest='shade_exag', type=float, default=1, help='Shade exaggeration factor (Default: 0.5)')

    parser = add_date_arguments(parser)
    parser = add_location_arguments(parser)
    parser = add_plot_parameters_arguments(parser)
    parser = add_map_parameters_arguments(parser)
    parser = add_save_arguments(parser)
    parser = add_gps_arguments(parser)

    inps = parser.parse_args()

    if len(inps.data_dir) < 1 or len(inps.data_dir) > 2:
        parser.error('USER ERROR: You must provide 1 or 2 directory paths.')

    if inps.plot_box:
        inps.plot_box = [float(val) for val in inps.plot_box.replace(':', ',').split(',')]  # converts to plot_box=[19.3, 19.6, -155.8, -155.4]

    if inps.polygon:
        inps.region = parse_polygon(inps.polygon)

    if inps.lalo:
        inps.lalo = parse_lalo(inps.lalo)

    if inps.ref_lalo:
        inps.ref_lalo = parse_lalo(inps.ref_lalo)

    if inps.dem_file and '$' in inps.dem_file:
        inps.dem_file = os.path.expandvars(inps.dem_file)

    if inps.period:
        for p in inps.period:
            delimiters = '[,:\-\s]'
            dates = re.split(delimiters, p)

            inps.start_date.append(dates[0])
            inps.end_date.append(dates[1])

    if inps.add_event:
        try:
            inps.add_event = tuple(datetime.strptime(date_string, '%Y-%m-%d').date() for date_string in inps.add_event)

        except ValueError:
            try:
                inps.add_event = tuple(datetime.strptime(date_string, '%Y%m%d').date() for date_string in inps.add_event)

            except ValueError:
                msg = 'Date format not valid, it must be in the format YYYYMMDD or YYYY-MM-DD'
                raise ValueError(msg)

    if inps.plot_type == 'ifgram':
        inps.style = 'ifgram'

    if inps.line:
        inps.line = [(inps.line[0],inps.line[1]), (inps.line[2],inps.line[3])]

##### Hardwired for Hawaii #####
    if 'GPSDIR' in os.environ:
        inps.gps_dir = os.getenv('GPSDIR') + '/data'
    else:
        inps.gps_dir = os.getenv('SCRATCHDIR') + '/MaunaLoa/MLtry/data'

    inps.gps_list_file = inps.gps_dir + '/GPS_BenBrooks_03-05full.txt'

    # TODO Hardwire line file for Hawaii data
    if ('Hawaii' in inps.data_dir or 'Mauna' in inps.data_dir or 'Kilauea' in inps.data_dir):
        inps.line_file = os.getenv('RSMASINSAR_HOME') + '/tools/PlotData' + '/data/hawaii_lines_new.mat'

    return inps
################################

def parse_polygon(polygon):
    """
    Parses a polygon string retrieved from ASF vertex tool and extracts the latitude and longitude coordinates.

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

    return [round(min(longitude),2), round(max(longitude),2), round(min(latitude),2), round(max(latitude),2)]


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


######################### MAIN #############################

def main(iargs=None):
    # logging_function.log(os.getcwd(), os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1:]))

    inps = create_parser()

    # import
    from Plot_data2.src.plotdata.process_data import run_prepare
    from Plot_data2.src.plotdata.plot import run_plot

    # extract_volcanoes_info('', 'Kilauea', inps.start_date, inps.end_date)
    plot_info = run_prepare(inps)

    if inps.show_flag:
        run_plot(plot_info, inps)

############################################################

if __name__ == '__main__':
    main(iargs=sys.argv)