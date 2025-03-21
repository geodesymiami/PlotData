#!/usr/bin/env python3
# Authors: Farzaneh Aziz Zanjani & Falk Amelung
# This script plots velocity, DEM error, and estimated elevation on the backscatter.
############################################################
import argparse
import os
import sys
import re
import time
from plotdata import logging_function

'''
PLOT REPO TODO:
    Subparser for editing style/format parameters
        o fig size
        o font size
        o point size
        o color map
    either as subparser or create parser that handles argparse.ArugmentParser
'''
EXAMPLE = """example:
            viewPS.py S1*PS.he5 --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 velocity --vlim -0.6 0.6
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124 --mask ../maskPS.h5
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026,-80.122124 --mask maskTempCoh.h5 --vlim -4 4
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --vlim -3 3 --ref-lalo 25.87609,-80.12213 --dem ../../DEM/MiamiBeach.tif
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --vlim -3 3 --ref-lalo 25.87609,-80.12213 --dem ../../DEM/MiamiBeach.tif --dem-noshade
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026,-80.122124 --lalo 25.878307,-80.121460 25.878176,-80.121483 --ylim -2 6
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --backscatter
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --satellite
            viewPS.py S1*PS.he5 elevation --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 dem_error --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 height --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026,-80.122124 --kml-2d
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026,-80.122124 --kml-3d
            viewPS.py S1*PS.he5 --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --save
            """
ADDITIONAL_TEXT = (
            "need to modify save_hdf5eos.py to include demErr.h5 in S1*PS.he5 file"
            )
DESCRIPTION = (
    "Plots PS displacement, velocity, DEM error or estimated elevation on open_street_map, geoTiff or backscatter."
)
def create_parser():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, epilog=EXAMPLE + ADDITIONAL_TEXT,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('data_file', metavar='FILE', help='HDFEOS data file.\n')
    parser.add_argument('dataset', metavar='DATASET', nargs='?', default='displacement',
                        help='dataset to plot [displacement, elevation, demErr, velocity] (Default: displacement).\n')
    parser.add_argument( "--subset-lalo", type=str, default=None, help="Latitude and longitude box in format 'lat1:lat2,lon1:lon2' (default: None)")
    parser.add_argument('--period', dest='period', metavar='YYYYMMDD-YYYYMMDD', default=None, help='time period (Default: full time period)')
    parser.add_argument("--mask", metavar='FILE', type=str, default='../maskPS.h5', help="Mask file. Default: ../maskPS.h5",)
    parser.add_argument("--geometry-file", metavar='FILE', type=str, default='inputs/geometryRadar.h5', help="Geolocation file (default: inputs/geometryRadar.h5)",)
    parser.add_argument("--ref-lalo", nargs='*',  metavar=('LAT,LON or LAT LON'), type=str, default=None, help="reference point (default: use existing reference point)")
    parser.add_argument("--lalo", nargs='*',  metavar=('LAT,LON or LAT LON or LAT1,LON1  LAT2,LON2'), type=str, default=None, help="lat/lon coords of  pixel for timeseries  (default: ?)")
    parser.add_argument("--no-marker-number", dest="marker_number",  action='store_false', default=True, help="add marker numbers to points (default: False)" )
    parser.add_argument("--noreference", dest="show_reference_point",  action='store_false', default=True, help="hide reference point (default: False)" )
    parser.add_argument("--geoid-height", metavar='NUM', type=float, default=-26,help="geoid height (default: -26 m for Miami)")
    parser.add_argument("--estimated-elevation", dest="estimated_elevation_flag", action='store_true', help="Display estimated elevation (default: False)")
    parser.add_argument("--satellite", dest="satellite", action='store_true', help="Satellite as background (default: open_streep_map)")
    parser.add_argument("--backscatter", dest="backscatter", action='store_true', help="Backscatter as background (default: open_streep_map)")
    parser.add_argument("--dem", dest="dem_file", type=str, metavar='FILE', default=None,help="Shaded relief/elevation as background (default: None)")
    parser.add_argument('--dem-noshade', dest='disp_dem_shade', action='store_false', help='do not show DEM shaded relief')
    parser.add_argument("--out-amplitude", metavar='FILE', type=str, default="mean_amplitude.npy", help="slcStack amplitude file (default: mean_amplitude.npy)")
    parser.add_argument("--vlim", nargs=2, metavar=("VMIN", "VMAX"), default=None,type=float, help="Velocity limit for the colorbar (default: None)")
    parser.add_argument("--ylim", nargs=2, metavar=("YMIN", "YMAX"), default=None,type=float, help="Y-axis limits for point plotting (default: None)")
    parser.add_argument("--kml-2d", dest="kml_2d",  action='store_true', default=False, help="create 2D color-coded kml file (default: False)")
    parser.add_argument("--kml-3d", dest="kml_3d",  action='store_true', default=False, help="create a 3D color-coded kml file (reads demErr.h5) (default: False)" )
    parser.add_argument("--correct-geo", dest="correct_geo",  action='store_true', default=False, help="correct geolocation using DEM error (default: False)")
    parser.add_argument("--flip-lr", dest="flip_lr",  action='store_true', default=False, help="Flip the figure Left-Right (default: False)" )
    parser.add_argument("--flip-ud", dest="flip_ud",  action='store_true', default=False, help="Flip the figure Up-Down (default: False)")
    parser.add_argument("--colormap", "-c", metavar="", type=str, default="jet",help="Colormap used for display (default: jet)")
    parser.add_argument("--point-size", metavar='NUM', default=50, type=float, help="Point size (Default: 50  (20 for backscatter))",)
    parser.add_argument("--fontsize", "-f", metavar="", type=float, default=10, help="Font size (Default: 10)")
    parser.add_argument("--figsize", metavar=("WID", "LEN"), type=float, nargs=2, default=(5,10), help="Width and length of the figure (default: 5 10)")
    parser.add_argument('--outfile', type=str,  default=None, help="filename to save figure (default=scatter.png).")
    parser.add_argument('--save', dest='save_fig', action='store_true',help='save the figure')
    parser.add_argument('--dpi', dest='fig_dpi', metavar='DPI', type=int, default=300, help='dot per inch for display/write (default: %(default)s).')
    # parser.add_argument('--nodisplay', dest='disp_fig', action='store_false',
    #                 help='save and do not display the figure')
    parser.add_argument('--nowhitespace', dest='disp_whitespace', action='store_false', help='do not display white space')

    inps = parser.parse_args()

    if inps.ref_lalo:
        inps.ref_lalo = parse_lalo(inps.ref_lalo)
        if len(inps.ref_lalo) == 1:   # if given as one string containing ','
            inps.ref_lalo = inps.ref_lalo[0]
    if inps.lalo:
        inps.lalo = parse_lalo(inps.lalo)
    # set background based on satellite, backscatter, dem
    inps.background = 'open_street_map'
    if  inps.satellite:
        inps.background = 'satellite'
    if  inps.backscatter:
        inps.background = 'backscatter'
        inps.point_size = 15
        inps.figsize = (5, 5)
    if  inps.dem_file:
        inps.background = 'dem'

    # # check: coupled option behaviors (FA: form view.py, not implemented because it was unclear
    # if not inps.disp_fig or inps.outfile:
    #     inps.save_fig = True

    if not inps.outfile:
        inps.outfile = 'scatter.png'

    inps.marker_list=['X','1','2','3','4','X','1','2','3','4']
    return inps

def parse_lalo(str_lalo):
    """Parse the lat/lon input from the command line."""
    if ',' in str_lalo[0]:
        lalo = [[float(coord) for coord in pair.split(',')] for pair in str_lalo]
    else:
        lalo = [[float(str_lalo[i]), float(str_lalo[i+1])] for i in range(0, len(str_lalo), 2)]
    if len(lalo) == 1:     # if given as one string containing ','
        lalo = lalo[0]
    return lalo

###################################################################################
def main(iargs=None):

    logging_function.log(os.getcwd(), os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1:]))

    # parse
    inps = create_parser()

    # import  (sys.path.pop(0) would remove the cli directory from sys.path if identical file names)

    from plotdata.persistent_scatterers import persistent_scatterers
    # run
    persistent_scatterers(inps)
    return

   ################################################################################
if __name__ == '__main__':
    main()
