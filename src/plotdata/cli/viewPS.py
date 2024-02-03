#!/usr/bin/env python3
# Authors: Farzaneh Aziz Zanjani & Falk Amelung
# This script plots velocity, DEM error, and estimated elevation on the backscatter.
############################################################
import argparse
import os
import sys
import re
import time
from minsar.objects import message_rsmas

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
            viewPS.py S1*PS.he5 velocity
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124 
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124 --mask ../maskPS.h5 
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124 --mask maskTempCoh.h5 --vlim -4 4
            viewPS.py S1*PS.he5 displacement --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --vlim -3 3 --ref-lalo 25.87609 -80.12213 --geotiff ../../DEM/MiamiBeach.tif
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --vlim -0.6 0.6 
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --backscatter 
            viewPS.py S1*PS.he5 elevation --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 dem_error --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 height --subset-lalo 25.8759:25.8787,-80.1223:-80.1205
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124 --kml-2d 
            viewPS.py S1*PS.he5 velocity --subset-lalo 25.8759:25.8787,-80.1223:-80.1205 --ref-lalo 25.876026 -80.122124 --kml-3d 
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
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'data_file', metavar='FILE', help='HDFEOS data file.\n'
    )
    parser.add_argument(
        'dataset', metavar='DATASET', nargs='?', default='displacement',  
        help='dataset to plot [displacement, elevation, demErr, velocity] (Default: displacement).\n'
    )
    parser.add_argument(
        "--subset-lalo", type=str, default=None,
        help="Latitude and longitude box in format 'lat1:lat2,lon1:lon2' (default: None)"
    )
    parser.add_argument(
        "--mask", metavar='FILE', type=str, default='../maskPS.h5',  
        help="Mask file. Default: ../maskPS.h5",
    )
    parser.add_argument(
        "--geometry-file", metavar='FILE', type=str, default='inputs/geometryRadar.h5', 
        help="Geolocation file (default: inputs/geometryRadar.h5)",
    )
    parser.add_argument( "--ref-lalo", nargs=2,  metavar=('LAT', 'LON'), type=float, 
        help="reference point (default: use existing reference point)"
    )
    parser.add_argument(
        "--dem-offset", metavar='NUM', type=float, default=26,
        help="DEM offset (geoid deviation) (default: 26 m for Miami)"
    )
    parser.add_argument(
        "--estimated-elevation", dest="estimated_elevation_flag", action='store_true',
        help="Display estimated elevation (default: False)"
    )
    parser.add_argument(
        "--backscatter", dest="backscatter", action='store_true',
        help="Use backscatter as background (default background: open_streep_map)"
    )
    parser.add_argument(
        "--geotiff", type=str, metavar='FILE', default=None,
        help="geotiff elevation file (default: None)",
    )
    parser.add_argument("--out-amplitude", metavar='FILE', type=str, default="mean_amplitude.npy",
        help="slcStack amplitude file (default: mean_amplitude.npy)",
    )
    parser.add_argument(
        "--vlim", nargs=2, metavar=("VMIN", "VMAX"), default=None,
        type=float, help="Velocity limit for the colorbar (default: None)",
    )
    parser.add_argument(
        "--kml-2d", dest="kml_2d",  action='store_true', default=False, 
        help="create a 2D color-coded kml file (default: False)" 
    )
    parser.add_argument(
        "--kml-3d", dest="kml_3d",  action='store_true', default=False, 
        help="create a 3D color-coded kml file (reads demErr.h5) (default: False)" 
    )
    parser.add_argument(
        "--correct-geo", dest="correct_geo",  action='store_true', default=False, 
        help="correct the geolocation using DEM error (default: False)" 
    )
    parser.add_argument(
        "--flip-lr", dest="flip_lr",  action='store_true', default=False, 
        help="Flip the figure Left-Right (default: False)" 
    )
    parser.add_argument("--flip-ud", dest="flip_ud",  action='store_true', default=False, 
        help="Flip the figure Up-Down (default: False)"
    )
    parser.add_argument(
        "--colormap", "-c", metavar="", type=str, default="jet",
        help="Colormap used for display (default: jet)",
    )
    parser.add_argument(
        "--point-size", metavar='NUM', default=50, type=float,
        help="Point size (Default: 50  (20 for backscatter))",
    )
    parser.add_argument(
        "--fontsize", "-f", metavar="", type=float, default=10,
        help="Font size (Default: 10)",
    )
    parser.add_argument(
        "--figsize", metavar=("WID", "LEN"), type=float, nargs=2,
        default=(5,10), help="Width and length of the figure"
    )
    parser.add_argument('--outfile', type=str,  default=None,
                    help="filename to save figure (default=scatter.png).")
    parser.add_argument('--save', dest='save_fig', action='store_true',
                    help='save the figure')
    parser.add_argument('--dpi', dest='fig_dpi', metavar='DPI', type=int, default=300,
                    help='DPI - dot per inch - for display/write (default: %(default)s).')
    # parser.add_argument('--nodisplay', dest='disp_fig', action='store_false',
    #                 help='save and do not display the figure')
    parser.add_argument('--nowhitespace', dest='disp_whitespace',
                    action='store_false', help='do not display white space')

    inps = parser.parse_args()
   
    # set background based on backscatter, geotiff
    inps.background = 'open_street_map'
    if  inps.backscatter:
        inps.background = 'backscatter'
        inps.point_size = 15
        inps.figsize = (5, 5)
    if  inps.geotiff:
        inps.background = 'geotiff'

    # # check: coupled option behaviors (FA: form view.py, not implemented because it was unclear
    # if not inps.disp_fig or inps.outfile:
    #     inps.save_fig = True

    if not inps.outfile:
        inps.outfile = 'scatter.png'
    
    return inps


###################################################################################
def main(iargs=None):
    
    message_rsmas.log(os.getcwd(), os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1:]))

    # parse
    inps = create_parser()

    # import  (sys.path.pop(0) would remove the cli directory from sys.path if identical file names)
    
    from plotdata.persistent_scatterers import persistent_scatterers
    # run
    persistent_scatterers(inps)

   ################################################################################
if __name__ == '__main__':
    main()
