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

import argparse
from datetime import datetime
from plotdata.utils.argument_parsers import add_date_arguments, add_location_arguments, add_plot_parameters_arguments, add_map_parameters_arguments, add_save_arguments,add_gps_arguments

############################################################
EXAMPLE = """example:
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=horzvert --ref-lalo 19.55,-155.45 --period 20220801:20221127 --resolution '01s' --isolines 2

        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=horzvert --period=20220101:20230831 --ref-lalo 0.8389,-77.902 --resolution '01s' --isolines 2

        plot_data.py MaunaLoaSenAT124/mintpy_5_20 --plot-type=velocity --period 20220101:20230831 20230831:20231001 --resolution '01s' --isolines 2 --section -77.968 -77.9309 0.793 0.793

        PLOT SHADED RELIEF AND ADD ISO-LINES
        give file
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 --plot-type=shaded_relief --period 20220101:20230831 --resolution '01s' --isolines 2

        give polygon
        plot_data.py --polygon "POLYGON((-155.8 19.3, -155.4 19.3, -155.4 19.6, -155.8 19.6, -155.8 19.3))" --plot-type=shaded_relief --period 20220101:20230831 --resolution '01s' --isolines 2

        PLOT BOTH VELOCITY FILES
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=velocity --period 20220101:20230831 --resolution '01s' --isolines 2

        PLOT BOTH VELOCITY FILES WITH DIFFERENT PERIODS
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=velocity --period 20220101:20230831 20200101:20220101 --resolution '01s' --isolines 2

        PLOT HORIZONTAL AND VERTICAL VELOCITY
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=horzvert --period 20220101:20230831 --ref-lalo 19.50068 -155.55856 --resolution '01s' --isolines 2

        PLOT HORIZONTAL
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=horzvert --period 20220101:20230831 --ref-lalo 19.50068 -155.55856--resolution '01s' --isolines 2 --plot-option horizontal

        PLOT VECTORS
        default: ascending and descending
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=vectors --period 20220101:20230831 --ref-lalo 19.50068 -155.55856 --resolution '01s' --isolines 2 --section -77.968 -77.9309 0.793 0.793

        plot horizontal and vertical instead
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 MaunaLoaSenAT124/mintpy_5_20 --plot-type=vectors --period 20220101:20230831 --ref-lalo 19.50068 -155.55856 --resolution '01s' --isolines 2 --section -77.968 -77.9309 0.793 0.793 --plot-option horzvert

        ADD EARTHQUAKES
        plot_data.py MaunaLoaSenDT87/mintpy_5_20 --plot-type=velocity --period 20220101:20230831  --ref-lalo 19.50068 -155.55856 --resolution '01s' --earthquake

        # FOR GIACOMO TO TEST
        plot_data.py Chiles-CerroNegroSenAT120/mintpy Chiles-CerroNegroSenDT142/mintpy --plot-type=horzvert --period=20220101:20230831 --ref-lalo 0.8389,-77.902 --resolution '01s' --isolines 2 --section -77.968 -77.9309 0.793 0.793

"""

def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('data_dir', nargs='*', help='Directory(s) with InSAR data.\n')
    parser.add_argument('--seismicity', dest='flag_seismicity', action='store_true', default=False, help='flag to add seismicity')
    parser.add_argument('--dem', dest='dem_file', default=None, help='external DEM file (Default: geo/geo_geometryRadar.h5)')
    parser.add_argument('--lines', dest='line_file', default=None, help='fault file (Default: None, but plotdata/data/hawaii_lines_new.mat for Hawaii)')
    parser.add_argument('--mask-thresh', dest='mask_vmin', type=float, default=0.7, help='coherence threshold for masking (Default: 0.7)')
    # parser.add_argument('--unit', dest='unit', default="cm", help='InSAR units (Default: cm)')
    # parser.add_argument("--noreference", dest="show_reference_point",  action='store_false', default=True, help="hide reference point (default: False)" )
    parser.add_argument("--section", dest="line", nargs='*', type=float, default=None, help="Section coordinates for deformation vectors")
    parser.add_argument("--resample-vector", dest="resample_vector", type=int, default=1, help="resample factor for deformation vectors (default: %(default)s).")
    parser.add_argument("--earthquake", dest="earthquake", action='store_true', default=False, help="plot earthquakes")

    parser = add_date_arguments(parser)
    parser = add_location_arguments(parser)
    parser = add_plot_parameters_arguments(parser)
    parser = add_map_parameters_arguments(parser)
    parser = add_save_arguments(parser)
    parser = add_gps_arguments(parser)

    inps = parser.parse_args()

    if inps.plot_type == 'vectors':
        if not inps.line:
            parser.error('USER ERROR: Section coordinates are required for deformation vectors')

    if len(inps.data_dir) > 2:
        parser.error('USER ERROR: To many files provided.')

    if inps.line:
        if len(inps.line) != 4:
            parser.error('USER ERROR: Section need 4 args, LON1 LON2, LAT1, LAT2')

    if inps.plot_box:
        inps.plot_box = [float(val) for val in inps.plot_box.replace(':', ',').split(',')]  # converts to plot_box=[19.3, 19.6, -155.8, -155.4]

    if inps.polygon:
        inps.region = parse_polygon(inps.polygon)
    else:
        inps.region = None

    if inps.lalo:
        inps.lalo = parse_lalo(inps.lalo)

    if inps.ref_lalo:
        inps.ref_lalo = parse_lalo(inps.ref_lalo)

    if inps.dem_file and '$' in inps.dem_file:
        inps.dem_file = os.path.expandvars(inps.dem_file)

    inps.vmax = max(inps.vlim) if inps.vlim else None
    inps.vmin = min(inps.vlim) if inps.vlim else None

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

    if True:
        from plotdata.objects.process_data import ProcessData

        processors = []
        for start_date, end_date in zip(inps.start_date, inps.end_date):
            processors.append(ProcessData(inps, start_date, end_date))

        for process in processors:
            process.process()

        if inps.show_flag:
            from plotdata.objects.plot_properties import PlotGrid
            from plotdata.objects.plotters import VelocityPlot, ShadedReliefPlot, VectorsPlot, TimeseriesPlot
            import matplotlib.pyplot as plt

            # Create plot grid, with columns as processors and rows as files
            pltgr = PlotGrid(processors=processors)

            # Select the correct plotter and corresponding files for rows
            plotter_map = {
                "velocity": (VelocityPlot, ["ascending", "descending"]),
                "horzvert": (VelocityPlot, ["horizontal", "vertical"]),
                "vectors": (VectorsPlot, ["ascending", "descending","horizontal", "vertical"]),
                "timeseries": [
                    (TimeseriesPlot, ["eos_file_ascending", "eos_file_descending"]),
                    (VelocityPlot, ["ascending", "descending"])
                ],
                "shaded_relief": (ShadedReliefPlot, ["velocity_file"]),  # Example
            }

            # Get plotter configuration for the selected plot type
            plotter_entries = plotter_map.get(inps.plot_type)
            if not plotter_entries:
                raise ValueError(f"Unsupported plot type: {inps.plot_type}")
            if not isinstance(plotter_entries, list):
                plotter_entries = [plotter_entries]

            # Iterate over each column (i.e., each processor)
            for col_idx, process in enumerate(processors):
                # For each plotter class and its corresponding file attributes
                for plotter_cls, file_attrs in plotter_entries:
                    files = [getattr(process, attr, None) for attr in file_attrs]
                    files = list(filter(lambda x: x is not None, files))

                    if plotter_cls is VectorsPlot:
                        if all(files):
                            plotter_cls(
                                ax=pltgr.axes[:, col_idx],
                                asc_file=files[0],
                                desc_file=files[1],
                                horz_file=files[2],
                                vert_file=files[3],
                                inps=process
                            )

                    elif plotter_cls is TimeseriesPlot:
                        for file in files:
                            if file:
                                plotter_cls(
                                    ax=pltgr.axes[-1, col_idx],  # Always plot in the last row
                                    file=file,
                                    inps=process
                                )

                    else:  # Default case (e.g., VelocityPlot, ShadedReliefPlot)
                        for row_idx, file in enumerate(files):
                            print(row_idx, file)
                            if file:
                                plotter_cls(
                                    ax=pltgr.axes[row_idx, col_idx],
                                    file=file,
                                    inps=process
                                )

            if inps.save:
                saving_path = os.path.join(inps.outdir, processors[0].project + '_' + inps.plot_type + '_' + inps.start_date[0] + '_' + inps.end_date[-1])
                print(f"Saving image in {saving_path}.png")
                plt.savefig(inps.outdir)

            plt.show()

############################################################

if __name__ == '__main__':
    main(iargs=sys.argv)