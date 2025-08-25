#!/usr/bin/env python3
############################################################
# Program is part of PlotData                              #
# Author: Giacomo Di Silvestro, Falk Amelung Dec 2023      #
############################################################

import os
import re
import sys

# !!! The asgeo import breaks when called by readfile.py unless I do the following !!!
from osgeo import gdal, osr

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from mintpy.utils import readfile
from plotdata.volcano_functions import get_volcano_event
from plotdata.helper_functions import prepend_scratchdir_if_needed, get_eos5_file
from plotdata.utils.argument_parsers import add_date_arguments, add_location_arguments, add_plot_parameters_arguments, add_map_parameters_arguments, add_save_arguments,add_gps_arguments, add_seismicity_arguments

############################################################
EXAMPLE = """
example:
        plot_data.py MaunaLoaSenDT87/mintpy MaunaLoaSenAT124/mintpy --period 20181001:20191031 --ref-lalo 19.50068,-155.55856 --lalo 19.47373,-155.59617 --resolution=01s --contour=2 --section 19.45,-155.75:19.45,-155.35 --resample-vector 40 --seismicity=3

        Add events on timeseries plot:
        plot_data.py MaunaLoaSenDT87/mintpy MaunaLoaSenAT124/mintpy --template default  --period 20181001:20191031 --ref-lalo 19.50068 -155.55856 --resolution '01s' --contour 2 --lalo 19.461,-155.558 --num-vectors 40 --add-event 20181201 --event-magnitude 5.0

        # FOR GIACOMO TO TEST
        plot_data.py ChilesSenAT120/mintpy ChilesSenDT142/mintpy --period=20220101:20230831 --ref-lalo 0.8389,-77.902 --resolution '01s' --contour 2 --section 0.793,-77.968:0.793,-77.9309 --lalo 0.78632 -77.92867

"""

def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('data_dir', nargs='*', help='Directory(s) with InSAR data.\n')
    parser.add_argument('--dem', dest='dem_file', default=None, help='external DEM file (Default: geo/geo_geometryRadar.h5)')
    parser.add_argument('--lines', dest='line_file', default=None, help='fault file (Default: None, but plotdata/data/hawaii_lines_new.mat for Hawaii)')
    parser.add_argument('--mask-thresh', dest='mask_vmin', type=float, default=0.55, help='coherence threshold for masking (Default: 0.7)')
    parser.add_argument('--unit', dest='unit', default="cm/yr", help='InSAR units (Default: cm)')
    # parser.add_argument("--noreference", dest="show_reference_point",  action='store_false', default=True, help="hide reference point (default: False)" )
    parser.add_argument("--section", dest="line", type=str, default=None, help="Section coordinates for deformation vectors, LAT,LON:LAT,LON")
    parser.add_argument("--num-vectors", dest="resample_vector", type=int, default=1, help="resample factor for deformation vectors (default: %(default)s).")
    parser.add_argument("--id", type=int, default=None, help="ID of the plot volcano")

    parser = add_date_arguments(parser)
    parser = add_location_arguments(parser)
    parser = add_plot_parameters_arguments(parser)
    parser = add_map_parameters_arguments(parser)
    parser = add_save_arguments(parser)
    parser = add_gps_arguments(parser)
    parser = add_seismicity_arguments(parser)

    inps = parser.parse_args()


    if len(inps.data_dir) > 2:
        parser.error('USER ERROR: Too many files provided.')

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

    if inps.line:
        if ":" in inps.line:
            inps.line = parse_section(inps.line)
        else:
            try:
                inps.line = float(inps.line)
            except ValueError:
                msg = 'Section format not corret, it must be in the format LAT,LON:LAT,LON or LAT'
                raise ValueError(msg)

    if inps.period:
        for p in inps.period:
            delimiters = '[,:\-\s]'
            dates = re.split(delimiters, p)

            if len(dates[0]) and len(dates[1]) != 8:
                msg = 'Date format not valid, it must be in the format YYYYMMDD'
                raise ValueError(msg)

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
    # TODO to change
    if False:
        inps.style = 'ifgram'

    if inps.add_event:
        if not inps.event_magnitude:
            inps.event_magnitude = [None] * len(inps.add_event)
        elif len(inps.add_event) != len(inps.event_magnitude):
            msg = 'Number of events and magnitudes do not match'
            raise ValueError(msg)

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


def parse_section(section):
    """
    Parses the section string and extracts the coordinates.

    Args:
        section (str): The section string in the format "lon1 lon2 lat1 lat2".

    Returns:
        tuple: A tuple containing the coordinates as floats.
               The first two elements are the longitude coordinates,
               and the last two elements are the latitude coordinates.
    """
    latitude = []
    longitude = []
    for coord in section.split(':'):
        latitude.append(float(coord.split(',')[0]))
        longitude.append(float(coord.split(',')[1]))

    return [(min(longitude), max(longitude)), (latitude[0], latitude[1])]


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
    pol = polygon.replace("POLYGON((", "").replace("))", "").replace("'", "").replace('"', "")

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

def initialize_dates_from_files(inps):
    scratch = os.getenv('SCRATCHDIR')
    for path in inps.data_dir:
        full_path = prepend_scratchdir_if_needed(path)
        file = get_eos5_file(full_path, scratch)

        atr = readfile.read_attribute(file)
        if atr['START_DATE'] not in inps.start_date:
            inps.start_date.append(atr['START_DATE'])
        if atr['END_DATE'] not in inps.end_date:
            inps.end_date.append(atr['END_DATE'])

    if inps.start_date and inps.end_date:
        coupled_dates = list(zip(inps.start_date, inps.end_date))
        if (min(inps.start_date), max(inps.end_date)) not in coupled_dates:
            inps.start_date.append(min(inps.start_date))
            inps.end_date.append(max(inps.end_date))


def try_initialize_from_volcano(inps):
    volcano = get_volcano_event(None, volcanoId=inps.id, strength=0)
    eruptions = []

    if volcano:
        first_key = next(iter(volcano), None)
        volcano_data = volcano.get(first_key, {})

        if 'eruptions' in volcano_data and 'Start' in volcano_data['eruptions']:
            eruptions = volcano_data['eruptions']['Start']
        else:
            print("Key 'eruptions' or 'Start' not found in volcano data.")
            return False

        initialize_dates_from_files(inps)

        if not inps.start_date or not inps.end_date:
            return False

        start_date = datetime.strptime(inps.start_date[0], '%Y%m%d').date()
        end_date = datetime.strptime(inps.end_date[0], '%Y%m%d').date()

        for e in eruptions:
            if start_date <= e <= end_date:
                one_month_later = e + relativedelta(months=1)
                end_str = one_month_later.strftime('%Y%m%d')
                if (one_month_later < end_date):
                    inps.end_date.append(end_str)
                    inps.start_date.append(min(inps.start_date))

        return True
    return False


def populate_dates(inps):
    if inps.start_date and inps.end_date:
        return inps

    if hasattr(inps, "id") and inps.id:
        success = try_initialize_from_volcano(inps)
        if success:
            return inps

    # Fallback to file-based method
    initialize_dates_from_files(inps)

    return inps

######################### MAIN #############################

def main(iargs=None):
    # logging_function.log(os.getcwd(), os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1:]))

    inps = create_parser()

    from plotdata.objects.process_data import ProcessData
    from plotdata.objects.plot_properties import PlotTemplate, PlotRenderer
    from plotdata.objects.plotters import VelocityPlot, VectorsPlot, TimeseriesPlot, EarthquakePlot
    from plotdata.objects.get_methods import DataExtractor
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('Agg')

    ###### TEST ######
    # inps.template = "test"  # Use a test template for demonstration
    ##################

    # 2. Build template object
    template = PlotTemplate(inps.template)

    # 3. Instantiate plotters with shared data
    # The plotter_map defines the mapping of plot types to their respective classes and required attributes
    # Attributes refer to the type of input file to get from the ProcessData object
    plotter_map = {
        "ascending": {"class": VelocityPlot, "attributes": ["ascending"]},
        "descending": {"class": VelocityPlot, "attributes": ["descending"]},
        "horizontal": {"class": VelocityPlot, "attributes": ["horizontal"]},
        "vertical": {"class": VelocityPlot, "attributes": ["vertical"]},
        "timeseries": {"class": TimeseriesPlot, "attributes": ["eos_file_ascending", "eos_file_descending"]},
        "vectors": {"class": VectorsPlot, "attributes": ["horizontal", "vertical"]},
        "seismicmap": {"class": VelocityPlot, "attributes": ["ascending_geometry", "descending_geometry"]},
        "seismicity": {"class": EarthquakePlot, "attributes": ["ascending", "descending"]},
        ########
    }

    figures = {}
    processors = []

    inps = populate_dates(inps)

    for start_date, end_date in zip(inps.start_date, inps.end_date):
        process = ProcessData(inps, template.layout, start_date, end_date)
        processors.append(process)
        process.process()

        datafethched = DataExtractor(plotter_map, process)

        # 4. Use PlotRenderer to populate the axes
        renderer = PlotRenderer(datafethched, template)
        fig = renderer.render()

        figures[id(process)] = fig if isinstance(fig, list) else [fig]

    # 5. Save or show
    if inps.save == 'pdf':
        from matplotlib.backends.backend_pdf import PdfPages

        for processor in processors:
            process_id = id(processor)
            if process_id in figures:
                if len(figures[process_id]) > 1:
                    saving_path = os.path.join(
                        inps.outdir,
                        processor.project,
                        f"{processor.project}_{figures[process_id][0].get_axes()[0].get_label().split('.')[0]}_{processor.start_date}_{processor.end_date}.pdf"
                    )
                else:
                    saving_path = os.path.join(
                        inps.outdir,
                        processor.project,
                        f"{processor.project}_{inps.template}_{processor.start_date}_{processor.end_date}.pdf"
                    )

                with PdfPages(saving_path) as pdf:
                    for fig in figures[process_id]:
                        pdf.savefig(fig, bbox_inches='tight', dpi=inps.dpi, transparent=True)
                        plt.close(fig)

    elif inps.save == 'png':
        # Save each figure as a PNG file
        for processor in processors:
            process_id = id(processor)
            if process_id in figures:
                for fig in figures[process_id]:
                    if len(figures[process_id]) > 1:
                        png_path = os.path.join(inps.outdir,processor.project,f"{processor.project}_{fig.get_axes()[0].get_label().split('.')[0]}_{processor.start_date}_{processor.end_date}.png")
                    else:
                        png_path = os.path.join(inps.outdir,processor.project,f"{processor.project}_{inps.template}_{processor.start_date}_{processor.end_date}.png")
                    fig.savefig(png_path, bbox_inches='tight', dpi=inps.dpi, transparent=True)
                    plt.close(fig)

    print(matplotlib.get_backend())

    if inps.show_flag:
        plt.show()

############################################################

if __name__ == '__main__':
     main(iargs=sys.argv)
