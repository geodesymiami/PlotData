import os

def add_date_arguments(parser):
    """
    Argument parser for the date range of the search.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        argparse.ArgumentParser: The argument parser object with added date arguments.
    """
    date = parser.add_argument_group('Date range of the search')
    date.add_argument('--start-date',
                        nargs='*',
                        metavar='YYYYMMDD',
                        type=str,
                        default=[],
                        help='Start date of the search')
    date.add_argument('--end-date',
                        nargs='*',
                        metavar='YYYYMMDD',
                        type=str,
                        default=[],
                        help='End date of the search')
    date.add_argument('--period',
                        nargs='*',
                        metavar='YYYYMMDD:YYYYMMDD, YYYYMMDD,YYYYMMDD',
                        type=str,
                        help='Period of the search')

    return parser


def add_location_arguments(parser):
    """
    Argument parser for the location of the volcano or area of interest.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        argparse.ArgumentParser: The argument parser object with added location arguments.
    """
    location = parser.add_argument_group('Location of the volcano or area of interest')
    location.add_argument('--latitude',
                        nargs='?',
                        metavar=('LATITUDE'),
                        help='Latitude')
    location.add_argument('--longitude',
                        nargs='?',
                        metavar=('LONGITUDE'),
                        help='Longitude')
    location.add_argument('--polygon',
                        metavar='POLYGON',
                        help='Polygon of the wanted area (Format from ASF Vertex Tool https://search.asf.alaska.edu/#/)')
    location.add_argument('--subset-lalo',
                        nargs='?',
                        dest='plot_box',
                        type=str,
                        help='Geographic area plotted')
    location.add_argument('--ref-lalo',
                        nargs='*',
                        metavar=('LATITUDE,LONGITUDE or LATITUDE LONGITUDE'),
                        default=None,
                        type=str,
                        help='reference point (default:  existing reference point)')
    location.add_argument("--lalo",
                        nargs='*',
                        default=None,
                        metavar=('LAT,LON or LAT LON or LAT1,LON1  LAT2,LON2'),
                        type=str,
                        help="lat/lon coords of  pixel for timeseries")
    location.add_argument('--window_size',
                        dest='window_size',
                        type=int,
                        default=3,
                        help='window size (square side in number of pixels) for reference point look up (default: %(default)s).')
    location.add_argument('--lat-step',
                        dest='lat_step',
                        type=float,
                        default=None,
                        help='latitude step for geocoding (default: %(default)s).')

    return parser


def add_plot_parameters_arguments(parser):
    """
    Argument parser for the plot parameters.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        argparse.ArgumentParser: The argument parser object with added plot parameters arguments.
    """
    plot_parameters = parser.add_argument_group('Plot parameters')
    plot_parameters.add_argument('--add-event',
                        nargs='*',
                        metavar=('YYYYMMDD, YYYY-MM-DD'),
                        help='Add event to the time series')
    plot_parameters.add_argument('--style',
                        default='scatter',
                        choices=['pixel', 'scatter', 'ifgram'],
                        help='Style of the plot (default: %(default)s).')
    plot_parameters.add_argument('--no-show',
                        dest='show_flag',
                        action='store_false',
                        default=True,
                        help='Do not show the plot')
    plot_parameters.add_argument('--fontsize',
                        dest='font_size',
                        default=15,
                        type=int,
                        help='fontsize for view.py (default: %(default)s).')
    plot_parameters.add_argument('--plot-option',
                        default=None,
                        metavar='horizontal, vertical, horzvert',
                        help='Limit the plot to horizontal or vertical component or replace ascending and descending with horizontal and vertical')
    plot_parameters.add_argument('--movement',
                        default='velocity',
                        choices=['velocity', 'displacement'],
                        help='Type of movement (default: %(default)s).')
    return parser


def add_map_parameters_arguments(parser):
    """
    Argument parser for the map parameters.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        argparse.ArgumentParser: The argument parser object with added map parameters arguments.
    """
    map_parameters = parser.add_argument_group('Map parameters')
    map_parameters.add_argument('--vlim',
                        nargs=2,
                        metavar=('VMIN', 'VMAX'),
                        help='Velocity limit for the colorbar')
    map_parameters.add_argument('--isolines',
                        nargs='?',
                        default=0,
                        type=int,
                        metavar='LEVELS',
                        help='Number of isolines to be plotted on the map (default: %(default)s).')
    map_parameters.add_argument('--colorbar',
                        default='viridis',
                        metavar='COLORBAR',
                        help='Colorbar (default: %(default)s).')
    map_parameters.add_argument('--isolines-color',
                        dest='iso_color',
                        type=str,
                        default='black',
                        metavar='COLOR',
                        help='Color of contour lines (default: %(default)s).')
    map_parameters.add_argument('--linewidth',
                        type=float,
                        default=0.5,
                        help='Line width for isolines (default: %(default)s).')
    map_parameters.add_argument('--inline',
                        action='store_true',
                        help='Display isolines inline')
    map_parameters.add_argument('--resolution',
                        type=str,
                        default='01m',
                        help='Resolution for the map (default: %(default)s).')
    map_parameters.add_argument('--color',
                        type=str,
                        default='black',
                        help='Color for the map (default: %(default)s).')
    map_parameters.add_argument('--scatter-size',
                        dest='scatter_marker_size',
                        type=float,
                        metavar='SIZE',
                        default=10,
                        help='Scatter marker size in points**2 (default: %(default)s).')
    map_parameters.add_argument('--no-dem',
                        action='store_true',
                        help='Add relief to the map')
    map_parameters.add_argument('--interpolate',
                        action='store_true',
                        help='Increase the resolution of the shaded dem')
    map_parameters.add_argument('--no-shade',
                        action='store_true',
                        help='Shade the dem')


    return parser


def add_save_arguments(parser):
    """
    Argument parser for the save options.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        argparse.ArgumentParser: The argument parser object with added save arguments.
    """
    save = parser.add_argument_group('Save options')
    save.add_argument('--save',
                      action='store_true',
                      default=False,
                      help=f'Save the plots (default path: {os.getenv("SCRATCHDIR")}).')
    save.add_argument('--outdir',
                      type=str,
                      default=os.getenv("SCRATCHDIR"),
                      metavar='PATH',
                      help='Folder to save the plot (default: %(default)s).')
    save.add_argument('--save-gbis',
                      dest='flag_save_gbis',
                      action='store_true',
                      default=False,
                      help='save GBIS files')

    return parser


def add_gps_arguments(parser):
    """
    Add GPS-related arguments to the given argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which the GPS arguments will be added.
    """
    gps = parser.add_argument_group('GPS options')
    gps.add_argument('--gps',
                        dest='flag_gps',
                        action='store_true',
                        help='flag to add GPS vectors')
    gps.add_argument('--gps-scale-fac',
                        dest='gps_scale_fac',
                        default=500,
                        type=int,
                        help='GPS scale factor (default: %(default)s).')
    gps.add_argument('--gps-key-length',
                        dest='gps_key_length',
                        default=4,
                        type=int,
                        help='GPS key length (default: %(default)s).')
    gps.add_argument('--gps-units',
                        dest='gps_unit',
                        default="cm",
                        help='GPS units (default: %(default)s).')
    gps.add_argument('--gps-dir',
                        dest='gps_dir',
                        default=None,
                        help='GPS directory (default: %(default)s).')

    return parser


def add_seismicity_arguments(parser):
    """
    Add seismicity arguments to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add the seismicity arguments to.

    Returns:
        argparse.ArgumentParser: The updated argument parser.
    """
    seismicity = parser.add_argument_group('GPS options')
    seismicity.add_argument('--seismicity',
                            nargs='?',
                            type=int,
                            default=None,
                            help='Add seismicity to the plot with magnitude above specified value (default: %(default)s).'
                            )

    return parser