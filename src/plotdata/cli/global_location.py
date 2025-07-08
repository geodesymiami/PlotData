#!/usr/bin/env python3

#################################################################
# Data extracted from:                                          #
# Global Volcanism Program, 2024.                               #
# [Database] Volcanoes of the World (v. 5.2.3; 20 Sep 2024).    #
# Distributed by Smithsonian Institution, compiled by Venzke, E.#
# https://doi.org/10.5479/si.GVP.VOTW5-2024.5.2                 #
# Source: https://volcano.si.edu/                               #
#################################################################

# Global Volcanism Program, 2024. [Database] Volcanoes of the World (v. 5.2.3; 20 Sep 2024). Distributed by Smithsonian Institution, compiled by Venzke, E. https://doi.org/10.5479/si.GVP.VOTW5-2024.5.2
from plotdata.volcano_functions import get_volcano_coord_name
from plotdata.objects.plotters import point_on_globe
import argparse
import os
import pygmt


SCRATCHDIR = os.getenv('SCRATCHDIR')
JSON_VOLCANO = 'volcanoes.json'
EXAMPLE = f"""
Plot volcanoes on the Earth
"""


def create_parser(iargs=None, namespace=None):
    """ 
    Creates command line argument parser object.

    Args:
        iargs (list): List of command line arguments (default: None)
        namespace (argparse.Namespace): Namespace object to store parsed arguments (default: None)

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Plot the volcanoes on the Earth',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)

    parser.add_argument('id',
                        nargs='*',
                        type=str,
                        help='Select the id of the vlocanoes you want to plot, e.g. 284305 355040')
    parser.add_argument('--lalo',
                        default=None,
                        type=str,
                        metavar='LAT1:LON1,LAT2:LON2',
                        help='Select the latitude and longitude of the volcanoes you want to plot, e.g. 37.75:-122.25')
    parser.add_argument('--name',
                        action='store_true',
                        help='Show names of volcanoes on the map')
    parser.add_argument('--size',
                        default=0.7,
                        type=float,
                        help='Size of the volcanoes on the map')
    parser.add_argument('--fsize',
                        default=10,
                        type=int,
                        help='Font size of the volcano names on the map')

    inps = parser.parse_args(iargs, namespace)

    inps.coords = []

    if inps.lalo:
        for coords in inps.lalo.split(','):
            try:
                inps.coords.append(list(map(float, coords.split(':'))))
                print(inps.coords)
            except ValueError:
                print(f"Invalid format for --lalo: {coords}. Use 'lat:lon' format.")
                exit(1)


    return inps


def main(iargs=None, namespace=None):
    """
    Main function to check precipitation files.

    Args:
        iargs (list): List of command line arguments (default: None)
        namespace (argparse.Namespace): Namespace object to store parsed arguments (default: None)
    """
    inps = create_parser(iargs, namespace)

    coords = []

    if inps.id:
        volcanoes = [get_volcano_coord_name(None, id_) for id_ in inps.id]
        coords = [v[0] for v in volcanoes]
        names = [v[1] for v in volcanoes]

        latitudes = [v[0] for v in coords]
        longitudes = [v[1] for v in coords]

    if inps.coords:
        latitudes = [coord[0] for coord in inps.coords]
        longitudes = [coord[1] for coord in inps.coords]

    if not inps.name:
        names = None

    fig = point_on_globe(latitudes, longitudes, names=names, size=inps.size, fsize=inps.fsize)
    fig.show()


if __name__ == "__main__":
    main()