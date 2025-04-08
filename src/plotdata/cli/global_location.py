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
                        help='Select the id of the vlocanoes you want to plot, e.g. 1234, 5678')

    inps = parser.parse_args(iargs, namespace)


    return inps


def main(iargs=None, namespace=None):
    """
    Main function to check precipitation files.

    Args:
        iargs (list): List of command line arguments (default: None)
        namespace (argparse.Namespace): Namespace object to store parsed arguments (default: None)
    """
    inps = create_parser(iargs, namespace)

    volcanoes = [get_volcano_coord_name(None, id_)[0] for id_ in inps.id]
    latitudes = [v[0] for v in volcanoes]
    longitudes = [v[1] for v in volcanoes]

    fig = point_on_globe(latitudes, longitudes)
    fig.show()


if __name__ == "__main__":
    main()