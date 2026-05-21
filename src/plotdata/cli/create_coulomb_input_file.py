#!/usr/bin/env python3

import os
import argparse
import numpy as np
from pathlib import Path
from mintpy.utils import readfile
from plotdata.helper_functions import read_best_values, convert_to_utm

FORMAT = (
    "{idx:3d} "
    "{x0:10.4f} {y0:10.4f} "
    "{x1:10.4f} {y1:10.4f} "
    "{kode:5d} "
    "{rt:8.1f} "
    "{rev:11.2E} "
    "{dip:8.2f} "
    "{top:10.1f} {bot:10.2f}"
)

def create_parser():
    parser = argparse.ArgumentParser(description="Generate Coulomb 3 input file from model and geometry.")
    parser.add_argument("dir", help="Directory containing model and geometry files")
    parser.add_argument("--period", help="Period to append to path (optional)")
    parser.add_argument("--model", required=True, help="Model name (e.g., penny, mogi, okada)")
    parser.add_argument("--geometry", dest='geometry_file', help="Geometry file name (optional, will search if not provided)")
    parser.add_argument("--mask", dest='mask_file', help="Mask file name (optional, will search if not provided)")
    parser.add_argument("--output", default=None, help="Output file")

    inps = parser.parse_args()

    if not inps.output:
        inps.output = os.path.join(os.getcwd(), "coulomb_input.inp")
    elif not os.path.isabs(inps.output):
        inps.output = os.path.join(os.getcwd(), inps.output)

    return inps

def resolve_dir(dir_arg, scratchdir=None):
	p = Path(dir_arg)
	if p.is_absolute():
		return p
	if scratchdir and (Path(scratchdir) / p).exists():
		return Path(scratchdir) / p

def return_mogi(source_params, xstep, ystep):
    v = source_params.get('dVol')

    center_x = source_params.get('xcen') / 1000.0
    center_y = source_params.get('ycen') / 1000.0
    center_d = source_params.get('depth') / 1000.0

    x_start = center_x - xstep
    y_start = center_y - ystep
    x_fin = center_x + xstep
    y_fin = center_y + ystep

    top = 0
    bot = 2 * center_d
    kode = 500
    rt_lat = 0
    reverse = round(v,2)
    dip = 90.0

    return x_start, y_start, x_fin, y_fin, kode, rt_lat, reverse, dip, top, bot


def main():
    inps = create_parser()

    scratch = os.getenv("SCRATCHDIR")

    # Resolve directory
    base_dir = resolve_dir(inps.dir, scratch)
    if inps.period:
        period_dir =base_dir / inps.period.replace(":", "_")

    # Find geometry file
    if inps.geometry_file:
        geometry_path = base_dir / inps.geometry_file
    else:
        files = os.listdir(base_dir)
        for f in files:
            if 'Sen' in f or 'Csk' in f:
                path = base_dir / f
                geometry_path = path / 'geo_geometryRadar.h5' if (path / 'geo_geometryRadar.h5').exists() else path / 'geometryRadar.h5'
                if not inps.mask_file:
                    mask_path = path / 'geo_maskTempCoh.h5' if (path / 'geo_maskTempCoh.h5').exists() else path / 'maskTempCoh.h5'
                else:
                    mask_path = inps.mask_file
                break


    # Read geometry attributes using mintpy.utils.readfile
    latitude, geometry_attr = readfile.read(str(geometry_path), datasetName='latitude')
    longitude = readfile.read(str(geometry_path), datasetName='longitude')[0]

    # Read source parameters
    model_dir =  period_dir / inps.model
    params_file =model_dir / "VSM_best.csv" if (model_dir / "VSM_best.csv").exists() else model_dir / "VSM_mean.csv"
    if not params_file.exists():
        raise FileNotFoundError(f"Source parameter file not found: {params_file}")
    params = ['dP_mu', 'dVol', 'opening', 'param1']
    source_params = next(iter(read_best_values(str(params_file), params).values()), {})

    x, y = convert_to_utm(longitude, latitude)

    xmin, xmax = np.nanmin(x)/1000, np.nanmax(x)/1000
    ymin, ymax = np.nanmin(y)/1000, np.nanmax(y)/1000

    xstep = (xmax - xmin) *0.05
    ystep = (ymax - ymin) * 0.05

    fault_rows = []

    if inps.model == 'mogi':
        x_start, y_start, x_fin, y_fin, kode, rt_lat, reverse, dip, top, bot = return_mogi(source_params, xstep, ystep)


    fault_rows.append(
        FORMAT.format(
            idx=1,
            x0=x_start, y0=y_start,
            x1=x_fin,   y1=y_fin,
            kode=kode,
            rt=rt_lat,
            rev=reverse,
            dip=dip,
            top=top,
            bot=bot,
        )
    )

    with open(inps.output, "w", encoding="utf-8") as f:
        f.write("This is a test file for the Coulomb 1.0\n")
        f.write("This file is prepared to check mainly thrust faulting calculation\n")
        f.write("#reg1=  0  #reg2=  0   #fixed=  1  sym=  1\n")
        f.write(" PR1=       .250      PR2=       .250    DEPTH=        7.5\n")
        f.write("  E1=   0.800000E+06   E2=   0.800000E+06\n")
        f.write("XSYM=       .000     YSYM=       .000\n")
        f.write("FRIC=       .400\n")
        f.write("S1DR=    19.0001     S1DP=     -0.0001    S1IN=    100.000     S1GD=   .000000\n")
        f.write("S3DR=    89.9999     S3DP=      89.999    S3IN=     30.000     S3GD=   .000000\n")
        f.write("S2DR=   109.0001     S2DP=     -0.0001    S2IN=      0.000     S2GD=   .000000\n")
        f.write("\n")
        f.write("  #   X-start    Y-start     X-fin      Y-fin   Kode  rt.lat    reverse   dip angle     top      bot\n")
        f.write("xxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx xxxxxxxxxx\n")

        for row in fault_rows:
            f.write(f"{row}\n")

        f.write("\n")
        f.write("    Grid Parameters\n")
        f.write(f"  1  ----------------------------  Start-x =    {xmin:10.5f}\n")
        f.write(f"  2  ----------------------------  Start-y =    {ymin:10.5f}\n")
        f.write(f"  3  --------------------------   Finish-x =    {xmax:10.5f}\n")
        f.write(f"  4  --------------------------   Finish-y =    {ymax:10.5f}\n")
        f.write(f"  5  ------------------------  x-increment =    {xstep:10.5f}\n")
        f.write(f"  6  ------------------------  y-increment =    {ystep:10.5f}\n")
        f.write("     Size Parameters\n")
        f.write("  1  --------------------------  Plot size =     2.00000\n")
        f.write("  2  --------------  Shade/Color increment =     1.00000\n")
        f.write("  3  ------  Exaggeration for disp.& dist. =  10000.00000\n")
        f.write("\n")
        f.write("Cross section default\n")
        f.write(f"  1  ----------------------------  Start-x =    {xmin:10.5f}\n")
        f.write(f"  2  ----------------------------  Start-y =    {ymin:10.5f}\n")
        f.write(f"  3  --------------------------   Finish-x =    {xmax:10.5f}\n")
        f.write(f"  4  --------------------------   Finish-y =    {ymax:10.5f}\n")
        f.write("  5  ------------------  Distant-increment =     1.00000\n")
        f.write("  6  ----------------------------  Z-depth =     30.00000\n")
        f.write("  7  ------------------------  Z-increment =     1.00000\n")

if __name__ == "__main__":
		main()