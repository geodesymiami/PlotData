#!/usr/bin/env python
# coding: utf-8

# ## Plot InSAR, GPS and seismicity data
# Assumes that the data are located in  `$SCRATCHDIR` (e.g.  `$SCRATCHDIR/MaunaLoaSenDT87/mintpy`).
# Output is  written into  `$SCRATCHDIR/MaunaLoa/SenDT87` and `$SCRATCHDIR/MaunaLoa/SenAT124`

import os
from mintpy.utils import readfile
from mintpy.cli import reference_point, asc_desc2horz_vert, save_gdal, mask, geocode
from plotdata.helper_functions import get_file_names
from plotdata.helper_functions import prepend_scratchdir_if_needed, find_nearest_start_end_date
from plotdata.helper_functions import  save_gbis_plotdata, find_longitude_degree, select_reference_point
from mintpy.cli import timeseries2velocity as ts2v


def run_prepare(inps):
    root_dir = os.getenv('SCRATCHDIR')
    os.chdir(root_dir)

    data_dir = inps.data_dir
    dem_file =  inps.dem_file if inps.dem_file else inps.data_dir[0] + '/geo/geo_geometryRadar.h5'
    plot_type = inps.plot_type
    ref_lalo = inps.ref_lalo
    mask_vmin = inps.mask_vmin
    flag_save_gbis =  inps.flag_save_gbis
    out_mskd_file = []
    horz_name = []
    vert_name = []
    project_base_dir = None
    plot_info = {}

    for start, end in zip(inps.start_date, inps.end_date):
        if plot_type != 'shaded_relief':
            for dir in data_dir:
                work_dir = prepend_scratchdir_if_needed(dir)
                eos_file, vel_file, geometry_file, project_base_dir, out_vel_file, inputs_folder = get_file_names(work_dir)
                out_vel_file = out_vel_file.replace('.h5', f'_{start}_{end}.h5')
                start_date, end_date = find_nearest_start_end_date(eos_file, start, end)
                temp_coh_file=out_vel_file.replace(f'velocity_{start}_{end}.h5','temporalCoherence.tif')
                metadata = None

                if start and end:
                    horz_name = os.path.join(project_base_dir, f'hz_{start}_{end}.h5')
                    vert_name = os.path.join(project_base_dir, f'up_{start}_{end}.h5')

                if os.path.exists(out_vel_file):
                    metadata = readfile.read(out_vel_file)[1]
                    if start_date != metadata['START_DATE'] or end_date != metadata['END_DATE']:
                        # Convert timeseries to velocity
                        run_timeseries2velocity(eos_file, start_date, end_date, out_vel_file)
                else:
                    run_timeseries2velocity(eos_file, start_date, end_date, out_vel_file)

                if not metadata:
                    metadata = readfile.read(out_vel_file)[1]

                if 'Y_STEP' in metadata:
                    print('-'*50)
                    print(f'{out_vel_file} already geocoded, skipping ...')

                # Geocode the velocity file
                else:
                    if not ref_lalo:
                        for key in ['LAT_REF1', 'REF_LAT']:
                            if key in metadata:
                                ref_lat = metadata[key]
                                break

                    lat_step = inps.lat_step if inps.lat_step else metadata['mintpy.geocode.laloStep'].split(',')[0]

                    run_geocode(ref_lat, lat_step, project_base_dir, vel_file)

                    # Go back to SCRACTDIR
                    os.chdir(root_dir)

                if not os.path.exists(temp_coh_file):
                    run_save_gdal(eos_file, temp_coh_file)

                if not os.path.exists(out_vel_file.replace('.h5', '_msk.h5')):
                    out_mskd_file.append(run_mask(out_vel_file, temp_coh_file, mask_vmin))

                else:
                    out_mskd_file.append(out_vel_file.replace('.h5', '_msk.h5'))

            if plot_type in ['horzvert','vectors']:
                if not os.path.exists(horz_name) or not os.path.exists(vert_name):
                    if len(out_mskd_file) != 2:
                        raise ValueError(f'Need two velocity files for {plot_type} plot')

                    if ref_lalo:
                        select_reference_point(out_mskd_file, inps.window_size, ref_lalo)

                        for geo_vel in out_mskd_file:
                            run_reference_point(geo_vel, ref_lalo)

                    run_asc_desc2horz_vert(out_mskd_file, horz_name=horz_name, vert_name=vert_name)

            if flag_save_gbis:
                for eos, vel in zip(eos_file, out_vel_file):
                    start_date, end_date = find_nearest_start_end_date(eos, start_date, end_date)
                    save_gbis_plotdata(eos, vel, start_date, end_date)

        plot_info[f"{start}:{end}"] = {
            'ascending': [item for item in out_mskd_file if 'SenA' in item],
            'descending': [item for item in out_mskd_file if 'SenD' in item],
            'horizontal': horz_name,
            'vertical': vert_name,
            'directory': project_base_dir,
            }
        out_mskd_file = []

    return plot_info


def run_timeseries2velocity(eos_file, start_date, end_date, output_file):
    cmd = f'{eos_file} --start-date {start_date} --end-date {end_date} --output {output_file}'
    ts2v.main(cmd.split())


def run_geocode(ref_lat, lat_step, outdir, file_fullpath):
    lon_step = find_longitude_degree(ref_lat, lat_step)

    # Go to folder with all the input files
    os.chdir(outdir)

    cmd = f"{file_fullpath} --lalo-step {lat_step} {lon_step} --outdir {outdir}"
    geocode.main(cmd.split())


def run_save_gdal(eos_file, temp_coh_file):
    cmd = f'{eos_file} --dset temporalCoherence --output {temp_coh_file}'
    save_gdal.main( cmd.split() )


def run_mask(out_vel_file, temp_coh_file, mask_vmin):
    out_mskd_file = out_vel_file.replace('.h5', '_msk.h5')
    cmd = f'{out_vel_file} --mask {temp_coh_file} --mask-vmin { mask_vmin} --outfile {out_mskd_file}'
    mask.main(cmd.split())

    return out_mskd_file


def run_reference_point(out_mskd_file, ref_lalo):
    cmd = f'{out_mskd_file} --lat {ref_lalo[0]} --lon {ref_lalo[1]}'
    reference_point.main( cmd.split() )


def run_asc_desc2horz_vert(mskd_file, horz_name, vert_name):
    cmd = f'{mskd_file[0]} {mskd_file[1]} --output {horz_name} {vert_name}'
    asc_desc2horz_vert.main( cmd.split() )