import os
from mintpy.utils import readfile
from mintpy.cli import (
    reference_point, asc_desc2horz_vert, save_gdal, mask, geocode, timeseries2velocity as ts2v
)
from plotdata.helper_functions import (
    get_file_names, prepend_scratchdir_if_needed, find_nearest_start_end_date,
    save_gbis_plotdata, find_longitude_degree, select_reference_point
)


class ProcessData:
    def __init__(self, inps, start_date=None, end_date=None):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))
        self.root_dir = os.getenv('SCRATCHDIR')
        self.start_date = start_date
        self.end_date = end_date

        # Attributes to store processing results
        self.ascending = None
        self.descending = None
        self.horizontal = None
        self.vertical = None
        self.directory = None
        self.velocity_file = None
        self.project = None

        # Extract file names once for all cases
        self._extract_file_names()

    def _extract_file_names(self):
        self.file_info = {}
        for dir in self.data_dir:
            work_dir = prepend_scratchdir_if_needed(dir)
            eos_file, vel_file, geometry_file, project_base_dir, out_vel_file, inputs_folder = get_file_names(work_dir)
            self.file_info[dir] = {
                'eos_file': eos_file,
                'vel_file': vel_file,
                'geometry_file': geometry_file,
                'project_base_dir': project_base_dir,
                'out_vel_file': out_vel_file,
                'inputs_folder': inputs_folder
            }

    def process(self):
        os.chdir(self.root_dir)
        for dir, files in self.file_info.items():
            self._process_data(files)

        if not  self.file_info:
            self.velocity_file = [None]

    def _process_data(self, files):
        eos_file = files['eos_file']
        vel_file = files['vel_file']
        out_vel_file = files['out_vel_file'].replace('.h5', f'_{self.start_date}_{self.end_date}.h5')
        project_base_dir = files['project_base_dir']

        if self.plot_type == 'shaded_relief':
            if self.start_date and self.end_date:
                start_date, end_date = find_nearest_start_end_date(eos_file, self.start_date, self.end_date)
                self._convert_timeseries_to_velocity(eos_file, start_date, end_date, out_vel_file)
                self.velocity_file = out_vel_file
            else:
                self.velocity_file = vel_file
            return

        temp_coh_file = out_vel_file.replace(f'velocity_{self.start_date}_{self.end_date}.h5', 'temporalCoherence.tif')
        start_date, end_date = find_nearest_start_end_date(eos_file, self.start_date, self.end_date)
        self._convert_timeseries_to_velocity(eos_file, start_date, end_date, out_vel_file)
        metadata = readfile.read(out_vel_file)[1] if os.path.exists(out_vel_file) else None

        if not metadata or 'Y_STEP' not in metadata:
            self._geocode_velocity_file(metadata, project_base_dir, files['vel_file'])

        if not os.path.exists(temp_coh_file):
            self._save_gdal(eos_file, temp_coh_file)

        out_mskd_file = self._apply_mask(out_vel_file, temp_coh_file)

        self.horizontal, self.vertical = self._process_vectors(out_mskd_file, project_base_dir)
        self.velocity_file = out_vel_file
        self.directory = project_base_dir
        self.project = project_base_dir.split('/')[-1]

        if self.flag_save_gbis:
            save_gbis_plotdata(eos_file, out_vel_file, start_date, end_date)

        if 'SenA' in out_mskd_file:
            self.ascending = out_mskd_file
        elif 'SenD' in out_mskd_file:
            self.descending = out_mskd_file

    def _convert_timeseries_to_velocity(self, eos_file, start_date, end_date, output_file):
        if not os.path.exists(output_file):
            cmd = f'{eos_file} --start-date {start_date} --end-date {end_date} --output {output_file}'
            ts2v.main(cmd.split())

    def _geocode_velocity_file(self, metadata, outdir, file_fullpath):
        ref_lat = metadata.get('LAT_REF1', metadata.get('REF_LAT', None))
        lat_step = metadata['mintpy.geocode.laloStep'].split(',')[0]
        lon_step = find_longitude_degree(ref_lat, lat_step)
        os.chdir(outdir)
        cmd = f"{file_fullpath} --lalo-step {lat_step} {lon_step} --outdir {outdir}"
        geocode.main(cmd.split())
        os.chdir(self.root_dir)

    def _save_gdal(self, eos_file, temp_coh_file):
        cmd = f'{eos_file} --dset temporalCoherence --output {temp_coh_file}'
        save_gdal.main(cmd.split())

    def _apply_mask(self, out_vel_file, temp_coh_file):
        out_mskd_file = out_vel_file.replace('.h5', '_msk.h5')
        if not os.path.exists(out_mskd_file):
            cmd = f'{out_vel_file} --mask {temp_coh_file} --mask-vmin {self.mask_vmin} --outfile {out_mskd_file}'
            mask.main(cmd.split())
        return out_mskd_file

    def _process_vectors(self, out_mskd_file, project_base_dir):
        horz_name = os.path.join(project_base_dir, f'hz_{self.start_date}_{self.end_date}.h5')
        vert_name = os.path.join(project_base_dir, f'up_{self.start_date}_{self.end_date}.h5')

        if self.plot_type in ['horzvert', 'vectors']:
            if not os.path.exists(horz_name) or not os.path.exists(vert_name):
                if self.ref_lalo:
                    select_reference_point([out_mskd_file], self.ref_lalo)
                    self._apply_reference_point(out_mskd_file)

                self._convert_to_horz_vert(out_mskd_file, horz_name, vert_name)

        if self.plot_option == 'horizontal':
            vert_name = None
        if self.plot_option == 'vertical':
            horz_name = None

        return horz_name, vert_name

    def _apply_reference_point(self, out_mskd_file):
        cmd = f'{out_mskd_file} --lat {self.ref_lalo[0]} --lon {self.ref_lalo[1]}'
        reference_point.main(cmd.split())

    def _convert_to_horz_vert(self, mskd_file, horz_name, vert_name):
        cmd = f'{mskd_file} {mskd_file} --output {horz_name} {vert_name}'
        asc_desc2horz_vert.main(cmd.split())