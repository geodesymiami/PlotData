import os
from mintpy.utils import readfile
from mintpy.cli import (
    reference_point, asc_desc2horz_vert, mask, geocode, timeseries2velocity as ts2v
)
from plotdata.helper_functions import (
    get_file_names, prepend_scratchdir_if_needed, find_nearest_start_end_date,
    find_longitude_degree, select_reference_point, create_geometry_file,
    create_mask_file
)


class ProcessData:
    def __init__(self, inps, layout, start_date=None, end_date=None):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))
        self.root_dir = os.getenv('SCRATCHDIR')
        self.start_date = start_date
        self.end_date = end_date

        # Attributes to store processing results
        self.ascending = None
        self.descending = None
        self.ascending_geometry = None
        self.descending_geometry = None
        self.horizontal = None
        self.vertical = None
        self.velocity_file = None
        self.directory = None
        self.project = None
        self.layout = layout

        # Extract file names once for all cases
        self._extract_file_names()

    def _extract_file_names(self):
        """Extracts necessary file names for each dataset directory."""
        self.file_info = {}
        for dir in self.data_dir:
            work_dir = prepend_scratchdir_if_needed(dir)
            eos_file, vel_file, geometry_file, project_base_dir, out_vel_file, inputs_folder = get_file_names(work_dir)
            self.directory = project_base_dir
            self.file_info[dir] = {
                'eos_file': eos_file,
                'vel_file': vel_file,
                'geometry_file': geometry_file,
                'project_base_dir': project_base_dir,
                'out_vel_file': out_vel_file,
                'inputs_folder': inputs_folder
            }

    def process(self):
        """Processes InSAR data by handling velocity files and computing horizontal/vertical components if needed."""
        os.chdir(self.root_dir)

        # First pass: Process ascending and descending separately, storing results
        for dir, files in self.file_info.items():
            out_mskd_file = self._process_data(files)

            if ('SenA' or 'CskA') in out_mskd_file:
                self.ascending = out_mskd_file
                self.eos_file_ascending = files['eos_file']
                self.ascending_geometry = files['geometry_file']
            elif ('SenD' or 'CskD') in out_mskd_file:
                self.descending = out_mskd_file
                self.eos_file_descending = files['eos_file']
                self.descending_geometry = files['geometry_file']

        masked_files = list(filter(lambda x: x is not None, [self.ascending, self.descending]))
        geo_masked_files = [file for file in masked_files if os.path.basename(file).startswith('geo_')]
        geometry_files = list(filter(lambda x: x is not None, [self.ascending_geometry, self.descending_geometry]))

        for file, geometry in zip(masked_files, geometry_files):
            metadata = readfile.read(file)[1] if os.path.exists(file) else None

            if not metadata or 'Y_STEP' not in metadata:
                for f in [file, geometry]:
                # Geocode velocity and geometry file
                    self._geocode_velocity_file(metadata, f, geometry)

                # Add 'geo_' to the front of the file name
                geo_masked_files.append(os.path.join(os.path.dirname(file), f"geo_{os.path.basename(file)}"))
                geo_geometry = os.path.join(os.path.dirname(geometry), f"geo_{os.path.basename(geometry)}")

                for i,f in enumerate(geometry_files):
                    if not f:
                        continue
                    if os.path.dirname(f) == os.path.dirname(geometry):
                        geometry_files[i] = geo_geometry
                self.ascending = next((file for file in geo_masked_files if 'SenA' in file or 'CskA' in file), None)
                self.descending = next((file for file in geo_masked_files if 'SenD' in file or 'CskD' in file), None)
                self.ascending_geometry = next((file for file in geometry_files if 'SenA' in file or 'CskA' in file), None)
                self.descending_geometry = next((file for file in geometry_files if 'SenD' in file or 'CskD' in file), None)

        if self.ref_lalo:
            self.ref_lalo = select_reference_point(geo_masked_files, self.window_size, self.ref_lalo)
            for file, ref_lalo in zip(geo_masked_files, self.ref_lalo):
                self._apply_reference_point(file, ref_lalo)
            self.ref_lalo = self.ref_lalo[0]

        # Assign directory and project name (assuming first dataset is representative)
        self.project = os.path.basename(self.directory) if self.directory else self.region

        # Second pass: Compute horizontal and vertical only if both asc & desc are available
        # TODO Probably have to remove the condition
        if any(('horizontal' in s or 'vertical' in s or 'vectors' in s) for sublist in self.layout for s in sublist) and self.ascending and self.descending:
            self.horizontal, self.vertical = self._process_vectors(self.ascending, self.descending, self.directory)

        if not self.file_info:
            self.velocity_file = [None]

    def _process_data(self, files):
        """Processes a single dataset and returns the masked velocity file."""
        eos_file = files['eos_file']
        date_dir = os.path.join(os.path.dirname(files['out_vel_file']), f'{self.start_date}_{self.end_date}')
        os.makedirs(date_dir, exist_ok=True)
        out_vel_file = os.path.join(date_dir, os.path.basename(files['out_vel_file']))

        create_geometry_file(eos_file, os.path.dirname(files['geometry_file']))
        mask = create_mask_file(eos_file, os.path.dirname(files['out_vel_file']), self.mask_vmin)

        start_date, end_date = find_nearest_start_end_date(eos_file, self.start_date, self.end_date)
        self._convert_timeseries_to_velocity(eos_file, start_date, end_date, out_vel_file)

        out_mskd_file = self._apply_mask(out_vel_file, mask)

        return out_mskd_file

    def _process_vectors(self, asc_file, desc_file, project_base_dir):
        """Computes horizontal and vertical velocity components from ascending and descending data."""
        horz_name = os.path.join(project_base_dir, f'hz_{self.start_date}_{self.end_date}.h5')
        vert_name = os.path.join(project_base_dir, f'up_{self.start_date}_{self.end_date}.h5')

        if not os.path.exists(horz_name) or not os.path.exists(vert_name):
            self._convert_to_horz_vert(asc_file, desc_file, horz_name, vert_name)

        return horz_name, vert_name

    def _convert_timeseries_to_velocity(self, eos_file, start_date, end_date, output_file):
        # TODO Overwrite option, this create conflicts if you use a new S1 file
        if not os.path.exists(output_file) or True:
            cmd = f'{eos_file} --start-date {start_date} --end-date {end_date} --output {output_file}'
            ts2v.main(cmd.split())

    def _geocode_velocity_file(self, metadata, file_fullpath, geometry):
        ref_lat = metadata.get('LAT_REF1', metadata.get('REF_LAT', None))

        lat_step =  metadata['mintpy.geocode.laloStep'].split(',')[0] if 'mintpy.geocode.laloStep' in metadata else self.lat_step
        lon_step = find_longitude_degree(ref_lat, lat_step)

        outdir = os.path.dirname(file_fullpath)
        os.chdir(outdir)
        cmd = f"{file_fullpath} --lalo-step {lat_step} {lon_step} --outdir {outdir} -l {geometry}"
        # TODO Overwrite option
        if not os.path.exists(outdir) or True:
            os.makedirs(outdir, exist_ok=True)
            geocode.main(cmd.split())
        os.chdir(self.root_dir)

    def _apply_mask(self, out_vel_file, temp_coh_file):
        out_mskd_file = out_vel_file.replace('.h5', '_msk.h5')
        if not os.path.exists(out_mskd_file) or True:
            cmd = f'{out_vel_file} --mask {temp_coh_file} --mask-vmin {self.mask_vmin} --outfile {out_mskd_file}'
            mask.main(cmd.split())
        return out_mskd_file

    def _apply_reference_point(self, out_mskd_file, ref_lalo):
        cmd = f'{out_mskd_file} --lat {ref_lalo[0]} --lon {ref_lalo[1]}'
        reference_point.main(cmd.split())

    def _convert_to_horz_vert(self, asc_file, desc_file, horz_name, vert_name):
        # TODO Overwrite option
        if not os.path.exists(horz_name) or not os.path.exists(vert_name) or True:
            cmd = f'{asc_file} {desc_file} --output {horz_name} {vert_name}'
            asc_desc2horz_vert.main(cmd.split())