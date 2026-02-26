import os
from mintpy.utils import readfile
from mintpy.cli import (
    reference_point, asc_desc2horz_vert, mask, geocode, timeseries2velocity as ts2v
)
from plotdata.helper_functions import (
    get_file_names, prepend_scratchdir_if_needed, find_nearest_start_end_date,
    find_longitude_degree, select_reference_point, create_geometry_file,
    create_mask_file, read_best_values, utm_to_latlon, latlon_to_utm_zone,
    meters_to_lon_deg, meters_to_lat_deg
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
        self.ascending_mask = None
        self.descending_mask = None
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
        for i, dir in enumerate(self.data_dir):
            work_dir = prepend_scratchdir_if_needed(dir)
            eos_file, vel_file, geometry_file, mask_file, project_base_dir, out_vel_file, inputs_folder = get_file_names(work_dir)
            self.directory = project_base_dir

            if True: # TODO overwrite option
                mask_file = None
            if self.mask and i < len(self.mask):
                mask_file = self.mask[i]

            self.file_info[dir] = {
                'eos_file': eos_file,
                'vel_file': vel_file,
                'mask_file': mask_file,
                'out_vel_file': out_vel_file,
                'geometry_file': geometry_file,
                'inputs_folder': inputs_folder,
                'project_base_dir': project_base_dir,
            }

    def process(self):
        """Processes InSAR data by handling velocity files and computing horizontal/vertical components if needed."""
        os.chdir(self.root_dir)

        # First pass: Process ascending and descending separately, storing results
        for dir, files in self.file_info.items():
            out_mskd_file, mask = self._process_data(files)
            self.file_info[dir]['mask_file'] = mask

            if 'SenA' in out_mskd_file or 'CskA' in out_mskd_file:
                self.ascending = out_mskd_file
                self.eos_file_ascending = files['eos_file']
                self.ascending_geometry = files['geometry_file']
                self.ascending_mask = files['mask_file']
            elif 'SenD' in out_mskd_file or 'CskD' in out_mskd_file:
                self.descending = out_mskd_file
                self.eos_file_descending = files['eos_file']
                self.descending_geometry = files['geometry_file']
                self.descending_mask = files['mask_file']

        masked_files = list(filter(lambda x: x is not None, [self.ascending, self.descending]))
        geo_masked_files = [file for file in masked_files if os.path.basename(file).startswith('geo_')]
        geometry_files = list(filter(lambda x: x is not None, [self.ascending_geometry, self.descending_geometry]))
        mask_files = list(filter(lambda x: x is not None, [self.ascending_mask, self.descending_mask]))

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
                for i,f in enumerate(mask_files):
                    if not f:
                        continue
                    if os.path.dirname(f) == os.path.dirname(file):
                        mask_files[i] = os.path.join(os.path.dirname(file), f"geo_{os.path.basename(file)}")
                self.ascending = next((file for file in geo_masked_files if 'SenA' in file or 'CskA' in file), None)
                self.descending = next((file for file in geo_masked_files if 'SenD' in file or 'CskD' in file), None)
                self.ascending_geometry = next((file for file in geometry_files if 'SenA' in file or 'CskA' in file), None)
                self.descending_geometry = next((file for file in geometry_files if 'SenD' in file or 'CskD' in file), None)
                self.ascending_mask = next((file for file in mask_files if 'SenA' in file or 'CskA' in file), None)
                self.descending_mask = next((file for file in mask_files if 'SenD' in file or 'CskD' in file), None)

        if self.model:
            self._read_model_input()

            if not self.no_sources:
                self.sources = self._read_model_parameters(self.model)

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

    def _read_model_parameters(self, folder):
        """
        Read model parameters and convert UTM coordinates to lat/lon
        and metric sizes (m) to angular distances (degrees).
        """

        # -------------------------------------------------
        # Read sources
        # -------------------------------------------------
        if os.path.exists(os.path.join(folder, 'VSM_best.csv')):
            sources = read_best_values(os.path.join(folder, 'VSM_best.csv'))
        elif os.path.exists(os.path.join(folder, 'VSM_mean.csv')):
            sources = read_best_values(os.path.join(folder, 'VSM_mean.csv'))
        else:
            print(f"VSM_best.csv not found in {folder}")
            return None

        # -------------------------------------------------
        # Determine reference lat/lon (for zone + scaling)
        # -------------------------------------------------
        if self.ref_lalo:
            ref_lat, ref_lon = self.ref_lalo
        elif self.region:
            ref_lat = (self.region[2] + self.region[3]) / 2
            ref_lon = (self.region[0] + self.region[1]) / 2
        else:
            raise ValueError(
                "Either ref_lalo or region must be provided "
                "to determine UTM zone and hemisphere."
            )

        zone_number, hemisphere = latlon_to_utm_zone(ref_lat, ref_lon)

        # -------------------------------------------------
        # Convert UTM → lat/lon + meters → degrees
        # -------------------------------------------------
        METRIC_PARAMS = {"radius", "s_axis_max", "length", "width"}

        for src_id, params in sources.items():

            # ---------------------------
            # Position conversion
            # ---------------------------
            if 'xcen' in params and 'ycen' in params:
                lat, lon = utm_to_latlon(
                    params['xcen'],
                    params['ycen'],
                    zone_number,
                    hemisphere,
                )
                params['ycen'] = lat
                params['xcen'] = lon

            if 'xtlc' in params and 'ytlc' in params:
                lat_tlc, lon_tlc = utm_to_latlon(
                    params['xtlc'],
                    params['ytlc'],
                    zone_number,
                    hemisphere,
                )
                params['ytlc'] = lat_tlc
                params['xtlc'] = lon_tlc

            # ---------------------------
            # Metric → degree conversion
            # ---------------------------
            # use source latitude if available, otherwise reference latitude
            lat0 = params.get('ycen', ref_lat)

            for key in METRIC_PARAMS:
                if key in params:
                    meters = params[key]

                    params[f"{key}"] = meters_to_lat_deg(meters)
                    params[f"{key}"] = meters_to_lon_deg(meters, lat0)

        return sources

    def _read_model_input(self):
        self.model = os.path.join(self.directory, f"{self.start_date}_{self.end_date}", "_".join(self.model) )

        with open(os.path.join(self.model, 'VSM_input.txt'), 'r') as i:
            lines = i.readlines()[1]
        lines = lines.replace(' \n', '').split(' ')

        for idx, line in enumerate(lines):
            if 'SenA' in line or 'CskA' in line:
                lines[idx] = 'ascending'
            elif 'SenD' in line or 'CskD' in line:
                lines[idx] = 'descending'

        synth_sar_files = [
            os.path.join(self.model, f) 
            for f in os.listdir(self.model) 
            if 'synth_sar' in f
        ]

        synth_sar_files.sort(key=lambda x: (
            0 if 'synth_sar1' in x else 
            1 if 'synth_sar2' in x else 
            2
        ))

        model_dict = {}
        if lines and synth_sar_files:
            if len(lines) == 1:
                model_dict[lines[0]] = synth_sar_files[0]
            elif len(lines) >= 2 and len(synth_sar_files) >= 2:
                sar1 = next((f for f in synth_sar_files if 'synth_sar1' in f), synth_sar_files[0])
                sar2 = next((f for f in synth_sar_files if 'synth_sar2' in f), synth_sar_files[1])

                model_dict = {
                    lines[0]: sar1,
                    lines[1]: sar2
                }

        if 'ascending' in model_dict:
            self.ascending_model = {'ascending': model_dict['ascending']}
        if 'descending' in model_dict:
            self.descending_model = {'descending': model_dict['descending']}

        if self.ascending:
            for f in os.listdir(os.path.dirname(self.ascending)):
                if 'downsampled' in f:
                    self.ascending_downsampled = os.path.join(os.path.dirname(self.ascending), f)

        if self.descending:
            for f in os.listdir(os.path.dirname(self.descending)):
                if 'downsampled' in f:
                    self.descending_downsampled = os.path.join(os.path.dirname(self.descending), f)

    def _process_data(self, files):
        """Processes a single dataset and returns the masked velocity file."""
        eos_file = files['eos_file']
        date_dir = os.path.join(os.path.dirname(files['out_vel_file']), f'{self.start_date}_{self.end_date}')
        os.makedirs(date_dir, exist_ok=True)
        out_vel_file = os.path.join(date_dir, os.path.basename(files['out_vel_file']))

        mask = files['mask_file']

        create_geometry_file(eos_file, os.path.dirname(files['geometry_file']))
        if not mask:
            mask = create_mask_file(eos_file, os.path.dirname(files['out_vel_file']), self.mask_vmin)

        start_date, end_date = find_nearest_start_end_date(eos_file, self.start_date, self.end_date)
        self._convert_timeseries_to_velocity(eos_file, start_date, end_date, out_vel_file)

        out_mskd_file = self._apply_mask(out_vel_file, mask)

        return out_mskd_file, mask

    def _process_vectors(self, asc_file, desc_file, project_base_dir):
        """Computes horizontal and vertical velocity components from ascending and descending data."""
        horz_name = os.path.join(project_base_dir, f'hz_{self.start_date}_{self.end_date}.h5')
        vert_name = os.path.join(project_base_dir, f'up_{self.start_date}_{self.end_date}.h5')

        # TODO Overwrite option
        if not os.path.exists(horz_name) or not os.path.exists(vert_name) or True:
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