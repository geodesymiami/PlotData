import pygmt
import numpy as np
from datetime import datetime
from plotdata.objects.section import Section
from plotdata.objects.create_map import Mapper, Isolines, Relief
from plotdata.objects.earthquakes import Earthquake
from mintpy.utils import readfile
from mintpy.objects.coord import coordinate
from mintpy.objects import timeseries, HDFEOS
from plotdata.helper_functions import draw_vectors, unpack_file


class ShadedReliefPlot:
    """Handles the generation of a shaded relief map."""
    def __init__(self, ax, file, inps):
        self.ax = ax
        self.file = unpack_file(file)
        self.region = inps.region
        self.resolution = inps.resolution
        self.interpolate = inps.interpolate
        self.no_shade = inps.no_shade
        self.iso_color = inps.iso_color
        self.linewidth = inps.linewidth
        self.isolines = inps.isolines
        self.inline = inps.inline
        self.seismicity = inps.seismicity
        self.start_date = inps.start_date if inps.start_date else None
        self.end_date = inps.end_date if inps.end_date else None

        self._create_map()

    def _create_map(self):
        """Create and configure the shaded relief map."""
        rel_map = Mapper(ax=self.ax, file=self.file, start_date=self.start_date, end_date=self.end_date, region=self.region)

        # Add relief and isolines
        Relief(map=rel_map, resolution=self.resolution, interpolate=self.interpolate, no_shade=self.no_shade)
        if self.isolines>0:
            Isolines(map=rel_map, resolution=self.resolution, color=self.iso_color, linewidth=self.linewidth,
                 levels=self.isolines, inline=self.inline)

        # Add earthquake markers if enabled
        if self.seismicity:
            Earthquake(map=rel_map, magnitude=self.seismicity).map()


class VelocityPlot:
    """Handles the plotting of velocity maps."""
    def __init__(self, ax, file, inps):
        self.ax = ax
        self.file = file
        self.resolution = inps.resolution
        self.interpolate = inps.interpolate
        self.no_shade = inps.no_shade
        self.style = inps.style
        self.vmin = inps.vmin
        self.vmax = inps.vmax
        self.movement = inps.movement
        self.isolines = inps.isolines
        self.iso_color = inps.iso_color
        self.linewidth = inps.linewidth
        self.inline = inps.inline
        self.seismicity = inps.seismicity
        self.no_dem = inps.no_dem
        self.lalo = inps.lalo
        self.ref_lalo = inps.ref_lalo

        self.map = self._create_map()

        if self.lalo:
            self.plot_point([self.lalo[0]], [self.lalo[1]], marker='x')
        if self.ref_lalo:
            self.plot_point([self.ref_lalo[0]], [self.ref_lalo[1]], marker='s')

    def _create_map(self):
        """Creates and configures the velocity map."""
        vel_map = Mapper(ax=self.ax, file=self.file)

        # Add relief if not disabled
        if not self.no_dem:
            Relief(map=vel_map, resolution=self.resolution, cmap='terrain',
                   interpolate=self.interpolate, no_shade=self.no_shade)

        # Add velocity file
        vel_map.add_file(style=self.style, vmin=self.vmin, vmax=self.vmax, movement=self.movement)

        # Add isolines if specified
        if self.isolines:
            Isolines(map=vel_map, resolution=self.resolution, color=self.iso_color, 
                     linewidth=self.linewidth, levels=self.isolines, inline=self.inline)

        # Add earthquake markers if enabled
        if self.seismicity:
            Earthquake(map=vel_map, magnitude=self.seismicity).map()

        if self.lalo:
            self.lat = self.lalo[0]
            self.lon = self.lalo[1]
            self.ax.scatter(self.lon, self.lat, color='black', marker='x')

        return vel_map

    def plot_point(self, lat, lon, marker='o'):
        """Plots a point on the map."""
        for x,y in zip(lon, lat):
            self.ax.scatter(x, y, color='black', marker=marker)


class VectorsPlot:
    """Handles the plotting of velocity maps, elevation profiles, and vector fields."""
    def __init__(self, ax, asc_file, desc_file, horz_file, vert_file, inps):
        self.ax = ax  # Expecting an array of axes
        self.asc_file = asc_file
        self.desc_file = desc_file
        self.horz_file = horz_file
        self.vert_file = vert_file
        self.ref_lalo = inps.ref_lalo
        self.seismicity = inps.seismicity
        self.inps = inps

        # Determine which files to plot
        self._set_plot_files()

        # Process the datasets
        self._process_velocity_maps()

        self._process_sections()

        # Compute and plot vectors
        self._compute_vectors()
        self._plot_vectors()

    def _set_plot_files(self):
        """Determines which velocity files to use based on plot options."""
        if self.inps.plot_option != 'horzvert':
            self.plot1_file = self.asc_file
            self.plot2_file = self.desc_file
        else:
            self.plot1_file = self.horz_file
            self.plot2_file = self.vert_file

    def plot_point(self, lat, lon, ax, marker='o'):
        """Plots a point on the map."""
        for x,y in zip(lon, lat):
            ax.scatter(x, y, color='black', marker=marker)


    def _create_map(self, ax, file):
        """Creates and configures a velocity map."""
        vel_map = Mapper(ax=ax, file=file)

        # Add relief if not disabled
        if not self.inps.no_dem:
            Relief(map=vel_map, resolution=self.inps.resolution, cmap='terrain',
                   interpolate=self.inps.interpolate, no_shade=self.inps.no_shade)

        # Add velocity file
        vel_map.add_file(style=self.inps.style, vmin=self.inps.vmin, vmax=self.inps.vmax, movement=self.inps.movement)

        # Add isolines if specified
        if self.inps.isolines:
            Isolines(map=vel_map, resolution=self.inps.resolution, color=self.inps.iso_color, 
                     linewidth=self.inps.linewidth, levels=self.inps.isolines, inline=self.inps.inline)

        # Add earthquake markers if enabled
        if self.seismicity:
            Earthquake(map=vel_map, magnitude=self.seismicity).map()

        if self.ref_lalo:
            self.plot_point([self.ref_lalo[0]], [self.ref_lalo[1]], ax, marker='s')

        return vel_map

    def _process_velocity_maps(self):
        """Processes and plots velocity maps."""
        self.asc_data = self._create_map(ax=self.ax[0], file=self.plot1_file)
        self.desc_data = self._create_map(ax=self.ax[1], file=self.plot2_file)

    def _process_sections(self):
        """Processes horizontal, vertical, and elevation sections."""
        self.horizontal_data = Mapper(file=self.horz_file)
        self.vertical_data = Mapper(file=self.vert_file)
        self.elevation_data = Relief(map=self.horizontal_data, resolution=self.inps.resolution)

        self.horizontal_section = Section(
            self.horizontal_data.velocity, self.horizontal_data.region, self.inps.line[1], self.inps.line[0]
        )
        self.vertical_section = Section(
            self.vertical_data.velocity, self.vertical_data.region, self.inps.line[1], self.inps.line[0]
        )
        self.elevation_section = Section(
            self.elevation_data.elevation, self.elevation_data.map.region, self.inps.line[1], self.inps.line[0]
        )

    def _compute_vectors(self):
        """Computes velocity vectors and scaling factors."""
        self.x, self.v, self.h = draw_vectors(
            self.elevation_section.values, self.vertical_section.values, self.horizontal_section.values, self.inps.line
        )

        fig = self.ax[0].get_figure()
        fig_width, fig_height = fig.get_size_inches()
        max_elevation = max(self.elevation_section.values)
        max_x = max(self.x)

        self.v_adj = 2 * max_elevation / max_x
        self.h_adj = 1 / self.v_adj

        self.rescale_h = self.h_adj / fig_width
        self.rescale_v = self.v_adj / fig_height

        # Resample vectors
        for i in range(len(self.h)):
            if i % self.inps.resample_vector != 0:
                self.h[i] = 0
                self.v[i] = 0

        # Filter out zero-length vectors
        non_zero_indices = np.where((self.h != 0) | (self.v != 0))
        self.filtered_x = self.x[non_zero_indices]
        self.filtered_h = self.h[non_zero_indices]
        self.filtered_v = self.v[non_zero_indices]
        self.filtered_elevation = self.elevation_section.values[non_zero_indices]

    def _plot_vectors(self):
        """Plots elevation profile and velocity vectors."""
        # Plot elevation profile
        self.ax[2].plot(self.x, self.elevation_section.values, color='#a8a8a8')
        self.ax[2].set_ylim([0, 2 * max(self.elevation_section.values)])
        self.ax[2].set_xlim([min(self.x), max(self.x)])

        # Plot velocity vectors
        #Probably right one
        self.ax[2].quiver(
            self.filtered_x, self.filtered_elevation,
            self.filtered_h, self.filtered_v,
            color='#ff7366', scale_units='xy', width=(1 / 10**(2.5))
        )

        # TODO test better the scaling issues
        # self.ax[2].quiver(
        #     self.filtered_x, self.filtered_elevation,
        #     self.filtered_h * self.rescale_h, self.filtered_v * self.rescale_v,
        #     color='#ff7366', scale_units='xy', width=(1 / 10**(2.5))
        # )

        # Add profile lines to velocity maps
        for i in range(2):
            self.ax[i].plot(self.inps.line[0], self.inps.line[1], '--', linewidth=1, alpha=0.7, color='black')

        # Mean velocity vector
        start_x = max(self.x) * 0.1
        start_y = (2 * max(self.elevation_section.values) * 0.8)
        mean_velocity = abs(np.mean(self.filtered_h))

        self.ax[2].quiver([start_x], [start_y], [mean_velocity], [0], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        # self.ax[2].quiver([start_x], [start_y], [0], [abs(np.mean(self.filtered_v))], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        self.ax[2].text(start_x, start_y * 1.03, f"{round(mean_velocity, 3)} m/yr", color='black', ha='left', fontsize=8)


class TimeseriesPlot:
    def __init__(self, ax, file, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))

        self.file = file
        self.ax = ax

        self._plot_timeseries()

    def _extract_timeseries_data(self, file):
        """Extracts timeseries data from the given file."""
        atr = readfile.read_attribute(file)

        # Identify file type and open it
        if atr['FILE_TYPE'] == 'timeseries':
            obj = timeseries(file)
        elif atr['FILE_TYPE'] == 'HDFEOS':
            obj = HDFEOS(file)
        else:
            raise ValueError(f'Input file is {atr["FILE_TYPE"]}, not timeseries.')

        obj.open(print_msg=False)
        date_list = obj.dateList

        # Filter dates if start/end dates are provided
        self.start_date = self.start_date if self.start_date else date_list[0]
        self.end_date = self.end_date if self.end_date else date_list[-1]

        self.start_date = datetime.strptime(self.start_date, "%Y%m%d")
        self.end_date = datetime.strptime(self.end_date, "%Y%m%d")

        # We want the whole timeseries
        # date_list = [d for d in date_list if int(start_date) <= int(d) <= int(end_date)]

        # Extract timeseries data
        data, atr = readfile.read(file, datasetName=date_list)

        # Handle complex data
        if atr['DATA_TYPE'].startswith('complex'):
            print('Input data is complex, calculating amplitude.')
            data = np.abs(data)

        # Convert geocoordinates to radar
        coord = coordinate(atr)
        lalo = coord.geo2radar(lat=self.lalo[0], lon=self.lalo[1])

        # Reference data to a specific point if provided
        if self.ref_lalo:
            ref_yx = coord.geo2radar(lat=self.ref_lalo[0], lon=self.ref_lalo[1])
            ref_phase = data[:, ref_yx[0], ref_yx[1]]
            data -= np.tile(ref_phase.reshape(-1, 1, 1), (1, data.shape[-2], data.shape[-1]))
            print(f'Referenced data to point: {self.ref_lalo}')

        ts = data[:, lalo[0], lalo[1]]
        date_list = [datetime.strptime(date, "%Y%m%d") for date in date_list]

        return date_list, ts

    def _plot_timeseries(self):
        """Plots timeseries data on the last axis."""
        ax_ts = self.ax
        color = "#ff7366" if "SenA" in self.file else "#5190cb"
        label = "ascending" if "SenA" in self.file else "descending"
        offsets = 0

        dates, ts = self._extract_timeseries_data(self.file)
        ax_ts.scatter(dates, ts + offsets, color=color, marker='o', label=label, alpha=0.5, s=7)

        # Plot vertical lines
        ax_ts.axvline(self.start_date, color='#a8a8a8', linestyle='--', linewidth=0.2,alpha=0.5)
        ax_ts.axvline(self.end_date, color='#a8a8a8', linestyle='--', linewidth=0.2, alpha=0.5)

        # Fill area between the vertical lines with a rectangle
        ax_ts.axvspan(self.start_date, self.end_date, color='#a8a8a8', alpha=0.1)
        ax_ts.set_ylabel('LOS displacement (m)')
        ax_ts.legend(fontsize='x-small')


def point_on_globe(latitude, longitude, names=None, size='0.7'):
    fig = pygmt.Figure()

    # Set up orthographic projection centered on your point
    fig.basemap(
        region="d",  # Global domain
        projection=f"G{np.mean(longitude)}/{np.mean(latitude)}/15c",  # Centered on your coordinates
        frame="g"  # Show gridlines only
    )

    # Add continent borders with black lines
    fig.coast(
        # shorelines="1/1p,black",  # Continent borders
        land="#f7f2f26b",  # Land color
        water="#7cc0ff"  # Water color
    )

    # Plot your central point
    fig.plot(
        x=longitude,
        y=latitude,
        style=f"t{size}c",  # Triangle marker
        fill="#ff7366",  # Marker color
        # pen="1p,black"  # Outline pen
    )

        # Add names if provided
    if names:
        fig.text(
            x=longitude,
            y=latitude,
            text=names,
            font="10p,Helvetica-Bold,black",  # Font size, style, and color
            justify="LM",  # Left-middle alignment
            offset="0.2c/0.2c"  # Offset to avoid overlapping with markers
        )

    return fig

if __name__ == '__main__':
    # Example usage
    file = "/Users/giacomo/onedrive/scratch/Chiles-CerroNegroSenAT120/mintpy/S1_IW2_120_1184_1185_20170112_XXXXXXXX.he5"
    # file = "/Users/giacomo/onedrive/scratch/AgungBaturSenAT156/mintpy/S1_IW12_156_1154_1155_20170121_XXXXXXXX.he5"
    ref_point = [0.8389,-77.902]
    TimeseriesPlot(file, lalo=[0.8, -77.95], ref_lalo=ref_point)