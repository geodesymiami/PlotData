import pygmt
import numpy as np
from datetime import datetime
from plotdata.objects.section import Section
from plotdata.objects.create_map import Mapper, Isolines, Relief
from plotdata.objects.earthquakes import Earthquake
from mintpy.utils import readfile
from mintpy.objects.coord import coordinate
from matplotlib.patheffects import withStroke
from mintpy.objects import timeseries, HDFEOS
from plotdata.helper_functions import draw_vectors, unpack_file, calculate_distance


# TODO Create a class to just extract data
class DataFetcher:
    def __init__(self, template, plotter_map, process) -> None:
        plotters={}
        for name, configs in self.plotter_map.items():
            if any(name in row for row in self.template.layout):
                cls = configs["class"]
                files = [getattr(process, attr) for attr in configs["attributes"]]

                # plotter_instance = cls(*files, self.inps)
                plotter_instance = cls(files, self.inps)
                plotters[name] = (plotter_instance)


class ShadedReliefPlot:
    """Handles the generation of a shaded relief map."""
    def __init__(self, file, inps):
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

    def plot(self, ax):
        self.ax = ax
        """Create and configure the shaded relief map."""
        rel_map = Mapper(ax=self.ax, file=self.file, start_date=self.start_date, end_date=self.end_date, region=self.region)

        # Add relief and isolines
        Relief(map=rel_map, resolution=self.resolution, interpolate=self.interpolate, no_shade=self.no_shade)
        if self.isolines>0:
            Isolines(map=rel_map, resolution=self.resolution, color=self.iso_color, linewidth=self.linewidth,
                 levels=self.isolines, inline=self.inline)

        # Add earthquake markers if enabled
        if self.seismicity or 'seismicmap' in self.ax.get_label():
            if not self.seismicity:
                self.seismicity = 1
            Earthquake(map=rel_map, magnitude=self.seismicity).map(self.ax)


class VelocityPlot:
    """Handles the plotting of velocity maps."""
    def __init__(self, file: str, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))
        self.file = file[0] if isinstance(file, list) else file

    def on_click(self, event):
        if event.inaxes == self.ax:  # Ensure the click is within the correct axis
            if event.inaxes == self.ax:  # Ensure the click is within the plot
                print(f"--lalo={event.ydata},{event.xdata}\n")
                print(f"--ref-lalo={event.ydata},{event.xdata}\n")

    def plot(self, ax):
        """Creates and configures the velocity map."""
        self.ax = ax
        vel_map = Mapper(ax=self.ax, file=self.file)
        self.region = vel_map.region

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
            Earthquake(map=vel_map, magnitude=self.seismicity).map(ax=self.ax)

        if 'ascending' in self.ax.get_label():
            label = "ASCENDING"
        elif 'descending' in self.ax.get_label():
            label = "DESCENDING"
        elif 'horizontal' in self.ax.get_label():
            label = "HORIZONTAL"
        elif 'vertical' in self.ax.get_label():
            label = "VERTICAL"
        else:
            label = None

        if label:
            self.ax.annotate(
                label,
                xy=(0.02, 0.98),
                xycoords='axes fraction',
                fontsize=10,
                ha='left',
                va='top',
                color='white',
                bbox=dict(facecolor='gray', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3')
            )

        if self.lalo and 'point' in self.ax.get_label():
            vel_map.plot_point([self.lalo[0]], [self.lalo[1]], marker='x')

        if 'section' in self.ax.get_label():
            if not self.line:
                self.line = self._set_default_section()
            self.ax.plot(self.line[0], self.line[1], '--', linewidth=1.5, alpha=0.7, color='black')

        if self.ref_lalo:
            vel_map.plot_point([self.ref_lalo[0]], [self.ref_lalo[1]], marker='s')

        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        return vel_map

    def _set_default_section(self):
        mid_lat = (max(self.region[2:4]) + min(self.region[2:4]))/2
        mid_lon = (max(self.region[0:2]) + min(self.region[0:2]))/2

        size = (max(self.region[0:2]) - min(self.region[0:2]))*0.25

        latitude = (mid_lat, mid_lat)
        longitude = (mid_lon - size, mid_lon + size)

        return [longitude, latitude]


class VectorsPlot:
    """Handles the plotting of velocity maps, elevation profiles, and vector fields."""
    def __init__(self, files: list, inps):
        # TODO to add later and change self.inps references
        # for attr in dir(inps):
        #     if not attr.startswith('__') and not callable(getattr(inps, attr)):
        #         setattr(self, attr, getattr(inps, attr))
        # TODO have to add attributes to https://github.com/insarlab/MintPy/blob/main/src/mintpy/asc_desc2horz_vert.py#L261
        # for f in files:
        #     attr = readfile.read_attribute(f)
        #     if attr['FILE_TYPE'] == 'VERTICAL':
        #         self.vert_file = f
        #         self.horz_file = None
        for f in files:
            if 'hz' in f:
                self.horz_file = f
            elif 'up' in f:
                self.vert_file = f

        self.ref_lalo = inps.ref_lalo
        self.seismicity = inps.seismicity
        self.inps = inps

        # Determine which files to plot
        self._set_plot_files()

        # Process the datasets
        # self._process_velocity_maps()

        self._process_sections()

    def _set_plot_files(self):
        """Determines which velocity files to use based on plot options."""
        # if self.inps.plot_option != 'horzvert':
        #     self.plot1_file = self.asc_file
        #     self.plot2_file = self.desc_file
        # else:
        self.plot1_file = self.horz_file
        self.plot2_file = self.vert_file

    def plot_point(self, lat, lon, ax, marker='o'):
        """Plots a point on the map."""
        for x,y in zip(lon, lat):
            ax.scatter(x, y, color='black', marker=marker)
    # TODO to remove
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
            Earthquake(map=vel_map, magnitude=self.seismicity).map(self.ax)

        if self.ref_lalo:
            vel_map.plot_point([self.ref_lalo[0]], [self.ref_lalo[1]], marker='s')

        return vel_map
    # TODO to remove
    def _process_velocity_maps(self):
        """Processes and plots velocity maps."""
        self.asc_data = self._create_map(ax=self.ax[0], file=self.plot1_file)
        self.desc_data = self._create_map(ax=self.ax[1], file=self.plot2_file)

    def _process_sections(self):
        """Processes horizontal, vertical, and elevation sections."""
        self.horizontal_data = Mapper(file=self.horz_file)
        self.vertical_data = Mapper(file=self.vert_file)
        self.elevation_data = Relief(map=self.horizontal_data, resolution=self.inps.resolution)

        self.region = self.elevation_data.map.region

        if not self.inps.line:
            self.inps.line = self._set_default_section()

        self.horizontal_section = Section(
            np.flipud(self.horizontal_data.velocity), self.horizontal_data.region, self.inps.line[1], self.inps.line[0]
        )
        self.vertical_section = Section(
            np.flipud(self.vertical_data.velocity), self.vertical_data.region, self.inps.line[1], self.inps.line[0]
        )
        self.elevation_section = Section(
            self.elevation_data.elevation.values, self.elevation_data.map.region, self.inps.line[1], self.inps.line[0]
        )

    def _set_default_section(self):
        mid_lat = (max(self.region[2:4]) + min(self.region[2:4]))/2
        mid_lon = (max(self.region[0:2]) + min(self.region[0:2]))/2

        size = (max(self.region[0:2]) - min(self.region[0:2]))*0.25

        latitude = (mid_lat, mid_lat)
        longitude = (mid_lon - size, mid_lon + size)

        return [longitude, latitude]

    def _compute_vectors(self):
        """Computes velocity vectors and scaling factors."""
        self.x, self.v, self.h, self.z = draw_vectors(
            self.elevation_section.values, self.vertical_section.values, self.horizontal_section.values, self.inps.line
        )

        fig = self.ax.get_figure()
        fig_width, fig_height = fig.get_size_inches()
        max_elevation = max(self.z)
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

        distance = calculate_distance(self.inps.line[1][0], self.inps.line[0][0], self.inps.line[1][1], self.inps.line[0][1])
        self.xrange = np.linspace(0, distance, len(self.x))
        # Filter out zero-length vectors
        non_zero_indices = np.where((self.h != 0) | (self.v != 0))
        self.filtered_x = self.xrange[non_zero_indices]
        self.filtered_h = self.h[non_zero_indices]
        self.filtered_v = self.v[non_zero_indices]
        self.filtered_elevation = self.z[non_zero_indices]

    def plot(self, ax):
        """Plots elevation profile and velocity vectors."""
        self.ax = ax

        # Compute and plot vectors
        self._compute_vectors()

        # Plot elevation profile
        self.ax.plot(self.xrange, self.z, color='#a8a8a8', alpha=0.5)
        self.ax.set_ylim([0, 2 * max(self.z)])
        self.ax.set_xlim([min(self.xrange), max(self.xrange)])

        # Plot velocity vectors
        #Probably right one
        self.ax.quiver(
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
        # for i in range(2):
        #     self.ax[i].plot(self.inps.line[0], self.inps.line[1], '--', linewidth=1, alpha=0.7, color='black')

        # Mean velocity vector
        start_x = max(self.xrange) * 0.1
        start_y = (2 * max(self.z) * 0.8)
        mean_velocity = np.sqrt(np.mean((self.vertical_section.values))**2 + np.mean((self.horizontal_section.values))**2)

        self.ax.quiver([start_x], [start_y], [mean_velocity], [0], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        # self.ax[2].quiver([start_x], [start_y], [0], [abs(np.mean(self.filtered_v))], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        self.ax.text(start_x, start_y * 1.03, f"{round(mean_velocity, 3)} m/yr", color='black', ha='left', fontsize=8)

        # Add labels
        self.ax.set_ylabel("Elevation (m)")
        self.ax.set_xlabel("Distance (km)")


class TimeseriesPlot:
    def __init__(self, files, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))

        self.start_date = self.start_date[0] if isinstance(self.start_date, list) else self.start_date
        self.end_date = self.end_date[0] if isinstance(self.end_date, list) else self.end_date

        self.files = files

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

        self.start_date = datetime.strptime(self.start_date, "%Y%m%d") if type(self.start_date) == str else self.start_date
        self.end_date = datetime.strptime(self.end_date, "%Y%m%d") if type(self.end_date) == str else self.end_date

        # We want the whole timeseries
        # date_list = [d for d in date_list if int(start_date) <= int(d) <= int(end_date)]

        # Extract timeseries data
        data, atr = readfile.read(file, datasetName=date_list)

        # Handle complex data
        if atr['DATA_TYPE'].startswith('complex'):
            print('Input data is complex, calculating amplitude.')
            data = np.abs(data)

        if 'X_FIRST' not in atr:
            geometry = self.ascending_geometry if 'SenA' in file else self.descending_geometry
        else:
            geometry = None

        # Convert geocoordinates to radar
        coord = coordinate(atr, lookup_file=geometry)
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

    def _plot_timeseries(self, file):
        """Plots timeseries data on the last axis."""
        ax_ts = self.ax
        color = "#ff7366" if "SenA" in file else "#5190cb"
        label = "ascending" if "SenA" in file else "descending"

        dates, ts = self._extract_timeseries_data(file)
        ax_ts.scatter(dates, ts + self.offset, color=color, marker='o', label=label, alpha=0.5, s=7)

        self.offset = 0

        # Plot vertical lines
        ax_ts.axvline(self.start_date, color='#a8a8a8', linestyle='--', linewidth=0.2, alpha=0.5)
        ax_ts.axvline(self.end_date, color='#a8a8a8', linestyle='--', linewidth=0.2, alpha=0.5)

        # Fill area between the vertical lines with a rectangle
        ax_ts.axvspan(self.start_date, self.end_date, color='#a8a8a8', alpha=0.1)
        ax_ts.set_ylabel('LOS displacement (m)')
        ax_ts.legend(fontsize='x-small')

    def _plot_event(self):
        if self.add_event:
            for event, magnitude in zip(self.add_event, self.magnitude):
                event = datetime.strptime(event, "%Y%m%d") if type(event) == str else event
                self.ax.axvline(event, color='#900C3F', linestyle='--', linewidth=1, alpha=0.3)

                # Add a number near the top of the line
                if magnitude:
                    self.ax.text(event, self.ax.get_ylim()[1] * (magnitude/10), f"{magnitude}", color='#900C3F', fontsize=7, alpha=1, ha='center',  path_effects=[withStroke(linewidth=0.5, foreground='black')])

    def plot(self, ax):
        self.ax = ax
        for file in self.files:
            self._plot_timeseries(file)

        self._plot_event()

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