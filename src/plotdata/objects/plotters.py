import pygmt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plotdata.objects.section import Section
from plotdata.objects.create_map import Mapper, Isolines, Relief
from plotdata.objects.earthquakes import Earthquake
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
        self.earthquake = inps.earthquake
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
        if self.earthquake:
            Earthquake(map=rel_map).map()


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
        self.earthquake = inps.earthquake
        self.no_dem = inps.no_dem

        self.map = self._create_map()

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
        if self.earthquake:
            Earthquake(map=vel_map).map()

        return vel_map


class VectorsPlot:
    """Handles the plotting of velocity maps, elevation profiles, and vector fields."""

    def __init__(self, ax, asc_file, desc_file, horz_file, vert_file, inps):
        self.ax = ax  # Expecting an array of axes
        self.asc_file = asc_file
        self.desc_file = desc_file
        self.horz_file = horz_file
        self.vert_file = vert_file
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
        if self.inps.plot_option and self.inps.plot_option != 'horzvert':
            self.plot1_file = self.asc_file
            self.plot2_file = self.desc_file
        else:
            self.plot1_file = self.horz_file
            self.plot2_file = self.vert_file

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
        if self.inps.earthquake:
            Earthquake(map=vel_map).map()

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
        self.ax[2].plot(self.x, self.elevation_section.values)
        self.ax[2].set_ylim([0, 2 * max(self.elevation_section.values)])
        self.ax[2].set_xlim([min(self.x), max(self.x)])

        # Plot velocity vectors
        self.ax[2].quiver(
            self.filtered_x, self.filtered_elevation,
            self.filtered_h * self.rescale_h, self.filtered_v * self.rescale_v,
            color='red', scale_units='xy', width=(1 / 10**(2.5))
        )

        # Add profile lines to velocity maps
        for i in range(2):
            self.ax[i].plot(self.inps.line[0], self.inps.line[1], '-', linewidth=2, alpha=0.7, color='black')

        # Mean velocity vector
        start_x = max(self.x) * 0.1
        start_y = (2 * max(self.elevation_section.values) * 0.8)
        mean_velocity = abs(np.mean(self.filtered_h * self.rescale_h))

        self.ax[2].quiver([start_x], [start_y], [mean_velocity], [0], color='red', scale_units='xy', width=(1 / 10**(2.5)))
        self.ax[2].text(start_x, start_y * 1.03, f"{round(mean_velocity, 3)} m/yr", color='black', ha='left', fontsize=8)


def point_on_globe(latitude, longitude, size='1'):
    fig = pygmt.Figure()

    # Set up orthographic projection centered on your point
    fig.basemap(
        region="d",  # Global domain
        projection=f"G{np.mean(longitude)}/{np.mean(latitude)}/15c",  # Centered on your coordinates
        frame="g"  # Show gridlines only
    )

    # Add continent borders with black lines
    fig.coast(
        shorelines="1/1p,black",  # Continent borders
        land="white",  # Land color
        water="white"  # Water color
    )

    # Plot your central point
    fig.plot(
        x=longitude,
        y=latitude,
        style=f"t{size}c",  # Triangle marker
        fill="red",  # Marker color
        pen="1p,black"  # Outline pen
    )