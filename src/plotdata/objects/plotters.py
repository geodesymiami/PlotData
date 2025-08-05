import math
import pygmt
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LightSource
from plotdata.objects.section import Section
from plotdata.objects.get_methods import DataFetcherFactory
from plotdata.objects.create_map import Mapper, Isolines, Relief
from plotdata.objects.earthquakes import Earthquake
from mintpy.utils import readfile
from mintpy.objects.coord import coordinate
from matplotlib.patheffects import withStroke
from mintpy.objects import timeseries, HDFEOS
from plotdata.helper_functions import draw_vectors, calculate_distance, get_bounding_box, expand_bbox, parse_polygon

def set_default_section(line, region):
    mid_lat = line if type(line) == float else (max(region[2:4]) + min(region[2:4]))/2
    mid_lon = (max(region[0:2]) + min(region[0:2]))/2

    size = (max(region[0:2]) - min(region[0:2]))*0.25

    latitude = (mid_lat, mid_lat)
    longitude = (mid_lon - size, mid_lon + size)

    return [longitude, latitude]

def plot_point(ax, lat, lon, marker='o', color='black', size=5, alpha=1, zorder=None):
    ax.plot(lon, lat, marker, color=color, markersize=size, alpha=alpha, zorder=zorder)

class VelocityPlot:
    """Handles the plotting of velocity maps."""
    def __init__(self, dataset, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))

        if 'data' in dataset:
            self.data = dataset["data"]
            self.attributes = dataset["attributes"]
        elif 'geometry' in dataset:
            self.attributes = dataset["geometry"]["attributes"]

        if "region" in self.attributes:
            self.region = self.attributes["region"]
        elif hasattr(self, 'region'):
            pass
        else:
            latitude, longitude = get_bounding_box(self.attributes)
            self.region = [longitude[0], longitude[1], latitude[0], latitude[1]]

        if not self.no_dem:
            for key in dataset:
                if "geometry" in key: 
                    self.geometry = dataset[key]['data']
                    self.attributes['longitude'] = dataset[key]['attributes']['longitude']
                    self.attributes['latitude'] = dataset[key]['attributes']['latitude']

                    break
                else:
                    self.geometry = None

        if "earthquakes" in dataset:
            self.earthquakes = dataset["earthquakes"]

        self.zorder = 0

    def _get_next_zorder(self):
        z = self.zorder
        self.zorder += 1
        return z

    def _plot_velocity(self):
        # TODO change to argparse
        zorder = self._get_next_zorder()
        cmap = 'jet'

        if not self.vmin and not self.vmax:
            lim = max(abs(np.nanmin(self.data)), abs(np.nanmax(self.data))) * 1.2
            self.vmin, self.vmax = -lim, lim

        if self.style == 'pixel':
            self.imdata = self.ax.imshow(self.data, cmap=cmap, extent=self.region, origin='upper', interpolation='none', zorder=zorder, vmin=self.vmin, vmax=self.vmax, rasterized=True)
            self.ax.set_aspect('auto')

        elif self.style == 'scatter':
            # Assuming self.velocity is a 2D numpy array
            nrows, ncols = self.data.shape

            x = np.linspace(self.region[0], self.region[1], ncols)
            y = np.linspace(self.region[2], self.region[3], nrows)
            X, Y = np.meshgrid(x, y)
            X = X.flatten()
            Y = np.flip(Y.flatten())
            C = self.data.flatten()

            self.imdata = self.ax.scatter(X, Y, c=C, cmap=cmap, marker='o', zorder=zorder, s=2, vmin=self.vmin, vmax=self.vmax, rasterized=True)

        cbar = self.ax.figure.colorbar(self.imdata, ax=self.ax, orientation='horizontal', aspect=13)
        cbar.set_label(self.unit)

        cbar.locator = ticker.MaxNLocator(3)
        cbar.update_ticks()

    def _plot_dem(self):
        print("-"*50)
        print("Plotting DEM data...\n")

        zorder = self._get_next_zorder()

        if not isinstance(self.geometry, np.ndarray):
            self.z = self.geometry.astype(float)
        else:
            self.z = self.geometry

        lat = self.attributes['latitude']
        lon = self.attributes['longitude']

        dlon = lon[1] - lon[0]
        dlat = lat[1] - lat[0]

        # Compute hillshade with real spacing
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(self.z, vert_exag=1.5, dx=dlon, dy=dlat)

        # # Create meshgrid of lon/lat edges for pcolormesh
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Use pcolormesh to plot hillshade using real coordinates
        self.im = self.ax.pcolormesh(lon2d,lat2d,hillshade,cmap='gray',shading='auto',zorder=zorder,alpha=0.5,)

    def _plot_isolines(self):
        print("Adding isolines...\n")

        zorder = self._get_next_zorder()

        if self.geometry is None:
            lines = pygmt.datasets.load_earth_relief(resolution=self.resolution, region=self.region)

            grid_np = lines.values

            # Remove negative values
            grid_np[grid_np < 0] = 0

            # Convert the numpy array back to a DataArray
            lines[:] = grid_np
        else:
            grid_np = self.geometry

            lat = self.attributes['latitude']
            lon = self.attributes['longitude']

            # Remove negative values
            grid_np[grid_np < 0] = 0

            # Convert the numpy array back to a DataArray
            lines = xr.DataArray(grid_np,dims=["lat", "lon"],coords={"lat": lat, "lon": lon},)

        # Extract coordinates and elevation values
        lon = lines.coords["lon"].values
        lat = lines.coords["lat"].values
        z = lines.values

        #Plot the isolines
        cont = self.ax.contour(lon, lat, z, levels=self.isolines, colors=self.color, linewidths=self.linewidth, alpha=0.7, zorder=zorder)

        if self.inline:
            self.ax.clabel(cont, inline=self.inline, fontsize=8)

    def _plot_earthquakes(self):
        """Plots earthquake data on the map."""
        print("Plotting earthquake data...\n")

        zorder = self._get_next_zorder()

        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(self.earthquakes['magnitude']), vmax=max(self.earthquakes['magnitude']))

        for lalo, magnitude, date in zip(self.earthquakes['lalo'], self.earthquakes['magnitude'], self.earthquakes['date']):
            self.ax.scatter(
                lalo[1], lalo[0], 
                s=10**(magnitude*0.5),  # Size based on magnitude
                c=cmap(norm(magnitude)),  # Color based on magnitude
                edgecolors='black',  # Edge color
                linewidths=0.5,  # Edge width
                marker='o',  # Circle marker
                alpha=0.6,  # Transparency
                label=f"{magnitude} {date}",
                zorder=zorder
            )

    def plot(self, ax):
        """Creates and configures the velocity map."""
        self.ax = ax

        if 'ascending' in self.ax.get_label():
            self.label = "ASCENDING"
        elif 'descending' in self.ax.get_label():
            self.label = "DESCENDING"
        elif 'horizontal' in self.ax.get_label():
            self.label = "HORIZONTAL"
        elif 'vertical' in self.ax.get_label():
            self.label = "VERTICAL"
        else:
            self.label = None

        if self.geometry is not None:
            self._plot_dem()

        if hasattr(self, 'data') and self.data is not None:
            self._plot_velocity()

        if self.isolines:
            self._plot_isolines()

        if hasattr(self, 'earthquakes') and self.earthquakes['date']:
            self._plot_earthquakes()

        if 'point' in self.ax.get_label():
            if not self.lalo:
                self.lalo = [(max(self.region[2:4]) + min(self.region[2:4]))/2, (max(self.region[0:2]) + min(self.region[0:2]))/2]
            plot_point(self.ax, [self.lalo[0]], [self.lalo[1]], marker='x', zorder=self._get_next_zorder())

        if 'section' in self.ax.get_label():
            if not self.line or type(self.line) == float:
                self.line = set_default_section(self.line, self.region)
            self.ax.plot(self.line[0], self.line[1], '--', linewidth=1.5, alpha=0.7, color='black', zorder=self._get_next_zorder())

        if self.ref_lalo and 'seismicmap' not in self.ax.get_label():
            plot_point(self.ax, [self.ref_lalo[0]], [self.ref_lalo[1]], marker='s', zorder=self._get_next_zorder())

        if self.label:
            self.ax.annotate(self.label,xy=(0.02, 0.98),xycoords='axes fraction',fontsize=5,ha='left',va='top',color='white',bbox=dict(facecolor='gray', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3'))

####################################################################################

class EarthquakePlot:
    def __init__(self, dataset, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))
        self.earthquakes = dataset

    def plot(self, ax):
        if not self.earthquakes['date']:
            ax.set_title('No Earthquake Data Available')
            ax.set_xlabel('Date')
            ax.set_ylabel('Magnitude')
            ax.set_xlim([datetime.strptime(self.start_date, '%Y%m%d').date() if isinstance(self.start_date, str) else self.start_date.date(),
                        datetime.strptime(self.end_date, '%Y%m%d').date() if isinstance(self.end_date, str) else self.end_date.date()])
            ax.set_ylim([0, 10])
            return

        if 'date' in ax.get_label():
            self.plot_by_date(ax)
        elif 'distance' in ax.get_label():
            self.plot_by_distance(ax)
        else:
            self.plot_by_date(ax)

    def plot_by_date(self, ax):
        # Plot EQs
        for i in range(len(self.earthquakes['date'])):
            ax.plot([self.earthquakes['date'][i], self.earthquakes['date'][i]], [self.earthquakes['magnitude'][i], 0], 'k-')

        ax.scatter(self.earthquakes['date'], self.earthquakes['magnitude'], c='black', marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Earthquake Magnitudes Over Time')
        s = datetime.strptime(self.start_date, '%Y%m%d') if type(self.start_date) == str else self.start_date
        e = datetime.strptime(self.end_date, '%Y%m%d') if type(self.end_date) == str else self.end_date
        ax.set_xlim([s.date(), e.date()])
        ax.set_ylim([0, 10])

    def plot_by_distance(self, ax):
        # Plot EQs
        dist = []
        for i in range(len(self.earthquakes['date'])):
            if not hasattr(self, 'coordinates') or not self.coordinates:
                self.coordinates = self.lalo if hasattr(self, 'lalo') else [(self.region[0]+self.region[1])/2, (self.region[2]+self.region[3])/2]
            dist.append(calculate_distance(self.earthquakes['lalo'][i][0], self.earthquakes['lalo'][i][1], self.coordinates[0], self.coordinates[1]))
            ax.plot([dist[i], dist[i]], [self.earthquakes['magnitude'][i], 0], 'k-')

        if not dist:
            dist = [0, 10]
            dist1 = (self.region[0]-self.region[1])/2 * 111.32 * math.cos(math.radians(float(self.region[-1])))
            dist2 = (self.region[2]-self.region[3])/2 * 111.32
            dist = [0, (dist1**2 + dist2**2)**0.5]
            # dist = [0, calculate_distance(abs(max(self.region[0]-self.region[1], self.region[2]-self.region[3]))/2)]
            self.earthquakes['magnitude'] = [None, None]

        ax.set_xlim([0, max(dist)+ (max(dist) * 0.05)])
        ax.set_ylim([0, 10])

        ax.scatter(dist, self.earthquakes['magnitude'], c='black', marker='o')

        ax.set_xlabel('Distance in KM')
        ax.set_ylabel('Magnitude')
        ax.set_title('Earthquake Magnitudes from Volcano')

####################################################################################

class TimeseriesPlot:
    def __init__(self, dataset, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))

        self.start_date = self.start_date[0] if isinstance(self.start_date, list) else self.start_date
        self.end_date = self.end_date[0] if isinstance(self.end_date, list) else self.end_date

        self.dataset = dataset

    def _plot_timeseries(self, dataset):
        """Plots timeseries data on the last axis."""
        ax_ts = self.ax
        if "passDirection" in dataset["attributes"]:
            color = "#ff7366" if dataset["attributes"]["passDirection"] == "ASCENDING" else "#5190cb"
            label = "ascending" if dataset["attributes"]["passDirection"] == "ASCENDING" else "descending"
        elif "FILE_PATH" in dataset["attributes"]:
            color = "#ff7366" if "SenA" in dataset["attributes"]["FILE_PATH"] else "#5190cb"
            label = "ascending" if "SenA" in dataset["attributes"]["FILE_PATH"] else "descending"

        dates, ts= dataset['dates'], dataset['data']

        ax_ts.scatter(dates, ts + self.offset, color=color, marker='o', label=label, alpha=0.5, s=7)

        self.offset = 0

        # Plot vertical lines
        ax_ts.axvline(self.start_date, color='#a8a8a8', linestyle='--', linewidth=0.2, alpha=0.5)
        ax_ts.axvline(self.end_date, color='#a8a8a8', linestyle='--', linewidth=0.2, alpha=0.5)

        # Fill area between the vertical lines with a rectangle
        ax_ts.axvspan(self.start_date, self.end_date, color='#a8a8a8', alpha=0.1)
        ax_ts.set_ylabel(f'LOS displacement ({self.unit.replace("/yr", "")})')
        ax_ts.legend(fontsize='x-small')

    def _plot_event(self):
        if self.add_event:
            for event, magnitude in zip(self.add_event, self.event_magnitude):
                event = datetime.strptime(event, "%Y%m%d") if type(event) == str else event
                self.ax.axvline(event, color='#900C3F', linestyle='--', linewidth=1, alpha=0.3)

                # Add a number near the top of the line
                if magnitude:
                    self.ax.text(event, self.ax.get_ylim()[1] * (magnitude/10), f"{magnitude}", color='#900C3F', fontsize=7, alpha=1, ha='center',  path_effects=[withStroke(linewidth=0.5, foreground='black')])

    def plot(self, ax):
        self.ax = ax
        for key in self.dataset:
            self._plot_timeseries(self.dataset[key])

        self._plot_event()


class VectorsPlot:
    """Handles the plotting of velocity maps, elevation profiles, and vector fields."""
    def __init__(self, dataset, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))

        # TODO have to add attributes to https://github.com/insarlab/MintPy/blob/main/src/mintpy/asc_desc2horz_vert.py#L261
        # for f in files:
        #     attr = readfile.read_attribute(f)
        #     if attr['FILE_TYPE'] == 'VERTICAL':
        #         self.vert_file = f
        #         self.horz_file = None
        self.horz = dataset["horizontal"]["data"]
        self.vert = dataset["vertical"]["data"]
        self.geometry = dataset["horizontal"].get("geometry").get("data") if "geometry" in dataset["horizontal"] else dataset["vertical"].get("geometry").get("data")

        self.horz_attr = dataset["horizontal"]["attributes"]
        self.vert_attr = dataset["vertical"]["attributes"]
        self.geometry_attr = dataset["horizontal"].get("geometry", {}).get("attributes", {})

        # Double check if the line is set
        if not self.line or type(self.line) == float:
            self.line = set_default_section(self.line, self.region)

        if not self.region:
            if "scene_footprint" in self.geometry_attr:
                self.region = parse_polygon(self.geometry_attr["scene_footprint"])
            elif "region" in self.geometry_attr:
                self.region = self.geometry_attr["region"]
            else:
                latitude, longitude = get_bounding_box(self.geometry_attr)
                self.region = [longitude[0], longitude[1], latitude[0], latitude[1]]

        self.horizontal_section = self._process_sections(np.flipud(self.horz), self.region)
        self.vertical_section = self._process_sections(np.flipud(self.vert), self.region)
        self.topography_section = self._process_sections(self.geometry, self.geometry_attr["region"])

    def _process_sections(self, data, region):
        """Processes the sections for horizontal and vertical components."""

        lat_indices, lon_indices = self._draw_line(data, region, self.line[1], self.line[0])

        # Extract the values data along the snapped path
        values = data[lat_indices, lon_indices]

        # TODO recheck
        return np.nan_to_num(values)

    def _draw_line(self, data, region, latitude, longitude):
        # Calculate the resolution in degrees
        lat_res = (region[3] - region[2]) / (data.shape[0] - 1)
        lon_res = (region[1] - region[0]) / (data.shape[1] - 1)

        # Calculate the distance between start and end points
        distance = np.sqrt((latitude[1] - latitude[0])**2 + (longitude[1] - longitude[0])**2)

        # Determine the number of points based on the distance
        num_points = int(distance / min(lat_res, lon_res))

        lon_points = np.linspace(longitude[0], longitude[1], num_points)
        lat_points = np.linspace(latitude[0], latitude[1], num_points)

        # Snap points to the nearest grid points
        lon_indices = np.round((lon_points - region[0]) / lon_res).astype(int)
        lat_indices = np.round((lat_points - region[2]) / lat_res).astype(int)

        # Ensure indices are within bounds
        lon_indices = np.clip(lon_indices, 0, data.shape[1] - 1)
        lat_indices = np.clip(lat_indices, 0, data.shape[0] - 1)

        # TODO do i need it?
        # Create a DataFrame to store the path data
        self.path_df = pd.DataFrame({
            'longitude': lon_points,
            'latitude': lat_points,
            'lon_index': lon_indices,
            'lat_index': lat_indices,
            'distance': np.linspace(0, 1, num_points)  # Normalized distance
        })

        return lat_indices, lon_indices

    def _compute_vectors(self):
        """Computes velocity vectors and scaling factors."""
        x, v, h, self.z = draw_vectors(self.topography_section, self.vertical_section, self.horizontal_section, self.line)

        fig = self.ax.get_figure()
        fig_width, fig_height = fig.get_size_inches()
        max_elevation = max(self.z)
        max_x = max(x)

        self.v_adj = 2 * max_elevation / max_x
        self.h_adj = 1 / self.v_adj

        self.rescale_h = self.h_adj / fig_width
        self.rescale_v = self.v_adj / fig_height

        # Resample vectors
        for i in range(len(h)):
            if i % self.resample_vector != 0:
                h[i] = 0
                v[i] = 0

        distance = calculate_distance(self.line[1][0], self.line[0][0], self.line[1][1], self.line[0][1])
        self.xrange = np.linspace(0, distance, len(x))
        # Filter out zero-length vectors
        non_zero_indices = np.where((h != 0) | (v != 0))
        self.filtered_x = self.xrange[non_zero_indices]
        self.filtered_h = h[non_zero_indices]
        self.filtered_v = v[non_zero_indices]
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
        mean_velocity = np.sqrt(np.mean((self.vertical_section[self.vertical_section!=0]))**2 + np.mean((self.horizontal_section[self.horizontal_section!=0]))**2)
        rounded_mean_velocity = round(mean_velocity, 4) if mean_velocity else round(mean_velocity, 3)

        self.ax.quiver([start_x], [start_y], [mean_velocity], [0], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        # self.ax[2].quiver([start_x], [start_y], [0], [abs(np.mean(self.filtered_v))], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        self.ax.text(start_x, start_y * 1.03, f"{rounded_mean_velocity:.4f} {self.unit}", color='black', ha='left', fontsize=8)

        # Add labels
        self.ax.set_ylabel("Elevation (m)")
        self.ax.set_xlabel("Distance (km)")


def point_on_globe(latitude, longitude, names=None, size='0.7', fsize=10):
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
            font=f"{fsize}p,Helvetica-Bold,black",  # Font size, style, and color
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