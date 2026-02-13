import math
import pygmt
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter, medfilt
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
from matplotlib.transforms import Affine2D
from matplotlib.patheffects import withStroke
from plotdata.volcano_functions import get_volcanoes_data
from plotdata.helper_functions import draw_vectors, calculate_distance, get_bounding_box, parse_polygon, resize_to_match


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
        elif 'synth' in dataset:
            self.synth = dataset["synth"]
            self.attributes = dataset["attributes"]
            self.east = dataset["east"]
            self.north = dataset["north"]
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

    def _update_axis_limits(self, x_min=None, x_max=None, y_min=None, y_max=None):
        if hasattr(self, 'subset') and self.subset:
            try:
                # Split the string into two parts
                coords1, coords2 = self.subset.split(':')

                # Split each part into lat and lon
                lat1, lon1 = map(float, coords1.split(','))
                lat2, lon2 = map(float, coords2.split(','))

                # Assign to x_min, x_max, y_min, y_max
                x_min, x_max = sorted([lon1, lon2])  # Longitude corresponds to x-axis
                y_min, y_max = sorted([lat1, lat2])  # Latitude corresponds to y-axis

            except ValueError:
                raise ValueError(f"Invalid subset format: {self.subset}. Expected format is 'lat,lon:lat2,lon2'.")

            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)

        elif self.zoom:
            # Get current axis limits
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()

            # Calculate the range
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Calculate the new limits
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            new_x_range = x_range / self.zoom
            new_y_range = y_range / self.zoom

            new_x_min = x_center - new_x_range / 2
            new_x_max = x_center + new_x_range / 2
            new_y_min = y_center - new_y_range / 2
            new_y_max = y_center + new_y_range / 2

            # Set the new axis limits
            self.ax.set_xlim(new_x_min, new_x_max)
            self.ax.set_ylim(new_y_min, new_y_max)

    def _get_next_zorder(self):
        z = self.zorder
        self.zorder += 1
        return z

    def _plot_source(self, sources):
        if sources:
            source_type = {
                "mogi": {"class": Mogi, "attributes": ["xcen", "ycen"]},
                "spheroid": {"class": Spheroid, "attributes": ["xcen", "ycen", "s_axis_max", "ratio", "strike", "dip"]},
                "penny": {"class": Penny,  "attributes": ["xcen", "ycen", "radius"]},
                "okada": {"class": Okada,  "attributes": ["ytlc", "xtlc", "length", "width", "strike", "dip"]},
            }
            for s in sources:
                s_keys = set(sources[s].keys())

                for key, value in source_type.items():
                    if set(value["attributes"]) == s_keys:
                        model = value["class"]
                        model(self.ax, **sources[s])

    def _plot_synthetic(self, data):
        zorder = self._get_next_zorder()

        if not self.vmin and not self.vmax:
            lim = max(abs(np.nanmin(data)), abs(np.nanmax(data))) * 1.2
            self.vmin, self.vmax = -lim, lim

        if self.style == 'scatter':
            self.imdata = self.ax.scatter(self.east, self.north, c=data, cmap=self.cmap, marker='o', zorder=zorder, s=20, vmin=self.vmin, vmax=self.vmax, rasterized=True)
        else:
            self.imdata = self.ax.imshow(data, cmap=self.cmap, extent=self.region, origin='upper', interpolation='none', zorder=zorder, vmin=self.vmin, vmax=self.vmax, rasterized=True)

        self._plot_source(self.sources) 

        self._update_axis_limits()

        self._plot_scale()

        if not self.no_colorbar:
            cbar = self.ax.figure.colorbar(self.imdata, ax=self.ax, orientation='horizontal', aspect=12, shrink=self.colorbar_size)
            cbar.set_label(self.unit)

            cbar.locator = ticker.MaxNLocator(3)
            cbar.update_ticks()

        self.imdata.set_alpha(0.7)

    def _plot_velocity(self, data):
        # TODO change to argparse
        zorder = self._get_next_zorder()

        if not self.vmin and not self.vmax:
            lim = max(abs(np.nanmin(data)), abs(np.nanmax(data))) * 1.2
            self.vmin, self.vmax = -lim, lim

        if self.style == 'pixel':
            self.imdata = self.ax.imshow(data, cmap=self.cmap, extent=self.region, origin='upper', interpolation='none', zorder=zorder, vmin=self.vmin, vmax=self.vmax, rasterized=True)

        elif self.style == 'scatter':
            # Assuming self.velocity is a 2D numpy array
            nrows, ncols = data.shape

            x = np.linspace(self.region[0], self.region[1], ncols)
            y = np.linspace(self.region[2], self.region[3], nrows)
            X, Y = np.meshgrid(x, y)
            X = X.flatten()
            Y = np.flip(Y.flatten())
            C = data.flatten()

            self.imdata = self.ax.scatter(X, Y, c=C, cmap=self.cmap, marker='o', zorder=zorder, s=10, vmin=self.vmin, vmax=self.vmax, rasterized=True)

        elif self.style == 'ifgram':
            # Wavelength for the interferogram
            wavelength = 0.05546576

            # Calculate the interferogram
            interferogram = (data) % (12 * np.pi)

            # Plot the interferogram
            self.imdata = self.ax.imshow(interferogram, cmap=self.cmap, extent=self.region, origin='upper', interpolation='none', zorder=zorder, rasterized=True)
            # TODO is this important?
            # self.ax.set_aspect('auto')

            #########################################################################################################################################

        self._update_axis_limits()

        self._plot_scale()

            #########################################################################################################################################

        if not self.no_colorbar:
            cbar = self.ax.figure.colorbar(self.imdata, ax=self.ax, orientation='horizontal', aspect=12, shrink=self.colorbar_size)
            cbar.set_label(self.unit)

            cbar.locator = ticker.MaxNLocator(3)
            cbar.update_ticks()

        self.imdata.set_alpha(0.7)


    def _plot_scale(self):
        zorder = self._get_next_zorder()

        lon1, lon2 = self.ax.get_xlim()
        lat1, lat2 = self.ax.get_ylim()
        lon_span = lon2 - lon1
        lat_span = lat2 - lat1

        dlon = lon_span / 4.0
        mean_lat = (lat1 + lat2) / 2.0
        km_per_deg = 111.32 * math.cos(math.radians(mean_lat))
        dist_km = abs(dlon) * km_per_deg

        # choose a location with a small margin (works if axes possibly inverted)
        x0 = min(lon1, lon2) + 0.05 * abs(lon_span)
        y0 = min(lat1, lat2) + 0.05 * abs(lat_span)

        # draw bar and end ticks
        self.ax.plot([x0, x0 + dlon], [y0, y0], color='k', lw=1)
        tick_h = 0.005 * abs(lat_span)
        self.ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color='k', lw=2)
        self.ax.plot([x0 + dlon, x0 + dlon], [y0 - tick_h, y0 + tick_h], color='k', lw=2)
        # label centered under the bar
        self.ax.text(x0 + dlon/2, y0 + 0.06 * abs(lat_span), f"{dist_km:.0f} km", ha='center', va='top', fontsize=8, path_effects=[withStroke(linewidth=1.5, foreground='white')], zorder=zorder)



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

        if lon.ndim > 1:
            dlon = float(self.attributes['X_STEP'])
            dlat = float(self.attributes['Y_STEP'])

            if hasattr(self, 'lat1d') and hasattr(self, 'lon1d'):
                lon1d = self.lon1d
                lat1d = self.lat1d
            else:
                lon_min = min(self.region[0:2])
                lat_max = max(self.region[2:4])
                ny, nx = self.z.shape

                lon1d = lon_min + np.arange(nx) * dlon
                lat1d = lat_max + np.arange(ny) * dlat

            lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        else:
            dlon = lon[1] - lon[0]
            dlat = lat[1] - lat[0]
            lon2d, lat2d = np.meshgrid(lon, lat)
 
        if hasattr(self, 'data') and self.data is not None:
            self.z = resize_to_match(self.z, self.data, 'DEM')
            lat2d = resize_to_match(lat2d, self.data, 'latitude')
            lon2d = resize_to_match(lon2d, self.data, 'longitude')


        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(np.nanmean(lat2d)))

        dx = dlon * meters_per_deg_lon
        dy = dlat * meters_per_deg_lat

        # Compute hillshade with real spacing
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(self.z, vert_exag=0.7, dx=dx, dy=dy)

        # Use pcolormesh to plot hillshade using real coordinates
        self.im = self.ax.pcolormesh(lon2d,lat2d,hillshade,cmap='gray',shading='auto',zorder=zorder,)

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
        cont = self.ax.contour(lon, lat, z, levels=self.contour, colors=self.color, linewidths=self.contour_linewidth, alpha=0.3, zorder=zorder)

        if self.inline:
            self.ax.clabel(cont, inline=self.inline, fontsize=8)

    def _plot_earthquakes(self):
        """Plots earthquake data on the map."""
        print("Plotting earthquake data...\n")

        zorder = self._get_next_zorder()

        vmin = min(self.earthquakes['magnitude'])  # Minimum magnitude
        vmax = max(self.earthquakes['magnitude'])  # Maximum magnitude

        for lalo, magnitude, date in zip(self.earthquakes['lalo'], self.earthquakes['magnitude'], self.earthquakes['date']):
            imdata = self.ax.scatter(
                lalo[1], lalo[0],
                s=10**(magnitude*0.5),  # Size based on magnitude
                c=magnitude, #cmap(norm(magnitude)),  # Color based on magnitude
                edgecolors='black',  # Edge color
                vmin=vmin,  # Set minimum value for colormap
                vmax=vmax,  # Set maximum value for colormap
                linewidths=0.5,  # Edge width
                marker='o',  # Circle marker
                alpha=0.6,  # Transparency
                label=f"{magnitude} {date}",
                zorder=zorder,
            )

        if not self.no_colorbar and (not hasattr(self, 'data') or self.data is None):
            cbar = self.ax.figure.colorbar(imdata, ax=self.ax, orientation='horizontal', fraction=0.03, pad=0.05)
            cbar.set_label('Magnitude')

            cbar.set_ticks([cbar.vmin, (cbar.vmin + cbar.vmax) / 2, cbar.vmax])
            cbar.formatter = ticker.FormatStrFormatter('%.1f')
            cbar.update_ticks()

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
            self._plot_velocity(self.data)

        if hasattr(self, 'synth') and self.synth is not None:
            self._plot_synthetic(self.synth)

        if self.contour:
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
            self.ax.annotate(self.label,xy=(0.02, 0.98),xycoords='axes fraction',fontsize=7,ha='left',va='top',color='white',bbox=dict(facecolor='black', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3'))

        if self.volcano:
            min_lon, max_lon, min_lat, max_lat = self.region
            volcanoName, volcanoId, volcanoCoordinates = get_volcanoes_data(bbox=[min_lon, min_lat, max_lon, max_lat])
            for name, id, coord in zip(volcanoName, volcanoId, volcanoCoordinates):
                lon, lat = coord
                print(f'Plotting volcano: {name}, id: {id}, coordinates: {lat}, {lon}')
                plot_point(self.ax, [lat], [lon], marker='^', color='#383838db', size=7, alpha=0.3, zorder=self._get_next_zorder())
                self.ax.text(lon, lat, name, fontsize=6, color='black', zorder=self._get_next_zorder())

        self.ax.set_aspect('equal', adjustable='datalim')

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
        ax.set_xticks([s.date(), s.date() + (e.date() - s.date()) / 2, e.date()])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


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

####################################################################################

class ProfilePlot:
    """Handles the plotting of deformation profiles for both model and observed data."""
    def __init__(self, dataset, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))

        self.data = dataset["data"]
        self.synth = resize_to_match(dataset["synth"], dataset["data"], 'Profile Data')
        self.geometry = dataset.get("geometry").get("data") if "geometry" in dataset else None
        self.attributes = dataset.get("geometry").get("attributes") if "geometry" in dataset else None
        self.region = self.attributes.get('region')

        resize_to_match(dataset["synth"], dataset["data"], 'Profile Data')

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

    def _draw_line(self, data, region, latitude, longitude):
        """Draws a line on the data grid and returns the indices of the path."""
        ny, nx = data.shape

        lon_min, lon_max = float(region[0]), float(region[1])
        lat_min, lat_max = float(region[2]), float(region[3])

        # avoid zero-division for degenerate regions
        lon_span = lon_max - lon_min if (lon_max - lon_min) != 0 else 1.0
        lat_span = lat_max - lat_min if (lat_max - lat_min) != 0 else 1.0

        # number of sample points along the profile
        distance_deg = math.hypot(latitude[1] - latitude[0], longitude[1] - longitude[0])
        num_points = max(2, int(distance_deg / min(lat_span / max(1, ny - 1), lon_span / max(1, nx - 1))) + 1)

        lon_points = np.linspace(longitude[0], longitude[1], num_points)
        lat_points = np.linspace(latitude[0], latitude[1], num_points)

        # fractional column index: 0..(nx-1) left->right
        col_f = (lon_points - lon_min) / lon_span * (nx - 1)

        # fractional row index: if row 0 == top (lat_max), map lat -> row via lat_max - lat
        row_f = (lat_max - lat_points) / lat_span * (ny - 1)

        # round/clip to integer array indices
        lon_indices = np.clip(np.round(col_f).astype(int), 0, nx - 1)
        lat_indices = np.clip(np.round(row_f).astype(int), 0, ny - 1)

        return lat_indices, lon_indices


    def _process_sections(self, data, region):
        """Processes the sections for horizontal and vertical components."""
        lat_indices, lon_indices = self._draw_line(data, region, self.line[1], self.line[0])

        # Extract the values data along the snapped path
        values = data[lat_indices, lon_indices]

        return values

    def _plot_profile(self, dataset, key, ax):
        style = {
            'model': dict(c='#5190cb', lw=2.5, ls='-', label='Model'),
            'data': dict(c='#ff7366', marker='o', ls='--', ms=3, label='Data', alpha=0.6),
            'smooth': dict(c='#79419e', lw=1, ls='-', label='Smoothed Data'),
        }.get(key, {})
        self.ax.plot(dataset[key], **style)

    def plot(self, ax):
        """Creates and configures the profile plot."""
        self.ax = ax

        profile_synth = self._process_sections(self.synth, self.region)
        profile_data = self._process_sections(self.data, self.region)
        # profile_topo = self._process_sections(self.geometry, self.region) if self.geometry is not None else None

        self.ax.set_ylabel(f'{self.unit}')

        if self.norm:
            profile_synth = (profile_synth - np.nanmin(profile_synth)) / (np.nanmax(profile_synth) - np.nanmin(profile_synth))
            profile_data = (profile_data - np.nanmin(profile_data)) / (np.nanmax(profile_data) - np.nanmin(profile_data))
            ax.set_ylabel('Normalized')

        self.profiles = {
            "model": profile_synth,
            "data": profile_data,
        }

        if self.denoise:
            window = self.denoise if self.denoise % 2 == 1 else self.denoise + 1
            # profile_data = savgol_filter(profile_data, window_length=window, polyorder=2,)
            self.profiles["smooth"] = np.convolve(profile_data, np.ones(window)/window, mode='valid')

        for key in self.profiles:
            self._plot_profile(self.profiles, key, self.ax)

        self.ax.legend(fontsize='xx-small')

        if 'ascending' in self.ax.get_label():
            self.label = "ASCENDING"
        elif 'descending' in self.ax.get_label():
            self.label = "DESCENDING"
        elif self.attributes.get('ORBIT_DIRECTION'):
            self.label = self.attributes['ORBIT_DIRECTION'].upper()

        if self.label:
            self.ax.annotate(self.label,xy=(0.02, 0.98),xycoords='axes fraction',fontsize=7,ha='left',va='top',color='white',bbox=dict(facecolor='black', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3'))

####################################################################################

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
        self.direction = dataset.get("geometry").get("attributes").get('ORBIT_DIRECTION').lower() if "geometry" in dataset else None


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

        self.horizontal_section = self._process_sections((self.horz), self.region)
        self.vertical_section = self._process_sections((self.vert), self.region)
        self.topography_section = self._process_sections(self.geometry, self.geometry_attr["region"])

    def _process_sections(self, data, region):
        """Processes the sections for horizontal and vertical components."""
        lat_indices, lon_indices = self._draw_line(data, region, self.line[1], self.line[0])

        # Extract the values data along the snapped path
        values = data[lat_indices, lon_indices]

        # TODO recheck
        return values

    def _draw_line(self, data, region, latitude, longitude):
        if False:
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
        else:
            ny, nx = data.shape

            lon_min, lon_max = float(region[0]), float(region[1])
            lat_min, lat_max = float(region[2]), float(region[3])

            # avoid zero-division for degenerate regions
            lon_span = lon_max - lon_min if (lon_max - lon_min) != 0 else 1.0
            lat_span = lat_max - lat_min if (lat_max - lat_min) != 0 else 1.0

            # number of sample points along the profile
            distance_deg = math.hypot(latitude[1] - latitude[0], longitude[1] - longitude[0])
            num_points = max(2, int(distance_deg / min(lat_span / max(1, ny - 1), lon_span / max(1, nx - 1))) + 1)

            lon_points = np.linspace(longitude[0], longitude[1], num_points)
            lat_points = np.linspace(latitude[0], latitude[1], num_points)

            # fractional column index: 0..(nx-1) left->right
            col_f = (lon_points - lon_min) / lon_span * (nx - 1)

            # fractional row index: if row 0 == top (lat_max), map lat -> row via lat_max - lat
            row_f = (lat_max - lat_points) / lat_span * (ny - 1)

            # round/clip to integer array indices
            lon_indices = np.clip(np.round(col_f).astype(int), 0, nx - 1)
            lat_indices = np.clip(np.round(row_f).astype(int), 0, ny - 1)

            return lat_indices, lon_indices

    def _compute_vectors(self):
        """Computes velocity vectors and scaling factors."""
        x, v, h, self.z = draw_vectors(self.topography_section, self.vertical_section, self.horizontal_section, self.line)
        fig = self.ax.get_figure()
        fig_width, fig_height = fig.get_size_inches()
        max_elevation = np.nanmax(self.z)
        max_x = np.nanmax(x)

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
        if self.vector_legend == 'mean_vector':
            # Mean velocity vector
            self.imdata = self.ax.quiver(self.filtered_x, self.filtered_elevation, self.filtered_h, self.filtered_v, color='#ff7366', width=(1 / 10**(2.5)))
            start_x = max(self.xrange) * 0.1
            start_y = (2 * max(self.z) * 0.8)
            mean_velocity = np.sqrt(np.nanmean((self.vertical_section[self.vertical_section!=0]))**2 + np.nanmean((self.horizontal_section[self.horizontal_section!=0]))**2)
            rounded_mean_velocity = round(mean_velocity, 4) if mean_velocity else round(mean_velocity, 3)

            self.ax.quiver([start_x], [start_y], [mean_velocity], [0], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
            self.ax.text(start_x, start_y * 1.03, f"{rounded_mean_velocity:.2f} {self.unit}", color='black', ha='left', fontsize=8)

        elif self.vector_legend == 'colorbar':
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            self.imdata = self.ax.quiver(self.filtered_x, self.filtered_elevation, self.filtered_h, self.filtered_v, np.hypot(self.filtered_h, self.filtered_v), cmap='viridis', width=(1 / 10**(2.5)))
            cax = inset_axes(self.ax, width="15%", height="2.8%", loc="lower left", borderpad=2.0)

            cb = self.ax.figure.colorbar(self.imdata, cax=cax, orientation="horizontal")

            mag = np.sqrt(((self.vertical_section[self.vertical_section!=0]))**2 + ((self.horizontal_section[self.horizontal_section!=0]))**2)
            vmin, vmax = np.nanmin(mag), np.nanmax(mag)
            # Get current normalized limits (usually 0–1)
            nmin, nmax = cax.get_xlim()

            # Place ticks at the colorbar ends
            cax.set_xticks([nmin, nmax])

            # Replace text only
            cax.set_xticklabels([f"{vmin:.1f}", f"{vmax:.1f}"])

            cb.set_label(self.unit)
            cb.ax.xaxis.set_label_position('top')

        # Add labels
        self.ax.set_ylabel("Elevation (m)")
        self.ax.set_xlabel("Distance (km)")


class Mogi():
    def __init__(self, ax, xcen, ycen):
        self.x = xcen
        self.y = ycen
        self._plot_source(ax)

    def _plot_source(self, ax):
        ax.scatter(self.x, self.y, s=15, color="black", linewidth=2, marker="x")


class Spheroid():
    def __init__(self, ax, xcen, ycen, s_axis_max, ratio, strike, dip):
        self.x = xcen
        self.y = ycen
        self.s_axis = s_axis_max
        self.ratio = ratio
        self.strike = strike
        self.dip = dip
        self._plot_source(ax)

    def _plot_source(self, ax):
        # Calculate semi-minor axis
        s_minor = self.s_axis * self.ratio

        # Convert angles to radians
        strike_rad = np.radians(self.strike - 90)
        dip_rad = np.radians(self.dip)

        # Adjust the semi-major axis length for the dip projection
        s_axis_projected = self.s_axis * np.sin(dip_rad)

        # Calculate endpoints of the major axis (with dip projection)
        dx_major = s_axis_projected * np.cos(strike_rad)
        dy_major = s_axis_projected * np.sin(strike_rad)
        x_major = [self.x - dx_major, self.x + dx_major]
        y_major = [self.y - dy_major, self.y + dy_major]

        # Calculate endpoints of the minor axis (without dip projection)
        dx_minor = s_minor * np.sin(strike_rad)
        dy_minor = s_minor * -np.cos(strike_rad)
        x_minor = [self.x - dx_minor, self.x + dx_minor]
        y_minor = [self.y - dy_minor, self.y + dy_minor]

        ax.plot(x_major, y_major, 'r-', label='Major Axis')  # Major axis in red
        ax.plot(x_minor, y_minor, 'b-', label='Minor Axis')  # Minor axis in blue


class Penny():
    def __init__(self, ax, xcen, ycen, radius):
        self.x = xcen
        self.y = ycen
        self.radius = radius
        self._plot_source(ax)

    def _plot_source(self, ax):
        circle = plt.Circle((self.x, self.y), self.radius, edgecolor='black', color="#7cc0ff", fill=True, alpha=0.7, label='Penny')
        ax.add_patch(circle)


class Okada:
    def __init__(self, ax, xtlc, ytlc, length, width, strike, dip):
        self.xtlc = xtlc
        self.ytlc = ytlc
        self.length = length
        self.width = width
        self.strike = strike
        self.dip = dip
        self._plot_source(ax)

    def _plot_source(self, ax):
        dip_radians = np.radians(self.dip)
        projected_width = self.width * np.cos(dip_radians)
        height = abs(projected_width)

        local_rect = Rectangle((0.0, -height),
                               self.length, height,
                            #    facecolor='black',
                               edgecolor='black',
                               lw=1,
                               alpha=0.2)
        # rotate around local origin (top-left) and translate to (xtlc, ytlc)
        t = Affine2D().rotate_deg(90 - self.strike).translate(self.xtlc, self.ytlc)
        local_rect.set_transform(t + ax.transData)
        ax.add_patch(local_rect)

        # add a single spike/triangle along the length that points in the down-dip direction
        # local coordinates: top edge is at y=0, down-dip is negative y
        try:
            from matplotlib.patches import Polygon
            # main triangle size
            base_half_main = max(0.04 * self.length, 0.01 * self.length)

            tri_color = 'black'
            tri_edge = 'black'
            tri_alpha = 0.3

            # add several smaller triangles along the fault length with same color/alpha
            n_extra = 6
            extra_positions = np.linspace(0.1, 0.9, n_extra)
            base_half_small = base_half_main * 0.5
            tip_offset_small = 0.6
            for pos in extra_positions:
                # skip center position to avoid overlapping the main triangle
                if abs(pos - 0.5) < 1e-6:
                    continue
                left = (pos * self.length - base_half_small, 0.0)
                right = (pos * self.length + base_half_small, 0.0)
                tip = (pos * self.length, -height * tip_offset_small)
                tri_small = Polygon([left, right, tip], closed=True,
                                    facecolor=tri_color, edgecolor=tri_edge, linewidth=0.6,
                                    zorder=29, alpha=tri_alpha)
                tri_small.set_transform(t + ax.transData)
                ax.add_patch(tri_small)
        except Exception:
            # non-fatal: continue without spikes if something goes wrong
            pass


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