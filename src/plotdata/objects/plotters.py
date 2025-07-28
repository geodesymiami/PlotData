import math
import pygmt
import numpy as np
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
from plotdata.helper_functions import draw_vectors, unpack_file, calculate_distance, get_bounding_box, expand_bbox


# TODO Create a class to just extract data
class DataExtractor:
    def __init__(self, plotter_map, process) -> None:
        for attr in dir(process):
            if not attr.startswith('__') and not callable(getattr(process, attr)):
                setattr(self, attr, getattr(process, attr))
        self.plotter_map = plotter_map

        self.dispatch_map = {
            "timeseries": self._extract_timeseries_data,
            "ascending": self._extract_velocity_data,
            "descending": self._extract_velocity_data,
            "horizontal": self._extract_velocity_data,
            "vertical": self._extract_velocity_data,
            "vectors": self._extract_vector_data,
            "seismicmap": self._make_seismicmap,
            "seismicity": self._make_seismicity,
        }

        self._fetch_data()

    def _fetch_data(self):
        self.dataset = {}
        for name, configs in self.plotter_map.items():
            if any(name in element for row in self.layout for element in row):
                attributes = configs.get("attributes", [])
                files = []

                for attr in attributes:
                    if hasattr(self, attr):
                        files.append(getattr(self, attr))
                    else:
                        print(f"[Warning] Attribute '{attr}' not found in process object.")

                handler = self.dispatch_map.get(name)
                if handler:
                    for file in files:
                        self.dataset.setdefault(name, {}).update(handler(file))
                else:
                    print(f"[Warning] No handler defined for plot type '{name}'.\n")

    def _extract_vector_data(self, file):
        if "up" in file:
            direction = "vertical"
        elif "hz" in file:
            direction = "horizontal"

        if direction in self.dataset:
            result = {direction: self.dataset[direction]}
            if "geometry" not in self.dataset["vectors"]:
                geometry_file = self.ascending_geometry or self.descending_geometry
                result["geometry"] = self._extract_geometry_data(geometry_file)
            return result

        if "vectors" in self.dataset:
            result = {direction: self._extract_velocity_data(file)}

            # TODO prob not needed now
            # if "geometry" in result:
            #     self.dataset["vectors"]["geometry"] = result["geometry"]

            # if "geometry" not in self.dataset["vectors"]:
            #     geometry_file = self.ascending_geometry or self.descending_geometry
            #     result["geometry"] = self._extract_geometry_data(geometry_file)
            return result

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

        # Extract timeseries data
        data = readfile.read(file, datasetName=date_list)[0]

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

        if 'passDirection' in atr:
            direction = atr['passDirection'].lower()
        else:
            direction = 'ascending' if 'SenA' in file else 'descending'

        dictionary = {
            direction:{
                'data': ts,
                'dates': date_list,
                'attributes': atr,}}

        return dictionary

    def _extract_velocity_data(self, file):
        """Extracts velocity data from the given file."""
        direction = "vertical" if "up" in file else "horizontal" if "hz" in file else None

        if (direction and self.dataset.get("vectors") and self.dataset["vectors"].get(direction)):
            vector_data = self.dataset["vectors"][direction]
            return {**vector_data, "geometry": self._extract_geometry_data(file)}

        velocity = readfile.read(file)[0]
        atr = readfile.read_attribute(file)
        latitude, longitude = get_bounding_box(atr)
        self.region = [longitude[0], longitude[1], latitude[0], latitude[1]]

        if self.region:
            atr['region'] = self.region

        dictionary = {
            'data': velocity,
            'attributes': atr,
        }

        if not self.no_dem:
            geometry = {}
            if 'passDirection' in atr:
                if atr['passDirection'] == 'ASCENDING':
                    geometry["geometry"] = self._extract_geometry_data(self.ascending_geometry)
                elif atr['passDirection'] == 'DESCENDING':
                    geometry["geometry"] = self._extract_geometry_data(self.descending_geometry)
            else:
                if 'SenA' in file:
                    geometry["geometry"] = self._extract_geometry_data(self.ascending_geometry)
                elif 'SenD' in file:
                    geometry["geometry"] = self._extract_geometry_data(self.descending_geometry)

            dictionary.update(geometry)

        if self.seismicity:
            earthquakes = self._extract_earthquakes()
            dictionary['earthquakes'] = earthquakes

        return dictionary

    def _make_seismicmap(self, file):
        dictionary = {}
        if "seismicmap" in self.dataset and self.dataset["seismicmap"]:
            return dictionary

        for key in self.dataset.keys():
            if any("geometry" in item for item in self.dataset[key]):
                for key, value in self.dataset[key].items():
                    if "geometry" in key:
                        dictionary['geometry'] = value
                        dictionary['earthquakes'] = self._extract_earthquakes()
                        return dictionary

        dictionary = self._extract_geometry_data(file)
        dictionary['earthquakes'] = self._extract_earthquakes()

        return dictionary

    def _extract_geometry_data(self, file=None):
        if file:
            atr = readfile.read_attribute(file)
            if atr['FILE_TYPE'] == 'geometry':
                elevation = readfile.read(file, datasetName='height')[0]
                elevation = np.flipud(elevation)
                elevation[np.isnan(elevation)] = 0

                dictionary = {
                    'data': elevation,
                    'attributes': atr,
                    }

                return dictionary
            else:
                latitude, longitude = get_bounding_box(atr)
                self.region = [longitude[0], longitude[1], latitude[0], latitude[1]]

        elevation = pygmt.datasets.load_earth_relief(resolution=self.resolution, region=self.region)
        dictionary = {
                'data': elevation,
                'attributes': atr,
                }

        return dictionary

    def _make_seismicity(self, file=None):
        earthquakes = self._extract_earthquakes(file)

        if "date" in earthquakes and not earthquakes["date"]:
            self.magnitude = 7
            original_region = self.region
            self.region = expand_bbox(original_region)

            earthquakes["attributes"] = {"region": self.region, "magnitude": self.magnitude}

            earthquakes = self._extract_earthquakes(file)

            # Reset
            self.magnitude = None
            self.region = original_region
        if "attributes" not in earthquakes:
            earthquakes["attributes"] = {"region": self.region, "magnitude": self.magnitude}

        return earthquakes

    def _extract_earthquakes(self, file=None):
        """Extracts earthquake data based on the specified parameters."""
        dictionary = {}
        if "seismicity" in self.dataset and self.dataset["seismicity"]:
            return dictionary

        website = self.website if hasattr(self, "website") else "usgs"

        if not hasattr(self, 'region') or not self.region:
            atr = readfile.read_attribute(file)
            latitude, longitude = get_bounding_box(atr)
            self.region = [min(longitude), max(longitude), min(latitude), max(latitude)]

        min_lon, max_lon, min_lat, max_lat = self.region

        if not hasattr(self, 'magnitude') or not self.magnitude:
            self.magnitude = self.seismicity if self.seismicity else 2

        start_date = datetime.strptime(self.start_date,'%Y%m%d') if isinstance(self.start_date, str) else self.start_date
        end_date = datetime.strptime(self.end_date,'%Y%m%d') if isinstance(self.end_date, str) else self.end_date

        fetcher = DataFetcherFactory.create_fetcher(
            website=website,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            magnitude=self.magnitude
        )
        data = fetcher.fetch_data(
            max_lat=max_lat,
            min_lat=min_lat,
            max_lon=max_lon,
            min_lon=min_lon
        )

        features = data['features']

        earthquakes = {
            "date" : [],
            "lalo" : [],
            "magnitude" : [],
            "moment" : []
        }
        for feature in features:
            timestamp = feature['properties']['time'] / 1000
            date_time = datetime.utcfromtimestamp(timestamp).date()

            latitude = feature['geometry']['coordinates'][1]
            longitude = feature['geometry']['coordinates'][0]

            earthquakes["date"].append(date_time)
            earthquakes["lalo"].append((latitude, longitude))
            earthquakes["magnitude"].append(float(feature['properties']['mag']))

        return earthquakes

###################################### TEST ########################################

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

############ # TODO Less readable??

        if False:
            self.data = dataset.get("data")
            # Get attributes from geometry if attribute doesnt exist, else None
            self.attributes = dataset.get("attributes", dataset.get("geometry", {}).get("attributes"))

            if "region" in self.attributes:
                self.region = self.attributes["region"]
            elif not hasattr(self, 'region'):
                latitude, longitude = get_bounding_box(self.attributes)
                self.region = [longitude[0], longitude[1], latitude[0], latitude[1]]

            self.geometry = None
            if not self.no_dem:
                self.geometry = next(
                    (dataset[key]["data"] for key in dataset if "geometry" in key),
                    None
                )

            self.earthquakes = dataset.get("earthquakes")

############ # TODO More readable??

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
                    break
                else:
                    self.geometry = None

        if "earthquakes" in dataset:
            self.earthquakes = dataset["earthquakes"]

############

        self.zorder = 0

    def _get_next_zorder(self):
        z = self.zorder
        self.zorder += 1
        return z

    def _plot_velocity(self):
        # TODO change to argparse
        zorder = self._get_next_zorder()
        cmap = 'jet'

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
        # TODO change with deformation m/yr
        # cbar.set_label(self.label)
        cbar.locator = ticker.MaxNLocator(3)
        cbar.update_ticks()

    def _plot_dem(self):
        print("Plotting DEM data...\n")

        zorder = self._get_next_zorder()

        if not isinstance(self.geometry, np.ndarray):
            self.z = self.geometry.astype(float)
        else:
            self.z = self.geometry

        lon_min, lon_max, lat_min, lat_max = self.region

        # Get the shape of the 2D array (e.g., elevation data)
        n_lat, n_lon = self.z.shape

        # Generate longitude and latitude arrays
        lon = np.linspace(lon_min, lon_max, n_lon)
        lat = np.linspace(lat_min, lat_max, n_lat)

        # Compute spacing for hillshade
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

            lon = np.linspace(self.region[0], self.region[1], grid_np.shape[1])
            lat = np.linspace(self.region[2], self.region[3], grid_np.shape[0])

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

        if hasattr(self, 'earthquakes') and self.earthquakes is not None:
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
        ax.set_title(f'Earthquake Magnitudes Over Time at {self.earthquakes.get("attributes", {}).get("region", "Unknown Region")}')
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
        ax_ts.set_ylabel('LOS displacement (m)')
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
        self.geometry = dataset["horizontal"].get("geometry") if "geometry" in dataset["horizontal"] else dataset["vertical"].get("geometry")

        self._process_sections()

####################################################################################

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


class VelocityPlot_old:
    """Handles the plotting of velocity maps."""
    def __init__(self, file: str, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))
        self.file = file[0] if isinstance(file, list) else file
        self.attr = readfile.read_attribute(self.file)

    def on_click(self, event):
        if event.inaxes == self.ax:  # Ensure the click is within the correct axis
            if event.inaxes == self.ax:  # Ensure the click is within the plot
                print(f"--lalo={event.ydata},{event.xdata}\n")
                print(f"--ref-lalo={event.ydata},{event.xdata}\n")

    def _set_default_section(self):
        mid_lat = self.line if type(self.line) == float else (max(self.region[2:4]) + min(self.region[2:4]))/2
        mid_lon = (max(self.region[0:2]) + min(self.region[0:2]))/2

        size = (max(self.region[0:2]) - min(self.region[0:2]))*0.25

        latitude = (mid_lat, mid_lat)
        longitude = (mid_lon - size, mid_lon + size)

        return [longitude, latitude]

    def plot(self, ax):
        """Creates and configures the velocity map."""
        self.ax = ax
        vel_map = Mapper(ax=self.ax, file=self.file)
        self.region = vel_map.region

        # Add relief if not disabled
        if not self.no_dem:
            if self.attr['passDirection'] == 'ASCENDING':
                geometry = self.ascending_geometry
            elif self.attr['passDirection'] == 'DESCENDING':
                geometry = self.descending_geometry
            else:
                geometry = None

            Relief(map=vel_map, geometry=geometry, resolution=self.resolution, cmap='terrain',
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
                fontsize=5,
                ha='left',
                va='top',
                color='white',
                bbox=dict(facecolor='gray', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3')
            )

        if self.lalo and 'point' in self.ax.get_label():
            vel_map.plot_point([self.lalo[0]], [self.lalo[1]], marker='x')

        if 'section' in self.ax.get_label():
            if not self.line or type(self.line) == float:
                self.line = self._set_default_section()
            self.ax.plot(self.line[0], self.line[1], '--', linewidth=1.5, alpha=0.7, color='black')

        if self.ref_lalo:
            vel_map.plot_point([self.ref_lalo[0]], [self.ref_lalo[1]], marker='s')

        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        return vel_map


class VectorsPlot_old:
    """Handles the plotting of velocity maps, elevation profiles, and vector fields."""
    def __init__(self, files: list, inps):
        for attr in dir(inps):
            if not attr.startswith('__') and not callable(getattr(inps, attr)):
                setattr(self, attr, getattr(inps, attr))
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

        # Determine which files to plot
        self._set_plot_files()

        self._process_sections()

    def _set_plot_files(self):
        """Determines which velocity files to use based on plot options."""
        self.plot1_file = self.horz_file
        self.plot2_file = self.vert_file

    def _process_sections(self):
        """Processes horizontal, vertical, and elevation sections."""
        self.horizontal_data = Mapper(file=self.horz_file)
        self.vertical_data = Mapper(file=self.vert_file)
        self.elevation_data = Relief(map=self.horizontal_data, resolution=self.resolution)

        self.region = self.elevation_data.map.region

        if not self.line or type(self.line) == float:
            self.line = self._set_default_section()

        self.horizontal_section = Section(
            np.flipud(self.horizontal_data.velocity), self.horizontal_data.region, self.line[1], self.line[0]
        )
        self.vertical_section = Section(
            np.flipud(self.vertical_data.velocity), self.vertical_data.region, self.line[1], self.line[0]
        )
        self.elevation_section = Section(
            self.elevation_data.elevation.values, self.elevation_data.map.region, self.line[1], self.line[0]
        )

    def _set_default_section(self):
        mid_lat = self.line if type(self.line) == float else (max(self.region[2:4]) + min(self.region[2:4]))/2
        mid_lon = (max(self.region[0:2]) + min(self.region[0:2]))/2

        size = (max(self.region[0:2]) - min(self.region[0:2]))*0.25

        latitude = (mid_lat, mid_lat)
        longitude = (mid_lon - size, mid_lon + size)

        return [longitude, latitude]

    def _compute_vectors(self):
        """Computes velocity vectors and scaling factors."""
        x, v, h, self.z = draw_vectors(
            self.elevation_section.values, self.vertical_section.values, self.horizontal_section.values, self.line
        )

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
        mean_velocity = np.sqrt(np.mean((self.vertical_section.values[self.vertical_section.values!=0]))**2 + np.mean((self.horizontal_section.values[self.horizontal_section.values!=0]))**2)
        rounded_mean_velocity = round(mean_velocity, 4) if mean_velocity else round(mean_velocity, 3)

        self.ax.quiver([start_x], [start_y], [mean_velocity], [0], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        # self.ax[2].quiver([start_x], [start_y], [0], [abs(np.mean(self.filtered_v))], color='#ff7366', scale_units='xy', width=(1 / 10**(2.5)))
        self.ax.text(start_x, start_y * 1.03, f"{rounded_mean_velocity:.4f} m/yr", color='black', ha='left', fontsize=8)

        # Add labels
        self.ax.set_ylabel("Elevation (m)")
        self.ax.set_xlabel("Distance (km)")


class TimeseriesPlot_old:
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