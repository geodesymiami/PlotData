from abc import ABC, abstractmethod
import requests
import pygmt
import numpy as np
from datetime import datetime
from mintpy.utils import readfile
from mintpy.objects.coord import coordinate
from mintpy.objects import timeseries, HDFEOS
from plotdata.helper_functions import get_bounding_box, expand_bbox, set_default_section


class DataFetcher(ABC):
    def __init__(self, base_url, params):
        self.base_url = base_url
        self.params = params

    @abstractmethod
    def construct_url(self, *args, **kwargs):
        pass

    def fetch_data(self, *args, **kwargs):
        url = self.construct_url(*args, **kwargs)
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


class DataFetcherFactory:
    @staticmethod
    def create_fetcher(website, **kwargs):
        if website == "usgs":
            print("-" * 50)
            print("USGS database\n")

            return USGSDataFetcher(**kwargs)
        elif website == "anotherwebsite":
            return AnotherWebsiteDataFetcher(**kwargs)
        else:
            raise ValueError(f"Unknown website: {website}")


class USGSDataFetcher(DataFetcher):
    API_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(self, start_date, end_date, magnitude, params=None):
        super().__init__(self.API_ENDPOINT, params or {})
        self.start_date = start_date
        self.end_date = end_date
        self.magnitude = magnitude

    def construct_url(self, max_lat, min_lat, max_lon, min_lon):
        self.params.update({
            "format": "geojson",
            "starttime": self.start_date,
            "endtime": self.end_date,
            "maxlatitude": max_lat,
            "minlatitude": min_lat,
            "maxlongitude": max_lon,
            "minlongitude": min_lon,
            "minmagnitude": self.magnitude
        })
        return f"{self.base_url}?{'&'.join([f'{k}={v}' for k, v in self.params.items()])}"


class AnotherWebsiteDataFetcher(DataFetcher):
    API_ENDPOINT = "https://anotherwebsite.com/api/data"

    def __init__(self, some_param, params=None):
        super().__init__(self.API_ENDPOINT, params or {})
        self.some_param = some_param

    def construct_url(self):
        self.params.update({
            "param": self.some_param
        })
        return f"{self.base_url}?{'&'.join([f'{k}={v}' for k, v in self.params.items()])}"


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
        self._define_unit_measure()

    def _define_unit_measure(self):
        if not hasattr(self, 'unit') or not self.unit:
            print(f"[Warning] Unit '{self.unit}' not recognized. No conversion applied.")

        for key, value in self.dataset.items():
            attributes = value.get('attributes', {})
            start_date_key = next((k for k in attributes if k.lower() == 'start_date'), None)
            end_date_key = next((k for k in attributes if k.lower() == 'end_date'), None)

            if start_date_key and end_date_key:
                start_date = datetime.strptime(attributes[start_date_key], "%Y%m%d")
                end_date = datetime.strptime(attributes[end_date_key], "%Y%m%d")

                number_of_days = (end_date - start_date).days
                attributes['days'] = number_of_days

            if key == 'vectors':
                for k, v in self.dataset[key].items():
                    if 'horizontal' in k or 'vertical' in k:
                        attributes = v.get('attributes', {})

                        start_date_key = next((attr_key for attr_key in attributes if attr_key.lower() == 'start_date'), None)
                        end_date_key = next((attr_key for attr_key in attributes if attr_key.lower() == 'end_date'), None)

                        if start_date_key and end_date_key:
                            start_date = datetime.strptime(attributes[start_date_key], "%Y%m%d")
                            end_date = datetime.strptime(attributes[end_date_key], "%Y%m%d")

                            number_of_days = (end_date - start_date).days
                            attributes['days'] = number_of_days

        for key, value in self.dataset.items():
            units = {
                'mm/yr': 1000,
                'cm/yr': 100,
                'm/yr': 1,
            }
            if key == 'timeseries':
                if 'mm' in self.unit or 'cm' in self.unit:
                    conversion_factor = 1000 if 'mm' in self.unit else 100
                    for k,v in value.items():
                        if 'data' in v:
                            v['data'] *= conversion_factor
                            v['attributes']['unit'] = self.unit
            if 'data' in value:
                days = value['attributes'].get('days', 1)
                units.update({
                    'mm': 1000 * 365.25 / days,
                    'cm': 100 * 365.25 / days,
                    'm': 365.25 / days,
                    })
                if self.unit in units:
                    value['data'] *= units[self.unit]
                    value['attributes']['unit'] = self.unit
                else:
                    raise ValueError(f"Unit '{self.unit}' is not recognized.")

            if key == 'vectors':
                for k, v in value.items():
                    if ('horizontal' in k and 'horizontal' not in self.dataset) or ('vertical' in k and 'vertical' not in self.dataset):
                        if self.unit in units:
                            v['data'] *= units[self.unit]
                            v['attributes']['unit'] = self.unit
                        else:
                            raise ValueError(f"Unit '{self.unit}' is not recognized.")

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
        # TODO Fix stupid mintpy behaviour
        if "geo_" in geometry:
            coord = coordinate(atr, lookup_file=geometry.replace("geo_",""))
        else:
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
        self.region = [min(longitude), max(longitude), min(latitude), max(latitude)]

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
                    geometry["geometry"]["data"] = geometry["geometry"]["data"]
            else:
                if 'SenA' in file:
                    geometry["geometry"] = self._extract_geometry_data(self.ascending_geometry)
                elif 'SenD' in file:
                    geometry["geometry"] = self._extract_geometry_data(self.descending_geometry)
                    geometry["geometry"]["data"] = geometry["geometry"]["data"]

            dictionary.update(geometry)

        if self.seismicity:
            earthquakes = self._extract_earthquakes()
            dictionary['earthquakes'] = earthquakes

        if not self.line or type(self.line) == float:
            self.line = set_default_section(self.line, self.region)

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

    def _get_pygmt_dem(self, region):
        relief = pygmt.datasets.load_earth_relief(resolution=self.resolution, region=region)
        elevation = relief.values.astype(float)
        elevation[elevation < 0] = 0

        lon = relief.coords["lon"].values
        lat = relief.coords["lat"].values

        return lon , lat, elevation

    def _extract_geometry_data(self, file=None):
        # TODO REVIEW FOR GEOMETRY FILE
        if file and True:
            atr = readfile.read_attribute(file)
            if atr['FILE_TYPE'] == 'geometry':
                # TODO DITCHED THE GEOMETRY FILE BECAUSE SUCKS
                elevation = readfile.read(file, datasetName='height')[0]
                # latitude = readfile.read(file, datasetName='latitude')[0]
                # longitude = readfile.read(file, datasetName='longitude')[0]
                latitude, longitude = get_bounding_box(atr)
                if not latitude or not longitude:
                    latitude, longitude = self.region[2:4], self.region[0:2]

                # TODO actually not needed
                # if atr['passDirection'] == 'DESCENDING' or 'SenD' in file:
                #     elevation = elevation#np.flip(elevation)

                if np.isnan(elevation).any() and False: # TODO Let nan be there for now
                    lon, lat, elevation = self._get_pygmt_dem(self.region)
                    atr["region"] = [min(lon), max(lon), min(lat), max(lat)]

                if not isinstance(elevation, np.ndarray):
                    self.elevation = self.elevation.where(self.elevation >= 0, 0)

                if 'Y_STEP' in atr:
                    dlon = float(atr['X_STEP'])
                    dlat = float(atr['Y_STEP'])
                else:
                    dlon = (max(longitude) - min(longitude)) / (int(atr['WIDTH']) - 1)
                    dlat = (min(latitude) - max(latitude)) / (int(atr['LENGTH']) - 1)

                lon1d = min(longitude) + np.arange(int(atr['WIDTH'])) * dlon
                lat1d = max(latitude) + np.arange(int(atr['LENGTH'])) * dlat

                atr["longitude"] = lon1d
                atr["latitude"] = lat1d
                atr["region"] = [np.nanmin(longitude), np.nanmax(longitude), np.nanmin(latitude), np.nanmax(latitude)]

                dictionary = {
                    'data': elevation,
                    'attributes': atr,
                    }

                return dictionary
            else:
                latitude, longitude = get_bounding_box(atr)
                self.region = [max(longitude), max(longitude), min(latitude), max(latitude)]

        atr = {}

        lon, lat, elevation = self._get_pygmt_dem(self.region)

        atr["longitude"] = lon
        atr["latitude"] = lat
        atr["region"] = [min(lon), max(lon), min(lat), max(lat)]

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
            "depth" : [],
            "moment" : []
        }
        for feature in features:
            timestamp = feature['properties']['time'] / 1000
            date_time = datetime.utcfromtimestamp(timestamp).date()

            latitude = feature['geometry']['coordinates'][1]
            longitude = feature['geometry']['coordinates'][0]
            depth = feature['geometry']['coordinates'][2]

            earthquakes["date"].append(date_time)
            earthquakes["lalo"].append((latitude, longitude))
            earthquakes["magnitude"].append(float(feature['properties']['mag']))
            earthquakes["depth"].append(depth)

        return earthquakes