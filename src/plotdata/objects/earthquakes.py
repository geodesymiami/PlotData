import sys
import os

sys.path.insert(0, '/Users/giacomo/code/Playground/Plot_data2/src')

import requests
from datetime import datetime
from matplotlib import pyplot as plt
from plotdata.objects.create_map import Mapper
from plotdata.helper_functions import draw_box, calculate_distance
from plotdata.volcano_functions import get_volcano_coord_id
from plotdata.objects.get_methods import DataFetcherFactory


class Earthquake():
    def __init__(self, start_date, end_date = None, distance_km = 20, distance_deg = None, magnitude = 3, volcano: str = None, region: list = None, map: Mapper = None):
        # Constants
        self.API_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
        self.PARAMS = {
            "eventtype": "earthquake",
            "orderby": "time",
        }
        self.magnitude = magnitude

        if volcano:
            self.coordinates, self.id = get_volcano_coord_id(None, volcano)
            self.region = draw_box(self.coordinates[0], self.coordinates[1], distance_km, distance_deg)

        if region:
            self.region = region

        if map:
            self.region = map.region

        self.start_date = datetime.strptime(start_date,'%Y%m%d') if isinstance(start_date, str) else start_date
        self.end_date = datetime.today() if not end_date else datetime.strptime(end_date, '%Y%m%d') if isinstance(end_date, str) else end_date

        self.get_earthquake_data(website="usgs")


    def get_earthquake_data(self, website="usgs"):
        min_lon, max_lon, min_lat, max_lat = self.region

        fetcher = DataFetcherFactory.create_fetcher(
            website="usgs",
            start_date=self.start_date.isoformat(),
            end_date=self.end_date.isoformat(),
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

        self.earthquakes = earthquakes


    def print(self):
        for i in range(len(self.earthquakes['date'])):
            print(self.earthquakes['date'][i])
            print(self.earthquakes['magnitude'][i])
            print(self.earthquakes['lalo'][i])
            print(f"Distance from {self.volcano}: {calculate_distance(self.earthquakes['lalo'][i][0], self.earthquakes['lalo'][i][1], self.volcano['lat'], self.volcano['lon'])} km\n")


    def plot(self):
        if hasattr(self, 'coordinates'):
            fig = plt.figure(figsize=(10, 10))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            self.plot_by_date(ax1)
            self.plot_by_distance(ax2)

        plt.show()


    def plot_by_date(self, ax):
        # Plot EQs
        for i in range(len(self.earthquakes['date'])):
            ax.plot([self.earthquakes['date'][i], self.earthquakes['date'][i]], [self.earthquakes['magnitude'][i], 0], 'k-')

        ax.scatter(self.earthquakes['date'], self.earthquakes['magnitude'], c='black', marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Magnitude')
        ax.set_title('Earthquake Magnitudes Over Time')
        ax.set_xlim([self.start_date.date(), self.end_date.date()])
        ax.set_ylim([0, 10])


    def plot_by_distance(self, ax):
        # Plot EQs
        dist = []
        for i in range(len(self.earthquakes['date'])):
            dist.append(calculate_distance(self.earthquakes['lalo'][i][0], self.earthquakes['lalo'][i][1], self.coordinates[0], self.coordinates[1]))
            ax.plot([dist[i], dist[i]], [self.earthquakes['magnitude'][i], 0], 'k-')

        ax.set_xlim([0, max(dist)+ (max(dist) * 0.05)])
        ax.set_ylim([0, 10])

        ax.scatter(dist, self.earthquakes['magnitude'], c='black', marker='o')

        ax.set_xlabel('Distance in KM')
        ax.set_ylabel('Magnitude')
        ax.set_title('Earthquake Magnitudes from Volcano')


if __name__ == "__main__":
    volcano = "Kilauea"
    start = "20220101"
    end = "20221201"
    dis = 50
    eq = Earthquake(volcano=volcano, start_date=start, end_date=end, distance_km=dis, magnitude=4)
    eq.plot()