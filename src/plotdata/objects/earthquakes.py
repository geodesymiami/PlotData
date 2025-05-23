import sys
import os

sys.path.insert(0, '/Users/giacomo/code/Plotdata/src')

from datetime import datetime
from matplotlib import pyplot as plt
from plotdata.objects.create_map import Mapper
from plotdata.helper_functions import draw_box, calculate_distance
from plotdata.volcano_functions import get_volcano_coord_id, get_volcano_coord_name
from plotdata.objects.get_methods import DataFetcherFactory


class Earthquake():
    def __init__(self, start_date=None, end_date = None, distance_km = 20, distance_deg = None, magnitude = 1, id=None, volcano: str = None, region: list = None, map: Mapper = None):
        # Constants
        self.API_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
        self.PARAMS = {
            "eventtype": "earthquake",
            "orderby": "time",
        }
        self.magnitude = magnitude

        if volcano or id:
            self.define_info(start_date, end_date, distance_km, distance_deg, volcano, id)

        if region:
            self.region = region
            self.start_date = datetime.strptime(start_date,'%Y%m%d') if isinstance(start_date, str) else start_date
            self.end_date = datetime.today() if not end_date else datetime.strptime(end_date, '%Y%m%d') if isinstance(end_date, str) else end_date

        if map:
            self.region = map.region
            self.start_date = map.start_date
            self.end_date = map.end_date
            self.ax = map.ax
            self.zorder = map.get_next_zorder()


        self.get_earthquake_data(website="usgs")


    def map(self):
        if not self.earthquakes['date']:
            print("No earthquake data available.")
            return

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
                zorder=self.zorder
            )

    def define_info(self, start_date, end_date, distance_km = 20, distance_deg = None, volcano=None, id=None):
        if volcano:
            self.coordinates, self.id = get_volcano_coord_id(None, volcano)
        if id:
            self.coordinates, self.volcano = get_volcano_coord_name(None, id)
        self.region = draw_box(self.coordinates[0], self.coordinates[1], distance_km, distance_deg)
        self.start_date = datetime.strptime(start_date,'%Y%m%d') if isinstance(start_date, str) else start_date
        self.end_date = datetime.today() if not end_date else datetime.strptime(end_date, '%Y%m%d') if isinstance(end_date, str) else end_date


    def get_earthquake_data(self, website="usgs"):
        min_lon, max_lon, min_lat, max_lat = self.region

        fetcher = DataFetcherFactory.create_fetcher(
            website=website,
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
    from plotdata.helper_functions import parse_polygon
    region = parse_polygon("POLYGON((130.3264 31.3226,130.9765 31.3226,130.9765 31.8198,130.3264 31.8198,130.3264 31.3226))")
    print(region)
    volcano = "Aira"
    start = "20170101"
    end = "20171010"
    dis = 20
    eq1 = Earthquake(volcano=volcano, start_date=start, end_date=end, distance_km=dis, magnitude=4)
    eq1.plot()
