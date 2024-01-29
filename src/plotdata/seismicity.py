import requests
from datetime import datetime, timezone
import pandas as pd

def get_earthquakes(start_date, end_date, plot_box, depth_range="0 10", mag_range="0 10"):
    # Define the API endpoint and parameters
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    depth_max, depth_min = map(lambda x: -float(x), depth_range.split())
    
    params = {
       "format": "geojson",
       "starttime": start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:],
       "endtime": end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:],
       "minlatitude": plot_box[0],
       "maxlatitude": plot_box[1],
       "minlongitude": plot_box[2],
       "maxlongitude": plot_box[3],
       "mindepth": depth_min,   # Minimum depth (in kilometers)
       "maxdepth": depth_max,   # Maximum depth (in kilometers)
    }
    
    # Make a request to the USGS API
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extract earthquake information from the API response
    earthquakes = data["features"]
    earthquake_data = []

    # Calculate the Unix timestamp in milliseconds of start and end time
    min_time = int(datetime.strptime(start_date, "%Y%m%d").timestamp() * 1000)
    max_time = int(datetime.strptime(end_date, "%Y%m%d").timestamp() * 1000)

    for quake in earthquakes:
        magnitude = quake["properties"]["mag"]
        latitude = quake["geometry"]["coordinates"][1]
        longitude = quake["geometry"]["coordinates"][0]
        depth = quake["geometry"]["coordinates"][2]
        time = quake["properties"]["time"]
        
        earthquake_data.append([time,latitude, longitude, depth, magnitude])
    
    # Create a DataFrame from the earthquake data
    columns = ["Time", "Latitude", "Longitude", "Depth", "Magnitude"]
    events_df = pd.DataFrame(earthquake_data, columns=columns)
    return events_df

def normalize_earthquake_times(events_df, start_date, end_date):
# Normalize times for colormap (use the Unix timestamp in milliseconds)
    min_time = int(datetime.strptime(start_date, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    max_time = int(datetime.strptime(end_date, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    norm_times = [(time - min_time) / (max_time - min_time) for time in events_df["Time"]]
    return norm_times