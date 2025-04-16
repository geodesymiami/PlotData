import numpy as np
import pandas as pd
from plotdata.helper_functions import calculate_distance



class Section():
    def __init__(self, data, region, latitude, longitude) -> None:
        self.data = data
        self.region = region

        lat_indices, lon_indices = self.draw_line(self.data, self.region, latitude, longitude)

        # TODO test if data needs to be flipped (apparantly needs to be flipped)
        if True:
            data = np.flipud(data)

        # Extract the values data along the snapped path
        values = data[lat_indices, lon_indices]

        # Add the data to the DataFrame
        self.path_df['values'] = values

        # Extract the values profile
        self.values = self.path_df['values']

        self.values = np.nan_to_num(self.values)


    def draw_line(self, data, region, latitude, longitude):
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

        # Create a DataFrame to store the path data
        self.path_df = pd.DataFrame({
            'longitude': lon_points,
            'latitude': lat_points,
            'lon_index': lon_indices,
            'lat_index': lat_indices,
            'distance': np.linspace(0, 1, num_points)  # Normalized distance
        })

        return lat_indices, lon_indices


    def plot_line(self, ax=None, zorder=None):
        total_distance = 0
        distances = [0]

        for i in range(len(self.path_df) - 1):
            total_distance += calculate_distance(
                self.path_df['latitude'][i], self.path_df['longitude'][i],
                self.path_df['latitude'][i + 1], self.path_df['longitude'][i + 1]
            )
            distances.append(total_distance)

        # Create a distance array that matches the length of the values data
        distance = np.linspace(0, total_distance, len(self.values))

        # Plot the values data against the distance array
        ax.plot(distance, self.values, 'k-', linewidth=1, zorder=zorder)
        ax.set_ylabel("values (m)")
        ax.set_xlabel("Length (km)")
        ax.set_ylim([min(self.values) * 0.9 , max(self.values) * 1.1])


    def plot_vectors(self, ax=None):
        total_distance = 0
        distances = [0]

        for i in range(len(self.path_df) - 1):
            total_distance += calculate_distance(
                self.path_df['latitude'][i], self.path_df['longitude'][i],
                self.path_df['latitude'][i + 1], self.path_df['longitude'][i + 1]
            )
            distances.append(total_distance)

        # Create a distance array that matches the length of the values data
        distance = np.linspace(0, total_distance, len(self.values))

        # Plot the values data against the distance array
        ax.scatter(distance, self.values, c='black', marker='o')