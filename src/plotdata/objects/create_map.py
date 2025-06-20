import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import re
import pygmt
import numpy as np
from mintpy.utils import readfile
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.ticker as ticker
from plotdata.helper_functions import parse_polygon, get_bounding_box


class Mapper():
    def __init__(self, region=None, polygon=None, start_date=None, end_date=None, location_types: dict = {}, ax=None, file=None):
        if ax:
            self.ax = ax
            self.fig = ax.get_figure()

        if region:
            region = [lon + 360 if i < 2 and lon < 0 else lon for i, lon in enumerate(region)]
            self.region = region
            self.start_date = datetime.strptime(start_date, '%Y%m%d') if isinstance(end_date, str) else start_date
            self.end_date = datetime.strptime(end_date, '%Y%m%d') if isinstance(end_date, str) else end_date
        # TODO this works only for geocoded
        elif file:
            self.velocity, self.metadata = readfile.read(file)
            self.start_date = datetime.strptime(self.metadata['START_DATE'], '%Y%m%d')
            self.end_date = datetime.strptime(self.metadata['END_DATE'], '%Y%m%d')
            latitude, longitude = get_bounding_box(self.metadata)
            self.region = [longitude[0], longitude[1], latitude[0], latitude[1]]

        elif polygon:
            self.region = parse_polygon(polygon)

        self.zorder = 0
        self.location_types = location_types


    def get_next_zorder(self):
        z = self.zorder
        self.zorder += 1
        return z


    def calculate_displacement(self):
        self.start_date = datetime.strptime(self.metadata['START_DATE'], '%Y%m%d')
        self.end_date = datetime.strptime(self.metadata['END_DATE'], '%Y%m%d')
        days = (self.end_date - self.start_date).days
        self.displacement = (self.velocity * days / 365.25) # In meters


    def plot(self):
        self.add_legend()
        plt.show()


    def plot_point(self, lat, lon, marker='o', color='black', size=5, alpha=1, zorder=None):
        if not zorder:
            zorder = self.get_next_zorder()

        self.ax.plot(lon, lat, marker, color=color, markersize=size, alpha=alpha, zorder=zorder)


    def add_legend(self):
        handles = []
        for type, props in self.location_types.items():
            handle = plt.Line2D([0], [0], marker=props['marker'], color='w', label=type, markersize=10, markerfacecolor=props['color'])
            handles.append(handle)
        self.ax.legend(handles=handles, loc='upper right')


    def add_section(self, latitude, longitude, color='black', zorder=None):
        if not zorder:
            zorder = self.get_next_zorder()

        self.ax.plot(longitude, latitude, '-', linewidth=2, alpha=0.7, color=color, zorder=zorder)
        # self.ax.text(longitude[0], latitude[0], 'A', fontsize=10, ha='right', color=color)
        # self.ax.text(longitude[1], latitude[1], 'B', fontsize=10, ha='left', color=color)


    def add_file(self, style='scatter', vmin=None, vmax=None, zorder=None, cmap='jet', movement='velocity'):
        if not zorder:
            zorder = self.get_next_zorder()

        if not hasattr(self, 'displacement'):
            self.calculate_displacement()

        data = self.displacement if movement == 'displacement' else self.velocity
        label = 'Displacement (m)' if movement == 'displacement' else 'Velocity (m/yr)'

        if not vmin and not vmax:
            lim = max(abs(np.nanmin(data)), abs(np.nanmax(data))) * 1.2
            vmin, vmax = -lim, lim

        if style == 'ifgram':
            label = 'Displacement (m)'
            data_phase = (2 * np.pi / float(self.metadata['WAVELENGTH'])) *  self.displacement #(self.displacement + float(self.metadata['HEIGHT']))
            data_wrapped = np.mod(data_phase, 2 * np.pi)
            self.imdata = self.ax.imshow(data_wrapped, cmap=cmap, extent=self.region, origin='upper', interpolation='none',zorder=self.zorder, vmin=0, vmax=2 * np.pi)

        if style == 'pixel':
            self.imdata = self.ax.imshow(data, cmap=cmap, extent=self.region, origin='upper', interpolation='none', zorder=self.zorder, vmin=vmin, vmax=vmax, rasterized=True)
            # TODO this might cause issues, to test more
            self.ax.set_aspect('auto')

        elif style == 'scatter':
            # Assuming self.velocity is a 2D numpy array
            nrows, ncols = data.shape
            x = np.linspace(self.region[0], self.region[1], ncols)
            y = np.linspace(self.region[2], self.region[3], nrows)
            X, Y = np.meshgrid(x, y)
            X = X.flatten()
            Y = np.flip(Y.flatten())
            C = data.flatten()

            self.imdata = self.ax.scatter(X, Y, c=C, cmap=cmap, marker='o', zorder=self.zorder, s=2, vmin=vmin, vmax=vmax, rasterized=True)

        cbar = self.ax.figure.colorbar(self.imdata, ax=self.ax, orientation='horizontal', aspect=13)
        cbar.set_label(label)
        cbar.locator = ticker.MaxNLocator(3)
        cbar.update_ticks()

class Isolines:
        def __init__(self, map: Mapper, resolution = '01m', color = 'black', linewidth = 0.5, levels = 10, inline = False, zorder = None):
            self.map = map
            self.resolution = resolution
            self.color = color
            self.linewidth = linewidth
            self.levels = levels
            self.inline = inline

            if not zorder:
                self.zorder = self.map.get_next_zorder()
            else:
                self.zorder = zorder
                self.map.zorder = zorder

            # Plot isolines
            print("Adding isolines\n")
            lines = pygmt.datasets.load_earth_relief(resolution=self.resolution, region=self.map.region)

            grid_np = lines.values

            # Remove negative values
            grid_np[grid_np < 0] = 0

            # Convert the numpy array back to a DataArray
            lines[:] = grid_np

            # Plot the data
            lon = lines.coords["lon"].values
            lat = lines.coords["lat"].values
            z = lines.values

            cont = self.map.ax.contour(lon, lat, z, levels=self.levels, colors=self.color,
                                    linewidths=self.linewidth, zorder=self.zorder)

            if inline:
                self.map.ax.clabel(cont, inline=inline, fontsize=8)


class Relief:
    def __init__(self, map: Mapper, cmap = 'terrain', resolution = '01m', interpolate=False, no_shade=False, zorder=None):
        self.map = map
        self.cmap = cmap
        self.resolution = resolution
        self.interpolate = interpolate
        self.no_shade = no_shade

        if not zorder:
            self.zorder = self.map.get_next_zorder()
        else:
            self.zorder = zorder
            self.map.zorder = zorder

        # Plot colormap
        # Load the relief data
        print("Adding elevation\n")
        self.elevation = pygmt.datasets.load_earth_relief(resolution=self.resolution, region=self.map.region)

        if interpolate:
            self.interpolate_relief(self.resolution)

        # Set all negative values to 0
        self.elevation = self.elevation.where(self.elevation >= 0, 0)

        if hasattr(map, 'ax'):
            if not no_shade:
                self.im = self.shade_elevation(zorder=self.zorder)
            else:
                print('here')
                self.im = self.map.ax.imshow(self.elevation.values, cmap=self.cmap, extent=self.map.region, origin='lower', zorder=self.zorder, rasterized=True)


    def interpolate_relief(self, resolution):
        print("!WARNING: Interpolating the data to a higher resolution grid")
        print("Accuracy may be lost\n")
        # Interpolate the relief data to the new higher resolution grid
        digits = re.findall(r'\d+', resolution)
        letter = re.findall(r'[a-z]', resolution)
        new_grid_spacing = f'{(int(digits[0]) / 10)}{letter[0]}'

        self.elevation = pygmt.grdsample(grid=self.elevation, spacing=new_grid_spacing, region=self.map.region)


    def shade_elevation(self, vert_exag=1.5, zorder=None):
        # Create hillshade
        print("Shading the elevation data...\n")

        # Get the coordinates and data
        elev = self.elevation.values.astype(float)
        lon = self.elevation.coords["lon"].values
        lat = self.elevation.coords["lat"].values

        # Compute spacing for hillshade
        dlon = lon[1] - lon[0]
        dlat = lat[1] - lat[0]

        # Compute hillshade with real spacing
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(elev, vert_exag=vert_exag, dx=dlon, dy=dlat)

        # Create meshgrid of lon/lat edges for pcolormesh
        lon2d, lat2d = np.meshgrid(lon, lat)

        # Use pcolormesh to plot hillshade using real coordinates
        self.im = self.map.ax.pcolormesh(lon2d,lat2d,hillshade,cmap='gray',shading='auto',zorder=zorder,alpha=0.5,)