"""Utility classes for plotting functionality.

This module contains helper classes that handle specific plotting concerns,
following the Single Responsibility Principle.
"""

import math
import numpy as np
from typing import Optional, List, Dict, Any
from matplotlib.axes import Axes
from matplotlib.patheffects import withStroke
from matplotlib.colors import LightSource
from plotdata.helper_functions import resize_to_match


class ScalePlotter:
    """Handles plotting of scale bars on maps."""
    
    def plot_scale(self, ax: Axes, zorder: int) -> None:
        """Plot a scale bar on the given axes.
        
        Args:
            ax: Matplotlib axes object
            zorder: Z-order for layer stacking
        """
        lon1, lon2 = ax.get_xlim()
        lat1, lat2 = ax.get_ylim()
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
        ax.plot([x0, x0 + dlon], [y0, y0], color='k', lw=1)
        tick_h = 0.005 * abs(lat_span)
        ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color='k', lw=2)
        ax.plot([x0 + dlon, x0 + dlon], [y0 - tick_h, y0 + tick_h], color='k', lw=2)
        
        # label centered under the bar
        ax.text(
            x0 + dlon/2, 
            y0 + 0.06 * abs(lat_span), 
            f"{dist_km:.0f} km", 
            ha='center', 
            va='top', 
            fontsize=8, 
            path_effects=[withStroke(linewidth=1.5, foreground='white')], 
            zorder=zorder
        )


class DEMPlotter:
    """Handles plotting of Digital Elevation Model (DEM) data."""
    
    def plot_dem(
        self, 
        ax: Axes, 
        geometry: np.ndarray, 
        attributes: Dict[str, Any], 
        region: List[float], 
        data: Optional[np.ndarray] = None, 
        zorder: int = 0
    ) -> None:
        """Plot DEM/hillshade data.
        
        Args:
            ax: Matplotlib axes object
            geometry: DEM data array
            attributes: Dictionary containing latitude, longitude, and step info
            region: Region bounds [lon_min, lon_max, lat_min, lat_max]
            data: Optional data array to resize DEM to match
            zorder: Z-order for layer stacking
        """
        print("-"*50)
        print("Plotting DEM data...\n")

        if not isinstance(geometry, np.ndarray):
            z = geometry.astype(float)
        else:
            z = geometry

        lat = attributes['latitude']
        lon = attributes['longitude']

        if lon.ndim > 1:
            dlon = float(attributes['X_STEP'])
            dlat = float(attributes['Y_STEP'])

            lon_min = min(region[0:2])
            lat_max = max(region[2:4])
            ny, nx = z.shape

            lon1d = lon_min + np.arange(nx) * dlon
            lat1d = lat_max + np.arange(ny) * dlat

            lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        else:
            dlon = lon[1] - lon[0]
            dlat = lat[1] - lat[0]
            lon2d, lat2d = np.meshgrid(lon, lat)
 
        if data is not None:
            z = resize_to_match(z, data, 'DEM')
            lat2d = resize_to_match(lat2d, data, 'latitude')
            lon2d = resize_to_match(lon2d, data, 'longitude')

        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * np.cos(np.radians(np.nanmean(lat2d)))

        dx = dlon * meters_per_deg_lon
        dy = dlat * meters_per_deg_lat

        # Compute hillshade with real spacing
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(z, vert_exag=0.7, dx=dx, dy=dy)

        # Plot hillshade
        ax.imshow(
            hillshade,
            cmap='gray',
            extent=region,
            origin='upper',
            alpha=0.5,
            zorder=zorder,
            rasterized=True
        )


class AxisLimitsManager:
    """Manages axis limits and zoom functionality."""
    
    def update_limits(
        self, 
        ax: Axes, 
        subset: Optional[str] = None, 
        zoom: Optional[float] = None
    ) -> None:
        """Update axis limits based on subset or zoom settings.
        
        Args:
            ax: Matplotlib axes object
            subset: Subset string in format 'lat,lon:lat2,lon2'
            zoom: Zoom factor (>1 zooms in)
        """
        if subset:
            try:
                # Split the string into two parts
                coords1, coords2 = subset.split(':')

                # Split each part into lat and lon
                lat1, lon1 = map(float, coords1.split(','))
                lat2, lon2 = map(float, coords2.split(','))

                # Assign to x_min, x_max, y_min, y_max
                x_min, x_max = sorted([lon1, lon2])  # Longitude corresponds to x-axis
                y_min, y_max = sorted([lat1, lat2])  # Latitude corresponds to y-axis

            except ValueError:
                raise ValueError(
                    f"Invalid subset format: {subset}. "
                    f"Expected format is 'lat,lon:lat2,lon2'."
                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        elif zoom:
            # Get current axis limits
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # Calculate the range
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Calculate the new limits
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            new_x_range = x_range / zoom
            new_y_range = y_range / zoom

            new_x_min = x_center - new_x_range / 2
            new_x_max = x_center + new_x_range / 2
            new_y_min = y_center - new_y_range / 2
            new_y_max = y_center + new_y_range / 2

            # Set the new axis limits
            ax.set_xlim(new_x_min, new_x_max)
            ax.set_ylim(new_y_min, new_y_max)
