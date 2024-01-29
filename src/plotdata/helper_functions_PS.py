import os
import numpy as np
import geopandas as gpd
import contextily as ctx
from pathlib import Path
import simplekml
import colorsys
from shapely.geometry import box
from mintpy.utils import readfile, arg_utils, utils as ut

def calculate_mean_amplitude(slcStack, out_amplitude):
    """
    Calculate the mean amplitude from the SLC stack and save it to a file.

    Args:
        slcStack (str): Path to the SLC stack file.
        out_amplitude (str): Path to the output amplitude file.

    Returns:
        None
    """

    with h5py.File(slcStack, 'r') as f:
        slcs = f['slc']
        s_shape = slcs.shape
        mean_amplitude = np.zeros((s_shape[1], s_shape[2]), dtype='float32')
        lines = np.arange(0, s_shape[1], 100)

        for t in lines:
            last = t + 100
            if t == lines[-1]:
                last = s_shape[1]  # Adjust the last index for the final block

            # Calculate mean amplitude for the current block
            mean_amplitude[t:last, :] = np.mean(np.abs(f['slc'][:, t:last, :]), axis=0)

        # Save the calculated mean amplitude to the output file
        np.save(out_amplitude, mean_amplitude)

def default_backscatter_file():
    options = ['mean_amplitude.npy', '../inputs/slcStack.h5']
    for option in options:
        if os.path.exists(option):
            print(f'Using {option} for backscatter.')
            return option
    raise FileNotFoundError(f'No backscatter file found {options}.')

def add_open_street_map_image(ax, coords):
    geometry = [box(coords['lon1'], coords['lat1'], coords['lon2'],coords['lat2'])]
    gdf = gpd.GeoDataFrame({'geometry': geometry}, crs='EPSG:4326')
    gdf.plot(ax=ax, facecolor="none", edgecolor='none')
    ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_xlim(coords['lon1'], coords['lon2'])
    ax.set_ylim(coords['lat1'], coords['lat2'])
    ax.set_axis_off()

def add_satellite_image(ax):
    pass

def add_geotiff_image(ax, gtif, coords, cmap='Greys_r'):
    data_coords = coords['lon1'], coords['lon2'], coords['lat1'], coords['lat2']
    my_image = georaster.MultiBandRaster(gtif,
                                         bands='all',
                                         load_data=data_coords,
                                         latlon=True)
    ax.imshow(my_image.r,
              extent=my_image.extent,
              cmap=cmap)

def add_dsm_image(inps, ax):
    pass

def add_backscatter_image(ax, amplitude):
    ax.imshow(amplitude, cmap='gray', vmin=0, vmax=300)

def change_reference_point(data, ref_lalo):
    """Change reference point of data to ref_lalo"""
    ref_lat = ref_lalo[0]
    ref_lon = ref_lalo[1]
    points_lalo = np.array([ref_lat, ref_lon])
    ref_y, ref_x = coord.geo2radar(points_lalo[0], points_lalo[1])[:2]
    data -= data[ref_y, ref_x]
    return data

def create_kml_3D_file(inps):
    """ create a 3D kml file """

    # Create a new KML object
    kml = simplekml.Kml()

    # Define the coordinates and altitudes

    if inps.kml_3d_key == 'velocity' or inps.kml_3d_key == 'displacement':
        coords = list(zip(inps.lon, inps.lat, inps.estimated_elevation_data, inps.data))
    else:
        coords = list(zip(inps.lon, inps.lat, inps.estimated_elevation_data, inps.estimated_elevation_data))

    # coords = [(-80.121244, 25.872155, alt, vel) for alt, vel in zip([4, 20, 45], [0.1, 0.3, 2.0])]
    # coords = [(-80.121244, 25.872155, alt, vel) for alt, vel in zip([4, 20, 45], [4, 20, 45])]

    min_key = min(key for _, _, _, key in coords)
    max_key = max(key for _, _, _, key in coords)

    # Create a point for each coordinate
    for i, coord in enumerate(coords):
        _, _, _, key = coord

        # Map the altitude to a value in the range [0, 0.7]
        key_norm = 0.7 * (key - min_key) / (max_key - min_key)

        # Convert the normalized altitude to a color in the RGB color space
        r, g, b = colorsys.hsv_to_rgb(key_norm, 1.0, 1.0)

        # Create the point without a name
        pnt = kml.newpoint()
        pnt.coords = [coord]  # Set the coordinates
        pnt.altitudemode = simplekml.AltitudeMode.relativetoground  # Set the altitude mode
        pnt.style.iconstyle.scale = 1.0  # Set the scale of the icon
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'  # Set the icon
        pnt.style.iconstyle.color = simplekml.Color.rgb(int(r * 255), int(g * 255), int(b * 255))  # Set the color

    # Save the KML file
    kml_file = "points.kml"
    kml.save(kml_file)

    # Print the command to open the KML file with Google Earth
    print(f'open -a "Google Earth Pro.app" {os.path.abspath(kml_file)}')


