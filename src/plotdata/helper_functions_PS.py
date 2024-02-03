import os
import h5py
import numpy as np
import geopandas as gpd
import contextily as ctx
import georaster
from pathlib import Path
import matplotlib.pyplot as plt
import simplekml
import zipfile
from PIL import Image
from shapely.geometry import box
from mintpy.utils import readfile, utils as ut
from mintpy import save_kmz

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
    raise FileNotFoundError(f'USER ERROR: No backscatter file found {options}.')

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

def change_reference_point(data, ref_lalo, file_type):
    """Change reference point of data to ref_lalo"""
    ref_lat = ref_lalo[0]
    ref_lon = ref_lalo[1]
    points_lalo = np.array([ref_lat, ref_lon])
    if file_type == 'HDFEOS':             # for data in radar coordinates (different for SARPROZ, Andreas)
        atr = readfile.read_attribute('velocity.h5')
        coord = ut.coordinate(atr, lookup_file='inputs/geometryRadar.h5')   # radar coord
        ref_y, ref_x = coord.geo2radar(points_lalo[0], points_lalo[1])[:2]
        data -= data[ref_y, ref_x]
    return data

def create_kml_file(inps):
    """ create a 3D kml file """

    print('create kml file, key:', inps.dataset)

    # Create a new KML object,  define the coordinates and altitudes
    kml = simplekml.Kml()
    coords = list(zip(inps.lon, inps.lat, inps.elevation, inps.data))

    min_key = min(key for _, _, _, key in coords)
    max_key = max(key for _, _, _, key in coords)

    for i, coord in enumerate(coords):
        _, _, _, key = coord

        # Map the altitude to a value in the range [0, 0.7]
        if inps.vlim:
            key_norm = 0.7 * (key - inps.vlim[0]) / (inps.vlim[1] - inps.vlim[0])
        else:
            key_norm = 0.7 (key - min_key) / (max_key - min_key)    

        # Convert the normalized altitude to a color in the RGB color space
        r, g, b, _ = plt.cm.jet(key_norm)

        # Create the point without a name
        pnt = kml.newpoint()
        
        if inps.kml_3d:
            pnt.coords = [coord]  # Set the coordinates
            pnt.altitudemode = simplekml.AltitudeMode.relativetoground  # Set the altitude mode
            pnt.style.iconstyle.scale = 1.0  # Set the scale of the icon
            pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'  # Set the icon
        else:
            pnt.coords = [(coord[0], coord[1])]  # Set the coordinates
            pnt.style.iconstyle.scale = 0.5  # Set the scale of the icon
            pnt.style.iconstyle.icon.href = 'https://maps.google.com/mapfiles/kml/shapes/road_shield3.png'  # Set the icon
     
        pnt.style.iconstyle.color = simplekml.Color.rgb(int(r * 255), int(g * 255), int(b * 255))  # Set the color
        pnt.description = get_balloon_description(coord, key, inps)  # Set the description

        # Define a balloon style for the point
        balloonstyle = simplekml.BalloonStyle()
        balloonstyle.text = pnt.description
        pnt.style.balloonstyle = balloonstyle

    # Create color scale
    save_kmz.generate_cbar_element(cbar_file='color_scale.png', cmap='jet',vmin=inps.vlim[0],vmax=inps.vlim[1],
                     unit=inps.label_dict['unit'],loc='lower left', nbins=None, label=inps.label_dict['str'])
    
    # Open the image file, Get the dimensions of the image, calculate the aspect ratio
    img = Image.open('color_scale.png')
    width, height = img.size
    aspect_ratio = width * 0.7 / height

    # Create a ScreenOverlay for the color scale
    overlay = kml.newscreenoverlay(name='Color Scale')
    overlay.icon.href = 'color_scale.png'
    overlay.overlayxy = simplekml.OverlayXY(x=0, y=0, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    overlay.screenxy = simplekml.ScreenXY(x=0, y=0, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)
    overlay.size = simplekml.Size(x=0.35 * aspect_ratio, y=0.35, xunits=simplekml.Units.fraction, yunits=simplekml.Units.fraction)

    # Add the overlay to the KML object
    kml.screenoverlay = overlay

    # Save the KML file
    kml_file = "points.kml"
    kml.save(kml_file)
    
    # Create a new zipfile object
    kmz_file = kml_file.replace('kml','kmz')
    with zipfile.ZipFile(kmz_file, 'w') as myzip:
        # Add the KML file to the zipfile
        myzip.write(kml_file)
        # Add the image file to the zipfile
        myzip.write('color_scale.png')

    print(f'open -a "Google Earth Pro.app" {kmz_file}')

def get_balloon_description(coord, key, inps ):
    """Get the balloon description for a point"""
    lat, lon, elevation, key = coord
    
    str = ''
    # if inps.dataset == 'velocity' or inps.dataset == 'displacement':
    str += f"{inps.label_dict['str']}: {key:.2f} {inps.label_dict['unit']}\n"
    str += f"Elevation: {elevation:.1f} m\n"
    str += f"Lat, Lon: {lon:.6f},{lat:.6f}"
    return str

def correct_geolocation(inps):
    """Correct the geolocation using DEM error"""
    print('Run geolocation correction ...')

    latitude = inps.lat
    longitude = inps.lon
    dem_error = inps.dem_error

    az_angle = np.deg2rad(float(inps.HEADING))
    inc_angle = np.deg2rad(inps.inc_angle)

    rad_latitude = np.deg2rad(latitude)

    one_degree_latitude = 111132.92 - 559.82 * np.cos(2*rad_latitude) + \
                            1.175 * np.cos(4 * rad_latitude) - 0.0023 * np.cos(6 * rad_latitude)

    one_degree_longitude = 111412.84 * np.cos(rad_latitude) - \
                            93.5 * np.cos(3 * rad_latitude) + 0.118 * np.cos(5 * rad_latitude)

    print('one_degree_latitude, one_degree_longitude:', np.mean(one_degree_latitude), np.mean(one_degree_longitude))

    dx = np.divide((dem_error) * (1 / np.tan(inc_angle)) * np.cos(az_angle), one_degree_longitude)  # converted to degree
    dy = np.divide((dem_error) * (1 / np.tan(inc_angle)) * np.sin(az_angle), one_degree_latitude)  # converted to degree

    sign = np.sign(latitude)
    latitude += sign * dy

    sign = np.sign(longitude)
    longitude += sign * dx

    inps.lat = latitude
    inps.lon = longitude    
    return 

