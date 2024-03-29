import os
import numpy as np
import geopandas as gpd
import contextily as ctx
import georaster
import matplotlib.pyplot as plt
import simplekml
import zipfile
from PIL import Image
from shapely.geometry import box
from mintpy import save_kmz
from mintpy.tsview import plot_ts_scatter
from mintpy.utils import  plot as pp, ptime

def plot_scatter(ax, inps, marker='o', colorbar=True):
    
    if  inps.background == 'open_street_map' or  inps.background == 'satellite' or inps.background == 'dem':

        im1 = ax.scatter(inps.lon, inps.lat, c=inps.data, s=inps.point_size, cmap=inps.colormap, marker=marker)
        
        if inps.ref_lalo and inps.show_reference_point:
            ax.scatter(inps.ref_lalo[1], inps.ref_lalo[0], color='black', s=inps.point_size*1.2, marker='s')
        
        if inps.lalo:
            i=0
            for point in inps.lalo:
                ax.scatter(point[1], point[0], color='white', s=inps.point_size*2.0, marker='s')
                ax.scatter(point[1], point[0], color='black', s=inps.point_size*1.0, marker=inps.marker_list[i])
                if inps.marker_number:
                    ax.text(point[1], point[0], str(i), color='black', fontsize=10)
                i += 1

    elif  inps.background == 'backscatter':
        # Create a boolean mask for the condition
        mask = (inps.yv < inps.amplitude.shape[0]) & (inps.xv < inps.amplitude.shape[1])
        xv_filtered = inps.xv[mask]
        yv_filtered = inps.yv[mask]
        data_filtered = inps.data[mask]
        # data_filtered = inps.data
        
        im1 = ax.scatter(xv_filtered, yv_filtered, c=data_filtered, s=inps.point_size, cmap=inps.colormap, marker=marker)
        # im = ax.scatter(inps.xv, inps.yv, c=inps.data, s=inps.point_size, cmap=inps.colormap, marker=marker)
   
    if colorbar:
        cbar = plt.colorbar(im1, ax=ax, shrink=1, orientation='horizontal', pad=0.02)
        cbar.set_label(inps.label_dict['str'] + ' [' + inps.label_dict['unit'] + ']' )
        if inps.vlim is not None:
            clim=(inps.vlim[0], inps.vlim[1])
            im1.set_clim(clim[0], clim[1])

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def add_open_street_map_image(ax, coords, background_type='open_street_map'):
    geometry = [box(coords['lon1'], coords['lat1'], coords['lon2'],coords['lat2'])]
    gdf = gpd.GeoDataFrame({'geometry': geometry}, crs='EPSG:4326')
    gdf.plot(ax=ax, facecolor="none", edgecolor='none')
    if background_type == 'open_street_map':
        ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
    elif background_type == 'satellite':
        ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.Esri.WorldImagery)

    dark_overlay = np.zeros((int(ax.figure.bbox.height), int(ax.figure.bbox.width), 4))
    dark_overlay[..., :3] = 0  # Set RGB channels to 0 (black)
    dark_overlay[..., 3] = 0.01  # Set alpha channel to 0.3 (30% opacity)
    ax.imshow(dark_overlay, extent=(*ax.get_xlim(), *ax.get_ylim()), zorder=1, origin='upper')

    ax.set_xlim(coords['lon1'], coords['lon2'])
    ax.set_ylim(coords['lat1'], coords['lat2'])
    ax.set_axis_off()

def add_dem_image(ax, dem_file, inps, cmap='Greys_r'):
    data_coords = inps.coords['lon1'], inps.coords['lon2'], inps.coords['lat1'], inps.coords['lat2']
    geo_box = inps.coords['lon1'], inps.coords['lat2'], inps.coords['lon2'], inps.coords['lat1'] 

    print('plotting DEM background ...')

    from mintpy.utils import plot as pp
    dem, dem_metadata, dem_pix_box = pp.read_dem(dem_file, geo_box=geo_box, print_msg=True )

    # use MintPy for shaded relief. MintPy does not support displaying elevation (FA 2/24)
    if inps.disp_dem_shade:
        # FA 2/24: needs inps from view.py for dem shading to work
        from mintpy.cli.view import cmd_line_parse
        iargs = ['velocity.h5']              # needs any file. inps.data_file may not work because it can be S*.he5
        tmp_inps = cmd_line_parse(iargs)                 
        tmp_inps.disp_dem_shade = inps.disp_dem_shade
        pp.plot_dem_background( ax=ax, geo_box=geo_box, dem=dem, inps=tmp_inps, print_msg=True)
    else:
        my_image = georaster.MultiBandRaster(dem_file, bands='all', load_data=data_coords, latlon=True)
        ax.imshow(my_image.r,extent=my_image.extent,cmap=cmap)

    return

def add_backscatter_image(ax, amplitude):
    ax.imshow(amplitude, cmap='gray', vmin=0, vmax=300)

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
            key_norm = 0.7 * (key - min_key) / (max_key - min_key)    

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
        os.remove(kml_file)     
        os.remove('color_scale.png')

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

def plot_timeseries(ax, index, inps):
    """Plot timeseries"""
        
    inps.marker = 'o' 
    inps.linewidth = 0
    inps.ex_date_list = None
    inps.font_size = None
    
    from argparse import Namespace
    ppar = Namespace(mfc='C0', ms=6.0, label = 'Displacement')
        
    inps.num_date = len(inps.date_list)
    inps.dates, inps.yearList = ptime.date_list2vector(inps.date_list)
    #(inps.ex_date_list, inps.ex_dates, inps.ex_flag) = read_exclude_date(inps.ex_date_list, inps.date_list)
    plot_ts_scatter(ax, inps.timeseries_at_point[index], inps, ppar)
    
    # axis format
    cbar_label = 'Displacement ' + ' [' + inps.label_dict['unit'] + ']' 
    ax.tick_params(which='both', direction='in', labelsize=inps.font_size,
                    bottom=True, top=True, left=True, right=True)
    pp.auto_adjust_xaxis_date(ax, inps.yearList, fontsize=inps.font_size)
    ax.set_ylabel(cbar_label, fontsize=inps.font_size)
    if inps.ylim:
        ax.set_ylim(inps.ylim)
    # if self.tick_right:
    #     ax.yaxis.tick_right()
    #     ax.yaxis.set_label_position("right")

    # title
    if inps. marker_number:
        title = f"Point: {inps.lalo[index][0]:.5f}, {inps.lalo[index][1]:.5f},  number: {index}"
    else:
        title = f"Point: {inps.lalo[index][0]:.5f}, {inps.lalo[index][1]:.5f}"

    ax.set_title(title, fontsize=inps.font_size)

    # legend
    # if len(self.ts_data) > 1:
    #     ax.legend(handles, labels)

    # Print to terminal
    print('\n---------------------------------------')
    print(title)
    float_formatter = lambda x: [float(f'{i:.2f}') for i in x]
    if len(inps.date_list) <= 1e3:
        print(float_formatter(inps.timeseries_at_point[0]))

    if not np.all(np.isnan(inps.timeseries_at_point)):
        # min/max displacement
        ts_min, ts_max = np.nanmin(inps.timeseries_at_point), np.nanmax(inps.timeseries_at_point)
        print( f"time-series range: [{ts_min:.2f}, {ts_max:.2f}] {inps.label_dict['unit']}")
