import os
import math
import pygmt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
from plotdata.objects.section import Section
from plotdata.objects.create_map import Mapper, Isolines, Relief
from plotdata.objects.earthquakes import Earthquake
from plotdata.helper_functions import draw_vectors

def run_plot(plot_info, inps):
    vmin = inps.vlim[0] if inps.vlim else None
    vmax = inps.vlim[1] if inps.vlim else None
    # fig = plt.figure(figsize=(10,10))
    plots = []
    title = plot_info[list(plot_info.keys())[0]]['directory'].split('/')[-1]

    if inps.plot_type == 'shaded_relief':
        main_gs = gridspec.GridSpec(1, 1, figure=fig) #rows, columns
        ax = fig.add_subplot(main_gs[0, 0])
        rel_map = Mapper(ax=ax, region=inps.region)
        Relief(map=rel_map, resolution = inps.resolution, interpolate=inps.interpolate, no_shade=inps.no_shade, zorder=None)

    if inps.plot_type == 'velocity':
        fig, axs = plt.subplots(nrows=len(inps.data_dir), ncols=len(plot_info.keys()),layout="constrained", squeeze=False)
        fig.suptitle(title)

        for col,key in enumerate(plot_info.keys()):
            # Extract files
            asc_file = plot_info[key]['ascending'][0] if plot_info[key]['ascending'] else None
            desc_file = plot_info[key]['descending'][0] if plot_info[key]['descending'] else None

            files = [file for file in [asc_file, desc_file] if file is not None]

            # Create subplots and set properties
            axes = [axs[i, col] for i in range(len(files))]

            for ax, file in zip(axes, files):
                if file:
                    processing_maps(ax=ax, file=file, no_dem=inps.no_dem, resolution=inps.resolution, interpolate=inps.interpolate, no_shade=inps.no_shade, style=inps.style, vmin=vmin, vmax=vmax, isolines=inps.isolines, iso_color=inps.iso_color, linewidth=inps.linewidth, inline=inps.inline, earthquake=inps.earthquake, movement=inps.movement)

                ax.set_box_aspect(0.5)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            # Title
            axs[0,col].set_title(key)

    if inps.plot_type == 'horzvert':
        rows = 1 if (inps.plot and inps.plot != 'horzvert') else 2

        fig, axs = plt.subplots(nrows=rows, ncols=len(plot_info.keys()),layout="constrained", squeeze=False)
        fig.suptitle(title)

        for col,key in enumerate(plot_info.keys()):
            # Extract files
            horz_file = plot_info[key]['horizontal'] if inps.plot != 'vertical' else None
            vert_file = plot_info[key]['vertical']if inps.plot != 'horizontal' else None

            files = [file for file in [horz_file, vert_file] if file is not None]

            # Create subplots and set properties
            axes = [axs[i, col] for i in range(len(files))]

            for ax, file in zip(axes, files):
                if file:
                    processing_maps(ax=ax, file=file, no_dem=inps.no_dem, resolution=inps.resolution, interpolate=inps.interpolate, no_shade=inps.no_shade, style=inps.style, vmin=vmin, vmax=vmax, isolines=inps.isolines, iso_color=inps.iso_color, linewidth=inps.linewidth, inline=inps.inline, earthquake=inps.earthquake, movement=inps.movement)

                ax.set_box_aspect(0.5)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
 
    if inps.plot_type == 'vectors':
        fig, axs = plt.subplots(nrows=3, ncols=len(plot_info.keys()),layout="constrained")
        fig.suptitle(title)

        if len(plot_info.keys()) == 1:
            axs = np.reshape(axs, (3, 1))
        else:
            axs = np.atleast_2d(axs)

        for col,key in enumerate(plot_info.keys()):
            # Extract files
            asc_file, desc_file, horz_file, vert_file = (
                plot_info[key]['ascending'][0],
                plot_info[key]['descending'][0],
                plot_info[key]['horizontal'],
                plot_info[key]['vertical']
            )

            axes = [
                axs[0, col],
                axs[1, col],
                axs[2, col]
            ]

            axs[0,col].set_title(key)

            for ax in axes:
                ax.set_box_aspect(0.5)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

            if inps.plot != 'horzvert':
                plot1_file = asc_file
                plot2_file = desc_file
            else:
                plot1_file = horz_file
                plot2_file = vert_file

            asc_data = processing_maps(ax=axes[0],
                            file=plot1_file,
                            no_dem=inps.no_dem,
                            resolution=inps.resolution,
                            interpolate=inps.interpolate,
                            no_shade=inps.no_shade,
                            style=inps.style,
                            vmin=vmin,
                            vmax=vmax,
                            isolines=inps.isolines,
                            iso_color=inps.iso_color,
                            linewidth=inps.linewidth,
                            inline=inps.inline,
                            earthquake=inps.earthquake,
                            movement=inps.movement)

            desc_data = processing_maps(ax=axes[1],
                            file=plot2_file,
                            no_dem=inps.no_dem,
                            resolution=inps.resolution,
                            interpolate=inps.interpolate,
                            no_shade=inps.no_shade,
                            style=inps.style,
                            vmin=vmin,
                            vmax=vmax,
                            isolines=inps.isolines,
                            iso_color=inps.iso_color,
                            linewidth=inps.linewidth,
                            inline=inps.inline,
                            earthquake=inps.earthquake,
                            movement=inps.movement)

            # Process horizontal data
            horizontal_data = Mapper(file=horz_file)
            # Get horizontal data section
            horizontal_section = Section(
                horizontal_data.velocity,
                horizontal_data.region,
                inps.line[1],  # Vertical coordinate
                inps.line[0]   # Horizontal coordinate
            )

            # Process vertical data
            vertical_data = Mapper(file=vert_file)
            # Get vertical data section
            vertical_section = Section(
                vertical_data.velocity,
                vertical_data.region,
                inps.line[1],
                inps.line[0]
            )

            # Create elevation data
            elevation = Relief(
                map=horizontal_data,
                resolution=inps.resolution
            )

            # Get elevation data section
            elevation_section = Section(
                elevation.elevation,
                elevation.map.region,
                inps.line[1],
                inps.line[0]
            )

            # Calculate vector components
            x, v, h = draw_vectors(
                elevation_section.values,
                vertical_section.values,
                horizontal_section.values,
                inps.line
            )

            # Get figure dimensions
            fig_width, fig_height = fig.get_size_inches()

            # Calculate vector scaling factors
            v_adj = 2 * max(elevation_section.values) / max(x)
            h_adj = 1 / v_adj

            rescale_h = h_adj / fig_width
            rescale_v = v_adj / fig_height

            # Plot elevation profile
            axes[2].plot(x, elevation_section.values)

            # Configure plot limits
            axes[2].set_ylim([0, 2 * max(elevation_section.values)])
            axes[2].set_xlim([min(x), max(x)])

            # Calculate vector scaling parameters
            start_x, start_y = max(x) * 0.1, (2 * max(elevation_section.values) * 0.8)
            magnitudes = np.sqrt(v**2 + h**2)
            # non_zero_magnitudes = magnitudes[magnitudes != 0] # No need??

            # Resample vectors
            for i in range(len(h)):
                if i % inps.resample_vector != 0:
                    h[i] = 0
                    v[i] = 0

            # Cancel vectors outside the plot
            for i in range(len(h)):
                if np.any(h[i] * rescale_h + x[i] < x[0]) or np.any(h[i] * rescale_h + x[i] > x[-1]):
                    h[i] = 0
                    v[i] = 0

            # Filter out zero-length vectors
            non_zero_indices = np.where((h != 0) | (v != 0))
            filtered_x = x[non_zero_indices]
            filtered_h = h[non_zero_indices]
            filtered_v = v[non_zero_indices]
            filtered_elevation = elevation_section.values[non_zero_indices]

            # Plot velocity vectors
            axes[2].quiver(
                filtered_x,
                filtered_elevation,
                filtered_h * rescale_h,
                filtered_v * rescale_v,
                color='red',
                scale_units='xy',
                width=(1 / 10**(2.5))
            )

            axes[0].plot(inps.line[0], inps.line[1], '-', linewidth=2, alpha=0.7, color='black')
            axes[1].plot(inps.line[0], inps.line[1], '-', linewidth=2, alpha=0.7, color='black')

            # Mean velocity
            axes[2].quiver([start_x], [start_y], [abs(np.mean(filtered_h * rescale_h))],[0], color='red', scale_units='xy', width=(1 / 10**(2.5)))
            axes[2].text(start_x, start_y*1.03, f"{round(abs(np.mean(filtered_h * rescale_h)), 3)} m/yr", color='black', ha='left', fontsize=8)
            # Control Vecor
            if False:
                axes[2].quiver([start_x], [start_y], [0],[np.mean(rescale_v)], color='red', scale_units='xy', width=(1 / 10**(2.5)))

    if inps.plot_type == 'timeseries':
        pass

    plt.show()


def processing_maps(ax, file, no_dem, resolution, interpolate, no_shade, style, vmin, vmax, isolines, iso_color, linewidth, inline, earthquake=False, movement=None):
    map = Mapper(ax=ax, file=file)

    if not no_dem:
        Relief(map=map, resolution = resolution, cmap = 'terrain', interpolate=interpolate, no_shade=no_shade, zorder=None)

    map.add_file(style=style, vmin=vmin, vmax=vmax, zorder=None, movement=movement)

    if isolines != 0:
        Isolines(map=map, resolution = resolution, color = iso_color, linewidth = linewidth, levels = isolines, inline = inline, zorder = None) # TODO add zorder

    if earthquake:
        Earthquake(map=map).map()

    return map


def point_on_globe(latitude, longitude, size='1'):
    fig = pygmt.Figure()

    # Set up orthographic projection centered on your point
    fig.basemap(
        region="d",  # Global domain
        projection=f"G{np.mean(longitude)}/{np.mean(latitude)}/15c",  # Centered on your coordinates
        frame="g"  # Show gridlines only
    )

    # Add continent borders with black lines
    fig.coast(
        shorelines="1/1p,black",  # Continent borders
        land="white",  # Land color
        water="white"  # Water color
    )

    # Plot your central point
    fig.plot(
        x=longitude,
        y=latitude,
        style=f"t{size}c",  # Triangle marker
        fill="red",  # Marker color
        pen="1p,black"  # Outline pen
    )