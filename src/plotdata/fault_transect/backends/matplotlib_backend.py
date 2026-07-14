#!/usr/bin/env python3
"""Matplotlib implementation of PlotBackend. Only this file imports matplotlib."""

import numpy as np

from plotdata.fault_transect.backends.base import PlotBackend
from plotdata.fault_transect.plot_api import (
    profile_axis_half_km, title_coords, map_view_lat_pad, map_view_lon_pad,
    compute_map_stack_step, stacked_map_axis_offset, stacked_map_ytick_pairs,
    stacked_map_xtick_pairs)
from plotdata.fault_transect.profiles import auto_stack_offset


class MatplotlibBackend(PlotBackend):

    def __init__(self):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._figures = []

    # ------------------------------------------------------------------ map
    def _offset_norm_limits(self, series, vlim):
        """Return (vmin, vmax) for offset coloring and curve y-axis auto-scale."""
        if vlim is not None:
            return vlim
        offsets = np.asarray(series.offset, dtype=float)
        finite_offs = offsets[np.isfinite(offsets)]
        olim = float(np.nanmax(np.abs(finite_offs))) if len(finite_offs) else 1.0
        if olim == 0:
            olim = 1.0
        return -olim, olim

    def _fault_geographic_bounds(self, spec):
        """Return (lon_min, lon_max, lat_min, lat_max) from fault segment vertices."""
        lons, lats = [], []
        for seg in spec.fault_segments:
            lons.extend(seg['lons'])
            lats.extend(seg['lats'])
        return min(lons), max(lons), min(lats), max(lats)

    def _map_stack_step(self, map_specs):
        """Stack step between period copies along map_stack_axis."""
        spec = map_specs[0]
        perp_km = spec.perp_width_km
        manual = spec.options.map_stack_offset
        axis = spec.options.map_stack_axis or 'lat'
        lon_min, lon_max, lat_min, lat_max = self._fault_geographic_bounds(spec)
        return compute_map_stack_step(
            axis, perp_km, lon_min, lon_max, lat_min, lat_max, manual)

    def _stacked_longitude_xticks(self, ax_map, lon_min, lon_max, pad, lon_step, n_periods):
        """X-axis ticks for the first (left) period only — true longitudes."""
        from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

        pairs = stacked_map_xtick_pairs(lon_min, lon_max, pad, lon_step, n_periods)
        if not pairs:
            return
        tick_locs = [loc for loc, _ in pairs]
        tick_labels = [f'{true_lon:.2f}' for _, true_lon in pairs]
        ax_map.xaxis.set_major_locator(FixedLocator(tick_locs))
        ax_map.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax_map.xaxis.set_minor_locator(NullLocator())

    def _stacked_latitude_yticks(self, ax_map, lat_min, lat_max, pad, lat_step, n_periods):
        """Y-axis ticks for the first (top) period only — true latitudes."""
        from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

        pairs = stacked_map_ytick_pairs(lat_min, lat_max, pad, lat_step, n_periods)
        if not pairs:
            return
        tick_locs = [loc for loc, _ in pairs]
        tick_labels = [f'{true_lat:.2f}' for _, true_lat in pairs]
        ax_map.yaxis.set_major_locator(FixedLocator(tick_locs))
        ax_map.yaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax_map.yaxis.set_minor_locator(NullLocator())

    def _shared_map_vlim(self, map_specs):
        """One colorscale for all periods in a combined map figure."""
        from plotdata.fault_transect.limits import shared_map_color_limits

        explicit = shared_map_color_limits([spec.options.vlim for spec in map_specs])
        if explicit is not None:
            return explicit
        vmin, vmax = None, None
        for spec in map_specs:
            lo, hi = self._offset_norm_limits(spec.offset_series, None)
            vmin = lo if vmin is None else min(vmin, lo)
            vmax = hi if vmax is None else max(vmax, hi)
        olim = max(abs(vmin), abs(vmax))
        if olim == 0:
            olim = 1.0
        return -olim, olim

    def _resolve_cmap(self, opts):
        from plotdata.fault_transect.colormaps import resolve_colormap
        return resolve_colormap(opts.colormap, vlist=opts.cmap_vlist)

    def _add_map_colorbar(self, fig, ax_map, mappable, opts):
        """Colorbar matched to the map axes height (never taller than the map panel)."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax_map)
        cax = divider.append_axes('right', size='4%', pad=0.08)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(f'offset ({opts.unit})', fontsize=opts.font_size)
        return cbar

    def _fault_colored_segments(self, series, vlim=None, lat_offset=0.0, lon_offset=0.0,
                                cmap=None):
        """Build LineCollection segments and colors along the fault sample path."""
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        lons = np.asarray(series.lon, dtype=float)
        lats = np.asarray(series.lat, dtype=float)
        along = np.asarray(series.along_km, dtype=float)
        offsets = np.asarray(series.offset, dtype=float)

        if len(along) < 2:
            return None

        finite_steps = np.diff(along[np.isfinite(along)])
        gap_break = 1.5 * float(np.median(finite_steps)) if len(finite_steps) else 1.0

        segs = []
        colors = []
        for i in range(len(along) - 1):
            if along[i + 1] - along[i] > gap_break:
                continue
            if not (np.isfinite(lons[i]) and np.isfinite(lats[i])
                    and np.isfinite(lons[i + 1]) and np.isfinite(lats[i + 1])):
                continue
            o0, o1 = offsets[i], offsets[i + 1]
            if np.isfinite(o0) and np.isfinite(o1):
                color_val = 0.5 * (o0 + o1)
            elif np.isfinite(o0):
                color_val = o0
            elif np.isfinite(o1):
                color_val = o1
            else:
                continue
            segs.append([
                (lons[i] + lon_offset, lats[i] + lat_offset),
                (lons[i + 1] + lon_offset, lats[i + 1] + lat_offset),
            ])
            colors.append(color_val)

        if not segs:
            return None

        vmin, vmax = self._offset_norm_limits(series, vlim)
        if cmap is None:
            cmap = 'RdBu_r'
        lc = LineCollection(segs, cmap=cmap, linewidths=10, capstyle='round',
                            norm=Normalize(vmin=vmin, vmax=vmax), zorder=3)
        lc.set_array(np.asarray(colors))
        return lc

    def _draw_map_text(self, ax_map, opts, lon_min, lon_max, lat_min, lat_max,
                       lat_offset=0.0, lon_offset=0.0, *, draw_title=True, draw_period=True):
        """Draw fault tag title and formatted period label on a map panel."""
        if draw_title and opts.title:
            x, y, ha, va = title_coords(
                opts.title_position, lon_min, lon_max, lat_min, lat_max,
                lat_offset=lat_offset,
                lon_offset=lon_offset,
                title_offset_lon=opts.title_offset_lon,
                title_offset_lat=opts.title_offset_lat)
            ax_map.text(x, y, opts.title, fontsize=opts.font_size + 1, va=va, ha=ha)
        if draw_period and opts.period_label:
            x, y, ha, va = title_coords(
                opts.period_label_position, lon_min, lon_max, lat_min, lat_max,
                lat_offset=lat_offset,
                lon_offset=lon_offset,
                title_offset_lon=opts.title_offset_lon,
                title_offset_lat=opts.title_offset_lat)
            ax_map.text(x, y, opts.period_label, fontsize=opts.font_size, va=va, ha=ha)

    def _draw_map_panel(self, fig, ax_map, ax_curve, spec, draw_curve=True, add_colorbar=True,
                        vlim=None, lat_offset=0.0, lon_offset=0.0, in_stack=False):
        opts = spec.options
        series = spec.offset_series
        color_lim = vlim if vlim is not None else opts.vlim
        cmap = self._resolve_cmap(opts)

        all_lons = []
        all_lats = []
        for seg in spec.fault_segments:
            all_lons.extend(seg['lons'])
            all_lats.extend(seg['lats'])

        ax_map.set_facecolor('white')

        lc = self._fault_colored_segments(series, color_lim, lat_offset=lat_offset,
                                          lon_offset=lon_offset, cmap=cmap)
        if lc is not None:
            ax_map.add_collection(lc)

        pad = map_view_lat_pad(spec.perp_width_km, min(all_lats), max(all_lats))
        if not in_stack:
            ax_map.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
            ax_map.set_ylim(min(all_lats) - pad + lat_offset, max(all_lats) + pad + lat_offset)
            ax_map.set_aspect('equal', adjustable='box')
            ax_map.set_xlabel('Longitude', fontsize=opts.font_size)
            ax_map.set_ylabel('Latitude', fontsize=opts.font_size)
        if lc is not None and add_colorbar:
            self._add_map_colorbar(fig, ax_map, lc, opts)
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        self._draw_map_text(ax_map, opts, lon_min, lon_max, lat_min, lat_max,
                            lat_offset=lat_offset, lon_offset=lon_offset,
                            draw_title=not in_stack, draw_period=True)

        if draw_curve and ax_curve is not None:
            ymin, ymax = self._offset_norm_limits(series, None)
            ax_curve.axhline(0, color='gray', linewidth=0.8)
            ax_curve.plot(series.along_km, series.offset, 'o-', color='#c0392b', markersize=4)
            ax_curve.set_ylim(ymin, ymax)
            ax_curve.set_xlabel('Distance along fault (km)', fontsize=opts.font_size)
            ax_curve.set_ylabel(f'{series.reference_side} - other ({opts.unit})', fontsize=opts.font_size)
            ax_curve.grid(alpha=0.3)
        return lc

    def _draw_stacked_map(self, fig, ax_map, map_specs):
        """Draw periods stacked along lat (vertical) or lon (horizontal)."""
        from matplotlib.colors import Normalize

        plt = self._plt
        opts = map_specs[0].options
        n_periods = len(map_specs)
        axis = opts.map_stack_axis or 'lat'
        stack_step = self._map_stack_step(map_specs)
        shared_lim = self._shared_map_vlim(map_specs)
        cmap = self._resolve_cmap(opts)

        lon_min, lon_max, lat_min, lat_max = self._fault_geographic_bounds(map_specs[0])
        lat_pad = map_view_lat_pad(map_specs[0].perp_width_km, lat_min, lat_max)
        lat_center = 0.5 * (lat_min + lat_max)
        lon_pad = map_view_lon_pad(map_specs[0].perp_width_km, lat_center, lon_min, lon_max)

        ax_map.set_facecolor('white')
        last_lc = None
        for i, spec in enumerate(map_specs):
            lat_offset, lon_offset = stacked_map_axis_offset(i, stack_step, axis)
            lc = self._draw_map_panel(fig, ax_map, None, spec, draw_curve=False,
                                      add_colorbar=False, vlim=shared_lim,
                                      lat_offset=lat_offset, lon_offset=lon_offset,
                                      in_stack=True)
            if lc is not None:
                last_lc = lc

        if axis == 'lon':
            x_left = lon_min - lon_pad
            x_right = lon_max + lon_pad + (n_periods - 1) * stack_step
            ax_map.set_xlim(x_left, x_right)
            ax_map.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
            self._stacked_longitude_xticks(ax_map, lon_min, lon_max, lon_pad, stack_step,
                                           n_periods)
        else:
            ax_map.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
            y_top = lat_max + lat_pad
            y_bottom = lat_min - lat_pad - (n_periods - 1) * stack_step
            ax_map.set_ylim(y_bottom, y_top)
            self._stacked_latitude_yticks(ax_map, lat_min, lat_max, lat_pad, stack_step,
                                          n_periods)

        ax_map.set_aspect('equal', adjustable='box')
        ax_map.set_xlabel('Longitude', fontsize=opts.font_size)
        ax_map.set_ylabel('Latitude', fontsize=opts.font_size)

        if last_lc is not None:
            self._add_map_colorbar(fig, ax_map, last_lc, opts)
        else:
            sm = plt.cm.ScalarMappable(norm=Normalize(*shared_lim), cmap=cmap)
            sm.set_array([])
            self._add_map_colorbar(fig, ax_map, sm, opts)

        if opts.title:
            fig.suptitle(opts.title, fontsize=opts.font_size + 2, y=0.98)

    def render_map_figure(self, map_specs, out_path):
        plt = self._plt

        n_periods = len(map_specs)
        opts = map_specs[0].options
        if n_periods == 1:
            fig, axes = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True,
                                     gridspec_kw={'height_ratios': [2.2, 1]}, squeeze=False)
            self._draw_map_panel(fig, axes[0, 0], axes[1, 0], map_specs[0])
        else:
            height = max(6, 3.5 * n_periods)
            fig, ax_map = plt.subplots(figsize=(9, height), constrained_layout=True)
            self._draw_stacked_map(fig, ax_map, map_specs)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    # ------------------------------------------------------------- profiles
    def _plot_profile_samples(self, ax, across_km, values, y_offset, color, connect, zorder=3,
                              markersize=3, linewidth=1.2, alpha=1.0):
        y = np.asarray(values, dtype=float) + y_offset
        x = np.asarray(across_km, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return
        if connect:
            ax.plot(x[finite], y[finite], color=color, linewidth=linewidth,
                    alpha=alpha, zorder=zorder - 1)
        ax.scatter(x[finite], y[finite], color=color, s=markersize ** 2, alpha=alpha,
                   linewidths=0, zorder=zorder)

    def _set_profile_xlim(self, ax, spec):
        half = spec.axis_half_km
        if half is None:
            half = profile_axis_half_km(spec.bundle.profile_length_km)
        ax.set_xlim(-half, half)

    def _draw_profile(self, ax, spec, main_index, y_offset=0.0):
        """Draw one main profile (black) plus its gray cloud on an axis."""
        bundle = spec.bundle
        for delta in range(-spec.cloud_profiles, spec.cloud_profiles + 1):
            if delta == 0:
                continue
            neighbor = bundle.get(main_index + delta)
            if neighbor is None:
                continue
            self._plot_profile_samples(ax, neighbor.across_km, neighbor.value, y_offset,
                                       color='gray', connect=spec.connect_lines,
                                       zorder=1, markersize=2, linewidth=0.4, alpha=0.6)
        main = bundle.get(main_index)
        if main is not None:
            self._plot_profile_samples(ax, main.across_km, main.value, y_offset,
                                       color='black', connect=spec.connect_lines,
                                       zorder=3, markersize=4, linewidth=1.2)
        return main

    def render_profiles_separate(self, spec, out_path_template):
        plt = self._plt
        opts = spec.options
        written = []
        for nn, idx in enumerate(spec.main_indices):
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            main = self._draw_profile(ax, spec, idx)
            if main is None:
                plt.close(fig)
                continue
            ax.axvline(0, color='#c0392b', linewidth=0.8, alpha=0.7)
            self._set_profile_xlim(ax, spec)
            ax.set_xlabel('Distance across fault (km)  [negative = left]', fontsize=opts.font_size)
            ax.set_ylabel(opts.unit, fontsize=opts.font_size)
            if opts.title:
                ax.set_title(opts.title, fontsize=opts.font_size + 1)
            ax.text(0.02, 0.92, f'profile {nn:02d} at {main.along_km:.1f} km',
                    transform=ax.transAxes, fontsize=opts.font_size - 1, va='top')
            if opts.period_label:
                ax.text(0.98, 0.98, opts.period_label, transform=ax.transAxes,
                        fontsize=opts.font_size, ha='right', va='top')
            ax.grid(alpha=0.3)
            path = out_path_template.replace('{nn}', f'{nn:02d}')
            fig.savefig(path, dpi=opts.dpi, bbox_inches='tight')
            self._figures.append(fig)
            written.append(path)
        return written

    def render_profiles_subplot(self, specs, out_path):
        plt = self._plt
        opts = specs[0].options
        rows = max(len(spec.main_indices) for spec in specs)
        # one column per period; subplot_cols only applies to a single period
        cols = specs[0].subplot_cols if len(specs) == 1 else len(specs)
        if len(specs) == 1:
            rows = int(np.ceil(len(specs[0].main_indices) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, max(1.8 * rows, 4)),
                                 sharex=True, constrained_layout=True, squeeze=False)
        if len(specs) == 1:
            spec = specs[0]
            for k, idx in enumerate(spec.main_indices):
                ax = axes[k // cols][k % cols]
                self._annotate_subplot(ax, spec, idx, opts)
            for k in range(len(spec.main_indices), rows * cols):
                axes[k // cols][k % cols].set_visible(False)
            if opts.period_label:
                fig.text(0.5, 0.995, opts.period_label, ha='center', va='top',
                         fontsize=opts.font_size)
        else:
            for j, spec in enumerate(specs):
                axes[0][j].set_title(spec.options.period_label, fontsize=opts.font_size + 1)
                for k, idx in enumerate(spec.main_indices):
                    self._annotate_subplot(axes[k][j], spec, idx, opts)
                for k in range(len(spec.main_indices), rows):
                    axes[k][j].set_visible(False)

        fig.supxlabel('Distance across fault (km)  [negative = left]', fontsize=opts.font_size)
        fig.supylabel(opts.unit, fontsize=opts.font_size)
        axis_half = specs[0].axis_half_km
        if axis_half is None:
            axis_half = profile_axis_half_km(specs[0].bundle.profile_length_km)
        for row in axes:
            for ax in row:
                if ax.get_visible():
                    ax.set_xlim(-axis_half, axis_half)
        if opts.title:
            fig.suptitle(opts.title, fontsize=opts.font_size + 2)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    def _annotate_subplot(self, ax, spec, idx, opts):
        main = self._draw_profile(ax, spec, idx)
        ax.axvline(0, color='#c0392b', linewidth=0.6, alpha=0.7)
        self._set_profile_xlim(ax, spec)
        label = f'{main.along_km:.1f} km' if main is not None else 'no data'
        ax.text(0.02, 0.82, label, transform=ax.transAxes, fontsize=opts.font_size - 1)
        ax.grid(alpha=0.3)

    def render_profiles_stacked(self, specs, out_path):
        plt = self._plt
        opts = specs[0].options
        ncols = len(specs)
        height = max(5, 0.8 * max(len(spec.main_indices) for spec in specs))
        fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, height),
                                 constrained_layout=True, squeeze=False)
        for j, spec in enumerate(specs):
            ax = axes[0][j]
            step = spec.stack_offset if spec.stack_offset else auto_stack_offset(spec.bundle)
            n_prof = len(spec.main_indices)
            for k, idx in enumerate(spec.main_indices):
                y_off = (n_prof - 1 - k) * step
                main = self._draw_profile(ax, spec, idx, y_offset=y_off)
                if main is not None:
                    ax.text(main.across_km[-1], y_off + np.nanmedian(main.value),
                            f' {main.along_km:.1f} km', fontsize=opts.font_size - 1,
                            va='center', color='#555555')
            ax.axvline(0, color='#c0392b', linewidth=0.8, alpha=0.7)
            self._set_profile_xlim(ax, spec)
            ax.set_xlabel('Distance across fault (km)  [negative = left]', fontsize=opts.font_size)
            ax.set_ylabel(f'{opts.unit} (profiles offset by {step:.2g})', fontsize=opts.font_size)
            ax.set_title(spec.options.period_label, fontsize=opts.font_size + 2)
            ax.grid(alpha=0.3)
        if opts.title:
            fig.suptitle(opts.title, fontsize=opts.font_size + 2)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    # ---------------------------------------------------------------- misc
    def show(self):
        if self._figures:
            self._plt.show()

    def close_all(self):
        for fig in self._figures:
            self._plt.close(fig)
        self._figures = []
