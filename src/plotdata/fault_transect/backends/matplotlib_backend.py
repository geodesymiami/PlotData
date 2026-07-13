#!/usr/bin/env python3
"""Matplotlib implementation of PlotBackend. Only this file imports matplotlib."""

import numpy as np

from plotdata.fault_transect.backends.base import PlotBackend
from plotdata.fault_transect.plot_api import profile_axis_half_km
from plotdata.fault_transect.profiles import auto_stack_offset


class MatplotlibBackend(PlotBackend):

    def __init__(self):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._figures = []

    # ------------------------------------------------------------------ map
    def _value_norm_limits(self, series, value_lim):
        """Return (vmin, vmax) for symmetric offset coloring and curve y-axis."""
        if value_lim is not None:
            return value_lim
        offsets = np.asarray(series.offset, dtype=float)
        finite_offs = offsets[np.isfinite(offsets)]
        olim = float(np.nanmax(np.abs(finite_offs))) if len(finite_offs) else 1.0
        if olim == 0:
            olim = 1.0
        return -olim, olim

    def _fault_colored_segments(self, series, value_lim=None):
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
            segs.append([(lons[i], lats[i]), (lons[i + 1], lats[i + 1])])
            colors.append(color_val)

        if not segs:
            return None

        finite_offs = offsets[np.isfinite(offsets)]
        vmin, vmax = self._value_norm_limits(series, value_lim)
        lc = LineCollection(segs, cmap='RdBu_r', linewidths=10, capstyle='round',
                            norm=Normalize(vmin=vmin, vmax=vmax), zorder=3)
        lc.set_array(np.asarray(colors))
        return lc

    def _draw_map_panel(self, fig, ax_map, ax_curve, spec, draw_curve=True):
        opts = spec.options
        series = spec.offset_series

        all_lons = []
        all_lats = []
        for seg in spec.fault_segments:
            all_lons.extend(seg['lons'])
            all_lats.extend(seg['lats'])

        ax_map.set_facecolor('white')

        lc = self._fault_colored_segments(series, opts.value_lim)
        if lc is not None:
            ax_map.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax_map, shrink=0.75, pad=0.02)
            cbar.set_label(f'offset ({opts.unit})', fontsize=opts.font_size)

        pad = max(4 * spec.perp_width_km / 111.19, 0.02)
        ax_map.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
        ax_map.set_ylim(min(all_lats) - pad, max(all_lats) + pad)
        ax_map.set_aspect('equal', adjustable='box')
        ax_map.set_xlabel('Longitude', fontsize=opts.font_size)
        ax_map.set_ylabel('Latitude', fontsize=opts.font_size)
        if opts.title:
            ax_map.text(0.02, 0.98, opts.title, transform=ax_map.transAxes,
                        fontsize=opts.font_size + 1, va='top', ha='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.25))

        if draw_curve and ax_curve is not None:
            ymin, ymax = self._value_norm_limits(series, opts.value_lim)
            ax_curve.axhline(0, color='gray', linewidth=0.8)
            ax_curve.plot(series.along_km, series.offset, 'o-', color='#c0392b', markersize=4)
            ax_curve.set_ylim(ymin, ymax)
            ax_curve.set_xlabel('Distance along fault (km)', fontsize=opts.font_size)
            ax_curve.set_ylabel(f'{series.reference_side} - other ({opts.unit})', fontsize=opts.font_size)
            ax_curve.grid(alpha=0.3)

    def render_map_figure(self, map_specs, out_path):
        plt = self._plt
        from matplotlib import gridspec

        n_periods = len(map_specs)
        opts = map_specs[0].options
        if n_periods == 1:
            fig, axes = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True,
                                     gridspec_kw={'height_ratios': [2.2, 1]}, squeeze=False)
            self._draw_map_panel(fig, axes[0, 0], axes[1, 0], map_specs[0])
        else:
            fig = plt.figure(figsize=(9, 5 * n_periods), constrained_layout=True)
            gs = gridspec.GridSpec(n_periods, 1, figure=fig)
            for i, spec in enumerate(map_specs):
                ax_map = fig.add_subplot(gs[i])
                self._draw_map_panel(fig, ax_map, None, spec, draw_curve=False)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    # ------------------------------------------------------------- profiles
    def _apply_profile_value_ylim(self, ax, spec, stacked_extra=0.0):
        lim = spec.options.value_lim
        if lim is None:
            return
        ymin, ymax = lim
        ax.set_ylim(ymin, ymax + stacked_extra)

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
            self._apply_profile_value_ylim(ax, spec)
            ax.set_xlabel('Distance across fault (km)  [negative = left]', fontsize=opts.font_size)
            ax.set_ylabel(opts.unit, fontsize=opts.font_size)
            ax.set_title(f'{opts.title}  profile {nn:02d} at {main.along_km:.1f} km',
                         fontsize=opts.font_size + 1)
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
        else:
            for j, spec in enumerate(specs):
                axes[0][j].set_title(spec.options.title, fontsize=opts.font_size + 1)
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
        if len(specs) == 1:
            fig.suptitle(opts.title, fontsize=opts.font_size + 2)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    def _annotate_subplot(self, ax, spec, idx, opts):
        main = self._draw_profile(ax, spec, idx)
        ax.axvline(0, color='#c0392b', linewidth=0.6, alpha=0.7)
        self._set_profile_xlim(ax, spec)
        self._apply_profile_value_ylim(ax, spec)
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
            stacked_extra = (n_prof - 1) * step if spec.options.value_lim is not None else 0.0
            self._apply_profile_value_ylim(ax, spec, stacked_extra=stacked_extra)
            ax.set_xlabel('Distance across fault (km)  [negative = left]', fontsize=opts.font_size)
            ax.set_ylabel(f'{opts.unit} (profiles offset by {step:.2g})', fontsize=opts.font_size)
            ax.set_title(spec.options.title, fontsize=opts.font_size + 2)
            ax.grid(alpha=0.3)
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
