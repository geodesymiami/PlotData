#!/usr/bin/env python3
"""Matplotlib implementation of PlotBackend. Only this file imports matplotlib."""

import numpy as np

from plotdata.fault_transect.backends.base import PlotBackend
from plotdata.fault_transect.profiles import auto_stack_offset


class MatplotlibBackend(PlotBackend):

    def __init__(self):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._figures = []

    # ------------------------------------------------------------------ map
    def _draw_map_panel(self, fig, ax_map, ax_curve, spec):
        opts = spec.options
        series = spec.offset_series

        vmin, vmax = (opts.vlim if opts.vlim else (None, None))
        extent = [spec.lons[0], spec.lons[-1], spec.lats[-1], spec.lats[0]]
        im = ax_map.imshow(spec.data, extent=extent, origin='upper',
                           cmap=opts.colormap, vmin=vmin, vmax=vmax, aspect='auto')
        cbar = fig.colorbar(im, ax=ax_map, shrink=0.75, pad=0.02)
        cbar.set_label(opts.unit, fontsize=opts.font_size)

        all_lons = []
        all_lats = []
        for seg in spec.fault_segments:
            ax_map.plot(seg['lons'], seg['lats'], 'k-', linewidth=1.5)
            all_lons.extend(seg['lons'])
            all_lats.extend(seg['lats'])

        offsets = np.asarray(series.offset, dtype=float)
        finite = np.isfinite(offsets)
        if np.any(finite):
            olim = np.nanmax(np.abs(offsets[finite]))
            sc = ax_map.scatter(np.asarray(series.lon)[finite], np.asarray(series.lat)[finite],
                                c=offsets[finite], cmap='RdBu_r', vmin=-olim, vmax=olim,
                                s=45, edgecolors='black', linewidths=0.4, zorder=5)
            cbar2 = fig.colorbar(sc, ax=ax_map, shrink=0.75, pad=0.08)
            cbar2.set_label(f'offset ({opts.unit})', fontsize=opts.font_size)

        pad = max(4 * spec.perp_width_km / 111.19, 0.02)
        ax_map.set_xlim(min(all_lons) - pad, max(all_lons) + pad)
        ax_map.set_ylim(min(all_lats) - pad, max(all_lats) + pad)
        ax_map.set_xlabel('Longitude', fontsize=opts.font_size)
        ax_map.set_ylabel('Latitude', fontsize=opts.font_size)
        ax_map.set_title(opts.title, fontsize=opts.font_size + 2)

        ax_curve.axhline(0, color='gray', linewidth=0.8)
        ax_curve.plot(series.along_km, series.offset, 'o-', color='#c0392b', markersize=4)
        ax_curve.set_xlabel('Distance along fault (km)', fontsize=opts.font_size)
        ax_curve.set_ylabel(f'{series.reference_side} - other ({opts.unit})', fontsize=opts.font_size)
        ax_curve.grid(alpha=0.3)

    def render_map_figure(self, map_specs, out_path):
        plt = self._plt
        ncols = len(map_specs)
        opts = map_specs[0].options
        fig, axes = plt.subplots(2, ncols, figsize=(9 * ncols, 10), constrained_layout=True,
                                 gridspec_kw={'height_ratios': [2.2, 1]}, squeeze=False)
        for j, spec in enumerate(map_specs):
            self._draw_map_panel(fig, axes[0][j], axes[1][j], spec)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    # ------------------------------------------------------------- profiles
    def _draw_profile(self, ax, spec, main_index, y_offset=0.0):
        """Draw one main profile (black) plus its gray cloud on an axis."""
        bundle = spec.bundle
        for delta in range(-spec.cloud_profiles, spec.cloud_profiles + 1):
            if delta == 0:
                continue
            neighbor = bundle.get(main_index + delta)
            if neighbor is None:
                continue
            ax.plot(neighbor.across_km, neighbor.value + y_offset,
                    color='gray', linewidth=0.4, alpha=0.6, zorder=1)
        main = bundle.get(main_index)
        if main is not None:
            ax.plot(main.across_km, main.value + y_offset,
                    color='black', linewidth=1.2, zorder=3)
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
        if len(specs) == 1:
            fig.suptitle(opts.title, fontsize=opts.font_size + 2)
        fig.savefig(out_path, dpi=opts.dpi, bbox_inches='tight')
        self._figures.append(fig)
        return out_path

    def _annotate_subplot(self, ax, spec, idx, opts):
        main = self._draw_profile(ax, spec, idx)
        ax.axvline(0, color='#c0392b', linewidth=0.6, alpha=0.7)
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
            for k, idx in enumerate(spec.main_indices):
                main = self._draw_profile(ax, spec, idx, y_offset=k * step)
                if main is not None:
                    ax.text(main.across_km[-1], k * step + np.nanmedian(main.value),
                            f' {main.along_km:.1f} km', fontsize=opts.font_size - 1,
                            va='center', color='#555555')
            ax.axvline(0, color='#c0392b', linewidth=0.8, alpha=0.7)
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
