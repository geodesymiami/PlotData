#!/usr/bin/env python3
"""Backend-agnostic rendering interface for fault-transect figures.

All plotting data is passed via the dataclasses in plot_api.py; backends only
render and save. Replacing matplotlib means implementing PlotBackend once.

Methods that accept a list of specs render one column per spec (used for
side-by-side comparison of multiple periods); a single-element list produces
a normal single-period figure.
"""

from abc import ABC, abstractmethod


class PlotBackend(ABC):

    @abstractmethod
    def render_map_figure(self, map_specs, out_path):
        """Map figure: per spec one column of (map panel, offset-curve panel).

        map_specs: list of plot_api.MapFigureSpec (one per period column).
        """

    @abstractmethod
    def render_profiles_separate(self, spec, out_path_template):
        """One figure per main profile (with optional gray cloud), one period.

        out_path_template contains '{nn}' replaced by the profile number.
        Returns list of written paths.
        """

    @abstractmethod
    def render_profiles_subplot(self, specs, out_path):
        """Profile grid; one column per spec (period), one row per profile."""

    @abstractmethod
    def render_profiles_stacked(self, specs, out_path):
        """Vertically offset profiles; one axis per spec (period)."""

    @abstractmethod
    def render_timeseries_stacked(self, spec, out_path):
        """Stacked across-fault displacement timeseries vs acquisition date."""

    @abstractmethod
    def show(self):
        """Open interactive windows for figures rendered so far (optional)."""

    @abstractmethod
    def close_all(self):
        """Release figure resources."""
