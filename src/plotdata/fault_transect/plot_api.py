#!/usr/bin/env python3
"""Backend-agnostic figure specifications for fault-transect plots."""

from dataclasses import dataclass, field


@dataclass
class PlotOptions:
    colormap: str = 'viridis'
    vlim: tuple = None            # (vmin, vmax) or None
    value_lim: tuple = None       # (vmin, vmax) for offset map colors and profile y-axis
    font_size: int = 10
    dpi: int = 300
    unit: str = 'cm/yr'
    title: str = ''


@dataclass
class MapFigureSpec:
    """Data for the map plot: background grid, fault trace, offset series."""
    data: object                  # 2D array (NaN-masked velocity)
    lats: object                  # 1D array
    lons: object                  # 1D array
    fault_segments: object        # list of segments: each {'lons': [...], 'lats': [...]}
    offset_series: object         # offset.OffsetSeries
    perp_width_km: float = 0.5
    options: PlotOptions = field(default_factory=PlotOptions)


def profile_axis_half_km(profile_length_km, padding_frac=0.1):
    """Half-width of the profile plot x-axis in km.

    Span is the profile length plus ``padding_frac`` (default 10%) total margin.
    """
    return (profile_length_km / 2.0) * (1.0 + padding_frac)


@dataclass
class ProfileFigureSpec:
    """Data for profile figures in any layout."""
    bundle: object                # profiles.ProfileBundle
    main_indices: list = field(default_factory=list)   # indices of main profiles
    cloud_profiles: int = 0       # adjacent profiles per side, gray thin lines
    stack_offset: float = None    # None -> auto
    subplot_cols: int = 1
    connect_lines: bool = False   # if True, connect profile samples with lines
    axis_half_km: float = None      # None -> derived from bundle.profile_length_km
    options: PlotOptions = field(default_factory=PlotOptions)
