#!/usr/bin/env python3
"""Backend-agnostic figure specifications for fault-transect plots."""

from dataclasses import dataclass, field


@dataclass
class PlotOptions:
    colormap: str = 'jet'
    cmap_vlist: tuple = None       # (vmin, vmid, vmax) for _truncate maps
    vlim: tuple = None            # (vmin, vmax) for offset map colorscale; None -> auto
    font_size: int = 10
    dpi: int = 300
    unit: str = 'cm/yr'
    title: str = ''                   # fault / tag name only
    dataset_label: str = ''           # ascending, descending, horizontal, or vertical
    period_label: str = ''            # formatted period range (no tag)
    title_position: str = 'upper-right'
    period_label_position: str = 'lower-left'
    title_offset_lon: float = 0.0
    title_offset_lat: float = 0.0
    map_stack_axis: str = 'lat'       # 'lat' or 'lon': axis for stacked multi-period maps
    map_stack_offset: float = None    # manual step (deg) along map_stack_axis
    scatter_size: float = 2.0         # marker diameter in points (profile/timeseries dots)


def title_coords(position, lon_min, lon_max, lat_min, lat_max, lat_offset=0.0,
                 lon_offset=0.0, margin_frac=0.02, title_offset_lon=0.0, title_offset_lat=0.0):
    """Return (x, y, ha, va) for a map title in geographic coordinates."""
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lat_base_min = lat_min + lat_offset
    lat_base_max = lat_max + lat_offset
    lon_base_min = lon_min + lon_offset
    lon_base_max = lon_max + lon_offset
    positions = {
        'upper-left': (lon_base_min + margin_frac * lon_span, lat_base_max - margin_frac * lat_span,
                       'left', 'top'),
        'upper-right': (lon_base_max - margin_frac * lon_span, lat_base_max - margin_frac * lat_span,
                        'right', 'top'),
        'lower-left': (lon_base_min + margin_frac * lon_span, lat_base_min + margin_frac * lat_span,
                       'left', 'bottom'),
        'lower-right': (lon_base_max - margin_frac * lon_span, lat_base_min + margin_frac * lat_span,
                        'right', 'bottom'),
    }
    try:
        x, y, ha, va = positions[position]
    except KeyError as exc:
        valid = ', '.join(sorted(positions))
        raise ValueError(f'title position must be one of: {valid}') from exc
    return x + title_offset_lon, y + title_offset_lat, ha, va


def map_view_lat_pad(perp_width_km, lat_min, lat_max):
    """Padding around fault lat extent for map axes (matches single-period maps)."""
    lat_span = lat_max - lat_min
    return max(4 * perp_width_km / 111.19, 0.02)


def map_view_lon_pad(perp_width_km, lat_center, lon_min, lon_max):
    """Padding around fault lon extent for map axes (mid-latitude km→deg)."""
    import math
    lon_span = lon_max - lon_min
    km_per_deg = 111.19 * max(abs(math.cos(math.radians(lat_center))), 0.01)
    return max(4 * perp_width_km / km_per_deg, 0.02)


def compute_map_lat_stack_step(perp_width_km, lat_min, lat_max, manual=None):
    """Latitude step (deg) between stacked multi-period map panels."""
    if manual is not None:
        return float(manual)
    pad = map_view_lat_pad(perp_width_km, lat_min, lat_max)
    lat_span = lat_max - lat_min
    return lat_span + max(pad, 0.15 * lat_span, 0.02)


def compute_map_lon_stack_step(perp_width_km, lat_center, lon_min, lon_max, manual=None):
    """Longitude step (deg) between stacked multi-period map panels."""
    if manual is not None:
        return float(manual)
    pad = map_view_lon_pad(perp_width_km, lat_center, lon_min, lon_max)
    lon_span = lon_max - lon_min
    return lon_span + max(pad, 0.15 * lon_span, 0.02)


def compute_map_stack_step(axis, perp_width_km, lon_min, lon_max, lat_min, lat_max, manual=None):
    """Stack step in degrees along ``axis`` ('lat' or 'lon')."""
    if axis == 'lon':
        lat_center = 0.5 * (lat_min + lat_max)
        return compute_map_lon_stack_step(perp_width_km, lat_center, lon_min, lon_max, manual)
    return compute_map_lat_stack_step(perp_width_km, lat_min, lat_max, manual)


def print_offset_summaries(summaries):
    """Print stack-offset summary collected during plotting."""
    if not summaries:
        return
    print('Stack / layout offsets used:')
    for entry in summaries:
        kind = entry['kind']
        if kind == 'map_stack':
            src = entry['source']
            axis = entry.get('axis', 'lat')
            axis_label = 'latitude' if axis == 'lat' else 'longitude'
            print(f"  {entry['project']} multi-period map ({entry['n_periods']} periods): "
                  f"{axis_label} step {entry['step_deg']:.4f} deg ({src})")
        elif kind == 'profile_stack':
            print(f"  {entry['project']} stacked profiles: vertical step "
                  f"{entry['stack_step']:.4g} {entry['unit']} ({entry['source']})")
        elif kind == 'title_offset':
            print(f"  map title nudge: {entry['lon_deg']:+.4f} deg lon, "
                  f"{entry['lat_deg']:+.4f} deg lat")


def stacked_curve_y_offset(index, count, step):
    """Vertical offset for stacked profile/timeseries curves.

    The first curve (index 0, start along fault) sits at the baseline; each
    subsequent curve is shifted upward by ``step``. Timeseries values are
    anchored at the first acquisition via ``stacked_timeseries_y`` so spacing
    is uniform at early dates.
    """
    del count
    return index * step


def single_period_lat_ticks(lat_min, lat_max, pad, nbins=5):
    """Latitude tick values for one map panel (true coordinates, no stack offset)."""
    y0, y1 = lat_min - pad, lat_max + pad
    try:
        from matplotlib.ticker import MaxNLocator
        values = MaxNLocator(nbins=nbins, min_n_ticks=3).tick_values(y0, y1)
    except ImportError:
        step = (y1 - y0) / max(nbins - 1, 1)
        values = [y0 + i * step for i in range(nbins)]
    return [float(v) for v in values if y0 <= v <= y1]


def stacked_map_lat_offset(period_index, step):
    """Shift south for period index > 0 when stacking along latitude."""
    return -period_index * step


def stacked_map_lon_offset(period_index, step):
    """Shift east for period index > 0 when stacking along longitude (period 1 left)."""
    return period_index * step


def stacked_map_axis_offset(period_index, step, axis):
    """Return (lat_offset, lon_offset) for stacked multi-period maps."""
    if axis == 'lon':
        return 0.0, stacked_map_lon_offset(period_index, step)
    return stacked_map_lat_offset(period_index, step), 0.0


def stacked_map_ytick_pairs(lat_min, lat_max, pad, lat_step, n_periods, nbins=5):
    """First (top) period only: same tick positions and labels as a single-period map."""
    del lat_step, n_periods
    true_ticks = single_period_lat_ticks(lat_min, lat_max, pad, nbins=nbins)
    return [(true_lat, true_lat) for true_lat in true_ticks]


def single_period_lon_ticks(lon_min, lon_max, pad, nbins=5):
    """Longitude tick values for one map panel (true coordinates, no stack offset)."""
    x0, x1 = lon_min - pad, lon_max + pad
    try:
        from matplotlib.ticker import MaxNLocator
        values = MaxNLocator(nbins=nbins, min_n_ticks=3).tick_values(x0, x1)
    except ImportError:
        step = (x1 - x0) / max(nbins - 1, 1)
        values = [x0 + i * step for i in range(nbins)]
    return [float(v) for v in values if x0 <= v <= x1]


def stacked_map_xtick_pairs(lon_min, lon_max, pad, lon_step, n_periods, nbins=5):
    """First (left) period only: true longitude labels at axis positions."""
    del lon_step, n_periods
    true_ticks = single_period_lon_ticks(lon_min, lon_max, pad, nbins=nbins)
    return [(true_lon, true_lon) for true_lon in true_ticks]


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


@dataclass
class TimeseriesFigureSpec:
    """Data for map + stacked across-fault displacement timeseries figure."""
    bundle: object                # timeseries.TimeseriesBundle
    period_boundary_dates: list = field(default_factory=list)
    stack_offset: float = None
    fault_segments: list = field(default_factory=list)
    offset_series: object = None    # offset.OffsetSeries for full-span map coloring
    sample_points: list = field(default_factory=list)
    perp_width_km: float = 0.5
    perp_offset_km: float = 0.5
    reference_side: str = 'left'
    options: PlotOptions = field(default_factory=PlotOptions)
