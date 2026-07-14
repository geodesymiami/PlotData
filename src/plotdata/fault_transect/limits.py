#!/usr/bin/env python3
"""Shared offset colorscale limits for map fault coloring."""

import numpy as np


def _finite_values(values):
    return [float(v) for v in values if np.isfinite(v)]


def global_symmetric_limit(offset_series_list):
    """Largest absolute finite offset value across map offset series."""
    vals = []
    for series in offset_series_list:
        vals.extend(_finite_values(series.offset))
    if not vals:
        return 1.0
    olim = float(np.max(np.abs(vals)))
    return olim if olim > 0 else 1.0


def build_offset_color_limits(num_periods, vlim, auto_colorscale, offset_series_list):
    """Return per-period (vmin, vmax) for the map offset colorscale, or None for auto."""
    if vlim is not None:
        lim = tuple(vlim)
        return [lim] * num_periods

    if auto_colorscale:
        olim = global_symmetric_limit(offset_series_list)
        lim = (-olim, olim)
        return [lim] * num_periods

    return [None] * num_periods


def shared_map_color_limits(period_limits):
    """Merge explicit per-period (vmin, vmax) for one multi-period map colorbar.

    Returns (vmin, vmax) when any period limit is set, else None (caller uses auto).
    """
    defined = [lim for lim in period_limits if lim is not None]
    if not defined:
        return None
    return min(lim[0] for lim in defined), max(lim[1] for lim in defined)
