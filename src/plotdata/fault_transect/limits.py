#!/usr/bin/env python3
"""Shared value limits for map offset colors and profile y-axes."""

import numpy as np


def _finite_values(values):
    return [float(v) for v in values if np.isfinite(v)]


def global_symmetric_limit(offset_series_list, profile_bundles=None, main_indices_per_period=None):
    """Largest absolute finite value across offset series and profile samples."""
    vals = []
    for series in offset_series_list:
        vals.extend(_finite_values(series.offset))
    if profile_bundles and main_indices_per_period:
        for bundle, main_indices in zip(profile_bundles, main_indices_per_period):
            for idx in main_indices:
                prof = bundle.get(idx)
                if prof is not None:
                    vals.extend(_finite_values(prof.value))
    if not vals:
        return 1.0
    olim = float(np.max(np.abs(vals)))
    return olim if olim > 0 else 1.0


def build_value_limits(num_periods, ylim_pairs, auto_colorscale,
                       offset_series_list, profile_bundles=None,
                       main_indices_per_period=None):
    """Return a list of (vmin, vmax) per period, or None entries for per-period auto."""
    if ylim_pairs:
        return list(ylim_pairs)

    if auto_colorscale:
        olim = global_symmetric_limit(offset_series_list, profile_bundles,
                                      main_indices_per_period)
        lim = (-olim, olim)
        return [lim] * num_periods

    return [None] * num_periods
