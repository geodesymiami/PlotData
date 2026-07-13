#!/usr/bin/env python3
"""Across-fault displacement offset along the fault (map plot quantity)."""

from dataclasses import dataclass, field

import numpy as np

from plotdata.fault_transect.sampling import sample_side_box


@dataclass
class OffsetSeries:
    along_km: list = field(default_factory=list)
    lat: list = field(default_factory=list)
    lon: list = field(default_factory=list)
    left_val: list = field(default_factory=list)
    right_val: list = field(default_factory=list)
    offset: list = field(default_factory=list)     # reference side minus other side
    reference_side: str = 'left'
    unit: str = ''


def compute_offset_series(data, lats, lons, points, perp_width_km, along_step_km,
                          sample_method='mean', reference_side='left', unit='',
                          perp_offset_km=0.5):
    """Compute the across-fault offset at every sampling point.

    offset = value(reference side) - value(other side); NaN when either side
    has no valid pixel in its search box.
    """
    series = OffsetSeries(reference_side=reference_side, unit=unit)
    half_along = along_step_km / 2.0

    for point in points:
        left = sample_side_box(data, lats, lons, point, 'left', half_along, perp_width_km,
                               sample_method, perp_offset_km=perp_offset_km)
        right = sample_side_box(data, lats, lons, point, 'right', half_along, perp_width_km,
                                sample_method, perp_offset_km=perp_offset_km)

        if reference_side == 'left':
            offset = left.value - right.value
        else:
            offset = right.value - left.value

        series.along_km.append(point.along_km)
        series.lat.append(point.lat)
        series.lon.append(point.lon)
        series.left_val.append(left.value)
        series.right_val.append(right.value)
        series.offset.append(offset if np.isfinite(offset) else np.nan)

    return series
