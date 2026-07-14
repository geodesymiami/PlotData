#!/usr/bin/env python3
"""Across-fault displacement timeseries at selected along-fault locations."""

from dataclasses import dataclass, field

import numpy as np

from plotdata.fault_transect.periods import (
    _eos_date_list, snap_end_date, snap_first_period_start, snap_gap_start)
from plotdata.fault_transect.sampling import grid_latlon_vectors, sample_side_box


@dataclass
class LocationTimeseries:
    point_index: int
    along_km: float
    lat: float
    lon: float
    offset: np.ndarray


@dataclass
class TimeseriesBundle:
    dates: list = field(default_factory=list)
    locations: list = field(default_factory=list)
    unit: str = 'cm'
    reference_side: str = 'left'


def interior_period_dates(periods):
    """Dates to mark as interior period boundaries (exclude first start, last end)."""
    if len(periods) < 2:
        return []
    dates = []
    for start, _end in periods[1:]:
        dates.append(start)
    for _start, end in periods[:-1]:
        dates.append(end)
    return sorted(set(dates))


def plotted_interior_period_dates(periods, plotted_dates):
    """Interior period boundaries snapped to acquisition dates shown in the plot."""
    if len(periods) < 2:
        return []
    date_list = sorted(str(d) for d in plotted_dates)
    dates = []
    for start, _end in periods[1:]:
        dates.append(snap_gap_start(date_list, start))
    for _start, end in periods[:-1]:
        dates.append(snap_end_date(date_list, end))
    return sorted(set(dates))


def snap_timeseries_span(eos_file, periods):
    """Return (start, end) snapped acquisition dates spanning all periods."""
    date_list = _eos_date_list(eos_file)
    start = snap_first_period_start(date_list, periods[0][0])
    end = snap_end_date(date_list, periods[-1][1])
    return start, end


def load_displacement_timeseries(eos_file, start_date, end_date, mask_thresh=0.55):
    """Load masked displacement cube (cm) and dates for ``start_date``..``end_date``."""
    from mintpy.objects import HDFEOS
    from mintpy.utils import readfile

    obj = HDFEOS(eos_file)
    obj.open(print_msg=False)
    try:
        data = np.asarray(obj.read(), dtype=float)
        all_dates = [str(d) for d in obj.get_date_list()]
    finally:
        obj.close()

    keep = [i for i, d in enumerate(all_dates)
            if int(start_date) <= int(d) <= int(end_date)]
    if not keep:
        raise ValueError(f'No acquisitions between {start_date} and {end_date} in {eos_file}')
    dates = [all_dates[i] for i in keep]
    data = data[keep, :, :]

    _, attr = readfile.read(
        eos_file, datasetName='HDFEOS/GRIDS/timeseries/observation/displacement')
    lats, lons = grid_latlon_vectors(attr)

    try:
        coh, _ = readfile.read(
            eos_file, datasetName='HDFEOS/GRIDS/timeseries/quality/temporalCoherence')
        if coh.shape == data.shape[1:]:
            bad = coh < mask_thresh
            for t in range(data.shape[0]):
                slice_ = data[t]
                slice_[bad] = np.nan
                data[t] = slice_
        else:
            print(f'WARNING: temporalCoherence shape {coh.shape} != grid shape; skipping mask')
    except Exception as exc:
        print(f'WARNING: could not read temporalCoherence for masking: {exc}')

    data[data == 0.0] = np.nan
    unit = attr.get('UNIT', 'm')
    if str(unit).startswith('m'):
        data = data * 100.0

    return data, dates, lats, lons


def compute_timeseries_offsets(data, dates, lats, lons, points, location_indices,
                               perp_width_km, along_step_km, sample_method='mean',
                               reference_side='left', perp_offset_km=0.5, unit='cm'):
    """Across-fault differential displacement timeseries at selected point indices."""
    half_along = along_step_km / 2.0
    bundle = TimeseriesBundle(dates=list(dates), unit=unit, reference_side=reference_side)

    for idx in location_indices:
        if idx < 0 or idx >= len(points):
            continue
        point = points[idx]
        offsets = []
        for t in range(len(dates)):
            left = sample_side_box(data[t], lats, lons, point, 'left', half_along,
                                   perp_width_km, sample_method,
                                   perp_offset_km=perp_offset_km)
            right = sample_side_box(data[t], lats, lons, point, 'right', half_along,
                                    perp_width_km, sample_method,
                                    perp_offset_km=perp_offset_km)
            if reference_side == 'left':
                off = left.value - right.value
            else:
                off = right.value - left.value
            offsets.append(off if np.isfinite(off) else np.nan)
        bundle.locations.append(LocationTimeseries(
            point_index=idx,
            along_km=point.along_km,
            lat=point.lat,
            lon=point.lon,
            offset=np.asarray(offsets, dtype=float)))
    return bundle


def auto_timeseries_stack_offset(bundle):
    """Vertical offset step for stacked timeseries curves."""
    ptps = [np.nanmax(loc.offset) - np.nanmin(loc.offset)
            for loc in bundle.locations if np.isfinite(loc.offset).any()]
    if not ptps:
        return 1.0
    return 1.2 * float(np.median(ptps))


def first_finite_value(values, default=0.0):
    """First finite sample in a 1D series (e.g. earliest acquisition)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else default


def stacked_timeseries_y(offsets, stack_index, stack_step):
    """Stacked timeseries values anchored at the first acquisition.

    Each curve is shifted to zero at the earliest date, then separated by a
    constant ``stack_index * stack_step`` so spacing is uniform on the left.
    """
    values = np.asarray(offsets, dtype=float)
    baseline = first_finite_value(values)
    return (values - baseline) + stack_index * stack_step


def stacked_profile_y(values, y_offset):
    """Center a profile on its median, then apply the stack offset."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    center = float(np.nanmedian(finite)) if finite.size else 0.0
    return (arr - center) + y_offset


def reference_side_center_latlon(point, reference_side, perp_offset_km, perp_width_km):
    """Geographic center of the reference-side search box at a sample point."""
    from plotdata.fault_transect.fault_sampling import offset_latlon

    ne, nn = point.left_normal
    sign = 1.0 if reference_side == 'left' else -1.0
    dist_km = perp_offset_km + 0.5 * perp_width_km
    return offset_latlon(point.lat, point.lon, sign * ne * dist_km, sign * nn * dist_km)
