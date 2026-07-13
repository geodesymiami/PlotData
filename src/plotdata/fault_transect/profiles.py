#!/usr/bin/env python3
"""Fault-perpendicular profiles extracted with MintPy transect_lalo."""

from dataclasses import dataclass, field

import numpy as np

from plotdata.fault_transect.fault_sampling import offset_latlon


@dataclass
class Profile:
    index: int
    along_km: float          # position of the profile along the fault
    center_lat: float
    center_lon: float
    across_km: np.ndarray    # signed distance across fault (negative = left side)
    lat: np.ndarray
    lon: np.ndarray
    value: np.ndarray


@dataclass
class ProfileBundle:
    profiles: list = field(default_factory=list)
    profile_length_km: float = 0.0
    unit: str = ''

    def get(self, index):
        for p in self.profiles:
            if p.index == index:
                return p
        return None


def extract_profiles(data, attr, points, profile_length_km, interpolation='nearest', unit=''):
    """Extract one perpendicular profile at every sampling point.

    Each profile is a single line of data points from the LEFT extreme through
    the fault (across_km = 0) to the RIGHT extreme, so across_km is negative on
    the left side of the fault and positive on the right side.
    """
    from mintpy.utils import utils as ut

    half = profile_length_km / 2.0
    bundle = ProfileBundle(profile_length_km=profile_length_km, unit=unit)

    for i, point in enumerate(points):
        ne, nn = point.left_normal
        start_lat, start_lon = offset_latlon(point.lat, point.lon, ne * half, nn * half)
        end_lat, end_lon = offset_latlon(point.lat, point.lon, -ne * half, -nn * half)

        try:
            txn = ut.transect_lalo(data, attr, [start_lat, start_lon], [end_lat, end_lon],
                                   interpolation=interpolation)
        except ValueError:
            continue    # profile endpoint outside data coverage
        if txn['value'].size == 0:
            continue

        coord_lats, coord_lons = _pixel_latlon(attr, txn['Y'], txn['X'])
        across = txn['distance'] / 1000.0 - half
        bundle.profiles.append(Profile(
            index=i,
            along_km=point.along_km,
            center_lat=point.lat,
            center_lon=point.lon,
            across_km=np.asarray(across, dtype=float),
            lat=np.asarray(coord_lats, dtype=float),
            lon=np.asarray(coord_lons, dtype=float),
            value=np.asarray(txn['value'], dtype=float),
        ))

    return bundle


def _pixel_latlon(attr, rows, cols):
    lat0 = float(attr['Y_FIRST'])
    lon0 = float(attr['X_FIRST'])
    dlat = float(attr['Y_STEP'])
    dlon = float(attr['X_STEP'])
    lats = lat0 + dlat * np.asarray(rows, dtype=float)
    lons = lon0 + dlon * np.asarray(cols, dtype=float)
    return lats, lons


def auto_stack_offset(bundle):
    """Vertical offset for the stacked layout: 1.2x the median peak-to-peak."""
    ptps = [np.nanmax(p.value) - np.nanmin(p.value)
            for p in bundle.profiles if np.isfinite(p.value).any()]
    if not ptps:
        return 1.0
    return 1.2 * float(np.median(ptps))
