#!/usr/bin/env python3
"""Sample gridded data in along/across-fault search boxes on each fault side."""

from dataclasses import dataclass

import numpy as np

from plotdata.fault_transect.fault_sampling import local_east_north_km


@dataclass
class BoxSample:
    value: float         # combined value (nan if no valid pixel)
    count: int           # number of valid pixels used
    nearest_lat: float   # location of nearest valid pixel (for method='nearest')
    nearest_lon: float


def grid_latlon_vectors(attr):
    """Return (lats, lons) 1D vectors for a geocoded MintPy attribute dict."""
    lat0 = float(attr['Y_FIRST'])
    lon0 = float(attr['X_FIRST'])
    dlat = float(attr['Y_STEP'])
    dlon = float(attr['X_STEP'])
    length = int(attr['LENGTH'])
    width = int(attr['WIDTH'])
    lats = lat0 + dlat * np.arange(length)
    lons = lon0 + dlon * np.arange(width)
    return lats, lons


def sample_side_box(data, lats, lons, point, side, half_along_km, perp_width_km,
                    method='mean', perp_offset_km=0.0):
    """Combine pixels in the search box on one side of the fault at a sample point.

    The box extends +-half_along_km along the fault tangent and from perp_offset_km
    to perp_offset_km + perp_width_km outward on the requested side ('left' or
    'right', where left is the counterclockwise side of the tangent in map view).
    """
    lat_c, lon_c = point.lat, point.lon
    te, tn = point.tangent
    ne, nn = point.left_normal
    sign = 1.0 if side == 'left' else -1.0

    # subwindow around the point to keep the search cheap
    margin_km = half_along_km + perp_offset_km + perp_width_km + 0.5
    km_per_deg = 111.19
    dlat = margin_km / km_per_deg
    dlon = margin_km / (km_per_deg * max(np.cos(np.radians(lat_c)), 0.01))
    row_sel = np.where((lats >= lat_c - dlat) & (lats <= lat_c + dlat))[0]
    col_sel = np.where((lons >= lon_c - dlon) & (lons <= lon_c + dlon))[0]
    if row_sel.size == 0 or col_sel.size == 0:
        return BoxSample(np.nan, 0, np.nan, np.nan)

    sub = data[np.ix_(row_sel, col_sel)]
    sub_lats = lats[row_sel]
    sub_lons = lons[col_sel]
    lon_grid, lat_grid = np.meshgrid(sub_lons, sub_lats)

    east, north = local_east_north_km(lat_c, lon_c, lat_grid.ravel(), lon_grid.ravel())
    along = east * te + north * tn
    across = (east * ne + north * nn) * sign     # >0 on requested side

    values = sub.ravel()
    in_box = ((np.abs(along) <= half_along_km)
              & (across > perp_offset_km)
              & (across <= perp_offset_km + perp_width_km))
    valid = in_box & np.isfinite(values)
    if not np.any(valid):
        return BoxSample(np.nan, 0, np.nan, np.nan)

    vals = values[valid]
    if method == 'mean':
        combined = float(np.nanmean(vals))
    elif method == 'median':
        combined = float(np.nanmedian(vals))
    elif method == 'nearest':
        dist2 = east[valid] ** 2 + north[valid] ** 2
        combined = float(vals[int(np.argmin(dist2))])
    else:
        raise ValueError(f'Unknown --sample-method: {method}')

    dist2 = east[valid] ** 2 + north[valid] ** 2
    nearest_i = int(np.argmin(dist2))
    return BoxSample(value=combined, count=int(vals.size),
                     nearest_lat=float(lat_grid.ravel()[valid][nearest_i]),
                     nearest_lon=float(lon_grid.ravel()[valid][nearest_i]))
