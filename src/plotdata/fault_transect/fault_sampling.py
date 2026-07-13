#!/usr/bin/env python3
"""Sampling points along a fault polyline: positions, tangents, left normals."""

import math
from dataclasses import dataclass

import numpy as np

from plotdata.fault_transect.kmz_fault import haversine_km

EARTH_RADIUS_KM = 6371.0


@dataclass
class SamplePoint:
    along_km: float
    lat: float
    lon: float
    tangent: tuple      # (east, north) unit vector of along-fault direction
    left_normal: tuple  # (east, north) unit vector pointing left of the fault


def _local_scale(lat0):
    """km per degree of (lat, lon) around latitude lat0 (equirectangular)."""
    km_per_deg_lat = math.pi / 180.0 * EARTH_RADIUS_KM
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(lat0))
    return km_per_deg_lat, km_per_deg_lon


def cumulative_distance_km(coords):
    """Cumulative along-polyline distance for (lon, lat) coords."""
    dist = [0.0]
    for (lon0, lat0), (lon1, lat1) in zip(coords[:-1], coords[1:]):
        dist.append(dist[-1] + haversine_km(lat0, lon0, lat1, lon1))
    return np.asarray(dist)


def sample_points(coords, along_step_km, along_start_km=0.0, along_end_km=None):
    """Generate SamplePoints every along_step_km along the (lon, lat) polyline.

    The tangent points in the direction of increasing along-distance; the left
    normal is the tangent rotated 90 degrees counterclockwise in map view
    (east/north), i.e. the left-hand side when walking along the fault.
    """
    coords = list(coords)
    if len(coords) < 2:
        raise ValueError('Fault polyline needs at least 2 vertices')

    cum = cumulative_distance_km(coords)
    total = float(cum[-1])
    end = min(along_end_km, total) if along_end_km is not None else total
    if along_start_km >= end:
        raise ValueError(f'--along-start ({along_start_km} km) must be smaller than '
                         f'--along-end / fault length ({end:.3f} km)')

    targets = np.arange(along_start_km, end + 1e-9, along_step_km)
    lons = np.asarray([c[0] for c in coords])
    lats = np.asarray([c[1] for c in coords])

    points = []
    for target in targets:
        seg = int(np.searchsorted(cum, target, side='right') - 1)
        seg = min(max(seg, 0), len(coords) - 2)
        seg_len = cum[seg + 1] - cum[seg]
        frac = 0.0 if seg_len == 0 else (target - cum[seg]) / seg_len
        lat = lats[seg] + frac * (lats[seg + 1] - lats[seg])
        lon = lons[seg] + frac * (lons[seg + 1] - lons[seg])

        km_lat, km_lon = _local_scale(lat)
        d_east = (lons[seg + 1] - lons[seg]) * km_lon
        d_north = (lats[seg + 1] - lats[seg]) * km_lat
        norm = math.hypot(d_east, d_north)
        if norm == 0:
            continue
        tangent = (d_east / norm, d_north / norm)
        left_normal = (-tangent[1], tangent[0])
        points.append(SamplePoint(along_km=float(target), lat=float(lat), lon=float(lon),
                                  tangent=tangent, left_normal=left_normal))
    return points


def sample_points_segments(segments, along_step_km, along_start_km=0.0, along_end_km=None):
    """Sample points along an ordered list of disconnected segments.

    Along-strike distance is computed by concatenating segment lengths in order
    (no gap penalty). The sampling points restart at the beginning of each
    segment, but their returned `along_km` is global cumulative distance.

    segments: list of (lon, lat) coordinate lists.
    """
    if not segments:
        return []
    # compute segment cumulative lengths (segments already include any <=300m connectors)
    seg_lengths = [float(cumulative_distance_km(seg)[-1]) for seg in segments]
    cum0 = [0.0]
    for L in seg_lengths:
        cum0.append(cum0[-1] + L)
    total = cum0[-1]
    end = min(along_end_km, total) if along_end_km is not None else total
    if along_start_km >= end:
        raise ValueError(f'--along-start ({along_start_km} km) must be smaller than '
                         f'--along-end / fault length ({end:.3f} km)')
    targets = np.arange(along_start_km, end + 1e-9, along_step_km)

    points = []
    for target in targets:
        seg_idx = int(np.searchsorted(cum0, target, side='right') - 1)
        seg_idx = min(max(seg_idx, 0), len(segments) - 1)
        local_target = target - cum0[seg_idx]
        seg_points = sample_points(segments[seg_idx], along_step_km=along_step_km,
                                   along_start_km=local_target, along_end_km=local_target)
        if seg_points:
            p = seg_points[0]
            p.along_km = float(target)
            points.append(p)
    return points


def offset_latlon(lat, lon, east_km, north_km):
    """Shift a (lat, lon) point by east/north km offsets."""
    km_lat, km_lon = _local_scale(lat)
    return lat + north_km / km_lat, lon + east_km / km_lon


def local_east_north_km(lat0, lon0, lats, lons):
    """East/north km offsets of (lats, lons) arrays relative to (lat0, lon0)."""
    km_lat, km_lon = _local_scale(lat0)
    east = (np.asarray(lons) - lon0) * km_lon
    north = (np.asarray(lats) - lat0) * km_lat
    return east, north
