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


def _tangent_from_delta(lon0, lat0, lon1, lat1):
    km_lat, km_lon = _local_scale((lat0 + lat1) / 2.0)
    d_east = (lon1 - lon0) * km_lon
    d_north = (lat1 - lat0) * km_lat
    norm = math.hypot(d_east, d_north)
    if norm == 0:
        return None
    tangent = (d_east / norm, d_north / norm)
    left_normal = (-tangent[1], tangent[0])
    return tangent, left_normal


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
    if along_end_km is not None and abs(along_end_km - along_start_km) < 1e-9:
        if along_start_km > total + 1e-9:
            raise ValueError(f'--along-start ({along_start_km} km) exceeds fault length ({total:.3f} km)')
        targets = np.array([along_start_km])
    else:
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

        tb = _tangent_from_delta(lons[seg], lats[seg], lons[seg + 1], lats[seg + 1])
        if tb is None:
            continue
        tangent, left_normal = tb
        points.append(SamplePoint(along_km=float(target), lat=float(lat), lon=float(lon),
                                  tangent=tangent, left_normal=left_normal))
    return points


def _segment_gaps_km(segments, gaps_km=None):
    """Return list of gap lengths between consecutive segments (len n-1)."""
    if gaps_km is not None:
        return list(gaps_km)
    gaps = []
    for i in range(1, len(segments)):
        end_lon, end_lat = segments[i - 1][-1]
        start_lon, start_lat = segments[i][0]
        gaps.append(haversine_km(end_lat, end_lon, start_lat, start_lon))
    return gaps


def sample_points_segments(segments, along_step_km, along_start_km=0.0, along_end_km=None,
                           gaps_km=None):
    """Sample points along an ordered list of disconnected segments.

    Along-strike distance runs along each segment, then includes the straight
    gap to the next segment start, then continues on the next segment.

    segments: list of (lon, lat) coordinate lists.
    gaps_km: optional precomputed gaps between segments (from QC); computed from
             endpoints when omitted.
    """
    if not segments:
        return []
    seg_lengths = [float(cumulative_distance_km(seg)[-1]) for seg in segments]
    gaps = _segment_gaps_km(segments, gaps_km=gaps_km)

    seg_starts = [0.0]
    for i in range(1, len(segments)):
        seg_starts.append(seg_starts[-1] + seg_lengths[i - 1] + gaps[i - 1])
    total = seg_starts[-1] + seg_lengths[-1]
    end = min(along_end_km, total) if along_end_km is not None else total
    if along_start_km >= end:
        raise ValueError(f'--along-start ({along_start_km} km) must be smaller than '
                         f'--along-end / fault length ({end:.3f} km)')
    targets = np.arange(along_start_km, end + 1e-9, along_step_km)

    points = []
    for target in targets:
        seg_idx = int(np.searchsorted(seg_starts, target, side='right') - 1)
        seg_idx = min(max(seg_idx, 0), len(segments) - 1)
        seg_start = seg_starts[seg_idx]
        seg_end = seg_start + seg_lengths[seg_idx]

        if target <= seg_end + 1e-9:
            local_target = target - seg_start
            seg_points = sample_points(segments[seg_idx], along_step_km=along_step_km,
                                       along_start_km=local_target, along_end_km=local_target)
            if seg_points:
                p = seg_points[0]
                p.along_km = float(target)
                points.append(p)
            continue

        if seg_idx >= len(segments) - 1:
            continue

        gap_start = seg_end
        gap_end = seg_starts[seg_idx + 1]
        gap_len = gap_end - gap_start
        if gap_len <= 0:
            continue
        frac = (target - gap_start) / gap_len
        end_lon, end_lat = segments[seg_idx][-1]
        start_lon, start_lat = segments[seg_idx + 1][0]
        lat = end_lat + frac * (start_lat - end_lat)
        lon = end_lon + frac * (start_lon - end_lon)
        tb = _tangent_from_delta(end_lon, end_lat, start_lon, start_lat)
        if tb is None:
            continue
        tangent, left_normal = tb
        points.append(SamplePoint(along_km=float(target), lat=float(lat), lon=float(lon),
                                  tangent=tangent, left_normal=left_normal))
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
