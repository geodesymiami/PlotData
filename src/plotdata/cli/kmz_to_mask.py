#!/usr/bin/env python3
"""
kmz_to_mask.py

Convert a KMZ (KML zipped) of polygons into a raster mask (GeoTIFF).

Usage examples:
  # Create mask with given pixel size (degrees if geometries are geographic)
  python kmz_to_mask.py -i polygons.kmz -o mask.tif --pixsize 0.0005 --burn 1

  # Match an existing raster (same transform/shape/CRS)
  python kmz_to_mask.py -i polygons.kmz -o mask.tif --like reference.tif --burn 1

Dependencies:
  pip install geopandas rasterio numpy shapely
"""
import argparse
import sys
import os
import zipfile
import tempfile
from typing import Optional
from mintpy.utils import readfile, writefile
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


EXAMPLE = "kmz_to_mask.py --input /Users/giacomo/onedrive/scratch/Chiles/Chiles_mask.kmz --output /Users/giacomo/onedrive/scratch/Chiles/SenAT120/Chiles_mask.h5 --geom /Users/giacomo/onedrive/scratch/Chiles/SenAT120/geo_geometryRadar.h5 --mask /Users/giacomo/onedrive/scratch/Chiles/SenAT120/geo_maskTempCoh.h5"
def create_parser():
    """Build and parse CLI for KMZ->mask tool (compatible style with create_parser() used elsewhere)."""
    synopsis = 'Convert KMZ polygons to a raster mask GeoTIFF'
    parser = argparse.ArgumentParser(description=synopsis, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input', required=True, nargs='*', help='Input KMZ file path(s)')
    parser.add_argument('-o', '--output', required=True, help='Output GeoTIFF mask path')
    parser.add_argument('-g', '--geom', required=True, help='Input geometry path')
    parser.add_argument('-m', '--mask', help='Input mask path')
    parser.add_argument('--mask-mode', dest='mmode', nargs='*', choices=['include', 'exclude'], default='exclude', help='Mask out area inside or outside polygons (default: %(default)s)')
    parser.add_argument('--like', help='Match transform/shape/CRS of this raster (optional)')
    parser.add_argument('--pixsize', type=float, default=0.001, help='Pixel size (units of CRS). Ignored when --like is used. Default: 0.001')
    parser.add_argument('--pixsize-y', type=float, help='Pixel size Y (if different). Use with --pixsize for X and --pixsize-y for Y')
    parser.add_argument('--burn', type=int, default=1, help='Value to burn inside polygons (default 1)')
    parser.add_argument('--nodata', type=int, default=0, help='Nodata / outside value (default 0)')
    parser.add_argument('--dissolve', action='store_true', help='Dissolve polygons into single geometry before rasterizing')

    return parser.parse_args()

def extract_kml_from_kmz(kmz_path: str, tmpdir: str) -> str:
    """Extract first .kml from KMZ to tmpdir and return path."""
    with zipfile.ZipFile(kmz_path, 'r') as z:
        kml_candidates = [n for n in z.namelist() if n.lower().endswith('.kml')]
        if not kml_candidates:
            raise RuntimeError('No .kml file found inside KMZ')
        kml_name = kml_candidates[0]
        out_path = os.path.join(tmpdir, os.path.basename(kml_name))
        with z.open(kml_name) as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
    return out_path


def read_kmz(kmz_path: str) -> gpd.GeoDataFrame:
    tmpd = tempfile.mkdtemp(prefix='kmz2mask_')
    try:
        kml = extract_kml_from_kmz(kmz_path, tmpd)
        # try the fast route via geopandas / GDAL first
        try:
            gdf = gpd.read_file(kml)
            return gdf
        except Exception as e:
            # If GDAL lacks LIBKML support (common), fall back to a pure-Python KML parser
            err = str(e).lower()
            if 'libkml' not in err and 'kml' not in err:
                # still attempt manual parse for any other read errors
                pass

        # Manual KML parsing (no LIBKML needed) - builds shapely polygons from <coordinates>
        import xml.etree.ElementTree as ET
        from shapely.geometry import Polygon, MultiPolygon

        def parse_coords_text(txt: str):
            pts = []
            if not txt:
                return pts
            for part in txt.strip().split():
                vals = part.split(',')
                if len(vals) >= 2:
                    lon = float(vals[0])
                    lat = float(vals[1])
                    pts.append((lon, lat))
            return pts

        def polygon_from_elem(poly_elem):
            # collect all <coordinates> text nodes under this Polygon; first is outer, rest are holes
            coords_texts = []
            for node in poly_elem.iter():
                if node.tag.endswith('coordinates'):
                    coords_texts.append(node.text)
            if not coords_texts:
                return None
            outer = parse_coords_text(coords_texts[0])
            holes = [parse_coords_text(t) for t in coords_texts[1:]]
            try:
                poly = Polygon(outer, holes if holes else None)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                return poly
            except Exception:
                return None

        def placemark_geoms(pm_elem):
            polys = []
            # find any Polygon elements under this placemark
            for node in pm_elem.iter():
                if node.tag.endswith('Polygon'):
                    p = polygon_from_elem(node)
                    if p is not None and not p.is_empty:
                        polys.append(p)
            if not polys:
                return None
            if len(polys) == 1:
                return polys[0]
            return MultiPolygon(polys)

        tree = ET.parse(kml)
        root = tree.getroot()

        geoms = []
        names = []
        # find all Placemark elements (namespace-agnostic)
        for pm in root.iter():
            if pm.tag.endswith('Placemark'):
                geom = placemark_geoms(pm)
                if geom is None:
                    continue
                # try to get a name if present
                name = None
                for ch in pm:
                    if ch.tag.endswith('name'):
                        name = ch.text
                        break
                geoms.append(geom)
                names.append(name)

        if not geoms:
            raise RuntimeError('No polygon geometries found in KML (manual parse)')

        gdf = gpd.GeoDataFrame({'name': names, 'geometry': geoms}, crs='EPSG:4326')
        return gdf

    finally:
        # keep tmp for debug if needed; you can remove if desired
        pass
    return gdf


def prepare_geoms(gdf: gpd.GeoDataFrame, dissolve: bool) -> gpd.GeoSeries:
    """Return a GeoSeries of polygon/multipolygon geometries to rasterize."""
    # keep only polygonal geometries
    poly_types = {'Polygon', 'MultiPolygon'}
    has_poly = gdf.geometry.geom_type.isin(poly_types)
    if not has_poly.any():
        # if only non-polygons, try taking convex hull / buffer
        raise RuntimeError('No polygon geometries found in input')
    polys = gdf.loc[has_poly].geometry
    # optionally dissolve to single geometry (useful to create single mask)
    if dissolve:
        geom = polys.unary_union
        return gpd.GeoSeries([geom], crs=gdf.crs)
    return polys.reset_index(drop=True)


def build_target_grid(
    geoms,
    pixsize: float | tuple[float, float] | None,
    like: Optional[str],
):
    """Return (transform, width, height, crs, bounds) for rasterization."""
    if like:
        with rasterio.open(like) as src:
            return src.transform, src.width, src.height, src.crs, src.bounds

    # use geometries bounds
    minx, miny, maxx, maxy = geoms.total_bounds  # [minx, miny, maxx, maxy]
    if isinstance(pixsize, tuple):
        xres, yres = pixsize
    else:
        xres = yres = float(pixsize or 0.001)  # default ~0.001 deg (~111m)
    if xres <= 0 or yres <= 0:
        raise ValueError('pixsize must be positive')

    # compute width/height (ensure we cover bounds exactly)
    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))
    # adjust maxx to align exactly
    maxx_adj = minx + width * xres
    maxy_adj = miny + height * yres
    transform = from_origin(minx, maxy_adj, xres, yres)
    bounds = (minx, miny, maxx_adj, maxy_adj)
    # default crs: use geoms crs if present, otherwise WGS84
    crs = geoms.crs if geoms.crs is not None else rasterio.crs.CRS.from_epsg(4326)
    return transform, width, height, crs, bounds


def rasterize_geoms(geoms, transform, width, height, burn_value: int = 1, nodata: int = 0,):
    """
    Rasterize geometries to a numpy array.

    Converts vector geometries to a raster representation as a 2D numpy array.

    Args:
        geoms: Iterable of geometry objects to rasterize.
        transform: Affine transformation matrix mapping pixel coordinates to 
            geographic coordinates.
        width: Width of the output raster in pixels.
        height: Height of the output raster in pixels.
        burn_value: Value to assign to pixels covered by geometries. Defaults to 1.
        nodata: Value to assign to pixels not covered by any geometry. Defaults to 0.

    Returns:
        numpy.ndarray: 2D array of shape (height, width) with dtype uint8, where
            pixels covered by geometries have the burn_value and uncovered pixels
            have the nodata value.
    """

    shapes = ((geom, burn_value) for geom in geoms)
    out = rasterize(
        shapes,
        out_shape=(height, width),
        fill=nodata,
        transform=transform,
        all_touched=False,
        dtype=np.uint8,
    )
    return out


def write_mask_tif(path: str, arr: np.ndarray, transform, crs, nodata: int = 0):
    """
    Write a 2D numpy array as a single-band GeoTIFF file.

    Parameters
    ----------
    path : str
        The file path where the GeoTIFF will be written.
    arr : np.ndarray
        A 2D numpy array containing the data to be written.
    transform : rasterio.transform.Affine
        The geospatial transform defining the location and resolution of the raster.
    crs : rasterio.crs.CRS
        The coordinate reference system of the raster.
    nodata : int, optional
        The value to be used as nodata/background value. Default is 0.

    Returns
    -------
    None
        Writes the array directly to the specified GeoTIFF file.

    Notes
    -----
    - The output file uses deflate compression.
    - The data type of the output file matches the input array's dtype.
    - The file is created as a single-band GeoTIFF with geospatial metadata.
    """
    """Write numpy 2D array as single-band GeoTIFF."""
    height, width = arr.shape
    profile = {
        'driver': 'GTiff',
        'dtype': arr.dtype,
        'count': 1,
        'width': width,
        'height': height,
        'transform': transform,
        'crs': crs,
        'nodata': nodata,
        'compress': 'deflate',
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


def adapt_mask(array, transform, width, height, geom):
    """
    Adapt a mask array to match the geospatial grid of a reference dataset.

    This function takes a rasterized mask array defined in pixel coordinates and
    resamples it to match the geospatial coordinates of a reference geometry file.
    The resampling uses nearest-neighbor interpolation on a regular grid.

    Parameters
    ----------
    array : numpy.ndarray
        2D binary mask array in the source coordinate system (width x height).
    transform : Affine
        Affine transformation matrix defining the geospatial coordinates of the mask.
        Expected to have attributes: a (pixel width), e (pixel height), c (left),
        f (top).
    width : int
        Width of the mask array in pixels.
    height : int
        Height of the mask array in pixels.
    geom : str
        File path to the reference geometry dataset containing 'latitude' and
        'longitude' datasets for the target geospatial grid.

    Returns
    -------
    numpy.ndarray
        Boolean mask array resampled to the target geospatial grid defined by
        the latitude and longitude datasets in the geometry file. Shape matches
        the geometry dataset dimensions.

    Notes
    -----
    - Uses nearest-neighbor interpolation for fast resampling.
    - NaN values in the latitude/longitude grids are treated as 0 for interpolation.
    - Out-of-bounds interpolation points are assigned a fill value of 0.
    """
    xres = float(transform.a)
    yres = abs(float(transform.e))
    left = float(transform.c)
    top = float(transform.f)

    # 1D centers
    lon_1d = left + (np.arange(width) + 0.5) * xres
    lat_1d = top - (np.arange(height) + 0.5) * yres

    latitude, meta = readfile.read(geom, datasetName='latitude')
    longitude, meta = readfile.read(geom, datasetName='longitude')

    interp = RegularGridInterpolator(
        (lat_1d, lon_1d),
        array.astype(int),
        method="nearest",
        bounds_error=False,
        fill_value=0
    )

    # evaluate on big grid
    pts = np.column_stack((latitude.ravel(), longitude.ravel()))
    out_mask = (interp(np.nan_to_num(pts, nan=0))).reshape(latitude.shape).astype(bool)

    return out_mask

def main():
    args = create_parser()
    kmz = args.input

    masks = []
    for i,k in enumerate(kmz):
        if not os.path.exists(k):
            raise RuntimeError(f'Input not found: {k}')

        # read kmz/kml
        try:
            gdf = read_kmz(k)
        except Exception as e:
            raise RuntimeError(f'Failed to read KMZ: {e}') from e
            sys.exit(1)

        if gdf.empty:
            raise RuntimeError('No features found in KMZ')

        try:
            geoms = prepare_geoms(gdf, args.dissolve)
        except Exception as e:
            raise RuntimeError(f'Geometry error: {e}') from e

        # if matching reference raster, ensure same CRS: reproject geoms to ref CRS later in build_target_grid
        pix_arg = args.pixsize
        if args.pixsize_y:
            pixsize = (args.pixsize or 0.001, args.pixsize_y)
        else:
            pixsize = args.pixsize

        try:
            transform, width, height, crs, _ = build_target_grid(geoms, pixsize, args.like)
        except Exception as e:
            raise RuntimeError(f'Failed to build target grid: {e}') from e

        # reproject geometries to target crs if needed
        if geoms.crs != crs:
            try:
                geoms = geoms.to_crs(crs)
            except Exception as e:
                raise RuntimeError(f'Failed to reproject geometries to target CRS: {e}') from e

        # rasterize
        arr = rasterize_geoms(geoms, transform, width, height, burn_value=args.burn, nodata=args.nodata)

        out_mask = adapt_mask(arr, transform, width, height, args.geom)

        if args.mmode[i] == 'include':
            out_mask = np.logical_not(out_mask)

        masks.append(out_mask)

    # combine masks from multiple inputs
    out_mask = np.prod(masks, axis=0).astype(bool)

    if args.mask:
        mask, meta = readfile.read(args.mask)
        out_mask = out_mask & (mask == 1)

    # TODO metadata is missing if mask is not imported
    writefile.write({'mask': out_mask.astype(np.uint8)}, args.output, metadata=meta)

if __name__ == '__main__':
    main()