import os
import random
import argparse
import glob
import shlex
import subprocess
import tempfile
import netCDF4
import numpy as np
from tqdm.auto import tqdm
from shapely import wkt as _wkt
from datetime import datetime, date, timedelta
from mintpy.objects import HDFEOS


DEFAULT_SSH_USER = "exouser"
DEFAULT_SSH_HOST = "149.165.154.65"
DEFAULT_SSH_KEY_PATH = "~/.ssh/id_rsa_jetstream"
DEFAULT_REMOTE_PARENT = "/data/HDF5EOS/opera_download/"


def parse_args():
    parser = argparse.ArgumentParser(description="Process OPERA .nc files from local directory or SSH source.")
    parser.add_argument("input_dir", help="Directory to scan for .nc files (local or remote when --ssh is set).")
    parser.add_argument("--ssh", action="store_true", help="Use SSH mode. If not set, local mode is used unless OPERA_SOURCE=ssh.")
    parser.add_argument("--plot-debug", action="store_true", help="Plot displacement for each processed file.")
    parser.add_argument("--ssh-user", default=None, help="SSH username override.")
    parser.add_argument("--ssh-host", default=None, help="SSH host override.")
    parser.add_argument("--ssh-key", default=None, help="SSH private key path override.")

    return parser.parse_args()


def to_date(x):
    return datetime.strptime(str(x), "%Y%m%d").date()


def polygon_corners_string(polygon_str: str) -> str:
    """
    Return corners from the polygon as S0081W09112_S0081W09130_S0100W09130_S0100W09112
    """

    def fmt_lat(lat: float) -> str:
        val = int(round(abs(lat) * 100))              # keep 2 decimals
        return f"{'N' if lat >= 0 else 'S'}{val:04d}" # 2 deg digits + 2 decimals

    def fmt_lon(lon: float) -> str:
        val = int(round(abs(lon) * 100))
        return f"{'E' if lon >= 0 else 'W'}{val:05d}" # 3 deg digits + 2 decimals

    poly = _wkt.loads(polygon_str)

    # polygon vertices in counter-clockwise order starting SW, drop duplicate last point
    coords = list(poly.exterior.coords)[:-1]
    corners = [(lat, lon) for lon, lat in coords]
    parts = [f"{fmt_lat(lat)}{fmt_lon(lon)}" for (lat, lon) in corners]

    corners_str = "_".join(parts)

    return  corners_str


def get_output_filename(metadata, template, direction=None):
    """Build output filename from OPERA identification metadata."""

    def mget(key, default=None):
        # supports dict metadata and argparse.Namespace(attrs=..., variables=...)
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        if hasattr(metadata, "attrs") and key in metadata.attrs:
            return metadata.attrs.get(key, default)
        if hasattr(metadata, "variables") and key in metadata.variables:
            return metadata.variables.get(key, default)
        return default

    def parse_ymd(value):
        if not value:
            return "00000000"
        s = str(value).strip()
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(s, fmt).strftime("%Y%m%d")
            except ValueError:
                pass
        # fallback for strings like "2017-01-07T04:30:28.815125Z"
        return s[:10].replace("-", "")

    sat_raw = mget("source_data_satellite_names", mget("mission", "OPERA"))
    sat_str = str(sat_raw).upper().replace(" ", "") if sat_raw is not None else ""

    if "S1A" in sat_str or "S1B" in sat_str:
        sat = "S1"
    else:
        sat = str(sat_raw).split(",")[0].strip() if sat_raw else "OPERA"

    relorb = f"{int(mget('relative_orbit', mget('track_number', 0))):03d}"
    relorb2 = f"{int(mget('relative_orbit_second', mget('frame_id', 0))):05d}"

    method_str = str(mget("post_processing_method", "opera")).lower()

    date1 = parse_ymd(
        mget("first_date", mget("reference_datetime", mget("reference_zero_doppler_start_time")))
    )
    date2 = parse_ymd(
        mget("last_date", mget("secondary_datetime", mget("secondary_zero_doppler_start_time")))
    )

    update_flag = str(mget("cfg.mintpy.save.hdfEos5.update", "")).lower() == "yes"
    if update_flag:
        date2 = "XXXXXXXX"

    direction_val = direction or mget("orbit_pass_direction", None)
    if direction_val:
        direction_upper = str(direction_val).strip().upper()
        if "ASC" in direction_upper:
            direction_val = "asc"
        elif "DES" in direction_upper:
            direction_val = "desc"
        else:
            direction_val = str(direction_val).strip().lower()

    if direction_val:
        out_name = f"{sat}_{direction_val}_{relorb}_{relorb2}_{method_str}_{date1}_{date2}.he5"
    else:
        out_name = f"{sat}_{relorb}_{relorb2}_{method_str}_{date1}_{date2}.he5"

    fbase, fext = os.path.splitext(out_name)
    polygon_str = mget("data_footprint", mget("bounding_polygon", None))

    if polygon_str:
        try:
            sub = polygon_corners_string(polygon_str)
            out_name = f"{fbase}_{sub}{fext}"
        except Exception:
            pass

    return out_name


def aaa(a_vals, b_vals, delta):
    delta = 12

    all_pairs = []

    for shift in range(delta + 1):
        date2 = a_vals + timedelta(days=shift)
        b_index = {d: idx for idx, d in enumerate(b_vals)}
        # build pairs as date tuples (a_date, b_date)
        pairs = [(a_vals[i], b_vals[b_index[date2[i]]]) for i in range(len(date2)) if date2[i] in b_index]

        if pairs:
            matched_a = np.array([p[0] for p in pairs])
            matched_b = np.array([p[1] for p in pairs])

            a_vals = a_vals[~np.isin(a_vals, matched_a)]
            b_vals = b_vals[~np.isin(b_vals, matched_b)]

            all_pairs.extend(pairs)

        print(f"shift={shift} pairs found={len(pairs)} total unique={len(all_pairs)}")

    return np.array(all_pairs)


def generate_date_lists():
    n = 30
    start = date(2025, 1, 1)
    list_a = [(start + timedelta(days=5 * i)).strftime("%Y%m%d") for i in range(n)]
    list_b = [(start + timedelta(days=5 * i + 3)).strftime("%Y%m%d") for i in range(n)]

    random.seed(0)
    drop_a = [1, 8, 9, 12, 13, 15, 16, 24, 27, 28]
    drop_b = [4, 6, 9, 11, 15, 16, 18, 21, 25]

    a_dropped = [v for i, v in enumerate(list_a) if i not in drop_a]
    b_dropped = [v for i, v in enumerate(list_b) if i not in drop_b]

    a_vals = np.array([to_date(x) for x in a_dropped])
    b_vals = np.array([to_date(x) for x in b_dropped])

    return a_vals, b_vals


def _normalize_meta_value(value):
    def _collapse_singleton(v):
        while isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        return v

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_meta_value(value.item())
        norm_list = [_normalize_meta_value(item) for item in value.tolist()]
        return _collapse_singleton(norm_list)
    if isinstance(value, (list, tuple)):
        norm_list = [_normalize_meta_value(item) for item in value]
        return _collapse_singleton(norm_list)
    return value


def extract_identification_metadata(opera):
    identification_group = opera.groups.get("identification") if hasattr(opera, "groups") else None
    if identification_group is None:
        return argparse.Namespace(attrs={}, variables={})

    identification_attrs = {
        attr_name: _normalize_meta_value(getattr(identification_group, attr_name))
        for attr_name in identification_group.ncattrs()
    }

    identification_variables = {}
    for var_name, var_obj in identification_group.variables.items():
        var_value = _normalize_meta_value(var_obj[...])
        identification_variables[var_name] = var_value

    return argparse.Namespace(attrs=identification_attrs, variables=identification_variables)


def _decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return value


def _parse_hdf_time_value(time_value, units):
    units_str = str(_decode_if_bytes(units or "")).strip()
    if " since " not in units_str.lower():
        raise ValueError(f"Unsupported time units: {units_str}")

    split_idx = units_str.lower().index(" since ")
    unit_name = units_str[:split_idx].strip().lower()
    origin_str = units_str[split_idx + len(" since "):].strip().rstrip("Z")

    origin = None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            origin = datetime.strptime(origin_str, fmt)
            break
        except ValueError:
            continue

    if origin is None:
        raise ValueError(f"Unsupported time origin: {origin_str}")

    value = float(np.asarray(time_value).squeeze())
    if unit_name.startswith("day"):
        delta = timedelta(days=value)
    elif unit_name.startswith("hour"):
        delta = timedelta(hours=value)
    elif unit_name.startswith("min"):
        delta = timedelta(minutes=value)
    elif unit_name.startswith("sec"):
        delta = timedelta(seconds=value)
    else:
        raise ValueError(f"Unsupported time unit: {unit_name}")

    return origin + delta


def get_orbit_pass_direction(meta):
    orbit_value = None

    if hasattr(meta, "variables"):
        orbit_value = meta.variables.get("orbit_pass_direction")

    if orbit_value is None and hasattr(meta, "attrs"):
        orbit_value = meta.attrs.get("orbit_pass_direction")

    if isinstance(orbit_value, list):
        orbit_value = orbit_value[0] if orbit_value else None

    if orbit_value is None:
        return "UNKNOWN"

    orbit_str = str(orbit_value).strip().upper()
    if "ASC" in orbit_str:
        return "ASCENDING"
    if "DES" in orbit_str:
        return "DESCENDING"
    return "UNKNOWN"


def process_opera_file(file_path, label=None):
    with netCDF4.Dataset(file_path, "r") as opera:
        displacement = opera.variables["displacement"][:]
        displacement_data = displacement.filled(np.nan) if hasattr(displacement, "filled") else np.asarray(displacement)
        displacement_data = np.asarray(displacement_data, dtype=np.float32)

        temporal_coh = np.asarray(opera.variables["temporal_coherence"][:], dtype=np.float32)
        mask = np.asarray(opera.variables["recommended_mask"][:])

        time_var = opera.variables["time"]
        time_dt = netCDF4.num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        time = time_dt[0].date()

        meta = extract_identification_metadata(opera)
        x = np.asarray(opera.variables["x"][:], dtype=float)
        y = np.asarray(opera.variables["y"][:], dtype=float)

    return displacement_data, mask, temporal_coh, datetime.strftime(time, "%Y%m%d"), y, x, meta


def _slice_for_overlap_axis(axis_vals, overlap_min, overlap_max):
    axis = np.asarray(axis_vals, dtype=float)
    lo = min(overlap_min, overlap_max)
    hi = max(overlap_min, overlap_max)

    if axis[0] <= axis[-1]:
        i0 = np.searchsorted(axis, lo, side="left")
        i1 = np.searchsorted(axis, hi, side="right")
    else:
        axis_rev = axis[::-1]
        j0 = np.searchsorted(axis_rev, lo, side="left")
        j1 = np.searchsorted(axis_rev, hi, side="right")
        i0 = len(axis) - j1
        i1 = len(axis) - j0

    if i0 >= i1:
        raise ValueError("No overlap found along one coordinate axis.")

    return slice(i0, i1)


def crop_chunks_to_common_xy(data_chunks, y_list, x_list):
    x_overlap_min = max(np.min(x) for x in x_list)
    x_overlap_max = min(np.max(x) for x in x_list)
    y_overlap_min = max(np.min(y) for y in y_list)
    y_overlap_max = min(np.max(y) for y in y_list)

    if not (x_overlap_min < x_overlap_max and y_overlap_min < y_overlap_max):
        raise ValueError("No overlapping area across files based on x/y coordinates.")

    cropped_data = []
    cropped_y = []
    cropped_x = []

    for chunk, y_vals, x_vals in zip(data_chunks, y_list, x_list):
        y_slice = _slice_for_overlap_axis(y_vals, y_overlap_min, y_overlap_max)
        x_slice = _slice_for_overlap_axis(x_vals, x_overlap_min, x_overlap_max)

        if chunk.ndim == 3:
            chunk_cropped = chunk[:, y_slice, x_slice]
        elif chunk.ndim == 2:
            chunk_cropped = chunk[y_slice, x_slice]
        else:
            raise ValueError(f"Unsupported chunk dimensions: {chunk.shape}")

        cropped_data.append(chunk_cropped)
        cropped_y.append(np.asarray(y_vals[y_slice], dtype=float))
        cropped_x.append(np.asarray(x_vals[x_slice], dtype=float))

    min_ny = min(arr.shape[-2] for arr in cropped_data)
    min_nx = min(arr.shape[-1] for arr in cropped_data)

    cropped_data = [arr[..., :min_ny, :min_nx] for arr in cropped_data]
    cropped_y = [arr[:min_ny] for arr in cropped_y]
    cropped_x = [arr[:min_nx] for arr in cropped_x]

    return cropped_data, cropped_y, cropped_x


def finalize_orbit_group(data_chunks, y_list, x_list):
    if not data_chunks:
        return None, None, None

    data_chunks, y_list, x_list = crop_chunks_to_common_xy(data_chunks, y_list, x_list)
    data = np.stack(data_chunks, axis=0)

    return data, y_list[0], x_list[0]


def _flatten_time_cube(data):
    arr = np.asarray(data)
    if arr.ndim == 4:
        n_files, n_time, ny, nx = arr.shape
        return arr.reshape(n_files * n_time, ny, nx), n_time
    if arr.ndim == 3:
        return arr, 1
    raise ValueError(f"Unsupported displacement dimensions: {arr.shape}")


def write_orbit_hdf5(group, out_file):
    data, y, x = finalize_orbit_group(group["data_chunks"], group["y_list"], group["x_list"])
    if data is None:
        return None

    data_3d, n_time_per_file = _flatten_time_cube(data)

    mask_stack = None
    if group["mask"]:
        mask_chunks, _, _ = crop_chunks_to_common_xy(group["mask"], group["y_list"], group["x_list"])
        mask_stack = np.stack(mask_chunks, axis=0)

    temporal_stack = None
    if group["temporal_coh"]:
        temporal_chunks, _, _ = crop_chunks_to_common_xy(group["temporal_coh"], group["y_list"], group["x_list"])
        temporal_stack = np.stack(temporal_chunks, axis=0)

    date_list = list(group["date_list"])
    if n_time_per_file > 1:
        expanded_dates = []
        for date_item in date_list:
            expanded_dates.extend([date_item] * n_time_per_file)
        date_list = expanded_dates

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    if mask_stack is not None:
        mask_2d = np.any(np.asarray(mask_stack) > 0, axis=0).astype(np.bool_)
    else:
        mask_2d = np.ones((data_3d.shape[1], data_3d.shape[2]), dtype=np.bool_)

    if temporal_stack is not None:
        temporal_arr = np.asarray(temporal_stack, dtype=np.float32)
        if temporal_arr.ndim == 4:
            temporal_2d = np.nanmean(temporal_arr, axis=(0, 1))
        elif temporal_arr.ndim == 3:
            temporal_2d = np.nanmean(temporal_arr, axis=0)
        elif temporal_arr.ndim == 2:
            temporal_2d = temporal_arr
        else:
            raise ValueError(f"Unsupported temporal coherence dimensions: {temporal_arr.shape}")
    else:
        temporal_2d = np.full((data_3d.shape[1], data_3d.shape[2]), np.nan, dtype=np.float32)

    with netCDF4.Dataset(out_file, "w", format="NETCDF4") as f:
        f.createDimension("time", len(date_list))
        f.createDimension("y", data_3d.shape[1])
        f.createDimension("x", data_3d.shape[2])

        hdfeos_group = f.createGroup("HDFEOS")
        grids_group = hdfeos_group.createGroup("GRIDS")
        ts_group = grids_group.createGroup("timeseries")
        obs_group = ts_group.createGroup("observation")
        qual_group = ts_group.createGroup("quality")
        geom_group = ts_group.createGroup("geometry")

        disp_var = obs_group.createVariable("displacement", "f4", ("time", "y", "x"), zlib=True, complevel=4)
        disp_var[:] = data_3d.astype(np.float32)

        date_var = obs_group.createVariable("date", str, ("time",))
        date_var[:] = np.asarray(date_list, dtype=object)

        bperp_var = obs_group.createVariable("bperp", "f4", ("time",))
        bperp_var[:] = np.zeros((len(date_list),), dtype=np.float32)

        tcoh_var = qual_group.createVariable("temporalCoherence", "f4", ("y", "x"), zlib=True, complevel=4)
        tcoh_var[:] = temporal_2d.astype(np.float32)

        mask_var = qual_group.createVariable("mask", "i1", ("y", "x"), zlib=True, complevel=4)
        mask_var[:] = mask_2d.astype(np.int8)

        y_var = geom_group.createVariable("y", "f4", ("y",))
        y_var[:] = np.asarray(y, dtype=np.float32)

        x_var = geom_group.createVariable("x", "f4", ("x",))
        x_var[:] = np.asarray(x, dtype=np.float32)

        f.FILE_TYPE = "HDFEOS"
        f.LENGTH = str(data_3d.shape[1])
        f.WIDTH = str(data_3d.shape[2])

        if group["meta_list"]:
            meta_attrs = _meta_to_dict(group["meta_list"][0])
            for key, value in meta_attrs.items():
                if value is None:
                    continue
                norm_val = _normalize_meta_value(value)
                setattr(f, key, str(norm_val) if isinstance(norm_val, (list, tuple, dict, np.ndarray)) else _decode_if_bytes(norm_val))

    validator = HDFEOS(out_file)
    validator.open(print_msg=False)
    validator.get_date_list()
    validator.close(print_msg=False)

    return out_file


def _meta_to_dict(meta):
    data = {}
    if isinstance(meta, dict):
        data.update(meta)
    if hasattr(meta, "attrs") and isinstance(meta.attrs, dict):
        data.update(meta.attrs)
    if hasattr(meta, "variables") and isinstance(meta.variables, dict):
        data.update(meta.variables)
    return data


def _normalize_satellite_token(value):
    if value is None:
        return None
    token = str(value).strip().upper().replace(" ", "")
    if "S1A" in token or "S1B" in token:
        return "S1"
    return token


def _build_filename_metadata(meta_list):
    if not meta_list:
        return {}

    all_meta_dicts = [_meta_to_dict(meta) for meta in meta_list]
    filename_meta = dict(all_meta_dicts[0])

    sat_tokens = []
    for meta_dict in all_meta_dicts:
        sat_raw = meta_dict.get("source_data_satellite_names", meta_dict.get("mission"))
        sat_token = _normalize_satellite_token(sat_raw)
        if sat_token:
            sat_tokens.append(sat_token)

    if sat_tokens and len(set(sat_tokens)) == 1:
        filename_meta["source_data_satellite_names"] = sat_tokens[0]
    else:
        filename_meta["source_data_satellite_names"] = "OPERA"

    return filename_meta


def save_hdfeos5(orbit_groups, output_dir):
    saved_files = []

    print(f"Saving HDFEOS outputs to: {output_dir}")

    for orbit_direction, group in orbit_groups.items():
        n_files = len(group.get("meta_list", []))
        print(f"Processing orbit group: {orbit_direction} ({n_files} files)")

        if not group["meta_list"]:
            print(f"Skipping {orbit_direction}: no metadata/files collected.")
            continue

        filename_meta = _build_filename_metadata(group["meta_list"])
        out_name = get_output_filename(filename_meta, None, direction=orbit_direction)
        out_file = os.path.join(output_dir, out_name)
        print(f"Writing {orbit_direction} output: {out_file}")
        written = write_orbit_hdf5(group, out_file)
        if written:
            saved_files.append(written)
            print(f"Finished {orbit_direction}: {written}")

    print(f"Completed save_hdfeos5: wrote {len(saved_files)} file(s).")

    return saved_files



def _ssh_common_options(ssh_key_path):
    return [
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=30",
        "-o", "TCPKeepAlive=yes",
        "-i", os.path.expanduser(ssh_key_path),
    ]


def list_remote_nc_files(remote_dir, ssh_user, ssh_host, ssh_key_path):
    remote_find = f"find {shlex.quote(remote_dir)} -maxdepth 1 -type f -name '*.nc' | sort"
    cmd = [
        "ssh",
        "-YC",
        *_ssh_common_options(ssh_key_path),
        f"{ssh_user}@{ssh_host}",
        remote_find,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def copy_remote_file_to_temp(remote_file, ssh_user, ssh_host, ssh_key_path):
    temp_file = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    cmd = [
        "scp",
        *_ssh_common_options(ssh_key_path),
        f"{ssh_user}@{ssh_host}:{remote_file}",
        temp_path,
    ]
    subprocess.run(cmd, check=True)
    return temp_path


def plot_displacement(displacement_data):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    data_to_plot = displacement_data[0] if displacement_data.ndim == 3 else displacement_data
    ax.imshow(data_to_plot)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    scratchdir = os.getenv("SCRATCHDIR")
    source_mode = "ssh" if args.ssh else os.getenv("OPERA_SOURCE", "local").strip().lower()

    default_parent = os.path.join(scratchdir, "opera_download") if scratchdir else None
    plot_debug = args.plot_debug

    if source_mode == "local":
        local_parent = args.input_dir or default_parent

        if not local_parent:
            raise EnvironmentError("SCRATCHDIR is not set.")

        if not scratchdir:
            raise EnvironmentError("SCRATCHDIR is not set.")

        local_parent_expanded = os.path.expanduser(local_parent)
        scratchdir_expanded = os.path.expanduser(scratchdir)
        if (not os.path.isabs(local_parent_expanded)) or (scratchdir_expanded not in local_parent_expanded):
            local_parent = os.path.join(scratchdir_expanded, local_parent_expanded.lstrip(os.sep))
        else:
            local_parent = local_parent_expanded

        if not os.path.isdir(local_parent):
            raise FileNotFoundError(f"Input folder not found: {local_parent}")

        source_files = sorted(glob.glob(os.path.join(local_parent, "*.nc")))
        if not source_files:
            raise FileNotFoundError(f"No .nc files found in: {local_parent}")

        ssh_user = ssh_host = ssh_key_path = None

    elif source_mode == "ssh":
        ssh_user = args.ssh_user or os.getenv("OPERA_SSH_USER", DEFAULT_SSH_USER)
        ssh_host = args.ssh_host or os.getenv("OPERA_SSH_HOST", DEFAULT_SSH_HOST)
        ssh_key_path = args.ssh_key or os.getenv("OPERA_SSH_KEY", DEFAULT_SSH_KEY_PATH)
        remote_parent = args.input_dir or os.getenv("OPERA_REMOTE_DIR", DEFAULT_REMOTE_PARENT)

        if not remote_parent:
            raise EnvironmentError("Set OPERA_REMOTE_DIR or SCRATCHDIR for ssh mode.")

        source_files = list_remote_nc_files(remote_parent, ssh_user, ssh_host, ssh_key_path)
        if not source_files:
            raise FileNotFoundError(f"No remote .nc files found in: {remote_parent}")

    else:
        raise ValueError("Invalid OPERA_SOURCE. Use 'local' or 'ssh'.")

    orbit_groups = {
        "ASCENDING": {"data_chunks": [], "mask": [], "temporal_coh": [], "date_list": [], "y_list": [], "x_list": [], "meta_list": []},
        "DESCENDING": {"data_chunks": [], "mask": [], "temporal_coh": [], "date_list": [], "y_list": [], "x_list": [], "meta_list": []},
        "UNKNOWN": {"data_chunks": [], "mask": [], "temporal_coh": [], "date_list": [], "y_list": [], "x_list": [], "meta_list": []},
    }

    for source_file in tqdm(source_files, desc=f"Processing ({source_mode})", unit="file"):
        temp_path = None
        file_path = source_file
        label = None

        if source_mode == "ssh":
            temp_path = copy_remote_file_to_temp(source_file, ssh_user, ssh_host, ssh_key_path)
            file_path = temp_path
            label = source_file

        try:
            displacement_data, mask, temporal_coh, time, y, x, meta = process_opera_file(file_path, label=label)
            orbit_direction = get_orbit_pass_direction(meta)
            group = orbit_groups[orbit_direction]

            group["data_chunks"].append(displacement_data)
            group['mask'].append(mask)
            group['temporal_coh'].append(temporal_coh)
            group["date_list"].append(time)
            group["y_list"].append(y)
            group["x_list"].append(x)
            group["meta_list"].append(meta)

            if plot_debug:
                plot_displacement(displacement_data)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    output_dir = os.path.abspath(local_parent if source_mode == "local" else os.getcwd())
    saved = save_hdfeos5(orbit_groups, output_dir)

    if saved:
        for out_path in saved:
            print(f"Wrote: {out_path}")
    else:
        print("No orbit groups with data to save.")