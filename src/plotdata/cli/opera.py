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
from datetime import datetime, date, timedelta


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
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_meta_value(value.item())
        return [_normalize_meta_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_meta_value(item) for item in value]
    return value


def extract_identification_metadata(opera):
    identification_group = opera.groups.get("identification")
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
        displacement = opera.variables["displacement"][:]  # (time, y, x)
        temporal_coh = opera.variables["temporal_coherence"][:]

        time_var = opera.variables["time"]
        time_dt = netCDF4.num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        meta = extract_identification_metadata(opera)  # for debugging
        time = time_dt[0].date()

        mask = np.asarray((temporal_coh > 0.65).filled(False), dtype=bool)
        displacement_data = displacement.filled(np.nan)

        x = np.asarray(opera.variables["x"][:], dtype=float)
        y = np.asarray(opera.variables["y"][:], dtype=float)

    return displacement_data, mask, datetime.strftime(time, "%Y%m%d"), y, x, meta


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

    finite_data = np.isfinite(data)
    axes_to_reduce = tuple(range(1, data.ndim - 2))
    if axes_to_reduce:
        valid_per_file = np.any(finite_data, axis=axes_to_reduce)
    else:
        valid_per_file = finite_data

    overlap_mask = np.all(valid_per_file, axis=0)
    mask_shape = (1,) * (data.ndim - 2) + overlap_mask.shape
    data = np.where(overlap_mask.reshape(mask_shape), data, np.nan)

    return data, y_list[0], x_list[0]



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
        "ASCENDING": {"data_chunks": [], "date_list": [], "y_list": [], "x_list": [], "meta_list": []},
        "DESCENDING": {"data_chunks": [], "date_list": [], "y_list": [], "x_list": [], "meta_list": []},
        "UNKNOWN": {"data_chunks": [], "date_list": [], "y_list": [], "x_list": [], "meta_list": []},
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
            displacement_data, mask, time, y, x, meta = process_opera_file(file_path, label=label)
            orbit_direction = get_orbit_pass_direction(meta)
            group = orbit_groups[orbit_direction]

            group["data_chunks"].append(displacement_data)
            group["date_list"].append(time)
            group["y_list"].append(y)
            group["x_list"].append(x)
            group["meta_list"].append(meta)

            if plot_debug:
                plot_displacement(displacement_data)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    ascending_data_list = orbit_groups["ASCENDING"]["data_chunks"]
    descending_data_list = orbit_groups["DESCENDING"]["data_chunks"]
    ascending_meta_list = orbit_groups["ASCENDING"]["meta_list"]
    descending_meta_list = orbit_groups["DESCENDING"]["meta_list"]

    ascending_data, ascending_y, ascending_x = finalize_orbit_group(
        orbit_groups["ASCENDING"]["data_chunks"],
        orbit_groups["ASCENDING"]["y_list"],
        orbit_groups["ASCENDING"]["x_list"],
    )
    descending_data, descending_y, descending_x = finalize_orbit_group(
        orbit_groups["DESCENDING"]["data_chunks"],
        orbit_groups["DESCENDING"]["y_list"],
        orbit_groups["DESCENDING"]["x_list"],
    )

    pass