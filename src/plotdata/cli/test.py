import os
import random
import netCDF4
import xarray as xr
import numpy as np
from datetime import datetime, date, timedelta


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


if __name__ == "__main__":
    scratchdir = os.getenv("SCRATCHDIR")
    if not scratchdir:
        raise EnvironmentError("SCRATCHDIR is not set.")

    file = os.path.join(
        scratchdir,
        "opera_download",
        "OPERA_L3_DISP-S1_IW_F16940_VV_20160707T014949Z_20160731T014950Z_v1.0_20250408T164134Z.nc",
    )

    if not os.path.exists(file):
        raise FileNotFoundError(f"Dataset file not found: {file}")

    opera = netCDF4.Dataset(file, "r")

    print("=== FILE OVERVIEW ===")
    print(f"Dimensions: {opera.dimensions.keys()}")
    for dim_name, dim in opera.dimensions.items():
        print(f"  {dim_name}: {len(dim)}")

    print("\n=== VARIABLES ===")
    print(opera.variables.keys())

    displacement = opera.variables['displacement'][:]  # Shape: (time, y, x)

    # Convert masked array to regular numpy array with NaN for invalid values
    displacement_data = displacement.filled(np.nan)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(displacement_data)
    plt.tight_layout()
    plt.show()
    pass