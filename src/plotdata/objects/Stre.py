import os
import re
import sys
import argparse
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from mintpy.utils import readfile
from matplotlib import pyplot as plt
from mintpy.objects import timeseries, HDFEOS
from scipy.interpolate import UnivariateSpline
from plotdata.objects.forward import Penny, Mogi, Okada, Yang
from plotdata.helper_functions import read_best_values, convert_to_utm, calculate_LOS

SCRATCHDIR = os.getenv('SCRATCHDIR')
EXAMPLE = ''

MODELS = {
    'penny': Penny,
    'mogi': Mogi,
    'okada': Okada,
    'spheroid': Yang,
}

def create_parser():
    synopsis = 'Plotting of InSAR, GPS and Seismicity data'
    epilog = EXAMPLE
    parser = argparse.ArgumentParser(description=synopsis, epilog=epilog, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('data_dir', help='Directory(s) with InSAR data.\n')
    parser.add_argument('--period', '-p', nargs='*')
    parser.add_argument('--model', '-m', nargs='*')
    parser.add_argument('--index', '-i', nargs='*')
    parser.add_argument('--timeseries', '-t', help='Vertical timeseries for inversion')
    parser.add_argument('--lookup', '-l', dest='geometry_file', help='Geometry file')
    parser.add_argument("--lalo", nargs='*', default=None, metavar=('LAT:LON or LAT,LON'), type=str, help="lat/lon coords of  pixel for timeseries")
    parser.add_argument('--ref-lalo', nargs='?', metavar=('LAT:LON or LAT,LON'), default=None, type=str, help='reference point (default:  existing reference point)')
    parser.add_argument("--offset", nargs='*', type=int, default=[0], help="offset to apply to each timeseries (default: %(default)s).")

    parser.add_argument('--unit', default='cm', help='Unit for velocity vectors (default: %(default)s).')
    parser.add_argument('--figsize', nargs=2, type=float, metavar=('WIDTH', 'HEIGHT'), default=[7.5, 12.0],
                        help='Figure size in inches (default: %(default)s).')

    inps = parser.parse_args()

    inps.start_date = []
    inps.end_date = []

    if inps.period:
        for p in inps.period:
            delimiters = '[,:\-\s]'
            dates = re.split(delimiters, p)

            if len(dates[0]) and len(dates[1]) != 8:
                msg = 'Date format not valid, it must be in the format YYYYMMDD'
                raise ValueError(msg)

            inps.start_date.append(dates[0])
            inps.end_date.append(dates[1])

    if inps.ref_lalo:
        split = inps.ref_lao.split(',') if ',' in inps.ref_lalo else inps.ref_lalo.split(':')
        inps.ref_la = float(split[0])
        inps.ref_lo = float(split[1])

    if inps.lalo:
        inps.latitudes = []
        inps.longitudes = []
        for i in inps.lalo:
            if ',' in i:
                split = i.split(',')
            else:
                split = i.split(':')
            inps.latitudes.append(float(split[0]))
            inps.longitudes.append(float(split[1]))

    if inps.timeseries:
        inps.layout = [['timeseries']]

    if len(inps.offset) != len(inps.period):
        for i in range(len(inps.period) - len(inps.offset)):
            inps.offset.append(0)

    return inps


def get_source_increment(value):
    """
    Calculate the source increment based on the provided value dictionary.

    The function computes an increment value (`inc`) depending on the keys present in the input dictionary `value`:
    - If "param1" is present (not None), it calculates the increment as the Euclidean norm of the slip and opening contributions over an area defined by "length" and "width".
    - If "dP_mu" and "radius" are present, it calculates the increment as the product of "dP_mu" and the cube of "radius".
    - If "dVol" is present, it uses its value as the increment.

    Works for the following source types:
    - Mogi: uses "dVol".
    - Penny-shaped crack: uses "dP_mu" and "radius".
    - Okada: uses "param1" (slip), "length", and "width".

    Parameters:
        value (dict): A dictionary containing source parameters. Expected keys are:
            - "param1" (float, optional): Slip value.
            - "length" (float, optional): Length of the source.
            - "width" (float, optional): Width of the source.
            - "opening" (float, optional): Opening value.
            - "dP_mu" (float, optional): Pressure change over shear modulus.
            - "radius" (float, optional): Radius of the source.
            - "dVol" (float, optional): Volume increment.

    Returns:
        float: The calculated source increment.
    """
    inc = 0.0
    if value.get("param1") is not None:
        L = value.get("length", 0.0)
        W = value.get("width", 0.0)
        A = L * W
        slip = value.get("param1", 0.0)
        opening = value.get("opening", 0.0)
        Ps = slip * A
        Pt = opening * A
        inc = (Ps**2 + Pt**2) ** 0.5
    elif value.get("dP_mu") is not None and value.get("radius") is not None:
        inc = value["dP_mu"] * (value["radius"] ** 3)
    elif value.get("dVol") is not None:
        inc = value["dVol"]

    return inc

def main(iargs):
    inps = create_parser()

    sources = {}
    if inps.timeseries:
        atr = readfile.read_attribute(inps.timeseries)

        # Identify file type and open it
        if atr['FILE_TYPE'] == 'timeseries':
            obj = timeseries(inps.timeseries)
        elif atr['FILE_TYPE'] == 'HDFEOS':
            obj = HDFEOS(inps.timeseries)
        else:
            raise ValueError(f'Input file is {atr["FILE_TYPE"]}, not timeseries.')

        obj.open(print_msg=False)
        if hasattr(obj, "datasetGroupNameDict"):
            obj.datasetGroupNameDict.update({
                "latitude": "geometry",
                "longitude": "geometry",
        })

        data = obj.read()  # (nt, ny, nx)
        mask = obj.read(datasetName='mask').astype(bool)  # (ny, nx)
        timeseries = []

        for lat, lon, start, end, m, i, off in zip(inps.latitudes, inps.longitudes, inps.start_date, inps.end_date, inps.model, inps.index, inps.offset):
            y = np.argmin(np.abs(obj.read(datasetName='latitude')[:, 0] - lat))
            x = np.argmin(np.abs(obj.read(datasetName='longitude')[0, :] - lon))

            ref_x = np.argmin(np.abs(obj.read(datasetName='longitude')[0, :] - inps.ref_lo))
            ref_y = np.argmin(np.abs(obj.read(datasetName='latitude')[:, 0] - inps.ref_la))

            # keep 3D shape; set masked pixels to NaN
            data = np.where(mask[None, :, :], data, np.nan)
            data = data - data[:, ref_y, ref_x][:, None, None]

            # then extract pixel time series normally
            ts = (data[:, y, x] * 100) + off
            dates = [datetime.strptime(str(int(date)), "%Y%m%d").date() for date in obj.read(datasetName='date')]
            start_dt = datetime.strptime(str(int(start)), "%Y%m%d").date()
            end_dt   = datetime.strptime(str(int(end)), "%Y%m%d").date()

            if obj.read(datasetName='incidenceAngle')[~np.isnan(obj.read(datasetName='incidenceAngle'))].size > 0:
                azimuth = obj.read(datasetName='azimuthAngle')[y, x]
                incidence = obj.read(datasetName='incidenceAngle')[y, x]

            i0 = int(np.argmin([abs((d - start_dt).days) for d in dates]))
            i1 = int(np.argmin([abs((d - end_dt).days) for d in dates]))

            if i0 > i1:
                i0, i1 = i1, i0

            if f"{m}_{i}" not in sources:
                sources[f"{m}_{i}"] = {}
            if f"{start}_{end}" not in sources[f"{m}_{i}"]:
                sources[f"{m}_{i}"][f"{start}_{end}"] = {}

            sources[f"{m}_{i}"][f"{start}_{end}"]['point_xy'] = convert_to_utm([lat], [lon])
            sources[f"{m}_{i}"][f"{start}_{end}"]['ts'] = ts[i0:i1+1]
            sources[f"{m}_{i}"][f"{start}_{end}"]['date_list'] = dates[i0:i1+1]

    data_path = os.path.join(SCRATCHDIR, inps.data_dir)
    params = ['dP_mu', 'dVol', 'opening', 'param1']

    for s, e, m, i in zip(inps.start_date, inps.end_date, inps.model, inps.index):
        model_inps = os.path.join(data_path, f"{s}_{e}", m, "VSM_best.csv")

        if not os.path.isfile(model_inps):
            raise ValueError(f"Model input file '{model_inps}' not found. Skipping this period and model.")

        if f"{m}_{i}" not in sources:
            sources[f"{m}_{i}"] = {}
        if f"{s}_{e}" not in sources[f"{m}_{i}"]:
            sources[f"{m}_{i}"][f"{s}_{e}"] = {}

        param_vals = next(iter(read_best_values(model_inps, params) .values()), {})
        sources[f"{m}_{i}"][f"{s}_{e}"]['params'] = param_vals

        frwd = MODELS[m]()
        x, y = sources[f"{m}_{i}"][f"{s}_{e}"]['point_xy']

        if 'dP_mu' in param_vals:
            param_vals['dP_mu'] = 1
        if 'dVol' in param_vals:
            param_vals['dVol'] = 1

        _, _, Gz = frwd.model(np.array([x]), np.array([y]), **param_vals)

        dP = sources[f"{m}_{i}"][f"{s}_{e}"]['ts'] / Gz[0]

    pass

def old_main(iargs):
    inps = create_parser()

    sources = {}
    if inps.timeseries:
        atr = readfile.read_attribute(inps.timeseries)

        # Identify file type and open it
        if atr['FILE_TYPE'] == 'timeseries':
            obj = timeseries(inps.timeseries)
        elif atr['FILE_TYPE'] == 'HDFEOS':
            obj = HDFEOS(inps.timeseries)
        else:
            raise ValueError(f'Input file is {atr["FILE_TYPE"]}, not timeseries.')

        obj.open(print_msg=False)
        if hasattr(obj, "datasetGroupNameDict"):
            obj.datasetGroupNameDict.update({
                "latitude": "geometry",
                "longitude": "geometry",
        })

        data = obj.read()  # (nt, ny, nx)
        mask = obj.read(datasetName='mask').astype(bool)  # (ny, nx)
        timeseries = []

        for lat, lon, start, end, m, i, off in zip(inps.latitudes, inps.longitudes, inps.start_date, inps.end_date, inps.model, inps.index, inps.offset):
            y = np.argmin(np.abs(obj.read(datasetName='latitude')[:, 0] - lat))
            x = np.argmin(np.abs(obj.read(datasetName='longitude')[0, :] - lon))

            ref_x = np.argmin(np.abs(obj.read(datasetName='longitude')[0, :] - inps.ref_lo))
            ref_y = np.argmin(np.abs(obj.read(datasetName='latitude')[:, 0] - inps.ref_la))

            # keep 3D shape; set masked pixels to NaN
            data = np.where(mask[None, :, :], data, np.nan)
            data = data - data[:, ref_y, ref_x][:, None, None]

            # then extract pixel time series normally
            ts = (data[:, y, x] * 100) + off
            dates = [datetime.strptime(str(int(date)), "%Y%m%d").date() for date in obj.read(datasetName='date')]
            start_dt = datetime.strptime(str(int(start)), "%Y%m%d").date()
            end_dt   = datetime.strptime(str(int(end)), "%Y%m%d").date()

            i0 = int(np.argmin([abs((d - start_dt).days) for d in dates]))
            i1 = int(np.argmin([abs((d - end_dt).days) for d in dates]))

            if i0 > i1:
                i0, i1 = i1, i0

            if f"{m}_{i}" not in sources:
                sources[f"{m}_{i}"] = {}
            if f"{start}_{end}" not in sources[f"{m}_{i}"]:
                sources[f"{m}_{i}"][f"{start}_{end}"] = {}

            sources[f"{m}_{i}"][f"{start}_{end}"]['ts'] = ts[i0:i1+1]
            sources[f"{m}_{i}"][f"{start}_{end}"]['date_list'] = dates[i0:i1+1]

    data_path = os.path.join(SCRATCHDIR, inps.data_dir)
    params = ['dP_mu', 'dVol', 'opening', 'param1']

    for s, e, m, i in zip(inps.start_date, inps.end_date, inps.model, inps.index):
        model_inps = os.path.join(data_path, f"{s}_{e}", m, "VSM_best.csv")

        if not os.path.isfile(model_inps):
            raise ValueError(f"Model input file '{model_inps}' not found. Skipping this period and model.")

        if f"{m}_{i}" not in sources:
            sources[f"{m}_{i}"] = {}
        if f"{s}_{e}" not in sources[f"{m}_{i}"]:
            sources[f"{m}_{i}"][f"{s}_{e}"] = {}

        param_vals = next(iter(read_best_values(model_inps, params) .values()), {})
        sources[f"{m}_{i}"][f"{s}_{e}"]['params'] = param_vals

    plt.rcParams['figure.figsize'] = (inps.figsize[0], inps.figsize[1])
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(inps.figsize[0], inps.figsize[1], forward=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1])
    title_size = 24
    label_size = 20
    tick_size = 18
    legend_size = 18
    plt.rcParams.update({
        'font.size': tick_size,
        'axes.labelsize': label_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': legend_size,
    })
    line_width = 3
    model_list = sorted(sources.keys())
    cmap = plt.get_cmap("plasma")   # try: "cividis", "plasma", "turbo"
    vals = np.linspace(0.1, 0.9, len(model_list))  # avoid very dark/light ends
    model_colors = {m: cmap(v) for m, v in zip(model_list, vals)}

    for model in sorted(sources.keys()):
        strength = 1.0

        for date in sorted(sources[model].keys()):
            value = sources[model][date]["params"]
            inc = get_source_increment(value)

            strength += inc
            sources[model][date]['strength'] = strength

    ts_dates = [
        dt
        for source_data in sources.values()
        for period_data in source_data.values()
        for dt in period_data.get('date_list', [])
    ]
    s = min(ts_dates) if ts_dates else min([datetime.strptime(x, "%Y%m%d").date() for x in inps.start_date])
    e = max([datetime.strptime(x, "%Y%m%d").date() for x in inps.end_date])
    quake_date = datetime(2022, 7, 22).date()
    first_event_date = datetime(2014, 10, 20).date()

    preferred_order = ['mogi_2', 'penny_3', 'penny_1', 'okada_4']

    def source_order_key(key):
        if key in preferred_order:
            return (preferred_order.index(key), key)
        return (len(preferred_order), key)

    source_keys = sorted(sources.keys(), key=source_order_key)
    n_source_axes = min(4, len(source_keys))
    plot_source_keys = source_keys[:n_source_axes]
    source_axes = []
    if n_source_axes > 0:
        source_height_weights = []
        for source in plot_source_keys:
            sorted_dates = sorted(sources[source].keys())
            final_strength = sources[source][sorted_dates[-1]]['strength'] if sorted_dates else 1.0
            strength_scale = max(float(final_strength), 1.0)
            strength_exp = int(np.floor(np.log10(strength_scale)))
            source_height_weights.append(max(strength_exp + 1, 1))

        source_subgs = gs[0, 0].subgridspec(n_source_axes, 1, height_ratios=source_height_weights, hspace=0.10)
        for idx in range(n_source_axes):
            source_axes.append(fig.add_subplot(source_subgs[idx, 0]))

    legend_handles = []

    for ax_src, source in zip(source_axes, plot_source_keys):
        color = model_colors[source]
        ax_src.tick_params(axis='both', labelsize=tick_size)
        x_all = []
        y_all = []
        sorted_dates = sorted(sources[source].keys())
        final_strength = sources[source][sorted_dates[-1]]['strength'] if sorted_dates else 0.0

        for date in sorted_dates:
            dlist = sources[source][date]['date_list']
            ts = sources[source][date]['ts']

            if len(dlist) < 2 or len(ts) < 2:
                continue

            x_all.extend(dlist)
            y_all.extend(ts)

        if len(x_all) < 2:
            ax_src.set_visible(False)
            continue

        order = np.argsort(x_all)
        x_all = np.array(x_all, dtype=object)[order]
        y_all = np.array(y_all, dtype=float)[order]

        y_all = y_all - y_all[0]
        end_value = y_all[-1]
        if np.isclose(end_value, 0.0):
            y_scaled = np.zeros_like(y_all)
        else:
            y_scaled = y_all / end_value * final_strength

        x_num_all = mdates.date2num(x_all)
        unique_x = []
        unique_y = []
        for x_val, y_val in zip(x_num_all, y_scaled):
            if unique_x and np.isclose(x_val, unique_x[-1]):
                unique_y[-1] = y_val
            else:
                unique_x.append(x_val)
                unique_y.append(y_val)

        x_num = np.asarray(unique_x, dtype=float)
        y_scaled = np.asarray(unique_y, dtype=float)

        if len(x_num) >= 4:
            smooth_factor = max(len(y_scaled) * np.nanvar(y_scaled) * 0.2, 1e-6)
            spline = UnivariateSpline(x_num, y_scaled, s=smooth_factor)
            x_smooth = np.linspace(x_num.min(), x_num.max(), 300)
            y_smooth = spline(x_smooth)
        else:
            x_smooth = x_num
            y_smooth = y_scaled

        y_smooth = y_smooth - y_smooth[0]
        if final_strength > 0 and not np.isclose(y_smooth[-1], 0.0):
            y_smooth = y_smooth * (final_strength / y_smooth[-1])

        first_date = min(x_all)
        last_date = max(x_all)
        if s < first_date:
            ax_src.plot([s, first_date], [0, 0], color=color, linewidth=line_width, alpha=0.7)
        ax_src.scatter(
            mdates.num2date(x_num),
            y_scaled,
            color=color,
            s=28,
            alpha=0.35,
            edgecolors='none',
            zorder=2,
        )
        main_line, = ax_src.plot(mdates.num2date(x_smooth), y_smooth, color=color, linewidth=line_width, label=source)
        legend_handles.append(main_line)
        if last_date < e:
            tail_y = float(y_smooth[-1])
            ax_src.plot([last_date, e], [tail_y, tail_y], color=color, linewidth=line_width, alpha=0.7)
        ax_src.set_xlim(s, e)
        ax_src.axvline(first_event_date, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
        ax_src.axvline(quake_date, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
        y_min = min(float(np.nanmin(y_scaled)), float(np.nanmin(y_smooth)), 0.0)
        y_max = max(float(np.nanmax(y_scaled)), float(np.nanmax(y_smooth)), final_strength)
        y_span = y_max - y_min
        y_pad = y_span * 0.04 if y_span > 0 else max(abs(y_max) * 0.04, 1.0)
        y_upper = y_max + y_pad
        ax_src.set_ylim(0, y_upper)
        y_tick = final_strength if final_strength > 0 else y_upper
        ax_src.set_yticks([0, y_tick])
        if np.isclose(y_tick, 0.0):
            y_tick_label = '0'
        else:
            y_exp = int(np.floor(np.log10(abs(y_tick))))
            y_mantissa = y_tick / (10 ** y_exp)
            y_mantissa = int(np.round(y_mantissa))
            if y_mantissa == 10:
                y_mantissa = 1
                y_exp += 1
            y_tick_label = rf'${y_mantissa}\times10^{{{y_exp}}}$'
        ax_src.set_yticklabels(['0', y_tick_label])

    if source_axes:
        # fig.text(0.715, 0.5, 'Volume change (m3)', va='center', ha='center', rotation='vertical', fontsize=max(label_size - 2, 10))
        for idx, ax_src in enumerate(source_axes):
            ax_src.tick_params(axis='x', which='both', bottom=False, labelbottom=False, labelsize=tick_size)
            ax_src.spines['bottom'].set_visible(False)
            ax_src.spines['top'].set_visible(idx == 0)
            ax_src.xaxis.set_major_formatter(mdates.DateFormatter(''))
            ax_src.xaxis.get_offset_text().set_visible(False)
        source_axes[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=tick_size)
        source_axes[-1].spines['bottom'].set_visible(True)
        year_span = max(e.year - s.year, 1)
        label_year_step = max(1, int(np.ceil(year_span / 6)))
        source_axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1))

        def year_label_formatter(x, pos):
            d = mdates.num2date(x)
            return f"{d.year}" if (d.year % label_year_step == 0) else ""

        source_axes[-1].xaxis.set_major_formatter(FuncFormatter(year_label_formatter))
        source_axes[-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7, bymonthday=1))
        source_axes[-1].tick_params(axis='x', which='major', bottom=True, labelbottom=True, length=8)
        source_axes[-1].tick_params(axis='x', which='minor', bottom=True, labelbottom=False, length=3)
        source_axes[-1].xaxis.get_offset_text().set_visible(False)
        for tick_label in source_axes[-1].get_xticklabels():
            tick_label.set_rotation(0)
            tick_label.set_ha('center')

    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')
    if legend_handles:
        legend_ax.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc='center',
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=max(legend_size - 4, 10),
            handlelength=2.0,
            columnspacing=1.0,
        )
    pass
    fig.savefig(
        os.path.join(SCRATCHDIR, "Chiles/images/Chiles_Strength_plots.png"),
        dpi=300,
        transparent=True,
        bbox_inches='tight',
    )


if __name__ == '__main__':
     main(iargs=sys.argv)
