#!/usr/bin/env python3
"""
Raw scratch runner for PlotData objects.

Edit the CONFIG block and run:

    python scripts/raw_plotdata_scratch.py

This intentionally avoids argparse. It is meant as a temporary notebook-like
script where file paths and plot choices are edited by hand.
"""

from pathlib import Path
from types import SimpleNamespace
import os
import sys

Path("/tmp/plotdata_mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/plotdata_mplconfig")

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CONFIG: this is the part you normally edit.
# ---------------------------------------------------------------------------

# If $SCRATCHDIR is already set in your shell, leave SCRATCHDIR = None.
# If not, set it to the folder that contains project directories such as
# MaunaLoaSenAT124, ChilesSenDT142, etc.
SCRATCHDIR = "/Users/giacomo/onedrive/scratch"

# Data dirs or .he5 files. Absolute paths are fine. Relative paths are resolved
# the same way as plot_data.py, through $SCRATCHDIR.
DATA_DIRS = [
    # Used only by the normal ProcessData route.
    # Manual file mode below bypasses this.
]

# Manual file mode feeds already-made velocity/geometry files directly into
# DataExtractor. This is the quickest path when you already have *_msk.h5.
USE_MANUAL_FILES = True
MANUAL_ASCENDING_VELOCITY = "/Users/giacomo/onedrive/scratch/Chiles/SenAT120/20150731_20260328/geo_velocity_msk.h5"
MANUAL_DESCENDING_VELOCITY = "/Users/giacomo/onedrive/scratch/Chiles/SenDT142/20150731_20260328/geo_velocity_msk.h5"
MANUAL_ASCENDING_GEOMETRY = "/Users/giacomo/onedrive/scratch/Chiles/SenAT120/geo_geometryRadar.h5"
MANUAL_DESCENDING_GEOMETRY = "/Users/giacomo/onedrive/scratch/Chiles/SenDT142/geo_geometryRadar.h5"

# Add these if you want the actual "timeseries" panel. Velocity files alone
# can draw maps, but they do not contain the full pixel time series.
MANUAL_ASCENDING_EOS = None
MANUAL_DESCENDING_EOS = None

# Direct model directory override. Leave as None to derive it from MODEL:
#   /project_root/start_end/<model_names_joined_with_underscore>
MANUAL_MODEL_DIR = None

# Common built-in templates:
#   "ascending", "descending", "velocity", "timeseries", "vectors",
#   "default", "default_with_seismicity", "model", "model_ascending",
#   "model_descending", "seismicmap", "test"
TEMPLATE = "model_descending"

# Use this when you want a one-off layout without editing PlotTemplate.
# Each cell is a plot name, optionally suffixed with ".point" and/or ".section".
# Examples:
#   [["velocity_ascending.point.section", "velocity_descending.point.section"]]
#   [["timeseries"]]
#   [["velocity_ascending.section"], ["vectors"]]
CUSTOM_LAYOUT = [
    ["velocity_descending.section"],
    ["model_descending.section"],
    ["profile_descending"],
]

# Dates are strings in YYYYMMDD format. Leave as None to read from the input
# files. For multiple periods, use matching lists:
#   START_DATES = ["20220101", "20230101"]
#   END_DATES   = ["20221231", "20231231"]
START_DATES = ["20150731"]
END_DATES = ["20260328"]

# Point for timeseries and ".point" maps: [lat, lon]
LALO = [0.7407, -77.8892]

# Reference point: [lat, lon]
REF_LALO = [0.84969, -77.9580]

# Section line for ".section", profiles, and vectors.
# Shape used by plotters: [[lon0, lon1], [lat0, lat1]]
# You may also use a float latitude to auto-create an east-west section.
LINE = [[-77.95, -77.84], [0.7407, 0.7407]]

# Plot style for velocity/model maps: "pixel", "scatter", or "ifgram".
STYLE = "pixel"
UNIT = "cm"
CMAP = "jet"
VLIM = [-30, 30]  # e.g. [-5, 5]
CONTOUR = 10
NO_DEM = False
NO_COLORBAR = False
ZOOM = None
SUBSET = "0.78:-77.96,0.68:-77.80"  # "lat0:lon0,lat1:lon1"

# Optional add-ons.
SEISMICITY = None  # minimum magnitude, e.g. 3.0
FOCAL = None       # minimum magnitude for focal mechanisms
VOLCANO = False
ADD_EVENT = []     # ["20220101", ...]
EVENT_MAGNITUDE = []

# Model folder support. This follows ProcessData._read_model_input().
MODEL = ["okada"]  # e.g. ["mogi"] or ["okada"]
NO_SOURCES = False
FULLRES = True
NORM = False
DENOISE = None

# Vector plot controls.
RESAMPLE_VECTOR = 4
VECTOR_LEGEND = "mean_vector"  # "mean_vector" or "colorbar"
VERTICAL_EXAG = 1

# Save/show behavior.
SAVE_FIGURES = False
SHOW_FIGURES = True
OUTDIR = Path("scratch_plots")
DPI = 300
FLAG_SAVE_AXIS = False
SAVE_AXIS_PANELS = False
FORCE_SOURCE_OVERLAY = True


# ---------------------------------------------------------------------------
# Plumbing below this line.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plotdata.cli.plot_data import populate_dates
from plotdata.helper_functions import (
    latlon_to_utm_zone,
    meters_to_lon_deg,
    read_best_values,
    utm_to_latlon,
)
from plotdata.objects.get_methods import DataExtractor
from plotdata.objects.plot_properties import PlotRenderer, PlotTemplate
from plotdata.objects.plotters import (
    EarthquakePlot,
    Mogi,
    Okada,
    Penny,
    ProfilePlot,
    Spheroid,
    TimeseriesPlot,
    VectorsPlot,
    VelocityPlot,
)
from plotdata.objects.process_data import ProcessData


PLOTTER_MAP = {
    # plot_name: class to render it, and ProcessData attributes needed to build data
    "model_ascending": {"class": VelocityPlot, "attributes": ["ascending_model"]},
    "model_descending": {"class": VelocityPlot, "attributes": ["descending_model"]},
    "profile_ascending": {"class": ProfilePlot, "attributes": ["ascending_model", "ascending"]},
    "profile_descending": {"class": ProfilePlot, "attributes": ["descending_model", "descending"]},
    "vectors": {"class": VectorsPlot, "attributes": ["horizontal", "vertical"]},
    "vertical": {"class": VelocityPlot, "attributes": ["vertical"]},
    "velocity_ascending": {"class": VelocityPlot, "attributes": ["ascending"]},
    "velocity_descending": {"class": VelocityPlot, "attributes": ["descending"]},
    "horizontal": {"class": VelocityPlot, "attributes": ["horizontal"]},
    "timeseries": {"class": TimeseriesPlot, "attributes": ["eos_file_ascending", "eos_file_descending"]},
    "seismicmap": {"class": VelocityPlot, "attributes": ["ascending_geometry", "descending_geometry"]},
    "seismicity": {"class": EarthquakePlot, "attributes": ["ascending", "descending"]},
}


def make_inps():
    """Create an argparse-like object with the attributes PlotData expects."""
    if SCRATCHDIR:
        os.environ["SCRATCHDIR"] = str(Path(SCRATCHDIR).expanduser())
    elif "SCRATCHDIR" not in os.environ:
        raise RuntimeError("Set SCRATCHDIR in this script or export it before running.")

    if not USE_MANUAL_FILES and not DATA_DIRS:
        raise RuntimeError("Add at least one path to DATA_DIRS in the CONFIG block.")

    start_dates = [] if START_DATES is None else list(START_DATES)
    end_dates = [] if END_DATES is None else list(END_DATES)
    outdir = Path(OUTDIR).expanduser()
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir

    vmin = min(VLIM) if VLIM else None
    vmax = max(VLIM) if VLIM else None

    return SimpleNamespace(
        data_dir=[str(Path(p).expanduser()) for p in DATA_DIRS],
        template=TEMPLATE,
        start_date=start_dates,
        end_date=end_dates,
        period=None,
        lalo=LALO,
        ref_lalo=REF_LALO,
        line=LINE,
        region=None,
        subset=SUBSET,
        dem_file=None,
        line_file=None,
        model=MODEL,
        no_sources=NO_SOURCES,
        fullres=FULLRES,
        norm=NORM,
        denoise=DENOISE,
        mask_thresh=0.55,
        mask_vmin=0.55,
        mask=None,
        unit=UNIT,
        style=STYLE,
        show_flag=SHOW_FIGURES,
        font_size=10,
        dpi=DPI,
        cmap=CMAP,
        colorbar_size=0.3,
        no_colorbar=NO_COLORBAR,
        zoom=ZOOM,
        vector_legend=VECTOR_LEGEND,
        vertical_exag=VERTICAL_EXAG,
        vlim=VLIM,
        vmin=vmin,
        vmax=vmax,
        contour=CONTOUR,
        colorbar="viridis",
        iso_color="black",
        contour_linewidth=0.5,
        inline=False,
        resolution="01m",
        color="black",
        scatter_marker_size=10,
        no_dem=NO_DEM,
        interpolate=False,
        no_shade=False,
        offset=0,
        save="png" if SAVE_FIGURES else None,
        flag_save_axis=FLAG_SAVE_AXIS,
        outdir=str(outdir),
        flag_save_gbis=False,
        flag_gps=False,
        gps_scale_fac=500,
        gps_key_length=4,
        gps_unit="cm",
        gps_dir=None,
        gps_list_file=None,
        seismicity=SEISMICITY,
        focal=FOCAL,
        event_magnitude=EVENT_MAGNITUDE,
        add_event=ADD_EVENT,
        volcano=VOLCANO,
        resample_vector=RESAMPLE_VECTOR,
        window_size=3,
        lat_step=-0.0002,
        lat=None,
        lon=None,
        polygon=None,
        sources=None,
        website="usgs",
    )


def make_template(inps):
    template = PlotTemplate(inps.template)
    if CUSTOM_LAYOUT is not None:
        template.layout = normalize_layout(CUSTOM_LAYOUT)
    return template


def normalize_layout(layout):
    """Accept 1-D or 2-D layouts and flatten accidental single-item cells."""
    if all(isinstance(row, str) for row in layout):
        layout = [[row] for row in layout]

    normalized = []
    for row in layout:
        if isinstance(row, str):
            row = [row]

        normalized_row = []
        for cell in row:
            if isinstance(cell, list):
                if len(cell) != 1 or not isinstance(cell[0], str):
                    raise ValueError(
                        "CUSTOM_LAYOUT cells must be strings, e.g. "
                        '[["velocity_descending.section", "model_descending.section"]].'
                    )
                cell = cell[0]
            normalized_row.append(cell)
        normalized.append(normalized_row)
    return normalized


def pad_layout(layout):
    """Pad ragged subplot_mosaic rows with a dummy cell.

    Do not use Matplotlib's "." sentinel here because PlotRenderer checks
    element.split(".")[0], and "." becomes an empty string that matches every
    plotter name.
    """
    if not layout:
        return layout

    width = max(len(row) for row in layout)
    return [row + ["__blank__"] * (width - len(row)) for row in layout]


def _first_existing(paths):
    for path in paths:
        if path and Path(path).exists():
            return str(Path(path))
    return None


def _manual_project_base_dir():
    """Infer the project root that contains SenAT/SenDT date folders."""
    if MANUAL_ASCENDING_VELOCITY:
        asc_dir = Path(MANUAL_ASCENDING_VELOCITY).expanduser().parent
        if len(asc_dir.parents) > 1:
            return asc_dir.parents[1]
    if MANUAL_DESCENDING_VELOCITY:
        desc_dir = Path(MANUAL_DESCENDING_VELOCITY).expanduser().parent
        if len(desc_dir.parents) > 1:
            return desc_dir.parents[1]
    return Path(os.environ["SCRATCHDIR"])


def _manual_model_dir(start_date=None, end_date=None):
    """Resolve the manual model folder from override or MODEL list."""
    if MANUAL_MODEL_DIR:
        return Path(MANUAL_MODEL_DIR).expanduser()

    if not MODEL:
        return None

    start = start_date or (START_DATES[0] if START_DATES else None)
    end = end_date or (END_DATES[0] if END_DATES else None)
    if not start or not end:
        return None

    model_name = "_".join(MODEL) if isinstance(MODEL, (list, tuple)) else str(MODEL)
    return _manual_project_base_dir() / f"{start}_{end}" / model_name


def _manual_model_attrs():
    """Return ascending_model/descending_model dicts for a direct VSM folder."""
    attrs = {}
    model_dir = _manual_model_dir()
    if not model_dir:
        return attrs

    synth_files = sorted(model_dir.glob("VSM_synth_sar*.csv"))
    if not synth_files:
        return attrs

    input_file = model_dir / "VSM_input.txt"
    input_text = input_file.read_text() if input_file.exists() else ""
    lower_text = input_text.lower()

    if len(synth_files) == 1:
        synth_file = str(synth_files[0])
        if "send" in lower_text or "cskd" in lower_text or "desc" in lower_text:
            attrs["descending_model"] = {"descending": synth_file}
        elif "senat" in lower_text or "sena" in lower_text or "cska" in lower_text or "asc" in lower_text:
            attrs["ascending_model"] = {"ascending": synth_file}
        else:
            attrs["ascending_model"] = {"ascending": synth_file}
        return attrs

    attrs["ascending_model"] = {"ascending": str(synth_files[0])}
    attrs["descending_model"] = {"descending": str(synth_files[1])}
    return attrs


def _filter_source_params(params):
    source_shapes = (
        ("mogi", ("xcen", "ycen")),
        ("spheroid", ("xcen", "ycen", "s_axis_max", "ratio", "strike", "dip")),
        ("penny", ("xcen", "ycen", "radius")),
        ("okada", ("ytlc", "xtlc", "length", "width", "strike", "dip")),
    )

    for _, keys in source_shapes:
        if all(key in params for key in keys):
            return {key: params[key] for key in keys}
    return None


def _manual_sources():
    """Read VSM source parameters for manual model folders."""
    model_dir = _manual_model_dir()
    if NO_SOURCES or not model_dir:
        return None

    source_file = model_dir / "VSM_best.csv"
    if not source_file.exists():
        source_file = model_dir / "VSM_mean.csv"
    if not source_file.exists():
        return None

    if REF_LALO:
        ref_lat, ref_lon = REF_LALO
    elif LALO:
        ref_lat, ref_lon = LALO
    else:
        ref_lat = (LINE[1][0] + LINE[1][1]) / 2
        ref_lon = (LINE[0][0] + LINE[0][1]) / 2

    zone_number, hemisphere = latlon_to_utm_zone(ref_lat, ref_lon)
    converted = {}

    raw_sources = read_best_values(str(source_file))
    print("Raw source keys from VSM:", raw_sources)

    for source_id, params in raw_sources.items():
        params = dict(params)

        if "xcen" in params and "ycen" in params:
            lat, lon = utm_to_latlon(params["xcen"], params["ycen"], zone_number, hemisphere)
            params["ycen"] = float(lat)
            params["xcen"] = float(lon)

        if "xtlc" in params and "ytlc" in params:
            lat, lon = utm_to_latlon(params["xtlc"], params["ytlc"], zone_number, hemisphere)
            params["ytlc"] = float(lat)
            params["xtlc"] = float(lon)

        source_lat = params.get("ycen", params.get("ytlc", ref_lat))
        for key in ("radius", "s_axis_max", "length", "width"):
            if key in params:
                params[key] = meters_to_lon_deg(params[key], source_lat)

        filtered = _filter_source_params(params)
        if filtered:
            converted[f"{model_dir.name}_{source_id}"] = filtered

    return converted or None


def overlay_sources(figures, sources):
    """Force source overlays onto rendered map axes for scratch/manual mode."""
    if not FORCE_SOURCE_OVERLAY or not sources:
        return

    source_type = {
        "mogi": {"class": Mogi, "attributes": {"xcen", "ycen"}},
        "spheroid": {"class": Spheroid, "attributes": {"xcen", "ycen", "s_axis_max", "ratio", "strike", "dip"}},
        "penny": {"class": Penny, "attributes": {"xcen", "ycen", "radius"}},
        "okada": {"class": Okada, "attributes": {"ytlc", "xtlc", "length", "width", "strike", "dip"}},
    }

    for fig in figures:
        for ax in fig.get_axes():
            label = ax.get_label()
            if "__blank__" in label or "profile" in label or "timeseries" in label:
                continue
            for name, params in sources.items():
                keys = set(params)
                for source in source_type.values():
                    if keys == source["attributes"]:
                        source["class"](ax, **params)
                        print(f"Overlayed source {name} on {label}: {params}")
                        break


def build_manual_process(inps, template, start_date, end_date):
    """Build a ProcessData-like object from direct file paths."""
    process = SimpleNamespace(**vars(inps))
    process.start_date = start_date
    process.end_date = end_date
    process.layout = template.layout

    process.ascending = MANUAL_ASCENDING_VELOCITY
    process.descending = MANUAL_DESCENDING_VELOCITY
    process.ascending_geometry = MANUAL_ASCENDING_GEOMETRY
    process.descending_geometry = MANUAL_DESCENDING_GEOMETRY
    process.eos_file_ascending = MANUAL_ASCENDING_EOS
    process.eos_file_descending = MANUAL_DESCENDING_EOS

    asc_dir = Path(MANUAL_ASCENDING_VELOCITY).expanduser().parent
    desc_dir = Path(MANUAL_DESCENDING_VELOCITY).expanduser().parent
    process.ascending_downsampled = _first_existing(asc_dir.glob("*downsampled*.h5"))
    process.descending_downsampled = _first_existing(desc_dir.glob("*downsampled*.h5"))

    process.directory = str(asc_dir.parents[1]) if len(asc_dir.parents) > 1 else str(asc_dir)
    process.project = Path(process.directory).name
    process.horizontal = None
    process.vertical = None
    process.sources = _manual_sources()

    process.file_info = {
        "ascending": {
            "mask_file": _first_existing(asc_dir.glob("*mask*.h5")),
            "out_vel_file": MANUAL_ASCENDING_VELOCITY,
        },
        "descending": {
            "mask_file": _first_existing(desc_dir.glob("*mask*.h5")),
            "out_vel_file": MANUAL_DESCENDING_VELOCITY,
        },
    }

    for name, value in _manual_model_attrs().items():
        setattr(process, name, value)

    template.update_layout(PLOTTER_MAP, process)
    template.layout = pad_layout(template.layout)
    process.layout = template.layout

    extracted = DataExtractor(PLOTTER_MAP, process)
    return process, extracted


def build_one_period(inps, template, start_date, end_date):
    """Run ProcessData -> DataExtractor for one period."""
    if USE_MANUAL_FILES:
        return build_manual_process(inps, template, start_date, end_date)

    process = ProcessData(inps, template.layout, start_date, end_date)
    process.process()

    template.update_layout(PLOTTER_MAP, process)
    template.layout = pad_layout(template.layout)
    process.layout = template.layout

    extracted = DataExtractor(PLOTTER_MAP, process)
    return process, extracted


def render(extracted, template):
    renderer = PlotRenderer(extracted, template)
    figures = renderer.render()
    figures = figures if isinstance(figures, list) else [figures]
    for fig in figures:
        for ax in fig.get_axes():
            if ax.get_label() == "__blank__":
                ax.set_visible(False)
    return figures


def save_figures(figures, process, inps):
    outdir = Path(inps.outdir).expanduser()
    root = outdir / str(process.project) / "images" / f"{process.start_date}_{process.end_date}"
    root.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, fig in enumerate(figures, start=1):
        axis_label = fig.get_axes()[0].get_label().split(".")[0]
        suffix = axis_label if len(figures) > 1 else inps.template
        path = root / f"{process.project}_{suffix}_{process.start_date}_{process.end_date}_{i}.png"
        fig.savefig(path, bbox_inches="tight", dpi=inps.dpi, transparent=True)
        paths.append(path)
        print(f"Saved {path}")
    return paths


def save_axis_panels(figures, process, inps):
    """Save each axis from a mosaic figure as a separate PNG."""
    outdir = Path(inps.outdir).expanduser()
    root = outdir / str(process.project) / "images" / f"{process.start_date}_{process.end_date}" / "axes"
    root.mkdir(parents=True, exist_ok=True)

    paths = []
    for fig in figures:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for i, ax in enumerate(fig.get_axes(), start=1):
            label = ax.get_label() or f"axis_{i}"
            if label == "__blank__":
                continue
            if label.startswith("<"):
                label = f"axis_{i}"
            clean_label = label.replace(".", "_").replace(" ", "_")
            bbox = ax.get_tightbbox(renderer).expanded(1.05, 1.12)
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
            path = root / f"{process.project}_{clean_label}_{process.start_date}_{process.end_date}.png"
            fig.savefig(path, bbox_inches=bbox, dpi=inps.dpi, transparent=True)
            paths.append(path)
            print(f"Saved axis {path}")
    return paths


def main():
    inps = populate_dates(make_inps())

    all_datasets = []
    all_figures = []

    for start_date, end_date in zip(inps.start_date, inps.end_date):
        template = make_template(inps)
        process, extracted = build_one_period(inps, template, start_date, end_date)
        print("Layout after filtering:", process.layout)
        print("Extracted datasets:", sorted(extracted.dataset.keys()))
        print("Manual model folder:", _manual_model_dir(start_date, end_date))
        print("Sources:", process.sources)

        # Raw extracted data lives here. Typical shapes:
        #   extracted.dataset["velocity_ascending"]["data"] -> 2-D numpy array
        #   extracted.dataset["timeseries"]["ascending"]["data"] -> 1-D array
        #   extracted.dataset["timeseries"]["ascending"]["dates"] -> datetimes
        #   extracted.dataset["vectors"]["horizontal"]["data"] -> 2-D array
        all_datasets.append(extracted.dataset)

        figures = render(extracted, template)
        overlay_sources(figures, process.sources)
        all_figures.extend(figures)

        if SAVE_FIGURES:
            save_figures(figures, process, inps)
            if SAVE_AXIS_PANELS:
                save_axis_panels(figures, process, inps)

    if SHOW_FIGURES:
        plt.show()
    else:
        for fig in all_figures:
            plt.close(fig)

    return all_datasets, all_figures


if __name__ == "__main__":
    main()
