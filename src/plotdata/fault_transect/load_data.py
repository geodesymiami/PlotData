#!/usr/bin/env python3
"""Resolve HDFEOS5 inputs and derive a masked velocity grid per period."""

import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityGrid:
    data: object          # 2D np.ndarray with NaN where masked
    attr: dict
    lats: object          # 1D np.ndarray
    lons: object          # 1D np.ndarray
    start_date: str
    end_date: str
    eos_file: str
    project: str          # e.g. EtnaSenA44
    source: str           # 'mintpy' or 'miaplpy'
    unit: str


def resolve_input(path):
    """Resolve an input (he5 file or mintpy/miaplpy dir) to (eos_file, project, source)."""
    from plotdata.helper_functions import prepend_scratchdir_if_needed, get_eos5_file

    full_path = prepend_scratchdir_if_needed(path)
    if os.path.isfile(full_path) and full_path.endswith('.he5'):
        eos_file = full_path
    else:
        eos_file = get_eos5_file(full_path, os.getenv('SCRATCHDIR'))

    parts = os.path.normpath(eos_file).split(os.sep)
    source = 'miaplpy' if any('miaplpy' in p for p in parts) else 'mintpy'

    project = None
    for part in parts:
        for keyword in ('SenAT', 'SenDT', 'SenA', 'SenD', 'CskAT', 'CskDT', 'CskA', 'CskD'):
            if keyword in part:
                project = part
                break
        if project:
            break
    if not project:
        project = os.path.basename(os.path.dirname(os.path.dirname(eos_file))) or 'project'

    return eos_file, project, source


def resolve_dataset_label(data_input, eos_file=None):
    """Return ``ascending``, ``descending``, ``horizontal``, or ``vertical``.

    Uses HDF attributes when available, then EOS filename tokens (``S1_horz_``,
    ``S1_vert_``), then path/project names (``SenA``/``SenD``).
    """
    paths = [str(data_input or ''), str(eos_file or '')]
    combined = ' '.join(paths)
    base = os.path.basename(eos_file or data_input or '').lower()

    if re.search(r'(?:^|[_-])horz(?:[_-]|$)', base):
        return 'horizontal'
    if re.search(r'(?:^|[_-])vert(?:[_-]|$)', base):
        return 'vertical'

    path_lower = combined.lower()
    if re.search(r'(?:^|/)(?:horz|horizontal)(?:/|_|\.|$)', path_lower):
        return 'horizontal'
    if re.search(r'(?:^|/)(?:vert|vertical|up)(?:/|_|\.|$)', path_lower):
        return 'vertical'

    file_to_read = eos_file if eos_file and os.path.isfile(eos_file) else None
    if file_to_read:
        try:
            from mintpy.utils import readfile

            attr = readfile.read_attribute(file_to_read)
            disp = (attr.get('displacement_type') or '').lower()
            if disp in ('horizontal', 'vertical'):
                return disp
            proc = (attr.get('processing_type') or '').lower()
            if 'horizontal' in proc:
                return 'horizontal'
            if 'vertical' in proc:
                return 'vertical'
            orbit = (attr.get('ORBIT_DIRECTION') or attr.get('ORBIT_DIRECTION_SECOND') or '')
            orbit = orbit.lower()
            if orbit in ('ascending', 'descending'):
                return orbit
        except Exception:
            pass

    s = path_lower
    tokens = set(re.findall(r'[a-z0-9]+', s))
    asc_tokens = {'sena', 'cska', 'senat', 'cskat', 'asc'}
    desc_tokens = {'send', 'cskd', 'sendt', 'cskdt', 'desc'}
    if tokens & asc_tokens:
        return 'ascending'
    if tokens & desc_tokens:
        return 'descending'
    for token in asc_tokens:
        if token in s:
            return 'ascending'
    for token in desc_tokens:
        if token in s:
            return 'descending'
    return ''


def default_output_dir(eos_file, project, source):
    """Default output dir: sibling of the processing dir at project level.

    EtnaSenA44/mintpy/...      -> <project_base>/EtnaSenA44/transects_mintpy
    EtnaSenA44/miaplpy/net_X/. -> <project_base>/EtnaSenA44/transects_miaplpy
    """
    parts = os.path.normpath(os.path.dirname(eos_file)).split(os.sep)
    for i, part in enumerate(parts):
        if part == project:
            project_dir = os.sep.join(parts[:i + 1])
            return os.path.join(project_dir, f'transects_{source}')
    return os.path.join(os.path.dirname(eos_file), f'transects_{source}')


def full_date_span(eos_file):
    from mintpy.utils import readfile

    attr = readfile.read_attribute(eos_file)
    return attr['START_DATE'], attr['END_DATE']


def load_velocity_grid(eos_file, project, source, start_date, end_date, work_dir,
                       mask_thresh=0.55, consecutive_start=False, gap_start=False,
                       force=False, tag_string=''):
    """timeseries2velocity for the period, mask by temporal coherence, return grid.

    Follows the plot_data pipeline (ProcessData._process_data) without
    modifying it: ts2v on the he5 file, then NaN-mask from the HDFEOS quality
    temporalCoherence dataset.
    """
    from mintpy.utils import readfile
    from mintpy.cli import timeseries2velocity as ts2v
    from plotdata.fault_transect.periods import snap_period_dates
    from plotdata.fault_transect.cache import cache_is_fresh

    os.makedirs(work_dir, exist_ok=True)
    tag_string = (tag_string or '').strip()
    if not tag_string:
        raise ValueError('fault tag is required for velocity cache filenames')
    start_date, end_date = snap_period_dates(
        eos_file, start_date, end_date,
        consecutive_start=consecutive_start, gap_start=gap_start)

    vel_file = os.path.join(work_dir,
                            f'velocity_{tag_string}_{start_date}_{end_date}.h5')
    if not force and cache_is_fresh(vel_file, eos_file):
        print(f'Using cached velocity grid: {vel_file}')
    else:
        cmd = f'{eos_file} --start-date {start_date} --end-date {end_date} --output {vel_file}'
        ts2v.main(cmd.split())

    data, attr = readfile.read(vel_file)
    if 'Y_FIRST' not in attr:
        raise ValueError(f'{vel_file} is not geocoded (no Y_FIRST); '
                         'plot_fault_transect.py requires geocoded HDFEOS5 input')

    data = np.asarray(data, dtype=float)
    data[data == 0.0] = np.nan

    try:
        coh, _ = readfile.read(eos_file, datasetName='HDFEOS/GRIDS/timeseries/quality/temporalCoherence')
        if coh.shape == data.shape:
            data[coh < mask_thresh] = np.nan
        else:
            print(f'WARNING: temporalCoherence shape {coh.shape} != velocity shape {data.shape}; skipping mask')
    except Exception as exc:
        print(f'WARNING: could not read temporalCoherence for masking: {exc}')

    from plotdata.fault_transect.sampling import grid_latlon_vectors
    lats, lons = grid_latlon_vectors(attr)

    unit = 'cm/yr'
    scale = 100.0 if attr.get('UNIT', 'm/year').startswith('m') else 1.0
    data = data * scale

    return VelocityGrid(data=data, attr=attr, lats=lats, lons=lons,
                        start_date=start_date, end_date=end_date,
                        eos_file=eos_file, project=project, source=source, unit=unit)
