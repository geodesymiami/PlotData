#!/usr/bin/env python3
############################################################
# Program is part of PlotData                              #
# Fault offset (map) and cross-fault profile plotting      #
############################################################
"""Plot across-fault displacement offset and fault-perpendicular profiles.

Takes a fault KMZ and 1-4 HDFEOS5 timeseries inputs (S1_*.he5 files or
mintpy/miaplpy directories). Produces a map plot (fault trace + sampling boxes
colored by across-fault offset + offset-vs-distance curve) and/or profile
plots (fault-perpendicular profiles at regular along-fault spacing). All
plotted data is always exported to companion .txt files and an index.html is
generated.
"""

import os
import re
import sys
import argparse

EXAMPLE = """example:
  plot_fault_transect.py PFS_Pernicana_faults_system_.kmz EtnaSenA44/mintpy --fault-segment 1,2,4,5,6,7,8,9,10,11 --period 20141001:20181224,20181225:20201224,20211225:20260701
  plot_fault_transect.py PFS_Pernicana_faults_system__joint.kmz EtnaSenA44/mintpy --period 20141020:20260626 --tag Pernicana
  plot_fault_transect.py FiandacaFault_FA.kmz EtnaSenA44/mintpy --map-stack-axis lon --period 20141001:20181222,20181228:20201225,20201225:20260701 --vlim -2 2
  plot_fault_transect.py fault.kmz EtnaSenA44/mintpy --fault-segment 2-8 --plot-type profile --profile-count 10 --plot-layout subplot --cloud-profiles 2
  plot_fault_transect.py fault_joint.kmz EtnaSenA44/mintpy --plot-type profile --plot-layout stacked --period 20141020:20181231,20190101:20260626 --display
"""


def create_parser():
    parser = argparse.ArgumentParser(
        description='Plot across-fault displacement offset (map) and fault-perpendicular profiles',
        epilog=EXAMPLE, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('fault_file', help='Fault trace KMZ/KML (single- or multi-segment)')
    parser.add_argument('data_dir', nargs='*', help='1-4 inputs: S1_*.he5 file or mintpy/miaplpy directory')

    fault = parser.add_argument_group('Fault handling')
    fault.add_argument('--fault-segment', dest='fault_segment', type=str, default='all',
                       help='Segments to use, in along-fault order: all, 3, 2-8, 1,2,4-11. '
                            'Numbers match placemark labels (PFS3 -> 3) when names have unique '
                            'suffixes; use --fault-segment-by index for 0-based KMZ read order.')
    fault.add_argument('--fault-segment-by', dest='fault_segment_by', type=str, default='auto',
                       choices=['auto', 'label', 'index'],
                       help='How to interpret --fault-segment numbers (default: %(default)s)')
    fault.add_argument('--flip-fault', dest='flip_fault', action='store_true', help='Reverse along-fault direction (swaps left/right and along-km origin)')

    data = parser.add_argument_group('Data selection')
    data.add_argument('--period', dest='period', nargs='*', default=[], metavar='YYYYMMDD:YYYYMMDD', help='Period(s); repeatable or comma-separated (default: full span)')
    data.add_argument('--mask-thresh', dest='mask_vmin', type=float, default=0.55, help='Coherence threshold for masking (default: %(default)s)')
    data.add_argument('--unit', dest='unit', default='cm/yr', help='Display unit (default: %(default)s)')

    geom = parser.add_argument_group('Sampling geometry')
    geom.add_argument('--along-step', dest='along_step', type=float, default=1.0, help='Spacing of sampling points along the fault in km (default: %(default)s)')
    geom.add_argument('--along-start', dest='along_start', type=float, default=0.0, help='Start distance along fault in km (default: %(default)s)')
    geom.add_argument('--along-end', dest='along_end', type=float, default=None, help='End distance along fault in km (default: fault end)')
    geom.add_argument('--perp-width', dest='perp_width', type=float, default=0.5, help='Search-zone width on each side of the fault in km (default: %(default)s)')
    geom.add_argument('--perp-offset', dest='perp_offset', type=float, default=0.5,
                      help='Gap excluded next to the fault before the search zone on each side, in km (default: %(default)s)')
    geom.add_argument('--sample-method', dest='sample_method', choices=['mean', 'median', 'nearest'], default='mean', help='Combining pixels in a search box (default: %(default)s)')
    geom.add_argument('--interpolation', dest='interpolation', choices=['nearest', 'linear', 'cubic'], default='nearest', help='Value extraction along profile lines (default: %(default)s)')

    plot = parser.add_argument_group('Plot selection and layout')
    plot.add_argument('--plot-type', dest='plot_type', choices=['map', 'profile', 'both'], default='both', help='What to plot (default: %(default)s)')
    plot.add_argument('--reference-side', dest='reference_side', choices=['left', 'right'], default='left', help='Reference side of fault; offset = reference - other (default: %(default)s)')
    plot.add_argument('--plot-layout', dest='plot_layout', choices=['separate', 'subplot', 'stacked', '3d'], default='stacked', help='Profile arrangement (default: %(default)s)')
    plot.add_argument('--profile-spacing', dest='profile_spacing', type=float, default=None, help='Spacing between profiles in km (default: --along-step)')
    plot.add_argument('--profile-length', dest='profile_length', type=float, default=4.0, help='Profile length across the fault in km (default: %(default)s)')
    plot.add_argument('--profile-count', dest='profile_count', type=int, default=None, help='Fixed number of profiles (overrides --profile-spacing)')
    plot.add_argument('--cloud-profiles', dest='cloud_profiles', type=int, default=0, help='Adjacent profiles per side as thin gray lines (default: %(default)s)')
    plot.add_argument('--profile-lines', dest='profile_lines', action='store_true',
                      help='Connect profile samples with lines (default: dots only)')
    plot.add_argument('--stack-offset', dest='stack_offset', type=float, default=None,
                      help='Vertical step between profiles in stacked profile layout (default: auto)')
    plot.add_argument('--subplot-cols', dest='subplot_cols', type=int, default=1, help='Columns for subplot layout (default: %(default)s)')
    plot.add_argument('--period-layout', dest='period_layout', choices=['auto', 'side-by-side', 'separate-page'], default='auto', help='Arrangement for multiple periods (default: %(default)s)')

    style = parser.add_argument_group('Plot parameters')
    style.add_argument('--vlim', dest='vlim', nargs=2, type=float, metavar=('VMIN', 'VMAX'),
                       default=None, help='Offset colorscale limits on the map (default: auto)')
    style.add_argument('--auto-colorscale', dest='auto_colorscale', action='store_true',
                       help='Use one automatic symmetric offset colorscale across all periods')
    style.add_argument('--colormap', dest='colormap', default='jet', metavar='COLORMAP',
                       help='Colormap: matplotlib names, MintPy names (cmy, dismph, temperature, vik, ...), '
                            'coherence (γ scale), or suffix _r / _truncate (default: %(default)s)')
    style.add_argument('--cmap-vlist', dest='cmap_vlist', nargs=3, type=float, default=None,
                       metavar=('VMIN', 'VMID', 'VMAX'),
                       help='Truncation limits for *_truncate colormaps (default: 0 0.7 1)')
    style.add_argument('--font-size', dest='font_size', type=int, default=10, help='Font size (default: %(default)s)')
    style.add_argument('--dpi', dest='dpi', type=int, default=300, help='Figure DPI (default: %(default)s)')
    style.add_argument('--title-position', dest='title_position',
                       choices=['upper-left', 'upper-right', 'lower-left', 'lower-right'],
                       default='upper-right',
                       help='Map title corner (default: %(default)s)')
    style.add_argument('--title-offset', dest='title_offset', nargs=2, type=float, default=None,
                       metavar=('LON_DEG', 'LAT_DEG'),
                       help='Nudge map title from its corner: east and north in deg (default: 0 0)')
    style.add_argument('--map-stack-axis', dest='map_stack_axis', choices=['lat', 'lon'],
                       default='lat',
                       help='Axis for stacked multi-period maps: lat (vertical, period 1 top) or '
                            'lon (horizontal, period 1 left)')
    style.add_argument('--map-stack-offset', dest='map_stack_offset', type=float, default=None,
                       help='Step in deg along --map-stack-axis between stacked map periods '
                            '(default: auto)')

    out = parser.add_argument_group('Output')
    out.add_argument('--save', dest='save', choices=['png', 'pdf'], default='png', help='Image format; images are always saved (default: %(default)s)')
    display = out.add_mutually_exclusive_group()
    display.add_argument('--display', dest='show_flag', action='store_true',
                         help='Open interactive figure windows (blocks until they are closed)')
    display.add_argument('--no-display', dest='show_flag', action='store_false',
                         help='Do not open interactive windows (default; figures are still saved)')
    out.set_defaults(show_flag=False)
    out.add_argument('--outdir', dest='outdir', type=str, default=None, help='Output directory (default: <project>/transects_mintpy or transects_miaplpy)')
    out.add_argument('--tag', dest='tag_string', type=str, default='',
                     help='Tag in output filenames (default: derived from fault KMZ name)')
    out.add_argument('--no-index', dest='no_index', action='store_true', help='Skip index.html generation')
    out.add_argument('--force', dest='force', action='store_true',
                     help='Recompute velocity grids and txt even when cached outputs are '
                          'newer than inputs (default: skip each step when up to date)')
    out.add_argument('--plots-only', dest='plots_only', action='store_true',
                     help='Build figures from existing txt only; do not read HDFEOS5 or rewrite txt')
    out.add_argument('--upload', dest='upload', action='store_true', default=False, help='Upload products (not implemented yet)')

    return parser


def parse_periods(tokens):
    """Parse --period tokens; each may contain comma-separated periods.

    Returns list of (start, end) strings. Empty list means full span.
    """
    from plotdata.fault_transect.periods import parse_period_chunks, validate_and_adjust_periods
    return validate_and_adjust_periods(parse_period_chunks(tokens))


def format_map_title(tag_string, start_date, end_date):
    """Deprecated: use format_period_display for labels and tag_string for titles."""
    from plotdata.fault_transect.periods import format_period_display
    return format_period_display(start_date, end_date)


def make_plot_options(inps, grid, *, vlim=None):
    """Build PlotOptions with fault tag title and formatted period label."""
    from plotdata.fault_transect.naming import format_fault_plot_title
    from plotdata.fault_transect.periods import format_period_display
    from plotdata.fault_transect.plot_api import PlotOptions

    title_off = inps.title_offset if inps.title_offset is not None else (0.0, 0.0)
    cmap_vlist = tuple(inps.cmap_vlist) if inps.cmap_vlist else None
    return PlotOptions(
        colormap=inps.colormap,
        cmap_vlist=cmap_vlist,
        vlim=vlim,
        font_size=inps.font_size,
        dpi=inps.dpi,
        unit=grid.unit,
        title=format_fault_plot_title(inps.tag_string),
        period_label=format_period_display(grid.start_date, grid.end_date),
        title_position=inps.title_position,
        title_offset_lon=title_off[0],
        title_offset_lat=title_off[1],
        map_stack_axis=inps.map_stack_axis,
        map_stack_offset=inps.map_stack_offset,
    )


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if not inps.data_dir:
        parser.error('at least one data input is required')
    if len(inps.data_dir) > 4:
        parser.error('at most 4 data inputs are supported (asc, desc, horz, vert)')
    if inps.plot_layout == '3d':
        parser.error('--plot-layout 3d is not implemented yet; use separate, subplot or stacked')
    if inps.plots_only and inps.plot_layout == 'separate':
        parser.error('--plots-only is not supported with --plot-layout separate; '
                     'use subplot or stacked')
    if inps.profile_spacing is None:
        inps.profile_spacing = inps.along_step

    try:
        inps.periods = parse_periods(inps.period)
    except ValueError as exc:
        parser.error(str(exc))
    if inps.vlim is not None and inps.vlim[0] >= inps.vlim[1]:
        parser.error('--vlim VMIN must be less than VMAX')
    if inps.map_stack_offset is not None and inps.map_stack_offset <= 0:
        parser.error('--map-stack-offset must be positive')

    from plotdata.fault_transect.naming import resolve_output_tag
    inps.tag_string = resolve_output_tag(inps.fault_file, inps.tag_string)

    inps.offset_summaries = []
    if inps.title_offset is not None:
        inps.offset_summaries.append({
            'kind': 'title_offset',
            'lon_deg': inps.title_offset[0],
            'lat_deg': inps.title_offset[1],
        })

    return inps


def _fault_lat_bounds(fault_segments_xy):
    lats = [lat for seg in fault_segments_xy for lat in seg['lats']]
    return min(lats), max(lats)


def _record_map_stack_summary(inps, project, n_periods, step_deg, manual):
    inps.offset_summaries.append({
        'kind': 'map_stack',
        'project': project,
        'n_periods': n_periods,
        'axis': inps.map_stack_axis,
        'step_deg': step_deg,
        'source': 'manual' if manual is not None else 'auto',
    })


def _record_profile_stack_summary(inps, project, stack_step, unit, manual):
    inps.offset_summaries.append({
        'kind': 'profile_stack',
        'project': project,
        'stack_step': stack_step,
        'unit': unit,
        'source': 'manual' if manual else 'auto',
    })


def _fault_cache_inputs(inps):
    """KMZ paths used for txt/figure staleness checks."""
    return getattr(inps, 'fault_cache_paths', (inps.fault_file,))


def _require_plots_only_cache(txt_path, eos_file, bracket_info, *fault_paths):
    from plotdata.fault_transect.cache import cache_is_fresh, parse_txt_header

    if not txt_path or not os.path.isfile(txt_path):
        raise SystemExit(f'ERROR: --plots-only requires cached txt at {txt_path or "(missing)"}')
    if not cache_is_fresh(txt_path, eos_file, *fault_paths):
        raise SystemExit(
            f'ERROR: {txt_path} is older than the HDFEOS5 or fault KMZ input; '
            f'rerun without --plots-only to update.')
    _, bracket = parse_txt_header(txt_path)
    if bracket != bracket_info:
        raise SystemExit(
            f'ERROR: {txt_path} was created with different options; '
            f'rerun without --plots-only to update.')


def _load_map_series_from_cache(inps, txt_path, eos_file, bracket_info):
    from plotdata.fault_transect.cache import txt_cache_hit
    from plotdata.fault_transect.export import read_offset_txt

    fault_paths = _fault_cache_inputs(inps)
    if inps.plots_only:
        _require_plots_only_cache(txt_path, eos_file, bracket_info, *fault_paths)
        series, _ = read_offset_txt(txt_path)
        print(f'Using cached map data from {txt_path}')
        return series
    if inps.force:
        return None
    if txt_path and txt_cache_hit(txt_path, eos_file, bracket_info, *fault_paths):
        series, _ = read_offset_txt(txt_path)
        print(f'Using cached map data from {txt_path}')
        return series
    return None


def _load_profiles_from_cache(inps, txt_path, eos_file, bracket_info):
    from plotdata.fault_transect.cache import txt_cache_hit
    from plotdata.fault_transect.export import read_profiles_txt

    fault_paths = _fault_cache_inputs(inps)
    if inps.plots_only:
        _require_plots_only_cache(txt_path, eos_file, bracket_info, *fault_paths)
        bundle, main_indices, _ = read_profiles_txt(txt_path)
        print(f'Using cached profile data from {txt_path}')
        return bundle, main_indices
    if inps.force:
        return None, None
    if txt_path and txt_cache_hit(txt_path, eos_file, bracket_info, *fault_paths):
        bundle, main_indices, _ = read_profiles_txt(txt_path)
        print(f'Using cached profile data from {txt_path}')
        return bundle, main_indices
    return None, None


def _select_profile_points(points, inps):
    """Pick main-profile sampling-point indices from spacing or count."""
    if not points:
        return []
    if inps.profile_count:
        n = min(inps.profile_count, len(points))
        idx = [int(round(k)) for k in
               [i * (len(points) - 1) / max(n - 1, 1) for i in range(n)]]
        return sorted(set(idx))
    stride = max(1, int(round(inps.profile_spacing / inps.along_step)))
    return list(range(0, len(points), stride))


def _side_by_side(inps, num_periods):
    """Decide whether periods share one figure (True) or get separate figures."""
    if num_periods < 2:
        return False
    if inps.period_layout == 'side-by-side':
        return True
    if inps.period_layout == 'separate-page':
        return False
    # auto: compact layouts side-by-side; full-page 'separate' layout per period
    return inps.plot_layout != 'separate' or inps.plot_type == 'map'


def _multi_period_stem(project, tag_string, plot_label, periods):
    from plotdata.fault_transect.naming import multi_period_stem
    return multi_period_stem(project, tag_string, plot_label, periods)


def _dates_from_product_txt(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'(\d{8})_(\d{8})$', stem)
    if not match:
        raise ValueError(f'Could not parse dates from cached file name: {path}')
    return match.group(1), match.group(2)


def _make_grid_stub(start_date, end_date, unit):
    from plotdata.fault_transect.load_data import VelocityGrid
    return VelocityGrid(data=None, attr={}, lats=None, lons=None,
                        start_date=start_date, end_date=end_date,
                        eos_file='', project='', source='', unit=unit)


def _process_input(inps, fault_segments, data_input, command):
    """Full pipeline for one data input: all periods, map + profile plots."""
    from plotdata.fault_transect.load_data import (
        resolve_input, default_output_dir, full_date_span, load_velocity_grid)
    from plotdata.fault_transect.fault_sampling import sample_points_segments
    from plotdata.fault_transect.offset import compute_offset_series
    from plotdata.fault_transect.profiles import extract_profiles
    from plotdata.fault_transect.plot_api import PlotOptions, MapFigureSpec, ProfileFigureSpec
    from plotdata.fault_transect.backends import get_backend
    from plotdata.fault_transect.export import write_offset_txt, write_profiles_txt
    from plotdata.fault_transect.log_io import append_command_log
    from plotdata.fault_transect.naming import MAP_LABEL, PROFILE_LABEL, build_basename
    from plotdata.fault_transect.cache import (
        should_write_txt, figure_is_fresh, combined_figure_is_fresh, write_figure_style,
        map_figure_style_key, profile_figure_style_key, find_cached_txt,
        discover_map_periods, map_period_bracket, profile_period_bracket)

    eos_file, project, source = resolve_input(data_input)
    out_dir = inps.outdir if inps.outdir else default_output_dir(eos_file, project, source)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, 'work')
    fault_paths = _fault_cache_inputs(inps)

    project_dir = os.path.dirname(out_dir)
    append_command_log(project_dir, command)

    if inps.periods:
        periods = inps.periods
    elif inps.plots_only:
        periods = discover_map_periods(out_dir, project, inps.tag_string)
        if not periods:
            raise SystemExit(f'ERROR: --plots-only found no map txt files under {out_dir}')
    else:
        periods = [full_date_span(eos_file)]

    points = sample_points_segments(fault_segments, inps.along_step, inps.along_start, inps.along_end)
    if not points:
        raise SystemExit('ERROR: no sampling points on the fault; check --along-* options')
    print(f'{project}: {len(points)} sampling points along {points[-1].along_km:.1f} km of fault')

    fault_segments_xy = [{'lons': [c[0] for c in seg], 'lats': [c[1] for c in seg]} for seg in fault_segments]

    backend = get_backend()
    index_entries = []

    from plotdata.fault_transect.periods import consecutive_start_flags, gap_start_flags
    grids = []
    all_series = []
    prof_bundles_data = []
    if inps.plots_only:
        consec_flags = [False] * len(periods)
        gap_flags = [False] * len(periods)
    else:
        consec_flags = consecutive_start_flags(periods)
        gap_flags = gap_start_flags(periods)

    for i, (start, end) in enumerate(periods):
        map_bracket = map_period_bracket(inps, start, end)
        prof_bracket = profile_period_bracket(inps, start, end)
        map_txt = find_cached_txt(out_dir, project, inps.tag_string, MAP_LABEL, start, end, map_bracket)
        prof_label = PROFILE_LABEL
        prof_txt = find_cached_txt(out_dir, project, inps.tag_string, prof_label, start, end, prof_bracket)

        grid = None
        if inps.plot_type in ('map', 'both'):
            series = _load_map_series_from_cache(inps, map_txt, eos_file, map_bracket)
            if series is None:
                if inps.plots_only:
                    raise SystemExit(
                        f'ERROR: --plots-only requires map txt for period {start}:{end}')
                grid = load_velocity_grid(eos_file, project, source, start, end, work_dir,
                                          inps.mask_vmin, consecutive_start=consec_flags[i],
                                          gap_start=gap_flags[i], force=inps.force,
                                          tag_string=inps.tag_string)
                series = compute_offset_series(
                    grid.data, grid.lats, grid.lons, points,
                    inps.perp_width, inps.along_step,
                    inps.sample_method, inps.reference_side, grid.unit,
                    perp_offset_km=inps.perp_offset)
            else:
                start_date, end_date = _dates_from_product_txt(map_txt)
                grid = _make_grid_stub(start_date, end_date, inps.unit)
            all_series.append(series)
        elif inps.plots_only:
            if prof_txt:
                start_date, end_date = _dates_from_product_txt(prof_txt)
            else:
                start_date, end_date = start, end
            grid = _make_grid_stub(start_date, end_date, inps.unit)
        else:
            grid = load_velocity_grid(eos_file, project, source, start, end, work_dir,
                                      inps.mask_vmin, consecutive_start=consec_flags[i],
                                      gap_start=gap_flags[i], force=inps.force,
                                      tag_string=inps.tag_string)
        grids.append(grid)

        if inps.plot_type in ('profile', 'both'):
            bundle, main_indices = _load_profiles_from_cache(
                inps, prof_txt, eos_file, prof_bracket)
            if bundle is None:
                if inps.plots_only:
                    raise SystemExit(
                        f'ERROR: --plots-only requires profile txt for period {start}:{end}')
                if grid.data is None:
                    grid = load_velocity_grid(eos_file, project, source, start, end, work_dir,
                                              inps.mask_vmin, consecutive_start=consec_flags[i],
                                              gap_start=gap_flags[i],
                                              tag_string=inps.tag_string)
                    grids[-1] = grid
                bundle = extract_profiles(grid.data, grid.attr, points, inps.profile_length,
                                          inps.interpolation, grid.unit)
                main_indices = [idx for idx in _select_profile_points(points, inps)
                                if bundle.get(idx) is not None]
            prof_bundles_data.append((bundle, main_indices))

    actual_periods = [(g.start_date, g.end_date) for g in grids]
    combine = _side_by_side(inps, len(grids))

    from plotdata.fault_transect.limits import build_offset_color_limits
    color_lims = build_offset_color_limits(
        len(grids),
        tuple(inps.vlim) if inps.vlim else None,
        inps.auto_colorscale,
        all_series)

    # -------------------------------------------------------------- map plot
    if inps.plot_type in ('map', 'both'):
        map_specs, map_txts = [], []
        for grid, series, color_lim, (start, end) in zip(
                grids, all_series, color_lims, periods):
            map_bracket = map_period_bracket(inps, start, end)
            stem = build_basename(project, inps.tag_string, MAP_LABEL, grid.start_date, grid.end_date)
            txt_path = find_cached_txt(out_dir, project, inps.tag_string, MAP_LABEL, start, end, map_bracket)
            if txt_path is None:
                txt_path = os.path.join(out_dir, f'{stem}.txt')
            if should_write_txt(inps, txt_path, eos_file, map_bracket, *fault_paths):
                write_offset_txt(txt_path, series, map_bracket)
            map_specs.append(MapFigureSpec(data=grid.data, lats=grid.lats, lons=grid.lons,
                                           fault_segments=fault_segments_xy,
                                           offset_series=series, perp_width_km=inps.perp_width,
                                           options=make_plot_options(inps, grid,
                                                             vlim=color_lim)))
            map_txts.append(txt_path)

        map_style = map_figure_style_key(inps, color_lims, len(map_specs))
        if combine:
            stem = _multi_period_stem(project, inps.tag_string, MAP_LABEL, actual_periods)
            img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
            if len(map_specs) > 1:
                from plotdata.fault_transect.plot_api import compute_map_stack_step
                lat_min, lat_max = _fault_lat_bounds(fault_segments_xy)
                lons = [lon for seg in fault_segments_xy for lon in seg['lons']]
                lon_min, lon_max = min(lons), max(lons)
                stack_step = compute_map_stack_step(
                    inps.map_stack_axis, inps.perp_width,
                    lon_min, lon_max, lat_min, lat_max, inps.map_stack_offset)
                _record_map_stack_summary(
                    inps, project, len(map_specs), stack_step, inps.map_stack_offset)
            if (not inps.plots_only
                    and combined_figure_is_fresh(img_path, map_txts, map_style,
                                                 eos_file, *fault_paths)):
                print(f'Skipping figure (up to date): {img_path}')
            else:
                backend.render_map_figure(map_specs, img_path)
                write_figure_style(img_path, map_style)
                print(f'Figure saved to {img_path}')
            index_entries.append({'group': f'{project} map', 'image': img_path, 'txt': map_txts})
        else:
            for spec, txt_path, grid in zip(map_specs, map_txts, grids):
                stem = build_basename(project, inps.tag_string, MAP_LABEL,
                                      grid.start_date, grid.end_date)
                img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
                period_style = map_figure_style_key(inps, [spec.options.vlim], 1)
                if (not inps.plots_only
                        and figure_is_fresh(img_path, txt_path, period_style,
                                            eos_file, *fault_paths)):
                    print(f'Skipping figure (up to date): {img_path}')
                else:
                    backend.render_map_figure([spec], img_path)
                    write_figure_style(img_path, period_style)
                    print(f'Figure saved to {img_path}')
                index_entries.append({'group': f'{project} map', 'image': img_path, 'txt': txt_path})

    # --------------------------------------------------------- profile plots
    if inps.plot_type in ('profile', 'both'):
        prof_specs, prof_bundles = [], []
        for grid_idx, grid in enumerate(grids):
            if grid_idx < len(prof_bundles_data):
                bundle, main_indices = prof_bundles_data[grid_idx]
            else:
                bundle = extract_profiles(grid.data, grid.attr, points, inps.profile_length,
                                          inps.interpolation, grid.unit)
                main_indices = [idx for idx in _select_profile_points(points, inps)
                                if bundle.get(idx) is not None]
            if not main_indices:
                print(f'WARNING: no valid profiles for {project} '
                      f'{grid.start_date}_{grid.end_date}')
                continue
            prof_specs.append(ProfileFigureSpec(bundle=bundle, main_indices=main_indices,
                                                cloud_profiles=inps.cloud_profiles,
                                                stack_offset=inps.stack_offset,
                                                subplot_cols=inps.subplot_cols,
                                                connect_lines=inps.profile_lines,
                                                options=make_plot_options(inps, grid)))
            prof_bundles.append((bundle, main_indices, grid, periods[grid_idx]))

        prof_style = profile_figure_style_key(inps, inps.plot_layout, len(prof_specs))
        if prof_specs and inps.plot_layout == 'separate':
            for spec, (bundle, main_indices, grid, (req_start, req_end)) in zip(prof_specs, prof_bundles):
                prof_bracket = profile_period_bracket(inps, req_start, req_end)
                stem = build_basename(project, inps.tag_string, f'{PROFILE_LABEL}_{{nn}}',
                                      grid.start_date, grid.end_date)
                template = os.path.join(out_dir, f'{stem}.{inps.save}')
                need_render = False
                planned = []
                sep_style = profile_figure_style_key(inps, 'separate', 1)
                for nn, idx in enumerate(spec.main_indices):
                    img_path = template.replace('{nn}', f'{nn:02d}')
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    if should_write_txt(inps, txt_path, eos_file, prof_bracket, *fault_paths):
                        write_profiles_txt(txt_path, bundle, [main_indices[nn]],
                                           inps.cloud_profiles, prof_bracket)
                    if (not inps.plots_only
                            and figure_is_fresh(img_path, txt_path, sep_style,
                                                eos_file, *fault_paths)):
                        print(f'Skipping figure (up to date): {img_path}')
                    else:
                        need_render = True
                    planned.append((img_path, txt_path))
                if need_render:
                    written = backend.render_profiles_separate(spec, template)
                    for img_path in written:
                        write_figure_style(img_path, sep_style)
                        print(f'Figure saved to {img_path}')
                for img_path, txt_path in planned:
                    index_entries.append({'group': f'{project} profiles {req_start}_{req_end}',
                                          'image': img_path, 'txt': txt_path})
        elif prof_specs:
            if inps.plot_layout == 'stacked':
                from plotdata.fault_transect.profiles import auto_stack_offset
                step = (inps.stack_offset if inps.stack_offset is not None
                        else auto_stack_offset(prof_specs[0].bundle))
                _record_profile_stack_summary(
                    inps, project, step, prof_specs[0].options.unit,
                    inps.stack_offset is not None)
            render = (backend.render_profiles_subplot if inps.plot_layout == 'subplot'
                      else backend.render_profiles_stacked)
            txt_paths = []
            for bundle, main_indices, grid, (req_start, req_end) in prof_bundles:
                prof_bracket = profile_period_bracket(inps, req_start, req_end)
                prof_label = PROFILE_LABEL
                txt_path = find_cached_txt(out_dir, project, inps.tag_string, prof_label,
                                           req_start, req_end, prof_bracket)
                if txt_path is None:
                    stem = build_basename(project, inps.tag_string, prof_label,
                                          grid.start_date, grid.end_date)
                    txt_path = os.path.join(out_dir, f'{stem}.txt')
                if should_write_txt(inps, txt_path, eos_file, prof_bracket, *fault_paths):
                    write_profiles_txt(txt_path, bundle, main_indices, inps.cloud_profiles,
                                       prof_bracket)
                txt_paths.append(txt_path)

            if combine:
                bundle_periods = [p for _, _, _, p in prof_bundles]
                stem = _multi_period_stem(project, inps.tag_string, PROFILE_LABEL, bundle_periods)
                img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
                if (not inps.plots_only
                        and combined_figure_is_fresh(img_path, txt_paths, prof_style,
                                                     eos_file, *fault_paths)):
                    print(f'Skipping figure (up to date): {img_path}')
                else:
                    render(prof_specs, img_path)
                    write_figure_style(img_path, prof_style)
                    print(f'Figure saved to {img_path}')
                index_entries.append({'group': f'{project} profiles',
                                      'image': img_path, 'txt': txt_paths})
            else:
                for spec, txt_path, (_, _, grid, (req_start, req_end)) in zip(
                        prof_specs, txt_paths, prof_bundles):
                    stem = build_basename(project, inps.tag_string, PROFILE_LABEL,
                                          grid.start_date, grid.end_date)
                    img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
                    one_style = profile_figure_style_key(inps, inps.plot_layout, 1)
                    if (not inps.plots_only
                            and figure_is_fresh(img_path, txt_path, one_style,
                                                eos_file, *fault_paths)):
                        print(f'Skipping figure (up to date): {img_path}')
                    else:
                        render([spec], img_path)
                        write_figure_style(img_path, one_style)
                        print(f'Figure saved to {img_path}')
                    index_entries.append({'group': f'{project} profiles',
                                          'image': img_path, 'txt': txt_path})

    if not inps.no_index and index_entries:
        from plotdata.fault_transect.html_index import write_index_html
        from plotdata.fault_transect.naming import index_html_stem
        stem = index_html_stem(project, inps.tag_string)
        write_index_html(out_dir, command, index_entries, stem)

    if inps.show_flag:
        print('Close all figure windows to finish.')
        backend.show()
    backend.close_all()

    return index_entries


def _print_done():
    print('Done.')


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    from plotdata.fault_transect.log_io import append_command_log
    command = f'{os.path.basename(sys.argv[0])} {" ".join(sys.argv[1:])}'.strip()
    append_command_log(os.getcwd(), command)

    from plotdata.fault_transect.kmz_fault import prepare_fault_geometry
    fault_segments, source_kmz, fault_cache_paths = prepare_fault_geometry(inps)
    inps.fault_source_kmz = source_kmz
    inps.fault_cache_paths = fault_cache_paths

    for data_input in inps.data_dir:
        _process_input(inps, fault_segments, data_input, command)

    if inps.upload:
        print('WARNING: --upload is not implemented yet; skipping upload.')

    from plotdata.fault_transect.plot_api import print_offset_summaries
    print_offset_summaries(getattr(inps, 'offset_summaries', []))

    _print_done()


if __name__ == '__main__':
    main()
