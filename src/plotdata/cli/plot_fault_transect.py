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
  plot_fault_transect.py PFS_Pernicana_faults_system_.kmz --dry-run
  plot_fault_transect.py PFS_Pernicana_faults_system__joint.kmz EtnaSenA44/mintpy --period 20141020:20260626 --tag Pernicana --no-display
  plot_fault_transect.py fault_joint.kmz EtnaSenA44/mintpy EtnaSenD124/mintpy --plot-type map --perp-width 0.5 --no-display
  plot_fault_transect.py fault.kmz EtnaSenA44/mintpy --fault-segment 2-8 --plot-type profile --profile-count 10 --plot-layout subplot --cloud-profiles 2 --no-display
  plot_fault_transect.py fault_joint.kmz EtnaSenA44/mintpy --plot-type profile --plot-layout stacked --period 20141020:20181231,20190101:20260626 --no-display
"""


def create_parser():
    parser = argparse.ArgumentParser(
        description='Plot across-fault displacement offset (map) and fault-perpendicular profiles',
        epilog=EXAMPLE, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('fault_file', help='Fault trace KMZ/KML (single- or multi-segment)')
    parser.add_argument('data_dir', nargs='*', help='1-4 inputs: S1_*.he5 file or mintpy/miaplpy directory')

    fault = parser.add_argument_group('Fault handling')
    fault.add_argument('--dry-run', dest='dry_run', action='store_true', help='Join segments, write {stem}_joint.kmz + {stem}_joint.txt, then exit')
    fault.add_argument('--fault-segment', dest='fault_segment', type=str, default='all', help='Segment selection: all, 3, 2-8, 0,2,5 (default: %(default)s)')
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
    geom.add_argument('--sample-method', dest='sample_method', choices=['mean', 'median', 'nearest'], default='mean', help='Combining pixels in a search box (default: %(default)s)')
    geom.add_argument('--interpolation', dest='interpolation', choices=['nearest', 'linear', 'cubic'], default='nearest', help='Value extraction along profile lines (default: %(default)s)')

    plot = parser.add_argument_group('Plot selection and layout')
    plot.add_argument('--plot-type', dest='plot_type', choices=['map', 'profile', 'both'], default='both', help='What to plot (default: %(default)s)')
    plot.add_argument('--reference-side', dest='reference_side', choices=['left', 'right'], default='left', help='Reference side of fault; offset = reference - other (default: %(default)s)')
    plot.add_argument('--plot-layout', dest='plot_layout', choices=['separate', 'subplot', 'stacked', '3d'], default='stacked', help='Profile arrangement (default: %(default)s)')
    plot.add_argument('--profile-spacing', dest='profile_spacing', type=float, default=None, help='Spacing between profiles in km (default: --along-step)')
    plot.add_argument('--profile-length', dest='profile_length', type=float, default=10.0, help='Profile length across the fault in km (default: %(default)s)')
    plot.add_argument('--profile-count', dest='profile_count', type=int, default=None, help='Fixed number of profiles (overrides --profile-spacing)')
    plot.add_argument('--cloud-profiles', dest='cloud_profiles', type=int, default=0, help='Adjacent profiles per side as thin gray lines (default: %(default)s)')
    plot.add_argument('--stack-offset', dest='stack_offset', type=float, default=None, help='Vertical offset between stacked profiles (default: auto)')
    plot.add_argument('--subplot-cols', dest='subplot_cols', type=int, default=1, help='Columns for subplot layout (default: %(default)s)')
    plot.add_argument('--period-layout', dest='period_layout', choices=['auto', 'side-by-side', 'separate-page'], default='auto', help='Arrangement for multiple periods (default: %(default)s)')

    style = parser.add_argument_group('Plot parameters')
    style.add_argument('--vlim', dest='vlim', nargs=2, type=float, metavar=('VMIN', 'VMAX'), default=None, help='Velocity limits for the map background')
    style.add_argument('--colormap', dest='colormap', default='viridis', help='Colormap (default: %(default)s)')
    style.add_argument('--font-size', dest='font_size', type=int, default=10, help='Font size (default: %(default)s)')
    style.add_argument('--dpi', dest='dpi', type=int, default=300, help='Figure DPI (default: %(default)s)')

    out = parser.add_argument_group('Output')
    out.add_argument('--save', dest='save', choices=['png', 'pdf'], default='png', help='Image format; images are always saved (default: %(default)s)')
    out.add_argument('--no-display', dest='show_flag', action='store_false', default=True, help='Do not open interactive windows')
    out.add_argument('--outdir', dest='outdir', type=str, default=None, help='Output directory (default: <project>/transects_mintpy or transects_miaplpy)')
    out.add_argument('--tag', dest='tag_string', type=str, default='', help='Tag inserted into output filenames (default: none)')
    out.add_argument('--no-index', dest='no_index', action='store_true', help='Skip index.html generation')
    out.add_argument('--upload', dest='upload', action='store_true', default=False, help='Upload products (not implemented yet)')

    return parser


def parse_periods(tokens):
    """Parse --period tokens; each may contain comma-separated periods.

    Returns list of (start, end) strings. Empty list means full span.
    """
    periods = []
    for token in tokens:
        for chunk in token.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            match = re.fullmatch(r'(\d{8}):(\d{8})', chunk)
            if not match:
                raise ValueError(f'Invalid --period "{chunk}": expected YYYYMMDD:YYYYMMDD')
            periods.append((match.group(1), match.group(2)))
    return periods


def build_basename(project, tag_string, plot_label, start_date, end_date):
    tag_string = (tag_string or '').strip()
    if tag_string:
        return f'{project}_{tag_string}_{plot_label}_{start_date}_{end_date}'
    return f'{project}_{plot_label}_{start_date}_{end_date}'


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if not inps.dry_run and not inps.data_dir:
        parser.error('at least one data input is required (or use --dry-run)')
    if len(inps.data_dir) > 4:
        parser.error('at most 4 data inputs are supported (asc, desc, horz, vert)')
    if inps.plot_layout == '3d':
        parser.error('--plot-layout 3d is not implemented yet; use separate, subplot or stacked')
    if inps.profile_spacing is None:
        inps.profile_spacing = inps.along_step

    try:
        inps.periods = parse_periods(inps.period)
    except ValueError as exc:
        parser.error(str(exc))

    return inps


def _load_fault(inps):
    """Read the KMZ and return one (lon, lat) polyline according to the options."""
    from plotdata.fault_transect.kmz_fault import (
        read_fault_kmz, join_segments, parse_segment_spec)

    segments = read_fault_kmz(inps.fault_file)
    indices = parse_segment_spec(inps.fault_segment, len(segments))
    selected = [segments[i] for i in indices]

    if len(selected) == 1:
        coords = list(selected[0].coords)
    elif inps.fault_segment.strip().lower() == 'all' and not inps.fault_file.endswith('_joint.kmz'):
        raise SystemExit(
            f'ERROR: {os.path.basename(inps.fault_file)} contains {len(segments)} segments. '
            'Run with --dry-run first to create and inspect the joint KMZ, '
            'or select segments with --fault-segment N / N-M.')
    else:
        coords, report = join_segments(selected)
        for warning in report.warnings:
            print(f'WARNING: {warning}')

    if inps.flip_fault:
        coords = list(reversed(coords))
    return coords


def _run_dry_run(inps):
    from plotdata.fault_transect.kmz_fault import (
        read_fault_kmz, join_segments, parse_segment_spec,
        write_fault_kmz, write_join_report, joint_output_paths)

    segments = read_fault_kmz(inps.fault_file)
    indices = parse_segment_spec(inps.fault_segment, len(segments))
    selected = [segments[i] for i in indices]
    print(f'Read {len(segments)} segment(s) from {inps.fault_file}; using {len(selected)}')

    coords, report = join_segments(selected)
    kmz_path, txt_path = joint_output_paths(inps.fault_file, inps.outdir)
    if inps.outdir:
        os.makedirs(inps.outdir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(kmz_path))[0]
    write_fault_kmz(kmz_path, coords, name=stem)
    write_join_report(txt_path, report, inps.fault_file, coords)

    print(f'Joint fault KMZ:    {kmz_path}')
    print(f'Join report:        {txt_path}')
    print(f'Total fault length: {report.total_length_km:.3f} km '
          f'({len(report.order)} segments, {len(coords)} vertices)')
    for warning in report.warnings:
        print(f'WARNING: {warning}')
    print('Inspect the joint KMZ in Google Earth; if satisfied, rerun with it (no --dry-run).')


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
    label = '_'.join(f'{s}-{e}' for s, e in periods)
    return build_basename(project, tag_string, plot_label, periods[0][0], periods[-1][1]) \
        if len(periods) == 1 else \
        (f'{project}_{tag_string}_{plot_label}_{label}' if tag_string else f'{project}_{plot_label}_{label}')


def _process_input(inps, coords, data_input, command):
    """Full pipeline for one data input: all periods, map + profile plots."""
    from plotdata.fault_transect.load_data import (
        resolve_input, default_output_dir, full_date_span, load_velocity_grid)
    from plotdata.fault_transect.fault_sampling import sample_points
    from plotdata.fault_transect.offset import compute_offset_series
    from plotdata.fault_transect.profiles import extract_profiles
    from plotdata.fault_transect.plot_api import PlotOptions, MapFigureSpec, ProfileFigureSpec
    from plotdata.fault_transect.backends import get_backend
    from plotdata.fault_transect.export import write_offset_txt, write_profiles_txt
    from plotdata.fault_transect.log_io import append_command_log

    eos_file, project, source = resolve_input(data_input)
    out_dir = inps.outdir if inps.outdir else default_output_dir(eos_file, project, source)
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.join(out_dir, 'work')

    project_dir = os.path.dirname(out_dir)
    append_command_log(project_dir, command)

    periods = inps.periods if inps.periods else [full_date_span(eos_file)]

    points = sample_points(coords, inps.along_step, inps.along_start, inps.along_end)
    if not points:
        raise SystemExit('ERROR: no sampling points on the fault; check --along-* options')
    print(f'{project}: {len(points)} sampling points along {points[-1].along_km:.1f} km of fault')

    fault_lons = [c[0] for c in coords]
    fault_lats = [c[1] for c in coords]

    section_info = (f'fault={os.path.basename(inps.fault_file)} '
                    f'along-step={inps.along_step} perp-width={inps.perp_width} '
                    f'sample-method={inps.sample_method} reference-side={inps.reference_side}')
    prof_info = section_info + f' profile-length={inps.profile_length} layout={inps.plot_layout}'

    backend = get_backend()
    index_entries = []

    # ---- load all periods first (needed for side-by-side figures)
    grids = [load_velocity_grid(eos_file, project, source, s, e, work_dir, inps.mask_vmin)
             for s, e in periods]
    actual_periods = [(g.start_date, g.end_date) for g in grids]
    combine = _side_by_side(inps, len(grids))

    def make_opts(grid):
        return PlotOptions(colormap=inps.colormap,
                           vlim=tuple(inps.vlim) if inps.vlim else None,
                           font_size=inps.font_size, dpi=inps.dpi, unit=grid.unit,
                           title=f'{project} {grid.start_date}:{grid.end_date}')

    # -------------------------------------------------------------- map plot
    if inps.plot_type in ('map', 'both'):
        map_specs, map_txts = [], []
        for grid in grids:
            series = compute_offset_series(grid.data, grid.lats, grid.lons, points,
                                           inps.perp_width, inps.along_step,
                                           inps.sample_method, inps.reference_side, grid.unit)
            stem = build_basename(project, inps.tag_string, 'map', grid.start_date, grid.end_date)
            txt_path = os.path.join(out_dir, f'{stem}.txt')
            write_offset_txt(txt_path, series, section_info)
            map_specs.append(MapFigureSpec(data=grid.data, lats=grid.lats, lons=grid.lons,
                                           fault_lons=fault_lons, fault_lats=fault_lats,
                                           offset_series=series, perp_width_km=inps.perp_width,
                                           options=make_opts(grid)))
            map_txts.append(txt_path)

        if combine:
            stem = _multi_period_stem(project, inps.tag_string, 'map', actual_periods)
            img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
            backend.render_map_figure(map_specs, img_path)
            print(f'Figure saved to {img_path}')
            index_entries.append({'group': f'{project} map', 'image': img_path, 'txt': map_txts})
        else:
            for spec, txt_path, (start, end) in zip(map_specs, map_txts, actual_periods):
                stem = build_basename(project, inps.tag_string, 'map', start, end)
                img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
                backend.render_map_figure([spec], img_path)
                print(f'Figure saved to {img_path}')
                index_entries.append({'group': f'{project} map', 'image': img_path, 'txt': txt_path})

    # --------------------------------------------------------- profile plots
    if inps.plot_type in ('profile', 'both'):
        prof_specs, prof_bundles = [], []
        for grid in grids:
            bundle = extract_profiles(grid.data, grid.attr, points, inps.profile_length,
                                      inps.interpolation, grid.unit)
            main_indices = [i for i in _select_profile_points(points, inps)
                            if bundle.get(i) is not None]
            if not main_indices:
                print(f'WARNING: no valid profiles for {project} '
                      f'{grid.start_date}_{grid.end_date}')
                continue
            prof_specs.append(ProfileFigureSpec(bundle=bundle, main_indices=main_indices,
                                                cloud_profiles=inps.cloud_profiles,
                                                stack_offset=inps.stack_offset,
                                                subplot_cols=inps.subplot_cols,
                                                options=make_opts(grid)))
            prof_bundles.append((bundle, main_indices, (grid.start_date, grid.end_date)))

        if prof_specs and inps.plot_layout == 'separate':
            for spec, (bundle, main_indices, (start, end)) in zip(prof_specs, prof_bundles):
                stem = build_basename(project, inps.tag_string, 'profile{nn}', start, end)
                template = os.path.join(out_dir, f'{stem}.{inps.save}')
                written = backend.render_profiles_separate(spec, template)
                for nn, img_path in enumerate(written):
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    write_profiles_txt(txt_path, bundle, [main_indices[nn]],
                                       inps.cloud_profiles, prof_info)
                    print(f'Figure saved to {img_path}')
                    index_entries.append({'group': f'{project} profiles {start}_{end}',
                                          'image': img_path, 'txt': txt_path})
        elif prof_specs:
            render = (backend.render_profiles_subplot if inps.plot_layout == 'subplot'
                      else backend.render_profiles_stacked)
            txt_paths = []
            for bundle, main_indices, (start, end) in prof_bundles:
                stem = build_basename(project, inps.tag_string,
                                      f'profiles_{inps.plot_layout}', start, end)
                txt_path = os.path.join(out_dir, f'{stem}.txt')
                write_profiles_txt(txt_path, bundle, main_indices, inps.cloud_profiles, prof_info)
                txt_paths.append(txt_path)

            if combine:
                bundle_periods = [p for _, _, p in prof_bundles]
                stem = _multi_period_stem(project, inps.tag_string,
                                          f'profiles_{inps.plot_layout}', bundle_periods)
                img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
                render(prof_specs, img_path)
                print(f'Figure saved to {img_path}')
                index_entries.append({'group': f'{project} profiles',
                                      'image': img_path, 'txt': txt_paths})
            else:
                for spec, txt_path, (_, _, (start, end)) in zip(prof_specs, txt_paths, prof_bundles):
                    stem = build_basename(project, inps.tag_string,
                                          f'profiles_{inps.plot_layout}', start, end)
                    img_path = os.path.join(out_dir, f'{stem}.{inps.save}')
                    render([spec], img_path)
                    print(f'Figure saved to {img_path}')
                    index_entries.append({'group': f'{project} profiles',
                                          'image': img_path, 'txt': txt_path})

    if not inps.no_index and index_entries:
        from plotdata.fault_transect.html_index import write_index_html
        write_index_html(out_dir, command, index_entries)

    if inps.show_flag:
        backend.show()
    backend.close_all()

    return index_entries


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    from plotdata.fault_transect.log_io import append_command_log
    command = f'{os.path.basename(sys.argv[0])} {" ".join(sys.argv[1:])}'.strip()
    append_command_log(os.getcwd(), command)

    if inps.dry_run:
        _run_dry_run(inps)
        return

    coords = _load_fault(inps)

    for data_input in inps.data_dir:
        _process_input(inps, coords, data_input, command)

    if inps.upload:
        print('WARNING: --upload is not implemented yet; skipping upload.')


if __name__ == '__main__':
    main()
