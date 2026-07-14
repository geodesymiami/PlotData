#!/usr/bin/env python3
"""Resolve colormap names the way MintPy view.py does, plus a coherence map."""

import os
import re

import numpy as np


def coherence_colormap(cmap_lut=256):
    """InSAR temporal-coherence colorscale (γ): 0 = blue, 1 = white."""
    from matplotlib.colors import LinearSegmentedColormap

    stops = [
        (0.00, (0.00, 0.04, 0.71)),
        (0.12, (0.05, 0.45, 0.90)),
        (0.28, (0.00, 0.75, 0.85)),
        (0.42, (0.20, 0.82, 0.75)),
        (0.50, (0.45, 0.88, 0.62)),
        (0.62, (0.70, 0.92, 0.55)),
        (0.75, (0.92, 0.95, 0.55)),
        (0.88, (0.98, 0.98, 0.82)),
        (1.00, (1.00, 1.00, 1.00)),
    ]
    return LinearSegmentedColormap.from_list('coherence', stops, N=cmap_lut)


def _cmy_colormap(cmap_lut=256):
    from matplotlib.colors import LinearSegmentedColormap

    rgbs = np.zeros((256, 3), dtype=np.uint8)
    for kk in range(85):
        rgbs[kk, 0] = kk * 3
        rgbs[kk, 1] = 255 - kk * 3
        rgbs[kk, 2] = 255
    rgbs[85:170, 0] = rgbs[0:85, 2]
    rgbs[85:170, 1] = rgbs[0:85, 0]
    rgbs[85:170, 2] = rgbs[0:85, 1]
    rgbs[170:255, 0] = rgbs[0:85, 1]
    rgbs[170:255, 1] = rgbs[0:85, 2]
    rgbs[170:255, 2] = rgbs[0:85, 0]
    rgbs[255, 0] = 0
    rgbs[255, 1] = 255
    rgbs[255, 2] = 255
    rgbs = np.roll(rgbs, int(256 / 2 - 214), axis=0)
    rgbs = np.flipud(rgbs)
    return LinearSegmentedColormap.from_list('cmy', rgbs / 255., N=cmap_lut)


def _dismph_colormap(cmap_lut=256):
    from matplotlib.colors import LinearSegmentedColormap

    clist = ['#f579cd', '#f67fc6', '#f686bf', '#f68cb9', '#f692b3', '#f698ad',
             '#f69ea7', '#f6a5a1', '#f6ab9a', '#f6b194', '#f6b78e', '#f6bd88',
             '#f6c482', '#f6ca7b', '#f6d075', '#f6d66f', '#f6dc69', '#f6e363',
             '#efe765', '#e5eb6b', '#dbf071', '#d0f477', '#c8f67d', '#c2f684',
             '#bbf68a', '#b5f690', '#aff696', '#a9f69c', '#a3f6a3', '#9cf6a9',
             '#96f6af', '#90f6b5', '#8af6bb', '#84f6c2', '#7df6c8', '#77f6ce',
             '#71f6d4', '#6bf6da', '#65f6e0', '#5ef6e7', '#58f0ed', '#52e8f3',
             '#4cdbf9', '#7bccf6', '#82c4f6', '#88bdf6', '#8eb7f6', '#94b1f6',
             '#9aabf6', '#a1a5f6', '#a79ef6', '#ad98f6', '#b392f6', '#b98cf6',
             '#bf86f6', '#c67ff6', '#cc79f6', '#d273f6', '#d86df6', '#de67f6',
             '#e561f6', '#e967ec', '#ed6de2', '#f173d7']
    cmap = LinearSegmentedColormap.from_list('dismph', clist, N=cmap_lut)
    cmap.set_bad('w', 0.0)
    return cmap


def _mintpy_cpt_dirs():
    dirs = []
    try:
        import mintpy
        dirs.append(os.path.join(os.path.dirname(mintpy.__file__), 'data', 'colormaps'))
    except ImportError:
        pass
    gmt_dir = '/opt/local/share/gmt/cpt'
    if os.path.isdir(gmt_dir):
        dirs.append(gmt_dir)
    return dirs


def _is_number(value):
    try:
        float(value)
    except ValueError:
        return False
    return True


def _read_cpt_file(cpt_file, cmap_lut=256):
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    with open(cpt_file) as handle:
        lines = handle.readlines()

    x, r, g, b = [], [], [], []
    color_model = 'RGB'
    for line in lines:
        if not line.strip():
            continue
        parts = re.split(r' |\t|\n|/', line)
        parts = [part for part in parts if part]
        if line[0] == '#':
            if parts and parts[-1] == 'HSV':
                color_model = 'HSV'
            continue
        if parts[0] in ('B', 'F', 'N'):
            continue
        if not _is_number(parts[1]):
            rgb = [int(255 * channel) for channel in to_rgb(parts[1])]
            parts = [parts[0], *map(str, rgb), *parts[2:]]
        if len(parts) >= 8 and not _is_number(parts[5]):
            rgb = [int(255 * channel) for channel in to_rgb(parts[5])]
            parts = parts[:5] + list(map(str, rgb)) + parts[6:]
        values = [float(part) for part in parts[:8]]
        x.extend([values[0], values[4]])
        r.extend([values[1], values[5]])
        g.extend([values[2], values[6]])
        b.extend([values[3], values[7]])

    x = np.asarray(x, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if color_model == 'RGB':
        r /= 255.
        g /= 255.
        b /= 255.

    x_norm = (x - x[0]) / (x[-1] - x[0])
    red = [(x_norm[i], r[i], r[i]) for i in range(len(x))]
    green = [(x_norm[i], g[i], g[i]) for i in range(len(x))]
    blue = [(x_norm[i], b[i], b[i]) for i in range(len(x))]
    cmap_name = os.path.splitext(os.path.basename(cpt_file))[0]
    return LinearSegmentedColormap(cmap_name, {'red': tuple(red),
                                               'green': tuple(green),
                                               'blue': tuple(blue)}, N=cmap_lut)


def _truncate_colormap(cmap, vlist, cmap_lut=256):
    from matplotlib.colors import LinearSegmentedColormap

    v0, v1, v2 = vlist
    n1_ratio = (v1 - v0) / (v2 - v0)
    n1 = int(np.rint(cmap_lut * n1_ratio))
    n2 = cmap_lut - n1
    colors1 = cmap(np.linspace(0.0, 0.3, max(n1, 1)))
    colors2 = cmap(np.linspace(0.6, 1.0, max(n2, 1)))
    return LinearSegmentedColormap.from_list(
        f'{cmap.name}_truncate',
        np.vstack((colors1, colors2)),
        N=cmap_lut,
    )


def _parse_colormap_name(name):
    reverse = False
    truncate = False
    repeat = 1
    base = name

    match = re.search(r'_\d+$', base)
    if match:
        repeat = int(match.group(0).split('_')[-1])
        base = base[:match.start()]

    if base.endswith('_truncate'):
        truncate = True
        base = base[:-len('_truncate')]

    if base.endswith('_r'):
        reverse = True
        base = base[:-len('_r')]

    return base, reverse, truncate, repeat


def _builtin_colormap(base_name, cmap_lut=256):
    if base_name == 'coherence':
        return coherence_colormap(cmap_lut)
    if base_name == 'cmy':
        return _cmy_colormap(cmap_lut)
    if base_name == 'dismph':
        return _dismph_colormap(cmap_lut)
    return None


def _cpt_colormap(base_name, cmap_lut=256):
    for cpt_dir in _mintpy_cpt_dirs():
        cpt_file = os.path.join(cpt_dir, f'{base_name}.cpt')
        if os.path.isfile(cpt_file):
            return _read_cpt_file(cpt_file, cmap_lut=cmap_lut)
    return None


def resolve_colormap(name, cmap_lut=256, vlist=None):
    """Return a matplotlib colormap object for a MintPy-style name.

    Supports matplotlib names (``jet``, ``RdBu``, ``viridis``, ...),
    MintPy builtins (``cmy``, ``dismph``, ``temperature``, ``vik``, ...),
    suffixes ``_r``, ``_truncate``, ``_N`` (repeat), and the InSAR
    coherence map ``coherence`` (γ scale: white = 1, blue = 0).
    """
    import matplotlib.pyplot as plt

    vlist = list(vlist) if vlist is not None else [0.0, 0.7, 1.0]
    if len(vlist) != 3:
        raise ValueError('--cmap-vlist requires exactly three values: VMIN VMID VMAX')

    try:
        from mintpy.objects.colors import ColormapExt
        return ColormapExt(name, cmap_lut=cmap_lut, vlist=vlist).colormap
    except ImportError:
        pass
    except ValueError:
        pass

    base, reverse, truncate, repeat = _parse_colormap_name(name)
    cmap = _builtin_colormap(base, cmap_lut=cmap_lut)
    if cmap is None:
        cmap = _cpt_colormap(base, cmap_lut=cmap_lut)
    if cmap is None:
        try:
            cmap = plt.get_cmap(base, lut=cmap_lut)
        except ValueError as exc:
            raise ValueError(
                f'Unrecognized colormap "{name}". Try jet, RdBu, cmy, coherence, '
                f'temperature, vik, or any matplotlib colormap.') from exc

    if truncate:
        cmap = _truncate_colormap(cmap, vlist, cmap_lut=cmap_lut)
    if reverse:
        cmap = cmap.reversed()
    if repeat > 1:
        from matplotlib.colors import LinearSegmentedColormap
        colors = np.tile(cmap(np.linspace(0., 1., cmap_lut)), (repeat, 1))
        cmap = LinearSegmentedColormap.from_list(f'{base}_{repeat}', colors,
                                                 N=cmap_lut * repeat)
    return cmap


def known_colormap_examples():
    """Short list for CLI help."""
    return ('jet (default), jet_r, jet_truncate, RdBu, RdBu_r, viridis, cmy, '
            'dismph, temperature, vik, coherence')
