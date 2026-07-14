#!/usr/bin/env python3
"""Output filename tag helpers for fault transect products."""

import os
import re


def extract_fault_tag(fault_path):
    """Derive a short tag from a fault KMZ/KML basename.

    Rules (first match wins):
    1. Text before ``Fault``/``fault`` (e.g. ``FiandacaFault_FA.kmz`` -> ``Fiandaca``).
    2. Else text before the first ``_`` (e.g. ``Fiandaca_FA.kmz`` -> ``Fiandaca``).
    3. Else the full basename stem (e.g. ``FiandacaFA.kmz`` -> ``FiandacaFA``).
    """
    stem = os.path.splitext(os.path.basename(fault_path))[0]
    match = re.search(r'(?i)fault', stem)
    if match:
        prefix = stem[:match.start()].rstrip('_')
        if prefix:
            return prefix
    if '_' in stem:
        return stem.split('_')[0]
    return stem


def resolve_output_tag(fault_path, tag_string=None):
    """Return explicit ``--tag`` or auto-extract from ``fault_path``."""
    explicit = (tag_string or '').strip()
    if explicit:
        return explicit
    return extract_fault_tag(fault_path)


def format_fault_plot_title(tag_string):
    """User-facing map/profile title from fault tag (e.g. ``Fiandaca Fault``)."""
    tag = (tag_string or '').strip()
    if not tag:
        return ''
    return f'{tag} Fault'


def format_dataset_display_label(dataset_label):
    """Capitalize dataset type for figure labels (e.g. ``Ascending``)."""
    label = (dataset_label or '').strip().lower()
    if not label:
        return ''
    return label.capitalize()


def format_figure_suptitle(title, dataset_label=''):
    """Combine fault title and dataset label for figure suptitles."""
    title = (title or '').strip()
    dataset = format_dataset_display_label(dataset_label)
    if title and dataset:
        return f'{title}\n{dataset}'
    return title or dataset


MAP_LABEL = 'map'
PROFILE_LABEL = 'profile'
TIMESERIES_LABEL = 'timeseries'


def build_basename(project, tag_string, plot_label, start_date, end_date):
    """Build ``{project}_{tag}_{plot_label}_{start}_{end}`` (tag always required)."""
    tag_string = (tag_string or '').strip()
    if not tag_string:
        raise ValueError('fault tag is required for output filenames (use --tag or a named KMZ)')
    return f'{project}_{tag_string}_{plot_label}_{start_date}_{end_date}'


def multi_period_stem(project, tag_string, plot_label, periods):
    """Filename stem for a combined multi-period figure or txt."""
    if len(periods) == 1:
        return build_basename(project, tag_string, plot_label, periods[0][0], periods[-1][1])
    label = '_'.join(f'{s}-{e}' for s, e in periods)
    tag_string = (tag_string or '').strip()
    if not tag_string:
        raise ValueError('fault tag is required for output filenames (use --tag or a named KMZ)')
    return f'{project}_{tag_string}_{plot_label}_{label}'


def index_html_stem(project, tag_string):
    """Basename for the named HTML index (e.g. ``EtnaSenA44_Fiandaca``)."""
    tag_string = (tag_string or '').strip()
    if tag_string:
        return f'{project}_{tag_string}'
    return project
