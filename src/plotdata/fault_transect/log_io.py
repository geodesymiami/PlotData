#!/usr/bin/env python3
"""Command logging in the minsar message_rsmas style (local copy, standalone)."""

import os
from datetime import datetime


def _abbreviate_paths(msg):
    for var in ('SAMPLESDIR', 'TEMPLATES', 'SCRATCHDIR'):
        value = os.getenv(var)
        if value:
            token = '$TE' if var == 'TEMPLATES' else f'${var}'
            msg = msg.replace(value, token)
    return msg


def append_command_log(log_dir, message):
    """Append 'YYYYMMDD-HH:MM + command' to log_dir/log (created if needed)."""
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    line = f'{datetime.now().strftime("%Y%m%d-%H:%M")} + {_abbreviate_paths(message)}'
    with open(os.path.join(log_dir, 'log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line)
