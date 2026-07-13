#!/usr/bin/env python3
"""Generate index.html listing all produced figures and data files."""

import os
import html
from datetime import datetime

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>plot_fault_transect results</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; color: #222; }}
  h1 {{ font-size: 1.4em; }}
  h2 {{ font-size: 1.15em; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 1.6em; }}
  .cmd {{ background: #f4f4f4; padding: 8px 12px; border-radius: 6px; font-family: monospace;
         font-size: 0.85em; overflow-x: auto; }}
  .item {{ display: inline-block; margin: 10px; text-align: center; vertical-align: top; }}
  .item img {{ max-width: 340px; border: 1px solid #ddd; border-radius: 4px; }}
  .item a {{ font-size: 0.85em; }}
  .txt {{ display: block; margin-top: 2px; }}
</style>
</head>
<body>
<h1>plot_fault_transect results</h1>
<p class="cmd">{command}</p>
<p>Generated: {timestamp}</p>
{sections}
</body>
</html>
"""


def write_index_html(out_dir, command, entries):
    """entries: list of dicts with keys: group (section title), image, txt.

    'txt' may be a single path or a list of paths. Paths are made relative to
    out_dir. Returns the index.html path.
    """
    groups = {}
    for entry in entries:
        groups.setdefault(entry['group'], []).append(entry)

    sections = []
    for group, items in groups.items():
        cards = []
        for item in items:
            img_rel = os.path.relpath(item['image'], out_dir)
            card = f'<div class="item"><a href="{img_rel}"><img src="{img_rel}" alt=""></a>'
            card += f'<a href="{img_rel}">{html.escape(os.path.basename(item["image"]))}</a>'
            txts = item.get('txt') or []
            if isinstance(txts, str):
                txts = [txts]
            for txt in txts:
                txt_rel = os.path.relpath(txt, out_dir)
                card += f'<a class="txt" href="{txt_rel}">{html.escape(os.path.basename(txt))}</a>'
            card += '</div>'
            cards.append(card)
        sections.append(f'<h2>{html.escape(group)}</h2>\n' + '\n'.join(cards))

    page = _PAGE.format(command=html.escape(command),
                        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        sections='\n'.join(sections))
    path = os.path.join(out_dir, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'Index written to {path}')
    return path
