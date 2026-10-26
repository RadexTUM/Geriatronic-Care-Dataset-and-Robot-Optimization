"""Minimal HTML preview per processed trial."""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_task_preview(processed_dir: Path, metadata: dict[str, Any]) -> Path:
    npz = processed_dir / "task_data.npz"
    rows = ""
    try:
        data = np.load(npz)
        for side in ["left", "right"]:
            pos = data[f"{side}_pos"]
            force = np.linalg.norm(data[f"{side}_force"], axis=1)
            rows += (
                f"<tr><td>{side}</td>"
                f"<td>{np.ptp(pos, axis=0).round(3).tolist()}</td>"
                f"<td>{force.max():.1f}</td></tr>"
            )
    except Exception as exc:  # the preview must never fail the preprocessing
        rows = f"<tr><td colspan='3'>preview error: {html_lib.escape(str(exc))}</td></tr>"
    title = html_lib.escape(str(metadata.get("trial_id") or metadata.get("task_name")))
    description = html_lib.escape(str(metadata.get("description") or ""))
    details = html_lib.escape(json.dumps(metadata, indent=2, default=str))
    page = f"""<html><head><meta charset='utf-8'><title>{title}</title></head>
<body><h2>{title}</h2>
<p>{description}</p>
<p>category: {html_lib.escape(str(metadata.get('category_name')))} | split: {metadata.get('split')} |
force measurement: {metadata.get('force_measurement')}</p>
<table border='1' cellpadding='4'><tr><th>side</th><th>range xyz [m]</th><th>max |F| [N]</th></tr>{rows}</table>
<details><summary>metadata</summary><pre>{details}</pre></details>
</body></html>"""
    out = processed_dir / "preview.html"
    out.write_text(page, encoding="utf-8")
    return out
