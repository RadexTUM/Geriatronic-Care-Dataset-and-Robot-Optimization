"""Report suite (``garmi-opt report-all``): baseline reports plus MuJoCo
simulation reports and videos per trial.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from garmi_optimization.evaluation import _profile_slug, run_baseline, set_base_motion_mode
from garmi_optimization.config import (
    DEFAULT_PROFILE,
    PATHS,
    ProjectPaths,
    ensure_dirs,
    load_baseline_config,
    load_group_config,
    timestamp_id,
    write_json,
)
from garmi_optimization.mujoco_backend import simulate_batch
from garmi_optimization.tasks import global_task_names, group_task_names


def _link(path: Path, label: str) -> str:
    return f'<a href="{path.resolve().as_uri()}">{label}</a>'


def _selected_groups(
    group_names: list[str] | None,
    all_groups: bool,
    global_opt: bool,
    paths: ProjectPaths,
) -> list[tuple[str, list[str]]]:
    groups = load_group_config(paths).get("groups", {})
    selected: list[tuple[str, list[str]]] = []
    seen: set[str] = set()

    def add_group(name: str, tasks: list[str]) -> None:
        if name in seen:
            return
        seen.add(name)
        selected.append((name, tasks))

    for name in group_names or []:
        if name == "all_tasks":
            add_group("all_tasks", global_task_names(paths))
            continue
        add_group(name, group_task_names(name, paths))

    if all_groups:
        for name in groups:
            if name != "all_tasks":
                add_group(name, group_task_names(name, paths))

    if global_opt:
        add_group("all_tasks", global_task_names(paths))

    if not selected:
        raise ValueError("Choose at least one of --group, --all-groups, or --global.")
    return selected


def run_report_suite(
    *,
    group_names: list[str] | None = None,
    all_groups: bool = False,
    global_opt: bool = False,
    baseline_profile: str = DEFAULT_PROFILE,
    paths: ProjectPaths = PATHS,
    workers: int | None = None,
    render_videos: bool = True,
    base_motion_mode: str | None = None,
) -> Path:
    ensure_dirs(paths)
    cfg = load_baseline_config(paths, profile=baseline_profile)
    if base_motion_mode:
        set_base_motion_mode(cfg, base_motion_mode)
    slug = _profile_slug(cfg, baseline_profile)
    run_dir = paths.runs / timestamp_id(f"report_suite_{slug}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for group_name, task_names in _selected_groups(group_names, all_groups, global_opt, paths):
        baseline_dir = run_baseline(group_name, task_names, paths, cfg=cfg, baseline_profile=baseline_profile)
        simulation_dir = simulate_batch(
            task_names,
            group_name,
            paths,
            workers=workers,
            render=render_videos,
            cfg=cfg,
            baseline_profile=baseline_profile,
        )
        videos = sorted(simulation_dir.glob("*/impedance_playback.mp4"))
        for video in videos:
            video_rows.append(
                {
                    "group_name": group_name,
                    "task_name": video.parent.name,
                    "video": str(video),
                    "report": str(video.parent / "report.html"),
                }
            )
        rows.append(
            {
                "group_name": group_name,
                "task_count": len(task_names),
                "baseline_report": str(baseline_dir / "report.html"),
                "baseline_dir": str(baseline_dir),
                "simulation_report": str(simulation_dir / "report.html"),
                "simulation_dir": str(simulation_dir),
                "video_count": len(videos),
            }
        )

    elapsed_s = time.perf_counter() - started
    pd.DataFrame(rows).to_csv(run_dir / "report_suite.csv", index=False)
    pd.DataFrame(video_rows).to_csv(run_dir / "videos.csv", index=False)
    write_json(
        run_dir / "report_suite.json",
        {
            "baseline_profile": cfg.get("_profile", baseline_profile),
            "baseline_name": cfg.get("baseline", {}).get("name", "unnamed"),
            "base_motion_mode": cfg.get("base", {}).get("motion_mode", "free_holonomic"),
            "render_videos": render_videos,
            "workers": workers,
            "elapsed_s": elapsed_s,
            "groups": rows,
            "videos": video_rows,
        },
    )
    _write_report_suite_html(run_dir, cfg, rows, video_rows, elapsed_s)
    return run_dir


def _write_report_suite_html(
    run_dir: Path,
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    elapsed_s: float,
) -> Path:
    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "group_name": row["group_name"],
                "task_count": row["task_count"],
                "baseline_report": _link(Path(row["baseline_report"]), "open"),
                "simulation_report": _link(Path(row["simulation_report"]), "open"),
                "video_count": row["video_count"],
            }
        )
    display_videos = []
    for row in video_rows:
        display_videos.append(
            {
                "group_name": row["group_name"],
                "task_name": row["task_name"],
                "video": _link(Path(row["video"]), "mp4"),
                "report": _link(Path(row["report"]), "report"),
            }
        )
    group_table = pd.DataFrame(display_rows).to_html(index=False, escape=False)
    video_table = pd.DataFrame(display_videos).to_html(index=False, escape=False) if display_videos else "<p>No videos rendered.</p>"
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Report Suite</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #182027; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d7dde2; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f3f6f8; }}
</style></head>
<body>
<h1>Report Suite</h1>
<p><strong>Baseline profile:</strong> {cfg.get("_profile", "baseline")} ({cfg.get("baseline", {}).get("name", "unnamed")})</p>
<p><strong>Elapsed:</strong> {elapsed_s:.1f} s</p>
<h2>Reports</h2>
{group_table}
<h2>Videos</h2>
{video_table}
</body></html>
"""
    output = run_dir / "index.html"
    output.write_text(html, encoding="utf-8")
    return output
