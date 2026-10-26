"""Morris (elementary effects) sensitivity screening of the mount parameters.

Which of the six mirrored mount parameters influence coverage and soft loss,
and how strongly does the base-motion mode mask that influence? Run once with
mode B and once with mode E and compare mu_star.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from garmi_optimization.evaluation import set_base_motion_mode
from garmi_optimization.config import DEFAULT_PROFILE, PATHS, ProjectPaths, ensure_dirs, load_baseline_config, timestamp_id, write_json
from garmi_optimization.optimization import (
    MountCandidate,
    _candidate_bounds,
    _clamp_candidate,
    _evaluate_mount_candidate_batch,
    _profile_slug,
)


def _morris_trajectories(dim: int, r: int, levels: int, seed: int) -> np.ndarray:
    """r Morris trajectories in the unit cube, shape (r, dim + 1, dim)."""
    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1.0))
    grid = np.arange(0, levels) / (levels - 1.0)
    usable = grid[grid <= 1.0 - delta]
    trajectories = np.empty((r, dim + 1, dim), dtype=float)
    for t in range(r):
        base = rng.choice(usable, size=dim)
        order = rng.permutation(dim)
        direction = rng.choice([-1.0, 1.0], size=dim)
        point = base.copy()
        # keep steps inside [0, 1]
        for d in range(dim):
            if direction[d] > 0 and point[d] + delta > 1.0:
                direction[d] = -1.0
            if direction[d] < 0 and point[d] - delta < 0.0:
                direction[d] = 1.0
        trajectories[t, 0] = point
        for step, d in enumerate(order, start=1):
            point = point.copy()
            point[d] = point[d] + direction[d] * delta
            trajectories[t, step] = point
    return trajectories


def run_sensitivity(
    group_name: str,
    task_names: list[str],
    paths: ProjectPaths = PATHS,
    baseline_profile: str = DEFAULT_PROFILE,
    trajectories: int = 8,
    levels: int = 6,
    seed: int = 7,
    workers: int | None = None,
    base_motion_mode: str | None = None,
    metrics: tuple[str, ...] = ("optimization_score", "success_fraction", "mean_soft_loss"),
) -> Path:
    """Morris screening of the 6 mount parameters at quick fidelity."""
    ensure_dirs(paths)
    cfg = load_baseline_config(paths, profile=baseline_profile)
    if base_motion_mode:
        set_base_motion_mode(cfg, base_motion_mode)
    bounds = _candidate_bounds(cfg)
    names = list(bounds)
    lower = np.asarray([bounds[name][0] for name in names], dtype=float)
    upper = np.asarray([bounds[name][1] for name in names], dtype=float)
    dim = len(names)

    unit = _morris_trajectories(dim, int(trajectories), int(levels), int(seed))
    candidates: list[MountCandidate] = []
    index_map: list[tuple[int, int]] = []
    for t in range(unit.shape[0]):
        for s in range(unit.shape[1]):
            values = dict(zip(names, lower + unit[t, s] * (upper - lower)))
            candidates.append(_clamp_candidate(values, f"morris_t{t:02d}_s{s:02d}", cfg))
            index_map.append((t, s))

    mode = str(cfg.get("base", {}).get("motion_mode", "?"))
    slug = _profile_slug(cfg, baseline_profile)
    run_dir = paths.runs / timestamp_id(f"sensitivity_{slug}_{group_name}_mode_{mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fidelity = {"name": "quick", "sample_stride": 12, "max_samples": 60, "base_search": "single"}
    summaries, _task_rows, _cfgs = _evaluate_mount_candidate_batch(
        candidates,
        base_cfg=cfg,
        task_names=task_names,
        paths=paths,
        group_name=group_name,
        workers=workers,
        base_search="single",
        stage_label="morris",
        fidelity=fidelity,
        fidelity_label="quick",
        cache_dir=run_dir / "evaluation_cache",
    )
    by_id = {str(s["candidate_id"]): s for s in summaries}

    delta = float(levels) / (2.0 * (levels - 1.0))
    records: list[dict[str, Any]] = []
    for metric in metrics:
        effects: dict[str, list[float]] = {name: [] for name in names}
        for t in range(unit.shape[0]):
            for s in range(1, unit.shape[1]):
                prev_id = f"morris_t{t:02d}_s{s - 1:02d}"
                curr_id = f"morris_t{t:02d}_s{s:02d}"
                if prev_id not in by_id or curr_id not in by_id:
                    continue
                diff_vec = unit[t, s] - unit[t, s - 1]
                changed = int(np.argmax(np.abs(diff_vec)))
                y0 = float(by_id[prev_id].get(metric, np.nan))
                y1 = float(by_id[curr_id].get(metric, np.nan))
                if not (np.isfinite(y0) and np.isfinite(y1)):
                    continue
                effects[names[changed]].append((y1 - y0) / (delta * np.sign(diff_vec[changed])))
        for name in names:
            values = np.asarray(effects[name], dtype=float)
            records.append(
                {
                    "metric": metric,
                    "parameter": name,
                    "mu": float(np.mean(values)) if len(values) else float("nan"),
                    "mu_star": float(np.mean(np.abs(values))) if len(values) else float("nan"),
                    "sigma": float(np.std(values)) if len(values) else float("nan"),
                    "n_effects": int(len(values)),
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(run_dir / "sensitivity.csv", index=False)
    pd.DataFrame(summaries).to_csv(run_dir / "morris_evaluations.csv", index=False)
    _write_tornado_html(run_dir, df, group_name, mode)
    write_json(
        run_dir / "sensitivity_summary.json",
        {
            "group_name": group_name,
            "baseline_profile": baseline_profile,
            "base_motion_mode": mode,
            "trajectories": int(trajectories),
            "levels": int(levels),
            "seed": int(seed),
            "parameters": names,
            "evaluations": len(candidates),
            "elapsed_s": float(time.perf_counter() - started),
            "interpretation": (
                "mu_star ~ 0 for all mount parameters means the mount has no "
                "measurable influence under this base-motion mode; optimizing "
                "it in that regime is not meaningful."
            ),
        },
    )
    return run_dir


def _write_tornado_html(run_dir: Path, df: pd.DataFrame, group_name: str, mode: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return
    metrics = list(df["metric"].unique())
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics, shared_yaxes=True)
    for col, metric in enumerate(metrics, start=1):
        sub = df[df["metric"] == metric].sort_values("mu_star")
        fig.add_trace(
            go.Bar(x=sub["mu_star"], y=sub["parameter"], orientation="h", name=metric,
                   error_x={"type": "data", "array": sub["sigma"], "visible": True}),
            row=1,
            col=col,
        )
    fig.update_layout(
        title=f"Morris sensitivity (mu*), group={group_name}, base mode={mode}",
        height=420,
        showlegend=False,
    )
    fig.write_html(run_dir / "sensitivity_tornado.html", include_plotlyjs="cdn")
