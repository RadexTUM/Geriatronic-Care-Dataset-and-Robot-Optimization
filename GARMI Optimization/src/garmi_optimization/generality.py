"""Price-of-generality analysis.

Central co-design question: how much coverage does a single global mount cost
compared to the per-group optima? This module evaluates a set of candidate
configurations (typically the ``best_config.yaml`` of each group run plus the
global run) against every task group and reports the coverage matrix and the
price of generality per group.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from garmi_optimization.evaluation import _aggregate, evaluate_tasks_parallel
from garmi_optimization.config import (
    PATHS,
    ProjectPaths,
    ensure_dirs,
    load_baseline_config,
    load_task_frames_config,
    timestamp_id,
    write_json,
)
from garmi_optimization.tasks import group_task_names, group_task_weights, load_tasks


def run_generality_report(
    configs: dict[str, str],
    groups: list[str],
    paths: ProjectPaths = PATHS,
    workers: int | None = None,
) -> Path:
    """Evaluate each named config on each group; write coverage matrix + price of generality.

    configs: mapping label -> profile name or YAML path
             (e.g. {"recommended": "config/recommended.yaml",
                    "specialist_transfer": "results/paper/configs/specialist_transfer.yaml"}).
    groups:  group names from task_groups.yaml.
    """
    ensure_dirs(paths)
    run_dir = paths.runs / timestamp_id("generality_report")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows: list[dict[str, Any]] = []
    for label, profile in configs.items():
        cfg = load_baseline_config(paths, profile=str(profile))
        if "task_frames" not in cfg:
            frames = load_task_frames_config(paths)
            if frames:
                cfg["task_frames"] = frames
        if workers is not None:
            cfg.setdefault("performance", {})["parallel_workers"] = int(workers)
        for group in groups:
            task_names = group_task_names(group, paths)
            cfg_group = dict(cfg)
            cfg_group["_task_weights"] = group_task_weights(group, paths)
            tasks = load_tasks(task_names, paths)
            task_rows = evaluate_tasks_parallel(tasks, cfg_group, group)
            aggregate = _aggregate(task_rows, group)
            rows.append(
                {
                    "config": label,
                    "profile": str(profile),
                    "group": group,
                    "task_count": aggregate["task_count"],
                    "success_fraction": aggregate["success_fraction"],
                    "mean_soft_loss": aggregate.get("mean_soft_loss", float("nan")),
                    "mean_position_error_m": aggregate["mean_position_error_m"],
                    "min_clearance_m": aggregate["min_clearance_m"],
                    "max_torque_utilization": aggregate["max_torque_utilization"],
                }
            )
            print(f"[generality] {label} x {group}: coverage {aggregate['success_fraction']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "coverage_matrix.csv", index=False)

    # Price of generality: per group, best specialist coverage minus each
    # config's coverage on that group.
    pivot = df.pivot_table(index="group", columns="config", values="success_fraction")
    best_specialist = pivot.max(axis=1)
    price = pivot.rsub(best_specialist, axis=0)
    price.columns = [f"price_{col}" for col in price.columns]
    matrix = pd.concat([pivot, price], axis=1).reset_index()
    matrix.to_csv(run_dir / "price_of_generality.csv", index=False)
    _write_html(run_dir, df, pivot)
    write_json(
        run_dir / "generality_summary.json",
        {
            "configs": {str(k): str(v) for k, v in configs.items()},
            "groups": list(groups),
            "elapsed_s": float(time.perf_counter() - started),
            "note": (
                "price_<config> = best specialist coverage of the group minus "
                "the config's coverage on that group. The central artifact for "
                "the group-vs-global co-design decision."
            ),
        },
    )
    return run_dir


def _write_html(run_dir: Path, df: pd.DataFrame, pivot: pd.DataFrame) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            zmin=0.0,
            zmax=1.0,
            colorscale="RdYlGn",
            text=[[f"{v:.2f}" for v in row] for row in pivot.to_numpy()],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Coverage: configuration x task group (price of generality)", height=420)
    fig.write_html(run_dir / "generality_report.html", include_plotlyjs="cdn")
