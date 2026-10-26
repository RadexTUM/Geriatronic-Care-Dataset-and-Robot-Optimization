"""Local browser tool (``garmi-opt gui``) for configuring and launching evaluations,
optimizations and MuJoCo simulations and for editing task groups.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from garmi_optimization.collision import BoxProxy, CapsuleProxy, SphereProxy, garmi_body_proxies
from garmi_optimization.evaluation import normalize_base_motion_mode
from garmi_optimization.config import DEFAULT_PROFILE, PATHS, ensure_dirs, load_baseline_config, load_group_config, read_json, timestamp_id, write_yaml
from garmi_optimization.optimization import MountCandidate, _candidate_bounds, apply_mount_candidate, mount_plausibility
from garmi_optimization.panda import PandaModel, forward_points
from garmi_optimization.tasks import configured_task_names, global_task_names, group_task_names


JOB_LOCK = threading.Lock()
GROUP_LOCK = threading.Lock()
JOBS: dict[str, "GuiJob"] = {}


@dataclass
class GuiJob:
    job_id: str
    kind: str
    status: str = "queued"
    started_s: float = field(default_factory=time.time)
    finished_s: float | None = None
    message: str = ""
    log: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "started_s": self.started_s,
            "finished_s": self.finished_s,
            "elapsed_s": (self.finished_s or time.time()) - self.started_s,
            "message": self.message,
            "log": self.log[-8000:],
            "result": self.result,
            "command": self.command,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _artifact_url(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return "/artifact?path=" + urllib.parse.quote(str(resolved))


def _html_link(path: str | Path, label: str) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return f'<a href="{_artifact_url(path)}">{label}</a>'


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _profiles() -> list[str]:
    ignored = {"project", "task_groups", "task_frames"}
    profiles = []
    for path in sorted(PATHS.config.glob("*.yaml")):
        if path.stem not in ignored:
            profiles.append(path.stem)
    return profiles


def _groups() -> dict[str, Any]:
    return load_group_config().get("groups", {})


def _unique_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _task_catalog() -> list[dict[str, Any]]:
    """Processed trials plus all trials referenced by a group."""
    configured = configured_task_names()
    configured_set = set(configured)
    manifest_by_id: dict[str, dict[str, Any]] = {}
    # Written by the preprocessing: all dataset trials with category and split.
    manifest_path = PATHS.processed_data / "manifest.json"
    if manifest_path.exists():
        for item in read_json(manifest_path).get("trials", []):
            trial_id = str(item.get("trial_id", "")).strip()
            if trial_id:
                manifest_by_id[trial_id] = item
    group_refs: list[str] = []
    for group in _groups().values():
        group_refs.extend(str(task) for task in group.get("tasks", []))
        group_refs.extend(str(task) for task in group.get("excluded_tasks", []))
    names = _unique_preserve([*configured, *group_refs])
    catalog = []
    for name in names:
        item = manifest_by_id.get(name, {})
        category = str(item.get("category_name") or "")
        catalog.append(
            {
                "task_id": name,
                "display_name": f"{category}: {name}" if category else name,
                "category": category,
                "trial_name": name,
                "status": "valid" if name in configured_set else str(item.get("status") or "unknown"),
                "configured": bool(name in configured_set),
            }
        )
    return catalog


def _groups_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, group in _groups().items():
        tasks = [str(task) for task in group.get("tasks", [])]
        excluded = _unique_preserve([str(task) for task in group.get("excluded_tasks", [])])
        effective = [task for task in tasks if task not in set(excluded)]
        weights = {str(k): float(v) for k, v in (group.get("weights", {}) or {}).items()}
        payload[name] = {
            "description": str(group.get("description", "")),
            "tasks": tasks,
            "excluded_tasks": excluded,
            "effective_tasks": effective,
            "weights": weights,
            "task_count": len(tasks),
            "effective_task_count": len(effective),
            "duplicate_count": len(tasks) - len(set(tasks)),
        }
    return payload


def _group_editor_payload() -> dict[str, Any]:
    return {"groups": _groups_payload(), "tasks": _task_catalog()}


def _sanitize_group_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    if not name:
        raise ValueError("Group name must not be empty.")
    return name


def _save_group(payload: dict[str, Any]) -> dict[str, Any]:
    group = payload.get("group", {})
    name = _sanitize_group_name(group.get("name"))
    original_name = str(payload.get("original_name") or group.get("original_name") or name).strip()
    tasks = [str(task).strip() for task in group.get("tasks", []) if str(task).strip()]
    excluded = _unique_preserve([str(task).strip() for task in group.get("excluded_tasks", []) if str(task).strip()])
    with GROUP_LOCK:
        data = load_group_config()
        groups = data.setdefault("groups", {})
        if original_name and original_name != name and original_name in groups:
            groups.pop(original_name)
        entry: dict[str, Any] = {
            "description": str(group.get("description", "")).strip(),
            "tasks": tasks,
            "excluded_tasks": excluded,
        }
        # Explicit task weights (preferred over duplicating task entries).
        raw_weights = group.get("weights", {}) or {}
        weights: dict[str, float] = {}
        for key, value in raw_weights.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if str(key).strip() and abs(weight - 1.0) > 1e-9:
                weights[str(key).strip()] = max(0.0, weight)
        if weights:
            entry["weights"] = weights
        groups[name] = entry
        write_yaml(PATHS.config / "task_groups.yaml", data)
    return {"saved": name, **_group_editor_payload()}


def _delete_group(payload: dict[str, Any]) -> dict[str, Any]:
    name = _sanitize_group_name(payload.get("name"))
    if name == "all_tasks":
        raise ValueError("The all_tasks group cannot be deleted.")
    with GROUP_LOCK:
        data = load_group_config()
        groups = data.setdefault("groups", {})
        if name not in groups:
            raise KeyError(f"Unknown group '{name}'. Available: {', '.join(sorted(groups))}")
        groups.pop(name)
        write_yaml(PATHS.config / "task_groups.yaml", data)
    return {"deleted": name, **_group_editor_payload()}


def _float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _list(value: Any, default: list[float]) -> list[float]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _set_if_present(target: dict[str, Any], settings: dict[str, Any], key: str, cast: Callable[[Any], Any]) -> None:
    if key in settings:
        target[key] = cast(settings[key])


def _apply_gui_overrides(cfg: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(_jsonable(cfg)))
    performance = cfg.setdefault("performance", {})
    if "performance_parallel_workers" in settings:
        workers = settings["performance_parallel_workers"]
        performance["parallel_workers"] = workers if str(workers).strip().lower() in {"auto", "off"} else int(workers)
    if "performance_progress" in settings:
        performance["progress"] = bool(settings["performance_progress"])

    if "simulation_use_garmi_follow" in settings:
        camera = "garmi_follow" if bool(settings["simulation_use_garmi_follow"]) else "overview"
        simulation = cfg.setdefault("simulation", {})
        simulation["render_camera"] = camera
        simulation["interactive_camera"] = camera

    base = cfg.setdefault("base", {})
    if "base_motion_mode" in settings:
        base["motion_mode"] = normalize_base_motion_mode(settings["base_motion_mode"])
    for key, default in [
        ("fixed_pose", [0.0, 0.0, 0.0]),
        ("preferred_reach_fraction_candidates", [0.62, 0.72, 0.82]),
        ("forward_offset_candidates_m", [-0.10, 0.0, 0.12]),
        ("lateral_offset_candidates_m", [-0.20, 0.0, 0.20]),
        ("yaw_offset_candidates_rad", [0.0]),
    ]:
        setting_key = f"base_{key}"
        if setting_key in settings:
            base[key] = _list(settings[setting_key], default)
    for key, cast in [
        ("max_speed_mps", float),
        ("max_acc_mps2", float),
        ("limited_max_path_m", float),
        ("limited_max_speed_mps", float),
        ("limited_max_yaw_rate_radps", float),
        ("nonholonomic_max_speed_mps", float),
        ("nonholonomic_max_yaw_rate_radps", float),
        ("smoothing_window", int),
    ]:
        setting_key = f"base_{key}"
        if setting_key in settings:
            base[key] = cast(settings[setting_key])

    ik = cfg.setdefault("ik", {})
    for key in [
        "iterations",
        "damping",
        "step_limit_rad",
        "position_tolerance_m",
        "orientation_weight",
        "orientation_tolerance_rad",
        "joint_center_weight",
        "continuity_weight",
    ]:
        setting_key = f"ik_{key}"
        if setting_key in settings:
            ik[key] = int(settings[setting_key]) if key == "iterations" else float(settings[setting_key])
    if "ik_use_orientation" in settings:
        ik["use_orientation"] = bool(settings["ik_use_orientation"])
    if "ik_orientation_mode" in settings:
        ik["orientation_mode"] = str(settings["ik_orientation_mode"])

    collision = cfg.setdefault("collision", {})
    for key in [
        "hard_clearance_m",
        "warning_clearance_m",
        "avoidance_start_m",
        "avoidance_target_m",
        "avoidance_step_rad",
        "avoidance_gradient_epsilon_rad",
        "avoidance_nullspace_damping",
        "avoidance_max_position_error_m",
        "avoidance_max_orientation_error_rad",
        "avoidance_min_improvement_m",
        "arm_link_radius_m",
        "joint_radius_m",
        "end_effector_radius_m",
        "shoulder_mount_exclusion_m",
    ]:
        setting_key = f"collision_{key}"
        if setting_key in settings:
            collision[key] = float(settings[setting_key])
    for key in ["enabled", "collision_aware_ik"]:
        setting_key = f"collision_{key}"
        if setting_key in settings:
            collision[key] = bool(settings[setting_key])
    if "collision_avoidance_iterations" in settings:
        collision["avoidance_iterations"] = int(settings["collision_avoidance_iterations"])
    if "collision_shoulder_mount_exclusion_points" in settings:
        collision["shoulder_mount_exclusion_points"] = int(settings["collision_shoulder_mount_exclusion_points"])

    success = cfg.setdefault("success", {})
    for key in [
        "mean_position_error_m",
        "max_position_error_m",
        "mean_orientation_error_rad",
        "max_orientation_error_rad",
        "min_manipulability",
        "max_torque_utilization",
        "min_clearance_m",
    ]:
        setting_key = f"success_{key}"
        if setting_key in settings:
            success[key] = float(settings[setting_key])

    simulation_validation = cfg.setdefault("simulation_validation", {})
    for key in [
        "mean_position_error_m",
        "max_position_error_m",
        "mean_orientation_error_rad",
        "max_orientation_error_rad",
        "max_joint_torque_clip_fraction",
    ]:
        setting_key = f"simulation_validation_{key}"
        if setting_key in settings:
            simulation_validation[key] = float(settings[setting_key])
    if "simulation_validation_max_contact_count" in settings:
        simulation_validation["max_contact_count"] = int(settings["simulation_validation_max_contact_count"])
    if "simulation_validation_require_no_exception" in settings:
        simulation_validation["require_no_exception"] = bool(settings["simulation_validation_require_no_exception"])

    controller = cfg.setdefault("controller", {})
    for key in [
        "position_kp_npm",
        "position_kd_nspm",
        "orientation_kp_nmprad",
        "orientation_kd_nmsprad",
        "nullspace_kp_nmprad",
        "nullspace_kd_nmsprad",
    ]:
        setting_key = f"controller_{key}"
        if setting_key in settings:
            controller[key] = _list(settings[setting_key], list(controller.get(key, [])))
    for key in ["max_task_force_n", "max_task_torque_nm", "nullspace_damping"]:
        setting_key = f"controller_{key}"
        if setting_key in settings:
            controller[key] = float(settings[setting_key])
    if "controller_bias_compensation" in settings:
        controller["bias_compensation"] = bool(settings["controller_bias_compensation"])
    base_pd = controller.setdefault("base_pd", {})
    if "controller_base_pd_kp" in settings:
        base_pd["kp"] = _list(settings["controller_base_pd_kp"], list(base_pd.get("kp", [2400.0, 2400.0, 850.0])))
    if "controller_base_pd_kd" in settings:
        base_pd["kd"] = _list(settings["controller_base_pd_kd"], list(base_pd.get("kd", [520.0, 520.0, 160.0])))

    optimization = cfg.setdefault("optimization", {})
    if "optimizer" in settings:
        optimization["optimizer"] = str(settings["optimizer"])
    if "optimization_promote_fraction" in settings:
        optimization["promote_fraction"] = float(settings["optimization_promote_fraction"])
    if "optimization_surrogate_pool_size" in settings:
        optimization["surrogate_pool_size"] = int(settings["optimization_surrogate_pool_size"])
    genetic = optimization.setdefault("genetic", {})
    for setting_key, config_key, cast in [
        ("ga_population", "population", int),
        ("ga_generations_quick", "quick_generations", int),
        ("ga_generations_reduced", "reduced_generations", int),
        ("ga_elite_fraction", "elite_fraction", float),
        ("ga_mutation_sigma", "mutation_sigma", float),
        ("ga_crossover_rate", "crossover_rate", float),
    ]:
        if setting_key in settings and settings[setting_key] not in {None, ""}:
            genetic[config_key] = cast(settings[setting_key])
    if "ga_local_seeds" in settings:
        genetic["local_seed_fraction"] = float(genetic.get("local_seed_fraction", 0.25)) if bool(settings["ga_local_seeds"]) else 0.0
    if "ga_seed_run" in settings:
        genetic["seed_run"] = str(settings.get("ga_seed_run", "") or "")
    if "ga_seed_candidates" in settings:
        genetic["seed_candidates"] = [
            item.strip()
            for item in str(settings.get("ga_seed_candidates", "") or "").split(",")
            if item.strip()
        ]
    if "ga_seed_top" in settings and settings["ga_seed_top"] not in {None, ""}:
        genetic["seed_top"] = int(settings["ga_seed_top"])
    if "mount_rotation_mode" in settings:
        optimization["mount_rotation_mode"] = str(settings["mount_rotation_mode"])
    bounds: dict[str, list[float]] = {}
    scales: dict[str, float] = {}
    for name in [
        "left_x_m",
        "left_y_m",
        "left_z_m",
        "left_delta_roll_rad",
        "left_delta_pitch_rad",
        "left_delta_yaw_rad",
    ]:
        min_key = f"bound_{name}_min"
        max_key = f"bound_{name}_max"
        scale_key = f"scale_{name}"
        if min_key in settings or max_key in settings:
            bounds[name] = [float(settings[min_key]), float(settings[max_key])]
        if scale_key in settings:
            scales[name] = float(settings[scale_key])
    if bounds:
        optimization["mount_bounds"] = bounds
        if str(optimization.get("mount_rotation_mode", "")).lower() in {"forward_outer_quarter_sphere", "quarter_sphere", "forward_outer"}:
            quarter_defaults = optimization.get("mount_rotation_quarter_sphere", {})
            optimization["mount_rotation_quarter_sphere"] = {
                "roll_rad": list(bounds.get("left_delta_roll_rad", quarter_defaults.get("roll_rad", [-0.65, 0.65]))),
                "elevation_rad": list(
                    bounds.get(
                        "left_delta_pitch_rad",
                        quarter_defaults.get("elevation_rad", [-1.5706963267948966, 1.5706963267948966]),
                    )
                ),
                "azimuth_rad": list(bounds.get("left_delta_yaw_rad", quarter_defaults.get("azimuth_rad", [0.0, 1.5707963267948966]))),
            }
    if scales:
        optimization["mount_local_scales"] = scales
    if "mount_local_sample_fraction" in settings:
        optimization["mount_local_sample_fraction"] = float(settings["mount_local_sample_fraction"])
    return cfg


def _set_base_search_mode(cfg: dict[str, Any], mode: str) -> None:
    mode = str(mode).lower()
    base = cfg.setdefault("base", {})
    if mode == "full":
        return
    if mode == "single":
        base["preferred_reach_fraction_candidates"] = [0.72]
        base["forward_offset_candidates_m"] = [0.0]
        base["lateral_offset_candidates_m"] = [0.0]
        base["yaw_offset_candidates_rad"] = [0.0]
        return
    if mode == "reduced":
        base["preferred_reach_fraction_candidates"] = list(base.get("preferred_reach_fraction_candidates", [0.62, 0.72, 0.82]))
        base["forward_offset_candidates_m"] = [0.0]
        base["lateral_offset_candidates_m"] = [0.0]
        base["yaw_offset_candidates_rad"] = [0.0]
        return
    raise ValueError(f"Unknown base search mode '{mode}'")


def _baseline_mount_candidate(cfg: dict[str, Any], candidate_id: str = "candidate_000_strict_garmi") -> MountCandidate:
    left = np.asarray(cfg["panda"]["mounts"]["left"]["xyz_m"], dtype=float)
    return MountCandidate(candidate_id, np.array([left[0], abs(left[1]), left[2]], dtype=float), np.zeros(3, dtype=float))


def _load_gui_config(settings: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    profile = str(settings.get("baseline_profile", DEFAULT_PROFILE))
    cfg = load_baseline_config(profile=profile)
    cfg = _apply_gui_overrides(cfg, settings)
    cfg["_profile"] = f"{profile}_gui"
    return profile, cfg


def _mount_candidate_from_settings(settings: dict[str, Any], candidate_id: str = "manual_mount") -> MountCandidate:
    xyz = np.array(
        [
            _float(settings.get("manual_left_x_m"), -0.14326),
            _float(settings.get("manual_left_y_m"), 0.20),
            _float(settings.get("manual_left_z_m"), 0.92451),
        ],
        dtype=float,
    )
    delta = np.array(
        [
            _float(settings.get("manual_left_delta_roll_rad"), 0.0),
            _float(settings.get("manual_left_delta_pitch_rad"), 0.0),
            _float(settings.get("manual_left_delta_yaw_rad"), 0.0),
        ],
        dtype=float,
    )
    return MountCandidate(candidate_id, xyz, delta)


def _scope_args(settings: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    group_name = str(settings.get("group_name", "basic_care"))
    if bool(settings.get("global_opt", False)) or group_name == "all_tasks":
        task_names = global_task_names()
        if not task_names:
            raise ValueError("The selected global task scope has no active tasks.")
        return "all_tasks", task_names, ["--global"]
    task_names = group_task_names(group_name)
    if not task_names:
        raise ValueError(f"The selected task group '{group_name}' has no active tasks.")
    return group_name, task_names, ["--group", group_name]


def _write_temp_config(cfg: dict[str, Any], prefix: str) -> Path:
    path = PATHS.runs / "gui_configs" / f"{timestamp_id(prefix)}.yaml"
    write_yaml(path, cfg)
    return path


def _run_cli(args: list[str], job: GuiJob | None = None) -> tuple[int, str]:
    env = os.environ.copy()
    src_path = str(PATHS.root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = subprocess.Popen(
        args,
        cwd=str(PATHS.root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines: list[str] = []
    existing_log = job.log if job is not None else ""
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        if job is not None:
            with JOB_LOCK:
                job.log = existing_log + "".join(lines)
    return_code = process.wait()
    output = "".join(lines)
    if job is not None:
        with JOB_LOCK:
            job.log = existing_log + output
    return return_code, output


def _parse_path_from_output(output: str, label: str) -> str:
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(output)
    return str(Path(match.group(1).strip()).resolve()) if match else ""


def _simulation_rows(configuration: str, batch_dir: Path, config_path: Path | str = "") -> list[dict[str, Any]]:
    summary_path = batch_dir / "batch_simulation_summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        task_name = str(row.get("task_name", ""))
        task_dir = batch_dir / task_name
        report = task_dir / "report.html"
        video = task_dir / "impedance_playback.mp4"
        rows.append(
            {
                "configuration": configuration,
                "task_name": task_name,
                "config_path": str(config_path),
                "simulation_ok": bool(row.get("simulation_ok", False)),
                "simulation_passed": bool(row.get("simulation_passed", row.get("simulation_ok", False))),
                "simulation_failure_reasons": str(row.get("simulation_failure_reasons", "")),
                "active_sides": str(row.get("active_sides", "")),
                "frames": int(row.get("frames", 0)) if not pd.isna(row.get("frames", np.nan)) else 0,
                "mean_tracking_error_m": float(row.get("mean_tracking_error_m", np.nan)),
                "max_tracking_error_m": float(row.get("max_tracking_error_m", np.nan)),
                "mean_left_tracking_error_m": float(row.get("mean_left_tracking_error_m", np.nan)),
                "mean_right_tracking_error_m": float(row.get("mean_right_tracking_error_m", np.nan)),
                "max_left_tracking_error_m": float(row.get("max_left_tracking_error_m", np.nan)),
                "max_right_tracking_error_m": float(row.get("max_right_tracking_error_m", np.nan)),
                "batch_dir": str(batch_dir),
                "batch_report": str(batch_dir / "report.html"),
                "task_report": str(report) if report.exists() else "",
                "video": str(video) if video.exists() else "",
                "error": str(row.get("error", "")),
            }
        )
    return rows


def _simulation_options_from_configs(configs: list[tuple[str, Path | str]], task_names: list[str], group_name: str) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for label, config_path in configs:
        if not config_path:
            continue
        for task_name in task_names:
            options.append(
                {
                    "configuration": label,
                    "task_name": task_name,
                    "group_name": group_name,
                    "config_path": str(config_path),
                    "task_report": "",
                    "video": "",
                }
            )
    return options


def _simulation_options_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "configuration": str(row.get("configuration", "")),
            "task_name": str(row.get("task_name", "")),
            "config_path": str(row.get("config_path", "")),
            "task_report": str(row.get("task_report", "")),
            "video": str(row.get("video", "")),
        }
        for row in rows
        if row.get("task_name") and row.get("config_path")
    ]


def _write_task_report_index(index_dir: Path, title: str, rows: list[dict[str, Any]]) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    csv_path = index_dir / "task_video_reports.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    display = df.copy()
    if len(display):
        display["batch_report"] = display["batch_report"].map(lambda path: _html_link(path, "batch") if path else "")
        display["task_report"] = display["task_report"].map(lambda path: _html_link(path, "report") if path else "")
        display["video"] = display["video"].map(lambda path: _html_link(path, "video") if path else "")
    table = display.to_html(index=False, escape=False, float_format=lambda value: f"{value:.5f}") if len(display) else "<p>No task reports generated.</p>"
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #182027; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
th, td {{ border: 1px solid #d6dde3; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f3f6f8; }}
a {{ color: #0a5c7a; font-weight: 600; text-decoration: none; }}
</style></head>
<body>
<h1>{title}</h1>
<p>Per-task MuJoCo impedance reports and optional MP4 videos.</p>
{table}
</body></html>
"""
    output = index_dir / "task_video_reports.html"
    output.write_text(html, encoding="utf-8")
    return output


def _run_task_video_reports(
    *,
    settings: dict[str, Any],
    configs: list[tuple[str, dict[str, Any]]],
    scope_args: list[str],
    group_name: str,
    index_dir: Path,
    job: GuiJob,
) -> dict[str, Any]:
    if not bool(settings.get("write_task_reports", False)):
        return {}
    render_videos = bool(settings.get("task_report_render_videos", True))
    workers = _int_or_none(settings.get("task_report_workers"))
    if workers is None:
        workers = _int_or_none(settings.get("workers")) or 1
    config_dir = index_dir / "task_report_configs"
    rows: list[dict[str, Any]] = []
    for label, cfg in configs:
        safe = _safe_label(label)
        cfg_path = config_dir / f"{safe}.yaml"
        cfg = json.loads(json.dumps(_jsonable(cfg)))
        cfg["_profile"] = f"{cfg.get('_profile', 'gui')}_{safe}_mujoco"
        cfg.setdefault("baseline", {})["name"] = safe
        write_yaml(cfg_path, cfg)
        cmd = [
            sys.executable,
            "-m",
            "garmi_optimization.cli",
            "simulate-batch",
            *scope_args,
            "--workers",
            str(workers),
            "--baseline-profile",
            str(cfg_path),
        ]
        if render_videos:
            cmd.append("--video")
        with JOB_LOCK:
            job.command = cmd
            job.log += f"\n[task-reports] {label}: starting MuJoCo batch render={render_videos}\n"
        return_code, output = _run_cli(cmd, job)
        if return_code != 0:
            raise RuntimeError(output)
        batch_dir = Path(_parse_path_from_output(output, "simulation_batch_run"))
        rows.extend(_simulation_rows(label, batch_dir, cfg_path))
    index = _write_task_report_index(index_dir, "MuJoCo Task Reports", rows)
    return {
        "task_report_index": str(index),
        "task_report_index_url": _artifact_url(index),
        "task_report_csv": str(index_dir / "task_video_reports.csv"),
        "task_report_count": len(rows),
        "task_report_render_videos": render_videos,
        "simulations": _simulation_options_from_rows(rows),
    }


def _candidate_from_record(record: pd.Series) -> MountCandidate:
    return MountCandidate(
        str(record["candidate_id"]),
        np.array([float(record["left_x_m"]), float(record["left_y_m"]), float(record["left_z_m"])], dtype=float),
        np.array(
            [
                float(record["left_delta_roll_rad"]),
                float(record["left_delta_pitch_rad"]),
                float(record["left_delta_yaw_rad"]),
            ],
            dtype=float,
        ),
    )


def _top_mount_candidates_for_reports(run_dir: Path, top_count: int) -> list[MountCandidate]:
    source = run_dir / "full_validation_candidates.csv"
    if not source.exists():
        source = run_dir / "candidates.csv"
    if not source.exists():
        return []
    df = pd.read_csv(source)
    sort_cols = [
        col
        for col in [
            "selection_ineligible",
            "selection_failed_task_count",
            "failed_task_count",
            "optimization_score",
            "hard_constraint_violation",
            "multi_objective_score",
        ]
        if col in df
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)
    out: list[MountCandidate] = []
    for _, row in df.iterrows():
        if str(row.get("candidate_id", "")) == "candidate_000_strict_garmi":
            continue
        out.append(_candidate_from_record(row))
        if len(out) >= max(0, int(top_count)):
            break
    return out


def _run_mount_optimization_job(settings: dict[str, Any], job: GuiJob) -> dict[str, Any]:
    _profile, cfg = _load_gui_config(settings)
    config_path = _write_temp_config(cfg, f"gui_mount_opt_config_{job.job_id}")
    group_name, task_names, scope_args = _scope_args(settings)
    cmd = [
        sys.executable,
        "-m",
        "garmi_optimization.cli",
        "optimize-mounts",
        *scope_args,
        "--samples",
        str(int(settings.get("samples", 24))),
        "--seed",
        str(int(settings.get("seed", 7))),
        "--validate-top",
        str(int(settings.get("validate_top", 3))),
        "--base-search",
        str(settings.get("base_search", "reduced")),
        "--search-strategy",
        str(settings.get("search_strategy", "adaptive")),
        "--optimizer",
        str(settings.get("optimizer", "adaptive")),
        "--baseline-profile",
        str(config_path),
    ]
    if settings.get("budget_hours") not in {None, ""}:
        budget_hours = float(settings.get("budget_hours") or 0.0)
        if budget_hours > 0.0:
            cmd.extend(["--budget-hours", f"{budget_hours:.6g}"])
    max_evaluations = _int_or_none(settings.get("max_evaluations"))
    if max_evaluations is not None and max_evaluations > 0:
        cmd.extend(["--max-evaluations", str(max_evaluations)])
    if bool(settings.get("optimize_controller", False)):
        cmd.append("--optimize-controller")
    if settings.get("optimizer") == "genetic":
        for setting_key, cli_key, cast in [
            ("ga_population", "--ga-population", int),
            ("ga_generations_quick", "--ga-generations-quick", int),
            ("ga_generations_reduced", "--ga-generations-reduced", int),
            ("ga_elite_fraction", "--ga-elite-fraction", float),
            ("ga_mutation_sigma", "--ga-mutation-sigma", float),
            ("ga_crossover_rate", "--ga-crossover-rate", float),
        ]:
            value = settings.get(setting_key)
            if value not in {None, ""}:
                cmd.extend([cli_key, str(cast(value))])
        if not bool(settings.get("ga_local_seeds", True)):
            cmd.append("--ga-no-local-seeds")
        seed_run = str(settings.get("ga_seed_run", "") or "").strip()
        if seed_run:
            cmd.extend(["--ga-seed-run", seed_run])
            for candidate_id in [
                item.strip()
                for item in str(settings.get("ga_seed_candidates", "") or "").split(",")
                if item.strip()
            ]:
                cmd.extend(["--ga-seed-candidate", candidate_id])
            seed_top = _int_or_none(settings.get("ga_seed_top"))
            if seed_top is not None and seed_top > 0:
                cmd.extend(["--ga-seed-top", str(seed_top)])
    resume = str(settings.get("resume_run_dir", "") or "").strip()
    if resume:
        cmd.extend(["--resume", resume])
    workers = _int_or_none(settings.get("workers"))
    if workers is not None and workers > 0:
        cmd.extend(["--workers", str(workers)])
    mujoco_validate_top = int(settings.get("mujoco_validate_top", 0))
    if mujoco_validate_top > 0:
        cmd.extend(["--mujoco-validate-top", str(mujoco_validate_top)])
        mujoco_workers = _int_or_none(settings.get("mujoco_workers"))
        if mujoco_workers is not None and mujoco_workers > 0:
            cmd.extend(["--mujoco-workers", str(mujoco_workers)])
        if bool(settings.get("mujoco_video", False)):
            cmd.append("--mujoco-video")
        if bool(settings.get("mujoco_contacts", False)):
            cmd.append("--mujoco-contacts")
    job.command = cmd
    return_code, output = _run_cli(cmd, job)
    if return_code != 0:
        raise RuntimeError(output)
    run_dir = _parse_path_from_output(output, "mount_optimization_run")
    report = _parse_path_from_output(output, "report")
    result = {
        "group_name": group_name,
        "task_count": len(task_names),
        "config_path": str(config_path),
        "run_dir": run_dir,
        "report": report,
        "report_url": _artifact_url(report) if report else "",
        "best_config": _parse_path_from_output(output, "best_config"),
        "candidates": _parse_path_from_output(output, "candidates"),
        "stdout": output,
    }
    result["simulations"] = _simulation_options_from_configs(
        [("baseline", config_path), ("best_candidate", result["best_config"])],
        task_names,
        group_name,
    )
    if bool(settings.get("write_task_reports", False)):
        report_base_search = str(settings.get("task_report_base_search", "full"))
        baseline_cfg = json.loads(json.dumps(_jsonable(cfg)))
        _set_base_search_mode(baseline_cfg, report_base_search)
        baseline_cfg["_profile"] = f"{_profile}_baseline_task_reports"
        report_configs: list[tuple[str, dict[str, Any]]] = [("baseline", baseline_cfg)]
        top_count = int(settings.get("task_report_top_count", settings.get("validate_top", 3)))
        for candidate in _top_mount_candidates_for_reports(Path(run_dir), top_count):
            candidate_cfg = apply_mount_candidate(cfg, candidate, base_search=report_base_search, serial_evaluation=False)
            candidate_cfg["_profile"] = f"{_profile}_{candidate.candidate_id}_task_reports"
            report_configs.append((candidate.candidate_id, candidate_cfg))
        result.update(
            _run_task_video_reports(
                settings=settings,
                configs=report_configs,
                scope_args=scope_args,
                group_name=group_name,
                index_dir=Path(run_dir) / "mujoco_task_reports",
                job=job,
            )
        )
    return result


def _summary_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "summary.csv")
    frame.insert(0, "configuration", label)
    return frame


def _task_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "task_success_matrix.csv")
    frame.insert(0, "configuration", label)
    return frame


def _write_manual_comparison_report(
    run_dir: Path,
    manual_run: Path,
    baseline_run: Path,
    candidate: MountCandidate,
    mount_info: dict[str, Any],
    simulation_index: Path | None = None,
) -> Path:
    manual_summary = _summary_frame(manual_run, "manual_mount")
    baseline_summary = _summary_frame(baseline_run, "baseline")
    summary = pd.concat([manual_summary, baseline_summary], ignore_index=True)
    manual_tasks = _task_frame(manual_run, "manual_mount")
    baseline_tasks = _task_frame(baseline_run, "baseline")
    tasks = pd.concat([manual_tasks, baseline_tasks], ignore_index=True)
    summary.to_csv(run_dir / "comparison_summary.csv", index=False)
    tasks.to_csv(run_dir / "comparison_task_metrics.csv", index=False)

    key_columns = [
        "configuration",
        "task_count",
        "success_count",
        "success_fraction",
        "mean_position_error_m",
        "max_position_error_m",
        "mean_orientation_error_rad",
        "max_orientation_error_rad",
        "min_manipulability",
        "max_torque_utilization",
        "max_joint_speed_utilization",
        "min_clearance_m",
        "worst_task_by_objective",
        "most_common_primary_failure",
    ]
    task_columns = [
        "configuration",
        "task_name",
        "success",
        "primary_failure",
        "failure_reasons",
        "mean_position_error_m",
        "max_position_error_m",
        "mean_orientation_error_rad",
        "max_orientation_error_rad",
        "min_manipulability",
        "max_torque_utilization",
        "max_joint_speed_utilization",
        "min_clearance_m",
    ]
    summary_html = summary[[col for col in key_columns if col in summary]].to_html(index=False, float_format=lambda value: f"{value:.5f}")
    task_html = tasks[[col for col in task_columns if col in tasks]].to_html(index=False, float_format=lambda value: f"{value:.5f}")
    mount_table = pd.DataFrame([{**candidate.as_record(), **mount_info}]).to_html(index=False, float_format=lambda value: f"{value:.5f}")
    simulation_html = ""
    if simulation_index is not None and simulation_index.exists():
        simulation_html = f'<h2>MuJoCo Task Reports</h2><p>{_html_link(simulation_index, "task reports and videos")}</p>'
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Manual Mount Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #182027; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
th, td {{ border: 1px solid #d6dde3; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f3f6f8; }}
a {{ color: #0a5c7a; }}
</style></head>
<body>
<h1>Manual Mount Comparison</h1>
<h2>Mount</h2>
{mount_table}
<h2>Summary</h2>
{summary_html}
<h2>Per Task</h2>
{task_html}
<h2>Source Reports</h2>
<p><a href="{_artifact_url(manual_run / "report.html")}">manual report</a> | <a href="{_artifact_url(baseline_run / "report.html")}">baseline report</a></p>
{simulation_html}
</body></html>
"""
    output = run_dir / "report.html"
    output.write_text(html, encoding="utf-8")
    return output


def _run_manual_evaluation_job(settings: dict[str, Any], job: GuiJob) -> dict[str, Any]:
    profile, base_cfg = _load_gui_config(settings)
    group_name, task_names, scope_args = _scope_args(settings)
    candidate = _mount_candidate_from_settings(settings)
    base_search = str(settings.get("manual_base_search", settings.get("base_search", "full")))
    manual_cfg = apply_mount_candidate(base_cfg, candidate, base_search=base_search, serial_evaluation=False)
    manual_cfg["_profile"] = f"{profile}_manual_mount"
    manual_cfg.setdefault("baseline", {})["name"] = "manual_mount_dual_panda"
    baseline_cfg = json.loads(json.dumps(_jsonable(base_cfg)))
    _set_base_search_mode(baseline_cfg, base_search)
    baseline_cfg["_profile"] = f"{profile}_reference"

    run_dir = PATHS.runs / timestamp_id(f"manual_mount_comparison_{group_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manual_config = run_dir / "manual_config.yaml"
    baseline_config = run_dir / "baseline_config.yaml"
    write_yaml(manual_config, manual_cfg)
    write_yaml(baseline_config, baseline_cfg)

    cmd_manual = [sys.executable, "-m", "garmi_optimization.cli", "baseline", *scope_args, "--baseline-profile", str(manual_config)]
    cmd_baseline = [sys.executable, "-m", "garmi_optimization.cli", "baseline", *scope_args, "--baseline-profile", str(baseline_config)]
    job.command = cmd_manual
    code_manual, output_manual = _run_cli(cmd_manual, job)
    if code_manual != 0:
        raise RuntimeError(output_manual)
    job.command = cmd_baseline
    code_baseline, output_baseline = _run_cli(cmd_baseline, job)
    if code_baseline != 0:
        raise RuntimeError(output_baseline)

    manual_run = Path(_parse_path_from_output(output_manual, "baseline_run"))
    baseline_run = Path(_parse_path_from_output(output_baseline, "baseline_run"))
    mount_info = mount_plausibility(candidate)
    task_report_result: dict[str, Any] = {}
    if bool(settings.get("write_task_reports", False)):
        task_report_result = _run_task_video_reports(
            settings=settings,
            configs=[("manual_mount", manual_cfg), ("baseline", baseline_cfg)],
            scope_args=scope_args,
            group_name=group_name,
            index_dir=run_dir / "mujoco_task_reports",
            job=job,
        )
    simulation_index = Path(task_report_result["task_report_index"]) if task_report_result.get("task_report_index") else None
    report = _write_manual_comparison_report(run_dir, manual_run, baseline_run, candidate, mount_info, simulation_index)
    result = {
        "group_name": group_name,
        "task_count": len(task_names),
        "run_dir": str(run_dir),
        "report": str(report),
        "report_url": _artifact_url(report),
        "manual_report": str(manual_run / "report.html"),
        "manual_report_url": _artifact_url(manual_run / "report.html"),
        "baseline_report": str(baseline_run / "report.html"),
        "baseline_report_url": _artifact_url(baseline_run / "report.html"),
        "manual_config": str(manual_config),
        "baseline_config": str(baseline_config),
        "stdout": output_manual + "\n" + output_baseline,
        "simulations": _simulation_options_from_configs(
            [("manual_mount", manual_config), ("baseline", baseline_config)],
            task_names,
            group_name,
        ),
    }
    result.update(task_report_result)
    return result


def _proxy_to_json(proxy: SphereProxy | CapsuleProxy | BoxProxy) -> dict[str, Any]:
    if isinstance(proxy, SphereProxy):
        return {"type": "sphere", "name": proxy.name, "center": proxy.center, "radius": proxy.radius}
    if isinstance(proxy, CapsuleProxy):
        return {"type": "capsule", "name": proxy.name, "p0": proxy.p0, "p1": proxy.p1, "radius": proxy.radius}
    return {"type": "box", "name": proxy.name, "center": proxy.center, "half_extents": proxy.half_extents}


def manual_geometry(settings: dict[str, Any]) -> dict[str, Any]:
    _profile, cfg = _load_gui_config(settings)
    candidate = _mount_candidate_from_settings(settings)
    candidate_cfg = apply_mount_candidate(cfg, candidate, base_search=str(settings.get("manual_base_search", "full")), serial_evaluation=False)
    model = PandaModel.from_config(candidate_cfg)
    base_pose = np.zeros(3, dtype=float)
    left_points = forward_points(model, model.q_nominal, "left", base_pose)
    right_points = forward_points(model, model.q_nominal, "right", base_pose)
    z_axis = np.array([0.0, 0.0, 1.0])
    axes = {
        side: model.shoulder_world(side, base_pose) + 0.22 * (model.mount_rotation_world(side, 0.0) @ z_axis)
        for side in ["left", "right"]
    }
    return _jsonable(
        {
            "candidate": candidate.as_record(),
            "mount_plausibility": mount_plausibility(candidate),
            "left_points": left_points,
            "right_points": right_points,
            "left_mount": model.shoulder_world("left", base_pose),
            "right_mount": model.shoulder_world("right", base_pose),
            "left_axis_end": axes["left"],
            "right_axis_end": axes["right"],
            "body": [_proxy_to_json(proxy) for proxy in garmi_body_proxies()],
        }
    )


def _launch_interactive_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    task_name = str(payload.get("task_name", "")).strip()
    if not task_name:
        raise ValueError("task_name is required")
    config_path = Path(str(payload.get("config_path", ""))).expanduser().resolve()
    root = PATHS.root.resolve()
    if not config_path.is_relative_to(root) or not config_path.exists() or not config_path.is_file():
        raise ValueError("config_path must point to an existing project config file")
    realtime_factor = max(0.05, float(payload.get("realtime_factor", 1.0) or 1.0))
    if "use_garmi_follow" in payload:
        camera = "garmi_follow" if bool(payload.get("use_garmi_follow")) else "overview"
    else:
        camera = str(payload.get("camera", "") or "")
    log_dir = PATHS.runs / "gui_viewers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp_id('viewer')}_{_safe_label(task_name)}.log"
    env = os.environ.copy()
    src_path = str(PATHS.root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    executable = Path(sys.executable)
    mjpython = executable.parent / "mjpython"
    viewer_executable = str(mjpython if sys.platform == "darwin" and mjpython.exists() else executable)
    cmd = [
        viewer_executable,
        "-m",
        "garmi_optimization.cli",
        "view-simulation",
        "--task",
        task_name,
        "--baseline-profile",
        str(config_path),
        "--realtime-factor",
        f"{realtime_factor:.6g}",
    ]
    if camera in {"garmi_follow", "overview"}:
        cmd.extend(["--camera", camera])
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(PATHS.root),
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(0.75)
    return_code = process.poll()
    if return_code is not None:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
        raise RuntimeError(f"MuJoCo viewer exited immediately with code {return_code}.\n{log_text}")
    return {
        "pid": process.pid,
        "task_name": task_name,
        "config_path": str(config_path),
        "log": str(log_path),
        "log_url": _artifact_url(log_path),
        "command": cmd,
    }


def _default_settings() -> dict[str, Any]:
    cfg = load_baseline_config(profile=DEFAULT_PROFILE)
    panda = cfg["panda"]
    left = panda.get("mounts", {}).get("left", {}).get("xyz_m", [panda.get("shoulder_x_m", -0.14326), 0.20, panda.get("shoulder_height_m", 0.92451)])
    optimization_cfg = cfg.get("optimization", {})
    mount_bounds = {name: [float(low), float(high)] for name, (low, high) in _candidate_bounds(cfg).items()}
    optimization_defaults = {
        "mount_bounds": mount_bounds,
        "mount_local_scales": {
            "left_x_m": 0.055,
            "left_y_m": 0.045,
            "left_z_m": 0.055,
            "left_delta_roll_rad": 0.16,
            "left_delta_pitch_rad": 0.14,
            "left_delta_yaw_rad": 0.18,
        },
        "mount_local_sample_fraction": float(optimization_cfg.get("mount_local_sample_fraction", 1.0 / 3.0)),
    }
    optimization_defaults["mount_local_scales"].update(optimization_cfg.get("mount_local_scales", {}))
    return {
        "baseline_profile": DEFAULT_PROFILE,
        "group_name": "basic_care",
        "global_opt": False,
        "samples": 24,
        "workers": max(1, min((os.cpu_count() or 8) - 2, 10)),
        "seed": 7,
        "validate_top": 3,
        "base_search": "reduced",
        "search_strategy": "adaptive",
        "optimizer": cfg.get("optimization", {}).get("optimizer", "multifidelity"),
        "budget_hours": "",
        "max_evaluations": "",
        "optimize_controller": False,
        "resume_run_dir": "",
        "optimization_promote_fraction": float(cfg.get("optimization", {}).get("promote_fraction", 0.45)),
        "optimization_surrogate_pool_size": int(cfg.get("optimization", {}).get("surrogate_pool_size", 512)),
        "ga_population": int(cfg.get("optimization", {}).get("genetic", {}).get("population", 64)),
        "ga_generations_quick": int(cfg.get("optimization", {}).get("genetic", {}).get("quick_generations", 5)),
        "ga_generations_reduced": int(cfg.get("optimization", {}).get("genetic", {}).get("reduced_generations", 4)),
        "ga_elite_fraction": float(cfg.get("optimization", {}).get("genetic", {}).get("elite_fraction", 0.20)),
        "ga_mutation_sigma": float(cfg.get("optimization", {}).get("genetic", {}).get("mutation_sigma", 0.12)),
        "ga_crossover_rate": float(cfg.get("optimization", {}).get("genetic", {}).get("crossover_rate", 0.85)),
        "ga_local_seeds": float(cfg.get("optimization", {}).get("genetic", {}).get("local_seed_fraction", 0.25)) > 0.0,
        "ga_seed_run": str(cfg.get("optimization", {}).get("genetic", {}).get("seed_run", "")),
        "ga_seed_candidates": ",".join(cfg.get("optimization", {}).get("genetic", {}).get("seed_candidates", [])),
        "ga_seed_top": int(cfg.get("optimization", {}).get("genetic", {}).get("seed_top", 0)),
        "mount_rotation_mode": str(optimization_cfg.get("mount_rotation_mode", "rpy_delta")),
        "mujoco_validate_top": 0,
        "mujoco_workers": max(1, min((os.cpu_count() or 8) - 2, 4)),
        "mujoco_video": False,
        "mujoco_contacts": False,
        "manual_base_search": "full",
        "write_task_reports": False,
        "task_report_top_count": 3,
        "task_report_workers": max(1, min((os.cpu_count() or 8) - 2, 4)),
        "task_report_base_search": "full",
        "task_report_render_videos": True,
        "simulation_use_garmi_follow": True,
        "performance_parallel_workers": "auto",
        "performance_progress": True,
        "manual_left_x_m": float(left[0]),
        "manual_left_y_m": float(abs(left[1])),
        "manual_left_z_m": float(left[2]),
        "manual_left_delta_roll_rad": 0.0,
        "manual_left_delta_pitch_rad": 0.0,
        "manual_left_delta_yaw_rad": 0.0,
        **{f"base_{key}": value for key, value in cfg.get("base", {}).items()},
        **{f"ik_{key}": value for key, value in cfg.get("ik", {}).items()},
        **{f"collision_{key}": value for key, value in cfg.get("collision", {}).items()},
        **{f"success_{key}": value for key, value in cfg.get("success", {}).items()},
        **{f"simulation_validation_{key}": value for key, value in cfg.get("simulation_validation", {}).items()},
        **{f"controller_{key}": value for key, value in cfg.get("controller", {}).items() if key != "base_pd"},
        "controller_base_pd_kp": cfg.get("controller", {}).get("base_pd", {}).get("kp", [2400.0, 2400.0, 850.0]),
        "controller_base_pd_kd": cfg.get("controller", {}).get("base_pd", {}).get("kd", [520.0, 520.0, 160.0]),
        **{
            f"bound_{name}_min": values[0]
            for name, values in optimization_defaults["mount_bounds"].items()
        },
        **{
            f"bound_{name}_max": values[1]
            for name, values in optimization_defaults["mount_bounds"].items()
        },
        **{
            f"scale_{name}": value
            for name, value in optimization_defaults["mount_local_scales"].items()
        },
        "mount_local_sample_fraction": optimization_defaults["mount_local_sample_fraction"],
    }


def _start_job(kind: str, settings: dict[str, Any], target: Callable[[dict[str, Any], GuiJob], dict[str, Any]]) -> GuiJob:
    ensure_dirs()
    job = GuiJob(job_id=timestamp_id(kind), kind=kind)
    with JOB_LOCK:
        JOBS[job.job_id] = job

    def runner() -> None:
        with JOB_LOCK:
            job.status = "running"
            job.message = "running"
        try:
            result = target(settings, job)
            with JOB_LOCK:
                job.status = "complete"
                job.message = "complete"
                job.result = result
                job.finished_s = time.time()
        except Exception:
            with JOB_LOCK:
                job.status = "failed"
                job.message = "failed"
                job.log = (job.log + "\n" + traceback.format_exc())[-12000:]
                job.finished_s = time.time()

    threading.Thread(target=runner, daemon=True).start()
    return job


def index_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GARMI Mount Optimization</title>
<style>
:root { --bg: #f5f7f8; --panel: #ffffff; --line: #d7dee4; --text: #17212b; --muted: #64717d; --accent: #0b6f86; --accent2: #b76019; --ok: #287447; --bad: #a33b35; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }
header { height: 56px; display: flex; align-items: center; justify-content: space-between; padding: 0 18px; border-bottom: 1px solid var(--line); background: #ffffff; }
h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: 0; }
main { display: grid; grid-template-columns: 360px minmax(0, 1fr); min-height: calc(100vh - 56px); }
aside { border-right: 1px solid var(--line); background: #fbfcfd; padding: 14px; overflow: auto; }
section { padding: 14px 16px 18px; overflow: auto; }
	.tabs { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-bottom: 12px; }
.tabs button, .primary, .secondary { border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 6px; height: 34px; font: inherit; cursor: pointer; }
.tabs button.active, .primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.secondary { background: #f7fafb; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.field { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
label { font-size: 12px; color: var(--muted); }
input, select { width: 100%; min-height: 32px; border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; background: #fff; color: var(--text); font: inherit; }
input[type="checkbox"] { width: 18px; height: 18px; min-height: 18px; padding: 0; }
.checkrow { display: flex; align-items: center; gap: 8px; min-height: 32px; }
.panel { display: none; }
.panel.active { display: block; }
.block { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; margin-bottom: 12px; }
.block h2 { font-size: 14px; margin: 0 0 10px; }
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.wide { grid-column: 1 / -1; }
.viewer { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.canvasPanel { border: 1px solid var(--line); border-radius: 8px; background: #fff; min-width: 0; }
.canvasPanel h2 { margin: 10px 12px 4px; font-size: 14px; }
canvas { width: 100%; height: 330px; display: block; }
.metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
.metric { border: 1px solid var(--line); background: #fff; border-radius: 8px; padding: 10px; min-height: 70px; }
.metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.metric strong { font-size: 17px; font-weight: 650; overflow-wrap: anywhere; }
.status { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background: #101820; color: #dbe7ee; border-radius: 8px; padding: 10px; max-height: 260px; overflow: auto; }
	.links a { display: inline-flex; align-items: center; margin: 0 10px 8px 0; color: var(--accent); text-decoration: none; font-weight: 600; }
	.listBox { border: 1px solid var(--line); border-radius: 8px; background: #fff; min-height: 160px; max-height: 340px; overflow: auto; }
	.listItem { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; padding: 7px 8px; border-bottom: 1px solid #edf1f4; }
	.listItem:last-child { border-bottom: 0; }
	.listItem strong { display: block; font-size: 13px; overflow-wrap: anywhere; }
	.listItem span { display: block; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
	.smallButton { min-width: 34px; height: 28px; border: 1px solid var(--line); border-radius: 6px; background: #f7fafb; cursor: pointer; }
	.danger { border-color: #d9a4a0; color: var(--bad); }
	.groupStats { color: var(--muted); font-size: 12px; margin-top: 8px; }
	table { width: 100%; border-collapse: collapse; background: #fff; }
td, th { border: 1px solid var(--line); padding: 6px 8px; font-size: 12px; text-align: left; }
th { background: #f1f5f7; }
@media (max-width: 1080px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid var(--line); } .viewer { grid-template-columns: 1fr; } .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
</head>
<body>
<header>
  <h1>GARMI Panda Mount Optimization</h1>
  <div id="headerState">idle</div>
</header>
<main>
<aside>
  <div class="block">
    <h2>Run</h2>
    <div class="field"><label>Baseline profile</label><select data-key="baseline_profile" id="baselineProfile"></select></div>
    <div class="field"><label>Task group</label><select data-key="group_name" id="groupName"></select></div>
    <div class="checkrow"><input type="checkbox" data-key="global_opt" id="globalOpt"><label for="globalOpt">global</label></div>
    <div class="grid">
      <div class="field"><label>samples</label><input type="number" data-key="samples"></div>
      <div class="field"><label>workers</label><input type="number" data-key="workers"></div>
      <div class="field"><label>seed</label><input type="number" data-key="seed"></div>
      <div class="field"><label>validate top</label><input type="number" data-key="validate_top"></div>
      <div class="field"><label>base search</label><select data-key="base_search"><option>reduced</option><option>single</option><option>full</option></select></div>
      <div class="field"><label>search strategy</label><select data-key="search_strategy"><option>adaptive</option><option>hybrid</option><option>lhs</option><option>sobol</option><option>random</option></select></div>
      <div class="field"><label>optimizer</label><select data-key="optimizer"><option>adaptive</option><option>multifidelity</option><option>surrogate</option><option>genetic</option></select></div>
      <div class="checkrow"><input type="checkbox" data-key="optimize_controller" id="optimizeController"><label for="optimizeController">optimize controller</label></div>
      <div class="field"><label>budget hours</label><input type="number" step="0.1" data-key="budget_hours"></div>
      <div class="field"><label>max evaluations</label><input type="number" data-key="max_evaluations"></div>
      <div class="field"><label>promote fraction</label><input type="number" step="0.01" data-key="optimization_promote_fraction"></div>
      <div class="field"><label>surrogate pool</label><input type="number" data-key="optimization_surrogate_pool_size"></div>
      <div class="field"><label>GA population</label><input type="number" data-key="ga_population"></div>
      <div class="field"><label>GA quick gens</label><input type="number" data-key="ga_generations_quick"></div>
      <div class="field"><label>GA reduced gens</label><input type="number" data-key="ga_generations_reduced"></div>
      <div class="field"><label>GA elite</label><input type="number" step="0.01" data-key="ga_elite_fraction"></div>
      <div class="field"><label>GA mutation</label><input type="number" step="0.01" data-key="ga_mutation_sigma"></div>
      <div class="field"><label>GA crossover</label><input type="number" step="0.01" data-key="ga_crossover_rate"></div>
      <div class="checkrow"><input type="checkbox" data-key="ga_local_seeds" id="gaLocalSeeds"><label for="gaLocalSeeds">GA local seeds</label></div>
      <div class="field wide"><label>GA seed run</label><input data-key="ga_seed_run"></div>
      <div class="field wide"><label>GA seed candidates</label><input data-key="ga_seed_candidates"></div>
      <div class="field"><label>GA seed top</label><input type="number" data-key="ga_seed_top"></div>
      <div class="field"><label>manual base</label><select data-key="manual_base_search"><option>full</option><option>reduced</option><option>single</option></select></div>
      <div class="field wide"><label>resume run dir</label><input data-key="resume_run_dir"></div>
    </div>
    <div class="actions">
      <button class="primary" id="startOptimization">Start optimization</button>
      <button class="secondary" id="evaluateManual">Evaluate manual</button>
    </div>
  </div>
  <div class="block">
    <h2>MuJoCo validation</h2>
    <div class="grid">
      <div class="field"><label>validate top</label><input type="number" data-key="mujoco_validate_top"></div>
      <div class="field"><label>workers</label><input type="number" data-key="mujoco_workers"></div>
      <div class="checkrow"><input type="checkbox" data-key="mujoco_video" id="mujocoVideo"><label for="mujocoVideo">render videos</label></div>
      <div class="checkrow"><input type="checkbox" data-key="mujoco_contacts" id="mujocoContacts"><label for="mujocoContacts">enable contacts</label></div>
    </div>
  </div>
  <div class="block">
    <h2>Task reports</h2>
    <div class="checkrow"><input type="checkbox" data-key="write_task_reports" id="writeTaskReports"><label for="writeTaskReports">write per-task reports</label></div>
    <div class="checkrow"><input type="checkbox" data-key="task_report_render_videos" id="taskReportVideos"><label for="taskReportVideos">include MuJoCo videos</label></div>
    <div class="checkrow"><input type="checkbox" data-key="simulation_use_garmi_follow" id="simulationUseFollow"><label for="simulationUseFollow">garmi follow camera</label></div>
    <div class="grid">
      <div class="field"><label>top candidates</label><input type="number" data-key="task_report_top_count"></div>
      <div class="field"><label>video workers</label><input type="number" data-key="task_report_workers"></div>
      <div class="field wide"><label>task report base</label><select data-key="task_report_base_search"><option>full</option><option>reduced</option><option>single</option></select></div>
    </div>
  </div>
  <div class="block">
    <h2>Manual mount</h2>
    <div class="grid3">
      <div class="field"><label>x m</label><input type="number" step="0.001" data-key="manual_left_x_m"></div>
      <div class="field"><label>y m</label><input type="number" step="0.001" data-key="manual_left_y_m"></div>
      <div class="field"><label>z m</label><input type="number" step="0.001" data-key="manual_left_z_m"></div>
      <div class="field"><label>roll d rad</label><input type="number" step="0.001" data-key="manual_left_delta_roll_rad"></div>
      <div class="field"><label>pitch d rad</label><input type="number" step="0.001" data-key="manual_left_delta_pitch_rad"></div>
      <div class="field"><label>yaw d rad</label><input type="number" step="0.001" data-key="manual_left_delta_yaw_rad"></div>
    </div>
  </div>
</aside>
<section>
  <div class="tabs">
    <button class="active" data-tab="visual">Visual</button>
	    <button data-tab="search">Search</button>
	    <button data-tab="eval">Evaluation</button>
	    <button data-tab="groups">Groups</button>
	    <button data-tab="results">Results</button>
	  </div>

  <div class="panel active" id="visual">
    <div class="metrics" id="mountMetrics"></div>
    <div class="viewer">
      <div class="canvasPanel"><h2>Top x/y</h2><canvas id="topCanvas" width="620" height="360"></canvas></div>
      <div class="canvasPanel"><h2>Front y/z</h2><canvas id="frontCanvas" width="620" height="360"></canvas></div>
      <div class="canvasPanel"><h2>Side x/z</h2><canvas id="sideCanvas" width="620" height="360"></canvas></div>
    </div>
  </div>

  <div class="panel" id="search">
    <div class="block">
      <h2>Bounds</h2>
      <div class="grid">
        <div class="field"><label>x min</label><input type="number" step="0.001" data-key="bound_left_x_m_min"></div>
        <div class="field"><label>x max</label><input type="number" step="0.001" data-key="bound_left_x_m_max"></div>
        <div class="field"><label>y min</label><input type="number" step="0.001" data-key="bound_left_y_m_min"></div>
        <div class="field"><label>y max</label><input type="number" step="0.001" data-key="bound_left_y_m_max"></div>
        <div class="field"><label>z min</label><input type="number" step="0.001" data-key="bound_left_z_m_min"></div>
        <div class="field"><label>z max</label><input type="number" step="0.001" data-key="bound_left_z_m_max"></div>
      </div>
    </div>
    <div class="block">
      <h2>Rotation model</h2>
      <div class="grid">
        <div class="field wide"><label>mode</label><select data-key="mount_rotation_mode"><option>forward_outer_quarter_sphere</option><option>rpy_delta</option></select></div>
        <div class="field"><label>roll min</label><input type="number" step="0.001" data-key="bound_left_delta_roll_rad_min"></div>
        <div class="field"><label>roll max</label><input type="number" step="0.001" data-key="bound_left_delta_roll_rad_max"></div>
        <div class="field"><label>elev min</label><input type="number" step="0.001" data-key="bound_left_delta_pitch_rad_min"></div>
        <div class="field"><label>elev max</label><input type="number" step="0.001" data-key="bound_left_delta_pitch_rad_max"></div>
        <div class="field"><label>azim min</label><input type="number" step="0.001" data-key="bound_left_delta_yaw_rad_min"></div>
        <div class="field"><label>azim max</label><input type="number" step="0.001" data-key="bound_left_delta_yaw_rad_max"></div>
      </div>
    </div>
    <div class="block">
      <h2>Local sampling</h2>
      <div class="grid3">
        <div class="field"><label>x scale</label><input type="number" step="0.001" data-key="scale_left_x_m"></div>
        <div class="field"><label>y scale</label><input type="number" step="0.001" data-key="scale_left_y_m"></div>
        <div class="field"><label>z scale</label><input type="number" step="0.001" data-key="scale_left_z_m"></div>
        <div class="field"><label>roll d scale</label><input type="number" step="0.001" data-key="scale_left_delta_roll_rad"></div>
        <div class="field"><label>pitch d scale</label><input type="number" step="0.001" data-key="scale_left_delta_pitch_rad"></div>
        <div class="field"><label>yaw d scale</label><input type="number" step="0.001" data-key="scale_left_delta_yaw_rad"></div>
        <div class="field"><label>local fraction</label><input type="number" step="0.01" data-key="mount_local_sample_fraction"></div>
      </div>
    </div>
  </div>

  <div class="panel" id="eval">
    <div class="block">
      <h2>Base variants</h2>
      <div class="grid">
        <div class="field wide"><label>motion mode</label><select data-key="base_motion_mode"><option value="fixed_world">A fixed world</option><option value="fixed_initial">B fixed initial</option><option value="limited_holonomic">C limited holonomic</option><option value="nonholonomic">D nonholonomic</option><option value="free_holonomic">E free holonomic</option></select></div>
        <div class="field wide"><label>fixed pose x,y,yaw</label><input data-key="base_fixed_pose"></div>
        <div class="field"><label>reach fractions</label><input data-key="base_preferred_reach_fraction_candidates"></div>
        <div class="field"><label>forward offsets m</label><input data-key="base_forward_offset_candidates_m"></div>
        <div class="field"><label>lateral offsets m</label><input data-key="base_lateral_offset_candidates_m"></div>
        <div class="field"><label>yaw offsets rad</label><input data-key="base_yaw_offset_candidates_rad"></div>
        <div class="field"><label>base max speed</label><input type="number" step="0.01" data-key="base_max_speed_mps"></div>
        <div class="field"><label>base max acc</label><input type="number" step="0.01" data-key="base_max_acc_mps2"></div>
        <div class="field"><label>C max path m</label><input type="number" step="0.01" data-key="base_limited_max_path_m"></div>
        <div class="field"><label>C max speed</label><input type="number" step="0.01" data-key="base_limited_max_speed_mps"></div>
        <div class="field"><label>C yaw rate</label><input type="number" step="0.01" data-key="base_limited_max_yaw_rate_radps"></div>
        <div class="field"><label>D max speed</label><input type="number" step="0.01" data-key="base_nonholonomic_max_speed_mps"></div>
        <div class="field"><label>D yaw rate</label><input type="number" step="0.01" data-key="base_nonholonomic_max_yaw_rate_radps"></div>
        <div class="field"><label>smoothing window</label><input type="number" data-key="base_smoothing_window"></div>
      </div>
    </div>
    <div class="block">
      <h2>IK</h2>
      <div class="grid3">
        <div class="field"><label>iterations</label><input type="number" data-key="ik_iterations"></div>
        <div class="field"><label>damping</label><input type="number" step="0.001" data-key="ik_damping"></div>
        <div class="field"><label>step limit</label><input type="number" step="0.001" data-key="ik_step_limit_rad"></div>
        <div class="field"><label>pos tolerance</label><input type="number" step="0.001" data-key="ik_position_tolerance_m"></div>
        <div class="field"><label>orientation weight</label><input type="number" step="0.001" data-key="ik_orientation_weight"></div>
        <div class="field"><label>orientation tol</label><input type="number" step="0.001" data-key="ik_orientation_tolerance_rad"></div>
        <div class="field"><label>joint center</label><input type="number" step="0.001" data-key="ik_joint_center_weight"></div>
        <div class="field"><label>continuity</label><input type="number" step="0.001" data-key="ik_continuity_weight"></div>
        <div class="field"><label>orientation mode</label><select data-key="ik_orientation_mode"><option>relative</option><option>absolute</option></select></div>
        <div class="checkrow"><input type="checkbox" data-key="ik_use_orientation" id="ikUseOrientation"><label for="ikUseOrientation">orientation</label></div>
      </div>
    </div>
    <div class="block">
      <h2>Impedance controller</h2>
      <div class="grid">
        <div class="field"><label>position kp</label><input data-key="controller_position_kp_npm"></div>
        <div class="field"><label>position kd</label><input data-key="controller_position_kd_nspm"></div>
        <div class="field"><label>orientation kp</label><input data-key="controller_orientation_kp_nmprad"></div>
        <div class="field"><label>orientation kd</label><input data-key="controller_orientation_kd_nmsprad"></div>
        <div class="field"><label>nullspace kp</label><input data-key="controller_nullspace_kp_nmprad"></div>
        <div class="field"><label>nullspace kd</label><input data-key="controller_nullspace_kd_nmsprad"></div>
        <div class="field"><label>max task force</label><input type="number" step="1" data-key="controller_max_task_force_n"></div>
        <div class="field"><label>max task torque</label><input type="number" step="1" data-key="controller_max_task_torque_nm"></div>
        <div class="field"><label>null damping</label><input type="number" step="0.001" data-key="controller_nullspace_damping"></div>
        <div class="field"><label>base kp</label><input data-key="controller_base_pd_kp"></div>
        <div class="field"><label>base kd</label><input data-key="controller_base_pd_kd"></div>
        <div class="checkrow"><input type="checkbox" data-key="controller_bias_compensation" id="biasCompensation"><label for="biasCompensation">bias compensation</label></div>
      </div>
    </div>
    <div class="block">
      <h2>Success and collision</h2>
      <div class="grid3">
        <div class="field"><label>mean pos limit</label><input type="number" step="0.001" data-key="success_mean_position_error_m"></div>
        <div class="field"><label>max pos limit</label><input type="number" step="0.001" data-key="success_max_position_error_m"></div>
        <div class="field"><label>mean rot limit</label><input type="number" step="0.001" data-key="success_mean_orientation_error_rad"></div>
        <div class="field"><label>max rot limit</label><input type="number" step="0.001" data-key="success_max_orientation_error_rad"></div>
        <div class="field"><label>min manip</label><input type="number" step="0.001" data-key="success_min_manipulability"></div>
        <div class="field"><label>max torque util</label><input type="number" step="0.001" data-key="success_max_torque_utilization"></div>
        <div class="field"><label>min clearance</label><input type="number" step="0.001" data-key="success_min_clearance_m"></div>
        <div class="field"><label>hard clearance</label><input type="number" step="0.001" data-key="collision_hard_clearance_m"></div>
        <div class="field"><label>warning clearance</label><input type="number" step="0.001" data-key="collision_warning_clearance_m"></div>
        <div class="field"><label>sim max pos</label><input type="number" step="0.001" data-key="simulation_validation_max_position_error_m"></div>
        <div class="field"><label>sim max rot</label><input type="number" step="0.001" data-key="simulation_validation_max_orientation_error_rad"></div>
        <div class="field"><label>sim max clip</label><input type="number" step="0.001" data-key="simulation_validation_max_joint_torque_clip_fraction"></div>
        <div class="field"><label>sim contacts</label><input type="number" step="1" data-key="simulation_validation_max_contact_count"></div>
        <div class="field"><label>link radius</label><input type="number" step="0.001" data-key="collision_arm_link_radius_m"></div>
        <div class="field"><label>joint radius</label><input type="number" step="0.001" data-key="collision_joint_radius_m"></div>
        <div class="field"><label>ee radius</label><input type="number" step="0.001" data-key="collision_end_effector_radius_m"></div>
        <div class="field"><label>mount exclusion</label><input type="number" step="0.001" data-key="collision_shoulder_mount_exclusion_m"></div>
        <div class="field"><label>mount points</label><input type="number" data-key="collision_shoulder_mount_exclusion_points"></div>
        <div class="checkrow"><input type="checkbox" data-key="collision_enabled" id="collisionEnabled"><label for="collisionEnabled">collision</label></div>
        <div class="checkrow"><input type="checkbox" data-key="collision_collision_aware_ik" id="collisionAware"><label for="collisionAware">avoidance IK</label></div>
        <div class="checkrow"><input type="checkbox" data-key="simulation_validation_require_no_exception" id="simRequireNoException"><label for="simRequireNoException">sim must finish</label></div>
      </div>
    </div>
  </div>

	  <div class="panel" id="results">
    <div class="block">
      <h2>Job</h2>
      <div class="links" id="jobLinks"></div>
      <div class="status" id="jobStatus">idle</div>
    </div>
    <div class="block">
      <h2>Interactive MuJoCo</h2>
      <div class="grid">
        <div class="field"><label>simulation</label><select id="simulationSelect"></select></div>
        <div class="field"><label>realtime factor</label><input id="realtimeFactor" type="number" min="0.05" step="0.1" value="1.0"></div>
      </div>
      <div class="checkrow"><input type="checkbox" id="interactiveUseFollow" checked><label for="interactiveUseFollow">garmi follow camera</label></div>
      <div class="actions">
        <button class="primary" id="launchSimulation">Launch viewer</button>
      </div>
    </div>
	  </div>

	  <div class="panel" id="groups">
	    <div class="block">
	      <h2>Task Groups</h2>
	      <div class="grid">
	        <div class="field"><label>edit group</label><select id="editGroupSelect"></select></div>
	        <div class="field"><label>group id</label><input id="editGroupName"></div>
	        <div class="field wide"><label>description</label><input id="editGroupDescription"></div>
	      </div>
	      <div class="actions">
	        <button class="secondary" id="newGroup">New group</button>
	        <button class="primary" id="saveGroup">Save group</button>
	      </div>
	      <div class="actions">
	        <button class="secondary danger" id="deleteGroup">Delete group</button>
	      </div>
	      <div class="groupStats" id="groupStats"></div>
	    </div>
	    <div class="block">
	      <h2>Group Tasks</h2>
	      <div class="grid">
	        <div class="field"><label>available task</label><select id="groupTaskSelect"></select></div>
	        <div class="field"><label>&nbsp;</label><button class="secondary" id="addGroupTask">Add task</button></div>
	      </div>
	      <div class="listBox" id="groupTaskList"></div>
	    </div>
	    <div class="block">
	      <h2>Excluded Tasks</h2>
	      <div class="grid">
	        <div class="field"><label>task to exclude</label><select id="excludeTaskSelect"></select></div>
	        <div class="field"><label>&nbsp;</label><button class="secondary" id="addExcludedTask">Exclude task</button></div>
	      </div>
	      <div class="listBox" id="excludedTaskList"></div>
	    </div>
	  </div>
	</section>
	</main>
	<script>
	const state = { defaults: {}, geometry: null, jobId: null, poll: null, simulations: [], groups: {}, tasks: [], activeGroup: "" };
const keysAsLists = new Set([
  "base_fixed_pose","base_preferred_reach_fraction_candidates","base_forward_offset_candidates_m","base_lateral_offset_candidates_m","base_yaw_offset_candidates_rad",
  "controller_position_kp_npm","controller_position_kd_nspm","controller_orientation_kp_nmprad","controller_orientation_kd_nmsprad",
  "controller_nullspace_kp_nmprad","controller_nullspace_kd_nmsprad","controller_base_pd_kp","controller_base_pd_kd"
]);

function $(sel) { return document.querySelector(sel); }
function $all(sel) { return Array.from(document.querySelectorAll(sel)); }
function setTab(id) {
  $all(".tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === id));
  $all(".panel").forEach(p => p.classList.toggle("active", p.id === id));
}
function parseValue(input) {
  if (input.type === "checkbox") return input.checked;
  if (keysAsLists.has(input.dataset.key)) return input.value.split(",").map(v => Number(v.trim())).filter(v => !Number.isNaN(v));
  if (input.type === "number") return Number(input.value);
  return input.value;
}
function collectSettings() {
  const settings = {};
  $all("[data-key]").forEach(input => { settings[input.dataset.key] = parseValue(input); });
  return settings;
}
function setInput(key, value) {
  const input = document.querySelector(`[data-key="${key}"]`);
  if (!input || value === undefined || value === null) return;
  if (input.type === "checkbox") input.checked = Boolean(value);
  else if (Array.isArray(value)) input.value = value.join(", ");
  else input.value = value;
}
async function api(path, body) {
  const opts = body ? { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function taskInfo(taskId) {
  return state.tasks.find(t => t.task_id === taskId) || { task_id: taskId, display_name: taskId, category: "", status: "unknown", configured: false };
}
function taskLabel(taskId) {
  const info = taskInfo(taskId);
  return `${info.task_id}${info.configured ? "" : " [not configured]"}`;
}
function populateTaskSelect(select) {
  if (!select) return;
  const previous = select.value;
  select.innerHTML = "";
  state.tasks.filter(t => t.configured).forEach(task => {
    const label = `${task.task_id} | ${task.category || task.status}`;
    select.add(new Option(label, task.task_id));
  });
  if (previous) select.value = previous;
}
function refreshRunGroupSelect(selected) {
  const select = $("#groupName");
  const previous = selected || select.value || state.defaults.group_name;
  select.innerHTML = "";
  Object.keys(state.groups).forEach(name => {
    const group = state.groups[name];
    select.add(new Option(`${name} (${group.effective_tasks.length}/${group.tasks.length})`, name));
  });
  if (previous && state.groups[previous]) select.value = previous;
}
function refreshEditGroupSelect(selected) {
  const select = $("#editGroupSelect");
  const previous = selected || state.activeGroup || select.value;
  select.innerHTML = "";
  Object.keys(state.groups).forEach(name => {
    const group = state.groups[name];
    select.add(new Option(`${name} (${group.effective_tasks.length}/${group.tasks.length})`, name));
  });
  if (previous && state.groups[previous]) select.value = previous;
  state.activeGroup = select.value || Object.keys(state.groups)[0] || "";
}
function currentGroupDraft() {
  const group = state.groups[state.activeGroup] || { tasks: [], excluded_tasks: [], description: "" };
  return {
    name: $("#editGroupName").value.trim(),
    description: $("#editGroupDescription").value,
    tasks: [...group.tasks],
    excluded_tasks: [...group.excluded_tasks],
  };
}
function selectGroupForEdit(name) {
  state.activeGroup = name;
  const group = state.groups[name] || { tasks: [], excluded_tasks: [], description: "" };
  $("#editGroupName").value = name || "";
  $("#editGroupDescription").value = group.description || "";
  renderGroupEditor();
}
function renderGroupEditor() {
  const group = state.groups[state.activeGroup] || { tasks: [], excluded_tasks: [], effective_tasks: [], description: "" };
  $("#groupTaskList").innerHTML = group.tasks.map((taskId, index) => {
    const info = taskInfo(taskId);
    const excluded = group.excluded_tasks.includes(taskId);
    return `<div class="listItem">
      <div><strong>${esc(taskLabel(taskId))}${excluded ? " [excluded]" : ""}</strong><span>${esc(info.display_name)} | occurrence ${index + 1}</span></div>
      <button class="smallButton danger" data-remove-task="${index}">x</button>
    </div>`;
  }).join("") || `<div class="listItem"><div><strong>No tasks</strong><span>Add tasks from the selector above.</span></div></div>`;
  $("#excludedTaskList").innerHTML = group.excluded_tasks.map(taskId => {
    const info = taskInfo(taskId);
    return `<div class="listItem">
      <div><strong>${esc(taskLabel(taskId))}</strong><span>${esc(info.display_name)}</span></div>
      <button class="smallButton danger" data-remove-excluded="${esc(taskId)}">x</button>
    </div>`;
  }).join("") || `<div class="listItem"><div><strong>No exclusions</strong><span>All listed task occurrences are active.</span></div></div>`;
  $("#groupStats").textContent = `${group.tasks.length} listed task occurrences, ${group.effective_tasks.length} active after exclusions, ${group.duplicate_count || 0} duplicate occurrences`;
  $all("[data-remove-task]").forEach(button => button.addEventListener("click", () => removeGroupTask(Number(button.dataset.removeTask))));
  $all("[data-remove-excluded]").forEach(button => button.addEventListener("click", () => removeExcludedTask(button.dataset.removeExcluded)));
}
function setGroupsData(data, selected) {
  state.groups = data.groups || {};
  state.tasks = data.tasks || state.tasks || [];
  populateTaskSelect($("#groupTaskSelect"));
  populateTaskSelect($("#excludeTaskSelect"));
  refreshRunGroupSelect(selected);
  refreshEditGroupSelect(selected);
  selectGroupForEdit(state.activeGroup);
}
function newGroupDraft() {
  const base = "new_group";
  let name = base;
  let idx = 1;
  while (state.groups[name]) {
    idx += 1;
    name = `${base}_${idx}`;
  }
  state.groups[name] = { description: "", tasks: [], excluded_tasks: [], effective_tasks: [], duplicate_count: 0 };
  refreshRunGroupSelect(name);
  refreshEditGroupSelect(name);
  selectGroupForEdit(name);
}
function addGroupTask() {
  const taskId = $("#groupTaskSelect").value;
  if (!taskId || !state.activeGroup) return;
  state.groups[state.activeGroup].tasks.push(taskId);
  state.groups[state.activeGroup].effective_tasks = state.groups[state.activeGroup].tasks.filter(t => !state.groups[state.activeGroup].excluded_tasks.includes(t));
  state.groups[state.activeGroup].duplicate_count = state.groups[state.activeGroup].tasks.length - new Set(state.groups[state.activeGroup].tasks).size;
  renderGroupEditor();
}
function removeGroupTask(index) {
  const group = state.groups[state.activeGroup];
  if (!group) return;
  group.tasks.splice(index, 1);
  group.effective_tasks = group.tasks.filter(t => !group.excluded_tasks.includes(t));
  group.duplicate_count = group.tasks.length - new Set(group.tasks).size;
  renderGroupEditor();
}
function addExcludedTask() {
  const taskId = $("#excludeTaskSelect").value;
  const group = state.groups[state.activeGroup];
  if (!taskId || !group) return;
  if (!group.excluded_tasks.includes(taskId)) group.excluded_tasks.push(taskId);
  group.effective_tasks = group.tasks.filter(t => !group.excluded_tasks.includes(t));
  renderGroupEditor();
}
function removeExcludedTask(taskId) {
  const group = state.groups[state.activeGroup];
  if (!group) return;
  group.excluded_tasks = group.excluded_tasks.filter(t => t !== taskId);
  group.effective_tasks = group.tasks.filter(t => !group.excluded_tasks.includes(t));
  renderGroupEditor();
}
async function saveCurrentGroup() {
  const original = state.activeGroup;
  const draft = currentGroupDraft();
  const data = await api("/api/groups/save", { original_name: original, group: draft });
  setGroupsData(data, data.saved);
  $("#headerState").textContent = `saved group ${data.saved}`;
}
async function deleteCurrentGroup() {
  if (!state.activeGroup) return;
  const data = await api("/api/groups/delete", { name: state.activeGroup });
  const selected = Object.keys(data.groups || {})[0] || "";
  setGroupsData(data, selected);
  $("#headerState").textContent = `deleted group ${data.deleted}`;
}
async function loadOptions() {
  const data = await api("/api/options");
  state.defaults = data.defaults;
  const profile = $("#baselineProfile");
  data.profiles.forEach(p => profile.add(new Option(p, p)));
  setGroupsData(data, data.defaults.group_name);
  Object.entries(data.defaults).forEach(([k, v]) => setInput(k, v));
  refreshRunGroupSelect($("#groupName").value);
  await updateGeometry();
}
function toScreen(point, view, w, h) {
  let a, b, minA, maxA, minB, maxB;
  if (view === "top") { a = point[0]; b = point[1]; minA = -0.75; maxA = 0.95; minB = -0.62; maxB = 0.62; }
  else if (view === "front") { a = point[1]; b = point[2]; minA = -0.62; maxA = 0.62; minB = -0.28; maxB = 1.55; }
  else { a = point[0]; b = point[2]; minA = -0.75; maxA = 0.95; minB = -0.28; maxB = 1.55; }
  return [40 + (a - minA) / (maxA - minA) * (w - 80), h - 35 - (b - minB) / (maxB - minB) * (h - 70)];
}
function drawProxy(ctx, proxy, view, w, h) {
  ctx.strokeStyle = "#aab4bc"; ctx.fillStyle = "rgba(110,122,132,0.18)"; ctx.lineWidth = 1;
  if (proxy.type === "box") {
    const c = proxy.center, e = proxy.half_extents;
    let p1, p2;
    if (view === "top") { p1 = [c[0]-e[0], c[1]-e[1], c[2]]; p2 = [c[0]+e[0], c[1]+e[1], c[2]]; }
    else if (view === "front") { p1 = [c[0], c[1]-e[1], c[2]-e[2]]; p2 = [c[0], c[1]+e[1], c[2]+e[2]]; }
    else { p1 = [c[0]-e[0], c[1], c[2]-e[2]]; p2 = [c[0]+e[0], c[1], c[2]+e[2]]; }
    const a = toScreen(p1, view, w, h), b = toScreen(p2, view, w, h);
    ctx.fillRect(a[0], b[1], b[0]-a[0], a[1]-b[1]); ctx.strokeRect(a[0], b[1], b[0]-a[0], a[1]-b[1]);
  } else if (proxy.type === "sphere") {
    const p = toScreen(proxy.center, view, w, h);
    ctx.beginPath(); ctx.arc(p[0], p[1], proxy.radius * 150, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  } else {
    const p0 = toScreen(proxy.p0, view, w, h), p1 = toScreen(proxy.p1, view, w, h);
    ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.lineWidth = Math.max(4, proxy.radius * 170); ctx.strokeStyle = "rgba(110,122,132,0.28)"; ctx.stroke();
  }
}
function drawArm(ctx, points, view, w, h, color) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3; ctx.beginPath();
  points.forEach((pt, i) => { const p = toScreen(pt, view, w, h); if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); });
  ctx.stroke();
  points.forEach(pt => { const p = toScreen(pt, view, w, h); ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill(); });
}
function drawAxis(ctx, a, b, view, w, h, color) {
  const p0 = toScreen(a, view, w, h), p1 = toScreen(b, view, w, h);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
}
function drawCanvas(id, view) {
  const canvas = $(id), ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h); ctx.fillStyle = "#ffffff"; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = "#e5eaee"; ctx.lineWidth = 1;
  for (let x = 40; x < w - 30; x += 60) { ctx.beginPath(); ctx.moveTo(x,20); ctx.lineTo(x,h-25); ctx.stroke(); }
  for (let y = 30; y < h - 25; y += 60) { ctx.beginPath(); ctx.moveTo(30,y); ctx.lineTo(w-30,y); ctx.stroke(); }
  if (!state.geometry) return;
  state.geometry.body.forEach(p => drawProxy(ctx, p, view, w, h));
  drawArm(ctx, state.geometry.left_points, view, w, h, "#0b6f86");
  drawArm(ctx, state.geometry.right_points, view, w, h, "#b76019");
  drawAxis(ctx, state.geometry.left_mount, state.geometry.left_axis_end, view, w, h, "#084d5c");
  drawAxis(ctx, state.geometry.right_mount, state.geometry.right_axis_end, view, w, h, "#8a450f");
}
function renderMetrics() {
  const m = state.geometry?.mount_plausibility || {};
  const c = state.geometry?.candidate || {};
  const items = [
    ["left x", c.left_x_m], ["left y", c.left_y_m], ["left z", c.left_z_m],
    ["mount clearance", m.mount_min_body_clearance_m], ["drift", m.mount_shoulder_drift_m], ["penalty", m.mount_plausibility_penalty],
  ];
  $("#mountMetrics").innerHTML = items.map(([k,v]) => `<div class="metric"><span>${k}</span><strong>${Number(v).toFixed(4)}</strong></div>`).join("");
}
async function updateGeometry() {
  try {
    state.geometry = await api("/api/manual/geometry", { settings: collectSettings() });
    renderMetrics();
    drawCanvas("#topCanvas", "top"); drawCanvas("#frontCanvas", "front"); drawCanvas("#sideCanvas", "side");
  } catch (err) { $("#headerState").textContent = err.message; }
}
async function startJob(path) {
  setTab("results");
  const data = await api(path, { settings: collectSettings() });
  state.jobId = data.job_id;
  state.simulations = [];
  renderSimulationSelect();
  $("#jobStatus").textContent = "queued " + data.job_id;
  $("#jobLinks").innerHTML = "";
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(pollJob, 1500);
  await pollJob();
}
async function pollJob() {
  if (!state.jobId) return;
  const job = await api(`/api/jobs/${state.jobId}`);
  $("#headerState").textContent = `${job.kind}: ${job.status}`;
  $("#jobStatus").textContent = `${job.kind} ${job.status} ${job.elapsed_s.toFixed(1)}s\n\n${job.log || ""}`;
  const result = job.result || {};
  const links = [];
  if (result.report_url) links.push(`<a target="_blank" href="${result.report_url}">comparison/report</a>`);
  if (result.task_report_index_url) links.push(`<a target="_blank" href="${result.task_report_index_url}">task videos</a>`);
  if (result.manual_report_url) links.push(`<a target="_blank" href="${result.manual_report_url}">manual</a>`);
  if (result.baseline_report_url) links.push(`<a target="_blank" href="${result.baseline_report_url}">baseline</a>`);
  $("#jobLinks").innerHTML = links.join("");
  if (Array.isArray(result.simulations)) {
    state.simulations = result.simulations;
    renderSimulationSelect();
  }
  if (job.status === "complete" || job.status === "failed") clearInterval(state.poll);
}
function renderSimulationSelect() {
  const select = $("#simulationSelect");
  if (!select) return;
  select.innerHTML = "";
  state.simulations.forEach((sim, index) => {
    select.add(new Option(`${sim.configuration} | ${sim.task_name}`, String(index)));
  });
  if (!state.simulations.length) select.add(new Option("no simulation available", ""));
}
async function launchSimulation() {
  const index = Number($("#simulationSelect").value);
  const sim = state.simulations[index];
  if (!sim) throw new Error("No simulation selected.");
  const result = await api("/api/simulations/launch", {
    task_name: sim.task_name,
    config_path: sim.config_path,
    realtime_factor: Number($("#realtimeFactor").value) || 1.0,
    use_garmi_follow: $("#interactiveUseFollow").checked
  });
  const logLine = result.log_url ? `\nviewer log: ${result.log_url}` : "";
  $("#headerState").textContent = `viewer pid ${result.pid}`;
  $("#jobStatus").textContent = `${$("#jobStatus").textContent}\n\nstarted interactive viewer pid ${result.pid}${logLine}`;
}
let geometryTimer = null;
$all(".tabs button").forEach(b => b.addEventListener("click", () => setTab(b.dataset.tab)));
document.addEventListener("input", evt => {
  if (evt.target.matches("[data-key]")) {
    clearTimeout(geometryTimer);
    geometryTimer = setTimeout(updateGeometry, 250);
  }
});
$("#startOptimization").addEventListener("click", () => startJob("/api/optimize-mounts").catch(err => $("#jobStatus").textContent = err.message));
$("#evaluateManual").addEventListener("click", () => startJob("/api/manual/evaluate").catch(err => $("#jobStatus").textContent = err.message));
$("#launchSimulation").addEventListener("click", () => launchSimulation().catch(err => $("#jobStatus").textContent = `${$("#jobStatus").textContent}\n\n${err.message}`));
$("#editGroupSelect").addEventListener("change", evt => selectGroupForEdit(evt.target.value));
$("#newGroup").addEventListener("click", newGroupDraft);
$("#addGroupTask").addEventListener("click", addGroupTask);
$("#addExcludedTask").addEventListener("click", addExcludedTask);
$("#saveGroup").addEventListener("click", () => saveCurrentGroup().catch(err => $("#headerState").textContent = err.message));
$("#deleteGroup").addEventListener("click", () => deleteCurrentGroup().catch(err => $("#headerState").textContent = err.message));
loadOptions().catch(err => $("#headerState").textContent = err.message);
</script>
</body>
</html>
"""


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "GarmiMountOptGui/1.0"

    def _send(self, status: int, body: bytes, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(_jsonable(data), indent=2).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send(200, index_html().encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/options":
                self._send_json(200, {"profiles": _profiles(), "defaults": _default_settings(), **_group_editor_payload()})
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                with JOB_LOCK:
                    job = JOBS.get(job_id)
                    if job is None:
                        self._send_json(404, {"error": "unknown job"})
                    else:
                        self._send_json(200, job.as_dict())
                return
            if parsed.path == "/artifact":
                query = urllib.parse.parse_qs(parsed.query)
                raw_path = query.get("path", [""])[0]
                path = Path(raw_path).resolve()
                root = PATHS.root.resolve()
                if not path.is_relative_to(root) or not path.exists() or not path.is_file():
                    self._send_json(404, {"error": "artifact not found"})
                    return
                content_type = "text/html; charset=utf-8" if path.suffix == ".html" else "text/plain; charset=utf-8"
                if path.suffix == ".mp4":
                    content_type = "video/mp4"
                elif path.suffix == ".csv":
                    content_type = "text/csv; charset=utf-8"
                self._send(200, path.read_bytes(), content_type)
                return
            self._send_json(404, {"error": "not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "traceback": traceback.format_exc()})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            settings = payload.get("settings", {})
            if self.path == "/api/manual/geometry":
                self._send_json(200, manual_geometry(settings))
                return
            if self.path == "/api/optimize-mounts":
                job = _start_job("mount_optimization", settings, _run_mount_optimization_job)
                self._send_json(202, job.as_dict())
                return
            if self.path == "/api/manual/evaluate":
                job = _start_job("manual_mount", settings, _run_manual_evaluation_job)
                self._send_json(202, job.as_dict())
                return
            if self.path == "/api/simulations/launch":
                self._send_json(200, _launch_interactive_simulation(payload))
                return
            if self.path == "/api/groups/save":
                self._send_json(200, _save_group(payload))
                return
            if self.path == "/api/groups/delete":
                self._send_json(200, _delete_group(payload))
                return
            self._send_json(404, {"error": "not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "traceback": traceback.format_exc()})

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[gui] " + (format % args) + "\n")


def run_gui(host: str = "127.0.0.1", port: int = 8765) -> None:
    ensure_dirs()
    server = ThreadingHTTPServer((host, int(port)), GuiHandler)
    print(f"gui_url: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
