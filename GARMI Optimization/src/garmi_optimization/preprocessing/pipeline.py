"""Preprocessing of the Geriatronics Interaction Dataset into task data.

Every valid trial is converted into ``<output>/<trial_id>/task_data.npz``
(waypoints of both demonstrator wrists, their orientation axes and the
measured wrenches in the room frame) plus ``metadata.json`` and a small HTML
preview. The optimization only reads these processed files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from garmi_optimization.config import PATHS, ProjectPaths, load_project_config, timestamp_id, write_json
from garmi_optimization.dataset import build_trial_manifest, dataset_root, preprocessing_params
from garmi_optimization.math_utils import as_float_or_none, interp_columns, normalize_rows
from garmi_optimization.preprocessing.sensors import reconstruct_sensor
from garmi_optimization.preprocessing.vicon import (
    clean_sensor_nodes,
    clean_therapist_nodes,
    convert_positions_mm_to_m,
    derive_therapist_helper_nodes,
    fill_missing_data,
    node_xyz,
    read_ft_csv,
    read_vicon_csv,
    smooth_data,
)
from garmi_optimization.preprocessing.preview import write_task_preview


def _time_vector_from_lengths(node_count: int, sensor_t_rel: np.ndarray, node_rate_hz: float) -> np.ndarray:
    nominal = np.arange(node_count, dtype=float) / float(node_rate_hz)
    if len(sensor_t_rel) < 2 or node_count < 2:
        return nominal
    sensor_duration = float(sensor_t_rel[-1] - sensor_t_rel[0])
    if sensor_duration <= 0:
        return nominal
    nominal_duration = nominal[-1]
    if nominal_duration <= 0:
        return nominal
    return nominal * (sensor_duration / nominal_duration)


def _sensor_time(sensor_data: dict[str, Any], sensor_rate_hz: float) -> np.ndarray:
    rel = np.asarray(sensor_data["General Data"]["Relative_time"], dtype=float)
    if len(rel) > 1 and np.nanmax(rel) > np.nanmin(rel):
        return rel - rel[0]
    return np.arange(len(rel), dtype=float) / float(sensor_rate_hz)


def _slice_window(time: np.ndarray, start: float, end: float) -> np.ndarray:
    return (time >= start) & (time <= end)


def _longest_true_window(mask: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(mask) - 1):
            end = idx if value and idx == len(mask) - 1 else idx - 1
            if best is None or end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    return best


def _frame_time(node_data: dict[str, Any], frame_value: Any) -> float | None:
    value = as_float_or_none(frame_value)
    if value is None:
        return None
    frames = np.asarray(node_data["General Data"]["Frame"], dtype=float)
    if len(frames) == 0:
        return None
    node_time = np.asarray(node_data["real_time"], dtype=float)
    idx = int(np.nanargmin(np.abs(frames - float(value))))
    return float(node_time[idx])


def _configured_time_window(node_data: dict[str, Any], params: dict[str, Any]) -> tuple[float | None, float | None, str]:
    start_time = as_float_or_none(params.get("start_time"))
    end_time = as_float_or_none(params.get("end_time"))
    if start_time is not None and end_time is not None:
        return float(start_time), float(end_time), "yaml"

    start_frame_time = _frame_time(node_data, params.get("start_frame"))
    end_frame_time = _frame_time(node_data, params.get("end_frame"))
    if start_frame_time is not None and end_frame_time is not None:
        return start_frame_time, end_frame_time, "metadata_frames"
    return start_time, end_time, "inferred"


def _clean_available_sensor_nodes(node_data: dict[str, Any], params: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    available = []
    notes = []
    cleaned = node_data
    for sensor_name in ["sensor1", "sensor2"]:
        try:
            cleaned = clean_sensor_nodes(cleaned, sensor_name)
            available.append(sensor_name)
        except ValueError as exc:
            notes.append(str(exc))
            params[f"{sensor_name}_available"] = False
    return cleaned, available, notes


def _infer_sensor_side_from_proximity(
    node_data: dict[str, Any],
    sensor_name: str,
    start_time: float | None,
    end_time: float | None,
) -> tuple[str | None, dict[str, float]]:
    node_time = np.asarray(node_data["real_time"], dtype=float)
    if start_time is None or end_time is None or end_time <= start_time:
        mask = np.ones(len(node_time), dtype=bool)
    else:
        mask = _slice_window(node_time, start_time, end_time)
        if not np.any(mask):
            mask = np.ones(len(node_time), dtype=bool)
    center = node_xyz(node_data, f"{sensor_name}:center")
    distances: dict[str, float] = {}
    for side, prefix in [("left", "L"), ("right", "R")]:
        wrist = node_xyz(node_data, f"therapeut:{prefix}_WRIST")
        distance = np.linalg.norm(center[mask] - wrist[mask], axis=1)
        distances[side] = float(np.nanmedian(distance))
    if not distances or not np.isfinite(list(distances.values())).any():
        return None, distances
    return min(distances, key=distances.get), distances


def _complete_end_effector_map(
    node_data: dict[str, Any],
    sensors: dict[str, dict[str, np.ndarray]],
    params: dict[str, Any],
    start_time: float | None,
    end_time: float | None,
) -> list[str]:
    notes = []
    for sensor_name in sensors:
        side = str(params.get(f"{sensor_name}_end_effector", "")).strip().lower()
        if side in {"left", "right"}:
            params[f"{sensor_name}_end_effector"] = side
            continue
        inferred, distances = _infer_sensor_side_from_proximity(node_data, sensor_name, start_time, end_time)
        if inferred is None:
            notes.append(f"{sensor_name}: no end-effector side could be inferred")
            continue
        params[f"{sensor_name}_end_effector"] = inferred
        notes.append(
            f"{sensor_name}: inferred {inferred} end-effector by wrist proximity "
            f"(left={distances.get('left', float('nan')):.3f} m, right={distances.get('right', float('nan')):.3f} m)"
        )
    return notes


def _infer_touch_window(
    side_sensor: dict[str, dict[str, np.ndarray]],
    node_data: dict[str, Any],
    params: dict[str, Any],
    assumed_hand_length_m: float,
    force_threshold_n: float,
    fallback_force_quantile: float,
) -> tuple[float, float, str]:
    windows: list[tuple[float, float, str]] = []
    for sensor_name, sensor in side_sensor.items():
        side = str(params.get(f"{sensor_name}_end_effector", "")).strip().lower()
        if side not in {"left", "right"}:
            continue
        prefix = "L" if side == "left" else "R"
        wrist = node_xyz(node_data, f"therapeut:{prefix}_WRIST")
        sensor_center = node_xyz(node_data, f"{sensor_name}:center")
        node_time = np.asarray(node_data["real_time"], dtype=float)
        sensor_time = np.asarray(sensor["sensor_time"], dtype=float)
        wrist_sensor = interp_columns(node_time, wrist - sensor_center, sensor_time)
        distance = np.linalg.norm(wrist_sensor, axis=1)
        force = np.asarray(sensor["force_resultant"], dtype=float)
        touch_mask = (distance < assumed_hand_length_m) & (force > force_threshold_n)
        window = _longest_true_window(touch_mask)
        source = "distance_and_force"
        if window is None:
            threshold = max(force_threshold_n, float(np.nanquantile(force, fallback_force_quantile)))
            touch_mask = force > threshold
            window = _longest_true_window(touch_mask)
            source = "force_quantile"
        if window is not None:
            windows.append((float(sensor_time[window[0]]), float(sensor_time[window[1]]), f"{sensor_name}:{source}"))

    if not windows:
        node_time = np.asarray(node_data["real_time"], dtype=float)
        return float(node_time[0]), float(node_time[-1]), "full_recording"
    return min(w[0] for w in windows), max(w[1] for w in windows), ",".join(w[2] for w in windows)


def _build_side_arrays(
    node_data: dict[str, Any],
    sensors: dict[str, dict[str, np.ndarray]],
    params: dict[str, Any],
    task_time: np.ndarray,
) -> dict[str, np.ndarray]:
    node_time = np.asarray(node_data["real_time"], dtype=float)
    arrays: dict[str, np.ndarray] = {}
    for side in ["left", "right"]:
        prefix = "L" if side == "left" else "R"
        pos = node_xyz(node_data, f"therapeut:{prefix}_WRIST")
        hand = node_xyz(node_data, f"therapeut:{prefix}_HAND")
        wristup = node_xyz(node_data, f"therapeut:{prefix}_WRISTUP")
        wrist = node_xyz(node_data, f"therapeut:{prefix}_WRIST")
        arrays[f"{side}_pos"] = interp_columns(node_time, pos, task_time)
        arrays[f"{side}_orient_primary"] = normalize_rows(interp_columns(node_time, hand - wrist, task_time))
        arrays[f"{side}_orient_secondary"] = normalize_rows(interp_columns(node_time, wristup - wrist, task_time))
        arrays[f"{side}_force"] = np.zeros((len(task_time), 3), dtype=float)
        arrays[f"{side}_torque"] = np.zeros((len(task_time), 3), dtype=float)

    for sensor_name, sensor in sensors.items():
        side = str(params.get(f"{sensor_name}_end_effector", "")).strip().lower()
        if side not in {"left", "right"}:
            continue
        sensor_time = np.asarray(sensor["sensor_time"], dtype=float)
        arrays[f"{side}_force"] = interp_columns(sensor_time, sensor["force_global"], task_time)
        arrays[f"{side}_torque"] = interp_columns(sensor_time, sensor["torque_global"], task_time)
    return arrays


def _preprocess_from_files(
    trial_id: str,
    mocap_path: Path,
    ft_path: Path | None,
    params: dict[str, Any],
    paths: ProjectPaths,
    output_dir: Path,
    source_root: Path,
    n_points: int | None = None,
) -> Path:
    """Convert one MoCap (+ optional force-torque) recording into task data.

    Steps: read both CSVs, align the 120 Hz motion capture with the 240 Hz
    force-torque time base, fill gaps, apply a zero-phase low-pass filter,
    reconstruct the sensor frames from their marker triads (Kabsch), rotate
    the wrenches into the room frame, assign each sensor to a wrist, cut the
    annotated contact window and resample it to ``n_points`` waypoints.
    """
    project_cfg = load_project_config(paths)
    prep_cfg = project_cfg.get("preprocessing", {})
    node_rate, node_data = read_vicon_csv(mocap_path, int(prep_cfg.get("vicon_default_rate_hz", 120)))
    node_count = len(node_data["General Data"]["Frame"])
    # Force data is optional (kinematics-only trials).
    has_ft = ft_path is not None
    if has_ft:
        ft_rate, sensor_data = read_ft_csv(ft_path, int(prep_cfg.get("ft_default_rate_hz", 240)))
        sensor_time = _sensor_time(sensor_data, ft_rate)
        node_time = _time_vector_from_lengths(node_count, sensor_time, node_rate)
    else:
        ft_rate, sensor_data = 0, None
        node_time = np.arange(node_count, dtype=float) / float(node_rate)
        sensor_time = node_time
    node_data["real_time"] = node_time
    if sensor_data is not None:
        sensor_data["real_time"] = sensor_time

    notes: list[str] = []
    node_data = clean_therapist_nodes(node_data, str(params.get("therapeut_name", "therapeut")))
    if has_ft:
        node_data, available_sensors, sensor_notes = _clean_available_sensor_nodes(node_data, params)
        notes.extend(sensor_notes)
    else:
        available_sensors = []
        notes.append("kinematics-only trial: no force-torque recording; forces set to zero")
    node_data = fill_missing_data(node_data)
    if sensor_data is not None:
        sensor_data = fill_missing_data(sensor_data)
    cutoff = float(prep_cfg.get("lowpass_cutoff_hz", 15.0))
    order = int(prep_cfg.get("filter_order", 4))
    node_data = smooth_data(node_data, node_rate, cutoff, order)
    if sensor_data is not None:
        sensor_data = smooth_data(sensor_data, ft_rate, cutoff, order)
    node_data = convert_positions_mm_to_m(node_data)
    derive_therapist_helper_nodes(node_data)

    sensors: dict[str, dict[str, np.ndarray]] = {}
    for sensor_name in available_sensors:
        local_matrix = np.asarray(params[f"{sensor_name}_node_matrix"], dtype=float)
        reconstructed = reconstruct_sensor(node_data, sensor_data, sensor_name, local_matrix)
        reconstructed["sensor_time"] = sensor_time
        sensors[sensor_name] = reconstructed

    start_time, end_time, cut_source = _configured_time_window(node_data, params)
    notes.extend(_complete_end_effector_map(node_data, sensors, params, start_time, end_time))
    if start_time is None or end_time is None:
        start_time, end_time, cut_source = _infer_touch_window(
            sensors,
            node_data,
            params,
            float(prep_cfg.get("assumed_hand_length_m", 0.20)),
            float(prep_cfg.get("force_threshold_n", 1.0)),
            float(prep_cfg.get("fallback_force_quantile", 0.75)),
        )
        notes.append(f"time window inferred from {cut_source}")
    start_time = max(float(start_time), float(node_time[0]), float(sensor_time[0]))
    end_time = min(float(end_time), float(node_time[-1]), float(sensor_time[-1]))
    if end_time <= start_time:
        start_time, end_time, cut_source = float(node_time[0]), float(node_time[-1]), "full_recording_invalid_cut"
        notes.append("annotated time window was invalid; full recording used")

    if n_points is None:
        n_points = int(params.get("number_of_task_points", 50) or 50)
    task_time_abs = np.linspace(start_time, end_time, int(n_points))
    arrays = _build_side_arrays(node_data, sensors, params, task_time_abs)
    task_time = task_time_abs - task_time_abs[0]

    processed_dir = output_dir / trial_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_dir / "task_data.npz", time=task_time, time_abs=task_time_abs, **arrays)

    all_positions = np.vstack([arrays["left_pos"], arrays["right_pos"]])
    metadata = {
        "task_name": trial_id,
        "trial_id": trial_id,
        "category": params.get("category"),
        "category_name": params.get("category_name"),
        "description": params.get("task_description"),
        "original_trial_name": params.get("original_trial_name"),
        "split": params.get("split"),
        "repeat_group": params.get("repeat_group"),
        "force_measurement": bool(has_ft),
        "source_files": {
            "mocap": _relative_to(mocap_path, source_root),
            "force_torque": _relative_to(ft_path, source_root),
        },
        "node_rate_hz": node_rate,
        "ft_rate_hz": ft_rate if has_ft else None,
        "n_task_points": int(n_points),
        "time_window_s": {"start": start_time, "end": end_time, "duration": end_time - start_time, "source": cut_source},
        "metadata_frames": {
            "start_frame": params.get("start_frame"),
            "end_frame": params.get("end_frame"),
        },
        "sensor_positions": {
            "sensor1": params.get("sensor1_position"),
            "sensor2": params.get("sensor2_position"),
        },
        "available_sensors": available_sensors,
        "end_effector_map": {
            "sensor1": params.get("sensor1_end_effector"),
            "sensor2": params.get("sensor2_end_effector"),
        },
        "force_summary_n": {
            name: {
                "max": float(np.max(sensor["force_resultant"])),
                "mean": float(np.mean(sensor["force_resultant"])),
            }
            for name, sensor in sensors.items()
        },
        "workspace_bbox_m": {
            "min": np.min(all_positions, axis=0).round(5).tolist(),
            "max": np.max(all_positions, axis=0).round(5).tolist(),
        },
        "preprocessing_notes": [
            "Positions are converted from millimeters to meters.",
            "End-effector targets are the demonstrator's wrist trajectories.",
            "Forces and torques are rotated into the motion-capture (room) frame.",
            *notes,
        ],
    }
    write_json(processed_dir / "metadata.json", metadata)
    write_task_preview(processed_dir, metadata)
    return processed_dir


def _relative_to(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def preprocess_trial(
    record: dict[str, Any],
    paths: ProjectPaths = PATHS,
    output_dir: Path | None = None,
    n_points: int | None = None,
) -> Path:
    """Preprocess one trial of the manifest (see :func:`build_trial_manifest`)."""
    if record.get("status") != "valid":
        raise ValueError(f"Trial '{record.get('trial_id')}' is not processable: {record.get('skip_reason')}")
    root = dataset_root(paths)
    prep_cfg = load_project_config(paths).get("preprocessing", {})
    default_points = int(prep_cfg.get("default_number_of_task_points", 50) or 50)
    params = preprocessing_params(record, default_points)
    mocap_path = root / str(record["mocap_path"])
    ft_path = root / str(record["ft_path"]) if record.get("ft_path") else None
    return _preprocess_from_files(
        str(record["trial_id"]),
        mocap_path,
        ft_path,
        params,
        paths,
        output_dir or paths.processed_data,
        root,
        n_points=n_points,
    )


def preprocess_dataset(
    paths: ProjectPaths = PATHS,
    trials: list[str] | None = None,
    output_dir: Path | None = None,
    n_points: int | None = None,
    continue_on_error: bool = True,
) -> tuple[list[Path], Path]:
    """Preprocess all valid trials of the dataset (or the selected ``trials``).

    Writes ``manifest.json`` (all trials incl. skipped ones and the study
    assignment) and ``preprocessing_status.json`` into the output folder.
    """
    output_dir = output_dir or paths.processed_data
    manifest = build_trial_manifest(paths)
    known = {record["trial_id"] for record in manifest}
    if trials:
        unknown = sorted(set(trials) - known)
        if unknown:
            raise KeyError(f"Unknown trial(s): {', '.join(unknown)}")
    selected = [r for r in manifest if r["status"] == "valid" and (not trials or r["trial_id"] in set(trials))]
    if not selected:
        raise ValueError(f"No processable trials found in {dataset_root(paths)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", {"trials": manifest})
    outputs: list[Path] = []
    rows: list[dict[str, Any]] = []
    for record in selected:
        try:
            output = preprocess_trial(record, paths, output_dir=output_dir, n_points=n_points)
            outputs.append(output)
            rows.append({"trial_id": record["trial_id"], "status": "processed", "error": None})
        except Exception as exc:
            rows.append({"trial_id": record["trial_id"], "status": "failed", "error": str(exc)})
            if not continue_on_error:
                raise
    for record in manifest:
        if record["status"] != "valid":
            rows.append({"trial_id": record["trial_id"], "status": "skipped", "error": record.get("skip_reason")})
    status_path = output_dir / "preprocessing_status.json"
    write_json(status_path, {"created": timestamp_id("preprocessing"), "trials": rows})
    if not outputs:
        raise RuntimeError(f"No trials were preprocessed successfully. Status report: {status_path}")
    return outputs, status_path
