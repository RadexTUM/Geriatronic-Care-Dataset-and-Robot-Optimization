"""Robustness perturbations for recorded task clouds.

A candidate mount that only works for the exact recorded scene is not a useful
recommendation. Each task can be evaluated on ``ensemble`` perturbed copies in
addition to the nominal recording: the whole task point cloud is translated in
the ground plane, rotated about the world vertical at its centroid, and scaled
about the centroid. Perturbation seeds are derived deterministically from the
robustness seed and the variant index, so every candidate sees exactly the same
perturbed scenes (common random numbers).

Perturbed variants are addressed with a task-name suffix ``<task>@p<k>``.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from garmi_optimization.math_utils import rotation_z
from garmi_optimization.tasks import TaskData

VARIANT_SEPARATOR = "@p"


def robustness_config(cfg: dict[str, Any]) -> dict[str, float]:
    section = cfg.get("robustness", {}) or {}
    return {
        "ensemble": int(section.get("ensemble", 0)),
        "translation_m": float(section.get("translation_m", 0.10)),
        "yaw_rad": float(section.get("yaw_rad", 0.17)),
        "scale": float(section.get("scale", 0.05)),
        "seed": int(section.get("seed", 20260705)),
    }


def split_variant_name(name: str) -> tuple[str, int | None]:
    """Return (base_task_name, variant_index) for names like 'task@p3'."""
    if VARIANT_SEPARATOR not in name:
        return name, None
    base, _, suffix = name.rpartition(VARIANT_SEPARATOR)
    try:
        return base, int(suffix)
    except ValueError:
        return name, None


def expand_task_names(task_names: list[str], ensemble: int) -> list[str]:
    if ensemble <= 0:
        return list(task_names)
    expanded: list[str] = []
    for name in task_names:
        expanded.append(name)
        expanded.extend(f"{name}{VARIANT_SEPARATOR}{k}" for k in range(1, ensemble + 1))
    return expanded


def _task_centroid(task: TaskData) -> np.ndarray:
    points = []
    for side in task.active_sides():
        points.append(task.side_position(side))
    stacked = np.vstack(points) if points else np.vstack([task.left_pos, task.right_pos])
    return np.asarray(np.mean(stacked, axis=0), dtype=float)


def _stable_name_hash(name: str) -> int:
    """Process-independent hash of a trial name.

    Python's built-in ``hash`` of a string is salted per interpreter process,
    so worker processes would draw different perturbations for the same
    trial. CRC-32 gives every process (and every run) the same value.
    """
    return zlib.crc32(name.encode("utf-8")) % 100003


def perturb_task(task: TaskData, variant: int, robustness: dict[str, float]) -> TaskData:
    """Deterministically perturbed copy of a task (common random numbers)."""
    rng = np.random.default_rng(int(robustness["seed"]) + 7919 * int(variant) + _stable_name_hash(task.name))
    dxy = rng.uniform(-robustness["translation_m"], robustness["translation_m"], size=2)
    yaw = float(rng.uniform(-robustness["yaw_rad"], robustness["yaw_rad"]))
    scale = float(1.0 + rng.uniform(-robustness["scale"], robustness["scale"]))
    rot = rotation_z(yaw)
    centroid = _task_centroid(task)
    offset = np.array([dxy[0], dxy[1], 0.0], dtype=float)

    def _points(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return (points - centroid) @ rot.T * scale + centroid + offset

    def _frames(frames: np.ndarray) -> np.ndarray:
        return np.einsum("ij,njk->nik", rot, np.asarray(frames, dtype=float))

    def _vectors(vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=float) @ rot.T

    metadata = dict(task.metadata)
    metadata["perturbation"] = {
        "variant": int(variant),
        "translation_m": [float(dxy[0]), float(dxy[1])],
        "yaw_rad": yaw,
        "scale": scale,
    }
    return TaskData(
        name=f"{task.name}{VARIANT_SEPARATOR}{variant}",
        time=task.time,
        time_abs=task.time_abs,
        left_pos=_points(task.left_pos),
        left_rot=_frames(task.left_rot),
        right_pos=_points(task.right_pos),
        right_rot=_frames(task.right_rot),
        left_force=_vectors(task.left_force),
        right_force=_vectors(task.right_force),
        left_torque=_vectors(task.left_torque),
        right_torque=_vectors(task.right_torque),
        metadata=metadata,
    )


def robustness_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Nominal vs perturbed coverage statistics from per-task result rows."""
    nominal: dict[str, bool] = {}
    perturbed: dict[str, list[bool]] = {}
    for row in rows:
        base, variant = split_variant_name(str(row.get("task_name", "")))
        success = bool(row.get("success", False))
        if variant is None:
            nominal[base] = success
        else:
            perturbed.setdefault(base, []).append(success)
    out: dict[str, float] = {}
    if nominal:
        out["success_fraction_nominal"] = float(np.mean([1.0 if v else 0.0 for v in nominal.values()]))
    if perturbed:
        per_task_rates = [float(np.mean([1.0 if v else 0.0 for v in flags])) for flags in perturbed.values()]
        out["success_fraction_perturbed_mean"] = float(np.mean(per_task_rates))
        out["success_fraction_perturbed_min"] = float(np.min(per_task_rates))
        out["robustness_gap"] = float(out.get("success_fraction_nominal", 0.0) - out["success_fraction_perturbed_mean"])
    return out
