"""Task-frame constraint handling.

Each task can declare which orientation degrees of freedom actually matter for
task success. This prevents the evaluator from penalizing deviations along
task-irrelevant directions (e.g. rotation about the tool axis while wiping) and
enforces absolute gravity alignment where it is physically required (e.g.
carrying a tray).

Orientation modes:

- ``relative`` (default): track the recorded orientation change relative to the
  anchored start orientation.
- ``gravity_aligned``: like ``relative``, but the anchoring offset is restricted
  to a rotation about the world vertical axis, and the scored error additionally
  keeps the recorded tool-axis tilt against gravity. Yaw about the world
  vertical stays free.
- ``free``: orientation is not scored at all; IK runs position-only.

Additional flag:

- ``free_tool_axis``: rotation about the tool (EE z) axis is task-irrelevant.
  The scored orientation error reduces to the angle between the target and the
  actual tool axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from garmi_optimization.math_utils import rotation_error_vector, rotation_z

VALID_MODES = {"relative", "gravity_aligned", "free"}
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class TaskFrameSpec:
    orientation_mode: str = "relative"
    free_tool_axis: bool = False
    tilt_weight: float = 1.0
    base_motion_mode: str | None = None
    """Per-task base-mode override. Locomotion tasks (walking assistance,
    moving around the care recipient) are impossible with a fixed base;
    they may declare e.g. 'nonholonomic' or 'limited_holonomic' here while the
    profile default stays strict for stationary tasks."""

    @property
    def scores_orientation(self) -> bool:
        return self.orientation_mode != "free"


def spec_for_task(cfg: dict[str, Any], task_name: str) -> TaskFrameSpec:
    section = cfg.get("task_frames", {}) or {}
    defaults = section.get("defaults", {}) or {}
    per_task = (section.get("tasks", {}) or {}).get(task_name, {}) or {}
    merged = {**defaults, **per_task}
    mode = str(merged.get("orientation_mode", "relative")).strip().lower()
    if mode not in VALID_MODES:
        mode = "relative"
    base_mode = merged.get("base_motion_mode")
    return TaskFrameSpec(
        orientation_mode=mode,
        free_tool_axis=bool(merged.get("free_tool_axis", False)),
        tilt_weight=float(merged.get("tilt_weight", 1.0)),
        base_motion_mode=str(base_mode).strip().lower() if base_mode else None,
    )


def _tool_axis(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=float)[:, 2]


def _axis_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def yaw_only_rotation(rotation: np.ndarray) -> np.ndarray:
    """Project a rotation matrix onto a pure rotation about the world z axis."""
    rotation = np.asarray(rotation, dtype=float)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return rotation_z(yaw)


def anchor_offset(spec: TaskFrameSpec, raw_rotation_0: np.ndarray, ee_rotation_0: np.ndarray) -> np.ndarray:
    """Offset applied to recorded rotations so targets start at the IK-found EE pose.

    For gravity-aligned tasks, the offset is restricted to a rotation about the
    world vertical so the recorded tilt against gravity is preserved in the
    target trajectory.
    """
    offset = np.asarray(raw_rotation_0, dtype=float).T @ np.asarray(ee_rotation_0, dtype=float)
    if spec.orientation_mode == "gravity_aligned":
        world_offset = np.asarray(raw_rotation_0, dtype=float) @ offset @ np.asarray(raw_rotation_0, dtype=float).T
        yaw_world = yaw_only_rotation(world_offset)
        return np.asarray(raw_rotation_0, dtype=float).T @ yaw_world @ np.asarray(raw_rotation_0, dtype=float)
    return offset


def orientation_error(spec: TaskFrameSpec, target_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    """Scored orientation error [rad] respecting task-frame constraints."""
    if spec.orientation_mode == "free":
        return 0.0
    target_rotation = np.asarray(target_rotation, dtype=float)
    current_rotation = np.asarray(current_rotation, dtype=float)
    if spec.orientation_mode == "gravity_aligned":
        # The task-relevant quantity is the tool-axis tilt against gravity plus
        # (unless the tool axis is free) the shape mismatch of the remaining
        # rotation. Yaw about world z is never penalized.
        tilt_target = _axis_angle(_tool_axis(target_rotation), WORLD_UP)
        tilt_current = _axis_angle(_tool_axis(current_rotation), WORLD_UP)
        tilt_error = abs(tilt_target - tilt_current)
        if spec.free_tool_axis:
            return spec.tilt_weight * tilt_error
        axis_error = _axis_angle(_tool_axis(target_rotation), _tool_axis(current_rotation))
        yaw_free_axis_error = max(0.0, axis_error - tilt_error)
        return spec.tilt_weight * tilt_error + 0.5 * yaw_free_axis_error
    if spec.free_tool_axis:
        return _axis_angle(_tool_axis(target_rotation), _tool_axis(current_rotation))
    return float(np.linalg.norm(rotation_error_vector(target_rotation, current_rotation)))
