"""Loading of processed trials and task groups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from garmi_optimization.config import PATHS, ProjectPaths, load_group_config, read_json
from garmi_optimization.math_utils import frames_from_primary_secondary


@dataclass(frozen=True)
class TaskData:
    name: str
    time: np.ndarray
    time_abs: np.ndarray
    left_pos: np.ndarray
    left_rot: np.ndarray
    right_pos: np.ndarray
    right_rot: np.ndarray
    left_force: np.ndarray
    right_force: np.ndarray
    left_torque: np.ndarray
    right_torque: np.ndarray
    metadata: dict

    @property
    def sample_count(self) -> int:
        return int(len(self.time))

    @property
    def duration_s(self) -> float:
        if len(self.time) < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    def side_position(self, side: str) -> np.ndarray:
        return self.left_pos if side == "left" else self.right_pos

    def side_rotation(self, side: str) -> np.ndarray:
        return self.left_rot if side == "left" else self.right_rot

    def side_force(self, side: str) -> np.ndarray:
        return self.left_force if side == "left" else self.right_force

    def side_torque(self, side: str) -> np.ndarray:
        return self.left_torque if side == "left" else self.right_torque

    def active_sides(self) -> tuple[str, ...]:
        explicit = self.metadata.get("active_robot_sides") or self.metadata.get("active_sides")
        sides: list[str] = []
        if explicit:
            values = explicit if isinstance(explicit, (list, tuple)) else [explicit]
            for value in values:
                side = str(value).strip().lower()
                if side in {"left", "right"} and side not in sides:
                    sides.append(side)
        if not sides:
            mapping = self.metadata.get("end_effector_map", {})
            if isinstance(mapping, dict):
                for value in mapping.values():
                    side = str(value).strip().lower() if value is not None else ""
                    if side in {"left", "right"} and side not in sides:
                        sides.append(side)
        return tuple(sides or ("left", "right"))

    def is_side_active(self, side: str) -> bool:
        return side in self.active_sides()


def _load_side_rotations(data: np.lib.npyio.NpzFile, side: str, sample_count: int) -> np.ndarray:
    rot_key = f"{side}_rot"
    if rot_key in data:
        return np.asarray(data[rot_key], dtype=float)
    primary_key = f"{side}_orient_primary"
    secondary_key = f"{side}_orient_secondary"
    if primary_key in data and secondary_key in data:
        return frames_from_primary_secondary(data[primary_key], data[secondary_key])
    return np.repeat(np.eye(3)[None, :, :], sample_count, axis=0)


def load_task(task_name: str, paths: ProjectPaths = PATHS) -> TaskData:
    task_dir = paths.processed_data / task_name
    npz_path = task_dir / "task_data.npz"
    metadata_path = task_dir / "metadata.json"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Processed trial '{task_name}' not found in {task_dir}. Run "
            "'garmi-opt preprocess' to generate it from the dataset (or set "
            "GARMI_OPT_PROCESSED_DIR to another processed-data folder)."
        )
    data = np.load(npz_path)
    time = data["time"]
    return TaskData(
        name=task_name,
        time=time,
        time_abs=data["time_abs"],
        left_pos=data["left_pos"],
        left_rot=_load_side_rotations(data, "left", len(time)),
        right_pos=data["right_pos"],
        right_rot=_load_side_rotations(data, "right", len(time)),
        left_force=data["left_force"],
        right_force=data["right_force"],
        left_torque=data["left_torque"],
        right_torque=data["right_torque"],
        metadata=read_json(metadata_path),
    )


def load_tasks(task_names: list[str], paths: ProjectPaths = PATHS) -> list[TaskData]:
    return [load_task(name, paths) for name in task_names]


def configured_task_names(paths: ProjectPaths = PATHS) -> list[str]:
    """Trials available in the processed-data folder."""
    root = paths.processed_data
    if not root.exists():
        return []
    return sorted(path.parent.name for path in root.glob("*/task_data.npz"))


def group_task_names(group_name: str, paths: ProjectPaths = PATHS) -> list[str]:
    groups = load_group_config(paths).get("groups", {})
    if group_name not in groups:
        raise KeyError(f"Unknown group '{group_name}'. Available: {', '.join(sorted(groups))}")
    group = groups[group_name] or {}
    excluded = {str(task_name) for task_name in group.get("excluded_tasks", [])}
    return [str(task_name) for task_name in group.get("tasks", []) if str(task_name) not in excluded]


def group_task_weights(group_name: str, paths: ProjectPaths = PATHS) -> dict[str, float]:
    """Explicit task weights for a group. Defaults to 1.0 per task.

    Configured in task_groups.yaml as:

        groups:
          my_group:
            tasks: [...]
            weights:
              some_task: 2.5
    """
    groups = load_group_config(paths).get("groups", {})
    group = groups.get(group_name) or {}
    raw = group.get("weights", {}) or {}
    names = group_task_names(group_name, paths) if group_name in groups else []
    weights = {name: 1.0 for name in names}
    for key, value in raw.items():
        try:
            weights[str(key)] = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return weights


def global_task_names(paths: ProjectPaths = PATHS) -> list[str]:
    groups = load_group_config(paths).get("groups", {})
    if "all_tasks" in groups:
        return group_task_names("all_tasks", paths)
    return configured_task_names(paths)
