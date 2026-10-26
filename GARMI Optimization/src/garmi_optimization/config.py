"""Project paths and YAML/JSON helpers.

All configuration files, resources, processed data and outputs are resolved
relative to the ``GARMI Optimization`` folder (two levels above this file in
the ``src`` layout). Install the package in editable mode
(``pip install -e .``) or set ``GARMI_OPT_ROOT`` to point at that folder.
The raw dataset location is resolved in :mod:`garmi_optimization.dataset`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(os.environ.get("GARMI_OPT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
# Profile used when a command is called without --baseline-profile: the
# original GARMI mounting evaluated with the measured wrenches (paper baseline).
DEFAULT_PROFILE = "stage2_forces"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def processed_data(self) -> Path:
        # GARMI_OPT_PROCESSED_DIR points evaluations at an alternative
        # processed-data folder (e.g. the 100/200-waypoint variants of the
        # discretization analysis). Relative paths resolve against the root.
        override = os.environ.get("GARMI_OPT_PROCESSED_DIR")
        if override:
            path = Path(override)
            return path if path.is_absolute() else self.root / path
        return self.root / "data" / "processed"

    @property
    def runs(self) -> Path:
        return self.root / "outputs" / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "outputs" / "reports"

    @property
    def resources(self) -> Path:
        return self.root / "resources"


PATHS = ProjectPaths()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)


def timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_dirs(paths: ProjectPaths = PATHS) -> None:
    for path in [paths.runs, paths.reports]:
        path.mkdir(parents=True, exist_ok=True)


def load_project_config(paths: ProjectPaths = PATHS) -> dict[str, Any]:
    return load_yaml(paths.config / "project.yaml")


def load_group_config(paths: ProjectPaths = PATHS) -> dict[str, Any]:
    return load_yaml(paths.config / "task_groups.yaml")


def baseline_config_path(profile: str = DEFAULT_PROFILE, paths: ProjectPaths = PATHS) -> Path:
    """Path of an evaluation profile: a name in ``config/`` or a YAML file path."""
    path = Path(profile)
    if path.is_absolute():
        return path
    if path.suffix in {".yaml", ".yml"} or len(path.parts) > 1:
        return (paths.root / path).resolve()
    return paths.config / f"{profile}.yaml"


def load_task_frames_config(paths: ProjectPaths = PATHS) -> dict[str, Any]:
    path = paths.config / "task_frames.yaml"
    if not path.exists():
        return {}
    return load_yaml(path).get("task_frames", {}) or {}


def _relative_to_root(path: Path, paths: ProjectPaths) -> str:
    """Path relative to the project root where possible (portable run records)."""
    try:
        return str(path.resolve().relative_to(paths.root.resolve()))
    except ValueError:
        return str(path)


def load_baseline_config(paths: ProjectPaths = PATHS, profile: str = DEFAULT_PROFILE) -> dict[str, Any]:
    profile_path = baseline_config_path(profile, paths)
    cfg = load_yaml(profile_path)
    label = _relative_to_root(Path(profile), paths) if Path(profile).is_absolute() else profile
    cfg.setdefault("_profile", label)
    cfg.setdefault("_profile_path", _relative_to_root(profile_path, paths))
    # Task-frame constraints live in config/task_frames.yaml; a profile can
    # override them by defining its own task_frames section.
    if "task_frames" not in cfg:
        frames = load_task_frames_config(paths)
        if frames:
            cfg["task_frames"] = frames
    return cfg
