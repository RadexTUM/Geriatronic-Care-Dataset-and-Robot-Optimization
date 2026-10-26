"""Access to the Geriatronics Interaction Dataset.

The dataset is organized by functional task category::

    Geriatronics Interaction Dataset/
        Trials_Description.xlsx              one sheet per category
        <Category>/MoCap/<trial>.csv         Vicon Nexus export (120 Hz)
        <Category>/FT-sensor/<trial>_ft.csv  force-torque recording (240 Hz)
        Objects/                             photos of the tracked objects

``Trials_Description.xlsx`` lists every trial with its description, the frame
window of the task-relevant contact phase, the placement of the two
force-torque sensors and the manipulated objects. This module turns that
description into one :class:`TrialRecord` per trial and adds the study
assignment (optimization split, held-out split and repeat group) that is
defined in ``config/task_groups.yaml``.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from garmi_optimization.config import PATHS, ProjectPaths, load_group_config, load_project_config, write_json

DATASET_FOLDER_NAME = "Geriatronics Interaction Dataset"
DESCRIPTION_FILE = "Trials_Description.xlsx"
MOCAP_FOLDER = "MoCap"
FT_FOLDER = "FT-sensor"

# Local marker coordinates [mm] of the three struts (x+, x-, y-) of the
# 3D-printed force-torque sensor housing, expressed in the sensor frame.
DEFAULT_SENSOR_NODE_MATRIX_MM = [
    [57.0, 0.0, -50.0],
    [-60.1, 0.0, -14.5],
    [0.0, -49.9, -14.5],
]


@dataclass(frozen=True)
class Category:
    slug: str
    folder: str


CATEGORIES: tuple[Category, ...] = (
    Category("basic_care", "Basic Care"),
    Category("meals", "Meals"),
    Category("transfer", "Transfer"),
    Category("rom", "ROM"),
    Category("geriatric_rehabilitation", "Geriatric Rehabilitation"),
)
CATEGORY_BY_SLUG = {category.slug: category for category in CATEGORIES}

# Column names of Trials_Description.xlsx (matched case-insensitively).
COLUMN_TRIAL = "trial name"
COLUMN_ORIGINAL_TRIAL = "trial name (old)"
COLUMN_FORCE = "trial with force measurement"
COLUMN_DESCRIPTION = "description"
COLUMN_START = "timeframe begin contact"
COLUMN_END = "timeframe end contact"
COLUMN_SENSOR1 = "position sensor 1"
COLUMN_SENSOR2 = "position sensor 2"


@dataclass(frozen=True)
class TrialRecord:
    """One trial of the dataset as described in ``Trials_Description.xlsx``."""

    trial_id: str
    category: str
    category_name: str
    description: str | None
    original_trial_name: str | None
    force_measurement: bool
    mocap_path: str | None
    ft_path: str | None
    start_frame: int | None
    end_frame: int | None
    sensor1_position: str | None
    sensor2_position: str | None
    objects: tuple[str, ...]
    status: str
    skip_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["objects"] = list(self.objects)
        return data


# ---------------------------------------------------------------------------
# Locations


def dataset_root(paths: ProjectPaths = PATHS) -> Path:
    """Folder of the Geriatronics Interaction Dataset.

    Resolution order: the environment variable ``GARMI_OPT_DATASET_DIR``,
    ``config/project.yaml -> dataset.root`` (relative paths resolve against
    the ``GARMI Optimization`` folder) and finally the dataset folder next to
    the ``GARMI Optimization`` folder in this repository.
    """
    env_root = os.environ.get("GARMI_OPT_DATASET_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    configured = (load_project_config(paths).get("dataset", {}) or {}).get("root")
    if configured:
        candidate = Path(str(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = paths.root / candidate
        return candidate.resolve()
    return (paths.root.parent / DATASET_FOLDER_NAME).resolve()


def _find_child(folder: Path, name: str) -> Path | None:
    """Child folder ``name`` of ``folder``, matched case-insensitively."""
    exact = folder / name
    if exact.is_dir():
        return exact
    if not folder.is_dir():
        return None
    for child in folder.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def _find_file(folder: Path | None, filename: str) -> Path | None:
    """File ``filename`` in ``folder``, matched case-insensitively."""
    if folder is None or not folder.is_dir():
        return None
    exact = folder / filename
    if exact.is_file():
        return exact
    for child in folder.iterdir():
        if child.is_file() and child.name.lower() == filename.lower():
            return child
    return None


def _relative(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Parsing helpers


def _norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _norm_key(value: Any) -> str:
    return re.sub(r"\s+", " ", _norm_text(value).lower()).strip()


def _first_number(value: Any) -> int | None:
    """First integer in a cell, e.g. ``"371 (contact drawer), 547"`` -> 371."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    return int(float(match.group(0).replace(",", ".")))


def _is_marked(value: Any) -> bool:
    return _norm_text(value).lower() in {"x", "yes", "true", "1", "y"}


def _object_columns(columns: dict[str, str]) -> list[str]:
    return [columns[key] for key in sorted(columns) if re.fullmatch(r"included object \d+", key)]


def read_trials_description(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Rows of ``Trials_Description.xlsx`` per category sheet (normalized keys)."""
    path = root / DESCRIPTION_FILE
    if not path.exists():
        raise FileNotFoundError(f"{DESCRIPTION_FILE} not found in {root}")
    sheets = pd.read_excel(path, sheet_name=None)
    rows_by_sheet: dict[str, list[dict[str, Any]]] = {}
    for sheet_name, frame in sheets.items():
        columns = {_norm_key(column): column for column in frame.columns}
        if COLUMN_TRIAL not in columns:
            continue
        object_columns = _object_columns(columns)
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            trial = _norm_text(row.get(columns[COLUMN_TRIAL]))
            if not trial:
                continue
            record = {key: row.get(column) for key, column in columns.items()}
            record["_objects"] = [
                _norm_text(row.get(column)) for column in object_columns if _norm_text(row.get(column))
            ]
            rows.append(record)
        rows_by_sheet[str(sheet_name)] = rows
    return rows_by_sheet


def infer_explicit_end_effector(sensor_position: str | None) -> str | None:
    """Robot arm side from a sensor-position text such as ``"right hand"``.

    Returns ``None`` when the text names no side of the demonstrator (e.g. a
    sensor on an object or on the care recipient); the preprocessing then
    assigns the sensor to the closer wrist.
    """
    text = _norm_text(sensor_position).lower()
    if not text:
        return None
    text = text.replace("rigth", "right")
    right_tokens = ("right hand", "right wrist", "therapist right", "therapeut right")
    left_tokens = ("left hand", "left wrist", "therapist left", "therapeut left")
    if any(token in text for token in right_tokens):
        return "right"
    if any(token in text for token in left_tokens):
        return "left"
    return None


# ---------------------------------------------------------------------------
# Discovery


def discover_trials(paths: ProjectPaths = PATHS) -> list[TrialRecord]:
    """All trials listed in ``Trials_Description.xlsx`` with their files.

    A trial is ``valid`` if its MoCap file exists and a contact frame window
    is annotated; force-torque data is optional (kinematics-only trials).
    """
    root = dataset_root(paths)
    rows_by_sheet = read_trials_description(root)
    records: list[TrialRecord] = []
    seen: set[str] = set()
    for category in CATEGORIES:
        rows = rows_by_sheet.get(category.folder)
        if rows is None:
            continue
        category_dir = root / category.folder
        mocap_dir = _find_child(category_dir, MOCAP_FOLDER)
        ft_dir = _find_child(category_dir, FT_FOLDER)
        for row in rows:
            trial_id = _norm_text(row.get(COLUMN_TRIAL))
            if trial_id in seen:
                raise ValueError(f"Trial name '{trial_id}' appears more than once in {DESCRIPTION_FILE}")
            seen.add(trial_id)
            mocap_path = _find_file(mocap_dir, f"{trial_id}.csv")
            ft_path = _find_file(ft_dir, f"{trial_id}_ft.csv")
            start_frame = _first_number(row.get(COLUMN_START))
            end_frame = _first_number(row.get(COLUMN_END))
            skip_reason = None
            if mocap_path is None:
                skip_reason = "missing_mocap_csv"
            elif start_frame is None or end_frame is None:
                skip_reason = "missing_frame_window"
            # Force data is used whenever the recording exists; the sheet
            # column documents which trials were recorded with the sensors.
            force_measurement = ft_path is not None
            records.append(
                TrialRecord(
                    trial_id=trial_id,
                    category=category.slug,
                    category_name=category.folder,
                    description=_norm_text(row.get(COLUMN_DESCRIPTION)) or None,
                    original_trial_name=_norm_text(row.get(COLUMN_ORIGINAL_TRIAL)) or None,
                    force_measurement=force_measurement,
                    mocap_path=_relative(mocap_path, root),
                    ft_path=_relative(ft_path, root),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    sensor1_position=_norm_text(row.get(COLUMN_SENSOR1)) or None,
                    sensor2_position=_norm_text(row.get(COLUMN_SENSOR2)) or None,
                    objects=tuple(row.get("_objects", [])),
                    status="valid" if skip_reason is None else "skipped",
                    skip_reason=skip_reason,
                )
            )
    return records


def trial_by_id(trial_id: str, paths: ProjectPaths = PATHS) -> TrialRecord | None:
    for record in discover_trials(paths):
        if record.trial_id == trial_id:
            return record
    return None


def trial_files(record: TrialRecord, paths: ProjectPaths = PATHS) -> tuple[Path, Path | None]:
    """Absolute MoCap and force-torque paths of a trial (FT may be ``None``)."""
    root = dataset_root(paths)
    if record.mocap_path is None:
        raise FileNotFoundError(f"Trial '{record.trial_id}' has no MoCap file ({record.skip_reason})")
    mocap = root / record.mocap_path
    ft = root / record.ft_path if record.ft_path else None
    return mocap, ft


# ---------------------------------------------------------------------------
# Study assignment (paper)


def study_assignment(paths: ProjectPaths = PATHS) -> dict[str, dict[str, Any]]:
    """Split and repeat group of every trial used in the paper's study.

    ``config/task_groups.yaml`` defines the optimization split (group
    ``all_tasks``), the held-out split (group ``heldout_all``) and the repeat
    groups. Trials that are not part of the study get ``split = None``.
    """
    config = load_group_config(paths)
    groups = config.get("groups", {}) or {}
    optimization = set((groups.get("all_tasks") or {}).get("tasks", []) or [])
    heldout = set((groups.get("heldout_all") or {}).get("tasks", []) or [])
    repeat_of: dict[str, str] = {}
    for group_name, members in (config.get("repeat_groups", {}) or {}).items():
        for trial in members or []:
            repeat_of[str(trial)] = str(group_name)
    assignment: dict[str, dict[str, Any]] = {}
    for trial in sorted(optimization | heldout | set(repeat_of)):
        split = "optimization" if trial in optimization else "heldout" if trial in heldout else None
        assignment[trial] = {"split": split, "repeat_group": repeat_of.get(trial, trial)}
    return assignment


def build_trial_manifest(paths: ProjectPaths = PATHS) -> list[dict[str, Any]]:
    """Trial records enriched with the study assignment."""
    study = study_assignment(paths)
    manifest = []
    for record in discover_trials(paths):
        item = record.to_dict()
        info = study.get(record.trial_id, {})
        item["split"] = info.get("split")
        item["repeat_group"] = info.get("repeat_group", record.trial_id)
        manifest.append(item)
    return manifest


def write_trial_manifest(output: Path, paths: ProjectPaths = PATHS) -> Path:
    """Write the trial manifest (file paths relative to the dataset folder)."""
    write_json(output, {"trials": build_trial_manifest(paths)})
    return output


def preprocessing_params(record: dict[str, Any], default_task_points: int = 50) -> dict[str, Any]:
    """Parameters of the preprocessing pipeline for one manifest entry."""
    return {
        "task_name": record["trial_id"],
        "trial_name": record["trial_id"],
        "task_description": record.get("description"),
        "original_trial_name": record.get("original_trial_name"),
        "category": record.get("category"),
        "category_name": record.get("category_name"),
        "split": record.get("split"),
        "repeat_group": record.get("repeat_group"),
        "therapeut_name": "therapeut",
        "start_frame": record.get("start_frame"),
        "end_frame": record.get("end_frame"),
        "sensor1_position": record.get("sensor1_position"),
        "sensor2_position": record.get("sensor2_position"),
        "sensor1_end_effector": infer_explicit_end_effector(record.get("sensor1_position")),
        "sensor2_end_effector": infer_explicit_end_effector(record.get("sensor2_position")),
        "sensor1_node_matrix": DEFAULT_SENSOR_NODE_MATRIX_MM,
        "sensor2_node_matrix": DEFAULT_SENSOR_NODE_MATRIX_MM,
        "number_of_task_points": default_task_points,
    }
