"""Search over the mirrored shoulder-mount pose and the base-placement policy
(paper Sec. IV-D).

Candidates are generated with Sobol/LHS sampling and Gaussian-process
Bayesian optimization inside a multi-fidelity scheme (quick, reduced and
full fidelity); the original GARMI mounting is evaluated as a protected
candidate in every run. Surrogate-assisted and genetic schedulers are
available as alternatives but were not used in the paper.
"""
from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from garmi_optimization.evaluation import _aggregate, _profile_slug, evaluate_task_baseline, evaluate_tasks_parallel
from garmi_optimization.collision import BoxProxy, CapsuleProxy, SphereProxy, _point_segment_distance, _point_box_clearance, garmi_body_proxies
from garmi_optimization.config import DEFAULT_PROFILE, PATHS, ProjectPaths, ensure_dirs, load_baseline_config, read_json, timestamp_id, write_json, write_yaml
from garmi_optimization.math_utils import rotation_from_rpy
from garmi_optimization.perturbations import (
    expand_task_names,
    perturb_task,
    robustness_config,
    split_variant_name,
)
from garmi_optimization.tasks import TaskData, group_task_weights, load_task, load_tasks


MIRROR_Y = np.diag([1.0, -1.0, 1.0])
# Summed constraint violation below this value does not gate candidate
# eligibility (it still contributes to scores). Units are mixed (meters for
# clearances, ratios for torque/speed); 0.01 tolerates ~1 cm or ~1 % overrun.
CONSTRAINT_EPSILON = 0.01
# Summed violation above this cap disqualifies a candidate regardless of
# coverage (deep penetration or gross torque/speed overruns); violations
# between epsilon and the cap only act as ranking tie-breakers.
HARD_CONSTRAINT_CAP = 0.5
STRICT_LEFT_MOUNT_ROTATION = rotation_from_rpy(-np.pi / 2.0, 0.0, -np.pi / 2.0)
PANDA_LOCAL_FORWARD_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class MountCandidate:
    candidate_id: str
    left_xyz_m: np.ndarray
    left_delta_rpy_rad: np.ndarray

    def as_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "left_x_m": float(self.left_xyz_m[0]),
            "left_y_m": float(self.left_xyz_m[1]),
            "left_z_m": float(self.left_xyz_m[2]),
            "left_delta_roll_rad": float(self.left_delta_rpy_rad[0]),
            "left_delta_pitch_rad": float(self.left_delta_rpy_rad[1]),
            "left_delta_yaw_rad": float(self.left_delta_rpy_rad[2]),
        }


@dataclass(frozen=True)
class DesignCandidate:
    candidate_id: str
    mount: MountCandidate
    base_policy: dict[str, float] = field(default_factory=dict)
    ik: dict[str, float] = field(default_factory=dict)
    collision_ik: dict[str, float] = field(default_factory=dict)
    controller: dict[str, float] = field(default_factory=dict)
    source: str = "sampled"

    def as_record(self) -> dict[str, Any]:
        record = self.mount.as_record()
        record["candidate_id"] = self.candidate_id
        record["candidate_source"] = self.source
        for prefix, values in [
            ("base", self.base_policy),
            ("ik", self.ik),
            ("collision", self.collision_ik),
            ("controller", self.controller),
        ]:
            for key, value in values.items():
                record[f"{prefix}_{key}"] = float(value)
        return record


CandidateLike = MountCandidate | DesignCandidate


def _right_from_left(left_xyz: np.ndarray, left_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right_xyz = np.asarray(left_xyz, dtype=float).copy()
    right_xyz[1] *= -1.0
    right_rotation = MIRROR_Y @ np.asarray(left_rotation, dtype=float) @ MIRROR_Y
    return right_xyz, right_rotation


def _rpy_from_rotation(rotation: np.ndarray) -> list[float]:
    return [float(value) for value in Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_euler("xyz")]


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source = source / max(float(np.linalg.norm(source)), 1e-12)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        axis = np.cross(source, np.array([0.0, 0.0, 1.0], dtype=float))
        if float(np.linalg.norm(axis)) < 1e-9:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = np.cross(source, target)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return Rotation.from_rotvec(float(np.arccos(dot)) * axis).as_matrix()


def _mount_rotation_mode(cfg: dict[str, Any] | None) -> str:
    mode = str((cfg or {}).get("optimization", {}).get("mount_rotation_mode", "rpy_delta")).lower()
    if mode in {"forward_outer_quarter_sphere", "quarter_sphere", "forward_outer"}:
        return "forward_outer_quarter_sphere"
    return "rpy_delta"


def _quarter_sphere_parameter_bounds(cfg: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    eps = 1e-4
    defaults = {
        "left_delta_roll_rad": (-0.65, 0.65),
        "left_delta_pitch_rad": (-0.5 * np.pi + eps, 0.5 * np.pi - eps),
        "left_delta_yaw_rad": (0.0, 0.5 * np.pi),
    }
    raw = (cfg or {}).get("optimization", {}).get("mount_rotation_quarter_sphere", {})
    mapping = {
        "left_delta_roll_rad": "roll_rad",
        "left_delta_pitch_rad": "elevation_rad",
        "left_delta_yaw_rad": "azimuth_rad",
    }
    out: dict[str, tuple[float, float]] = {}
    for target_key, source_key in mapping.items():
        low, high = defaults[target_key]
        value = raw.get(source_key, [low, high]) if isinstance(raw, dict) else [low, high]
        if isinstance(value, dict):
            value_low = float(value.get("min", low))
            value_high = float(value.get("max", high))
        else:
            value_low = float(value[0]) if len(value) > 0 else float(low)
            value_high = float(value[1]) if len(value) > 1 else float(high)
        if target_key == "left_delta_pitch_rad":
            value_low = float(np.clip(value_low, -0.5 * np.pi + eps, 0.5 * np.pi - eps))
            value_high = float(np.clip(value_high, -0.5 * np.pi + eps, 0.5 * np.pi - eps))
        if target_key == "left_delta_yaw_rad":
            value_low = float(np.clip(value_low, 0.0, 0.5 * np.pi))
            value_high = float(np.clip(value_high, 0.0, 0.5 * np.pi))
        if value_low > value_high:
            value_low, value_high = value_high, value_low
        out[target_key] = (value_low, value_high)
    return out


def _left_rotation_from_quarter_sphere(roll_rad: float, elevation_rad: float, azimuth_rad: float) -> np.ndarray:
    azimuth_rad = float(np.clip(azimuth_rad, 0.0, 0.5 * np.pi))
    elevation_rad = float(np.clip(elevation_rad, -0.5 * np.pi + 1e-4, 0.5 * np.pi - 1e-4))
    target_forward = np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ],
        dtype=float,
    )
    target_forward = target_forward / max(float(np.linalg.norm(target_forward)), 1e-12)
    strict_forward = STRICT_LEFT_MOUNT_ROTATION @ PANDA_LOCAL_FORWARD_AXIS
    align = _rotation_between_vectors(strict_forward, target_forward)
    roll_world = Rotation.from_rotvec(float(roll_rad) * target_forward).as_matrix()
    return roll_world @ align @ STRICT_LEFT_MOUNT_ROTATION


def _quarter_sphere_delta_rpy(roll_rad: float, elevation_rad: float, azimuth_rad: float) -> np.ndarray:
    left_rotation = _left_rotation_from_quarter_sphere(roll_rad, elevation_rad, azimuth_rad)
    delta = STRICT_LEFT_MOUNT_ROTATION.T @ left_rotation
    return np.asarray(_rpy_from_rotation(delta), dtype=float)


def _quarter_sphere_parameters_from_delta(delta_rpy: np.ndarray) -> np.ndarray:
    left_rotation = STRICT_LEFT_MOUNT_ROTATION @ rotation_from_rpy(
        float(delta_rpy[0]),
        float(delta_rpy[1]),
        float(delta_rpy[2]),
    )
    target_forward = left_rotation @ PANDA_LOCAL_FORWARD_AXIS
    target_forward = target_forward / max(float(np.linalg.norm(target_forward)), 1e-12)
    azimuth = float(np.clip(np.arctan2(target_forward[1], target_forward[0]), 0.0, 0.5 * np.pi))
    elevation = float(np.arcsin(np.clip(target_forward[2], -1.0 + 1e-9, 1.0 - 1e-9)))
    zero_roll_rotation = _left_rotation_from_quarter_sphere(0.0, elevation, azimuth)
    roll_vec = Rotation.from_matrix(left_rotation @ zero_roll_rotation.T).as_rotvec()
    roll = float(np.dot(roll_vec, target_forward))
    return np.array([roll, elevation, azimuth], dtype=float)


def _mount_search_values(candidate: MountCandidate, cfg: dict[str, Any] | None = None) -> dict[str, float]:
    if _mount_rotation_mode(cfg) == "forward_outer_quarter_sphere":
        roll, elevation, azimuth = _quarter_sphere_parameters_from_delta(candidate.left_delta_rpy_rad)
    else:
        roll, elevation, azimuth = candidate.left_delta_rpy_rad
    return {
        "left_x_m": float(candidate.left_xyz_m[0]),
        "left_y_m": float(candidate.left_xyz_m[1]),
        "left_z_m": float(candidate.left_xyz_m[2]),
        "left_delta_roll_rad": float(roll),
        "left_delta_pitch_rad": float(elevation),
        "left_delta_yaw_rad": float(azimuth),
    }


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
    raise ValueError(f"Unknown base search mode '{mode}'. Use full, reduced, or single.")


def apply_mount_candidate(
    cfg: dict[str, Any],
    candidate: MountCandidate,
    base_search: str = "full",
    serial_evaluation: bool = False,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    panda = out["panda"]
    left_rotation = STRICT_LEFT_MOUNT_ROTATION @ rotation_from_rpy(
        float(candidate.left_delta_rpy_rad[0]),
        float(candidate.left_delta_rpy_rad[1]),
        float(candidate.left_delta_rpy_rad[2]),
    )
    right_xyz, right_rotation = _right_from_left(candidate.left_xyz_m, left_rotation)
    panda["mounts"] = {
        "left": {"xyz_m": [float(v) for v in candidate.left_xyz_m], "rpy_rad": _rpy_from_rotation(left_rotation)},
        "right": {"xyz_m": [float(v) for v in right_xyz], "rpy_rad": _rpy_from_rotation(right_rotation)},
    }
    panda["shoulder_x_m"] = float(candidate.left_xyz_m[0])
    panda["shoulder_width_m"] = float(2.0 * abs(candidate.left_xyz_m[1]))
    panda["shoulder_height_m"] = float(candidate.left_xyz_m[2])
    out["baseline"] = copy.deepcopy(out.get("baseline", {}))
    out["baseline"]["name"] = f"{out['baseline'].get('name', 'candidate')}_{candidate.candidate_id}"
    out["performance"] = copy.deepcopy(out.get("performance", {}))
    if serial_evaluation:
        out["performance"]["parallel_workers"] = "off"
        out["performance"]["progress"] = False
    _set_base_search_mode(out, base_search)
    out["_candidate_id"] = candidate.candidate_id
    return out


def _mount_candidate(candidate: CandidateLike) -> MountCandidate:
    return candidate.mount if isinstance(candidate, DesignCandidate) else candidate


def _candidate_record(candidate: CandidateLike, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    record = candidate.as_record() if isinstance(candidate, DesignCandidate) else candidate.as_record()
    if _mount_rotation_mode(cfg) == "forward_outer_quarter_sphere":
        search_values = _mount_search_values(_mount_candidate(candidate), cfg)
        record.update(
            {
                "left_search_roll_rad": search_values["left_delta_roll_rad"],
                "left_search_elevation_rad": search_values["left_delta_pitch_rad"],
                "left_search_azimuth_rad": search_values["left_delta_yaw_rad"],
            }
        )
    return record


def _as_float_array(values: Any, fallback: list[float]) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return np.asarray(fallback, dtype=float)
    if arr.shape != (len(fallback),):
        return np.asarray(fallback, dtype=float)
    return arr


def _scale_list(values: Any, multiplier: float, fallback: list[float]) -> list[float]:
    return [float(value) for value in (_as_float_array(values, fallback) * float(multiplier))]


def _base_policy_grid(value: float, spread: float, base_search: str, lo: float | None = None, hi: float | None = None) -> list[float]:
    if base_search == "single" or spread <= 0.0:
        values = [float(value)]
    else:
        values = [float(value) - spread, float(value), float(value) + spread]
    if lo is not None:
        values = [max(lo, v) for v in values]
    if hi is not None:
        values = [min(hi, v) for v in values]
    out: list[float] = []
    for v in values:
        if not any(abs(v - existing) < 1e-9 for existing in out):
            out.append(v)
    return out


def _apply_design_overlay(cfg: dict[str, Any], candidate: DesignCandidate, base_search: str = "full") -> None:
    if candidate.base_policy:
        # The candidate's base policy defines the center of the same per-task
        # base-placement search that the strict baseline receives, not a single
        # fixed placement. Per-task base placement is behavior, not hardware,
        # so every mount gets the same freedom.
        base = cfg.setdefault("base", {})
        policy = candidate.base_policy
        if "preferred_reach_fraction" in policy:
            base["preferred_reach_fraction_candidates"] = _base_policy_grid(
                policy["preferred_reach_fraction"], 0.10 if base_search != "single" else 0.0, base_search, lo=0.50, hi=0.90
            )
        wide = base_search == "full"
        if "forward_offset_m" in policy:
            base["forward_offset_candidates_m"] = _base_policy_grid(policy["forward_offset_m"], 0.11 if wide else 0.0, base_search)
        if "lateral_offset_m" in policy:
            base["lateral_offset_candidates_m"] = _base_policy_grid(policy["lateral_offset_m"], 0.20 if wide else 0.0, base_search)
        if "yaw_offset_rad" in policy:
            base["yaw_offset_candidates_rad"] = [float(policy["yaw_offset_rad"])]
        cfg.setdefault("optimization", {})["optimized_base_policy"] = {key: float(value) for key, value in candidate.base_policy.items()}

    if candidate.ik:
        ik_cfg = cfg.setdefault("ik", {})
        for key, value in candidate.ik.items():
            ik_cfg[key] = float(value)
        cfg.setdefault("optimization", {})["optimized_ik"] = {key: float(value) for key, value in candidate.ik.items()}

    if candidate.collision_ik:
        collision_cfg = cfg.setdefault("collision", {})
        collision_cfg["collision_aware_ik"] = True
        for key, value in candidate.collision_ik.items():
            collision_cfg[key] = float(value)
        cfg.setdefault("optimization", {})["optimized_collision_ik"] = {key: float(value) for key, value in candidate.collision_ik.items()}

    if candidate.controller:
        controller_cfg = cfg.setdefault("controller", {})
        default_pos_kp = [1200.0, 1200.0, 1200.0]
        default_pos_kd = [110.0, 110.0, 110.0]
        default_rot_kp = [62.0, 62.0, 62.0]
        default_rot_kd = [7.5, 7.5, 7.5]
        default_null_kp = [20.0, 18.0, 15.0, 12.0, 7.0, 5.0, 3.0]
        default_null_kd = [3.0, 2.6, 2.2, 1.8, 1.0, 0.7, 0.45]
        if "position_kp_multiplier" in candidate.controller:
            controller_cfg["position_kp_npm"] = _scale_list(
                controller_cfg.get("position_kp_npm", default_pos_kp),
                candidate.controller["position_kp_multiplier"],
                default_pos_kp,
            )
        if "position_kd_multiplier" in candidate.controller:
            controller_cfg["position_kd_nspm"] = _scale_list(
                controller_cfg.get("position_kd_nspm", default_pos_kd),
                candidate.controller["position_kd_multiplier"],
                default_pos_kd,
            )
        if "orientation_kp_multiplier" in candidate.controller:
            controller_cfg["orientation_kp_nmprad"] = _scale_list(
                controller_cfg.get("orientation_kp_nmprad", default_rot_kp),
                candidate.controller["orientation_kp_multiplier"],
                default_rot_kp,
            )
        if "orientation_kd_multiplier" in candidate.controller:
            controller_cfg["orientation_kd_nmsprad"] = _scale_list(
                controller_cfg.get("orientation_kd_nmsprad", default_rot_kd),
                candidate.controller["orientation_kd_multiplier"],
                default_rot_kd,
            )
        if "nullspace_kp_multiplier" in candidate.controller:
            controller_cfg["nullspace_kp_nmprad"] = _scale_list(
                controller_cfg.get("nullspace_kp_nmprad", default_null_kp),
                candidate.controller["nullspace_kp_multiplier"],
                default_null_kp,
            )
        if "nullspace_kd_multiplier" in candidate.controller:
            controller_cfg["nullspace_kd_nmsprad"] = _scale_list(
                controller_cfg.get("nullspace_kd_nmsprad", default_null_kd),
                candidate.controller["nullspace_kd_multiplier"],
                default_null_kd,
            )
        for key in ["nullspace_damping", "max_task_force_n", "max_task_torque_nm"]:
            if key in candidate.controller:
                controller_cfg[key] = float(candidate.controller[key])
        base_pd = dict(controller_cfg.get("base_pd", {}))
        base_pd.setdefault("kp", [2400.0, 2400.0, 850.0])
        base_pd.setdefault("kd", [520.0, 520.0, 160.0])
        if "base_kp_multiplier" in candidate.controller:
            base_pd["kp"] = _scale_list(base_pd.get("kp"), candidate.controller["base_kp_multiplier"], [2400.0, 2400.0, 850.0])
        if "base_kd_multiplier" in candidate.controller:
            base_pd["kd"] = _scale_list(base_pd.get("kd"), candidate.controller["base_kd_multiplier"], [520.0, 520.0, 160.0])
        controller_cfg["base_pd"] = base_pd
        cfg.setdefault("optimization", {})["optimized_controller"] = {key: float(value) for key, value in candidate.controller.items()}


def apply_design_candidate(
    cfg: dict[str, Any],
    candidate: DesignCandidate,
    base_search: str = "full",
    serial_evaluation: bool = False,
) -> dict[str, Any]:
    out = apply_mount_candidate(cfg, candidate.mount, base_search=base_search, serial_evaluation=serial_evaluation)
    out["_candidate_id"] = candidate.candidate_id
    base_name = str(cfg.get("baseline", {}).get("name", "candidate"))
    out.setdefault("baseline", {})["name"] = f"{base_name}_{candidate.candidate_id}"
    _apply_design_overlay(out, candidate, base_search=base_search)
    return out


def _apply_candidate_config(
    cfg: dict[str, Any],
    candidate: CandidateLike,
    base_search: str = "full",
    serial_evaluation: bool = False,
) -> dict[str, Any]:
    if isinstance(candidate, DesignCandidate):
        return apply_design_candidate(cfg, candidate, base_search=base_search, serial_evaluation=serial_evaluation)
    return apply_mount_candidate(cfg, candidate, base_search=base_search, serial_evaluation=serial_evaluation)


def _candidate_bounds(cfg: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    defaults = {
        "left_x_m": (-0.27, -0.02),
        "left_y_m": (0.18, 0.46),
        "left_z_m": (0.78, 1.16),
        "left_delta_roll_rad": (-0.45, 0.45),
        "left_delta_pitch_rad": (-0.40, 0.40),
        "left_delta_yaw_rad": (-0.50, 0.50),
    }
    if _mount_rotation_mode(cfg) == "forward_outer_quarter_sphere":
        defaults.update(_quarter_sphere_parameter_bounds(cfg))
    overrides = (cfg or {}).get("optimization", {}).get("mount_bounds", {})
    bounds: dict[str, tuple[float, float]] = {}
    for name, (low, high) in defaults.items():
        raw = overrides.get(name, [low, high])
        if isinstance(raw, dict):
            value_low = float(raw.get("min", low))
            value_high = float(raw.get("max", high))
        else:
            value_low = float(raw[0]) if len(raw) > 0 else float(low)
            value_high = float(raw[1]) if len(raw) > 1 else float(high)
        if value_low > value_high:
            value_low, value_high = value_high, value_low
        bounds[name] = (value_low, value_high)
    return bounds


def _clamp_candidate(values: dict[str, float], candidate_id: str, cfg: dict[str, Any] | None = None) -> MountCandidate:
    bounds = _candidate_bounds(cfg)
    clamped = {name: float(np.clip(value, low, high)) for name, value in values.items() for low, high in [bounds[name]]}
    if _mount_rotation_mode(cfg) == "forward_outer_quarter_sphere":
        delta_rpy = _quarter_sphere_delta_rpy(
            clamped["left_delta_roll_rad"],
            clamped["left_delta_pitch_rad"],
            clamped["left_delta_yaw_rad"],
        )
    else:
        delta_rpy = np.array(
            [clamped["left_delta_roll_rad"], clamped["left_delta_pitch_rad"], clamped["left_delta_yaw_rad"]],
            dtype=float,
        )
    return MountCandidate(
        candidate_id=candidate_id,
        left_xyz_m=np.array([clamped["left_x_m"], clamped["left_y_m"], clamped["left_z_m"]], dtype=float),
        left_delta_rpy_rad=delta_rpy,
    )


def _unit_samples(count: int, dim: int, seed: int, strategy: str) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dim), dtype=float)
    strategy = str(strategy).lower()
    if strategy == "sobol":
        sampler = qmc.Sobol(d=dim, scramble=True, seed=int(seed))
        return np.asarray(sampler.random(count), dtype=float)
    if strategy in {"lhs", "latin_hypercube", "hybrid", "adaptive"}:
        sampler = qmc.LatinHypercube(d=dim, seed=int(seed))
        return np.asarray(sampler.random(count), dtype=float)
    rng = np.random.default_rng(int(seed))
    return rng.random((count, dim))


def _bounds_vector(bounds: dict[str, tuple[float, float]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(bounds)
    lower = np.asarray([bounds[name][0] for name in names], dtype=float)
    upper = np.asarray([bounds[name][1] for name in names], dtype=float)
    return names, lower, upper


def generate_mount_candidates(cfg: dict[str, Any], samples: int, seed: int, strategy: str = "hybrid") -> list[MountCandidate]:
    rng = np.random.default_rng(int(seed))
    bounds = _candidate_bounds(cfg)
    names, lower, upper = _bounds_vector(bounds)
    left_mount = np.asarray(cfg["panda"]["mounts"]["left"]["xyz_m"], dtype=float)
    base_values = {
        "left_x_m": float(left_mount[0]),
        "left_y_m": float(abs(left_mount[1])),
        "left_z_m": float(left_mount[2]),
        "left_delta_roll_rad": 0.0,
        "left_delta_pitch_rad": 0.0,
        "left_delta_yaw_rad": 0.0,
    }
    candidates = [_clamp_candidate(base_values, "candidate_000_strict_garmi", cfg)]
    if samples <= 1:
        return candidates[:samples]
    strategy = str(strategy).lower()

    default_local_scales = {
        "left_x_m": 0.055,
        "left_y_m": 0.045,
        "left_z_m": 0.055,
        "left_delta_roll_rad": 0.16,
        "left_delta_pitch_rad": 0.14,
        "left_delta_yaw_rad": 0.18,
    }
    local_scale_overrides = cfg.get("optimization", {}).get("mount_local_scales", {})
    local_scales = {
        name: float(local_scale_overrides.get(name, default))
        for name, default in default_local_scales.items()
    }
    local_fraction = float(cfg.get("optimization", {}).get("mount_local_sample_fraction", 1.0 / 3.0))
    local_count = 0 if strategy in {"random", "sobol", "lhs", "latin_hypercube"} else max(4, int(np.ceil(int(samples) * np.clip(local_fraction, 0.0, 1.0))))
    local_count = min(max(0, local_count), max(0, int(samples) - 1))
    global_count = max(0, int(samples) - 1 - local_count)
    for idx in range(1, local_count + 1):
        values = {
            name: float(base_values[name] + rng.normal(0.0, local_scales[name]))
            for name in bounds
        }
        candidates.append(_clamp_candidate(values, f"candidate_{idx:03d}", cfg))
    if global_count:
        unit = _unit_samples(global_count, len(names), int(seed) + 1009, strategy)
        scaled = lower[None, :] + unit * (upper - lower)[None, :]
        for row_idx, row in enumerate(scaled, start=len(candidates)):
            values = {name: float(value) for name, value in zip(names, row)}
            candidates.append(_clamp_candidate(values, f"candidate_{row_idx:03d}", cfg))
    return candidates[: int(samples)]


def generate_refined_mount_candidates(
    cfg: dict[str, Any],
    ranked_df: pd.DataFrame,
    candidate_by_id: dict[str, MountCandidate],
    count: int,
    seed: int,
    start_index: int,
) -> list[MountCandidate]:
    if count <= 0 or ranked_df.empty:
        return []
    rng = np.random.default_rng(int(seed))
    top_ids = [
        str(row["candidate_id"])
        for _, row in ranked_df.iterrows()
        if str(row.get("candidate_id", "")) in candidate_by_id
    ]
    if not top_ids:
        return []
    top_ids = top_ids[: max(1, min(4, len(top_ids)))]
    default_scales = {
        "left_x_m": 0.030,
        "left_y_m": 0.025,
        "left_z_m": 0.030,
        "left_delta_roll_rad": 0.090,
        "left_delta_pitch_rad": 0.080,
        "left_delta_yaw_rad": 0.100,
    }
    overrides = cfg.get("optimization", {}).get("mount_refine_scales", {})
    scales = {name: float(overrides.get(name, default)) for name, default in default_scales.items()}
    candidates: list[MountCandidate] = []
    for idx in range(int(count)):
        center = candidate_by_id[top_ids[idx % len(top_ids)]]
        center_values = _mount_search_values(center, cfg)
        values = {
            "left_x_m": float(center_values["left_x_m"] + rng.normal(0.0, scales["left_x_m"])),
            "left_y_m": float(center_values["left_y_m"] + rng.normal(0.0, scales["left_y_m"])),
            "left_z_m": float(center_values["left_z_m"] + rng.normal(0.0, scales["left_z_m"])),
            "left_delta_roll_rad": float(center_values["left_delta_roll_rad"] + rng.normal(0.0, scales["left_delta_roll_rad"])),
            "left_delta_pitch_rad": float(center_values["left_delta_pitch_rad"] + rng.normal(0.0, scales["left_delta_pitch_rad"])),
            "left_delta_yaw_rad": float(center_values["left_delta_yaw_rad"] + rng.normal(0.0, scales["left_delta_yaw_rad"])),
        }
        candidates.append(_clamp_candidate(values, f"candidate_{start_index + idx:03d}_refined", cfg))
    return candidates


def _bounds_from_config(section: dict[str, Any], defaults: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    raw_bounds = section.get("bounds", {}) if isinstance(section, dict) else {}
    out: dict[str, tuple[float, float]] = {}
    for key, (low, high) in defaults.items():
        raw = raw_bounds.get(key, [low, high]) if isinstance(raw_bounds, dict) else [low, high]
        if isinstance(raw, dict):
            value_low = float(raw.get("min", low))
            value_high = float(raw.get("max", high))
        else:
            value_low = float(raw[0]) if len(raw) > 0 else float(low)
            value_high = float(raw[1]) if len(raw) > 1 else float(high)
        if value_low > value_high:
            value_low, value_high = value_high, value_low
        out[key] = (value_low, value_high)
    return out


def _design_variable_sections(cfg: dict[str, Any], optimize_controller: bool) -> dict[str, dict[str, Any]]:
    design = cfg.get("optimization", {}).get("design_variables", {})
    sections = {
        "base_policy": {
            "enabled": True,
            "bounds": {
                "preferred_reach_fraction": [0.58, 0.86],
                "forward_offset_m": [-0.16, 0.16],
                "lateral_offset_m": [-0.24, 0.24],
                "yaw_offset_rad": [-0.35, 0.35],
            },
        },
        "ik": {
            "enabled": bool(optimize_controller),
            "bounds": {
                "damping": [0.015, 0.085],
                "step_limit_rad": [0.08, 0.28],
                "orientation_weight": [0.01, 0.10],
                "joint_center_weight": [0.0, 0.04],
                "continuity_weight": [0.0, 0.05],
            },
        },
        "collision_ik": {
            "enabled": bool(optimize_controller),
            "bounds": {
                "avoidance_start_m": [0.05, 0.14],
                "avoidance_target_m": [0.03, 0.10],
                "avoidance_step_rad": [0.02, 0.09],
                "avoidance_nullspace_damping": [0.02, 0.09],
            },
        },
        "controller": {
            "enabled": bool(optimize_controller),
            "bounds": {
                "position_kp_multiplier": [0.55, 1.65],
                "position_kd_multiplier": [0.55, 1.80],
                "orientation_kp_multiplier": [0.55, 1.80],
                "orientation_kd_multiplier": [0.55, 1.90],
                "nullspace_kp_multiplier": [0.35, 1.80],
                "nullspace_kd_multiplier": [0.35, 2.00],
                "nullspace_damping": [0.015, 0.08],
                "max_task_force_n": [180.0, 340.0],
                "max_task_torque_nm": [35.0, 75.0],
                "base_kp_multiplier": [0.55, 1.75],
                "base_kd_multiplier": [0.55, 1.85],
            },
        },
    }
    for name, defaults in list(sections.items()):
        override = design.get(name, {}) if isinstance(design, dict) else {}
        merged = copy.deepcopy(defaults)
        if isinstance(override, dict):
            if "enabled" in override:
                merged["enabled"] = bool(override["enabled"])
            if "bounds" in override and isinstance(override["bounds"], dict):
                merged["bounds"].update(override["bounds"])
        if name in {"ik", "collision_ik", "controller"} and optimize_controller:
            merged["enabled"] = True
        sections[name] = merged
    return sections


def _design_bounds(cfg: dict[str, Any], optimize_controller: bool) -> list[tuple[str, str, float, float]]:
    sections = _design_variable_sections(cfg, optimize_controller)
    specs: list[tuple[str, str, float, float]] = []
    specs.extend(("mount", key, low, high) for key, (low, high) in _candidate_bounds(cfg).items())
    defaults_by_section = {
        "base_policy": {
            "preferred_reach_fraction": (0.58, 0.86),
            "forward_offset_m": (-0.16, 0.16),
            "lateral_offset_m": (-0.24, 0.24),
            "yaw_offset_rad": (-0.35, 0.35),
        },
        "ik": {
            "damping": (0.015, 0.085),
            "step_limit_rad": (0.08, 0.28),
            "orientation_weight": (0.01, 0.10),
            "joint_center_weight": (0.0, 0.04),
            "continuity_weight": (0.0, 0.05),
        },
        "collision_ik": {
            "avoidance_start_m": (0.05, 0.14),
            "avoidance_target_m": (0.03, 0.10),
            "avoidance_step_rad": (0.02, 0.09),
            "avoidance_nullspace_damping": (0.02, 0.09),
        },
        "controller": {
            "position_kp_multiplier": (0.55, 1.65),
            "position_kd_multiplier": (0.55, 1.80),
            "orientation_kp_multiplier": (0.55, 1.80),
            "orientation_kd_multiplier": (0.55, 1.90),
            "nullspace_kp_multiplier": (0.35, 1.80),
            "nullspace_kd_multiplier": (0.35, 2.00),
            "nullspace_damping": (0.015, 0.08),
            "max_task_force_n": (180.0, 340.0),
            "max_task_torque_nm": (35.0, 75.0),
            "base_kp_multiplier": (0.55, 1.75),
            "base_kd_multiplier": (0.55, 1.85),
        },
    }
    for section_name, defaults in defaults_by_section.items():
        section = sections[section_name]
        if not bool(section.get("enabled", False)):
            continue
        for key, (low, high) in _bounds_from_config(section, defaults).items():
            specs.append((section_name, key, low, high))
    return specs


def _baseline_design_candidate(cfg: dict[str, Any]) -> DesignCandidate:
    mount = generate_mount_candidates(cfg, samples=1, seed=0, strategy="lhs")[0]
    return DesignCandidate("candidate_000_strict_garmi", mount, source="strict_garmi")


def _design_candidate_from_values(
    cfg: dict[str, Any],
    candidate_id: str,
    specs: list[tuple[str, str, float, float]],
    values: dict[tuple[str, str], float],
    source: str,
) -> DesignCandidate:
    mount_values = {
        key: float(values[("mount", key)])
        for section, key, _low, _high in specs
        if section == "mount"
    }
    mount = _clamp_candidate(mount_values, candidate_id, cfg)
    payload: dict[str, dict[str, float]] = {
        "base_policy": {},
        "ik": {},
        "collision_ik": {},
        "controller": {},
    }
    for section, key, _low, _high in specs:
        if section == "mount":
            continue
        payload[section][key] = float(values[(section, key)])
    return DesignCandidate(
        candidate_id=candidate_id,
        mount=mount,
        base_policy=payload["base_policy"],
        ik=payload["ik"],
        collision_ik=payload["collision_ik"],
        controller=payload["controller"],
        source=source,
    )


def _candidate_vector(candidate: DesignCandidate, specs: list[tuple[str, str, float, float]], cfg: dict[str, Any] | None = None) -> np.ndarray:
    record = candidate.as_record()
    mount_values = _mount_search_values(candidate.mount, cfg)
    values = []
    for section, key, low, high in specs:
        if section == "mount":
            value = float(mount_values.get(key, 0.5 * (low + high)))
        elif section == "base_policy":
            record_key = f"base_{key}"
            value = float(record.get(record_key, 0.5 * (low + high)))
        elif section == "collision_ik":
            record_key = f"collision_{key}"
            value = float(record.get(record_key, 0.5 * (low + high)))
        else:
            record_key = f"{section}_{key}"
            value = float(record.get(record_key, 0.5 * (low + high)))
        values.append((value - low) / max(high - low, 1e-12))
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def generate_design_candidates(
    cfg: dict[str, Any],
    samples: int,
    seed: int,
    strategy: str = "hybrid",
    optimize_controller: bool = False,
    start_index: int = 0,
    source: str = "sampled",
) -> list[DesignCandidate]:
    if int(samples) <= 0:
        return []
    specs = _design_bounds(cfg, optimize_controller)
    candidates: list[DesignCandidate] = []
    if start_index <= 0:
        candidates.append(_baseline_design_candidate(cfg))
    remaining = int(samples) - len(candidates)
    if remaining <= 0:
        return candidates[: int(samples)]
    unit = _unit_samples(remaining, len(specs), int(seed), strategy)
    for row_idx, row in enumerate(unit, start=max(1, int(start_index))):
        values = {
            (section, key): float(low + float(row[col_idx]) * (high - low))
            for col_idx, (section, key, low, high) in enumerate(specs)
        }
        candidates.append(_design_candidate_from_values(cfg, f"candidate_{row_idx:03d}", specs, values, source))
    return candidates[: int(samples)]


def _design_candidate_from_unit_vector(
    cfg: dict[str, Any],
    candidate_id: str,
    specs: list[tuple[str, str, float, float]],
    unit_vector: np.ndarray,
    source: str,
) -> DesignCandidate:
    unit = np.clip(np.asarray(unit_vector, dtype=float), 0.0, 1.0)
    if unit.shape != (len(specs),):
        raise ValueError(f"Expected genome with shape ({len(specs)},), got {unit.shape}.")
    values = {
        (section, key): float(low + unit[idx] * (high - low))
        for idx, (section, key, low, high) in enumerate(specs)
    }
    return _design_candidate_from_values(cfg, candidate_id, specs, values, source)


def _is_finite_record_value(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _design_candidate_from_record(
    cfg: dict[str, Any],
    record: dict[str, Any] | pd.Series,
    specs: list[tuple[str, str, float, float]],
    candidate_id: str,
    source: str,
) -> DesignCandidate:
    row = dict(record)
    if _mount_rotation_mode(cfg) == "forward_outer_quarter_sphere" and all(
        _is_finite_record_value(row.get(key))
        for key in ["left_search_roll_rad", "left_search_elevation_rad", "left_search_azimuth_rad"]
    ):
        mount_values = {
            "left_x_m": float(row["left_x_m"]),
            "left_y_m": float(row["left_y_m"]),
            "left_z_m": float(row["left_z_m"]),
            "left_delta_roll_rad": float(row["left_search_roll_rad"]),
            "left_delta_pitch_rad": float(row["left_search_elevation_rad"]),
            "left_delta_yaw_rad": float(row["left_search_azimuth_rad"]),
        }
    else:
        mount_values = {
            "left_x_m": float(row["left_x_m"]),
            "left_y_m": float(row["left_y_m"]),
            "left_z_m": float(row["left_z_m"]),
            "left_delta_roll_rad": float(row["left_delta_roll_rad"]),
            "left_delta_pitch_rad": float(row["left_delta_pitch_rad"]),
            "left_delta_yaw_rad": float(row["left_delta_yaw_rad"]),
        }
    values: dict[tuple[str, str], float] = {}
    for section, key, low, high in specs:
        if section == "mount":
            values[(section, key)] = float(np.clip(mount_values[key], low, high))
            continue
        prefix = "base" if section == "base_policy" else "collision" if section == "collision_ik" else section
        record_key = f"{prefix}_{key}"
        if _is_finite_record_value(row.get(record_key)):
            values[(section, key)] = float(np.clip(float(row[record_key]), low, high))
    for section, key, low, high in specs:
        values.setdefault((section, key), 0.5 * (low + high))
    return _design_candidate_from_values(cfg, candidate_id, specs, values, source)


def _append_unique_genome(genomes: list[np.ndarray], genome: np.ndarray, *, atol: float = 1e-9) -> None:
    unit = np.clip(np.asarray(genome, dtype=float), 0.0, 1.0)
    if any(np.allclose(unit, existing, atol=atol, rtol=0.0) for existing in genomes):
        return
    genomes.append(unit)


def _load_genetic_seed_genomes(
    cfg: dict[str, Any],
    specs: list[tuple[str, str, float, float]],
    seed_run: str | Path | None,
    seed_candidates: list[str] | None,
    seed_top: int,
) -> list[np.ndarray]:
    if seed_run in {None, ""}:
        return []
    run_dir = Path(seed_run).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"GA seed run does not exist: {run_dir}")
    source_path = None
    for name in ["full_validation_candidates.csv", "candidates.csv", "evaluation_archive.csv"]:
        candidate_path = run_dir / name
        if candidate_path.exists():
            source_path = candidate_path
            break
    if source_path is None:
        raise FileNotFoundError(f"No candidate CSV found in GA seed run: {run_dir}")
    df = pd.read_csv(source_path)
    if df.empty:
        return []
    if seed_candidates:
        requested = {str(candidate_id) for candidate_id in seed_candidates}
        selected = df[df["candidate_id"].astype(str).isin(requested)].copy()
        missing = sorted(requested - set(selected["candidate_id"].astype(str)))
        if missing:
            raise ValueError(f"GA seed candidates not found in {source_path}: {', '.join(missing)}")
    else:
        selected = _rank_candidate_frame(df).head(max(1, int(seed_top))).copy()
    genomes: list[np.ndarray] = []
    for _, row in selected.iterrows():
        candidate = _design_candidate_from_record(
            cfg,
            row,
            specs,
            candidate_id=f"seed_{row['candidate_id']}",
            source="genetic_seed",
        )
        _append_unique_genome(genomes, _candidate_vector(candidate, specs, cfg))
    return genomes


def _genetic_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("optimization", {}).get("genetic", {})
    raw = raw if isinstance(raw, dict) else {}
    return {
        "population": int(raw.get("population", 64)),
        "quick_generations": int(raw.get("quick_generations", 5)),
        "reduced_generations": int(raw.get("reduced_generations", 4)),
        "elite_fraction": float(raw.get("elite_fraction", 0.20)),
        "mutation_sigma": float(raw.get("mutation_sigma", 0.12)),
        "crossover_rate": float(raw.get("crossover_rate", 0.85)),
        "local_seed_fraction": float(raw.get("local_seed_fraction", 0.25)),
        "random_immigrant_fraction": float(raw.get("random_immigrant_fraction", 0.08)),
        "full_fraction": float(raw.get("full_fraction", 0.35)),
        "tournament_size": int(raw.get("tournament_size", 3)),
    }


def _genetic_baseline_genome(cfg: dict[str, Any], specs: list[tuple[str, str, float, float]]) -> np.ndarray:
    return _candidate_vector(_baseline_design_candidate(cfg), specs, cfg)


def _genetic_initial_genomes(
    cfg: dict[str, Any],
    specs: list[tuple[str, str, float, float]],
    population_size: int,
    seed: int,
    search_strategy: str,
    local_seed_fraction: float,
    seed_genomes: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    population_size = max(1, int(population_size))
    rng = np.random.default_rng(int(seed))
    baseline = _genetic_baseline_genome(cfg, specs)
    genomes = [baseline]
    seed_genomes = seed_genomes or []
    for seed_genome in seed_genomes:
        if len(genomes) >= population_size:
            break
        _append_unique_genome(genomes, seed_genome)
    if population_size <= 1:
        return genomes
    local_count = min(
        max(0, int(np.ceil((population_size - 1) * float(np.clip(local_seed_fraction, 0.0, 0.8))))),
        population_size - 1,
    )
    local_centers = [baseline, *seed_genomes] if seed_genomes else [baseline]
    for _idx in range(local_count):
        if len(genomes) >= population_size:
            break
        center = np.asarray(local_centers[_idx % len(local_centers)], dtype=float)
        _append_unique_genome(genomes, center + rng.normal(0.0, 0.08, size=len(specs)))
    remaining = population_size - len(genomes)
    if remaining > 0:
        unit = _unit_samples(remaining, len(specs), int(seed) + 1777, search_strategy)
        for row in unit:
            if len(genomes) >= population_size:
                break
            _append_unique_genome(genomes, row)
    return genomes[:population_size]


def _genetic_population_candidates(
    cfg: dict[str, Any],
    specs: list[tuple[str, str, float, float]],
    genomes: list[np.ndarray],
    *,
    generation: int,
    stage_label: str,
    source: str,
) -> list[DesignCandidate]:
    candidates: list[DesignCandidate] = []
    baseline = _genetic_baseline_genome(cfg, specs)
    for idx, genome in enumerate(genomes):
        if idx == 0 and np.allclose(np.asarray(genome, dtype=float), baseline, atol=1e-12, rtol=0.0):
            candidates.append(_baseline_design_candidate(cfg))
            continue
        candidate_id = f"candidate_ga_{stage_label}_g{generation:02d}_{idx:03d}"
        candidates.append(_design_candidate_from_unit_vector(cfg, candidate_id, specs, genome, source))
    return candidates


def _crowding_distance(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if values.shape[0] <= 2:
        return np.full(values.shape[0], np.inf, dtype=float)
    values = np.nan_to_num(values, nan=1e12, posinf=1e12, neginf=-1e12)
    distance = np.zeros(values.shape[0], dtype=float)
    for col in range(values.shape[1]):
        order = np.argsort(values[:, col])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = float(values[order[-1], col] - values[order[0], col])
        if span < 1e-12:
            continue
        for pos in range(1, len(order) - 1):
            distance[order[pos]] += float((values[order[pos + 1], col] - values[order[pos - 1], col]) / span)
    return distance


def _genetic_rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = _rank_candidate_frame(df)
    objective_cols = [
        col
        for col in [
            "failed_task_count",
            "mean_position_error_m",
            "max_position_error_m",
            "mean_orientation_error_rad",
            "max_orientation_error_rad",
            "torque_violation",
            "joint_speed_violation",
            "clearance_violation_m",
            "clearance_warning_violation_m",
            "mount_clearance_violation_m",
            "mount_plausibility_penalty",
            "architecture_plausibility_penalty",
        ]
        if col in ranked
    ]
    if objective_cols:
        ranked["ga_crowding_distance"] = _crowding_distance(ranked[objective_cols].astype(float).to_numpy())
    else:
        ranked["ga_crowding_distance"] = 0.0
    ranked["ga_rank_key"] = np.arange(len(ranked), dtype=int)
    return ranked


def _genetic_survivor_ids(df: pd.DataFrame, count: int, include_strict: bool = True) -> list[str]:
    if df.empty or int(count) <= 0:
        return []
    ranked = _genetic_rank_frame(df)
    survivors: list[str] = []
    strict_id = "candidate_000_strict_garmi"
    if include_strict and strict_id in set(ranked["candidate_id"].astype(str)):
        survivors.append(strict_id)

    eligible = ranked[ranked.get("eligible_candidate", False).astype(bool)].copy() if "eligible_candidate" in ranked else ranked.iloc[0:0].copy()
    infeasible = ranked[~ranked.index.isin(eligible.index)].copy()
    blocks: list[tuple[pd.DataFrame, list[str]]] = [
        (
            eligible,
            [
                "selection_failed_task_count",
                "pareto_rank",
                "hard_constraint_violation",
                "multi_objective_score",
                "optimization_score",
            ],
        ),
        (
            infeasible,
            [
                "hard_constraint_violation",
                "torque_violation",
                "clearance_violation_m",
                "mount_clearance_violation_m",
                "joint_speed_violation",
                "selection_failed_task_count",
                "multi_objective_score",
                "optimization_score",
            ],
        ),
    ]
    for front, preferred_cols in blocks:
        if front.empty:
            continue
        sort_cols = [col for col in preferred_cols if col in front]
        if sort_cols:
            front = front.sort_values([*sort_cols, "ga_crowding_distance"], ascending=[*[True] * len(sort_cols), False])
        else:
            front = front.sort_values("ga_crowding_distance", ascending=False)
        for candidate_id in front["candidate_id"].astype(str).tolist():
            if candidate_id not in survivors:
                survivors.append(candidate_id)
            if len(survivors) >= int(count):
                return survivors
    return survivors[: int(count)]


def _genetic_tournament(
    rng: np.random.Generator,
    ranked: pd.DataFrame,
    pool_ids: list[str],
    tournament_size: int,
) -> str:
    if not pool_ids:
        raise ValueError("Genetic tournament needs at least one parent.")
    sample_size = min(max(1, int(tournament_size)), len(pool_ids))
    sampled = rng.choice(pool_ids, size=sample_size, replace=False)
    rank_lookup = {str(row["candidate_id"]): int(pos) for pos, row in ranked.reset_index(drop=True).iterrows()}
    crowding = {str(row["candidate_id"]): float(row.get("ga_crowding_distance", 0.0)) for _, row in ranked.iterrows()}
    return min(
        [str(candidate_id) for candidate_id in sampled],
        key=lambda candidate_id: (rank_lookup.get(candidate_id, 10**9), -crowding.get(candidate_id, 0.0)),
    )


def _genetic_next_generation(
    df: pd.DataFrame,
    genome_by_id: dict[str, np.ndarray],
    population_size: int,
    seed: int,
    generation: int,
    *,
    elite_fraction: float,
    mutation_sigma: float,
    crossover_rate: float,
    random_immigrant_fraction: float,
    tournament_size: int,
) -> list[np.ndarray]:
    population_size = max(1, int(population_size))
    ranked = _genetic_rank_frame(df)
    survivor_ids = [candidate_id for candidate_id in _genetic_survivor_ids(ranked, population_size, include_strict=True) if candidate_id in genome_by_id]
    if not survivor_ids:
        return []
    rng = np.random.default_rng(int(seed) + 7919 * max(1, int(generation)))
    elite_count = min(len(survivor_ids), max(1, int(np.ceil(population_size * float(np.clip(elite_fraction, 0.02, 0.8))))))
    next_genomes = [genome_by_id[candidate_id].copy() for candidate_id in survivor_ids[:elite_count]]
    dim = len(next_genomes[0])
    immigrant_count = int(np.floor(population_size * float(np.clip(random_immigrant_fraction, 0.0, 0.4))))
    child_target = max(0, population_size - immigrant_count)
    parent_pool = survivor_ids
    while len(next_genomes) < child_target:
        first_id = _genetic_tournament(rng, ranked, parent_pool, tournament_size)
        second_id = _genetic_tournament(rng, ranked, parent_pool, tournament_size)
        first = genome_by_id[first_id]
        second = genome_by_id[second_id]
        if rng.random() < float(np.clip(crossover_rate, 0.0, 1.0)):
            blend = rng.random(dim)
            child = blend * first + (1.0 - blend) * second
        else:
            child = first.copy()
        child = child + rng.normal(0.0, float(max(0.0, mutation_sigma)), size=dim)
        next_genomes.append(np.clip(child, 0.0, 1.0))
    while len(next_genomes) < population_size:
        next_genomes.append(rng.random(dim))
    return next_genomes[:population_size]


def _renamed_design_candidate(candidate: DesignCandidate, candidate_id: str, source: str) -> DesignCandidate:
    mount = MountCandidate(candidate_id, candidate.mount.left_xyz_m.copy(), candidate.mount.left_delta_rpy_rad.copy())
    return DesignCandidate(
        candidate_id=candidate_id,
        mount=mount,
        base_policy=dict(candidate.base_policy),
        ik=dict(candidate.ik),
        collision_ik=dict(candidate.collision_ik),
        controller=dict(candidate.controller),
        source=source,
    )


def _point_proxy_clearance(point: np.ndarray, radius: float, ignored_names: set[str]) -> tuple[float, str]:
    best = float("inf")
    body = ""
    point = np.asarray(point, dtype=float)
    for proxy in garmi_body_proxies():
        if proxy.name in ignored_names:
            continue
        if isinstance(proxy, SphereProxy):
            clearance = float(np.linalg.norm(point - proxy.center) - radius - proxy.radius)
        elif isinstance(proxy, CapsuleProxy):
            clearance = _point_segment_distance(point, proxy.p0, proxy.p1) - radius - proxy.radius
        elif isinstance(proxy, BoxProxy):
            clearance = _point_box_clearance(point, radius, proxy)
        else:
            continue
        if clearance < best:
            best = float(clearance)
            body = proxy.name
    return best, body


def mount_plausibility(candidate: MountCandidate) -> dict[str, Any]:
    left_clearance, left_body = _point_proxy_clearance(
        candidate.left_xyz_m,
        0.045,
        {"left_shoulder_pad", "right_shoulder_pad"},
    )
    right_xyz, _right_rotation = _right_from_left(candidate.left_xyz_m, np.eye(3))
    right_clearance, right_body = _point_proxy_clearance(
        right_xyz,
        0.045,
        {"left_shoulder_pad", "right_shoulder_pad"},
    )
    min_clearance = min(left_clearance, right_clearance)
    target_clearance = 0.015
    penetration_penalty = 6000.0 * max(0.0, target_clearance - min_clearance) ** 2
    lateral_penalty = 18.0 * max(0.0, 0.21 - float(candidate.left_xyz_m[1])) ** 2
    shoulder_drift = np.linalg.norm(candidate.left_xyz_m - np.array([-0.14326, 0.20, 0.92451], dtype=float))
    drift_penalty = 1.8 * max(0.0, shoulder_drift - 0.18) ** 2
    orientation_penalty = 0.08 * float(np.linalg.norm(candidate.left_delta_rpy_rad) ** 2)
    penalty = float(penetration_penalty + lateral_penalty + drift_penalty + orientation_penalty)
    return {
        "mount_min_body_clearance_m": float(min_clearance),
        "mount_left_body": left_body,
        "mount_right_body": right_body,
        "mount_shoulder_drift_m": float(shoulder_drift),
        "mount_plausibility_penalty": penalty,
    }


def design_plausibility(candidate: CandidateLike) -> dict[str, Any]:
    mount = _mount_candidate(candidate)
    info = mount_plausibility(mount)
    if not isinstance(candidate, DesignCandidate):
        return info
    penalty = 0.0
    for key, neutral in {
        "position_kp_multiplier": 1.0,
        "position_kd_multiplier": 1.0,
        "orientation_kp_multiplier": 1.0,
        "orientation_kd_multiplier": 1.0,
        "nullspace_kp_multiplier": 1.0,
        "nullspace_kd_multiplier": 1.0,
        "base_kp_multiplier": 1.0,
        "base_kd_multiplier": 1.0,
    }.items():
        if key in candidate.controller:
            penalty += 0.12 * float(candidate.controller[key] - neutral) ** 2
    if candidate.base_policy:
        penalty += 0.45 * abs(float(candidate.base_policy.get("lateral_offset_m", 0.0)))
        penalty += 0.20 * abs(float(candidate.base_policy.get("yaw_offset_rad", 0.0)))
    if candidate.ik:
        penalty += 0.18 * max(0.0, float(candidate.ik.get("step_limit_rad", 0.18)) - 0.20) ** 2
    info["architecture_plausibility_penalty"] = float(info.get("architecture_plausibility_penalty", 0.0) + penalty)
    return info


def _objective_columns(summary: dict[str, Any]) -> dict[str, float]:
    failed = float(summary.get("failed_task_count", 0.0))
    task_count = max(float(summary.get("task_count", 0.0)), 1.0)
    torque_over = max(0.0, float(summary.get("max_torque_utilization", 0.0)) - 1.0)
    speed_over = max(0.0, float(summary.get("max_joint_speed_utilization", 0.0)) - 1.0)
    clearance_margin = float(summary.get("min_clearance_margin_m", summary.get("min_clearance_m", 0.0)))
    clearance_over = max(0.0, -clearance_margin)
    warning_clearance_over = max(0.0, -float(summary.get("min_warning_clearance_margin_m", 0.0)))
    mount_over = max(0.0, -float(summary.get("mount_min_body_clearance_m", 0.0)))
    base_speed = max(0.0, float(summary.get("max_base_speed_mps", 0.0)) - 0.85)
    base_acc = max(0.0, float(summary.get("max_base_acc_mps2", 0.0)) - 1.6)
    return {
        "objective_coverage_loss": failed / task_count,
        "objective_mean_pose_error": float(summary.get("mean_position_error_m", 0.0))
        + 0.15 * float(summary.get("mean_orientation_error_rad", 0.0)),
        "objective_worst_pose_error": float(summary.get("max_position_error_m", 0.0))
        + 0.08 * float(summary.get("max_orientation_error_rad", 0.0)),
        "objective_torque_overrun": float(torque_over),
        "objective_joint_speed_overrun": float(speed_over),
        "objective_collision_violation_m": float(max(clearance_over, mount_over)),
        "objective_clearance_warning_m": float(warning_clearance_over),
        "objective_base_motion": float(summary.get("mean_base_path_length_m", 0.0)) + 2.0 * base_speed + 0.5 * base_acc,
        "objective_plausibility": float(summary.get("mount_plausibility_penalty", 0.0))
        + float(summary.get("architecture_plausibility_penalty", 0.0)),
    }


def _score_candidate(rows: list[dict[str, Any]], plausibility: dict[str, Any]) -> dict[str, Any]:
    aggregate = _aggregate(rows, "candidate")
    df = pd.DataFrame(rows)
    failed = int(len(df) - int(df["success"].sum())) if len(df) else 0
    success_fraction = float(aggregate["success_fraction"])
    mean_objective = float(df["objective"].mean()) if len(df) else float("inf")
    worst_objective = float(df["objective"].max()) if len(df) else float("inf")
    # Continuous, threshold-normalized soft loss as the search signal. Binary
    # coverage stays available for reporting and the final selection.
    if len(df) and "soft_loss" in df:
        weights = df["task_weight"].astype(float).clip(lower=0.0) if "task_weight" in df else pd.Series(np.ones(len(df)))
        weight_sum = max(float(weights.sum()), 1e-9)
        mean_soft = float((df["soft_loss"].astype(float) * weights).sum() / weight_sum)
        worst_soft = float(df["soft_loss"].max())
    else:
        mean_soft = float("inf")
        worst_soft = float("inf")
    coverage_loss = 1.0 - success_fraction
    score = (
        mean_soft
        + 0.25 * worst_soft
        + 2.0 * coverage_loss
        + float(plausibility["mount_plausibility_penalty"])
        + float(plausibility.get("architecture_plausibility_penalty", 0.0))
    )
    # Coverage-dominated scalar score, reported for comparison only.
    coverage_weighted_score = (
        10000.0 * coverage_loss
        + mean_objective
        + 0.12 * worst_objective
        + float(plausibility["mount_plausibility_penalty"])
        + float(plausibility.get("architecture_plausibility_penalty", 0.0))
    )
    summary = {
        **aggregate,
        "failed_task_count": failed,
        "mean_task_objective": mean_objective,
        "worst_task_objective": worst_objective,
        "mean_soft_loss": mean_soft,
        "worst_soft_loss": worst_soft,
        "optimization_score": float(score),
        "coverage_weighted_score": float(coverage_weighted_score),
        **plausibility,
    }
    summary.update(_objective_columns(summary))
    return summary


def _fidelity_constraint_guard(cfg: dict[str, Any], fidelity_label: str | None) -> dict[str, float]:
    label = str(fidelity_label or "full").lower()
    defaults = {
        "quick": {
            "torque_utilization_limit": 0.70,
            "joint_speed_utilization_limit": 0.80,
            "min_clearance_m": 0.035,
            "mount_min_body_clearance_m": 0.015,
        },
        "reduced": {
            "torque_utilization_limit": 0.90,
            "joint_speed_utilization_limit": 0.95,
            "min_clearance_m": 0.015,
            "mount_min_body_clearance_m": 0.005,
        },
        "full": {
            "torque_utilization_limit": 1.00,
            "joint_speed_utilization_limit": 1.00,
            "min_clearance_m": 0.0,
            "mount_min_body_clearance_m": 0.0,
        },
    }
    guard = dict(defaults.get(label, defaults["full"]))
    raw = cfg.get("optimization", {}).get("fidelity", {}).get("constraint_guards", {})
    if isinstance(raw, dict):
        override = raw.get(label, {})
        if isinstance(override, dict):
            for key in guard:
                if key in override:
                    guard[key] = float(override[key])
    guard["torque_utilization_limit"] = max(1e-6, float(guard["torque_utilization_limit"]))
    guard["joint_speed_utilization_limit"] = max(1e-6, float(guard["joint_speed_utilization_limit"]))
    guard["min_clearance_m"] = float(guard["min_clearance_m"])
    guard["mount_min_body_clearance_m"] = float(guard["mount_min_body_clearance_m"])
    return guard


def _guard_violation_value(summary: dict[str, Any], guard: dict[str, float]) -> float:
    clearance_margin = float(summary.get("min_clearance_margin_m", summary.get("min_clearance_m", 0.0)))
    mount_clearance = float(summary.get("mount_min_body_clearance_m", 0.0))
    torque = float(summary.get("max_torque_utilization", 0.0))
    speed = float(summary.get("max_joint_speed_utilization", 0.0))
    return float(
        max(0.0, guard["min_clearance_m"] - clearance_margin)
        + max(0.0, guard["mount_min_body_clearance_m"] - mount_clearance)
        + max(0.0, torque / max(guard["torque_utilization_limit"], 1e-9) - 1.0)
        + max(0.0, speed / max(guard["joint_speed_utilization_limit"], 1e-9) - 1.0)
    )


def _apply_fidelity_constraint_guard(summary: dict[str, Any], cfg: dict[str, Any], fidelity_label: str | None) -> dict[str, Any]:
    """Attach constraint columns for ranking.

    The screening levels (quick/reduced) log their guard violations but do
    not gate candidate eligibility; only the full-fidelity guard decides.
    This prevents cheap but inaccurate screening evaluations from pruning
    candidates that would pass at full fidelity. Set
    optimization.fidelity.enforce_guards: true to prune at every level.
    """
    label = str(fidelity_label or "full").lower()
    screening_guard = _fidelity_constraint_guard(cfg, fidelity_label)
    enforce = bool(cfg.get("optimization", {}).get("fidelity", {}).get("enforce_guards", False))
    effective = screening_guard if (enforce or label == "full") else _fidelity_constraint_guard(cfg, "full")
    out = dict(summary)
    out["constraint_torque_utilization_limit"] = float(effective["torque_utilization_limit"])
    out["constraint_joint_speed_utilization_limit"] = float(effective["joint_speed_utilization_limit"])
    out["constraint_min_clearance_m"] = float(effective["min_clearance_m"])
    out["constraint_mount_min_body_clearance_m"] = float(effective["mount_min_body_clearance_m"])
    out["screening_guard_violation"] = _guard_violation_value(summary, screening_guard)
    out["screening_guard_enforced"] = bool(enforce or label == "full")
    return out


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _pareto_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=int)
    values = np.nan_to_num(values, nan=1e12, posinf=1e12, neginf=-1e12)
    remaining = set(range(values.shape[0]))
    ranks = np.zeros(values.shape[0], dtype=int)
    rank = 0
    while remaining:
        front: list[int] = []
        for idx in list(remaining):
            if not any(_dominates(values[other], values[idx]) for other in remaining if other != idx):
                front.append(idx)
        for idx in front:
            ranks[idx] = rank
            remaining.remove(idx)
        rank += 1
    return ranks


def _rank_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.copy()

    def numeric_column(name: str, default: float = 0.0) -> pd.Series:
        if name in ranked:
            return ranked[name].astype(float)
        return pd.Series(np.full(len(ranked), float(default)), index=ranked.index, dtype=float)

    if "min_clearance_margin_m" in ranked:
        clearance_margin = ranked["min_clearance_margin_m"].astype(float)
    elif "clearance_margin_m" in ranked:
        clearance_margin = ranked["clearance_margin_m"].astype(float)
    else:
        clearance_margin = numeric_column("min_clearance_m")
    clearance_required = numeric_column("constraint_min_clearance_m", 0.0)
    ranked["clearance_violation_m"] = np.maximum(0.0, clearance_required - clearance_margin)
    if "min_warning_clearance_margin_m" in ranked:
        warning_clearance_margin = ranked["min_warning_clearance_margin_m"].astype(float)
    elif "warning_clearance_margin_m" in ranked:
        warning_clearance_margin = ranked["warning_clearance_margin_m"].astype(float)
    else:
        warning_clearance_margin = clearance_margin
    ranked["clearance_warning_violation_m"] = np.maximum(0.0, -warning_clearance_margin)
    mount_clearance_required = numeric_column("constraint_mount_min_body_clearance_m", 0.0)
    ranked["mount_clearance_violation_m"] = np.maximum(0.0, mount_clearance_required - numeric_column("mount_min_body_clearance_m"))
    torque_limit = np.maximum(numeric_column("constraint_torque_utilization_limit", 1.0), 1e-6)
    speed_limit = np.maximum(numeric_column("constraint_joint_speed_utilization_limit", 1.0), 1e-6)
    ranked["torque_violation"] = np.maximum(0.0, numeric_column("max_torque_utilization") / torque_limit - 1.0)
    ranked["joint_speed_violation"] = np.maximum(0.0, numeric_column("max_joint_speed_utilization") / speed_limit - 1.0)
    ranked["hard_constraint_violation"] = (
        ranked["clearance_violation_m"]
        + ranked["mount_clearance_violation_m"]
        + ranked["torque_violation"]
        + ranked["joint_speed_violation"]
    )
    # Eligibility with tolerance: a near-zero threshold would rank a candidate
    # with a 2 mm clearance shortfall behind clearly worse but "clean"
    # candidates. Violations below epsilon still count in the score; they just
    # do not gate the ranking.
    ranked["eligible_candidate"] = ranked["hard_constraint_violation"] <= CONSTRAINT_EPSILON
    objectives = [
        "failed_task_count",
        "mean_position_error_m",
        "max_position_error_m",
        "mean_orientation_error_rad",
        "max_orientation_error_rad",
        "torque_violation",
        "joint_speed_violation",
        "clearance_violation_m",
        "clearance_warning_violation_m",
        "mount_clearance_violation_m",
        "mount_plausibility_penalty",
        "architecture_plausibility_penalty",
    ]
    matrix = ranked[[col for col in objectives if col in ranked]].astype(float).to_numpy()
    ranked["pareto_rank"] = _pareto_ranks(matrix)
    normalized = np.zeros(len(ranked), dtype=float)
    weights = {
        "failed_task_count": 4.0,
        "mean_position_error_m": 1.0,
        "max_position_error_m": 0.8,
        "mean_orientation_error_rad": 0.35,
        "max_orientation_error_rad": 0.25,
        "torque_violation": 0.9,
        "joint_speed_violation": 0.7,
        "clearance_violation_m": 2.0,
        "clearance_warning_violation_m": 0.35,
        "mount_clearance_violation_m": 2.0,
        "mount_plausibility_penalty": 0.35,
        "architecture_plausibility_penalty": 0.35,
    }
    for col in objectives:
        if col not in ranked:
            continue
        values = ranked[col].astype(float).to_numpy()
        span = float(np.nanmax(values) - np.nanmin(values)) if len(values) else 0.0
        scaled = np.zeros_like(values) if span < 1e-12 else (values - np.nanmin(values)) / span
        normalized += float(weights.get(col, 1.0)) * np.nan_to_num(scaled, nan=1.0, posinf=1.0, neginf=0.0)
    ranked["multi_objective_score"] = normalized
    ranked["selection_failed_task_count"] = numeric_column("failed_task_count", 1e6)
    ranked["selection_ineligible"] = (~ranked["eligible_candidate"]).astype(int)
    # Two-tier constraint handling: extreme violations (deep penetrations,
    # torque/speed far beyond the limits) disqualify a candidate outright.
    # Small violations must not be ranked before coverage, otherwise a
    # candidate that solves no task (and therefore violates nothing) would
    # outrank a high-coverage candidate with a minor overrun. Small violations
    # act as tie-breakers between candidates of comparable coverage.
    ranked["selection_hard_ineligible"] = (
        ranked["hard_constraint_violation"] > HARD_CONSTRAINT_CAP
    ).astype(int)
    sort_columns = [
        col
        for col in [
            "selection_hard_ineligible",
            "selection_failed_task_count",
            "selection_ineligible",
            "optimization_score",
            "hard_constraint_violation",
            "pareto_rank",
            "multi_objective_score",
            "mean_position_error_m",
        ]
        if col in ranked
    ]
    return ranked.sort_values(sort_columns).reset_index(drop=True)


_OPT_CFG: dict[str, Any] | None = None
_OPT_TASKS: list[TaskData] | None = None
_OPT_BASE_SEARCH = "reduced"


def _downsample_task(task: TaskData, sample_stride: int = 1, max_samples: int = 0) -> TaskData:
    stride = max(1, int(sample_stride))
    indices = np.arange(0, task.sample_count, stride, dtype=int)
    if int(max_samples) > 0 and len(indices) > int(max_samples):
        indices = np.linspace(0, task.sample_count - 1, int(max_samples), dtype=int)
        indices = np.unique(indices)
    if len(indices) == task.sample_count and np.all(indices == np.arange(task.sample_count)):
        return task
    return TaskData(
        name=task.name,
        time=task.time[indices],
        time_abs=task.time_abs[indices],
        left_pos=task.left_pos[indices],
        left_rot=task.left_rot[indices],
        right_pos=task.right_pos[indices],
        right_rot=task.right_rot[indices],
        left_force=task.left_force[indices],
        right_force=task.right_force[indices],
        left_torque=task.left_torque[indices],
        right_torque=task.right_torque[indices],
        metadata={**task.metadata, "fidelity_sample_stride": stride, "fidelity_max_samples": int(max_samples)},
    )


def _load_fidelity_tasks(
    task_names: list[str],
    paths: ProjectPaths,
    fidelity: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[TaskData]:
    """Load tasks; names with an '@p<k>' suffix become perturbed robustness variants."""
    robustness = robustness_config(cfg or {})
    tasks: list[TaskData] = []
    for name in task_names:
        base_name, variant = split_variant_name(name)
        task = load_task(base_name, paths)
        if variant is not None:
            task = perturb_task(task, variant, robustness)
        tasks.append(task)
    fidelity = fidelity or {}
    stride = int(fidelity.get("sample_stride", 1))
    max_samples = int(fidelity.get("max_samples", 0))
    return [_downsample_task(task, sample_stride=stride, max_samples=max_samples) for task in tasks]


def _init_optimization_worker(
    cfg: dict[str, Any],
    task_names: list[str],
    root: str,
    base_search: str,
    fidelity: dict[str, Any] | None = None,
) -> None:
    global _OPT_CFG, _OPT_TASKS, _OPT_BASE_SEARCH
    _OPT_CFG = cfg
    _OPT_TASKS = _load_fidelity_tasks(task_names, ProjectPaths(Path(root)), fidelity, cfg)
    _OPT_BASE_SEARCH = str(base_search)


def _evaluate_candidate_worker(candidate: CandidateLike) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if _OPT_CFG is None or _OPT_TASKS is None:
        raise RuntimeError("Optimization worker is not initialized.")
    cfg = _apply_candidate_config(_OPT_CFG, candidate, base_search=_OPT_BASE_SEARCH, serial_evaluation=True)
    record = _candidate_record(candidate, _OPT_CFG)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task in _OPT_TASKS:
        result = evaluate_task_baseline(task, cfg)
        rows.append({"candidate_id": candidate.candidate_id, **record, **result["metrics"]})
    plausibility = design_plausibility(candidate)
    summary = {
        "candidate_id": candidate.candidate_id,
        **record,
        **_score_candidate(rows, plausibility),
        "elapsed_s": float(time.perf_counter() - started),
    }
    return summary, rows, cfg


def _worker_count(workers: int | None, candidate_count: int) -> int:
    if workers is None:
        value = os.cpu_count() or 1
    else:
        value = int(workers)
    return max(1, min(value, max(1, candidate_count)))


def candidate_hash(candidate: CandidateLike, task_names: list[str], fidelity_label: str, base_search: str) -> str:
    payload = {
        "evaluation_version": "task_masks_wrench_constraint_guards_v2",
        "candidate": _candidate_record(candidate),
        "task_names": list(task_names),
        "fidelity": str(fidelity_label),
        "base_search": str(base_search),
    }
    text = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / key / "summary.json", cache_dir / key / "task_rows.csv"


def _load_candidate_cache(cache_dir: Path | None, key: str) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if cache_dir is None:
        return None
    summary_path, rows_path = _cache_paths(cache_dir, key)
    if not summary_path.exists() or not rows_path.exists():
        return None
    summary = read_json(summary_path)
    rows = pd.read_csv(rows_path).to_dict("records")
    summary["cache_hit"] = True
    return summary, rows


def _write_candidate_cache(cache_dir: Path | None, key: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if cache_dir is None:
        return
    summary_path, rows_path = _cache_paths(cache_dir, key)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, _jsonable(summary))
    pd.DataFrame(rows).to_csv(rows_path, index=False)


def _evaluate_mount_candidate_batch(
    candidates: list[CandidateLike],
    *,
    base_cfg: dict[str, Any],
    task_names: list[str],
    paths: ProjectPaths,
    group_name: str,
    workers: int | None,
    base_search: str,
    stage_label: str,
    fidelity: dict[str, Any] | None = None,
    fidelity_label: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    fidelity_label = str(fidelity_label or stage_label)
    # Fidelity levels may override collision-aware IK (screening levels run
    # plain IK; only full fidelity pays for the avoidance refinement).
    level_cfg = base_cfg
    if fidelity and "collision_aware_ik" in fidelity:
        level_cfg = dict(base_cfg)
        level_cfg["collision"] = dict(base_cfg.get("collision", {}))
        level_cfg["collision"]["collision_aware_ik"] = bool(fidelity["collision_aware_ik"])
    summaries: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    candidate_cfgs: dict[str, dict[str, Any]] = {}
    pending: list[CandidateLike] = []
    cache_keys: dict[str, str] = {}
    for candidate in candidates:
        key = candidate_hash(candidate, task_names, fidelity_label, base_search)
        cache_keys[candidate.candidate_id] = key
        cached = _load_candidate_cache(cache_dir, key)
        if cached is None:
            pending.append(candidate)
            continue
        summary, rows = cached
        summary = _apply_fidelity_constraint_guard(summary, base_cfg, fidelity_label)
        summary["fidelity"] = fidelity_label
        summary["stage"] = stage_label
        summaries.append(summary)
        task_rows.extend(rows)
        candidate_cfgs[candidate.candidate_id] = _apply_candidate_config(base_cfg, candidate, base_search=base_search, serial_evaluation=False)
    if not pending:
        return summaries, task_rows, candidate_cfgs
    worker_count = _worker_count(workers, len(pending))
    started = time.perf_counter()
    print(
        f"[mount-opt:{group_name}:{stage_label}] {len(pending)} candidates x {len(task_names)} task evaluations "
        f"(base_search={base_search}, collision_aware_ik={bool(level_cfg.get('collision', {}).get('collision_aware_ik', False))}); "
        "first result can take several minutes at reduced/full fidelity",
        flush=True,
    )
    if worker_count <= 1:
        global _OPT_CFG, _OPT_TASKS, _OPT_BASE_SEARCH
        _OPT_CFG = level_cfg
        _OPT_TASKS = _load_fidelity_tasks(task_names, paths, fidelity, level_cfg)
        _OPT_BASE_SEARCH = str(base_search)
        for idx, candidate in enumerate(pending, start=1):
            summary, rows, candidate_cfg = _evaluate_candidate_worker(candidate)
            summary = _apply_fidelity_constraint_guard(summary, base_cfg, fidelity_label)
            summary["fidelity"] = fidelity_label
            summary["stage"] = stage_label
            summary["candidate_hash"] = cache_keys[candidate.candidate_id]
            summary["cache_hit"] = False
            summaries.append(summary)
            task_rows.extend(rows)
            candidate_cfgs[candidate.candidate_id] = candidate_cfg
            _write_candidate_cache(cache_dir, cache_keys[candidate.candidate_id], summary, rows)
            print(f"[mount-opt:{group_name}:{stage_label}] {idx}/{len(pending)} {candidate.candidate_id}", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_init_optimization_worker,
            initargs=(level_cfg, task_names, str(paths.root), base_search, fidelity),
        ) as executor:
            futures = {executor.submit(_evaluate_candidate_worker, candidate): candidate.candidate_id for candidate in pending}
            for idx, future in enumerate(as_completed(futures), start=1):
                summary, rows, candidate_cfg = future.result()
                summary = _apply_fidelity_constraint_guard(summary, base_cfg, fidelity_label)
                summary["fidelity"] = fidelity_label
                summary["stage"] = stage_label
                summary["candidate_hash"] = cache_keys[summary["candidate_id"]]
                summary["cache_hit"] = False
                summaries.append(summary)
                task_rows.extend(rows)
                candidate_cfgs[summary["candidate_id"]] = candidate_cfg
                _write_candidate_cache(cache_dir, cache_keys[summary["candidate_id"]], summary, rows)
                elapsed = time.perf_counter() - started
                print(f"[mount-opt:{group_name}:{stage_label}] {idx}/{len(pending)} candidates in {elapsed:.1f}s", flush=True)
    return summaries, task_rows, candidate_cfgs


def _write_optimization_report(
    run_dir: Path,
    group_name: str,
    candidate_df: pd.DataFrame,
    task_df: pd.DataFrame,
    best_summary: dict[str, Any],
    validation_df: pd.DataFrame | None = None,
    validation_task_df: pd.DataFrame | None = None,
    mujoco_validation_df: pd.DataFrame | None = None,
    title: str = "Mount Optimization",
    description: str = "Franka/Panda-only architecture optimization: mirrored GARMI shoulder mounts with unchanged original Panda 7-DoF arm kinematics.",
) -> Path:
    try:
        import plotly.express as px
        from plotly.offline import plot

        scatter = px.scatter(
            candidate_df,
            x="multi_objective_score" if "multi_objective_score" in candidate_df else "optimization_score",
            y="success_fraction",
            color="pareto_rank" if "pareto_rank" in candidate_df else "mount_min_body_clearance_m",
            size=np.clip(candidate_df["mean_manipulability"], 0.02, None),
            hover_name="candidate_id",
            hover_data=[
                "pareto_rank",
                "failed_task_count",
                "optimization_score",
                "mean_position_error_m",
                "max_position_error_m",
                "mean_orientation_error_rad",
                "min_clearance_m",
                "max_torque_utilization",
                "max_joint_speed_utilization",
            ],
            title=f"{title} candidates: {group_name}",
        )
        scatter_html = plot(scatter, include_plotlyjs="cdn", output_type="div")
        task_plot = px.bar(
            task_df.sort_values(["candidate_id", "objective"]),
            x="task_name",
            y="objective",
            color="success",
            facet_row="candidate_id" if len(candidate_df) <= 8 else None,
            title="Per-task objective by candidate",
        )
        task_html = plot(task_plot, include_plotlyjs=False, output_type="div")
    except Exception:
        scatter_html = "<p>Plotly unavailable.</p>"
        task_html = "<p>Plotly unavailable.</p>"

    best_table = pd.DataFrame([best_summary]).to_html(index=False, float_format=lambda value: f"{value:.5f}")
    candidates_table = candidate_df.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    pareto_html = ""
    pareto_path = run_dir / "pareto_candidates.csv"
    if pareto_path.exists():
        try:
            pareto_table = pd.read_csv(pareto_path).to_html(index=False, float_format=lambda value: f"{value:.5f}")
            pareto_html = f"""
<h2>Pareto Candidates</h2>
<p>Non-dominated candidates from the final fidelity stage.</p>
{pareto_table}
"""
        except Exception:
            pareto_html = ""
    archive_html = ""
    archive_path = run_dir / "evaluation_archive.csv"
    if archive_path.exists():
        try:
            archive_df = pd.read_csv(archive_path)
            archive_view = archive_df[
                [
                    col
                    for col in [
                        "stage",
                        "fidelity",
                        "candidate_id",
                        "cache_hit",
                        "success_fraction",
                        "failed_task_count",
                        "optimization_score",
                        "multi_objective_score",
                        "hard_constraint_violation",
                        "elapsed_s",
                    ]
                    if col in archive_df
                ]
            ].copy()
            archive_table = archive_view.to_html(index=False, float_format=lambda value: f"{value:.5f}")
            archive_html = f"""
<h2>Fidelity History</h2>
<p>Candidate evaluations across quick, reduced and full stages. Cache hits are reused evaluations from the run cache.</p>
{archive_table}
"""
        except Exception:
            archive_html = ""
    failures = task_df[task_df["success"] == False].copy()  # noqa: E712
    failures_table = failures.to_html(index=False, float_format=lambda value: f"{value:.5f}") if len(failures) else "<p>No failed tasks.</p>"
    validation_html = ""
    if validation_df is not None and len(validation_df):
        validation_table = validation_df.to_html(index=False, float_format=lambda value: f"{value:.5f}")
        validation_failures = validation_task_df[validation_task_df["success"] == False].copy() if validation_task_df is not None else pd.DataFrame()  # noqa: E712
        validation_failure_table = (
            validation_failures.to_html(index=False, float_format=lambda value: f"{value:.5f}") if len(validation_failures) else "<p>No failed tasks.</p>"
        )
        validation_html = f"""
<h2>Full Validation</h2>
<p>Top search candidates plus the strict GARMI baseline re-evaluated with the full base variant grid.</p>
{validation_table}
<h2>Full Validation Failed Tasks</h2>
{validation_failure_table}
"""
    mujoco_html = ""
    if mujoco_validation_df is not None and len(mujoco_validation_df):
        display = mujoco_validation_df.copy()
        for column in ["mujoco_report", "mujoco_batch_dir"]:
            if column in display:
                display[column] = display[column].map(lambda path: f'<a href="{Path(path).resolve().as_uri()}">open</a>' if path else "")
        mujoco_table = display.to_html(index=False, escape=False, float_format=lambda value: f"{value:.5f}")
        mujoco_html = f"""
<h2>MuJoCo Top Candidate Validation</h2>
<p>Optional operational-space impedance simulation for selected top mount candidates. This remains Panda-only.</p>
{mujoco_table}
"""
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{title} {group_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #182027; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d7dde2; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f3f6f8; }}
</style></head>
<body>
<h1>{title}: {group_name}</h1>
<p>{description}</p>
<h2>Best Candidate</h2>
{best_table}
<h2>Candidate Map</h2>
{scatter_html}
<h2>Ranking Notes</h2>
<p>Candidates are sorted task-first: eligible candidates, failed task count, weighted task score, hard constraint violation, Pareto rank, and multi-objective score. Extreme torque, joint-speed, or collision violations are marked ineligible and cannot beat an eligible candidate.</p>
<h2>Failed Tasks</h2>
{failures_table}
{pareto_html}
{archive_html}
{validation_html}
{mujoco_html}
<h2>Per-Task Objective</h2>
{task_html}
<h2>All Candidates</h2>
{candidates_table}
</body></html>
"""
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _validate_top_candidates_full(
    candidate_df: pd.DataFrame,
    candidate_by_id: dict[str, CandidateLike],
    base_cfg: dict[str, Any],
    task_names: list[str],
    group_name: str,
    paths: ProjectPaths,
    validate_top: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], CandidateLike]:
    strict_id = "candidate_000_strict_garmi"
    top_ids = [str(value) for value in candidate_df.head(max(0, int(validate_top)))["candidate_id"].tolist()]
    if strict_id in candidate_by_id and strict_id not in top_ids:
        top_ids.append(strict_id)
    tasks = load_tasks(task_names, paths)
    validation_summaries: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for idx, candidate_id in enumerate(top_ids, start=1):
        candidate = candidate_by_id[candidate_id]
        cfg = _apply_candidate_config(base_cfg, candidate, base_search="full", serial_evaluation=False)
        rows = evaluate_tasks_parallel(tasks, cfg, f"{group_name}_{candidate_id}_full")
        record = _candidate_record(candidate, base_cfg)
        rows = [{"candidate_id": candidate_id, **record, **row} for row in rows]
        plausibility = design_plausibility(candidate)
        summary = {
            "candidate_id": candidate_id,
            **record,
            **_score_candidate(rows, plausibility),
            "validation_rank": idx,
        }
        validation_summaries.append(summary)
        validation_rows.extend(rows)
    validation_df = _rank_candidate_frame(pd.DataFrame(validation_summaries))
    validation_task_df = pd.DataFrame(validation_rows).sort_values(["candidate_id", "task_name"]).reset_index(drop=True)
    best_summary = validation_df.iloc[0].to_dict()
    best_candidate = candidate_by_id[str(best_summary["candidate_id"])]
    return validation_df, validation_task_df, best_summary, best_candidate


def _validate_mujoco_top_candidates(
    candidate_df: pd.DataFrame,
    candidate_by_id: dict[str, CandidateLike],
    base_cfg: dict[str, Any],
    task_names: list[str],
    group_name: str,
    paths: ProjectPaths,
    run_dir: Path,
    count: int,
    workers: int | None,
    render: bool,
    enable_contacts: bool,
) -> pd.DataFrame:
    if int(count) <= 0 or candidate_df.empty:
        return pd.DataFrame()
    from garmi_optimization.mujoco_backend import simulate_batch

    # MuJoCo validation always runs on the nominal recordings; robustness
    # perturbation variants (task@p<k>) are an IK-level construct.
    task_names = [name for name in task_names if split_variant_name(name)[1] is None]
    strict_id = "candidate_000_strict_garmi"
    top_ids = [str(value) for value in candidate_df.head(max(0, int(count)))["candidate_id"].tolist()]
    if strict_id in candidate_by_id and strict_id not in top_ids:
        top_ids.append(strict_id)
    config_dir = run_dir / "mujoco_validation_configs"
    rows: list[dict[str, Any]] = []
    for candidate_id in top_ids:
        candidate = candidate_by_id[candidate_id]
        cfg = _apply_candidate_config(base_cfg, candidate, base_search="full", serial_evaluation=False)
        cfg["_profile"] = f"{base_cfg.get('_profile', DEFAULT_PROFILE)}_{candidate_id}_mujoco"
        cfg.setdefault("simulation", {})["enable_contacts"] = bool(enable_contacts)
        cfg_path = config_dir / f"{candidate_id}.yaml"
        write_yaml(cfg_path, cfg)
        batch_dir = simulate_batch(
            task_names,
            f"{group_name}_{candidate_id}",
            paths=paths,
            workers=workers,
            render=render,
            cfg=cfg,
            baseline_profile=str(cfg_path),
        )
        summary = read_json(batch_dir / "batch_summary.json")
        rows.append(
            {
                "candidate_id": candidate_id,
                **_candidate_record(candidate, base_cfg),
                "mujoco_batch_dir": str(batch_dir),
                "mujoco_report": str(batch_dir / "report.html"),
                "mujoco_rendered_videos": bool(render),
                "mujoco_contacts_enabled": bool(enable_contacts),
                "mujoco_success_fraction": float(summary.get("simulation_success_fraction", 0.0)),
                "mujoco_success_count": int(summary.get("simulation_success_count", 0)),
                "mujoco_mean_left_tracking_error_m": float(summary.get("mean_left_tracking_error_m", np.nan)),
                "mujoco_mean_right_tracking_error_m": float(summary.get("mean_right_tracking_error_m", np.nan)),
                "mujoco_max_left_tracking_error_m": float(summary.get("max_left_tracking_error_m", np.nan)),
                "mujoco_max_right_tracking_error_m": float(summary.get("max_right_tracking_error_m", np.nan)),
                "mujoco_max_left_joint_torque_clip_fraction": float(summary.get("max_left_joint_torque_clip_fraction", np.nan)),
                "mujoco_max_right_joint_torque_clip_fraction": float(summary.get("max_right_joint_torque_clip_fraction", np.nan)),
                "mujoco_max_contact_count": int(summary.get("max_contact_count", 0)),
                "mujoco_min_contact_distance_m": float(summary.get("min_contact_distance_m", np.nan)),
                "mujoco_elapsed_s": float(summary.get("elapsed_s", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _sort_mujoco_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.copy()
    max_tracking_cols = [
        col
        for col in [
            "mujoco_max_left_tracking_error_m",
            "mujoco_max_right_tracking_error_m",
        ]
        if col in ranked
    ]
    mean_tracking_cols = [
        col
        for col in [
            "mujoco_mean_left_tracking_error_m",
            "mujoco_mean_right_tracking_error_m",
        ]
        if col in ranked
    ]
    clip_cols = [
        col
        for col in [
            "mujoco_max_left_joint_torque_clip_fraction",
            "mujoco_max_right_joint_torque_clip_fraction",
        ]
        if col in ranked
    ]
    ranked["_mujoco_max_tracking"] = (
        np.nanmax(ranked[max_tracking_cols].astype(float).to_numpy(), axis=1) if max_tracking_cols else np.inf
    )
    ranked["_mujoco_mean_tracking"] = (
        np.nanmean(ranked[mean_tracking_cols].astype(float).to_numpy(), axis=1) if mean_tracking_cols else np.inf
    )
    ranked["_mujoco_max_clip"] = np.nanmax(ranked[clip_cols].astype(float).to_numpy(), axis=1) if clip_cols else np.inf
    sort_cols = [
        col
        for col in [
            "mujoco_success_count",
            "_mujoco_max_clip",
            "_mujoco_max_tracking",
            "_mujoco_mean_tracking",
            "mujoco_max_contact_count",
        ]
        if col in ranked
    ]
    ascending = [False if col == "mujoco_success_count" else True for col in sort_cols]
    if sort_cols:
        ranked = ranked.sort_values(sort_cols, ascending=ascending)
    return ranked.reset_index(drop=True)


def _write_best_mujoco_config(
    run_dir: Path,
    base_cfg: dict[str, Any],
    candidate_by_id: dict[str, CandidateLike],
    mujoco_validation_df: pd.DataFrame,
) -> dict[str, Any]:
    if mujoco_validation_df.empty:
        return {}
    ranked = _sort_mujoco_candidates(mujoco_validation_df)
    best = ranked.iloc[0].drop(labels=[col for col in ranked.columns if col.startswith("_mujoco_")], errors="ignore").to_dict()
    candidate_id = str(best.get("candidate_id", ""))
    candidate = candidate_by_id.get(candidate_id)
    if candidate is None:
        return best
    cfg = _apply_candidate_config(base_cfg, candidate, base_search="full", serial_evaluation=False)
    cfg["_profile"] = f"{base_cfg.get('_profile', DEFAULT_PROFILE)}_{candidate_id}_best_mujoco"
    write_yaml(run_dir / "best_mujoco_config.yaml", cfg)
    write_json(run_dir / "best_mujoco_candidate.json", _jsonable(best))
    return best


def _fidelity_levels(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw_levels = cfg.get("optimization", {}).get("fidelity", {}).get("levels", {})
    defaults = {
        # Collision-aware IK only at full fidelity (the expensive, decisive
        # evaluation); the screening levels run plain IK.
        "quick": {"sample_stride": 12, "max_samples": 60, "base_search": "single", "collision_aware_ik": False},
        "reduced": {"sample_stride": 6, "max_samples": 140, "base_search": "reduced", "collision_aware_ik": False},
        "full": {"sample_stride": 1, "max_samples": 0, "base_search": "full"},
    }
    levels = []
    for name in ["quick", "reduced", "full"]:
        level = dict(defaults[name])
        if isinstance(raw_levels, dict) and isinstance(raw_levels.get(name), dict):
            level.update(raw_levels[name])
        level["name"] = name
        levels.append(level)
    return levels


def _candidate_ids_from_frame(df: pd.DataFrame, count: int, include_strict: bool = True) -> list[str]:
    if df.empty or int(count) <= 0:
        return ["candidate_000_strict_garmi"] if include_strict and "candidate_000_strict_garmi" in set(df.get("candidate_id", [])) else []
    ranked = _rank_candidate_frame(df)
    ids = [str(value) for value in ranked.head(int(count))["candidate_id"].tolist()]
    strict_id = "candidate_000_strict_garmi"
    if include_strict and strict_id in set(df["candidate_id"].astype(str)) and strict_id not in ids:
        ids.append(strict_id)
    return ids


def _candidate_ids_by_success(df: pd.DataFrame, count: int, include_strict: bool = True) -> list[str]:
    if df.empty or int(count) <= 0:
        return []
    ranked = _rank_candidate_frame(df)
    sort_cols = [
        col
        for col in [
            "selection_ineligible",
            "success_count",
            "selection_failed_task_count",
            "optimization_score",
            "hard_constraint_violation",
            "multi_objective_score",
        ]
        if col in ranked
    ]
    ascending = []
    for col in sort_cols:
        ascending.append(False if col == "success_count" else True)
    success_ranked = ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True) if sort_cols else ranked
    ids = [str(value) for value in success_ranked.head(int(count))["candidate_id"].tolist()]
    strict_id = "candidate_000_strict_garmi"
    if include_strict and strict_id in set(df["candidate_id"].astype(str)) and strict_id not in ids:
        ids.append(strict_id)
    return ids


def _promoted_candidates(
    df: pd.DataFrame,
    candidate_by_id: dict[str, CandidateLike],
    minimum: int,
    fraction: float,
    maximum: int | None = None,
) -> list[CandidateLike]:
    count = max(int(minimum), int(np.ceil(len(df) * float(np.clip(fraction, 0.05, 1.0)))))
    if maximum is not None:
        # Absolute cap: with BO proposals the archive can hold about twice the
        # initial samples, and promoting a fixed fraction of it to the
        # expensive levels would dominate the run time.
        count = min(count, max(int(minimum), int(maximum)))
    ids: list[str] = []
    for candidate_id in [
        *_candidate_ids_by_success(df, count=max(int(minimum), count // 2), include_strict=True),
        *_candidate_ids_from_frame(df, count=count, include_strict=True),
    ]:
        if candidate_id not in ids:
            ids.append(candidate_id)
    return [candidate_by_id[candidate_id] for candidate_id in ids if candidate_id in candidate_by_id]


def _pareto_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = _rank_candidate_frame(df)
    if "eligible_candidate" in ranked and bool(ranked["eligible_candidate"].any()):
        ranked = ranked[ranked["eligible_candidate"]].copy()
    return ranked[ranked["pareto_rank"] == 0].copy().reset_index(drop=True)


def _surrogate_candidate_proposals(
    cfg: dict[str, Any],
    archive_df: pd.DataFrame,
    candidate_by_id: dict[str, CandidateLike],
    count: int,
    seed: int,
    optimize_controller: bool,
    start_index: int,
) -> list[DesignCandidate]:
    if int(count) <= 0:
        return []
    existing_ids = set(candidate_by_id)
    specs = _design_bounds(cfg, optimize_controller)
    design_candidates = [candidate for candidate in candidate_by_id.values() if isinstance(candidate, DesignCandidate)]
    if len(design_candidates) < 6 or archive_df.empty:
        generated = generate_design_candidates(
            cfg,
            samples=int(count),
            seed=int(seed),
            strategy="sobol",
            optimize_controller=optimize_controller,
            start_index=start_index,
            source="surrogate_fallback",
        )
        return [candidate for candidate in generated if candidate.candidate_id not in existing_ids][: int(count)]

    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        generated = generate_design_candidates(
            cfg,
            samples=int(count),
            seed=int(seed),
            strategy="sobol",
            optimize_controller=optimize_controller,
            start_index=start_index,
            source="surrogate_fallback",
        )
        return [candidate for candidate in generated if candidate.candidate_id not in existing_ids][: int(count)]

    latest = archive_df.sort_values(["fidelity", "stage"]).drop_duplicates("candidate_id", keep="last")
    rows = []
    y = []
    for _, row in latest.iterrows():
        candidate_id = str(row.get("candidate_id", ""))
        candidate = candidate_by_id.get(candidate_id)
        if not isinstance(candidate, DesignCandidate):
            continue
        rows.append(_candidate_vector(candidate, specs, cfg))
        y.append(float(row.get("optimization_score", 1e6)))
    if len(rows) < 6:
        return []
    x_train = np.vstack(rows)
    y_train = np.asarray(y, dtype=float)
    model = RandomForestRegressor(
        n_estimators=96,
        min_samples_leaf=2,
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    pool_size = int(cfg.get("optimization", {}).get("surrogate_pool_size", 512))
    pool = generate_design_candidates(
        cfg,
        samples=max(pool_size, int(count) * 16),
        seed=int(seed) + 911,
        strategy="sobol",
        optimize_controller=optimize_controller,
        start_index=start_index,
        source="surrogate_pool",
    )
    pool = [candidate for candidate in pool if candidate.candidate_id not in existing_ids]
    if not pool:
        return []
    x_pool = np.vstack([_candidate_vector(candidate, specs, cfg) for candidate in pool])
    tree_preds = np.vstack([tree.predict(x_pool) for tree in model.estimators_])
    pred = tree_preds.mean(axis=0)
    uncertainty = tree_preds.std(axis=0)
    selected: list[DesignCandidate] = []
    selected_vectors: list[np.ndarray] = []
    order = np.argsort(pred - 0.10 * uncertainty)
    for idx in order:
        candidate = pool[int(idx)]
        vector = x_pool[int(idx)]
        if selected_vectors:
            nearest = min(float(np.linalg.norm(vector - other)) for other in selected_vectors)
            if nearest < 0.06:
                continue
        renamed = _renamed_design_candidate(candidate, f"candidate_{start_index + len(selected):03d}_surrogate", "surrogate")
        selected.append(renamed)
        selected_vectors.append(vector)
        if len(selected) >= int(count):
            break
    return selected


def _bayesian_candidate_proposals(
    cfg: dict[str, Any],
    archive_df: pd.DataFrame,
    candidate_by_id: dict[str, CandidateLike],
    count: int,
    seed: int,
    optimize_controller: bool,
    start_index: int,
    round_index: int = 0,
) -> list[DesignCandidate]:
    """GP + expected-improvement proposals over the unit-cube design space.

    With up to ~15 design variables and expensive evaluations, Bayesian
    optimization with a Gaussian process is a better fit than a small-budget
    GA. The GP is fit on the continuous soft optimization score.
    """
    if int(count) <= 0:
        return []
    specs = _design_bounds(cfg, optimize_controller)
    lower = np.asarray([spec[2] for spec in specs], dtype=float)
    upper = np.asarray([spec[3] for spec in specs], dtype=float)
    span = np.maximum(upper - lower, 1e-12)
    existing_ids = set(candidate_by_id)

    def _fallback() -> list[DesignCandidate]:
        generated = generate_design_candidates(
            cfg,
            samples=int(count),
            seed=int(seed),
            strategy="sobol",
            optimize_controller=optimize_controller,
            start_index=start_index,
            source="bayesian_fallback",
        )
        return [candidate for candidate in generated if candidate.candidate_id not in existing_ids][: int(count)]

    latest = archive_df.sort_values(["fidelity", "stage"]).drop_duplicates("candidate_id", keep="last") if len(archive_df) else archive_df
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for _, row in latest.iterrows():
        candidate = candidate_by_id.get(str(row.get("candidate_id", "")))
        if not isinstance(candidate, DesignCandidate):
            continue
        score = float(row.get("optimization_score", np.nan))
        if not np.isfinite(score):
            continue
        x_rows.append((_candidate_vector(candidate, specs, cfg) - lower) / span)
        y_rows.append(score)
    if len(x_rows) < max(6, len(specs) + 2):
        return _fallback()

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except Exception:
        return _fallback()

    x_train = np.vstack(x_rows)
    y_train = np.asarray(y_rows, dtype=float)
    # Robust target transform: degenerate candidates can score two orders of
    # magnitude above the baseline, and a plain standardization would let
    # those outliers flatten the GP around the interesting region. log1p of
    # the score above the incumbent keeps the ordering but compresses outliers.
    y_train = np.log1p(np.maximum(y_train - float(np.min(y_train)), 0.0))
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    y_norm = (y_train - y_mean) / y_std
    # Wide length-scale bounds (a nearly flat dimension is legitimate; it just
    # means that the parameter barely matters). sklearn ConvergenceWarnings
    # are harmless here and suppressed.
    kernel = ConstantKernel(1.0, (0.05, 20.0)) * Matern(
        length_scale=np.full(x_train.shape[1], 0.35), length_scale_bounds=(0.02, 50.0), nu=2.5
    ) + WhiteKernel(1e-4, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=int(seed))
    try:
        import warnings as _warnings

        from sklearn.exceptions import ConvergenceWarning as _SkConvergenceWarning

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", category=_SkConvergenceWarning)
            gp.fit(x_train, y_norm)
    except Exception:
        return _fallback()

    bo_cfg = cfg.get("optimization", {}).get("bayesian", {}) or {}
    pool_size = int(bo_cfg.get("pool", 2048))
    pool = generate_design_candidates(
        cfg,
        samples=pool_size,
        seed=int(seed) + 131 * (int(round_index) + 1),
        strategy="sobol",
        optimize_controller=optimize_controller,
        start_index=start_index,
        source="bayesian_pool",
    )
    pool = [candidate for candidate in pool if candidate.candidate_id not in existing_ids]
    if not pool:
        return []
    x_pool = (np.vstack([_candidate_vector(candidate, specs, cfg) for candidate in pool]) - lower) / span
    mu, sigma = gp.predict(x_pool, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    best = float(np.min(y_norm))
    # Expected improvement for minimization.
    from scipy.stats import norm as _norm

    improvement = best - mu
    z = improvement / sigma
    ei = improvement * _norm.cdf(z) + sigma * _norm.pdf(z)
    order = np.argsort(-ei)
    min_dist = float(bo_cfg.get("min_proposal_distance", 0.05))
    selected: list[DesignCandidate] = []
    selected_x: list[np.ndarray] = []
    for idx in order:
        vector = x_pool[int(idx)]
        if selected_x and min(float(np.linalg.norm(vector - other)) for other in selected_x) < min_dist:
            continue
        renamed = _renamed_design_candidate(
            pool[int(idx)],
            f"candidate_bo_r{int(round_index):02d}_{start_index + len(selected):03d}",
            "bayesian",
        )
        selected.append(renamed)
        selected_x.append(vector)
        if len(selected) >= int(count):
            break
    return selected


def _write_best_configs(
    run_dir: Path,
    group_name: str,
    base_cfg: dict[str, Any],
    candidate_by_id: dict[str, CandidateLike],
    best_summary: dict[str, Any],
    search_best_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_candidate = candidate_by_id[str(best_summary["candidate_id"])]
    search_candidate = candidate_by_id[str(search_best_summary["candidate_id"])]
    best_cfg = _apply_candidate_config(base_cfg, best_candidate, base_search="full", serial_evaluation=False)
    search_best_cfg = _apply_candidate_config(base_cfg, search_candidate, base_search="full", serial_evaluation=False)
    write_yaml(run_dir / "best_config.yaml", best_cfg)
    write_yaml(run_dir / "best_search_config.yaml", search_best_cfg)
    best_by_group = run_dir / "best_by_group"
    best_by_group.mkdir(parents=True, exist_ok=True)
    write_yaml(best_by_group / f"{group_name}.yaml", best_cfg)
    write_yaml(run_dir / "best_overall.yaml", best_cfg)
    return best_cfg, search_best_cfg


def _run_genetic_optimization(
    group_name: str,
    task_names: list[str],
    paths: ProjectPaths,
    base_cfg: dict[str, Any],
    baseline_profile: str,
    samples: int,
    workers: int | None,
    seed: int,
    base_search: str,
    validate_top: int,
    search_strategy: str,
    mujoco_validate_top: int,
    mujoco_workers: int | None,
    mujoco_render: bool,
    mujoco_contacts: bool,
    budget_hours: float | None,
    max_evaluations: int | None,
    optimize_controller: bool,
    resume: str | Path | None,
    ga_population: int | None = None,
    ga_generations_quick: int | None = None,
    ga_generations_reduced: int | None = None,
    ga_elite_fraction: float | None = None,
    ga_mutation_sigma: float | None = None,
    ga_crossover_rate: float | None = None,
    ga_local_seeds: bool = True,
    ga_seed_run: str | Path | None = None,
    ga_seed_candidates: list[str] | None = None,
    ga_seed_top: int = 0,
) -> Path:
    started = time.perf_counter()
    slug = _profile_slug(base_cfg, baseline_profile)
    run_dir = Path(resume).resolve() if resume else paths.runs / timestamp_id(f"genetic_mount_optimization_{slug}_{group_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "evaluation_cache"
    deadline = started + float(budget_hours) * 3600.0 if budget_hours else None

    defaults = _genetic_defaults(base_cfg)
    population_size = max(2, int(ga_population or samples or defaults["population"]))
    quick_generations = max(0, int(defaults["quick_generations"] if ga_generations_quick is None else ga_generations_quick))
    reduced_generations = max(0, int(defaults["reduced_generations"] if ga_generations_reduced is None else ga_generations_reduced))
    if quick_generations + reduced_generations <= 0:
        quick_generations = 1
    elite_fraction = float(defaults["elite_fraction"] if ga_elite_fraction is None else ga_elite_fraction)
    mutation_sigma = float(defaults["mutation_sigma"] if ga_mutation_sigma is None else ga_mutation_sigma)
    crossover_rate = float(defaults["crossover_rate"] if ga_crossover_rate is None else ga_crossover_rate)
    local_seed_fraction = float(defaults["local_seed_fraction"] if ga_local_seeds else 0.0)
    random_immigrant_fraction = float(defaults["random_immigrant_fraction"])
    full_fraction = float(defaults["full_fraction"])
    tournament_size = int(defaults["tournament_size"])
    default_eval_budget = population_size * (quick_generations + reduced_generations) + max(
        int(validate_top) + 1,
        int(np.ceil(population_size * np.clip(full_fraction, 0.05, 1.0))),
    )
    max_evaluations = int(max_evaluations or default_eval_budget)

    specs = _design_bounds(base_cfg, optimize_controller)
    imported_seed_genomes = _load_genetic_seed_genomes(
        base_cfg,
        specs,
        seed_run=ga_seed_run,
        seed_candidates=ga_seed_candidates,
        seed_top=int(ga_seed_top or 1),
    )
    genomes = _genetic_initial_genomes(
        base_cfg,
        specs,
        population_size=population_size,
        seed=seed,
        search_strategy=search_strategy if search_strategy not in {"adaptive"} else "hybrid",
        local_seed_fraction=local_seed_fraction,
        seed_genomes=imported_seed_genomes,
    )
    levels = {str(level["name"]): level for level in _fidelity_levels(base_cfg)}
    schedule = [("quick", quick_generations), ("reduced", reduced_generations)]

    candidate_by_id: dict[str, CandidateLike] = {"candidate_000_strict_garmi": _baseline_design_candidate(base_cfg)}
    genome_by_id: dict[str, np.ndarray] = {}
    candidate_cfgs: dict[str, dict[str, Any]] = {}
    all_summaries: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    evaluations_used = 0
    latest_df = pd.DataFrame()
    latest_task_rows: list[dict[str, Any]] = []
    generation_index = 0

    for level_idx, (level_name, generations) in enumerate(schedule):
        if int(generations) <= 0:
            continue
        fidelity = levels.get(level_name, {"name": level_name})
        for local_generation in range(int(generations)):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            eval_budget = max(0, max_evaluations - evaluations_used)
            if eval_budget <= 0:
                break
            stage_label = f"ga_{level_name}_g{generation_index:02d}"
            candidates = _genetic_population_candidates(
                base_cfg,
                specs,
                genomes[: min(len(genomes), eval_budget)],
                generation=generation_index,
                stage_label=level_name,
                source="genetic",
            )
            for candidate, genome in zip(candidates, genomes):
                candidate_by_id[candidate.candidate_id] = candidate
                genome_by_id[candidate.candidate_id] = np.asarray(genome, dtype=float)
            summaries, task_rows, cfgs = _evaluate_mount_candidate_batch(
                candidates,
                base_cfg=base_cfg,
                task_names=task_names,
                paths=paths,
                group_name=group_name,
                workers=workers,
                base_search=str(fidelity.get("base_search", base_search)),
                stage_label=stage_label,
                fidelity=fidelity,
                fidelity_label=level_name,
                cache_dir=cache_dir,
            )
            evaluations_used += len(candidates)
            candidate_cfgs.update(cfgs)
            all_summaries.extend(summaries)
            all_task_rows.extend(task_rows)
            latest_df = _genetic_rank_frame(pd.DataFrame(summaries))
            latest_task_rows = task_rows
            if not latest_df.empty:
                best = latest_df.iloc[0].to_dict()
                generation_rows.append(
                    {
                        "generation": int(generation_index),
                        "stage": stage_label,
                        "fidelity": level_name,
                        "population": int(len(candidates)),
                        "evaluations_used": int(evaluations_used),
                        "best_candidate_id": str(best.get("candidate_id", "")),
                        "best_success_count": int(best.get("success_count", 0)),
                        "best_failed_task_count": int(best.get("failed_task_count", 0)),
                        "best_optimization_score": float(best.get("optimization_score", np.nan)),
                        "best_hard_constraint_violation": float(best.get("hard_constraint_violation", np.nan)),
                    }
                )
                sigma_decay = max(0.35, 1.0 - 0.5 * (local_generation + 1) / max(1, int(generations)))
                stage_sigma = mutation_sigma * (0.72 ** level_idx) * sigma_decay
                genomes = _genetic_next_generation(
                    latest_df,
                    genome_by_id,
                    population_size=population_size,
                    seed=int(seed) + 101 * level_idx,
                    generation=generation_index + 1,
                    elite_fraction=elite_fraction,
                    mutation_sigma=stage_sigma,
                    crossover_rate=crossover_rate,
                    random_immigrant_fraction=random_immigrant_fraction,
                    tournament_size=tournament_size,
                )
            generation_index += 1
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if evaluations_used >= max_evaluations:
            break

    archive_df = _genetic_rank_frame(pd.DataFrame(all_summaries))
    archive_task_df = pd.DataFrame(all_task_rows)
    full_count = max(1, int(validate_top) + 1, int(np.ceil(population_size * np.clip(full_fraction, 0.05, 1.0))))
    final_summaries: list[dict[str, Any]]
    final_task_rows: list[dict[str, Any]]
    if not latest_df.empty and max_evaluations - evaluations_used > 0:
        remaining_full_budget = max(1, max_evaluations - evaluations_used)
        final_ids = _genetic_survivor_ids(latest_df, count=min(full_count, remaining_full_budget), include_strict=True)
        strict_id = "candidate_000_strict_garmi"
        if strict_id in candidate_by_id and strict_id not in final_ids:
            if len(final_ids) >= remaining_full_budget:
                final_ids = final_ids[: max(0, remaining_full_budget - 1)]
            final_ids.append(strict_id)
        final_candidates = [candidate_by_id[candidate_id] for candidate_id in final_ids if candidate_id in candidate_by_id]
        if final_candidates:
            summaries, task_rows, cfgs = _evaluate_mount_candidate_batch(
                final_candidates,
                base_cfg=base_cfg,
                task_names=task_names,
                paths=paths,
                group_name=group_name,
                workers=workers,
                base_search="full",
                stage_label="ga_full_final",
                fidelity=levels.get("full", {"name": "full"}),
                fidelity_label="full",
                cache_dir=cache_dir,
            )
            evaluations_used += len(final_candidates)
            candidate_cfgs.update(cfgs)
            all_summaries.extend(summaries)
            all_task_rows.extend(task_rows)
            final_summaries = summaries
            final_task_rows = task_rows
        else:
            final_summaries = latest_df.to_dict("records")
            final_task_rows = latest_task_rows
    else:
        final_summaries = latest_df.to_dict("records") if not latest_df.empty else archive_df.head(max(1, full_count)).to_dict("records")
        final_task_rows = latest_task_rows

    candidate_df = _genetic_rank_frame(pd.DataFrame(final_summaries))
    task_df = pd.DataFrame(final_task_rows)
    if not task_df.empty and {"candidate_id", "task_name"}.issubset(task_df.columns):
        task_df = task_df.sort_values(["candidate_id", "task_name"]).reset_index(drop=True)
    archive_df = _genetic_rank_frame(pd.DataFrame(all_summaries))
    archive_task_df = pd.DataFrame(all_task_rows)
    pareto_df = _pareto_candidates(candidate_df)
    best_summary = candidate_df.iloc[0].to_dict() if not candidate_df.empty else {}
    search_best_summary = archive_df.iloc[0].to_dict() if not archive_df.empty else best_summary
    best_cfg, search_best_cfg = _write_best_configs(run_dir, group_name, base_cfg, candidate_by_id, best_summary, search_best_summary)
    evaluated_best_cfg = candidate_cfgs.get(str(search_best_summary["candidate_id"]), search_best_cfg)

    candidate_df.to_csv(run_dir / "candidates.csv", index=False)
    candidate_df.to_csv(run_dir / "full_validation_candidates.csv", index=False)
    pareto_df.to_csv(run_dir / "pareto_candidates.csv", index=False)
    task_df.to_csv(run_dir / "candidate_task_metrics.csv", index=False)
    task_df.to_csv(run_dir / "full_validation_task_metrics.csv", index=False)
    archive_df.to_csv(run_dir / "evaluation_archive.csv", index=False)
    archive_df.to_csv(run_dir / "fidelity_history.csv", index=False)
    archive_task_df.to_csv(run_dir / "evaluation_archive_task_metrics.csv", index=False)
    pd.DataFrame(generation_rows).to_csv(run_dir / "genetic_history.csv", index=False)
    write_yaml(run_dir / "best_config_evaluated_search.yaml", evaluated_best_cfg)

    mujoco_validation_df = _validate_mujoco_top_candidates(
        candidate_df,
        candidate_by_id,
        base_cfg,
        task_names,
        group_name,
        paths,
        run_dir,
        count=int(mujoco_validate_top),
        workers=mujoco_workers,
        render=bool(mujoco_render),
        enable_contacts=bool(mujoco_contacts),
    )
    mujoco_best: dict[str, Any] = {}
    if len(mujoco_validation_df):
        mujoco_validation_df.to_csv(run_dir / "mujoco_validation_candidates.csv", index=False)
        mujoco_best = _write_best_mujoco_config(run_dir, base_cfg, candidate_by_id, mujoco_validation_df)

    genetic_state = {
        "enabled": True,
        "population": int(population_size),
        "quick_generations": int(quick_generations),
        "reduced_generations": int(reduced_generations),
        "elite_fraction": float(elite_fraction),
        "mutation_sigma": float(mutation_sigma),
        "crossover_rate": float(crossover_rate),
        "local_seed_fraction": float(local_seed_fraction),
        "random_immigrant_fraction": float(random_immigrant_fraction),
        "full_fraction": float(full_fraction),
        "tournament_size": int(tournament_size),
        "generation_count": int(generation_index),
        "seed_run": str(Path(ga_seed_run).expanduser().resolve()) if ga_seed_run not in {None, ""} else "",
        "seed_candidates": list(ga_seed_candidates or []),
        "seed_top": int(ga_seed_top or 0),
        "imported_seed_count": int(len(imported_seed_genomes)),
    }
    write_json(run_dir / "genetic_state.json", _jsonable(genetic_state))
    write_json(
        run_dir / "optimization_summary.json",
        _jsonable(
            {
                "group_name": group_name,
                "baseline_profile": base_cfg.get("_profile", baseline_profile),
                "optimizer": "genetic",
                "samples": int(samples),
                "population": int(population_size),
                "max_evaluations": int(max_evaluations),
                "evaluations_used": int(evaluations_used),
                "workers": int(_worker_count(workers, max(1, population_size))),
                "seed": int(seed),
                "base_search": str(base_search),
                "validate_top": int(validate_top),
                "search_strategy": str(search_strategy),
                "optimize_controller": bool(optimize_controller),
                "budget_hours": budget_hours,
                "mujoco_validate_top": int(mujoco_validate_top),
                "mujoco_workers": mujoco_workers,
                "mujoco_render": bool(mujoco_render),
                "mujoco_contacts": bool(mujoco_contacts),
                "elapsed_s": float(time.perf_counter() - started),
                "best_source": "genetic_full",
                "search_best_candidate": search_best_summary,
                "best_candidate": best_summary,
                "best_mujoco_candidate": mujoco_best,
                "task_names": task_names,
                "genetic": genetic_state,
                "note": "Genetic Franka/Panda-only co-design optimizer for mirrored GARMI shoulder mounts and holonomic base policy. IK/controller variables are included only when --optimize-controller is set.",
            }
        ),
    )
    _write_optimization_report(
        run_dir,
        group_name,
        candidate_df,
        task_df,
        best_summary,
        candidate_df,
        task_df,
        mujoco_validation_df,
        title="Genetic Mount/Base Optimization",
        description="NSGA-II-style genetic search over mirrored GARMI Panda mounts and holonomic base policy. The strict GARMI baseline is protected as an elite candidate and every final report remains comparable to the existing multi-fidelity optimizer.",
    )
    return run_dir


def _run_multifidelity_optimization(
    group_name: str,
    task_names: list[str],
    paths: ProjectPaths,
    base_cfg: dict[str, Any],
    baseline_profile: str,
    samples: int,
    workers: int | None,
    seed: int,
    base_search: str,
    validate_top: int,
    search_strategy: str,
    mujoco_validate_top: int,
    mujoco_workers: int | None,
    mujoco_render: bool,
    mujoco_contacts: bool,
    optimizer: str,
    budget_hours: float | None,
    max_evaluations: int | None,
    optimize_controller: bool,
    resume: str | Path | None,
) -> Path:
    started = time.perf_counter()
    slug = _profile_slug(base_cfg, baseline_profile)
    run_dir = Path(resume).resolve() if resume else paths.runs / timestamp_id(f"mount_optimization_{slug}_{group_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "evaluation_cache"
    deadline = started + float(budget_hours) * 3600.0 if budget_hours else None
    promote_fraction = float(base_cfg.get("optimization", {}).get("promote_fraction", 0.45))
    minimum_promote = max(2, int(validate_top) + 1)
    if max_evaluations:
        max_evaluations = int(max_evaluations)
    elif optimizer == "bayesian":
        # The default budget covers the initial batch, the BO proposal rounds
        # and the reduced/full promotions; a budget equal to the number of
        # initial samples would end the search before the first BO round.
        bo_cfg_budget = base_cfg.get("optimization", {}).get("bayesian", {}) or {}
        bo_budget = int(bo_cfg_budget.get("rounds", 4)) * int(bo_cfg_budget.get("batch", max(6, int(samples) // 4)))
        max_evaluations = int(samples) + bo_budget + 3 * minimum_promote + max(8, int(samples) // 2)
    else:
        max_evaluations = max(samples, minimum_promote * 3)

    initial_count = max(1, min(int(samples), max_evaluations))
    candidates: list[CandidateLike] = generate_design_candidates(
        base_cfg,
        samples=initial_count,
        seed=seed,
        strategy=search_strategy if search_strategy not in {"adaptive"} else "hybrid",
        optimize_controller=optimize_controller,
    )
    candidate_by_id: dict[str, CandidateLike] = {candidate.candidate_id: candidate for candidate in candidates}
    levels = _fidelity_levels(base_cfg)
    all_summaries: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    final_summaries: list[dict[str, Any]] = []
    final_task_rows: list[dict[str, Any]] = []
    candidate_cfgs: dict[str, dict[str, Any]] = {}
    evaluations_used = 0
    current_candidates = candidates
    surrogate_state: dict[str, Any] = {
        "enabled": optimizer in {"surrogate", "bayesian"},
        "optimizer": optimizer,
        "used": False,
        "proposal_count": 0,
        "fallback": False,
    }
    # Robustness perturbation ensembles (common random numbers across
    # candidates); _level_task_names selects the levels that use them.
    robustness = robustness_config(base_cfg)

    def _level_task_names(label: str) -> list[str]:
        # The perturbation ensemble multiplies the evaluation cost and only
        # matters for the final ranking, so it runs at full fidelity only
        # (quick screens and reduced pre-ranks use the nominal recordings).
        if label != "full" or robustness["ensemble"] <= 0:
            return list(task_names)
        return expand_task_names(task_names, robustness["ensemble"])

    for level_idx, level in enumerate(levels):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        label = str(level["name"])
        eval_budget = max(0, max_evaluations - evaluations_used)
        if eval_budget <= 0:
            break
        current_candidates = current_candidates[:eval_budget]
        summaries, task_rows, cfgs = _evaluate_mount_candidate_batch(
            current_candidates,
            base_cfg=base_cfg,
            task_names=_level_task_names(label),
            paths=paths,
            group_name=group_name,
            workers=workers,
            base_search=str(level.get("base_search", base_search)),
            stage_label=label,
            fidelity=level,
            fidelity_label=label,
            cache_dir=cache_dir,
        )
        evaluations_used += len(current_candidates)
        candidate_cfgs.update(cfgs)
        all_summaries.extend(summaries)
        all_task_rows.extend(task_rows)

        # Iterative Bayesian-optimization rounds at the quick level.
        if optimizer == "bayesian" and label == "quick":
            bo_cfg = base_cfg.get("optimization", {}).get("bayesian", {}) or {}
            bo_rounds = int(bo_cfg.get("rounds", 4))
            bo_batch = int(bo_cfg.get("batch", max(6, initial_count // 4)))
            for round_index in range(bo_rounds):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                remaining = max(0, max_evaluations - evaluations_used)
                if remaining <= 0:
                    break
                proposals = _bayesian_candidate_proposals(
                    base_cfg,
                    pd.DataFrame(all_summaries),
                    candidate_by_id,
                    count=min(bo_batch, remaining),
                    seed=int(seed) + 37 * (round_index + 1),
                    optimize_controller=optimize_controller,
                    start_index=len(candidate_by_id) + 1,
                    round_index=round_index,
                )
                if not proposals:
                    break
                surrogate_state["used"] = True
                surrogate_state["proposal_count"] = int(surrogate_state["proposal_count"]) + len(proposals)
                for candidate in proposals:
                    candidate_by_id[candidate.candidate_id] = candidate
                bo_summaries, bo_task_rows, bo_cfgs = _evaluate_mount_candidate_batch(
                    proposals,
                    base_cfg=base_cfg,
                    task_names=_level_task_names(label),
                    paths=paths,
                    group_name=group_name,
                    workers=workers,
                    base_search=str(level.get("base_search", base_search)),
                    stage_label=f"{label}_bo_r{round_index}",
                    fidelity=level,
                    fidelity_label=label,
                    cache_dir=cache_dir,
                )
                evaluations_used += len(proposals)
                candidate_cfgs.update(bo_cfgs)
                summaries.extend(bo_summaries)
                all_summaries.extend(bo_summaries)
                all_task_rows.extend(bo_task_rows)

        level_df = _rank_candidate_frame(pd.DataFrame(summaries))
        if label == "full" or level_idx == len(levels) - 1:
            final_summaries = summaries
            final_task_rows = task_rows
            break

        promoted = _promoted_candidates(
            level_df,
            candidate_by_id,
            minimum=minimum_promote,
            fraction=promote_fraction,
            maximum=int(base_cfg.get("optimization", {}).get("promote_max", 16)),
        )

        if optimizer == "surrogate" and label == "reduced":
            remaining_budget = max(0, max_evaluations - evaluations_used - len(promoted))
            proposal_count = min(max(0, int(np.ceil(initial_count * 0.25))), remaining_budget)
            proposals = _surrogate_candidate_proposals(
                base_cfg,
                pd.DataFrame(all_summaries),
                candidate_by_id,
                count=proposal_count,
                seed=int(seed) + 6001,
                optimize_controller=optimize_controller,
                start_index=len(candidate_by_id) + 1,
            )
            if proposals:
                surrogate_state["used"] = True
                surrogate_state["proposal_count"] = len(proposals)
                for candidate in proposals:
                    candidate_by_id[candidate.candidate_id] = candidate
                promoted.extend(proposals)
        current_candidates = promoted

    if not final_summaries:
        latest_df = _rank_candidate_frame(pd.DataFrame(all_summaries))
        final_ids = _candidate_ids_from_frame(latest_df, count=max(1, minimum_promote), include_strict=True)
        final_candidates = [candidate_by_id[candidate_id] for candidate_id in final_ids if candidate_id in candidate_by_id]
        summaries, task_rows, cfgs = _evaluate_mount_candidate_batch(
            final_candidates,
            base_cfg=base_cfg,
            task_names=_level_task_names("full"),
            paths=paths,
            group_name=group_name,
            workers=workers,
            base_search="full",
            stage_label="full",
            fidelity=levels[-1],
            fidelity_label="full",
            cache_dir=cache_dir,
        )
        final_summaries = summaries
        final_task_rows = task_rows
        all_summaries.extend(summaries)
        all_task_rows.extend(task_rows)
        candidate_cfgs.update(cfgs)

    candidate_df = _rank_candidate_frame(pd.DataFrame(final_summaries))
    task_df = pd.DataFrame(final_task_rows).sort_values(["candidate_id", "task_name"]).reset_index(drop=True)
    archive_df = _rank_candidate_frame(pd.DataFrame(all_summaries))
    archive_task_df = pd.DataFrame(all_task_rows)
    pareto_df = _pareto_candidates(candidate_df)
    best_summary = candidate_df.iloc[0].to_dict()
    search_best_summary = _rank_candidate_frame(archive_df).iloc[0].to_dict() if len(archive_df) else best_summary
    best_cfg, search_best_cfg = _write_best_configs(run_dir, group_name, base_cfg, candidate_by_id, best_summary, search_best_summary)
    evaluated_best_cfg = candidate_cfgs.get(str(search_best_summary["candidate_id"]), search_best_cfg)

    candidate_df.to_csv(run_dir / "candidates.csv", index=False)
    candidate_df.to_csv(run_dir / "full_validation_candidates.csv", index=False)
    pareto_df.to_csv(run_dir / "pareto_candidates.csv", index=False)
    task_df.to_csv(run_dir / "candidate_task_metrics.csv", index=False)
    task_df.to_csv(run_dir / "full_validation_task_metrics.csv", index=False)
    archive_df.to_csv(run_dir / "evaluation_archive.csv", index=False)
    archive_df.to_csv(run_dir / "fidelity_history.csv", index=False)
    archive_task_df.to_csv(run_dir / "evaluation_archive_task_metrics.csv", index=False)
    write_yaml(run_dir / "best_config_evaluated_search.yaml", evaluated_best_cfg)
    write_json(run_dir / "surrogate_state.json", _jsonable(surrogate_state))
    # Diagnostic log: which candidates would strict screening guards have
    # pruned, and at which fidelity?
    guard_cols = [
        col
        for col in [
            "candidate_id",
            "fidelity",
            "stage",
            "screening_guard_violation",
            "screening_guard_enforced",
            "success_fraction",
            "mean_soft_loss",
            "optimization_score",
        ]
        if col in archive_df
    ]
    if guard_cols:
        archive_df[guard_cols].to_csv(run_dir / "guard_log.csv", index=False)

    mujoco_validation_df = _validate_mujoco_top_candidates(
        candidate_df,
        candidate_by_id,
        base_cfg,
        task_names,
        group_name,
        paths,
        run_dir,
        count=int(mujoco_validate_top),
        workers=mujoco_workers,
        render=bool(mujoco_render),
        enable_contacts=bool(mujoco_contacts),
    )
    mujoco_best: dict[str, Any] = {}
    if len(mujoco_validation_df):
        mujoco_validation_df.to_csv(run_dir / "mujoco_validation_candidates.csv", index=False)
        mujoco_best = _write_best_mujoco_config(run_dir, base_cfg, candidate_by_id, mujoco_validation_df)

    write_json(
        run_dir / "optimization_summary.json",
        _jsonable(
            {
                "group_name": group_name,
                "baseline_profile": base_cfg.get("_profile", baseline_profile),
                "optimizer": optimizer,
                "samples": int(initial_count),
                "max_evaluations": int(max_evaluations),
                "evaluations_used": int(evaluations_used),
                "workers": int(_worker_count(workers, max(1, initial_count))),
                "seed": int(seed),
                "base_search": str(base_search),
                "validate_top": int(validate_top),
                "search_strategy": str(search_strategy),
                "optimize_controller": bool(optimize_controller),
                "budget_hours": budget_hours,
                "mujoco_validate_top": int(mujoco_validate_top),
                "mujoco_workers": mujoco_workers,
                "mujoco_render": bool(mujoco_render),
                "mujoco_contacts": bool(mujoco_contacts),
                "elapsed_s": float(time.perf_counter() - started),
                "best_source": "multifidelity_full",
                "search_best_candidate": search_best_summary,
                "best_candidate": best_summary,
                "best_mujoco_candidate": mujoco_best,
                "task_names": task_names,
                "robustness": robustness,
                "surrogate": surrogate_state,
                "note": "Franka/Panda-only multi-fidelity co-design optimizer for mirrored GARMI shoulder mounts, holonomic base policy, IK and impedance controller parameters.",
            }
        ),
    )
    _write_optimization_report(
        run_dir,
        group_name,
        candidate_df,
        task_df,
        best_summary,
        candidate_df,
        task_df,
        mujoco_validation_df,
        title="Multi-Fidelity Mount/Controller Optimization",
        description="Task-driven Franka/Panda-only co-design run. The strict GARMI baseline is always included; candidates can include mirrored mount, holonomic base, IK and impedance-controller variables.",
    )
    return run_dir


def run_mount_optimization(
    group_name: str,
    task_names: list[str],
    paths: ProjectPaths = PATHS,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
    samples: int = 24,
    workers: int | None = None,
    seed: int = 7,
    base_search: str = "reduced",
    validate_top: int = 3,
    search_strategy: str = "adaptive",
    mujoco_validate_top: int = 0,
    mujoco_workers: int | None = None,
    mujoco_render: bool = False,
    mujoco_contacts: bool = False,
    optimizer: str = "adaptive",
    budget_hours: float | None = None,
    max_evaluations: int | None = None,
    optimize_controller: bool = False,
    resume: str | Path | None = None,
    ga_population: int | None = None,
    ga_generations_quick: int | None = None,
    ga_generations_reduced: int | None = None,
    ga_elite_fraction: float | None = None,
    ga_mutation_sigma: float | None = None,
    ga_crossover_rate: float | None = None,
    ga_local_seeds: bool = True,
    ga_seed_run: str | Path | None = None,
    ga_seed_candidates: list[str] | None = None,
    ga_seed_top: int = 0,
) -> Path:
    ensure_dirs(paths)
    base_cfg = cfg or load_baseline_config(paths, profile=baseline_profile)
    # Explicit per-group task weights reach the evaluator through the config.
    try:
        base_cfg["_task_weights"] = group_task_weights(group_name, paths)
    except Exception:
        base_cfg["_task_weights"] = {}
    optimizer = str(optimizer or search_strategy).lower()
    if optimizer == "genetic":
        return _run_genetic_optimization(
            group_name=group_name,
            task_names=task_names,
            paths=paths,
            base_cfg=base_cfg,
            baseline_profile=baseline_profile,
            samples=samples,
            workers=workers,
            seed=seed,
            base_search=base_search,
            validate_top=validate_top,
            search_strategy=search_strategy,
            mujoco_validate_top=mujoco_validate_top,
            mujoco_workers=mujoco_workers,
            mujoco_render=mujoco_render,
            mujoco_contacts=mujoco_contacts,
            budget_hours=budget_hours,
            max_evaluations=max_evaluations,
            optimize_controller=optimize_controller,
            resume=resume,
            ga_population=ga_population,
            ga_generations_quick=ga_generations_quick,
            ga_generations_reduced=ga_generations_reduced,
            ga_elite_fraction=ga_elite_fraction,
            ga_mutation_sigma=ga_mutation_sigma,
            ga_crossover_rate=ga_crossover_rate,
            ga_local_seeds=ga_local_seeds,
            ga_seed_run=ga_seed_run,
            ga_seed_candidates=ga_seed_candidates,
            ga_seed_top=ga_seed_top,
        )
    if optimizer in {"multifidelity", "surrogate", "bayesian"}:
        return _run_multifidelity_optimization(
            group_name=group_name,
            task_names=task_names,
            paths=paths,
            base_cfg=base_cfg,
            baseline_profile=baseline_profile,
            samples=samples,
            workers=workers,
            seed=seed,
            base_search=base_search,
            validate_top=validate_top,
            search_strategy=search_strategy,
            mujoco_validate_top=mujoco_validate_top,
            mujoco_workers=mujoco_workers,
            mujoco_render=mujoco_render,
            mujoco_contacts=mujoco_contacts,
            optimizer=optimizer,
            budget_hours=budget_hours,
            max_evaluations=max_evaluations,
            optimize_controller=optimize_controller,
            resume=resume,
        )
    search_strategy = str(search_strategy).lower()
    if search_strategy == "adaptive" and int(samples) > 8:
        initial_count = max(6, int(np.ceil(int(samples) * 0.65)))
        initial_count = min(initial_count, int(samples))
        candidates = generate_mount_candidates(base_cfg, samples=initial_count, seed=seed, strategy="hybrid")
    else:
        initial_count = int(samples)
        candidates = generate_mount_candidates(base_cfg, samples=samples, seed=seed, strategy=search_strategy)
    worker_count = _worker_count(workers, len(candidates))
    slug = _profile_slug(base_cfg, baseline_profile)
    run_dir = paths.runs / timestamp_id(f"mount_optimization_{slug}_{group_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    summaries, task_rows, candidate_cfgs = _evaluate_mount_candidate_batch(
        candidates,
        base_cfg=base_cfg,
        task_names=task_names,
        paths=paths,
        group_name=group_name,
        workers=workers,
        base_search=base_search,
        stage_label="initial",
    )
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if search_strategy == "adaptive" and int(samples) > len(candidates):
        interim_df = _rank_candidate_frame(pd.DataFrame(summaries))
        refined = generate_refined_mount_candidates(
            base_cfg,
            interim_df,
            candidate_by_id,
            count=int(samples) - len(candidates),
            seed=int(seed) + 4099,
            start_index=len(candidates),
        )
        if refined:
            refined_summaries, refined_rows, refined_cfgs = _evaluate_mount_candidate_batch(
                refined,
                base_cfg=base_cfg,
                task_names=task_names,
                paths=paths,
                group_name=group_name,
                workers=workers,
                base_search=base_search,
                stage_label="refine",
            )
            candidates.extend(refined)
            summaries.extend(refined_summaries)
            task_rows.extend(refined_rows)
            candidate_cfgs.update(refined_cfgs)
            candidate_by_id.update({candidate.candidate_id: candidate for candidate in refined})

    candidate_df = _rank_candidate_frame(pd.DataFrame(summaries))
    task_df = pd.DataFrame(task_rows).sort_values(["candidate_id", "task_name"]).reset_index(drop=True)
    search_best_summary = candidate_df.iloc[0].to_dict()
    search_best_candidate_id = str(search_best_summary["candidate_id"])
    search_best_cfg = _apply_candidate_config(base_cfg, candidate_by_id[search_best_candidate_id], base_search="full")
    evaluated_best_cfg = candidate_cfgs[search_best_candidate_id]

    validation_df: pd.DataFrame | None = None
    validation_task_df: pd.DataFrame | None = None
    if int(validate_top) > 0:
        validation_df, validation_task_df, best_summary, best_candidate = _validate_top_candidates_full(
            candidate_df,
            candidate_by_id,
            base_cfg,
            task_names,
            group_name,
            paths,
            validate_top,
        )
        best_cfg = _apply_candidate_config(base_cfg, best_candidate, base_search="full", serial_evaluation=False)
        best_source = "full_validation"
    else:
        best_summary = search_best_summary
        best_cfg = search_best_cfg
        best_source = "search_only"

    candidate_df.to_csv(run_dir / "candidates.csv", index=False)
    task_df.to_csv(run_dir / "candidate_task_metrics.csv", index=False)
    write_yaml(run_dir / "best_search_config.yaml", search_best_cfg)
    write_yaml(run_dir / "best_config.yaml", best_cfg)
    write_yaml(run_dir / "best_config_evaluated_search.yaml", evaluated_best_cfg)
    if validation_df is not None and validation_task_df is not None:
        validation_df.to_csv(run_dir / "full_validation_candidates.csv", index=False)
        validation_task_df.to_csv(run_dir / "full_validation_task_metrics.csv", index=False)
    mujoco_validation_df = _validate_mujoco_top_candidates(
        validation_df if validation_df is not None else candidate_df,
        candidate_by_id,
        base_cfg,
        task_names,
        group_name,
        paths,
        run_dir,
        count=int(mujoco_validate_top),
        workers=mujoco_workers,
        render=bool(mujoco_render),
        enable_contacts=bool(mujoco_contacts),
    )
    mujoco_best: dict[str, Any] = {}
    if len(mujoco_validation_df):
        mujoco_validation_df.to_csv(run_dir / "mujoco_validation_candidates.csv", index=False)
        mujoco_best = _write_best_mujoco_config(run_dir, base_cfg, candidate_by_id, mujoco_validation_df)
    write_json(
        run_dir / "optimization_summary.json",
        _jsonable({
            "group_name": group_name,
            "baseline_profile": base_cfg.get("_profile", baseline_profile),
            "samples": int(len(candidates)),
            "workers": int(worker_count),
            "seed": int(seed),
            "base_search": str(base_search),
            "validate_top": int(validate_top),
            "search_strategy": str(search_strategy),
            "initial_candidate_count": int(initial_count),
            "mujoco_validate_top": int(mujoco_validate_top),
            "mujoco_workers": mujoco_workers,
            "mujoco_render": bool(mujoco_render),
            "mujoco_contacts": bool(mujoco_contacts),
            "elapsed_s": float(time.perf_counter() - started),
            "best_source": best_source,
            "search_best_candidate": search_best_summary,
            "best_candidate": best_summary,
            "best_mujoco_candidate": mujoco_best,
            "task_names": task_names,
            "note": "Franka/Panda-only optimizer keeps original arm kinematics fixed and optimizes strictly mirrored GARMI shoulder mount pose.",
        }),
    )
    _write_optimization_report(run_dir, group_name, candidate_df, task_df, best_summary, validation_df, validation_task_df, mujoco_validation_df)
    return run_dir
