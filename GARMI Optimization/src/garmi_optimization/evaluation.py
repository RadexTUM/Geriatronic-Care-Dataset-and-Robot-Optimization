"""Kinematic coverage of a configuration on recorded trials (paper Sec. IV-C).

For every trial the evaluator places the base, solves damped-least-squares
inverse kinematics along the wrist trajectories, and checks the coverage
criteria of the profile (position and task-frame orientation errors,
manipulability, joint-speed and torque utilization with the measured
wrenches as external loads, time scaling, collision clearance). It also
writes the baseline reports (``garmi-opt baseline``).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from garmi_optimization.collision import arm_body_clearance, inter_arm_clearance
from garmi_optimization.config import (
    DEFAULT_PROFILE,
    PATHS,
    ProjectPaths,
    ensure_dirs,
    load_baseline_config,
    load_group_config,
    load_project_config,
    timestamp_id,
    write_json,
)
from garmi_optimization.math_utils import finite_gradient, moving_average, rotation_z, wrap_angle
from garmi_optimization.panda import (
    PandaModel,
    end_effector,
    end_effector_rotation,
    forward_points,
    manipulability,
    pose_jacobian,
    position_jacobian,
    solve_position_ik,
    solve_pose_ik,
)
from garmi_optimization.perturbations import robustness_summary, split_variant_name
from garmi_optimization.task_frames import TaskFrameSpec, anchor_offset, spec_for_task
from garmi_optimization.task_frames import orientation_error as frame_orientation_error
from garmi_optimization.tasks import TaskData, global_task_names, group_task_names, load_tasks


BASE_MOTION_MODES = {
    "a": "fixed_world",
    "mode_a": "fixed_world",
    "fixed_world": "fixed_world",
    "fixed": "fixed_world",
    "b": "fixed_initial",
    "mode_b": "fixed_initial",
    "fixed_initial": "fixed_initial",
    "initial": "fixed_initial",
    "initial_only": "fixed_initial",
    "c": "limited_holonomic",
    "mode_c": "limited_holonomic",
    "limited": "limited_holonomic",
    "limited_holonomic": "limited_holonomic",
    "d": "nonholonomic",
    "mode_d": "nonholonomic",
    "nonholonomic": "nonholonomic",
    "non_holonomic": "nonholonomic",
    "e": "free_holonomic",
    "mode_e": "free_holonomic",
    "free": "free_holonomic",
    "holonomic": "free_holonomic",
    "free_holonomic": "free_holonomic",
}


def normalize_base_motion_mode(mode: Any) -> str:
    text = str(mode or "free_holonomic").strip().lower().replace("-", "_")
    if text not in BASE_MOTION_MODES:
        choices = ", ".join(["A", "B", "C", "D", "E", "fixed_world", "fixed_initial", "limited_holonomic", "nonholonomic", "free_holonomic"])
        raise ValueError(f"Unknown base motion mode '{mode}'. Choose one of: {choices}.")
    return BASE_MOTION_MODES[text]


def set_base_motion_mode(cfg: dict[str, Any], mode: Any) -> dict[str, Any]:
    cfg.setdefault("base", {})["motion_mode"] = normalize_base_motion_mode(mode)
    return cfg


def _float_list(value: Any, fallback: list[float]) -> list[float]:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def base_variants(cfg: dict[str, Any]) -> list[dict[str, float]]:
    base = cfg["base"]
    fractions = _float_list(base.get("preferred_reach_fraction_candidates"), [0.72])
    forward = _float_list(base.get("forward_offset_candidates_m"), [0.0])
    lateral = _float_list(base.get("lateral_offset_candidates_m"), [0.0])
    yaw = _float_list(base.get("yaw_offset_candidates_rad"), [0.0])
    variants = []
    for fraction in fractions:
        for fwd in forward:
            for lat in lateral:
                for yaw_offset in yaw:
                    variants.append(
                        {
                            "preferred_reach_fraction": float(fraction),
                            "forward_offset_m": float(fwd),
                            "lateral_offset_m": float(lat),
                            "yaw_offset_rad": float(yaw_offset),
                        }
                    )
    return variants


def _reference_base_pose(task: TaskData, model: PandaModel, variant: dict[str, float]) -> np.ndarray:
    preferred_reach = float(variant["preferred_reach_fraction"]) * model.reach_m
    active_sides = task.active_sides()
    yaw = np.zeros(task.sample_count, dtype=float)

    base_xy = np.empty((task.sample_count, 2), dtype=float)
    if len(active_sides) == 1:
        side = active_sides[0]
        target = task.side_position(side)
        shoulder_y = float(model.shoulder_local(side)[1])
        yaw += float(variant.get("yaw_offset_rad", 0.0))
        for idx in range(task.sample_count):
            offset = rotation_z(float(yaw[idx])) @ np.array(
                [
                    preferred_reach + float(variant.get("forward_offset_m", 0.0)),
                    shoulder_y + float(variant.get("lateral_offset_m", 0.0)),
                    0.0,
                ],
                dtype=float,
            )
            base_xy[idx] = target[idx, :2] - offset[:2]
    else:
        left = task.left_pos
        right = task.right_pos
        pair = left[:, :2] - right[:, :2]
        midpoint = 0.5 * (left[:, :2] + right[:, :2])
        for idx, vec in enumerate(pair):
            if np.linalg.norm(vec) < 1e-6:
                yaw[idx] = yaw[idx - 1] if idx else 0.0
            else:
                yaw[idx] = np.arctan2(-vec[0], vec[1])
        for idx in range(1, len(yaw)):
            yaw[idx] = yaw[idx - 1] + wrap_angle(float(yaw[idx] - yaw[idx - 1]))
        yaw += float(variant.get("yaw_offset_rad", 0.0))

        for idx in range(task.sample_count):
            offset = rotation_z(float(yaw[idx])) @ np.array(
                [
                    preferred_reach + float(variant.get("forward_offset_m", 0.0)),
                    float(variant.get("lateral_offset_m", 0.0)),
                    0.0,
                ],
                dtype=float,
            )
            base_xy[idx] = midpoint[idx] - offset[:2]
    return np.column_stack([base_xy, yaw])


def _smooth_pose(pose: np.ndarray, window: int) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if len(pose) == 0:
        return pose.copy()
    base_xy = moving_average(pose[:, :2], int(window))
    yaw = moving_average(pose[:, 2:3], int(window))[:, 0]
    return np.column_stack([base_xy, yaw])


def _pose_motion_metrics(task: TaskData, pose: np.ndarray) -> dict[str, float]:
    pose = np.asarray(pose, dtype=float)
    base_xy = pose[:, :2]
    dt = np.gradient(task.time) if len(task.time) > 1 else np.ones(task.sample_count)
    xy_vel = np.gradient(base_xy, axis=0) / np.maximum(dt[:, None], 1e-6)
    xy_acc = np.gradient(xy_vel, axis=0) / np.maximum(dt[:, None], 1e-6)
    yaw = pose[:, 2]
    yaw_rate = np.gradient(yaw) / np.maximum(dt, 1e-6) if len(yaw) else np.zeros(0)
    return {
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(base_xy, axis=0), axis=1))) if len(base_xy) > 1 else 0.0,
        "max_speed_mps": float(np.max(np.linalg.norm(xy_vel, axis=1))) if len(xy_vel) else 0.0,
        "max_acc_mps2": float(np.max(np.linalg.norm(xy_acc, axis=1))) if len(xy_acc) else 0.0,
        "max_yaw_rate_radps": float(np.max(np.abs(yaw_rate))) if len(yaw_rate) else 0.0,
    }


def _fixed_world_pose(task: TaskData, cfg: dict[str, Any], variant: dict[str, float]) -> np.ndarray:
    base_cfg = cfg.get("base", {})
    fixed = np.asarray(base_cfg.get("fixed_pose", [0.0, 0.0, 0.0]), dtype=float)
    if fixed.shape != (3,):
        fixed = np.array([0.0, 0.0, float(variant.get("yaw_offset_rad", 0.0))], dtype=float)
    fixed = fixed.copy()
    fixed[2] += float(variant.get("yaw_offset_rad", 0.0))
    return np.repeat(fixed[None, :], task.sample_count, axis=0)


def _fixed_initial_pose(reference_pose: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(reference_pose[0], dtype=float)[None, :], len(reference_pose), axis=0)


def _limited_holonomic_pose(task: TaskData, reference_pose: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    base_cfg = cfg.get("base", {})
    max_path = float(base_cfg.get("limited_max_path_m", 0.75))
    max_speed = float(base_cfg.get("limited_max_speed_mps", min(float(base_cfg.get("max_speed_mps", 0.85)), 0.20)))
    max_yaw_rate = float(base_cfg.get("limited_max_yaw_rate_radps", 0.35))
    pose = np.asarray(reference_pose, dtype=float).copy()
    if len(pose) <= 1:
        return pose
    out = np.empty_like(pose)
    out[0] = pose[0]
    used_path = 0.0
    dt = np.gradient(task.time) if len(task.time) > 1 else np.ones(task.sample_count)
    for idx in range(1, len(pose)):
        remaining = max(0.0, max_path - used_path)
        step_limit = max(0.0, min(max_speed * float(max(dt[idx], 1e-6)), remaining))
        delta = pose[idx, :2] - out[idx - 1, :2]
        distance = float(np.linalg.norm(delta))
        if distance > step_limit > 0.0:
            delta = delta / distance * step_limit
            distance = step_limit
        elif step_limit <= 0.0:
            delta = np.zeros(2, dtype=float)
            distance = 0.0
        out[idx, :2] = out[idx - 1, :2] + delta
        used_path += distance
        yaw_delta = wrap_angle(float(pose[idx, 2] - out[idx - 1, 2]))
        yaw_limit = max_yaw_rate * float(max(dt[idx], 1e-6))
        out[idx, 2] = out[idx - 1, 2] + float(np.clip(yaw_delta, -yaw_limit, yaw_limit))
    return out


def _nonholonomic_pose(task: TaskData, reference_pose: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    base_cfg = cfg.get("base", {})
    max_speed = float(base_cfg.get("nonholonomic_max_speed_mps", base_cfg.get("max_speed_mps", 0.85)))
    max_yaw_rate = float(base_cfg.get("nonholonomic_max_yaw_rate_radps", 0.75))
    pose = np.asarray(reference_pose, dtype=float).copy()
    if len(pose) <= 1:
        return pose
    out = np.empty_like(pose)
    out[0] = pose[0]
    dt = np.gradient(task.time) if len(task.time) > 1 else np.ones(task.sample_count)
    for idx in range(1, len(pose)):
        step_dt = float(max(dt[idx], 1e-6))
        yaw_delta = wrap_angle(float(pose[idx, 2] - out[idx - 1, 2]))
        yaw_limit = max_yaw_rate * step_dt
        out[idx, 2] = out[idx - 1, 2] + float(np.clip(yaw_delta, -yaw_limit, yaw_limit))
        heading = np.array([np.cos(out[idx, 2]), np.sin(out[idx, 2])], dtype=float)
        target_delta = pose[idx, :2] - out[idx - 1, :2]
        forward_step = float(np.dot(target_delta, heading))
        step_limit = max_speed * step_dt
        forward_step = float(np.clip(forward_step, -step_limit, step_limit))
        out[idx, :2] = out[idx - 1, :2] + forward_step * heading
    return out


def estimate_base_motion(task: TaskData, model: PandaModel, cfg: dict[str, Any], variant: dict[str, float]) -> dict[str, Any]:
    mode = normalize_base_motion_mode(cfg.get("base", {}).get("motion_mode", "free_holonomic"))
    reference_pose = _smooth_pose(_reference_base_pose(task, model, variant), int(cfg["base"].get("smoothing_window", 7)))
    if mode == "fixed_world":
        pose = _fixed_world_pose(task, cfg, variant)
    elif mode == "fixed_initial":
        pose = _fixed_initial_pose(reference_pose)
    elif mode == "limited_holonomic":
        pose = _limited_holonomic_pose(task, reference_pose, cfg)
    elif mode == "nonholonomic":
        pose = _nonholonomic_pose(task, reference_pose, cfg)
    elif mode == "free_holonomic":
        pose = reference_pose
    else:
        raise ValueError(f"Unsupported base motion mode: {mode}")
    metrics = _pose_motion_metrics(task, pose)
    return {
        "pose": pose,
        "base_motion_mode": mode,
        **metrics,
        **variant,
    }


def _trajectory_torque_utilization(
    model: PandaModel,
    task: TaskData,
    q_traj: np.ndarray,
    tau_ext: np.ndarray,
    side: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Velocity/torque estimates with IK-discontinuity masking and time scaling.

    1. IK discontinuities (elbow flips between neighboring samples) produce
       finite-difference velocities and inverse-dynamics torques that are off
       by orders of magnitude. Samples adjacent to a joint jump larger than
       ik.discontinuity_jump_rad are excluded from the dynamics estimate; the
       jump itself is reported as ik_discontinuity_fraction and penalized
       separately.
    2. The robot is not required to execute a task at recorded human speed.
       If the trajectory exceeds joint speed limits, playback is virtually
       slowed by a factor of up to success.max_time_scale (velocities /s,
       accelerations /s^2) and the used factor is reported as time_scale.
    """
    cfg = cfg or {}
    time = np.asarray(task.time, dtype=float)
    qdot = finite_gradient(q_traj, time)
    max_time_scale = float(cfg.get("success", {}).get("max_time_scale", 1.5))
    # A step only counts as an IK discontinuity if the joint could not have
    # covered it physically within the sample interval, i.e. faster than
    # velocity_limit * dt * max_time_scale * margin. A plain absolute
    # threshold would misclassify legitimate fast motion as a jump whenever
    # the sampling is coarse (50 samples over ~15 s -> dt ~ 0.3 s).
    jump_floor = float(cfg.get("ik", {}).get("discontinuity_jump_rad", 0.6))
    jump_margin = float(cfg.get("ik", {}).get("discontinuity_speed_margin", 1.35))
    sample_mask = np.zeros(len(q_traj), dtype=bool)
    if len(q_traj) > 1 and jump_floor > 0:
        steps = np.abs(np.diff(np.asarray(q_traj, dtype=float), axis=0))
        dt = np.maximum(np.diff(time), 1e-9)
        physical_max = model.velocity_limits_radps[None, :] * dt[:, None] * max_time_scale * jump_margin
        jump_intervals = np.any(steps > np.maximum(physical_max, jump_floor), axis=1)
        sample_mask[:-1] |= jump_intervals
        sample_mask[1:] |= jump_intervals
    discontinuity_fraction = float(np.mean(sample_mask)) if len(sample_mask) else 0.0
    qdot_masked = qdot.copy()
    qdot_masked[sample_mask] = 0.0
    raw_speed_util = np.max(np.abs(qdot_masked) / np.maximum(model.velocity_limits_radps[None, :], 1e-9)) if len(qdot_masked) else 0.0
    time_scale = float(np.clip(raw_speed_util, 1.0, max(1.0, max_time_scale)))

    qdot_eff = qdot_masked / time_scale
    qddot_eff = finite_gradient(qdot_masked, time) / (time_scale**2)
    # The acceleration estimate needs one extra sample of margin: an unmasked
    # sample next to a zeroed one would otherwise see an artificial spike.
    accel_mask = sample_mask.copy()
    if len(accel_mask) > 1:
        accel_mask[:-1] |= sample_mask[1:]
        accel_mask[1:] |= sample_mask[:-1]
    qddot_eff[accel_mask] = 0.0
    tau_dyn = np.asarray(
        [model.inverse_dynamics(q, qd, qdd, side) for q, qd, qdd in zip(q_traj, qdot_eff, qddot_eff)],
        dtype=float,
    )
    tau_total = tau_dyn + tau_ext
    util = tau_total / np.maximum(model.torque_limits_nm[None, :], 1e-9)
    return np.abs(util), qdot_eff, tau_dyn, tau_total, time_scale, discontinuity_fraction


def _pose_error_values(
    model: PandaModel,
    q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    frame_spec: TaskFrameSpec | None = None,
) -> tuple[float, float]:
    from garmi_optimization.math_utils import rotation_error_vector

    pos_error = float(np.linalg.norm(np.asarray(target_position, dtype=float) - end_effector(model, q, side, base_pose)))
    ee_rotation = end_effector_rotation(model, q, side, base_pose)
    if frame_spec is not None:
        rot_error = float(frame_orientation_error(frame_spec, target_rotation, ee_rotation))
    else:
        rot_error = float(np.linalg.norm(rotation_error_vector(target_rotation, ee_rotation)))
    return pos_error, rot_error


def _task_frame_spec(cfg: dict[str, Any], task: TaskData) -> TaskFrameSpec:
    base_name, _variant = split_variant_name(task.name)
    return spec_for_task(cfg, base_name)


def _cfg_for_frame_spec(cfg: dict[str, Any], spec: TaskFrameSpec) -> dict[str, Any]:
    """Return a config copy with position-only IK for orientation-free tasks."""
    if spec.orientation_mode != "free":
        return cfg
    local = dict(cfg)
    ik = dict(local.get("ik", {}))
    ik["use_orientation"] = False
    ik["orientation_weight"] = 0.0
    local["ik"] = ik
    return local


def _ik_seed_set(model: PandaModel, side: str) -> np.ndarray:
    if hasattr(model, "ik_seed_set"):
        seeds = np.asarray(model.ik_seed_set(side), dtype=float)
    else:
        seeds = np.asarray([model.q_nominal], dtype=float)
    if seeds.ndim != 2 or seeds.shape[1] != model.dof:
        seeds = np.asarray([model.q_nominal], dtype=float)
    return np.asarray([np.clip(seed, model.q_lower, model.q_upper) for seed in seeds], dtype=float)


def _pose_solution_score(q: np.ndarray, previous_q: np.ndarray, pos_error: float, orientation_error: float, cfg: dict[str, Any]) -> float:
    continuity = float(np.linalg.norm(np.asarray(q, dtype=float) - np.asarray(previous_q, dtype=float))) / max(np.sqrt(len(q)), 1.0)
    return (
        float(pos_error)
        + float(cfg["ik"].get("serial_retry_orientation_weight", 0.045)) * float(orientation_error)
        + float(cfg["ik"].get("serial_retry_continuity_weight", 0.012)) * continuity
    )


def _solve_position_ik_from_seeds(
    model: PandaModel,
    target_position: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    previous_q: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, float]:
    best_q, best_error = solve_position_ik(model, target_position, side, base_pose, previous_q, cfg["ik"])
    best_score = float(best_error)
    for seed in _ik_seed_set(model, side)[1 : int(cfg["ik"].get("serial_initial_seed_count", 8))]:
        q_try, error_try = solve_position_ik(model, target_position, side, base_pose, seed, cfg["ik"])
        score = float(error_try) + 0.01 * float(np.linalg.norm(q_try - previous_q)) / max(np.sqrt(model.dof), 1.0)
        if score < best_score:
            best_q = q_try
            best_error = float(error_try)
            best_score = score
    return best_q, best_error


def _solve_pose_ik_with_retries(
    model: PandaModel,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    previous_q: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, float, float]:
    q_best, pos_best, rot_best = solve_pose_ik(model, target_position, target_rotation, side, base_pose, previous_q, cfg["ik"])
    best_score = _pose_solution_score(q_best, previous_q, pos_best, rot_best, cfg)
    if not hasattr(model, "ik_seed_set"):
        return q_best, pos_best, rot_best

    retry_position = float(cfg["ik"].get("serial_retry_position_error_m", 0.055))
    retry_orientation = float(cfg["ik"].get("serial_retry_orientation_error_rad", 0.90))
    if pos_best <= retry_position and rot_best <= retry_orientation:
        return q_best, pos_best, rot_best

    seed_count = int(cfg["ik"].get("serial_retry_seed_count", 7))
    for seed in _ik_seed_set(model, side)[1:seed_count]:
        q_try, pos_try, rot_try = solve_pose_ik(model, target_position, target_rotation, side, base_pose, seed, cfg["ik"])
        score = _pose_solution_score(q_try, previous_q, pos_try, rot_try, cfg)
        if score < best_score:
            q_best = q_try
            pos_best = float(pos_try)
            rot_best = float(rot_try)
            best_score = score
    return q_best, pos_best, rot_best


def _weighted_pose_jacobian(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    jac_pos, jac_rot = pose_jacobian(model, q, side, base_pose)
    return np.vstack([jac_pos, float(cfg["ik"].get("orientation_weight", 0.08)) * jac_rot])


def _nullspace_projector(jac: np.ndarray, damping: float) -> np.ndarray:
    lhs = jac @ jac.T + float(damping * damping) * np.eye(jac.shape[0])
    try:
        pinv = jac.T @ np.linalg.solve(lhs, np.eye(lhs.shape[0]))
    except np.linalg.LinAlgError:
        pinv = jac.T @ np.linalg.pinv(lhs)
    return np.eye(jac.shape[1]) - pinv @ jac


def _body_clearance_for_q(
    model: PandaModel,
    q: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    collision_cfg: dict[str, Any],
) -> tuple[float, str, np.ndarray]:
    path = forward_points(model, q, side, base_pose)
    clearance, body = arm_body_clearance(path, base_pose, collision_cfg)
    return clearance, body, path


def _body_clearance_gradient(
    model: PandaModel,
    q: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    collision_cfg: dict[str, Any],
    eps: float,
    current_clearance: float,
) -> np.ndarray:
    grad = np.zeros(model.dof, dtype=float)
    for joint_idx in range(model.dof):
        shifted = q.copy()
        shifted[joint_idx] = np.clip(shifted[joint_idx] + eps, model.q_lower[joint_idx], model.q_upper[joint_idx])
        if shifted[joint_idx] == q[joint_idx]:
            continue
        shifted_clearance, _body, _path = _body_clearance_for_q(model, shifted, side, base_pose, collision_cfg)
        grad[joint_idx] = (shifted_clearance - current_clearance) / max(abs(shifted[joint_idx] - q[joint_idx]), 1e-9)
    return grad


def _refine_body_clearance(
    model: PandaModel,
    q: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    collision_cfg = cfg["collision"]
    if not bool(collision_cfg.get("collision_aware_ik", False)):
        return q
    start = float(collision_cfg.get("avoidance_start_m", 0.08))
    target = float(collision_cfg.get("avoidance_target_m", 0.055))
    max_pos = float(collision_cfg.get("avoidance_max_position_error_m", 0.075))
    max_rot = float(collision_cfg.get("avoidance_max_orientation_error_rad", 0.75))
    eps = float(collision_cfg.get("avoidance_gradient_epsilon_rad", 0.01))
    max_step = float(collision_cfg.get("avoidance_step_rad", 0.055))
    min_improvement = float(collision_cfg.get("avoidance_min_improvement_m", 0.0005))
    damping = float(collision_cfg.get("avoidance_nullspace_damping", 0.045))
    q_current = np.asarray(q, dtype=float).copy()
    for _ in range(int(collision_cfg.get("avoidance_iterations", 5))):
        clearance, _body, _path = _body_clearance_for_q(model, q_current, side, base_pose, collision_cfg)
        if clearance >= target or clearance >= start:
            break
        grad = _body_clearance_gradient(model, q_current, side, base_pose, collision_cfg, eps, clearance)
        if float(np.linalg.norm(grad)) < 1e-9:
            break
        jac = _weighted_pose_jacobian(model, q_current, side, base_pose, cfg)
        direction = _nullspace_projector(jac, damping) @ grad
        if float(np.linalg.norm(direction)) < 1e-8:
            direction = 0.25 * grad
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-9:
            break
        direction = direction / direction_norm
        accepted = False
        for scale in [1.0, 0.5, 0.25, 0.125]:
            q_trial = np.clip(q_current + scale * max_step * direction, model.q_lower, model.q_upper)
            trial_clearance, _trial_body, _trial_path = _body_clearance_for_q(model, q_trial, side, base_pose, collision_cfg)
            if trial_clearance <= clearance + min_improvement:
                continue
            pos_error, rot_error = _pose_error_values(model, q_trial, target_position, target_rotation, side, base_pose)
            if pos_error <= max_pos and rot_error <= max_rot:
                q_current = q_trial
                accepted = True
                break
        if not accepted:
            break
    return q_current


def _inter_clearance_gradient(
    model: PandaModel,
    q_left: np.ndarray,
    q_right: np.ndarray,
    base_pose: np.ndarray,
    collision_cfg: dict[str, Any],
    eps: float,
    current_clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_path = forward_points(model, q_left, "left", base_pose)
    right_path = forward_points(model, q_right, "right", base_pose)
    grad_left = np.zeros(model.dof, dtype=float)
    grad_right = np.zeros(model.dof, dtype=float)
    for joint_idx in range(model.dof):
        shifted = q_left.copy()
        shifted[joint_idx] = np.clip(shifted[joint_idx] + eps, model.q_lower[joint_idx], model.q_upper[joint_idx])
        if shifted[joint_idx] != q_left[joint_idx]:
            value = inter_arm_clearance(forward_points(model, shifted, "left", base_pose), right_path, collision_cfg)
            grad_left[joint_idx] = (value - current_clearance) / max(abs(shifted[joint_idx] - q_left[joint_idx]), 1e-9)
        shifted = q_right.copy()
        shifted[joint_idx] = np.clip(shifted[joint_idx] + eps, model.q_lower[joint_idx], model.q_upper[joint_idx])
        if shifted[joint_idx] != q_right[joint_idx]:
            value = inter_arm_clearance(left_path, forward_points(model, shifted, "right", base_pose), collision_cfg)
            grad_right[joint_idx] = (value - current_clearance) / max(abs(shifted[joint_idx] - q_right[joint_idx]), 1e-9)
    return grad_left, grad_right


def _refine_inter_arm_clearance(
    model: PandaModel,
    left_q: np.ndarray,
    right_q: np.ndarray,
    left_target_position: np.ndarray,
    right_target_position: np.ndarray,
    left_target_rotation: np.ndarray,
    right_target_rotation: np.ndarray,
    base_pose: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    collision_cfg = cfg["collision"]
    if not bool(collision_cfg.get("collision_aware_ik", False)):
        return left_q, right_q
    target = float(collision_cfg.get("avoidance_target_m", 0.055))
    start = float(collision_cfg.get("avoidance_start_m", 0.08))
    max_pos = float(collision_cfg.get("avoidance_max_position_error_m", 0.075))
    max_rot = float(collision_cfg.get("avoidance_max_orientation_error_rad", 0.75))
    eps = float(collision_cfg.get("avoidance_gradient_epsilon_rad", 0.01))
    max_step = float(collision_cfg.get("avoidance_step_rad", 0.055))
    min_improvement = float(collision_cfg.get("avoidance_min_improvement_m", 0.0005))
    damping = float(collision_cfg.get("avoidance_nullspace_damping", 0.045))
    q_left = np.asarray(left_q, dtype=float).copy()
    q_right = np.asarray(right_q, dtype=float).copy()
    for _ in range(int(collision_cfg.get("avoidance_iterations", 5))):
        left_path = forward_points(model, q_left, "left", base_pose)
        right_path = forward_points(model, q_right, "right", base_pose)
        clearance = inter_arm_clearance(left_path, right_path, collision_cfg)
        if clearance >= target or clearance >= start:
            break
        grad_left, grad_right = _inter_clearance_gradient(model, q_left, q_right, base_pose, collision_cfg, eps, clearance)
        if float(np.linalg.norm(grad_left) + np.linalg.norm(grad_right)) < 1e-9:
            break
        n_left = _nullspace_projector(_weighted_pose_jacobian(model, q_left, "left", base_pose, cfg), damping)
        n_right = _nullspace_projector(_weighted_pose_jacobian(model, q_right, "right", base_pose, cfg), damping)
        direction = np.concatenate([n_left @ grad_left, n_right @ grad_right])
        if float(np.linalg.norm(direction)) < 1e-8:
            direction = 0.25 * np.concatenate([grad_left, grad_right])
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-9:
            break
        direction = direction / direction_norm
        accepted = False
        for scale in [1.0, 0.5, 0.25, 0.125]:
            trial_left = np.clip(q_left + scale * max_step * direction[: model.dof], model.q_lower, model.q_upper)
            trial_right = np.clip(q_right + scale * max_step * direction[model.dof :], model.q_lower, model.q_upper)
            trial_left_path = forward_points(model, trial_left, "left", base_pose)
            trial_right_path = forward_points(model, trial_right, "right", base_pose)
            trial_clearance = inter_arm_clearance(trial_left_path, trial_right_path, collision_cfg)
            if trial_clearance <= clearance + min_improvement:
                continue
            left_body, _body_name_l = arm_body_clearance(trial_left_path, base_pose, collision_cfg)
            right_body, _body_name_r = arm_body_clearance(trial_right_path, base_pose, collision_cfg)
            if min(left_body, right_body) < -0.01:
                continue
            left_pos_error, left_rot_error = _pose_error_values(model, trial_left, left_target_position, left_target_rotation, "left", base_pose)
            right_pos_error, right_rot_error = _pose_error_values(model, trial_right, right_target_position, right_target_rotation, "right", base_pose)
            if (
                left_pos_error <= max_pos
                and right_pos_error <= max_pos
                and left_rot_error <= max_rot
                and right_rot_error <= max_rot
            ):
                q_left = trial_left
                q_right = trial_right
                accepted = True
                break
        if not accepted:
            break
    return q_left, q_right


def _build_side_result_from_q(
    task: TaskData,
    model: PandaModel,
    cfg: dict[str, Any],
    side: str,
    base_pose: np.ndarray,
    target_rotations: np.ndarray,
    q_arr: np.ndarray,
    target_positions: np.ndarray | None = None,
    active: bool = True,
) -> dict[str, Any]:
    frame_spec = _task_frame_spec(cfg, task) if active else None
    # evaluation.use_measured_forces=false runs the force-free stage (measured
    # wrenches ignored; the torque criterion covers the rigid-body dynamics of
    # the motion only).
    use_forces = active and bool(cfg.get("evaluation", {}).get("use_measured_forces", True))
    target_positions = np.asarray(target_positions if target_positions is not None else task.side_position(side), dtype=float)
    errors = []
    orientation_errors = []
    manips = []
    tau_ext = []
    paths = []
    for idx, q in enumerate(q_arr):
        pos_error, rot_error = _pose_error_values(
            model, q, target_positions[idx], target_rotations[idx], side, base_pose[idx], frame_spec
        )
        path = forward_points(model, q, side, base_pose[idx])
        jac_pos, jac_rot = pose_jacobian(model, q, side, base_pose[idx])
        force = task.side_force(side)[idx] if use_forces else np.zeros(3)
        torque = task.side_torque(side)[idx] if use_forces else np.zeros(3)
        tau = jac_pos.T @ force + jac_rot.T @ torque
        errors.append(pos_error)
        orientation_errors.append(rot_error)
        manips.append(manipulability(model, q, side, base_pose[idx]))
        tau_ext.append(tau)
        paths.append(path)
    tau_ext_arr = np.asarray(tau_ext, dtype=float)
    torque_util, qdot, tau_dyn, tau_total, time_scale, discontinuity_fraction = _trajectory_torque_utilization(
        model, task, q_arr, tau_ext_arr, side, cfg
    )
    speed_util = np.abs(qdot) / np.maximum(model.velocity_limits_radps[None, :], 1e-9)
    limit_tol = float(cfg.get("success", {}).get("joint_limit_margin_rad", 0.035))
    near_limit = (q_arr - model.q_lower[None, :] < limit_tol) | (model.q_upper[None, :] - q_arr < limit_tol)
    joint_limit_proximity = float(np.mean(np.any(near_limit, axis=1))) if active and len(q_arr) else 0.0
    return {
        "time_scale": float(time_scale),
        "ik_discontinuity_fraction": float(discontinuity_fraction),
        "joint_limit_proximity_fraction": joint_limit_proximity,
        "errors": np.asarray(errors, dtype=float),
        "orientation_errors": np.asarray(orientation_errors, dtype=float),
        "target_positions": target_positions,
        "target_rotations": target_rotations,
        "manipulability": np.asarray(manips, dtype=float),
        "q": np.asarray(q_arr, dtype=float),
        "paths": np.asarray(paths, dtype=float),
        "torque_utilization": torque_util,
        "dynamic_torque_nm": tau_dyn,
        "external_torque_nm": tau_ext_arr,
        "total_torque_nm": tau_total,
        "speed_utilization": speed_util,
        "measured_force_n": np.asarray(task.side_force(side) if use_forces else np.zeros((task.sample_count, 3)), dtype=float),
        "measured_task_torque_nm": np.asarray(task.side_torque(side) if use_forces else np.zeros((task.sample_count, 3)), dtype=float),
        "active": bool(active),
    }


def _evaluate_side(
    task: TaskData,
    model: PandaModel,
    cfg: dict[str, Any],
    side: str,
    base_pose: np.ndarray,
) -> dict[str, Any]:
    frame_spec = _task_frame_spec(cfg, task)
    cfg = _cfg_for_frame_spec(cfg, frame_spec)
    q = model.q_nominal.copy()
    raw_rotations = task.side_rotation(side)
    target_rotations = raw_rotations
    if bool(cfg["ik"].get("use_orientation", True)) and str(cfg["ik"].get("orientation_mode", "relative")).lower() == "relative":
        q_seed, _seed_error = _solve_position_ik_from_seeds(model, task.side_position(side)[0], side, base_pose[0], q, cfg)
        offset = anchor_offset(frame_spec, raw_rotations[0], end_effector_rotation(model, q_seed, side, base_pose[0]))
        target_rotations = raw_rotations @ offset
        q = q_seed
    errors = []
    orientation_errors = []
    q_traj = []
    for idx in range(task.sample_count):
        q, error, orientation_error = _solve_pose_ik_with_retries(
            model,
            task.side_position(side)[idx],
            target_rotations[idx],
            side,
            base_pose[idx],
            q,
            cfg,
        )
        q = _refine_body_clearance(model, q, task.side_position(side)[idx], target_rotations[idx], side, base_pose[idx], cfg)
        error, orientation_error = _pose_error_values(
            model, q, task.side_position(side)[idx], target_rotations[idx], side, base_pose[idx], frame_spec
        )
        errors.append(error)
        orientation_errors.append(orientation_error)
        q_traj.append(q.copy())
    q_arr = np.asarray(q_traj)
    return _build_side_result_from_q(task, model, cfg, side, base_pose, target_rotations, q_arr, active=True)


def _evaluate_inactive_side(task: TaskData, model: PandaModel, cfg: dict[str, Any], side: str, base_pose: np.ndarray) -> dict[str, Any]:
    q_arr = np.repeat(model.q_nominal[None, :], task.sample_count, axis=0)
    target_positions = np.asarray([end_effector(model, q, side, pose) for q, pose in zip(q_arr, base_pose)], dtype=float)
    target_rotations = np.asarray([end_effector_rotation(model, q, side, pose) for q, pose in zip(q_arr, base_pose)], dtype=float)
    return _build_side_result_from_q(
        task,
        model,
        cfg,
        side,
        base_pose,
        target_rotations,
        q_arr,
        target_positions=target_positions,
        active=False,
    )


def _evaluate_variant(task: TaskData, model: PandaModel, cfg: dict[str, Any], variant: dict[str, float]) -> dict[str, Any]:
    base_motion = estimate_base_motion(task, model, cfg, variant)
    pose = np.asarray(base_motion["pose"], dtype=float)
    active_sides = task.active_sides()
    side = {
        "left": _evaluate_side(task, model, cfg, "left", pose)
        if "left" in active_sides
        else _evaluate_inactive_side(task, model, cfg, "left", pose),
        "right": _evaluate_side(task, model, cfg, "right", pose)
        if "right" in active_sides
        else _evaluate_inactive_side(task, model, cfg, "right", pose),
    }
    if bool(cfg["collision"].get("collision_aware_ik", False)) and len(active_sides) == 2:
        left_q = side["left"]["q"].copy()
        right_q = side["right"]["q"].copy()
        changed = False
        for idx in range(task.sample_count):
            refined_left, refined_right = _refine_inter_arm_clearance(
                model,
                left_q[idx],
                right_q[idx],
                task.left_pos[idx],
                task.right_pos[idx],
                side["left"]["target_rotations"][idx],
                side["right"]["target_rotations"][idx],
                pose[idx],
                cfg,
            )
            if not np.allclose(refined_left, left_q[idx]) or not np.allclose(refined_right, right_q[idx]):
                changed = True
                left_q[idx] = refined_left
                right_q[idx] = refined_right
        if changed:
            side["left"] = _build_side_result_from_q(
                task, model, cfg, "left", pose, side["left"]["target_rotations"], left_q, active=True
            )
            side["right"] = _build_side_result_from_q(
                task, model, cfg, "right", pose, side["right"]["target_rotations"], right_q, active=True
            )

    metric_sides = list(active_sides or ("left", "right"))
    max_time_scale_used = max(float(side[name].get("time_scale", 1.0)) for name in metric_sides)
    max_ik_discontinuity = max(float(side[name].get("ik_discontinuity_fraction", 0.0)) for name in metric_sides)
    max_joint_limit_proximity = max(float(side[name].get("joint_limit_proximity_fraction", 0.0)) for name in metric_sides)
    all_errors = np.concatenate([side[name]["errors"] for name in metric_sides])
    all_orientation_errors = np.concatenate([side[name]["orientation_errors"] for name in metric_sides])
    all_manips = np.concatenate([side[name]["manipulability"] for name in metric_sides])
    all_torque = np.vstack([side[name]["torque_utilization"] for name in metric_sides])
    all_dynamic_torque = np.vstack([side[name]["dynamic_torque_nm"] for name in metric_sides])
    all_external_torque = np.vstack([side[name]["external_torque_nm"] for name in metric_sides])
    all_total_torque = np.vstack([side[name]["total_torque_nm"] for name in metric_sides])
    all_speed = np.vstack([side[name]["speed_utilization"] for name in metric_sides])
    all_forces = np.vstack([side[name]["measured_force_n"] for name in metric_sides])
    all_task_torques = np.vstack([side[name]["measured_task_torque_nm"] for name in metric_sides])

    body_clearances = []
    inter_clearances = []
    worst_body = ""
    for idx in range(task.sample_count):
        left_clearance, left_body = arm_body_clearance(side["left"]["paths"][idx], pose[idx], cfg["collision"])
        right_clearance, right_body = arm_body_clearance(side["right"]["paths"][idx], pose[idx], cfg["collision"])
        if left_clearance <= right_clearance:
            body_clearances.append(left_clearance)
            frame_body = f"left:{left_body}"
        else:
            body_clearances.append(right_clearance)
            frame_body = f"right:{right_body}"
        if body_clearances[-1] <= min(body_clearances):
            worst_body = frame_body
        inter_clearances.append(inter_arm_clearance(side["left"]["paths"][idx], side["right"]["paths"][idx], cfg["collision"]))

    min_body = float(np.min(body_clearances))
    min_inter = float(np.min(inter_clearances))
    min_clearance = min(min_body, min_inter)
    success_cfg = cfg["success"]
    hard = float(success_cfg.get("min_clearance_m", 0.05))
    warning_clearance = float(cfg.get("collision", {}).get("warning_clearance_m", max(hard, 0.05)))
    mean_position_limit = float(success_cfg.get("mean_position_error_m", 0.05))
    max_position_limit = float(success_cfg.get("max_position_error_m", 0.12))
    mean_orientation_limit = float(success_cfg.get("mean_orientation_error_rad", 0.90))
    max_orientation_limit = float(success_cfg.get("max_orientation_error_rad", 1.75))
    min_manip_limit = float(success_cfg.get("min_manipulability", 0.08))
    max_torque_limit = float(success_cfg.get("max_torque_utilization", 1.0))
    max_joint_speed_limit = float(success_cfg.get("max_joint_speed_utilization", 1.0))
    max_base_speed_limit = float(cfg["base"].get("max_speed_mps", float("inf")))
    max_base_acc_limit = float(cfg["base"].get("max_acc_mps2", float("inf")))
    mean_error = float(np.mean(all_errors))
    max_error = float(np.max(all_errors))
    mean_orientation_error = float(np.mean(all_orientation_errors))
    max_orientation_error = float(np.max(all_orientation_errors))
    min_manip = float(np.min(all_manips))
    max_torque_util = float(np.max(all_torque))
    max_dynamic_torque = float(np.max(np.abs(all_dynamic_torque)))
    max_external_torque = float(np.max(np.abs(all_external_torque)))
    max_total_torque = float(np.max(np.abs(all_total_torque)))
    mean_total_torque = float(np.mean(np.abs(all_total_torque)))
    max_speed_util = float(np.max(all_speed))
    max_measured_force = float(np.max(np.linalg.norm(all_forces, axis=1))) if len(all_forces) else 0.0
    max_measured_task_torque = float(np.max(np.linalg.norm(all_task_torques, axis=1))) if len(all_task_torques) else 0.0
    reachable_fraction = float(np.mean(all_errors <= float(success_cfg.get("max_position_error_m", 0.12))))
    mean_position_ok = bool(mean_error <= mean_position_limit)
    max_position_ok = bool(max_error <= max_position_limit)
    mean_orientation_ok = bool(mean_orientation_error <= mean_orientation_limit)
    max_orientation_ok = bool(max_orientation_error <= max_orientation_limit)
    manipulability_ok = bool(min_manip >= min_manip_limit)
    garmi_collision_free = bool(min_body >= hard)
    inter_arm_collision_free = bool(min_inter >= hard)
    collision_free = bool(garmi_collision_free and inter_arm_collision_free)
    clearance_warning = bool(min_clearance < warning_clearance)
    torque_ok = bool(max_torque_util <= max_torque_limit)
    joint_speed_ok = bool(max_speed_util <= max_joint_speed_limit)
    base_speed_ok = bool(float(base_motion["max_speed_mps"]) <= max_base_speed_limit)
    base_acc_ok = bool(float(base_motion["max_acc_mps2"]) <= max_base_acc_limit)
    max_ik_discontinuity_limit = float(success_cfg.get("max_ik_discontinuity_fraction", 0.05))
    continuity_ok = bool(max_ik_discontinuity <= max_ik_discontinuity_limit)
    motion_ok = bool(joint_speed_ok and base_speed_ok and base_acc_ok and continuity_ok)
    position_ok = bool(mean_position_ok and max_position_ok)
    orientation_ok = bool(mean_orientation_ok and max_orientation_ok)
    kinematic_ok = bool(position_ok and orientation_ok and manipulability_ok)
    success = bool(kinematic_ok and torque_ok and collision_free and motion_ok)
    failure_reasons = []
    if not max_position_ok:
        failure_reasons.append("max_position_error")
    if not mean_position_ok:
        failure_reasons.append("mean_position_error")
    if not max_orientation_ok:
        failure_reasons.append("max_orientation_error")
    if not mean_orientation_ok:
        failure_reasons.append("mean_orientation_error")
    if not manipulability_ok:
        failure_reasons.append("low_manipulability")
    if not torque_ok:
        failure_reasons.append("torque_limit")
    if not garmi_collision_free:
        failure_reasons.append("garmi_collision")
    if not inter_arm_collision_free:
        failure_reasons.append("inter_arm_collision")
    if not joint_speed_ok:
        failure_reasons.append("joint_speed_limit")
    if not base_speed_ok:
        failure_reasons.append("base_speed_limit")
    if not base_acc_ok:
        failure_reasons.append("base_acc_limit")
    if not continuity_ok:
        failure_reasons.append("ik_discontinuity")
    primary_failure = failure_reasons[0] if failure_reasons else "ok"
    objective = (
        120.0 * mean_error
        + 35.0 * max_error
        + 9.0 * mean_orientation_error
        + 4.0 * max(0.0, max_orientation_error - max_orientation_limit)
        + 25.0 * max(0.0, min_manip_limit - min_manip)
        + 55.0 * max(0.0, max_torque_util - max_torque_limit)
        + 1000.0 * max(0.0, hard - min_clearance)
        + 45.0 * max(0.0, warning_clearance - min_clearance)
        + 1.5 * float(base_motion["path_length_m"])
        + 12.0 * max(0.0, max_speed_util - 1.0)
    )
    # Soft loss: continuous, threshold-normalized distance to task success.
    # Zero when every success criterion is met with margin; grows smoothly with
    # each normalized overshoot so the optimizer receives gradient information
    # between the binary success plateaus.
    frame_spec = _task_frame_spec(cfg, task)
    soft_terms = {
        "mean_position": max(0.0, mean_error / max(mean_position_limit, 1e-9) - 1.0),
        "max_position": max(0.0, max_error / max(max_position_limit, 1e-9) - 1.0),
        "mean_orientation": max(0.0, mean_orientation_error / max(mean_orientation_limit, 1e-9) - 1.0)
        if frame_spec.scores_orientation
        else 0.0,
        "max_orientation": max(0.0, max_orientation_error / max(max_orientation_limit, 1e-9) - 1.0)
        if frame_spec.scores_orientation
        else 0.0,
        "manipulability": max(0.0, (min_manip_limit - min_manip) / max(min_manip_limit, 1e-9)),
        "torque": max(0.0, max_torque_util / max(max_torque_limit, 1e-9) - 1.0),
        "joint_speed": max(0.0, max_speed_util / max(max_joint_speed_limit, 1e-9) - 1.0),
        "penetration": max(0.0, -min_clearance) / 0.05,
        "clearance_warning": max(0.0, (warning_clearance - min_clearance) / max(warning_clearance, 1e-9))
        if min_clearance >= 0.0
        else 1.0,
        "base_speed": max(0.0, float(base_motion["max_speed_mps"]) / max(max_base_speed_limit, 1e-9) - 1.0)
        if np.isfinite(max_base_speed_limit)
        else 0.0,
        "base_acc": max(0.0, float(base_motion["max_acc_mps2"]) / max(max_base_acc_limit, 1e-9) - 1.0)
        if np.isfinite(max_base_acc_limit)
        else 0.0,
        "ik_discontinuity": max_ik_discontinuity / max(max_ik_discontinuity_limit, 1e-9),
        # Working close to the joint limits is undesirable; it is scored as a
        # bounded penalty rather than through joint-limit constraint forces.
        "joint_limit_proximity": max_joint_limit_proximity,
    }
    # Cap each term so that a single degenerate task cannot dominate the
    # candidate score (and thus the GP surrogate).
    soft_term_cap = 10.0
    soft_terms = {key: min(float(value), soft_term_cap) for key, value in soft_terms.items()}
    soft_weights = {
        "mean_position": 1.0,
        "max_position": 0.6,
        "mean_orientation": 0.6,
        "max_orientation": 0.35,
        "manipulability": 0.5,
        "torque": 1.0,
        "joint_speed": 0.7,
        "penetration": 3.0,
        "clearance_warning": 0.25,
        "base_speed": 0.5,
        "base_acc": 0.35,
        "ik_discontinuity": 1.5,
        "joint_limit_proximity": 0.5,
    }
    soft_loss = float(sum(soft_weights[key] * soft_terms[key] for key in soft_terms))
    task_weight = float(cfg.get("_task_weights", {}).get(split_variant_name(task.name)[0], 1.0))
    metrics = {
        "task_name": task.name,
        "active_sides": ",".join(metric_sides),
        "active_left": bool("left" in active_sides),
        "active_right": bool("right" in active_sides),
        "active_side_count": int(len(metric_sides)),
        "success": success,
        "kinematic_ok": kinematic_ok,
        "position_ok": position_ok,
        "orientation_ok": orientation_ok,
        "manipulability_ok": manipulability_ok,
        "torque_ok": torque_ok,
        "collision_free": collision_free,
        "garmi_collision_free": garmi_collision_free,
        "inter_arm_collision_free": inter_arm_collision_free,
        "clearance_warning": clearance_warning,
        "motion_ok": motion_ok,
        "joint_speed_ok": joint_speed_ok,
        "continuity_ok": continuity_ok,
        "time_scale": max_time_scale_used,
        "ik_discontinuity_fraction": max_ik_discontinuity,
        "joint_limit_proximity_fraction": max_joint_limit_proximity,
        "base_speed_ok": base_speed_ok,
        "base_acc_ok": base_acc_ok,
        "primary_failure": primary_failure,
        "failure_reasons": ";".join(failure_reasons) if failure_reasons else "ok",
        "objective": float(objective),
        "soft_loss": soft_loss,
        "task_weight": task_weight,
        "orientation_mode": frame_spec.orientation_mode,
        "free_tool_axis": bool(frame_spec.free_tool_axis),
        "reachable_fraction": reachable_fraction,
        "mean_position_error_m": mean_error,
        "max_position_error_m": max_error,
        "max_position_margin_m": max_position_limit - max_error,
        "mean_orientation_error_rad": mean_orientation_error,
        "max_orientation_error_rad": max_orientation_error,
        "max_orientation_margin_rad": max_orientation_limit - max_orientation_error,
        "min_manipulability": min_manip,
        "manipulability_margin": min_manip - min_manip_limit,
        "mean_manipulability": float(np.mean(all_manips)),
        "max_torque_utilization": max_torque_util,
        "torque_utilization_margin": max_torque_limit - max_torque_util,
        "mean_torque_utilization": float(np.mean(all_torque)),
        "max_dynamic_torque_nm": max_dynamic_torque,
        "max_external_torque_nm": max_external_torque,
        "max_total_torque_nm": max_total_torque,
        "mean_abs_total_torque_nm": mean_total_torque,
        "max_measured_force_n": max_measured_force,
        "max_measured_task_torque_nm": max_measured_task_torque,
        "max_joint_speed_utilization": max_speed_util,
        "joint_speed_utilization_margin": max_joint_speed_limit - max_speed_util,
        "min_garmi_clearance_m": min_body,
        "min_inter_arm_clearance_m": min_inter,
        "min_clearance_m": min_clearance,
        "clearance_margin_m": min_clearance - hard,
        "warning_clearance_m": warning_clearance,
        "warning_clearance_margin_m": min_clearance - warning_clearance,
        "worst_collision_body": worst_body,
        "base_path_length_m": float(base_motion["path_length_m"]),
        "base_motion_mode": str(base_motion.get("base_motion_mode", cfg.get("base", {}).get("motion_mode", "free_holonomic"))),
        "base_max_speed_mps": float(base_motion["max_speed_mps"]),
        "base_speed_margin_mps": max_base_speed_limit - float(base_motion["max_speed_mps"]),
        "base_max_acc_mps2": float(base_motion["max_acc_mps2"]),
        "base_acc_margin_mps2": max_base_acc_limit - float(base_motion["max_acc_mps2"]),
        "base_max_yaw_rate_radps": float(base_motion.get("max_yaw_rate_radps", 0.0)),
        "base_preferred_reach_fraction": float(base_motion["preferred_reach_fraction"]),
        "base_forward_offset_m": float(base_motion["forward_offset_m"]),
        "base_lateral_offset_m": float(base_motion["lateral_offset_m"]),
        "base_yaw_offset_rad": float(base_motion["yaw_offset_rad"]),
    }
    return {"metrics": metrics, "base_motion": base_motion, "side": side}


def evaluate_task_baseline(task: TaskData, cfg: dict[str, Any]) -> dict[str, Any]:
    # Locomotion tasks may override the base-motion mode per task
    # (config/task_frames.yaml); walking assistance is impossible with a base
    # that stays at its initial placement.
    spec = _task_frame_spec(cfg, task)
    cfg = dict(cfg)
    if spec.base_motion_mode:
        cfg["base"] = dict(cfg.get("base", {}))
        set_base_motion_mode(cfg, spec.base_motion_mode)
    # Collision proxies (shoulder pads) follow the candidate mounts.
    cfg["collision"] = {**cfg.get("collision", {}), "_mounts": cfg.get("panda", {}).get("mounts")}
    model = PandaModel.from_config(cfg)
    variants = base_variants(cfg)
    sort_key = lambda item: (not item["metrics"]["success"], not item["metrics"]["collision_free"], item["metrics"]["objective"])  # noqa: E731

    # Collision-aware IK is expensive, so the evaluation runs in two stages:
    # screen all base variants with plain IK, then re-evaluate only the best
    # variant with collision-aware IK.
    aware = bool(cfg.get("collision", {}).get("collision_aware_ik", False))
    if aware and len(variants) > 1:
        fast_cfg = dict(cfg)
        fast_cfg["collision"] = dict(cfg["collision"])
        fast_cfg["collision"]["collision_aware_ik"] = False
        fast_results = [_evaluate_variant(task, model, fast_cfg, variant) for variant in variants]
        order = sorted(range(len(variants)), key=lambda i: sort_key(fast_results[i]))
        best_idx = order[0]
        refined = _evaluate_variant(task, model, cfg, variants[best_idx])
        results = [refined, *fast_results]
    else:
        results = [_evaluate_variant(task, model, cfg, variant) for variant in variants]
    results.sort(key=sort_key)
    return results[0]


_WORKER_CFG: dict[str, Any] | None = None


def _init_worker(cfg: dict[str, Any]) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _evaluate_task_worker(task: TaskData) -> dict[str, Any]:
    if _WORKER_CFG is None:
        raise RuntimeError("Baseline worker is not initialized.")
    result = evaluate_task_baseline(task, _WORKER_CFG)
    return {"task": task.name, **result["metrics"]}


def _worker_count(cfg: dict[str, Any], task_count: int) -> int:
    value = cfg.get("performance", {}).get("parallel_workers", "auto")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"auto", "all", ""}:
            count = os.cpu_count() or 1
        elif text in {"off", "serial", "none", "false"}:
            count = 1
        else:
            count = int(text)
    else:
        count = int(value)
    return max(1, min(count, max(1, task_count)))


def _profile_slug(cfg: dict[str, Any], fallback: str = "baseline") -> str:
    value = str(cfg.get("_profile") or cfg.get("baseline", {}).get("name") or fallback)
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or fallback


def evaluate_tasks_parallel(tasks: list[TaskData], cfg: dict[str, Any], group_name: str) -> list[dict[str, Any]]:
    workers = _worker_count(cfg, len(tasks))
    progress = bool(cfg.get("performance", {}).get("progress", True))
    started = time.perf_counter()
    if workers <= 1:
        rows = []
        for idx, task in enumerate(tasks, start=1):
            result = evaluate_task_baseline(task, cfg)
            rows.append({"task": task.name, **result["metrics"]})
            if progress:
                print(f"[baseline:{group_name}] {idx}/{len(tasks)} {task.name}", flush=True)
        return rows
    if progress:
        print(f"[baseline:{group_name}] evaluating {len(tasks)} tasks with {workers} worker processes", flush=True)
    rows = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(cfg,)) as executor:
        futures = {executor.submit(_evaluate_task_worker, task): task.name for task in tasks}
        completed = 0
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if progress:
                elapsed = time.perf_counter() - started
                print(f"[baseline:{group_name}] {completed}/{len(tasks)} tasks in {elapsed:.1f}s", flush=True)
    rows.sort(key=lambda row: row["task_name"])
    return rows


def _aggregate(rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if len(df) and "task_weight" in df:
        weights = df["task_weight"].astype(float).clip(lower=0.0)
        weight_sum = float(weights.sum())
        weighted_success = float((df["success"].astype(float) * weights).sum() / weight_sum) if weight_sum > 0 else 0.0
        weighted_soft = (
            float((df["soft_loss"].astype(float) * weights).sum() / weight_sum)
            if weight_sum > 0 and "soft_loss" in df
            else 0.0
        )
    else:
        weighted_success = float(df["success"].mean()) if len(df) else 0.0
        weighted_soft = float(df["soft_loss"].mean()) if len(df) and "soft_loss" in df else 0.0
    return {
        "group_name": group_name,
        "task_count": int(len(df)),
        "success_count": int(df["success"].sum()),
        "success_fraction": weighted_success,
        "success_fraction_unweighted": float(df["success"].mean()) if len(df) else 0.0,
        "mean_soft_loss": weighted_soft,
        "max_soft_loss": float(df["soft_loss"].max()) if len(df) and "soft_loss" in df else 0.0,
        **robustness_summary(rows),
        "kinematic_ok_fraction": float(df["kinematic_ok"].mean()) if len(df) else 0.0,
        "position_ok_fraction": float(df["position_ok"].mean()) if len(df) and "position_ok" in df else 0.0,
        "orientation_ok_fraction": float(df["orientation_ok"].mean()) if len(df) and "orientation_ok" in df else 0.0,
        "manipulability_ok_fraction": float(df["manipulability_ok"].mean()) if len(df) and "manipulability_ok" in df else 0.0,
        "torque_ok_fraction": float(df["torque_ok"].mean()) if len(df) else 0.0,
        "collision_free_fraction": float(df["collision_free"].mean()) if len(df) else 0.0,
        "motion_ok_fraction": float(df["motion_ok"].mean()) if len(df) and "motion_ok" in df else 0.0,
        "mean_position_error_m": float(df["mean_position_error_m"].mean()) if len(df) else 0.0,
        "max_position_error_m": float(df["max_position_error_m"].max()) if len(df) else 0.0,
        "mean_orientation_error_rad": float(df["mean_orientation_error_rad"].mean()) if len(df) else 0.0,
        "max_orientation_error_rad": float(df["max_orientation_error_rad"].max()) if len(df) else 0.0,
        "min_manipulability": float(df["min_manipulability"].min()) if len(df) else 0.0,
        "max_torque_utilization": float(df["max_torque_utilization"].max()) if len(df) else 0.0,
        "max_dynamic_torque_nm": float(df["max_dynamic_torque_nm"].max()) if len(df) and "max_dynamic_torque_nm" in df else 0.0,
        "max_external_torque_nm": float(df["max_external_torque_nm"].max()) if len(df) and "max_external_torque_nm" in df else 0.0,
        "max_total_torque_nm": float(df["max_total_torque_nm"].max()) if len(df) and "max_total_torque_nm" in df else 0.0,
        "max_measured_force_n": float(df["max_measured_force_n"].max()) if len(df) and "max_measured_force_n" in df else 0.0,
        "max_measured_task_torque_nm": float(df["max_measured_task_torque_nm"].max()) if len(df) and "max_measured_task_torque_nm" in df else 0.0,
        "max_joint_speed_utilization": float(df["max_joint_speed_utilization"].max()) if len(df) else 0.0,
        "min_clearance_m": float(df["min_clearance_m"].min()) if len(df) else 0.0,
        "min_clearance_margin_m": float(df["clearance_margin_m"].min()) if len(df) and "clearance_margin_m" in df else 0.0,
        "min_warning_clearance_margin_m": float(df["warning_clearance_margin_m"].min()) if len(df) and "warning_clearance_margin_m" in df else 0.0,
        "clearance_warning_fraction": float(df["clearance_warning"].mean()) if len(df) and "clearance_warning" in df else 0.0,
        "base_motion_mode": str(df["base_motion_mode"].mode().iloc[0]) if len(df) and "base_motion_mode" in df else "",
        "mean_base_path_length_m": float(df["base_path_length_m"].mean()) if len(df) and "base_path_length_m" in df else 0.0,
        "max_base_speed_mps": float(df["base_max_speed_mps"].max()) if len(df) and "base_max_speed_mps" in df else 0.0,
        "max_base_acc_mps2": float(df["base_max_acc_mps2"].max()) if len(df) and "base_max_acc_mps2" in df else 0.0,
        "max_base_yaw_rate_radps": float(df["base_max_yaw_rate_radps"].max()) if len(df) and "base_max_yaw_rate_radps" in df else 0.0,
        "worst_task_by_objective": str(df.sort_values("objective", ascending=False).iloc[0]["task_name"]) if len(df) else "",
        "most_common_primary_failure": str(df["primary_failure"].mode().iloc[0]) if len(df) and "primary_failure" in df else "",
    }


def _failure_breakdown(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["failure", "task_count", "task_fraction"])
    df = pd.DataFrame(rows)
    records = []
    for failure, group in df.groupby("primary_failure", dropna=False):
        records.append(
            {
                "failure": str(failure),
                "task_count": int(len(group)),
                "task_fraction": float(len(group) / max(len(df), 1)),
            }
        )
    out = pd.DataFrame(records).sort_values(["task_count", "failure"], ascending=[False, True])
    return out.reset_index(drop=True)


def _failure_taxonomy(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    columns = [
        "task_name",
        "active_sides",
        "success",
        "primary_failure",
        "failure_reasons",
        "position_ok",
        "orientation_ok",
        "manipulability_ok",
        "torque_ok",
        "garmi_collision_free",
        "inter_arm_collision_free",
        "joint_speed_ok",
        "base_speed_ok",
        "base_acc_ok",
        "max_position_error_m",
        "max_orientation_error_rad",
        "min_manipulability",
        "max_torque_utilization",
        "max_dynamic_torque_nm",
        "max_external_torque_nm",
        "max_total_torque_nm",
        "max_measured_force_n",
        "max_measured_task_torque_nm",
        "max_joint_speed_utilization",
        "min_garmi_clearance_m",
        "min_inter_arm_clearance_m",
        "clearance_margin_m",
        "warning_clearance_margin_m",
        "clearance_warning",
        "base_motion_mode",
        "base_path_length_m",
        "base_max_speed_mps",
        "base_max_acc_mps2",
        "base_max_yaw_rate_radps",
        "worst_collision_body",
    ]
    df = pd.DataFrame(rows)
    return df[[column for column in columns if column in df.columns]].copy()


def _write_html_report(
    run_dir: Path,
    group_name: str,
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> Path:
    df = pd.DataFrame(rows)
    breakdown = _failure_breakdown(rows)
    taxonomy = _failure_taxonomy(rows)
    try:
        import plotly.express as px
        from plotly.offline import plot

        fig = px.scatter(
            df,
            x="mean_position_error_m",
            y="min_manipulability",
            size=np.clip(df["max_torque_utilization"], 0.02, None),
            color="success",
            hover_name="task_name",
            hover_data=[
                "primary_failure",
                "failure_reasons",
                "collision_free",
                "torque_ok",
                "motion_ok",
                "mean_orientation_error_rad",
                "max_orientation_error_rad",
                "min_clearance_m",
                "max_torque_utilization",
                "max_joint_speed_utilization",
                "base_motion_mode",
                "base_path_length_m",
                "base_max_speed_mps",
                "base_max_yaw_rate_radps",
            ],
            title=f"Garmi/Panda baseline task coverage: {group_name}",
        )
        plot_html = plot(fig, include_plotlyjs="cdn", output_type="div")
        failure_fig = px.bar(
            breakdown,
            x="failure",
            y="task_count",
            title=f"Primary failure breakdown: {group_name}",
        )
        failure_plot_html = plot(failure_fig, include_plotlyjs=False, output_type="div")
    except Exception:
        plot_html = "<p>Plotly unavailable.</p>"
        failure_plot_html = "<p>Plotly unavailable.</p>"
    summary = pd.DataFrame([aggregate]).to_html(index=False, float_format=lambda value: f"{value:.5f}")
    failure_table = breakdown.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    taxonomy_table = taxonomy.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    table = df.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Baseline {group_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #182027; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d7dde2; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f3f6f8; }}
</style></head>
<body>
<h1>Garmi/Panda Baseline: {group_name}</h1>
<p><strong>Baseline profile:</strong> {cfg.get("_profile", "baseline")} ({cfg.get("baseline", {}).get("name", "unnamed")})</p>
<h2>Summary</h2>
{summary}
<h2>Task Map</h2>
{plot_html}
<h2>Failure Breakdown</h2>
{failure_plot_html}
{failure_table}
<h2>Failure Taxonomy</h2>
{taxonomy_table}
<h2>Per Task Metrics</h2>
{table}
</body></html>
"""
    output = run_dir / "report.html"
    output.write_text(html, encoding="utf-8")
    return output


def run_baseline(
    group_name: str,
    task_names: list[str],
    paths: ProjectPaths = PATHS,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
) -> Path:
    ensure_dirs(paths)
    cfg = cfg or load_baseline_config(paths, profile=baseline_profile)
    if "_task_weights" not in cfg:
        try:
            from garmi_optimization.tasks import group_task_weights

            cfg["_task_weights"] = group_task_weights(group_name, paths)
        except Exception:
            cfg["_task_weights"] = {}
    tasks = load_tasks(task_names, paths)
    rows = evaluate_tasks_parallel(tasks, cfg, group_name)
    aggregate = _aggregate(rows, group_name)
    slug = _profile_slug(cfg, baseline_profile)
    prefix = f"baseline_{group_name}" if slug == "baseline" else f"baseline_{slug}_{group_name}"
    run_dir = paths.runs / timestamp_id(prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "task_success_matrix.csv", index=False)
    _failure_taxonomy(rows).to_csv(run_dir / "failure_taxonomy.csv", index=False)
    _failure_breakdown(rows).to_csv(run_dir / "failure_breakdown.csv", index=False)
    pd.DataFrame([aggregate]).to_csv(run_dir / "summary.csv", index=False)
    write_json(
        run_dir / "run_config.json",
        {
            "baseline_profile": cfg.get("_profile", baseline_profile),
            "baseline": cfg,
            "task_names": task_names,
            "aggregate": aggregate,
        },
    )
    _write_html_report(run_dir, group_name, aggregate, rows, cfg)
    return run_dir


def run_baseline_group(
    group_name: str,
    paths: ProjectPaths = PATHS,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
) -> Path:
    return run_baseline(group_name, group_task_names(group_name, paths), paths, cfg=cfg, baseline_profile=baseline_profile)


def run_baseline_global(
    paths: ProjectPaths = PATHS,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
) -> Path:
    return run_baseline("all_tasks", global_task_names(paths), paths, cfg=cfg, baseline_profile=baseline_profile)


def run_baseline_all_groups(
    paths: ProjectPaths = PATHS,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
) -> list[Path]:
    groups = load_group_config(paths).get("groups", {})
    outputs = []
    for group_name in groups:
        if group_name == "all_tasks":
            continue
        outputs.append(run_baseline_group(group_name, paths, cfg=cfg, baseline_profile=baseline_profile))
    return outputs
