"""MuJoCo model of GARMI with two Franka Emika Panda arms and the
operational-space impedance tracking simulation (paper Sec. IV-E).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from garmi_optimization.evaluation import _profile_slug, evaluate_task_baseline
from garmi_optimization.collision import BoxProxy, CapsuleProxy, SphereProxy, garmi_body_proxies
from garmi_optimization.config import DEFAULT_PROFILE, PATHS, ProjectPaths, ensure_dirs, load_baseline_config, read_json, timestamp_id, write_json
from garmi_optimization.math_utils import finite_gradient, quat_wxyz_from_rotation, rotation_error_vector, rotation_z
from garmi_optimization.panda import PandaModel
from garmi_optimization.tasks import load_task


def _vec(values: list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _copy_source_default_and_assets(root: ET.Element, source_root: ET.Element, source_path: Path) -> None:
    source_default = source_root.find("default")
    if source_default is not None:
        default = copy.deepcopy(source_default)
        ET.SubElement(default, "joint", {"limited": "true"})
        ET.SubElement(default, "geom", {"contype": "0", "conaffinity": "0"})
        root.append(default)
    source_asset = source_root.find("asset")
    if source_asset is not None:
        root.append(copy.deepcopy(source_asset))
    meshdir = source_root.find("compiler").attrib.get("meshdir", "assets") if source_root.find("compiler") is not None else "assets"
    root.find("compiler").set("meshdir", str((source_path.parent / meshdir).resolve()))


def _remove_matching_children(element: ET.Element) -> None:
    for child in list(element):
        name = child.attrib.get("name", "")
        geom_class = child.attrib.get("class", "")
        if child.tag == "body" and "finger" in name:
            element.remove(child)
            continue
        if child.tag == "geom" and "collision" in geom_class:
            element.remove(child)
            continue
        _remove_matching_children(child)


def _rename_panda_subtree(element: ET.Element, source_prefix: str, target_prefix: str) -> None:
    source = f"{source_prefix}_"
    target = f"{target_prefix}_"
    for attr in ["name", "joint", "body1", "body2", "tendon", "site"]:
        if attr in element.attrib:
            element.attrib[attr] = element.attrib[attr].replace(source, target)
    for child in element:
        _rename_panda_subtree(child, source_prefix, target_prefix)


def _configure_panda_subtree(element: ET.Element, side: str, model: PandaModel, contacts_enabled: bool = False) -> None:
    if element.tag == "joint":
        name = element.attrib.get("name", "")
        for idx in range(7):
            if name == f"{side}_joint{idx + 1}":
                element.set("range", f"{model.q_lower[idx]:.6g} {model.q_upper[idx]:.6g}")
                element.set("armature", "0.05")
                damping = float(element.attrib.get("damping", "0.08"))
                element.set("damping", f"{max(damping, 0.04):.6g}")
                element.set("limited", "true")
                break
    elif element.tag == "geom":
        element.set("contype", "1" if contacts_enabled else "0")
        element.set("conaffinity", "1" if contacts_enabled else "0")
    elif element.tag == "site":
        if element.attrib.get("name") == f"{side}_ee_site":
            element.set("name", f"{side}_ee")
            element.set("type", "sphere")
            element.set("size", "0.035")
            element.set("rgba", "0.05 0.45 0.95 1" if side == "left" else "0.95 0.25 0.18 1")
    for child in element:
        _configure_panda_subtree(child, side, model, contacts_enabled=contacts_enabled)


def _copy_panda_arm_body(source_root: ET.Element, side: str, model: PandaModel, contacts_enabled: bool = False) -> ET.Element:
    source_body = source_root.find(".//body[@name='mj_left_link0']")
    if source_body is None:
        raise KeyError("mj_left_link0 not found in Franka reference MJCF")
    body = copy.deepcopy(source_body)
    _remove_matching_children(body)
    _rename_panda_subtree(body, "mj_left", side)
    body.set("pos", "0 0 0")
    body.set("quat", "1 0 0 0")
    _configure_panda_subtree(body, side, model, contacts_enabled=contacts_enabled)
    return body


def _proxy_rgba(name: str) -> str:
    if "platform" in name or "wheel" in name:
        return "0.18 0.20 0.22 1"
    if "base" in name:
        return "0.32 0.35 0.38 1"
    if "torso" in name or "shoulder_pad" in name:
        return "0.50 0.53 0.55 1"
    if "head" in name:
        return "0.76 0.78 0.78 1"
    return "0.05 0.05 0.05 1"


def _add_garmi_body_proxy_geoms(base: ET.Element, contacts_enabled: bool = False, mounts: dict[str, Any] | None = None) -> None:
    for proxy in garmi_body_proxies(mounts):
        common = {
            "name": proxy.name,
            "rgba": _proxy_rgba(proxy.name),
            "mass": "1",
            "contype": "1" if contacts_enabled else "0",
            "conaffinity": "1" if contacts_enabled else "0",
        }
        if isinstance(proxy, BoxProxy):
            ET.SubElement(
                base,
                "geom",
                common | {"type": "box", "pos": _vec(proxy.center), "size": _vec(proxy.half_extents)},
            )
        elif isinstance(proxy, SphereProxy):
            ET.SubElement(
                base,
                "geom",
                common | {"type": "sphere", "pos": _vec(proxy.center), "size": f"{proxy.radius:.8g}"},
            )
        elif isinstance(proxy, CapsuleProxy):
            ET.SubElement(
                base,
                "geom",
                common | {"type": "capsule", "fromto": f"{_vec(proxy.p0)} {_vec(proxy.p1)}", "size": f"{proxy.radius:.8g}"},
            )


def _environment_spec(cfg: dict[str, Any], task_metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Parametric per-category scene proxies from the task workspace bbox.

    These proxies make expected end-effector contact simulatable. Placement is
    derived from the recorded workspace bounding box and is provisional; use
    environment.overrides in the profile config for per-task adjustments.
    """
    env_cfg = cfg.get("environment", {}) or {}
    if not bool(env_cfg.get("enabled", False)) or not task_metadata:
        return []
    bbox = task_metadata.get("workspace_bbox_m") or {}
    if "min" not in bbox or "max" not in bbox:
        return []
    lo = np.asarray(bbox["min"], dtype=float)
    hi = np.asarray(bbox["max"], dtype=float)
    center = 0.5 * (lo + hi)
    category = str(task_metadata.get("category", ""))
    params = dict((env_cfg.get("categories", {}) or {}).get(category, {}) or {})
    task_name = str(task_metadata.get("task_name", ""))
    override = (env_cfg.get("overrides", {}) or {}).get(task_name, {}) or {}
    for key, value in override.items():
        params[key] = {**params.get(key, {}), **value} if isinstance(value, dict) else value

    geoms: list[dict[str, Any]] = []
    table = params.get("table")
    if isinstance(table, dict):
        margin = float(table.get("margin_xy_m", 0.15))
        top_offset = float(table.get("top_offset_m", 0.03))
        thickness = float(table.get("thickness_m", 0.04))
        top_z = float(lo[2]) - top_offset
        half = np.array(
            [0.5 * (hi[0] - lo[0]) + margin, 0.5 * (hi[1] - lo[1]) + margin, 0.5 * thickness],
            dtype=float,
        )
        geoms.append(
            {
                "name": "env_table",
                "type": "box",
                "pos": np.array([center[0], center[1], top_z - 0.5 * thickness], dtype=float),
                "size": half,
                "rgba": "0.55 0.42 0.28 1",
            }
        )
    patient = params.get("patient")
    if isinstance(patient, dict):
        radius = float(patient.get("radius_m", 0.14))
        seat_height = float(patient.get("seat_height_m", 0.45))
        top_z = max(float(hi[2]), seat_height + 0.3)
        geoms.append(
            {
                "name": "env_patient",
                "type": "capsule",
                "fromto": np.array([center[0], center[1], seat_height, center[0], center[1], top_z], dtype=float),
                "radius": radius,
                "rgba": "0.85 0.72 0.60 1",
            }
        )
        geoms.append(
            {
                "name": "env_seat",
                "type": "box",
                "pos": np.array([center[0], center[1], 0.5 * seat_height], dtype=float),
                "size": np.array([0.25, 0.25, 0.5 * seat_height], dtype=float),
                "rgba": "0.35 0.35 0.40 1",
            }
        )
    return geoms


def _add_environment_geoms(world: ET.Element, cfg: dict[str, Any], task_metadata: dict[str, Any] | None, contacts_enabled: bool) -> None:
    for spec in _environment_spec(cfg, task_metadata):
        attrs = {
            "name": str(spec["name"]),
            "type": str(spec["type"]),
            "rgba": str(spec["rgba"]),
            "contype": "1" if contacts_enabled else "0",
            "conaffinity": "1" if contacts_enabled else "0",
        }
        if spec["type"] == "box":
            attrs["pos"] = _vec(spec["pos"])
            attrs["size"] = _vec(spec["size"])
        elif spec["type"] == "capsule":
            fromto = np.asarray(spec["fromto"], dtype=float)
            attrs["fromto"] = f"{_vec(fromto[:3])} {_vec(fromto[3:])}"
            attrs["size"] = f"{float(spec['radius']):.8g}"
        ET.SubElement(world, "geom", attrs)


def build_baseline_mjcf(cfg: dict[str, Any], output: Path, task_metadata: dict[str, Any] | None = None) -> Path:
    model = PandaModel.from_config(cfg)
    contacts_enabled = bool(cfg.get("simulation", {}).get("enable_contacts", False))
    source_path = model.reference_mjcf
    source_root = ET.parse(source_path).getroot()
    root = ET.Element("mujoco", {"model": str(cfg.get("baseline", {}).get("name", "garmi_dual_panda_baseline"))})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"timestep": "0.002", "iterations": "40", "solver": "Newton"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})
    _copy_source_default_and_assets(root, source_root, source_path)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"name": "top", "pos": "0 0 3.0"})
    ET.SubElement(world, "camera", {"name": "overview", "pos": "4.6 -4.6 3.6", "xyaxes": "0.707 0.707 0 -0.35 0.35 0.87"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "pos": "0 0 -0.21", "size": "4 4 0.02", "rgba": "0.93 0.94 0.92 1"})
    _add_environment_geoms(world, cfg, task_metadata, contacts_enabled)

    base = ET.SubElement(world, "body", {"name": "base", "pos": "0 0 0"})
    ET.SubElement(base, "joint", {"name": "base_x", "type": "slide", "axis": "1 0 0", "range": "-4 4"})
    ET.SubElement(base, "joint", {"name": "base_y", "type": "slide", "axis": "0 1 0", "range": "-4 4"})
    ET.SubElement(base, "joint", {"name": "base_yaw", "type": "hinge", "axis": "0 0 1", "range": "-6.283 6.283"})
    ET.SubElement(base, "camera", {"name": "garmi_follow", "pos": "4.2 -4.2 3.3", "xyaxes": "0.707 0.707 0 -0.35 0.35 0.87"})
    _add_garmi_body_proxy_geoms(base, contacts_enabled=contacts_enabled, mounts=cfg.get("panda", {}).get("mounts"))

    for side in ["left", "right"]:
        shoulder = model.shoulder_local(side)
        shoulder_quat = quat_wxyz_from_rotation(model.mount_rotations[side])
        shoulder_body = ET.SubElement(
            base,
            "body",
            {"name": f"{side}_shoulder", "pos": _vec(shoulder), "quat": _vec(shoulder_quat), "gravcomp": "1"},
        )
        ET.SubElement(shoulder_body, "geom", {"name": f"{side}_shoulder_mount", "type": "sphere", "size": "0.055", "rgba": "0.1 0.1 0.1 1", "mass": "1"})
        shoulder_body.append(_copy_panda_arm_body(source_root, side, model, contacts_enabled=contacts_enabled))

    actuator = ET.SubElement(root, "actuator")
    for name, limit in [("base_x", 1800), ("base_y", 1800), ("base_yaw", 750)]:
        ET.SubElement(actuator, "motor", {"name": f"{name}_motor", "joint": name, "gear": "1", "forcerange": f"{-limit} {limit}"})
    for side in ["left", "right"]:
        for idx, limit in enumerate(model.torque_limits_nm):
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": f"{side}_joint{idx + 1}_motor",
                    "joint": f"{side}_joint{idx + 1}",
                    "gear": "1",
                    "forcerange": f"{-float(limit):.6g} {float(limit):.6g}",
                },
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _qpos_indices(model: mujoco.MjModel) -> dict[str, int]:
    return {model.joint(i).name: int(model.jnt_qposadr[i]) for i in range(model.njnt)}


def _dof_indices(model: mujoco.MjModel) -> dict[str, int]:
    return {model.joint(i).name: int(model.jnt_dofadr[i]) for i in range(model.njnt)}


def _site_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))


def compute_baseline_reference(task_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    task = load_task(task_name)
    baseline = evaluate_task_baseline(task, cfg)
    base_pose = np.asarray(baseline["base_motion"]["pose"], dtype=float)
    refs: dict[str, Any] = {
        "task_name": task_name,
        "time": task.time,
        "base_pose": base_pose,
        "active_sides": tuple(task.active_sides()),
    }
    for side in ["left", "right"]:
        side_result = baseline["side"][side]
        refs[f"{side}_q"] = np.asarray(side_result["q"], dtype=float)
        refs[f"{side}_ik_error"] = np.asarray(side_result["errors"], dtype=float)
        refs[f"{side}_ik_orientation_error"] = np.asarray(side_result["orientation_errors"], dtype=float)
        refs[f"{side}_target"] = np.asarray(side_result["target_positions"], dtype=float)
        refs[f"{side}_target_rot"] = np.asarray(side_result["target_rotations"], dtype=float)
        refs[f"{side}_active"] = bool(side_result.get("active", side in refs["active_sides"]))
    refs["baseline_metrics"] = baseline["metrics"]
    return refs


def _trajectory_angular_velocity(rotations: np.ndarray, time: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=float)
    time = np.asarray(time, dtype=float)
    omega = np.zeros((len(rotations), 3), dtype=float)
    if len(rotations) < 2:
        return omega
    for idx in range(len(rotations)):
        if idx == 0:
            dt = max(float(time[1] - time[0]), 1e-6)
            omega[idx] = rotation_error_vector(rotations[1], rotations[0]) / dt
        elif idx == len(rotations) - 1:
            dt = max(float(time[-1] - time[-2]), 1e-6)
            omega[idx] = rotation_error_vector(rotations[-1], rotations[-2]) / dt
        else:
            dt = max(float(time[idx + 1] - time[idx - 1]), 1e-6)
            omega[idx] = rotation_error_vector(rotations[idx + 1], rotations[idx - 1]) / dt
    return omega


def _arm_q_and_qvel(data: mujoco.MjData, qpos: dict[str, int], dof: dict[str, int], side: str) -> tuple[np.ndarray, np.ndarray]:
    q = np.array([data.qpos[qpos[f"{side}_joint{i + 1}"]] for i in range(7)], dtype=float)
    qvel = np.array([data.qvel[dof[f"{side}_joint{i + 1}"]] for i in range(7)], dtype=float)
    return q, qvel


def _site_jacobians(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp, jacr


def _damped_nullspace_projector(jac: np.ndarray, damping: float) -> np.ndarray:
    lhs = jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0])
    try:
        pinv = jac.T @ np.linalg.solve(lhs, np.eye(lhs.shape[0]))
    except np.linalg.LinAlgError:
        pinv = jac.T @ np.linalg.pinv(lhs)
    return np.eye(jac.shape[1]) - pinv @ jac


def _controller_config(cfg: dict[str, Any]) -> dict[str, Any]:
    controller = dict(cfg.get("controller", {}))
    controller.setdefault("mode", "operational_space_impedance")
    controller.setdefault("position_kp_npm", [1200.0, 1200.0, 1200.0])
    controller.setdefault("position_kd_nspm", [110.0, 110.0, 110.0])
    controller.setdefault("orientation_kp_nmprad", [62.0, 62.0, 62.0])
    controller.setdefault("orientation_kd_nmsprad", [7.5, 7.5, 7.5])
    controller.setdefault("max_task_force_n", 260.0)
    controller.setdefault("max_task_torque_nm", 55.0)
    base_pd = dict(controller.get("base_pd", {}))
    base_pd.setdefault("kp", [2400.0, 2400.0, 850.0])
    base_pd.setdefault("kd", [520.0, 520.0, 160.0])
    controller["base_pd"] = base_pd
    controller.setdefault("nullspace_kp_nmprad", [20.0, 18.0, 15.0, 12.0, 7.0, 5.0, 3.0])
    controller.setdefault("nullspace_kd_nmsprad", [3.0, 2.6, 2.2, 1.8, 1.0, 0.7, 0.45])
    controller.setdefault("nullspace_damping", 0.035)
    controller.setdefault("bias_compensation", True)
    return controller


def _base_pd_gains(controller: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    base_pd = dict(controller.get("base_pd", {}))
    kp = np.asarray(base_pd.get("kp", [2400.0, 2400.0, 850.0]), dtype=float)
    kd = np.asarray(base_pd.get("kd", [520.0, 520.0, 160.0]), dtype=float)
    if kp.shape != (3,):
        kp = np.asarray([2400.0, 2400.0, 850.0], dtype=float)
    if kd.shape != (3,):
        kd = np.asarray([520.0, 520.0, 160.0], dtype=float)
    return kp, kd


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > float(max_norm) > 0.0:
        return vector * (float(max_norm) / norm)
    return vector


def _interpolation_window(time: np.ndarray, t: float) -> tuple[int, int, float]:
    if len(time) <= 1 or t <= float(time[0]):
        return 0, 0, 0.0
    if t >= float(time[-1]):
        last = len(time) - 1
        return last, last, 0.0
    upper = int(np.searchsorted(time, t, side="right"))
    lower = max(0, upper - 1)
    upper = min(upper, len(time) - 1)
    dt = max(float(time[upper] - time[lower]), 1e-9)
    alpha = float(np.clip((float(t) - float(time[lower])) / dt, 0.0, 1.0))
    return lower, upper, alpha


def _interpolate_array(values: np.ndarray, lower: int, upper: int, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if lower == upper:
        return values[lower].copy()
    return ((1.0 - alpha) * values[lower] + alpha * values[upper]).copy()


def _interpolate_rotation(rotations: np.ndarray, lower: int, upper: int, alpha: float) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=float)
    if lower == upper:
        return rotations[lower].copy()
    delta = Rotation.from_rotvec(alpha * rotation_error_vector(rotations[upper], rotations[lower])).as_matrix()
    return delta @ rotations[lower]


def _reference_sample(
    refs: dict[str, Any],
    target_velocities: dict[str, np.ndarray],
    target_omegas: dict[str, np.ndarray],
    task_time: np.ndarray,
    t: float,
) -> dict[str, Any]:
    lower, upper, alpha = _interpolation_window(task_time, t)
    sample: dict[str, Any] = {
        "ref_index": lower,
        "ref_upper_index": upper,
        "ref_alpha": alpha,
        "base_pose": _interpolate_array(refs["base_pose"], lower, upper, alpha),
    }
    for side in ["left", "right"]:
        sample[f"{side}_q"] = _interpolate_array(refs[f"{side}_q"], lower, upper, alpha)
        sample[f"{side}_target"] = _interpolate_array(refs[f"{side}_target"], lower, upper, alpha)
        sample[f"{side}_target_rot"] = _interpolate_rotation(refs[f"{side}_target_rot"], lower, upper, alpha)
        sample[f"{side}_target_vel"] = _interpolate_array(target_velocities[side], lower, upper, alpha)
        sample[f"{side}_target_omega"] = _interpolate_array(target_omegas[side], lower, upper, alpha)
    return sample


def _active_sides_from_refs(refs: dict[str, Any]) -> tuple[str, ...]:
    sides = tuple(str(side) for side in refs.get("active_sides", ("left", "right")) if str(side) in {"left", "right"})
    return sides or ("left", "right")


def _active_summary_value(summary: dict[str, Any], sides: tuple[str, ...], key_template: str, reducer: str) -> float:
    values = []
    for side in sides:
        try:
            value = float(summary[key_template.format(side=side)])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    if reducer == "max":
        return float(max(values))
    return float(np.mean(values))


def _simulation_pass_status(summary: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, str]:
    success_cfg = cfg.get("success", {})
    validation = cfg.get("simulation_validation", {}) or cfg.get("simulation", {}).get("validation", {})
    mean_pos_limit = float(validation.get("mean_position_error_m", success_cfg.get("mean_position_error_m", 0.05)))
    max_pos_limit = float(validation.get("max_position_error_m", success_cfg.get("max_position_error_m", 0.12)))
    mean_rot_limit = float(validation.get("mean_orientation_error_rad", success_cfg.get("mean_orientation_error_rad", 0.90)))
    max_rot_limit = float(validation.get("max_orientation_error_rad", success_cfg.get("max_orientation_error_rad", 1.75)))
    max_clip = float(validation.get("max_joint_torque_clip_fraction", 0.0))
    # Only unexpected contacts fail validation; expected end-effector contact
    # with the environment is allowed within a force budget.
    # max_contact_count is accepted as an alias of max_unexpected_contact_count.
    max_unexpected = int(validation.get("max_unexpected_contact_count", validation.get("max_contact_count", 0)))
    force_budget = float(validation.get("expected_contact_force_budget_n", float("inf")))
    require_no_exception = bool(validation.get("require_no_exception", True))

    failures = []
    if require_no_exception and not bool(summary.get("simulation_ok", True)):
        failures.append("simulation_exception")
    if float(summary.get("mean_tracking_error_m", float("inf"))) > mean_pos_limit:
        failures.append("mean_tracking_error")
    if float(summary.get("max_tracking_error_m", float("inf"))) > max_pos_limit:
        failures.append("max_tracking_error")
    if float(summary.get("mean_orientation_error_rad", float("inf"))) > mean_rot_limit:
        failures.append("mean_orientation_error")
    if float(summary.get("max_orientation_error_rad", float("inf"))) > max_rot_limit:
        failures.append("max_orientation_error")
    if float(summary.get("max_joint_torque_clip_fraction", float("inf"))) > max_clip:
        failures.append("joint_torque_clipping")
    unexpected = summary.get("max_unexpected_contact_count")
    if unexpected is None:
        unexpected = summary.get("max_contact_count", 0)
    if int(unexpected) > max_unexpected:
        failures.append("unexpected_contacts")
    if float(summary.get("max_expected_contact_force_n", 0.0)) > force_budget:
        failures.append("expected_contact_force_budget")
    return not failures, ";".join(failures) if failures else "ok"


def _operational_space_arm_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: dict[str, int],
    dof: dict[str, int],
    side: str,
    site_id: int,
    q_ref: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    target_vel: np.ndarray,
    target_omega: np.ndarray,
    controller: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    arm_dofs = np.array([dof[f"{side}_joint{i + 1}"] for i in range(7)], dtype=int)
    q_actual, qvel_actual = _arm_q_and_qvel(data, qpos, dof, side)
    site_pos = data.site_xpos[site_id].copy()
    site_rot = data.site_xmat[site_id].reshape(3, 3).copy()
    jacp, jacr = _site_jacobians(model, data, site_id)
    linear_vel = jacp @ data.qvel
    angular_vel = jacr @ data.qvel

    pos_error = target_pos - site_pos
    rot_error = rotation_error_vector(target_rot, site_rot)

    kp_pos = np.asarray(controller["position_kp_npm"], dtype=float)
    kd_pos = np.asarray(controller["position_kd_nspm"], dtype=float)
    kp_rot = np.asarray(controller["orientation_kp_nmprad"], dtype=float)
    kd_rot = np.asarray(controller["orientation_kd_nmsprad"], dtype=float)
    task_force = kp_pos * pos_error + kd_pos * (target_vel - linear_vel)
    task_torque = kp_rot * rot_error + kd_rot * (target_omega - angular_vel)
    task_force = _clip_norm(task_force, float(controller["max_task_force_n"]))
    task_torque = _clip_norm(task_torque, float(controller["max_task_torque_nm"]))

    jacp_arm = jacp[:, arm_dofs]
    jacr_arm = jacr[:, arm_dofs]
    tau_task = jacp_arm.T @ task_force + jacr_arm.T @ task_torque

    null_kp = np.asarray(controller["nullspace_kp_nmprad"], dtype=float)
    null_kd = np.asarray(controller["nullspace_kd_nmsprad"], dtype=float)
    posture_tau = null_kp * (q_ref - q_actual) - null_kd * qvel_actual
    jac_task = np.vstack([jacp_arm, 0.35 * jacr_arm])
    null_projector = _damped_nullspace_projector(jac_task, float(controller["nullspace_damping"]))
    tau = tau_task + null_projector @ posture_tau
    if bool(controller.get("bias_compensation", True)):
        # qfrc_bias contains gravity, but the Panda bodies carry gravcomp=1
        # (MuJoCo compensates gravity internally, like the real Franka).
        # Subtract the gravcomp share so gravity is not compensated twice.
        bias = data.qfrc_bias[arm_dofs].copy()
        gravcomp = getattr(data, "qfrc_gravcomp", None)
        if gravcomp is not None:
            bias = bias - gravcomp[arm_dofs]
        tau = tau + bias
    return tau, {
        f"{side}_task_force_norm_n": float(np.linalg.norm(task_force)),
        f"{side}_task_torque_norm_nm": float(np.linalg.norm(task_torque)),
        f"{side}_pos_error_m": float(np.linalg.norm(pos_error)),
        f"{side}_rot_error_rad": float(np.linalg.norm(rot_error)),
    }


_GARMI_FLOOR_TOKENS = ("wheel", "platform", "base", "caster", "tire")


def _classify_contacts(model: mujoco.MjModel, data: mujoco.MjData, geom_names: list[str]) -> dict[str, float]:
    """Split active contacts into expected / unexpected / ignored.

    Expected: end-effector geoms (hand/finger) against environment proxies.
    Ignored: GARMI platform geoms against the floor.
    Unexpected: everything else (arm vs. GARMI body, arm vs. arm, non-EE links
    vs. environment, ...).
    Returns counts and the summed normal-force magnitude on expected contacts.
    """
    expected = 0
    unexpected = 0
    ignored = 0
    expected_force = 0.0
    force6 = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        contact = data.contact[i]
        name1 = geom_names[int(contact.geom1)].lower()
        name2 = geom_names[int(contact.geom2)].lower()

        def _is_env(name: str) -> bool:
            return name.startswith("env_")

        def _is_ee(name: str) -> bool:
            return "hand" in name or "finger" in name

        def _is_floor_pair(a: str, b: str) -> bool:
            return a == "floor" and any(token in b for token in _GARMI_FLOOR_TOKENS)

        if _is_floor_pair(name1, name2) or _is_floor_pair(name2, name1):
            ignored += 1
            continue
        if (_is_env(name1) and _is_ee(name2)) or (_is_env(name2) and _is_ee(name1)):
            expected += 1
            mujoco.mj_contactForce(model, data, i, force6)
            expected_force += float(np.linalg.norm(force6[:3]))
            continue
        unexpected += 1
    return {
        "expected_contact_count": float(expected),
        "unexpected_contact_count": float(unexpected),
        "ignored_contact_count": float(ignored),
        "expected_contact_force_n": expected_force,
    }


def _auto_duration_scale(refs: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Playback slowdown for the impedance simulation.

    The kinematic evaluation allows executing a task up to
    success.max_time_scale slower than the human recording. The simulation
    inherits the time scale chosen by the kinematic stage, times an optional
    simulation.playback_slowdown factor.
    """
    metrics = refs.get("baseline_metrics", {}) or {}
    scale = max(1.0, float(metrics.get("time_scale", 1.0)))
    return scale * float(cfg.get("simulation", {}).get("playback_slowdown", 1.0))


def simulate_impedance_task(
    task_name: str,
    output_dir: Path,
    cfg: dict[str, Any] | None = None,
    duration_scale: float | None = None,
    render: bool = True,
) -> Path:
    cfg = cfg or load_baseline_config()
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_obj = load_task(task_name)
    xml_path = build_baseline_mjcf(cfg, output_dir / "garmi_dual_panda_baseline.xml", task_metadata=task_obj.metadata)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    refs = compute_baseline_reference(task_name, cfg)
    if duration_scale is None:
        duration_scale = _auto_duration_scale(refs, cfg)
    # Contact classification labels: geom name, or the parent body name for the
    # many unnamed collision geoms in the Panda subtrees (e.g. 'left_hand').
    geom_names = []
    for i in range(model.ngeom):
        label = str(model.geom(i).name or "")
        if not label:
            label = str(model.body(int(model.geom_bodyid[i])).name or "")
        geom_names.append(label)
    has_environment = any(name.startswith("env_") for name in geom_names)
    # Measured endpoint force magnitude per reference sample (max over active
    # sides) for the contact-force validity comparison against the simulation.
    active_sides = _active_sides_from_refs(refs)
    measured_force_norm = np.zeros(len(task_obj.time), dtype=float)
    for side in active_sides:
        measured_force_norm = np.maximum(measured_force_norm, np.linalg.norm(task_obj.side_force(side), axis=1))
    qpos = _qpos_indices(model)
    dof = _dof_indices(model)
    left_site = _site_id(model, "left_ee")
    right_site = _site_id(model, "right_ee")
    ctrl_names = [model.actuator(i).name for i in range(model.nu)]
    actuator_index = {name: idx for idx, name in enumerate(ctrl_names)}

    sim_dt = float(model.opt.timestep)
    task_time = np.asarray(refs["time"], dtype=float)
    sim_duration = float(task_time[-1] - task_time[0]) * float(duration_scale)
    n_steps = max(1, int(sim_duration / sim_dt))
    panda_cfg = cfg["panda"]
    torque_limits = np.asarray(panda_cfg["torque_limits_nm"], dtype=float)
    controller = _controller_config(cfg)
    target_velocities = {
        "left": finite_gradient(refs["left_target"], task_time),
        "right": finite_gradient(refs["right_target"], task_time),
    }
    target_omegas = {
        "left": _trajectory_angular_velocity(refs["left_target_rot"], task_time),
        "right": _trajectory_angular_velocity(refs["right_target_rot"], task_time),
    }
    base_kp, base_kd = _base_pd_gains(controller)

    # Initialize at first reference pose.
    data.qpos[qpos["base_x"]] = refs["base_pose"][0, 0]
    data.qpos[qpos["base_y"]] = refs["base_pose"][0, 1]
    data.qpos[qpos["base_yaw"]] = refs["base_pose"][0, 2]
    for side in ["left", "right"]:
        for idx in range(7):
            data.qpos[qpos[f"{side}_joint{idx + 1}"]] = refs[f"{side}_q"][0, idx]
    mujoco.mj_forward(model, data)

    rows = []
    frames = []
    render_every = max(1, int(1.0 / (30.0 * sim_dt)))
    renderer = None
    render_camera = str(cfg.get("simulation", {}).get("render_camera", "garmi_follow"))
    if render:
        renderer = mujoco.Renderer(model, width=1280, height=720)

    for step in range(n_steps):
        t = min(task_time[-1], task_time[0] + step * sim_dt / max(duration_scale, 1e-9))
        sample = _reference_sample(refs, target_velocities, target_omegas, task_time, t)
        desired_base = sample["base_pose"]
        for name, desired, gain_p, gain_d in [
            ("base_x", desired_base[0], base_kp[0], base_kd[0]),
            ("base_y", desired_base[1], base_kp[1], base_kd[1]),
            ("base_yaw", desired_base[2], base_kp[2], base_kd[2]),
        ]:
            qi = qpos[name]
            di = dof[name]
            data.ctrl[actuator_index[f"{name}_motor"]] = gain_p * (desired - data.qpos[qi]) - gain_d * data.qvel[di]
        arm_metrics: dict[str, float] = {}
        for side in ["left", "right"]:
            site_id = left_site if side == "left" else right_site
            tau, metrics = _operational_space_arm_torque(
                model,
                data,
                qpos,
                dof,
                side,
                site_id,
                sample[f"{side}_q"],
                sample[f"{side}_target"],
                sample[f"{side}_target_rot"],
                sample[f"{side}_target_vel"],
                sample[f"{side}_target_omega"],
                controller,
            )
            clipped_tau = np.clip(tau, -torque_limits, torque_limits)
            arm_metrics.update(metrics)
            arm_metrics[f"{side}_max_abs_joint_torque_nm"] = float(np.max(np.abs(clipped_tau)))
            arm_metrics[f"{side}_mean_abs_joint_torque_nm"] = float(np.mean(np.abs(clipped_tau)))
            arm_metrics[f"{side}_joint_torque_clip_fraction"] = float(np.mean(np.abs(tau) > torque_limits))
            for idx in range(7):
                joint = f"{side}_joint{idx + 1}"
                data.ctrl[actuator_index[f"{joint}_motor"]] = float(clipped_tau[idx])
        mujoco.mj_step(model, data)
        if step % max(1, int(0.02 / sim_dt)) == 0 or step == n_steps - 1:
            left_pos = data.site_xpos[left_site].copy()
            right_pos = data.site_xpos[right_site].copy()
            left_rot = data.site_xmat[left_site].reshape(3, 3).copy()
            right_rot = data.site_xmat[right_site].reshape(3, 3).copy()
            base_actual = np.array([data.qpos[qpos["base_x"]], data.qpos[qpos["base_y"]], data.qpos[qpos["base_yaw"]]], dtype=float)
            left_q_actual = np.array([data.qpos[qpos[f"left_joint{i + 1}"]] for i in range(7)], dtype=float)
            right_q_actual = np.array([data.qpos[qpos[f"right_joint{i + 1}"]] for i in range(7)], dtype=float)
            contact_distances = [float(data.contact[i].dist) for i in range(data.ncon)]
            contact_classes = _classify_contacts(model, data, geom_names)
            measured_now = float(
                (1.0 - sample["ref_alpha"]) * measured_force_norm[int(sample["ref_index"])]
                + sample["ref_alpha"] * measured_force_norm[int(sample["ref_upper_index"])]
            ) if len(measured_force_norm) else 0.0
            rows.append(
                {
                    "time": float(data.time),
                    "ref_index": int(sample["ref_index"]),
                    "ref_upper_index": int(sample["ref_upper_index"]),
                    "ref_alpha": float(sample["ref_alpha"]),
                    "left_tracking_error_m": float(np.linalg.norm(left_pos - sample["left_target"])),
                    "right_tracking_error_m": float(np.linalg.norm(right_pos - sample["right_target"])),
                    "left_orientation_error_rad": float(np.linalg.norm(rotation_error_vector(sample["left_target_rot"], left_rot))),
                    "right_orientation_error_rad": float(np.linalg.norm(rotation_error_vector(sample["right_target_rot"], right_rot))),
                    "base_xy_error_m": float(np.linalg.norm(base_actual[:2] - desired_base[:2])),
                    "base_yaw_error_rad": float(abs(base_actual[2] - desired_base[2])),
                    "left_q_rms_error_rad": float(np.sqrt(np.mean((left_q_actual - sample["left_q"]) ** 2))),
                    "right_q_rms_error_rad": float(np.sqrt(np.mean((right_q_actual - sample["right_q"]) ** 2))),
                    "max_abs_ctrl": float(np.max(np.abs(data.ctrl))),
                    "mean_abs_ctrl": float(np.mean(np.abs(data.ctrl))),
                    "contact_count": int(data.ncon),
                    "min_contact_distance_m": float(np.min(contact_distances)) if contact_distances else float("inf"),
                    "expected_contact_count": int(contact_classes["expected_contact_count"]),
                    "unexpected_contact_count": int(contact_classes["unexpected_contact_count"]),
                    "ignored_contact_count": int(contact_classes["ignored_contact_count"]),
                    "expected_contact_force_n": float(contact_classes["expected_contact_force_n"]),
                    "measured_force_n": measured_now,
                    "controller_mode": str(controller["mode"]),
                    **arm_metrics,
                }
            )
        if renderer is not None and step % render_every == 0:
            renderer.update_scene(data, camera=render_camera)
            frames.append(renderer.render())

    if renderer is not None:
        renderer.close()
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "simulation_metrics.csv", index=False)
    if frames:
        imageio.mimsave(output_dir / "impedance_playback.mp4", frames, fps=30)
    active_sides = _active_sides_from_refs(refs)
    summary = {
        "task_name": task_name,
        "baseline_profile": cfg.get("_profile", "baseline"),
        "baseline_name": cfg.get("baseline", {}).get("name", "unnamed"),
        "base_motion_mode": cfg.get("base", {}).get("motion_mode", "free_holonomic"),
        "controller_mode": str(controller["mode"]),
        "active_sides": ",".join(active_sides),
        "xml": str(xml_path),
        "frames": len(frames),
        "duration_s": float(sim_duration),
        "mean_left_tracking_error_m": float(df["left_tracking_error_m"].mean()) if len(df) else float("nan"),
        "mean_right_tracking_error_m": float(df["right_tracking_error_m"].mean()) if len(df) else float("nan"),
        "max_left_tracking_error_m": float(df["left_tracking_error_m"].max()) if len(df) else float("nan"),
        "max_right_tracking_error_m": float(df["right_tracking_error_m"].max()) if len(df) else float("nan"),
        "mean_left_orientation_error_rad": float(df["left_orientation_error_rad"].mean()) if len(df) else float("nan"),
        "mean_right_orientation_error_rad": float(df["right_orientation_error_rad"].mean()) if len(df) else float("nan"),
        "max_left_orientation_error_rad": float(df["left_orientation_error_rad"].max()) if len(df) else float("nan"),
        "max_right_orientation_error_rad": float(df["right_orientation_error_rad"].max()) if len(df) else float("nan"),
        "mean_base_xy_error_m": float(df["base_xy_error_m"].mean()) if len(df) else float("nan"),
        "mean_left_q_rms_error_rad": float(df["left_q_rms_error_rad"].mean()) if len(df) else float("nan"),
        "mean_right_q_rms_error_rad": float(df["right_q_rms_error_rad"].mean()) if len(df) else float("nan"),
        "max_abs_ctrl": float(df["max_abs_ctrl"].max()) if len(df) else float("nan"),
        "max_contact_count": int(df["contact_count"].max()) if "contact_count" in df and len(df) else 0,
        "min_contact_distance_m": float(df["min_contact_distance_m"].replace([np.inf, -np.inf], np.nan).min()) if "min_contact_distance_m" in df and len(df) else float("nan"),
        "mean_left_task_force_norm_n": float(df["left_task_force_norm_n"].mean()) if "left_task_force_norm_n" in df and len(df) else float("nan"),
        "mean_right_task_force_norm_n": float(df["right_task_force_norm_n"].mean()) if "right_task_force_norm_n" in df and len(df) else float("nan"),
        "max_left_task_force_norm_n": float(df["left_task_force_norm_n"].max()) if "left_task_force_norm_n" in df and len(df) else float("nan"),
        "max_right_task_force_norm_n": float(df["right_task_force_norm_n"].max()) if "right_task_force_norm_n" in df and len(df) else float("nan"),
        "mean_left_task_torque_norm_nm": float(df["left_task_torque_norm_nm"].mean()) if "left_task_torque_norm_nm" in df and len(df) else float("nan"),
        "mean_right_task_torque_norm_nm": float(df["right_task_torque_norm_nm"].mean()) if "right_task_torque_norm_nm" in df and len(df) else float("nan"),
        "max_left_task_torque_norm_nm": float(df["left_task_torque_norm_nm"].max()) if "left_task_torque_norm_nm" in df and len(df) else float("nan"),
        "max_right_task_torque_norm_nm": float(df["right_task_torque_norm_nm"].max()) if "right_task_torque_norm_nm" in df and len(df) else float("nan"),
        "max_left_joint_torque_clip_fraction": float(df["left_joint_torque_clip_fraction"].max()) if "left_joint_torque_clip_fraction" in df and len(df) else float("nan"),
        "max_right_joint_torque_clip_fraction": float(df["right_joint_torque_clip_fraction"].max()) if "right_joint_torque_clip_fraction" in df and len(df) else float("nan"),
        "baseline_metrics": refs["baseline_metrics"],
    }
    # Contact semantics and contact-force validity check.
    summary["duration_scale"] = float(duration_scale)
    summary["environment_proxies"] = bool(has_environment)
    summary["max_unexpected_contact_count"] = int(df["unexpected_contact_count"].max()) if "unexpected_contact_count" in df and len(df) else 0
    summary["max_expected_contact_count"] = int(df["expected_contact_count"].max()) if "expected_contact_count" in df and len(df) else 0
    summary["max_expected_contact_force_n"] = float(df["expected_contact_force_n"].max()) if "expected_contact_force_n" in df and len(df) else 0.0
    if has_environment and "expected_contact_force_n" in df and "measured_force_n" in df and len(df):
        simulated = df["expected_contact_force_n"].astype(float).to_numpy()
        measured = df["measured_force_n"].astype(float).to_numpy()
        summary["contact_force_rmse_n"] = float(np.sqrt(np.mean((simulated - measured) ** 2)))
        summary["contact_force_peak_simulated_n"] = float(np.max(simulated))
        summary["contact_force_peak_measured_n"] = float(np.max(measured))
    summary["mean_tracking_error_m"] = _active_summary_value(summary, active_sides, "mean_{side}_tracking_error_m", "mean")
    summary["max_tracking_error_m"] = _active_summary_value(summary, active_sides, "max_{side}_tracking_error_m", "max")
    summary["mean_orientation_error_rad"] = _active_summary_value(
        summary, active_sides, "mean_{side}_orientation_error_rad", "mean"
    )
    summary["max_orientation_error_rad"] = _active_summary_value(summary, active_sides, "max_{side}_orientation_error_rad", "max")
    summary["max_joint_torque_clip_fraction"] = _active_summary_value(
        summary, active_sides, "max_{side}_joint_torque_clip_fraction", "max"
    )
    summary["simulation_ok"] = True
    passed, failure_reasons = _simulation_pass_status(summary, cfg)
    summary["simulation_passed"] = bool(passed)
    summary["simulation_failure_reasons"] = failure_reasons
    write_json(output_dir / "summary.json", summary)
    _write_simulation_report(output_dir, summary)
    return output_dir


def _initialize_mujoco_reference_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: dict[str, int], refs: dict[str, Any]) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[qpos["base_x"]] = refs["base_pose"][0, 0]
    data.qpos[qpos["base_y"]] = refs["base_pose"][0, 1]
    data.qpos[qpos["base_yaw"]] = refs["base_pose"][0, 2]
    for side in ["left", "right"]:
        for idx in range(7):
            data.qpos[qpos[f"{side}_joint{idx + 1}"]] = refs[f"{side}_q"][0, idx]
    mujoco.mj_forward(model, data)


def view_impedance_task(
    task_name: str,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
    realtime_factor: float = 1.0,
    camera: str | None = None,
) -> Path:
    cfg = cfg or load_baseline_config(profile=baseline_profile)
    ensure_dirs()
    slug = _profile_slug(cfg, baseline_profile)
    output_dir = PATHS.runs / timestamp_id(f"interactive_mujoco_{slug}_{task_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = build_baseline_mjcf(cfg, output_dir / "garmi_dual_panda_baseline.xml", task_metadata=load_task(task_name).metadata)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    refs = compute_baseline_reference(task_name, cfg)
    qpos = _qpos_indices(model)
    dof = _dof_indices(model)
    left_site = _site_id(model, "left_ee")
    right_site = _site_id(model, "right_ee")
    ctrl_names = [model.actuator(i).name for i in range(model.nu)]
    actuator_index = {name: idx for idx, name in enumerate(ctrl_names)}

    sim_dt = float(model.opt.timestep)
    task_time = np.asarray(refs["time"], dtype=float)
    n_steps = max(1, int(float(task_time[-1] - task_time[0]) / sim_dt))
    panda_cfg = cfg["panda"]
    torque_limits = np.asarray(panda_cfg["torque_limits_nm"], dtype=float)
    controller = _controller_config(cfg)
    target_velocities = {
        "left": finite_gradient(refs["left_target"], task_time),
        "right": finite_gradient(refs["right_target"], task_time),
    }
    target_omegas = {
        "left": _trajectory_angular_velocity(refs["left_target_rot"], task_time),
        "right": _trajectory_angular_velocity(refs["right_target_rot"], task_time),
    }
    base_kp, base_kd = _base_pd_gains(controller)

    _initialize_mujoco_reference_state(model, data, qpos, refs)
    try:
        from mujoco import viewer as mujoco_viewer
    except Exception as exc:
        raise RuntimeError("mujoco.viewer is not available in this Python environment.") from exc

    camera_name = str(camera or cfg.get("simulation", {}).get("interactive_camera", cfg.get("simulation", {}).get("render_camera", "garmi_follow")))
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = int(camera_id)
        step = 0
        last_wall = time.perf_counter()
        while viewer.is_running():
            if step >= n_steps:
                _initialize_mujoco_reference_state(model, data, qpos, refs)
                step = 0
            t = min(task_time[-1], task_time[0] + step * sim_dt)
            sample = _reference_sample(refs, target_velocities, target_omegas, task_time, t)
            desired_base = sample["base_pose"]
            for name, desired, gain_p, gain_d in [
                ("base_x", desired_base[0], base_kp[0], base_kd[0]),
                ("base_y", desired_base[1], base_kp[1], base_kd[1]),
                ("base_yaw", desired_base[2], base_kp[2], base_kd[2]),
            ]:
                qi = qpos[name]
                di = dof[name]
                data.ctrl[actuator_index[f"{name}_motor"]] = gain_p * (desired - data.qpos[qi]) - gain_d * data.qvel[di]
            for side in ["left", "right"]:
                site_id = left_site if side == "left" else right_site
                tau, _metrics = _operational_space_arm_torque(
                    model,
                    data,
                    qpos,
                    dof,
                    side,
                    site_id,
                    sample[f"{side}_q"],
                    sample[f"{side}_target"],
                    sample[f"{side}_target_rot"],
                    sample[f"{side}_target_vel"],
                    sample[f"{side}_target_omega"],
                    controller,
                )
                clipped_tau = np.clip(tau, -torque_limits, torque_limits)
                for idx in range(7):
                    joint = f"{side}_joint{idx + 1}"
                    data.ctrl[actuator_index[f"{joint}_motor"]] = float(clipped_tau[idx])
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            target_dt = sim_dt / max(float(realtime_factor), 1e-6)
            elapsed = time.perf_counter() - last_wall
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            last_wall = time.perf_counter()
    return output_dir


def _write_simulation_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    metrics = pd.read_csv(output_dir / "simulation_metrics.csv")
    table = pd.DataFrame([summary | {"baseline_metrics": "<nested>"}]).to_html(index=False, float_format=lambda v: f"{v:.5f}")
    plot_html = ""
    effort_html = ""
    try:
        import plotly.express as px
        from plotly.offline import plot

        fig = px.line(
            metrics,
            x="time",
            y=[
                "left_tracking_error_m",
                "right_tracking_error_m",
                "left_orientation_error_rad",
                "right_orientation_error_rad",
            ],
            title="MuJoCo operational-space impedance tracking",
        )
        plot_html = plot(fig, include_plotlyjs="cdn", output_type="div")
        effort_columns = [
            col
            for col in [
                "left_task_force_norm_n",
                "right_task_force_norm_n",
                "left_task_torque_norm_nm",
                "right_task_torque_norm_nm",
                "left_max_abs_joint_torque_nm",
                "right_max_abs_joint_torque_nm",
            ]
            if col in metrics
        ]
        if effort_columns:
            effort_fig = px.line(metrics, x="time", y=effort_columns, title="Task-space command and joint torque usage")
            effort_html = plot(effort_fig, include_plotlyjs=False, output_type="div")
    except Exception:
        plot_html = "<p>Plotly unavailable.</p>"
    video = ""
    if (output_dir / "impedance_playback.mp4").exists():
        video = '<video controls width="960" src="impedance_playback.mp4"></video>'
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MuJoCo Impedance Simulation</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#182027}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7dde2;padding:6px 8px;font-size:13px;text-align:left}}th{{background:#f3f6f8}}</style>
</head><body>
<h1>MuJoCo Impedance Simulation: {summary["task_name"]}</h1>
<p><strong>Baseline profile:</strong> {summary.get("baseline_profile", "baseline")} ({summary.get("baseline_name", "unnamed")})</p>
<h2>Summary</h2>{table}
<h2>Video</h2>{video}
<h2>Tracking</h2>{plot_html}
<h2>Controller Effort</h2>{effort_html}
</body></html>"""
    out = output_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def simulate_task_cli(
    task_name: str,
    paths: ProjectPaths = PATHS,
    render: bool = True,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
) -> Path:
    cfg = cfg or load_baseline_config(paths, profile=baseline_profile)
    slug = _profile_slug(cfg, baseline_profile)
    prefix = f"mujoco_impedance_{task_name}" if slug == "baseline" else f"mujoco_impedance_{slug}_{task_name}"
    output = paths.runs / timestamp_id(prefix)
    return simulate_impedance_task(task_name, output, cfg=cfg, render=render)


def _batch_worker_count(cfg: dict[str, Any], task_count: int, workers: int | None = None) -> int:
    if workers is not None:
        return max(1, min(int(workers), max(1, task_count)))
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


def _simulate_batch_worker(task_name: str, run_dir: str, cfg: dict[str, Any], render: bool) -> dict[str, Any]:
    task_output = Path(run_dir) / task_name
    try:
        simulate_impedance_task(task_name, task_output, cfg=cfg, render=render)
        summary = read_json(task_output / "summary.json")
        summary["simulation_ok"] = True
        summary["error"] = ""
        return summary
    except Exception as exc:
        task_output.mkdir(parents=True, exist_ok=True)
        summary = {
            "task_name": task_name,
            "baseline_profile": cfg.get("_profile", "baseline"),
            "baseline_name": cfg.get("baseline", {}).get("name", "unnamed"),
            "base_motion_mode": cfg.get("base", {}).get("motion_mode", "free_holonomic"),
            "controller_mode": str(_controller_config(cfg)["mode"]),
            "simulation_ok": False,
            "simulation_passed": False,
            "simulation_failure_reasons": "simulation_exception",
            "error": repr(exc),
            "active_sides": "",
            "frames": 0,
            "duration_s": float("nan"),
            "mean_tracking_error_m": float("nan"),
            "max_tracking_error_m": float("nan"),
            "mean_left_tracking_error_m": float("nan"),
            "mean_right_tracking_error_m": float("nan"),
            "max_left_tracking_error_m": float("nan"),
            "max_right_tracking_error_m": float("nan"),
            "mean_orientation_error_rad": float("nan"),
            "max_orientation_error_rad": float("nan"),
            "mean_left_orientation_error_rad": float("nan"),
            "mean_right_orientation_error_rad": float("nan"),
            "max_left_orientation_error_rad": float("nan"),
            "max_right_orientation_error_rad": float("nan"),
            "mean_base_xy_error_m": float("nan"),
            "mean_left_q_rms_error_rad": float("nan"),
            "mean_right_q_rms_error_rad": float("nan"),
            "max_abs_ctrl": float("nan"),
            "max_contact_count": 0,
            "max_unexpected_contact_count": 0,
            "max_expected_contact_force_n": 0.0,
            "min_contact_distance_m": float("nan"),
            "mean_left_task_force_norm_n": float("nan"),
            "mean_right_task_force_norm_n": float("nan"),
            "max_left_task_force_norm_n": float("nan"),
            "max_right_task_force_norm_n": float("nan"),
            "mean_left_task_torque_norm_nm": float("nan"),
            "mean_right_task_torque_norm_nm": float("nan"),
            "max_left_task_torque_norm_nm": float("nan"),
            "max_right_task_torque_norm_nm": float("nan"),
            "max_joint_torque_clip_fraction": float("nan"),
            "max_left_joint_torque_clip_fraction": float("nan"),
            "max_right_joint_torque_clip_fraction": float("nan"),
            "baseline_metrics": {},
        }
        write_json(task_output / "summary.json", summary)
        _write_failed_simulation_report(task_output, summary)
        return summary


def _write_failed_simulation_report(output_dir: Path, summary: dict[str, Any]) -> Path:
    table = pd.DataFrame([{key: value for key, value in summary.items() if key != "baseline_metrics"}]).to_html(index=False)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MuJoCo Simulation Failed</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#182027}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d7dde2;padding:6px 8px;font-size:13px;text-align:left}}th{{background:#f3f6f8}}</style>
</head><body>
<h1>MuJoCo Simulation Failed: {summary["task_name"]}</h1>
<p><strong>Baseline profile:</strong> {summary.get("baseline_profile", "baseline")} ({summary.get("baseline_name", "unnamed")})</p>
{table}
</body></html>"""
    out = output_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def simulate_batch(
    task_names: list[str],
    group_name: str,
    paths: ProjectPaths = PATHS,
    workers: int | None = None,
    render: bool = False,
    cfg: dict[str, Any] | None = None,
    baseline_profile: str = DEFAULT_PROFILE,
) -> Path:
    cfg = cfg or load_baseline_config(paths, profile=baseline_profile)
    slug = _profile_slug(cfg, baseline_profile)
    prefix = f"mujoco_batch_{group_name}" if slug == "baseline" else f"mujoco_batch_{slug}_{group_name}"
    run_dir = paths.runs / timestamp_id(prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    worker_count = _batch_worker_count(cfg, len(task_names), workers)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if worker_count <= 1:
        for idx, task_name in enumerate(task_names, start=1):
            rows.append(_simulate_batch_worker(task_name, str(run_dir), cfg, render))
            print(f"[mujoco-batch:{group_name}] {idx}/{len(task_names)} {task_name}", flush=True)
    else:
        context = mp.get_context("spawn")
        print(f"[mujoco-batch:{group_name}] simulating {len(task_names)} tasks with {worker_count} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(_simulate_batch_worker, task_name, str(run_dir), cfg, render): task_name
                for task_name in task_names
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                elapsed = time.perf_counter() - started
                print(f"[mujoco-batch:{group_name}] {idx}/{len(task_names)} tasks in {elapsed:.1f}s", flush=True)
    rows.sort(key=lambda row: row["task_name"])
    df = pd.DataFrame([{key: value for key, value in row.items() if key != "baseline_metrics"} for row in rows])
    df.to_csv(run_dir / "batch_simulation_summary.csv", index=False)
    pass_column = "simulation_passed" if "simulation_passed" in df else "simulation_ok"
    aggregate = {
        "group_name": group_name,
        "baseline_profile": cfg.get("_profile", baseline_profile),
        "baseline_name": cfg.get("baseline", {}).get("name", "unnamed"),
        "base_motion_mode": cfg.get("base", {}).get("motion_mode", "free_holonomic"),
        "controller_mode": str(_controller_config(cfg)["mode"]),
        "task_count": len(rows),
        "simulation_ok_count": int(df["simulation_ok"].sum()) if "simulation_ok" in df and len(df) else 0,
        "simulation_ok_fraction": float(df["simulation_ok"].mean()) if "simulation_ok" in df and len(df) else 0.0,
        "simulation_success_count": int(df[pass_column].sum()) if pass_column in df and len(df) else 0,
        "simulation_success_fraction": float(df[pass_column].mean()) if pass_column in df and len(df) else 0.0,
        "workers": worker_count,
        "render": render,
        "mean_tracking_error_m": float(df["mean_tracking_error_m"].mean()) if "mean_tracking_error_m" in df and len(df) else float("nan"),
        "max_tracking_error_m": float(df["max_tracking_error_m"].max()) if "max_tracking_error_m" in df and len(df) else float("nan"),
        "mean_left_tracking_error_m": float(df["mean_left_tracking_error_m"].mean()) if len(df) else float("nan"),
        "mean_right_tracking_error_m": float(df["mean_right_tracking_error_m"].mean()) if len(df) else float("nan"),
        "max_left_tracking_error_m": float(df["max_left_tracking_error_m"].max()) if len(df) else float("nan"),
        "max_right_tracking_error_m": float(df["max_right_tracking_error_m"].max()) if len(df) else float("nan"),
        "mean_orientation_error_rad": float(df["mean_orientation_error_rad"].mean()) if "mean_orientation_error_rad" in df and len(df) else float("nan"),
        "max_orientation_error_rad": float(df["max_orientation_error_rad"].max()) if "max_orientation_error_rad" in df and len(df) else float("nan"),
        "mean_left_orientation_error_rad": float(df["mean_left_orientation_error_rad"].mean()) if len(df) else float("nan"),
        "mean_right_orientation_error_rad": float(df["mean_right_orientation_error_rad"].mean()) if len(df) else float("nan"),
        "max_left_orientation_error_rad": float(df["max_left_orientation_error_rad"].max()) if len(df) else float("nan"),
        "max_right_orientation_error_rad": float(df["max_right_orientation_error_rad"].max()) if len(df) else float("nan"),
        "mean_base_xy_error_m": float(df["mean_base_xy_error_m"].mean()) if len(df) else float("nan"),
        "max_left_task_force_norm_n": float(df["max_left_task_force_norm_n"].max()) if "max_left_task_force_norm_n" in df and len(df) else float("nan"),
        "max_right_task_force_norm_n": float(df["max_right_task_force_norm_n"].max()) if "max_right_task_force_norm_n" in df and len(df) else float("nan"),
        "max_left_task_torque_norm_nm": float(df["max_left_task_torque_norm_nm"].max()) if "max_left_task_torque_norm_nm" in df and len(df) else float("nan"),
        "max_right_task_torque_norm_nm": float(df["max_right_task_torque_norm_nm"].max()) if "max_right_task_torque_norm_nm" in df and len(df) else float("nan"),
        "max_joint_torque_clip_fraction": float(df["max_joint_torque_clip_fraction"].max()) if "max_joint_torque_clip_fraction" in df and len(df) else float("nan"),
        "max_left_joint_torque_clip_fraction": float(df["max_left_joint_torque_clip_fraction"].max()) if "max_left_joint_torque_clip_fraction" in df and len(df) else float("nan"),
        "max_right_joint_torque_clip_fraction": float(df["max_right_joint_torque_clip_fraction"].max()) if "max_right_joint_torque_clip_fraction" in df and len(df) else float("nan"),
        "max_contact_count": int(df["max_contact_count"].max()) if "max_contact_count" in df and len(df) else 0,
        "max_unexpected_contact_count": int(df["max_unexpected_contact_count"].max()) if "max_unexpected_contact_count" in df and len(df) else 0,
        "max_expected_contact_force_n": float(df["max_expected_contact_force_n"].max()) if "max_expected_contact_force_n" in df and len(df) else 0.0,
        "mean_contact_force_rmse_n": float(df["contact_force_rmse_n"].mean()) if "contact_force_rmse_n" in df and len(df) else float("nan"),
        "min_contact_distance_m": float(df["min_contact_distance_m"].min()) if "min_contact_distance_m" in df and len(df) else float("nan"),
        "worst_task_left": str(df.sort_values("max_left_tracking_error_m", ascending=False).iloc[0]["task_name"]) if len(df) else "",
        "worst_task_right": str(df.sort_values("max_right_tracking_error_m", ascending=False).iloc[0]["task_name"]) if len(df) else "",
        "elapsed_s": float(time.perf_counter() - started),
    }
    write_json(run_dir / "batch_summary.json", aggregate)
    _write_batch_report(run_dir, aggregate, df)
    return run_dir


def _write_batch_report(run_dir: Path, aggregate: dict[str, Any], df: pd.DataFrame) -> Path:
    try:
        import plotly.express as px
        from plotly.offline import plot

        fig = px.scatter(
            df,
            x="mean_left_tracking_error_m",
            y="mean_right_tracking_error_m",
            size=np.clip(df["mean_base_xy_error_m"], 0.005, None),
            hover_name="task_name",
            hover_data=[
                "active_sides",
                "simulation_passed",
                "simulation_failure_reasons",
                "max_left_tracking_error_m",
                "max_right_tracking_error_m",
                "max_tracking_error_m",
                "mean_left_orientation_error_rad",
                "mean_right_orientation_error_rad",
                "max_orientation_error_rad",
                "max_joint_torque_clip_fraction",
                "mean_base_xy_error_m",
                "simulation_ok",
                "error",
            ],
            title=f"MuJoCo impedance batch validation: {aggregate['group_name']}",
        )
        plot_html = plot(fig, include_plotlyjs="cdn", output_type="div")
    except Exception:
        plot_html = "<p>Plotly unavailable.</p>"
    summary = pd.DataFrame([aggregate]).to_html(index=False, float_format=lambda value: f"{value:.5f}")
    table = df.to_html(index=False, float_format=lambda value: f"{value:.5f}")
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>MuJoCo Batch {aggregate["group_name"]}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #182027; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d7dde2; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f3f6f8; }}
</style></head>
<body>
<h1>MuJoCo Impedance Batch: {aggregate["group_name"]}</h1>
<p><strong>Baseline profile:</strong> {aggregate.get("baseline_profile", "baseline")} ({aggregate.get("baseline_name", "unnamed")})</p>
<h2>Summary</h2>
{summary}
<h2>Task Map</h2>
{plot_html}
<h2>Per Task Metrics</h2>
{table}
</body></html>
"""
    output = run_dir / "report.html"
    output.write_text(html, encoding="utf-8")
    return output
