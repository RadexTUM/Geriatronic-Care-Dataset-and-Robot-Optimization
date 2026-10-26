"""Primitive-geometry collision model of the GARMI body and clearance checks of the arms.

The torso, head, base and shoulder covers are approximated by spheres,
capsules and boxes; the shoulder covers move with the candidate mounts so
that every candidate is checked against the same body model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from garmi_optimization.math_utils import rotation_z


@dataclass(frozen=True)
class SphereProxy:
    name: str
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class CapsuleProxy:
    name: str
    p0: np.ndarray
    p1: np.ndarray
    radius: float


@dataclass(frozen=True)
class BoxProxy:
    name: str
    center: np.ndarray
    half_extents: np.ndarray


# Original GARMI mount frames the static pad positions were modeled for.
_REFERENCE_MOUNT_LEFT = np.array([-0.14326, 0.20000, 0.92451])
# Pad center offset relative to its mount (measured from the original model).
_PAD_OFFSET_LEFT = np.array([-0.16, 0.145, 0.925]) - _REFERENCE_MOUNT_LEFT


def _shoulder_pad_proxies(mounts: dict[str, Any] | None) -> list[BoxProxy]:
    """Shoulder-pad proxies that move with the mounts.

    The pads keep their offset relative to the mount, so a moved mount is
    never checked against a pad at the original mount location.
    """
    half = np.array([0.12, 0.055, 0.13])
    if not mounts:
        return [
            BoxProxy("left_shoulder_pad", _REFERENCE_MOUNT_LEFT + _PAD_OFFSET_LEFT, half),
            BoxProxy("right_shoulder_pad", (_REFERENCE_MOUNT_LEFT + _PAD_OFFSET_LEFT) * np.array([1.0, -1.0, 1.0]), half),
        ]
    pads = []
    for side in ["left", "right"]:
        mount = np.asarray((mounts.get(side, {}) or {}).get("xyz_m", _REFERENCE_MOUNT_LEFT * (1 if side == "left" else np.array([1.0, -1.0, 1.0]))), dtype=float)
        offset = _PAD_OFFSET_LEFT if side == "left" else _PAD_OFFSET_LEFT * np.array([1.0, -1.0, 1.0])
        pads.append(BoxProxy(f"{side}_shoulder_pad", mount + offset, half))
    return pads


def garmi_body_proxies(mounts: dict[str, Any] | None = None) -> list[SphereProxy | CapsuleProxy | BoxProxy]:
    return [
        BoxProxy("mobile_platform_main", np.array([0.0, 0.0, -0.105]), np.array([0.40, 0.35, 0.105])),
        BoxProxy("mobile_platform_crossbar", np.array([0.0, 0.0, -0.105]), np.array([0.60, 0.05, 0.105])),
        BoxProxy("base_lower_box", np.array([0.0, 0.0, 0.20]), np.array([0.30, 0.30, 0.25])),
        BoxProxy("torso_front_box", np.array([0.0, 0.0, 0.75]), np.array([0.15, 0.15, 0.30])),
        BoxProxy("torso_backpack_center", np.array([-0.15, 0.0, 0.75]), np.array([0.15, 0.105, 0.30])),
        *_shoulder_pad_proxies(mounts),
        SphereProxy("head", np.array([0.00859, 0.00621, 1.25]), 0.20),
        CapsuleProxy("left_wheel", np.array([0.0, 0.35, -0.125]), np.array([0.0, 0.40, -0.125]), 0.08),
        CapsuleProxy("right_wheel", np.array([0.0, -0.40, -0.125]), np.array([0.0, -0.35, -0.125]), 0.08),
        BoxProxy("front_lidar", np.array([0.42, 0.0, 0.01]), np.array([0.05, 0.05, 0.05])),
        BoxProxy("back_lidar", np.array([-0.42, 0.0, 0.01]), np.array([0.05, 0.05, 0.05])),
    ]


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def _segment_segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    denom = a * c - b * b
    small = 1e-12
    if a <= small and c <= small:
        return float(np.linalg.norm(a0 - b0))
    if a <= small:
        t = np.clip(e / max(c, small), 0.0, 1.0)
        return float(np.linalg.norm(a0 - (b0 + t * v)))
    if c <= small:
        s = np.clip(-d / max(a, small), 0.0, 1.0)
        return float(np.linalg.norm((a0 + s * u) - b0))
    s = 0.0 if denom <= small else np.clip((b * e - c * d) / denom, 0.0, 1.0)
    t = (b * s + e) / c
    if t < 0.0:
        t = 0.0
        s = np.clip(-d / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - d) / a, 0.0, 1.0)
    return float(np.linalg.norm((a0 + s * u) - (b0 + t * v)))


def _point_box_clearance(point: np.ndarray, radius: float, box: BoxProxy) -> float:
    delta = np.abs(point - box.center) - box.half_extents
    outside = np.maximum(delta, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    if outside_distance > 0.0:
        return outside_distance - radius
    penetration = float(np.min(box.half_extents - np.abs(point - box.center)))
    return -penetration - radius


def _segment_box_clearance(a: np.ndarray, b: np.ndarray, radius: float, box: BoxProxy) -> float:
    samples = np.linspace(a, b, 9)
    return min(_point_box_clearance(point, radius, box) for point in samples)


def _segment_body_clearance(a: np.ndarray, b: np.ndarray, radius: float, proxies: list | None = None) -> tuple[float, str]:
    best = float("inf")
    body = ""
    for proxy in proxies if proxies is not None else garmi_body_proxies():
        if isinstance(proxy, SphereProxy):
            clearance = _point_segment_distance(proxy.center, a, b) - radius - proxy.radius
        elif isinstance(proxy, CapsuleProxy):
            clearance = _segment_segment_distance(a, b, proxy.p0, proxy.p1) - radius - proxy.radius
        else:
            clearance = _segment_box_clearance(a, b, radius, proxy)
        if clearance < best:
            best = float(clearance)
            body = proxy.name
    return best, body


def _point_body_clearance(point: np.ndarray, radius: float, proxies: list | None = None) -> tuple[float, str]:
    best = float("inf")
    body = ""
    for proxy in proxies if proxies is not None else garmi_body_proxies():
        if isinstance(proxy, SphereProxy):
            clearance = float(np.linalg.norm(point - proxy.center) - radius - proxy.radius)
        elif isinstance(proxy, CapsuleProxy):
            clearance = _point_segment_distance(point, proxy.p0, proxy.p1) - radius - proxy.radius
        else:
            clearance = _point_box_clearance(point, radius, proxy)
        if clearance < best:
            best = float(clearance)
            body = proxy.name
    return best, body


def _world_to_base_local(points: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    origin = np.array([base_pose[0], base_pose[1], 0.0])
    rot = rotation_z(-float(base_pose[2]))
    return np.asarray([rot @ (point - origin) for point in points], dtype=float)


def arm_body_clearance(points_world: np.ndarray, base_pose: np.ndarray, cfg: dict[str, Any]) -> tuple[float, str]:
    points = _world_to_base_local(np.asarray(points_world, dtype=float), base_pose)
    link_radius = float(cfg.get("arm_link_radius_m", 0.035))
    joint_radius = float(cfg.get("joint_radius_m", 0.045))
    ee_radius = float(cfg.get("end_effector_radius_m", 0.06))
    mount_exclusion = float(cfg.get("shoulder_mount_exclusion_m", 0.22))
    mount_points = int(cfg.get("shoulder_mount_exclusion_points", 1))
    # Shoulder pads follow the candidate mounts (cfg["_mounts"] is injected by
    # evaluate_task_baseline from panda.mounts).
    proxies = garmi_body_proxies(cfg.get("_mounts"))
    best = float("inf")
    best_body = ""
    for idx in range(len(points) - 1):
        a = points[idx]
        b = points[idx + 1]
        if idx == 0 and mount_exclusion > 0:
            length = float(np.linalg.norm(b - a))
            if length <= mount_exclusion:
                continue
            a = a + (b - a) * (mount_exclusion / length)
        clearance, body = _segment_body_clearance(a, b, link_radius, proxies)
        if clearance < best:
            best = clearance
            best_body = body
    for idx, point in enumerate(points):
        if idx <= mount_points:
            continue
        radius = ee_radius if idx == len(points) - 1 else joint_radius
        clearance, body = _point_body_clearance(point, radius, proxies)
        if clearance < best:
            best = clearance
            best_body = body
    return best, best_body


def inter_arm_clearance(left_points: np.ndarray, right_points: np.ndarray, cfg: dict[str, Any]) -> float:
    link_radius = float(cfg.get("arm_link_radius_m", 0.035))
    ee_radius = float(cfg.get("end_effector_radius_m", 0.06))
    best = float("inf")
    for i in range(len(left_points) - 1):
        for j in range(len(right_points) - 1):
            best = min(
                best,
                _segment_segment_distance(left_points[i], left_points[i + 1], right_points[j], right_points[j + 1])
                - 2.0 * link_radius,
            )
    for ee, other in [(left_points[-1], right_points), (right_points[-1], left_points)]:
        for idx in range(len(other) - 1):
            best = min(best, _point_segment_distance(ee, other[idx], other[idx + 1]) - ee_radius - link_radius)
    return float(best)
