"""Kinematics and rigid-body dynamics of the two Franka Emika Panda arms of GARMI
(MuJoCo-based) and damped-least-squares inverse kinematics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from garmi_optimization.config import PATHS
from garmi_optimization.math_utils import quat_wxyz_from_rotation, rotation_error_vector, rotation_from_rpy, rotation_y, rotation_z


class ReferencePandaKinematics:
    """MuJoCo-backed FK/Jacobian for the Panda structure copied from multipanda_ros2."""

    def __init__(self, mjcf_path: Path, prefix: str, root_offset_m: np.ndarray) -> None:
        self.mjcf_path = Path(mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.prefix = prefix
        self.root_offset_m = np.asarray(root_offset_m, dtype=float)
        self.qpos_ids = []
        self.dof_ids = []
        for idx in range(7):
            name = f"{prefix}_joint{idx + 1}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in {mjcf_path}")
            self.qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
        self.joint_site_ids = [self._site_id(f"{prefix}_joint{idx + 1}_site") for idx in range(7)]
        self.ee_site_id = self._site_id(f"{prefix}_ee_site")
        self._inverse_cache: dict[tuple[float, ...], tuple[mujoco.MjModel, mujoco.MjData, list[int], list[int]]] = {}

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"Site '{name}' not found in MuJoCo Panda reference")
        return int(site_id)

    def _set_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        for idx, qpos_id in enumerate(self.qpos_ids):
            self.data.qpos[qpos_id] = q[idx]
        mujoco.mj_forward(self.model, self.data)

    def points(self, q: np.ndarray) -> np.ndarray:
        self._set_q(q)
        points = [np.zeros(3, dtype=float)]
        for site_id in self.joint_site_ids:
            points.append(self.data.site_xpos[site_id].copy() - self.root_offset_m)
        points.append(self.data.site_xpos[self.ee_site_id].copy() - self.root_offset_m)
        return np.asarray(points, dtype=float)

    def ee(self, q: np.ndarray) -> np.ndarray:
        self._set_q(q)
        return self.data.site_xpos[self.ee_site_id].copy() - self.root_offset_m

    def rotation(self, q: np.ndarray) -> np.ndarray:
        self._set_q(q)
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        jacp, _jacr = self.pose_jacobian(q)
        return jacp

    def pose_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._set_q(q)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:, self.dof_ids].copy(), jacr[:, self.dof_ids].copy()

    def _inverse_context(self, mount_rotation: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData, list[int], list[int]]:
        quat = quat_wxyz_from_rotation(mount_rotation)
        key = tuple(np.round(quat, 12))
        if key in self._inverse_cache:
            return self._inverse_cache[key]

        model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{self.prefix}_link0")
        if root_body < 0:
            raise KeyError(f"Body '{self.prefix}_link0' not found in MuJoCo Panda reference")
        model.body_quat[root_body] = quat
        model.body_gravcomp[:] = 0.0
        # Pure rigid-body inverse dynamics (M(q)*qacc + C + g). With
        # constraints enabled, mj_inverse would also report joint-limit
        # constraint forces as required torque and penalize every mount that
        # operates close to its joint limits. Limit proximity is scored
        # separately (joint_limit_proximity_fraction).
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

        qpos_ids = []
        dof_ids = []
        for idx in range(7):
            name = f"{self.prefix}_joint{idx + 1}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"Joint '{name}' not found in {self.mjcf_path}")
            qpos_ids.append(int(model.jnt_qposadr[joint_id]))
            dof_id = int(model.jnt_dofadr[joint_id])
            dof_ids.append(dof_id)
            model.dof_damping[dof_id] = max(float(model.dof_damping[dof_id]), 0.04)
            model.dof_armature[dof_id] = max(float(model.dof_armature[dof_id]), 0.05)

        data = mujoco.MjData(model)
        mujoco.mj_setConst(model, data)
        data = mujoco.MjData(model)
        self._inverse_cache[key] = (model, data, qpos_ids, dof_ids)
        return model, data, qpos_ids, dof_ids

    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, mount_rotation: np.ndarray) -> np.ndarray:
        model, data, qpos_ids, dof_ids = self._inverse_context(mount_rotation)
        mujoco.mj_resetData(model, data)
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        qddot = np.asarray(qddot, dtype=float)
        for idx in range(7):
            data.qpos[qpos_ids[idx]] = q[idx]
            data.qvel[dof_ids[idx]] = qdot[idx]
            data.qacc[dof_ids[idx]] = qddot[idx]
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_inverse(model, data)
        return data.qfrc_inverse[dof_ids].copy()


def _preset_mount_rotation(panda_cfg: dict[str, Any]) -> np.ndarray:
    mount_rotation = str(panda_cfg.get("mount_rotation", "y90")).lower()
    if mount_rotation in {"y90", "horizontal_forward"}:
        return rotation_y(np.pi / 2.0)
    if mount_rotation in {"identity", "none"}:
        return np.eye(3)
    raise ValueError(f"Unsupported panda.mount_rotation '{panda_cfg.get('mount_rotation')}'")


def _mounts_from_config(
    panda_cfg: dict[str, Any],
    *,
    shoulder_width_m: float,
    shoulder_height_m: float,
    shoulder_x_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if "mounts" in panda_cfg:
        positions = {}
        rotations = {}
        for side in ["left", "right"]:
            mount = panda_cfg["mounts"][side]
            positions[side] = np.asarray(mount["xyz_m"], dtype=float)
            rpy = np.asarray(mount.get("rpy_rad", [0.0, 0.0, 0.0]), dtype=float)
            rotations[side] = rotation_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        return positions, rotations

    rotation = _preset_mount_rotation(panda_cfg)
    positions = {}
    rotations = {}
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        positions[side] = np.array([shoulder_x_m, sign * shoulder_width_m / 2.0, shoulder_height_m], dtype=float)
        rotations[side] = rotation.copy()
    return positions, rotations


@dataclass
class PandaModel:
    shoulder_width_m: float
    shoulder_height_m: float
    shoulder_x_m: float
    mount_positions_m: dict[str, np.ndarray]
    mount_rotations: dict[str, np.ndarray]
    tool_offset_m: float
    reference_mjcf: Path
    reference_prefix: str
    reference_root_offset_m: np.ndarray
    q_lower: np.ndarray
    q_upper: np.ndarray
    q_nominal: np.ndarray
    torque_limits_nm: np.ndarray
    velocity_limits_radps: np.ndarray
    acceleration_limit_radps2: float
    inertia_proxy_kgm2: np.ndarray
    manipulability_reference: float
    _reference: ReferencePandaKinematics | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PandaModel":
        panda = cfg["panda"]
        kinematic_model = str(panda.get("kinematic_model", "panda")).lower()
        if kinematic_model not in {"panda", "franka", "franka_panda"}:
            raise ValueError(
                "garmi_optimization currently supports only the original Franka Panda arm model. "
                f"Unsupported panda.kinematic_model: {panda.get('kinematic_model')!r}"
            )
        limits = panda["joint_limits_rad"]
        reference_mjcf = Path(panda.get("reference_mjcf", "resources/mujoco/franka/mj_dual.xml"))
        if not reference_mjcf.is_absolute():
            reference_mjcf = PATHS.root / reference_mjcf
        shoulder_width_m = float(panda.get("shoulder_width_m", 0.74))
        shoulder_height_m = float(panda.get("shoulder_height_m", 1.12))
        shoulder_x_m = float(panda.get("shoulder_x_m", 0.0))
        mount_positions, mount_rotations = _mounts_from_config(
            panda,
            shoulder_width_m=shoulder_width_m,
            shoulder_height_m=shoulder_height_m,
            shoulder_x_m=shoulder_x_m,
        )
        return cls(
            shoulder_width_m=shoulder_width_m,
            shoulder_height_m=shoulder_height_m,
            shoulder_x_m=shoulder_x_m,
            mount_positions_m=mount_positions,
            mount_rotations=mount_rotations,
            tool_offset_m=float(panda.get("tool_offset_m", 0.1034)),
            reference_mjcf=reference_mjcf,
            reference_prefix=str(panda.get("reference_prefix", "mj_left")),
            reference_root_offset_m=np.asarray(panda.get("reference_root_offset_m", [0.0, 0.26, 0.0]), dtype=float),
            q_lower=np.asarray(limits["lower"], dtype=float),
            q_upper=np.asarray(limits["upper"], dtype=float),
            q_nominal=np.asarray(panda["nominal_q_rad"], dtype=float),
            torque_limits_nm=np.asarray(panda["torque_limits_nm"], dtype=float),
            velocity_limits_radps=np.asarray(panda["velocity_limits_radps"], dtype=float),
            acceleration_limit_radps2=float(panda["acceleration_limit_radps2"]),
            inertia_proxy_kgm2=np.asarray(panda["inertia_proxy_kgm2"], dtype=float),
            manipulability_reference=float(panda.get("manipulability_reference", 0.08)),
        )

    @property
    def dof(self) -> int:
        return 7

    @property
    def reach_m(self) -> float:
        return 0.93

    def shoulder_local(self, side: str) -> np.ndarray:
        return self.mount_positions_m[side].copy()

    def shoulder_world(self, side: str, base_pose: np.ndarray) -> np.ndarray:
        base_xy = np.array([base_pose[0], base_pose[1], 0.0])
        return base_xy + rotation_z(float(base_pose[2])) @ self.shoulder_local(side)

    def mount_rotation_world(self, side: str, base_yaw: float) -> np.ndarray:
        return rotation_z(base_yaw) @ self.mount_rotations[side]

    def reference(self) -> ReferencePandaKinematics:
        if self._reference is None:
            self._reference = ReferencePandaKinematics(
                self.reference_mjcf,
                self.reference_prefix,
                self.reference_root_offset_m,
            )
        return self._reference

    def inverse_dynamics(self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, side: str) -> np.ndarray:
        return self.reference().inverse_dynamics(q, qdot, qddot, self.mount_rotations[side])

def forward_points(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray) -> np.ndarray:
    local = model.reference().points(q)
    shoulder = model.shoulder_world(side, base_pose)
    rotation = model.mount_rotation_world(side, float(base_pose[2]))
    return shoulder + local @ rotation.T


def end_effector(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray) -> np.ndarray:
    local = model.reference().ee(q)
    shoulder = model.shoulder_world(side, base_pose)
    rotation = model.mount_rotation_world(side, float(base_pose[2]))
    return shoulder + rotation @ local


def end_effector_rotation(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray) -> np.ndarray:
    rotation = model.mount_rotation_world(side, float(base_pose[2]))
    return rotation @ model.reference().rotation(q)


def position_jacobian(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray) -> np.ndarray:
    rotation = model.mount_rotation_world(side, float(base_pose[2]))
    return rotation @ model.reference().jacobian(q)


def pose_jacobian(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = model.mount_rotation_world(side, float(base_pose[2]))
    jacp, jacr = model.reference().pose_jacobian(q)
    return rotation @ jacp, rotation @ jacr


def manipulability(model: PandaModel, q: np.ndarray, side: str, base_pose: np.ndarray) -> float:
    jac = position_jacobian(model, q, side, base_pose)
    gram = jac @ jac.T
    value = float(np.sqrt(max(np.linalg.det(gram), 0.0)))
    return value / max(model.manipulability_reference, 1e-9)


def solve_pose_ik(
    model: PandaModel,
    target: np.ndarray,
    target_rotation: np.ndarray | None,
    side: str,
    base_pose: np.ndarray,
    q0: np.ndarray,
    ik_cfg: dict[str, Any],
) -> tuple[np.ndarray, float, float]:
    q = np.asarray(q0, dtype=float).copy()
    damping = float(ik_cfg.get("damping", 0.035))
    step_limit = float(ik_cfg.get("step_limit_rad", 0.18))
    joint_center_weight = float(ik_cfg.get("joint_center_weight", 0.015))
    continuity_weight = float(ik_cfg.get("continuity_weight", 0.02))
    orientation_weight = float(ik_cfg.get("orientation_weight", 0.08))
    use_orientation = bool(ik_cfg.get("use_orientation", True)) and target_rotation is not None
    target_rotation = None if target_rotation is None else np.asarray(target_rotation, dtype=float)
    q_center = 0.5 * (model.q_lower + model.q_upper)
    q_span = np.maximum(model.q_upper - model.q_lower, 1e-6)
    for _ in range(int(ik_cfg.get("iterations", 35))):
        end = end_effector(model, q, side, base_pose)
        pos_error = np.asarray(target, dtype=float) - end
        rot_error = np.zeros(3, dtype=float)
        if use_orientation and target_rotation is not None:
            rot_error = rotation_error_vector(target_rotation, end_effector_rotation(model, q, side, base_pose))
        if (
            float(np.linalg.norm(pos_error)) < float(ik_cfg.get("position_tolerance_m", 0.02))
            and (not use_orientation or float(np.linalg.norm(rot_error)) < float(ik_cfg.get("orientation_tolerance_rad", 0.20)))
        ):
            break
        if use_orientation:
            jac_pos, jac_rot = pose_jacobian(model, q, side, base_pose)
            jac = np.vstack([jac_pos, orientation_weight * jac_rot])
            error = np.concatenate([pos_error, orientation_weight * rot_error])
        else:
            jac = position_jacobian(model, q, side, base_pose)
            error = pos_error
        lhs = jac @ jac.T + (damping * damping) * np.eye(jac.shape[0])
        try:
            dq = jac.T @ np.linalg.solve(lhs, error)
        except np.linalg.LinAlgError:
            dq = jac.T @ np.linalg.pinv(lhs) @ error
        dq += joint_center_weight * (q_center - q) / q_span
        dq += continuity_weight * (np.asarray(q0, dtype=float) - q)
        norm = float(np.linalg.norm(dq))
        if norm > step_limit:
            dq *= step_limit / norm
        q = np.clip(q + dq, model.q_lower, model.q_upper)
    final_error = float(np.linalg.norm(np.asarray(target, dtype=float) - end_effector(model, q, side, base_pose)))
    if use_orientation and target_rotation is not None:
        final_orientation_error = float(np.linalg.norm(rotation_error_vector(target_rotation, end_effector_rotation(model, q, side, base_pose))))
    else:
        final_orientation_error = 0.0
    return q, final_error, final_orientation_error


def solve_position_ik(
    model: PandaModel,
    target: np.ndarray,
    side: str,
    base_pose: np.ndarray,
    q0: np.ndarray,
    ik_cfg: dict[str, Any],
) -> tuple[np.ndarray, float]:
    q, pos_error, _orient_error = solve_pose_ik(model, target, None, side, base_pose, q0, ik_cfg)
    return q, pos_error
