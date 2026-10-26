import numpy as np
import pytest

from garmi_optimization.config import load_baseline_config
from garmi_optimization.math_utils import rotation_from_rpy, rotation_z
from garmi_optimization.mujoco_backend import _environment_spec
from garmi_optimization.optimization import _apply_fidelity_constraint_guard, _score_candidate
from garmi_optimization.perturbations import (
    expand_task_names,
    perturb_task,
    robustness_summary,
    split_variant_name,
)
from garmi_optimization.sensitivity import _morris_trajectories
from garmi_optimization.task_frames import TaskFrameSpec, anchor_offset, orientation_error, spec_for_task
from garmi_optimization.tasks import group_task_weights, load_task

PROFILE = "stage2_forces"


# ---------------------------------------------------------------------------
# task frames


def test_free_mode_scores_zero():
    spec = TaskFrameSpec(orientation_mode="free")
    r1 = rotation_from_rpy(0.3, -0.2, 0.9)
    assert orientation_error(spec, np.eye(3), r1) == 0.0


def test_free_tool_axis_invariant_under_tool_rotation():
    spec = TaskFrameSpec(orientation_mode="relative", free_tool_axis=True)
    target = rotation_from_rpy(0.2, 0.1, -0.4)
    # rotate current about the tool (z) axis of target: tool axes stay aligned
    current = target @ rotation_z(1.1)
    assert orientation_error(spec, target, current) < 1e-9
    # full geodesic error would be large
    strict = TaskFrameSpec(orientation_mode="relative", free_tool_axis=False)
    assert orientation_error(strict, target, current) > 0.5


def test_gravity_aligned_free_yaw_about_world_z():
    spec = TaskFrameSpec(orientation_mode="gravity_aligned")
    target = rotation_from_rpy(0.25, 0.0, 0.0)
    current = rotation_z(0.8) @ target  # same tilt, different world yaw
    assert orientation_error(spec, target, current) < 0.15
    tilted = rotation_from_rpy(0.85, 0.0, 0.0)  # different tilt
    assert orientation_error(spec, target, tilted) > 0.4


def test_gravity_aligned_anchor_offset_preserves_tilt():
    spec = TaskFrameSpec(orientation_mode="gravity_aligned")
    raw0 = rotation_from_rpy(0.3, 0.1, 0.2)
    ee0 = rotation_from_rpy(-0.2, 0.4, 1.2)
    offset = anchor_offset(spec, raw0, ee0)
    anchored = raw0 @ offset
    up = np.array([0.0, 0.0, 1.0])
    tilt_raw = np.arccos(np.clip(raw0[:, 2] @ up, -1, 1))
    tilt_anchored = np.arccos(np.clip(anchored[:, 2] @ up, -1, 1))
    assert abs(tilt_raw - tilt_anchored) < 1e-9


def test_spec_for_task_reads_config():
    cfg = load_baseline_config(profile=PROFILE)
    spec = spec_for_task(cfg, "carrying_tray_force")
    assert spec.orientation_mode == "gravity_aligned"
    spec2 = spec_for_task(cfg, "wiping_table_force")
    assert spec2.free_tool_axis
    default = spec_for_task(cfg, "slicing_bread_force")
    assert default.orientation_mode == "relative" and not default.free_tool_axis


# ---------------------------------------------------------------------------
# perturbations


def test_variant_names_roundtrip():
    assert split_variant_name("task_a") == ("task_a", None)
    assert split_variant_name("task_a@p3") == ("task_a", 3)
    expanded = expand_task_names(["t1", "t2"], 2)
    assert expanded == ["t1", "t1@p1", "t1@p2", "t2", "t2@p1", "t2@p2"]


@pytest.mark.requires_data
def test_perturb_task_deterministic_and_rigid():
    task = load_task("slicing_bread_force")
    robustness = {"translation_m": 0.1, "yaw_rad": 0.2, "scale": 0.0, "seed": 123}
    p1 = perturb_task(task, 1, robustness)
    p2 = perturb_task(task, 1, robustness)
    assert p1.name.endswith("@p1")
    np.testing.assert_allclose(p1.left_pos, p2.left_pos)
    # scale=0 -> rigid transform: pairwise distances preserved
    d0 = np.linalg.norm(task.left_pos[0] - task.right_pos[0])
    d1 = np.linalg.norm(p1.left_pos[0] - p1.right_pos[0])
    assert abs(d0 - d1) < 1e-9
    # different variant differs
    p3 = perturb_task(task, 2, robustness)
    assert not np.allclose(p1.left_pos, p3.left_pos)


def test_perturbations_do_not_depend_on_the_process():
    """Every worker process draws the same perturbations (common random numbers)."""
    import os
    import subprocess
    import sys

    code = "from garmi_optimization.perturbations import _stable_name_hash; print(_stable_name_hash('slicing_bread_force'))"
    values = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        values.add(result.stdout.strip())
    assert len(values) == 1


def test_robustness_summary():
    rows = [
        {"task_name": "a", "success": True},
        {"task_name": "a@p1", "success": True},
        {"task_name": "a@p2", "success": False},
        {"task_name": "b", "success": False},
        {"task_name": "b@p1", "success": False},
    ]
    out = robustness_summary(rows)
    assert out["success_fraction_nominal"] == pytest.approx(0.5)
    assert out["success_fraction_perturbed_mean"] == pytest.approx(0.25)
    assert out["robustness_gap"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# soft score / guards


def _fake_row(name: str, success: bool, soft: float, weight: float = 1.0) -> dict:
    return {
        "task_name": name,
        "success": success,
        "kinematic_ok": success,
        "position_ok": success,
        "orientation_ok": success,
        "manipulability_ok": success,
        "torque_ok": True,
        "collision_free": True,
        "motion_ok": True,
        "clearance_warning": False,
        "objective": soft * 10.0,
        "soft_loss": soft,
        "task_weight": weight,
        "primary_failure": "ok" if success else "max_position_error",
        "mean_position_error_m": 0.01,
        "max_position_error_m": 0.02,
        "mean_orientation_error_rad": 0.1,
        "max_orientation_error_rad": 0.2,
        "min_manipulability": 0.1,
        "max_torque_utilization": 0.5,
        "max_dynamic_torque_nm": 10.0,
        "max_external_torque_nm": 5.0,
        "max_total_torque_nm": 12.0,
        "max_measured_force_n": 20.0,
        "max_measured_task_torque_nm": 3.0,
        "max_joint_speed_utilization": 0.4,
        "min_clearance_m": 0.08,
        "clearance_margin_m": 0.08,
        "warning_clearance_margin_m": 0.03,
        "base_motion_mode": "fixed_initial",
        "base_path_length_m": 0.0,
        "base_max_speed_mps": 0.0,
        "base_max_acc_mps2": 0.0,
        "base_max_yaw_rate_radps": 0.0,
    }


def test_score_candidate_is_smooth():
    plaus = {"mount_plausibility_penalty": 0.0, "architecture_plausibility_penalty": 0.0}
    rows_good = [_fake_row("a", True, 0.0), _fake_row("b", True, 0.1)]
    rows_slightly_worse = [_fake_row("a", True, 0.0), _fake_row("b", False, 0.15)]
    s_good = _score_candidate(rows_good, plaus)
    s_worse = _score_candidate(rows_slightly_worse, plaus)
    assert s_good["optimization_score"] < s_worse["optimization_score"]
    # a single near-miss failure costs little more than its soft loss
    assert s_worse["optimization_score"] - s_good["optimization_score"] < 10.0
    assert "coverage_weighted_score" in s_good


def test_task_weights_affect_score():
    plaus = {"mount_plausibility_penalty": 0.0, "architecture_plausibility_penalty": 0.0}
    rows_a = [_fake_row("a", True, 0.0, weight=1.0), _fake_row("b", False, 1.0, weight=1.0)]
    rows_b = [_fake_row("a", True, 0.0, weight=1.0), _fake_row("b", False, 1.0, weight=3.0)]
    assert (
        _score_candidate(rows_b, plaus)["optimization_score"]
        > _score_candidate(rows_a, plaus)["optimization_score"]
    )


def test_screening_guards_do_not_gate_eligibility():
    cfg = load_baseline_config(profile=PROFILE)
    summary = {
        "min_clearance_margin_m": 0.02,  # violates the quick-fidelity guard (0.035)
        "mount_min_body_clearance_m": 0.05,
        "max_torque_utilization": 0.85,  # violates the quick-fidelity guard (0.70)
        "max_joint_speed_utilization": 0.5,
    }
    out = _apply_fidelity_constraint_guard(summary, cfg, "quick")
    # effective constraints are the full-level ones -> candidate stays eligible
    assert out["constraint_min_clearance_m"] == pytest.approx(0.0)
    assert out["constraint_torque_utilization_limit"] == pytest.approx(1.0)
    # but the screening violation is logged
    assert out["screening_guard_violation"] > 0.0
    assert out["screening_guard_enforced"] is False
    out_full = _apply_fidelity_constraint_guard(summary, cfg, "full")
    assert out_full["screening_guard_enforced"] is True


# ---------------------------------------------------------------------------
# environment proxies


@pytest.mark.requires_data
def test_environment_spec_table_and_patient():
    cfg = load_baseline_config(profile=PROFILE)
    # Environment proxies are opt-in (bbox placement collides with the
    # reference trajectories); the spec generation itself stays testable.
    cfg["environment"] = {
        "enabled": True,
        "categories": {"meals": {"table": {}}, "geriatric_rehabilitation": {"patient": {}}},
        "overrides": {},
    }
    table_meta = load_task("slicing_bread_force").metadata
    geoms = _environment_spec(cfg, table_meta)
    assert [g["name"] for g in geoms] == ["env_table"]
    top_z = float(geoms[0]["pos"][2]) + float(geoms[0]["size"][2])
    assert top_z < table_meta["workspace_bbox_m"]["min"][2]

    physio_meta = load_task("wiping_table_exercise_force").metadata
    names = [g["name"] for g in _environment_spec(cfg, physio_meta)]
    assert "env_patient" in names and "env_seat" in names

    cfg_disabled = dict(cfg)
    cfg_disabled["environment"] = {"enabled": False}
    assert _environment_spec(cfg_disabled, table_meta) == []


# ---------------------------------------------------------------------------
# weights / sensitivity plumbing


def test_group_task_weights_defaults():
    weights = group_task_weights("basic_care")
    assert weights and all(value == 1.0 for value in weights.values())


@pytest.mark.requires_data
def test_dynamics_time_scaling_and_jump_masking():
    """Fast but feasible motion is time-scaled and IK flips are masked."""
    from garmi_optimization.evaluation import _trajectory_torque_utilization
    from garmi_optimization.panda import PandaModel

    cfg = load_baseline_config(profile=PROFILE)
    model = PandaModel.from_config(cfg)
    task = load_task("slicing_bread_force")
    n = task.sample_count

    # smooth trajectory slightly above speed limits -> time scale kicks in
    t = np.asarray(task.time)
    fast = np.outer(np.sin(2.0 * np.pi * (t - t[0]) / max(t[-1] - t[0], 1e-9)), np.ones(7)) * 1.2
    fast = np.clip(model.q_nominal[None, :] + fast, model.q_lower, model.q_upper)
    tau_ext = np.zeros((n, 7))
    util, _qd, _td, _tt, scale, disc = _trajectory_torque_utilization(model, task, fast, tau_ext, "left", cfg)
    assert disc == pytest.approx(0.0)
    assert 1.0 <= scale <= float(cfg["success"]["max_time_scale"])

    # single pi-flip in one sample -> masked as discontinuity, torque stays sane
    flip = np.repeat(model.q_nominal[None, :], n, axis=0)
    flip[n // 2 :, 3] = model.q_nominal[3] + 2.5
    flip = np.clip(flip, model.q_lower, model.q_upper)
    util2, _qd2, _td2, _tt2, _s2, disc2 = _trajectory_torque_utilization(model, task, flip, tau_ext, "left", cfg)
    assert disc2 > 0.0
    assert float(np.max(util2)) < 5.0  # without masking this reaches O(100)


def test_base_mode_override_from_task_frames():
    cfg = load_baseline_config(profile=PROFILE)
    spec = spec_for_task(cfg, "walking_AAN_kin")
    assert spec.base_motion_mode == "nonholonomic"
    assert spec_for_task(cfg, "slicing_bread_force").base_motion_mode is None


def test_inverse_dynamics_bounded_at_joint_limits():
    """No joint-limit constraint forces in the torque estimate."""
    from garmi_optimization.panda import PandaModel

    cfg = load_baseline_config(profile=PROFILE)
    model = PandaModel.from_config(cfg)
    q = model.q_upper.copy() - 1e-6
    tau = model.inverse_dynamics(q, np.full(7, 0.5), np.full(7, 2.0), "left")
    util = np.abs(tau) / np.asarray(cfg["panda"]["torque_limits_nm"], dtype=float)
    assert float(np.max(util)) < 2.0  # was O(100) with constraint forces


def test_eligibility_epsilon_tolerates_small_violations():
    """A 2 mm clearance shortfall must not gate the ranking."""
    import pandas as pd

    from garmi_optimization.optimization import _rank_candidate_frame

    rows = []
    for cid, coverage, violation_m in [("good_tiny_violation", 0.75, 0.002), ("clean_but_poor", 0.30, 0.0)]:
        rows.append(
            {
                "candidate_id": cid,
                "success_fraction": coverage,
                "failed_task_count": int((1 - coverage) * 8),
                "optimization_score": 1.0 if cid == "good_tiny_violation" else 4.0,
                "min_clearance_margin_m": -violation_m,
                "min_warning_clearance_margin_m": -violation_m,
                "mount_min_body_clearance_m": 0.05,
                "max_torque_utilization": 0.5,
                "max_joint_speed_utilization": 0.5,
                "mean_position_error_m": 0.01,
                "max_position_error_m": 0.02,
                "mean_orientation_error_rad": 0.1,
                "max_orientation_error_rad": 0.2,
                "mount_plausibility_penalty": 0.0,
                "architecture_plausibility_penalty": 0.0,
                "constraint_min_clearance_m": 0.0,
                "constraint_mount_min_body_clearance_m": 0.0,
                "constraint_torque_utilization_limit": 1.0,
                "constraint_joint_speed_utilization_limit": 1.0,
            }
        )
    ranked = _rank_candidate_frame(pd.DataFrame(rows))
    assert ranked.iloc[0]["candidate_id"] == "good_tiny_violation"
    assert bool(ranked.iloc[0]["eligible_candidate"])


def test_shoulder_pads_follow_mounts():
    from garmi_optimization.collision import garmi_body_proxies

    mounts = {"left": {"xyz_m": [-0.10, 0.30, 1.05]}, "right": {"xyz_m": [-0.10, -0.30, 1.05]}}
    moved = {p.name: p for p in garmi_body_proxies(mounts)}
    static = {p.name: p for p in garmi_body_proxies(None)}
    assert moved["left_shoulder_pad"].center[2] > static["left_shoulder_pad"].center[2]
    assert abs(moved["left_shoulder_pad"].center[1] - 0.245) < 1e-6  # 0.30 - 0.055 offset
    # unrelated proxies unchanged
    np.testing.assert_allclose(moved["head"].center, static["head"].center)


def test_auto_duration_scale_inherits_kinematic_time_scale():
    from garmi_optimization.mujoco_backend import _auto_duration_scale

    cfg = {"simulation": {}}
    assert _auto_duration_scale({"baseline_metrics": {"time_scale": 1.4}}, cfg) == pytest.approx(1.4)
    assert _auto_duration_scale({"baseline_metrics": {}}, cfg) == pytest.approx(1.0)
    cfg_slow = {"simulation": {"playback_slowdown": 2.0}}
    assert _auto_duration_scale({"baseline_metrics": {"time_scale": 1.4}}, cfg_slow) == pytest.approx(2.8)


def test_morris_trajectories_valid():
    unit = _morris_trajectories(dim=6, r=4, levels=6, seed=3)
    assert unit.shape == (4, 7, 6)
    assert np.all(unit >= -1e-12) and np.all(unit <= 1.0 + 1e-12)
    for t in range(4):
        for s in range(1, 7):
            diff = unit[t, s] - unit[t, s - 1]
            assert int(np.sum(np.abs(diff) > 1e-12)) == 1
