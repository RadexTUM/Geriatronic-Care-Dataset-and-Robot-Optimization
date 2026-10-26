import numpy as np
import mujoco
import pandas as pd
import pytest

from garmi_optimization.evaluation import estimate_base_motion, normalize_base_motion_mode
from garmi_optimization.cli import _exclude_task_names
from garmi_optimization.config import ProjectPaths, load_baseline_config, write_yaml
from garmi_optimization.gui import _apply_gui_overrides, _simulation_rows, _top_mount_candidates_for_reports, _write_task_report_index
from garmi_optimization.mujoco_backend import _simulation_pass_status, build_baseline_mjcf
from garmi_optimization.optimization import (
    DesignCandidate,
    MIRROR_Y,
    MountCandidate,
    apply_design_candidate,
    apply_mount_candidate,
    candidate_hash,
    _candidate_bounds,
    _candidate_vector,
    _design_bounds,
    _design_candidate_from_unit_vector,
    _load_genetic_seed_genomes,
    _genetic_initial_genomes,
    _genetic_next_generation,
    _genetic_survivor_ids,
    generate_mount_candidates,
    generate_design_candidates,
    _promoted_candidates,
    _rank_candidate_frame,
)
from garmi_optimization.panda import PandaModel, end_effector, end_effector_rotation, forward_points, solve_position_ik
from garmi_optimization.tasks import global_task_names, group_task_names, load_task

PROFILE = "stage2_forces"


def test_horizontal_panda_mount_extends_forward_from_shoulder():
    cfg = load_baseline_config()
    model = PandaModel.from_config(cfg)
    base_pose = np.zeros(3)
    points = forward_points(model, model.q_nominal, "left", base_pose)
    shoulder = model.shoulder_world("left", base_pose)
    assert points[1, 0] > shoulder[0] + 0.30
    assert points.shape[1] == 3


def test_strict_garmi_mounts_match_xacro_values():
    cfg = load_baseline_config(profile=PROFILE)
    model = PandaModel.from_config(cfg)

    assert cfg["success"]["min_clearance_m"] == 0.0
    assert cfg["collision"]["warning_clearance_m"] == 0.05
    np.testing.assert_allclose(model.shoulder_local("left"), [-0.14326, 0.2, 0.92451])
    np.testing.assert_allclose(model.shoulder_local("right"), [-0.14326, -0.2, 0.92451])
    np.testing.assert_allclose(
        model.q_nominal,
        [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],
    )
    for side in ["left", "right"]:
        forward_axis = model.mount_rotation_world(side, 0.0) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(forward_axis, [1.0, 0.0, 0.0], atol=1e-12)


def test_panda_model_rejects_parametric_arm_models():
    cfg = load_baseline_config(profile=PROFILE)
    cfg["panda"]["kinematic_model"] = "serial7"

    with pytest.raises(ValueError, match="original Franka Panda"):
        PandaModel.from_config(cfg)


def test_base_motion_mode_aliases_are_normalized():
    assert normalize_base_motion_mode("A") == "fixed_world"
    assert normalize_base_motion_mode("b") == "fixed_initial"
    assert normalize_base_motion_mode("C") == "limited_holonomic"
    assert normalize_base_motion_mode("D") == "nonholonomic"
    assert normalize_base_motion_mode("E") == "free_holonomic"
    assert normalize_base_motion_mode("non-holonomic") == "nonholonomic"


@pytest.mark.requires_data
def test_base_motion_modes_project_reference_motion():
    cfg = load_baseline_config(profile=PROFILE)
    model = PandaModel.from_config(cfg)
    task = load_task("wiping_table_exercise_force")
    variant = {
        "preferred_reach_fraction": 0.72,
        "forward_offset_m": 0.0,
        "lateral_offset_m": 0.0,
        "yaw_offset_rad": 0.0,
    }

    cfg["base"]["motion_mode"] = "free_holonomic"
    free = estimate_base_motion(task, model, cfg, variant)
    cfg["base"]["motion_mode"] = "fixed_initial"
    fixed = estimate_base_motion(task, model, cfg, variant)
    cfg["base"]["motion_mode"] = "limited_holonomic"
    limited = estimate_base_motion(task, model, cfg, variant)
    cfg["base"]["motion_mode"] = "nonholonomic"
    nonholonomic = estimate_base_motion(task, model, cfg, variant)

    assert fixed["base_motion_mode"] == "fixed_initial"
    assert fixed["path_length_m"] == pytest.approx(0.0, abs=1e-12)
    assert limited["base_motion_mode"] == "limited_holonomic"
    assert limited["path_length_m"] <= cfg["base"]["limited_max_path_m"] + 1e-9
    assert free["path_length_m"] >= limited["path_length_m"] - 1e-9
    assert nonholonomic["base_motion_mode"] == "nonholonomic"

    pose = nonholonomic["pose"]
    deltas = np.diff(pose[:, :2], axis=0)
    headings = pose[1:, 2]
    lateral_steps = -np.sin(headings) * deltas[:, 0] + np.cos(headings) * deltas[:, 1]
    np.testing.assert_allclose(lateral_steps, np.zeros_like(lateral_steps), atol=1e-10)


def test_mujoco_inverse_dynamics_includes_gravity_load():
    cfg = load_baseline_config(profile=PROFILE)
    model = PandaModel.from_config(cfg)
    qdot = np.zeros(7)
    qddot = np.zeros(7)
    tau_left = model.inverse_dynamics(model.q_nominal, qdot, qddot, "left")
    tau_right = model.inverse_dynamics(model.q_nominal, qdot, qddot, "right")

    assert np.max(np.abs(tau_left)) > 1.0
    assert np.max(np.abs(tau_right)) > 1.0
    np.testing.assert_allclose(np.abs(tau_left), np.abs(tau_right), rtol=0.15, atol=0.35)


def test_position_ik_reduces_error_for_reachable_target():
    cfg = load_baseline_config()
    model = PandaModel.from_config(cfg)
    base_pose = np.zeros(3)
    q0 = model.q_nominal.copy()
    target = forward_points(model, q0, "right", base_pose)[-1] + np.array([0.04, -0.02, 0.03])
    _q, err = solve_position_ik(model, target, "right", base_pose, q0, cfg["ik"])
    assert err < 0.04


def test_generated_mujoco_model_matches_reference_fk(tmp_path):
    cfg = load_baseline_config()
    reference = PandaModel.from_config(cfg)
    xml_path = build_baseline_mjcf(cfg, tmp_path / "garmi_dual_panda_baseline.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    qpos = {model.joint(idx).name: int(model.jnt_qposadr[idx]) for idx in range(model.njnt)}
    base_pose = np.array([0.32, -0.18, 0.41])
    left_q = np.array([0.2, -0.45, 0.35, -2.1, 0.15, 1.7, -0.22])
    right_q = np.array([-0.25, -0.32, -0.28, -1.9, -0.1, 1.55, 0.3])

    data.qpos[qpos["base_x"]] = base_pose[0]
    data.qpos[qpos["base_y"]] = base_pose[1]
    data.qpos[qpos["base_yaw"]] = base_pose[2]
    for idx in range(7):
        data.qpos[qpos[f"left_joint{idx + 1}"]] = left_q[idx]
        data.qpos[qpos[f"right_joint{idx + 1}"]] = right_q[idx]
    mujoco.mj_forward(model, data)

    assert model.njnt == 17
    follow_camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "garmi_follow")
    overview_camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
    assert follow_camera >= 0
    assert overview_camera >= 0
    assert model.cam_pos[follow_camera, 0] > 2.5
    assert model.cam_pos[follow_camera, 2] > 3.0
    assert model.cam_pos[overview_camera, 0] > 2.8
    assert model.cam_pos[overview_camera, 2] > 3.3
    for side, q in [("left", left_q), ("right", right_q)]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
        np.testing.assert_allclose(data.site_xpos[site_id], end_effector(reference, q, side, base_pose), atol=1e-12)
        np.testing.assert_allclose(
            data.site_xmat[site_id].reshape(3, 3),
            end_effector_rotation(reference, q, side, base_pose),
            atol=1e-12,
        )


@pytest.mark.requires_data
def test_task_orientations_are_valid_rotation_frames():
    task = load_task("wiping_table_exercise_force")
    for rotations in [task.left_rot, task.right_rot]:
        assert rotations.shape == (task.sample_count, 3, 3)
        gram = np.einsum("nij,nik->njk", rotations, rotations)
        np.testing.assert_allclose(gram, np.repeat(np.eye(3)[None, :, :], task.sample_count, axis=0), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(rotations), np.ones(task.sample_count), atol=1e-8)


@pytest.mark.requires_data
def test_task_active_sides_follow_preprocessing_metadata():
    curtain = load_task("drawing_curtain_force")
    shirt = load_task("tucking_shirt_exercise_force")
    washup = load_task("washing_upper_body_force")

    assert curtain.active_sides() == ("right",)
    assert shirt.active_sides() == ("left",)
    assert washup.active_sides() == ("left", "right")
    assert np.max(np.linalg.norm(washup.side_torque("left"), axis=1)) > 0.0


def test_mount_candidate_is_strictly_mirrored():
    cfg = load_baseline_config(profile=PROFILE)
    candidate = MountCandidate(
        "candidate_test",
        np.array([-0.15, 0.27, 0.97]),
        np.array([0.12, -0.08, 0.19]),
    )
    candidate_cfg = apply_mount_candidate(cfg, candidate)
    model = PandaModel.from_config(candidate_cfg)

    np.testing.assert_allclose(model.shoulder_local("left") * [1.0, -1.0, 1.0], model.shoulder_local("right"))
    np.testing.assert_allclose(model.mount_rotations["right"], MIRROR_Y @ model.mount_rotations["left"] @ MIRROR_Y, atol=1e-12)


def test_exported_mount_candidate_keeps_parallel_performance():
    cfg = load_baseline_config(profile=PROFILE)
    candidate = MountCandidate("candidate_test", np.array([-0.15, 0.24, 0.94]), np.zeros(3))
    exported = apply_mount_candidate(cfg, candidate)
    evaluated = apply_mount_candidate(cfg, candidate, serial_evaluation=True)

    assert exported["performance"]["parallel_workers"] == "auto"
    assert exported["performance"]["progress"] is True
    assert evaluated["performance"]["parallel_workers"] == "off"
    assert evaluated["performance"]["progress"] is False


def test_mount_candidate_generation_accepts_configurable_bounds():
    cfg = load_baseline_config(profile=PROFILE)
    cfg["optimization"] = {
        "mount_bounds": {
            "left_x_m": [-0.16, -0.14],
            "left_y_m": [0.24, 0.26],
            "left_z_m": [0.93, 0.95],
            "left_delta_roll_rad": [-0.02, 0.02],
            "left_delta_pitch_rad": [-0.03, 0.03],
            "left_delta_yaw_rad": [-0.04, 0.04],
        },
        "mount_local_sample_fraction": 0.0,
    }
    candidates = generate_mount_candidates(cfg, samples=8, seed=5, strategy="lhs")

    for candidate in candidates:
        assert -0.16 <= candidate.left_xyz_m[0] <= -0.14
        assert 0.24 <= candidate.left_xyz_m[1] <= 0.26
        assert 0.93 <= candidate.left_xyz_m[2] <= 0.95
        assert -0.02 <= candidate.left_delta_rpy_rad[0] <= 0.02
        assert -0.03 <= candidate.left_delta_rpy_rad[1] <= 0.03
        assert -0.04 <= candidate.left_delta_rpy_rad[2] <= 0.04


def test_garmi_strict_search_uses_expanded_quarter_sphere_mount_space():
    cfg = load_baseline_config(profile=PROFILE)
    bounds = _candidate_bounds(cfg)

    assert bounds["left_y_m"] == (0.18, 0.46)
    assert bounds["left_z_m"] == (0.78, 1.16)
    assert bounds["left_delta_roll_rad"] == (-0.65, 0.65)
    assert bounds["left_delta_pitch_rad"] == pytest.approx((-1.5706963267948966, 1.5706963267948966))
    assert bounds["left_delta_yaw_rad"] == pytest.approx((0.0, 0.5 * np.pi))

    # The general configuration of the paper is bounded to +-5 cm around the
    # original shoulder height.
    bounded = _candidate_bounds(load_baseline_config(profile="stage2_forces_height_bound"))
    assert bounded["left_z_m"] == pytest.approx((0.92451 - 0.05, 0.92451 + 0.05))
    assert {key: value for key, value in bounded.items() if key != "left_z_m"} == {
        key: value for key, value in bounds.items() if key != "left_z_m"
    }


def test_quarter_sphere_mount_generation_points_forward_outer():
    cfg = load_baseline_config(profile=PROFILE)
    cfg["optimization"]["mount_local_sample_fraction"] = 0.0
    candidates = generate_mount_candidates(cfg, samples=18, seed=23, strategy="lhs")

    for candidate in candidates:
        candidate_cfg = apply_mount_candidate(cfg, candidate)
        model = PandaModel.from_config(candidate_cfg)
        left_forward = model.mount_rotation_world("left", 0.0) @ np.array([0.0, 0.0, 1.0])
        right_forward = model.mount_rotation_world("right", 0.0) @ np.array([0.0, 0.0, 1.0])
        assert left_forward[0] >= -1e-9
        assert left_forward[1] >= -1e-9
        assert right_forward[0] >= -1e-9
        assert right_forward[1] <= 1e-9
        np.testing.assert_allclose(np.linalg.norm(left_forward), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(right_forward), 1.0, atol=1e-12)


def test_mount_candidate_generation_supports_adaptive_global_strategy():
    cfg = load_baseline_config(profile=PROFILE)
    candidates = generate_mount_candidates(cfg, samples=9, seed=11, strategy="sobol")

    assert len(candidates) == 9
    assert candidates[0].candidate_id == "candidate_000_strict_garmi"
    assert all(candidate.left_xyz_m.shape == (3,) for candidate in candidates)
    assert all(candidate.left_delta_rpy_rad.shape == (3,) for candidate in candidates)


def test_candidate_frame_gets_pareto_rank_and_multi_objective_score():
    ranked = _rank_candidate_frame(
        pd.DataFrame(
            [
                {
                    "candidate_id": "bad",
                    "failed_task_count": 2,
                    "mean_position_error_m": 0.04,
                    "max_position_error_m": 0.10,
                    "mean_orientation_error_rad": 0.4,
                    "max_orientation_error_rad": 1.0,
                    "max_torque_utilization": 0.8,
                    "max_joint_speed_utilization": 0.9,
                    "min_clearance_m": 0.02,
                    "mount_plausibility_penalty": 0.0,
                    "optimization_score": 100.0,
                },
                {
                    "candidate_id": "good",
                    "failed_task_count": 0,
                    "mean_position_error_m": 0.03,
                    "max_position_error_m": 0.09,
                    "mean_orientation_error_rad": 0.3,
                    "max_orientation_error_rad": 0.8,
                    "max_torque_utilization": 0.7,
                    "max_joint_speed_utilization": 0.8,
                    "min_clearance_m": 0.03,
                    "mount_plausibility_penalty": 0.0,
                    "optimization_score": 10.0,
                },
            ]
        )
    )

    assert ranked.iloc[0]["candidate_id"] == "good"
    assert "pareto_rank" in ranked
    assert "multi_objective_score" in ranked


def test_candidate_ranking_prioritizes_hard_constraints_before_coverage():
    ranked = _rank_candidate_frame(
        pd.DataFrame(
            [
                {
                    "candidate_id": "baseline",
                    "failed_task_count": 6,
                    "success_count": 26,
                    "task_count": 32,
                    "mean_position_error_m": 0.008,
                    "max_position_error_m": 0.18,
                    "mean_orientation_error_rad": 0.18,
                    "max_orientation_error_rad": 1.51,
                    "max_torque_utilization": 0.97,
                    "max_joint_speed_utilization": 0.90,
                    "min_clearance_m": 0.02,
                    "min_clearance_margin_m": 0.02,
                    "mount_min_body_clearance_m": 0.005,
                    "mount_plausibility_penalty": 0.0,
                    "optimization_score": 1920.0,
                },
                {
                    "candidate_id": "lower_constraint_but_worse_coverage",
                    "failed_task_count": 8,
                    "success_count": 24,
                    "task_count": 32,
                    "mean_position_error_m": 0.008,
                    "max_position_error_m": 0.74,
                    "mean_orientation_error_rad": 0.35,
                    "max_orientation_error_rad": 2.25,
                    "max_torque_utilization": 0.76,
                    "max_joint_speed_utilization": 0.83,
                    "min_clearance_m": -0.20,
                    "min_clearance_margin_m": -0.20,
                    "mount_min_body_clearance_m": 0.14,
                    "mount_plausibility_penalty": 0.0,
                    "optimization_score": 2550.0,
                },
            ]
        )
    )

    assert ranked.iloc[0]["candidate_id"] == "baseline"
    assert bool(ranked[ranked["candidate_id"] == "lower_constraint_but_worse_coverage"]["eligible_candidate"].iloc[0]) is False


def test_clearance_warning_does_not_make_candidate_ineligible():
    ranked = _rank_candidate_frame(
        pd.DataFrame(
            [
                {
                    "candidate_id": "close_but_not_penetrating",
                    "failed_task_count": 0,
                    "success_count": 1,
                    "task_count": 1,
                    "mean_position_error_m": 0.01,
                    "max_position_error_m": 0.02,
                    "mean_orientation_error_rad": 0.1,
                    "max_orientation_error_rad": 0.2,
                    "max_torque_utilization": 0.5,
                    "max_joint_speed_utilization": 0.5,
                    "min_clearance_m": 0.015,
                    "min_clearance_margin_m": 0.015,
                    "min_warning_clearance_margin_m": -0.035,
                    "mount_min_body_clearance_m": 0.02,
                    "optimization_score": 1.0,
                }
            ]
        )
    )

    assert bool(ranked.iloc[0]["eligible_candidate"]) is True
    assert ranked.iloc[0]["clearance_violation_m"] == 0.0
    assert ranked.iloc[0]["clearance_warning_violation_m"] > 0.0


def test_candidate_ranking_excludes_extreme_torque_even_with_high_coverage():
    ranked = _rank_candidate_frame(
        pd.DataFrame(
            [
                {
                    "candidate_id": "eligible",
                    "failed_task_count": 2,
                    "success_count": 30,
                    "task_count": 32,
                    "mean_position_error_m": 0.02,
                    "max_position_error_m": 0.10,
                    "mean_orientation_error_rad": 0.2,
                    "max_orientation_error_rad": 1.0,
                    "max_torque_utilization": 0.9,
                    "max_joint_speed_utilization": 0.9,
                    "min_clearance_m": 0.01,
                    "mount_min_body_clearance_m": 0.03,
                    "mount_plausibility_penalty": 0.0,
                    "optimization_score": 900.0,
                },
                {
                    "candidate_id": "extreme_torque",
                    "failed_task_count": 0,
                    "success_count": 32,
                    "task_count": 32,
                    "mean_position_error_m": 0.01,
                    "max_position_error_m": 0.08,
                    "mean_orientation_error_rad": 0.1,
                    "max_orientation_error_rad": 0.8,
                    "max_torque_utilization": 25.0,
                    "max_joint_speed_utilization": 0.8,
                    "min_clearance_m": 0.01,
                    "mount_min_body_clearance_m": 0.03,
                    "mount_plausibility_penalty": 0.0,
                    "optimization_score": 100.0,
                },
            ]
        )
    )

    assert ranked.iloc[0]["candidate_id"] == "eligible"
    assert bool(ranked[ranked["candidate_id"] == "extreme_torque"]["eligible_candidate"].iloc[0]) is False


def test_fidelity_constraint_guards_make_quick_candidates_conservative():
    ranked = _rank_candidate_frame(
        pd.DataFrame(
            [
                {
                    "candidate_id": "quick_guarded",
                    "failed_task_count": 0,
                    "success_count": 1,
                    "task_count": 1,
                    "optimization_score": 1.0,
                    "max_torque_utilization": 0.75,
                    "constraint_torque_utilization_limit": 0.70,
                    "max_joint_speed_utilization": 0.70,
                    "constraint_joint_speed_utilization_limit": 0.80,
                    "min_clearance_m": 0.04,
                    "constraint_min_clearance_m": 0.035,
                    "mount_min_body_clearance_m": 0.02,
                    "constraint_mount_min_body_clearance_m": 0.015,
                }
            ]
        )
    )

    assert bool(ranked.iloc[0]["eligible_candidate"]) is False
    assert ranked.iloc[0]["torque_violation"] > 0.0


def test_promotion_keeps_top_success_candidate():
    cfg = load_baseline_config(profile=PROFILE)
    candidates = {
        "candidate_000_strict_garmi": MountCandidate("candidate_000_strict_garmi", np.array([-0.14326, 0.20, 0.92451]), np.zeros(3)),
        "ranked_best": MountCandidate("ranked_best", np.array([-0.14, 0.22, 0.94]), np.zeros(3)),
        "top_success": MountCandidate("top_success", np.array([-0.12, 0.24, 0.96]), np.zeros(3)),
    }
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_000_strict_garmi",
                "failed_task_count": 4,
                "success_count": 28,
                "task_count": 32,
                "optimization_score": 1200.0,
                "max_torque_utilization": 0.9,
                "max_joint_speed_utilization": 0.9,
                "min_clearance_m": 0.01,
                "mount_min_body_clearance_m": 0.01,
            },
            {
                "candidate_id": "ranked_best",
                "failed_task_count": 3,
                "success_count": 29,
                "task_count": 32,
                "optimization_score": 800.0,
                "max_torque_utilization": 0.9,
                "max_joint_speed_utilization": 0.9,
                "min_clearance_m": 0.01,
                "mount_min_body_clearance_m": 0.01,
            },
            {
                "candidate_id": "top_success",
                "failed_task_count": 2,
                "success_count": 30,
                "task_count": 32,
                "optimization_score": 5000.0,
                "max_torque_utilization": 0.95,
                "max_joint_speed_utilization": 0.95,
                "min_clearance_m": 0.05,
                "min_clearance_margin_m": 0.05,
                "mount_min_body_clearance_m": 0.02,
            },
        ]
    )

    promoted = _promoted_candidates(frame, candidates, minimum=1, fraction=0.34)

    assert "top_success" in [candidate.candidate_id for candidate in promoted]
    assert cfg["panda"]["dof"] == 7


def test_simulation_pass_status_uses_tracking_clipping_and_contacts():
    cfg = load_baseline_config(profile=PROFILE)
    passing = {
        "simulation_ok": True,
        "mean_tracking_error_m": 0.01,
        "max_tracking_error_m": 0.04,
        "mean_orientation_error_rad": 0.1,
        "max_orientation_error_rad": 0.2,
        "max_joint_torque_clip_fraction": 0.0,
        "max_contact_count": 0,
    }
    failed = passing | {"max_tracking_error_m": 0.30, "max_joint_torque_clip_fraction": 0.2}

    ok, reasons = _simulation_pass_status(passing, cfg)
    bad, bad_reasons = _simulation_pass_status(failed, cfg)

    assert ok is True
    assert reasons == "ok"
    assert bad is False
    assert "max_tracking_error" in bad_reasons
    assert "joint_torque_clipping" in bad_reasons


def test_generated_mujoco_model_can_enable_contact_geoms(tmp_path):
    cfg = load_baseline_config(profile=PROFILE)
    cfg["simulation"] = {"enable_contacts": True}
    xml_path = build_baseline_mjcf(cfg, tmp_path / "contacts.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    contypes = [int(model.geom_contype[idx]) for idx in range(model.ngeom)]

    assert max(contypes) > 0


def test_group_task_names_preserve_duplicates_and_apply_exclusions(tmp_path):
    paths = ProjectPaths(tmp_path)
    write_yaml(
        paths.config / "task_groups.yaml",
        {
            "groups": {
                "custom": {
                    "tasks": ["task_a", "task_a", "task_b", "task_c"],
                    "excluded_tasks": ["task_b"],
                },
                "all_tasks": {
                    "tasks": ["task_a", "task_b", "task_c"],
                    "excluded_tasks": ["task_a"],
                },
            }
        },
    )

    assert group_task_names("custom", paths) == ["task_a", "task_a", "task_c"]
    assert global_task_names(paths) == ["task_b", "task_c"]


def test_cli_exclude_task_names_accepts_repeated_and_csv_names():
    task_names = ["washing_upper_body_force", "washing_upper_body_kin", "wiping_table_exercise_force"]

    assert _exclude_task_names(task_names, ["washing_upper_body_force"]) == [
        "washing_upper_body_kin",
        "wiping_table_exercise_force",
    ]
    assert _exclude_task_names(task_names, ["washing_upper_body_force", "washing_upper_body_kin"]) == [
        "wiping_table_exercise_force"
    ]
    assert _exclude_task_names(task_names, ["washing_upper_body_force,washing_upper_body_kin"]) == [
        "wiping_table_exercise_force"
    ]
    assert _exclude_task_names(task_names, None) == task_names

    with pytest.raises(SystemExit, match="not found"):
        _exclude_task_names(task_names, ["does_not_exist"])
    with pytest.raises(SystemExit, match="nothing to run"):
        _exclude_task_names(task_names, [",".join(task_names)])


def test_gui_camera_toggle_sets_mujoco_camera_mode():
    cfg = load_baseline_config(profile=PROFILE)
    overview = _apply_gui_overrides(cfg, {"simulation_use_garmi_follow": False})
    follow = _apply_gui_overrides(cfg, {"simulation_use_garmi_follow": True})

    assert overview["simulation"]["render_camera"] == "overview"
    assert overview["simulation"]["interactive_camera"] == "overview"
    assert follow["simulation"]["render_camera"] == "garmi_follow"
    assert follow["simulation"]["interactive_camera"] == "garmi_follow"


def test_gui_task_report_index_and_top_candidate_selection(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate_000_strict_garmi",
                "optimization_score": 1.0,
                "failed_task_count": 0,
                "left_x_m": -0.14326,
                "left_y_m": 0.20,
                "left_z_m": 0.92451,
                "left_delta_roll_rad": 0.0,
                "left_delta_pitch_rad": 0.0,
                "left_delta_yaw_rad": 0.0,
            },
            {
                "candidate_id": "candidate_010",
                "optimization_score": 2.0,
                "failed_task_count": 1,
                "left_x_m": -0.12,
                "left_y_m": 0.28,
                "left_z_m": 0.96,
                "left_delta_roll_rad": 0.1,
                "left_delta_pitch_rad": -0.2,
                "left_delta_yaw_rad": 0.3,
            },
            {
                "candidate_id": "candidate_011",
                "optimization_score": 3.0,
                "failed_task_count": 1,
                "left_x_m": -0.11,
                "left_y_m": 0.30,
                "left_z_m": 0.98,
                "left_delta_roll_rad": -0.1,
                "left_delta_pitch_rad": 0.2,
                "left_delta_yaw_rad": -0.3,
            },
        ]
    ).to_csv(run_dir / "full_validation_candidates.csv", index=False)

    candidates = _top_mount_candidates_for_reports(run_dir, top_count=1)
    assert [candidate.candidate_id for candidate in candidates] == ["candidate_010"]

    task_report = tmp_path / "candidate_010" / "task_a" / "report.html"
    video = tmp_path / "candidate_010" / "task_a" / "impedance_playback.mp4"
    task_report.parent.mkdir(parents=True)
    task_report.write_text("<html></html>", encoding="utf-8")
    video.write_bytes(b"video")
    pd.DataFrame(
        [
            {
                "task_name": "task_a",
                "simulation_ok": True,
                "frames": 1,
                "mean_left_tracking_error_m": 0.01,
                "mean_right_tracking_error_m": 0.02,
                "max_left_tracking_error_m": 0.03,
                "max_right_tracking_error_m": 0.04,
                "error": "",
            }
        ]
    ).to_csv(tmp_path / "candidate_010" / "batch_simulation_summary.csv", index=False)
    rows = _simulation_rows("candidate_010", tmp_path / "candidate_010", tmp_path / "candidate_010.yaml")
    assert rows[0]["config_path"].endswith("candidate_010.yaml")

    index = _write_task_report_index(
        tmp_path / "index",
        "Task Reports",
        rows,
    )
    html = index.read_text(encoding="utf-8")
    assert "candidate_010" in html
    assert "task_a" in html
    assert "video" in html
    assert (tmp_path / "index" / "task_video_reports.csv").exists()


def test_design_candidate_hash_is_stable_and_sensitive():
    cfg = load_baseline_config(profile=PROFILE)
    candidate = generate_design_candidates(cfg, samples=2, seed=13, strategy="lhs", optimize_controller=True)[1]

    first = candidate_hash(candidate, ["task_a", "task_b"], "quick", "single")
    second = candidate_hash(candidate, ["task_a", "task_b"], "quick", "single")
    changed = candidate_hash(candidate, ["task_a", "task_b"], "full", "single")

    assert first == second
    assert first != changed


def test_design_candidate_applies_base_ik_and_controller_overlay():
    cfg = load_baseline_config(profile=PROFILE)
    mount = MountCandidate("candidate_design", np.array([-0.15, 0.24, 0.96]), np.array([0.05, -0.02, 0.03]))
    candidate = DesignCandidate(
        candidate_id="candidate_design",
        mount=mount,
        base_policy={
            "preferred_reach_fraction": 0.68,
            "forward_offset_m": 0.04,
            "lateral_offset_m": -0.03,
            "yaw_offset_rad": 0.12,
        },
        ik={"damping": 0.05, "step_limit_rad": 0.16},
        collision_ik={"avoidance_target_m": 0.07},
        controller={
            "position_kp_multiplier": 1.2,
            "base_kp_multiplier": 0.8,
            "base_kd_multiplier": 1.1,
            "max_task_force_n": 300.0,
        },
    )

    out = apply_design_candidate(cfg, candidate, base_search="full")

    # The candidate base policy defines the center of the same per-task
    # base-placement grid the strict baseline gets.
    np.testing.assert_allclose(out["base"]["preferred_reach_fraction_candidates"], [0.58, 0.68, 0.78])
    np.testing.assert_allclose(out["base"]["forward_offset_candidates_m"], [-0.07, 0.04, 0.15])
    np.testing.assert_allclose(out["base"]["lateral_offset_candidates_m"], [-0.23, -0.03, 0.17])
    assert out["base"]["yaw_offset_candidates_rad"] == [0.12]

    out_single = apply_design_candidate(cfg, candidate, base_search="single")
    assert out_single["base"]["preferred_reach_fraction_candidates"] == [0.68]
    assert out_single["base"]["forward_offset_candidates_m"] == [0.04]
    assert out["ik"]["damping"] == 0.05
    assert out["ik"]["step_limit_rad"] == 0.16
    assert out["collision"]["collision_aware_ik"] is True
    assert out["collision"]["avoidance_target_m"] == 0.07
    np.testing.assert_allclose(out["controller"]["position_kp_npm"], [1440.0, 1440.0, 1440.0])
    np.testing.assert_allclose(out["controller"]["base_pd"]["kp"], [1920.0, 1920.0, 680.0])
    np.testing.assert_allclose(out["controller"]["base_pd"]["kd"], [572.0, 572.0, 176.0])
    assert out["controller"]["max_task_force_n"] == 300.0


def test_generate_design_candidates_keeps_strict_garmi_baseline_first():
    cfg = load_baseline_config(profile=PROFILE)
    candidates = generate_design_candidates(cfg, samples=5, seed=19, strategy="sobol", optimize_controller=True)

    assert candidates[0].candidate_id == "candidate_000_strict_garmi"
    assert candidates[0].source == "strict_garmi"
    assert isinstance(candidates[1], DesignCandidate)
    assert candidates[1].base_policy
    assert candidates[1].controller


def test_design_candidate_from_unit_vector_stays_inside_bounds():
    cfg = load_baseline_config(profile=PROFILE)
    specs = _design_bounds(cfg, optimize_controller=False)
    unit = np.full(len(specs), 0.5, dtype=float)

    candidate = _design_candidate_from_unit_vector(cfg, "candidate_unit", specs, unit, "test")
    roundtrip = _candidate_vector(candidate, specs, cfg)

    assert candidate.candidate_id == "candidate_unit"
    assert candidate.base_policy
    assert np.all(roundtrip >= 0.0)
    assert np.all(roundtrip <= 1.0)


def test_genetic_population_and_selection_keep_strict_baseline():
    cfg = load_baseline_config(profile=PROFILE)
    specs = _design_bounds(cfg, optimize_controller=False)
    genomes = _genetic_initial_genomes(
        cfg,
        specs,
        population_size=5,
        seed=31,
        search_strategy="sobol",
        local_seed_fraction=0.4,
    )
    assert len(genomes) == 5
    np.testing.assert_allclose(genomes[0], _candidate_vector(generate_design_candidates(cfg, 1, 0)[0], specs, cfg))
    assert all(np.all(genome >= 0.0) and np.all(genome <= 1.0) for genome in genomes)

    frame = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_000_strict_garmi",
                "failed_task_count": 4,
                "success_count": 28,
                "task_count": 32,
                "optimization_score": 400.0,
                "max_torque_utilization": 0.8,
                "max_joint_speed_utilization": 0.9,
                "min_clearance_m": 0.01,
                "mount_min_body_clearance_m": 0.02,
            },
            {
                "candidate_id": "candidate_a",
                "failed_task_count": 3,
                "success_count": 29,
                "task_count": 32,
                "optimization_score": 300.0,
                "max_torque_utilization": 1.4,
                "max_joint_speed_utilization": 0.9,
                "min_clearance_m": 0.01,
                "mount_min_body_clearance_m": 0.02,
            },
            {
                "candidate_id": "candidate_b",
                "failed_task_count": 5,
                "success_count": 27,
                "task_count": 32,
                "optimization_score": 500.0,
                "max_torque_utilization": 0.7,
                "max_joint_speed_utilization": 0.8,
                "min_clearance_m": 0.02,
                "mount_min_body_clearance_m": 0.02,
            },
        ]
    )
    survivors = _genetic_survivor_ids(frame, count=2, include_strict=True)
    assert "candidate_000_strict_garmi" in survivors

    genome_by_id = {
        "candidate_000_strict_garmi": genomes[0],
        "candidate_a": genomes[1],
        "candidate_b": genomes[2],
    }
    next_generation = _genetic_next_generation(
        frame,
        genome_by_id,
        population_size=5,
        seed=31,
        generation=1,
        elite_fraction=0.4,
        mutation_sigma=0.05,
        crossover_rate=0.8,
        random_immigrant_fraction=0.0,
        tournament_size=2,
    )
    assert len(next_generation) == 5
    assert all(np.all(genome >= 0.0) and np.all(genome <= 1.0) for genome in next_generation)


def test_genetic_seed_genomes_load_candidate_records(tmp_path):
    cfg = load_baseline_config(profile=PROFILE)
    specs = _design_bounds(cfg, optimize_controller=False)
    seed_run = tmp_path / "seed_run"
    seed_run.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate_seed",
                "left_x_m": -0.06,
                "left_y_m": 0.34,
                "left_z_m": 0.98,
                "left_delta_roll_rad": 0.01,
                "left_delta_pitch_rad": -0.02,
                "left_delta_yaw_rad": 0.03,
                "left_search_roll_rad": 0.25,
                "left_search_elevation_rad": 0.12,
                "left_search_azimuth_rad": 0.40,
                "base_preferred_reach_fraction": 0.72,
                "base_forward_offset_m": 0.04,
                "base_lateral_offset_m": -0.03,
                "base_yaw_offset_rad": 0.08,
                "failed_task_count": 4,
                "success_count": 28,
                "task_count": 32,
                "optimization_score": 10.0,
                "max_torque_utilization": 0.8,
                "max_joint_speed_utilization": 0.7,
                "min_clearance_m": 0.05,
                "mount_min_body_clearance_m": 0.05,
            }
        ]
    ).to_csv(seed_run / "full_validation_candidates.csv", index=False)

    genomes = _load_genetic_seed_genomes(
        cfg,
        specs,
        seed_run=seed_run,
        seed_candidates=["candidate_seed"],
        seed_top=1,
    )

    assert len(genomes) == 1
    assert np.all(genomes[0] >= 0.0)
    assert np.all(genomes[0] <= 1.0)
