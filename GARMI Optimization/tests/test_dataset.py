"""Dataset discovery, study split and processed trials."""
import json

import pytest

from garmi_optimization.config import PATHS, load_group_config, load_task_frames_config
from garmi_optimization.dataset import (
    CATEGORIES,
    build_trial_manifest,
    discover_trials,
    infer_explicit_end_effector,
    study_assignment,
    trial_by_id,
    trial_files,
)


def test_explicit_end_effector_from_sensor_position():
    assert infer_explicit_end_effector("right hand") == "right"
    assert infer_explicit_end_effector("Therapist left wrist") == "left"
    assert infer_explicit_end_effector("rigth hand") == "right"
    assert infer_explicit_end_effector("on the chair") is None
    assert infer_explicit_end_effector(None) is None


def test_study_split_sizes():
    study = study_assignment()
    splits = [item["split"] for item in study.values()]
    assert splits.count("optimization") == 55
    assert splits.count("heldout") == 26
    # every held-out trial shares its repeat group with an optimization trial
    optimization_groups = {item["repeat_group"] for item in study.values() if item["split"] == "optimization"}
    for item in study.values():
        if item["split"] == "heldout":
            assert item["repeat_group"] in optimization_groups


def test_category_groups_partition_the_optimization_split():
    groups = load_group_config()["groups"]
    optimization = set(groups["all_tasks"]["tasks"])
    heldout = set(groups["heldout_all"]["tasks"])
    by_category = [set(groups[category.slug]["tasks"]) for category in CATEGORIES]
    assert set().union(*by_category) == optimization
    assert sum(len(tasks) for tasks in by_category) == len(optimization)
    assert not optimization & heldout


@pytest.mark.requires_dataset
def test_dataset_trials_and_files():
    records = discover_trials()
    assert len(records) == 86
    assert all(record.status == "valid" for record in records)
    for record in records:
        mocap, ft = trial_files(record)
        assert mocap.is_file()
        assert record.force_measurement == (ft is not None)
        if ft is not None:
            assert ft.name == f"{record.trial_id}_ft.csv"
    record = trial_by_id("slicing_bread_force")
    assert record is not None and record.category == "meals" and record.force_measurement


@pytest.mark.requires_dataset
def test_configuration_refers_to_dataset_trials():
    trials = {record.trial_id for record in discover_trials()}
    config = load_group_config()
    for name, group in config["groups"].items():
        missing = set(group["tasks"]) - trials
        assert not missing, f"group {name}: {sorted(missing)}"
    for name, members in config["repeat_groups"].items():
        assert set(members) <= trials, name
    annotated = set(load_task_frames_config().get("tasks", {}))
    assert annotated and annotated <= trials


@pytest.mark.requires_dataset
@pytest.mark.requires_data
def test_processed_trials_match_the_manifest():
    manifest = {item["trial_id"]: item for item in build_trial_manifest()}
    processed = PATHS.processed_data
    folders = {path.parent.name for path in processed.glob("*/task_data.npz")}
    assert folders == set(manifest)
    for trial_id in folders:
        metadata = json.loads((processed / trial_id / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["trial_id"] == trial_id
        assert metadata["split"] == manifest[trial_id]["split"]
        assert metadata["force_measurement"] == manifest[trial_id]["force_measurement"]
