"""Command-line interface ``garmi-opt``.

Typical workflow (see the repository README for details)::

    garmi-opt preprocess                      # dataset -> data/processed
    garmi-opt baseline --global               # coverage of the original GARMI mounting
    garmi-opt optimize-mounts --global ...    # mount optimization
    garmi-opt generality --config ...         # price of generality
    garmi-opt simulate-batch --global ...     # MuJoCo tracking simulation
    garmi-opt gui                             # browser tool on http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
from pathlib import Path

from garmi_optimization.evaluation import run_baseline_all_groups, run_baseline_global, run_baseline_group, set_base_motion_mode
from garmi_optimization.config import DEFAULT_PROFILE, PATHS, ensure_dirs, load_baseline_config
from garmi_optimization.gui import run_gui
from garmi_optimization.mujoco_backend import build_baseline_mjcf, simulate_batch, simulate_task_cli, view_impedance_task
from garmi_optimization.optimization import run_mount_optimization
from garmi_optimization.reporting import run_report_suite
from garmi_optimization.tasks import global_task_names, group_task_names


BASE_MOTION_MODE_CHOICES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "a",
    "b",
    "c",
    "d",
    "e",
    "mode_a",
    "mode_b",
    "mode_c",
    "mode_d",
    "mode_e",
    "fixed_world",
    "fixed_initial",
    "limited_holonomic",
    "nonholonomic",
    "free_holonomic",
]


def _add_base_motion_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-motion-mode",
        choices=BASE_MOTION_MODE_CHOICES,
        help=(
            "Base mobility model: A=fixed world, B=fixed after initial placement, "
            "C=limited/slow holonomic, D=nonholonomic forward/backward plus yaw, "
            "E=free holonomic."
        ),
    )


def _load_cli_config(args: argparse.Namespace) -> dict:
    cfg = load_baseline_config(profile=args.baseline_profile)
    if getattr(args, "base_motion_mode", None):
        set_base_motion_mode(cfg, args.base_motion_mode)
    if getattr(args, "no_contacts", False):
        # Without contact geoms MuJoCo cannot detect self-contact, so the
        # tracking criterion reduces to tracking errors and torque clipping.
        cfg.setdefault("simulation", {})["enable_contacts"] = False
    return cfg


def _exclude_task_names(task_names: list[str], excludes: list[str] | None) -> list[str]:
    """Remove the trial IDs given via --exclude-task (repeatable, comma-separated)."""
    if not excludes:
        return list(task_names)
    requested = {item.strip() for value in excludes for item in str(value).split(",") if item.strip()}
    if not requested:
        return list(task_names)
    missing = sorted(requested - set(task_names))
    if missing:
        raise SystemExit(f"Excluded trial(s) not found in the selected scope: {', '.join(missing)}")
    filtered = [task_name for task_name in task_names if task_name not in requested]
    if not filtered:
        raise SystemExit("All selected trials were excluded; nothing to run.")
    return filtered


def _cmd_baseline(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = _load_cli_config(args)
    outputs = []
    if args.all_groups:
        outputs.extend(run_baseline_all_groups(cfg=cfg, baseline_profile=args.baseline_profile))
    if args.global_opt:
        outputs.append(run_baseline_global(cfg=cfg, baseline_profile=args.baseline_profile))
    if args.group:
        outputs.append(run_baseline_group(args.group, cfg=cfg, baseline_profile=args.baseline_profile))
    if not outputs:
        raise SystemExit("Choose --group <name>, --all-groups, or --global.")
    for output in outputs:
        print(f"baseline_run: {output}")
        print(f"report: {output / 'report.html'}")


def _cmd_open(args: argparse.Namespace) -> None:
    path = Path(args.path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    print(path)


def _cmd_gui(args: argparse.Namespace) -> None:
    run_gui(host=args.host, port=args.port)


def _cmd_mujoco_model(args: argparse.Namespace) -> None:
    ensure_dirs()
    output = Path(args.output).resolve() if args.output else PATHS.resources / "mujoco" / "generated" / "garmi_dual_panda_baseline.xml"
    path = build_baseline_mjcf(_load_cli_config(args), output)
    print(f"mujoco_model: {path}")


def _cmd_simulate(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = _load_cli_config(args)
    output = simulate_task_cli(args.task, render=not args.no_video, cfg=cfg, baseline_profile=args.baseline_profile)
    print(f"simulation_run: {output}")
    print(f"report: {output / 'report.html'}")
    print(f"metrics: {output / 'simulation_metrics.csv'}")
    if not args.no_video:
        print(f"video: {output / 'impedance_playback.mp4'}")


def _cmd_simulate_batch(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = _load_cli_config(args)
    if args.global_opt:
        group_name = "all_tasks"
        task_names = global_task_names()
    else:
        if not args.group:
            raise SystemExit("Choose --group <name> or --global.")
        group_name = args.group
        task_names = group_task_names(group_name)
    task_names = _exclude_task_names(task_names, args.exclude_task)
    output = simulate_batch(task_names, group_name, workers=args.workers, render=args.video, cfg=cfg, baseline_profile=args.baseline_profile)
    print(f"simulation_batch_run: {output}")
    print(f"report: {output / 'report.html'}")
    print(f"summary: {output / 'batch_simulation_summary.csv'}")


def _cmd_view_simulation(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = _load_cli_config(args)
    output = view_impedance_task(
        args.task,
        cfg=cfg,
        baseline_profile=args.baseline_profile,
        realtime_factor=args.realtime_factor,
        camera=args.camera,
    )
    print(f"interactive_simulation_run: {output}")
    print(f"model: {output / 'garmi_dual_panda_baseline.xml'}")


def _cmd_report_all(args: argparse.Namespace) -> None:
    ensure_dirs()
    output = run_report_suite(
        group_names=args.group,
        all_groups=args.all_groups,
        global_opt=args.global_opt,
        baseline_profile=args.baseline_profile,
        workers=args.workers,
        render_videos=not args.no_video,
        base_motion_mode=args.base_motion_mode,
    )
    print(f"report_suite_run: {output}")
    print(f"index: {output / 'index.html'}")
    print(f"summary: {output / 'report_suite.csv'}")
    print(f"videos: {output / 'videos.csv'}")


def _cmd_optimize_mounts(args: argparse.Namespace) -> None:
    ensure_dirs()
    cfg = _load_cli_config(args)
    if getattr(args, "robust_ensemble", None) is not None:
        cfg.setdefault("robustness", {})["ensemble"] = int(args.robust_ensemble)
    if args.global_opt:
        group_name = "all_tasks"
        task_names = global_task_names()
    else:
        if not args.group:
            raise SystemExit("Choose --group <name> or --global.")
        group_name = args.group
        task_names = group_task_names(group_name)
    task_names = _exclude_task_names(task_names, args.exclude_task)
    output = run_mount_optimization(
        group_name,
        task_names,
        cfg=cfg,
        baseline_profile=args.baseline_profile,
        samples=args.samples,
        workers=args.workers,
        seed=args.seed,
        base_search=args.base_search,
        validate_top=args.validate_top,
        search_strategy=args.search_strategy,
        mujoco_validate_top=args.mujoco_validate_top,
        mujoco_workers=args.mujoco_workers,
        mujoco_render=args.mujoco_video,
        mujoco_contacts=args.mujoco_contacts,
        optimizer=args.optimizer,
        budget_hours=args.budget_hours,
        max_evaluations=args.max_evaluations,
        optimize_controller=args.optimize_controller,
        resume=args.resume,
        ga_population=args.ga_population,
        ga_generations_quick=args.ga_generations_quick,
        ga_generations_reduced=args.ga_generations_reduced,
        ga_elite_fraction=args.ga_elite_fraction,
        ga_mutation_sigma=args.ga_mutation_sigma,
        ga_crossover_rate=args.ga_crossover_rate,
        ga_local_seeds=not args.ga_no_local_seeds,
        ga_seed_run=args.ga_seed_run,
        ga_seed_candidates=args.ga_seed_candidate,
        ga_seed_top=args.ga_seed_top,
    )
    print(f"mount_optimization_run: {output}")
    print(f"report: {output / 'report.html'}")
    print(f"best_config: {output / 'best_config.yaml'}")
    print(f"candidates: {output / 'candidates.csv'}")
    if (output / "pareto_candidates.csv").exists():
        print(f"pareto_candidates: {output / 'pareto_candidates.csv'}")
    if (output / "evaluation_archive.csv").exists():
        print(f"evaluation_archive: {output / 'evaluation_archive.csv'}")


def _cmd_sensitivity(args: argparse.Namespace) -> None:
    ensure_dirs()
    from garmi_optimization.sensitivity import run_sensitivity

    if args.global_opt:
        group_name = "all_tasks"
        task_names = global_task_names()
    else:
        if not args.group:
            raise SystemExit("Choose --group <name> or --global.")
        group_name = args.group
        task_names = group_task_names(group_name)
    task_names = _exclude_task_names(task_names, args.exclude_task)
    output = run_sensitivity(
        group_name,
        task_names,
        baseline_profile=args.baseline_profile,
        trajectories=args.trajectories,
        levels=args.levels,
        seed=args.seed,
        workers=args.workers,
        base_motion_mode=args.base_motion_mode,
    )
    print(f"sensitivity_run: {output}")
    print(f"table: {output / 'sensitivity.csv'}")
    print(f"tornado: {output / 'sensitivity_tornado.html'}")


def _cmd_generality(args: argparse.Namespace) -> None:
    ensure_dirs()
    from garmi_optimization.generality import run_generality_report

    configs: dict[str, str] = {}
    for item in args.config:
        if "=" not in item:
            raise SystemExit(f"--config expects label=profile_or_path, got '{item}'")
        label, _, profile = item.partition("=")
        configs[label.strip()] = profile.strip()
    groups = list(args.group or [])
    if not groups:
        from garmi_optimization.config import load_group_config

        groups = [name for name in load_group_config().get("groups", {}) if name != "all_tasks"]
    output = run_generality_report(configs, groups, workers=args.workers)
    print(f"generality_run: {output}")
    print(f"coverage_matrix: {output / 'coverage_matrix.csv'}")
    print(f"price_of_generality: {output / 'price_of_generality.csv'}")
    print(f"report: {output / 'generality_report.html'}")


def _cmd_preprocess(args: argparse.Namespace) -> None:
    from garmi_optimization.dataset import dataset_root
    from garmi_optimization.preprocessing.pipeline import preprocess_dataset

    output = Path(args.output).resolve() if args.output else None
    outputs, status = preprocess_dataset(
        trials=list(args.trial or []) or None,
        output_dir=output,
        n_points=args.points,
        continue_on_error=not args.stop_on_error,
    )
    print(f"dataset: {dataset_root()}")
    print(f"processed_trials: {len(outputs)}")
    print(f"output: {outputs[0].parent if outputs else output}")
    print(f"status: {status}")


def _cmd_trials(args: argparse.Namespace) -> None:
    from garmi_optimization.dataset import build_trial_manifest, dataset_root

    manifest = build_trial_manifest()
    print(f"dataset: {dataset_root()}")
    header = f"{'trial':42s} {'category':26s} {'force':5s} {'split':12s} status"
    print(header)
    print("-" * len(header))
    for item in manifest:
        if args.split and (item.get("split") or "unused") != args.split:
            continue
        force = "yes" if item["force_measurement"] else "no"
        status = item["status"] if item["status"] == "valid" else f"{item['status']} ({item['skip_reason']})"
        print(f"{item['trial_id']:42s} {item['category_name']:26s} {force:5s} {str(item.get('split') or 'unused'):12s} {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garmi-opt")
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Convert the raw dataset into processed trials (data/processed).")
    preprocess.add_argument("--trial", action="append", help="Trial to process. Repeat for several; defaults to all valid trials.")
    preprocess.add_argument("--points", type=int, help="Waypoints per trial (default: preprocessing.default_number_of_task_points = 50).")
    preprocess.add_argument("--output", help="Output folder. Defaults to data/processed.")
    preprocess.add_argument("--stop-on-error", action="store_true", help="Abort at the first failing trial.")
    preprocess.set_defaults(func=_cmd_preprocess)

    trials = sub.add_parser("trials", help="List the trials of the dataset with category, force data and study split.")
    trials.add_argument("--split", choices=["optimization", "heldout", "unused"], help="Show only trials of this split.")
    trials.set_defaults(func=_cmd_trials)

    baseline = sub.add_parser("baseline", help="Evaluate a configuration profile on recorded trials (coverage and failure analysis).")
    baseline.add_argument("--group", help="Task group from config/task_groups.yaml.")
    baseline.add_argument("--all-groups", action="store_true", help="Evaluate every non-global task group.")
    baseline.add_argument("--global", dest="global_opt", action="store_true", help="Evaluate the optimization split (group all_tasks).")
    baseline.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(baseline)
    baseline.set_defaults(func=_cmd_baseline)

    open_cmd = sub.add_parser("open", help="Print an absolute artifact path.")
    open_cmd.add_argument("path")
    open_cmd.set_defaults(func=_cmd_open)

    gui = sub.add_parser("gui", help="Start the browser tool for configuring and launching evaluations and optimizations.")
    gui.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    gui.add_argument("--port", type=int, default=8765, help="HTTP bind port.")
    gui.set_defaults(func=_cmd_gui)

    mujoco_model = sub.add_parser("mujoco-model", help="Generate the MuJoCo model (MJCF) of GARMI with two Panda arms for a profile.")
    mujoco_model.add_argument("--output", help="Output XML path. Defaults to resources/mujoco/generated/.")
    mujoco_model.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(mujoco_model)
    mujoco_model.set_defaults(func=_cmd_mujoco_model)

    simulate = sub.add_parser("simulate", help="Run MuJoCo operational-space impedance simulation for one task.")
    simulate.add_argument("--task", required=True, help="Processed trial ID.")
    simulate.add_argument("--no-video", action="store_true", help="Skip MP4 rendering for faster metric-only simulation.")
    simulate.add_argument("--no-contacts", action="store_true", help="Disable MuJoCo contact geoms (no self-contact check).")
    simulate.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(simulate)
    simulate.set_defaults(func=_cmd_simulate)

    simulate_batch_cmd = sub.add_parser("simulate-batch", help="Run metric-only MuJoCo operational-space impedance validation for a task group.")
    simulate_batch_cmd.add_argument("--group", help="Task group from config/task_groups.yaml.")
    simulate_batch_cmd.add_argument("--global", dest="global_opt", action="store_true", help="Simulate the optimization split (group all_tasks).")
    simulate_batch_cmd.add_argument(
        "--exclude-task",
        action="append",
        help="Trial ID to remove from this run. Can be repeated or comma-separated.",
    )
    simulate_batch_cmd.add_argument("--workers", type=int, help="Override worker process count.")
    simulate_batch_cmd.add_argument("--video", action="store_true", help="Render per-task MP4s. Slow and not recommended for large batches.")
    simulate_batch_cmd.add_argument("--no-contacts", action="store_true", help="Disable MuJoCo contact geoms (no self-contact check).")
    simulate_batch_cmd.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(simulate_batch_cmd)
    simulate_batch_cmd.set_defaults(func=_cmd_simulate_batch)

    view_simulation = sub.add_parser("view-simulation", help="Open an interactive MuJoCo impedance simulation for one task.")
    view_simulation.add_argument("--task", required=True, help="Processed trial ID.")
    view_simulation.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(view_simulation)
    view_simulation.add_argument("--realtime-factor", type=float, default=1.0, help="Playback speed relative to real time.")
    view_simulation.add_argument("--camera", choices=["garmi_follow", "overview"], help="Interactive viewer camera.")
    view_simulation.set_defaults(func=_cmd_view_simulation)

    report_all = sub.add_parser("report-all", help="Write baseline reports plus MuJoCo simulation reports and task videos.")
    report_all.add_argument("--group", action="append", help="Task group from config/task_groups.yaml. Can be used multiple times.")
    report_all.add_argument("--all-groups", action="store_true", help="Evaluate every non-global task group.")
    report_all.add_argument("--global", dest="global_opt", action="store_true", help="Evaluate all configured tasks.")
    report_all.add_argument("--workers", type=int, help="Override worker process count for simulation batches.")
    report_all.add_argument("--no-video", action="store_true", help="Skip MP4 rendering and write metric/report files only.")
    report_all.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(report_all)
    report_all.set_defaults(func=_cmd_report_all)

    sensitivity = sub.add_parser(
        "sensitivity",
        help="Morris elementary-effects screening of the six mount parameters.",
    )
    sensitivity.add_argument("--group", help="Task group from config/task_groups.yaml.")
    sensitivity.add_argument("--global", dest="global_opt", action="store_true", help="Screen on the optimization split (group all_tasks).")
    sensitivity.add_argument("--exclude-task", action="append", help="Trial ID to exclude. Can be repeated or comma-separated.")
    sensitivity.add_argument("--trajectories", type=int, default=8, help="Number of Morris trajectories (evaluations = r*(d+1)).")
    sensitivity.add_argument("--levels", type=int, default=6, help="Morris grid levels.")
    sensitivity.add_argument("--seed", type=int, default=7)
    sensitivity.add_argument("--workers", type=int, help="Candidate-level worker process count.")
    sensitivity.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(sensitivity)
    sensitivity.set_defaults(func=_cmd_sensitivity)

    generality = sub.add_parser(
        "generality",
        help="Price-of-generality report: evaluate named configs (group optima, global optimum) on every group.",
    )
    generality.add_argument(
        "--config",
        action="append",
        required=True,
        help="label=profile_or_yaml_path. Repeat for each configuration (e.g. general=reports/optimization/general_height_bound_seed07/best_config.yaml).",
    )
    generality.add_argument("--group", action="append", help="Group to evaluate on. Defaults to all non-global groups.")
    generality.add_argument("--workers", type=int, help="Worker process count.")
    generality.set_defaults(func=_cmd_generality)

    optimize_mounts = sub.add_parser(
        "optimize-mounts",
        help="Optimize the mirrored GARMI shoulder-mount pose and the base-placement policy.",
    )
    optimize_mounts.add_argument("--group", help="Task group from config/task_groups.yaml.")
    optimize_mounts.add_argument("--global", dest="global_opt", action="store_true", help="Optimize on the optimization split (group all_tasks).")
    optimize_mounts.add_argument(
        "--exclude-task",
        action="append",
        help="Trial ID to remove from this run. Can be repeated or comma-separated.",
    )
    optimize_mounts.add_argument("--samples", type=int, default=24, help="Number of mount candidates including the strict GARMI baseline.")
    optimize_mounts.add_argument("--workers", type=int, help="Candidate-level worker process count.")
    optimize_mounts.add_argument("--seed", type=int, default=7, help="Random seed for candidate generation.")
    optimize_mounts.add_argument(
        "--search-strategy",
        choices=["adaptive", "hybrid", "lhs", "sobol", "random"],
        default="adaptive",
        help="Generation of the initial candidates. adaptive (default) samples Sobol points and refines around the best mounts.",
    )
    optimize_mounts.add_argument(
        "--optimizer",
        choices=["adaptive", "multifidelity", "surrogate", "genetic", "bayesian"],
        default="bayesian",
        help=(
            "Optimization scheduler. bayesian (default) runs GP/expected-improvement proposal rounds at quick "
            "fidelity inside the multi-fidelity scheme; multifidelity, surrogate and genetic are alternative "
            "schedulers; adaptive is a mount-only sampling scheme."
        ),
    )
    optimize_mounts.add_argument(
        "--robust-ensemble",
        type=int,
        help="Override robustness.ensemble: number of perturbed task-cloud variants per task at reduced/full fidelity (0 disables).",
    )
    optimize_mounts.add_argument("--budget-hours", type=float, help="Optional wall-time budget. Checked between fidelity stages.")
    optimize_mounts.add_argument("--max-evaluations", type=int, help="Maximum candidate-fidelity evaluations for multifidelity, surrogate and genetic runs.")
    optimize_mounts.add_argument("--optimize-controller", action="store_true", help="Include IK, collision-IK, impedance and base-PD variables in DesignCandidates.")
    optimize_mounts.add_argument("--resume", help="Existing run directory whose evaluation_cache should be reused.")
    genetic = optimize_mounts.add_argument_group("genetic scheduler (--optimizer genetic; alternative scheduler, not used in the paper)")
    genetic.add_argument("--ga-population", type=int, help="Population size for --optimizer genetic. Defaults to --samples or optimization.genetic.population.")
    genetic.add_argument("--ga-generations-quick", type=int, help="Quick-fidelity generations for --optimizer genetic.")
    genetic.add_argument("--ga-generations-reduced", type=int, help="Reduced-fidelity generations for --optimizer genetic.")
    genetic.add_argument("--ga-elite-fraction", type=float, help="Elite fraction preserved by the genetic optimizer.")
    genetic.add_argument("--ga-mutation-sigma", type=float, help="Gaussian mutation sigma in normalized design-variable space.")
    genetic.add_argument("--ga-crossover-rate", type=float, help="Arithmetic crossover probability for the genetic optimizer.")
    genetic.add_argument("--ga-no-local-seeds", action="store_true", help="Disable local initial mutations around the strict GARMI baseline.")
    genetic.add_argument("--ga-seed-run", help="Previous optimization run directory whose candidates should seed a local genetic search.")
    genetic.add_argument("--ga-seed-candidate", action="append", help="Candidate id from --ga-seed-run to import. Can be passed multiple times.")
    genetic.add_argument("--ga-seed-top", type=int, default=0, help="Import this many ranked candidates from --ga-seed-run when no --ga-seed-candidate is given.")
    optimize_mounts.add_argument(
        "--validate-top",
        type=int,
        default=3,
        help="Re-evaluate this many top search candidates plus strict GARMI with the full base grid. Use 0 to skip.",
    )
    optimize_mounts.add_argument(
        "--base-search",
        choices=["reduced", "single", "full"],
        default="reduced",
        help="Base variant search used during candidate optimization. best_config.yaml keeps the full base search.",
    )
    optimize_mounts.add_argument("--mujoco-validate-top", type=int, default=0, help="Optionally validate this many top mount candidates in MuJoCo impedance simulation.")
    optimize_mounts.add_argument("--mujoco-workers", type=int, help="Worker process count for optional MuJoCo top-candidate validation.")
    optimize_mounts.add_argument("--mujoco-video", action="store_true", help="Render videos during optional MuJoCo top-candidate validation.")
    optimize_mounts.add_argument("--mujoco-contacts", action="store_true", help="Enable MuJoCo contact geoms during optional top-candidate validation.")
    optimize_mounts.add_argument("--baseline-profile", default=DEFAULT_PROFILE, help="Config profile from config/<name>.yaml.")
    _add_base_motion_arg(optimize_mounts)
    optimize_mounts.set_defaults(func=_cmd_optimize_mounts)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
