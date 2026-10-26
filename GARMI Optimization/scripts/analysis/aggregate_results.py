#!/usr/bin/env python3
"""Aggregate the statistical and robustness analyses of the paper.

The script reads a folder with the layout of ``reports/`` (the shipped paper
reports, or ``outputs/paper/`` written by the reproduction scripts) and
computes the numbers reported in Sec. V of the paper:

  optimization_runs.csv     best candidate of every optimization run (seeds table)
  convergence.csv           best-so-far full-fidelity coverage per evaluation
  bootstrap_ci.csv          paired bootstrap CIs of the coverage difference
                            recommended - baseline (trial level, exact; and
                            clustered by repeat group), optimization and
                            held-out split
  force_subset.csv          coverage on the force-equipped trials and per-category
                            availability of force data
  contingency.csv           kinematic vs dynamic outcome per configuration with
                            exact McNemar test
  kin_dyn_correlation.csv   Spearman correlation of the kinematic objective and
                            the simulated tracking error
  threshold_sensitivity.csv ranking under +-20 % threshold variation
  discretization.csv        coverage with 50/100/200 waypoints
  sobol_vs_bayesian.csv     best coverage of Sobol search and Bayesian optimization
  summary.md                human-readable summary of all of the above

Usage (from the "GARMI Optimization" folder):
    python3 scripts/analysis/aggregate_results.py                   # shipped reports
    python3 scripts/analysis/aggregate_results.py --reports outputs/paper
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ID = "candidate_000_strict_garmi"  # protected original GARMI mounting
VARIANT = re.compile(r"@p\d+$")
MAX_EXACT_COMPOSITIONS = 2_000_000  # larger enumerations fall back to Monte Carlo
N_BOOT = 10_000
BOOT_SEED = 20260715
CATEGORIES = ["basic_care", "meals", "transfer", "rom", "geriatric_rehabilitation"]

lines: list[str] = ["# Aggregated results", ""]


def note(text: str = "") -> None:
    print(text)
    lines.append(text)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trial_metadata(processed: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(processed.glob("*/metadata.json")):
        meta = read_json(path)
        rows.append(
            {
                "trial": meta["task_name"],
                "category": meta.get("category"),
                "repeat_group": meta.get("repeat_group") or meta["task_name"],
                "force_measurement": bool(meta.get("force_measurement")),
            }
        )
    if not rows:
        raise SystemExit(f"no processed trials found in {processed}")
    return pd.DataFrame(rows).set_index("trial")


def nominal(frame: pd.DataFrame, column: str = "task_name") -> pd.DataFrame:
    return frame[~frame[column].astype(str).str.contains(VARIANT)]


def best_candidate(run: Path) -> str:
    return str(pd.read_csv(run / "candidates.csv").iloc[0].candidate_id)


# ---------------------------------------------------------------------------
# Optimization runs and convergence


def optimization_runs(reports: Path, out: Path) -> pd.DataFrame:
    rows = []
    for run in sorted((reports / "optimization").glob("*")):
        if not (run / "candidates.csv").exists():
            continue
        summary = read_json(run / "optimization_summary.json")
        best = pd.read_csv(run / "candidates.csv").iloc[0]
        rows.append(
            {
                "run": run.name,
                "profile": summary.get("baseline_profile"),
                "group": summary.get("group_name"),
                "seed": summary.get("seed"),
                "samples": summary.get("samples"),
                "evaluations_used": summary.get("evaluations_used"),
                "best_candidate": best.candidate_id,
                "coverage_with_perturbations": float(best.success_fraction),
                "coverage_nominal": float(best.get("success_fraction_nominal", np.nan)),
                "x_m": float(best.left_x_m),
                "y_m": float(best.left_y_m),
                "z_m": float(best.left_z_m),
                "roll_rad": float(best.left_delta_roll_rad),
                "pitch_rad": float(best.left_delta_pitch_rad),
                "yaw_rad": float(best.left_delta_yaw_rad),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        note("## Optimization runs: none found")
        note()
        return frame
    frame.to_csv(out / "optimization_runs.csv", index=False)
    note(f"## Optimization runs ({len(frame)})")
    for _, r in frame.iterrows():
        note(
            f"- {r.run}: nominal coverage {r.coverage_nominal:.3f} (with perturbations {r.coverage_with_perturbations:.3f}), "
            f"x={r.x_m:+.3f} y={r.y_m:.3f} z={r.z_m:.3f} pitch={r.pitch_rad:+.3f}"
        )
    seeds = frame[frame.run.str.startswith("general_height_bound_seed")]
    if len(seeds) > 1:
        note(
            f"- height-bounded seeds: nominal coverage spread {seeds.coverage_nominal.max() - seeds.coverage_nominal.min():.3f}, "
            f"y range {seeds.y_m.max() - seeds.y_m.min():.3f} m, pitch {seeds.pitch_rad.max():+.2f} to {seeds.pitch_rad.min():+.2f} rad"
        )
    note()
    return frame


def convergence(reports: Path, out: Path) -> None:
    frames = []
    for run in sorted((reports / "optimization").glob("*")):
        archive_path = run / "evaluation_archive.csv"
        if not archive_path.exists():
            continue
        archive = pd.read_csv(archive_path).reset_index(drop=True)
        full = archive["success_fraction"].where(archive.get("fidelity") == "full")
        frames.append(
            pd.DataFrame(
                {
                    "run": run.name,
                    "evaluation": np.arange(1, len(archive) + 1),
                    "fidelity": archive.get("fidelity"),
                    "success_fraction": archive["success_fraction"],
                    "best_so_far_full_fidelity": full.cummax().ffill(),
                }
            )
        )
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "convergence.csv", index=False)


def sobol_vs_bayesian(runs: pd.DataFrame, out: Path) -> None:
    rows = []
    for _, r in runs.iterrows():
        if r.run.startswith("sobol_search_seed"):
            rows.append({"method": "sobol", "seed": r.seed, "best_coverage": r.coverage_with_perturbations, "evaluations": r.evaluations_used})
        elif r.run.startswith("general_height_bound_seed"):
            rows.append({"method": "bayesian", "seed": r.seed, "best_coverage": r.coverage_with_perturbations, "evaluations": r.evaluations_used})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "sobol_vs_bayesian.csv", index=False)
    note("## Sobol search vs Bayesian optimization (height-bounded design space, identical budget)")
    for method, sub in frame.groupby("method"):
        note(f"- {method}: best coverage {', '.join(f'{v:.3f}' for v in sub.best_coverage)} (mean {sub.best_coverage.mean():.3f})")
    note()


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals


def exact_paired_bootstrap(a: np.ndarray, b: np.ndarray, clusters: list[str] | None = None) -> dict | None:
    """Exact paired (optionally cluster) bootstrap of mean(b) - mean(a).

    The success flags are binary, so a resample of the m clusters is fully
    described by how many clusters of each type (sum of the paired
    differences, cluster size) it contains, which is multinomially
    distributed. Enumerating these counts yields the bootstrap distribution
    exactly: the limit of the Monte-Carlo bootstrap for infinitely many
    resamples, independent of random seeds and of the order of the trials.
    The 95 % interval uses the 2.5 % and 97.5 % quantiles of this
    distribution. Returns None if the enumeration would be too large.
    """
    d = np.asarray(b, dtype=int) - np.asarray(a, dtype=int)
    labels = list(clusters) if clusters is not None else list(range(len(d)))
    cluster_sum: dict = {}
    cluster_size: dict = {}
    for label, value in zip(labels, d):
        cluster_sum[label] = cluster_sum.get(label, 0) + int(value)
        cluster_size[label] = cluster_size.get(label, 0) + 1
    m = len(cluster_sum)
    kinds = list(Counter((cluster_sum[c], cluster_size[c]) for c in cluster_sum).items())
    if math.comb(m + len(kinds) - 1, len(kinds) - 1) > MAX_EXACT_COMPOSITIONS:
        return None
    log_p = [math.log(count / m) for _, count in kinds]
    log_fact = [math.lgamma(i + 1) for i in range(m + 1)]
    distribution: dict[Fraction, float] = {}

    def enumerate_counts(index: int, remaining: int, log_weight: float, total_d: int, total_size: int) -> None:
        (d_sum, size), _ = kinds[index]
        counts = [remaining] if index == len(kinds) - 1 else range(remaining + 1)
        for k in counts:
            weight = log_weight - log_fact[k] + k * log_p[index]
            if index == len(kinds) - 1:
                key = Fraction(total_d + k * d_sum, total_size + k * size)
                distribution[key] = distribution.get(key, 0.0) + math.exp(weight)
            else:
                enumerate_counts(index + 1, remaining - k, weight, total_d + k * d_sum, total_size + k * size)

    enumerate_counts(0, m, log_fact[m], 0, 0)
    values = sorted(distribution)
    probs = np.array([distribution[v] for v in values])
    cdf = np.cumsum(probs) / probs.sum()

    def quantile(alpha: float) -> float:
        return float(values[int(np.searchsorted(cdf, alpha - 1e-12))])

    return {
        "delta": float(d.mean()),
        "ci_low": quantile(0.025),
        "ci_high": quantile(0.975),
        "p_delta_le_0": float(sum(distribution[v] for v in values if v <= 0) / probs.sum()),
        "n_trials": len(d),
        "n_clusters": m,
        "method": "exact",
    }


def monte_carlo_bootstrap(a: np.ndarray, b: np.ndarray, clusters: list[str], rng: np.random.Generator) -> dict:
    """Monte-Carlo paired cluster bootstrap (fallback for large enumerations)."""
    labels = pd.Series(clusters)
    unique = labels.unique()
    members = {c: np.flatnonzero(labels.values == c) for c in unique}
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([members[c] for c in chosen])
        diffs[i] = b[idx].mean() - a[idx].mean()
    return {
        "delta": float(b.mean() - a.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_delta_le_0": float(np.mean(diffs <= 0.0)),
        "n_trials": len(a),
        "n_clusters": len(unique),
        "method": f"monte_carlo_{N_BOOT}",
    }


def bootstrap(reports: Path, meta: pd.DataFrame, out: Path) -> None:
    """Coverage difference recommended - baseline on both splits.

    The recommended configuration is the best candidate of the height-bounded
    search with seed 7 (config/recommended.yaml); its per-trial outcomes on
    the optimization split come from the full-fidelity validation of that run.
    """
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    run = reports / "optimization/general_height_bound_seed07"
    metrics = nominal(pd.read_csv(run / "full_validation_task_metrics.csv"))
    winner = best_candidate(run)
    paired = metrics.pivot_table(index="task_name", columns="candidate_id", values="success", aggfunc="first")
    paired = paired[[BASELINE_ID, winner]].dropna().astype(float).sort_index()
    splits = [("optimization", paired[BASELINE_ID], paired[winner])]

    base = pd.read_csv(reports / "baseline/heldout/task_success_matrix.csv").set_index("task")["success"].astype(float)
    rec = pd.read_csv(reports / "recommended/heldout/task_success_matrix.csv").set_index("task")["success"].astype(float)
    common = base.index.intersection(rec.index).sort_values()
    splits.append(("heldout", base.loc[common], rec.loc[common]))

    for split, a_series, b_series in splits:
        a, b = a_series.to_numpy(), b_series.to_numpy()
        common_fields = {"split": split, "baseline": a.mean(), "recommended": b.mean()}
        # Trial level as in the paper; repeat-group level resamples whole
        # repeat groups (recordings of the same scenario) as a check.
        for level, clusters in [("trial", None), ("repeat_group", [meta.repeat_group.get(t, t) for t in a_series.index])]:
            result = exact_paired_bootstrap(a, b, clusters)
            if result is None:
                result = monte_carlo_bootstrap(a, b, clusters or list(range(len(a))), rng)
            rows.append({**common_fields, "level": level, **result})

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "bootstrap_ci.csv", index=False)
    note("## Paired bootstrap, coverage difference recommended - baseline (exact bootstrap distribution)")
    for _, r in frame.iterrows():
        note(
            f"- {r.split} split, {r.level} level (n={r.n_trials}, clusters={r.n_clusters}): "
            f"{r.baseline:.3f} -> {r.recommended:.3f}, delta {r.delta:+.3f}, "
            f"95 % CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}], P(delta <= 0) = {r.p_delta_le_0:.4f}"
        )
    note()


# ---------------------------------------------------------------------------
# Force subset


def force_subset(reports: Path, meta: pd.DataFrame, out: Path) -> None:
    run = reports / "optimization/general_height_bound_seed07"
    metrics = nominal(pd.read_csv(run / "full_validation_task_metrics.csv"))
    winner = best_candidate(run)
    metrics = metrics[metrics.candidate_id.isin([BASELINE_ID, winner])].copy()
    metrics["force"] = metrics.task_name.map(meta.force_measurement)
    metrics["category"] = metrics.task_name.map(meta.category)
    rows = []
    for label, subset in [("all", metrics), ("force_equipped", metrics[metrics.force]), ("kinematics_only", metrics[~metrics.force])]:
        for candidate, sub in subset.groupby("candidate_id"):
            rows.append(
                {
                    "subset": label,
                    "configuration": "baseline" if candidate == BASELINE_ID else "recommended",
                    "n_trials": sub.task_name.nunique(),
                    "coverage": float(sub.success.mean()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "force_subset.csv", index=False)
    note("## Force-equipped subset (optimization split)")
    for (subset, n), sub in frame.groupby(["subset", "n_trials"], sort=False):
        values = dict(zip(sub.configuration, sub.coverage))
        note(f"- {subset} (n={n}): baseline {values.get('baseline', np.nan):.3f}, recommended {values.get('recommended', np.nan):.3f}")
    availability = metrics[metrics.candidate_id == winner].groupby("category")["force"].agg(["sum", "count"])
    for category in CATEGORIES:
        if category in availability.index:
            r = availability.loc[category]
            note(f"  - {category}: {int(r['sum'])}/{int(r['count'])} trials with force data")
    note()


# ---------------------------------------------------------------------------
# Kinematic vs dynamic stage


def contingency(reports: Path, out: Path) -> None:
    run = reports / "optimization/general_height_bound_seed07"
    metrics = nominal(pd.read_csv(run / "full_validation_task_metrics.csv"))
    candidates = {"baseline": BASELINE_ID, "recommended": best_candidate(run)}
    rows, corr_rows = [], []
    for label, candidate in candidates.items():
        batch = reports / "tracking_simulation" / label / "batch_simulation_summary.csv"
        if not batch.exists():
            continue
        sim = pd.read_csv(batch).set_index("task_name")
        kin_rows = metrics[metrics.candidate_id == candidate].set_index("task_name")
        common = kin_rows.index.intersection(sim.index)
        kin = kin_rows.loc[common, "success"].astype(bool)
        dyn = sim.loc[common, "simulation_passed"].astype(bool)
        both, kin_only, dyn_only, none = int((kin & dyn).sum()), int((kin & ~dyn).sum()), int((~kin & dyn).sum()), int((~kin & ~dyn).sum())
        discordant = kin_only + dyn_only
        p_value = float(stats.binomtest(min(kin_only, dyn_only), discordant, 0.5).pvalue) if discordant else 1.0
        rows.append(
            {
                "configuration": label,
                "n_trials": len(common),
                "kinematic_coverage": float(kin.mean()),
                "dynamic_pass_rate": float(dyn.mean()),
                "dynamic_passes": int(dyn.sum()),
                "mean_tracking_error_m": float(sim.loc[common, "mean_tracking_error_m"].mean()),
                "kin_pass_dyn_pass": both,
                "kin_pass_dyn_fail": kin_only,
                "kin_fail_dyn_pass": dyn_only,
                "kin_fail_dyn_fail": none,
                "mcnemar_exact_p": p_value,
            }
        )
        joint = pd.concat([kin_rows.loc[common, "objective"].astype(float), sim.loc[common, "mean_tracking_error_m"].astype(float)], axis=1).dropna()
        rho, p = stats.spearmanr(joint.iloc[:, 0], joint.iloc[:, 1])
        corr_rows.append({"configuration": label, "spearman_rho": float(rho), "p_value": float(p), "n_trials": len(joint)})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "contingency.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "kin_dyn_correlation.csv", index=False)
    note("## Kinematic vs dynamic (MuJoCo tracking simulation)")
    for _, r in frame.iterrows():
        note(
            f"- {r.configuration}: {r.dynamic_passes}/{r.n_trials} dynamic passes, mean tracking error {100 * r.mean_tracking_error_m:.1f} cm; "
            f"kinematic pass & dynamic pass {r.kin_pass_dyn_pass}, kinematic pass & dynamic fail {r.kin_pass_dyn_fail}, "
            f"kinematic fail & dynamic pass {r.kin_fail_dyn_pass}, both fail {r.kin_fail_dyn_fail}; McNemar p = {r.mcnemar_exact_p:.2g}"
        )
    for r in corr_rows:
        note(f"- Spearman(kinematic objective, mean tracking error), {r['configuration']}: rho = {r['spearman_rho']:.2f} (p = {r['p_value']:.2g}, n = {r['n_trials']})")
    note()


# ---------------------------------------------------------------------------
# Threshold and discretization variants


def variant_table(folder: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(folder.glob("*")):
        if not (run / "summary.csv").exists():
            continue
        config, _, variant = run.name.partition("__")
        summary = pd.read_csv(run / "summary.csv").iloc[0]
        rows.append({"configuration": config, "variant": variant, "coverage": float(summary.success_fraction), "n_trials": int(summary.task_count)})
    return pd.DataFrame(rows)


def threshold_sensitivity(reports: Path, out: Path) -> None:
    frame = variant_table(reports / "robustness/threshold_variants")
    if frame.empty:
        return
    wide = frame.pivot_table(index="variant", columns="configuration", values="coverage")
    wide["gap"] = wide["recommended"] - wide["baseline"]
    wide.to_csv(out / "threshold_sensitivity.csv")
    preserved = int((wide.gap > 0).sum())
    note("## Threshold sensitivity (+-20 %, individually and jointly)")
    note(f"- ranking preserved in {preserved}/{len(wide)} variants, gap {wide.gap.min():+.2f} to {wide.gap.max():+.2f}")
    note()


def discretization(reports: Path, out: Path) -> None:
    frame = variant_table(reports / "robustness/discretization")
    if frame.empty:
        return
    frame["waypoints"] = frame.variant.str.extract(r"waypoints_(\d+)").astype(int)
    nominal_rows = []
    for config, folder in [("baseline", reports / "baseline/optimization_split"), ("recommended", None)]:
        if folder is not None and (folder / "summary.csv").exists():
            nominal_rows.append({"configuration": config, "waypoints": 50, "coverage": float(pd.read_csv(folder / "summary.csv").iloc[0].success_fraction)})
    run = reports / "optimization/general_height_bound_seed07"
    metrics = nominal(pd.read_csv(run / "full_validation_task_metrics.csv"))
    nominal_rows.append({"configuration": "recommended", "waypoints": 50, "coverage": float(metrics[metrics.candidate_id == best_candidate(run)].success.mean())})
    table = pd.concat([pd.DataFrame(nominal_rows), frame[["configuration", "waypoints", "coverage"]]], ignore_index=True)
    wide = table.pivot_table(index="waypoints", columns="configuration", values="coverage").sort_index()
    wide.to_csv(out / "discretization.csv")
    note("## Temporal discretization (waypoints per trial)")
    for waypoints, r in wide.iterrows():
        note(f"- {waypoints} waypoints: baseline {r.get('baseline', np.nan):.3f}, recommended {r.get('recommended', np.nan):.3f}")
    note()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", default=str(PROJECT_ROOT / "reports"), help="Folder with the layout of reports/.")
    parser.add_argument("--processed", default=str(PROJECT_ROOT / "data/processed"), help="Processed-data folder (trial metadata).")
    parser.add_argument("--output", help="Output folder. Defaults to <reports>/robustness/summary.")
    args = parser.parse_args()
    reports = Path(args.reports).resolve()
    out = Path(args.output).resolve() if args.output else reports / "robustness/summary"
    out.mkdir(parents=True, exist_ok=True)
    meta = trial_metadata(Path(args.processed).resolve())

    def section(function, *section_args) -> object:
        # A partial reproduction may lack some runs; skip the affected analyses.
        try:
            return function(*section_args)
        except FileNotFoundError as exc:
            note(f"## {function.__name__}: skipped (missing {exc.filename})")
            note()
            return None

    runs = section(optimization_runs, reports, out)
    section(convergence, reports, out)
    if runs is not None and not runs.empty:
        section(sobol_vs_bayesian, runs, out)
    section(bootstrap, reports, meta, out)
    section(force_subset, reports, meta, out)
    section(contingency, reports, out)
    section(threshold_sensitivity, reports, out)
    section(discretization, reports, out)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
