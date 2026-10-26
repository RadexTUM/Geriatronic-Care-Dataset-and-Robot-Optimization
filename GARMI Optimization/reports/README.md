# Reports of the paper

This folder contains the results of exactly those runs whose numbers appear in
the paper. Every subfolder is the output folder of one `garmi-opt` command
(see `scripts/reproduce_paper.sh` for the commands). Bulky intermediate files
such as evaluation caches, per-task HTML pages and videos were removed; all
tables that enter the paper are kept.

Open any `report.html` in a browser for an interactive view of a run. The
folder `robustness/summary/` is written by `scripts/analysis/aggregate_results.py`
and collects the statistics of Sec. V in a few CSV files plus `summary.md`.

## Where the numbers of the paper come from

| Paper | Content | Folder or file |
|---|---|---|
| Sec. V-B | Elementary-effects (Morris) screening, stationary and freely moving base | `sensitivity_screening/fixed_base`, `sensitivity_screening/holonomic_base` |
| Sec. V-B | Coverage thresholds varied by ±20 % (16 variants) | `robustness/threshold_variants`, `robustness/summary/threshold_sensitivity.csv` |
| Sec. V-B | Temporal discretization with 50/100/200 waypoints | `robustness/discretization`, `robustness/summary/discretization.csv` |
| Sec. V-B | Sobol search with the budget of the Bayesian optimization | `optimization/sobol_search_seed{07,11,23}`, `robustness/summary/sobol_vs_bayesian.csv` |
| Sec. V-C | Baseline coverage on the optimization split and per category | `baseline/optimization_split`, `baseline/<category>` |
| Sec. V-C, Fig. 3 | General configuration within the height bound, three seeds | `optimization/general_height_bound_seed{07,11,23}`, `robustness/summary/optimization_runs.csv` |
| Sec. V-C | Paired bootstrap of the coverage gain | `robustness/summary/bootstrap_ci.csv` |
| Sec. V-C | Held-out split, baseline and recommended configuration | `baseline/heldout`, `recommended/heldout` |
| Sec. V-C | Stage 1 (kinematics only) vs. stage 2 (measured wrenches), force-equipped subset | `optimization/general_free_height_stage1_seed07`, `optimization/general_free_height_stage2_seed07`, `robustness/summary/force_subset.csv` |
| Sec. V-D, Fig. 4 | Category specialists and price of generality | `optimization/specialist_<category>`, `generality/` |
| Sec. V-E | MuJoCo tracking simulation of the 55 optimization trials | `tracking_simulation/baseline`, `tracking_simulation/recommended`, `robustness/summary/contingency.csv`, `robustness/summary/kin_dyn_correlation.csv` |
| Sec. V-E | Primary failure reasons of the trials not covered by the recommended configuration | `optimization/general_height_bound_seed07/full_validation_task_metrics.csv` (column `primary_failure`) |

The recommended configuration (`config/recommended.yaml`) is the best
candidate of `optimization/general_height_bound_seed07`
(`best_config.yaml`, candidate `candidate_bo_r00_065`).

## Folder contents

* `baseline/`, `recommended/`: evaluation of a fixed configuration
  (`garmi-opt baseline`). `summary.csv` holds the coverage,
  `task_success_matrix.csv` the per-trial outcome and metrics,
  `failure_breakdown.csv`/`failure_taxonomy.csv` the failure reasons and
  `run_config.json` the complete configuration snapshot.
* `optimization/<run>/`: one search (`garmi-opt optimize-mounts`).
  `candidates.csv` ranks the candidates (first row = best),
  `evaluation_archive.csv` lists every candidate evaluation in order,
  `full_validation_task_metrics.csv` holds the per-trial results of the
  full-fidelity re-evaluations (including the baseline candidate and the
  perturbed scene copies `<trial>@p<k>`), `best_config.yaml` is the best
  configuration as a ready-to-use profile, `optimization_summary.json` stores
  the search settings and `guard_log.csv` the plausibility checks. Runs with
  MuJoCo validation additionally contain `mujoco_validation_candidates.csv`
  and `best_mujoco_candidate.json`.
* `generality/`: every configuration evaluated on every functional category
  (`coverage_matrix.csv`, `price_of_generality.csv`).
* `tracking_simulation/<configuration>/`: MuJoCo impedance tracking of the 55
  optimization trials (`batch_simulation_summary.csv`, one `summary.json` per
  trial in `trials/`).
* `robustness/threshold_variants/<configuration>__<threshold>__<down|up>/`
  and `robustness/discretization/<configuration>__waypoints_<n>/`: baseline
  evaluations under modified thresholds or waypoint counts.
* `sensitivity_screening/<mode>/`: Morris screening of the six mount
  parameters (`sensitivity.csv` with mu, mu* and sigma per metric,
  `morris_evaluations.csv` with all evaluated candidates).

## Labels

| Label | Meaning |
|---|---|
| `candidate_000_strict_garmi` | Original GARMI mounting; evaluated as a protected candidate in every search (the baseline) |
| `strict` (generality) | Baseline, profile `stage2_forces` |
| `global` (generality) | Best general configuration without height bound (`optimization/general_free_height_stage2_seed07`); not shown in Fig. 4 |
| `global_z5cm` (generality) | Recommended general configuration within the ±5 cm height bound (`optimization/general_height_bound_seed07`) |
| `<category>` (generality) | Specialist of that category (`optimization/specialist_<category>`) |
| `stage1_kinematic` | Profile without measured wrenches |
| `stage2_forces` | Profile with measured wrenches where force-torque recordings exist |
| `stage2_forces_height_bound` | `stage2_forces` with mount height z in [0.875, 0.975] m |
| `sobol_search` | `stage2_forces_height_bound` without Bayesian-optimization rounds |
| `baseline`, `recommended` (robustness, tracking simulation) | Original GARMI mounting and recommended configuration |

## Notes

* The runs were computed with an earlier internal version of this code that
  addressed the trials by their original recording names. All trial
  identifiers, profile names and paths in this folder were converted to the
  published names (the column `Trial name (old)` of
  `Trials_Description.xlsx` lists the original names); numerical values are
  unchanged. The inactive environment-proxy section (`environment.enabled:
  false`) was emptied in the configuration snapshots.
* The Morris screening in `sensitivity_screening/` was computed in an earlier
  stage of the study on a preliminary selection of 32 trials (profile
  `garmi_strict_baseline`, equivalent to the original mounting). Two of these
  recordings are not part of the published dataset; they appear only as
  `care_therapeut_patient__walkassist4` and
  `care_therapeut_patient__washup6` in the column `worst_task_by_objective`.
  `scripts/reproduce_paper.sh` repeats the screening on the 55 trials of the
  optimization split.
* The searches drew the perturbed scene copies with a seed that included
  Python's per-process salted string hash, so the perturbations of a trial
  differed between worker processes and runs. The current code uses a
  process-independent hash (`garmi_optimization/perturbations.py`), which
  makes the perturbations identical for all candidates (common random
  numbers) and every run reproducible. Re-running a search therefore gives
  results close to, but not identical with, the shipped runs. Evaluations
  of fixed configurations on the nominal scenes (baseline, held-out,
  generality, robustness variants, tracking simulation) are not affected.
* The two tracking simulations were run with different contact settings.
  `tracking_simulation/recommended` comes from `garmi-opt simulate-batch`
  with the profile setting `simulation.enable_contacts: true`, so a
  self-contact disqualifies a trial. `tracking_simulation/baseline` comes
  from the MuJoCo validation inside the search
  `optimization/general_free_height_stage2_seed07` (candidate
  `candidate_000_strict_garmi`), which runs without contact geoms unless
  `--mujoco-contacts` is given. `garmi-opt simulate-batch --global --no-contacts`
  reproduces this baseline run (53 of 55 trials identical; two trials differ
  slightly in the time scaling taken over from the kinematic stage, with the
  same outcome). `scripts/reproduce_paper.sh` simulates both configurations
  with contact geoms. With identical settings the baseline passes 8 of 55
  trials (mean tracking error 5.9 cm) with contact geoms, and the recommended
  configuration passes 26 of 55 (3.1 cm) without them.
* The trial-level bootstrap in `robustness/summary/bootstrap_ci.csv` is
  evaluated exactly (the limit of infinitely many resamples) instead of with
  10 000 random resamples, so it does not depend on a random seed.
