#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the experiments of the paper (Sec. IV and V).
#
# Usage, from the "GARMI Optimization" folder with the package installed
# (pip install -e ".[all]"):
#
#   scripts/reproduce_paper.sh                   all evaluation steps; the
#                                                configurations come from the
#                                                searches in reports/optimization
#   scripts/reproduce_paper.sh --search          additionally repeat all searches
#   scripts/reproduce_paper.sh heldout tracking  selected steps only
#   scripts/reproduce_paper.sh --list            list the steps
#
# Options:
#   --search      include the search steps (about 15 h with 8 workers)
#   --workers N   worker processes (default: $WORKERS or 8)
#   --out DIR     result folder, relative to "GARMI Optimization" (default: outputs/paper)
#
# Every step registers its run folders (outputs/runs/<run>) as symbolic links
# in the result folder, using the layout of reports/, for example
#   outputs/paper/baseline/optimization_split -> ../../runs/baseline_...
# so that every result can be compared with the shipped reports file by file.
# The step "aggregate" finally writes the statistics of Sec. V to
# <out>/robustness/summary (see scripts/analysis/aggregate_results.py).
#
# Without --search the evaluation steps use the configurations found by the
# searches of the paper (reports/optimization/<run>/best_config.yaml and
# config/recommended.yaml). With --search they use the new search results.
# New searches give results close to, but not identical with, the shipped
# ones (see the notes in reports/README.md).
#
# The script is compatible with the Bash 3.2 of macOS. On macOS it restarts
# itself under "caffeinate -i" so that the computer does not go to sleep.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT/scripts/$(basename "$0")"
cd "$ROOT"

if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  export CAFFEINATED=1
  exec caffeinate -i "$SCRIPT" "$@"
fi

WORKERS="${WORKERS:-8}"
OUT="outputs/paper"
SEARCH=0
STEPS=""
RECOMMENDED_RUN="general_height_bound_seed07"   # source of config/recommended.yaml
CATEGORIES="basic_care meals transfer rom geriatric_rehabilitation"
SEEDS="7 11 23"

ALL_STEPS="preprocess baselines search_free_height search_specialists search_height_bound search_sobol \
generality heldout tracking thresholds discretization screening aggregate"
SEARCH_STEPS="search_free_height search_specialists search_height_bound search_sobol"

usage() { sed -n '3,34p' "$SCRIPT" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --search) SEARCH=1 ;;
    --workers) WORKERS="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --list) echo "$ALL_STEPS" | tr ' ' '\n' | sed '/^$/d'; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) STEPS="$STEPS $1" ;;
  esac
  shift
done

if command -v garmi-opt >/dev/null 2>&1; then
  OPT="garmi-opt"
else
  OPT="python3 -m garmi_optimization.cli"
fi

mkdir -p "$OUT" outputs/logs
OUT="$(cd "$OUT" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/outputs/logs/paper_$STAMP"
mkdir -p "$LOG_DIR"

# --- helpers ----------------------------------------------------------------

info() { printf '\033[1;36m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

relpath() { python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"; }

# run <log name> <command ...>: run a garmi-opt command, keep a log and store
# the run folder it reports in LAST_RUN.
LAST_RUN=""
run() {
  local name="$1"; shift
  local log="$LOG_DIR/$name.log"
  info "$name: garmi-opt $*"
  # shellcheck disable=SC2086
  $OPT "$@" 2>&1 | tee "$log"
  LAST_RUN="$( (grep -E '^[a-z_]+_run: ' "$log" || true) | tail -1 | cut -d' ' -f2-)"
}

# register <target in the reports layout> <run folder>
register() {
  local target="$OUT/$1" run_dir="$2"
  [ -n "$run_dir" ] && [ -d "$run_dir" ] || die "no run folder for $1"
  mkdir -p "$(dirname "$target")"
  rm -rf "$target"
  ln -s "$(relpath "$run_dir" "$(dirname "$target")")" "$target"
  info "registered $1"
}

# Folder with the search runs: a real folder after --search, otherwise a link
# to the shipped searches.
use_new_searches() {
  if [ -L "$OUT/optimization" ]; then rm "$OUT/optimization"; fi
  mkdir -p "$OUT/optimization"
}
ensure_searches() {
  if [ ! -e "$OUT/optimization" ]; then
    ln -s "$(relpath "$ROOT/reports/optimization" "$OUT")" "$OUT/optimization"
    info "using the searches of the paper (reports/optimization)"
  fi
}

# best_config <search run>: configuration found by a search (path relative to
# the "GARMI Optimization" folder)
best_config() {
  ensure_searches
  local path="$OUT/optimization/$1/best_config.yaml"
  if [ -L "$OUT/optimization" ] || [ ! -f "$path" ]; then
    path="$ROOT/reports/optimization/$1/best_config.yaml"
    [ -L "$OUT/optimization" ] || info "search $1 was not repeated, using reports/optimization/$1"
  else
    path="$(cd "$(dirname "$path")" && pwd -P)/best_config.yaml"
  fi
  relpath "$path" "$ROOT"
}

# Recommended configuration: config/recommended.yaml unless the search was repeated.
recommended_config() {
  ensure_searches
  if [ ! -L "$OUT/optimization" ] && [ -f "$OUT/optimization/$RECOMMENDED_RUN/best_config.yaml" ]; then
    relpath "$(cd "$OUT/optimization/$RECOMMENDED_RUN" && pwd -P)/best_config.yaml" "$ROOT"
  else
    echo "config/recommended.yaml"
  fi
}

# --- steps --------------------------------------------------------------------

# Dataset -> data/processed (86 trials, 50 waypoints each).
step_preprocess() {
  run preprocess preprocess
}

# Original GARMI mounting with measured wrenches (profile stage2_forces).
step_baselines() {
  run baseline_optimization_split baseline --global --baseline-profile stage2_forces
  register baseline/optimization_split "$LAST_RUN"
  for category in $CATEGORIES; do
    run "baseline_$category" baseline --group "$category" --baseline-profile stage2_forces
    register "baseline/$category" "$LAST_RUN"
  done
  run baseline_heldout baseline --group heldout_all --baseline-profile stage2_forces
  register baseline/heldout "$LAST_RUN"
}

# General configuration without height bound, stage 1 (kinematics only) and
# stage 2 (measured wrenches); the top candidates are also simulated in MuJoCo.
step_search_free_height() {
  use_new_searches
  run general_free_height_stage1 optimize-mounts --global --samples 64 --seed 7 \
    --mujoco-validate-top 3 --workers "$WORKERS" --baseline-profile stage1_kinematic
  register optimization/general_free_height_stage1_seed07 "$LAST_RUN"
  run general_free_height_stage2 optimize-mounts --global --samples 64 --seed 7 \
    --mujoco-validate-top 3 --workers "$WORKERS" --baseline-profile stage2_forces
  register optimization/general_free_height_stage2_seed07 "$LAST_RUN"
}

# Category specialists over the full height range; transfer and ROM with a
# doubled budget.
step_search_specialists() {
  use_new_searches
  local samples
  for category in $CATEGORIES; do
    samples=48
    if [ "$category" = "transfer" ] || [ "$category" = "rom" ]; then samples=96; fi
    run "specialist_$category" optimize-mounts --group "$category" --samples "$samples" --seed 7 \
      --workers "$WORKERS" --baseline-profile stage2_forces
    register "optimization/specialist_$category" "$LAST_RUN"
  done
}

# General configuration within the +-5 cm height bound, three seeds.
step_search_height_bound() {
  use_new_searches
  for seed in $SEEDS; do
    run "general_height_bound_seed$seed" optimize-mounts --global --samples 64 --seed "$seed" \
      --workers "$WORKERS" --baseline-profile stage2_forces_height_bound
    register "optimization/general_height_bound_seed$(printf '%02d' "$seed")" "$LAST_RUN"
  done
}

# Quasi-random Sobol search with the budget of the Bayesian optimization
# (64 initial samples + 4 x 12 proposals = 112 samples).
step_search_sobol() {
  use_new_searches
  for seed in $SEEDS; do
    run "sobol_search_seed$seed" optimize-mounts --global --samples 112 --seed "$seed" \
      --workers "$WORKERS" --baseline-profile sobol_search
    register "optimization/sobol_search_seed$(printf '%02d' "$seed")" "$LAST_RUN"
  done
}

# Every configuration on every category (Fig. 4).
step_generality() {
  local category
  set -- --config "strict=stage2_forces" \
    --config "global=$(best_config general_free_height_stage2_seed07)" \
    --config "global_z5cm=$(best_config general_height_bound_seed07)"
  for category in $CATEGORIES; do
    set -- "$@" --config "$category=$(best_config "specialist_$category")" --group "$category"
  done
  run generality generality "$@" --workers "$WORKERS"
  register generality "$LAST_RUN"
}

# Recommended configuration on the 26 held-out trials.
step_heldout() {
  run recommended_heldout baseline --group heldout_all --baseline-profile "$(recommended_config)"
  register recommended/heldout "$LAST_RUN"
}

# MuJoCo impedance tracking of the 55 optimization trials. Both
# configurations are simulated with contact geoms, so that self-contact
# disqualifies a trial (see reports/README.md for the baseline run of the paper).
step_tracking() {
  run tracking_baseline simulate-batch --global --workers "$WORKERS" --baseline-profile stage2_forces
  register tracking_simulation/baseline "$LAST_RUN"
  run tracking_recommended simulate-batch --global --workers "$WORKERS" --baseline-profile "$(recommended_config)"
  register tracking_simulation/recommended "$LAST_RUN"
}

make_variants() {
  python3 scripts/analysis/make_variants.py --recommended "$(recommended_config)" --output outputs/variants
}

# Seven coverage thresholds varied by +-20 %, individually and jointly.
step_thresholds() {
  make_variants
  local config variant name
  for config in baseline recommended; do
    for variant in mean_position_error_m max_position_error_m mean_orientation_error_rad \
                   max_orientation_error_rad min_manipulability max_time_scale \
                   max_ik_discontinuity_fraction all; do
      for direction in down up; do
        name="${config}__${variant}__${direction}"
        run "threshold_$name" baseline --global --baseline-profile "outputs/variants/$name.yaml"
        register "robustness/threshold_variants/$name" "$LAST_RUN"
      done
    done
  done
}

# Trials re-processed with 100 and 200 waypoints.
step_discretization() {
  make_variants
  local points config name
  for points in 100 200; do
    run "preprocess_$points" preprocess --points "$points" --output "outputs/processed_$points"
    export GARMI_OPT_PROCESSED_DIR="outputs/processed_$points"
    for config in baseline recommended; do
      name="${config}__waypoints_$points"
      run "discretization_$name" baseline --global --baseline-profile "outputs/variants/$name.yaml"
      register "robustness/discretization/$name" "$LAST_RUN"
    done
    unset GARMI_OPT_PROCESSED_DIR
  done
}

# Elementary-effects screening of the six mount parameters with a stationary
# and with a freely moving holonomic base.
step_screening() {
  run screening_fixed_base sensitivity --global --trajectories 8 --levels 6 --seed 7 \
    --workers "$WORKERS" --baseline-profile stage2_forces --base-motion-mode fixed_initial
  register sensitivity_screening/fixed_base "$LAST_RUN"
  run screening_holonomic_base sensitivity --global --trajectories 8 --levels 6 --seed 7 \
    --workers "$WORKERS" --baseline-profile stage2_forces --base-motion-mode free_holonomic
  register sensitivity_screening/holonomic_base "$LAST_RUN"
}

# Statistics of Sec. V (bootstrap, contingency, correlations, robustness tables).
step_aggregate() {
  ensure_searches
  local run_dir name
  if [ ! -L "$OUT/optimization" ]; then
    for run_dir in "$ROOT"/reports/optimization/*; do
      name="$(basename "$run_dir")"
      if [ ! -e "$OUT/optimization/$name" ]; then
        info "search $name was not repeated, linking reports/optimization/$name"
        ln -s "$(relpath "$run_dir" "$OUT/optimization")" "$OUT/optimization/$name"
      fi
    done
  fi
  info "aggregate: scripts/analysis/aggregate_results.py --reports $OUT"
  python3 scripts/analysis/aggregate_results.py --reports "$OUT" 2>&1 | tee "$LOG_DIR/aggregate.log"
}

# --- main ---------------------------------------------------------------------

if [ -z "$STEPS" ]; then
  for step in $ALL_STEPS; do
    case " $SEARCH_STEPS " in
      *" $step "*) if [ "$SEARCH" = "1" ]; then STEPS="$STEPS $step"; fi ;;
      *) STEPS="$STEPS $step" ;;
    esac
  done
fi

for step in $STEPS; do
  case " $ALL_STEPS " in
    *" $step "*) ;;
    *) die "unknown step '$step' (see --list)" ;;
  esac
done

info "result folder: $OUT"
info "logs:          $LOG_DIR"
info "steps:        $STEPS"
for step in $STEPS; do
  info "=== step $step ==="
  "step_$step"
done
info "done. Compare $OUT with reports/ (summary: $OUT/robustness/summary/summary.md)"
