# Geriatronics Interaction Dataset and Task-Based Optimization of GARMI

This repository accompanies the paper

> S. Nertinger, P. Radecker, A. König, D. J. Rixen, and K. Bengler,
> "A Multimodal Dataset for Assistive Robotics in Elderly Care: A Case Study on
> Task-Based Kinematic Optimization of Humanoid Robots,"
> IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2026.

It contains

* the **Geriatronics Interaction Dataset**: motion capture and force-torque
  recordings of 86 trials of elderly-care and geriatric-rehabilitation
  activities performed by trained experts, and
* **GARMI Optimization**: the Python package `garmi-optimization` that
  processes the dataset, evaluates how well the two Franka Emika Panda arms of
  the humanoid robot GARMI can execute the recorded tasks, optimizes the
  mirrored shoulder-mount pose, simulates the tasks in MuJoCo, and the reports
  of all runs that produced the numbers of the paper.

| Configuration | Optimization trials (55) | Held-out trials (26) |
|---|---|---|
| Original GARMI mounting | 0.62 | 0.46 |
| Recommended general mounting (`config/recommended.yaml`) | 0.78 | 0.58 |

Kinematic task coverage, i.e. the fraction of trials for which all coverage
criteria of the paper (Table II) hold.

## Contents

1. [Repository layout](#repository-layout)
2. [Getting the repository](#getting-the-repository)
3. [The Geriatronics Interaction Dataset](#the-geriatronics-interaction-dataset)
4. [GARMI Optimization](#garmi-optimization)
5. [Reproducing the paper](#reproducing-the-paper)
6. [Citation](#citation)
7. [License](#license)

## Repository layout

```text
Geriatronics Interaction Dataset/      raw dataset (stored with Git LFS)
    Trials_Description.xlsx            one sheet per category: description, contact window,
                                       sensor placement and objects of every trial
    Basic Care/ Meals/ Transfer/ ROM/ Geriatric Rehabilitation/
        MoCap/<trial>.csv              Vicon Nexus export, 120 Hz
        FT-sensor/<trial>_ft.csv       force-torque recording, 240 Hz
    Objects/                           photos of the manipulated objects
    LICENSE                            CC BY 4.0
GARMI Optimization/                    code (Python package garmi-optimization)
    config/                            evaluation profiles, task groups, task frames
    data/processed/                    processed trials (input of all evaluations)
    reports/                           results of the runs reported in the paper
    resources/mujoco/                  Franka Emika Panda MuJoCo model (third party)
    scripts/reproduce_paper.sh         reproduction of all experiments
    scripts/analysis/                  robustness variants and statistics of Sec. V
    src/garmi_optimization/            package source
    tests/                             test suite
README.md  LICENSE  CITATION.cff
```

## Getting the repository

The raw dataset (about 1.9 GB) is stored with [Git LFS](https://git-lfs.com).
Install Git LFS once (`brew install git-lfs` on macOS, `sudo apt install git-lfs`
on Ubuntu) and clone:

```bash
git lfs install
git clone https://github.com/RadexTUM/Geriatronic-Care-Dataset-and-Robot-Optimization.git
```

Downloads without Git LFS, e.g. ZIP archives of the repository, may contain
small pointer files instead of the raw data files.

The processed trials in `GARMI Optimization/data/processed` are regular Git
files, so all evaluations and optimizations work without the raw data. To
skip the download of the raw data, or to fetch only parts of it:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RadexTUM/Geriatronic-Care-Dataset-and-Robot-Optimization.git
cd Geriatronic-Care-Dataset-and-Robot-Optimization
git lfs pull --include "Geriatronics Interaction Dataset/Trials_Description.xlsx"
git lfs pull --include "Geriatronics Interaction Dataset/Meals/**"
git lfs pull                       # everything
```

## The Geriatronics Interaction Dataset

### Activities

The activities were selected from about 100 hours of non-participatory
observations in inpatient long-term care and recreated in a laboratory
furnished like a care facility (paper Sec. III, Table I). Activities that
involve only objects were performed by a researcher who took part in the
observations; activities with human-human interaction were performed by a
vocationally trained nurse or a physical therapist together with a healthy
participant acting as care recipient.

| Category | Folder | Trials | With force data | Examples |
|---|---|---|---|---|
| Basic care | `Basic Care` | 18 | 10 | combing hair, putting on a jacket, washing, changing an underpad |
| Moderated meals | `Meals` | 20 | 9 | slicing bread, pouring tea, carrying a tray, wiping the table |
| Transfer | `Transfer` | 9 | 3 | walking assistance, standing up, turning to the side, moving a chair |
| Functional movement assessment (ROM) | `ROM` | 20 | 9 | range-of-motion and maximal-strength assessment (Janda) of shoulder and elbow |
| Geriatric rehabilitation | `Geriatric Rehabilitation` | 19 | 7 | guided exercises (wiping table, opening window, combing hair, tucking shirt), passive joint mobilization |

Activities that consist of several movements (joint mobilization and the
functional movement assessments) are split into one trial per movement.

### Recording setup

* **Motion capture**: 13-camera Vicon system at 120 Hz. 20 markers on the
  practitioner (torso, arms and hands), 28 markers on the care recipient
  (full body), plus all manipulated objects and relevant furniture.
* **Force-torque sensing**: up to two 6-axis force-torque sensors (Schunk
  FTE-Nano-25 SI-250-6) at 240 Hz in a two-part 3D-printed housing. The base
  is strapped to the care recipient's upper arm, forearm, thigh or lower leg,
  or attached to an object; the upper part with the sensor is held by the
  practitioner and couples magnetically to the base during contact. Three
  orthogonal marker struts on the housing (subject `sensor1`/`sensor2`,
  markers `xplus`, `xminus`, `yminus`) give the sensor orientation.
* All participants gave written informed consent. The study was approved by
  the ethics committee of the Technical University of Munich (reference
  2023-90-125040-S).
* Participants appear only under the role names `therapeut` and `patient`.
  Recording paths of the acquisition software, document properties and photo
  metadata (location, camera, capture time) were removed, and people visible
  in the object photos were blacked out.

### Files

**`Trials_Description.xlsx`** has one sheet per category and one row per
trial with the columns

| Column | Content |
|---|---|
| `Trial name (old)` | name of the original recording session |
| `Trial name` | trial name used for all files of the trial |
| `Trial with force measurement` | `x` if force-torque data exist |
| `Description` | short description of the activity |
| `Timeframe Begin Contact`, `Timeframe End Contact` | MoCap frames of the task-relevant contact phase |
| `Position Sensor 1`, `Position Sensor 2` | placement of the two force-torque sensors |
| `Included Object 1-3`, `Weight of object 1-3 [g]`, `Marker names of object 1-3` | manipulated objects (with in-cell photos) |

**`<Category>/MoCap/<trial>.csv`** is a Vicon Nexus CSV export with the
sections `Joints`, `Segments` and `Trajectories`. Each section starts with its
name, the sample rate (120), a row with `<subject>:<name>` labels, a row with
the components (`RX RY RZ TX TY TZ` for joints and segments, `X Y Z` for
marker trajectories, followed by their first and second derivatives) and a
unit row (`mm`, `deg`, `mm/s`, ...), followed by one row per frame. Marker
trajectories and segment poses are given in the room frame of the
motion-capture system (z up); joints describe the relative pose of two
segments. Subjects are `therapeut` (the practitioner), `patient` (the care
recipient, markers `PAT_*`), `sensor1` and `sensor2`, and the objects and
furniture of the scene. Some object
subjects have German names: `Bett` bed, `brot` bread, `brotscheibe` slice of
bread, `bürste` brush, `kelle` ladle, `löffel` spoon, `messer` knife, `mob`
mop, `schüssel` bowl, `semmel` bun, `stuhl` chair, `tablett` tray, `tasse`
mug, `thermo` thermos flask, `vorhang` curtain.

**`<Category>/FT-sensor/<trial>_ft.csv`** starts with comment lines of the
acquisition software (`# trial_name=<original recording name>`,
`# start_iso=<start time>`, `# rate_hz=240`), followed by the columns

```text
sample, t_rel_s, wall_iso, mono_ns,
s1_Fx, s1_Fy, s1_Fz, s1_Tx, s1_Ty, s1_Tz, s1_missed,
s2_Fx, s2_Fy, s2_Fz, s2_Tx, s2_Ty, s2_Tz, s2_missed
```

with forces in N and torques in N m in the frame of the respective sensor,
the time `t_rel_s` in seconds since the start of the recording, the wall-clock
and monotonic time stamps of the acquisition computer and a flag for missed
samples. Motion capture and force-torque recording cover the same interval;
the preprocessing maps the MoCap frames linearly onto the force-torque time
base.

**`Objects/`** contains photos of the manipulated objects; files ending in
`_ft` show the object with the mounted sensor housing and the labels of its
markers.

### Trial names

Trial names have the form `<activity>[_<variant>]_<kin|force>[_2]`. Variants
are `AAN` (assist-as-needed: the support is adapted to the physical condition
of the target care recipient), `passive` (the participant stays as passive as
possible) and `janda` (maximal-strength assessment after Janda); `mobi_*`
denotes passive joint mobilization and `_2` a second recording. The suffixes
`_kin` and `_force` distinguish the recordings of one scenario: `_force`
recordings were made with the force-torque sensors. The column
`Trial with force measurement` and the presence of the `_ft.csv` file are
authoritative; four trials deviate from the suffix rule
(`washing_lower_body_kin` has force data, `putting_on_jacket_force`,
`folding_newspaper_force` and `refilling_tea_side_force` have none).

### Use in the paper

81 of the 86 trials entered the study (paper Sec. IV-B): 55 trials form the
optimization split, and 26 further recordings of scenarios contained in the
optimization split form the held-out split, which is never used in a search.
The splits, the functional categories and the
groups of repeat recordings are defined in
`GARMI Optimization/config/task_groups.yaml`; `garmi-opt trials` lists every
trial with its split. The trials `setting_table_kin`, `setting_table_force`,
`reaching_overhead_exercise_kin`, `elbow_extension_janda_kin` and
`shoulder_internal_rotation_janda_force` were not part of the study.

## GARMI Optimization

### What the code does

For every trial, the wrist poses of the practitioner's active hands are the
end-effector targets of GARMI's arms, and the measured wrenches act as
external end-effector loads. A configuration (mirrored shoulder-mount pose
plus a per-task base-placement policy) covers a trial if damped-least-squares
inverse kinematics along the trajectory keeps the position and task-relevant
orientation errors, manipulability, joint-speed and torque utilization,
time scaling and collision clearance within the thresholds of the profile.
A multi-fidelity Bayesian optimization searches the design space; the
original GARMI mounting is evaluated as a protected candidate in every run.
A MuJoCo operational-space impedance simulation checks the dynamic
trackability of the results.

### Installation

Python 3.10 or newer on macOS or Linux. The package reads `config/`, `data/`
and `resources/` from the `GARMI Optimization` folder, so install it in
editable mode:

```bash
cd "GARMI Optimization"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q          # tests that need the raw dataset are skipped without it
garmi-opt --help
```

The runs of the paper were computed with Python 3.12 on macOS (Apple
silicon); the evaluations were also verified on Linux with Python 3.10. For
headless video rendering on Linux set `MUJOCO_GL=egl`; on macOS start the
interactive viewer (`garmi-opt view-simulation`) with `mjpython`.

### Quick start

All commands run from the `GARMI Optimization` folder; every run writes a
folder `outputs/runs/<run>/` with a `report.html` and CSV/JSON tables.

```bash
# list the trials with category, force data and study split
garmi-opt trials

# coverage of the original GARMI mounting on the 55 optimization trials
garmi-opt baseline --global --baseline-profile stage2_forces

# coverage of the recommended mounting on the held-out trials
garmi-opt baseline --group heldout_all --baseline-profile config/recommended.yaml

# optimize the mounting for one category (the paper used 48-96 samples, 8 workers)
garmi-opt optimize-mounts --group transfer --samples 48 --seed 7 --workers 8 \
    --baseline-profile stage2_forces

# MuJoCo impedance tracking of one trial (writes an MP4 unless --no-video)
garmi-opt simulate --task slicing_bread_force --baseline-profile config/recommended.yaml

# browser tool for configuring and launching evaluations and optimizations
garmi-opt gui                # then open http://127.0.0.1:8765
```

Every `best_config.yaml` written by a search can be passed back as
`--baseline-profile`. The command `baseline` evaluates any profile; its name
refers to the original mounting it was first used for.

### Commands

| Command | Purpose |
|---|---|
| `preprocess` | Raw dataset → `data/processed` (`--points` waypoints per trial, `--output`, `--trial`) |
| `trials` | List the trials of the dataset with category, force data and split |
| `baseline` | Evaluate a profile on a task group (`--group`, `--all-groups`, `--global`) |
| `optimize-mounts` | Search the mount pose and base policy (`--group` or `--global`, `--samples`, `--seed`, `--workers`, `--mujoco-validate-top`) |
| `generality` | Evaluate several configurations on several groups (price of generality, Fig. 4) |
| `sensitivity` | Morris elementary-effects screening of the six mount parameters |
| `simulate`, `simulate-batch` | MuJoCo impedance tracking of one trial or a group (`--no-contacts` disables contact geoms) |
| `view-simulation` | Interactive MuJoCo viewer for one trial |
| `mujoco-model` | Write the MJCF model of GARMI for a profile |
| `report-all` | Baseline reports plus MuJoCo reports and videos |
| `gui` | Browser tool for evaluations, optimizations, simulations and task groups |
| `open` | Print the absolute path of an artifact |

`garmi-opt <command> --help` lists all options. The base-motion model can be
overridden with `--base-motion-mode` (`fixed_world`, `fixed_initial`,
`limited_holonomic`, `nonholonomic`, `free_holonomic`).

### Configuration

| File in `config/` | Content |
|---|---|
| `stage1_kinematic.yaml` | Original GARMI mounting, measured wrenches not applied (stage 1) |
| `stage2_forces.yaml` | Original GARMI mounting with measured wrenches (stage 2, baseline of the paper, default profile) |
| `stage2_forces_height_bound.yaml` | Stage 2 with the mount height bounded to ±5 cm around the original shoulder |
| `sobol_search.yaml` | Height-bounded design space, Sobol sampling without Bayesian optimization |
| `recommended.yaml` | Recommended general configuration of the paper (Fig. 3) |
| `task_groups.yaml` | Functional categories, optimization and held-out split, repeat groups |
| `task_frames.yaml` | Per-trial orientation constraints and base-motion annotations |
| `project.yaml` | Dataset location and preprocessing parameters |

A profile contains the coverage thresholds (`success`), the design space and
optimizer settings (`optimization`), the Panda arms and mounts (`panda`),
the base policy (`base`), IK, collision, controller and simulation settings.
The browser tool can create additional task groups.

Environment variables: `GARMI_OPT_DATASET_DIR` (location of the raw dataset,
default: the dataset folder next to `GARMI Optimization`),
`GARMI_OPT_PROCESSED_DIR` (alternative processed-data folder) and
`GARMI_OPT_ROOT` (location of the `GARMI Optimization` folder if the package
is not installed in editable mode).

### Processed trials

`garmi-opt preprocess` aligns motion capture and force-torque data in time,
fills gaps, applies a 15 Hz fourth-order zero-phase Butterworth filter,
reconstructs the sensor frames from the marker struts (Kabsch algorithm),
rotates the wrenches into the room frame, assigns each sensor to the hand
named in `Position Sensor 1/2` (otherwise to the closer wrist), cuts the
annotated contact window and resamples it to 50 waypoints. The wrist point
of each hand lies on the extension of the line from the `WRISTUP` marker
through the midpoint of `WRISTMED` and `WRISTLAT`. Every trial folder in
`data/processed/<trial>/` contains

* `task_data.npz`: `time` (s from the start of the contact window) and
  `time_abs` (s from the start of the recording); for `left` and `right`:
  `*_pos` (wrist point, m, room frame), `*_orient_primary` (unit vector from
  the wrist point to the midpoint of the index- and ring-finger markers),
  `*_orient_secondary` (unit vector from the wrist point to the `WRISTUP`
  marker), `*_force` (N) and `*_torque` (N m) in the room frame (zero for
  trials without force data),
* `metadata.json`: category, description, split, repeat group, source files,
  contact window, sensor placement and assignment, force summary, workspace
  bounding box,
* `preview.html`: a short overview.

`manifest.json` lists all trials of the dataset and `preprocessing_status.json`
the result of the last preprocessing run. The shipped folder was produced with
`garmi-opt preprocess` from the dataset in this repository.

## Reproducing the paper

`GARMI Optimization/scripts/reproduce_paper.sh` runs all experiments of the
paper in order and registers every run, as a link to `outputs/runs/<run>`, in
`outputs/paper/` with the layout of the shipped `reports/` folder, so that the
results can be compared file by file:

```bash
cd "GARMI Optimization"
scripts/reproduce_paper.sh                 # evaluation steps, a few hours with 8 workers
scripts/reproduce_paper.sh --search        # additionally all searches, about 15 h more
scripts/reproduce_paper.sh --list          # steps; e.g. scripts/reproduce_paper.sh heldout tracking
```

| Step | Paper | Content |
|---|---|---|
| `preprocess` | Sec. IV-B | Dataset → `data/processed` |
| `baselines` | Sec. V-C | Original mounting on the optimization split, per category and on the held-out split |
| `search_free_height` | Sec. V-C | General search without height bound, stage 1 and stage 2 (force ablation) |
| `search_specialists` | Sec. V-D | One search per category over the full height range |
| `search_height_bound` | Sec. V-C, Fig. 3 | General search within the ±5 cm height bound, seeds 7, 11, 23 |
| `search_sobol` | Sec. V-B | Sobol search with the same budget, seeds 7, 11, 23 |
| `generality` | Sec. V-D, Fig. 4 | Every configuration on every category |
| `heldout` | Sec. V-C | Recommended configuration on the held-out split |
| `tracking` | Sec. V-E | MuJoCo tracking of the 55 optimization trials, baseline and recommended |
| `thresholds` | Sec. V-B | Coverage thresholds varied by ±20 % |
| `discretization` | Sec. V-B | 100 and 200 waypoints per trial |
| `screening` | Sec. V-B | Morris screening with stationary and holonomic base |
| `aggregate` | Sec. V | Bootstrap intervals, McNemar test, correlations, robustness tables |

Without `--search` the evaluation steps use the configurations found by the
searches of the paper (`reports/optimization/*/best_config.yaml`,
`config/recommended.yaml`). The script restarts itself under `caffeinate` on
macOS so that the computer does not sleep; `WORKERS=10` or `--workers 10`
changes the number of worker processes.

The shipped results are documented in
[`GARMI Optimization/reports/README.md`](GARMI%20Optimization/reports/README.md),
including a table that maps every number of the paper to its source file.
Points to be aware of when comparing new runs with the shipped ones:

* Evaluations of a fixed configuration are deterministic and reproduce the
  shipped results. New searches reproduce them closely but not exactly: the
  searches of the paper drew the perturbed scene copies with Python's
  per-process salted string hash, which the code now replaces with a
  process-independent hash (common random numbers across all candidates).
* The shipped tracking simulation of the baseline was run without contact
  geoms, the one of the recommended configuration with contact geoms; the
  script simulates both with contact geoms (details in the reports README).
* The Morris screening in `reports/sensitivity_screening` was computed on a
  preliminary selection of 32 trials; the script repeats it on the 55 trials
  of the optimization split.

## Citation

If you use the dataset or the code, please cite the paper (see also
[`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{nertinger2026multimodal,
  author    = {Nertinger, Simone and Radecker, Philipp and K{\"o}nig, Alexander and Rixen, Daniel J. and Bengler, Klaus},
  title     = {A Multimodal Dataset for Assistive Robotics in Elderly Care: A Case Study on
               Task-Based Kinematic Optimization of Humanoid Robots},
  booktitle = {IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year      = {2026}
}
```

## License

* Code in `GARMI Optimization` (except `resources/mujoco/franka`): MIT License,
  see [`LICENSE`](LICENSE).
* Geriatronics Interaction Dataset: Creative Commons Attribution 4.0
  International (CC BY 4.0), see
  [`Geriatronics Interaction Dataset/LICENSE`](Geriatronics%20Interaction%20Dataset/LICENSE).
* Franka Emika Panda MuJoCo model in `GARMI Optimization/resources/mujoco/franka`:
  Apache License 2.0, see
  [`GARMI Optimization/THIRD_PARTY_NOTICES.md`](GARMI%20Optimization/THIRD_PARTY_NOTICES.md).

Contact: Philipp Radecker (philipp.radecker@tum.de), Simone Nertinger
(simone.nertinger@tum.de), Technical University of Munich.
