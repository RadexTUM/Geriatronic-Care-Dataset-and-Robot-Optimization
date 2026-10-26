# GARMI Optimization

Python package `garmi-optimization` (command `garmi-opt`) that processes the
Geriatronics Interaction Dataset, evaluates and optimizes the shoulder-mount
configuration of the humanoid robot GARMI on the recorded tasks and simulates
them in MuJoCo. Installation, usage and the reproduction of the paper are
described in the [README of the repository](../README.md).

```text
config/                  evaluation profiles, task groups, task frames, preprocessing settings
data/processed/          processed trials (garmi-opt preprocess)
reports/                 results of the runs reported in the paper (see reports/README.md)
resources/mujoco/        Franka Emika Panda MuJoCo model (third party, Apache-2.0)
scripts/reproduce_paper.sh
scripts/analysis/        robustness variants (make_variants.py) and statistics (aggregate_results.py)
src/garmi_optimization/  package source
    config.py            project paths and profile loading
    dataset.py           dataset discovery and study split
    preprocessing/       raw recordings -> processed trials
    tasks.py             processed trials and task groups
    task_frames.py       task-frame orientation constraints
    panda.py             Panda kinematics, dynamics and inverse kinematics
    collision.py         primitive collision model of the GARMI body
    evaluation.py        kinematic coverage of a configuration
    optimization.py      multi-fidelity Bayesian optimization of mount pose and base policy
    perturbations.py     perturbed scene copies (robustness ensemble)
    generality.py        configurations x categories (price of generality)
    sensitivity.py       Morris screening
    mujoco_backend.py    MuJoCo model and impedance tracking simulation
    reporting.py         report suite (garmi-opt report-all)
    gui.py               browser tool
    cli.py               command-line interface
tests/                   pytest suite
outputs/                 created at run time (runs, logs, variants); not versioned
```

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
garmi-opt --help
```
