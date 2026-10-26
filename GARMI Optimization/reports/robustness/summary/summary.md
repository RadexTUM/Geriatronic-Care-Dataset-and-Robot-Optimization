# Aggregated results

## Optimization runs (13)
- general_free_height_stage1_seed07: nominal coverage 0.764 (with perturbations 0.773), x=-0.040 y=0.352 z=1.111 pitch=-0.082
- general_free_height_stage2_seed07: nominal coverage 0.764 (with perturbations 0.782), x=-0.040 y=0.352 z=1.111 pitch=-0.082
- general_height_bound_seed07: nominal coverage 0.782 (with perturbations 0.782), x=-0.055 y=0.313 z=0.966 pitch=-0.296
- general_height_bound_seed11: nominal coverage 0.727 (with perturbations 0.732), x=-0.120 y=0.299 z=0.959 pitch=-0.242
- general_height_bound_seed23: nominal coverage 0.782 (with perturbations 0.786), x=-0.027 y=0.225 z=0.929 pitch=-0.474
- sobol_search_seed07: nominal coverage 0.709 (with perturbations 0.709), x=-0.034 y=0.259 z=0.971 pitch=-0.322
- sobol_search_seed11: nominal coverage 0.745 (with perturbations 0.745), x=-0.051 y=0.204 z=0.926 pitch=-0.016
- sobol_search_seed23: nominal coverage 0.691 (with perturbations 0.691), x=-0.070 y=0.247 z=0.952 pitch=-0.589
- specialist_basic_care: nominal coverage 0.545 (with perturbations 0.545), x=-0.040 y=0.352 z=1.111 pitch=-0.082
- specialist_geriatric_rehabilitation: nominal coverage 0.812 (with perturbations 0.844), x=-0.027 y=0.357 z=1.112 pitch=-0.168
- specialist_meals: nominal coverage 0.800 (with perturbations 0.725), x=-0.029 y=0.266 z=1.088 pitch=-0.238
- specialist_rom: nominal coverage 0.917 (with perturbations 0.917), x=-0.098 y=0.432 z=1.108 pitch=-0.249
- specialist_transfer: nominal coverage 1.000 (with perturbations 0.958), x=-0.058 y=0.223 z=0.911 pitch=-0.373
- height-bounded seeds: nominal coverage spread 0.055, y range 0.087 m, pitch -0.24 to -0.47 rad

## Sobol search vs Bayesian optimization (height-bounded design space, identical budget)
- bayesian: best coverage 0.782, 0.732, 0.786 (mean 0.767)
- sobol: best coverage 0.709, 0.745, 0.691 (mean 0.715)

## Paired bootstrap, coverage difference recommended - baseline (exact bootstrap distribution)
- optimization split, trial level (n=55, clusters=55): 0.618 -> 0.782, delta +0.164, 95 % CI [+0.055, +0.291], P(delta <= 0) = 0.0049
- optimization split, repeat_group level (n=55, clusters=55): 0.618 -> 0.782, delta +0.164, 95 % CI [+0.055, +0.291], P(delta <= 0) = 0.0049
- heldout split, trial level (n=26, clusters=26): 0.462 -> 0.577, delta +0.115, 95 % CI [+0.000, +0.231], P(delta <= 0) = 0.0413
- heldout split, repeat_group level (n=26, clusters=25): 0.462 -> 0.577, delta +0.115, 95 % CI [+0.000, +0.240], P(delta <= 0) = 0.0409

## Force-equipped subset (optimization split)
- all (n=55): baseline 0.618, recommended 0.782
- force_equipped (n=34): baseline 0.618, recommended 0.824
- kinematics_only (n=21): baseline 0.619, recommended 0.714
  - basic_care: 9/11 trials with force data
  - meals: 8/10 trials with force data
  - transfer: 2/6 trials with force data
  - rom: 8/12 trials with force data
  - geriatric_rehabilitation: 7/16 trials with force data

## Kinematic vs dynamic (MuJoCo tracking simulation)
- baseline: 19/55 dynamic passes, mean tracking error 4.3 cm; kinematic pass & dynamic pass 18, kinematic pass & dynamic fail 16, kinematic fail & dynamic pass 1, both fail 20; McNemar p = 0.00027
- recommended: 22/55 dynamic passes, mean tracking error 3.8 cm; kinematic pass & dynamic pass 22, kinematic pass & dynamic fail 21, kinematic fail & dynamic pass 0, both fail 12; McNemar p = 9.5e-07
- Spearman(kinematic objective, mean tracking error), baseline: rho = 0.83 (p = 9.4e-15, n = 55)
- Spearman(kinematic objective, mean tracking error), recommended: rho = 0.79 (p = 7.2e-13, n = 55)

## Threshold sensitivity (+-20 %, individually and jointly)
- ranking preserved in 16/16 variants, gap +0.13 to +0.20

## Temporal discretization (waypoints per trial)
- 50 waypoints: baseline 0.618, recommended 0.782
- 100 waypoints: baseline 0.527, recommended 0.655
- 200 waypoints: baseline 0.400, recommended 0.564

