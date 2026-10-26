# Franka Emika Panda MuJoCo model (third party)

The files in this folder are an unmodified copy of
`franka_description/mujoco/franka/` from
[multipanda_ros2](https://github.com/tenfoldpaper/multipanda_ros2)
(commit `6448a6e6dd503ec5f07b48945d0590c3031683de`), licensed under the
Apache License 2.0 (see `LICENSE` in this folder). The Franka Emika Panda
description is © Franka Emika GmbH; the meshes appear to originate from an
earlier version of the `franka_emika_panda` model of
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(also Apache-2.0), which is derived from Franka Emika's `franka_ros`
description.

`garmi_optimization` reads `mj_dual.xml` as the reference model for the Panda
kinematics, inertias and joint limits (`panda.reference_mjcf` in the profiles)
and generates the GARMI dual-arm model with primitive torso geometry from it.
The remaining scene files are kept for completeness.
