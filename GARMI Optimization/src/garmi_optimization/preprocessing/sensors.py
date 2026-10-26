"""Force-torque sensor frames from the three-marker struts (Kabsch algorithm)
and rotation of the measured wrenches into the room frame.
"""
from __future__ import annotations

import numpy as np

from garmi_optimization.math_utils import interp_columns
from garmi_optimization.preprocessing.vicon import node_xyz, set_node_xyz


def kabsch_pose(local_points: np.ndarray, global_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid_local = np.mean(local_points, axis=0)
    centroid_global = np.mean(global_points, axis=0)
    centered_local = local_points - centroid_local
    centered_global = global_points - centroid_global
    h = centered_local.T @ centered_global
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = centroid_global - rotation @ centroid_local
    return rotation, translation


def reconstruct_sensor(
    node_data: dict,
    sensor_data: dict,
    sensor_name: str,
    local_marker_matrix_mm: np.ndarray,
) -> dict[str, np.ndarray]:
    local_points = np.asarray(local_marker_matrix_mm, dtype=float) / 1000.0
    node_time = np.asarray(node_data["real_time"], dtype=float)
    sensor_time = np.asarray(sensor_data["real_time"], dtype=float)

    marker_names = ["xplus", "xminus", "yminus"]
    marker_node_points = [node_xyz(node_data, f"{sensor_name}:{marker}") for marker in marker_names]

    rotations_node = []
    centers_node = []
    for frame in range(len(node_time)):
        global_points = np.vstack([marker_points[frame] for marker_points in marker_node_points])
        rotation, translation = kabsch_pose(local_points, global_points)
        rotations_node.append(rotation)
        centers_node.append(translation)
    centers_node_arr = np.asarray(centers_node)
    set_node_xyz(node_data, f"{sensor_name}:center", centers_node_arr, node_time)

    interpolated_markers = [
        interp_columns(node_time, marker_points, sensor_time) for marker_points in marker_node_points
    ]
    rotations_sensor = []
    for frame in range(len(sensor_time)):
        global_points = np.vstack([marker_points[frame] for marker_points in interpolated_markers])
        rotation, _translation = kabsch_pose(local_points, global_points)
        rotations_sensor.append(rotation)

    sensor_key = "S1" if sensor_name == "sensor1" else "S2"
    ft = sensor_data[sensor_key]
    force_local = np.column_stack([ft["Fx"], ft["Fy"], ft["Fz"]])
    torque_local = np.column_stack([ft["Tx"], ft["Ty"], ft["Tz"]])
    force_global = np.empty_like(force_local)
    torque_global = np.empty_like(torque_local)
    for idx, rotation in enumerate(rotations_sensor):
        force_global[idx] = rotation @ force_local[idx]
        torque_global[idx] = rotation @ torque_local[idx]

    return {
        "center_node_time": centers_node_arr,
        "force_global": force_global,
        "torque_global": torque_global,
        "force_local": force_local,
        "torque_local": torque_local,
        "force_resultant": np.linalg.norm(force_local, axis=1),
        "torque_resultant": np.linalg.norm(torque_local, axis=1),
        "rotations_node": np.asarray(rotations_node),
        "rotations_sensor": np.asarray(rotations_sensor),
    }
