"""Rotation, homogeneous-transform and interpolation helpers.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rotation_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def quat_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(translation, dtype=float)
    return out


def translate(x: float, y: float, z: float) -> np.ndarray:
    return transform(np.eye(3), np.array([x, y, z], dtype=float))


def rot_x4(angle: float) -> np.ndarray:
    return transform(rotation_x(angle), np.zeros(3))


def rot_z4(angle: float) -> np.ndarray:
    return transform(rotation_z(angle), np.zeros(3))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = int(window)
    if window <= 1 or len(values) <= 2:
        return values
    window = min(window, len(values))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, [(pad_left, pad_right), (0, 0)], mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.vstack([np.convolve(padded[:, col], kernel, mode="valid") for col in range(values.shape[1])]).T


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def finite_gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, time, axis=0, edge_order=1)


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9 and np.isfinite(norm):
        return vector / norm
    if fallback is None:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize_vector(np.asarray(fallback, dtype=float), None)


def frames_from_primary_secondary(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    frames = np.empty((len(primary), 3, 3), dtype=float)
    previous_z = np.array([0.0, 0.0, 1.0], dtype=float)
    for idx, (primary_vec, secondary_vec) in enumerate(zip(primary, secondary)):
        x_axis = normalize_vector(primary_vec, frames[idx - 1, :, 0] if idx else np.array([1.0, 0.0, 0.0]))
        z_raw = secondary_vec - float(np.dot(secondary_vec, x_axis)) * x_axis
        z_axis = normalize_vector(z_raw, previous_z)
        y_axis = normalize_vector(np.cross(z_axis, x_axis), frames[idx - 1, :, 1] if idx else np.array([0.0, 1.0, 0.0]))
        z_axis = normalize_vector(np.cross(x_axis, y_axis), z_axis)
        frames[idx] = np.column_stack([x_axis, y_axis, z_axis])
        previous_z = z_axis
    return frames


def rotation_error_vector(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)
    return Rotation.from_matrix(target @ current.T).as_rotvec()


# --- Helpers for the preprocessing pipeline ---
def is_none_like(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"none", "null", ""}
    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def as_float_or_none(value: object) -> float | None:
    if is_none_like(value):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalize_rows(vectors: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    out = np.empty_like(vectors)
    previous = fallback if fallback is not None else np.array([1.0, 0.0, 0.0])
    for idx, vec in enumerate(vectors):
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9 and np.isfinite(norm):
            previous = vec / norm
        out[idx] = previous
    return out


def interp_columns(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    target = np.empty((len(target_t), values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        target[:, idx] = np.interp(target_t, source_t, values[:, idx])
    return target
