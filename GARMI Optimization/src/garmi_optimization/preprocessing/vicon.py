"""Reading of the marker trajectories in the Vicon Nexus CSV exports and of the
force-torque CSV files, gap filling and zero-phase low-pass filtering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


THERAPIST_NODES = [
    "CLAV",
    "STRN",
    "C7",
    "T10",
    "LSHO",
    "LELB",
    "LELB_LAT",
    "L_WRISTMED",
    "L_WRISTLAT",
    "L_WRISTUP",
    "L_ZEIG",
    "L_RING",
    "RSHO",
    "RELB",
    "RELB_LAT",
    "R_WRISTMED",
    "R_WRISTLAT",
    "R_WRISTUP",
    "R_ZEIG",
    "R_RING",
]

SENSOR_ALIASES = {
    "xplus": "xplus",
    "x_plus": "xplus",
    "xminus": "xminus",
    "x_minus": "xminus",
    "yminus": "yminus",
    "y_minus": "yminus",
    "ymimnus": "yminus",
}


def normalize_key(desc: Any) -> str:
    desc = str(desc).strip()
    mapping = {
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "X'": "Vx",
        "Y'": "Vy",
        "Z'": "Vz",
        "X''": "Ax",
        "Y''": "Ay",
        "Z''": "Az",
    }
    return mapping.get(desc, desc)


def normalize_key_ft(sensor: Any) -> str:
    text = str(sensor).strip().lower()
    if "fx" in text:
        return "Fx"
    if "fy" in text:
        return "Fy"
    if "fz" in text:
        return "Fz"
    if "tx" in text:
        return "Tx"
    if "ty" in text:
        return "Ty"
    if "tz" in text:
        return "Tz"
    if "missed" in text:
        return "Missed"
    return str(sensor).strip()


def read_vicon_rate(path: Path, default: int = 120) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        try:
            next(handle)
            return int(float(next(handle).strip()))
        except (StopIteration, ValueError):
            return default


def read_ft_rate(path: Path, default: int = 240) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if "rate_hz" in line:
                try:
                    return int(float(line.split("=", maxsplit=1)[1].strip()))
                except (IndexError, ValueError):
                    return default
    return default


def _find_segment_start(path: Path, needle: str) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if needle.lower() in line.lower():
                return idx
    return 0


def read_vicon_csv(path: Path, default_rate: int = 120) -> tuple[int, dict[str, Any]]:
    fps = read_vicon_rate(path, default_rate)
    segment_start = _find_segment_start(path, "trajectories")
    df = pd.read_csv(path, header=None, skiprows=segment_start + 2, engine="python")

    node_row = df.iloc[0].tolist()
    desc_row = df.iloc[1].tolist()
    unit_row = df.iloc[2].tolist()
    data = df.iloc[3:].reset_index(drop=True)

    filled_nodes: list[str | None] = []
    last_node: str | None = None
    for node in node_row:
        if pd.isna(node):
            filled_nodes.append(last_node)
        else:
            last_node = str(node).strip()
            filled_nodes.append(last_node)

    frames = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    valid_rows = ~np.isnan(frames)
    data = data.loc[valid_rows].reset_index(drop=True)
    frames = frames[valid_rows].astype(int)
    subframes = pd.to_numeric(data.iloc[:, 1], errors="coerce").fillna(0).to_numpy(dtype=float)

    node_data: dict[str, Any] = {"General Data": {"Frame": frames, "Subframe": subframes}}
    for col_idx, (node, desc, _unit) in enumerate(zip(filled_nodes, desc_row, unit_row)):
        if col_idx < 2 or node is None or pd.isna(desc):
            continue
        key = normalize_key(desc)
        if key not in {"X", "Y", "Z", "Vx", "Vy", "Vz", "Ax", "Ay", "Az"}:
            continue
        clean_node = str(node).strip()
        if clean_node.startswith("|") and clean_node.endswith("|"):
            clean_node = clean_node[1:-1].strip()
        node_data.setdefault(clean_node, {})
        node_data[clean_node][key] = pd.to_numeric(data.iloc[:, col_idx], errors="coerce").to_numpy(dtype=float)
    return fps, node_data


def read_ft_csv(path: Path, default_rate: int = 240) -> tuple[int, dict[str, Any]]:
    fps = read_ft_rate(path, default_rate)
    segment_start = _find_segment_start(path, "sample")
    df = pd.read_csv(path, header=None, skiprows=segment_start, engine="python")
    sensor_names = df.iloc[0].tolist()
    data = df.iloc[1:].reset_index(drop=True)

    sample = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(sample)
    data = data.loc[valid].reset_index(drop=True)
    sample = sample[valid].astype(int)

    t_rel = pd.to_numeric(data.iloc[:, 1], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sensor_data: dict[str, Any] = {
        "General Data": {
            "Sample": sample,
            "Relative_time": t_rel,
            "Time": data.iloc[:, 2].to_numpy(),
            "Monotonic_timestamp": data.iloc[:, 3].to_numpy(),
        }
    }
    for col_idx, sensor_name in enumerate(sensor_names):
        if col_idx < 4 or pd.isna(sensor_name):
            continue
        text = str(sensor_name).strip().lower()
        if "s1" in text:
            sensor = "S1"
        elif "s2" in text:
            sensor = "S2"
        else:
            continue
        key = normalize_key_ft(sensor_name)
        sensor_data.setdefault(sensor, {})
        sensor_data[sensor][key] = pd.to_numeric(data.iloc[:, col_idx], errors="coerce").to_numpy(dtype=float)
    return fps, sensor_data


def fill_missing_data(data: dict[str, Any]) -> dict[str, Any]:
    def fill(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=float).copy()
        missing = np.isnan(array)
        if not np.any(missing):
            return array
        if np.all(missing):
            return array
        x = np.arange(len(array))
        array[missing] = np.interp(x[missing], x[~missing], array[~missing])
        return array

    output: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"real_time", "General Data"}:
            output[key] = value
            continue
        output[key] = {inner: fill(np.asarray(arr, dtype=float)) for inner, arr in value.items()}
    return output


def smooth_data(data: dict[str, Any], sample_rate_hz: float, cutoff_hz: float, order: int) -> dict[str, Any]:
    if cutoff_hz <= 0 or cutoff_hz >= 0.5 * sample_rate_hz:
        return data
    b, a = butter(order, cutoff_hz / (0.5 * sample_rate_hz), btype="low", analog=False)

    def smooth(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=float)
        if len(array) <= 3 * max(len(a), len(b)) or np.all(np.isnan(array)):
            return array
        return filtfilt(b, a, array)

    output: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"real_time", "General Data"}:
            output[key] = value
            continue
        output[key] = {inner: smooth(np.asarray(arr, dtype=float)) for inner, arr in value.items()}
    return output


def convert_positions_mm_to_m(node_data: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in node_data.items():
        if key in {"real_time", "General Data"}:
            output[key] = value
            continue
        output[key] = {}
        for inner, arr in value.items():
            arr_np = np.asarray(arr, dtype=float)
            if inner in {"X", "Y", "Z", "Vx", "Vy", "Vz", "Ax", "Ay", "Az"}:
                output[key][inner] = arr_np / 1000.0
            else:
                output[key][inner] = arr_np
    return output


def clean_therapist_nodes(node_data: dict[str, Any], role_name: str = "therapeut") -> dict[str, Any]:
    cleaned = dict(node_data)
    for node in THERAPIST_NODES:
        target = f"therapeut:{node}"
        if target in cleaned:
            continue
        node_lower = node.lower()
        for key in list(cleaned.keys()):
            key_lower = key.lower()
            if role_name.lower() in key_lower and node_lower in key_lower:
                cleaned[target] = cleaned.pop(key)
                break
    missing = [f"therapeut:{node}" for node in THERAPIST_NODES if f"therapeut:{node}" not in cleaned]
    if missing:
        raise ValueError(f"Missing therapist nodes: {', '.join(missing)}")
    return cleaned


def clean_sensor_nodes(node_data: dict[str, Any], sensor_name: str) -> dict[str, Any]:
    cleaned = dict(node_data)
    for key in list(cleaned.keys()):
        key_lower = key.lower()
        if sensor_name.lower() not in key_lower:
            continue
        for alias, canonical in SENSOR_ALIASES.items():
            if alias in key_lower:
                cleaned[f"{sensor_name}:{canonical}"] = cleaned.pop(key)
                break
    missing = [f"{sensor_name}:{name}" for name in ["xplus", "xminus", "yminus"] if f"{sensor_name}:{name}" not in cleaned]
    if missing:
        raise ValueError(f"Missing {sensor_name} nodes: {', '.join(missing)}")
    return cleaned


def node_xyz(node_data: dict[str, Any], node_name: str) -> np.ndarray:
    node = node_data[node_name]
    return np.column_stack([node["X"], node["Y"], node["Z"]]).astype(float)


def set_node_xyz(node_data: dict[str, Any], node_name: str, xyz: np.ndarray, time: np.ndarray) -> None:
    node_data[node_name] = {
        "X": xyz[:, 0],
        "Y": xyz[:, 1],
        "Z": xyz[:, 2],
    }
    if len(time) > 1:
        velocity = np.gradient(xyz, time, axis=0, edge_order=1)
        acceleration = np.gradient(velocity, time, axis=0, edge_order=1)
        node_data[node_name].update(
            {
                "Vx": velocity[:, 0],
                "Vy": velocity[:, 1],
                "Vz": velocity[:, 2],
                "Ax": acceleration[:, 0],
                "Ay": acceleration[:, 1],
                "Az": acceleration[:, 2],
            }
        )


def derive_therapist_helper_nodes(node_data: dict[str, Any]) -> None:
    time = np.asarray(node_data["real_time"], dtype=float)
    for side in ["L", "R"]:
        wrist_med = node_xyz(node_data, f"therapeut:{side}_WRISTMED")
        wrist_lat = node_xyz(node_data, f"therapeut:{side}_WRISTLAT")
        wrist_up = node_xyz(node_data, f"therapeut:{side}_WRISTUP")
        ring = node_xyz(node_data, f"therapeut:{side}_RING")
        zeig = node_xyz(node_data, f"therapeut:{side}_ZEIG")
        overwrist = 0.5 * (wrist_med + wrist_lat)
        wrist = wrist_up + 1.5 * (overwrist - wrist_up)
        hand = 0.5 * (ring + zeig)
        set_node_xyz(node_data, f"therapeut:{side}_OVERWRIST", overwrist, time)
        set_node_xyz(node_data, f"therapeut:{side}_WRIST", wrist, time)
        set_node_xyz(node_data, f"therapeut:{side}_HAND", hand, time)
