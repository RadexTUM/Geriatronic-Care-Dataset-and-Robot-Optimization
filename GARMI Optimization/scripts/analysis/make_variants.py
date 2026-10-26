#!/usr/bin/env python3
"""Write the configuration variants of the robustness analyses (paper Sec. V-B).

For the baseline (config/stage2_forces.yaml) and the recommended configuration
(config/recommended.yaml, or --recommended) the script writes to
outputs/variants/:

  <config>__<threshold>__<down|up>.yaml  one coverage threshold scaled by 0.8 / 1.2
  <config>__all__<down|up>.yaml          all seven thresholds scaled together
  <config>__waypoints_<n>.yaml           unchanged copy, evaluated on the
                                         n-waypoint data (discretization analysis)
  manifest.json                          list of all variants

with <config> in {baseline, recommended}. Evaluating both configurations with
every threshold variant shows whether their ranking depends on the chosen
thresholds.

Usage (from the "GARMI Optimization" folder):
    python3 scripts/analysis/make_variants.py [--recommended path/to/best_config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Binary coverage thresholds of the "success" section (Table II).
THRESHOLD_KEYS = [
    "mean_position_error_m",
    "max_position_error_m",
    "mean_orientation_error_rad",
    "max_orientation_error_rad",
    "min_manipulability",
    "max_time_scale",
    "max_ik_discontinuity_fraction",
]
FACTORS = {"down": 0.8, "up": 1.2}
WAYPOINTS = (100, 200)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def write_variant(source: Path, config: str, variant: str, description: str, out_dir: Path, mutate=None) -> dict:
    cfg = load_yaml(source)
    if mutate is not None:
        mutate(cfg["success"])
    cfg.setdefault("baseline", {})["name"] = f"{config}__{variant}"
    cfg["baseline"]["description"] = description
    path = out_dir / f"{config}__{variant}.yaml"
    dump_yaml(path, cfg)
    return {"config": config, "variant": variant, "profile": relative(path)}


def make_variants(source: Path, config: str, out_dir: Path) -> list[dict]:
    if "success" not in load_yaml(source):
        raise SystemExit(f"{source} has no success section")
    entries = []

    def threshold_description(variant: str) -> str:
        return f"Threshold variant '{variant}' of the {config} configuration ({source.name})."

    for key in THRESHOLD_KEYS:
        for direction, factor in FACTORS.items():
            def scale_one(success: dict, key: str = key, factor: float = factor) -> None:
                success[key] = round(float(success[key]) * factor, 6)

            variant = f"{key}__{direction}"
            entries.append(write_variant(source, config, variant, threshold_description(variant), out_dir, scale_one))
    for direction, factor in FACTORS.items():
        def scale_all(success: dict, factor: float = factor) -> None:
            for key in THRESHOLD_KEYS:
                success[key] = round(float(success[key]) * factor, 6)

        variant = f"all__{direction}"
        entries.append(write_variant(source, config, variant, threshold_description(variant), out_dir, scale_all))
    for points in WAYPOINTS:
        description = f"Discretization variant: {points} waypoints, {config} configuration."
        entries.append(write_variant(source, config, f"waypoints_{points}", description, out_dir))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default=str(PROJECT_ROOT / "config/stage2_forces.yaml"), help="Baseline profile.")
    parser.add_argument("--recommended", default=str(PROJECT_ROOT / "config/recommended.yaml"), help="Recommended configuration.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/variants"), help="Output folder.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    entries = make_variants(Path(args.baseline), "baseline", out_dir)
    entries += make_variants(Path(args.recommended), "recommended", out_dir)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"variants": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(entries)} variants written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
