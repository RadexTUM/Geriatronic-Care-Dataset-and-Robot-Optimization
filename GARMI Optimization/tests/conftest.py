"""Shared pytest configuration.

Tests marked ``requires_data`` need the processed trials in
``data/processed/<trial>/task_data.npz`` (shipped with the repository and
reproducible with ``garmi-opt preprocess``). Tests marked ``requires_dataset``
need the raw Geriatronics Interaction Dataset next to the "GARMI Optimization"
folder. Both are skipped automatically when the files are not available, e.g.
when the Git LFS objects have not been downloaded.
"""
from __future__ import annotations

import pytest

from garmi_optimization.config import PATHS
from garmi_optimization.dataset import DESCRIPTION_FILE, dataset_root


def _processed_data_available() -> bool:
    root = PATHS.processed_data
    return root.is_dir() and any(root.glob("*/task_data.npz"))


def _dataset_available() -> bool:
    description = dataset_root() / DESCRIPTION_FILE
    # A Git LFS pointer file is a small text file, not an Excel workbook.
    return description.is_file() and description.stat().st_size > 1024


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_data: needs the processed trials in data/processed")
    config.addinivalue_line("markers", "requires_dataset: needs the raw Geriatronics Interaction Dataset")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skips = []
    if not _processed_data_available():
        skips.append(("requires_data", "processed trials not available (run garmi-opt preprocess)"))
    if not _dataset_available():
        skips.append(("requires_dataset", "Geriatronics Interaction Dataset not available (git lfs pull)"))
    for marker, reason in skips:
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
