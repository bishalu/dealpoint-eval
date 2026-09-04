"""Shared fixtures for the dealpoint offline test suite.

Tests that need the extracted MAUD dataset under `data/raw/` use the
`dataset_available` fixture, which skips cleanly (not an error) when the
dataset is absent, per the M0 spec: the dataset "may be assumed present ...
for the offline suite only if a fixture skips cleanly when it is absent."
"""

from __future__ import annotations

import pytest

from dealpoint.config import CONTRACTS_DIR, CSV_PATHS, N_CONTRACTS


def _dataset_present() -> bool:
    if not CONTRACTS_DIR.is_dir():
        return False
    if len(list(CONTRACTS_DIR.glob("contract_*.txt"))) != N_CONTRACTS:
        return False
    return all(path.exists() for path in CSV_PATHS.values())


@pytest.fixture(scope="session")
def dataset_available() -> bool:
    return _dataset_present()


@pytest.fixture(autouse=False)
def require_dataset(dataset_available: bool) -> None:
    if not dataset_available:
        pytest.skip("MAUD dataset not present under data/raw/ (offline fixture skip)")
