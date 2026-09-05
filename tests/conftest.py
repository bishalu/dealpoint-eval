"""Shared fixtures for the dealpoint offline test suite.

Tests that need the extracted MAUD dataset under `data/raw/` use the
`dataset_available` fixture, which skips cleanly (not an error) when the
dataset is absent, per the M0 spec: the dataset "may be assumed present ...
for the offline suite only if a fixture skips cleanly when it is absent."
"""

from __future__ import annotations

import pytest

from dealpoint.config import CONTRACTS_DIR, CSV_PATHS, N_CONTRACTS


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch):
    """M4.1: `call_with_retries`'s backoff is real exponential backoff based
    at `dealpoint.config.API_RETRY_BASE_DELAY_S` (default ~1.0s) so a real
    retried sweep actually waits between attempts. The offline suite (incl.
    the pre-existing `gate_m1` API-error test) must stay sub-second, so every
    test gets a tiny base delay unless it opts out by monkeypatching this
    back up itself.
    """
    monkeypatch.setattr("dealpoint.agent._common.API_RETRY_BASE_DELAY_S", 0.001)


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
