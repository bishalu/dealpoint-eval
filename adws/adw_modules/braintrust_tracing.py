"""Braintrust: the run's phase tree mirrored to a hosted trace.

The local trace (events.jsonl + sssf.db) stays the source of truth. This adds a
second view of the same spine — one root span per run, one child per phase, and
one llm span per send inside an agent phase — so a run is readable off-machine
without shipping its sqlite file.

Nesting rides braintrust's contextvar: the root span is entered for the life of
the Run, so `braintrust.start_span` anywhere beneath it lands in the right
place by construction. The API key is read from .env.braintrust by the SDK.
"""

from __future__ import annotations

import braintrust

_logger: braintrust.Logger | None = None


def logger() -> braintrust.Logger:
    """Initialize once per ADW process, on the first span."""
    global _logger
    if _logger is None:
        braintrust.auto_instrument()
        _logger = braintrust.init_logger(project="sssf-dealpoint")
    return _logger


def flush() -> None:
    """Ship pending spans. An ADW is a short-lived process, so it must not exit
    on the async flush's timing."""
    if _logger is not None:
        _logger.flush()
