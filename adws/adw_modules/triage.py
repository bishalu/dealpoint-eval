"""Failure triage: was it the infrastructure, or was it the work?

A milestone can die for two very different reasons, and the right response
differs: a provider that killed a turn after five minutes, a context window
that overflowed, a permission breach, an empty response — none of those are
defects a builder can fix, so spending a corrective cycle on them only burns
the budget meant for real defects. Red checks, a reviewer's REVISE, a builder
that declared failure — those are the work, and the corrective cycle is for
them.

Classification is code over evidence: the exception text the harness raised,
and the coding agent's own raw stream (every turn pi records carries a
`stopReason` and, on failure, an `errorMessage`). The most frequent provider
error message is the diagnosis; the class decides the remedy.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .data_types import Triage

PATTERNS: list[tuple[str, str, str]] = [
    (r"Unknown provider|not logged in|not authenticated|authentication|invalid api key|"
     r"credentials|please run .*login|OAuth|token expired",
     "infra:provider_auth",
     "the coding-agent provider is not registered or not authenticated — run `claude login` "
     "(or renew the provider's credentials), confirm with `claude auth status`, then resume"),
    (r"request exceeded \d+ms|timed out|ETIMEDOUT|idle timeout|deadline",
     "infra:provider_timeout",
     "fresh agent session (smaller context per turn); raise pi `retry.provider.timeoutMs` / "
     "`httpIdleTimeoutMs` and PI_CLAUDE_CODE_PROVIDER_*_TIMEOUT_MS if it recurs"),
    (r"context_length_exceeded|prompt is too long|context window exceeded|context length",
     "infra:context_overflow",
     "fresh agent session; the correction prompt asks for narrower reads and shorter tool output"),
    (r"overloaded|rate.?limit|too many requests|\b429\b|\b529\b|\b5\d\d\b.*(error|status)|ECONNRESET|"
     r"socket hang up|usage limit|out of usage|quota|service unavailable",
     "infra:provider_error",
     "fresh agent session after a pause; the provider, not the work, failed"),
    (r"PermissionBreach|is limited to .* but modified|is barred from|is read-only but modified",
     "infra:permission_breach",
     "an agent (or the operator, concurrently) wrote outside its allowlist; the change was "
     "rolled back — retry with a fresh session and nothing else touching the tree"),
]

JSON_MISS = re.compile(r"never produced valid \w+ JSON|no JSON object found", re.I)


def _since_ms(iso: str) -> int:
    try:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, AttributeError):
        return 0


def _scan_raw(path: Path, since_ms: int = 0) -> tuple[Counter, int, bool]:
    """Provider error messages, error-turn count, and whether any assistant text exists —
    counted only for turns at or after `since_ms`, so a resumed session's old errors do
    not colour the verdict on the phase that just died."""
    messages: Counter = Counter()
    error_turns = 0
    any_text = False
    if not path.exists():
        return messages, error_turns, any_text
    with path.open() as fh:
        for line in fh:
            if '"errorMessage"' not in line and '"stopReason":"error"' not in line \
                    and '"type":"text"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            msg = event.get("message") if isinstance(event.get("message"), dict) else event
            if not isinstance(msg, dict):
                continue
            stamp = msg.get("timestamp") or event.get("timestamp") or 0
            if since_ms and isinstance(stamp, (int, float)) and stamp < since_ms:
                continue
            if msg.get("stopReason") == "error":
                error_turns += 1
                text = str(msg.get("errorMessage") or "").strip()
                if text:
                    messages[text[:200]] += 1
            for part in msg.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip():
                    any_text = True
    return messages, error_turns, any_text


def classify(run, agent: str, error_text: str, since_iso: str = "") -> Triage:
    """Decide infra vs product from the raised error plus the agent's raw stream."""
    raw = run.session_dir / agent / "raw_output.jsonl"
    if not since_iso and getattr(run, "phases", None):
        since_iso = run.phases[-1].started_at or ""
    messages, error_turns, any_text = _scan_raw(raw, _since_ms(since_iso))
    top = messages.most_common(1)[0][0] if messages else ""
    haystack = f"{error_text}\n{top}"

    for pattern, cls, remedy in PATTERNS:
        if re.search(pattern, haystack, re.I):
            evidence = (f"{agent}: {cls} — provider reported {error_turns} error turn(s); "
                        f"most frequent: {top or error_text[:200]!r}")
            return Triage(cls=cls, infra=True, agent=agent, evidence=evidence, remedy=remedy,
                          error_counts=dict(messages.most_common(5)))

    if JSON_MISS.search(error_text) and (error_turns or not any_text):
        # No usable final response and either provider errors or no words at all —
        # the agent never got to speak, which is not a defect in what it built.
        return Triage(cls="infra:empty_response", infra=True, agent=agent,
                      evidence=f"{agent}: no JSON and {error_turns} provider error turn(s); "
                               f"assistant text present: {any_text}",
                      remedy="fresh agent session; check provider health if it recurs",
                      error_counts=dict(messages.most_common(5)))

    return Triage(cls="product", infra=False, agent=agent,
                  evidence=error_text[:400], remedy="corrective cycle against the recorded failure",
                  error_counts=dict(messages.most_common(5)))
