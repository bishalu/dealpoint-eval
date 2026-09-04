"""Pins the absence of skill content from the shared system prompt (M4 only).

specs/grilled-product-brief.md Appendix B skill rule 6 ("Before abstaining,
run >= 2 searches with different phrasing") must not appear in
`dealpoint.agent.prompts.system_prompt()`, which is shown in every arm
before M4. This test exists so that leak cannot come back silently.
"""

from __future__ import annotations

import pytest

from dealpoint.agent.prompts import system_prompt

pytestmark = pytest.mark.gate_m1


def test_system_prompt_has_no_skill_content():
    prompt = system_prompt()
    assert "different phrasing" not in prompt
    assert "two searches" not in prompt


def test_system_prompt_still_states_abstain_only_when_genuinely_absent():
    prompt = system_prompt()
    assert "genuinely absent" in prompt
    assert "not merely hard to find" in prompt
