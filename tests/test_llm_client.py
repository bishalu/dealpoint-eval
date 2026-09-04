import json

import pytest

from dealpoint.llm.client import FakeClient, ScriptedToolCall, ScriptedTurn, build_system_message

pytestmark = pytest.mark.gate_m1


def test_fake_client_never_touches_ledger(tmp_path):
    ledger_path = tmp_path / "spend_ledger.jsonl"
    client = FakeClient(script=[ScriptedTurn(content='{"answer": "ABSTAIN"}')])
    result = client.chat(messages=[{"role": "user", "content": "hi"}], model="fake/model")
    assert result.content == '{"answer": "ABSTAIN"}'
    assert result.cost_usd == 0.0
    assert not ledger_path.exists()


def test_fake_client_records_messages_it_was_given():
    client = FakeClient(script=[ScriptedTurn(content="ok")])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    client.chat(messages=messages, model="fake/model", max_tokens=100, temperature=0)
    assert len(client.calls) == 1
    assert client.calls[0]["messages"] == messages
    assert client.calls[0]["max_tokens"] == 100
    assert client.calls[0]["temperature"] == 0


def test_fake_client_replays_scripted_tool_calls_then_final_answer():
    client = FakeClient(
        script=[
            ScriptedTurn(tool_calls=[ScriptedToolCall(name="search_agreement", arguments={"query": "x"})]),
            ScriptedTurn(content='{"answer": "ABSTAIN", "evidence": [], "rationale": "n/a"}'),
        ]
    )
    first = client.chat(messages=[], model="fake/model")
    assert first.finish_reason == "tool_calls"
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].name == "search_agreement"

    second = client.chat(messages=[], model="fake/model")
    assert second.finish_reason == "stop"
    assert second.content is not None
    payload = json.loads(second.content)
    assert payload["answer"] == "ABSTAIN"


def test_fake_client_raises_scripted_error():
    boom = RuntimeError("boom")
    client = FakeClient(script=[ScriptedTurn(error=boom)])
    with pytest.raises(RuntimeError):
        client.chat(messages=[], model="fake/model")


def test_build_system_message_carries_cache_control_on_static_prefix():
    msg = build_system_message("some static prefix")
    assert msg["role"] == "system"
    assert isinstance(msg["content"], list)
    block = msg["content"][-1]
    assert block["text"] == "some static prefix"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_openrouter_client_chat_payload_carries_usage_include_temperature_and_cache_control(
    tmp_path, monkeypatch
):
    pytest.importorskip("openai")
    from dealpoint.llm.client import OpenRouterClient

    captured = {}

    class _FakeUsageDetails:
        cached_tokens = 5
        cache_write_tokens = 7

    class _FakeUsage:
        prompt_tokens = 100
        completion_tokens = 20
        cost = 0.001
        prompt_tokens_details = _FakeUsageDetails()

    class _FakeMessage:
        content = '{"answer": "ABSTAIN", "evidence": [], "rationale": "n/a"}'
        tool_calls = None

    class _FakeChoice:
        message = _FakeMessage()
        finish_reason = "stop"

    class _FakeResponse:
        choices = [_FakeChoice()]  # noqa: RUF012 - test double, not a dataclass
        usage = _FakeUsage()

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeSDKClient:
        chat = _FakeChat()

    monkeypatch.setattr(
        "openai.OpenAI", lambda **kwargs: _FakeSDKClient(), raising=True
    )

    ledger_path = tmp_path / "spend_ledger.jsonl"
    client = OpenRouterClient(api_key="fake-key", ledger_path=ledger_path)
    messages = [build_system_message("static prefix"), {"role": "user", "content": "hi"}]
    result = client.chat(messages=messages, model="anthropic/claude-haiku-4.5", max_tokens=300, temperature=0)

    assert captured["extra_body"] == {"usage": {"include": True}}
    assert captured["temperature"] == 0
    assert captured["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    assert result.cost_usd == 0.001
    assert result.cached_tokens == 5
    assert result.cache_write_tokens == 7

    assert ledger_path.exists()
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["usd"] == 0.001
    assert "usd" in rows[0]
