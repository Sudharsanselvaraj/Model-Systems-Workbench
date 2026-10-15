"""Claude Haiku 5.5 through the existing chat adapter, validators and runner: mocked transport only, no network."""

import json

import pytest

from msu_llm_lab.classification.chat_adapter import ChatClassificationAdapter
from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.records import Task
from msu_llm_lab.models import registry
from msu_llm_lab.providers.openrouter import OpenRouterAdapter
from synthetic_fixtures import SYN_INTENT_PROMPT, SYN_LEAD_PROMPT
from test_runner_adapters import chat_body, net, unblock  # noqa: F401

HAIKU = "anthropic/claude-haiku-5.5"
STATE = "[HISTORY]\n[NO PRIOR HISTORY]\n[CURRENT USER MESSAGE]\nwhat are the fees?"


def adapter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    return ChatClassificationAdapter(model_id=HAIKU, provider=OpenRouterAdapter(registry_stage="pilot", sleep=lambda s: None))


def reply(text, model="anthropic/claude-haiku-5.5-20261007", provider="Anthropic"):
    return chat_body(text, model=model, provider=provider)


def test_haiku_intent_and_lead_requests_use_the_pinned_anthropic_route_without_temperature(net, monkeypatch):
    a = adapter(monkeypatch)
    net.script = [reply("QUERY"), reply('{"need_cta": true}')]
    r1 = a.classify(Task.INTENT, STATE, SYN_INTENT_PROMPT, live_authorized=True)
    r2 = a.classify(Task.LEAD, "asked for the application link", SYN_LEAD_PROMPT, live_authorized=True)
    b1, b2 = net.bodies
    for b in (b1, b2):
        assert b["model"] == HAIKU and b["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
        assert "temperature" not in b and "reasoning" not in b and b["messages"][0]["role"] == "system"
    assert b1["max_tokens"] == 16 and b2["max_tokens"] == 24
    assert r1.ok and r1.predicted_label == IntentLabel.QUERY and r2.ok and r2.predicted_need_cta is True
    assert r1.resolved_model_id == "anthropic/claude-haiku-5.5-20261007" and r1.serving_provider == "Anthropic" and r1.requested_route == {"order": ["anthropic"], "allow_fallbacks": False}


@pytest.mark.parametrize("text,issue", [("QUERY because it is about fees", "EXTRA_TEXT"), ("", "EMPTY_OUTPUT"), ("pricing", "UNKNOWN_LABEL")])
def test_haiku_outputs_that_break_the_intent_contract_are_invalid_not_predictions(net, monkeypatch, text, issue):
    a = adapter(monkeypatch)
    net.script = [reply(text)]
    r = a.classify(Task.INTENT, STATE, SYN_INTENT_PROMPT, live_authorized=True)
    assert not r.ok and r.predicted_label is None and issue in r.issues


def test_haiku_lead_output_must_be_exactly_the_one_key_json(net, monkeypatch):
    a = adapter(monkeypatch)
    net.script = [reply('{"need_cta": false}'), reply('{"need_cta": "yes"}'), reply('Sure: {"need_cta": true}')]
    ok, bad1, bad2 = (a.classify(Task.LEAD, "x", SYN_LEAD_PROMPT, live_authorized=True) for _ in range(3))
    assert ok.ok and ok.predicted_need_cta is False
    assert not bad1.ok and bad1.predicted_need_cta is None and not bad2.ok and bad2.predicted_need_cta is None


def test_haiku_usage_native_cost_and_estimate_are_kept_apart(net, monkeypatch):
    a = adapter(monkeypatch)
    body = reply("GREETING")
    body["usage"] = {"prompt_tokens": 2000, "completion_tokens": 3, "cost": 0.0002015}
    net.script = [body]
    r = a.classify(Task.INTENT, STATE, SYN_INTENT_PROMPT, live_authorized=True)
    assert (r.prompt_tokens, r.completion_tokens, r.usage_source) == (2000, 3, "PROVIDER_REPORTED")
    assert float(r.native_cost) == pytest.approx(0.0002015) and r.actual_cost_usd is None                    # credits are never booked as USD
    assert float(r.estimated_cost_usd) == pytest.approx(2000 * 0.10 / 1e6 + 3 * 0.50 / 1e6, abs=1e-6)       # catalogue price x reported tokens
    assert r.model_latency_ms is None and r.latency_ms is not None


def test_haiku_is_refused_before_any_request_when_its_route_is_blocked(net, monkeypatch):
    from registry_helpers import block_route
    block_route(monkeypatch, "claude-haiku-5.5")
    a = adapter(monkeypatch)
    from msu_llm_lab.providers.base import ProviderError
    with pytest.raises(ProviderError, match="registry blockers"):
        a.classify(Task.INTENT, STATE, SYN_INTENT_PROMPT, live_authorized=True)
    assert net.requests == []
