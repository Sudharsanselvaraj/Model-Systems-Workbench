"""Offline tests for the Jev adapter, the Chat adapter and the classification runner.

Only hand-written SYNTHETIC fixtures and a fake urlopen are used. Sockets are blocked by conftest.py, and
`urllib.request.urlopen` is replaced, so no real request can happen. Registry blockers are patched away ONLY in tests
that need an adapter to reach the fake transport; separate tests show the real registry blocks both models.
"""

import io
import json
import urllib.error
import urllib.request
from decimal import Decimal

import pytest

from msu_llm_lab.classification import runner as runner_mod
from msu_llm_lab.classification.chat_adapter import ChatClassificationAdapter, NANO_MODEL_ID
from msu_llm_lab.classification.contracts import IntentLabel, ReviewStatus
from msu_llm_lab.classification.jev_adapter import (
    JEV_ENDPOINT, JEV_MODEL_ID, JevDecisionsAdapter, parse_decisions_response, version_consistent,
)
from msu_llm_lab.classification.records import AdapterResult, PredictionRecord, PredictionStore, Task
from msu_llm_lab.classification.runner import (
    BatchRejectedError, ClassificationRunner, ModelNotReadyError, prompt_version,
)
from msu_llm_lab.classification.threshold import ProbItem, ThresholdError, evaluate_held_out, select_threshold_on_dev
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.config import BudgetPolicy
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.models import registry
from msu_llm_lab.models.registry import Protocol
from msu_llm_lab.providers.base import (
    LiveNotAuthorizedError, ProviderError, ProviderResponse, UsageUnavailableError,
)
from msu_llm_lab.providers.http import RetryingPoster
from msu_llm_lab.providers.openrouter import OpenRouterAdapter
from msu_llm_lab.schemas.evaluation import DataClassification as DC, DatasetSplit
from msu_llm_lab.storage.db import DatabaseManager
from registry_helpers import block_route
from synthetic_fixtures import SYN_INTENT_PROMPT, SYN_LEAD_PROMPT, A, U, intent_case, lead_case

FAKE_KEY = "test-not-a-real-key"
LABELS = [l.value for l in IntentLabel]
INTENT_PROMPT, LEAD_PROMPT = SYN_INTENT_PROMPT, SYN_LEAD_PROMPT


# ------------------------------------------------------------------ fake transport
class FakeNet:
    def __init__(self):
        self.script, self.requests = [], []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if not self.script:
            raise AssertionError("unexpected extra request")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        body = item if isinstance(item, bytes) else json.dumps(item).encode()
        return _Resp(body)

    @property
    def bodies(self):
        return [json.loads(r.data) for r in self.requests]


class _Resp:
    def __init__(self, b):
        self.b = b

    def read(self):
        return self.b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_err(code, body=b"{}"):
    import email.message
    return urllib.error.HTTPError("https://example.invalid", code, "x", email.message.Message(), io.BytesIO(body))


@pytest.fixture
def net(monkeypatch):
    n = FakeNet()
    monkeypatch.setattr(urllib.request, "urlopen", n)
    return n


@pytest.fixture
def unblock(monkeypatch):
    """Pretend registry blockers are resolved, so adapters can reach the FAKE transport. Real registry blocks both models."""
    monkeypatch.setattr(registry, "synthetic_live_blockers", lambda model_id, stage='pilot': [])


def jev_choice(label="QUERY", conf=0.7, **over):
    probs = {l: 0.025 for l in LABELS}
    probs[label] = 0.9
    ans = {"type": "choice", "choice": label, "probabilities": probs, "confidence": conf}
    ans.update(over)
    return ans


def jev_body(answer=None, task="intent", **over):
    body = {"id": "gen-synthetic-1", "model": "typesafe/jev-1.14-20261001", "provider": "SynthProvider",
            "answers": {task: answer if answer is not None else jev_choice()},
            "usage": {"input_tokens": 1200, "output_tokens": 3, "cost": 0.00005}}
    body.update(over)
    return body


def noul_body(p=0.37, **over):
    return jev_body({"type": "noul", "noul": p}, task="lead", **over)


def jev_adapter(**kw):
    return JevDecisionsAdapter(api_key=FAKE_KEY, poster=RetryingPoster(sleep=lambda s: None), **kw)


STATE = "[HISTORY]\n[NO PRIOR HISTORY]\n[CURRENT USER MESSAGE]\nwhat are the fees?"


# ------------------------------------------------------------------ Jev: Choice
def test_valid_choice_response_captures_everything(net, unblock):
    net.script = [jev_body()]
    r = jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert r.ok and r.predicted_label == IntentLabel.QUERY and r.issues == []
    assert r.label_probs["QUERY"] == 0.9 and r.native_confidence == 0.7 and r.confidence_source == "NATIVE_DISTRIBUTION"
    assert (r.prompt_tokens, r.completion_tokens, r.usage_source) == (1200, 3, "PROVIDER_REPORTED")
    assert r.native_cost == Decimal("0.00005") and r.native_cost_unit == "OPENROUTER_CREDITS_USD_RATE_UNVERIFIED" and r.actual_cost_usd is None
    assert r.estimated_cost_usd == Decimal("0.000050") and r.cost_basis == "ESTIMATED_FROM_CATALOG"  # 1200 tokens x $0.042/M, TypeSafe route
    assert r.resolved_model_id == "typesafe/jev-1.14-20261001" and r.serving_provider == "SynthProvider"
    assert r.latency_ms is not None and r.model_latency_ms is None and r.attempts == 1
    assert r.predicted_need_cta is None and r.yes_prob is None


def test_request_uses_documented_fields_only_and_never_sends_labels(net, unblock):
    net.script = [jev_body(), noul_body()]
    case = intent_case(history=[U(0, "hi"), A(1, "hello")], current=U(2, "fees?"), review_status=ReviewStatus.APPROVED,
                       expected_intent=IntentLabel.GREETING)
    adapter = jev_adapter()
    # 1. Intent task: Choice question
    adapter.classify(Task.INTENT, runner_mod._state_text(Task.INTENT, case), INTENT_PROMPT, live_authorized=True)
    req, body = net.requests[0], net.bodies[0]
    assert req.get_method() == "POST"
    assert req.full_url == "https://openrouter.ai/api/alpha/decisions"
    assert req.full_url == JEV_ENDPOINT
    assert req.get_header("Authorization") == f"Bearer {FAKE_KEY}"
    assert req.get_header("Content-type") == "application/json"
    assert set(body.keys()) == {"model", "state", "questions"} and body["model"] == "typesafe/jev-1.14"
    assert isinstance(body["state"], list) and isinstance(body["state"][0], str)
    q = body["questions"]["intent"]
    assert q["type"] == "choice" and list(q["criteria"]) == LABELS and "Synthetic first step" in q["instructions"]
    assert "expected" not in json.dumps(body).lower()

    # 2. Lead Qualifier task: Noul question
    l_case = lead_case()
    adapter.classify(Task.LEAD, runner_mod._state_text(Task.LEAD, l_case), LEAD_PROMPT, live_authorized=True)
    req2, body2 = net.requests[1], net.bodies[1]
    assert req2.get_method() == "POST"
    assert req2.full_url == "https://openrouter.ai/api/alpha/decisions"
    assert req2.get_header("Authorization") == f"Bearer {FAKE_KEY}"
    assert req2.get_header("Content-type") == "application/json"
    assert set(body2.keys()) == {"model", "state", "questions"} and body2["model"] == "typesafe/jev-1.14"
    assert isinstance(body2["state"], list) and isinstance(body2["state"][0], str)
    q2 = body2["questions"]["lead"]
    assert q2["type"] == "noul" and "instructions" in q2
    assert "expected" not in json.dumps(body2).lower()


@pytest.mark.parametrize("answer,issue", [
    (jev_choice(choice="greeting"), "CHOICE_NOT_A_VALID_LABEL"),
    (jev_choice(probabilities={"QUERY": 1.0}), "PROBABILITY_KEYS_MISMATCH"),
    ({**jev_choice(), "confidence": None}, "CONFIDENCE_MISSING_OR_OUT_OF_RANGE"),
    ({**jev_choice(), "confidence": 1.7}, "CONFIDENCE_MISSING_OR_OUT_OF_RANGE"),
    ({"type": "noul", "noul": 0.5}, "NOT_A_CHOICE_ANSWER"),
    ("QUERY", "NOT_A_CHOICE_ANSWER"),
])
def test_malformed_choice_answers_are_rejected_without_a_label(net, unblock, answer, issue):
    net.script = [jev_body(answer)]
    r = jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert not r.ok and issue in r.issues and r.predicted_label is None and r.label_probs is None


@pytest.mark.parametrize("mutate,issue", [
    (lambda b: b.pop("answers"), "ANSWERS_MISSING_OR_MALFORMED"),
    (lambda b: b.update(answers=[]), "ANSWERS_MISSING_OR_MALFORMED"),
    (lambda b: b.update(answers={}), "ANSWER_MISSING_FOR_QUESTION"),
    (lambda b: b["answers"].update(extra=jev_choice()), "UNEXPECTED_ANSWER_KEYS"),
    (lambda b: b.update(usage="lots"), "USAGE_MALFORMED"),
    (lambda b: b["usage"].update(input_tokens="12"), "USAGE_TOKEN_FIELD_MALFORMED"),
    (lambda b: b["usage"].update(cost="free"), "USAGE_COST_MALFORMED"),
    (lambda b: b.update(model=123), "MODEL_FIELD_MALFORMED"),
    (lambda b: b.update(provider=["x"]), "PROVIDER_FIELD_MALFORMED"),
    (lambda b: b.update(model="typesafe/jev-1.13-20260917"), "RESOLVED_VERSION_DIFFERS_FROM_REQUESTED"),
    (lambda b: b.update(error={"message": "x"}), "ERROR_BODY_IN_SUCCESS_RESPONSE"),
])
def test_malformed_or_unexpected_envelopes_fail_safely(net, unblock, mutate, issue):
    body = jev_body()
    mutate(body)
    net.script = [body]
    r = jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert not r.ok and issue in r.issues and r.predicted_label is None


def test_non_object_and_non_json_responses():
    assert parse_decisions_response(Task.INTENT, JEV_MODEL_ID, ["x"]).issues == ["ENVELOPE_NOT_AN_OBJECT"]


def test_non_json_body_raises_without_retry(net, unblock):
    net.script = [b"<html>not json</html>", jev_body()]
    with pytest.raises(ProviderError) as ei:
        jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert ei.value.retryable is False and len(net.requests) == 1


def test_missing_usage_and_cost_stay_null_not_fabricated(net, unblock):
    body = jev_body()
    del body["usage"], body["model"], body["provider"], body["id"]
    net.script = [body]
    r = jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert r.ok  # the answer itself is valid; absent documented fields are warnings, not inventions
    assert r.prompt_tokens is None and r.completion_tokens is None and r.usage_source == "UNAVAILABLE"
    assert r.actual_cost_usd is None and r.estimated_cost_usd is None and r.cost_basis == "UNAVAILABLE"
    assert r.resolved_model_id is None and r.serving_provider is None
    assert {"USAGE_MISSING", "RESOLVED_MODEL_MISSING", "PROVIDER_MISSING", "REQUEST_ID_MISSING"} <= set(r.warnings)


def test_cost_missing_but_tokens_present_gives_a_separate_estimate_on_the_verified_route(net, unblock):
    body = jev_body()
    del body["usage"]["cost"]
    net.script = [body]
    r = jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert r.prompt_tokens == 1200 and r.actual_cost_usd is None
    assert r.estimated_cost_usd == (Decimal(1200) * Decimal("0.042") / Decimal(1_000_000)).quantize(Decimal("0.000001"))
    assert r.cost_basis == "ESTIMATED_FROM_CATALOG" and "COST_MISSING" in r.warnings


def test_estimate_withheld_when_route_price_is_not_verified(net, unblock, monkeypatch):
    monkeypatch.setattr(registry, "estimate_permitted", lambda model_id: False)
    body = jev_body()
    del body["usage"]["cost"]
    net.script = [body]
    r = jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert r.prompt_tokens == 1200 and r.estimated_cost_usd is None and r.actual_cost_usd is None and r.cost_basis == "UNAVAILABLE"


def test_version_consistency_rule():
    assert version_consistent("typesafe/jev-1.14", "typesafe/jev-1.14")
    assert version_consistent("typesafe/jev-1.14", "typesafe/jev-1.14-20261001")
    assert not version_consistent("typesafe/jev-1.14", "typesafe/jev-1.13-20260917")
    assert not version_consistent("typesafe/jev-1.14", "typesafe/jev-1.140")
    assert not version_consistent("typesafe/jev-1.14", "typesafe/jev-1.14-latest")
    assert not version_consistent("typesafe/jev-1.14", "typesafe/jev-1.14-20261301")  # not a real date


# ------------------------------------------------------------------ Jev: Noul
def test_valid_noul_response_gives_probability_only(net, unblock):
    net.script = [noul_body(0.37)]
    r = jev_adapter().classify(Task.LEAD, "asked about scholarship options", LEAD_PROMPT, live_authorized=True)
    assert r.ok and r.yes_prob == 0.37 and r.predicted_need_cta is None  # no silent threshold
    assert r.native_confidence is None and r.confidence_source == "NONE" and r.label_probs is None
    body = net.bodies[0]
    assert body["questions"]["lead"]["type"] == "noul" and "criteria" not in body["questions"]["lead"]


@pytest.mark.parametrize("answer,issue", [
    ({"type": "noul"}, "YES_PROBABILITY_MISSING_OR_OUT_OF_RANGE"),
    ({"type": "noul", "noul": 1.2}, "YES_PROBABILITY_MISSING_OR_OUT_OF_RANGE"),
    ({"type": "noul", "noul": "0.4"}, "YES_PROBABILITY_MISSING_OR_OUT_OF_RANGE"),
    ({"type": "noul", "noul": True}, "YES_PROBABILITY_MISSING_OR_OUT_OF_RANGE"),
    ({"type": "noul", "noul": 0.4, "confidence": 0.9}, "UNEXPECTED_CONFIDENCE_FIELD"),
    (jev_choice(), "NOT_A_NOUL_ANSWER"),
])
def test_malformed_noul_answers_are_rejected(net, unblock, answer, issue):
    net.script = [jev_body(answer, task="lead")]
    r = jev_adapter().classify(Task.LEAD, "x", LEAD_PROMPT, live_authorized=True)
    assert not r.ok and issue in r.issues and r.yes_prob is None


# ------------------------------------------------------------------ Jev: guards before network
def test_jev_unauthorized_unblocked_and_keyless_requests_never_reach_network(net, monkeypatch):
    with pytest.raises(LiveNotAuthorizedError):
        jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=False)
    assert registry.synthetic_live_blockers(JEV_MODEL_ID, "pilot") == []   # schema facts are LIVE_OBSERVED
    with monkeypatch.context() as m:
        block_route(m, "jev-1.14")
        with pytest.raises(ProviderError, match="registry blockers"):       # a real blocker still stops the request before the network
            jev_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        JevDecisionsAdapter(api_key=None).classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert net.requests == []


# ------------------------------------------------------------------ Chat adapter
def chat_adapter():
    return ChatClassificationAdapter(provider=OpenRouterAdapter(api_key=FAKE_KEY, sleep=lambda s: None))


def chat_body(text, usage=True, **over):
    b = {"choices": [{"message": {"content": text}}], "model": "openai/gpt-4.1-nano-2025-04-14", "provider": "OpenAI"}
    if usage:
        b["usage"] = {"prompt_tokens": 800, "completion_tokens": 4, "cost": 0.00009}
    b.update(over)
    return b


def test_chat_intent_valid_label_pinned_route_and_no_schema_mode(net, unblock):
    net.script = [chat_body("GREETING")]
    r = chat_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert r.ok and r.predicted_label == IntentLabel.GREETING and r.confidence_source == "NONE" and r.label_probs is None
    body = net.bodies[0]
    assert body["provider"] == {"order": ["azure"], "allow_fallbacks": False}
    assert "response_format" not in body and body["temperature"] == 0.0 and body["max_tokens"] == 16
    assert body["messages"][0] == {"role": "system", "content": INTENT_PROMPT}
    assert (r.resolved_model_id, r.serving_provider) == ("openai/gpt-4.1-nano-2025-04-14", "OpenAI")
    assert (r.prompt_tokens, r.completion_tokens) == (800, 4) and r.cost_basis == "ESTIMATED_FROM_CATALOG" and r.native_cost == Decimal("0.00009")


@pytest.mark.parametrize("text,reason", [
    ("query", "WRONG_CASE"), ("QUERY.", "EXTRA_TEXT"), ("I think QUERY", "EXTRA_TEXT"), ("", "EMPTY_OUTPUT"), ("HELLO", "UNKNOWN_LABEL"),
])
def test_chat_invalid_intent_labels(net, unblock, text, reason):
    net.script = [chat_body(text)]
    r = chat_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert not r.ok and r.predicted_label is None and r.issues == [reason] and r.raw_output == text


def test_chat_lead_json_contract(net, unblock):
    net.script = [chat_body('{"need_cta": true}')]
    r = chat_adapter().classify(Task.LEAD, "asked how to pay the deposit", LEAD_PROMPT, live_authorized=True)
    assert r.ok and r.predicted_need_cta is True and net.bodies[0]["max_tokens"] == 24


@pytest.mark.parametrize("text,reason", [
    ('```json\n{"need_cta": true}\n```', "CODE_FENCE"), ('{"need_cta": "yes"}', "NON_BOOLEAN"),
    ('{"need_cta": true, "why": "x"}', "EXTRA_KEYS"), ("true", "NOT_OBJECT"), ('{"cta": true}', "MISSING_KEY"),
])
def test_chat_malformed_lead_json(net, unblock, text, reason):
    net.script = [chat_body(text)]
    r = chat_adapter().classify(Task.LEAD, "x", LEAD_PROMPT, live_authorized=True)
    assert not r.ok and r.predicted_need_cta is None and r.issues == [reason]


def test_chat_missing_usage_is_null_and_cost_unavailable(net, unblock):
    net.script = [chat_body("QUERY", usage=False)]
    r = chat_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert r.prompt_tokens is None and r.usage_source == "UNAVAILABLE" and r.cost_basis == "UNAVAILABLE" and "USAGE_MISSING" in r.warnings


def test_chat_blocked_route_is_refused_before_network(net, monkeypatch):
    block_route(monkeypatch, "gpt-4.1-nano")
    with pytest.raises(ProviderError, match="registry blockers"):
        chat_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    with pytest.raises(LiveNotAuthorizedError):
        chat_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=False)
    assert net.requests == []


# ------------------------------------------------------------------ Runner
@pytest.fixture
def env(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "t.db")
    budget = BudgetService(db=db)
    store = PredictionStore(tmp_path / "runs" / "preds.jsonl")
    return db, budget, store


def make_runner(env, adapters=None, **kw):
    _, budget, store = env
    return ClassificationRunner(adapters or {JEV_MODEL_ID: jev_adapter()}, store, budget, **kw)


AUTH = LiveAuthorization(models=(JEV_MODEL_ID, NANO_MODEL_ID))


def cases(n=2, **kw):
    return [intent_case(case_id=f"syn-i-{i}", current=U(0, f"synthetic question {i}"), **kw) for i in range(n)]


class SpyAdapter:
    protocol = Protocol.DECISIONS_API
    model_id = JEV_MODEL_ID

    def __init__(self):
        self.calls = 0

    def supports(self, task):
        return True

    def max_output_tokens(self, task):
        return 0

    def classify(self, *a, **k):
        self.calls += 1
        raise AssertionError("adapter must not be reached")


def test_authorization_failure_happens_before_anything_else(env, net, monkeypatch):
    spy = SpyAdapter()
    monkeypatch.setattr(runner_mod.serialization, "serialize_intent_case", lambda c: pytest.fail("request material built"))
    r = make_runner(env, {JEV_MODEL_ID: spy})
    for bad in (None, LiveAuthorization(models=("other/model",))):
        with pytest.raises(LiveNotAuthorizedError):
            r.run(task=Task.INTENT, cases=cases(), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=bad)
    assert spy.calls == 0 and net.requests == [] and env[2].load() == []


def test_one_ineligible_case_rejects_the_whole_batch_before_any_request_is_built(env, net, monkeypatch, unblock):
    spy = SpyAdapter()
    monkeypatch.setattr(runner_mod.serialization, "serialize_intent_case", lambda c: pytest.fail("request material built"))
    mixed = cases(3)
    mixed[1] = intent_case(case_id="syn-i-bad", classification=DC.UNCLASSIFIED)
    mixed.append(intent_case(case_id="syn-i-conf", classification=DC.CONFIDENTIAL))
    with pytest.raises(BatchRejectedError) as ei:
        make_runner(env, {JEV_MODEL_ID: spy}).run(task=Task.INTENT, cases=mixed, model_id=JEV_MODEL_ID,
                                                 prompt_text=INTENT_PROMPT, authorization=AUTH)
    assert set(ei.value.reasons) == {"syn-i-bad", "syn-i-conf"}
    assert spy.calls == 0 and net.requests == [] and env[2].load() == []


def test_wrong_case_type_and_duplicate_ids_rejected(env, net, unblock):
    r = make_runner(env)
    with pytest.raises(BatchRejectedError):
        r.run(task=Task.LEAD, cases=cases(1), model_id=JEV_MODEL_ID, prompt_text=LEAD_PROMPT, authorization=AUTH)
    dup = [intent_case(case_id="same"), intent_case(case_id="same")]
    with pytest.raises(BatchRejectedError):
        r.run(task=Task.INTENT, cases=dup, model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH)
    assert net.requests == []


def test_unknown_unverified_and_blocked_models_stop_before_network(env, net, monkeypatch):
    block_route(monkeypatch, "gpt-4.1-nano")
    block_route(monkeypatch, "jev-1.14")
    r = make_runner(env)
    with pytest.raises(ModelNotReadyError, match="No adapter"):
        r.run(task=Task.INTENT, cases=cases(), model_id="mystery/model", prompt_text=INTENT_PROMPT,
              authorization=LiveAuthorization(models=("mystery/model",)))
    # unknown registry record
    class NoRec(SpyAdapter):
        model_id = "mystery/model"
    with pytest.raises(ModelNotReadyError, match="unknown model configuration"):
        make_runner(env, {"mystery/model": NoRec()}).run(task=Task.INTENT, cases=cases(), model_id="mystery/model",
                                                          prompt_text=INTENT_PROMPT, authorization=LiveAuthorization(models=("mystery/model",)))
    # blocked routes block both adapters before any request
    for mid, ad in ((JEV_MODEL_ID, jev_adapter()), (NANO_MODEL_ID, chat_adapter())):
        with pytest.raises(ModelNotReadyError, match="Registry blockers"):
            make_runner(env, {mid: ad}).run(task=Task.INTENT, cases=cases(), model_id=mid, prompt_text=INTENT_PROMPT, authorization=AUTH)
    assert net.requests == [] and env[2].load() == []


def test_budget_failure_stops_before_network(tmp_path, net, unblock):
    db = DatabaseManager(db_path=tmp_path / "b.db")
    tiny = BudgetService(db=db, policy=BudgetPolicy(per_run_ceiling_usd=Decimal("0.000000001")))
    store = PredictionStore(tmp_path / "p.jsonl")
    r = ClassificationRunner({JEV_MODEL_ID: jev_adapter()}, store, tiny)
    with pytest.raises(BudgetExceededError):
        r.run(task=Task.INTENT, cases=cases(), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH)
    per_case = BudgetService(db=db, policy=BudgetPolicy(per_case_ceiling_usd=Decimal("0.000000001")))
    with pytest.raises(BudgetExceededError, match="per-case"):
        ClassificationRunner({JEV_MODEL_ID: jev_adapter()}, store, per_case).run(
            task=Task.INTENT, cases=cases(), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH)
    assert net.requests == [] and store.load() == []


def test_oversized_state_rejected_before_network(env, net, unblock):
    big = [intent_case(case_id="big", current=U(0, "x" * 200))]
    with pytest.raises(BatchRejectedError, match="size limit"):
        make_runner(env, max_state_bytes=100).run(task=Task.INTENT, cases=big, model_id=JEV_MODEL_ID,
                                                  prompt_text=INTENT_PROMPT, authorization=AUTH)
    assert net.requests == []


def test_successful_run_stores_records_without_ground_truth_and_records_spend(env, net, unblock):
    net.script = [jev_body(jev_choice("QUERY")), jev_body(jev_choice("GREETING"))]
    approved = cases(2, review_status=ReviewStatus.APPROVED, expected_intent=IntentLabel.OUT_OF_SCOPE)
    s = make_runner(env).run(task=Task.INTENT, cases=approved, model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT,
                             authorization=AUTH, run_id="run-1", dataset_version="syn-v0")
    assert (s.n_ok, s.n_invalid, s.n_error) == (2, 0, 0) and s.recorded_cost_usd == Decimal("0.00010")
    recs = env[2].load()
    assert [r.predicted_label for r in recs] == [IntentLabel.QUERY, IntentLabel.GREETING]
    r0 = recs[0]
    assert r0.is_ground_truth is False and r0.status == "OK" and r0.raw_output is None
    assert r0.requested_model_id == JEV_MODEL_ID and r0.resolved_model_id == "typesafe/jev-1.14-20261001"
    assert r0.prompt_version == prompt_version(INTENT_PROMPT) and r0.input_contract_version == "v1" and r0.dataset_version == "syn-v0"
    assert r0.parity_status == "PRODUCTION_PARITY_NOT_ESTABLISHED"
    stored = [json.loads(l) for l in env[2].path.read_text().splitlines()]
    assert all(not k.startswith("expected") and k not in ("review_status", "exclusion_reason") for d in stored for k in d)
    assert all(d["predicted_label"] != "OUT_OF_SCOPE" for d in stored)  # approved label differs from predictions; never copied in
    txs = env[1].get_status().recent_transactions
    assert len(txs) == 2 and all(t['is_simulated'] == 0 and t['is_preflight_estimate'] == 0 and t['cost_basis'] == "ESTIMATED_FROM_CATALOG" for t in txs)
    # request material is identical with and without an approved label
    plain = cases(2)
    assert runner_mod._state_text(Task.INTENT, approved[0]) == runner_mod._state_text(Task.INTENT, plain[0])
    assert "OUT_OF_SCOPE" not in json.dumps(net.bodies[0]["state"])


def test_nulls_are_persisted_without_fabricated_telemetry(env, net, unblock):
    body = jev_body()
    del body["usage"]["cost"]  # tokens reported, provider cost missing
    net.script = [body]
    make_runner(env).run(task=Task.INTENT, cases=cases(1), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    d = json.loads(env[2].path.read_text().splitlines()[0])
    assert d["model_latency_ms"] is None and d["native_cost"] is None and d["actual_cost_usd"] is None
    assert d["estimated_cost_usd"] == "0.000050" and d["cost_basis"] == "ESTIMATED_FROM_CATALOG" and d["latency_ms"] is not None


def test_native_cost_without_tokens_is_preserved_but_leaves_unresolved_spend(env, net, unblock):
    body = jev_body()
    del body["usage"]["input_tokens"], body["usage"]["output_tokens"]  # native cost present, tokens missing
    net.script = [body]
    with pytest.raises(UsageUnavailableError):
        make_runner(env).run(task=Task.INTENT, cases=cases(1), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    d = json.loads(env[2].path.read_text().splitlines()[0])
    assert d["prompt_tokens"] is None and d["usage_source"] == "UNAVAILABLE" and d["native_cost"] == "0.00005"   # kept as received
    assert d["actual_cost_usd"] is None and d["estimated_cost_usd"] is None and d["recorded_cost_usd"] is None and d["cost_basis"] == "UNAVAILABLE"
    assert env[1].get_status().unresolved_spend_count == 1


def test_cost_unavailable_stores_the_record_then_stops_the_run(env, net, unblock):
    body = jev_body()
    del body["usage"]
    net.script = [body, jev_body()]
    with pytest.raises(UsageUnavailableError):
        make_runner(env).run(task=Task.INTENT, cases=cases(2), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    recs = env[2].load()
    assert len(recs) == 1 and recs[0].recorded_cost_usd is None and recs[0].cost_basis == "UNAVAILABLE" and len(net.requests) == 1
    st = env[1].get_status()
    assert st.unresolved_spend_count == 1 and st.cumulative_spend_usd == 0   # explicit unresolved entry; no invented amount


def test_invalid_outputs_are_stored_as_invalid_not_dropped(env, net, unblock):
    net.script = [jev_body(jev_choice(choice="greeting"))]
    s = make_runner(env).run(task=Task.INTENT, cases=cases(1), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    rec = env[2].load()[0]
    assert s.n_invalid == 1 and rec.status == "INVALID_OUTPUT" and rec.predicted_label is None and "CHOICE_NOT_A_VALID_LABEL" in rec.validation_issues


def test_provider_error_is_recorded_with_only_what_the_failure_produced(env, net, unblock):
    net.script = [http_err(400, b'{"error":{"message":"bad Bearer abcdefgh12345678"}}'), jev_body()]
    s = make_runner(env).run(task=Task.INTENT, cases=cases(2), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    e, ok = env[2].load()
    assert e.status == "ERROR" and e.error_type == "ProviderError" and "abcdefgh12345678" not in e.error_message
    assert (e.latency_ms, e.prompt_tokens, e.resolved_model_id, e.recorded_cost_usd) == (None, None, None, None)
    assert e.attempts == 1 and e.retry_errors == ["HTTP_400"] and ok.status == "OK" and s.n_error == 1 and s.n_ok == 1


def test_systemic_errors_and_consecutive_failures_abort(env, net, unblock):
    net.script = [http_err(401), jev_body()]
    s = make_runner(env).run(task=Task.INTENT, cases=cases(2), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="a")
    assert s.aborted_reason.startswith("systemic") and len(net.requests) == 1
    net.script, net.requests = [http_err(400)] * 3 + [jev_body()], []
    s2 = make_runner(env).run(task=Task.INTENT, cases=cases(4), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="b")
    assert s2.aborted_reason == "3 consecutive errors" and len(net.requests) == 3


def test_resume_does_not_duplicate_work(env, net, unblock):
    net.script = [jev_body(), jev_body()]
    kw = dict(task=Task.INTENT, model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    make_runner(env).run(cases=cases(1), **kw)
    s = make_runner(env).run(cases=cases(2), **kw)
    assert s.n_skipped_existing == 1 and s.n_ok == 1 and len(net.requests) == 2 and len(env[2].load()) == 2
    with pytest.raises(ValueError):
        env[2].append(env[2].load()[0])


def test_per_case_cost_over_ceiling_stops_run_after_recording(tmp_path, net, unblock):
    db = DatabaseManager(db_path=tmp_path / "c.db")
    budget = BudgetService(db=db)
    store = PredictionStore(tmp_path / "p.jsonl")
    net.script = [jev_body(usage={"input_tokens": 1, "output_tokens": 1, "cost": 0.5})]
    with pytest.raises(BudgetExceededError, match="per-case"):
        ClassificationRunner({JEV_MODEL_ID: jev_adapter()}, store, budget).run(
            task=Task.INTENT, cases=cases(2), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="r")
    assert len(store.load()) == 1 and len(net.requests) == 1


def test_lead_run_with_jev_and_chat_records_the_right_fields(env, net, unblock):
    net.script = [noul_body(0.8)]
    s = make_runner(env).run(task=Task.LEAD, cases=[lead_case()], model_id=JEV_MODEL_ID, prompt_text=LEAD_PROMPT, authorization=AUTH, run_id="j")
    j = env[2].load()[0]
    assert j.yes_prob == 0.8 and j.predicted_need_cta is None and j.native_confidence is None and j.protocol == Protocol.DECISIONS_API
    net.script = [chat_body('{"need_cta": false}')]
    ch = ClassificationRunner({NANO_MODEL_ID: chat_adapter()}, env[2], env[1])
    ch.run(task=Task.LEAD, cases=[lead_case("syn-l-2")], model_id=NANO_MODEL_ID, prompt_text=LEAD_PROMPT, authorization=AUTH, run_id="n")
    n = [r for r in env[2].load() if r.run_id == "n"][0]
    assert n.predicted_need_cta is False and n.yes_prob is None and n.protocol == Protocol.CHAT_COMPLETIONS


# ------------------------------------------------------------------ predictions are never ground truth
def test_predictions_can_never_be_marked_as_ground_truth():
    with pytest.raises(Exception):
        PredictionRecord(record_id="x", run_id="r", case_id="c", task=Task.INTENT, model_key="m", protocol=Protocol.DECISIONS_API,
                         requested_model_id="m", prompt_version="p", status="OK", is_ground_truth=True)


# ------------------------------------------------------------------ thresholds from stored predictions
def test_threshold_selected_on_dev_only_and_test_needs_frozen_threshold(env, net, unblock):
    probs = [0.9, 0.8, 0.4, 0.6, 0.3, 0.2]
    net.script = [noul_body(p) for p in probs]
    lead_cases = [lead_case(f"syn-l-{i}") for i in range(6)]
    make_runner(env).run(task=Task.LEAD, cases=lead_cases, model_id=JEV_MODEL_ID, prompt_text=LEAD_PROMPT, authorization=AUTH, run_id="t")
    recs = {r.case_id: r for r in env[2].load()}
    truth = {f"syn-l-{i}": t for i, t in enumerate([True, True, True, False, False, False])}  # synthetic labels, held apart from records
    items = [ProbItem(cid, truth[cid], r.yes_prob, DatasetSplit.DEV if i < 6 else DatasetSplit.TEST) for i, (cid, r) in enumerate(recs.items())]
    frozen = select_threshold_on_dev(items)
    assert frozen.selected_on == "DEV" and frozen.threshold == pytest.approx(0.35)
    test_items = [ProbItem(i.case_id, i.expected, i.yes_prob, DatasetSplit.TEST) for i in items]
    with pytest.raises(ThresholdError):
        evaluate_held_out(test_items, None)
    with pytest.raises(ThresholdError):
        select_threshold_on_dev(test_items)
    assert evaluate_held_out(test_items, frozen).frozen_threshold == frozen.threshold


# ------------------------------------------------------------------ route-aware requests (registry pins)
def test_temperature_is_omitted_when_the_pinned_route_does_not_list_it(net, unblock):
    net.script = [chat_body("QUERY"), chat_body("QUERY")]
    sonnet = ChatClassificationAdapter(model_id="anthropic/claude-sonnet-5.5", provider=OpenRouterAdapter(api_key=FAKE_KEY, sleep=lambda s: None))
    nano = chat_adapter()
    sonnet.classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    nano.classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    s_body, n_body = net.bodies
    assert "temperature" not in s_body and s_body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
    assert n_body["temperature"] == 0.0 and n_body["provider"] == {"order": ["azure"], "allow_fallbacks": False}


def test_requested_route_is_recorded_without_claiming_who_served_the_request(net, unblock, env):
    net.script = [chat_body("QUERY", provider=None)]
    r = ClassificationRunner({NANO_MODEL_ID: chat_adapter()}, env[2], env[1]).run(
        task=Task.INTENT, cases=cases(1), model_id=NANO_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="rt")
    rec = env[2].load()[0]
    assert rec.requested_route == {"order": ["azure"], "allow_fallbacks": False} and rec.serving_provider is None


def test_chat_adapter_404_routing_funnel_zdr_failure(net, unblock):
    """Reproduces OpenRouter HTTP 404 when Guardrails remove OpenAI due to ZDR and fallback is disallowed."""
    err_body = json.dumps({
        "error": {
            "message": "No endpoints found for openai/gpt-4.1-nano. Every candidate endpoint was removed during routing: Filter by Regional Surcharge removed azure/swedencentral; Filter by Guardrails removed openai (ZDR violation (account settings)); Filter by Fallback removed azure.",
            "code": 404,
            "metadata": {
                "routing_funnel": [
                    {"step": "Initial Endpoints", "endpoint_count": 3},
                    {"step": "Filter by Regional Surcharge", "endpoint_count": 2},
                    {"step": "Filter by Guardrails", "endpoint_count": 1},
                ]
            }
        }
    }).encode()
    net.script = [http_err(404, err_body)]
    with pytest.raises(ProviderError) as exc_info:
        chat_adapter().classify(Task.INTENT, STATE, INTENT_PROMPT, live_authorized=True)
    assert exc_info.value.status == 404
    assert exc_info.value.retryable is False
    assert "No endpoints found" in str(exc_info.value)
    assert "ZDR violation" in str(exc_info.value)


def test_runner_aborts_on_consecutive_routing_404_errors(net, unblock, env):
    """Verifies that consecutive 404 routing errors abort the run at max_consecutive_errors (3),
    reporting UNAVAILABLE cost basis without recording ungrounded ledger spend."""
    err_body = json.dumps({
        "error": {
            "message": "No endpoints found for openai/gpt-4.1-nano. Every candidate endpoint was removed during routing.",
            "code": 404,
        }
    }).encode()
    net.script = [http_err(404, err_body), http_err(404, err_body), http_err(404, err_body)]
    runner = ClassificationRunner({NANO_MODEL_ID: chat_adapter()}, env[2], env[1], max_consecutive_errors=3)
    five_cases = cases(5)
    summary = runner.run(task=Task.INTENT, cases=five_cases, model_id=NANO_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="abort_run")

    assert summary.aborted_reason == "3 consecutive errors"
    assert summary.n_error == 3
    assert len(summary.records) == 3
    for rec in summary.records:
        assert rec.status == "ERROR"
        assert rec.cost_basis == "UNAVAILABLE"
        assert rec.usage_source == "UNAVAILABLE"
        assert rec.prompt_tokens is None
        assert rec.completion_tokens is None
        assert rec.recorded_cost_usd is None
    # Ledger must remain clean (0 spend recorded for failed requests)
    status = env[1].get_status()
    assert status.cumulative_spend_usd == Decimal("0")
    assert status.unresolved_spend_count == 0
