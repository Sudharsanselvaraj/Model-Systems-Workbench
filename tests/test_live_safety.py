"""Offline tests for live-run safety (C1), honest usage/latency (C2) and safe retries (C3).

Everything here is synthetic. urllib.request.urlopen is replaced by fakes and sockets are blocked,
so a stray real request fails the test instead of leaving the machine.
"""

import email.message
import io
import json
import socket
import urllib.error
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.live_guard import (
    LiveAuthorization, confirm_live_run, expected_phrase, require_live_authorization,
)
from msu_llm_lab.dataset.manager import DatasetManager
from msu_llm_lab.evaluation.runner import BenchmarkRunner
from msu_llm_lab.providers import openrouter as or_mod
from msu_llm_lab.providers.base import (
    BaseProvider, LiveNotAuthorizedError, ProviderError, ProviderRequest, ProviderResponse,
    RateLimitError, UsageUnavailableError, require_complete_usage, scrub_secrets,
)
from msu_llm_lab.providers.openrouter import MAX_ATTEMPTS_CAP, OpenRouterAdapter
from msu_llm_lab.rag.runner import RAGBenchmarkRunner
from msu_llm_lab.schemas.evaluation import (
    DataClassification, DatasetSplit, EvaluationCase, VerificationRecord, VerificationStatus,
)
from msu_llm_lab.storage.db import DatabaseManager

MODEL = "google/gemini-3.5-flash-lite"
FAKE_KEY = "test-not-a-real-key"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def _deny(*a, **k):
        raise AssertionError("real network access attempted")
    monkeypatch.setattr(socket.socket, "connect", _deny)
    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(socket, "getaddrinfo", _deny)


# ------------------------------------------------------------------ helpers
class FakeHTTPResponse:
    def __init__(self, payload):
        self._body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body=b"{}", retry_after=None):
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://example.invalid", code, "x", hdrs, io.BytesIO(body))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds

    sleeps: list


def make_adapter(script, monkeypatch, step=0.2, **kw):
    """Adapter whose urlopen replays `script` (responses or exceptions); time advances only via the fake clock."""
    clock = Clock()
    clock.sleeps = []
    calls = []
    items = list(script)

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        clock.t += step
        item = items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeHTTPResponse(item)

    monkeypatch.setattr(or_mod.urllib.request, "urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(api_key=FAKE_KEY, sleep=clock.sleep, clock=clock, **kw)
    return adapter, clock, calls


def live_request(**kw):
    base = dict(model_id=MODEL, messages=[{"role": "user", "content": "synthetic question"}], live_authorized=True)
    base.update(kw)
    return ProviderRequest(**base)


OK_BODY = {"choices": [{"message": {"content": "synthetic answer"}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}


# ------------------------------------------------------------------ C1: scrubber
def test_credentials_are_redacted():
    out = scrub_secrets({
        "api_key": "x", "Authorization": "Bearer abcdefgh12345678", "access_token": "t", "refresh_token": "t",
        "password": "p", "client_secret": "s", "x-api-key": "k", "cookie": "c", "nested": {"private_key": "k"},
    })
    assert all(v == "[REDACTED_SECRET]" for k, v in out.items() if k != "nested")
    assert out["nested"]["private_key"] == "[REDACTED_SECRET]"


def test_credentials_in_strings_are_redacted():
    text = "Authorization: Bearer abcdefgh12345678 and sk-or-v1-abcdefghijklmnop1234 and sk-abcdefghijklmnop1234"
    out = scrub_secrets(text)
    assert "abcdefgh12345678" not in out and "abcdefghijklmnop1234" not in out


def test_usage_fields_survive_scrubbing():
    usage = {
        "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19, "input_tokens": 12, "output_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 3}, "completion_tokens_details": {"reasoning_tokens": 2},
    }
    assert scrub_secrets({"usage": usage})["usage"] == usage


def test_not_every_token_key_is_exempt():
    out = scrub_secrets({"token_count": 5, "access_token": 12345, "prompt_tokens": "not-a-number", "completion_tokens": True,
                         "tokenizer_secret": "s"})
    assert set(out.values()) == {"[REDACTED_SECRET]"}


def test_secret_inside_usage_details_still_redacted():
    out = scrub_secrets({"prompt_tokens_details": {"cached_tokens": 3, "api_key": "k"}})
    assert out["prompt_tokens_details"] == {"cached_tokens": 3, "api_key": "[REDACTED_SECRET]"}


# ------------------------------------------------------------------ C1: confirmation
def boom_prompt(msg):
    raise AssertionError("prompt must not be reached")


def test_non_interactive_never_confirms_even_with_correct_phrase():
    with pytest.raises(LiveNotAuthorizedError):
        confirm_live_run([MODEL], is_interactive=lambda: False, prompt=lambda m: expected_phrase([MODEL]), echo=lambda l: None)


def test_wrong_or_empty_phrase_refused():
    for answer in ["", "yes", "RUN LIVE", "run live " + MODEL, "RUN LIVE other/model"]:
        with pytest.raises(LiveNotAuthorizedError):
            confirm_live_run([MODEL], is_interactive=lambda: True, prompt=lambda m, a=answer: a, echo=lambda l: None)


def test_exact_phrase_in_interactive_session_authorizes_only_listed_models():
    auth = confirm_live_run([MODEL, "b/m"], is_interactive=lambda: True,
                            prompt=lambda m: expected_phrase([MODEL, "b/m"]), echo=lambda l: None)
    assert auth.covers(MODEL) and auth.covers("b/m") and not auth.covers("c/m")


def test_no_models_refused():
    with pytest.raises(LiveNotAuthorizedError):
        confirm_live_run([], is_interactive=lambda: True, prompt=boom_prompt, echo=lambda l: None)


def test_require_live_authorization():
    require_live_authorization(LiveAuthorization(models=(MODEL,)), MODEL)
    for bad in (None, LiveAuthorization(models=("x/y",))):
        with pytest.raises(LiveNotAuthorizedError):
            require_live_authorization(bad, MODEL)


def test_adapter_refuses_unauthorized_live_request_before_any_http(monkeypatch):
    monkeypatch.setattr(or_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("urlopen called"))
    adapter = OpenRouterAdapter(api_key=FAKE_KEY)
    with pytest.raises(LiveNotAuthorizedError):
        adapter.generate(live_request(live_authorized=False))


def test_adapter_refuses_live_without_key_or_with_unknown_model(monkeypatch):
    monkeypatch.setattr(or_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("urlopen called"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenRouterAdapter(api_key=None).generate(live_request())
    with pytest.raises(ProviderError):
        OpenRouterAdapter(api_key=FAKE_KEY).generate(live_request(model_id="unregistered/model"))


def _dataset(tmp_path):
    cases = [EvaluationCase(id="syn-1", query="synthetic query", expected_answer="synthetic answer",
                            classification=DataClassification.SYNTHETIC, split=DatasetSplit.TEST,
                            verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="synthetic"))]
    p = tmp_path / "ds.jsonl"
    DatasetManager.save_jsonl(cases, p)
    return p


class CountingProvider(BaseProvider):
    def __init__(self, response=None):
        super().__init__(api_key=None)
        self.calls = 0
        self.response = response

    def _get_pricing_for_model(self, model_id):
        return Decimal("0.1"), Decimal("0.1")

    def generate(self, request):
        self.calls += 1
        return self.response


@pytest.mark.parametrize("runner_cls", [BenchmarkRunner, RAGBenchmarkRunner])
def test_runners_refuse_live_without_authorization(tmp_path, runner_cls):
    prov = CountingProvider()
    extra = {"retriever": __import__("live_fakes").ExpectedContextRetriever()} if runner_cls is RAGBenchmarkRunner else {}
    runner = runner_cls(db_mgr=DatabaseManager(db_path=tmp_path / "t.db"), provider=prov, **extra)
    with pytest.raises(LiveNotAuthorizedError):
        runner.run(model_id=MODEL, dataset_path=_dataset(tmp_path), split=DatasetSplit.TEST)
    assert prov.calls == 0


@pytest.mark.parametrize("cmd", [["benchmark"], ["benchmark-rag"]])
def test_cli_live_without_confirmation_makes_no_request(tmp_path, monkeypatch, cmd):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(or_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("urlopen called"))
    monkeypatch.setattr(OpenRouterAdapter, "generate", lambda self, req: pytest.fail("provider generate called"))
    ds = _dataset(tmp_path)
    args = cmd + ["--model", MODEL, "--dataset", str(ds), "--split", "TEST", "--live"]
    if cmd == ["benchmark-rag"]:  # needs audited KB records for its real BM25 retriever
        (tmp_path / "data" / "audit").mkdir(parents=True)
        (tmp_path / "data" / "audit" / "parsed_records_general_kb.jsonl").write_text(
            '{"record_id": "REC-1", "source_file": "x.md", "question_clean": "fees?", "answer_clean": "RM 1", "parsing_status": "VALID", "programme_name": "P"}\n')
    # Piped stdin containing the exact phrase must NOT bypass the interactive requirement.
    result = CliRunner().invoke(app, args, input=expected_phrase([MODEL]) + "\n")
    assert result.exit_code == 1
    assert "refused" in result.output.lower()
    assert not (tmp_path / "data" / "workbench.db").exists()


# ------------------------------------------------------------------ C2: usage and latency
def test_provider_reported_usage_and_estimated_catalog_cost(monkeypatch):
    adapter, clock, _ = make_adapter([OK_BODY], monkeypatch)
    r = adapter.generate(live_request())
    assert (r.prompt_tokens, r.completion_tokens, r.usage_source) == (1000, 500, "PROVIDER_REPORTED")
    assert r.cost_usd == Decimal("0.001550") and r.cost_basis == "ESTIMATED_FROM_CATALOG"
    assert r.attempts == 1 and r.retry_errors == []
    assert r.raw_response["usage"]["prompt_tokens"] == 1000  # scrubber keeps usage


def test_provider_cost_is_kept_as_native_credits_and_not_booked_as_usd_until_the_unit_is_verified(monkeypatch):
    body = {**OK_BODY, "usage": {**OK_BODY["usage"], "cost": 0.0042}}
    r = make_adapter([body], monkeypatch)[0].generate(live_request())
    assert r.native_cost == Decimal("0.0042") and r.native_cost_unit == "OPENROUTER_CREDITS_USD_RATE_UNVERIFIED" and r.actual_cost_usd is None
    assert r.cost_usd == Decimal("0.001550") and r.cost_basis == "ESTIMATED_FROM_CATALOG"  # accounting falls back to the catalog estimate


def test_provider_cost_is_booked_as_usd_only_when_the_unit_is_verified(monkeypatch):
    from msu_llm_lab.models import registry as reg
    monkeypatch.setattr(reg, "OPENROUTER_COST_UNIT", reg.fact("USD (1 credit = $1)", reg.FactStatus.VERIFIED_PRIMARY, ["https://openrouter.ai/docs"], "test"))
    body = {**OK_BODY, "usage": {**OK_BODY["usage"], "cost": 0.0042}}
    r = make_adapter([body], monkeypatch)[0].generate(live_request())
    assert r.actual_cost_usd == Decimal("0.0042") and r.cost_usd == Decimal("0.0042") and r.cost_basis == "PROVIDER_REPORTED" and r.native_cost == Decimal("0.0042")


def test_missing_tokens_with_only_a_native_cost_cannot_be_booked(monkeypatch):
    body = {"choices": OK_BODY["choices"], "usage": {"cost": 0.0042}}
    r = make_adapter([body], monkeypatch)[0].generate(live_request())
    assert r.native_cost == Decimal("0.0042") and r.actual_cost_usd is None and r.estimated_cost_usd is None and r.cost_basis == "UNAVAILABLE"


def test_missing_usage_is_unavailable_not_estimated(monkeypatch):
    body = {"choices": [{"message": {"content": "a fairly long synthetic answer with many words in it"}}]}
    r = make_adapter([body], monkeypatch)[0].generate(live_request())
    assert r.prompt_tokens is None and r.completion_tokens is None
    assert r.usage_source == "UNAVAILABLE" and r.cost_basis == "UNAVAILABLE" and r.cost_usd == Decimal("0")


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 10},
    {"prompt_tokens": "10", "completion_tokens": "5"},
    {"prompt_tokens": True, "completion_tokens": -1},
])
def test_partial_or_invalid_usage_is_not_trusted(monkeypatch, usage):
    r = make_adapter([{**OK_BODY, "usage": usage}], monkeypatch)[0].generate(live_request())
    assert r.usage_source in ("PARTIAL", "UNAVAILABLE") and r.cost_basis == "UNAVAILABLE"
    if usage == {"prompt_tokens": 10}:
        assert r.prompt_tokens == 10 and r.completion_tokens is None and r.usage_source == "PARTIAL"
    else:
        assert r.prompt_tokens is None


def test_model_latency_unavailable_unless_provider_reports_it(monkeypatch):
    r = make_adapter([OK_BODY], monkeypatch)[0].generate(live_request())
    assert r.model_latency_ms is None and r.model_latency_source == "UNAVAILABLE"
    assert r.latency_source == "MEASURED_CLIENT" and r.latency_ms == pytest.approx(200.0) and r.attempt_latency_ms == pytest.approx(200.0)


def test_provider_reported_model_latency_is_captured_with_provenance(monkeypatch):
    body = {**OK_BODY, "usage": {**OK_BODY["usage"], "generation_ms": 85}}
    adapter, _, _ = make_adapter([body], monkeypatch, provider_latency_field=("usage", "generation_ms"))
    r = adapter.generate(live_request())
    assert r.model_latency_ms == 85.0 and r.model_latency_source == "PROVIDER_REPORTED"
    assert r.latency_ms != r.model_latency_ms  # end-to-end is measured separately, never copied


def test_end_to_end_includes_retry_wait_but_attempt_latency_does_not(monkeypatch):
    adapter, clock, _ = make_adapter([http_error(503), OK_BODY], monkeypatch, step=0.2)
    r = adapter.generate(live_request())
    assert r.latency_ms == pytest.approx(1400.0) and r.attempt_latency_ms == pytest.approx(200.0)
    assert r.attempts == 2 and r.retry_errors == ["HTTP_503"] and r.model_latency_ms is None


def test_require_complete_usage():
    ok = ProviderResponse(output_text="x", prompt_tokens=1, completion_tokens=1, usage_source="PROVIDER_REPORTED",
                          latency_ms=1.0, cost_basis="ESTIMATED_FROM_CATALOG", model_id="m")
    require_complete_usage(ok)
    for bad in (ok.model_copy(update={"usage_source": "UNAVAILABLE"}), ok.model_copy(update={"cost_basis": "UNAVAILABLE"}),
                ok.model_copy(update={"usage_source": "PARTIAL"})):
        with pytest.raises(UsageUnavailableError):
            require_complete_usage(bad)


def test_runner_aborts_when_live_usage_missing_and_records_unresolved_spend(tmp_path):
    missing = ProviderResponse(output_text="x", latency_ms=5.0, model_id=MODEL)
    prov = CountingProvider(missing)
    db = DatabaseManager(db_path=tmp_path / "t.db")
    runner = BenchmarkRunner(db_mgr=db, provider=prov)
    with pytest.raises(UsageUnavailableError):
        runner.run(model_id=MODEL, dataset_path=_dataset(tmp_path), split=DatasetSplit.TEST,
                   live_authorization=LiveAuthorization(models=(MODEL,)))
    assert prov.calls == 1
    st = runner.budget_service.get_status()
    assert st.cumulative_spend_usd == 0 and st.unresolved_spend_count == 1      # no amount invented, but the gap is on the ledger
    assert runner.budget_service.get_status().recent_transactions[0]["is_unresolved"] == 1


# ------------------------------------------------------------------ C3: retries
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_statuses_retry_then_succeed(monkeypatch, status):
    adapter, clock, calls = make_adapter([http_error(status), OK_BODY], monkeypatch)
    r = adapter.generate(live_request())
    assert len(calls) == 2 and r.attempts == 2 and r.retry_errors == [f"HTTP_{status}"]
    assert clock.sleeps == [1.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_permanent_errors_are_not_retried(monkeypatch, status):
    adapter, clock, calls = make_adapter([http_error(status, b'{"error":"bad"}'), OK_BODY], monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    e = ei.value
    assert len(calls) == 1 and clock.sleeps == []
    assert (e.attempts, e.status, e.retryable, e.errors) == (1, status, False, [f"HTTP_{status}"])


def test_exhausted_retries_never_sleep_after_final_attempt(monkeypatch):
    adapter, clock, calls = make_adapter([http_error(503)] * 3, monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    e = ei.value
    assert len(calls) == 3 and len(clock.sleeps) == 2 and clock.sleeps == [1.0, 2.0]
    assert (e.attempts, e.errors, e.status, e.retryable) == (3, ["HTTP_503"] * 3, 503, True)
    assert not isinstance(e, RateLimitError)


def test_exhausted_429_raises_rate_limit_error(monkeypatch):
    adapter, clock, _ = make_adapter([http_error(429)] * 2, monkeypatch, max_attempts=2)
    with pytest.raises(RateLimitError) as ei:
        adapter.generate(live_request())
    assert ei.value.attempts == 2 and clock.sleeps == [1.0]


def test_single_attempt_never_sleeps(monkeypatch):
    adapter, clock, calls = make_adapter([http_error(503)], monkeypatch, max_attempts=1)
    with pytest.raises(ProviderError):
        adapter.generate(live_request())
    assert len(calls) == 1 and clock.sleeps == []


def test_attempts_are_bounded(monkeypatch):
    assert make_adapter([], monkeypatch, max_attempts=99)[0].max_attempts == MAX_ATTEMPTS_CAP
    assert make_adapter([], monkeypatch, max_attempts=0)[0].max_attempts == 1
    assert make_adapter([], monkeypatch, max_retries=2)[0].max_attempts == 2  # legacy name = total attempts


def test_retry_after_is_honoured_but_capped(monkeypatch):
    adapter, clock, _ = make_adapter([http_error(429, retry_after=3), http_error(429, retry_after=500), OK_BODY], monkeypatch)
    adapter.generate(live_request())
    assert clock.sleeps == [3.0, 8.0]


def test_network_errors_are_classified_and_retried(monkeypatch):
    script = [urllib.error.URLError(TimeoutError()), ConnectionResetError(), OK_BODY]
    adapter, clock, _ = make_adapter(script, monkeypatch)
    r = adapter.generate(live_request())
    assert r.attempts == 3 and r.retry_errors == ["TIMEOUT", "NETWORK_ERROR"]


def test_malformed_success_body_is_not_retried(monkeypatch):
    adapter, clock, calls = make_adapter([b"not json at all", OK_BODY], monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    assert len(calls) == 1 and clock.sleeps == [] and ei.value.errors == ["INVALID_RESPONSE"] and ei.value.retryable is False


def test_error_body_in_200_is_permanent(monkeypatch):
    adapter, clock, calls = make_adapter([{"error": {"message": "nope"}}, OK_BODY], monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    assert len(calls) == 1 and ei.value.retryable is False


def test_unexpected_programming_errors_are_not_retried(monkeypatch):
    adapter, clock, calls = make_adapter([RuntimeError("bug"), OK_BODY], monkeypatch)
    with pytest.raises(RuntimeError):
        adapter.generate(live_request())
    assert len(calls) == 1 and clock.sleeps == []


def test_error_bodies_are_scrubbed_in_exceptions(monkeypatch):
    adapter, _, _ = make_adapter([http_error(401, b'{"detail":"bad Bearer abcdefgh12345678"}')], monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    assert "abcdefgh12345678" not in str(ei.value)


# ------------------------------------------------------------------ Phase 4: cost provenance and retry codes
from msu_llm_lab.providers.base import require_recordable_cost  # noqa: E402


def test_actual_and_estimated_costs_are_kept_separate(monkeypatch):
    body = {**OK_BODY, "usage": {**OK_BODY["usage"], "cost": 0.0042}}
    r = make_adapter([body], monkeypatch)[0].generate(live_request())
    assert r.native_cost == Decimal("0.0042") and r.actual_cost_usd is None and r.estimated_cost_usd == Decimal("0.001550")
    assert r.cost_usd == Decimal("0.001550") and r.cost_basis == "ESTIMATED_FROM_CATALOG"  # native, actual and estimated stay separate


def test_estimate_requires_both_reported_token_counts(monkeypatch):
    r = make_adapter([{**OK_BODY, "usage": {"prompt_tokens": 1000}}], monkeypatch)[0].generate(live_request())
    assert r.estimated_cost_usd is None and r.actual_cost_usd is None and r.cost_basis == "UNAVAILABLE"
    r2 = make_adapter([{"choices": OK_BODY["choices"]}], monkeypatch)[0].generate(live_request())
    assert r2.estimated_cost_usd is None and r2.actual_cost_usd is None


def test_estimate_only_when_provider_reports_no_cost(monkeypatch):
    r = make_adapter([OK_BODY], monkeypatch)[0].generate(live_request())
    assert r.actual_cost_usd is None and r.estimated_cost_usd == Decimal("0.001550")
    assert r.cost_basis == "ESTIMATED_FROM_CATALOG" and r.cost_usd == r.estimated_cost_usd


def test_recordable_cost_rules():
    base = ProviderResponse(output_text="x", latency_ms=1.0, model_id="m")
    for basis in ("PROVIDER_REPORTED", "ESTIMATED_FROM_CATALOG"):
        require_recordable_cost(base.model_copy(update={"cost_basis": basis}))
    with pytest.raises(UsageUnavailableError):
        require_recordable_cost(base)  # UNAVAILABLE
    # cost reported but tokens missing: recordable here, still rejected by the token-persisting legacy check
    cost_only = base.model_copy(update={"cost_basis": "PROVIDER_REPORTED", "usage_source": "UNAVAILABLE"})
    require_recordable_cost(cost_only)
    with pytest.raises(UsageUnavailableError):
        require_complete_usage(cost_only)


@pytest.mark.parametrize("status", [524, 529])
def test_524_and_529_are_retried_within_bounds(monkeypatch, status):
    adapter, clock, calls = make_adapter([http_error(status)] * 3, monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    assert len(calls) == 3 and clock.sleeps == [1.0, 2.0] and ei.value.errors == [f"HTTP_{status}"] * 3
    adapter2, _, calls2 = make_adapter([http_error(status), OK_BODY], monkeypatch)
    assert adapter2.generate(live_request()).retry_errors == [f"HTTP_{status}"] and len(calls2) == 2


def test_402_and_403_remain_permanent(monkeypatch):
    for status in (402, 403):
        adapter, clock, calls = make_adapter([http_error(status), OK_BODY], monkeypatch)
        with pytest.raises(ProviderError) as ei:
            adapter.generate(live_request())
        assert len(calls) == 1 and ei.value.retryable is False


# ------------------------------------------------------------------ Phase 4b: 402, typed errors, registry live guard, route
def err_body(error_type=None, limit_source=None, code=402):
    meta = {k: v for k, v in (("error_type", error_type), ("limit_source", limit_source)) if v}
    return json.dumps({"error": {"code": code, "message": "m", "metadata": meta}}).encode()


def test_402_retried_only_for_in_flight_budget_with_valid_retry_after(monkeypatch):
    adapter, clock, calls = make_adapter(
        [http_error(402, err_body("budget", "openrouter_in_flight_budget"), retry_after=2), OK_BODY], monkeypatch)
    r = adapter.generate(live_request())
    assert len(calls) == 2 and clock.sleeps == [2.0] and r.retry_errors == ["HTTP_402/budget"]


@pytest.mark.parametrize("body,retry_after", [
    (err_body(None, "openrouter_in_flight_budget"), None),            # no Retry-After
    (err_body(None, "openrouter_in_flight_budget"), "soon"),          # invalid Retry-After
    (err_body(None, "openrouter_in_flight_budget"), "-5"),            # negative
    (err_body(None, "openrouter_in_flight_budget"), "nan"),           # not finite
    (err_body(None, "openrouter_in_flight_budget"), "Wed, 21 Oct 2026 07:28:00 GMT"),  # HTTP-date not trusted
    (err_body("budget", "openrouter_key_limit"), "2"),                # other limit source
    (b"{}", "2"),                                                     # plain credits error
    (b"not json", "2"),
])
def test_other_402_responses_are_never_retried(monkeypatch, body, retry_after):
    adapter, clock, calls = make_adapter([http_error(402, body, retry_after=retry_after), OK_BODY], monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    assert len(calls) == 1 and clock.sleeps == [] and ei.value.retryable is False and ei.value.status == 402


def test_exhausted_in_flight_402_retries_stay_bounded_and_never_sleep_last(monkeypatch):
    errs = [http_error(402, err_body(None, "openrouter_in_flight_budget"), retry_after=1) for _ in range(3)]
    adapter, clock, calls = make_adapter(errs, monkeypatch)
    with pytest.raises(ProviderError) as ei:
        adapter.generate(live_request())
    assert len(calls) == 3 and len(clock.sleeps) == 2 and ei.value.status == 402 and ei.value.attempts == 3


def test_typed_error_type_is_recorded_when_present_and_unsafe_values_are_dropped(monkeypatch):
    adapter, _, _ = make_adapter([http_error(503, err_body("provider_overloaded", code=503)), OK_BODY], monkeypatch)
    assert adapter.generate(live_request()).retry_errors == ["HTTP_503/provider_overloaded"]
    adapter2, _, _ = make_adapter([http_error(503, err_body("bad type with spaces & Bearer abcdefgh12345678", code=503)), OK_BODY], monkeypatch)
    assert adapter2.generate(live_request()).retry_errors == ["HTTP_503"]


def test_retry_after_non_numeric_is_ignored_for_backoff(monkeypatch):
    adapter, clock, _ = make_adapter([http_error(429, retry_after="tomorrow"), OK_BODY], monkeypatch)
    adapter.generate(live_request())
    assert clock.sleeps == [1.0]


def test_registry_blocks_live_for_unverified_route_before_any_http(monkeypatch):
    from registry_helpers import block_route
    block_route(monkeypatch, "gpt-4.1-nano")
    block_route(monkeypatch, "jev-1.14")
    monkeypatch.setattr(or_mod.urllib.request, "urlopen", lambda *a, **k: pytest.fail("urlopen called"))
    adapter = OpenRouterAdapter(api_key=FAKE_KEY)
    for model in ("openai/gpt-4.1-nano", "typesafe/jev-1.14"):  # both routes blocked
        with pytest.raises(ProviderError, match="registry blockers"):
            adapter.generate(live_request(model_id=model))


def test_provider_preferences_are_sent_only_when_set_and_resolved_values_recorded(monkeypatch):
    adapter, _, calls = make_adapter([{**OK_BODY, "model": "gemini-resolved-snapshot", "provider": "SomeProvider"}, OK_BODY], monkeypatch)
    prefs = {"order": ["openai"], "allow_fallbacks": False}
    r = adapter.generate(live_request(provider_preferences=prefs))
    assert json.loads(calls[0].data)["provider"] == prefs
    assert (r.resolved_model_id, r.serving_provider) == ("gemini-resolved-snapshot", "SomeProvider")
    r2 = adapter.generate(live_request())
    assert "provider" not in json.loads(calls[1].data) and r2.resolved_model_id is None and r2.serving_provider is None


def test_estimate_withheld_for_registry_models_without_verified_route_price(monkeypatch):
    from registry_helpers import block_route
    block_route(monkeypatch, "gpt-4.1-nano")
    monkeypatch.setattr(or_mod.registry, "synthetic_live_blockers", lambda model_id, stage='pilot': [])  # isolate the estimate rule
    adapter, _, _ = make_adapter([OK_BODY], monkeypatch)
    r = adapter.generate(live_request(model_id="openai/gpt-4.1-nano"))
    assert r.estimated_cost_usd is None and r.cost_basis == "UNAVAILABLE" and r.prompt_tokens == 1000


def test_registry_prices_support_preflight_budgeting_for_nano():
    assert OpenRouterAdapter(api_key=FAKE_KEY)._get_pricing_for_model("openai/gpt-4.1-nano") == (Decimal("0.10"), Decimal("0.40"))
