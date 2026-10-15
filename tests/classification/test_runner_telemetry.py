"""The runner appends real request lifecycle events (start, success/failure, retries) and nothing else."""

import json

from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.classification.runner import ClassificationRunner
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.observability.aggregate import aggregate
from msu_llm_lab.observability.events import EventStore, RequestEvent
from msu_llm_lab.storage.db import DatabaseManager
from test_runner_adapters import (  # noqa: F401  (fixtures and helpers shared with the adapter tests)
    AUTH, INTENT_PROMPT, JEV_MODEL_ID, cases, env, http_err, jev_adapter, jev_body, net, unblock,
)


def run_one(env, events, n=1, **kw):
    _, budget, store = env
    runner = ClassificationRunner({JEV_MODEL_ID: jev_adapter()}, store, budget, telemetry=events, **kw)
    return runner, runner.run(task=Task.INTENT, cases=cases(n), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=AUTH, run_id="t1")


def test_a_successful_request_emits_start_then_success_with_measured_fields(env, net, unblock, tmp_path):
    ev = EventStore(tmp_path / "ev.jsonl")
    net.script = [jev_body()]
    run_one(env, ev)
    events = ev.read_all()
    assert [e.kind for e in events] == ["REQUEST_STARTED", "REQUEST_SUCCEEDED"]
    done = events[1]
    assert done.status == "OK" and done.latency_ms is not None and done.prompt_tokens == 1200 and done.completion_tokens == 3
    assert done.model_latency_ms is None                        # Jev reported none: not substituted
    assert done.provenance == "LIVE" and done.request_id == events[0].request_id
    assert aggregate(events, 2e9).overall.n_ok == 1


def test_events_carry_no_prompt_or_output_text(env, net, unblock, tmp_path):
    ev = EventStore(tmp_path / "ev.jsonl")
    net.script = [jev_body()]
    run_one(env, ev)
    raw = ev.path.read_text()
    assert "synthetic question" not in raw and "raw_output" not in raw and "messages" not in raw
    assert set(json.loads(raw.splitlines()[0])) <= set(RequestEvent.model_fields)


def test_timeouts_emit_retry_events_then_a_failed_event_marked_timed_out(env, net, unblock, tmp_path):
    ev = EventStore(tmp_path / "ev.jsonl")
    net.script = [TimeoutError("t")] * 3
    run_one(env, ev)
    kinds = [e.kind for e in ev.read_all()]
    assert kinds == ["REQUEST_STARTED", "REQUEST_RETRY", "REQUEST_RETRY", "REQUEST_FAILED"]
    failed = ev.read_all()[-1]
    assert failed.timed_out and failed.status == "ERROR" and failed.attempts == 3
    g = aggregate(ev.read_all(), 2e9).overall
    assert (g.n_timeouts, g.n_retries, g.n_error) == (1, 2, 1)


def test_http_failures_record_the_status(env, net, unblock, tmp_path):
    ev = EventStore(tmp_path / "ev.jsonl")
    net.script = [http_err(401)]
    try:
        run_one(env, ev)
    except Exception:
        pass
    failed = [e for e in ev.read_all() if e.kind == "REQUEST_FAILED"][0]
    assert failed.http_status == 401 and not failed.timed_out


def test_retried_then_successful_request_reports_its_retries(env, net, unblock, tmp_path):
    ev = EventStore(tmp_path / "ev.jsonl")
    net.script = [http_err(503), jev_body()]
    run_one(env, ev)
    events = ev.read_all()
    assert [e.kind for e in events] == ["REQUEST_STARTED", "REQUEST_RETRY", "REQUEST_SUCCEEDED"]
    assert events[1].retry_cause == "HTTP_503" and events[2].attempts == 2


def test_a_telemetry_write_failure_is_counted_and_never_changes_the_run(env, net, unblock, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    ev = EventStore(blocker / "ev.jsonl")                        # parent is a file: every append raises OSError
    net.script = [jev_body()]
    runner, summary = run_one(env, ev)
    assert summary.n_ok == 1 and runner.telemetry_errors >= 2
    assert env[1].get_status().cumulative_spend_usd > 0           # spend was still booked


def test_no_event_is_written_when_the_run_is_refused_before_any_request(env, net, tmp_path):
    ev = EventStore(tmp_path / "ev.jsonl")
    runner = ClassificationRunner({JEV_MODEL_ID: jev_adapter()}, env[2], env[1], telemetry=ev)
    try:
        runner.run(task=Task.INTENT, cases=cases(1), model_id=JEV_MODEL_ID, prompt_text=INTENT_PROMPT, authorization=None)
    except Exception:
        pass
    assert not ev.path.exists() and net.requests == []


def test_a_real_runner_flow_exports_with_event_timestamps_and_provenance(env, net, unblock, tmp_path):
    from msu_llm_lab.observability.export import ExportSelection, build_trace_rows
    from msu_llm_lab.observability.labels import ApprovedLabels

    ev = EventStore(tmp_path / "ev.jsonl")
    net.script = [jev_body(), http_err(401)]
    try:
        run_one(env, ev, n=2)
    except Exception:
        pass
    ts = build_trace_rows(env[2].load(), ev.read_all(), ApprovedLabels(), {}, ExportSelection())
    ok, failed = ts.rows
    assert ok["markers"]["source"] == "PREDICTION_RECORD" and ok["request"]["request_started_at"] is not None
    assert ok["usage"]["prompt_tokens"] == 1200 and ok["cost"]["cost_basis"] in ("PROVIDER_REPORTED", "ESTIMATED_FROM_CATALOG", "UNAVAILABLE")
    assert ok["confidence"]["selected_label_probability"] is not None and ok["confidence"]["native_confidence"] is not None
    assert failed["markers"]["failed_call"] and failed["request"]["error_category"] == "AUTH"
