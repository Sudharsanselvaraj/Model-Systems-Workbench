"""Trace Explorer and exports: synthetic data only, no network, sources stay read-only."""

import csv
import hashlib
import io
import json
import zipfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from msu_llm_lab.classification.contracts import CONTRACT_VERSION, IntentLabel
from msu_llm_lab.classification.records import PredictionRecord, Task
from msu_llm_lab.core.config import settings
from msu_llm_lab.models.registry import Protocol, roster_status
from msu_llm_lab.observability import export as X
from msu_llm_lab.observability.events import EventStore, RequestEvent
from msu_llm_lab.observability.labels import ApprovedLabels, load_approved_labels
from msu_llm_lab.tui.app import WorkbenchTUI
from obs_helpers import finished, intent_pred, label_row, lead_pred, make_prep, started, ts, write_labels, write_preds

NOLABELS = ApprovedLabels()
STAMP = "2026-10-15T12:30:00+00:00"
EMAIL, PHONE, KEY = "student.test@example.invalid", "0123456789012", "sk-or-v1-abcdefghijklmnopqrstuvwxyz0123"


def error_rec(case, run="run1", model="jev-1.14", retry=("TIMEOUT", "TIMEOUT", "TIMEOUT"), msg=None):
    return PredictionRecord(
        record_id=f"{run}:{case}:INTENT:{model}", run_id=run, case_id=case, task=Task.INTENT, model_key=model, protocol=Protocol.DECISIONS_API,
        requested_model_id=model, prompt_version="sha256:test", dataset_version="test-ds", status="ERROR", attempts=len(retry), retry_errors=list(retry),
        error_type="ProviderError", error_message=msg or "request failed", created_at="2026-10-15T12:00:05+00:00")


def sample_preds():
    ok = [intent_pred(f"C{i}", "QUERY", top=0.9, latency_ms=100.0 + i, prompt_tokens=50, completion_tokens=3, usage_source="PROVIDER_REPORTED",
                      actual_cost_usd=Decimal("0.00010"), cost_basis="PROVIDER_REPORTED", native_cost=Decimal("0.0001"), native_cost_unit="credits(unverified)")
          for i in range(3)]
    other = [intent_pred(f"C{i}", "GREETING", model="gpt-4.1-nano", run="run2") for i in range(2)]
    bad = [intent_pred("C9", "QUERY", status="INVALID_OUTPUT"), error_rec("C8")]
    lead = [lead_pred("L1", yes_prob=0.8), lead_pred("L2", need_cta=False, model="gpt-4.1-nano", run="lrun2")]
    return ok + other + bad + lead


def build(preds=None, events=(), labels=NOLABELS, frozen=None, sel=X.ExportSelection(), **kw):
    return X.build_trace_rows(preds if preds is not None else sample_preds(), list(events), labels, frozen or {}, sel, **kw)


# ------------------------------------------------------------------ filters and scopes
def test_scopes_and_filters():
    assert X.ExportSelection().scope == "ALL_MODELS_ALL_RUNS"
    assert X.ExportSelection(models=("a",)).scope == "ONE_MODEL" and X.ExportSelection(runs=("r",)).scope == "ONE_RUN"
    assert X.ExportSelection(task="LEAD").scope == "ONE_TASK" and X.ExportSelection(models=("a",), task="LEAD").scope == "FILTERED_SELECTION"
    everything = build()
    assert len(everything.rows) == len(sample_preds())
    one_model = build(sel=X.ExportSelection(models=("gpt-4.1-nano",)))
    assert {r["identity"]["model_key"] for r in one_model.rows} == {"gpt-4.1-nano"} and one_model.skipped["NOT_SELECTED"] == len(sample_preds()) - len(one_model.rows)
    assert {r["identity"]["run_id"] for r in build(sel=X.ExportSelection(runs=("run2",))).rows} == {"run2"}
    assert {r["identity"]["task"] for r in build(sel=X.ExportSelection(task="LEAD")).rows} == {"LEAD"}
    failed = build(sel=X.ExportSelection(statuses=("ERROR", "INVALID_OUTPUT")))
    assert {r["request"]["status"] for r in failed.rows} == {"ERROR", "INVALID_OUTPUT"}
    assert [r["identity"]["case_id"] for r in build(sel=X.ExportSelection(case_ids=("C1",))).rows] == ["C1", "C1"]


# ------------------------------------------------------------------ completeness, nulls, provenance fields
def test_every_trace_field_is_present_and_nulls_are_preserved():
    row = next(r for r in build().rows if r["identity"]["case_id"] == "C0" and r["identity"]["model_key"] == "jev-1.14" and r["identity"]["task"] == "INTENT")
    assert row["schema_version"] == X.SCHEMA_VERSION
    assert row["versions"] == {"dataset_version": "test-ds", "prompt_fingerprint": "sha256:test", "input_contract_version": CONTRACT_VERSION,
                               "parity_status": "PRODUCTION_PARITY_NOT_ESTABLISHED"}
    assert row["confidence"]["selected_label_probability"] == pytest.approx(0.9) and sum(row["confidence"]["label_probabilities"].values()) == pytest.approx(1.0)
    assert row["confidence"]["native_confidence"] is None                         # absent stays null, never filled
    assert row["request"]["model_latency_ms"] is None and row["request"]["model_latency_source"] == "UNAVAILABLE"
    assert row["cost"] == {**row["cost"], "actual_cost_usd": "0.00010", "estimated_cost_usd": None, "native_cost": "0.0001", "native_cost_unit": "credits(unverified)",
                           "cost_basis": "PROVIDER_REPORTED", "unresolved_spend": False}
    assert row["markers"]["simulated"] is False and row["markers"]["provenance"] == "LIVE" and "model_latency_ms" in row["markers"]["unavailable_telemetry"]
    assert row["ground_truth"] == {"status": "PENDING", "label": None, "correct": None}


def test_failed_and_invalid_requests_are_exported_with_their_markers():
    rows = {(r["identity"]["case_id"]): r for r in build().rows if r["identity"]["task"] == "INTENT" and r["identity"]["model_key"] == "jev-1.14"}
    err, inv = rows["C8"], rows["C9"]
    assert err["markers"]["failed_call"] and err["request"]["error_category"] == "TIMEOUT" and err["request"]["attempts"] == 3
    assert err["request"]["retry_causes"] == ["TIMEOUT"] * 3 and err["output"]["predicted_label"] is None and err["usage"]["prompt_tokens"] is None
    assert inv["markers"]["invalid_output"] and inv["request"]["error_category"] == "INVALID_OUTPUT" and not inv["markers"]["failed_call"]


def test_events_add_timestamps_retries_and_event_only_failures():
    evs = [started("run1:C0:INTENT:jev-1.14", 5), finished("run1:C0:INTENT:jev-1.14", 6),
           started("run3:Z1:INTENT:gpt-4o-mini", 7, model="gpt-4o-mini", run="run3", case="Z1"),
           finished("run3:Z1:INTENT:gpt-4o-mini", 8, model="gpt-4o-mini", run="run3", case="Z1", status="ERROR", http_status=429, task="INTENT", actual=None, basis=None),
           RequestEvent(kind="REQUEST_RETRY", request_id="run1:C0:INTENT:jev-1.14", run_id="run1", case_id="C0", model_key="jev-1.14", ts=ts(5.5), retry_cause="HTTP_503")]
    ts_ = build(events=evs)
    c0 = next(r for r in ts_.rows if r["request"]["request_id"] == "run1:C0:INTENT:jev-1.14")
    assert c0["request"]["request_started_at"] == ts(5) and "HTTP_503" in c0["request"]["retry_causes"]
    only = next(r for r in ts_.rows if r["identity"]["run_id"] == "run3")
    assert only["markers"]["source"] == "EVENT_ONLY_NO_PREDICTION_RECORD" and only["markers"]["failed_call"] and only["request"]["error_category"] == "RATE_LIMIT"
    assert only["output"]["predicted_label"] is None


def test_lead_rows_carry_the_frozen_threshold_and_decision_source():
    frozen = {"jev-1.14": SimpleNamespace(threshold=0.55)}
    lead = [r for r in build(frozen=frozen).rows if r["identity"]["task"] == "LEAD"]
    jev = next(r for r in lead if r["identity"]["model_key"] == "jev-1.14")
    assert jev["confidence"]["yes_probability"] == 0.8 and jev["confidence"]["frozen_threshold"] == 0.55
    assert jev["output"]["decision"] is True and jev["output"]["decision_source"] == "FROZEN_DEV_THRESHOLD"
    jev0 = next(r for r in build().rows if r["identity"]["task"] == "LEAD" and r["identity"]["model_key"] == "jev-1.14")
    assert jev0["output"]["decision"] is None and jev0["output"]["decision_source"] == "THRESHOLD_NOT_FROZEN"
    chat = next(r for r in lead if r["identity"]["model_key"] == "gpt-4.1-nano")
    assert chat["output"]["decision"] is False and chat["output"]["decision_source"] == "MODEL_OUTPUT"


def test_ground_truth_and_correctness_only_after_approval(tmp_path):
    prep = make_prep(tmp_path / "p")
    write_labels(prep, [label_row("C0", "INTENT", "QUERY"), label_row("C1", "INTENT", "GREETING"), label_row("C2", "INTENT", "QUERY", status="PENDING_REVIEW")])
    labels = load_approved_labels(prep)
    rows = {r["identity"]["case_id"]: r for r in build(labels=labels).rows if r["identity"]["model_key"] == "jev-1.14" and r["identity"]["task"] == "INTENT"}
    assert rows["C0"]["ground_truth"] == {"status": "APPROVED", "label": "QUERY", "correct": True}
    assert rows["C1"]["ground_truth"] == {"status": "APPROVED", "label": "GREETING", "correct": False}
    assert rows["C2"]["ground_truth"] == {"status": "NOT_LABELLED", "label": None, "correct": None}


# ------------------------------------------------------------------ simulated exclusion
def test_simulated_events_never_enter_real_only_exports():
    sim = finished("simreq", 3, model="m-sim", run="sim", status="ERROR", provenance="SIMULATED", task="INTENT")
    ts_ = build(events=[sim, started("x", 1, provenance="SIMULATED")])
    assert ts_.quarantined_simulated == 2 and ts_.skipped["SIMULATED_EXCLUDED"] == 2
    assert all(r["identity"]["run_id"] != "sim" and r["markers"]["simulated"] is False for r in ts_.rows)


# ------------------------------------------------------------------ privacy
def test_redaction_secrets_and_user4_never_reach_an_export(tmp_path):
    msg = f"call failed for {EMAIL} phone {PHONE} Authorization: Bearer abcdefgh12345678 key {KEY}"
    rec = error_rec("C8", msg=msg)
    rec = rec.model_copy(update={"requested_route": {"order": ["p"], "user4": "0123", "Authorization": "Bearer abcdefgh12345678", "api_key": KEY}})
    ts_ = build(preds=[rec])
    assert ts_.records_redacted == 1
    for fmt in ("jsonl", "csv", "zip"):
        res = X.export_traces(ts_, X.ExportSelection(), roster_status(), NOLABELS, fmt, tmp_path / fmt, exported_at=STAMP)
        blob = res.path.read_bytes()
        if fmt == "zip":
            blob = b"".join(zipfile.ZipFile(res.path).read(n) for n in zipfile.ZipFile(res.path).namelist())
        for secret in (EMAIL.encode(), PHONE.encode(), KEY.encode(), b"abcdefgh12345678", b'"user4":', b'"0123"'):
            assert secret not in blob, (fmt, secret)
        assert res.manifest["privacy"]["user4_excluded"] and res.manifest["privacy"]["secrets_scrubbed"] and not res.manifest["privacy"]["this_export_transmitted_externally"]


def test_raw_text_is_withheld_unless_the_data_owner_authorizes(tmp_path):
    rec = intent_pred("C1", "QUERY", raw_output="QUERY")
    default = build(preds=[rec])
    assert default.rows[0]["output"]["raw_output"] is None and default.raw_text_withheld == 1 and default.rows[0]["output"]["raw_output_present_in_store"]
    auth = X.RawTextAuthorization("data owner", "pilot review", "2026-10-15")
    allowed = build(preds=[rec], authorization=auth)
    assert allowed.rows[0]["output"]["raw_output"] == "QUERY"
    res = X.export_traces(allowed, X.ExportSelection(), [], NOLABELS, "jsonl", tmp_path, authorization=auth, exported_at=STAMP)
    assert res.manifest["privacy"]["raw_text_included"] and res.manifest["privacy"]["raw_text_authorization"]["owner"] == "data owner"
    with pytest.raises(X.ExportError, match="authorization is incomplete"):
        X.export_traces(allowed, X.ExportSelection(), [], NOLABELS, "jsonl", tmp_path, authorization=X.RawTextAuthorization("", "x", "y"))


# ------------------------------------------------------------------ files
def test_jsonl_and_csv_match_the_rows_and_preserve_empty_values(tmp_path):
    ts_ = build()
    j = X.export_traces(ts_, X.ExportSelection(), roster_status(), NOLABELS, "jsonl", tmp_path, exported_at=STAMP)
    lines = j.path.read_text().splitlines()
    assert len(lines) == j.n_exported == len(ts_.rows) and json.loads(lines[0])["schema_version"] == X.SCHEMA_VERSION
    assert j.path.with_name(j.path.name + ".manifest.json").exists()
    c = X.export_traces(ts_, X.ExportSelection(), roster_status(), NOLABELS, "csv", tmp_path, exported_at=STAMP)
    recs = list(csv.DictReader(io.StringIO(c.path.read_text())))
    assert len(recs) == len(ts_.rows) and list(recs[0].keys()) == X.CSV_COLUMNS
    err = next(r for r in recs if r["case_id"] == "C8")
    assert err["prompt_tokens"] == "" and err["actual_cost_usd"] == "" and err["failed_call"] == "True" and err["simulated"] == "False"
    ok = next(r for r in recs if r["case_id"] == "C0" and r["model_key"] == "jev-1.14")
    assert float(ok["p_QUERY"]) == pytest.approx(0.9) and ok["ground_truth_status"] == "PENDING" and ok["correct"] == ""


def test_zip_contents_manifest_and_participant_coverage(tmp_path):
    ts_ = build()
    res = X.export_traces(ts_, X.ExportSelection(), roster_status(), NOLABELS, "zip", tmp_path, exported_at=STAMP)
    zf = zipfile.ZipFile(res.path)
    names = set(zf.namelist())
    assert {"manifest.json", "all_records.jsonl", "all_records.csv", "consolidated_comparison.csv", "participants.csv",
            "models/jev-1.14.jsonl", "models/jev-1.14.csv", "models/gpt-4.1-nano.jsonl", "models/gpt-4.1-nano.csv"} <= names
    man = json.loads(zf.read("manifest.json"))
    assert man["schema_version"] == X.SCHEMA_VERSION and man["exported_at"] == STAMP and man["format"] == "zip"
    assert man["run_ids"] == ["lrun", "lrun2", "run1", "run2"] and man["model_ids"] == ["gpt-4.1-nano", "jev-1.14"] and man["tasks"] == ["INTENT", "LEAD"]
    assert man["dataset_versions"] == ["test-ds"] and man["prompt_versions"] == ["sha256:test"] and man["input_contract_versions"] == [CONTRACT_VERSION]
    assert man["parity_status"] == "PRODUCTION_PARITY_NOT_ESTABLISHED" and man["telemetry_provenance"]["live_only"] and not man["telemetry_provenance"]["simulated_records_included"]
    assert "models" not in man["provider_routes"] and set(man["provider_routes"]) == {"gpt-4.1-nano", "jev-1.14"}
    assert any("GROUND TRUTH PENDING" in w for w in man["warnings"]) and any("PRODUCTION_PARITY_NOT_ESTABLISHED" in w for w in man["warnings"])
    for name, digest in man["files"].items():
        assert hashlib.sha256(zf.read(name)).hexdigest() == digest
    assert man["counts"]["exported"] == len(ts_.rows) and man["counts"]["failed_calls"] == 1
    # every roster participant is listed; models that never ran have no trace rows
    parts = {p["key"]: p for p in man["participants"]}
    assert len([p for p in man["participants"] if p["readiness"] != "NOT_ON_ROSTER"]) == len(roster_status()) == 11
    assert parts["gpt-4o-mini"]["execution_state"] == "NOT_EXECUTED" and parts["gpt-4o-mini"]["n_trace_records"] == 0
    assert parts["gpt-4.1-nano"]["execution_state"] in ("EXECUTED", "EXECUTED_WITH_FAILURES") and parts["jev-1.14"]["execution_state"] == "EXECUTED_WITH_FAILURES"
    assert all(r["identity"]["model_key"] != "gpt-4o-mini" for r in ts_.rows)
    comp = list(csv.DictReader(io.StringIO(zf.read("consolidated_comparison.csv").decode())))
    row = next(r for r in comp if r["case_id"] == "C0" and r["task"] == "INTENT")
    assert row["pred::jev-1.14"] == "QUERY" and row["pred::gpt-4.1-nano"] == "GREETING" and row["models_agree"] == "False"


def test_empty_export_is_valid_and_creates_no_placeholder_rows(tmp_path):
    ts_ = build(preds=[], events=[])
    res = X.export_traces(ts_, X.ExportSelection(), roster_status(), NOLABELS, "zip", tmp_path, exported_at=STAMP)
    assert res.n_exported == 0
    zf = zipfile.ZipFile(res.path)
    man = json.loads(zf.read("manifest.json"))
    assert zf.read("all_records.jsonl") == b"" and any("EMPTY EXPORT" in w for w in man["warnings"])
    assert {p["execution_state"] for p in man["participants"]} <= {"NOT_EXECUTED", "NOT_EXECUTED_BLOCKED"} and len(man["participants"]) == len(roster_status())
    assert not [n for n in zf.namelist() if n.startswith("models/")]
    none_match = build(sel=X.ExportSelection(runs=("does-not-exist",)))
    assert none_match.rows == [] and none_match.skipped["NOT_SELECTED"] == len(sample_preds())


def test_exports_are_reproducible_never_overwrite_and_leave_sources_untouched(tmp_path):
    preds_dir = tmp_path / "preds"
    write_preds(preds_dir, "r", sample_preds())
    events = EventStore(tmp_path / "ev.jsonl")
    events.append(started("run1:C0:INTENT:jev-1.14", 1))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    sel = X.ExportSelection(models=("jev-1.14",))
    a = X.export_traces(build(sel=sel, events=events.read_all()), sel, roster_status(), NOLABELS, "zip", tmp_path / "out", exported_at=STAMP)
    b = X.export_traces(build(sel=sel, events=events.read_all()), sel, roster_status(), NOLABELS, "zip", tmp_path / "out", exported_at=STAMP)
    assert a.path != b.path and b.path.name.endswith("-r2.zip") and a.path.read_bytes() == b.path.read_bytes()
    c = X.export_traces(build(), X.ExportSelection(), roster_status(), NOLABELS, "zip", tmp_path / "out", exported_at=STAMP)
    assert c.path.name != a.path.name
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file() and "out" not in p.parts}
    assert after == before


def test_unknown_format_is_refused(tmp_path):
    with pytest.raises(X.ExportError, match="unknown format"):
        X.export_traces(build(), X.ExportSelection(), [], NOLABELS, "xlsx", tmp_path)


# ------------------------------------------------------------------ TUI
def make_tui():
    return WorkbenchTUI(console=Console(file=io.StringIO(), width=207, height=55, force_terminal=False))


def test_explorer_screen_filters_preview_and_export(monkeypatch):
    write_preds(settings.reports_dir / "classification", "r", sample_preds())
    tui = make_tui()
    tui.execute_command(":export")
    assert tui.section == "explorer"
    vm = tui.get_view()
    assert len(vm.rows) == len(sample_preds()) and any("NOT_EXECUTED" in n for n in vm.footnotes)
    out = io.StringIO()
    Console(file=out, width=207, height=55).print(tui.build_layout())
    for needle in ("TRACE EXPLORER & EXPORT", "Destination", "GROUND TRUTH PENDING"):
        assert needle in out.getvalue()
    tui.handle_key("t")
    assert tui.ex.task == "INTENT" and all(r.cells[3] == "INTENT" for r in tui.get_view().rows)
    tui.handle_key("m")
    assert tui.ex.model in ("gpt-4.1-nano", "jev-1.14")
    tui.handle_key("c")
    assert (tui.ex.model, tui.ex.task, tui.ex.run, tui.ex.status) == (None, None, None, None)
    tui.handle_key("s")
    assert tui.ex.status == "OK"
    tui.handle_key("f")
    tui.handle_key("f")
    assert tui.ex.fmt == "zip"
    tui.handle_key("x")
    r = tui.ex.last_result
    assert r is not None and r.path.exists() and r.path.parent == settings.reports_dir / "exports" and tui.ex.last_error is None
    assert str(r.path) in tui.feedback_message and r.n_exported == len(tui.get_view().rows)
    assert "EXPORT OK" in (lambda o: (Console(file=o, width=207, height=55).print(tui.build_layout()), o.getvalue())[1])(io.StringIO())


def test_explorer_reports_export_failure_and_empty_state(tmp_path):
    tui = make_tui()
    tui.goto("explorer")
    assert not tui.get_view().rows and "NO TRACE RECORDS" in tui.get_view().empty
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    tui.ex.dest = blocker / "exports"
    tui.handle_key("x")
    assert tui.ex.last_result is None and tui.ex.last_error and "FAILED" in tui.feedback_message


def test_explorer_never_sends_a_request(monkeypatch):
    import socket
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("trace export must not use the network")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    write_preds(settings.reports_dir / "classification", "r", sample_preds())
    tui = make_tui()
    tui.goto("explorer")
    for k in "mtusfcx":
        tui.handle_key(k)
    tui.handle_key("ENTER")


def test_simulated_event_in_the_log_is_excluded_from_the_explorer_export():
    p = settings.data_dir / "telemetry" / "request_events.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(finished("sim1", 1, model="m-sim", run="simrun", status="ERROR", provenance="SIMULATED", task="INTENT").to_json_line() + "\n")
    tui = make_tui()
    tui.goto("explorer")
    assert tui.get_view().rows == [] and tui._trace_set().quarantined_simulated == 1


# ------------------------------------------------------------------ CLI
def test_export_traces_cli_writes_a_zip_under_reports_exports():
    from msu_llm_lab.cli import app
    write_preds(settings.reports_dir / "classification", "r", sample_preds())
    res = CliRunner().invoke(app, ["export-traces", "--format", "zip", "--task", "INTENT", "--model", "jev-1.14"])
    assert res.exit_code == 0, res.output
    assert "records exported 5" in res.output and "GROUND TRUTH PENDING" in res.output and str(settings.reports_dir) in res.output.replace("\n", "")
    zips = list((settings.reports_dir / "exports").glob("*.zip"))
    assert len(zips) == 1
    assert CliRunner().invoke(app, ["export-traces", "--task", "NOPE"]).exit_code == 1


# ------------------------------------------------------------------ cleaner viewer, honest data
def test_explorer_table_shows_dashes_not_repeated_unavailable_and_never_shows_errors_as_predictions():
    write_preds(settings.reports_dir / "classification", "r", sample_preds())
    tui = make_tui()
    tui.goto("explorer")
    vm = tui.get_view()
    out = io.StringIO()
    Console(file=out, width=207, height=60).print(tui.build_layout())
    text = out.getvalue()
    assert text.count("UNAVAILABLE") <= 1                        # at most the legacy header word; no per-cell noise
    err = next(r for r in vm.rows if r.cells[1] == "C8" and r.cells[4] == "ERROR")
    assert err.cells[5] == "—" and err.cells[7] == "—" and err.cells[8] == "—"          # no prediction, tokens or cost for a failed call
    inv = next(r for r in vm.rows if r.cells[1] == "C9")
    assert inv.cells[4] == "INVALID_OUTPUT" and inv.cells[5] == "—"                    # an invalid output is not a prediction either
    ok = next(r for r in vm.rows if r.cells[1] == "C0" and r.cells[2] != "—" and r.cells[3] == "INTENT" and r.cells[4] == "OK")
    assert ok.cells[5] == "QUERY"
    joined = "\n".join(ok.detail)
    assert "model latency" not in joined.split("not measured:")[0] and "not measured:" in joined and "not reported by provider" in joined
    assert "privacy: free text redacted by default" in joined and "prompt sha256:test" in joined and "data prep test-ds" in joined


def test_machine_readable_records_keep_nulls_and_reasons_even_when_the_viewer_hides_them():
    row = next(r for r in build().rows if r["identity"]["case_id"] == "C8")
    assert row["usage"]["prompt_tokens"] is None and row["request"]["model_latency_ms"] is None and "tokens" in row["markers"]["unavailable_telemetry"]


def test_synthetic_fixture_traces_are_labelled_and_never_look_like_real_data():
    smoke = intent_pred("S1", "QUERY", run="smoke_abc-jev-1.14-INTENT", dataset_version="smoke-synthetic-v1")
    real = intent_pred("R1", "QUERY", run="bench-0123-jev-1.14-INTENT", dataset_version="prep-v2")
    other = intent_pred("O1", "QUERY", run="adhoc", dataset_version="x")
    kinds = {r["identity"]["case_id"]: r["markers"]["data_kind"] for r in build(preds=[smoke, real, other]).rows}
    assert kinds == {"S1": "SYNTHETIC_FIXTURE", "R1": "REAL_DATA_BENCHMARK", "O1": "UNCLASSIFIED"}
    only_real = build(preds=[smoke, real, other], sel=X.ExportSelection(data_kinds=("REAL_DATA_BENCHMARK",)))
    assert [r["identity"]["case_id"] for r in only_real.rows] == ["R1"]
    write_preds(settings.reports_dir / "classification", "r", [smoke, real])
    tui = make_tui()
    tui.goto("explorer")
    vm = tui.get_view()
    assert {r.cells[9].split(" ")[0] for r in vm.rows} == {"SYNTHETIC", "REAL-DATA"}
    tui.handle_key("d")
    assert tui.ex.kind in ("REAL_DATA_BENCHMARK", "SYNTHETIC_FIXTURE") and len(tui.get_view().rows) == 1


def test_error_class_filter_and_per_run_summary_with_planned_attempted_failed_invalid_and_unattempted():
    ts_ = build(sel=X.ExportSelection(error_categories=("TIMEOUT",)))
    assert [r["identity"]["case_id"] for r in ts_.rows] == ["C8"]
    plans = {"bench-abc": {"participants": [{"key": "jev-1.14", "per_task": {"INTENT": {"requests_full": 10}}}]}}
    recs = [intent_pred(f"C{i}", "QUERY", run="bench-abc-jev-1.14-INTENT") for i in range(3)] + [intent_pred("C9", "QUERY", run="bench-abc-jev-1.14-INTENT", status="INVALID_OUTPUT")]
    summ = X.run_summaries(build(preds=recs).rows, plans)
    assert summ == [{"run_id": "bench-abc-jev-1.14-INTENT", "model_key": "jev-1.14", "task": "INTENT", "data_kind": "REAL_DATA_BENCHMARK", "attempted": 4, "succeeded": 3,
                     "failed": 0, "invalid": 1, "planned_full_dataset": 10, "unattempted": 6}]
    none = X.run_summaries(build().rows, {})
    assert all(s["planned_full_dataset"] is None and s["unattempted"] is None for s in none)       # unknown stays unknown


def test_unusable_outputs_are_not_votes_in_the_comparison_file():
    a = [intent_pred("C1", "QUERY", model="m-a"), intent_pred("C2", "QUERY", model="m-a")]
    b = [intent_pred("C1", "QUERY", model="m-b"), intent_pred("C2", "QUERY", model="m-b", status="INVALID_OUTPUT")]
    cols, comp = X.comparison_rows(build(preds=a + b).rows)
    c2 = next(r for r in comp if r["case_id"] == "C2")
    assert c2["pred::m-b"] is None and c2["n_models_answered"] == 1 and c2["n_models_unusable"] == 1 and c2["models_agree"] is None


def test_explorer_filter_keys_cover_error_class_and_data_kind_and_clear():
    write_preds(settings.reports_dir / "classification", "r", sample_preds())
    tui = make_tui()
    tui.goto("explorer")
    tui.handle_key("e")
    assert tui.ex.error_class in ("TIMEOUT", "INVALID_OUTPUT")
    assert all(r.cells[4] != "OK" for r in tui.get_view().rows)
    tui.handle_key("c")
    assert tui.ex.error_class is None and len(tui.get_view().rows) == len(sample_preds())
