"""Real-data benchmark: plan, authorization, checkpointed execution, quarantine and budget gates. Synthetic data and a fake transport only."""

import json
import urllib.request
from decimal import Decimal
from pathlib import Path

import pytest

from msu_llm_lab.classification import benchmark as bm
from msu_llm_lab.classification import dataprep as dp
from msu_llm_lab.classification import realdata
from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.classification.runner import BatchRejectedError, ClassificationRunner
from msu_llm_lab.classification.smoke import build_smoke_adapters
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.models import registry
from msu_llm_lab.observability.events import EventStore
from msu_llm_lab.providers.base import LiveNotAuthorizedError
from msu_llm_lab.storage.db import DatabaseManager
from synthetic_fixtures import SYN_INTENT_PROMPT, SYN_LEAD_PROMPT
from test_dataprep import write_csv
from test_runner_adapters import http_err, jev_body, noul_body, chat_body, _Resp

MODELS = ["jev-1.14", "gpt-4.1-nano"]
PROMPTS = {Task.INTENT: SYN_INTENT_PROMPT, Task.LEAD: SYN_LEAD_PROMPT}
SECRET = "private-test-secret-0123456789"


@pytest.fixture
def prep(tmp_path, monkeypatch):
    monkeypatch.setenv(dp.SALT_ENV, SECRET)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    src = tmp_path / "syn.csv"
    write_csv(src)
    dp.prepare(src, tmp_path / "prep-t")
    return tmp_path / "prep-t"


@pytest.fixture
def budget(tmp_path):
    return BudgetService(db=DatabaseManager(db_path=tmp_path / "b.db"))


class SmartNet:
    """Answers each request according to its protocol and task; can fail a protocol on demand. Records every request."""

    def __init__(self):
        self.requests, self.fail_jev, self.fail_chat, self.fail_models = [], None, None, {}

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        body = json.loads(req.data)
        if req.full_url.endswith("/decisions"):
            if self.fail_jev:
                raise self.fail_jev()
            return _Resp(json.dumps(noul_body(0.8) if "lead" in body["questions"] else jev_body()).encode())
        if self.fail_chat:
            raise self.fail_chat()
        if body["model"] in self.fail_models:
            raise self.fail_models[body["model"]]()
        is_intent = any(SYN_INTENT_PROMPT in m["content"] for m in body["messages"])
        return _Resp(json.dumps(chat_body("QUERY" if is_intent else '{"need_cta": false}')).encode())


@pytest.fixture
def net(monkeypatch):
    n = SmartNet()
    monkeypatch.setattr(urllib.request, "urlopen", n)
    return n


def make_plan(prep, budget, policy="2h", models=MODELS, block=2, cap="0.05"):
    loaded = bm.load_prep_cases(prep, policy)
    return loaded, bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=block, hard_cap_usd=Decimal(cap))


def issue(plan, models=None, phrase=None):
    models = models or [p["model_id"] for p in plan["participants"]]
    return realdata.confirm_real_data(
        dataset_version=plan["dataset"]["version"], session_policy=plan["dataset"]["session_policy"], models=models,
        routes={p["model_id"]: p["route"]["pinned_tag"] or "none" for p in plan["participants"]}, tasks=["INTENT", "LEAD"],
        plan_fingerprint=plan["fingerprint"], cost_ceiling_usd=plan["limits"]["hard_cap_usd"],
        provider_policy_status={p["model_id"]: p["data_handling"]["data_policy"] for p in plan["participants"]}, recipients=plan["recipients"],
        operator="Test Operator", is_interactive=lambda: True, prompt=lambda m: phrase or realdata.expected_phrase(plan["dataset"]["version"], plan["fingerprint"]),
        echo=lambda s: None)


def run(plan, loaded, budget, tmp_path, auth=None, live=None, **kw):
    models = [p["model_id"] for p in plan["participants"]]
    return bm.execute(plan, loaded, PROMPTS, auth or issue(plan), live or LiveAuthorization(models=tuple(models)),
                      build_smoke_adapters([p["key"] for p in plan["participants"]], bm.STAGE), budget, root=tmp_path / "bench",
                      reports_dir=tmp_path / "reports", events=EventStore(tmp_path / "ev.jsonl"), **kw)


# ------------------------------------------------------------------ plan
def test_plan_states_counts_hashes_routes_caps_and_a_deterministic_fingerprint(prep, budget):
    loaded, plan = make_plan(prep, budget)
    assert plan["dataset"]["version"] == "prep-t" and plan["dataset"]["session_policy"] == "2h" and plan["blockers"] == []
    assert plan["dataset"]["eligible"] == {"INTENT": len(loaded.intent), "LEAD": len(loaded.lead)} and plan["dataset"]["labels"].startswith("ALL PENDING")
    assert plan["prompts"]["INTENT"]["sha256"] and plan["prompts"]["LEAD"]["version"].startswith("sha256:")
    assert plan["limits"]["per_case_usd"] == "0.02" and plan["limits"]["pilot_usd"] == "2.00" and plan["limits"]["total_usd"] == "10.00"
    assert plan["limits"]["max_attempts"] == 3 and plan["limits"]["timeout_seconds"] == 30.0
    by = {p["key"]: p for p in plan["participants"]}
    assert by["gpt-4.1-nano"]["route"]["pinned_tag"] == "azure" and by["gpt-4.1-nano"]["route"]["fallbacks"] == "disabled"
    assert by["jev-1.14"]["data_handling"]["data_policy"] == "UNKNOWN" and by["jev-1.14"]["readiness"]["schema_facts"]["response_shape"] == "LIVE_OBSERVED"
    assert plan["totals"]["requests_full_dataset"] == sum(v["requests_full"] for p in plan["participants"] for v in p["per_task"].values())
    assert all(plan["recipients"][m] for m in plan["recipients"])
    again = make_plan(prep, budget)[1]
    assert again["fingerprint"] == plan["fingerprint"] and again["bench_id"] == plan["bench_id"]
    assert make_plan(prep, budget, block=3)[1]["fingerprint"] != plan["fingerprint"]
    assert make_plan(prep, budget, policy="30m")[1]["fingerprint"] != plan["fingerprint"]
    assert make_plan(prep, budget, models=["gpt-4.1-nano"])[1]["fingerprint"] != plan["fingerprint"]
    changed = bm.build_plan(loaded, {**PROMPTS, Task.INTENT: PROMPTS[Task.INTENT] + " "}, budget, model_keys=MODELS, block_size=2, hard_cap_usd=Decimal("0.05"))
    assert changed["fingerprint"] != plan["fingerprint"]


def test_full_roster_keeps_every_participant_with_its_task_scope(prep, budget):
    _, plan = make_plan(prep, budget, models=None)
    assert [p["key"] for p in plan["participants"]] == list(registry.REGISTRY) and len(plan["participants"]) == len(registry.REGISTRY) == 11
    by = {p["key"]: p["tasks"] for p in plan["participants"]}
    assert by["baseline-intent-gemini-3.5-flash-lite"] == ["INTENT"] and by["baseline-lead-gpt-4.1-mini"] == ["LEAD"] and by["claude-sonnet-5.5"] == ["INTENT", "LEAD"]
    assert "benchmark route only" in "\n".join(bm.render_plan(plan))


def test_planning_sends_nothing_and_contains_no_student_text(prep, budget, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("planning must not use the network")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    loaded, plan = make_plan(prep, budget)
    dump = json.dumps(plan) + "\n".join(bm.render_plan(plan))
    sample = [t.text for c in loaded.intent for t in ([c.current_message] + list(c.history))][:20]
    assert sample and not any(len(t) > 15 and t in dump for t in sample)


def test_a_public_salt_or_unhardened_dataset_blocks_the_plan(tmp_path, budget, monkeypatch):
    monkeypatch.delenv(dp.SALT_ENV, raising=False)
    src = tmp_path / "syn.csv"
    write_csv(src)
    dp.prepare(src, tmp_path / "prep-public")
    _, plan = make_plan(tmp_path / "prep-public", budget)
    assert any("public legacy salt" in b for b in plan["blockers"])


def test_session_policy_must_be_chosen_explicitly(prep):
    with pytest.raises(bm.PlanError, match="session policy"):
        bm.load_prep_cases(prep, "")


def test_blocks_hold_identical_cases_for_every_participant_and_exceeding_caps_gives_a_checkpointed_schedule(prep, budget):
    loaded, plan = make_plan(prep, budget, block=2, cap="0.0001")
    assert plan["totals"]["full_dataset_fits_caps"] is False and plan["totals"]["guaranteed_blocks"] < plan["ordering"]["n_blocks"]
    assert sum(b["requests"] for b in plan["blocks"]) == plan["totals"]["requests_full_dataset"]
    assert plan["limits"]["pilot_usd"] == "2.00"                    # budgets are never raised to make the plan fit
    order = [c.case_id for c in bm.ordered(loaded.intent)]
    assert order == [c.case_id for c in bm.ordered(list(reversed(loaded.intent)))]   # deterministic regardless of input order


# ------------------------------------------------------------------ authorization
def test_real_data_confirmation_needs_the_exact_phrase_an_interactive_terminal_and_a_named_operator(prep, budget):
    _, plan = make_plan(prep, budget)
    with pytest.raises(LiveNotAuthorizedError, match="did not match"):
        issue(plan, phrase="yes")
    with pytest.raises(LiveNotAuthorizedError, match="interactive"):
        realdata.confirm_real_data(dataset_version="d", session_policy="2h", models=["m"], routes={}, tasks=["INTENT"], plan_fingerprint="f", cost_ceiling_usd="1",
                                   provider_policy_status={}, recipients={}, operator="x", is_interactive=lambda: False, prompt=lambda m: "", echo=lambda s: None)
    with pytest.raises(LiveNotAuthorizedError, match="operator"):
        realdata.confirm_real_data(dataset_version="d", session_policy="2h", models=["m"], routes={}, tasks=["INTENT"], plan_fingerprint="f", cost_ceiling_usd="1",
                                   provider_policy_status={}, recipients={}, operator=" ", is_interactive=lambda: True, prompt=lambda m: "", echo=lambda s: None)
    auth = issue(plan)
    assert auth.plan_fingerprint == plan["fingerprint"] and auth.real_admissions_derived and auth.data_classification == "CONFIDENTIAL"
    assert set(auth.provider_policy_status.values()) == {"UNKNOWN"} and "does not verify" in auth.policy_note


def test_the_authorization_phrase_is_bound_to_the_dataset_version_and_plan():
    assert realdata.expected_phrase("prep-v2", "abc") == "AUTHORIZE REAL DATA prep-v2 PLAN abc"


def test_a_hand_built_authorization_is_never_accepted(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    forged = realdata.RealDataAuthorization(dataset_version="prep-t", session_policy="2h", models=tuple(p["model_id"] for p in plan["participants"]),
                                            routes={}, tasks=("INTENT", "LEAD"), plan_fingerprint=plan["fingerprint"], cost_ceiling_usd="1", operator="x",
                                            statement="x", provider_policy_status={})
    with pytest.raises(LiveNotAuthorizedError, match="issued"):
        run(plan, loaded, budget, tmp_path, auth=forged)
    assert net.requests == []


def test_authorization_for_another_plan_dataset_or_model_set_is_refused(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    other_plan = {**plan, "fingerprint": "deadbeef0000"}
    with pytest.raises(LiveNotAuthorizedError, match="different plan"):
        run(other_plan, loaded, budget, tmp_path, auth=issue(plan))
    partial = issue(plan, models=[plan["participants"][0]["model_id"]])
    with pytest.raises(LiveNotAuthorizedError, match="cover every participant"):
        run(plan, loaded, budget, tmp_path, auth=partial)
    assert net.requests == []


def test_runner_refuses_confidential_cases_without_a_covering_issued_authorization(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    adapters = build_smoke_adapters(MODELS, bm.STAGE)
    runner = ClassificationRunner(adapters, PredictionStore(tmp_path / "p.jsonl"), budget, stage="pilot", telemetry=EventStore(tmp_path / "e.jsonl"))
    mid = "openai/gpt-4.1-nano"
    live = LiveAuthorization(models=(mid,))
    with pytest.raises(BatchRejectedError, match="Batch rejected"):
        runner.run(task=Task.INTENT, cases=loaded.intent[:2], model_id=mid, prompt_text=SYN_INTENT_PROMPT, authorization=live, dataset_version="prep-t")
    auth = issue(plan)
    with pytest.raises(BatchRejectedError):      # wrong dataset version
        runner.run(task=Task.INTENT, cases=loaded.intent[:2], model_id=mid, prompt_text=SYN_INTENT_PROMPT, authorization=live, dataset_version="prep-OTHER", real_data=auth)
    assert net.requests == []


def test_excluded_records_can_never_be_dispatched(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    case = loaded.intent[0].model_copy(update={"exclusion_reason": __import__("msu_llm_lab.classification.contracts", fromlist=["ExclusionReason"]).ExclusionReason.INSUFFICIENT_CONTEXT})
    adapters = build_smoke_adapters(MODELS, bm.STAGE)
    runner = ClassificationRunner(adapters, PredictionStore(tmp_path / "p.jsonl"), budget, stage="pilot", telemetry=EventStore(tmp_path / "e.jsonl"))
    mid = "openai/gpt-4.1-nano"
    with pytest.raises(BatchRejectedError) as e:
        runner.run(task=Task.INTENT, cases=[case], model_id=mid, prompt_text=SYN_INTENT_PROMPT, authorization=LiveAuthorization(models=(mid,)), dataset_version="prep-t", real_data=issue(plan))
    assert "excluded record" in str(e.value.reasons)
    assert net.requests == []


# ------------------------------------------------------------------ execution
def test_execution_records_every_outcome_checkpoints_and_resumes_without_duplicates(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    first = run(plan, loaded, budget, tmp_path, max_blocks=1)
    assert first.blocks_completed == 1 and not first.quarantined
    sent_one = len(net.requests)
    assert sent_one == plan["blocks"][0]["requests"]
    recs = [r for p in (tmp_path / "reports" / "classification").glob("*.jsonl") for r in PredictionStore(p).load()]
    assert len(recs) == sent_one and all(r.status == "OK" and r.dataset_version == "prep-t" for r in recs)
    resumed = run(plan, loaded, budget, tmp_path)           # same plan: continues from the checkpoint
    assert resumed.blocks_completed == plan["ordering"]["n_blocks"] and resumed.halted is None
    assert len(net.requests) == plan["totals"]["requests_full_dataset"]       # every planned request exactly once
    again = run(plan, loaded, budget, tmp_path)
    assert len(net.requests) == plan["totals"]["requests_full_dataset"] and again.blocks_completed == plan["ordering"]["n_blocks"]
    ids = [(r.run_id, r.case_id) for p in (tmp_path / "reports" / "classification").glob("*.jsonl") for r in PredictionStore(p).load()]
    assert len(ids) == len(set(ids)) == plan["totals"]["requests_full_dataset"]
    events = EventStore(tmp_path / "ev.jsonl").read_all()
    assert sum(e.kind == "REQUEST_STARTED" for e in events) == len(net.requests)


def test_same_cases_go_to_every_participant(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    run(plan, loaded, budget, tmp_path)
    cases = {}
    for p in (tmp_path / "reports" / "classification").glob("*.jsonl"):
        for r in PredictionStore(p).load():
            cases.setdefault((r.task.value, r.model_key), set()).add(r.case_id)
    for t in ("INTENT", "LEAD"):
        assert cases[(t, "typesafe/jev-1.14")] == cases[(t, "openai/gpt-4.1-nano")]


def test_a_failing_participant_is_quarantined_not_skipped_silently(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    net.fail_jev = lambda: urllib.error.HTTPError("https://x", 404, "nf", None, __import__("io").BytesIO(b"{}"))
    import urllib.error
    res = run(plan, loaded, budget, tmp_path)
    assert "typesafe/jev-1.14" in res.quarantined and "openai/gpt-4.1-nano" not in res.quarantined
    assert res.blocks_completed == plan["ordering"]["n_blocks"]          # the other participant finished
    st = bm.load_state(tmp_path / "bench", plan["bench_id"])
    assert st["quarantined"] and any(u["status"] in ("FAILED", "QUARANTINED") for u in st["units"].values())
    jev_recs = PredictionStore(bm.prediction_path(tmp_path / "reports", plan["bench_id"], "jev-1.14", "INTENT")).load()
    assert jev_recs and all(r.status == "ERROR" and r.predicted_label is None for r in jev_recs)     # an error is never a prediction


def test_the_hard_cap_gate_stops_before_any_request_and_never_raises_a_budget(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget, cap="0.000001")
    res = run(plan, loaded, budget, tmp_path)
    assert net.requests == [] and res.blocks_completed == 0 and "BUDGET_GATE_BEFORE_BLOCK_0" in res.halted
    assert plan["limits"]["pilot_usd"] == "2.00"


def test_unresolved_spend_halts_the_benchmark_before_dispatch(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    budget.record_unresolved_spend(transaction_id="tx_x", model_id="openai/gpt-4.1-nano", provider="openrouter", description="test", run_id="r")
    res = run(plan, loaded, budget, tmp_path)
    assert net.requests == [] and "unresolved" in res.halted


def test_manifest_is_versioned_records_the_authorization_and_holds_no_secret(prep, budget, net, tmp_path, monkeypatch):
    loaded, plan = make_plan(prep, budget)
    auth = issue(plan)
    live = LiveAuthorization(models=tuple(p["model_id"] for p in plan["participants"]))
    path = bm.write_manifest(tmp_path / "bench", plan, auth, live, "codefp")
    bm.write_manifest(tmp_path / "bench", plan, auth, live, "codefp")      # a resume appends a session, nothing is overwritten
    m = json.loads(path.read_text())
    assert m["manifest_version"] == "real-benchmark-manifest-v1" and len(m["sessions"]) == 2 and m["plan_fingerprint"] == plan["fingerprint"]
    a = m["sessions"][0]["authorization"]
    assert a["dataset_version"] == "prep-t" and a["real_admissions_derived"] and set(a["provider_policy_status"].values()) == {"UNKNOWN"}
    assert m["prompts"]["INTENT"]["sha256"] and "does not verify" in m["provider_policy_caveat"]
    run(plan, loaded, budget, tmp_path)
    blob = "".join(p.read_text() for p in (tmp_path / "bench").rglob("*") if p.is_file()) + "".join(p.read_text() for p in (tmp_path / "reports").rglob("*.jsonl"))
    assert SECRET not in blob and "test-not-a-real-key" not in blob


# ------------------------------------------------------------------ report and CLI
def _report(plan, tmp_path, prep=None, events=None):
    from msu_llm_lab.classification import benchmark_report as br
    return br.build_report(plan, bm.load_state(tmp_path / "bench", plan["bench_id"]), tmp_path / "reports", events or EventStore(tmp_path / "ev.jsonl"), prep)


def test_report_marks_a_partial_run_partial_and_a_complete_run_complete(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    run(plan, loaded, budget, tmp_path, max_blocks=1)
    part = _report(plan, tmp_path, prep)
    assert part["benchmark_state"] == "PARTIAL_OR_INCOMPLETE" and {p["completion_state"] for p in part["participants"]} == {"PARTIAL"}
    assert all(u["unattempted"] > 0 for p in part["participants"] for u in p["units"])
    run(plan, loaded, budget, tmp_path)
    full = _report(plan, tmp_path, prep)
    assert full["benchmark_state"] == "FULL_PLANNED_SCOPE_COMPLETE" and {p["completion_state"] for p in full["participants"]} == {"COMPLETE_FOR_PLANNED_SCOPE"}
    assert all(u["unattempted"] == 0 and u["valid_output_rate"] == 1.0 for p in full["participants"] for u in p["units"])


def test_full_roster_report_lists_all_ten_with_exact_states_including_never_executed(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget, models=None)
    run(plan, loaded, budget, tmp_path, max_blocks=1)
    rep = _report(plan, tmp_path, prep)
    assert len(rep["participants"]) == len(registry.REGISTRY) and rep["accuracy"]["status"] == "GROUND TRUTH PENDING"
    assert all(p["completion_state"] == "PARTIAL" for p in rep["participants"])
    # a participant that never ran is reported as such, not omitted and not counted complete
    fresh = _report(plan, tmp_path / "elsewhere", prep)
    assert len(fresh["participants"]) == len(registry.REGISTRY) and {p["completion_state"] for p in fresh["participants"]} == {"NOT_EXECUTED"}
    assert fresh["benchmark_state"] == "PARTIAL_OR_INCOMPLETE" and all(u["attempted"] == 0 for p in fresh["participants"] for u in p["units"])


def test_report_separates_cost_kinds_keeps_missing_metrics_missing_and_contains_no_student_text(prep, budget, net, tmp_path):
    from msu_llm_lab.classification import benchmark_report as br
    loaded, plan = make_plan(prep, budget)
    run(plan, loaded, budget, tmp_path)
    rep = _report(plan, tmp_path, prep)
    jev = next(p for p in rep["participants"] if p["key"] == "jev-1.14")
    u = jev["units"][0]
    assert u["provider_model_latency_ms"]["n"] == 0 and u["provider_model_latency_ms"]["note"] == "not reported by provider"
    assert "UNVERIFIED" in u["cost"]["native_unit"] and u["latency_ms_end_to_end"]["p95"] is None       # P95 withheld below 20 samples
    md = br.render_markdown(rep)
    assert "not reported by provider" in md and "Participant coverage (all ten)" in md and "GROUND TRUTH PENDING" in md and "not correctness" in md
    sample = [t.text for c in loaded.intent for t in ([c.current_message] + list(c.history))]
    assert not any(len(t) > 15 and t in md + json.dumps(rep) for t in sample)


def test_cli_plan_is_plan_only_and_run_without_live_never_sends(prep, budget, monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from msu_llm_lab.cli import app
    from msu_llm_lab.core.config import settings

    def boom(*a, **k):
        raise AssertionError("no request may be sent")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.chdir(tmp_path)
    for name in ("main_intent_classifier.md", "lead_qualification_prompt.md"):
        (tmp_path / name).write_text(SYN_INTENT_PROMPT if "intent" in name else SYN_LEAD_PROMPT)
    base = ["--prep-dir", str(prep), "--session-policy", "2h", "--block-size", "2"]
    r = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["real-benchmark", "plan", *base, "--model", "jev-1.14", "--model", "gpt-4.1-nano"])
    assert r.exit_code == 0 and "plan fingerprint:" in r.output and "plan only" not in r.output.lower() or "Plan stored" in r.output
    assert (settings.data_dir / "benchmarks").exists() and list((settings.data_dir / "benchmarks").glob("bench-*/plan.json"))
    dry = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["real-benchmark", "run", *base, "--operator", "T", "--model", "jev-1.14", "--model", "gpt-4.1-nano"])
    assert dry.exit_code == 0 and "Plan only: nothing was sent" in dry.output
    live = CliRunner().invoke(app, ["real-benchmark", "run", *base, "--operator", "T", "--model", "jev-1.14", "--model", "gpt-4.1-nano", "--live"])
    assert live.exit_code == 1 and "refused" in live.output.lower()            # non-interactive: the typed approval cannot be given
    unstored = CliRunner().invoke(app, ["real-benchmark", "run", "--prep-dir", str(prep), "--session-policy", "30m", "--operator", "T", "--model", "jev-1.14", "--live"])
    assert unstored.exit_code == 1 and "never stored" in unstored.output


def test_cli_report_writes_markdown_json_and_a_redacted_trace_export(prep, budget, net, tmp_path):
    from typer.testing import CliRunner
    from msu_llm_lab.cli import app
    from msu_llm_lab.core.config import settings

    loaded, plan = make_plan(prep, budget)
    root = settings.data_dir / "benchmarks"
    bm.save_plan(root, plan)
    models = [p["model_id"] for p in plan["participants"]]
    bm.execute(plan, loaded, PROMPTS, issue(plan), LiveAuthorization(models=tuple(models)), build_smoke_adapters(MODELS, bm.STAGE), budget, root=root,
               reports_dir=settings.reports_dir, max_blocks=1)
    r = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["real-benchmark", "report", plan["bench_id"], "--prep-dir", str(prep)])
    assert r.exit_code == 0, r.output
    md = settings.reports_dir / f"real_benchmark_{plan['bench_id']}.md"
    assert md.exists() and (settings.reports_dir / f"real_benchmark_{plan['bench_id']}.json").exists()
    zips = list((settings.reports_dir / "exports").glob("*.zip"))
    assert len(zips) == 1 and "Trace export (redacted by default)" in r.output
    import zipfile
    man = json.loads(zipfile.ZipFile(zips[0]).read("manifest.json"))
    assert man["telemetry_provenance"]["live_only"] and man["parity_status"] == "PRODUCTION_PARITY_NOT_ESTABLISHED" and set(m for m in man["model_ids"]) == set(models)
    rows = [json.loads(l) for l in zipfile.ZipFile(zips[0]).read("all_records.jsonl").decode().splitlines()]
    assert rows and {x["markers"]["data_kind"] for x in rows} == {"REAL_DATA_BENCHMARK"}


# ------------------------------------------------------------------ stratified sampling, max-blocks semantics, comparison bundle
def test_stratified_order_puts_every_pattern_bucket_first_then_proportional_and_is_deterministic(prep):
    loaded = bm.load_prep_cases(prep, "2h")
    for task in (Task.INTENT, Task.LEAD):
        strat = bm.order_cases(loaded, task, "stratified")
        hashed = bm.order_cases(loaded, task, "hash")
        buckets = set(loaded.strata[task.value].values())
        assert [c.case_id for c in strat] != [c.case_id for c in hashed] or len(buckets) == 1
        assert {loaded.strata[task.value][c.case_id] for c in strat[:len(buckets)]} == buckets      # variety first
        assert sorted(c.case_id for c in strat) == sorted(c.case_id for c in hashed)                # same cases, different order
        assert [c.case_id for c in strat] == [c.case_id for c in bm.order_cases(loaded, task, "stratified")]
    with pytest.raises(bm.PlanError, match="sampling"):
        bm.order_cases(loaded, Task.INTENT, "random")


def test_plan_records_the_sampling_and_executes_in_that_order(prep, budget, net, tmp_path):
    loaded = bm.load_prep_cases(prep, "2h")
    plan_s = bm.build_plan(loaded, PROMPTS, budget, model_keys=MODELS, block_size=2, hard_cap_usd=Decimal("0.05"), sampling="stratified")
    plan_h = bm.build_plan(loaded, PROMPTS, budget, model_keys=MODELS, block_size=2, hard_cap_usd=Decimal("0.05"), sampling="hash")
    assert plan_s["ordering"]["sampling"] == "stratified" and "NOT labels" in plan_s["ordering"]["note"] and plan_s["fingerprint"] != plan_h["fingerprint"]
    run(plan_s, loaded, budget, tmp_path, max_blocks=1)
    sent = {r.case_id for p in (tmp_path / "reports" / "classification").glob("*INTENT.jsonl") for r in PredictionStore(p).load()}
    assert sent == {c.case_id for c in bm.order_cases(loaded, Task.INTENT, "stratified")[:2]}


def test_max_blocks_is_cumulative_from_the_start_of_the_plan(prep, budget, net, tmp_path):
    loaded, plan = make_plan(prep, budget)
    r1 = run(plan, loaded, budget, tmp_path, max_blocks=1)
    n1 = len(net.requests)
    r3 = run(plan, loaded, budget, tmp_path, max_blocks=3)
    assert r1.blocks_completed == 1 and r3.blocks_completed == 3
    assert len(net.requests) == sum(b["requests"] for b in plan["blocks"][:3]) and n1 == plan["blocks"][0]["requests"]    # blocks 0-2 once each, no repeats
    again = run(plan, loaded, budget, tmp_path, max_blocks=3)
    assert again.blocks_completed == 3 and len(net.requests) == sum(b["requests"] for b in plan["blocks"][:3])


def test_comparison_bundle_has_every_file_hashes_no_overwrite_and_no_student_text(prep, budget, net, tmp_path):
    from msu_llm_lab.classification import benchmark_report as br
    from msu_llm_lab.models.registry import roster_status
    net.fail_models = {"openai/gpt-4o-mini": _404}
    loaded, plan = make_plan(prep, budget, models=None)
    run(plan, loaded, budget, tmp_path, max_blocks=2)
    state = bm.load_state(tmp_path / "bench", plan["bench_id"])
    bm.write_manifest(tmp_path / "bench", plan, issue(plan), LiveAuthorization(models=tuple(p["model_id"] for p in plan["participants"])), "fp")
    kwargs = dict(plan=plan, state=state, reports_dir=tmp_path / "reports", events=EventStore(tmp_path / "ev.jsonl"), roster=roster_status(), out_root=tmp_path / "out",
                  prep_dir=prep, loaded=loaded, manifest_path=tmp_path / "bench" / plan["bench_id"] / "manifest.json")
    a = br.write_bundle(**kwargs)
    b = br.write_bundle(**kwargs)
    assert a["dir"] != b["dir"] and b["dir"].name.endswith("-r2")                           # never overwrites an earlier bundle
    names = {p.name for p in a["dir"].iterdir()}
    assert set(br.BUNDLE_FILES) <= names
    man = json.loads((a["dir"] / "benchmark_manifest.json").read_text())
    import hashlib
    for name, digest in man["files"].items():
        assert hashlib.sha256((a["dir"] / name).read_bytes()).hexdigest() == digest
    assert man["benchmark_state"] == "PARTIAL_OR_INCOMPLETE" and len(man["participants"]) == len(registry.REGISTRY) and man["privacy"]["export"]["raw_student_text_in_files"] is False and man["privacy"]["inference_payloads"]["real_admissions_derived_data_sent_to_external_providers"] is True
    assert any("SAMPLED OPERATIONAL BENCHMARK" in w for w in man["warnings"]) and man["authorization_sessions"][0]["operator"] == "Test Operator"
    assert set(man["authorization_sessions"][0]["provider_policy_status"].values()) == {"UNKNOWN"}
    import csv as _csv
    summ = list(_csv.DictReader((a["dir"] / "model_summary.csv").open()))
    assert {r["model_key"] for r in summ} == set(registry.REGISTRY) and {r["task"] for r in summ} <= {"INTENT", "LEAD"}
    assert all(r["actual_usd"] == "" or float(r["actual_usd"]) >= 0 for r in summ) and {r["model_key"]: r["completion_state"] for r in summ}["gpt-4o-mini"] == "QUARANTINED_PARTIAL" and {r["completion_state"] for r in summ if r["model_key"] != "gpt-4o-mini"} == {"PARTIAL"}
    results = list(_csv.DictReader((a["dir"] / "model_results.csv").open()))
    assert len(results) == sum(1 for _ in (a["dir"] / "traces.jsonl").open()) == man["counts"]["trace_records"] > 0
    comp = list(_csv.DictReader((a["dir"] / "model_comparison.csv").open()))
    assert comp and "models_agree" in comp[0] and "n_models_unusable" in comp[0]
    unatt = list(_csv.DictReader((a["dir"] / "unattempted_in_executed_blocks.csv").open()))
    assert unatt and {r["status"] for r in unatt} == {"NOT_ATTEMPTED"} and {(r["model_key"], r["task"]) for r in unatt} == {("gpt-4o-mini", "INTENT"), ("gpt-4o-mini", "LEAD")} and len(unatt) == 4                       # a participant that was quarantined mid-block leaves cases unattempted inside the executed blocks
    sample = [t.text for c in loaded.intent for t in ([c.current_message] + list(c.history))]
    blob = "".join(p.read_text() for p in a["dir"].rglob("*") if p.is_file())
    assert not any(len(t) > 15 and t in blob for t in sample)


# ------------------------------------------------------------------ re-pins, request overrides and carry-over
def _404():
    import io
    return urllib.error.HTTPError("https://x", 404, "no endpoints", None, io.BytesIO(b"{}"))


def test_owner_approved_repins_are_in_the_registry_with_the_same_listed_price():
    pins = {k: registry.get_record(k) for k in ("gpt-4o-mini", "gpt-4.1", "baseline-lead-gpt-4.1-mini", "baseline-intent-gemini-3.5-flash-lite")}
    for k in ("gpt-4o-mini", "gpt-4.1", "baseline-lead-gpt-4.1-mini"):
        assert pins[k].pinned_tag == "azure" and pins[k].route.provider_preferences == {"order": ["azure"], "allow_fallbacks": False}
        assert any("Zero-Data-Retention" in a for a in pins[k].route.assumptions)
        openai = next(c for c in pins[k].candidate_routes if c.tag == "openai")
        assert (pins[k].pinned_route().input_per_m, pins[k].pinned_route().output_per_m) == (openai.input_per_m, openai.output_per_m)
    g = pins["baseline-intent-gemini-3.5-flash-lite"]
    assert g.pinned_tag == "google-vertex/global" and g.route.provider_preferences["allow_fallbacks"] is False
    studio = next(c for c in g.candidate_routes if c.tag == "google-ai-studio")
    assert (g.pinned_route().input_per_m, g.pinned_route().output_per_m) == (studio.input_per_m, studio.output_per_m)
    assert registry.get_record("qwen3.5-35b-a3b").request_overrides == {"reasoning": {"enabled": False}}
    assert not any(registry.get_record(k).request_overrides for k in registry.REGISTRY if k != "qwen3.5-35b-a3b")


def test_qwen_requests_carry_reasoning_off_and_other_models_send_nothing_extra(net, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    adapters = build_smoke_adapters(["qwen3.5-35b-a3b", "gpt-4.1-nano"], bm.STAGE)
    for mid in ("qwen/qwen3.5-35b-a3b", "openai/gpt-4.1-nano"):
        adapters[mid].classify(Task.INTENT, "[HISTORY]\n[NO PRIOR HISTORY]\n[CURRENT USER MESSAGE]\nhello", SYN_INTENT_PROMPT, live_authorized=True)
    bodies = [json.loads(r.data) for r in net.requests]
    assert bodies[0]["reasoning"] == {"enabled": False} and bodies[0]["provider"] == {"order": ["deepinfra/fp8"], "allow_fallbacks": False}
    assert "reasoning" not in bodies[1] and bodies[1]["provider"]["order"] == ["azure"]


def test_extra_request_fields_outside_the_approved_list_are_refused_before_sending(net):
    from msu_llm_lab.providers.base import ProviderError, ProviderRequest
    from msu_llm_lab.providers.openrouter import OpenRouterAdapter
    req = ProviderRequest(model_id="openai/gpt-4.1-nano", messages=[{"role": "user", "content": "x"}], live_authorized=True, extra_fields={"tools": []})
    with pytest.raises(ProviderError, match="approved extra-field list"):
        OpenRouterAdapter(api_key="k").generate(req)
    assert net.requests == []


def test_carry_over_keeps_clean_results_reruns_failed_participants_and_never_resends(prep, budget, net, tmp_path, monkeypatch):
    models = ["jev-1.14", "gpt-4.1-nano", "gpt-4o-mini"]
    loaded = bm.load_prep_cases(prep, "2h")
    parent = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"))
    net.fail_models = {"openai/gpt-4o-mini": _404}
    res = bm.execute(parent, loaded, PROMPTS, issue(parent), LiveAuthorization(models=tuple(p["model_id"] for p in parent["participants"])),
                     build_smoke_adapters(models, bm.STAGE), budget, root=tmp_path / "bench", reports_dir=tmp_path / "reports", events=EventStore(tmp_path / "ev.jsonl"), max_blocks=2)
    assert "openai/gpt-4o-mini" in res.quarantined and res.blocks_completed == 2
    sent_parent = len(net.requests)
    bm.save_plan(tmp_path / "bench", parent)
    pstate = bm.load_state(tmp_path / "bench", parent["bench_id"])

    # the failed participant is re-pinned/fixed (here: simply fixed upstream); the clean ones are carried over
    net.fail_models = {}
    child = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"), parent=(parent, pstate), reports_dir=tmp_path / "reports")
    assert child["inherits"]["participants"] == ["jev-1.14", "gpt-4.1-nano"] and child["inherits"]["parent_bench"] == parent["bench_id"]
    assert child["fingerprint"] != parent["fingerprint"]
    inh = {p["key"]: p for p in child["participants"]}
    assert inh["jev-1.14"]["store_bench"] == parent["bench_id"] and "store_bench" not in inh["gpt-4o-mini"]
    assert child["blocks"][0]["requests"] < parent["blocks"][0]["requests"] and child["blocks"][2]["requests"] == parent["blocks"][2]["requests"]    # new requests only, from block 2 on all are new
    assert "CARRY-OVER" in "\n".join(bm.render_plan(child))
    r2 = bm.execute(child, loaded, PROMPTS, issue(child), LiveAuthorization(models=tuple(p["model_id"] for p in child["participants"])),
                    build_smoke_adapters(models, bm.STAGE), budget, root=tmp_path / "bench", reports_dir=tmp_path / "reports", events=EventStore(tmp_path / "ev.jsonl"), max_blocks=2)
    assert r2.blocks_completed == 2 and not r2.quarantined
    new_requests = len(net.requests) - sent_parent
    assert new_requests == child["blocks"][0]["requests"] + child["blocks"][1]["requests"]              # only gpt-4o-mini was sent
    assert all(json.loads(r.data)["model"] == "openai/gpt-4o-mini" for r in net.requests[sent_parent:])
    # reports read the carried-over stores plus the new ones, all ten-style states stay explicit
    from msu_llm_lab.classification import benchmark_report as br
    rep = br.build_report(child, bm.load_state(tmp_path / "bench", child["bench_id"]), tmp_path / "reports", EventStore(tmp_path / "ev.jsonl"), prep)
    by = {p["key"]: p for p in rep["participants"]}
    assert by["jev-1.14"]["units"][0]["attempted"] == 4 and by["gpt-4o-mini"]["units"][0]["valid"] == 4 and by["gpt-4o-mini"]["completion_state"] == "PARTIAL"


def test_a_changed_participant_is_not_carried_over(prep, budget, net, tmp_path, monkeypatch):
    models = ["jev-1.14", "gpt-4.1-nano"]
    loaded = bm.load_prep_cases(prep, "2h")
    parent = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"))
    bm.execute(parent, loaded, PROMPTS, issue(parent), LiveAuthorization(models=tuple(p["model_id"] for p in parent["participants"])), build_smoke_adapters(models, bm.STAGE),
               budget, root=tmp_path / "bench", reports_dir=tmp_path / "reports", events=EventStore(tmp_path / "ev.jsonl"), max_blocks=1)
    pstate = bm.load_state(tmp_path / "bench", parent["bench_id"])
    changed = registry.get_record("gpt-4.1-nano").model_copy(update={"request_overrides": {"reasoning": {"enabled": False}}})
    monkeypatch.setitem(registry.REGISTRY, "gpt-4.1-nano", changed)
    child = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"), parent=(parent, pstate), reports_dir=tmp_path / "reports")
    assert child["inherits"]["participants"] == ["jev-1.14"]
    other_prompts = {**PROMPTS, Task.LEAD: PROMPTS[Task.LEAD] + " "}
    none = bm.build_plan(loaded, other_prompts, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"), parent=(parent, pstate), reports_dir=tmp_path / "reports")
    assert none["inherits"] is None                                      # different prompt: nothing is reused


def test_bundle_contains_the_detailed_comparison_one_report_per_participant_and_observability(prep, budget, net, tmp_path):
    from msu_llm_lab.classification import benchmark_report as br
    from msu_llm_lab.models.registry import roster_status
    net.fail_models = {"openai/gpt-4o-mini": _404}
    loaded, plan = make_plan(prep, budget, models=None)
    run(plan, loaded, budget, tmp_path, max_blocks=2)
    state = bm.load_state(tmp_path / "bench", plan["bench_id"])
    out = br.write_bundle(plan, state, tmp_path / "reports", EventStore(tmp_path / "ev.jsonl"), roster_status(), tmp_path / "out", prep, loaded,
                          ledger={"counted_against_caps_usd": "0.01"})
    d = out["dir"]
    assert {"comparison_report.md", "observability_report.md", "observability.json"} <= {p.name for p in d.iterdir()}
    models = {p.stem for p in (d / "models").iterdir()}
    assert models == set(registry.REGISTRY)                                         # one individual report for ALL ten, including failed ones
    cmp_md = (d / "comparison_report.md").read_text()
    for needle in ("Coverage and completion", "Reliability", "Speed and tokens", "Cost (actual, estimated, native; never mixed)", "not correctness", "pending approved labels", "QUARANTINED_PARTIAL"):
        assert needle in cmp_md, needle
    gm = (d / "models" / "gpt-4o-mini.md").read_text()
    assert "QUARANTINED" in gm and "request errors" in gm and "HTTP" in gm.upper() or "ProviderError" in gm
    jev = (d / "models" / "jev-1.14.md").read_text()
    assert "Native confidence field" in jev and "Selected-label probability" in jev and "P(yes)" in jev and "not reported by provider" in jev
    obs = json.loads((d / "observability.json").read_text())
    assert obs["integrity"]["started_without_end"] == 0 and obs["integrity"]["requests_finished"] == len(net.requests) and obs["ledger"]["counted_against_caps_usd"] == "0.01"
    assert obs["http_status_breakdown"].get("404", 0) >= 3 and "typesafe/jev-1.14" in obs["by_model"] and obs["timeline_per_minute"]
    assert "Telemetry integrity" in (d / "observability_report.md").read_text()
    man = json.loads((d / "benchmark_manifest.json").read_text())
    import hashlib
    assert all(hashlib.sha256((d / n).read_bytes()).hexdigest() == h for n, h in man["files"].items())
    sample = [t.text for c in loaded.intent for t in ([c.current_message] + list(c.history))]
    blob = "".join(p.read_text() for p in d.rglob("*") if p.is_file())
    assert not any(len(t) > 15 and t in blob for t in sample)


# ------------------------------------------------------------------ corrected bundle semantics, reconciliation, diagnostics, verification
def _bundle_after_a_run(prep, budget, net, tmp_path, fail=None, blocks=2, parent=None):
    from msu_llm_lab.classification import benchmark_report as br
    from msu_llm_lab.models.registry import roster_status
    net.fail_models = {m: _404 for m in (fail or [])}
    loaded, plan = make_plan(prep, budget, models=None)
    run(plan, loaded, budget, tmp_path, max_blocks=blocks)
    state = bm.load_state(tmp_path / "bench", plan["bench_id"])
    bm.write_manifest(tmp_path / "bench", plan, issue(plan), LiveAuthorization(models=tuple(p["model_id"] for p in plan["participants"])), "fp")
    out = br.write_bundle(plan, state, tmp_path / "reports", EventStore(tmp_path / "ev.jsonl"), roster_status(), tmp_path / "out", prep, loaded,
                          manifest_path=tmp_path / "bench" / plan["bench_id"] / "manifest.json", ledger={"counted_against_caps_usd": "0.01"}, prompts=PROMPTS)
    return plan, loaded, out


def test_the_report_disagreement_count_equals_the_comparison_csv_even_with_partial_participants(prep, budget, net, tmp_path):
    import re, csv as _csv
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path, fail=["openai/gpt-4o-mini"])
    d = out["dir"]
    comp = list(_csv.DictReader((d / "model_comparison.csv").open()))
    n_false = sum(1 for r in comp if r["models_agree"] == "False")
    md = (d / "comparison_report.md").read_text()
    m = re.search(r"differed: (\d+)", md)
    assert m and int(m.group(1)) == n_false == json.loads((d / "benchmark_manifest.json").read_text())["counts"]["disagreement_cases"]


def test_manifest_separates_export_privacy_from_inference_transmission(prep, budget, net, tmp_path):
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path)
    pr = json.loads((out["dir"] / "benchmark_manifest.json").read_text())["privacy"]
    assert pr["export"]["contains_raw_student_text"] is False and pr["export"]["this_bundle_transmitted_externally"] is False
    inf = pr["inference_payloads"]
    assert inf["real_admissions_derived_data_sent_to_external_providers"] is True and "does not verify any provider policy" in inf["meaning"]
    assert {v["data_policy"] for v in inf["recipients"].values()} == {"UNKNOWN"} and len(inf["recipients"]) == len(registry.REGISTRY)
    assert "transmitted_externally" not in pr and "transmitted_externally" not in json.dumps(pr).replace("this_bundle_transmitted_externally", "")


def test_error_rows_without_cost_evidence_are_counted_not_treated_as_zero_cost(prep, budget, net, tmp_path):
    import csv as _csv
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path, fail=["openai/gpt-4o-mini"])
    summ = [r for r in _csv.DictReader((out["dir"] / "model_summary.csv").open()) if r["model_key"] == "gpt-4o-mini"]
    assert sum(int(r["errors_no_response"]) for r in summ) >= 2 and sum(int(r["requests_without_cost_evidence"]) for r in summ) >= 2
    assert all(r["unknown_cost_after_response"] == "0" and r["actual_usd"] == "" and r["estimated_usd"] == "" for r in summ)          # blank, never 0
    assert json.loads((out["dir"] / "benchmark_manifest.json").read_text())["cost_semantics"]["rule"] == "an empty cost cell means not measured, never zero"


def test_jev_lead_probabilities_are_not_binary_decisions(prep, budget, net, tmp_path):
    import csv as _csv
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path)
    rows = [r for r in _csv.DictReader((out["dir"] / "model_results.csv").open()) if r["task"] == "LEAD" and r["requested_model_id"] == "typesafe/jev-1.14"]
    assert rows and all(r["yes_probability"] and r["decision"] == "" and r["predicted_need_cta"] == "" and r["binary_decision_available"] == "False" and r["probability_only"] == "True" for r in rows)
    chat = [r for r in _csv.DictReader((out["dir"] / "model_results.csv").open()) if r["task"] == "LEAD" and r["requested_model_id"] == "openai/gpt-4.1-nano"]
    assert chat and all(r["binary_decision_available"] == "True" and r["predicted_need_cta"] != "" for r in chat)
    summ = {(r["model_key"], r["task"]): r for r in _csv.DictReader((out["dir"] / "model_summary.csv").open())}
    assert summ[("jev-1.14", "LEAD")]["binary_decisions"] == "0" and int(summ[("jev-1.14", "LEAD")]["probability_only_results"]) > 0
    assert "NOT binary decisions" in (out["dir"] / "comparison_report.md").read_text()


def test_the_intent_distribution_diagnostic_is_labelled_as_observation_not_evidence(prep, budget, net, tmp_path):
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path)
    man = json.loads((out["dir"] / "benchmark_manifest.json").read_text())
    assert "UPLOAD_STATUS_CHECK" in man["sample_diagnostics"]["upload_status_check_note"] and "observed prediction distribution only" in man["sample_diagnostics"]["interpretation"] or "neither shows" in man["sample_diagnostics"]["interpretation"]
    assert set(next(iter(man["sample_diagnostics"]["intent_prediction_distribution_valid_outputs"].values()))) == {"GREETING", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST", "QUERY"}
    md = (out["dir"] / "comparison_report.md").read_text()
    assert "observed prediction distribution only" in md and "does not show that the category is absent" in md


def test_request_counts_reconcile_planned_dispatched_outcomes_attempts_and_unattempted(prep, budget, net, tmp_path):
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path, fail=["openai/gpt-4o-mini"], blocks=2)
    rc = json.loads((out["dir"] / "request_reconciliation.json").read_text())
    t = rc["totals"]
    assert all(v in (True, None) for v in rc["checks"].values()), rc["checks"]
    assert t["planned_new_logical_requests"] == t["dispatched_logical_requests"] + t["unattempted"] and t["unattempted"] > 0
    assert t["outcomes_recorded"] == t["ok"] + t["invalid_output"] + t["error"] and t["provider_http_attempts"] == t["outcomes_recorded"] + t["retries"]
    assert t["rows_in_results_files"] == len((out["dir"] / "model_results.csv").read_text().splitlines()) - 1
    md = (out["dir"] / "comparison_report.md").read_text()
    assert "Request-count reconciliation" in md and "NOT finished" in md
    man = json.loads((out["dir"] / "benchmark_manifest.json").read_text())
    assert man["matrix_completeness"]["complete_for_executed_blocks"] is False and man["request_reconciliation"]["totals"] == t


def test_carried_over_results_are_distinguishable_and_never_listed_as_unattempted(prep, budget, net, tmp_path):
    import csv as _csv
    from msu_llm_lab.classification import benchmark_report as br
    from msu_llm_lab.models.registry import roster_status
    models = ["jev-1.14", "gpt-4.1-nano", "gpt-4o-mini"]
    loaded = bm.load_prep_cases(prep, "2h")
    parent = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"))
    net.fail_models = {"openai/gpt-4o-mini": _404}
    live = LiveAuthorization(models=tuple(p["model_id"] for p in parent["participants"]))
    kw = dict(root=tmp_path / "bench", reports_dir=tmp_path / "reports", events=EventStore(tmp_path / "ev.jsonl"))
    bm.execute(parent, loaded, PROMPTS, issue(parent), live, build_smoke_adapters(models, bm.STAGE), budget, max_blocks=2, **kw)
    bm.save_plan(tmp_path / "bench", parent)
    net.fail_models = {}
    child = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"), parent=(parent, bm.load_state(tmp_path / "bench", parent["bench_id"])), reports_dir=tmp_path / "reports")
    bm.execute(child, loaded, PROMPTS, issue(child), live, build_smoke_adapters(models, bm.STAGE), budget, max_blocks=2, **kw)
    out = br.write_bundle(child, bm.load_state(tmp_path / "bench", child["bench_id"]), tmp_path / "reports", EventStore(tmp_path / "ev.jsonl"), roster_status(), tmp_path / "out", prep, loaded, prompts=PROMPTS)
    res = list(_csv.DictReader((out["dir"] / "model_results.csv").open()))
    origins = {r["requested_model_id"]: r["origin"] for r in res}
    assert origins["typesafe/jev-1.14"] == f"CARRIED_OVER_FROM:{parent['bench_id']}" and origins["openai/gpt-4o-mini"] == "NEW_THIS_PLAN"
    assert not [u for u in _csv.DictReader((out["dir"] / "unattempted_in_executed_blocks.csv").open()) if u["model_key"] in ("jev-1.14", "gpt-4.1-nano")]
    rc = json.loads((out["dir"] / "request_reconciliation.json").read_text())
    assert rc["totals"]["carried_over_records_not_resent"] == 16 and rc["totals"]["dispatched_logical_requests"] == rc["totals"]["outcomes_recorded"] == 8


def test_diagnostics_separate_verified_facts_from_hypotheses_and_prepare_synthetic_only_plans(prep, budget, net, tmp_path):
    from msu_llm_lab.classification import diagnostics as dg
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path, fail=["openai/gpt-4o-mini"])
    d = out["dir"] / "diagnostics"
    assert {"rate_limits.md", "empty_outputs.md", "rate_limits.json", "empty_outputs.json"} <= {p.name for p in d.iterdir()}
    assert "Synthetic-only test plan" in (d / "empty_outputs.md").read_text() or not json.loads((d / "empty_outputs.json").read_text())["participants"]
    # direct unit check on synthetic stored records: empties at the cap are verified; the reason stays a hypothesis
    from obs_helpers import intent_pred
    recs = [intent_pred(f"C{i}", "QUERY", model="claude-x", status="INVALID_OUTPUT", completion_tokens=16, validation_issues=["EMPTY_OUTPUT"]) for i in range(3)]
    recs += [intent_pred("C9", "QUERY", model="claude-x", completion_tokens=6)]
    fake_plan = {"bench_id": "bench-x", "participants": [{"key": "claude-x", "model_id": "claude-x", "tasks": ["INTENT"], "route": {"pinned_tag": "anthropic"}}]}
    eo = dg.empty_output_diagnostics(fake_plan, {"claude-x": {"INTENT": recs}})["participants"]["claude-x"]
    assert eo["empty_outputs"] == 3 and eo["empty_with_completion_tokens_equal_to_cap"] == 3 and eo["reasoning_tokens_reported"] == 0 and eo["finish_reason_values"] == {"not captured": 4}
    assert any("reasoning/thinking" in h for h in eo["hypotheses_not_verified"]) and any("equal to the request's max_tokens" in v for v in eo["verified"])
    md = dg.empty_output_markdown({"token_caps": {"INTENT": 16}, "participants": {"claude-x": eo}})
    assert "NOT run" in md and "synthetic fixtures only" in md and "Variant A" in md and "Variant B" in md and "separate run labels" in md


def test_finish_reason_and_reasoning_tokens_are_captured_when_the_provider_reports_them(monkeypatch):
    from msu_llm_lab.classification.chat_adapter import ChatClassificationAdapter
    from msu_llm_lab.providers.openrouter import OpenRouterAdapter
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    body = chat_body("", usage=True)
    body["choices"][0]["finish_reason"] = "length"
    body["usage"].update({"completion_tokens": 16, "completion_tokens_details": {"reasoning_tokens": 16}})
    script = [body, chat_body("QUERY")]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(json.dumps(script.pop(0)).encode()))
    a = ChatClassificationAdapter(model_id="openai/gpt-4.1-nano", provider=OpenRouterAdapter(registry_stage="pilot", sleep=lambda s: None))
    r1 = a.classify(Task.INTENT, "[HISTORY]\n[NO PRIOR HISTORY]\n[CURRENT USER MESSAGE]\nx", SYN_INTENT_PROMPT, live_authorized=True)
    r2 = a.classify(Task.INTENT, "[HISTORY]\n[NO PRIOR HISTORY]\n[CURRENT USER MESSAGE]\nx", SYN_INTENT_PROMPT, live_authorized=True)
    assert (r1.finish_reason, r1.reasoning_tokens, r1.ok) == ("length", 16, False)
    assert (r2.finish_reason, r2.reasoning_tokens) == (None, None)                         # absent stays None, never guessed


def test_bundle_verification_passes_on_a_good_bundle_and_catches_tampering(prep, budget, net, tmp_path):
    from msu_llm_lab.observability import bundle_verify as bv
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path, fail=["openai/gpt-4o-mini"])
    executed = {t.value: {c.case_id for c in bm.order_cases(loaded, t, plan["ordering"]["sampling"])[: 2 * plan["ordering"]["block_size"]]} for t in (Task.INTENT, Task.LEAD)}
    v = bv.verify_bundle(out["dir"], executed)
    assert v["ok"], v["failed"]
    csvp = out["dir"] / "model_summary.csv"
    csvp.write_text(csvp.read_text().replace(",OK,", ",OK,", 1) + "tampered\n")
    v2 = bv.verify_bundle(out["dir"], executed)
    assert not v2["ok"] and "manifest hashes match every listed file" in v2["failed"]


# ------------------------------------------------------------------ expanded benchmark: sizing, carry-over reasons, rankings, annotation templates
def _parent_run(prep, budget, net, tmp_path, models, fail=None, blocks=2):
    loaded = bm.load_prep_cases(prep, "2h")
    parent = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"))
    net.fail_models = {m: _404 for m in (fail or [])}
    live = LiveAuthorization(models=tuple(p["model_id"] for p in parent["participants"]))
    bm.execute(parent, loaded, PROMPTS, issue(parent), live, build_smoke_adapters(models, bm.STAGE), budget, root=tmp_path / "bench", reports_dir=tmp_path / "reports",
               events=EventStore(tmp_path / "ev.jsonl"), max_blocks=blocks)
    bm.save_plan(tmp_path / "bench", parent)
    return loaded, parent, bm.load_state(tmp_path / "bench", parent["bench_id"])


def test_sizing_reports_worst_case_and_expected_blocks_with_a_reserve_that_is_never_planned_away(prep, budget, net, tmp_path):
    models = ["jev-1.14", "gpt-4.1-nano"]
    loaded, parent, pstate = _parent_run(prep, budget, net, tmp_path, models)
    child = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("5.00"), parent=(parent, pstate), reports_dir=tmp_path / "reports",
                          reserve_usd=Decimal("1.00"))
    z = child["sizing"]
    head = min(budget.get_status().remaining_total_usd, budget.get_status().remaining_pilot_usd)
    assert Decimal(z["spendable_usd"]) == head - Decimal("1.00") and Decimal(child["limits"]["hard_cap_usd"]) <= head - Decimal("1.00") and child["limits"]["reserve_usd"] == "1.000000"
    assert z["largest_blocks_fitting_expected_cost"] >= 0 and z["largest_blocks_fitting_worst_case"] >= 0
    assert z["observed_basis"] and all(v["n"] > 0 for v in z["observed_basis"].values())
    big = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("5.00"), parent=(parent, pstate), reports_dir=tmp_path / "reports", reserve_usd=Decimal("0.10"))
    rows = {r["blocks"]: r for r in big["sizing"]["table"]}
    assert float(big["sizing"]["spendable_usd"]) > float(z["spendable_usd"]) and "SIZING:" in "\n".join(bm.render_plan(big))


def test_a_participant_with_request_errors_is_not_carried_over_and_the_plan_says_why_and_how_much_is_re_sent(prep, budget, net, tmp_path):
    models = ["jev-1.14", "gpt-4o-mini"]
    loaded, parent, pstate = _parent_run(prep, budget, net, tmp_path, models, fail=["openai/gpt-4o-mini"])
    child = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"), parent=(parent, pstate), reports_dir=tmp_path / "reports")
    nc = child["inherits"]["not_carried_over"]["gpt-4o-mini"]
    assert child["inherits"]["participants"] == ["jev-1.14"] and "quarantined" in nc["reason"] and nc["failed_requests_retried"] > 0
    txt = "\n".join(bm.render_plan(child))
    assert "NOT carried over" in txt and "gpt-4o-mini" in txt


def test_bundle_has_operational_rankings_labelled_as_not_accuracy_and_cost_per_1000(prep, budget, net, tmp_path):
    import csv as _csv
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path, fail=["openai/gpt-4o-mini"])
    d = out["dir"]
    ranks = list(_csv.DictReader((d / "operational_ranking.csv").open()))
    assert ranks and {r["ranking"] for r in ranks} == {"OPERATIONAL (not accuracy)"} and {r["metric"] for r in ranks} == {"valid_output_rate", "median_e2e_latency_ms", "estimated_usd_per_1000_cases"}
    assert all(r["sample_note"] == "n < 20: indicative only" for r in ranks if int(r["n_requests"]) < 20)
    md = (d / "comparison_report.md").read_text()
    assert "NOT accuracy rankings" in md and "Accuracy ranking: **unavailable**" in md and "Route and configuration failures" in md and "gpt-4o-mini" in md
    summ = list(_csv.DictReader((d / "model_summary.csv").open()))
    assert {"est_usd_per_request", "est_usd_per_1000_cases"} <= set(summ[0]) and any(r["est_usd_per_1000_cases"] for r in summ)
    assert all(r["est_usd_per_1000_cases"] == "" for r in summ if r["estimated_n"] == "0")


def test_annotation_templates_hold_opaque_ids_only_and_never_overwrite(prep, tmp_path):
    import csv as _csv
    from msu_llm_lab.observability import annotation_templates as at
    loaded = bm.load_prep_cases(prep, "2h")
    paths = at.write_templates(prep, loaded, "stratified", 2, 3)
    ti = list(_csv.DictReader(paths["INTENT"].open()))
    tl = list(_csv.DictReader(paths["LEAD"].open()))
    assert len(ti) == min(6, len(loaded.intent)) and len(tl) == min(6, len(loaded.lead))
    assert list(ti[0]) == at.INTENT_COLUMNS and "final_expected_need_cta" in tl[0] and ti[0]["review_status"] == "PENDING" and ti[0]["approval_status"] == "NOT_APPROVED"
    assert all(not r["reviewer_1_label"] and not r["final_expected_intent"] and not r["adjudicated_label"] for r in ti)       # nothing is pre-filled or suggested
    blob = paths["INTENT"].read_text() + paths["LEAD"].read_text()
    sample = [t.text for c in loaded.intent for t in ([c.current_message] + list(c.history))]
    assert not any(len(t) > 15 and t in blob for t in sample) and "model" not in blob.lower().split("\n")[0]
    assert [c.case_id for c in bm.order_cases(loaded, Task.INTENT, "stratified")[:6]] == [r["case_id"] for r in ti]
    with pytest.raises(__import__("msu_llm_lab.classification.annotation", fromlist=["x"]).AnnotationError, match="never overwritten"):
        at.write_templates(prep, loaded, "stratified", 2, 3)
    assert (prep / "annotations" / "templates" / "README.md").exists()


def test_filled_sheets_flow_into_the_annotation_workflow_without_approving_anything(prep, tmp_path):
    import csv as _csv
    from msu_llm_lab.classification import annotation as an
    from msu_llm_lab.observability import annotation_templates as at
    from msu_llm_lab.observability.labels import load_approved_labels
    loaded = bm.load_prep_cases(prep, "2h")
    paths = at.write_templates(prep, loaded, "stratified", 1, 3)
    rows = list(_csv.DictReader(paths["INTENT"].open()))
    for slot, who in ((1, "alice"), (2, "bob")):
        for r in rows[:2]:
            r[f"reviewer_{slot}_id"], r[f"reviewer_{slot}_label"] = who, "QUERY"
    sheet = tmp_path / "filled.csv"
    with sheet.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=at.INTENT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    valid = an.eligible_case_ids(loaded)
    assert at.import_sheet(sheet, prep, valid, 1) == 2 and at.import_sheet(sheet, prep, valid, 2) == 2
    states = an.case_states(prep)
    assert {s.state for s in states.values()} == {"AGREED"} and not load_approved_labels(prep).available
    with pytest.raises(an.AnnotationError, match="slot"):
        at.import_sheet(sheet, prep, valid, 3)


def test_diagnostics_serialize_when_some_records_have_a_finish_reason_and_some_do_not():
    from msu_llm_lab.classification import diagnostics as dg
    from obs_helpers import intent_pred
    recs = [intent_pred("C1", "QUERY", model="claude-x", status="INVALID_OUTPUT", completion_tokens=16, validation_issues=["EMPTY_OUTPUT"], finish_reason="length", reasoning_tokens=16),
            intent_pred("C2", "QUERY", model="claude-x", status="INVALID_OUTPUT", completion_tokens=16, validation_issues=["EMPTY_OUTPUT"]), intent_pred("C3", "QUERY", model="claude-x", completion_tokens=6)]
    plan = {"bench_id": "bench-x", "participants": [{"key": "claude-x", "model_id": "claude-x", "tasks": ["INTENT"], "route": {"pinned_tag": "anthropic"}}]}
    d = dg.empty_output_diagnostics(plan, {"claude-x": {"INTENT": recs}})
    text = json.dumps(d, sort_keys=True)                                    # must not raise on mixed None / str keys
    x = d["participants"]["claude-x"]
    assert x["finish_reason_values"] == {"length": 1, "not captured": 2} and x["reasoning_tokens_reported"] == 1 and "length" in text


def test_a_partly_carried_participant_reconciles_and_each_row_says_which_part_is_new(prep, budget, net, tmp_path):
    import csv as _csv
    from msu_llm_lab.classification import benchmark_report as br
    from msu_llm_lab.models.registry import roster_status
    models = ["jev-1.14", "gpt-4o-mini"]
    loaded, parent, pstate = _parent_run(prep, budget, net, tmp_path, models, blocks=2)
    child = bm.build_plan(loaded, PROMPTS, budget, model_keys=models, block_size=2, hard_cap_usd=Decimal("0.05"), parent=(parent, pstate), reports_dir=tmp_path / "reports")
    assert child["inherits"]["participants"] == ["jev-1.14", "gpt-4o-mini"]
    live = LiveAuthorization(models=tuple(p["model_id"] for p in child["participants"]))
    bm.execute(child, loaded, PROMPTS, issue(child), live, build_smoke_adapters(models, bm.STAGE), budget, root=tmp_path / "bench", reports_dir=tmp_path / "reports",
               events=EventStore(tmp_path / "ev.jsonl"), max_blocks=4)
    out = br.write_bundle(child, bm.load_state(tmp_path / "bench", child["bench_id"]), tmp_path / "reports", EventStore(tmp_path / "ev.jsonl"), roster_status(), tmp_path / "out", prep, loaded, prompts=PROMPTS)
    rc = json.loads((out["dir"] / "request_reconciliation.json").read_text())
    assert all(v in (True, None) for v in rc["checks"].values()), rc["checks"]
    t = rc["totals"]
    assert t["planned_new_logical_requests"] == t["plan_block_requests_for_executed_blocks"] == t["dispatched_logical_requests"] == t["outcomes_recorded"] and t["carried_over_records_not_resent"] == 16
    res = list(_csv.DictReader((out["dir"] / "model_results.csv").open()))
    o = {(r["requested_model_id"], r["task"], r["case_id"]): r["origin"] for r in res}
    expected_rows = 2 * min(8, len(loaded.intent)) + 2 * min(8, len(loaded.lead))
    assert len(res) == t["rows_in_results_files"] == expected_rows
    assert sum(v.startswith("CARRIED_OVER_FROM") for v in o.values()) == 16 and sum(v == "NEW_THIS_PLAN" for v in o.values()) == expected_rows - 16
    summ = {(r["model_key"], r["task"]): r["origin"] for r in _csv.DictReader((out["dir"] / "model_summary.csv").open())}
    assert "(blocks 0-1) + NEW_THIS_PLAN (later blocks)" in summ[("jev-1.14", "INTENT")]
    from msu_llm_lab.observability import bundle_verify as bv
    executed = {tt.value: {c.case_id for c in bm.order_cases(loaded, tt, child["ordering"]["sampling"])[: 4 * 2]} for tt in (Task.INTENT, Task.LEAD)}
    assert bv.verify_bundle(out["dir"], executed)["ok"]


def test_operational_ranking_gives_tied_values_the_same_rank(prep, budget, net, tmp_path):
    import csv as _csv
    plan, loaded, out = _bundle_after_a_run(prep, budget, net, tmp_path)
    rows = [r for r in _csv.DictReader((out["dir"] / "operational_ranking.csv").open()) if r["metric"] == "valid_output_rate" and r["task"] == "INTENT"]
    perfect = [r for r in rows if float(r["value"]) == 1.0]
    assert len(perfect) >= 2 and {r["rank"] for r in perfect} == {"1"}
