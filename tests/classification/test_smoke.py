"""Offline tests for the synthetic smoke-test plan, typed approval, execution and CLI, plus Jev version-form rules.

Only hand-written SYNTHETIC fixtures, fake urlopen, and a temp working directory are used. Nothing is sent anywhere.
"""

import json
import urllib.request
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from msu_llm_lab.classification import jev_adapter as ja
from msu_llm_lab.classification import smoke
from msu_llm_lab.classification.contracts import IntentLabel, ReviewStatus
from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.cli import app
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.config import BudgetPolicy
from msu_llm_lab.core.live_guard import LiveAuthorization, confirm_live_run, expected_phrase
from msu_llm_lab.providers.base import LiveNotAuthorizedError
from msu_llm_lab.schemas.evaluation import DataClassification as DC
from msu_llm_lab.storage.db import DatabaseManager
from registry_helpers import block_route
from synthetic_fixtures import SYN_INTENT_PROMPT, SYN_LEAD_PROMPT
from test_runner_adapters import FakeNet, jev_body, jev_choice, noul_body

PROMPTS = {Task.INTENT: SYN_INTENT_PROMPT, Task.LEAD: SYN_LEAD_PROMPT}
JEV = "jev-1.14"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def budget(tmp_path):
    return BudgetService(db=DatabaseManager(db_path=tmp_path / "smoke.db"))


# ------------------------------------------------------------------ fixtures
def test_fixtures_are_five_per_task_all_synthetic_unscored_and_unlabelled():
    fx = smoke.build_fixtures()
    assert [sum(f.task == t for f in fx) for t in (Task.INTENT, Task.LEAD)] == [5, 5]
    for f in fx:
        c = f.case
        assert c.classification == DC.SYNTHETIC and c.review_status == ReviewStatus.NOT_REVIEWED and f.scored is False
        assert getattr(c, "expected_intent", None) is None and getattr(c, "expected_need_cta", None) is None
        assert c.case_id.startswith("smoke-")
    ids = [f.case.case_id for f in fx]
    assert len(set(ids)) == 10


def test_intent_fixtures_cover_all_five_labels_with_contextual_cases_and_no_unresolved_rules():
    fx = [f for f in smoke.build_fixtures() if f.task == Task.INTENT]
    assert {f.intended for f in fx} == set(IntentLabel)
    assert any(f.kind == "contextual" for f in fx)
    text = " ".join(t.text.lower() for f in fx for t in f.case.history + [f.case.current_message])
    assert "any update" not in text and "no thanks" not in text  # O1 and handoff follow-ups excluded
    lead = [f for f in smoke.build_fixtures() if f.task == Task.LEAD]
    summary = " ".join(" ".join(f.case.action_summary).lower() for f in lead)
    for unresolved in ("expensive", "discount", "later session", "deadline", "offered to upload", "recognition"):
        assert unresolved not in summary  # Lead L4-L8 situations are not in the smoke set


def test_fixture_module_never_touches_the_real_csv_or_network():
    src = Path(smoke.__file__).read_text()
    assert "september_to_october" not in src and "import csv" not in src and "import socket" not in src and "import urllib" not in src


# ------------------------------------------------------------------ plan
def test_plan_contains_everything_the_approver_needs(budget):
    plan = smoke.build_plan([JEV], [Task.INTENT, Task.LEAD], PROMPTS, budget)
    text = "\n".join(smoke.render_plan(plan))
    for needed in ("typesafe/jev-1.14", "DECISIONS_API", "TypeSafe", "VERIFIED_PRIMARY", "cases: 5", "INTENT", "LEAD",
                   "upper-bound spend", "Hard cap for this run: $0.05", "per-run ceiling: $0.50", "Pilot ceiling: $2.00", plan.fingerprint,
                   "schema verification only"):
        assert needed in text, needed
    assert not plan.blocked and plan.total_upper_bound_usd > 0 and plan.total_upper_bound_usd < plan.hard_cap_usd
    assert "STATUS: no blockers" in text


def test_plan_blocks_a_blocked_route_and_models_outside_the_smoke_pair(budget, monkeypatch):
    block_route(monkeypatch, "gpt-4.1-nano")
    plan = smoke.build_plan([JEV, "gpt-4.1-nano", "gpt-4o-mini"], [Task.INTENT], PROMPTS, budget)
    joined = " | ".join(plan.blocked_reasons)
    assert plan.blocked and "gpt-4.1-nano (INTENT): route" in joined and "gpt-4o-mini: not part of the two-model smoke test" in joined
    nano = [r for r in plan.rows if r.model_key == "gpt-4.1-nano"][0]
    assert "BLOCKED" in nano.route and '"allow_fallbacks": false' in nano.route and "[VERIFIED_PRIMARY]" in nano.pricing
    assert nano.upper_bound_usd is not None  # the price is verified, so the bound exists; the blocked route is what blocks


def test_plan_fingerprint_is_stable_and_binds_the_details(budget):
    a = smoke.build_plan([JEV], [Task.INTENT], PROMPTS, budget)
    assert a.fingerprint == smoke.build_plan([JEV], [Task.INTENT], PROMPTS, budget).fingerprint
    assert a.fingerprint != smoke.build_plan([JEV], [Task.INTENT], PROMPTS, budget, Decimal("0.04")).fingerprint
    assert a.fingerprint != smoke.build_plan([JEV], [Task.LEAD], PROMPTS, budget).fingerprint
    assert a.fingerprint != smoke.build_plan([JEV], [Task.INTENT], {**PROMPTS, Task.INTENT: PROMPTS[Task.INTENT] + "x"}, budget).fingerprint


def test_cap_rules_use_the_unchanged_pilot_policy(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "p.db")
    base = BudgetService(db=db)
    assert base.policy.pilot_budget_usd == Decimal("2.00") and base.policy.per_run_ceiling_usd == Decimal("0.50")
    assert any("per-run ceiling" in b for b in smoke.build_plan([JEV], [Task.INTENT], PROMPTS, base, Decimal("0.75")).blocked_reasons)
    assert any("exceeds the hard cap" in b for b in smoke.build_plan([JEV], [Task.INTENT], PROMPTS, base, Decimal("0.0000001")).blocked_reasons)
    assert any("must be positive" in b for b in smoke.build_plan([JEV], [Task.INTENT], PROMPTS, base, Decimal("0")).blocked_reasons)
    tight = BudgetService(db=db, policy=BudgetPolicy(pilot_budget_usd=Decimal("0.01")))
    assert any("pilot ceiling" in b for b in smoke.build_plan([JEV], [Task.INTENT], PROMPTS, tight, Decimal("0.05")).blocked_reasons)


# ------------------------------------------------------------------ typed approval is bound to the plan
def test_approval_phrase_binds_the_exact_plan_and_needs_an_interactive_terminal(budget):
    plan = smoke.build_plan([JEV], [Task.INTENT], PROMPTS, budget)
    phrase = expected_phrase(plan.model_ids, plan.fingerprint)
    assert plan.fingerprint in phrase
    ok = confirm_live_run(plan.model_ids, is_interactive=lambda: True, prompt=lambda m: phrase, echo=lambda l: None, plan_fingerprint=plan.fingerprint)
    assert ok.covers("typesafe/jev-1.14")
    for wrong in (expected_phrase(plan.model_ids), expected_phrase(plan.model_ids, "deadbeef0000"), ""):
        with pytest.raises(LiveNotAuthorizedError):
            confirm_live_run(plan.model_ids, is_interactive=lambda: True, prompt=lambda m, w=wrong: w, echo=lambda l: None, plan_fingerprint=plan.fingerprint)
    with pytest.raises(LiveNotAuthorizedError):
        confirm_live_run(plan.model_ids, is_interactive=lambda: False, prompt=lambda m: phrase, echo=lambda l: None, plan_fingerprint=plan.fingerprint)


# ------------------------------------------------------------------ execution with a fake transport
@pytest.fixture
def fake_net(monkeypatch):
    n = FakeNet()
    monkeypatch.setattr(urllib.request, "urlopen", n)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    return n


def jev_script():
    return [jev_body(jev_choice(l)) for l in ("GREETING", "QUERY", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST")] + \
           [noul_body(p) for p in (0.9, 0.1, 0.8, 0.85, 0.2)]


def test_execute_plan_writes_records_and_reports_schema_checks_without_accuracy(tmp_path, budget, fake_net):
    plan = smoke.build_plan([JEV], [Task.INTENT, Task.LEAD], PROMPTS, budget)
    fake_net.script = jev_script()
    store = PredictionStore(tmp_path / "out" / "r.jsonl")
    sums = smoke.execute_plan(plan, LiveAuthorization(models=tuple(plan.model_ids)), smoke.build_smoke_adapters([JEV]), store, budget, PROMPTS, "smoke_t")
    assert len(fake_net.requests) == 10 and all(s.aborted_reason is None for s in sums)
    recs = store.load()
    assert len(recs) == 10 and {r.dataset_version for r in recs} == {smoke.SMOKE_DATASET_VERSION} and all(r.status == "OK" for r in recs)
    assert all(not r.is_ground_truth for r in recs)
    rows = smoke.summarize_schema_checks(recs)
    assert {(r["model"], r["task"], r["n"], r["valid"]) for r in rows} == {("typesafe/jev-1.14", "INTENT", 5, 5), ("typesafe/jev-1.14", "LEAD", 5, 5)}
    assert all("accuracy" not in k and "correct" not in k for r in rows for k in r)
    lead = [r for r in recs if r.task == Task.LEAD]
    assert all(r.yes_prob is not None and r.predicted_need_cta is None for r in lead)
    assert all(r.model_latency_ms is None for r in recs)  # never fabricated


def test_execute_refuses_blocked_plan_and_stops_at_the_hard_cap(tmp_path, budget, fake_net, monkeypatch):
    block_route(monkeypatch, "gpt-4.1-nano")
    blocked = smoke.build_plan([JEV, "gpt-4.1-nano"], [Task.INTENT], PROMPTS, budget)
    with pytest.raises(RuntimeError, match="blocked plan"):
        smoke.execute_plan(blocked, LiveAuthorization(models=("typesafe/jev-1.14",)), smoke.build_smoke_adapters([JEV]),
                           PredictionStore(tmp_path / "x.jsonl"), budget, PROMPTS, "r")
    assert fake_net.requests == []
    plan = smoke.build_plan([JEV], [Task.INTENT], PROMPTS, budget, Decimal("0.0015"))
    assert not plan.blocked
    fake_net.script = [jev_body(jev_choice(), usage={"input_tokens": 10, "output_tokens": 1, "cost": 0.001}) for _ in range(5)]
    store = PredictionStore(tmp_path / "cap.jsonl")
    with pytest.raises(BudgetExceededError, match="hard cap"):
        smoke.execute_plan(plan, LiveAuthorization(models=("typesafe/jev-1.14",)), smoke.build_smoke_adapters([JEV]), store, budget, PROMPTS, "cap")
    assert len(fake_net.requests) == 2 and len(store.load()) == 2


# ------------------------------------------------------------------ CLI
@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "main_intent_classifier.md").write_text(SYN_INTENT_PROMPT)
    (tmp_path / "lead_qualification_prompt.md").write_text(SYN_LEAD_PROMPT)
    return tmp_path


def run_cli(*args, **kw):
    return CliRunner(env={"COLUMNS": "240"}).invoke(app, ["smoke-test", *args], **kw)


def test_cli_default_is_plan_only_and_sends_nothing(cli_env, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    r = run_cli("--model", JEV)
    assert r.exit_code == 0 and "PREFLIGHT PLAN" in r.output and "Plan only: nothing was sent" in r.output
    assert run_cli().exit_code == 0  # the default pair is clear for the smoke stage
    block_route(monkeypatch, "gpt-4.1-nano")
    blocked = run_cli()  # the default pair includes GPT-4.1 Nano; a blocked route blocks the plan
    assert blocked.exit_code == 2 and "STATUS: BLOCKED" in blocked.output and "gpt-4.1-nano" in blocked.output
    assert not (cli_env / "reports").exists()


def test_cli_live_refuses_without_interactive_approval_even_with_the_right_phrase(cli_env, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    plan = smoke.build_plan([JEV], [Task.INTENT, Task.LEAD], PROMPTS, BudgetService())
    phrase = expected_phrase(plan.model_ids, plan.fingerprint)
    r = run_cli("--model", JEV, "--live", input=phrase + "\n")
    assert r.exit_code == 1 and "refused" in r.output.lower()
    assert not (cli_env / "reports").exists()


def test_cli_live_refuses_a_blocked_plan_before_asking_for_approval(cli_env, monkeypatch):
    block_route(monkeypatch, "gpt-4.1-nano")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    import msu_llm_lab.core.live_guard as lg
    monkeypatch.setattr(lg, "confirm_live_run", lambda *a, **k: pytest.fail("approval asked for a blocked plan"))
    r = run_cli("--live")  # includes gpt-4.1-nano
    assert r.exit_code == 1 and "plan is blocked" in r.output


def test_cli_validates_arguments_and_missing_prompts(cli_env):
    assert run_cli("--hard-cap", "abc").exit_code == 1
    assert run_cli("--task", "nonsense").exit_code == 1
    (cli_env / "main_intent_classifier.md").unlink()
    r = run_cli("--model", JEV)
    assert r.exit_code == 1 and "Authoritative prompt not found" in r.output


def test_cli_full_live_path_with_fake_transport_and_approval_bound_to_the_plan(cli_env, fake_net, monkeypatch):
    import msu_llm_lab.core.live_guard as lg
    seen = {}

    def fake_confirm(models, summary_lines=(), **kw):
        seen["fingerprint"] = kw.get("plan_fingerprint")
        return LiveAuthorization(models=tuple(models))
    monkeypatch.setattr(lg, "confirm_live_run", fake_confirm)
    fake_net.script = jev_script()
    r = run_cli("--model", JEV, "--live", "--out-dir", str(cli_env / "out"))
    assert r.exit_code == 0 and seen["fingerprint"] and f"Plan fingerprint: {seen['fingerprint']}" in r.output
    assert "Schema verification only (no accuracy computed)" in r.output
    files = list((cli_env / "out").glob("smoke_*.jsonl"))
    assert len(files) == 1 and len(files[0].read_text().splitlines()) == 10 and len(fake_net.requests) == 10


def test_models_roster_command_lists_all_ten_with_precise_readiness(cli_env):
    r = CliRunner(env={"COLUMNS": "300"}).invoke(app, ["models", "roster"])
    assert r.exit_code == 0
    for name in ("Gemini", "GPT-4.1-mini", "Jev 1.14", "GPT-4.1 Nano", "GPT-4o", "Qwen3.5", "DeepSeek", "Mistral", "Claude Sonnet 5"):
        assert name in r.output, name
    assert "SMOKE_READY" in r.output and "NOT_YET_EVALUATED" in r.output and "PRODUCTION_BASELINE" in r.output and "FRONTIER_REFERENCE" in r.output


# ------------------------------------------------------------------ Jev version forms (documented for 1.13 only)
REQ = "typesafe/jev-1.14"


def parse(model_value, present=True):
    body = jev_body()
    if present:
        body["model"] = model_value
    else:
        del body["model"]
    return ja.parse_decisions_response(Task.INTENT, REQ, body)


def test_exact_and_dated_1_14_identifiers_are_accepted_with_an_explicit_warning_for_the_snapshot_form():
    exact = parse("typesafe/jev-1.14")
    assert exact.ok and exact.resolved_model_id == "typesafe/jev-1.14" and "DATED_SNAPSHOT_NOT_YET_SEEN_IN_A_1_14_RESPONSE" not in exact.warnings
    dated = parse("typesafe/jev-1.14-20261001")
    assert dated.ok and dated.resolved_model_id == "typesafe/jev-1.14-20261001" and "DATED_SNAPSHOT_NOT_YET_SEEN_IN_A_1_14_RESPONSE" in dated.warnings
    assert "NOT_YET_SEEN_IN_A_1.14_RESPONSE" in ja.SNAPSHOT_RULE_STATUS


@pytest.mark.parametrize("returned", ["typesafe/jev-1.13-20260917", "typesafe/jev-1.13", "typesafe/jev-1.15-20261101", "typesafe/jev-1.140"])
def test_other_versions_are_rejected_for_a_1_14_request(returned):
    r = parse(returned)
    assert not r.ok and "RESOLVED_VERSION_DIFFERS_FROM_REQUESTED" in r.issues and r.predicted_label is None


@pytest.mark.parametrize("returned", ["typesafe/jev-1.14-latest", "typesafe/jev-1.14-2026", "typesafe/jev-1.14-20261301", "typesafe/jev-1.14-20261001-b",
                                      "jev-1.14", "other/model-1.14", "TYPESAFE/jev-1.14", "typesafe/jev-1.14 "])
def test_unrecognised_forms_are_fatal_not_guessed(returned):
    r = parse(returned)
    assert not r.ok and "RESOLVED_VERSION_FORM_UNRECOGNISED" in r.issues and r.predicted_label is None


def test_missing_version_is_a_warning_and_malformed_is_fatal_and_nothing_is_invented():
    missing = parse(None, present=False)
    assert missing.ok and missing.resolved_model_id is None and "RESOLVED_MODEL_MISSING" in missing.warnings
    for bad in (123, "", ["typesafe/jev-1.14"], {"id": "x"}):
        r = parse(bad)
        assert not r.ok and "MODEL_FIELD_MALFORMED" in r.issues and r.resolved_model_id is None


def test_classify_resolved_version_kinds():
    assert ja.classify_resolved_version(REQ, "typesafe/jev-1.14") == ("EXACT", None)
    assert ja.classify_resolved_version(REQ, "typesafe/jev-1.14-20261001") == ("DATED_SNAPSHOT", "20261001")
    assert ja.classify_resolved_version(REQ, "typesafe/jev-1.13-20260917") == ("DIFFERENT_VERSION", "20260917")
    assert ja.classify_resolved_version(REQ, "garbage")[0] == "UNRECOGNISED"


# ------------------------------------------------------------------ api-preflight never reveals the key or touches the network
def test_api_preflight_reports_presence_without_revealing_the_key(monkeypatch):
    sentinel = "sk-or-v1-SYNTHETICSENTINEL1234567890"
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    r = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["api-preflight"])
    assert r.exit_code == 0 and "yes" in r.output and "no request is sent" in r.output
    assert sentinel not in r.output and "SENTINEL" not in r.output and "sk-or" not in r.output
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    missing = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["api-preflight"])
    assert missing.exit_code == 1 and "export OPENROUTER_API_KEY" in missing.output
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert CliRunner().invoke(app, ["api-preflight"]).exit_code == 1  # blank counts as missing


# ------------------------------------------------------------------ expanded pilot plan (every eligible participant, strict gating)
def test_pilot_plan_covers_every_participant_with_per_task_support_and_exact_blockers(budget):
    from msu_llm_lab.models.registry import REGISTRY
    plan = smoke.build_plan(list(REGISTRY), [Task.INTENT, Task.LEAD], PROMPTS, budget, Decimal("0.25"), stage="pilot")
    by = {}
    for r in plan.rows:
        by.setdefault(r.model_key, set()).add(r.task.value)
    assert set(by) == set(REGISTRY)                                    # nobody is silently dropped
    assert by["baseline-intent-gemini-3.5-flash-lite"] == {"INTENT"} and by["baseline-lead-gpt-4.1-mini"] == {"LEAD"}
    assert by["gpt-4.1"] == {"INTENT", "LEAD"} and by["jev-1.14"] == {"INTENT", "LEAD"}
    assert plan.stage == "pilot" and "EXPANDED PILOT" in "\n".join(smoke.render_plan(plan))
    # strict gating: Jev stays blocked until a smoke test confirms its response schema; every chat route is clear
    assert plan.blocked and all("jev-1.14" in b for b in plan.blocked_reasons)
    assert {b.split(" ")[0] + ":" + b.rsplit(": ", 1)[1] for b in plan.blocked_reasons} == {"jev-1.14:response_shape", "jev-1.14:usage_fields"}
    assert plan.total_upper_bound_usd <= plan.hard_cap_usd


def test_pilot_plan_blocked_participants_stay_visible_with_their_reason(budget, monkeypatch):
    from msu_llm_lab.models.registry import REGISTRY
    block_route(monkeypatch, "qwen3.5-35b-a3b")
    plan = smoke.build_plan(list(REGISTRY), [Task.INTENT], PROMPTS, budget, Decimal("0.25"), stage="pilot")
    assert any(r.model_key == "qwen3.5-35b-a3b" and r.blockers == ["route"] for r in plan.rows)
    assert any("qwen3.5-35b-a3b (INTENT): route" in b for b in plan.blocked_reasons)


def test_smoke_stage_ignores_participants_outside_the_pair_and_says_why(budget):
    plan = smoke.build_plan(["jev-1.14", "gpt-4o-mini"], [Task.INTENT], PROMPTS, budget)
    assert [r.model_key for r in plan.rows] == ["jev-1.14"] and any("gpt-4o-mini: not part of the two-model smoke test" in b for b in plan.blocked_reasons)
