"""Guarded executor for labeled-eval-v1: gate refusal, permission binding, caps, order, quarantine, resume. Synthetic data and a fake transport only."""

import json
import urllib.error
from decimal import Decimal

import pandas as pd
import pytest

from msu_llm_lab.classification import realdata
from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.classification.smoke import build_smoke_adapters
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.labeled_eval import data as D
from msu_llm_lab.labeled_eval import execute as X
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.labeled_eval import stage2
from msu_llm_lab.observability.events import EventStore
from synthetic_fixtures import SYN_INTENT_PROMPT, SYN_LEAD_PROMPT
from test_real_benchmark import SmartNet, budget, net  # noqa: F401  (fixtures)

PROMPTS = {Task.INTENT: SYN_INTENT_PROMPT, Task.LEAD: SYN_LEAD_PROMPT}
SMALL = ["jev-1.14", "gpt-4.1-nano"]


def _csvs(tmp_path, n=12):
    labs = list(D.INTENT_LABELS)
    pd.DataFrame({"case_id": [f"I{i}" for i in range(n)], "history": ["User: hi\nAssistant: hello" if i % 2 else "" for i in range(n)], "current_message": [f"message number {i}" for i in range(n)],
                  "expected_label": [labs[i % 5] for i in range(n)]}).to_csv(tmp_path / "i.csv", index=False)
    pd.DataFrame({"id": [f"L{i}" for i in range(n)], "action_summary": [f"Asked about course {i}" for i in range(n)], "user_msg": [f"msg {i}" for i in range(n)],
                  "expected_cta": ["true" if i % 2 else "false" for i in range(n)]}).to_csv(tmp_path / "l.csv", index=False)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    _csvs(tmp_path)
    cfg = R.Cfg(intent=str(tmp_path / "i.csv"), lead=str(tmp_path / "l.csv"), lead_alt=str(tmp_path / "none.csv"), out=str(tmp_path / "rep"), private=str(tmp_path / "priv"), prep_dir=str(tmp_path / "prep"))
    prep = R.prepare(cfg)
    man = prep["manifest"]
    stage2.label_attestation(cfg, man)
    X.record_decisions(cfg, man)
    return cfg, man


def _plan(ws, budget, keys=SMALL):
    cfg, _ = ws
    cases = X.load_cases(cfg)
    parts = [p for p in X._participants() if p["key"] in keys]
    return cases, X.build_plan(cfg, budget, cases=cases, prompts=PROMPTS, participants=parts)


def _perm(cfg, plan, **over):
    d = {"verified": True, "verified_by": "data owner", "evidence_reference": "ticket-1", "evidence_type": "DOCUMENT", "dataset_hashes": plan["datasets"], "models": [p["model_id"] for p in plan["participants"]],
         "routes": {p["model_id"]: p["route"] for p in plan["participants"]}, "tasks": ["INTENT", "LEAD"], "processing_path": "OpenRouter, routed to the pinned provider", "plan_fingerprint": plan["fingerprint"]}
    d.update(over)
    R._write_private(cfg.private / R.VERSION / X.PERMISSION_FILE, json.dumps(d))


def _auth(plan, phrase=None):
    return realdata.confirm_real_data(dataset_version=R.VERSION, session_policy="t", models=[p["model_id"] for p in plan["participants"]], routes={p["model_id"]: p["route"] or "none" for p in plan["participants"]},
                                      tasks=["INTENT", "LEAD"], plan_fingerprint=plan["fingerprint"], cost_ceiling_usd="1", provider_policy_status={p["model_id"]: "UNKNOWN" for p in plan["participants"]},
                                      recipients=plan["recipients"], operator="Test Operator", is_interactive=lambda: True,
                                      prompt=lambda m: phrase or realdata.expected_phrase(R.VERSION, plan["fingerprint"]), echo=lambda s: None)


def _run(ws, budget, plan, cases, tmp_path, gate_ok=True, auth=None, **kw):
    cfg, _ = ws
    adapters = build_smoke_adapters([p["key"] for p in plan["participants"]], X.EXECUTION_STAGE)
    live = LiveAuthorization(models=tuple(p["model_id"] for p in plan["participants"]))
    gate = (lambda: [{"id": "T", "item": "t", "ok": True, "blocking": True, "detail": ""}]) if gate_ok else (lambda: X.check_gates(cfg, plan, budget))
    return X.execute(plan, cases, PROMPTS, auth or _auth(plan), live, adapters, budget, root=tmp_path / "state", reports_dir=tmp_path / "rep", gate_fn=gate, events=EventStore(tmp_path / "ev.jsonl"), **kw)


def test_cases_are_classified_confidential_never_synthetic(ws):
    cfg, _ = ws
    cs = X.load_cases(cfg)
    assert all(c.classification.value == "CONFIDENTIAL" for t in cs.values() for c in t)
    dec = json.loads((cfg.out / R.VERSION / "decisions.json").read_text())
    assert dec["G2_data_classification"]["described_as_synthetic"] is False and dec["G2_data_classification"]["origin_verified"] is False
    assert dec["G3_lead_contract"]["user_msg_sent"] is False and dec["G3_lead_contract"]["production_parity"] == "PRODUCTION_PARITY_NOT_ESTABLISHED"


def test_missing_permission_blocks_live_execution_before_any_request(ws, budget, net, tmp_path):
    cfg, _ = ws
    cases, plan = _plan(ws, budget)
    gates = X.check_gates(cfg, plan, budget)
    assert not next(g for g in gates if g["id"] == "G1b")["ok"]
    with pytest.raises(X.GateFailure) as e:
        _run(ws, budget, plan, cases, tmp_path, gate_ok=False)
    assert "G1b" in str(e.value) and net.requests == []


def test_permission_must_name_the_dataset_hashes_models_and_routes(ws, budget):
    cfg, _ = ws
    _, plan = _plan(ws, budget)
    _perm(cfg, plan)
    assert next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == "G1b")["ok"]
    for over in ({"verified": False}, {"verified_by": ""}, {"evidence_reference": ""}, {"models": ["openai/gpt-4.1-nano"]}, {"routes": {"x": "y"}}, {"dataset_hashes": {"intent": "0", "lead": "0"}}):
        _perm(cfg, plan, **over)
        assert not next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == "G1b")["ok"], over


def test_raised_caps_are_refused(ws, budget):
    cfg, _ = ws
    _, plan = _plan(ws, budget)
    assert next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == "B1")["ok"] is False
    budget.policy = budget.policy.model_copy(update={"pilot_budget_usd": Decimal("10.00"), "per_run_ceiling_usd": Decimal("0.75")})
    assert next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == "B1")["ok"] is False
    budget.policy = budget.policy.model_copy(update={"per_run_ceiling_usd": Decimal("0.50")})
    assert next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == "B1")["ok"] is True


def test_changed_output_caps_or_lead_user_msg_fail_their_gates(ws, budget, tmp_path):
    cfg, _ = ws
    _, plan = _plan(ws, budget)
    bad = dict(plan, output_caps={"INTENT": 64, "LEAD": 24})
    assert not next(g for g in X.check_gates(cfg, bad, budget) if g["id"] == "G6")["ok"]
    p = cfg.private / R.VERSION / "inference_inputs_lead.jsonl"
    rows = [json.loads(l) for l in open(p)]
    rows[0]["user_msg"] = "leak"
    R._write_private(p, "\n".join(json.dumps(r) for r in rows) + "\n")
    assert not next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == "G3")["ok"]


def test_authorisation_for_another_plan_or_model_set_is_refused(ws, budget, net, tmp_path):
    cases, plan = _plan(ws, budget)
    with pytest.raises(realdata.LiveNotAuthorizedError):
        _run(ws, budget, plan, cases, tmp_path, auth=_auth(plan, phrase="wrong"))
    other = dict(plan, fingerprint="deadbeef0000")
    with pytest.raises(realdata.LiveNotAuthorizedError):
        _run(ws, budget, plan, cases, tmp_path, auth=_auth(other))
    assert net.requests == []


def test_execution_sends_every_case_once_records_every_outcome_and_resumes_without_duplicates(ws, budget, net, tmp_path):
    cases, plan = _plan(ws, budget)
    res = _run(ws, budget, plan, cases, tmp_path)
    n = sum(len(v) for v in cases.values())
    assert res.units_done == res.units_total and not res.quarantined and len(net.requests) == 2 * n
    recs = PredictionStore(X.prediction_path(tmp_path / "rep", "jev-1.14", "INTENT")).load()
    assert {r.case_id for r in recs} == {f"I{i}" for i in range(12)} and all(r.dataset_version == R.VERSION for r in recs)
    before = len(net.requests)
    _run(ws, budget, plan, cases, tmp_path)
    assert len(net.requests) == before                              # checkpoint: nothing is re-sent
    stored = json.loads((tmp_path / "state" / "exec_state.json").read_text())
    assert stored["fingerprint"] == plan["fingerprint"]


def test_a_failing_participant_is_quarantined_and_never_replaced(ws, budget, net, tmp_path):
    cases, plan = _plan(ws, budget)
    net.fail_jev = lambda: urllib.error.HTTPError("https://x", 404, "nf", None, __import__("io").BytesIO(b"{}"))
    res = _run(ws, budget, plan, cases, tmp_path)
    assert "typesafe/jev-1.14" in res.quarantined and "openai/gpt-4.1-nano" not in res.quarantined
    nano = PredictionStore(X.prediction_path(tmp_path / "rep", "gpt-4.1-nano", "INTENT")).load()
    assert len(nano) == 12                                           # the other participant completed; no substitution


def test_the_budget_gate_stops_before_any_request(ws, budget, net, tmp_path):
    cases, plan = _plan(ws, budget)
    plan = dict(plan, reserve_usd="999")                             # headroom minus reserve is negative: nothing may be sent
    res = _run(ws, budget, plan, cases, tmp_path)
    assert net.requests == [] and "BUDGET_GATE" in res.halted


def test_unresolved_spend_halts_before_dispatch(ws, budget, net, tmp_path):
    cases, plan = _plan(ws, budget)
    budget.record_unresolved_spend(transaction_id="tx_x", model_id="openai/gpt-4.1-nano", provider="openrouter", description="t", run_id="r")
    res = _run(ws, budget, plan, cases, tmp_path)
    assert net.requests == [] and "unresolved" in res.halted


def test_mistral_is_ordered_last_and_the_plan_is_deterministic(ws, budget):
    cfg, _ = ws
    parts = X._participants()
    assert parts[-1]["key"] == "mistral-small-3.2-24b-instruct" and len(parts) == 9
    cases = X.load_cases(cfg)
    a = X.build_plan(cfg, budget, cases=cases, prompts=PROMPTS)
    b = X.build_plan(cfg, budget, cases=cases, prompts=PROMPTS)
    assert a["fingerprint"] == b["fingerprint"] and a["n_requests"] == (12 + 12) * 8


def test_operator_approval_is_bound_to_the_fingerprint_and_never_satisfies_the_permission_gate(ws, budget):
    cfg, _ = ws
    _, plan = _plan(ws, budget)
    gid = lambda i: next(g for g in X.check_gates(cfg, plan, budget) if g["id"] == i)["ok"]  # noqa: E731
    assert not gid("G1c")
    X.record_operator_approval(cfg, plan)
    assert gid("G1c") and not gid("G1b")                       # approval alone never verifies the processing permission
    assert not next(g for g in X.check_gates(cfg, dict(plan, fingerprint="000000000000"), budget) if g["id"] == "G1c")["ok"]


def test_recorded_attestation_satisfies_g1b_but_is_flagged_as_not_a_document(ws, budget):
    cfg, _ = ws
    _, plan = _plan(ws, budget)
    with pytest.raises(ValueError):
        X.record_processing_permission(cfg, plan, evidence_type="DOCUMENT", verified_by="", evidence_reference="x", approvers="NOT PROVIDED", approved_on="NOT PROVIDED", source="s")
    with pytest.raises(ValueError):
        X.record_processing_permission(cfg, plan, evidence_type="VERBAL", verified_by="a", evidence_reference="x", approvers="a", approved_on="a", source="s")
    rec = X.record_processing_permission(cfg, plan, evidence_type="OPERATOR_CHAT_ATTESTATION", verified_by="op", evidence_reference="chat", approvers="NOT PROVIDED", approved_on="NOT PROVIDED", source="chat")
    assert rec["limitations"] and "NOT a verification" in rec["meaning"]
    gates = {g["id"]: g for g in X.check_gates(cfg, plan, budget)}
    assert gates["G1b"]["ok"] and not gates["G1b-evidence"]["ok"] and not gates["G1b-evidence"]["blocking"]
    other = dict(plan, fingerprint="000000000000")
    assert not {g["id"]: g for g in X.check_gates(cfg, other, budget)}["G1b"]["ok"]
