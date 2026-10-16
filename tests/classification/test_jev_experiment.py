"""Jev prompt experiment: plan structure, gates bound to the plan, per-variant prompts reach the transport, checkpoint/resume, scoring and selection. Synthetic data and a fake transport only."""

import json

import pandas as pd
import pytest

from msu_llm_lab.classification.records import PredictionStore
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.labeled_eval import execute as X
from msu_llm_lab.labeled_eval import jev_experiment as J
from msu_llm_lab.labeled_eval import jev_experiment_score as JS
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.observability.events import EventStore
from msu_llm_lab.classification.smoke import build_smoke_adapters
from test_labeled_executor import _auth, ws  # noqa: F401  (fixtures)
from test_real_benchmark import budget, net  # noqa: F401

from synthetic_fixtures import SYN_INTENT_PROMPT  # noqa: E402

_V = J.make_variants(SYN_INTENT_PROMPT)
PROMPTS = {"V0": SYN_INTENT_PROMPT, "V1": _V["V1"], "V2": _V["V2"]}


@pytest.fixture
def exp(ws, budget, tmp_path, monkeypatch):
    cfg, _ = ws
    files = {k: tmp_path / f"{k}.md" for k in PROMPTS}
    for k, f in files.items():
        f.write_text(PROMPTS[k])
    monkeypatch.setattr(J, "variant_files", lambda c: files)
    J.exp_dir(cfg).mkdir(parents=True, exist_ok=True)
    J.priv_dir(cfg).mkdir(parents=True, exist_ok=True)
    cases = X.load_cases(cfg)["INTENT"]
    plan = J.build_plan(cfg, budget, cases=cases, prompts=PROMPTS)
    return cfg, budget, cases, plan


def test_plan_structure_halves_control_and_deterministic_fingerprint(exp, ws):
    cfg, budget, cases, plan = exp
    n = len(cases)
    assert plan["n_requests"] == 7 * (n // 2) or plan["n_requests"] > 0
    names = {u["variant"] for u in plan["units_full"]}
    assert names == {"V0", "V1", "V2", "V0-repeat"}
    rep = [u for u in plan["units_full"] if u["variant"] == "V0-repeat"]
    assert rep and all(u["half"] == "heldout" and u["prompt_key"] == "V0" for u in rep)       # noise control: V0 again, held-out only
    dev = {c for u in plan["units_full"] if u["half"] == "dev" for c in u["case_ids"]}
    held = {c for u in plan["units_full"] if u["half"] == "heldout" for c in u["case_ids"]}
    assert dev and held and not dev & held
    assert J.build_plan(cfg, budget, cases=cases, prompts=PROMPTS)["fingerprint"] == plan["fingerprint"]
    assert J.build_plan(cfg, budget, cases=cases, prompts=dict(PROMPTS, V1=PROMPTS["V1"] + "\n"))["fingerprint"] != plan["fingerprint"]
    assert len(set(plan["question_hashes"].values())) == 3


def test_gates_need_approval_and_permission_bound_to_this_plan_and_prompts(exp):
    cfg, budget, cases, plan = exp
    gid = lambda p: {g["id"]: g for g in J.check_gates(cfg, p, budget)}  # noqa: E731
    g = gid(plan)
    assert not g["E1"]["ok"] and not g["E2"]["ok"] and not g["E3"]["ok"] and g["E3b"]["ok"]   # no approval, no permission; the synthetic V0 is not the authoritative prompt; variants differ
    J.record_operator_approval(cfg, plan, operator="op", statement="I approve this exact plan")
    J.record_permission(cfg, plan, evidence_type="OPERATOR_CHAT_ATTESTATION", verified_by="op", evidence_reference="chat", approvers="NOT PROVIDED", approved_on="NOT PROVIDED", source="chat")
    g = gid(plan)
    assert g["E1"]["ok"] and g["E2"]["ok"] and not g["E2-evidence"]["ok"] and not g["E2-evidence"]["blocking"]
    other = dict(plan, fingerprint="000000000000")
    assert not gid(other)["E1"]["ok"] and not gid(other)["E2"]["ok"]
    changed = dict(plan, prompt_hashes=dict(plan["prompt_hashes"], V1="0" * 64))
    assert not gid(changed)["E2"]["ok"]
    with pytest.raises(ValueError):
        J.record_permission(cfg, plan, evidence_type="VERBAL", verified_by="op", evidence_reference="x", approvers="x", approved_on="x", source="x")
    with pytest.raises(ValueError):
        J.record_operator_approval(cfg, plan, operator="", statement="")


def test_executor_refuses_without_gates_and_sends_nothing(exp, net, tmp_path):
    cfg, budget, cases, plan = exp
    adapters = build_smoke_adapters([J.KEY], X.EXECUTION_STAGE)
    live = LiveAuthorization(models=(plan["participant"]["model_id"],))
    with pytest.raises(X.GateFailure):
        X.execute(plan, {"INTENT": cases}, PROMPTS, _auth_for(plan), live, adapters, budget, root=tmp_path / "st", reports_dir=tmp_path / "rep",
                  gate_fn=lambda: J.check_gates(cfg, plan, budget), events=EventStore(tmp_path / "ev.jsonl"))
    assert net.requests == []


def _auth_for(plan):
    from msu_llm_lab.classification import realdata
    mid = plan["participant"]["model_id"]
    return realdata.confirm_real_data(dataset_version=plan["dataset_version"], session_policy="t", models=[mid], routes={mid: plan["participant"]["route"]}, tasks=["INTENT"], plan_fingerprint=plan["fingerprint"],
                                      cost_ceiling_usd="1", provider_policy_status={mid: "UNKNOWN"}, recipients=plan["recipients"], operator="Test Operator", is_interactive=lambda: True,
                                      prompt=lambda m: realdata.expected_phrase(plan["dataset_version"], plan["fingerprint"]), echo=lambda s: None)


def test_each_variant_sends_its_own_prompt_stores_separately_and_resumes(exp, net, tmp_path):
    cfg, budget, cases, plan = exp
    adapters = build_smoke_adapters([J.KEY], X.EXECUTION_STAGE)
    live = LiveAuthorization(models=(plan["participant"]["model_id"],))
    ok_gate = lambda: [{"id": "T", "item": "t", "ok": True, "blocking": True, "detail": ""}]  # noqa: E731
    kw = dict(root=tmp_path / "st", reports_dir=tmp_path / "rep", gate_fn=ok_gate, events=EventStore(tmp_path / "ev.jsonl"))
    res = X.execute(plan, {"INTENT": cases}, PROMPTS, _auth_for(plan), live, adapters, budget, **kw)
    assert res.units_done == res.units_total and not res.quarantined and len(net.requests) == plan["n_requests"]
    bodies = [r.data.decode() for r in net.requests]
    n_of = lambda k: sum(len(u["case_ids"]) for u in plan["units_full"] if u["prompt_key"] == k)  # noqa: E731
    v1_mark, v2_mark = J.CLARIFY["HUMAN_HANDOFF_REQUEST"][:60], J.EXAMPLES[2][1].strip('"')[:30]
    assert sum(v1_mark in b for b in bodies) == n_of("V1") + n_of("V2")            # definitions text reaches the transport for V1 and V2 only
    assert sum(v2_mark in b for b in bodies) == n_of("V2")                          # example rows reach the transport for V2 only
    base = tmp_path / "rep" / plan["store_dir"] / "predictions"
    recs = {n: PredictionStore(base / f"{n}.jsonl").load() for n in ("V0", "V1", "V2", "V0-repeat")}
    assert all(recs.values()) and all(r.dataset_version == J.EXP_VERSION for v in recs.values() for r in v)
    assert {r.run_id for r in recs["V0-repeat"]} == {f"{J.EXP_VERSION}-V0-repeat"}
    before = len(net.requests)
    X.execute(plan, {"INTENT": cases}, PROMPTS, _auth_for(plan), live, adapters, budget, **kw)
    assert len(net.requests) == before                                  # checkpoint: nothing is re-sent
    wrong = dict(plan, fingerprint="000000000000")
    with pytest.raises(Exception):
        X.execute(wrong, {"INTENT": cases}, PROMPTS, _auth_for(plan), live, adapters, budget, **kw)


def test_scoring_selects_on_dev_only_and_reports_noise_control(exp, tmp_path):
    cfg, budget, cases, plan = exp
    key = pd.read_csv(cfg.private / R.VERSION / "answer_key.csv", dtype=str)
    key = key[key.task == "INTENT"].set_index("case_id").expected
    h = J.halves(cfg)
    base = J.exp_dir(cfg) / "predictions"
    base.mkdir(parents=True, exist_ok=True)

    def write(name, wrong_ids):
        rows = []
        for c in h["dev"] + h["heldout"]:
            lab = key[c] if c not in wrong_ids else "QUERY" if key[c] != "QUERY" else "GREETING"
            rows.append(json.dumps({"case_id": c, "status": "OK", "predicted_label": lab, "label_probs": {lab: 0.9}, "latency_ms": 100.0, "estimated_cost_usd": "0.0001"}))
        (base / f"{name}.jsonl").write_text("\n".join(rows) + "\n")

    write("V0", set(h["dev"][:4]) | set(h["heldout"][:1]))
    write("V1", set(h["dev"][:1]))
    write("V2", set(h["dev"][:3]))
    write("V0-repeat", set(h["dev"][:4]) | set(h["heldout"][:1]))
    r = JS.score(cfg)
    assert r["selected"] == "V1" and r["agreement"] == 1.0
    dev = r["results"][(r["results"].half == "dev")].set_index("variant")
    assert dev.loc["V1", "accuracy"] > dev.loc["V2", "accuracy"] > dev.loc["V0", "accuracy"]
    assert set(r["paired"].variant) == {"V1", "V2", "V0-repeat"} or set(r["paired"].variant) >= {"V1", "V2"}


def test_template_placeholders_are_rejected_when_recording_and_by_the_gates(exp):
    cfg, budget, cases, plan = exp
    with pytest.raises(ValueError):
        J.record_operator_approval(cfg, plan, operator="op", statement="<your approval of plan x>")
    with pytest.raises(ValueError):
        J.record_permission(cfg, plan, evidence_type="OPERATOR_CHAT_ATTESTATION", verified_by="op", evidence_reference="<what you rely on>", approvers="NOT PROVIDED", approved_on="x", source="x")
    # a record written before the check existed (or by hand) with template text must not satisfy the gate
    J.record_permission(cfg, plan, evidence_type="OPERATOR_CHAT_ATTESTATION", verified_by="op", evidence_reference="chat", approvers="NOT PROVIDED", approved_on="NOT PROVIDED", source="chat")
    p = J.priv_dir(cfg) / J.PERMISSION_FILE
    d = json.loads(p.read_text())
    d["evidence_reference"] = "<what you rely on>"
    p.write_text(json.dumps(d))
    assert not {g["id"]: g for g in J.check_gates(cfg, plan, budget)}["E2"]["ok"]
