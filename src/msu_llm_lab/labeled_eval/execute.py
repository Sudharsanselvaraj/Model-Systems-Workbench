"""Guarded executor for the labelled evaluation (labeled-eval-v1).

Nothing here sends a request unless EVERY blocking gate passes, the plan fingerprint matches, a typed live approval and a typed real-data authorisation
(bound to this dataset version, models, routes, tasks and plan) were issued in this process, and every existing budget guard allows the next chunk.
No cap is read from a setting that this module can change: the effective policy must equal the recorded one or the run is refused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from msu_llm_lab.classification import realdata
from msu_llm_lab.classification.benchmark import _case_bound, _state_bytes
from msu_llm_lab.classification.chat_adapter import MAX_OUTPUT_TOKENS
from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.classification.runner import ClassificationRunner
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.labeled_eval import budget_plan as BP
from msu_llm_lab.labeled_eval import cases as C
from msu_llm_lab.labeled_eval import data as D
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.models import registry
from msu_llm_lab.observability.events import EventStore
from msu_llm_lab.providers.base import LiveNotAuthorizedError

DECISION_SCHEMA = "labeled-eval-decisions-v1"
PERMISSION_FILE = "processing_permission.json"
RECORDED_CAPS = {"total_budget_usd": "10.00", "pilot_budget_usd": "10.00", "per_run_ceiling_usd": "0.50", "per_case_ceiling_usd": "0.02"}
CAP_TOKENS = {"INTENT": 16, "LEAD": 24}
MAX_CONSECUTIVE = 3
EXECUTION_STAGE = "pilot"


class GateFailure(LiveNotAuthorizedError):
    def __init__(self, failed: List[Dict[str, Any]]):
        self.failed = failed
        super().__init__("live execution refused, gates not satisfied: " + "; ".join(f"{g['id']} {g['item']}" for g in failed))


def _gate(id_, item, ok, detail, blocking=True):
    return {"id": id_, "item": item, "ok": bool(ok), "detail": detail, "blocking": blocking}


# ------------------------------------------------------------------ decisions (recorded from the operator's statements in chat)
def record_decisions(cfg: R.Cfg, man: Dict[str, Any]) -> Dict[str, Any]:
    d = {"schema": DECISION_SCHEMA, "dataset_version": R.VERSION, "recorded_from": "operator statements in chat, 2026-10-16", "operator": "Sudharsanselvaraj",
         "datasets": {"intent": {"sha256": man["datasets"]["intent"]["sha256"], "cases": 120}, "lead": {"sha256": man["datasets"]["lead"]["sha256"], "cases": 200}},
         "G1_operator_approval": {"approved": True, "scope": "this evaluation: 120 Main Intent cases, 200 Lead Qualifier cases, the eight-participant cohort",
                                  "condition": "subject to verification that the applicable data-processing permissions permit the selected external provider routes",
                                  "caps_unchanged": RECORDED_CAPS, "reserve_usd": str(BP.RESERVE_USD), "note": "Approval does not establish provenance or provider data policies."},
         "G2_data_classification": {"classification": "CONFIDENTIAL_POTENTIALLY_REAL", "origin_verified": False, "described_as_synthetic": False, "described_as_anonymised": False,
                                    "basis": "text identity with prep-v2 messages (114/120 and 196/200); origin to be verified by the data owner"},
         "G3_lead_contract": {"input": "action_summary", "user_msg_sent": False, "file": "Intent:lead dataset/lead_qualification_test_cases (1).csv", "rows": 200, "production_parity": "PRODUCTION_PARITY_NOT_ESTABLISHED",
                              "reason": "Dify export unavailable"},
         "G6_output_limits": {"max_output_tokens": CAP_TOKENS, "change_during_run": False, "empty_outputs": "counted as failures; finish_reason and token usage recorded",
                              "alternative_caps_or_reasoning": "separate pre-registered experiment only"},
         "G7_mistral": {"position": "last", "retry_policy": "existing (max 3 attempts)", "circuit_breaker": "existing (3 consecutive errors)", "http_429": "kept separate from invalid_output (status ERROR/RATE_LIMIT vs INVALID_OUTPUT)",
                        "on_failure": "stop the participant, document the incomplete evaluation, no substitution"},
         "G8_executor": {"refuses_without": ["processing permission verified", "typed live approval", "typed real-data authorisation bound to labeled-eval-v1"]},
         "cohort": BP.COHORT}
    (cfg.out / R.VERSION / "decisions.json").write_text(json.dumps(d, indent=2, sort_keys=True))
    R._write_private(cfg.private / R.VERSION / "decisions.json", json.dumps(d, indent=2, sort_keys=True))
    return d


def record_operator_approval(cfg: R.Cfg, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Operator's explicit approval for the real-request evaluation, bound to one plan fingerprint. It does NOT verify processing permission (gate G1b)."""
    a = {"schema": "labeled-eval-operator-approval-v1", "operator": "Sudharsanselvaraj", "recorded_from": "operator statement in chat, 2026-10-16", "plan_fingerprint": plan["fingerprint"],
         "dataset_hashes": plan["datasets"], "models": sorted(p["model_id"] for p in plan["participants"]), "routes": {p["model_id"]: p["route"] for p in plan["participants"]},
         "tasks": ["INTENT", "LEAD"], "caps": RECORDED_CAPS, "reserve_usd": plan["reserve_usd"], "guard_bound_total_usd": plan["guard_bound_total_usd"],
         "statement": "Explicit approval to proceed with the real-request evaluation of 120 Main Intent and 200 Lead Qualifier cases across the eight-participant cohort, within the existing limits.",
         "does_not_verify": ["processing permission for any provider or route (G1b)", "dataset provenance", "provider retention, training or processing policies"],
         "typed_approvals_still_required": ["live approval phrase", "real-data authorisation phrase"]}
    (cfg.out / R.VERSION / "operator_approval.json").write_text(json.dumps(a, indent=2, sort_keys=True))
    R._write_private(cfg.private / R.VERSION / "operator_approval.json", json.dumps(a, indent=2, sort_keys=True))
    return a


def record_processing_permission(cfg: R.Cfg, plan: Dict[str, Any], *, evidence_type: str, verified_by: str, evidence_reference: str, approvers: str, approved_on: str, source: str,
                                 searched: Optional[List[str]] = None) -> Dict[str, Any]:
    """Write the permission record exactly as the operator states it. Every field is supplied by the operator; nothing is defaulted or inferred.
    `evidence_type` is DOCUMENT or OPERATOR_CHAT_ATTESTATION; an attestation is never described as a document or as a verified provider policy."""
    if evidence_type not in ("DOCUMENT", "OPERATOR_CHAT_ATTESTATION"):
        raise ValueError("evidence_type must be DOCUMENT or OPERATOR_CHAT_ATTESTATION")
    for name, val in (("verified_by", verified_by), ("evidence_reference", evidence_reference), ("approvers", approvers), ("approved_on", approved_on), ("source", source)):
        if not str(val).strip():
            raise ValueError(f"{name} is required (state 'NOT PROVIDED' if it was not given; it is recorded as such)")
    rec = {"schema": "labeled-eval-processing-permission-v1", "verified": True, "verified_by": verified_by, "evidence_type": evidence_type, "evidence_reference": evidence_reference,
           "approvers": approvers, "approved_on": approved_on, "source": source, "dataset_hashes": plan["datasets"], "models": sorted(p["model_id"] for p in plan["participants"]),
           "routes": {p["model_id"]: p["route"] for p in plan["participants"]}, "tasks": ["INTENT", "LEAD"], "processing_path": "OpenRouter, routed to the pinned provider (fallbacks disabled)",
           "plan_fingerprint": plan["fingerprint"], "documents_searched_without_result": searched or [],
           "meaning": "Documented or attested APPROVAL by the responsible parties to process these datasets through these routes. It is NOT a verification of any provider's retention, training or data-processing policy; those stay UNKNOWN in the registry.",
           "limitations": [] if evidence_type == "DOCUMENT" else ["no permission document was located", "approver identities may be unnamed", "scope is attested by the operator, not independently confirmed"]}
    R._write_private(cfg.private / R.VERSION / PERMISSION_FILE, json.dumps(rec, indent=2, sort_keys=True))
    (cfg.out / R.VERSION / PERMISSION_FILE).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def _participants(order_last: str = "mistral-small-3.2-24b-instruct") -> List[Dict[str, Any]]:
    tasks_by_key: Dict[str, List[str]] = {}
    for t, keys in BP.COHORT.items():
        for k in keys:
            tasks_by_key.setdefault(k, []).append(t)
    keys = sorted(tasks_by_key, key=lambda k: (k == order_last, k))
    out = []
    for k in keys:
        rec = registry.get_record(k)
        pr = rec.pinned_route()
        out.append({"key": k, "model_id": rec.requested_model_id.value, "tasks": sorted(tasks_by_key[k]), "route": rec.pinned_tag, "protocol": rec.protocol.value,
                    "route_listed_provider": getattr(rec.route, "listed_provider", None), "fallbacks_disabled": bool(pr is not None)})
    return out


def _listed_provider(key: str) -> str:
    """Provider name shown to the operator in the authorisation text. Display only: not part of the plan fingerprint."""
    rec = registry.get_record(key)
    pr = rec.pinned_route()
    return str(pr.provider if pr else rec.provider_route.value)


def load_cases(cfg: R.Cfg) -> Dict[str, List[Any]]:
    inp = BP._inputs(cfg)
    return {"INTENT": [C.intent_case(r) for r in inp["INTENT"]], "LEAD": [C.lead_case(r) for r in inp["LEAD"]]}


def load_prompts() -> Dict[Task, str]:
    return {Task.INTENT: Path("main_intent_classifier.md").read_text(), Task.LEAD: Path("lead_qualification_prompt.md").read_text()}


def build_plan(cfg: R.Cfg, budget: BudgetService, cases: Optional[Dict[str, List[Any]]] = None, prompts: Optional[Dict[Task, str]] = None, participants: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Pure planning. Opens the ledger for read-only status; sends nothing."""
    cases = cases or load_cases(cfg)
    prompts = prompts or load_prompts()
    parts = participants or _participants()
    man = json.loads((cfg.out / R.VERSION / "classifier_evaluation_manifest.json").read_text())
    pb = {t.value: len(p.encode()) for t, p in prompts.items()}
    units: List[Dict[str, Any]] = []
    for p in parts:
        prices = registry.model_level_prices(p["model_id"])
        for t in p["tasks"]:
            cs = cases[t]
            bounds = [_case_bound(Task(t), _state_bytes(Task(t), c), pb[t], prices, CAP_TOKENS[t]) for c in cs]
            for i in range(0, len(cs), BP.BLOCK):
                ids = [c.case_id for c in cs[i:i + BP.BLOCK]]
                units.append({"unit": f"{p['key']}|{t}|{i // BP.BLOCK}", "key": p["key"], "model_id": p["model_id"], "task": t, "chunk": i // BP.BLOCK, "case_ids": ids,
                              "bound_usd": str(sum(bounds[i:i + BP.BLOCK], Decimal("0"))), "max_case_bound_usd": str(max(bounds[i:i + BP.BLOCK]))})
    body = {"version": R.VERSION, "datasets": {k: man["datasets"][k]["sha256"] for k in ("intent", "lead")}, "participants": parts,
            "prompts": {t.value: hashlib.sha256(p.encode()).hexdigest()[:12] for t, p in prompts.items()}, "output_caps": CAP_TOKENS, "caps": RECORDED_CAPS, "chunk": BP.BLOCK,
            "max_attempts": BP.MAX_ATTEMPTS, "reserve_usd": str(BP.RESERVE_USD), "units": [{k: u[k] for k in ("unit", "case_ids", "bound_usd")} for u in units],
            "request_overrides": {p["key"]: json.dumps(registry.request_overrides(p["model_id"]), sort_keys=True, default=str) for p in parts if hasattr(registry, "request_overrides")}}
    fp = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]
    st = budget.get_status()
    total_bound = sum((Decimal(u["bound_usd"]) for u in units), Decimal("0"))
    return {**body, "fingerprint": fp, "units_full": units, "guard_bound_total_usd": str(total_bound), "n_requests": sum(len(u["case_ids"]) for u in units),
            "recipients": {p["model_id"]: f"OpenRouter, routed to {_listed_provider(p['key'])}" + (f" (pinned: {p['route']})" if p["route"] else "") for p in parts},
            "ledger_at_plan_time": {"remaining_pilot_usd": str(st.remaining_pilot_usd), "remaining_total_usd": str(st.remaining_total_usd), "unresolved": st.unresolved_spend_count}}


# ------------------------------------------------------------------ gates
def check_gates(cfg: R.Cfg, plan: Dict[str, Any], budget: BudgetService) -> List[Dict[str, Any]]:
    out = cfg.out / R.VERSION
    priv = cfg.private / R.VERSION
    g: List[Dict[str, Any]] = []

    def read(p: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    man = read(out / "classifier_evaluation_manifest.json") or {}
    dec = read(out / "decisions.json")
    ok_dec = bool(dec and dec.get("schema") == DECISION_SCHEMA and dec.get("G1_operator_approval", {}).get("approved") and dec["datasets"]["intent"]["sha256"] == plan["datasets"]["intent"]
                  and dec["datasets"]["lead"]["sha256"] == plan["datasets"]["lead"])
    g.append(_gate("G1a", "operator approval recorded for exactly these datasets", ok_dec, "decisions.json present and bound to the dataset hashes" if ok_dec else "decisions.json missing or not bound to the current dataset hashes"))
    appr = read(out / "operator_approval.json")
    ok_appr = bool(appr and appr.get("plan_fingerprint") == plan["fingerprint"] and appr.get("dataset_hashes") == plan["datasets"] and appr.get("caps") == RECORDED_CAPS)
    g.append(_gate("G1c", "operator approval bound to this plan fingerprint, datasets and unchanged caps", ok_appr, "approval recorded for this fingerprint" if ok_appr else "no operator approval recorded for this exact plan"))
    perm = read(priv / PERMISSION_FILE)
    models = sorted(p["model_id"] for p in plan["participants"])
    routes = {p["model_id"]: p["route"] for p in plan["participants"]}
    ok_perm = bool(perm and perm.get("verified") is True and perm.get("verified_by") and perm.get("evidence_reference") and perm.get("evidence_type") in ("DOCUMENT", "OPERATOR_CHAT_ATTESTATION")
                   and perm.get("dataset_hashes") == plan["datasets"] and sorted(perm.get("models", [])) == models and perm.get("routes") == routes
                   and sorted(perm.get("tasks", [])) == ["INTENT", "LEAD"] and "OpenRouter" in str(perm.get("processing_path", "")) and perm.get("plan_fingerprint") == plan["fingerprint"])
    g.append(_gate("G1b", "data-processing permission recorded for this dataset, these models, routes, tasks and the OpenRouter path", ok_perm,
                   "permission record matches the plan (dataset hashes, models, routes, tasks, OpenRouter path, fingerprint)" if ok_perm else
                   f"{PERMISSION_FILE} missing or does not cover the dataset hashes, models, pinned routes, both tasks, the OpenRouter path and this plan fingerprint (needs verified, verified_by, evidence_type, evidence_reference)"))
    is_doc = bool(perm and perm.get("evidence_type") == "DOCUMENT")
    g.append(_gate("G1b-evidence", "permission evidence is a document (not only an attestation)", is_doc,
                   f"evidence_type={perm.get('evidence_type') if perm else 'none'}: " + ("a document" if is_doc else "operator chat attestation; no document located; provider policies remain UNKNOWN. Accepted as recorded, not independently verifiable by the executor"), blocking=False))
    ok_cls = bool(dec and dec["G2_data_classification"]["classification"] == "CONFIDENTIAL_POTENTIALLY_REAL" and not dec["G2_data_classification"]["described_as_synthetic"]
                  and not dec["G2_data_classification"]["described_as_anonymised"])
    g.append(_gate("G2", "datasets classified CONFIDENTIAL (potentially real); origin unverified is recorded", ok_cls, "classification recorded; origin_verified=False (documented)" if ok_cls else "classification not recorded"))
    lead_rows = [json.loads(l) for l in open(priv / "inference_inputs_lead.jsonl")] if (priv / "inference_inputs_lead.jsonl").exists() else []
    ok_lead = bool(dec and dec["G3_lead_contract"]["input"] == "action_summary" and not dec["G3_lead_contract"]["user_msg_sent"] and lead_rows and all(set(r) == {"case_id", "task", "split", "action_summary"} for r in lead_rows)
                   and len(lead_rows) == 200 and dec["G3_lead_contract"]["production_parity"] == "PRODUCTION_PARITY_NOT_ESTABLISHED")
    g.append(_gate("G3", "Lead input is action_summary only (200 rows); parity recorded as not established", ok_lead, "inference inputs hold only case_id, task, split, action_summary" if ok_lead else "Lead inputs or decision do not match"))
    sep = read(out / "separation_verification.json")
    from msu_llm_lab.labeled_eval import stage2
    live_sep = stage2.separation_check(cfg, man) if man else {"ok": False}
    g.append(_gate("G5a", "dataset hashes unchanged; answer key separate from inference inputs", live_sep["ok"], "all separation checks pass" if live_sep["ok"] else "a separation check failed"))
    att = read(out / "label_attestation.json")
    g.append(_gate("G4", "expected labels attested by the operator", bool(att and att.get("status") == "OPERATOR_ATTESTED_APPROVED"), "attestation present; kappa unavailable (no second annotation)", blocking=False))
    caps_ok = plan["output_caps"] == CAP_TOKENS == dict(MAX_OUTPUT_TOKENS_STR())
    g.append(_gate("G6", "output caps are the existing 16/24 tokens and unchanged", caps_ok, f"plan caps {plan['output_caps']}, adapter caps {MAX_OUTPUT_TOKENS_STR()}"))
    order = [p["key"] for p in plan["participants"]]
    g.append(_gate("G7", "Mistral is scheduled last", order[-1] == "mistral-small-3.2-24b-instruct", f"order tail: {order[-2:]}"))
    pol = budget.policy
    eff = {"total_budget_usd": str(pol.total_budget_usd), "pilot_budget_usd": str(pol.pilot_budget_usd), "per_run_ceiling_usd": str(pol.per_run_ceiling_usd), "per_case_ceiling_usd": str(pol.per_case_ceiling_usd)}
    g.append(_gate("B1", "effective caps equal the recorded caps (none raised)", {k: Decimal(v) for k, v in eff.items()} == {k: Decimal(v) for k, v in RECORDED_CAPS.items()}, f"effective {eff}"))
    st = budget.get_status()
    head = min(st.remaining_pilot_usd, st.remaining_total_usd)
    avail = head - Decimal(plan["reserve_usd"])
    g.append(_gate("B2", "no unresolved spend", st.unresolved_spend_count == 0, f"{st.unresolved_spend_count} unresolved entries"))
    g.append(_gate("B3", "guard bound of the whole plan fits the headroom minus the reserve", Decimal(plan["guard_bound_total_usd"]) <= avail, f"bound ${plan['guard_bound_total_usd']} vs headroom ${head} - reserve ${plan['reserve_usd']} = ${avail}"))
    over_run = [u["unit"] for u in plan["units_full"] if Decimal(u["bound_usd"]) > pol.per_run_ceiling_usd]
    over_case = [u["unit"] for u in plan["units_full"] if Decimal(u["max_case_bound_usd"]) > pol.per_case_ceiling_usd]
    g.append(_gate("B4", "every 10-case chunk is within the per-run cap and every case within the per-case cap", not over_run and not over_case, f"chunks over cap {over_run[:3]}, cases over cap {over_case[:3]}"))
    bad = {}
    for p in plan["participants"]:
        rec = registry.get_record(p["key"])
        b = registry.synthetic_live_blockers(p["model_id"], EXECUTION_STAGE) + [x for x in rec.student_data_blockers() if x != "data_policy"]
        if not p["fallbacks_disabled"] or not p["route"]:
            b.append("route not pinned")
        if b:
            bad[p["key"]] = b
    g.append(_gate("R1", "registry readiness and pinned routes for every participant (data_policy stays UNKNOWN and is covered only by the typed authorisation)", not bad, f"blockers: {bad}" if bad else "all participants ready"))
    return g


def MAX_OUTPUT_TOKENS_STR() -> Dict[str, int]:
    return {t.value: n for t, n in MAX_OUTPUT_TOKENS.items()}


# ------------------------------------------------------------------ execution
@dataclass
class ExecResult:
    units_done: int
    units_total: int
    halted: Optional[str]
    quarantined: Dict[str, str]
    units: Dict[str, Dict[str, Any]]
    spent_guard_usd: Decimal


def _state_path(root: Path) -> Path:
    return Path(root) / "exec_state.json"


def load_state(root: Path, fingerprint: str) -> Dict[str, Any]:
    p = _state_path(root)
    if p.exists():
        s = json.loads(p.read_text())
        if s.get("fingerprint") != fingerprint:
            raise LiveNotAuthorizedError("a checkpoint from a different plan exists; refusing to resume it")
        return s
    return {"fingerprint": fingerprint, "units": {}, "quarantined": {}, "streak": {}, "spent_guard_usd": "0", "halted": None}


def prediction_path(reports: Path, key: str, task: str) -> Path:
    return Path(reports) / R.VERSION / "predictions" / f"{key}__{task}.jsonl"


def execute(plan: Dict[str, Any], cases: Dict[str, List[Any]], prompts: Dict[Task, str], auth: realdata.RealDataAuthorization, live: LiveAuthorization, adapters: Dict[str, object],
            budget: BudgetService, *, root: Path, reports_dir: Path, gate_fn: Callable[[], List[Dict[str, Any]]], events: Optional[EventStore] = None,
            progress: Callable[[str], None] = lambda s: None) -> ExecResult:
    failed = [g for g in gate_fn() if g["blocking"] and not g["ok"]]
    if failed:
        raise GateFailure(failed)
    if not realdata.is_issued(auth):
        raise LiveNotAuthorizedError("a real-data authorisation issued by the typed confirmation in this process is required")
    dv = plan.get("dataset_version", R.VERSION)               # an experiment plan binds the typed authorisation to its own dataset version
    if auth.plan_fingerprint != plan["fingerprint"] or auth.dataset_version != dv:
        raise LiveNotAuthorizedError("the authorisation is bound to a different plan or dataset version")
    model_ids = {p["model_id"] for p in plan["participants"]}
    if not model_ids <= set(auth.models) or not model_ids <= set(live.models):
        raise LiveNotAuthorizedError("the authorisations do not cover every participant in the plan")
    by_id = {t: {c.case_id: c for c in cs} for t, cs in cases.items()}
    state = load_state(root, plan["fingerprint"])
    spent = Decimal(state["spent_guard_usd"])
    ev = events if events is not None else EventStore()
    st0 = budget.get_status()
    hard = min(st0.remaining_pilot_usd, st0.remaining_total_usd) - Decimal(plan["reserve_usd"])
    try:
        for u in plan["units_full"]:
            if state["units"].get(u["unit"], {}).get("status") in ("DONE", "QUARANTINED", "SKIPPED_QUARANTINE"):
                continue
            if u["model_id"] in state["quarantined"]:
                state["units"][u["unit"]] = {"status": "SKIPPED_QUARANTINE"}
                continue
            bound = Decimal(u["bound_usd"])
            st = budget.get_status()
            if spent + bound > hard or bound > st.remaining_pilot_usd or bound > st.remaining_total_usd or st.unresolved_spend_count:
                state["halted"] = f"BUDGET_GATE_BEFORE_{u['unit']}: guard spent {spent} + bound {bound} vs available {hard}; ledger remaining pilot {st.remaining_pilot_usd}, total {st.remaining_total_usd}; unresolved {st.unresolved_spend_count}"
                break
            chunk = [by_id[u["task"]][cid] for cid in u["case_ids"]]
            store = PredictionStore(prediction_path(reports_dir, u["key"], u["task"]) if "store_name" not in u else Path(reports_dir) / plan.get("store_dir", R.VERSION) / "predictions" / f"{u['store_name']}.jsonl")
            runner = ClassificationRunner(adapters, store, budget, stage=EXECUTION_STAGE, telemetry=ev)
            progress(f"{u['unit']}: {len(chunk)} requests")
            try:
                s = runner.run(task=Task(u["task"]), cases=chunk, model_id=u["model_id"], prompt_text=prompts[u["prompt_key"]] if "prompt_key" in u else prompts[Task(u["task"])], authorization=live,
                               run_id=u.get("run_id", f"{R.VERSION}-{u['key']}-{u['task']}"), dataset_version=dv, hard_cap_usd=hard - spent, real_data=auth)
            except (BudgetExceededError, LiveNotAuthorizedError) as e:
                state["halted"] = f"{type(e).__name__}: {e}"
                _save(root, state, spent)
                raise
            except Exception as e:  # noqa: BLE001  quarantine the participant, never substitute another model
                state["quarantined"][u["model_id"]] = f"{type(e).__name__}: {str(e)[:200]} ({u['unit']})"
                state["units"][u["unit"]] = {"status": "FAILED", "reason": state["quarantined"][u["model_id"]]}
                _save(root, state, spent)
                continue
            spent += s.guard_cost_usd
            state["units"][u["unit"]] = {"status": "DONE" if not s.aborted_reason else "QUARANTINED", "n_cases": len(chunk), "ok": s.n_ok, "invalid": s.n_invalid, "error": s.n_error, "guard_cost_usd": str(s.guard_cost_usd), "aborted": s.aborted_reason}
            streak = state["streak"]
            streak[u["model_id"]] = (streak.get(u["model_id"], 0) + s.n_error) if (s.n_ok + s.n_invalid == 0 and s.n_error) else 0
            if streak[u["model_id"]] >= MAX_CONSECUTIVE and not s.aborted_reason:
                s.aborted_reason = f"{streak[u['model_id']]} consecutive errors across chunks"
            if s.aborted_reason:
                state["units"][u["unit"]]["status"] = "QUARANTINED"
                state["quarantined"][u["model_id"]] = f"circuit breaker: {s.aborted_reason} ({u['unit']})"
            _save(root, state, spent)
        else:
            state["halted"] = None
    except KeyboardInterrupt:
        state["halted"] = "INTERRUPTED_BY_OPERATOR (resumable)"
        _save(root, state, spent)
        raise
    _save(root, state, spent)
    done = sum(1 for v in state["units"].values() if v.get("status") in ("DONE", "QUARANTINED", "SKIPPED_QUARANTINE", "FAILED"))
    return ExecResult(done, len(plan["units_full"]), state["halted"], dict(state["quarantined"]), dict(state["units"]), spent)


def _save(root: Path, state: Dict[str, Any], spent: Decimal) -> None:
    state["spent_guard_usd"] = str(spent)
    realdata.write_json_atomic(_state_path(root), state)


def save_plan(cfg: R.Cfg, plan: Dict[str, Any]) -> Path:
    p = cfg.out / R.VERSION / "plan.json"
    if p.exists() and json.loads(p.read_text()).get("fingerprint") != plan["fingerprint"]:
        raise ValueError(f"a different plan {json.loads(p.read_text()).get('fingerprint')} is already stored; review it or choose a new output folder (nothing is overwritten)")
    realdata.write_json_atomic(p, {k: v for k, v in plan.items()})        # case ids and bounds only; no text
    return p


def render_plan(plan: Dict[str, Any], gates: List[Dict[str, Any]]) -> List[str]:
    lines = [f"labeled-eval {R.VERSION}   plan {plan['fingerprint']}   requests {plan['n_requests']}   guard bound ${plan['guard_bound_total_usd']}   chunks of {plan['chunk']}",
             f"datasets: intent {plan['datasets']['intent'][:12]}  lead {plan['datasets']['lead'][:12]}   output caps {plan['output_caps']}   caps {plan['caps']}   reserve ${plan['reserve_usd']}",
             "order: " + ", ".join(p["key"] for p in plan["participants"]) + "   (Mistral last)"]
    lines += [f"  recipient {m}: {r}" for m, r in plan["recipients"].items()]
    lines += ["GATES:"] + [f"  [{'PASS' if g['ok'] else 'FAIL'}] {g['id']} {g['item']}: {g['detail']}" for g in gates]
    return lines


def write_gate_report(cfg: R.Cfg, plan: Dict[str, Any], gates: List[Dict[str, Any]], cost: Dict[str, Any]) -> None:
    out = cfg.out / R.VERSION
    pd.DataFrame(gates).to_csv(out / "live_run_gates.csv", index=False)
    failed = [g for g in gates if g["blocking"] and not g["ok"]]
    t, v, pol = cost["totals"], cost["verdict"], cost["policy"]
    md = ["# Gates before the first API request", "", f"Plan `{plan['fingerprint']}`: {plan['n_requests']} requests, 8 participants per task, Mistral last, 10-case chunks, output caps {plan['output_caps']}.", "",
          "## Gate status (checked now by the same code the executor uses)", "", "| Gate | Item | Status | Detail |", "|---|---|---|---|"]
    md += [f"| {g['id']} | {g['item']} | {'PASS' if g['ok'] else ('FAIL' if g['blocking'] else 'NOTE')} | {g['detail'][:240]} |" for g in gates]
    md += ["", "## Remaining blockers", ""]
    md += [f"- **{g['id']} {g['item']}.** {g['detail']}" for g in failed] or ["- none"]
    md += ["", "## Not gates, but required at run time", "- Interactive typed live approval bound to the plan fingerprint, and the typed real-data authorisation bound to `labeled-eval-v1`, the models, routes, tasks, plan and ceiling. Neither can be given by a script or a non-interactive shell.",
           "- The provider API key set in your own shell (run `api-preflight` to check; it is never printed).",
           "", "## Estimated cost before live execution",
           f"- Expected ${t['expected_usd']:.3f}; with observed retries ${t['expected_with_observed_retries_usd']:.3f}; guard bound ${t['guard_bound_usd']:.3f}; absolute worst (3 billed attempts every request) ${t['worst_case_3_attempts_usd']:.3f} (does not fit, theoretical).",
           f"- Headroom ${v['headroom_usd']:.3f} (spent ${pol['spent_total_usd']} of ${pol['total_ceiling_usd']}); reserve ${pol['reserve_usd']} kept; caps unchanged (total ${pol['total_ceiling_usd']}, pilot ${pol['pilot_ceiling_usd']}, per run ${pol['per_run_ceiling_usd']}, per case ${pol['per_case_ceiling_usd']}).",
           "", "## Accepted limitations (recorded, not blockers)",
           "- Data origin unverified: both datasets are classified CONFIDENTIAL (potentially real), never described as synthetic or anonymised.",
           "- Lead input is `action_summary` only; `user_msg` is never sent; PRODUCTION_PARITY_NOT_ESTABLISHED (Dify export unavailable).",
           "- Labels are an operator attestation; no second annotation, so reviewer agreement and kappa are unavailable.",
           "- Claude 16/24-token caps are kept: empty outputs (hidden reasoning consuming the cap) count as failures; alternative caps or reasoning settings are a separate pre-registered experiment.",
           "- Mistral runs last with the existing retry policy and circuit breaker; HTTP 429 (status ERROR) is kept separate from invalid-format outputs; if blocked it is stopped and documented, never replaced.",
           "- Provider data policies are UNKNOWN for every route; the typed authorisation does not verify them.", ""]
    (out / "gates_before_first_request.md").write_text("\n".join(md))
