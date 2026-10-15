"""Benchmark report: what actually ran, per participant and task, from persisted records, events and the stored plan. No student text.

Completion states are strict. COMPLETE_FOR_PLANNED_SCOPE needs every planned request to have been attempted; anything less is PARTIAL, and
a participant that failed, was quarantined, was blocked or was never executed says so with its reason. Accuracy-type metrics are pending
unless independently approved labels exist; model agreement is reported on valid common outputs only and is never correctness.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from msu_llm_lab.classification import benchmark as bm
from msu_llm_lab.classification.records import PredictionRecord, PredictionStore, Task
from msu_llm_lab.observability import analysis as A
from msu_llm_lab.observability.aggregate import percentile
from msu_llm_lab.observability.events import EventStore
from msu_llm_lab.observability.labels import PENDING, load_approved_labels

REPORT_SCHEMA = "real-benchmark-report-v1"


def _d(x) -> Optional[str]:
    return None if x is None else str(x)


def _unit_stats(recs: List[PredictionRecord], events, run_id: str) -> Dict[str, Any]:
    ok = [r for r in recs if r.status == "OK"]
    inv = [r for r in recs if r.status == "INVALID_OUTPUT"]
    err = [r for r in recs if r.status == "ERROR"]
    lat = [r.latency_ms for r in recs if r.latency_ms is not None and r.status != "ERROR"]
    mlat = [r.model_latency_ms for r in recs if r.model_latency_ms is not None]
    tok = [r for r in recs if r.prompt_tokens is not None and r.completion_tokens is not None]
    act = [r.actual_cost_usd for r in recs if r.actual_cost_usd is not None]
    est = [r.estimated_cost_usd for r in recs if r.estimated_cost_usd is not None]
    nat = [r.native_cost for r in recs if r.native_cost is not None]
    ev = [e for e in events if e.run_id == run_id and e.provenance == "LIVE"]
    http = Counter(str(e.http_status) for e in ev if e.kind == "REQUEST_FAILED" and e.http_status is not None)
    return {
        "attempted": len(recs), "valid": len(ok), "invalid_output": len(inv), "errors": len(err),
        "valid_output_rate": (len(ok) / len(recs)) if recs else None,
        "error_classes": dict(Counter(r.error_type for r in err)), "http_status": dict(http),
        "timeouts": sum(1 for e in ev if e.kind == "REQUEST_FAILED" and e.timed_out), "retried_requests": sum(1 for r in recs if (r.attempts or 1) > 1),
        "retry_events": sum(1 for e in ev if e.kind == "REQUEST_RETRY"),
        "resolved_models": sorted({r.resolved_model_id for r in recs if r.resolved_model_id}), "serving_providers": sorted({r.serving_provider for r in recs if r.serving_provider}),
        "prompt_fingerprints": sorted({r.prompt_version for r in recs}),
        "latency_ms_end_to_end": {"n": len(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95) if len(lat) >= 20 else None, "p99": percentile(lat, 99) if len(lat) >= 20 else None,
                                  "note": "client-measured; P95/P99 withheld below 20 samples"},
        "provider_model_latency_ms": {"n": len(mlat), "p50": percentile(mlat, 50) if mlat else None, "note": "reported only when the provider supplied it" if mlat else "not reported by provider"},
        "tokens": {"requests_with_reported_tokens": len(tok), "prompt": sum(r.prompt_tokens for r in tok), "completion": sum(r.completion_tokens for r in tok),
                   "requests_without_tokens": len(recs) - len(tok)},
        "cost": {"actual_usd": _d(sum(act, Decimal("0"))) if act else None, "actual_n": len(act), "estimated_usd": _d(sum(est, Decimal("0"))) if est else None, "estimated_n": len(est),
                 "native_credits": _d(sum(nat, Decimal("0"))) if nat else None, "native_n": len(nat), "native_unit": "credits; USD ratio UNVERIFIED, never converted",
                 "unknown_cost_requests": sum(1 for r in recs if r.cost_basis == "UNAVAILABLE" and r.status != "ERROR"), "coverage": f"{len(act) + len(est)} of {len(recs)} requests have a USD figure (actual or estimated)"},
    }


def build_report(plan: Dict[str, Any], state: Dict[str, Any], reports_dir: Path, events: EventStore, prep_dir: Optional[Path] = None) -> Dict[str, Any]:
    bid = plan["bench_id"]
    all_events = events.read_all()
    labels = load_approved_labels(prep_dir)
    preds: List[PredictionRecord] = []
    units: List[Dict[str, Any]] = []
    participants: List[Dict[str, Any]] = []
    n_blocks = plan["ordering"]["n_blocks"]
    done_blocks = state.get("blocks_completed", 0)
    for p in plan["participants"]:
        rows = []
        store_bid = p.get("store_bench") or bid
        for tname in p["tasks"]:
            path = bm.prediction_path(reports_dir, store_bid, p["key"], tname)
            recs = PredictionStore(path).load() if path.exists() else []
            preds += recs
            planned = p["per_task"][tname]["requests_full"]
            st = _unit_stats(recs, all_events, f"{store_bid}-{p['key']}-{tname}")
            st.update({"task": tname, "planned_full_dataset": planned, "unattempted": max(0, planned - len(recs)), "excluded_per_case_cap": p["per_task"][tname]["excluded_per_case_cap"]})
            rows.append(st)
            units.append({"model_key": p["key"], "model_id": p["model_id"], **st})
        attempted = sum(r["attempted"] for r in rows)
        planned_total = sum(r["planned_full_dataset"] for r in rows)
        quarantined = state.get("quarantined", {}).get(p["model_id"])
        if p["readiness"]["blockers"]:
            status, reason = "BLOCKED", "; ".join(p["readiness"]["blockers"])
        elif attempted == 0:
            status, reason = ("NOT_EXECUTED_QUARANTINED" if quarantined else "NOT_EXECUTED"), quarantined or "no request was dispatched (plan not approved or halted before this participant)"
        elif quarantined:
            status, reason = "QUARANTINED_PARTIAL", quarantined
        elif attempted >= planned_total and done_blocks >= n_blocks:
            status, reason = "COMPLETE_FOR_PLANNED_SCOPE", "every planned request was attempted"
        else:
            status, reason = "PARTIAL", f"{attempted} of {planned_total} planned requests attempted (blocks completed {done_blocks} of {n_blocks}); {state.get('halted') or 'stopped'}"
        participants.append({"key": p["key"], "model_id": p["model_id"], "display_name": p["display_name"], "roles": p["roles"], "tasks": p["tasks"], "route": p["route"],
                             "data_policy": p["data_handling"]["data_policy"], "completion_state": status, "reason": reason, "units": rows})
    runs = A.latest_runs(preds)
    agree = {t.value: A.agreement_report({k: v for k, v in runs.items() if k[1] == t.value}, t, labels) for t in (Task.INTENT, Task.LEAD)}
    exec_states = Counter(x["completion_state"] for x in participants)
    full = all(x["completion_state"] == "COMPLETE_FOR_PLANNED_SCOPE" for x in participants) and done_blocks >= n_blocks
    return {
        "schema": REPORT_SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(), "bench_id": bid, "plan_fingerprint": plan["fingerprint"],
        "benchmark_state": "FULL_PLANNED_SCOPE_COMPLETE" if full else "PARTIAL_OR_INCOMPLETE", "blocks": {"completed": done_blocks, "planned": n_blocks, "halted": state.get("halted")},
        "dataset": plan["dataset"], "prompts": plan["prompts"], "limits": plan["limits"], "evaluation_contract_version": plan["evaluation_contract_version"],
        "participants": participants, "completion_counts": dict(exec_states), "guard_cost_usd": state.get("spent_guard_usd"),
        "coverage_cases": {t: {"eligible": plan["dataset"]["eligible"][t], "scheduled_blocks_completed": min(done_blocks * plan["ordering"]["block_size"], plan["dataset"]["eligible"][t])} for t in ("INTENT", "LEAD")},
        "agreement": {t: {"models": r.models, "n_cases_all_models": r.n_cases_all_models, "pairs": [{"a": x.a, "b": x.b, "n_common": x.n_common, "agreement": x.agreement, "kappa": x.kappa,
                                                                                                         "n_unusable_excluded": x.n_unusable, "note": x.note} for x in r.pairs],
                       "disagreement_case_ids": [d[0] for d in r.disagreements[:50]], "note": "agreement is not correctness; invalid or errored outputs are excluded, not counted as disagreements"}
                      for t, r in agree.items()},
        "accuracy": {"status": labels.status, "approved_labels": {"intent": len(labels.intent), "lead": len(labels.lead)},
                     "metrics": f"accuracy, macro-F1, per-class P/R/F1, confusion, Brier and ECE: {PENDING if not labels.available else 'compute with the dashboard or export from approved labels'}"},
        "caveats": ["Operator authorization does not verify any provider's retention, training or data-processing policy; every route's policy status is the registry value (UNKNOWN unless a primary source verified it).",
                    "PRODUCTION_PARITY_NOT_ESTABLISHED: the Dify workflow export is missing.", "Pseudonymization is not anonymization; redaction is pattern-based without named-entity recognition.",
                    "Native provider cost is in credits with an unverified USD ratio and is never converted to USD."],
    }


def render_markdown(r: Dict[str, Any]) -> str:
    L = [f"# Real-data benchmark report: {r['bench_id']}", "", f"- State: **{r['benchmark_state']}** (blocks {r['blocks']['completed']} of {r['blocks']['planned']}; halted: {r['blocks']['halted'] or 'no'})",
         f"- Plan fingerprint: `{r['plan_fingerprint']}`   evaluation contract: `{r['evaluation_contract_version']}`   generated: {r['generated_at']}",
         f"- Dataset preparation: `{r['dataset']['version']}` (session policy {r['dataset']['session_policy']}), parity `{r['dataset']['parity_status']}`",
         f"- Eligible cases: {r['dataset']['eligible']}; excluded: {r['dataset']['excluded']}; coverage so far: {r['coverage_cases']}",
         "- Prompts: " + "; ".join(f"{k} `{v['version']}` (sha256 `{v['sha256'][:16]}…`)" for k, v in r["prompts"].items()),
         f"- Caps: per case ${r['limits']['per_case_usd']}, per run ${r['limits']['per_run_usd']}, pilot ${r['limits']['pilot_usd']}, total ${r['limits']['total_usd']}, hard cap ${r['limits']['hard_cap_usd']}; guard cost so far ${r['guard_cost_usd']}",
         "", "## Participant coverage (all ten)", "", "| Participant | Roles | Tasks | Route | Data policy | State | Reason |", "|---|---|---|---|---|---|---|"]
    for p in r["participants"]:
        L.append(f"| {p['display_name']} | {', '.join(p['roles'])} | {', '.join(p['tasks'])} | {p['route']['pinned_tag'] or 'none'} | {p['data_policy']} | **{p['completion_state']}** | {p['reason']} |")
    L += ["", "## Operational and protocol metrics", "", "| Participant | Task | Planned (full) | Attempted | Valid | Invalid | Errors | Unattempted | Valid rate | E2E P50 ms | P95 ms | Model latency | Tokens (in/out) | Actual USD | Estimated USD | Native credits |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    dash = lambda v, f="{}": "—" if v is None else f.format(v)  # noqa: E731
    for p in r["participants"]:
        for u in p["units"]:
            L.append(f"| {p['display_name']} | {u['task']} | {u['planned_full_dataset']} | {u['attempted']} | {u['valid']} | {u['invalid_output']} | {u['errors']} | {u['unattempted']} | "
                     f"{dash(u['valid_output_rate'] and round(100 * u['valid_output_rate'], 1), '{}%')} | {dash(u['latency_ms_end_to_end']['p50'], '{:.0f}')} | {dash(u['latency_ms_end_to_end']['p95'], '{:.0f}')} | "
                     f"{dash(u['provider_model_latency_ms']['p50'], '{:.0f}') if u['provider_model_latency_ms']['n'] else 'not reported by provider'} | {u['tokens']['prompt']}/{u['tokens']['completion']} ({u['tokens']['requests_with_reported_tokens']} reqs) | "
                     f"{dash(u['cost']['actual_usd'])} | {dash(u['cost']['estimated_usd'])} | {dash(u['cost']['native_credits'])} |")
    L += ["", "Error classes, HTTP statuses, timeouts and retries per participant and task:", ""]
    for p in r["participants"]:
        for u in p["units"]:
            if u["errors"] or u["retried_requests"] or u["timeouts"]:
                L.append(f"- {p['display_name']} {u['task']}: errors {u['error_classes']} http {u['http_status']} timeouts {u['timeouts']} retried requests {u['retried_requests']} (retry events {u['retry_events']})")
    L += ["", "Resolved model and serving provider (as reported):", ""]
    for p in r["participants"]:
        for u in p["units"]:
            if u["attempted"]:
                L.append(f"- {p['display_name']} {u['task']}: resolved {u['resolved_models'] or '—'}; provider {u['serving_providers'] or '—'}; prompt {u['prompt_fingerprints']}")
    L += ["", "## Model agreement (valid common outputs only; not correctness)", ""]
    for t, a in r["agreement"].items():
        L.append(f"### {t}: {len(a['models'])} participants with valid outputs; {a['n_cases_all_models']} cases answered validly by all")
        for x in a["pairs"]:
            L.append(f"- {x['a']} vs {x['b']}: n={x['n_common']} agreement {dash(x['agreement'] and round(100 * x['agreement'], 1), '{}%')} kappa {dash(x['kappa'] and round(x['kappa'], 3))} excluded unusable {x['n_unusable_excluded']} {x['note']}")
    L += ["", "## Accuracy and calibration", "", f"- Ground truth: **{r['accuracy']['status']}**. {r['accuracy']['metrics']}", "", "## Caveats", ""] + [f"- {c}" for c in r["caveats"]]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ comparison bundle (CSV + traces + manifest)
BUNDLE_FILES = ("comparison_report.md", "operational_ranking.csv", "observability_report.md", "observability.json", "model_results.csv", "model_summary.csv", "model_comparison.csv", "model_agreement_pairs.csv",
                "unattempted_in_executed_blocks.csv", "request_reconciliation.json", "traces.jsonl", "benchmark_manifest.json")  # plus models/<participant>.md and diagnostics/*
SUMMARY_COLUMNS = ["model_key", "model_id", "display_name", "roles", "task", "origin", "route_pinned", "data_policy", "completion_state", "planned_full_dataset", "attempted", "valid",
                   "invalid_output", "errors", "unattempted_in_executed_blocks", "unattempted_rest_of_full_dataset", "valid_output_rate", "binary_decisions", "probability_only_results", "timeouts", "retried_requests", "e2e_latency_n",
                   "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms", "provider_latency_n", "provider_latency_p50_ms", "requests_with_tokens", "prompt_tokens", "completion_tokens", "actual_usd",
                   "actual_n", "estimated_usd", "estimated_n", "native_credits", "est_usd_per_request", "est_usd_per_1000_cases", "requests_without_cost_evidence", "errors_no_response", "unknown_cost_after_response",
                   "resolved_models", "serving_providers", "prompt_fingerprints", "reason"]
COST_SEMANTICS = {
    "requests_without_cost_evidence": "records with no actual USD, no estimated USD and no native credit figure (includes errors that returned no response)",
    "errors_no_response": "requests that ended in an error and returned no usable response; no cost evidence exists for them. This is NOT proof they cost nothing; a provider could still have metered them",
    "unknown_cost_after_response": "a response arrived but neither a cost nor token counts allowed any figure; these raise unresolved-spend ledger entries",
    "ledger_unresolved_entries": "separate: unresolved-spend entries on the budget ledger (counted in the manifest, not per row)",
    "rule": "an empty cost cell means not measured, never zero",
}
INTENT_LABELS = ["GREETING", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST", "QUERY"]


def _csv(rows, cols) -> bytes:
    from msu_llm_lab.observability.export import csv_bytes
    return csv_bytes(rows, cols, flat=True)


def _executed_cases(plan, loaded, blocks_done: int, t: Task):
    return [c.case_id for c in bm.order_cases(loaded, t, plan["ordering"].get("sampling", "hash"))[: blocks_done * plan["ordering"]["block_size"]]]


def _unattempted(plan: Dict[str, Any], loaded, reports_dir: Path, blocks_done: int, cap_excluded: Optional[Dict[tuple, set]] = None) -> List[Dict[str, Any]]:
    """Cases inside the blocks that were actually executed (not merely planned) that have no stored record for a NEW participant/task. Opaque ids only.

    Carried-over participants are not listed: their completed results were not re-sent on purpose. Cases a participant was excluded from by the
    per-case cost cap are not 'unattempted' either; they are excluded by policy."""
    if loaded is None:
        return []
    out = []
    for t in (Task.INTENT, Task.LEAD):
        scheduled = _executed_cases(plan, loaded, blocks_done, t)
        for p in plan["participants"]:
            if t.value not in p["tasks"] or (p.get("store_bench") and p.get("inherited_completed_blocks", 0) >= blocks_done):
                continue
            path = bm.prediction_path(reports_dir, p.get("store_bench") or plan["bench_id"], p["key"], t.value)
            have = {r.case_id for r in PredictionStore(path).load()} if path.exists() else set()
            skip = (cap_excluded or {}).get((p["key"], t.value), set())
            for cid in scheduled:
                if cid not in have and cid not in skip:
                    out.append({"model_key": p["key"], "task": t.value, "case_id": cid, "status": "NOT_ATTEMPTED"})
    return out


def _origin(plan: Dict[str, Any], p: Dict[str, Any], done: Optional[int] = None) -> str:
    if not p.get("store_bench"):
        return "NEW_THIS_PLAN"
    inh = p.get("inherited_completed_blocks", 0)
    if done is None or inh >= done:
        return f"CARRIED_OVER_FROM:{p['store_bench']}"
    return f"CARRIED_OVER_FROM:{p['store_bench']} (blocks 0-{inh - 1}) + NEW_THIS_PLAN (later blocks)"


def reconcile(plan: Dict[str, Any], state: Dict[str, Any], reports_dir: Path, events: Sequence[Any], loaded, unatt: List[Dict[str, Any]], cap_excluded: Dict[tuple, set]) -> Dict[str, Any]:
    """One table that ties planned logical requests to dispatched requests, outcomes, provider attempts, retries and unattempted cases."""
    done = state.get("blocks_completed", 0)
    bs = plan["ordering"]["block_size"]
    units: List[Dict[str, Any]] = []
    un_by = Counter((u["model_key"], u["task"]) for u in unatt)
    for p in plan["participants"]:
        inh = p.get("inherited_completed_blocks", 0) if p.get("store_bench") else 0
        sb = p.get("store_bench") or plan["bench_id"]
        for t in p["tasks"]:
            path = bm.prediction_path(reports_dir, sb, p["key"], t)
            recs = PredictionStore(path).load() if path.exists() else []
            rid = f"{sb}-{p['key']}-{t}"
            if loaded is not None:
                order = bm.order_cases(loaded, Task(t), plan["ordering"].get("sampling", "hash"))
                carried_cases = {c.case_id for c in order[: min(inh, done) * bs]}
                sched = [c.case_id for c in order[: done * bs] if c.case_id not in carried_cases and c.case_id not in cap_excluded.get((p["key"], t), set())]
                new_recs = [r for r in recs if r.case_id not in carried_cases]
                carried_recs = [r for r in recs if r.case_id in carried_cases]
                new_set = set(sched) | {r.case_id for r in new_recs}
                ev = [e for e in events if e.run_id == rid and e.provenance == "LIVE" and e.case_id in new_set]
                planned_new = len(sched)
            else:
                carried_cases, new_recs, carried_recs = set(), (recs if inh < done else []), (recs if inh >= done else [])
                ev = [e for e in events if e.run_id == rid and e.provenance == "LIVE"] if inh < done else []
                planned_new = None
            attempts = sum((r.attempts or 1) for r in new_recs)
            units.append({
                "model_key": p["key"], "task": t, "origin": _origin(plan, p) if inh else "NEW_THIS_PLAN", "carried_over_cases": len(carried_cases) if loaded is not None else None,
                "carried_over_not_resent": inh >= done and bool(inh),
                "planned_new_logical_requests": planned_new, "dispatched_logical_requests": sum(e.kind == "REQUEST_STARTED" for e in ev),
                "outcomes_recorded": len(new_recs), "carried_over_records": len(carried_recs),
                "ok": sum(r.status == "OK" for r in new_recs), "invalid_output": sum(r.status == "INVALID_OUTPUT" for r in new_recs), "error": sum(r.status == "ERROR" for r in new_recs),
                "provider_http_attempts": attempts, "retries": attempts - len(new_recs),
                "unattempted": un_by.get((p["key"], t), 0), "quarantined": p["model_id"] in state.get("quarantined", {}),
            })
    new = units
    tot = {k: sum((u[k] or 0) for u in new) for k in ("planned_new_logical_requests", "dispatched_logical_requests", "outcomes_recorded", "ok", "invalid_output", "error", "provider_http_attempts", "retries", "unattempted")}
    tot["carried_over_records_not_resent"] = sum(u["carried_over_records"] for u in units)
    tot["rows_in_results_files"] = tot["outcomes_recorded"] + tot["carried_over_records_not_resent"]
    tot["plan_block_requests_for_executed_blocks"] = sum(b["requests"] for b in plan["blocks"][:done])
    checks = {
        "planned_new_equals_dispatched_plus_unattempted": tot["planned_new_logical_requests"] == tot["dispatched_logical_requests"] + tot["unattempted"],
        "dispatched_equals_outcomes": tot["dispatched_logical_requests"] == tot["outcomes_recorded"],
        "outcomes_equal_ok_invalid_error": tot["outcomes_recorded"] == tot["ok"] + tot["invalid_output"] + tot["error"],
        "attempts_equal_outcomes_plus_retries": tot["provider_http_attempts"] == tot["outcomes_recorded"] + tot["retries"],
        "planned_matches_plan_block_requests": tot["planned_new_logical_requests"] == tot["plan_block_requests_for_executed_blocks"] if loaded is not None else None,
    }
    return {"blocks_completed": done, "units": units, "totals": tot, "checks": checks,
            "explanation": ("Planned new logical requests are what the plan scheduled for the executed blocks. Dispatched = requests actually started. A participant quarantined by its circuit "
                            "breaker leaves the rest of its scheduled cases unattempted. One logical request can use several provider HTTP attempts (retries). Carried-over participants "
                            "keep results from the earlier benchmark and are not re-sent.")}


def _distribution(stats: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for k, v in stats.items():
        if "INTENT" in v:
            d = v["INTENT"]["predicted_distribution"]
            out[k] = {l: d.get(l, 0) for l in INTENT_LABELS}
    return out


def write_bundle(plan: Dict[str, Any], state: Dict[str, Any], reports_dir: Path, events: EventStore, roster, out_root: Path, prep_dir: Optional[Path] = None,
                 loaded=None, manifest_path: Optional[Path] = None, exported_at: Optional[str] = None, ledger: Optional[Dict[str, Any]] = None,
                 prompts: Optional[Dict[Task, str]] = None) -> Dict[str, Any]:
    """Write the comparison files for one benchmark into a NEW folder (never overwrites). Redacted by default; no student text; nulls stay empty."""
    import hashlib

    from msu_llm_lab.classification import benchmark_detail as bd, diagnostics as dg
    from msu_llm_lab.observability import export as X

    report = build_report(plan, state, reports_dir, events, prep_dir)
    bid = plan["bench_id"]
    all_events = events.read_all()
    done = state.get("blocks_completed", 0)
    preds = []
    rec_origin: Dict[tuple, str] = {}
    for p in plan["participants"]:
        inh = p.get("inherited_completed_blocks", 0) if p.get("store_bench") else 0
        for t in p["tasks"]:
            path = bm.prediction_path(reports_dir, p.get("store_bench") or bid, p["key"], t)
            if path.exists():
                recs_ = PredictionStore(path).load()
                preds += recs_
                carried_cases = ({c.case_id for c in bm.order_cases(loaded, Task(t), plan["ordering"].get("sampling", "hash"))[: min(inh, done) * plan["ordering"]["block_size"]]}
                                 if (loaded is not None and inh) else set())
                for r in recs_:
                    carried = bool(inh) and ((r.case_id in carried_cases) if loaded is not None else inh >= done)
                    rec_origin[(r.run_id, r.case_id)] = _origin(plan, p) if carried else "NEW_THIS_PLAN"
    labels = load_approved_labels(prep_dir)
    sel = X.ExportSelection(runs=tuple(sorted({r.run_id for r in preds})) or None)
    ts = X.build_trace_rows(preds, all_events, labels, {}, sel)
    for r in ts.rows:
        r["markers"]["origin"] = rec_origin.get((r["identity"]["run_id"], r["identity"]["case_id"]))
    # cases excluded from a participant by the per-case cost cap are excluded by policy, not unattempted
    cap_excluded: Dict[tuple, set] = {}
    if loaded is not None and prompts:
        adapters = bm.build_smoke_adapters([p["key"] for p in plan["participants"]], bm.STAGE)
        for p in plan["participants"]:
            for t in p["tasks"]:
                chunk = [c for c in bm.order_cases(loaded, Task(t), plan["ordering"].get("sampling", "hash"))[: done * plan["ordering"]["block_size"]]]
                over = bm._over_cap_ids(plan, p, Task(t), chunk, prompts, adapters)
                if over:
                    cap_excluded[(p["key"], t)] = over
    unatt = _unattempted(plan, loaded, reports_dir, done, cap_excluded)
    recon = reconcile(plan, state, reports_dir, all_events, loaded, unatt, cap_excluded)

    stats_by = bd.collect(plan, reports_dir)
    stats = {k: {t: bd._unit(rs) for t, rs in v.items()} for k, v in stats_by.items()}
    pmap = {p["key"]: p for p in plan["participants"]}
    un_exec = {(x["model_key"], x["task"]): x["unattempted"] for x in recon["units"]}
    summary_rows = []
    for p in report["participants"]:
        for u in p["units"]:
            st = stats[p["key"]][u["task"]]
            c = st["cost"]
            recs_u = stats_by[p["key"]][u["task"]]
            lead = u["task"] == "LEAD"
            binary = sum(1 for r in recs_u if r.status == "OK" and (r.predicted_need_cta is not None)) if lead else None
            prob_only = sum(1 for r in recs_u if r.status == "OK" and r.predicted_need_cta is None and r.yes_prob is not None) if lead else None
            no_evidence = sum(1 for r in recs_u if r.actual_cost_usd is None and r.estimated_cost_usd is None and r.native_cost is None)
            est_per = (Decimal(c["estimated_usd"]) / c["estimated_n"]) if c["estimated_n"] else None
            summary_rows.append({
                "model_key": p["key"], "model_id": p["model_id"], "display_name": p["display_name"], "roles": "|".join(p["roles"]), "task": u["task"], "origin": _origin(plan, pmap[p["key"]], done),
                "route_pinned": p["route"]["pinned_tag"], "data_policy": p["data_policy"], "completion_state": p["completion_state"], "planned_full_dataset": u["planned_full_dataset"],
                "attempted": u["attempted"], "valid": u["valid"], "invalid_output": u["invalid_output"], "errors": u["errors"], "unattempted_in_executed_blocks": un_exec.get((p["key"], u["task"]), 0), "unattempted_rest_of_full_dataset": u["unattempted"], "valid_output_rate": u["valid_output_rate"],
                "binary_decisions": binary, "probability_only_results": prob_only, "timeouts": u["timeouts"], "retried_requests": u["retried_requests"], "e2e_latency_n": u["latency_ms_end_to_end"]["n"],
                "e2e_p50_ms": u["latency_ms_end_to_end"]["p50"], "e2e_p95_ms": u["latency_ms_end_to_end"]["p95"], "e2e_p99_ms": u["latency_ms_end_to_end"]["p99"],
                "provider_latency_n": u["provider_model_latency_ms"]["n"], "provider_latency_p50_ms": u["provider_model_latency_ms"]["p50"],
                "requests_with_tokens": u["tokens"]["requests_with_reported_tokens"], "prompt_tokens": u["tokens"]["prompt"] if u["tokens"]["requests_with_reported_tokens"] else None,
                "completion_tokens": u["tokens"]["completion"] if u["tokens"]["requests_with_reported_tokens"] else None, "actual_usd": u["cost"]["actual_usd"], "actual_n": u["cost"]["actual_n"],
                "estimated_usd": u["cost"]["estimated_usd"], "estimated_n": u["cost"]["estimated_n"], "native_credits": u["cost"]["native_credits"],
                "est_usd_per_request": (f"{est_per:.6f}" if est_per is not None else None), "est_usd_per_1000_cases": (f"{est_per * 1000:.4f}" if est_per is not None else None),
                "requests_without_cost_evidence": no_evidence, "errors_no_response": st["error"], "unknown_cost_after_response": c["unknown"],
                "resolved_models": "|".join(u["resolved_models"]), "serving_providers": "|".join(u["serving_providers"]), "prompt_fingerprints": "|".join(u["prompt_fingerprints"]), "reason": p["reason"]})
    cols, comp = X.comparison_rows(ts.rows)
    disagreements = [r for r in comp if r.get("models_agree") is False]
    dis_by_task = Counter(r["task"] for r in disagreements)
    pairs = [{"task": t, "model_a": x["a"], "model_b": x["b"], "n_common_valid": x["n_common"], "agreement": x["agreement"], "cohens_kappa": x["kappa"],
              "n_unusable_excluded": x["n_unusable_excluded"], "note": x["note"]} for t, a in report["agreement"].items() for x in a["pairs"]]
    # matrix completeness: required cases remain unattempted => not finished
    incomplete = sorted({(u["model_key"], u["task"]) for u in unatt})
    matrix = {"complete_for_executed_blocks": not incomplete and not state.get("quarantined"),
              "incomplete_units": [{"model_key": m, "task": t, "unattempted": recon["units"][[(x["model_key"], x["task"]) for x in recon["units"]].index((m, t))]["unattempted"]} for m, t in incomplete],
              "quarantined": state.get("quarantined", {}),
              "statement": ("The model/task matrix is NOT finished: required cases remain unattempted." if incomplete else "Every planned case of the executed blocks was attempted.")}
    sample_hints = None
    if loaded is not None:
        sampled = set(_executed_cases(plan, loaded, done, Task.INTENT))
        sample_hints = {"sampled_intent_cases": len(sampled), "with_selection_hints": sum(1 for cid in sampled if (loaded.strata.get("INTENT", {}).get(cid) or (0, 0, False))[2]),
                        "note": "selection hints are phrase heuristics (handoff / upload status / out of scope wording) set during preparation; they are not labels"}
    dist = _distribution(stats)

    ranking_rows = []
    for t in ("INTENT", "LEAD"):
        units = [r for r in summary_rows if r["task"] == t and int(r["attempted"] or 0) > 0]
        for metric, key, rev in (("valid_output_rate", "valid_output_rate", True), ("median_e2e_latency_ms", "e2e_p50_ms", False), ("estimated_usd_per_1000_cases", "est_usd_per_1000_cases", False)):
            have = [u for u in units if u[key] not in (None, "")]
            have.sort(key=lambda u: float(u[key]), reverse=rev)
            prev, rank = None, 0
            for pos, u in enumerate(have, 1):
                if prev is None or float(u[key]) != prev:
                    rank, prev = pos, float(u[key])          # equal values share a rank: a tie is not an ordering
                ranking_rows.append({"task": t, "ranking": "OPERATIONAL (not accuracy)", "metric": metric, "rank": rank, "model_key": u["model_key"], "value": u[key], "n_requests": u["attempted"],
                                     "origin": u["origin"], "sample_note": "" if int(u["attempted"]) >= 20 else "n < 20: indicative only"})
    route_failures = {}
    for p in plan["participants"]:
        recs_all = [r for tt in p["tasks"] for r in stats_by[p["key"]][tt]]
        codes = Counter(c for r in recs_all for c in r.retry_errors)
        errs = [r for r in recs_all if r.status == "ERROR"]
        q = state.get("quarantined", {}).get(p["model_id"])
        if codes or errs or q:
            route_failures[p["key"]] = {"route": p["route"]["pinned_tag"], "request_overrides": p.get("request_overrides") or {}, "http_or_attempt_codes": dict(codes), "failed_requests": len(errs),
                                        "quarantined": q, "origin": _origin(plan, pmap[p["key"]], done)}
    files = {
        "operational_ranking.csv": _csv(ranking_rows, ["task", "ranking", "metric", "rank", "model_key", "value", "n_requests", "origin", "sample_note"]),
        "model_results.csv": X.csv_bytes(ts.rows), "model_summary.csv": _csv(summary_rows, SUMMARY_COLUMNS), "model_comparison.csv": _csv(comp, cols),
        "model_agreement_pairs.csv": _csv(pairs, ["task", "model_a", "model_b", "n_common_valid", "agreement", "cohens_kappa", "n_unusable_excluded", "note"]),
        "unattempted_in_executed_blocks.csv": _csv(unatt, ["model_key", "task", "case_id", "status"]), "traces.jsonl": X.jsonl_bytes(ts.rows),
        "request_reconciliation.json": (json.dumps(recon, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    flat_pairs = [{"task": t, "a": x["a"], "b": x["b"], "n_common": x["n_common"], "agreement": x["agreement"], "kappa": x["kappa"], "n_unusable_excluded": x["n_unusable_excluded"], "note": x["note"]}
                  for t, a in report["agreement"].items() for x in a["pairs"]]
    files["comparison_report.md"] = bd.comparison_markdown(report, stats, plan, extra={"disagreements": disagreements, "dis_by_task": dict(dis_by_task), "recon": recon, "matrix": matrix,
                                                                                         "distribution": dist, "sample_hints": sample_hints, "comparison_rows": len(comp),
                                                                                         "origins": {p["key"]: _origin(plan, p, done) for p in plan["participants"]}, "ranking": ranking_rows, "route_failures": route_failures}).encode("utf-8")
    for rp in report["participants"]:
        files[f"models/{rp['key']}.md"] = bd.model_markdown(pmap[rp["key"]], rp, stats[rp["key"]], flat_pairs, bid).encode("utf-8")
    obs = bd.observability(plan, all_events, stats, ledger or {"note": "ledger snapshot not supplied"})
    files["observability.json"] = (json.dumps(obs, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    files["observability_report.md"] = bd.observability_markdown(obs).encode("utf-8")
    for name, data in dg.diagnostics_files(plan, state, reports_dir, all_events, stats_by, recon).items():
        files[f"diagnostics/{name}"] = data

    out_root = Path(out_root)
    out = out_root / f"real_benchmark_{bid}"
    n = 1
    while out.exists():
        n += 1
        out = out_root / f"real_benchmark_{bid}-r{n}"
    sessions = []
    if manifest_path and Path(manifest_path).exists():
        m = json.loads(Path(manifest_path).read_text())
        sessions = [{"started_at": s["started_at"], "operator": s["authorization"]["operator"], "authorized_at": s["authorization"]["authorized_at"],
                     "dataset_version": s["authorization"]["dataset_version"], "real_admissions_derived": s["authorization"]["real_admissions_derived"],
                     "provider_policy_status": s["authorization"]["provider_policy_status"], "plan_fingerprint": s["authorization"]["plan_fingerprint"]} for s in m.get("sessions", [])]
    sent = any(r["markers"]["data_kind"] == "REAL_DATA_BENCHMARK" and r["markers"]["source"] == "PREDICTION_RECORD" for r in ts.rows)
    warnings = ["SAMPLED OPERATIONAL BENCHMARK: accuracy is not measurable until independent ground-truth labels are approved." if not labels.available else "Approved labels exist; accuracy needs the frozen label set and DEV/TEST discipline.",
                "PRODUCTION_PARITY_NOT_ESTABLISHED", f"{report['benchmark_state']}: {report['coverage_cases']}", matrix["statement"]] + report["caveats"]
    manifest = {
        "schema": "real-benchmark-bundle-v2", "created_at": exported_at or datetime.now(timezone.utc).isoformat(), "bench_id": bid, "plan_fingerprint": plan["fingerprint"],
        "benchmark_state": report["benchmark_state"], "blocks": report["blocks"], "dataset": plan["dataset"], "ordering": plan["ordering"], "prompts": plan["prompts"], "limits": plan["limits"],
        "evaluation_contract_version": plan["evaluation_contract_version"], "authorization_sessions": sessions, "inherits": plan.get("inherits"),
        "participants": [{**{k: p[k] for k in ("key", "model_id", "display_name", "roles", "tasks", "route", "data_policy", "completion_state", "reason")}, "origin": _origin(plan, pmap[p["key"]], done)} for p in report["participants"]],
        "request_override_fields": {p["key"]: p["request_overrides"] for p in plan["participants"] if p.get("request_overrides")},
        "completion_counts": report["completion_counts"], "matrix_completeness": matrix, "request_reconciliation": {"totals": recon["totals"], "checks": recon["checks"]},
        "counts": {"result_rows": len(ts.rows), "trace_records": len(ts.rows), "failed_calls": sum(r["markers"]["failed_call"] for r in ts.rows), "invalid_outputs": sum(r["markers"]["invalid_output"] for r in ts.rows),
                   "records_redacted": ts.records_redacted, "unattempted_listed": len(unatt), "model_comparison_rows": len(comp), "disagreement_cases": len(disagreements),
                   "lead_probability_only_results": sum(1 for r in ts.rows if r["markers"].get("probability_only"))},
        "cost_semantics": COST_SEMANTICS, "actual_cost": {"guard_cost_usd": report["guard_cost_usd"], "ledger": ledger, "note": "ACTUAL = provider-reported USD; ESTIMATED = catalogue price x reported tokens; native credits are never converted"},
        "privacy": {
            "export": {"contains_raw_student_text": False, "raw_student_text_in_files": False, "this_bundle_transmitted_externally": False,
                       "redaction": "free-text fields pattern-redacted (no named-entity recognition); opaque case ids; no user4; no secrets; pseudonymization is not anonymization"},
            "inference_payloads": {"real_admissions_derived_data_sent_to_external_providers": sent,
                                   "meaning": "the benchmark itself sent redacted admissions-derived inputs to the recipients below through OpenRouter, under the operator's typed authorization; the authorization does not verify any provider policy",
                                   "recipients": {p["key"]: {"model_id": p["model_id"], "route": p["route"]["pinned_tag"], "data_policy": p["data_policy"]} for p in report["participants"]}},
        },
        "sample_diagnostics": {"intent_prediction_distribution_valid_outputs": dist, "sample_hints": sample_hints,
                               "upload_status_check_note": "No valid prediction used UPLOAD_STATUS_CHECK" if all(v.get("UPLOAD_STATUS_CHECK", 0) == 0 for v in dist.values()) else "UPLOAD_STATUS_CHECK was predicted by some participants",
                               "interpretation": "an observed prediction distribution only; with every ground-truth label pending it neither shows the class is absent from the sample nor that any model misclassified it"},
        "warnings": warnings, "files": {name: hashlib.sha256(b).hexdigest() for name, b in sorted(files.items())},
    }
    out.mkdir(parents=True)
    for name, data in files.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"dir": out, "manifest": manifest, "report": report, "n_trace_records": len(ts.rows), "reconciliation": recon}
