"""Detailed comparison and observability documents for a real-data benchmark. Built only from stored records and request events; no student text.

Everything descriptive (label distributions, latency, tokens, cost, failures) is measured. Nothing here says which model is more accurate: accuracy needs
independently approved labels, and agreement is never correctness. Absent measurements are omitted or marked, never replaced.
"""

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from msu_llm_lab.classification import benchmark as bm
from msu_llm_lab.classification.records import PredictionRecord, PredictionStore
from msu_llm_lab.observability.aggregate import aggregate, percentile
from msu_llm_lab.observability.events import RequestEvent, parse_ts


def collect(plan: Dict[str, Any], reports_dir: Path) -> Dict[str, Dict[str, List[PredictionRecord]]]:
    out: Dict[str, Dict[str, List[PredictionRecord]]] = {}
    for p in plan["participants"]:
        sb = p.get("store_bench") or plan["bench_id"]
        out[p["key"]] = {}
        for t in p["tasks"]:
            path = bm.prediction_path(reports_dir, sb, p["key"], t)
            out[p["key"]][t] = PredictionStore(path).load() if path.exists() else []
    return out


def run_ids(plan: Dict[str, Any]) -> set:
    return {f"{p.get('store_bench') or plan['bench_id']}-{p['key']}-{t}" for p in plan["participants"] for t in p["tasks"]}


def _f(v, nd=0, suf=""):
    return "—" if v is None else f"{v:.{nd}f}{suf}"


def _dist(vals: Sequence[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": min(vals), "mean": statistics.fmean(vals), "p50": percentile(vals, 50), "p90": percentile(vals, 90) if len(vals) >= 10 else None,
            "p95": percentile(vals, 95) if len(vals) >= 20 else None, "p99": percentile(vals, 99) if len(vals) >= 20 else None, "max": max(vals)}


def _dline(d: Dict[str, Any], unit: str = "ms") -> str:
    if not d.get("n"):
        return "no measurements"
    s = f"n={d['n']}  min {_f(d['min'])}  mean {_f(d['mean'])}  P50 {_f(d['p50'])}"
    if d.get("p90") is not None:
        s += f"  P90 {_f(d['p90'])}"
    if d.get("p95") is not None:
        s += f"  P95 {_f(d['p95'])}  P99 {_f(d['p99'])}"
    return s + f"  max {_f(d['max'])} {unit}" + ("" if d["n"] >= 20 else "  (P95/P99 withheld below 20 samples)")


def _unit(recs: List[PredictionRecord]) -> Dict[str, Any]:
    ok = [r for r in recs if r.status == "OK"]
    labels = Counter((r.predicted_label.value if r.predicted_label else ("YES" if r.predicted_need_cta else "NO") if r.predicted_need_cta is not None else None) for r in ok)
    labels.pop(None, None)
    undecided = sum(1 for r in ok if r.predicted_label is None and r.predicted_need_cta is None)
    conf = [r.native_confidence for r in recs if r.native_confidence is not None]
    sel = [r.label_probs[r.predicted_label.value] for r in ok if r.label_probs and r.predicted_label and r.predicted_label.value in r.label_probs]
    yes = [r.yes_prob for r in ok if r.yes_prob is not None]
    return {
        "n": len(recs), "ok": len(ok), "invalid": sum(r.status == "INVALID_OUTPUT" for r in recs), "error": sum(r.status == "ERROR" for r in recs),
        "issues": dict(Counter(i for r in recs for i in r.validation_issues)), "warnings": dict(Counter(w for r in recs for w in r.warnings)),
        "error_types": dict(Counter(r.error_type for r in recs if r.status == "ERROR")), "retry_codes": dict(Counter(c for r in recs for c in r.retry_errors)),
        "predicted_distribution": dict(sorted(labels.items())), "undecided_p_yes_only": undecided,
        "e2e_ms": _dist([r.latency_ms for r in recs if r.latency_ms is not None and r.status != "ERROR"]),
        "model_latency_ms": _dist([r.model_latency_ms for r in recs if r.model_latency_ms is not None]),
        "tokens": {"requests_with_tokens": sum(1 for r in recs if r.prompt_tokens is not None and r.completion_tokens is not None),
                   "prompt_total": sum(r.prompt_tokens or 0 for r in recs), "completion_total": sum(r.completion_tokens or 0 for r in recs),
                   "prompt_per_request": _dist([r.prompt_tokens for r in recs if r.prompt_tokens is not None]), "completion_per_request": _dist([r.completion_tokens for r in recs if r.completion_tokens is not None]),
                   "sources": dict(Counter(r.usage_source for r in recs))},
        "cost": {"basis": dict(Counter(r.cost_basis for r in recs)), "actual_usd": str(sum((r.actual_cost_usd for r in recs if r.actual_cost_usd is not None), Decimal("0"))),
                 "actual_n": sum(1 for r in recs if r.actual_cost_usd is not None), "estimated_usd": str(sum((r.estimated_cost_usd for r in recs if r.estimated_cost_usd is not None), Decimal("0"))),
                 "estimated_n": sum(1 for r in recs if r.estimated_cost_usd is not None), "native_credits": str(sum((r.native_cost for r in recs if r.native_cost is not None), Decimal("0"))),
                 "native_n": sum(1 for r in recs if r.native_cost is not None), "unknown": sum(1 for r in recs if r.cost_basis == "UNAVAILABLE" and r.status != "ERROR")},
        "native_confidence": _dist(conf), "selected_label_probability": _dist(sel), "p_yes": _dist(yes),
        "resolved_models": sorted({r.resolved_model_id for r in recs if r.resolved_model_id}), "serving_providers": sorted({r.serving_provider for r in recs if r.serving_provider}),
        "prompt_fingerprints": sorted({r.prompt_version for r in recs}), "dataset_versions": sorted({r.dataset_version for r in recs if r.dataset_version}),
        "attempts": dict(Counter(str(r.attempts) for r in recs)), "first_record": min((r.created_at for r in recs), default=None), "last_record": max((r.created_at for r in recs), default=None),
    }


def model_markdown(p: Dict[str, Any], part: Dict[str, Any], units: Dict[str, Dict[str, Any]], pairs: List[Dict[str, Any]], bench_id: str) -> str:
    L = [f"# {p['display_name']}", "", f"Benchmark `{bench_id}`. Operational comparison only: **accuracy is pending approved ground truth**; agreement is not correctness.", "",
         "## Identity and route", "", f"- Model id: `{p['model_id']}`  protocol `{p['protocol']}`  roles {', '.join(p['roles'])}" + ("  (benchmark route only; production parity not established)" if p["benchmark_route_only"] else ""),
         f"- Route: pinned `{p['route']['pinned_tag']}`, listed provider {p['route']['listed_provider']}, fallbacks {p['route']['fallbacks']}",
         f"- Price per 1M tokens (in/out): {p['pricing_usd_per_1m']}  status {p['price_status']}",
         f"- Approved request fields: {p.get('request_overrides') or 'none'}",
         f"- Data policy: **{p['data_handling']['data_policy']}** (UNKNOWN = nothing verified from a primary source)", f"- Native fields: {p['native_fields']}",
         f"- Completion state: **{part['completion_state']}**  ({part['reason']})", f"- Results stored under bench `{p.get('store_bench') or bench_id}`" + ("  (carried over)" if p.get("store_bench") else ""), ""]
    for t, u in units.items():
        pu = next((x for x in part["units"] if x["task"] == t), {})
        L += [f"## {t}", "", f"- Planned (full dataset) {pu.get('planned_full_dataset')}  attempted {u['n']}  unattempted {pu.get('unattempted')}  excluded by per-case cap {pu.get('excluded_per_case_cap')}",
              f"- Outcomes: valid {u['ok']}, invalid output {u['invalid']}, request errors {u['error']}" + (f"  (valid rate {100 * u['ok'] / u['n']:.1f}%)" if u["n"] else "")]
        if u["issues"]:
            L.append(f"- Invalid-output reasons: {u['issues']}")
        if u["error_types"] or u["retry_codes"]:
            L.append(f"- Errors: {u['error_types']}  retry/attempt codes {u['retry_codes']}  attempts per request {u['attempts']}")
        if u["warnings"]:
            L.append(f"- Warnings: {u['warnings']}")
        L.append(f"- Predicted distribution among valid outputs (descriptive, not correctness): {u['predicted_distribution'] or '—'}" + (f"  undecided (P(yes) only) {u['undecided_p_yes_only']}" if u["undecided_p_yes_only"] else ""))
        L += [f"- End-to-end latency (client-measured): {_dline(u['e2e_ms'])}", f"- Provider-reported model latency: {_dline(u['model_latency_ms']) if u['model_latency_ms'].get('n') else ('not reported by provider' if u['e2e_ms'].get('n') else '— (no successful response)')}"]
        tk = u["tokens"]
        L.append(f"- Tokens: {tk['requests_with_tokens']} of {u['n']} requests report tokens; prompt total {tk['prompt_total']}, completion total {tk['completion_total']}; per request prompt {_dline(tk['prompt_per_request'], 'tok')}; completion {_dline(tk['completion_per_request'], 'tok')}; sources {tk['sources']}")
        c = u["cost"]
        L.append(f"- Cost: actual USD {c['actual_usd']} ({c['actual_n']} req)  estimated USD {c['estimated_usd']} ({c['estimated_n']} req)  native credits {c['native_credits']} ({c['native_n']} req; USD ratio unverified, not converted)  basis {c['basis']}  unknown-cost requests {c['unknown']}")
        if u["native_confidence"].get("n"):
            L.append(f"- Native confidence field: {_dline(u['native_confidence'], '')}")
        if u["selected_label_probability"].get("n"):
            L.append(f"- Selected-label probability: {_dline(u['selected_label_probability'], '')}")
        if u["p_yes"].get("n"):
            L.append(f"- P(yes): {_dline(u['p_yes'], '')}")
        L += [f"- Resolved model(s): {u['resolved_models'] or '—'}  serving provider(s): {u['serving_providers'] or '—'}  prompt {u['prompt_fingerprints']}  dataset {u['dataset_versions']}", ""]
    mine = [x for x in pairs if p["key"] in (x["a"], x["b"])]
    L += ["## Agreement with other participants (valid common outputs only)", ""]
    if not mine:
        L.append("No comparable pairs yet.")
    for x in mine:
        other = x["b"] if x["a"] == p["key"] else x["a"]
        L.append(f"- vs `{other}` ({x['task']}): n={x['n_common']}  agreement {_f(x['agreement'] and 100 * x['agreement'], 1, '%')}  kappa {_f(x['kappa'], 3)}  unusable pairs excluded {x['n_unusable_excluded']} {x['note']}")
    return "\n".join(L) + "\n"


def comparison_markdown(report: Dict[str, Any], stats: Dict[str, Dict[str, Dict[str, Any]]], plan: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    extra = extra or {}
    d = report["dataset"]
    L = [f"# Detailed comparison: all participants ({report['bench_id']})", "",
         f"**{report['benchmark_state']}.** A sampled operational benchmark: it measures reliability, speed, tokens and cost on the same prepared cases. It does **not** say which model is more accurate (ground truth: {report['accuracy']['status']}).", "",
         f"- Dataset `{d['version']}`, session policy {d['session_policy']}, parity `{d['parity_status']}`; eligible {d['eligible']}; excluded {d['excluded']}",
         f"- Sample: {plan['ordering']['method']}; blocks completed {report['blocks']['completed']} of {report['blocks']['planned']} ({report['coverage_cases']})",
         "- Prompts: " + "; ".join(f"{k} `{v['version']}`" for k, v in report["prompts"].items()),
         f"- Caps: per case ${report['limits']['per_case_usd']}, per run ${report['limits']['per_run_usd']}, pilot ${report['limits']['pilot_usd']}, total ${report['limits']['total_usd']}; guard cost this plan ${report['guard_cost_usd']}",
         ""]
    if plan.get("inherits"):
        L += [f"- Carry-over from `{plan['inherits']['parent_bench']}`: {', '.join(plan['inherits']['participants'])}", ""]
    if extra.get("matrix"):
        m = extra["matrix"]
        L += [f"> **{m['statement']}** " + ("Incomplete: " + ", ".join(f"{u['model_key']} {u['task']} ({u['unattempted']} unattempted)" for u in m["incomplete_units"]) + "." if m["incomplete_units"] else ""),
              (f"> Quarantined by the circuit breaker: {list(m['quarantined'])}" if m["quarantined"] else ""), ""]
    L += ["## 1. Coverage and completion", "", "| Participant | Roles | Tasks | Route | Origin | State | Reason |", "|---|---|---|---|---|---|---|"]
    for p in report["participants"]:
        L.append(f"| {p['display_name']} | {', '.join(p['roles'])} | {', '.join(p['tasks'])} | {p['route']['pinned_tag'] or 'none'} | {extra.get('origins', {}).get(p['key'], '—')} | **{p['completion_state']}** | {p['reason']} |")
    if extra.get("recon"):
        rc = extra["recon"]
        t, ck = rc["totals"], rc["checks"]
        L += ["", "### Request-count reconciliation", "",
              f"- Planned new logical requests for the {rc['blocks_completed']} executed blocks: **{t['planned_new_logical_requests']}** (the plan's block table says {t['plan_block_requests_for_executed_blocks']})",
              f"- Dispatched (started) logical requests: **{t['dispatched_logical_requests']}**; not dispatched (unattempted, cases left by a quarantined participant): **{t['unattempted']}**",
              f"- Outcomes recorded: **{t['outcomes_recorded']}** = {t['ok']} valid + {t['invalid_output']} invalid output + {t['error']} request errors",
              f"- Provider HTTP attempts: **{t['provider_http_attempts']}** = {t['outcomes_recorded']} outcomes + {t['retries']} retries",
              f"- Carried-over records (earlier benchmark, not re-sent): **{t['carried_over_records_not_resent']}**; rows in the result files: **{t['rows_in_results_files']}** = new outcomes + carried over",
              "- Consistency checks: " + ", ".join(f"{k} = {v}" for k, v in ck.items()), f"- {rc['explanation']}", "",
              "| Participant | Task | Origin | Planned new | Dispatched | Outcomes | Valid | Invalid | Errors | HTTP attempts | Retries | Unattempted |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for u in rc["units"]:
            L.append(f"| {u['model_key']} | {u['task']} | {u['origin']} | {u['planned_new_logical_requests'] if u['planned_new_logical_requests'] is not None else '—'} | {u['dispatched_logical_requests']} | {u['outcomes_recorded'] or (str(u['carried_over_records']) + ' carried') } | {u['ok']} | {u['invalid_output']} | {u['error']} | {u['provider_http_attempts']} | {u['retries']} | {u['unattempted']}{' (quarantined)' if u['quarantined'] and u['unattempted'] else ''} |")
    for title, key in (("2. Reliability (valid, invalid, errors)", "rel"), ("3. Speed and tokens", "spd"), ("4. Cost (actual, estimated, native; never mixed)", "cost")):
        L += ["", f"## {title}", ""]
        if key == "rel":
            L += ["| Participant | Task | Attempted | Valid | Invalid | Errors | Valid rate | Main invalid reasons | Error classes |", "|---|---|---|---|---|---|---|---|---|"]
        elif key == "spd":
            L += ["| Participant | Task | E2E P50 ms | E2E mean | E2E max | Model latency | Prompt tok/req (mean) | Completion tok/req (mean) | Requests with tokens |", "|---|---|---|---|---|---|---|---|---|"]
        else:
            L += ["| Participant | Task | Actual USD | Estimated USD | Native credits | Unknown-cost requests | Est. USD per request | Est. USD per 1,000 cases |", "|---|---|---|---|---|---|---|---|"]
        for p in report["participants"]:
            for t in p["tasks"]:
                u = stats[p["key"]][t]
                n = u["n"]
                if key == "rel":
                    L.append(f"| {p['display_name']} | {t} | {n} | {u['ok']} | {u['invalid']} | {u['error']} | {_f(100 * u['ok'] / n if n else None, 1, '%')} | {u['issues'] or '—'} | {u['error_types'] or '—'} |")
                elif key == "spd":
                    e, tk = u["e2e_ms"], u["tokens"]
                    L.append(f"| {p['display_name']} | {t} | {_f(e.get('p50'))} | {_f(e.get('mean'))} | {_f(e.get('max'))} | {_f(u['model_latency_ms'].get('p50')) if u['model_latency_ms'].get('n') else ('not reported by provider' if e.get('n') else '—')} | "
                             f"{_f(tk['prompt_per_request'].get('mean'))} | {_f(tk['completion_per_request'].get('mean'), 1)} | {tk['requests_with_tokens']} of {n} |")
                else:
                    c = u["cost"]
                    per = (Decimal(c["estimated_usd"]) / c["estimated_n"]) if c["estimated_n"] else None
                    L.append(f"| {p['display_name']} | {t} | {c['actual_usd'] if c['actual_n'] else '—'} | {c['estimated_usd'] if c['estimated_n'] else '—'} | {c['native_credits'] if c['native_n'] else '—'} | {c['unknown']} | "
                             f"{('%.6f' % per) if per is not None else '—'} | {('%.4f' % (per * 1000)) if per is not None else '—'} |")
    L += ["", "Cost columns: an empty cell means not measured, never zero. Requests that ended in an error returned no response and carry no cost evidence; that is not proof they cost nothing. "
          "`requests_without_cost_evidence`, `errors_no_response` and `unknown_cost_after_response` in model_summary.csv separate these; unresolved ledger entries are counted separately in the manifest."]
    L += ["", "## 5. Agreement on valid common outputs (not correctness)", ""]
    if extra.get("disagreements") is not None:
        dis = extra["disagreements"]
        L += [f"**Cases where at least two valid participants answered and differed: {len(dis)}** ({extra.get('dis_by_task', {})}), counted from model_comparison.csv (`models_agree` = False). "
              f"The CSV has {extra.get('comparison_rows')} comparison rows. Case ids (opaque, first 40): {', '.join(r['case_id'] for r in dis[:40]) or 'none'}", ""]
    for t, a in report["agreement"].items():
        L.append(f"### {t}: {len(a['models'])} participants with valid outputs")
        L += ["", "| A | B | n common valid | Agreement | Kappa | Excluded unusable | Note |", "|---|---|---|---|---|---|---|"]
        for x in a["pairs"]:
            L.append(f"| {x['a']} | {x['b']} | {x['n_common']} | {_f(x['agreement'] and 100 * x['agreement'], 1, '%')} | {_f(x['kappa'], 3)} | {x['n_unusable_excluded']} | {x['note']} |")
        L += [""]
    L += ["## 5b. LEAD semantics (binary comparison eligibility)", "",
          "Participants that return a boolean contribute a binary decision. Jev returns only P(yes); until a threshold is selected on the DEV split and frozen, its results are probability observations and are NOT binary decisions (`binary_decision_available` = False, `probability_only` = True) and are excluded from YES/NO agreement.", ""]
    L += ["| Participant | LEAD valid outputs | Binary decisions | Probability-only results |", "|---|---|---|---|"]
    for p in report["participants"]:
        if "LEAD" in p["tasks"]:
            u = stats[p["key"]]["LEAD"]
            ok = u["ok"]
            und = u["undecided_p_yes_only"]
            L.append(f"| {p['display_name']} | {ok} | {ok - und} | {und} |")
    if extra.get("distribution"):
        L += ["", "## 5c. Predicted INTENT distribution (valid outputs only)", "", "| Participant | " + " | ".join(["GREETING", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST", "QUERY"]) + " |", "|---|---|---|---|---|---|"]
        for k, d in extra["distribution"].items():
            L.append(f"| {k} | " + " | ".join(str(d[l]) for l in ["GREETING", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST", "QUERY"]) + " |")
        tot = {l: sum(d[l] for d in extra["distribution"].values()) for l in ["GREETING", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST", "QUERY"]}
        L += ["", f"Totals across participants: {tot}.", "This is an **observed prediction distribution only**. With every ground-truth label pending, a zero for a category (for example UPLOAD_STATUS_CHECK) does not show that the category is absent from the sample or that any model misclassified it."]
        if extra.get("sample_hints"):
            h = extra["sample_hints"]
            L.append(f"Sample context: {h['with_selection_hints']} of {h['sampled_intent_cases']} sampled INTENT cases carry a preparation selection hint ({h['note']}).")
    if extra.get("route_failures") is not None:
        L += ["", "## 5d. Route and configuration failures", "", "These are request-level failures (HTTP status, rate limits, quarantine), separate from outputs that arrived but broke the output contract.", ""]
        if not extra["route_failures"]:
            L.append("No participant had a request failure or retry.")
        for k, v in extra["route_failures"].items():
            L.append(f"- **{k}** ({v['origin']}) on route `{v['route']}`: failed requests {v['failed_requests']}; attempt/retry codes {v['http_or_attempt_codes'] or 'none'}; approved request fields {v['request_overrides'] or 'none'}; quarantined: {v['quarantined'] or 'no'}")
    if extra.get("ranking") is not None:
        L += ["", "## 5e. Operational rankings (these are NOT accuracy rankings)", "", "Accuracy ranking: **unavailable** (every ground-truth label is pending). The tables below rank reliability, speed and estimated cost only, each on its own sample size; a participant with fewer than 20 requests is marked indicative.", ""]
        for t in ("INTENT", "LEAD"):
            for metric in ("valid_output_rate", "median_e2e_latency_ms", "estimated_usd_per_1000_cases"):
                rows_ = [r for r in extra["ranking"] if r["task"] == t and r["metric"] == metric]
                if rows_:
                    L += [f"**{t}: {metric}**", "", "| Rank | Participant | Value | n requests | Origin | Note |", "|---|---|---|---|---|---|"]
                    L += [f"| {r['rank']} | {r['model_key']} | {r['value']} | {r['n_requests']} | {r['origin']} | {r['sample_note']} |" for r in rows_]
                    L.append("")
    L += ["", "## 6. What this does not show", "", "- Accuracy, macro-F1, per-class precision/recall, confusion matrices, Brier score and calibration: **pending approved labels**.",
          "- Provider data policies are UNKNOWN for every route; the operator authorization does not verify them.", "- " + report["caveats"][1], "- Native cost is credits with an unverified USD ratio.", ""]
    return "\n".join(L)


def observability(plan: Dict[str, Any], events: Sequence[RequestEvent], stats: Dict[str, Dict[str, Dict[str, Any]]], ledger: Dict[str, Any]) -> Dict[str, Any]:
    rids = run_ids(plan)
    ev = [e for e in events if e.run_id in rids]
    now = max((parse_ts(e.ts) for e in ev), default=datetime.now(timezone.utc).timestamp())
    agg = aggregate(ev, now, window_s=3600 * 24, bucket_s=60)
    live = [e for e in ev if e.provenance == "LIVE"]
    starts = {e.request_id for e in live if e.kind == "REQUEST_STARTED"}
    ends = {e.request_id for e in live if e.kind in ("REQUEST_SUCCEEDED", "REQUEST_FAILED")}
    by_min: Dict[str, Dict[str, Any]] = {}
    for e in live:
        if e.kind in ("REQUEST_SUCCEEDED", "REQUEST_FAILED"):
            m = e.ts[:16]
            b = by_min.setdefault(m, {"requests": 0, "failed": 0, "latencies": []})
            b["requests"] += 1
            b["failed"] += e.status == "ERROR"
            if e.latency_ms is not None:
                b["latencies"].append(e.latency_ms)
    timeline = [{"minute_utc": m, "finished_requests": b["requests"], "failed": b["failed"], "mean_e2e_ms": (sum(b["latencies"]) / len(b["latencies"])) if b["latencies"] else None}
                for m, b in sorted(by_min.items())]

    def g(x):
        return {"finished": x.n_terminal, "ok": x.n_ok, "invalid": x.n_invalid, "errors": x.n_error, "timeouts": x.n_timeouts, "rate_limited": x.n_rate_limited, "provider_5xx": x.n_provider_failures,
                "retries": x.n_retries, "e2e_ms": _dist(x.latencies_ms), "provider_model_latency_ms": _dist(x.model_latencies_ms) if x.model_latencies_ms else "not reported by provider",
                "tokens_known_requests": x.n_token_known, "tokens_unknown_requests": x.n_token_unknown, "prompt_tokens": x.prompt_tokens, "completion_tokens": x.completion_tokens,
                "tokens_per_second": x.tokens_per_second, "tokens_per_second_basis": x.tps_source, "actual_usd": str(x.actual_cost) if x.n_actual_cost else None,
                "estimated_usd": str(x.estimated_cost) if x.n_estimated_cost else None, "native_credits": str(x.native_cost) if x.n_native_cost else None, "cost_unresolved": x.n_cost_unresolved}
    http = Counter(str(e.http_status) for e in live if e.kind == "REQUEST_FAILED" and e.http_status is not None)
    return {
        "schema": "real-benchmark-observability-v1", "bench_id": plan["bench_id"], "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {"events": "request event log (data/telemetry/request_events.jsonl)", "records": "prediction stores under reports/classification", "ledger": "SQLite budget ledger"},
        "integrity": {"events_in_scope": len(ev), "simulated_events_excluded": agg.n_quarantined_simulated, "requests_started": len(starts), "requests_finished": len(ends),
                      "started_without_end": len(starts - ends), "ended_without_start": len(ends - starts), "retry_events": agg.n_retry_events,
                      "records_vs_events": {k: {t: {"records": u["n"]} for t, u in v.items()} for k, v in stats.items()}},
        "window": {"first_event": min((e.ts for e in ev), default=None), "last_event": max((e.ts for e in ev), default=None)},
        "overall": g(agg.overall), "by_model": {k: g(v) for k, v in sorted(agg.by_model.items())}, "by_provider": {k: g(v) for k, v in sorted(agg.by_provider.items())},
        "http_status_breakdown": dict(http), "timeline_per_minute": timeline,
        "ledger": ledger,
        "notes": ["Latency is client-measured end to end; provider/model latency appears only where the provider reported it.", "Concurrency is 1 (requests are sent sequentially).",
                  "Native cost is credits with an unverified USD ratio; estimated USD is catalogue price x reported tokens; actual USD needs a verified provider USD figure."],
    }


def observability_markdown(o: Dict[str, Any]) -> str:
    ov, ig = o["overall"], o["integrity"]
    L = [f"# Observability report ({o['bench_id']})", "", f"Window {o['window']['first_event']} to {o['window']['last_event']}.  Sources: {o['sources']}", "",
         "## Telemetry integrity", "", f"- Events in scope {ig['events_in_scope']}; requests started {ig['requests_started']}, finished {ig['requests_finished']}; started without an end {ig['started_without_end']}; ended without a start {ig['ended_without_start']}; retry events {ig['retry_events']}; simulated events excluded {ig['simulated_events_excluded']}",
         "", "## Overall", "", f"- Finished {ov['finished']}: ok {ov['ok']}, invalid {ov['invalid']}, errors {ov['errors']}  (timeouts {ov['timeouts']}, rate-limited {ov['rate_limited']}, 5xx {ov['provider_5xx']}, retries {ov['retries']})",
         f"- End-to-end latency: {_dline(ov['e2e_ms'])}", f"- HTTP failures: {o['http_status_breakdown'] or 'none'}", "", "## By model", "", "| Model | Finished | OK | Invalid | Errors | Retries | E2E P50 ms | Tokens in/out | Tokens/s (basis) | Estimated USD | Native credits |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in o["by_model"].items():
        L.append(f"| {k} | {x['finished']} | {x['ok']} | {x['invalid']} | {x['errors']} | {x['retries']} | {_f(x['e2e_ms'].get('p50'))} | {x['prompt_tokens']}/{x['completion_tokens']} | {_f(x['tokens_per_second'], 1)} ({x['tokens_per_second_basis'] or '—'}) | {x['estimated_usd'] or '—'} | {x['native_credits'] or '—'} |")
    L += ["", "## By serving provider (as reported)", "", "| Provider | Finished | OK | Errors | E2E P50 ms | Estimated USD |", "|---|---|---|---|---|---|"]
    for k, x in o["by_provider"].items():
        L.append(f"| {k} | {x['finished']} | {x['ok']} | {x['errors']} | {_f(x['e2e_ms'].get('p50'))} | {x['estimated_usd'] or '—'} |")
    L += ["", "## Timeline (per minute, UTC)", "", "| Minute | Finished | Failed | Mean E2E ms |", "|---|---|---|---|"]
    for t in o["timeline_per_minute"]:
        L.append(f"| {t['minute_utc']} | {t['finished_requests']} | {t['failed']} | {_f(t['mean_e2e_ms'])} |")
    L += ["", "## Ledger", "", f"{o['ledger']}", "", "## Notes", ""] + [f"- {n}" for n in o["notes"]]
    return "\n".join(L) + "\n"
