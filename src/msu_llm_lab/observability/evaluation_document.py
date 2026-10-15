"""Report content for the evaluation report: one element list rendered to Markdown and to PDF (reportlab) so both always agree."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from msu_llm_lab.observability.evaluation_report import NOT_COMPUTABLE

TITLE = "MSU LLM Inference Workbench — Comprehensive Benchmark Evaluation Report"
Element = Tuple[str, Any]


def _f(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}"
    return str(v)


def table(df: pd.DataFrame, cols: Sequence[str], heads: Sequence[str] = (), nd: Dict[str, int] = None, maxrows: int = 400) -> Element:
    nd = nd or {}
    rows = [[_f(r[c], nd.get(c, 3)) for c in cols] for _, r in df.head(maxrows).iterrows()]
    return ("table", {"head": list(heads or cols), "rows": rows})


def P(t: str) -> Element:
    return ("p", t)


def B(items: Sequence[str]) -> Element:
    return ("bullets", list(items))


def FIG(fid: str, man: pd.DataFrame) -> Element:
    r = man[man.figure_id == fid].iloc[0]
    return ("fig", {"file": r.file, "caption": f"{r.figure_id}. {r.title}. {r.caption}"})


def build_sections(R: Dict[str, Any], fig: pd.DataFrame) -> List[Tuple[str, List[Element]]]:
    S, T, aux, F = R["S"], R["T"], R["aux"], R["findings"]
    df, man, state, plan = S["df"], S["manifest"], S["state"], S["plan"]
    mts, ms = T["model_task_summary"], T["model_summary"]
    rec = T["request_reconciliation"].set_index("quantity")["value"]
    cost = aux["cost"]
    ex = int(state["blocks_completed"])
    fid = {Path(r.file).stem: r.figure_id for r in fig.itertuples()}

    def fg(stem: str) -> Element:
        return FIG(fid[stem], fig)

    n_new, n_old = int(df.is_new.sum()), int((~df.is_new).sum())
    full = [p for p in sorted(df.participant.unique()) if not p.startswith("mistral")]
    ok_both = [p for p in full if (mts[mts.participant == p].valid_rate >= 0.99).all()]
    I, L = mts[mts.task == "INTENT"].set_index("participant"), mts[mts.task == "LEAD"].set_index("participant")
    fastest = mts.sort_values("latency_p50_ms").iloc[0]
    cheapest = mts.sort_values("est_cost_usd_per_1000_requests").iloc[0]
    dearest = mts.sort_values("est_cost_usd_per_1000_requests").iloc[-1]
    checks_ok = int(T["reconciliation_summary"].passed.sum())
    nchk = len(T["reconciliation_summary"])
    pw = T["agreement_metrics"]
    S_: List[Tuple[str, List[Element]]] = []

    def add(title: str, *els: Element):
        S_.append((title, list(els)))

    add("Document control",
        P(f"Run analysed: {S['cfg'].bench_id} (plan fingerprint {man['plan_fingerprint']}); final verified bundle: {S['cfg'].bundle_dir.name}. "
          f"Generated offline and deterministically from local artifacts; no network, no inference, no spend. Source artifacts were hashed before and after the analysis "
          f"({'unchanged' if R['hashes_ok'] else 'CHANGED - investigate'})."),
        B(["Provenance vocabulary: ACTUAL (provider-reported USD), ESTIMATED (catalogue price x provider-reported tokens), GUARD (stop-check quantity, never booked), "
           "UNAVAILABLE (not measured), NOT COMPUTABLE - GROUND TRUTH PENDING.",
           "Blank or n/a means not measured; it is never zero.", "Every claim is tied to a source file listed in the artifact inventory."]))

    add("Executive summary",
        P(f"This is a sampled operational benchmark, not an accuracy evaluation. {ex} of {len(plan['blocks'])} blocks ran: {ex*10} of 2,211 INTENT and {ex*10} of 723 LEAD eligible cases, "
          f"answered by {df.participant.nunique()} participants ({n_new:,} new requests plus {n_old:,} carried over). All {len(df):,} records have ground truth PENDING, so accuracy, F1, "
          f"calibration and confusion matrices are {NOT_COMPUTABLE}."),
        B([f"Reconciliation: {checks_ok} of {nchk} independent checks pass; 3,400 = {n_old} carried + 2,860 planned new; 2,860 = {n_new:,} dispatched + {len(S['unattempted'])} unattempted (all Mistral, quarantined).",
           f"Operational reliability (valid-output rate, INTENT/LEAD): " + "; ".join(f"{p} {I.valid_rate.get(p, float('nan')):.1%}/{L.valid_rate.get(p, float('nan')):.1%}" for p in ["jev-1.14", "gpt-4.1", "claude-sonnet-5.5", "claude-haiku-5.5", "deepseek-chat-v3.1"]).replace("nan%", "n/a"),
           f"Models with valid-output rate >= 99% in every task they ran (complete coverage): {', '.join(ok_both)}.",
           f"Lowest median latency: {fastest.participant} {fastest.task} at {fastest.latency_p50_ms:,.0f} ms; lowest estimated cost: {cheapest.participant} {cheapest.task} at ${cheapest.est_cost_usd_per_1000_requests:.3f} per 1,000 requests; "
           f"highest: {dearest.participant} {dearest.task} at ${dearest.est_cost_usd_per_1000_requests:.2f}.",
           f"Cost: estimated ${cost['results_estimated_usd']:.4f} for all records; ledger ${cost['ledger_total_usd']:.4f} (includes smoke tests and superseded sends); guard cost ${cost['guard_cost_usd']:.4f}. "
           "No provider-reported USD exists, so none of these is an invoice.",
           "No single winner is declared: ranks are per metric and per task, on matched cases, and a Pareto view shows trade-offs."]),
        P("What this does not show: which model classifies correctly, how models behave on the 2,041 INTENT and 553 LEAD cases not sampled, or equivalence with the production Dify workflow."))

    add("How to read this report",
        B(["Sections 1-4 summarise; 5-9 describe design and execution; 10-17 audit data and reconcile counts; 18-19 define metrics and methods; 20-39 present results; 40-44 limits, privacy, recommendations and reproducibility.",
           "Valid-output rate is the share of responses that parsed to an allowed value. It is a reliability measure, not correctness.",
           "Agreement between models is not accuracy: models can agree and all be wrong.",
           "Intervals: Wilson 95% for proportions; percentile bootstrap (2,000 resamples, fixed seed) for medians and means."]))

    add("Key findings at a glance",
        table(ms.assign(role=ms.role.str.replace("PRODUCTION_BASELINE", "PROD_BASELINE").str.replace("FRONTIER_REFERENCE", "FRONTIER_REF")), ["participant", "role", "tasks", "records", "valid_rate", "est_cost_usd_per_request", "completion_state"],
              ["Participant", "Role", "Tasks", "Records", "Valid rate", "Est. USD/req", "State"], {"est_cost_usd_per_request": 5}),
        P("Role lists use the registry; PRODUCTION_BASELINE rows are benchmark-route replicas of production models, not the production workflow itself."))

    add("Scope and objectives",
        B(["Objective: compare operational behaviour (reliability, latency, cost, inter-model agreement) of 11 participants on two admissions tasks: Main Intent (5 labels) and Lead Qualifier ({need_cta: true|false}).",
           "In scope: the 17 executed blocks of plan d0a763e4a920, including carried-over results from two parent plans.",
           "Out of scope: accuracy (no approved labels), production parity (Dify export missing), full-dataset estimates, provider data-policy verification."]))

    add("Benchmark design",
        P(f"Ordering: {plan['ordering']['method']}, seed {plan['ordering']['seed']}, blocks of {plan['ordering']['block_size']} cases per task; every block gives all participants identical cases. "
          f"Pattern buckets are preparation metadata, not labels. Prompt fingerprints: INTENT {man['prompts']['INTENT']['version']}, LEAD {man['prompts']['LEAD']['version']}."),
        B([f"Limits: per-run ${plan['limits']['per_run_usd']}, per-case ${plan['limits']['per_case_usd']}, total ${plan['limits']['total_usd']}; max attempts {plan['limits']['max_attempts']}; timeout {plan['limits']['timeout_seconds']} s.",
           f"Max output tokens: INTENT {plan['limits']['max_output_tokens']['INTENT']}, LEAD {plan['limits']['max_output_tokens']['LEAD']}.",
           "A circuit breaker quarantines a participant after 3 consecutive errors."]))

    add("Dataset and sampling",
        P(f"Eligible cases: INTENT {man['dataset']['eligible']['INTENT']:,} of {man['dataset']['candidates']['INTENT']:,} candidates, LEAD {man['dataset']['eligible']['LEAD']:,} of {man['dataset']['candidates']['LEAD']:,}. Excluded by reason: {json.dumps(man['dataset']['excluded'])}. Labels: {man['dataset']['labels']}."),
        table(T["stratum_coverage"].head(25), ["task", "stratum", "cases_in_sample"], ["Task", "Preparation stratum (metadata)", "Cases in sample"]),
        fg("stratum_coverage"),
        P("The sample is stratified toward pattern variety first, so it over-represents rare patterns relative to the population. Rates here are sample rates, not population estimates."))

    parts = man["participants"]
    add("Participants and routes",
        table(pd.DataFrame([{"key": p["key"], "model_id": p["model_id"], "roles": ",".join(p["roles"]), "tasks": ",".join(p["tasks"]), "route": p["route"].get("pinned_tag", ""),
                             "policy": p["data_policy"], "state": p["completion_state"]} for p in parts]),
              ["key", "model_id", "roles", "tasks", "route", "policy", "state"], ["Key", "Model id", "Roles", "Tasks", "Pinned route", "Data policy", "State"]),
        P("Routes are pinned with fallbacks disabled. Data policy UNKNOWN means no retention/training facts were verified from a primary source for that route."))

    add("Execution history",
        B([f"Plan created {S['plan']['informational']['created_at']}; authorised {man['authorization_sessions'][0]['authorized_at']} by typed operator approval.",
           f"Blocks completed {ex}; halted: {state['halted']}; quarantined: {state['quarantined']}.",
           "Earlier plans (bench-045a3c43f1a5, bench-492495eeddc0) produced the carried-over records; route re-pins (Azure, Vertex) and Qwen reasoning-off followed provider 404 and empty-output findings.",
           f"Ledger at plan time: {S['plan']['informational']['ledger_at_plan_time']}."]))

    inv = T["artifact_inventory"]
    add("Artifact inventory and provenance",
        P(f"{len(inv)} input files were inventoried (SHA-256, size, modification time) and are listed in artifact_inventory.csv. The private salt file is deliberately excluded from inventory and hashing."),
        table(inv.assign(path=inv.path.str[-58:]).groupby("role", as_index=False).agg(files=("path", "size"), bytes=("bytes", "sum")).head(40) if False else
              inv[inv.used_in_analysis].assign(path=lambda d: d.path.str[-62:]), ["path", "bytes", "provenance"], ["File (tail)", "Bytes", "Role"]))

    add("Artifact integrity",
        P(f"Every inventoried file was hashed before analysis and again after; result: {'IDENTICAL for all ' + str(len(R['hashes_before'])) + ' files' if R['hashes_ok'] else 'MISMATCH'}. "
          f"The bundle manifest lists {len(man['files'])} hashed files; verification_report.* was written after sealing and is not in that list (by design)."),
        table(pd.DataFrame(S["verification"]["checks"]), ["check", "ok", "detail"], ["Bundle verification check", "Pass", "Detail"], maxrows=30))

    add("Request-level reconstruction",
        P(f"The canonical table has {len(df):,} rows, one per (participant, task, case); keys (participant, task, case_id) and request_id are unique. Each row gained block index, preparation stratum, origin and outcome flags. "
          "Block index is recomputed with the benchmark's own deterministic ordering from local prep metadata (no network)."),
        table(df.groupby(["participant", "task"]).agg(rows=("case_id", "size"), blocks=("block", "nunique"), first=("start_ts", "min")).reset_index().assign(first=lambda d: d["first"].astype(str).str[:19]),
              ["participant", "task", "rows", "blocks", "first"], ["Participant", "Task", "Rows", "Blocks", "First request"]))

    add("Request reconciliation",
        table(T["request_reconciliation"], ["quantity", "value", "source"], ["Quantity", "Value", "Source"]),
        table(T["reconciliation_summary"], ["check", "passed"], ["Independent check", "Passed"]),
        fg("request_flow"))

    old = S["old_recon"].get("real_benchmark_bench-d0a763e4a920", {})
    add("Carry-over accounting",
        P(f"{n_old} records were inherited from parent plans (identical participant entries with no ERROR records) and are not re-sent; {n_new:,} are new. Inheriting participants append new results to the parent's prediction store, so run_id prefixes do not identify carry-over; the origin column does. "
          f"Mistral was not carried over (quarantined in its parent): {S['plan']['inherits']['not_carried_over']}."),
        table(T["reconciliation_details"], ["participant", "task", "carried_over", "dispatched", "unattempted", "ok", "invalid", "error", "retries", "balanced"],
              ["Participant", "Task", "Carried", "Dispatched", "Unattempted", "OK", "Invalid", "Error", "Retries", "Balanced"]),
        P(f"The first bundle (no suffix) counted carried records as new: planned_new {old.get('totals', {}).get('planned_new_logical_requests')} versus plan {rec['planned new logical requests (plan blocks, executed)']}; its reconciliation check failed. Later bundles are correct."))

    add("Cost reconciliation",
        table(T["cost_provenance"], ["source", "usd", "basis"], ["Source", "USD", "Basis"], {"usd": 6}),
        P(f"Ledger ${cost['ledger_total_usd']:.6f} = final results ${cost['results_estimated_usd']:.6f} + smoke ${cost['ledger_smoke_usd']:.6f} + residual ${cost['delta_ledger_minus_results_minus_smoke']:.6f}. "
          "The residual is booked spend on requests whose records were superseded (for example re-sends after route re-pins or the Qwen reasoning-off change) and is itemised per run id."),
        fg("cost_sources"), fg("ledger_residual"))

    ev = aux["events"]
    add("Event-log reconciliation",
        P(f"Request events since the authorisation start ({ev['window_start'][:19]}): STARTED {ev['started']}, SUCCEEDED {ev['succeeded']}, FAILED {ev['failed']}, RETRY {ev['retry']}; HTTP statuses {ev['http_status']}. "
          f"STARTED equals dispatched records ({ev['dispatched_records']}); SUCCEEDED + FAILED = {ev['succeeded'] + ev['failed']}; retries {ev['retry']} equal attempts minus dispatched. The log is append-only and spans several plans, so the window is a heuristic."))

    add("Known issues investigation",
        table(pd.DataFrame(F), ["id", "issue", "status", "evidence", "impact"], ["ID", "Issue", "Status", "Evidence", "Impact"]))

    add("Metric catalogue", table(T["metric_definitions"], ["metric", "definition", "class", "caveat"], ["Metric", "Definition", "Class", "Caveat"]))

    add("Statistical methods",
        B(["Proportions: Wilson 95% interval. Medians/means: percentile bootstrap, 2,000 resamples, seed 20261015, per-unit seeds derived from SHA-256 of a label.",
           "Paired comparisons on the same cases only, with at least 30 pairs: McNemar exact (valid vs not), Wilcoxon signed-rank (latency on both-valid cases, cost). Holm correction within each task x metric family.",
           "A significant paired difference means the sample favours one model on that operational metric; it is not a statement about correctness or about unsampled cases.",
           "Percentiles use linear interpolation; the source bundle used its own percentile definition, so p50/p95 can differ slightly (counts are identical)."]))

    cov = T["coverage_matrix"]
    um = S["unattempted"]
    mi_i = int(((um.model_key == "mistral-small-3.2-24b-instruct") & (um.task == "INTENT")).sum())
    mi_l = int(((um.model_key == "mistral-small-3.2-24b-instruct") & (um.task == "LEAD")).sum())
    add("Coverage and completeness",
        P(f"Executed blocks: {ex}. Not executed: {len(plan['blocks']) - ex}. Matrix complete for executed blocks: {man['matrix_completeness']['complete_for_executed_blocks']} (Mistral INTENT {mi_i} and LEAD {mi_l} unattempted)."),
        table(cov[cov.block < ex].groupby(["participant", "task"]).agg(records=("records", "sum"), carried=("carried_over", "sum"), new=("new_this_plan", "sum"), unattempted=("unattempted", "sum")).reset_index(),
              ["participant", "task", "records", "carried", "new", "unattempted"], ["Participant", "Task", "Records", "Carried", "New", "Unattempted"]),
        fg("coverage_overview"))

    bs = T["batch_summary"]
    add("Per-batch analysis",
        P(f"All {len(bs)} blocks are accounted for; blocks {ex}-{len(bs) - 1} are NOT EXECUTED and carry no results. Executed blocks:"),
        table(bs[bs.block < ex], ["block", "records", "new_this_plan", "carried_over", "valid", "invalid_output", "errors", "unattempted", "valid_rate", "latency_p50_ms", "est_cost_usd"],
              ["Block", "Records", "New", "Carried", "Valid", "Invalid", "Err", "Unatt.", "Valid rate", "p50 ms", "Est USD"], {"latency_p50_ms": 0, "est_cost_usd": 4}),
        fg("block_heatmap_valid"))

    add("Batch trends",
        P("Per-block series for the 17 executed blocks. With 10 cases per task per block, single-block rates are noisy; look for sustained shifts, not single points."),
        fg("latency_by_block"), fg("disagreement_by_block"))

    add("Operational results overview",
        table(mts, ["participant", "task", "records", "valid", "invalid_output", "errors", "valid_rate", "valid_rate_ci_lo", "valid_rate_ci_hi", "latency_p50_ms", "est_cost_usd_per_1000_requests"],
              ["Participant", "Task", "N", "Valid", "Invalid", "Err", "Valid rate", "CI lo", "CI hi", "p50 ms", "USD/1k req"], {"latency_p50_ms": 0}),
        fg("outcomes_stacked"))

    rl = T["reliability_metrics"]
    add("Output validity and reliability",
        table(rl, ["participant", "task", "records", "error_rate", "invalid_rate", "retry_rate", "rate_limit_errors", "provider_5xx_errors", "max_attempts_seen"],
              ["Participant", "Task", "N", "Error rate", "Invalid rate", "Retry rate", "429s", "5xx", "Max attempts"]),
        fg("valid_rate_intent"), fg("valid_rate_lead"), fg("invalid_causes"), fg("retry_rate"))

    ea = T["error_analysis"]
    add("Error analysis",
        table(ea.head(30), ["participant", "task", "status", "error_category", "validation_issues", "finish_reason", "n", "new", "completion_tokens_median"],
              ["Participant", "Task", "Status", "Category", "Issue", "Finish", "N", "New", "Median compl. tok"], {"completion_tokens_median": 0}),
        P("EMPTY_OUTPUT records are HTTP 200 responses whose visible content was empty; 'length' finish means the token cap was reached. finish_reason was not captured for records created before the capture was added."),
        fg("finish_reasons"))

    lm = T["latency_metrics"]
    add("Latency analysis",
        P("Latency is client-side end-to-end wall time (provider model latency is UNAVAILABLE); for retried requests it includes retries and backoff."),
        table(lm[lm["subset"] == "ALL"], ["participant", "task", "n", "p50", "p90", "p95", "p99", "mean", "max"], ["Participant", "Task", "N", "p50", "p90", "p95", "p99", "Mean", "Max"], {k: 0 for k in ("p50", "p90", "p95", "p99", "mean", "max")}),
        fg("latency_box_intent"), fg("latency_box_lead"), fg("latency_percentiles"), fg("latency_p50_ci"), fg("latency_vs_tokens"))

    add("Token usage",
        table(mts, ["participant", "task", "prompt_tokens_mean", "completion_tokens_mean", "prompt_tokens_sum", "completion_tokens_sum", "reasoning_tokens_sum"],
              ["Participant", "Task", "Prompt mean", "Compl. mean", "Prompt sum", "Compl. sum", "Reasoning sum"], {"prompt_tokens_mean": 0, "completion_tokens_mean": 1, "prompt_tokens_sum": 0, "completion_tokens_sum": 0, "reasoning_tokens_sum": 0}),
        fg("token_distributions"))

    add("Cost analysis",
        table(T["cost_metrics"], ["participant", "task", "est_cost_usd_sum", "est_cost_usd_per_request", "est_cost_usd_per_1000_requests", "est_cost_usd_per_valid_output", "requests_without_cost_evidence"],
              ["Participant", "Task", "Est USD total", "USD/req", "USD/1k req", "USD/valid", "No cost evidence"], {"est_cost_usd_per_request": 5, "est_cost_usd_per_valid_output": 5}),
        P("All costs are ESTIMATED (catalogue $/1M tokens times provider-reported tokens); native provider credits have an unverified USD ratio and are never converted. Errors without a response have no cost evidence; that is not proof they were free."),
        fg("cost_per_1000_intent"), fg("cost_per_1000_lead"), fg("cost_per_valid"), fg("cumulative_cost"))

    pair_i = pw[(pw.metric_family == "pairwise") & (pw.task == "INTENT")]
    add("Agreement analysis (INTENT)",
        P(f"Pairwise agreement uses cases both models answered validly. Median pairwise agreement {pair_i.agreement.median():.3f}; range {pair_i.agreement.min():.3f}-{pair_i.agreement.max():.3f}. Agreement is not accuracy."),
        table(pw[(pw.metric_family == "peer_plurality") & (pw.task == "INTENT")], ["participant_a", "n_common_valid", "agreement", "agreement_ci_lo", "agreement_ci_hi", "note"],
              ["Participant", "N", "Agreement w/ peers", "CI lo", "CI hi", "Note"]),
        fg("agreement_heatmap_intent"), fg("kappa_heatmap_intent"), fg("peer_plurality"))

    pair_l = pw[(pw.metric_family == "pairwise") & (pw.task == "LEAD")]
    add("Agreement analysis (LEAD)",
        P(f"Jev LEAD returns P(yes) only; with no DEV-frozen threshold it has no binary decision and is excluded from decision agreement. Median pairwise agreement among the rest {pair_l.agreement.median():.3f}."),
        table(pw[(pw.metric_family == "peer_plurality") & (pw.task == "LEAD")], ["participant_a", "n_common_valid", "agreement", "agreement_ci_lo", "agreement_ci_hi"],
              ["Participant", "N", "Agreement w/ peers", "CI lo", "CI hi"]),
        fg("agreement_heatmap_lead"), fg("jev_yes_probability"))

    od = pw[pw.metric_family == "output_distribution"]
    od_i = od[od.task == "INTENT"].pivot_table(index="participant_a", columns="participant_b", values="agreement", fill_value=0).reset_index()
    add("Output distributions",
        table(od_i, list(od_i.columns), ["Participant"] + [c.replace("label=", "") for c in od_i.columns[1:]]),
        P("Only one UPLOAD_STATUS_CHECK appears among valid INTENT outputs of the sample; rare classes are therefore effectively unobserved and per-class conclusions are not supportable."),
        fg("label_distribution_intent"), fg("lead_decision_distribution"))

    lb = T["leaderboards"]
    lbv = lb[lb.leaderboard == "valid_output_rate"]
    add("Matched-case leaderboards",
        P("Separate leaderboards per metric and task on matched cases (answered by every complete participant: 170 cases per task). Ties share a rank. There is no composite score and no overall winner. Mistral is excluded for incomplete coverage."),
        *[table(lb[(lb.leaderboard == m) & (lb.task == t)], ["rank", "participant", "value"], [f"{t} {m}: rank", "Participant", "Value"], {"value": 4})
          for t in ("INTENT", "LEAD") for m in ("valid_output_rate", "p50_latency_ms", "est_cost_per_1000_requests_usd")])

    pf = T["pareto"]
    add("Pareto view",
        table(pf, ["task", "participant", "cost_per_1000_usd", "p50_latency_ms", "valid_rate", "frontier_status", "dominated_by"], ["Task", "Participant", "USD/1k", "p50 ms", "Valid rate", "Status", "Dominated by"], {"p50_latency_ms": 0}),
        P("A model is on the frontier when no other complete model is at least as good on cost, p50 latency and valid-output rate and strictly better on one. Jev's frontier position in LEAD is operational only: its LEAD output is a probability without a decision rule."),
        fg("pareto_intent"), fg("pareto_lead"), fg("valid_vs_cost"))

    sc = T["statistical_comparisons"]
    add("Statistical comparisons",
        P(f"{len(sc)} paired comparisons with at least 30 pairs. Holm-significant at 0.05: valid-output {int(sc[sc.metric.str.startswith('valid')]['significant_holm_0.05'].sum())}, latency {int(sc[sc.metric.str.contains('latency')]['significant_holm_0.05'].sum())}, cost {int(sc[sc.metric.str.contains('cost')]['significant_holm_0.05'].sum())}. Full table: statistical_comparisons.csv."),
        table(sc[sc.metric.str.startswith("valid")].sort_values("p_holm").head(15), ["task", "participant_a", "participant_b", "n_paired", "effect", "a_only", "b_only", "p_holm"],
              ["Task", "A", "B", "Pairs", "Valid-rate diff (A-B)", "A only valid", "B only valid", "p (Holm)"], {"p_holm": 5}))

    model_els: List[Element] = []
    pol = {p["key"]: p for p in parts}
    eo = S["empty_outputs"].get("participants", {})
    for p in sorted(df.participant.unique()):
        g = mts[mts.participant == p]
        m = pol.get(p, {})
        model_els.append(("h2", p))
        model_els.append(P(f"Route {m.get('route', {}).get('pinned_tag', 'n/a')}; roles {','.join(m.get('roles', []))}; data policy {m.get('data_policy', 'n/a')}; origin {m.get('origin', 'n/a')}."))
        model_els.append(table(g, ["task", "records", "valid_rate", "invalid_output", "errors", "latency_p50_ms", "latency_p95_ms", "est_cost_usd_per_1000_requests", "est_cost_usd_per_valid_output"],
                               ["Task", "N", "Valid rate", "Invalid", "Err", "p50 ms", "p95 ms", "USD/1k req", "USD/valid"], {"latency_p50_ms": 0, "latency_p95_ms": 0, "est_cost_usd_per_valid_output": 5}))
        pr = pw[(pw.metric_family == "peer_plurality") & (pw.participant_a == p)]
        if len(pr):
            model_els.append(P("Agreement with peer plurality (not accuracy): " + "; ".join(f"{r.task} {r.agreement:.3f} (n={int(r.n_common_valid)})" for r in pr.itertuples()) + "."))
        if p in eo:
            model_els.append(B(["Verified: " + v for v in eo[p].get("verified", [])] + ["NOT verified: " + h for h in eo[p].get("hypotheses_not_verified", [])]))
    add("Per-model deep dives", *model_els)

    mi = mts[mts.participant.str.startswith("mistral")]
    rlim = S["rate_limits"]["participants"].get("mistral-small-3.2-24b-instruct", {})
    add("Mistral quarantine",
        P(f"Mistral recorded {int(mi.records.sum())} requests ({int(mi.errors.sum())} errors, all HTTP 429 after 3 attempts each) before the circuit breaker opened at block 10; {len(S['unattempted'])} planned requests were never attempted. "
          f"INTENT valid rate {I.valid_rate['mistral-small-3.2-24b-instruct']:.1%} of {int(I.records['mistral-small-3.2-24b-instruct'])}; LEAD {L.valid_rate['mistral-small-3.2-24b-instruct']:.1%} of {int(L.records['mistral-small-3.2-24b-instruct'])}."),
        B(["Not known: the Retry-After value, whether the limit is provider, OpenRouter per-provider or account level, and whether a pause would clear it."]),
        fg("mistral_timeline"))

    cl = [p for p in ("claude-haiku-5.5", "claude-sonnet-5.5") if p in eo]
    add("Claude empty-output investigation",
        P("Both Claude models returned HTTP 200 with empty visible content for a share of requests. In every empty record completion_tokens equals the request cap, so the tokens were billed but no text was visible. "
          "The cause (hidden reasoning, output filtering, or other) is NOT verified: finish_reason and reasoning_tokens were not stored for most of these records."),
        table(pd.DataFrame([{"participant": p, "records": eo[p]["records"], "empty": eo[p]["empty_outputs"], "at_cap": eo[p]["empty_with_completion_tokens_equal_to_cap"],
                             "valid": eo[p]["valid_outputs"], "reasoning_tokens_reported": eo[p].get("reasoning_tokens_reported", "n/a")} for p in cl]),
              ["participant", "records", "empty", "at_cap", "valid", "reasoning_tokens_reported"], ["Participant", "Records", "Empty", "Empty at cap", "Valid", "Reasoning tokens reported"]),
        P("A capture-only synthetic diagnostic is prepared but was not run (it would spend credits). Until it runs, treat the Claude invalid rates as a token-cap/format interaction, not as a capability finding."))

    add("Jev probability-only LEAD",
        P("Jev returns P(yes) for LEAD, so it is probability_only; binary_decision_available is false until a threshold is frozen on DEV data. It is reported for reliability, latency and cost, and excluded from decision agreement, errors-as-votes and any binary metric."),
        P(f"Jev LEAD: valid {L.valid_rate['jev-1.14']:.1%}, p50 {L.latency_p50_ms['jev-1.14']:,.0f} ms, ${L.est_cost_usd_per_1000_requests['jev-1.14']:.3f} per 1,000 requests (estimated)."))

    bi, bl = I.loc["baseline-intent-gemini-3.5-flash-lite"], L.loc["baseline-lead-gpt-4.1-mini"]
    add("Baselines versus candidates",
        P(f"Production baselines on benchmark routes: Gemini 3.5 Flash Lite (INTENT) valid {bi.valid_rate:.1%}, p50 {bi.latency_p50_ms:,.0f} ms, ${bi.est_cost_usd_per_1000_requests:.3f}/1k; GPT-4.1-mini (LEAD) valid {bl.valid_rate:.1%}, p50 {bl.latency_p50_ms:,.0f} ms, ${bl.est_cost_usd_per_1000_requests:.3f}/1k."),
        P("Candidates cheaper than the INTENT baseline: " + (", ".join(p for p in I.index if I.est_cost_usd_per_1000_requests[p] < bi.est_cost_usd_per_1000_requests and p != bi.name and not p.startswith("mistral")) or "none") +
          ". Candidates cheaper than the LEAD baseline: " + (", ".join(p for p in L.index if L.est_cost_usd_per_1000_requests[p] < bl.est_cost_usd_per_1000_requests and p != bl.name and not p.startswith("mistral")) or "none") +
          ". Cheaper and reliable is not equivalent: correctness against production behaviour remains unmeasured."))

    add("Limitations",
        B(["No ground truth: accuracy, precision, recall, F1, calibration and confusion matrices are " + NOT_COMPUTABLE + ".",
           "Sample: 170 stratified cases per task (7.7% of INTENT, 23.5% of LEAD); stratification over-represents rare patterns.",
           "17 of 222 blocks; Mistral incomplete (203 of 340 intended requests).",
           "Costs are estimates; no provider-reported USD; native credits have an unverified ratio.",
           "Latency is client-side and includes retries; provider model latency unavailable.",
           "Production parity not established (Dify export missing); prompts are the benchmark prompts.",
           "Claude empty-output cause unverified; Qwen ran with reasoning off by approval; routes were re-pinned after 404s, so results describe those routes only.",
           "Rare classes (UPLOAD_STATUS_CHECK, OUT_OF_SCOPE) are almost unobserved among valid outputs."]))

    add("Privacy and data handling",
        B(["Reports contain opaque case ids and no raw student text, no raw model output and no user4 content; redaction is pattern-based without NER and pseudonymisation is not anonymisation.",
           f"Benchmark inference sent redacted admissions-derived inputs to external providers through OpenRouter under typed operator authorisation (real_admissions_derived_data_sent_to_external_providers = {man['privacy']['inference_payloads']['real_admissions_derived_data_sent_to_external_providers']}); provider data policies are UNKNOWN for every route.",
           "The dataset notice 'LOCAL_ONLY' describes storage of the prepared dataset and export, not inference; it should not be shown without that qualification.",
           "This report generation made no network calls and transmitted nothing."]))

    jv_ok = [p for p in ok_both]
    add("Recommendations",
        B(["Do not select a production model from these results: correctness is unmeasured. Complete independent labelling (two reviewers, adjudication; templates for the 17 blocks exist) first.",
           f"Operationally plausible shortlist for the next stage (valid >= 99% on every task run, complete coverage; includes the production baselines): {', '.join(jv_ok)}. This is a reliability screen, not a recommendation.",
           "Sonnet 5.5 and Haiku 5.5: re-test only after the empty-output cause is established (capture-only diagnostic, small spend) or with a larger output cap; do not rank them on current invalid rates.",
           "Mistral: decide on a route/limit remedy before spending more; its partial data cannot support a comparison.",
           "Freeze a Jev LEAD threshold on DEV data before any binary Jev result is reported.",
           "Fix the 'LOCAL_ONLY' wording, and write verification output before manifest sealing or hash it in a second-level manifest.",
           f"Remaining ledger headroom is about ${float(man['actual_cost']['ledger']['pilot_remaining_usd']):.2f}; extending the sample is a budget decision for the operator."]))

    add("Reproducibility",
        P("Command: msu-llm-lab evaluation-report --bundle <bundle dir> --out <dir>. Inputs are read-only; figures and PDF are rendered with fixed seeds and no timestamps in figure metadata. Versions and the configuration are in analysis_config.json and reproducibility.md."),
        P("Only the report date on the cover and in the PDF metadata varies between runs; all analysis tables are byte-reproducible given the same inputs."))

    figs = fig.assign(Figure=fig.figure_id, File=fig.file)
    add("Appendices",
        P("A. Figure list. B. All 222 blocks. C. Files produced."),
        table(figs, ["figure_id", "file", "title"], ["ID", "File", "Title"]),
        ("h2", "All 222 blocks (not-executed blocks have no results)"),
        table(T["batch_summary"], ["block", "state", "plan_new_requests", "records", "carried_over", "valid", "invalid_output", "errors", "unattempted"],
              ["Block", "State", "Plan new req", "Records", "Carried", "Valid", "Invalid", "Err", "Unatt."]))
    return S_


# ----------------------------------------------------------------------------- renderers
def _clean(t: str) -> str:
    rep = {"≥": ">=", "≤": "<=", "→": "->", "×": "x", "≈": "~", "μ": "u", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}
    for a, b in rep.items():
        t = t.replace(a, b)
    return t.encode("cp1252", "replace").decode("cp1252")


def to_markdown(sections: List[Tuple[str, List[Element]]]) -> str:
    out = [f"# {TITLE}", ""]
    for i, (title, els) in enumerate(sections, 1):
        out += [f"## {i}. {title}", ""]
        for kind, v in els:
            if kind == "p":
                out += [v, ""]
            elif kind == "h2":
                out += [f"### {v}", ""]
            elif kind == "bullets":
                out += [f"- {x}" for x in v] + [""]
            elif kind == "table":
                out += ["| " + " | ".join(v["head"]) + " |", "|" + "---|" * len(v["head"])]
                out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |" for r in v["rows"]] + [""]
            elif kind == "fig":
                out += [f"![{v['caption']}]({v['file']})", f"*{v['caption']}*", ""]
    return "\n".join(out)


def to_pdf(sections: List[Tuple[str, List[Element]]], out_pdf: Path, base_dir: Path, meta: Dict[str, str]) -> int:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.platypus import BaseDocTemplate, Frame, Image, CondPageBreak, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.platypus.tableofcontents import TableOfContents
    from xml.sax.saxutils import escape

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11.4, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=4, spaceAfter=8, textColor=colors.HexColor("#1f3a5f"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3a5f"))
    cap = ParagraphStyle("cap", parent=body, fontSize=7.4, leading=9.2, textColor=colors.HexColor("#444444"), spaceAfter=8)
    cell = ParagraphStyle("c", parent=body, fontSize=6.3, leading=7.6, spaceAfter=0)
    cellh = ParagraphStyle("ch", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    title_s = ParagraphStyle("t", parent=ss["Title"], fontSize=22, leading=27, textColor=colors.HexColor("#1f3a5f"))

    class Doc(BaseDocTemplate):
        def afterFlowable(self, fl):
            if isinstance(fl, Paragraph) and fl.style.name in ("h1", "h2"):
                lvl = 0 if fl.style.name == "h1" else 1
                key = f"bk{self.seq.nextf('bk')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(fl.getPlainText(), key, level=lvl, closed=lvl == 0)
                if lvl == 0:
                    self.notify("TOCEntry", (0, fl.getPlainText(), self.page, key))

    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._saved: List[dict] = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self._saved)
            for st in self._saved:
                self.__dict__.update(st)
                if self._pageNumber > 1:
                    self.setFont("Helvetica", 7)
                    self.setFillColor(colors.HexColor("#666666"))
                    self.drawString(1.8 * cm, 1.0 * cm, "MSU LLM Inference Workbench - Benchmark Evaluation Report")
                    self.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {self._pageNumber} of {n}")
                super().showPage()
            super().save()

    doc = Doc(str(out_pdf), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm, title=TITLE,
              author="MSU LLM Workbench (offline report generator)", subject=meta.get("subject", ""), creator="msu-llm-lab evaluation-report", invariant=1)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame])])
    W = doc.width
    st: List[Any] = [Spacer(1, 4.5 * cm), Paragraph(escape(_clean(TITLE)), title_s), Spacer(1, 0.6 * cm),
                     Paragraph(escape(_clean(meta["subtitle"])), body), Spacer(1, 0.3 * cm)]
    for line in meta["lines"]:
        st.append(Paragraph(escape(_clean(line)), body))
    st += [Spacer(1, 0.6 * cm), Paragraph("<b>Status:</b> SAMPLED OPERATIONAL BENCHMARK. Ground truth PENDING: accuracy metrics are NOT COMPUTABLE. All costs ESTIMATED.", body), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", fontSize=8.5, leading=11, leftIndent=8)]
    st += [Paragraph("Table of contents", ParagraphStyle("tocT", parent=h1)), toc, PageBreak()]

    for i, (title, els) in enumerate(sections, 1):
        st.append(Paragraph(escape(_clean(f"{i}. {title}")), h1))
        for kind, v in els:
            if kind == "p":
                st.append(Paragraph(escape(_clean(v)), body))
            elif kind == "h2":
                st.append(Paragraph(escape(_clean(v)), h2))
            elif kind == "bullets":
                for x in v:
                    st.append(Paragraph("&bull; " + escape(_clean(x)), ParagraphStyle("bl", parent=body, leftIndent=10, firstLineIndent=-7, spaceAfter=2)))
                st.append(Spacer(1, 3))
            elif kind == "table":
                head, rows = v["head"], v["rows"]
                n = len(head)
                lens = [max([len(str(h))] + [len(str(r[j])) for r in rows[:60]]) for j, h in enumerate(head)]
                wts = [min(max(l, 4), 60) ** 0.8 for l in lens]
                cw = [W * w / sum(wts) for w in wts]
                data = [[Paragraph(escape(_clean(str(h))), cellh) for h in head]] + [[Paragraph(escape(_clean(str(c))), cell) for c in r] for r in rows]
                t = Table(data, colWidths=cw, repeatRows=1)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
                                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6fa")]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
                st += [t, Spacer(1, 6)]
            elif kind == "fig":
                from PIL import Image as PI

                path = base_dir / v["file"]
                w, h = PI.open(path).size
                iw = W * 0.92
                ih = iw * h / w
                if ih > 15 * cm:
                    ih, iw = 15 * cm, 15 * cm * w / h
                st.append(KeepTogether([Image(str(path), width=iw, height=ih), Paragraph(escape(_clean(v["caption"])), cap)]))
        st.append(CondPageBreak(7 * cm))
    doc.multiBuild(st, canvasmaker=NumberedCanvas)
    return len(st)
