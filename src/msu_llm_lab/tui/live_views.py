"""Experimental dashboard views built from recorded telemetry, stored predictions and approved labels.

Each view is a ViewModel (summary lines, a selectable table and a per-row inspector). Builders are pure functions of a LiveContext:
they read nothing, send nothing, and print UNAVAILABLE / GROUND TRUTH PENDING / NO LIVE INFERENCE YET instead of inventing values.
Provenance words are fixed: ACTUAL (provider-reported), ESTIMATED (catalogue price x reported tokens), UNRESOLVED (cost unknown),
LEGACY SIMULATED (old ledger rows only), UNAVAILABLE, PENDING.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from msu_llm_lab.core.budget import BudgetStatus
from msu_llm_lab.observability.aggregate import Aggregate, GroupStats, percentile
from msu_llm_lab.observability.analysis import FRONTIER_KEYS, MIN_SAMPLE
from msu_llm_lab.observability.labels import PENDING
from msu_llm_lab.observability.monitor import Analysis, DatasetInfo
from msu_llm_lab.tui.theme import (
    BORDER_CHARCOAL, BORDER_FOCUSED, COLOR_ERROR, COLOR_SUCCESS, COLOR_WARNING, STYLE_PANEL, STYLE_ROW_SELECTED,
    TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY,
)

NO_LIVE = "NO LIVE INFERENCE YET"
UNAVAILABLE = "UNAVAILABLE"
DASH = "—"
PARITY = "PRODUCTION_PARITY_NOT_ESTABLISHED"
SPARK = " ▁▂▃▄▅▆▇█"


@dataclass
class LiveContext:
    now: float
    width: int
    budget: Optional[BudgetStatus]
    agg: Aggregate
    analysis: Analysis
    dataset: DatasetInfo
    roster: List[Dict[str, Any]]
    window_label: str = "5 min"
    events_malformed: int = 0
    review_ids: List[str] = field(default_factory=list)       # builder-flagged lead cases (opaque ids)
    review_count: Optional[int] = None


@dataclass
class Row:
    cells: List[str]
    detail: List[str]


@dataclass
class ViewModel:
    title: str
    source: str
    summary: List[Text]
    headers: List[str]
    rows: List[Row]
    footnotes: List[str] = field(default_factory=list)
    empty: Optional[str] = None


# ------------------------------------------------------------------ formatting
def _f(v: Optional[float], nd: int = 1, suffix: str = "") -> str:
    return DASH if v is None else f"{v:.{nd}f}{suffix}"


def _pct(v: Optional[float], nd: int = 1) -> str:
    return DASH if v is None else f"{100 * v:.{nd}f}%"


def _usd(v: Optional[Decimal], nd: int = 6) -> str:
    return DASH if v is None else f"${v:.{nd}f}"


def _ci(ci: Optional[Tuple[float, float]]) -> str:
    return "" if ci is None else f" (95% CI {100 * ci[0]:.0f}-{100 * ci[1]:.0f}%)"


def spark(values: Sequence[Optional[float]], width: int = 40) -> str:
    """One character per real time bucket; a bucket with no measurement is blank, never zero."""
    vals = list(values)[-width:]
    nums = [v for v in vals if v is not None]
    if not nums:
        return "·" * min(width, max(1, len(vals)))
    hi = max(nums) or 1.0
    return "".join(" " if v is None else SPARK[max(1, min(8, int(v / hi * 8)))] if v > 0 else "▁" for v in vals)


def _line(label: str, value: str, style: str = TEXT_PRIMARY) -> Text:
    t = Text()
    t.append(f"{label:<22}", style=TEXT_SECONDARY)
    t.append(value, style=style)
    return t


def _model_name(ctx: LiveContext, key: str) -> str:
    for r in ctx.roster:
        if r["key"] == key or r["model_id"] == key:
            return r["display_name"] or r["key"]
    return key


# ------------------------------------------------------------------ overview
def view_overview(ctx: LiveContext) -> ViewModel:
    a, b, ds, lab = ctx.agg, ctx.budget, ctx.dataset, ctx.analysis.labels
    ready = sum(r["readiness"] == "READY" for r in ctx.roster)
    smoke = sum(r["readiness"] == "SMOKE_READY" for r in ctx.roster)
    blocked = len(ctx.roster) - ready - smoke
    summary = [
        _line("Live inference", NO_LIVE if not a.has_live_inference else f"{a.n_started} requests started, {a.n_terminal} finished, {a.active} active"),
        _line("Roster readiness", f"{ready} ready / {smoke} smoke-ready / {blocked} blocked of {len(ctx.roster)}"),
        _line("Dataset", f"{ds.version or 'NOT PREPARED'}  ({ds.intent_rows if ds.intent_rows is not None else UNAVAILABLE} intent candidates)"),
        _line("Ground truth", lab.status, COLOR_WARNING if not lab.available else COLOR_SUCCESS),
        _line("Parity", PARITY, COLOR_WARNING),
        _line("Session policy", ds.session_policy, COLOR_WARNING),
    ]
    q_state = lab.status if ctx.analysis.intent_quality else "NO PREDICTIONS RECORDED"
    cal = ctx.analysis.calibration
    rows = [
        Row(["1", "Classification quality", q_state], [
            "Accuracy, macro-F1, per-class P/R/F1 and the confusion matrix are computed only from human-approved labels.",
            "Schema validity and invalid-label rate are measured from stored predictions and need no labels.",
            f"Predictions on disk: {len(ctx.analysis.predictions)}.  Approved labels: {lab.n_approved}."]),
        Row(["2", "Confidence and calibration", "UNAVAILABLE until a native distribution and approved labels exist" if not cal else lab.status if not lab.available else "SCORED"], [
            "Only native probability distributions are calibratable; textual or self-reported confidence is never used."]),
        Row(["3", "Model agreement", f"{len(ctx.analysis.agreement_intent.models) if ctx.analysis.agreement_intent else 0} models with intent predictions"], [
            "Agreement is not correctness; it only shows how often models give the same answer."]),
        Row(["4", "Lead Qualifier", PENDING if not lab.lead else "SCORED"], [
            "YES precision/recall need approved lead labels. The Jev threshold is chosen on DEV only, then frozen."]),
        Row(["5", "Latency, reliability, cost", NO_LIVE if not a.has_live_inference else f"{a.n_terminal} recorded requests"], [
            "Measured from the request event log only."]),
        Row(["6", "Robustness and readiness", f"{blocked} roster entries blocked; student data blocked by unverified data policy"], [
            "Student-derived text may not be sent to any provider until a data policy is verified and approved."]),
    ]
    return ViewModel("OVERVIEW", "recorded events, ledger, preparation manifest", summary, ["#", "Experimental category", "State"], rows,
                     ["Open a category with its number key or the sidebar. All views are read-only."])


# ------------------------------------------------------------------ 1 quality
def _valid_rate(recs) -> Tuple[int, int]:
    ok = sum(r.status == "OK" for r in recs)
    scored = sum(r.status in ("OK", "INVALID_OUTPUT") for r in recs)
    return ok, scored


def view_quality(ctx: LiveContext) -> ViewModel:
    an = ctx.analysis
    rows: List[Row] = []
    for q in an.intent_quality:
        recs = an.runs[(q.model_key, "INTENT")]
        ok, scored = _valid_rate(recs)
        m = q.metrics
        detail = [f"Main Intent | model {_model_name(ctx, q.model_key)} | run {q.run_id} | {q.n_predictions} predictions, {q.n_request_errors} request errors",
                  f"Schema/label validity (measured, no labels needed): {ok}/{scored}",
                  f"Invalid-label rate: {_pct(m.invalid_rate) if m else UNAVAILABLE + ' (' + PENDING + ')'}"]
        if m:
            detail += [f"Scored n={m.n_scored}  accuracy {_pct(m.accuracy)}{_ci(q.accuracy_ci)}  macro-F1 {_pct(m.macro_f1)}", "", "Per class (P / R / F1 / support):"]
            for c, s in m.per_class.items():
                detail.append(f"  {c:<22}{_pct(s.precision):>9}{_pct(s.recall):>9}{_pct(s.f1):>9}{s.support:>6}" + ("  INSUFFICIENT SUPPORT" if s.support < 5 else ""))
            cols = list(m.confusion[next(iter(m.confusion))].keys())
            detail += ["", "Confusion matrix (rows expected, columns predicted): " + " | ".join(c[:6] for c in cols)]
            for c, r in m.confusion.items():
                detail.append(f"  {c[:22]:<22}" + " ".join(f"{r[x]:>6}" for x in cols))
        else:
            detail.append(f"Accuracy, macro-F1, per-class and confusion matrix: {PENDING}. Predictions are never used as labels.")
        detail += q.flags
        rows.append(Row(["INTENT", _model_name(ctx, q.model_key), str(q.n_predictions), _pct(ok / scored) if scored else UNAVAILABLE,
                         _pct(m.accuracy) if m else PENDING, _pct(m.macro_f1) if m else PENDING, "; ".join(q.flags[:1])], detail))
    for lr in an.lead:
        recs = an.runs[(lr.model_key, "LEAD")]
        ok, scored = _valid_rate(recs)
        m = lr.metrics
        detail = [f"Lead Qualifier | model {_model_name(ctx, lr.model_key)} | run {lr.run_id} | {lr.n_predictions} predictions",
                  f"JSON/schema validity (measured): {ok}/{scored}",
                  f"YES {lr.n_yes}  NO {lr.n_no}  undecided {lr.n_undecided}"]
        if m:
            detail += [f"Scored n={m.n_scored}  accuracy {_pct(m.accuracy)}  YES P {_pct(m.yes_precision)} R {_pct(m.yes_recall)} F1 {_pct(m.yes_f1)}",
                       f"False-positive rate {_pct(m.false_positive_rate)}  false-negative rate {_pct(m.false_negative_rate)}  (TP {m.tp} FP {m.fp} FN {m.fn} TN {m.tn})"]
        else:
            detail.append(f"Accuracy and YES precision/recall/F1: {PENDING}")
        detail += lr.flags
        rows.append(Row(["LEAD", _model_name(ctx, lr.model_key), str(lr.n_predictions), _pct(ok / scored) if scored else UNAVAILABLE,
                         _pct(m.accuracy) if m else PENDING, _pct(m.yes_f1) if m else PENDING, "; ".join(lr.flags[:1])], detail))
    summary = [_line("Ground truth", an.labels.status, COLOR_WARNING if not an.labels.available else COLOR_SUCCESS),
               _line("Approved labels", f"intent {len(an.labels.intent)}  lead {len(an.labels.lead)}  (rows not approved: {an.labels.n_not_approved})")]
    if not an.labels.available:
        summary.append(Text("Supervised metrics are withheld: historical chatbot answers and model predictions are never used as labels.", style=COLOR_WARNING))
    return ViewModel("CLASSIFICATION QUALITY", "stored prediction records; approved labels", summary,
                     ["Task", "Model", "N", "Valid", "Accuracy", "F1 (macro / YES)", "Flag"], rows,
                     empty="NO PREDICTION RECORDS YET: run the smoke test or a benchmark to create some")


# ------------------------------------------------------------------ 2 calibration
def view_calibration(ctx: LiveContext) -> ViewModel:
    rows = []
    an = ctx.analysis
    for c in an.calibration:
        rec = an.runs[(c.model_key, "INTENT")]
        label_means: Dict[str, List[float]] = {}
        for r in rec:
            for k, v in (r.label_probs or {}).items():
                label_means.setdefault(k, []).append(v)
        detail = [f"{_model_name(ctx, c.model_key)}: {c.n_with_distribution}/{c.n_records} records carry a native probability distribution ({c.n_invalid_distribution} malformed)",
                  f"Selected-label probability (mean): {_f(c.mean_selected_prob, 3)}",
                  f"Separate native confidence field (mean, n={c.n_with_native_confidence}): {_f(c.mean_native_confidence, 3)}"]
        if label_means:
            detail += ["Mean per-label probability:"] + [f"  {k:<22}{sum(v) / len(v):.3f}  (n={len(v)})" for k, v in sorted(label_means.items())]
        if c.unsupported_reason:
            detail.append(c.unsupported_reason)
        if c.status == "SCORED":
            detail += [f"Calibration sample n={c.n_calibration}  ECE {_f(c.ece, 4)}  Brier {_f(c.brier, 4)}", "Reliability diagram (confidence bin: n, mean confidence, accuracy):"]
            detail += [f"  [{b.lo:.1f},{b.hi:.1f})  n={b.n:<4} conf {b.mean_confidence:.3f}  acc {b.accuracy:.3f}" for b in c.bins]
            detail += ["Coverage vs accuracy by threshold:"] + [f"  >= {t:.2f}: kept {n}, accuracy {_pct(a)}" for t, n, a in c.coverage]
            detail += [f"High-confidence errors (>= 0.9): {len(c.high_conf_errors)}"] + [f"  {cid}: {conf:.3f} predicted {p}, approved {e}" for cid, conf, p, e in c.high_conf_errors[:10]]
        elif c.status == "PENDING" or c.status == PENDING:
            detail.append(f"ECE, Brier, reliability diagram, high-confidence errors, coverage curve: {PENDING}")
        detail += c.flags
        state = c.status if c.status != PENDING else PENDING
        rows.append(Row([_model_name(ctx, c.model_key), str(c.n_with_distribution), _f(c.mean_selected_prob, 3), _f(c.mean_native_confidence, 3),
                         _f(c.ece, 4) if c.ece is not None else state, _f(c.brier, 4) if c.brier is not None else state], detail))
    summary = [_line("Rule", "only native probability distributions are calibrated; text such as '90% confident' is never used"),
               _line("Ground truth", an.labels.status, COLOR_WARNING if not an.labels.available else COLOR_SUCCESS)]
    return ViewModel("CONFIDENCE AND CALIBRATION", "stored predictions; approved labels", summary,
                     ["Model", "N dist.", "Selected P (mean)", "Native conf. (mean)", "ECE", "Brier"], rows,
                     empty="NO INTENT PREDICTIONS RECORDED: confidence data does not exist yet")


# ------------------------------------------------------------------ 3 agreement
def view_agreement(ctx: LiveContext) -> ViewModel:
    an = ctx.analysis
    rows = []
    summary: List[Text] = []
    have = {k[0] for k in an.runs}
    missing = [r["display_name"] or r["key"] for r in ctx.roster if r["key"] not in have]
    for rep in (an.agreement_intent, an.agreement_lead):
        if rep is None:
            continue
        summary.append(_line(f"{rep.task} contributors", ", ".join(f"{_model_name(ctx, m)}" for m in rep.models) or "none"))
        summary.append(_line(f"{rep.task} cases", f"{rep.n_cases_any} seen, {rep.n_cases_all_models} answered by every contributing model"))
        for m, why in rep.excluded.items():
            summary.append(Text(f"  excluded {_model_name(ctx, m)}: {why}", style=COLOR_WARNING))
        for p in rep.pairs:
            fr = rep.frontier_agreement
            detail = [f"{rep.task}: {_model_name(ctx, p.a)} vs {_model_name(ctx, p.b)}", f"Cases compared: {p.n_common} (cases both models answered)",
                      f"Agreement {_pct(p.agreement)}  Cohen's kappa {_f(p.kappa, 3)}  {p.note}",
                      "This is agreement, not correctness: two models can agree and both be wrong.", "",
                      "Agreement with frontier references:"]
            for mk in (p.a, p.b):
                detail.append(f"  {_model_name(ctx, mk)}: " + ", ".join(f"{_model_name(ctx, fk)} {_pct(v)}" for fk, v in fr.get(mk, {}).items()) if fr.get(mk) else f"  {_model_name(ctx, mk)}: no frontier reference predictions")
            detail += ["", f"Disagreement cases across all contributors: {len(rep.disagreements)} (opaque ids, first 8):"]
            detail += [f"  {cid}: " + ", ".join(f"{_model_name(ctx, m)}={v}" for m, v in preds.items()) for cid, preds in rep.disagreements[:8]]
            ue = rep.unanimous_errors
            detail += ["", "Unanimous errors vs approved labels: " + (PENDING if ue is None else f"{len(ue)} ({rep.unanimous_error_basis})")]
            rows.append(Row([rep.task, _model_name(ctx, p.a), _model_name(ctx, p.b), str(p.n_common), _pct(p.agreement), _f(p.kappa, 3), p.note], detail))
    summary.append(_line("No predictions from", ", ".join(missing) or "none"))
    summary.append(Text("Agreement is not correctness. Unanimous errors need approved labels (" + an.labels.status + ").", style=COLOR_WARNING))
    return ViewModel("MODEL AGREEMENT", "stored prediction records", summary, ["Task", "Model A", "Model B", "N common", "Agreement", "Kappa", "Note"], rows,
                     empty="NO MODEL PAIRS TO COMPARE: at least two models need predictions on shared cases")


# ------------------------------------------------------------------ 4 lead
def view_lead(ctx: LiveContext) -> ViewModel:
    an = ctx.analysis
    rows = []
    for lr in an.lead:
        m = lr.metrics
        detail = [f"{_model_name(ctx, lr.model_key)} | run {lr.run_id}", f"Distribution: YES {lr.n_yes}  NO {lr.n_no}  undecided {lr.n_undecided}  (native P(yes) on {lr.n_with_prob} cases)"]
        if lr.n_with_prob and lr.frozen is None:
            detail.append(lr.threshold_note or f"No frozen threshold: a dev-set threshold needs approved lead labels on the DEV split. 0.5 is a reference only.")
        if lr.frozen:
            f = lr.frozen
            detail += [f"Frozen threshold {f.threshold:.4f} ({f.criterion} on {f.selected_on}, n={f.n_dev_scored}, dev F1 {_f(f.dev_f1_at_threshold, 3)}); 0.5 reference dev F1 {_f(f.dev_f1_at_reference, 3)}"]
        if lr.held_out:
            h = lr.held_out
            detail += [f"Held-out TEST (threshold not tuned on it): n={h.n_scored}  P {_pct(h.at_frozen.precision)} R {_pct(h.at_frozen.recall)} F1 {_pct(h.at_frozen.f1)}  FPR {_pct(h.at_frozen.false_positive_rate)} FNR {_pct(h.at_frozen.false_negative_rate)}"]
        elif lr.frozen:
            detail.append("Held-out TEST: no approved TEST labels with predictions yet")
        detail.append(f"YES precision/recall/F1: " + (f"{_pct(m.yes_precision)} / {_pct(m.yes_recall)} / {_pct(m.yes_f1)}" if m else PENDING))
        detail.append(f"False positives / negatives: " + (f"{m.fp} / {m.fn}" if m else PENDING))
        detail += lr.flags
        rows.append(Row([_model_name(ctx, lr.model_key), f"{lr.n_yes}/{lr.n_no}/{lr.n_undecided}",
                         _pct(m.yes_precision) if m else PENDING, _pct(m.yes_recall) if m else PENDING, _pct(m.yes_f1) if m else PENDING,
                         f"{lr.frozen.threshold:.3f}" if lr.frozen else "NOT FROZEN"], detail))
    summary = [_line("Threshold rule", "chosen on DEV only (max F1), frozen, then applied unchanged to TEST; 0.5 shown as reference"),
               _line("Lead ground truth", f"{len(an.labels.lead)} approved" if an.labels.lead else PENDING, COLOR_WARNING if not an.labels.lead else COLOR_SUCCESS),
               _line("Needs human review", f"{ctx.review_count} sessions with builder ambiguity or unresolved rules" if ctx.review_count is not None else UNAVAILABLE + " (no preparation)")]
    if ctx.review_ids:
        summary.append(_line("  first ids", ", ".join(ctx.review_ids[:6])))
    return ViewModel("LEAD QUALIFIER", "stored predictions; approved labels; builder metadata (no text)", summary,
                     ["Model", "YES/NO/undecided", "YES precision", "YES recall", "YES F1", "Threshold"], rows,
                     empty="NO LEAD PREDICTIONS RECORDED YET")


# ------------------------------------------------------------------ 5 latency, reliability, cost
def _group_row(ctx: LiveContext, g: GroupStats, label: str) -> Row:
    tail = "" if g.p_tail_reliable else f"  (n={len(g.latencies_ms)} < 20: P95/P99 indicative)"
    detail = [f"{label}: {g.n_terminal} finished requests (OK {g.n_ok}, invalid {g.n_invalid}, errors {g.n_error})",
              f"End-to-end latency P50 {_f(g.p(50), 0, ' ms')}  P95 {_f(g.p(95), 0, ' ms')}  P99 {_f(g.p(99), 0, ' ms')}{tail}",
              "Provider-reported model latency: " + (f"P50 {_f(percentile(g.model_latencies_ms, 50), 0, ' ms')} (n={len(g.model_latencies_ms)})" if g.model_latencies_ms
                                                     else UNAVAILABLE + " (provider reports none; client round-trip is not substituted)"),
              f"Tokens (provider-reported): prompt {g.prompt_tokens} completion {g.completion_tokens} on {g.n_token_known} requests; unknown on {g.n_token_unknown}",
              f"Throughput: {_f(g.tokens_per_second, 1, ' tok/s')} from known completion tokens over " + (g.tps_source or UNAVAILABLE) + f" ({g.tps_n} requests)",
              f"Timeouts {g.n_timeouts}  rate limits (429) {g.n_rate_limited}  5xx {g.n_provider_failures}  retries {g.n_retries}",
              f"Cost ACTUAL (provider-reported) {_usd(g.actual_cost if g.n_actual_cost else None)} on {g.n_actual_cost} requests",
              f"Cost ESTIMATED (catalogue price x tokens) {_usd(g.estimated_cost if g.n_estimated_cost else None)} on {g.n_estimated_cost} requests",
              f"Cost UNRESOLVED requests: {g.n_cost_unresolved}   native provider cost (credits, unit unverified): {g.native_cost if g.n_native_cost else UNAVAILABLE}",
              f"Per request actual {_usd(g.cost_per_request('actual'))} / estimated {_usd(g.cost_per_request('estimated'))};  per 1,000 cases actual {_usd(g.cost_per_1000('actual'), 4)} / estimated {_usd(g.cost_per_1000('estimated'), 4)}",
              "Cost per correct prediction: " + PENDING + " (needs approved labels)"]
    return Row([label, str(g.n_terminal), _f(g.p(50), 0), _f(g.p(95), 0), _f(g.p(99), 0), _f(g.tokens_per_second, 1),
                f"{g.n_timeouts}/{g.n_rate_limited}/{g.n_provider_failures}/{g.n_retries}", _usd(g.cost_per_request("actual")), _usd(g.cost_per_request("estimated"))], detail)


def view_latency(ctx: LiveContext) -> ViewModel:
    a = ctx.agg
    rows = [_group_row(ctx, g, _model_name(ctx, k)) for k, g in sorted(a.by_model.items()) if g.n_terminal]
    rows += [_group_row(ctx, g, f"provider: {k}") for k, g in sorted(a.by_provider.items()) if g.n_terminal]
    summary: List[Text] = []
    if not a.has_live_inference:
        summary.append(Text(NO_LIVE, style=f"bold {COLOR_WARNING}"))
    else:
        summary.append(_line("Active requests", f"{a.active}" + (f"  ({a.stale} stale: no end event after 15 min)" if a.stale else "")))
        for name, label in (("latency_e2e", "E2E latency (mean/bucket)"), ("latency_model", "Model latency (reported)"), ("tokens_per_second", "Tokens per second"),
                            ("concurrency", "Concurrency"), ("error_rate", "Error rate"), ("cost_actual", "Cost actual"), ("cost_estimated", "Cost estimated"), ("cost_unresolved", "Cost unresolved")):
            ser = [v for _, v in a.series.get(name, [])]
            last = next((v for v in reversed(ser) if v is not None), None)
            note = "" if any(v is not None for v in ser) else "  (none reported)" if name == "latency_model" else "  (no data in window)"
            summary.append(_line(f"{label} [{a.series_units.get(name, '')}]", f"{spark(ser)}  last {_f(last, 2)}{note}"))
        summary.append(Text(f"Window {ctx.window_label}, {a.bucket_s:.0f}s buckets; blank = no measurement in that bucket.  Malformed log lines skipped: {ctx.events_malformed}", style=TEXT_MUTED))
    if a.n_quarantined_simulated:
        summary.append(Text(f"{a.n_quarantined_simulated} SIMULATED event(s) quarantined and excluded from every figure here", style=COLOR_ERROR))
    return ViewModel("LIVE LATENCY, RELIABILITY AND COST", "request event log (ACTUAL requests only)", summary,
                     ["Group", "Reqs", "P50 ms", "P95 ms", "P99 ms", "tok/s", "TO/429/5xx/retry", "$/req actual", "$/req est."], rows,
                     empty=NO_LIVE)


# ------------------------------------------------------------------ 6 robustness
def view_robustness(ctx: LiveContext) -> ViewModel:
    an = ctx.analysis
    rows = []
    for r in ctx.roster:
        key = r["key"]
        recs = [x for (mk, _), rs in an.runs.items() if mk == key or mk == r["model_id"] for x in rs]
        g = ctx.agg.by_model.get(r["model_id"]) or ctx.agg.by_model.get(key)
        rep = [x for x in an.repeat if x.model_key in (key, r["model_id"])]
        detail = [f"{r['display_name'] or key}  ({r['model_id']})  protocol {r['protocol']}",
                  f"Pinned route: {r['pinned_route'] or UNAVAILABLE}   route status {r['route_status']}   price {r['route_price'] or UNAVAILABLE}",
                  f"Readiness {r['readiness']}   blockers: {', '.join(r['blockers']) or 'none for synthetic data'}",
                  f"Student-data blockers: {', '.join(r['student_data_blockers']) or 'none'}   data policy {r['data_policy']}"]
        if recs:
            detail += ["Prompt fingerprints: " + ", ".join(sorted({x.prompt_version for x in recs})),
                       "Dataset versions: " + ", ".join(sorted({x.dataset_version or 'none' for x in recs})),
                       "Input contract: " + ", ".join(sorted({x.input_contract_version for x in recs})),
                       "Run ids: " + ", ".join(sorted({x.run_id for x in recs})[:4]),
                       "Serving provider(s) reported: " + (", ".join(sorted({x.serving_provider for x in recs if x.serving_provider})) or UNAVAILABLE)]
            invalid = sum(x.status == "INVALID_OUTPUT" for x in recs)
            detail.append(f"Invalid responses: {invalid}/{len(recs)}   errors: {sum(x.status == 'ERROR' for x in recs)}   retried requests: {sum((x.attempts or 1) > 1 for x in recs)}")
        else:
            detail.append("No prediction records for this model.")
        detail.append("Repeatability: " + ("; ".join(f"{x.task} {x.n_identical}/{x.n_cases_repeated} identical across repeated runs" + ("" if x.n_cases_repeated >= MIN_SAMPLE else " (INSUFFICIENT SAMPLE)") for x in rep) or "UNAVAILABLE (needs two or more runs on the same cases)"))
        hs = [h for h in an.history if h[0] in (key, r["model_id"])]
        detail.append("History-completeness slices: " + ("; ".join(f"{s}: n={n} acc {_pct(acc)}" + ("" if n >= MIN_SAMPLE else " INSUFFICIENT") for _, s, n, acc in hs) if hs else f"{PENDING} (needs approved labels and case metadata)"))
        detail.append("Language slices: UNAVAILABLE (the prepared data carries no reviewed language field)")
        rows.append(Row([r["display_name"] or key, r["readiness"], r["pinned_route"] or "-", str(len(recs)), ", ".join(r["blockers"][:2]) or "none"], detail))
    summary = [_line("Parity", PARITY, COLOR_WARNING), _line("Session policy", ctx.dataset.session_policy, COLOR_WARNING),
               _line("Privacy", "no provider data policy verified: student-derived data may not be sent", COLOR_WARNING),
               _line("Dataset / contract", f"{ctx.dataset.version or 'NOT PREPARED'}  /  input_contract_v1")]
    return ViewModel("ROBUSTNESS AND OPERATIONAL READINESS", "registry, stored predictions, preparation manifest", summary,
                     ["Participant", "Readiness", "Pinned route", "Records", "Blockers"], rows, ["Slices below the minimum sample are flagged INSUFFICIENT, never hidden."])


# ------------------------------------------------------------------ dataset review and labels
def view_dataset(ctx: LiveContext) -> ViewModel:
    ds, lab = ctx.dataset, ctx.analysis.labels
    ann = ds.prep_dir / "annotations" if ds.prep_dir else None
    rows = [
        Row(["Selected dataset", f"{ds.version or 'NOT PREPARED'}"], [f"Folder: {ds.prep_dir or 'none'}", "Derived locally, redacted, git-ignored. Raw student text is never shown on the dashboard."]),
        Row(["Intent candidates", str(ds.intent_rows) if ds.intent_rows is not None else UNAVAILABLE], ["One per student message; every expected label is empty until a human approves it."]),
        Row(["Lead sessions", (f"{ds.lead_sessions.get('2h')} (2 h) / {ds.lead_sessions.get('30m')} (30 min)" if ds.lead_sessions else UNAVAILABLE)],
            ["Two provisional policies; neither is selected. Choosing one is a human decision this screen cannot make."]),
        Row(["Session policy", ds.session_policy], ["Not selected. The dashboard never chooses a boundary."]),
        Row(["Approved labels", f"intent {len(lab.intent)} / lead {len(lab.lead)}"], [lab.status, f"Source: {lab.source or 'no annotations file'}", f"Rows present {lab.n_rows}, not approved {lab.n_not_approved}, malformed {lab.n_malformed}", "Approval needs a named reviewer; predictions and suggested labels never count."]),
        Row(["Annotations folder", "PRESENT (protected)" if ds.labels_dir_present else "none yet"], [f"{ann or 'n/a'}", "Read-only here. prepare-classification-data refuses to run over an existing annotations folder; refresh never writes."]),
        Row(["Challenge view", "heuristic, unvalidated"], [ds.views_note]),
        Row(["Production parity", PARITY], ["The Dify workflow export is missing; reconstructed inputs are not claimed to match production."]),
        Row(["Label approval", "workflow not built"], ["There is no label-approval screen yet. Review and approve outside the dashboard; this view only reads the result."]),
    ]
    return ViewModel("DATASET REVIEW AND LABELS", "preparation manifest (counts only, no student text)", [_line("Status", lab.status, COLOR_WARNING if not lab.available else COLOR_SUCCESS)],
                     ["Item", "State"], rows, ["Nothing on this screen approves labels, selects a session policy, regenerates files or edits annotations."])


FORMAT_CHOICES = ("jsonl", "csv", "zip")
PREVIEW_LIMIT = 200
KIND_SHORT = {"SYNTHETIC_FIXTURE": "SYNTHETIC", "REAL_DATA_BENCHMARK": "REAL-DATA", "UNCLASSIFIED": "UNCLASSIFIED"}


@dataclass
class ExplorerState:
    """Filters and export settings for the Trace Explorer. Changing them never sends anything and never writes a file."""
    dest: Path = Path("reports/exports")
    model: Optional[str] = None
    task: Optional[str] = None
    run: Optional[str] = None
    status: Optional[str] = None
    error_class: Optional[str] = None
    kind: Optional[str] = None
    fmt: str = "jsonl"
    last_result: Any = None
    last_error: Optional[str] = None
    progress: Optional[str] = None

    def selection(self):
        from msu_llm_lab.observability.export import ExportSelection
        return ExportSelection(models=(self.model,) if self.model else None, runs=(self.run,) if self.run else None, task=self.task,
                               statuses=(self.status,) if self.status else None, error_categories=(self.error_class,) if self.error_class else None,
                               data_kinds=(self.kind,) if self.kind else None)

    def key(self):
        return (self.model, self.task, self.run, self.status, self.error_class, self.kind)


def _short_run(run_id: str, n: int = 22) -> str:
    return run_id if len(run_id) <= n else run_id[: n - 9] + "…" + run_id[-8:]


def _num(v: Optional[float], nd: int = 0) -> str:
    return DASH if v is None else f"{v:.{nd}f}"


def _not_measured(r: Dict[str, Any]) -> List[str]:
    """Honest reasons for absent measurements, shown only in the detail note. The machine-readable record keeps the nulls."""
    q, u, k, m = r["request"], r["usage"], r["cost"], r["markers"]
    failed = m["failed_call"]
    out = []
    if q["model_latency_ms"] is None:
        out.append("model latency: " + ("request failed" if failed else "not reported by provider"))
    if u["prompt_tokens"] is None or u["completion_tokens"] is None:
        out.append("tokens: " + ("request failed" if failed else "not reported by provider"))
    if k["actual_cost_usd"] is None and not failed:
        out.append("actual USD cost: " + ("provider reports credits only (USD ratio unverified)" if k["native_cost"] else "not reported by provider"))
    if k["estimated_cost_usd"] is None and not failed:
        out.append("estimated cost: needs reported tokens and a verified price")
    if q["request_started_at"] is None:
        out.append("start time: no event-log entry")
    return out


def _row_detail(r: Dict[str, Any]) -> List[str]:
    i, v, q, o, c, u, k, g, m = (r[x] for x in ("identity", "versions", "request", "output", "confidence", "usage", "cost", "ground_truth", "markers"))
    lines = [f"{i['model_key']}  |  {i['task']}  |  run {i['run_id']}  |  case {i['case_id']}  |  {KIND_SHORT.get(m['data_kind'], m['data_kind'])}"]
    st = f"status {q['status']}"
    if q["error_category"]:
        st += f"  class {q['error_category']}"
    if q["attempts"] is not None:
        st += f"  attempts {q['attempts']}"
    if q["retry_causes"]:
        st += f"  retries: {', '.join(q['retry_causes'])}"
    lines.append(st)
    if q["error_message"]:
        lines.append(f"error (scrubbed): {q['error_message']}")
    if q["validation_issues"]:
        lines.append("validation: " + ", ".join(q["validation_issues"]))
    timing = []
    if q["request_started_at"]:
        timing.append(f"started {q['request_started_at']}")
    if q["latency_ms_end_to_end"] is not None:
        timing.append(f"end-to-end {q['latency_ms_end_to_end']:.0f} ms (client-measured)")
    if q["model_latency_ms"] is not None:
        timing.append(f"model latency {q['model_latency_ms']:.0f} ms (provider-reported)")
    if timing:
        lines.append("timing: " + "  ".join(timing))
    if q["status"] == "OK":
        pred = o["predicted_label"] or (None if o["decision"] is None else ("YES" if o["decision"] else "NO"))
        parts = [f"prediction {pred}" if pred else "prediction: none (P(yes) only, no frozen threshold)"]
        if o["decision_source"]:
            parts.append(f"source {o['decision_source']}")
        if c["selected_label_probability"] is not None:
            parts.append(f"selected-label P {c['selected_label_probability']:.3f}")
        if c["native_confidence"] is not None:
            parts.append(f"native confidence {c['native_confidence']:.3f}")
        if c["yes_probability"] is not None:
            parts.append(f"P(yes) {c['yes_probability']:.3f}")
        if c["frozen_threshold"] is not None:
            parts.append(f"frozen threshold {c['frozen_threshold']:.3f}")
        lines.append("output: " + "  ".join(parts))
    if u["prompt_tokens"] is not None and u["completion_tokens"] is not None:
        lines.append(f"tokens: in {u['prompt_tokens']} / out {u['completion_tokens']}  (source {u['usage_source']})")
    cost = []
    if k["actual_cost_usd"]:
        cost.append(f"actual ${k['actual_cost_usd']}")
    if k["estimated_cost_usd"]:
        cost.append(f"estimated ${k['estimated_cost_usd']}")
    if k["native_cost"]:
        cost.append(f"native {k['native_cost']} {k['native_cost_unit'] or 'credits'} (not USD)")
    if cost or k["cost_basis"] not in (None, "UNAVAILABLE"):
        lines.append("cost: " + "  ".join(cost) + f"  basis {k['cost_basis']}")
    elif k["unresolved_spend"]:
        lines.append("cost: UNRESOLVED (the response gave neither a cost nor tokens); reconcile on the ledger")
    lines.append(f"ground truth: {g['status']}" + (f"  label {g['label']}  correct {g['correct']}" if g["status"] == "APPROVED" else ""))
    ver = [f"prompt {v['prompt_fingerprint']}" if v["prompt_fingerprint"] else None, f"data prep {v['dataset_version']}" if v["dataset_version"] else None,
           f"contract {v['input_contract_version']}" if v["input_contract_version"] else None]
    lines.append("versions: " + "  ".join(x for x in ver if x))
    route = [f"pinned {i['pinned_route']}" if i["pinned_route"] else None, f"resolved {i['resolved_model_id']}" if i["resolved_model_id"] else None,
             f"served by {i['serving_provider']}" if i["serving_provider"] else None]
    if any(route):
        lines.append("route: " + "  ".join(x for x in route if x))
    red = m.get("redaction") or {}
    lines.append(f"privacy: free text redacted by default ({red.get('replacements_in_free_text', 0)} replacement(s)); no raw student text in this record; pattern-based only")
    nm = _not_measured(r)
    if nm:
        lines.append("not measured: " + "; ".join(nm))
    return lines


def view_explorer(ctx: LiveContext, ex: ExplorerState, ts, coverage: List[Dict[str, Any]], labels_status: str, run_summaries: Optional[List[Dict[str, Any]]] = None) -> ViewModel:
    """Preview of the selected trace records plus run and participant coverage. Raw student text is never part of a row."""
    sel = ex.selection()
    f = lambda v: v if v else "ALL"  # noqa: E731
    n_missing = sum(1 for r in ts.rows if r["markers"]["unavailable_telemetry"])
    summary = [
        _line("Filters", f"model {f(ex.model)}  task {f(ex.task)}  run {_short_run(ex.run) if ex.run else 'ALL'}  status {f(ex.status)}  error class {f(ex.error_class)}  data {f(ex.kind)}"),
        _line("Export format", f"{ex.fmt.upper()}   keys: m model  t task  u run  s status  e error class  d data kind  f format  c clear  x export"),
        _line("Selected records", f"{len(ts.rows)}   skipped {sum(ts.skipped.values())}   redacted {ts.records_redacted}   with some measurement absent {n_missing}   scope {sel.scope}"),
        _line("Destination", str(ex.dest.resolve() if ex.dest else "")),
    ]
    warn = ["GROUND TRUTH PENDING (correctness columns stay empty)"] if labels_status == PENDING else []
    warn.append(PARITY)
    if any(r["markers"]["data_kind"] == "SYNTHETIC_FIXTURE" for r in ts.rows):
        warn.append("SYNTHETIC FIXTURE records are present: schema checks only, not production or real-data results")
    summary.append(Text("Warnings: " + "; ".join(warn), style=COLOR_WARNING))
    if ex.progress:
        summary.append(_line("Progress", ex.progress))
    if ex.last_error:
        summary.append(Text(f"EXPORT FAILED: {ex.last_error}", style=f"bold {COLOR_ERROR}"))
    elif ex.last_result is not None:
        r = ex.last_result
        summary.append(Text(f"EXPORT OK: {r.n_exported} records ({r.n_skipped} skipped, {r.n_redacted} redacted, {r.n_missing_telemetry} with an absent measurement) -> {r.path}", style=f"bold {COLOR_SUCCESS}"))
    for rs in (run_summaries or [])[:6]:
        planned = DASH if rs["planned_full_dataset"] is None else str(rs["planned_full_dataset"])
        un = DASH if rs["unattempted"] is None else str(rs["unattempted"])
        summary.append(Text(f"  run {_short_run(rs['run_id'], 40)}  [{KIND_SHORT.get(rs['data_kind'], rs['data_kind'])}]  planned {planned}  attempted {rs['attempted']}  "
                            f"succeeded {rs['succeeded']}  failed {rs['failed']}  invalid {rs['invalid']}  unattempted {un}", style=TEXT_SECONDARY))
    if run_summaries and len(run_summaries) > 6:
        summary.append(Text(f"  … {len(run_summaries) - 6} more runs", style=TEXT_MUTED))
    rows = []
    for r in ts.rows[:PREVIEW_LIMIT]:
        i, q, o, k, m, u = r["identity"], r["request"], r["output"], r["cost"], r["markers"], r["usage"]
        pred = DASH
        if q["status"] == "OK":
            pred = o["predicted_label"] or (DASH if o["decision"] is None else ("YES" if o["decision"] else "NO"))
        tok = DASH if u["prompt_tokens"] is None else f"{u['prompt_tokens']}/{u['completion_tokens']}"
        cost = f"${k['actual_cost_usd']} A" if k["actual_cost_usd"] else (f"${k['estimated_cost_usd']} E" if k["estimated_cost_usd"] else DASH)
        flags = ",".join(x for x, on in (("FAILED", m["failed_call"]), ("INVALID", m["invalid_output"]), ("NO-RECORD", m["source"] != "PREDICTION_RECORD"), ("UNRESOLVED$", k["unresolved_spend"])) if on)
        rows.append(Row([_short_run(i["run_id"], 20), i["case_id"], _model_name(ctx, i["model_key"]), i["task"] or DASH, q["status"] or DASH, pred, _num(q["latency_ms_end_to_end"]), tok, cost,
                         KIND_SHORT.get(m["data_kind"], m["data_kind"]) + (f" {flags}" if flags else "")], _row_detail(r)))
    notes = ["A dash means the field was not measured or does not apply; the export keeps the null and its reason. A = actual, E = estimated cost.", "Participant coverage (never-run models have no trace rows):"]
    for p in coverage:
        notes.append(f"  {p['display_name'] or p['key']:<34}{p['execution_state']:<32}records {p['n_trace_records']}  failed {p['n_failed_calls']}" + (f"  blockers: {', '.join(p['blockers'])}" if p["execution_state"].startswith("NOT_EXECUTED") and p["blockers"] else ""))
    if len(ts.rows) > PREVIEW_LIMIT:
        notes.insert(0, f"Preview shows the first {PREVIEW_LIMIT} of {len(ts.rows)} selected records; the export contains all of them.")
    return ViewModel("TRACE EXPLORER & EXPORT", "stored prediction records + request event log (read-only)", summary,
                     ["Run", "Case", "Model", "Task", "Status", "Prediction", "E2E ms", "Tokens in/out", "Cost", "Data / flags"], rows, notes,
                     empty="NO TRACE RECORDS MATCH THE SELECTION (an export would be empty; no placeholder rows are ever created)")


VIEWS = {"overview": view_overview, "quality": view_quality, "calibration": view_calibration, "agreement": view_agreement,
         "lead": view_lead, "latency": view_latency, "robustness": view_robustness, "dataset": view_dataset}


# ------------------------------------------------------------------ rendering
def render_view(vm: ViewModel, cursor: int, focused: bool, width: int) -> Panel:
    wide = width >= 150
    table = Table(expand=True, show_header=True, header_style=f"bold {TEXT_PRIMARY}", box=None, padding=(0, 1))
    for i, h in enumerate(vm.headers):
        longest = max([len(h)] + [len(r.cells[i]) for r in vm.rows if i < len(r.cells)])
        # Narrow columns keep their natural width; long values wrap inside their cell instead of being squeezed or cut.
        table.add_column(h, overflow="fold", min_width=min(longest, 14), ratio=max(1, min(longest, 36)))
    if not vm.rows:
        body: RenderableType = Text(vm.empty or UNAVAILABLE, style=f"bold {COLOR_WARNING}")
        inspector: RenderableType = Text("")
    else:
        cursor = max(0, min(cursor, len(vm.rows) - 1))
        for i, r in enumerate(vm.rows):
            sel = focused and i == cursor
            table.add_row(*r.cells, style=STYLE_ROW_SELECTED if sel else None)
        body = table
        inspector = Panel(Text("\n".join(vm.rows[cursor].detail), style=TEXT_PRIMARY, overflow="fold"), title="[b]INSPECTOR[/b]", title_align="left",
                          border_style=BORDER_CHARCOAL, padding=(0, 1))
    parts: List[RenderableType] = list(vm.summary) + [Text("")]
    if wide and vm.rows:
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=3)
        grid.add_column(ratio=2)
        grid.add_row(body, inspector)
        parts.append(grid)
    else:
        parts += [body, inspector] if vm.rows else [body]
    for n in vm.footnotes:
        parts.append(Text(n, style=TEXT_MUTED))
    return Panel(Group(*parts), title=f"[{TEXT_PRIMARY}]{vm.title}[/{TEXT_PRIMARY}]  [{TEXT_MUTED}]source: {vm.source}[/{TEXT_MUTED}]", title_align="left",
                 border_style=BORDER_FOCUSED if focused else BORDER_CHARCOAL, style=STYLE_PANEL, padding=(0, 1))


# ------------------------------------------------------------------ header
def build_execution_header(ctx: LiveContext, project: str = "msu-llm-lab") -> Table:
    a, b = ctx.agg, ctx.budget
    g = Table.grid(expand=True)
    g.add_column(ratio=1, no_wrap=False, overflow="fold")
    stamp = datetime.fromtimestamp(ctx.now).strftime("%a %d %b %Y  %H:%M:%S")
    mode = Text()
    mode.append("[MSU] LLM WORKBENCH ", style=f"bold {TEXT_PRIMARY}")
    mode.append(f"{project}  {stamp}   ", style=TEXT_SECONDARY)
    if a.active:
        mode.append("LIVE: REQUESTS IN FLIGHT", style=f"bold {COLOR_WARNING}")
    elif a.has_live_inference:
        mode.append("IDLE (live requests recorded)", style=f"bold {COLOR_SUCCESS}")
    else:
        mode.append(NO_LIVE, style=f"bold {TEXT_SECONDARY}")
    mode.append(f"   API requests {a.n_started} (active {a.active})", style=TEXT_SECONDARY)
    g.add_row(mode)
    sp = Text()
    if b is not None:
        by = b.spend_by_basis
        actual = by.get("PROVIDER_REPORTED", Decimal("0")) + by.get("RECONCILED", Decimal("0"))
        est = by.get("ESTIMATED_FROM_CATALOG", Decimal("0"))
        other = b.cumulative_spend_usd - actual - est
        sp.append(f"COUNTED vs CAPS ${b.cumulative_spend_usd:.4f} = ACTUAL ${actual:.4f} + ESTIMATED ${est:.4f}" + (f" + OTHER ${other:.4f}" if abs(other) > Decimal("0.00005") else "")
                  + f"   LEGACY-SIMULATED (not counted) ${b.simulated_spend_usd:.4f}   ", style=TEXT_SECONDARY)
        sp.append(f"UNRESOLVED {b.unresolved_spend_count}" + (" (live runs blocked)" if b.unresolved_spend_count else ""), style=f"bold {COLOR_ERROR}" if b.unresolved_spend_count else TEXT_SECONDARY)
        sp.append(f"   pilot left ${b.remaining_pilot_usd:.2f}  total left ${b.remaining_total_usd:.2f}", style=TEXT_SECONDARY)
    else:
        sp.append("Budget UNAVAILABLE", style=COLOR_WARNING)
    g.add_row(sp)
    ready = sum(r["readiness"] == "READY" for r in ctx.roster)
    smoke = sum(r["readiness"] == "SMOKE_READY" for r in ctx.roster)
    st = Text()
    st.append(f"Roster {ready} ready/{smoke} smoke/{len(ctx.roster) - ready - smoke} blocked   ", style=TEXT_SECONDARY)
    st.append(f"Dataset {ctx.dataset.version or 'NOT PREPARED'}   ", style=TEXT_SECONDARY)
    st.append(f"{ctx.analysis.labels.status}   ", style=COLOR_WARNING if not ctx.analysis.labels.available else COLOR_SUCCESS)
    st.append(PARITY, style=f"bold {COLOR_WARNING}")
    g.add_row(st)
    return g
