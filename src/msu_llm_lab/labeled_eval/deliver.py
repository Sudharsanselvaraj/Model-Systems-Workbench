"""Offline deliverables for the labelled evaluation: protocol, metric dictionary, blocked-run results table, plan costs and the report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from msu_llm_lab.labeled_eval import run as R

METRICS = [
    ("accuracy", "correct / n_cases_expected (missing or invalid counts as wrong)", "ratio", "both", "needs approved labels"),
    ("macro_f1", "mean per-class F1 over the 5 labels (Intent) or both classes (Lead), end-to-end", "ratio", "both", "unstable for classes with support < 10"),
    ("per_class_metrics_json", "precision/recall/F1/support/TP/FP/FN per class, end-to-end and conditional on valid outputs", "json", "intent", ""),
    ("cta_recall", "TP / (TP + FN); missing predictions count as FN", "ratio", "lead", "primary: missing a lead"),
    ("cta_precision", "TP / (TP + FP)", "ratio", "lead", ""),
    ("false_positive_rate", "FP / (FP + TN + missing negatives)", "ratio", "lead", ""),
    ("valid_output_rate", "valid / n_cases_expected", "ratio", "both", "format reliability, not correctness"),
    ("schema_compliance_rate", "valid / dispatched", "ratio", "both", ""),
    ("coverage_rate", "dispatched / n_cases_expected", "ratio", "both", ""),
    ("accuracy_ci95_*", "Wilson 95% interval", "ratio", "both", ""),
    ("macro_f1_ci95_*", "percentile bootstrap over cases, fixed seed", "ratio", "intent", "small n: wide"),
    ("selected_threshold", "max-F1 threshold chosen on DEV only, then frozen", "score", "lead", "only if a continuous score exists"),
    ("pr_auc / roc_auc", "from continuous yes-probability scores", "ratio", "lead", "needs scores"),
    ("brier_score / expected_calibration_error", "10 equal-width bins", "ratio", "lead", "only for genuine probabilities"),
    ("latency_*", "client end-to-end wall time", "ms", "both", "provider latency unavailable"),
    ("estimated_cost_*", "catalogue price x provider-reported tokens", "USD", "both", "not an invoice"),
    ("estimated_cost_per_correct_prediction_usd", "estimated cost / correct predictions", "USD", "both", "needs labels and valid predictions"),
]


def protocol_md(man: Dict[str, Any], blockers: List[Dict[str, str]]) -> str:
    d = man["datasets"]
    return f"""# Evaluation protocol ({R.VERSION}) - written before any labelled result exists

## Datasets (read-only)
- Labels: operator-attested as reviewed and correct (`label_attestation.json`); no second annotation, so reviewer agreement and kappa are unavailable. Data origin is unconfirmed (see `provenance_and_authorisation_scope.md`).
- Main Intent: `{d['intent']['path']}`, sha256 `{d['intent']['sha256'][:16]}`, {d['intent']['rows']} cases, labels {d['intent']['label_counts']}.
- Lead Qualifier: `{d['lead']['path']}`, sha256 `{d['lead']['sha256'][:16]}`, {d['lead']['rows']} cases, labels {d['lead']['label_counts']}.
- Alternative Lead file (198 rows) is NOT merged or substituted; see `lead_version_comparison` in the manifest.

## Tasks and contracts
- Main Intent: input = conversation history + current message; output = one of GREETING, OUT_OF_SCOPE, UPLOAD_STATUS_CHECK, HUMAN_HANDOFF_REQUEST, QUERY. Invalid output is a miss, never a class.
- Lead Qualifier: the repository contract and `lead_qualification_prompt.md` define the input as the action summary only. The files also carry `user_msg`. Which fields are sent is an OPEN decision (gate G3); the protocol is fixed to the repository contract (action summary only) unless the data owner decides otherwise, and a user_msg variant would be a separately named experiment.
- Expected labels and the answer key never enter inference inputs (`inference_inputs_*.jsonl` select allowed fields only; `answer_key.csv` is mode 0600 and read only by the scorer).

## Splits
- Intent: no parameter is tuned (prompts, models and thresholds are frozen), so all 120 cases are the evaluation split. Rare classes (UPLOAD_STATUS_CHECK 4, OUT_OF_SCOPE 4) cannot support a dev/test split; their metrics are flagged unstable. A stratified half is recorded as `exploratory_half` only.
- Lead: stratified dev/test (100/100, 53 positives each), families of identical normalised action summaries kept together, seed `{man['split']['seed']}`. Case ids per split: `split_manifest.csv`.

## Metrics and statistics
- Intent primary: macro-F1 (end-to-end, invalid counts as a miss); secondary accuracy, per-class P/R/F1, confusion matrix, weighted metrics, conditional-on-valid view.
- Lead primary: CTA recall and CTA F1; secondary precision, specificity, balanced accuracy, FPR/FNR.
- Intervals: Wilson 95% for proportions; percentile bootstrap (2,000, fixed seed) for macro-F1. Paired comparisons: McNemar exact on correctness over matched cases with Holm correction; effect sizes reported.
- Missing outputs: counted in the denominator (end-to-end); a conditional view over valid outputs is reported next to it.
- Jev threshold: only if Jev returns a continuous score for these cases; DEV only; criterion = maximum F1 (ties: lowest threshold); frozen; applied unchanged to TEST; also report the native decision rule and other operating points. No cost ratio is assumed; sensitivity scenarios are labelled hypothetical.
- Cost: estimated from catalogue prices and provider-reported tokens; native credits are never added to USD.
- Retries: the existing bounded retry policy (max 3 attempts); invalid outputs are not re-sent.
- Stopping: any budget gate, quarantine (3 consecutive errors) or approval refusal stops that model; no cap is raised.

## Live-run gates (all must pass; see live_run_gates.csv and gates_before_first_request.md)
""" + "\n".join(f"- **{b['id']} {b['blocker']}:** {b['detail']}" for b in blockers) + "\n"


def results_table(cfg: R.Cfg, man: Dict[str, Any]) -> pd.DataFrame:
    from msu_llm_lab.models import registry

    sm = pd.read_csv(cfg.bench_bundle / "model_summary.csv", dtype=str, keep_default_na=False)
    bm = json.loads((cfg.bench_bundle / "benchmark_manifest.json").read_text())
    pol = {p["key"]: p for p in bm["participants"]}
    pf = {k: v["version"] for k, v in bm["prompts"].items()}
    rows = []
    for r in sm.itertuples():
        p = pol.get(r.model_key, {})
        for split, n in ((("test", 120),) if r.task == "INTENT" else (("dev", 100), ("test", 100))):
            row = {c: "" for c in R.RESULT_COLUMNS}
            row.update({"evaluation_run_id": "NOT_EXECUTED", "dataset_name": "intent_test_cases" if r.task == "INTENT" else "lead_qualification_test_cases",
                        "dataset_version": R.VERSION, "dataset_sha256": man["datasets"]["intent" if r.task == "INTENT" else "lead"]["sha256"], "task": r.task, "split": split, "model_id": r.model_id,
                        "provider": r.serving_providers, "provider_route": p.get("route", {}).get("pinned_tag", ""), "prompt_version": pf.get(r.task, ""), "prompt_fingerprint": pf.get(r.task, ""),
                        "evaluation_status": ("EXCLUDED_PENDING_BUDGET_DECISION" if r.model_key in ("gpt-4.1", "claude-sonnet-5.5") else "NOT_EXECUTED_BLOCKED"), "comparability_status": "NOT_APPLICABLE_UNTIL_ALL_MODELS_RUN_ON_MATCHED_CASES", "n_cases_expected": n,
                        "n_requests_dispatched": 0, "n_unattempted": n, "coverage_rate": 0.0, "privacy_status": "UNKNOWN_DATA_POLICY; dataset origin unconfirmed (treat as confidential)",
                        "production_parity_status": "PRODUCTION_PARITY_NOT_ESTABLISHED", "known_limitations": "no live inference; labels not approved; see classifier_evaluation_limitations.md",
                        "notes": "All metric cells are intentionally empty: no predictions exist for these labelled cases. Empty means not measured, never zero."})
            if r.task == "LEAD" and r.model_key == "jev-1.14":
                row["threshold_selection_method"] = "max F1 on dev (planned); not run"
            rows.append(row)
    return pd.DataFrame(rows, columns=R.RESULT_COLUMNS)


def deliver(cfg: R.Cfg) -> Dict[str, Any]:
    prep = R.prepare(cfg)
    out, man = prep["out"], prep["manifest"]
    bl = R.blockers(cfg, man)
    (out / "evaluation_protocol.md").write_text(protocol_md(man, bl))
    pd.DataFrame(METRICS, columns=["metric", "formula_or_definition", "unit", "task", "caveat"]).to_csv(out / "classifier_metric_definitions.csv", index=False)
    costs = R.plan_costs(cfg)
    costs.to_csv(out / "classifier_cost_breakdown.csv", index=False)
    res = results_table(cfg, man)
    res.to_csv(out / "classifier_evaluation_results.csv", index=False)
    for name, cols in (("classifier_predictions.csv", ["case_id", "task", "model_key", "status", "predicted"]), ("classifier_error_analysis.csv", ["case_id", "task", "model_key", "error_type"]),
                       ("classifier_batch_metrics.csv", ["task", "batch", "model_key", "n"]), ("classifier_threshold_sweep.csv", ["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "specificity"]),
                       ("classifier_threshold_test_results.csv", ["threshold", "split", "tp", "fp", "fn", "tn"]), ("classifier_pairwise_comparisons.csv", ["task", "model_a", "model_b", "n_pairs", "p_holm"])):
        pd.DataFrame(columns=cols).to_csv(out / name, index=False)           # header-only: no predictions exist, nothing is fabricated
    tot_exp, tot_worst = costs.expected_usd.sum(), costs.worst_case_usd.sum()
    try:
        from msu_llm_lab.core.budget import BudgetService
        st = BudgetService().get_status()
        remaining = f"${st.remaining_pilot_usd} pilot / ${st.remaining_total_usd} total"
    except Exception as e:  # noqa: BLE001
        remaining = f"unavailable ({type(e).__name__})"
    per_model = costs.groupby("participant")[["expected_usd", "worst_case_usd"]].sum().round(3)
    over = costs[costs.worst_case_usd > 0.50].assign(expected_usd=lambda d: d.expected_usd.round(3), worst_case_usd=lambda d: d.worst_case_usd.round(3))
    (out / "classifier_evaluation_limitations.md").write_text("# Limitations and unavailable metrics\n\n" + "\n".join(f"- **{b['id']} {b['blocker']}:** {b['detail']}" for b in bl) +
        "\n\n## Unavailable until inference and approved labels exist\nAccuracy, macro-F1, per-class metrics, confusion matrices, CTA precision/recall/F1, AUCs, calibration, cost per correct prediction, paired tests, Jev threshold selection.\n\n"
        "## Unavailable regardless\nTime to first token, decode speed, provider-side latency, billed USD, repeatability (needs separately authorised repeated runs), language slices (no language field), reviewer agreement (no second annotation).\n")
    rep = f"""# Classifier evaluation report ({R.VERSION}): preparation stage

**Status: prepared, not executed.** No live inference was run, so there are no accuracy, F1, threshold or paired-comparison results. Live execution is blocked by one gate: verification that the applicable data-processing permission covers the selected external provider routes for this dataset (`gates_before_first_request.md`). The operator's decisions (Lead input, 16/24 caps, Mistral last, cohort) are recorded in `decisions.json`; the guarded executor exists and refuses to dispatch while that gate fails.

## 1. Datasets
Main Intent: {man['datasets']['intent']['rows']} cases, {man['datasets']['intent']['label_counts']}. Lead Qualifier: {man['datasets']['lead']['rows']} cases, {man['datasets']['lead']['label_counts']}. Hashes in `classifier_evaluation_manifest.json`.

## 2. Audit findings (details: `classifier_dataset_audit.csv`)
- No duplicate ids, no empty current messages, no invalid labels or boolean mismatches (the lead file uses lower-case true/false; the alternative uses True/False).
- Main Intent: 30 cases have no history (unknown, not evidence), 30 current messages repeat across cases with different history, 8 cases have suspect characters, UPLOAD_STATUS_CHECK and OUT_OF_SCOPE have only 4 cases each: per-class metrics for them will be unstable.
- Lead: 4 very short user messages. The 198-row alternative has the same 198 action summaries and the same labels but different user_msg for {198 - man['lead_version_comparison']['user_msg_identical']} of 198 rows; the 200-row file adds Test 199 and Test 200.
- Contamination: {man['contamination_check'].get('overlap_count')} test messages are identical (normalised, hash-compared) to current messages of the prepared benchmark candidates, which strongly suggests the cases are real, redacted conversation messages. They were not used to tune anything, but they cannot be treated as synthetic. 114 of 120 intent messages and 196 of 200 lead user_msg values match intent candidates, so the lead user_msg values are intent-style messages rather than lead-candidate messages.
- Label quality: the operator has attested in chat that the labels were reviewed and are correct and should be treated as the approved ground truth (`label_attestation.json`). No reviewer identities, second annotation or independent agreement exist, so reviewer agreement and Cohen\'s kappa are UNAVAILABLE. This is a single-operator attestation, not an independent two-reviewer process.

## 3. Design
See `evaluation_protocol.md` (fixed before any result). Splits: `split_manifest.csv`. Leakage control: inference inputs omit labels; the answer key is separate (0600) and read only by the scorer.

## 4. Budget
See `budget_verification_and_cohort_cost.md` (guards verified unchanged; eight-participant cohort expected cost, retry scenario, guard bound and absolute worst case) and `eight_model_cost_plan.csv`. The ten-participant reference in `classifier_cost_breakdown.csv` (expected about ${tot_exp:.2f}) is informational only; its per-run comparison was withdrawn as the wrong unit. Ledger headroom: {remaining}.

## 4b. Offline findings since the first stage
Lead input contract: `lead_input_contract_decision.md`. Claude and Mistral: `model_issue_investigation.md`. Provenance and the scope of the earlier authorisation: `provenance_and_authorisation_scope.md`. Separation checks: `separation_verification.json`. Remaining gates: `gates_before_first_request.md`.

## 5. What is in the primary CSV
`classifier_evaluation_results.csv` has one row per model x task x split ({len(res)} rows) with `evaluation_status = NOT_EXECUTED_BLOCKED` and empty metric cells. The scorer (`msu-llm-lab labeled-eval score`) fills them once predictions exist.
"""
    (out / "classifier_evaluation_report.md").write_text(rep)
    (out / "classifier_evaluation_summary.md").write_text(f"# Summary\n\nPrepared, not executed. 120 intent and 200 lead cases audited; splits and protocol frozen; no results because live inference is blocked by gate G1b (data-processing permission verification, see gates_before_first_request.md). Estimated spend for the full run about ${tot_exp:.2f} (worst case ${tot_worst:.2f}); remaining ledger {remaining}.\n")
    from msu_llm_lab.labeled_eval import stage2
    s2 = stage2.write_all(cfg)
    from msu_llm_lab.core.budget import BudgetService
    from msu_llm_lab.labeled_eval import execute as X
    X.record_decisions(cfg, man)
    bsvc = BudgetService()
    plan = X.build_plan(cfg, bsvc)
    if (out / "operator_approval.json").exists() and json.loads((out / "operator_approval.json").read_text()).get("plan_fingerprint") == plan["fingerprint"]:
        X.record_operator_approval(cfg, plan)          # keep the approval current for the same fingerprint
    try:
        X.save_plan(cfg, plan)
    except ValueError as e:                                   # a different plan is already stored: keep it, say so
        (out / "plan_conflict.txt").write_text(str(e))
    gates = X.check_gates(cfg, plan, bsvc)
    X.write_gate_report(cfg, plan, gates, s2["bp"])
    return {"plan": plan, "gates": gates, "cohort_totals": s2["bp"]["totals"], "cohort_verdict": s2["bp"]["verdict"], "out": out, "blockers": bl, "costs": costs, "per_model": per_model, "total_expected": tot_exp, "total_worst": tot_worst, "remaining": remaining, "manifest": man}


THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]
DEFAULT_REFERENCE_THRESHOLD = 0.5


def _score_matrix(g: pd.DataFrame, key: pd.DataFrame, split_of: Dict[str, str], split: str):
    k = key[(key.task == "LEAD") & key.case_id.map(lambda c: split_of.get(c) == split)]
    m = k.merge(g, on="case_id", how="inner")
    m = m[m.status.eq("OK") & pd.to_numeric(m.yes_prob, errors="coerce").notna()]
    return [e == "true" for e in m.expected], pd.to_numeric(m.yes_prob).tolist()


def jev_threshold(g: pd.DataFrame, key: pd.DataFrame, split_of: Dict[str, str], model: str) -> Dict[str, Any]:
    """Threshold from DEV scores only (max F1, ties to the lowest threshold), frozen, then applied unchanged to TEST. TEST labels are never read before the threshold is fixed."""
    y_dev, s_dev = _score_matrix(g, key, split_of, "dev")
    if len(y_dev) < 10 or all(y_dev) or not any(y_dev):
        return {"supported": False, "reason": "too few or single-class dev scores"}
    sweep = R.S.threshold_sweep(y_dev, s_dev, THRESHOLD_GRID)
    t = R.S.choose_threshold_max_f1(sweep)
    out = {"supported": True, "threshold": t, "n_dev": len(y_dev), "dev_positives": int(sum(y_dev)), "sweep": [dict(r, model_id=model, split="dev") for r in sweep]}
    y_te, s_te = _score_matrix(g, key, split_of, "test")             # first read of TEST labels, after the threshold is frozen
    out["n_test"], out["test_positives"] = len(y_te), int(sum(y_te))
    out["test_frozen"] = R.S.threshold_sweep(y_te, s_te, [t])[0]
    out["test_default"] = R.S.threshold_sweep(y_te, s_te, [DEFAULT_REFERENCE_THRESHOLD])[0]
    out["test_sweep_for_reference_only"] = R.S.threshold_sweep(y_te, s_te, THRESHOLD_GRID)
    return out


def score_all(cfg: R.Cfg, prediction_files: List[Path], out_csv: Path) -> pd.DataFrame:
    """Join stored prediction records to the protected answer key and fill the results table. Evaluator only: run after inference.
    Probability-only Lead participants (Jev) get a DEV-selected, frozen threshold; every other participant is scored on its own decisions."""
    key = pd.read_csv(cfg.private / R.VERSION / "answer_key.csv", dtype=str)
    split = pd.read_csv(cfg.out / R.VERSION / "split_manifest.csv", dtype=str)
    split_of = {t: dict(zip(g.case_id, g.split)) for t, g in split.groupby("task")}
    recs = []
    for f in prediction_files:
        recs += [json.loads(ln) for ln in open(f) if ln.strip()]
    df = pd.DataFrame(recs)
    if df.empty:
        raise ValueError("no prediction records")
    if "dataset_version" in df:
        df = df[df.dataset_version.eq(R.VERSION)]
    rows, sweeps, tests, reports = [], [], [], []
    for (mk, task), g in df.groupby(["model_key", "task"]):
        meta = {"model_id": mk, "dataset_version": R.VERSION, "evaluation_status": "EXECUTED"}
        prob_only = task == "LEAD" and "yes_prob" in g and pd.to_numeric(g.yes_prob, errors="coerce").notna().any() and g.predicted_need_cta.isna().all()
        thr = jev_threshold(g, key, split_of["LEAD"], mk) if prob_only else None
        for sp in (("test",) if task == "INTENT" else ("dev", "test")):
            if prob_only and not thr["supported"]:
                row = {c: "" for c in R.RESULT_COLUMNS}
                row.update(meta, task=task, split=sp, evaluation_status="THRESHOLD_NOT_SUPPORTED", notes=f"probability-only output; {thr['reason']}; no threshold was invented")
                rows.append(row)
                continue
            row = R.score_model_task(g, key, task, split_of[task], sp, threshold=(thr["threshold"] if prob_only else None), meta=meta)
            if prob_only:
                row.update({"selected_threshold": thr["threshold"], "threshold_selection_method": "max F1 on DEV (ties: lowest threshold), grid 0.05-0.95", "threshold_frozen": True, "threshold_test_evaluated": sp == "test"})
            rows.append(row)
        if prob_only and thr["supported"]:
            sweeps += thr["sweep"]
            tests += [dict(model_id=mk, split="test", rule="frozen DEV threshold", **thr["test_frozen"]), dict(model_id=mk, split="test", rule=f"default reference {DEFAULT_REFERENCE_THRESHOLD} (not an approved rule)", **thr["test_default"])]
            reports.append(f"- {mk}: DEV n={thr['n_dev']} ({thr['dev_positives']} positive) -> frozen threshold {thr['threshold']}; TEST n={thr['n_test']} ({thr['test_positives']} positive): "
                           f"F1 {thr['test_frozen']['f1']:.3f}, recall {thr['test_frozen']['recall']:.3f}, precision {thr['test_frozen']['precision']:.3f} at the frozen threshold; at the {DEFAULT_REFERENCE_THRESHOLD} reference F1 {thr['test_default']['f1']:.3f}.")
        elif prob_only:
            reports.append(f"- {mk}: THRESHOLD OPTIMISATION NOT SUPPORTED BY AVAILABLE OUTPUT ({thr['reason']}).")
    out = pd.DataFrame(rows, columns=R.RESULT_COLUMNS)
    out.to_csv(out_csv, index=False)
    d = Path(out_csv).parent
    if sweeps:
        pd.DataFrame(sweeps).to_csv(d / "classifier_threshold_sweep.csv", index=False)
        pd.DataFrame(tests).to_csv(d / "classifier_threshold_test_results.csv", index=False)
    _side_by_side(out, d)
    _prediction_tables(df, key, split_of, d)
    (d / "jev_threshold_report.md").write_text("# Jev threshold selection\n\nMethod: maximise F1 on the DEV split only (ties to the lowest threshold), freeze it, apply it unchanged to TEST. The default 0.5 row is a reference, not an approved rule. "
                                               "Small samples: preliminary.\n\n" + ("\n".join(reports) if reports else "No probability-only Lead participant had scores.") + "\n")
    return out


INTENT_COLS = ["model_id", "n_cases_expected", "n_valid_outputs", "valid_output_rate", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1", "macro_f1_ci95_low", "macro_f1_ci95_high",
               "macro_precision", "macro_recall", "latency_p50_ms", "latency_p95_ms", "estimated_cost_per_1000_requests_usd", "estimated_cost_per_correct_prediction_usd"]
LEAD_COLS = ["model_id", "split", "n_cases_expected", "n_valid_outputs", "valid_output_rate", "accuracy", "cta_precision", "cta_recall", "cta_f1", "specificity", "balanced_accuracy", "false_positive_rate",
             "false_negative_rate", "roc_auc", "pr_auc", "selected_threshold", "latency_p50_ms", "latency_p95_ms", "estimated_cost_per_1000_requests_usd", "estimated_cost_per_correct_prediction_usd"]


def _side_by_side(res: pd.DataFrame, d: Path) -> None:
    """Tasks stay separate (own files); the single summary file stacks them with a task column and never mixes their metrics."""
    i = res[res.task == "INTENT"][INTENT_COLS].sort_values("model_id")
    l = res[res.task == "LEAD"][LEAD_COLS].sort_values(["split", "model_id"])
    i.to_csv(d / "classifier_comparison_intent.csv", index=False)
    l.to_csv(d / "classifier_comparison_lead.csv", index=False)
    pd.concat([i.assign(task="INTENT", split="test"), l.assign(task="LEAD")], ignore_index=True).to_csv(d / "classifier_summary_side_by_side.csv", index=False)


def _prediction_tables(df: pd.DataFrame, key: pd.DataFrame, split_of: Dict[str, Dict[str, str]], d: Path) -> None:
    """Evaluator-side tables only: opaque case ids, no case text and no raw model output. Expected labels appear here because this runs after inference."""
    m = df.merge(key.rename(columns={"expected": "expected"}), on=["task", "case_id"], how="left")
    m["predicted"] = m.predicted_label.where(m.task.eq("INTENT"), m.predicted_need_cta.map(lambda v: None if v != v or v is None else str(bool(v)).lower()))
    m.loc[(m.task == "LEAD") & m.predicted.isna() & m.yes_prob.notna(), "predicted"] = "score_only:" + m.yes_prob.round(4).astype(str)
    m["split"] = [split_of.get(t, {}).get(c, "") for t, c in zip(m.task, m.case_id)]
    m["correct"] = [(p == e) if (s == "OK" and not str(p).startswith("score_only")) else "" for p, e, s in zip(m.predicted, m.expected, m.status)]
    cols = ["task", "split", "case_id", "model_key", "status", "predicted", "yes_prob", "expected", "correct", "validation_issues", "error_type", "finish_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens", "attempts", "latency_ms", "estimated_cost_usd"]
    m = m.assign(validation_issues=m.validation_issues.map(lambda v: "|".join(v) if isinstance(v, list) else v))
    m[cols].sort_values(["task", "model_key", "case_id"]).to_csv(d / "classifier_predictions.csv", index=False)
    bad = m[(m.status != "OK") | (m.correct == False)]  # noqa: E712
    bad.assign(reason=lambda x: x.apply(lambda r: "INVALID_OUTPUT:" + str(r.validation_issues) if r.status == "INVALID_OUTPUT" else ("ERROR:" + str(r.error_type) if r.status == "ERROR" else "WRONG"), axis=1))[cols + ["reason"]].to_csv(d / "classifier_error_analysis.csv", index=False)
