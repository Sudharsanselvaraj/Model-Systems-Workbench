"""Prepare, plan and score the labelled evaluation. Offline only: nothing here sends a request or reads an API key."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from msu_llm_lab.labeled_eval import data as D
from msu_llm_lab.labeled_eval import stats as S

VERSION = "labeled-eval-v1"
UNAVAILABLE_REASON = {"cta": "task is Main Intent", "intent": "task is Lead Qualifier"}


class Cfg:
    def __init__(self, intent="Intent:lead dataset/intent_test_cases - intent_test_cases.csv.csv", lead="Intent:lead dataset/lead_qualification_test_cases (1).csv",
                 lead_alt=os.path.expanduser("~/Downloads/lead_qualification_test_cases.csv"), prep_dir="data/classification_prep/prep-v2",
                 out="reports/labeled_eval", private="data/labeled_eval", bench_bundle="reports/real_benchmark_bench-d0a763e4a920-r4",
                 prior_report="reports/evaluation_report_bench-d0a763e4a920"):
        self.intent, self.lead, self.lead_alt = Path(intent), Path(lead), Path(lead_alt)
        self.prep_dir, self.out, self.private = Path(prep_dir), Path(out), Path(private)
        self.bench_bundle, self.prior_report = Path(bench_bundle), Path(prior_report)
        self.policy_name = "2h"


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def inference_inputs(intent: pd.DataFrame, lead: pd.DataFrame, split: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Allowed inputs only. Label columns are not selected here, so they cannot reach a request payload."""
    i_rows = [{"case_id": r.case_id, "task": "INTENT", "split": split["intent"][r.case_id], "history_turns": [{"role": ro, "text": t} for ro, t in D.parse_history(r.history)],
               "current_message": r.current_message} for r in intent.itertuples()]
    l_rows = [{"case_id": r.case_id, "task": "LEAD", "split": split["lead"][r.case_id], "action_summary": r.action_summary} for r in lead.itertuples()]
    return {"INTENT": i_rows, "LEAD": l_rows}


def prepare(cfg: Cfg) -> Dict[str, Any]:
    out = cfg.out / VERSION
    out.mkdir(parents=True, exist_ok=True)
    intent, lead = D.load_intent(cfg.intent), D.load_lead(cfg.lead)
    lead_alt = D.load_lead(cfg.lead_alt) if cfg.lead_alt.exists() else None
    i_iss, i_stats = D.audit_intent(intent)
    l_iss, l_stats = D.audit_lead(lead)
    cmp_ = D.compare_lead_versions(lead, lead_alt) if lead_alt is not None else {"note": "alternative lead file not found"}
    cont = D.contamination({**{f"intent:{r.case_id}": r.current_message for r in intent.itertuples()}, **{f"lead:{r.case_id}": r.user_msg for r in lead.itertuples()}}, cfg.prep_dir)
    issues = pd.DataFrame(i_iss + l_iss)
    for cid in cont.get("overlapping_case_ids", []):
        issues = pd.concat([issues, pd.DataFrame([{"dataset": cid.split(":")[0], "case_id": cid.split(":")[1], "issue": "OVERLAP_WITH_BENCHMARK_CANDIDATE", "severity": "WARN",
                                                  "detail": "normalised current/user message equals a benchmark candidate message"}])])
    issues.to_csv(out / "classifier_dataset_audit.csv", index=False)
    # splits
    i_split = {cid: "test" for cid in intent.case_id}                       # Intent: no parameter is tuned, so the whole set is the evaluation split (see protocol)
    i_dev = D.stratified_group_split(intent, "expected_label", lambda r: D.norm(r.current_message), 0.5)
    l_split = D.stratified_group_split(lead, "expected_cta", lambda r: D.norm(r.action_summary), 0.5)
    manifest_rows = [{"task": "INTENT", "case_id": c, "split": s, "exploratory_half": i_dev[c]} for c, s in i_split.items()] + [{"task": "LEAD", "case_id": c, "split": s, "exploratory_half": ""} for c, s in l_split.items()]
    pd.DataFrame(manifest_rows).to_csv(out / "split_manifest.csv", index=False)
    inputs = inference_inputs(intent, lead, {"intent": i_split, "lead": l_split})
    for k, rows in inputs.items():
        _write_private(cfg.private / VERSION / f"inference_inputs_{k.lower()}.jsonl", "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    key = pd.concat([intent.assign(task="INTENT", expected=intent.expected_label)[["task", "case_id", "expected"]], lead.assign(task="LEAD", expected=lead.expected_cta.str.lower())[["task", "case_id", "expected"]]])
    _write_private(cfg.private / VERSION / "answer_key.csv", key.to_csv(index=False))
    l_lab = lead.assign(split=lead.case_id.map(l_split), y=lead.expected_cta.str.lower())
    man = {"schema": VERSION, "datasets": {"intent": {"path": str(cfg.intent), "sha256": D.sha256_file(cfg.intent), **i_stats}, "lead": {"path": str(cfg.lead), "sha256": D.sha256_file(cfg.lead), **l_stats},
                                           "lead_alternative": ({"path": str(cfg.lead_alt), "sha256": D.sha256_file(cfg.lead_alt)} if lead_alt is not None else None)},
           "lead_version_comparison": cmp_, "contamination_check": {k: v for k, v in cont.items() if k != "overlapping_case_ids"} | {"overlap_count": len(cont.get("overlapping_case_ids", []))},
           "split": {"seed": D.SPLIT_SEED, "intent": "all 120 cases form the evaluation split; a stratified half is recorded for exploratory use only", "lead_dev": int((l_lab.split == "dev").sum()),
                     "lead_test": int((l_lab.split == "test").sum()), "lead_dev_positive": int(((l_lab.split == "dev") & (l_lab.y == "true")).sum()), "lead_test_positive": int(((l_lab.split == "test") & (l_lab.y == "true")).sum())},
           "answer_key_location": str(cfg.private / VERSION / "answer_key.csv"), "answer_key_mode": "0600; never read by inference code"}
    (out / "classifier_evaluation_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True, default=str))
    return {"manifest": man, "issues": issues, "out": out}


def plan_costs(cfg: Cfg, n_intent: int = 120, n_lead: int = 200) -> pd.DataFrame:
    """Expected and worst-case spend per participant from the previous benchmark's observed cost per request and the registry prices. No request is made."""
    from msu_llm_lab.models import registry

    mts = pd.read_csv(cfg.prior_report / "model_task_summary.csv")
    mid = dict(zip(*pd.read_csv(cfg.bench_bundle / "model_summary.csv", dtype=str)[["model_key", "model_id"]].values.T))
    rows = []
    for r in mts.itertuples():
        n = n_intent if r.task == "INTENT" else n_lead
        p = registry.model_level_prices(mid.get(r.participant, ""))
        cap = 16 if r.task == "INTENT" else 24
        worst = (float(r.prompt_tokens_mean) * 1.5 * float(p[0]) + cap * float(p[1])) / 1e6 * n if p else float("nan")
        rows.append({"participant": r.participant, "task": r.task, "n_requests": n, "expected_usd": r.est_cost_usd_per_request * n, "worst_case_usd": worst,
                     "basis": "observed mean estimated USD per request in the earlier benchmark (different case text; indicative); worst case = 1.5x mean prompt tokens + output cap at registry prices"})
    return pd.DataFrame(rows)


def blockers(cfg: Cfg, man: Dict[str, Any]) -> List[Dict[str, str]]:
    """Remaining blockers after the operator's decisions (G3, G6, G7 decided; G8 executor built). `id` keys match live_run_gates.csv."""
    return [{"id": "G1b", "blocker": "Data-processing permission verification", "detail": "The operator approved this evaluation conditionally. A record (processing_permission.json, verified=true with verified_by and an evidence reference) must verify that the selected external provider routes are permitted for these dataset hashes and models. The earlier prep-v2 authorisation does not cover this dataset. No such record exists, so the executor refuses to dispatch."},
            {"id": "G2", "blocker": "Data origin unverified (accepted limitation, not a gate)", "detail": "Both datasets are classified CONFIDENTIAL (potentially real) because 114/120 intent messages and 196/200 lead user_msg values equal prep-v2 messages. They are not described as synthetic or anonymised. The origin must still be verified by the data owner."}]


# ----------------------------------------------------------------------------- scoring (post-inference, evaluator only)
RESULT_COLUMNS = ["evaluation_run_id", "evaluation_timestamp", "dataset_name", "dataset_version", "dataset_sha256", "task", "split", "model_id", "provider", "provider_route", "prompt_version",
    "prompt_fingerprint", "configuration_fingerprint", "evaluation_status", "comparability_status", "n_cases_expected", "n_requests_dispatched", "n_http_attempts", "n_retries", "n_completed",
    "n_valid_outputs", "n_invalid_outputs", "n_errors", "n_timeouts", "n_unattempted", "n_quarantined", "coverage_rate", "completion_rate", "valid_output_rate", "invalid_output_rate", "error_rate",
    "retry_rate", "schema_compliance_rate", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "macro_precision", "macro_recall", "macro_f1", "macro_f1_ci95_low", "macro_f1_ci95_high",
    "weighted_precision", "weighted_recall", "weighted_f1", "accuracy_conditional_valid", "macro_f1_conditional_valid", "per_class_metrics_json", "confusion_matrix_json", "rare_class_summary_json",
    "cta_precision", "cta_recall", "cta_f1", "no_cta_precision", "no_cta_recall", "no_cta_f1", "specificity", "balanced_accuracy", "false_positive_rate", "false_negative_rate", "pr_auc", "roc_auc",
    "selected_threshold", "threshold_selection_method", "threshold_frozen", "threshold_test_evaluated", "brier_score", "expected_calibration_error", "lead_confusion_matrix_json",
    "latency_mean_ms", "latency_p50_ms", "latency_p90_ms", "latency_p95_ms", "latency_p99_ms", "latency_min_ms", "latency_max_ms", "latency_std_ms", "latency_measurement_boundary",
    "provider_latency_available", "time_to_first_token_ms", "throughput_requests_per_second", "input_tokens_total", "output_tokens_total", "total_tokens", "input_tokens_mean", "output_tokens_mean",
    "token_metric_coverage", "provider_reported_cost_usd", "estimated_cost_usd", "ledger_recorded_cost_usd", "guard_cost_usd", "native_provider_credits", "native_credit_unit", "unresolved_cost_usd",
    "estimated_cost_per_request_usd", "estimated_cost_per_1000_requests_usd", "estimated_cost_per_1000_valid_outputs_usd", "estimated_cost_per_correct_prediction_usd", "pricing_source",
    "pricing_version", "cost_provenance", "confidence_interval_method", "statistical_comparison_group", "paired_comparison_available", "significance_adjustment", "privacy_status",
    "production_parity_status", "known_limitations", "notes"]


def _j(x) -> str:
    return json.dumps(x, sort_keys=True, default=lambda o: None if o != o else o)


def score_model_task(preds: pd.DataFrame, key: pd.DataFrame, task: str, split_of: Dict[str, str], split: str, *, threshold: Optional[float] = None,
                     n_boot: int = 2000, seed: int = 20261015, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One result row for one model x task x split. `preds` has one row per dispatched case (PredictionRecord fields); missing cases count as unattempted."""
    meta = meta or {}
    k = key[(key.task == task)]
    k = k[k.case_id.map(lambda c: split_of.get(c) == split)] if split != "all" else k
    if k.case_id.duplicated().any():
        raise ValueError("duplicate case ids in the answer key")
    if preds.case_id.duplicated().any():
        raise ValueError("duplicate predictions for one case: resolve before scoring")
    unknown = set(preds.case_id) - set(key[key.task == task].case_id)
    if unknown:
        raise ValueError(f"predictions for cases not in the answer key: {sorted(unknown)[:3]}")
    m = k.merge(preds, on="case_id", how="left")
    n = len(k)
    disp = m.status.notna()
    ok = m.status.eq("OK")
    row: Dict[str, Any] = {c: "" for c in RESULT_COLUMNS}
    row.update(meta)
    row.update({"task": task, "split": split, "n_cases_expected": n, "n_requests_dispatched": int(disp.sum()), "n_completed": int(disp.sum()), "n_valid_outputs": int(ok.sum()),
                "n_invalid_outputs": int(m.status.eq("INVALID_OUTPUT").sum()), "n_errors": int(m.status.eq("ERROR").sum()), "n_unattempted": int((~disp).sum()),
                "coverage_rate": disp.mean() if n else "", "completion_rate": disp.mean() if n else "", "valid_output_rate": ok.sum() / n if n else "",
                "invalid_output_rate": m.status.eq("INVALID_OUTPUT").sum() / n if n else "", "error_rate": m.status.eq("ERROR").sum() / n if n else "", "schema_compliance_rate": ok.sum() / int(disp.sum()) if disp.any() else ""})
    if "attempts" in m:
        a = pd.to_numeric(m.attempts, errors="coerce")
        row.update({"n_http_attempts": int(a.sum()), "n_retries": int((a.dropna() - 1).clip(lower=0).sum()), "retry_rate": float((a > 1).sum() / max(1, disp.sum()))})
    row["n_timeouts"] = int(m.get("error_category", pd.Series(dtype=str)).eq("TIMEOUT").sum()) if "error_category" in m else ""
    if task == "INTENT":
        res = S.multiclass(list(m.expected), [p if s == "OK" else None for p, s in zip(m.get("predicted_label"), m.status)], list(D.INTENT_LABELS), n_expected=n)
        lo, hi = res["accuracy_end_to_end_ci95"]
        row.update({"accuracy": res["accuracy_end_to_end"], "accuracy_ci95_low": lo, "accuracy_ci95_high": hi, "macro_precision": res["macro_precision_end_to_end"], "macro_recall": res["macro_recall_end_to_end"],
                    "macro_f1": res["macro_f1_end_to_end"], "weighted_precision": res["weighted_precision_end_to_end"], "weighted_recall": res["weighted_recall_end_to_end"], "weighted_f1": res["weighted_f1_end_to_end"],
                    "accuracy_conditional_valid": res["accuracy_conditional"], "macro_f1_conditional_valid": res["macro_f1_conditional"],
                    "per_class_metrics_json": _j(res["per_class"]), "confusion_matrix_json": _j(res["confusion_matrix"]),
                    "rare_class_summary_json": _j({"unstable_classes_support_below_10": res["unstable_classes"], "support": {r["label"]: r["support"] for r in res["per_class"]["end_to_end"]}})})
        blo, bhi = S.macro_f1_ci(list(m.expected), [p if s == "OK" else None for p, s in zip(m.get("predicted_label"), m.status)], list(D.INTENT_LABELS), f"{meta.get('model_id')}|{split}", n_boot, seed)
        row.update({"macro_f1_ci95_low": blo, "macro_f1_ci95_high": bhi, "confidence_interval_method": "accuracy: Wilson 95%; macro-F1: percentile bootstrap (cases resampled), seed fixed"})
    else:
        yt = [e == "true" for e in m.expected]
        yp = [(bool(p) if (s == "OK" and p == p and p is not None) else None) for p, s in zip(m.get("predicted_need_cta"), m.status)]
        if threshold is not None and "yes_prob" in m:
            sc = pd.to_numeric(m.yes_prob, errors="coerce")
            yp = [(bool(v >= threshold) if (s == "OK" and v == v) else None) for v, s in zip(sc, m.status)]
        res = S.binary(yt, yp, n_expected=n)
        lo, hi = res["accuracy_ci95"]
        row.update({k2: res[k2] for k2 in ("accuracy", "cta_precision", "cta_recall", "cta_f1", "no_cta_precision", "no_cta_recall", "no_cta_f1", "specificity", "balanced_accuracy", "false_positive_rate", "false_negative_rate", "macro_f1")})
        row.update({"accuracy_ci95_low": lo, "accuracy_ci95_high": hi, "lead_confusion_matrix_json": _j({k2: res[k2] for k2 in ("tp", "fp", "fn", "tn", "missing_positive", "missing_negative")}),
                    "confidence_interval_method": "accuracy: Wilson 95%"})
        if "yes_prob" in m and pd.to_numeric(m.yes_prob, errors="coerce").notna().all() and ok.all():
            sc = pd.to_numeric(m.yes_prob).tolist()
            roc, pr = S.roc_pr_auc(yt, sc)
            e, _ = S.ece(yt, sc)
            row.update({"roc_auc": roc, "pr_auc": pr, "brier_score": S.brier(yt, sc), "expected_calibration_error": e})
    lat = pd.to_numeric(m.get("latency_ms"), errors="coerce").dropna() if "latency_ms" in m else pd.Series(dtype=float)
    if len(lat):
        row.update({"latency_mean_ms": lat.mean(), "latency_p50_ms": lat.quantile(.5), "latency_p90_ms": lat.quantile(.9), "latency_p95_ms": lat.quantile(.95), "latency_p99_ms": lat.quantile(.99),
                    "latency_min_ms": lat.min(), "latency_max_ms": lat.max(), "latency_std_ms": lat.std(), "latency_measurement_boundary": "client end-to-end wall time (includes retries)", "provider_latency_available": False})
    pt, ct = (pd.to_numeric(m.get(c), errors="coerce") for c in ("prompt_tokens", "completion_tokens")) if "prompt_tokens" in m else (pd.Series(dtype=float), pd.Series(dtype=float))
    if len(pt) and pt.notna().any():
        row.update({"input_tokens_total": int(pt.sum()), "output_tokens_total": int(ct.sum()), "total_tokens": int(pt.sum() + ct.sum()), "input_tokens_mean": pt.mean(), "output_tokens_mean": ct.mean(), "token_metric_coverage": float(pt.notna().mean())})
    if "estimated_cost_usd" in m:
        est = pd.to_numeric(m.estimated_cost_usd, errors="coerce")
        tot = est.sum()
        row.update({"estimated_cost_usd": tot, "estimated_cost_per_request_usd": est.mean(), "estimated_cost_per_1000_requests_usd": est.mean() * 1000,
                    "estimated_cost_per_1000_valid_outputs_usd": tot / ok.sum() * 1000 if ok.sum() else "", "cost_provenance": "ESTIMATED (catalogue price x provider-reported tokens); not an invoice"})
        if task == "INTENT":
            corr = sum(1 for e, p, s in zip(m.expected, m.get("predicted_label"), m.status) if s == "OK" and e == p)
        else:
            corr = sum(1 for e, p, s in zip(m.expected, m.get("predicted_need_cta"), m.status) if s == "OK" and p == p and (e == "true") == bool(p))
        row["estimated_cost_per_correct_prediction_usd"] = tot / corr if corr else ""
    return row
