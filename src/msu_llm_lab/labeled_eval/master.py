"""Model-comparison master tables for labeled-eval-v1: one CSV for Main Intent and one for Lead Qualifier, every metric recomputed from per-case predictions.

Evaluator side only: reads stored predictions, the protected answer key and local accounting; sends nothing. Missing data is a status string, never a guess.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from msu_llm_lab.labeled_eval import data as D
from msu_llm_lab.labeled_eval import deliver as DL
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.observability.evaluation_report import boot_ci, holm, rng_for, wilson

NA = "NOT_AVAILABLE"
NC = "NOT_COMPUTED"
INSUFFICIENT = "INSUFFICIENT_DATA"
INVALID = "<INVALID>"
SEED = 20261015
N_BOOT = 2000
P99_MIN_N = 100                     # a p99 needs at least 100 observations to rest on more than the single maximum
RELIABILITY_FLOOR = 0.90            # analyst assumption: below this valid-output rate a model is flagged not recommendable
REFERENCE = {"INTENT": "google/gemini-3.5-flash-lite", "LEAD": "openai/gpt-4.1-mini"}      # the production baseline of each task
WEIGHTS = {"INTENT": {"quality_macro_f1": 0.30, "quality_effective_accuracy": 0.20, "reliability": 0.15, "latency": 0.15, "cost": 0.20},
           "LEAD": {"quality_cta_f1": 0.25, "quality_cta_recall": 0.30, "reliability": 0.15, "latency": 0.10, "cost": 0.20}}
WEIGHT_NOTE = ("Analyst assumptions, not approved business weights. Intent: quality 50% (macro-F1 30, effective accuracy 20), reliability 15, latency 15, cost 20. "
               "Lead: quality 55% (CTA recall 30, CTA F1 25, because a missed lead is assumed costlier than a false alert), reliability 15, latency 10, cost 20.")


def f(x):
    return x if (isinstance(x, str) or (x == x and x is not None)) else ""


# ------------------------------------------------------------------ metrics (independent of stats.py)
def confusion(y_true: List[str], y_pred: List[str], classes: List[str]) -> np.ndarray:
    cols = classes + [INVALID]
    m = np.zeros((len(classes), len(cols)), dtype=int)
    ci = {c: i for i, c in enumerate(cols)}
    for t, p in zip(y_true, y_pred):
        m[classes.index(t), ci[p]] += 1
    return m


def per_class(m: np.ndarray, classes: List[str]) -> List[Dict[str, Any]]:
    out = []
    for i, c in enumerate(classes):
        tp = int(m[i, i])
        support = int(m[i].sum())
        fp = int(m[:, i].sum() - tp)
        fn = support - tp
        prec = tp / (tp + fp) if (tp + fp) else math.nan
        rec = tp / support if support else math.nan
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else math.nan
        out.append({"label": c, "support": support, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1})
    return out


def aggregate(m: np.ndarray, classes: List[str]) -> Dict[str, float]:
    pc = per_class(m, classes)
    n = int(m.sum())
    correct = int(sum(r["tp"] for r in pc))
    sup = np.array([r["support"] for r in pc], float)
    res: Dict[str, float] = {"n": n, "correct": correct, "accuracy": correct / n if n else math.nan}
    for k in ("precision", "recall", "f1"):
        v = np.array([r[k] for r in pc], float)
        res[f"macro_{k}"] = float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
        ok = np.isfinite(v) & (sup > 0)
        res[f"weighted_{k}"] = float(np.sum(v[ok] * sup[ok]) / sup[ok].sum()) if ok.any() else math.nan
    pred_total = m.sum(axis=0)                       # includes the invalid column as its own "predicted class"
    true_total = np.append(m.sum(axis=1), 0)
    s = n
    c = correct
    num = c * s - float(np.dot(pred_total, true_total))
    den = math.sqrt(max(s * s - float(np.sum(pred_total ** 2)), 0) * max(s * s - float(np.sum(true_total ** 2)), 0))
    res["mcc"] = num / den if den else math.nan
    pe = float(np.dot(pred_total, true_total)) / (s * s) if s else math.nan
    res["cohen_kappa"] = (res["accuracy"] - pe) / (1 - pe) if (s and pe == pe and pe < 1) else math.nan
    valid = int(m[:, :-1].sum())
    mp = correct / valid if valid else math.nan
    mr = correct / n if n else math.nan
    res["micro_precision"], res["micro_recall"] = mp, mr
    res["micro_f1"] = 2 * mp * mr / (mp + mr) if (mp == mp and mr == mr and (mp + mr) > 0) else math.nan
    rec = np.array([r["recall"] for r in pc if r["support"] > 0], float)
    res["balanced_accuracy"] = float(np.nanmean(rec)) if rec.size else math.nan
    return res


def pctl(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q)) if len(v) else math.nan


# ------------------------------------------------------------------ inputs
def load_inputs(cfg: R.Cfg) -> Dict[str, Any]:
    out = cfg.out / R.VERSION
    recs = []
    for f_ in sorted((out / "predictions").glob("*.jsonl")):
        recs += [json.loads(l) for l in open(f_) if l.strip()]
    df = pd.DataFrame(recs)
    df = df[df.dataset_version.eq(R.VERSION)].copy()
    key = pd.read_csv(cfg.private / R.VERSION / "answer_key.csv", dtype=str)
    split = pd.read_csv(out / "split_manifest.csv", dtype=str)
    split_of = {t: dict(zip(g.case_id, g.split)) for t, g in split.groupby("task")}
    return {"df": df, "key": key, "split_of": split_of, "man": json.loads((out / "classifier_evaluation_manifest.json").read_text()), "plan": json.loads((out / "plan.json").read_text())}


def decisions(df: pd.DataFrame, key: pd.DataFrame, split_of: Dict[str, Dict[str, str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """One row per (task, model, case) with the final label, `<INVALID>` for every unusable output, and the expected label from the answer key."""
    thr_info: Dict[str, Any] = {}
    rows = []
    for (task, mk), g in df.groupby(["task", "model_key"]):
        g = g.drop_duplicates("case_id")
        prob_only = task == "LEAD" and g.predicted_need_cta.isna().all() and pd.to_numeric(g.yes_prob, errors="coerce").notna().any()
        thr = None
        if prob_only:
            thr_info[mk] = DL.jev_threshold(g, key, split_of["LEAD"], mk)
            thr = thr_info[mk]["threshold"] if thr_info[mk]["supported"] else None
        for r in g.itertuples():
            if r.status != "OK":
                pred = INVALID
            elif task == "INTENT":
                pred = r.predicted_label if r.predicted_label in D.INTENT_LABELS else INVALID
            elif prob_only:
                pred = INVALID if (thr is None or r.yes_prob != r.yes_prob or r.yes_prob is None) else ("true" if r.yes_prob >= thr else "false")
            else:
                pred = INVALID if r.predicted_need_cta is None or r.predicted_need_cta != r.predicted_need_cta else ("true" if bool(r.predicted_need_cta) else "false")
            rows.append((task, mk, r.case_id, pred))
    d = pd.DataFrame(rows, columns=["task", "model_key", "case_id", "pred"])
    d = d.merge(key.rename(columns={"expected": "expected"}), on=["task", "case_id"], how="left")
    if d.expected.isna().any():
        raise ValueError("predictions for cases that are not in the answer key")
    d["split"] = [split_of[t][c] for t, c in zip(d.task, d.case_id)]
    return d, thr_info


# ------------------------------------------------------------------ one row per (task, model, split)
def build_rows(cfg: R.Cfg, cohort_note: str = "") -> Dict[str, Any]:
    I = load_inputs(cfg)
    df, key, split_of, man, plan = I["df"], I["key"], I["split_of"], I["man"], I["plan"]
    dec, thr_info = decisions(df, key, split_of)
    from msu_llm_lab.models import registry

    prompt_files = {"INTENT": Path("main_intent_classifier.md"), "LEAD": Path("lead_qualification_prompt.md")}
    prompt_hash = {t: hashlib.sha256(p.read_bytes()).hexdigest() for t, p in prompt_files.items()}
    ledger = {}
    con = sqlite3.connect(f"file:{Path('data/workbench.db').resolve()}?mode=ro", uri=True)
    for rid, usd in con.execute("select run_id, amount_usd from budget_ledger where run_id like 'labeled-eval-v1%'"):
        ledger[rid] = ledger.get(rid, 0.0) + float(usd)
    con.close()
    route = {p["model_id"]: p["route"] for p in plan["participants"]}
    key_of = {p["model_id"]: p["key"] for p in plan["participants"]}
    boot_rows: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    rows: Dict[str, List[Dict[str, Any]]] = {"INTENT": [], "LEAD": []}
    classes_of = {"INTENT": list(D.INTENT_LABELS), "LEAD": ["true", "false"]}
    cm_rows: List[Dict[str, Any]] = []
    correct_vec: Dict[Tuple[str, str, str], pd.Series] = {}
    for task in ("INTENT", "LEAD"):
        for sp in (["test"] if task == "INTENT" else ["dev", "test"]):
            for mk in sorted(dec[dec.task == task].model_key.unique()):
                g = dec[(dec.task == task) & (dec.model_key == mk) & (dec.split == sp)].sort_values("case_id")
                rec = df[(df.task == task) & (df.model_key == mk) & df.case_id.isin(g.case_id)].drop_duplicates("case_id").set_index("case_id").loc[g.case_id]
                classes = classes_of[task]
                m = confusion(list(g.expected), list(g.pred), classes)
                agg = aggregate(m, classes)
                pc = per_class(m, classes)
                vm = m[:, :-1]
                agg_v = aggregate(np.hstack([vm, np.zeros((len(classes), 1), dtype=int)]), classes)
                n = len(g)
                valid_n = int((g.pred != INVALID).sum())
                correct = int(agg["correct"])
                is_ok = rec.status.eq("OK")
                lat = pd.to_numeric(rec.latency_ms, errors="coerce").dropna().to_numpy()
                att = pd.to_numeric(rec.attempts, errors="coerce")
                retried = int((att > 1).sum())
                cost_est = pd.to_numeric(rec.estimated_cost_usd, errors="coerce")
                tin, tout = pd.to_numeric(rec.prompt_tokens, errors="coerce"), pd.to_numeric(rec.completion_tokens, errors="coerce")
                issues = rec.validation_issues.map(lambda v: v if isinstance(v, list) else [])
                empty = int(issues.map(lambda v: "EMPTY_OUTPUT" in v).sum())
                malformed = int(issues.map(lambda v: any(x in ("EXTRA_TEXT", "UNKNOWN_LABEL", "MALFORMED_JSON", "TYPE_MISMATCH", "MISSING_FIELD") for x in v)).sum())
                errors = int(rec.status.eq("ERROR").sum())
                completed = int(rec.status.isin(["OK", "INVALID_OUTPUT"]).sum())
                timeouts = int(rec.retry_errors.map(lambda v: sum(1 for c in (v or []) if c == "TIMEOUT")).sum())
                cp = rec.requested_route.map(lambda r: (r or {}).get("order", [None])[0] if isinstance(r, dict) else None)
                lo_a, hi_a = wilson(correct, n)
                rr = {"task": task, "split": sp, "model_id": mk, "model_key": key_of.get(mk, ""), "provider": rec.serving_provider.dropna().iloc[0] if rec.serving_provider.notna().any() else NA,
                      "provider_route": route.get(mk, NA), "dataset_name": "intent_test_cases" if task == "INTENT" else "lead_qualification_test_cases",
                      "dataset_version": R.VERSION, "dataset_hash": man["datasets"]["intent" if task == "INTENT" else "lead"]["sha256"], "sample_count": n,
                      "valid_prediction_count": valid_n, "correct_prediction_count": correct, "invalid_prediction_count": n - valid_n,
                      "accuracy": agg["accuracy"], "effective_accuracy": agg["accuracy"], "successful_case_accuracy": (correct / valid_n) if valid_n else NC,
                      "accuracy_95ci_lower": lo_a, "accuracy_95ci_upper": hi_a, "confidence_interval_method": "accuracy: Wilson 95% (effective, n=sample_count); F1: percentile bootstrap over cases, 2000 resamples, fixed seed",
                      "macro_precision": agg["macro_precision"], "macro_recall": agg["macro_recall"], "macro_f1": agg["macro_f1"], "weighted_precision": agg["weighted_precision"], "weighted_recall": agg["weighted_recall"],
                      "weighted_f1": agg["weighted_f1"], "micro_f1": agg["micro_f1"], "balanced_accuracy": agg["balanced_accuracy"], "matthews_correlation_coefficient": agg["mcc"], "cohen_kappa": agg["cohen_kappa"],
                      "macro_f1_valid_only": agg_v["macro_f1"], "accuracy_valid_only": agg_v["accuracy"],
                      "metric_denominator_note": "end-to-end metrics treat every invalid output as a miss for its true class and as an extra '<INVALID>' prediction; *_valid_only metrics use valid outputs only",
                      "attempted_requests": n, "completed_requests": completed, "valid_output_rate": valid_n / n, "invalid_output_rate": (n - valid_n) / n, "empty_response_rate": empty / n,
                      "request_failure_rate": errors / n, "retry_count": int((att - 1).clip(lower=0).sum()), "retried_request_count": retried, "retry_rate": retried / n, "timeout_count": timeouts,
                      "malformed_output_count": malformed, "http_success_but_invalid_count": int(rec.status.eq("INVALID_OUTPUT").sum()),
                      "mean_latency_ms": float(lat.mean()) if len(lat) else NA, "median_latency_ms": pctl(lat, 50), "p50_latency_ms": pctl(lat, 50), "p90_latency_ms": pctl(lat, 90), "p95_latency_ms": pctl(lat, 95),
                      "p99_latency_ms": pctl(lat, 99) if len(lat) >= P99_MIN_N else f"{INSUFFICIENT} (n<{P99_MIN_N})", "latency_standard_deviation_ms": float(lat.std(ddof=1)) if len(lat) > 1 else NA,
                      "latency_iqr_ms": pctl(lat, 75) - pctl(lat, 25) if len(lat) else NA, "retry_inclusive_latency_ms": float(lat.mean()) if len(lat) else NA,
                      "first_attempt_latency_ms": (float(pd.to_numeric(rec.latency_ms, errors="coerce")[att == 1].mean()) if (att == 1).any() else NA),
                      "first_attempt_latency_note": "mean over requests that needed no retry; first-attempt timing of retried requests was not stored",
                      "throughput_requests_per_second": NA, "latency_definition": "client end-to-end wall time per request, retries and backoff included; provider-internal latency not available",
                      "total_input_tokens": float(tin.sum()), "total_output_tokens": float(tout.sum()), "mean_input_tokens_per_case": float(tin.mean()), "mean_output_tokens_per_case": float(tout.mean()),
                      "total_estimated_cost_usd": float(cost_est.sum()), "provider_reported_cost_usd": NA if rec.actual_cost_usd.isna().all() else float(pd.to_numeric(rec.actual_cost_usd).sum()),
                      "native_provider_credits_total": float(pd.to_numeric(rec.native_cost, errors="coerce").sum()) if rec.native_cost.notna().any() else NA,
                      "native_credit_unit": (rec.native_cost_unit.dropna().iloc[0] if rec.native_cost_unit.notna().any() else NA),
                      "ledger_recorded_cost_usd_model_task_total": ledger.get(f"{R.VERSION}-{key_of.get(mk, '')}-{task}", NA),
                      "ledger_note": "ledger rows are per model-task run (dev and test together); do not add this column across splits",
                      "estimated_cost_per_attempt_usd": float(cost_est.sum()) / n, "estimated_cost_per_valid_result_usd": float(cost_est.sum()) / valid_n if valid_n else NC,
                      "estimated_cost_per_correct_result_usd": float(cost_est.sum()) / correct if correct else NC, "estimated_cost_per_1000_cases_usd": float(cost_est.sum()) / n * 1000,
                      "projected_monthly_cost_usd": f"{NC} (no workload assumption supplied)",
                      "run_id": f"{R.VERSION}-{key_of.get(mk, '')}-{task}", "evaluation_version": R.VERSION, "model_name": mk.split("/")[-1],
                      "prompt_version": rec.prompt_version.iloc[0], "prompt_hash": prompt_hash[task], "configuration_hash": plan["fingerprint"],
                      "model_parameters": "temperature 0.0 where the pinned route lists it (omitted otherwise; per-route detail not stored); fallbacks disabled; provider order pinned",
                      "output_token_limit": plan["output_caps"][task], "input_contract": ("conversation history + current message" if task == "INTENT" else "action_summary only (user_msg never sent)"),
                      "provider_policy_status": registry.get_record(key_of[mk]).data_policy.status.value if mk in key_of else NA,
                      "data_authorization_status": "operator attestation of processing permission (no document); typed live approval and typed real-data authorisation entered at run time; no authorisation manifest was persisted",
                      "production_parity_status": "PRODUCTION_PARITY_NOT_ESTABLISHED", "provenance_status": "CONFIDENTIAL_POTENTIALLY_REAL; origin unverified; labels are an operator attestation (no second annotation)",
                      "test_sample_support": f"n={n}; smallest class support {min(r['support'] for r in pc if r['support'] > 0)}" + ("; rare classes <10: per-class metrics unstable" if min(r['support'] for r in pc if r['support'] > 0) < 10 else "")}
                for r in pc:
                    c = r["label"]
                    rr.update({f"support_{c}": r["support"], f"precision_{c}": r["precision"], f"recall_{c}": r["recall"], f"f1_{c}": r["f1"], f"tp_{c}": r["tp"], f"fp_{c}": r["fp"], f"fn_{c}": r["fn"]})
                if task == "INTENT":
                    rr["false_positive_count_total"] = int(sum(r["fp"] for r in pc))
                    rr["false_negative_count_total"] = int(sum(r["fn"] for r in pc))
                    rr["false_count_definition"] = "sum over classes: FP = valid wrong predictions; FN = true cases not predicted correctly (includes invalid)"
                else:
                    t = pc[0]
                    neg_total = int(m[1].sum())
                    rr.update({"positive_class": "need_cta = true", "cta_precision": t["precision"], "cta_recall": t["recall"], "cta_f1": t["f1"], "no_cta_precision": pc[1]["precision"], "no_cta_recall": pc[1]["recall"],
                               "no_cta_f1": pc[1]["f1"], "specificity": pc[1]["tp"] / neg_total if neg_total else NC, "false_positive_count": int(m[1, 0]), "false_negative_count": int(m[0, 1] + m[0, 2]),
                               "false_positive_rate": m[1, 0] / neg_total if neg_total else NC, "false_negative_rate": (m[0, 1] + m[0, 2]) / int(m[0].sum()) if m[0].sum() else NC,
                               "true_positive_count": t["tp"], "true_negative_count": pc[1]["tp"], "invalid_on_positive_count": int(m[0, 2]), "invalid_on_negative_count": int(m[1, 2]),
                               "false_negative_definition": "positives not predicted true, invalid outputs included", "selected_decision_threshold": NA, "threshold_selection_split": NA})
                    if mk in thr_info and thr_info[mk].get("supported"):
                        ti = thr_info[mk]
                        yt = [e == "true" for e in g.expected]
                        sc = rec.yes_prob
                        rr.update({"selected_decision_threshold": ti["threshold"], "threshold_selection_split": "dev (max F1, grid 0.05-0.95, ties to lowest); frozen before test labels were read",
                                   "roc_auc": float(DL.R.S.roc_pr_auc(yt, sc.astype(float).tolist())[0]), "pr_auc": float(DL.R.S.roc_pr_auc(yt, sc.astype(float).tolist())[1]),
                                   "brier_score": DL.R.S.brier(yt, sc.astype(float).tolist()), "expected_calibration_error": DL.R.S.ece(yt, sc.astype(float).tolist())[0]})
                    else:
                        rr.update({"roc_auc": NA, "pr_auc": NA, "brier_score": NA, "expected_calibration_error": NA})
                # bootstrap CIs over cases (end-to-end)
                idx_true, idx_pred = np.array(g.expected), np.array(g.pred)
                g_rng = rng_for(f"master|{task}|{mk}|{sp}", SEED)
                draws = g_rng.integers(0, n, size=(N_BOOT, n))
                mf, c_f1 = [], []
                for ix in draws:
                    mm = confusion(list(idx_true[ix]), list(idx_pred[ix]), classes)
                    a = aggregate(mm, classes)
                    mf.append(a["macro_f1"])
                    c_f1.append(per_class(mm, classes)[0]["f1"])
                rr["macro_f1_95ci_lower"], rr["macro_f1_95ci_upper"] = float(np.nanpercentile(mf, 2.5)), float(np.nanpercentile(mf, 97.5))
                if task == "LEAD":
                    rr["cta_f1_95ci_lower"], rr["cta_f1_95ci_upper"] = float(np.nanpercentile(c_f1, 2.5)), float(np.nanpercentile(c_f1, 97.5))
                rows[task].append(rr)
                correct_vec[(task, sp, mk)] = pd.Series((g.expected.values == g.pred.values), index=g.case_id.values)
                for i, ec in enumerate(classes):
                    for j, pcl in enumerate(classes + [INVALID]):
                        cm_rows.append({"task": task, "split": sp, "model_id": mk, "true_label": ec, "predicted_label": pcl, "count": int(m[i, j])})
    return {"rows": rows, "cm": pd.DataFrame(cm_rows), "correct": correct_vec, "thr": thr_info, "I": I, "ledger": ledger}


# ------------------------------------------------------------------ statistics, ranks, composite, sensitivity
def pairwise(correct_vec: Dict[Tuple[str, str, str], pd.Series]) -> pd.DataFrame:
    from scipy import stats as sps

    out = []
    groups: Dict[Tuple[str, str], List[str]] = {}
    for (t, sp, mk) in correct_vec:
        groups.setdefault((t, sp), []).append(mk)
    for (t, sp), models in groups.items():
        block = []
        for a, b in combinations(sorted(models), 2):
            va, vb = correct_vec[(t, sp, a)], correct_vec[(t, sp, b)]
            ids = va.index.intersection(vb.index)
            x, y = va[ids].values.astype(bool), vb[ids].values.astype(bool)
            n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
            p = float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0
            block.append({"task": t, "split": sp, "model_a": a, "model_b": b, "n_paired_cases": len(ids), "a_only_correct": n10, "b_only_correct": n01, "accuracy_a": float(x.mean()), "accuracy_b": float(y.mean()),
                          "accuracy_difference_a_minus_b": float(x.mean() - y.mean()), "p_value": p, "method": "McNemar exact (invalid output = incorrect); same cases"})
        adj = holm([r["p_value"] for r in block])
        for r, q in zip(block, adj):
            r["adjusted_p_value_holm"] = q
            r["significant_at_0.05_after_holm"] = bool(q < 0.05)
        out += block
    return pd.DataFrame(out)


def attach_stats_and_ranks(rows: Dict[str, List[Dict[str, Any]]], pw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    res = {}
    for task, rs in rows.items():
        d = pd.DataFrame(rs)
        # paired comparison against the task's production baseline, Holm over the candidates in the same (task, split)
        d["comparison_reference_model"] = REFERENCE[task]
        d["paired_comparison_method"] = "McNemar exact vs the baseline on the same cases (invalid = incorrect); Holm across candidates"
        pv, adj, sig = [], [], []
        for sp, g in d.groupby("split"):
            ref = REFERENCE[task]
            cand = [m for m in g.model_id if m != ref]
            sub = pw[(pw.task == task) & (pw.split == sp)]
            ps = []
            for m in cand:
                r = sub[((sub.model_a == m) & (sub.model_b == ref)) | ((sub.model_a == ref) & (sub.model_b == m))].iloc[0]
                ps.append(r.p_value)
            ad = dict(zip(cand, holm(ps)))
            pm = dict(zip(cand, ps))
            for m in g.model_id:
                pv.append((m, sp, pm.get(m, math.nan)))
                adj.append((m, sp, ad.get(m, math.nan)))
        pmap = {(m, sp): p for m, sp, p in pv}
        amap = {(m, sp): p for m, sp, p in adj}
        d["paired_comparison_p_value"] = [pmap[(m, s)] if pmap[(m, s)] == pmap[(m, s)] else "REFERENCE_MODEL" for m, s in zip(d.model_id, d.split)]
        d["adjusted_p_value"] = [amap[(m, s)] if amap[(m, s)] == amap[(m, s)] else "REFERENCE_MODEL" for m, s in zip(d.model_id, d.split)]
        d["statistically_significant_difference"] = [(bool(amap[(m, s)] < 0.05) if amap[(m, s)] == amap[(m, s)] else "REFERENCE_MODEL") for m, s in zip(d.model_id, d.split)]
        # ranks within (task, split)
        for sp, idx in d.groupby("split").groups.items():
            g = d.loc[idx]
            qual = g.macro_f1 if task == "INTENT" else g.cta_f1
            d.loc[idx, "quality_rank"] = qual.rank(ascending=False, method="min")
            d.loc[idx, "reliability_rank"] = pd.concat([g.valid_output_rate, -g.request_failure_rate], axis=1).apply(tuple, axis=1).rank(ascending=False, method="min")
            d.loc[idx, "latency_rank"] = g.p50_latency_ms.rank(method="min")
            d.loc[idx, "cost_rank"] = g.estimated_cost_per_1000_cases_usd.rank(method="min")
            d.loc[idx, "cost_efficiency_rank_per_correct_result"] = pd.to_numeric(g.estimated_cost_per_correct_result_usd, errors="coerce").rank(method="min")
            comp = composite(g, task, WEIGHTS[task])
            d.loc[idx, "composite_score"] = comp
            d.loc[idx, "overall_rank"] = comp.rank(ascending=False, method="min")
        d["rank_scope"] = "ranks are within (task, split); ties share a rank"
        res[task] = d
    return res


def _norm(v: pd.Series, ref: pd.Series, invert: bool = False, log: bool = False) -> pd.Series:
    """Min-max over the eligible models only (valid-output rate at or above the floor), so a model that fails reliability cannot stretch the scale; values outside the range are clipped."""
    x = np.log10(v.astype(float)) if log else v.astype(float)
    r = np.log10(ref.astype(float)) if log else ref.astype(float)
    lo, hi = r.min(), r.max()
    s = ((x - lo) / (hi - lo)).clip(0, 1) if hi > lo else pd.Series(1.0, index=v.index)
    return 1 - s if invert else s


def components(g: pd.DataFrame, task: str) -> pd.DataFrame:
    e = g[g.valid_output_rate >= RELIABILITY_FLOOR]
    e = e if len(e) else g
    c = pd.DataFrame(index=g.index)
    if task == "INTENT":
        c["quality_macro_f1"] = _norm(g.macro_f1, e.macro_f1)
        c["quality_effective_accuracy"] = _norm(g.effective_accuracy, e.effective_accuracy)
    else:
        c["quality_cta_f1"] = _norm(g.cta_f1, e.cta_f1)
        c["quality_cta_recall"] = _norm(g.cta_recall, e.cta_recall)
    c["reliability"] = _norm(g.valid_output_rate, e.valid_output_rate)
    c["latency"] = _norm(g.p50_latency_ms, e.p50_latency_ms, invert=True, log=True)
    c["cost"] = _norm(g.estimated_cost_per_1000_cases_usd, e.estimated_cost_per_1000_cases_usd, invert=True, log=True)
    return c


def composite(g: pd.DataFrame, task: str, w: Dict[str, float]) -> pd.Series:
    c = components(g, task)
    tot = sum(w.values())
    return sum(c[k] * (v / tot) for k, v in w.items())


def sensitivity(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for task, d in tables.items():
        g = d[d.split == "test"].copy()
        base = WEIGHTS[task]
        keys = list(base)
        qk = [k for k in keys if k.startswith("quality")]
        scen = {"baseline weights": base, "equal weights": {k: 1.0 for k in keys}, "quality only": {k: (1.0 if k in qk else 0.0) for k in keys}}
        scen["quality-heavy (75%)"] = {k: (0.75 / len(qk) if k in qk else 0.25 / (len(keys) - len(qk))) for k in keys}
        scen["cost-heavy"] = {k: (0.40 if k == "cost" else 0.60 / (len(keys) - 1)) for k in keys}
        scen["latency-heavy"] = {k: (0.40 if k == "latency" else 0.60 / (len(keys) - 1)) for k in keys}
        scen["reliability-heavy"] = {k: (0.40 if k == "reliability" else 0.60 / (len(keys) - 1)) for k in keys}
        for name, w in scen.items():
            s = composite(g, task, w)
            top = g.loc[s.idxmax(), "model_id"]
            eligible = g[g.valid_output_rate >= RELIABILITY_FLOOR]
            s_e = s.loc[eligible.index]
            out.append({"task": task, "scenario": name, "winner_all_models": top, "winner_among_models_with_valid_rate_>=_0.90": g.loc[s_e.idxmax(), "model_id"] if len(s_e) else NA,
                        "winner_score": float(s.max()), "weights": json.dumps({k: round(v, 3) for k, v in w.items()})})
        rng = np.random.default_rng(SEED)
        wins: Dict[str, int] = {}
        eligible_idx = g.index[g.valid_output_rate >= RELIABILITY_FLOOR]
        for _ in range(2000):
            w = dict(zip(keys, rng.dirichlet(np.ones(len(keys)))))
            s = composite(g, task, w)
            m = g.loc[s.loc[eligible_idx].idxmax(), "model_id"]
            wins[m] = wins.get(m, 0) + 1
        for m, c in sorted(wins.items(), key=lambda kv: -kv[1]):
            out.append({"task": task, "scenario": "random weights (Dirichlet, 2000 draws), eligible models only", "winner_all_models": m, "winner_among_models_with_valid_rate_>=_0.90": m,
                        "winner_score": c / 2000, "weights": "winner_score = share of draws won"})
    return pd.DataFrame(out)


def recommendation(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for task, d in tables.items():
        rec, lim = [], []
        top = {sp: d[(d.split == sp) & (d.valid_output_rate >= RELIABILITY_FLOOR)].sort_values("composite_score", ascending=False).head(1).model_id.tolist() for sp in d.split.unique()}
        for r in d.itertuples():
            limits = []
            if r.valid_output_rate < RELIABILITY_FLOOR:
                limits.append(f"valid-output rate {r.valid_output_rate:.0%} is below the {RELIABILITY_FLOOR:.0%} floor")
            if r.request_failure_rate > 0:
                limits.append("request failures present")
            limits.append("sample is small; ranking is preliminary" if r.sample_count < 200 else "")
            if "unstable" in str(r.test_sample_support):
                limits.append("rare-class metrics unstable")
            lim.append("; ".join(x for x in limits if x))
            if r.valid_output_rate < RELIABILITY_FLOOR:
                rec.append("NOT_RECOMMENDED_RELIABILITY")
            elif top[r.split] and r.model_id == top[r.split][0] and r.split == "test":
                rec.append("BEST_OBSERVED_CANDIDATE (not a statistically established winner; not approved for production)")
            elif r.split == "dev":
                rec.append("DEV_SPLIT_FOR_THRESHOLD_SELECTION_ONLY")
            else:
                rec.append("CANDIDATE")
        d["recommendation"] = rec
        d["limitations"] = lim
    return tables


# ------------------------------------------------------------------ reconciliation, outputs, summary
def reconcile(cfg: R.Cfg, built: Dict[str, Any], tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    I = built["I"]
    df, key, split_of = I["df"], I["key"], I["split_of"]
    chk: List[Dict[str, Any]] = []

    def add(name, expected, observed, ok, note="", explained=False):
        chk.append({"check": name, "expected_or_reference": expected, "observed": observed, "status": "RECONCILED" if ok else ("EXPLAINED_DIFFERENT_QUANTITY" if explained else "DISCREPANCY"), "note": note})

    att = pd.to_numeric(df.attempts, errors="coerce")
    ev = pd.DataFrame([json.loads(l) for l in open("data/telemetry/request_events.jsonl") if l.strip()])
    ev = ev[ev.run_id.fillna("").str.startswith(R.VERSION)]
    n_retry_events = int((ev.kind == "REQUEST_RETRY").sum())
    add("requests that needed at least one retry", "14 (earlier report)", int((att > 1).sum()), int((att > 1).sum()) == 14, "counts requests, not attempts")
    add("extra HTTP attempts (retries) = sum(attempts-1)", "15 (earlier report)", int((att - 1).sum()), int((att - 1).sum()) == 15, "13 requests had 1 retry, 1 request had 2 retries (3 attempts)")
    add("retry events in the request-event log", "sum(attempts-1)", n_retry_events, n_retry_events == int((att - 1).sum()), "14 vs 15: 14 requests retried; 15 retry events because one request retried twice; no discrepancy")
    add("total HTTP attempts", "2560 requests + retries", int(att.sum()), int(att.sum()) == 2560 + int((att - 1).sum()))
    est = float(pd.to_numeric(df.estimated_cost_usd).sum())
    led = float(sum(built["ledger"].values()))
    st = json.loads(Path("data/labeled_eval/labeled-eval-v1/exec_state.json").read_text())
    guard = float(st["spent_guard_usd"])
    add("estimated cost (sum of per-request catalogue estimates) vs ledger rows for this run", est, led, abs(est - led) < 1e-6, "ledger books the same estimated amounts; 2560 rows")
    add("guard cost (executor stop-check quantity) vs estimated", est, guard, abs(est - guard) <= 1e-6,
        f"guard = max(booked, native credits) per request, never booked; differs by {guard - est:+.6f} USD. Planned expected cost was 1.08 and guard bound 2.23: all are estimates, none is an invoice", explained=True)
    nat = float(pd.to_numeric(df.native_cost, errors="coerce").sum())
    add("native provider credits (not USD)", NA, nat, True, "unit OPENROUTER_CREDITS_USD_RATE_UNVERIFIED; kept separate, never added to USD; provider-reported USD is not available")
    add("provider-reported USD present", NA, int(df.actual_cost_usd.notna().sum()), int(df.actual_cost_usd.notna().sum()) == 0, "0 rows: every USD figure here is ESTIMATED")
    # per-case counts
    for t, n_exp in (("INTENT", 120), ("LEAD", 200)):
        c = df[df.task == t].groupby("model_key").case_id.nunique()
        add(f"{t}: every model answered each case exactly once", f"{n_exp} per model, no duplicates", f"min {c.min()} max {c.max()}; duplicates {int(df[df.task == t].duplicated(['model_key', 'case_id']).sum())}", bool((c == n_exp).all()) and not df[df.task == t].duplicated(["model_key", "case_id"]).any())
    dev = {c for c, s in split_of["LEAD"].items() if s == "dev"}
    tst = {c for c, s in split_of["LEAD"].items() if s == "test"}
    add("Lead dev and test case sets are disjoint and complete", "100 + 100, disjoint", f"{len(dev)} + {len(tst)}, overlap {len(dev & tst)}", len(dev & tst) == 0 and len(dev) + len(tst) == 200)
    # threshold uses dev only
    flip = key.copy()
    m = flip.task.eq("LEAD") & flip.case_id.isin(tst)
    flip.loc[m, "expected"] = flip.loc[m, "expected"].map({"true": "false", "false": "true"})
    for mk, info in built["thr"].items():
        again = DL.jev_threshold(df[(df.task == "LEAD") & (df.model_key == mk)], flip, split_of["LEAD"], mk)
        add(f"{mk}: threshold unchanged when TEST labels are flipped", info.get("threshold"), again.get("threshold"), info.get("threshold") == again.get("threshold"), "threshold depends on dev labels only")
    # cross-check against the earlier scored file
    ref = pd.read_csv(cfg.out / R.VERSION / "classifier_evaluation_results.csv")
    worst = 0.0
    for t, d in tables.items():
        for r in d.itertuples():
            x = ref[(ref.model_id == r.model_id) & (ref.task == t) & (ref.split == r.split)]
            if x.empty:
                continue
            x = x.iloc[0]
            pairs = [(r.accuracy, x.accuracy), (r.macro_f1 if t == "INTENT" else r.cta_f1, x.macro_f1 if t == "INTENT" else x.cta_f1)]
            if t == "LEAD":
                pairs.append((r.cta_recall, x.cta_recall))
            worst = max([worst] + [abs(a - b) for a, b in pairs if a == a and b == b])
    add("independent recomputation vs the earlier scored file (accuracy, macro-F1 / CTA F1, CTA recall)", "difference 0", f"max abs difference {worst:.2e}", worst < 1e-9, "computed by separate code paths from the same per-case predictions and answer key")
    return pd.DataFrame(chk)


def lead_all200(built: Dict[str, Any]) -> pd.DataFrame:
    """All 200 Lead cases per model without a held-out split. Probability-only models get out-of-fold decisions (5-fold stratified, threshold chosen on the other four folds), so no case is scored by a threshold that saw it; a fixed-0.5 row is a reference only."""
    from msu_llm_lab.labeled_eval import stats as S

    I = built["I"]
    lead = I["df"][I["df"].task == "LEAD"]
    ans = I["key"][I["key"].task == "LEAD"].set_index("case_id").expected
    rows = []

    def met(y, p):
        y = np.array(y, bool)
        tp, fp = int(((p == "true") & y).sum()), int(((p == "true") & ~y).sum())
        fn, tn = int(y.sum() - tp), int(((p == "false") & ~y).sum())
        prec = tp / (tp + fp) if tp + fp else math.nan
        rec = tp / y.sum()
        return {"accuracy": (tp + tn) / len(y), "cta_precision": prec, "cta_recall": rec, "cta_f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "invalid_count": int((p == INVALID).sum())}

    for mk, g in lead.groupby("model_key"):
        g = g.drop_duplicates("case_id").set_index("case_id").sort_index()
        y = np.array([ans[c] == "true" for c in g.index])
        base = {"model_id": mk, "n_cases": len(g), "valid_output_rate": float((g.status == "OK").mean())}
        if g.predicted_need_cta.isna().all() and pd.to_numeric(g.yes_prob, errors="coerce").notna().any():
            sc = pd.to_numeric(g.yes_prob).values
            order = sorted(range(len(g)), key=lambda i: hashlib.sha256(g.index[i].encode()).hexdigest())
            folds = np.zeros(len(g), int)
            for lab in (True, False):
                for j, i in enumerate([i for i in order if y[i] == lab]):
                    folds[i] = j % 5
            pred = np.empty(len(g), object)
            ths = []
            for k in range(5):
                tr, te = folds != k, folds == k
                t = S.choose_threshold_max_f1(S.threshold_sweep(list(y[tr]), list(sc[tr]), DL.THRESHOLD_GRID))
                ths.append(t)
                pred[te] = np.where(sc[te] >= t, "true", "false")
            rows.append({**base, "basis": "all 200; threshold by 5-fold cross-validation (out-of-fold decisions)", **met(y, pred), "fold_thresholds": json.dumps(ths)})
            rows.append({**base, "basis": "all 200; fixed 0.5 (reference, not an approved rule)", **met(y, np.where(sc >= 0.5, "true", "false")), "fold_thresholds": ""})
        else:
            pred = np.where(g.status.eq("OK") & g.predicted_need_cta.notna(), np.where(g.predicted_need_cta.fillna(False).astype(bool), "true", "false"), INVALID)
            rows.append({**base, "basis": "all 200; the model's own decisions", **met(y, pred), "fold_thresholds": ""})
    return pd.DataFrame(rows).sort_values(["basis", "cta_f1"], ascending=[True, False])


def write_all(cfg: R.Cfg, monthly_cases: Optional[int] = None) -> Dict[str, Any]:
    out = cfg.out / R.VERSION
    built = build_rows(cfg)
    pw = pairwise(built["correct"])
    tables = recommendation(attach_stats_and_ranks(built["rows"], pw))
    if monthly_cases:
        for t, d in tables.items():
            d["projected_monthly_cost_usd"] = d.estimated_cost_per_attempt_usd * monthly_cases
            d["monthly_workload_assumption"] = f"{monthly_cases} cases per month (operator-supplied)"
    sens = sensitivity(tables)
    rec = reconcile(cfg, built, tables)
    files = {}
    for t, name in (("INTENT", "classifier_model_comparison_master_intent.csv"), ("LEAD", "classifier_model_comparison_master_lead.csv")):
        d = tables[t].sort_values(["split", "overall_rank", "model_id"])
        d.to_csv(out / name, index=False)
        files[t] = (out / name, d)
    lead_all200(built).to_csv(out / "classifier_model_comparison_lead_all200.csv", index=False)
    pw.to_csv(out / "classifier_model_comparison_pairwise.csv", index=False)
    built["cm"].to_csv(out / "classifier_model_comparison_confusion_long.csv", index=False)
    sens.to_csv(out / "classifier_model_comparison_sensitivity.csv", index=False)
    rec.to_csv(out / "classifier_model_comparison_reconciliation.csv", index=False)
    (out / "classifier_model_comparison_summary.md").write_text(summary_md(tables, sens, rec, built))
    return {"tables": tables, "sens": sens, "rec": rec, "pw": pw, "files": {k: str(v[0]) for k, v in files.items()}, "built": built}


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float, np.floating)) and x == x else str(x)


def summary_md(tables: Dict[str, pd.DataFrame], sens: pd.DataFrame, rec: pd.DataFrame, built: Dict[str, Any]) -> str:
    L = ["# Classifier model comparison: summary (labeled-eval-v1)", "",
         "Every number is recomputed from per-case predictions and the protected answer key; nothing was re-run and no data was sent to a provider. Main Intent and Lead Qualifier are compared separately.", "",
         "**Status of the conclusions:** the *best observed benchmark candidate* named below is not a statistically established winner (small samples, wide intervals) and is not approved for production (no provider policy verified, production parity not established, data origin unverified, labels attested by one operator).", ""]
    for t, title in (("INTENT", "Main Intent (120 cases, test split = all)"), ("LEAD", "Lead Qualifier (200 cases; test half = 100; Jev threshold chosen on the 100 dev cases)")):
        d = tables[t]
        g = d[d.split == "test"].sort_values("overall_rank")
        L += [f"## {title}", "", f"Weights (documented assumptions): {WEIGHT_NOTE}", ""]
        cols = ["model_id", "valid_output_rate", "accuracy", "macro_f1" if t == "INTENT" else "cta_f1"] + (["cta_recall", "cta_precision"] if t == "LEAD" else ["balanced_accuracy"]) + ["p50_latency_ms", "estimated_cost_per_1000_cases_usd", "composite_score", "overall_rank", "recommendation"]
        L += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in g.itertuples():
            L.append("| " + " | ".join(_fmt(getattr(r, c)) if c != "p50_latency_ms" else _fmt(getattr(r, c), 0) for c in cols) + " |")
        best = g[g.recommendation.str.startswith("BEST")]
        if len(best):
            b = best.iloc[0]
            L += ["", f"**Best observed candidate (baseline weights, models with valid-output rate >= {RELIABILITY_FLOOR:.0%}): {b.model_id}.**"]
        s = sens[(sens.task == t) & (sens.scenario != "random weights (Dirichlet, 2000 draws), eligible models only")]
        r_ = sens[(sens.task == t) & (sens.scenario.str.startswith("random"))]
        L += ["", "Sensitivity of the winner (eligible models): " + "; ".join(f"{a.scenario}: {a['winner_among_models_with_valid_rate_>=_0.90']}" for _, a in s.iterrows()) + ".",
              "Random weights: " + ", ".join(f"{a.winner_all_models} {a.winner_score:.0%}" for _, a in r_.iterrows()) + " of 2,000 draws.", ""]
        best_q = g.sort_values("quality_rank").iloc[0]
        L += [f"Highest quality alone ({'macro-F1' if t == 'INTENT' else 'CTA F1'}): {best_q.model_id} ({_fmt(best_q.macro_f1 if t == 'INTENT' else best_q.cta_f1)}). Fastest: {g.sort_values('latency_rank').iloc[0].model_id}. Cheapest per 1,000 cases: {g.sort_values('cost_rank').iloc[0].model_id}. Most reliable: {g.sort_values('reliability_rank').iloc[0].model_id}.", ""]
    L += ["## Methodology", "- End-to-end metrics count every invalid output as a miss and as an extra `<INVALID>` prediction; `*_valid_only` columns give the conditional view. Both are in the CSVs with their denominators.",
          "- Intervals: accuracy Wilson 95%; macro-F1 and CTA F1 percentile bootstrap over cases (2,000 resamples, fixed seed). Paired comparison: McNemar exact on the same cases, invalid = incorrect, versus the task's production baseline, Holm-adjusted; all pairs are in `classifier_model_comparison_pairwise.csv`.",
          "- Composite score: min-max normalisation within task and split over the models that meet the valid-output floor (values clipped; latency and cost on a log scale, inverted), weighted as above. The weights are analyst assumptions; the sensitivity table shows how the winner moves.",
          "- Lead: the positive class is `need_cta = true`; Jev's decision uses the frozen dev threshold; dev rows are informational and are not test results.", "",
          "## Reconciliation", "| check | reference | observed | status | note |", "|---|---|---|---|---|"]
    for r in rec.itertuples():
        L.append(f"| {r.check} | {r.expected_or_reference} | {r.observed} | {r.status} | {r.note} |")
    L += ["", "## Limitations and missing data",
          "- Rare Intent classes (UPLOAD_STATUS_CHECK, OUT_OF_SCOPE) have 4 cases each: their per-class metrics are unstable and drive macro-F1 differences.",
          "- Throughput, first-attempt latency of retried requests, provider-reported USD, provider-internal latency and a monthly cost projection are NOT AVAILABLE or NOT COMPUTED (no concurrency design, no stored first-attempt timing, no provider USD, no workload assumption).",
          "- Cost columns are estimates (catalogue price x provider-reported tokens). Native credits (unit unverified) are listed separately. Ledger spend equals the estimates for this run.",
          "- GPT-4.1 and Claude Sonnet 5.5 were outside the approved eight-model cohort and are not in these tables.",
          "- Claude Haiku's and Mistral's validity problems come from the fixed 16/24-token caps and output formatting (documented earlier); a different cap would be a separate pre-registered experiment.",
          "- The executor did not persist the typed real-data authorisation as a manifest; the permission record is an operator attestation without a document.", ""]
    return "\n".join(L) + "\n"
