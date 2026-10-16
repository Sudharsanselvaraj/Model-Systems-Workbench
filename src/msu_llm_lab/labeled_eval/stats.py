"""Metrics and statistics for the labelled evaluation. Every function states its denominator; unavailable stays NaN, never zero."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from msu_llm_lab.observability.evaluation_report import boot_ci, holm, wilson

NAN = float("nan")


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else NAN
    r = tp / (tp + fn) if (tp + fn) else NAN
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else (NAN if not (tp + fp + fn) else 0.0)
    return p, r, f


def multiclass(y_true: Sequence[str], y_pred: Sequence[Optional[str]], labels: Sequence[str], *, n_expected: int) -> Dict:
    """End-to-end view: `y_pred` None marks a missing/invalid prediction, which counts as a miss for its true class (never dropped).
    Also returns the conditional view over valid predictions only."""
    out: Dict = {"n_expected": n_expected}
    yt, yp = list(y_true), list(y_pred)
    valid = [(t, p) for t, p in zip(yt, yp) if p is not None]
    out["n_valid"] = len(valid)
    cm = {t: {p: 0 for p in list(labels) + ["<INVALID>"]} for t in labels}
    for t, p in zip(yt, yp):
        cm[t][p if p is not None else "<INVALID>"] += 1
    per: Dict[str, Dict] = {}
    for view, pairs in (("end_to_end", list(zip(yt, [p if p is not None else "<INVALID>" for p in yp]))), ("conditional", valid)):
        rows = []
        for k in labels:
            tp = sum(1 for t, p in pairs if t == k and p == k)
            fp = sum(1 for t, p in pairs if t != k and p == k)
            fn = sum(1 for t, p in pairs if t == k and p != k)
            pr, rc, f1 = prf(tp, fp, fn)
            rows.append({"label": k, "support": sum(1 for t, _ in pairs if t == k), "tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": f1})
        per[view] = rows
        acc_n = len(pairs)
        correct = sum(1 for t, p in pairs if t == p)
        out[f"accuracy_{view}"] = correct / acc_n if acc_n else NAN
        lo, hi = wilson(correct, acc_n)
        out[f"accuracy_{view}_ci95"] = (lo, hi)
        sup = np.array([r["support"] for r in rows], float)
        for key in ("precision", "recall", "f1"):
            v = np.array([r[key] for r in rows], float)
            out[f"macro_{key}_{view}"] = float(np.nanmean(v)) if np.isfinite(v).any() else NAN
            m = np.isfinite(v) & (sup > 0)
            out[f"weighted_{key}_{view}"] = float(np.sum(v[m] * sup[m]) / sup[m].sum()) if m.any() else NAN
    out["per_class"] = per
    out["confusion_matrix"] = cm
    out["unstable_classes"] = [r["label"] for r in per["end_to_end"] if r["support"] < 10]
    return out


def macro_f1_ci(y_true, y_pred, labels, label: str, n_boot: int, seed: int) -> Tuple[float, float]:
    idx = np.arange(len(y_true))
    yt, yp = np.array(y_true, dtype=object), np.array([p if p is not None else "<INVALID>" for p in y_pred], dtype=object)

    def stat(i):
        i = i.astype(int)
        fs = []
        for k in labels:
            tp = np.sum((yt[i] == k) & (yp[i] == k)); fp = np.sum((yt[i] != k) & (yp[i] == k)); fn = np.sum((yt[i] == k) & (yp[i] != k))
            fs.append(prf(int(tp), int(fp), int(fn))[2])
        return np.nanmean(fs) if np.isfinite(fs).any() else NAN

    return boot_ci(idx.astype(float), stat, label, n_boot, seed)


def binary(y_true: Sequence[bool], y_pred: Sequence[Optional[bool]], *, n_expected: int) -> Dict:
    yt, yp = list(y_true), list(y_pred)
    tp = sum(1 for t, p in zip(yt, yp) if t and p is True)
    fn = sum(1 for t, p in zip(yt, yp) if t and p is not True)       # includes missing/invalid: counts as a miss
    fp = sum(1 for t, p in zip(yt, yp) if (not t) and p is True)
    tn = sum(1 for t, p in zip(yt, yp) if (not t) and p is False)
    miss_neg = sum(1 for t, p in zip(yt, yp) if (not t) and p is None)
    pr, rc, f1 = prf(tp, fp, fn)
    pos_pred_false = sum(1 for t, p in zip(yt, yp) if t and p is False)
    npr, nrc, nf1 = prf(tn, pos_pred_false, fp + miss_neg)
    spec = tn / (tn + fp + miss_neg) if (tn + fp + miss_neg) else NAN
    n_valid = sum(1 for p in yp if p is not None)
    correct = tp + tn
    lo, hi = wilson(correct, len(yt))
    return {"n_expected": n_expected, "n_valid": n_valid, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "missing_positive": sum(1 for t, p in zip(yt, yp) if t and p is None),
            "missing_negative": miss_neg, "accuracy": correct / len(yt) if yt else NAN, "accuracy_ci95": (lo, hi), "cta_precision": pr, "cta_recall": rc, "cta_f1": f1,
            "no_cta_precision": npr, "no_cta_recall": nrc, "no_cta_f1": nf1, "macro_f1": float(np.nanmean([f1, nf1])), "specificity": spec,
            "balanced_accuracy": float(np.nanmean([rc, spec])), "false_positive_rate": fp / (fp + tn + miss_neg) if (fp + tn + miss_neg) else NAN,
            "false_negative_rate": (fn / (tp + fn)) if (tp + fn) else NAN}


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict:
    from scipy import stats as sps

    a, b = np.array(a_correct, bool), np.array(b_correct, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0
    return {"n_pairs": len(a), "a_only_correct": n10, "b_only_correct": n01, "p_value": p, "effect_acc_diff": float(a.mean() - b.mean())}


def threshold_sweep(y_true: Sequence[bool], scores: Sequence[float], thresholds: Sequence[float]) -> List[Dict]:
    y, s = np.array(y_true, bool), np.array(scores, float)
    rows = []
    for t in thresholds:
        pred = s >= t
        tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
        pr, rc, f1 = prf(tp, fp, fn)
        rows.append({"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": pr, "recall": rc, "f1": f1, "specificity": tn / (tn + fp) if (tn + fp) else NAN,
                     "false_positive_rate": fp / (fp + tn) if (fp + tn) else NAN, "false_negative_rate": fn / (fn + tp) if (fn + tp) else NAN})
    return rows


def choose_threshold_max_f1(rows: List[Dict]) -> float:
    """Pre-declared rule: maximise F1 on DEV; ties go to the lowest threshold above the first maximum (deterministic)."""
    best = max(r["f1"] for r in rows if r["f1"] == r["f1"])
    return min(r["threshold"] for r in rows if r["f1"] == best)


def roc_pr_auc(y_true: Sequence[bool], scores: Sequence[float]) -> Tuple[float, float]:
    """ROC-AUC via the rank statistic (ties averaged) and average precision; NaN when only one class is present."""
    from scipy.stats import rankdata

    y, s = np.array(y_true, bool), np.array(scores, float)
    if y.all() or (~y).all():
        return NAN, NAN
    r = rankdata(s)
    n1, n0 = int(y.sum()), int((~y).sum())
    auc = float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    ap = float(np.sum(prec[ys]) / n1)
    return auc, ap


def brier(y_true: Sequence[bool], p: Sequence[float]) -> float:
    y, p = np.array(y_true, float), np.array(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else NAN


def ece(y_true: Sequence[bool], p: Sequence[float], bins: int = 10) -> Tuple[float, List[Dict]]:
    y, p = np.array(y_true, float), np.array(p, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, tot = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            gap = abs(y[m].mean() - p[m].mean())
            tot += m.sum() / len(y) * gap
            rows.append({"bin": f"[{edges[i]:.1f},{edges[i + 1]:.1f}{']' if i == bins - 1 else ')'}", "n": int(m.sum()), "mean_score": float(p[m].mean()), "positive_rate": float(y[m].mean())})
    return float(tot), rows
