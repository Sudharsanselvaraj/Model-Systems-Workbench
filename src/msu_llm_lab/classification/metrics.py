"""Classification metrics computed from approved labels only. Never from dry-run or unreviewed data."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel

from msu_llm_lab.classification.contracts import IntentLabel

INVALID = "INVALID"


class MetricsError(ValueError):
    pass


class ConfidenceSource(str, Enum):
    NATIVE_DISTRIBUTION = "NATIVE_DISTRIBUTION"
    NATIVE_CONFIDENCE_FIELD = "NATIVE_CONFIDENCE_FIELD"
    TOKEN_LOGPROB = "TOKEN_LOGPROB"
    SELF_REPORTED_TEXT = "SELF_REPORTED_TEXT"
    NONE = "NONE"


_CALIBRATION_OK = {ConfidenceSource.NATIVE_DISTRIBUTION, ConfidenceSource.NATIVE_CONFIDENCE_FIELD, ConfidenceSource.TOKEN_LOGPROB}


@dataclass
class IntentItem:
    expected: Optional[IntentLabel]
    predicted: Optional[IntentLabel]  # None = invalid output
    exclusion_reason: Optional[str] = None
    is_dry_run: bool = False


@dataclass
class LeadItem:
    expected: Optional[bool]
    predicted: Optional[bool]  # None = invalid output
    exclusion_reason: Optional[str] = None
    is_dry_run: bool = False


class ClassStats(BaseModel):
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    support: int


class IntentMetrics(BaseModel):
    n_total: int
    n_excluded: int
    n_unlabeled: int
    n_scored: int
    invalid_count: int
    invalid_rate: Optional[float]
    accuracy: Optional[float]
    macro_f1: Optional[float]
    per_class: Dict[str, ClassStats]
    confusion: Dict[str, Dict[str, int]]


class LeadMetrics(BaseModel):
    n_total: int
    n_excluded: int
    n_unlabeled: int
    n_scored: int
    invalid_count: int
    invalid_rate: Optional[float]
    accuracy: Optional[float]
    yes_precision: Optional[float]
    yes_recall: Optional[float]
    yes_f1: Optional[float]
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]
    tp: int
    fp: int
    fn: int
    tn: int


def _div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def _f1(p: Optional[float], r: Optional[float], tp: int, fp: int, fn: int) -> Optional[float]:
    if p is None and r is None:
        return None
    if tp == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _scorable(items, kind: str):
    if any(i.is_dry_run for i in items):
        raise MetricsError("Dry-run/simulated records must never enter a metrics report")
    excluded = [i for i in items if i.exclusion_reason is not None]
    rest = [i for i in items if i.exclusion_reason is None]
    unlabeled = [i for i in rest if i.expected is None]
    scored = [i for i in rest if i.expected is not None]
    return excluded, unlabeled, scored


def intent_metrics(items: Sequence[IntentItem]) -> IntentMetrics:
    """Invalid outputs count as incorrect. Macro-F1 averages labels with a defined F1."""
    excluded, unlabeled, scored = _scorable(items, "intent")
    labels = [l.value for l in IntentLabel]
    confusion: Dict[str, Dict[str, int]] = {l: {c: 0 for c in labels + [INVALID]} for l in labels}
    for i in scored:
        confusion[i.expected.value][i.predicted.value if i.predicted else INVALID] += 1
    per_class: Dict[str, ClassStats] = {}
    f1s: List[float] = []
    for l in labels:
        tp = confusion[l][l]
        fn = sum(confusion[l].values()) - tp
        fp = sum(confusion[o][l] for o in labels if o != l)
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        f1 = _f1(p, r, tp, fp, fn)
        if f1 is not None:
            f1s.append(f1)
        per_class[l] = ClassStats(precision=p, recall=r, f1=f1, support=tp + fn)
    n = len(scored)
    correct = sum(1 for i in scored if i.predicted == i.expected)
    invalid = sum(1 for i in scored if i.predicted is None)
    return IntentMetrics(
        n_total=len(items), n_excluded=len(excluded), n_unlabeled=len(unlabeled), n_scored=n,
        invalid_count=invalid, invalid_rate=_div(invalid, n), accuracy=_div(correct, n),
        macro_f1=(sum(f1s) / len(f1s)) if f1s else None, per_class=per_class, confusion=confusion,
    )


def lead_metrics(items: Sequence[LeadItem]) -> LeadMetrics:
    """YES is the positive class. An invalid output on an expected-YES case is a miss; on expected-NO it is wrong but not a false positive."""
    excluded, unlabeled, scored = _scorable(items, "lead")
    tp = sum(1 for i in scored if i.expected is True and i.predicted is True)
    fp = sum(1 for i in scored if i.expected is False and i.predicted is True)
    tn = sum(1 for i in scored if i.expected is False and i.predicted is False)
    fn = sum(1 for i in scored if i.expected is True and i.predicted is not True)
    neg = sum(1 for i in scored if i.expected is False)
    n = len(scored)
    invalid = sum(1 for i in scored if i.predicted is None)
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return LeadMetrics(
        n_total=len(items), n_excluded=len(excluded), n_unlabeled=len(unlabeled), n_scored=n,
        invalid_count=invalid, invalid_rate=_div(invalid, n), accuracy=_div(tp + tn, n),
        yes_precision=p, yes_recall=r, yes_f1=_f1(p, r, tp, fp, fn),
        false_positive_rate=_div(fp, neg), false_negative_rate=_div(fn, tp + fn), tp=tp, fp=fp, fn=fn, tn=tn,
    )


def _require_calibratable(source: ConfidenceSource) -> None:
    if source not in _CALIBRATION_OK:
        raise MetricsError(f"confidence_source {source.value} cannot be used for calibration metrics")


def brier_multiclass(probs: Sequence[Dict[str, float]], truths: Sequence[str], source: ConfidenceSource) -> float:
    """Mean over cases of the sum of squared errors across all labels. Needs a native distribution."""
    _require_calibratable(source)
    if not probs or len(probs) != len(truths):
        raise MetricsError("probs and truths must be non-empty and equal length")
    total = 0.0
    for dist, truth in zip(probs, truths):
        if abs(sum(dist.values()) - 1.0) > 1e-6:
            raise MetricsError("each probability distribution must sum to 1")
        total += sum((p - (1.0 if k == truth else 0.0)) ** 2 for k, p in dist.items())
    return total / len(probs)


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], n_bins: int, source: ConfidenceSource) -> float:
    """Equal-count bins ordered by confidence. Ties are broken by input order, so results are deterministic."""
    _require_calibratable(source)
    if n_bins < 1 or not confidences or len(confidences) != len(correct):
        raise MetricsError("invalid calibration input")
    if any(not 0.0 <= c <= 1.0 for c in confidences):
        raise MetricsError("confidences must lie in [0, 1]")
    order = sorted(range(len(confidences)), key=lambda i: (confidences[i], i))
    n = len(order)
    ece = 0.0
    for b in range(n_bins):
        idx = order[b * n // n_bins:(b + 1) * n // n_bins]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        ece += len(idx) / n * abs(avg_conf - acc)
    return ece
