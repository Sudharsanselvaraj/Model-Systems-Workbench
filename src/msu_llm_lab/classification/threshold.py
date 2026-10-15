"""Probability-based Lead Qualifier analysis: threshold sweep, dev-only threshold selection, frozen evaluation.

Jev's Noul answer is P(yes) only. No threshold is documented, so one is chosen here from the DEVELOPMENT split
only (initial criterion: maximum F1), frozen, and then applied unchanged to held-out data. Nothing in this module
ever selects a threshold from held-out data. 0.5 is always reported as a reference.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from msu_llm_lab.schemas.evaluation import DatasetSplit

REFERENCE_THRESHOLD = 0.5
CRITERION_MAX_F1 = "MAX_F1"


class ThresholdError(ValueError):
    pass


@dataclass
class ProbItem:
    case_id: str
    expected: Optional[bool]       # approved ground truth; None = unlabeled
    yes_prob: Optional[float]      # native P(yes); None = unavailable (never imputed)
    split: DatasetSplit
    exclusion_reason: Optional[str] = None


class SweepRow(BaseModel):
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]
    accuracy: Optional[float]


class FrozenThreshold(BaseModel):
    """Immutable record of a threshold selected on the dev split. Must exist before any held-out evaluation."""
    model_config = ConfigDict(frozen=True)

    threshold: float
    criterion: str
    selected_on: str
    n_dev_scored: int
    dev_f1_at_threshold: float
    dev_f1_at_reference: Optional[float]
    dev_case_fingerprint: str


class HeldOutReport(BaseModel):
    n_scored: int
    n_missing_prob: int
    frozen_threshold: float
    at_frozen: SweepRow
    at_reference: SweepRow
    roc_auc: Optional[float]
    average_precision: Optional[float]
    brier: Optional[float]


def _usable(items: Sequence[ProbItem]) -> Tuple[List[ProbItem], int]:
    scored, missing = [], 0
    for i in items:
        if i.exclusion_reason is not None or i.expected is None:
            continue
        if i.yes_prob is None:
            missing += 1
            continue
        if not (isinstance(i.yes_prob, (int, float)) and math.isfinite(i.yes_prob) and 0.0 <= i.yes_prob <= 1.0):
            raise ThresholdError(f"yes_prob for {i.case_id} must be a number in [0, 1]")
        scored.append(i)
    return scored, missing


def _div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def evaluate_at(items: Sequence[ProbItem], threshold: float) -> SweepRow:
    """Predict YES when yes_prob >= threshold."""
    scored, _ = _usable(items)
    tp = sum(1 for i in scored if i.expected and i.yes_prob >= threshold)
    fn = sum(1 for i in scored if i.expected and i.yes_prob < threshold)
    fp = sum(1 for i in scored if not i.expected and i.yes_prob >= threshold)
    tn = sum(1 for i in scored if not i.expected and i.yes_prob < threshold)
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = None if (p is None or r is None) else (0.0 if tp == 0 else 2 * p * r / (p + r))
    return SweepRow(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=f1,
                    false_positive_rate=_div(fp, fp + tn), false_negative_rate=_div(fn, fn + tp),
                    accuracy=_div(tp + tn, len(scored)))


def candidate_thresholds(items: Sequence[ProbItem]) -> List[float]:
    """Midpoints between consecutive distinct probabilities, plus the 0.5 reference. Deterministic and sorted."""
    scored, _ = _usable(items)
    vals = sorted({float(i.yes_prob) for i in scored})
    mids = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    return sorted(set(mids) | {REFERENCE_THRESHOLD})


def sweep(items: Sequence[ProbItem]) -> List[SweepRow]:
    """Descriptive table for any split. Reporting a sweep is analysis; only select_threshold_on_dev may choose a threshold."""
    return [evaluate_at(items, t) for t in candidate_thresholds(items)]


def select_threshold_on_dev(items: Sequence[ProbItem]) -> FrozenThreshold:
    """Maximum F1 on DEV items only. Ties: higher precision, then higher threshold (the more conservative cut)."""
    if not items:
        raise ThresholdError("No items supplied")
    if any(i.split != DatasetSplit.DEV for i in items):
        raise ThresholdError("Threshold selection accepts DEV items only; held-out data must never influence it")
    scored, _ = _usable(items)
    if not any(i.expected for i in scored) or all(i.expected for i in scored):
        raise ThresholdError("Dev data needs both YES and NO labelled cases with probabilities to select a threshold")
    rows = [r for r in sweep(items) if r.f1 is not None]
    best = max(rows, key=lambda r: (r.f1, r.precision or 0.0, r.threshold))
    ref = evaluate_at(items, REFERENCE_THRESHOLD)
    fp = hashlib.sha256("|".join(sorted(i.case_id for i in scored)).encode()).hexdigest()
    return FrozenThreshold(threshold=best.threshold, criterion=CRITERION_MAX_F1, selected_on="DEV", n_dev_scored=len(scored),
                           dev_f1_at_threshold=best.f1, dev_f1_at_reference=ref.f1, dev_case_fingerprint=fp)


def roc_auc(items: Sequence[ProbItem]) -> Optional[float]:
    """Rank-based AUC with ties counted as half. None if one class is absent."""
    scored, _ = _usable(items)
    pos = [i.yes_prob for i in scored if i.expected]
    neg = [i.yes_prob for i in scored if not i.expected]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def average_precision(items: Sequence[ProbItem]) -> Optional[float]:
    scored, _ = _usable(items)
    n_pos = sum(1 for i in scored if i.expected)
    if n_pos == 0:
        return None
    ordered = sorted(scored, key=lambda i: (-i.yes_prob, i.case_id))
    tp, ap = 0, 0.0
    for rank, i in enumerate(ordered, 1):
        if i.expected:
            tp += 1
            ap += tp / rank
    return ap / n_pos


def brier_binary(items: Sequence[ProbItem]) -> Optional[float]:
    scored, _ = _usable(items)
    if not scored:
        return None
    return sum((i.yes_prob - (1.0 if i.expected else 0.0)) ** 2 for i in scored) / len(scored)


def evaluate_held_out(items: Sequence[ProbItem], frozen: Optional[FrozenThreshold]) -> HeldOutReport:
    """Apply the frozen dev threshold, unchanged, to TEST items. Refuses without a frozen threshold or with non-TEST data."""
    if frozen is None:
        raise ThresholdError("A frozen dev-selected threshold is required before held-out evaluation")
    if not items or any(i.split != DatasetSplit.TEST for i in items):
        raise ThresholdError("Held-out evaluation accepts TEST items only")
    scored, missing = _usable(items)
    return HeldOutReport(
        n_scored=len(scored), n_missing_prob=missing, frozen_threshold=frozen.threshold,
        at_frozen=evaluate_at(items, frozen.threshold), at_reference=evaluate_at(items, REFERENCE_THRESHOLD),
        roc_auc=roc_auc(items), average_precision=average_precision(items), brier=brier_binary(items),
    )
