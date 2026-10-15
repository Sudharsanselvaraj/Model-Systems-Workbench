"""Analysis of stored prediction records. Pure functions over records, approved labels and non-text case metadata.

Supervised numbers (accuracy, F1, calibration, unanimous errors, held-out thresholds) need human-approved labels and are
reported as GROUND TRUTH PENDING without them. Model agreement is never reported as correctness. Nothing here sends a request.
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from msu_llm_lab.classification import metrics as M
from msu_llm_lab.classification import threshold as T
from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.records import PredictionRecord, Task
from msu_llm_lab.observability.labels import PENDING, ApprovedLabels
from msu_llm_lab.schemas.evaluation import DatasetSplit

MIN_SAMPLE = 30          # below this a metric is flagged as indicative only
MIN_CLASS_SUPPORT = 5    # below this a per-class figure is flagged insufficient
FRONTIER_KEYS = ("gpt-4.1", "claude-sonnet-5.5")
HIGH_CONFIDENCE = 0.9


# ---------------------------------------------------------------- loading
def load_predictions(dirs: Iterable[Path]) -> List[PredictionRecord]:
    out: List[PredictionRecord] = []
    for d in dirs:
        d = Path(d)
        for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.append(PredictionRecord.model_validate_json(line))
            except Exception:
                continue  # an unreadable file is skipped; it is never repaired or rewritten
    return out


def latest_runs(preds: Sequence[PredictionRecord]) -> Dict[Tuple[str, str], List[PredictionRecord]]:
    """(model_key, task) -> records of the most recent run for that pair."""
    groups: Dict[Tuple[str, str, str], List[PredictionRecord]] = defaultdict(list)
    for r in preds:
        groups[(r.model_key, r.task.value, r.run_id)].append(r)
    best: Dict[Tuple[str, str], Tuple[str, List[PredictionRecord]]] = {}
    for (mk, task, run_id), recs in groups.items():
        newest = max(r.created_at for r in recs)
        cur = best.get((mk, task))
        if cur is None or newest > max(r.created_at for r in cur[1]):
            best[(mk, task)] = (run_id, recs)
    return {k: v[1] for k, v in best.items()}


@dataclass
class CaseMeta:
    case_id: str
    split: Optional[str] = None
    view: Optional[str] = None
    history_status: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)
    ambiguity_count: int = 0
    unresolved_rules: List[str] = field(default_factory=list)
    tier_a: Optional[int] = None
    tier_b: Optional[int] = None


_META_CACHE: Dict[Tuple, Dict[str, CaseMeta]] = {}


def load_case_meta(prep_dir: Optional[Path]) -> Dict[str, CaseMeta]:
    """Non-text metadata of prepared candidates (split, view, history status, builder flags). Student text is parsed and discarded."""
    if prep_dir is None or not Path(prep_dir).exists():
        return {}
    files = sorted(Path(prep_dir).glob("intent_candidates_*.jsonl")) + sorted(Path(prep_dir).glob("lead_candidates_*.jsonl"))
    key = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    if key in _META_CACHE:
        return _META_CACHE[key]
    meta: Dict[str, CaseMeta] = {}
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            c, cand = d.get("case", {}), d.get("candidate", {})
            cid = c.get("case_id")
            if not cid:
                continue
            tally = cand.get("builder_tally_not_a_label") or {}
            meta[cid] = CaseMeta(
                case_id=cid, split=cand.get("split"), view=cand.get("view"), history_status=c.get("history_status"),
                quality_flags=list(cand.get("quality_flags") or []), ambiguity_count=int(cand.get("ambiguity_count") or 0),
                unresolved_rules=list(cand.get("unresolved_rules") or []),
                tier_a=tally.get("tier_a_actions"), tier_b=tally.get("tier_b_distinct_types"),
            )
    _META_CACHE.clear()
    _META_CACHE[key] = meta
    return meta


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def intent_pred(r: PredictionRecord) -> Optional[IntentLabel]:
    return r.predicted_label if r.status == "OK" else None


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return None if pe >= 1.0 else (po - pe) / (1 - pe)


# ---------------------------------------------------------------- quality
@dataclass
class IntentQuality:
    model_key: str
    run_id: str
    n_predictions: int
    n_request_errors: int
    status: str                              # PENDING text or "SCORED"
    metrics: Optional[M.IntentMetrics] = None
    accuracy_ci: Optional[Tuple[float, float]] = None
    flags: List[str] = field(default_factory=list)


def intent_quality(runs: Dict[Tuple[str, str], List[PredictionRecord]], labels: ApprovedLabels) -> List[IntentQuality]:
    out = []
    for (mk, task), recs in sorted(runs.items()):
        if task != Task.INTENT.value:
            continue
        q = IntentQuality(mk, recs[0].run_id, len(recs), sum(r.status == "ERROR" for r in recs), PENDING)
        scorable = [r for r in recs if r.case_id in labels.intent]
        if labels.intent and scorable:
            items = [M.IntentItem(expected=labels.intent[r.case_id], predicted=intent_pred(r)) for r in scorable]
            q.metrics = M.intent_metrics(items)
            q.status = "SCORED"
            n = q.metrics.n_scored
            if q.metrics.accuracy is not None:
                q.accuracy_ci = wilson(round(q.metrics.accuracy * n), n)
            if n < MIN_SAMPLE:
                q.flags.append(f"INSUFFICIENT SAMPLE (n={n} < {MIN_SAMPLE}): indicative only")
            thin = [c for c, s in q.metrics.per_class.items() if s.support < MIN_CLASS_SUPPORT]
            if thin:
                q.flags.append("RARE CLASSES BELOW SUPPORT %d: %s" % (MIN_CLASS_SUPPORT, ", ".join(f"{c}({q.metrics.per_class[c].support})" for c in thin)))
        out.append(q)
    return out


def lead_pred(r: PredictionRecord, frozen: Optional[T.FrozenThreshold]) -> Optional[bool]:
    if r.status != "OK":
        return None
    if r.predicted_need_cta is not None:
        return r.predicted_need_cta
    if r.yes_prob is not None and frozen is not None:
        return r.yes_prob >= frozen.threshold
    return None


@dataclass
class LeadModelReport:
    model_key: str
    run_id: str
    n_predictions: int
    n_yes: int
    n_no: int
    n_undecided: int                          # invalid, errored, or P(yes) with no frozen threshold yet
    n_with_prob: int
    status: str = PENDING
    metrics: Optional[M.LeadMetrics] = None
    frozen: Optional[T.FrozenThreshold] = None
    held_out: Optional[T.HeldOutReport] = None
    threshold_note: str = ""
    flags: List[str] = field(default_factory=list)


def lead_reports(runs, labels: ApprovedLabels, meta: Dict[str, CaseMeta]) -> List[LeadModelReport]:
    out = []
    for (mk, task), recs in sorted(runs.items()):
        if task != Task.LEAD.value:
            continue
        frozen: Optional[T.FrozenThreshold] = None
        note = ""
        probs = [r for r in recs if r.yes_prob is not None]
        if probs:
            dev = [T.ProbItem(r.case_id, labels.lead.get(r.case_id), r.yes_prob, DatasetSplit.DEV) for r in probs
                   if meta.get(r.case_id) and meta[r.case_id].split == "DEV" and r.case_id in labels.lead]
            try:
                frozen = T.select_threshold_on_dev(dev)
            except T.ThresholdError as e:
                note = f"NO FROZEN THRESHOLD: {e}"
        rep = LeadModelReport(mk, recs[0].run_id, len(recs), 0, 0, 0, len(probs), frozen=frozen, threshold_note=note)
        for r in recs:
            p = lead_pred(r, frozen)
            if p is True:
                rep.n_yes += 1
            elif p is False:
                rep.n_no += 1
            else:
                rep.n_undecided += 1
        scorable = [r for r in recs if r.case_id in labels.lead]
        if scorable:
            rep.metrics = M.lead_metrics([M.LeadItem(expected=labels.lead[r.case_id], predicted=lead_pred(r, frozen)) for r in scorable])
            rep.status = "SCORED"
            if rep.metrics.n_scored < MIN_SAMPLE:
                rep.flags.append(f"INSUFFICIENT SAMPLE (n={rep.metrics.n_scored} < {MIN_SAMPLE})")
        if frozen is not None:
            test = [T.ProbItem(r.case_id, labels.lead[r.case_id], r.yes_prob, DatasetSplit.TEST) for r in probs
                    if meta.get(r.case_id) and meta[r.case_id].split == "TEST" and r.case_id in labels.lead]
            if test:
                rep.held_out = T.evaluate_held_out(test, frozen)
        out.append(rep)
    return out


# ---------------------------------------------------------------- calibration
@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float


@dataclass
class CalibrationReport:
    model_key: str
    n_records: int
    n_with_distribution: int
    n_invalid_distribution: int
    n_with_native_confidence: int
    mean_selected_prob: Optional[float]
    mean_native_confidence: Optional[float]
    unsupported_reason: Optional[str] = None
    status: str = PENDING
    n_calibration: int = 0
    ece: Optional[float] = None
    brier: Optional[float] = None
    bins: List[ReliabilityBin] = field(default_factory=list)
    high_conf_errors: List[Tuple[str, float, str, str]] = field(default_factory=list)   # (case_id, confidence, predicted, expected)
    coverage: List[Tuple[float, int, Optional[float]]] = field(default_factory=list)     # (threshold, n kept, accuracy of kept)
    flags: List[str] = field(default_factory=list)


def calibration_reports(runs, labels: ApprovedLabels) -> List[CalibrationReport]:
    out = []
    for (mk, task), recs in sorted(runs.items()):
        if task != Task.INTENT.value:
            continue
        dists = []
        invalid = 0
        for r in recs:
            if r.label_probs and r.confidence_source == "NATIVE_DISTRIBUTION" and r.status == "OK" and r.predicted_label is not None:
                if abs(sum(r.label_probs.values()) - 1.0) <= 1e-6 and r.predicted_label.value in r.label_probs:
                    dists.append(r)
                else:
                    invalid += 1
        nat = [r.native_confidence for r in recs if r.native_confidence is not None]
        rep = CalibrationReport(mk, len(recs), len(dists), invalid, len(nat),
                                (sum(r.label_probs[r.predicted_label.value] for r in dists) / len(dists)) if dists else None,
                                (sum(nat) / len(nat)) if nat else None)
        if not dists:
            rep.unsupported_reason = "UNAVAILABLE: this model returned no native probability distribution (self-reported or textual confidence is never used)"
            rep.status = "UNAVAILABLE"
            out.append(rep)
            continue
        lab = [r for r in dists if r.case_id in labels.intent]
        if lab:
            truths = [labels.intent[r.case_id].value for r in lab]
            confs = [r.label_probs[r.predicted_label.value] for r in lab]
            correct = [r.predicted_label.value == t for r, t in zip(lab, truths)]
            src = M.ConfidenceSource.NATIVE_DISTRIBUTION
            rep.n_calibration = len(lab)
            rep.status = "SCORED"
            rep.brier = M.brier_multiclass([r.label_probs for r in lab], truths, src)
            rep.ece = M.expected_calibration_error(confs, correct, max(1, min(10, len(lab) // 5)), src)  # >= 5 cases per bin on average
            for b in range(10):
                lo, hi = b / 10, (b + 1) / 10
                idx = [i for i, c in enumerate(confs) if lo <= c < hi or (b == 9 and c == 1.0)]
                if idx:
                    rep.bins.append(ReliabilityBin(lo, hi, len(idx), sum(confs[i] for i in idx) / len(idx), sum(correct[i] for i in idx) / len(idx)))
            rep.high_conf_errors = [(r.case_id, c, r.predicted_label.value, t) for r, c, ok, t in zip(lab, confs, correct, truths)
                                    if c >= HIGH_CONFIDENCE and not ok]
            for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                kept = [i for i, c in enumerate(confs) if c >= thr]
                rep.coverage.append((thr, len(kept), (sum(correct[i] for i in kept) / len(kept)) if kept else None))
            if len(lab) < MIN_SAMPLE:
                rep.flags.append(f"INSUFFICIENT SAMPLE (n={len(lab)} < {MIN_SAMPLE}): calibration is indicative only")
        out.append(rep)
    return out


# ---------------------------------------------------------------- agreement
@dataclass
class PairAgreement:
    a: str
    b: str
    n_common: int                        # cases where BOTH models produced a valid output
    agreement: Optional[float]
    kappa: Optional[float]
    note: str = ""
    n_unusable: int = 0                  # cases both models were run on where at least one output was an error, invalid or undecided (not a disagreement)


@dataclass
class AgreementReport:
    task: str
    models: List[str]
    pairs: List[PairAgreement]
    n_cases_any: int
    n_cases_all_models: int
    disagreements: List[Tuple[str, Dict[str, str]]]      # case_id -> predictions by model
    unanimous_errors: Optional[List[str]]                # None = GROUND TRUTH PENDING
    unanimous_error_basis: str
    excluded: Dict[str, str]                             # model -> why it is not comparable
    frontier_agreement: Dict[str, Dict[str, Optional[float]]]


def agreement_report(runs, task: Task, labels: ApprovedLabels, frozen_by_model: Optional[Dict[str, T.FrozenThreshold]] = None) -> AgreementReport:
    """Agreement on VALID common outputs only. An error, an invalid output or an undecided P(yes) is never a vote and never a disagreement."""
    frozen_by_model = frozen_by_model or {}
    cat: Dict[str, Dict[str, str]] = {}           # model -> case -> valid prediction
    attempted: Dict[str, set] = {}                # model -> cases it was run on (valid or not)
    excluded: Dict[str, str] = {}
    for (mk, t), recs in runs.items():
        if t != task.value:
            continue
        m: Dict[str, str] = {}
        attempted[mk] = {r.case_id for r in recs}
        undecided = 0
        for r in recs:
            if task == Task.INTENT:
                p = intent_pred(r)
                if p is not None:
                    m[r.case_id] = p.value
            else:
                p = lead_pred(r, frozen_by_model.get(mk))
                if p is not None:
                    m[r.case_id] = "YES" if p else "NO"
                elif r.status == "OK" and r.yes_prob is not None:
                    undecided += 1
        if task == Task.LEAD and undecided and not m:
            excluded[mk] = "P(yes) only and no frozen dev threshold: no YES/NO decision to compare"
            continue
        cat[mk] = m
    models = sorted(cat)
    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            common = sorted(set(cat[a]) & set(cat[b]))
            unusable = len((attempted[a] & attempted[b]) - set(common))
            xs, ys = [cat[a][c] for c in common], [cat[b][c] for c in common]
            note = "" if len(common) >= MIN_SAMPLE else f"n={len(common)} < {MIN_SAMPLE}: indicative only"
            pairs.append(PairAgreement(a, b, len(common), (sum(x == y for x, y in zip(xs, ys)) / len(common)) if common else None, cohen_kappa(xs, ys), note, unusable))
    all_cases = set().union(*[set(m) for m in cat.values()]) if cat else set()
    in_all = set.intersection(*[set(m) for m in cat.values()]) if cat else set()
    dis = []
    for c in sorted(in_all):
        preds = {mk: cat[mk][c] for mk in models}
        if len(set(preds.values())) > 1:
            dis.append((c, preds))
    frontier: Dict[str, Dict[str, Optional[float]]] = {}
    for mk in models:
        frontier[mk] = {}
        for fk in FRONTIER_KEYS:
            if fk in cat and mk != fk:
                p = next((x for x in pairs if {x.a, x.b} == {mk, fk}), None)
                frontier[mk][fk] = p.agreement if p else None
    ue: Optional[List[str]] = None
    basis = PENDING
    truth = labels.intent if task == Task.INTENT else {k: ("YES" if v else "NO") for k, v in labels.lead.items()}
    if truth and in_all:
        lab_cases = sorted(c for c in in_all if c in truth)
        tv = {c: (truth[c].value if task == Task.INTENT else truth[c]) for c in lab_cases}
        ue = [c for c in lab_cases if all(cat[mk][c] != tv[c] for mk in models) and len({cat[mk][c] for mk in models}) == 1]
        basis = f"{len(lab_cases)} approved-label cases shared by {len(models)} models"
    return AgreementReport(task.value, models, pairs, len(all_cases), len(in_all), dis, ue, basis, excluded, frontier)


# ---------------------------------------------------------------- robustness
@dataclass
class Repeatability:
    model_key: str
    task: str
    n_cases_repeated: int
    n_identical: int


def repeatability(preds: Sequence[PredictionRecord]) -> List[Repeatability]:
    by: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    for r in preds:
        val = (intent_pred(r).value if intent_pred(r) else "INVALID") if r.task == Task.INTENT else (
            "INVALID" if r.status != "OK" else str(r.predicted_need_cta if r.predicted_need_cta is not None else r.yes_prob))
        by[(r.model_key, r.task.value, r.case_id)][r.run_id] = val
    agg: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for (mk, task, _), runs in by.items():
        if len(runs) >= 2:
            agg[(mk, task)][0] += 1
            agg[(mk, task)][1] += len(set(runs.values())) == 1
    return [Repeatability(mk, t, a, b) for (mk, t), (a, b) in sorted(agg.items())]


def history_slices(runs, labels: ApprovedLabels, meta: Dict[str, CaseMeta]) -> List[Tuple[str, str, int, Optional[float]]]:
    """(model, history_status, n_scored, accuracy). Needs approved labels and case metadata; otherwise empty."""
    out = []
    for (mk, task), recs in sorted(runs.items()):
        if task != Task.INTENT.value or not labels.intent:
            continue
        groups: Dict[str, List[PredictionRecord]] = defaultdict(list)
        for r in recs:
            if r.case_id in labels.intent and meta.get(r.case_id) and meta[r.case_id].history_status:
                groups[meta[r.case_id].history_status].append(r)
        for hs, rs in sorted(groups.items()):
            ok = sum(intent_pred(r) == labels.intent[r.case_id] for r in rs)
            out.append((mk, hs, len(rs), ok / len(rs)))
    return out
