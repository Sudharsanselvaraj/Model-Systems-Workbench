"""Independent annotation, adjudication, approval and freezing of ground-truth labels. Local files only; nothing is ever inferred.

Event log `<prep>/annotations/events.jsonl` (append-only). Labels become ground truth only through this path:

  ANNOTATION  two DISTINCT reviewers label a case independently (the queue never shows another reviewer's label)
  agreement   -> the case is AGREED;  disagreement -> CONFLICT, which a THIRD reviewer settles with an ADJUDICATION
  APPROVAL    a named approver explicitly approves AGREED/ADJUDICATED cases; this writes `approved_labels.jsonl`
  FREEZE      the approved set is hashed and recorded so a later benchmark can prove which labels it used

Model predictions, suggested labels and historical chatbot answers are never inputs here. Nothing is approved automatically.
"""

import hashlib
import json
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from msu_llm_lab.classification import benchmark as bm
from msu_llm_lab.classification import serialization
from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.records import Task
from msu_llm_lab.observability.labels import ANNOTATIONS_DIRNAME, LABELS_FILENAME

EVENTS_FILENAME = "events.jsonl"
QUEUE_DIRNAME = "review_queue"


class AnnotationError(ValueError):
    pass


def ann_dir(prep_dir: Path) -> Path:
    return Path(prep_dir) / ANNOTATIONS_DIRNAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def parse_label(task: str, raw: str):
    if task == "INTENT":
        try:
            return IntentLabel(raw.strip().upper()).value
        except ValueError:
            raise AnnotationError(f"'{raw}' is not one of {[l.value for l in IntentLabel]}")
    if raw.strip().lower() in ("true", "yes", "1"):
        return True
    if raw.strip().lower() in ("false", "no", "0"):
        return False
    raise AnnotationError("a LEAD label must be true or false")


def eligible_case_ids(loaded: "bm.LoadedCases") -> Dict[str, set]:
    return {"INTENT": {c.case_id for c in loaded.intent}, "LEAD": {c.case_id for c in loaded.lead}}


def write_queue(prep_dir: Path, loaded: "bm.LoadedCases", task: Task, limit: Optional[int] = None) -> Path:
    """Write the deterministic review queue (redacted case text for a human reviewer). LOCAL ONLY; never contains any label or prediction."""
    cases = bm.ordered(loaded.for_task(task))[: limit or None]
    path = ann_dir(prep_dir) / QUEUE_DIRNAME / f"{task.value.lower()}_{loaded.policy}.jsonl"
    if path.exists():
        raise AnnotationError(f"{path} already exists; delete it deliberately if you want a new queue")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps({"case_id": c.case_id, "task": task.value, "history_status": c.history_status.value,
                                 "text_for_review": bm._state_text(task, c)}, ensure_ascii=False) + "\n")
    return path


def submit(prep_dir: Path, *, case_id: str, task: str, label: str, reviewer: str, valid_ids: Dict[str, set], note: str = "") -> Dict[str, Any]:
    if not reviewer.strip():
        raise AnnotationError("a reviewer name is required")
    if case_id not in valid_ids.get(task, set()):
        raise AnnotationError(f"{case_id} is not an eligible {task} case of this preparation (excluded or unknown cases cannot be labelled)")
    row = {"event_id": uuid.uuid4().hex, "ts": _now(), "type": "ANNOTATION", "task": task, "case_id": case_id, "label": parse_label(task, label),
           "reviewer": reviewer.strip(), "note": note[:200]}
    _append(ann_dir(prep_dir) / EVENTS_FILENAME, row)
    return row


def adjudicate(prep_dir: Path, *, case_id: str, task: str, label: str, adjudicator: str, note: str = "") -> Dict[str, Any]:
    st = case_states(prep_dir).get((task, case_id))
    if st is None or st.state != "CONFLICT":
        raise AnnotationError("only a case with two disagreeing independent annotations can be adjudicated")
    if adjudicator.strip() in st.annotators:
        raise AnnotationError("the adjudicator must be a third reviewer, distinct from both annotators")
    row = {"event_id": uuid.uuid4().hex, "ts": _now(), "type": "ADJUDICATION", "task": task, "case_id": case_id, "label": parse_label(task, label),
           "reviewer": adjudicator.strip(), "note": note[:200]}
    _append(ann_dir(prep_dir) / EVENTS_FILENAME, row)
    return row


@dataclass
class CaseState:
    task: str
    case_id: str
    annotations: Dict[str, Any] = field(default_factory=dict)       # reviewer -> label (last wins per reviewer)
    adjudication: Optional[Tuple[str, Any]] = None
    approved: Optional[Dict[str, Any]] = None

    @property
    def annotators(self) -> List[str]:
        return sorted(self.annotations)

    @property
    def state(self) -> str:
        if self.approved is not None:
            return "APPROVED"
        if self.adjudication is not None:
            return "ADJUDICATED"
        labels = list(self.annotations.values())
        if len(labels) < 2:
            return "PENDING_SECOND_ANNOTATION" if labels else "UNLABELLED"
        return "AGREED" if len(set(labels)) == 1 else "CONFLICT"

    @property
    def proposed(self) -> Optional[Tuple[Any, str]]:
        if self.adjudication is not None:
            return self.adjudication[1], "ADJUDICATION"
        if self.state == "AGREED":
            return next(iter(self.annotations.values())), "AGREEMENT"
        return None


def read_events(prep_dir: Path) -> List[Dict[str, Any]]:
    p = ann_dir(prep_dir) / EVENTS_FILENAME
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def case_states(prep_dir: Path) -> Dict[Tuple[str, str], CaseState]:
    out: Dict[Tuple[str, str], CaseState] = {}
    for e in read_events(prep_dir):
        st = out.setdefault((e["task"], e["case_id"]), CaseState(e["task"], e["case_id"]))
        if e["type"] == "ANNOTATION":
            st.annotations[e["reviewer"]] = e["label"]
        elif e["type"] == "ADJUDICATION":
            st.adjudication = (e["reviewer"], e["label"])
        elif e["type"] == "APPROVAL":
            st.approved = e
        elif e["type"] == "REVOCATION":
            st.approved = None
    return out


def summary(prep_dir: Path, valid_ids: Optional[Dict[str, set]] = None) -> Dict[str, Any]:
    states = case_states(prep_dir)
    by: Dict[str, Counter] = defaultdict(Counter)
    for (task, _), st in states.items():
        by[task][st.state] += 1
    out = {t: dict(c) for t, c in by.items()}
    if valid_ids:
        for t, ids in valid_ids.items():
            out.setdefault(t, {})["UNLABELLED"] = len(ids - {cid for (tt, cid) in states if tt == t})
    return out


def approve(prep_dir: Path, *, approver: str, case_ids: Optional[Sequence[Tuple[str, str]]] = None, all_ready: bool = False) -> List[Dict[str, Any]]:
    """Approve AGREED / ADJUDICATED cases. Needs an explicit list or all_ready=True; appends to the approved-label file."""
    if not approver.strip():
        raise AnnotationError("a named approver is required")
    states = case_states(prep_dir)
    chosen = [s for (t, c), s in sorted(states.items()) if (all_ready and s.proposed is not None and s.state != "APPROVED") or (case_ids and (t, c) in set(case_ids))]
    rows = []
    for s in chosen:
        if s.proposed is None:
            raise AnnotationError(f"{s.case_id} ({s.task}) is {s.state}: only AGREED or ADJUDICATED cases can be approved")
        if s.state == "APPROVED":
            continue
        label, basis = s.proposed
        rows.append({"case_id": s.case_id, "task": s.task, "expected": label, "status": "APPROVED", "approved_by": approver.strip(), "approved_at": _now(),
                     "basis": basis, "annotators": s.annotators, "adjudicator": s.adjudication[0] if s.adjudication else None})
    for r in rows:
        _append(ann_dir(prep_dir) / LABELS_FILENAME, r)
        _append(ann_dir(prep_dir) / EVENTS_FILENAME, {"event_id": uuid.uuid4().hex, "ts": r["approved_at"], "type": "APPROVAL", "task": r["task"], "case_id": r["case_id"],
                                                     "label": r["expected"], "reviewer": r["approved_by"]})
    return rows


def revoke(prep_dir: Path, *, task: str, case_id: str, by: str, reason: str) -> None:
    """Withdraw an approval. The history stays; the later REVOKED row overrides the earlier APPROVED row in the loader."""
    st = case_states(prep_dir).get((task, case_id))
    if st is None or st.approved is None:
        raise AnnotationError("that case is not approved")
    now = _now()
    _append(ann_dir(prep_dir) / LABELS_FILENAME, {"case_id": case_id, "task": task, "expected": None, "status": "REVOKED", "approved_by": by, "approved_at": now, "reason": reason[:200]})
    _append(ann_dir(prep_dir) / EVENTS_FILENAME, {"event_id": uuid.uuid4().hex, "ts": now, "type": "REVOCATION", "task": task, "case_id": case_id, "reviewer": by, "note": reason[:200]})


def freeze(prep_dir: Path, *, by: str, dataset_version: str) -> Path:
    """Record the exact approved label set (hash + counts) so a later label-backed benchmark can prove what it used. Never overwrites."""
    from msu_llm_lab.observability.labels import load_approved_labels

    labels = load_approved_labels(prep_dir)
    if not labels.available:
        raise AnnotationError("there are no approved labels to freeze")
    rows = sorted([("INTENT", k, v.value) for k, v in labels.intent.items()] + [("LEAD", k, v) for k, v in labels.lead.items()], key=str)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    n = 1 + len(list(ann_dir(prep_dir).glob("frozen_labels_v*.json")))
    path = ann_dir(prep_dir) / f"frozen_labels_v{n}.json"
    path.write_text(json.dumps({"version": n, "dataset_version": dataset_version, "frozen_by": by, "frozen_at": _now(), "approved_sha256": digest,
                                "n_intent": len(labels.intent), "n_lead": len(labels.lead),
                                "rule": "thresholds are selected on DEV only and applied unchanged to TEST; later label edits create a new freeze"}, indent=2) + "\n")
    return path


def frozen_matches_current(prep_dir: Path) -> Optional[bool]:
    """True if the latest freeze equals the current approved set, False if labels changed since, None if never frozen."""
    from msu_llm_lab.observability.labels import load_approved_labels

    frozen = sorted(ann_dir(prep_dir).glob("frozen_labels_v*.json"))
    if not frozen:
        return None
    labels = load_approved_labels(prep_dir)
    rows = sorted([("INTENT", k, v.value) for k, v in labels.intent.items()] + [("LEAD", k, v) for k, v in labels.lead.items()], key=str)
    return json.loads(frozen[-1].read_text())["approved_sha256"] == hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
