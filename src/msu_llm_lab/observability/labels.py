"""Approved ground-truth labels, read-only.

Labels live in `<prep>/annotations/approved_labels.jsonl`, written only by the annotation workflow (`classification/annotation.py`:
two independent annotations, adjudication of conflicts, explicit named approval). This module only READS it, never creates,
rewrites or deletes anything there. A row counts only when its status is APPROVED and it names an approver; a LATER row for the same
case with any other status (for example REVOKED) withdraws the approval. Predictions, suggested labels and historical chatbot
answers are never labels.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from msu_llm_lab.classification.contracts import IntentLabel

ANNOTATIONS_DIRNAME = "annotations"
LABELS_FILENAME = "approved_labels.jsonl"
PENDING = "GROUND TRUTH PENDING"


@dataclass
class ApprovedLabels:
    source: Optional[Path] = None
    intent: Dict[str, IntentLabel] = field(default_factory=dict)
    lead: Dict[str, bool] = field(default_factory=dict)
    n_rows: int = 0
    n_not_approved: int = 0
    n_malformed: int = 0

    @property
    def n_approved(self) -> int:
        return len(self.intent) + len(self.lead)

    @property
    def available(self) -> bool:
        return self.n_approved > 0

    @property
    def status(self) -> str:
        return "APPROVED LABELS PRESENT" if self.available else PENDING


def labels_path(prep_dir: Path) -> Path:
    return Path(prep_dir) / ANNOTATIONS_DIRNAME / LABELS_FILENAME


def load_approved_labels(prep_dir: Optional[Path]) -> ApprovedLabels:
    out = ApprovedLabels()
    if prep_dir is None:
        return out
    path = labels_path(prep_dir)
    if not path.exists():
        return out
    out.source = path
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out.n_rows += 1
        try:
            d = json.loads(line)
            if d["task"] not in ("INTENT", "LEAD"):
                out.n_malformed += 1
                continue
            bucket = out.intent if d["task"] == "INTENT" else out.lead
            if d.get("status") != "APPROVED" or not str(d.get("approved_by") or "").strip():
                out.n_not_approved += 1
                bucket.pop(str(d.get("case_id")), None)      # a later non-approved row withdraws an earlier approval
                continue
            if d["task"] == "INTENT":
                out.intent[str(d["case_id"])] = IntentLabel(d["expected"])
            elif isinstance(d["expected"], bool):
                out.lead[str(d["case_id"])] = d["expected"]
            else:
                out.n_malformed += 1
        except Exception:
            out.n_malformed += 1
    return out
