"""Annotation templates and sheet import. Templates hold opaque ids and metadata only (no student text, no identifiers, no predictions, no suggested labels).

A reviewer fills one label per case in their own column; two independent reviewers, an adjudicator for conflicts and a named approver then go through the
`annotate` workflow (this module only reads filled sheets and forwards the labels to it). Nothing is ever approved from here.
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from msu_llm_lab.classification import annotation as an
from msu_llm_lab.classification import benchmark as bm
from msu_llm_lab.classification.records import Task
from msu_llm_lab.observability.analysis import load_case_meta
from msu_llm_lab.observability.labels import ANNOTATIONS_DIRNAME

TEMPLATES_DIRNAME = "templates"
COMMON = ["case_id", "row_ref", "task", "split", "view", "history_status", "builder_ambiguity_flag", "sample_block", "review_status",
          "reviewer_1_id", "reviewer_1_label", "reviewer_1_date", "reviewer_2_id", "reviewer_2_label", "reviewer_2_date", "labels_agree",
          "adjudicator_id", "adjudicated_label", "adjudication_note"]
INTENT_COLUMNS = COMMON + ["final_expected_intent", "approval_status", "approver_id", "approved_date", "notes"]
LEAD_COLUMNS = COMMON + ["final_expected_need_cta", "approval_status", "approver_id", "approved_date", "notes"]
README = """# Annotation templates

- One row per sampled case. `case_id` and `row_ref` are opaque ids; no student text is in these files. Review text comes from the local review queue (`annotate queue`), never from here.
- INTENT labels (exactly one): GREETING, OUT_OF_SCOPE, UPLOAD_STATUS_CHECK, HUMAN_HANDOFF_REQUEST, QUERY.
- LEAD labels: true or false (`need_cta`).
- Two DIFFERENT reviewers fill their own label columns independently, without seeing the other's. If they disagree, a third person fills the adjudication columns.
- `final_*`, `approval_status`, `approver_id` and `approved_date` are filled only through `annotate approve`; nothing here counts as ground truth until approved.
- Model predictions and model consensus must never be used as labels.
"""


def _row_ref(case_id: str) -> str:
    m = re.search(r"R\d+", case_id)
    return m.group(0) if m else ""


def write_templates(prep_dir: Path, loaded: "bm.LoadedCases", sampling: str, blocks: int, block_size: int, out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Create the INTENT and LEAD templates for the first `blocks` blocks. Never overwrites an existing file."""
    prep_dir = Path(prep_dir)
    out_dir = Path(out_dir) if out_dir else prep_dir / ANNOTATIONS_DIRNAME / TEMPLATES_DIRNAME
    meta = load_case_meta(prep_dir)
    paths: Dict[str, Path] = {}
    for task, cols in ((Task.INTENT, INTENT_COLUMNS), (Task.LEAD, LEAD_COLUMNS)):
        path = out_dir / f"{task.value.lower()}_review_template_{loaded.policy}_{blocks}blocks.csv"
        if path.exists():
            raise an.AnnotationError(f"{path} already exists; templates are never overwritten")
        cases = bm.order_cases(loaded, task, sampling)[: blocks * block_size]
        out_dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for i, c in enumerate(cases):
                m = meta.get(c.case_id)
                w.writerow({"case_id": c.case_id, "row_ref": _row_ref(c.case_id), "task": task.value, "split": getattr(m, "split", "") or "", "view": getattr(m, "view", "") or "",
                            "history_status": c.history_status.value, "builder_ambiguity_flag": (getattr(m, "ambiguity_count", 0) > 0) if (task == Task.LEAD and m) else "",
                            "sample_block": i // block_size, "review_status": "PENDING", "approval_status": "NOT_APPROVED"})
        paths[task.value] = path
    (out_dir / "README.md").write_text(README, encoding="utf-8") if not (out_dir / "README.md").exists() else None
    return paths


def import_sheet(sheet: Path, prep_dir: Path, valid_ids: Dict[str, set], slot: int) -> int:
    """Forward the labels one reviewer typed in slot 1 or 2 to the annotation workflow. Rows without a label or reviewer id are skipped."""
    if slot not in (1, 2):
        raise an.AnnotationError("slot must be 1 or 2")
    n = 0
    with Path(sheet).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label, who = (row.get(f"reviewer_{slot}_label") or "").strip(), (row.get(f"reviewer_{slot}_id") or "").strip()
            if label and who:
                an.submit(prep_dir, case_id=row["case_id"], task=row["task"], label=label, reviewer=who, valid_ids=valid_ids, note="imported from review sheet")
                n += 1
    return n
