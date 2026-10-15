"""LiveMonitor: the one read-only subscription the dashboard uses.

It owns a single EventTail and a cache of analysis results. `poll()` only reads files; it never sends a request, writes a
label or touches the preparation folder. Polling is driven by the UI loop (no threads), so there is nothing to leak: `close()`
drops the tail and caches.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from msu_llm_lab.classification.records import PredictionRecord, Task
from msu_llm_lab.observability import analysis as A
from msu_llm_lab.observability.aggregate import Aggregate, aggregate
from msu_llm_lab.observability.events import EventTail, RequestEvent, default_events_path
from msu_llm_lab.observability.labels import ApprovedLabels, load_approved_labels


@dataclass
class DatasetInfo:
    prep_dir: Optional[Path] = None
    version: Optional[str] = None
    parity: str = "PRODUCTION_PARITY_NOT_ESTABLISHED"
    intent_rows: Optional[int] = None
    lead_sessions: Optional[Dict[str, int]] = None
    session_policy: str = "NOT SELECTED (30 min and 2 h both provisional)"
    views_note: str = "challenge pool is a heuristic candidate set, not validated as a rare-class benchmark"
    labels_dir_present: bool = False


def discover_dataset(base: Path) -> DatasetInfo:
    info = DatasetInfo()
    manifests = sorted(Path(base).glob("*/manifest.json")) if Path(base).exists() else []
    if not manifests:
        return info
    p = manifests[-1]
    try:
        m = json.loads(p.read_text())
        info.prep_dir = p.parent
        info.version = m.get("prep_version")
        info.parity = m.get("parity_status", info.parity)
        info.intent_rows = m["candidates"]["intent_2h"]["cases"]
        info.lead_sessions = dict(m["session_comparison"]["sessions"])
    except Exception:
        info.prep_dir = p.parent
    info.labels_dir_present = (p.parent / "annotations").exists()
    return info


def default_prediction_dirs() -> List[Path]:
    from msu_llm_lab.core.config import settings
    return [settings.reports_dir / "smoke", settings.reports_dir / "classification"]


@dataclass
class Analysis:
    predictions: List[PredictionRecord] = field(default_factory=list)
    labels: ApprovedLabels = field(default_factory=ApprovedLabels)
    meta: Dict[str, A.CaseMeta] = field(default_factory=dict)
    runs: Dict[Tuple[str, str], List[PredictionRecord]] = field(default_factory=dict)
    intent_quality: List[A.IntentQuality] = field(default_factory=list)
    calibration: List[A.CalibrationReport] = field(default_factory=list)
    lead: List[A.LeadModelReport] = field(default_factory=list)
    agreement_intent: Optional[A.AgreementReport] = None
    agreement_lead: Optional[A.AgreementReport] = None
    repeat: List[A.Repeatability] = field(default_factory=list)
    history: List[Tuple[str, str, int, Optional[float]]] = field(default_factory=list)


class LiveMonitor:
    def __init__(
        self,
        events_path: Optional[Path] = None,
        prediction_dirs: Optional[List[Path]] = None,
        dataset_base: Optional[Path] = None,
        window_s: float = 300.0,
        bucket_s: float = 10.0,
        clock: Callable[[], float] = time.time,
    ):
        self.tail = EventTail(events_path or default_events_path())
        self.prediction_dirs = prediction_dirs if prediction_dirs is not None else default_prediction_dirs()
        self.dataset_base = Path(dataset_base) if dataset_base is not None else Path("data/classification_prep")
        self.window_s, self.bucket_s, self.clock = window_s, bucket_s, clock
        self.events: List[RequestEvent] = []
        self.dataset = DatasetInfo()
        self.analysis = Analysis()
        self._sig: Optional[Tuple] = None
        self.closed = False
        self.poll_count = 0
        self.refresh_files()

    def _signature(self) -> Tuple:
        sig: List[Any] = []
        for d in self.prediction_dirs:
            if Path(d).exists():
                sig += [(str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in sorted(Path(d).glob("*.jsonl"))]
        if self.dataset.prep_dir is not None:
            lp = self.dataset.prep_dir / "annotations" / "approved_labels.jsonl"
            if lp.exists():
                sig.append((str(lp), lp.stat().st_mtime_ns, lp.stat().st_size))
        return tuple(sig)

    def refresh_files(self) -> None:
        """Re-read dataset status and, only if the files changed, the stored predictions and labels. Read-only."""
        self.dataset = discover_dataset(self.dataset_base)
        sig = self._signature()
        if sig == self._sig:
            return
        self._sig = sig
        a = Analysis()
        a.predictions = A.load_predictions(self.prediction_dirs)
        a.labels = load_approved_labels(self.dataset.prep_dir)
        if a.predictions and self.dataset.prep_dir is not None:
            a.meta = A.load_case_meta(self.dataset.prep_dir)
        a.runs = A.latest_runs(a.predictions)
        a.intent_quality = A.intent_quality(a.runs, a.labels)
        a.calibration = A.calibration_reports(a.runs, a.labels)
        a.lead = A.lead_reports(a.runs, a.labels, a.meta)
        frozen = {r.model_key: r.frozen for r in a.lead if r.frozen is not None}
        a.agreement_intent = A.agreement_report(a.runs, Task.INTENT, a.labels)
        a.agreement_lead = A.agreement_report(a.runs, Task.LEAD, a.labels, frozen)
        a.repeat = A.repeatability(a.predictions)
        a.history = A.history_slices(a.runs, a.labels, a.meta)
        self.analysis = a

    def poll(self) -> int:
        """Read newly appended events. Returns how many were new. Never sends anything; safe to call at any rate."""
        if self.closed:
            return 0
        self.poll_count += 1
        new = self.tail.poll()
        self.events.extend(new)
        self.refresh_files()
        return len(new)

    def trace_inputs(self):
        """(predictions, events, approved labels, frozen thresholds by model) for the Trace Explorer. Read-only views of the loaded sources."""
        a = self.analysis
        return a.predictions, list(self.events), a.labels, {r.model_key: r.frozen for r in a.lead if r.frozen is not None}

    def lead_review(self) -> Tuple[Optional[int], List[str]]:
        """(count, first ids) of prepared lead sessions the summary builder flagged for ambiguity or unresolved rules. Metadata only."""
        if self.dataset.prep_dir is None:
            return None, []
        meta = A.load_case_meta(self.dataset.prep_dir)
        flagged = sorted(c for c, m in meta.items() if m.tier_a is not None and (m.ambiguity_count > 0 or m.unresolved_rules))
        return len(flagged), flagged[:20]

    def aggregate(self) -> Aggregate:
        return aggregate(self.events, self.clock(), self.window_s, self.bucket_s)

    def close(self) -> None:
        self.closed = True
        self.events = []
        self.analysis = Analysis()
