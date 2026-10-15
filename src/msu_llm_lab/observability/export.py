"""Trace Explorer data and downloadable exports (JSONL, CSV, ZIP) built from the existing stores.

Sources, all read-only: stored prediction records (the prediction store), the request-event log (timestamps, retries, real-time
failures) and approved labels. No parallel trace store is created. Nothing is invented: a participant that never ran appears only in
the manifest's participant coverage, never as a trace row. Exports are redacted by default, contain no secrets and no `user4`, and
are reproducible: the same selection, sources, schema version and `exported_at` give byte-identical files.
"""

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from msu_llm_lab.classification.contracts import PARITY_STATUS, IntentLabel
from msu_llm_lab.classification.dataprep import redact
from msu_llm_lab.classification.records import PredictionRecord, Task
from msu_llm_lab.observability.analysis import lead_pred
from msu_llm_lab.observability.events import RequestEvent
from msu_llm_lab.observability.labels import ApprovedLabels
from msu_llm_lab.providers.base import scrub_secrets

SCHEMA_VERSION = "trace-export-v1"
LABELS = [l.value for l in IntentLabel]
SMOKE_PREFIX = "smoke-synthetic"
_DROP_KEY = re.compile(r"(authorization|cookie|api[_-]?key|secret|password|bearer|user4)", re.I)
FORMATS = ("jsonl", "csv", "zip")


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawTextAuthorization:
    """Written authorization by the data owner. Without all three fields no raw text is ever exported."""
    owner: str
    scope: str
    authorized_on: str

    def valid(self) -> bool:
        return all(str(x).strip() for x in (self.owner, self.scope, self.authorized_on))


@dataclass(frozen=True)
class ExportSelection:
    """Which trace records to include. None means no restriction on that dimension."""
    models: Optional[Tuple[str, ...]] = None
    runs: Optional[Tuple[str, ...]] = None
    task: Optional[str] = None
    statuses: Optional[Tuple[str, ...]] = None
    case_ids: Optional[Tuple[str, ...]] = None
    error_categories: Optional[Tuple[str, ...]] = None
    data_kinds: Optional[Tuple[str, ...]] = None

    @property
    def scope(self) -> str:
        active = [n for n, v in (("models", self.models), ("runs", self.runs), ("task", self.task), ("statuses", self.statuses), ("case_ids", self.case_ids),
                                 ("error_categories", self.error_categories), ("data_kinds", self.data_kinds)) if v]
        if not active:
            return "ALL_MODELS_ALL_RUNS"
        if active == ["models"] and len(self.models) == 1:
            return "ONE_MODEL"
        if active == ["runs"] and len(self.runs) == 1:
            return "ONE_RUN"
        if active == ["task"]:
            return "ONE_TASK"
        return "FILTERED_SELECTION"

    def matches(self, model: str, run: str, task: str, status: Optional[str], case: str) -> bool:
        return ((self.models is None or model in self.models) and (self.runs is None or run in self.runs) and (self.task is None or task == self.task)
                and (self.statuses is None or status in self.statuses) and (self.case_ids is None or case in self.case_ids))

    def matches_row(self, row: Dict[str, Any]) -> bool:
        """Filters that need the built row: the error class and whether the record is synthetic or real-data."""
        return ((self.error_categories is None or row["request"]["error_category"] in self.error_categories)
                and (self.data_kinds is None or row["markers"]["data_kind"] in self.data_kinds))

    def as_dict(self) -> Dict[str, Any]:
        return {"scope": self.scope, "models": list(self.models) if self.models else None, "runs": list(self.runs) if self.runs else None,
                "task": self.task, "statuses": list(self.statuses) if self.statuses else None, "case_ids": list(self.case_ids) if self.case_ids else None,
                "error_categories": list(self.error_categories) if self.error_categories else None, "data_kinds": list(self.data_kinds) if self.data_kinds else None}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:10]


# ------------------------------------------------------------------ row building
def _s(v: Any) -> Any:
    return str(v) if isinstance(v, Decimal) else v


def data_kind(dataset_version: Optional[str], run_id: str) -> str:
    """Never let a synthetic fixture pass for production or real data: the kind is derived from the dataset version and run id only."""
    if (dataset_version or "").startswith(SMOKE_PREFIX) or run_id.startswith("smoke_"):
        return "SYNTHETIC_FIXTURE"
    if run_id.startswith("bench-"):
        return "REAL_DATA_BENCHMARK"
    return "UNCLASSIFIED"


def _error_category(rec: PredictionRecord, ev: Optional[RequestEvent]) -> Optional[str]:
    if rec.status == "OK":
        return None
    if rec.status == "INVALID_OUTPUT":
        return "INVALID_OUTPUT"
    et = rec.error_type or ""
    errs = " ".join(rec.retry_errors)
    if (ev and ev.timed_out) or "TIMEOUT" in errs:
        return "TIMEOUT"
    code = ev.http_status if ev else None
    if code == 429 or "HTTP_429" in errs:
        return "RATE_LIMIT"
    if code in (401, 403):
        return "AUTH"
    if code == 402:
        return "PAYMENT_OR_BUDGET"
    if (code or 0) >= 500 or "HTTP_5" in errs:
        return "PROVIDER_5XX"
    if et == "UsageUnavailableError":
        return "USAGE_UNAVAILABLE"
    if et in ("LiveNotAuthorizedError", "BudgetExceededError"):
        return "GUARD_REFUSED"
    return "OTHER_ERROR"


def _unavailable(rec: PredictionRecord, ev: Optional[RequestEvent]) -> List[str]:
    miss = []
    if rec.latency_ms is None:
        miss.append("latency_ms")
    if rec.model_latency_ms is None:
        miss.append("model_latency_ms")
    if rec.prompt_tokens is None or rec.completion_tokens is None:
        miss.append("tokens")
    if rec.actual_cost_usd is None:
        miss.append("actual_cost")
    if rec.estimated_cost_usd is None:
        miss.append("estimated_cost")
    if rec.status != "ERROR" and rec.confidence_source == "NONE":
        miss.append("native_confidence")
    if rec.status != "ERROR" and rec.finish_reason is None:
        miss.append("finish_reason")
    if ev is None:
        miss.append("request_started_at")
    return miss


def _truth(rec: PredictionRecord, labels: ApprovedLabels, decision: Optional[bool]) -> Dict[str, Any]:
    if rec.task == Task.INTENT:
        lab = labels.intent.get(rec.case_id)
        got = lab.value if lab else None
        correct = (rec.predicted_label.value == got if rec.predicted_label else False) if lab else None
    else:
        lab = labels.lead.get(rec.case_id)
        got = lab
        correct = (decision is lab) if (lab is not None and decision is not None) else (False if lab is not None else None)
    status = "APPROVED" if got is not None else ("NOT_LABELLED" if labels.available else "PENDING")
    return {"status": status, "label": got, "correct": correct}


def row_from_record(rec: PredictionRecord, started: Optional[RequestEvent], retries: List[str], labels: ApprovedLabels,
                    frozen_by_model: Dict[str, Any], authorization: Optional[RawTextAuthorization]) -> Dict[str, Any]:
    request_id = f"{rec.run_id}:{rec.case_id}:{rec.task.value}:{rec.model_key}"
    frozen = frozen_by_model.get(rec.model_key)
    decision = lead_pred(rec, frozen) if rec.task == Task.LEAD else None
    if rec.task == Task.LEAD:
        source = "MODEL_OUTPUT" if rec.predicted_need_cta is not None else ("FROZEN_DEV_THRESHOLD" if (frozen is not None and rec.yes_prob is not None) else
                                                                           ("THRESHOLD_NOT_FROZEN" if rec.yes_prob is not None else None))
    else:
        source = "MODEL_OUTPUT" if rec.predicted_label else None
    sel = rec.label_probs.get(rec.predicted_label.value) if (rec.label_probs and rec.predicted_label) else None
    route = rec.requested_route or {}
    pinned = (route.get("order") or [None])[0] if isinstance(route, dict) else None
    return {
        "schema_version": SCHEMA_VERSION, "record_type": "REQUEST_TRACE",
        "identity": {"run_id": rec.run_id, "case_id": rec.case_id, "task": rec.task.value, "model_key": rec.model_key, "requested_model_id": rec.requested_model_id,
                     "resolved_model_id": rec.resolved_model_id, "serving_provider": rec.serving_provider, "requested_route": rec.requested_route,
                     "pinned_route": pinned, "protocol": rec.protocol.value},
        "versions": {"dataset_version": rec.dataset_version, "prompt_fingerprint": rec.prompt_version, "input_contract_version": rec.input_contract_version,
                     "parity_status": rec.parity_status},
        "request": {"request_id": request_id, "request_started_at": started.ts if started else None, "record_created_at": rec.created_at,
                    "latency_ms_end_to_end": rec.latency_ms, "model_latency_ms": rec.model_latency_ms,
                    "model_latency_source": "PROVIDER_REPORTED" if rec.model_latency_ms is not None else "UNAVAILABLE", "finish_reason": rec.finish_reason,
                    "attempts": rec.attempts, "retry_causes": list(retries or rec.retry_errors), "status": rec.status,
                    "error_category": _error_category(rec, None), "error_type": rec.error_type, "error_message": rec.error_message,
                    "validation_issues": list(rec.validation_issues), "warnings": list(rec.warnings)},
        "output": {"predicted_label": rec.predicted_label.value if rec.predicted_label else None, "predicted_need_cta": rec.predicted_need_cta,
                   "decision": decision, "decision_source": source,
                   "raw_output": rec.raw_output if (authorization and authorization.valid()) else None, "raw_output_present_in_store": rec.raw_output is not None},
        "confidence": {"confidence_source": rec.confidence_source, "label_probabilities": rec.label_probs, "selected_label_probability": sel,
                       "native_confidence": rec.native_confidence, "yes_probability": rec.yes_prob,
                       "frozen_threshold": frozen.threshold if (frozen is not None and rec.task == Task.LEAD) else None},
        "usage": {"prompt_tokens": rec.prompt_tokens, "completion_tokens": rec.completion_tokens, "usage_source": rec.usage_source, "reasoning_tokens": rec.reasoning_tokens},
        "cost": {"actual_cost_usd": _s(rec.actual_cost_usd), "estimated_cost_usd": _s(rec.estimated_cost_usd), "native_cost": _s(rec.native_cost),
                 "native_cost_unit": rec.native_cost_unit, "recorded_cost_usd": _s(rec.recorded_cost_usd), "cost_basis": rec.cost_basis,
                 "unresolved_spend": rec.cost_basis == "UNAVAILABLE" and rec.status != "ERROR"},
        "ground_truth": _truth(rec, labels, decision),
        "markers": {"provenance": "LIVE", "simulated": False, "failed_call": rec.status == "ERROR", "invalid_output": rec.status == "INVALID_OUTPUT",
                    "synthetic_fixture": (rec.dataset_version or "").startswith(SMOKE_PREFIX), "data_kind": data_kind(rec.dataset_version, rec.run_id),
                    "source": "PREDICTION_RECORD", "origin": None,
                    "binary_decision_available": (decision is not None) if rec.task == Task.LEAD else None,
                    "probability_only": (rec.task == Task.LEAD and decision is None and rec.yes_prob is not None),
                    "unavailable_telemetry": _unavailable(rec, started)},
    }


def row_from_event(ev: RequestEvent, started: Optional[RequestEvent], retries: List[str]) -> Dict[str, Any]:
    """A finished request that has no stored prediction record (for example an aborted run). Only what the events state."""
    return {
        "schema_version": SCHEMA_VERSION, "record_type": "REQUEST_TRACE",
        "identity": {"run_id": ev.run_id, "case_id": ev.case_id, "task": ev.task, "model_key": ev.model_key, "requested_model_id": ev.requested_model_id,
                     "resolved_model_id": ev.resolved_model_id, "serving_provider": ev.serving_provider, "requested_route": None, "pinned_route": ev.pinned_route, "protocol": None},
        "versions": {"dataset_version": None, "prompt_fingerprint": None, "input_contract_version": None, "parity_status": PARITY_STATUS},
        "request": {"request_id": ev.request_id, "request_started_at": started.ts if started else None, "record_created_at": None,
                    "latency_ms_end_to_end": ev.latency_ms, "model_latency_ms": ev.model_latency_ms,
                    "model_latency_source": "PROVIDER_REPORTED" if ev.model_latency_ms is not None else "UNAVAILABLE", "finish_reason": None, "attempts": ev.attempts,
                    "retry_causes": retries, "status": ev.status, "error_category": ("TIMEOUT" if ev.timed_out else "RATE_LIMIT" if ev.http_status == 429 else
                                                                                  "PROVIDER_5XX" if (ev.http_status or 0) >= 500 else "OTHER_ERROR") if ev.status == "ERROR" else None,
                    "error_type": ev.error_type, "error_message": None, "validation_issues": [], "warnings": []},
        "output": {"predicted_label": None, "predicted_need_cta": None, "decision": None, "decision_source": None, "raw_output": None, "raw_output_present_in_store": False},
        "confidence": {"confidence_source": "NONE", "label_probabilities": None, "selected_label_probability": None, "native_confidence": None,
                       "yes_probability": None, "frozen_threshold": None},
        "usage": {"prompt_tokens": ev.prompt_tokens, "completion_tokens": ev.completion_tokens, "usage_source": "UNAVAILABLE" if ev.prompt_tokens is None else "PROVIDER_REPORTED", "reasoning_tokens": None},
        "cost": {"actual_cost_usd": _s(ev.actual_cost_usd), "estimated_cost_usd": _s(ev.estimated_cost_usd), "native_cost": _s(ev.native_cost), "native_cost_unit": None,
                 "recorded_cost_usd": None, "cost_basis": ev.cost_basis or "UNAVAILABLE", "unresolved_spend": (ev.cost_basis == "UNAVAILABLE" and ev.status != "ERROR")},
        "ground_truth": {"status": "PENDING", "label": None, "correct": None},
        "markers": {"provenance": "LIVE", "simulated": False, "failed_call": ev.status == "ERROR", "invalid_output": ev.status == "INVALID_OUTPUT", "synthetic_fixture": ev.run_id.startswith("smoke_"),
                    "data_kind": data_kind(None, ev.run_id), "source": "EVENT_ONLY_NO_PREDICTION_RECORD", "origin": None, "binary_decision_available": None, "probability_only": False, "unavailable_telemetry": ["prediction_record"]},
    }


_FREE_TEXT_KEYS = frozenset({"error_message", "validation_issues", "warnings", "raw_output", "retry_causes"})


def _clean(obj: Any, counts: Counter, free_text: bool = False) -> Any:
    """Drop secret/user4 keys and scrub secret-looking strings everywhere; redact personal identifiers in free-text fields.

    Identifier fields (run ids, case ids, timestamps, fingerprints) are opaque and are not pattern-redacted, because a digit run in
    them is not personal data and redacting it would corrupt the join keys.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _DROP_KEY.search(str(k)):
                counts["dropped_keys"] += 1
                continue
            out[k] = _clean(v, counts, free_text or k in _FREE_TEXT_KEYS)
        return out
    if isinstance(obj, list):
        return [_clean(v, counts, free_text) for v in obj]
    if isinstance(obj, str):
        s = scrub_secrets(obj)
        if s != obj:
            counts["secrets_scrubbed"] += 1
        if free_text:
            s, c = redact(s)
            counts.update({f"redacted{t}": n for t, n in c.items()})
        return s
    return obj


@dataclass
class TraceSet:
    rows: List[Dict[str, Any]]
    skipped: Counter = field(default_factory=Counter)
    redactions: Counter = field(default_factory=Counter)
    records_redacted: int = 0
    quarantined_simulated: int = 0
    raw_text_withheld: int = 0


def build_trace_rows(
    predictions: Sequence[PredictionRecord], events: Iterable[RequestEvent], labels: ApprovedLabels, frozen_by_model: Dict[str, Any],
    selection: ExportSelection = ExportSelection(), authorization: Optional[RawTextAuthorization] = None, include_event_only: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> TraceSet:
    """Join records with events, apply the selection, then clean. Sources are only read."""
    ts = TraceSet(rows=[])
    started: Dict[str, RequestEvent] = {}
    terminal: Dict[str, RequestEvent] = {}
    retries: Dict[str, List[str]] = {}
    for ev in events:
        if ev.provenance != "LIVE":
            ts.quarantined_simulated += 1
            ts.skipped["SIMULATED_EXCLUDED"] += 1
            continue
        if ev.kind == "REQUEST_STARTED":
            started.setdefault(ev.request_id, ev)
        elif ev.kind == "REQUEST_RETRY":
            retries.setdefault(ev.request_id, []).append(ev.retry_cause or "UNKNOWN")
        else:
            terminal[ev.request_id] = ev
    raw: List[Dict[str, Any]] = []
    seen_ids = set()
    for rec in sorted(predictions, key=lambda r: (r.run_id, r.model_key, r.task.value, r.case_id)):
        rid = f"{rec.run_id}:{rec.case_id}:{rec.task.value}:{rec.model_key}"
        seen_ids.add(rid)
        if not selection.matches(rec.model_key, rec.run_id, rec.task.value, rec.status, rec.case_id):
            ts.skipped["NOT_SELECTED"] += 1
            continue
        if rec.raw_output is not None and not (authorization and authorization.valid()):
            ts.raw_text_withheld += 1
        row = row_from_record(rec, started.get(rid), retries.get(rid, []), labels, frozen_by_model, authorization)
        te = terminal.get(rid)
        if row["request"]["error_category"] is not None and te is not None:
            row["request"]["error_category"] = _error_category(rec, te)
        raw.append(row)
    if include_event_only:
        for rid, ev in sorted(terminal.items()):
            if rid in seen_ids:
                continue
            if not selection.matches(ev.model_key, ev.run_id, ev.task or "", ev.status, ev.case_id):
                ts.skipped["NOT_SELECTED"] += 1
                continue
            raw.append(row_from_event(ev, started.get(rid), retries.get(rid, [])))
    total = len(raw)
    for i, row in enumerate(raw, 1):
        if not selection.matches_row(row):
            ts.skipped["NOT_SELECTED"] += 1
            continue
        before = Counter()
        cleaned = _clean(row, before)
        replaced = sum(n for k, n in before.items() if k.startswith("redacted"))
        if replaced or before.get("secrets_scrubbed"):
            ts.records_redacted += 1
        cleaned["markers"]["redaction"] = {"default_redaction_applied": True, "replacements_in_free_text": replaced, "secrets_scrubbed": before.get("secrets_scrubbed", 0),
                                           "raw_student_text_in_record": False, "pattern_based_only": True}
        ts.redactions.update(before)
        ts.rows.append(cleaned)
        if progress:
            progress(i, total)
    ts.rows.sort(key=lambda r: (r["identity"]["run_id"], r["identity"]["model_key"], r["identity"]["task"] or "", r["identity"]["case_id"]))
    return ts


# ------------------------------------------------------------------ flattening and files
CSV_COLUMNS = [
    "schema_version", "run_id", "case_id", "task", "model_key", "requested_model_id", "resolved_model_id", "serving_provider", "pinned_route", "requested_route_json",
    "protocol", "dataset_version", "prompt_fingerprint", "input_contract_version", "parity_status", "request_id", "request_started_at", "record_created_at",
    "latency_ms_end_to_end", "model_latency_ms", "model_latency_source", "attempts", "retry_causes", "status", "error_category", "error_type", "error_message",
    "validation_issues", "warnings", "predicted_label", "predicted_need_cta", "decision", "decision_source", "raw_output",
    "confidence_source", *[f"p_{l}" for l in LABELS], "selected_label_probability", "native_confidence", "yes_probability", "frozen_threshold",
    "prompt_tokens", "completion_tokens", "usage_source", "reasoning_tokens", "finish_reason", "actual_cost_usd", "estimated_cost_usd", "native_cost", "native_cost_unit", "recorded_cost_usd", "cost_basis",
    "unresolved_spend", "ground_truth_status", "ground_truth_label", "correct", "provenance", "simulated", "failed_call", "invalid_output", "synthetic_fixture",
    "data_kind", "origin", "binary_decision_available", "probability_only", "redaction_replacements", "record_source", "unavailable_telemetry",
]


def flatten(row: Dict[str, Any]) -> Dict[str, Any]:
    i, v, r, o, c, u, k, g, m = (row[x] for x in ("identity", "versions", "request", "output", "confidence", "usage", "cost", "ground_truth", "markers"))
    probs = c.get("label_probabilities") or {}
    flat = {
        "schema_version": row["schema_version"], "run_id": i["run_id"], "case_id": i["case_id"], "task": i["task"], "model_key": i["model_key"],
        "requested_model_id": i["requested_model_id"], "resolved_model_id": i["resolved_model_id"], "serving_provider": i["serving_provider"],
        "pinned_route": i["pinned_route"], "requested_route_json": json.dumps(i["requested_route"], sort_keys=True) if i["requested_route"] is not None else None,
        "protocol": i["protocol"], "dataset_version": v["dataset_version"], "prompt_fingerprint": v["prompt_fingerprint"],
        "input_contract_version": v["input_contract_version"], "parity_status": v["parity_status"], "request_id": r["request_id"],
        "request_started_at": r["request_started_at"], "record_created_at": r["record_created_at"], "latency_ms_end_to_end": r["latency_ms_end_to_end"],
        "model_latency_ms": r["model_latency_ms"], "model_latency_source": r["model_latency_source"], "attempts": r["attempts"],
        "retry_causes": "|".join(r["retry_causes"]), "status": r["status"], "error_category": r["error_category"], "error_type": r["error_type"],
        "error_message": r["error_message"], "validation_issues": "|".join(r["validation_issues"]), "warnings": "|".join(r["warnings"]),
        "predicted_label": o["predicted_label"], "predicted_need_cta": o["predicted_need_cta"], "decision": o["decision"], "decision_source": o["decision_source"],
        "raw_output": o["raw_output"], "confidence_source": c["confidence_source"], "selected_label_probability": c["selected_label_probability"],
        "native_confidence": c["native_confidence"], "yes_probability": c["yes_probability"], "frozen_threshold": c["frozen_threshold"],
        "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"], "usage_source": u["usage_source"], "reasoning_tokens": u.get("reasoning_tokens"), "finish_reason": r.get("finish_reason"),
        "actual_cost_usd": k["actual_cost_usd"], "estimated_cost_usd": k["estimated_cost_usd"], "native_cost": k["native_cost"], "native_cost_unit": k["native_cost_unit"],
        "recorded_cost_usd": k["recorded_cost_usd"], "cost_basis": k["cost_basis"], "unresolved_spend": k["unresolved_spend"],
        "ground_truth_status": g["status"], "ground_truth_label": g["label"], "correct": g["correct"], "provenance": m["provenance"], "simulated": m["simulated"],
        "failed_call": m["failed_call"], "invalid_output": m["invalid_output"], "synthetic_fixture": m["synthetic_fixture"], "data_kind": m["data_kind"], "origin": m.get("origin"), "binary_decision_available": m.get("binary_decision_available"), "probability_only": m.get("probability_only"),
        "redaction_replacements": (m.get("redaction") or {}).get("replacements_in_free_text"), "record_source": m["source"],
        "unavailable_telemetry": "|".join(m["unavailable_telemetry"]),
    }
    for l in LABELS:
        flat[f"p_{l}"] = probs.get(l)
    return flat


def jsonl_bytes(rows: Iterable[Dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def csv_bytes(rows: Iterable[Dict[str, Any]], columns: Sequence[str] = CSV_COLUMNS, flat: bool = False) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        f = r if flat else flatten(r)
        w.writerow({k: ("" if f.get(k) is None else f.get(k)) for k in columns})   # an empty cell means null/unavailable, never a substituted value
    return buf.getvalue().encode("utf-8")


# ------------------------------------------------------------------ participants and comparison
def participant_coverage(roster: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every participant is listed, distinguishing models with real requests from models never executed. No trace rows are made for the latter."""
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["identity"]["model_key"], []).append(r)
    out = []
    known = set()
    for p in roster:
        keys = {p["key"], p["model_id"]}
        known |= keys
        rs = [r for k in keys for r in by_model.get(k, [])]
        ok = sum(r["request"]["status"] == "OK" for r in rs)
        failed = sum(r["markers"]["failed_call"] for r in rs)
        if not rs:
            state = "NOT_EXECUTED_BLOCKED" if p["readiness"] == "BLOCKED" else "NOT_EXECUTED"
        elif ok == 0:
            state = "EXECUTED_ALL_FAILED_OR_INVALID"
        elif ok < len(rs):
            state = "EXECUTED_WITH_FAILURES"
        else:
            state = "EXECUTED"
        out.append({"key": p["key"], "display_name": p["display_name"], "model_id": p["model_id"], "roles": p["roles"], "tasks": p["tasks"], "readiness": p["readiness"],
                    "blockers": p["blockers"], "pinned_route": p["pinned_route"], "execution_state": state, "n_trace_records": len(rs), "n_ok": ok, "n_failed_calls": failed,
                    "runs": sorted({r["identity"]["run_id"] for r in rs})})
    for mk in sorted(set(by_model) - known):  # executed models that are not on the roster (for example test or legacy ids)
        rs = by_model[mk]
        out.append({"key": mk, "display_name": mk, "model_id": mk, "roles": [], "tasks": sorted({r["identity"]["task"] for r in rs}), "readiness": "NOT_ON_ROSTER", "blockers": [],
                    "pinned_route": None, "execution_state": "EXECUTED", "n_trace_records": len(rs), "n_ok": sum(r["request"]["status"] == "OK" for r in rs),
                    "n_failed_calls": sum(r["markers"]["failed_call"] for r in rs), "runs": sorted({r["identity"]["run_id"] for r in rs})})
    return out


def comparison_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """One line per (run family, task, case) with each model's prediction side by side. Cells for models that did not answer stay empty."""
    models = sorted({r["identity"]["model_key"] for r in rows})
    cases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (r["identity"]["task"] or "", r["identity"]["case_id"])
        d = cases.setdefault(key, {"task": key[0], "case_id": key[1], "ground_truth_label": r["ground_truth"]["label"], "ground_truth_status": r["ground_truth"]["status"]})
        o = r["output"]
        ok = r["request"]["status"] == "OK"
        val = None if not ok else (o["predicted_label"] if o["predicted_label"] is not None else (None if o["decision"] is None else ("YES" if o["decision"] else "NO")))
        # Only a valid output is shown as a prediction. An error, an invalid output or an undecided P(yes) leaves the cell empty and is counted apart.
        d[f"pred::{r['identity']['model_key']}"] = val
        if val is None:
            d["n_models_unusable"] = d.get("n_models_unusable", 0) + 1
    out = []
    for key in sorted(cases):
        d = cases[key]
        preds = [d.get(f"pred::{m}") for m in models if d.get(f"pred::{m}") is not None]
        d["n_models_answered"] = len(preds)
        d.setdefault("n_models_unusable", 0)
        d["models_agree"] = (len(set(preds)) == 1) if len(preds) > 1 else None   # agreement only; never correctness
        out.append(d)
    cols = ["task", "case_id", "ground_truth_status", "ground_truth_label", "n_models_answered", "n_models_unusable", "models_agree"] + [f"pred::{m}" for m in models]
    return cols, out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# ------------------------------------------------------------------ export
@dataclass
class ExportResult:
    path: Path
    format: str
    manifest: Dict[str, Any]
    n_exported: int
    n_skipped: int
    n_redacted: int
    n_missing_telemetry: int
    warnings: List[str]


def _manifest(ts: TraceSet, sel: ExportSelection, roster, labels: ApprovedLabels, exported_at: str, fmt: str, files: Dict[str, str],
              authorization: Optional[RawTextAuthorization]) -> Dict[str, Any]:
    rows = ts.rows
    routes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = routes.setdefault(r["identity"]["model_key"], {"pinned_routes": set(), "serving_providers": set(), "resolved_models": set()})
        if r["identity"]["pinned_route"]:
            d["pinned_routes"].add(r["identity"]["pinned_route"])
        if r["identity"]["serving_provider"]:
            d["serving_providers"].add(r["identity"]["serving_provider"])
        if r["identity"]["resolved_model_id"]:
            d["resolved_models"].add(r["identity"]["resolved_model_id"])
    missing = sum(1 for r in rows if r["markers"]["unavailable_telemetry"])
    warnings = []
    if not labels.available:
        warnings.append("GROUND TRUTH PENDING: no human-approved labels; correctness fields are null and no supervised metric can be derived from this export")
    if any(r["versions"]["parity_status"] == PARITY_STATUS for r in rows) or not rows:
        warnings.append("PRODUCTION_PARITY_NOT_ESTABLISHED: these traces are not claimed to match the production workflow")
    if any(r["markers"]["synthetic_fixture"] for r in rows):
        warnings.append("Some records are SYNTHETIC smoke-test fixtures, not benchmark data")
    if not rows:
        warnings.append("EMPTY EXPORT: no trace records matched the selection (no placeholder rows are created)")
    if ts.raw_text_withheld:
        warnings.append(f"{ts.raw_text_withheld} stored raw output(s) withheld (no data-owner authorization)")
    return {
        "schema_version": SCHEMA_VERSION, "exported_at": exported_at, "format": fmt, "selection": sel.as_dict(),
        "run_ids": sorted({r["identity"]["run_id"] for r in rows}), "model_ids": sorted({r["identity"]["model_key"] for r in rows}),
        "provider_routes": {m: {k: sorted(v) for k, v in d.items()} for m, d in sorted(routes.items())},
        "tasks": sorted({r["identity"]["task"] for r in rows if r["identity"]["task"]}),
        "dataset_versions": sorted({r["versions"]["dataset_version"] for r in rows if r["versions"]["dataset_version"]}),
        "prompt_versions": sorted({r["versions"]["prompt_fingerprint"] for r in rows if r["versions"]["prompt_fingerprint"]}),
        "input_contract_versions": sorted({r["versions"]["input_contract_version"] for r in rows if r["versions"]["input_contract_version"]}),
        "parity_status": PARITY_STATUS,
        "telemetry_provenance": {"sources": ["stored prediction records", "request event log"], "live_only": True, "simulated_records_included": False,
                                 "simulated_events_excluded": ts.quarantined_simulated, "event_only_records": sum(r["markers"]["source"] != "PREDICTION_RECORD" for r in rows)},
        "counts": {"exported": len(rows), "skipped": dict(sorted(ts.skipped.items())), "skipped_total": sum(ts.skipped.values()), "records_redacted": ts.records_redacted,
                   "redaction_events": dict(sorted(ts.redactions.items())), "records_missing_telemetry": missing,
                   "failed_calls": sum(r["markers"]["failed_call"] for r in rows), "invalid_outputs": sum(r["markers"]["invalid_output"] for r in rows)},
        "ground_truth": {"status": labels.status, "approved_labels": labels.n_approved},
        "privacy": {"redaction": "pattern-based (email, ID, phone, introduced names); no named-entity recognition", "secrets_scrubbed": True, "user4_excluded": True,
                    "raw_text_included": bool(authorization and authorization.valid()),
                    "raw_text_authorization": ({"owner": authorization.owner, "scope": authorization.scope, "authorized_on": authorization.authorized_on}
                                               if authorization and authorization.valid() else None),
                    "this_export_transmitted_externally": False,
                    "note": "This describes the export files only. Whether inference payloads were sent to providers is recorded in the benchmark manifest, not here."},
        "participants": participant_coverage(roster, rows), "files": files, "warnings": warnings,
    }


def export_traces(
    ts: TraceSet, selection: ExportSelection, roster: Sequence[Dict[str, Any]], labels: ApprovedLabels, fmt: str, out_dir: Path,
    exported_at: Optional[str] = None, authorization: Optional[RawTextAuthorization] = None,
) -> ExportResult:
    """Write one export atomically to `out_dir` and return its exact path. Never overwrites: an existing name gets a numeric suffix."""
    if fmt not in FORMATS:
        raise ExportError(f"unknown format '{fmt}' (choose from {', '.join(FORMATS)})")
    if authorization is not None and not authorization.valid():
        raise ExportError("raw-text export refused: the data-owner authorization is incomplete")
    exported_at = exported_at or datetime.now(timezone.utc).isoformat()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trace_export_{selection.scope.lower()}_{selection.fingerprint()}"
    files: Dict[str, bytes] = {}
    if fmt == "jsonl":
        files[f"{stem}.jsonl"] = jsonl_bytes(ts.rows)
    elif fmt == "csv":
        files[f"{stem}.csv"] = csv_bytes(ts.rows)
    else:
        files["all_records.jsonl"] = jsonl_bytes(ts.rows)
        files["all_records.csv"] = csv_bytes(ts.rows)
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for r in ts.rows:
            by_model.setdefault(r["identity"]["model_key"], []).append(r)
        for m, rs in sorted(by_model.items()):
            files[f"models/{_safe(m)}.jsonl"] = jsonl_bytes(rs)
            files[f"models/{_safe(m)}.csv"] = csv_bytes(rs)
        cols, comp = comparison_rows(ts.rows)
        files["consolidated_comparison.csv"] = csv_bytes(comp, cols, flat=True)
        pcols = ["key", "display_name", "model_id", "execution_state", "readiness", "n_trace_records", "n_ok", "n_failed_calls", "pinned_route", "blockers", "runs"]
        part = [{**p, "blockers": "|".join(p["blockers"]), "runs": "|".join(p["runs"])} for p in participant_coverage(roster, ts.rows)]
        files["participants.csv"] = csv_bytes(part, pcols, flat=True)
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())}
    manifest = _manifest(ts, selection, roster, labels, exported_at, fmt, hashes, authorization)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    ext = "zip" if fmt == "zip" else fmt
    target = out_dir / f"{stem}.{ext}"
    n = 1
    while target.exists():
        n += 1
        target = out_dir / f"{stem}-r{n}.{ext}"
    tmp = target.with_name(target.name + ".part")
    try:
        if fmt == "zip":
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in sorted(files.items()) + [("manifest.json", manifest_bytes)]:
                    zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))   # fixed timestamp: identical input gives identical bytes
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zi.external_attr = 0o644 << 16
                    zf.writestr(zi, data)
        else:
            tmp.write_bytes(next(iter(files.values())))
            manifest_path = target.with_name(target.name + ".manifest.json")
            if manifest_path.exists():
                raise ExportError(f"refusing to overwrite {manifest_path}")
            manifest_path.write_bytes(manifest_bytes)
        tmp.replace(target)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    c = manifest["counts"]
    return ExportResult(target, fmt, manifest, c["exported"], c["skipped_total"], c["records_redacted"], c["records_missing_telemetry"], list(manifest["warnings"]))


def load_bench_plans(root: Path) -> Dict[str, Dict[str, Any]]:
    """Stored benchmark plans by bench id (local files, read-only)."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(Path(root).glob("bench-*/plan.json")) if Path(root).exists() else []:
        try:
            out[p.parent.name] = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return out


def run_summaries(rows: Sequence[Dict[str, Any]], plans: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Per run id: planned (full dataset, when a stored plan covers it), attempted, succeeded, failed, invalid and unattempted. Unknowns stay None."""
    plans = plans or {}
    agg: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        i = r["identity"]
        d = agg.setdefault(i["run_id"], {"run_id": i["run_id"], "model_key": i["model_key"], "task": i["task"], "data_kind": r["markers"]["data_kind"],
                                         "attempted": 0, "succeeded": 0, "failed": 0, "invalid": 0})
        d["attempted"] += 1
        d["succeeded"] += r["request"]["status"] == "OK"
        d["failed"] += bool(r["markers"]["failed_call"])
        d["invalid"] += bool(r["markers"]["invalid_output"])
    for rid, d in agg.items():
        d["planned_full_dataset"] = None
        for bid, plan in plans.items():
            if rid.startswith(bid + "-"):
                for p in plan["participants"]:
                    if rid == f"{bid}-{p['key']}-{d['task']}" and d["task"] in p["per_task"]:
                        d["planned_full_dataset"] = p["per_task"][d["task"]]["requests_full"]
        d["unattempted"] = None if d["planned_full_dataset"] is None else max(0, d["planned_full_dataset"] - d["attempted"])
    return [agg[k] for k in sorted(agg)]
