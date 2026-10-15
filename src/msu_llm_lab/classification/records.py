"""Adapter results and persisted prediction records.

Nullable means unavailable: tokens, latency, confidence and cost are never fabricated. A prediction is never ground
truth: records carry no expected label and `is_ground_truth` can only be False.
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from msu_llm_lab.classification.contracts import CONTRACT_VERSION, PARITY_STATUS, IntentLabel
from msu_llm_lab.models.registry import Protocol
from msu_llm_lab.providers.base import scrub_secrets

ConfidenceSource = Literal["NATIVE_DISTRIBUTION", "NATIVE_CONFIDENCE_FIELD", "NONE"]


class Task(str, Enum):
    INTENT = "INTENT"
    LEAD = "LEAD"


class AdapterResult(BaseModel):
    """What one protocol adapter returns for one case. `ok` is False when output or envelope failed validation."""
    model_config = ConfigDict(extra="forbid")

    ok: bool
    task: Task
    protocol: Protocol
    requested_model_id: str
    issues: List[str] = Field(default_factory=list)      # fatal: output or envelope unusable
    warnings: List[str] = Field(default_factory=list)    # non-fatal: a documented field was absent
    resolved_model_id: Optional[str] = None
    serving_provider: Optional[str] = None
    requested_route: Optional[Dict[str, Any]] = None     # routing preferences sent (pin, no fallback); not proof of who served it
    predicted_label: Optional[IntentLabel] = None
    predicted_need_cta: Optional[bool] = None            # Chat lead only; Jev Noul gives yes_prob and a frozen threshold decides
    raw_output: Optional[str] = None
    label_probs: Optional[Dict[str, float]] = None
    native_confidence: Optional[float] = None
    confidence_source: ConfidenceSource = "NONE"
    yes_prob: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    usage_source: Literal["PROVIDER_REPORTED", "PARTIAL", "UNAVAILABLE"] = "UNAVAILABLE"
    latency_ms: Optional[float] = None                   # measured end-to-end, including retry waits
    attempt_latency_ms: Optional[float] = None
    model_latency_ms: Optional[float] = None             # provider-reported only
    attempts: int = Field(default=1, ge=1)
    retry_errors: List[str] = Field(default_factory=list)
    actual_cost_usd: Optional[Decimal] = None
    estimated_cost_usd: Optional[Decimal] = None
    cost_usd: Decimal = Decimal("0")
    cost_basis: Literal["PROVIDER_REPORTED", "ESTIMATED_FROM_CATALOG", "UNAVAILABLE"] = "UNAVAILABLE"
    is_dry_run: bool = False


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    run_id: str
    case_id: str
    task: Task
    model_key: str
    protocol: Protocol
    requested_model_id: str
    resolved_model_id: Optional[str] = None
    serving_provider: Optional[str] = None
    requested_route: Optional[Dict[str, Any]] = None
    prompt_version: str
    input_contract_version: str = CONTRACT_VERSION
    parity_status: str = PARITY_STATUS
    dataset_version: Optional[str] = None
    status: Literal["OK", "INVALID_OUTPUT", "ERROR"]
    validation_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    predicted_label: Optional[IntentLabel] = None
    predicted_need_cta: Optional[bool] = None
    raw_output: Optional[str] = None
    label_probs: Optional[Dict[str, float]] = None
    native_confidence: Optional[float] = None
    confidence_source: ConfidenceSource = "NONE"
    yes_prob: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    usage_source: Literal["PROVIDER_REPORTED", "PARTIAL", "UNAVAILABLE"] = "UNAVAILABLE"
    latency_ms: Optional[float] = None
    attempt_latency_ms: Optional[float] = None
    model_latency_ms: Optional[float] = None
    attempts: Optional[int] = None
    retry_errors: List[str] = Field(default_factory=list)
    actual_cost_usd: Optional[Decimal] = None
    estimated_cost_usd: Optional[Decimal] = None
    recorded_cost_usd: Optional[Decimal] = None
    cost_basis: Literal["PROVIDER_REPORTED", "ESTIMATED_FROM_CATALOG", "UNAVAILABLE"] = "UNAVAILABLE"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    is_dry_run: Literal[False] = False
    is_ground_truth: Literal[False] = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json_line(self) -> str:
        return json.dumps(json.loads(self.model_dump_json()), sort_keys=True, ensure_ascii=False)


def record_from_result(
    result: AdapterResult, *, run_id: str, case_id: str, model_key: str, prompt_version: str,
    dataset_version: Optional[str], store_raw_output: bool, input_contract_version: str = CONTRACT_VERSION,
) -> PredictionRecord:
    return PredictionRecord(
        record_id=f"{run_id}:{case_id}:{result.task.value}:{model_key}", run_id=run_id, case_id=case_id, task=result.task,
        model_key=model_key, protocol=result.protocol, requested_model_id=result.requested_model_id,
        resolved_model_id=result.resolved_model_id, serving_provider=result.serving_provider, requested_route=result.requested_route,
        prompt_version=prompt_version,
        input_contract_version=input_contract_version, dataset_version=dataset_version,
        status="OK" if result.ok else "INVALID_OUTPUT", validation_issues=list(result.issues), warnings=list(result.warnings),
        predicted_label=result.predicted_label, predicted_need_cta=result.predicted_need_cta,
        raw_output=result.raw_output if store_raw_output else None, label_probs=result.label_probs,
        native_confidence=result.native_confidence, confidence_source=result.confidence_source, yes_prob=result.yes_prob,
        prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens, usage_source=result.usage_source,
        latency_ms=result.latency_ms, attempt_latency_ms=result.attempt_latency_ms, model_latency_ms=result.model_latency_ms,
        attempts=result.attempts, retry_errors=list(result.retry_errors), actual_cost_usd=result.actual_cost_usd,
        estimated_cost_usd=result.estimated_cost_usd,
        recorded_cost_usd=result.cost_usd if result.cost_basis != "UNAVAILABLE" else None, cost_basis=result.cost_basis,
    )


def error_record(
    exc: BaseException, *, run_id: str, case_id: str, task: Task, model_key: str, protocol: Protocol,
    requested_model_id: str, prompt_version: str, dataset_version: Optional[str],
    input_contract_version: str = CONTRACT_VERSION,
) -> PredictionRecord:
    """A failed request. Telemetry the failure did not produce stays None; only attempts/errors the exception carried are kept."""
    return PredictionRecord(
        record_id=f"{run_id}:{case_id}:{task.value}:{model_key}", run_id=run_id, case_id=case_id, task=task, model_key=model_key,
        protocol=protocol, requested_model_id=requested_model_id, prompt_version=prompt_version,
        input_contract_version=input_contract_version, dataset_version=dataset_version, status="ERROR",
        attempts=getattr(exc, "attempts", None), retry_errors=list(getattr(exc, "errors", []) or []),
        error_type=type(exc).__name__, error_message=str(scrub_secrets(str(exc)))[:500],
    )


class PredictionStore:
    """Append-only JSONL store. Nulls are written as nulls; a (run_id, case_id) pair is never written twice."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def existing_keys(self) -> Set[Tuple[str, str]]:
        if not self.path.exists():
            return set()
        keys = set()
        for line in self.path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                keys.add((d["run_id"], d["case_id"]))
        return keys

    def append(self, record: PredictionRecord) -> None:
        if (record.run_id, record.case_id) in self.existing_keys():
            raise ValueError(f"record for ({record.run_id}, {record.case_id}) already stored")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json_line() + "\n")

    def load(self) -> List[PredictionRecord]:
        if not self.path.exists():
            return []
        return [PredictionRecord.model_validate_json(l) for l in self.path.read_text().splitlines() if l.strip()]
