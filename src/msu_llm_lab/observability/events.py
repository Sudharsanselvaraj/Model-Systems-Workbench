"""Request lifecycle events: the single authoritative telemetry source for live monitoring.

Events are appended by the classification runner when a real request starts, succeeds, fails (including timeouts) or
retries. Events never contain prompt text, model output or student text. Nullable means unavailable and is never imputed.
Only LIVE events are accepted: there is no simulation mode, so a SIMULATED event is refused on write and quarantined on read.
"""

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

EventKind = Literal["REQUEST_STARTED", "REQUEST_SUCCEEDED", "REQUEST_FAILED", "REQUEST_RETRY"]
TERMINAL_KINDS = ("REQUEST_SUCCEEDED", "REQUEST_FAILED")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


class RequestEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = Field(default_factory=now_iso)
    kind: EventKind
    provenance: Literal["LIVE", "SIMULATED"] = "LIVE"
    request_id: str
    run_id: str
    case_id: str                               # opaque candidate id, never text
    task: Optional[str] = None
    model_key: str
    requested_model_id: Optional[str] = None
    pinned_route: Optional[str] = None
    serving_provider: Optional[str] = None
    resolved_model_id: Optional[str] = None
    status: Optional[Literal["OK", "INVALID_OUTPUT", "ERROR"]] = None
    error_type: Optional[str] = None
    http_status: Optional[int] = None
    timed_out: bool = False
    retry_cause: Optional[str] = None
    attempts: Optional[int] = None
    latency_ms: Optional[float] = None         # measured end-to-end, including retry waits
    model_latency_ms: Optional[float] = None   # provider-reported only
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_basis: Optional[str] = None           # PROVIDER_REPORTED / ESTIMATED_FROM_CATALOG / UNAVAILABLE
    actual_cost_usd: Optional[Decimal] = None
    estimated_cost_usd: Optional[Decimal] = None
    native_cost: Optional[Decimal] = None

    def to_json_line(self) -> str:
        return json.dumps(json.loads(self.model_dump_json()), sort_keys=True, ensure_ascii=False)


def default_events_path() -> Path:
    from msu_llm_lab.core.config import settings
    return settings.data_dir / "telemetry" / "request_events.jsonl"


class EventStore:
    """Append-only JSONL event log. Never rewrites or truncates."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else default_events_path()

    def append(self, event: RequestEvent) -> None:
        if event.provenance != "LIVE":
            raise ValueError("only LIVE events may be recorded; simulated telemetry is not accepted")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(event.to_json_line() + "\n")

    def read_all(self) -> List[RequestEvent]:
        tail = EventTail(self.path)
        return tail.poll()


class EventTail:
    """Incremental reader: remembers its byte offset and the event ids it has delivered, so polling never duplicates.

    Only complete lines are consumed (a half-written final line waits for the next poll). If the file shrinks, the offset
    resets and `truncations` is incremented rather than silently losing track. Polling is read-only.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.seen: Set[str] = set()
        self.malformed = 0
        self.truncations = 0

    def poll(self) -> List[RequestEvent]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            self.offset = 0
            self.truncations += 1
        if size == self.offset:
            return []
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        out: List[RequestEvent] = []
        for raw in chunk[: end + 1].splitlines():
            if not raw.strip():
                continue
            try:
                ev = RequestEvent.model_validate_json(raw)
            except Exception:
                self.malformed += 1
                continue
            if ev.event_id in self.seen:
                continue
            self.seen.add(ev.event_id)
            out.append(ev)
        return out

