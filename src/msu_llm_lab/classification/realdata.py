"""Operator authorization to transmit REAL admissions-derived records to model providers, and the run manifest that records it.

What this is: an auditable, versioned statement by the data owner that a specific, reviewed dataset may be used for a specific
benchmark. It is bound to the dataset preparation version, the exact models and routes, the task scope, the plan fingerprint and the
cost ceiling. What it is not: verification of any provider's retention, training or data-processing policy. Those stay UNKNOWN in the
registry and in the record, whatever the operator authorizes.

An authorization object can only be created by `confirm_real_data` (typed phrase, interactive terminal) or loaded back from a manifest
for inspection (a loaded one is never accepted by the runner unless it was created in the current process).
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from msu_llm_lab.providers.base import LiveNotAuthorizedError

AUTH_SCHEMA = "real-data-authorization-v1"
CONFIRM_PREFIX = "AUTHORIZE REAL DATA"
POLICY_NOTE = ("The operator's authorization does not verify any provider's retention, training or data-processing policy; "
               "every route's policy status is recorded as it is in the registry (UNKNOWN unless a primary source verified it).")
DATA_CATEGORIES = {
    "INTENT": ["redacted student messages (current message and prior turns of the same session)", "prior assistant replies as exported (redacted)",
               "the authoritative INTENT prompt text"],
    "LEAD": ["a rule-built summary of student actions (no transcript)", "the authoritative LEAD prompt text"],
}
NOT_TRANSMITTED = ["pseudonymous group or trace ids", "timestamps", "user4 (never read)", "excluded records", "API keys or credentials", "unredacted direct identifiers (pattern-based redaction only; no named-entity recognition)"]


class RealDataAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = AUTH_SCHEMA
    dataset_version: str
    session_policy: str
    models: Tuple[str, ...]
    routes: Dict[str, str]
    tasks: Tuple[str, ...]
    plan_fingerprint: str
    cost_ceiling_usd: str
    data_classification: str = "CONFIDENTIAL"
    real_admissions_derived: bool = True
    operator: str
    statement: str
    authorized_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provider_policy_status: Dict[str, str]
    policy_note: str = POLICY_NOTE

    def covers(self, *, dataset_version: Optional[str], model_id: str, task: str, plan_fingerprint: Optional[str] = None) -> bool:
        return (dataset_version == self.dataset_version and model_id in self.models and task in self.tasks
                and (plan_fingerprint is None or plan_fingerprint == self.plan_fingerprint))


_ISSUED: set = set()   # ids of authorizations created by confirm_real_data in this process


def is_issued(auth: Optional[RealDataAuthorization]) -> bool:
    return auth is not None and id(auth) in _ISSUED


def expected_phrase(dataset_version: str, plan_fingerprint: str) -> str:
    return f"{CONFIRM_PREFIX} {dataset_version} PLAN {plan_fingerprint}"


def confirm_real_data(
    *, dataset_version: str, session_policy: str, models: Sequence[str], routes: Dict[str, str], tasks: Sequence[str], plan_fingerprint: str,
    cost_ceiling_usd: str, provider_policy_status: Dict[str, str], recipients: Dict[str, str], operator: str, summary_lines: Sequence[str] = (),
    is_interactive: Optional[Callable[[], bool]] = None, prompt: Optional[Callable[[str], str]] = None, echo: Callable[[str], None] = print,
) -> RealDataAuthorization:
    """Show the recipients and data categories, then require the exact typed phrase from an interactive terminal."""
    interactive = is_interactive if is_interactive is not None else (lambda: sys.stdin.isatty() and sys.stdout.isatty())
    if not interactive():
        raise LiveNotAuthorizedError("Real-data authorization needs an interactive terminal; non-interactive execution is refused.")
    if not operator.strip():
        raise LiveNotAuthorizedError("Real-data authorization needs a named operator.")
    echo("REAL ADMISSIONS-DERIVED DATA WILL BE SENT TO EXTERNAL PROVIDERS")
    echo(f"  dataset preparation: {dataset_version} (session policy {session_policy})   tasks: {', '.join(tasks)}")
    for m in models:
        echo(f"  recipient: {m}  via route {routes.get(m, 'none')}  [{recipients.get(m, 'unknown provider')}]  provider data policy: {provider_policy_status.get(m, 'UNKNOWN')}")
    for t in tasks:
        echo(f"  {t} data sent: " + "; ".join(DATA_CATEGORIES[t]))
    echo("  never sent: " + "; ".join(NOT_TRANSMITTED))
    echo(f"  {POLICY_NOTE}")
    for line in summary_lines:
        echo(line)
    phrase = expected_phrase(dataset_version, plan_fingerprint)
    echo(f"Type exactly: {phrase}")
    if prompt is None:
        import typer
        prompt = lambda msg: typer.prompt(msg, default="", show_default=False)  # noqa: E731
    if prompt("Real-data confirmation").strip() != phrase:
        raise LiveNotAuthorizedError("Real-data confirmation phrase did not match; nothing was authorized.")
    auth = RealDataAuthorization(
        dataset_version=dataset_version, session_policy=session_policy, models=tuple(sorted(models)), routes=dict(sorted(routes.items())),
        tasks=tuple(sorted(tasks)), plan_fingerprint=plan_fingerprint, cost_ceiling_usd=cost_ceiling_usd, operator=operator.strip(),
        statement=("Operator states that they reviewed the prepared admissions dataset and authorize its use for this API benchmark, "
                   "and typed the plan-bound confirmation phrase."),
        provider_policy_status=dict(sorted(provider_policy_status.items())),
    )
    _ISSUED.add(id(auth))
    return auth


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
