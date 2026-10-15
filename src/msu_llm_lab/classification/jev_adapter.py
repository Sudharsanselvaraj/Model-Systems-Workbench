"""Dedicated adapter for Jev on OpenRouter's Decisions API. NOT a chat adapter.

Endpoint: POST https://openrouter.ai/api/alpha/decisions with the documented fields `model`, `state`, `questions`.
The response envelope below is documented only by a jev-1.13 example, so every field is parsed strictly and any
deviation fails the case safely instead of being guessed. Nothing here sends student data: the caller supplies a
state string that has already passed the privacy gate, plus an explicit live authorization flag.
"""

import json
import os
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from msu_llm_lab.classification import jev_payloads as jp
from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.records import AdapterResult, Task
from msu_llm_lab.classification.validators import validate_intent_output
from msu_llm_lab.models import registry
from msu_llm_lab.models.registry import Protocol
from msu_llm_lab.providers.base import LiveNotAuthorizedError, ProviderError
from msu_llm_lab.providers.http import RetryingPoster

JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL_ID = "typesafe/jev-1.14"
QUESTION_KEYS = {Task.INTENT: "intent", Task.LEAD: "lead"}


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _count(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


# OpenRouter's example response shows the served snapshot of 1.13 as `typesafe/jev-1.13-20260917`, and its endpoint metadata
# names the 1.14 endpoint `typesafe/jev-1.14-20261014` (base id + "-" + 8-digit date). Both show the form, but no actual 1.14
# RESPONSE had been seen before the first live smoke run. Snapshots recorded in the registry's live evidence are accepted without a warning; a new
# dated snapshot is accepted with a warning, and any other suffix is fatal.
SNAPSHOT_RULE_STATUS = "1.14_SNAPSHOT_20261014_LIVE_OBSERVED_IN_STORED_SMOKE_RECORDS"
_RESOLVED = re.compile(r"^(?P<base>typesafe/jev-\d+\.\d+)(?:-(?P<snap>\d{8}))?$")


def classify_resolved_version(requested: str, resolved: str) -> Tuple[str, Optional[str]]:
    """('EXACT' | 'DATED_SNAPSHOT' | 'DIFFERENT_VERSION' | 'UNRECOGNISED', snapshot-or-None). Never guesses."""
    m = _RESOLVED.match(resolved)
    if not m:
        return "UNRECOGNISED", None
    snap = m.group("snap")
    if snap is not None:
        try:
            datetime.strptime(snap, "%Y%m%d")
        except ValueError:
            return "UNRECOGNISED", None
    if m.group("base") != requested:
        return "DIFFERENT_VERSION", snap
    return ("DATED_SNAPSHOT", snap) if snap else ("EXACT", None)


def version_consistent(requested: str, resolved: str) -> bool:
    return classify_resolved_version(requested, resolved)[0] in ("EXACT", "DATED_SNAPSHOT")


def parse_decisions_response(
    task: Task, requested_model_id: str, data: Any, *, price_estimator: Optional[Callable[[int, int], Optional[Decimal]]] = None,
) -> AdapterResult:
    """Strictly parse one Decisions-API response body (already JSON-decoded) into an AdapterResult."""
    base = dict(task=task, protocol=Protocol.DECISIONS_API, requested_model_id=requested_model_id)
    issues: List[str] = []
    warnings: List[str] = []
    if not isinstance(data, dict):
        return AdapterResult(ok=False, issues=["ENVELOPE_NOT_AN_OBJECT"], **base)
    if data.get("error"):
        return AdapterResult(ok=False, issues=["ERROR_BODY_IN_SUCCESS_RESPONSE"], **base)

    # --- model / provider (informational but type-checked; absence is a warning, never invented)
    resolved = data.get("model")
    if resolved is None:
        warnings.append("RESOLVED_MODEL_MISSING")
    elif not isinstance(resolved, str) or not resolved:
        issues.append("MODEL_FIELD_MALFORMED")
        resolved = None
    else:
        kind, _snap = classify_resolved_version(requested_model_id, resolved)
        if kind == "DIFFERENT_VERSION":
            issues.append("RESOLVED_VERSION_DIFFERS_FROM_REQUESTED")
        elif kind == "UNRECOGNISED":
            issues.append("RESOLVED_VERSION_FORM_UNRECOGNISED")
        elif kind == "DATED_SNAPSHOT" and resolved not in registry.observed_snapshots(requested_model_id):
            warnings.append("DATED_SNAPSHOT_NOT_PREVIOUSLY_OBSERVED")  # a snapshot seen in stored live evidence carries no warning
    provider = data.get("provider")
    if provider is None:
        warnings.append("PROVIDER_MISSING")
    elif not isinstance(provider, str) or not provider:
        issues.append("PROVIDER_FIELD_MALFORMED")
        provider = None
    if "id" not in data:
        warnings.append("REQUEST_ID_MISSING")

    # --- usage
    p_tok = c_tok = None
    actual: Optional[Decimal] = None
    native: Optional[Decimal] = None
    usage = data.get("usage")
    if usage is None:
        warnings.append("USAGE_MISSING")
    elif not isinstance(usage, dict):
        issues.append("USAGE_MALFORMED")
    else:
        for name in ("input_tokens", "output_tokens"):
            if name in usage and _count(usage[name]) is None:
                issues.append("USAGE_TOKEN_FIELD_MALFORMED")
        p_tok, c_tok = _count(usage.get("input_tokens")), _count(usage.get("output_tokens"))
        if "cost" in usage:
            if _num(usage["cost"]) and usage["cost"] >= 0:
                native = Decimal(str(usage["cost"]))
                actual = native if registry.provider_cost_is_usd() else None  # credits are not booked as USD until the unit is verified
            else:
                issues.append("USAGE_COST_MALFORMED")
        else:
            warnings.append("COST_MISSING")
        if p_tok is None and c_tok is None and "input_tokens" not in usage and "output_tokens" not in usage:
            warnings.append("TOKEN_COUNTS_MISSING")
    usage_source = "PROVIDER_REPORTED" if (p_tok is not None and c_tok is not None) else (
        "PARTIAL" if (p_tok is not None or c_tok is not None) else "UNAVAILABLE")
    estimated = price_estimator(p_tok, c_tok) if (price_estimator and p_tok is not None and c_tok is not None) else None
    if actual is not None:
        cost, basis = actual, "PROVIDER_REPORTED"
    elif estimated is not None:
        cost, basis = estimated, "ESTIMATED_FROM_CATALOG"
    else:
        cost, basis = Decimal("0"), "UNAVAILABLE"

    telemetry = dict(
        resolved_model_id=resolved, serving_provider=provider, prompt_tokens=p_tok, completion_tokens=c_tok, usage_source=usage_source,
        native_cost=native, native_cost_unit=registry.NATIVE_COST_UNIT_LABEL if native is not None else None,
        actual_cost_usd=actual, estimated_cost_usd=estimated, cost_usd=cost, cost_basis=basis, warnings=warnings,
    )

    # --- answers
    answers = data.get("answers")
    key = QUESTION_KEYS[task]
    if not isinstance(answers, dict):
        return AdapterResult(ok=False, issues=issues + ["ANSWERS_MISSING_OR_MALFORMED"], **base, **telemetry)
    if key not in answers:
        issues.append("ANSWER_MISSING_FOR_QUESTION")
    if set(answers) - {key}:
        issues.append("UNEXPECTED_ANSWER_KEYS")
    answer = answers.get(key)

    out: Dict[str, Any] = {}
    if key in answers and task == Task.INTENT:
        chk = jp.check_choice_answer(answer)
        issues += chk.issues
        if chk.ok:
            label = validate_intent_output(chk.values["label"].value)  # authoritative validator on the chosen option
            if label.valid:
                out = dict(predicted_label=label.label, label_probs={k: float(v) for k, v in chk.values["probabilities"].items()},
                           native_confidence=float(chk.values["confidence"]), confidence_source="NATIVE_DISTRIBUTION")
            else:
                issues.append("CHOICE_REJECTED_BY_INTENT_VALIDATOR")
    elif key in answers and task == Task.LEAD:
        chk = jp.check_noul_answer(answer)
        issues += chk.issues
        if chk.ok:
            out = dict(yes_prob=chk.values["yes_prob"])  # no threshold applied here; a frozen dev threshold decides later
    if issues:  # an untrusted response never yields a prediction, even if the answer object looked valid
        out = {}
    return AdapterResult(ok=not issues, issues=issues, **base, **telemetry, **out)


class JevDecisionsAdapter:
    protocol = Protocol.DECISIONS_API

    def __init__(
        self,
        model_id: str = JEV_MODEL_ID,
        api_key: Optional[str] = None,
        endpoint: str = JEV_ENDPOINT,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        poster: Optional[RetryingPoster] = None,
        registry_stage: str = "pilot",
    ):
        self.registry_stage = registry_stage
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.endpoint = endpoint
        self.poster = poster or RetryingPoster(timeout_seconds=timeout_seconds, max_attempts=max_attempts)

    def supports(self, task: Task) -> bool:
        return task in QUESTION_KEYS

    def max_output_tokens(self, task: Task) -> int:
        return 0  # typed answers; output tokens are documented as free

    def build_request_body(self, task: Task, state_text: str, prompt_text: str) -> Dict[str, Any]:
        question = jp.build_intent_question(prompt_text) if task == Task.INTENT else jp.build_lead_question(prompt_text)
        return jp.build_decisions_request(self.model_id, state_text, {QUESTION_KEYS[task]: question})

    def _estimator(self) -> Callable[[int, int], Optional[Decimal]]:
        def estimate(p: int, c: int) -> Optional[Decimal]:
            if not registry.estimate_permitted(self.model_id):
                return None
            prices = registry.model_level_prices(self.model_id)
            if prices is None:
                return None
            return ((Decimal(p) * prices[0] + Decimal(c) * prices[1]) / Decimal("1000000")).quantize(Decimal("0.000001"))
        return estimate

    def classify(self, task: Task, state_text: str, prompt_text: str, *, live_authorized: bool) -> AdapterResult:
        if not live_authorized:
            raise LiveNotAuthorizedError("Jev request refused: no explicit live authorization.")
        if not self.supports(task):
            raise ProviderError(f"Jev adapter does not support task {task}")
        blockers = registry.synthetic_live_blockers(self.model_id, self.registry_stage)
        if blockers:
            raise ProviderError(f"Live request refused: registry blockers for '{self.model_id}': {', '.join(blockers)}.")
        if not self.api_key or not self.api_key.strip():
            raise ProviderError("OPENROUTER_API_KEY environment variable not configured.")
        body = self.build_request_body(task, state_text, prompt_text)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        posted = self.poster.post(self.endpoint, headers, body)
        try:
            data = json.loads(posted.body.decode("utf-8"))
        except ValueError as e:
            raise ProviderError("Decisions API returned an unreadable (non-JSON) response.", attempts=posted.attempts,
                                errors=posted.errors + ["INVALID_RESPONSE"], retryable=False) from e
        result = parse_decisions_response(task, self.model_id, data, price_estimator=self._estimator())
        return result.model_copy(update=dict(
            latency_ms=round(posted.e2e_ms, 2), attempt_latency_ms=round(posted.attempt_ms, 2),
            model_latency_ms=None, attempts=posted.attempts, retry_errors=list(posted.errors),
        ))
