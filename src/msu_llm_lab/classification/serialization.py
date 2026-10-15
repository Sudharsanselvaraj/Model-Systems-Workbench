"""Canonical deterministic serialization for offline test cases (not production Dify format)."""

import hashlib
import json

from msu_llm_lab.classification.contracts import HistoryStatus, IntentCase, LeadCase, Turn

EMPTY_MARKER = "[EMPTY]"
NO_HISTORY_MARKER = "[NO PRIOR HISTORY]"
NO_ACTIONS_MARKER = "[NO ACTIONS EXTRACTED]"


def _norm(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _line(turn: Turn) -> str:
    body = _norm(turn.text) or EMPTY_MARKER
    return f"{turn.role.value}: {body}"


def serialize_intent_case(case: IntentCase) -> str:
    """Render history and current message. Missing history is stated, never invented."""
    if case.history:
        history_block = "\n".join(_line(t) for t in case.history)
    else:
        history_block = NO_HISTORY_MARKER
    return f"[HISTORY]\n{history_block}\n[CURRENT USER MESSAGE]\n{_norm(case.current_message.text)}"


def serialize_lead_input(case: LeadCase) -> str:
    """Render only the action summary. The marker does NOT assert that no actions occurred."""
    if not case.action_summary:
        return NO_ACTIONS_MARKER
    return ", ".join(case.action_summary)


def case_fingerprint(case) -> str:
    """Stable SHA-256 over the case's canonical JSON (key-sorted)."""
    payload = case.model_dump_json(exclude_none=False)
    canonical = json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "HistoryStatus",
    "serialize_intent_case",
    "serialize_lead_input",
    "case_fingerprint",
    "EMPTY_MARKER",
    "NO_HISTORY_MARKER",
    "NO_ACTIONS_MARKER",
]
