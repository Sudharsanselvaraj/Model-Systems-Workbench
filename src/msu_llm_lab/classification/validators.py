"""Strict output validators for the two classifiers."""

import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from msu_llm_lab.classification.contracts import IntentLabel


class InvalidReason(str, Enum):
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    UNKNOWN_LABEL = "UNKNOWN_LABEL"
    EXTRA_TEXT = "EXTRA_TEXT"
    WRONG_CASE = "WRONG_CASE"
    CODE_FENCE = "CODE_FENCE"
    INVALID_JSON = "INVALID_JSON"
    NOT_OBJECT = "NOT_OBJECT"
    MISSING_KEY = "MISSING_KEY"
    EXTRA_KEYS = "EXTRA_KEYS"
    DUPLICATE_KEYS = "DUPLICATE_KEYS"
    NON_BOOLEAN = "NON_BOOLEAN"


class IntentValidation(BaseModel):
    valid: bool
    label: Optional[IntentLabel] = None
    reason: Optional[InvalidReason] = None


class LeadValidation(BaseModel):
    valid: bool
    need_cta: Optional[bool] = None
    reason: Optional[InvalidReason] = None


_LABELS = {l.value: l for l in IntentLabel}
_UPPER = {l.value.upper() for l in IntentLabel}


def validate_intent_output(raw: Optional[str]) -> IntentValidation:
    """Exactly one label, case-sensitive. Only leading/trailing whitespace is tolerated."""
    if raw is None or not raw.strip():
        return IntentValidation(valid=False, reason=InvalidReason.EMPTY_OUTPUT)
    text = raw.strip()
    if text in _LABELS:
        return IntentValidation(valid=True, label=_LABELS[text])
    if text.upper() in _UPPER:
        return IntentValidation(valid=False, reason=InvalidReason.WRONG_CASE)
    if any(lbl in text.upper() for lbl in _UPPER):
        return IntentValidation(valid=False, reason=InvalidReason.EXTRA_TEXT)
    return IntentValidation(valid=False, reason=InvalidReason.UNKNOWN_LABEL)


def _reject_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys")
    return dict(pairs)


def validate_lead_output(raw: Optional[str]) -> LeadValidation:
    """Exactly `{"need_cta": <bool>}`: no fences, no prose, no extra keys, real boolean only."""
    if raw is None or not raw.strip():
        return LeadValidation(valid=False, reason=InvalidReason.EMPTY_OUTPUT)
    text = raw.strip()
    if text.startswith("```"):
        return LeadValidation(valid=False, reason=InvalidReason.CODE_FENCE)
    try:
        obj = json.loads(text, object_pairs_hook=_reject_duplicates)
    except ValueError as e:
        reason = InvalidReason.DUPLICATE_KEYS if "duplicate keys" in str(e) else InvalidReason.INVALID_JSON
        return LeadValidation(valid=False, reason=reason)
    if not isinstance(obj, dict):
        return LeadValidation(valid=False, reason=InvalidReason.NOT_OBJECT)
    if "need_cta" not in obj:
        return LeadValidation(valid=False, reason=InvalidReason.MISSING_KEY)
    if set(obj) != {"need_cta"}:
        return LeadValidation(valid=False, reason=InvalidReason.EXTRA_KEYS)
    if not isinstance(obj["need_cta"], bool):
        return LeadValidation(valid=False, reason=InvalidReason.NON_BOOLEAN)
    return LeadValidation(valid=True, need_cta=obj["need_cta"])
