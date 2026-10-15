import pytest

from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.validators import InvalidReason, validate_intent_output, validate_lead_output


@pytest.mark.parametrize("label", [l.value for l in IntentLabel])
def test_exact_labels_accepted(label):
    assert validate_intent_output(label).label.value == label
    assert validate_intent_output(f"  {label}\n").valid


@pytest.mark.parametrize("raw,reason", [
    (None, InvalidReason.EMPTY_OUTPUT), ("", InvalidReason.EMPTY_OUTPUT), ("   ", InvalidReason.EMPTY_OUTPUT),
    ("query", InvalidReason.WRONG_CASE), ("Greeting", InvalidReason.WRONG_CASE),
    ("QUERY.", InvalidReason.EXTRA_TEXT), ("The answer is QUERY", InvalidReason.EXTRA_TEXT),
    ("QUERY\nGREETING", InvalidReason.EXTRA_TEXT), ('{"intent":"QUERY"}', InvalidReason.EXTRA_TEXT),
    ("HANDOFF", InvalidReason.UNKNOWN_LABEL), ("I cannot help", InvalidReason.UNKNOWN_LABEL),
])
def test_invalid_intent_outputs(raw, reason):
    v = validate_intent_output(raw)
    assert not v.valid and v.label is None and v.reason == reason


def test_lead_valid():
    assert validate_lead_output('{"need_cta": true}').need_cta is True
    assert validate_lead_output(' {"need_cta":false}\n').need_cta is False


@pytest.mark.parametrize("raw,reason", [
    (None, InvalidReason.EMPTY_OUTPUT),
    ('```json\n{"need_cta": true}\n```', InvalidReason.CODE_FENCE),
    ("need_cta: true", InvalidReason.INVALID_JSON),
    ('Here you go: {"need_cta": true}', InvalidReason.INVALID_JSON),
    ('[true]', InvalidReason.NOT_OBJECT),
    ('{"cta": true}', InvalidReason.MISSING_KEY),
    ('{"need_cta": true, "why": "x"}', InvalidReason.EXTRA_KEYS),
    ('{"need_cta": "true"}', InvalidReason.NON_BOOLEAN),
    ('{"need_cta": 1}', InvalidReason.NON_BOOLEAN),
    ('{"need_cta": null}', InvalidReason.NON_BOOLEAN),
    ('{"need_cta": true, "need_cta": false}', InvalidReason.DUPLICATE_KEYS),
])
def test_invalid_lead_outputs(raw, reason):
    v = validate_lead_output(raw)
    assert not v.valid and v.need_cta is None and v.reason == reason
