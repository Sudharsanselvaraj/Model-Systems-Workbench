import pytest
from pydantic import ValidationError

from msu_llm_lab.classification.contracts import (
    ExclusionReason, HistoryStatus, IntentCase, IntentLabel, LeadCase, ReviewStatus, Role,
)
from msu_llm_lab.classification.serialization import (
    NO_ACTIONS_MARKER, NO_HISTORY_MARKER, case_fingerprint, serialize_intent_case, serialize_lead_input,
)
from synthetic_fixtures import A, U, intent_case, turn


def test_valid_intent_case_defaults_are_safe():
    c = intent_case(history=[U(0, "hi"), A(1, "Hello, how can I help?")], current=U(2, "what are the fees"))
    assert c.contract_version == "v1"
    assert c.parity_status == "PRODUCTION_PARITY_NOT_ESTABLISHED"
    assert c.expected_intent is None and c.review_status == ReviewStatus.NOT_REVIEWED
    assert c.previous_assistant_message.text.startswith("Hello")
    assert not c.is_scorable


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        IntentCase(case_id="x", current_message=U(0, "hi"), user4="0123")


def test_history_must_be_strictly_ordered():
    with pytest.raises(ValidationError):
        intent_case(history=[U(1, "a"), A(1, "b")], current=U(2, "c"))


def test_current_message_must_be_user_and_last():
    with pytest.raises(ValidationError):
        intent_case(history=[U(0, "a")], current=A(1, "assistant text"))
    with pytest.raises(ValidationError):
        intent_case(history=[U(5, "a")], current=U(3, "b"))


def test_previous_assistant_must_match_history():
    with pytest.raises(ValidationError):
        intent_case(history=[U(0, "a"), A(1, "b")], current=U(2, "c"), previous_assistant_message=A(1, "different"))
    with pytest.raises(ValidationError):
        intent_case(history=[U(0, "a")], current=U(1, "c"), previous_assistant_message=A(0, "invented"))


def test_history_status_consistency():
    with pytest.raises(ValidationError):
        intent_case(history=[], current=U(0, "hi"), history_status=HistoryStatus.COMPLETE)
    with pytest.raises(ValidationError):
        intent_case(history=[U(0, "a")], current=U(1, "b"), history_status=HistoryStatus.NONE_AVAILABLE)


def test_expected_label_requires_approval_and_vice_versa():
    with pytest.raises(ValidationError):
        intent_case(expected_intent=IntentLabel.QUERY)
    with pytest.raises(ValidationError):
        intent_case(review_status=ReviewStatus.APPROVED)
    ok = intent_case(review_status=ReviewStatus.APPROVED, expected_intent=IntentLabel.GREETING)
    assert ok.is_scorable


def test_exclusion_removes_case_from_scoring():
    c = intent_case(
        review_status=ReviewStatus.APPROVED, expected_intent=IntentLabel.QUERY,
        exclusion_reason=ExclusionReason.AMBIGUOUS_HANDOFF_RULE,
    )
    assert not c.is_scorable


def test_truncation_defaults_to_unknown_not_false():
    assert intent_case().known_truncation is None


def test_intent_serialization_is_canonical_and_deterministic():
    c = intent_case(history=[U(0, "hi"), A(1, "")], current=U(2, " fees?\r\n"))
    out = serialize_intent_case(c)
    assert out == "[HISTORY]\nUSER: hi\nASSISTANT: [EMPTY]\n[CURRENT USER MESSAGE]\nfees?"
    assert out == serialize_intent_case(c)
    assert case_fingerprint(c) == case_fingerprint(c.model_copy(deep=True))


def test_missing_history_is_stated_not_invented():
    assert NO_HISTORY_MARKER in serialize_intent_case(intent_case())


def test_lead_action_must_reference_user_turn():
    base = dict(case_id="l1", history=[U(0, "x"), A(1, "y")], history_status=HistoryStatus.COMPLETE, summary_builder_version="t")
    from msu_llm_lab.classification.contracts import ExtractedAction
    bad = ExtractedAction(action_code="X", tier="A", rule_id="r", turn_index=1, source_ref="s", summary_text="t")
    with pytest.raises(ValidationError):
        LeadCase(extracted_actions=[bad], **base)
    missing = ExtractedAction(action_code="X", tier="A", rule_id="r", turn_index=9, source_ref="s", summary_text="t")
    with pytest.raises(ValidationError):
        LeadCase(extracted_actions=[missing], **base)


def test_lead_expected_requires_approval():
    with pytest.raises(ValidationError):
        LeadCase(case_id="l", summary_builder_version="t", expected_need_cta=True)
    with pytest.raises(ValidationError):
        LeadCase(case_id="l", summary_builder_version="t", review_status=ReviewStatus.APPROVED)


def test_lead_serialization_marker_does_not_assert_absence():
    c = LeadCase(case_id="l", summary_builder_version="t")
    assert serialize_lead_input(c) == NO_ACTIONS_MARKER
    c2 = LeadCase(case_id="l", summary_builder_version="t", action_summary=["a", "b"])
    assert serialize_lead_input(c2) == "a, b"
