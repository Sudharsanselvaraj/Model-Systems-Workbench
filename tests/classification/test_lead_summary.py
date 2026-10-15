import json

import pytest

from msu_llm_lab.classification.contracts import HistoryStatus
from msu_llm_lab.classification.lead_summary import (
    BUILDER_VERSION, PROVISIONAL_DEFAULTS, UNIMPLEMENTED_RULES, SessionEvidence,
    build_action_summary, build_lead_case, tier_tally,
)
from msu_llm_lab.schemas.evaluation import DataClassification
from synthetic_fixtures import A, U


def codes(res):
    return {a.action_code for a in res.extracted_actions}


def reasons(res):
    return {a.reason_code for a in res.ambiguities}


def build(*turns, status=HistoryStatus.COMPLETE, **kw):
    return build_action_summary(list(turns), status, **kw)


def test_defaults_and_unimplemented_are_documented():
    assert set(PROVISIONAL_DEFAULTS) == {f"B{i}" for i in range(1, 11)}
    assert "B_RETURN_VISIT" in UNIMPLEMENTED_RULES


def test_acknowledgement_only_creates_nothing():
    for t in ["ok", "Okay thanks!", "thank you", "noted", "terima kasih", "ic"]:
        r = build(U(0, t))
        assert r.extracted_actions == [] and r.ambiguities == [], t


def test_ack_does_not_suppress_real_action_in_same_turn():
    assert "APPLICATION_LINK_REQUEST" in codes(build(U(0, "ok thanks, can you send the application link?")))


def test_assistant_text_never_creates_actions():
    r = build(U(0, "hi"), A(1, "You can apply via the application link, and the hostel is available. Scholarships exist."))
    assert r.extracted_actions == []


@pytest.mark.parametrize("text,code", [
    ("Where is the application link?", "APPLICATION_LINK_REQUEST"),
    ("What documents are needed to apply?", "DOCUMENTS_REQUIRED_QUESTION"),
    ("How do I pay the deposit?", "DEPOSIT_PAYMENT_QUESTION"),
    ("I want to speak to a counsellor", "HUMAN_CONTACT_REQUEST"),
    ("Can someone call me?", "CALLBACK_REQUEST"),
    ("I already submitted my SPM results", "SUBMISSION_OR_COMMITMENT"),
    ("I will upload my transcript tomorrow", "SUBMISSION_OR_COMMITMENT"),
    ("Can my credits be transferred?", "CREDIT_TRANSFER_OR_EXEMPTION"),
    ("Do you accept APEL?", "APEL_OR_WORK_EXPERIENCE"),
    ("Can I change my programme later?", "INTRA_MSU_TRANSFER"),
    ("Am I eligible?", "ELIGIBILITY_QUESTION"),
    ("Can I still join the March intake?", "INTAKE_NAMED"),
    ("Can I still apply?", "LATE_APPLICATION_QUESTION"),
    ("How long does the approval take?", "APPLICATION_DURATION_QUESTION"),
    ("This is urgent", "URGENCY"),
    ("When I start, where do I collect my timetable?", "OWNERSHIP_LANGUAGE"),
    ("Is there a hostel?", "POST_DECISION_LOGISTICS"),
    ("Is there an open day?", "OPEN_DAY_OR_VISIT"),
    ("Any discount available?", "PRICE_OBJECTION"),
    ("Is it recognised by MQA?", "RECOGNITION_QUESTION"),
    ("Boleh saya tahu pautan permohonan?", "APPLICATION_LINK_REQUEST"),
    ("Adakah saya layak?", "ELIGIBILITY_QUESTION"),
])
def test_tier_a_rules_fire(text, code):
    r = build(U(0, text))
    assert code in codes(r)
    assert all(a.tier == "A" for a in r.extracted_actions if a.action_code == code)


@pytest.mark.parametrize("text,code", [
    ("Do you have scholarships?", "FUNDING_QUESTION"),
    ("Is the campus tour video available?", "CAMPUS_TOUR"),
    ("My father wants to know the details", "SPONSOR_OR_DECISION_MAKER"),
    ("Ada biasiswa?", "FUNDING_QUESTION"),
])
def test_tier_b_rules_fire(text, code):
    r = build(U(0, text))
    assert code in codes(r) and all(a.tier == "B" for a in r.extracted_actions if a.action_code == code)


def test_near_misses_do_not_fire():
    for t in ["What courses do you offer?", "Where is the campus?", "hello", "What is the weather", "Tell me more"]:
        assert build(U(0, t)).extracted_actions == [], t


def test_capability_question_is_ambiguity_not_action():  # B1
    r = build(U(0, "Can I upload my IGCSE results here?"))
    assert "SUBMISSION_OR_COMMITMENT" not in codes(r)
    assert "CAPABILITY_QUESTION_NOT_ACTION" in reasons(r)


def test_negated_submission_not_counted():
    r = build(U(0, "I have not submitted my results yet"))
    assert "SUBMISSION_OR_COMMITMENT" not in codes(r) and "NEGATION_PRESENT" in reasons(r)


def test_stated_results_require_specifics():  # B2
    specific = build(U(0, "I got 5 A in SPM"))
    vague = build(U(0, "I have my SPM results"))
    assert "RESULTS_STATED" in codes(specific)
    assert "RESULTS_STATED" not in codes(vague) and "NONSPECIFIC_RESULTS" in reasons(vague)
    assert "RESULTS_STATED" in codes(build(U(0, "My CGPA is 3.2 from my diploma")))


def test_status_vs_results_precedence():  # B7
    status_only = build(U(0, "I just finished SPM"))
    assert codes(status_only) == {"STATED_ACADEMIC_STATUS"}
    both = build(U(0, "I just finished SPM and got 6 A"))
    assert "RESULTS_STATED" in codes(both) and "STATED_ACADEMIC_STATUS" not in codes(both)


def test_status_not_unprompted_when_assistant_asked():
    r = build(U(0, "hi"), A(1, "What is your current qualification?"), U(2, "I have a diploma in business"))
    assert "STATED_ACADEMIC_STATUS" not in codes(r)


def test_broad_scope_rules_are_flagged_b4_b5():
    assert "BROAD_SCOPE_REVIEW" in reasons(build(U(0, "Is there a hostel?")))
    assert "SEMANTIC_OBJECTION_REVIEW" in reasons(build(U(0, "Any discount?")))


def test_distinct_tier_b_types_not_repeated_wording():  # B3
    r = build(U(0, "Any scholarship?"), U(1, "what scholarships do you have"), U(2, "tell me about scholarships again"))
    assert len(r.extracted_actions) == 1 and r.extracted_actions[0].occurrences == 3
    assert tier_tally(r.extracted_actions) == {"tier_a_actions": 0, "tier_b_distinct_types": 1}
    assert r.action_summary == ["asked about scholarships or funding (3 times)"]


def test_repeated_programme_fee_enquiry_not_double_counted():
    r = build(U(0, "How much is the fee for the Diploma in Business?"), U(1, "what is the fee for diploma in business again?"))
    fee = [a for a in r.extracted_actions if a.action_code == "PROGRAMME_FEE_QUESTION"]
    assert len(fee) == 1 and fee[0].occurrences == 2
    assert "REPEAT_COUNTING_RULE_UNAPPROVED" not in reasons(r)
    assert tier_tally(r.extracted_actions)["tier_b_distinct_types"] == 1


def test_fee_without_resolvable_programme_is_ambiguity():
    r = build(U(0, "How much is the fee?"))
    assert "PROGRAMME_FEE_QUESTION" not in codes(r) and "PROGRAMME_NOT_RESOLVED" in reasons(r)


def test_prompt_few_shot_shapes_via_tally_only():
    ex1 = build(U(0, "I submitted my SPM results"), U(1, "I uploaded my IELTS certificate"))
    assert tier_tally(ex1.extracted_actions)["tier_a_actions"] >= 1
    ex2 = build(U(0, "Is the campus tour video available?"))
    assert tier_tally(ex2.extracted_actions) == {"tier_a_actions": 0, "tier_b_distinct_types": 1}
    ex3 = build(U(0, "campus tour video?"), U(1, "fee for diploma in business?"), U(2, "any scholarship options"))
    assert tier_tally(ex3.extracted_actions) == {"tier_a_actions": 0, "tier_b_distinct_types": 3}
    ex4 = build(U(0, "what time is the admissions office open"), A(1, "9am to 5pm"), U(2, "okay thanks"))
    assert ex4.extracted_actions == []


def test_volunteered_contact_requires_unprompted():
    unprompted = build(U(0, "my email is syn.student@example.test"))
    assert "VOLUNTEERED_CONTACT" in codes(unprompted)
    asked = build(U(0, "hi"), A(1, "Please share your email so we can follow up."), U(2, "syn.student@example.test"))
    assert "VOLUNTEERED_CONTACT" not in codes(asked)


def test_no_raw_text_in_output():
    secret = "syn.student@example.test"
    r = build(U(0, f"my email is {secret} and I submitted my SPM results"))
    dumped = json.dumps([a.model_dump() for a in r.extracted_actions] + [a.model_dump() for a in r.ambiguities])
    assert secret not in dumped and "my email" not in dumped


def test_provenance_points_to_supporting_turn():
    r = build(U(0, "hello"), A(1, "hi"), turn_with_ref := U(2, "How do I pay the deposit?"))
    a = r.extracted_actions[0]
    assert (a.turn_index, a.source_ref) == (2, turn_with_ref.source_ref)


def test_return_visit_needs_approved_evidence():  # B9
    t = [U(0, "hello")]
    assert "RETURN_VISIT" not in codes(build(*t))
    assert "RETURN_VISIT_UNVERIFIABLE" in build(*t).unknown_context
    assert "RETURN_VISIT" not in codes(build(*t, session_evidence=SessionEvidence(approved=False, session_count=3)))
    approved = build(*t, session_evidence=SessionEvidence(approved=True, session_count=2))
    assert "RETURN_VISIT" in codes(approved) and "RETURN_VISIT_UNVERIFIABLE" not in approved.unknown_context


def test_missing_history_is_unknown_not_negative():  # rule 9
    r = build(status=HistoryStatus.UNKNOWN)
    assert r.extracted_actions == [] and "HISTORY_UNKNOWN" in r.unknown_context
    assert "HISTORY_INCOMPLETE" in build(U(0, "hi"), status=HistoryStatus.INCOMPLETE).unknown_context


def test_unsupported_script_flagged_b8():
    r = build(U(0, "我想申请 application link"))
    assert "UNSUPPORTED_SCRIPT" in reasons(r) and "APPLICATION_LINK_REQUEST" in codes(r)
    assert "UNSUPPORTED_SCRIPT" not in reasons(build(U(0, "Boleh saya tahu yuran?")))


def test_unimplemented_context_rules_raise_ambiguity_not_actions():
    r = build(U(0, "What is the deadline?"), U(1, "let me think about it"))
    assert r.extracted_actions == [] and "PROGRAMME_CONTEXT_UNRESOLVED" in reasons(r)


def test_determinism():
    turns = [U(0, "Any scholarship?"), A(1, "Yes."), U(2, "How do I pay the deposit?")]
    a, b = build(*turns), build(*turns)
    assert a == b


def test_build_lead_case_has_no_label_and_marks_provenance():
    c = build_lead_case("syn-l-1", [U(0, "Is there a hostel?")], DataClassification.SYNTHETIC, HistoryStatus.COMPLETE)
    assert c.expected_need_cta is None and c.summary_builder_version == BUILDER_VERSION
    assert c.summary_provenance == "OFFLINE_RULE_BUILDER_NOT_PRODUCTION" and c.parity_status == "PRODUCTION_PARITY_NOT_ESTABLISHED"
    assert not c.is_scorable


# ---- L1 (confirmed): repeated wording of the same enquiry is one Tier B signal ----
ENTRY_Q1 = "What are the entry requirements for the Diploma in Business?"
ENTRY_Q2 = "entry requirements for diploma in business please"


def test_repeated_entry_requirements_count_once():
    r = build(U(0, ENTRY_Q1), U(1, ENTRY_Q2))
    acts = [a for a in r.extracted_actions if a.tier == "B"]
    assert [a.action_code for a in acts] == ["PROGRAMME_ENQUIRY_REPEATED"]
    assert acts[0].occurrences == 2 and acts[0].turn_index == 0
    assert tier_tally(r.extracted_actions)["tier_b_distinct_types"] == 1
    assert "REPEAT_COUNTING_RULE_UNAPPROVED" not in reasons(r)


def test_single_entry_requirements_question_is_not_a_signal():
    r = build(U(0, ENTRY_Q1))
    assert r.extracted_actions == []


def test_entry_requirements_without_resolvable_programme_not_counted():
    r = build(U(0, "What are the entry requirements?"), U(1, "entry requirements please"))
    assert r.extracted_actions == []


def test_entry_requirements_for_different_programmes_are_not_repeats():
    r = build(U(0, ENTRY_Q1), U(1, "What are the entry requirements for the Bachelor of Nursing?"))
    assert r.extracted_actions == []


def test_each_programme_repeated_gives_one_distinct_type_not_two():
    r = build(U(0, ENTRY_Q1), U(1, ENTRY_Q2),
              U(2, "entry requirements for bachelor of nursing"), U(3, "bachelor of nursing entry requirements again"))
    assert len(r.extracted_actions) == 2
    assert tier_tally(r.extracted_actions)["tier_b_distinct_types"] == 1


def test_fee_and_entry_for_same_programme_never_double_counted():
    r = build(U(0, "What is the fee for the Diploma in Business?"), U(1, ENTRY_Q1), U(2, ENTRY_Q2))
    b = [a for a in r.extracted_actions if a.tier == "B"]
    assert [a.action_code for a in b] == ["PROGRAMME_FEE_QUESTION"]
    assert tier_tally(r.extracted_actions)["tier_b_distinct_types"] == 1
    assert "MIXED_ENQUIRY_TOPICS_COUNTED_ONCE" in reasons(r)


def test_fee_twice_for_different_programmes_is_one_type():
    r = build(U(0, "fee for diploma in business?"), U(1, "fee for bachelor of nursing?"))
    assert len(r.extracted_actions) == 2
    assert tier_tally(r.extracted_actions)["tier_b_distinct_types"] == 1


def test_repeated_entry_requirements_deterministic_and_no_raw_text():
    a, b = build(U(0, ENTRY_Q1), U(1, ENTRY_Q2)), build(U(0, ENTRY_Q1), U(1, ENTRY_Q2))
    assert a == b
    dumped = json.dumps([x.model_dump() for x in a.extracted_actions])
    assert "entry requirements" not in dumped.replace("entry requirements more", "")
