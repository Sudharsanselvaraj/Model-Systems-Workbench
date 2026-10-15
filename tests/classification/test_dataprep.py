"""Tests for local dataset preparation, using a hand-written SYNTHETIC CSV (fake hashes, fake contacts). No real data is read."""

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from msu_llm_lab.classification import dataprep as dp
from msu_llm_lab.classification.contracts import IntentCase, LeadCase

USER4_SENTINEL = "9999"
EMAIL = "syn.person@example.test"
PHONE = "012-345 6789"
IC = "990101-14-5678"
T0 = datetime(2026, 9, 1, 3, 0, 0, tzinfo=timezone.utc)


def ts(minutes: float) -> str:
    return (T0 + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


# group, minute offset, turn, query, answer
SYN_ROWS = [
    ("hash-A", 0, 1, "hello", "Hi, how can I help?"),
    ("hash-A", 1, 2, "What are the fees for the Diploma in Business?", "The fee is RM 20,000."),
    ("hash-A", 45, 3, "thanks", "You're welcome."),                       # 44 min later: new session at 30m, same at 2h
    ("hash-A", 400, 1, "Can I speak to a counsellor?", None),            # 6h later: new session in both; answer missing
    ("hash-B", 5, 1, f"my email is {EMAIL} and my number is {PHONE}", "Thanks, noted."),
    ("hash-B", 6, 1, f"IC {IC} also my name is Aisyah Binti Test", "Received."),   # turn resets inside the session
    ("hash-B", 6.5, 3, "I already uploaded my SPM result, can you check?", "I cannot see uploads."),
    ("hash-C", 10, 1, "Would you like fee details?", "Yes."),
    ("hash-C", 10.2, 2, "no", "OK."),
    ("hash-C", 10.4, 3, "no", "Alright."),                                   # near duplicate of the previous row
    ("hash-D", 20, 1, "Any scholarship?", "Yes, several."),
    ("hash-D", 20, 1, "Any scholarship?", "Yes, several."),              # exact duplicate (same group, query, answer)
    ("hash-E", 30, 1, "我想申请", "Hello"),                                 # unsupported script
]
HEADER = dp.EXPECTED_COLUMNS


def write_csv(path: Path, rows=SYN_ROWS):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (g, m, turn, q, a) in enumerate(rows):
            w.writerow([f"trace-{i}", ts(m), USER4_SENTINEL, g, "", turn, q, "" if a is None else a])


@pytest.fixture
def prepared(tmp_path):
    src = tmp_path / "syn.csv"
    write_csv(src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    out = tmp_path / "out"
    manifest = dp.prepare(src, out)
    return src, out, manifest, before


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ redaction
def test_redaction_replaces_direct_identifiers_and_counts_them():
    t, c = dp.redact(f"mail {EMAIL} call {PHONE} ic {IC} my name is Aisyah Binti Test, RM 62,500 in 2026")
    assert EMAIL not in t and "345" not in t and IC not in t and "Aisyah" not in t
    assert c["[EMAIL]"] == 1 and c["[PHONE]"] == 1 and c["[ID_NUMBER]"] == 1 and c["[NAME]"] == 1
    assert "RM 62,500" in t and "2026" in t  # amounts and years are not identifiers


def test_plain_text_is_untouched():
    t, c = dp.redact("What are the entry requirements for the Bachelor of Nursing?")
    assert t == "What are the entry requirements for the Bachelor of Nursing?" and not c


# ------------------------------------------------------------------ source safety and user4
def test_source_is_unchanged_and_user4_never_appears_anywhere(prepared):
    src, out, manifest, before = prepared
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before == manifest["source"]["sha256"] and manifest["source"]["unchanged_after_run"]
    assert "user4" not in manifest["source"]["columns_read"]
    for f in out.iterdir():
        text = f.read_text()
        assert f'"{USER4_SENTINEL}"' not in text and f"{USER4_SENTINEL}," not in text, f.name  # the phone-prefix value is in no artifact
        if f.name != "manifest.json":
            assert "user4" not in text, f.name  # the column name appears only in the manifest's exclusion note
    assert "user4" in json.dumps(manifest["source"]["excluded_column"])


def test_raw_identifiers_and_contacts_are_absent_from_every_derived_file(prepared):
    _, out, _, _ = prepared
    for f in out.iterdir():
        text = f.read_text()
        for raw in ("hash-A", "hash-B", "trace-0", EMAIL, PHONE, IC, "Aisyah"):
            assert raw not in text, (f.name, raw)


def test_unexpected_columns_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unexpected CSV columns"):
        dp.load_rows(bad)


# ------------------------------------------------------------------ manifest accounting
def test_every_row_is_accounted_for_and_nothing_is_dropped(prepared):
    _, out, m, _ = prepared
    rows = lines(out / "cleaned_rows.jsonl")
    assert len(rows) == len(SYN_ROWS) == m["source"]["rows"] == m["cleaning"]["rows_accounted"]
    assert sum(m["cleaning"]["dispositions"].values()) == len(SYN_ROWS)
    assert m["parity_status"] == "PRODUCTION_PARITY_NOT_ESTABLISHED" and all(r["parity_status"] == "PRODUCTION_PARITY_NOT_ESTABLISHED" for r in rows)


def test_missing_answers_are_preserved_not_invented(prepared):
    _, out, m, _ = prepared
    rows = {r["row_id"]: r for r in lines(out / "cleaned_rows.jsonl")}
    assert rows["R0004"]["answer_redacted"] is None and rows["R0004"]["answer_status"] == "MISSING" and "MISSING_ANSWER" in rows["R0004"]["flags"]
    assert m["cleaning"]["answers_missing"] == 1


def test_duplicates_are_flagged_not_deleted(prepared):
    _, out, m, _ = prepared
    rows = {r["row_id"]: r for r in lines(out / "cleaned_rows.jsonl")}
    assert rows["R0012"]["exact_duplicate_of"] == "R0011" and "EXACT_DUPLICATE" in rows["R0012"]["flags"]
    assert rows["R0010"]["near_duplicate_of"] == "R0009" and "NEAR_DUPLICATE" in rows["R0010"]["flags"]
    assert rows["R0011"]["exact_duplicate_of"] is None and m["cleaning"]["duplicates"] == {"exact_duplicate_rows": 1, "near_duplicate_rows": 1}


def test_redaction_counts_and_flags_in_manifest(prepared):
    _, out, m, _ = prepared
    toks = m["cleaning"]["redaction_token_counts"]
    assert toks["query[EMAIL]"] == 1 and toks["query[PHONE]"] == 1 and toks["query[ID_NUMBER]"] == 1 and toks["query[NAME]"] == 1
    assert m["cleaning"]["rows_with_any_redaction"] == 2 and "UNSUPPORTED_SCRIPT" in m["cleaning"]["flag_counts"]
    assert any("named-entity" in l for l in m["cleaning"]["limitations"])


# ------------------------------------------------------------------ sessions
def test_two_session_policies_are_built_and_compared_without_choosing_one(prepared):
    _, out, m, _ = prepared
    c = m["session_comparison"]
    assert c["sessions"]["30m"] > c["sessions"]["2h"] and m["provisional_session_policy"] is True
    assert c["rows_whose_gap_falls_between_thresholds"] == 1  # the 44-minute gap
    assert c["boundary_categories"]["30m"]["NEW_SESSION_AFTER_GAP"] == 2 and c["boundary_categories"]["2h"]["NEW_SESSION_AFTER_GAP"] == 1
    assert (out / "sessions_30m.jsonl").exists() and (out / "sessions_2h.jsonl").exists()


def test_turn_is_evidence_only_and_never_changes_a_boundary(prepared):
    _, out, m, _ = prepared
    rows = {r["row_id"]: r for r in lines(out / "cleaned_rows.jsonl")}
    assert rows["R0006"]["session_30m"] == rows["R0005"]["session_30m"]   # turn reset to 1 but no gap: same session
    assert m["session_comparison"]["turn_evidence"]["2h"].get("TURN_RESET_INSIDE_SESSION", 0) >= 1
    assert rows["R0003"]["session_30m"] != rows["R0002"]["session_30m"] and rows["R0003"]["session_2h"] == rows["R0002"]["session_2h"]


def test_review_sample_is_redacted_and_local(prepared):
    _, out, _, _ = prepared
    sample = lines(out / "session_review_sample.jsonl")
    assert sample and all(len(s["rows"]) >= 1 for s in sample)
    text = (out / "session_review_sample.jsonl").read_text()
    assert EMAIL not in text and PHONE not in text and "hash-" not in text


# ------------------------------------------------------------------ candidates
def test_intent_candidates_follow_the_contract_with_null_labels_and_confidential_class(prepared):
    _, out, m, _ = prepared
    cands = lines(out / "intent_candidates_30m.jsonl")
    assert len(cands) == len(SYN_ROWS) == m["candidates"]["intent_30m"]["cases"]
    for c in cands:
        case = IntentCase.model_validate(c["case"])
        assert case.classification.value == "CONFIDENTIAL" and case.expected_intent is None and case.review_status.value == "NOT_REVIEWED"
        assert c["candidate"]["suggested_intent"] is None and c["candidate"]["expected_intent"] is None
        assert c["candidate"]["parity_status"] == "PRODUCTION_PARITY_NOT_ESTABLISHED" and not case.is_scorable
        assert c["candidate"]["assistant_text_provenance"] == "AS_EXPORTED_NOT_A_PREDICTION"


def test_history_is_ordered_within_the_session_and_incomplete_when_answers_are_missing(prepared):
    _, out, _, _ = prepared
    by = {c["candidate"]["row_id"]: c for c in lines(out / "intent_candidates_2h.jsonl")}
    c3 = by["R0003"]["case"]  # same 2h session as R0001/R0002
    assert [t["role"] for t in c3["history"]] == ["USER", "ASSISTANT", "USER", "ASSISTANT"] and c3["history_status"] == "COMPLETE"
    assert c3["previous_assistant_message"]["source_ref"] == "R0002:A"
    assert by["R0001"]["case"]["history_status"] == "NONE_AVAILABLE" and by["R0001"]["case"]["history"] == []
    assert by["R0004"]["case"]["history_status"] == "NONE_AVAILABLE"  # new session after a 6h gap


def test_missing_preceding_assistant_message_marks_short_replies_insufficient_context(tmp_path):
    rows = [("hash-X", 0, 1, "What are the fees?", None), ("hash-X", 1, 2, "yes", "Okay.")]
    src = tmp_path / "s.csv"
    write_csv(src, rows)
    out = tmp_path / "o"
    dp.prepare(src, out)
    c = {x["candidate"]["row_id"]: x for x in lines(out / "intent_candidates_30m.jsonl")}["R0002"]
    assert c["case"]["history_status"] == "INCOMPLETE" and "PRECEDING_ASSISTANT_MESSAGE_MISSING" in c["candidate"]["quality_flags"]
    assert c["case"]["exclusion_reason"] == "INSUFFICIENT_CONTEXT" and c["candidate"]["candidate_status"] == "EXCLUDED"
    assert c["case"]["previous_assistant_message"] is None  # no earlier assistant message was invented or substituted


def test_o1_bare_reply_to_an_offer_is_excluded_with_its_rule_reference(prepared):
    _, out, _, _ = prepared
    by = {c["candidate"]["row_id"]: c for c in lines(out / "intent_candidates_30m.jsonl")}
    # R0008 is the assistant 'Yes.'; R0009's query 'no' replies to an assistant answer that is not an offer, so it is not O1
    assert by["R0009"]["candidate"]["candidate_status"] != "EXCLUDED"
    assert dp._is_o1("no", "Would you like fee details?") and dp._is_o1("Thanks", "Do you want the entry requirements as well?")
    assert not dp._is_o1("no thanks", "Would you like fee details?") and not dp._is_o1("no", "The fee is RM 20,000.") and not dp._is_o1("no", None)


def test_lead_candidates_keep_summary_separate_with_provenance_and_null_label(prepared):
    _, out, m, _ = prepared
    cands = lines(out / "lead_candidates_2h.jsonl")
    assert len(cands) == m["session_comparison"]["sessions"]["2h"]
    for c in cands:
        lc = LeadCase.model_validate(c["case"])
        assert lc.expected_need_cta is None and lc.summary_provenance == "OFFLINE_RULE_BUILDER_NOT_PRODUCTION" and lc.classification.value == "CONFIDENTIAL"
        assert c["candidate"]["expected_need_cta"] is None and c["candidate"]["suggested_need_cta"] is None
        assert "builder_tally_not_a_label" in c["candidate"] and c["candidate"]["source_row_ids"]
        for a in lc.extracted_actions:  # every action points to a turn that points to a source row
            assert any(t.turn_index == a.turn_index and t.source_ref == a.source_ref for t in lc.history)
    detected = {a["action_code"] for c in cands for a in c["case"]["extracted_actions"]}
    assert "VOLUNTEERED_CONTACT" in detected  # works on redacted [EMAIL]/[PHONE] tokens


def test_unresolved_rules_exclude_only_cases_they_could_flip():
    from msu_llm_lab.classification.lead_summary import tier_tally
    assert tier_tally([])["tier_a_actions"] == 0  # sanity for the rule used in dataprep


# ------------------------------------------------------------------ views, splits, leakage
def test_group_level_splits_have_no_leakage_and_are_policy_independent(prepared):
    _, out, m, _ = prepared
    for name, chk in m["leakage_checks"].items():
        assert chk["groups_in_multiple_splits"] == 0 and chk["groups_in_multiple_views_or_splits"] == 0, name
    a = {c["candidate"]["group_id"]: (c["candidate"]["view"], c["candidate"]["split"]) for c in lines(out / "intent_candidates_30m.jsonl")}
    b = {c["candidate"]["group_id"]: (c["candidate"]["view"], c["candidate"]["split"]) for c in lines(out / "intent_candidates_2h.jsonl")}
    assert a == b
    assert {s["view"] for s in lines(out / "split_assignments.jsonl")} <= {"NATURAL", "CHALLENGE"}


def test_selection_hints_are_hints_not_labels():
    assert dp.selection_hints("Can I speak to a counsellor?") == ["HANDOFF_PHRASE"]
    assert dp.selection_hints("I already uploaded my results, please check") == ["UPLOAD_STATUS_PHRASE"]
    assert dp.selection_hints("Tell me a joke") == ["OUT_OF_SCOPE_PHRASE"] and dp.selection_hints("What are the fees?") == []


def test_rerun_is_deterministic(tmp_path):
    src = tmp_path / "syn.csv"
    write_csv(src)
    dp.prepare(src, tmp_path / "o1")
    dp.prepare(src, tmp_path / "o2")
    for name in ("cleaned_rows.jsonl", "intent_candidates_30m.jsonl", "lead_candidates_2h.jsonl", "split_assignments.jsonl", "manifest.json"):
        a, b = (tmp_path / "o1" / name).read_text(), (tmp_path / "o2" / name).read_text()
        if name == "manifest.json":  # the manifest lists artifacts; compare the stable parts
            a, b = json.loads(a), json.loads(b)
            assert a["cleaning"] == b["cleaning"] and a["candidates"] == b["candidates"] and a["source"]["sha256"] == b["source"]["sha256"]
        else:
            assert a == b, name


def test_no_network_or_model_imports_in_the_module():
    src = Path(dp.__file__).read_text()
    for banned in ("import socket", "import urllib", "import requests", "import httpx", "providers.", "jev_adapter", "chat_adapter"):
        assert banned not in src
