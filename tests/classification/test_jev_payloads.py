"""Semantic-parity tests: Jev question objects must carry the authoritative prompt text, not a competing copy.

Parser tests run on a small hand-written SYNTHETIC prompt so they always run. Parity tests read the real authoritative
files and are skipped (visibly) when they are absent, as on a clean clone where the files are git-ignored.
"""

import re
from pathlib import Path

import pytest

from msu_llm_lab.classification import jev_payloads as jp
from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.validators import validate_intent_output

ROOT = Path(__file__).resolve().parents[2]
INTENT_FILE, LEAD_FILE = ROOT / "main_intent_classifier.md", ROOT / "lead_qualification_prompt.md"

SYN_INTENT = """## Role

Synthetic role text.

## Input

Synthetic input text.

## General Rules (MUST BE FOLLOWED)

- Synthetic rule.

## Category Definitions

**OUT_OF_SCOPE** — synthetic out of scope definition.

**GREETING** — synthetic greeting definition.

**UPLOAD_STATUS_CHECK** — synthetic upload definition.

**HUMAN_HANDOFF_REQUEST** — synthetic handoff definition.

**QUERY** — synthetic query definition.

## Decision Procedure (apply in this order)

**Step 1 — Synthetic.** Do the first thing.

## Synthetic Examples

| 1 | a | b | `QUERY` | 6 |

## OUTPUT FORMAT

Output exactly one label and nothing else.
"""


def test_parse_sections_keeps_bodies_verbatim_and_strips_parenthetical_heading_text():
    s = jp.parse_sections(SYN_INTENT)
    assert list(s) == ["Role", "Input", "General Rules", "Category Definitions", "Decision Procedure", "Synthetic Examples", "OUTPUT FORMAT"]
    assert s["Role"] == "Synthetic role text."


def test_synthetic_intent_question_shape_and_omitted_output_section():
    q = jp.build_intent_question(SYN_INTENT)
    assert q["type"] == "choice" and set(q) == {"type", "instructions", "criteria"}
    assert list(q["criteria"]) == [l.value for l in IntentLabel]
    assert q["criteria"]["QUERY"] == "synthetic query definition."
    assert q["instructions"].startswith(jp.INTENT_WRAPPER)
    assert "Do the first thing." in q["instructions"] and "Output exactly one label" not in q["instructions"]
    assert "synthetic query definition" not in q["instructions"]  # definitions travel as criteria, once


def test_missing_sections_or_labels_fail_loudly():
    with pytest.raises(jp.PromptParseError):
        jp.build_intent_question(SYN_INTENT.replace("## Decision Procedure", "## Something Else"))
    with pytest.raises(jp.PromptParseError):
        jp.intent_criteria(SYN_INTENT.replace("**GREETING** —", "**GREET** —"))
    with pytest.raises(jp.PromptParseError):
        jp.build_lead_question(SYN_INTENT)


def test_decisions_request_uses_only_documented_top_level_fields():
    body = jp.build_decisions_request("typesafe/jev-1.14", "state text", {"intent": jp.build_intent_question(SYN_INTENT)})
    assert set(body) == {"model", "state", "questions"} and body["state"] == ["state text"]
    with pytest.raises(ValueError):
        jp.build_decisions_request("m", "", {"q": {}})


LABELS = [l.value for l in IntentLabel]


def good_choice(label="QUERY"):
    probs = {l: 0.05 for l in LABELS}
    probs[label] = 0.8
    return {"type": "choice", "choice": label, "probabilities": probs, "confidence": 0.7}


def test_choice_answer_checks():
    ok = jp.check_choice_answer(good_choice("GREETING"))
    assert ok.ok and ok.values["label"] == IntentLabel.GREETING and ok.values["confidence"] == 0.7
    bad = {
        "CHOICE_NOT_A_VALID_LABEL": {**good_choice(), "choice": "greeting"},
        "PROBABILITY_KEYS_MISMATCH": {**good_choice(), "probabilities": {"QUERY": 1.0}},
        "PROBABILITY_OUT_OF_RANGE": {**good_choice(), "probabilities": {**good_choice()["probabilities"], "QUERY": 1.5}},
        "PROBABILITIES_DO_NOT_SUM_TO_ONE": {**good_choice(), "probabilities": {l: 0.1 for l in LABELS}},
        "CHOICE_IS_NOT_THE_MODE": {**good_choice("QUERY"), "choice": "GREETING"},
        "CONFIDENCE_MISSING_OR_OUT_OF_RANGE": {k: v for k, v in good_choice().items() if k != "confidence"},
    }
    for issue, ans in bad.items():
        assert issue in jp.check_choice_answer(ans).issues, issue
    assert jp.check_choice_answer({"type": "noul", "noul": 0.5}).issues == ["NOT_A_CHOICE_ANSWER"]


def test_choice_labels_are_accepted_by_the_authoritative_validator_unchanged():
    for l in LABELS:
        assert validate_intent_output(jp.check_choice_answer(good_choice(l)).values["label"].value).valid


def test_noul_answer_checks_and_no_invented_confidence():
    assert jp.check_noul_answer({"type": "noul", "noul": 0.37}).values["yes_prob"] == 0.37
    for bad in ({"type": "noul"}, {"type": "noul", "noul": 1.2}, {"type": "noul", "noul": True}, {"type": "noul", "noul": "0.4"}):
        assert not jp.check_noul_answer(bad).ok
    assert "UNEXPECTED_CONFIDENCE_FIELD" in jp.check_noul_answer({"type": "noul", "noul": 0.4, "confidence": 0.9}).issues


# ---------------------------------------------------------------- parity with the real authoritative prompts
needs_intent = pytest.mark.skipif(not INTENT_FILE.exists(), reason="authoritative prompt file not present (git-ignored)")
needs_lead = pytest.mark.skipif(not LEAD_FILE.exists(), reason="authoritative prompt file not present (git-ignored)")


@needs_intent
def test_intent_instructions_contain_every_authoritative_step_verbatim():
    text = INTENT_FILE.read_text()
    q = jp.build_intent_question(text)
    sections = jp.parse_sections(text)
    for name in jp.INTENT_INSTRUCTION_SECTIONS:
        assert sections[name] in q["instructions"], name
    steps = re.findall(r"^\*\*Step \d+ — .*$", sections["Decision Procedure"], flags=re.M)
    assert len(steps) == 7 and all(s in q["instructions"] for s in steps)
    for omitted in jp.OMITTED_INTENT_SECTIONS:
        assert sections[omitted] not in q["instructions"]


@needs_intent
def test_intent_instructions_add_nothing_beyond_wrapper_and_authoritative_sections():
    text = INTENT_FILE.read_text()
    q = jp.build_intent_question(text)
    sections = jp.parse_sections(text)
    rest = q["instructions"].replace(jp.INTENT_WRAPPER, "")
    for name in jp.INTENT_INSTRUCTION_SECTIONS:
        rest = rest.replace(f"## {name}", "").replace(sections[name], "")
    assert rest.strip() == ""


@needs_intent
def test_intent_criteria_are_the_five_authoritative_definitions_verbatim():
    text = INTENT_FILE.read_text()
    q = jp.build_intent_question(text)
    assert list(q["criteria"]) == LABELS
    body = jp.parse_sections(text)["Category Definitions"]
    for label, definition in q["criteria"].items():
        assert f"**{label}** — {definition}" in body
    # the confirmed rules are reflected in the carried text
    assert "any update?" in q["criteria"]["QUERY"] and "not a handoff" in q["criteria"]["HUMAN_HANDOFF_REQUEST"]
    assert "no thanks" in q["criteria"]["GREETING"]


@needs_intent
def test_every_authoritative_example_label_is_a_valid_option():
    text = INTENT_FILE.read_text()
    rows = re.findall(r"^\| (\d+) \|.*\| `([A-Z_]+)` \|", jp.parse_sections(text)["Synthetic Examples"], flags=re.M)
    assert len(rows) >= 20 and {l for _, l in rows} <= set(LABELS)


@needs_lead
def test_lead_noul_instructions_carry_all_rules_and_every_tier_bullet_verbatim():
    text = LEAD_FILE.read_text()
    q = jp.build_lead_question(text)
    assert q["type"] == "noul" and set(q) == {"type", "instructions"}
    sections = jp.parse_sections(text)
    for name in jp.LEAD_INSTRUCTION_SECTIONS:
        assert sections[name] in q["instructions"], name
    bullets = [b for b in sections["Action Signal Tiers"].splitlines() if b.startswith("- ")]
    assert len(bullets) == 41 and all(b in q["instructions"] for b in bullets)
    assert "Count each bullet at most once" in q["instructions"] and "Repeating the same enquiry" in q["instructions"]
    for omitted in jp.OMITTED_LEAD_SECTIONS:
        assert sections[omitted] not in q["instructions"]
    assert "need_cta" not in q["instructions"].replace("`need_cta`", "")  # raw-JSON output instructions are not sent
