"""Pure builders that adapt the AUTHORITATIVE prompts into Jev Decisions-API question objects.

No network, no HTTP client, no second prompt: every instruction and criterion is extracted verbatim from
`main_intent_classifier.md` / `lead_qualification_prompt.md` at build time. Only the one-line wrappers below are
added, and only the sections that describe the free-text/JSON output format are left out, because Jev answers
are typed (Choice / Noul). Jev's documented `state`/`questions` shape is used; the OpenRouter response envelope is
still unverified, so these helpers parse a single answer object only.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from msu_llm_lab.classification.contracts import IntentLabel

INTENT_WRAPPER = (
    "Classify the CURRENT user message in the state. Choose exactly one option. "
    "The option descriptions are the category definitions."
)
LEAD_WRAPPER = (
    "Read the action summary in the state. Answer yes if the student should be handed to a human counsellor now "
    "(the rules below decide), otherwise no."
)

INTENT_INSTRUCTION_SECTIONS = ("Role", "Input", "General Rules", "Decision Procedure", "Synthetic Examples")
LEAD_INSTRUCTION_SECTIONS = ("Role", "Input", "Decision Rules", "Action Signal Tiers")
OMITTED_INTENT_SECTIONS = ("OUTPUT FORMAT",)
OMITTED_LEAD_SECTIONS = ("Output Format", "Synthetic Examples")

PROBABILITY_SUM_TOLERANCE = 0.02  # TypeSafe documents rounding without stating its size; kept loose and recorded


class PromptParseError(ValueError):
    pass


def parse_sections(prompt_text: str) -> Dict[str, str]:
    """Split a prompt into `## ` sections, keeping each body verbatim. Key = heading without the part in parentheses."""
    parts = re.split(r"^## (.+)$", prompt_text, flags=re.M)
    sections: Dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name = re.sub(r"\s*\(.*\)\s*$", "", parts[i]).strip()
        sections[name] = parts[i + 1].strip("\n")
    return sections


def _section_text(sections: Dict[str, str], names: Tuple[str, ...]) -> str:
    missing = [n for n in names if n not in sections]
    if missing:
        raise PromptParseError(f"authoritative prompt is missing section(s): {missing}")
    return "\n\n".join(f"## {n}\n\n{sections[n]}" for n in names)


def intent_criteria(prompt_text: str) -> Dict[str, str]:
    """Five option -> definition pairs, taken verbatim from the Category Definitions section. Option names are the labels."""
    body = parse_sections(prompt_text).get("Category Definitions")
    if body is None:
        raise PromptParseError("Category Definitions section not found")
    found = {m.group(1): m.group(2).strip() for m in re.finditer(r"^\*\*([A-Z_]+)\*\* — (.+)$", body, flags=re.M)}
    expected = {l.value for l in IntentLabel}
    if set(found) != expected:
        raise PromptParseError(f"category definitions {sorted(found)} do not match the five labels")
    return {l.value: found[l.value] for l in IntentLabel}


def build_intent_question(prompt_text: str) -> Dict[str, Any]:
    sections = parse_sections(prompt_text)
    return {
        "type": "choice",
        "instructions": INTENT_WRAPPER + "\n\n" + _section_text(sections, INTENT_INSTRUCTION_SECTIONS),
        "criteria": intent_criteria(prompt_text),
    }


def build_lead_question(prompt_text: str) -> Dict[str, Any]:
    """Noul question. `criteria` is optional in Jev's docs and its format for yes/no is not documented, so none is sent."""
    sections = parse_sections(prompt_text)
    return {"type": "noul", "instructions": LEAD_WRAPPER + "\n\n" + _section_text(sections, LEAD_INSTRUCTION_SECTIONS)}


def build_decisions_request(model: str, state_text: str, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Body for POST /api/alpha/decisions: documented top-level fields `model`, `state`, `questions` only."""
    if not state_text or not questions:
        raise ValueError("state and at least one question are required")
    return {"model": model, "state": [state_text], "questions": questions}


class AnswerCheck:
    def __init__(self, ok: bool, issues: List[str], **values: Any):
        self.ok, self.issues, self.values = ok, issues, values


def check_choice_answer(answer: Any) -> AnswerCheck:
    """Validate one Choice answer: option must be one of the five labels; probabilities cover exactly those labels."""
    issues: List[str] = []
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return AnswerCheck(False, ["NOT_A_CHOICE_ANSWER"])
    labels = {l.value for l in IntentLabel}
    choice = answer.get("choice")
    if choice not in labels:
        issues.append("CHOICE_NOT_A_VALID_LABEL")
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != labels:
        issues.append("PROBABILITY_KEYS_MISMATCH")
    else:
        vals = list(probs.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1 for v in vals):
            issues.append("PROBABILITY_OUT_OF_RANGE")
        elif abs(sum(vals) - 1.0) > PROBABILITY_SUM_TOLERANCE:
            issues.append("PROBABILITIES_DO_NOT_SUM_TO_ONE")
        elif choice in labels and probs[choice] < max(vals) - 1e-9:
            issues.append("CHOICE_IS_NOT_THE_MODE")
    conf = answer.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        issues.append("CONFIDENCE_MISSING_OR_OUT_OF_RANGE")
    label: Optional[IntentLabel] = IntentLabel(choice) if choice in labels else None
    return AnswerCheck(not issues, issues, label=label, probabilities=probs if isinstance(probs, dict) else None,
                       confidence=conf if not issues or "CONFIDENCE_MISSING_OR_OUT_OF_RANGE" not in issues else None)


def check_noul_answer(answer: Any) -> AnswerCheck:
    """Validate one Noul answer: only P(yes). A `confidence` key is reported, never invented or accepted as meaningful."""
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        return AnswerCheck(False, ["NOT_A_NOUL_ANSWER"])
    issues: List[str] = []
    p = answer.get("noul")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not (0.0 <= p <= 1.0):
        issues.append("YES_PROBABILITY_MISSING_OR_OUT_OF_RANGE")
    if "confidence" in answer:
        issues.append("UNEXPECTED_CONFIDENCE_FIELD")
    return AnswerCheck(not issues, issues, yes_prob=float(p) if not issues else None)
