"""Hand-written SYNTHETIC fixtures. None of this text comes from real student data."""

from typing import List, Optional

from msu_llm_lab.classification.contracts import (
    HistoryStatus, IntentCase, ReviewStatus, Role, Turn,
)
from msu_llm_lab.schemas.evaluation import DataClassification


def turn(i: int, role: Role, text: str, ref: Optional[str] = None) -> Turn:
    return Turn(turn_index=i, role=role, text=text, source_ref=ref or f"SYN-{i}")


def U(i: int, text: str) -> Turn:
    return turn(i, Role.USER, text)


def A(i: int, text: str) -> Turn:
    return turn(i, Role.ASSISTANT, text)


def intent_case(case_id="syn-i-1", history: Optional[List[Turn]] = None, current: Optional[Turn] = None, **kw) -> IntentCase:
    history = history if history is not None else []
    last_a = next((t for t in reversed(history) if t.role == Role.ASSISTANT), None)
    base = dict(
        case_id=case_id,
        classification=DataClassification.SYNTHETIC,
        history=history,
        history_status=HistoryStatus.COMPLETE if history else HistoryStatus.NONE_AVAILABLE,
        current_message=current or U(len(history), "hello there"),
        previous_assistant_message=last_a,
    )
    base.update(kw)
    return IntentCase(**base)


SYN_INTENT_PROMPT = """## Role

Synthetic intent role.

## Input

Synthetic input description.

## General Rules (MUST BE FOLLOWED)

- Synthetic general rule.

## Category Definitions

**OUT_OF_SCOPE** — synthetic out-of-scope definition.

**GREETING** — synthetic greeting definition.

**UPLOAD_STATUS_CHECK** — synthetic upload definition.

**HUMAN_HANDOFF_REQUEST** — synthetic handoff definition.

**QUERY** — synthetic query definition.

## Decision Procedure (apply in this order)

**Step 1 — Synthetic.** Synthetic first step.

## Synthetic Examples

| 1 | a | b | `QUERY` | 6 |

## OUTPUT FORMAT

Output exactly one label and nothing else.
"""

SYN_LEAD_PROMPT = """## Role

Synthetic lead role.

## Input

Synthetic lead input.

## Decision Rules (NON-NEGOTIABLE)

1. Synthetic rule.

## Action Signal Tiers

**Tier A**

- Synthetic tier A bullet

**Tier B**

- Synthetic tier B bullet

## Output Format

Return one JSON object.

## Synthetic Examples

Input: x
{"need_cta": false}
"""


def lead_case(case_id="syn-l-1", summary=None, classification=DataClassification.SYNTHETIC, **kw):
    from msu_llm_lab.classification.contracts import LeadCase
    return LeadCase(case_id=case_id, classification=classification, summary_builder_version="synthetic-test",
                    action_summary=summary if summary is not None else ["asked about scholarship options"], **kw)
