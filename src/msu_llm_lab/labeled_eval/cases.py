"""Cases are classified CONFIDENTIAL (potentially real, origin unverified): never described as synthetic or anonymised.
Build contract cases from the label-free inference inputs (never from the answer key) so the existing serializers and runner can be reused."""

from __future__ import annotations

from typing import Any, Dict

from msu_llm_lab.classification.contracts import DataClassification, HistoryStatus, IntentCase, LeadCase, Role, Turn


def intent_case(row: Dict[str, Any]) -> IntentCase:
    turns = [Turn(turn_index=i, role=Role(t["role"]), text=t["text"], source_ref=f"labeled-eval:{row['case_id']}:{i}") for i, t in enumerate(row["history_turns"])]
    cur = Turn(turn_index=len(turns), role=Role.USER, text=row["current_message"], source_ref=f"labeled-eval:{row['case_id']}:cur")
    last_a = next((t for t in reversed(turns) if t.role == Role.ASSISTANT), None)
    return IntentCase(case_id=row["case_id"], history=turns, history_status=HistoryStatus.UNKNOWN, classification=DataClassification.CONFIDENTIAL, current_message=cur, previous_assistant_message=last_a)


def lead_case(row: Dict[str, Any]) -> LeadCase:
    """Action summary only. `user_msg` is deliberately not accepted here (contract decision pending)."""
    return LeadCase(case_id=row["case_id"], action_summary=[row["action_summary"]], summary_builder_version="labeled-test-file-verbatim", history_status=HistoryStatus.UNKNOWN, classification=DataClassification.CONFIDENTIAL)
