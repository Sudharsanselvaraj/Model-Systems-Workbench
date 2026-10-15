"""`input_contract_v1`: versioned offline input contracts for the two classification tasks.

PRODUCTION PARITY IS NOT ESTABLISHED. These contracts define an offline benchmark
representation. They do not claim to match the production Dify serialization.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from msu_llm_lab.schemas.evaluation import DataClassification

CONTRACT_VERSION = "v1"
PARITY_STATUS = "PRODUCTION_PARITY_NOT_ESTABLISHED"
SUMMARY_PROVENANCE = "OFFLINE_RULE_BUILDER_NOT_PRODUCTION"


class IntentLabel(str, Enum):
    """The five (and only) valid Main Intent labels."""
    GREETING = "GREETING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UPLOAD_STATUS_CHECK = "UPLOAD_STATUS_CHECK"
    HUMAN_HANDOFF_REQUEST = "HUMAN_HANDOFF_REQUEST"
    QUERY = "QUERY"


class Role(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class HistoryStatus(str, Enum):
    """How much history the case carries. UNKNOWN is never treated as 'none happened'."""
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    NONE_AVAILABLE = "NONE_AVAILABLE"
    UNKNOWN = "UNKNOWN"


class ReviewStatus(str, Enum):
    NOT_REVIEWED = "NOT_REVIEWED"
    APPROVED = "APPROVED"


class ExclusionReason(str, Enum):
    AMBIGUOUS_HANDOFF_RULE = "AMBIGUOUS_HANDOFF_RULE"
    UNRESOLVED_BUSINESS_RULE = "UNRESOLVED_BUSINESS_RULE"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    OTHER = "OTHER"


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    role: Role
    text: str
    source_ref: str = Field(min_length=1, description="Pseudonymous pointer to the source row(s)")


class _CaseBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    contract_version: Literal["v1"] = CONTRACT_VERSION
    parity_status: Literal["PRODUCTION_PARITY_NOT_ESTABLISHED"] = PARITY_STATUS
    classification: DataClassification = DataClassification.UNCLASSIFIED
    group_id: Optional[str] = Field(
        default=None,
        description="Pseudonymous grouping key (provisional user_hash surrogate) used for leakage-safe splits",
    )
    history: List[Turn] = Field(default_factory=list)
    history_status: HistoryStatus = HistoryStatus.UNKNOWN
    known_truncation: Optional[bool] = Field(default=None, description="None means unknown, never 'false'")
    review_status: ReviewStatus = ReviewStatus.NOT_REVIEWED
    exclusion_reason: Optional[ExclusionReason] = None

    @model_validator(mode="after")
    def _check_history(self):
        idx = [t.turn_index for t in self.history]
        if any(b <= a for a, b in zip(idx, idx[1:])):
            raise ValueError("history turn_index values must be strictly increasing")
        if self.history_status == HistoryStatus.NONE_AVAILABLE and self.history:
            raise ValueError("history_status NONE_AVAILABLE requires an empty history")
        if not self.history and self.history_status in (HistoryStatus.COMPLETE, HistoryStatus.INCOMPLETE):
            raise ValueError("empty history must be NONE_AVAILABLE or UNKNOWN")
        return self

    @property
    def is_scorable(self) -> bool:
        """Approved label present and not excluded."""
        return self.review_status == ReviewStatus.APPROVED and self.exclusion_reason is None


class IntentCase(_CaseBase):
    """Main Intent Classifier case. `previous_assistant_message` is the last ASSISTANT turn in history."""
    current_message: Turn
    previous_assistant_message: Optional[Turn] = None
    assistant_text_provenance: Literal["AS_EXPORTED"] = "AS_EXPORTED"
    truncation_rule: Optional[str] = None
    expected_intent: Optional[IntentLabel] = None

    @model_validator(mode="after")
    def _check_intent(self):
        if self.current_message.role != Role.USER:
            raise ValueError("current_message must be a USER turn")
        if self.history and self.current_message.turn_index <= self.history[-1].turn_index:
            raise ValueError("current_message must come after all history turns")
        last_assistant = next((t for t in reversed(self.history) if t.role == Role.ASSISTANT), None)
        if self.previous_assistant_message != last_assistant:
            raise ValueError("previous_assistant_message must equal the last ASSISTANT turn in history (or be None)")
        if self.expected_intent is not None and self.review_status != ReviewStatus.APPROVED:
            raise ValueError("expected_intent may only be set once review_status is APPROVED")
        if self.review_status == ReviewStatus.APPROVED and self.expected_intent is None:
            raise ValueError("an APPROVED intent case needs expected_intent")
        return self


class ExtractedAction(BaseModel):
    """One action signal. Evidence is by reference only: no raw student text is stored."""
    model_config = ConfigDict(extra="forbid")

    action_code: str
    tier: Literal["A", "B"]
    rule_id: str
    turn_index: int = Field(ge=0)
    source_ref: str
    occurrences: int = Field(default=1, ge=1)
    programme_key: Optional[str] = None
    summary_text: str


class AmbiguityRecord(BaseModel):
    """A signal the builder could not resolve; routed to human review rather than guessed."""
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    reason_code: str
    turn_index: Optional[int] = Field(default=None, ge=0)
    source_ref: Optional[str] = None
    detail: str = ""


class LeadCase(_CaseBase):
    """Lead Qualifier case. The model sees only the action summary, never the transcript."""
    action_summary: List[str] = Field(default_factory=list)
    extracted_actions: List[ExtractedAction] = Field(default_factory=list)
    ambiguities: List[AmbiguityRecord] = Field(default_factory=list)
    unknown_context: List[str] = Field(default_factory=list)
    summary_builder_version: str
    summary_provenance: Literal["OFFLINE_RULE_BUILDER_NOT_PRODUCTION"] = SUMMARY_PROVENANCE
    expected_need_cta: Optional[bool] = None

    @model_validator(mode="after")
    def _check_lead(self):
        by_index = {t.turn_index: t for t in self.history}
        for a in self.extracted_actions:
            t = by_index.get(a.turn_index)
            if t is None:
                raise ValueError(f"extracted action references unknown turn_index {a.turn_index}")
            if t.role != Role.USER:
                raise ValueError("extracted actions may only reference USER turns")
        if self.expected_need_cta is not None and self.review_status != ReviewStatus.APPROVED:
            raise ValueError("expected_need_cta may only be set once review_status is APPROVED")
        if self.review_status == ReviewStatus.APPROVED and self.expected_need_cta is None:
            raise ValueError("an APPROVED lead case needs expected_need_cta")
        return self
