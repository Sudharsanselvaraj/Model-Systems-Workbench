"""Evaluation schemas and confidentiality classifications for MSU LLM Inference Workbench."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class DataClassification(str, Enum):
    """Fail-closed confidentiality classification levels.
    
    Only PUBLIC and SYNTHETIC records may ever be routed to external model APIs.
    INTERNAL, CONFIDENTIAL, and UNCLASSIFIED records are blocked before network dispatch.
    """
    PUBLIC = "PUBLIC"
    SYNTHETIC = "SYNTHETIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    UNCLASSIFIED = "UNCLASSIFIED"


CaseConfidentiality = DataClassification


class VerificationStatus(str, Enum):
    """Verification state for evaluation ground truths."""
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"


class DifficultyLevel(str, Enum):
    """Categorized query complexity level."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    EDGE_CASE = "EDGE_CASE"


class DatasetSplit(str, Enum):
    """Target partition for model evaluation."""
    TRAIN = "TRAIN"
    DEV = "DEV"
    TEST = "TEST"
    UNASSIGNED = "UNASSIGNED"


class VerificationRecord(BaseModel):
    """Verification provenance for ground truth citations.
    
    Acceptance Criteria: Any record claiming VERIFIED status MUST provide an explicit,
    authoritative reference source citation.
    """
    status: VerificationStatus = Field(default=VerificationStatus.UNVERIFIED)
    reference_source: Optional[str] = Field(
        default=None,
        description="Authoritative citation (e.g. Official Academic Guide 2026, Dean's Office bulletin)."
    )
    verifier: Optional[str] = Field(
        default=None,
        description="Identity of the academic reviewer or domain expert."
    )
    verified_at: Optional[datetime] = Field(default=None)
    notes: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_verified_source(self) -> "VerificationRecord":
        if self.status == VerificationStatus.VERIFIED:
            if not self.reference_source or not self.reference_source.strip():
                raise ValueError(
                    "Records claiming VERIFIED status must specify an authoritative reference_source."
                )
        return self


class EvaluationCase(BaseModel):
    """Standard evaluation test case for student admissions queries."""
    id: str = Field(description="Unique case identifier (e.g. msu-eval-001)")
    query: str = Field(description="Student query prompt")
    expected_answer: Optional[str] = Field(
        default=None,
        description="Authoritative ground truth answer"
    )
    programme_category: str = Field(
        default="General",
        description="Faculty or programme category (e.g. Accounting, Broadcasting, Medicine)"
    )
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    classification: DataClassification = Field(
        default=DataClassification.UNCLASSIFIED,
        description="Confidentiality classification. Defaults to UNCLASSIFIED (fail-closed)."
    )
    split: DatasetSplit = Field(default=DatasetSplit.UNASSIGNED)
    verification: VerificationRecord = Field(default_factory=VerificationRecord)
    expected_context: Optional[str] = Field(
        default=None,
        description="Authoritative grounding context/chunks that the retriever is expected to fetch."
    )
    expected_retrieval_keywords: List[str] = Field(
        default_factory=list,
        description="Key terms or section headings expected to be returned by retriever."
    )
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_eligible_for_external_dispatch(self) -> bool:
        """Determines if record is approved for external provider routing."""
        return self.classification in (DataClassification.PUBLIC, DataClassification.SYNTHETIC)
