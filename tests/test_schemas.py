"""Unit tests for Phase 2 Pydantic schemas and confidentiality invariants."""

from decimal import Decimal
import pytest
from pydantic import ValidationError

from msu_llm_lab.schemas.evaluation import (
    DataClassification,
    DatasetSplit,
    DifficultyLevel,
    EvaluationCase,
    VerificationRecord,
    VerificationStatus,
)
from msu_llm_lab.schemas.execution import ModelExecutionRecord
from msu_llm_lab.schemas.scoring import EvaluationScore


def test_evaluation_case_defaults_to_fail_closed_unclassified():
    """Unclassified cases must default to UNCLASSIFIED and block external dispatch."""
    case = EvaluationCase(
        id="case-001",
        query="What is the tuition fee for MBBS at MSU?",
    )
    assert case.classification == DataClassification.UNCLASSIFIED
    assert case.split == DatasetSplit.UNASSIGNED
    assert case.difficulty == DifficultyLevel.MEDIUM
    assert case.verification.status == VerificationStatus.UNVERIFIED
    # Invariant: Must NOT be eligible for external API network dispatch
    assert not case.is_eligible_for_external_dispatch


def test_verified_status_requires_explicit_reference_citation():
    """Invariant: A record claiming VERIFIED status MUST provide an explicit source citation."""
    # Attempting VERIFIED with None reference_source
    with pytest.raises(ValidationError) as exc_info:
        VerificationRecord(
            status=VerificationStatus.VERIFIED,
            reference_source=None,
        )
    assert "Records claiming VERIFIED status must specify an authoritative reference_source" in str(exc_info.value)

    # Attempting VERIFIED with empty string
    with pytest.raises(ValidationError) as exc_info_empty:
        VerificationRecord(
            status=VerificationStatus.VERIFIED,
            reference_source="   ",
        )
    assert "Records claiming VERIFIED status must specify an authoritative reference_source" in str(exc_info_empty.value)

    # Valid verified record
    valid_record = VerificationRecord(
        status=VerificationStatus.VERIFIED,
        reference_source="MSU Academic Prospectus 2026, Section 4.2",
        verifier="Dean of Academic Affairs",
    )
    assert valid_record.status == VerificationStatus.VERIFIED
    assert valid_record.reference_source is not None


def test_confidentiality_gatekeeper_eligibility():
    """Only PUBLIC or SYNTHETIC records can be routed externally."""
    for allowed in (DataClassification.PUBLIC, DataClassification.SYNTHETIC):
        case = EvaluationCase(
            id=f"case-{allowed.value}",
            query="Sample query",
            classification=allowed,
        )
        assert case.is_eligible_for_external_dispatch is True

    for blocked in (
        DataClassification.INTERNAL,
        DataClassification.CONFIDENTIAL,
        DataClassification.UNCLASSIFIED,
    ):
        case = EvaluationCase(
            id=f"case-{blocked.value}",
            query="Sample query",
            classification=blocked,
        )
        assert case.is_eligible_for_external_dispatch is False


def test_model_execution_record_validation():
    """Model execution captures latency, tokens, and Decimal pricing."""
    record = ModelExecutionRecord(
        execution_id="exec_001",
        case_id="case_001",
        model_id="google/gemini-3.5-flash-lite",
        output_text="The Bachelor in Accounting requires STPM C+.",
        latency_ms=245.8,
        prompt_tokens=120,
        completion_tokens=45,
        cost_usd=Decimal("0.0001"),
    )
    assert record.latency_ms == 245.8
    assert record.prompt_tokens == 120
    assert record.cost_usd == Decimal("0.0001")


def test_evaluation_score_weighted_calculation():
    """Evaluation score computes default weighted composite score."""
    score = EvaluationScore(
        score_id="score_001",
        execution_id="exec_001",
        accuracy=0.90,
        groundedness=0.80,
        format_adherence=1.00,
    )
    # 0.50 * 0.90 + 0.35 * 0.80 + 0.15 * 1.00 = 0.45 + 0.28 + 0.15 = 0.88
    assert pytest.approx(score.overall_score, 0.001) == 0.88
