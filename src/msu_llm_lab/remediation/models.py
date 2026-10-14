"""Data models for knowledge base remediation and patch management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RemediationClassification(str, Enum):
    """Categorization of KB defect types determining remediation action."""
    CONFIRMED_CONTENT_DEFECT = "CONFIRMED_CONTENT_DEFECT"      # Proven mismatch with clear authoritative context
    CONTRADICTORY_FACT = "CONTRADICTORY_FACT"                  # Conflicting numbers/facts; requires official verification
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"                        # Duplicate questions/answers needing comparison before merge
    STRUCTURAL_ANOMALY = "STRUCTURAL_ANOMALY"                  # Missing or unpaired markers, stray tokens
    POLICY_INTAKE_DATE = "POLICY_INTAKE_DATE"                  # Dates, intakes, eligibility rules needing current verification
    CURRICULUM_MPU_RULE = "CURRICULUM_MPU_RULE"                # MPU or language requirement needing KB owner escalation


class RemediationStatus(str, Enum):
    """Lifecycle state of an individual remediation finding."""
    DRAFT = "DRAFT"
    INVESTIGATION_PENDING = "INVESTIGATION_PENDING"
    OFFICIAL_VERIFICATION_PENDING = "OFFICIAL_VERIFICATION_PENDING"
    KB_OWNER_ESCALATED = "KB_OWNER_ESCALATED"
    APPROVED_FOR_RELEASE = "APPROVED_FOR_RELEASE"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"


@dataclass
class RemediationFinding:
    """A documented knowledge base defect with provenance, classification, and resolution rule."""
    finding_id: str
    source_file: str
    line_number: int
    programme_name: str
    classification: RemediationClassification
    status: RemediationStatus
    issue_title: str
    issue_description: str
    original_text: str
    proposed_text: Optional[str]
    evidence_and_rationale: str
    authority_source: str
    requires_official_verification: bool
    requires_kb_owner_approval: bool
    resolution_notes: str = ""


@dataclass
class RemediationReport:
    """Summary of remediation evaluation and patch execution."""
    source_file: str
    sha256_original: str
    sha256_remediated: Optional[str]
    total_findings: int
    approved_count: int
    pending_official_verification_count: int
    pending_investigation_count: int
    escalated_kb_owner_count: int
    applied_count: int
    is_dry_run: bool
    findings: List[RemediationFinding] = field(default_factory=list)
