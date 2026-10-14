"""Data models and enums for knowledge base auditing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class IssueSeverity(str, Enum):
    """Severity classification for audit findings."""
    ERROR = "ERROR"        # Deterministic flaw: missing answer, empty field, broken syntax
    WARNING = "WARNING"    # Heuristic detection: duplicate question, conflicting numbers, mismatch


class IssueCategory(str, Enum):
    """Categorical grouping for audit issues."""
    MISSING_FIELD = "MISSING_FIELD"
    MALFORMED_STRUCTURE = "MALFORMED_STRUCTURE"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    NUMERICAL_CONFLICT = "NUMERICAL_CONFLICT"
    PROGRAMME_MISMATCH = "PROGRAMME_MISMATCH"
    TIME_SENSITIVE = "TIME_SENSITIVE"


@dataclass
class Heading:
    """Represents a markdown section or programme header."""
    level: int
    text: str
    raw_text: str
    line_number: int


@dataclass
class KBRecord:
    """Represents a parsed Question-and-Answer entity."""
    record_id: str
    source_file: str
    start_line: int
    end_line: int
    raw_text: str
    programme_name: str
    section_hierarchy: List[str]
    question_raw: str
    question_clean: str
    answer_raw: str
    answer_clean: str
    has_question_tag: bool
    has_answer_tag: bool
    is_multiline: bool
    is_inline: bool
    answer_marker_count: int = 1
    source_file_hash: str = ""
    parsing_status: str = "VALID"
    validation_warnings: List[str] = field(default_factory=list)


class FindingStatus(str, Enum):
    """Review and confidence status for audit findings."""
    CONFIRMED = "CONFIRMED"
    CANDIDATE = "CANDIDATE"
    OFFICIAL_VERIFICATION_REQUIRED = "OFFICIAL_VERIFICATION_REQUIRED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class FindingClassification(str, Enum):
    """Classification taxonomy matching audit governance requirements."""
    CONFIRMED_STRUCTURAL_DEFECT = "CONFIRMED_STRUCTURAL_DEFECT"
    CONFIRMED_EXACT_DUPLICATE = "CONFIRMED_EXACT_DUPLICATE"
    POTENTIAL_SEMANTIC_DEFECT = "POTENTIAL_SEMANTIC_DEFECT"
    POTENTIAL_CONTRADICTION = "POTENTIAL_CONTRADICTION"
    POTENTIALLY_OUTDATED_INFO = "POTENTIALLY_OUTDATED_INFO"
    HUMAN_VERIFICATION_REQUIRED = "HUMAN_VERIFICATION_REQUIRED"


@dataclass
class AuditIssue:
    """Represents a validated flaw or heuristic warning."""
    issue_id: str
    source_file: str
    start_line: int
    end_line: int
    severity: IssueSeverity
    category: IssueCategory
    issue_code: str
    programme: str
    question_snippet: str
    description: str
    raw_snippet: str
    suggested_action: str
    requires_human_verification: bool = True
    status: FindingStatus = FindingStatus.CANDIDATE
    record_id: Optional[str] = None
    classification: FindingClassification = FindingClassification.POTENTIAL_SEMANTIC_DEFECT
    confidence_level: str = "HIGH"
    requires_human_approval: bool = True


@dataclass
class AuditSummary:
    """Aggregated audit statistics for a source file."""
    source_file: str
    file_size_bytes: int
    line_count: int
    sha256_before: str
    sha256_after: str
    integrity_verified: bool
    raw_question_markers: int
    raw_answer_markers: int
    parsed_records_count: int
    error_count: int
    warning_count: int
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    issues: List[AuditIssue] = field(default_factory=list)
