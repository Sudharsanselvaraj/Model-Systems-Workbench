"""Validation engine for detecting deterministic errors and heuristic warnings in KB records."""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from msu_llm_lab.audit.models import (
    KBRecord,
    Heading,
    AuditIssue,
    IssueSeverity,
    IssueCategory,
)


# Regex patterns for heuristic checks
RM_FEE_REGEX = re.compile(r'RM\s?([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{2})?)', re.IGNORECASE)
YEAR_REGEX = re.compile(r'\b(202[0-9])\b')
DEADLINE_KEYWORDS = re.compile(r'\b(deadline|intake|valid until|closing date|expire|expiry|validity)\b', re.IGNORECASE)


def normalize_question(q: str) -> str:
    """Normalizes a question string for duplicate detection."""
    # Lowercase, remove markdown artifacts, punctuation, and extra whitespace
    q_norm = q.lower()
    q_norm = re.sub(r'[^\w\s]', '', q_norm)
    return " ".join(q_norm.split()).strip()


def extract_rm_fees(text: str) -> List[str]:
    """Extracts all RM currency amounts from text."""
    matches = RM_FEE_REGEX.findall(text)
    # Normalize: remove spaces and commas to compare numerical values
    cleaned = []
    for m in matches:
        val = m.replace(',', '').strip()
        cleaned.append(val)
    return cleaned


class KBValidator:
    """Validates parsed KB records for structural, deterministic, and heuristic issues."""

    def __init__(
        self,
        source_file: str,
        records: List[KBRecord],
        headings: List[Heading],
        orphan_lines: List[Tuple[int, str]],
    ):
        self.source_file = source_file
        self.records = records
        self.headings = headings
        self.orphan_lines = orphan_lines
        self.issues: List[AuditIssue] = []
        self._issue_counter = 0

    def _add_issue(
        self,
        start_line: int,
        end_line: int,
        severity: IssueSeverity,
        category: IssueCategory,
        issue_code: str,
        programme: str,
        question_snippet: str,
        description: str,
        raw_snippet: str,
        suggested_action: str,
    ) -> None:
        self._issue_counter += 1
        issue_id = f"ISSUE-{self._issue_counter:04d}"
        issue = AuditIssue(
            issue_id=issue_id,
            source_file=self.source_file,
            start_line=start_line,
            end_line=end_line,
            severity=severity,
            category=category,
            issue_code=issue_code,
            programme=programme,
            question_snippet=question_snippet[:100],
            description=description,
            raw_snippet=raw_snippet[:120].replace("\n", " "),
            suggested_action=suggested_action,
            requires_human_verification=True,
        )
        self.issues.append(issue)

    def validate_all(self) -> List[AuditIssue]:
        """Runs all deterministic and heuristic validations."""
        self._validate_headings()
        self._validate_orphan_lines()
        self._validate_records_deterministic()
        self._validate_duplicates()
        self._validate_numerical_conflicts()
        self._validate_programme_mismatches()
        self._validate_time_sensitivity()
        return self.issues

    def _validate_headings(self) -> None:
        """Flags empty or malformed headings."""
        for h in self.headings:
            if not h.text:
                self._add_issue(
                    start_line=h.line_number,
                    end_line=h.line_number,
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MALFORMED_STRUCTURE,
                    issue_code="EMPTY_HEADING",
                    programme="General",
                    question_snippet="N/A (Heading)",
                    description=f"Heading at line {h.line_number} is empty (missing section or programme title).",
                    raw_snippet=h.raw_text,
                    suggested_action="Supply missing section name or remove empty header marker.",
                )

    def _validate_orphan_lines(self) -> None:
        """Flags stray non-blank text outside headings and Q&A blocks."""
        for line_no, raw_text in self.orphan_lines:
            # Skip introductory overview paragraphs if longer than 150 chars
            if len(raw_text) > 150 and ("Management and Science University" in raw_text or "MSU" in raw_text):
                continue

            self._add_issue(
                start_line=line_no,
                end_line=line_no,
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MALFORMED_STRUCTURE,
                issue_code="STRAY_TEXT",
                programme="General",
                question_snippet="N/A (Orphan Text)",
                description=f"Stray token or unassociated text found outside Q&A blocks at line {line_no}.",
                raw_snippet=raw_text,
                suggested_action="Inspect source line to verify if it represents an incomplete edit or missing tag.",
            )

    def _validate_records_deterministic(self) -> None:
        """Deterministic checks on Question and Answer integrity."""
        for r in self.records:
            # Missing answer tag or empty answer
            if not r.has_answer_tag:
                self._add_issue(
                    start_line=r.start_line,
                    end_line=r.end_line,
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MISSING_FIELD,
                    issue_code="MISSING_ANSWER_TAG",
                    programme=r.programme_name,
                    question_snippet=r.question_clean,
                    description=f"Question at line {r.start_line} has no corresponding Answer tag before next block.",
                    raw_snippet=r.raw_text[:120],
                    suggested_action="Add explicit '**Answer:**' tag and verified answer text.",
                )
            elif not r.answer_clean:
                self._add_issue(
                    start_line=r.start_line,
                    end_line=r.end_line,
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MISSING_FIELD,
                    issue_code="EMPTY_ANSWER_CONTENT",
                    programme=r.programme_name,
                    question_snippet=r.question_clean,
                    description=f"Record at line {r.start_line} contains an Answer tag but empty content.",
                    raw_snippet=r.raw_text[:120],
                    suggested_action="Supply verified answer content for this question.",
                )

            # Missing question tag or empty question
            if not r.has_question_tag:
                self._add_issue(
                    start_line=r.start_line,
                    end_line=r.end_line,
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MISSING_FIELD,
                    issue_code="ORPHAN_ANSWER_TAG",
                    programme=r.programme_name,
                    question_snippet="N/A",
                    description=f"Answer tag at line {r.start_line} has no preceding Question tag.",
                    raw_snippet=r.raw_text[:120],
                    suggested_action="Supply preceding '**Question:**' block.",
                )
            elif not r.question_clean:
                self._add_issue(
                    start_line=r.start_line,
                    end_line=r.end_line,
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.MISSING_FIELD,
                    issue_code="EMPTY_QUESTION_CONTENT",
                    programme=r.programme_name,
                    question_snippet="N/A",
                    description=f"Record at line {r.start_line} contains a Question tag but empty content.",
                    raw_snippet=r.raw_text[:120],
                    suggested_action="Supply question text.",
                )

            # Check repetitive answer markers within a single record
            if r.answer_marker_count > 1:
                self._add_issue(
                    start_line=r.start_line,
                    end_line=r.end_line,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.MALFORMED_STRUCTURE,
                    issue_code="REPETITIVE_ANSWER_MARKERS",
                    programme=r.programme_name,
                    question_snippet=r.question_clean,
                    description=f"Record at line {r.start_line} contains {r.answer_marker_count} repeated Answer markers inside a single answer.",
                    raw_snippet=r.raw_text[:120],
                    suggested_action="Normalize sub-answers into bullet points or list items.",
                )

            # Check for stray dangling tokens on their own line inside the answer (e.g. 'cou')
            for line_offset, raw_l in enumerate(r.answer_raw.splitlines()):
                token_strip = raw_l.strip()
                if token_strip and len(token_strip) <= 5 and token_strip.isalpha() and not token_strip.isupper():
                    token_line = r.start_line + line_offset
                    self._add_issue(
                        start_line=token_line,
                        end_line=token_line,
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.MALFORMED_STRUCTURE,
                        issue_code="STRAY_TOKEN",
                        programme=r.programme_name,
                        question_snippet=r.question_clean,
                        description=f"Potential stray or orphaned token '{token_strip}' found on line {token_line}.",
                        raw_snippet=raw_l,
                        suggested_action="Review if this token is a typographical artifact or incomplete word.",
                    )

    def _validate_duplicates(self) -> None:
        """Detects exact and normalized duplicate questions."""
        exact_map: Dict[str, List[KBRecord]] = defaultdict(list)
        norm_map: Dict[str, List[KBRecord]] = defaultdict(list)

        for r in self.records:
            if r.question_clean:
                exact_map[r.question_clean].append(r)
                norm_q = normalize_question(r.question_clean)
                if norm_q:
                    norm_map[norm_q].append(r)

        # Flag exact duplicates
        flagged_exact_records: Set[str] = set()
        for q_text, rec_list in exact_map.items():
            if len(rec_list) > 1:
                locs = ", ".join(f"L{r.start_line}" for r in rec_list)
                for r in rec_list[1:]:  # Flag secondary occurrences
                    flagged_exact_records.add(r.record_id)
                    self._add_issue(
                        start_line=r.start_line,
                        end_line=r.end_line,
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.DUPLICATE_CONTENT,
                        issue_code="DUPLICATE_QUESTION_EXACT",
                        programme=r.programme_name,
                        question_snippet=r.question_clean,
                        description=f"Exact duplicate question found at multiple locations: {locs}.",
                        raw_snippet=r.raw_text[:120],
                        suggested_action="Review duplicate entries; deduplicate or differentiate programme contexts.",
                    )

        # Flag normalized duplicates (excluding those already flagged as exact)
        for norm_q, rec_list in norm_map.items():
            if len(rec_list) > 1:
                unflagged = [r for r in rec_list if r.record_id not in flagged_exact_records]
                if len(unflagged) > 1:
                    locs = ", ".join(f"L{r.start_line}" for r in rec_list)
                    for r in unflagged[1:]:
                        self._add_issue(
                            start_line=r.start_line,
                            end_line=r.end_line,
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.DUPLICATE_CONTENT,
                            issue_code="DUPLICATE_QUESTION_NORMALIZED",
                            programme=r.programme_name,
                            question_snippet=r.question_clean,
                            description=f"Normalized near-duplicate question found at: {locs}.",
                            raw_snippet=r.raw_text[:120],
                            suggested_action="Verify if these questions intend to ask the same thing in different wording.",
                        )

    def _validate_numerical_conflicts(self) -> None:
        """Identifies conflicting numerical amounts (e.g. tuition fees) in the same entry."""
        for r in self.records:
            if not r.answer_clean:
                continue

            fees = extract_rm_fees(r.answer_raw)
            # If multiple distinct fees appear in the same answer, inspect for conflicts
            distinct_fees = list(set(fees))
            if len(distinct_fees) > 1:
                # Filter out obvious ratios / instalments (e.g., if one fee is < 15,000 and the other is > 50,000)
                # But flag if two substantial amounts appear (e.g., 62,500 and 63,500)
                substantial_fees = [float(f) for f in distinct_fees if float(f) > 1000]
                if len(substantial_fees) >= 2:
                    # Check if they are close enough to be conflicting figures rather than semester breakdowns
                    max_f = max(substantial_fees)
                    min_f = min(substantial_fees)
                    diff_pct = (max_f - min_f) / max_f
                    if 0 < diff_pct < 0.20:  # Within 20% of each other: high probability of conflicting figures!
                        formatted_fees = ", ".join(f"RM{int(f):,}" for f in substantial_fees)
                        self._add_issue(
                            start_line=r.start_line,
                            end_line=r.end_line,
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.NUMERICAL_CONFLICT,
                            issue_code="CONFLICTING_TUITION_FEES",
                            programme=r.programme_name,
                            question_snippet=r.question_clean,
                            description=f"Conflicting fee amounts ({formatted_fees}) found within the same answer.",
                            raw_snippet=r.raw_text[:120],
                            suggested_action="Verify authoritative fee schedule with MSU finance registrar. Do not guess correct fee.",
                        )

    def _validate_programme_mismatches(self) -> None:
        """Detects when a question references a programme different from its section heading or answer."""
        degree_patterns = [
            "doctor of philosophy", "phd", "bachelor in", "bachelor of",
            "diploma in", "master in", "master of", "foundation in"
        ]

        for r in self.records:
            if not r.question_clean or not r.programme_name or r.programme_name == "General":
                continue

            prog_lower = r.programme_name.lower()
            q_lower = r.question_clean.lower()
            a_lower = r.answer_clean.lower()

            # Check if Question contains a degree pattern
            q_degree_matches = [d for d in degree_patterns if d in q_lower]
            prog_degree_matches = [d for d in degree_patterns if d in prog_lower]

            if q_degree_matches and prog_degree_matches:
                # If question explicitly names a degree level or discipline, check for mismatch
                # Example: Section is "Doctor of Philosophy (Food Service Technology)", but Q is "What is Doctor of Philosophy (Applied Science)?"
                # Extract subject in parentheses if present
                prog_paren = re.findall(r'\((.*?)\)', r.programme_name)
                q_paren = re.findall(r'\((.*?)\)', r.question_clean)

                if prog_paren and q_paren and prog_paren[0].lower() != q_paren[0].lower():
                    self._add_issue(
                        start_line=r.start_line,
                        end_line=r.end_line,
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.PROGRAMME_MISMATCH,
                        issue_code="SUSPICIOUS_PROGRAMME_MISMATCH",
                        programme=r.programme_name,
                        question_snippet=r.question_clean,
                        description=f"Question references '({q_paren[0]})' but section header specifies '({prog_paren[0]})'.",
                        raw_snippet=r.raw_text[:120],
                        suggested_action="Review if question was copied from another programme section without updating discipline name.",
                    )

    def _validate_time_sensitivity(self) -> None:
        """Identifies time-sensitive dates, years, or deadlines requiring human review."""
        for r in self.records:
            full_text = f"{r.question_clean} {r.answer_clean}"
            year_matches = set(YEAR_REGEX.findall(full_text))
            deadline_match = DEADLINE_KEYWORDS.search(full_text)

            if year_matches or deadline_match:
                details = []
                if year_matches:
                    details.append(f"years: {', '.join(sorted(year_matches))}")
                if deadline_match:
                    details.append(f"term: '{deadline_match.group(0)}'")

                detail_str = "; ".join(details)
                self._add_issue(
                    start_line=r.start_line,
                    end_line=r.end_line,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.TIME_SENSITIVE,
                    issue_code="TIME_SENSITIVE_CONTENT",
                    programme=r.programme_name,
                    question_snippet=r.question_clean,
                    description=f"Record references potentially time-sensitive information ({detail_str}).",
                    raw_snippet=r.raw_text[:120],
                    suggested_action="Verify validity for current academic year (intakes, fees, ranking dates).",
                )
