"""Comprehensive test suite for Knowledge Base parser, validator, reporter, and CLI."""

import hashlib
from pathlib import Path
import pytest
from typer.testing import CliRunner

from msu_llm_lab.audit.models import IssueSeverity, IssueCategory
from msu_llm_lab.audit.parser import KBParser, clean_text, unescape_markdown
from msu_llm_lab.audit.validator import KBValidator
from msu_llm_lab.audit.reporter import KBReporter
from msu_llm_lab.cli import app, compute_sha256

runner = CliRunner()


def test_unescape_and_clean_text():
    """Verify markdown escape removal and whitespace collapsing."""
    raw = r"\*\*Question:\*\* What is MSU\'s address\? Visit \[MSU\]\(http\:\/\/msu\.edu\.my\)\."
    cleaned = clean_text(raw)
    assert r"\*" not in cleaned
    assert r"\:" not in cleaned
    assert "What is MSU's address? Visit [MSU](http://msu.edu.my)." in cleaned


def test_valid_single_turn_and_multiline_record(tmp_kb_dir: Path):
    """Verify clean parsing and validation of well-formed Q&A entries."""
    content = (
        "# \\# General Information about MSU\n\n"
        "## \\#\\# Legitimacy\n\n"
        "\\*\\*Question:\\*\\* Is this MSU’s official portal?\n\n"
        "\\*\\*Answer:\\*\\* Yes, this is the official admissions portal.\n\n"
        "## \\#\\# Semester Breakdown\n\n"
        "\\*\\*Question:\\*\\* How long is each semester?\n\n"
        "\\*\\*Answer:\\*\\* Instruction: 14 weeks.\n"
        "Exams: 2 weeks.\n"
    )
    kb_file = tmp_kb_dir / "valid_sample.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    q_markers, a_markers, records, headings, orphans = parser.parse()

    assert q_markers == 2
    assert a_markers == 2
    assert len(records) == 2
    assert len(headings) == 3
    assert len(orphans) == 0

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
    assert len(errors) == 0


def test_missing_answer_detection(tmp_kb_dir: Path):
    """Verify deterministic detection of questions with missing answers."""
    content = (
        "# \\# Admissions\n\n"
        "\\*\\*Question:\\*\\* What is the Affin Education Financing-I?\n\n"
        "\\*\\*Question:\\*\\* What is PTPTN?\n\n"
        "\\*\\*Answer:\\*\\* PTPTN is the National Higher Education Fund.\n"
    )
    kb_file = tmp_kb_dir / "missing_answer.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()
    assert len(records) == 2
    assert not records[0].has_answer_tag
    assert records[1].has_answer_tag

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    missing_ans_issues = [i for i in issues if i.issue_code == "MISSING_ANSWER_TAG"]
    assert len(missing_ans_issues) == 1
    assert missing_ans_issues[0].severity == IssueSeverity.ERROR
    assert "Affin Education Financing-I" in missing_ans_issues[0].question_snippet


def test_empty_question_and_empty_answer(tmp_kb_dir: Path):
    """Verify detection of empty question and empty answer tags."""
    content = (
        "# \\# General\n\n"
        "\\*\\*Question:\\*\\* \n\n"
        "\\*\\*Answer:\\*\\* Some answer text.\n\n"
        "\\*\\*Question:\\*\\* What are the discount requirements?\n\n"
        "\\*\\*Answer:\\*\\* \n\n"
    )
    kb_file = tmp_kb_dir / "empty_fields.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    empty_q = [i for i in issues if i.issue_code == "EMPTY_QUESTION_CONTENT"]
    empty_a = [i for i in issues if i.issue_code == "EMPTY_ANSWER_CONTENT"]

    assert len(empty_q) == 1
    assert empty_q[0].severity == IssueSeverity.ERROR
    assert len(empty_a) == 1
    assert empty_a[0].severity == IssueSeverity.ERROR


def test_duplicate_questions_exact_and_normalized(tmp_kb_dir: Path):
    """Verify heuristic warning for exact and normalized duplicate questions."""
    content = (
        "# \\# Careers\n\n"
        "\\*\\*Question:\\*\\* What is the job demand for English language graduates in Malaysia?\n\n"
        "\\*\\*Answer:\\*\\* Demand is high in media and education.\n\n"
        "\\*\\*Question:\\*\\* What is the job demand for English language graduates in Malaysia?\n\n"
        "\\*\\*Answer:\\*\\* Demand is high in media and education.\n\n"
        "\\*\\*Question:\\*\\* What is the job demand for English language graduates in Malaysia???  \n\n"
        "\\*\\*Answer:\\*\\* High demand.\n"
    )
    kb_file = tmp_kb_dir / "duplicates.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    exact_dup = [i for i in issues if i.issue_code == "DUPLICATE_QUESTION_EXACT"]
    norm_dup = [i for i in issues if i.issue_code == "DUPLICATE_QUESTION_NORMALIZED"]

    assert len(exact_dup) >= 1
    assert all(i.severity == IssueSeverity.WARNING for i in exact_dup)
    assert len(norm_dup) >= 1


def test_conflicting_numbers_in_tuition_fees(tmp_kb_dir: Path):
    """Verify heuristic warning for contradictory fees within the same entry."""
    content = (
        "# \\# Doctor of Philosophy (Applied Science)\n\n"
        "\\*\\*Question:\\*\\* How much are the local fees for Doctor of Philosophy (Applied Science)?\n\n"
        "\\*\\*Answer:\\*\\* The fees for Doctor of Philosophy (Applied Science) is RM 62,500. "
        "Since the course duration is 2-6 years, you can equally divide RM63,500 by the amount of years.\n"
    )
    kb_file = tmp_kb_dir / "conflicting_fees.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    fee_conflicts = [i for i in issues if i.issue_code == "CONFLICTING_TUITION_FEES"]
    assert len(fee_conflicts) == 1
    assert fee_conflicts[0].severity == IssueSeverity.WARNING
    assert "RM62,500" in fee_conflicts[0].description or "RM 62,500" in fee_conflicts[0].raw_snippet
    assert fee_conflicts[0].requires_human_verification is True


def test_suspicious_programme_mismatch(tmp_kb_dir: Path):
    """Verify detection when question references a different programme than the section header."""
    content = (
        "# \\# Doctor of Philosophy (Food Service Technology)\n\n"
        "\\*\\*Question:\\*\\* What is the Doctor of Philosophy (Applied Science)?\n\n"
        "\\*\\*Answer:\\*\\* MSU's Doctor of Philosophy (Food Service Technology) is a research program.\n"
    )
    kb_file = tmp_kb_dir / "programme_mismatch.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    mismatches = [i for i in issues if i.issue_code == "SUSPICIOUS_PROGRAMME_MISMATCH"]
    assert len(mismatches) == 1
    assert mismatches[0].severity == IssueSeverity.WARNING
    assert "Food Service Technology" in mismatches[0].description
    assert "Applied Science" in mismatches[0].description


def test_time_sensitive_content_flagging(tmp_kb_dir: Path):
    """Verify heuristic warning for specific calendar years and deadline terms."""
    content = (
        "# \\# Ranking\n\n"
        "\\*\\*Question:\\*\\* What’s MSU’s QS ranking for 2027?\n\n"
        "\\*\\*Answer:\\*\\* MSU places #608 on the QS World University Rankings.\n\n"
        "\\*\\*Question:\\*\\* What is the application deadline?\n\n"
        "\\*\\*Answer:\\*\\* Applications close in July 2026.\n"
    )
    kb_file = tmp_kb_dir / "time_sensitive.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    time_issues = [i for i in issues if i.issue_code == "TIME_SENSITIVE_CONTENT"]
    assert len(time_issues) == 2
    assert all(i.category == IssueCategory.TIME_SENSITIVE for i in time_issues)


def test_malformed_syntax_and_stray_tokens(tmp_kb_dir: Path):
    """Verify detection of stray orphan tokens and empty headings."""
    content = (
        "# \\# General\n\n"
        "## \n\n"  # Empty heading
        "\\*\\*Question:\\*\\* When are opening hours?\n\n"
        "\\*\\*Answer:\\*\\* 9:00 AM to 6:00 PM.\n"
        "cou\n\n"  # Stray token
        "## \\#\\# Accreditation\n\n"
        "\\*\\*Question:\\*\\* Is MSU accredited?\n\n"
        "\\*\\*Answer:\\*\\* Yes, accredited by MQA.\n"
    )
    kb_file = tmp_kb_dir / "malformed.md"
    kb_file.write_text(content, encoding="utf-8")

    parser = KBParser(kb_file)
    _, _, records, headings, orphans = parser.parse()

    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    empty_hd = [i for i in issues if i.issue_code == "EMPTY_HEADING"]
    stray_tk = [i for i in issues if i.issue_code in ("STRAY_TEXT", "STRAY_TOKEN")]

    assert len(empty_hd) == 1
    assert empty_hd[0].severity == IssueSeverity.ERROR
    assert len(stray_tk) >= 1
    assert any("cou" in i.raw_snippet for i in stray_tk)


def test_source_immutability_and_hash_integrity(tmp_kb_dir: Path):
    """Verify that running the audit leaves the source file 100% unaltered."""
    content = (
        "# \\# Integrity Test\n\n"
        "\\*\\*Question:\\*\\* Is source data altered?\n\n"
        "\\*\\*Answer:\\*\\* Never.\n"
    )
    kb_file = tmp_kb_dir / "immutability.md"
    kb_file.write_text(content, encoding="utf-8")

    hash_before = compute_sha256(kb_file)

    # Execute parser, validator, and reporter
    parser = KBParser(kb_file)
    q_cnt, a_cnt, records, headings, orphans = parser.parse()
    validator = KBValidator(str(kb_file), records, headings, orphans)
    issues = validator.validate_all()

    # Re-compute hash
    hash_after = compute_sha256(kb_file)
    assert hash_before == hash_after
    assert kb_file.read_text(encoding="utf-8") == content


def test_cli_dry_run_and_report_generation(tmp_kb_dir: Path):
    """Test CLI audit-kb command in both dry-run and file-export modes."""
    content = (
        "# \\# CLI Test\n\n"
        "\\*\\*Question:\\*\\* Test Question?\n\n"
        "\\*\\*Answer:\\*\\* Test Answer.\n"
    )
    kb_file = tmp_kb_dir / "cli_sample.md"
    kb_file.write_text(content, encoding="utf-8")
    reports_dir = tmp_kb_dir / "reports"

    # Test Dry Run
    dry_result = runner.invoke(app, ["audit-kb", str(kb_file), "--dry-run", "-o", str(reports_dir)])
    assert dry_result.exit_code == 0
    assert "Dry-run mode enabled" in dry_result.stdout
    assert not reports_dir.exists()

    # Test Actual Report Generation
    run_result = runner.invoke(app, ["audit-kb", str(kb_file), "-o", str(reports_dir)])
    assert run_result.exit_code == 0
    assert "CSV issue log saved" in run_result.stdout
    assert "Markdown summary saved" in run_result.stdout

    csv_files = list(reports_dir.glob("*.csv"))
    md_files = list(reports_dir.glob("*.md"))
    assert len(csv_files) == 1
    assert len(md_files) == 1

    # Verify CSV has expected headers
    csv_text = csv_files[0].read_text(encoding="utf-8")
    assert "issue_id,source_file,start_line,end_line,severity" in csv_text
