"""Unit tests for the knowledge base remediation and patch subsystem."""

from pathlib import Path
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.remediation.engine import RemediationEngine
from msu_llm_lab.remediation.models import (
    RemediationClassification,
    RemediationStatus,
)
from msu_llm_lab.remediation.registry import get_default_remediation_findings

runner = CliRunner()
PROGRAMMES_KB = Path("Copy of MSU Programmes KB - Local - Sudharsan.md")


def test_remediation_findings_registry():
    """Verify curated remediation findings correctly classify defects and verification requirements."""
    findings = get_default_remediation_findings(str(PROGRAMMES_KB))
    findings_map = {f.finding_id: f for f in findings}

    # REM-001 to REM-004: Confirmed Content Defects (Approved)
    assert "REM-001" in findings_map
    assert findings_map["REM-001"].classification == RemediationClassification.CONFIRMED_CONTENT_DEFECT
    assert findings_map["REM-001"].status == RemediationStatus.APPROVED_FOR_RELEASE
    assert "Broadcasting" in findings_map["REM-001"].proposed_text

    assert "REM-002" in findings_map
    assert findings_map["REM-002"].classification == RemediationClassification.CONFIRMED_CONTENT_DEFECT
    assert findings_map["REM-002"].status == RemediationStatus.APPROVED_FOR_RELEASE
    assert "Product Design" in findings_map["REM-002"].proposed_text

    assert "REM-003" in findings_map
    assert findings_map["REM-003"].classification == RemediationClassification.CONFIRMED_CONTENT_DEFECT
    assert findings_map["REM-003"].status == RemediationStatus.APPROVED_FOR_RELEASE
    assert "Master of Design" in findings_map["REM-003"].proposed_text

    assert "REM-004" in findings_map
    assert findings_map["REM-004"].classification == RemediationClassification.CONFIRMED_CONTENT_DEFECT
    assert findings_map["REM-004"].status == RemediationStatus.APPROVED_FOR_RELEASE
    assert "Food Service Technology" in findings_map["REM-004"].proposed_text

    # REM-005: Contradictory fee must NOT be arbitrarily guessed
    assert "REM-005" in findings_map
    assert findings_map["REM-005"].classification == RemediationClassification.CONTRADICTORY_FACT
    assert findings_map["REM-005"].status == RemediationStatus.OFFICIAL_VERIFICATION_PENDING
    assert findings_map["REM-005"].proposed_text is None
    assert findings_map["REM-005"].requires_official_verification is True


def test_remediation_engine_dry_run_preserves_source_hash():
    """Verify dry-run in-memory transformation strictly preserves original file immutability."""
    engine = RemediationEngine(PROGRAMMES_KB)
    initial_sha = engine.sha256_initial

    report, remediated_text = engine.execute_remediation(dry_run=True)

    assert report.is_dry_run is True
    assert report.sha256_original == initial_sha
    assert engine.sha256_initial == initial_sha
    assert report.applied_count == 4

    # Confirmed defect replacements took place in remediated text
    assert "What types of jobs/careers can I expect to get with a  \nBachelor in Broadcasting (Hons)?" in remediated_text
    assert "How long is Bachelor of Product Design (Honours)?" in remediated_text
    assert "What is the Master of Design?   " in remediated_text
    assert "What is the Doctor of Philosophy (Food Service Technology)?  " in remediated_text

    # Contradictory fee remained flagged but unpatched
    assert "RM 62,500" in remediated_text
    assert "RM63,500" in remediated_text


def test_remediation_cli_command():
    """Verify CLI 'remediate-kb' command runs in dry-run mode and displays findings table."""
    res = runner.invoke(app, ["remediate-kb", str(PROGRAMMES_KB)], env={"COLUMNS": "120"})
    assert res.exit_code == 0
    assert "MSU Knowledge Base Remediation Engine" in res.stdout
    assert "REM-001" in res.stdout
    assert "CONFIRMED_CONTENT" in res.stdout
    assert "APPLIED (CLEAN COPY)" in res.stdout
    assert "OFFICIAL_VERIF_PENDING" in res.stdout
    assert "DRY-RUN (In-Memory Evaluation)" in res.stdout
    assert "Verified 100% Immutable" in res.stdout

