"""Safe, repeatable remediation engine for MSU knowledge-base content defects."""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from msu_llm_lab.audit.parser import KBParser
from msu_llm_lab.audit.validator import KBValidator
from msu_llm_lab.remediation.models import (
    RemediationFinding,
    RemediationClassification,
    RemediationStatus,
    RemediationReport,
)
from msu_llm_lab.remediation.registry import get_default_remediation_findings


def _compute_sha256(content: str) -> str:
    """Computes SHA-256 hash of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a local file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class RemediationEngine:
    """Executes validated, versioned remediations while guaranteeing source file immutability."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        if not self.source_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {source_path}")

        self.sha256_initial = _compute_file_sha256(self.source_path)
        with open(self.source_path, "r", encoding="utf-8") as f:
            self.original_content = f.read()

        self.findings: List[RemediationFinding] = get_default_remediation_findings(str(self.source_path))

    def evaluate_plan(self) -> RemediationReport:
        """Evaluates findings and prepares a remediation execution plan."""
        approved = [f for f in self.findings if f.status == RemediationStatus.APPROVED_FOR_RELEASE]
        pending_verif = [f for f in self.findings if f.status == RemediationStatus.OFFICIAL_VERIFICATION_PENDING]
        pending_inv = [f for f in self.findings if f.status == RemediationStatus.INVESTIGATION_PENDING]
        escalated = [f for f in self.findings if f.status == RemediationStatus.KB_OWNER_ESCALATED]

        return RemediationReport(
            source_file=str(self.source_path),
            sha256_original=self.sha256_initial,
            sha256_remediated=None,
            total_findings=len(self.findings),
            approved_count=len(approved),
            pending_official_verification_count=len(pending_verif),
            pending_investigation_count=len(pending_inv),
            escalated_kb_owner_count=len(escalated),
            applied_count=0,
            is_dry_run=True,
            findings=self.findings,
        )

    def execute_remediation(
        self,
        output_path: Optional[Path] = None,
        dry_run: bool = True,
    ) -> Tuple[RemediationReport, str]:
        """Applies approved remediations in-memory or to a separate versioned destination.
        
        CRITICAL INVARIANT: The original source file is never modified under any circumstances.
        """
        # Ensure source file has not been altered
        pre_sha = _compute_file_sha256(self.source_path)
        if pre_sha != self.sha256_initial:
            raise RuntimeError(
                f"Source file {self.source_path} hash changed during session! Immutability violated."
            )

        working_content = self.original_content
        applied_findings = []

        # Only apply findings that are explicitly APPROVED_FOR_RELEASE
        for finding in self.findings:
            if finding.status == RemediationStatus.APPROVED_FOR_RELEASE and finding.proposed_text:
                # Target exact string replacement
                if finding.finding_id == "REM-001":
                    target = "\\*\\*Question:\\*\\* What types of jobs/careers can I expect to get with a  \nBachelor in Accounting?"
                    replacement = "\\*\\*Question:\\*\\* What types of jobs/careers can I expect to get with a  \nBachelor in Broadcasting (Hons)?"
                elif finding.finding_id == "REM-002":
                    target = "\\*\\*Question:\\*\\* How long is Bachelor of New Media (Hons)?"
                    replacement = "\\*\\*Question:\\*\\* How long is Bachelor of Product Design (Honours)?"
                elif finding.finding_id == "REM-003":
                    target = "\\*\\*Question:\\*\\* What is the Master of Education (TESL)?   "
                    replacement = "\\*\\*Question:\\*\\* What is the Master of Design?   "
                elif finding.finding_id == "REM-004":
                    target = "\\*\\*Question:\\*\\* What is the Doctor of Philosophy (Applied Science)?  "
                    replacement = "\\*\\*Question:\\*\\* What is the Doctor of Philosophy (Food Service Technology)?  "
                else:
                    target = ""
                    replacement = ""


                if target and target in working_content:
                    working_content = working_content.replace(target, replacement, 1)
                    applied_findings.append(finding)
                    finding.status = RemediationStatus.APPLIED

        sha_remediated = _compute_sha256(working_content)

        # Write to separate versioned destination if not dry-run
        if not dry_run and output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(working_content)

        # Verify that original source file remains pristine
        post_sha = _compute_file_sha256(self.source_path)
        if post_sha != self.sha256_initial:
            raise RuntimeError(
                f"Source file {self.source_path} was modified! Hash mismatch detected."
            )

        report = RemediationReport(
            source_file=str(self.source_path),
            sha256_original=self.sha256_initial,
            sha256_remediated=sha_remediated,
            total_findings=len(self.findings),
            approved_count=len([f for f in self.findings if f.status in (RemediationStatus.APPROVED_FOR_RELEASE, RemediationStatus.APPLIED)]),
            pending_official_verification_count=len([f for f in self.findings if f.status == RemediationStatus.OFFICIAL_VERIFICATION_PENDING]),
            pending_investigation_count=len([f for f in self.findings if f.status == RemediationStatus.INVESTIGATION_PENDING]),
            escalated_kb_owner_count=len([f for f in self.findings if f.status == RemediationStatus.KB_OWNER_ESCALATED]),
            applied_count=len(applied_findings),
            is_dry_run=dry_run,
            findings=self.findings,
        )

        return report, working_content
