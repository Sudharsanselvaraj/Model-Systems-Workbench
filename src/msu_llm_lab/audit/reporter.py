import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from collections import Counter

from msu_llm_lab.audit.models import AuditIssue, AuditSummary, IssueSeverity


class KBReporter:
    """Generates machine-readable CSV logs, JSONL records, and readable Markdown summaries."""

    def __init__(self, summary: AuditSummary, issues: List[AuditIssue], output_dir: Path):
        self.summary = summary
        self.issues = issues
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, filename: str) -> Path:
        """Exports audit issues to a structured CSV file."""
        csv_path = self.output_dir / filename
        fieldnames = [
            "issue_id",
            "source_file",
            "record_id",
            "start_line",
            "end_line",
            "severity",
            "status",
            "category",
            "issue_code",
            "programme",
            "question_snippet",
            "description",
            "suggested_action",
            "requires_human_verification",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for issue in self.issues:
                writer.writerow({
                    "issue_id": issue.issue_id,
                    "source_file": Path(issue.source_file).name,
                    "record_id": issue.record_id or f"L{issue.start_line}-L{issue.end_line}",
                    "start_line": issue.start_line,
                    "end_line": issue.end_line,
                    "severity": issue.severity.value,
                    "status": issue.status.value,
                    "category": issue.category.value,
                    "issue_code": issue.issue_code,
                    "programme": issue.programme,
                    "question_snippet": issue.question_snippet,
                    "description": issue.description,
                    "suggested_action": issue.suggested_action,
                    "requires_human_verification": "YES" if issue.requires_human_verification else "NO",
                })
        return csv_path

    def export_jsonl(self, filename: str) -> Path:
        """Exports audit findings in machine-readable JSONL format."""
        jsonl_path = self.output_dir / filename
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for issue in self.issues:
                record = {
                    "finding_id": issue.issue_id,
                    "severity": issue.severity.value,
                    "classification": issue.classification.value,
                    "confidence_level": issue.confidence_level,
                    "requires_human_approval": issue.requires_human_approval,
                    "source_filename": Path(issue.source_file).name,
                    "record_id": issue.record_id or f"L{issue.start_line}-L{issue.end_line}",
                    "start_line": issue.start_line,
                    "end_line": issue.end_line,
                    "programme": issue.programme,
                    "category": issue.category.value,
                    "issue_code": issue.issue_code,
                    "evidence_excerpt": issue.raw_snippet,
                    "question_snippet": issue.question_snippet,
                    "explanation": issue.description,
                    "recommended_action": issue.suggested_action,
                    "status": issue.status.value,
                    "requires_official_verification": issue.status.value == "OFFICIAL_VERIFICATION_REQUIRED",
                }
                f.write(json.dumps(record) + "\n")
        return jsonl_path

    def export_markdown(self, filename: str) -> Path:
        """Exports an executive summary in Markdown format."""
        md_path = self.output_dir / filename
        source_name = Path(self.summary.source_file).name
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cat_counts = Counter(i.category.value for i in self.issues)
        sev_counts = Counter(i.severity.value for i in self.issues)

        deterministic_errors = [i for i in self.issues if i.severity == IssueSeverity.ERROR]
        high_priority_warnings = [
            i for i in self.issues
            if i.severity == IssueSeverity.WARNING and i.issue_code in (
                "CONFLICTING_TUITION_FEES", "SUSPICIOUS_PROGRAMME_MISMATCH"
            )
        ]

        lines = [
            f"# Knowledge Base Audit Report: `{source_name}`",
            "",
            f"**Generated**: {timestamp}  ",
            f"**Source Path**: `{self.summary.source_file}`  ",
            f"**File Size**: {self.summary.file_size_bytes:,} bytes ({self.summary.line_count:,} lines)  ",
            "",
            "---",
            "",
            "## 1. Source File Integrity Verification",
            "",
            "The knowledge base source file is strictly read-only and was verified cryptographically before and after the audit:",
            "",
            "| Check | SHA-256 Hash | Integrity Status |",
            "| :--- | :--- | :--- |",
            f"| **Pre-Audit** | `{self.summary.sha256_before}` | Baseline recorded |",
            f"| **Post-Audit** | `{self.summary.sha256_after}` | {'MATCH (Zero modifications)' if self.summary.integrity_verified else 'FAIL (Hash mismatch detected!)'} |",
            "",
            "---",
            "",
            "## 2. Reconciled Metrics Summary",
            "",
            "> [!NOTE]",
            "> **Counting Methodology Reconciliation**:",
            "> - **Raw Marker Counts**: Count of regex tag occurrences in the source text.",
            "> - **Parsed Records**: Structural Q&A entities extracted by the parser.",
            "> - Discrepancies between marker counts and parsed records arise from multi-marker answers (e.g., repeated `**Answer:**` tags within OR branches in entry requirements), missing tags, or non-standard formatting.",
            "",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **Raw Question Markers** | {self.summary.raw_question_markers:,} | Total Question tags in file |",
            f"| **Raw Answer Markers** | {self.summary.raw_answer_markers:,} | Total Answer tags in file |",
            f"| **Parsed Q&A Records** | {self.summary.parsed_records_count:,} | Structured entities parsed |",
            f"| **Deterministic Errors** | {sev_counts['ERROR']:,} | Missing answers, empty fields, malformed syntax |",
            f"| **Heuristic Warnings** | {sev_counts['WARNING']:,} | Duplicates, conflicting fees, mismatches, dates |",
            f"| **Total Issues Flagged** | {len(self.issues):,} | Requiring review / verification |",
            "",
            "---",
            "",
            "## 3. Issue Breakdown by Category",
            "",
            "| Issue Category | Error Count | Warning Count | Total |",
            "| :--- | :--- | :--- | :--- |",
        ]

        all_categories = sorted(list(set(i.category.value for i in self.issues)))
        for cat in all_categories:
            err_c = sum(1 for i in self.issues if i.category.value == cat and i.severity == IssueSeverity.ERROR)
            warn_c = sum(1 for i in self.issues if i.category.value == cat and i.severity == IssueSeverity.WARNING)
            lines.append(f"| `{cat}` | {err_c:,} | {warn_c:,} | {err_c + warn_c:,} |")

        lines.extend([
            "",
            "---",
            "",
            "## 4. Deterministic Errors (Immediate Attention Required)",
            "",
        ])

        if deterministic_errors:
            lines.extend([
                "| Line | Issue Code | Programme | Question / Field | Description |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for err in deterministic_errors[:20]:
                q_disp = err.question_snippet.replace("|", "/")
                lines.append(
                    f"| L{err.start_line} | `{err.issue_code}` | {err.programme[:30]} | {q_disp[:40]} | {err.description} |"
                )
            if len(deterministic_errors) > 20:
                lines.append(f"*(... and {len(deterministic_errors) - 20} more errors recorded in CSV)*")
        else:
            lines.append("No deterministic errors detected.")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Critical Heuristic Warnings (Fees & Programme Mismatches)",
            "",
        ])

        if high_priority_warnings:
            lines.extend([
                "| Line | Issue Code | Programme | Question Snippet | Conflict / Mismatch Description |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ])
            for warn in high_priority_warnings:
                q_disp = warn.question_snippet.replace("|", "/")
                lines.append(
                    f"| L{warn.start_line} | `{warn.issue_code}` | {warn.programme[:30]} | {q_disp[:40]} | {warn.description} |"
                )
        else:
            lines.append("No conflicting fees or programme mismatches detected.")

        lines.extend([
            "",
            "---",
            "",
            "## 6. Audit Invariants & Human Review Policy",
            "",
            "> [!IMPORTANT]",
            "> **Non-Negotiable Research Policy**:",
            "> 1. **Zero Automatic Fixes**: The audit tool never automatically determines which conflicting tuition fee, duration, or entry requirement is correct.",
            "> 2. **Authoritative Verification**: All flagged discrepancies are exported to CSV for verification against official MSU registrar records.",
            "> 3. **Source Immutability**: The original knowledge base files remain untouched.",
            "",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return md_path
