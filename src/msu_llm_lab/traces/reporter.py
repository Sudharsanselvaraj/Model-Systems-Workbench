"""Reporter formatting diagnostic trace analyses into CSV and Markdown artifacts."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from msu_llm_lab.traces.models import TraceDiagnosticReport


class DiagnosticReporter:
    """Exports trace diagnostic hypotheses to CSV and Markdown reports."""

    @staticmethod
    def export_csv(report: TraceDiagnosticReport, output_path: Path) -> Path:
        """Writes diagnoses out to a structured CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trace_id",
                "failure_category",
                "confidence",
                "candidate_hypothesis",
                "evidence_summary",
                "suggested_action",
            ])
            for d in report.diagnoses:
                writer.writerow([
                    d.trace_id,
                    d.category.value,
                    f"{d.confidence:.2f}",
                    d.hypothesis,
                    " | ".join(d.evidence),
                    d.suggested_action,
                ])
        return output_path

    @staticmethod
    def export_markdown(report: TraceDiagnosticReport, output_path: Path) -> Path:
        """Writes a human-readable Markdown summary report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "# MSU LLM Workbench — Trace Diagnostic Report",
            "",
            f"**Generated At**: `{now_str}`  ",
            f"**Total Analyzed Traces**: `{report.total_traces}`  ",
            f"**Identified Candidate Failures**: `{report.diagnosed_failures}`  ",
            f"**Mean Trace Latency**: `{report.mean_latency_ms:.1f} ms`  ",
            "",
            "---",
            "",
            "## Failure Distribution",
            "",
            "| Failure Category | Candidate Incidents | % of Total |",
            "|---|---|---|",
        ]

        total = max(1, report.total_traces)
        for cat, count in report.category_counts.items():
            pct = (count / total) * 100
            lines.append(f"| `{cat}` | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "---",
            "",
            "## Detailed Candidate Hypotheses",
            "",
            "> [!NOTE]",
            "> All findings are candidate diagnostic hypotheses derived from heuristics and telemetry,",
            "> and must be reviewed prior to deploying system prompt or router modifications.",
            "",
        ])

        for idx, d in enumerate(report.diagnoses, 1):
            lines.extend([
                f"### {idx}. Trace `{d.trace_id}` — {d.category.value} (Confidence: {d.confidence:.0%})",
                f"- **Hypothesis**: {d.hypothesis}",
                f"- **Suggested Action**: {d.suggested_action}",
                "- **Evidence**:",
            ])
            for ev in d.evidence:
                lines.append(f"  - {ev}")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
