"""CLI entry point for msu-llm-lab."""

import hashlib
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from msu_llm_lab.audit.models import AuditSummary, IssueSeverity
from msu_llm_lab.audit.parser import KBParser
from msu_llm_lab.audit.reporter import KBReporter
from msu_llm_lab.audit.validator import KBValidator
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.models.catalog import get_portfolio

import sys

app = typer.Typer(
    name="msu-llm-lab",
    help="MSU Admissions LLM Research Workbench: evaluation, audit, and benchmark CLI.",
    no_args_is_help=False,
)
budget_app = typer.Typer(
    name="budget",
    help="Budget ledger, ceilings, and spend tracking.",
    no_args_is_help=True,
)
models_app = typer.Typer(
    name="models",
    help="Model portfolio catalog and validation.",
    no_args_is_help=True,
)

app.add_typer(budget_app, name="budget")
app.add_typer(models_app, name="models")

console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch interactive terminal UI.",
    ),
) -> None:
    """MSU Admissions LLM Research Workbench."""
    if ctx.invoked_subcommand is None:
        if interactive or sys.stdin.isatty():
            from msu_llm_lab.tui.app import run_tui
            run_tui()
        else:
            console.print(ctx.get_help())


@app.command(name="ui")
def launch_ui() -> None:
    """Launch the interactive terminal UI (welcome screen, model catalogue, budget)."""
    from msu_llm_lab.tui.app import run_tui
    run_tui()


@app.command(name="interactive", hidden=True)
def launch_interactive() -> None:
    """Alias to launch the interactive terminal UI."""
    from msu_llm_lab.tui.app import run_tui
    run_tui()


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Knowledge Base Audit Command (Phase 1)
# ---------------------------------------------------------------------------


@app.command(name="audit-kb")
def audit_kb(
    file_path: Path = typer.Argument(
        ...,
        help="Path to the knowledge base Markdown file to audit.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("reports"),
        "--output-dir",
        "-o",
        help="Directory to save audit CSV and Markdown reports.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run audit in memory and display summary without writing reports to disk.",
    ),
    report_format: str = typer.Option(
        "both",
        "--format",
        "-f",
        help="Report export format: 'both', 'csv', or 'md'.",
    ),
) -> None:
    """Audit a knowledge base Markdown file for structure, syntax, and content integrity."""
    console.print(Panel(f"[bold cyan]Knowledge Base Audit[/bold cyan]: {file_path.name}", expand=False))

    # 1. Pre-audit SHA-256 verification
    sha_before = compute_sha256(file_path)
    file_size = file_path.stat().st_size

    # 2. Parse source file
    parser = KBParser(file_path)
    q_markers, a_markers, records, headings, orphan_lines = parser.parse()

    # 3. Validate parsed records
    validator = KBValidator(str(file_path), records, headings, orphan_lines)
    issues = validator.validate_all()

    # 4. Post-audit SHA-256 verification (guarantee zero mutation)
    sha_after = compute_sha256(file_path)
    integrity_ok = (sha_before == sha_after)

    if not integrity_ok:
        console.print("[bold red]CRITICAL INTEGRITY FAILURE: Source file hash changed during audit![/bold red]")
        raise typer.Exit(code=2)

    # 5. Build summary
    error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
    warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

    summary = AuditSummary(
        source_file=str(file_path.resolve()),
        file_size_bytes=file_size,
        line_count=len(parser.lines),
        sha256_before=sha_before,
        sha256_after=sha_after,
        integrity_verified=integrity_ok,
        raw_question_markers=q_markers,
        raw_answer_markers=a_markers,
        parsed_records_count=len(records),
        error_count=error_count,
        warning_count=warning_count,
        issues=issues,
    )

    # 6. Render terminal tables
    metric_table = Table(title="Audit Metrics Summary", show_header=True, header_style="bold magenta")
    metric_table.add_column("Metric", style="dim")
    metric_table.add_column("Value", justify="right")
    metric_table.add_column("Notes", style="italic")

    metric_table.add_row("Source SHA-256", f"{sha_before[:16]}...", "Pre- and post-audit match verified")
    metric_table.add_row("Total Lines", f"{len(parser.lines):,}", "Original line structure preserved")
    metric_table.add_row("Raw Question Markers", f"{q_markers:,}", "Regex hits across document")
    metric_table.add_row("Raw Answer Markers", f"{a_markers:,}", "Regex hits across document")
    metric_table.add_row("Parsed Q&A Records", f"{len(records):,}", "Structural entities extracted")
    metric_table.add_row(
        "Deterministic Errors",
        f"[bold red]{error_count}[/bold red]" if error_count else "[green]0[/green]",
        "Missing answers, empty fields, malformed syntax",
    )
    metric_table.add_row(
        "Heuristic Warnings",
        f"[bold yellow]{warning_count}[/bold yellow]" if warning_count else "[green]0[/green]",
        "Duplicates, conflicting fees, mismatches, dates",
    )
    console.print(metric_table)

    cat_table = Table(title="Issues Breakdown by Category", show_header=True, header_style="bold blue")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Errors", justify="right", style="red")
    cat_table.add_column("Warnings", justify="right", style="yellow")
    cat_table.add_column("Total", justify="right", style="bold")

    categories = sorted(list(set(i.category.value for i in issues)))
    for cat in categories:
        c_err = sum(1 for i in issues if i.category.value == cat and i.severity == IssueSeverity.ERROR)
        c_warn = sum(1 for i in issues if i.category.value == cat and i.severity == IssueSeverity.WARNING)
        cat_table.add_row(cat, str(c_err), str(c_warn), str(c_err + c_warn))
    console.print(cat_table)

    # 7. Write reports if not dry-run
    if dry_run:
        console.print("[yellow]Dry-run mode enabled: No report files written to disk.[/yellow]")
    else:
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = file_path.stem.replace(" ", "_").lower()
        reporter = KBReporter(summary, issues, output_dir)

        if report_format in ("both", "csv"):
            csv_file = reporter.export_csv(f"audit_issues_{safe_name}_{timestamp_slug}.csv")
            console.print(f"[green]✓ CSV issue log saved:[/green] {csv_file}")

        if report_format in ("both", "md"):
            md_file = reporter.export_markdown(f"audit_summary_{safe_name}_{timestamp_slug}.md")
            console.print(f"[green]✓ Markdown summary saved:[/green] {md_file}")

    console.print("[dim]Source file integrity verified: zero modifications made to source.[/dim]")


# ---------------------------------------------------------------------------
# Knowledge Base Remediation Subsystem (Step 2)
# ---------------------------------------------------------------------------


@app.command(name="remediate-kb")
def remediate_kb(
    file_path: Path = typer.Argument(
        Path("Copy of MSU Programmes KB - Local - Sudharsan.md"),
        help="Path to the source knowledge base Markdown file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path for versioned clean copy (e.g., data/remediated/MSU_Programmes_KB_v1.1_clean.md).",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Write remediated clean copy to output destination. Defaults to dry-run.",
    ),
) -> None:
    """Inspect defect findings, evaluate remediation rules, and produce versioned clean copies."""
    from msu_llm_lab.remediation.engine import RemediationEngine
    from msu_llm_lab.remediation.models import RemediationStatus

    console.print(Panel(f"[bold cyan]MSU Knowledge Base Remediation Engine[/bold cyan]: {file_path.name}", expand=False))
    console.print("[yellow]SAFETY INVARIANT: Original knowledge-base source files remain 100% read-only and immutable.[/yellow]\n")

    engine = RemediationEngine(file_path)
    dry_run = not apply

    target_out = output_path or Path("data/remediated") / f"{file_path.stem}_v1.1_clean.md"
    report, _ = engine.execute_remediation(output_path=target_out if apply else None, dry_run=dry_run)

    # 1. Findings Overview Table
    table = Table(title="Remediation Findings & Action Classification", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="bold cyan", width=9)
    table.add_column("Programme", style="white", width=26)
    table.add_column("Classification", style="dim", width=22)
    table.add_column("Status", justify="center", width=24)
    table.add_column("Action / Resolution", style="italic")

    status_styles = {
        RemediationStatus.APPROVED_FOR_RELEASE: "[bold green]APPROVED_FOR_RELEASE[/bold green]",
        RemediationStatus.APPLIED: "[bold green]APPLIED (CLEAN COPY)[/bold green]",
        RemediationStatus.OFFICIAL_VERIFICATION_PENDING: "[bold yellow]OFFICIAL_VERIF_PENDING[/bold yellow]",
        RemediationStatus.INVESTIGATION_PENDING: "[bold cyan]INVESTIGATION_PENDING[/bold cyan]",
        RemediationStatus.KB_OWNER_ESCALATED: "[bold magenta]KB_OWNER_ESCALATED[/bold magenta]",
    }

    for f in report.findings:
        st_label = status_styles.get(f.status, str(f.status.value))
        table.add_row(f.finding_id, f.programme_name[:26], f.classification.value, st_label, f.resolution_notes or f.issue_title)

    console.print(table)

    # 2. Summary Card
    mode_text = "[bold green]LIVE VERSIONED WRITE[/bold green]" if apply else "[bold yellow]DRY-RUN (In-Memory Evaluation)[/bold yellow]"
    summary_card = (
        f"Execution Mode:       {mode_text}\n"
        f"Source SHA-256:       {report.sha256_original[:16]}... (Verified 100% Immutable)\n"
        f"Total Findings:       {report.total_findings}\n"
        f"Approved for Clean:   {report.approved_count} (Confirmed Content Defects)\n"
        f"Official Verif Req:   {report.pending_official_verification_count} (Contradictory Fees / Intakes — Zero Guessing)\n"
        f"KB Owner Escalations: {report.escalated_kb_owner_count} (Curriculum / Language Standards)\n"
    )
    if apply and output_path:
        summary_card += f"Output Destination:   {target_out}\nRemediated SHA-256:   {report.sha256_remediated[:16]}...\n"
    elif dry_run:
        summary_card += f"\n[dim]Run with '--apply --output {target_out}' to generate versioned clean copy.[/dim]"

    console.print(Panel(summary_card, title="[bold]Remediation Execution Summary[/bold]", border_style="cyan"))



# ---------------------------------------------------------------------------
# Evaluation Dataset Subsystem (Phase 2)
# ---------------------------------------------------------------------------


@app.command(name="validate-dataset")
def validate_dataset(
    dataset_path: Path = typer.Argument(
        Path("data/evaluation/synthetic_seed_eval.jsonl"),
        help="Path to the JSONL evaluation dataset file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    check_leakage: bool = typer.Option(
        True,
        "--check-leakage/--no-check-leakage",
        help="Perform cross-split (train/dev/test) leakage and n-gram overlap detection.",
    ),
) -> None:
    """Validate evaluation dataset schemas, confidentiality boundaries, and cross-split leakage."""
    from msu_llm_lab.dataset.manager import DatasetManager

    console.print(Panel(f"[bold cyan]Dataset Validation[/bold cyan]: {dataset_path.name}", expand=False))

    manager = DatasetManager()
    try:
        cases = manager.load_jsonl(dataset_path)
    except Exception as e:
        console.print(f"[bold red]Validation Failure:[/bold red] Failed to load dataset: {e}")
        raise typer.Exit(code=1)

    summary = manager.validate_dataset(cases, source_path=str(dataset_path), check_leakage=check_leakage)

    # 1. Dataset Overview Table
    table = Table(title="Dataset Partition & Confidentiality Overview", show_header=True, header_style="bold magenta")
    table.add_column("Partition Split", style="cyan")
    table.add_column("Case Count", justify="right")
    table.add_column("External API Status", justify="center")
    table.add_column("Notes", style="italic")

    table.add_row("TRAIN", str(summary.train_count), "[green]Internal Model Training / Tuning[/green]", "Isolated partition")
    table.add_row("DEV (Validation)", str(summary.dev_count), "[green]Hyperparameter & Prompt Dev[/green]", "Isolated partition")
    table.add_row("TEST (Evaluation)", str(summary.test_count), "[green]Final Benchmark Evaluation[/green]", "Held-out partition")
    if summary.unassigned_count > 0:
        table.add_row("UNASSIGNED", str(summary.unassigned_count), "[yellow]Pending Stratification[/yellow]", "Not yet split")
    console.print(table)

    # 2. Privacy & Governance Breakdown Table
    gov_table = Table(title="Confidentiality & Provenance Status", show_header=True, header_style="bold blue")
    gov_table.add_column("Metric", style="dim")
    gov_table.add_column("Count", justify="right")
    gov_table.add_column("Governance Policy", style="italic")

    gov_table.add_row("Eligible for External Dispatch", f"[green]{summary.external_eligible_count}[/green]", "PUBLIC or SYNTHETIC records")
    gov_table.add_row("Blocked from Outbound Network", f"[bold yellow]{summary.blocked_count}[/bold yellow]", "INTERNAL, CONFIDENTIAL or UNCLASSIFIED")
    gov_table.add_row("Verified Ground Truths", str(summary.verification_breakdown.get("VERIFIED", 0)), "Explicit citation attached")
    gov_table.add_row("Unverified Ground Truths", str(summary.verification_breakdown.get("UNVERIFIED", 0)), "Requires domain expert review")
    gov_table.add_row("Disputed Ground Truths", str(summary.verification_breakdown.get("DISPUTED", 0)), "Contradictory facts flagged")
    console.print(gov_table)

    # 3. Leakage Findings
    if summary.leakage_report and summary.leakage_report.has_leakage:
        leak_table = Table(title="[bold red]Contamination Alert: Cross-Split Leakage Detected[/bold red]", show_header=True, header_style="bold red")
        leak_table.add_column("Case A (Split)", style="yellow")
        leak_table.add_column("Case B (Split)", style="red")
        leak_table.add_column("Type", style="bold")
        leak_table.add_column("Similarity", justify="right")
        leak_table.add_column("Query Snippet", style="italic")

        for f in summary.leakage_report.findings:
            leak_table.add_row(
                f"{f.case_a_id} ({f.split_a.value})",
                f"{f.case_b_id} ({f.split_b.value})",
                f.leakage_type.value,
                f"{f.similarity_score:.2f}",
                f.query_a[:40] + ("..." if len(f.query_a) > 40 else ""),
            )
        console.print(leak_table)

    # 4. Final Verdict
    if summary.is_valid:
        console.print(Panel(
            f"[bold green]✓ DATASET VALIDATION PASSED[/bold green]\n"
            f"Total Cases: {summary.total_cases} | Zero Leakage Detected | Confidentiality Invariants Satisfied",
            border_style="green",
        ))
    else:
        err_msg = "\n".join(f"• {e}" for e in summary.validation_errors)
        console.print(Panel(
            f"[bold red]✗ DATASET VALIDATION FAILED[/bold red]\n{err_msg}",
            border_style="red",
        ))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Budget Subcommands
# ---------------------------------------------------------------------------


@budget_app.command(name="status")
def budget_status() -> None:
    """Display current spend ledger, remaining balances, and local policy ceilings."""
    service = BudgetService()
    status = service.get_status()

    console.print(Panel("[bold cyan]MSU LLM Inference Workbench — Budget Status[/bold cyan]", expand=False))
    console.print("[yellow]NOTE: Ceilings below are local experiment tracking policies, NOT provider-enforced account limits.[/yellow]\n")

    table = Table(title="Spend & Policy Overview", show_header=True, header_style="bold magenta")
    table.add_column("Budget Metric", style="dim")
    table.add_column("Configured Ceiling", justify="right")
    table.add_column("Current Spend", justify="right")
    table.add_column("Remaining Balance", justify="right", style="bold")
    table.add_column("Status", justify="center")

    pilot_status_str = "[bold red]EXHAUSTED[/bold red]" if status.is_pilot_exhausted else "[green]ACTIVE[/green]"
    total_status_str = "[bold red]EXHAUSTED[/bold red]" if status.is_total_exhausted else "[green]ACTIVE[/green]"

    table.add_row(
        "Initial Pilot Budget",
        f"${status.pilot_ceiling_usd:.4f}",
        f"${status.pilot_spend_usd:.4f}",
        f"${status.remaining_pilot_usd:.4f}",
        pilot_status_str,
    )
    table.add_row(
        "Total Experiment Ceiling",
        f"${status.total_ceiling_usd:.4f}",
        f"${status.cumulative_spend_usd:.4f}",
        f"${status.remaining_total_usd:.4f}",
        total_status_str,
    )
    console.print(table)

    rule_table = Table(title="Per-Execution Safety Rules", show_header=True, header_style="bold blue")
    rule_table.add_column("Safety Rule", style="cyan")
    rule_table.add_column("Limit (USD)", justify="right")
    rule_table.add_column("Enforcement Action", style="italic")

    rule_table.add_row("Max Single Run Spend", f"${status.per_run_ceiling_usd:.4f}", "Preflight check rejects run if exceeded")
    rule_table.add_row("Max Single Case Spend", f"${status.per_case_ceiling_usd:.4f}", "Per-case circuit breaker halts run")
    rule_table.add_row("Execution Mode", "Dry-Run by Default", "Requires --live and explicit confirmation")
    console.print(rule_table)

    if status.recent_transactions:
        tx_table = Table(title="Recent Ledger Transactions", show_header=True, header_style="bold green")
        tx_table.add_column("Timestamp (UTC)", style="dim")
        tx_table.add_column("Model ID", style="cyan")
        tx_table.add_column("Provider", style="magenta")
        tx_table.add_column("Amount (USD)", justify="right")
        tx_table.add_column("Type")
        tx_table.add_column("Description", style="italic")

        for tx in status.recent_transactions:
            tx_type = "[yellow]ESTIMATE[/yellow]" if tx["is_preflight_estimate"] else "[bold green]ACTUAL[/bold green]"
            tx_table.add_row(
                tx["timestamp"][:19].replace("T", " "),
                tx["model_id"],
                tx["provider"],
                f"${Decimal(tx['amount_micros'])/Decimal('1000000'):.4f}",
                tx_type,
                tx["description"],
            )
        console.print(tx_table)
    else:
        console.print("[dim]No paid transactions recorded yet. Spend ledger is at baseline ($0.0000).[/dim]")


# ---------------------------------------------------------------------------
# Models Subcommands
# ---------------------------------------------------------------------------


@models_app.command(name="list")
def models_list() -> None:
    """List all candidate models in the evaluation portfolio with status and pricing."""
    candidates = get_portfolio()

    console.print(Panel("[bold cyan]MSU LLM Inference Workbench — Candidate Portfolio[/bold cyan]", expand=False))
    console.print(
        "[yellow]NOTE: Model IDs and pricing below are preliminary entries. "
        "They have NOT yet been validated against OpenRouter live API (scheduled for Step 2).[/yellow]\n"
    )

    table = Table(title="Evaluation Model Candidates", show_header=True, header_style="bold magenta")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="bold")
    table.add_column("Deployment Mode", style="blue")
    table.add_column("Est. $/1M (In / Out)", justify="right")
    table.add_column("License", style="dim")
    table.add_column("Verification Status", justify="center")
    table.add_column("Experimental Role", style="italic")

    for m in candidates:
        status_disp = f"[bold yellow]{m.verification_status}[/bold yellow]"
        pricing_str = f"${m.input_price_per_1m_usd:.2f} / ${m.output_price_per_1m_usd:.2f}"
        table.add_row(
            m.model_id,
            m.display_name,
            m.deployment_mode,
            pricing_str,
            m.license,
            status_disp,
            m.experimental_purpose,
        )

    console.print(table)
    console.print("\n[dim]Hosted OpenRouter runs are distinct from self-hosted vLLM serving. Every evaluation record logs its deployment mode.[/dim]")


if __name__ == "__main__":
    app()
