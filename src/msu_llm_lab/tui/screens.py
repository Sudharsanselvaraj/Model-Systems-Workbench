"""Screen renderers for the MSU LLM Workbench terminal interface."""

from decimal import Decimal
from typing import List, Optional
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from msu_llm_lab.core.budget import BudgetStatus
from msu_llm_lab.models.catalog import ModelCandidate
from msu_llm_lab.tui.banner import BANNER_EMBLEM, APP_TITLE, APP_VERSION, APP_SUBTITLE
from msu_llm_lab.tui.health import HealthCheckItem


MENU_ITEMS = [
    ("Model Catalogue", "Inspect candidates, verification status, and pricing"),
    ("Budget & Usage", "Ledger transactions, remaining balance, and local ceilings"),
    ("Dataset Validation", "[NOT IMPLEMENTED - Step 2] Evaluation case schemas & leakage checks"),
    ("Evaluation Runs", "[NOT IMPLEMENTED - Step 4] Preflight planning & dry-run evaluation"),
    ("Traces & Observability", "[NOT IMPLEMENTED - Step 6] Local span traces & latency metrics"),
    ("Project Health", "Real diagnostics: Python env, CLI entry point, DB, and KB integrity"),
    ("Exit", "Return to terminal shell"),
]


def render_welcome_screen(console: Console, selected_index: int = 0) -> None:
    """Renders the dark welcome screen inspired by Cline / OpenCode TUIs."""
    console.clear()

    # Emblem and Header
    console.print(Align.center(BANNER_EMBLEM))
    console.print(Align.center(f"{APP_TITLE}  {APP_VERSION}"))
    console.print(Align.center(APP_SUBTITLE))
    console.print(Align.center("[dim]Local-First · Zero Untracked Spend · Immutability Verified[/dim]\n"))

    # Status summary strip
    status_strip = (
        "[dim]Budget Policy:[/dim] [cyan]$10.00 Total[/cyan] / [cyan]$2.00 Pilot[/cyan]  "
        "[dim]│[/dim]  [dim]Inference:[/dim] [bold green]DRY-RUN DEFAULT[/bold green]  "
        "[dim]│[/dim]  [dim]Storage:[/dim] [dim]SQLite WAL[/dim]"
    )
    console.print(Align.center(status_strip))
    console.print()

    # Interactive Navigation Menu Box
    menu_lines = []
    menu_lines.append("[bold bright_white]How would you like to get started?[/bold bright_white]\n")

    for idx, (title, desc) in enumerate(MENU_ITEMS):
        is_selected = (idx == selected_index)
        num = idx + 1
        if is_selected:
            if "NOT IMPLEMENTED" in desc:
                menu_lines.append(f"[bold bright_cyan]› {num}. {title}[/bold bright_cyan] [dim yellow]— {desc}[/dim yellow]")
            else:
                menu_lines.append(f"[bold bright_cyan]› {num}. {title}[/bold bright_cyan] [bright_white]— {desc}[/bright_white]")
        else:
            if "NOT IMPLEMENTED" in desc:
                menu_lines.append(f"  [dim white]{num}. {title}[/dim white] [dim]— {desc}[/dim]")
            else:
                menu_lines.append(f"  [white]{num}. {title}[/white] [dim]— {desc}[/dim]")

    menu_lines.append("\n[dim]Use arrow keys ↑/↓ or 1-7, Enter to select, 'p' for Command Prompt, 'q' to quit[/dim]")

    panel = Panel(
        "\n".join(menu_lines),
        border_style="bright_black",
        expand=False,
        padding=(1, 2),
    )
    console.print(Align.center(panel))


def render_model_list_screen(console: Console, candidates: List[ModelCandidate], selected_index: int = 0) -> None:
    """Renders the interactive candidate models menu."""
    console.clear()

    header_text = (
        f"{APP_TITLE}  {APP_VERSION}\n"
        "[dim]Candidate Model Portfolio · Preliminary Local Catalog[/dim]"
    )
    console.print(Align.center(header_text))
    console.print(Align.center(
        "[yellow]NOTE: Model IDs and prices are preliminary and marked UNVERIFIED_CATALOG_ENTRY.\n"
        "Validation against OpenRouter live API catalog will occur in Step 2.[/yellow]\n"
    ))

    lines = []
    lines.append("[bold bright_white]Choose a candidate model to inspect:[/bold bright_white]\n")

    for idx, m in enumerate(candidates):
        is_selected = (idx == selected_index)
        price_str = f"${m.input_price_per_1m_usd:.2f} / ${m.output_price_per_1m_usd:.2f} per 1M tokens"

        if is_selected:
            lines.append(f"[bold bright_cyan]› {m.display_name}[/bold bright_cyan] [dim]({m.deployment_mode})[/dim]")
            lines.append(f"  [dim]ID:[/dim] [cyan]{m.model_id}[/cyan]")
            lines.append(f"  [dim]Status:[/dim] [yellow]{m.verification_status}[/yellow]  [dim]│ Est. Price:[/dim] [bright_white]{price_str}[/bright_white]")
            lines.append(f"  [dim italic]{m.experimental_purpose}[/dim italic]\n")
        else:
            lines.append(f"  [bold white]{m.display_name}[/bold white] [dim]({m.deployment_mode})[/dim]")
            lines.append(f"  [dim]ID: {m.model_id} │ Status: {m.verification_status} │ Est. Price: {price_str}[/dim]\n")

    lines.append("[dim]Arrows ↑/↓ to navigate · Enter to inspect details · Esc or 'b' to return to menu[/dim]")

    panel = Panel(
        "\n".join(lines),
        border_style="bright_black",
        expand=False,
        padding=(1, 2),
    )
    console.print(Align.center(panel))


def render_model_detail_screen(console: Console, candidate: ModelCandidate) -> None:
    """Renders a detailed inspection card for a single model candidate."""
    console.clear()

    header_text = (
        f"{APP_TITLE}  {APP_VERSION}\n"
        f"[bold cyan]Model Inspector: {candidate.model_id}[/bold cyan]\n"
    )
    console.print(Align.center(header_text))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="dim", justify="right")
    table.add_column("Value", style="bold white")

    table.add_row("Display Name:", candidate.display_name)
    table.add_row("Model Identifier:", f"[cyan]{candidate.model_id}[/cyan]")
    table.add_row("Provider:", candidate.provider)
    table.add_row("Deployment Mode:", f"[blue]{candidate.deployment_mode}[/blue]")
    table.add_row("Context Window:", f"{candidate.context_length:,} tokens" if candidate.context_length else "N/A")
    table.add_row("Est. Input Price / 1M:", f"${candidate.input_price_per_1m_usd:.4f} USD")
    table.add_row("Est. Output Price / 1M:", f"${candidate.output_price_per_1m_usd:.4f} USD")
    table.add_row("Verification Status:", f"[bold yellow]{candidate.verification_status}[/bold yellow]")
    table.add_row("Price Verification:", "[bold yellow]UNVERIFIED_ESTIMATE[/bold yellow]")
    table.add_row("License:", candidate.license)
    table.add_row("Experimental Purpose:", f"[italic]{candidate.experimental_purpose}[/italic]")

    guardrail_notice = (
        "\n[bold yellow]⚠ INFERENCE LOCKED / EVALUATION SAFEGUARD[/bold yellow]\n"
        "[dim]Selecting a model does NOT initiate API calls or consume API credits.\n"
        "Model ID must be validated against the live provider API in Step 2 before evaluation runs.[/dim]"
    )

    detail_panel = Panel(
        table,
        title=f"[bold bright_white]{candidate.display_name}[/bold bright_white]",
        border_style="bright_black",
        expand=False,
        padding=(1, 2),
    )
    console.print(Align.center(detail_panel))
    console.print(Align.center(Panel(guardrail_notice, border_style="yellow", expand=False)))
    console.print(Align.center("\n[dim]Press Enter, Esc, or 'b' to return to candidate catalogue[/dim]"))


def render_budget_screen(console: Console, status: BudgetStatus) -> None:
    """Renders the persistent local budget ledger status and policy limits."""
    console.clear()

    header_text = (
        f"{APP_TITLE}  {APP_VERSION}\n"
        "[bold cyan]Local Budget Ledger & Spending Governance[/bold cyan]\n"
    )
    console.print(Align.center(header_text))

    disclaimer = (
        "[bold yellow]LOCAL POLICY NOTICE:[/bold yellow]\n"
        "Ceilings below are local experiment tracking policies enforced by this workbench.\n"
        "They are NOT provider-enforced account limits."
    )
    console.print(Align.center(Panel(disclaimer, border_style="yellow", expand=False)))
    console.print()

    # Spend Overview Table
    table = Table(title="Spend & Policy Overview", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Budget Metric", style="dim", no_wrap=True)
    table.add_column("Configured Ceiling", justify="right")
    table.add_column("Recorded Spend", justify="right")
    table.add_column("Remaining Balance", justify="right", style="bold")
    table.add_column("Status", justify="center")

    pilot_status = "[bold red]EXHAUSTED[/bold red]" if status.is_pilot_exhausted else "[bold green]ACTIVE[/bold green]"
    total_status = "[bold red]EXHAUSTED[/bold red]" if status.is_total_exhausted else "[bold green]ACTIVE[/bold green]"

    table.add_row(
        "Initial Pilot Budget",
        f"${status.pilot_ceiling_usd:.4f}",
        f"${status.pilot_spend_usd:.4f}",
        f"${status.remaining_pilot_usd:.4f}",
        pilot_status,
    )
    table.add_row(
        "Total Experiment Ceiling",
        f"${status.total_ceiling_usd:.4f}",
        f"${status.cumulative_spend_usd:.4f}",
        f"${status.remaining_total_usd:.4f}",
        total_status,
    )
    console.print(Align.center(table))
    console.print()

    # Governance Details Table
    rule_table = Table(title="Per-Execution Safety Policies", show_header=True, header_style="bold blue", expand=False)
    rule_table.add_column("Safety Policy", style="cyan")
    rule_table.add_column("Value / Limit", justify="right")
    rule_table.add_column("Enforcement Mechanism", style="italic")

    rule_table.add_row("Provider-Reported Spend", "$0.0000 USD", "No provider synchronization; local tracking only")
    rule_table.add_row("Live Inference Enabled", "[red]NO[/red]", "Dry-Run mode by default; requires explicit approval")
    rule_table.add_row("Max Single Run Spend", f"${status.per_run_ceiling_usd:.4f} USD", "Preflight check rejects run if exceeded")
    rule_table.add_row("Max Single Case Spend", f"${status.per_case_ceiling_usd:.4f} USD", "Per-case circuit breaker halts run")
    console.print(Align.center(rule_table))
    console.print()

    # Ledger Transactions
    if status.recent_transactions:
        tx_table = Table(title="Recent Ledger Transactions", show_header=True, header_style="bold green", expand=False)
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
        console.print(Align.center(tx_table))
    else:
        console.print(Align.center("[dim]Spend ledger is currently at baseline ($0.0000 spent). No API charges incurred.[/dim]"))

    console.print(Align.center("\n[dim]Press Enter, Esc, or 'b' to return to menu[/dim]"))


def render_health_screen(console: Console, health_items: List[HealthCheckItem]) -> None:
    """Renders live system health and diagnostic checks."""
    console.clear()

    header_text = (
        f"{APP_TITLE}  {APP_VERSION}\n"
        "[bold cyan]Project Health & Diagnostics[/bold cyan]\n"
    )
    console.print(Align.center(header_text))

    table = Table(title="Live System Checks", show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Component", style="bold white")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="cyan")
    table.add_column("Notes", style="dim italic")

    for item in health_items:
        if item.status == "PASS":
            status_badge = "[bold green]PASS[/bold green]"
        elif item.status == "WARN":
            status_badge = "[bold yellow]WARN[/bold yellow]"
        else:
            status_badge = "[bold red]FAIL[/bold red]"

        table.add_row(item.component, status_badge, item.details, item.note)

    console.print(Align.center(table))
    console.print(Align.center("\n[dim]Press Enter, Esc, or 'b' to return to menu[/dim]"))


def render_unimplemented_screen(console: Console, feature_name: str, step_info: str) -> None:
    """Renders a clean placeholder for roadmap phases not yet implemented."""
    console.clear()

    header_text = (
        f"{APP_TITLE}  {APP_VERSION}\n"
        f"[bold yellow]{feature_name}[/bold yellow]\n"
    )
    console.print(Align.center(header_text))

    content = (
        f"[bold white]This component has not been implemented yet.[/bold white]\n\n"
        f"Scheduled Milestone: [cyan]{step_info}[/cyan]\n\n"
        "[dim]Safety Invariant: The workbench enforces strict step-by-step progress.\n"
        "No fake evaluation records or unverified network connections are made.[/dim]"
    )

    panel = Panel(
        content,
        title="[bold yellow]Roadmap Stage[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 3),
    )
    console.print(Align.center(panel))
    console.print(Align.center("\n[dim]Press Enter, Esc, or 'b' to return to menu[/dim]"))
