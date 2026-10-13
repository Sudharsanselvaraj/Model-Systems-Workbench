"""State machine and interactive loop for the MSU LLM Workbench TUI."""

import sys
from enum import Enum, auto
from typing import Optional, List
from rich.console import Console

from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.models.catalog import get_portfolio, ModelCandidate
from msu_llm_lab.tui.health import run_system_health_checks
from msu_llm_lab.tui.input import KeyReader
from msu_llm_lab.tui.screens import (
    render_welcome_screen,
    render_model_list_screen,
    render_model_detail_screen,
    render_budget_screen,
    render_health_screen,
    render_unimplemented_screen,
    MENU_ITEMS,
)


class ScreenState(Enum):
    WELCOME = auto()
    MODELS = auto()
    MODEL_DETAIL = auto()
    BUDGET = auto()
    HEALTH = auto()
    UNIMPLEMENTED = auto()
    PROMPT = auto()
    EXIT = auto()


class WorkbenchTUI:
    """Main interactive terminal application for MSU LLM Workbench."""

    def __init__(self, console: Optional[Console] = None, key_reader: Optional[KeyReader] = None) -> None:
        self.console = console or Console()
        self.key_reader = key_reader or KeyReader()
        self.budget_service = BudgetService()
        self.state = ScreenState.WELCOME
        self.menu_cursor = 0
        self.model_cursor = 0
        self.candidates: List[ModelCandidate] = []
        self.unimplemented_feature = ("", "")

    def refresh_data(self) -> None:
        """Loads latest data from SQLite and local services."""
        self.candidates = get_portfolio()

    def run(self) -> None:
        """Runs the interactive UI event loop until EXIT is reached."""
        self.refresh_data()

        # If running in non-interactive mode (e.g. redirected or test pipe), render welcome and exit cleanly
        if not self.key_reader.is_tty:
            render_welcome_screen(self.console, self.menu_cursor)
            return

        try:
            while self.state != ScreenState.EXIT:
                if self.state == ScreenState.WELCOME:
                    self._handle_welcome()
                elif self.state == ScreenState.MODELS:
                    self._handle_models()
                elif self.state == ScreenState.MODEL_DETAIL:
                    self._handle_model_detail()
                elif self.state == ScreenState.BUDGET:
                    self._handle_budget()
                elif self.state == ScreenState.HEALTH:
                    self._handle_health()
                elif self.state == ScreenState.UNIMPLEMENTED:
                    self._handle_unimplemented()
                elif self.state == ScreenState.PROMPT:
                    self._handle_prompt()

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user. Exiting workbench.[/yellow]")
        finally:
            self.console.print("[dim]Exited MSU LLM Inference Workbench.[/dim]")

    def _handle_welcome(self) -> None:
        render_welcome_screen(self.console, self.menu_cursor)
        key = self.key_reader.read_key()

        if key == "UP":
            self.menu_cursor = (self.menu_cursor - 1) % len(MENU_ITEMS)
        elif key == "DOWN":
            self.menu_cursor = (self.menu_cursor + 1) % len(MENU_ITEMS)
        elif key in ("q", "Q", "EOF"):
            self.state = ScreenState.EXIT
        elif key in ("p", "P"):
            self.state = ScreenState.PROMPT
        elif key in ("1", "2", "3", "4", "5", "6", "7"):
            self.menu_cursor = int(key) - 1
            self._dispatch_menu_selection()
        elif key == "ENTER":
            self._dispatch_menu_selection()

    def _dispatch_menu_selection(self) -> None:
        if self.menu_cursor == 0:
            self.model_cursor = 0
            self.state = ScreenState.MODELS
        elif self.menu_cursor == 1:
            self.state = ScreenState.BUDGET
        elif self.menu_cursor == 2:
            self.unimplemented_feature = ("Dataset Validation", "Step 2: Evaluation Case Schemas & Leakage Checks")
            self.state = ScreenState.UNIMPLEMENTED
        elif self.menu_cursor == 3:
            self.unimplemented_feature = ("Evaluation Runs", "Step 4: Local Evaluation Runner & Dry-Run Scorer")
            self.state = ScreenState.UNIMPLEMENTED
        elif self.menu_cursor == 4:
            self.unimplemented_feature = ("Traces & Observability", "Step 6: Local Span Tracer & Redacted Langfuse Adapter")
            self.state = ScreenState.UNIMPLEMENTED
        elif self.menu_cursor == 5:
            self.state = ScreenState.HEALTH
        elif self.menu_cursor == 6:
            self.state = ScreenState.EXIT

    def _handle_models(self) -> None:
        if not self.candidates:
            self.refresh_data()
        render_model_list_screen(self.console, self.candidates, self.model_cursor)
        key = self.key_reader.read_key()

        if key == "UP":
            if self.candidates:
                self.model_cursor = (self.model_cursor - 1) % len(self.candidates)
        elif key == "DOWN":
            if self.candidates:
                self.model_cursor = (self.model_cursor + 1) % len(self.candidates)
        elif key in ("ESC", "b", "B", "q", "Q"):
            self.state = ScreenState.WELCOME
        elif key == "ENTER":
            if self.candidates:
                self.state = ScreenState.MODEL_DETAIL

    def _handle_model_detail(self) -> None:
        if 0 <= self.model_cursor < len(self.candidates):
            candidate = self.candidates[self.model_cursor]
            render_model_detail_screen(self.console, candidate)
        self.key_reader.read_key()
        self.state = ScreenState.MODELS

    def _handle_budget(self) -> None:
        status = self.budget_service.get_status()
        render_budget_screen(self.console, status)
        self.key_reader.read_key()
        self.state = ScreenState.WELCOME

    def _handle_health(self) -> None:
        items = run_system_health_checks()
        render_health_screen(self.console, items)
        self.key_reader.read_key()
        self.state = ScreenState.WELCOME

    def _handle_unimplemented(self) -> None:
        name, step = self.unimplemented_feature
        render_unimplemented_screen(self.console, name, step)
        self.key_reader.read_key()
        self.state = ScreenState.WELCOME

    def _handle_prompt(self) -> None:
        """Screen E: Interactive command prompt dispatcher."""
        self.console.clear()
        self.console.print("[bold bright_white]MSU LLM Workbench — Command Centre[/bold bright_white]")
        self.console.print("[dim]Type 'help' for supported commands, 'menu' for visual UI, or 'exit' to quit.[/dim]\n")

        while True:
            try:
                cmd_line = self.console.input("[bold cyan]msu-workbench>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.state = ScreenState.WELCOME
                break

            if not cmd_line:
                continue

            parts = cmd_line.split()
            cmd = parts[0].lower()

            if cmd in ("exit", "quit", "q"):
                self.state = ScreenState.EXIT
                break
            elif cmd in ("menu", "ui"):
                self.state = ScreenState.WELCOME
                break
            elif cmd in ("clear", "cls"):
                self.console.clear()
            elif cmd == "help":
                self._print_prompt_help()
            elif cmd == "models":
                self._run_prompt_models()
            elif cmd == "budget":
                self._run_prompt_budget()
            elif cmd == "health":
                self._run_prompt_health()
            elif cmd == "audit":
                self._run_prompt_audit()
            else:
                self.console.print(f"[red]Unknown command:[/red] '{cmd}'. Type '[bold]help[/bold]' for available commands.")

    def _print_prompt_help(self) -> None:
        from rich.table import Table
        t = Table(title="Available Workbench Commands", show_header=True, header_style="bold blue")
        t.add_column("Command", style="cyan")
        t.add_column("Description")
        t.add_row("models", "List candidate models and their verification status")
        t.add_row("budget", "Display budget ledger, ceilings, and remaining allowance")
        t.add_row("health", "Run live system and knowledge base integrity checks")
        t.add_row("audit", "Run knowledge base structure audit summary")
        t.add_row("menu", "Return to full-screen visual interactive navigation")
        t.add_row("clear", "Clear terminal screen")
        t.add_row("exit / quit", "Exit the workbench and return to shell")
        self.console.print(t)

    def _run_prompt_models(self) -> None:
        candidates = get_portfolio()
        from rich.table import Table
        t = Table(title="Candidate Models Portfolio", show_header=True, header_style="bold magenta")
        t.add_column("Model ID", style="cyan")
        t.add_column("Display Name")
        t.add_column("Mode", style="blue")
        t.add_column("Price (In/Out 1M)", justify="right")
        t.add_column("Status", justify="center")
        for m in candidates:
            t.add_row(
                m.model_id,
                m.display_name,
                m.deployment_mode,
                f"${m.input_price_per_1m_usd:.2f} / ${m.output_price_per_1m_usd:.2f}",
                f"[yellow]{m.verification_status}[/yellow]",
            )
        self.console.print(t)

    def _run_prompt_budget(self) -> None:
        status = self.budget_service.get_status()
        from rich.table import Table
        t = Table(title="Local Budget Status", show_header=True, header_style="bold magenta")
        t.add_column("Metric", style="dim")
        t.add_column("Ceiling", justify="right")
        t.add_column("Spend", justify="right")
        t.add_column("Remaining", justify="right", style="bold")
        t.add_row("Pilot Ceiling", f"${status.pilot_ceiling_usd:.4f}", f"${status.pilot_spend_usd:.4f}", f"${status.remaining_pilot_usd:.4f}")
        t.add_row("Total Ceiling", f"${status.total_ceiling_usd:.4f}", f"${status.cumulative_spend_usd:.4f}", f"${status.remaining_total_usd:.4f}")
        self.console.print(t)
        self.console.print("[dim]Local tracking policy only, not provider-enforced account limit.[/dim]")

    def _run_prompt_health(self) -> None:
        items = run_system_health_checks()
        from rich.table import Table
        t = Table(title="Live System Diagnostics", show_header=True, header_style="bold magenta")
        t.add_column("Component", style="bold white")
        t.add_column("Status", justify="center")
        t.add_column("Details", style="cyan")
        for item in items:
            badge = "[green]PASS[/green]" if item.status == "PASS" else ("[yellow]WARN[/yellow]" if item.status == "WARN" else "[red]FAIL[/red]")
            t.add_row(item.component, badge, item.details)
        self.console.print(t)

    def _run_prompt_audit(self) -> None:
        from pathlib import Path
        from msu_llm_lab.audit.parser import KBParser
        from msu_llm_lab.audit.validator import KBValidator
        gen_kb = Path("Copy of MSU General KB - Local - Sudarshan.md")
        if not gen_kb.exists():
            self.console.print("[red]Source General KB file not found.[/red]")
            return
        parser = KBParser(gen_kb)
        q_m, a_m, recs, heads, orphans = parser.parse()
        validator = KBValidator(str(gen_kb), recs, heads, orphans)
        issues = validator.validate_all()
        self.console.print(f"[bold green]Audit Summary for {gen_kb.name}:[/bold green]")
        self.console.print(f"  Records: {len(recs):,} | Total Issues: {len(issues)} (Deterministic and heuristic)")
        self.console.print("  [dim]Run 'msu-llm-lab audit-kb' for full report export.[/dim]")


def run_tui() -> None:
    """Convenience launcher for the terminal UI."""
    app = WorkbenchTUI()
    app.run()
