"""Persistent, panel-based interactive terminal user interface for MSU LLM Workbench."""

import sys
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from msu_llm_lab.core.budget import BudgetService, BudgetStatus
from msu_llm_lab.models.catalog import get_portfolio, ModelCandidate
from msu_llm_lab.traces.analyzer import TraceDiagnosticAnalyzer
from msu_llm_lab.traces.importer import TraceImporter
from msu_llm_lab.traces.models import TraceRecord, TraceDiagnosticReport
from msu_llm_lab.tui.health import run_system_health_checks, HealthCheckItem
from msu_llm_lab.tui.input import KeyReader
from msu_llm_lab.tui.telemetry import SystemTelemetry, InferenceTelemetry, LogBuffer
from msu_llm_lab.tui.theme import BG_DARK
from msu_llm_lab.tui.screens import (
    NAV_ITEMS,
    build_top_header,
    build_navigation_panel,
    build_system_resources_panel,
    build_request_metrics_panel,
    build_system_and_metrics_panel,
    build_inference_status_panel,
    build_model_catalogue_panel,
    build_roster_panel,
    build_budget_main_panel,
    build_dataset_validation_panel,
    build_evaluation_main_panel,
    build_traces_main_panel,
    build_roadmap_panel,
    build_health_main_panel,
    build_settings_main_panel,
    build_help_main_panel,
    build_recent_logs_panel,
    build_budget_usage_bottom_panel,
    build_datasets_bottom_panel,
    build_command_bar,
    build_footer_bar,
)


class InputMode(Enum):
    NORMAL = auto()
    COMMAND = auto()
    SEARCH = auto()
    HELP = auto()


class WorkbenchTUI:
    """Single-instance, persistent panel-based TUI for msu-llm-lab inspired by btop."""

    def __init__(
        self,
        console: Optional[Console] = None,
        key_reader: Optional[KeyReader] = None,
    ) -> None:
        self.console = console or Console()
        self.key_reader = key_reader or KeyReader()
        self.budget_service = BudgetService()

        # Telemetry & Logging
        self.sys_telemetry = SystemTelemetry()
        self.inf_telemetry = InferenceTelemetry()
        self.log_buffer = LogBuffer()

        # State
        self.nav_index: int = 0          # Active loaded section (0-7)
        self.sidebar_cursor: int = 0     # Cursor position in sidebar list (0-7)
        self.focused_pane: str = "NAV"   # "NAV", "MAIN", or "COMMAND"
        self.model_cursor: int = 0       # Cursor for Model Catalogue table
        self.show_roster: bool = True    # the verified comparison roster is the default home; :catalogue shows the legacy planning table
        self.content_cursor: int = 0     # Cursor for current active main panel table rows
        self.input_mode: InputMode = InputMode.NORMAL

        # Editable Command / Search Bar State
        self.text_buffer: str = ""
        self.cursor_pos: int = 0
        self.command_history: List[str] = []
        self.history_index: int = -1
        self.search_query: str = ""
        self.feedback_message: str = ""

        # Data cache
        self.candidates: List[ModelCandidate] = []
        self.budget_status: Optional[BudgetStatus] = None
        self.health_items: List[HealthCheckItem] = []
        self.eval_runs: List[Dict[str, Any]] = []
        self.selected_run_results: List[Dict[str, Any]] = []
        self.traces: List[TraceRecord] = []
        self.trace_report: Optional[TraceDiagnosticReport] = None
        self.should_exit: bool = False

        # Initial data load
        self.refresh_data()

    def get_filtered_candidates(self) -> List[ModelCandidate]:
        """Returns candidate models filtered by search_query."""
        if not self.search_query:
            return self.candidates
        q = self.search_query.lower()
        return [
            m
            for m in self.candidates
            if q in m.model_id.lower()
            or q in m.display_name.lower()
            or q in m.provider.lower()
        ]

    def _model_rows(self) -> list:
        """Rows the Model pane is currently showing: the ten roster participants, or the legacy filtered catalogue."""
        if self.show_roster:
            from msu_llm_lab.models.registry import REGISTRY
            return list(REGISTRY)
        return self.get_filtered_candidates()

    def _clamp_model_cursor(self) -> None:
        """Ensures model_cursor remains within bounds of current filtered list."""
        if self.show_roster and self.nav_index == 0:
            from msu_llm_lab.models.registry import REGISTRY
            self.model_cursor = max(0, min(self.model_cursor, len(REGISTRY) - 1))
            return
        filtered = self.get_filtered_candidates()
        if not filtered:
            self.model_cursor = 0
        else:
            self.model_cursor = max(0, min(self.model_cursor, len(filtered) - 1))

    def refresh_data(self) -> None:
        """Loads latest data from SQLite ledger, portfolio registry, diagnostics, benchmarks, and traces."""
        self.candidates = get_portfolio()
        self.budget_status = self.budget_service.get_status()
        self.health_items = run_system_health_checks()
        self.sys_telemetry.sample()
        self._clamp_model_cursor()

        # Load benchmark evaluation runs
        try:
            self.eval_runs = self.budget_service.db.get_all_evaluation_runs(limit=50)
            if self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                sel_id = self.eval_runs[self.content_cursor].get("run_id")
                self.selected_run_results = (
                    self.budget_service.db.get_evaluation_results_for_run(sel_id)
                    if sel_id
                    else []
                )
            else:
                self.selected_run_results = []
        except Exception:
            self.eval_runs = []
            self.selected_run_results = []

        # Load production traces from local exports
        try:
            trace_candidates = [
                Path("data/traces/synthetic_langfuse_traces.json"),
                Path("data/traces/synthetic_langfuse_traces.csv"),
            ]
            loaded_traces: List[TraceRecord] = []
            for p in trace_candidates:
                if p.exists():
                    loaded_traces = TraceImporter.load_from_file(p)
                    break
            self.traces = loaded_traces
            if self.traces:
                analyzer = TraceDiagnosticAnalyzer()
                self.trace_report = analyzer.analyze_batch(self.traces)
            else:
                self.trace_report = None
        except Exception:
            self.traces = []
            self.trace_report = None

    def build_layout(self) -> Layout:
        """Constructs the single, persistent layout shell adapted to terminal size."""
        width, height = self.console.size

        # Graceful handling for very small terminal dimensions
        if width < 75 or height < 18:
            layout = Layout()
            warning = Panel(
                f"[bold yellow]Terminal size ({width}x{height}) is too compact.[/bold yellow]\n\n"
                "[dim]Please expand your terminal window to at least 80 columns x 24 rows for the full btop dashboard layout.[/dim]",
                title="[bold red]Display Notice[/bold red]",
                border_style="yellow",
            )
            layout.update(warning)
            return layout

        # Adjust component sizing dynamically based on height
        top_size = 14 if height >= 34 else 12
        bottom_size = 7 if height >= 34 else 6

        layout = Layout()
        layout.split(
            Layout(name="header", size=1),
            Layout(name="top_row", size=top_size),
            Layout(name="main", ratio=1),
            Layout(name="bottom_row", size=bottom_size),
            Layout(name="footer", size=4),
        )

        layout["top_row"].split_row(
            Layout(name="nav", size=28),
            Layout(name="sys_metrics", ratio=1),
            Layout(name="inf_status", size=26),
        )

        layout["top_row"]["sys_metrics"].split(
            Layout(name="sys_res", ratio=1),
            Layout(name="req_metrics", ratio=1),
        )

        layout["bottom_row"].split_row(
            Layout(name="logs", ratio=1),
            Layout(name="budget_usage", ratio=1),
            Layout(name="datasets", ratio=1),
        )

        layout["footer"].split(
            Layout(name="command", size=3),
            Layout(name="status_line", size=1),
        )

        # 1. Header Bar (Single top line)
        layout["header"].update(build_top_header())

        # 2. Navigation Sidebar
        layout["top_row"]["nav"].update(
            build_navigation_panel(
                self.nav_index,
                self.focused_pane == "NAV",
                self.sidebar_cursor,
            )
        )

        # 3. System Resources and Request Metrics (Separate stacked panels)
        layout["top_row"]["sys_metrics"]["sys_res"].update(
            build_system_resources_panel(self.sys_telemetry)
        )
        layout["top_row"]["sys_metrics"]["req_metrics"].update(
            build_request_metrics_panel(self.inf_telemetry)
        )

        # 4. Inference Status
        if self.budget_status:
            layout["top_row"]["inf_status"].update(
                build_inference_status_panel(self.budget_status)
            )

        # 5. Main Content Panel (Dynamic central area)
        if self.input_mode == InputMode.HELP or self.nav_index == 7:
            content_panel = build_help_main_panel(self.focused_pane == "MAIN")
        elif self.nav_index == 0 and self.show_roster:
            from msu_llm_lab.models.registry import roster_status
            content_panel = build_roster_panel(roster_status(), self.model_cursor, self.focused_pane == "MAIN")
        elif self.nav_index == 0:
            content_panel = build_model_catalogue_panel(
                self.candidates,
                self.model_cursor,
                self.search_query,
                self.focused_pane == "MAIN",
            )
        elif self.nav_index == 1:
            if self.budget_status:
                content_panel = build_budget_main_panel(
                    self.budget_status,
                    self.focused_pane == "MAIN",
                    self.content_cursor,
                )
            else:
                content_panel = Panel(
                    "Budget data unavailable",
                    title="BUDGET & USAGE",
                )
        elif self.nav_index == 2:
            content_panel = build_dataset_validation_panel(self.focused_pane == "MAIN")

        elif self.nav_index == 3:
            content_panel = build_evaluation_main_panel(
                self.eval_runs,
                self.content_cursor,
                self.focused_pane == "MAIN",
                self.selected_run_results,
            )
        elif self.nav_index == 4:
            content_panel = build_traces_main_panel(
                self.traces,
                self.trace_report,
                self.content_cursor,
                self.focused_pane == "MAIN",
            )
        elif self.nav_index == 5:
            content_panel = build_health_main_panel(
                self.health_items,
                self.focused_pane == "MAIN",
                self.content_cursor,
            )
        elif self.nav_index == 6:
            content_panel = build_settings_main_panel(self.focused_pane == "MAIN")
        else:
            content_panel = build_model_catalogue_panel(
                self.candidates,
                self.model_cursor,
                self.search_query,
                self.focused_pane == "MAIN",
            )

        layout["main"].update(content_panel)

        # 6. Bottom Panels (Logs, Budget Progress, Datasets)
        layout["bottom_row"]["logs"].update(build_recent_logs_panel(self.log_buffer))
        if self.budget_status:
            layout["bottom_row"]["budget_usage"].update(
                build_budget_usage_bottom_panel(self.budget_status)
            )
        layout["bottom_row"]["datasets"].update(build_datasets_bottom_panel())

        # 7. Command Input Bar & Status Line
        layout["footer"]["command"].update(
            build_command_bar(
                self.input_mode.name,
                self.text_buffer,
                self.cursor_pos,
                self.feedback_message,
                self.focused_pane == "COMMAND",
            )
        )
        if self.budget_status:
            layout["footer"]["status_line"].update(build_footer_bar(self.budget_status))

        return layout

    def run(self) -> None:
        """Main event loop using Live alternate screen buffer to prevent duplicate rendering."""
        self.refresh_data()

        is_interactive = self.key_reader.is_tty and self.console.is_terminal

        # Non-interactive / piped / automated test execution
        if not is_interactive:
            self.console.print(self.build_layout())
            return

        from rich.live import Live

        # Apply slight grey slate background matching OpenSRE (#212633)
        try:
            sys.stdout.write(f"\033]11;{BG_DARK}\007")
            sys.stdout.flush()
        except Exception:
            pass

        try:
            with Live(
                self.build_layout(),
                console=self.console,
                screen=True,
                auto_refresh=False,
            ) as live:
                while not self.should_exit:
                    live.update(self.build_layout(), refresh=True)
                    key = self.key_reader.read_key()
                    self.handle_key(key)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                # Restore original terminal window background
                sys.stdout.write("\033]111\007")
                sys.stdout.flush()
            except Exception:
                pass
            self.console.print("[dim]Exited MSU LLM Workbench.[/dim]")

    def handle_key(self, key: str) -> None:
        """Processes keystrokes according to active input mode."""
        if not key or key == "TIMEOUT":
            return

        # 1. HELP Mode Key Handling
        if self.input_mode == InputMode.HELP:
            if key in ("?", "ESC", "ENTER", "q", "Q"):
                self.input_mode = InputMode.NORMAL
            return

        # 2. COMMAND Mode (:prompt) Key Handling
        if self.input_mode == InputMode.COMMAND:
            if key == "ESC":
                self.input_mode = InputMode.NORMAL
                self.text_buffer = ""
                self.cursor_pos = 0
                self.feedback_message = "Command cancelled"
            elif key == "ENTER":
                cmd = self.text_buffer.strip()
                if cmd:
                    self.command_history.append(cmd)
                    self.history_index = len(self.command_history)
                    self.execute_command(cmd)
                self.input_mode = InputMode.NORMAL
                self.text_buffer = ""
                self.cursor_pos = 0
            elif key == "BACKSPACE":
                if self.cursor_pos > 0:
                    self.text_buffer = (
                        self.text_buffer[: self.cursor_pos - 1]
                        + self.text_buffer[self.cursor_pos :]
                    )
                    self.cursor_pos -= 1
                elif not self.text_buffer:
                    self.input_mode = InputMode.NORMAL
            elif key == "LEFT":
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif key == "RIGHT":
                self.cursor_pos = min(len(self.text_buffer), self.cursor_pos + 1)
            elif key == "HOME":
                self.cursor_pos = 0
            elif key == "END":
                self.cursor_pos = len(self.text_buffer)
            elif key == "UP":
                if self.command_history and self.history_index > 0:
                    self.history_index -= 1
                    self.text_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.text_buffer)
            elif key == "DOWN":
                if self.command_history and self.history_index < len(self.command_history) - 1:
                    self.history_index += 1
                    self.text_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.text_buffer)
                elif self.history_index >= len(self.command_history) - 1:
                    self.history_index = len(self.command_history)
                    self.text_buffer = ""
                    self.cursor_pos = 0
            elif len(key) == 1 and key.isprintable():
                self.text_buffer = (
                    self.text_buffer[: self.cursor_pos]
                    + key
                    + self.text_buffer[self.cursor_pos :]
                )
                self.cursor_pos += 1
            return

        # 3. SEARCH / FILTER Mode (/prompt) Key Handling
        if self.input_mode == InputMode.SEARCH:
            if key == "ESC":
                self.input_mode = InputMode.NORMAL
                self.text_buffer = ""
                self.search_query = ""
                self.cursor_pos = 0
                self._clamp_model_cursor()
                self.feedback_message = "Filter cleared"
            elif key == "ENTER":
                self.search_query = self.text_buffer.strip()
                self.input_mode = InputMode.NORMAL
                self._clamp_model_cursor()
                self.feedback_message = (
                    f"Filter applied: '{self.search_query}'"
                    if self.search_query
                    else "Filter cleared"
                )
            elif key == "BACKSPACE":
                if self.cursor_pos > 0:
                    self.text_buffer = (
                        self.text_buffer[: self.cursor_pos - 1]
                        + self.text_buffer[self.cursor_pos :]
                    )
                    self.cursor_pos -= 1
                    self.search_query = self.text_buffer
                    self._clamp_model_cursor()
                elif not self.text_buffer:
                    self.search_query = ""
                    self._clamp_model_cursor()
                    self.input_mode = InputMode.NORMAL
            elif key == "LEFT":
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif key == "RIGHT":
                self.cursor_pos = min(len(self.text_buffer), self.cursor_pos + 1)
            elif len(key) == 1 and key.isprintable():
                self.text_buffer = (
                    self.text_buffer[: self.cursor_pos]
                    + key
                    + self.text_buffer[self.cursor_pos :]
                )
                self.cursor_pos += 1
                self.search_query = self.text_buffer
                self._clamp_model_cursor()
            return

        # 4. NORMAL Mode Navigation
        if key in (":",):
            self.input_mode = InputMode.COMMAND
            self.focused_pane = "COMMAND"
            self.text_buffer = ""
            self.cursor_pos = 0
            self.feedback_message = ""
        elif key in ("/",):
            self.nav_index = 0
            self.sidebar_cursor = 0
            self.focused_pane = "MAIN"
            self.input_mode = InputMode.SEARCH
            self.text_buffer = ""
            self.cursor_pos = 0
            self.feedback_message = "Search mode: type query, Enter to apply, Esc to cancel"
        elif key in ("?",):
            self.input_mode = InputMode.HELP
        elif key in ("q", "Q", "EOF"):
            self.should_exit = True
        elif key in ("ESC",):
            if self.search_query:
                self.search_query = ""
                self._clamp_model_cursor()
                self.feedback_message = "Filter cleared"
            elif self.focused_pane in ("MAIN", "COMMAND"):
                self.focused_pane = "NAV"
        elif key in ("TAB",):
            # Predictable focus cycle: NAV -> MAIN -> COMMAND -> NAV
            if self.focused_pane == "NAV":
                self.focused_pane = "MAIN"
            elif self.focused_pane == "MAIN":
                self.focused_pane = "COMMAND"
            else:
                self.focused_pane = "NAV"
        elif key in ("SHIFT_TAB",):
            # Reverse focus cycle: COMMAND -> MAIN -> NAV -> COMMAND
            if self.focused_pane == "NAV":
                self.focused_pane = "COMMAND"
            elif self.focused_pane == "COMMAND":
                self.focused_pane = "MAIN"
            else:
                self.focused_pane = "NAV"
        elif key in ("LEFT",):
            # Move between panes towards sidebar
            if self.focused_pane == "COMMAND":
                self.focused_pane = "MAIN"
                self.feedback_message = f"Focused {NAV_ITEMS[self.nav_index][1]} — use ↑/↓ to navigate"
            elif self.focused_pane == "MAIN":
                self.focused_pane = "NAV"
                self.feedback_message = "Focused Navigation sidebar — use ↑/↓ to browse sections"
        elif key in ("RIGHT",):
            # Move between panes towards command input
            if self.focused_pane == "NAV":
                self.focused_pane = "MAIN"
                self.feedback_message = f"Focused {NAV_ITEMS[self.nav_index][1]} — use ↑/↓ to navigate"
            elif self.focused_pane == "MAIN":
                self.focused_pane = "COMMAND"
                self.feedback_message = "Focused Command input — press Enter or type : to execute"
        elif key in ("HOME",):
            if self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    self.model_cursor = 0
                else:
                    self.content_cursor = 0
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = 0
        elif key in ("END",):
            if self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    filtered = self._model_rows()
                    self.model_cursor = max(0, len(filtered) - 1)
                elif self.nav_index == 1:
                    self.content_cursor = 5
                elif self.nav_index == 3:
                    self.content_cursor = max(0, len(self.eval_runs) - 1)
                elif self.nav_index == 4:
                    self.content_cursor = max(0, len(self.traces) - 1)
                elif self.nav_index == 5:
                    self.content_cursor = max(0, len(self.health_items) - 1)
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = len(NAV_ITEMS) - 1
        elif key in ("PAGE_UP",):
            if self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    self.model_cursor = max(0, self.model_cursor - 3)
                else:
                    self.content_cursor = max(0, self.content_cursor - 3)
                    if self.nav_index == 3 and self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                        sel_id = self.eval_runs[self.content_cursor].get("run_id")
                        try:
                            self.selected_run_results = self.budget_service.db.get_evaluation_results_for_run(sel_id) if sel_id else []
                        except Exception:
                            pass
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = max(0, self.sidebar_cursor - 3)
        elif key in ("PAGE_DOWN",):
            if self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    filtered = self._model_rows()
                    self.model_cursor = min(max(0, len(filtered) - 1), self.model_cursor + 3)
                elif self.nav_index == 1:
                    self.content_cursor = min(5, self.content_cursor + 3)
                elif self.nav_index == 3:
                    self.content_cursor = min(max(0, len(self.eval_runs) - 1), self.content_cursor + 3)
                    if self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                        sel_id = self.eval_runs[self.content_cursor].get("run_id")
                        try:
                            self.selected_run_results = self.budget_service.db.get_evaluation_results_for_run(sel_id) if sel_id else []
                        except Exception:
                            pass
                elif self.nav_index == 4:
                    self.content_cursor = min(max(0, len(self.traces) - 1), self.content_cursor + 3)
                elif self.nav_index == 5:
                    self.content_cursor = min(max(0, len(self.health_items) - 1), self.content_cursor + 3)
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = min(len(NAV_ITEMS) - 1, self.sidebar_cursor + 3)
        elif key in ("UP", "k", "K"):
            if self.focused_pane == "NAV":
                self.sidebar_cursor = max(0, self.sidebar_cursor - 1)
            elif self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    self.model_cursor = max(0, self.model_cursor - 1)
                else:
                    self.content_cursor = max(0, self.content_cursor - 1)
                    if self.nav_index == 3 and self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                        sel_id = self.eval_runs[self.content_cursor].get("run_id")
                        try:
                            self.selected_run_results = self.budget_service.db.get_evaluation_results_for_run(sel_id) if sel_id else []
                        except Exception:
                            pass
            elif self.focused_pane == "COMMAND":
                if self.command_history and self.history_index > 0:
                    self.history_index -= 1
                    self.text_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.text_buffer)
        elif key in ("DOWN", "j", "J"):
            if self.focused_pane == "NAV":
                self.sidebar_cursor = min(len(NAV_ITEMS) - 1, self.sidebar_cursor + 1)
            elif self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    filtered = self._model_rows()
                    max_idx = max(0, len(filtered) - 1)
                    self.model_cursor = min(max_idx, self.model_cursor + 1)
                elif self.nav_index == 1:
                    self.content_cursor = min(5, self.content_cursor + 1)
                elif self.nav_index == 3:
                    max_idx = max(0, len(self.eval_runs) - 1)
                    self.content_cursor = min(max_idx, self.content_cursor + 1)
                    if self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                        sel_id = self.eval_runs[self.content_cursor].get("run_id")
                        try:
                            self.selected_run_results = self.budget_service.db.get_evaluation_results_for_run(sel_id) if sel_id else []
                        except Exception:
                            pass
                elif self.nav_index == 4:
                    max_idx = max(0, len(self.traces) - 1)
                    self.content_cursor = min(max_idx, self.content_cursor + 1)
                elif self.nav_index == 5:
                    self.content_cursor = min(max(0, len(self.health_items) - 1), self.content_cursor + 1)
                else:
                    self.content_cursor = min(10, self.content_cursor + 1)
            elif self.focused_pane == "COMMAND":
                if self.command_history and self.history_index < len(self.command_history) - 1:
                    self.history_index += 1
                    self.text_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.text_buffer)
                elif self.history_index >= len(self.command_history) - 1:
                    self.history_index = len(self.command_history)
                    self.text_buffer = ""
                    self.cursor_pos = 0
        elif key in ("1", "2", "3", "4", "5", "6", "7", "8"):
            idx = int(key) - 1
            self.nav_index = idx
            self.sidebar_cursor = idx
            self.content_cursor = 0
            self.focused_pane = "MAIN"
            self.log_buffer.add("INFO", f"Navigated to {NAV_ITEMS[self.nav_index][1]}")
            self.feedback_message = f"Switched to {NAV_ITEMS[self.nav_index][1]}"
        elif key in ("r", "R"):
            self.refresh_data()
            self._clamp_model_cursor()
            self.feedback_message = "Refreshed models, telemetry, benchmarks, and traces"
            self.log_buffer.add("INFO", "Telemetry and SQLite ledger refreshed")
        elif key in ("ENTER",):
            if self.focused_pane == "NAV":
                self.nav_index = self.sidebar_cursor
                self.content_cursor = 0
                self.focused_pane = "MAIN"
                self.log_buffer.add("INFO", f"Opened {NAV_ITEMS[self.nav_index][1]}")
                self.feedback_message = f"Focused {NAV_ITEMS[self.nav_index][1]} — use ↑/↓ to navigate"
            elif self.focused_pane == "MAIN":
                if self.nav_index == 0:
                    filtered = self._model_rows()
                    if filtered and 0 <= self.model_cursor < len(filtered):
                        cand = filtered[self.model_cursor]
                        self.feedback_message = (
                            f"Inspected: {cand.model_id} ({cand.provider}) | "
                            f"Context: {cand.context_length:,} tokens | "
                            f"Price: ${cand.input_price_per_1m_usd:.2f}/${cand.output_price_per_1m_usd:.2f} per 1M"
                        )
                        self.log_buffer.add("INFO", f"Inspected candidate {cand.model_id}")
                    else:
                        self.feedback_message = "No model selected"
                elif self.nav_index == 1:
                    budget_rules = [
                        "Pilot Budget: $2.00 ceiling. Isolated pilot experiment spending ceiling.",
                        "Total Ceiling: $10.00 experiment ceiling. Hard stop across all experiments.",
                        "Provider Spend: $0.0000 USD. Purely local SQLite tracking without outbound sync.",
                        "Live inference is not available from the dashboard. Use the CLI (smoke-test) with typed approval.",
                        "Max Single Run: $0.50. Preflight check halts batches exceeding limit.",
                        "Max Single Case: $0.02. Per-case circuit breaker halts run immediately.",
                    ]
                    idx = min(self.content_cursor, len(budget_rules) - 1)
                    self.feedback_message = budget_rules[idx]
                    self.log_buffer.add("INFO", f"Inspected budget rule #{idx+1}")
                elif self.nav_index == 3:
                    if self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                        sel_run = self.eval_runs[self.content_cursor]
                        run_id = sel_run.get("run_id")
                        try:
                            self.selected_run_results = (
                                self.budget_service.db.get_evaluation_results_for_run(run_id)
                                if run_id
                                else []
                            )
                        except Exception:
                            self.selected_run_results = []
                        acc_pct = float(sel_run.get("mean_accuracy", 0)) * 100
                        grd_pct = float(sel_run.get("mean_groundedness", 0)) * 100
                        self.feedback_message = (
                            f"Inspected Run: {run_id} ({sel_run.get('model_id')}) | "
                            f"Acc: {acc_pct:.1f}% | Faithfulness: {grd_pct:.1f}% | Lat: {float(sel_run.get('mean_latency_ms', 0)):.1f}ms"
                        )
                        self.log_buffer.add("INFO", f"Inspected benchmark run {run_id}")
                    else:
                        self.feedback_message = "No evaluation runs recorded"
                elif self.nav_index == 4:
                    if self.traces and 0 <= self.content_cursor < len(self.traces):
                        trace = self.traces[self.content_cursor]
                        diag = (
                            self.trace_report.diagnoses[self.content_cursor]
                            if (self.trace_report and self.content_cursor < len(self.trace_report.diagnoses))
                            else None
                        )
                        cat_str = diag.category.value if diag else "UNANALYZED"
                        self.feedback_message = (
                            f"Inspected Trace: {trace.id} [{cat_str}] | Latency: {trace.latency_ms:.0f}ms | Tokens: {trace.total_tokens}"
                        )
                        self.log_buffer.add("INFO", f"Inspected trace {trace.id}")
                    else:
                        self.feedback_message = "No traces loaded"
                elif self.nav_index == 5:
                    if self.health_items and 0 <= self.content_cursor < len(self.health_items):
                        item = self.health_items[self.content_cursor]
                        self.feedback_message = f"{item.component} [{item.status}]: {item.details} — {item.note}"
                        self.log_buffer.add("INFO", f"Inspected diagnostic: {item.component}")
                else:
                    self.feedback_message = f"Viewing {NAV_ITEMS[self.nav_index][1]}"
            elif self.focused_pane == "COMMAND":
                self.input_mode = InputMode.COMMAND
                self.text_buffer = ""
                self.cursor_pos = 0

    def execute_command(self, cmd_line: str) -> None:
        """Parses and executes a command entered via the command bar."""
        raw_cmd = cmd_line.lstrip(":").strip()
        if not raw_cmd:
            return

        parts = raw_cmd.split()
        cmd = parts[0].lower()

        if cmd == "roster":
            self.show_roster = True
            self.nav_index = 0
            self.sidebar_cursor = 0
            self.focused_pane = "MAIN"
            self.feedback_message = "Comparison roster: all participants, readiness and exact blockers"
            self.log_buffer.add("INFO", "Navigated to Comparison Roster")
        elif cmd in ("catalogue", "catalog", "legacy"):
            self.show_roster = False
            self.nav_index = 0
            self.sidebar_cursor = 0
            self.focused_pane = "MAIN"
            self.feedback_message = "Legacy planning catalogue (unverified prices); :roster shows the verified comparison roster"
        elif cmd in ("models", "m", "1"):
            self.show_roster = True
            self.nav_index = 0
            self.sidebar_cursor = 0
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Model Catalogue"
            self.log_buffer.add("INFO", "Navigated to Model Catalogue")
        elif cmd in ("budget", "b", "2"):
            self.nav_index = 1
            self.sidebar_cursor = 1
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Budget & Usage"
            self.log_buffer.add("INFO", "Navigated to Budget & Usage")
        elif cmd in ("dataset", "d", "3"):
            self.nav_index = 2
            self.sidebar_cursor = 2
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Dataset Validation"
            self.log_buffer.add("INFO", "Navigated to Dataset Validation")
        elif cmd in ("eval", "e", "4"):
            self.nav_index = 3
            self.sidebar_cursor = 3
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Evaluation Runs"
            self.log_buffer.add("INFO", "Navigated to Evaluation Runs")
        elif cmd in ("traces", "t", "5"):
            self.nav_index = 4
            self.sidebar_cursor = 4
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Traces & Observability"
            self.log_buffer.add("INFO", "Navigated to Traces & Observability")
        elif cmd in ("health", "h", "6"):
            self.nav_index = 5
            self.sidebar_cursor = 5
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Project Health"
            self.log_buffer.add("INFO", "Navigated to Project Health")
        elif cmd in ("settings", "s", "7"):
            self.nav_index = 6
            self.sidebar_cursor = 6
            self.focused_pane = "MAIN"
            self.feedback_message = "Switched to Settings"
            self.log_buffer.add("INFO", "Navigated to Settings")
        elif cmd in ("audit",):
            self._execute_audit_check()
        elif cmd in ("refresh", "r"):
            self.refresh_data()
            self._clamp_model_cursor()
            self.feedback_message = "Refreshed models, telemetry, and health diagnostics"
            self.log_buffer.add("INFO", "Telemetry and SQLite ledger refreshed")
        elif cmd in ("clear", "cls"):
            self.search_query = ""
            self._clamp_model_cursor()
            self.feedback_message = ""
        elif cmd in ("help", "?", "8"):
            self.nav_index = 7
            self.sidebar_cursor = 7
            self.input_mode = InputMode.NORMAL
            self.feedback_message = "Displaying Help & Keyboard Shortcuts"
        elif cmd in ("q", "quit", "exit"):
            self.should_exit = True
        else:
            self.feedback_message = f"Unknown command: ':{cmd}'. Type ':help' for shortcuts."

    def _execute_audit_check(self) -> None:
        """Executes live knowledge-base audit verification and updates feedback."""
        from pathlib import Path
        from msu_llm_lab.audit.parser import KBParser
        from msu_llm_lab.audit.validator import KBValidator

        gen_kb = Path("Copy of MSU General KB - Local - Sudarshan.md")
        if not gen_kb.exists():
            self.feedback_message = "Audit error: Source General KB not found"
            self.log_buffer.add("ERROR", "Audit check failed: KB not found")
            return

        parser = KBParser(gen_kb)
        _, _, recs, heads, orphans = parser.parse()
        validator = KBValidator(str(gen_kb), recs, heads, orphans)
        issues = validator.validate_all()
        self.feedback_message = f"Audit check: {len(recs):,} parsed records, {len(issues)} issues found in {gen_kb.name}"
        self.log_buffer.add("INFO", f"KB audit completed: {len(recs):,} records, {len(issues)} issues")


def run_tui() -> None:
    """Convenience entry point to launch the persistent TUI."""
    app = WorkbenchTUI()
    app.run()
