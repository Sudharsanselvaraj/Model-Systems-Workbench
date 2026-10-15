"""Persistent, panel-based interactive terminal user interface for MSU LLM Workbench."""

import sys
import time
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
from msu_llm_lab.observability.monitor import LiveMonitor
from msu_llm_lab.tui.live_views import (
    FORMAT_CHOICES, ExplorerState, LiveContext, VIEWS, ViewModel, build_execution_header, render_view, view_explorer,
)
from msu_llm_lab.tui.theme import BG_DARK
from msu_llm_lab.tui.screens import (
    NAV_ITEMS,
    NAV_KEYS,
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
        # One read-only subscription to the request-event log (polled by the UI loop; no threads, nothing to leak).
        self.monitor = LiveMonitor()
        self.live_interval_s: float = 2.0
        self.last_live_poll: float = 0.0
        self.live_version: int = 0
        self.live_ctx: Optional[LiveContext] = None
        self._view_cache: Dict[str, Any] = {}
        from msu_llm_lab.core.config import settings as _settings
        self.ex = ExplorerState(dest=_settings.reports_dir / "exports")
        self._trace_cache: Optional[tuple] = None
        self._redraw_hook = None

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
        self.refresh_live()

    # ------------------------------------------------------------------ live telemetry
    @property
    def section(self) -> str:
        return NAV_KEYS[self.nav_index] if 0 <= self.nav_index < len(NAV_KEYS) else "overview"

    def goto(self, key: str) -> None:
        """Activate a section by key and focus it. Entering a view never starts a poller or sends anything."""
        idx = NAV_KEYS.index(key)
        self.nav_index = idx
        self.sidebar_cursor = idx
        self.content_cursor = 0
        self.focused_pane = "MAIN"
        self._view_cache.clear()

    def refresh_live(self) -> int:
        """Poll the event log once (read-only), recompute the aggregate, refresh ledger and system samples. Returns new event count."""
        from msu_llm_lab.models.registry import roster_status
        new = self.monitor.poll()
        agg = self.monitor.aggregate()
        self.inf_telemetry.update_from(agg)
        self.sys_telemetry.sample()
        try:
            self.budget_status = self.budget_service.get_status()
        except Exception:
            pass
        review_ids: List[str] = []
        review_count: Optional[int] = None
        if self.section == "lead":
            review_count, review_ids = self.monitor.lead_review()
        self.live_ctx = LiveContext(
            now=time.time(), width=self.console.size[0], budget=self.budget_status, agg=agg, analysis=self.monitor.analysis,
            dataset=self.monitor.dataset, roster=roster_status(), events_malformed=self.monitor.tail.malformed,
            review_ids=review_ids, review_count=review_count,
        )
        self.live_version += 1
        self._view_cache.clear()
        self.last_live_poll = time.monotonic()
        return new

    def tick(self) -> None:
        """Called when the keyboard has been idle: refresh from the telemetry source when the interval has elapsed."""
        if time.monotonic() - self.last_live_poll >= self.live_interval_s:
            self.refresh_live()

    def _trace_set(self):
        """Selected trace rows for the explorer, cached until the data or the filters change. Read-only."""
        from msu_llm_lab.observability.export import build_trace_rows
        key = (self.live_version,) + self.ex.key()
        if self._trace_cache is None or self._trace_cache[0] != key:
            preds, events, labels, frozen = self.monitor.trace_inputs()
            self._trace_cache = (key, build_trace_rows(preds, events, labels, frozen, self.ex.selection()))
        return self._trace_cache[1]

    def _plans(self):
        from msu_llm_lab.core.config import settings
        from msu_llm_lab.observability.export import load_bench_plans
        return load_bench_plans(settings.data_dir / "benchmarks")

    def _explorer_choices(self):
        """Models, runs, error classes and data kinds present in ALL stored traces (unfiltered), cached per refresh."""
        from msu_llm_lab.observability.export import build_trace_rows
        if getattr(self, "_choices_cache", (None,))[0] != self.live_version:
            preds, events, labels, frozen = self.monitor.trace_inputs()
            rows = build_trace_rows(preds, events, labels, frozen).rows
            self._choices_cache = (self.live_version, {
                "models": sorted({r["identity"]["model_key"] for r in rows}), "runs": sorted({r["identity"]["run_id"] for r in rows}),
                "errors": sorted({r["request"]["error_category"] for r in rows if r["request"]["error_category"]}),
                "kinds": sorted({r["markers"]["data_kind"] for r in rows})})
        return self._choices_cache[1]

    def get_view(self) -> Optional[ViewModel]:
        if self.section == "explorer" and self.live_ctx is not None:
            if "explorer" not in self._view_cache:
                from msu_llm_lab.observability.export import load_bench_plans, participant_coverage, run_summaries
                ts = self._trace_set()
                plans = self._plans()
                self._view_cache["explorer"] = view_explorer(self.live_ctx, self.ex, ts, participant_coverage(self.live_ctx.roster, ts.rows),
                                                             self.monitor.analysis.labels.status, run_summaries(ts.rows, plans))
            return self._view_cache["explorer"]
        builder = VIEWS.get(self.section)
        if builder is None or self.live_ctx is None:
            return None
        if self.section not in self._view_cache:
            self.live_ctx.width = self.console.size[0]
            self._view_cache[self.section] = builder(self.live_ctx)
        return self._view_cache[self.section]

    def _row_count(self) -> int:
        sec = self.section
        if sec == "roster":
            return len(self._model_rows())
        if sec == "budget":
            return 6
        if sec == "evaluations":
            return len(self.eval_runs)
        if sec == "traces":
            return len(self.traces)
        if sec == "health":
            return len(self.health_items)
        vm = self.get_view()
        return len(vm.rows) if vm else 0

    def _cursor(self) -> int:
        return self.model_cursor if self.section == "roster" else self.content_cursor

    def _set_cursor(self, value: int) -> None:
        if self.section == "roster":
            self.model_cursor = value
        else:
            self.content_cursor = value
            if self.section == "evaluations" and self.eval_runs and 0 <= value < len(self.eval_runs):
                sel_id = self.eval_runs[value].get("run_id")
                try:
                    self.selected_run_results = self.budget_service.db.get_evaluation_results_for_run(sel_id) if sel_id else []
                except Exception:
                    pass

    def _move_main(self, delta: Optional[int] = None, to: Optional[str] = None) -> None:
        n = self._row_count()
        if to == "home":
            new = 0
        elif to == "end":
            new = n - 1
        else:
            new = self._cursor() + (delta or 0)
        self._set_cursor(max(0, min(new, max(0, n - 1))))

    def close(self) -> None:
        """Release the telemetry subscription."""
        self.monitor.close()

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
            return list(REGISTRY.values())
        return self.get_filtered_candidates()

    def _clamp_model_cursor(self) -> None:
        """Ensures model_cursor remains within bounds of current filtered list."""
        if self.show_roster and self.section == "roster":
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
        header_size = 3

        layout = Layout()
        layout.split(
            Layout(name="header", size=header_size),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=4),
        )
        layout["body"].split_row(
            Layout(name="nav", size=34 if width >= 140 else 30),
            Layout(name="right", ratio=1),
        )
        layout["body"]["right"].split(
            Layout(name="top_row", size=top_size),
            Layout(name="main", ratio=1),
            Layout(name="bottom_row", size=bottom_size),
        )
        top_row = layout["body"]["right"]["top_row"]
        top_row.split_row(
            Layout(name="sys_metrics", ratio=1),
            Layout(name="inf_status", size=30),
        )
        top_row["sys_metrics"].split(
            Layout(name="sys_res", ratio=1),
            Layout(name="req_metrics", ratio=1),
        )
        layout["body"]["right"]["bottom_row"].split_row(
            Layout(name="logs", ratio=1),
            Layout(name="budget_usage", ratio=1),
            Layout(name="datasets", ratio=1),
        )
        layout["footer"].split(
            Layout(name="command", size=3),
            Layout(name="status_line", size=1),
        )

        # 1. Execution-state header (mode, requests, spend by kind, readiness, dataset, parity)
        if self.live_ctx is not None:
            layout["header"].update(build_execution_header(self.live_ctx))
        else:
            layout["header"].update(build_top_header())

        # 2. Navigation sidebar (full height, windowed around the cursor on short terminals)
        layout["body"]["nav"].update(
            build_navigation_panel(
                self.nav_index,
                self.focused_pane == "NAV",
                self.sidebar_cursor,
                max_rows=max(6, height - header_size - 4),
            )
        )

        # 3. System resources and request metrics
        top_row["sys_metrics"]["sys_res"].update(build_system_resources_panel(self.sys_telemetry))
        top_row["sys_metrics"]["req_metrics"].update(build_request_metrics_panel(self.inf_telemetry))

        # 4. Inference status
        agg = self.live_ctx.agg if self.live_ctx else None
        if self.budget_status:
            top_row["inf_status"].update(build_inference_status_panel(self.budget_status, agg))

        # 5. Main content panel
        layout["body"]["right"]["main"].update(self._build_main_panel())

        # 6. Bottom panels
        bottom = layout["body"]["right"]["bottom_row"]
        bottom["logs"].update(build_recent_logs_panel(self.log_buffer))
        if self.budget_status:
            bottom["budget_usage"].update(build_budget_usage_bottom_panel(self.budget_status))
        bottom["datasets"].update(build_datasets_bottom_panel(labels_status=self.monitor.analysis.labels.status))

        # 7. Command bar and status line
        layout["footer"]["command"].update(
            build_command_bar(self.input_mode.name, self.text_buffer, self.cursor_pos, self.feedback_message, self.focused_pane == "COMMAND")
        )
        if self.budget_status:
            layout["footer"]["status_line"].update(build_footer_bar(self.budget_status, agg))

        return layout

    def _build_main_panel(self):
        focused = self.focused_pane == "MAIN"
        sec = self.section
        if self.input_mode == InputMode.HELP:
            return build_help_main_panel(focused)
        if sec == "settings":
            from rich.console import Group
            return Panel(Group(build_settings_main_panel(focused), build_help_main_panel(False)), title="SETTINGS & HELP", border_style="grey50")
        if sec == "roster":
            if self.show_roster:
                from msu_llm_lab.models.registry import roster_status
                return build_roster_panel(roster_status(), self.model_cursor, focused, self.console.size[0])
            return build_model_catalogue_panel(self.candidates, self.model_cursor, self.search_query, focused)
        if sec == "budget":
            if self.budget_status:
                return build_budget_main_panel(self.budget_status, focused, self.content_cursor)
            return Panel("Budget data unavailable", title="BUDGET & SPEND")
        if sec == "validation":
            return build_dataset_validation_panel(focused)
        if sec == "evaluations":
            return build_evaluation_main_panel(self.eval_runs, self.content_cursor, focused, self.selected_run_results)
        if sec == "traces":
            return build_traces_main_panel(self.traces, self.trace_report, self.content_cursor, focused)
        if sec == "health":
            return build_health_main_panel(self.health_items, focused, self.content_cursor)
        vm = self.get_view()
        if vm is None:
            return Panel("Live data unavailable", title=NAV_ITEMS[self.nav_index][1])
        return render_view(vm, self.content_cursor, focused, self.console.size[0])

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
                self._redraw_hook = lambda: live.update(self.build_layout(), refresh=True)
                while not self.should_exit:
                    live.update(self.build_layout(), refresh=True)
                    key = self.key_reader.read_key(timeout=self.live_interval_s)
                    if key == "TIMEOUT":
                        self.tick()  # idle: refresh from the recorded telemetry only
                        continue
                    self.handle_key(key)
        except KeyboardInterrupt:
            pass
        finally:
            self._redraw_hook = None
            self.close()
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
        title = NAV_ITEMS[self.nav_index][1]
        if self.section == "explorer" and self.focused_pane == "MAIN" and key in ("m", "t", "u", "s", "e", "d", "f", "c", "x"):
            self._explorer_key(key)
        elif key in (":",):
            self.input_mode = InputMode.COMMAND
            self.focused_pane = "COMMAND"
            self.text_buffer = ""
            self.cursor_pos = 0
            self.feedback_message = ""
        elif key in ("/",):
            self.goto("roster")
            self.sidebar_cursor = self.nav_index
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
            elif self.focused_pane == "COMMAND":
                self.focused_pane = "NAV"
            else:
                self.focused_pane = "COMMAND"
        elif key in ("SHIFT_TAB",):
            self.focused_pane = {"NAV": "COMMAND", "MAIN": "NAV", "COMMAND": "MAIN"}[self.focused_pane]
        elif key in ("LEFT",):
            if self.focused_pane == "COMMAND":
                self.focused_pane = "MAIN"
                self.feedback_message = f"Focused {title} — use ↑/↓ to navigate"
            elif self.focused_pane == "MAIN":
                self.focused_pane = "NAV"
                self.feedback_message = "Focused Navigation sidebar — use ↑/↓ to browse sections"
        elif key in ("RIGHT",):
            if self.focused_pane == "NAV":
                self.focused_pane = "MAIN"
                self.feedback_message = f"Focused {title} — use ↑/↓ to navigate"
            elif self.focused_pane == "MAIN":
                self.focused_pane = "COMMAND"
                self.feedback_message = "Focused Command input — press Enter or type : to execute"
        elif key in ("HOME",):
            if self.focused_pane == "MAIN":
                self._move_main(to="home")
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = 0
        elif key in ("END",):
            if self.focused_pane == "MAIN":
                self._move_main(to="end")
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = len(NAV_ITEMS) - 1
        elif key in ("PAGE_UP",):
            if self.focused_pane == "MAIN":
                self._move_main(delta=-3)
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = max(0, self.sidebar_cursor - 3)
        elif key in ("PAGE_DOWN",):
            if self.focused_pane == "MAIN":
                self._move_main(delta=3)
            elif self.focused_pane == "NAV":
                self.sidebar_cursor = min(len(NAV_ITEMS) - 1, self.sidebar_cursor + 3)
        elif key in ("UP", "k", "K"):
            if self.focused_pane == "NAV":
                self.sidebar_cursor = max(0, self.sidebar_cursor - 1)
            elif self.focused_pane == "MAIN":
                self._move_main(delta=-1)
            elif self.focused_pane == "COMMAND":
                if self.command_history and self.history_index > 0:
                    self.history_index -= 1
                    self.text_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.text_buffer)
        elif key in ("DOWN", "j", "J"):
            if self.focused_pane == "NAV":
                self.sidebar_cursor = min(len(NAV_ITEMS) - 1, self.sidebar_cursor + 1)
            elif self.focused_pane == "MAIN":
                self._move_main(delta=1)
            elif self.focused_pane == "COMMAND":
                if self.command_history and self.history_index < len(self.command_history) - 1:
                    self.history_index += 1
                    self.text_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.text_buffer)
                elif self.history_index >= len(self.command_history) - 1:
                    self.history_index = len(self.command_history)
                    self.text_buffer = ""
                    self.cursor_pos = 0
        elif key in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="):
            idx = {"0": 9, "-": 10, "=": 11}.get(key, int(key) - 1 if key.isdigit() else 0)
            self.goto(NAV_KEYS[idx])
            self.log_buffer.add("INFO", f"Navigated to {NAV_ITEMS[self.nav_index][1]}")
            self.feedback_message = f"Switched to {NAV_ITEMS[self.nav_index][1]}"
        elif key in ("[", "]"):
            idx = (self.nav_index + (1 if key == "]" else -1)) % len(NAV_KEYS)
            self.goto(NAV_KEYS[idx])
            self.feedback_message = f"Switched to {NAV_ITEMS[self.nav_index][1]}"
        elif key in ("r", "R"):
            self.refresh_data()
            self.refresh_live()
            self._clamp_model_cursor()
            self.feedback_message = "Refreshed from the telemetry log, ledger and local files (read-only; nothing was sent)"
            self.log_buffer.add("INFO", "Telemetry and SQLite ledger refreshed")
        elif key in ("ENTER",):
            if self.focused_pane == "NAV":
                self.goto(NAV_KEYS[self.sidebar_cursor])
                self.log_buffer.add("INFO", f"Opened {NAV_ITEMS[self.nav_index][1]}")
                self.feedback_message = f"Focused {NAV_ITEMS[self.nav_index][1]} — use ↑/↓ to navigate"
            elif self.focused_pane == "MAIN":
                self._inspect_selected()
            elif self.focused_pane == "COMMAND":
                self.input_mode = InputMode.COMMAND
                self.text_buffer = ""
                self.cursor_pos = 0

    @staticmethod
    def _cycle(options, current):
        opts = [None] + list(options)
        return opts[(opts.index(current) + 1) % len(opts)] if current in opts else None

    def _explorer_key(self, key: str) -> None:
        """Filter, format and export keys for the Trace Explorer. Only 'x' writes anything, and only a new file under the exports folder."""
        ex = self.ex
        choices = self._explorer_choices()
        if key == "m":
            ex.model = self._cycle(choices["models"], ex.model)
        elif key == "e":
            ex.error_class = self._cycle(choices["errors"], ex.error_class)
        elif key == "d":
            ex.kind = self._cycle(choices["kinds"], ex.kind)
        elif key == "t":
            ex.task = self._cycle(("INTENT", "LEAD"), ex.task)
        elif key == "u":
            ex.run = self._cycle(choices["runs"], ex.run)
        elif key == "s":
            ex.status = self._cycle(("OK", "INVALID_OUTPUT", "ERROR"), ex.status)
        elif key == "f":
            ex.fmt = FORMAT_CHOICES[(FORMAT_CHOICES.index(ex.fmt) + 1) % len(FORMAT_CHOICES)]
        elif key == "c":
            ex.model = ex.task = ex.run = ex.status = ex.error_class = ex.kind = None
        elif key == "x":
            self._run_export()
            return
        self.content_cursor = 0
        self._view_cache.pop("explorer", None)
        self.feedback_message = (f"Trace filters: model {ex.model or 'ALL'}, task {ex.task or 'ALL'}, run {ex.run or 'ALL'}, status {ex.status or 'ALL'}, "
                                 f"error class {ex.error_class or 'ALL'}, data {ex.kind or 'ALL'}; format {ex.fmt.upper()}")

    def _run_export(self) -> None:
        from msu_llm_lab.observability.export import ExportError, build_trace_rows, export_traces
        ex = self.ex
        ex.last_error, ex.last_result = None, None
        preds, events, labels, frozen = self.monitor.trace_inputs()
        sel = ex.selection()
        last_pct = [-1]

        def progress(done: int, total: int) -> None:
            pct = int(100 * done / total) if total else 100
            if pct // 10 != last_pct[0]:
                last_pct[0] = pct // 10
                ex.progress = f"preparing {done}/{total} records"
                if self._redraw_hook:
                    self._redraw_hook()
        try:
            ex.progress = "starting"
            ts = build_trace_rows(preds, events, labels, frozen, sel, progress=progress)
            ex.progress = f"writing {ex.fmt.upper()} to {ex.dest.resolve()}"
            if self._redraw_hook:
                self._redraw_hook()
            ex.last_result = export_traces(ts, sel, self.live_ctx.roster if self.live_ctx else [], labels, ex.fmt, ex.dest)
            ex.progress = "done"
            r = ex.last_result
            self.feedback_message = f"Exported {r.n_exported} records to {r.path}"
            self.log_buffer.add("INFO", f"Trace export written: {r.n_exported} records ({r.format})")
        except (ExportError, OSError) as e:
            ex.last_error = f"{type(e).__name__}: {e}"
            ex.progress = "failed"
            self.feedback_message = f"Export FAILED: {e}"
            self.log_buffer.add("ERROR", f"Trace export failed: {type(e).__name__}")
        self._view_cache.pop("explorer", None)

    def _inspect_selected(self) -> None:
        sec = self.section
        if sec == "roster":
            filtered = self._model_rows()
            if filtered and 0 <= self.model_cursor < len(filtered):
                cand = filtered[self.model_cursor]
                if self.show_roster:
                    d_name = getattr(cand, "display_name", str(cand))
                    req = getattr(cand, "requested_model_id", None)
                    req_id = getattr(req, "value", str(req)) if req else getattr(cand, "model_id", str(cand))
                    proto = getattr(cand, "protocol", "")
                    proto_val = getattr(proto, "value", str(proto))
                    self.feedback_message = f"Inspected: {d_name} ({req_id}) | {proto_val}"
                else:
                    m_id = getattr(cand, "model_id", str(cand))
                    prov = getattr(cand, "provider", "unknown")
                    ctx = getattr(cand, "context_length", 0)
                    in_p = getattr(cand, "input_price_per_1m_usd", 0)
                    out_p = getattr(cand, "output_price_per_1m_usd", 0)
                    self.feedback_message = (
                        f"Inspected: {m_id} ({prov}) | Context: {ctx:,} tokens | "
                        f"Price: ${in_p:.2f}/${out_p:.2f} per 1M"
                    )
                self.log_buffer.add("INFO", "Inspected a roster entry")
            else:
                self.feedback_message = "No model selected"
        elif sec == "budget":
            bs = self.budget_status
            budget_rules = [
                f"Pilot Budget: ${bs.pilot_ceiling_usd:.2f} ceiling (changed only by the typed, audited `budget raise-pilot-ceiling`).",
                f"Total Ceiling: ${bs.total_ceiling_usd:.2f} lifetime experiment ceiling. Hard stop across all experiments.",
                "Spend is shown by basis: ACTUAL (provider-reported), ESTIMATED, UNRESOLVED; legacy simulated rows never count.",
                "Live inference is not available from the dashboard. Use the CLI (smoke-test) with typed approval.",
                f"Max Single Run: ${bs.per_run_ceiling_usd:.2f}. Preflight check halts batches exceeding limit.",
                f"Max Single Case: ${bs.per_case_ceiling_usd:.2f}. Per-case circuit breaker halts run immediately.",
            ]
            idx = min(self.content_cursor, len(budget_rules) - 1)
            self.feedback_message = budget_rules[idx]
            self.log_buffer.add("INFO", f"Inspected budget rule #{idx+1}")
        elif sec == "evaluations":
            if self.eval_runs and 0 <= self.content_cursor < len(self.eval_runs):
                sel_run = self.eval_runs[self.content_cursor]
                run_id = sel_run.get("run_id")
                try:
                    self.selected_run_results = self.budget_service.db.get_evaluation_results_for_run(run_id) if run_id else []
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
        elif sec == "traces":
            if self.traces and 0 <= self.content_cursor < len(self.traces):
                trace = self.traces[self.content_cursor]
                diag = (self.trace_report.diagnoses[self.content_cursor]
                        if (self.trace_report and self.content_cursor < len(self.trace_report.diagnoses)) else None)
                cat_str = diag.category.value if diag else "UNANALYZED"
                self.feedback_message = f"Inspected Trace: {trace.id} [{cat_str}] | Latency: {trace.latency_ms:.0f}ms | Tokens: {trace.total_tokens}"
                self.log_buffer.add("INFO", f"Inspected trace {trace.id}")
            else:
                self.feedback_message = "No traces loaded"
        elif sec == "health":
            if self.health_items and 0 <= self.content_cursor < len(self.health_items):
                item = self.health_items[self.content_cursor]
                self.feedback_message = f"{item.component} [{item.status}]: {item.details} — {item.note}"
                self.log_buffer.add("INFO", f"Inspected diagnostic: {item.component}")
        else:
            vm = self.get_view()
            if vm and vm.rows and 0 <= self.content_cursor < len(vm.rows):
                self.feedback_message = vm.rows[self.content_cursor].detail[0][:160] if vm.rows[self.content_cursor].detail else ""
            else:
                self.feedback_message = f"Viewing {NAV_ITEMS[self.nav_index][1]}"

    COMMAND_SECTIONS = {
        "overview": "overview", "o": "overview", "home": "overview",
        "roster": "roster", "models": "roster", "m": "roster",
        "quality": "quality", "classification": "quality", "q1": "quality",
        "calibration": "calibration", "confidence": "calibration", "cal": "calibration",
        "agreement": "agreement", "agree": "agreement",
        "lead": "lead", "leads": "lead",
        "latency": "latency", "cost": "latency", "reliability": "latency", "live": "latency",
        "robustness": "robustness", "readiness": "robustness",
        "dataset": "dataset", "labels": "dataset", "d": "dataset",
        "budget": "budget", "spend": "budget", "b": "budget",
        "traces": "traces", "t": "traces",
        "settings": "settings", "s": "settings",
        "eval": "evaluations", "evals": "evaluations", "e": "evaluations", "rag": "evaluations",
        "health": "health", "h": "health",
        "explorer": "explorer", "export": "explorer", "trace": "explorer",
        "validation": "validation", "validate": "validation",
    }

    def execute_command(self, cmd_line: str) -> None:
        """Parses and executes a command entered via the command bar."""
        raw_cmd = cmd_line.lstrip(":").strip()
        if not raw_cmd:
            return

        parts = raw_cmd.split()
        cmd = parts[0].lower()

        if cmd == "roster":
            self.show_roster = True
            self.goto("roster")
            self.feedback_message = "Comparison roster: all participants, readiness and exact blockers"
            self.log_buffer.add("INFO", "Navigated to Comparison Roster")
        elif cmd in ("catalogue", "catalog", "legacy"):
            self.show_roster = False
            self.goto("roster")
            self.feedback_message = "Legacy planning catalogue (unverified prices); :roster shows the verified comparison roster"
        elif cmd in self.COMMAND_SECTIONS:
            if cmd in ("models", "m"):
                self.show_roster = True
            self.goto(self.COMMAND_SECTIONS[cmd])
            self.feedback_message = f"Switched to {NAV_ITEMS[self.nav_index][1]}"
            self.log_buffer.add("INFO", f"Navigated to {NAV_ITEMS[self.nav_index][1]}")
        elif cmd in ("audit",):
            self._execute_audit_check()
        elif cmd in ("refresh", "r"):
            self.refresh_data()
            self.refresh_live()
            self._clamp_model_cursor()
            self.feedback_message = "Refreshed models, telemetry, and health diagnostics (read-only)"
            self.log_buffer.add("INFO", "Telemetry and SQLite ledger refreshed")
        elif cmd in ("clear", "cls"):
            self.search_query = ""
            self._clamp_model_cursor()
            self.feedback_message = ""
        elif cmd in ("help", "?"):
            self.input_mode = InputMode.HELP
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
