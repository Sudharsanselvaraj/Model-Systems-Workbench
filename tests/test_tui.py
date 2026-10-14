"""Unit tests for the btop-inspired terminal user interface, telemetry, and diagnostics."""

import io
from decimal import Decimal
from rich.console import Console
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.budget import BudgetStatus
from msu_llm_lab.models.catalog import get_portfolio
from msu_llm_lab.tui.app import WorkbenchTUI, InputMode
from msu_llm_lab.tui.health import run_system_health_checks
from msu_llm_lab.tui.input import KeyReader
from msu_llm_lab.tui.telemetry import SystemTelemetry, InferenceTelemetry, LogBuffer
from msu_llm_lab.tui.theme import render_sparkline
from msu_llm_lab.tui.screens import (
    build_top_header,
    build_navigation_panel,
    build_system_and_metrics_panel,
    build_inference_status_panel,
    build_model_catalogue_panel,
    build_budget_main_panel,
    build_roadmap_panel,
    build_health_main_panel,
    build_settings_main_panel,
    build_help_main_panel,
    build_recent_logs_panel,
    build_budget_usage_bottom_panel,
    build_datasets_bottom_panel,
    build_command_bar,
    build_footer_bar,
    # Compatibility wrappers
    build_header_panel,
    build_models_panel,
    build_budget_panel,
    build_dataset_panel,
    build_eval_panel,
    build_traces_panel,
    build_health_panel,
    build_help_panel,
    build_status_bar,
)

runner = CliRunner()


def test_health_checks_live():
    """Verify live system diagnostics produce real PASS results for current environment."""
    items = run_system_health_checks()
    comp_map = {item.component: item for item in items}

    assert "Python Runtime" in comp_map
    assert comp_map["Python Runtime"].status == "PASS"

    assert "SQLite Database" in comp_map
    assert comp_map["SQLite Database"].status == "PASS"

    assert "Audit Subsystem" in comp_map
    assert comp_map["Audit Subsystem"].status == "PASS"

    assert "MSU General KB" in comp_map
    assert comp_map["MSU General KB"].status == "PASS"
    assert "MSU Programmes KB" in comp_map
    assert comp_map["MSU Programmes KB"].status == "PASS"


def test_system_telemetry_sampling():
    """Verify local hardware telemetry samples without error or network calls."""
    sys_t = SystemTelemetry(history_len=10)
    sys_t.sample()
    cpu_pct, cpu_detail, cpu_hist = sys_t.get_cpu_info()
    assert 0.0 <= cpu_pct <= 100.0
    assert "cores" in cpu_detail
    assert len(cpu_hist) >= 1

    mem_pct, mem_detail, mem_hist = sys_t.get_mem_info()
    assert 0.0 <= mem_pct <= 100.0
    assert "GB" in mem_detail
    assert len(mem_hist) >= 1

    disk_pct, disk_detail, disk_hist = sys_t.get_disk_info()
    assert 0.0 <= disk_pct <= 100.0
    assert "GB" in disk_detail
    assert len(disk_hist) >= 1


def test_render_sparkline():
    """Verify sparkline rendering generates correct Unicode block glyphs."""
    data = [0.0, 25.0, 50.0, 75.0, 100.0]
    spark = render_sparkline(data, 0.0, 100.0, length=5)
    assert len(spark) == 5
    assert spark[0] == " "
    assert spark[-1] == "█"

    # Clamped bounds
    clamped = render_sparkline([150.0], 0.0, 100.0, length=1)
    assert clamped == "█"

    # Empty history returns baseline dashes
    empty_spark = render_sparkline([], 0.0, 100.0, length=4)
    assert empty_spark == "────"


def test_render_2row_histogram_and_empty_state():
    """Verify 2-row histogram graphs render both tiers accurately with explicit empty states."""
    from msu_llm_lab.tui.theme import render_2row_histogram

    # 1. Empty state produces clean baseline with no fabricated bars
    top_empty, bot_empty, base_empty = render_2row_histogram([], 0.0, 100.0, width=10)
    assert top_empty == " " * 10
    assert bot_empty == " " * 10
    assert base_empty == "─" * 10

    # 2. Values in lower tier (0 to 50%) populate bottom row only
    top_low, bot_low, base_low = render_2row_histogram([25.0], 0.0, 100.0, width=1)
    assert top_low == " "
    assert bot_low != " "
    assert base_low == "─"

    # 3. Values in upper tier (50% to 100%) saturate bottom row and populate top row
    top_high, bot_high, base_high = render_2row_histogram([100.0], 0.0, 100.0, width=1)
    assert top_high == "█"
    assert bot_high == "█"
    assert base_high == "─"


def test_log_buffer_rolling_history():
    """Verify LogBuffer appends messages and truncates at max_size."""
    buf = LogBuffer(max_size=5)
    for i in range(10):
        buf.add("INFO", f"Message {i}")
    recent = buf.get_recent(5)
    assert len(recent) == 5
    assert recent[-1][1] == "INFO"
    assert recent[-1][2] == "Message 9"


def test_key_reader_non_interactive(monkeypatch):
    """Verify KeyReader behavior in non-interactive piped/mocked input."""
    # Test reading single token
    mock_stdin = io.StringIO("q\n")
    monkeypatch.setattr("sys.stdin", mock_stdin)
    reader = KeyReader()
    assert reader.is_tty is False
    assert reader.read_key() == "q"

    # Test reading command string
    mock_cmd = io.StringIO(":models\n")
    monkeypatch.setattr("sys.stdin", mock_cmd)
    reader_cmd = KeyReader()
    assert reader_cmd.read_key() == ":models"

    # Test reading EOF
    mock_empty = io.StringIO("")
    monkeypatch.setattr("sys.stdin", mock_empty)
    reader_empty = KeyReader()
    assert reader_empty.read_key() == "EOF"


def test_panel_renderers():
    """Verify all Rich TUI panels render cleanly without exceptions."""
    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    # 1. Header panel
    header = build_top_header()
    console.print(header)
    assert "MSU" in console.file.getvalue()
    assert "LLM INFERENCE WORKBENCH" in console.file.getvalue()

    # 2. Navigation panel
    console.file = io.StringIO()
    nav = build_navigation_panel(active_index=0, is_focused=True)
    console.print(nav)
    out_nav = console.file.getvalue()
    assert "Model Catalogue" in out_nav
    assert "Budget & Usage" in out_nav

    # 3. Models panel
    candidates = get_portfolio()
    console.file = io.StringIO()
    models_panel = build_model_catalogue_panel(
        candidates, selected_index=0, search_query="", is_focused=True
    )
    console.print(models_panel)
    out_models = console.file.getvalue()
    assert "MODEL CATALOGUE" in out_models
    assert "google/gemini-flash-1.5" in out_models
    assert "UNVERIFIED" in out_models
    assert "live provider" in out_models

    # 4. Budget panel
    fake_status = BudgetStatus(
        total_ceiling_usd=Decimal("10.00"),
        cumulative_spend_usd=Decimal("0.00"),
        remaining_total_usd=Decimal("10.00"),
        pilot_ceiling_usd=Decimal("2.00"),
        pilot_spend_usd=Decimal("0.00"),
        remaining_pilot_usd=Decimal("2.00"),
        per_run_ceiling_usd=Decimal("0.50"),
        per_case_ceiling_usd=Decimal("0.02"),
        is_pilot_exhausted=False,
        is_total_exhausted=False,
    )
    console.file = io.StringIO()
    budget_panel = build_budget_main_panel(fake_status, is_focused=True)
    console.print(budget_panel)
    out_budget = console.file.getvalue()
    assert "BUDGET & USAGE LEDGER" in out_budget
    assert "Total Experiment Ceiling" in out_budget
    assert "provider-enforced" in out_budget

    # 5. Roadmap panels
    console.file = io.StringIO()
    console.print(build_dataset_panel(is_focused=False))
    console.print(build_eval_panel(is_focused=False))
    console.print(build_traces_panel(is_focused=False))
    out_roadmap = console.file.getvalue()
    assert "ROADMAP STEP 2" in out_roadmap
    assert "ROADMAP STEP 4" in out_roadmap
    assert "ROADMAP STEP 6" in out_roadmap

    # 6. Health panel
    items = run_system_health_checks()
    console.file = io.StringIO()
    console.print(build_health_main_panel(items, is_focused=True))
    assert "PROJECT HEALTH DIAGNOSTICS" in console.file.getvalue()

    # 7. Help panel
    console.file = io.StringIO()
    console.print(build_help_main_panel())
    assert "KEYBOARD SHORTCUTS" in console.file.getvalue()

    # 8. Bottom Panels
    console.file = io.StringIO()
    console.print(build_recent_logs_panel(LogBuffer()))
    console.print(build_budget_usage_bottom_panel(fake_status))
    console.print(build_datasets_bottom_panel())
    console.print(build_command_bar("NORMAL", "", 0, ""))
    console.print(build_footer_bar(fake_status))
    out_bottom = console.file.getvalue()
    assert "RECENT LOGS" in out_bottom
    assert "BUDGET USAGE" in out_bottom
    assert "DATASETS" in out_bottom
    assert "msu-llm-lab >" in out_bottom


def test_monochrome_selection_and_palette():
    """Verify restrained monochrome palette and inverted selection highlights."""
    from msu_llm_lab.tui.theme import STYLE_ROW_SELECTED, BG_DARK, TEXT_PRIMARY, BORDER_CHARCOAL

    assert BG_DARK == "#1C1C1C"
    assert TEXT_PRIMARY == "#D0D0D0"
    assert BORDER_CHARCOAL == "#383838"
    # Inverted selection: dark text on light gray background
    assert STYLE_ROW_SELECTED.color.name.lower() == "#1c1c1c"
    assert STYLE_ROW_SELECTED.bgcolor.name.lower() == "#d0d0d0"


def test_workbench_tui_btop_layout():
    """Verify WorkbenchTUI instantiates the 5-tier btop layout shell."""
    out_buf = io.StringIO()
    console = Console(file=out_buf, force_terminal=False, width=120, height=36)
    app_tui = WorkbenchTUI(console=console)
    app_tui.refresh_data()

    layout = app_tui.build_layout()
    assert layout["header"] is not None
    assert layout["top_row"]["nav"] is not None
    assert layout["top_row"]["sys_metrics"] is not None
    assert layout["top_row"]["inf_status"] is not None
    assert layout["main"] is not None
    assert layout["bottom_row"]["logs"] is not None
    assert layout["bottom_row"]["budget_usage"] is not None
    assert layout["bottom_row"]["datasets"] is not None
    assert layout["footer"]["command"] is not None
    assert layout["footer"]["status_line"] is not None

    console.print(layout)
    output = out_buf.getvalue()
    assert "MSU" in output
    assert "LLM INFERENCE WORKBENCH" in output
    assert "NAVIGATION" in output
    assert "SYSTEM RESOURCES" in output
    assert "MODEL CATALOGUE" in output
    assert "RECENT LOGS" in output
    assert "BUDGET USAGE" in output


def test_menu_navigation_and_pane_switching():
    """Verify arrow keys and numbers navigate menu items without reinitializing."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()
    assert app_tui.nav_index == 0
    assert app_tui.focused_pane == "NAV"

    # Navigate down to Budget & Usage
    app_tui.handle_key("DOWN")
    assert app_tui.nav_index == 1

    # Jump directly to Project Health using '6'
    app_tui.handle_key("6")
    assert app_tui.nav_index == 5
    assert app_tui.focused_pane == "MAIN"

    # Switch pane back to NAV using Tab
    app_tui.handle_key("TAB")
    assert app_tui.focused_pane == "NAV"

    # Switch pane to MAIN using Enter
    app_tui.handle_key("ENTER")
    assert app_tui.focused_pane == "MAIN"


def test_command_bar_editing_and_dispatch():
    """Verify editable command bar text entry, backspace, cursor movement, and command execution."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # 1. Activate command mode with ':'
    app_tui.handle_key(":")
    assert app_tui.input_mode == InputMode.COMMAND
    assert app_tui.text_buffer == ""

    # 2. Type 'budget'
    for ch in "budget":
        app_tui.handle_key(ch)
    assert app_tui.text_buffer == "budget"
    assert app_tui.cursor_pos == 6

    # 3. Test Backspace
    app_tui.handle_key("BACKSPACE")
    assert app_tui.text_buffer == "budge"
    assert app_tui.cursor_pos == 5

    # 4. Re-add 't'
    app_tui.handle_key("t")
    assert app_tui.text_buffer == "budget"

    # 5. Execute command with ENTER
    app_tui.handle_key("ENTER")
    assert app_tui.input_mode == InputMode.NORMAL
    assert app_tui.nav_index == 1  # Switched to Budget & Usage!
    assert "Budget & Usage" in app_tui.feedback_message

    # 6. Test invalid command feedback
    app_tui.handle_key(":")
    for ch in "unknowncmd":
        app_tui.handle_key(ch)
    app_tui.handle_key("ENTER")
    assert "Unknown command: ':unknowncmd'" in app_tui.feedback_message

    # 7. Test Esc cancellation
    app_tui.handle_key(":")
    app_tui.handle_key("a")
    app_tui.handle_key("ESC")
    assert app_tui.input_mode == InputMode.NORMAL
    assert app_tui.text_buffer == ""
    assert app_tui.feedback_message == "Command cancelled"


def test_search_filter_mode():
    """Verify search filter mode (/prompt) filters candidate models in real time."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # Activate search with '/'
    app_tui.handle_key("/")
    assert app_tui.input_mode == InputMode.SEARCH

    # Type 'qwen'
    for ch in "qwen":
        app_tui.handle_key(ch)
    assert app_tui.search_query == "qwen"

    # Confirm filter with ENTER
    app_tui.handle_key("ENTER")
    assert app_tui.input_mode == InputMode.NORMAL
    assert app_tui.search_query == "qwen"

    # Clear filter with ESC
    app_tui.handle_key("ESC")
    assert app_tui.search_query == ""


def test_help_modal_toggle():
    """Verify '?' toggles the keyboard shortcut help overlay."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    app_tui.handle_key("?")
    assert app_tui.input_mode == InputMode.HELP

    app_tui.handle_key("ESC")
    assert app_tui.input_mode == InputMode.NORMAL


def test_graceful_handling_small_terminal():
    """Verify small terminal dimensions render a notice without crashing."""
    console = Console(file=io.StringIO(), force_terminal=True, width=40, height=10)
    app_tui = WorkbenchTUI(console=console)
    app_tui.refresh_data()

    layout = app_tui.build_layout()
    console.print(layout)
    output = console.file.getvalue()
    assert "Terminal size (40x10)" in output


def test_cli_ui_and_interactive_invocation():
    """Verify CLI 'ui' and '--interactive' commands launch TUI without error."""
    res_ui = runner.invoke(app, ["ui"], env={"COLUMNS": "120", "LINES": "36"})
    assert res_ui.exit_code == 0
    assert "LLM INFERENCE WORKBENCH" in res_ui.stdout
    assert "NAVIGATION" in res_ui.stdout

    res_inter = runner.invoke(app, ["--interactive"], env={"COLUMNS": "120", "LINES": "36"})
    assert res_inter.exit_code == 0
    assert "LLM INFERENCE WORKBENCH" in res_inter.stdout
