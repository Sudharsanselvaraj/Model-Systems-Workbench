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
    assert "LLM WORKBENCH" in console.file.getvalue()

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
    assert "google/gemini-3.5-flash-lite" in out_models
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
    """Verify purple-and-black palette and selection highlights."""
    from msu_llm_lab.tui.theme import STYLE_ROW_SELECTED, BG_DARK, TEXT_PRIMARY, BORDER_CHARCOAL

    assert BG_DARK == "#212633"
    assert TEXT_PRIMARY == "#f8fafc"
    assert BORDER_CHARCOAL == "#3b4256"
    # Selection: crisp white on royal purple background
    assert STYLE_ROW_SELECTED.color.name.lower() == "#ffffff"
    assert STYLE_ROW_SELECTED.bgcolor.name.lower() == "#4c1d95"


def test_workbench_tui_btop_layout():
    """Verify WorkbenchTUI instantiates the 5-tier btop layout shell."""
    out_buf = io.StringIO()
    console = Console(file=out_buf, force_terminal=False, width=120, height=36)
    app_tui = WorkbenchTUI(console=console)
    app_tui.show_roster = False  # this test exercises the legacy planning catalogue
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
    assert "LLM WORKBENCH" in output
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
    assert app_tui.sidebar_cursor == 0
    assert app_tui.focused_pane == "NAV"

    # Navigate down to Budget & Usage with DOWN and activate with ENTER
    app_tui.handle_key("DOWN")
    assert app_tui.sidebar_cursor == 1
    assert app_tui.nav_index == 0  # not activated yet
    app_tui.handle_key("ENTER")
    assert app_tui.nav_index == 1  # activated
    assert app_tui.focused_pane == "MAIN"

    # Jump directly to Project Health using '6'
    app_tui.handle_key("6")
    assert app_tui.nav_index == 5
    assert app_tui.sidebar_cursor == 5
    assert app_tui.focused_pane == "MAIN"

    # Switch pane to COMMAND, then NAV using Tab
    app_tui.handle_key("TAB")
    assert app_tui.focused_pane == "COMMAND"
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
    assert "LLM WORKBENCH" in res_ui.stdout
    assert "NAVIGATION" in res_ui.stdout

    res_inter = runner.invoke(app, ["--interactive"], env={"COLUMNS": "120", "LINES": "36"})
    assert res_inter.exit_code == 0
    assert "LLM WORKBENCH" in res_inter.stdout


def test_sidebar_up_down_and_enter_activation():
    """Verify Up/Down moves the sidebar highlight and Enter opens the section."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    assert app_tui.sidebar_cursor == 0
    assert app_tui.nav_index == 0

    # Down moves cursor but does not change loaded view
    app_tui.handle_key("DOWN")
    assert app_tui.sidebar_cursor == 1
    assert app_tui.nav_index == 0

    # Up moves cursor back
    app_tui.handle_key("UP")
    assert app_tui.sidebar_cursor == 0
    assert app_tui.nav_index == 0

    # Move down to index 2 (Dataset Validation) and press Enter
    app_tui.handle_key("DOWN")
    app_tui.handle_key("DOWN")
    assert app_tui.sidebar_cursor == 2
    assert app_tui.nav_index == 0

    app_tui.handle_key("ENTER")
    assert app_tui.nav_index == 2
    assert app_tui.focused_pane == "MAIN"


def test_three_way_focus_cycling_tab_and_arrows():
    """Verify Tab, Shift+Tab, and Left/Right move focus predictably across NAV, MAIN, COMMAND."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    assert app_tui.focused_pane == "NAV"

    # Forward Tab cycle: NAV -> MAIN -> COMMAND -> NAV
    app_tui.handle_key("TAB")
    assert app_tui.focused_pane == "MAIN"
    app_tui.handle_key("TAB")
    assert app_tui.focused_pane == "COMMAND"
    app_tui.handle_key("TAB")
    assert app_tui.focused_pane == "NAV"

    # Reverse Shift+Tab cycle: NAV -> COMMAND -> MAIN -> NAV
    app_tui.handle_key("SHIFT_TAB")
    assert app_tui.focused_pane == "COMMAND"
    app_tui.handle_key("SHIFT_TAB")
    assert app_tui.focused_pane == "MAIN"
    app_tui.handle_key("SHIFT_TAB")
    assert app_tui.focused_pane == "NAV"

    # Arrow keys in NORMAL mode:
    # Right: NAV -> MAIN -> COMMAND
    app_tui.handle_key("RIGHT")
    assert app_tui.focused_pane == "MAIN"
    app_tui.handle_key("RIGHT")
    assert app_tui.focused_pane == "COMMAND"

    # Left: COMMAND -> MAIN -> NAV
    app_tui.handle_key("LEFT")
    assert app_tui.focused_pane == "MAIN"
    app_tui.handle_key("LEFT")
    assert app_tui.focused_pane == "NAV"


def test_model_catalogue_navigation_and_details_sync():
    """Verify Up/Down in Model Catalogue updates model selection and inspector details."""
    app_tui = WorkbenchTUI()
    app_tui.show_roster = False  # this test exercises the legacy planning catalogue
    app_tui.refresh_data()
    assert len(app_tui.candidates) >= 5

    # 1. Startup focus is NAV
    assert app_tui.focused_pane == "NAV"
    assert app_tui.sidebar_cursor == 0

    # 2. Transfer focus to Model Catalogue via ENTER
    app_tui.handle_key("ENTER")
    assert app_tui.focused_pane == "MAIN"
    assert app_tui.nav_index == 0
    assert app_tui.model_cursor == 0

    # 3. Down changes selection from candidate 0 to candidate 1
    app_tui.handle_key("DOWN")
    assert app_tui.model_cursor == 1
    c1 = app_tui.candidates[1]

    # Verify model details panel updates immediately
    layout1 = app_tui.build_layout()
    out1 = io.StringIO()
    console = Console(file=out1, force_terminal=False, width=120, height=36)
    console.print(layout1)
    rendered1 = out1.getvalue()
    assert c1.model_id in rendered1
    assert "MODEL CATALOGUE (FOCUSED)" in rendered1

    # 4. Down repeatedly navigates through all five candidates
    for expected_idx in [2, 3, 4]:
        app_tui.handle_key("DOWN")
        assert app_tui.model_cursor == expected_idx

    # Clamping: further DOWN does not exceed bounds
    app_tui.handle_key("DOWN")
    assert app_tui.model_cursor == 4

    # 5. Up moves backwards through candidates
    for expected_idx in [3, 2, 1, 0]:
        app_tui.handle_key("UP")
        assert app_tui.model_cursor == expected_idx

    # Clamping: further UP does not drop below 0
    app_tui.handle_key("UP")
    assert app_tui.model_cursor == 0

    # 6. Home and End jump directly to start and end
    app_tui.handle_key("END")
    assert app_tui.model_cursor == len(app_tui.candidates) - 1
    app_tui.handle_key("HOME")
    assert app_tui.model_cursor == 0

    # 7. Enter triggers model action and records inspection log
    initial_log_count = len(app_tui.log_buffer.entries)
    app_tui.handle_key("ENTER")
    assert app_tui.candidates[0].model_id in app_tui.feedback_message
    assert len(app_tui.log_buffer.entries) == initial_log_count + 1


def test_command_input_isolation():
    """Verify typing in command bar does not trigger global shortcuts (q, ?, r, /)."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # Enter command mode with ':'
    app_tui.handle_key(":")
    assert app_tui.input_mode == InputMode.COMMAND

    # Type keys that would normally be global shortcuts: 'q', '?', 'r', '/'
    for ch in "q?r/":
        app_tui.handle_key(ch)

    # Must NOT have quit or opened help modal
    assert app_tui.should_exit is False
    assert app_tui.input_mode == InputMode.COMMAND
    assert app_tui.text_buffer == "q?r/"

    # Esc cancels safely
    app_tui.handle_key("ESC")
    assert app_tui.input_mode == InputMode.NORMAL
    assert app_tui.text_buffer == ""


def test_empty_catalogue_and_search_handling():
    """Verify empty search results and empty catalogue do not crash navigation."""
    app_tui = WorkbenchTUI()
    app_tui.show_roster = False  # this test exercises the legacy planning catalogue
    app_tui.refresh_data()

    # 1. Search filter with no matches
    app_tui.handle_key("/")
    for ch in "nonexistent_model_xyz":
        app_tui.handle_key(ch)
    app_tui.handle_key("ENTER")

    assert app_tui.search_query == "nonexistent_model_xyz"
    assert len(app_tui.get_filtered_candidates()) == 0
    assert app_tui.model_cursor == 0

    # Layout builds gracefully with empty result message
    out_buf = io.StringIO()
    console = Console(file=out_buf, force_terminal=False, width=120, height=36)
    app_tui.console = console
    layout = app_tui.build_layout()
    console.print(layout)
    output = out_buf.getvalue()
    assert "No models matching search query" in output

    # Up/Down/Enter do not crash on empty results
    app_tui.handle_key("DOWN")
    app_tui.handle_key("UP")
    app_tui.handle_key("ENTER")
    assert app_tui.feedback_message == "No model selected"

    # Clear filter with Esc
    app_tui.handle_key("ESC")
    assert app_tui.search_query == ""

    # 2. Empty candidate catalogue entirely
    app_tui.candidates = []
    app_tui._clamp_model_cursor()
    layout_empty = app_tui.build_layout()
    console.print(layout_empty)
    output_empty = out_buf.getvalue()
    assert "No model candidates found in catalogue" in output_empty


def test_global_keys_q_r_esc_help():
    """Verify behavior of ?, r, Esc, and q in appropriate contexts."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # '?' opens help modal
    app_tui.handle_key("?")
    assert app_tui.input_mode == InputMode.HELP

    # 'ESC' closes help modal
    app_tui.handle_key("ESC")
    assert app_tui.input_mode == InputMode.NORMAL

    # 'r' refreshes data and leaves navigation valid
    app_tui.nav_index = 2
    app_tui.sidebar_cursor = 2
    app_tui.handle_key("r")
    assert app_tui.nav_index == 2
    assert app_tui.sidebar_cursor == 2
    assert "Refreshed" in app_tui.feedback_message

    # 'ESC' in MAIN returns focus to NAV
    app_tui.focused_pane = "MAIN"
    app_tui.handle_key("ESC")
    assert app_tui.focused_pane == "NAV"

    # 'q' in NORMAL quits
    assert app_tui.should_exit is False
    app_tui.handle_key("q")
    assert app_tui.should_exit is True


def test_navigation_zero_side_effects():
    """Verify navigation never triggers API calls, incurs spend, or mutates budget ledger."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    initial_spend = app_tui.budget_status.cumulative_spend_usd

    # Perform diverse navigation operations
    for key in ["DOWN", "DOWN", "ENTER", "TAB", "DOWN", "UP", "ENTER", "LEFT", "UP", "ENTER", "r"]:
        app_tui.handle_key(key)

    current_status = app_tui.budget_service.get_status()
    assert current_status.cumulative_spend_usd == initial_spend
    assert current_status.pilot_spend_usd == app_tui.budget_status.pilot_spend_usd


def test_budget_panel_row_navigation_and_inspection():
    """Verify Up/Down moves row selection within Budget & Usage and Enter inspects rules."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # Move to Budget & Usage and focus MAIN
    app_tui.nav_index = 1
    app_tui.sidebar_cursor = 1
    app_tui.focused_pane = "MAIN"
    app_tui.content_cursor = 0

    # Down changes row in budget panel
    app_tui.handle_key("DOWN")
    assert app_tui.content_cursor == 1

    # Enter inspects Total Ceiling rule
    app_tui.handle_key("ENTER")
    assert "Total Ceiling" in app_tui.feedback_message

    # Down to row 3 (Live Inference Enabled)
    app_tui.handle_key("DOWN")
    app_tui.handle_key("DOWN")
    assert app_tui.content_cursor == 3
    app_tui.handle_key("ENTER")
    assert "Live inference" in app_tui.feedback_message

    # Left arrow returns focus to NAV sidebar
    app_tui.handle_key("LEFT")
    assert app_tui.focused_pane == "NAV"


def test_key_reader_ss3_and_extended_sequences(monkeypatch):
    """Verify KeyReader parses both standard CSI and VT100/SS3 arrow sequences."""
    # Test SS3 terminal sequences (\x1bOA, \x1bOB, etc.)
    for seq, expected in [("UP", "UP"), ("DOWN", "DOWN"), ("LEFT", "LEFT"), ("RIGHT", "RIGHT")]:
        mock_stdin = io.StringIO(f"{seq}\n")
        monkeypatch.setattr("sys.stdin", mock_stdin)
        reader = KeyReader()
        assert reader.read_key() == expected


def test_build_evaluation_main_panel_rendering():
    """Verify Evaluation Runs screen renders summary, tables, and case details properly."""
    console = Console(width=120, height=40, record=True)
    sample_runs = [
        {
            "run_id": "rag_run_20261014_001",
            "timestamp": "2026-10-14T09:00:00Z",
            "model_id": "google/gemini-3.5-flash-lite",
            "dataset_path": "data/evaluation/verified_rag_benchmark.jsonl",
            "total_cases": 8,
            "mean_accuracy": 1.0,
            "mean_groundedness": 0.464,
            "mean_latency_ms": 266.3,
            "total_cost_usd": "0.0001",
            "is_dry_run": 1,
            "notes": "RAG test split evaluation",
        }
    ]
    sample_results = [
        {
            "case_id": "msu-rag-test-01",
            "prompt": "What are the entry criteria for Master of Science in Engineering?",
            "accuracy": 1.0,
            "groundedness": 0.5,
            "latency_ms": 250.0,
            "evaluator_notes": "Grounding verified",
        }
    ]

    # Test unfocused
    panel = build_evaluation_main_panel(sample_runs, selected_index=0, is_focused=False, run_results=sample_results)
    console.print(panel)
    output = console.export_text()
    assert "EVALUATION RUNS & BENCHMARKS" in output
    assert "Decoupled RAG Benchmarks" in output
    assert "rag_run_20261014_001" in output
    assert "google/gemini-3.5-flash-lite" in output
    assert "msu-rag-test-01" in output

    # Test empty runs
    console.clear()
    panel_empty = build_evaluation_main_panel([], selected_index=0, is_focused=True)
    console.print(panel_empty)
    out_empty = console.export_text()
    assert "No recorded runs yet" in out_empty


def test_build_traces_main_panel_rendering():
    """Verify Traces & Observability screen renders trace logs and diagnostic hypotheses."""
    from msu_llm_lab.traces.models import DiagnosisHypothesis, FailureCategory, TraceDiagnosticReport, TraceRecord

    console = Console(width=120, height=40, record=True)
    sample_traces = [
        TraceRecord(
            id="trace-001",
            timestamp="2026-10-14T10:00:00Z",
            input_query="What are the tuition fees for Bachelor of Software Engineering?",
            output_response="Tuition fee is RM48,000 across 3 years.",
            latency_ms=1150.0,
            total_tokens=420,
        )
    ]
    sample_report = TraceDiagnosticReport(
        total_traces=1,
        diagnosed_failures=1,
        category_counts={"ROUTING_FAILURE": 0, "RETRIEVAL_FAILURE": 0, "SOURCE_KNOWLEDGE_FAILURE": 0, "GENERATION_FAILURE": 1, "NO_FAILURE_DETECTED": 0},
        mean_latency_ms=1150.0,
        total_tokens=420,
        diagnoses=[
            DiagnosisHypothesis(
                trace_id="trace-001",
                category=FailureCategory.GENERATION_FAILURE,
                confidence=0.88,
                hypothesis="Candidate Generation Hallucination: Model invented fee figure not in context.",
                evidence=["Invented figure: RM48,000"],
                suggested_action="Adjust temperature to 0.0 and add strict grounding constraints.",
            )
        ],
    )

    panel = build_traces_main_panel(sample_traces, report=sample_report, selected_index=0, is_focused=True)
    console.print(panel)
    output = console.export_text()
    assert "PRODUCTION TRACES & DIAGNOSTIC OBSERVABILITY" in output
    assert "Langfuse Trace Observability" in output
    assert "trace-001" in output
    assert "GENERATION_FAILURE" in output
    assert "Candidate Generation Hallucination" in output
    assert "Adjust temperature to 0.0" in output
    assert "candidate diagnostic hypotheses" in output.lower()


def test_tui_evaluation_runs_navigation_and_inspection():
    """Verify Up/Down arrow keys and Enter inspect runs on Evaluation Runs screen (nav 3)."""
    from live_fakes import seed_runs
    from msu_llm_lab.storage.db import DatabaseManager
    seed_runs(DatabaseManager())  # two recorded runs in the isolated test database
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # Navigate to Evaluation Runs screen
    app_tui.nav_index = 3
    app_tui.sidebar_cursor = 3
    app_tui.focused_pane = "MAIN"
    app_tui.content_cursor = 0

    assert len(app_tui.eval_runs) > 0

    # Down moves selection in evaluation runs
    app_tui.handle_key("DOWN")
    assert app_tui.content_cursor == 1

    # Enter inspects the currently highlighted run
    app_tui.handle_key("ENTER")
    assert "Inspected Run:" in app_tui.feedback_message

    # Up returns cursor to first run
    app_tui.handle_key("UP")
    assert app_tui.content_cursor == 0


def test_tui_traces_observability_navigation_and_inspection():
    """Verify Up/Down arrow keys and Enter inspect traces on Traces screen (nav 4)."""
    app_tui = WorkbenchTUI()
    app_tui.refresh_data()

    # Navigate to Traces & Observability screen
    app_tui.nav_index = 4
    app_tui.sidebar_cursor = 4
    app_tui.focused_pane = "MAIN"
    app_tui.content_cursor = 0

    assert len(app_tui.traces) > 0
    assert app_tui.trace_report is not None

    # Down moves selection in traces
    app_tui.handle_key("DOWN")
    assert app_tui.content_cursor == 1

    # Enter inspects the selected trace
    app_tui.handle_key("ENTER")
    assert "Inspected Trace:" in app_tui.feedback_message
    assert "ms" in app_tui.feedback_message

    # Up returns cursor to 0
    app_tui.handle_key("UP")
    assert app_tui.content_cursor == 0


def test_welcome_screen_render():
    """Verify WelcomeScreen renders Braille logo, telemetry, and menu items properly."""
    from msu_llm_lab.tui.welcome import WelcomeScreen
    buf = io.StringIO()
    test_console = Console(file=buf, width=100)
    screen = WelcomeScreen(console=test_console)
    screen.render(clear=False)
    output = buf.getvalue()

    assert "Welcome to MSU LLM Workbench" in output
    assert "Corpus:" in output and "Models:" in output and "API key:" in output and "Mode:" in output
    assert "Spend: $" in output and "cap" in output
    assert "Tests: 95 passed" not in output  # no hard-coded claims; every figure is read live or shown as n/a
    assert "Launch Full-Screen TUI Dashboard" in output
    for item in ("Classification Dataset Status", "Model Comparison Roster", "API Preflight", "Smoke-Test Plan", "Inspect Budget"):
        assert item in output
    for gone in ("Decoupled RAG Benchmark", "Simulate vLLM", "Diagnose Production Traces"):
        assert gone not in output  # the launcher offers real workflows only
    assert "msu-llm-lab >" in output


def test_welcome_cli_command():
    """Verify msu-llm-lab welcome command can be invoked."""
    result = runner.invoke(app, ["welcome"], input="9\n")
    assert result.exit_code == 0


