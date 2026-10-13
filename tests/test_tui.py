"""Unit tests for the interactive terminal UI, navigation, health diagnostics, and CLI invocation."""

import io
from decimal import Decimal
from rich.console import Console
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.budget import BudgetStatus
from msu_llm_lab.models.catalog import ModelCandidate, get_portfolio
from msu_llm_lab.tui.app import WorkbenchTUI, ScreenState
from msu_llm_lab.tui.health import run_system_health_checks
from msu_llm_lab.tui.input import KeyReader
from msu_llm_lab.tui.screens import (
    render_welcome_screen,
    render_model_list_screen,
    render_model_detail_screen,
    render_budget_screen,
    render_health_screen,
    render_unimplemented_screen,
)

runner = CliRunner()


def test_health_checks_live():
    """Verify live system diagnostics produce real PASS results for current environment."""
    items = run_system_health_checks()
    comp_map = {item.component: item for item in items}

    # Python version check
    assert "Python Runtime" in comp_map
    assert comp_map["Python Runtime"].status == "PASS"

    # Database check
    assert "SQLite Database" in comp_map
    assert comp_map["SQLite Database"].status == "PASS"

    # Audit subsystem check
    assert "Audit Subsystem" in comp_map
    assert comp_map["Audit Subsystem"].status == "PASS"

    # Source knowledge bases immutability check
    assert "MSU General KB" in comp_map
    assert comp_map["MSU General KB"].status == "PASS"
    assert "MSU Programmes KB" in comp_map
    assert comp_map["MSU Programmes KB"].status == "PASS"


def test_key_reader_non_interactive(monkeypatch):
    """Verify KeyReader behavior in non-interactive piped/mocked input."""
    # Test reading a single command
    mock_stdin = io.StringIO("q\n")
    monkeypatch.setattr("sys.stdin", mock_stdin)
    reader = KeyReader()
    assert reader.is_tty is False
    assert reader.read_key() == "Q"

    # Test reading EOF
    mock_empty = io.StringIO("")
    monkeypatch.setattr("sys.stdin", mock_empty)
    reader_empty = KeyReader()
    assert reader_empty.read_key() == "EOF"


def test_screens_rendering_outputs():
    """Verify all Rich TUI screens render cleanly without formatting exceptions."""
    console = Console(file=io.StringIO(), force_terminal=True, width=120)

    # 1. Welcome screen
    render_welcome_screen(console, selected_index=0)
    out_welcome = console.file.getvalue()
    assert "MSU LLM INFERENCE WORKBENCH" in out_welcome
    assert "Model Catalogue" in out_welcome
    assert "Budget & Usage" in out_welcome
    assert "NOT IMPLEMENTED" in out_welcome

    # 2. Model list screen
    candidates = get_portfolio()
    console.file = io.StringIO()
    render_model_list_screen(console, candidates, selected_index=0)
    out_models = console.file.getvalue()
    assert "Candidate Model Portfolio" in out_models
    assert "UNVERIFIED_CATALOG_ENTRY" in out_models
    assert "google/gemini-flash-1.5" in out_models

    # 3. Model detail inspector
    console.file = io.StringIO()
    render_model_detail_screen(console, candidates[0])
    out_detail = console.file.getvalue()
    assert "Model Inspector:" in out_detail
    assert "INFERENCE LOCKED / EVALUATION SAFEGUARD" in out_detail
    assert "Selecting a model does NOT initiate API calls" in out_detail

    # 4. Budget screen
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
    render_budget_screen(console, fake_status)
    out_budget = console.file.getvalue()
    assert "Local Budget Ledger" in out_budget
    assert "Initial Pilot Budget" in out_budget
    assert "NOT provider-enforced account limits" in out_budget

    # 5. Health screen
    items = run_system_health_checks()
    console.file = io.StringIO()
    render_health_screen(console, items)
    out_health = console.file.getvalue()
    assert "Project Health & Diagnostics" in out_health
    assert "PASS" in out_health

    # 6. Unimplemented roadmap notice
    console.file = io.StringIO()
    render_unimplemented_screen(console, "Dataset Validation", "Step 2")
    out_unimp = console.file.getvalue()
    assert "Dataset Validation" in out_unimp
    assert "Step 2" in out_unimp


def test_workbench_tui_headless_loop(monkeypatch):
    """Verify WorkbenchTUI initializes and exits cleanly when invoked non-interactively."""
    out_buf = io.StringIO()
    console = Console(file=out_buf, force_terminal=True, width=120)
    mock_stdin = io.StringIO("")
    monkeypatch.setattr("sys.stdin", mock_stdin)

    app_tui = WorkbenchTUI(console=console)
    assert not app_tui.key_reader.is_tty
    app_tui.run()
    assert "MSU LLM INFERENCE WORKBENCH" in out_buf.getvalue()


def test_cli_ui_and_interactive_invocation():
    """Verify CLI 'ui' and '--interactive' commands launch TUI without error."""
    # Test 'msu-llm-lab ui'
    res_ui = runner.invoke(app, ["ui"], env={"COLUMNS": "120"})
    assert res_ui.exit_code == 0
    assert "MSU LLM INFERENCE WORKBENCH" in res_ui.stdout
    assert "Model Catalogue" in res_ui.stdout

    # Test 'msu-llm-lab --interactive'
    res_inter = runner.invoke(app, ["--interactive"], env={"COLUMNS": "120"})
    assert res_inter.exit_code == 0
    assert "MSU LLM INFERENCE WORKBENCH" in res_inter.stdout
