"""The TUI roster must show every participant with readiness, exact blockers, source and evaluation state."""

from io import StringIO

from rich.console import Console

from msu_llm_lab.models.registry import roster_status
from msu_llm_lab.tui.app import WorkbenchTUI
from msu_llm_lab.tui.screens import build_roster_panel


def render(panel, width=520):
    buf = StringIO()
    Console(file=buf, width=width, force_terminal=False, color_system=None).print(panel)
    return buf.getvalue()


def test_roster_panel_lists_all_ten_with_status_blockers_source_and_eval_state():
    text = render(build_roster_panel(roster_status(), 0, True))
    for name in ("Gemini 3.5 Flash Lite", "GPT-4.1-mini", "Jev 1.14", "GPT-4.1 Nano", "GPT-4o mini", "Qwen3.5", "DeepSeek", "Mistral Small", "GPT-4.1 (frontier", "Claude Sonnet 5.5"):
        assert name in text, name
    assert "Production Baseline" in text and "Frontier Reference" in text and "Candidate" in text
    assert "SMOKE_READY" in text and "READY" in text and "response_shape" in text and "usage_fields" in text
    assert "NOT_YET_EVALUATED" in text and "UNKNOWN" in text and "openrouter.ai" in text and "2026-10-15" in text
    flat = " ".join(text.split())
    assert "benchmark route only, production parity not established" in flat and "student-derived data blocked" in flat


def test_blocked_participants_show_their_exact_reason(monkeypatch):
    from registry_helpers import block_route
    block_route(monkeypatch, "qwen3.5-35b-a3b")
    text = render(build_roster_panel(roster_status(), 0, False))
    assert "BLOCKED" in text and "route" in text


def test_roster_command_toggles_the_view_without_disturbing_the_catalogue():
    tui = WorkbenchTUI(console=Console(file=StringIO(), width=200))
    assert tui.show_roster is False
    tui.execute_command(":roster")
    assert tui.show_roster is True and tui.nav_index == 0 and "roster" in tui.feedback_message.lower()
    tui.execute_command(":models")
    assert tui.show_roster is False and tui.nav_index == 0
