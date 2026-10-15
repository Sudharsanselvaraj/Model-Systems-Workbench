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
    assert "READY" in text
    assert "not run" in text and "UNKNOWN" in text and "openrouter.ai" in text and "2026-10-15" in text
    flat = " ".join(text.split())
    assert "benchmark route only" in flat and "production parity not established" in flat and "real-data authorization" in flat


def test_long_names_and_blockers_wrap_inside_their_cells_and_never_collapse_the_columns(monkeypatch):
    from registry_helpers import block_route
    block_route(monkeypatch, "mistral-small-3.2-24b-instruct")
    rows = roster_status()
    rows[0]["blockers"] = ["route", "response_shape", "usage_fields", "data_policy", "some_very_long_blocker_name_that_used_to_squeeze_columns"]
    for width in (120, 160, 207):
        buf = StringIO()
        Console(file=buf, width=width, force_terminal=False).print(build_roster_panel(rows, 0, True, width))
        lines = buf.getvalue().splitlines()
        header = next(l for l in lines if "Participant" in l)
        for col in ("Participant", "Role", "Readiness", "Blockers"):
            assert col in header, (width, col)          # no header was truncated to a letter or two
        assert "Mistral Small" in buf.getvalue() and "BLOCKED" in buf.getvalue()


def test_blocked_participants_show_their_exact_reason(monkeypatch):
    from registry_helpers import block_route
    block_route(monkeypatch, "qwen3.5-35b-a3b")
    text = render(build_roster_panel(roster_status(), 0, False))
    assert "BLOCKED" in text and "route" in text


def test_roster_command_toggles_the_view_without_disturbing_the_catalogue():
    tui = WorkbenchTUI(console=Console(file=StringIO(), width=200))
    assert tui.show_roster is True  # the verified roster is the default home
    tui.execute_command(":catalogue")
    assert tui.show_roster is False and tui.section == "roster" and "legacy" in tui.feedback_message.lower()
    tui.execute_command(":roster")
    assert tui.show_roster is True and tui.section == "roster" and "roster" in tui.feedback_message.lower()
    tui.execute_command(":catalogue")
    tui.execute_command(":models")
    assert tui.show_roster is True


def test_roster_cursor_covers_all_ten_rows():
    tui = WorkbenchTUI(console=Console(file=StringIO(), width=200))
    tui.refresh_data()
    tui.goto("roster")
    for _ in range(30):
        tui.handle_key("DOWN")
    assert tui.model_cursor == len(__import__('msu_llm_lab.models.registry', fromlist=['REGISTRY']).REGISTRY) - 1
    tui.execute_command(":catalogue")
    tui.model_cursor = 99
    tui._clamp_model_cursor()
    assert tui.model_cursor == len(tui.get_filtered_candidates()) - 1
