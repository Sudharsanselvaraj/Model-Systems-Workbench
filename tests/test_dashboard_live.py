"""The live dashboard: truthful empty states, read-only refresh, no duplicated subscriptions, keyboard navigation, narrow terminals."""

import threading
from decimal import Decimal
from io import StringIO

import pytest
from rich.console import Console

from msu_llm_lab.core.budget import BudgetStatus
from msu_llm_lab.observability.events import EventStore
from msu_llm_lab.observability.monitor import LiveMonitor
from msu_llm_lab.tui.app import WorkbenchTUI
from msu_llm_lab.tui.live_views import build_execution_header
from msu_llm_lab.tui.screens import NAV_KEYS
from obs_helpers import finished, intent_pred, label_row, make_prep, started, write_labels, write_preds


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run inside an empty folder so relative data paths (data/classification_prep) never touch the real project files."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def preds_dir():
    from msu_llm_lab.core.config import settings
    return settings.reports_dir / "classification"


def make_tui(width=207, height=55):
    buf = StringIO()
    return WorkbenchTUI(console=Console(file=buf, width=width, height=height, force_terminal=False)), buf


def render(tui, buf) -> str:
    buf.seek(0)
    buf.truncate()
    tui.console.print(tui.build_layout())
    return buf.getvalue()


def now_events(store, n=3, model="jev-1.14"):
    import time
    from datetime import datetime, timezone
    from msu_llm_lab.observability.events import RequestEvent
    t = time.time()
    for i in range(n):
        iso = lambda x: datetime.fromtimestamp(x, timezone.utc).isoformat()  # noqa: E731
        store.append(RequestEvent(kind="REQUEST_STARTED", request_id=f"q{i}", run_id="r1", case_id=f"C{i}", model_key=model, ts=iso(t - 3 + i * 0.1)))
        store.append(RequestEvent(kind="REQUEST_SUCCEEDED", request_id=f"q{i}", run_id="r1", case_id=f"C{i}", model_key=model, ts=iso(t - 2 + i * 0.1), status="OK",
                                  latency_ms=120.0 + i, prompt_tokens=100, completion_tokens=10, cost_basis="PROVIDER_REPORTED", actual_cost_usd=Decimal("0.0001")))


def test_empty_state_is_explicit_everywhere(workdir):
    tui, buf = make_tui()
    out = render(tui, buf)
    assert "NO LIVE INFERENCE YET" in out and "PRODUCTION_PARITY_NOT_ESTABLISHED" in out and "GROUND TRUTH PENDING" in out
    assert "API requests 0" in out
    tui.goto("latency")
    assert "NO LIVE INFERENCE YET" in render(tui, buf)
    tui.goto("quality")
    assert "NO PREDICTION RECORDS YET" in render(tui, buf)


def test_graphs_and_counts_update_when_real_events_are_recorded(workdir):
    tui, buf = make_tui()
    assert not tui.inf_telemetry.is_live
    now_events(EventStore(), n=3)
    assert tui.refresh_live() == 6
    assert tui.inf_telemetry.is_live and tui.inf_telemetry.sample_count == 3
    out = render(tui, buf)
    assert "API requests 3" in out and "IDLE" in out
    tui.goto("latency")
    out = render(tui, buf)
    assert "P50" in out or "P50 ms" in out
    vm = tui.get_view()
    assert vm.rows and vm.rows[0].cells[1] == "3"


def test_refresh_sends_nothing(workdir, monkeypatch):
    import socket
    import urllib.request
    from msu_llm_lab.providers.http import RetryingPoster
    from msu_llm_lab.providers.openrouter import OpenRouterAdapter

    def boom(*a, **k):
        raise AssertionError("the dashboard must never send a request")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(RetryingPoster, "post", boom)
    monkeypatch.setattr(OpenRouterAdapter, "generate", boom)
    tui, buf = make_tui()
    now_events(EventStore(), n=2)
    for key in NAV_KEYS:
        tui.goto(key)
        tui.refresh_live()
        tui.tick()
        tui.handle_key("r")
        tui.handle_key("ENTER")
        render(tui, buf)
    assert tui.monitor.poll_count > 5


def test_repeated_refresh_never_duplicates_events_or_subscriptions(workdir):
    tui, _ = make_tui()
    store = EventStore()
    now_events(store, n=2)
    monitor, tail = tui.monitor, tui.monitor.tail
    threads = threading.active_count()
    tui.refresh_live()
    count = len(tui.monitor.events)
    for _ in range(25):
        tui.refresh_live()
        tui.goto("latency")
        tui.goto("overview")
    assert len(tui.monitor.events) == count == 4
    assert tui.monitor is monitor and tui.monitor.tail is tail          # one subscription for the whole session
    assert threading.active_count() == threads                           # polling is cooperative: no background task per view or redraw
    assert len(tail.seen) == 4


def test_tick_polls_only_when_the_interval_has_elapsed(workdir):
    tui, _ = make_tui()
    before = tui.monitor.poll_count
    tui.tick()
    assert tui.monitor.poll_count == before                              # just refreshed in __init__: not due yet
    tui.last_live_poll -= 10
    tui.tick()
    assert tui.monitor.poll_count == before + 1


def test_close_releases_the_subscription(workdir):
    tui, _ = make_tui()
    tui.close()
    assert tui.monitor.closed and tui.monitor.events == [] and tui.monitor.poll() == 0


def test_quality_view_needs_approved_labels_and_never_substitutes_predictions(workdir):
    prep = make_prep(workdir / "data" / "classification_prep")
    write_preds(preds_dir(), "run1", [intent_pred("C1", "QUERY", top=0.9), intent_pred("C2", "GREETING", top=0.9)])
    tui, buf = make_tui()
    tui.goto("quality")
    out = render(tui, buf)
    assert "GROUND TRUTH PENDING" in out and "Accuracy" in out
    vm = tui.get_view()
    assert vm.rows[0].cells[4] == "GROUND TRUTH PENDING"
    write_labels(prep, [label_row("C1", "INTENT", "QUERY"), label_row("C2", "INTENT", "QUERY")])
    tui.refresh_live()
    assert tui.get_view().rows[0].cells[4] == "50.0%"


def test_agreement_view_labels_agreement_as_not_correctness(workdir):
    write_preds(preds_dir(), "a", [intent_pred(f"C{i}", "QUERY", model="m-a", run="ra") for i in range(3)])
    write_preds(preds_dir(), "b", [intent_pred(f"C{i}", "QUERY", model="m-b", run="rb") for i in range(3)])
    tui, buf = make_tui()
    tui.goto("agreement")
    out = render(tui, buf)
    assert "not correctness" in out.lower()
    assert tui.get_view().rows[0].detail[-1].startswith("Unanimous errors vs approved labels: GROUND TRUTH PENDING")


def test_every_section_renders_at_the_reference_size_and_on_small_terminals(workdir):
    now_events(EventStore(), n=2)
    for width, height in ((207, 55), (120, 40), (90, 24)):
        tui, buf = make_tui(width, height)
        for key in NAV_KEYS:
            tui.goto(key)
            out = render(tui, buf)
            assert out.strip(), key


def test_too_narrow_terminal_gets_a_notice_instead_of_a_broken_layout(workdir):
    tui, buf = make_tui(60, 20)
    assert "too compact" in render(tui, buf)


def test_keyboard_reaches_all_twelve_primary_sections_and_back(workdir):
    tui, _ = make_tui()
    seen = []
    for key in "1234567890-=":
        tui.handle_key(key)
        seen.append(tui.section)
    assert seen == NAV_KEYS[:12]
    tui.goto("overview")
    for _ in range(len(NAV_KEYS)):
        tui.handle_key("]")
    assert tui.section == "overview"
    tui.handle_key("[")
    assert tui.section == NAV_KEYS[-1]
    tui.handle_key("TAB")
    tui.handle_key("SHIFT_TAB")
    tui.handle_key("SHIFT_TAB")
    assert tui.focused_pane in ("NAV", "MAIN", "COMMAND")


def test_row_cursor_moves_inside_views_and_stays_in_range(workdir):
    prep = make_prep(workdir / "data" / "classification_prep")
    write_preds(preds_dir(), "r", [intent_pred("C1", "QUERY", model=m, run=f"r-{m}") for m in ("m1", "m2", "m3")])
    tui, _ = make_tui()
    tui.goto("quality")
    for _ in range(10):
        tui.handle_key("DOWN")
    assert tui.content_cursor == 2
    tui.handle_key("HOME")
    assert tui.content_cursor == 0
    tui.handle_key("END")
    assert tui.content_cursor == 2
    tui.handle_key("ENTER")
    assert tui.feedback_message.startswith("Main Intent")


def test_commands_open_the_new_and_the_preserved_workflows(workdir):
    tui, _ = make_tui()
    for cmd, sec in ((":latency", "latency"), (":calibration", "calibration"), (":agreement", "agreement"), (":lead", "lead"), (":labels", "dataset"),
                     (":eval", "evaluations"), (":health", "health"), (":validate", "validation"), (":traces", "traces"), (":budget", "budget"),
                     (":settings", "settings"), (":roster", "roster"), (":overview", "overview")):
        tui.execute_command(cmd)
        assert tui.section == sec, cmd


def test_simulated_spend_never_appears_as_actual_in_the_header(workdir):
    tui, _ = make_tui()
    b = BudgetStatus(total_ceiling_usd=Decimal("10"), cumulative_spend_usd=Decimal("0.2"), remaining_total_usd=Decimal("9.8"), pilot_ceiling_usd=Decimal("2"),
                     pilot_spend_usd=Decimal("0"), remaining_pilot_usd=Decimal("2"), per_run_ceiling_usd=Decimal("0.5"), per_case_ceiling_usd=Decimal("0.02"),
                     simulated_spend_usd=Decimal("3.5"), spend_by_basis={"ESTIMATED_FROM_CATALOG": Decimal("0.2")}, unresolved_spend_count=1,
                     is_pilot_exhausted=False, is_total_exhausted=False)
    tui.live_ctx.budget = b
    console = Console(file=StringIO(), width=207)
    console.print(build_execution_header(tui.live_ctx))
    out = console.file.getvalue()
    assert "COUNTED vs CAPS $0.2000 = ACTUAL $0.0000 + ESTIMATED $0.2000" in out and "OTHER" not in out and "LEGACY-SIMULATED (not counted) $3.5000" in out and "UNRESOLVED 1 (live runs blocked)" in out


def test_dataset_view_reports_statuses_and_refresh_never_touches_annotations(workdir):
    prep = make_prep(workdir / "data" / "classification_prep")
    write_labels(prep, [label_row("C1", "INTENT", "QUERY", status="PENDING_REVIEW")])
    before = {p: p.read_bytes() for p in prep.rglob("*") if p.is_file()}
    tui, buf = make_tui()
    tui.goto("dataset")
    out = render(tui, buf)
    for needle in ("prep-test", "NOT SELECTED", "PRODUCTION_PARITY_NOT_ESTABLISHED", "heuristic", "workflow not built"):
        assert needle in out
    for _ in range(5):
        tui.handle_key("r")
    assert {p: p.read_bytes() for p in prep.rglob("*") if p.is_file()} == before


def test_system_resource_values_are_measured_or_unavailable_never_placeholders(workdir):
    tui, _ = make_tui()
    st = tui.sys_telemetry
    assert len(st.cpu) == len(st.mem) == len(st.disk) == 1               # one real sample at start, nothing pre-filled
    cpu, _, _ = st.get_cpu_info()
    assert cpu is None or cpu >= 0
    st.mem.append((0.0, None))
    assert st.get_mem_info()[0] is None


def test_dashboard_shows_the_full_roster_of_ten_in_the_robustness_view(workdir):
    tui, _ = make_tui()
    tui.goto("robustness")
    assert len(tui.get_view().rows) == len(__import__('msu_llm_lab.models.registry', fromlist=['REGISTRY']).REGISTRY) == 11


def test_request_graphs_distinguish_an_empty_current_window_from_recorded_history(workdir):
    import time
    from datetime import datetime, timezone
    from msu_llm_lab.observability.events import RequestEvent
    store = EventStore()
    old = time.time() - 3600
    iso = lambda x: datetime.fromtimestamp(x, timezone.utc).isoformat()  # noqa: E731
    store.append(RequestEvent(kind="REQUEST_STARTED", request_id="q", run_id="r", case_id="C", model_key="m", ts=iso(old)))
    store.append(RequestEvent(kind="REQUEST_SUCCEEDED", request_id="q", run_id="r", case_id="C", model_key="m", ts=iso(old + 1), status="OK", latency_ms=50.0))
    tui, buf = make_tui()
    assert tui.inf_telemetry.is_live and tui.inf_telemetry.n_in_window == 0
    out = render(tui, buf)
    assert "NO REQUESTS IN THE CURRENT 5-MINUTE WINDOW" in out and "1 historical request" in out
    assert "NO LIVE INFERENCE YET: graphs" not in out
    store.append(RequestEvent(kind="REQUEST_STARTED", request_id="q2", run_id="r", case_id="C2", model_key="m", ts=iso(time.time() - 5)))
    store.append(RequestEvent(kind="REQUEST_SUCCEEDED", request_id="q2", run_id="r", case_id="C2", model_key="m", ts=iso(time.time() - 4), status="OK", latency_ms=70.0))
    tui.refresh_live()
    assert tui.inf_telemetry.n_in_window == 1 and "NO REQUESTS IN THE CURRENT" not in render(tui, buf)


def test_spend_is_presented_consistently_as_counted_actual_and_estimated(workdir):
    from msu_llm_lab.tui.screens import build_inference_status_panel
    tui, buf = make_tui()
    b = BudgetStatus(total_ceiling_usd=Decimal("10"), cumulative_spend_usd=Decimal("0.0040"), remaining_total_usd=Decimal("9.996"), pilot_ceiling_usd=Decimal("2"),
                     pilot_spend_usd=Decimal("0.0040"), remaining_pilot_usd=Decimal("1.996"), per_run_ceiling_usd=Decimal("0.5"), per_case_ceiling_usd=Decimal("0.02"),
                     spend_by_basis={"ESTIMATED_FROM_CATALOG": Decimal("0.0040")}, is_pilot_exhausted=False, is_total_exhausted=False)
    out = StringIO()
    Console(file=out, width=60).print(build_inference_status_panel(b, tui.live_ctx.agg))
    t = out.getvalue()
    assert "Counted" in t and "$0.0040" in t and "actual" in t and "$0.0000" in t and "estimated" in t
    tui.live_ctx.budget = b
    o2 = StringIO()
    Console(file=o2, width=207).print(build_execution_header(tui.live_ctx))
    assert "COUNTED vs CAPS $0.0040 = ACTUAL $0.0000 + ESTIMATED $0.0040" in o2.getvalue()
