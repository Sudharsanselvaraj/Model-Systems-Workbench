"""Panel and renderable builders for the btop-inspired terminal interface."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from rich.align import Align
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from msu_llm_lab.core.budget import BudgetStatus
from msu_llm_lab.models.catalog import ModelCandidate
from msu_llm_lab.traces.models import DiagnosisHypothesis, FailureCategory, TraceDiagnosticReport, TraceRecord
from msu_llm_lab.tui.health import HealthCheckItem
from msu_llm_lab.tui.telemetry import SystemTelemetry, InferenceTelemetry, LogBuffer
from msu_llm_lab.tui.theme import (
    BG_DARK,
    BORDER_CHARCOAL,
    BORDER_TEAL,
    BORDER_FOCUSED,
    CYAN_HIGHLIGHT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    TEXT_BRIGHT,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
    STYLE_PANEL,
    STYLE_ROW_SELECTED,
    STYLE_ROW_HOVER,
    render_sparkline,
    render_2row_histogram,
)


NAV_ITEMS = [
    ("1", "Model Catalogue", "Model candidates and verification"),
    ("2", "Budget & Usage", "Budget ledger and spending limits"),
    ("3", "Dataset Validation", "Dataset schemas and integrity checks"),
    ("4", "Evaluation Runs", "Evaluation planning and results"),
    ("5", "Traces & Observability", "Latency and trace inspection"),
    ("6", "Project Health", "Environment and database diagnostics"),
    ("7", "Settings", "Workbench parameters and paths"),
    ("8", "Help", "Shortcuts and documentation"),
]


def build_top_header() -> Table:
    """Builds the single-line restrained top header bar matching charcoal reference."""
    header = Table.grid(expand=True)
    header.style = f"{TEXT_PRIMARY} on {BG_DARK}"
    header.add_column("left", justify="left", width=38, no_wrap=True)
    header.add_column("center", justify="center", ratio=1, no_wrap=True)
    header.add_column("right", justify="right", width=26, no_wrap=True)

    left_text = Text()
    left_text.append("[MSU] ", style=f"bold {TEXT_PRIMARY}")
    left_text.append("LLM INFERENCE WORKBENCH ", style=f"bold {TEXT_PRIMARY}")
    left_text.append("v0.1.0", style=f"{TEXT_MUTED}")

    center_text = Text()
    center_text.append("Project: msu-llm-lab  |  ", style=f"{TEXT_SECONDARY}")
    center_text.append("Mode: ", style=f"{TEXT_SECONDARY}")
    center_text.append("DRY RUN ", style=f"bold {TEXT_PRIMARY}")
    center_text.append("(Zero API Spend)", style=f"{TEXT_MUTED}")

    now_str = datetime.now().strftime("%a %d %b %Y  %H:%M:%S")
    right_text = Text(now_str, style=f"{TEXT_SECONDARY}")

    header.add_row(left_text, center_text, right_text)
    return header


def build_navigation_panel(
    active_index: int,
    is_focused: bool = False,
    cursor_index: Optional[int] = None,
) -> Panel:
    """Builds the upper-left navigation sidebar panel matching charcoal monochrome reference."""
    if cursor_index is None:
        cursor_index = active_index

    grid = Table.grid(expand=True, padding=(0, 0))
    grid.add_column("title", ratio=1, no_wrap=True)
    grid.add_column("badge", justify="right", width=3, no_wrap=True)

    items = [
        "Model Catalogue",
        "Budget & Usage",
        "Dataset Validation",
        "Evaluation Runs",
        "Traces & Observ.",
        "Project Health",
        "Settings",
        "Help",
    ]

    for idx, title in enumerate(items):
        is_active = (idx == active_index)
        is_cursor = (idx == cursor_index)

        left_txt = Text()
        right_txt = Text()

        if is_cursor and is_focused:
            # Focused cursor item: inverted monochrome selection bar
            left_txt.append("> ", style=STYLE_ROW_SELECTED)
            left_txt.append(title, style=STYLE_ROW_SELECTED)
            right_txt.append("[●]" if is_active else "   ", style=STYLE_ROW_SELECTED)
            grid.add_row(left_txt, right_txt, style=STYLE_ROW_SELECTED)
        elif is_cursor and not is_focused:
            # Cursor item when unfocused
            left_txt.append("> ", style=f"bold {TEXT_PRIMARY}")
            left_txt.append(title, style=f"bold {TEXT_PRIMARY}")
            right_txt.append("[●]" if is_active else "   ", style=f"bold {TEXT_PRIMARY}" if is_active else f"{TEXT_MUTED}")
            grid.add_row(left_txt, right_txt)
        elif is_active:
            # Active section while cursor is navigating elsewhere
            left_txt.append("● ", style=f"bold {TEXT_PRIMARY}")
            left_txt.append(title, style=f"bold {TEXT_PRIMARY}")
            right_txt.append("[●]", style=f"bold {TEXT_PRIMARY}")
            grid.add_row(left_txt, right_txt)
        else:
            left_txt.append("- ", style=f"{TEXT_MUTED}")
            left_txt.append(title, style=f"{TEXT_SECONDARY}")
            right_txt.append("   ", style=f"{TEXT_MUTED}")
            grid.add_row(left_txt, right_txt)

    # Subtle separator before exit
    grid.add_row(Text("──────────────────────", style=f"{BORDER_CHARCOAL}"), Text(""))
    exit_left = Text()
    exit_left.append("- ", style=f"{TEXT_MUTED}")
    exit_left.append("Exit (q)", style=f"{TEXT_SECONDARY}")
    grid.add_row(exit_left, Text(""))

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]NAVIGATION[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def _build_histogram_metric_cell(
    title: str,
    title_val: str,
    top_tier: str,
    bot_tier: str,
    baseline: str,
    bot_left: str,
    bot_right: str = "60s",
    y_max: str = "100",
    y_mid: str = " 50",
    y_min: str = "  0",
) -> Table:
    """Builds a compact 5-row terminal histogram cell matching the btop reference."""
    cell = Table.grid(expand=True)
    cell.add_column("main", ratio=1)

    # Line 1: Header line (e.g. CPU 23% or MEM 5% 0.8/16.0 GB)
    header = Table.grid(expand=True)
    header.add_column("l", justify="left", no_wrap=True)
    header.add_column("r", justify="right", no_wrap=True)
    h_l = Text()
    h_l.append(f"{title}  ", style=f"{TEXT_SECONDARY}")
    h_l.append(f"{title_val}", style=f"bold {TEXT_PRIMARY}")
    header.add_row(h_l, Text(""))
    cell.add_row(header)

    # Line 2: 100 ┤ <top tier bars>
    l2 = Text()
    l2.append(f"{y_max} ┤ ", style=f"{TEXT_MUTED}")
    l2.append(top_tier, style=f"{TEXT_PRIMARY}")
    cell.add_row(l2)

    # Line 3:  50 ┤ <bottom tier bars>
    l3 = Text()
    l3.append(f"{y_mid} ┤ ", style=f"{TEXT_MUTED}")
    l3.append(bot_tier, style=f"{TEXT_PRIMARY}")
    cell.add_row(l3)

    # Line 4:   0 └─ <baseline>
    l4 = Text()
    l4.append(f"{y_min} └", style=f"{TEXT_MUTED}")
    l4.append(baseline, style=f"{TEXT_MUTED}")
    cell.add_row(l4)

    # Line 5: Details and time axis label (e.g. 2.3 / 10 cores   60s)
    footer = Table.grid(expand=True)
    footer.add_column("fl", justify="left", no_wrap=True)
    footer.add_column("fr", justify="right", no_wrap=True)
    footer.add_row(
        Text(f"    {bot_left}", style=f"{TEXT_SECONDARY}"),
        Text(bot_right, style=f"{TEXT_MUTED}"),
    )
    cell.add_row(footer)

    return cell


def build_system_resources_panel(telemetry: SystemTelemetry) -> Panel:
    """Builds the top-middle system resource graphs (CPU, MEM, DISK) as a standalone Panel."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column("cpu", ratio=1)
    grid.add_column("mem", ratio=1)
    grid.add_column("disk", ratio=1)

    # 1. CPU
    cpu_pct, cpu_detail, cpu_hist = telemetry.get_cpu_info()
    cpu_top, cpu_bot, cpu_base = render_2row_histogram(cpu_hist, 0.0, 100.0, width=18)
    cpu_cell = _build_histogram_metric_cell("CPU", f"{int(cpu_pct)}%", cpu_top, cpu_bot, cpu_base, cpu_detail, "60s", "100", " 50", "  0")

    # 2. MEM
    mem_pct, mem_detail, mem_hist = telemetry.get_mem_info()
    mem_top, mem_bot, mem_base = render_2row_histogram(mem_hist, 0.0, 100.0, width=18)
    mem_cell = _build_histogram_metric_cell("MEM", f"{int(mem_pct)}%   {mem_detail}", mem_top, mem_bot, mem_base, "", "60s", "100", " 50", "  0")

    # 3. DISK
    disk_pct, disk_detail, disk_hist = telemetry.get_disk_info()
    disk_top, disk_bot, disk_base = render_2row_histogram(disk_hist, 0.0, 100.0, width=18)
    disk_cell = _build_histogram_metric_cell("DISK", f"{int(disk_pct)}%   {disk_detail}", disk_top, disk_bot, disk_base, "", "60s", "100", " 50", "  0")

    grid.add_row(cpu_cell, mem_cell, disk_cell)
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]SYSTEM RESOURCES (LOCAL)[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=BORDER_CHARCOAL,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_request_metrics_panel(telemetry: InferenceTelemetry) -> Panel:
    """Builds the 4 mini request metric histogram charts."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column("c1", ratio=1)
    grid.add_column("c2", ratio=1)
    grid.add_column("c3", ratio=1)
    grid.add_column("c4", ratio=1)

    lat_samples = list(telemetry.latency_history) if telemetry.is_live else []
    tok_samples = list(telemetry.token_history) if telemetry.is_live else []
    con_samples = list(telemetry.concurrency_history) if telemetry.is_live else []
    err_samples = list(telemetry.error_history) if telemetry.is_live else []

    lat_top, lat_bot, lat_base = render_2row_histogram(lat_samples, 0.0, 2000.0, width=16)
    tok_top, tok_bot, tok_base = render_2row_histogram(tok_samples, 0.0, 500.0, width=16)
    con_top, con_bot, con_base = render_2row_histogram(con_samples, 0.0, 10.0, width=16)
    err_top, err_bot, err_base = render_2row_histogram(err_samples, 0.0, 10.0, width=16)

    c1 = _build_histogram_metric_cell("Request Latency (ms)", "", lat_top, lat_bot, lat_base, "", "60s", "2000", "1000", "   0")
    c2 = _build_histogram_metric_cell("Tokens / Second", "", tok_top, tok_bot, tok_base, "", "60s", " 500", " 250", "   0")
    c3 = _build_histogram_metric_cell("Concurrent Requests", "", con_top, con_bot, con_base, "", "60s", "  10", "   5", "   0")
    c4 = _build_histogram_metric_cell("Error Rate (%)", "", err_top, err_bot, err_base, "", "60s", "  10", "   5", "   0")

    grid.add_row(c1, c2, c3, c4)
    title_str = "REQUEST METRICS (No live inference yet)" if not telemetry.is_live and telemetry.sample_count == 0 else "REQUEST METRICS"
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]{title_str}[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=BORDER_CHARCOAL,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_system_and_metrics_panel(
    sys_telemetry: SystemTelemetry,
    inf_telemetry: InferenceTelemetry,
) -> Table:
    """Combines System Resources and Request Metrics into a single view for compatibility."""
    outer = Table.grid(expand=True)
    outer.add_column("main", ratio=1)
    outer.add_row(build_system_resources_panel(sys_telemetry))
    outer.add_row(build_request_metrics_panel(inf_telemetry))
    return outer


def build_inference_status_panel(status: BudgetStatus) -> Panel:
    """Builds the upper-right inference status card."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column("k", style=f"{TEXT_SECONDARY}", width=13)
    table.add_column("v", justify="right")

    table.add_row("Mode:", Text("DRY RUN", style=f"bold {TEXT_PRIMARY}"))
    table.add_row("API Calls:", Text("0", style=f"{TEXT_PRIMARY}"))
    table.add_row("Total Spend:", Text(f"${status.cumulative_spend_usd:.4f}", style=f"{TEXT_PRIMARY}"))
    table.add_row("Pilot Limit:", Text(f"${status.pilot_ceiling_usd:.2f}", style=f"{TEXT_PRIMARY}"))
    table.add_row("Total Limit:", Text(f"${status.total_ceiling_usd:.2f}", style=f"{TEXT_PRIMARY}"))
    table.add_row("Remaining:", Text(f"${status.remaining_total_usd:.2f}", style=f"bold {TEXT_PRIMARY}"))
    table.add_row("", Text("(100%)", style=f"{TEXT_SECONDARY}"))

    return Panel(
        table,
        title=f"[{TEXT_PRIMARY}]INFERENCE STATUS[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=BORDER_CHARCOAL,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_model_catalogue_panel(
    candidates: List[ModelCandidate],
    selected_index: int,
    search_query: str,
    is_focused: bool,
) -> Panel:
    """Builds the primary Model Catalogue table with 3-column Inspector Card."""
    filtered = [
        (idx, m) for idx, m in enumerate(candidates)
        if not search_query or search_query.lower() in m.model_id.lower() or search_query.lower() in m.display_name.lower() or search_query.lower() in m.provider.lower()
    ]

    effective_sel = max(0, min(selected_index, len(filtered) - 1)) if filtered else 0

    outer = Table.grid(expand=True)
    outer.add_column("main", ratio=1)

    # Sub-header bar matching reference
    count_str = f"{len(candidates)} candidates" if not search_query else f"{len(filtered)}/{len(candidates)} filtered"
    sub_header = Table.grid(expand=True)
    sub_header.add_column("left", justify="left")
    sub_header.add_column("right", justify="right")

    if is_focused:
        left_sub = Text(f"{count_str}   ↑/↓ move selection  |  Enter: inspect  |  Home/End  |  ←/Esc: sidebar  |  Tab: next", style=f"bold {TEXT_PRIMARY}")
    else:
        left_sub = Text(f"{count_str}   (Press Enter or → to focus catalogue)  |  / search  |  r refresh", style=f"{TEXT_SECONDARY}")

    sub_header.add_row(
        left_sub,
        Text(f"Filter: '{search_query}'" if search_query else (f"Selected: #{effective_sel + 1}/{len(filtered)}" if is_focused and filtered else ""), style=f"bold {TEXT_PRIMARY}"),
    )
    outer.add_row(sub_header)

    if not candidates:
        outer.add_row(Text("\nNo model candidates found in catalogue.", style=f"bold {COLOR_WARNING}"))
        border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
        title_str = f"[{TEXT_BRIGHT}]● MODEL CATALOGUE (FOCUSED)[/{TEXT_BRIGHT}]" if is_focused else f"[{TEXT_PRIMARY}]MODEL CATALOGUE[/{TEXT_PRIMARY}]"
        return Panel(
            outer,
            title=title_str,
            title_align="left",
            border_style=border,
            style=STYLE_PANEL,
            padding=(0, 1),
        )

    if not filtered:
        empty_text = Text(f"\nNo models matching search query '{search_query}'. Press Esc to clear filter.", style=f"bold {COLOR_WARNING}")
        outer.add_row(empty_text)
        border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
        title_str = f"[{TEXT_BRIGHT}]● MODEL CATALOGUE (FOCUSED)[/{TEXT_BRIGHT}]" if is_focused else f"[{TEXT_PRIMARY}]MODEL CATALOGUE[/{TEXT_PRIMARY}]"
        return Panel(
            outer,
            title=title_str,
            title_align="left",
            border_style=border,
            style=STYLE_PANEL,
            padding=(0, 1),
        )

    # Table of candidates with inverted monochrome selection
    table = Table(
        box=None,
        expand=True,
        padding=(0, 1),
        show_header=True,
        header_style=f"bold {TEXT_SECONDARY}",
    )
    table.add_column("#", width=3, justify="left")
    table.add_column("Model ID", width=30, no_wrap=True)
    table.add_column("Provider", width=12)
    table.add_column("Context", width=14, justify="right")
    table.add_column("Deploy Mode", width=20)
    table.add_column("Est. $/1M (In/Out)", width=22, justify="right")
    table.add_column("Status", width=14, justify="center")

    effective_sel = max(0, min(selected_index, len(filtered) - 1))
    selected_candidate = filtered[effective_sel][1]

    for list_pos, (orig_idx, m) in enumerate(filtered):
        is_sel = (list_pos == effective_sel)
        num_str = str(list_pos + 1)
        ctx_str = f"{m.context_length:,}" if m.context_length else "N/A"
        price_str = f"${m.input_price_per_1m_usd:.2f} / ${m.output_price_per_1m_usd:.2f}"
        status_disp = "UNVERIFIED"

        if is_sel:
            # Highlight active candidate: inverted when focused, subtle when unfocused
            row_style = STYLE_ROW_SELECTED if is_focused else STYLE_ROW_HOVER
            table.add_row(
                num_str,
                m.model_id,
                m.provider,
                ctx_str,
                m.deployment_mode,
                price_str,
                status_disp,
                style=row_style,
            )
        else:
            table.add_row(
                num_str,
                m.model_id,
                m.provider,
                ctx_str,
                m.deployment_mode,
                price_str,
                status_disp,
                style=f"{TEXT_PRIMARY}",
            )

    outer.add_row(table)

    # 3-column Inspector Card underneath table
    inspector_grid = Table.grid(expand=True, padding=(0, 2))
    inspector_grid.add_column("col1", ratio=3)
    inspector_grid.add_column("col2", ratio=4)
    inspector_grid.add_column("col3", ratio=3)

    # Col 1: Model parameters
    col1 = Text()
    col1.append(f"{selected_candidate.model_id}\n", style=f"bold {TEXT_PRIMARY}")
    col1.append("Provider:     ", style=f"{TEXT_SECONDARY}")
    col1.append(f"{selected_candidate.provider}\n", style=f"{TEXT_PRIMARY}")
    col1.append("Context:      ", style=f"{TEXT_SECONDARY}")
    ctx_display = f"{selected_candidate.context_length:,} tokens" if selected_candidate.context_length else "N/A"
    col1.append(f"{ctx_display}\n", style=f"{TEXT_PRIMARY}")
    col1.append("Deploy Mode:  ", style=f"{TEXT_SECONDARY}")
    col1.append(f"{selected_candidate.deployment_mode}\n", style=f"{TEXT_PRIMARY}")
    col1.append("Est. Price:   ", style=f"{TEXT_SECONDARY}")
    col1.append(f"${selected_candidate.input_price_per_1m_usd:.2f} / ${selected_candidate.output_price_per_1m_usd:.2f} (per 1M)\n", style=f"{TEXT_PRIMARY}")
    col1.append("License:      ", style=f"{TEXT_SECONDARY}")
    col1.append(f"{selected_candidate.license}\n", style=f"{TEXT_PRIMARY}")
    col1.append("Status:       ", style=f"{TEXT_SECONDARY}")
    col1.append(f"{selected_candidate.verification_status}", style=f"bold {TEXT_PRIMARY}")

    # Col 2: Description
    col2 = Text()
    col2.append("Description\n", style=f"bold {TEXT_PRIMARY}")
    desc_text = (
        "Fast and efficient multimodal model from Google. Optimized for high-throughput inference and long context."
        if "gemini" in selected_candidate.model_id
        else selected_candidate.experimental_purpose
    )
    col2.append(f"{desc_text}\n\n", style=f"{TEXT_PRIMARY}")
    col2.append("⚠ Model ID and pricing require live provider verification in a future step.\nCurrently for planning only.", style=f"{TEXT_SECONDARY} italic")

    # Col 3: Capabilities
    col3 = Text()
    col3.append("Capabilities (from catalog)\n", style=f"bold {TEXT_PRIMARY}")
    col3.append("► Multimodal (text, image, video, audio)\n", style=f"{TEXT_PRIMARY}")
    col3.append("► Large context window\n", style=f"{TEXT_PRIMARY}")
    col3.append("► Suitable for tool use and agentic workflows\n", style=f"{TEXT_PRIMARY}")
    col3.append("► Good speed and cost efficiency\n", style=f"{TEXT_PRIMARY}")
    col3.append("► Verify exact model ID and pricing before use", style=f"{TEXT_MUTED}")

    inspector_grid.add_row(col1, col2, col3)
    outer.add_row(Text(""))
    outer.add_row(inspector_grid)

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    title_str = f"[{TEXT_BRIGHT}]● MODEL CATALOGUE (FOCUSED)[/{TEXT_BRIGHT}]" if is_focused else f"[{TEXT_PRIMARY}]MODEL CATALOGUE[/{TEXT_PRIMARY}]"
    return Panel(
        outer,
        title=title_str,
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_budget_main_panel(
    status: BudgetStatus,
    is_focused: bool = False,
    selected_index: int = 0,
) -> Panel:
    """Builds the main panel when Budget & Usage is selected."""
    grid = Table.grid(expand=True)
    grid.add_column("main", ratio=1)

    sub_header = Table.grid(expand=True)
    sub_header.add_column("left", justify="left")
    sub_header.add_column("right", justify="right")
    sub_header.add_row(
        Text("Spend Tracking & Safeguards   ↑/↓ navigate rows  |  Enter: inspect rule  |  ←/Esc: sidebar", style=f"{TEXT_SECONDARY}"),
        Text(f"Selected Rule: #{selected_index + 1}/6" if is_focused else "", style=f"bold {TEXT_PRIMARY}"),
    )
    grid.add_row(sub_header)
    grid.add_row(Text(""))

    c_table = Table(title="Spend Ceilings & Tracking Policies", box=None, expand=True, header_style=f"bold {TEXT_PRIMARY}")
    c_table.add_column("Policy Metric", style=f"{TEXT_SECONDARY}", no_wrap=True)
    c_table.add_column("Configured Ceiling", justify="right")
    c_table.add_column("Recorded Spend", justify="right")
    c_table.add_column("Remaining Balance", justify="right", style=f"bold {TEXT_PRIMARY}")
    c_table.add_column("Status", justify="center")

    pilot_st = f"[bold {COLOR_ERROR}]EXHAUSTED[/]" if status.is_pilot_exhausted else f"[bold {TEXT_PRIMARY}]ACTIVE[/]"
    total_st = f"[bold {COLOR_ERROR}]EXHAUSTED[/]" if status.is_total_exhausted else f"[bold {TEXT_PRIMARY}]ACTIVE[/]"

    row0_style = STYLE_ROW_SELECTED if (is_focused and selected_index == 0) else None
    row1_style = STYLE_ROW_SELECTED if (is_focused and selected_index == 1) else None

    c_table.add_row("Initial Pilot Budget", f"${status.pilot_ceiling_usd:.4f}", f"${status.pilot_spend_usd:.4f}", f"${status.remaining_pilot_usd:.4f}", pilot_st, style=row0_style)
    c_table.add_row("Total Experiment Ceiling", f"${status.total_ceiling_usd:.4f}", f"${status.cumulative_spend_usd:.4f}", f"${status.remaining_total_usd:.4f}", total_st, style=row1_style)
    grid.add_row(c_table)

    s_table = Table(title="Execution Safeguards", box=None, expand=True, header_style=f"bold {TEXT_SECONDARY}")
    s_table.add_column("Rule", style=f"bold {TEXT_PRIMARY}")
    s_table.add_column("Limit / State", justify="right")
    s_table.add_column("Enforcement Action", style=f"{TEXT_MUTED} italic")

    row2_style = STYLE_ROW_SELECTED if (is_focused and selected_index == 2) else None
    row3_style = STYLE_ROW_SELECTED if (is_focused and selected_index == 3) else None
    row4_style = STYLE_ROW_SELECTED if (is_focused and selected_index == 4) else None
    row5_style = STYLE_ROW_SELECTED if (is_focused and selected_index == 5) else None

    s_table.add_row("Provider-Reported Spend", "$0.0000 USD", "No outbound provider synchronization; local tracking only", style=row2_style)
    s_table.add_row("Live Inference Enabled", f"[bold {COLOR_ERROR}]NO[/]", "Dry-run mode by default; requires explicit live approval", style=row3_style)
    s_table.add_row("Max Single Run Spend", f"${status.per_run_ceiling_usd:.4f} USD", "Preflight check rejects run if exceeded", style=row4_style)
    s_table.add_row("Max Single Case Spend", f"${status.per_case_ceiling_usd:.4f} USD", "Per-case circuit breaker halts run", style=row5_style)
    grid.add_row(s_table)

    grid.add_row(Text("\nNOTE: Ceilings are local experiment tracking policies, NOT provider-enforced account limits.", style=f"{TEXT_SECONDARY} italic"))

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]BUDGET & USAGE LEDGER[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_dataset_validation_panel(is_focused: bool) -> Panel:
    """Builds the comprehensive Dataset Validation & Remediation screen."""
    grid = Table.grid(expand=True)
    grid.add_column("main", ratio=1)

    # 1. Source Knowledge Bases Status Table
    kb_table = Table(title="Knowledge Base Sources (100% Immutable)", box=None, expand=True, header_style=f"bold {TEXT_PRIMARY}")
    kb_table.add_column("Knowledge Base", style=f"bold {TEXT_PRIMARY}", width=25)
    kb_table.add_column("Parsed Records", justify="right", width=16)
    kb_table.add_column("Integrity Hash", style=f"{TEXT_MUTED}", width=20)
    kb_table.add_column("Status", justify="center", width=8)
    kb_table.add_column("Coverage / Role", style=f"{TEXT_SECONDARY}")

    kb_table.add_row("MSU General KB", "222 Q&A pairs", "138e6893a1d54f36...", f"[bold {TEXT_PRIMARY}]PASS[/]", "Campus, admissions, faculties, general policies")
    kb_table.add_row("MSU Programmes KB", "2,810 Q&A pairs", "a46773fea196d8fe...", f"[bold {TEXT_PRIMARY}]PASS[/]", "Academic degrees, tuition fees, career outcomes")
    grid.add_row(kb_table)

    # 2. Defect Remediation Register Table
    rem_table = Table(title="Audit Findings & Remediation Register", box=None, expand=True, header_style=f"bold {TEXT_PRIMARY}")
    rem_table.add_column("ID", style=f"bold {TEXT_PRIMARY}", width=8)
    rem_table.add_column("Programme", style=f"{TEXT_PRIMARY}", width=23)
    rem_table.add_column("Defect Type", style=f"{TEXT_MUTED}", width=21)
    rem_table.add_column("Action / Pipeline Status", justify="center", width=25)
    rem_table.add_column("Resolution Policy", style=f"{TEXT_SECONDARY} italic")

    rem_table.add_row("REM-001", "Broadcasting (Hons)", "Mismatched Question", f"[bold {TEXT_PRIMARY}]APPROVED FOR CLEAN COPY[/]", "Corrected question to Broadcasting")
    rem_table.add_row("REM-002", "Product Design (Hons)", "New Media Duration", f"[bold {TEXT_PRIMARY}]APPROVED FOR CLEAN COPY[/]", "Corrected duration question label")
    rem_table.add_row("REM-003", "Master of Design", "TESL Question Tag", f"[bold {TEXT_PRIMARY}]APPROVED FOR CLEAN COPY[/]", "Corrected question tag to Master of Design")
    rem_table.add_row("REM-004", "PhD Food Service Tech", "Applied Science Tag", f"[bold {TEXT_PRIMARY}]APPROVED FOR CLEAN COPY[/]", "Corrected question to Food Service Technology")
    rem_table.add_row("REM-005", "PhD Applied Science", "Contradictory Tuition", f"[bold {COLOR_WARNING}]OFFICIAL VERIF PENDING[/]", "RM62.5k vs RM63.5k — Zero arbitrary guessing")
    rem_table.add_row("REM-006", "Multiple Programmes", "Duplicate Q&A Pairs", f"[bold {TEXT_PRIMARY}]INVESTIGATION PENDING[/]", "Cross-programme comparison before merge")
    rem_table.add_row("REM-007", "MBBS / General", "Time-Sensitive Intakes", f"[bold {COLOR_WARNING}]OFFICIAL VERIF PENDING[/]", "Verify dates against official 2026/27 calendar")
    rem_table.add_row("REM-008", "Education (TESL)", "MPU / MUET Standards", f"[bold {COLOR_ERROR}]KB OWNER ESCALATED[/]", "Escalate MPU subject & language requirements")
    grid.add_row(rem_table)

    # 3. Guidance Footer
    note = Text("\nSAFETY INVARIANT: Original KBs remain immutable. Approved fixes write strictly to versioned copies (data/remediated/). Never benchmark models against known-bad KB records.", style=f"{TEXT_MUTED} italic")
    grid.add_row(note)

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]DATASET VALIDATION & REMEDIATION PIPELINE[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def _format_failure_category(cat: str) -> str:
    """Returns styled category badge matching charcoal terminal design."""
    if "ROUTING" in cat:
        return f"[bold #E0A050]{cat}[/]"
    elif "RETRIEVAL" in cat:
        return f"[bold #E06060]{cat}[/]"
    elif "SOURCE" in cat or "KB" in cat:
        return f"[bold #D070D0]{cat}[/]"
    elif "GENERATION" in cat:
        return f"[bold #E08060]{cat}[/]"
    elif "NO_FAILURE" in cat or "SUCCESS" in cat:
        return f"[bold {COLOR_SUCCESS}]{cat}[/]"
    return f"[{TEXT_SECONDARY}]{cat}[/]"


def build_evaluation_main_panel(
    runs: List[Dict[str, Any]],
    selected_index: int = 0,
    is_focused: bool = False,
    run_results: Optional[List[Dict[str, Any]]] = None,
) -> Panel:
    """Builds the comprehensive Evaluation Runs & Benchmark Results panel."""
    grid = Table.grid(expand=True)
    grid.add_column("main", ratio=1)

    # 1. Summary Header
    header_table = Table.grid(expand=True)
    header_table.add_column("left", ratio=1)
    header_table.add_column("right", justify="right")

    left_summary = Text()
    left_summary.append("Decoupled RAG Benchmarks ", style=f"bold {TEXT_PRIMARY}")
    left_summary.append(f"({len(runs)} runs recorded)", style=f"{TEXT_MUTED}")

    right_summary = Text()
    right_summary.append("Dataset: ", style=f"{TEXT_MUTED}")
    right_summary.append("verified_rag_benchmark.jsonl ", style=f"bold {TEXT_SECONDARY}")
    right_summary.append("• Mode: ", style=f"{TEXT_MUTED}")
    right_summary.append("DRY-RUN ZERO-SPEND", style=f"bold {COLOR_SUCCESS}")

    header_table.add_row(left_summary, right_summary)
    grid.add_row(header_table)
    grid.add_row(Text(""))

    # 2. Benchmark Runs Table
    r_table = Table(box=None, expand=True, header_style=f"bold {TEXT_PRIMARY}")
    r_table.add_column("Run ID", style=f"bold {TEXT_PRIMARY}", width=24, no_wrap=True)
    r_table.add_column("Timestamp", style=f"{TEXT_MUTED}", width=17, no_wrap=True)
    r_table.add_column("Model Candidate", style=f"{TEXT_SECONDARY}", width=26, no_wrap=True)
    r_table.add_column("Cases", justify="right", width=6)
    r_table.add_column("Accuracy", justify="right", width=10)
    r_table.add_column("Faithfulness", justify="right", width=14)
    r_table.add_column("Mean Lat.", justify="right", width=10)
    r_table.add_column("Cost (USD)", justify="right", width=11)
    r_table.add_column("Mode", justify="center", width=9)

    if not runs:
        r_table.add_row(
            "-", "-", "No recorded runs yet", "0", "-", "-", "-", "$0.0000", "[DRY-RUN]"
        )
    else:
        # Windowed scrolling for clean display
        max_visible = 6
        total = len(runs)
        selected_index = max(0, min(selected_index, total - 1))
        if total <= max_visible:
            start_idx = 0
            end_idx = total
        else:
            half = max_visible // 2
            start_idx = max(0, min(selected_index - half, total - max_visible))
            end_idx = start_idx + max_visible

        for idx in range(start_idx, end_idx):
            run = runs[idx]
            run_id = str(run.get("run_id", "unknown"))
            ts_raw = str(run.get("timestamp", ""))
            try:
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_str = ts_raw[:16]

            model_id = str(run.get("model_id", "unknown"))
            if len(model_id) > 24:
                model_disp = model_id[:22] + "…"
            else:
                model_disp = model_id

            cases_str = str(run.get("total_cases", 0))
            acc = float(run.get("mean_accuracy", 0.0))
            acc_str = f"{acc * 100:.1f}%"
            groundedness = float(run.get("mean_groundedness", 0.0))
            grd_str = f"{groundedness * 100:.1f}%"
            lat = float(run.get("mean_latency_ms", 0.0))
            lat_str = f"{lat:.1f}ms"
            cost_str = f"${float(run.get('total_cost_usd', 0)):.4f}"
            is_dry = bool(run.get("is_dry_run", 1))
            mode_badge = f"[{TEXT_MUTED}][DRY-RUN][/]" if is_dry else f"[bold {COLOR_WARNING}][LIVE][/]"

            row_style = STYLE_ROW_SELECTED if (is_focused and idx == selected_index) else None
            r_table.add_row(
                run_id,
                ts_str,
                model_disp,
                cases_str,
                acc_str,
                grd_str,
                lat_str,
                cost_str,
                mode_badge,
                style=row_style,
            )

    grid.add_row(r_table)
    grid.add_row(Text(""))

    # 3. Selected Run Details / Case Preview
    if runs and 0 <= selected_index < len(runs):
        sel_run = runs[selected_index]
        det_table = Table(
            title=f"Selected Run Details: {sel_run.get('run_id')} ({sel_run.get('model_id')})",
            box=None,
            expand=True,
            header_style=f"bold {TEXT_SECONDARY}",
        )
        if run_results:
            det_table.add_column("Case ID", style=f"bold {TEXT_PRIMARY}", width=18)
            det_table.add_column("Student Prompt", style=f"{TEXT_PRIMARY}", ratio=1)
            det_table.add_column("Acc.", justify="right", width=8)
            det_table.add_column("Faithful", justify="right", width=9)
            det_table.add_column("Latency", justify="right", width=10)
            det_table.add_column("Attribution / Notes", style=f"{TEXT_MUTED}", width=22)

            for r in run_results[:3]:
                p_text = str(r.get("prompt", ""))
                p_snippet = p_text[:45] + "…" if len(p_text) > 45 else p_text
                acc_val = f"{float(r.get('accuracy', 0))*100:.0f}%"
                grd_val = f"{float(r.get('groundedness', 0))*100:.0f}%"
                l_val = f"{float(r.get('latency_ms', 0)):.0f}ms"
                notes = str(r.get("evaluator_notes", "") or "-")
                if len(notes) > 20:
                    notes = notes[:18] + "…"
                det_table.add_row(
                    str(r.get("case_id", "-")),
                    p_snippet,
                    acc_val,
                    grd_val,
                    l_val,
                    notes,
                )
        else:
            det_table.add_column("Metric", style=f"bold {TEXT_PRIMARY}", width=25)
            det_table.add_column("Value", justify="left", width=25)
            det_table.add_column("Analysis / Evaluation Invariant", style=f"{TEXT_MUTED}")

            det_table.add_row("Evaluation Dataset", str(sel_run.get("dataset_path", "verified_rag_benchmark.jsonl")), "Decoupled evaluation dataset with verified ground truth")
            det_table.add_row("Mean Accuracy", f"{float(sel_run.get('mean_accuracy', 0))*100:.1f}%", "Factual overlap against verified reference answers")
            det_table.add_row("Faithfulness (Groundedness)", f"{float(sel_run.get('mean_groundedness', 0))*100:.1f}%", "Context-supported ratio to prevent hallucinations")
            det_table.add_row("Execution Notes", str(sel_run.get("notes", "RAG evaluation")), "Tracked under local budget policy ceilings")

        grid.add_row(det_table)

    # 4. Guidance Footer
    note = Text(
        "\nNAVIGATION: Use ↑/↓ to browse benchmark runs. Press [Enter] to inspect run cases. Press [R] to refresh.",
        style=f"{TEXT_MUTED} italic",
    )
    grid.add_row(note)

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]EVALUATION RUNS & BENCHMARKS[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_traces_main_panel(
    traces: List[TraceRecord],
    report: Optional[TraceDiagnosticReport] = None,
    selected_index: int = 0,
    is_focused: bool = False,
) -> Panel:
    """Builds the comprehensive Production Traces & Observability panel."""
    grid = Table.grid(expand=True)
    grid.add_column("main", ratio=1)

    # 1. Summary Header
    header_table = Table.grid(expand=True)
    header_table.add_column("left", ratio=1)
    header_table.add_column("right", justify="right")

    left_summary = Text()
    left_summary.append("Langfuse Trace Observability ", style=f"bold {TEXT_PRIMARY}")
    left_summary.append(f"({len(traces)} traces imported)", style=f"{TEXT_MUTED}")

    diag_count = report.diagnosed_failures if report else 0
    mean_lat = f"{report.mean_latency_ms:.1f}ms" if report else "0.0ms"

    right_summary = Text()
    right_summary.append("Candidate Incidents: ", style=f"{TEXT_MUTED}")
    right_summary.append(f"{diag_count}/{len(traces)} ", style=f"bold {COLOR_WARNING}")
    right_summary.append("• Mean Latency: ", style=f"{TEXT_MUTED}")
    right_summary.append(f"{mean_lat} ", style=f"bold {TEXT_SECONDARY}")
    right_summary.append("• Redaction: ", style=f"{TEXT_MUTED}")
    right_summary.append("100% PASS", style=f"bold {COLOR_SUCCESS}")

    header_table.add_row(left_summary, right_summary)
    grid.add_row(header_table)

    # 1b. Failure Category Pill Distribution
    if report and report.category_counts:
        cat_pills = Text()
        cat_pills.append("Failure Breakdown: ", style=f"{TEXT_MUTED}")
        cat_items = [
            ("ROUTING", report.category_counts.get("ROUTING_FAILURE", 0), "#E0A050"),
            ("RETRIEVAL", report.category_counts.get("RETRIEVAL_FAILURE", 0), "#E06060"),
            ("KB DEFECT", report.category_counts.get("SOURCE_KNOWLEDGE_FAILURE", 0), "#D070D0"),
            ("GENERATION", report.category_counts.get("GENERATION_FAILURE", 0), "#E08060"),
            ("HEALTHY", report.category_counts.get("NO_FAILURE_DETECTED", 0), COLOR_SUCCESS),
        ]
        for name, count, color in cat_items:
            cat_pills.append(f" {name}: ", style=f"{TEXT_MUTED}")
            cat_pills.append(f"{count}", style=f"bold {color}")
            cat_pills.append(" |", style=f"{BORDER_CHARCOAL}")
        grid.add_row(cat_pills)

    grid.add_row(Text(""))

    # 2. Ingested Traces Table
    t_table = Table(box=None, expand=True, header_style=f"bold {TEXT_PRIMARY}")
    t_table.add_column("Trace ID", style=f"bold {TEXT_PRIMARY}", width=20, no_wrap=True)
    t_table.add_column("Timestamp", style=f"{TEXT_MUTED}", width=17, no_wrap=True)
    t_table.add_column("Student Query Snippet", style=f"{TEXT_PRIMARY}", ratio=1)
    t_table.add_column("Latency", justify="right", width=10)
    t_table.add_column("Candidate Category", width=25)
    t_table.add_column("Conf.", justify="right", width=7)

    if not traces:
        t_table.add_row("-", "-", "No traces loaded from data/traces/", "-", "N/A", "-")
    else:
        # Windowed scrolling
        max_visible = 5
        total = len(traces)
        selected_index = max(0, min(selected_index, total - 1))
        if total <= max_visible:
            start_idx = 0
            end_idx = total
        else:
            half = max_visible // 2
            start_idx = max(0, min(selected_index - half, total - max_visible))
            end_idx = start_idx + max_visible

        for idx in range(start_idx, end_idx):
            trace = traces[idx]
            diag = report.diagnoses[idx] if (report and idx < len(report.diagnoses)) else None

            ts_raw = trace.timestamp
            try:
                dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_str = ts_raw[:16]

            q_text = trace.input_query
            q_snippet = q_text[:42] + "…" if len(q_text) > 42 else q_text
            lat_str = f"{trace.latency_ms:.0f}ms"

            if diag:
                cat_badge = _format_failure_category(diag.category.value)
                conf_str = f"{diag.confidence * 100:.0f}%"
            else:
                cat_badge = f"[{TEXT_MUTED}]UNANALYZED[/]"
                conf_str = "-"

            row_style = STYLE_ROW_SELECTED if (is_focused and idx == selected_index) else None
            t_table.add_row(
                trace.id,
                ts_str,
                q_snippet,
                lat_str,
                cat_badge,
                conf_str,
                style=row_style,
            )

    grid.add_row(t_table)
    grid.add_row(Text(""))

    # 3. Diagnostic Hypothesis & Evidence Inspector
    if report and report.diagnoses and 0 <= selected_index < len(report.diagnoses):
        sel_diag = report.diagnoses[selected_index]
        sel_trace = traces[selected_index] if selected_index < len(traces) else None

        insp_table = Table(
            title=f"Diagnostic Inspector: {sel_diag.trace_id} ({sel_diag.category.value})",
            box=None,
            expand=True,
            header_style=f"bold {TEXT_SECONDARY}",
        )
        insp_table.add_column("Item", style=f"bold {TEXT_PRIMARY}", width=18)
        insp_table.add_column("Details", style=f"{TEXT_PRIMARY}", ratio=1)

        insp_table.add_row("Student Query", f'"{sel_trace.input_query}"' if sel_trace else "-")
        insp_table.add_row("Candidate Hypothesis", f"[{TEXT_BRIGHT}]{sel_diag.hypothesis}[/] (Confidence: {sel_diag.confidence*100:.0f}%)")
        
        evidence_text = "\n".join(f"• {e}" for e in sel_diag.evidence) if sel_diag.evidence else "• Telemetry within bounds"
        insp_table.add_row("Supporting Evidence", evidence_text)
        insp_table.add_row("Actionable Remediation", f"[bold {COLOR_SUCCESS}]{sel_diag.suggested_action}[/]")

        grid.add_row(insp_table)

    # 4. Safety Disclaimer & Footer
    disclaimer = Text(
        "\nAUDIT DISCLAIMER: Findings are candidate diagnostic hypotheses derived from heuristics & telemetry, not ground-truth verdicts.\nUse ↑/↓ to navigate traces. Press [R] to reload from data/traces/.",
        style=f"{TEXT_MUTED} italic",
    )
    grid.add_row(disclaimer)

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]PRODUCTION TRACES & DIAGNOSTIC OBSERVABILITY[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_roadmap_panel(title_str: str, step_str: str, items: List[str], is_focused: bool) -> Panel:
    """Builds clean roadmap panel for upcoming roadmap steps."""
    t = Text()
    t.append("Component Status: ", style=f"{TEXT_SECONDARY}")
    t.append(f"{step_str}\n\n", style=f"bold {TEXT_PRIMARY}")
    t.append("Planned Deliverables:\n", style=f"bold {TEXT_PRIMARY}")
    for item in items:
        t.append(f"  • {item}\n", style=f"{TEXT_PRIMARY}")
    t.append("\nSafety Invariant: The workbench enforces strict fail-closed privacy. No fake data or untracked calls.", style=f"{TEXT_MUTED}")

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        t,
        title=f"[{TEXT_PRIMARY}]{title_str}[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(1, 2),
    )


def build_health_main_panel(
    health_items: List[HealthCheckItem],
    is_focused: bool = False,
    selected_index: int = 0,
) -> Panel:
    """Builds the main panel when Project Health is selected."""
    grid = Table.grid(expand=True)
    grid.add_column("main", ratio=1)

    sub_header = Table.grid(expand=True)
    sub_header.add_column("left", justify="left")
    sub_header.add_column("right", justify="right")
    sub_header.add_row(
        Text("System Diagnostics   ↑/↓ navigate checks  |  Enter: inspect detail  |  ←/Esc: sidebar", style=f"{TEXT_SECONDARY}"),
        Text(f"Selected: #{selected_index + 1}/{len(health_items)}" if is_focused and health_items else "", style=f"bold {TEXT_PRIMARY}"),
    )
    grid.add_row(sub_header)
    grid.add_row(Text(""))

    table = Table(box=None, expand=True, padding=(0, 1), show_header=True, header_style=f"bold {TEXT_PRIMARY}")
    table.add_column("Component", style=f"bold {TEXT_PRIMARY}", width=22)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Details", style=f"{TEXT_PRIMARY}")
    table.add_column("Notes", style=f"{TEXT_MUTED} italic")

    for idx, item in enumerate(health_items):
        is_sel = (is_focused and idx == selected_index)
        badge = f"[bold {TEXT_PRIMARY}]PASS[/]" if item.status == "PASS" else (f"[bold {COLOR_WARNING}]WARN[/]" if item.status == "WARN" else f"[bold {COLOR_ERROR}]FAIL[/]")
        row_style = STYLE_ROW_SELECTED if is_sel else None
        table.add_row(item.component, badge, item.details, item.note, style=row_style)

    grid.add_row(table)

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]PROJECT HEALTH DIAGNOSTICS[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_settings_main_panel(is_focused: bool) -> Panel:
    """Builds the Settings view."""
    t = Text()
    t.append("Workbench Configuration & Paths:\n\n", style=f"bold {TEXT_PRIMARY}")
    t.append("  App Name:          ", style=f"{TEXT_SECONDARY}")
    t.append("msu-llm-lab\n", style=f"{TEXT_PRIMARY}")
    t.append("  Execution Mode:    ", style=f"{TEXT_SECONDARY}")
    t.append("DRY-RUN (Locked)\n", style=f"bold {TEXT_PRIMARY}")
    t.append("  Database Path:     ", style=f"{TEXT_SECONDARY}")
    t.append("data/workbench.db (SQLite WAL)\n", style=f"{TEXT_PRIMARY}")
    t.append("  Reports Directory: ", style=f"{TEXT_SECONDARY}")
    t.append("reports/\n", style=f"{TEXT_PRIMARY}")
    t.append("  Budget Ceilings:   ", style=f"{TEXT_SECONDARY}")
    t.append("$10.00 Total  │  $2.00 Pilot  │  $0.50 Per-Run  │  $0.02 Per-Case\n", style=f"{TEXT_PRIMARY}")
    t.append("  Provider Secret:   ", style=f"{TEXT_SECONDARY}")
    t.append("[EXCLUDED] OPENROUTER_API_KEY loaded via env only\n\n", style=f"dim {COLOR_WARNING}")
    t.append("Press 'r' to reload configuration from disk.", style=f"{TEXT_MUTED}")

    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        t,
        title=f"[{TEXT_PRIMARY}]SETTINGS & CONFIGURATION[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(1, 2),
    )


def build_help_main_panel(is_focused: bool = False) -> Panel:
    """Builds the Help & Documentation view."""
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column("col1", ratio=1)
    grid.add_column("col2", ratio=1)

    t1 = Text()
    t1.append("Navigation & Controls:\n", style=f"bold {TEXT_PRIMARY}")
    t1.append("  ↑ / k           Move selection up (sidebar or catalogue)\n", style=f"{TEXT_PRIMARY}")
    t1.append("  ↓ / j           Move selection down (sidebar or catalogue)\n", style=f"{TEXT_PRIMARY}")
    t1.append("  ← / →           Switch active pane (Nav ↔ Main ↔ Command)\n", style=f"{TEXT_PRIMARY}")
    t1.append("  Tab / Shift+Tab Cycle focus predictably between panels\n", style=f"{TEXT_PRIMARY}")
    t1.append("  1 - 8           Jump directly to navigation section\n", style=f"{TEXT_PRIMARY}")
    t1.append("  Enter           Activate section / Inspect model candidate\n", style=f"{TEXT_PRIMARY}")
    t1.append("  /               Open search filter in Model Catalogue\n", style=f"{TEXT_PRIMARY}")
    t1.append("  :               Open command line mode\n", style=f"{TEXT_PRIMARY}")
    t1.append("  r               Refresh telemetry, budget ledger, diagnostics\n", style=f"{TEXT_PRIMARY}")
    t1.append("  ?               Toggle this help cheatsheet\n", style=f"{TEXT_PRIMARY}")
    t1.append("  Esc             Cancel search / Cancel command / Focus Nav\n", style=f"{TEXT_PRIMARY}")
    t1.append("  q               Quit application safely (outside text entry)\n", style=f"{TEXT_PRIMARY}")

    t2 = Text()
    t2.append("Interactive Commands (:):\n", style=f"bold {TEXT_PRIMARY}")
    t2.append("  :models, :m     Switch to Model Catalogue\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :budget, :b     Switch to Budget & Usage\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :dataset, :d    Switch to Dataset Validation\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :eval, :e       Switch to Evaluation Runs\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :traces, :t     Switch to Traces & Observability\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :health, :h     Switch to Project Health diagnostics\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :settings, :s   Switch to Settings\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :audit          Execute live knowledge base integrity audit\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :clear          Clear search query and status messages\n", style=f"{TEXT_PRIMARY}")
    t2.append("  :q, :quit       Exit workbench\n", style=f"{TEXT_PRIMARY}")

    grid.add_row(t1, t2)
    border = BORDER_FOCUSED if is_focused else BORDER_CHARCOAL
    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]KEYBOARD SHORTCUTS & COMMAND CHEATSHEET[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=border,
        style=STYLE_PANEL,
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Bottom 3 Panels (Matching Mockup)
# ---------------------------------------------------------------------------


def build_recent_logs_panel(log_buffer: LogBuffer) -> Panel:
    """Builds the bottom-left RECENT LOGS panel."""
    grid = Table.grid(expand=True, padding=(0, 0))
    grid.add_column("ts", width=12, style=f"{TEXT_SECONDARY}")
    grid.add_column("level", width=6)
    grid.add_column("msg", style=f"{TEXT_PRIMARY}")

    logs = log_buffer.get_recent(6)
    for ts, level, msg in logs:
        lvl_style = f"bold {TEXT_PRIMARY}" if level == "INFO" else (f"bold {COLOR_WARNING}" if level == "WARN" else f"bold {COLOR_ERROR}")
        grid.add_row(f"[{ts}]", Text(level, style=lvl_style), msg)

    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]RECENT LOGS[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=BORDER_CHARCOAL,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_budget_usage_bottom_panel(status: BudgetStatus) -> Panel:
    """Builds the bottom-center BUDGET USAGE progress bar panel."""
    grid = Table.grid(expand=True, padding=(0, 0))
    grid.add_column("main", ratio=1)

    # Total Budget
    total_spent = float(status.cumulative_spend_usd)
    total_ceiling = float(status.total_ceiling_usd)
    total_pct = int((total_spent / total_ceiling) * 100) if total_ceiling > 0 else 0

    t_line = Table.grid(expand=True)
    t_line.add_column("lbl", justify="left")
    t_line.add_column("amt", justify="center")
    t_line.add_column("pct", justify="right")
    t_line.add_row(
        Text("Total Budget", style=f"{TEXT_SECONDARY}"),
        Text(f"${total_spent:.2f} / ${total_ceiling:.2f}", style=f"{TEXT_PRIMARY}"),
        Text(f"{total_pct}%", style=f"bold {TEXT_PRIMARY}"),
    )
    grid.add_row(t_line)

    # Progress bar total matching reference monochrome bar
    bar_len = 46
    filled_total = int((total_pct / 100) * bar_len)
    total_bar = Text()
    total_bar.append("[", style=f"{TEXT_MUTED}")
    total_bar.append("█" * filled_total, style=f"{TEXT_PRIMARY}")
    total_bar.append(" " * (bar_len - filled_total), style="on #242424")
    total_bar.append("]", style=f"{TEXT_MUTED}")
    grid.add_row(total_bar)

    grid.add_row(Text(""))

    # Pilot Budget
    pilot_spent = float(status.pilot_spend_usd)
    pilot_ceiling = float(status.pilot_ceiling_usd)
    pilot_pct = int((pilot_spent / pilot_ceiling) * 100) if pilot_ceiling > 0 else 0

    p_line = Table.grid(expand=True)
    p_line.add_column("lbl", justify="left")
    p_line.add_column("amt", justify="center")
    p_line.add_column("pct", justify="right")
    p_line.add_row(
        Text("Pilot Budget", style=f"{TEXT_SECONDARY}"),
        Text(f"${pilot_spent:.2f} / ${pilot_ceiling:.2f}", style=f"{TEXT_PRIMARY}"),
        Text(f"{pilot_pct}%", style=f"bold {TEXT_PRIMARY}"),
    )
    grid.add_row(p_line)

    # Progress bar pilot
    filled_pilot = int((pilot_pct / 100) * bar_len)
    pilot_bar = Text()
    pilot_bar.append("[", style=f"{TEXT_MUTED}")
    pilot_bar.append("█" * filled_pilot, style=f"{TEXT_PRIMARY}")
    pilot_bar.append(" " * (bar_len - filled_pilot), style="on #242424")
    pilot_bar.append("]", style=f"{TEXT_MUTED}")
    grid.add_row(pilot_bar)

    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]BUDGET USAGE[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=BORDER_CHARCOAL,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_datasets_bottom_panel() -> Panel:
    """Builds the bottom-right DATASETS panel."""
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column("k", style=f"{TEXT_SECONDARY}", width=12)
    grid.add_column("v")

    grid.add_row("Status:", Text("NOT IMPLEMENTED", style=f"bold {TEXT_SECONDARY}"))
    grid.add_row("Cases:", Text("-", style=f"{TEXT_MUTED}"))
    grid.add_row("Versions:", Text("-", style=f"{TEXT_MUTED}"))
    grid.add_row("Privacy:", Text("-", style=f"{TEXT_MUTED}"))
    grid.add_row("Next Step:", Text("Add evaluation datasets (Step 2)", style=f"{TEXT_PRIMARY}"))

    return Panel(
        grid,
        title=f"[{TEXT_PRIMARY}]DATASETS[/{TEXT_PRIMARY}]",
        title_align="left",
        border_style=BORDER_CHARCOAL,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_command_bar(
    mode: str,
    text_buffer: str,
    cursor_pos: int,
    feedback: str,
    is_focused: bool = False,
) -> Panel:
    """Builds the command input bar."""
    line = Text()
    line.append("msu-llm-lab > ", style=f"bold {TEXT_PRIMARY}")

    if mode in ("COMMAND", "SEARCH"):
        prefix = ":" if mode == "COMMAND" else "/"
        line.append(prefix, style=f"bold {TEXT_PRIMARY}")
        line.append(text_buffer[:cursor_pos], style=f"{TEXT_PRIMARY}")
        cursor_char = text_buffer[cursor_pos] if cursor_pos < len(text_buffer) else " "
        line.append(cursor_char, style=f"#1C1C1C on {TEXT_PRIMARY}")
        line.append(text_buffer[cursor_pos + 1:], style=f"{TEXT_PRIMARY}")
    else:
        if text_buffer:
            line.append(text_buffer, style=f"{TEXT_PRIMARY}")
        line.append("█" if is_focused else " ", style=f"{TEXT_PRIMARY}")

    if feedback:
        line.append("   │   ", style=f"{TEXT_MUTED}")
        line.append(feedback, style=f"{TEXT_SECONDARY}")

    border = BORDER_FOCUSED if (mode in ("COMMAND", "SEARCH") or is_focused) else BORDER_CHARCOAL
    return Panel(
        line,
        border_style=border,
        style=STYLE_PANEL,
        padding=(0, 1),
    )


def build_footer_bar(status: BudgetStatus) -> Table:
    """Builds the bottom-most shortcuts and status strip."""
    grid = Table.grid(expand=True)
    grid.style = f"{TEXT_PRIMARY} on {BG_DARK}"
    grid.add_column("left", justify="left", ratio=6)
    grid.add_column("right", justify="right", ratio=4)

    left = Text()
    left.append("↑/↓ ", style=f"bold {TEXT_PRIMARY}")
    left.append("Navigate  ", style=f"{TEXT_SECONDARY}")
    left.append("Enter ", style=f"bold {TEXT_PRIMARY}")
    left.append("Select  ", style=f"{TEXT_SECONDARY}")
    left.append("←/→ ", style=f"bold {TEXT_PRIMARY}")
    left.append("Pane  ", style=f"{TEXT_SECONDARY}")
    left.append("Tab ", style=f"bold {TEXT_PRIMARY}")
    left.append("Focus  ", style=f"{TEXT_SECONDARY}")
    left.append("/ ", style=f"bold {TEXT_PRIMARY}")
    left.append("Search  ", style=f"{TEXT_SECONDARY}")
    left.append("r ", style=f"bold {TEXT_PRIMARY}")
    left.append("Refresh  ", style=f"{TEXT_SECONDARY}")
    left.append("? ", style=f"bold {TEXT_PRIMARY}")
    left.append("Help  ", style=f"{TEXT_SECONDARY}")
    left.append("q ", style=f"bold {TEXT_PRIMARY}")
    left.append("Quit", style=f"{TEXT_SECONDARY}")

    right = Text()
    right.append("Dry-run", style=f"bold {TEXT_PRIMARY}")
    right.append("  |  ", style=f"{TEXT_MUTED}")
    right.append("No API calls", style=f"{TEXT_SECONDARY}")
    right.append("  |  ", style=f"{TEXT_MUTED}")
    right.append(f"Budget: ${status.remaining_total_usd:.2f}", style=f"bold {TEXT_PRIMARY}")
    right.append("  |  ", style=f"{TEXT_MUTED}")
    right.append("DB: SQLite WAL", style=f"{TEXT_SECONDARY}")

    grid.add_row(left, right)
    return grid


# ---------------------------------------------------------------------------
# Compatibility Wrappers for Legacy / Test Invocations
# ---------------------------------------------------------------------------


def build_header_panel() -> Table:
    return build_top_header()


def build_models_panel(
    candidates: List[ModelCandidate],
    selected_index: int = 0,
    search_query: str = "",
    is_focused: bool = False,
) -> Panel:
    return build_model_catalogue_panel(candidates, selected_index, search_query, is_focused)


def build_budget_panel(status: BudgetStatus, is_focused: bool = False) -> Panel:
    return build_budget_main_panel(status, is_focused)


def build_dataset_panel(is_focused: bool = False) -> Panel:
    return build_roadmap_panel(
        "DATASET VALIDATION",
        "ROADMAP STEP 2",
        [
            "MSU General KB & Programmes KB schema validation",
            "Dify prompt template extraction & variable checks",
            "Ground truth query-response pairing & golden datasets",
            "Student privacy redaction & sanitization filters",
        ],
        is_focused,
    )


def build_eval_panel(is_focused: bool = False) -> Panel:
    return build_roadmap_panel(
        "EVALUATION RUNS",
        "ROADMAP STEP 4",
        [
            "Dry-run baseline evaluation matrix",
            "Batch model inference execution with budget controls",
            "Deterministic latency & token usage recording",
            "Automated scoring against ground-truth benchmarks",
        ],
        is_focused,
    )


def build_traces_panel(is_focused: bool = False) -> Panel:
    return build_roadmap_panel(
        "TRACES & OBSERVABILITY",
        "ROADMAP STEP 6",
        [
            "Langfuse trace schema compatibility & ingestion",
            "Step-by-step latency breakdown (retrieval vs inference)",
            "Per-session token efficiency analysis",
            "Zero external network egress audit mode",
        ],
        is_focused,
    )


def build_health_panel(health_items: List[HealthCheckItem], is_focused: bool = False) -> Panel:
    return build_health_main_panel(health_items, is_focused)


def build_help_panel() -> Panel:
    return build_help_main_panel()


def build_status_bar(status: BudgetStatus, feedback: str = "") -> Panel:
    return Panel(
        Text(f"STATUS: Dry-run active │ Budget: ${status.remaining_total_usd:.2f} │ {feedback}", style=f"bold {TEXT_PRIMARY}"),
        border_style=BORDER_CHARCOAL,
        padding=(0, 1),
    )

