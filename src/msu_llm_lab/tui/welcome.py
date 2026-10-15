"""Interactive, developer-grade welcome screen and quick launcher for MSU LLM Workbench.

Features:
- Custom high-resolution Braille logo matching the MSU LLM Lab 3-plane branching architecture
- Mathematically centered layout across any terminal width (80 to 250+ columns)
- Zero emojis (clean, minimalist, developer-terminal typography)
- Deep purple and pitch black cyberpunk/hacker visual aesthetic
- Keyboard-driven interactive menu (arrow keys, number shortcuts, Enter to launch)
- Instant dispatch to full TUI dashboard, RAG benchmark, audit, traces, serving simulation, or budget
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from msu_llm_lab.tui.input import KeyReader
from msu_llm_lab.tui.theme import BG_DARK


# Precise Braille artwork matching vector logo geometry (26 columns x 13 lines)
LOGO_BRAILLE_STRIPPED: List[str] = [
    "                 ⢸⣶⣤⣀",
    "        ⢠⣄⡀      ⢸⣿⣿⣿⣿⣶⣤⣀",
    "        ⢸⣿⣿⣷⣦⣄⡀  ⢸⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣷⣦⣤⣀    ⢸⣿⣿⣿⣿⣿⣿⣷⡆⢸⣿⣿⡿⠿⠻⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣶⣤⣀⢸⣿⣿⣿⣿⣿⣿⣿⡇⢸⣿⡿⠃  ⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⠟⢉⣉⣉⣀⣈⣉⣠⣾⣿⣿⣿⣿⣿",
    "⣿⣿⠛⠛⢻⣿⣿⣿⠘⠿⠛⢁⣴⣿⣿⣿⡇⢹⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣀⣀⣠⣤⣤⣤⢠⣤⣤⡈⠻⣿⣿⣿⣇⣸⣿⣿⣿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣦⣄⣀⣀⡀⢀⣀⠙⢿⣿⣿⣿⣿⣿",
    "⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⣿⣿⡇⢸⣿⣷⡄  ⣿⣿⣿",
    "⠈⠙⠻⠿⣿⣿⣿⣿ ⠈⠙⠻⢿⣿⣿⣿⡇⢸⣿⣿⣷⣶⣶⣿⣿⣿",
    "     ⠉⠛⠿     ⠈⠙⠻⠇ ⠈⠙⠻⢿⣿⣿⣿⣿",
    "                      ⠈⠙⠻⢿",
]

MENU_ITEMS: List[Tuple[str, str, Optional[str]]] = [
    ("1.  Launch Full-Screen TUI Dashboard", "Live telemetry and monitors", None),
    ("2.  Run Knowledge Base Audit & Remediation", "parsed records", None),
    ("3.  Run Decoupled RAG Benchmark", "v2.1 baseline, Recall@k sweep", None),
    ("4.  Model Comparison Roster", "Readiness, routes and blockers", "NEW"),
    ("5.  Diagnose Production Traces", "Langfuse ingestion and analysis", None),
    ("6.  Simulate vLLM Serving & Hardware Sizing", "A100 / Mac Studio", None),
    ("7.  Inspect Budget & Spend Ceilings", "actual spend vs ceilings", None),
    ("8.  Settings & KB Hashes", "Providers, paths, cryptographic baseline", None),
    ("9.  Exit to Shell", "", None),
]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def launcher_status() -> Dict[str, str]:
    """Live facts for the launcher header. Every value is read from local files or the ledger; unknown values say 'n/a'."""
    from msu_llm_lab import __version__

    ver = f"v{__version__}"
    sha, branch, date = _git("rev-parse", "--short", "HEAD"), _git("rev-parse", "--abbrev-ref", "HEAD"), _git("log", "-1", "--format=%cd", "--date=format:%Y.%m.%d")
    if sha:
        ver = f"{ver}-{date or 'dev'}+{branch or 'detached'}.{sha}"
    status: Dict[str, str] = {"version": ver, "corpus": "n/a", "records": "n/a", "spend": "n/a", "key": "n/a", "models": "n/a"}
    try:
        m = json.loads(Path("reports/rag_corpus_manifest.json").read_text())
        status["corpus"] = f"{int(m['indexed_records_count']):,} clean"
        status["records"] = f"{int(m['total_records_scanned']):,} parsed records"
    except Exception:
        pass
    try:
        from msu_llm_lab.core.budget import BudgetService
        b = BudgetService().get_status()
        status["spend"] = f"${b.cumulative_spend_usd:.2f} / ${b.total_ceiling_usd:.0f} cap"
        status["ledger"] = f"${b.cumulative_spend_usd:.2f} actual (cap: ${b.total_ceiling_usd:.2f})"
    except Exception:
        pass
    key = os.environ.get("OPENROUTER_API_KEY")
    status["key"] = "set" if key and key.strip() else "not set"
    try:
        from msu_llm_lab.models.registry import roster_status
        rows = roster_status()
        ready, smoke = sum(r["readiness"] == "READY" for r in rows), sum(r["readiness"] == "SMOKE_READY" for r in rows)
        status["models"] = f"{ready} ready, {smoke} smoke-ready, {len(rows) - ready - smoke} blocked"
    except Exception:
        pass
    return status


class WelcomeScreen:
    """Renders the developer-grade launcher screen and handles interactive menu selection."""

    def __init__(self, console: Optional[Console] = None, key_reader: Optional[KeyReader] = None) -> None:
        self.console = console or Console()
        self.key_reader = key_reader or KeyReader()
        self.selected_index: int = 0

    def _build(self, compact: bool = False):
        """Builds the launcher layout. `compact` drops decorative lines so the frame fits short terminals."""
        status = launcher_status()
        self.status = status
        items = [(t, (status.get("records") if i == 1 and status.get("records") not in (None, "n/a") else
                      status.get("ledger") if i == 6 and status.get("ledger") else sub), b)
                 for i, (t, sub, b) in enumerate(MENU_ITEMS)]

        # Determine terminal and content width matching compact OpenSRE proportion
        term_cols, _ = shutil.get_terminal_size((100, 30))
        if self.console.width:
            term_cols = max(term_cols, self.console.width)
        content_width = min(max(74, term_cols - 4), 88)

        container = Table.grid(padding=(0, 0))
        container.add_column("main", width=content_width)

        # 1. Centered Braille Logo (26 cols wide, matching OpenSRE)
        logo_pad = max(0, (content_width - 26) // 2)
        if not compact:
            for line in LOGO_BRAILLE_STRIPPED:
                container.add_row(Text.from_markup(f"[bold #c4b5fd]{' ' * logo_pad}{line}[/]"))
            container.add_row(Text(""))

        # 2. Version, Title, and Subtitle (matching OpenSRE restrained hierarchy)
        container.add_row(Align.center(Text.from_markup(f"[dim #a78bfa]{status['version']}[/]")))
        container.add_row(Align.center(Text.from_markup("[bold #d8b4fe]Welcome to MSU LLM Workbench[/]")))
        container.add_row(
            Align.center(Text.from_markup("[dim #9ca3af]The local-first research laboratory for university admissions AI evaluation & serving.[/]"))
        )
        if not compact:
            container.add_row(Text(""))

        # 3. Telemetry Status Row (Zero Emojis, Clean Minimalist Monospace)
        key_style = "green" if status["key"] == "set" else "yellow"
        pills = Text.from_markup(f"Corpus: {status['corpus']}    │    Models: {status['models']}")
        pills2 = Text.from_markup(f"Spend: {status['spend']}    │    API key: [{key_style}]{status['key']}[/]    │    Mode: [bold #38bdf8]DRY-RUN[/]")
        container.add_row(Align.center(pills))
        container.add_row(Align.center(pills2))
        if not compact:
            container.add_row(Text(""))

        # 4. Announcement Box with subtle slate border (matching OpenSRE prompt box style)
        box_content = (
            "[bold #f1f3f9]MSU Admissions AI Research Environment[/]\n"
            "[dim #9ca3af]Audit the knowledge base. Benchmark retrieval with isolated ground truth. Compare models.[/]\n"
            "[dim #9ca3af]Diagnose traces. Plan and size vLLM serving. Maintain budget controls and data confidentiality.[/]"
        )
        panel = Panel(
            box_content,
            border_style="#373a4d",
            padding=(0, 2),
            width=content_width,
            expand=False,
        )
        if not compact:
            container.add_row(panel)
            container.add_row(Text(""))

        # 5. Menu Section
        container.add_row(Text.from_markup("[bold #f1f3f9]Select a workbench workflow:[/]"))
        if not compact:
            container.add_row(Text(""))

        for idx, (title, subtitle, badge) in enumerate(items):
            is_active = (idx == self.selected_index)
            prefix = " > " if is_active else "   "
            badge_str = f"  [bold #f5d0fe on #432874] {badge} [/]" if badge else ""

            # Target exact 94-column balance
            title_w = 44
            sub_w = 34

            if is_active:
                row_text = Text.from_markup(f"[bold #ffffff on #432874]{prefix}{title:<{title_w}} {subtitle:<{sub_w}}{badge_str}[/]")
            else:
                row_text = Text.from_markup(f"[dim #64748b]{prefix}[/][#f1f3f9]{title:<{title_w}}[/] [dim #8e95a5]{subtitle:<{sub_w}}[/]{badge_str}")

            container.add_row(row_text)

        if not compact:
            container.add_row(Text(""))

        # 6. Informational Disclaimer (Zero Emojis)
        container.add_row(Text.from_markup("[dim #64748b]All commands run locally. No external data is sent unless you explicitly enable a provider.[/]"))

        # 7. Prompt Line with Hotkey Hints
        prompt = Text.from_markup("[bold #c4b5fd]msu-llm-lab >[/] █")
        hints = Text.from_markup("[dim #64748b]↑↓ Navigate    Enter Select    ? Help    q Exit[/]")
        spacing = max(2, content_width - len(prompt.plain) - len(hints.plain))
        prompt_row = Text.assemble(prompt, " " * spacing, hints)
        container.add_row(prompt_row)

        return Align.center(container)

    def render(self, clear: bool = True) -> None:
        """Prints one frame (used for non-interactive output and tests). The interactive loop redraws in place instead."""
        if clear and self.key_reader.is_tty:
            self.console.clear()
        self.console.print(self._build())

    def frame_lines(self, rows: int) -> List[str]:
        """The frame as terminal lines that are guaranteed to fit `rows`: the full layout, else compact, else truncated."""
        for compact in (False, True):
            with self.console.capture() as cap:
                self.console.print(self._build(compact=compact))
            lines = cap.get().rstrip("\n").split("\n")
            if len(lines) <= rows:
                return lines
        return lines[:rows]

    def run(self) -> int:
        """Runs the interactive loop until a choice is selected. Returns the 1-based action code."""
        if not self.key_reader.is_tty:
            self.render(clear=False)
            return 0

        # Apply slate-grey background matching OpenSRE (#212633)
        try:
            sys.stdout.write(f"\033]11;{BG_DARK}\007")
            sys.stdout.flush()
        except Exception:
            pass

        # The alternate screen has no scrollback, so frames can never pile up; each frame is drawn in place.
        sys.stdout.write("\033[?1049h\033[?25l")
        sys.stdout.flush()
        last_size = None
        dirty = True
        try:
            while True:
                size = shutil.get_terminal_size((100, 30))
                if dirty or size != last_size:
                    rows = max(8, size.lines)
                    lines = self.frame_lines(rows)
                    sys.stdout.write("\033[H" + "\033[K\n".join(lines) + "\033[K\033[J")
                    sys.stdout.flush()
                    last_size, dirty = size, False
                key = self.key_reader.read_key(timeout=0.3)
                if key == "TIMEOUT":
                    continue
                dirty = True

                if key in ("UP", "k", "K"):
                    self.selected_index = (self.selected_index - 1) % len(MENU_ITEMS)
                elif key in ("DOWN", "j", "J"):
                    self.selected_index = (self.selected_index + 1) % len(MENU_ITEMS)
                elif key in ("ENTER", "SPACE"):
                    return self.selected_index + 1
                elif key in ("q", "Q", "ESC"):
                    return len(MENU_ITEMS)  # Exit to Shell
                elif key == "EOF":
                    return len(MENU_ITEMS)
                elif key.isdigit() and 1 <= int(key) <= len(MENU_ITEMS):
                    return int(key)
                elif key == "?":
                    self.selected_index = 0
                    return 1  # Open TUI dashboard for help
        finally:
            try:
                sys.stdout.write("\033[?25h\033[?1049l")  # show cursor, leave the alternate screen
                sys.stdout.write("\033]111\007")           # restore the original terminal background
                sys.stdout.flush()
            except Exception:
                pass


def show_welcome_screen() -> int:
    """Convenience helper to launch the welcome screen."""
    screen = WelcomeScreen()
    return screen.run()
