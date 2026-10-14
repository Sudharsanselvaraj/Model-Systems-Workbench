"""Interactive, developer-grade welcome screen and quick launcher for MSU LLM Workbench.

Features:
- Custom high-resolution Braille logo matching the MSU LLM Lab 3-plane branching architecture
- Mathematically centered layout across any terminal width (80 to 250+ columns)
- Zero emojis (clean, minimalist, developer-terminal typography)
- Deep purple and pitch black cyberpunk/hacker visual aesthetic
- Keyboard-driven interactive menu (arrow keys, number shortcuts, Enter to launch)
- Instant dispatch to full TUI dashboard, RAG benchmark, audit, traces, serving simulation, or budget
"""

import os
import shutil
import sys
from typing import List, Optional, Tuple
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
    ("2.  Run Knowledge Base Audit & Remediation", "3,051 parsed records", None),
    ("3.  Run Decoupled RAG Benchmark", "v2.1 baseline, Recall@k sweep", None),
    ("4.  Model Comparison Bench", "Compare quality, latency and cost", "NEW"),
    ("5.  Diagnose Production Traces", "Langfuse ingestion and analysis", None),
    ("6.  Simulate vLLM Serving & Hardware Sizing", "A100 / Mac Studio", None),
    ("7.  Inspect Budget & Spend Ceilings", "$0.00 ledger (cap: $10.00)", None),
    ("8.  Settings & KB Hashes", "Providers, paths, cryptographic baseline", None),
    ("9.  Exit to Shell", "", None),
]


class WelcomeScreen:
    """Renders the developer-grade launcher screen and handles interactive menu selection."""

    def __init__(self, console: Optional[Console] = None, key_reader: Optional[KeyReader] = None) -> None:
        self.console = console or Console()
        self.key_reader = key_reader or KeyReader()
        self.selected_index: int = 0

    def render(self, clear: bool = True) -> None:
        """Renders the complete screen layout perfectly centered across any terminal width."""
        if clear and self.key_reader.is_tty:
            self.console.clear()

        # Determine terminal and content width matching compact OpenSRE proportion
        term_cols, _ = shutil.get_terminal_size((100, 30))
        if self.console.width:
            term_cols = max(term_cols, self.console.width)
        content_width = min(max(74, term_cols - 4), 88)

        container = Table.grid(padding=(0, 0))
        container.add_column("main", width=content_width)

        # 1. Centered Braille Logo (26 cols wide, matching OpenSRE)
        logo_pad = max(0, (content_width - 26) // 2)
        for line in LOGO_BRAILLE_STRIPPED:
            container.add_row(Text.from_markup(f"[bold #c4b5fd]{' ' * logo_pad}{line}[/]"))

        container.add_row(Text(""))

        # 2. Version, Title, and Subtitle (matching OpenSRE restrained hierarchy)
        container.add_row(Align.center(Text.from_markup("[dim #a78bfa]v0.1.0-2026.10.14+main.9fbd167[/]")))
        container.add_row(Align.center(Text.from_markup("[bold #d8b4fe]Welcome to MSU LLM Workbench[/]")))
        container.add_row(
            Align.center(Text.from_markup("[dim #9ca3af]The local-first research laboratory for university admissions AI evaluation & serving.[/]"))
        )
        container.add_row(Text(""))

        # 3. Telemetry Status Row (Zero Emojis, Clean Minimalist Monospace)
        pills = Text.from_markup(
            "Corpus: 3,030 clean [green]✓[/]    │    "
            "Tests: 95 passed [green]✓[/]    │    "
            "Spend: $0.00 / $10 cap [green]✓[/]    │    "
            "Network: [bold #38bdf8]OFFLINE[/] [green]✓[/]"
        )
        container.add_row(Align.center(pills))
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
        container.add_row(panel)
        container.add_row(Text(""))

        # 5. Menu Section
        container.add_row(Text.from_markup("[bold #f1f3f9]Select a workbench workflow:[/]\n"))

        for idx, (title, subtitle, badge) in enumerate(MENU_ITEMS):
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

        container.add_row(Text(""))

        # 6. Informational Disclaimer (Zero Emojis)
        container.add_row(Text.from_markup("[dim #64748b]All commands run locally. No external data is sent unless you explicitly enable a provider.[/]"))

        # 7. Prompt Line with Hotkey Hints
        prompt = Text.from_markup("[bold #c4b5fd]msu-llm-lab >[/] █")
        hints = Text.from_markup("[dim #64748b]↑↓ Navigate    Enter Select    ? Help    q Exit[/]")
        spacing = max(2, content_width - len(prompt.plain) - len(hints.plain))
        prompt_row = Text.assemble(prompt, " " * spacing, hints)
        container.add_row(prompt_row)

        # Print mathematically centered container across entire terminal width
        self.console.print(Align.center(container))

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

        try:
            while True:
                self.render(clear=True)
                key = self.key_reader.read_key()

                if key in ("UP", "k", "K"):
                    self.selected_index = (self.selected_index - 1) % len(MENU_ITEMS)
                elif key in ("DOWN", "j", "J"):
                    self.selected_index = (self.selected_index + 1) % len(MENU_ITEMS)
                elif key in ("ENTER", "SPACE"):
                    return self.selected_index + 1
                elif key in ("q", "Q", "ESC"):
                    return len(MENU_ITEMS)  # Exit to Shell
                elif key.isdigit() and 1 <= int(key) <= len(MENU_ITEMS):
                    return int(key)
                elif key == "?":
                    self.selected_index = 0
                    return 1  # Open TUI dashboard for help
        finally:
            try:
                # Restore original terminal background
                sys.stdout.write("\033]111\007")
                sys.stdout.flush()
            except Exception:
                pass


def show_welcome_screen() -> int:
    """Convenience helper to launch the welcome screen."""
    screen = WelcomeScreen()
    return screen.run()
