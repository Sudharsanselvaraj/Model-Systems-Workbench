"""Visual design system, color palette, and styling constants for charcoal monochrome TUI."""

from typing import List, Optional, Tuple
from rich.style import Style

# OpenSRE-inspired Slate Grey Palette with Lavender Accents
BG_DARK = "#212633"              # Distinct, comfortable slate-grey background (sampled from OpenSRE)
PANEL_BG = "#262b3a"             # Subtle elevated slate-grey panel background
PANEL_BG_SECONDARY = "#2c3243"   # Elevated container background
BORDER_CHARCOAL = "#3b4256"      # Refined slate border (crisp and visible against grey canvas)
BORDER_FOCUSED = "#c084fc"       # Radiant lavender border for active focused panel
BORDER_TEAL = BORDER_CHARCOAL    # Backwards compatibility alias
CYAN_HIGHLIGHT = "#d8b4fe"       # Soft lilac accent for titles and indicators

TEXT_PRIMARY = "#f8fafc"         # Crisp, clean off-white primary text
TEXT_SECONDARY = "#94a3b8"       # Balanced slate gray secondary text
TEXT_MUTED = "#64748b"           # Deep restrained slate for inactive items and axes
TEXT_BRIGHT = "#ffffff"          # Crisp white highlights

# Clean Status Colors (Zero emojis)
COLOR_SUCCESS = "#4ade80"        # Clean emerald check
COLOR_WARNING = "#fbbf24"        # Warm amber warning
COLOR_ERROR = "#f87171"          # Soft coral red error
COLOR_INFO = "#c084fc"           # Elegant lavender info

# Rich Style Objects
STYLE_TITLE = Style(color="#e9d5ff", bold=True)
STYLE_MUTED = Style(color=TEXT_MUTED)
STYLE_SECONDARY = Style(color=TEXT_SECONDARY)
STYLE_SUCCESS = Style(color=COLOR_SUCCESS, bold=True)
STYLE_WARNING = Style(color=COLOR_WARNING, bold=True)
STYLE_ERROR = Style(color=COLOR_ERROR, bold=True)
STYLE_CYAN = Style(color="#d8b4fe", bold=True)
STYLE_PANEL = Style(color=TEXT_PRIMARY, bgcolor=PANEL_BG)

# Refined Royal Purple Selection Highlights
STYLE_ROW_SELECTED = Style(color="#ffffff", bgcolor="#4c1d95", bold=True)
STYLE_ROW_HOVER = Style(color="#f8fafc", bgcolor="#2e1065")

# Sparkline characters (Unicode Block Elements U+2581 to U+2588)
SPARK_CHARS = (" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█")


def render_sparkline(
    values: list[float],
    min_val: float = 0.0,
    max_val: float = 100.0,
    length: int = 24,
) -> str:
    """Renders a compact horizontal sparkline from a series of numeric samples."""
    if not values:
        return "─" * length

    samples = values[-length:]
    if len(samples) < length:
        samples = [min_val] * (length - len(samples)) + samples

    span = max_val - min_val if max_val > min_val else 1.0
    chars = []
    for v in samples:
        norm = max(0.0, min(1.0, (v - min_val) / span))
        idx = int(norm * (len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def render_2row_histogram(
    values: List[float],
    min_val: float = 0.0,
    max_val: float = 100.0,
    width: int = 18,
) -> Tuple[str, str, str]:
    """Renders a 2-tier terminal histogram matching btop systems engineering reference.
    
    Returns:
        (top_tier_chars, bottom_tier_chars, baseline_str)
        - top_tier_chars: Represents values in range [(min+max)/2 .. max]
        - bottom_tier_chars: Represents values in range [min .. (min+max)/2]
        - baseline_str: Horizontal axis line ('─' * width)
    """
    if not values:
        # Empty state: empty space between axis marks and baseline
        return (" " * width, " " * width, "─" * width)

    # Take latest `width` samples
    samples = list(values[-width:])
    if len(samples) < width:
        # Left-pad with min_val so chart advances from right
        samples = [min_val] * (width - len(samples)) + samples

    midpoint = (min_val + max_val) / 2.0
    half_span = (max_val - min_val) / 2.0 if max_val > min_val else 1.0

    top_row = []
    bot_row = []

    for v in samples:
        if v <= min_val:
            bot_row.append(" ")
            top_row.append(" ")
        elif v <= midpoint:
            ratio = (v - min_val) / half_span
            idx = max(1, min(len(SPARK_CHARS) - 1, int(ratio * (len(SPARK_CHARS) - 1))))
            bot_row.append(SPARK_CHARS[idx])
            top_row.append(" ")
        else:
            # Bottom row is fully saturated
            bot_row.append("█")
            ratio = (v - midpoint) / half_span
            idx = max(0, min(len(SPARK_CHARS) - 1, int(ratio * (len(SPARK_CHARS) - 1))))
            top_row.append(SPARK_CHARS[idx])

    return ("".join(top_row), "".join(bot_row), "─" * width)

