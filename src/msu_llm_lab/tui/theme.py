"""Visual design system, color palette, and styling constants for charcoal monochrome TUI."""

from typing import List, Optional, Tuple
from rich.style import Style

# Charcoal Monochrome Palette (matching systems engineering reference)
BG_DARK = "#1C1C1C"              # Near-black charcoal background
PANEL_BG = "#1C1C1C"             # Panel background
PANEL_BG_SECONDARY = "#222222"   # Secondary container background
BORDER_CHARCOAL = "#383838"      # Subtle charcoal border
BORDER_FOCUSED = "#666666"       # Active focused border (subtle lighter gray)
BORDER_TEAL = BORDER_CHARCOAL    # Backwards compatibility alias
CYAN_HIGHLIGHT = "#D0D0D0"       # Backwards compatibility alias (mapped to soft light gray)

TEXT_PRIMARY = "#D0D0D0"         # Soft light gray primary text
TEXT_SECONDARY = "#888888"       # Muted gray secondary text
TEXT_MUTED = "#555555"           # Dark muted gray / inactive elements
TEXT_BRIGHT = "#FFFFFF"          # Crisp white highlights

# Restrained Monochrome Status Colors (Zero vibrant neon)
COLOR_SUCCESS = "#A0A0A0"        # Subdued gray for active/pass status
COLOR_WARNING = "#C0A060"        # Restrained muted amber warning
COLOR_ERROR = "#A85050"          # Subdued dark red error
COLOR_INFO = "#888888"           # Informational text

# Rich Style Objects
STYLE_TITLE = Style(color=TEXT_PRIMARY, bold=True)
STYLE_MUTED = Style(color=TEXT_MUTED)
STYLE_SECONDARY = Style(color=TEXT_SECONDARY)
STYLE_SUCCESS = Style(color=TEXT_PRIMARY, bold=True)
STYLE_WARNING = Style(color=COLOR_WARNING, bold=True)
STYLE_ERROR = Style(color=COLOR_ERROR, bold=True)
STYLE_CYAN = Style(color=TEXT_PRIMARY, bold=True)

# Inverted Monochrome Selection Highlights
STYLE_ROW_SELECTED = Style(color="#1C1C1C", bgcolor="#D0D0D0", bold=True)
STYLE_ROW_HOVER = Style(color=TEXT_PRIMARY, bgcolor="#2A2A2A")

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

