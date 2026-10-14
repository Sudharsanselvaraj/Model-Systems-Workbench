"""Visual design system, color palette, and styling constants for btop-inspired TUI."""

from rich.style import Style

# Color Palette inspired by btop / systems monitoring
BG_DARK = "#080F14"              # Near-black navy
PANEL_BG = "#0C151C"             # Primary panel background
PANEL_BG_SECONDARY = "#101C24"   # Secondary panel background
BORDER_TEAL = "#24545A"          # Muted teal borders
BORDER_FOCUSED = "#55D6D0"       # Active focused cyan border
CYAN_HIGHLIGHT = "#55D6D0"       # Primary restrained cyan
TEXT_PRIMARY = "#DCE7EC"         # Soft white primary text
TEXT_SECONDARY = "#91A3AE"       # Cool gray secondary text
TEXT_MUTED = "#5A6E7A"           # Muted dark gray
COLOR_SUCCESS = "#4EBA6F"        # Muted green
COLOR_WARNING = "#E5A93C"        # Amber warning
COLOR_ERROR = "#D9534F"          # Restrained red

# Rich Style Objects
STYLE_TITLE = Style(color=TEXT_PRIMARY, bold=True)
STYLE_CYAN = Style(color=CYAN_HIGHLIGHT, bold=True)
STYLE_MUTED = Style(color=TEXT_MUTED)
STYLE_SECONDARY = Style(color=TEXT_SECONDARY)
STYLE_SUCCESS = Style(color=COLOR_SUCCESS, bold=True)
STYLE_WARNING = Style(color=COLOR_WARNING, bold=True)
STYLE_ERROR = Style(color=COLOR_ERROR, bold=True)

# Active table row selection highlight
STYLE_ROW_SELECTED = Style(color="#080F14", bgcolor=CYAN_HIGHLIGHT, bold=True)
STYLE_ROW_HOVER = Style(color=TEXT_PRIMARY, bgcolor="#162834")

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

    # Take the latest `length` values
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
