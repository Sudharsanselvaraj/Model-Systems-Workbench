"""Keyboard navigation and non-blocking input reader for terminal interface."""

import os
import select
import sys
from typing import Optional


class KeyReader:
    """Reads single keystrokes including ANSI arrow sequences and alphanumeric commands."""

    def __init__(self) -> None:
        self.is_tty = sys.stdin.isatty()

    def read_key(self, timeout: Optional[float] = None) -> str:
        """Reads a single key or key sequence.
        
        Returns:
            Key identifier like 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ENTER', 'ESC',
            'BACKSPACE', 'TAB', 'SHIFT_TAB', single characters, or 'EOF'.
        """
        if not self.is_tty:
            # Fallback for non-interactive environments, piped input, and automated tests
            line = sys.stdin.readline()
            if not line:
                return "EOF"
            raw = line.rstrip("\r\n")
            if not raw:
                return "ENTER"
            upper = raw.upper()
            if upper in ("UP", "DOWN", "LEFT", "RIGHT", "ENTER", "ESC", "BACKSPACE", "TAB", "SHIFT_TAB", "EOF"):
                return upper
            return raw

        # Interactive POSIX terminal input
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Check timeout if specified
            if timeout is not None:
                r, _, _ = select.select([fd], [], [], timeout)
                if not r:
                    return "TIMEOUT"

            first_bytes = os.read(fd, 1)
            if not first_bytes:
                return "EOF"

            # Handle Escape sequences
            if first_bytes == b"\x1b":
                seq_bytes = b""
                while True:
                    r, _, _ = select.select([fd], [], [], 0.04)
                    if r:
                        chunk = os.read(fd, 32)
                        if not chunk:
                            break
                        seq_bytes += chunk
                        if seq_bytes.endswith((b"A", b"B", b"C", b"D", b"Z", b"H", b"F", b"~")):
                            break
                    else:
                        break

                if not seq_bytes:
                    return "ESC"

                seq = seq_bytes.decode("utf-8", errors="replace")

                # Arrow keys (support both standard ANSI CSI \x1b[ and VT100/SS3 \x1bO)
                if seq in ("[A", "OA", "[1;2A", "[1;5A"):
                    return "UP"
                elif seq in ("[B", "OB", "[1;2B", "[1;5B"):
                    return "DOWN"
                elif seq in ("[C", "OC", "[1;2C", "[1;5C"):
                    return "RIGHT"
                elif seq in ("[D", "OD", "[1;2D", "[1;5D"):
                    return "LEFT"
                elif seq in ("[Z",):
                    return "SHIFT_TAB"
                elif seq in ("[H", "OH", "[1~"):
                    return "HOME"
                elif seq in ("[F", "OF", "[4~"):
                    return "END"
                elif seq in ("[3~",):
                    return "DELETE"
                elif seq in ("[5~",):
                    return "PAGE_UP"
                elif seq in ("[6~",):
                    return "PAGE_DOWN"
                return "ESC"

            elif first_bytes in (b"\r", b"\n"):
                return "ENTER"
            elif first_bytes in (b"\x7f", b"\x08"):
                return "BACKSPACE"
            elif first_bytes == b"\t":
                return "TAB"
            elif first_bytes == b"\x03":  # Ctrl+C
                raise KeyboardInterrupt
            elif first_bytes == b"\x04":  # Ctrl+D
                return "EOF"

            return first_bytes.decode("utf-8", errors="replace")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
