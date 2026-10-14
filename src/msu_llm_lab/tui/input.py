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
                r, _, _ = select.select([sys.stdin], [], [], timeout)
                if not r:
                    return "TIMEOUT"

            ch = sys.stdin.read(1)

            # Handle Escape sequences
            if ch == "\x1b":
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        ch3 = sys.stdin.read(1)
                        if ch3 == "A":
                            return "UP"
                        elif ch3 == "B":
                            return "DOWN"
                        elif ch3 == "C":
                            return "RIGHT"
                        elif ch3 == "D":
                            return "LEFT"
                        elif ch3 == "Z":
                            return "SHIFT_TAB"
                        elif ch3 == "H":
                            return "HOME"
                        elif ch3 == "F":
                            return "END"
                        elif ch3 in ("1", "2", "3", "4", "5", "6"):
                            # Extended sequences like [3~ (Delete)
                            r_tilde, _, _ = select.select([sys.stdin], [], [], 0.02)
                            if r_tilde:
                                ch4 = sys.stdin.read(1)
                                if ch4 == "~":
                                    if ch3 == "3":
                                        return "DELETE"
                                    elif ch3 == "1":
                                        return "HOME"
                                    elif ch3 == "4":
                                        return "END"
                    return "ESC"
                return "ESC"

            elif ch in ("\r", "\n"):
                return "ENTER"
            elif ch == "\x7f" or ch == "\x08":
                return "BACKSPACE"
            elif ch == "\t":
                return "TAB"
            elif ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            elif ch == "\x04":  # Ctrl+D
                return "EOF"

            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
