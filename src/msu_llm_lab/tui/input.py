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
            Key identifier like 'UP', 'DOWN', 'ENTER', 'ESC', 'q', '1', or EOF.
        """
        if not self.is_tty:
            # Fallback for non-interactive environments, piped input, and automated tests
            line = sys.stdin.readline()
            if not line:
                return "EOF"
            token = line.strip().upper()
            return token if token else "ENTER"

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

            # Handle Escape sequences (arrow keys)
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
