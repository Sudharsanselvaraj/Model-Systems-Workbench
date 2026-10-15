"""Run the real launcher in a pseudo-terminal and check that moving through the menu redraws in place (no piled-up frames)."""

import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pty")


def run_launcher(keys, rows=55, cols=207, settle=0.6):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(cols), "LINES": str(rows)}
    env.pop("OPENROUTER_API_KEY", None)
    proc = subprocess.Popen([sys.executable, "-c", "from msu_llm_lab.cli import app; app()"], stdin=slave, stdout=slave, stderr=slave,
                            cwd=ROOT, env=env, close_fds=True)
    os.close(slave)
    out = b""

    def drain(t):
        nonlocal out
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([master], [], [], 0.05)
            if r:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                out += chunk

    drain(settle * 3)
    for k in keys:
        os.write(master, k)
        drain(settle)
    drain(settle)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    os.close(master)
    return out.decode("utf-8", errors="replace"), proc.returncode


DOWN, UP, QUIT = b"\x1b[B", b"\x1b[A", b"q"


def test_moving_through_the_menu_redraws_in_place_on_the_alternate_screen():
    text, code = run_launcher([DOWN, DOWN, UP, DOWN, QUIT])
    assert code == 0
    assert text.count("\x1b[?1049h") == 1 and text.count("\x1b[?1049l") == 1      # one alternate-screen session, closed cleanly
    assert "\x1b[2J" not in text and "\x1b[3J" not in text                         # never a full clear that pushes frames into scrollback
    assert len(re.findall(r"\x1b\[H", text)) >= 5                                   # every frame starts from the home position
    assert text.count("\x1b[?25l") == 1 and text.count("\x1b[?25h") == 1            # cursor hidden once, restored once


def test_each_frame_fits_the_terminal_height_and_shows_live_status_not_hard_coded_claims():
    rows = 40
    text, _ = run_launcher([DOWN, QUIT], rows=rows)
    frames = re.split(r"\x1b\[H", text)[1:]
    assert frames
    for f in frames:
        body = f.split("\x1b[?25h")[0]
        assert body.count("\x1b[K\n") + 1 <= rows
    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07", "", text)
    assert "Welcome to MSU LLM Workbench" in plain and "Tests: 95 passed" not in plain and "API key: not set" in plain


def test_short_terminals_get_a_compact_frame_instead_of_scrolling():
    text, _ = run_launcher([QUIT], rows=20)
    frames = re.split(r"\x1b\[H", text)[1:]
    assert frames and all(f.split("\x1b[?25h")[0].count("\x1b[K\n") + 1 <= 20 for f in frames)
    assert "Select a workbench workflow" in re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


ENTER = b"\r"


def plain(text):
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07", "", text)


def test_a_workflow_returns_to_the_menu_instead_of_exiting_and_never_crashes():
    # 3 = dataset status, Enter returns to the menu, 5 = API preflight, Enter returns, q leaves
    text, code = run_launcher([b"3", ENTER, b"5", ENTER, QUIT], settle=1.2)
    p = plain(text)
    assert code == 0 and "Traceback" not in p and "AttributeError" not in p and "OptionInfo" not in p
    assert text.count("\x1b[?1049h") == 3 and text.count("\x1b[?1049l") == 3          # menu, menu again, menu again: it came back each time
    assert p.count("Press Enter to return to the menu") == 2
    assert "expected labels: ALL NULL" in p or "No prepared classification dataset" in p
    assert "API preflight" in p and "Exiting MSU LLM Workbench" in p


def test_every_menu_workflow_runs_without_a_traceback():
    for key in (b"2", b"4", b"6", b"7", b"8"):
        text, code = run_launcher([key, ENTER, QUIT], settle=1.5)
        p = plain(text)
        assert code == 0 and "Traceback" not in p and "Command failed" not in p, (key, p[-400:])
