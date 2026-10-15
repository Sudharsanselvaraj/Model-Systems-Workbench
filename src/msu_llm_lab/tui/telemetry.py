"""Local system resource telemetry and request-metric view-models.

Nothing here is generated: system samples come from the OS (psutil when installed, otherwise standard-library fallbacks that are
labelled as such), and request metrics come only from the recorded event log. A metric that cannot be measured is None and the
panels print UNAVAILABLE. Histories start empty; there is no pre-filled or placeholder data.
"""

import os
import shutil
import time
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional, Tuple

try:  # optional: pip install "msu-llm-lab[monitor]"
    import psutil  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    psutil = None

Sample = Tuple[float, Optional[float]]  # (epoch seconds, value or None when that sample could not be measured)


class SystemTelemetry:
    """Samples genuine local CPU, memory and disk usage at a minimum interval and keeps a bounded timestamped history."""

    def __init__(self, history_len: int = 60, min_interval_s: float = 1.0, clock: Callable[[], float] = time.time) -> None:
        self.history_len = history_len
        self.min_interval_s = min_interval_s
        self._clock = clock
        self.cpu: Deque[Sample] = deque(maxlen=history_len)
        self.mem: Deque[Sample] = deque(maxlen=history_len)
        self.disk: Deque[Sample] = deque(maxlen=history_len)
        self.cores = os.cpu_count() or 1
        self.last_update: Optional[float] = None
        self.cpu_source = "psutil" if psutil else "loadavg"
        self.mem_source = "psutil" if psutil else "UNAVAILABLE (needs psutil)"
        self._mem_detail = ""
        self._disk_detail = ""
        if psutil:
            try:
                psutil.cpu_percent(interval=None)  # the first call has no interval to measure; discard it
            except Exception:
                pass

    def sample(self, force: bool = False) -> bool:
        """Takes one measurement of each metric. Returns False when called again inside the minimum interval."""
        now = self._clock()
        if not force and self.last_update is not None and now - self.last_update < self.min_interval_s:
            return False
        self.last_update = now
        cpu = mem = disk = None
        try:
            if psutil:
                cpu = float(psutil.cpu_percent(interval=None))
            else:
                cpu = min(100.0, os.getloadavg()[0] / self.cores * 100.0)  # 1-minute load average per core, not CPU utilisation
        except Exception:
            cpu = None
        try:
            if psutil:
                vm = psutil.virtual_memory()
                mem = float(vm.percent)
                self._mem_detail = f"{vm.used / 1024 ** 3:.1f}/{vm.total / 1024 ** 3:.1f} GB"
        except Exception:
            mem = None
        try:
            du = shutil.disk_usage(".")
            disk = du.used / du.total * 100.0
            self._disk_detail = f"{du.used / 1024 ** 3:.0f}/{du.total / 1024 ** 3:.0f} GB"
        except Exception:
            disk = None
        self.cpu.append((now, cpu))
        self.mem.append((now, mem))
        self.disk.append((now, disk))
        return True

    @staticmethod
    def _latest(hist: Deque[Sample]) -> Optional[float]:
        return hist[-1][1] if hist else None

    @staticmethod
    def _values(hist: Deque[Sample]) -> List[float]:
        return [v if v is not None else 0.0 for _, v in hist]

    def get_cpu_info(self) -> Tuple[Optional[float], str, List[float]]:
        cur = self._latest(self.cpu)
        unit = "load/core" if self.cpu_source == "loadavg" else "util"
        detail = "UNAVAILABLE" if cur is None else f"{self.cores} cores, {unit} ({self.cpu_source})"
        return cur, detail, self._values(self.cpu)

    def get_mem_info(self) -> Tuple[Optional[float], str, List[float]]:
        cur = self._latest(self.mem)
        return cur, (self._mem_detail if cur is not None else self.mem_source), self._values(self.mem)

    def get_disk_info(self) -> Tuple[Optional[float], str, List[float]]:
        cur = self._latest(self.disk)
        return cur, (self._disk_detail if cur is not None else "UNAVAILABLE"), self._values(self.disk)


class InferenceTelemetry:
    """View-model for request metrics. It is filled only from an Aggregate built out of recorded request events."""

    SERIES = ("latency_e2e", "latency_model", "tokens_per_second", "concurrency", "error_rate", "cost_actual", "cost_estimated", "cost_unresolved")

    def __init__(self, history_len: int = 60) -> None:
        self.history_len = history_len
        self.series: Dict[str, List[Tuple[float, Optional[float]]]] = {k: [] for k in self.SERIES}
        self.units: Dict[str, str] = {}
        self.sample_count = 0      # recorded terminal events behind these graphs
        self.is_live = False       # True once any real request event exists
        self.active = 0
        self.last_event_ts: Optional[float] = None
        self.n_in_window = 0               # finished requests inside the current rolling window
        self.last_terminal_ts: Optional[float] = None
        self.window_s = 300.0

    def update_from(self, agg) -> None:
        """Replace the series with the aggregate's real time buckets. Called only when the event log changed or time advanced."""
        self.series = {k: list(agg.series.get(k, [])) for k in self.SERIES}
        self.units = dict(agg.series_units)
        self.sample_count = agg.n_terminal
        self.is_live = agg.has_live_inference
        self.active = agg.active
        self.last_event_ts = agg.last_event_ts
        self.n_in_window = agg.n_terminal_in_window
        self.last_terminal_ts = agg.last_terminal_ts
        self.window_s = agg.window_s

    def values(self, name: str) -> List[Optional[float]]:
        return [v for _, v in self.series.get(name, [])]


class LogBuffer:
    """Bounded circular buffer of timestamped application and diagnostic logs."""

    def __init__(self, max_entries: int = 100, max_size: Optional[int] = None) -> None:
        capacity = max_size if max_size is not None else max_entries
        self.entries: Deque[Tuple[str, str, str]] = deque(maxlen=capacity)
        # Startup entries are real facts read at launch (never mock-up text); anything unavailable is simply omitted.
        now = datetime.now().strftime("%H:%M:%S")
        self.add_log("INFO", "Workbench initialized (local only; no live requests from this screen)", now)
        try:
            from msu_llm_lab.models.registry import roster_status
            rows = roster_status()
            ready = sum(r["readiness"] == "READY" for r in rows)
            smoke = sum(r["readiness"] == "SMOKE_READY" for r in rows)
            self.add_log("INFO", f"Roster: {len(rows)} participants ({ready} ready, {smoke} smoke-ready, {len(rows) - ready - smoke} blocked)", now)
            self.add_log("WARN", "No provider data policy is verified: student-derived data stays blocked", now)
        except Exception:
            pass
        try:
            from msu_llm_lab.core.config import settings
            from msu_llm_lab.core.budget_policy import effective_policy
            pol = effective_policy(settings.budget, settings.data_dir)
            self.add_log("INFO", f"Budget policy: ${pol.total_budget_usd:.2f} total / ${pol.pilot_budget_usd:.2f} pilot", now)
            from msu_llm_lab.storage.db import DatabaseManager
            import sqlite3
            db = DatabaseManager()
            db.init_db()
            with sqlite3.connect(str(db.db_path)) as c:
                runs, live = c.execute("SELECT COUNT(*), COALESCE(SUM(CASE WHEN is_dry_run = 0 THEN 1 ELSE 0 END), 0) FROM evaluation_runs").fetchone()
            self.add_log("INFO", f"Database: SQLite WAL; {runs} recorded runs ({live} live)", now)
        except Exception:
            pass

    def add_log(self, level: str, message: str, timestamp: str = "") -> None:
        ts = timestamp or datetime.now().strftime("%H:%M:%S")
        self.entries.append((ts, level, message))

    def add(self, level: str, message: str, timestamp: str = "") -> None:
        """Alias for add_log."""
        self.add_log(level, message, timestamp)

    def get_recent(self, count: int = 6) -> List[Tuple[str, str, str]]:
        return list(self.entries)[-count:]

