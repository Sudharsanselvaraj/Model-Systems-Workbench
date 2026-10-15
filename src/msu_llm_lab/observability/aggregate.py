"""Aggregates derived from request events only. Nothing here is sampled, generated or smoothed.

Rules: an unknown token count, latency or cost stays unknown (it is excluded from the aggregate that needs it and the
exclusion is counted). Estimated cost is never added to actual cost. SIMULATED events are quarantined and counted.
"""

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from msu_llm_lab.observability.events import RequestEvent, TERMINAL_KINDS, parse_ts

MIN_FOR_TAIL_PERCENTILES = 20  # below this P95/P99 are flagged as indicative only
DEFAULT_STALE_AFTER_S = 900.0


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile; None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class GroupStats:
    key: str
    n_terminal: int = 0
    n_ok: int = 0
    n_invalid: int = 0
    n_error: int = 0
    n_timeouts: int = 0
    n_rate_limited: int = 0
    n_provider_failures: int = 0
    n_retries: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    model_latencies_ms: List[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_token_known: int = 0
    n_token_unknown: int = 0
    tps_tokens: int = 0
    tps_seconds: float = 0.0
    tps_n: int = 0
    tps_source: Optional[str] = None          # MODEL_REPORTED_LATENCY or CLIENT_E2E_LATENCY
    actual_cost: Decimal = Decimal("0")
    estimated_cost: Decimal = Decimal("0")
    n_actual_cost: int = 0
    n_estimated_cost: int = 0
    n_cost_unresolved: int = 0
    native_cost: Decimal = Decimal("0")
    n_native_cost: int = 0

    def p(self, q: float) -> Optional[float]:
        return percentile(self.latencies_ms, q)

    @property
    def p_tail_reliable(self) -> bool:
        return len(self.latencies_ms) >= MIN_FOR_TAIL_PERCENTILES

    @property
    def tokens_per_second(self) -> Optional[float]:
        return (self.tps_tokens / self.tps_seconds) if self.tps_seconds > 0 else None

    @property
    def error_rate(self) -> Optional[float]:
        return (self.n_error / self.n_terminal) if self.n_terminal else None

    @property
    def timeout_rate(self) -> Optional[float]:
        return (self.n_timeouts / self.n_terminal) if self.n_terminal else None

    def cost_per_request(self, kind: str) -> Optional[Decimal]:
        total, n = (self.actual_cost, self.n_actual_cost) if kind == "actual" else (self.estimated_cost, self.n_estimated_cost)
        return (total / n) if n else None

    def cost_per_1000(self, kind: str) -> Optional[Decimal]:
        c = self.cost_per_request(kind)
        return c * 1000 if c is not None else None


@dataclass
class Aggregate:
    now: float
    window_s: float
    bucket_s: float
    n_events: int = 0
    n_quarantined_simulated: int = 0
    n_started: int = 0
    n_terminal: int = 0
    n_retry_events: int = 0
    active: int = 0
    stale: int = 0
    last_event_ts: Optional[float] = None
    last_terminal_ts: Optional[float] = None
    n_terminal_in_window: int = 0
    overall: GroupStats = field(default_factory=lambda: GroupStats("ALL"))
    by_model: Dict[str, GroupStats] = field(default_factory=dict)
    by_provider: Dict[str, GroupStats] = field(default_factory=dict)
    series: Dict[str, List[Tuple[float, Optional[float]]]] = field(default_factory=dict)  # name -> [(bucket_start_epoch, value|None)]
    series_units: Dict[str, str] = field(default_factory=dict)
    run_ids: List[str] = field(default_factory=list)

    @property
    def has_live_inference(self) -> bool:
        return self.n_started > 0 or self.n_terminal > 0


def _apply_terminal(g: GroupStats, ev: RequestEvent) -> None:
    g.n_terminal += 1
    if ev.status == "OK":
        g.n_ok += 1
    elif ev.status == "INVALID_OUTPUT":
        g.n_invalid += 1
    else:
        g.n_error += 1
        if ev.timed_out:
            g.n_timeouts += 1
        if ev.http_status == 429:
            g.n_rate_limited += 1
        if ev.http_status is not None and ev.http_status >= 500:
            g.n_provider_failures += 1
    if ev.latency_ms is not None:
        g.latencies_ms.append(ev.latency_ms)
    if ev.model_latency_ms is not None:
        g.model_latencies_ms.append(ev.model_latency_ms)
    if ev.prompt_tokens is not None and ev.completion_tokens is not None:
        g.n_token_known += 1
        g.prompt_tokens += ev.prompt_tokens
        g.completion_tokens += ev.completion_tokens
        # Throughput needs a known completion count and a real interval; the provider-reported interval is preferred.
        interval_ms, source = (ev.model_latency_ms, "MODEL_REPORTED_LATENCY") if ev.model_latency_ms else (ev.latency_ms, "CLIENT_E2E_LATENCY")
        if interval_ms and interval_ms > 0 and ev.status == "OK":
            g.tps_tokens += ev.completion_tokens
            g.tps_seconds += interval_ms / 1000.0
            g.tps_n += 1
            g.tps_source = source if g.tps_source in (None, source) else "MIXED"
    else:
        g.n_token_unknown += 1
    if ev.actual_cost_usd is not None:
        g.actual_cost += ev.actual_cost_usd
        g.n_actual_cost += 1
    if ev.estimated_cost_usd is not None:
        g.estimated_cost += ev.estimated_cost_usd
        g.n_estimated_cost += 1
    if ev.native_cost is not None:
        g.native_cost += ev.native_cost
        g.n_native_cost += 1
    if ev.cost_basis == "UNAVAILABLE":
        g.n_cost_unresolved += 1


def aggregate(events: Iterable[RequestEvent], now: float, window_s: float = 300.0, bucket_s: float = 10.0,
              stale_after_s: float = DEFAULT_STALE_AFTER_S) -> Aggregate:
    agg = Aggregate(now=now, window_s=window_s, bucket_s=bucket_s)
    starts: Dict[str, float] = {}
    ends: Dict[str, float] = {}
    terminals: List[Tuple[float, RequestEvent]] = []
    runs: Dict[str, None] = {}
    for ev in sorted(events, key=lambda e: parse_ts(e.ts)):
        if ev.provenance != "LIVE":
            agg.n_quarantined_simulated += 1
            continue
        t = parse_ts(ev.ts)
        agg.n_events += 1
        agg.last_event_ts = t
        runs[ev.run_id] = None
        model_g = agg.by_model.setdefault(ev.model_key, GroupStats(ev.model_key))
        if ev.kind == "REQUEST_STARTED":
            agg.n_started += 1
            starts.setdefault(ev.request_id, t)
        elif ev.kind == "REQUEST_RETRY":
            agg.n_retry_events += 1
            for g in (agg.overall, model_g):
                g.n_retries += 1
        elif ev.kind in TERMINAL_KINDS:
            agg.n_terminal += 1
            agg.last_terminal_ts = t
            if now - window_s <= t <= now:
                agg.n_terminal_in_window += 1
            ends[ev.request_id] = t
            prov = agg.by_provider.setdefault(ev.serving_provider or "UNKNOWN", GroupStats(ev.serving_provider or "UNKNOWN"))
            for g in (agg.overall, model_g, prov):
                _apply_terminal(g, ev)
            terminals.append((t, ev))
    agg.run_ids = list(runs)
    for rid, t0 in starts.items():
        if rid not in ends:
            if now - t0 > stale_after_s:
                agg.stale += 1
            else:
                agg.active += 1

    # Rolling window of real time buckets ending at `now`; None means no measurement in that bucket (a gap, never zero).
    n_buckets = max(1, int(window_s // bucket_s))
    first = now - n_buckets * bucket_s
    lat: List[Tuple[float, Optional[float]]] = []
    mlat: List[Tuple[float, Optional[float]]] = []
    tps: List[Tuple[float, Optional[float]]] = []
    conc: List[Tuple[float, Optional[float]]] = []
    err: List[Tuple[float, Optional[float]]] = []
    cost_a: List[Tuple[float, Optional[float]]] = []
    cost_e: List[Tuple[float, Optional[float]]] = []
    cost_u: List[Tuple[float, Optional[float]]] = []
    for b in range(n_buckets):
        lo, hi = first + b * bucket_s, first + (b + 1) * bucket_s
        in_b = [ev for t, ev in terminals if lo <= t < hi]
        lats = [e.latency_ms for e in in_b if e.latency_ms is not None]
        mls = [e.model_latency_ms for e in in_b if e.model_latency_ms is not None]
        lat.append((lo, sum(lats) / len(lats) if lats else None))
        mlat.append((lo, sum(mls) / len(mls) if mls else None))
        tk = [(e.completion_tokens, (e.model_latency_ms or e.latency_ms)) for e in in_b
              if e.status == "OK" and e.completion_tokens is not None and (e.model_latency_ms or e.latency_ms)]
        tps.append((lo, (sum(a for a, _ in tk) / (sum(s for _, s in tk) / 1000.0)) if tk else None))
        # concurrency at the bucket end, from real start/terminal times
        c = sum(1 for rid, t0 in starts.items() if t0 < hi and (rid not in ends or ends[rid] >= hi) and hi - t0 <= stale_after_s)
        conc.append((lo, float(c) if starts else None))
        err.append((lo, (100.0 * sum(1 for e in in_b if e.status == "ERROR") / len(in_b)) if in_b else None))
        cost_a.append((lo, float(sum((e.actual_cost_usd for e in in_b if e.actual_cost_usd is not None), Decimal("0"))) if any(e.actual_cost_usd is not None for e in in_b) else None))
        cost_e.append((lo, float(sum((e.estimated_cost_usd for e in in_b if e.estimated_cost_usd is not None), Decimal("0"))) if any(e.estimated_cost_usd is not None for e in in_b) else None))
        cost_u.append((lo, float(sum(1 for e in in_b if e.cost_basis == "UNAVAILABLE")) if in_b else None))
    agg.series = {"latency_e2e": lat, "latency_model": mlat, "tokens_per_second": tps, "concurrency": conc,
                  "error_rate": err, "cost_actual": cost_a, "cost_estimated": cost_e, "cost_unresolved": cost_u}
    agg.series_units = {"latency_e2e": "ms", "latency_model": "ms", "tokens_per_second": "tok/s", "concurrency": "requests",
                        "error_rate": "%", "cost_actual": "USD/bucket", "cost_estimated": "USD/bucket", "cost_unresolved": "requests/bucket"}
    return agg
