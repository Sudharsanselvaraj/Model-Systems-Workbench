"""Event log, tail polling and aggregation: real measurements only, unknowns stay unknown."""

import json
from decimal import Decimal

import pytest

from msu_llm_lab.observability.aggregate import aggregate, percentile
from msu_llm_lab.observability.events import EventStore, EventTail, RequestEvent
from msu_llm_lab.observability.monitor import LiveMonitor
from obs_helpers import epoch, finished, started, ts


def test_store_refuses_simulated_events(tmp_path):
    store = EventStore(tmp_path / "e.jsonl")
    with pytest.raises(ValueError, match="only LIVE"):
        store.append(started("a", 0, provenance="SIMULATED"))
    assert not (tmp_path / "e.jsonl").exists()


def test_simulated_events_found_in_a_file_are_quarantined_not_counted(tmp_path):
    p = tmp_path / "e.jsonl"
    live, sim = finished("a", 1), finished("b", 2, provenance="SIMULATED", latency=1.0)
    p.write_text(live.to_json_line() + "\n" + sim.to_json_line() + "\n")
    agg = aggregate(EventStore(p).read_all(), epoch(10))
    assert agg.n_terminal == 1 and agg.n_quarantined_simulated == 1
    assert agg.overall.latencies_ms == [100.0]            # the simulated 1.0 ms never entered a figure
    assert agg.overall.actual_cost == Decimal("0.001")


def test_tail_never_duplicates_and_waits_for_a_complete_line(tmp_path):
    p = tmp_path / "e.jsonl"
    tail = EventTail(p)
    assert tail.poll() == []                              # missing file: empty, no error
    e1, e2 = started("a", 0), finished("a", 1)
    p.write_text(e1.to_json_line() + "\n" + e2.to_json_line()[:20])   # second line only half written
    assert [e.event_id for e in tail.poll()] == [e1.event_id]
    assert tail.poll() == []                              # nothing new: no duplicate
    with p.open("a") as fh:
        fh.write(e2.to_json_line()[20:] + "\n")
    assert [e.event_id for e in tail.poll()] == [e2.event_id]
    assert tail.poll() == [] and tail.poll() == []


def test_tail_skips_malformed_lines_and_survives_truncation(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("not json\n" + started("a", 0).to_json_line() + "\n")
    tail = EventTail(p)
    assert len(tail.poll()) == 1 and tail.malformed == 1
    p.write_text("")                                      # file replaced by an empty one
    assert tail.poll() == [] and tail.truncations == 1


def test_percentiles_and_unreliable_tail_flag():
    assert percentile([], 50) is None
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([1, 2, 3, 4], 99) == 4
    agg = aggregate([finished(f"r{i}", i, latency=float(i + 1)) for i in range(5)], epoch(10))
    assert agg.overall.p(95) == 5.0 and agg.overall.p_tail_reliable is False   # n=5 < 20: indicative only


def test_unknown_tokens_costs_and_latency_stay_unavailable():
    ev = finished("a", 1, p=None, c=None, latency=None, basis="UNAVAILABLE", actual=None)
    g = aggregate([ev], epoch(5)).overall
    assert g.n_token_unknown == 1 and g.n_token_known == 0
    assert g.tokens_per_second is None and g.p(50) is None
    assert g.cost_per_request("actual") is None and g.n_cost_unresolved == 1


def test_estimated_cost_is_never_added_to_actual():
    evs = [finished("a", 1, actual="0.01", est="0.02"), finished("b", 2, actual=None, est="0.03", basis="ESTIMATED_FROM_CATALOG")]
    g = aggregate(evs, epoch(5)).overall
    assert g.actual_cost == Decimal("0.01") and g.n_actual_cost == 1
    assert g.estimated_cost == Decimal("0.05") and g.n_estimated_cost == 2


def test_model_latency_is_not_invented_from_round_trip_time():
    g = aggregate([finished("a", 1, latency=900.0)], epoch(5)).overall
    assert g.model_latencies_ms == [] and g.p(50) == 900.0
    g2 = aggregate([finished("a", 1, latency=900.0, model_latency_ms=400.0)], epoch(5)).overall
    assert g2.model_latencies_ms == [400.0] and g2.tps_source == "MODEL_REPORTED_LATENCY"


def test_tokens_per_second_uses_only_known_tokens_and_real_intervals():
    evs = [finished("a", 1, c=100, latency=1000.0), finished("b", 2, c=None, p=None, latency=1000.0)]
    g = aggregate(evs, epoch(5)).overall
    assert g.tokens_per_second == pytest.approx(100.0) and g.tps_n == 1 and g.tps_source == "CLIENT_E2E_LATENCY"


def test_active_requests_come_from_unmatched_starts_and_stale_ones_are_flagged():
    evs = [started("a", 0), started("b", 1), finished("a", 2), started("old", -2000)]
    agg = aggregate(evs, epoch(10))
    assert agg.active == 1 and agg.stale == 1             # b is in flight; "old" never ended and is older than 15 min


def test_timeouts_rate_limits_and_retries_are_counted_from_events():
    evs = [finished("a", 1, status="ERROR", timed_out=True, error_type="ProviderError"),
           finished("b", 2, status="ERROR", http_status=429), finished("c", 3, status="ERROR", http_status=503),
           RequestEvent(kind="REQUEST_RETRY", request_id="b", run_id="r1", case_id="C1", model_key="m1", ts=ts(2), retry_cause="HTTP_429")]
    g = aggregate(evs, epoch(10)).overall
    assert (g.n_timeouts, g.n_rate_limited, g.n_provider_failures, g.n_retries, g.n_error) == (1, 1, 1, 1, 3)


def test_no_events_means_idle_series_with_no_invented_points():
    agg = aggregate([], epoch(100), window_s=60, bucket_s=10)
    assert not agg.has_live_inference and agg.active == 0
    assert all(v is None for ser in agg.series.values() for _, v in ser)       # gaps, not zeros or noise


def test_graph_buckets_follow_real_timestamps():
    evs = [finished("a", 41, latency=200.0), finished("b", 42, latency=400.0), finished("c", 55, latency=100.0)]
    agg = aggregate(evs, epoch(60), window_s=60, bucket_s=10)
    vals = dict(agg.series["latency_e2e"])
    assert vals[epoch(0) + 40] == pytest.approx(300.0) and vals[epoch(0) + 50] == pytest.approx(100.0)
    assert sum(v is not None for v in vals.values()) == 2


def test_monitor_picks_up_new_events_without_duplicating_and_close_releases(tmp_path):
    store = EventStore(tmp_path / "e.jsonl")
    clock = [epoch(10)]
    mon = LiveMonitor(events_path=store.path, prediction_dirs=[], dataset_base=tmp_path / "none", clock=lambda: clock[0])
    assert mon.poll() == 0 and mon.aggregate().n_terminal == 0
    store.append(started("a", 1))
    store.append(finished("a", 2))
    assert mon.poll() == 2 and mon.poll() == 0 and mon.poll() == 0
    assert mon.aggregate().n_terminal == 1 and len(mon.events) == 2
    mon.close()
    assert mon.events == [] and mon.poll() == 0
