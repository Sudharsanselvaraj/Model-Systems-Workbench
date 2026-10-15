"""Registry honesty tests: nothing is VERIFIED without a source, and live blockers are visible."""

import pytest
from pydantic import ValidationError

from msu_llm_lab.models.registry import (
    CHECKED_ON, REGISTRY, Fact, FactStatus, Protocol, get_record, live_blockers,
)
from msu_llm_lab.models.catalog import DEFAULT_PORTFOLIO


def all_facts(record):
    return {n: getattr(record, n) for n in type(record).model_fields if isinstance(getattr(record, n), Fact)}


def test_full_roster_of_ten_is_retained_with_distinct_protocols():
    assert len(REGISTRY) == 10
    assert set(REGISTRY) == {"baseline-intent-gemini-3.5-flash-lite", "baseline-lead-gpt-4.1-mini", "jev-1.14", "gpt-4.1-nano", "gpt-4o-mini",
                             "qwen3.5-35b-a3b", "deepseek-chat-v3.1", "mistral-small-3.2-24b-instruct", "gpt-4.1", "claude-sonnet-5.5"}
    assert get_record("jev-1.14").protocol == Protocol.DECISIONS_API
    assert get_record("gpt-4.1-nano").protocol == Protocol.CHAT_COMPLETIONS
    assert get_record("jev-1.14").requested_model_id.value == "typesafe/jev-1.14"
    assert "decisions" in get_record("jev-1.14").endpoint.value and "chat/completions" in get_record("gpt-4.1-nano").endpoint.value


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_known_fact_has_source_and_date_and_unknown_has_no_value(key):
    for name, f in all_facts(get_record(key)).items():
        if f.status == FactStatus.UNKNOWN:
            assert f.value is None, name
        else:
            assert f.checked_on == CHECKED_ON, name
            assert all(u.startswith("https://") for u in f.source_urls), name
            if f.status != FactStatus.USER_SUPPLIED:
                assert f.source_urls, name


def test_fact_validation_rejects_unsourced_or_valued_unknowns():
    with pytest.raises(ValidationError):
        Fact(value="x", status=FactStatus.VERIFIED_PRIMARY)
    with pytest.raises(ValidationError):
        Fact(value="x", status=FactStatus.UNKNOWN)


def test_unverifiable_details_are_not_marked_verified():
    jev, nano = get_record("jev-1.14"), get_record("gpt-4.1-nano")
    for key in REGISTRY:  # no model's data policy content has been verified
        assert get_record(key).data_policy.status == FactStatus.UNKNOWN
    assert jev.error_codes.status == FactStatus.UNKNOWN and nano.resolved_version_reporting.status == FactStatus.VERIFIED_PRIMARY
    assert (jev.input_price_per_1m_usd.value, jev.output_price_per_1m_usd.value) == ("0.042", "0")
    for f in (jev.input_price_per_1m_usd, jev.output_price_per_1m_usd, jev.context_limit_tokens, jev.provider_route,
              jev.route.route_input_price_per_1m_usd, jev.route.route_output_price_per_1m_usd):
        assert f.status == FactStatus.VERIFIED_PRIMARY
    assert jev.response_shape.status == FactStatus.SECONDARY_ONLY  # envelope seen for 1.13 only
    # GPT-4.1 Nano: provider-level OpenAI endpoint is listed in OpenRouter's endpoints metadata
    assert (nano.input_price_per_1m_usd.value, nano.output_price_per_1m_usd.value) == ("0.10", "0.40")
    assert nano.route.route_input_price_per_1m_usd.status == FactStatus.VERIFIED_PRIMARY and nano.pinned_tag == "openai"
    assert nano.route.serving_provider_field.status == FactStatus.UNKNOWN  # chat responses are not documented to carry `provider`


def test_live_runs_are_blocked_until_blockers_are_resolved():
    jev, nano = get_record("jev-1.14"), get_record("gpt-4.1-nano")
    assert jev.synthetic_blockers("pilot") == ["response_shape", "usage_fields"]   # strict stage blocks on an unconfirmed schema
    assert jev.synthetic_blockers("smoke") == []                                  # a smoke test exists to confirm it
    assert nano.synthetic_blockers("smoke") == [] and nano.synthetic_blockers("pilot") == []
    for key in REGISTRY:  # unknown data policy: student-derived data stays blocked everywhere
        assert "data_policy" in get_record(key).student_data_blockers()


def test_unknown_or_conflicting_facts_block_even_the_smoke_stage():
    from registry_helpers import blocked_copy
    assert blocked_copy("gpt-4o-mini").synthetic_blockers("smoke") == ["route"]
    rec = get_record("gpt-4o-mini")
    degraded = rec.model_copy(update={"endpoint": Fact(), "request_fields": Fact()})
    assert {"endpoint", "request_fields"} <= set(degraded.synthetic_blockers("smoke"))
    assert get_record("jev-1.14").response_shape.status == FactStatus.SECONDARY_ONLY


def test_roster_keeps_every_participant_with_role_tasks_and_precise_blockers():
    from msu_llm_lab.models.registry import roster_status
    rows = {r["key"]: r for r in roster_status()}
    assert len(rows) == 10 and all(r["evaluation"] == "NOT_YET_EVALUATED" for r in rows.values())
    assert rows["jev-1.14"]["readiness"] == "SMOKE_READY" and rows["jev-1.14"]["blockers"] == ["response_shape", "usage_fields"]
    for k in rows:
        if k != "jev-1.14":
            assert rows[k]["readiness"] == "READY" and rows[k]["blockers"] == [], k
    g = rows["baseline-intent-gemini-3.5-flash-lite"]
    assert g["roles"] == ["PRODUCTION_BASELINE"] and g["tasks"] == ["INTENT"] and g["model_id"] == "google/gemini-3.5-flash-lite"
    assert rows["baseline-lead-gpt-4.1-mini"]["tasks"] == ["LEAD"]
    assert rows["gpt-4.1"]["roles"] == ["FRONTIER_REFERENCE"] and rows["claude-sonnet-5.5"]["roles"] == ["FRONTIER_REFERENCE"]
    for k in ("gpt-4o-mini", "qwen3.5-35b-a3b", "deepseek-chat-v3.1", "mistral-small-3.2-24b-instruct", "gpt-4.1-nano"):
        assert rows[k]["roles"] == ["CANDIDATE"] and rows[k]["tasks"] == ["INTENT", "LEAD"]
    assert all(r["data_policy"] == "UNKNOWN" and "data_policy" in r["student_data_blockers"] for r in rows.values())


def test_production_reported_names_stay_separate_from_unverified_benchmark_routes():
    for key in ("baseline-intent-gemini-3.5-flash-lite", "baseline-lead-gpt-4.1-mini"):
        rec = get_record(key)
        assert rec.production_reported_name and rec.production_parity.status == FactStatus.UNKNOWN
        assert any("Dify" in d or "benchmark-route" in d for d in rec.open_decisions)
    assert get_record("gpt-4.1-nano").production_reported_name is None


def test_every_chat_participant_has_candidate_routes_and_a_pinned_no_fallback_route():
    for key, rec in REGISTRY.items():
        if rec.protocol == Protocol.CHAT_COMPLETIONS:
            assert rec.candidate_routes and rec.pinned_route() is not None, key
            assert rec.route.provider_preferences == {"order": [rec.pinned_tag], "allow_fallbacks": False}, key
            assert rec.route.status == RouteStatus.READY and rec.route.route_input_price_per_1m_usd.value == rec.pinned_route().input_per_m
    assert get_record("claude-sonnet-5.5").pinned_route().supports("temperature") is False
    assert get_record("gpt-4.1-nano").pinned_route().supports("temperature") is True


def test_noul_threshold_is_recorded_as_an_open_decision_not_defaulted():
    jev = get_record("jev-1.14")
    assert any("threshold" in d.lower() for d in jev.open_decisions)
    assert "threshold" not in str(jev.response_shape.value).lower()


def test_registry_does_not_change_the_pricing_catalog():
    ids = {c.model_id for c in DEFAULT_PORTFOLIO}
    assert "openai/gpt-4.1-nano" not in ids and "typesafe/jev-1.14" not in ids


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        get_record("nope")


# ---- Phase 4b: routes, estimates, student-data policy ----
from decimal import Decimal  # noqa: E402

from msu_llm_lab.classification.privacy import assert_student_data_allowed  # noqa: E402
from msu_llm_lab.models.registry import (  # noqa: E402
    RouteStatus, estimate_permitted, model_level_prices, synthetic_live_blockers,
)
from msu_llm_lab.providers.base import ConfidentialityViolationError  # noqa: E402


def test_gpt_route_is_pinned_without_fallback_and_ready_for_synthetic_use():
    route = get_record("gpt-4.1-nano").route
    assert route.status == RouteStatus.READY
    assert route.provider_preferences == {"order": ["openai"], "allow_fallbacks": False}
    assert route.assumptions and not route.blocked_reasons


def test_catalog_style_prices_available_for_budgeting_only_from_sourced_facts():
    assert model_level_prices("openai/gpt-4.1-nano") == (Decimal("0.10"), Decimal("0.40"))
    assert model_level_prices("typesafe/jev-1.14") == (Decimal("0.042"), Decimal("0"))
    assert model_level_prices("google/gemini-3.5-flash-lite") == (Decimal("0.30"), Decimal("2.50"))
    assert model_level_prices("not/in-registry") is None


def test_estimates_need_verified_route_prices(monkeypatch):
    from registry_helpers import block_route
    assert estimate_permitted("openai/gpt-4.1-nano") is True and estimate_permitted("typesafe/jev-1.14") is True
    block_route(monkeypatch, "gpt-4.1-nano")
    assert estimate_permitted("openai/gpt-4.1-nano") is False   # a blocked/unverified route never yields an estimate
    assert estimate_permitted("not/in-registry") is True        # legacy entries unchanged


def test_unknown_policy_means_no_student_data():
    for key in REGISTRY:
        with pytest.raises(ConfidentialityViolationError):
            assert_student_data_allowed(key)
    assert synthetic_live_blockers("not/in-registry") == []


