"""Operator-authorized pilot-ceiling change and the read-only ledger reconciliation."""

import json
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core import budget_policy as bp
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.core.config import BudgetPolicy, settings
from msu_llm_lab.providers.base import LiveNotAuthorizedError
from msu_llm_lab.storage.db import DatabaseManager

BASE = BudgetPolicy()


def raise_to(data_dir, amount="10.00", phrase=None, interactive=True, operator="Test Operator"):
    new = Decimal(amount)
    return bp.raise_pilot_ceiling(data_dir, BASE, new, operator=operator, ledger_total_spent=Decimal("0.915803"), ledger_pilot_spent=Decimal("0.915803"),
                                  is_interactive=lambda: interactive, prompt=lambda m: phrase if phrase is not None else bp.expected_phrase(new, BASE.total_budget_usd), echo=lambda s: None)


def test_the_defaults_are_unchanged_without_an_override(tmp_path):
    p = bp.effective_policy(BASE, tmp_path)
    assert (p.pilot_budget_usd, p.total_budget_usd, p.per_run_ceiling_usd, p.per_case_ceiling_usd) == (Decimal("2.00"), Decimal("10.00"), Decimal("0.50"), Decimal("0.02"))


def test_the_pilot_ceiling_changes_only_after_the_exact_typed_phrase_and_nothing_else_changes(tmp_path):
    with pytest.raises(LiveNotAuthorizedError, match="did not match"):
        raise_to(tmp_path, phrase="yes")
    with pytest.raises(LiveNotAuthorizedError, match="interactive"):
        raise_to(tmp_path, interactive=False)
    with pytest.raises(LiveNotAuthorizedError, match="operator"):
        raise_to(tmp_path, operator=" ")
    assert not bp.policy_path(tmp_path).exists() and bp.effective_policy(BASE, tmp_path).pilot_budget_usd == Decimal("2.00")
    entry = raise_to(tmp_path)
    p = bp.effective_policy(BASE, tmp_path)
    assert p.pilot_budget_usd == Decimal("10.00") and p.total_budget_usd == Decimal("10.00") and p.per_run_ceiling_usd == Decimal("0.50") and p.per_case_ceiling_usd == Decimal("0.02")
    assert entry["previous_pilot_budget_usd"] == "2.00" and entry["ledger_total_spent_usd"] == "0.915803" and entry["operator"] == "Test Operator"
    assert bp.history(tmp_path)[0]["new_pilot_budget_usd"] == "10.00"
    assert bp.expected_phrase(Decimal("10"), Decimal("10")) == "AUTHORIZE PILOT CEILING 10.00 USD TOTAL CEILING 10.00 USD UNCHANGED"


def test_the_pilot_ceiling_can_never_exceed_the_total_or_be_lowered_here(tmp_path):
    with pytest.raises(bp.PolicyError, match="cannot exceed the total"):
        raise_to(tmp_path, "10.01")
    with pytest.raises(bp.PolicyError, match="only raises"):
        raise_to(tmp_path, "2.00")
    assert not bp.policy_path(tmp_path).exists()


def test_a_damaged_or_inconsistent_override_file_is_ignored_not_trusted(tmp_path):
    bp.policy_path(tmp_path).write_text("{ not json")
    assert bp.effective_policy(BASE, tmp_path).pilot_budget_usd == Decimal("2.00") and bp.history(tmp_path) == []
    bp.policy_path(tmp_path).write_text(json.dumps({"schema": bp.SCHEMA, "history": [{"new_pilot_budget_usd": "99.00", "total_budget_usd": "10.00"}]}))
    assert bp.effective_policy(BASE, tmp_path).pilot_budget_usd == Decimal("2.00")           # above the total: refused
    bp.policy_path(tmp_path).write_text(json.dumps({"schema": bp.SCHEMA, "history": [{"new_pilot_budget_usd": "10.00", "total_budget_usd": "50.00"}]}))
    assert bp.effective_policy(BASE, tmp_path).pilot_budget_usd == Decimal("2.00")           # recorded against a different total: refused


def test_the_budget_service_applies_the_override_and_keeps_the_ledger(tmp_path, monkeypatch):
    db = DatabaseManager(db_path=tmp_path / "t.db")
    svc = BudgetService(db=db, policy=BASE)
    svc.record_spend("t1", "m", "openrouter", Decimal("1.90"), "x", run_id="r", cost_basis="ESTIMATED_FROM_CATALOG")
    blocked, _ = svc.check_spend_permitted(Decimal("0.20"))
    assert not blocked                                                                       # the old $2.00 pilot ceiling stops it
    raise_to(settings.data_dir)
    svc2 = BudgetService(db=db)
    assert svc2.policy.pilot_budget_usd == Decimal("10.00")
    ok, _ = svc2.check_spend_permitted(Decimal("0.20"))
    st = svc2.get_status()
    assert ok and st.cumulative_spend_usd == Decimal("1.900000") and st.remaining_pilot_usd == Decimal("8.100000") and st.remaining_total_usd == Decimal("8.100000")
    over, reason = svc2.check_spend_permitted(Decimal("0.60"))
    assert not over and "per-run ceiling" in reason                                          # per-run ceiling unchanged
    svc2.record_spend("t2", "m", "openrouter", Decimal("8.00"), "x", run_id="r", cost_basis="ESTIMATED_FROM_CATALOG")
    no, why = svc2.check_spend_permitted(Decimal("0.40"))
    assert not no and "total budget ceiling" in why                                          # the lifetime ceiling stays hard


def test_cli_refuses_without_an_interactive_terminal(tmp_path):
    r = CliRunner().invoke(app, ["budget", "raise-pilot-ceiling", "--to", "10.00", "--operator", "T"])
    assert r.exit_code == 1 and "interactive" in r.output and not bp.policy_path(settings.data_dir).exists()
    r2 = CliRunner().invoke(app, ["budget", "policy"])
    assert r2.exit_code == 0 and "pilot $2.00" in r2.output


def test_audit_reconciles_ledger_with_stored_results_and_separates_guard_from_ledger(tmp_path):
    from msu_llm_lab.classification.records import PredictionStore
    from msu_llm_lab.observability import budget_audit as ba
    from obs_helpers import intent_pred
    db = DatabaseManager(db_path=tmp_path / "a.db")
    svc = BudgetService(db=db, policy=BASE)
    reports = tmp_path / "reports"
    store = PredictionStore(reports / "classification" / "bench-0123456789ab__m__INTENT.jsonl")
    for i, (cost, native) in enumerate([("0.001000", "0.001500"), ("0.002000", None)]):
        rec = intent_pred(f"C{i}", "QUERY", run="bench-0123456789ab-m-INTENT", recorded_cost_usd=Decimal(cost), native_cost=Decimal(native) if native else None, cost_basis="ESTIMATED_FROM_CATALOG")
        store.append(rec)
        svc.record_spend(f"tx{i}", "m", "openrouter", Decimal(cost), "x", run_id=rec.run_id, cost_basis="ESTIMATED_FROM_CATALOG")
    store.append(intent_pred("E1", "QUERY", run="bench-0123456789ab-m-INTENT", status="ERROR"))
    a = ba.audit(db.db_path, BASE, reports, tmp_path / "benchmarks")
    g = a["groups"][0]
    assert a["reconciled"] and a["ledger_total_counted_usd"] == "0.003" and g["ledger_equals_records"]
    assert g["records_native_credits"] == "0.0015" and g["guard_from_records_usd"] == "0.0035"        # guard = max(booked, native) per request: a different meaning from the ledger
    assert g["records_error_no_cost"] == 1 and a["remaining_usd"]["total"] == "9.997"
    svc.record_spend("tx-extra", "m", "openrouter", Decimal("0.5"), "x", run_id="bench-0123456789ab-m-INTENT", cost_basis="ESTIMATED_FROM_CATALOG")
    assert not ba.audit(db.db_path, BASE, reports, tmp_path / "benchmarks")["reconciled"]           # a ledger row with no stored result is a discrepancy
