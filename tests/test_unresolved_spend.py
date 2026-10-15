"""A request whose cost cannot be established must stay visible on the ledger and block further live runs until reconciled."""

from decimal import Decimal

import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.storage.db import DatabaseManager

MODEL = "google/gemini-3.5-flash-lite"


@pytest.fixture
def svc(tmp_path):
    return BudgetService(db=DatabaseManager(db_path=tmp_path / "u.db"))


def test_unresolved_entry_has_no_invented_amount_and_blocks_preflight(svc):
    svc.record_unresolved_spend("tx-1", MODEL, "openrouter", "response lacked cost", run_id="r")
    st = svc.get_status()
    assert st.unresolved_spend_count == 1 and st.cumulative_spend_usd == 0 and st.pilot_spend_usd == 0
    row = svc.db.get_recent_transactions()[0]
    assert row["is_unresolved"] == 1 and row["amount_micros"] == 0 and row["cost_basis"] == "UNRESOLVED" and row["description"].startswith("UNRESOLVED SPEND")
    ok, reason = svc.check_spend_permitted(Decimal("0.0001"))
    assert ok is False and "unresolved" in reason.lower() and "reconcile" in reason.lower()
    with pytest.raises(BudgetExceededError, match="unresolved"):
        svc.check_preflight_budget(estimated_cost_usd=Decimal("0.0001"))


def test_reconciling_with_a_real_amount_unblocks_and_books_it_as_a_distinct_basis(svc):
    svc.record_unresolved_spend("tx-1", MODEL, "openrouter", "response lacked cost")
    svc.db.reconcile_transaction("tx-1", Decimal("0.000321"), "dashboard 2026-10-15")
    assert svc.unresolved() == []
    st = svc.get_status()
    assert st.unresolved_spend_count == 0 and st.cumulative_spend_usd == Decimal("0.000321")
    assert st.spend_by_basis == {"RECONCILED_MANUAL": Decimal("0.000321")}
    assert svc.check_spend_permitted(Decimal("0.0001"))[0] is True
    with pytest.raises(ValueError):
        svc.db.reconcile_transaction("tx-1", Decimal("1"))   # already resolved
    with pytest.raises(KeyError):
        svc.db.reconcile_transaction("nope", Decimal("1"))


def test_actual_estimated_simulated_and_unresolved_spend_stay_distinct(svc):
    svc.record_spend("a", MODEL, "openrouter", Decimal("0.001"), "x", cost_basis="PROVIDER_REPORTED")
    svc.record_spend("b", MODEL, "openrouter", Decimal("0.002"), "x", cost_basis="ESTIMATED_FROM_CATALOG")
    svc.record_spend("c", MODEL, "openrouter", Decimal("0.5"), "x", is_simulated=True, cost_basis="SIMULATED")
    svc.record_spend("d", MODEL, "openrouter", Decimal("0.9"), "x", is_preflight_estimate=True, cost_basis="ESTIMATED_FROM_CATALOG")
    svc.record_unresolved_spend("e", MODEL, "openrouter", "unknown")
    st = svc.get_status()
    assert st.spend_by_basis == {"PROVIDER_REPORTED": Decimal("0.001"), "ESTIMATED_FROM_CATALOG": Decimal("0.002")}
    assert st.simulated_spend_usd == Decimal("0.5") and st.unresolved_spend_count == 1 and st.cumulative_spend_usd == Decimal("0.003")


def test_legacy_ledgers_gain_the_new_columns_without_losing_rows(tmp_path):
    import sqlite3
    path = tmp_path / "legacy.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE budget_ledger (transaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, run_id TEXT, model_id TEXT NOT NULL, provider TEXT NOT NULL, "
              "amount_micros INTEGER NOT NULL, amount_usd TEXT NOT NULL, is_pilot INTEGER NOT NULL DEFAULT 1, is_preflight_estimate INTEGER NOT NULL DEFAULT 0, description TEXT NOT NULL)")
    c.execute("INSERT INTO budget_ledger VALUES ('t', 'now', NULL, 'm', 'p', 5000, '0.005000', 1, 0, 'old')")
    c.commit()
    c.close()
    db = DatabaseManager(db_path=path)
    db.init_db()
    with sqlite3.connect(path) as c2:
        cols = [r[1] for r in c2.execute("PRAGMA table_info(budget_ledger)")]
        assert {"is_simulated", "is_unresolved", "cost_basis"} <= set(cols)
        assert c2.execute("SELECT COUNT(*), SUM(amount_micros), MIN(cost_basis) FROM budget_ledger").fetchone() == (1, 5000, "UNSPECIFIED")


def test_budget_cli_shows_unresolved_entries_and_reconcile_clears_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = BudgetService()
    b.record_unresolved_spend("tx-cli", MODEL, "openrouter", "no cost")
    r = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["budget", "status"])
    assert r.exit_code == 0 and "UNRESOLVED SPEND: 1" in r.output and "tx-cli" in r.output
    assert CliRunner().invoke(app, ["budget", "reconcile", "--transaction-id", "tx-cli", "--amount-usd", "abc"]).exit_code == 1
    assert CliRunner().invoke(app, ["budget", "reconcile", "--transaction-id", "missing", "--amount-usd", "0.1"]).exit_code == 1
    ok = CliRunner().invoke(app, ["budget", "reconcile", "--transaction-id", "tx-cli", "--amount-usd", "0.0004", "--note", "dashboard"])
    assert ok.exit_code == 0 and "Unresolved entries remaining: 0" in ok.output
    after = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["budget", "status"])
    assert "have unknown cost" not in after.output and "RECONCILED_MANUAL" in after.output
