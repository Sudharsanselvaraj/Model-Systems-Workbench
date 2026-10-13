"""Unit tests for budget ledger calculations, limits, database idempotency, and CLI behavior."""

from decimal import Decimal
from pathlib import Path
import pytest
from typer.testing import CliRunner

from msu_llm_lab.core.config import BudgetPolicy
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.storage.db import DatabaseManager
from msu_llm_lab.models.catalog import get_portfolio, DEFAULT_PORTFOLIO
from msu_llm_lab.cli import app

runner = CliRunner()


@pytest.fixture
def test_db(tmp_path: Path) -> DatabaseManager:
    """Provides an isolated SQLite DatabaseManager in a temporary directory."""
    db_file = tmp_path / "test_workbench.db"
    mgr = DatabaseManager(db_path=db_file)
    mgr.init_db()
    return mgr


@pytest.fixture
def budget_service(test_db: DatabaseManager) -> BudgetService:
    """Provides a BudgetService configured with default policies on test_db."""
    policy = BudgetPolicy(
        total_budget_usd=Decimal("10.00"),
        pilot_budget_usd=Decimal("2.00"),
        per_run_ceiling_usd=Decimal("0.50"),
        per_case_ceiling_usd=Decimal("0.02"),
    )
    return BudgetService(db=test_db, policy=policy)


def test_database_idempotent_initialization(tmp_path: Path):
    """Verify that multiple init_db calls do not fail or wipe data."""
    db_file = tmp_path / "idempotent.db"
    mgr = DatabaseManager(db_path=db_file)
    mgr.init_db()

    # Insert a record
    mgr.record_transaction(
        transaction_id="TX-001",
        model_id="test/model",
        provider="test_provider",
        amount_usd=Decimal("0.0500"),
        description="First transaction",
    )

    # Re-run init_db
    mgr.init_db()

    # Verify data is still intact
    micros, total = mgr.get_cumulative_spend()
    assert micros == 50_000
    assert total == Decimal("0.0500")


def test_safe_decimal_and_micro_unit_storage(test_db: DatabaseManager):
    """Verify that sub-cent fractional amounts avoid floating-point drift."""
    # Record token-scale costs (e.g. $0.0015 and $0.0001)
    test_db.record_transaction(
        transaction_id="TX-SUB-1",
        model_id="test/model",
        provider="test",
        amount_usd=Decimal("0.0015"),
        description="Sub-cent test 1",
    )
    test_db.record_transaction(
        transaction_id="TX-SUB-2",
        model_id="test/model",
        provider="test",
        amount_usd=Decimal("0.0001"),
        description="Sub-cent test 2",
    )

    micros, total_usd = test_db.get_cumulative_spend()
    assert micros == 1600  # 1500 + 100
    assert total_usd == Decimal("0.0016")

    txs = test_db.get_recent_transactions()
    assert len(txs) == 2
    assert txs[0]["amount_micros"] == 100
    assert txs[1]["amount_micros"] == 1500


def test_budget_status_baseline_and_spend_accumulation(budget_service: BudgetService):
    """Verify baseline status and accurate spend/remaining balance accumulation."""
    status = budget_service.get_status()
    assert status.total_ceiling_usd == Decimal("10.00")
    assert status.cumulative_spend_usd == Decimal("0.00")
    assert status.remaining_total_usd == Decimal("10.00")
    assert status.pilot_ceiling_usd == Decimal("2.00")
    assert status.pilot_spend_usd == Decimal("0.00")
    assert status.remaining_pilot_usd == Decimal("2.00")
    assert not status.is_pilot_exhausted
    assert not status.is_total_exhausted

    # Record $0.45 spend in pilot
    budget_service.record_spend(
        transaction_id="TX-P1",
        model_id="anthropic/claude-3.5-haiku",
        provider="openrouter",
        amount_usd=Decimal("0.4500"),
        description="Batch 1",
        is_pilot=True,
    )

    status2 = budget_service.get_status()
    assert status2.cumulative_spend_usd == Decimal("0.4500")
    assert status2.remaining_total_usd == Decimal("9.5500")
    assert status2.pilot_spend_usd == Decimal("0.4500")
    assert status2.remaining_pilot_usd == Decimal("1.5500")

    # Record another $1.55 spend in pilot (reaching exactly $2.00)
    budget_service.record_spend(
        transaction_id="TX-P2",
        model_id="openai/gpt-4o-mini",
        provider="openrouter",
        amount_usd=Decimal("1.5500"),
        description="Batch 2",
        is_pilot=True,
    )

    status3 = budget_service.get_status()
    assert status3.pilot_spend_usd == Decimal("2.0000")
    assert status3.remaining_pilot_usd == Decimal("0.0000")
    assert status3.is_pilot_exhausted is True
    assert status3.remaining_total_usd == Decimal("8.0000")
    assert status3.is_total_exhausted is False


def test_budget_limit_checks_and_circuit_breaker(budget_service: BudgetService):
    """Verify fail-closed limit checks against per-run, pilot, and total ceilings."""
    # 1. Negative spend rejected
    ok, msg = budget_service.check_spend_permitted(Decimal("-0.0100"))
    assert not ok
    assert "cannot be negative" in msg

    # 2. Exceeding per-run ceiling ($0.50) rejected
    ok, msg = budget_service.check_spend_permitted(Decimal("0.5001"))
    assert not ok
    assert "exceeds the per-run ceiling" in msg

    # 3. Valid spend within per-run and pilot ceiling permitted
    ok, msg = budget_service.check_spend_permitted(Decimal("0.4000"))
    assert ok
    assert "permitted" in msg

    # 4. Spend that would breach pilot ceiling rejected
    # First spend $1.80 across multiple small runs
    budget_service.record_spend("T1", "m", "p", Decimal("0.45"), "r1", is_pilot=True)
    budget_service.record_spend("T2", "m", "p", Decimal("0.45"), "r2", is_pilot=True)
    budget_service.record_spend("T3", "m", "p", Decimal("0.45"), "r3", is_pilot=True)
    budget_service.record_spend("T4", "m", "p", Decimal("0.45"), "r4", is_pilot=True)
    # Current pilot spend: $1.80. Remaining pilot: $0.20

    # A new run of $0.25 is <= $0.50 per-run ceiling, but would cause pilot to reach $2.05 > $2.00!
    ok, msg = budget_service.check_spend_permitted(Decimal("0.2500"), is_pilot=True)
    assert not ok
    assert "would exceed the pilot evaluation ceiling" in msg

    # A run of $0.15 is permitted (1.80 + 0.15 = 1.95 <= 2.00)
    ok, msg = budget_service.check_spend_permitted(Decimal("0.1500"), is_pilot=True)
    assert ok


def test_preflight_case_cost_calculation(budget_service: BudgetService):
    """Verify exact Decimal calculation for individual cases."""
    # 250 prompt tokens at $0.80 / 1M, 400 completion tokens at $4.00 / 1M
    cost = budget_service.calculate_case_cost(
        prompt_tokens=250,
        completion_tokens=400,
        input_price_per_1m_usd=Decimal("0.80"),
        output_price_per_1m_usd=Decimal("4.00"),
    )
    # Expected: (250/1M * 0.80) + (400/1M * 4.00) = 0.000200 + 0.001600 = 0.001800
    assert cost == Decimal("0.0018")


def test_model_catalog_seeding_and_unverified_labels(test_db: DatabaseManager):
    """Verify default portfolio seeding and strict unverified labeling."""
    portfolio = get_portfolio(test_db)
    assert len(portfolio) == len(DEFAULT_PORTFOLIO)

    # Every candidate must be labeled UNVERIFIED_CATALOG_ENTRY
    for model in portfolio:
        assert model.verification_status == "UNVERIFIED_CATALOG_ENTRY"
        assert model.deployment_mode in ("HOSTED_OPENROUTER", "HOSTED_DIRECT", "SELF_HOSTED_VLLM")
        assert model.input_price_per_1m_usd > Decimal("0.00")
        assert model.output_price_per_1m_usd > Decimal("0.00")

    # Verify Gemini Flash, Claude Haiku, GPT-6 Luna, Qwen3-8B, Qwen3-14B exist
    ids = [m.model_id for m in portfolio]
    assert "google/gemini-flash-1.5" in ids
    assert "anthropic/claude-haiku-5.5" in ids
    assert "openai/gpt-6-luna" in ids
    assert "qwen/qwen3-8b" in ids
    assert "qwen/qwen3-14b" in ids


def test_cli_budget_status_output():
    """Verify CLI budget status command displays ceilings and disclaimer."""
    result = runner.invoke(app, ["budget", "status"], env={"COLUMNS": "180"})
    assert result.exit_code == 0
    assert "Budget Status" in result.stdout
    assert "Initial Pilot Budget" in result.stdout
    assert "Total Experiment Ceiling" in result.stdout
    assert "local experiment tracking policies" in result.stdout


def test_cli_models_list_output():
    """Verify CLI models list command displays candidates and unverified disclaimer."""
    result = runner.invoke(app, ["models", "list"], env={"COLUMNS": "180"})
    assert result.exit_code == 0
    assert "Candidate Portfolio" in result.stdout
    assert "UNVERIFIED" in result.stdout
    assert "NOT yet been validated against OpenRouter live API" in result.stdout
    assert "google/gemini-flash-1.5" in result.stdout
    assert "qwen/qwen3-8b" in result.stdout
