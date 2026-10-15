"""Simulated (dry-run) spend must never count against a ceiling, and legacy ledgers migrate on evidence only."""

import sqlite3
from decimal import Decimal

import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.config import BudgetPolicy
from msu_llm_lab.dataset.manager import DatasetManager
from msu_llm_lab.evaluation.runner import BenchmarkRunner
from msu_llm_lab.rag.runner import RAGBenchmarkRunner
from msu_llm_lab.schemas.evaluation import (
    DataClassification, DatasetSplit, EvaluationCase, VerificationRecord, VerificationStatus,
)
from msu_llm_lab.storage.db import DatabaseManager

MODEL = "google/gemini-3.5-flash-lite"


def svc(tmp_path, **policy):
    db = DatabaseManager(db_path=tmp_path / "l.db")
    return db, BudgetService(db=db, policy=BudgetPolicy(**policy) if policy else None)


def test_fresh_ledger_separates_simulated_from_actual(tmp_path):
    db, b = svc(tmp_path)
    b.record_spend("t1", MODEL, "openrouter", Decimal("0.01"), "simulated", run_id="r", is_simulated=True)
    b.record_spend("t2", MODEL, "openrouter", Decimal("0.002"), "actual", run_id="r2")
    b.record_spend("t3", MODEL, "openrouter", Decimal("0.5"), "estimate", is_preflight_estimate=True)
    st = b.get_status()
    assert st.cumulative_spend_usd == Decimal("0.002") and st.pilot_spend_usd == Decimal("0.002")
    assert st.simulated_spend_usd == Decimal("0.01") and st.remaining_pilot_usd == Decimal("1.998")
    rows = {r["transaction_id"]: r["is_simulated"] for r in db.get_recent_transactions()}
    assert rows == {"t1": 1, "t2": 0, "t3": 0}


def test_simulated_spend_never_blocks_a_run_or_exhausts_the_pilot(tmp_path):
    db, b = svc(tmp_path, pilot_budget_usd=Decimal("0.05"))
    b.record_spend("sim", MODEL, "openrouter", Decimal("0.049"), "simulated", is_simulated=True)
    assert b.get_status().is_pilot_exhausted is False
    b.check_preflight_budget(estimated_cost_usd=Decimal("0.04"))          # fits: simulated spend is ignored
    b.record_spend("real", MODEL, "openrouter", Decimal("0.049"), "actual")
    with pytest.raises(BudgetExceededError):
        b.check_preflight_budget(estimated_cost_usd=Decimal("0.04"))      # actual spend does count


def make_legacy_db(path):
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE budget_ledger (transaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, run_id TEXT, model_id TEXT NOT NULL,
        provider TEXT NOT NULL, amount_micros INTEGER NOT NULL, amount_usd TEXT NOT NULL, is_pilot INTEGER NOT NULL DEFAULT 1,
        is_preflight_estimate INTEGER NOT NULL DEFAULT 0, description TEXT NOT NULL);
    CREATE TABLE evaluation_runs (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, model_id TEXT NOT NULL, dataset_path TEXT NOT NULL,
        total_cases INTEGER NOT NULL, completed_cases INTEGER NOT NULL, failed_cases INTEGER NOT NULL, total_cost_usd TEXT NOT NULL,
        mean_latency_ms REAL NOT NULL, mean_accuracy REAL NOT NULL, mean_groundedness REAL NOT NULL, mean_overall_score REAL NOT NULL,
        is_dry_run INTEGER NOT NULL DEFAULT 1, notes TEXT);
    """)
    for rid, dry in (("run_dry", 1), ("run_live", 0)):
        c.execute("INSERT INTO evaluation_runs VALUES (?, 't', ?, 'd', 1, 1, 0, '0', 0, 0, 0, 0, ?, NULL)", (rid, MODEL, dry))
    rows = [("a", "run_dry", 20_000), ("b", "run_dry", 3_000), ("c", "run_live", 7_000), ("d", None, 4_000), ("e", "run_unknown", 1_000)]
    for tid, rid, micros in rows:
        c.execute("INSERT INTO budget_ledger VALUES (?, 't', ?, ?, 'openrouter', ?, ?, 1, 0, 'legacy')", (tid, rid, MODEL, micros, f"{micros/1e6:.6f}"))
    c.commit()
    c.close()


def test_legacy_ledger_migrates_on_evidence_only_and_is_idempotent(tmp_path):
    path = tmp_path / "legacy.db"
    make_legacy_db(path)
    db = DatabaseManager(db_path=path)
    db.init_db()
    db.init_db()  # idempotent
    with sqlite3.connect(path) as c:
        got = dict(c.execute("SELECT transaction_id, is_simulated FROM budget_ledger").fetchall())
        assert c.execute("SELECT COUNT(*), SUM(amount_micros) FROM budget_ledger").fetchone() == (5, 35_000)  # nothing lost or altered
    # only rows tied to a recorded dry-run are simulated; live-run, unlinked and unknown-run rows stay counted as actual
    assert got == {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0}
    assert db.get_simulated_spend()[1] == Decimal("0.023")
    assert db.get_cumulative_spend()[1] == Decimal("0.012") and db.get_pilot_spend()[1] == Decimal("0.012")


def dataset(tmp_path):
    cases = [EvaluationCase(id=f"syn-{i}", query="synthetic question", expected_answer="synthetic answer",
                            classification=DataClassification.SYNTHETIC, split=DatasetSplit.TEST,
                            verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="synthetic"))
             for i in range(3)]
    p = tmp_path / "ds.jsonl"
    DatasetManager.save_jsonl(cases, p)
    return p


def test_dry_run_benchmarks_record_simulated_spend_only(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "r.db")
    ds = dataset(tmp_path)
    BenchmarkRunner(db_mgr=db).run(model_id=MODEL, dataset_path=ds, split=DatasetSplit.TEST, dry_run=True)
    RAGBenchmarkRunner(db_mgr=db).run(model_id=MODEL, dataset_path=ds, split=DatasetSplit.TEST, dry_run=True)
    with sqlite3.connect(tmp_path / "r.db") as c:
        n, sim = c.execute("SELECT COUNT(*), SUM(is_simulated) FROM budget_ledger").fetchone()
    assert n >= 6 and sim == n
    assert db.get_cumulative_spend()[1] == 0 and db.get_pilot_spend()[1] == 0 and db.get_simulated_spend()[1] > 0


def test_budget_status_cli_labels_simulated_rows_and_shows_the_separate_total(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = BudgetService()
    b.record_spend("s1", MODEL, "openrouter", Decimal("0.001"), "Evaluation case x (dry_run=True)", run_id="r", is_simulated=True)
    b.record_spend("a1", MODEL, "openrouter", Decimal("0.002"), "real-looking actual", run_id="r2")
    r = CliRunner(env={"COLUMNS": "250"}).invoke(app, ["budget", "status"])
    assert r.exit_code == 0 and "SIMULATED" in r.output and "ACTUAL" in r.output
    assert "Simulated dry-run spend (not counted against any ceiling): $0.001000" in r.output
