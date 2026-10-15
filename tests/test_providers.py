"""Unit tests for Phase 3 provider adapters, confidentiality gate, and evaluation runner."""

import os
from decimal import Decimal
from pathlib import Path
import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.config import BudgetPolicy
from msu_llm_lab.evaluation.evaluator import RubricEvaluator
from msu_llm_lab.evaluation.runner import BenchmarkRunner
from msu_llm_lab.providers.base import (
    ConfidentialityViolationError,
    ProviderRequest,
    scrub_secrets,
)
from msu_llm_lab.providers.openrouter import OpenRouterAdapter
from msu_llm_lab.schemas.evaluation import (
    CaseConfidentiality,
    DataClassification,
    DatasetSplit,
    EvaluationCase,
    VerificationRecord,
    VerificationStatus,
)
from msu_llm_lab.storage.db import DatabaseManager

runner = CliRunner()


def test_scrub_secrets():
    """Confidentiality scrubber redacts API keys and sensitive tokens from nested structures."""
    payload = {
        "headers": {
            "Authorization": "Bearer sk-or-v1-abcdef1234567890abcdef1234567890",
            "Content-Type": "application/json",
        },
        "params": ["sk-test-secret-key-12345", "open-public-query"],
        "nested": {
            "token": "sk-or-v1-secret9999",
            "number": 42,
        },
    }

    scrubbed = scrub_secrets(payload)
    assert scrubbed["headers"]["Authorization"] == "[REDACTED_SECRET]"
    assert "sk-or-v1-abcdef" not in str(scrubbed)
    assert "[REDACTED_API_KEY]" in scrubbed["params"][0]
    assert scrubbed["params"][1] == "open-public-query"
    assert scrubbed["nested"]["token"] == "[REDACTED_SECRET]"
    assert scrubbed["nested"]["number"] == 42


def test_confidentiality_gatekeeper_halt():
    """Fail-closed confidentiality gate halts on non-public evaluation cases."""
    public_case = EvaluationCase(
        id="case-pub",
        query="What are the entrance requirements for Bachelor of Medicine?",
        expected_answer="Minimum 3.5 CGPA in STPM/A-Levels.",
        programme_category="Medicine",
        classification=DataClassification.PUBLIC,
        split=DatasetSplit.TEST,
        verification=VerificationRecord(
            status=VerificationStatus.VERIFIED,
            reference_source="Handbook 2026",
        ),
    )
    # Should not raise
    OpenRouterAdapter.validate_case_confidentiality(public_case)

    # Confidential case must raise ConfidentialityViolationError
    restricted_case = EvaluationCase(
        id="case-restricted",
        query="Confidential faculty staffing budget for 2026",
        expected_answer="Restricted internal figure.",
        programme_category="Administration",
        classification=DataClassification.CONFIDENTIAL,
        split=DatasetSplit.TEST,
        verification=VerificationRecord(
            status=VerificationStatus.VERIFIED,
            reference_source="Internal Senate Minutes",
        ),
    )
    with pytest.raises(ConfidentialityViolationError) as exc_info:
        OpenRouterAdapter.validate_case_confidentiality(restricted_case)
    assert "CONFIDENTIAL" in str(exc_info.value)
    assert "case-restricted" in str(exc_info.value)


def test_rubric_evaluator_scoring():
    """RubricEvaluator accurately scores groundedness, factual overlap, and format adherence."""
    evaluator = RubricEvaluator()
    score = evaluator.evaluate(
        score_id="score-test-1",
        execution_id="exec-test-1",
        generated_output="MSU provides 100% PTPTN loan facilities and scholarship waivers.",
        expected_answer="PTPTN loan options and merit scholarships are available at MSU.",
    )
    assert score.accuracy > 0.0
    assert score.groundedness > 0.0
    assert score.overall_score > 0.0
    assert score.format_adherence == 1.0


def _eval_dataset(tmp_path: Path) -> Path:
    cases = [
        EvaluationCase(
            id=f"case-bench-{i}",
            query=f"What is the curriculum for Semester {i} in Software Engineering?",
            expected_answer=f"Semester {i} covers core software architectures and algorithms.",
            programme_category="Information Technology",
            classification=DataClassification.SYNTHETIC,
            split=DatasetSplit.TEST,
            verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="Curriculum Guide 2026"),
        )
        for i in range(3)
    ]
    from msu_llm_lab.dataset.manager import DatasetManager
    path = tmp_path / "test_eval.jsonl"
    DatasetManager.save_jsonl(cases, path)
    return path


def test_benchmark_runner_flow(tmp_path: Path):
    """A real (live-shaped) run executes the suite, books spend on the ledger and persists results to SQLite."""
    from live_fakes import LiveFakeProvider
    from msu_llm_lab.core.live_guard import LiveAuthorization
    model = "google/gemini-3.5-flash-lite"
    db_mgr = DatabaseManager(db_path=tmp_path / "bench_test.db")
    db_mgr.init_db()
    dataset = _eval_dataset(tmp_path)
    runner_engine = BenchmarkRunner(db_mgr=db_mgr, provider=LiveFakeProvider.echoing_expected(dataset))
    report = runner_engine.run(model_id=model, dataset_path=dataset, split=DatasetSplit.TEST,
                               live_authorization=LiveAuthorization(models=(model,)))
    assert report.total_cases == 3 and report.completed_cases == 3 and report.failed_cases == 0 and report.mean_accuracy > 0.0
    saved_run = db_mgr.get_evaluation_run(report.run_id)
    assert saved_run is not None and saved_run["model_id"] == model and saved_run["completed_cases"] == 3 and not saved_run["is_dry_run"]
    assert len(db_mgr.get_evaluation_results_for_run(report.run_id)) == 3
    txs = db_mgr.get_recent_transactions()
    assert len(txs) == 3 and all(t["is_simulated"] == 0 and t["cost_basis"] == "ESTIMATED_FROM_CATALOG" for t in txs)
    assert db_mgr.get_cumulative_spend()[1] == Decimal("0.000300")


def test_benchmark_runner_budget_ceiling_enforced(tmp_path: Path):
    """BenchmarkRunner aborts if projected evaluation run exceeds policy ceiling."""
    db_path = tmp_path / "budget_test.db"
    db_mgr = DatabaseManager(db_path=db_path)
    db_mgr.init_db()

    # Configure a tiny budget policy of $0.0001 per run
    policy = BudgetPolicy(
        total_budget_usd=Decimal("1.00"),
        pilot_budget_usd=Decimal("0.50"),
        per_run_ceiling_usd=Decimal("0.00001"),
        per_case_ceiling_usd=Decimal("0.00001"),
    )
    budget_service = BudgetService(db=db_mgr, policy=policy)

    dataset_path = tmp_path / "test_eval_cases.jsonl"
    cases = [
        EvaluationCase(
            id=f"case-{i}",
            query=f"Query {i}",
            expected_answer=f"Answer {i}",
            programme_category="Engineering",
            classification=DataClassification.SYNTHETIC,
            split=DatasetSplit.TEST,
            verification=VerificationRecord(
                status=VerificationStatus.VERIFIED,
                reference_source="Guide 2026",
            ),
        )
        for i in range(5)
    ]
    from msu_llm_lab.dataset.manager import DatasetManager
    DatasetManager.save_jsonl(cases, dataset_path)

    from live_fakes import LiveFakeProvider
    from msu_llm_lab.core.live_guard import LiveAuthorization
    runner_engine = BenchmarkRunner(db_mgr=db_mgr, budget_service=budget_service, provider=LiveFakeProvider())

    with pytest.raises(BudgetExceededError) as exc_info:
        runner_engine.run(
            model_id="google/gemini-3.5-flash-lite",
            dataset_path=dataset_path,
            split=DatasetSplit.TEST,
            live_authorization=LiveAuthorization(models=("google/gemini-3.5-flash-lite",)),
        )
    assert "exceeds the per-run ceiling" in str(exc_info.value)


def test_cli_benchmark_never_simulates_and_compare_runs_reads_stored_real_runs(tmp_path: Path, monkeypatch):
    """`benchmark` has no dry-run mode: without --live it sends nothing and stores nothing; compare-runs works on recorded runs."""
    import urllib.request
    from live_fakes import seed_runs
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    plain = runner.invoke(app, ["benchmark", "--model", "google/gemini-3.5-flash-lite", "--split", "TEST"])
    assert plain.exit_code == 2 and "Nothing was sent" in plain.output and "REAL REQUESTS" in plain.output
    assert runner.invoke(app, ["benchmark", "--dry-run"]).exit_code == 2  # the simulated flag no longer exists (usage error)
    db = DatabaseManager()
    db.init_db()
    assert db.get_all_evaluation_runs() == [] and db.get_recent_transactions() == []
    seed_runs(db)
    cmp_ = runner.invoke(app, ["compare-runs", "run_test_0", "run_test_1"])
    assert cmp_.exit_code == 0
