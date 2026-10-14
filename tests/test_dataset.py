"""Unit tests for Phase 2 dataset management, partitioning, and leakage detection."""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.dataset.leakage import LeakageDetector, LeakageType
from msu_llm_lab.dataset.manager import DatasetManager
from msu_llm_lab.schemas.evaluation import (
    DataClassification,
    DatasetSplit,
    EvaluationCase,
    VerificationRecord,
    VerificationStatus,
)


runner = CliRunner()


def test_dataset_jsonl_roundtrip(tmp_path: Path):
    """DatasetManager loads and saves JSONL cases with high fidelity."""
    cases = [
        EvaluationCase(
            id=f"case-{i}",
            query=f"Query {i} for programme {i}",
            expected_answer=f"Answer {i}",
            programme_category="Engineering",
            classification=DataClassification.SYNTHETIC,
            split=DatasetSplit.TRAIN,
            verification=VerificationRecord(
                status=VerificationStatus.VERIFIED,
                reference_source="Handbook 2026",
            ),
        )
        for i in range(5)
    ]
    file_path = tmp_path / "test_dataset.jsonl"
    DatasetManager.save_jsonl(cases, file_path)
    assert file_path.exists()

    loaded = DatasetManager.load_jsonl(file_path)
    assert len(loaded) == 5
    assert loaded[0].id == "case-0"
    assert loaded[0].verification.status == VerificationStatus.VERIFIED


def test_dataset_partitioning():
    """Partitioning divides cases according to requested ratios without dropping records."""
    cases = [
        EvaluationCase(
            id=f"case-{i:03d}",
            query=f"Student query {i}",
            classification=DataClassification.SYNTHETIC,
        )
        for i in range(20)
    ]
    splits = DatasetManager.partition(cases, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, seed=123)

    assert len(splits[DatasetSplit.TRAIN]) == 14
    assert len(splits[DatasetSplit.DEV]) == 3
    assert len(splits[DatasetSplit.TEST]) == 3

    # All items are assigned
    total_assigned = (
        len(splits[DatasetSplit.TRAIN])
        + len(splits[DatasetSplit.DEV])
        + len(splits[DatasetSplit.TEST])
    )
    assert total_assigned == 20


def test_leakage_detector_exact_and_normalized_overlap():
    """Leakage detector catches verbatim and normalized text matches between TRAIN and TEST."""
    cases = [
        EvaluationCase(
            id="train-01",
            query="What are the minimum entry requirements for Master of Pharmacy?",
            split=DatasetSplit.TRAIN,
        ),
        EvaluationCase(
            id="test-01",
            query="What are the minimum entry requirements for Master of Pharmacy?",  # Verbatim duplicate
            split=DatasetSplit.TEST,
        ),
        EvaluationCase(
            id="train-02",
            query="How much does the Bachelor in Accounting cost per semester?",
            split=DatasetSplit.TRAIN,
        ),
        EvaluationCase(
            id="dev-01",
            query="how much does the bachelor in accounting cost per semester???",  # Normalized match
            split=DatasetSplit.DEV,
        ),
    ]
    detector = LeakageDetector()
    report = detector.analyze(cases)

    assert report.has_leakage is True
    assert len(report.findings) == 2
    types = [f.leakage_type for f in report.findings]
    assert LeakageType.EXACT_QUERY_MATCH in types
    assert LeakageType.NORMALIZED_QUERY_MATCH in types


def test_leakage_detector_high_ngram_overlap():
    """Leakage detector catches high n-gram overlap between train and test queries."""
    cases = [
        EvaluationCase(
            id="train-01",
            query="What are the admission requirements and English language proficiency for Bachelor of Computer Science at MSU Malaysia?",
            split=DatasetSplit.TRAIN,
        ),
        EvaluationCase(
            id="test-01",
            query="What are the admission requirements and English language proficiency for Bachelor of Information Technology at MSU Malaysia?",
            split=DatasetSplit.TEST,
        ),
    ]
    # Similarity threshold 0.50 (Jaccard for these queries is ~0.58)
    detector = LeakageDetector(n_gram_size=3, similarity_threshold=0.50)
    report = detector.analyze(cases)

    assert report.has_leakage is True
    assert report.near_duplicate_leak_count == 1
    assert report.findings[0].leakage_type == LeakageType.HIGH_NGRAM_OVERLAP


def test_cli_validate_dataset_command():
    """CLI validate-dataset command successfully validates synthetic seed file."""
    result = runner.invoke(app, ["validate-dataset", "data/evaluation/synthetic_seed_eval.jsonl"])
    assert result.exit_code == 0
    assert "DATASET VALIDATION PASSED" in result.stdout
    assert "Zero Leakage Detected" in result.stdout
