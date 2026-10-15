"""Unit tests for Phase 5 end-to-end RAG benchmarking and decoupled model evaluation."""

from decimal import Decimal
from pathlib import Path
import pytest
from typer.testing import CliRunner

from live_fakes import ExpectedContextRetriever, LiveFakeProvider
from msu_llm_lab.cli import app
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.dataset.manager import DatasetManager
from msu_llm_lab.rag.evaluator import RAGSystemEvaluator
from msu_llm_lab.rag.models import RetrievalResult
from msu_llm_lab.rag.retriever import BaseRetriever, KeywordBM25Retriever
from msu_llm_lab.rag.runner import RAGBenchmarkRunner
from msu_llm_lab.schemas.evaluation import (
    DataClassification,
    DatasetSplit,
    EvaluationCase,
    VerificationRecord,
    VerificationStatus,
)
from msu_llm_lab.storage.db import DatabaseManager

runner = CliRunner()


def test_keyword_bm25_retriever():
    """KeywordBM25Retriever indexes corpus passages and computes lexical hit rate and precision."""
    cases = [
        EvaluationCase(
            id="c1",
            query="What is the fee for Bachelor of Accounting?",
            expected_context="Bachelor of Accounting total tuition fee is RM 46,500.",
            expected_retrieval_keywords=["accounting", "fee"],
            classification=DataClassification.SYNTHETIC,
            verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="Source 1"),
        ),
        EvaluationCase(
            id="c2",
            query="How long is the MBBS medical degree?",
            expected_context="The MBBS degree is 5 years full-time comprising pre-clinical and clinical phases.",
            expected_retrieval_keywords=["mbbs", "5 years"],
            classification=DataClassification.SYNTHETIC,
            verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="Source 2"),
        ),
    ]

    bm25 = KeywordBM25Retriever(corpus_cases=cases)
    res = bm25.retrieve(cases[0])
    assert res.hit_rate == 1.0
    assert res.precision >= 0.50
    assert res.latency_ms > 0
    assert len(res.chunks) >= 1
    assert "Accounting" in res.chunks[0]


def test_keyword_bm25_independent_corpus_and_anti_leakage():
    """KeywordBM25Retriever supports independent corpus passages and detects test context leakage."""
    corpus = [
        "MSU Faculty of Business: Bachelor of Accounting total fee is RM 46,500.",
        "MSU International Medical School: MBBS is a 5-year full-time programme.",
    ]
    bm25 = KeywordBM25Retriever(corpus_documents=corpus)
    assert len(bm25.documents) == 2

    # Query with clean case
    case_clean = EvaluationCase(
        id="c-clean",
        query="Tell me about MBBS length",
        expected_context="Non-identical context summary",
        classification=DataClassification.SYNTHETIC,
        verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="Source 2"),
    )
    assert bm25.verify_no_leakage([case_clean]) == []

    # Query with leaked case (exact match of expected_context in corpus)
    case_leaked = EvaluationCase(
        id="c-leaked",
        query="What is the fee?",
        expected_context="MSU Faculty of Business: Bachelor of Accounting total fee is RM 46,500.",
        classification=DataClassification.SYNTHETIC,
        verification=VerificationRecord(status=VerificationStatus.VERIFIED, reference_source="Source 1"),
    )
    assert bm25.verify_no_leakage([case_leaked]) == ["c-leaked"]


def test_rag_evaluator_scoring_and_attribution():
    """RAGSystemEvaluator decouples answer correctness from context groundedness and attributes failures."""
    evaluator = RAGSystemEvaluator()

    # 1. Grounded accurate answer -> SYSTEM_SUCCESS
    corr, grd = evaluator.evaluate_generation(
        generated_output="The Bachelor of Computer Science requires STPM Grade C and Add Math credit.",
        retrieved_chunks=["STPM Grade C with credit in Add Math."],
        expected_answer="STPM Grade C with credit in Add Math.",
    )
    assert corr > 0.80
    assert grd > 0.50
    attr = evaluator.attribute_failure(retrieval_hit_rate=1.0, answer_correctness=corr, groundedness=grd)
    assert attr == "SYSTEM_SUCCESS"

    # 2. Hallucinated fee not in context -> penalized groundedness
    corr2, grd2 = evaluator.evaluate_generation(
        generated_output="Tuition fee is RM 99,000.",
        retrieved_chunks=["Tuition fee is RM 46,500."],
        expected_answer="Tuition fee is RM 46,500.",
    )
    assert grd2 < 0.50

    # 3. Retrieval Failure attribution
    attr_ret = evaluator.attribute_failure(retrieval_hit_rate=0.0, answer_correctness=0.2, groundedness=0.2)
    assert attr_ret == "RETRIEVAL_FAILURE"

    # 4. Generation Failure attribution (retrieval succeeded, but generation was wrong)
    attr_gen = evaluator.attribute_failure(retrieval_hit_rate=1.0, answer_correctness=0.3, groundedness=0.3)
    assert attr_gen == "GENERATION_FAILURE"


def test_rag_benchmark_runner_full_flow(tmp_path: Path):
    """A real (live-shaped) RAG run computes metrics, books spend and writes the run to SQLite. There is no simulated mode."""
    model = "google/gemini-3.5-flash-lite"
    db_mgr = DatabaseManager(db_path=tmp_path / "rag_bench_test.db")
    db_mgr.init_db()
    dataset_path = Path("data/evaluation/verified_rag_benchmark.jsonl")

    with pytest.raises(Exception, match="live authorization"):  # no authorization, no request
        RAGBenchmarkRunner(retriever=ExpectedContextRetriever(), db_mgr=db_mgr, provider=LiveFakeProvider()).run(
            model_id=model, dataset_path=dataset_path, split=DatasetSplit.TEST)

    report = RAGBenchmarkRunner(retriever=ExpectedContextRetriever(), db_mgr=db_mgr, provider=LiveFakeProvider.echoing_expected(dataset_path)).run(
        model_id=model, dataset_path=dataset_path, split=DatasetSplit.TEST, live_authorization=LiveAuthorization(models=(model,)))

    assert report.total_cases == 8
    assert report.mean_retrieval_hit_rate == 1.0
    assert report.mean_retrieval_latency_ms > 0
    assert report.mean_answer_correctness > 0.50
    assert report.mean_e2e_latency_ms > report.mean_retrieval_latency_ms
    assert len(report.case_results) == 8
    assert report.total_cost_usd == Decimal("0.0008")
    saved_run = db_mgr.get_evaluation_run(report.run_id)
    assert saved_run is not None and saved_run["completed_cases"] == 8 and not saved_run["is_dry_run"]
    assert len(db_mgr.get_recent_transactions()) == 8 and db_mgr.get_cumulative_spend()[1] == Decimal("0.000800")
    assert report.verified_cases_count == 3
    assert report.unresolved_cases_count == 5


def test_grounded_retrieval_metrics_hand_calculated(tmp_path: Path):
    """Verifies hand-calculated Recall@k, Precision@k, and MRR formulas on controlled inputs."""
    db_mgr = DatabaseManager(db_path=tmp_path / "test_metrics.db")
    db_mgr.init_db()

    # Hand-crafted test cases
    case_1 = EvaluationCase(
        id="c1",
        query="What is query 1?",
        expected_answer="Answer 1",
        relevant_record_ids=["REC-A"],
        source_status="SOURCE_VERIFIED",
        classification=DataClassification.SYNTHETIC,
    )
    case_2 = EvaluationCase(
        id="c2",
        query="What is query 2?",
        expected_answer="Answer 2",
        relevant_record_ids=["REC-B", "REC-C"],
        source_status="SOURCE_VERIFIED",
        classification=DataClassification.SYNTHETIC,
    )

    # Custom mock retriever with controlled record IDs
    class ControlledRetriever(BaseRetriever):
        def retrieve(self, case, top_k=2):
            if case.id == "c1":
                # Hits REC-A at rank 2
                return RetrievalResult(
                    chunks=["Chunk 1", "Chunk 2"],
                    retriever_id="ctrl",
                    retrieved_record_ids=["REC-X", "REC-A"],
                )
            else:
                # Hits REC-B at rank 1, misses REC-C
                return RetrievalResult(
                    chunks=["Chunk 3", "Chunk 4"],
                    retriever_id="ctrl",
                    retrieved_record_ids=["REC-B", "REC-Z"],
                )

    runner_engine = RAGBenchmarkRunner(db_mgr=db_mgr, retriever=ControlledRetriever(), top_k=2, provider=LiveFakeProvider())
    dataset_file = tmp_path / "test_hand_cases.jsonl"
    with open(dataset_file, "w") as f:
        f.write(case_1.model_dump_json() + "\n")
        f.write(case_2.model_dump_json() + "\n")

    report = runner_engine.run(model_id="google/gemini-3.5-flash-lite", dataset_path=dataset_file, live_authorization=LiveAuthorization(models=("google/gemini-3.5-flash-lite",)))

    # Hand calculation:
    # Case 1: rel={"REC-A"}, ret=["REC-X", "REC-A"] (top-2)
    #   hits = 1 -> Recall@2 = 1/1 = 1.0, Precision@2 = 1/2 = 0.5, RR = 1/2 = 0.50
    # Case 2: rel={"REC-B", "REC-C"}, ret=["REC-B", "REC-Z"] (top-2)
    #   hits = 1 -> Recall@2 = 1/2 = 0.5, Precision@2 = 1/2 = 0.5, RR = 1/1 = 1.00
    # Aggregate:
    #   Mean Recall@2 = (1.0 + 0.5) / 2 = 0.7500
    #   Mean Precision@2 = (0.5 + 0.5) / 2 = 0.5000
    #   MRR = (0.5 + 1.0) / 2 = 0.7500
    #   Hit Rate = 2/2 = 1.0000
    assert report.verified_cases_count == 2
    assert report.mean_recall_at_k == 0.7500
    assert report.mean_precision_at_k == 0.5000
    assert report.mean_reciprocal_rank == 0.7500
    assert report.verified_retrieval_hit_rate == 1.0000


def test_unresolved_abstention_exclusion_from_retrieval_denominator(tmp_path: Path):
    """Unresolved cases expecting abstention do not dilute verified retrieval Recall/MRR metrics."""
    db_mgr = DatabaseManager(db_path=tmp_path / "test_abstain.db")
    db_mgr.init_db()

    case_verified = EvaluationCase(
        id="c-ver",
        query="Verified query",
        expected_answer="Known answer",
        relevant_record_ids=["REC-FOUND"],
        source_status="SOURCE_VERIFIED",
        classification=DataClassification.SYNTHETIC,
    )
    case_unresolved = EvaluationCase(
        id="c-unres",
        query="Unknown query not in KB",
        expected_answer="KB does not contain details. Consult admissions.",
        relevant_record_ids=[],
        source_status="UNRESOLVED",
        classification=DataClassification.SYNTHETIC,
        abstention_expected=True,
    )

    class SingleHitRetriever(BaseRetriever):
        def retrieve(self, case, top_k=2):
            return RetrievalResult(
                chunks=["Some text"],
                retriever_id="single",
                retrieved_record_ids=["REC-FOUND"],
            )

    runner_engine = RAGBenchmarkRunner(db_mgr=db_mgr, retriever=SingleHitRetriever(), top_k=2,
                                     provider=LiveFakeProvider({"c-ver": "Known answer", "c-unres": "KB does not contain details. Consult admissions."}))
    dataset_file = tmp_path / "test_abstain_cases.jsonl"
    with open(dataset_file, "w") as f:
        f.write(case_verified.model_dump_json() + "\n")
        f.write(case_unresolved.model_dump_json() + "\n")

    report = runner_engine.run(model_id="google/gemini-3.5-flash-lite", dataset_path=dataset_file, live_authorization=LiveAuthorization(models=("google/gemini-3.5-flash-lite",)))

    # Only 1 verified case, which succeeded -> Recall = 1.0, MRR = 1.0
    assert report.verified_cases_count == 1
    assert report.unresolved_cases_count == 1
    assert report.mean_recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    # Unresolved case scored on abstention
    assert report.abstention_accuracy == 1.0


def test_cli_benchmark_rag_never_simulates_and_sends_nothing_without_live(monkeypatch, tmp_path):
    """Without --live the command only explains itself; without audited KB records it refuses (no evaluation-passage fallback)."""
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    dataset = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "verified_rag_benchmark.jsonl"
    monkeypatch.chdir(tmp_path)  # no data/audit here
    missing = runner.invoke(app, ["benchmark-rag", "--dataset", str(dataset), "--live"])
    assert missing.exit_code == 1 and "Audited knowledge-base records not found" in missing.output
    for flag in ("--dry-run", "--retriever"):  # the simulated options no longer exist
        assert runner.invoke(app, ["benchmark-rag", flag]).exit_code == 2
    (tmp_path / "data" / "audit").mkdir(parents=True)
    (tmp_path / "data" / "audit" / "parsed_records_general_kb.jsonl").write_text("")
    plain = runner.invoke(app, ["benchmark-rag", "--dataset", str(dataset)])
    assert plain.exit_code in (1, 2)  # 2 = explained and sent nothing; 1 = the (empty) index could not be built
