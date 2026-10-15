"""A provider stand-in that returns LIVE-shaped responses (not simulated), so persistence and ledger code can be tested without any network."""

from decimal import Decimal

from msu_llm_lab.providers.base import BaseProvider, ProviderResponse
from msu_llm_lab.rag.models import RetrievalResult
from msu_llm_lab.rag.retriever import BaseRetriever


class LiveFakeProvider(BaseProvider):
    """Returns the scripted output for a case id (default 'answer'). Test double only: nothing in src simulates a provider."""

    def __init__(self, answers=None):
        super().__init__(api_key=None)
        self.calls = 0
        self.answers = answers or {}

    @classmethod
    def echoing_expected(cls, dataset_path):
        from msu_llm_lab.dataset.manager import DatasetManager
        return cls({c.id: (c.expected_answer or "answer") for c in DatasetManager.load_jsonl(dataset_path)})

    def _get_pricing_for_model(self, model_id):
        return Decimal("0.1"), Decimal("0.1")

    def generate(self, request):
        self.calls += 1
        return ProviderResponse(
            output_text=self.answers.get(request.case_id, "answer"), prompt_tokens=100, completion_tokens=20, usage_source="PROVIDER_REPORTED",
            latency_ms=50.0, cost_usd=Decimal("0.0001"), cost_basis="ESTIMATED_FROM_CATALOG", model_id=request.model_id, 
        )


def seed_runs(db, n=2):
    """Insert n recorded (non-simulated) evaluation runs into a test database."""
    db.init_db()
    for i in range(n):
        db.save_evaluation_run({
            "run_id": f"run_test_{i}", "timestamp": f"2026-10-15T0{i}:00:00+00:00", "model_id": "openai/gpt-4.1-nano", "dataset_path": "synthetic.jsonl",
            "total_cases": 3, "completed_cases": 3, "failed_cases": 0, "total_cost_usd": "0.0003", "mean_latency_ms": 50.0, "mean_accuracy": 0.9,
            "mean_groundedness": 0.9, "mean_overall_score": 0.9, "is_dry_run": False, "notes": "test fixture",
        })


class ExpectedContextRetriever(BaseRetriever):
    """Test double that returns each case's expected context (a retrieval ceiling for metric tests)."""

    def retrieve(self, case, top_k=2):
        chunks = [case.expected_context] if case.expected_context else []
        return RetrievalResult(chunks=chunks, retriever_id="test-double", latency_ms=22.0, hit_rate=1.0 if chunks else 0.0,
                               precision=1.0 if chunks else 0.0)
