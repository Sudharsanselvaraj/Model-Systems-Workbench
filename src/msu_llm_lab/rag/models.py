"""Data models for decoupled RAG benchmarking: retrieval, generation, and end-to-end telemetry."""

from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Output telemetry from an independent retrieval pass."""
    chunks: List[str] = Field(default_factory=list, description="Retrieved context passages")
    retriever_id: str = Field(description="Identifier of retriever configuration (e.g. 'perfect', 'noisy', 'bm25')")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Time taken to perform retrieval")
    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Whether expected context/keywords were found")
    precision: float = Field(default=0.0, ge=0.0, le=1.0, description="Proportion of expected keywords retrieved")
    retrieved_record_ids: List[str] = Field(default_factory=list, description="IDs of the retrieved records in the corpus")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGCaseEvaluation(BaseModel):
    """Granular evaluation record for a single case in an end-to-end RAG benchmark."""
    case_id: str
    model_id: str
    retriever_id: str
    query: str
    retrieved_chunks: List[str]
    retrieved_record_ids: List[str] = Field(default_factory=list, description="IDs of the retrieved records")
    generated_answer: str
    expected_answer: Optional[str] = None
    
    # Grounded Provenance Tracking
    source_status: str = Field(default="SYNTHETIC", description="SOURCE_VERIFIED, SYNTHETIC, or UNRESOLVED")
    relevant_record_ids: List[str] = Field(default_factory=list, description="Target record IDs from source KB")
    abstention_expected: bool = Field(default=False)
    
    # Ground-Truth Source Retrieval Metrics (Evaluated against relevant_record_ids)
    recall_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    precision_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    reciprocal_rank: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_hit: bool = Field(default=False)
    
    # Diagnostic Lexical Keyword Metrics (Secondary)
    retrieval_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Diagnostic keyword hit rate")
    retrieval_precision: float = Field(default=0.0, ge=0.0, le=1.0, description="Diagnostic keyword precision")
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0)

    # Decoupled Generation Metrics
    answer_correctness: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    abstention_correctness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    generation_latency_ms: float = Field(ge=0.0)

    # End-to-End System Metrics
    e2e_latency_ms: float = Field(ge=0.0)
    cost_usd: Decimal = Field(default=Decimal("0.0000"))
    failure_attribution: str = Field(description="SYSTEM_SUCCESS, RETRIEVAL_FAILURE, GENERATION_FAILURE, DUAL_FAILURE, or ABSTENTION_SUCCESS")


class RAGBenchmarkReport(BaseModel):
    """Aggregated benchmark report isolating retrieval performance from model generation quality."""
    run_id: str
    timestamp: str
    model_id: str
    retriever_id: str
    split: str
    total_cases: int
    
    # Case Provenance Partitioning
    verified_cases_count: int = Field(default=0, description="Cases backed by verified source KB records")
    unresolved_cases_count: int = Field(default=0, description="Cases unsupported by KB expecting abstention")
    
    # Source-Grounded Retrieval Metrics (Evaluated strictly over verified cases)
    mean_recall_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_precision_at_k: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(default=0.0, ge=0.0, le=1.0)
    verified_retrieval_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_retrieval_latency_ms: float = Field(default=0.0)
    
    # Abstention Metric (Evaluated over unresolved cases)
    abstention_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # Diagnostic Secondary Metrics
    diagnostic_keyword_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_retrieval_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_retrieval_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Aggregated Generation Metrics
    mean_answer_correctness: float = Field(default=0.0)
    mean_groundedness: float = Field(default=0.0)
    mean_generation_latency_ms: float = Field(default=0.0)
    
    # Aggregated End-to-End Metrics
    mean_e2e_latency_ms: float = Field(default=0.0)
    total_cost_usd: Decimal = Field(default=Decimal("0.0000"))
    failure_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    case_results: List[RAGCaseEvaluation] = Field(default_factory=list)
