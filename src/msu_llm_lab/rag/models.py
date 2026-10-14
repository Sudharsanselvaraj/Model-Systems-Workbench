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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGCaseEvaluation(BaseModel):
    """Granular evaluation record for a single case in an end-to-end RAG benchmark."""
    case_id: str
    model_id: str
    retriever_id: str
    query: str
    retrieved_chunks: List[str]
    generated_answer: str
    expected_answer: Optional[str] = None
    
    # Decoupled Retrieval Metrics
    retrieval_hit_rate: float = Field(ge=0.0, le=1.0)
    retrieval_precision: float = Field(ge=0.0, le=1.0)
    retrieval_latency_ms: float = Field(ge=0.0)

    # Decoupled Generation Metrics
    answer_correctness: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    generation_latency_ms: float = Field(ge=0.0)

    # End-to-End System Metrics
    e2e_latency_ms: float = Field(ge=0.0)
    cost_usd: Decimal = Field(default=Decimal("0.0000"))
    failure_attribution: str = Field(description="SYSTEM_SUCCESS, RETRIEVAL_FAILURE, GENERATION_FAILURE, or DUAL_FAILURE")


class RAGBenchmarkReport(BaseModel):
    """Aggregated benchmark report isolating retrieval performance from model generation quality."""
    run_id: str
    timestamp: str
    model_id: str
    retriever_id: str
    split: str
    total_cases: int
    
    # Aggregated Retrieval Metrics
    mean_retrieval_hit_rate: float
    mean_retrieval_precision: float
    mean_retrieval_latency_ms: float
    
    # Aggregated Generation Metrics
    mean_answer_correctness: float
    mean_groundedness: float
    mean_generation_latency_ms: float
    
    # Aggregated End-to-End Metrics
    mean_e2e_latency_ms: float
    total_cost_usd: Decimal
    failure_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    case_results: List[RAGCaseEvaluation] = Field(default_factory=list)
