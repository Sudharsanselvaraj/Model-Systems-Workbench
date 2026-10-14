"""Retriever abstractions and configurations for independent retrieval measurement."""

import math
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

from msu_llm_lab.rag.models import RetrievalResult
from msu_llm_lab.schemas.evaluation import EvaluationCase


class BaseRetriever(ABC):
    """Abstract interface for RAG context retrieval engines."""

    @abstractmethod
    def retrieve(self, case: EvaluationCase) -> RetrievalResult:
        """Executes retrieval for an evaluation case, returning chunks and telemetry."""
        pass


class MockRetriever(BaseRetriever):
    """Deterministic mock retriever supporting perfect, noisy, and empty retrieval scenarios."""

    def __init__(self, mode: str = "perfect", simulated_latency_ms: float = 22.0):
        self.mode = mode.lower()
        self.simulated_latency_ms = simulated_latency_ms

    def retrieve(self, case: EvaluationCase) -> RetrievalResult:
        """Retrieves passages based on configured test mode."""
        if self.mode == "empty":
            return RetrievalResult(
                chunks=[],
                retriever_id=f"mock-{self.mode}",
                latency_ms=self.simulated_latency_ms,
                hit_rate=0.0,
                precision=0.0,
                metadata={"mode": self.mode, "reason": "Simulated empty retrieval failure"},
            )

        if self.mode == "noisy":
            # Injects distractor chunk with the expected context
            chunks = []
            if case.expected_context:
                chunks.append(case.expected_context)
            chunks.append(
                "MSU Faculty of Business offers various diploma programmes in banking and finance. "
                "Applications are open throughout the calendar year."
            )
            return RetrievalResult(
                chunks=chunks,
                retriever_id=f"mock-{self.mode}",
                latency_ms=self.simulated_latency_ms + 10.0,
                hit_rate=1.0 if case.expected_context else 0.0,
                precision=0.50,
                metadata={"mode": self.mode, "distractor_count": 1},
            )

        # Default: Perfect Ground Truth Context
        chunks = [case.expected_context] if case.expected_context else []
        return RetrievalResult(
            chunks=chunks,
            retriever_id=f"mock-{self.mode}",
            latency_ms=self.simulated_latency_ms,
            hit_rate=1.0 if chunks else 0.0,
            precision=1.0 if chunks else 0.0,
            metadata={"mode": self.mode},
        )


class KeywordBM25Retriever(BaseRetriever):
    """Lightweight in-memory BM25-style lexical retriever across corpus passages."""

    def __init__(self, corpus_cases: Optional[List[EvaluationCase]] = None):
        self.documents: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}

        if corpus_cases:
            self.index(corpus_cases)

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 2]

    def index(self, cases: List[EvaluationCase]) -> None:
        """Indexes context passages from evaluation cases."""
        self.documents = []
        self.doc_tokens = []
        df: Dict[str, int] = {}

        for c in cases:
            if c.expected_context:
                self.documents.append(c.expected_context)
                toks = self._tokenize(c.expected_context)
                self.doc_tokens.append(toks)
                unique_toks = set(toks)
                for t in unique_toks:
                    df[t] = df.get(t, 0) + 1

        n_docs = len(self.documents)
        if n_docs > 0:
            self.avg_doc_len = sum(len(d) for d in self.doc_tokens) / n_docs
            for term, freq in df.items():
                self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def retrieve(self, case: EvaluationCase, top_k: int = 2) -> RetrievalResult:
        """Scores indexed passages and returns top-k matching chunks with latency and precision."""
        start_t = time.perf_counter()
        q_toks = self._tokenize(case.query)

        k1 = 1.5
        b = 0.75
        scores: List[tuple[float, int]] = []

        for idx, doc_toks in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = len(doc_toks)
            tf_counts: Dict[str, int] = {}
            for t in doc_toks:
                tf_counts[t] = tf_counts.get(t, 0) + 1

            for q in q_toks:
                if q in tf_counts:
                    idf_val = self.idf.get(q, 0.5)
                    tf_val = tf_counts[q]
                    denom = tf_val + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_len)))
                    score += idf_val * (tf_val * (k1 + 1.0)) / max(1e-5, denom)

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for s, idx in scores[:top_k] if s > 0.0]

        retrieved_chunks = [self.documents[i] for i in top_indices]
        lat_ms = (time.perf_counter() - start_t) * 1000.0 + 8.5  # include simulated indexing overhead

        # Calculate hit rate & precision based on expected keywords
        hit_rate = 0.0
        precision = 0.0
        if case.expected_retrieval_keywords:
            combined_retrieved = " ".join(retrieved_chunks).lower()
            matches = [kw for kw in case.expected_retrieval_keywords if kw.lower() in combined_retrieved]
            precision = round(len(matches) / len(case.expected_retrieval_keywords), 4)
            hit_rate = 1.0 if precision >= 0.50 else 0.0
        else:
            hit_rate = 1.0 if retrieved_chunks else 0.0
            precision = 1.0 if retrieved_chunks else 0.0

        return RetrievalResult(
            chunks=retrieved_chunks,
            retriever_id="keyword-bm25",
            latency_ms=round(lat_ms, 2),
            hit_rate=hit_rate,
            precision=precision,
            metadata={"scored_candidates": len(scores), "top_score": scores[0][0] if scores else 0.0},
        )
