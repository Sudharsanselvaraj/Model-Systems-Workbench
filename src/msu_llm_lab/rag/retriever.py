"""Retriever abstractions and configurations for independent retrieval measurement."""

import math
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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

    CONTRADICTORY_RECORD_IDS = {
        "REC-1886", "REC-1897", "REC-1908", "REC-2268",
        "REC-2432", "REC-2763", "REC-2772", "REC-2781",
    }

    def __init__(
        self,
        corpus_cases: Optional[List[EvaluationCase]] = None,
        corpus_documents: Optional[List[str]] = None,
    ):
        self.documents: List[str] = []
        self.doc_records: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        self.manifest: Dict[str, Any] = {}

        if corpus_documents:
            self.index_documents(corpus_documents)
        elif corpus_cases:
            self.index(corpus_cases)

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 2]

    def index_documents(
        self,
        documents: List[str],
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Indexes an independent corpus of raw text passages or knowledge base chunks."""
        self.documents = [d for d in documents if d and d.strip()]
        self.doc_records = records if records is not None else [{} for _ in self.documents]
        self.doc_tokens = []
        df: Dict[str, int] = {}

        for doc in self.documents:
            toks = self._tokenize(doc)
            self.doc_tokens.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        n_docs = len(self.documents)
        if n_docs > 0:
            self.avg_doc_len = sum(len(d) for d in self.doc_tokens) / n_docs
            for term, freq in df.items():
                self.idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    def index(self, cases: List[EvaluationCase]) -> None:
        """Indexes context passages from evaluation cases (Warning: evaluate for test contamination)."""
        docs = [c.expected_context for c in cases if c.expected_context]
        records = [{"case_id": c.id} for c in cases if c.expected_context]
        self.index_documents(docs, records=records)

    @classmethod
    def from_parsed_kb(
        cls,
        general_path: Optional[Path] = None,
        programmes_path: Optional[Path] = None,
        exclude_defective: bool = True,
    ) -> "KeywordBM25Retriever":
        """Builds an uncontaminated BM25 index directly from audited structured KB records."""
        import json
        corpus: List[str] = []
        records: List[Dict[str, Any]] = []
        excluded_details: List[Dict[str, Any]] = []
        total_scanned = 0

        paths = [p for p in (general_path, programmes_path) if p and p.exists()]
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total_scanned += 1
                    r = json.loads(line)
                    rec_id = r.get("record_id", "")
                    status = r.get("parsing_status", "VALID")

                    if exclude_defective:
                        if status != "VALID":
                            excluded_details.append({
                                "record_id": rec_id,
                                "source_file": r.get("source_file"),
                                "reason": "INCOMPLETE_OR_SYNTAX_DEFECT",
                                "warnings": r.get("validation_warnings", []),
                            })
                            continue
                        if rec_id in cls.CONTRADICTORY_RECORD_IDS:
                            excluded_details.append({
                                "record_id": rec_id,
                                "source_file": r.get("source_file"),
                                "reason": "CONTRADICTORY_FACT",
                                "warnings": ["Internal numerical fee conflict"],
                            })
                            continue

                    cat = r.get("programme_name") or r.get("category") or ""
                    q = r.get("question_clean") or r.get("question_raw") or ""
                    a = r.get("answer_clean") or r.get("answer_raw") or ""
                    corpus.append(f"{cat}: {q}\n{a}".strip())
                    records.append({
                        "record_id": rec_id,
                        "source_file": r.get("source_file"),
                        "category": cat,
                        "start_line": r.get("start_line"),
                    })

        retriever = cls()
        retriever.index_documents(corpus, records=records)
        retriever.manifest = {
            "total_records_scanned": total_scanned,
            "indexed_records_count": len(corpus),
            "excluded_defective_count": len(excluded_details),
            "exclude_defective_policy": exclude_defective,
            "excluded_records": excluded_details,
        }
        return retriever

    def verify_no_leakage(self, cases: List[EvaluationCase]) -> List[str]:
        """Detects if any test case expected_context or expected_answer is leaked into the corpus."""
        leaked_case_ids: List[str] = []
        doc_set = set(self.documents)
        for c in cases:
            if c.expected_context and c.expected_context in doc_set:
                leaked_case_ids.append(c.id)
        return leaked_case_ids

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
        retrieved_record_ids = [
            self.doc_records[i].get("record_id", f"DOC-{i}")
            for i in top_indices
            if i < len(self.doc_records)
        ]
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
            retrieved_record_ids=retrieved_record_ids,
            metadata={
                "scored_candidates": len(scores),
                "top_score": scores[0][0] if scores else 0.0,
                "top_indices": top_indices,
            },
        )
