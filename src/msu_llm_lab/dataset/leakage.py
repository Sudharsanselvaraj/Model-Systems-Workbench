"""Cross-split evaluation dataset leakage detector."""

import re
from enum import Enum
from typing import List, Set, Tuple
from pydantic import BaseModel, Field

from msu_llm_lab.schemas.evaluation import DatasetSplit, EvaluationCase


class LeakageType(str, Enum):
    """Classification of detected cross-split contamination."""
    EXACT_QUERY_MATCH = "EXACT_QUERY_MATCH"
    NORMALIZED_QUERY_MATCH = "NORMALIZED_QUERY_MATCH"
    HIGH_NGRAM_OVERLAP = "HIGH_NGRAM_OVERLAP"


class LeakageFinding(BaseModel):
    """Specific instance of cross-split contamination between two evaluation cases."""
    case_a_id: str
    split_a: DatasetSplit
    case_b_id: str
    split_b: DatasetSplit
    leakage_type: LeakageType
    similarity_score: float = Field(ge=0.0, le=1.0)
    query_a: str
    query_b: str
    shared_ngrams: List[str] = Field(default_factory=list)


class LeakageReport(BaseModel):
    """Aggregated leakage evaluation report."""
    total_cases_analyzed: int = 0
    splits_evaluated: List[DatasetSplit] = Field(default_factory=list)
    has_leakage: bool = False
    findings: List[LeakageFinding] = Field(default_factory=list)
    verbatim_leak_count: int = 0
    near_duplicate_leak_count: int = 0


class LeakageDetector:
    """Detects train/dev/test data leakage and test-set contamination."""

    def __init__(self, n_gram_size: int = 3, similarity_threshold: float = 0.70):
        self.n_gram_size = n_gram_size
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercases, removes punctuation, and normalizes whitespace."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return " ".join(cleaned.split())

    def extract_ngrams(self, text: str) -> Set[Tuple[str, ...]]:
        """Extracts word n-grams from normalized text."""
        words = self.normalize_text(text).split()
        if len(words) < self.n_gram_size:
            return {tuple(words)} if words else set()
        return {
            tuple(words[i : i + self.n_gram_size])
            for i in range(len(words) - self.n_gram_size + 1)
        }

    def compute_jaccard_similarity(
        self, set_a: Set[Tuple[str, ...]], set_b: Set[Tuple[str, ...]]
    ) -> float:
        """Computes Jaccard similarity between two n-gram sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def analyze(self, cases: List[EvaluationCase]) -> LeakageReport:
        """Analyzes a set of evaluation cases for cross-split contamination."""
        report = LeakageReport(total_cases_analyzed=len(cases))
        
        # Filter for partitioned cases only
        partitioned = [
            c for c in cases if c.split in (DatasetSplit.TRAIN, DatasetSplit.DEV, DatasetSplit.TEST)
        ]
        active_splits = sorted(list(set(c.split for c in partitioned)), key=lambda s: s.value)
        report.splits_evaluated = active_splits

        # Pairwise cross-split comparison
        for i in range(len(partitioned)):
            for j in range(i + 1, len(partitioned)):
                case_a = partitioned[i]
                case_b = partitioned[j]

                # Only check across distinct splits (e.g. TRAIN vs TEST, DEV vs TEST)
                if case_a.split == case_b.split:
                    continue

                q_raw_a = case_a.query.strip()
                q_raw_b = case_b.query.strip()

                # 1. Exact raw match
                if q_raw_a == q_raw_b:
                    finding = LeakageFinding(
                        case_a_id=case_a.id,
                        split_a=case_a.split,
                        case_b_id=case_b.id,
                        split_b=case_b.split,
                        leakage_type=LeakageType.EXACT_QUERY_MATCH,
                        similarity_score=1.0,
                        query_a=case_a.query,
                        query_b=case_b.query,
                    )
                    report.findings.append(finding)
                    report.verbatim_leak_count += 1
                    continue

                # 2. Normalized match
                q_norm_a = self.normalize_text(case_a.query)
                q_norm_b = self.normalize_text(case_b.query)
                if q_norm_a == q_norm_b and q_norm_a != "":
                    finding = LeakageFinding(
                        case_a_id=case_a.id,
                        split_a=case_a.split,
                        case_b_id=case_b.id,
                        split_b=case_b.split,
                        leakage_type=LeakageType.NORMALIZED_QUERY_MATCH,
                        similarity_score=1.0,
                        query_a=case_a.query,
                        query_b=case_b.query,
                    )
                    report.findings.append(finding)
                    report.verbatim_leak_count += 1
                    continue

                # 3. High n-gram overlap
                ngrams_a = self.extract_ngrams(case_a.query)
                ngrams_b = self.extract_ngrams(case_b.query)
                jaccard = self.compute_jaccard_similarity(ngrams_a, ngrams_b)

                if jaccard >= self.similarity_threshold:
                    shared = [" ".join(ng) for ng in ngrams_a.intersection(ngrams_b)]
                    finding = LeakageFinding(
                        case_a_id=case_a.id,
                        split_a=case_a.split,
                        case_b_id=case_b.id,
                        split_b=case_b.split,
                        leakage_type=LeakageType.HIGH_NGRAM_OVERLAP,
                        similarity_score=round(jaccard, 4),
                        query_a=case_a.query,
                        query_b=case_b.query,
                        shared_ngrams=shared[:5],
                    )
                    report.findings.append(finding)
                    report.near_duplicate_leak_count += 1

        report.has_leakage = len(report.findings) > 0
        return report
