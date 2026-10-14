"""Diagnostic engine categorizing RAG trace failures across routing, retrieval, generation, and KB."""

import re
from typing import List, Optional, Tuple

from msu_llm_lab.traces.models import (
    DiagnosisHypothesis,
    FailureCategory,
    ObservationSpan,
    TraceDiagnosticReport,
    TraceRecord,
)


class TraceDiagnosticAnalyzer:
    """Diagnoses production trace failures across the 4 RAG failure categories.
    
    Acceptance Criteria: All findings are formulated as candidate hypotheses with
    explicit confidence scores and supporting evidence, never as unverified ground truth.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def analyze_trace(self, trace: TraceRecord) -> DiagnosisHypothesis:
        """Analyzes a single trace and assigns the most probable candidate diagnosis."""
        q_lower = trace.input_query.lower()
        ans_lower = trace.output_response.lower()

        # Extract retrieval context and routing span if present
        retrieval_text = self._extract_retrieval_context(trace)
        routing_info = self._extract_routing_info(trace)

        # 1. Evaluate Routing Failure Candidate
        is_routing, r_conf, r_hyp, r_ev, r_act = self._check_routing_failure(trace, q_lower, routing_info)
        if is_routing and r_conf >= self.min_confidence:
            return DiagnosisHypothesis(
                trace_id=trace.id,
                category=FailureCategory.ROUTING_FAILURE,
                confidence=r_conf,
                hypothesis=r_hyp,
                evidence=r_ev,
                suggested_action=r_act,
            )

        # 2. Evaluate Retrieval Failure Candidate
        is_retrieval, ret_conf, ret_hyp, ret_ev, ret_act = self._check_retrieval_failure(trace, q_lower, retrieval_text)
        if is_retrieval and ret_conf >= self.min_confidence:
            return DiagnosisHypothesis(
                trace_id=trace.id,
                category=FailureCategory.RETRIEVAL_FAILURE,
                confidence=ret_conf,
                hypothesis=ret_hyp,
                evidence=ret_ev,
                suggested_action=ret_act,
            )

        # 3. Evaluate Source Knowledge Defect Candidate
        is_kb, kb_conf, kb_hyp, kb_ev, kb_act = self._check_source_knowledge_failure(trace, retrieval_text, ans_lower)
        if is_kb and kb_conf >= self.min_confidence:
            return DiagnosisHypothesis(
                trace_id=trace.id,
                category=FailureCategory.SOURCE_KNOWLEDGE_FAILURE,
                confidence=kb_conf,
                hypothesis=kb_hyp,
                evidence=kb_ev,
                suggested_action=kb_act,
            )

        # 4. Evaluate Generation Failure Candidate
        is_gen, g_conf, g_hyp, g_ev, g_act = self._check_generation_failure(trace, q_lower, retrieval_text, ans_lower)
        if is_gen and g_conf >= self.min_confidence:
            return DiagnosisHypothesis(
                trace_id=trace.id,
                category=FailureCategory.GENERATION_FAILURE,
                confidence=g_conf,
                hypothesis=g_hyp,
                evidence=g_ev,
                suggested_action=g_act,
            )

        # Default: No significant failure detected
        return DiagnosisHypothesis(
            trace_id=trace.id,
            category=FailureCategory.NO_FAILURE_DETECTED,
            confidence=0.85,
            hypothesis="Trace response appears grounded, coherent, and within normal operational bounds.",
            evidence=[f"Latency: {trace.latency_ms:.1f}ms", f"Tokens: {trace.total_tokens}"],
            suggested_action="No remediation necessary.",
        )

    def analyze_batch(self, traces: List[TraceRecord]) -> TraceDiagnosticReport:
        """Analyzes a collection of traces and generates an aggregated diagnostic report."""
        diagnoses: List[DiagnosisHypothesis] = []
        category_counts = {cat.value: 0 for cat in FailureCategory}
        total_lat = 0.0
        total_tokens = 0
        failure_count = 0

        for t in traces:
            diag = self.analyze_trace(t)
            diagnoses.append(diag)
            category_counts[diag.category.value] += 1
            if diag.category != FailureCategory.NO_FAILURE_DETECTED:
                failure_count += 1
            total_lat += t.latency_ms
            total_tokens += t.total_tokens

        n = len(traces)
        mean_lat = round(total_lat / n, 2) if n > 0 else 0.0

        return TraceDiagnosticReport(
            total_traces=n,
            diagnosed_failures=failure_count,
            category_counts=category_counts,
            mean_latency_ms=mean_lat,
            total_tokens=total_tokens,
            diagnoses=diagnoses,
        )

    def _extract_retrieval_context(self, trace: TraceRecord) -> str:
        """Extracts text content from retrieval spans or metadata."""
        for s in trace.spans:
            if "retriev" in s.name.lower() or s.type == "retrieval":
                if isinstance(s.output_data, str):
                    return s.output_data
                elif isinstance(s.output_data, list):
                    return " ".join(str(item) for item in s.output_data)
                elif isinstance(s.output_data, dict):
                    return str(s.output_data)
        return str(trace.metadata.get("retrieval_context") or trace.metadata.get("context") or "")

    def _extract_routing_info(self, trace: TraceRecord) -> str:
        """Extracts route classification tag or intent span."""
        for s in trace.spans:
            if any(k in s.name.lower() for k in ("router", "intent", "classifier", "route")):
                return str(s.output_data or s.metadata.get("route") or "")
        return " ".join(trace.tags)

    def _check_routing_failure(
        self, trace: TraceRecord, q_lower: str, routing_info: str
    ) -> Tuple[bool, float, str, List[str], str]:
        """Checks for intent/level/faculty routing mismatch."""
        route_lower = routing_info.lower()

        # Check Postgraduate vs Undergraduate mismatch
        is_postgrad_query = any(k in q_lower for k in ("phd", "doctor of philosophy", "master", "postgraduate"))
        is_undergrad_route = "undergraduate" in route_lower or "diploma" in route_lower or "foundation" in route_lower

        if is_postgrad_query and is_undergrad_route:
            return (
                True,
                0.90,
                "Candidate Routing Mismatch: Postgraduate inquiry misrouted to undergraduate admissions flow.",
                [f"Query terms: '{q_lower[:60]}...'", f"Assigned Route: '{routing_info}'"],
                "Retrain intent classifier embeddings with explicit postgraduate qualification discriminators.",
            )

        # Check Medical / Faculty domain mismatch
        is_med_query = any(k in q_lower for k in ("mbbs", "medicine", "pharmacy", "medical"))
        is_business_route = any(k in route_lower for k in ("business", "accounting", "management"))

        if is_med_query and is_business_route:
            return (
                True,
                0.88,
                "Candidate Routing Mismatch: Medical sciences query directed to Faculty of Business router.",
                [f"Query snippet: '{q_lower[:60]}...'", f"Assigned Route: '{routing_info}'"],
                "Adjust router topic priority rules to prevent cross-faculty misclassification.",
            )

        return False, 0.0, "", [], ""

    def _check_retrieval_failure(
        self, trace: TraceRecord, q_lower: str, retrieval_text: str
    ) -> Tuple[bool, float, str, List[str], str]:
        """Checks for empty chunks, low similarity, or missing programme context."""
        ret_lower = retrieval_text.lower().strip()

        # 1. Empty or near-empty retrieval
        if not ret_lower or ret_lower == "[]" or ret_lower == "none":
            return (
                True,
                0.95,
                "Candidate Retrieval Failure: Vector search returned zero qualifying context chunks.",
                ["Retrieved context is empty or null.", f"Query: '{q_lower[:70]}'"],
                "Lower similarity threshold ceiling or inspect dense vector embedding index coverage.",
            )

        # 2. Programme divergence (e.g. Query asked about Computer Science, but retrieved only Accounting)
        cs_terms = ("computer science", "software engineering", "cybersecurity", "information technology")
        biz_terms = ("accounting", "finance", "business administration", "marketing")

        if any(term in q_lower for term in cs_terms) and not any(term in ret_lower for term in cs_terms):
            if any(term in ret_lower for term in biz_terms):
                return (
                    True,
                    0.86,
                    "Candidate Retrieval Divergence: Retrieved context belongs to unrelated faculty.",
                    [f"Student Query: '{q_lower[:60]}...'", f"Retrieved Context: '{retrieval_text[:80]}...'"],
                    "Implement pre-retrieval metadata filtering by faculty or degree level.",
                )

        return False, 0.0, "", [], ""

    def _check_source_knowledge_failure(
        self, trace: TraceRecord, retrieval_text: str, ans_lower: str
    ) -> Tuple[bool, float, str, List[str], str]:
        """Checks if response faithfully mirrors defective or contradictory knowledge-base text."""
        # 1. Contradictory tuition fees in retrieved context
        rm_matches = re.findall(r"rm\s*[\d,]+", retrieval_text.lower())
        unique_rms = list(set(rm_matches))
        if len(unique_rms) >= 2:
            # Check if answer also repeats the conflicting figures
            if all(rm in ans_lower for rm in unique_rms[:2]):
                return (
                    True,
                    0.92,
                    "Candidate Knowledge Base Defect: Model faithfully relayed conflicting fee figures present in source KB.",
                    [f"Conflicting figures in context: {unique_rms[:2]}", f"Output quotes: '{ans_lower[:80]}...'"],
                    "Execute 'msu-llm-lab remediate-kb' to reconcile conflicting figures with Dean's office records.",
                )

        # 2. Stale intake calendar references (e.g. 2022, 2023, 2024 in context)
        stale_years = re.findall(r"\b(202[0-4])\b", retrieval_text)
        if stale_years and any(yr in ans_lower for yr in stale_years):
            return (
                True,
                0.80,
                f"Candidate Knowledge Base Stale Data: Ingested chunk contains outdated calendar year ({stale_years[0]}).",
                [f"Outdated years detected in context: {stale_years}", f"Output: '{ans_lower[:70]}...'"],
                "Update admissions calendar references in KB to current 2026/2027 intake cycles.",
            )

        return False, 0.0, "", [], ""

    def _check_generation_failure(
        self, trace: TraceRecord, q_lower: str, retrieval_text: str, ans_lower: str
    ) -> Tuple[bool, float, str, List[str], str]:
        """Checks for hallucinations, ungrounded fee amounts, or model refusals."""
        # 1. False refusal despite context present
        refusal_phrases = ("i do not have access", "i don't know", "information is not available", "cannot find")
        if any(rp in ans_lower for rp in refusal_phrases) and len(retrieval_text) > 100:
            return (
                True,
                0.85,
                "Candidate Generation Failure: Model emitted refusal despite valid context present in prompt.",
                [f"Model refusal snippet: '{ans_lower[:70]}...'", f"Context length: {len(retrieval_text)} chars"],
                "Refine system prompt instructions to encourage direct citation of available context.",
            )

        # 2. Ungrounded fee figures (hallucination)
        out_rms = re.findall(r"rm\s*[\d,]{4,}", ans_lower)
        ctx_lower = retrieval_text.lower()
        ungrounded = [rm for rm in out_rms if rm not in ctx_lower]
        if ungrounded:
            return (
                True,
                0.88,
                f"Candidate Generation Hallucination: Model invented fee figure '{ungrounded[0]}' not in context.",
                [f"Invented figure: {ungrounded[0]}", f"Context available: '{retrieval_text[:80]}...'"],
                "Adjust generation temperature to 0.0 and add strict grounding constraints to system prompt.",
            )

        return False, 0.0, "", [], ""
