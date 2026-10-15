"""Runner orchestrating decoupled end-to-end RAG benchmarking across candidate models and retrievers."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.dataset.manager import DatasetManager
from msu_llm_lab.core.live_guard import LiveAuthorization, require_live_authorization
from msu_llm_lab.providers.base import BaseProvider, ProviderRequest, UsageUnavailableError, require_complete_usage
from msu_llm_lab.providers.openrouter import OpenRouterAdapter
from msu_llm_lab.rag.evaluator import RAGSystemEvaluator
from msu_llm_lab.rag.models import RAGBenchmarkReport, RAGCaseEvaluation
from msu_llm_lab.rag.retriever import BaseRetriever
from msu_llm_lab.schemas.evaluation import DatasetSplit, EvaluationCase
from msu_llm_lab.storage.db import DatabaseManager


class RAGBenchmarkRunner:
    """Orchestrates decoupled RAG evaluation measuring retrieval and generation separately."""

    def __init__(
        self,
        retriever: BaseRetriever,
        db_mgr: Optional[DatabaseManager] = None,
        budget_service: Optional[BudgetService] = None,
        provider: Optional[BaseProvider] = None,
        top_k: int = 2,
    ):
        self.db_mgr = db_mgr or DatabaseManager()
        self.budget_service = budget_service or BudgetService(db=self.db_mgr)
        self.provider = provider or OpenRouterAdapter()
        self.retriever = retriever
        self.evaluator = RAGSystemEvaluator()
        self.top_k = top_k

    def run(
        self,
        model_id: str,
        dataset_path: Path,
        split: Optional[DatasetSplit] = None,
        live_authorization: Optional[LiveAuthorization] = None,
    ) -> RAGBenchmarkReport:
        """Runs the RAG benchmark with real requests. Needs a live authorization covering the model; there is no simulated mode."""
        require_live_authorization(live_authorization, model_id)
        run_id = f"rag_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Load dataset
        manager = DatasetManager()
        all_cases = manager.load_jsonl(dataset_path)

        if split and split != DatasetSplit.UNASSIGNED:
            cases = [c for c in all_cases if c.split == split]
        else:
            cases = all_cases

        if not cases:
            raise ValueError(f"No cases found for evaluation in {dataset_path} (split: {split})")

        # 2. Budget Preflight Check (model-specific pricing based on estimated prompt & output tokens)
        in_price, out_price = self.provider._get_pricing_for_model(model_id)
        
        # Estimate ~600 prompt tokens (system + context + query) and ~150 completion tokens per case
        est_cost_per_case = (Decimal("600") * in_price / Decimal("1000000")) + (Decimal("150") * out_price / Decimal("1000000"))
        estimated_cost = max(Decimal("0.0001"), round(est_cost_per_case * Decimal(len(cases)), 6))
        self.budget_service.check_preflight_budget(
            estimated_cost_usd=estimated_cost,
            is_pilot=True,
            model_id=model_id,
        )

        case_results: List[RAGCaseEvaluation] = []
        total_cost = Decimal("0.0000")
        failure_breakdown: Dict[str, int] = {
            "SYSTEM_SUCCESS": 0,
            "RETRIEVAL_FAILURE": 0,
            "GENERATION_FAILURE": 0,
            "PARAMETRIC_MEMORY_RESCUE": 0,
        }

        # 3. Execution Loop
        for case in cases:
            OpenRouterAdapter.validate_case_confidentiality(case)

            # Step 3a: Decoupled Retrieval Pass
            import inspect
            sig = inspect.signature(self.retriever.retrieve)
            if "top_k" in sig.parameters:
                ret_result = self.retriever.retrieve(case, top_k=self.top_k)
            else:
                ret_result = self.retriever.retrieve(case)

            # Format grounded RAG prompt
            context_block = "\n---\n".join(ret_result.chunks) if ret_result.chunks else "NO_CONTEXT_RETRIEVED"
            system_prompt = (
                "You are the official MSU Admissions AI Counsellor for Management and Science University, Malaysia. "
                "Answer the student's question accurately, grounded strictly in the provided Context. "
                "Never invent fees or admission criteria. If the context does not contain sufficient details, state that clearly."
            )
            user_content = f"Context:\n{context_block}\n\nStudent Query: {case.query}"

            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # Step 3b: Decoupled Generation Pass
            request = ProviderRequest(
                model_id=model_id,
                messages=prompt_messages,
                live_authorized=True,
                case_id=case.id,
            )

            response = self.provider.generate(request)
            try:
                require_complete_usage(response)
            except UsageUnavailableError:
                self.budget_service.record_unresolved_spend(f"tx_{run_id}_{case.id}", model_id, "openrouter", f"RAG case {case.id}: response lacked usable cost or token counts", run_id)
                raise

            # Step 3c: Decoupled Scoring & Failure Attribution
            ans_corr, groundedness = self.evaluator.evaluate_generation(
                generated_output=response.output_text,
                retrieved_chunks=ret_result.chunks,
                expected_answer=case.expected_answer,
            )

            # Ground-Truth Source Retrieval Metrics against case.relevant_record_ids
            rel_set = set(case.relevant_record_ids)
            ret_ids = ret_result.retrieved_record_ids

            recall_k = 0.0
            precision_k = 0.0
            rr = 0.0
            ret_hit = False

            if rel_set:
                hits = [r for r in ret_ids if r in rel_set]
                recall_k = round(len(hits) / len(rel_set), 4)
                precision_k = round(len(hits) / len(ret_ids), 4) if ret_ids else 0.0
                ret_hit = len(hits) > 0

                for rank_idx, r in enumerate(ret_ids, 1):
                    if r in rel_set:
                        rr = round(1.0 / rank_idx, 4)
                        break

            # Abstention evaluation
            abstention_corr = None
            if case.abstention_expected:
                lower_gen = response.output_text.lower()
                abstention_phrases = [
                    "does not contain", "not mentioned", "not specified", "consult",
                    "counselor", "contact", "insufficient", "unable to", "not found",
                    "official portal", "reach out", "no information", "lacks details",
                ]
                abstention_corr = 1.0 if any(p in lower_gen for p in abstention_phrases) else 0.0

            # Updated Failure Attribution
            if case.abstention_expected:
                attribution = "ABSTENTION_SUCCESS" if abstention_corr == 1.0 else "OVERCONFIDENT_HALLUCINATION"
            elif rel_set:
                if ret_hit and ans_corr >= 0.70 and groundedness >= 0.40:
                    attribution = "SYSTEM_SUCCESS"
                elif not ret_hit:
                    attribution = "RETRIEVAL_FAILURE"
                else:
                    attribution = "GENERATION_FAILURE"
            else:
                attribution = self.evaluator.attribute_failure(
                    retrieval_hit_rate=ret_result.hit_rate,
                    answer_correctness=ans_corr,
                    groundedness=groundedness,
                )
            failure_breakdown[attribution] = failure_breakdown.get(attribution, 0) + 1

            self.budget_service.record_spend(
                transaction_id=f"tx_{run_id}_{case.id}",
                model_id=model_id,
                provider="openrouter",
                amount_usd=response.cost_usd,
                description=f"RAG Case {case.id} (model={model_id}, ret={ret_result.retriever_id})",
                run_id=run_id,
                is_pilot=True,
                is_preflight_estimate=False,
                cost_basis=response.cost_basis,
            )
            total_cost += response.cost_usd

            e2e_lat = round(ret_result.latency_ms + response.latency_ms, 2)

            case_eval = RAGCaseEvaluation(
                case_id=case.id,
                model_id=model_id,
                retriever_id=ret_result.retriever_id,
                query=case.query,
                retrieved_chunks=ret_result.chunks,
                retrieved_record_ids=ret_result.retrieved_record_ids,
                generated_answer=response.output_text,
                expected_answer=case.expected_answer,
                source_status=str(getattr(case, "source_status", "SYNTHETIC").value if hasattr(getattr(case, "source_status", None), "value") else str(getattr(case, "source_status", "SYNTHETIC"))),
                relevant_record_ids=case.relevant_record_ids,
                abstention_expected=case.abstention_expected,
                recall_at_k=recall_k,
                precision_at_k=precision_k,
                reciprocal_rank=rr,
                retrieval_hit=ret_hit,
                retrieval_hit_rate=ret_result.hit_rate,
                retrieval_precision=ret_result.precision,
                retrieval_latency_ms=ret_result.latency_ms,
                answer_correctness=ans_corr,
                groundedness=groundedness,
                abstention_correctness=abstention_corr,
                generation_latency_ms=response.latency_ms,
                e2e_latency_ms=e2e_lat,
                cost_usd=response.cost_usd,
                failure_attribution=attribution,
            )
            case_results.append(case_eval)

        # 4. Compute Aggregated Metrics
        n = len(case_results)
        verified_cases = [c for c in case_results if c.relevant_record_ids]
        n_ver = len(verified_cases)

        unresolved_cases = [c for c in case_results if c.abstention_expected]
        n_unres = len(unresolved_cases)

        mean_recall = round(sum(c.recall_at_k for c in verified_cases) / n_ver, 4) if n_ver else 0.0
        mean_precision = round(sum(c.precision_at_k for c in verified_cases) / n_ver, 4) if n_ver else 0.0
        mean_mrr = round(sum(c.reciprocal_rank for c in verified_cases) / n_ver, 4) if n_ver else 0.0
        ver_hit_rate = round(sum(1.0 if c.retrieval_hit else 0.0 for c in verified_cases) / n_ver, 4) if n_ver else 0.0

        abstention_acc = (
            round(sum(c.abstention_correctness or 0.0 for c in unresolved_cases) / n_unres, 4)
            if n_unres
            else 0.0
        )

        diag_kw_hit = round(sum(c.retrieval_hit_rate for c in case_results) / n, 4) if n else 0.0
        diag_kw_prec = round(sum(c.retrieval_precision for c in case_results) / n, 4) if n else 0.0
        mean_ret_lat = round(sum(c.retrieval_latency_ms for c in case_results) / n, 2) if n else 0.0

        mean_ans_corr = round(sum(c.answer_correctness for c in case_results) / n, 4) if n else 0.0
        mean_grd = round(sum(c.groundedness for c in case_results) / n, 4) if n else 0.0
        mean_gen_lat = round(sum(c.generation_latency_ms for c in case_results) / n, 2) if n else 0.0
        mean_e2e_lat = round(sum(c.e2e_latency_ms for c in case_results) / n, 2) if n else 0.0

        # 5. Persist run to SQLite
        run_record = {
            "run_id": run_id,
            "timestamp": now_iso,
            "model_id": model_id,
            "dataset_path": str(dataset_path),
            "total_cases": len(cases),
            "completed_cases": n,
            "failed_cases": 0,
            "total_cost_usd": str(total_cost),
            "mean_latency_ms": mean_e2e_lat,
            "mean_accuracy": mean_ans_corr,
            "mean_groundedness": mean_grd,
            "mean_overall_score": round((mean_ans_corr + mean_grd) / 2.0, 4),
            "is_dry_run": False,  # legacy column: this tool no longer produces simulated runs
            "notes": (
                f"RAG benchmark (retriever={self.retriever.__class__.__name__}, split={split or 'ALL'}, "
                f"top_k={self.top_k}, recall@k={mean_recall}, mrr={mean_mrr})"
            ),
        }
        self.db_mgr.save_evaluation_run(run_record)

        return RAGBenchmarkReport(
            run_id=run_id,
            timestamp=now_iso,
            model_id=model_id,
            retriever_id=self.retriever.__class__.__name__,
            split=str(split or "ALL"),
            total_cases=n,
            verified_cases_count=n_ver,
            unresolved_cases_count=n_unres,
            mean_recall_at_k=mean_recall,
            mean_precision_at_k=mean_precision,
            mean_reciprocal_rank=mean_mrr,
            verified_retrieval_hit_rate=ver_hit_rate,
            mean_retrieval_latency_ms=mean_ret_lat,
            abstention_accuracy=abstention_acc,
            diagnostic_keyword_hit_rate=diag_kw_hit,
            mean_retrieval_hit_rate=diag_kw_hit,
            mean_retrieval_precision=diag_kw_prec,
            mean_answer_correctness=mean_ans_corr,
            mean_groundedness=mean_grd,
            mean_generation_latency_ms=mean_gen_lat,
            mean_e2e_latency_ms=mean_e2e_lat,
            total_cost_usd=total_cost,
            failure_breakdown=failure_breakdown,
            case_results=case_results,
        )
