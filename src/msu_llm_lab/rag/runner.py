"""Runner orchestrating decoupled end-to-end RAG benchmarking across candidate models and retrievers."""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.dataset.manager import DatasetManager
from msu_llm_lab.providers.base import BaseProvider, ProviderRequest
from msu_llm_lab.providers.openrouter import OpenRouterAdapter
from msu_llm_lab.rag.evaluator import RAGSystemEvaluator
from msu_llm_lab.rag.models import RAGBenchmarkReport, RAGCaseEvaluation
from msu_llm_lab.rag.retriever import BaseRetriever, MockRetriever
from msu_llm_lab.schemas.evaluation import DatasetSplit, EvaluationCase
from msu_llm_lab.storage.db import DatabaseManager


class RAGBenchmarkRunner:
    """Orchestrates decoupled RAG evaluation measuring retrieval and generation separately."""

    def __init__(
        self,
        db_mgr: Optional[DatabaseManager] = None,
        budget_service: Optional[BudgetService] = None,
        provider: Optional[BaseProvider] = None,
        retriever: Optional[BaseRetriever] = None,
    ):
        self.db_mgr = db_mgr or DatabaseManager()
        self.budget_service = budget_service or BudgetService(db=self.db_mgr)
        self.provider = provider or OpenRouterAdapter()
        self.retriever = retriever or MockRetriever(mode="perfect")
        self.evaluator = RAGSystemEvaluator()

    def run(
        self,
        model_id: str,
        dataset_path: Path,
        split: Optional[DatasetSplit] = None,
        dry_run: bool = True,
    ) -> RAGBenchmarkReport:
        """Executes full RAG benchmark over specified dataset split."""
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

        # 2. Budget Preflight Check ($0.0004 estimated per case)
        estimated_cost = Decimal("0.0004") * Decimal(len(cases))
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
            if not dry_run:
                OpenRouterAdapter.validate_case_confidentiality(case)

            # Step 3a: Decoupled Retrieval Pass
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
                dry_run=dry_run,
                case_id=case.id,
                expected_answer=case.expected_answer,
            )

            response = self.provider.generate(request)

            # Step 3c: Decoupled Scoring & Failure Attribution
            ans_corr, groundedness = self.evaluator.evaluate_generation(
                generated_output=response.output_text,
                retrieved_chunks=ret_result.chunks,
                expected_answer=case.expected_answer,
            )

            attribution = self.evaluator.attribute_failure(
                retrieval_hit_rate=ret_result.hit_rate,
                answer_correctness=ans_corr,
                groundedness=groundedness,
            )
            failure_breakdown[attribution] = failure_breakdown.get(attribution, 0) + 1

            # Record in spend ledger
            self.budget_service.record_spend(
                transaction_id=f"tx_{run_id}_{case.id}",
                model_id=model_id,
                provider="openrouter",
                amount_usd=response.cost_usd,
                description=f"RAG Case {case.id} (model={model_id}, ret={ret_result.retriever_id})",
                run_id=run_id,
                is_pilot=True,
                is_preflight_estimate=False,
            )
            total_cost += response.cost_usd

            e2e_lat = round(ret_result.latency_ms + response.latency_ms, 2)

            case_eval = RAGCaseEvaluation(
                case_id=case.id,
                model_id=model_id,
                retriever_id=ret_result.retriever_id,
                query=case.query,
                retrieved_chunks=ret_result.chunks,
                generated_answer=response.output_text,
                expected_answer=case.expected_answer,
                retrieval_hit_rate=ret_result.hit_rate,
                retrieval_precision=ret_result.precision,
                retrieval_latency_ms=ret_result.latency_ms,
                answer_correctness=ans_corr,
                groundedness=groundedness,
                generation_latency_ms=response.latency_ms,
                e2e_latency_ms=e2e_lat,
                cost_usd=response.cost_usd,
                failure_attribution=attribution,
            )
            case_results.append(case_eval)

        # 4. Compute Aggregated Metrics
        n = len(case_results)
        mean_ret_hit = round(sum(c.retrieval_hit_rate for c in case_results) / n, 4) if n else 0.0
        mean_ret_prec = round(sum(c.retrieval_precision for c in case_results) / n, 4) if n else 0.0
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
            "is_dry_run": dry_run,
            "notes": f"RAG benchmark (retriever={self.retriever.__class__.__name__}, split={split or 'ALL'})",
        }
        self.db_mgr.save_evaluation_run(run_record)

        return RAGBenchmarkReport(
            run_id=run_id,
            timestamp=now_iso,
            model_id=model_id,
            retriever_id=self.retriever.__class__.__name__,
            split=str(split or "ALL"),
            total_cases=n,
            mean_retrieval_hit_rate=mean_ret_hit,
            mean_retrieval_precision=mean_ret_prec,
            mean_retrieval_latency_ms=mean_ret_lat,
            mean_answer_correctness=mean_ans_corr,
            mean_groundedness=mean_grd,
            mean_generation_latency_ms=mean_gen_lat,
            mean_e2e_latency_ms=mean_e2e_lat,
            total_cost_usd=total_cost,
            failure_breakdown=failure_breakdown,
            case_results=case_results,
        )
