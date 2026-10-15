"""Classification runner. Offline-testable; every guard runs before any request exists.

Order of operations (each step can stop the run before the next one starts):
 1. live authorization covers the model
 2. every case's data classification passes the privacy gate (whole batch rejected if any case fails), and cases are well formed
 3. registry readiness, budget preflight (upper-bound estimate), and context-size checks
 4. requests are built and sent through the protocol-specific adapter
 5. outputs are validated by the authoritative validators (inside the adapters)
 6. prediction records are stored with nullable telemetry
A prediction is never ground truth; expected labels are never sent or stored here.
"""

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Union

from msu_llm_lab.classification import serialization
from msu_llm_lab.classification.contracts import IntentCase, LeadCase
from msu_llm_lab.classification.privacy import assert_dispatch_eligible
from msu_llm_lab.classification.records import (
    AdapterResult, PredictionRecord, PredictionStore, Task, error_record, record_from_result,
)
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.live_guard import LiveAuthorization, require_live_authorization
from msu_llm_lab.models import registry
from msu_llm_lab.providers.base import (
    ConfidentialityViolationError, LiveNotAuthorizedError, ProviderError, UsageUnavailableError,
)

Case = Union[IntentCase, LeadCase]
MAX_STATE_BYTES = 32_000  # Jev's documented long-context threshold; numeric context limits are unverified, so stay below it
SYSTEMIC_STATUSES = frozenset({401, 402, 403})


class BatchRejectedError(ConfidentialityViolationError):
    """The whole batch is refused before any request is built."""

    def __init__(self, message: str, reasons: Dict[str, str]):
        super().__init__(message)
        self.reasons = dict(reasons)


class ModelNotReadyError(ProviderError):
    pass


@dataclass
class RunSummary:
    run_id: str
    task: Task
    model_id: str
    n_cases: int
    n_ok: int = 0
    n_invalid: int = 0
    n_error: int = 0
    n_skipped_existing: int = 0
    recorded_cost_usd: Decimal = Decimal("0")
    guard_cost_usd: Decimal = Decimal("0")     # larger of booked and native cost per case; used only for the stop checks
    aborted_reason: Optional[str] = None
    records: List[PredictionRecord] = field(default_factory=list)


def prompt_version(prompt_text: str) -> str:
    """Content fingerprint of the authoritative prompt actually used."""
    return "sha256:" + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def _state_text(task: Task, case: Case) -> str:
    return serialization.serialize_intent_case(case) if task == Task.INTENT else serialization.serialize_lead_input(case)


def estimate_upper_bound(task: Task, adapter, cases: Sequence[Case], prompt_text: str, prices) -> "tuple[Decimal, Decimal]":
    """(total, worst single case) pessimistic cost in USD: 2 bytes per token for the prompt and state, full max output tokens.
    Used only to guard spend and to print the plan; it is never stored as a cost."""
    prompt_bytes = len(prompt_text.encode("utf-8"))
    total, worst = Decimal("0"), Decimal("0")
    for c in cases:
        toks = math.ceil((prompt_bytes + len(_state_text(task, c).encode("utf-8"))) / 2)
        est = (Decimal(toks) * prices[0] + Decimal(adapter.max_output_tokens(task)) * prices[1]) / Decimal("1000000")
        total += est
        worst = max(worst, est)
    return total, worst


class ClassificationRunner:
    def __init__(
        self,
        adapters: Dict[str, object],
        store: PredictionStore,
        budget_service: BudgetService,
        max_consecutive_errors: int = 3,
        max_state_bytes: int = MAX_STATE_BYTES,
        stage: str = "pilot",
    ):
        self.stage = stage
        self.adapters = adapters
        self.store = store
        self.budget = budget_service
        self.max_consecutive_errors = max_consecutive_errors
        self.max_state_bytes = max_state_bytes

    # ---- step 2: privacy gate and case validation (no request-building code may run before this finishes)
    def _validate_batch(self, task: Task, cases: Sequence[Case]) -> None:
        want = IntentCase if task == Task.INTENT else LeadCase
        reasons: Dict[str, str] = {}
        seen = set()
        for c in cases:
            cid = getattr(c, "case_id", "?")
            if not isinstance(c, want):
                reasons[cid] = f"not a {want.__name__}"
                continue
            if cid in seen:
                reasons[cid] = "duplicate case_id"
            seen.add(cid)
            try:
                assert_dispatch_eligible(c.classification)
            except ConfidentialityViolationError:
                reasons[cid] = f"classification {c.classification.value} is barred from outbound dispatch"
        if reasons:
            raise BatchRejectedError(f"Batch rejected before any request: {len(reasons)} case(s) failed the gate.", reasons)

    # ---- step 3: readiness, size and budget (still no request)
    def _check_ready_and_budget(self, task: Task, adapter, cases: Sequence[Case], prompt_text: str) -> None:
        model_id = adapter.model_id
        if registry.record_for_model_id(model_id) is None:
            raise ModelNotReadyError(f"No sourced registry record for '{model_id}'; unknown model configuration.")
        blockers = registry.synthetic_live_blockers(model_id, self.stage)
        if blockers:
            raise ModelNotReadyError(f"Registry blockers for '{model_id}': {', '.join(blockers)}.")
        prices = registry.model_level_prices(model_id)
        if prices is None:
            raise ModelNotReadyError(f"No usable price for '{model_id}'; budget preflight is impossible.")
        if not adapter.supports(task):
            raise ModelNotReadyError(f"'{model_id}' does not support task {task.value}.")
        too_big = {c.case_id: "state exceeds the size limit"
                   for c in cases if len(_state_text(task, c).encode("utf-8")) > self.max_state_bytes}
        if too_big:
            raise BatchRejectedError("Batch rejected: state exceeds the conservative size limit.", too_big)
        total, worst = estimate_upper_bound(task, adapter, cases, prompt_text, prices)
        per_case_cap = self.budget.policy.per_case_ceiling_usd
        if worst > per_case_cap:
            raise BudgetExceededError(f"Preflight: a case's upper-bound cost ${worst:.6f} exceeds the per-case ceiling ${per_case_cap}.")
        self.budget.check_preflight_budget(estimated_cost_usd=max(total, Decimal("0.000001")), is_pilot=True, model_id=model_id)

    def run(
        self, *, task: Task, cases: Sequence[Case], model_id: str, prompt_text: str,
        authorization: Optional[LiveAuthorization], run_id: Optional[str] = None,
        dataset_version: Optional[str] = None, store_raw_output: bool = False, hard_cap_usd: Optional[Decimal] = None,
    ) -> RunSummary:
        # 1. authorization
        require_live_authorization(authorization, model_id)
        adapter = self.adapters.get(model_id)
        if adapter is None:
            raise ModelNotReadyError(f"No adapter configured for '{model_id}'.")
        # 2. privacy gate over the WHOLE batch
        self._validate_batch(task, cases)
        # 3. readiness and budget
        self._check_ready_and_budget(task, adapter, cases, prompt_text)
        # 4-6. send, validate, store
        run_id = run_id or f"cls_{uuid.uuid4().hex[:10]}"
        done = {cid for rid, cid in self.store.existing_keys() if rid == run_id}
        summary = RunSummary(run_id=run_id, task=task, model_id=model_id, n_cases=len(cases))
        pv = prompt_version(prompt_text)
        consecutive = 0
        for case in cases:
            if case.case_id in done:
                summary.n_skipped_existing += 1
                continue
            try:
                result: AdapterResult = adapter.classify(task, _state_text(task, case), prompt_text, live_authorized=True)
            except (LiveNotAuthorizedError, BudgetExceededError):
                raise
            except ProviderError as exc:
                rec = error_record(exc, run_id=run_id, case_id=case.case_id, task=task, model_key=model_id, protocol=adapter.protocol,
                                   requested_model_id=model_id, prompt_version=pv, dataset_version=dataset_version,
                                   input_contract_version=case.contract_version)
                self.store.append(rec)
                summary.records.append(rec)
                summary.n_error += 1
                consecutive += 1
                if getattr(exc, "status", None) in SYSTEMIC_STATUSES or consecutive >= self.max_consecutive_errors:
                    summary.aborted_reason = (f"systemic provider error (status {exc.status})" if getattr(exc, "status", None) in SYSTEMIC_STATUSES
                                              else f"{consecutive} consecutive errors")
                    break
                continue
            consecutive = 0
            rec = record_from_result(result, run_id=run_id, case_id=case.case_id, model_key=model_id, prompt_version=pv,
                                     dataset_version=dataset_version, store_raw_output=store_raw_output,
                                     input_contract_version=case.contract_version)
            self.store.append(rec)
            summary.records.append(rec)
            if result.ok:
                summary.n_ok += 1
            else:
                summary.n_invalid += 1
            if result.cost_basis == "UNAVAILABLE":
                summary.aborted_reason = "cost unavailable: spend cannot be recorded"
                # The request may have been billed. Record an explicit unresolved entry (no amount is invented); further live runs are
                # blocked by the budget preflight until it is reconciled with a real amount.
                self.budget.record_unresolved_spend(
                    transaction_id=f"tx_{run_id}_{case.case_id}", model_id=model_id, provider="openrouter",
                    description=f"Classification {task.value} case {case.case_id}: response lacked usable cost or token counts", run_id=run_id,
                )
                raise UsageUnavailableError(
                    "Live response has neither a provider-reported cost nor reported tokens for an estimate; run stopped. "
                    "Check the provider dashboard for this request.", attempts=result.attempts,
                )
            self.budget.record_spend(
                transaction_id=f"tx_{run_id}_{case.case_id}", model_id=model_id, provider="openrouter", amount_usd=result.cost_usd,
                description=f"Classification {task.value} case {case.case_id} [{result.cost_basis}]", run_id=run_id, is_pilot=True,
                is_preflight_estimate=False, cost_basis=result.cost_basis,
            )
            summary.recorded_cost_usd += result.cost_usd
            # Guard on the larger of the booked cost and the provider's native cost, so an anomalous native figure stops the run even
            # though it is not booked as USD.
            guard = max(result.cost_usd, result.native_cost or Decimal("0"))
            summary.guard_cost_usd += guard
            if hard_cap_usd is not None and summary.guard_cost_usd > hard_cap_usd:
                summary.aborted_reason = "hard cap exceeded"
                raise BudgetExceededError(f"Guard cost ${summary.guard_cost_usd} exceeds the hard cap ${hard_cap_usd}; run stopped.")
            if guard > self.budget.policy.per_case_ceiling_usd:
                summary.aborted_reason = "per-case ceiling exceeded"
                raise BudgetExceededError(
                    f"Case {case.case_id} cost ${guard} exceeds the per-case ceiling ${self.budget.policy.per_case_ceiling_usd}; run stopped.")
        return summary
