"""Budget calculations, limit checks, and spend governance logic."""

from decimal import Decimal
from typing import Optional, Tuple, List, Dict, Any
from pydantic import BaseModel, Field

from msu_llm_lab.core.config import settings, BudgetPolicy
from msu_llm_lab.storage.db import DatabaseManager


class BudgetExceededError(RuntimeError):
    """Raised when projected or actual spend exceeds local governance policy ceilings."""
    pass


class BudgetStatus(BaseModel):
    """Snapshot of current spend vs local governance ceilings."""
    total_ceiling_usd: Decimal
    cumulative_spend_usd: Decimal
    remaining_total_usd: Decimal
    
    pilot_ceiling_usd: Decimal
    pilot_spend_usd: Decimal
    remaining_pilot_usd: Decimal
    
    per_run_ceiling_usd: Decimal
    per_case_ceiling_usd: Decimal
    
    unresolved_spend_count: int = 0                   # billed-or-not requests with unknown cost; live runs are blocked while > 0
    spend_by_basis: Dict[str, Decimal] = Field(default_factory=dict)
    simulated_spend_usd: Decimal = Decimal("0")  # dry-run simulations; shown for transparency, never counted against a ceiling

    is_pilot_exhausted: bool
    is_total_exhausted: bool
    
    recent_transactions: List[Dict[str, Any]] = Field(default_factory=list)


class BudgetService:
    """Service enforcing budget limits and managing spend records."""

    def __init__(self, db: Optional[DatabaseManager] = None, policy: Optional[BudgetPolicy] = None):
        self.db = db or DatabaseManager()
        self.policy = policy or settings.budget

    def get_status(self) -> BudgetStatus:
        """Computes current budget status against configured local ceilings."""
        self.db.init_db()
        _, total_spend = self.db.get_cumulative_spend(include_estimates=False)
        _, pilot_spend = self.db.get_pilot_spend(include_estimates=False)
        _, simulated = self.db.get_simulated_spend()
        unresolved = len(self.db.get_unresolved())
        by_basis = self.db.get_spend_by_basis()

        rem_total = max(Decimal("0.00"), self.policy.total_budget_usd - total_spend)
        rem_pilot = max(Decimal("0.00"), self.policy.pilot_budget_usd - pilot_spend)

        return BudgetStatus(
            total_ceiling_usd=self.policy.total_budget_usd,
            cumulative_spend_usd=total_spend,
            remaining_total_usd=rem_total,
            pilot_ceiling_usd=self.policy.pilot_budget_usd,
            pilot_spend_usd=pilot_spend,
            simulated_spend_usd=simulated,
            unresolved_spend_count=unresolved,
            spend_by_basis=by_basis,
            remaining_pilot_usd=rem_pilot,
            per_run_ceiling_usd=self.policy.per_run_ceiling_usd,
            per_case_ceiling_usd=self.policy.per_case_ceiling_usd,
            is_pilot_exhausted=(pilot_spend >= self.policy.pilot_budget_usd),
            is_total_exhausted=(total_spend >= self.policy.total_budget_usd),
            recent_transactions=self.db.get_recent_transactions(limit=10),
        )

    def check_spend_permitted(
        self,
        projected_spend_usd: Decimal,
        is_pilot: bool = True,
    ) -> Tuple[bool, str]:
        """Validates whether a projected expense is permitted under local policies.
        
        Returns:
            (is_permitted, reason_message)
        """
        self.db.init_db()

        if projected_spend_usd < Decimal("0.00"):
            return False, "Projected spend cannot be negative."

        pending = self.db.get_unresolved()
        if pending:
            return False, (f"{len(pending)} request(s) have unresolved spend (cost unknown). Reconcile them with "
                           "`msu-llm-lab budget reconcile` before any further live run.")

        if projected_spend_usd > self.policy.per_run_ceiling_usd:
            return (
                False,
                f"Projected spend (${projected_spend_usd:.4f}) exceeds the per-run ceiling (${self.policy.per_run_ceiling_usd:.4f}).",
            )

        _, total_spend = self.db.get_cumulative_spend(include_estimates=False)
        if total_spend + projected_spend_usd > self.policy.total_budget_usd:
            return (
                False,
                f"Projected spend (${projected_spend_usd:.4f}) would exceed the total budget ceiling (${self.policy.total_budget_usd:.4f}). Current total spend: ${total_spend:.4f}.",
            )

        if is_pilot:
            _, pilot_spend = self.db.get_pilot_spend(include_estimates=False)
            if pilot_spend + projected_spend_usd > self.policy.pilot_budget_usd:
                return (
                    False,
                    f"Projected spend (${projected_spend_usd:.4f}) would exceed the pilot evaluation ceiling (${self.policy.pilot_budget_usd:.4f}). Current pilot spend: ${pilot_spend:.4f}.",
                )

        return True, "Spend is permitted under local experiment policies."

    def check_preflight_budget(
        self,
        estimated_cost_usd: Decimal,
        is_pilot: bool = True,
        model_id: Optional[str] = None,
    ) -> None:
        """Validates that projected evaluation run does not breach ceilings, raising BudgetExceededError on failure."""
        permitted, reason = self.check_spend_permitted(estimated_cost_usd, is_pilot=is_pilot)
        if not permitted:
            raise BudgetExceededError(f"Budget preflight check failed: {reason}")

    def record_spend(
        self,
        transaction_id: str,
        model_id: str,
        provider: str,
        amount_usd: Decimal,
        description: str,
        run_id: Optional[str] = None,
        is_pilot: bool = True,
        is_preflight_estimate: bool = False,
        is_simulated: bool = False,
        cost_basis: str = "UNSPECIFIED",
    ) -> None:
        """Commits an actual, estimated or simulated (dry-run) transaction. Simulated entries never count against ceilings."""
        self.db.init_db()
        self.db.record_transaction(
            transaction_id=transaction_id,
            model_id=model_id,
            provider=provider,
            amount_usd=amount_usd,
            description=description,
            run_id=run_id,
            is_pilot=is_pilot,
            is_preflight_estimate=is_preflight_estimate,
            is_simulated=is_simulated,
            cost_basis=cost_basis,
        )

    def calculate_case_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        input_price_per_1m_usd: Decimal,
        output_price_per_1m_usd: Decimal,
    ) -> Decimal:
        """Calculates exact cost for an individual case execution using Decimal arithmetic."""
        in_cost = (Decimal(prompt_tokens) / Decimal("1000000")) * input_price_per_1m_usd
        out_cost = (Decimal(completion_tokens) / Decimal("1000000")) * output_price_per_1m_usd
        return in_cost + out_cost


    def record_unresolved_spend(self, transaction_id: str, model_id: str, provider: str, description: str, run_id: Optional[str] = None) -> None:
        """Record that a request may have been billed but no cost could be established. No amount is invented (it is stored as 0 and
        flagged); live runs stay blocked until the entry is reconciled with a real amount."""
        self.db.init_db()
        self.db.record_transaction(
            transaction_id=transaction_id, model_id=model_id, provider=provider, amount_usd=Decimal("0"),
            description=f"UNRESOLVED SPEND: {description}", run_id=run_id, is_pilot=True, is_preflight_estimate=False,
            is_unresolved=True, cost_basis="UNRESOLVED",
        )

    def unresolved(self):
        self.db.init_db()
        return self.db.get_unresolved()
