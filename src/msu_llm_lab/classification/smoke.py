"""Synthetic smoke test: plan, typed approval, execution. It checks that the adapters work, NOT which model is better.

Fixtures are hand-written, marked SYNTHETIC, and contain no student text. `intended` labels record the fixture author's
design intent only; they are not ground truth and the report never computes accuracy from them. Nothing here loads the
real CSV. A live run needs an interactive typed confirmation bound to the exact plan (fingerprint).
"""

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from msu_llm_lab.classification import serialization
from msu_llm_lab.classification.chat_adapter import ChatClassificationAdapter
from msu_llm_lab.classification.contracts import HistoryStatus, IntentCase, IntentLabel, LeadCase, Role, Turn
from msu_llm_lab.classification.jev_adapter import JevDecisionsAdapter
from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.classification.runner import ClassificationRunner, estimate_upper_bound, prompt_version
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.models import registry
from msu_llm_lab.providers.openrouter import OpenRouterAdapter
from msu_llm_lab.schemas.evaluation import DataClassification

SMOKE_MODELS = ("jev-1.14", "gpt-4.1-nano")  # the two protocol paths; the full roster stays in the registry
PROMPT_FILES = {Task.INTENT: "main_intent_classifier.md", Task.LEAD: "lead_qualification_prompt.md"}
DEFAULT_HARD_CAP = Decimal("0.05")
SMOKE_DATASET_VERSION = "smoke-synthetic-v1"


@dataclass(frozen=True)
class SmokeFixture:
    case: Union[IntentCase, LeadCase]
    task: Task
    intended: Union[IntentLabel, bool]
    kind: str
    rationale: str
    scored: bool = False  # smoke fixtures are never scored for quality


def _t(i: int, role: Role, text: str, case: str) -> Turn:
    return Turn(turn_index=i, role=role, text=text, source_ref=f"SMOKE-{case}-{i}")


def _intent(cid: str, history: List[Tuple[Role, str]], current: str) -> IntentCase:
    turns = [_t(i, r, tx, cid) for i, (r, tx) in enumerate(history)]
    last_a = next((t for t in reversed(turns) if t.role == Role.ASSISTANT), None)
    return IntentCase(
        case_id=cid, classification=DataClassification.SYNTHETIC, history=turns,
        history_status=HistoryStatus.COMPLETE if turns else HistoryStatus.NONE_AVAILABLE,
        current_message=_t(len(turns), Role.USER, current, cid), previous_assistant_message=last_a,
    )


def _lead(cid: str, summary: List[str]) -> LeadCase:
    return LeadCase(case_id=cid, classification=DataClassification.SYNTHETIC, summary_builder_version="smoke-handwritten",
                    action_summary=summary)


def build_fixtures() -> List[SmokeFixture]:
    """Five hand-written cases per task. Cases that depend on unresolved rules (Lead L4-L8, Intent O1) are deliberately absent."""
    I, L = Task.INTENT, Task.LEAD
    return [
        SmokeFixture(_intent("smoke-i-1", [], "hello"), I, IntentLabel.GREETING, "simple", "plain greeting with no information need"),
        SmokeFixture(_intent("smoke-i-2", [], "What are the tuition fees for the Diploma in Business?"), I, IntentLabel.QUERY, "simple",
                     "genuine information need"),
        SmokeFixture(_intent("smoke-i-3", [], "Tell me a joke"), I, IntentLabel.OUT_OF_SCOPE, "simple", "cannot be answered from a knowledge base"),
        SmokeFixture(_intent("smoke-i-4", [], "I already uploaded my SPM result, can you check if you received it?"), I,
                     IntentLabel.UPLOAD_STATUS_CHECK, "simple", "past submission plus a status request"),
        SmokeFixture(_intent("smoke-i-5", [(Role.USER, "Do you offer scholarships?"),
                                           (Role.ASSISTANT, "Yes, several scholarships exist. Would you like me to connect you to a counsellor?")],
                             "yes please"), I, IntentLabel.HUMAN_HANDOFF_REQUEST, "contextual",
                     "clear affirmative to a specific counsellor-connect offer (confirmed rule R1)"),
        SmokeFixture(_lead("smoke-l-1", ["submitted SPM results", "uploaded IELTS certificate"]), L, True, "simple", "two Tier A actions"),
        SmokeFixture(_lead("smoke-l-2", ["asked about the campus tour video"]), L, False, "simple", "a single Tier B action is not enough"),
        SmokeFixture(_lead("smoke-l-3", ["asked about the campus tour video", "asked about scholarship options"]), L, True, "simple",
                     "two distinct Tier B actions"),
        SmokeFixture(_lead("smoke-l-4", ["asked whether their diploma credits can be transferred"]), L, True, "contextual",
                     "transfer signal auto-qualifies"),
        SmokeFixture(_lead("smoke-l-5", ["asked what time the admissions office is open", "assistant answered", "student replied 'okay thanks'"]),
                     L, False, "contextual", "an acknowledgement alone is never a signal"),
    ]


def load_prompts(prompt_dir: Path) -> Dict[Task, str]:
    out = {}
    for task, name in PROMPT_FILES.items():
        path = Path(prompt_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"Authoritative prompt not found: {path}")
        out[task] = path.read_text(encoding="utf-8")
    return out


def build_smoke_adapters(model_keys: Sequence[str], stage: str = "smoke") -> Dict[str, object]:
    """One protocol-specific adapter per participant that has a model id: Jev uses the Decisions adapter, every chat model the chat adapter."""
    adapters: Dict[str, object] = {}
    for key in model_keys:
        rec = registry.get_record(key)
        mid = rec.requested_model_id.value
        if not mid:
            continue
        if rec.protocol == registry.Protocol.DECISIONS_API:
            adapters[mid] = JevDecisionsAdapter(model_id=mid, registry_stage=stage)
        else:
            adapters[mid] = ChatClassificationAdapter(model_id=mid, provider=OpenRouterAdapter(registry_stage=stage))
    return adapters


@dataclass
class PlanRow:
    model_key: str
    model_id: Optional[str]
    task: Task
    protocol: str
    route: str
    pricing: str
    n_cases: int
    upper_bound_usd: Optional[Decimal]
    blockers: List[str]


@dataclass
class SmokePlan:
    stage: str
    rows: List[PlanRow]
    hard_cap_usd: Decimal
    per_run_ceiling_usd: Decimal
    per_case_ceiling_usd: Decimal
    pilot_ceiling_usd: Decimal
    pilot_spent_usd: Decimal
    prompt_versions: Dict[str, str]
    case_ids: Dict[str, List[str]]
    data_policy_note: str
    blocked_reasons: List[str] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def total_upper_bound_usd(self) -> Decimal:
        return sum((r.upper_bound_usd or Decimal("0") for r in self.rows), Decimal("0"))

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reasons)

    @property
    def model_ids(self) -> List[str]:
        return sorted({r.model_id for r in self.rows if r.model_id})


def _price_text(rec: registry.ModelRecord) -> str:
    i, o = rec.input_price_per_1m_usd, rec.output_price_per_1m_usd
    r = rec.route
    return (f"in ${i.value}/M [{i.status.value}] out ${o.value}/M [{o.status.value}]; route price in [{r.route_input_price_per_1m_usd.status.value}] "
            f"out [{r.route_output_price_per_1m_usd.status.value}]" if i.value is not None and o.value is not None
            else f"in [{i.status.value}] out [{o.status.value}]; route price in [{r.route_input_price_per_1m_usd.status.value}]")


def _route_text(rec: registry.ModelRecord) -> str:
    prefs = rec.route.provider_preferences
    listed = rec.provider_route.value
    return (f"{rec.route.status.value}; listed providers: {listed if listed else 'UNKNOWN'} [{rec.provider_route.status.value}]; "
            f"routing prefs: {json.dumps(prefs, sort_keys=True) if prefs else 'none sent'}")


def build_plan(
    model_keys: Sequence[str], tasks: Sequence[Task], prompts: Dict[Task, str], budget: BudgetService,
    hard_cap_usd: Decimal = DEFAULT_HARD_CAP, fixtures: Optional[List[SmokeFixture]] = None, stage: str = "smoke",
) -> SmokePlan:
    """stage="smoke": the two-model protocol check, lenient on unconfirmed response schemas. stage="pilot": any roster participant,
    strict registry gating (a schema still unconfirmed blocks, so Jev stays blocked until a smoke test confirms it)."""
    fixtures = fixtures if fixtures is not None else build_fixtures()
    adapters = build_smoke_adapters([k for k in model_keys if k in registry.REGISTRY and (stage == "pilot" or k in SMOKE_MODELS)], stage)
    policy = budget.policy
    budget.db.init_db()  # idempotent; a fresh database has no ledger table yet
    _, pilot_spent = budget.db.get_pilot_spend(include_estimates=False)
    blocked: List[str] = []
    rows: List[PlanRow] = []
    for key in model_keys:
        rec = registry.REGISTRY.get(key)
        if rec is None:
            blocked.append(f"{key}: not in the registry")
            continue
        mid = rec.requested_model_id.value
        if stage == "smoke" and key not in SMOKE_MODELS:
            blocked.append(f"{key}: not part of the two-model smoke test; it remains in the roster for the expanded pilot")
            continue
        for task in tasks:
            if task.value not in rec.tasks:  # e.g. a production baseline serves only one task
                continue
            cases = [f.case for f in fixtures if f.task == task]
            blockers = rec.synthetic_blockers(stage)
            adapter = adapters.get(mid)
            prices = registry.model_level_prices(mid)
            ub = None
            if adapter is not None and prices is not None:
                ub, _worst = estimate_upper_bound(task, adapter, cases, prompts[task], prices)
            else:
                blockers = blockers + ["price or adapter unavailable: no upper-bound estimate"]
            rows.append(PlanRow(key, mid, task, rec.protocol.value, _route_text(rec), _price_text(rec), len(cases), ub, blockers))
            for b in blockers:
                blocked.append(f"{key} ({task.value}): {b}")
    total = sum((r.upper_bound_usd or Decimal("0") for r in rows), Decimal("0"))
    pending = budget.unresolved()
    if pending:
        blocked.append(f"{len(pending)} unresolved-spend ledger entr{'y' if len(pending) == 1 else 'ies'}: reconcile with `msu-llm-lab budget reconcile` first")
    if hard_cap_usd <= 0:
        blocked.append("hard cap must be positive")
    if hard_cap_usd > policy.per_run_ceiling_usd:
        blocked.append(f"hard cap ${hard_cap_usd} exceeds the per-run ceiling ${policy.per_run_ceiling_usd}")
    if total > hard_cap_usd:
        blocked.append(f"estimated upper bound ${total:.6f} exceeds the hard cap ${hard_cap_usd}")
    if Decimal(str(pilot_spent)) + hard_cap_usd > policy.pilot_budget_usd:
        blocked.append(f"hard cap plus pilot spend so far (${pilot_spent}) exceeds the pilot ceiling ${policy.pilot_budget_usd}")
    plan = SmokePlan(
        stage=stage, rows=rows, hard_cap_usd=hard_cap_usd, per_run_ceiling_usd=policy.per_run_ceiling_usd, per_case_ceiling_usd=policy.per_case_ceiling_usd,
        pilot_ceiling_usd=policy.pilot_budget_usd, pilot_spent_usd=Decimal(str(pilot_spent)),
        prompt_versions={t.value: prompt_version(p) for t, p in prompts.items() if t in tasks},
        case_ids={t.value: [f.case.case_id for f in fixtures if f.task == t] for t in tasks},
        data_policy_note="Synthetic text and the authoritative prompt texts only. Provider data policies are UNKNOWN, so no student-derived data may be sent.",
        blocked_reasons=blocked,
    )
    canon = {
        "stage": stage,
        "rows": [(r.model_key, r.model_id, r.task.value, r.route, r.n_cases, str(r.upper_bound_usd)) for r in rows],
        "cap": str(hard_cap_usd), "prompts": plan.prompt_versions, "cases": plan.case_ids,
        "states": {f.case.case_id: hashlib.sha256((serialization.serialize_intent_case(f.case) if f.task == Task.INTENT
                                                    else serialization.serialize_lead_input(f.case)).encode()).hexdigest()[:12]
                   for f in fixtures if f.task in tasks},
    }
    plan.fingerprint = hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:12]
    return plan


def render_plan(plan: SmokePlan) -> List[str]:
    title = ("SYNTHETIC SMOKE TEST: PREFLIGHT PLAN" if plan.stage == "smoke" else "SYNTHETIC EXPANDED PILOT: PREFLIGHT PLAN (every eligible participant)")
    out = [f"{title} (schema verification only; no quality or accuracy conclusion can come from it)", ""]
    for r in plan.rows:
        out += [f"- {r.model_key}  [{r.protocol}]  model id: {r.model_id}", f"    task: {r.task.value}   cases: {r.n_cases}",
                f"    route: {r.route}", f"    pricing: {r.pricing}",
                f"    upper-bound spend: " + (f"${r.upper_bound_usd:.6f}" if r.upper_bound_usd is not None else "UNAVAILABLE (blocked)"),
                f"    blockers: {', '.join(r.blockers) if r.blockers else 'none for the ' + plan.stage + ' stage'}"]
    out += ["", f"Total estimated upper bound: ${plan.total_upper_bound_usd:.6f}   (pessimistic: 2 bytes per token, full max output tokens)",
            f"Hard cap for this run: ${plan.hard_cap_usd}   per-run ceiling: ${plan.per_run_ceiling_usd}   per-case ceiling: ${plan.per_case_ceiling_usd}",
            f"Pilot ceiling: ${plan.pilot_ceiling_usd}   pilot spent so far: ${plan.pilot_spent_usd}",
            f"Prompt versions sent: {plan.prompt_versions}", f"Data transmitted: {plan.data_policy_note}",
            f"Case ids: {plan.case_ids}", f"Plan fingerprint: {plan.fingerprint}", ""]
    if plan.blocked_reasons:
        out += ["STATUS: BLOCKED. A live run would be refused (each entry names the exact reason):"] + [f"  * {b}" for b in plan.blocked_reasons]
    else:
        out += ["STATUS: no blockers for the smoke stage. A live run still needs interactive typed approval of this exact plan."]
    return out


def summarize_schema_checks(records) -> List[Dict[str, Any]]:
    """Per (model, task): validity counts and which telemetry was actually available. No accuracy is computed."""
    groups: Dict[Tuple[str, str], List] = {}
    for r in records:
        groups.setdefault((r.model_key, r.task.value), []).append(r)
    rows = []
    for (m, t), rs in sorted(groups.items()):
        rows.append({
            "model": m, "task": t, "n": len(rs),
            "valid": sum(r.status == "OK" for r in rs), "invalid_output": sum(r.status == "INVALID_OUTPUT" for r in rs),
            "errors": sum(r.status == "ERROR" for r in rs),
            "tokens_reported": sum(r.usage_source == "PROVIDER_REPORTED" for r in rs),
            "cost_basis": {b: sum(r.cost_basis == b for r in rs) for b in sorted({r.cost_basis for r in rs})},
            "resolved_model_present": sum(r.resolved_model_id is not None for r in rs),
            "provider_present": sum(r.serving_provider is not None for r in rs),
            "native_confidence_present": sum(r.native_confidence is not None or r.yes_prob is not None for r in rs),
            "model_latency_reported": sum(r.model_latency_ms is not None for r in rs),
            "issues": sorted({i for r in rs for i in r.validation_issues}), "warnings": sorted({w for r in rs for w in r.warnings}),
        })
    return rows


def execute_plan(
    plan: SmokePlan, authorization: LiveAuthorization, adapters: Dict[str, object], store: PredictionStore, budget: BudgetService,
    prompts: Dict[Task, str], run_id: str, fixtures: Optional[List[SmokeFixture]] = None,
):
    """Run the approved plan. Refuses a blocked plan. Stops at the hard cap. Records are written exactly as received."""
    if plan.blocked:
        raise RuntimeError("Refusing to execute a blocked plan: " + "; ".join(plan.blocked_reasons))
    fixtures = fixtures if fixtures is not None else build_fixtures()
    runner = ClassificationRunner(adapters, store, budget, stage=plan.stage)
    remaining = plan.hard_cap_usd
    summaries = []
    for row in plan.rows:
        cases = [f.case for f in fixtures if f.task == row.task]
        s = runner.run(task=row.task, cases=cases, model_id=row.model_id, prompt_text=prompts[row.task], authorization=authorization,
                       run_id=f"{run_id}-{row.model_key}-{row.task.value}", dataset_version=SMOKE_DATASET_VERSION, hard_cap_usd=remaining)
        remaining -= s.recorded_cost_usd
        summaries.append(s)
        if s.aborted_reason:
            break
    return summaries
