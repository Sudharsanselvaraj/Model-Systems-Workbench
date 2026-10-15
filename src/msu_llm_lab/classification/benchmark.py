"""Real-data benchmark: plan, deterministic checkpointed schedule and resumable executor.

Nothing here sends a request while planning. The plan is a pure function of the prepared dataset version, the prompts, the registry and the
configured ceilings, and its fingerprint changes if any of them changes. Execution needs (1) the existing typed live approval, (2) the
typed real-data authorization bound to the same fingerprint, and re-derives the plan before sending anything.

Scheduling: eligible cases of each task are put in a deterministic pseudo-random order (SHA-256 of seed+case id) and cut into fixed-size
blocks. A block runs EVERY approved participant on the SAME cases, so models stay comparable. A block is dispatched only while the
recorded guard cost plus that block's pessimistic upper bound still fits the hard cap and the ledger ceilings; the ceilings are never raised.
Results of a run that stops early are PARTIAL and say how many cases they cover.
"""

import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from msu_llm_lab.classification import realdata, serialization
from msu_llm_lab.classification.contracts import IntentCase, LeadCase
from msu_llm_lab.classification.records import PredictionStore, Task
from msu_llm_lab.classification.runner import ClassificationRunner, MAX_STATE_BYTES, prompt_version
from msu_llm_lab.classification.smoke import PROMPT_FILES, build_smoke_adapters, load_prompts
from msu_llm_lab.core.budget import BudgetExceededError, BudgetService
from msu_llm_lab.core.live_guard import LiveAuthorization
from msu_llm_lab.models import registry
from msu_llm_lab.observability.events import EventStore
from msu_llm_lab.providers.base import LiveNotAuthorizedError, ProviderError, UsageUnavailableError

PLAN_SCHEMA = "real-benchmark-plan-v1"
EVAL_CONTRACT = "classification-benchmark-contract-v1"
ORDER_SEED = "msu-bench-order-v1"
DEFAULT_BLOCK = 10
STAGE = "pilot"
TIMEOUT_S = 30.0
MAX_ATTEMPTS = 3
MAX_CONSECUTIVE_ERRORS = 3
Case = Union[IntentCase, LeadCase]


class PlanError(RuntimeError):
    pass


# ------------------------------------------------------------------ dataset loading (local files only)
@dataclass
class LoadedCases:
    prep_dir: Path
    policy: str
    manifest: Dict[str, Any]
    manifest_sha256: str
    intent: List[IntentCase] = field(default_factory=list)
    lead: List[LeadCase] = field(default_factory=list)
    excluded: Dict[str, Counter] = field(default_factory=dict)      # task -> reason -> count
    candidates: Dict[str, int] = field(default_factory=dict)
    strata: Dict[str, Dict[str, tuple]] = field(default_factory=dict)   # task -> case id -> heuristic stratum (metadata, NOT a label)

    def for_task(self, task: Task) -> List[Case]:
        return self.intent if task == Task.INTENT else self.lead


def load_prep_cases(prep_dir: Path, policy: str) -> LoadedCases:
    """Load the prepared cases of one session policy. Excluded candidates are counted by reason and never returned."""
    prep_dir = Path(prep_dir)
    mpath = prep_dir / "manifest.json"
    if policy not in ("2h", "30m"):
        raise PlanError("session policy must be '2h' or '30m' (the choice is the operator's; neither is assumed)")
    if not mpath.exists():
        raise PlanError(f"no preparation manifest at {mpath}")
    out = LoadedCases(prep_dir, policy, json.loads(mpath.read_text()), realdata.file_sha256(mpath))
    for task, fname, cls, bucket in ((Task.INTENT, f"intent_candidates_{policy}.jsonl", IntentCase, out.intent),
                                     (Task.LEAD, f"lead_candidates_{policy}.jsonl", LeadCase, out.lead)):
        path = prep_dir / fname
        if not path.exists():
            raise PlanError(f"missing candidate file {path}")
        exc: Counter = Counter()
        n = 0
        out.strata[task.value] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            case = cls.model_validate(rec["case"])
            if case.exclusion_reason is not None:
                exc[case.exclusion_reason.value] += 1
                continue
            if _state_bytes(task, case) > MAX_STATE_BYTES:
                exc["STATE_EXCEEDS_SIZE_LIMIT"] += 1
                continue
            bucket.append(case)
            out.strata[task.value][case.case_id] = _stratum(task, rec["candidate"], case)
        out.excluded[task.value] = exc
        out.candidates[task.value] = n
    return out


def _state_text(task: Task, case: Case) -> str:
    return serialization.serialize_intent_case(case) if task == Task.INTENT else serialization.serialize_lead_input(case)


def _state_bytes(task: Task, case: Case) -> int:
    return len(_state_text(task, case).encode("utf-8"))


def _stratum(task: Task, cand: Dict[str, Any], case: Case) -> tuple:
    """Heuristic pattern bucket from preparation metadata. It describes the kind of conversation, never the correct answer."""
    view = str(cand.get("view"))
    if task == Task.INTENT:
        return (view, case.history_status.value, bool(cand.get("selection_hints")))
    tally = cand.get("builder_tally_not_a_label") or {}
    tier = "A" if (tally.get("tier_a_actions") or 0) > 0 else ("B2+" if (tally.get("tier_b_distinct_types") or 0) >= 2 else ("B1" if (tally.get("tier_b_distinct_types") or 0) == 1 else "none"))
    return (view, tier, (cand.get("ambiguity_count") or 0) > 0)


def order_key(case_id: str) -> str:
    return hashlib.sha256(f"{ORDER_SEED}|{case_id}".encode("utf-8")).hexdigest()


def ordered(cases: Sequence[Case]) -> List[Case]:
    return sorted(cases, key=lambda c: order_key(c.case_id))


SAMPLINGS = ("stratified", "hash")


def order_cases(loaded: "LoadedCases", task: Task, sampling: str = "stratified") -> List[Case]:
    """Deterministic case order. `hash`: SHA-256 order. `stratified`: the first case of EVERY pattern bucket (variety first), then everyone else in hash
    order (proportional). Buckets come from preparation metadata (view, history completeness, selection hints, builder tier, ambiguity): they are heuristics, not labels."""
    if sampling not in SAMPLINGS:
        raise PlanError(f"sampling must be one of {SAMPLINGS}")
    base = ordered(loaded.for_task(task))
    if sampling == "hash":
        return base
    seen, first, rest = set(), [], []
    strata = loaded.strata.get(task.value, {})
    for c in base:
        st = strata.get(c.case_id)
        if st not in seen:
            seen.add(st)
            first.append(c)
        else:
            rest.append(c)
    return first + rest


def prompt_manifest(prompts: Dict[Task, str]) -> Dict[str, Dict[str, Any]]:
    return {t.value: {"file": PROMPT_FILES[t], "sha256": hashlib.sha256(p.encode("utf-8")).hexdigest(), "version": prompt_version(p), "bytes": len(p.encode("utf-8"))}
            for t, p in prompts.items()}


# ------------------------------------------------------------------ plan
def _case_bound(task: Task, state_bytes: int, prompt_bytes: int, prices: Tuple[Decimal, Decimal], max_out: int) -> Decimal:
    toks = math.ceil((prompt_bytes + state_bytes) / 2)          # pessimistic: 2 bytes per token (same rule as the runner's preflight)
    return (Decimal(toks) * prices[0] + Decimal(max_out) * prices[1]) / Decimal("1000000")


def _q(d: Decimal) -> str:
    return f"{d:.6f}"


def build_plan(
    loaded: LoadedCases, prompts: Dict[Task, str], budget: BudgetService, *, model_keys: Optional[Sequence[str]] = None, block_size: int = DEFAULT_BLOCK,
    hard_cap_usd: Optional[Decimal] = None, sampling: str = "stratified", parent: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    reports_dir: Optional[Path] = None, reserve_usd: Optional[Decimal] = None,
) -> Dict[str, Any]:
    """Pure planning. Returns a JSON-serialisable plan including the schedule and a fingerprint. Never contacts a provider."""
    keys = list(model_keys) if model_keys is not None else list(registry.REGISTRY)
    unknown = [k for k in keys if k not in registry.REGISTRY]
    if unknown:
        raise PlanError(f"unknown participant(s): {', '.join(unknown)}")
    pol = budget.policy
    status = budget.get_status()
    blockers: List[str] = []
    ps = loaded.manifest.get("pseudonymization", {})
    if not ps.get("protected_secret_used"):
        blockers.append("dataset was prepared with the public legacy salt: prepare a new version with a private secret (--prompt-salt)")
    if not loaded.manifest.get("run", {}).get("code_fingerprint"):
        blockers.append("dataset predates the hardened preparation (no code fingerprint): prepare a new version")
    cap = min(hard_cap_usd if hard_cap_usd is not None else pol.pilot_budget_usd, pol.pilot_budget_usd, pol.total_budget_usd)
    headroom = min(status.remaining_total_usd, status.remaining_pilot_usd)
    reserve = reserve_usd if reserve_usd is not None else Decimal("0")
    if reserve_usd is not None:
        cap = min(cap, max(Decimal("0"), headroom - reserve))      # the reserve for retries and unresolved costs is never planned away
    adapters = build_smoke_adapters(keys, STAGE)
    pbytes = {t: len(prompts[t].encode("utf-8")) for t in prompts}
    order = {t: order_cases(loaded, t, sampling) for t in (Task.INTENT, Task.LEAD)}
    sbytes = {t: {c.case_id: _state_bytes(t, c) for c in order[t]} for t in order}
    n_blocks = max(math.ceil(len(order[t]) / block_size) for t in order) if any(order.values()) else 0

    participants: List[Dict[str, Any]] = []
    unit_bound: Dict[Tuple[int, str, str], Tuple[int, Decimal]] = {}   # (block, model_key, task) -> (requests, bound)
    for key in keys:
        rec = registry.get_record(key)
        mid = rec.requested_model_id.value
        adapter = adapters.get(mid)
        prices = registry.model_level_prices(mid)
        stage_blockers = registry.synthetic_live_blockers(mid, STAGE)
        sd_blockers = [b for b in rec.student_data_blockers() if b != "data_policy"]
        pinned = rec.pinned_route()
        per_task: Dict[str, Any] = {}
        for tname in rec.tasks:
            task = Task(tname)
            elig, over_cap = [], []
            for c in order[task]:
                if prices is None or adapter is None:
                    over_cap.append(c.case_id)
                    continue
                b = _case_bound(task, sbytes[task][c.case_id], pbytes[task], prices, adapter.max_output_tokens(task))
                (over_cap if b > pol.per_case_ceiling_usd else elig).append(c.case_id)
            per_task[tname] = {"eligible_cases": len(elig), "excluded_per_case_cap": len(over_cap), "excluded_per_case_cap_ids_sha256": hashlib.sha256("|".join(sorted(over_cap)).encode()).hexdigest()[:12]}
            elig_set = set(elig)
            for k in range(n_blocks):
                blk = [c for c in order[task][k * block_size:(k + 1) * block_size] if c.case_id in elig_set]
                if not blk:
                    continue
                total = sum((_case_bound(task, sbytes[task][c.case_id], pbytes[task], prices, adapter.max_output_tokens(task)) for c in blk), Decimal("0"))
                if total > pol.per_run_ceiling_usd:
                    blockers.append(f"{key} {tname} block {k}: bound {_q(total)} exceeds the per-run ceiling {pol.per_run_ceiling_usd}")
                unit_bound[(k, key, tname)] = (len(blk), total)
            per_task[tname]["requests_full"] = len(elig)
            per_task[tname]["upper_bound_full_usd"] = _q(sum((unit_bound[(k, key, tname)][1] for k in range(n_blocks) if (k, key, tname) in unit_bound), Decimal("0")))
        if stage_blockers:
            blockers.append(f"{key}: registry blockers {', '.join(stage_blockers)}")
        if sd_blockers:
            blockers.append(f"{key}: student-data blockers other than the data policy: {', '.join(sd_blockers)}")
        if prices is None:
            blockers.append(f"{key}: no usable price, budget preflight impossible")
        participants.append({
            "key": key, "model_id": mid, "display_name": rec.display_name, "roles": rec.roles, "protocol": rec.protocol.value, "tasks": list(rec.tasks),
            "route": {"pinned_tag": rec.pinned_tag, "status": rec.route.status.value, "provider_preferences": pinned and rec.route.provider_preferences,
                      "listed_provider": (pinned.provider if pinned else str(rec.provider_route.value)), "fallbacks": "disabled" if rec.route.provider_preferences and rec.route.provider_preferences.get("allow_fallbacks") is False else "n/a or provider-managed"},
            "production_reported_name": rec.production_reported_name, "benchmark_route_only": bool(rec.production_reported_name),
            "pricing_usd_per_1m": {"input": str(prices[0]), "output": str(prices[1])} if prices else None,
            "price_status": {"input": rec.input_price_per_1m_usd.status.value, "output": rec.output_price_per_1m_usd.status.value},
            "data_handling": {"data_policy": rec.data_policy.status.value, "student_data_blockers_beyond_policy": sd_blockers,
                              "note": "UNKNOWN means no provider retention, training or processing facts were verified from a primary source"},
            "readiness": {"schema_facts": {"response_shape": rec.response_shape.status.value, "usage_fields": rec.usage_fields.status.value}, "blockers": stage_blockers},
            "native_fields": _native_fields(rec), "request_overrides": dict(rec.request_overrides),
            "per_task": per_task,
        })
    # carry-over: a participant whose entry is IDENTICAL to the parent's (same route, request, prices, task scope) and whose stored results contain no
    # ERROR record keeps its completed results; nothing is re-sent for it. Anything else is re-run fresh under this plan.
    inherits: Optional[Dict[str, Any]] = None
    if parent is not None:
        pplan, pstate = parent
        same_setup = (pplan["dataset"]["version"] == plan_dataset(loaded)["version"] and pplan["dataset"]["session_policy"] == loaded.policy
                      and pplan["dataset"]["manifest_sha256"] == loaded.manifest_sha256 and pplan["prompts"] == prompt_manifest(prompts)
                      and pplan["ordering"]["sampling"] == sampling and pplan["ordering"]["block_size"] == block_size and pplan["ordering"]["seed"] == ORDER_SEED
                      and pplan["evaluation_contract_version"] == EVAL_CONTRACT)
        done_blocks = pstate.get("blocks_completed", 0)
        inherited_keys: List[str] = []
        not_carried: Dict[str, Any] = {}
        rd = Path(reports_dir) if reports_dir is not None else None
        if same_setup and rd is not None:
            ppart = {x["key"]: x for x in pplan["participants"]}
            for part in participants:
                old = ppart.get(part["key"])
                if old is None:
                    continue
                root_bid = old.get("store_bench") or pplan["bench_id"]
                stored = []
                for tname in part["tasks"]:
                    path = prediction_path(rd, root_bid, part["key"], tname)
                    if path.exists():
                        stored += PredictionStore(path).load()
                clean = not any(r.status == "ERROR" for r in stored)
                why = None
                if part["model_id"] in pstate.get("quarantined", {}):
                    why = f"quarantined in the parent run: {pstate['quarantined'][part['model_id']]}"
                elif _identity(old) != _identity(part):
                    why = "route, request fields, prices or task scope differ from the parent's entry"
                elif not clean:
                    why = "its stored results contain request errors, so they cannot be resumed partially"
                if why:
                    not_carried[part["key"]] = {"reason": why, "completed_requests_re_sent": sum(1 for r in stored if r.status != "ERROR"), "failed_requests_retried": sum(1 for r in stored if r.status == "ERROR")}
                    continue
                if clean:
                    part["store_bench"] = root_bid
                    part["inherited_completed_blocks"] = done_blocks
                    inherited_keys.append(part["key"])
        if inherited_keys or not_carried:
            inherits = {"parent_bench": pplan["bench_id"], "parent_fingerprint": pplan["fingerprint"], "parent_blocks_completed": done_blocks, "participants": inherited_keys,
                        "not_carried_over": not_carried,
                        "rule": "identical participant entry and no ERROR record in its stored results; its completed requests are not re-sent. Anything else runs fresh and its earlier results stay in the parent run"}
            for (k, key, tname) in list(unit_bound):
                if key in inherited_keys and k < done_blocks:
                    unit_bound[(k, key, tname)] = (0, Decimal("0"))
    # schedule: blocks of identical cases for every participant
    blocks = []
    cum_bound, cum_req = Decimal("0"), 0
    guaranteed = 0
    budget_room = min(cap, status.remaining_pilot_usd, status.remaining_total_usd)
    for k in range(n_blocks):
        reqs = sum(v[0] for (kk, _, _), v in unit_bound.items() if kk == k)
        bound = sum((v[1] for (kk, _, _), v in unit_bound.items() if kk == k), Decimal("0"))
        n_i = len(order[Task.INTENT][k * block_size:(k + 1) * block_size])
        n_l = len(order[Task.LEAD][k * block_size:(k + 1) * block_size])
        if cum_bound + bound <= budget_room and guaranteed == k:
            guaranteed = k + 1
        cum_bound += bound
        cum_req += reqs
        blocks.append({"index": k, "intent_cases": n_i, "lead_cases": n_l, "requests": reqs, "upper_bound_usd": _q(bound)})
    # sizing: what the remaining budget buys. Worst-case uses the pessimistic bounds; expected uses the mean booked cost and latency actually observed for the same
    # participant and task in stored results (never invented: with no observation the bound is used).
    observed: Dict[Tuple[str, str], Tuple[Decimal, float, int]] = {}
    if parent is not None and reports_dir is not None:
        for part in participants:
            roots = [part.get("store_bench"), parent[0]["bench_id"]]
            for tname in part["tasks"]:
                recs = []
                for root in dict.fromkeys(r for r in roots if r):
                    path = prediction_path(Path(reports_dir), root, part["key"], tname)
                    if path.exists():
                        recs += [r for r in PredictionStore(path).load() if r.recorded_cost_usd is not None and r.status != "ERROR"]
                if recs:
                    lat = [r.latency_ms for r in recs if r.latency_ms is not None]
                    observed[(part["key"], tname)] = (sum((r.recorded_cost_usd for r in recs), Decimal("0")) / len(recs), (sum(lat) / len(lat)) if lat else 0.0, len(recs))
    cum_b, cum_e, cum_r, cum_s = Decimal("0"), Decimal("0"), 0, 0.0
    sizing_rows: List[Dict[str, Any]] = []
    fits_worst = fits_expected = 0
    for k in range(n_blocks):
        for (kk, key, tname), (nreq, bound) in unit_bound.items():
            if kk != k or nreq == 0:
                continue
            mean = observed.get((key, tname))
            cum_e += (mean[0] * nreq) if mean else bound
            cum_s += (mean[1] / 1000.0 * nreq) if mean else 0.0
        cum_b += Decimal(blocks[k]["upper_bound_usd"])
        cum_r += blocks[k]["requests"]
        if cum_b <= budget_room:
            fits_worst = k + 1
        if cum_e <= budget_room:
            fits_expected = k + 1
        if (k + 1) in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100):
            sizing_rows.append({"blocks": k + 1, "cases_per_task": (k + 1) * block_size, "new_requests": cum_r, "worst_case_bound_usd": _q(cum_b), "expected_cost_usd": _q(cum_e),
                                "expected_minutes_sequential": round(cum_s / 60.0, 1), "fits_worst_case": cum_b <= budget_room, "fits_expected": cum_e <= budget_room})
    sizing = {"ledger_remaining_total_usd": str(status.remaining_total_usd), "ledger_remaining_pilot_usd": str(status.remaining_pilot_usd), "reserve_usd": _q(reserve) if reserve_usd is not None else None,
              "spendable_usd": _q(budget_room), "largest_blocks_fitting_worst_case": fits_worst, "largest_blocks_fitting_expected_cost": fits_expected,
              "cases_per_task_worst_case": fits_worst * block_size, "cases_per_task_expected": fits_expected * block_size, "table": sizing_rows,
              "observed_basis": {f"{k[0]}|{k[1]}": {"mean_booked_usd_per_request": str(v[0]), "mean_latency_ms": round(v[1]), "n": v[2]} for k, v in sorted(observed.items())},
              "note": "Worst case assumes every request costs its pessimistic bound. Expected uses observed mean booked (estimated) USD per request; it is a forecast, not a guarantee. "
                      "Dispatch stops by the recorded guard cost plus the next block's bound, so a run continues while real spend stays low and halts before the spendable amount is exceeded."}
    total_req_full = sum(p["per_task"][t]["requests_full"] for p in participants for t in p["per_task"])
    total_bound_full = sum((Decimal(p["per_task"][t]["upper_bound_full_usd"]) for p in participants for t in p["per_task"]), Decimal("0"))
    g_req = sum(b["requests"] for b in blocks[:guaranteed])
    g_bound = sum((Decimal(b["upper_bound_usd"]) for b in blocks[:guaranteed]), Decimal("0"))
    recipients = {p["model_id"]: f"OpenRouter, routed to {p['route']['listed_provider']}" + (f" (pinned: {p['route']['pinned_tag']})" if p["route"]["pinned_tag"] else "") for p in participants}
    plan: Dict[str, Any] = {
        "schema": PLAN_SCHEMA, "evaluation_contract_version": EVAL_CONTRACT,
        "dataset": {"version": loaded.manifest.get("prep_version"), "session_policy": loaded.policy, "manifest_sha256": loaded.manifest_sha256,
                    "source_csv_sha256": loaded.manifest.get("source", {}).get("sha256"), "parity_status": loaded.manifest.get("parity_status"),
                    "pseudonymization": loaded.manifest.get("pseudonymization", {}), "privacy_notice": loaded.manifest.get("privacy_notice"),
                    "candidates": loaded.candidates, "eligible": {t.value: len(loaded.for_task(t)) for t in (Task.INTENT, Task.LEAD)},
                    "excluded": {t: dict(sorted(c.items())) for t, c in loaded.excluded.items()}, "labels": "ALL PENDING: accuracy metrics are not computed"},
        "prompts": prompt_manifest(prompts),
        "ordering": {"method": ("stratified: first case of every pattern bucket, then SHA-256(seed|case_id) order" if sampling == "stratified" else "SHA-256(seed|case_id) ascending, per task"),
                     "sampling": sampling, "seed": ORDER_SEED, "block_size": block_size, "n_blocks": n_blocks,
                     "strata": {t.value: len(set(loaded.strata.get(t.value, {}).values())) for t in (Task.INTENT, Task.LEAD)},
                     "note": "pattern buckets are preparation metadata (view, history completeness, selection hints, builder tier, ambiguity), NOT labels; every block runs all approved participants on identical cases"},
        "participants": participants,
        "limits": {"per_case_usd": str(pol.per_case_ceiling_usd), "per_run_usd": str(pol.per_run_ceiling_usd), "pilot_usd": str(pol.pilot_budget_usd),
                   "total_usd": str(pol.total_budget_usd), "hard_cap_usd": _q(cap), "max_state_bytes": MAX_STATE_BYTES, "timeout_seconds": TIMEOUT_S,
                   "max_attempts": MAX_ATTEMPTS, "reserve_usd": _q(reserve) if reserve_usd is not None else None, "max_output_tokens": {"INTENT": 16, "LEAD": 24, "Jev": "typed answers (output tokens documented as free)"},
                   "token_estimate": "pessimistic: 2 bytes per token, full max output tokens"},
        "totals": {"requests_full_dataset": total_req_full, "upper_bound_full_dataset_usd": _q(total_bound_full), "full_dataset_fits_caps": total_bound_full <= budget_room,
                   "guaranteed_blocks": guaranteed, "guaranteed_requests": g_req, "guaranteed_upper_bound_usd": _q(g_bound),
                   "dispatch_rule": "a block is sent only while recorded guard cost + that block's upper bound <= the hard cap AND the ledger pilot/total ceilings; caps are never raised",
                   "coverage_if_only_guaranteed": f"{min(guaranteed * block_size, len(order[Task.INTENT]))} of {len(order[Task.INTENT])} INTENT and "
                                                  f"{min(guaranteed * block_size, len(order[Task.LEAD]))} of {len(order[Task.LEAD])} LEAD eligible cases"},
        "blocks": blocks, "recipients": recipients, "inherits": inherits, "sizing": sizing,
        "blocks_note": "requests and upper bounds count NEW requests only (carried-over results are excluded)" if inherits else "requests and upper bounds per block",
        "data_categories": realdata.DATA_CATEGORIES, "never_sent": realdata.NOT_TRANSMITTED,
        "blockers": sorted(blockers),
    }
    plan["fingerprint"] = fingerprint(plan)
    plan["bench_id"] = f"bench-{plan['fingerprint']}"
    plan["informational"] = {"created_at": datetime.now(timezone.utc).isoformat(), "ledger_at_plan_time": {
        "pilot_spent_usd": str(status.pilot_spend_usd), "total_spent_usd": str(status.cumulative_spend_usd), "unresolved_entries": status.unresolved_spend_count}}
    if status.unresolved_spend_count:
        plan["blockers"].append(f"{status.unresolved_spend_count} unresolved spend entr(y/ies) on the ledger (run `budget reconcile`)")
    return plan


def _identity(part: Dict[str, Any]) -> Dict[str, Any]:
    """What makes two plan entries the same participant setup: model, protocol, tasks, route, prices, approved request fields and per-task eligible cases."""
    return {"model_id": part["model_id"], "protocol": part["protocol"], "tasks": part["tasks"], "route": part["route"], "pricing": part["pricing_usd_per_1m"],
            "overrides": part.get("request_overrides") or {}, "eligible": {t: v["eligible_cases"] for t, v in part["per_task"].items()},
            "excluded_per_case_cap": {t: v["excluded_per_case_cap"] for t, v in part["per_task"].items()}}


def plan_dataset(loaded: "LoadedCases") -> Dict[str, Any]:
    return {"version": loaded.manifest.get("prep_version")}


def _native_fields(rec) -> Dict[str, str]:
    if rec.protocol == registry.Protocol.DECISIONS_API:
        return {"tokens": "provider-reported (LIVE_OBSERVED)", "native_cost": "credits; USD ratio unverified, never booked as USD",
                "confidence": "INTENT: native distribution + separate confidence; LEAD: P(yes) only", "model_latency": "not reported in stored evidence"}
    return {"tokens": "provider-reported usage", "native_cost": "credits when present; USD ratio unverified", "confidence": "none (plain-text label or one-key JSON)",
            "model_latency": "not documented; client end-to-end latency only"}


def fingerprint(plan: Dict[str, Any]) -> str:
    body = {k: v for k, v in plan.items() if k not in ("fingerprint", "bench_id", "informational")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def render_plan(plan: Dict[str, Any]) -> List[str]:
    d, lim, t = plan["dataset"], plan["limits"], plan["totals"]
    out = ["REAL-DATA BENCHMARK: PREFLIGHT PLAN (plan only; nothing has been sent)", "",
           f"dataset: {d['version']}  session policy: {d['session_policy']}  parity: {d['parity_status']}",
           f"source csv sha256: {str(d['source_csv_sha256'])[:16]}...  manifest sha256: {d['manifest_sha256'][:16]}...",
           f"pseudonymization: {d.get('pseudonymization', {}).get('salt_id', 'unknown')} (protected secret: {d.get('pseudonymization', {}).get('protected_secret_used')})  labels: {d['labels']}",
           f"candidates: {d['candidates']}  eligible: {d['eligible']}  excluded: {d['excluded']}",
           "prompts: " + "; ".join(f"{k} {v['version']} (sha256 {v['sha256'][:16]}...)" for k, v in plan["prompts"].items()),
           f"evaluation contract: {plan['evaluation_contract_version']}   ordering: {plan['ordering']['method']}, blocks of {plan['ordering']['block_size']} ({plan['ordering']['n_blocks']} blocks)", ""]
    for p in plan["participants"]:
        out.append(f"- {p['key']}  [{p['protocol']}]  {p['model_id']}  roles {','.join(p['roles'])}" + ("  (benchmark route only; parity not established)" if p["benchmark_route_only"] else ""))
        out.append(f"    route: pinned={p['route']['pinned_tag']} status={p['route']['status']} listed provider={p['route']['listed_provider']} fallbacks={p['route']['fallbacks']}")
        out.append(f"    price per 1M (in/out): {p['pricing_usd_per_1m']}  status {p['price_status']}   schema facts: {p['readiness']['schema_facts']}")
        if p.get("request_overrides"):
            out.append(f"    owner-approved request fields: {p['request_overrides']}")
        out.append(f"    data handling: policy {p['data_handling']['data_policy']} (unknown = nothing verified); native fields: {p['native_fields']}")
        for tn, v in p["per_task"].items():
            out.append(f"    {tn}: eligible {v['eligible_cases']}  excluded by per-case cap {v['excluded_per_case_cap']}  full requests {v['requests_full']}  full upper bound ${v['upper_bound_full_usd']}")
        if p["readiness"]["blockers"]:
            out.append(f"    BLOCKERS: {p['readiness']['blockers']}")
    if plan.get("inherits"):
        i = plan["inherits"]
        out += ["", f"CARRY-OVER from {i['parent_bench']} ({i['parent_blocks_completed']} blocks done): {', '.join(i['participants'])} keep their completed results and are NOT re-sent; "
                    f"every other participant runs fresh. Block bounds below count new requests only."]
    fresh = [(p["key"], tn) for p in plan["participants"] if not p.get("store_bench") for tn in p["tasks"]]
    allc = [(p["key"], tn) for p in plan["participants"] for tn in p["tasks"]]
    cum_r, cum_b, rows = 0, Decimal("0"), []
    for b in plan["blocks"][:6]:
        cum_r += b["requests"]
        cum_b += Decimal(b["upper_bound_usd"])
        rows.append(f"  --max-blocks {b['index'] + 1}: block {b['index']} adds {b['requests']} new requests; cumulative new {cum_r}, upper bound ${cum_b:.6f}")
    out += ["", f"MODEL-TASK MATRIX: {len(allc)} combinations across {len(plan['participants'])} participants; {len(fresh)} run fresh, {len(allc) - len(fresh)} are carried over."
                f" Each block gives every fresh combination {plan['ordering']['block_size']} cases (a block's requests = fresh combinations x {plan['ordering']['block_size']}, or all combinations once its blocks pass the carry-over point).",
            "NEW requests by --max-blocks (cumulative from the start of the plan):"] + rows + [""]
    if plan.get("inherits") and plan["inherits"].get("not_carried_over"):
        out += ["NOT carried over (run fresh; earlier results stay in the parent run):"] + [
            f"  {k}: {v['reason']}; completed requests re-sent {v['completed_requests_re_sent']}, failed requests retried {v['failed_requests_retried']}" for k, v in plan["inherits"]["not_carried_over"].items()]
    if plan.get("sizing"):
        z = plan["sizing"]
        out += ["", f"SIZING: ledger remaining ${z['ledger_remaining_total_usd']} total / ${z['ledger_remaining_pilot_usd']} pilot; reserve ${z['reserve_usd']}; spendable by this plan ${z['spendable_usd']}",
                f"  worst case (pessimistic bounds): up to {z['largest_blocks_fitting_worst_case']} blocks = {z['cases_per_task_worst_case']} cases per task",
                f"  expected (observed mean booked cost): up to {z['largest_blocks_fitting_expected_cost']} blocks = {z['cases_per_task_expected']} cases per task",
                "  blocks | cases/task | new requests | worst-case bound | expected cost | expected minutes | fits worst | fits expected"]
        out += [f"  {r['blocks']:>6} | {r['cases_per_task']:>10} | {r['new_requests']:>12} | ${r['worst_case_bound_usd']:>15} | ${r['expected_cost_usd']:>12} | {r['expected_minutes_sequential']:>16} | {str(r['fits_worst_case']):>10} | {r['fits_expected']}" for r in z["table"]]
        out += [f"  {z['note']}", ""]
    out += [f"FULL DATASET: {t['requests_full_dataset']} requests, upper bound ${t['upper_bound_full_dataset_usd']} (pessimistic); fits caps: {t['full_dataset_fits_caps']}",
            f"caps: per case ${lim['per_case_usd']}  per run ${lim['per_run_usd']}  pilot ${lim['pilot_usd']}  total ${lim['total_usd']}  hard cap this benchmark ${lim['hard_cap_usd']}",
            f"timeout {lim['timeout_seconds']}s, at most {lim['max_attempts']} attempts per request (bounded retry), state limit {lim['max_state_bytes']} bytes, output tokens {lim['max_output_tokens']}",
            f"GUARANTEED WITHIN CAPS: {t['guaranteed_blocks']} blocks = {t['guaranteed_requests']} requests, upper bound ${t['guaranteed_upper_bound_usd']}; coverage {t['coverage_if_only_guaranteed']}",
            f"dispatch rule: {t['dispatch_rule']}",
            "A run that stops early is PARTIAL and will be reported as such; it is not a full benchmark.", "",
            "recipients: " + "; ".join(f"{m} -> {r}" for m, r in plan["recipients"].items()),
            "data sent: " + " | ".join(f"{k}: {'; '.join(v)}" for k, v in plan["data_categories"].items()),
            "never sent: " + "; ".join(plan["never_sent"]),
            f"plan fingerprint: {plan['fingerprint']}   bench id: {plan['bench_id']}"]
    out.append("STATUS: " + ("BLOCKED: " + "; ".join(plan["blockers"]) if plan["blockers"] else "no blockers; a live run still needs typed approval of this exact plan"))
    return out


# ------------------------------------------------------------------ persistence
def bench_dir(root: Path, bench_id: str) -> Path:
    return Path(root) / bench_id


def save_plan(root: Path, plan: Dict[str, Any]) -> Path:
    path = bench_dir(root, plan["bench_id"]) / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("fingerprint") != plan["fingerprint"]:
            raise PlanError("a different plan is already stored under this bench id")
        return path           # same plan: keep the original (its creation time is part of the audit trail)
    realdata.write_json_atomic(path, plan)
    return path


def load_state(root: Path, bench_id: str) -> Dict[str, Any]:
    p = bench_dir(root, bench_id) / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"bench_id": bench_id, "units": {}, "quarantined": {}, "halted": None, "blocks_completed": 0}


def save_state(root: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    realdata.write_json_atomic(bench_dir(root, state["bench_id"]) / "state.json", state)


def write_manifest(root: Path, plan: Dict[str, Any], auth: realdata.RealDataAuthorization, live: LiveAuthorization, code_fp: Optional[str]) -> Path:
    """Versioned run manifest: authorization, hashes, caps and code version. Resumes append to `sessions`; nothing is overwritten."""
    path = bench_dir(root, plan["bench_id"]) / "manifest.json"
    session = {"started_at": datetime.now(timezone.utc).isoformat(), "authorization": auth.model_dump(), "live_models": list(live.models), "python": sys.version.split()[0]}
    if path.exists():
        m = json.loads(path.read_text())
        m["sessions"].append(session)
    else:
        m = {"manifest_version": "real-benchmark-manifest-v1", "bench_id": plan["bench_id"], "plan_fingerprint": plan["fingerprint"],
             "dataset": plan["dataset"], "prompts": plan["prompts"], "limits": plan["limits"], "evaluation_contract_version": plan["evaluation_contract_version"],
             "participants": [{k: p[k] for k in ("key", "model_id", "protocol", "tasks", "route", "data_handling")} for p in plan["participants"]],
             "code_fingerprint": code_fp, "inherits": plan.get("inherits"), "sessions": [session],
             "provider_policy_caveat": realdata.POLICY_NOTE}
    realdata.write_json_atomic(path, m)
    return path


# ------------------------------------------------------------------ execution
@dataclass
class ExecResult:
    bench_id: str
    blocks_completed: int
    halted: Optional[str]
    quarantined: Dict[str, str]
    units: Dict[str, Dict[str, Any]]
    spent_guard_usd: Decimal


def prediction_path(reports_dir: Path, bench_id: str, model_key: str, task: str) -> Path:
    return Path(reports_dir) / "classification" / f"{bench_id}__{model_key}__{task}.jsonl"


def execute(
    plan: Dict[str, Any], loaded: LoadedCases, prompts: Dict[Task, str], auth: realdata.RealDataAuthorization, live: LiveAuthorization,
    adapters: Dict[str, object], budget: BudgetService, *, root: Path, reports_dir: Path, events: Optional[EventStore] = None,
    max_blocks: Optional[int] = None, progress: Callable[[str], None] = lambda s: None, sleep: Callable[[float], None] = time.sleep,
) -> ExecResult:
    """Run the plan's blocks in order, resuming from the checkpoint. Re-checks every authorization before the first request."""
    if not realdata.is_issued(auth):
        raise LiveNotAuthorizedError("a real-data authorization issued by the typed confirmation in this process is required")
    if auth.plan_fingerprint != plan["fingerprint"] or auth.dataset_version != plan["dataset"]["version"]:
        raise LiveNotAuthorizedError("the authorization is bound to a different plan or dataset version")
    models = {p["model_id"]: p for p in plan["participants"]}
    if not set(models) <= set(auth.models) or not set(models) <= set(live.models):
        raise LiveNotAuthorizedError("the authorizations do not cover every participant in the plan")
    if plan["blockers"]:
        raise PlanError("the plan is blocked: " + "; ".join(plan["blockers"]))
    bid = plan["bench_id"]
    state = load_state(root, bid)
    block_size = plan["ordering"]["block_size"]
    order = {t: order_cases(loaded, t, plan["ordering"].get("sampling", "hash")) for t in (Task.INTENT, Task.LEAD)}
    hard_cap = Decimal(plan["limits"]["hard_cap_usd"])
    spent = Decimal(state.get("spent_guard_usd", "0"))
    store_events = events if events is not None else EventStore()
    n_blocks = plan["ordering"]["n_blocks"]
    last = n_blocks if max_blocks is None else min(n_blocks, max_blocks)
    try:
        for k in range(state.get("blocks_completed", 0), last):
            bound = Decimal(plan["blocks"][k]["upper_bound_usd"])
            live_models = [p for p in plan["participants"] if p["model_id"] not in state["quarantined"]]
            status = budget.get_status()
            if spent + bound > hard_cap or bound > status.remaining_pilot_usd or bound > status.remaining_total_usd or status.unresolved_spend_count:
                state["halted"] = f"BUDGET_GATE_BEFORE_BLOCK_{k}: spent {spent} + bound {bound} vs hard cap {hard_cap}; ledger remaining pilot {status.remaining_pilot_usd}, total {status.remaining_total_usd}; unresolved {status.unresolved_spend_count}"
                break
            for p in live_models:
                for tname in p["tasks"]:
                    task = Task(tname)
                    unit = f"{k}|{p['key']}|{tname}"
                    if state["units"].get(unit, {}).get("status") == "DONE":
                        continue
                    if k < p.get("inherited_completed_blocks", 0):
                        state["units"][unit] = {"status": "DONE", "inherited_from": p["store_bench"], "n_cases": 0}
                        continue
                    store_bid = p.get("store_bench") or bid
                    chunk = order[task][k * block_size:(k + 1) * block_size]
                    skip = _over_cap_ids(plan, p, task, chunk, prompts, adapters)
                    chunk = [c for c in chunk if c.case_id not in skip]
                    if not chunk:
                        state["units"][unit] = {"status": "DONE", "n_cases": 0}
                        continue
                    store = PredictionStore(prediction_path(reports_dir, store_bid, p["key"], tname))
                    runner = ClassificationRunner(adapters, store, budget, stage=STAGE, telemetry=store_events)
                    progress(f"block {k} {p['key']} {tname}: {len(chunk)} requests")
                    try:
                        s = runner.run(task=task, cases=chunk, model_id=p["model_id"], prompt_text=prompts[task], authorization=live,
                                       run_id=f"{store_bid}-{p['key']}-{tname}", dataset_version=plan["dataset"]["version"], hard_cap_usd=hard_cap - spent,
                                       real_data=auth)
                    except (BudgetExceededError, LiveNotAuthorizedError) as e:
                        state["halted"] = f"{type(e).__name__}: {e}"
                        save_state(root, {**state, "spent_guard_usd": str(spent)})
                        raise
                    except (ProviderError, UsageUnavailableError, Exception) as e:   # quarantine the participant, keep the benchmark honest
                        state["quarantined"][p["model_id"]] = f"{type(e).__name__}: {str(e)[:200]} (block {k}, {tname})"
                        state["units"][unit] = {"status": "FAILED", "reason": state["quarantined"][p["model_id"]]}
                        save_state(root, {**state, "spent_guard_usd": str(spent)})
                        break
                    spent += s.guard_cost_usd
                    status_u = "DONE" if not s.aborted_reason else "QUARANTINED"
                    state["units"][unit] = {"status": status_u, "n_cases": len(chunk), "ok": s.n_ok, "invalid": s.n_invalid, "error": s.n_error,
                                            "skipped_existing": s.n_skipped_existing, "guard_cost_usd": str(s.guard_cost_usd), "aborted": s.aborted_reason}
                    # Errors that repeat across blocks trip the same breaker even when each block is too small to trip it alone.
                    streak = state.setdefault("consecutive_errors", {})
                    streak[p["model_id"]] = (streak.get(p["model_id"], 0) + s.n_error) if (s.n_ok + s.n_invalid == 0 and s.n_error) else 0
                    if streak[p["model_id"]] >= MAX_CONSECUTIVE_ERRORS and not s.aborted_reason:
                        s.aborted_reason = f"{streak[p['model_id']]} consecutive errors across blocks"
                    if s.aborted_reason:
                        state["units"][unit].update({"status": "QUARANTINED", "aborted": s.aborted_reason})
                        state["quarantined"][p["model_id"]] = f"circuit breaker: {s.aborted_reason} (block {k}, {tname})"
                        save_state(root, {**state, "spent_guard_usd": str(spent)})
                        break
                    save_state(root, {**state, "spent_guard_usd": str(spent)})
            state["blocks_completed"] = k + 1   # a quarantined participant stops only its own work; the block still counts for the others
            save_state(root, {**state, "spent_guard_usd": str(spent)})
        if state["blocks_completed"] >= n_blocks:
            state["halted"] = None
    except KeyboardInterrupt:
        state["halted"] = "INTERRUPTED_BY_OPERATOR (resumable)"
        save_state(root, {**state, "spent_guard_usd": str(spent)})
        raise
    state["spent_guard_usd"] = str(spent)
    save_state(root, state)
    return ExecResult(bid, state["blocks_completed"], state["halted"], dict(state["quarantined"]), dict(state["units"]), spent)


def _over_cap_ids(plan, participant, task: Task, chunk: Sequence[Case], prompts, adapters) -> set:
    prices = registry.model_level_prices(participant["model_id"])
    adapter = adapters[participant["model_id"]]
    cap = Decimal(plan["limits"]["per_case_usd"])
    pb = len(prompts[task].encode("utf-8"))
    return {c.case_id for c in chunk if _case_bound(task, _state_bytes(task, c), pb, prices, adapter.max_output_tokens(task)) > cap}
