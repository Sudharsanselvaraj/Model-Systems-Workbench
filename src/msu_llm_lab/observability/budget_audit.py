"""Offline reconciliation of the budget ledger with the stored results. Reads the ledger and result files; writes nothing to either.

Two different numbers are easy to confuse:
  * the LEDGER total is what the budget layer counts against the ceilings: every recorded transaction, booked at the estimated USD figure
    (catalogue price x reported tokens) because the provider's native cost is in credits with an unverified USD ratio;
  * the GUARD cost in a benchmark's state/manifest is an in-process stop check for ONE plan: per request the larger of the booked USD and the native
    credit figure, summed over that plan's new requests only. It exists to stop a run early and is never booked.
Neither is a provider invoice.
"""

import json
import re
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from msu_llm_lab.classification.records import PredictionRecord, PredictionStore

_BENCH = re.compile(r"^(bench-[0-9a-f]{12})-")
_SMOKE = re.compile(r"^(smoke_[0-9a-f]{8})-")


def _group(run_id: Optional[str]) -> str:
    rid = run_id or "(no run id)"
    for pat in (_BENCH, _SMOKE):
        m = pat.match(rid)
        if m:
            return m.group(1)
    return rid.split("-")[0]


def _micros(d: Optional[Decimal]) -> int:
    return int(d * 1_000_000) if d is not None else 0


def audit(db_path: Path, policy, reports_dir: Path, bench_root: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT run_id, cost_basis, amount_micros, is_pilot, is_unresolved, is_simulated, is_preflight_estimate FROM budget_ledger")]
    conn.close()
    counted = [r for r in rows if not r["is_simulated"] and not r["is_preflight_estimate"]]
    by_group: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"ledger_rows": 0, "ledger_micros": 0, "basis": defaultdict(int), "unresolved_rows": 0})
    for r in counted:
        g = by_group[_group(r["run_id"])]
        g["ledger_rows"] += 1
        g["ledger_micros"] += r["amount_micros"]
        g["basis"][r["cost_basis"]] += r["amount_micros"]
        g["unresolved_rows"] += r["is_unresolved"]
    recs_by: Dict[str, List[PredictionRecord]] = defaultdict(list)
    for d in ("classification", "smoke"):
        base = Path(reports_dir) / d
        for f in sorted(base.glob("*.jsonl")) if base.exists() else []:
            try:
                for r in PredictionStore(f).load():
                    recs_by[_group(r.run_id)].append(r)
            except Exception:
                continue
    groups = []
    for name in sorted(set(by_group) | set(recs_by)):
        g = by_group.get(name, {"ledger_rows": 0, "ledger_micros": 0, "basis": {}, "unresolved_rows": 0})
        recs = recs_by.get(name, [])
        booked = [r for r in recs if r.status != "ERROR" and r.cost_basis != "UNAVAILABLE"]
        rec_micros = sum(_micros(r.recorded_cost_usd) for r in booked)
        guard = sum(max(_micros(r.recorded_cost_usd), _micros(r.native_cost)) for r in recs)
        native = sum(_micros(r.native_cost) for r in recs)
        state_guard = None
        sp = Path(bench_root) / name / "state.json"
        if sp.exists():
            try:
                state_guard = int(Decimal(json.loads(sp.read_text()).get("spent_guard_usd", "0")) * 1_000_000)
            except (OSError, ValueError):
                state_guard = None
        groups.append({
            "group": name, "kind": "benchmark" if name.startswith("bench-") else "smoke" if name.startswith("smoke_") else "other", "ledger_rows": g["ledger_rows"],
            "ledger_usd": str(Decimal(g["ledger_micros"]) / 1_000_000), "ledger_basis_usd": {k: str(Decimal(v) / 1_000_000) for k, v in g["basis"].items()},
            "records": len(recs), "records_booked": len(booked), "records_error_no_cost": sum(1 for r in recs if r.status == "ERROR"),
            "records_recorded_cost_usd": str(Decimal(rec_micros) / 1_000_000), "records_native_credits": str(Decimal(native) / 1_000_000),
            "guard_from_records_usd": str(Decimal(guard) / 1_000_000), "guard_in_state_usd": None if state_guard is None else str(Decimal(state_guard) / 1_000_000),
            "ledger_equals_records": g["ledger_micros"] == rec_micros, "guard_equals_state": (state_guard is None) or abs(state_guard - guard) <= max(1, len(recs)),
            "ledger_minus_records_usd": str(Decimal(g["ledger_micros"] - rec_micros) / 1_000_000),
        })
    total = sum(r["amount_micros"] for r in counted)
    pilot = sum(r["amount_micros"] for r in counted if r["is_pilot"])
    unresolved = sum(1 for r in counted if r["is_unresolved"])
    sum_groups = sum(int(Decimal(x["ledger_usd"]) * 1_000_000) for x in groups)
    ok = all(x["ledger_equals_records"] for x in groups) and sum_groups == total and unresolved == 0
    return {
        "schema": "budget-reconciliation-v1", "ledger_total_counted_usd": str(Decimal(total) / 1_000_000), "ledger_pilot_counted_usd": str(Decimal(pilot) / 1_000_000),
        "ledger_rows_total": len(rows), "ledger_rows_counted": len(counted), "excluded_rows": {"simulated": sum(1 for r in rows if r["is_simulated"]), "preflight_estimates": sum(1 for r in rows if r["is_preflight_estimate"])},
        "unresolved_entries": unresolved, "ceilings_usd": {"total": str(policy.total_budget_usd), "pilot": str(policy.pilot_budget_usd), "per_run": str(policy.per_run_ceiling_usd), "per_case": str(policy.per_case_ceiling_usd)},
        "remaining_usd": {"total": str(max(Decimal(0), policy.total_budget_usd - Decimal(total) / 1_000_000)), "pilot": str(max(Decimal(0), policy.pilot_budget_usd - Decimal(pilot) / 1_000_000))},
        "groups": groups, "groups_sum_equals_ledger_total": sum_groups == total, "reconciled": ok,
        "meanings": {
            "ledger_total": "what the budget layer counts against the ceilings: recorded transactions booked at ESTIMATED USD (catalogue price x reported tokens)",
            "guard_cost": "an in-process stop check for ONE plan: per request max(booked USD, native credits), summed over that plan's new requests; never booked",
            "native_credits": "the provider's usage.cost in credits with an unverified USD ratio; kept as evidence, never booked as USD",
            "not_an_invoice": "none of these figures is a verified provider charge",
        },
    }


def audit_markdown(a: Dict[str, Any]) -> str:
    L = ["# Budget reconciliation", "", f"**{'RECONCILED' if a['reconciled'] else 'NOT RECONCILED'}.** Ledger counted against the ceilings: **${a['ledger_total_counted_usd']}** (pilot ${a['ledger_pilot_counted_usd']}); "
         f"remaining total ${a['remaining_usd']['total']}, pilot ${a['remaining_usd']['pilot']}. Unresolved entries: {a['unresolved_entries']}. Ceilings: {a['ceilings_usd']}.", "",
         "Ledger rows: " + f"{a['ledger_rows_total']} total, {a['ledger_rows_counted']} counted; excluded {a['excluded_rows']}.", "", "## What the numbers mean", ""]
    L += [f"- **{k}**: {v}" for k, v in a["meanings"].items()]
    L += ["", "## By run group", "", "| Group | Kind | Ledger rows | Ledger USD | Records (booked / errors) | Recorded USD (sum of records) | Ledger = records | Native credits | Guard from records | Guard in state |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in a["groups"]:
        L.append(f"| {g['group']} | {g['kind']} | {g['ledger_rows']} | {g['ledger_usd']} | {g['records']} ({g['records_booked']} / {g['records_error_no_cost']}) | {g['records_recorded_cost_usd']} | {'yes' if g['ledger_equals_records'] else '**NO** (' + g['ledger_minus_records_usd'] + ')'} | {g['records_native_credits']} | {g['guard_from_records_usd']} | {g['guard_in_state_usd'] or '—'} |")
    L += ["", f"Group totals equal the ledger total: **{a['groups_sum_equals_ledger_total']}**."]
    return "\n".join(L) + "\n"
