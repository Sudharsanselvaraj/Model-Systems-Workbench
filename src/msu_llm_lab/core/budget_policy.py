"""Operator-authorized changes to the pilot ceiling, kept as an append-only audit trail.

Only the pilot-stage ceiling can be raised here, never above the total lifetime ceiling, and only through an interactive typed phrase that names
the amounts. The ledger is never touched. The per-run and per-case ceilings and the total ceiling come from the code defaults and cannot be changed
by this mechanism. A damaged or inconsistent override file is ignored (the default ceilings apply) and reported.
"""

import json
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from msu_llm_lab.core.config import BudgetPolicy
from msu_llm_lab.providers.base import LiveNotAuthorizedError

POLICY_FILE = "budget_policy.json"
SCHEMA = "budget-policy-overrides-v1"
STATEMENT = ("Operator authorizes raising the pilot-stage ceiling for the real-data benchmark. The total lifetime ceiling, the per-run ceiling and the per-case "
             "ceiling are unchanged and the ledger is not reset.")


class PolicyError(ValueError):
    pass


def policy_path(data_dir: Path) -> Path:
    return Path(data_dir) / POLICY_FILE


def _read(data_dir: Path) -> Dict[str, Any]:
    p = policy_path(data_dir)
    if not p.exists():
        return {"schema": SCHEMA, "history": []}
    try:
        d = json.loads(p.read_text())
        if d.get("schema") != SCHEMA or not isinstance(d.get("history"), list):
            raise ValueError("unrecognised structure")
        return d
    except (OSError, ValueError) as e:
        raise PolicyError(f"{p} is unreadable ({e}); the default ceilings apply until it is fixed") from e


def history(data_dir: Path) -> List[Dict[str, Any]]:
    try:
        return list(_read(data_dir)["history"])
    except PolicyError:
        return []


def effective_policy(base: BudgetPolicy, data_dir: Path) -> BudgetPolicy:
    """The base policy with the latest valid pilot override applied. Never exceeds the total ceiling and never touches any other limit."""
    try:
        hist = _read(data_dir)["history"]
    except PolicyError:
        return base
    for entry in reversed(hist):
        try:
            new = Decimal(entry["new_pilot_budget_usd"])
            if Decimal("0") < new <= base.total_budget_usd and entry.get("total_budget_usd") == str(base.total_budget_usd):
                return base.model_copy(update={"pilot_budget_usd": new})
        except (KeyError, InvalidOperation, TypeError):
            continue
        break
    return base


def expected_phrase(new_pilot: Decimal, total: Decimal) -> str:
    return f"AUTHORIZE PILOT CEILING {new_pilot:.2f} USD TOTAL CEILING {total:.2f} USD UNCHANGED"


def raise_pilot_ceiling(
    data_dir: Path, base: BudgetPolicy, new_pilot: Decimal, *, operator: str, ledger_total_spent: Decimal, ledger_pilot_spent: Decimal,
    is_interactive: Optional[Callable[[], bool]] = None, prompt: Optional[Callable[[str], str]] = None, echo: Callable[[str], None] = print,
) -> Dict[str, Any]:
    interactive = is_interactive if is_interactive is not None else (lambda: sys.stdin.isatty() and sys.stdout.isatty())
    current = effective_policy(base, data_dir)
    if not interactive():
        raise LiveNotAuthorizedError("Raising the pilot ceiling needs an interactive terminal; non-interactive execution is refused.")
    if not operator.strip():
        raise LiveNotAuthorizedError("A named operator is required.")
    if new_pilot <= current.pilot_budget_usd:
        raise PolicyError(f"the pilot ceiling is already ${current.pilot_budget_usd}; this command only raises it")
    if new_pilot > base.total_budget_usd:
        raise PolicyError(f"the pilot ceiling cannot exceed the total lifetime ceiling (${base.total_budget_usd})")
    echo(f"Pilot ceiling: ${current.pilot_budget_usd} -> ${new_pilot:.2f}.  Total lifetime ceiling stays ${base.total_budget_usd}.  Per-run ${base.per_run_ceiling_usd}, per-case ${base.per_case_ceiling_usd} unchanged.")
    echo(f"Ledger (kept as is): ${ledger_total_spent} counted in total, ${ledger_pilot_spent} counted as pilot; ${base.total_budget_usd - ledger_total_spent} of the lifetime ceiling remains.")
    echo(STATEMENT)
    phrase = expected_phrase(new_pilot, base.total_budget_usd)
    echo(f"Type exactly: {phrase}")
    if prompt is None:
        import typer
        prompt = lambda msg: typer.prompt(msg, default="", show_default=False)  # noqa: E731
    if prompt("Budget confirmation").strip() != phrase:
        raise LiveNotAuthorizedError("Confirmation phrase did not match; the ceiling was not changed.")
    doc = _read(data_dir) if policy_path(data_dir).exists() else {"schema": SCHEMA, "history": []}
    entry = {"at": datetime.now(timezone.utc).isoformat(), "operator": operator.strip(), "previous_pilot_budget_usd": str(current.pilot_budget_usd),
             "new_pilot_budget_usd": f"{new_pilot:.2f}", "total_budget_usd": str(base.total_budget_usd), "per_run_ceiling_usd": str(base.per_run_ceiling_usd),
             "per_case_ceiling_usd": str(base.per_case_ceiling_usd), "ledger_total_spent_usd": str(ledger_total_spent), "ledger_pilot_spent_usd": str(ledger_pilot_spent),
             "statement": STATEMENT, "confirmation": "typed phrase matched"}
    doc["history"].append(entry)
    p = policy_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return entry
