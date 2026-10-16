"""Controlled prompt comparison for Jev 1.14 on Main Intent (pre-registered in jev_prompt_experiment/PREREGISTRATION.md).

Same guarded executor, its own plan, its own approvals and permission record (bound to this plan's fingerprint), its own typed authorisation (dataset version
`labeled-eval-v1-jev-prompt-exp`). Only the prompt text differs between variants. Nothing is sent unless every blocking gate passes.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from msu_llm_lab.classification.benchmark import _case_bound, _state_bytes
from msu_llm_lab.classification.chat_adapter import MAX_OUTPUT_TOKENS
from msu_llm_lab.classification.records import Task
from msu_llm_lab.core.budget import BudgetService
from msu_llm_lab.labeled_eval import budget_plan as BP
from msu_llm_lab.labeled_eval import execute as X
from msu_llm_lab.labeled_eval import master as M
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.models import registry
from msu_llm_lab.observability.evaluation_report import holm

EXP_VERSION = "labeled-eval-v1-jev-prompt-exp"
KEY = "jev-1.14"
EXPECTED_HASH12 = {"V0": "0071512ad6aa", "V1": "", "V2": ""}        # V1/V2 hashes are fixed by write_variants and recorded in PREREGISTRATION.md
CLARIFY = {
    "UPLOAD_STATUS_CHECK": (" Clarification: when the student says they already uploaded, submitted, sent or attached something and asks whether it was received, checked or accepted, the label is UPLOAD_STATUS_CHECK "
                            "even if the wording is informal or in Malay; it is QUERY only under Exception A (grades or data pasted) or Exception B (asks how or when to edit or resubmit)."),
    "HUMAN_HANDOFF_REQUEST": (" Clarification: a short affirmative reply (for example yes, yes please, ok, sure, boleh) to a last Assistant message that offered to connect the student with a counsellor, advisor or staff member "
                              "is an acceptance and is HUMAN_HANDOFF_REQUEST; a short affirmative reply to any other offer or question is QUERY."),
    "QUERY": (" Clarification: choose QUERY only when none of the earlier labels applies after checking the decision steps in order; it is not a substitute for UPLOAD_STATUS_CHECK or HUMAN_HANDOFF_REQUEST "
              "when their conditions are met."),
}
EXAMPLES = [('Last A: "Would you like me to connect you with a counsellor?"', '"yes please"', "HUMAN_HANDOFF_REQUEST", "2b"),
            ('Last A: "I can also share the fee breakdown. Want it?"', '"yes"', "QUERY", "4"),
            ("no history", '"dah hantar result spm semalam, ada received tak?"', "UPLOAD_STATUS_CHECK", "3"),
            ("no history", '"I uploaded the wrong file, how do I change it?"', "QUERY", "3 (Exception B)"),
            ('Last A: "The fee is RM 12,000 per year."', '"ok thanks"', "GREETING", "5"),
            ("no history", '"can someone from your team call me about the scholarship?"', "HUMAN_HANDOFF_REQUEST", "2a"),
            ('Last A: "Would you like me to connect you with a counsellor?"', '"no thanks"', "GREETING", "4"),
            ("no history", '"I sent my transcript by email, can you confirm you got it? my CGPA is 3.2 and I took physics, chemistry and biology"', "QUERY", "3 (Exception A)")]


def make_variants(base: str) -> Dict[str, str]:
    """V1 changes only label definitions (text sent to Jev as its criteria); V2 adds hand-written synthetic example rows to the examples table. Nothing else changes."""
    import re

    v1 = base
    for label, add in CLARIFY.items():
        pat = re.compile(rf"^(\*\*{label}\*\* — .+)$", re.M)
        if not pat.search(v1):
            raise ValueError(f"definition line for {label} not found")
        v1 = pat.sub(lambda m: m.group(1) + add, v1, count=1)
    lines = v1.split("\n")
    idx = [i for i, ln in enumerate(lines) if re.match(r"^\| \d+ \|", ln)]
    if not idx:
        raise ValueError("examples table not found")
    n = max(int(re.match(r"^\| (\d+) \|", lines[i]).group(1)) for i in idx)
    rows = [f"| {n + 1 + k} | {c} | {m} | `{lab}` | {st} |" for k, (c, m, lab, st) in enumerate(EXAMPLES)]
    lines[idx[-1] + 1:idx[-1] + 1] = rows
    return {"V1": v1, "V2": "\n".join(lines)}


def write_variants(cfg: R.Cfg) -> Dict[str, str]:
    base = Path("main_intent_classifier.md").read_text()
    v = make_variants(base)
    names = {"V1": "V1_label_definitions.md", "V2": "V2_definitions_plus_examples.md"}
    exp_dir(cfg).mkdir(parents=True, exist_ok=True)
    for k, n in names.items():
        (exp_dir(cfg) / n).write_text(v[k])
    return {k: hashlib.sha256(v[k].encode()).hexdigest()[:12] for k in v}


def question_hash(prompt: str) -> str:
    """Hash of exactly what Jev receives (instructions + criteria); a variant must change it, or the experiment would be a no-op."""
    from msu_llm_lab.classification import jev_payloads as JP

    return hashlib.sha256(json.dumps(JP.build_intent_question(prompt), sort_keys=True).encode()).hexdigest()[:12]

APPROVAL_FILE = "operator_approval_jev_experiment.json"
PERMISSION_FILE = "processing_permission_jev_experiment.json"


def is_placeholder(v: Any) -> bool:
    """An unfilled template such as '<your approval>' or '<date>' is not a statement; it is rejected when recording and again by the gates."""
    import re

    return bool(re.search(r"<[^<>]{1,80}>", str(v)))


def exp_dir(cfg: R.Cfg) -> Path:
    return cfg.out / R.VERSION / "jev_prompt_experiment"


def priv_dir(cfg: R.Cfg) -> Path:
    return cfg.private / R.VERSION / "jev_prompt_experiment"


def variant_files(cfg: R.Cfg) -> Dict[str, Path]:
    return {"V0": Path("main_intent_classifier.md"), "V1": exp_dir(cfg) / "V1_label_definitions.md", "V2": exp_dir(cfg) / "V2_definitions_plus_examples.md"}


def load_prompts(cfg: R.Cfg, files: Optional[Dict[str, Path]] = None) -> Dict[str, str]:
    return {k: p.read_text() for k, p in (files or variant_files(cfg)).items()}


def halves(cfg: R.Cfg) -> Dict[str, List[str]]:
    sp = pd.read_csv(cfg.out / R.VERSION / "split_manifest.csv", dtype=str)
    sp = sp[sp.task == "INTENT"]
    return {"dev": sorted(sp[sp.exploratory_half == "dev"].case_id), "heldout": sorted(sp[sp.exploratory_half == "test"].case_id)}


def build_plan(cfg: R.Cfg, budget: BudgetService, cases: Optional[List[Any]] = None, prompts: Optional[Dict[str, str]] = None, hash_check: bool = True) -> Dict[str, Any]:
    cases = cases if cases is not None else X.load_cases(cfg)["INTENT"]
    by_id = {c.case_id: c for c in cases}
    prompts = prompts or load_prompts(cfg)
    h = halves(cfg)
    rec = registry.get_record(KEY)
    mid, prices = rec.requested_model_id.value, registry.model_level_prices(rec.requested_model_id.value)
    man = json.loads((cfg.out / R.VERSION / "classifier_evaluation_manifest.json").read_text())
    units: List[Dict[str, Any]] = []
    sched = [("V0", "V0", "dev"), ("V0", "V0", "heldout"), ("V1", "V1", "dev"), ("V1", "V1", "heldout"), ("V2", "V2", "dev"), ("V2", "V2", "heldout"), ("V0-repeat", "V0", "heldout")]
    for name, pkey, half in sched:
        ids = h[half]
        pb = len(prompts[pkey].encode())
        for i in range(0, len(ids), BP.BLOCK):
            chunk = ids[i:i + BP.BLOCK]
            bounds = [_case_bound(Task.INTENT, _state_bytes(Task.INTENT, by_id[c]), pb, prices, X.CAP_TOKENS["INTENT"]) for c in chunk]
            units.append({"unit": f"{KEY}|INTENT|{name}|{half}|{i // BP.BLOCK}", "key": KEY, "model_id": mid, "task": "INTENT", "chunk": i // BP.BLOCK, "variant": name, "half": half, "case_ids": chunk,
                          "bound_usd": str(sum(bounds, Decimal("0"))), "max_case_bound_usd": str(max(bounds)), "prompt_key": pkey, "store_name": name, "run_id": f"{EXP_VERSION}-{name}"})
    pr_hash = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.items()}
    body = {"version": EXP_VERSION, "dataset_hash_intent": man["datasets"]["intent"]["sha256"], "participant": {"key": KEY, "model_id": mid, "route": rec.pinned_tag}, "prompt_hashes": pr_hash, "question_hashes": {k: question_hash(v) for k, v in prompts.items()},
            "output_caps": X.CAP_TOKENS, "caps": X.RECORDED_CAPS, "chunk": BP.BLOCK, "max_attempts": BP.MAX_ATTEMPTS, "reserve_usd": str(BP.RESERVE_USD),
            "halves": {k: hashlib.sha256("|".join(v).encode()).hexdigest()[:12] for k, v in h.items()},
            "units": [{k: u[k] for k in ("unit", "case_ids", "bound_usd", "prompt_key")} for u in units]}
    fp = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    st = budget.get_status()
    total = sum((Decimal(u["bound_usd"]) for u in units), Decimal("0"))
    pr = rec.pinned_route()
    return {**body, "fingerprint": fp, "dataset_version": EXP_VERSION, "store_dir": f"{R.VERSION}/jev_prompt_experiment", "units_full": units, "guard_bound_total_usd": str(total),
            "n_requests": sum(len(u["case_ids"]) for u in units), "participants": [{"key": KEY, "model_id": mid, "route": rec.pinned_tag, "tasks": ["INTENT"]}],
            "recipients": {mid: f"OpenRouter, routed to {pr.provider if pr else rec.provider_route.value}" + (f" (pinned: {rec.pinned_tag})" if rec.pinned_tag else "")},
            "ledger_at_plan_time": {"remaining_pilot_usd": str(st.remaining_pilot_usd), "remaining_total_usd": str(st.remaining_total_usd), "unresolved": st.unresolved_spend_count}}


def _read(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def record_operator_approval(cfg: R.Cfg, plan: Dict[str, Any], *, operator: str, statement: str) -> Dict[str, Any]:
    if not statement.strip() or not operator.strip():
        raise ValueError("operator and statement are required")
    if is_placeholder(statement) or is_placeholder(operator):
        raise ValueError("the statement still contains template text in <angle brackets>; write your own words")
    a = {"schema": "jev-experiment-operator-approval-v1", "operator": operator, "statement": statement, "plan_fingerprint": plan["fingerprint"], "dataset_hash_intent": plan["dataset_hash_intent"],
         "prompt_hashes": plan["prompt_hashes"], "caps": X.RECORDED_CAPS, "reserve_usd": plan["reserve_usd"], "n_requests": plan["n_requests"],
         "does_not_verify": ["processing permission (separate record)", "provider policies", "data provenance"]}
    (exp_dir(cfg) / APPROVAL_FILE).write_text(json.dumps(a, indent=2, sort_keys=True))
    R._write_private(priv_dir(cfg) / APPROVAL_FILE, json.dumps(a, indent=2, sort_keys=True))
    return a


def record_permission(cfg: R.Cfg, plan: Dict[str, Any], *, evidence_type: str, verified_by: str, evidence_reference: str, approvers: str, approved_on: str, source: str) -> Dict[str, Any]:
    if evidence_type not in ("DOCUMENT", "OPERATOR_CHAT_ATTESTATION"):
        raise ValueError("evidence_type must be DOCUMENT or OPERATOR_CHAT_ATTESTATION")
    for n, v in (("verified_by", verified_by), ("evidence_reference", evidence_reference), ("approvers", approvers), ("approved_on", approved_on), ("source", source)):
        if not str(v).strip():
            raise ValueError(f"{n} is required (state 'NOT PROVIDED' if it was not given)")
        if is_placeholder(v):
            raise ValueError(f"{n} still contains template text in <angle brackets>; replace it with the real value or 'NOT PROVIDED'")
    rec = {"schema": "jev-experiment-processing-permission-v1", "verified": True, "verified_by": verified_by, "evidence_type": evidence_type, "evidence_reference": evidence_reference, "approvers": approvers,
           "approved_on": approved_on, "source": source, "dataset_hash_intent": plan["dataset_hash_intent"], "models": [plan["participant"]["model_id"]], "routes": {plan["participant"]["model_id"]: plan["participant"]["route"]},
           "tasks": ["INTENT"], "processing_path": "OpenRouter, routed to the pinned provider (fallbacks disabled)", "plan_fingerprint": plan["fingerprint"], "prompt_hashes": plan["prompt_hashes"],
           "meaning": "Approval to process the 120 Main Intent cases through this route with these prompt variants. NOT a verification of any provider's policy.",
           "limitations": [] if evidence_type == "DOCUMENT" else ["operator attestation; no document located"]}
    (exp_dir(cfg) / PERMISSION_FILE).write_text(json.dumps(rec, indent=2, sort_keys=True))
    R._write_private(priv_dir(cfg) / PERMISSION_FILE, json.dumps(rec, indent=2, sort_keys=True))
    return rec


def check_gates(cfg: R.Cfg, plan: Dict[str, Any], budget: BudgetService, files: Optional[Dict[str, Path]] = None) -> List[Dict[str, Any]]:
    from msu_llm_lab.labeled_eval import stage2

    g: List[Dict[str, Any]] = []
    man = _read(cfg.out / R.VERSION / "classifier_evaluation_manifest.json") or {}
    files = files or variant_files(cfg)
    appr = _read(exp_dir(cfg) / APPROVAL_FILE)
    ok = bool(appr and not is_placeholder(appr.get("statement", "")) and appr.get("plan_fingerprint") == plan["fingerprint"] and appr.get("prompt_hashes") == plan["prompt_hashes"] and appr.get("dataset_hash_intent") == plan["dataset_hash_intent"] and appr.get("caps") == X.RECORDED_CAPS)
    g.append(X._gate("E1", "operator approval bound to this experiment plan, prompts and unchanged caps", ok, "recorded for this fingerprint" if ok else "no operator approval recorded for this exact plan"))
    perm = _read(priv_dir(cfg) / PERMISSION_FILE)
    mid = plan["participant"]["model_id"]
    okp = bool(perm and not any(is_placeholder(perm.get(k, "")) for k in ("verified_by", "evidence_reference", "approvers", "approved_on", "source")) and perm.get("verified") is True and perm.get("verified_by") and perm.get("evidence_reference") and perm.get("evidence_type") in ("DOCUMENT", "OPERATOR_CHAT_ATTESTATION")
               and perm.get("dataset_hash_intent") == plan["dataset_hash_intent"] and perm.get("models") == [mid] and perm.get("routes") == {mid: plan["participant"]["route"]} and perm.get("tasks") == ["INTENT"]
               and "OpenRouter" in str(perm.get("processing_path", "")) and perm.get("plan_fingerprint") == plan["fingerprint"] and perm.get("prompt_hashes") == plan["prompt_hashes"])
    g.append(X._gate("E2", "data-processing permission recorded for this dataset, model, route, task, prompts and plan", okp,
                     "permission record matches the plan" if okp else f"{PERMISSION_FILE} missing or does not cover this plan (dataset hash, Jev route, INTENT, OpenRouter path, prompt hashes, fingerprint)"))
    isdoc = bool(perm and perm.get("evidence_type") == "DOCUMENT")
    g.append(X._gate("E2-evidence", "permission evidence is a document", isdoc, f"evidence_type={perm.get('evidence_type') if perm else 'none'}", blocking=False))
    hh = {k: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for k, p in files.items() if p.exists()}
    okh = hh.get("V0") == EXPECTED_HASH12["V0"] and {k: hashlib.sha256(files[k].read_bytes()).hexdigest() for k in files} == plan["prompt_hashes"]
    g.append(X._gate("E3", "prompt files match the plan and V0 is the unchanged authoritative prompt", okh, f"hashes {hh}"))
    qh = plan["question_hashes"]
    okq = len({qh["V0"], qh["V1"], qh["V2"]}) == 3
    g.append(X._gate("E3b", "each variant changes what Jev actually receives (the parser drops unmapped sections)", okq, f"question hashes {qh}"))
    sep = stage2.separation_check(cfg, man) if man else {"ok": False}
    g.append(X._gate("E4", "dataset hash unchanged; answer key separate from inference inputs", sep["ok"], "all separation checks pass" if sep["ok"] else "a separation check failed"))
    h = halves(cfg)
    g.append(X._gate("E5", "pre-registered halves intact (59 dev / 61 held-out, disjoint, 120 total)", len(h["dev"]) == 59 and len(h["heldout"]) == 61 and not set(h["dev"]) & set(h["heldout"]), f"{len(h['dev'])}/{len(h['heldout'])}"))
    g.append(X._gate("E6", "output caps are the existing 16 tokens", plan["output_caps"] == X.CAP_TOKENS and X.MAX_OUTPUT_TOKENS_STR()["INTENT"] == 16, f"{plan['output_caps']}"))
    pol = budget.policy
    eff = {"total_budget_usd": str(pol.total_budget_usd), "pilot_budget_usd": str(pol.pilot_budget_usd), "per_run_ceiling_usd": str(pol.per_run_ceiling_usd), "per_case_ceiling_usd": str(pol.per_case_ceiling_usd)}
    g.append(X._gate("B1", "effective caps equal the recorded caps (none raised)", {k: Decimal(v) for k, v in eff.items()} == {k: Decimal(v) for k, v in X.RECORDED_CAPS.items()}, f"effective {eff}"))
    st = budget.get_status()
    avail = min(st.remaining_pilot_usd, st.remaining_total_usd) - Decimal(plan["reserve_usd"])
    g.append(X._gate("B2", "no unresolved spend", st.unresolved_spend_count == 0, f"{st.unresolved_spend_count} unresolved"))
    g.append(X._gate("B3", "guard bound fits the headroom minus the reserve", Decimal(plan["guard_bound_total_usd"]) <= avail, f"bound ${plan['guard_bound_total_usd']} vs available ${avail}"))
    over = [u["unit"] for u in plan["units_full"] if Decimal(u["bound_usd"]) > pol.per_run_ceiling_usd or Decimal(u["max_case_bound_usd"]) > pol.per_case_ceiling_usd]
    g.append(X._gate("B4", "every chunk within the per-run cap and every case within the per-case cap", not over, f"over cap: {over[:3]}"))
    rec = registry.get_record(KEY)
    bl = registry.synthetic_live_blockers(plan["participant"]["model_id"], X.EXECUTION_STAGE) + [x for x in rec.student_data_blockers() if x != "data_policy"]
    g.append(X._gate("R1", "registry readiness and pinned route for Jev (data_policy UNKNOWN is covered only by the typed authorisation)", not bl and bool(rec.pinned_tag), f"blockers {bl}"))
    return g


def render_plan(plan: Dict[str, Any], gates: List[Dict[str, Any]]) -> List[str]:
    return [f"jev prompt experiment {EXP_VERSION}   plan {plan['fingerprint']}   requests {plan['n_requests']}   guard bound ${plan['guard_bound_total_usd']}",
            "variants: " + ", ".join(f"{k} {v[:12]}" for k, v in plan["prompt_hashes"].items()) + "   (V0 repeat on the held-out half is the noise control)",
            *[f"  recipient {m}: {r}" for m, r in plan["recipients"].items()], "GATES:",
            *[f"  [{'PASS' if g['ok'] else 'FAIL'}] {g['id']} {g['item']}: {g['detail']}" for g in gates]]


def save_plan(cfg: R.Cfg, plan: Dict[str, Any]) -> Path:
    p = exp_dir(cfg) / "plan.json"
    if p.exists() and json.loads(p.read_text()).get("fingerprint") != plan["fingerprint"]:
        raise ValueError("a different experiment plan is already stored")
    X.realdata.write_json_atomic(p, plan)
    return p
