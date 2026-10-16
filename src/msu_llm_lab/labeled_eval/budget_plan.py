"""Projected spend for a model cohort on the labelled suites, under the existing budget policy. Read-only: opens the ledger for status, sends nothing."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from msu_llm_lab.classification.benchmark import _case_bound, _state_bytes, load_prep_cases
from msu_llm_lab.classification.records import Task
from msu_llm_lab.labeled_eval import cases as C
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.models import registry

COHORT = {"INTENT": ["jev-1.14", "gpt-4.1-nano", "gpt-4o-mini", "qwen3.5-35b-a3b", "deepseek-chat-v3.1", "mistral-small-3.2-24b-instruct", "claude-haiku-5.5", "baseline-intent-gemini-3.5-flash-lite"],
          "LEAD": ["jev-1.14", "gpt-4.1-nano", "gpt-4o-mini", "qwen3.5-35b-a3b", "deepseek-chat-v3.1", "mistral-small-3.2-24b-instruct", "claude-haiku-5.5", "baseline-lead-gpt-4.1-mini"]}
EXCLUDED = ["gpt-4.1", "claude-sonnet-5.5"]
BLOCK = 10          # same chunk size as the real benchmark: one runner invocation = one "run" for the per-run ceiling
MAX_ATTEMPTS = 3
RESERVE_USD = Decimal("0.50")


def _inputs(cfg: R.Cfg, version: str = R.VERSION) -> Dict[str, List[Dict[str, Any]]]:
    d = cfg.private / version
    return {t: [json.loads(l) for l in open(d / f"inference_inputs_{t.lower()}.jsonl")] for t in ("INTENT", "LEAD")}


def build(cfg: R.Cfg) -> Dict[str, Any]:
    from msu_llm_lab.classification.chat_adapter import MAX_OUTPUT_TOKENS
    from msu_llm_lab.core.budget import BudgetService

    bs = BudgetService()
    st = bs.get_status()
    pol = bs.policy
    inputs = _inputs(cfg)
    pbytes = {"INTENT": len(Path("main_intent_classifier.md").read_bytes()), "LEAD": len(Path("lead_qualification_prompt.md").read_bytes())}
    sb = {"INTENT": [_state_bytes(Task.INTENT, C.intent_case(r)) for r in inputs["INTENT"]], "LEAD": [_state_bytes(Task.LEAD, C.lead_case(r)) for r in inputs["LEAD"]]}
    # mean serialized input size of the earlier benchmark's cases (local prep metadata, bytes only), used to scale observed token counts
    loaded = load_prep_cases(cfg.prep_dir, cfg.policy_name)
    bench_sb = {"INTENT": sum(_state_bytes(Task.INTENT, c) for c in loaded.intent) / max(1, len(loaded.intent)), "LEAD": sum(_state_bytes(Task.LEAD, c) for c in loaded.lead) / max(1, len(loaded.lead))}
    mts = pd.read_csv(cfg.prior_report / "model_task_summary.csv").set_index(["participant", "task"])
    res = pd.read_csv(cfg.bench_bundle / "model_results.csv", dtype=str, keep_default_na=False)
    mid = dict(zip(*pd.read_csv(cfg.bench_bundle / "model_summary.csv", dtype=str)[["model_key", "model_id"]].drop_duplicates().values.T))
    res["participant"] = res.model_key.map({v: k for k, v in mid.items()})
    res["attempts"] = pd.to_numeric(res.attempts, errors="coerce")
    rows = []
    for task, keys in COHORT.items():
        n = len(sb[task])
        for k in keys:
            m = mts.loc[(k, task)]
            prices = registry.model_level_prices(mid[k])
            pin, pout = float(prices[0]), float(prices[1])
            out_cap = MAX_OUTPUT_TOKENS[Task(task)]
            scale = [(pbytes[task] + b) / (pbytes[task] + bench_sb[task]) for b in sb[task]]
            exp_tok_in = [float(m.prompt_tokens_mean) * x for x in scale]
            expected = sum((t * pin + float(m.completion_tokens_mean) * pout) / 1e6 for t in exp_tok_in)
            bounds = [_case_bound(Task(task), b, pbytes[task], prices, out_cap) for b in sb[task]]       # the runner's own pessimistic per-case bound
            att = res[(res.participant == k) & (res.task == task)].attempts.dropna()
            mean_att = float(att.mean()) if len(att) else 1.0
            blocks = [sum(bounds[i:i + BLOCK], Decimal("0")) for i in range(0, n, BLOCK)]
            rows.append({"task": task, "participant": k, "n_cases": n, "expected_usd": expected, "expected_with_observed_retries_usd": expected * mean_att,
                         "observed_mean_attempts": mean_att, "guard_bound_usd": float(sum(bounds, Decimal("0"))), "worst_case_3_attempts_usd": float(sum(bounds, Decimal("0"))) * MAX_ATTEMPTS,
                         "max_case_bound_usd": float(max(bounds)), "max_block_bound_usd": float(max(blocks)), "case_cap_ok": max(bounds) <= pol.per_case_ceiling_usd,
                         "block_cap_ok": max(blocks) <= pol.per_run_ceiling_usd, "input_scale_vs_benchmark_mean": sum(scale) / len(scale),
                         "basis": "expected = observed mean tokens per request of the earlier benchmark scaled by input size, at registry prices; guard bound = runner's 2-bytes-per-token pessimistic bound + output cap"})
    df = pd.DataFrame(rows)
    headroom = min(st.remaining_pilot_usd, st.remaining_total_usd)
    tot = {c: float(df[c].sum()) for c in ("expected_usd", "expected_with_observed_retries_usd", "guard_bound_usd", "worst_case_3_attempts_usd")}
    policy = {"total_ceiling_usd": str(pol.total_budget_usd), "pilot_ceiling_usd": str(pol.pilot_budget_usd), "per_run_ceiling_usd": str(pol.per_run_ceiling_usd), "per_case_ceiling_usd": str(pol.per_case_ceiling_usd),
              "spent_total_usd": str(st.cumulative_spend_usd), "spent_pilot_usd": str(st.pilot_spend_usd), "remaining_total_usd": str(st.remaining_total_usd), "remaining_pilot_usd": str(st.remaining_pilot_usd),
              "unresolved_spend_entries": st.unresolved_spend_count, "reserve_usd": str(RESERVE_USD), "chunk_size_cases": BLOCK, "max_attempts": MAX_ATTEMPTS, "override_file": "data/budget_policy.json (typed pilot-ceiling override, total unchanged)"}
    verdict = {"headroom_usd": float(headroom), "fits_expected": tot["expected_with_observed_retries_usd"] <= float(headroom - RESERVE_USD), "fits_guard_bound": tot["guard_bound_usd"] <= float(headroom - RESERVE_USD),
               "fits_absolute_worst": tot["worst_case_3_attempts_usd"] <= float(headroom - RESERVE_USD), "all_case_caps_ok": bool(df.case_cap_ok.all()), "all_block_caps_ok": bool(df.block_cap_ok.all())}
    return {"table": df, "totals": tot, "policy": policy, "verdict": verdict}
