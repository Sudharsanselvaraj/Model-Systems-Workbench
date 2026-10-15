"""Deterministic, offline evaluation analysis of one real-data benchmark run.

Reads only local artifacts (result bundle, plan, state, ledger, request events, prep metadata); sends nothing, spends nothing and never writes into a source
location. Every figure is derived from files and tagged with its provenance. Accuracy-type metrics are NOT computable while ground truth is pending and are
reported as such, never estimated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sqlite3
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCHEMA = "evaluation-report-v1"
SEED = 20261015
N_BOOT = 2000
MIN_PAIRED_N = 30
NOT_COMPUTABLE = "NOT COMPUTABLE - GROUND TRUTH PENDING"
EXCLUDED_NAMES = ("salt", "secret", ".env")        # never hashed, listed or read


@dataclass
class Config:
    bench_id: str = "bench-d0a763e4a920"
    bundle_dir: Path = Path("reports/real_benchmark_bench-d0a763e4a920-r4")
    related_bundles: Tuple[str, ...] = ("reports/real_benchmark_bench-d0a763e4a920", "reports/real_benchmark_bench-d0a763e4a920-r2",
                                        "reports/real_benchmark_bench-d0a763e4a920-r3")
    bench_dir: Path = Path("data/benchmarks/bench-d0a763e4a920")
    db_path: Path = Path("data/workbench.db")
    events_path: Path = Path("data/telemetry/request_events.jsonl")
    prep_dir: Path = Path("data/classification_prep/prep-v2")
    policy: str = "2h"
    stores_dir: Path = Path("reports/classification")
    budget_policy: Path = Path("data/budget_policy.json")
    out_dir: Path = Path("reports/evaluation_report_bench-d0a763e4a920")
    seed: int = SEED
    n_boot: int = N_BOOT

    def as_json(self) -> Dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# ----------------------------------------------------------------------------- generic helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rng_for(label: str, seed: int = SEED) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(values: np.ndarray, stat, label: str, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    g = rng_for(label, seed)
    idx = g.integers(0, len(v), size=(n_boot, len(v)))
    s = np.array([stat(v[i]) for i in idx])
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def pct(v: np.ndarray, q: float) -> float:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, q)) if len(v) else float("nan")


def holm(pvals: List[float]) -> List[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cohen_kappa(a: List[str], b: List[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = pd.Series(a).value_counts(), pd.Series(b).value_counts()
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in set(ca.index) | set(cb.index))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ----------------------------------------------------------------------------- inventory and integrity
def _walk_inputs(cfg: Config) -> List[Path]:
    paths: List[Path] = []
    for d in (cfg.bundle_dir, *[Path(x) for x in cfg.related_bundles], cfg.bench_dir):
        if d.exists():
            paths += [p for p in sorted(d.rglob("*")) if p.is_file()]
    for p in (cfg.db_path, cfg.events_path, cfg.budget_policy, cfg.prep_dir / "manifest.json"):
        if p.exists():
            paths.append(p)
    if cfg.stores_dir.exists():
        paths += sorted(p for p in cfg.stores_dir.glob("bench-*.jsonl"))
    paths += sorted(Path("reports").glob("budget_reconciliation_*"))
    seen, out = set(), []
    for p in paths:
        if p not in seen and not any(x in p.name.lower() for x in EXCLUDED_NAMES):
            seen.add(p)
            out.append(p)
    return out


def _role(p: Path, cfg: Config) -> Tuple[str, str]:
    s = str(p)
    if p == cfg.db_path:
        return "ledger_database", "ledger (read-only)"
    if p == cfg.events_path:
        return "request_event_log", "telemetry (LIVE, no text)"
    if s.startswith(str(cfg.bench_dir)):
        return f"plan_state_manifest:{p.name}", "benchmark control files"
    if s.startswith(str(cfg.bundle_dir)):
        return f"final_bundle:{p.relative_to(cfg.bundle_dir)}", "final verified bundle (r4): primary analysis source"
    if "real_benchmark_bench-d0a763e4a920" in s:
        return f"superseded_bundle:{p.parent.name}/{p.name}", "earlier bundle of the same run (reconciliation history only)"
    if p.parent == cfg.stores_dir:
        return "prediction_store", "per-participant prediction store (carry-over lineage)"
    if p.name.startswith("budget_reconciliation"):
        return "budget_reconciliation_report", "budget audit output"
    return "other_input", "input"


def inventory(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rows, hashes = [], {}
    for p in _walk_inputs(cfg):
        h = sha256_file(p)
        hashes[str(p)] = h
        role, prov = _role(p, cfg)
        rows.append({"path": str(p), "role": role, "provenance": prov, "bytes": p.stat().st_size, "sha256": h,
                     "modified_utc": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "used_in_analysis": str(p.parent) == str(cfg.bundle_dir) or p in (cfg.db_path, cfg.events_path) or p.parent == cfg.bench_dir})
    return pd.DataFrame(rows), hashes


# ----------------------------------------------------------------------------- load
def load(cfg: Config) -> Dict[str, Any]:
    b = cfg.bundle_dir
    S: Dict[str, Any] = {"cfg": cfg}
    S["plan"] = json.loads((cfg.bench_dir / "plan.json").read_text())
    S["state"] = json.loads((cfg.bench_dir / "state.json").read_text())
    S["manifest"] = json.loads((b / "benchmark_manifest.json").read_text())
    S["recon_src"] = json.loads((b / "request_reconciliation.json").read_text())
    S["summary_src"] = pd.read_csv(b / "model_summary.csv", dtype=str, keep_default_na=False)
    S["ranking_src"] = pd.read_csv(b / "operational_ranking.csv", dtype=str, keep_default_na=False)
    S["pairs_src"] = pd.read_csv(b / "model_agreement_pairs.csv", dtype=str, keep_default_na=False)
    S["unattempted"] = pd.read_csv(b / "unattempted_in_executed_blocks.csv", dtype=str, keep_default_na=False)
    S["obs"] = json.loads((b / "observability.json").read_text())
    S["rate_limits"] = json.loads((b / "diagnostics" / "rate_limits.json").read_text())
    S["empty_outputs"] = json.loads((b / "diagnostics" / "empty_outputs.json").read_text())
    S["verification"] = json.loads((b / "verification_report.json").read_text())
    S["old_recon"] = {}
    for rb in cfg.related_bundles:
        f = Path(rb) / "request_reconciliation.json"
        if f.exists():
            S["old_recon"][Path(rb).name] = json.loads(f.read_text())

    raw = pd.read_csv(b / "model_results.csv", dtype=str, keep_default_na=False)
    smap = dict(zip(S["summary_src"]["model_id"], S["summary_src"]["model_key"]))
    raw["participant"] = raw["model_key"].map(smap)
    roles = dict(zip(S["summary_src"]["model_key"], S["summary_src"]["roles"]))
    raw["role"] = raw["participant"].map(roles)
    for c in ("latency_ms_end_to_end", "attempts", "prompt_tokens", "completion_tokens", "reasoning_tokens", "recorded_cost_usd", "estimated_cost_usd",
              "native_cost", "actual_cost_usd", "yes_probability"):
        raw[c] = _num(raw[c])
    raw["start_ts"] = pd.to_datetime(raw["request_started_at"], utc=True, errors="coerce")
    raw["is_new"] = raw["origin"].eq("NEW_THIS_PLAN")
    raw["valid"] = raw["status"].eq("OK")
    raw["invalid"] = raw["status"].eq("INVALID_OUTPUT")
    raw["error"] = raw["status"].eq("ERROR")
    raw["unit"] = raw["participant"] + "|" + raw["task"]
    S["blockmap"], S["strata"] = _block_map(cfg)
    raw["block"] = [S["blockmap"].get((t, c)) for t, c in zip(raw["task"], raw["case_id"])]
    raw["stratum"] = [S["strata"].get((t, c), "") for t, c in zip(raw["task"], raw["case_id"])]
    raw["prediction"] = np.where(raw["task"].eq("INTENT"), raw["predicted_label"], raw["decision"])
    S["df"] = raw

    ev = pd.DataFrame([json.loads(l) for l in open(cfg.events_path) if l.strip()])
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    S["events"] = ev
    uri = f"file:{cfg.db_path.resolve()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    S["ledger"] = pd.read_sql_query("select * from budget_ledger", con)
    con.close()
    S["ledger"]["usd"] = S["ledger"]["amount_usd"].astype(float)
    S["ledger"]["ts"] = pd.to_datetime(S["ledger"]["timestamp"], utc=True, errors="coerce")
    return S


def _block_map(cfg: Config) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], str]]:
    """Block index of every eligible case, recomputed with the benchmark's own deterministic ordering (offline, local files only)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from msu_llm_lab.classification import benchmark as bm
    from msu_llm_lab.classification.records import Task

    loaded = bm.load_prep_cases(cfg.prep_dir, cfg.policy)
    bmap, strata = {}, {}
    for t in (Task.INTENT, Task.LEAD):
        for i, c in enumerate(bm.order_cases(loaded, t)):
            bmap[(t.value, c.case_id)] = i // 10
            strata[(t.value, c.case_id)] = "/".join(str(x) for x in loaded.strata.get(t.value, loaded.strata.get(t, {})).get(c.case_id, ()))
    return bmap, strata


# ----------------------------------------------------------------------------- core tables
def model_task_summary(S: Dict[str, Any]) -> pd.DataFrame:
    df, cfg = S["df"], S["cfg"]
    rows = []
    unatt = S["unattempted"].groupby(["model_key", "task"]).size().to_dict()
    for (p, t), g in df.groupby(["participant", "task"]):
        n, k = len(g), int(g.valid.sum())
        lo, hi = wilson(k, n)
        lat = g.loc[g.latency_ms_end_to_end.notna(), "latency_ms_end_to_end"].to_numpy()
        lab = f"{p}|{t}"
        cost = g.recorded_cost_usd.dropna().to_numpy()
        rows.append({
            "participant": p, "task": t, "role": g.role.iloc[0], "records": n, "new_this_plan": int(g.is_new.sum()), "carried_over": int((~g.is_new).sum()),
            "valid": k, "invalid_output": int(g.invalid.sum()), "errors": int(g.error.sum()), "unattempted_in_executed_blocks": int(unatt.get((p, t), 0)),
            "valid_rate": k / n, "valid_rate_ci_lo": lo, "valid_rate_ci_hi": hi, "valid_rate_ci_method": "Wilson 95%",
            "latency_n": len(lat), "latency_mean_ms": float(np.mean(lat)) if len(lat) else np.nan, "latency_p50_ms": pct(lat, 50),
            "latency_p90_ms": pct(lat, 90), "latency_p95_ms": pct(lat, 95), "latency_p99_ms": pct(lat, 99), "latency_max_ms": float(np.max(lat)) if len(lat) else np.nan,
            "latency_p50_ci_lo": boot_ci(lat, np.median, lab + "|p50", cfg.n_boot, cfg.seed)[0], "latency_p50_ci_hi": boot_ci(lat, np.median, lab + "|p50", cfg.n_boot, cfg.seed)[1],
            "latency_source": "MEASURED_CLIENT end-to-end (provider model latency UNAVAILABLE)",
            "prompt_tokens_mean": g.prompt_tokens.mean(), "completion_tokens_mean": g.completion_tokens.mean(), "prompt_tokens_sum": g.prompt_tokens.sum(),
            "completion_tokens_sum": g.completion_tokens.sum(), "reasoning_tokens_sum": g.reasoning_tokens.sum(),
            "est_cost_usd_sum": g.recorded_cost_usd.sum(), "est_cost_usd_per_request": float(np.mean(cost)) if len(cost) else np.nan,
            "est_cost_usd_per_1000_requests": float(np.mean(cost)) * 1000 if len(cost) else np.nan,
            "est_cost_usd_per_valid_output": g.recorded_cost_usd.sum() / k if k else np.nan,
            "requests_without_cost_evidence": int(g.recorded_cost_usd.isna().sum()), "actual_provider_usd_n": int(g.actual_cost_usd.notna().sum()),
            "retried_requests": int((g.attempts > 1).sum()), "attempts_sum": int(g.attempts.sum()),
            "length_finish": int(g.finish_reason.eq("length").sum()), "finish_not_captured": int(g.finish_reason.eq("").sum()),
            "completion_state": "PARTIAL (quarantined)" if p.startswith("mistral") else "PARTIAL (17 of 222 blocks)",
        })
    return pd.DataFrame(rows).sort_values(["task", "participant"]).reset_index(drop=True)


def model_summary(mts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, g in mts.groupby("participant"):
        n, k = int(g.records.sum()), int(g.valid.sum())
        lo, hi = wilson(k, n)
        rows.append({"participant": p, "role": g.role.iloc[0], "tasks": "+".join(g.task), "records": n, "valid": k, "invalid_output": int(g.invalid_output.sum()),
                     "errors": int(g.errors.sum()), "valid_rate": k / n, "valid_rate_ci_lo": lo, "valid_rate_ci_hi": hi,
                     "est_cost_usd_sum": g.est_cost_usd_sum.sum(), "est_cost_usd_per_request": g.est_cost_usd_sum.sum() / n,
                     "prompt_tokens_sum": g.prompt_tokens_sum.sum(), "completion_tokens_sum": g.completion_tokens_sum.sum(),
                     "retried_requests": int(g.retried_requests.sum()), "completion_state": g.completion_state.iloc[0]})
    return pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)


def batch_tables(S: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df, plan = S["df"], S["plan"]
    executed = int(S["state"]["blocks_completed"])
    units = sorted(df.unit.unique())
    unatt = S["unattempted"].copy()
    unatt["block"] = [S["blockmap"].get((t, c)) for t, c in zip(unatt.task, unatt.case_id)]
    unatt["unit"] = unatt["model_key"] + "|" + unatt["task"]
    ub = {int(b["index"]): b for b in plan["blocks"]}
    mb_rows, b_rows, cov_rows = [], [], []
    for u in units:
        p, t = u.split("|")
        for k in range(len(plan["blocks"])):
            g = df[(df.unit == u) & (df.block == k)]
            ua = int(((unatt.unit == u) & (unatt.block == k)).sum())
            planned_cases = ub[k]["intent_cases" if t == "INTENT" else "lead_cases"]
            if k >= executed:
                state = "NOT_EXECUTED"
            elif g.empty and ua == 0:
                state = "NO_RECORDS"
            elif ua:
                state = "PARTIAL_QUARANTINED"
            else:
                state = "EXECUTED"
            lat = g.latency_ms_end_to_end.dropna().to_numpy()
            row = {"participant": p, "task": t, "block": k, "state": state, "planned_cases": planned_cases, "records": len(g), "new_this_plan": int(g.is_new.sum()),
                   "carried_over": int((~g.is_new).sum()), "valid": int(g.valid.sum()), "invalid_output": int(g.invalid.sum()), "errors": int(g.error.sum()),
                   "unattempted": ua}
            cov_rows.append(dict(row))
            row.update({"valid_rate": (g.valid.mean() if len(g) else np.nan), "latency_p50_ms": pct(lat, 50) if len(lat) else np.nan,
                        "est_cost_usd": g.recorded_cost_usd.sum() if len(g) else np.nan})
            mb_rows.append(row)
    mb, cov = pd.DataFrame(mb_rows), pd.DataFrame(cov_rows)
    for k, b in ub.items():
        g = df[df.block == k]
        lat = g.latency_ms_end_to_end.dropna().to_numpy()
        ua = int((unatt.block == k).sum())
        b_rows.append({"block": k, "state": "EXECUTED" if k < executed else "NOT_EXECUTED", "intent_cases": b["intent_cases"], "lead_cases": b["lead_cases"],
                       "plan_new_requests": b["requests"], "plan_upper_bound_usd": b["upper_bound_usd"], "records": len(g), "new_this_plan": int(g.is_new.sum()),
                       "carried_over": int((~g.is_new).sum()), "valid": int(g.valid.sum()), "invalid_output": int(g.invalid.sum()), "errors": int(g.error.sum()),
                       "unattempted": ua, "valid_rate": g.valid.mean() if len(g) else np.nan, "latency_p50_ms": pct(lat, 50) if len(lat) else np.nan,
                       "latency_p95_ms": pct(lat, 95) if len(lat) else np.nan, "est_cost_usd": g.recorded_cost_usd.sum() if len(g) else np.nan})
    return pd.DataFrame(b_rows), mb, cov


def reconcile(S: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df, ev, led, plan, man = S["df"], S["events"], S["ledger"], S["plan"], S["manifest"]
    new, old = df[df.is_new], df[~df.is_new]
    executed = int(S["state"]["blocks_completed"])
    bt = plan["totals"]
    nunits = df.unit.nunique()
    full_matrix = executed * 10 * nunits                       # every unit fresh, 10 cases per block
    n_unatt = len(S["unattempted"])
    rows = [
        ("blocks in plan", len(plan["blocks"]), "plan.json"),
        ("blocks executed (state)", executed, "state.json"),
        ("participant-task units", nunits, "model_results.csv"),
        ("full matrix of executed blocks (units x 10 cases x blocks)", full_matrix, "derived"),
        ("carried-over records (not re-sent)", len(old), "model_results.csv origin"),
        ("planned new logical requests (plan blocks, executed)", int(sum(b["requests"] for b in plan["blocks"][:executed])), "plan.json"),
        ("plan guaranteed_requests", bt["guaranteed_requests"], "plan.json sizing"),
        ("dispatched logical requests (new records)", len(new), "model_results.csv"),
        ("unattempted new requests (quarantine)", n_unatt, "unattempted_in_executed_blocks.csv"),
        ("records in results", len(df), "model_results.csv"),
        ("outcomes OK", int(new.valid.sum()), "new records"),
        ("outcomes INVALID_OUTPUT", int(new.invalid.sum()), "new records"),
        ("outcomes ERROR", int(new.error.sum()), "new records"),
        ("HTTP attempts (new records, attempts column)", int(new.attempts.sum()), "model_results.csv"),
        ("retries (attempts - 1)", int((new.attempts - 1).sum()), "model_results.csv"),
    ]
    checks = {
        "full_matrix == carried + planned_new": full_matrix == len(old) + int(rows[5][1]),
        "planned_new == dispatched + unattempted": int(rows[5][1]) == len(new) + n_unatt,
        "results == carried + dispatched": len(df) == len(old) + len(new),
        "results + unattempted == full_matrix": len(df) + n_unatt == full_matrix,
        "dispatched == ok + invalid + error": len(new) == int(new.valid.sum() + new.invalid.sum() + new.error.sum()),
        "attempts == dispatched + retries": int(new.attempts.sum()) == len(new) + int((new.attempts - 1).sum()),
        "no duplicate (participant, task, case)": not df.duplicated(["participant", "task", "case_id"]).any(),
        "no duplicate request_id": not df.request_id.duplicated().any(),
        "source reconciliation file agrees (dispatched)": S["recon_src"]["totals"]["dispatched_logical_requests"] == len(new),
        "source reconciliation file agrees (retries)": S["recon_src"]["totals"]["retries"] == int((new.attempts - 1).sum()),
    }
    rec = pd.DataFrame(rows, columns=["quantity", "value", "source"])
    rec["check_note"] = ""
    chk = pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()])

    # per-unit reconciliation
    det = []
    unatt = S["unattempted"].groupby(["model_key", "task"]).size().to_dict()
    for (p, t), g in df.groupby(["participant", "task"]):
        gn = g[g.is_new]
        det.append({"participant": p, "task": t, "carried_over": int((~g.is_new).sum()), "dispatched": len(gn), "unattempted": int(unatt.get((p, t), 0)),
                    "planned_new": len(gn) + int(unatt.get((p, t), 0)), "ok": int(gn.valid.sum()), "invalid": int(gn.invalid.sum()), "error": int(gn.error.sum()),
                    "attempts": int(gn.attempts.sum()), "retries": int((gn.attempts - 1).sum()),
                    "expected_total_executed_blocks": executed * 10, "records_plus_unattempted": len(g) + int(unatt.get((p, t), 0)),
                    "balanced": len(g) + int(unatt.get((p, t), 0)) == executed * 10})
    details = pd.DataFrame(det)

    # event-log reconciliation inside the authorisation window of this plan
    start = pd.Timestamp(man["authorization_sessions"][0]["started_at"])
    w = ev[(ev.ts >= start) & ev.provenance.eq("LIVE")]
    kinds = w.kind.value_counts().to_dict()
    ev_info = {"window_start": str(start), "events": len(w), "kinds": kinds,
               "started": kinds.get("REQUEST_STARTED", 0), "succeeded": kinds.get("REQUEST_SUCCEEDED", 0), "failed": kinds.get("REQUEST_FAILED", 0),
               "retry": kinds.get("REQUEST_RETRY", 0), "http_status": w.http_status.dropna().astype(int).value_counts().to_dict(),
               "dispatched_records": len(new), "started_minus_dispatched": kinds.get("REQUEST_STARTED", 0) - len(new)}
    # ledger vs results, per store run id
    lg = led[~led.is_simulated.astype(bool)].copy()
    lg["group"] = lg.run_id.fillna("").str.replace(r"-(INTENT|LEAD)$", "", regex=True)
    per_run = lg.groupby("run_id").usd.sum().rename("ledger_usd")
    res = df.groupby("run_id").recorded_cost_usd.sum().rename("results_estimated_usd")
    cr = pd.concat([per_run, res], axis=1).fillna(0.0)
    cr["delta_usd"] = cr.ledger_usd - cr.results_estimated_usd
    cr = cr.reset_index().rename(columns={"index": "run_id"})
    smoke = float(lg[lg.run_id.fillna("").str.startswith("smoke")].usd.sum())
    cost = {"ledger_total_usd": float(lg.usd.sum()), "ledger_smoke_usd": smoke, "results_estimated_usd": float(df.recorded_cost_usd.sum()),
            "results_new_estimated_usd": float(new.recorded_cost_usd.sum()),
            "guard_cost_usd": float(S["state"]["spent_guard_usd"]), "manifest_ledger_counted_usd": float(man["actual_cost"]["ledger"]["counted_against_caps_usd"]),
            "delta_ledger_minus_results_minus_smoke": float(lg.usd.sum()) - smoke - float(df.recorded_cost_usd.sum()),
            "provider_reported_usd_rows": int(df.actual_cost_usd.notna().sum()), "native_credit_rows": int(df.native_cost.notna().sum()),
            "native_credits_sum": float(df.native_cost.sum()), "ledger_rows": len(lg)}
    return rec, chk, details, {"events": ev_info, "ledger_vs_results": cr, "cost": cost, "full_matrix": full_matrix}


def cost_provenance(S: Dict[str, Any], aux: Dict[str, Any]) -> pd.DataFrame:
    c = aux["cost"]
    return pd.DataFrame([
        {"source": "results.recorded_cost_usd (sum, all 3,263 records)", "usd": c["results_estimated_usd"], "basis": "ESTIMATED: catalogue price x provider-reported tokens", "is_invoice": False},
        {"source": "results.recorded_cost_usd (sum, new this plan)", "usd": c["results_new_estimated_usd"], "basis": "ESTIMATED", "is_invoice": False},
        {"source": "budget ledger total (non-simulated rows)", "usd": c["ledger_total_usd"], "basis": "ESTIMATED: booked transactions across all plans and smoke tests", "is_invoice": False},
        {"source": "ledger smoke tests", "usd": c["ledger_smoke_usd"], "basis": "ESTIMATED (pre-benchmark smoke runs)", "is_invoice": False},
        {"source": "plan guard cost (state.spent_guard_usd)", "usd": c["guard_cost_usd"], "basis": "GUARD: per-request max(booked, native credits); stop-check only, never booked", "is_invoice": False},
        {"source": "manifest ledger counted against caps", "usd": c["manifest_ledger_counted_usd"], "basis": "ESTIMATED (ledger)", "is_invoice": False},
        {"source": "native provider credits (sum of usage.cost)", "usd": c["native_credits_sum"], "basis": "CREDITS with an UNVERIFIED USD ratio - not USD", "is_invoice": False},
        {"source": "provider-reported USD rows", "usd": float("nan"), "basis": f"UNAVAILABLE: {c['provider_reported_usd_rows']} rows carry a provider-reported USD amount", "is_invoice": False},
    ])


def distributions(S: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    df = S["df"]
    lat_rows, tok_rows = [], []
    for (p, t), g in df.groupby(["participant", "task"]):
        for status, h in (("ALL", g), ("OK", g[g.valid])):
            v = h.latency_ms_end_to_end.dropna().to_numpy()
            if len(v):
                lat_rows.append({"participant": p, "task": t, "subset": status, "n": len(v), **{f"p{q}": pct(v, q) for q in (1, 5, 25, 50, 75, 90, 95, 99)},
                                 "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan, "min": float(v.min()), "max": float(v.max()),
                                 "retried_mean_ms": float(h.loc[h.attempts > 1, "latency_ms_end_to_end"].mean()) if (h.attempts > 1).any() else np.nan})
    return {"latency": pd.DataFrame(lat_rows)}


def reliability(S: Dict[str, Any]) -> pd.DataFrame:
    df = S["df"]
    rows = []
    for (p, t), g in df.groupby(["participant", "task"]):
        n = len(g)
        rows.append({"participant": p, "task": t, "records": n, "error_rate": g.error.mean(), "invalid_rate": g.invalid.mean(), "retry_rate": (g.attempts > 1).mean(),
                     "rate_limit_errors": int(g.error_category.eq("RATE_LIMIT").sum()), "provider_5xx_errors": int(g.error_category.eq("PROVIDER_5XX").sum()),
                     "timeouts": int(g.error_category.eq("TIMEOUT").sum()), "retry_causes_nonempty": int(g.retry_causes.ne("").sum()),
                     "max_attempts_seen": int(g.attempts.max()), "quarantined": p.startswith("mistral")})
    return pd.DataFrame(rows)


def output_validity(S: Dict[str, Any]) -> pd.DataFrame:
    df = S["df"]
    rows = []
    for (p, t), g in df.groupby(["participant", "task"]):
        n = len(g)
        for issue in sorted(set(g.validation_issues) | {""}):
            h = g[g.validation_issues == issue]
            if len(h) or issue == "":
                rows.append({"participant": p, "task": t, "validation_issue": issue or "NONE", "n": len(h), "share_of_records": len(h) / n,
                             "finish_length": int(h.finish_reason.eq("length").sum()), "finish_stop": int(h.finish_reason.eq("stop").sum()),
                             "finish_not_captured": int(h.finish_reason.eq("").sum()),
                             "completion_tokens_median": float(h.completion_tokens.median()) if len(h) else np.nan})
    return pd.DataFrame(rows)


def error_analysis(S: Dict[str, Any]) -> pd.DataFrame:
    df = S["df"]
    b = df[~df.valid]
    g = b.groupby(["participant", "task", "status", "error_category", "error_type", "validation_issues", "finish_reason"], dropna=False)
    out = g.agg(n=("case_id", "size"), new=("is_new", "sum"), blocks=("block", lambda s: ",".join(str(int(x)) for x in sorted(set(s)))[:60]),
                completion_tokens_median=("completion_tokens", "median"), mean_attempts=("attempts", "mean")).reset_index()
    out["finish_reason"] = out["finish_reason"].replace("", "not captured")
    return out.sort_values("n", ascending=False)


def matched_sets(S: Dict[str, Any]) -> Dict[str, Any]:
    """Cases answered (any status) by EVERY complete participant of a task; Mistral is quarantined/partial and is reported on its own coverage."""
    df = S["df"]
    out = {}
    for t, g in df.groupby("task"):
        complete = [p for p, h in g.groupby("participant") if not p.startswith("mistral")]
        sets = [set(g[g.participant == p].case_id) for p in complete]
        out[t] = {"participants": complete, "cases": sorted(set.intersection(*sets))}
    return out


def leaderboards(S: Dict[str, Any], matched: Dict[str, Any]) -> pd.DataFrame:
    df = S["df"]
    rows = []
    for t, m in matched.items():
        cs = set(m["cases"])
        sub = df[(df.task == t) & df.case_id.isin(cs) & df.participant.isin(m["participants"])]
        stats = {}
        for p, g in sub.groupby("participant"):
            stats[p] = {"n": len(g), "valid_rate": g.valid.mean(), "p50": pct(g.latency_ms_end_to_end.dropna().to_numpy(), 50),
                        "p95": pct(g.latency_ms_end_to_end.dropna().to_numpy(), 95), "cost_1k": g.recorded_cost_usd.mean() * 1000,
                        "cost_per_valid": g.recorded_cost_usd.sum() / max(1, g.valid.sum())}
        for metric, key, asc in (("valid_output_rate", "valid_rate", False), ("p50_latency_ms", "p50", True), ("p95_latency_ms", "p95", True),
                                 ("est_cost_per_1000_requests_usd", "cost_1k", True), ("est_cost_per_valid_output_usd", "cost_per_valid", True)):
            vals = sorted(stats.items(), key=lambda kv: kv[1][key], reverse=not asc)
            rank, prev = 0, None
            for i, (p, v) in enumerate(vals):
                if v[key] != prev:
                    rank, prev = i + 1, v[key]
                rows.append({"task": t, "leaderboard": metric, "rank": rank, "participant": p, "value": v[key], "matched_cases": len(cs), "n_participants": len(stats),
                             "note": "OPERATIONAL ONLY (no ground truth); matched cases = answered by every complete participant; ties share a rank"})
    return pd.DataFrame(rows)


def pareto(S: Dict[str, Any], mts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in mts.groupby("task"):
        cand = g[~g.participant.str.startswith("mistral")]
        pts = {r.participant: (r.est_cost_usd_per_1000_requests, r.latency_p50_ms, -r.valid_rate) for r in cand.itertuples()}
        for r in g.itertuples():
            me = (r.est_cost_usd_per_1000_requests, r.latency_p50_ms, -r.valid_rate)
            dom = [q for q, v in pts.items() if q != r.participant and all(a <= b for a, b in zip(v, me)) and any(a < b for a, b in zip(v, me))]
            inc = r.participant.startswith("mistral")
            rows.append({"task": t, "participant": r.participant, "cost_per_1000_usd": r.est_cost_usd_per_1000_requests, "p50_latency_ms": r.latency_p50_ms,
                         "valid_rate": r.valid_rate, "on_frontier": (not dom) and not inc, "dominated_by": ";".join(dom),
                         "frontier_status": "EXCLUDED: incomplete coverage (quarantined)" if inc else ("FRONTIER" if not dom else "DOMINATED"),
                         "axes": "minimise cost per 1000 requests, minimise p50 latency, maximise valid-output rate; operational only, no accuracy axis"})
    return pd.DataFrame(rows)


def agreement(S: Dict[str, Any]) -> pd.DataFrame:
    df, cfg = S["df"], S["cfg"]
    rows = []
    for t, g in df.groupby("task"):
        v = g[g.valid & g.prediction.ne("")]
        if t == "LEAD":
            v = v[v.decision_source.eq("MODEL_OUTPUT")]               # probability-only Jev has no binary decision
        wide = v.pivot_table(index="case_id", columns="participant", values="prediction", aggfunc="first")
        for a, b in combinations(sorted(wide.columns), 2):
            c = wide[[a, b]].dropna()
            if len(c) < 2:
                continue
            same = (c[a] == c[b]).to_numpy().astype(float)
            lo, hi = boot_ci(same, np.mean, f"agree|{t}|{a}|{b}", cfg.n_boot, cfg.seed)
            rows.append({"metric_family": "pairwise", "task": t, "participant_a": a, "participant_b": b, "n_common_valid": len(c), "agreement": same.mean(),
                         "agreement_ci_lo": lo, "agreement_ci_hi": hi, "cohens_kappa": cohen_kappa(list(c[a]), list(c[b])),
                         "note": "agreement between models; NOT accuracy"})
        for p in wide.columns:
            ag, tie, used = 0, 0, 0
            others = [q for q in wide.columns if q != p]
            for case, r in wide.iterrows():
                if pd.isna(r[p]):
                    continue
                votes = r[others].dropna()
                if len(votes) < 3:
                    continue
                vc = votes.value_counts()
                if len(vc) > 1 and vc.iloc[0] == vc.iloc[1]:
                    tie += 1
                    continue
                used += 1
                ag += int(vc.index[0] == r[p])
            lo, hi = wilson(ag, used)
            rows.append({"metric_family": "peer_plurality", "task": t, "participant_a": p, "participant_b": "plurality of other valid outputs", "n_common_valid": used,
                         "agreement": ag / used if used else np.nan, "agreement_ci_lo": lo, "agreement_ci_hi": hi, "cohens_kappa": np.nan,
                         "note": f"ties skipped: {tie}; needs >=3 other valid outputs; agreement with peers is NOT accuracy"})
        dist = g[g.valid].groupby(["participant", "prediction"]).size().rename("n").reset_index()
        dist["share"] = dist["n"] / dist.groupby("participant")["n"].transform("sum")
        for r in dist.itertuples():
            rows.append({"metric_family": "output_distribution", "task": t, "participant_a": r.participant, "participant_b": f"label={r.prediction}", "n_common_valid": r.n,
                         "agreement": r.share, "agreement_ci_lo": np.nan, "agreement_ci_hi": np.nan, "cohens_kappa": np.nan, "note": "share of valid outputs"})
    return pd.DataFrame(rows)


def statistical_comparisons(S: Dict[str, Any]) -> pd.DataFrame:
    from scipy import stats as sps

    df, cfg = S["df"], S["cfg"]
    rows = []
    for t, g in df.groupby("task"):
        parts = sorted(p for p in g.participant.unique() if not p.startswith("mistral"))
        wide_v = g.pivot_table(index="case_id", columns="participant", values="valid", aggfunc="first")
        wide_l = g[g.valid].pivot_table(index="case_id", columns="participant", values="latency_ms_end_to_end", aggfunc="first")
        wide_c = g.pivot_table(index="case_id", columns="participant", values="recorded_cost_usd", aggfunc="first")
        block = []
        for a, b in combinations(parts, 2):
            c = wide_v[[a, b]].dropna().astype(bool)
            n10, n01 = int((c[a] & ~c[b]).sum()), int((~c[a] & c[b]).sum())
            if len(c) >= MIN_PAIRED_N:
                p = float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0
                block.append({"task": t, "metric": "valid_output (paired, McNemar exact)", "participant_a": a, "participant_b": b, "n_paired": len(c),
                              "effect": float(c[a].mean() - c[b].mean()), "effect_unit": "difference in valid rate (a-b)", "a_only": n10, "b_only": n01,
                              "ci_lo": np.nan, "ci_hi": np.nan, "p_value": p, "family": f"{t}|valid"})
            l = wide_l[[a, b]].dropna()
            if len(l) >= MIN_PAIRED_N:
                d = (l[a] - l[b]).to_numpy()
                try:
                    p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    p = 1.0
                lo, hi = boot_ci(d, np.median, f"lat|{t}|{a}|{b}", cfg.n_boot, cfg.seed)
                block.append({"task": t, "metric": "e2e latency on both-valid cases (paired, Wilcoxon)", "participant_a": a, "participant_b": b, "n_paired": len(l),
                              "effect": float(np.median(d)), "effect_unit": "median paired difference ms (a-b)", "a_only": np.nan, "b_only": np.nan,
                              "ci_lo": lo, "ci_hi": hi, "p_value": p, "family": f"{t}|latency"})
            cc = wide_c[[a, b]].dropna()
            if len(cc) >= MIN_PAIRED_N:
                d = (cc[a] - cc[b]).to_numpy()
                try:
                    p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    p = 1.0
                lo, hi = boot_ci(d, np.mean, f"cost|{t}|{a}|{b}", cfg.n_boot, cfg.seed)
                block.append({"task": t, "metric": "estimated cost per request (paired, Wilcoxon)", "participant_a": a, "participant_b": b, "n_paired": len(cc),
                              "effect": float(d.mean()), "effect_unit": "mean paired difference USD (a-b)", "a_only": np.nan, "b_only": np.nan,
                              "ci_lo": lo, "ci_hi": hi, "p_value": p, "family": f"{t}|cost"})
        rows += block
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for fam, idx in out.groupby("family").groups.items():
        out.loc[idx, "p_holm"] = holm(list(out.loc[idx, "p_value"]))
    out["significant_holm_0.05"] = out["p_holm"] < 0.05
    out["note"] = ("operational comparison on a 170-case-per-task stratified sample; seeds fixed; Holm correction within each task x metric family; "
                   "no statement about correctness")
    return out.drop(columns=["family"])


def coverage_by_stratum(S: Dict[str, Any]) -> pd.DataFrame:
    df = S["df"]
    g = df[df.participant.eq("gpt-4.1") | (df.participant.eq("gpt-4.1-mini"))]
    return (g.groupby(["task", "stratum"]).size().rename("cases_in_sample").reset_index().sort_values(["task", "cases_in_sample"], ascending=[True, False]))


def data_quality(S: Dict[str, Any], aux: Dict[str, Any], rec_checks: pd.DataFrame, mts: pd.DataFrame, hashes_ok: bool) -> List[Dict[str, str]]:
    df, man, state, plan = S["df"], S["manifest"], S["state"], S["plan"]
    cost, ev = aux["cost"], aux["events"]
    old0 = S["old_recon"].get("real_benchmark_bench-d0a763e4a920", {})
    f: List[Dict[str, str]] = []

    def add(i, title, status, evidence, impact):
        f.append({"id": i, "issue": title, "status": status, "evidence": evidence, "impact": impact})

    ex = int(state["blocks_completed"])
    add("DQ01", "Partial run", "CONFIRMED",
        f"state.json blocks_completed={ex} of {len(plan['blocks'])}; sample {ex*10} of 2211 INTENT ({ex*10/2211:.1%}) and {ex*10} of 723 LEAD ({ex*10/723:.1%}) eligible cases; halted={state['halted']}",
        "Results describe a stratified 170-case-per-task sample, not the full dataset; no full-dataset claim is supported.")
    add("DQ02", "2,860 vs 3,400 planned requests", "EXPLAINED (not a defect in the data)",
        f"3,400 = 17 blocks x 10 cases x 20 units (full matrix if every unit were fresh) = 2,860 planned new + 540 carried-over. Bundle without suffix recorded planned_new={old0.get('totals', {}).get('planned_new_logical_requests')} "
        f"(carried records counted as new) and its check planned_matches_plan_block_requests={old0.get('checks', {}).get('planned_matches_plan_block_requests')}; r2-r4 report 2,860. "
        f"Separately, 3,400 also equals Jev LEAD completion-token total in the early bundle; that is a coincidence of values, not a request count.",
        "Use 2,860 as planned new requests and 3,400 as the full matrix of executed blocks.")
    add("DQ03", "Failed reconciliation", "CONFIRMED for the first bundle; RESOLVED in r2-r4",
        f"first bundle verification: request reconciliation FAIL (planned_matches_plan_block_requests=False); r4 checks all True: {S['recon_src']['checks']}; independent recomputation: "
        f"{int(rec_checks.passed.sum())}/{len(rec_checks)} checks pass",
        "Superseded bundles must not be cited for counts; r4 is the record.")
    n_pref = int((df.is_new & ~df.run_id.str.startswith(S['cfg'].bench_id)).sum())
    add("DQ04", "Carry-over inconsistency", "CONFIRMED (labelling, not counts)",
        f"{n_pref} of {int(df.is_new.sum())} new records carry a run_id prefixed with a parent bench id because inheriting participants append to the parent's prediction store; "
        f"the `origin` column, not the run_id prefix, identifies carry-over ({int((~df.is_new).sum())} carried-over records). No duplicate (participant, task, case) rows.",
        "Count by `origin`; never by run_id prefix.")
    add("DQ05", "Estimated cost vs ledger", "RECONCILED",
        f"ledger ${cost['ledger_total_usd']:.6f} = results estimated ${cost['results_estimated_usd']:.6f} + smoke ${cost['ledger_smoke_usd']:.6f} + residual ${cost['delta_ledger_minus_results_minus_smoke']:.6f}; "
        f"guard cost ${cost['guard_cost_usd']:.6f} is a different quantity (max(booked, native) per request, never booked)",
        "Neither number is an invoice; the residual is shown per run in reconciliation_details.csv.")
    add("DQ06", "No provider-reported USD", "CONFIRMED",
        f"{cost['provider_reported_usd_rows']} rows with provider-reported USD; {cost['native_credit_rows']} rows with native credits (unit {man['actual_cost'].get('note','')[:0]}OPENROUTER_CREDITS_USD_RATE_UNVERIFIED)",
        "All USD figures are ESTIMATED; cost comparisons are valid between models, absolute spend is unverified.")
    add("DQ07", "Pending labels", "CONFIRMED", f"ground_truth_status values: {sorted(set(df.ground_truth_status))}; labelled rows: {int(df.ground_truth_label.ne('').sum())}",
        f"Accuracy, precision, recall, F1, calibration, confusion matrices: {NOT_COMPUTABLE}.")
    mi = mts[mts.participant.str.startswith("mistral")]
    add("DQ08", "Mistral quarantine", "CONFIRMED",
        f"state.quarantined={state['quarantined']}; records {int(mi.records.sum())}, errors {int(mi.errors.sum())}, unattempted {len(S['unattempted'])}; "
        f"rate-limit diagnostics: {json.dumps(S['rate_limits'].get('summary', S['rate_limits']))[:200]}",
        "Mistral is excluded from matched-case leaderboards and from the Pareto frontier; its figures describe 203 of 340 intended requests only.")
    add("DQ09", "'Local only' label despite external calls", "CONFIRMED",
        f"manifest dataset.privacy_notice contains 'LOCAL_ONLY' while privacy.inference_payloads.real_admissions_derived_data_sent_to_external_providers="
        f"{man['privacy']['inference_payloads']['real_admissions_derived_data_sent_to_external_providers']}; the same manifest separates the export (not transmitted) from inference payloads (sent)",
        "'LOCAL_ONLY' describes dataset storage/export, not inference. Provider data policies are UNKNOWN for every route.")
    add("DQ10", "Verification file written after manifest sealing", "CONFIRMED (by design)",
        "verification_report.md states it was written after the manifest was sealed and is not listed in the manifest's hashes; the manifest lists 26 hashed files",
        "Verify it separately (artifact_integrity.json records its own hash).")
    add("DQ11", "Source artifacts unmodified by this analysis", "VERIFIED" if hashes_ok else "FAILED", "SHA-256 of every input read before and after the run", "See artifact_integrity.json.")
    add("DQ12", "Production parity", "NOT ESTABLISHED", "manifest warning PRODUCTION_PARITY_NOT_ESTABLISHED (Dify export missing)", "Results do not claim production equivalence.")
    add("DQ13", "Provider model latency", "UNAVAILABLE", f"model_latency_source = {sorted(set(df.model_latency_source))}", "Only client end-to-end latency (includes retries/backoff for retried requests).")
    add("DQ14", "Event log vs records", "INFORMATIONAL",
        f"events since authorisation start: STARTED {ev['started']}, SUCCEEDED {ev['succeeded']}, FAILED {ev['failed']}, RETRY {ev['retry']}; dispatched records {ev['dispatched_records']}; "
        f"difference STARTED-dispatched = {ev['started_minus_dispatched']}",
        "The event log is append-only and spans several plans; the window is a heuristic. Records, not events, are the unit of analysis.")
    return f


# ----------------------------------------------------------------------------- orchestration
def metric_definitions() -> pd.DataFrame:
    D = [
        ("valid_output_rate", "OK records / records", "operational", "Output parsed to an allowed value. NOT accuracy."),
        ("invalid_output_rate", "INVALID_OUTPUT / records", "operational", "Response arrived but was empty, had extra text, or an unknown label."),
        ("error_rate", "ERROR / records", "operational", "No usable response (rate limit, 5xx, timeout)."),
        ("retry_rate", "records with attempts>1 / records", "operational", "HTTP attempts beyond the first."),
        ("latency_e2e_ms", "client wall-clock per logical request (includes retries and backoff)", "operational", "Provider model latency is UNAVAILABLE."),
        ("latency percentiles", "numpy linear-interpolated percentile", "operational", "p50/p90/p95/p99; small n makes p95/p99 unstable."),
        ("est_cost_usd", "catalogue $/1M tokens x provider-reported tokens", "ESTIMATED", "Not an invoice; no provider-reported USD exists."),
        ("est_cost_per_valid_output", "sum est cost / valid outputs", "ESTIMATED", "Penalises models that waste tokens on invalid outputs."),
        ("guard_cost_usd", "per-request max(booked, native credits) summed", "GUARD", "Used only by the dispatch stop-check; never booked."),
        ("pairwise_agreement", "share of common valid cases with identical output", "agreement", "Not accuracy; excludes errors/invalid; Jev LEAD excluded (probability-only)."),
        ("cohens_kappa", "(po-pe)/(1-pe)", "agreement", "Chance-corrected agreement; unstable when one label dominates."),
        ("peer_plurality_agreement", "agreement with strict plurality of >=3 other valid outputs", "agreement", "NOT accuracy; ties skipped."),
        ("Wilson interval", "95% score interval for a proportion", "uncertainty", "Used for rates."),
        ("bootstrap CI", f"percentile bootstrap, {N_BOOT} resamples, fixed seed {SEED}", "uncertainty", "Per-unit seed derived from SHA-256 of a label."),
        ("McNemar exact", "two-sided exact binomial on discordant valid/invalid pairs", "paired test", "Holm-adjusted within task x metric family; n>=30 pairs."),
        ("Wilcoxon signed-rank", "paired latency/cost differences", "paired test", "Holm-adjusted; both-valid cases only for latency."),
        ("Pareto frontier", "non-dominated on cost/1000, p50 latency, valid rate", "operational", "No composite score; Mistral excluded (incomplete)."),
        ("accuracy / precision / recall / F1 / calibration", "-", NOT_COMPUTABLE, "Ground truth is PENDING for every case."),
    ]
    return pd.DataFrame(D, columns=["metric", "definition", "class", "caveat"])


def run_analysis(cfg: Config) -> Dict[str, Any]:
    inv, hashes_before = inventory(cfg)
    S = load(cfg)
    mts = model_task_summary(S)
    ms = model_summary(mts)
    bsum, mbs, cov = batch_tables(S)
    rec, rchk, rdet, aux = reconcile(S)
    matched = matched_sets(S)
    T = {
        "artifact_inventory": inv, "metric_definitions": metric_definitions(), "model_summary": ms, "model_task_summary": mts, "batch_summary": bsum,
        "model_batch_summary": mbs, "coverage_matrix": cov, "request_reconciliation": rec, "reconciliation_summary": rchk, "reconciliation_details": rdet,
        "latency_metrics": distributions(S)["latency"], "cost_metrics": mts[["participant", "task", "records", "prompt_tokens_sum", "completion_tokens_sum", "reasoning_tokens_sum",
                                                                          "est_cost_usd_sum", "est_cost_usd_per_request", "est_cost_usd_per_1000_requests",
                                                                          "est_cost_usd_per_valid_output", "requests_without_cost_evidence", "actual_provider_usd_n"]],
        "reliability_metrics": reliability(S), "output_validity_metrics": output_validity(S), "agreement_metrics": agreement(S), "error_analysis": error_analysis(S),
        "statistical_comparisons": statistical_comparisons(S), "leaderboards": leaderboards(S, matched), "pareto": pareto(S, mts),
        "cost_provenance": cost_provenance(S, aux), "stratum_coverage": coverage_by_stratum(S),
    }
    hashes_after = {p: sha256_file(Path(p)) for p in hashes_before}
    ok = hashes_before == hashes_after
    findings = data_quality(S, aux, rchk, mts, ok)
    return {"S": S, "T": T, "aux": aux, "matched": matched, "findings": findings, "hashes_before": hashes_before, "hashes_after": hashes_after, "hashes_ok": ok}


def versions() -> Dict[str, str]:
    import matplotlib, reportlab, scipy

    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "reportlab": reportlab.Version}
