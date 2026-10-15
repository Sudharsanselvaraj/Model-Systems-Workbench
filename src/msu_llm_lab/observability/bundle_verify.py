"""Offline verification of a written benchmark bundle: hashes, row counts, retained predictions and errors, cost provenance, unattempted scope,
and agreement between the narrative report and the CSV files. Reads files only."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verify_bundle(bundle: Path, executed_case_ids: Optional[Dict[str, set]] = None) -> Dict[str, Any]:
    bundle = Path(bundle)
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    man = json.loads((bundle / "benchmark_manifest.json").read_text())
    bad = [n for n, h in man["files"].items() if not (bundle / n).exists() or hashlib.sha256((bundle / n).read_bytes()).hexdigest() != h]
    add("manifest hashes match every listed file", not bad, f"{len(man['files'])} files; mismatched: {bad}")
    try:
        results, summary, comp = _rows(bundle / "model_results.csv"), _rows(bundle / "model_summary.csv"), _rows(bundle / "model_comparison.csv")
        unatt = _rows(bundle / "unattempted_in_executed_blocks.csv")
        traces = [json.loads(l) for l in (bundle / "traces.jsonl").read_text().splitlines() if l.strip()]
        add("results, traces and manifest row counts agree", len(results) == len(traces) == man["counts"]["result_rows"] == man["counts"]["trace_records"], f"results {len(results)}, traces {len(traces)}, manifest {man['counts']['result_rows']}")
        add("comparison and disagreement counts match the manifest", len(comp) == man["counts"]["model_comparison_rows"] and sum(r["models_agree"] == "False" for r in comp) == man["counts"]["disagreement_cases"],
            f"comparison rows {len(comp)}, disagreements {sum(r['models_agree'] == 'False' for r in comp)}")
        keys = {p["key"] for p in man["participants"]}
        add("all manifest participants are in the summary", keys == {r["model_key"] for r in summary}, f"missing: {sorted(keys - {r['model_key'] for r in summary})}")
        # per unit statuses
        res_by = defaultdict(Counter)
        for r in results:
            res_by[(r["requested_model_id"], r["task"])][r["status"]] += 1
        key_map = {}
        for s in summary:
            key_map[(s["model_id"], s["task"])] = s
        mismatch = []
        for s in summary:
            c = res_by.get((s["model_id"], s["task"]), Counter())
            if (int(s["attempted"]), int(s["valid"]), int(s["invalid_output"]), int(s["errors"])) != (sum(c.values()), c["OK"], c["INVALID_OUTPUT"], c["ERROR"]):
                mismatch.append((s["model_key"], s["task"]))
        add("summary counts equal the result rows per model and task", not mismatch, f"mismatches: {mismatch}")
        # predictions and errors retained
        probs = []
        for r in results:
            if r["status"] == "OK" and r["task"] == "INTENT" and not r["predicted_label"]:
                probs.append(("intent-no-label", r["case_id"]))
            if r["status"] == "OK" and r["task"] == "LEAD" and r["binary_decision_available"] == "True" and r["predicted_need_cta"] == "":
                probs.append(("lead-no-boolean", r["case_id"]))
            if r["status"] == "OK" and r["task"] == "LEAD" and r["binary_decision_available"] != "True" and r["probability_only"] != "True":
                probs.append(("lead-undefined", r["case_id"]))
            if r["status"] != "OK" and (r["predicted_label"] or r["predicted_need_cta"] or r["decision"]):
                probs.append(("prediction-on-non-ok", r["case_id"]))
            if r["status"] == "ERROR" and (not r["error_category"] or r["failed_call"] != "True"):
                probs.append(("error-without-category", r["case_id"]))
        add("predictions retained for valid rows; none on failed or invalid rows; errors keep their category", not probs, f"{probs[:5]}")
        # unattempted scope
        kid = {s["model_key"]: s["model_id"] for s in summary}          # registry key -> requested model id (the results files name models by id)
        in_results = {(r["requested_model_id"], r["task"], r["case_id"]) for r in results}
        overlap = [u for u in unatt if (kid[u["model_key"]], u["task"], u["case_id"]) in in_results]
        add("no case is both unattempted and recorded", not overlap, f"overlap {len(overlap)}")
        if executed_case_ids is not None:
            outside = [u for u in unatt if u["case_id"] not in executed_case_ids.get(u["task"], set())]
            add("the unattempted list contains only cases of the executed blocks", not outside, f"outside the executed blocks: {len(outside)}")
        carried = {p["key"] for p in man["participants"] if str(p.get("origin", "")).startswith("CARRIED_OVER")}
        add("carried-over participants are not listed as unattempted", not [u for u in unatt if u["model_key"] in carried], f"carried: {sorted(carried)}")
        # cost provenance
        cost_bad = []
        for r in results:
            if r["actual_cost_usd"] and r["cost_basis"] != "PROVIDER_REPORTED":
                cost_bad.append(("actual-basis", r["case_id"]))
            if not r["actual_cost_usd"] and r["estimated_cost_usd"] and r["cost_basis"] not in ("ESTIMATED_FROM_CATALOG", "PROVIDER_REPORTED"):
                cost_bad.append(("estimated-basis", r["case_id"]))
            if r["status"] != "ERROR" and not (r["actual_cost_usd"] or r["estimated_cost_usd"] or r["native_cost"]) and r["cost_basis"] != "UNAVAILABLE":
                cost_bad.append(("no-cost-but-basis", r["case_id"]))
        add("cost basis is consistent with the cost fields (empty is never zero)", not cost_bad, f"{cost_bad[:5]}")
        est_mismatch = []
        for s in summary:
            n = sum(1 for r in results if r["requested_model_id"] == s["model_id"] and r["task"] == s["task"] and r["estimated_cost_usd"])
            if int(s["estimated_n"]) != n:
                est_mismatch.append((s["model_key"], s["task"]))
        add("summary estimated-cost counts equal the result rows", not est_mismatch, f"{est_mismatch}")
        no_ev = sum(1 for r in results if not (r["actual_cost_usd"] or r["estimated_cost_usd"] or r["native_cost"]))
        add("requests without cost evidence are counted in the summary", sum(int(s["requests_without_cost_evidence"]) for s in summary) == no_ev, f"summary {sum(int(s['requests_without_cost_evidence']) for s in summary)}, results {no_ev}")
        # raw text
        add("no raw output is exported", not any(r["raw_output"] for r in results) and not any((t["output"].get("raw_output")) for t in traces), "")
        add("privacy sections separate the export from the inference payloads", "export" in man["privacy"] and "inference_payloads" in man["privacy"], "")
        # narrative vs CSV
        md = (bundle / "comparison_report.md").read_text()
        m = re.search(r"Cases where at least two valid participants answered and differed: (\d+)", md)
        add("the report's disagreement count equals model_comparison.csv", bool(m) and int(m.group(1)) == man["counts"]["disagreement_cases"], f"report {m.group(1) if m else 'missing'}, csv {man['counts']['disagreement_cases']}")
        rel = {}
        section = md.split("## 2. Reliability", 1)[-1].split("## 3.", 1)[0]
        for line in section.splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 9 and cells[1] in ("INTENT", "LEAD") and cells[2].isdigit():
                rel[(cells[0], cells[1])] = tuple(int(x) for x in cells[2:6])
        name = {s["model_key"]: s["display_name"] for s in summary}
        rep_bad = [(s["model_key"], s["task"]) for s in summary if rel.get((name[s["model_key"]], s["task"])) not in (None, (int(s["attempted"]), int(s["valid"]), int(s["invalid_output"]), int(s["errors"])))]
        add("the report's reliability table equals model_summary.csv", bool(rel) and not rep_bad, f"{len(rel)} rows compared; mismatches {rep_bad}")
        recon = json.loads((bundle / "request_reconciliation.json").read_text())
        add("request reconciliation checks all hold", all(v in (True, None) for v in recon["checks"].values()), f"{recon['checks']}")
        add("matrix completeness is stated and matches the unattempted file", (not man["matrix_completeness"]["complete_for_executed_blocks"]) == bool(unatt or man["matrix_completeness"]["quarantined"]),
            man["matrix_completeness"]["statement"])
    except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
        add("bundle files are well-formed and readable", False, f"{type(e).__name__}: {e}")
    ok = all(c["ok"] for c in checks)
    return {"bundle": str(bundle), "ok": ok, "n_checks": len(checks), "failed": [c["check"] for c in checks if not c["ok"]], "checks": checks}


def verification_markdown(v: Dict[str, Any]) -> str:
    L = [f"# Bundle verification: {'PASS' if v['ok'] else 'FAIL'}", "", f"`{v['bundle']}`: {v['n_checks']} checks, {len(v['failed'])} failed.", "",
         "Written after the manifest was sealed, so this file is not listed in the manifest's hashes.", "", "| Check | Result | Detail |", "|---|---|---|"]
    L += [f"| {c['check']} | {'PASS' if c['ok'] else '**FAIL**'} | {c['detail']} |" for c in v["checks"]]
    return "\n".join(L) + "\n"
