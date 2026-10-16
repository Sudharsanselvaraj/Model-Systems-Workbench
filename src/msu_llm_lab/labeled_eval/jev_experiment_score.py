"""Scoring and selection for the Jev prompt experiment. Evaluator side only: it reads the protected answer key, so the executor-side module never does."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from msu_llm_lab.labeled_eval import master as M
from msu_llm_lab.labeled_eval import run as R
from msu_llm_lab.labeled_eval.jev_experiment import exp_dir, halves


def score(cfg: R.Cfg) -> Dict[str, Any]:
    key = pd.read_csv(cfg.private / R.VERSION / "answer_key.csv", dtype=str)
    key = key[key.task == "INTENT"].set_index("case_id").expected
    h = halves(cfg)
    classes = list(M.D.INTENT_LABELS)
    rows, preds, correct = [], {}, {}
    for name in ("V0", "V1", "V2", "V0-repeat"):
        f = Path(cfg.out) / R.VERSION / "jev_prompt_experiment" / "predictions" / f"{name}.jsonl"
        if not f.exists():
            continue
        recs = {r["case_id"]: r for r in map(json.loads, open(f))}
        for half, ids in h.items():
            ids = [c for c in ids if c in recs]
            if not ids:
                continue
            y = [key[c] for c in ids]
            p = [recs[c]["predicted_label"] if recs[c]["status"] == "OK" and recs[c].get("predicted_label") in classes else M.INVALID for c in ids]
            m = M.confusion(y, p, classes)
            agg = M.aggregate(m, classes)
            pc = {r["label"]: r for r in M.per_class(m, classes)}
            probs = [((recs[c].get("label_probs") or {}).get(key[c]) or 0.0) for c, pp in zip(ids, p) if pp != key[c]]
            rows.append({"variant": name, "half": half, "n": len(ids), "valid_output_rate": float(np.mean([x != M.INVALID for x in p])), "accuracy": agg["accuracy"], "macro_f1": agg["macro_f1"], "macro_recall": agg["macro_recall"],
                         "recall_HANDOFF": pc["HUMAN_HANDOFF_REQUEST"]["recall"], "recall_UPLOAD": pc["UPLOAD_STATUS_CHECK"]["recall"], "recall_OUT_OF_SCOPE": pc["OUT_OF_SCOPE"]["recall"], "recall_QUERY": pc["QUERY"]["recall"],
                         "support_UPLOAD": pc["UPLOAD_STATUS_CHECK"]["support"], "support_OUT_OF_SCOPE": pc["OUT_OF_SCOPE"]["support"], "query_prediction_share": float(np.mean([x == "QUERY" for x in p])),
                         "true_query_share": float(np.mean([x == "QUERY" for x in y])), "wrong_count": int(sum(a != b for a, b in zip(y, p))), "mean_prob_of_true_label_on_errors": float(np.mean(probs)) if probs else np.nan,
                         "latency_p50_ms": float(np.median([recs[c]["latency_ms"] for c in ids if recs[c].get("latency_ms")])), "estimated_cost_usd": float(sum(float(recs[c].get("estimated_cost_usd") or 0) for c in ids))})
            preds[(name, half)] = dict(zip(ids, p))
            correct[(name, half)] = pd.Series([a == b for a, b in zip(y, p)], index=ids)
    res = pd.DataFrame(rows)
    pw = []
    from scipy import stats as sps

    for half in ("dev", "heldout"):
        base = correct.get(("V0", half))
        for v in ("V1", "V2", "V0-repeat"):
            c = correct.get((v, half))
            if base is None or c is None or half == "dev" and v == "V0-repeat":
                continue
            ids = base.index.intersection(c.index)
            a, b = base[ids].values, c[ids].values
            n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
            pw.append({"half": half, "variant": v, "reference": "V0", "n_paired": len(ids), "v0_only_correct": n10, "variant_only_correct": n01, "p_value": float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0})
    pwd = pd.DataFrame(pw)
    selected = None
    dev = res[(res.half == "dev") & res.variant.isin(["V1", "V2"])]
    if len(dev) == 2:
        dv = dev.sort_values(["macro_f1", "valid_output_rate"], ascending=[False, False])
        selected = "V1" if (dv.iloc[0].macro_f1 == dv.iloc[1].macro_f1 and dv.iloc[0].valid_output_rate == dv.iloc[1].valid_output_rate) else dv.iloc[0].variant
    agree = None
    if ("V0", "heldout") in preds and ("V0-repeat", "heldout") in preds:
        a, b = preds[("V0", "heldout")], preds[("V0-repeat", "heldout")]
        agree = float(np.mean([a[c] == b[c] for c in a if c in b]))
    out = exp_dir(cfg)
    res.to_csv(out / "jev_prompt_experiment_results.csv", index=False)
    pwd.to_csv(out / "jev_prompt_experiment_paired.csv", index=False)
    md = ["# Jev prompt experiment results", "", "Selection rule (pre-registered): V1 versus V2 by dev-half macro-F1 only. Held-out results for the non-selected variant are exploratory.", "",
          res.round(3).to_markdown(index=False) if not res.empty else "no predictions yet", "", f"Selected on dev: {selected}", f"V0 versus V0-repeat prediction agreement on the held-out half (run-to-run noise): {agree}", "",
          pwd.round(4).to_markdown(index=False) if len(pwd) else "", "", "Caveat: the held-out half has 1 UPLOAD_STATUS_CHECK and 2 OUT_OF_SCOPE cases; Jev had only 3 held-out errors at baseline, so a held-out gain cannot be distinguished from noise."]
    (out / "jev_prompt_experiment_results.md").write_text("\n".join(md) + "\n")
    return {"results": res, "paired": pwd, "selected": selected, "agreement": agree}
