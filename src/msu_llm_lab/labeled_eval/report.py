"""Final results report for labeled-eval-v1 (PDF + Markdown), built only from the master tables and stored per-case predictions. Offline; sends nothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from msu_llm_lab.labeled_eval import data as D  # noqa: E402
from msu_llm_lab.labeled_eval import master as M  # noqa: E402
from msu_llm_lab.labeled_eval import run as R  # noqa: E402
from msu_llm_lab.observability.evaluation_document import table, to_markdown  # noqa: E402
from msu_llm_lab.observability.evaluation_redesign import to_pdf_redesign  # noqa: E402

PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150})
NOTE = "labeled-eval-v1: 120 Main Intent and 200 Lead cases; labels operator-attested; costs estimated; provider policies unknown."
SHORT = {"INTENT": {"GREETING": "GREET", "OUT_OF_SCOPE": "OOS", "UPLOAD_STATUS_CHECK": "UPLOAD", "HUMAN_HANDOFF_REQUEST": "HANDOFF", "QUERY": "QUERY", "<INVALID>": "INVALID"}}


def short(m: str) -> str:
    return m.split("/")[-1].replace("-instruct", "").replace("-chat-v3.1", "").replace("-flash-lite", "-fl")


def colors(models: List[str]) -> Dict[str, str]:
    return {m: PAL[i % len(PAL)] for i, m in enumerate(sorted(models))}


def _save(fig, out: Path, name: str) -> str:
    fig.text(0.01, 0.005, NOTE, fontsize=6, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out / f"{name}.png", metadata={"Software": None})
    plt.close(fig)
    return f"figures/{name}.png"


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that mix numbers with status strings (NOT_AVAILABLE, NOT_COMPUTED, INSUFFICIENT_DATA...) become numeric; the status strings become NaN and print as n/a."""
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            bad = df[c][conv.isna() & df[c].notna()].astype(str)
            if conv.notna().any() and bad.str.match(r"^(NOT_AVAILABLE|NOT_COMPUTED|INSUFFICIENT_DATA|REFERENCE_MODEL)").all():
                df[c] = conv
    return df


def load(cfg: R.Cfg) -> Dict[str, Any]:
    d = cfg.out / R.VERSION
    T = {"intent": pd.read_csv(d / "classifier_model_comparison_master_intent.csv"), "lead": pd.read_csv(d / "classifier_model_comparison_master_lead.csv"),
         "pw": pd.read_csv(d / "classifier_model_comparison_pairwise.csv"), "cm": pd.read_csv(d / "classifier_model_comparison_confusion_long.csv"),
         "sens": pd.read_csv(d / "classifier_model_comparison_sensitivity.csv"), "rec": pd.read_csv(d / "classifier_model_comparison_reconciliation.csv"),
         "all200": pd.read_csv(d / "classifier_model_comparison_lead_all200.csv"), "pred": pd.read_csv(d / "classifier_predictions.csv"),
         "thr": pd.read_csv(d / "classifier_threshold_sweep.csv"), "thr_test": pd.read_csv(d / "classifier_threshold_test_results.csv"),
         "man": json.loads((d / "classifier_evaluation_manifest.json").read_text()), "plan": json.loads((d / "plan.json").read_text()),
         "state": json.loads((cfg.private / R.VERSION / "exec_state.json").read_text()) if (cfg.private / R.VERSION / "exec_state.json").exists() else json.loads(Path("data/labeled_eval/labeled-eval-v1/exec_state.json").read_text())}
    for k in ("intent", "lead", "all200"):
        T[k] = _numeric(T[k])
    return T


# ------------------------------------------------------------------ figures
def figures(T: Dict[str, Any], out: Path) -> List[Dict[str, str]]:
    out = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    it, ld = T["intent"], T["lead"]
    lt = ld[ld.split == "test"].copy()
    allm = sorted(set(it.model_id) | set(ld.model_id))
    col = colors(allm)
    reg: List[Dict[str, str]] = []

    def add(name, title, cap, src, fig):
        reg.append({"figure_id": f"L{len(reg) + 1:02d}", "file": _save(fig, out, name), "title": title, "caption": cap, "source": src})

    g = it.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(g))
    ax.barh(y, g.macro_f1, color=[col[m] for m in g.model_id], xerr=np.clip([g.macro_f1 - g.macro_f1_95ci_lower, g.macro_f1_95ci_upper - g.macro_f1], 0, None), capsize=2, ecolor="#333333")
    ax.scatter(g.accuracy, y, marker="D", color="black", s=14, zorder=3, label="accuracy")
    ax.set_yticks(y, [short(m) for m in g.model_id])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("macro-F1 (bars, bootstrap 95% CI) and accuracy (diamonds), end-to-end")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Main Intent: quality (invalid outputs count as misses)")
    add("intent_quality", "Main Intent quality", "Macro-F1 with bootstrap 95% CI and accuracy for each model on all 120 cases.", "master_intent", fig)

    classes = list(D.INTENT_LABELS)
    rec = it.set_index("model_id")[[f"recall_{c}" for c in classes]]
    sup = [int(it[f"support_{c}"].iloc[0]) for c in classes]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(rec.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5), [f"{SHORT['INTENT'][c]}\nn={s}" for c, s in zip(classes, sup)])
    ax.set_yticks(range(len(rec)), [short(m) for m in rec.index])
    for i in range(rec.shape[0]):
        for j in range(5):
            v = rec.values[i, j]
            ax.text(j, i, "n/a" if v != v else f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Main Intent: per-class recall (4-case classes are unstable)")
    add("intent_recall_heatmap", "Per-class recall (Main Intent)", "Recall per class and model; UPLOAD_STATUS_CHECK and OUT_OF_SCOPE have only 4 cases each.", "master_intent", fig)

    cm = T["cm"][T["cm"].task == "INTENT"]
    models = list(it.sort_values("macro_f1", ascending=False).model_id)
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.4))
    labs = classes + ["<INVALID>"]
    for ax, mk in zip(axs.ravel(), models):
        x = cm[cm.model_id == mk].pivot_table(index="true_label", columns="predicted_label", values="count", aggfunc="sum").reindex(index=classes, columns=labs).fillna(0).values
        ax.imshow(x, cmap="Blues")
        ax.set_xticks(range(6), [SHORT["INTENT"][c][:3] for c in labs], fontsize=5, rotation=90)
        ax.set_yticks(range(5), [SHORT["INTENT"][c][:3] for c in classes], fontsize=5)
        for i in range(5):
            for j in range(6):
                if x[i, j]:
                    ax.text(j, i, int(x[i, j]), ha="center", va="center", fontsize=5.5, color="white" if x[i, j] > x.max() / 2 else "black")
        ax.set_title(short(mk), fontsize=7)
    fig.suptitle("Main Intent confusion matrices (rows true, columns predicted)", fontsize=9)
    add("intent_confusion_grid", "Intent confusion matrices", "Counts for every model; the INVALID column holds outputs that could not be used.", "confusion_long", fig)

    pr = T["pred"]
    iss = pr[pr.status == "INVALID_OUTPUT"].assign(cause=lambda d: d.validation_issues.fillna("UNKNOWN"))
    iss = iss.groupby(["task", "model_key", "cause"]).size().reset_index(name="n")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        s = iss[iss.task == t].pivot_table(index="model_key", columns="cause", values="n", fill_value=0)
        full = (it if t == "INTENT" else lt).set_index("model_id")
        v = full.valid_output_rate * full.sample_count
        base = pd.DataFrame({"valid": v.values}, index=full.index.map(lambda m: m))
        s.index = s.index.map(lambda m: m)
        tab = base.join(s, how="left").fillna(0)
        tab.index = [short(i) for i in tab.index]
        tab.plot(kind="barh", stacked=True, ax=ax, color=["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"][: tab.shape[1]], legend=(t == "INTENT"))
        ax.set_title(f"{t} ({'120' if t == 'INTENT' else '100 test'} cases per model)")
        ax.set_xlabel("cases")
    add("validity_stack", "Valid versus invalid outputs", "Valid outputs and the causes of invalid ones (EMPTY_OUTPUT, EXTRA_TEXT, UNKNOWN_LABEL).", "predictions", fig)

    g = lt.sort_values("cta_f1")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(g))
    for k, (c, lab) in enumerate((("cta_precision", "precision"), ("cta_recall", "recall"), ("cta_f1", "F1"))):
        ax.barh(y + (k - 1) * 0.26, g[c], height=0.26, label=lab)
    ax.set_yticks(y, [short(m) for m in g.model_id])
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_xlabel("positive class: need_cta = true (test half, 100 cases)")
    ax.set_title("Lead Qualifier: CTA precision, recall and F1")
    add("lead_quality", "Lead Qualifier quality", "CTA precision, recall and F1 on the held-out test half.", "master_lead", fig)

    dv = ld.pivot_table(index="model_id", columns="split", values="cta_f1")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    dv = dv.sort_values("test")
    y = np.arange(len(dv))
    ax.hlines(y, dv.dev, dv.test, color="#999999")
    ax.scatter(dv.dev, y, label="dev (100)", color="#1f77b4")
    ax.scatter(dv.test, y, label="test (100)", color="#d62728")
    ax.set_yticks(y, [short(m) for m in dv.index])
    ax.legend(frameon=False)
    ax.set_xlabel("CTA F1")
    ax.set_title("Lead: dev versus test (Jev's threshold was tuned on dev)")
    add("lead_dev_test", "Dev versus test CTA F1", "A large gap would indicate instability; Jev's dev value is optimistic by construction.", "master_lead", fig)

    thr = T["thr"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for c, lab in (("precision", "precision"), ("recall", "recall"), ("f1", "F1")):
        ax.plot(thr.threshold, thr[c], marker="o", ms=3, label=f"dev {lab}")
    tt = T["thr_test"]
    ax.axvline(float(lt[lt.model_id.str.contains("jev")].selected_decision_threshold.iloc[0]), color="black", ls="--", lw=1, label="frozen threshold")
    ax.axvline(0.5, color="#999999", ls=":", lw=1, label="0.5 reference")
    ax.set_xlabel("decision threshold on P(yes)")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.set_title("Jev 1.14: threshold sweep on the dev half only")
    add("jev_threshold_sweep", "Jev threshold sweep", "Precision, recall and F1 on the 100 dev cases for thresholds 0.05-0.95; the choice maximises dev F1 (ties to the lowest).", "threshold_sweep", fig)

    jv = pr[(pr.model_key == "typesafe/jev-1.14") & (pr.task == "LEAD")].copy()
    key = pd.read_csv(Path("data/labeled_eval/labeled-eval-v1/answer_key.csv"), dtype=str)
    jv = jv.merge(key[key.task == "LEAD"][["case_id", "expected"]].rename(columns={"expected": "truth"}), on="case_id")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bins = np.linspace(0, 1, 21)
    ax.hist([jv[jv.truth == "false"].yes_prob, jv[jv.truth == "true"].yes_prob], bins=bins, label=["not a lead", "lead"], color=["#7f7f7f", "#2ca02c"])
    ax.axvline(float(lt[lt.model_id.str.contains("jev")].selected_decision_threshold.iloc[0]), color="black", ls="--", lw=1)
    ax.set_xlabel("Jev P(yes), all 200 cases")
    ax.set_ylabel("cases")
    ax.legend(frameon=False)
    ax.set_title("Jev score distribution by true class (dashed: frozen threshold)")
    add("jev_scores", "Jev score distribution", "Scores separate the classes well (ROC-AUC 0.99), so the threshold matters little.", "predictions + answer key", fig)

    a = T["all200"].copy()
    a["label"] = a.model_id.map(short) + a.basis.map(lambda b: " (CV)" if "cross" in b else (" (0.5 ref)" if "0.5" in b else ""))
    a = a[a.model_id != "anthropic/claude-haiku-5.5"].sort_values("cta_f1")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(a.label, a.cta_f1, color=["#999999" if "ref" in l else "#1f77b4" for l in a.label])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CTA F1 on all 200 Lead cases")
    ax.set_title("Lead on all 200 cases (no split; Jev by 5-fold cross-validation)")
    add("lead_all200", "Lead on all 200 cases", "Out-of-fold decisions for Jev; every other model uses its own decisions. Claude Haiku (0.055) omitted from the plot.", "lead_all200", fig)

    pr2 = pr[pr.latency_ms.notna()]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=False)
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        g = pr2[pr2.task == t]
        order = list(g.groupby("model_key").latency_ms.median().sort_values().index)
        ax.boxplot([g[g.model_key == m].latency_ms for m in order], vert=False, tick_labels=[short(m) for m in order], showfliers=False)
        ax.set_xscale("log")
        ax.set_title(t)
        ax.set_xlabel("end-to-end latency ms (log)")
    add("latency_box", "Latency distributions", "Client-side wall time per request, retries included; whiskers 1.5 IQR, outliers hidden.", "predictions", fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (t, d) in zip(axs, (("INTENT", it), ("LEAD", lt))):
        g = d.sort_values("estimated_cost_per_1000_cases_usd")
        ax.barh([short(m) for m in g.model_id], g.estimated_cost_per_1000_cases_usd, color=[col[m] for m in g.model_id])
        ax.set_title(t)
        ax.set_xlabel("estimated USD per 1,000 cases")
    add("cost_bars", "Estimated cost per 1,000 cases", "Catalogue price x provider-reported tokens; not an invoice.", "master tables", fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (t, d, q) in zip(axs, (("INTENT", it, "macro_f1"), ("LEAD", lt, "cta_f1"))):
        for r in d.itertuples():
            ax.scatter(r.estimated_cost_per_1000_cases_usd, getattr(r, q), s=20 + 150 * (1 / (1 + r.p50_latency_ms / 800)), color=col[r.model_id])
            ax.annotate(short(r.model_id), (r.estimated_cost_per_1000_cases_usd, getattr(r, q)), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("estimated USD per 1,000 cases (log)")
        ax.set_ylabel("macro-F1" if t == "INTENT" else "CTA F1")
        ax.set_title(f"{t} (bubble size: faster is larger)")
    add("quality_vs_cost", "Quality versus cost", "Models in the top-left are cheaper and better.", "master tables", fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (t, d) in zip(axs, (("INTENT", it), ("LEAD", lt))):
        c = M.components(d.reset_index(drop=True), t)
        w = M.WEIGHTS[t]
        contrib = pd.DataFrame({k: c[k] * v for k, v in w.items()})
        contrib.index = [short(m) for m in d.model_id]
        contrib = contrib.loc[contrib.sum(axis=1).sort_values().index]
        contrib.plot(kind="barh", stacked=True, ax=ax, legend=(t == "INTENT"), fontsize=6)
        ax.set_title(f"{t}: weighted composite")
        ax.set_xlabel("score (analyst weights)")
        if t == "INTENT":
            ax.legend(fontsize=5, frameon=False)
    add("composite", "Composite score components", "Weighted contribution of each criterion; the weights are analyst assumptions, see the sensitivity table.", "master tables", fig)

    pw = T["pw"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        p = pw[(pw.task == t) & (pw.split == "test")]
        ms = sorted(set(p.model_a) | set(p.model_b))
        mat = pd.DataFrame(np.nan, index=ms, columns=ms)
        for r in p.itertuples():
            mat.loc[r.model_a, r.model_b] = r.accuracy_difference_a_minus_b
            mat.loc[r.model_b, r.model_a] = -r.accuracy_difference_a_minus_b
        ax.imshow(mat.values.astype(float), cmap="RdBu", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(ms)), [short(m) for m in ms], rotation=70, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(ms)), [short(m) for m in ms], fontsize=5.5)
        for r in p.to_dict("records"):
            if r["significant_at_0.05_after_holm"]:
                i, j = ms.index(r["model_a"]), ms.index(r["model_b"])
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
                ax.text(i, j, "*", ha="center", va="center", fontsize=9)
        ax.set_title(f"{t}: accuracy difference (row minus column)")
    add("pairwise", "Paired accuracy differences", "Blue: row model is better. * marks a difference that is significant after Holm correction (McNemar exact, invalid = incorrect).", "pairwise", fig)

    rt = pr.assign(retried=pr.attempts > 1).groupby(["model_key"]).agg(retried=("retried", "sum"), extra=("attempts", lambda s: int((s - 1).sum()))).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.bar([short(m) for m in rt.model_key], rt.retried, color="#17becf")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("requests retried")
    ax.set_title("Retried requests per model (timeouts 13, HTTP 502 and 529 one each)")
    add("retries", "Retries", "Requests that needed more than one HTTP attempt; 14 requests, 15 extra attempts.", "predictions", fig)
    return reg


# ------------------------------------------------------------------ sections
def P(t):
    return ("p", t)


def B(x):
    return ("bullets", x)


def FIG(reg, name):
    r = next(x for x in reg if x["file"].endswith(f"/{name}.png"))
    return ("fig", {"file": r["file"], "caption": f"{r['figure_id']}. {r['title']}. {r['caption']}"})


def _model_notes(it: pd.DataFrame, lt: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for r in it.itertuples():
        rows.setdefault(r.model_id, {"model": r.model_id, "intent": "not run", "lead": "not run"})["intent"] = f"macro-F1 {r.macro_f1:.3f}, valid {r.valid_output_rate:.0%}, {r.p50_latency_ms:.0f} ms, ${r.estimated_cost_per_1000_cases_usd:.3f}/1k"
    for r in lt.itertuples():
        rows.setdefault(r.model_id, {"model": r.model_id, "intent": "not run", "lead": "not run"})["lead"] = f"CTA F1 {r.cta_f1:.3f}, recall {r.cta_recall:.2f}, valid {r.valid_output_rate:.0%}, {r.p50_latency_ms:.0f} ms, ${r.estimated_cost_per_1000_cases_usd:.3f}/1k"
    return pd.DataFrame(list(rows.values())).sort_values("model")


def sections(T: Dict[str, Any], reg: List[Dict[str, str]]) -> List[Tuple[str, List[Any]]]:
    it, ld, rec, sens, pw, all200, man, st = T["intent"], T["lead"], T["rec"], T["sens"], T["pw"], T["all200"], T["man"], T["state"]
    lt = ld[ld.split == "test"]
    bi, bl = it[it.recommendation.str.startswith("BEST")].iloc[0], lt[lt.recommendation.str.startswith("BEST")].iloc[0]
    S: List[Tuple[str, List[Any]]] = []
    jev = lt[lt.model_id.str.contains("jev")].iloc[0]
    qw_l = lt[lt.model_id.str.contains("qwen")].iloc[0]
    jev_i = it[it.model_id.str.contains("jev")].iloc[0]
    random_i = sens[(sens.task == "INTENT") & sens.scenario.str.startswith("random")]
    S.append(("Executive summary", [
        P("This report gives the labelled results for the Main Intent and Lead Qualifier classifiers: 8 participants per task, 120 intent cases and 200 lead cases, 2,560 requests in all, with expected labels (an operator attestation) and a protected answer key. "
          "It replaces the earlier operational-only benchmark, which could not measure accuracy. The two tasks are analysed separately."),
        B([f"Main Intent: best observed candidate {bi.model_id.split('/')[-1]} (macro-F1 {bi.macro_f1:.3f}, accuracy {bi.accuracy:.3f}, valid {bi.valid_output_rate:.0%}). It wins at baseline weights; Jev 1.14 wins only when cost or latency dominates "
           f"(random weights: {', '.join(f'{r.winner_all_models.split(chr(47))[-1]} {r.winner_score:.0%}' for r in random_i.itertuples())}). Jev's Intent macro-F1 is {jev_i.macro_f1:.3f}: it found none of the four UPLOAD_STATUS_CHECK cases.",
           f"Lead Qualifier: best observed candidate {bl.model_id.split('/')[-1]} (CTA F1 {bl.cta_f1:.3f}, recall {bl.cta_recall:.2f}, precision {bl.cta_precision:.3f}, test half). Jev's frozen threshold is {bl.selected_decision_threshold}; ROC-AUC {bl.roc_auc:.3f}. It wins every weight scenario. "
           f"On all 200 cases (cross-validated threshold) Jev scores CTA F1 0.932 and Qwen 0.903.",
           "Reliability: Claude Haiku (Lead 15% valid) and, on Intent, DeepSeek (79%), Mistral (7%) and Haiku (68%) fall below the 90% valid-output floor under the fixed 16/24-token caps; they are flagged not recommendable as configured.",
           f"Run: {len(st['units'])} of {len(T['plan']['units_full'])} chunks, 2,560 requests, no errors, 14 requests retried, nothing quarantined. Estimated cost ${float(rec[rec.check.str.startswith('estimated cost')].observed.iloc[0]):.4f}; no provider-reported USD exists.",
           "Not established: a statistically significant winner (small samples), production approval, production parity, provider data policies, data origin."]),
        P("Reading guide: 'valid' means the output parsed to an allowed value; 'effective' metrics count an invalid output as a miss; 'valid-only' metrics use valid outputs only. Agreement between models is never used as accuracy.")]))
    S.append(("Scope, datasets, design and approvals", [
        B([f"Main Intent: {man['datasets']['intent']['rows']} cases, labels {man['datasets']['intent']['label_counts']}; history and current message are the model input. Lead Qualifier: {man['datasets']['lead']['rows']} cases (106 true, 94 false); `action_summary` only is the input and `user_msg` is never sent, per the prompt and repository contract.",
           "Splits: Intent has no tuned parameter, so all 120 cases are the evaluation set. Lead is split 100 dev / 100 test (53 positives each, identical action summaries kept together), used only to select and freeze Jev's threshold; all 200 are also scored without a split (Jev out-of-fold).",
           "Models: 9 keys (eight per task); Mistral ran last; GPT-4.1 and Claude Sonnet 5.5 were outside the approved cohort. Output caps 16 (Intent) and 24 (Lead) tokens, unchanged. Chunks of 10; retries up to 3 attempts.",
           "Labels: operator-attested as reviewed and correct; there is no second annotation, so reviewer agreement and Cohen's kappa are unavailable.",
           "Data origin: unverified. 114 of 120 intent messages and 196 of 200 lead messages equal prepared real-conversation messages, so the data is treated as CONFIDENTIAL and potentially real, never as synthetic or anonymised.",
           "Approvals: operator approval bound to plan 27b0c17efd03; processing permission recorded as an operator attestation (no document, approvers unnamed); typed live approval and typed real-data authorisation entered at run time. Provider data policies remain UNKNOWN for all nine routes."]),
        table(pd.DataFrame([{"participant": p["key"], "model": p["model_id"], "route": p["route"], "tasks": ",".join(p["tasks"])} for p in T["plan"]["participants"]]), ["participant", "model", "route", "tasks"], ["Participant", "Model id", "Pinned route", "Tasks"])]))
    S.append(("Execution, integrity and reconciliation", [
        P(f"All {len(T['plan']['units_full'])} chunks completed ({sum(1 for u in st['units'].values() if u['status'] == 'DONE')} DONE); halted: {st['halted']}; quarantined: {st['quarantined'] or 'none'}. Guard cost ${float(st['spent_guard_usd']):.4f} (the executor's stop-check quantity, not booked)."),
        table(rec, ["check", "expected_or_reference", "observed", "status", "note"], ["Check", "Reference", "Observed", "Status", "Note"], nd={"expected_or_reference": 6, "observed": 6}),
        P("The 14-versus-15 retry difference is explained: 14 requests were retried and 15 retry events exist because one request retried twice. Estimated cost equals the ledger's 2,560 rows; the guard cost is a different quantity. Answer-key separation, dataset hashes and the dev/test split were verified before and after the run."),
        FIG(reg, "retries")]))
    ic = ["model_id", "valid_output_rate", "accuracy", "accuracy_95ci_lower", "accuracy_95ci_upper", "macro_f1", "macro_f1_95ci_lower", "macro_f1_95ci_upper", "balanced_accuracy", "matthews_correlation_coefficient", "successful_case_accuracy"]
    S.append(("Main Intent results", [
        table(it.sort_values("macro_f1", ascending=False), ic, ["Model", "Valid", "Acc.", "CI lo", "CI hi", "Macro-F1", "CI lo", "CI hi", "Bal. acc.", "MCC", "Acc. valid only"]),
        P("Accuracy and macro-F1 are end-to-end: an invalid output is a miss. 'Acc. valid only' is the conditional view. Macro-F1 weights the five classes equally, so the two 4-case classes move it a lot."),
        FIG(reg, "intent_quality"), FIG(reg, "intent_recall_heatmap"),
        table(it.sort_values("macro_f1", ascending=False), ["model_id"] + [f"{m}_{c}" for c in D.INTENT_LABELS for m in ("precision", "recall")], ["Model"] + [f"{m[:4]}. {c[:6]}" for c in D.INTENT_LABELS for m in ("precision", "recall")], nd={}),
        FIG(reg, "intent_confusion_grid"),
        P("Common errors: HUMAN_HANDOFF_REQUEST is under-called (about 75-80% recall for the strong models); QUERY absorbs many of the misses; UPLOAD_STATUS_CHECK is rarely found. With only 4 cases each, UPLOAD_STATUS_CHECK and OUT_OF_SCOPE cannot support firm per-class conclusions.")]))
    lc = ["model_id", "valid_output_rate", "accuracy", "cta_precision", "cta_recall", "cta_f1", "cta_f1_95ci_lower", "cta_f1_95ci_upper", "specificity", "false_positive_rate", "false_negative_rate", "balanced_accuracy", "matthews_correlation_coefficient"]
    S.append(("Lead Qualifier results", [
        P("Positive class: need_cta = true. Test half (100 cases, 53 positive). A missing or invalid output on a positive case is a false negative."),
        table(lt.sort_values("cta_f1", ascending=False), lc, ["Model", "Valid", "Acc.", "Prec.", "Recall", "F1", "CI lo", "CI hi", "Spec.", "FPR", "FNR", "Bal. acc.", "MCC"]),
        FIG(reg, "lead_quality"), FIG(reg, "lead_dev_test"),
        table(ld[ld.split == "dev"].sort_values("cta_f1", ascending=False), ["model_id", "accuracy", "cta_precision", "cta_recall", "cta_f1"], ["Model (dev half)", "Acc.", "Prec.", "Recall", "F1"]),
        P("Dev rows are informational; for Jev they are optimistic because the threshold was chosen on them."),
        ("h2", "All 200 cases"), table(all200.sort_values("cta_f1", ascending=False), ["model_id", "basis", "accuracy", "cta_precision", "cta_recall", "cta_f1", "invalid_count"], ["Model", "Basis", "Acc.", "Prec.", "Recall", "F1", "Invalid"]),
        FIG(reg, "lead_all200")]))
    S.append(("Jev 1.14 threshold", [
        P(f"Jev returns only P(yes). The threshold was chosen on the 100 dev cases by maximising F1 on a 0.05 grid (ties to the lowest), frozen at {jev.selected_decision_threshold}, then applied unchanged to the test half. Flipping the test labels leaves it at 0.35 (verified). "
          f"Test result at the frozen threshold: recall {jev.cta_recall:.2f}, precision {jev.cta_precision:.3f}, F1 {jev.cta_f1:.3f}; ROC-AUC {jev.roc_auc:.3f}, PR-AUC {jev.pr_auc:.3f}, Brier {jev.brier_score:.3f}, ECE {jev.expected_calibration_error:.3f}."),
        table(T["thr_test"], ["rule", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1"], ["Rule (test half)", "Threshold", "TP", "FP", "FN", "TN", "Prec.", "Recall", "F1"]),
        FIG(reg, "jev_threshold_sweep"), FIG(reg, "jev_scores"),
        P("The F1 plateau is flat from about 0.3 to 0.7, so the exact threshold barely matters on this sample; the choice between higher recall (lower threshold) and higher precision is a business decision, and no cost ratio was supplied. With 100 cases per half, treat the threshold as preliminary.")]))
    S.append(("Output validity and reliability", [
        table(pd.concat([it.assign(task="INTENT"), lt.assign(task="LEAD")]), ["task", "model_id", "valid_output_rate", "invalid_output_rate", "empty_response_rate", "malformed_output_count", "request_failure_rate", "retried_request_count", "retry_count", "timeout_count"],
              ["Task", "Model", "Valid", "Invalid", "Empty", "Malformed", "Req. fail", "Retried req.", "Retries", "Timeouts"]),
        FIG(reg, "validity_stack"),
        B(["Claude Haiku: hidden reasoning tokens use up the 16/24-token cap (finish reason 'length'); 176 of its 200 Lead outputs were empty. This is a cap interaction, not a measure of classification ability; a changed cap would be a separate pre-registered experiment.",
           "DeepSeek Intent: 25 outputs had text around the label (stop finish). Mistral Intent: 112 of 120 outputs were invalid format, while its Lead output was fully valid; no HTTP 429 occurred this time.",
           "No request failed: all 2,560 produced a response. 14 requests were retried (13 timeouts, one HTTP 502, one HTTP 529)."])]))
    S.append(("Latency and tokens", [
        table(pd.concat([it.assign(task="INTENT"), lt.assign(task="LEAD")]), ["task", "model_id", "median_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms", "mean_latency_ms", "latency_standard_deviation_ms", "mean_input_tokens_per_case", "mean_output_tokens_per_case"],
              ["Task", "Model", "p50 ms", "p90", "p95", "p99", "Mean", "Std", "In tok", "Out tok"], nd={"median_latency_ms": 0, "p90_latency_ms": 0, "p95_latency_ms": 0, "p99_latency_ms": 0, "mean_latency_ms": 0, "latency_standard_deviation_ms": 0, "mean_input_tokens_per_case": 0, "mean_output_tokens_per_case": 1}),
        P("Latency is client-side wall time, retries included; provider-internal latency, throughput and first-attempt time of retried requests are not available. p99 needs at least 100 observations."),
        FIG(reg, "latency_box")]))
    S.append(("Cost and cost efficiency", [
        table(pd.concat([it.assign(task="INTENT"), lt.assign(task="LEAD")]), ["task", "model_id", "total_estimated_cost_usd", "estimated_cost_per_1000_cases_usd", "estimated_cost_per_valid_result_usd", "estimated_cost_per_correct_result_usd", "native_provider_credits_total"],
              ["Task", "Model", "Est. total USD", "USD / 1,000 cases", "USD / valid", "USD / correct", "Native credits"], nd={"total_estimated_cost_usd": 5, "estimated_cost_per_valid_result_usd": 5, "estimated_cost_per_correct_result_usd": 5, "native_provider_credits_total": 5}),
        P("All USD values are catalogue price times provider-reported tokens. Native provider credits (unit unverified) are shown separately and are never added to USD. Provider-reported USD is not available. Ledger spend for this run equals the estimated total; the ledger now shows $6.12 of $10 with $3.88 left."),
        FIG(reg, "cost_bars"), FIG(reg, "quality_vs_cost")]))
    S.append(("Statistical comparisons and uncertainty", [
        P("Paired McNemar exact tests on the same cases (invalid = incorrect), Holm-adjusted within each task and split; accuracy intervals are Wilson, F1 intervals bootstrap. With 100-120 cases only large gaps are significant."),
        table(pw[(pw.split == "test") & pw["significant_at_0.05_after_holm"]].sort_values(["task", "adjusted_p_value_holm"]).head(24), ["task", "model_a", "model_b", "n_paired_cases", "accuracy_a", "accuracy_b", "adjusted_p_value_holm"], ["Task", "A", "B", "Pairs", "Acc. A", "Acc. B", "Holm p"], nd={"adjusted_p_value_holm": 4}),
        P("Only significant pairs are listed (the full set is in classifier_model_comparison_pairwise.csv). The top models in each task are mostly not significantly different from each other: the ranking among them is preliminary."),
        FIG(reg, "pairwise")]))
    S.append(("Rankings, composite and sensitivity", [
        P(M.WEIGHT_NOTE + f" Models below a {M.RELIABILITY_FLOOR:.0%} valid-output rate are flagged and never recommended; normalisation uses eligible models only."),
        table(it.sort_values("overall_rank"), ["model_id", "quality_rank", "reliability_rank", "latency_rank", "cost_rank", "composite_score", "overall_rank", "recommendation"], ["Intent model", "Quality", "Reliab.", "Latency", "Cost", "Score", "Overall", "Recommendation"], nd={"composite_score": 3, "quality_rank": 0, "reliability_rank": 0, "latency_rank": 0, "cost_rank": 0, "overall_rank": 0}),
        table(lt.sort_values("overall_rank"), ["model_id", "quality_rank", "reliability_rank", "latency_rank", "cost_rank", "composite_score", "overall_rank", "recommendation"], ["Lead model", "Quality", "Reliab.", "Latency", "Cost", "Score", "Overall", "Recommendation"], nd={"composite_score": 3, "quality_rank": 0, "reliability_rank": 0, "latency_rank": 0, "cost_rank": 0, "overall_rank": 0}),
        table(sens, ["task", "scenario", "winner_among_models_with_valid_rate_>=_0.90", "winner_score"], ["Task", "Weight scenario", "Winner (eligible models)", "Score / draw share"]),
        FIG(reg, "composite")]))
    S.append(("Model-by-model notes", [
        table(_model_notes(it, lt), ["model", "intent", "lead"], ["Model", "Main Intent", "Lead Qualifier"]),
        B(["Qwen3.5-35B: best Intent and second best Lead; mid-range latency and cost; no invalid outputs. Reasoning was turned off by earlier approval.",
           "Jev 1.14: best Lead on every axis and the fastest and cheapest overall; weaker on Intent, especially the rare classes. Lead output is a probability and needs the frozen threshold.",
           "Gemini 3.5 Flash Lite (Intent baseline) and GPT-4.1-mini (Lead baseline): solid and fully valid; the benchmark routes are replicas, not the production workflow.",
           "GPT-4.1-nano and GPT-4o-mini: valid and cheap; mid quality; GPT-4o-mini under-detects leads (recall 0.60).",
           "DeepSeek, Mistral, Claude Haiku: reliability problems under the fixed caps and strict format validation; see the validity section."])]))
    S.append(("Limitations, privacy and readiness", [
        B(["Small samples: 120 intent cases (two classes with 4), 100 lead cases per half; intervals are wide and the top models overlap.", "Labels are one operator's attestation; no second annotation, so no reviewer agreement or kappa.",
           "Data origin unverified; potentially real admissions-derived conversations; redaction is pattern-based with no name recognition and pseudonymisation is not anonymisation.",
           "Processing permission is an operator attestation with no document and unnamed approvers; provider retention, training and processing policies are UNKNOWN for every route (public documentation for several routes was summarised separately and verifies nothing for these records).",
           "Production parity is not established (the Dify export is missing); the benchmark prompts and routes are not the production workflow.", "Costs are estimates; no provider-reported USD; native credits have an unverified USD ratio.",
           "Latency is client-side; throughput, first-attempt timing and provider latency are not available.", "The executor did not persist the typed real-data authorisation as a manifest.",
           "Claude and Mistral results reflect the fixed 16/24-token caps and strict format validation, not their capability ceilings."])]))
    S.append(("Recommendations", [
        B([f"Main Intent: shortlist {bi.model_id.split('/')[-1]} as the best observed candidate; keep Jev in view if latency and cost dominate; do not rely on any model for UPLOAD_STATUS_CHECK until more cases of that class exist.",
           f"Lead Qualifier: shortlist {bl.model_id.split('/')[-1]} with its frozen threshold; choose the threshold's recall/precision trade-off once the business cost of a missed lead versus a false alert is known.",
           "Before any production decision: confirm data origin and the processing permission with the data owner, obtain a second annotation (at least for the rare Intent classes), add cases for UPLOAD_STATUS_CHECK and OUT_OF_SCOPE, and verify provider retention and training terms for the chosen routes.",
           "Treat Claude Haiku, DeepSeek and Mistral as untested for classification quality under adequate output limits; re-test only through a separate pre-registered experiment with changed caps or format handling.",
           "Check parity with the production workflow once the Dify export exists."]),
        P("Reproduce with: msu-llm-lab labeled-eval prepare, labeled-eval run (after all gates), labeled-eval score, labeled-eval master, then this report. Inputs are read-only; no network is used by scoring, the master tables or this report.")]))
    S.append(("Appendix: files and definitions", [
        table(pd.DataFrame([("classifier_model_comparison_master_intent.csv", "one row per model, 141 columns"), ("classifier_model_comparison_master_lead.csv", "one row per model and split, 142 columns"),
                            ("classifier_model_comparison_lead_all200.csv", "all 200 Lead cases, Jev cross-validated"), ("classifier_model_comparison_pairwise.csv", "all paired tests"),
                            ("classifier_model_comparison_confusion_long.csv", "confusion counts"), ("classifier_model_comparison_sensitivity.csv", "weight scenarios"),
                            ("classifier_model_comparison_reconciliation.csv", "retry/cost/metric checks"), ("classifier_evaluation_results.csv", "primary results table"),
                            ("classifier_predictions.csv / classifier_error_analysis.csv", "per-case predictions and misses (case ids only)"), ("classifier_threshold_sweep.csv / classifier_threshold_test_results.csv", "Jev threshold"),
                            ("gates_before_first_request.md, provider_terms_and_requirements.md, operator_attestation.json", "approvals and terms")], columns=["file", "content"]), ["file", "content"], ["File", "Content"]),
        table(pd.DataFrame([("effective (end-to-end)", "invalid output counts as a miss and as an extra INVALID prediction"), ("valid-only", "only valid outputs counted"), ("macro-F1", "mean per-class F1, classes weighted equally"),
                            ("CTA precision/recall/F1", "positive class need_cta = true; missing positives are false negatives"), ("MCC / kappa", "multiclass formulas with INVALID as an extra predicted class"),
                            ("composite", "min-max over eligible models, weighted; analyst weights")], columns=["term", "definition"]), ["term", "definition"], ["Term", "Definition"])]))
    return S


def renumber(secs: List[Tuple[str, List[Any]]], reg: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Figure ids follow the order of appearance in the report."""
    order = [e[1]["file"] for _, els in secs for e in els if e[0] == "fig"]
    new = {f: f"F{i + 1:02d}" for i, f in enumerate(dict.fromkeys(order))}
    old = {r["file"]: r["figure_id"] for r in reg}
    for _, els in secs:
        for e in els:
            if e[0] == "fig":
                e[1]["caption"] = e[1]["caption"].replace(old[e[1]["file"]] + ".", new[e[1]["file"]] + ".", 1)
    for r in reg:
        r["figure_id"] = new.get(r["file"], r["figure_id"])
    return sorted(reg, key=lambda r: r["figure_id"])


def _fmtnum(v: Any) -> Any:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return round(x, 6)


def build(cfg: R.Cfg) -> Dict[str, Any]:
    T = load(cfg)
    out = cfg.out / R.VERSION
    old = out / "classifier_evaluation_report.md"
    if old.exists() and "preparation stage" in old.read_text()[:400] and not (out / "classifier_evaluation_report_prerun.md").exists():
        old.rename(out / "classifier_evaluation_report_prerun.md")          # keep the preparation-stage report; the final report takes its name
    reg = figures(T, out)
    T["rec"] = T["rec"].assign(expected_or_reference=T["rec"].expected_or_reference.map(_fmtnum), observed=T["rec"].observed.map(_fmtnum))
    secs = sections(T, reg)
    reg = renumber(secs, reg)
    pd.DataFrame(reg).to_csv(out / "classifier_evaluation_report_figures.csv", index=False)
    md = to_markdown(secs).replace("# MSU LLM Inference Workbench — Comprehensive Benchmark Evaluation Report", "# MSU LLM Inference Workbench: Classifier Evaluation Report (labeled-eval-v1)", 1)
    (out / "classifier_evaluation_report.md").write_text(md)
    it, ld = T["intent"], T["lead"]
    lt = ld[ld.split == "test"]
    bi, bl = it[it.recommendation.str.startswith("BEST")].iloc[0], lt[lt.recommendation.str.startswith("BEST")].iloc[0]
    (out / "classifier_evaluation_summary.md").write_text(
        "# Summary (labeled-eval-v1, executed)\n\n"
        f"- Run: 2,560 requests (120 Main Intent + 200 Lead cases, eight participants per task), no errors, 14 requests retried, nothing quarantined; estimated cost about $1.08; no provider-reported USD.\n"
        f"- Main Intent best observed candidate: {bi.model_id} (macro-F1 {bi.macro_f1:.3f}, accuracy {bi.accuracy:.3f}). Lead best observed candidate: {bl.model_id} (CTA F1 {bl.cta_f1:.3f}, recall {bl.cta_recall:.2f}; frozen threshold {bl.selected_decision_threshold}).\n"
        "- Not a statistically established winner, not production-approved. Details: classifier_evaluation_report.pdf, classifier_model_comparison_summary.md.\n")
    (out / "classifier_evaluation_limitations.md").write_text(
        "# Limitations and unavailable metrics\n\n- Small samples (120 intent, 100 lead per half); two Intent classes have 4 cases each.\n- Labels are one operator's attestation; no second annotation, so reviewer agreement and Cohen's kappa are unavailable.\n"
        "- Data origin unverified (potentially real, CONFIDENTIAL); processing permission is an operator attestation without a document; provider data policies UNKNOWN for all nine routes.\n- PRODUCTION_PARITY_NOT_ESTABLISHED (Dify export missing).\n"
        "- Costs are estimates; native credits unit unverified; provider-reported USD not available.\n- Latency is client-side; throughput, first-attempt timing, provider latency, time to first token are not available; repeatability and language slices were not measured.\n"
        "- Claude Haiku, DeepSeek and Mistral results reflect the fixed 16/24-token caps and strict output validation.\n- The executor did not persist the typed real-data authorisation as a manifest.\n")
    cover = {"bench_id": "labeled-eval-v1", "revision": "labeled-eval-v1", "header": "MSU LLM Inference Workbench  |  Classifier Evaluation Report", "footer": "Labels operator-attested; costs estimated; provider policies unknown; not production-approved",
             "cover_lines": ("MSU LLM Inference", "Workbench", "Classifier Evaluation Report", "Main Intent and Lead Qualifier: Accuracy, Reliability, Latency and Cost"),
             "pdf_title": "MSU LLM Inference Workbench - Classifier Evaluation Report (labeled-eval-v1)"}
    to_pdf_redesign(secs, out / "classifier_evaluation_report.pdf", out, cover)
    return {"sections": secs, "figures": reg, "out": out}
