"""Figures for the evaluation report: deterministic matplotlib (Agg) renders from the analysis tables. Every figure states its n and caveats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "msu", "figure.autolayout": False})
NOTE = "Operational metrics on a 170-case-per-task stratified sample; ground truth PENDING (no accuracy)."


def _colors(parts: List[str]) -> Dict[str, str]:
    return {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(sorted(parts))}


def _save(fig, out: Path, name: str, note: str = NOTE) -> str:
    fig.text(0.01, 0.005, note, fontsize=6, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    path = out / f"{name}.png"
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return str(path)


def build_figures(R: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    S, T = R["S"], R["T"]
    df: pd.DataFrame = S["df"]
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    mts, mbs, bsum = T["model_task_summary"], T["model_batch_summary"], T["batch_summary"]
    parts = sorted(df.participant.unique())
    col = _colors(parts)
    man: List[Dict[str, str]] = []
    n_fig = [0]

    def reg(name: str, title: str, caption: str, source: str, path: str):
        n_fig[0] += 1
        man.append({"figure_id": f"F{n_fig[0]:02d}", "file": f"figures/{name}.png", "title": title, "caption": caption, "source_tables": source})

    def hbar(name, title, task, col_name, xlabel, caption, source, err=None, fmt="{:.3f}"):
        g = mts[mts.task == task].sort_values(col_name)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        xerr = None
        if err:
            xerr = np.clip(np.array([g[col_name] - g[err[0]], g[err[1]] - g[col_name]]), 0, None)
        ax.barh(g.participant, g[col_name], color=[col[p] for p in g.participant], xerr=xerr, capsize=2, ecolor="#333333")
        hi = (g[err[1]] if err else g[col_name]).to_numpy()
        for y, (v, h) in enumerate(zip(g[col_name], hi)):
            ax.text(max(v, h), y, " " + fmt.format(v), va="center", fontsize=6.5)
        ax.set_xlim(0, float(np.nanmax(hi)) * 1.15)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        reg(name, title, caption, source, _save(fig, out, name))

    # 1 coverage of the plan
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ex = int(S["state"]["blocks_completed"])
    ax.barh(["blocks"], [ex], color="#2ca02c", label=f"executed ({ex})")
    ax.barh(["blocks"], [len(bsum) - ex], left=[ex], color="#cccccc", label=f"not executed ({len(bsum) - ex})")
    ax.set_xlabel("block index (10 cases per task per block)")
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.45))
    ax.set_title("Plan coverage: 17 of 222 blocks executed", pad=28)
    reg("coverage_overview", "Plan coverage", "Executed versus not-executed blocks of the 222-block plan.", "batch_summary.csv", _save(fig, out, "coverage_overview"))

    # 2 request flow
    rec = T["request_reconciliation"].set_index("quantity")["value"]
    labels = ["full matrix\n(executed blocks)", "carried over\n(not re-sent)", "planned new", "dispatched", "unattempted\n(quarantine)", "OK", "invalid", "error"]
    keys = ["full matrix of executed blocks (units x 10 cases x blocks)", "carried-over records (not re-sent)", "planned new logical requests (plan blocks, executed)",
            "dispatched logical requests (new records)", "unattempted new requests (quarantine)", "outcomes OK", "outcomes INVALID_OUTPUT", "outcomes ERROR"]
    vals = [int(rec[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    b = ax.bar(labels, vals, color=["#7f7f7f", "#9467bd", "#1f77b4", "#2ca02c", "#d62728", "#2ca02c", "#ff7f0e", "#d62728"])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Request flow: 3,400 = 540 carried + 2,860 planned = 2,723 dispatched + 137 unattempted")
    plt.setp(ax.get_xticklabels(), fontsize=6.5)
    reg("request_flow", "Request flow reconciliation", "Counts from the full matrix to outcomes.", "request_reconciliation.csv", _save(fig, out, "request_flow"))

    # 3 outcomes stacked
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=False)
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        g = mts[mts.task == t].sort_values("valid_rate")
        y = np.arange(len(g))
        ax.barh(y, g.valid, color="#2ca02c", label="valid")
        ax.barh(y, g.invalid_output, left=g.valid, color="#ff7f0e", label="invalid")
        ax.barh(y, g.errors, left=g.valid + g.invalid_output, color="#d62728", label="error")
        ax.barh(y, g.unattempted_in_executed_blocks, left=g.valid + g.invalid_output + g.errors, color="#cccccc", label="unattempted")
        ax.set_yticks(y)
        ax.set_yticklabels(g.participant, fontsize=6.5)
        ax.set_title(t)
    axs[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle("Outcomes per participant (records in executed blocks)", fontsize=9)
    reg("outcomes_stacked", "Outcomes per participant", "Valid, invalid, error and unattempted counts per task.", "model_task_summary.csv", _save(fig, out, "outcomes_stacked"))

    for t in ("INTENT", "LEAD"):
        hbar(f"valid_rate_{t.lower()}", f"{t}: valid-output rate with Wilson 95% CI", t, "valid_rate", "valid-output rate (NOT accuracy)",
             f"Share of records that parsed to an allowed output, {t}.", "model_task_summary.csv", ("valid_rate_ci_lo", "valid_rate_ci_hi"))

    # invalid causes
    ov = T["output_validity_metrics"]
    ov = ov[ov.validation_issue != "NONE"]
    if len(ov):
        pv = ov.pivot_table(index="participant", columns="validation_issue", values="n", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        pv.plot(kind="barh", stacked=True, ax=ax, color=["#ff7f0e", "#d62728", "#9467bd", "#8c564b"][: pv.shape[1]])
        ax.set_title("Invalid-output causes (both tasks)")
        ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_xlabel("records")
        reg("invalid_causes", "Invalid-output causes", "EMPTY_OUTPUT, EXTRA_TEXT and UNKNOWN_LABEL counts per participant.", "output_validity_metrics.csv", _save(fig, out, "invalid_causes"))

    # finish reasons
    fr = df.assign(finish=df.finish_reason.replace("", "not captured")).groupby(["participant", "finish"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fr.plot(kind="barh", stacked=True, ax=ax, color=["#2ca02c", "#d62728", "#bbbbbb"][: fr.shape[1]] if fr.shape[1] <= 3 else None)
    ax.set_title("Finish reason (not captured for early carried-over records)")
    reg("finish_reasons", "Finish reasons", "'length' means the token cap truncated the output.", "model_results.csv", _save(fig, out, "finish_reasons"))

    # retries
    rl = T["reliability_metrics"].copy()
    rl["label"] = rl.participant + "|" + rl.task.str[0]
    rl = rl.sort_values("retry_rate")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(rl.label, rl.retry_rate * 100, color="#17becf")
    ax.set_xlabel("% of records needing >1 HTTP attempt")
    ax.set_title("Retry rate per participant and task (I=INTENT, L=LEAD)")
    reg("retry_rate", "Retry rate", "Records with attempts above one.", "reliability_metrics.csv", _save(fig, out, "retry_rate"))

    # latency boxplots
    for t in ("INTENT", "LEAD"):
        g = df[(df.task == t) & df.latency_ms_end_to_end.notna()]
        order = list(mts[mts.task == t].sort_values("latency_p50_ms").participant)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.boxplot([g[g.participant == p].latency_ms_end_to_end for p in order], vert=False, labels=order, showfliers=False)
        ax.set_xscale("log")
        ax.set_xlabel("end-to-end latency ms (log), fliers hidden")
        ax.set_title(f"{t}: end-to-end latency distribution")
        reg(f"latency_box_{t.lower()}", f"{t} latency distribution", "Boxes show quartiles, whiskers 1.5 IQR; client-side latency.", "model_results.csv", _save(fig, out, f"latency_box_{t.lower()}"))

    # latency percentile bars
    lm = T["latency_metrics"]
    lm = lm[lm["subset"] == "ALL"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        g = lm[lm.task == t].sort_values("p50")
        y = np.arange(len(g))
        for j, q in enumerate(("p50", "p95", "p99")):
            ax.barh(y + (j - 1) * 0.27, g[q], height=0.27, label=q)
        ax.set_yticks(y)
        ax.set_yticklabels(g.participant, fontsize=6.5)
        ax.set_xscale("log")
        ax.set_title(t)
    axs[0].legend(frameon=False, fontsize=6)
    fig.suptitle("Latency percentiles (ms, log scale); p99 from 170 records is unstable", fontsize=9)
    reg("latency_percentiles", "Latency percentiles", "p50/p95/p99 per participant and task.", "latency_metrics.csv", _save(fig, out, "latency_percentiles"))

    # p50 CI
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    g = mts.assign(label=mts.participant + "|" + mts.task.str[0]).sort_values("latency_p50_ms")
    ax.errorbar(g.latency_p50_ms, g.label, xerr=np.clip([g.latency_p50_ms - g.latency_p50_ci_lo, g.latency_p50_ci_hi - g.latency_p50_ms], 0, None), fmt="o", ms=3, color="#1f77b4", ecolor="#999999")
    ax.set_xscale("log")
    ax.set_xlabel("median latency ms with bootstrap 95% CI (2,000 resamples, fixed seed)")
    ax.set_title("Median latency uncertainty")
    reg("latency_p50_ci", "Median latency with bootstrap CI", "Percentile bootstrap, seeded.", "model_task_summary.csv", _save(fig, out, "latency_p50_ci"))

    # latency over blocks
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        g = mbs[(mbs.task == t) & mbs.state.isin(["EXECUTED", "PARTIAL_QUARANTINED"])]
        for p, h in g.groupby("participant"):
            ax.plot(h.block, h.latency_p50_ms, color=col[p], lw=1, label=p)
        ax.set_yscale("log")
        ax.set_title(t)
        ax.set_xlabel("block")
    axs[0].set_ylabel("block p50 latency ms")
    axs[1].legend(fontsize=5, frameon=False, ncol=2)
    fig.suptitle("Median latency per executed block", fontsize=9)
    reg("latency_by_block", "Latency by block", "Block-level p50 for the 17 executed blocks.", "model_batch_summary.csv", _save(fig, out, "latency_by_block"))

    # latency vs tokens
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    g = df[df.latency_ms_end_to_end.notna() & df.prompt_tokens.notna()]
    for p, h in g.groupby("participant"):
        ax.scatter(h.prompt_tokens, h.latency_ms_end_to_end, s=3, alpha=0.4, color=col[p], label=p)
    ax.set_yscale("log")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("latency ms")
    ax.legend(fontsize=5, ncol=2, frameon=False, markerscale=3)
    ax.set_title("Latency versus prompt tokens")
    reg("latency_vs_tokens", "Latency versus prompt tokens", "Each point is one request.", "model_results.csv", _save(fig, out, "latency_vs_tokens"))

    # token distributions
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, c, ttl in zip(axs, ("prompt_tokens", "completion_tokens"), ("prompt tokens", "completion tokens")):
        order = sorted(parts)
        ax.boxplot([df[(df.participant == p) & df[c].notna()][c] for p in order], vert=False, labels=order, showfliers=False)
        ax.set_title(ttl)
        ax.tick_params(axis="y", labelsize=6)
    reg("token_distributions", "Token distributions", "Provider-reported tokens per request, both tasks.", "model_results.csv", _save(fig, out, "token_distributions"))

    # cost
    for t in ("INTENT", "LEAD"):
        hbar(f"cost_per_1000_{t.lower()}", f"{t}: estimated cost per 1,000 requests (USD, ESTIMATED)", t, "est_cost_usd_per_1000_requests", "USD per 1,000 requests (estimated, not invoiced)",
             "Catalogue price times provider-reported tokens.", "cost_metrics.csv", fmt="{:.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    g = mts.assign(label=mts.participant + "|" + mts.task.str[0]).sort_values("est_cost_usd_per_valid_output")
    ax.barh(g.label, g.est_cost_usd_per_valid_output * 1000, color="#8c564b")
    ax.set_xlabel("estimated USD per 1,000 valid outputs")
    ax.set_title("Cost per valid output (invalid outputs still cost money)")
    reg("cost_per_valid", "Cost per valid output", "Total estimated cost divided by valid outputs.", "cost_metrics.csv", _save(fig, out, "cost_per_valid"))

    # cumulative cost over time (new records)
    nw = df[df.is_new & df.start_ts.notna()].sort_values("start_ts")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(nw.start_ts, nw.recorded_cost_usd.fillna(0).cumsum(), color="#1f77b4")
    ax.set_ylabel("cumulative estimated USD")
    ax.set_title("Cumulative estimated cost of this plan's new requests")
    fig.autofmt_xdate()
    reg("cumulative_cost", "Cumulative cost over time", "New requests only; estimated.", "model_results.csv", _save(fig, out, "cumulative_cost"))

    # cost sources
    cp = T["cost_provenance"].dropna(subset=["usd"])
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.barh(cp.source, cp.usd, color="#7f7f7f")
    for y, v in enumerate(cp.usd):
        ax.text(v, y, f" {v:.4f}", va="center", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("Cost sources are different quantities (none is an invoice)")
    reg("cost_sources", "Cost source comparison", "Results estimate, ledger, guard cost and native credits.", "cost_provenance.csv", _save(fig, out, "cost_sources"))

    # ledger vs results by run
    lv = R["aux"]["ledger_vs_results"].copy()
    lv["short"] = lv.run_id.str.replace("bench-", "b-").str[:34]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(lv.short, lv.delta_usd * 1000, color=["#d62728" if d > 0.0005 else "#2ca02c" for d in lv.delta_usd])
    ax.tick_params(axis="y", labelsize=5)
    ax.set_xlabel("ledger minus final results, milli-USD (red: booked spend with no final record)")
    ax.set_title("Ledger-to-results residual per store run id")
    reg("ledger_residual", "Ledger residual per run", "Booked requests whose records were superseded or not carried.", "reconciliation_details.csv", _save(fig, out, "ledger_residual"))

    # pareto
    pf = T["pareto"]
    for t in ("INTENT", "LEAD"):
        g = pf[pf.task == t]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for r in g.itertuples():
            ax.scatter(r.cost_per_1000_usd, r.p50_latency_ms, s=20 + 160 * r.valid_rate ** 4, color=col[r.participant], edgecolor="black" if r.on_frontier else "none", linewidth=1.5)
            ax.annotate(r.participant + ("*" if r.on_frontier else ""), (r.cost_per_1000_usd, r.p50_latency_ms), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("estimated USD per 1,000 requests")
        ax.set_ylabel("p50 latency ms")
        ax.set_title(f"{t}: cost-latency (bubble = valid-output rate; * = Pareto frontier; Mistral excluded)")
        reg(f"pareto_{t.lower()}", f"{t} operational Pareto view", "Non-dominated on cost, p50 latency and valid rate; no accuracy axis.", "pareto.csv", _save(fig, out, f"pareto_{t.lower()}"))
    g = mts
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for r in g.itertuples():
        ax.scatter(r.est_cost_usd_per_1000_requests, r.valid_rate, color=col[r.participant], marker="o" if r.task == "INTENT" else "s", s=22)
        ax.annotate(f"{r.participant}|{r.task[0]}", (r.est_cost_usd_per_1000_requests, r.valid_rate), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("estimated USD per 1,000 requests")
    ax.set_ylabel("valid-output rate")
    ax.set_title("Valid-output rate versus cost (circle INTENT, square LEAD)")
    reg("valid_vs_cost", "Valid rate versus cost", "Each point is a participant-task unit.", "model_task_summary.csv", _save(fig, out, "valid_vs_cost"))

    # label distributions
    am = T["agreement_metrics"]
    od = am[am.metric_family == "output_distribution"].copy()
    od["label"] = od.participant_b.str.replace("label=", "")
    for t, name in (("INTENT", "label_distribution_intent"), ("LEAD", "lead_decision_distribution")):
        g = od[od.task == t].pivot_table(index="participant_a", columns="label", values="agreement", fill_value=0)
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        g.plot(kind="barh", stacked=True, ax=ax, width=0.75)
        ax.set_xlabel("share of valid outputs")
        ax.set_title("INTENT: predicted label distribution (valid outputs)" if t == "INTENT" else "LEAD: need_cta decision distribution (Jev probability-only excluded)")
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        reg(name, f"{t} output distribution", "Distribution of valid outputs; says nothing about correctness.", "agreement_metrics.csv", _save(fig, out, name))

    # agreement heatmaps
    pw = am[am.metric_family == "pairwise"]
    for t, metric, name in (("INTENT", "agreement", "agreement_heatmap_intent"), ("INTENT", "cohens_kappa", "kappa_heatmap_intent"), ("LEAD", "agreement", "agreement_heatmap_lead")):
        g = pw[pw.task == t]
        ps = sorted(set(g.participant_a) | set(g.participant_b))
        M = pd.DataFrame(np.nan, index=ps, columns=ps)
        for r in g.itertuples():
            M.loc[r.participant_a, r.participant_b] = getattr(r, metric)
            M.loc[r.participant_b, r.participant_a] = getattr(r, metric)
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(M.values.astype(float), cmap="viridis", vmin=(-0.2 if metric == "cohens_kappa" else 0), vmax=1)
        ax.set_xticks(range(len(ps)))
        ax.set_yticks(range(len(ps)))
        ax.set_xticklabels(ps, rotation=70, ha="right", fontsize=6)
        ax.set_yticklabels(ps, fontsize=6)
        for i in range(len(ps)):
            for j in range(len(ps)):
                if not np.isnan(M.values[i, j]):
                    ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{t}: pairwise {'Cohen kappa' if metric == 'cohens_kappa' else 'agreement'} on common valid cases")
        reg(name, f"{t} pairwise {metric}", "Agreement between models; NOT accuracy.", "agreement_metrics.csv", _save(fig, out, name))

    pp = am[am.metric_family == "peer_plurality"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, t in zip(axs, ("INTENT", "LEAD")):
        g = pp[pp.task == t].sort_values("agreement")
        ax.barh(g.participant_a, g.agreement, xerr=np.clip([g.agreement - g.agreement_ci_lo, g.agreement_ci_hi - g.agreement], 0, None), color="#9467bd", capsize=2)
        ax.set_title(t)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Agreement with the plurality of the other valid outputs (NOT accuracy)", fontsize=9)
    reg("peer_plurality", "Peer-plurality agreement", "Strict plurality of at least 3 other valid outputs; ties skipped; Wilson 95% CI.", "agreement_metrics.csv", _save(fig, out, "peer_plurality"))

    # block heatmap valid rate
    ex = int(S["state"]["blocks_completed"])
    g = mbs[mbs.block < ex].assign(unit=lambda d: d.participant + "|" + d.task.str[0]).pivot_table(index="unit", columns="block", values="valid_rate")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(g.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=5.5)
    ax.set_xticks(range(g.shape[1]))
    ax.set_xlabel("block")
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Valid-output rate per block (blank = no records: quarantined)")
    reg("block_heatmap_valid", "Valid rate by block and participant", "Executed blocks 0-16 only; blocks 17-221 were not executed.", "model_batch_summary.csv", _save(fig, out, "block_heatmap_valid"))

    # disagreement by block
    cmp_ = pd.read_csv(Path(S["cfg"].bundle_dir) / "model_comparison.csv", dtype=str, keep_default_na=False)
    cmp_["block"] = [S["blockmap"].get((t, c)) for t, c in zip(cmp_.task, cmp_.case_id)]
    dis = cmp_[cmp_.models_agree.str.lower().isin(["false", "0", "no"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    d = dis.groupby(["block", "task"]).size().unstack(fill_value=0).reindex(range(ex), fill_value=0)
    d.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e"][: d.shape[1]])
    ax.set_title(f"Cases where valid models disagree, per block ({len(dis)} cases)")
    ax.set_xlabel("block")
    reg("disagreement_by_block", "Disagreements by block", "Cases flagged models_agree=false in model_comparison.csv.", "model_comparison.csv", _save(fig, out, "disagreement_by_block"))

    # mistral timeline
    mi = df[df.participant.str.startswith("mistral")].sort_values("start_ts")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.scatter(mi.start_ts, mi.status.map({"OK": 1, "INVALID_OUTPUT": 0.5, "ERROR": 0}), c=mi.status.map({"OK": "#2ca02c", "INVALID_OUTPUT": "#ff7f0e", "ERROR": "#d62728"}), s=8)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["error", "invalid", "valid"])
    fig.autofmt_xdate()
    ax.set_title("Mistral requests over time (circuit breaker opened at block 10; 137 requests unattempted)")
    reg("mistral_timeline", "Mistral outcome timeline", "Outcomes of all 203 Mistral records.", "model_results.csv", _save(fig, out, "mistral_timeline"))

    # stratum coverage
    sc = T["stratum_coverage"]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    lab = sc.task.str[0] + ":" + sc.stratum.str[:40]
    ax.barh(lab, sc.cases_in_sample, color=np.where(sc.task == "INTENT", "#1f77b4", "#ff7f0e"))
    ax.tick_params(axis="y", labelsize=4.5)
    ax.set_title("Sample composition by preparation stratum (metadata, not labels)")
    reg("stratum_coverage", "Stratum coverage", "Pattern buckets in the 170-case samples.", "stratum_coverage.csv", _save(fig, out, "stratum_coverage"))

    # Jev yes-probability
    jv = df[(df.participant == "jev-1.14") & (df.task == "LEAD") & df.yes_probability.notna()]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(jv.yes_probability, bins=20, color="#393b79")
    ax.set_xlabel("P(yes) (probability-only; no frozen threshold, no binary decision)")
    ax.set_title("Jev LEAD yes-probability distribution")
    reg("jev_yes_probability", "Jev LEAD probability distribution", "Jev returns P(yes) only; no decision is derived.", "model_results.csv", _save(fig, out, "jev_yes_probability"))

    return pd.DataFrame(man)
