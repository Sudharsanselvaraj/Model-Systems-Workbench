"""Compact black-on-white redesign of the evaluation report.

Reuses the analysis tables, figures and section content of `evaluation_document`; only regroups them into a 15-section flow, pairs related figures into
labelled panels, and renders with a different style (no table of contents, art-directed cover). No numbers are recomputed here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from msu_llm_lab.observability.evaluation_document import Element, _clean, build_sections, table

REVISION = "redesign-r1"
INK = "#111111"
PANELS = {  # panel name -> (caption, [stems])
    "valid_rate_pair": ("Valid-output rate with Wilson 95% CI. (a) INTENT, (b) LEAD. Reliability, not accuracy.", ["valid_rate_intent", "valid_rate_lead"]),
    "latency_box_pair": ("End-to-end latency distribution (log axis, fliers hidden). (a) INTENT, (b) LEAD. Client-side latency.", ["latency_box_intent", "latency_box_lead"]),
    "cost_pair": ("Estimated cost per 1,000 requests (ESTIMATED, not invoiced). (a) INTENT, (b) LEAD.", ["cost_per_1000_intent", "cost_per_1000_lead"]),
    "pareto_pair": ("Operational cost-latency view; * marks the Pareto frontier; Mistral excluded. (a) INTENT, (b) LEAD.", ["pareto_intent", "pareto_lead"]),
    "distribution_pair": ("Distribution of valid outputs (not correctness). (a) INTENT label shares, (b) LEAD need_cta shares (Jev probability-only excluded).",
                          ["label_distribution_intent", "lead_decision_distribution"]),
    "flow_pair": ("(a) Plan coverage: executed versus not executed blocks. (b) Request flow from the full matrix to outcomes.", ["coverage_overview", "request_flow"]),
}


def make_panels(out_dir: Path) -> Dict[str, Path]:
    """Stack the original figures vertically with (a)/(b) labels. Originals are untouched; panels are written under figures/panels/."""
    pdir = Path(out_dir) / "figures" / "panels"
    pdir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        font = ImageFont.load_default()
    made = {}
    for name, (_, stems) in PANELS.items():
        ims = [Image.open(Path(out_dir) / "figures" / f"{s}.png").convert("RGB") for s in stems]
        w = max(i.width for i in ims)
        canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
        y = 0
        d = ImageDraw.Draw(canvas)
        for k, im in enumerate(ims):
            canvas.paste(im, (0, y))
            d.text((8, y + 4), f"({chr(97 + k)})", fill=(0, 0, 0), font=font)
            y += im.height
        path = pdir / f"{name}.png"
        canvas.save(path, optimize=False)
        made[name] = path
    return made


def build_redesign(R: Dict[str, Any], fig: pd.DataFrame, panels: Dict[str, Path]) -> Tuple[List[Tuple[str, List[Element]]], Dict[str, Any]]:
    old = {t: els for t, els in build_sections(R, fig)}
    T, S, F = R["T"], R["S"], R["findings"]
    fid = {Path(r.file).stem: r.figure_id for r in fig.itertuples()}
    paneled = {s for _, (_, stems) in PANELS.items() for s in stems}
    used: set = set()

    def f(stem: str, cap_extra: str = "") -> Element:
        used.add(stem)
        r = fig[fig.file == f"figures/{stem}.png"].iloc[0]
        return ("fig", {"file": r.file, "caption": f"{r.figure_id}. {r.title}. {r.caption}"})

    def pn(name: str) -> Element:
        cap, stems = PANELS[name]
        used.update(stems)
        ids = "/".join(fid[s] for s in stems)
        return ("fig", {"file": f"figures/panels/{name}.png", "caption": f"{ids}. {cap}", "tall": True})

    def els_of(title: str, drop_fig: bool = True, drop_tables: Tuple[int, ...] = ()) -> List[Element]:
        out, ti = [], 0
        for k, e in old[title]:
            if k == "fig" and drop_fig:
                continue
            if k == "table":
                ti += 1
                if ti - 1 in drop_tables:
                    continue
            out.append((k, e))
        return out

    def only(title: str, kinds: Tuple[str, ...]) -> List[Element]:
        return [(k, e) for k, e in old[title] if k in kinds]

    df, mts = S["df"], T["model_task_summary"]
    ex = int(S["state"]["blocks_completed"])
    sec: List[Tuple[str, List[Element]]] = []

    sec.append(("Executive summary and key findings", only("Executive summary", ("p", "bullets")) + [("h2", "Reading guide")] + only("Document control", ("bullets",))
                + [("bullets", [x for x in only("How to read this report", ("bullets",))[0][1] if not x.startswith("Sections 1-4")])]))

    stratum = [(k, e) for k, e in old["Dataset and sampling"] if k == "table"]
    stratum_small = [table(T["stratum_coverage"].head(12), ["task", "stratum", "cases_in_sample"], ["Task", "Preparation stratum (metadata, not labels)", "Cases in sample"])]
    sec.append(("Benchmark scope, design, dataset and model roster",
                only("Scope and objectives", ("bullets",)) + only("Benchmark design", ("p", "bullets")) + [e for e in only("Dataset and sampling", ("p",))] + stratum_small
                + [P_("The ten most frequent strata are shown; the full list is in stratum_coverage.csv."), f("stratum_coverage")]
                + only("Participants and routes", ("table", "p"))))

    sec.append(("Execution status, artifact integrity and request reconciliation",
                only("Execution history", ("bullets",)) + only("Artifact integrity", ("p", "table")) + only("Request-level reconstruction", ("p",))
                + only("Request reconciliation", ("table",)) + [pn("flow_pair")] + only("Carry-over accounting", ("p", "table")) + only("Event-log reconciliation", ("p",))
                + [("h2", "Known issues investigated against the real files")] + only("Known issues investigation", ("table",))))

    sec.append(("Overall results and model coverage",
                only("Coverage and completeness", ("p", "table")) + [("h2", "Per-model summary (both tasks)")] + only("Key findings at a glance", ("table", "p"))
                + [("h2", "Per-task results")] + only("Operational results overview", ("table",)) + [f("outcomes_stacked")]))

    sec.append(("Output validity and reliability",
                only("Output validity and reliability", ("table",)) + [P_("Valid-output rate is the share of responses that parsed to an allowed output. It measures format reliability, not whether the label is correct."),
                                                                      pn("valid_rate_pair"), f("invalid_causes")]))

    sec.append(("Batch-wise and block-wise analysis",
                only("Per-batch analysis", ("p", "table")) + [f("block_heatmap_valid")] + only("Batch trends", ("p",)) + [f("latency_by_block"), f("disagreement_by_block")]))

    sec.append(("Latency and throughput", only("Latency analysis", ("p", "table")) + [pn("latency_box_pair"), f("latency_percentiles"), f("latency_p50_ci"), f("latency_vs_tokens")]))

    sec.append(("Token usage and cost reconciliation",
                only("Token usage", ("table",)) + [f("token_distributions")] + only("Cost analysis", ("table", "p")) + [pn("cost_pair"), f("cost_per_valid"), f("cumulative_cost")]
                + [("h2", "Cost reconciliation")] + only("Cost reconciliation", ("table", "p")) + [f("cost_sources"), f("ledger_residual")]))

    sec.append(("Model agreement and output distributions",
                only("Agreement analysis (INTENT)", ("p", "table")) + [f("agreement_heatmap_intent"), f("kappa_heatmap_intent"), f("peer_plurality")]
                + [("h2", "LEAD agreement")] + only("Agreement analysis (LEAD)", ("p", "table")) + [f("agreement_heatmap_lead")]
                + [("h2", "Output distributions")] + only("Output distributions", ("table", "p")) + [pn("distribution_pair")]))

    sec.append(("Matched-case comparisons and Pareto analysis",
                only("Matched-case leaderboards", ("p", "table")) + only("Pareto view", ("table", "p")) + [pn("pareto_pair"), f("valid_vs_cost")]
                + [("h2", "Paired statistical comparisons")] + only("Statistical comparisons", ("p", "table"))))

    # compact model deep dives: one row per model, no repeated per-task tables
    ms = T["model_summary"]
    pol = {p["key"]: p for p in S["manifest"]["participants"]}
    rows = []
    for r in ms.itertuples():
        g = mts[mts.participant == r.participant]
        rows.append({"participant": r.participant, "route": pol.get(r.participant, {}).get("route", {}).get("pinned_tag", "n/a"),
                     "intent": "; ".join(f"{x.valid_rate:.1%} / {x.latency_p50_ms:,.0f} ms / ${x.est_cost_usd_per_1000_requests:.3f}" for x in g.itertuples() if x.task == "INTENT") or "not run",
                     "lead": "; ".join(f"{x.valid_rate:.1%} / {x.latency_p50_ms:,.0f} ms / ${x.est_cost_usd_per_1000_requests:.3f}" for x in g.itertuples() if x.task == "LEAD") or "not run",
                     "note": ("probability-only LEAD; no decision rule" if r.participant == "jev-1.14" else "quarantined after block 10; 137 requests unattempted" if r.participant.startswith("mistral")
                              else "empty outputs reach the token cap; cause unverified" if r.participant.startswith("claude") else "reasoning off by approval" if r.participant.startswith("qwen")
                              else "production baseline on a benchmark route" if r.participant.startswith("baseline") else "")})
    dd = [("p", "Each row: valid-output rate / median end-to-end latency / estimated USD per 1,000 requests (ESTIMATED). Per-task counts and intervals are in the overview table above.")]
    dd.append(table(pd.DataFrame(rows), ["participant", "route", "intent", "lead", "note"], ["Participant", "Pinned route", "INTENT", "LEAD", "Model-specific note"]))
    pw = T["agreement_metrics"]
    pp = pw[pw.metric_family == "peer_plurality"]
    dd.append(("p", "Agreement with the plurality of other valid outputs (not accuracy): " + "; ".join(
        f"{p} " + "/".join(f"{r.task[0]} {r.agreement:.2f}" for r in pp[pp.participant_a == p].itertuples()) for p in sorted(pp.participant_a.unique())) + "."))
    sec.append(("Individual model deep dives", dd))

    sec.append(("Error, retry and quarantine investigations",
                only("Error analysis", ("table", "p")) + [f("finish_reasons"), f("retry_rate")] + [("h2", "Mistral quarantine")] + only("Mistral quarantine", ("p", "bullets")) + [f("mistral_timeline")]
                + [("h2", "Claude empty outputs")] + only("Claude empty-output investigation", ("p", "table")) + [("h2", "Jev probability-only LEAD")] + only("Jev probability-only LEAD", ("p",))
                + [f("jev_yes_probability")]))

    sec.append(("Limitations, privacy and operational readiness",
                only("Limitations", ("bullets",)) + [("h2", "Privacy and data handling")] + only("Privacy and data handling", ("bullets",)) + [("h2", "Baselines versus candidates")]
                + only("Baselines versus candidates", ("p",))))

    sec.append(("Recommendations and reproducibility", only("Recommendations", ("bullets",)) + [("h2", "Reproducibility")] + only("Reproducibility", ("p",))))

    bs = T["batch_summary"]
    ne = bs[bs.block >= ex]
    rng = pd.DataFrame([{"block": f"{ex}-{int(ne.block.max())}", "state": "NOT_EXECUTED", "blocks": len(ne), "plan_new": int(ne.plan_new_requests.sum()),
                         "results": int(ne.records.sum())}])
    inv = T["artifact_inventory"]
    used_inv = inv[inv.used_in_analysis].assign(path=lambda d: d.path.str[-60:], sha=lambda d: d.sha256.str[:12])
    figs_left = [s for s in fid if s not in used]
    assert not figs_left, f"figures not placed: {figs_left}"
    sec.append(("Technical appendix",
                [("h2", "A. Metric definitions")] + only("Metric catalogue", ("table",)) + [("h2", "B. Statistical methods")] + only("Statistical methods", ("bullets",))
                + [("h2", "C. Block ledger")] + [("p", f"Blocks 0-{ex - 1} are tabulated in section 6. Blocks {ex}-{int(ne.block.max())} were not executed and hold no results; the per-block table for all {len(bs)} blocks is batch_summary.csv.")]
                + [table(rng, ["block", "state", "blocks", "plan_new", "results"], ["Blocks", "State", "Count", "Plan new requests (total)", "Result records"])]
                + [("h2", "D. Artifact provenance (files used in the analysis)")] + [table(used_inv, ["path", "bytes", "sha", "provenance"], ["File (tail)", "Bytes", "SHA-256 (12)", "Role"])]
                + [("h2", "E. Figure list")] + [table(fig, ["figure_id", "file", "title"], ["ID", "File", "Title"])]))
    return sec, {"figures_total": len(fid), "figures_individual": len([s for s in used if s not in paneled]), "figures_paneled": len([s for s in used if s in paneled]),
                 "panels": {k: v[1] for k, v in PANELS.items()}}


def P_(t: str) -> Element:
    return ("p", t)


# ----------------------------------------------------------------------------- renderer
def to_pdf_redesign(sections: List[Tuple[str, List[Element]]], out_pdf: Path, base_dir: Path, cover: Dict[str, Any]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import BaseDocTemplate, CondPageBreak, Frame, Image as RLImage, KeepTogether, NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    ink, grey = colors.HexColor(INK), colors.HexColor("#8a8a8a")
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=8.4, leading=10.4, textColor=ink, spaceAfter=3.5)
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=15, textColor=ink, spaceBefore=9, spaceAfter=5)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.4, leading=11.5, textColor=ink, spaceBefore=6, spaceAfter=3)
    cap = ParagraphStyle("cap", parent=body, fontSize=7.2, leading=8.8, spaceBefore=1.5, spaceAfter=7)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=6.5, leading=7.7, textColor=ink)
    cellr = ParagraphStyle("cellr", parent=cell, alignment=TA_RIGHT)
    cellh = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold")
    cellhr = ParagraphStyle("cellhr", parent=cellh, alignment=TA_RIGHT)
    bl = ParagraphStyle("bl", parent=body, leftIndent=9, firstLineIndent=-7, spaceAfter=1.8)
    num = re.compile(r"^[-+$]?[\d,]*\.?\d+(%| ms)?$|^n/a$|^yes$|^no$")

    class Doc(BaseDocTemplate):
        pass

    def deco(c, d):
        c.saveState()
        c.setStrokeColor(grey)
        c.setLineWidth(0.4)
        c.line(1.7 * cm, A4[1] - 1.2 * cm, A4[0] - 1.7 * cm, A4[1] - 1.2 * cm)
        c.setFont("Helvetica", 6.8)
        c.setFillColor(ink)
        c.drawString(1.7 * cm, A4[1] - 1.0 * cm, "MSU LLM Inference Workbench  |  Comprehensive Benchmark Evaluation")
        c.drawRightString(A4[0] - 1.7 * cm, A4[1] - 1.0 * cm, cover["bench_id"])
        c.line(1.7 * cm, 1.2 * cm, A4[0] - 1.7 * cm, 1.2 * cm)
        c.drawRightString(A4[0] - 1.7 * cm, 0.75 * cm, f"{d.page}")
        c.drawString(1.7 * cm, 0.75 * cm, "Operational benchmark; ground truth pending; costs estimated")
        c.restoreState()

    def draw_cover(c, d):
        W, H = A4
        L = 2.0 * cm
        y0 = H * 0.58
        c.saveState()
        c.setFillColor(ink)
        c.setFont("Helvetica-Bold", 34)
        c.drawString(L, y0, "MSU LLM Inference")
        c.drawString(L, y0 - 1.3 * cm, "Workbench")
        c.setFont("Helvetica", 19)
        c.drawString(L, y0 - 2.7 * cm, "Comprehensive Benchmark Evaluation")
        c.setFont("Helvetica", 10.5)
        c.drawString(L, y0 - 3.5 * cm, "Operational Reliability, Latency, Cost and Cross-Model Analysis")
        c.restoreState()

    doc = Doc(str(out_pdf), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm, invariant=1,
              title="MSU LLM Inference Workbench — Comprehensive Benchmark Evaluation", author="MSU LLM Workbench (offline report generator)",
              subject=f"Operational benchmark evaluation, {cover['revision']}", keywords=f"{cover['bench_id']}, {cover['revision']}", creator="msu-llm-lab evaluation-report --redesign")
    fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    cover_fr = Frame(0, 0, 1, 1, id="cv")
    doc.addPageTemplates([PageTemplate(id="cover", frames=[cover_fr], onPage=draw_cover), PageTemplate(id="main", frames=[fr], onPage=deco)])
    W = doc.width

    def table_flow(v):
        head, rows = v["head"], v["rows"]
        isnum = [sum(bool(num.match(str(r[j]).strip())) for r in rows) >= max(1, 0.7 * len(rows)) for j in range(len(head))]
        lens = [max([len(str(head[j]))] + [len(str(r[j])) for r in rows[:80]]) for j in range(len(head))]
        wts = [min(max(l, 5), 70) ** 0.85 for l in lens]
        cw = [W * w / sum(wts) for w in wts]
        data = [[Paragraph(escape(_clean(str(h))), cellhr if isnum[j] else cellh) for j, h in enumerate(head)]]
        data += [[Paragraph(escape(_clean(str(c))), cellr if isnum[j] else cell) for j, c in enumerate(r)] for r in rows]
        t = Table(data, colWidths=cw, repeatRows=1)
        sty = [("LINEABOVE", (0, 0), (-1, 0), 0.9, ink), ("LINEBELOW", (0, 0), (-1, 0), 0.6, ink), ("LINEBELOW", (0, -1), (-1, -1), 0.9, ink),
               ("INNERGRID", (0, 1), (-1, -1), 0.2, colors.HexColor("#c8c8c8")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
               ("LEFTPADDING", (0, 0), (-1, -1), 2.2), ("RIGHTPADDING", (0, 0), (-1, -1), 2.2), ("TOPPADDING", (0, 0), (-1, -1), 1.1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.1)]
        sty += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f3f3")) for i in range(2, len(data), 2)]
        t.setStyle(TableStyle(sty))
        return t

    st: List[Any] = [NextPageTemplate("main"), PageBreak()]
    for i, (title, els) in enumerate(sections, 1):
        st.append(CondPageBreak(5 * cm))
        st.append(Paragraph(escape(_clean(f"{i}. {title}")), h1))
        for kind, v in els:
            if kind == "p":
                st.append(Paragraph(escape(_clean(v)), body))
            elif kind == "h2":
                st += [CondPageBreak(3.2 * cm), Paragraph(escape(_clean(v)), h2)]
            elif kind == "bullets":
                st += [Paragraph("&bull; " + escape(_clean(x)), bl) for x in v] + [Spacer(1, 2)]
            elif kind == "table":
                st += [table_flow(v), Spacer(1, 5)]
            elif kind == "fig":
                w, h = Image.open(base_dir / v["file"]).size
                iw = W * (0.70 if not v.get("tall") else 0.74)
                ih = iw * h / w
                maxh = 13.5 * cm if not v.get('tall') else 17 * cm
                if ih > maxh:
                    ih, iw = maxh, maxh * w / h
                st.append(KeepTogether([RLImage(str(base_dir / v["file"]), width=iw, height=ih, hAlign="CENTER"), Paragraph(escape(_clean(v["caption"])), cap)]))
    doc.build(st)


def cover_data(R: Dict[str, Any], bundle_name: str, date: str) -> Dict[str, Any]:
    S, T = R["S"], R["T"]
    ex = int(S["state"]["blocks_completed"])
    mbs = T["model_batch_summary"]
    g = mbs[mbs.block < ex].pivot_table(index=["participant", "task"], columns="block", values="valid_rate")
    motif = [[1.0 if pd.isna(v) else float(v) for v in row] for row in g.to_numpy()]
    df = S["df"]
    return {"bench_id": S["cfg"].bench_id, "revision": REVISION, "motif": motif,
            "stats": [(str(df.participant.nunique()), "participants"), (f"{len(df):,}", "records"), (f"{ex} of {len(S['plan']['blocks'])}", "blocks executed"), ("2", "tasks")],
            "meta": [("Benchmark ID", S["cfg"].bench_id), ("Source bundle", bundle_name), ("Report date", date),
                     ("Tasks", "Main Intent (5 labels) and Lead Qualifier (need_cta true/false)"),
                     ("Sample", f"{ex * 10} of 2,211 INTENT and {ex * 10} of 723 LEAD eligible cases (stratified)"),
                     ("Records", f"{int(df.is_new.sum()):,} new and {int((~df.is_new).sum()):,} carried over from earlier plans")],
            "status": f"Partial execution of a sampled operational benchmark: {ex} of {len(S['plan']['blocks'])} blocks, Mistral quarantined. Ground truth is pending, so accuracy, F1, calibration and "
                      "confusion matrices are not computable. Costs are estimates, not provider invoices. Latency is client-side. Production parity is not established."}
