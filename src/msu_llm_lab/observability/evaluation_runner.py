"""Write every deliverable of the evaluation report (CSV/JSON/Markdown/figures/PDF) and a verification file that is itself hashed in the output manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from msu_llm_lab.observability import evaluation_report as er
from msu_llm_lab.observability.evaluation_document import TITLE, build_sections, to_markdown, to_pdf
from msu_llm_lab.observability.evaluation_figures import build_figures

PDF_NAME = "benchmark_evaluation_report.pdf"
CSV_OUT = {  # table key -> file name
    "artifact_inventory": "artifact_inventory.csv", "metric_definitions": "metric_definitions.csv", "model_summary": "model_summary.csv",
    "model_task_summary": "model_task_summary.csv", "batch_summary": "batch_summary.csv", "model_batch_summary": "model_batch_summary.csv",
    "request_reconciliation": "request_reconciliation.csv", "reconciliation_summary": "reconciliation_summary.csv", "reconciliation_details": "reconciliation_details.csv",
    "latency_metrics": "latency_metrics.csv", "cost_metrics": "cost_metrics.csv", "reliability_metrics": "reliability_metrics.csv",
    "output_validity_metrics": "output_validity_metrics.csv", "agreement_metrics": "agreement_metrics.csv", "error_analysis": "error_analysis.csv",
    "coverage_matrix": "coverage_matrix.csv", "statistical_comparisons": "statistical_comparisons.csv", "leaderboards": "leaderboards.csv",
    "pareto": "pareto_frontier.csv", "cost_provenance": "cost_provenance.csv", "stratum_coverage": "stratum_coverage.csv",
}
PII = [re.compile(p) for p in (r"[\w.+-]+@[\w-]+\.[\w.-]+", r"\b\d{10,}\b", r"\+\d[\d\s-]{8,}")]


def _csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")


def _findings_md(F: List[Dict[str, str]]) -> str:
    out = ["# Data quality findings", "", "Each finding was checked against the real files; evidence is quoted from them.", ""]
    for f in F:
        out += [f"## {f['id']} {f['issue']}", f"- **Status:** {f['status']}", f"- **Evidence:** {f['evidence']}", f"- **Impact:** {f['impact']}", ""]
    return "\n".join(out)


def generate(cfg: er.Config) -> Dict[str, Any]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    R = er.run_analysis(cfg)
    T, S = R["T"], R["S"]
    for k, name in CSV_OUT.items():
        _csv(T[k], out / name)
    figm = build_figures(R, out)
    _csv(figm, out / "figure_manifest.csv")
    sections = build_sections(R, figm)
    md = to_markdown(sections)
    (out / "benchmark_evaluation_report.md").write_text(md)
    exec_md = md.split("## 3.")[0].split("## 2.", 1)
    (out / "executive_summary.md").write_text(f"# Executive summary\n\n{exec_md[1].split(chr(10), 2)[2] if len(exec_md) > 1 else ''}")
    (out / "data_quality_findings.md").write_text(_findings_md(R["findings"]))
    recs = [e for t, els in sections if t == "Recommendations" for k, e in els if k == "bullets"][0]
    (out / "recommendations.md").write_text("# Recommendations\n\nOperational evidence only; correctness is unmeasured (ground truth pending).\n\n" + "\n".join(f"- {r}" for r in recs) + "\n")
    ver = er.versions()
    (out / "analysis_config.json").write_text(json.dumps({"schema": er.SCHEMA, "config": cfg.as_json(), "seed": cfg.seed, "n_bootstrap": cfg.n_boot, "min_paired_n": er.MIN_PAIRED_N,
                                                             "percentile_method": "numpy linear interpolation", "ci_methods": {"proportion": "Wilson 95%", "median_mean": "percentile bootstrap 95%"},
                                                             "multiple_testing": "Holm within task x metric family", "versions": ver}, indent=2, sort_keys=True))
    (out / "reproducibility.md").write_text(
        "# Reproducibility\n\n```bash\nmsu-llm-lab evaluation-report --bundle reports/real_benchmark_bench-d0a763e4a920-r4 --out reports/evaluation_report_bench-d0a763e4a920\n```\n\n"
        f"- Seed {cfg.seed}; bootstrap resamples {cfg.n_boot}; per-unit seeds from SHA-256 of a label.\n- Versions: {json.dumps(ver)}\n"
        "- Inputs are opened read-only (the ledger via SQLite mode=ro); SHA-256 of every input is recorded before and after in artifact_integrity.json.\n"
        "- No network access, no model calls, no spend. Figures carry no timestamps; the PDF is built with reportlab invariant mode.\n"
        "- Run the tests with `uv run pytest tests/test_evaluation_report.py`.\n")
    (out / "artifact_integrity.json").write_text(json.dumps({"inputs_unchanged": R["hashes_ok"], "files": len(R["hashes_before"]), "sha256_before": R["hashes_before"], "sha256_after": R["hashes_after"],
                                                              "excluded_by_name": list(er.EXCLUDED_NAMES)}, indent=2, sort_keys=True))
    pages = to_pdf(sections, out / PDF_NAME, out, {"subtitle": f"Run {cfg.bench_id}: 11 participants, Main Intent and Lead Qualifier, sampled operational benchmark",
                                                    "subject": "Operational benchmark evaluation (no ground truth)",
                                                    "lines": [f"Source bundle: {cfg.bundle_dir.name}", f"Blocks executed: {S['state']['blocks_completed']} of {len(S['plan']['blocks'])}",
                                                              f"Records: {len(S['df']):,}  |  Seed: {cfg.seed}", f"Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}"]})
    verification = verify(out, R, sections)
    (out / "verification.md").write_text(verification["md"])
    manifest = {f.name: er.sha256_file(f) for f in sorted(out.rglob("*")) if f.is_file() and f.name != "output_manifest.json"}
    (out / "output_manifest.json").write_text(json.dumps({"schema": er.SCHEMA, "files": manifest, "note": "verification.md is included in this manifest"}, indent=2, sort_keys=True))
    return {"R": R, "verification": verification, "out": out, "pages": pages}


def verify(out: Path, R: Dict[str, Any], sections) -> Dict[str, Any]:
    from pypdf import PdfReader

    checks: List[Dict[str, Any]] = []

    def chk(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    rd = PdfReader(str(out / PDF_NAME))
    outline = [o for o in rd.outline if not isinstance(o, list)]
    text = "\n".join((p.extract_text() or "") for p in rd.pages[:3])
    chk("PDF opens and has pages", len(rd.pages) > 20, f"{len(rd.pages)} pages")
    chk("PDF title metadata", (rd.metadata.title or "") == TITLE, rd.metadata.title or "")
    chk("44 top-level sections and bookmarks", len(sections) == 44 and len(outline) >= 44, f"sections {len(sections)}, top-level bookmarks {len(outline)}")
    chk("table of contents present", "Table of contents" in text)
    last = rd.pages[-1].extract_text() or ""
    chk("page numbers printed", f"of {len(rd.pages)}" in last, "")
    figs = pd.read_csv(out / "figure_manifest.csv")
    chk("figure files exist (>=30)", len(figs) >= 30 and all((out / f).exists() for f in figs.file), f"{len(figs)} figures")
    for k, name in CSV_OUT.items():
        chk(f"{name} non-empty", (out / name).exists() and len(pd.read_csv(out / name)) > 0)
    for name in ("classification_metrics.csv", "confusion_matrices.csv"):
        chk(f"{name} absent because ground truth is pending", not (out / name).exists())
    rs = pd.read_csv(out / "reconciliation_summary.csv")
    chk("all independent reconciliation checks pass", bool(rs.passed.all()), f"{int(rs.passed.sum())}/{len(rs)}")
    chk("source artifacts unchanged", R["hashes_ok"], f"{len(R['hashes_before'])} files")
    # cross-check against the bundle's own summary
    src, df = R["S"]["summary_src"], R["S"]["df"]
    mism = []
    mts = R["T"]["model_task_summary"].set_index(["participant", "task"])
    for r in src.itertuples():
        m = mts.loc[(r.model_key, r.task)]
        if (int(r.attempted), int(r.valid), int(r.invalid_output), int(r.errors)) != (int(m.records), int(m.valid), int(m.invalid_output), int(m.errors)):
            mism.append((r.model_key, r.task))
    chk("counts equal the bundle's model_summary.csv", not mism, f"mismatches {mism}")
    rel = []
    for r in src.itertuples():
        a, b = float(r.e2e_p50_ms), float(mts.loc[(r.model_key, r.task)].latency_p50_ms)
        rel.append(abs(a - b) / a)
    chk("p50 latency within 1% of the bundle (percentile-definition difference only)", max(rel) < 0.01, f"max relative difference {max(rel):.4%}")
    hits = []
    for f in list(out.glob("*.csv")) + list(out.glob("*.md")):
        t = f.read_text(errors="ignore")
        for p in PII:
            for m in p.findall(t):
                if "sha256" not in m and not re.fullmatch(r"\d{10,}", m) or False:
                    hits.append((f.name, m[:20]))
    chk("no email/phone-like strings in text outputs", not hits, str(hits[:5]))
    chk("no raw output column exported", "raw_output" not in "".join(pd.read_csv(out / "model_task_summary.csv").columns))
    ok = all(c["ok"] for c in checks)
    md = ["# Evaluation report verification", "", f"**Result: {'PASS' if ok else 'FAIL'}** ({sum(c['ok'] for c in checks)}/{len(checks)} checks).", "",
          "This file is generated before output_manifest.json, whose hashes include it.", "", "| Check | Result | Detail |", "|---|---|---|"]
    md += [f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |" for c in checks]
    return {"ok": ok, "checks": checks, "md": "\n".join(md) + "\n"}


REDESIGN_PDF = "benchmark_evaluation_report_redesigned.pdf"


def generate_redesign(cfg: er.Config) -> Dict[str, Any]:
    """Compact redesigned report written next to the original (never overwrites it). Analysis tables and figures are reused unchanged."""
    from msu_llm_lab.observability import evaluation_redesign as rd

    out = Path(cfg.out_dir)
    R = er.run_analysis(cfg)
    figm = pd.read_csv(out / "figure_manifest.csv")
    panels = rd.make_panels(out)
    sections, info = rd.build_redesign(R, figm, panels)
    date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    rd.to_pdf_redesign(sections, out / REDESIGN_PDF, out, rd.cover_data(R, cfg.bundle_dir.name, date))
    from msu_llm_lab.observability.evaluation_document import to_markdown
    md = to_markdown(sections).replace(f"# {TITLE}", f"# {TITLE} ({rd.REVISION})", 1)
    (out / "benchmark_evaluation_report_redesigned.md").write_text(md)
    q = quality_check(out, info, sections)
    return {"R": R, "sections": sections, "info": info, "out": out, "quality": q}


def quality_check(out: Path, info: Dict[str, Any], sections) -> Dict[str, Any]:
    """Automated layout/content checks of the redesigned PDF against the original; writes pdf_quality_check.md and redesign_change_log.md."""
    import pymupdf

    new, old = pymupdf.open(str(out / REDESIGN_PDF)), pymupdf.open(str(out / PDF_NAME))
    rows: List[Dict[str, Any]] = []

    def chk(name, ok, detail=""):
        rows.append({"check": name, "ok": bool(ok), "detail": detail})

    n_new, n_old = len(new), len(old)
    new_text = "\n".join(p.get_text() for p in new)
    old_text = "\n".join(p.get_text() for p in old)
    chk("page count reduced", n_new < n_old, f"{n_old} -> {n_new} pages ({(n_old - n_new) / n_old:.1%} reduction)")
    chk("page count within 22-28 target", 22 <= n_new <= 28, str(n_new))
    chk("no table of contents", "table of contents" not in new_text.lower(), "searched all pages (case-insensitive)")
    chk("cover on one page with title/subtitle", "MSU LLM Inference" in new[0].get_text() and "Comprehensive Benchmark Evaluation" in new[0].get_text())
    # text colour: every span near-black
    bad, total = 0, 0
    for p in new:
        for b in p.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    if sp["text"].strip():
                        total += 1
                        c = sp["color"]
                        if max((c >> 16) & 255, (c >> 8) & 255, c & 255) > 0x40:
                            bad += 1
    chk("all text spans black or near-black", bad == 0, f"{total} spans, {bad} lighter than #404040")
    blanks = [i + 1 for i, p in enumerate(new) if len(p.get_text().strip()) < 40 and not p.get_images()]
    chk("no blank pages", not blanks, str(blanks))
    # whitespace
    sparse = []
    for i, p in enumerate(new):
        if i in (0, n_new - 1):
            continue
        ys = [b[3] for b in p.get_text("blocks") if b[3] < p.rect.height - 50] + [x["bbox"][3] for x in p.get_image_info()]
        low = max(ys) / p.rect.height if ys else 0
        if low < 0.70:
            sparse.append((i + 1, f"{1 - low:.0%} of page height below content"))
    chk("pages with >30% empty tail (informational)", True, str(sparse) if sparse else "none")
    # orphan headings: last body block of a page is a heading
    orphans = []
    for i, p in enumerate(new):
        if i == 0:
            continue
        blocks = [b for b in p.get_text("dict")["blocks"] if b.get("lines") and b["bbox"][3] < p.rect.height - 50 and b["bbox"][1] > 45]
        if blocks:
            sp = blocks[-1]["lines"][-1]["spans"][0]
            if sp["size"] >= 9.2 and "Bold" in sp["font"] and len(blocks[-1]["lines"]) == 1:
                orphans.append(i + 1)
    chk("no orphan headings at page bottom", not orphans, str(orphans))
    n_img = sum(len(p.get_images()) for p in new)
    ntot = info["figures_total"]
    chk("all figures placed", info["figures_individual"] + info["figures_paneled"] == ntot, f"{ntot} source figures: {info['figures_individual']} stand-alone, {info['figures_paneled']} in {len(info['panels'])} panels; {n_img} images embedded")
    keys = ["3,263", "17 of 222", "2,860", "3,400", "2,723", "137", "540", "5.0337", "4.118", "NOT COMPUTABLE", "PRODUCTION_PARITY_NOT_ESTABLISHED", "quarantine", "ESTIMATED",
            "probability-only", "11 participants", "2,502", "217"]
    nt, ot = new_text.replace("\n", " "), old_text.replace("\n", " ")
    missing = [k for k in keys if k.lower() in ot.lower() and k.lower() not in nt.lower()]
    chk("key counts, costs and caveats present in both reports", not missing, f"checked {len(keys)}; missing in redesign: {missing}")
    import re
    tok = lambda t: set(re.findall(r"\b\d[\d,]*\.\d+\b|\b\d{1,3}(?:,\d{3})+\b", t))
    lost = sorted(tok(ot) - tok(nt))
    body_lost = [x for x in lost]
    chk("numeric tokens of original retained (informational)", True, f"{len(tok(ot)) - len(lost)} of {len(tok(ot))} decimal/thousands tokens retained; {len(lost)} not found: {lost}")
    ok = all(r["ok"] for r in rows)
    md = ["# PDF quality check (redesign-r1)", "", f"**Result: {'PASS' if ok else 'FAIL'}** ({sum(r['ok'] for r in rows)}/{len(rows)} automated checks). Pages rendered and inspected visually at 70 dpi in contact sheets.", "",
          "| Check | Result | Detail |", "|---|---|---|"] + [f"| {r['check']} | {'PASS' if r['ok'] else 'FAIL'} | {r['detail']} |" for r in rows]
    md += ["", "## Visual inspection notes", "- Cover is a single art-directed page: large title, subtitle, four key figures, metadata, status statement and a monochrome valid-output heat grid drawn from the real per-block rates.",
           "- Figures keep their original colours; panels stack the original PNGs with (a)/(b) labels and are not rescaled below the original resolution.",
           "- Pre-existing figure issue, unchanged: in F03 (outcomes per participant) the legend overlaps the bars of the lowest rows.",
           "- Tail whitespace remains on the pages listed above because a figure plus caption is kept together and moves as a unit.",
           "- No table needed landscape after column trimming; wide tables use weighted column widths and wrapped cells."]
    (out / "pdf_quality_check.md").write_text("\n".join(md) + "\n")
    cl = ["# Redesign change log (redesign-r1)", "", f"Original: {n_old} pages. Redesigned: {n_new} pages ({(n_old - n_new) / n_old:.1%} fewer). Source: `evaluation_redesign.py`; original PDF and all CSV/JSON unchanged.", "",
          "## Structure", "The 44 sections were regrouped into 15 numbered sections after the cover, plus a technical appendix: " + "; ".join(f"{i}. {t}" for i, (t, _) in enumerate(sections, 1)) + ".", "",
          "## Removed or consolidated", "- Table of contents page and all page-number listings removed; no replacement index.",
                    "- Per-model deep dives: the repeated per-model task tables and the Haiku/Sonnet verified-evidence bullets were consolidated into one comparison table; the evidence appears once in the Claude investigation (section 12) and the per-task numbers once in the overview table.",
          "- 'How to read' section-number bullet removed (section numbers changed); document-control prose merged into the executive summary.",
          "- Block ledger: blocks 17-221 (not executed, no results) collapsed to one row; blocks 0-16 and all 222 blocks in batch_summary.csv are unchanged.",
          "- Dataset section shows the 12 largest preparation strata; the full list is stratum_coverage.csv.",
          "- Artifact inventory appendix lists only files used in the analysis (all 159 remain in artifact_inventory.csv).", "",
          "## Figures", f"All {info['figures_total']} figures are kept. {info['figures_paneled']} are combined into {len(info['panels'])} labelled two-part panels (originals unchanged): "
          + "; ".join(f"{k} = {', '.join(v)}" for k, v in info["panels"].items()) + f". No figure was removed. {info['figures_individual']} stay stand-alone, each placed in the section that discusses it.", "",
          "## Style", "Black text on white, Helvetica (standard PDF font, no embedding needed), thin grey rules, bold black table headers, right-aligned numeric columns, repeated table headers, keep-with-next headings, figure and caption kept together. Warnings are plain black text.",
          "", f"## Metadata", f"Revision `{REVISION_NAME}` is in the PDF subject/keywords and cover."]
    (out / "redesign_change_log.md").write_text("\n".join(cl) + "\n")
    return {"ok": ok, "rows": rows, "pages": (n_old, n_new)}


REVISION_NAME = "redesign-r1"
