"""Evaluation report: statistics helpers (synthetic) and, when the local benchmark artifacts exist, end-to-end reconciliation and determinism."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("reportlab")
from msu_llm_lab.observability import evaluation_report as er  # noqa: E402

ART = Path("reports/real_benchmark_bench-d0a763e4a920-r4/model_results.csv")


def test_wilson_bounds_and_edge_cases():
    lo, hi = er.wilson(170, 170)
    assert 0.97 < lo < 1.0 and hi == pytest.approx(1.0)
    assert np.isnan(er.wilson(0, 0)[0])
    lo, hi = er.wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35


def test_holm_is_monotone_and_capped():
    adj = er.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert all(a <= 1 for a in er.holm([0.9, 0.95]))


def test_kappa_perfect_and_chance():
    assert er.cohen_kappa(list("abab"), list("abab")) == pytest.approx(1.0)
    assert er.cohen_kappa(list("aabb"), list("abab")) == pytest.approx(0.0)


def test_bootstrap_is_seeded_and_deterministic():
    v = np.arange(50, dtype=float)
    assert er.boot_ci(v, np.median, "x") == er.boot_ci(v, np.median, "x")
    assert er.boot_ci(v, np.median, "x") != er.boot_ci(v, np.median, "y")


@pytest.mark.skipif(not ART.exists(), reason="local benchmark artifacts not present")
def test_end_to_end_reconciliation_and_no_ground_truth(tmp_path):
    from msu_llm_lab.observability.evaluation_runner import generate

    res = generate(er.Config(out_dir=tmp_path))
    assert res["verification"]["ok"], [c for c in res["verification"]["checks"] if not c["ok"]]
    assert not (tmp_path / "classification_metrics.csv").exists()
    rec = res["R"]["T"]["request_reconciliation"].set_index("quantity")["value"]
    assert rec["full matrix of executed blocks (units x 10 cases x blocks)"] == 3400
    assert rec["planned new logical requests (plan blocks, executed)"] == 2860
    assert res["R"]["hashes_ok"]
    import hashlib
    h1 = hashlib.sha256((tmp_path / "model_task_summary.csv").read_bytes()).hexdigest()
    res2 = generate(er.Config(out_dir=tmp_path / "again"))
    assert h1 == hashlib.sha256((tmp_path / "again" / "model_task_summary.csv").read_bytes()).hexdigest()
    assert res2["verification"]["ok"]


@pytest.mark.skipif(not ART.exists(), reason="local benchmark artifacts not present")
def test_redesign_is_shorter_has_no_toc_and_keeps_every_figure(tmp_path):
    import pymupdf
    from msu_llm_lab.observability.evaluation_runner import REDESIGN_PDF, PDF_NAME, generate, generate_redesign

    cfg = er.Config(out_dir=tmp_path)
    generate(cfg)
    res = generate_redesign(cfg)
    assert res["quality"]["ok"], [r for r in res["quality"]["rows"] if not r["ok"]]
    new, old = pymupdf.open(str(tmp_path / REDESIGN_PDF)), pymupdf.open(str(tmp_path / PDF_NAME))
    assert len(new) < len(old)
    assert "table of contents" not in "".join(p.get_text() for p in new).lower()
    assert res["info"]["figures_individual"] + res["info"]["figures_paneled"] == res["info"]["figures_total"]
