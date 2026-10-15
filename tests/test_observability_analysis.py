"""Supervised metrics, calibration, agreement and the Lead threshold: labels and splits are respected, nothing is substituted."""

import json

import pytest

from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.records import Task
from msu_llm_lab.observability import analysis as A
from msu_llm_lab.observability.labels import PENDING, load_approved_labels
from msu_llm_lab.observability.monitor import LiveMonitor
from obs_helpers import intent_pred, label_row, lead_pred, make_prep, write_labels, write_preds


def labels_for(tmp_path, rows):
    prep = make_prep(tmp_path / "prep")
    write_labels(prep, rows)
    return prep, load_approved_labels(prep)


def test_without_approved_labels_every_supervised_metric_is_pending(tmp_path):
    runs = A.latest_runs([intent_pred("C1", "QUERY", top=0.9), intent_pred("C2", "GREETING", top=0.9)])
    labels = load_approved_labels(make_prep(tmp_path / "prep"))
    assert labels.status == PENDING and not labels.available
    q = A.intent_quality(runs, labels)[0]
    assert q.status == PENDING and q.metrics is None and q.accuracy_ci is None
    cal = A.calibration_reports(runs, labels)[0]
    assert cal.status == PENDING and cal.ece is None and cal.brier is None and cal.bins == []
    assert cal.mean_selected_prob == pytest.approx(0.9)       # descriptive, label-free figures are still shown


def test_predictions_and_unapproved_rows_never_become_labels(tmp_path):
    rows = [label_row("C1", "INTENT", "QUERY", status="PENDING_REVIEW"), label_row("C2", "INTENT", "QUERY", by=" ")]
    _, labels = labels_for(tmp_path, rows)
    assert labels.intent == {} and labels.n_not_approved == 2 and labels.status == PENDING


def test_quality_metrics_come_only_from_approved_labels(tmp_path):
    preds = [intent_pred("C1", "QUERY"), intent_pred("C2", "GREETING"), intent_pred("C3", "QUERY"), intent_pred("C4", "QUERY", status="INVALID_OUTPUT")]
    _, labels = labels_for(tmp_path, [label_row("C1", "INTENT", "QUERY"), label_row("C2", "INTENT", "QUERY"), label_row("C4", "INTENT", "QUERY")])
    q = A.intent_quality(A.latest_runs(preds), labels)[0]
    assert q.status == "SCORED" and q.metrics.n_scored == 3        # C3 has no approved label: not scored
    assert q.metrics.accuracy == pytest.approx(1 / 3) and q.metrics.invalid_count == 1
    assert any("INSUFFICIENT SAMPLE" in f for f in q.flags) and any("RARE CLASSES" in f for f in q.flags)
    assert q.accuracy_ci is not None and q.accuracy_ci[0] < 1 / 3 < q.accuracy_ci[1]


def test_agreement_is_not_correctness(tmp_path):
    a = [intent_pred(f"C{i}", "QUERY", model="m-a") for i in range(4)]
    b = [intent_pred(f"C{i}", "QUERY", model="m-b") for i in range(4)]
    runs = A.latest_runs(a + b)
    none = load_approved_labels(make_prep(tmp_path / "p0"))
    rep = A.agreement_report(runs, Task.INTENT, none)
    assert rep.pairs[0].agreement == 1.0 and rep.unanimous_errors is None and rep.unanimous_error_basis == PENDING
    _, labels = labels_for(tmp_path, [label_row(f"C{i}", "INTENT", "GREETING") for i in range(4)])
    rep = A.agreement_report(runs, Task.INTENT, labels)
    assert rep.pairs[0].agreement == 1.0                         # they agree ...
    assert rep.unanimous_errors == ["C0", "C1", "C2", "C3"]       # ... and are all wrong
    assert "4 approved-label cases shared by 2 models" in rep.unanimous_error_basis


def test_agreement_lists_contributors_disagreements_and_frontier_agreement():
    cases = [f"C{i}" for i in range(6)]
    a = [intent_pred(c, "QUERY", model="cand") for c in cases]
    f = [intent_pred(c, "QUERY" if i < 4 else "GREETING", model="gpt-4.1") for i, c in enumerate(cases)]
    rep = A.agreement_report(A.latest_runs(a + f), Task.INTENT, load_approved_labels(None))
    assert rep.models == ["cand", "gpt-4.1"] and rep.n_cases_all_models == 6
    assert [d[0] for d in rep.disagreements] == ["C4", "C5"]
    assert rep.frontier_agreement["cand"]["gpt-4.1"] == pytest.approx(4 / 6)
    assert rep.pairs[0].note.startswith("n=6 < 30")


def test_kappa_is_chance_corrected_and_undefined_without_variation():
    assert A.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert A.cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert A.cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == -1.0


def test_calibration_requires_a_native_distribution(tmp_path):
    textual = [intent_pred("C1", "QUERY", native_confidence=0.9), intent_pred("C2", "QUERY", native_confidence=0.9)]
    rep = A.calibration_reports(A.latest_runs(textual), load_approved_labels(None))[0]
    assert rep.status == "UNAVAILABLE" and "never used" in rep.unsupported_reason and rep.ece is None


def test_calibration_with_labels_reports_ece_brier_coverage_and_high_confidence_errors(tmp_path):
    preds = [intent_pred(f"C{i}", "QUERY", top=0.95) for i in range(6)]
    rows = [label_row(f"C{i}", "INTENT", "QUERY" if i < 4 else "GREETING") for i in range(6)]
    _, labels = labels_for(tmp_path, rows)
    rep = A.calibration_reports(A.latest_runs(preds), labels)[0]
    assert rep.status == "SCORED" and rep.n_calibration == 6 and rep.ece == pytest.approx(abs(0.95 - 4 / 6))
    assert rep.brier is not None and len(rep.high_conf_errors) == 2 and rep.coverage[0][1] == 6
    assert any("INSUFFICIENT SAMPLE" in f for f in rep.flags)


def test_lead_threshold_is_selected_on_dev_only_and_frozen_before_test(tmp_path):
    prep = make_prep(tmp_path / "prep")
    lines = []
    cases = {"D1": ("DEV", True, 0.9), "D2": ("DEV", True, 0.8), "D3": ("DEV", False, 0.2), "D4": ("DEV", False, 0.1),
             "T1": ("TEST", True, 0.7), "T2": ("TEST", False, 0.6)}
    for cid, (split, _, _) in cases.items():
        lines.append(json.dumps({"case": {"case_id": cid, "history_status": "COMPLETE"},
                                 "candidate": {"split": split, "view": "NATURAL", "ambiguity_count": 0, "unresolved_rules": [], "builder_tally_not_a_label": {"tier_a_actions": 1, "tier_b_distinct_types": 0}}}))
    (prep / "lead_candidates_2h.jsonl").write_text("\n".join(lines) + "\n")
    write_labels(prep, [label_row(c, "LEAD", v[1]) for c, v in cases.items()])
    recs = [lead_pred(c, yes_prob=v[2]) for c, v in cases.items()]
    labels, meta = load_approved_labels(prep), A.load_case_meta(prep)
    rep = A.lead_reports(A.latest_runs(recs), labels, meta)[0]
    assert rep.frozen is not None and rep.frozen.selected_on == "DEV" and rep.frozen.n_dev_scored == 4
    assert rep.held_out is not None and rep.held_out.n_scored == 2 and rep.held_out.frozen_threshold == rep.frozen.threshold
    # changing TEST-only probabilities cannot move the frozen threshold
    recs2 = [lead_pred(c, yes_prob=(0.99 if c.startswith("T") else v[2])) for c, v in cases.items()]
    rep2 = A.lead_reports(A.latest_runs(recs2), labels, meta)[0]
    assert rep2.frozen.threshold == rep.frozen.threshold


def test_lead_without_a_frozen_threshold_gives_undecided_not_yes_no():
    recs = [lead_pred("L1", yes_prob=0.9), lead_pred("L2", yes_prob=0.1)]
    rep = A.lead_reports(A.latest_runs(recs), load_approved_labels(None), {})[0]
    assert rep.frozen is None and (rep.n_yes, rep.n_no, rep.n_undecided) == (0, 0, 2) and rep.status == PENDING


def test_repeatability_and_latest_run_selection():
    p1 = [intent_pred("C1", "QUERY", run="r1", created="2026-10-15T12:00:00+00:00"), intent_pred("C2", "QUERY", run="r1", created="2026-10-15T12:00:00+00:00")]
    p2 = [intent_pred("C1", "QUERY", run="r2", created="2026-10-15T13:00:00+00:00"), intent_pred("C2", "GREETING", run="r2", created="2026-10-15T13:00:00+00:00")]
    assert {r.run_id for r in A.latest_runs(p1 + p2)[("jev-1.14", "INTENT")]} == {"r2"}
    rep = A.repeatability(p1 + p2)[0]
    assert (rep.n_cases_repeated, rep.n_identical) == (2, 1)


def test_label_loader_and_monitor_refresh_are_read_only(tmp_path):
    prep = make_prep(tmp_path / "data" / "classification_prep")
    write_labels(prep, [label_row("C1", "INTENT", "QUERY")])
    preds_dir = tmp_path / "reports"
    write_preds(preds_dir, "run1", [intent_pred("C1", "QUERY", top=0.9)])
    before = {p: p.read_bytes() for p in prep.rglob("*") if p.is_file()}
    mtimes = {p: p.stat().st_mtime_ns for p in before}
    mon = LiveMonitor(events_path=tmp_path / "e.jsonl", prediction_dirs=[preds_dir], dataset_base=tmp_path / "data" / "classification_prep")
    for _ in range(3):
        mon.poll()
    assert mon.analysis.labels.intent == {"C1": IntentLabel.QUERY}
    assert {p: p.read_bytes() for p in prep.rglob("*") if p.is_file()} == before
    assert {p: p.stat().st_mtime_ns for p in before} == mtimes
    assert not (tmp_path / "e.jsonl").exists()                  # polling never creates the telemetry log either


def test_errors_and_invalid_outputs_are_not_counted_as_disagreements():
    a = [intent_pred(f"C{i}", "QUERY", model="m-a") for i in range(4)]
    b = [intent_pred("C0", "QUERY", model="m-b"), intent_pred("C1", "QUERY", model="m-b", status="INVALID_OUTPUT"),
         intent_pred("C2", "GREETING", model="m-b"), intent_pred("C3", "QUERY", model="m-b")]
    rep = A.agreement_report(A.latest_runs(a + b), Task.INTENT, load_approved_labels(None))
    pair = rep.pairs[0]
    assert pair.n_common == 3 and pair.n_unusable == 1                       # the invalid output is set aside, not a vote
    assert pair.agreement == pytest.approx(2 / 3) and [d[0] for d in rep.disagreements] == ["C2"]
