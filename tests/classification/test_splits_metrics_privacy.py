import math
import pathlib
import re

import pytest

from msu_llm_lab.classification import splits
from msu_llm_lab.classification.contracts import IntentLabel as L
from msu_llm_lab.classification.metrics import (
    ConfidenceSource as CS, IntentItem, LeadItem, MetricsError, brier_multiclass,
    expected_calibration_error, intent_metrics, lead_metrics,
)
from msu_llm_lab.classification.privacy import (
    assert_dispatch_eligible, assert_synthetic_fixture, contains_contact_details, redact_for_log,
)
from msu_llm_lab.providers.base import ConfidentialityViolationError
from msu_llm_lab.schemas.evaluation import DataClassification as DC, DatasetSplit as S

RATIOS = {S.TRAIN: 0.5, S.DEV: 0.25, S.TEST: 0.25}
GROUPS = [f"syn-g-{i}" for i in range(40)]


# ---------------- splits ----------------
def test_split_deterministic_and_seed_sensitive():
    a = splits.assign_group_splits(GROUPS, RATIOS, "seed-1")
    assert a == splits.assign_group_splits(list(reversed(GROUPS)), RATIOS, "seed-1")
    assert a != splits.assign_group_splits(GROUPS, RATIOS, "seed-2")
    counts = {s: sum(1 for v in a.values() if v == s) for s in RATIOS}
    assert counts == {S.TRAIN: 20, S.DEV: 10, S.TEST: 10}


def test_no_group_leaks_across_splits():
    cases = {f"c{i}": GROUPS[i % 10] for i in range(60)}  # many cases share a group
    group_split = splits.assign_group_splits(list(cases.values()), RATIOS, "s")
    case_split = {c: group_split[g] for c, g in cases.items()}
    assert splits.find_group_leaks(cases, case_split) == []
    case_split["c0"] = S.TEST if case_split["c0"] != S.TEST else S.TRAIN
    assert splits.find_group_leaks(cases, case_split) == [cases["c0"]]


def test_split_input_validation():
    with pytest.raises(splits.SplitError):
        splits.assign_group_splits(GROUPS, RATIOS, "")
    with pytest.raises(splits.SplitError):
        splits.assign_group_splits(["a", ""], RATIOS, "s")
    with pytest.raises(splits.SplitError):
        splits.assign_group_splits(GROUPS, {S.TRAIN: 0.5, S.TEST: 0.4}, "s")
    with pytest.raises(splits.SplitError):
        splits.assign_group_splits(GROUPS, {S.UNASSIGNED: 1.0}, "s")


def test_user4_rejected_and_views_disjoint():
    with pytest.raises(splits.SplitError):
        splits.assert_no_user4(["case_id", "User4"])
    splits.assert_no_user4(["case_id", "group_id"])
    splits.assert_views_disjoint({"a", "b"}, {"c"})
    with pytest.raises(splits.SplitError):
        splits.assert_views_disjoint({"a", "b"}, {"b"})


# ---------------- metrics ----------------
def test_intent_metrics_hand_computed():
    items = [
        IntentItem(L.QUERY, L.QUERY), IntentItem(L.QUERY, L.QUERY), IntentItem(L.QUERY, L.GREETING),
        IntentItem(L.GREETING, L.GREETING), IntentItem(L.GREETING, None),
        IntentItem(L.OUT_OF_SCOPE, L.QUERY),
    ]
    m = intent_metrics(items)
    assert (m.n_scored, m.invalid_count) == (6, 1)
    assert m.accuracy == pytest.approx(3 / 6) and m.invalid_rate == pytest.approx(1 / 6)
    q = m.per_class["QUERY"]
    assert (q.precision, q.recall, q.support) == (pytest.approx(2 / 3), pytest.approx(2 / 3), 3)
    g = m.per_class["GREETING"]
    assert (g.precision, g.recall) == (pytest.approx(1 / 2), pytest.approx(1 / 2))
    assert m.per_class["OUT_OF_SCOPE"].f1 == 0.0 and m.per_class["HUMAN_HANDOFF_REQUEST"].f1 is None
    assert m.macro_f1 == pytest.approx((2 / 3 + 1 / 2 + 0.0) / 3)
    assert m.confusion["GREETING"]["INVALID"] == 1 and m.confusion["QUERY"]["GREETING"] == 1


def test_exclusions_and_unlabeled_never_scored():
    items = [
        IntentItem(L.QUERY, L.QUERY),
        IntentItem(L.HUMAN_HANDOFF_REQUEST, L.QUERY, exclusion_reason="AMBIGUOUS_HANDOFF_RULE"),
        IntentItem(None, L.QUERY),
    ]
    m = intent_metrics(items)
    assert (m.n_total, m.n_excluded, m.n_unlabeled, m.n_scored, m.accuracy) == (3, 1, 1, 1, 1.0)


def test_empty_metrics_are_none_not_zero():
    m = intent_metrics([])
    assert m.accuracy is None and m.macro_f1 is None and m.invalid_rate is None


def test_lead_metrics_hand_computed():
    items = [LeadItem(True, True), LeadItem(True, True), LeadItem(True, False), LeadItem(True, None),
             LeadItem(False, True), LeadItem(False, False), LeadItem(False, False), LeadItem(False, None)]
    m = lead_metrics(items)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 2, 2)
    assert m.yes_precision == pytest.approx(2 / 3) and m.yes_recall == pytest.approx(2 / 4)
    assert m.yes_f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m.false_positive_rate == pytest.approx(1 / 4) and m.false_negative_rate == pytest.approx(2 / 4)
    assert m.accuracy == pytest.approx(4 / 8) and m.invalid_count == 2


def test_calibration_requires_native_confidence():
    for bad in (CS.SELF_REPORTED_TEXT, CS.NONE):
        with pytest.raises(MetricsError):
            expected_calibration_error([0.9], [True], 1, bad)
        with pytest.raises(MetricsError):
            brier_multiclass([{"a": 1.0}], ["a"], bad)


def test_ece_and_brier_fixed_values():
    ece = expected_calibration_error([0.9, 0.9, 0.6, 0.6], [True, True, True, False], 2, CS.NATIVE_DISTRIBUTION)
    assert ece == pytest.approx(0.5 * abs(0.6 - 0.5) + 0.5 * abs(0.9 - 1.0))
    b = brier_multiclass([{"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}], ["a", "b"], CS.NATIVE_DISTRIBUTION)
    assert b == pytest.approx(((0.3 ** 2 + 0.3 ** 2) + (0.2 ** 2 + 0.2 ** 2)) / 2)
    with pytest.raises(MetricsError):
        brier_multiclass([{"a": 0.5, "b": 0.2}], ["a"], CS.NATIVE_DISTRIBUTION)
    with pytest.raises(MetricsError):
        expected_calibration_error([1.2], [True], 1, CS.NATIVE_DISTRIBUTION)


# ---------------- privacy ----------------
@pytest.mark.parametrize("c", [DC.INTERNAL, DC.CONFIDENTIAL, DC.UNCLASSIFIED])
def test_gate_rejects_non_public_synthetic(c):
    with pytest.raises(ConfidentialityViolationError):
        assert_dispatch_eligible(c)


def test_gate_allows_only_public_and_synthetic_and_fixtures_are_synthetic():
    assert_dispatch_eligible(DC.PUBLIC)
    assert_dispatch_eligible(DC.SYNTHETIC)
    assert_synthetic_fixture(DC.SYNTHETIC)
    with pytest.raises(ConfidentialityViolationError):
        assert_synthetic_fixture(DC.PUBLIC)


def test_redaction_and_contact_detection():
    text = "call me on 012-345 6789 or syn@example.test"
    assert contains_contact_details(text) and not contains_contact_details("what are the fees")
    out = redact_for_log(text)
    assert "012" not in out and "example" not in out and out.startswith("<redacted len=")


# ---------------- synthetic-only / no-network evidence ----------------
PKG = pathlib.Path(__file__).resolve().parents[2] / "src" / "msu_llm_lab" / "classification"
FORBIDDEN_IMPORTS = re.compile(r"^\s*(?:import|from)\s+(?:socket|urllib|http|requests|httpx|aiohttp|ftplib|smtplib|csv)\b", re.M)
NETWORK_IMPORTS = re.compile(r"^\s*(?:import|from)\s+(?:socket|urllib|http|requests|httpx|aiohttp|ftplib|smtplib)\b", re.M)
NETWORKED_MODULES = {"jev_adapter.py", "chat_adapter.py", "smoke.py", "runner.py"}  # reach providers only through the shared poster / providers package


def test_package_has_no_network_or_csv_imports_and_no_real_data_reference():
    for f in PKG.glob("*.py"):
        src = f.read_text()
        # only the local data-preparation module may read a CSV; nothing in the package may import a network library directly
        assert not (NETWORK_IMPORTS if f.name == "dataprep.py" else FORBIDDEN_IMPORTS).search(src), f.name
        assert "september_to_october" not in src, f.name


def test_network_is_actually_blocked_in_this_suite():
    import socket
    with pytest.raises(AssertionError):
        socket.create_connection(("example.invalid", 80))


def test_fixtures_are_all_synthetic():
    from synthetic_fixtures import intent_case
    assert intent_case().classification == DC.SYNTHETIC
    src = (pathlib.Path(__file__).parent / "synthetic_fixtures.py").read_text()
    assert "SYNTHETIC" in src and ".csv" not in src
