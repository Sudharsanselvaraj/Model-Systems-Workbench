import pytest

from msu_llm_lab.classification.threshold import (
    REFERENCE_THRESHOLD, FrozenThreshold, ProbItem, ThresholdError, average_precision, brier_binary,
    candidate_thresholds, evaluate_at, evaluate_held_out, roc_auc, select_threshold_on_dev, sweep,
)
from msu_llm_lab.schemas.evaluation import DatasetSplit as S

DEV, TEST = S.DEV, S.TEST


def items(spec, split=DEV, prefix="c"):
    return [ProbItem(f"{prefix}{i}", exp, p, split) for i, (exp, p) in enumerate(spec)]


# synthetic: positives 0.9, 0.8, 0.4 ; negatives 0.6, 0.3, 0.2
DEV_SPEC = [(True, 0.9), (True, 0.8), (True, 0.4), (False, 0.6), (False, 0.3), (False, 0.2)]


def test_evaluate_at_counts_use_ge_threshold():
    r = evaluate_at(items(DEV_SPEC), 0.5)
    assert (r.tp, r.fp, r.fn, r.tn) == (2, 1, 1, 2)
    assert r.precision == pytest.approx(2 / 3) and r.recall == pytest.approx(2 / 3) and r.f1 == pytest.approx(2 / 3)
    assert r.false_positive_rate == pytest.approx(1 / 3) and r.false_negative_rate == pytest.approx(1 / 3)
    assert evaluate_at(items(DEV_SPEC), 0.9).tp == 1  # >= includes equality


def test_candidates_are_midpoints_plus_reference_and_deterministic():
    c = candidate_thresholds(items(DEV_SPEC))
    assert REFERENCE_THRESHOLD in c and c == sorted(c) and c == candidate_thresholds(list(reversed(items(DEV_SPEC))))
    assert pytest.approx(0.35) in c and pytest.approx(0.85) in c


def test_selection_maximises_f1_on_dev_and_reports_reference():
    f = select_threshold_on_dev(items(DEV_SPEC))
    # cut at 0.35 keeps all three positives and one false positive (0.6): F1 = 2*0.75*1/(1.75) = 0.857
    assert f.threshold == pytest.approx(0.35) and f.dev_f1_at_threshold == pytest.approx(6 / 7)
    assert f.criterion == "MAX_F1" and f.selected_on == "DEV" and f.n_dev_scored == 6
    assert f.dev_f1_at_reference == pytest.approx(2 / 3)
    assert select_threshold_on_dev(items(DEV_SPEC)) == f  # deterministic


def test_tie_break_prefers_higher_precision_then_higher_threshold():
    # two perfect separations are impossible to tell apart; identical F1 resolves to the higher threshold
    spec = [(True, 0.9), (True, 0.8), (False, 0.2), (False, 0.1)]
    f = select_threshold_on_dev(items(spec))
    assert f.dev_f1_at_threshold == 1.0 and f.threshold == pytest.approx(0.5)  # 0.5 is a candidate and the highest perfect cut in range


def test_selection_refuses_non_dev_items():
    mixed = items(DEV_SPEC) + [ProbItem("t1", True, 0.7, TEST)]
    with pytest.raises(ThresholdError):
        select_threshold_on_dev(mixed)
    with pytest.raises(ThresholdError):
        select_threshold_on_dev(items(DEV_SPEC, split=TEST))


def test_selection_needs_both_classes_and_probabilities():
    with pytest.raises(ThresholdError):
        select_threshold_on_dev(items([(True, 0.9), (True, 0.8)]))
    with pytest.raises(ThresholdError):
        select_threshold_on_dev(items([(True, None), (False, None)]))
    with pytest.raises(ThresholdError):
        select_threshold_on_dev([])


def test_unlabelled_excluded_and_missing_prob_not_imputed():
    spec = items(DEV_SPEC) + [ProbItem("u", None, 0.5, DEV), ProbItem("x", True, 0.1, DEV, exclusion_reason="AMBIGUOUS"),
                              ProbItem("m", True, None, DEV)]
    assert select_threshold_on_dev(spec).n_dev_scored == 6
    with pytest.raises(ThresholdError):
        evaluate_at([ProbItem("bad", True, 1.5, DEV)], 0.5)


def test_held_out_needs_frozen_threshold_and_test_split_only():
    frozen = select_threshold_on_dev(items(DEV_SPEC))
    test = items([(True, 0.7), (True, 0.3), (False, 0.6), (False, 0.1)], split=TEST, prefix="t")
    with pytest.raises(ThresholdError):
        evaluate_held_out(test, None)
    with pytest.raises(ThresholdError):
        evaluate_held_out(items(DEV_SPEC), frozen)
    rep = evaluate_held_out(test, frozen)
    assert rep.frozen_threshold == frozen.threshold
    assert (rep.at_frozen.tp, rep.at_frozen.fp, rep.at_frozen.fn, rep.at_frozen.tn) == (1, 1, 1, 1)
    assert rep.at_reference.threshold == 0.5


def test_frozen_threshold_is_immutable_and_unaffected_by_held_out_results():
    frozen = select_threshold_on_dev(items(DEV_SPEC))
    with pytest.raises(Exception):
        frozen.threshold = 0.1
    before = frozen.threshold
    evaluate_held_out(items([(True, 0.99), (False, 0.01)], split=TEST, prefix="t"), frozen)
    assert frozen.threshold == before
    assert isinstance(frozen, FrozenThreshold)


def test_probability_metrics_hand_computed():
    it = items([(True, 0.9), (True, 0.4), (False, 0.6), (False, 0.1)])
    assert roc_auc(it) == pytest.approx(0.75)  # 3 of 4 pos/neg pairs ordered correctly
    assert average_precision(it) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert brier_binary(it) == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)
    assert roc_auc(items([(True, 0.5), (False, 0.5)])) == 0.5  # ties count half
    assert roc_auc(items([(True, 0.5)])) is None and average_precision(items([(False, 0.5)])) is None


def test_sweep_is_descriptive_for_any_split():
    rows = sweep(items(DEV_SPEC, split=TEST))
    assert rows and all(r.threshold >= 0 for r in rows)
