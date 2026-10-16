"""Labelled evaluation: leakage isolation, metric denominators, splits and threshold handling, all on synthetic data."""
import json

import pandas as pd
import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
from msu_llm_lab.labeled_eval import data as D  # noqa: E402
from msu_llm_lab.labeled_eval import run as R  # noqa: E402
from msu_llm_lab.labeled_eval import stats as S  # noqa: E402


def _intent(n=40):
    labs = list(D.INTENT_LABELS)
    return pd.DataFrame({"case_id": [f"I{i}" for i in range(n)], "history": ["User: hi\nAssistant: hello" if i % 3 else "" for i in range(n)],
                         "current_message": [f"message {i}" for i in range(n)], "expected_label": [labs[i % 5] for i in range(n)]})


def _lead(n=40):
    return pd.DataFrame({"case_id": [f"L{i}" for i in range(n)], "action_summary": [f"Asked about course {i}" for i in range(n)], "user_msg": [f"msg {i}" for i in range(n)],
                         "expected_cta": ["true" if i % 2 else "false" for i in range(n)]})


def test_inference_inputs_never_contain_labels():
    i, l = _intent(), _lead()
    inp = R.inference_inputs(i, l, {"intent": {c: "test" for c in i.case_id}, "lead": {c: "dev" for c in l.case_id}})
    blob = json.dumps(inp)
    for f in D.LABEL_FIELDS:
        assert f not in blob
    for row in inp["INTENT"] + inp["LEAD"]:
        assert not (set(row) & set(D.LABEL_FIELDS))
    assert "user_msg" not in inp["LEAD"][0]            # contract: action summary only


def test_history_parser_handles_continuations_and_empty():
    t = D.parse_history("User: a\nstill a\nAssistant: b")
    assert t == [("USER", "a\nstill a"), ("ASSISTANT", "b")]
    assert D.parse_history("") == []


def test_split_is_deterministic_stratified_and_keeps_families_together():
    l = _lead(60)
    l.loc[1, "action_summary"] = l.loc[0, "action_summary"]            # a family of two
    a = D.stratified_group_split(l, "expected_cta", lambda r: D.norm(r.action_summary), 0.5)
    b = D.stratified_group_split(l, "expected_cta", lambda r: D.norm(r.action_summary), 0.5)
    assert a == b and a["L0"] == a["L1"]
    pos_dev = sum(1 for c, s in a.items() if s == "dev" and l.set_index("case_id").expected_cta[c] == "true")
    assert 12 <= pos_dev <= 18


def test_multiclass_counts_invalid_as_miss_and_reports_both_views():
    yt = ["QUERY", "QUERY", "GREETING", "GREETING"]
    yp = ["QUERY", None, "GREETING", "QUERY"]
    r = S.multiclass(yt, yp, list(D.INTENT_LABELS), n_expected=4)
    assert r["accuracy_end_to_end"] == pytest.approx(0.5) and r["accuracy_conditional"] == pytest.approx(2 / 3)
    assert r["n_valid"] == 3 and "UPLOAD_STATUS_CHECK" in r["unstable_classes"]
    assert r["confusion_matrix"]["QUERY"]["<INVALID>"] == 1


def test_binary_missing_positive_counts_as_false_negative_and_denominators():
    r = S.binary([True, True, False, False], [True, None, False, True], n_expected=4)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["cta_recall"] == pytest.approx(0.5) and r["false_positive_rate"] == pytest.approx(0.5)


def test_threshold_choice_uses_dev_scores_only_and_is_deterministic():
    dev_y, dev_s = [True, True, False, False, True, False], [0.9, 0.7, 0.4, 0.2, 0.6, 0.55]
    rows = S.threshold_sweep(dev_y, dev_s, [0.3, 0.5, 0.65, 0.8])
    t = S.choose_threshold_max_f1(rows)
    assert t == S.choose_threshold_max_f1(rows) and t in (0.5, 0.65)


def _preds(case_ids, status="OK", **kw):
    d = pd.DataFrame({"case_id": case_ids, "status": status})
    for k, v in kw.items():
        d[k] = v
    return d


def test_scorer_joins_by_id_keeps_missing_as_unattempted_and_rejects_duplicates():
    key = pd.DataFrame({"task": "INTENT", "case_id": ["a", "b", "c"], "expected": ["QUERY", "GREETING", "QUERY"]})
    split = {"a": "test", "b": "test", "c": "test"}
    p = _preds(["a", "b"], predicted_label=["QUERY", "QUERY"])
    row = R.score_model_task(p, key, "INTENT", split, "test", meta={"model_id": "m"}, n_boot=50)
    assert row["n_cases_expected"] == 3 and row["n_unattempted"] == 1 and row["coverage_rate"] == pytest.approx(2 / 3)
    assert row["accuracy"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        R.score_model_task(pd.concat([p, p]), key, "INTENT", split, "test")
    with pytest.raises(ValueError):
        R.score_model_task(_preds(["zzz"], predicted_label=["QUERY"]), key, "INTENT", split, "test")


def test_lead_frozen_threshold_is_applied_unchanged_and_cost_stays_estimated():
    key = pd.DataFrame({"task": "LEAD", "case_id": ["a", "b", "c", "d"], "expected": ["true", "true", "false", "false"]})
    split = {c: "test" for c in "abcd"}
    p = _preds(list("abcd"), yes_prob=[0.9, 0.4, 0.3, 0.6], predicted_need_cta=[None] * 4, estimated_cost_usd=[0.001] * 4)
    r = R.score_model_task(p, key, "LEAD", split, "test", threshold=0.5, meta={"model_id": "jev"})
    assert json.loads(r["lead_confusion_matrix_json"]) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "missing_positive": 0, "missing_negative": 0}
    assert "ESTIMATED" in r["cost_provenance"] and r["provider_reported_cost_usd"] == ""


def test_labeled_eval_package_has_no_direct_network_access_and_only_the_executor_reaches_the_runner():
    import pathlib
    import re
    for f in pathlib.Path("src/msu_llm_lab/labeled_eval").glob("*.py"):
        t = f.read_text()
        assert not re.search(r"^\s*(import|from)\s+(urllib|requests|http\.client|socket|msu_llm_lab\.providers\.(openrouter|http))", t, re.M), f
        assert "OPENROUTER_API_KEY" not in t
        if f.name != "execute.py":
            assert not re.search(r"^\s*(import|from)\s+msu_llm_lab\.classification\.(runner|smoke|jev_adapter)", t, re.M), f.name


def test_threshold_is_chosen_on_dev_only_frozen_and_unaffected_by_test_labels(tmp_path):
    from msu_llm_lab.labeled_eval import deliver
    ids = [f"L{i}" for i in range(40)]
    split_of = {c: ("dev" if i < 20 else "test") for i, c in enumerate(ids)}
    exp = ["true" if i % 2 else "false" for i in range(40)]
    key = pd.DataFrame({"task": "LEAD", "case_id": ids, "expected": exp})
    scores = [0.8 if e == "true" else 0.2 for e in exp]
    g = pd.DataFrame({"case_id": ids, "status": "OK", "yes_prob": scores})
    a = deliver.jev_threshold(g, key, split_of, "jev")
    flipped = key.copy()
    flipped.loc[20:, "expected"] = ["false" if e == "true" else "true" for e in key.expected[20:]]
    b = deliver.jev_threshold(g, flipped, split_of, "jev")
    assert a["supported"] and a["threshold"] == b["threshold"]                    # TEST labels cannot move the DEV threshold
    assert a["test_frozen"]["f1"] > b["test_frozen"]["f1"]                         # but they do change the TEST result
    assert deliver.jev_threshold(g.assign(yes_prob=0.5), key.assign(expected="true"), split_of, "jev")["supported"] is False


def test_score_all_never_invents_a_threshold_for_single_class_dev(tmp_path):
    from msu_llm_lab.labeled_eval import deliver
    ids = [f"L{i}" for i in range(12)]
    key = pd.DataFrame({"task": "LEAD", "case_id": ids, "expected": ["true"] * 12})
    g = pd.DataFrame({"case_id": ids, "status": "OK", "yes_prob": [0.6] * 12})
    r = deliver.jev_threshold(g, key, {c: "dev" for c in ids}, "jev")
    assert r["supported"] is False and "single-class" in r["reason"]


def test_master_metrics_match_hand_computed_confusion_values():
    from msu_llm_lab.labeled_eval import master as M
    classes = ["a", "b"]
    m = M.confusion(["a", "a", "a", "b", "b", "b"], ["a", "a", "b", "b", "b", M.INVALID], classes)
    agg = M.aggregate(m, classes)
    assert agg["accuracy"] == pytest.approx(4 / 6) and agg["correct"] == 4
    pc = {r["label"]: r for r in M.per_class(m, classes)}
    assert pc["a"]["precision"] == pytest.approx(1.0) and pc["a"]["recall"] == pytest.approx(2 / 3) and pc["b"]["precision"] == pytest.approx(2 / 3) and pc["b"]["recall"] == pytest.approx(2 / 3)
    assert agg["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert agg["micro_recall"] == pytest.approx(4 / 6) and agg["micro_precision"] == pytest.approx(4 / 5)
    assert -1 <= agg["mcc"] <= 1 and agg["balanced_accuracy"] == pytest.approx(2 / 3)
    perfect = M.aggregate(M.confusion(["a", "b", "a"], ["a", "b", "a"], classes), classes)
    assert perfect["mcc"] == pytest.approx(1.0) and perfect["cohen_kappa"] == pytest.approx(1.0)


def test_invalid_outputs_lower_effective_metrics_and_are_never_correct():
    from msu_llm_lab.labeled_eval import master as M
    classes = ["a", "b"]
    all_valid = M.aggregate(M.confusion(["a", "b"], ["a", "b"], classes), classes)
    one_invalid = M.aggregate(M.confusion(["a", "b"], ["a", M.INVALID], classes), classes)
    assert all_valid["accuracy"] == 1.0 and one_invalid["accuracy"] == pytest.approx(0.5) and one_invalid["macro_f1"] < all_valid["macro_f1"]


def test_composite_normalises_over_eligible_models_only_and_floor_blocks_recommendation():
    from msu_llm_lab.labeled_eval import master as M
    g = pd.DataFrame({"model_id": ["good", "fast_broken", "slow"], "macro_f1": [0.8, 0.05, 0.7], "effective_accuracy": [0.9, 0.05, 0.8], "valid_output_rate": [1.0, 0.1, 1.0],
                      "p50_latency_ms": [1000.0, 300.0, 2000.0], "estimated_cost_per_1000_cases_usd": [1.0, 0.1, 2.0]})
    s = M.composite(g, "INTENT", M.WEIGHTS["INTENT"])
    assert s.idxmax() == 0 and s[1] < s[0]                       # a broken fast/cheap model cannot win
    t = pd.DataFrame({"model_id": g.model_id, "split": "test", "valid_output_rate": g.valid_output_rate, "request_failure_rate": 0.0, "sample_count": 120, "test_sample_support": "n=120", "composite_score": s})
    out = M.recommendation({"INTENT": t})["INTENT"]
    assert out.recommendation.iloc[1] == "NOT_RECOMMENDED_RELIABILITY" and out.recommendation.iloc[0].startswith("BEST_OBSERVED_CANDIDATE")


def test_pairwise_mcnemar_uses_the_same_cases_and_holm():
    from msu_llm_lab.labeled_eval import master as M
    ids = [f"c{i}" for i in range(40)]
    a = pd.Series([True] * 40, index=ids)
    b = pd.Series([True] * 20 + [False] * 20, index=ids)
    c = pd.Series([True] * 38 + [False] * 2, index=ids)
    pw = M.pairwise({("INTENT", "test", "a"): a, ("INTENT", "test", "b"): b, ("INTENT", "test", "c"): c})
    ab = pw[(pw.model_a == "a") & (pw.model_b == "b")].iloc[0]
    assert ab.a_only_correct == 20 and ab.b_only_correct == 0 and ab.significant_at_0_05_after_holm if False else ab["significant_at_0.05_after_holm"]
    ac = pw[(pw.model_a == "a") & (pw.model_b == "c")].iloc[0]
    assert ac.adjusted_p_value_holm >= ac.p_value and not ac["significant_at_0.05_after_holm"]


def test_report_numeric_coercion_keeps_status_strings_out_of_numbers_and_ids_follow_appearance():
    from msu_llm_lab.labeled_eval import report as RP
    df = pd.DataFrame({"a": ["0.5", "NOT_AVAILABLE", "0.25"], "b": ["x", "y", "z"], "h": ["0cc06b", "ada2f3", "ef57ac"]})
    out = RP._numeric(df.copy())
    assert out.a.tolist()[0] == 0.5 and out.a.isna().tolist() == [False, True, False] and out.b.tolist() == ["x", "y", "z"] and out.h.tolist() == ["0cc06b", "ada2f3", "ef57ac"]
    reg = [{"file": "figures/b.png", "figure_id": "L01"}, {"file": "figures/a.png", "figure_id": "L02"}]
    secs = [("s", [("fig", {"file": "figures/a.png", "caption": "L02. A. x"}), ("fig", {"file": "figures/b.png", "caption": "L01. B. y"})])]
    new = RP.renumber(secs, reg)
    assert secs[0][1][0][1]["caption"].startswith("F01.") and secs[0][1][1][1]["caption"].startswith("F02.") and [r["figure_id"] for r in new] == ["F01", "F02"]
