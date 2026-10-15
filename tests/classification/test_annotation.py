"""Annotation workflow: independence, adjudication, explicit approval, revocation, freeze, and protection of the label file."""

import json

import pytest
from typer.testing import CliRunner

from msu_llm_lab.classification import annotation as an
from msu_llm_lab.classification import benchmark as bm
from msu_llm_lab.classification import dataprep as dp
from msu_llm_lab.classification.records import Task
from msu_llm_lab.cli import app
from msu_llm_lab.observability.labels import load_approved_labels
from test_dataprep import write_csv


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv(dp.SALT_ENV, "private-test-secret-0123456789")
    src = tmp_path / "s.csv"
    write_csv(src)
    dp.prepare(src, tmp_path / "prep-t")
    prep = tmp_path / "prep-t"
    loaded = bm.load_prep_cases(prep, "2h")
    return prep, loaded, an.eligible_case_ids(loaded)


def ids(loaded, n=3):
    return [c.case_id for c in loaded.intent[:n]]


def test_two_independent_agreeing_annotations_make_a_case_ready_but_nothing_is_approved_automatically(ctx):
    prep, loaded, valid = ctx
    cid = ids(loaded)[0]
    an.submit(prep, case_id=cid, task="INTENT", label="query", reviewer="alice", valid_ids=valid)
    assert an.case_states(prep)[("INTENT", cid)].state == "PENDING_SECOND_ANNOTATION"
    an.submit(prep, case_id=cid, task="INTENT", label="QUERY", reviewer="bob", valid_ids=valid)
    assert an.case_states(prep)[("INTENT", cid)].state == "AGREED"
    assert not load_approved_labels(prep).available                      # ready is not approved
    assert not (prep / "annotations" / "approved_labels.jsonl").exists()


def test_the_same_reviewer_twice_is_not_two_independent_annotations(ctx):
    prep, loaded, valid = ctx
    cid = ids(loaded)[0]
    an.submit(prep, case_id=cid, task="INTENT", label="QUERY", reviewer="alice", valid_ids=valid)
    an.submit(prep, case_id=cid, task="INTENT", label="QUERY", reviewer="alice", valid_ids=valid)
    assert an.case_states(prep)[("INTENT", cid)].state == "PENDING_SECOND_ANNOTATION"


def test_conflicts_need_a_third_distinct_adjudicator(ctx):
    prep, loaded, valid = ctx
    cid = ids(loaded)[0]
    an.submit(prep, case_id=cid, task="INTENT", label="QUERY", reviewer="alice", valid_ids=valid)
    an.submit(prep, case_id=cid, task="INTENT", label="GREETING", reviewer="bob", valid_ids=valid)
    assert an.case_states(prep)[("INTENT", cid)].state == "CONFLICT"
    with pytest.raises(an.AnnotationError, match="distinct"):
        an.adjudicate(prep, case_id=cid, task="INTENT", label="QUERY", adjudicator="alice")
    with pytest.raises(an.AnnotationError, match="not ready|AGREED|ADJUDICATED"):
        an.approve(prep, approver="carol", case_ids=[("INTENT", cid)])
    an.adjudicate(prep, case_id=cid, task="INTENT", label="OUT_OF_SCOPE", adjudicator="dave")
    rows = an.approve(prep, approver="carol", case_ids=[("INTENT", cid)])
    assert rows[0]["expected"] == "OUT_OF_SCOPE" and rows[0]["basis"] == "ADJUDICATION" and rows[0]["adjudicator"] == "dave"
    assert load_approved_labels(prep).intent[cid].value == "OUT_OF_SCOPE"


def test_only_valid_labels_eligible_cases_and_named_people_are_accepted(ctx):
    prep, loaded, valid = ctx
    cid = ids(loaded)[0]
    with pytest.raises(an.AnnotationError, match="not one of"):
        an.submit(prep, case_id=cid, task="INTENT", label="MAYBE", reviewer="a", valid_ids=valid)
    with pytest.raises(an.AnnotationError, match="eligible"):
        an.submit(prep, case_id="NOPE", task="INTENT", label="QUERY", reviewer="a", valid_ids=valid)
    with pytest.raises(an.AnnotationError, match="reviewer"):
        an.submit(prep, case_id=cid, task="INTENT", label="QUERY", reviewer=" ", valid_ids=valid)
    with pytest.raises(an.AnnotationError, match="approver"):
        an.approve(prep, approver="", all_ready=True)
    lead_id = loaded.lead[0].case_id
    an.submit(prep, case_id=lead_id, task="LEAD", label="yes", reviewer="a", valid_ids=valid)
    assert an.case_states(prep)[("LEAD", lead_id)].annotations["a"] is True
    with pytest.raises(an.AnnotationError, match="true or false"):
        an.submit(prep, case_id=lead_id, task="LEAD", label="perhaps", reviewer="a", valid_ids=valid)


def test_review_queue_has_redacted_text_but_no_labels_or_predictions_and_is_never_overwritten(ctx):
    prep, loaded, _ = ctx
    path = an.write_queue(prep, loaded, Task.INTENT, limit=3)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 3 and set(rows[0]) == {"case_id", "task", "history_status", "text_for_review"}
    with pytest.raises(an.AnnotationError, match="already exists"):
        an.write_queue(prep, loaded, Task.INTENT)


def test_approval_revocation_and_freeze_keep_a_complete_history(ctx):
    prep, loaded, valid = ctx
    a, b = ids(loaded)[:2]
    for cid in (a, b):
        for r in ("alice", "bob"):
            an.submit(prep, case_id=cid, task="INTENT", label="QUERY", reviewer=r, valid_ids=valid)
    assert len(an.approve(prep, approver="carol", all_ready=True)) == 2
    assert an.approve(prep, approver="carol", all_ready=True) == []       # nothing new: no duplicate approvals
    f1 = an.freeze(prep, by="carol", dataset_version="prep-t")
    assert an.frozen_matches_current(prep) is True
    an.revoke(prep, task="INTENT", case_id=a, by="carol", reason="reviewer found an error")
    assert load_approved_labels(prep).intent.keys() == {b} and an.frozen_matches_current(prep) is False
    f2 = an.freeze(prep, by="carol", dataset_version="prep-t")
    assert f1 != f2 and f1.exists() and f2.exists()
    assert len(an.read_events(prep)) == 7   # 4 annotations, 2 approvals, 1 revocation


def test_freeze_needs_approved_labels(ctx):
    prep, _, _ = ctx
    with pytest.raises(an.AnnotationError, match="no approved labels"):
        an.freeze(prep, by="x", dataset_version="prep-t")


def test_the_preparation_command_cannot_touch_a_folder_with_annotations(ctx, tmp_path):
    prep, loaded, valid = ctx
    an.submit(prep, case_id=ids(loaded)[0], task="INTENT", label="QUERY", reviewer="alice", valid_ids=valid)
    before = (prep / "annotations" / "events.jsonl").read_text()
    src = tmp_path / "s2.csv"
    write_csv(src)
    with pytest.raises(dp.PrepOverwriteError):
        dp.prepare(src, prep)
    assert (prep / "annotations" / "events.jsonl").read_text() == before


def test_cli_end_to_end_bulk_approval_needs_the_typed_phrase(ctx, monkeypatch):
    prep, loaded, _ = ctx
    cid = ids(loaded)[0]
    runner = CliRunner()
    base = ["--prep-dir", str(prep), "--session-policy", "2h"]
    for who in ("alice", "bob"):
        r = runner.invoke(app, ["annotate", "submit", *base, "--task", "INTENT", "--reviewer", who, "--case", cid, "--label", "QUERY"])
        assert r.exit_code == 0, r.output
    st = runner.invoke(app, ["annotate", "status", *base])
    assert "AGREED" in st.output and "never frozen" in st.output
    refused = runner.invoke(app, ["annotate", "approve", "--prep-dir", str(prep), "--approver", "carol", "--all-ready"])
    assert refused.exit_code == 1 and not load_approved_labels(prep).available      # non-interactive: refused
    one = runner.invoke(app, ["annotate", "approve", "--prep-dir", str(prep), "--approver", "carol", "--task", "INTENT", "--case", cid])
    assert one.exit_code == 0 and load_approved_labels(prep).intent[cid].value == "QUERY"
    bad = runner.invoke(app, ["annotate", "submit", *base, "--task", "INTENT", "--reviewer", "x", "--case", "NOPE", "--label", "QUERY"])
    assert bad.exit_code == 1
