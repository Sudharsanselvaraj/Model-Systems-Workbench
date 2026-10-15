"""Hardening tests for local dataset preparation: narrowed O1, overwrite protection, versioning, pseudonymisation and visible limitations."""

import hashlib
import json
import re

import pytest
from typer.testing import CliRunner

from msu_llm_lab.classification import dataprep as dp
from test_dataprep import lines, prepared, write_csv  # noqa: F401  (prepared is a fixture)


# ------------------------------------------------------------------ O1 is narrowed to real non-counsellor offers
@pytest.mark.parametrize("assistant,expected", [
    ("Would you like fee details for this programme?", True),
    ("Do you want the entry requirements as well?", True),
    ("Shall I list the scholarships?", True),
    ("Which campus do you mean, Kuala Lumpur or Shah Alam?", False),     # ordinary clarifying question
    ("What is your current qualification?", False),                      # ordinary clarifying question
    ("Anything else? Let me know.", False),                              # generic closing, not an offer
    ("Would you like me to connect you to a counsellor?", False),        # counsellor-connect offers follow Step 2, not O1
    ("The fee is RM 20,000.", False),
])
@pytest.mark.parametrize("reply", ["no", "thanks"])
def test_o1_applies_only_to_a_bare_reply_after_a_non_counsellor_offer(reply, assistant, expected):
    assert dp._is_o1(reply, assistant) is expected


# ------------------------------------------------------------------ overwrite protection and versioning
def test_rerun_into_an_existing_directory_is_refused_and_leaves_everything_untouched(prepared):
    src, out, _, _ = prepared
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()}
    with pytest.raises(dp.PrepOverwriteError, match="non-empty"):
        dp.prepare(src, out)
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()} == before


def test_a_directory_holding_human_annotations_is_protected_with_a_specific_message(tmp_path):
    src = tmp_path / "syn.csv"
    write_csv(src)
    out = tmp_path / "out"
    (out / "annotations").mkdir(parents=True)
    note = out / "annotations" / "labels_v1.jsonl"
    note.write_text('{"case_id": "x", "expected_intent": "QUERY"}\n')
    with pytest.raises(dp.PrepOverwriteError, match="annotations"):
        dp.prepare(src, out)
    assert note.read_text().startswith('{"case_id"')


def test_next_free_version_dir_and_a_new_version_run_leave_the_original_alone(prepared, tmp_path):
    src, out, _, _ = prepared
    snapshot = (out / "manifest.json").read_text()
    new = dp.next_free_version_dir(out)
    assert new.name == out.name + "-r2" and not new.exists()
    dp.prepare(src, new)
    assert dp.next_free_version_dir(out).name == out.name + "-r3"
    assert (out / "manifest.json").read_text() == snapshot and (new / "manifest.json").exists()
    empty = tmp_path / "fresh"
    assert dp.next_free_version_dir(empty) == empty  # an unused directory is used as is


def test_manifest_records_code_and_config_version(prepared):
    _, _, m, _ = prepared
    run = m["run"]
    assert re.fullmatch(r"[0-9a-f]{16}", run["code_fingerprint"]) and run["created_utc"] and "git_commit" in run
    assert run["config"]["seed"] == dp.SEED and run["config"]["session_policies_seconds"] == {"30m": 1800, "2h": 7200}
    assert run["config"]["split_ratios"] == {"DEV": 0.5, "TEST": 0.5} and run["config"]["challenge_fraction"] == 0.5


# ------------------------------------------------------------------ pseudonymisation is visible, and a protected salt is supported
def test_public_salt_is_flagged_in_the_manifest_and_readme(prepared):
    _, out, m, _ = prepared
    p = m["pseudonymization"]
    assert p["salt_id"] == "public-legacy-v1" and p["protected_secret_used"] is False and "PUBLIC legacy salt" in p["limitation"]
    assert "stay stable" in p["migration"] and "never by group id" in p["migration"]
    assert "PUBLIC legacy constant" in (out / "README.txt").read_text()


def test_a_private_salt_changes_group_ids_but_not_row_ids_and_is_never_written(tmp_path, monkeypatch):
    src = tmp_path / "syn.csv"
    write_csv(src)
    monkeypatch.delenv(dp.SALT_ENV, raising=False)
    dp.prepare(src, tmp_path / "pub")
    monkeypatch.setenv(dp.SALT_ENV, "a-private-test-secret-value")
    m = dp.prepare(src, tmp_path / "prot")
    assert m["pseudonymization"]["protected_secret_used"] is True and m["pseudonymization"]["salt_id"].startswith("secret-")
    pub, prot = lines(tmp_path / "pub" / "cleaned_rows.jsonl"), lines(tmp_path / "prot" / "cleaned_rows.jsonl")
    assert [r["row_id"] for r in pub] == [r["row_id"] for r in prot]
    assert {r["group_id"] for r in pub}.isdisjoint({r["group_id"] for r in prot})
    for f in (tmp_path / "prot").iterdir():
        assert "a-private-test-secret-value" not in f.read_text(), f.name
    assert "PUBLIC legacy constant" not in (tmp_path / "prot" / "README.txt").read_text()


# ------------------------------------------------------------------ limitations are visible in every derived artifact
def test_privacy_and_view_notices_are_in_every_derived_artifact(prepared):
    _, out, m, _ = prepared
    notice = dp.PRIVACY_NOTICE
    assert notice in (out / "README.txt").read_text() and m["privacy_notice"] == notice
    for name in ("cleaned_rows.jsonl", "sessions_30m.jsonl", "sessions_2h.jsonl", "split_assignments.jsonl", "session_review_sample.jsonl"):
        recs = lines(out / name)
        assert recs and all(r["privacy_notice"] == notice for r in recs), name
    for name in ("intent_candidates_30m.jsonl", "intent_candidates_2h.jsonl", "lead_candidates_30m.jsonl", "lead_candidates_2h.jsonl"):
        recs = lines(out / name)
        assert all(r["candidate"]["privacy_notice"] == notice and r["candidate"]["view_status"] == "HEURISTIC_UNVALIDATED" for r in recs), name
    assert all(r["view_status"] == "HEURISTIC_UNVALIDATED" for r in lines(out / "split_assignments.jsonl"))


def test_view_semantics_say_the_challenge_set_is_not_a_validated_rare_class_set(prepared):
    _, out, m, _ = prepared
    assert "not a validated rare-class set" in m["view_semantics"]["CHALLENGE"] and "NOT validated as representative" in m["view_semantics"]["NATURAL"]
    assert "unvalidated" in (out / "README.txt").read_text().lower()


# ------------------------------------------------------------------ CLI: refuses overwrite, versions on request
def test_prepare_command_refuses_to_overwrite_and_creates_a_new_version_on_request(tmp_path):
    from typer.testing import CliRunner
    from msu_llm_lab.cli import app
    src = tmp_path / "syn.csv"
    write_csv(src)
    out = tmp_path / "data" / "prep-v1"
    first = CliRunner(env={"COLUMNS": "200"}).invoke(app, ["prepare-classification-data", "--source", str(src), "--out-dir", str(out), "--allow-public-salt"])
    assert first.exit_code == 0 and "pseudonymization: public-legacy-v1 (protected: False)" in first.output and dp.PRIVACY_NOTICE in first.output
    snap = (out / "cleaned_rows.jsonl").read_text()
    again = CliRunner().invoke(app, ["prepare-classification-data", "--source", str(src), "--out-dir", str(out), "--allow-public-salt"])
    assert again.exit_code == 1 and "Refused" in again.output and (out / "cleaned_rows.jsonl").read_text() == snap
    versioned = CliRunner().invoke(app, ["prepare-classification-data", "--source", str(src), "--out-dir", str(out), "--new-version", "--allow-public-salt"])
    assert versioned.exit_code == 0 and (tmp_path / "data" / "prep-v1-r2" / "manifest.json").exists()


def test_dashboard_label_files_stay_protected_from_the_preparation_command(tmp_path):
    from obs_helpers import label_row, make_prep, write_labels
    prep = make_prep(tmp_path / "classification_prep")
    write_labels(prep, [label_row("C1", "INTENT", "QUERY")])
    src = tmp_path / "syn.csv"
    write_csv(src)
    snapshot = {p: p.read_bytes() for p in prep.rglob("*") if p.is_file()}
    with pytest.raises(dp.PrepOverwriteError):
        dp.prepare(src, prep)
    assert {p: p.read_bytes() for p in prep.rglob("*") if p.is_file()} == snapshot


def test_cli_refuses_the_public_salt_unless_explicitly_allowed(tmp_path, monkeypatch):
    from msu_llm_lab.cli import app
    monkeypatch.delenv(dp.SALT_ENV, raising=False)
    src = tmp_path / "syn.csv"
    write_csv(src)
    res = CliRunner().invoke(app, ["prepare-classification-data", "--source", str(src), "--out-dir", str(tmp_path / "o")])
    assert res.exit_code == 1 and "No private salt" in res.output and not (tmp_path / "o").exists()


def test_hidden_salt_prompt_is_never_echoed_stored_or_printed(tmp_path, monkeypatch):
    from msu_llm_lab.cli import app
    monkeypatch.delenv(dp.SALT_ENV, raising=False)
    secret = "correct-horse-battery-staple-9"
    answers = iter([secret, secret])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    src = tmp_path / "syn.csv"
    write_csv(src)
    out = tmp_path / "prep-v9"
    res = CliRunner().invoke(app, ["prepare-classification-data", "--source", str(src), "--out-dir", str(out), "--prompt-salt"])
    assert res.exit_code == 0, res.output
    assert secret not in res.output and "protected: True" in res.output
    assert all(secret not in p.read_text() for p in out.iterdir())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["prep_version"] == "prep-v9" and manifest["pseudonymization"]["protected_secret_used"] is True
    monkeypatch.delenv(dp.SALT_ENV, raising=False)


def test_mismatched_or_short_hidden_secrets_write_nothing(tmp_path, monkeypatch):
    from msu_llm_lab.cli import app
    monkeypatch.delenv(dp.SALT_ENV, raising=False)
    answers = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    src = tmp_path / "syn.csv"
    write_csv(src)
    res = CliRunner().invoke(app, ["prepare-classification-data", "--source", str(src), "--out-dir", str(tmp_path / "o"), "--prompt-salt"])
    assert res.exit_code == 1 and not (tmp_path / "o").exists()


def test_version_recorded_in_artifacts_is_the_folder_name_and_is_restored(tmp_path):
    src = tmp_path / "syn.csv"
    write_csv(src)
    m = dp.prepare(src, tmp_path / "prep-v7")
    assert m["prep_version"] == "prep-v7" and dp.PREP_VERSION == "prep-v1"
    first = json.loads((tmp_path / "prep-v7" / "intent_candidates_2h.jsonl").read_text().splitlines()[0])
    assert first["candidate"]["prep_version"] == "prep-v7"


def test_extended_redaction_covers_passports_introductions_and_relatives_without_eating_ordinary_words():
    text, c = dp.redact("Hi I am Aisha Rahman, passport A12345678. My father's name is Ahmad Zaki. I am interested in IT. I am Looking for fees")
    assert "Aisha" not in text and "A12345678" not in text and "Ahmad" not in text
    assert "I am interested in IT" in text and "I am Looking for fees" in text and c["[NAME]"] >= 2 and c["[ID_NUMBER]"] == 1
