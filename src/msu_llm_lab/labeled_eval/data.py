"""Load, audit, split and separate the expected-label test suites. Inference inputs never contain labels; the answer key is a separate file."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

INTENT_LABELS = ("GREETING", "OUT_OF_SCOPE", "UPLOAD_STATUS_CHECK", "HUMAN_HANDOFF_REQUEST", "QUERY")
LABEL_FIELDS = ("expected_label", "expected_cta", "expected_intent", "expected_need_cta")
SPLIT_SEED = "msu-labeled-eval-v1"
MIN_RARE_SUPPORT = 10          # below this a per-class estimate is flagged unstable
TURN_RE = re.compile(r"^(User|Assistant):\s?(.*)$")
BOOL = {"true": True, "false": False}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def load_intent(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"case_id", "history", "current_message", "expected_label"}
    if not need <= set(d.columns):
        raise ValueError(f"intent file lacks columns {sorted(need - set(d.columns))}")
    return d


def load_lead(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"id", "action_summary", "user_msg", "expected_cta"}
    if not need <= set(d.columns):
        raise ValueError(f"lead file lacks columns {sorted(need - set(d.columns))}")
    return d.rename(columns={"id": "case_id"})


def parse_history(text: str) -> List[Tuple[str, str]]:
    """Turns start at 'User:'/'Assistant:'; any other line continues the previous turn. Empty text means no history (unknown, not 'none happened')."""
    turns: List[List[str]] = []
    for ln in (text or "").replace("\r\n", "\n").split("\n"):
        m = TURN_RE.match(ln)
        if m:
            turns.append([m.group(1).upper(), m.group(2)])
        elif turns:
            turns[-1][1] += "\n" + ln
        elif ln.strip():
            turns.append(["USER", ln])           # unlabeled leading text: recorded, flagged by the audit
    return [(r, t.strip()) for r, t in turns]


def _issue(rows, dataset, case_id, kind, severity, detail):
    rows.append({"dataset": dataset, "case_id": case_id, "issue": kind, "severity": severity, "detail": detail})


def _near_dups(items: List[Tuple[str, str]], thr: float = 0.92) -> List[Tuple[str, str, float]]:
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a and b and abs(len(a) - len(b)) <= 0.2 * max(len(a), len(b)):
                r = difflib.SequenceMatcher(None, a, b).quick_ratio()
                if r >= thr and difflib.SequenceMatcher(None, a, b).ratio() >= thr:
                    out.append((items[i][0], items[j][0], round(difflib.SequenceMatcher(None, a, b).ratio(), 3)))
    return out


def audit_intent(d: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    iss: List[Dict[str, Any]] = []
    ds = "intent"
    for cid in d.case_id[d.case_id.duplicated(keep=False)]:
        _issue(iss, ds, cid, "DUPLICATE_ID", "ERROR", "case_id repeated")
    for r in d.itertuples():
        if r.expected_label not in INTENT_LABELS:
            _issue(iss, ds, r.case_id, "INVALID_LABEL", "ERROR", f"label {r.expected_label!r} not in the five-class contract")
        if not r.current_message.strip():
            _issue(iss, ds, r.case_id, "EMPTY_CURRENT_MESSAGE", "ERROR", "")
        if not r.history.strip():
            _issue(iss, ds, r.case_id, "NO_HISTORY", "INFO", "history empty: unknown, not evidence that nothing happened")
        else:
            ts = parse_history(r.history)
            if not ts or ts[-1][0] != "ASSISTANT":
                _issue(iss, ds, r.case_id, "HISTORY_DOES_NOT_END_WITH_ASSISTANT", "INFO", "labels that depend on the last assistant message may not apply")
        if len(r.history) + len(r.current_message) > 3000:
            _issue(iss, ds, r.case_id, "LONG_INPUT", "INFO", f"{len(r.history) + len(r.current_message)} characters")
        if any(ord(c) > 0xFFFF or c == "�" for c in r.history + r.current_message):
            _issue(iss, ds, r.case_id, "ENCODING_SUSPECT", "WARN", "replacement or astral characters")
    for a, b, s in _near_dups([(r.case_id, norm(r.history + " || " + r.current_message)) for r in d.itertuples()]):
        _issue(iss, ds, f"{a}~{b}", "NEAR_DUPLICATE_CASE", "WARN", f"similarity {s}")
    rep = d[d.current_message.map(norm).duplicated(keep=False)]
    for cid in rep.case_id:
        _issue(iss, ds, cid, "CURRENT_MESSAGE_REPEATED", "INFO", "same current message appears with different history; kept in one split family")
    sup = d.expected_label.value_counts().to_dict()
    for k in INTENT_LABELS:
        if sup.get(k, 0) < MIN_RARE_SUPPORT:
            _issue(iss, ds, k, "LOW_CLASS_SUPPORT", "WARN", f"{sup.get(k, 0)} cases: per-class metrics are unstable")
    stats = {"rows": len(d), "unique_ids": int(d.case_id.nunique()), "label_counts": sup, "empty_history": int((~d.history.str.strip().astype(bool)).sum()),
             "history_chars_median": float(d.history.str.len().median()), "message_chars_median": float(d.current_message.str.len().median())}
    return iss, stats


def audit_lead(d: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    iss: List[Dict[str, Any]] = []
    ds = "lead"
    for cid in d.case_id[d.case_id.duplicated(keep=False)]:
        _issue(iss, ds, cid, "DUPLICATE_ID", "ERROR", "id repeated")
    reps = set(d.expected_cta)
    if not reps <= {"true", "false", "True", "False", "TRUE", "FALSE"}:
        _issue(iss, ds, "*", "INVALID_BOOLEAN", "ERROR", f"values {sorted(reps)}")
    if len({x.lower() for x in reps}) != len(reps):
        _issue(iss, ds, "*", "INCONSISTENT_BOOLEAN_CASE", "WARN", f"representations {sorted(reps)}; normalised to lower case")
    for r in d.itertuples():
        if not r.action_summary.strip():
            _issue(iss, ds, r.case_id, "EMPTY_ACTION_SUMMARY", "ERROR", "")
        if not r.user_msg.strip():
            _issue(iss, ds, r.case_id, "EMPTY_USER_MSG", "WARN", "")
        if len(r.user_msg.strip()) < 5:
            _issue(iss, ds, r.case_id, "VERY_SHORT_USER_MSG", "INFO", f"{len(r.user_msg.strip())} characters")
    for a, b, s in _near_dups([(r.case_id, norm(r.action_summary + " || " + r.user_msg)) for r in d.itertuples()]):
        _issue(iss, ds, f"{a}~{b}", "NEAR_DUPLICATE_CASE", "WARN", f"similarity {s}")
    dup_sum = d[d.action_summary.map(norm).duplicated(keep=False)]
    for cid in dup_sum.case_id:
        _issue(iss, ds, cid, "SAME_ACTION_SUMMARY_REPEATED", "INFO", "kept in one split family")
    lab = d.expected_cta.str.lower()
    clash = d.assign(l=lab).groupby(d.action_summary.map(norm)).l.nunique()
    for k in clash[clash > 1].index:
        for cid in d[d.action_summary.map(norm) == k].case_id:
            _issue(iss, ds, cid, "SAME_SUMMARY_CONFLICTING_LABELS", "WARN", "identical action summary with different expected_cta: label needs review (or user_msg decides)")
    stats = {"rows": len(d), "unique_ids": int(d.case_id.nunique()), "label_counts": lab.value_counts().to_dict(), "summary_chars_median": float(d.action_summary.str.len().median()),
             "user_msg_chars_median": float(d.user_msg.str.len().median())}
    return iss, stats


def compare_lead_versions(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, Any]:
    m = a.merge(b, on="case_id", suffixes=("_a", "_b"))
    return {"rows_a": len(a), "rows_b": len(b), "shared_ids": len(m), "only_in_a": sorted(set(a.case_id) - set(b.case_id)), "only_in_b": sorted(set(b.case_id) - set(a.case_id)),
            "action_summary_identical": int((m.action_summary_a == m.action_summary_b).sum()), "user_msg_identical": int((m.user_msg_a == m.user_msg_b).sum()),
            "expected_cta_identical": int((m.expected_cta_a.str.lower() == m.expected_cta_b.str.lower()).sum())}


def contamination(test_texts: Dict[str, str], prep_dir: Path) -> Dict[str, Any]:
    """Count exact (normalised, hashed) overlaps between test messages and the local benchmark candidates. Only hashes are compared; no text is printed."""
    seen = set()
    for f in ("intent_candidates_2h.jsonl", "lead_candidates_2h.jsonl"):
        p = prep_dir / f
        if not p.exists():
            return {"checked": False, "reason": f"{p} not found"}
        for ln in open(p):
            r = json.loads(ln).get("case", {})
            cm = (r.get("current_message") or {}).get("text") if isinstance(r.get("current_message"), dict) else None
            if cm:
                seen.add(hashlib.sha256(norm(cm).encode()).hexdigest())
    hit = [k for k, t in test_texts.items() if hashlib.sha256(norm(t).encode()).hexdigest() in seen]
    return {"checked": True, "benchmark_messages_hashed": len(seen), "test_cases_with_exact_overlap": len(hit), "overlapping_case_ids": sorted(hit)}


def _rank(cid: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{cid}".encode()).hexdigest()


def stratified_group_split(d: pd.DataFrame, label_col: str, group_key, dev_fraction: float = 0.5, seed: str = SPLIT_SEED) -> Dict[str, str]:
    """Deterministic dev/test split: families (identical normalised key) stay together; whole families are assigned greedily to hit the per-class dev share."""
    d = d.assign(_fam=[group_key(r) for r in d.itertuples()], _lab=d[label_col].str.lower())
    fams = d.groupby("_fam").agg(ids=("case_id", list), labs=("_lab", list)).reset_index()
    fams["rank"] = fams._fam.map(lambda f: _rank(f, seed))
    fams = fams.sort_values("rank")
    total = d._lab.value_counts().to_dict()
    want_dev = {k: v * dev_fraction for k, v in total.items()}
    got_dev = {k: 0 for k in total}
    got_test = {k: 0 for k in total}
    out: Dict[str, str] = {}
    for r in fams.itertuples():
        dev_def = sum(want_dev[l] - got_dev[l] for l in r.labs)
        test_def = sum((total[l] - want_dev[l]) - got_test[l] for l in r.labs)
        dev = dev_def > test_def or (dev_def == test_def and int(r.rank[:2], 16) % 2 == 0)
        for cid, l in zip(r.ids, r.labs):
            out[cid] = "dev" if dev else "test"
            (got_dev if dev else got_test)[l] += 1
    return out
