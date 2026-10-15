"""Local, deterministic preparation of the historical conversation export for human review. Nothing leaves the machine.

The source CSV is read only (its SHA-256 is recorded and re-checked). `user4` (phone-number prefix digits) is never read into
any structure. `user_hash` and `traceId` are replaced by salted, truncated hashes; text is redacted before any derived file is
written. Everything is labelled PRODUCTION_PARITY_NOT_ESTABLISHED, session policy is provisional, expected labels stay null,
and the historical assistant answer is context only (never a prediction or a label).
"""

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from msu_llm_lab.classification.contracts import (
    PARITY_STATUS, ExclusionReason, HistoryStatus, IntentCase, LeadCase, Role, Turn,
)
from msu_llm_lab.classification.lead_summary import build_lead_case, tier_tally
from msu_llm_lab.classification.splits import assert_no_user4, assign_group_splits, find_group_leaks
from msu_llm_lab.schemas.evaluation import DataClassification, DatasetSplit

PREP_VERSION = "prep-v1"
SALT = "msu-llm-lab/prep-v1"
SEED = "msu-prep-v1-seed"
EXPECTED_COLUMNS = ["traceId", "time", "user4", "user_hash", "conversation_id", "turn", "query", "answer"]
POLICIES = {"30m": 1800, "2h": 7200}
SPLIT_RATIOS = {DatasetSplit.DEV: 0.5, DatasetSplit.TEST: 0.5}  # provisional
CHALLENGE_FRACTION = 0.5                                         # share of rare-class-hint groups moved to the challenge view
SHORT_REPLY_TOKENS = 4
REVIEW_SAMPLE_GROUPS = 12

# ------------------------------------------------------------------ redaction
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_IC = re.compile(r"(?<!\d)\d{6}-?\d{2}-?\d{4}(?!\d)")
_MY_MOBILE = re.compile(r"(?<![\w])(?:\+?60|0)\s?1\d(?:[\s\-]?\d){7,8}(?!\d)")
_LONG_DIGITS = re.compile(r"(?<!\d)\+?\d[\d\s\-()]{7,18}\d(?!\d)")
_NAME = re.compile(r"(?i)\b(my name is|nama saya|nama\s*:|name\s*:)\s+([A-Za-z'@.\-]+(?:\s+[A-Za-z'@.\-]+){0,2})")


def redact(text: str) -> Tuple[str, Counter]:
    """Replace direct identifiers with tokens. Names are only caught after explicit introductions (no NER)."""
    counts: Counter = Counter()

    def sub(pattern, token, s, min_digits=0):
        def repl(m):
            if min_digits and sum(c.isdigit() for c in m.group(0)) < min_digits:
                return m.group(0)
            counts[token] += 1
            return token
        return pattern.sub(repl, s)

    out = sub(_EMAIL, "[EMAIL]", text)
    out = sub(_IC, "[ID_NUMBER]", out)
    out = sub(_MY_MOBILE, "[PHONE]", out)
    out = sub(_LONG_DIGITS, "[PHONE]", out, min_digits=9)

    def name_repl(m):
        counts["[NAME]"] += 1
        return f"{m.group(1)} [NAME]"
    out = _NAME.sub(name_repl, out)
    return out, counts


def _h(value: str, n: int = 10) -> str:
    return hashlib.sha256(f"{SALT}|{value}".encode("utf-8")).hexdigest()[:n]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _norm_loose(text: str) -> str:
    return re.sub(r"[^\w\s]", "", _norm(text)).strip()


def _unsupported_script(text: str) -> bool:
    return any(ch.isalpha() and ord(ch) > 0x24F for ch in text)


# ------------------------------------------------------------------ rows
@dataclass
class Row:
    row_id: str
    trace_ref: str
    group_id: str
    ts: datetime
    turn: Optional[int]
    query: str
    answer: Optional[str]
    flags: List[str] = field(default_factory=list)
    redactions: Dict[str, int] = field(default_factory=dict)
    exact_dup_of: Optional[str] = None
    near_dup_of: Optional[str] = None
    disposition: str = "INCLUDED"
    exclusion_reason: Optional[str] = None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> Tuple[List[Row], Dict[str, Any]]:
    """Read the CSV read-only. Column `user4` is never accessed."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected CSV columns: {header}")
        idx = {name: i for i, name in enumerate(header)}
        rows: List[Row] = []
        stats: Counter = Counter()
        for n, rec in enumerate(reader, 1):
            if len(rec) != len(header):
                raise ValueError(f"Row {n} has {len(rec)} fields")
            raw_q, raw_a = rec[idx["query"]], rec[idx["answer"]]
            q, qc = redact(raw_q)
            a, ac = (redact(raw_a) if raw_a.strip() else (None, Counter()))
            red = Counter(qc) + Counter(ac)
            for tok, c in qc.items():
                stats[f"query{tok}"] += c
            for tok, c in ac.items():
                stats[f"answer{tok}"] += c
            ts = datetime.fromisoformat(rec[idx["time"]].replace("Z", "+00:00"))
            turn_raw = rec[idx["turn"]].strip()
            row = Row(
                row_id=f"R{n:04d}", trace_ref=_h(rec[idx["traceId"]], 12), group_id="G" + _h(rec[idx["user_hash"]], 10), ts=ts,
                turn=int(turn_raw) if turn_raw.isdigit() else None, query=q, answer=a, redactions=dict(red),
            )
            if red:
                row.flags.append("REDACTED")
            if not q.strip():
                row.flags.append("EMPTY_QUERY")
            if a is None:
                row.flags.append("MISSING_ANSWER")
            if "\n" in raw_q:
                row.flags.append("MULTILINE_QUERY")
            if _unsupported_script(raw_q):
                row.flags.append("UNSUPPORTED_SCRIPT")
            if len(raw_q) > 600:
                row.flags.append("VERY_LONG_QUERY")
            if rec[idx["conversation_id"]].strip():
                row.flags.append("HAS_CONVERSATION_ID")
            rows.append(row)
    stats["rows"] = len(rows)
    return rows, dict(stats)


def mark_duplicates(rows: List[Row]) -> Dict[str, int]:
    """Flag (never delete) exact duplicates and quick near-duplicates (same normalised query re-sent within 5 minutes)."""
    first_exact: Dict[Tuple[str, str, str], str] = {}
    last_by_query: Dict[Tuple[str, str], Row] = {}
    counts: Counter = Counter()
    for r in sorted(rows, key=lambda r: (r.ts, r.row_id)):
        key = (r.group_id, _norm(r.query), _norm(r.answer or ""))
        if key in first_exact:
            r.exact_dup_of = first_exact[key]
            r.flags.append("EXACT_DUPLICATE")
            counts["exact_duplicate_rows"] += 1
        else:
            first_exact[key] = r.row_id
            qkey = (r.group_id, _norm_loose(r.query))
            prev = last_by_query.get(qkey)
            if prev is not None and qkey[1] and (r.ts - prev.ts).total_seconds() <= 300:
                r.near_dup_of = prev.row_id
                r.flags.append("NEAR_DUPLICATE")
                counts["near_duplicate_rows"] += 1
        last_by_query[(r.group_id, _norm_loose(r.query))] = r
    return dict(counts)


def classify_dispositions(rows: List[Row]) -> Dict[str, int]:
    """Every row gets a disposition; nothing is dropped. Only rows with no usable current message are EXCLUDED."""
    counts: Counter = Counter()
    for r in rows:
        if "EMPTY_QUERY" in r.flags:
            r.disposition, r.exclusion_reason = "EXCLUDED", "EMPTY_QUERY"
        elif set(r.flags) & {"MISSING_ANSWER", "EXACT_DUPLICATE", "NEAR_DUPLICATE", "REDACTED", "UNSUPPORTED_SCRIPT", "VERY_LONG_QUERY", "MULTILINE_QUERY"}:
            r.disposition = "INCLUDED_FLAGGED"
        counts[r.disposition] += 1
    return dict(counts)


# ------------------------------------------------------------------ sessions
@dataclass
class SessionPolicy:
    name: str
    gap_seconds: int
    session_of: Dict[str, str] = field(default_factory=dict)
    sessions: Dict[str, List[str]] = field(default_factory=dict)       # session_id -> ordered row ids
    boundary: Dict[str, str] = field(default_factory=dict)             # row id -> category
    turn_evidence: Dict[str, str] = field(default_factory=dict)
    gap_before: Dict[str, Optional[float]] = field(default_factory=dict)


def build_sessions(rows: List[Row], name: str, gap_seconds: int) -> SessionPolicy:
    """Group by pseudonymous user group, order by timestamp (then file order), split on gaps over the threshold.
    `turn` is supporting evidence only and never creates or removes a boundary."""
    pol = SessionPolicy(name, gap_seconds)
    by_group: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_group[r.group_id].append(r)
    for g, rs in by_group.items():
        rs.sort(key=lambda r: (r.ts, r.row_id))
        n, prev = 0, None
        for r in rs:
            gap = None if prev is None else (r.ts - prev.ts).total_seconds()
            if prev is None:
                n += 1
                cat = "FIRST_ROW_OF_GROUP"
            elif gap > gap_seconds:
                n += 1
                cat = "NEW_SESSION_AFTER_GAP"
            else:
                cat = "CONTINUATION"
            sid = f"S{name}-{g}-{n}"
            pol.session_of[r.row_id] = sid
            pol.sessions.setdefault(sid, []).append(r.row_id)
            pol.boundary[r.row_id] = cat
            pol.gap_before[r.row_id] = gap
            if cat == "CONTINUATION":
                ev = ("AGREES" if (r.turn is not None and prev.turn is not None and r.turn == prev.turn + 1)
                      else "TURN_RESET_INSIDE_SESSION" if r.turn == 1 else "TURN_IRREGULAR")
            else:
                ev = "AGREES" if r.turn == 1 else "SESSION_STARTS_WITHOUT_TURN_1"
            pol.turn_evidence[r.row_id] = ev
            prev = r
    return pol


def compare_policies(a: SessionPolicy, b: SessionPolicy, rows: List[Row]) -> Dict[str, Any]:
    gaps = [a.gap_before[r.row_id] for r in rows if a.gap_before[r.row_id] is not None]
    between = sum(1 for g in gaps if a.gap_seconds < g <= b.gap_seconds)
    merged = Counter()
    for sid_b, rids in b.sessions.items():
        merged[len({a.session_of[r] for r in rids})] += 1
    return {
        "sessions": {a.name: len(a.sessions), b.name: len(b.sessions)},
        "boundary_categories": {p.name: dict(Counter(p.boundary.values())) for p in (a, b)},
        "turn_evidence": {p.name: dict(Counter(p.turn_evidence.values())) for p in (a, b)},
        "rows_whose_gap_falls_between_thresholds": between,
        "short_sessions_merged_into_one_longer_session": {str(k): v for k, v in sorted(merged.items())},
        "single_row_sessions": {p.name: sum(1 for v in p.sessions.values() if len(v) == 1) for p in (a, b)},
    }


def review_sample(rows: List[Row], a: SessionPolicy, b: SessionPolicy, n_groups: int = REVIEW_SAMPLE_GROUPS) -> List[Dict[str, Any]]:
    """Small, redacted, seeded sample of user groups for manual boundary review. Written to a local file only."""
    by_group: Dict[str, List[Row]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r.ts, r.row_id)):
        by_group[r.group_id].append(r)
    cats: Dict[str, List[str]] = {"gap_between_thresholds": [], "turn_reset_inside_session": [], "gap_over_24h": []}
    for g, rs in by_group.items():
        for r in rs:
            gap = a.gap_before[r.row_id]
            if gap is not None and a.gap_seconds < gap <= b.gap_seconds:
                cats["gap_between_thresholds"].append(g)
            if b.turn_evidence[r.row_id] == "TURN_RESET_INSIDE_SESSION":
                cats["turn_reset_inside_session"].append(g)
            if gap is not None and gap > 86400:
                cats["gap_over_24h"].append(g)
    chosen: List[Tuple[str, str]] = []
    per = max(1, n_groups // 3)
    for cat, gs in cats.items():
        for g in sorted(set(gs), key=lambda g: _h(SEED + g))[:per]:
            if g not in {c[0] for c in chosen}:
                chosen.append((g, cat))
    out = []
    for g, cat in chosen:
        out.append({"group_id": g, "selected_because": cat, "rows": [{
            "row_id": r.row_id, "time": r.ts.isoformat(), "gap_s": a.gap_before[r.row_id], "turn": r.turn,
            "session_30m": a.session_of[r.row_id][-6:], "session_2h": b.session_of[r.row_id][-6:], "boundary_30m": a.boundary[r.row_id],
            "boundary_2h": b.boundary[r.row_id], "turn_evidence_2h": b.turn_evidence[r.row_id], "flags": r.flags,
            "query_redacted_excerpt": r.query[:90], "answer_redacted_excerpt": (r.answer or "")[:90] if r.answer else None,
        } for r in by_group[g]]})
    return out


# ------------------------------------------------------------------ selection hints (NOT labels)
_HINTS = {
    "HANDOFF_PHRASE": re.compile(r"\b(?:speak|talk|call|contact|connect|hubungi)\b.*\b(?:counsel+or|advis[eo]r|someone|agent|human|staff|person|kaunselor)\b|\bcall\s*me\b|\bcall\s*back\b", re.I),
    "UPLOAD_STATUS_PHRASE": re.compile(r"(?=.*\b(?:uploaded|submitted|sent|attached|hantar|muat\s+naik)\b)(?=.*\b(?:check|received|receive|status|confirm|sampai|terima)\b)", re.I | re.S),
    "OUT_OF_SCOPE_PHRASE": re.compile(r"\b(?:ignore\s+(?:all|your|previous)|system\s+prompt|jailbreak|joke|pretend|you\s+are\s+now)\b", re.I),
}
_OFFER = re.compile(r"\?\s*$|\bwould you like\b|\bdo you want\b|\bshall i\b|\bwant me to\b|\blet me know\b", re.I)
_O1_TOKENS = {"no", "nope", "nah", "thanks", "thank you", "thx", "tq", "terima kasih"}


def selection_hints(text: str) -> List[str]:
    return [name for name, rx in _HINTS.items() if rx.search(text or "")]


def _is_o1(current: str, prev_answer: Optional[str]) -> bool:
    """Unresolved Intent O1: a bare 'no'/'thanks' replying to an offer or question from the assistant."""
    return bool(prev_answer) and _norm_loose(current) in _O1_TOKENS and bool(_OFFER.search(prev_answer or ""))


# ------------------------------------------------------------------ case construction
def _turns_for(rows_in_session: List[Row]) -> Tuple[List[Turn], List[str]]:
    turns, missing = [], []
    for r in rows_in_session:
        turns.append(Turn(turn_index=len(turns), role=Role.USER, text=r.query, source_ref=f"{r.row_id}:Q"))
        if r.answer is not None:
            turns.append(Turn(turn_index=len(turns), role=Role.ASSISTANT, text=r.answer, source_ref=f"{r.row_id}:A"))
        else:
            missing.append(r.row_id)
    return turns, missing


def build_intent_candidates(rows: List[Row], pol: SessionPolicy, group_view: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    by_id = {r.row_id: r for r in rows}
    out = []
    for sid, rids in pol.sessions.items():
        srows = [by_id[x] for x in rids]
        for i, cur in enumerate(srows):
            prior = srows[:i]
            hist_turns, missing = _turns_for(prior)
            last_a = next((t for t in reversed(hist_turns) if t.role == Role.ASSISTANT), None)
            immediate_missing = bool(prior) and prior[-1].answer is None
            status = (HistoryStatus.NONE_AVAILABLE if not hist_turns else HistoryStatus.INCOMPLETE if missing else HistoryStatus.COMPLETE)
            if not hist_turns and prior:  # earlier rows exist but contributed no turns (cannot happen: every row has a query)
                status = HistoryStatus.UNKNOWN
            flags: List[str] = []
            if immediate_missing:
                flags.append("PRECEDING_ASSISTANT_MESSAGE_MISSING")
            if missing:
                flags.append("EARLIER_ASSISTANT_ANSWER_MISSING")
            flags += [f for f in cur.flags if f in ("MULTILINE_QUERY", "UNSUPPORTED_SCRIPT", "EXACT_DUPLICATE", "NEAR_DUPLICATE", "VERY_LONG_QUERY", "REDACTED")]
            excl: Optional[ExclusionReason] = None
            rule_ref = None
            short = len(_norm_loose(cur.query).split()) <= SHORT_REPLY_TOKENS
            if "EMPTY_QUERY" in cur.flags:
                excl, rule_ref = ExclusionReason.OTHER, "EMPTY_QUERY"
            elif _is_o1(cur.query, prior[-1].answer if prior else None):
                excl, rule_ref = ExclusionReason.UNRESOLVED_BUSINESS_RULE, "INTENT_O1"
            elif immediate_missing and short:
                excl, rule_ref = ExclusionReason.INSUFFICIENT_CONTEXT, "PRECEDING_ASSISTANT_MESSAGE_MISSING_FOR_SHORT_REPLY"
            case = IntentCase(
                case_id=f"I-{pol.name}-{cur.row_id}", classification=DataClassification.CONFIDENTIAL, group_id=cur.group_id,
                history=hist_turns, history_status=status,
                current_message=Turn(turn_index=len(hist_turns), role=Role.USER, text=cur.query, source_ref=f"{cur.row_id}:Q"),
                previous_assistant_message=last_a, exclusion_reason=excl,
            )
            gv = group_view[cur.group_id]
            out.append({"case": case.model_dump(mode="json"), "candidate": {
                "candidate_status": "EXCLUDED" if excl else ("NEEDS_CONTEXT" if flags and "PRECEDING_ASSISTANT_MESSAGE_MISSING" in flags else "PENDING_REVIEW"),
                "exclusion_rule_ref": rule_ref, "policy": pol.name, "session_id": sid, "row_id": cur.row_id, "group_id": cur.group_id,
                "source_row_ids": [r.row_id for r in prior] + [cur.row_id], "quality_flags": flags,
                "suggested_intent": None, "expected_intent": None, "review_status": "NOT_REVIEWED",
                "selection_hints": selection_hints(cur.query), "view": gv["view"], "split": gv["split"],
                "parity_status": PARITY_STATUS, "prep_version": PREP_VERSION, "assistant_text_provenance": "AS_EXPORTED_NOT_A_PREDICTION",
            }})
    return out


L_RULE_MAP = {"SEMANTIC_OBJECTION_REVIEW": "L4", "PROGRAMME_CONTEXT_UNRESOLVED": "L6_L7", "CAPABILITY_QUESTION_NOT_ACTION": "L8"}


def build_lead_candidates(rows: List[Row], pol: SessionPolicy, group_view: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """One candidate per session, summarising the actions in that session's history (a provisional unit)."""
    by_id = {r.row_id: r for r in rows}
    session_index: Dict[str, int] = {}
    seen: Counter = Counter()
    for sid in sorted(pol.sessions, key=lambda s: (pol.sessions[s][0])):
        g = by_id[pol.sessions[sid][0]].group_id
        seen[g] += 1
        session_index[sid] = seen[g]
    out = []
    for sid, rids in pol.sessions.items():
        srows = [by_id[x] for x in rids]
        turns, missing = _turns_for(srows)
        status = HistoryStatus.INCOMPLETE if missing else HistoryStatus.COMPLETE
        g = srows[0].group_id
        lc = build_lead_case(f"L-{pol.name}-{sid}", turns, DataClassification.CONFIDENTIAL, status, g)
        tally = tier_tally(lc.extracted_actions)
        l_rules = sorted({L_RULE_MAP[a.reason_code] for a in lc.ambiguities if a.reason_code in L_RULE_MAP})
        later_session = session_index[sid] > 1
        possible_return = later_session and tally["tier_a_actions"] == 0 and tally["tier_b_distinct_types"] == 1
        if possible_return:
            l_rules.append("L5")
        excl = None
        if tally["tier_a_actions"] == 0 and l_rules:   # an unresolved rule could flip a 'no signal / one Tier B' case
            excl = ExclusionReason.UNRESOLVED_BUSINESS_RULE
        lc = lc.model_copy(update={"exclusion_reason": excl})
        flags = []
        if missing:
            flags.append("ASSISTANT_ANSWER_MISSING_IN_HISTORY")
        out.append({"case": lc.model_dump(mode="json"), "candidate": {
            "candidate_status": "EXCLUDED" if excl else "PENDING_REVIEW", "unresolved_rules": l_rules, "policy": pol.name, "session_id": sid,
            "group_id": g, "source_row_ids": rids, "quality_flags": flags, "builder_tally_not_a_label": tally,
            "ambiguity_count": len(lc.ambiguities), "later_session_for_group": later_session,
            "expected_need_cta": None, "suggested_need_cta": None, "review_status": "NOT_REVIEWED",
            "view": group_view[g]["view"], "split": group_view[g]["split"], "parity_status": PARITY_STATUS, "prep_version": PREP_VERSION,
            "summary_provenance": "OFFLINE_RULE_BUILDER_NOT_PRODUCTION",
        }})
    return out


# ------------------------------------------------------------------ views, splits, leakage
def assign_views_and_splits(rows: List[Row]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """Policy-independent, group-level assignment: rare-class-hint groups are partly moved to a separate challenge view."""
    by_group: Dict[str, List[Row]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r.ts, r.row_id)):
        by_group[r.group_id].append(r)
    hint_groups = []
    for g, rs in by_group.items():
        text_hit = any(selection_hints(r.query) for r in rs)
        lead_hit = False
        turns, _ = _turns_for(rs)
        if turns:
            lc = build_lead_case(f"H-{g}", turns, DataClassification.CONFIDENTIAL, HistoryStatus.UNKNOWN, g)
            lead_hit = tier_tally(lc.extracted_actions)["tier_a_actions"] > 0
        if text_hit or lead_hit:
            hint_groups.append(g)
    ordered = sorted(hint_groups, key=lambda g: _h(SEED + g))
    n_ch = math.ceil(len(ordered) * CHALLENGE_FRACTION)
    challenge = set(ordered[:n_ch])
    view = {g: ("CHALLENGE" if g in challenge else "NATURAL") for g in by_group}
    splits: Dict[str, DatasetSplit] = {}
    for v in ("NATURAL", "CHALLENGE"):
        gs = sorted(g for g in by_group if view[g] == v)
        if gs:
            splits.update(assign_group_splits(gs, SPLIT_RATIOS, f"{SEED}|{v}"))
    mapping = {g: {"view": view[g], "split": splits[g].value} for g in by_group}
    info = {"groups": len(by_group), "hint_groups": len(hint_groups), "challenge_groups": len(challenge), "natural_groups": len(by_group) - len(challenge),
            "challenge_fraction": CHALLENGE_FRACTION, "split_ratios": {k.value: v for k, v in SPLIT_RATIOS.items()}, "seed": SEED}
    return mapping, info


def leakage_checks(candidate_sets: Dict[str, List[Dict[str, Any]]], group_view: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for name, cands in candidate_sets.items():
        case_groups = {c["candidate"]["row_id" if "row_id" in c["candidate"] else "session_id"]: c["candidate"]["group_id"] for c in cands}
        case_splits = {k: DatasetSplit(group_view[g]["split"]) for k, g in case_groups.items()}
        leaks = find_group_leaks(case_groups, case_splits)
        view_by_group: Dict[str, set] = defaultdict(set)
        for c in cands:
            view_by_group[c["candidate"]["group_id"]].add((c["candidate"]["view"], c["candidate"]["split"]))
        multi = [g for g, v in view_by_group.items() if len(v) > 1]
        by_split_msgs: Dict[str, Counter] = {"DEV": Counter(), "TEST": Counter()}
        if name.startswith("intent"):
            for c in cands:
                by_split_msgs[c["candidate"]["split"]][_norm_loose(c["case"]["current_message"]["text"])] += 1
        shared = set(by_split_msgs["DEV"]) & set(by_split_msgs["TEST"])
        res[name] = {"groups_in_multiple_splits": len(leaks), "groups_in_multiple_views_or_splits": len(multi),
                     "distinct_current_messages_shared_between_dev_and_test": len(shared) if name.startswith("intent") else None,
                     "cases": len(cands)}
    return res


# ------------------------------------------------------------------ orchestration
def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def prepare(source: Path, out_dir: Path) -> Dict[str, Any]:
    """Run the whole local pipeline and write artifacts. Returns the manifest (aggregates only)."""
    source, out_dir = Path(source), Path(out_dir)
    sha_before = file_sha256(source)
    rows, redaction_stats = load_rows(source)
    dup_counts = mark_duplicates(rows)
    disp = classify_dispositions(rows)
    pol_a = build_sessions(rows, "30m", POLICIES["30m"])
    pol_b = build_sessions(rows, "2h", POLICIES["2h"])
    comparison = compare_policies(pol_a, pol_b, rows)
    group_view, view_info = assign_views_and_splits(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned = [{
        "row_id": r.row_id, "trace_ref": r.trace_ref, "group_id": r.group_id, "time": r.ts.isoformat(), "turn_source": r.turn,
        "query_redacted": r.query, "answer_redacted": r.answer, "answer_status": "PRESENT" if r.answer is not None else "MISSING",
        "flags": r.flags, "redactions": r.redactions, "disposition": r.disposition, "exclusion_reason": r.exclusion_reason,
        "exact_duplicate_of": r.exact_dup_of, "near_duplicate_of": r.near_dup_of,
        "session_30m": pol_a.session_of[r.row_id], "session_2h": pol_b.session_of[r.row_id],
        "boundary_30m": pol_a.boundary[r.row_id], "boundary_2h": pol_b.boundary[r.row_id], "parity_status": PARITY_STATUS,
    } for r in rows]
    assert_no_user4(cleaned[0].keys())
    _write_jsonl(out_dir / "cleaned_rows.jsonl", cleaned)

    for pol in (pol_a, pol_b):
        _write_jsonl(out_dir / f"sessions_{pol.name}.jsonl", ({"session_id": sid, "n_rows": len(r), "row_ids": r, "policy": pol.name,
                                                               "gap_seconds": pol.gap_seconds, "provisional": True, "parity_status": PARITY_STATUS}
                                                              for sid, r in pol.sessions.items()))
    _write_jsonl(out_dir / "session_review_sample.jsonl", review_sample(rows, pol_a, pol_b))
    _write_jsonl(out_dir / "split_assignments.jsonl", ({"group_id": g, **v} for g, v in sorted(group_view.items())))

    cand: Dict[str, List[Dict[str, Any]]] = {}
    for pol in (pol_a, pol_b):
        cand[f"intent_{pol.name}"] = build_intent_candidates(rows, pol, group_view)
        cand[f"lead_{pol.name}"] = build_lead_candidates(rows, pol, group_view)
        for kind in ("intent", "lead"):
            _write_jsonl(out_dir / f"{kind}_candidates_{pol.name}.jsonl", cand[f"{kind}_{pol.name}"])

    def summarize(cs: List[Dict[str, Any]]) -> Dict[str, Any]:
        c = Counter(); v = Counter(); sp = Counter(); rl = Counter()
        for x in cs:
            c[x["candidate"]["candidate_status"]] += 1
            v[(x["candidate"]["view"], x["candidate"]["split"])] += 1
            if x["candidate"].get("exclusion_rule_ref"):
                rl[x["candidate"]["exclusion_rule_ref"]] += 1
            for u in x["candidate"].get("unresolved_rules", []):
                rl[u] += 1
        return {"cases": len(cs), "by_status": dict(c), "by_view_split": {f"{a}/{b}": n for (a, b), n in sorted(v.items())}, "unresolved_rule_refs": dict(rl)}

    sha_after = file_sha256(source)
    all_flags = Counter(f for r in rows for f in r.flags)
    manifest = {
        "prep_version": PREP_VERSION, "parity_status": PARITY_STATUS, "provisional_session_policy": True,
        "source": {"file": source.name, "sha256": sha_before, "unchanged_after_run": sha_before == sha_after, "rows": len(rows),
                   "columns_read": [c for c in EXPECTED_COLUMNS if c != "user4"], "excluded_column": "user4 (never read into any structure)"},
        "cleaning": {"rows_accounted": len(cleaned), "dispositions": disp, "flag_counts": dict(all_flags), "duplicates": dup_counts,
                     "redaction_token_counts": {k: v for k, v in redaction_stats.items() if k != "rows"},
                     "rows_with_any_redaction": sum(1 for r in rows if r.redactions), "answers_missing": sum("MISSING_ANSWER" in r.flags for r in rows),
                     "limitations": ["Names are redacted only after explicit introductions; there is no named-entity recognition.",
                                     "Programme names, free-text addresses and other quasi-identifiers are not redacted."]},
        "session_comparison": comparison, "views_and_splits": view_info,
        "candidates": {k: summarize(v) for k, v in cand.items()},
        "leakage_checks": leakage_checks(cand, group_view),
        "unit_decisions": ["Intent candidates: one per student row (current message = that row's query; history = earlier rows in the same session).",
                           "Lead candidates: one per session (summary of that session's history); a per-turn unit is not implemented.",
                           "Both session policies are produced; neither is chosen."],
        "artifacts": sorted(p.name for p in out_dir.iterdir()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "README.txt").write_text(
        f"{PARITY_STATUS}\nLocal artifacts derived from the historical export ({PREP_VERSION}). Contains redacted student text. Never commit or upload.\n"
        "Session policies 30m and 2h are provisional. Expected labels are null until human review. Historical assistant answers are context, not predictions.\n")
    return manifest
