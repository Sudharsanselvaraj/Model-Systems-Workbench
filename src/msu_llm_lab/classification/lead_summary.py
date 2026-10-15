"""Deterministic Lead Qualifier action-summary builder (OFFLINE BENCHMARK COMPONENT).

This is NOT a reproduction of the production summary builder. Rule defaults B1-B10 are
provisional business-rule defaults and are listed in `PROVISIONAL_DEFAULTS` /
`UNIMPLEMENTED_RULES`. The builder never calls a model and never touches the network.
Evidence is stored by reference (turn_index + source_ref); raw student text is never stored.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from msu_llm_lab.classification.contracts import (
    SUMMARY_PROVENANCE,
    AmbiguityRecord,
    ExtractedAction,
    HistoryStatus,
    LeadCase,
    Role,
    Turn,
)
from msu_llm_lab.schemas.evaluation import DataClassification

BUILDER_VERSION = "offline-rules-0.1.0"

PROVISIONAL_DEFAULTS: Dict[str, str] = {
    "B1": "Count explicit student commitments/actions only when supported by available evidence.",
    "B2": "Stated-results signal requires specific results, grades, scores or qualification outcomes.",
    "B3": "Count distinct Tier B action types, not repeated wording; no double-counting of the same evidence.",
    "B4": "Hostel/campus/transport questions stay Tier A as written; broad scope is flagged for review.",
    "B5": "Explicit price/recognition objections by deterministic patterns; semantic ambiguity flagged.",
    "B6": "Pathway evidence combined only within the supplied history.",
    "B7": "Tier A only when explicit criteria are met; otherwise keep a supported Tier B.",
    "B8": "English and Malay supported in v1; other scripts flagged for review.",
    "B9": "No return-visit signal without approved session-boundary evidence.",
    "B10": "Sessions are never merged by the builder; only supplied history is used.",
}

# Prompt rules that cannot be implemented unambiguously. Never silently approximated.
UNIMPLEMENTED_RULES: Dict[str, str] = {
    "A_DEADLINE_AFTER_PROGRAMME": "Needs a programme-name resolver against the KB; builder records an ambiguity instead.",
    "A_COMPETITOR_DECISION": "Needs a named-competitor list (business decision).",
    "A_OTHER_INSTITUTION_OFFER_OR_REJECTION": "Needs named-institution handling; only flagged.",
    "A_THINK_IT_OVER_AFTER_PROGRAMME": "Needs programme-context resolution; only flagged.",
    "A_PATHWAY_EXPLORATION": "Cross-turn semantic pairing of qualification and 'what can I do' (B6); only flagged.",
    "B_PROGRAMME_COMPARISON": "Needs two resolved MSU programme names; only flagged when a heuristic pattern matches.",
    "B_CAMPUS_TOUR_SENT": "'Was sent the video' is an assistant action; never inferred from assistant text.",
    "B_RETURN_VISIT": "Emitted only with approved session-boundary evidence (B9).",
    "B_PARENT_ENQUIRY_SEMANTICS": "Parent-on-behalf detection is heuristic; only emitted with child level/results in the same turn.",
}


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip()


@dataclass(frozen=True)
class Rule:
    rule_id: str
    action_code: str
    tier: str
    summary: str
    patterns: Tuple[str, ...]
    negation_sensitive: bool = False
    review_flag: Optional[str] = None  # emits an ambiguity record alongside the action


_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|januari|februari|mac|mei|julai|ogos|disember)"
_EXAM = r"(?:spm|stpm|uec|igcse|a-?levels?|o-?levels?|diploma|cgpa|gpa|stam|matriculation|foundation)"

RULES: Tuple[Rule, ...] = (
    # ---------------- Tier A ----------------
    Rule("A01", "APPLICATION_LINK_REQUEST", "A", "asked for the application link",
         (r"\b(?:application|registration|apply(?:ing)?)\s+link\b", r"\blink\s+(?:to|for)\s+(?:apply|application|registration|register)\b",
          r"\bpautan\s+(?:permohonan|pendaftaran)\b")),
    Rule("A02", "DOCUMENTS_REQUIRED_QUESTION", "A", "asked what documents are needed to apply",
         (r"\bwhat\s+documents?\s+(?:are\s+|do\s+i\s+)?(?:needed|required|need)\b", r"\bdocuments?\s+(?:needed|required)\s+(?:to|for)\s+(?:apply|application)\b",
          r"\bdokumen\s+(?:yang\s+)?(?:diperlukan|perlu|diperlukan)\b")),
    Rule("A03", "DEPOSIT_PAYMENT_QUESTION", "A", "asked how to pay the deposit or registration fee",
         (r"\bhow\s+(?:to|do\s+i|can\s+i)\s+pay\s+(?:the\s+|my\s+)?(?:deposit|registration\s+fee)\b", r"\bbayar\s+(?:deposit|yuran\s+pendaftaran)\b")),
    Rule("A04", "HUMAN_CONTACT_REQUEST", "A", "asked to speak to a counsellor",
         (r"\b(?:speak|talk)\s+(?:to|with)\s+(?:a\s+|an\s+|the\s+)?(?:counsel+or|advis[eo]r|staff|someone|a\s+person|human)\b",
          r"\bconnect\s+me\s+(?:to|with)\s+(?:a\s+|an\s+|the\s+)?(?:counsel+or|advis[eo]r|staff|someone)\b",
          r"\bcakap\s+dengan\s+(?:kaunselor|penasihat)\b")),
    Rule("A05", "CALLBACK_REQUEST", "A", "requested a callback",
         (r"\b(?:can|could|will)\s+(?:someone|anyone|you)\s+call\s+me\b", r"\bcall\s*back\b", r"\bplease\s+call\s+me\b", r"\bboleh\s+(?:call|telefon)\s+saya\b")),
    Rule("A06", "SUBMISSION_OR_COMMITMENT", "A", "submitted or offered to submit results or documents",
         (rf"\b(?:i\s+)?(?:have\s+|already\s+|just\s+)?(?:uploaded|submitted|sent)\s+(?:my\s+)?(?:{_EXAM}|results?|transcripts?|certificates?|documents?)\b",
          rf"\bi(?:'ll|\s+will|\s+can|\s+want\s+to|\s+would\s+like\s+to)\s+(?:upload|submit|send)\s+(?:my\s+)?(?:{_EXAM}|results?|transcripts?|certificates?|documents?)\b",
          r"\bsaya\s+(?:dah|telah|sudah)\s+(?:upload|hantar|muat\s+naik)\b"), negation_sensitive=True),
    Rule("A07", "CREDIT_TRANSFER_OR_EXEMPTION", "A", "asked about credit transfer or exemption",
         (r"\b(?:transfer|exempt)\w*\s+(?:my\s+)?credits?\b", r"\bcredit\s+transfers?\b", r"\bcredits?\s+(?:be\s+)?(?:transferred|exempted)\b",
          r"\bhow\s+many\s+semesters?\s+(?:can\s+i\s+)?skip\b", r"\bpindah\s+kredit\b", r"\bpengecualian\s+kredit\b")),
    Rule("A08", "APEL_OR_WORK_EXPERIENCE", "A", "asked about APEL or work-experience recognition",
         (r"\bapel\b", r"\bpengalaman\s+kerja\s+(?:untuk|bagi)\s+(?:masuk|kemasukan)\b")),
    Rule("A09", "INTRA_MSU_TRANSFER", "A", "asked about changing programme or faculty within MSU",
         (r"\b(?:change|switch|transfer)\s+(?:my\s+)?(?:programme|program|faculty|course)\b", r"\btukar\s+(?:program|fakulti|kursus)\b")),
    Rule("A10", "ELIGIBILITY_QUESTION", "A", "asked whether eligible with their results",
         (r"\bam\s+i\s+eligible\b", r"\bcan\s+i\s+get\s+in(?:to)?\b", r"\bdo\s+i\s+qualify\b", r"\bwill\s+i\s+qualify\b", r"\bsaya\s+layak\b", r"\badakah\s+saya\s+layak\b")),
    Rule("A11", "INTAKE_NAMED", "A", "named a target intake",
         (rf"\b{_MONTHS}\s+intake\b", rf"\bintake\s+(?:in|of|for)\s+{_MONTHS}\b", rf"\bpengambilan\s+{_MONTHS}\b")),
    Rule("A12", "LATE_APPLICATION_QUESTION", "A", "asked whether they can still apply",
         (r"\bstill\s+(?:apply|join|open)\b", r"\bapply\s+late\b", r"\blate\s+application\b", r"\bmasih\s+boleh\s+(?:mohon|daftar)\b")),
    Rule("A13", "APPLICATION_DURATION_QUESTION", "A", "asked how long application or approval takes",
         (r"\bhow\s+long\s+(?:does|will|is)\s+(?:the\s+)?(?:application|approval|process)\b", r"\bberapa\s+lama\s+(?:permohonan|kelulusan)\b")),
    Rule("A14", "URGENCY", "A", "used urgency language",
         (r"\burgent(?:ly)?\b", r"\basap\b", r"\bas\s+soon\s+as\s+possible\b", r"\bresults?\s+just\s+came\s+out\b", r"\bneed\s+to\s+decide\s+this\s+week\b", r"\bsegera\b")),
    Rule("A15", "OWNERSHIP_LANGUAGE", "A", "used ownership language about studying at MSU",
         (r"\bwhen\s+i\s+(?:start|join|begin)\b", r"\bmy\s+(?:classes|intake|lecturers?|class\s+schedule)\b")),
    Rule("A16", "POST_DECISION_LOGISTICS", "A", "asked about hostel, transport or campus logistics",
         (r"\bhostel\b", r"\baccommodation\b", r"\btransport(?:ation)?\b", r"\bclass\s+schedule\b", r"\bcampus\s+location\b",
          r"\bpart[- ]time\b", r"\bweekend\s+(?:class|classes|programme|program)\b", r"\bwork(?:ing)?\s+while\s+studying\b", r"\basrama\b", r"\bpenginapan\b"),
         review_flag="BROAD_SCOPE_REVIEW"),
    Rule("A17", "OPEN_DAY_OR_VISIT", "A", "asked about an open day or campus visit",
         (r"\bopen\s+day\b", r"\bvisit\s+(?:the\s+)?campus\b", r"\bcampus\s+visit\b", r"\bhari\s+terbuka\b")),
    Rule("A18", "PRICE_OBJECTION", "A", "pushed back on price",
         (r"\bisn't\s+that\s+expensive\b", r"\btoo\s+expensive\b", r"\bany\s+discounts?\b", r"\bdiskaun\b", r"\bterlalu\s+mahal\b"),
         review_flag="SEMANTIC_OBJECTION_REVIEW"),
    Rule("A19", "RECOGNITION_QUESTION", "A", "questioned recognition (MQA, JPA or similar)",
         (r"\bmqa\b", r"\bjpa\b", r"\b(?:recogni[sz]ed|recognition)\b", r"\bdiiktiraf\b"), review_flag="SEMANTIC_OBJECTION_REVIEW"),
    # ---------------- Tier B ----------------
    Rule("B01", "FUNDING_QUESTION", "B", "asked about scholarships or funding",
         (r"\bscholarships?\b", r"\bptptn\b", r"\bepf\b", r"\bmara\b", r"\binstal+ments?\b", r"\bbiasiswa\b", r"\bpinjaman\s+pendidikan\b")),
    Rule("B02", "CAMPUS_TOUR", "B", "asked about the campus tour video",
         (r"\bcampus\s+tour\b", r"\btour\s+video\b", r"\bvirtual\s+tour\b")),
    Rule("B03", "SPONSOR_OR_DECISION_MAKER", "B", "mentioned a sponsor or decision-maker",
         (r"\bmy\s+(?:father|mother|parents?|company|employer)\s+(?:wants|want|sponsors?|will\s+pay|is\s+paying)\b", r"\bsponsor(?:s|ed)?\s+me\b")),
)

_RULES_BY_ID = {r.rule_id: r for r in RULES}
_COMPILED = {r.rule_id: tuple(re.compile(p, re.IGNORECASE) for p in r.patterns) for r in RULES}

_ACK_TOKENS = {
    "ok", "okay", "okey", "k", "thanks", "thank you", "thankyou", "thx", "tq", "noted", "got it", "alright", "ic", "i see",
    "ohh", "oh", "understood", "bye", "terima kasih", "baik", "faham", "okay thanks", "ok thanks", "ok thank you",
}
_NEGATION = re.compile(r"\b(?:not|never|haven't|havent|hasn't|didn't|didnt|no|belum|tak|tidak)\b")
_CONTACT_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CONTACT_PHONE = re.compile(r"(?<!\d)(?:\+?6?0?1\d[\s-]?\d{3,4}[\s-]?\d{4}|\+?\d{9,13})(?!\d)")
_ASKED_FOR_CONTACT = re.compile(r"\b(?:phone|mobile|whatsapp|email|e-mail|contact)\s+(?:number|no\.?|address|details)?\b.*\b(?:share|provide|give|leave|send)\b|"
                               r"\b(?:share|provide|give|leave|send)\b.*\b(?:phone|mobile|whatsapp|email|contact)\b", re.IGNORECASE)
_EXAM_RE = re.compile(rf"\b{_EXAM}\b", re.IGNORECASE)
_GRADE_RE = re.compile(r"\b(?:\d{1,2}\s*(?:a\+?|a-|b\+?|c\+?|d|e)s?\b|grade\s+[a-e][+-]?\b|cgpa\s*(?:of|is|:)?\s*\d(?:\.\d+)?|gpa\s*(?:of|is|:)?\s*\d(?:\.\d+)?|\d{1,3}\s*%|\d\s+credits?\b|\ball\s+a\b|\bpass(?:ed)?\b)", re.IGNORECASE)
_RESULT_WORD = re.compile(r"\b(?:results?|grades?|keputusan|markah)\b", re.IGNORECASE)
_STATUS_RE = re.compile(rf"\bi(?:'ve|\s+have)?\s+(?:just\s+)?(?:finished|completed|graduated\s+from|got\s+(?:a|my)|have\s+(?:a|an))\s+(?:{_EXAM})\b|\bsaya\s+(?:baru\s+)?(?:habis|tamat|ada)\s+{_EXAM}\b", re.IGNORECASE)
_STATUS_ASKED = re.compile(r"\b(?:what|which)\b.*\b(?:qualification|level|results?|studying)\b", re.IGNORECASE)
_FEE_WORD = re.compile(r"\b(?:fees?|yuran|tuition|cost)\b", re.IGNORECASE)
_ENTRY_WORD = re.compile(r"\b(?:entry\s+requirements?|requirements?|syarat\s+kemasukan)\b", re.IGNORECASE)
_PROG_RE = re.compile(r"\b(bachelor|diploma|master|phd|doctorate|foundation|degree|sarjana|ijazah)\b(?:\s+(?:of|in|dalam))?\s+((?:[a-z]+\s?){0,5})", re.IGNORECASE)
_PROG_STOP = {"fee", "fees", "requirement", "requirements", "for", "at", "msu", "how", "what", "is", "are", "the", "and", "or", "cost", "yuran", "tuition", "program", "programme", "course", "to", "can", "do", "does", "price", "much", "many", "when", "i", "my", "please", "pls", "again", "also", "now", "per", "semester", "year", "a", "an", "if", "there", "that", "this", "with", "only", "total", "first", "too", "is", "was", "will", "does", "did", "have", "has", "entry", "admission", "admissions", "intake", "duration", "length", "details", "info", "information", "eligibility", "about", "offered", "available"}
_CAPABILITY_UPLOAD = re.compile(r"\b(?:can|could|may|am\s+i\s+able\s+to|boleh)\s+(?:i|saya)\s+(?:upload|submit|send|hantar|muat\s+naik)\b", re.IGNORECASE)
_DEADLINE = re.compile(r"\b(?:deadline|closing\s+date|tarikh\s+tutup)\b", re.IGNORECASE)
_THINK = re.compile(r"\blet\s+me\s+think\b|\bdiscuss\s+with\s+my\s+parents\b", re.IGNORECASE)
_COMPARE = re.compile(r"\b(?:difference\s+between|compare|better)\b.*\b(?:bachelor|diploma|bba|degree)\b.*\b(?:or|and|vs)\b", re.IGNORECASE)
_PARENT = re.compile(r"\bmy\s+(?:son|daughter|child)\b|\banak\s+saya\b", re.IGNORECASE)
_CHILD_LEVEL = re.compile(rf"\b(?:{_EXAM}|form\s+\d|year\s+\d|grade\s+\d+)\b", re.IGNORECASE)


def _is_ack_only(norm: str) -> bool:
    stripped = re.sub(r"[^\w\s]", "", norm).strip()
    return stripped in _ACK_TOKENS


def _has_unsupported_script(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and ord(ch) > 0x24F:
            return True
    return False


def _programme_key(norm: str) -> Optional[str]:
    m = _PROG_RE.search(norm)
    if not m:
        return None
    words: List[str] = []
    for w in m.group(2).split():
        if w in _PROG_STOP:
            break
        words.append(w)
    if not words:
        return None
    return f"{m.group(1).lower()}:{' '.join(words[:4])}"


@dataclass
class SessionEvidence:
    """Approved session-boundary evidence (B9). Without approval the builder emits no return-visit signal."""
    approved: bool
    session_count: int


@dataclass
class SummaryResult:
    action_summary: List[str]
    extracted_actions: List[ExtractedAction]
    ambiguities: List[AmbiguityRecord]
    unknown_context: List[str]
    builder_version: str = BUILDER_VERSION
    summary_provenance: str = SUMMARY_PROVENANCE


def tier_tally(actions: Sequence[ExtractedAction]) -> Dict[str, int]:
    """Counts only (Tier A actions, distinct Tier B action types). This is NOT a label and NOT ground truth."""
    return {
        "tier_a_actions": sum(1 for a in actions if a.tier == "A"),
        "tier_b_distinct_types": len({a.action_code for a in actions if a.tier == "B"}),
    }


def build_action_summary(
    history: Sequence[Turn],
    history_status: HistoryStatus = HistoryStatus.UNKNOWN,
    session_evidence: Optional[SessionEvidence] = None,
) -> SummaryResult:
    """Extract auditable action signals from USER turns of the supplied history only."""
    actions: Dict[Tuple[str, Optional[str]], ExtractedAction] = {}
    order: List[Tuple[str, Optional[str]]] = []
    ambiguities: List[AmbiguityRecord] = []
    unknown: List[str] = []
    fee_turns: Dict[str, List[Turn]] = {}
    entry_turns: Dict[str, List[Turn]] = {}

    def add(rule_id: str, code: str, tier: str, summary: str, turn: Turn, programme: Optional[str] = None):
        key = (code, programme)
        if key in actions:
            actions[key].occurrences += 1
            return
        text = summary if programme is None else f"{summary} ({programme.split(':', 1)[1]})"
        actions[key] = ExtractedAction(
            action_code=code, tier=tier, rule_id=rule_id, turn_index=turn.turn_index,
            source_ref=turn.source_ref, programme_key=programme, summary_text=text,
        )
        order.append(key)

    def amb(rule_id: str, reason: str, turn: Optional[Turn], detail: str = ""):
        ambiguities.append(AmbiguityRecord(
            rule_id=rule_id, reason_code=reason,
            turn_index=turn.turn_index if turn else None, source_ref=turn.source_ref if turn else None, detail=detail,
        ))

    if history_status in (HistoryStatus.INCOMPLETE, HistoryStatus.UNKNOWN, HistoryStatus.NONE_AVAILABLE):
        unknown.append(f"HISTORY_{history_status.value}")
    if session_evidence is None or not session_evidence.approved:
        unknown.append("RETURN_VISIT_UNVERIFIABLE")

    prev_assistant: Optional[Turn] = None
    for turn in history:
        if turn.role == Role.ASSISTANT:
            prev_assistant = turn
            continue
        text = turn.text or ""
        norm = _normalise(text)
        if not norm:
            continue
        if _has_unsupported_script(text):
            amb("B8", "UNSUPPORTED_SCRIPT", turn, "Non-Latin script present; only English and Malay are supported in v1")
        if _is_ack_only(norm):
            prev_assistant = prev_assistant
            continue

        fired_codes = set()
        for rule in RULES:
            for rx in _COMPILED[rule.rule_id]:
                m = rx.search(norm)
                if not m:
                    continue
                if rule.negation_sensitive and _NEGATION.search(norm[max(0, m.start() - 30): m.start()]):
                    amb(rule.rule_id, "NEGATION_PRESENT", turn, "Pattern preceded by a negation; not counted")
                    break
                if rule.rule_id == "A06" and _CAPABILITY_UPLOAD.search(norm):
                    amb("A06", "CAPABILITY_QUESTION_NOT_ACTION", turn, "Asked whether upload is possible; not a submission (B1)")
                    break
                add(rule.rule_id, rule.action_code, rule.tier, rule.summary, turn)
                fired_codes.add(rule.action_code)
                if rule.review_flag:
                    amb(rule.rule_id, rule.review_flag, turn, "Provisional default flags this rule for review")
                break

        # A06 capability question without any pattern hit still deserves a flag
        if "SUBMISSION_OR_COMMITMENT" not in fired_codes and _CAPABILITY_UPLOAD.search(norm):
            amb("A06", "CAPABILITY_QUESTION_NOT_ACTION", turn, "Asked whether upload is possible; not a submission (B1)")

        # Volunteered contact details: only when the immediately preceding assistant turn did not ask for them
        if _CONTACT_EMAIL.search(text) or _CONTACT_PHONE.search(text) or "[EMAIL]" in text or "[PHONE]" in text:
            if prev_assistant is None:
                if history and history[0].turn_index != turn.turn_index:
                    amb("A_VOLUNTEERED_CONTACT", "ASSISTANT_CONTEXT_MISSING", turn, "Cannot tell whether contact details were requested")
                else:
                    add("A20", "VOLUNTEERED_CONTACT", "A", "volunteered contact details", turn)
            elif _ASKED_FOR_CONTACT.search(prev_assistant.text or ""):
                pass
            else:
                add("A20", "VOLUNTEERED_CONTACT", "A", "volunteered contact details", turn)

        # Stated results (A, needs specificity) vs stated status (B); A wins (B7)
        has_exam = bool(_EXAM_RE.search(norm))
        specific = has_exam and bool(_GRADE_RE.search(norm))
        if specific:
            add("A21", "RESULTS_STATED", "A", "stated specific results", turn)
        elif has_exam and _RESULT_WORD.search(norm):
            amb("A21", "NONSPECIFIC_RESULTS", turn, "Results mentioned without specific grades or scores (B2)")
        if not specific and _STATUS_RE.search(norm):
            if prev_assistant is not None and _STATUS_ASKED.search(prev_assistant.text or ""):
                pass
            elif prev_assistant is None and history and history[0].turn_index != turn.turn_index:
                amb("B04", "ASSISTANT_CONTEXT_MISSING", turn, "Cannot tell whether status was unprompted")
            else:
                add("B04", "STATED_ACADEMIC_STATUS", "B", "stated their academic status unprompted", turn)

        # Programme fee / entry enquiries (heuristic programme key)
        key = _programme_key(norm)
        if _FEE_WORD.search(norm):
            if key:
                add("B05", "PROGRAMME_FEE_QUESTION", "B", "asked the fee for a named programme", turn, key)
                fee_turns.setdefault(key, []).append(turn)
                amb("B05", "HEURISTIC_PROGRAMME_KEY", turn, "Programme resolved by heuristic pattern, not a KB lookup")
            else:
                amb("B05", "PROGRAMME_NOT_RESOLVED", turn, "Fee question without a resolvable named programme")
        if _ENTRY_WORD.search(norm) and key:
            entry_turns.setdefault(key, []).append(turn)
        if _DEADLINE.search(norm):
            amb("A_DEADLINE_AFTER_PROGRAMME", "PROGRAMME_CONTEXT_UNRESOLVED", turn, UNIMPLEMENTED_RULES["A_DEADLINE_AFTER_PROGRAMME"])
        if _THINK.search(norm):
            amb("A_THINK_IT_OVER_AFTER_PROGRAMME", "PROGRAMME_CONTEXT_UNRESOLVED", turn, UNIMPLEMENTED_RULES["A_THINK_IT_OVER_AFTER_PROGRAMME"])
        if _COMPARE.search(norm):
            amb("B_PROGRAMME_COMPARISON", "PROGRAMME_CONTEXT_UNRESOLVED", turn, UNIMPLEMENTED_RULES["B_PROGRAMME_COMPARISON"])
        if _PARENT.search(norm):
            if _CHILD_LEVEL.search(norm):
                add("B06", "PARENT_ENQUIRY", "B", "parent enquiring on behalf of a child with level or results", turn)
            else:
                amb("B06", "PARENT_CHILD_LEVEL_MISSING", turn, "Parent enquiry without the child's level or results")

    # Confirmed L1: repeated wording of the same enquiry is one Tier B signal, and the same evidence is never counted twice.
    for key, ent in entry_turns.items():
        if key in fee_turns:
            # The fee action already covers this programme; entry-requirement asks add no second signal.
            amb("B05", "MIXED_ENQUIRY_TOPICS_COUNTED_ONCE", ent[0], "Fee and entry requirements asked for the same programme; counted as one signal")
        elif len(ent) >= 2:
            for t in ent:
                add("B08", "PROGRAMME_ENQUIRY_REPEATED", "B", "asked about the same programme's entry requirements more than once", t, key)

    if session_evidence is not None and session_evidence.approved and session_evidence.session_count >= 2:
        first = history[0] if history else None
        if first is not None:
            add("B07", "RETURN_VISIT", "B", "returned to the chat in a later session", first)

    summary: List[str] = []
    for key in order:
        a = actions[key]
        summary.append(a.summary_text if a.occurrences == 1 else f"{a.summary_text} ({a.occurrences} times)")
    return SummaryResult(
        action_summary=summary,
        extracted_actions=[actions[k] for k in order],
        ambiguities=ambiguities,
        unknown_context=unknown,
    )


def build_lead_case(
    case_id: str,
    history: Sequence[Turn],
    classification: DataClassification = DataClassification.UNCLASSIFIED,
    history_status: HistoryStatus = HistoryStatus.UNKNOWN,
    group_id: Optional[str] = None,
    session_evidence: Optional[SessionEvidence] = None,
) -> LeadCase:
    """Build an unreviewed LeadCase. `expected_need_cta` stays None: ground truth comes only from human review."""
    result = build_action_summary(history, history_status, session_evidence)
    return LeadCase(
        case_id=case_id,
        classification=classification,
        group_id=group_id,
        history=list(history),
        history_status=history_status,
        action_summary=result.action_summary,
        extracted_actions=result.extracted_actions,
        ambiguities=result.ambiguities,
        unknown_context=result.unknown_context,
        summary_builder_version=result.builder_version,
    )
