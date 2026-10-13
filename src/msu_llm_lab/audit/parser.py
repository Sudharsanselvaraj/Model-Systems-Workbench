"""Parser for MSU Knowledge Base Markdown documents."""

import re
from pathlib import Path
from typing import List, Tuple, Optional

from msu_llm_lab.audit.models import Heading, KBRecord


# Regexes constructed with re.escape to guarantee matching escaped and unescaped variants
B_ESC = re.escape(r'\*\*')
B_RAW = re.escape(r'**')
BOLD = f'(?:{B_ESC}|{B_RAW})'

q_variants = [
    rf'{BOLD}\s*Question\s*:\s*{BOLD}:?',
    rf'{BOLD}\s*Question\s*{BOLD}\s*:\s*',
    rf'{BOLD}\s*Question\s*{BOLD}',
    r'(?:^|(?<=\n)|(?<=\s)|[\"\'\>])Question\s*:\s*',
]
Q_MARKER_REGEX = re.compile('|'.join(q_variants), re.IGNORECASE)

a_variants = [
    rf'{BOLD}\s*Answer\s*:\s*{BOLD}:?',
    rf'{BOLD}\s*Answer\s*{BOLD}\s*:\s*',
    rf'{BOLD}\s*Answer\s*{BOLD}',
    r'(?:^|(?<=\n)|(?<=\s)|[\"\'\>])Answer\s*:\s*',
]
A_MARKER_REGEX = re.compile('|'.join(a_variants), re.IGNORECASE)

HEADING_REGEX = re.compile(r'^(?:#{1,6}|(?:\\\\#){1,6})\s*(.*)$')
DIVIDER_REGEX = re.compile(r'^\s*\\?--{3,}\s*$')


def unescape_markdown(text: str) -> str:
    """Removes backslash escapes for markdown characters."""
    return re.sub(r'\\([*#+\-.:~()\[\]&_`"\'\?\\/!])', r'\1', text)


def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces/newlines into clean single spaces."""
    return " ".join(text.split()).strip()


def clean_text(text: str) -> str:
    """Unescapes and normalizes whitespace."""
    return normalize_whitespace(unescape_markdown(text))


class KBParser:
    """Robust parser for MSU Markdown knowledge bases."""

    def __init__(self, source_path: Path):
        self.source_path = Path(source_path)
        self.lines: List[str] = []
        self.raw_question_markers: int = 0
        self.raw_answer_markers: int = 0
        self.records: List[KBRecord] = []
        self.headings: List[Heading] = []
        self.orphan_lines: List[Tuple[int, str]] = []

    def parse(self) -> Tuple[int, int, List[KBRecord], List[Heading], List[Tuple[int, str]]]:
        """Parses the source file into structured records and metadata."""
        with open(self.source_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Count raw markers across full text
        self.raw_question_markers = len(list(Q_MARKER_REGEX.finditer(content)))
        self.raw_answer_markers = len(list(A_MARKER_REGEX.finditer(content)))

        self.lines = content.splitlines()
        self._parse_lines()
        return (
            self.raw_question_markers,
            self.raw_answer_markers,
            self.records,
            self.headings,
            self.orphan_lines,
        )

    def _parse_lines(self) -> None:
        """State-machine line parser that preserves line numbers and original text."""
        current_programme = "General"
        section_hierarchy: List[str] = ["General Information"]
        record_counter = 0

        # Accumulator for currently open record
        in_record = False
        in_answer_phase = False
        start_line = 0
        question_lines: List[Tuple[int, str]] = []
        answer_lines: List[Tuple[int, str]] = []
        raw_lines: List[Tuple[int, str]] = []
        has_question_tag = False
        has_answer_tag = False
        answer_marker_count = 0

        def finalize_record() -> None:
            nonlocal in_record, in_answer_phase, start_line, question_lines, answer_lines
            nonlocal raw_lines, has_question_tag, has_answer_tag, answer_marker_count, record_counter

            if not in_record:
                return

            end_line = raw_lines[-1][0] if raw_lines else start_line
            raw_text = "\n".join(txt for _, txt in raw_lines)
            q_raw = "\n".join(txt for _, txt in question_lines).strip()
            a_raw = "\n".join(txt for _, txt in answer_lines).strip()

            q_clean = clean_text(q_raw)
            a_clean = clean_text(a_raw)

            record_counter += 1
            rec_id = f"REC-{record_counter:04d}"

            is_multiline = (end_line > start_line)
            is_inline = has_question_tag and has_answer_tag and len(question_lines) > 0 and len(answer_lines) > 0 and (question_lines[0][0] == answer_lines[0][0])

            record = KBRecord(
                record_id=rec_id,
                source_file=str(self.source_path),
                start_line=start_line,
                end_line=end_line,
                raw_text=raw_text,
                programme_name=current_programme,
                section_hierarchy=list(section_hierarchy),
                question_raw=q_raw,
                question_clean=q_clean,
                answer_raw=a_raw,
                answer_clean=a_clean,
                has_question_tag=has_question_tag,
                has_answer_tag=has_answer_tag,
                is_multiline=is_multiline,
                is_inline=is_inline,
                answer_marker_count=max(answer_marker_count, 1 if has_answer_tag else 0),
            )
            self.records.append(record)

            in_record = False
            in_answer_phase = False
            start_line = 0
            question_lines = []
            answer_lines = []
            raw_lines = []
            has_question_tag = False
            has_answer_tag = False
            answer_marker_count = 0

        for line_idx, line in enumerate(self.lines, start=1):
            stripped = line.strip()

            # 1. Heading detection
            heading_match = HEADING_REGEX.match(stripped)
            if heading_match:
                finalize_record()
                raw_hash_match = re.match(r'^(#+)', stripped)
                if raw_hash_match:
                    level = len(raw_hash_match.group(1))
                else:
                    level = stripped.count(r'\#') if r'\#' in stripped else 1
                level = min(max(level, 1), 6)

                h_text_raw = heading_match.group(1).strip()
                h_text_clean = clean_text(h_text_raw)
                h_text_clean = re.sub(r'^[#\s]+', '', h_text_clean).strip()

                heading = Heading(
                    level=level,
                    text=h_text_clean,
                    raw_text=line,
                    line_number=line_idx,
                )
                self.headings.append(heading)

                if level == 1:
                    if h_text_clean:
                        current_programme = h_text_clean
                        section_hierarchy = [h_text_clean]
                elif level == 2:
                    if h_text_clean:
                        if len(section_hierarchy) > 1:
                            section_hierarchy[1] = h_text_clean
                        else:
                            section_hierarchy.append(h_text_clean)
                continue

            # 2. Divider detection
            if DIVIDER_REGEX.match(stripped):
                finalize_record()
                continue

            # 3. Scan for Question and Answer markers on this line
            q_match = Q_MARKER_REGEX.search(line)
            a_match = A_MARKER_REGEX.search(line)

            if q_match:
                finalize_record()

                in_record = True
                start_line = line_idx
                raw_lines.append((line_idx, line))
                has_question_tag = True

                if a_match and a_match.start() > q_match.end():
                    q_part = line[q_match.end():a_match.start()]
                    a_part = line[a_match.end():]

                    question_lines.append((line_idx, q_part))
                    answer_lines.append((line_idx, a_part))
                    in_answer_phase = True
                    has_answer_tag = True
                    answer_marker_count = 1
                else:
                    q_part = line[q_match.end():]
                    question_lines.append((line_idx, q_part))
                    in_answer_phase = False
                continue

            elif a_match:
                if in_record and not in_answer_phase:
                    in_answer_phase = True
                    has_answer_tag = True
                    answer_marker_count = 1
                    raw_lines.append((line_idx, line))

                    pre_a = line[:a_match.start()].strip()
                    if pre_a:
                        question_lines.append((line_idx, pre_a))

                    post_a = line[a_match.end():]
                    answer_lines.append((line_idx, post_a))
                elif in_record and in_answer_phase:
                    raw_lines.append((line_idx, line))
                    answer_marker_count += 1
                    post_a = line[a_match.end():]
                    answer_lines.append((line_idx, post_a))
                else:
                    finalize_record()
                    in_record = True
                    start_line = line_idx
                    raw_lines.append((line_idx, line))
                    has_question_tag = False
                    has_answer_tag = True
                    in_answer_phase = True
                    answer_marker_count = 1
                    post_a = line[a_match.end():]
                    answer_lines.append((line_idx, post_a))
                continue

            # 4. Continuation lines
            if in_record:
                raw_lines.append((line_idx, line))
                if in_answer_phase:
                    answer_lines.append((line_idx, line))
                else:
                    question_lines.append((line_idx, line))
            else:
                if stripped:
                    self.orphan_lines.append((line_idx, stripped))

        finalize_record()
