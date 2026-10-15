# Dataset cleaning: how the historical export was prepared

Status: local preparation only. `PRODUCTION_PARITY_NOT_ESTABLISHED`. No labels approved. Nothing was sent to any provider.

- **Source:** `september_to_october.csv` (read only; git-ignored). SHA-256 prefix `fd6b06a2d4ad`, checked before and after the run: unchanged.
- **Code:** `src/msu_llm_lab/classification/dataprep.py`, run with `msu-llm-lab prepare-classification-data`.
- **Output folder:** `data/classification_prep/prep-v1/` (git-ignored; it contains redacted student text, so it must never be committed or uploaded).
- **Version:** `prep-v1`, deterministic (the same input gives the same output).

## 1. What happened to each source column

The export has 8 columns and 2,224 data rows.

| Source column | What was done | In the cleaned output? |
|---|---|---|
| `traceId` | Replaced by a salted SHA-256 prefix (`trace_ref`, 12 characters). The raw value is not stored. | Hash only |
| `time` | Parsed as ISO-8601 UTC and kept, as `time`. | Yes |
| **`user4`** | **Dropped.** It holds the first digits of phone numbers, so it is sensitive and not a user ID. It is never accessed, copied, hashed, grouped on or used for splitting. The preparation code refuses any output structure that has a `user4` key. | **No** |
| `user_hash` | Replaced by a salted SHA-256 prefix (`group_id`, "G" plus 10 characters). It is a provisional grouping key, not a verified identity. The raw value is not stored. | Hash only |
| `conversation_id` | Empty in every row (audited earlier), so it carries no information. It is only checked for presence and is not stored. | No |
| `turn` | Kept as `turn_source`, as supporting evidence only. It never creates or removes a session boundary. | Yes |
| `query` | Identifiers redacted (section 3), then kept as `query_redacted`. | Redacted text |
| `answer` | Same redaction, kept as `answer_redacted`. An empty answer stays `null` with status `MISSING`; no reply is ever invented. | Redacted text or null |

A scan of every output file found none of the raw `user_hash` or `traceId` values. No `user4` key or value appears in any file except one note in the manifest saying the column was excluded.

## 2. Row accounting

All 2,224 rows are accounted for in `cleaned_rows.jsonl`. No row was deleted.

| Disposition | Rows |
|---|---|
| `INCLUDED` | 1,924 |
| `INCLUDED_FLAGGED` | 300 |
| `EXCLUDED` | 0 |

A row is `EXCLUDED` only if it has no usable query (none did). It is `INCLUDED_FLAGGED` if it has any of the flags below.

| Flag | Rows | Meaning |
|---|---|---|
| `MISSING_ANSWER` | 107 | The answer field was empty. |
| `REDACTED` | 68 | At least one identifier was replaced. |
| `EXACT_DUPLICATE` | 22 | Same user group, same query and same answer as an earlier row. Flagged, not removed. |
| `NEAR_DUPLICATE` | 70 | Same normalised query re-sent by the same group within 5 minutes. Flagged, not removed. |
| `MULTILINE_QUERY` | 55 | The query contains line breaks, so it may combine several messages. |
| `UNSUPPORTED_SCRIPT` | 11 | Non-Latin script; only English and Malay are supported by the rules. |
| `VERY_LONG_QUERY` | 4 | Over 600 characters. |

## 3. Redaction

Redaction happens before anything is written. The tokens are `[EMAIL]`, `[ID_NUMBER]`, `[PHONE]` and `[NAME]`.

| Identifier | How it is found | Replaced in queries | Replaced in answers |
|---|---|---|---|
| Email address | Standard email pattern | 2 | 15 |
| National ID (12 digits, `yymmdd-pb-####`) | Pattern | 0 | 0 |
| Phone number | Malaysian mobile pattern, plus any 9 or more digit run | 11 | 62 |
| Name | Only the words after "my name is", "nama saya" or "name:" | 6 | 0 |

68 rows had at least one replacement.

Limits, stated plainly:
- **No named-entity recognition.** A name that isn't introduced explicitly is not redacted.
- Programme names, addresses and other indirect identifiers are not redacted.
- The assistant answers also have phone numbers and emails redacted. Many of those are probably the university's own contact details, which the redaction doesn't tell apart from a student's. That loses a little context but does not leak anything.
- A text scan afterwards found no remaining 9-or-more digit runs inside the query and answer fields.

## 4. Sessions (two provisional options; neither is chosen)

`conversation_id` is empty and `turn` is unreliable, so sessions are rebuilt from user group plus time order. A new session starts when the gap since the user's previous row exceeds the threshold.

| | 30 minutes | 2 hours |
|---|---|---|
| Sessions | 778 | 736 |
| Rows that open a new session after a gap | 136 | 94 |
| Rows that continue a session | 1,446 | 1,488 |
| Single-row sessions | 344 | 316 |
| Turn counter agrees with the session | 2,060 | 2,098 |
| Turn restarts at 1 inside a session | 37 | 39 |
| Session starts without turn 1 | 126 | 86 |

42 rows have a gap between 30 minutes and 2 hours (these are the rows that differ between the policies). A 12-group redacted review sample is in `session_review_sample.jsonl` for manual inspection.

## 5. Candidate datasets (labels are all null)

| Dataset | Unit | Cases per policy |
|---|---|---|
| Main Intent | One per student row. History is the earlier rows in the same session; the current message is that row's query. | 2,224 |
| Lead Qualifier | One per session. Contains the history and a separate action summary from the offline builder, with the source row of every action. | 736 (2 h), 778 (30 min) |

- **Labels:** `expected_intent` and `expected_need_cta` are null. No suggested labels were produced. The historical assistant answer is context only, never a prediction or a label.
- **Classification:** every case is `CONFIDENTIAL`, so the privacy gate blocks all of them from any provider.
- **Intent exclusions (16 per policy):** 5 under unresolved rule O1 (a bare "no"/"thanks" replying to an offer) and 11 short replies whose preceding assistant message is missing. A further 20 are marked `NEEDS_CONTEXT`.
- **Lead exclusions (13 for 2 h, 17 for 30 min):** unresolved rules L4–L8, only where no Tier A action already settles the answer.
- **Not claimed:** the reconstructed history and the summary do not reproduce the production input. The Dify export is missing.

## 6. Splits and leakage

- **Grouping:** all rows of one provisional user group stay in one split.
- **Views:** 642 groups. 121 contain phrases that hint at rare classes (handoff, upload status, out of scope); 61 of those form a separate **challenge** view (355 intent cases). The other 581 groups form the **natural** view (1,869 cases). The natural view therefore under-represents rare intents. The two views are never combined into one accuracy figure.
- **Splits:** DEV and TEST at 50/50 within each view, seed `msu-prep-v1-seed`. The assignment is the same under both session policies.
- **Checks:** 0 groups appear in more than one split or view. 67 distinct stock messages (greetings and similar) appear in both DEV and TEST, which is expected but reported.

## 7. Files written (`data/classification_prep/prep-v1/`)

| File | Contents |
|---|---|
| `manifest.json` | Counts, parameters, source hash, checks. No student text. |
| `cleaned_rows.jsonl` | One redacted row per source row, with flags, duplicates and session ids. |
| `sessions_30m.jsonl`, `sessions_2h.jsonl` | Session id and row ids. |
| `intent_candidates_30m.jsonl`, `intent_candidates_2h.jsonl` | Intent cases. |
| `lead_candidates_30m.jsonl`, `lead_candidates_2h.jsonl` | Lead cases. |
| `split_assignments.jsonl` | Group to view and split. |
| `session_review_sample.jsonl` | Redacted sample for manual boundary review. |
| `README.txt` | The parity warning. |

## 8. What still needs a human decision

1. 30-minute or 2-hour sessions.
2. Whether Lead cases should be per session (as prepared) or per turn.
3. Review and approval of all labels, independent of the builder's summary.
4. Whether the assistant-answer redactions (institutional phones and emails) should be loosened.
5. A verified provider data policy before any of this text can go to a model.

## 9. Hardening pass (after an external code review)

These changes apply to the code and to any NEW preparation run. The existing `prep-v1` folder was not regenerated or modified.

- **No silent overwrite.** `prepare-classification-data` now refuses to write into a non-empty output folder (it checks before reading the CSV). A folder that holds an `annotations/` directory gets a specific protective message. Use `--new-version` to write a sibling version (`prep-v1-r2`, `prep-v1-r3`, ...). Human annotations belong in a separate versioned `annotations/` folder that no rerun may touch, keyed by **row id** (`R0001...`, file order), not by group id.
- **Run provenance.** The manifest now records a code fingerprint (hash of the preparation, summary-builder, contract and split sources), the git commit, creation time and all parameters (session thresholds, seed, split ratios, challenge fraction).
- **O1 narrowed.** A bare "no" or "thanks" is treated as the unresolved O1 case only when the assistant's previous message made a concrete non-counsellor offer ("would you like...", "do you want...", "shall I...", "can I help with..."). A plain clarifying question ending in "?" no longer qualifies, a generic "let me know" closing no longer qualifies, and counsellor-connect offers follow the handoff rule instead. The existing `prep-v1` candidates were made with the broader rule (5 intent cases per policy were excluded under O1), so some of those exclusions may be over-broad until a new version is prepared.
- **Pseudonymisation is not anonymisation.** Group and trace ids are salted hashes. `prep-v1` used a PUBLIC constant salt, so anyone holding the export can re-link them. A new version can use a private secret by setting `MSU_PREP_SALT` in the shell before running. The salt is never written to any file; only a short `salt_id` is recorded. **Migration:** changing the salt changes every group id and therefore the group-level splits, so a protected version is a new dataset; annotations must be migrated by row id. Existing ids were not rotated.
- **Limitations are visible everywhere.** Every derived record now carries `privacy_notice` (`REDACTION_INCOMPLETE_NO_NER;PSEUDONYMS_ARE_NOT_ANONYMOUS;LOCAL_ONLY`), and the README and manifest say so.
- **Views are unvalidated.** The challenge view is a candidate pool chosen by phrase hints before any label review, and the natural view is not validated as representative. Candidates and split assignments carry `view_status: HEURISTIC_UNVALIDATED`, and the manifest explains both.
