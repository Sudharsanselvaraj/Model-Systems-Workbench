# Live experimental dashboard

Read-only. It shows what has actually been recorded and never sends a model request, never approves a label, never selects a session policy and never regenerates or edits preparation files.

## Where the numbers come from

| Source | What it holds | Written by |
|---|---|---|
| `data/telemetry/request_events.jsonl` | One event per request lifecycle step: `REQUEST_STARTED`, `REQUEST_SUCCEEDED` (transport success; status `OK` or `INVALID_OUTPUT`), `REQUEST_FAILED` (status `ERROR`, with `http_status` and `timed_out`), `REQUEST_RETRY` (one per retried attempt, reported when the request ends) | `ClassificationRunner` only, around each real request |
| Prediction records (`reports/smoke/*.jsonl`, `reports/classification/*.jsonl`) | Stored predictions with nullable tokens, latency, probabilities, cost | the runner |
| SQLite ledger (`data/workbench.db`) | Spend by basis, unresolved entries | the budget service |
| `data/classification_prep/<version>/manifest.json` | Counts and statuses (no student text) | `prepare-classification-data` |
| `<prep>/annotations/approved_labels.jsonl` | Human-approved labels (`status: APPROVED` and a named `approved_by`) | a human review process (not built yet); the dashboard only reads it |

Events carry opaque case ids and numbers only: no prompt text, no model output, no student text.

## Refresh model

One `LiveMonitor` per dashboard session holds a single `EventTail` (byte offset plus delivered event ids). The UI loop polls it every 2 seconds while the keyboard is idle and on `r`. There are no threads and no per-view or per-redraw subscriptions, so nothing accumulates when you change view; `close()` releases it on exit. A half-written last line waits for the next poll, duplicates are impossible, and a truncated log resets safely. If the log is missing the dashboard shows `NO LIVE INFERENCE YET`; graphs stay idle (a bucket with no measurement is blank, never zero).

## Provenance words

| Word | Meaning |
|---|---|
| ACTUAL | provider-reported (or manually reconciled) cost; measured request data |
| ESTIMATED | sourced catalogue price x provider-reported tokens; never added to ACTUAL |
| UNRESOLVED | request whose cost could not be established; blocks further live runs until `budget reconcile` |
| LEGACY-SIMULATED | old ledger rows from versions that had a dry-run mode; shown for transparency, never counted. There is no simulation mode any more, so a SIMULATED telemetry event is refused on write and quarantined (counted, excluded) on read |
| UNAVAILABLE | the value was not supplied or could not be measured |
| GROUND TRUTH PENDING | needs human-approved labels that do not exist yet |

## Views

1. **Overview.** Status of all categories.
2. **Model Roster.** Ten participants, pinned routes, readiness, exact blockers.
3. **Classification Quality.** Schema validity and invalid-label rate (measured, no labels needed). Accuracy, macro-F1, per-class P/R/F1, five-class confusion matrix and a Wilson 95% interval only from approved labels. Samples under 30 and classes under 5 cases are flagged. Request errors count as invalid predictions.
4. **Confidence & Calibration.** Only a native probability distribution is calibratable (Jev). Shows selected-label probability, mean per-label probabilities and Jev's separate `confidence` field. With approved labels: ECE (equal-count bins, at least 5 cases per bin on average), multiclass Brier, reliability bins, high-confidence (>= 0.9) errors and coverage-versus-accuracy. Textual or self-reported confidence is never used.
5. **Model Agreement.** Pairwise agreement and Cohen's kappa on the cases both models answered, agreement with `gpt-4.1` and `claude-sonnet-5.5`, per-case disagreements and which models contributed. Agreement is not correctness; unanimous errors need approved labels. A P(yes)-only Jev lead run is excluded from lead agreement until a dev threshold is frozen.
6. **Lead Qualifier.** YES/NO/undecided counts, YES precision/recall/F1, FPR/FNR, native P(yes), frozen dev threshold (max F1 on DEV only), held-out TEST result with that threshold unchanged, sessions the builder flagged for ambiguity or unresolved rules.
7. **Live Latency & Cost.** P50/P95/P99 end-to-end latency (P95/P99 flagged under 20 requests), provider-reported model latency only where supplied, tokens, tokens per second (known completion tokens over the provider-reported interval if given, otherwise the client round trip, labelled), timeouts, 429s, 5xx, retries, cost per request and per 1,000 cases for ACTUAL and ESTIMATED separately, per model and per provider. Cost per correct prediction needs labels.
8. **Robustness & Readiness.** Per participant: route, readiness, blockers, prompt fingerprints, dataset and input-contract versions, run ids, serving provider, invalid and retried requests, repeatability across runs, history-completeness slices (need labels), and the data-policy, parity and session-policy status. There is no reviewed language field in the prepared data, so language slices are reported as UNAVAILABLE.
9. **Dataset Review & Labels.** Selected preparation, counts, approved-label status, protected annotations folder, unselected session policy, the unvalidated challenge pool and missing parity. The label-approval workflow itself is not built.
10-12. Budget & Spend, Traces & Observability, Settings & Help (existing screens, kept). Evaluation Runs (generation/RAG benchmark), Project Health and Dataset (JSONL) Validation stay reachable as entries 14-16 and by `:eval`, `:health`, `:validate`. The knowledge-base audit remains `:audit`.

## Trace Explorer & Export (section 13, `:export`)

Preview and export stored trace records (prediction records joined with the request-event log and approved labels; no new store). Keys in the screen: `m` model, `t` task, `u` run, `s` status, `e` error class, `d` data kind (synthetic fixture vs real-data benchmark), `f` format (JSONL, CSV, ZIP), `c` clear filters, `x` export. The table shows a dash for anything not measured (the inspector lists why, for example "not reported by provider"); errors and invalid outputs are never shown as predictions; a per-run summary lists planned, attempted, succeeded, failed, invalid and unattempted counts; synthetic fixture traces are labelled SYNTHETIC and never look like real data. The same export is available as `msu-llm-lab export-traces`.

- Scopes: all models and runs, one model, one run, one task, or a filtered selection. Files go to `reports/exports/` (git-ignored); the exact path, counts (exported, skipped, redacted, missing telemetry) and warnings are shown. Existing files are never overwritten (`-r2` suffix); identical inputs and `exported_at` give identical bytes.
- JSONL = full nested records (`trace-export-v1`); CSV = flattened with empty cells for null; ZIP = `manifest.json`, `all_records.*`, `models/<model>.jsonl|csv`, `consolidated_comparison.csv`, `participants.csv`. JSONL/CSV get a `<file>.manifest.json` sidecar.
- Manifest: run ids, model ids, provider routes, tasks, dataset/prompt/input-contract versions, parity status, export time, telemetry provenance, counts, privacy statement, file hashes, warnings, and **every roster participant** with its execution state (`EXECUTED`, `EXECUTED_WITH_FAILURES`, `EXECUTED_ALL_FAILED_OR_INVALID`, `NOT_EXECUTED`, `NOT_EXECUTED_BLOCKED`). Models that never ran have no trace rows.
- Privacy: free-text fields are redacted (email, ID, phone, introduced names), secret-looking strings and keys are scrubbed, `user4` is dropped, raw model output is withheld unless a complete data-owner authorization is supplied through the API (the TUI never offers it). Simulated events are excluded and counted. Ground truth and correctness are filled only for human-approved labels.

## Keys

`1`-`9`, `0`, `-`, `=` jump to sections 1-12; `[` `]` step through all 15; arrows, `j`/`k`, `Home`/`End`, `PgUp`/`PgDn` move; `Enter` inspects; `/` filters the roster; `:` opens commands (`:overview`, `:roster`, `:quality`, `:calibration`, `:agreement`, `:lead`, `:latency`, `:robustness`, `:labels`, `:budget`, `:traces`, `:settings`, `:eval`, `:health`, `:validate`, `:audit`); `r` refreshes; `q` quits. On wide terminals each table has a side inspector with the full detail of the selected row.

## Local resources

CPU, memory and disk come from `psutil` when installed (`pip install "msu-llm-lab[monitor]"`). Without it, CPU is the 1-minute load average per core (labelled LOAD/core), memory is UNAVAILABLE, and disk comes from the standard library. Histories start empty and each sample is timestamped.

## Known gaps

- No pairwise agreement with human reviewers, no label-approval screen, no dataset browser (by design: raw conversations are never shown).
- Dify parity is unknown; hardware-sizing and serving simulations were removed with the simulation features.
- The generation/RAG benchmarks do not emit request events yet (only the classification runner does).
- Retry events are reported when a request ends, not at the instant of each retry.
