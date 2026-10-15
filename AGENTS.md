# AGENTS.md — MSU LLM Workbench

Guide for AI coding agents (and humans) working in this repository. Read it fully before changing anything. It describes what the project is, how the folders fit together, the rules that must never be broken, and how to verify your work.

## 1. What this project is

A local-first research workbench that:

1. audits MSU admissions knowledge bases (KBs) deterministically;
2. prepares **classification datasets** locally from a real WhatsApp conversation export (redacted, session-reconstructed, leakage-safe; **every label empty until a human approves it**);
3. compares **10 model participants** on two classification tasks, on identical inputs, under hard spend caps:
   - **Main Intent**: `GREETING`, `OUT_OF_SCOPE`, `UPLOAD_STATUS_CHECK`, `HUMAN_HANDOFF_REQUEST`, `QUERY`
   - **Lead Qualifier**: `{"need_cta": true|false}`
4. observes everything through a read-only terminal dashboard and exports redacted trace records.

Status: alpha. **No live provider request has ever been made.** `PRODUCTION_PARITY_NOT_ESTABLISHED` (the Dify workflow export is missing). No provider data policy is verified, so **student-derived text may not be sent to any provider.**

Python 3.11+, `uv`, Typer (CLI), Rich (TUI), Pydantic v2, SQLite (WAL), pytest.

## 2. Non-negotiable rules

Break none of these. If a task seems to require it, stop and ask the user.

1. **No simulation, no mock data, no fabricated values.** There is no dry-run mode (removed on purpose). Missing tokens, latency, confidence, cost or telemetry stay `null`/`UNAVAILABLE`. A prediction is never ground truth. A historical chatbot answer is never a label. Never create placeholder trace rows for models that did not run.
2. **No network call without explicit user approval of an exact plan.** Live requests need an interactive, typed approval phrase that includes the plan fingerprint (`core/live_guard.py`). Never call a model API, spend credits, configure or print an API key, or transmit student data while implementing or testing. Tests run offline (fake transports; sockets blocked).
3. **Student data stays local unless the operator authorizes a specific benchmark.** `PUBLIC` and `SYNTHETIC` cases may be dispatched; `CONFIDENTIAL` real records only through `real-benchmark run`, which needs the live approval phrase AND the `AUTHORIZE REAL DATA <version> PLAN <fingerprint>` phrase, binds to the dataset version, models, routes, tasks, plan fingerprint and ceiling, and never marks any provider policy VERIFIED (`classification/realdata.py`, `docs/REAL_BENCHMARK.md`). Do not open, print, summarise or commit `september_to_october.csv` or anything derived from it. Do not display raw conversations in the TUI, logs or exports. `user4` (first digits of phone numbers) is never read, hashed, grouped on, exported or shown.
4. **Secrets never enter the repo, logs, events or exports.** The API key comes only from the user's own shell (`OPENROUTER_API_KEY`). Never ask for it in chat.
5. **Budget guards stay effective.** Local caps: $10 total, $2 pilot, $0.50 per run, $0.02 per case. Unresolved-spend entries block live runs until `budget reconcile`. Spend kinds stay separate: ACTUAL (provider-reported), ESTIMATED (catalogue price × reported tokens), UNRESOLVED, legacy SIMULATED (old rows only, never counted). Provider `usage.cost` is "credits" of unverified USD ratio: kept as `native_cost`, not booked as USD.
6. **Ground truth only after human approval.** Supervised metrics (accuracy, F1, calibration, unanimous errors, held-out thresholds) are computed only from `annotations/approved_labels.jsonl` rows with `status: APPROVED` and a named `approved_by`. Otherwise show `GROUND TRUTH PENDING`.
7. **Never tune on the held-out split.** The Lead threshold is chosen on DEV only (max F1), frozen, then applied unchanged to TEST.
8. **Agreement is not correctness.** Never present model–model agreement as accuracy.
9. **Source data is read-only.** KBs, the conversation export, prediction stores, annotations and the preparation folder are never modified by dashboards, exports or refreshes. Preparation refuses to overwrite a non-empty output folder or any folder with an `annotations/` directory (`--new-version` writes a sibling).
10. **Do not change the authoritative prompts** (`main_intent_classifier.md`, `lead_qualification_prompt.md`) or anything about production Dify without explicit instruction.
11. **Do not make business decisions silently.** Open decisions (session policy 30 min vs 2 h, lead unit per session vs per turn, rules L4–L8, intent rule O1, provider route pins, data-policy verification, Stage 3 cap) belong to the user. Surface them; do not pick.
12. **Be honest in reports.** State exactly what was run and what failed. Say what you did not verify.

## 3. Git and workflow conventions (this user's preferences)

- Commit **one file per commit**, each with its own descriptive message, using the configured git identity (`Sudharsanselvaraj`).
- **No Claude co-author line, no "Generated with Claude" trailer, no attribution** in commits or PRs, even if tooling suggests it.
- Commit or push **only when asked**. Never commit git-ignored files (below). Never force-push.
- Preserve uncommitted work and unrelated functionality. Inspect before modifying; prefer extending existing modules to adding parallel ones.
- Keep the visual identity of the TUI (dark slate/navy background, muted purple selection, restrained accents). No emojis in UI text.
- Match the surrounding code: comment density, naming, idiom.

## 4. Repository map

```text
.
├── AGENTS.md                     # this file
├── README.md                     # user-facing overview, commands, safety model
├── LICENSE                       # Apache 2.0
├── pyproject.toml / uv.lock      # deps; extras: dev (pytest), monitor (psutil)
├── assets/                       # logo and README screenshots
├── docs/                         # design and status documents (section 9)
├── src/msu_llm_lab/              # the package (section 5)
├── tests/                        # offline test suite (section 8)
├── data/                         # LOCAL, git-ignored except data/evaluation/
├── reports/                      # LOCAL, git-ignored generated reports and exports
└── (git-ignored confidential files at the root: see section 6)
```

### 5. Package layout (`src/msu_llm_lab/`)

| Folder / file | Responsibility | Key facts |
|---|---|---|
| `cli.py` | Typer entry point and launcher loop | Commands: `smoke-test`, `api-preflight`, `prepare-classification-data`, `classification-data-status`, `export-traces`, `models roster/list`, `budget status/reconcile/purge-simulated`, `audit-kb`, `remediate-kb`, `validate-dataset`, `benchmark --live`, `benchmark-rag --live`, `compare-runs`, `import-traces`, `ui`. Bare `msu-llm-lab` opens the launcher, which returns to the menu after each workflow. |
| `core/` | `config.py` (settings, `BudgetPolicy`, paths), `budget.py` (`BudgetService`, status, preflight, unresolved spend), `live_guard.py` (typed approval, `LiveAuthorization`) | Tests monkeypatch `settings.data_dir`/`reports_dir`; never hardcode `data/`. |
| `storage/db.py` | SQLite WAL ledger + evaluation tables | Ledger columns `is_simulated`, `is_unresolved`, `cost_basis`. Additive migrations only. |
| `providers/` | `base.py` (request/response, errors, `scrub_secrets`, usage/cost guards), `http.py` (shared bounded-retry poster), `openrouter.py` (chat adapter, live only) | Retries 408/429/5xx and in-flight-budget 402 with valid `Retry-After`; no sleep after the last attempt. |
| `models/` | `registry.py` (**authoritative** sourced participant registry: facts with status VERIFIED_PRIMARY / SECONDARY_ONLY / USER_SUPPLIED / CONFLICTING / UNKNOWN, pinned routes, readiness, student-data blockers), `catalog.py` (legacy planning catalogue) | The 10 participants: baselines `baseline-intent-gemini-3.5-flash-lite`, `baseline-lead-gpt-4.1-mini` (benchmark routes, parity unverified); candidates `jev-1.14`, `gpt-4.1-nano`, `gpt-4o-mini`, `qwen3.5-35b-a3b`, `deepseek-chat-v3.1`, `mistral-small-3.2-24b-instruct`; frontier references `gpt-4.1`, `claude-sonnet-5.5`. |
| `classification/` | Offline classification stack | `contracts` (input_contract_v1, cases, labels), `serialization`, `validators` (strict outputs), `lead_summary` (deterministic rule builder B1–B10), `splits`, `metrics`, `threshold`, `privacy`, `jev_payloads` + `jev_adapter` (Decisions API, probabilities + separate confidence), `chat_adapter`, `records` (`AdapterResult`, `PredictionRecord`, append-only `PredictionStore`), `runner` (guard order below), `smoke` (synthetic plan/exec), `dataprep` (local cleaning, redaction, sessions, candidates; the recorded version is the output folder name), `realdata` (operator authorization + atomic json), `benchmark` (plan, deterministic blocks, checkpointed executor, quarantine), `benchmark_report`, `annotation` (two independent labels, adjudication, named approval, freeze). |
| `observability/` | Read-only telemetry, analysis, export | `events` (request lifecycle log + `EventTail`), `aggregate` (percentiles, throughput, cost kinds, time buckets), `labels` (approved-label loader), `analysis` (quality, calibration, agreement, lead threshold, repeatability), `monitor` (`LiveMonitor`: single polled subscription), `export` (trace rows, JSONL/CSV/ZIP, manifest). |
| `tui/` | Rich terminal UI | `welcome.py` (launcher), `app.py` (`WorkbenchTUI`, keys, commands), `screens.py` (nav, panels, legacy screens), `live_views.py` (experimental views, Trace Explorer, execution header), `telemetry.py` (real system samples, request view-model, log buffer), `theme.py`, `input.py`, `health.py`, `banner.py`. |
| `audit/`, `remediation/` | KB parser/validator/reporter; SHA-256-guarded clean-copy engine | Source KBs are hashed before/after and never modified. |
| `dataset/`, `schemas/` | JSONL dataset manager + leakage detector; Pydantic models | `DataClassification`, `DatasetSplit`. |
| `evaluation/`, `rag/` | Generation benchmark and BM25 retrieval benchmark | Live-only; need typed approval; retriever over the audited KB is required. |
| `traces/` | Langfuse export importer and failure classifier | Works on redacted exports. |

**Runner guard order** (`classification/runner.py`; do not reorder): authorization → whole-batch privacy gate → registry/budget/size preflight → send → strict validation → store → book spend (unresolved spend recorded if cost unknown). Lifecycle events are appended around each request; an event-write failure is counted and never changes a run.

**One telemetry source.** `data/telemetry/request_events.jsonl` (events), prediction JSONL under `reports/smoke/` and `reports/classification/` (records), the SQLite ledger (spend), `data/classification_prep/<version>/` (dataset facts and annotations). All dashboard views and exports derive from these; do not invent a parallel store.

## 6. Git-ignored and confidential (never read aloud, print, commit or upload)

`september_to_october.csv`, `main_intent_classifier.md`, `lead_qualification_prompt.md`, `Copy of MSU *.md` (KBs), `data/*` (except `data/evaluation/`), `reports/`, `*.db`, `.env*`, keys. `data/classification_prep/prep-v1/` contains redacted student text; `data/workbench.db` is the real ledger. Tests never touch them (the autouse fixture in `tests/conftest.py` isolates paths); if a test needs a file like these it must create a synthetic one in `tmp_path`.

## 7. Dashboard and exports (how to extend safely)

- Views are pure functions of a `LiveContext` returning a `ViewModel` (summary lines, selectable rows, inspector detail). Add a view in `tui/live_views.py`, register it in `VIEWS`, add its key to `NAV_KEYS`/`NAV_ITEMS` in `tui/screens.py` (the two lists must stay aligned) and a command in `WorkbenchTUI.COMMAND_SECTIONS`.
- Refresh is cooperative: `WorkbenchTUI.tick()` polls `LiveMonitor` every 2 s when idle. No threads, no per-view subscriptions, no provider calls. Keep it that way.
- Provenance vocabulary is fixed: ACTUAL, ESTIMATED, UNRESOLVED, LEGACY-SIMULATED, UNAVAILABLE, `GROUND TRUTH PENDING`, `NO LIVE INFERENCE YET`. Never relabel.
- Trace exports (`observability/export.py`): schema `trace-export-v1`; redaction by default (free-text fields only: error messages, issues, warnings, raw output; identifiers and timestamps are left intact); secret and `user4` keys dropped; raw text withheld unless a complete `RawTextAuthorization` from the data owner is supplied (the TUI never offers it); atomic writes, never overwrites (`-r2` suffix); deterministic ZIPs (fixed timestamps, sorted members); manifest lists participants including never-run models. Destination: `reports/exports/`.

## 8. Testing

```bash
uv sync
uv run pytest                      # full suite, ~1 minute (pty launcher tests are the slow part)
uv run pytest tests/test_trace_export.py -q
uv run pytest tests/classification -q
```

Current state: **583 passing**. Rules for tests:

- Synthetic, hand-written fixtures only (`tests/classification/synthetic_fixtures.py`, `tests/obs_helpers.py`, `tests/live_fakes.py`, `tests/registry_helpers.py`). Test doubles live under `tests/`; nothing in `src/` may simulate a provider.
- Fake transport by replacing `urllib.request.urlopen`; sockets are blocked in classification tests. Registry blockers are patched away only in tests that need an adapter to reach the fake transport; other tests assert the real registry blocks.
- Tests that touch relative paths use `monkeypatch.chdir(tmp_path)` or the isolated `settings` paths.
- `tests/classification/` imports helpers from sibling test modules (for example `test_runner_adapters`); `tests/` top-level modules cannot import from `classification/`.
- Add or update tests with every behaviour change. Report real pass/fail numbers; never claim a run you did not make.

| Test area | Files |
|---|---|
| Classification stack | `tests/classification/*` (contracts, validators, lead summary, splits/metrics/privacy, threshold, Jev payloads, runner and adapters, runner telemetry, smoke, dataprep and hardening, roster TUI) |
| Safety and ledger | `test_env_config.py`, `test_live_safety.py`, `test_budget.py`, `test_ledger_simulated.py`, `test_unresolved_spend.py`, `test_model_registry.py`, `test_providers.py` |
| Dashboard and observability | `test_dashboard_live.py`, `test_observability_events.py`, `test_observability_analysis.py`, `test_trace_export.py`, `test_tui.py`, `test_launcher_pty.py` |
| KB, datasets, RAG, traces | `test_audit.py`, `test_remediation.py`, `test_dataset.py`, `test_schemas.py`, `test_rag_benchmark.py`, `test_traces.py` |

## 9. Documents (`docs/`)

| File | Content |
|---|---|
| `ARCHITECTURE.md`, `CURRENT_ARCHITECTURE.md` | System architecture (current one reflects the real-only design) |
| `PROJECT_MAP.md` | Folder map and test counts (update when files or counts change) |
| `IMPLEMENTATION_PLAN.md` | Phase plan (serving-simulation phase marked removed) |
| `DATASET_CLEANING.md` | How the conversation export was prepared, columns kept/dropped, redaction, sessions, hardening pass |
| `PROMPT_DECISIONS.md` | Decisions on the two authoritative prompts |
| `LIVE_DASHBOARD.md` | Telemetry sources, views, provenance words, keys, known gaps |
| `REAL_BENCHMARK.md` | Authorized real-data benchmark: flow, plan contents, budget gating, authorization, quarantine, annotation workflow |

Keep docs truthful when behaviour changes. Do not document features that do not exist.

## 10. Common commands

```bash
uv run msu-llm-lab                         # launcher (returns to menu after each workflow)
uv run msu-llm-lab ui                      # dashboard
uv run msu-llm-lab models roster           # 10 participants: readiness, routes, blockers
uv run msu-llm-lab api-preflight           # is the key set? (never prints it; sends nothing)
uv run msu-llm-lab smoke-test              # prints the exact plan; sends nothing without --live
uv run msu-llm-lab export-traces -f zip    # local redacted trace export under reports/exports/
uv run msu-llm-lab prepare-classification-data --out-dir data/classification_prep/prep-v2 --prompt-salt   # hidden private salt
uv run msu-llm-lab real-benchmark plan --prep-dir data/classification_prep/prep-v2 --session-policy 2h     # plan only
uv run msu-llm-lab annotate status --prep-dir data/classification_prep/prep-v2 --session-policy 2h
uv run msu-llm-lab budget status
uv run msu-llm-lab classification-data-status
```

Dashboard keys: `1`–`9` `0` `-` `=` jump to sections, `[` `]` step through all, arrows/`j`/`k` move, `Enter` inspects, `:` commands (`:quality`, `:calibration`, `:agreement`, `:lead`, `:latency`, `:robustness`, `:labels`, `:export`, `:budget`, `:eval`, `:health`, `:audit`), `r` refresh, `q` quit. Trace Explorer: `m` model, `t` task, `u` run, `s` status, `f` format, `c` clear, `x` export.

## 11. Known gaps (do not paper over)

- No approved labels exist yet (the annotation workflow exists in `annotate`, but nobody has used it); no dataset browser (by design).
- Only a synthetic 20-request smoke run has happened (Jev and GPT-4.1 Nano, 20/20 valid). Jev's response schema is `LIVE_OBSERVED` (not primary-verified); its `usage.cost` unit is unverified; provider data policy UNKNOWN for all.
- Dify workflow export missing; session policy, lead unit and rules L4–L8 / O1 undecided; `prep-v1` was produced with the older broader O1 rule and a public salt (not regenerated).
- Only the classification runner emits request events; generation/RAG benchmarks do not yet. Retry events are reported when a request ends.
- No language slices (no reviewed language field); no pairwise human-agreement/kappa screen; no cross-model benchmark orchestration UI.

## 12. Definition of done for a change

1. Rules in section 2 still hold (re-read items 1–9 if you touched data, spend, telemetry or exports).
2. Tests added/updated; full suite run; real result reported.
3. Docs and `PROJECT_MAP.md` counts updated if behaviour or file layout changed.
4. No git-ignored or confidential file staged; nothing committed unless asked; commits one file each, no co-author.
5. The final report lists what was done, what was not verified, and any blocked feature.
