# Authorized real-data benchmark

For the operator-authorized benchmark of all ten participants on the prepared admissions dataset. Nothing here sends a request until **two typed approvals** are given in an interactive terminal.

## Flow

```bash
# 1. Prepare a NEW dataset version with a private pseudonymization secret (typed twice, hidden; never stored or printed)
uv run msu-llm-lab prepare-classification-data --out-dir data/classification_prep/prep-v2 --prompt-salt

# 2. Build and store the plan (plan only, no provider call). The session policy is YOUR choice: 2h or 30m.
uv run msu-llm-lab real-benchmark plan --prep-dir data/classification_prep/prep-v2 --session-policy 2h

# 3. Execute (or resume). Needs the live approval phrase AND the real-data authorization phrase for this exact plan.
uv run msu-llm-lab real-benchmark run --prep-dir data/classification_prep/prep-v2 --session-policy 2h --operator "Your Name" --live

# --max-blocks N means "N blocks in TOTAL from the start of the plan" (the plan has hundreds of blocks; 1 block = 10 cases per task for every participant).
# The first 3 blocks (30 cases per task) are the part that fits the $2 pilot ceiling even at worst-case cost; run with --max-blocks 3, or omit it to continue
# block by block while the recorded cost still leaves room under the caps. Re-running resumes; completed requests are never repeated.

# 4. Status, report, comparison bundle and redacted trace export (read-only)
uv run msu-llm-lab real-benchmark status
uv run msu-llm-lab real-benchmark report bench-<id> --prep-dir data/classification_prep/prep-v2
```

## What the plan states

Dataset version, session policy, source-file hash, manifest hash; eligible and excluded cases per task with reasons; prompt hashes (full SHA-256 and the 12-character version used in records); evaluation contract version; per participant the exact model id, protocol, pinned route and listed provider, sourced prices and their status, schema-fact status, data-handling facts (policy status stays `UNKNOWN` unless a primary source verified it), which native fields it can return, eligible cases per task, cases excluded by the $0.02 per-case cap, and the pessimistic upper bound; request counts; timeout (30 s), at most 3 attempts per request, state-size and output-token limits; all caps; recipients and data categories; a fingerprint derived from all of it. Prices come from the registry, never from constants, and a different prompt, model set, block size, policy or ceiling gives a different fingerprint.

## Budgets are never raised

The pilot ($2.00), total ($10.00), per-run ($0.50) and per-case ($0.02) ceilings stay as configured. When the full dataset does not fit, the plan is a deterministic, checkpointed schedule:

- eligible cases of each task are ordered and cut into blocks. Default `--sampling stratified`: the first case of EVERY pattern bucket comes first (variety), then everyone else in `SHA-256(seed|case id)` order (proportional). Buckets are preparation metadata (view, history completeness, selection hints, builder tier, ambiguity), not labels; `--sampling hash` gives the plain hash order. The sampling is part of the plan fingerprint;
- a block runs **every** participant on the **same** cases, so models stay comparable;
- a block is dispatched only while recorded guard cost + that block's pessimistic bound fits the hard cap and the ledger ceilings; unresolved spend halts the run;
- progress is checkpointed after every (block, participant, task) unit; re-running the same command resumes without repeating a completed request (the prediction store refuses duplicates);
- a run that stops early is reported as **PARTIAL** with its exact coverage (for example "40 of 2,208 INTENT cases"), never as a full benchmark.

## Authorization

`real-benchmark run --live` needs (1) the existing `RUN LIVE <models> PLAN <fingerprint>` phrase and (2) `AUTHORIZE REAL DATA <dataset version> PLAN <fingerprint>` after the recipients, routes, provider-policy status and data categories are shown. The resulting authorization is bound to dataset version, session policy, model set and routes, task scope, plan fingerprint, cost ceiling and the fact that the records are real admissions-derived data. It is created only by the typed confirmation in the current process (a hand-built or loaded object is refused) and is written to the versioned run manifest `data/benchmarks/<bench id>/manifest.json`, which also records prompt hashes, caps, code fingerprint and every resume session. **It does not verify any provider's retention, training or processing policy**, and the policy status stays as the registry has it.

Never sent: pseudonymous group or trace ids, timestamps, `user4`, excluded records, API keys, anything outside the reviewed dataset and task scope. Sent: redacted student messages (INTENT: current message and prior turns of the same session, plus prior assistant replies as exported) and the authoritative prompt; for LEAD, the rule-built action summary and prompt. Redaction is pattern-based without named-entity recognition and pseudonymization is not anonymization.

## Failure handling

Each participant has the existing circuit breaker (3 consecutive errors or a systemic 401/402/403 stops it) plus a cross-block breaker. A failing participant is **quarantined** with its reason, the others continue, and the report lists it as `QUARANTINED_PARTIAL` or `NOT_EXECUTED_QUARANTINED`. Diagnose before resuming. An error is never a prediction and a malformed output is never a valid result; both stay in reliability counts.

## Results and honesty rules

`real-benchmark report` writes `reports/real_benchmark_<id>.md/.json`, a NEW (never overwritten) folder `reports/real_benchmark_<id>/` with `model_results.csv` (one row per case/model), `model_summary.csv` (per model and task: validity, failures, latency percentiles gated by sample size, tokens, actual / estimated / native cost, completion state), `model_comparison.csv` (side-by-side valid predictions; unusable outputs are counted apart), `model_agreement_pairs.csv`, `unattempted_in_executed_blocks.csv` (with `--prep-dir` and `--session-policy`; only new participants' cases inside the blocks actually executed), `request_reconciliation.json` (planned, dispatched, outcomes, HTTP attempts, retries, unattempted, carried-over; every identity checked), `diagnostics/` (rate-limit and empty-output analyses with verified facts separated from hypotheses, plus prepared synthetic-only test plans), `models/<participant>.md`, `comparison_report.md`, `observability_report.md`, `traces.jsonl` and `benchmark_manifest.json` (hashes, authorization sessions, caveats), and a redacted ZIP trace export. Completion states: `COMPLETE_FOR_PLANNED_SCOPE` (every planned request attempted), `PARTIAL`, `QUARANTINED_PARTIAL`, `NOT_EXECUTED`, `NOT_EXECUTED_QUARANTINED`, `BLOCKED`. All ten participants always appear. End-to-end latency is client-measured; provider/model latency appears only if reported; native cost is in credits (USD ratio unverified, never converted); estimated USD is a catalogue price times reported tokens. Agreement uses valid common outputs only and is not correctness. Accuracy, macro-F1, per-class metrics, confusion, Brier and ECE stay `GROUND TRUTH PENDING` until approved labels exist.

## Ground truth workflow

```bash
uv run msu-llm-lab annotate queue  --prep-dir ... --session-policy 2h --task INTENT        # redacted review queue (local)
uv run msu-llm-lab annotate submit --prep-dir ... --session-policy 2h --task INTENT --reviewer alice --file alice.jsonl
uv run msu-llm-lab annotate status --prep-dir ... --session-policy 2h
uv run msu-llm-lab annotate adjudicate --prep-dir ... --task INTENT --case I-2h-R0001 --label QUERY --adjudicator dave
uv run msu-llm-lab annotate approve --prep-dir ... --approver carol --all-ready            # typed phrase required
uv run msu-llm-lab annotate freeze  --prep-dir ... --by carol
```

Two distinct reviewers label independently; agreement makes a case ready, disagreement needs a third reviewer; approval is explicit and named (bulk approval needs a typed phrase); revocation keeps history; a freeze records the SHA-256 of the approved set. Thresholds are selected on DEV only and applied unchanged to TEST. Nothing is approved automatically and predictions are never inputs.


## Bundle semantics worth knowing

- `origin` marks each row `NEW_THIS_PLAN` or `CARRIED_OVER_FROM:<bench>`; carried-over participants are never listed as unattempted.
- **Cost columns:** an empty cell means not measured, never zero. `requests_without_cost_evidence`, `errors_no_response` (an error returned nothing to meter; that is not proof it cost nothing) and `unknown_cost_after_response` are separate; unresolved ledger entries are counted separately in the manifest.
- **Jev LEAD** results are probabilities until a threshold is frozen from DEV: `binary_decision_available` is False and `probability_only` True, and they are excluded from YES/NO agreement.
- **Privacy block:** `privacy.export` describes the files (no raw student text, this bundle was not sent anywhere); `privacy.inference_payloads` records that redacted admissions-derived inputs WERE sent to the listed recipients under the operator's authorization (provider policies UNKNOWN).
- `finish_reason` and `reasoning_tokens` are stored for new requests (null for records made before they were captured).
- `real-benchmark verify <bundle> --prep-dir ... --session-policy ...` re-checks hashes, row counts, retained predictions, cost provenance, unattempted scope and that the narrative report agrees with the CSV files.
