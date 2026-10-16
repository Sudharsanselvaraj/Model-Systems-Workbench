# Labelled classifier evaluation (labeled-eval-v1)

Offline preparation, a guarded executor and a scorer for the expected-label test suites (Main Intent 120 cases, Lead Qualifier 200 cases; eight participants per task).

```bash
uv run msu-llm-lab labeled-eval prepare              # audit, frozen splits, answer-key separation, decisions, plan, gate report (offline)
uv run msu-llm-lab labeled-eval gates --save-plan    # show the plan and every gate; exit 2 while any blocking gate fails
uv run msu-llm-lab labeled-eval run --operator NAME  # refuses before any typed phrase if a gate fails; --live adds the two typed approvals
uv run msu-llm-lab labeled-eval score <prediction.jsonl ...>   # evaluator only, after inference
```

- Inputs stay in `Intent:lead dataset/` (git-ignored, read-only). Label-free inference inputs and the answer key (mode 0600) live under `data/labeled_eval/`; reports go to `reports/labeled_eval/labeled-eval-v1/`.
- Both datasets are classified CONFIDENTIAL (potentially real): their text matches prepared real-conversation messages and the origin is unverified. They are never described as synthetic or anonymised.
- Lead input is `action_summary` only; `user_msg` is never sent. Output caps stay 16 (INTENT) and 24 (LEAD) tokens. Mistral runs last. No cap is raised: the effective policy must equal the recorded one.
- The executor dispatches only when every blocking gate passes (operator approval record, a verified data-processing permission record bound to the dataset hashes, models and routes, separation checks, caps, unresolved spend, headroom minus reserve, registry readiness) and the typed live approval and typed real-data authorisation (bound to the plan fingerprint) were issued in the same process.
- Invalid or missing outputs count as misses in the end-to-end view; a conditional view over valid outputs is reported next to it. Empty cells mean not measured, never zero.
- Jev threshold selection uses DEV scores only and is frozen before TEST; it needs a continuous score for these cases.

## Jev prompt experiment

`labeled-eval jev-experiment gates|record-approval|record-permission|run|score` runs the pre-registered V0/V1/V2 comparison on Main Intent (`reports/labeled_eval/labeled-eval-v1/jev_prompt_experiment/PREREGISTRATION.md`). It has its own plan fingerprint, approval and permission records, and typed authorisation; it refuses to run unless every blocking gate passes, including that each variant changes what Jev receives (Jev's payload builder drops unmapped prompt sections). Selection uses the dev half only; the held-out half is too small to confirm gains on the rare classes.
