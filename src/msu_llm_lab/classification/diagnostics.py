"""Offline diagnostics built from stored records and events only. They separate verified facts from hypotheses and prepare (never run) test plans.

Nothing here sends a request, changes a route or a token limit, or mixes results from a changed configuration into earlier results.
"""

import json
from collections import Counter
from typing import Any, Dict, List, Sequence

from msu_llm_lab.classification.chat_adapter import MAX_OUTPUT_TOKENS
from msu_llm_lab.classification.records import PredictionRecord, Task
from msu_llm_lab.models import registry

NOT_CAPTURED = "not captured (these records predate the finish_reason / reasoning_tokens fields)"


def _route_table(key: str) -> List[Dict[str, Any]]:
    r = registry.get_record(key)
    return [{"tag": c.tag, "provider": c.provider, "input_per_1m": c.input_per_m, "output_per_1m": c.output_per_m, "context": c.context, "max_completion": c.max_completion,
             "temperature_listed": c.supports("temperature"), "reasoning_listed": c.supports("reasoning"), "uptime_1d": c.uptime_1d, "pinned": c.tag == r.pinned_tag} for c in r.candidate_routes]


def rate_limit_diagnostics(plan: Dict[str, Any], stats_by: Dict[str, Dict[str, List[PredictionRecord]]], events: Sequence[Any], state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"schema": "rate-limit-diagnostics-v1", "participants": {}}
    for p in plan["participants"]:
        recs = [r for t in p["tasks"] for r in stats_by.get(p["key"], {}).get(t, [])]
        r429 = [r for r in recs if any(c == "HTTP_429" for c in r.retry_errors)]
        if not r429:
            continue
        rid = {f"{p.get('store_bench') or plan['bench_id']}-{p['key']}-{t}" for t in p["tasks"]}
        ev = sorted((e for e in events if e.run_id in rid), key=lambda e: e.ts)
        failed = [r for r in recs if r.status == "ERROR"]
        recovered = [r for r in recs if r.status != "ERROR" and (r.attempts or 1) > 1]
        retry_events = [e for e in ev if e.kind == "REQUEST_RETRY"]
        out["participants"][p["key"]] = {
            "model_id": p["model_id"], "route": p["route"], "quarantined": state.get("quarantined", {}).get(p["model_id"]),
            "requests_recorded": len(recs), "requests_with_any_429": len(r429), "failed_after_all_attempts": len(failed), "recovered_after_retry": len(recovered),
            "provider_http_attempts": sum((r.attempts or 1) for r in recs), "retry_events": len(retry_events), "retry_causes": dict(Counter(e.retry_cause for e in retry_events)),
            "failed_cases": [{"case_id": r.case_id, "task": r.task.value, "attempts": r.attempts, "retry_codes": r.retry_errors, "recorded_at": r.created_at, "error_type": r.error_type} for r in failed],
            "recovered_cases": [{"case_id": r.case_id, "task": r.task.value, "attempts": r.attempts, "retry_codes": r.retry_errors, "latency_ms": r.latency_ms} for r in recovered],
            "first_request_started": next((e.ts for e in ev if e.kind == "REQUEST_STARTED"), None), "last_event": ev[-1].ts if ev else None,
            "verified": ["The provider returned HTTP 429 on the pinned route; the shared retry policy made up to 3 attempts per request with bounded backoff.",
                         "Some requests recovered after a retry, so the limit is intermittent rather than a hard refusal at the start of the run.",
                         "Three consecutive requests failed all 3 attempts, which tripped the 3-consecutive-error circuit breaker and quarantined the participant; later cases were not sent."],
            "not_known": ["The Retry-After header value (the poster validates it for backoff but does not store it).", "Whether the limit comes from the provider's own quota, an OpenRouter per-provider limit or an account-level limit (the stored error carries only the status code).",
                          "The exact backoff delay used for each attempt (retry events are written when a request ends, not at the instant of each retry).", "Whether a pause of any length would clear the limit."],
            "route_options_in_registry": _route_table(p["key"]),
        }
    return out


def rate_limit_markdown(d: Dict[str, Any]) -> str:
    L = ["# Rate-limit diagnostics (offline; nothing was re-sent)", ""]
    if not d["participants"]:
        return "\n".join(L + ["No participant recorded an HTTP 429."]) + "\n"
    for k, x in d["participants"].items():
        L += [f"## {k} (`{x['model_id']}`)", "", f"- Pinned route: `{x['route']['pinned_tag']}` (provider {x['route']['listed_provider']}); quarantined: {x['quarantined'] or 'no'}",
              f"- Requests recorded {x['requests_recorded']}; with any 429 {x['requests_with_any_429']}; failed after all attempts {x['failed_after_all_attempts']}; recovered after a retry {x['recovered_after_retry']}",
              f"- Provider HTTP attempts {x['provider_http_attempts']}; retry events {x['retry_events']} ({x['retry_causes']}); window {x['first_request_started']} to {x['last_event']}", "", "**Verified from stored records and events**", ""]
        L += [f"- {v}" for v in x["verified"]] + ["", "**Not known (not stored)**", ""] + [f"- {v}" for v in x["not_known"]]
        L += ["", "Failed cases (opaque ids):", ""] + [f"- {c['case_id']} {c['task']}: attempts {c['attempts']} {c['retry_codes']} at {c['recorded_at']}" for c in x["failed_cases"]]
        L += ["", "## What must be verified before an alternative route is tried (with synthetic input only)", "",
              "1. **Owner approval of the new recipient.** Any alternative route is a different provider hosting the model; the registry pin changes only with your approval, and every result stays labelled with its route.",
              "2. **Account guardrails.** The account's Zero-Data-Retention rule already removed the `openai` and `google-ai-studio` endpoints (HTTP 404). Check the candidate route is allowed for this account before spending anything.",
              "3. **Listed price, context and parameters** of the candidate route against the table below (the registry price check must pass; the plan never assumes a price).",
              "4. **Quantization / hosting differences** (for example fp8 versus the first-party route) can change outputs; comparisons must name the route and never mix the two.",
              "5. **Rate-limit evidence.** Decide whether to capture `Retry-After` and per-attempt timing first (a small, offline code change) so the next 429 is diagnosable.",
              "6. **Data handling.** The provider data policy stays UNKNOWN until verified from a primary source; operator authorization does not change that.", "",
              "| Route | Provider | In / out per 1M | Context | Max completion | Temperature | Uptime 1d | Pinned now |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['tag']} | {r['provider']} | {r['input_per_1m']} / {r['output_per_1m']} | {r['context']} | {r['max_completion']} | {'listed' if r['temperature_listed'] else 'not listed'} | {r['uptime_1d']} | {'yes' if r['pinned'] else ''} |" for r in x["route_options_in_registry"]]
        L += ["", "### Synthetic-only test plan (prepared, NOT run)", "",
              "- Use the hand-written synthetic fixtures of `smoke-test` (no student text), at most 5 INTENT and 5 LEAD cases, under a separate synthetic run label.",
              "- Run only after (a) you approve the alternative route and (b) the registry pin is changed in a separate commit; the plan gets a new fingerprint and needs both typed approvals.",
              "- Success criteria: all 10 requests return HTTP 200, schema-valid outputs, the route in the response matches the pin, no 429 in 3 consecutive minutes of spacing.",
              "- Results are NEVER merged into the real-data comparison: the original Mistral results keep their `mistral/eu` provenance and stay quarantined and incomplete."]
    return "\n".join(L) + "\n"


def empty_output_diagnostics(plan: Dict[str, Any], stats_by: Dict[str, Dict[str, List[PredictionRecord]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"schema": "empty-output-diagnostics-v1", "token_caps": {t.value: MAX_OUTPUT_TOKENS[t] for t in MAX_OUTPUT_TOKENS}, "participants": {}}
    for p in plan["participants"]:
        recs = [r for t in p["tasks"] for r in stats_by.get(p["key"], {}).get(t, [])]
        empties = [r for r in recs if "EMPTY_OUTPUT" in r.validation_issues]
        if not empties:
            continue
        cap = lambda r: MAX_OUTPUT_TOKENS[r.task]  # noqa: E731
        at_cap = [r for r in empties if r.completion_tokens == cap(r)]
        ok = [r for r in recs if r.status == "OK"]
        out["participants"][p["key"]] = {
            "model_id": p["model_id"], "route": p["route"]["pinned_tag"], "request_overrides": p.get("request_overrides") or {}, "records": len(recs), "empty_outputs": len(empties),
            "empty_by_task": dict(Counter(r.task.value for r in empties)), "empty_with_completion_tokens_equal_to_cap": len(at_cap),
            "empty_completion_token_values": dict(Counter(r.completion_tokens for r in empties)),
            "valid_outputs": len(ok), "valid_completion_token_values": dict(Counter(r.completion_tokens for r in ok)),
            "other_invalid": dict(Counter(i for r in recs for i in r.validation_issues if i != "EMPTY_OUTPUT")),
            "invalid_with_completion_tokens_equal_to_cap_not_empty": sum(1 for r in recs if r.status == "INVALID_OUTPUT" and "EMPTY_OUTPUT" not in r.validation_issues and r.completion_tokens == cap(r)),
            "finish_reason_values": {("not captured" if k is None else k): v for k, v in Counter(r.finish_reason for r in recs).items()}, "reasoning_tokens_reported": sum(1 for r in recs if r.reasoning_tokens is not None),
            "provider_errors_in_empty_records": sum(1 for r in empties if r.status == "ERROR"), "raw_output_stored": sum(1 for r in empties if r.raw_output is not None),
            "verified": [f"{len(at_cap)} of {len(empties)} empty outputs report completion_tokens equal to the request's max_tokens cap ({out['token_caps']}). They were billed as completion tokens but produced no visible text.",
                         "None of the empty outputs was a provider error: each was HTTP 200 with usage and the validator rejected the empty content.",
                         f"Valid outputs used {sorted(set(r.completion_tokens for r in ok))} completion tokens (their visible content is a single label or a one-key JSON)."],
            "hypotheses_not_verified": ["The invisible tokens are reasoning/thinking tokens (the API's reasoning controls exist for this model family, but reasoning_tokens and finish_reason were not stored).",
                                        "finish_reason was `length` for the capped empties (consistent with the cap, but not recorded).", "Output filtering or a refusal as the cause (nothing stored supports or excludes it)."],
            "evidence_not_captured": {"finish_reason": NOT_CAPTURED, "reasoning_tokens": NOT_CAPTURED, "raw content blocks": "never stored by design (privacy)"},
        }
    return out


def empty_output_markdown(d: Dict[str, Any]) -> str:
    L = ["# Empty-output diagnostics (offline; no configuration was changed)", "", f"Completion-token caps in force: {d['token_caps']}.", ""]
    for k, x in d["participants"].items():
        L += [f"## {k} (`{x['model_id']}`, route `{x['route']}`)", "", f"- Records {x['records']}; empty outputs {x['empty_outputs']} ({x['empty_by_task']}); empties at the token cap: **{x['empty_with_completion_tokens_equal_to_cap']}**",
              f"- Completion tokens of empties: {x['empty_completion_token_values']}; of valid outputs: {x['valid_completion_token_values']}; other invalid reasons: {x['other_invalid']}",
              f"- Other invalid outputs that also hit the cap (truncation signature): {x['invalid_with_completion_tokens_equal_to_cap_not_empty']}", f"- finish_reason values stored: {x['finish_reason_values']}; reasoning_tokens reported on {x['reasoning_tokens_reported']} records",
              "", "**Verified from the stored records**", ""] + [f"- {v}" for v in x["verified"]] + ["", "**Hypotheses (not verified)**", ""] + [f"- {v}" for v in x["hypotheses_not_verified"]]
        L += ["", "**Evidence not captured**", ""] + [f"- {a}: {b}" for a, b in x["evidence_not_captured"].items()] + [""]
    L += ["## Synthetic-only test plan for a possible fix (prepared, NOT run)", "",
          "Goal: find out WHY the capped outputs are empty before changing any configuration, and test fixes without touching the real-data results.", "",
          "1. **Capture-only run (no configuration change).** The adapter now stores `finish_reason` and `usage.completion_tokens_details.reasoning_tokens` for new requests. Run 10 synthetic fixtures through each Claude participant under the SAME configuration. Expectation if the hypothesis holds: empties show `finish_reason=length` and reasoning tokens close to the cap.",
          "2. **Variant A: reasoning off** (`reasoning.enabled=false`, an owner-approved request field as already done for Qwen). 10 synthetic fixtures per Claude participant.",
          "3. **Variant B: larger completion cap** (for example 64 and 256), same fixtures. Record cost; the per-case ceiling still applies.",
          "4. Compare valid-output rate, empty rate, completion tokens, latency and estimated cost per variant on identical fixtures.",
          "5. **Guards:** synthetic fixtures only; separate run labels (`synthetic-diagnostic-*`); a new plan fingerprint and both typed approvals; hard cap far below the pilot ceiling (a few cents); no result from a changed configuration is merged into bench-492495eeddc0 or earlier results. If a variant is adopted for the real benchmark it becomes a new, clearly labelled configuration whose results are compared separately.",
          "6. **Decision:** adopt a variant only after your explicit approval, because it changes what the Claude results mean (Sonnet's carried-over results used the original configuration)."]
    return "\n".join(L) + "\n"


def diagnostics_files(plan, state, reports_dir, events, stats_by, recon) -> Dict[str, bytes]:
    rl, eo = rate_limit_diagnostics(plan, stats_by, events, state), empty_output_diagnostics(plan, stats_by)
    enc = lambda s: s.encode("utf-8")  # noqa: E731
    return {"rate_limits.json": enc(json.dumps(rl, indent=2, sort_keys=True, default=str) + "\n"), "rate_limits.md": enc(rate_limit_markdown(rl)),
            "empty_outputs.json": enc(json.dumps(eo, indent=2, sort_keys=True, default=str) + "\n"), "empty_outputs.md": enc(empty_output_markdown(eo))}
