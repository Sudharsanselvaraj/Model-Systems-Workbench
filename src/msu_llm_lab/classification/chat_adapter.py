"""Chat Completions adapter for classification (GPT-4.1 Nano via OpenRouter). Plain text output, strict validation.

Intent: single plain-text label validated by the authoritative intent validator. Lead: the existing raw-JSON contract
validated by the authoritative lead validator. No JSON-schema constrained output is used for the initial comparison.
Provider routing comes from the registry route (pinned, no silent fallback) unless explicitly overridden. Execution
stays blocked while the registry route is BLOCKED, because the underlying adapter checks the registry first.
"""

from typing import Any, Dict, Optional

from msu_llm_lab.classification.contracts import IntentLabel
from msu_llm_lab.classification.records import AdapterResult, Task
from msu_llm_lab.classification.validators import validate_intent_output, validate_lead_output
from msu_llm_lab.models import registry
from msu_llm_lab.models.registry import Protocol
from msu_llm_lab.providers.base import LiveNotAuthorizedError, ProviderError, ProviderRequest
from msu_llm_lab.providers.openrouter import OpenRouterAdapter

NANO_MODEL_ID = "openai/gpt-4.1-nano"
MAX_OUTPUT_TOKENS = {Task.INTENT: 16, Task.LEAD: 24}  # small caps: labels and the one-key JSON are tiny; also bounds spend


class ChatClassificationAdapter:
    protocol = Protocol.CHAT_COMPLETIONS

    def __init__(
        self,
        model_id: str = NANO_MODEL_ID,
        provider: Optional[OpenRouterAdapter] = None,
        provider_preferences: Optional[Dict[str, Any]] = None,
        use_registry_route: bool = True,
    ):
        self.model_id = model_id
        self.provider = provider or OpenRouterAdapter()
        if provider_preferences is None and use_registry_route:
            rec = registry.record_for_model_id(model_id)
            provider_preferences = rec.route.provider_preferences if rec else None
        self.provider_preferences = provider_preferences

    def supports(self, task: Task) -> bool:
        return task in MAX_OUTPUT_TOKENS

    def max_output_tokens(self, task: Task) -> int:
        return MAX_OUTPUT_TOKENS[task]

    def build_request(self, task: Task, state_text: str, prompt_text: str, live_authorized: bool) -> ProviderRequest:
        rec = registry.record_for_model_id(self.model_id)
        pinned = rec.pinned_route() if rec else None
        temperature = 0.0 if (pinned is None or pinned.supports("temperature")) else None  # never send what the route does not list
        return ProviderRequest(
            model_id=self.model_id,
            messages=[{"role": "system", "content": prompt_text}, {"role": "user", "content": state_text}],
            temperature=temperature, max_tokens=MAX_OUTPUT_TOKENS[task], live_authorized=live_authorized,
            provider_preferences=self.provider_preferences, extra_fields=(dict(rec.request_overrides) if rec and rec.request_overrides else None),
        )

    def classify(self, task: Task, state_text: str, prompt_text: str, *, live_authorized: bool) -> AdapterResult:
        if not live_authorized:
            raise LiveNotAuthorizedError("Chat classification request refused: no explicit live authorization.")
        if not self.supports(task):
            raise ProviderError(f"Chat adapter does not support task {task}")
        resp = self.provider.generate(self.build_request(task, state_text, prompt_text, live_authorized))
        issues, warnings = [], []
        label = need = None
        if task == Task.INTENT:
            v = validate_intent_output(resp.output_text)
            label = v.label if v.valid else None
            if not v.valid:
                issues.append(v.reason.value)
        else:
            v = validate_lead_output(resp.output_text)
            need = v.need_cta if v.valid else None
            if not v.valid:
                issues.append(v.reason.value)
        if resp.resolved_model_id is None:
            warnings.append("RESOLVED_MODEL_MISSING")
        if resp.serving_provider is None:
            warnings.append("PROVIDER_MISSING")
        if resp.usage_source == "UNAVAILABLE":
            warnings.append("USAGE_MISSING")
        return AdapterResult(
            ok=not issues, task=task, protocol=self.protocol, requested_model_id=self.model_id, issues=issues, warnings=warnings,
            resolved_model_id=resp.resolved_model_id, serving_provider=resp.serving_provider,
            predicted_label=label, predicted_need_cta=need, raw_output=resp.output_text,
            requested_route=self.provider_preferences, confidence_source="NONE", prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            usage_source=resp.usage_source, latency_ms=resp.latency_ms, attempt_latency_ms=resp.attempt_latency_ms,
            model_latency_ms=resp.model_latency_ms if resp.model_latency_source == "PROVIDER_REPORTED" else None,
            attempts=resp.attempts, retry_errors=list(resp.retry_errors), native_cost=resp.native_cost, native_cost_unit=resp.native_cost_unit, finish_reason=resp.finish_reason, reasoning_tokens=resp.reasoning_tokens,
            actual_cost_usd=resp.actual_cost_usd,
            estimated_cost_usd=resp.estimated_cost_usd, cost_usd=resp.cost_usd, cost_basis=resp.cost_basis,
        )
