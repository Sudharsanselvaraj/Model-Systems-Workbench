"""OpenRouter provider adapter with fail-closed confidentiality gate and dry-run engine."""

import json
import os
import time
import urllib.error  # noqa: F401
import urllib.request  # noqa: F401
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from msu_llm_lab.models.catalog import DEFAULT_PORTFOLIO, ModelCandidate
from msu_llm_lab.models import registry
from msu_llm_lab.providers.base import (
    BaseProvider,
    ConfidentialityViolationError,
    LiveNotAuthorizedError,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    RateLimitError,
)
from msu_llm_lab.providers.http import (  # noqa: F401  (re-exported for callers and tests)
    IN_FLIGHT_BUDGET,
    MAX_ATTEMPTS_CAP,
    RETRYABLE_STATUS,
    PostResult,
    RetryingPoster,
)
from msu_llm_lab.schemas.evaluation import EvaluationCase


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class OpenRouterAdapter(BaseProvider):
    """Adapter for OpenRouter API with fail-closed privacy gatekeeper and dry-run engine."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: Optional[int] = None,
        max_attempts: Optional[int] = None,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 8.0,
        provider_latency_field: Optional[Tuple[str, ...]] = None,
        registry_stage: str = "pilot",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(api_key=key, base_url=base_url or self.DEFAULT_BASE_URL)
        self.timeout_seconds = timeout_seconds
        # `max_retries` is the legacy name for the TOTAL number of attempts; both are bounded.
        requested = max_attempts if max_attempts is not None else (max_retries if max_retries is not None else 3)
        self.max_attempts = max(1, min(int(requested), MAX_ATTEMPTS_CAP))
        self.max_retries = self.max_attempts
        self.backoff_base_seconds = backoff_base_seconds
        self.max_backoff_seconds = max_backoff_seconds
        # Key path to a provider-reported model latency in the response body; None until a real field is verified.
        self.provider_latency_field = provider_latency_field
        self._sleep = sleep
        self._clock = clock
        self.registry_stage = registry_stage

    @staticmethod
    def validate_case_confidentiality(case: EvaluationCase) -> None:
        """Fail-Closed Confidentiality Gatekeeper.
        
        Strictly prevents any record not explicitly classified as PUBLIC or SYNTHETIC
        from being routed to external network APIs.
        """
        if not case.is_eligible_for_external_dispatch:
            raise ConfidentialityViolationError(
                f"Confidentiality Gate: Case '{case.id}' has classification "
                f"'{case.classification.value}', which is barred from outbound network dispatch. "
                "Only PUBLIC and SYNTHETIC records are eligible."
            )

    def _get_pricing_for_model(self, model_id: str) -> tuple[Decimal, Decimal]:
        """Resolves input and output pricing per 1M tokens for model."""
        sourced = registry.model_level_prices(model_id)  # the sourced registry is authoritative and is consulted first
        if sourced is not None:
            return sourced
        for candidate in DEFAULT_PORTFOLIO:
            if candidate.model_id == model_id:
                return candidate.input_price_per_1m_usd, candidate.output_price_per_1m_usd
        raise ProviderError(
            f"No pricing metadata for model '{model_id}'. Register it in the model catalog "
            "before use; costs are never guessed."
        )

    def _calculate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Calculates exact cost in Decimal USD."""
        in_price, out_price = self._get_pricing_for_model(model_id)
        cost_in = (Decimal(prompt_tokens) * in_price) / Decimal("1000000")
        cost_out = (Decimal(completion_tokens) * out_price) / Decimal("1000000")
        return (cost_in + cost_out).quantize(Decimal("0.000001"))

    def _simulate_dry_run(self, request: ProviderRequest) -> ProviderResponse:
        """Deterministic dry-run mock response generator with zero network dispatch.

        Every number here is SIMULATED and labelled so; it must never be reported as a provider measurement.
        """
        user_msg = ""
        for m in request.messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")

        p_tok = max(10, int(len(user_msg.split()) * 1.35))
        
        if request.expected_answer:
            output = f"Based on MSU official academic regulations: {request.expected_answer}"
        else:
            output = f"MSU Admissions Advisory: Information regarding '{user_msg[:60]}' is governed by faculty guidelines."

        c_tok = max(15, int(len(output.split()) * 1.35))
        latency = round(140.0 + (c_tok * 2.8), 2)
        cost = self._calculate_cost(request.model_id, p_tok, c_tok)

        return ProviderResponse(
            output_text=output,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            usage_source="SIMULATED",
            latency_ms=latency,
            latency_source="SIMULATED",
            model_latency_source="SIMULATED",
            cost_usd=cost,
            cost_basis="SIMULATED",
            model_id=request.model_id,
            is_dry_run=True,
            raw_response={"simulation": True, "provider": "openrouter_mock"},
        )

    def _extract_provider_latency(self, data: Dict[str, Any]) -> Optional[float]:
        if not self.provider_latency_field:
            return None
        node: Any = data
        for part in self.provider_latency_field:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        if isinstance(node, (int, float)) and not isinstance(node, bool) and node >= 0:
            return float(node)
        return None

    def _build_live_response(
        self, request: ProviderRequest, body: bytes, e2e_ms: float, attempt_ms: float, attempts: int, errors: List[str]
    ) -> ProviderResponse:
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")
        if data.get("error"):
            raise ProviderError(
                f"OpenRouter returned an error body: {self.scrub_secrets(json.dumps(data.get('error'))[:300])}",
                attempts=attempts, errors=errors + ["ERROR_BODY"], retryable=False,
            )
        choices = data.get("choices") or []
        out_text = ""
        if choices and isinstance(choices[0], dict):
            content = (choices[0].get("message") or {}).get("content")
            out_text = content if isinstance(content, str) else ""
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        p_tok, c_tok = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
        usage_source = "PROVIDER_REPORTED" if (p_tok is not None and c_tok is not None) else (
            "PARTIAL" if (p_tok is not None or c_tok is not None) else "UNAVAILABLE")

        reported = usage.get("cost")
        actual = Decimal(str(reported)) if isinstance(reported, (int, float)) and not isinstance(reported, bool) and reported >= 0 else None
        # Estimates need provider-reported tokens AND a verified route-specific price (legacy catalog models are unchanged).
        estimated = (
            self._calculate_cost(request.model_id, p_tok, c_tok)
            if (p_tok is not None and c_tok is not None and registry.estimate_permitted(request.model_id))
            else None
        )
        if actual is not None:
            cost, basis = actual, "PROVIDER_REPORTED"
        elif estimated is not None:
            cost, basis = estimated, "ESTIMATED_FROM_CATALOG"
        else:
            cost, basis = Decimal("0.0000"), "UNAVAILABLE"

        model_ms = self._extract_provider_latency(data)
        resolved = data.get("model") if isinstance(data.get("model"), str) else None
        serving = data.get("provider") if isinstance(data.get("provider"), str) else None
        return ProviderResponse(
            resolved_model_id=resolved,
            serving_provider=serving,
            output_text=out_text,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            usage_source=usage_source,
            latency_ms=round(e2e_ms, 2),
            latency_source="MEASURED_CLIENT",
            attempt_latency_ms=round(attempt_ms, 2),
            model_latency_ms=model_ms,
            model_latency_source="PROVIDER_REPORTED" if model_ms is not None else "UNAVAILABLE",
            attempts=attempts,
            retry_errors=list(errors),
            actual_cost_usd=actual,
            estimated_cost_usd=estimated,
            cost_usd=cost,
            cost_basis=basis,
            model_id=request.model_id,
            is_dry_run=False,
            raw_response=self.scrub_secrets(data),
        )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Executes model generation in dry-run mode or dispatches a live network call."""
        if request.dry_run:
            return self._simulate_dry_run(request)

        # Live execution gates, all checked BEFORE any request is built or sent.
        if not request.live_authorized:
            raise LiveNotAuthorizedError("Live request refused: it carries no explicit live authorization.")
        if not self.api_key or not self.api_key.strip():
            raise ProviderError(
                "OPENROUTER_API_KEY environment variable not configured. "
                "Live network calls require a valid API key. Use --dry-run for local offline simulation."
            )
        blockers = registry.synthetic_live_blockers(request.model_id, self.registry_stage)
        if blockers:
            raise ProviderError(f"Live request refused: registry blockers for '{request.model_id}': {', '.join(blockers)}.")
        self._get_pricing_for_model(request.model_id)  # unknown model fails here, before any spend

        payload = {
            "model": request.model_id,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
        }
        if request.temperature is not None:  # omitted when the pinned route does not list temperature
            payload["temperature"] = request.temperature
        if request.provider_preferences:
            payload["provider"] = request.provider_preferences
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Sudharsanselvaraj/Model-Systems-Workbench",
            "X-Title": "MSU LLM Inference Workbench",
        }
        poster = RetryingPoster(
            timeout_seconds=self.timeout_seconds, max_attempts=self.max_attempts,
            backoff_base_seconds=self.backoff_base_seconds, max_backoff_seconds=self.max_backoff_seconds,
            sleep=self._sleep, clock=self._clock,
        )
        result = poster.post(self.base_url, headers, payload)
        try:
            return self._build_live_response(request, result.body, result.e2e_ms, result.attempt_ms, result.attempts, result.errors)
        except ProviderError:
            raise
        except (ValueError, KeyError, TypeError) as e:
            # A malformed 200 response is NOT retried: the request may already have been billed.
            raise ProviderError(
                f"OpenRouter returned an unreadable response ({type(e).__name__}).",
                attempts=result.attempts, errors=result.errors + ["INVALID_RESPONSE"], retryable=False,
            ) from e
