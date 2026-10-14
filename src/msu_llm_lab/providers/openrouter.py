"""OpenRouter provider adapter with fail-closed confidentiality gate and dry-run engine."""

import json
import os
import time
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Any, Dict, Optional

from msu_llm_lab.models.catalog import DEFAULT_PORTFOLIO, ModelCandidate
from msu_llm_lab.providers.base import (
    BaseProvider,
    ConfidentialityViolationError,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    RateLimitError,
)
from msu_llm_lab.schemas.evaluation import EvaluationCase


class OpenRouterAdapter(BaseProvider):
    """Adapter for OpenRouter API with fail-closed privacy gatekeeper and dry-run engine."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        super().__init__(api_key=key, base_url=base_url or self.DEFAULT_BASE_URL)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

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
        for candidate in DEFAULT_PORTFOLIO:
            if candidate.model_id == model_id:
                return candidate.input_price_per_1m_usd, candidate.output_price_per_1m_usd
        # Fallback default pricing
        return Decimal("0.1500"), Decimal("0.6000")

    def _calculate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Calculates exact cost in Decimal USD."""
        in_price, out_price = self._get_pricing_for_model(model_id)
        cost_in = (Decimal(prompt_tokens) * in_price) / Decimal("1000000")
        cost_out = (Decimal(completion_tokens) * out_price) / Decimal("1000000")
        return (cost_in + cost_out).quantize(Decimal("0.000001"))

    def _simulate_dry_run(self, request: ProviderRequest) -> ProviderResponse:
        """Deterministic dry-run mock response generator with zero network dispatch."""
        user_msg = ""
        for m in request.messages:
            if m.get("role") == "user":
                user_msg = m.get("content", "")

        # Estimate tokens safely
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
            latency_ms=latency,
            cost_usd=cost,
            model_id=request.model_id,
            is_dry_run=True,
            raw_response={"simulation": True, "provider": "openrouter_mock"},
        )

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Executes model generation in dry-run mode or dispatches live network call."""
        if request.dry_run:
            return self._simulate_dry_run(request)

        # Live Execution Gate
        if not self.api_key or not self.api_key.strip():
            raise ProviderError(
                "OPENROUTER_API_KEY environment variable not configured. "
                "Live network calls require a valid API key. Use --dry-run for local offline simulation."
            )

        payload = {
            "model": request.model_id,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        json_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Sudharsanselvaraj/Model-Systems-Workbench",
            "X-Title": "MSU LLM Inference Workbench",
        }

        req = urllib.request.Request(self.base_url, data=json_bytes, headers=headers, method="POST")

        start_time = time.time()
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            attempts += 1
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    latency_ms = round((time.time() - start_time) * 1000.0, 2)
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))

                    choices = data.get("choices", [])
                    out_text = choices[0]["message"]["content"] if choices else ""
                    usage = data.get("usage", {})
                    p_tok = usage.get("prompt_tokens", len(request.messages[0]["content"].split()))
                    c_tok = usage.get("completion_tokens", len(out_text.split()))
                    cost = self._calculate_cost(request.model_id, p_tok, c_tok)

                    return ProviderResponse(
                        output_text=out_text,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        latency_ms=latency_ms,
                        cost_usd=cost,
                        model_id=request.model_id,
                        is_dry_run=False,
                        raw_response=self.scrub_secrets(data),
                    )

            except urllib.error.HTTPError as e:
                status_code = e.code
                error_body = e.read().decode("utf-8", errors="replace")
                scrubbed_body = self.scrub_secrets(error_body)

                if status_code == 429:
                    last_error = RateLimitError(f"OpenRouter Rate Limit (429): {scrubbed_body}")
                    time.sleep(2.0 * attempts)
                elif 500 <= status_code < 600:
                    last_error = ProviderError(f"OpenRouter Upstream Error ({status_code}): {scrubbed_body}")
                    time.sleep(1.5 * attempts)
                else:
                    raise ProviderError(f"OpenRouter HTTP Error ({status_code}): {scrubbed_body}") from e
            except Exception as e:
                scrubbed_msg = self.scrub_secrets(str(e))
                last_error = ProviderError(f"OpenRouter Network Request Failed: {scrubbed_msg}")
                time.sleep(1.0 * attempts)

        raise last_error or ProviderError("OpenRouter request failed after maximum retries.")
