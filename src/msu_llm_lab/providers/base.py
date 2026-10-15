"""Abstract base provider interface, request/response models, and secret scrubbers."""

import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ConfidentialityViolationError(Exception):
    """Raised when an unapproved or confidential case is dispatched to an external network provider."""
    pass


class ProviderError(Exception):
    """Base exception for model provider errors.

    Optional keyword metadata records what actually happened: how many attempts were made,
    the scrubbed per-attempt error codes, the final HTTP status, and whether it was retryable.
    """

    def __init__(
        self,
        message: str = "",
        *,
        attempts: Optional[int] = None,
        errors: Optional[List[str]] = None,
        status: Optional[int] = None,
        retryable: Optional[bool] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.errors = list(errors) if errors else []
        self.status = status
        self.retryable = retryable


class RateLimitError(ProviderError):
    """Raised on HTTP 429 rate limit errors."""
    pass


class LiveNotAuthorizedError(ProviderError):
    """Raised when a request is attempted without explicit live authorization."""
    pass


class UsageUnavailableError(ProviderError):
    """Raised by runners when a live response lacks usage or cost, so spend cannot be recorded honestly."""
    pass


class ProviderRequest(BaseModel):
    """Standardized model request payload."""
    model_id: str
    messages: List[Dict[str, str]] = Field(description="List of role/content chat messages")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0, description="None omits the field (for routes that do not list it)")
    max_tokens: int = Field(default=512, ge=1)
    live_authorized: bool = Field(
        default=False,
        description="Must be True for every request; set only after explicit live confirmation",
    )
    provider_preferences: Optional[Dict[str, Any]] = Field(
        default=None, description="OpenRouter `provider` routing object (e.g. pin + no fallbacks); sent only when set"
    )
    case_id: Optional[str] = None


class ProviderResponse(BaseModel):
    """Standardized model response telemetry. Unavailable values are None, never zero or estimated."""
    output_text: str
    prompt_tokens: Optional[int] = Field(default=None, description="None when the provider did not report it")
    completion_tokens: Optional[int] = Field(default=None, description="None when the provider did not report it")
    usage_source: Literal["PROVIDER_REPORTED", "PARTIAL", "UNAVAILABLE"] = "UNAVAILABLE"
    latency_ms: float = Field(
        description="End-to-end request duration measured by this client (including retry waits)"
    )
    latency_source: Literal["MEASURED_CLIENT"] = "MEASURED_CLIENT"
    attempt_latency_ms: Optional[float] = Field(default=None, description="Measured duration of the final, successful attempt")
    model_latency_ms: Optional[float] = Field(default=None, description="Only when the provider reports it; never derived")
    model_latency_source: Literal["PROVIDER_REPORTED", "UNAVAILABLE"] = "UNAVAILABLE"
    resolved_model_id: Optional[str] = Field(default=None, description="'model' field of the response, only if present")
    serving_provider: Optional[str] = Field(default=None, description="'provider' field of the response, only if present")
    attempts: int = Field(default=1, ge=1)
    retry_errors: List[str] = Field(default_factory=list, description="Scrubbed error codes of failed attempts before success")
    native_cost: Optional[Decimal] = Field(default=None, description="Provider-reported usage.cost in its own unit (credits); preserved as received")
    native_cost_unit: Optional[str] = Field(default=None, description="Unit label of native_cost; it is booked as USD only once the unit is verified")
    actual_cost_usd: Optional[Decimal] = Field(default=None, description="USD cost, only when the provider's cost unit is verified as USD; never derived")
    estimated_cost_usd: Optional[Decimal] = Field(
        default=None, description="Catalog price x PROVIDER-REPORTED tokens; None unless both token counts were reported"
    )
    cost_usd: Decimal = Field(
        default=Decimal("0.0000"),
        description="Recorded spend: actual if reported, else the estimate, else 0 with cost_basis UNAVAILABLE",
    )
    cost_basis: Literal["PROVIDER_REPORTED", "ESTIMATED_FROM_CATALOG", "UNAVAILABLE"] = "UNAVAILABLE"
    model_id: str
    raw_response: Dict[str, Any] = Field(default_factory=dict)


# Standard provider usage counters that are numeric and must survive scrubbing.
USAGE_KEYS = frozenset({
    "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
    "cached_tokens", "reasoning_tokens", "audio_tokens", "image_tokens",
    "prompt_tokens_details", "completion_tokens_details", "input_tokens_details", "output_tokens_details",
})
_SENSITIVE_KEY_PARTS = ("key", "secret", "token", "auth", "password", "credential", "bearer", "cookie")


def require_complete_usage(response: "ProviderResponse") -> None:
    """Fail closed when a LIVE response lacks provider usage or cost: spend cannot be recorded honestly."""
    if response.usage_source != "PROVIDER_REPORTED" or response.cost_basis == "UNAVAILABLE":
        raise UsageUnavailableError(
            "Live response is missing provider token usage or cost, so spend cannot be recorded. "
            "The run is stopped; check the provider dashboard for this request.",
            attempts=response.attempts,
        )


def require_recordable_cost(response: "ProviderResponse") -> None:
    """Spend must be recordable: provider-reported cost, or a catalog estimate from reported tokens.

    Unlike `require_complete_usage`, this does not demand token counts, so a response that reports cost but
    omits tokens is still recordable. Runners that persist tokens use `require_complete_usage` instead.
    """
    if response.cost_basis not in ("PROVIDER_REPORTED", "ESTIMATED_FROM_CATALOG"):
        raise UsageUnavailableError(
            "Live response has neither a provider-reported cost nor reported token counts for a catalog estimate; "
            "spend cannot be recorded. The run is stopped; check the provider dashboard for this request.",
            attempts=response.attempts,
        )


class BaseProvider(ABC):
    """Abstract interface for LLM provider adapters."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @staticmethod
    def scrub_secrets(data: Any) -> Any:
        """Recursively scrubs API keys, auth headers, and bearer tokens from logs/data structures."""
        if isinstance(data, str):
            # Scrub Bearer tokens and generic API key patterns
            scrubbed = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{8,}", r"\1[REDACTED_SECRET]", data, flags=re.IGNORECASE)
            scrubbed = re.sub(r"(sk-or-v1-[a-zA-Z0-9_\-]{16,}|sk-[a-zA-Z0-9_\-]{16,})", "[REDACTED_API_KEY]", scrubbed)
            scrubbed = re.sub(r"(authorization\s*[:=]\s*)\S+(?:\s+\S+)?", r"\1[REDACTED_SECRET]", scrubbed, flags=re.IGNORECASE)
            return scrubbed
        elif isinstance(data, dict):
            clean_dict = {}
            for k, v in data.items():
                lk = str(k).lower()
                is_usage = lk in USAGE_KEYS and (isinstance(v, dict) or (isinstance(v, (int, float)) and not isinstance(v, bool)))
                if is_usage:
                    clean_dict[k] = BaseProvider.scrub_secrets(v) if isinstance(v, dict) else v
                elif any(sec in lk for sec in _SENSITIVE_KEY_PARTS):
                    clean_dict[k] = "[REDACTED_SECRET]"
                else:
                    clean_dict[k] = BaseProvider.scrub_secrets(v)
            return clean_dict
        elif isinstance(data, list):
            return [BaseProvider.scrub_secrets(item) for item in data]
        return data

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Dispatches one real request to the provider."""
        pass


scrub_secrets = BaseProvider.scrub_secrets

