"""Abstract base provider interface, request/response models, and secret scrubbers."""

import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConfidentialityViolationError(Exception):
    """Raised when an unapproved or confidential case is dispatched to an external network provider."""
    pass


class ProviderError(Exception):
    """Base exception for model provider errors."""
    pass


class RateLimitError(ProviderError):
    """Raised on HTTP 429 rate limit errors."""
    pass


class ProviderRequest(BaseModel):
    """Standardized model request payload."""
    model_id: str
    messages: List[Dict[str, str]] = Field(description="List of role/content chat messages")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    dry_run: bool = Field(default=True, description="When True, prevents any external network requests")
    case_id: Optional[str] = None
    expected_answer: Optional[str] = None


class ProviderResponse(BaseModel):
    """Standardized model response telemetry."""
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: Decimal = Field(default=Decimal("0.0000"))
    model_id: str
    is_dry_run: bool = True
    raw_response: Dict[str, Any] = Field(default_factory=dict)


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
            return scrubbed
        elif isinstance(data, dict):
            clean_dict = {}
            for k, v in data.items():
                if any(sec in k.lower() for sec in ("key", "secret", "token", "auth", "password")):
                    clean_dict[k] = "[REDACTED_SECRET]"
                else:
                    clean_dict[k] = BaseProvider.scrub_secrets(v)
            return clean_dict
        elif isinstance(data, list):
            return [BaseProvider.scrub_secrets(item) for item in data]
        return data

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Dispatches request to provider or executes deterministic dry-run."""
        pass


scrub_secrets = BaseProvider.scrub_secrets

