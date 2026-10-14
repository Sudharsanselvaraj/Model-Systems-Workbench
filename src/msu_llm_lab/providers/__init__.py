"""Provider abstraction package for external API and local serving backends."""

from msu_llm_lab.providers.base import (
    BaseProvider,
    ConfidentialityViolationError,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    RateLimitError,
    scrub_secrets,
)
from msu_llm_lab.providers.openrouter import OpenRouterAdapter

__all__ = [
    "BaseProvider",
    "OpenRouterAdapter",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderError",
    "ConfidentialityViolationError",
    "RateLimitError",
    "scrub_secrets",
]
