"""Shared bounded-retry JSON POST used by every OpenRouter protocol adapter (Chat Completions and Decisions API).

Retry policy lives in exactly one place. Only `urllib.request.urlopen` touches the network, so tests can replace it.
"""

import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from msu_llm_lab.providers.base import ProviderError, RateLimitError, scrub_secrets

# HTTP statuses that are worth retrying: request timeout, rate limit, and transient upstream errors.
# 524 (timeout) and 529 (overloaded) are listed by OpenRouter only as status constants without a description,
# so they are treated like 504 and 503. Timeouts may still have been billed upstream, which is why attempts
# stay bounded and the run's budget ceilings still apply.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
MAX_ATTEMPTS_CAP = 5
IN_FLIGHT_BUDGET = "openrouter_in_flight_budget"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def valid_retry_after(value: Optional[str]) -> Optional[float]:
    """A Retry-After counts only as a non-negative finite number of seconds (HTTP-date forms are not trusted)."""
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def typed_error(body_text: str) -> Tuple[Optional[str], Optional[str]]:
    """(error_type, limit_source) from OpenRouter's `error.metadata`, when present and well-formed."""
    try:
        err = json.loads(body_text).get("error")
        meta = err.get("metadata") if isinstance(err, dict) else None
    except (ValueError, AttributeError):
        return None, None
    if not isinstance(meta, dict):
        return None, None
    pick = lambda k: meta[k] if isinstance(meta.get(k), str) and _SAFE_TOKEN.match(meta[k]) else None  # noqa: E731
    return pick("error_type"), pick("limit_source")


def is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)


@dataclass
class PostResult:
    body: bytes
    attempts: int
    errors: List[str]
    e2e_ms: float
    attempt_ms: float


class RetryingPoster:
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        backoff_base_seconds: float = 1.0,
        max_backoff_seconds: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS_CAP))
        self.backoff_base_seconds = backoff_base_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep
        self._clock = clock

    def _backoff_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        delay = self.backoff_base_seconds * (2 ** (attempt - 1))
        ra = valid_retry_after(retry_after)
        if ra is not None:
            delay = max(delay, ra)
        return max(0.0, min(delay, self.max_backoff_seconds))

    def post(self, url: str, headers: dict, payload: dict) -> PostResult:
        """POST JSON. Retries only transient failures, never sleeps after the final attempt, records every attempt."""
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        started = self._clock()
        errors: List[str] = []
        for attempt in range(1, self.max_attempts + 1):
            attempt_started = self._clock()
            retry_after: Optional[str] = None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    body = resp.read()
                now = self._clock()
                return PostResult(body, attempt, list(errors), (now - started) * 1000.0, (now - attempt_started) * 1000.0)
            except urllib.error.HTTPError as e:
                status = e.code
                raw_body = e.read().decode("utf-8", errors="replace")
                scrubbed = scrub_secrets(raw_body)
                error_type, limit_source = typed_error(raw_body)
                code = f"HTTP_{status}" + (f"/{error_type}" if error_type else "")
                retry_after = e.headers.get("Retry-After") if e.headers else None
                in_flight_402 = (
                    status == 402 and limit_source == IN_FLIGHT_BUDGET and valid_retry_after(retry_after) is not None
                )
                if status not in RETRYABLE_STATUS and not in_flight_402:
                    raise ProviderError(
                        f"OpenRouter HTTP Error ({status}): {scrubbed}",
                        attempts=attempt, errors=errors + [code], status=status, retryable=False,
                    ) from e
                errors.append(code)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                errors.append("TIMEOUT" if is_timeout(e) else "NETWORK_ERROR")

            if attempt < self.max_attempts:  # never sleep after the final attempt
                self._sleep(self._backoff_delay(attempt, retry_after))

        exc_cls = RateLimitError if errors and errors[-1].startswith("HTTP_429") else ProviderError
        raise exc_cls(
            f"OpenRouter request failed after {self.max_attempts} attempt(s): {', '.join(errors)}",
            attempts=self.max_attempts, errors=errors,
            status=int(re.match(r"HTTP_(\d+)", errors[-1]).group(1)) if errors[-1].startswith("HTTP_") else None,
            retryable=True,
        )
