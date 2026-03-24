"""Shared HTTP transport helpers for external judge providers."""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional, Sequence


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def should_retry_http_status(status_code: Optional[int]) -> bool:
    """Return whether an HTTP status should be retried."""
    return status_code in RETRYABLE_HTTP_STATUS_CODES


def compute_retry_delay_seconds(
    attempt: int,
    *,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    jitter_ratio: float = 0.25,
    rng: Optional[Callable[[float, float], float]] = None,
) -> float:
    """Compute exponential backoff with optional jitter.

    `attempt` is zero-based for the failed attempt index.
    """
    delay = min(max_delay, base_delay * (2 ** attempt))
    if jitter_ratio <= 0:
        return delay

    low = delay * (1.0 - jitter_ratio)
    high = delay * (1.0 + jitter_ratio)
    sampler = rng or random.uniform
    return sampler(low, high)


def post_json_with_retry(
    *,
    url: str,
    payload: Dict,
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    provider_name: str,
) -> Dict:
    """POST JSON with provider-aware retry/backoff logging."""
    request_body = json.dumps(payload).encode("utf-8")
    request_body_bytes = len(request_body)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            data=request_body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt + 1 < max_retries and should_retry_http_status(exc.code):
                delay = compute_retry_delay_seconds(attempt)
                print(
                    f"[{provider_name}] HTTP {exc.code} on attempt {attempt + 1}/{max_retries}; "
                    f"retrying in {delay:.1f}s (payload_bytes={request_body_bytes})",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_error = exc
                continue
            raise RuntimeError(f"{provider_name} API request failed ({exc.code}): {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt + 1 < max_retries:
                delay = compute_retry_delay_seconds(attempt)
                print(
                    f"[{provider_name}] transient network error on attempt {attempt + 1}/{max_retries}; "
                    f"retrying in {delay:.1f}s (payload_bytes={request_body_bytes}): {exc}",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_error = exc
                continue
            raise RuntimeError(f"{provider_name} request failed after retries: {exc}") from exc

    raise RuntimeError(f"{provider_name} request failed after retries.") from last_error
