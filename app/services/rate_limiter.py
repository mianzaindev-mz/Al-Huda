"""
services/rate_limiter.py
------------------------
Simple sliding-window rate limiter to avoid exceeding Mistral API
quotas, plus an async retry decorator with exponential backoff.
"""

import asyncio
import logging
import time
from functools import wraps

from app.core.config import MAX_RETRIES, RETRY_DELAY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter (default: 60 req / 60 s)."""

    def __init__(self, max_requests: int = 60, window: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window
        self._requests: list[float] = []

    def is_allowed(self) -> bool:
        """Return True if a new request is within the rate limit."""
        now = time.time()
        # Discard timestamps outside the current window
        self._requests = [t for t in self._requests if now - t < self.window]
        if len(self._requests) < self.max_requests:
            self._requests.append(now)
            return True
        return False

    def get_wait_time(self) -> float:
        """Seconds to wait before the next request is allowed."""
        now = time.time()
        # Clean up old requests first to avoid stale entries
        self._requests = [t for t in self._requests if now - t < self.window]
        if not self._requests:
            return 0.0
        oldest = self._requests[0]
        return max(0.0, self.window - (now - oldest))

    @property
    def request_count(self) -> int:
        """Current number of requests in the active sliding window."""
        now = time.time()
        self._requests = [t for t in self._requests if now - t < self.window]
        return len(self._requests)


# Singleton used across the app
rate_limiter = RateLimiter(max_requests=60, window=60)


# ---------------------------------------------------------------------------
# Retry Decorator
# ---------------------------------------------------------------------------

def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_DELAY):
    """Decorator: retry an async function with exponential backoff on failure."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}: {exc}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}. "
                        f"Retrying in {delay}s…"
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator
