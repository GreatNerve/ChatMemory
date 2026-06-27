"""Gemini API rate limiter for build-time persona analysis calls.

Limits: 14 RPM (requests per minute), 100k TPM (tokens per minute).
Uses a simple sliding-window reset: when a limit is exceeded, sleeps until
the current 60-second window expires, then resets counters.

Usage (sync, workspace.py context):
    _rate_limiter.acquire(estimated_tokens)
    result = gemini_service.chat(...)
    _rate_limiter.record(estimated_tokens)
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("chatmemory.rate_limit")


def estimate_tokens(text: str) -> int:
    """Rough chars-to-tokens estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


class GeminiRateLimiter:
    """Tracks RPM and TPM usage; sleeps when limits would be exceeded.

    Thread safety: persona builds run single-file behind the GPU mutex,
    so no locking is needed for the standard use-case.
    """

    def __init__(self, max_rpm: int = 14, max_tpm: int = 100_000) -> None:
        self._max_rpm = max_rpm
        self._max_tpm = max_tpm
        # Monotonic timestamp of the current window's start
        self._window_start: float = time.monotonic()
        self._req_count: int = 0
        self._token_count: int = 0

    def _maybe_reset_window(self) -> None:
        """Roll over to a new window if 60 seconds have elapsed."""
        if time.monotonic() - self._window_start >= 60.0:
            self._window_start = time.monotonic()
            self._req_count = 0
            self._token_count = 0

    def acquire(self, estimated_tokens: int) -> None:
        """Sleep until a new request fits within the current rate limits.

        Checks both RPM and TPM.  If either limit would be exceeded by this
        request, sleeps until the current minute window expires (plus a small
        0.25 s grace period), then resets counters.

        Args:
            estimated_tokens: Estimated input token count for this request.
                               Use estimate_tokens() for a quick approximation.
        """
        self._maybe_reset_window()

        will_exceed_rpm = self._req_count + 1 > self._max_rpm
        will_exceed_tpm = self._token_count + estimated_tokens > self._max_tpm

        if will_exceed_rpm or will_exceed_tpm:
            elapsed = time.monotonic() - self._window_start
            sleep_secs = max(0.0, 60.25 - elapsed)  # 0.25 s grace buffer
            logger.info(
                "GeminiRateLimiter sleeping %.1fs "
                "(rpm=%d/%d, tpm=%d/%d, new_req_tokens=%d)",
                sleep_secs,
                self._req_count,
                self._max_rpm,
                self._token_count,
                self._max_tpm,
                estimated_tokens,
            )
            time.sleep(sleep_secs)
            # Reset window after sleep
            self._window_start = time.monotonic()
            self._req_count = 0
            self._token_count = 0

    def record(self, tokens: int) -> None:
        """Record a completed request and its token usage in the current window.

        Call this immediately after a successful Gemini API call.

        Args:
            tokens: Estimated or actual input token count for the completed call.
        """
        self._maybe_reset_window()
        self._req_count += 1
        self._token_count += tokens


# Module-level singleton; shared across all workspace build functions so
# concurrent builds (if ever parallelised) respect the same quota window.
_rate_limiter = GeminiRateLimiter(max_rpm=14, max_tpm=100_000)
