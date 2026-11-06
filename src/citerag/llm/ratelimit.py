"""Client-side sliding-window rate limiter (free tier: 20 requests per minute; we default to 15).

Pure and clock-injectable so it can be unit-tested. The async gateway does:
    while (wait := limiter.try_acquire()) > 0:
        await asyncio.sleep(wait)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    def __init__(
        self, max_calls: int, per_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic
    ):
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls, self.per, self._clock = max_calls, per_seconds, clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] >= self.per:
            self._stamps.popleft()

    def try_acquire(self, n: int = 1) -> float:
        """Take n slots if available and return 0.0; otherwise take nothing and return seconds to wait.
        Use n > 1 to reserve slots for a CrewAI run (it calls the provider outside the gateway)."""
        if n > self.max_calls:
            raise ValueError("n exceeds the window size")
        with self._lock:
            now = self._clock()
            self._purge(now)
            if len(self._stamps) + n <= self.max_calls:
                self._stamps.extend([now] * n)
                return 0.0
            need = len(self._stamps) + n - self.max_calls  # how many old stamps must expire first
            return max(self._stamps[need - 1] + self.per - now, 0.001)

    def in_window(self) -> int:
        with self._lock:
            self._purge(self._clock())
            return len(self._stamps)
