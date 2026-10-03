"""Fixed-grid resampler: irregular frame timestamps -> 10 Hz grid, nearest-before with staleness."""
from __future__ import annotations

import math
from typing import Any


class GridResampler:
    """push(t_ms, item) -> [(grid_t_ms, item | None, age_ms)].

    Each grid step T uses the latest frame with t <= T (never the future). If that frame is older than
    max_age_ms, or none exists yet, item is None (=> frame_valid=False). Duplicate / out-of-order frames
    (t <= last t) are dropped. A step is emitted only once a frame at t >= T has arrived, so a long gap
    yields one (T, None, age) per missed step — never interpolated.
    """

    def __init__(self, hz: int = 10, max_age_ms: int = 250):
        self.step = 1000 // hz
        self.max_age_ms = max_age_ms
        self._next: int | None = None
        self._prev: tuple[int, Any] | None = None

    def _emit(self, T: int) -> tuple[int, Any, float]:
        if self._prev is None:
            return T, None, math.inf
        age = T - self._prev[0]
        return T, (self._prev[1] if age <= self.max_age_ms else None), float(age)

    def push(self, t_ms: int, item: Any) -> list[tuple[int, Any, float]]:
        if self._prev is not None and t_ms <= self._prev[0]:
            return []
        if self._next is None:
            self._next = -(-t_ms // self.step) * self.step  # first grid step at/after the first frame
        out = []
        while self._next < t_ms:
            out.append(self._emit(self._next))
            self._next += self.step
        self._prev = (t_ms, item)
        if self._next == t_ms:
            out.append(self._emit(self._next))
            self._next += self.step
        return out
