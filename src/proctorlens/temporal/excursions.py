"""Off-screen excursion counter behind REPEATED_GLANCING (E5)."""
from __future__ import annotations

from collections import Counter, deque

from ..core.config import GlanceCfg


class ExcursionTracker:
    """An excursion = run of steps with off_score >= on_thr lasting [min_s, max_s] (longer ones belong to
    OFF_SCREEN_SUSTAINED). Score 1.0 when >= n excursions ended in the last window_s and >= same_zone_frac
    of them share a zone. None steps neither start nor end an excursion (they stay inside its duration)."""

    def __init__(self, cfg: GlanceCfg, step_ms: int):
        self.cfg, self.step = cfg, step_ms
        ms = lambda s: round(s * 1000)  # noqa: E731
        self._lo, self._hi, self._win = ms(cfg.min_s), ms(cfg.max_s), ms(cfg.window_s)
        self._cur: tuple[int, int, Counter] | None = None  # start, end of last off step, zones seen
        self._ex: deque[dict] = deque()
        self._t = 0

    def update(self, t_ms: int, off_score: float | None, zone: str) -> float | None:
        self._t = t_ms
        if off_score is not None and off_score == off_score:  # NaN counts as None
            if off_score >= self.cfg.on_thr:
                s, _, z = self._cur or (t_ms, 0, Counter())
                z[zone] += 1
                self._cur = (s, t_ms + self.step, z)
            elif self._cur:
                s, e, z = self._cur
                self._cur = None
                if self._lo <= e - s <= self._hi:
                    self._ex.append({"start_ms": s, "end_ms": e, "zone": z.most_common(1)[0][0]})
        while self._ex and self._ex[0]["end_ms"] <= t_ms - self._win:
            self._ex.popleft()
        if off_score is None or off_score != off_score:
            return None
        n = len(self._ex)
        if n < self.cfg.n:
            return 0.0
        top = Counter(x["zone"] for x in self._ex).most_common(1)[0][1]
        return 1.0 if top / n >= self.cfg.same_zone_frac else 0.0

    def excursions(self) -> list[dict]:
        return [dict(x) for x in self._ex]
