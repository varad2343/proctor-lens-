"""Shared event state machine: IDLE -> PENDING -> ACTIVE -> COOLDOWN, one instance per score key.

Time model: a step at t covers [t, t+step_ms). Event = [first step >= on_thr, last in-event step + step_ms).
Debounce/hysteresis are counted in scored steps, so None (unreliable) steps pause them.
A finished event is held until no later event can merge into it (gap < merge_gap_s), then min_dur_s
filtered and emitted. Invariants: events never overlap, end_ms >= start_ms, consecutive events are
>= max(merge_gap_s, cooldown_s) apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.config import EventCfg
from ..core.types import Event

IDLE, PENDING, ACTIVE, COOLDOWN = "IDLE", "PENDING", "ACTIVE", "COOLDOWN"


@dataclass
class _Run:
    start: int
    end: int = 0
    total: float = 0.0
    n: int = 0
    peak: float = 0.0

    def add(self, s: float) -> None:
        self.total += s
        self.n += 1
        self.peak = max(self.peak, s)

    def absorb(self, o: _Run) -> None:
        self.total += o.total
        self.n += o.n
        self.peak = max(self.peak, o.peak)


class EventMachine:
    def __init__(self, key: str, cfg: EventCfg, event_type: str, step_ms: int, detector: str):
        self.key, self.cfg, self.type, self.step, self.detector = key, cfg, event_type, step_ms, detector
        steps = lambda s: max(1, math.ceil(round(s * 1000 / step_ms, 6)))  # noqa: E731
        self._n_on, self._n_off, self._n_hold = steps(cfg.t_on_s), steps(cfg.t_off_s), steps(cfg.max_hold_s)
        self._merge_ms, self._cool_ms = round(cfg.merge_gap_s * 1000), round(cfg.cooldown_s * 1000)
        self._reset()

    def _reset(self) -> None:
        self.state = IDLE
        self._act: _Run | None = None  # ACTIVE event
        self._pend: _Run | None = None  # run of steps >= on_thr, not yet long enough
        self._off: _Run | None = None  # tentative steps < off_thr inside an ACTIVE event
        self._held: _Run | None = None  # ended event still waiting for a possible merge
        self._none = 0  # consecutive None steps
        self._cool_until = -math.inf

    def update(self, t_ms: int, score: float | None) -> list[Event]:
        out: list[Event] = []
        if score is None or score != score:
            self._none += 1
            if self._none >= self._n_hold:  # unreliable for too long: stop holding
                self._pend = None
                if self._act:
                    self._end_active()
        else:
            self._none = 0
            self._step(t_ms, min(1.0, max(0.0, float(score))))
        if self._held:
            nxt = self._pend.start if self._pend else t_ms + self.step  # earliest start of a later event
            if nxt - self._held.end >= self._merge_ms:
                out += self._emit(self._held)
                self._held = None
        if self._act:
            self.state = ACTIVE
        elif self._pend:
            self.state = PENDING
        elif self._held or t_ms + self.step < self._cool_until:
            self.state = COOLDOWN
        else:
            self.state = IDLE
        return out

    def _step(self, t: int, s: float) -> None:
        c, a = self.cfg, self._act
        if a:
            if s >= c.off_thr:
                if self._off:  # dip shorter than t_off: stays one event
                    a.absorb(self._off)
                    self._off = None
                a.add(s)
                a.end = t + self.step
            else:
                self._off = self._off or _Run(t)
                self._off.add(s)
                if self._off.n >= self._n_off:
                    self._end_active()
        elif s >= c.on_thr and (self._held or t >= self._cool_until):
            p = self._pend = self._pend or _Run(t)
            p.add(s)
            if p.n >= self._n_on:
                self._pend = None
                if self._held:  # gap < merge_gap: continue the earlier event
                    self._held.absorb(p)
                    p, self._held = self._held, None
                p.end = t + self.step
                self._act = p
        else:
            self._pend = None

    def _end_active(self) -> None:
        self._held, self._act, self._off = self._act, None, None
        self._cool_until = self._held.end + self._cool_ms

    def _emit(self, r: _Run) -> list[Event]:
        if r.end - r.start < round(self.cfg.min_dur_s * 1000):
            return []
        return [self._event(r, "final")]

    def _event(self, r: _Run, status: str) -> Event:
        mean = r.total / r.n
        return Event(self.type, r.start, r.end, min(1.0, mean), self.detector, status=status,
                     details={"key": self.key, "score_peak": r.peak, "score_mean": mean, "n_steps": r.n})

    def ongoing(self) -> Event | None:
        return self._event(self._act, "ongoing") if self._act else None

    def close(self, t_ms: int) -> list[Event]:
        """End of stream: finalize whatever is open. End time = last evidence (t_ms is informational)."""
        if self._act:
            self._end_active()
        out = self._emit(self._held) if self._held else []
        self._reset()
        return out
