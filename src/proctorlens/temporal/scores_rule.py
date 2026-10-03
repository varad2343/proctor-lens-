"""Rule score provider: one feature row -> Scores (soft thresholds), causal and stateful.

None = cannot observe (quality gating): state machines start nothing and count no evidence.
Face-derived scores (face_absent, off_screen, glancing, speaking, id_mismatch) are None when the row is not
`reliable`; object/people scores are None only when the frame itself is invalid; `degraded` is never None.
"""
from __future__ import annotations

import math
from collections import deque

from ..core.config import Config
from ..core.types import Scores
from .excursions import ExcursionTracker

# ponytail: speaking constants live here (no cfg fields in the frozen contract). mouth_energy_1s is
# sqrt(sum of variances) of jaw/funnel/pucker over 1 s (features/extractor.py): ~0.1 speech, <0.015 at rest.
# Retune on data. The learned scorer overrides this.
MOUTH_ENERGY_THR, MOUTH_ENERGY_W = 0.03, 0.03
YAWN_JAW = 0.7  # jaw_open above this = yawn / wide-open mouth, not speech
SPEAK_WINDOW_S, SPEAK_ACTIVE_S = 10.0, 3.0  # spec E6: >= 3 s active within a 10 s window


def soft(x: float, thr: float, width: float) -> float:
    """Linear ramp: 0 at thr-width/2, 0.5 at thr, 1 at thr+width/2."""
    return min(1.0, max(0.0, (x - thr) / width + 0.5))


def _g(row: dict, k: str, d: float = math.nan) -> float:
    try:
        v = float(row.get(k))
    except (TypeError, ValueError):
        return d
    return d if v != v else v


class SpeakingWindow:
    """min(1, active_seconds_in_last_10s / 3) from per-step activity in [0,1]; None adds no evidence."""

    def __init__(self, hz: int):
        self.dt = 1.0 / hz
        self._w: deque[float] = deque(maxlen=round(SPEAK_WINDOW_S * hz))

    def update(self, act: float | None) -> float | None:
        if act is not None and act != act:
            act = None
        self._w.append(0.0 if act is None else act)
        return None if act is None else min(1.0, sum(self._w) * self.dt / SPEAK_ACTIVE_S)


class RuleScores:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        hz = cfg.pipeline.grid_hz
        self._exc = ExcursionTracker(cfg.glance, 1000 // hz)
        self._speak = SpeakingWindow(hz)
        self._bad, self._id, self._prev_sim = 0, 0.0, math.nan

    def update(self, row: dict) -> Scores:
        pol, idc = self.cfg.policy, self.cfg.identity
        valid, rel = bool(row.get("frame_valid", True)), bool(row.get("reliable", True))
        zone = row.get("zone", "none")
        clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731

        off = _g(row, "off_screen_score")
        if pol.allow_looking_down and zone == "down" and off == off:
            off = 0.0
        off_v = off if rel and off == off else None
        e, jaw = _g(row, "mouth_energy_1s"), _g(row, "jaw_open", 0.0)
        act = soft(e, MOUTH_ENERGY_THR, MOUTH_ENERGY_W) * (jaw < YAWN_JAW) if rel and e == e else None
        speak = self._speak.update(act)

        # id_mismatch: a *check* = finite similarity that differs from the previous row's (a check held
        # across >1 grid step repeats the same value; rows between checks are NaN). Only quality_ok checks
        # count; low-quality ones neither count nor reset. Output is step-held; a good check clears it.
        sim = _g(row, "id_similarity")
        is_check = sim == sim and sim != self._prev_sim
        self._prev_sim = sim
        if rel and is_check and row.get("id_quality_ok", False):
            self._bad = self._bad + 1 if sim < idc.tau else 0
            self._id = float(self._bad >= idc.consecutive)

        return {
            "face_absent": None if not rel else float(
                not row.get("primary_face_present", True) and not row.get("turned_away", False)
                and _g(row, "n_persons", 0.0) == 0),
            "multiple_people": None if not valid else
            soft(max(_g(row, "n_faces", 0.0), _g(row, "n_persons", 0.0)), 1.5, 1.0),
            "phone": None if not valid else clip(_g(row, "phone_conf", 0.0)),
            "notes": None if not valid else 0.0 if pol.allow_notes else clip(_g(row, "notes_conf", 0.0)),
            "off_screen": None if off_v is None else clip(off_v),
            "glancing": self._exc.update(int(row["t_ms"]), off_v, zone),
            "speaking": 0.0 if speak is not None and pol.allow_reading_aloud else speak,
            "id_mismatch": self._id if rel else None,
            "degraded": 0.0 if rel else 1.0,
        }
