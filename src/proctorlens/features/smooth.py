"""One Euro filter (Casiez et al. 2012) for the pose / gaze signals: causal and dt-aware, steady at rest and quick
on fast moves because the cutoff rises with the filtered speed. FaceSmoother applies it to the Face fields that
feed the pose columns and the gaze regression, and can hold the eye signals through a blink.

Deviation from the paper: speed below speed_floor is treated as noise and does not open the cutoff. At 10 Hz the
speed estimate of a noisy still signal is large enough to open a plain One Euro cutoff and cancel the smoothing."""
from __future__ import annotations

import dataclasses
import math

from proctorlens.core.config import SmoothCfg
from proctorlens.core.types import Face
from proctorlens.perception.gaze import EYELOOK

# ponytail: one beta / floor for every signal needs one unit; ~50 deg of head or gaze travel ~ 1.0 in the [0,1]
# signals (iris ratios, blendshapes, face centre). Per-signal parameters if that proves too coarse.
_UNIT = 50.0


def _alpha(cutoff: float, dt: float) -> float:
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))  # dt / (dt + tau), tau = 1 / (2 pi cutoff)


class OneEuro:
    """x_hat follows x through a low-pass with cutoff min_cutoff + beta * max(0, speed - floor) Hz, where speed is
    the low-passed |dx/dt| times `unit` (degree-equivalents per second)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float, floor: float = 0.0, unit: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff, self.floor, self.unit = min_cutoff, beta, d_cutoff, floor, unit
        self.reset()

    def reset(self) -> None:
        self.t: int | None = None  # time of the last update, ms
        self.x: float | None = None  # last output
        self.dx = 0.0  # filtered speed, signal units / s

    def __call__(self, x: float, t_ms: int) -> float:
        if not math.isfinite(x):  # NaN must not poison the state: restart at the next finite sample
            self.reset()
            return x
        if self.t is None:
            self.t, self.x = t_ms, x
            return x
        dt = (t_ms - self.t) / 1000.0
        if dt <= 0:
            return self.x
        self.dx += _alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        self.x += _alpha(self.min_cutoff + self.beta * max(0.0, abs(self.dx) * self.unit - self.floor), dt) * (x - self.x)
        self.t = t_ms
        return self.x


class FaceSmoother:
    """Smooths the Face fields gaze_features() and the pose columns read: yaw, pitch, roll, centre, scale, iris
    ratios and the eyeLook* blendshapes. Mouth blendshapes, bbox and eye_open are left alone."""

    def __init__(self, cfg: SmoothCfg):
        self.cfg = cfg
        self._f: dict[str, OneEuro] = {}

    def reset(self) -> None:
        for f in self._f.values():
            f.reset()

    def _s(self, key: str, x: float, t_ms: int, unit: float = 1.0, hold: bool = False) -> float:
        f = self._f.get(key)
        if f is None:
            c = self.cfg
            f = self._f[key] = OneEuro(c.min_cutoff, c.beta, c.d_cutoff, c.speed_floor, unit)
        if hold and f.x is not None:
            return f.x  # blink: keep the last good value, filter state frozen
        return f(x, t_ms)

    def __call__(self, face: Face, t_ms: int, hold_eyes: bool = False) -> Face:
        s = self._s
        blend = dict(face.blend)
        for _, k in EYELOOK:
            if k in blend:
                blend[k] = s(k, blend[k], t_ms, _UNIT, hold_eyes)
        return dataclasses.replace(
            face, yaw=s("yaw", face.yaw, t_ms), pitch=s("pitch", face.pitch, t_ms), roll=s("roll", face.roll, t_ms),
            center=(s("cx", face.center[0], t_ms, _UNIT), s("cy", face.center[1], t_ms, _UNIT)),
            scale=s("scale", face.scale, t_ms, _UNIT),
            iris=tuple(s(f"iris{i}", v, t_ms, _UNIT, hold_eyes) for i, v in enumerate(face.iris)), blend=blend)
