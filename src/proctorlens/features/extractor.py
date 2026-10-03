"""FeatureExtractor: one grid step of Perceived -> a row dict holding every features.schema.COLUMNS key.

Undefined numerics are NaN, strings '' and bool columns real bools. Definitions:
  d_yaw/d_pitch/d_roll, head_x/head_y, head_scale = face minus calib.baseline (zero baseline if no calib)
  head_speed      = |d(d_yaw, d_pitch)| / dt in deg/s of the RAW pose, NaN on the first step after the face (re)appears
  blink           = 1.0 while mean eyelid openness < _BLINK_FRAC x its running open level
  mouth_energy_1s = sqrt(sum of variances) of (jawOpen, mouthFunnel, mouthPucker) over the last 1 s
                    (blendshape units); NaN when those blendshapes are missing
cfg.smooth.enabled: d_*, head_x/y/scale, iris_* and the gaze_x/y, off_screen_score, zone derived from them use One Euro
filtered values (features/smooth.py), and the iris / eye-look inputs are held while the lids are shut (a blink).
Filters restart after a face gap or a different face; TURNED_AWAY and head_speed stay on the raw pose.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Box, Perceived
from proctorlens.features.schema import COLUMNS
from proctorlens.features.smooth import FaceSmoother
from proctorlens.perception.gaze import EYELOOK, Baseline, Calibration, off_screen

_BOOL = ("frame_valid", "primary_face_present", "turned_away", "id_quality_ok", "reliable")
_MOUTH = ("jawOpen", "mouthFunnel", "mouthPucker")
_BLINK_FRAC, _EYE_ALPHA = 0.6, 0.05  # ponytail: fixed relative lid drop; 10 Hz sampling misses some blinks
_SWITCH_IOU = 0.3  # primary-face box overlap with the previous step below this = a different face (filters restart)
_GAP_MS = 1000  # a box track that goes unseen longer than this starts over (unless already static)
_BOOT_MS = 1000  # boxes already in view when the recording starts count as static after holding still this long
nan = math.nan


def iou(a: Box, b: Box) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0.0) * max(h, 0.0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


@dataclass
class _Track:
    """A box that keeps overlapping its first position (anchor) at IoU >= static_iou."""
    anchor: Box
    since: int  # when it last (re)started holding that position
    seen: int
    need: int  # how long it must hold it to count as static
    static: bool = False


class FeatureExtractor:
    def __init__(self, cfg: Config, calib: Calibration | None = None):
        self.cfg, self.calib = cfg, calib
        self.base = calib.baseline if calib else Baseline()
        self._t0: int | None = None  # first step
        self._last: tuple[int, float] | None = None  # (t, yaw) of the last step with a primary face
        self._prev: tuple[int, float, float] | None = None  # (t, d_yaw, d_pitch) if previous step had a face
        self._eye_ref: float | None = None
        self._mouth: deque = deque(maxlen=cfg.pipeline.grid_hz)
        self._faces: list[_Track] = []  # non-primary face boxes (static suppression)
        self._people: list[_Track] = []  # person boxes
        self._seat: tuple[float, float] | None = None  # where the primary face last was
        self._smooth = FaceSmoother(cfg.smooth)
        self._pbox: Box | None = None  # bbox of the last primary face (a jump = a different face)
        self._lid_t0: int | None = None  # when the current run of lowered lids began

    def _static(self, T: int, fb: list[Box], pb: list[Box]) -> tuple[list[bool], list[bool]]:
        """Which boxes sit still (poster, photo, TV). A box joins the track whose anchor it overlaps at IoU >=
        static_iou; a track is static once it held that position static_window_s (only _BOOT_MS if the box was
        already in view when the recording started, so a poster does not count for 20 s). Static tracks are sticky,
        so a dark or blocked stretch does not turn a poster into a person; other tracks restart after a gap.
        ponytail: a TV switched on mid-recording counts as a live person for static_window_s."""
        w, thr = int(self.cfg.pipeline.static_window_s * 1000), self.cfg.pipeline.static_iou

        def flag(tracks: list[_Track], boxes: list[Box]) -> list[bool]:
            out = []
            for b in boxes:
                k = next((k for k in tracks if iou(b, k.anchor) >= thr), None)
                if k is None:
                    tracks.append(k := _Track(b, T, T, min(w, _BOOT_MS) if T - self._t0 <= _BOOT_MS else w))
                elif not k.static and T - k.seen > _GAP_MS:
                    k.since = T
                k.seen, k.static = T, k.static or T - k.since >= k.need
                out.append(k.static)
            tracks[:] = [k for k in tracks if k.static or T - k.seen <= _GAP_MS]  # moving boxes leave one track per step
            return out

        return flag(self._faces, fb), flag(self._people, pb)

    def step(self, T_ms: int, p: Perceived | None, age_ms: float, fps: float) -> dict:
        cfg, pc = self.cfg, self.cfg.pipeline
        if self._t0 is None:
            self._t0 = T_ms
        since = T_ms - (self._t0 if self._last is None else self._last[0])
        row: dict = {c: nan for c in COLUMNS}
        row.update(dict.fromkeys(_BOOL, False), zone="none", quality_reasons="frame_gap" if p is None else "")
        row.update(t_ms=int(T_ms), frame_age_ms=age_ms if math.isfinite(age_ms) else nan, effective_fps=fps,
                   time_since_face_ms=float(since))
        if p is None:  # frame gap: nothing observed this step
            self._prev = None
            self._mouth.clear()
            return _clean(row)

        # a known-static face (poster) that outlived the user must not become "the user": it goes behind the live faces
        bg = [any(k.static and iou(f.bbox, k.anchor) >= pc.static_iou for k in self._faces) for f in p.faces]
        live = [f for f, b in zip(p.faces, bg) if not b]
        face, others = (live[0] if live else None), live[1:] + [f for f, b in zip(p.faces, bg) if b]
        pboxes = list(p.det.person_boxes)
        sf, sp = self._static(T_ms, [f.bbox for f in others], pboxes)
        self._seat = face.center if face else self._seat
        if self._seat:  # the user's own body (tightest person box around where the face is / was) is never background
            cx, cy = self._seat
            own = [i for i, b in enumerate(pboxes) if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
            if own:
                sp[min(own, key=lambda i: (pboxes[i][2] - pboxes[i][0]) * (pboxes[i][3] - pboxes[i][1]))] = False
        big = [f.scale >= pc.min_second_face_frac for f in others]  # tiny background faces are ignored
        for f, s in zip(others, sf):
            f.static = s
        row.update(
            frame_valid=True, luma=p.luma, blur=p.blur, overexp_frac=p.overexp_frac, quality=p.quality,
            quality_reasons=",".join(p.quality_reasons),
            phone_conf=p.det.phone_conf, notes_conf=p.det.notes_conf,
            n_faces=(face is not None) + sum(b and not s for b, s in zip(big, sf)),
            n_persons=sum(not s for s in sp),
            static_face_flags=sum(b and s for b, s in zip(big, sf)) + sum(sp),
            id_similarity=p.id.similarity if p.id else nan, id_quality_ok=bool(p.id and p.id.quality_ok),
            reliable=bool(p.quality >= cfg.quality.min_quality))  # NaN quality -> False

        if face is None:
            self._prev = None
            self._mouth.clear()
            last = self._last  # vanished right after a near-profile pose => TURNED_AWAY, not FACE_ABSENT, for
            # the hold, and for as long as a person is still detected (a turned head, not someone who left)
            if last and abs(last[1]) >= pc.turned_away_yaw and (
                    T_ms - last[0] <= pc.turned_away_hold_ms or row["n_persons"] > 0):
                row.update(turned_away=True, off_screen_score=1.0, gaze_in_screen_prob=0.0,
                           zone="right" if last[1] > 0 else "left")
            return _clean(row)

        b, bl, sm = self.base, face.blend, cfg.smooth
        dy, dp = face.yaw - b.yaw, face.pitch - b.pitch  # raw: head_speed and TURNED_AWAY never see the filter
        eo, low = float(np.mean(face.eye_open)), False
        if math.isfinite(eo):
            self._eye_ref = eo if self._eye_ref is None else self._eye_ref
            blink, low = eo < _BLINK_FRAC * self._eye_ref, eo < sm.blink_frac * self._eye_ref
            if not blink:
                self._eye_ref += _EYE_ALPHA * (eo - self._eye_ref)
            row["blink"] = float(blink)
        fs = face
        if sm.enabled:
            if (self._last and T_ms - self._last[0] > sm.reset_gap_s * 1000) or (
                    self._pbox and iou(face.bbox, self._pbox) < _SWITCH_IOU):  # gap or a different face: start over
                self._smooth.reset()
                self._lid_t0 = None
            self._pbox = face.bbox
            self._lid_t0 = (T_ms if self._lid_t0 is None else self._lid_t0) if low else None
            hold = low and T_ms - self._lid_t0 < sm.hold_s * 1000  # mid-blink: iris / eye-look values are not trusted
            fs = self._smooth(face, T_ms, hold)
        row.update(zip(("face_x0", "face_y0", "face_x1", "face_y1"), face.bbox))
        row.update(zip(("iris_lx", "iris_ly", "iris_rx", "iris_ry"), fs.iris))
        row.update(zip(("eye_open_l", "eye_open_r"), face.eye_open))
        row.update((col, bl.get(k, nan)) for col, k in EYELOOK)
        row.update(primary_face_present=True, face_size_frac=face.scale, landmark_conf=face.conf,
                   time_since_face_ms=0.0, d_yaw=fs.yaw - b.yaw, d_pitch=fs.pitch - b.pitch, d_roll=fs.roll - b.roll,
                   head_x=fs.center[0] - b.cx, head_y=fs.center[1] - b.cy,
                   head_scale=fs.scale / b.scale, jaw_open=bl.get("jawOpen", nan),
                   mouth_close=bl.get("mouthClose", nan), mar=bl.get("mar", nan))
        g = off_screen(fs, self.calib, cfg.gaze)
        row.update(gaze_x=g.gx, gaze_y=g.gy, gaze_in_screen_prob=g.in_screen_prob, off_screen_score=g.score,
                   zone=g.zone)

        if self._prev and T_ms > self._prev[0]:
            dt = (T_ms - self._prev[0]) / 1000
            row["head_speed"] = math.hypot(dy - self._prev[1], dp - self._prev[2]) / dt
        self._last, self._prev = (T_ms, face.yaw), (T_ms, dy, dp)

        m = [bl.get(k, nan) for k in _MOUTH]
        if all(map(math.isfinite, m)):
            self._mouth.append(m)
            row["mouth_energy_1s"] = math.sqrt(float(np.var(self._mouth, axis=0).sum()))
        return _clean(row)


def _clean(row: dict) -> dict:
    """numpy scalars -> Python scalars (JSON/parquet friendly, real bools)."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
