"""Per-frame image quality (classical CV, no training).

quality = min over soft component scores; every component is 0.5 exactly at its config threshold and 1.0
at twice the threshold (so the default min_quality=0.5 agrees with the reason codes). `no_face` is a code
only: it does not lower quality, otherwise FACE_ABSENT could never be told apart from blindness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from proctorlens.core.config import QualityCfg
from proctorlens.core.types import Box


@dataclass
class QualityResult:
    luma: float
    blur: float  # variance of Laplacian on the face ROI (full frame when no face)
    overexp_frac: float
    face_frac: float  # face-box height / frame height (0 when no face)
    quality: float  # [0,1]
    reasons: list[str] = field(default_factory=list)  # dark, bright, blur, small_face, face_cut, no_face, blocked


def _s(x: float, thr: float) -> float:
    """Higher-is-better soft score: 0.5 at thr, 1 at 2*thr."""
    return float(np.clip(x / (2 * max(thr, 1e-9)), 0.0, 1.0))


def assess(frame_bgr: np.ndarray, face_bbox: Box | None, cfg: QualityCfg) -> QualityResult:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(g)
    luma, sd = float(mean[0, 0]), float(std[0, 0])
    over = float(np.count_nonzero(g >= 250)) / g.size
    roi, face_frac, cut = g, 0.0, False
    if face_bbox is not None:
        x0, y0, x1, y1 = face_bbox
        face_frac, cut = y1 - y0, min(x0, y0) < 0.005 or max(x1, y1) > 0.995
        h, w = g.shape
        a, b, c, d = (np.clip(face_bbox, 0, 1) * [w, h, w, h]).astype(int)
        if d - b >= 8 and c - a >= 8:
            roi = g[b:d, a:c]
    blur = float(cv2.Laplacian(roi, cv2.CV_64F).var())

    comps = [_s(luma, cfg.min_luma), _s(255 - luma, 255 - cfg.max_luma),
             1 - _s(over, cfg.overexp_max), _s(blur, cfg.min_blur)]
    small = face_bbox is not None and face_frac < cfg.min_face_frac
    blocked = sd < cfg.blocked_luma_std
    if face_bbox is not None:
        comps.append(_s(face_frac, cfg.min_face_frac))
    if cut:
        comps.append(0.25)
    if blocked:
        comps.append(0.0)
    reasons = [r for r, on in (
        ("dark", luma < cfg.min_luma), ("bright", luma > cfg.max_luma or over > cfg.overexp_max),
        ("blur", blur < cfg.min_blur), ("small_face", small), ("face_cut", cut),
        ("no_face", face_bbox is None), ("blocked", blocked)) if on]
    return QualityResult(luma, blur, over, max(face_frac, 0.0), min(comps), reasons)
