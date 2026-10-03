"""OpenCV overlay: face box, head-pose axes, gaze arrow + zone, object boxes, mouth dot, quality banner,
ongoing-event ticker. Draws observations only. For error analysis and the demo (`replay --render`, live window)."""
from __future__ import annotations

import math

import cv2
import numpy as np

from proctorlens.core.types import Event, Perceived

# BGR
_GREEN, _GRAY, _RED = (80, 200, 80), (150, 150, 150), (60, 60, 230)
_AMBER, _BLUE, _WHITE = (0, 180, 255), (230, 140, 0), (255, 255, 255)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def draw(frame_bgr: np.ndarray, perceived: Perceived | None, row: dict | None, ongoing: list[Event], *,
         fps: float | None = None, zone: str | None = None, calib_mode: str | None = None) -> np.ndarray:
    """Return an annotated copy of frame_bgr; perceived/row may be None (frame gap).
    fps / zone / calib_mode (all optional) add a small HUD strip under the banner; none given = the old picture."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    fs, th = max(0.35, h / 700), max(1, h // 240)
    px = lambda x, y: (int(x * w), int(y * h))  # noqa: E731  normalized -> pixels
    text = lambda s, org, col: cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)  # noqa: E731
    row = row or {}
    reasons = ""
    quality = math.nan
    if perceived is not None:
        quality, reasons = perceived.quality, ",".join(perceived.quality_reasons)
        for name, conf, b in perceived.det.boxes:
            cv2.rectangle(img, px(b[0], b[1]), px(b[2], b[3]), _AMBER, th)
            text(f"{name} {conf:.2f}", px(b[0], b[1]), _AMBER)
        for i, f in enumerate(perceived.faces):
            cv2.rectangle(img, px(f.bbox[0], f.bbox[1]), px(f.bbox[2], f.bbox[3]),
                          _GREEN if i == 0 and not f.static else _GRAY, th)
            if i:
                continue
            o, L = np.array([f.center[0] * w, f.center[1] * h]), f.scale * h * 0.6
            y, p, r = np.radians([f.yaw, f.pitch, f.roll])  # convention: core/types.py Face
            for d, c in (((math.sin(y) * math.cos(p), -math.sin(p)), _BLUE),  # forward (nose)
                         ((math.cos(r), math.sin(r)), _RED), ((math.sin(r), -math.cos(r)), _GREEN)):  # right, up
                cv2.arrowedLine(img, (int(o[0]), int(o[1])), (int(o[0] + L * d[0]), int(o[1] + L * d[1])), c, th)
            gx, gy = _num(row.get("gaze_x")), _num(row.get("gaze_y"))
            if math.isfinite(gx) and math.isfinite(gy):  # where on the screen the calibrated model says attention is
                cv2.arrowedLine(img, (int(o[0]), int(o[1])),
                                px(f.center[0] + (gx - 0.5) * 0.3, f.center[1] + (gy - 0.5) * 0.3), _AMBER, th)
            if "zone" in row:
                text(f"{row['zone']} {_num(row.get('off_screen_score')):.2f}", px(f.bbox[0], f.bbox[3]), _WHITE)
            e = _num(row.get("mouth_energy_1s"))  # ponytail: assumes energy in ~[0,1]; dot radius = activity
            if math.isfinite(e):
                cv2.circle(img, px(f.bbox[2], f.bbox[3]), 2 + int(6 * min(1.0, e)), _AMBER if e >= 0.5 else _GRAY, -1)
    if "quality" in row:
        quality, reasons = _num(row["quality"]), str(row.get("quality_reasons") or reasons)
    bad = ("reliable" in row and not row["reliable"]) or (perceived is None and not row)
    bar = 4 + int(24 * fs)
    cv2.rectangle(img, (0, 0), (w, bar), _RED if bad else (40, 40, 40), -1)
    text(("LOW QUALITY " if bad else "") + f"q={quality:.2f} {reasons}", (2, bar - 4), _WHITE)
    hud = [s for s in (f"{fps:.1f} fps" if math.isfinite(_num(fps)) else "", f"zone {zone}" if zone else "",
                       f"calib {calib_mode}" if calib_mode else "") if s]
    if hud:
        strip = img[bar + 1:2 * bar + 1]  # rows bar+1 .. 2*bar, a view: glyphs are clipped to it, never spill into the banner
        strip[:] = (40, 40, 40)
        cv2.putText(strip, "  ".join(hud), (2, bar - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, _WHITE, th, cv2.LINE_AA)
    for i, e in enumerate(ongoing):
        text(f"{e.type} {(e.end_ms - e.start_ms) / 1000:.1f}s", (2, h - 4 - i * bar), _AMBER)
    return img
