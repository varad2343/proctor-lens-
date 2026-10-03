"""Calibration procedure: dot schedule + log, and fitting a Calibration from recorded frames.

Times are ms from the start of the calibration recording. Retries (cfg.gaze.max_tries) are the caller's loop:
rerun and `dataclasses.replace(calib, tries=n)`.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from proctorlens.core.config import Config, GazeCfg
from proctorlens.perception.gaze import Calibration, CalibSample, fit_calibration

_GRID = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
_MIN_GRID_DIST = 0.15  # validation dots stay this far from every grid dot
_STABLE_N = 4  # faces in the stability window (~0.13 s at 30 fps)


@dataclass
class CalibDot:
    x: float
    y: float
    phase: str  # 'neutral' | 'grid' | 'validation'
    dot_id: int
    t_start_ms: int
    t_end_ms: int


_QUICK = [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]


def dot_schedule(cfg: GazeCfg, rng=None, mode: str = "full") -> list[CalibDot]:
    """Neutral centre hold, then the grid (row-major), then random validation dots away from the grid.
    mode: full = 3x3 grid + cfg.n_validation dots; quick = centre+corners + 2 dots (~half the time, coarser
    fit); baseline = neutral hold only (head-pose baseline, no gaze regression). rng: seed or Generator.
    dot_id = position in the schedule."""
    grid, n_val_total = {"full": (_GRID, cfg.n_validation), "quick": (_QUICK, min(2, cfg.n_validation)),
                         "baseline": ([], 0)}[mode]
    rng = np.random.default_rng(rng)
    pts = [("neutral", 0.5, 0.5, cfg.neutral_s)] + [("grid", x, y, cfg.dot_s) for x, y in grid]
    n_val = 0
    while n_val < n_val_total:
        x, y = rng.uniform(0.05, 0.95, 2)
        if min(math.dist((x, y), g) for g in grid) >= _MIN_GRID_DIST:
            pts.append(("validation", float(x), float(y), cfg.dot_s))
            n_val += 1
    dots, t = [], 0
    for i, (ph, x, y, dur) in enumerate(pts):
        dots.append(CalibDot(x, y, ph, i, t, t + round(dur * 1000)))
        t = dots[-1].t_end_ms
    return dots


_COLS = ["dot_id", "phase", "x", "y", "t_start_ms", "t_end_ms"]


def write_log(dots: list[CalibDot], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, _COLS)
        w.writeheader()
        w.writerows({k: getattr(d, k) for k in _COLS} for d in dots)


def read_log(path: str | Path) -> list[CalibDot]:
    with open(path, newline="", encoding="utf-8") as f:
        return [CalibDot(float(r["x"]), float(r["y"]), r["phase"], int(r["dot_id"]),
                         int(r["t_start_ms"]), int(r["t_end_ms"])) for r in csv.DictReader(f)]


def calibrate_from_frames(frames: Iterable[tuple[float, np.ndarray]], dots: list[CalibDot], perceiver,
                          cfg: Config) -> Calibration:
    """Keep the primary face of frames with t in [t_start + settle, t_end] of a dot (first settle_s =
    saccade). Every frame goes through the perceiver so its tracking state stays continuous."""
    settle, end = int(cfg.gaze.settle_s * 1000), max((d.t_end_ms for d in dots), default=0)
    samples = []
    for i, (t, frame) in enumerate(frames):
        if t > end:
            break
        p = perceiver.perceive(frame, int(t), i)
        d = next((d for d in dots if d.t_start_ms + settle <= t <= d.t_end_ms), None)
        if d and p.faces:
            samples.append(CalibSample(p.faces[0], d.x, d.y, d.phase, d.dot_id))
    n = {ph: sum(s.phase == ph for s in samples) for ph in ("neutral", "grid", "validation")}
    if not all(n.values()):  # otherwise the failure is silent (error=None, head_pose_only)
        print(f"calibration: no face seen during some dots (face samples per phase: {n}); "
              "sit centred and well lit, looking at each dot", file=sys.stderr)
    return fit_calibration(samples, cfg.gaze)


def collect_adaptive(faces_fn, read, show, cfg: Config, dots: list[CalibDot]):
    """Interactive calibration. Each dot stays up until cfg.gaze.dwell_samples faces were seen after settle_s
    (or dot_timeout_s), not for a fixed time: about 2x faster than the recorded schedule when the face is visible.
    With cfg.gaze.robust only *stable* faces count: yaw and pitch spanned < stable_deg over the last _STABLE_N faces,
    so a sample taken mid-saccade or mid-head-turn does not (comparing with the previous face alone lets a slow
    turn through: measured). The rest are kept aside: if the timeout hits first, the latest of them top the dot up
    to dwell_samples (a restless subject still gets a fit; fit_calibration drops the outliers). robust=False =
    every face counts, the original behaviour.
    faces_fn(frame, t_ms) -> list[Face]; read() -> frame | None; show(dot, n, total, face_seen, frame) -> key code
    (-1 = none). Returns (samples, "done" | "skip" | "quit"); a key press can always end it.
    ponytail: the window counts faces, not seconds, so stable_deg assumes a ~30 fps camera; a subject who has not
    started moving yet looks stable too (settle_s covers that, and fit_calibration drops what slips through)."""
    g, t0 = cfg.gaze, time.monotonic()
    samples, last_ok, recent = [], t0, []
    for n, d in enumerate(dots):
        start, got, loose = time.monotonic(), 0, []
        while True:
            now = time.monotonic()
            frame = read()
            if frame is None:  # tolerate a dropped read, not a dead camera
                if now - last_ok > 2:
                    raise RuntimeError("camera read failed during calibration")
                time.sleep(0.01)
                continue
            last_ok = now
            faces = faces_fn(frame, int((now - t0) * 1000))
            h = [*recent, (faces[0].yaw, faces[0].pitch)] if faces else []  # recent poses (settle window too)
            if faces and now - start >= g.settle_s:
                still = not g.robust or (len(h) == _STABLE_N and np.ptp(h, axis=0).max() < g.stable_deg)
                (samples if still else loose).append(CalibSample(faces[0], d.x, d.y, d.phase, d.dot_id))
                got += still
            recent = h[1 - _STABLE_N:]  # a lost face empties it: no reference until _STABLE_N faces are back
            key = show(d, n, len(dots), bool(faces), frame)
            if key in (27, ord("q")):
                return samples, "quit"
            if key == ord("s"):
                return samples, "skip"
            if got >= g.dwell_samples or now - start >= g.dot_timeout_s:
                break
        if got < g.dwell_samples:
            samples += loose[-(g.dwell_samples - got):]
    return samples, "done"


def _video_frames(path: str | Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        i = 0
        while (r := cap.read())[0]:
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield (t if t > 0 or i == 0 else 1000.0 * i / fps), r[1]
            i += 1
    finally:
        cap.release()


def calibrate_from_video(video_path: str | Path, log_path: str | Path, cfg: Config, perceiver) -> Calibration:
    return calibrate_from_frames(_video_frames(video_path), read_log(log_path), perceiver, cfg)
