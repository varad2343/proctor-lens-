"""The one pipeline: frames -> perception -> 10 Hz feature rows -> scores -> event state machines -> events.
Replay, live and extract-features all call this class, so evaluation describes the demoed system."""
from __future__ import annotations

import bisect
import collections
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from proctorlens import __version__
from proctorlens.core.clock import GridResampler
from proctorlens.core.config import Config, EventCfg
from proctorlens.core.types import FACE_DERIVED, MACHINE_EVENT, Event, Perceived
from proctorlens.explain.attribution import attribute_event
from proctorlens.explain.details import build_details
from proctorlens.explain.overlays import draw
from proctorlens.features.extractor import FeatureExtractor
from proctorlens.features.schema import COLUMNS
from proctorlens.temporal.scores_learned import make_scores
from proctorlens.temporal.state_machine import EventMachine

_ATTR = {"off_screen": "off_screen", "speaking": "speaking"}  # score key -> learned-model target


class Pipeline:
    """process(frame, t_ms) per frame; finish() at end of stream. perceiver needs .perceive(frame, t_ms, idx)."""

    def __init__(self, cfg: Config, perceiver: Any = None, calib: Any = None, provider: Any = None):
        if perceiver is None:
            from proctorlens.perception import Perceiver  # lazy: builds heavy models on first frame only

            perceiver = Perceiver(cfg)
        self.cfg, self.perceiver, self.calib = cfg, perceiver, calib
        self.provider = provider or make_scores(cfg)
        self.extractor = FeatureExtractor(cfg, calib)
        self.grid = GridResampler(cfg.pipeline.grid_hz, cfg.pipeline.max_age_ms)
        detector = f"{cfg.scorer.provider}@{__version__}"
        # keys not in policy.active get no machine (= score 0, never an event); degraded first so it can gate the rest
        keys = sorted(cfg.policy.active, key=lambda k: k != "degraded")
        self.machines = {k: EventMachine(k, cfg.events.get(k, EventCfg()), MACHINE_EVENT[k], self.grid.step, detector)
                         for k in keys}
        self.rows: list[dict] = []  # ponytail: all rows stay in memory (~36k/h); spill to disk for multi-hour live runs
        self.events: list[Event] = []
        self.last_perceived: Perceived | None = None  # for overlays
        self.last_row: dict | None = None
        self._T: list[int] = []  # grid time of each row (bisect index for event windows)
        self._times: collections.deque[int] = collections.deque(maxlen=20)  # recent frame times -> fps
        self._last_t: int | None = None
        self._idx = 0
        self._invalid_run = 0
        self._snap: dict[str, tuple[int, float, dict]] = {}  # key -> (t, peak score, attribution at that peak)
        self._glance_end = 0  # end of the last REPEATED_GLANCING episode
        self._finished = False

    def _fps(self) -> float:
        t = self._times
        return 1000.0 * (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else math.nan

    def process(self, frame_bgr, t_ms: int) -> list[Event]:
        """One frame in; returns events finalized by the grid steps this frame completed."""
        t_ms = int(t_ms)
        if self._last_t is not None and t_ms <= self._last_t:
            return []  # duplicate / out-of-order frame: landmarker VIDEO mode needs monotonic time
        self._last_t = t_ms
        self._times.append(t_ms)
        self.last_perceived = p = self.perceiver.perceive(frame_bgr, t_ms, self._idx)
        self._idx += 1
        done: list[Event] = []
        for T, item, age in self.grid.push(t_ms, p):
            done += self._step(T, item, age)
        return done

    def tick(self, t_ms: int) -> list[Event]:
        """Live only: no frame arrived (camera stalled or unplugged). Once the last frame is older than
        max_age_ms, advance the grid with invalid steps so a stall becomes frame_gap -> MONITORING_DEGRADED."""
        t_ms = int(t_ms)
        if self._last_t is None or t_ms - self._last_t <= self.cfg.pipeline.max_age_ms:
            return []
        self._last_t = t_ms
        done: list[Event] = []
        for T, item, age in self.grid.push(t_ms, None):
            done += self._step(T, item, age)
        return done

    def _step(self, T: int, p: Perceived | None, age: float) -> list[Event]:
        row = self.extractor.step(T, p, age, self._fps())
        self._invalid_run = 0 if row["frame_valid"] else self._invalid_run + 1
        if self._invalid_run * self.grid.step >= self.cfg.pipeline.stall_ms:  # stalled stream = blindness, not absence
            row["quality_reasons"], row["reliable"] = "frame_gap", False
        self.rows.append(row)
        self._T.append(T)
        self.last_row = row
        scores = self.provider.update(row)
        deg, done = self.machines.get("degraded"), []
        for k, m in self.machines.items():
            s = scores.get(k)
            if k in FACE_DERIVED and deg is not None and deg.ongoing() is not None:
                s = None  # cross-event suppression while monitoring is degraded
            done += [self._finalize(ev, k) for ev in m.update(T, s)]
            if k in _ATTR:
                self._snapshot(k, m, s, T)
        return done

    def _snapshot(self, k: str, m: EventMachine, s: float | None, T: int) -> None:
        """Attribution must describe the event, not the quiet window after it (an ended event waits merge_gap
        before it is emitted): keep the one taken at the highest score while the event was ongoing."""
        ev = m.ongoing()
        if ev is None or s is None:
            return
        _, top, _ = cur = self._snap.get(k, (0, -1.0, {}))
        if cur[0] < ev.start_ms:
            top = -1.0  # left over from an earlier event that was dropped (too short)
        if s > top + 0.02 and (a := attribute_event(self.provider, _ATTR[k])):  # only re-score on a clearly higher peak
            self._snap[k] = (T, s, a)

    def _finalize(self, ev: Event, k: str) -> Event:
        ev.status = "final"
        # glancing excursions happened before the event start (it fires when the n-th one completes)
        lo = ev.start_ms - (int(self.cfg.glance.window_s * 1000) if k == "glancing" else 0)
        i, j = bisect.bisect_left(self._T, lo), bisect.bisect_left(self._T, ev.end_ms)  # end_ms is exclusive
        ev.details = build_details(ev, self.rows[i:j], self.cfg, self.calib)
        if k == "glancing" and (ex := ev.details["excursions"]):
            # the score stays 1 until the glances slide out of the window; the labeled episode (docs/DATA_PROTOCOL.md) runs
            # from the first glance to the end of the last: back-date the start, trim the tail, keep episodes disjoint
            ev.start_ms = max(ex[0]["start_ms"], self._glance_end)
            ev.end_ms = self._glance_end = max(ev.start_ms, ex[-1]["end_ms"])
            ev.details["duration_s"] = round((ev.end_ms - ev.start_ms) / 1000, 2)
        snap = self._snap.pop(k, None)
        if snap and ev.start_ms <= snap[0] < ev.end_ms:
            ev.attribution = snap[2]
        elif k in _ATTR:  # ponytail: no in-event snapshot (e.g. window not full then) => window at finalization time
            ev.attribution = attribute_event(self.provider, _ATTR[k])
        self.events.append(ev)
        return ev

    def ongoing(self) -> list[Event]:
        return [e for m in self.machines.values() if (e := m.ongoing()) is not None]

    def finish(self, t_ms: int | None = None) -> list[Event]:
        """End of stream: close open events; returns ALL events sorted by start."""
        if not self._finished:
            self._finished = True
            t = t_ms if t_ms is not None else (self._T[-1] if self._T else 0)
            for k, m in self.machines.items():
                for ev in m.close(t):
                    self._finalize(ev, k)
            self.events.sort(key=lambda e: (e.start_ms, e.end_ms, e.type))
        return list(self.events)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)


def video_frames(cap):
    """(t_ms, frame) for every frame of an open capture. The one timestamp rule for replay and evidence keyframes:
    container time, or idx * 1000 / fps when the container gives no usable (increasing) timestamp."""
    fps, idx, last = cap.get(cv2.CAP_PROP_FPS) or 30.0, 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        t = round(cap.get(cv2.CAP_PROP_POS_MSEC))
        if t <= last:
            t = max(last + 1, round(idx * 1000 / fps))
        yield t, frame
        last, idx = t, idx + 1


def run_video(path: str | Path, cfg: Config, calib: Any = None, perceiver: Any = None,
              render_path: str | Path | None = None, progress: bool = False) -> tuple[pd.DataFrame, list[Event]]:
    """Replay a video file through Pipeline. Writes nothing unless render_path (annotated video) is given."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pl, writer = Pipeline(cfg, perceiver, calib), None
    try:
        for idx, (t, frame) in enumerate(video_frames(cap), 1):
            pl.process(frame, t)
            if render_path:
                img = draw(frame, pl.last_perceived, pl.last_row, pl.ongoing())
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if str(render_path).endswith(".avi") else "mp4v"))
                    writer = cv2.VideoWriter(str(render_path), fourcc, fps, (img.shape[1], img.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot write video: {render_path}")
                writer.write(img)
            if progress and idx % 50 == 0:
                print(f"\rframe {idx}/{n or '?'}", end="", file=sys.stderr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if progress:
        print(file=sys.stderr)
    events = pl.finish()
    return pl.frame(), events
