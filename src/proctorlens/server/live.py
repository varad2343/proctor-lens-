"""Server side of one candidate connection (master spec 10.3, 11.1): environment check -> gaze calibration -> exam,
fed JPEG frames from the browser. The exam runs the same Pipeline as replay and live; this adds the client clock, an
in-memory ring buffer for event evidence (nothing else of the video is kept), and browser telemetry
(BROWSER_INTEGRITY). Every public method returns (messages for the candidate, messages for proctors) and must run on
`self.worker`: one thread per session serializes frames and control messages and keeps the models on one thread."""
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import datetime as dt
import json
import logging
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

from proctorlens import __version__
from proctorlens.cli import summarize
from proctorlens.core.config import Config
from proctorlens.core.types import BROWSER_INTEGRITY, Event
from proctorlens.explain.evidence import caption, clip_window, keyframe_times, write_clip_jpegs
from proctorlens.explain.review import explain, ts
from proctorlens.io import make_header, save_events, save_features
from proctorlens.perception.gaze import CalibSample, fit_calibration
from proctorlens.perception.quality import assess
from proctorlens.pipeline.calibration import dot_schedule
from proctorlens.pipeline.runner import Pipeline
from proctorlens.server.db import DB

log = logging.getLogger("proctorlens.server")
RING_MS = 40_000  # ponytail: evidence comes from the last ~40 s of frames in memory; a longer event keeps its tail
STATUS_MS = 500  # status message rate (candidate + proctors)
GUIDE = {"dark": "Add light in front of you", "bright": "Reduce backlight or glare behind you",
         "blur": "Hold still, or clean the camera lens", "small_face": "Move closer to the camera",
         "face_cut": "Centre your face in the picture", "no_face": "Face the camera", "blocked": "Uncover the camera"}
_OPEN = {"hidden": "visible", "blur": "focus", "fullscreen_exit": "fullscreen_enter", "offline": "online"}
_CLOSE = {v: k for k, v in _OPEN.items()}  # closing kind -> the opening kind it ends
_ONESHOT = ("paste", "copy", "contextmenu", "beforeunload")
EVIDENCE = concurrent.futures.ThreadPoolExecutor(2, thread_name_prefix="evidence")  # encoding never blocks frames


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _log_failure(f: concurrent.futures.Future) -> None:
    if (e := f.exception()) is not None:
        log.error("evidence encoding failed", exc_info=e)


class LiveSession:
    def __init__(self, sid: str, cfg: Config, folder: str | Path, db: DB, landmarker_factory, perceiver_factory):
        self.sid, self.cfg, self.dir, self.db = sid, cfg, Path(folder), db
        self._lm_factory, self._perc_factory = landmarker_factory, perceiver_factory
        self.worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=f"session-{sid}")
        self.phase = "check"  # check <-> calib, then exam, then ended
        self.lm = None  # one VIDEO-mode landmarker for the whole session (timestamps stay monotonic)
        self.t0: float | None = None  # client clock (epoch ms) of the first message = session time 0
        self.last_t: int | None = None
        self.last_mono = 0.0
        self.latest_jpeg: bytes | None = None  # proctor live view
        self.schedule, self.dot, self.dot_t, self.samples = [], None, 0, []
        self.calib, self.tries = None, 0
        self.pl: Pipeline | None = None
        self.ring: collections.deque[tuple[int, bytes]] = collections.deque()
        self.jobs: list[tuple[int, Event]] = []  # (event id, event) waiting for frames up to the clip end
        self.futures: list[concurrent.futures.Future] = []
        self.open: dict[str, int] = {}  # open browser interval: opening kind -> start t
        self.extra: list[Event] = []  # BROWSER_INTEGRITY events (not from the pipeline)
        self._stored: set[int] = set()  # id() of events already in the DB
        self._ongoing: set[tuple[str, int]] = set()
        self._status_t = -math.inf

    def _t(self, t_client: float) -> int:
        """Session ms from the client's epoch-ms clock (performance.timeOrigin + now(): monotonic within a page and
        comparable across a reload, so a reconnect continues the same timeline and the gap reads as a frame gap)."""
        if self.t0 is None:
            self.t0 = float(t_client)
        return round(float(t_client) - self.t0)

    # ---------------------------------------------------------------- frames

    def frame(self, seq: int, t_client: float, jpeg: bytes):
        ack, t = [{"type": "ack", "seq": seq}], self._t(t_client)
        if self.phase == "ended" or (self.last_t is not None and t <= self.last_t):
            return ack, []  # duplicate / out of order
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ack, []
        self.last_t, self.last_mono, self.latest_jpeg = t, time.monotonic(), jpeg
        if self.lm is None:
            self.lm = self._lm_factory(self.cfg)
        extra: dict = {}
        if self.phase == "exam":
            self.ring.append((t, jpeg))
            while t - self.ring[0][0] > RING_MS:
                self.ring.popleft()
            out = self._events(self.pl.process(img, t))
            self._flush(t)
            row = self.pl.last_row or {}
            q, reasons = row.get("quality"), [r for r in re.split(r"[,;|\s]+", str(row.get("quality_reasons") or "")) if r]
            extra = {"zone": row.get("zone"), "fps": row.get("effective_fps")}
        else:
            faces, out = self.lm.process(img, t), []
            r = assess(img, faces[0].bbox if faces else None, self.cfg.quality)
            q, reasons = r.quality, r.reasons
            if self.phase == "calib" and self.dot and faces and t >= self.dot_t + self.cfg.gaze.settle_s * 1000:
                d = self.dot  # first settle_s of a dot = saccade, not a sample
                self.samples.append(CalibSample(faces[0], d.x, d.y, d.phase, d.dot_id))
        if t - self._status_t < STATUS_MS:
            return ack, out
        self._status_t = t
        st = {"phase": self.phase, "quality": q, "reasons": reasons, "guidance": {r: GUIDE[r] for r in reasons if r in GUIDE}}
        # ack last: a client that waits for it has every message this frame produced
        return [{"type": "status", **st}, *ack], out + [{"type": "session_status", "session_id": self.sid, **st, **extra}]

    def tick(self):
        """No frame lately (camera stalled, tab throttled): advance the grid on the server clock, so a stall becomes
        frame_gap -> MONITORING_DEGRADED while it happens, not only once frames resume."""
        if self.phase != "exam" or self.last_t is None:
            return [], []
        t = self.last_t + round((time.monotonic() - self.last_mono) * 1000)
        out = self._events(self.pl.tick(t))
        self._flush(t)
        return [], out

    # ---------------------------------------------------------------- control messages

    def control(self, m: dict):
        k = m.get("type")
        if k == "hello":
            return [{"type": "hello", "phase": self.phase, "calibration": self.calib.mode if self.calib else "none",
                     "tries": self.tries, "max_tries": self.cfg.gaze.max_tries,
                     "enroll_ms": int(self.cfg.pipeline.enroll_seconds * 1000)}], []
        if k == "ping":
            return [{"type": "pong"}], []
        if k == "browser_event":
            return [], self._browser(str(m["kind"]), float(m["t"]), str(m.get("detail") or "")[:200])
        if k == "exam_end" and self.phase == "exam":
            return [{"type": "ended"}], self.end(m.get("answers"))
        if self.phase not in ("check", "calib"):
            return [], []  # setup messages after the exam started are ignored
        if k == "calib_start":
            mode = m.get("mode") if m.get("mode") in ("full", "quick", "baseline") else "full"
            self.phase, self.samples, self.dot, self.schedule = "calib", [], None, dot_schedule(self.cfg.gaze, mode=mode)
            return [{"type": "calib_schedule", "dots": [
                {"dot_id": d.dot_id, "x": d.x, "y": d.y, "phase": d.phase, "dur_ms": d.t_end_ms - d.t_start_ms}
                for d in self.schedule]}], []
        if k == "calib_point" and self.phase == "calib" and 0 <= int(m["dot_id"]) < len(self.schedule):
            self.dot, self.dot_t = self.schedule[int(m["dot_id"])], self._t(float(m["t"]))
            return [], []
        if k == "calib_end" and self.phase == "calib":
            self.phase, self.dot, self.tries = "check", None, self.tries + 1
            if any(s.phase in ("neutral", "grid") for s in self.samples):
                self.calib = dataclasses.replace(fit_calibration(self.samples, self.cfg.gaze), tries=self.tries)
                self.db.run("UPDATE sessions SET calibration_json=?, mode=? WHERE id=?",
                            json.dumps(self.calib.to_dict()), self.calib.mode, self.sid)
            c = self.calib
            return [{"type": "calib_result", "mode": c.mode if c else "none", "accepted": bool(c and c.accepted),
                     "error": c.error if c else None, "tries": self.tries, "max_tries": self.cfg.gaze.max_tries,
                     "n_samples": len(self.samples)}], []
        if k == "exam_start":
            return self._start()
        return [], []

    def _start(self):
        if self.lm is None:
            self.lm = self._lm_factory(self.cfg)
        perc = self._perc_factory(self.cfg, self.lm)
        # load the detector / identity models now: a multi-second stall on the first exam frame would read as a gap
        perc.perceive(np.zeros((480, 640, 3), np.uint8), (self.last_t or 0) + 1, 0)
        self.pl, self.phase = Pipeline(self.cfg, perc, self.calib), "exam"
        mode = self.calib.mode if self.calib else "none"
        self.db.run("UPDATE sessions SET status='live', started_at=?, mode=? WHERE id=?", now_iso(), mode, self.sid)
        return ([{"type": "exam_started", "enroll_ms": int(self.cfg.pipeline.enroll_seconds * 1000)}],
                [{"type": "session_status", "session_id": self.sid, "phase": "exam"}])

    # ---------------------------------------------------------------- events + evidence

    def _store(self, ev: Event) -> dict:
        eid = self.db.add_event(self.sid, ev, explain(ev))
        self._stored.add(id(ev))
        self.jobs.append((eid, ev))
        return {"type": "event_ended", "session_id": self.sid,
                "event": {"id": eid, "type": ev.type, "start_ms": ev.start_ms, "end_ms": ev.end_ms}}

    def _events(self, done: list[Event]) -> list[dict]:
        """Finalized events -> DB + evidence job; ongoing ones -> event_started for the live feed."""
        out = [self._store(e) for e in done]
        now = {(e.type, e.start_ms) for e in self.pl.ongoing()}
        out += [{"type": "event_started", "session_id": self.sid, "event": {"type": ty, "start_ms": s}}
                for ty, s in sorted(now - self._ongoing)]
        self._ongoing = now
        return out

    def _flush(self, t: int, force: bool = False) -> None:
        """Start encoding every job whose clip window has passed (all of them when force)."""
        wait = []
        for eid, ev in self.jobs:
            a, b = clip_window(ev.start_ms, ev.end_ms)
            if t < b and not force:
                wait.append((eid, ev))
                continue
            f = EVIDENCE.submit(self._evidence, eid, ev, [x for x in self.ring if a <= x[0] < b])
            f.add_done_callback(_log_failure)
            self.futures.append(f)
        self.jobs, self.futures = wait, [f for f in self.futures if not f.done()]

    def _evidence(self, eid: int, ev: Event, frames: list[tuple[int, bytes]]) -> None:
        """Keyframes (onset / peak / end) + H.264 clip from ring-buffer JPEGs -> evidence/<event id>*; paths -> row."""
        thumbs, clip = {}, None
        if frames:
            (self.dir / "evidence").mkdir(parents=True, exist_ok=True)
            for name, t in keyframe_times(ev, 1000 // self.cfg.pipeline.grid_hz).items():
                img = cv2.imdecode(np.frombuffer(min(frames, key=lambda x: abs(x[0] - t))[1], np.uint8), cv2.IMREAD_COLOR)
                p = thumbs[name] = f"evidence/{eid}_{name}.jpg"
                (self.dir / p).write_bytes(cv2.imencode(".jpg", caption(img, f"{ev.type}  {name}  {ts(t)}"))[1].tobytes())
            span = (frames[-1][0] - frames[0][0]) / 1000
            if write_clip_jpegs([j for _, j in frames], (len(frames) - 1) / span if span > 0 else 10.0,
                                self.dir / f"evidence/{eid}.mp4"):
                clip = f"evidence/{eid}.mp4"
        self.db.run("UPDATE events SET clip_path=?, thumbs_json=? WHERE id=?", clip, json.dumps(thumbs), eid)

    def _browser(self, kind: str, t_client: float, detail: str) -> list[dict]:
        """Raw telemetry -> browser_events; during the exam, intervals (hidden..visible, blur..focus, fullscreen exit..
        enter, offline..online) and one-shot actions (paste, copy, context menu, unload) -> BROWSER_INTEGRITY."""
        t = self._t(t_client)
        self.db.run("INSERT INTO browser_events (session_id, kind, t_ms, detail) VALUES (?,?,?,?)", self.sid, kind, t, detail)
        if self.phase != "exam":
            return []
        if kind in _OPEN:
            self.open.setdefault(kind, t)
        elif kind in _CLOSE and (s := self.open.pop(_CLOSE[kind], None)) is not None:
            return [self._browser_event(_CLOSE[kind], s, t)]
        elif kind in _ONESHOT:
            return [self._browser_event(kind, t, t)]
        return []

    def _browser_event(self, kind: str, a: int, b: int) -> dict:
        b = max(a, b)
        ev = Event(BROWSER_INTEGRITY, a, b, 1.0, f"browser@{__version__}",
                   {"key": "browser", "kind": kind, "duration_s": round((b - a) / 1000, 2)})
        self.extra.append(ev)
        return self._store(ev)

    # ---------------------------------------------------------------- end

    def end(self, answers=None) -> list[dict]:
        """Close open events and browser intervals, finish the evidence, write features / events / summary files."""
        out: list[dict] = []
        if self.pl is not None:
            t = self.last_t or 0
            out += [self._store(e) for e in self.pl.finish() if id(e) not in self._stored]
            out += [self._browser_event(k, s, t) for k, s in self.open.items()]
            self.open.clear()
            self._flush(t, force=True)
            concurrent.futures.wait(self.futures)
            df = self.pl.frame()
            save_features(df, self.dir / "features.parquet")
            save_events(sorted(self.pl.events + self.extra, key=lambda e: (e.start_ms, e.end_ms, e.type)),
                        self.dir / "events.json", make_header(self.cfg, self.calib))
            (self.dir / "summary.json").write_text(json.dumps(summarize(df, self.pl.events, self.calib), indent=1,
                                                              allow_nan=False), encoding="utf-8")
        self.phase = "ended"
        self.db.run("UPDATE sessions SET status='ended', ended_at=?, answers_json=? WHERE id=?",
                    now_iso(), json.dumps(answers), self.sid)
        return out + [{"type": "session_status", "session_id": self.sid, "phase": "ended"}]
