"""End to end on a scripted fake perceiver: frames -> features -> scores -> state machines -> events.json."""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from proctorlens import cli
from proctorlens.core.config import Config
from proctorlens.core.types import Detections, Face, Perceived
from proctorlens.features.schema import COLUMNS
from proctorlens.io import load_events, make_header, save_events
from proctorlens.perception.gaze import Baseline, Calibration
from proctorlens.pipeline.runner import Pipeline, run_video
from proctorlens.temporal.scores_rule import RuleScores

_BLEND = dict.fromkeys(("jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
                        *(f"eyeLook{d}{s}" for d in ("In", "Out", "Up", "Down") for s in ("Left", "Right"))), 0.0)
_QUALITY = {"dark": (10.0, 0.0, 0.05, ["dark", "no_face"]), "absent": (120.0, 200.0, 0.9, ["no_face"])}
# (end_s, scene): normal, phone 3 s, normal, no face 5 s, normal, 2 faces 3 s, normal, head turned 6 s, normal, dark to the end
SCRIPT = [(10, "normal"), (13, "phone"), (18, "normal"), (23, "absent"), (28, "normal"), (31, "two"),
          (36, "normal"), (42, "turn"), (47, "normal"), (60, "dark")]
EXPECTED = {"PROHIBITED_OBJECT": (10, 13), "FACE_ABSENT": (18, 23), "MULTIPLE_PEOPLE": (28, 31),
            "OFF_SCREEN_SUSTAINED": (36, 42), "MONITORING_DEGRADED": (47, 60)}
CALIB = Calibration(Baseline(0.0, 0.0, 0.0, 0.5, 0.5, 0.4), None, float("nan"), "head_pose_only", False)


def _face(cx=0.5, yaw=0.0):
    return Face(bbox=(cx - 0.15, 0.3, cx + 0.15, 0.7), yaw=yaw, pitch=0.0, roll=0.0, iris=(0.5,) * 4,
                eye_open=(0.6, 0.6), blend=dict(_BLEND), center=(cx, 0.5), scale=0.4)


class FakePerceiver:
    """No models: the scene is a function of time only (so runs are repeatable)."""

    def __init__(self, script=SCRIPT):
        self.script = script

    def perceive(self, frame, t_ms, idx):
        scene = next(sc for end, sc in self.script if t_ms / 1000 < end)
        luma, blur, q, reasons = _QUALITY.get(scene, (120.0, 200.0, 0.9, []))
        faces = {"normal": [_face()], "phone": [_face()], "two": [_face(), _face(0.8)], "turn": [_face(yaw=35.0)]}.get(scene, [])
        det = Detections(0.9, 0.0, [], [("phone", 0.9, (0.6, 0.5, 0.8, 0.8))]) if scene == "phone" else Detections()
        return Perceived(t_ms, faces, det, None, luma, blur, 0.0, q, reasons)


def _run(calib=CALIB, script=SCRIPT, n=600):
    pl, frame, emitted = Pipeline(Config(), FakePerceiver(script), calib), np.zeros((48, 64, 3), np.uint8), []
    for i in range(n):  # 10 fps
        emitted += pl.process(frame, i * 100)
    return pl, emitted


def test_scripted_session_events():
    pl, emitted = _run()
    # the dark stretch runs to the end of the stream: MONITORING_DEGRADED is still open before finish()
    assert [e.type for e in pl.ongoing()] == ["MONITORING_DEGRADED"] and pl.ongoing()[0].status == "ongoing"
    assert sorted(e.type for e in emitted) == sorted(EXPECTED.keys() - {"MONITORING_DEGRADED"})
    events = pl.finish()
    assert pl.ongoing() == [] and pl.finish() == events  # finish closes open events, and is idempotent
    assert [e.type for e in events if e not in emitted] == ["MONITORING_DEGRADED"]
    assert sorted(e.type for e in events) == sorted(EXPECTED)  # one of each, nothing else (no glancing, speaking, ...)
    assert [e.start_ms for e in events] == sorted(e.start_ms for e in events)
    for e in events:
        s, t = EXPECTED[e.type]
        assert abs(e.start_ms - s * 1000) <= 1000 and abs(e.end_ms - t * 1000) <= 1000, (e.type, e.start_ms, e.end_ms)
        assert e.status == "final" and e.end_ms >= e.start_ms and 0 < e.confidence <= 1 and e.detector.startswith("rule@")
        assert e.attribution is None  # rule provider
        assert e.details["signal"] and e.details["duration_s"] > 0 and e.details["calib_mode"] == "head_pose_only"
        assert e.details["quality_mean"] is not None and "reduced_calibration" in e.details["benign_flags"]
        json.dumps(e.to_dict(), allow_nan=False)
    by = {e.type: e for e in events}
    assert by["OFF_SCREEN_SUSTAINED"].details["zone"] == "right" and by["OFF_SCREEN_SUSTAINED"].details["d_yaw_deg"] == 35.0
    assert by["PROHIBITED_OBJECT"].details["object"] == "phone"
    assert by["MONITORING_DEGRADED"].details["reasons"][0] == "dark"
    # dark = blindness, never absence: no FACE_ABSENT after the lights go out, and those rows are unreliable
    assert not any(e.type == "FACE_ABSENT" and e.end_ms > 48000 for e in events)
    assert all(not r["reliable"] for r in pl.rows if r["t_ms"] >= 47000)
    df = pl.frame()
    assert list(df.columns) == COLUMNS and len(df) == len(pl.rows) >= 590


def test_track_numbers_follow_boxes_and_restart_after_absence():
    def tracks():
        pl, frame, seen = Pipeline(Config(), FakePerceiver(), CALIB), np.zeros((48, 64, 3), np.uint8), {}
        for i in range(600):
            pl.process(frame, i * 100)
            if i in (50, 110, 250, 300):
                p = pl.last_perceived
                seen[i] = ([f.track for f in p.faces], list(p.det.ids))
        return seen

    s = tracks()
    assert s[50] == ([1], []) and s[110] == ([1], [1])  # the same face throughout; the phone at 11 s
    assert s[250] == ([2], [])  # back after the 5 s absence: a new number, never the old one
    assert s[300] == ([2, 3], [])  # the second person gets the next number
    assert tracks() == s  # deterministic


def test_replay_is_deterministic():
    (a, _), (b, _) = _run(), _run()
    ea, eb = a.finish(), b.finish()
    d = Path(tempfile.mkdtemp())
    for name, evs, pl in (("a.json", ea, a), ("b.json", eb, b)):
        save_events(evs, d / name, make_header(pl.cfg, CALIB))
    assert (d / "a.json").read_text() == (d / "b.json").read_text()
    assert a.frame().equals(b.frame())
    assert load_events(d / "a.json")[1] == ea  # events.json loads back to the same events


class AttrProvider:
    """Rule scores + an `attribute` hook that reports when it was called (stands in for LearnedScores)."""

    def __init__(self, cfg):
        self.rule, self.t = RuleScores(cfg), 0

    def update(self, row):
        self.t = row["t_ms"]
        return self.rule.update(row)

    def attribute(self, target):
        return {"head_pose": float(self.t)}


def test_attribution_describes_the_event_not_the_window_after_it():
    cfg, frame = Config(), np.zeros((48, 64, 3), np.uint8)
    pl = Pipeline(cfg, FakePerceiver(), CALIB, provider=AttrProvider(cfg))
    for i in range(600):
        pl.process(frame, i * 100)
    by = {e.type: e for e in pl.finish()}
    off = by["OFF_SCREEN_SUSTAINED"]
    assert off.start_ms <= off.attribution["head_pose"] <= off.end_ms  # snapshot at the peak, inside the event
    assert by["PROHIBITED_OBJECT"].attribution is None and by["FACE_ABSENT"].attribution is None  # not learned targets


def test_camera_stall_becomes_degraded_not_absent():
    pl, frame = Pipeline(Config(), FakePerceiver([(99, "normal")]), CALIB), np.zeros((48, 64, 3), np.uint8)
    for i in range(51):  # frames until 5.0 s, then the camera goes silent
        pl.process(frame, i * 100)
    done = []
    for t in range(5050, 9001, 50):  # live loop ticking with no frames
        done += pl.tick(t)
    assert [e.type for e in pl.ongoing()] == ["MONITORING_DEGRADED"] and done == []
    assert pl.rows[-1]["quality_reasons"] == "frame_gap" and not pl.rows[-1]["reliable"]
    (ev,) = pl.finish()
    assert ev.type == "MONITORING_DEGRADED" and abs(ev.start_ms - 5300) <= 500 and ev.end_ms >= 8000


def _write_avi(path, n_frames):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert w.isOpened()
    for i in range(n_frames):
        w.write(np.full((48, 64, 3), i % 200, np.uint8))
    w.release()


def test_run_video_and_replay_cli_write_only_features_and_events():
    d = Path(tempfile.mkdtemp())
    _write_avi(d / "v.avi", 80)  # 8 s, nobody in view, good image quality => FACE_ABSENT
    fake = FakePerceiver([(99, "absent")])
    df, events = run_video(d / "v.avi", Config(), perceiver=fake, render_path=d / "r.avi")
    assert list(df.columns) == COLUMNS and 70 <= len(df) <= 81
    assert (np.diff(df.t_ms) == 100).all()  # container timestamps -> a clean 10 Hz grid
    assert [e.type for e in events] == ["FACE_ABSENT"] and events[0].status == "final" and events[0].details["signal"]
    assert cv2.VideoCapture(str(d / "r.avi")).read()[0]  # the overlay render is a playable video
    # the CLI path: no --render => exactly features + events, nothing else; calibration is recorded in the header
    CALIB.save(d / "c.json")
    out = d / "out"
    args = cli.build_parser().parse_args(["replay", str(d / "v.avi"), "--out", str(out), "--calib", str(d / "c.json")])
    assert cli._replay(args, perceiver=fake) == 0
    assert sorted(p.name for p in out.iterdir()) in (["events.json", "features.csv", "summary.json"],
                                                     ["events.json", "features.parquet", "summary.json"])  # + run summary
    header, evs = load_events(out / "events.json")
    assert header["calibration"]["mode"] == "head_pose_only" and [e.type for e in evs] == ["FACE_ABSENT"]


def test_replay_report_flag_cuts_evidence_from_the_render():
    d = Path(tempfile.mkdtemp())
    _write_avi(d / "v.avi", 80)  # FACE_ABSENT, as above
    args = cli.build_parser().parse_args(["replay", str(d / "v.avi"), "--out", str(d / "o"), "--render",
                                          str(d / "r.avi"), "--report"])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        assert cli._replay(args, perceiver=FakePerceiver([(99, "absent")])) == 0
    assert "no face and no person were visible" in (d / "o" / "report.html").read_text(encoding="utf-8")
    assert (d / "o" / "evidence" / "ev0_onset.jpg").exists() and (d / "o" / "segments.json").exists()


def test_cli_argument_wiring():
    p = cli.build_parser()
    a = p.parse_args(["replay", "v.mp4", "--config", "a.yaml", "--config", "b.yaml", "--calib", "c.json",
                      "--out", "o", "--render", "r.mp4"])
    assert (a.fn, a.video, a.config, a.calib, a.out, a.render) == (cli._replay, "v.mp4", ["a.yaml", "b.yaml"], "c.json", "o", "r.mp4")
    a = p.parse_args(["replay", "v.mp4"])
    assert a.config == [] and a.calib is None and a.render is None  # no --render => no video
    a = p.parse_args(["live", "--camera", "2", "--calib", "c.json"])
    assert (a.fn, a.camera, a.calib, a.out) == (cli._live, 2, "c.json", None)  # live writes nothing unless --out
    a = p.parse_args(["calibrate", "--camera", "1"])
    assert (a.fn, a.camera, a.out) == (cli._calibrate, 1, "calib.json")
    a = p.parse_args(["extract-features", "data/recordings", "--config", "x.yaml"])
    assert (a.fn, a.dir, a.config) == (cli._extract, "data/recordings", ["x.yaml"])
    for bad in ([], ["nope"], ["replay"]):
        with contextlib.redirect_stderr(io.StringIO()):  # argparse prints usage before exiting
            try:
                p.parse_args(bad)
            except SystemExit:
                continue
        raise AssertionError(bad)
