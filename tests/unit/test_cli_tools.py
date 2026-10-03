"""CLI tooling: run summary, doctor checks (with fakes), fetch-models planning (tmp dirs, never the network),
parser wiring, quick_bench helpers (no models). Nothing here opens a camera or a window."""
import contextlib
import io
import json
import math
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml.evaluation import quick_bench as qb
from proctorlens import cli
from proctorlens.core.config import Config
from proctorlens.core.types import EVENT_TYPES, Detections, Event, Face, Perceived


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _ev(t, a, b):
    return Event(t, a, b, 0.9, "rule@0.1.0")


def _face(yaw=0.0, eye=(0.9, 0.7)):
    return Face(bbox=(0.3, 0.2, 0.6, 0.8), yaw=yaw, pitch=0.0, roll=0.0, iris=(0.5,) * 4, eye_open=eye, blend={},
                center=(0.45, 0.5), scale=0.6)


# ---- summary -------------------------------------------------------------------------------------------------

def test_summarize_counts_seconds_and_percentages():
    df = pd.DataFrame({"t_ms": range(0, 10000, 100), "primary_face_present": [True] * 50 + [math.nan] * 50})
    events = [_ev("OFF_SCREEN_SUSTAINED", 1000, 3000), _ev("OFF_SCREEN_SUSTAINED", 5000, 6000),
              _ev("PROHIBITED_OBJECT", 2000, 4000), _ev("PROHIBITED_OBJECT", 3000, 5000),  # phone + notes overlap
              _ev("MONITORING_DEGRADED", 6000, 8000)]
    s = cli.summarize(df, events, None)
    assert s["duration_s"] == 10.0 and s["n_rows"] == 100
    assert s["face_seen_pct"] == 50.0 and s["degraded_pct"] == 20.0  # NaN rows (frame gaps) are "not seen"
    assert s["events"]["OFF_SCREEN_SUSTAINED"] == {"count": 2, "total_s": 3.0}
    assert s["events"]["PROHIBITED_OBJECT"] == {"count": 2, "total_s": 3.0}  # union 2000-5000, not 4.0
    assert s["events"]["FACE_ABSENT"] == {"count": 0, "total_s": 0.0}  # every type is listed
    assert set(s["events"]) == set(EVENT_TYPES)
    assert s["calibration"] == dict(mode="none", error=None, accepted=None, tries=None)
    c = SimpleNamespace(mode="full", error=0.08249, accepted=True, tries=2)
    assert cli.summarize(df, [], c)["calibration"] == dict(mode="full", error=0.082, accepted=True, tries=2)
    assert cli.summarize(df, [], SimpleNamespace(mode="head_pose_only", error=math.nan, accepted=False,
                                                  tries=1))["calibration"]["error"] is None
    json.dumps(s, allow_nan=False)  # JSON-safe


def test_summarize_empty_run_and_summary_file():
    e = cli.summarize(pd.DataFrame({"t_ms": [], "primary_face_present": []}), [], None)
    assert e["duration_s"] == 0.0 and e["face_seen_pct"] is None and e["degraded_pct"] is None
    d = _tmp()
    df = pd.DataFrame({"t_ms": range(0, 3000, 100), "primary_face_present": [True] * 30})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli._summary(d, df, [_ev("MOUTH_ACTIVITY", 0, 3000)], None)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    assert s["events"]["MOUTH_ACTIVITY"] == {"count": 1, "total_s": 3.0} and s["face_seen_pct"] == 100.0
    assert "MOUTH_ACTIVITY: 1 event(s), 3.0 s" in buf.getvalue() and "calibration none" in buf.getvalue()


def test_fps_from_frame_times():
    assert abs(cli._fps([0.0, 0.1, 0.2, 0.3]) - 10.0) < 1e-9
    assert math.isnan(cli._fps([1.0])) and math.isnan(cli._fps([])) and math.isnan(cli._fps([2.0, 2.0]))


# ---- doctor --------------------------------------------------------------------------------------------------

def test_check_deps_required_fail_optional_warn():
    def fake(name):
        if name in ("mediapipe", "insightface"):
            raise ImportError(f"No module named {name}")
        return SimpleNamespace(__version__="1.2")

    got = {n: s for s, n, _ in cli.check_deps(fake)}
    assert got["mediapipe"] == "FAIL" and got["insightface"] == "WARN"  # optional: identity is skippable
    assert got["numpy"] == got["cv2"] == got["torch"] == "PASS"
    assert ("PASS", "numpy", "1.2") in cli.check_deps(fake)


def _cfg_in(d: Path, lm=True, det=True, ident=False) -> Config:
    cfg = Config()
    cfg.models.landmarker, cfg.models.detector, cfg.models.identity = str(d / "lm.task"), str(d / "yolo11n.pt"), str(d / "buf")
    if lm:
        (d / "lm.task").write_bytes(b"x")
    if det:
        (d / "yolo11n.pt").write_bytes(b"x")
    if ident:
        (d / "buf").mkdir()
        (d / "buf" / "w.onnx").write_bytes(b"x")
    return cfg


def test_check_models_and_config():
    d = _tmp()
    assert [s for s, _, _ in cli.check_models(_cfg_in(d))] == ["PASS", "PASS", "WARN"]  # no identity: optional
    d = _tmp()
    assert [s for s, _, _ in cli.check_models(_cfg_in(d, det=False, ident=True))] == ["PASS", "FAIL", "PASS"]
    (d / "ok.yaml").write_text("pipeline: {grid_hz: 10}\n", encoding="utf-8")
    (d / "bad.yaml").write_text("pipeline: {nope: 1}\n", encoding="utf-8")
    (checks, cfg), (bad, none) = cli.check_config([str(d / "ok.yaml")]), cli.check_config([str(d / "bad.yaml")])
    assert checks[0][0] == "PASS" and cfg is not None
    assert bad[0][0] == "FAIL" and "nope" in bad[0][2] and none is None


class _Lm:
    def __init__(self, faces):
        self.faces = faces

    def process(self, frame, t_ms):
        return self.faces


class _Det:
    def __init__(self, n):
        self.n = n

    def process(self, frame):
        return Detections(person_boxes=[(0.1, 0.1, 0.4, 0.9)] * self.n)


class _Id:
    def embed(self, frame, face):
        return np.zeros(512, np.float32)


def test_selftest_with_fakes():
    cfg, imgs = _cfg_in(_tmp()), {"zidane_crop": np.zeros((9, 9, 3), np.uint8), "bus": np.zeros((9, 9, 3), np.uint8)}
    ok = cli.selftest(cfg, imgs, _Lm([_face(9.5)]), _Det(4), _Id())
    assert [s for s, _, _ in ok] == ["PASS"] * 3 and "yaw 9.5" in ok[0][2] and "512-d" in ok[2][2]
    assert [s for s, _, _ in cli.selftest(cfg, imgs, _Lm([_face()]), _Det(4))] == ["PASS", "PASS", "SKIP"]  # no identity model
    few = cli.selftest(cfg, imgs, _Lm([_face()]), _Det(1), _Id())
    assert few[1][0] == "FAIL" and "1 person" in few[1][2]  # needs >= 2 persons
    none = cli.selftest(cfg, imgs, _Lm([]), _Det(2), _Id())
    assert [s for s, _, _ in none] == ["FAIL", "PASS", "SKIP"]  # no face -> identity has nothing to embed

    class Boom:
        def process(self, *a):
            raise RuntimeError("model file corrupt")

    boom = cli.selftest(cfg, imgs, Boom(), _Det(2), _Id())
    assert boom[0][0] == "FAIL" and "model file corrupt" in boom[0][2]
    assert cli.selftest(cfg, {}) == [("SKIP", "selftest", "ultralytics sample images not found")]


class _Cap:
    """Fake capture + clock: every clock() call advances 0.25 s, so the ~3 s loop ends after a few reads."""

    def __init__(self, frame, ok=True):
        self.frame, self.ok, self.reads, self.t = frame, ok, 0, 0.0

    def read(self):
        self.reads += 1
        return self.ok, self.frame

    def clock(self):
        self.t += 0.25
        return self.t


def _textured(level):
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(level, 40, (120, 160, 3)), 0, 255).astype(np.uint8)


def test_camera_report_logic_with_fake_capture():
    cfg = Config()
    cap = _Cap(_textured(130))
    good = cli.camera_report(cap, _Lm([_face()]), cfg, seconds=3.0, clock=cap.clock)
    assert [s for s, _, _ in good] == ["PASS"] * 3 and cap.reads >= 5 and "frames" in good[0][2]
    cap = _Cap(_textured(130))
    away = cli.camera_report(cap, _Lm([]), cfg, clock=cap.clock)
    assert away[0][0] == "WARN" and "face the camera" in away[0][2]  # advice, not an error exit
    cap = _Cap(_textured(12))
    dark = cli.camera_report(cap, _Lm([_face()]), cfg, clock=cap.clock)
    assert dark[1][0] == "WARN" and "too dark" in dark[1][2]
    cap = _Cap(np.full((120, 160, 3), 130, np.uint8))  # flat image: no detail => soft
    assert cli.camera_report(cap, _Lm([_face()]), cfg, clock=cap.clock)[2][0] == "WARN"
    cap = _Cap(None, ok=False)
    assert cli.camera_report(cap, _Lm([]), cfg, clock=cap.clock)[0][0] == "FAIL"  # no frames at all


@contextlib.contextmanager
def _patch(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _models_yaml(d: Path, cfg: Config) -> str:
    m = cfg.models
    (d / "c.yaml").write_text(f"models: {{landmarker: {json.dumps(m.landmarker)}, detector: {json.dumps(m.detector)}, "
                              f"identity: {json.dumps(m.identity)}}}\n", encoding="utf-8")
    return str(d / "c.yaml")


def test_doctor_exit_code_follows_failures():
    with _patch(cli, check_deps=lambda: [("PASS", "numpy", "1.0"), ("WARN", "torch", "missing")]):
        for ok in (True, False):
            d = _tmp()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli._doctor(SimpleNamespace(config=[_models_yaml(d, _cfg_in(d, det=ok))], selftest=False, camera=None))
            assert rc == (0 if ok else 1)
            assert ("FAIL model detector" in buf.getvalue()) == (not ok) and "PASS model landmarker" in buf.getvalue()


# ---- live loop, headless -------------------------------------------------------------------------------------

class _LiveCam:
    def read(self):
        time.sleep(0.01)
        return True, np.full((240, 320, 3), 120, np.uint8)

    def release(self):
        pass


class _FakePerceiver:
    def __init__(self, cfg, landmarker=None):
        pass

    def perceive(self, frame, t_ms, idx):
        return Perceived(t_ms, [_face()], Detections(), None, 120.0, 200.0, 0.0, 0.9, [])


def test_live_loop_passes_hud_values_and_writes_summary():
    """The real _live loop with a fake camera and stubbed cv2 GUI calls: no camera, no window."""
    import cv2

    from proctorlens import perception
    from proctorlens.explain import overlays
    from proctorlens.perception.gaze import Baseline, Calibration

    d, end, calls = _tmp(), time.monotonic() + 0.8, []
    Calibration(Baseline(0.0, 0.0, 0.0, 0.5, 0.5, 0.4), None, math.nan, "head_pose_only", False).save(d / "c.json")
    real_draw = overlays.draw

    def spy(*a, **kw):
        calls.append(kw)
        return real_draw(*a, **kw)

    def forbidden(*a):
        raise AssertionError("no window may be created")

    args = cli.build_parser().parse_args(["live", "--calib", str(d / "c.json"), "--out", str(d / "out")])
    buf = io.StringIO()
    with _patch(cli, _open_camera=lambda i: _LiveCam()), _patch(perception, Perceiver=_FakePerceiver), \
            _patch(overlays, draw=spy), _patch(cv2, imshow=lambda *a: None, destroyAllWindows=lambda: None,
                                               getWindowProperty=lambda *a: 1.0, namedWindow=forbidden,
                                               waitKey=lambda *a: 27 if time.monotonic() > end else -1), \
            contextlib.redirect_stdout(buf):
        assert cli._live(args) == 0
    assert calls and all(c["calib_mode"] == "head_pose_only" for c in calls)
    assert any(c["fps"] > 1 for c in calls)  # ~100 fake frames/s, measured from frame times (NaN on the first call)
    assert any(c["zone"] in ("on_screen", "none") for c in calls)  # the zone arrives once the first grid row exists
    s = json.loads((d / "out" / "summary.json").read_text(encoding="utf-8"))
    assert s["calibration"]["mode"] == "head_pose_only" and s["face_seen_pct"] > 50 and "summary:" in buf.getvalue()


# ---- fetch-models --------------------------------------------------------------------------------------------

def test_plan_fetch_lists_only_missing_files():
    d = _tmp()
    plan = {n: (p, s) for n, p, s in cli.plan_fetch(_cfg_in(d, lm=False, det=False))}
    assert set(plan) == {"landmarker", "detector", "identity"}
    assert plan["landmarker"][1].endswith("face_landmarker.task") and plan["detector"][1] == "ultralytics yolo11n.pt"
    assert plan["identity"][1] == "insightface buf"
    assert [n for n, _, _ in cli.plan_fetch(_cfg_in(_tmp(), ident=True))] == []  # all present: nothing, never overwritten
    d = _tmp()
    (d / "buf").mkdir()  # an empty identity folder is still missing
    cfg = _cfg_in(d, lm=True, det=False)
    cfg.models.detector = str(d / "custom.onnx")  # not yolo11n.pt: no standard download
    assert {n: s for n, _, s in cli.plan_fetch(cfg)} == {"detector": None, "identity": "insightface buf"}


def test_fetch_models_dry_run_changes_nothing():
    def no_fetch(*a):
        raise AssertionError("must not fetch")

    d = _tmp()
    conf = _models_yaml(d, _cfg_in(d, lm=False, det=False))
    before = sorted(p.name for p in d.iterdir())
    with _patch(cli, _fetch=no_fetch):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli._fetch_models(SimpleNamespace(config=[conf], dry_run=True)) == 0
        out = buf.getvalue()
        assert out.count("would fetch:") == 3 and "AGPL-3.0" in out and "non-commercial" in out
        assert sorted(p.name for p in d.iterdir()) == before  # nothing created
        _cfg_in(d, ident=True)  # now everything exists: even without --dry-run there is nothing to fetch
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli._fetch_models(SimpleNamespace(config=[conf], dry_run=False)) == 0
        assert "nothing to fetch" in buf.getvalue() and "AGPL-3.0" in buf.getvalue()  # licenses are always shown


def test_tool_parser_wiring():
    p = cli.build_parser()
    a = p.parse_args(["doctor", "--camera", "1", "--selftest", "--config", "x.yaml"])
    assert (a.fn, a.camera, a.selftest, a.config) == (cli._doctor, 1, True, ["x.yaml"])
    a = p.parse_args(["doctor"])
    assert a.camera is None and a.selftest is False and a.config == []  # the camera is opt-in
    a = p.parse_args(["fetch-models", "--dry-run"])
    assert (a.fn, a.dry_run) == (cli._fetch_models, True)
    assert p.parse_args(["fetch-models"]).dry_run is False


# ---- quick_bench (pure helpers) ------------------------------------------------------------------------------

def test_quick_bench_perturbations_and_measure():
    img = _textured(130)
    for name, fn in qb.perturbations().items():
        out = fn(img)
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3, name
    perts = qb.perturbations()
    assert (perts["clean"](img) == img).all() and perts["clean"](img) is not img
    assert perts["gamma_2.0"](img).mean() < img.mean()
    assert perts["gamma_5.0"](img).mean() < perts["gamma_4.0"](img).mean() < perts["gamma_3.0"](img).mean()
    assert perts["down_x0.5"](img).shape[:2] == (60, 80) and perts["down_x0.25"](img).shape[0] == 30
    det = Detections(person_boxes=[(0, 0, 1, 1)] * 2)
    got = qb.measure(Perceived(0, [_face(7.0, (1.0, 0.5))], det, None, 100.0, 50.0, 0.0, 0.9, []))
    assert got == dict(face=1.0, yaw=7.0, eye=0.75, persons=2)
    none = qb.measure(Perceived(0, [], det, None, 100.0, 50.0, 0.0, 0.9, ["no_face"]))
    assert none["face"] == 0.0 and math.isnan(none["yaw"]) and math.isnan(none["eye"]) and none["persons"] == 2


def _rec(image, pert, config, yaw, face=1.0, persons=2):
    return dict(image=image, pert=pert, config=config, face=face, yaw=yaw, eye=1.0 if face else math.nan, persons=persons)


def test_quick_bench_dyaw_tables_and_report():
    recs = []
    for img in ("crop", "full"):
        for cfg in ("base", "impr"):
            nan = math.nan
            has = not (img == "full" and cfg == "base")  # base never finds the small faces
            recs += [_rec(img, "clean", cfg, 10.0 if has else nan, float(has)),
                     _rec(img, "gamma_3.0", cfg, 14.0 if has else nan, float(has))]
    qb.add_dyaw(recs)
    d = {(r["image"], r["pert"], r["config"]): r["dyaw"] for r in recs}
    assert d[("crop", "clean", "base")] == 0.0 and d[("crop", "gamma_3.0", "impr")] == 4.0
    assert math.isnan(d[("full", "gamma_3.0", "base")])  # no clean face to compare with
    t = qb.tables(recs)
    assert list(t["pert"]) == ["clean", "gamma_3.0", "ALL"]
    assert list(t.columns)[:5] == ["pert", "face rate base", "face rate impr", "|dyaw| deg base", "|dyaw| deg impr"]
    row = t[t["pert"] == "ALL"].iloc[0]
    assert row["face rate base"] == 0.5 and row["face rate impr"] == 1.0 and row["|dyaw| deg impr"] == 2.0
    md = qb.render(recs)
    assert md.startswith("# Quick perception bench") and "## crop" in md and "## full" in md and "| ALL |" in md
    assert md.count("\n|---|") == 3  # separator rows: all-images table + one per image
    assert "cheat" not in md.lower()
