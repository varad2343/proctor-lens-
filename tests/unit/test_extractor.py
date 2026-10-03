import dataclasses
import itertools
import math

import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Detections, Face, IdCheck, Perceived
from proctorlens.features.extractor import FeatureExtractor
from proctorlens.features.schema import COLUMNS, STRING_COLUMNS
from proctorlens.perception.gaze import Baseline, Calibration, CalibSample, fit_calibration, off_screen

BOOLS = {"frame_valid", "primary_face_present", "turned_away", "id_quality_ok", "reliable"}
BLEND = {"jawOpen": 0.1, "mouthClose": 0.05, "mar": 0.2, "mouthFunnel": 0.0, "mouthPucker": 0.0,
         "eyeLookInLeft": 0.3}


def _face(yaw=0.0, pitch=0.0, bbox=(0.3, 0.2, 0.7, 0.8), **kw) -> Face:
    x0, y0, x1, y1 = bbox
    f = Face(bbox, yaw, pitch, 0.0, (0.5, 0.5, 0.5, 0.5), (1.0, 1.0), dict(BLEND),
             ((x0 + x1) / 2, (y0 + y1) / 2), y1 - y0)
    return dataclasses.replace(f, **kw)


def _p(t, faces, det=None, id=None, quality=0.9) -> Perceived:
    return Perceived(t, faces, det or Detections(), id, 120.0, 80.0, 0.0, quality, [])


def test_valid_row_schema_types_and_baseline_deltas():
    calib = Calibration(Baseline(yaw=10, pitch=-5, roll=2, cx=0.4, cy=0.5, scale=0.5), None, math.nan,
                        "head_pose_only", False)
    ex = FeatureExtractor(Config(), calib)
    row = ex.step(1000, _p(1000, [_face(yaw=25, pitch=-2, roll=0.0)], id=IdCheck(0.7, True)), 30.0, 29.5)
    assert list(row) == COLUMNS
    for c in COLUMNS:
        want = str if c in STRING_COLUMNS else bool if c in BOOLS else None
        assert type(row[c]) is want if want else type(row[c]) in (int, float), c
    assert (row["d_yaw"], row["d_pitch"], row["d_roll"]) == (15, 3, -2)
    assert abs(row["head_x"] - 0.1) < 1e-12 and abs(row["head_scale"] - 1.2) < 1e-12
    assert row["frame_valid"] and row["reliable"] and row["primary_face_present"] and row["n_faces"] == 1
    assert row["n_persons"] == 0 and row["time_since_face_ms"] == 0 and row["zone"] == "on_screen"
    assert row["frame_age_ms"] == 30.0 and row["effective_fps"] == 29.5 and row["quality_reasons"] == ""
    assert row["jaw_open"] == 0.1 and row["mar"] == 0.2 and row["eyelook_in_l"] == 0.3
    assert math.isnan(row["eyelook_out_l"])  # blendshape absent -> undefined
    assert row["id_similarity"] == 0.7 and row["id_quality_ok"] is True
    assert math.isnan(row["gaze_x"])  # head_pose_only: no gaze point
    far = ex.step(1100, _p(1100, [_face(yaw=50)], quality=0.2), 0.0, 10.0)  # d_yaw = 40 > limit; low quality
    assert far["off_screen_score"] == 1.0 and far["zone"] == "right" and far["gaze_in_screen_prob"] == 0.0
    assert far["reliable"] is False and math.isnan(far["id_similarity"]) and far["id_quality_ok"] is False


def test_frame_gap_row():
    ex = FeatureExtractor(Config(), None)
    ex.step(0, _p(0, [_face()]), 0.0, 10.0)
    rows = [ex.step(t, None, math.inf, 10.0) for t in (100, 200, 300)]
    assert [r["time_since_face_ms"] for r in rows] == [100, 200, 300]  # keeps growing
    r = rows[0]
    assert list(r) == COLUMNS and r["frame_valid"] is False and r["reliable"] is False and r["zone"] == "none"
    assert r["quality_reasons"] == "frame_gap" and r["primary_face_present"] is False
    assert r["turned_away"] is False
    nans = ("d_yaw", "off_screen_score", "quality", "phone_conf", "frame_age_ms")
    assert all(math.isnan(r[c]) for c in nans)
    assert ex.step(400, _p(400, [_face()]), 0.0, 10.0)["time_since_face_ms"] == 0


def test_turned_away_vs_face_absent():
    cfg = Config()  # turned_away_yaw 50
    assert cfg.pipeline.turned_away_hold_ms >= 1000 * cfg.events["off_screen"].t_on_s + 1000  # E4 can fire on it
    cfg.pipeline.turned_away_hold_ms = 1500
    ex = FeatureExtractor(cfg, None)
    ex.step(0, _p(0, [_face(yaw=20)]), 0.0, 10.0)
    ex.step(100, _p(100, [_face(yaw=-60)]), 0.0, 10.0)
    r = ex.step(200, _p(200, []), 0.0, 10.0)
    assert r["turned_away"] is True and r["off_screen_score"] == 1.0 and r["zone"] == "left"
    assert r["primary_face_present"] is False and r["n_faces"] == 0 and r["reliable"] is True
    assert ex.step(1600, _p(1600, []), 0.0, 10.0)["turned_away"] is True  # still within the hold
    r = ex.step(1700, _p(1700, []), 0.0, 10.0)
    assert r["turned_away"] is False and r["zone"] == "none" and math.isnan(r["off_screen_score"])
    assert r["time_since_face_ms"] == 1600
    ex = FeatureExtractor(cfg, None)  # frontal face that vanishes is an absence; positive yaw -> right
    ex.step(0, _p(0, [_face(yaw=10)]), 0.0, 10.0)
    assert ex.step(100, _p(100, []), 0.0, 10.0)["turned_away"] is False
    ex.step(200, _p(200, [_face(yaw=70)]), 0.0, 10.0)
    assert ex.step(300, _p(300, []), 0.0, 10.0)["zone"] == "right"
    # past the hold it stays TURNED_AWAY while a person is still detected (a turned head, not someone who left)
    ex = FeatureExtractor(cfg, None)
    ex.step(0, _p(0, [_face(yaw=60)]), 0.0, 10.0)
    body = Detections(person_boxes=[(0.2, 0.2, 0.8, 1.0)])
    assert ex.step(5000, _p(5000, [], body), 0.0, 10.0)["turned_away"] is True
    assert ex.step(5100, _p(5100, []), 0.0, 10.0)["turned_away"] is False


def test_static_faces_and_persons_suppressed():
    cfg = Config()
    cfg.pipeline.static_window_s = 3.0
    ex = FeatureExtractor(cfg, None)
    poster_b, poster_p, tiny_b, late_b = (0.05, 0.1, 0.2, 0.3), (0.0, 0.0, 0.3, 0.9), (0.8, 0.1, 0.83, 0.14), (0.75, 0.5, 0.9, 0.7)
    rows = {}
    for k in range(80):
        x = 0.4 + 0.005 * k  # a real second person keeps moving
        poster, tiny, late = _face(bbox=poster_b), _face(bbox=tiny_b), _face(bbox=late_b)
        mover = _face(bbox=(x, 0.6, x + 0.1, 0.8))
        det = Detections(person_boxes=[poster_p, (x, 0.3, x + 0.3, 0.9)])
        faces = [_face(), poster, mover, tiny] + ([late] if k >= 20 else [])  # `late` shows up at 2 s
        rows[k * 100] = ex.step(k * 100, _p(k * 100, faces, det), 0.0, 10.0)
    key = lambda r: (r["n_faces"], r["n_persons"], r["static_face_flags"])  # noqa: E731
    assert key(rows[900]) == (3, 2, 0)  # the tiny face is never counted; the poster is new to the tracker...
    assert key(rows[1000]) == (2, 1, 2)  # ...but it was in view at the start: static after 1 s, not after 3 s
    assert key(rows[4900]) == (3, 1, 2)  # `late` needs the full static_window_s (3 s) to count as background
    assert key(rows[5000]) == (2, 1, 3) and key(rows[7900]) == (2, 1, 3)
    assert poster.static and late.static and not mover.static  # flagged on the Face objects for overlays


def test_static_poster_survives_dark_stretch_and_own_body_is_never_static():
    cfg = Config()
    cfg.pipeline.static_window_s = 1.0
    ex = FeatureExtractor(cfg, None)
    poster, body = (0.05, 0.1, 0.2, 0.3), (0.3, 0.1, 0.7, 1.0)  # body = the user's person box (face centre inside)

    def step(t, seen=True):
        det = Detections(person_boxes=[body]) if seen else Detections()
        return ex.step(t, _p(t, [_face(), _face(bbox=poster)] if seen else [], det), 0.0, 10.0)

    for k in range(30):
        r = step(k * 100)
    assert (r["n_faces"], r["n_persons"], r["static_face_flags"]) == (1, 1, 1)  # poster static, body still counted
    for k in range(30, 130):
        step(k * 100, seen=False)  # 10 s of dark / blocked camera: no boxes at all
    r = step(13000)
    assert (r["n_faces"], r["n_persons"], r["static_face_flags"]) == (1, 1, 1)  # no 20 s of a fake second face


def test_head_speed_blink_and_mouth_energy():
    ex = FeatureExtractor(Config(), None)
    r0 = ex.step(0, _p(0, [_face()]), 0.0, 10.0)
    assert math.isnan(r0["head_speed"]) and r0["blink"] == 0.0 and r0["mouth_energy_1s"] == 0.0
    r1 = ex.step(100, _p(100, [_face(yaw=3.0, pitch=4.0)]), 0.0, 10.0)
    assert abs(r1["head_speed"] - 50.0) < 1e-9  # hypot(3, 4) deg in 0.1 s
    closed = _face(eye_open=(0.1, 0.1))
    assert ex.step(200, _p(200, [closed]), 0.0, 10.0)["blink"] == 1.0
    assert ex.step(300, _p(300, [_face()]), 0.0, 10.0)["blink"] == 0.0
    for k in range(10):  # jaw alternating 0.1 / 0.5: std 0.2 over the last second
        f = _face(blend={**BLEND, "jawOpen": 0.1 if k % 2 else 0.5})
        r = ex.step(400 + 100 * k, _p(400 + 100 * k, [f]), 0.0, 10.0)
    assert abs(r["mouth_energy_1s"] - 0.2) < 1e-9


# ---- signal quality: One Euro smoothing + blink-aware gaze (cfg.smooth) --------------------------------------------
_GRID = list(itertools.product((0.1, 0.5, 0.9), repeat=2))
_SMOOTHED = {"d_yaw", "d_pitch", "d_roll", "head_x", "head_y", "head_scale", "iris_lx", "iris_ly", "iris_rx", "iris_ry",
             "gaze_x", "gaze_y", "gaze_in_screen_prob", "off_screen_score", "zone"}


def _gface(x, y, head=1.0, rng=None, nz=1.0) -> Face:
    """Synthetic face whose pose / iris / eye-look encode screen point (x, y); sensor noise (yaw sigma 0.5 deg x nz) if rng."""
    n = (lambda s: rng.normal(0, s * nz)) if rng is not None else (lambda s: 0.0)
    dx, dy = x - 0.5, y - 0.5
    bl = {"eyeLookInLeft": 0.3 - 0.4 * dx, "eyeLookOutLeft": 0.3 + 0.4 * dx, "eyeLookInRight": 0.3 + 0.4 * dx,
          "eyeLookOutRight": 0.3 - 0.4 * dx, "eyeLookUpLeft": 0.3 - 0.4 * dy, "eyeLookDownLeft": 0.3 + 0.4 * dy,
          "eyeLookUpRight": 0.3 - 0.4 * dy, "eyeLookDownRight": 0.3 + 0.4 * dy, "jawOpen": 0.1, "mouthClose": 0.05}
    c = (0.5 + 0.03 * head * dx + n(0.002), 0.5 + 0.02 * head * dy + n(0.002))
    iris = (0.5 + 0.3 * dx + n(0.005), 0.5 + 0.3 * dy + n(0.005), 0.5 + 0.3 * dx + n(0.005), 0.5 + 0.3 * dy + n(0.005))
    return Face((c[0] - 0.2, c[1] - 0.3, c[0] + 0.2, c[1] + 0.3), head * 40 * dx + n(0.5), -head * 30 * dy + n(0.5),
                n(0.5), iris, (1.0, 1.0), {k: v + n(0.01) for k, v in bl.items()}, c, 0.6 + n(0.002))


def _gcalib(head=1.0) -> Calibration:
    """Full gaze calibration fitted on synthetic faces (3x3 grid + 4 validation dots)."""
    rng = np.random.default_rng(0)
    s = [CalibSample(_gface(.5, .5, head, rng), .5, .5, "neutral", 0) for _ in range(10)]
    for i, (x, y) in enumerate(_GRID, 1):
        s += [CalibSample(_gface(x, y, head, rng), x, y, "grid", i) for _ in range(10)]
    for i, (x, y) in enumerate([(.25, .3), (.75, .3), (.3, .75), (.7, .7)], 10):
        s += [CalibSample(_gface(x, y, head, rng), x, y, "validation", i) for _ in range(10)]
    cal = fit_calibration(s, Config().gaze)
    assert cal.mode == "full", cal.error
    return cal


def _rows(faces, calib=None, **smooth) -> list[dict]:
    """One grid row per entry of faces (None = no face); smooth overrides cfg.smooth fields."""
    cfg = Config()
    for k, v in smooth.items():
        setattr(cfg.smooth, k, v)
    ex = FeatureExtractor(cfg, calib)
    return [ex.step(100 * i, _p(100 * i, [] if f is None else [f]), 0.0, 10.0) for i, f in enumerate(faces)]


def _col(rows, c) -> np.ndarray:
    return np.array([r[c] for r in rows], float)


def test_smooth_disabled_rows_are_the_raw_signals():
    rng, cal = np.random.default_rng(0), _gcalib()
    faces = [_gface(0.5, 0.5, 1.0, rng, 2.0) for _ in range(60)]
    off, on = _rows(faces, cal, enabled=False), _rows(faces, cal)
    for f, r in zip(faces, off):  # enabled=False: exactly the old arithmetic on the raw face
        g = off_screen(f, cal, Config().gaze)
        assert r["d_yaw"] == f.yaw - cal.baseline.yaw and r["iris_ry"] == f.iris[3]
        assert r["head_x"] == f.center[0] - cal.baseline.cx and r["head_scale"] == f.scale / cal.baseline.scale
        assert (r["gaze_x"], r["gaze_y"], r["off_screen_score"], r["zone"]) == (g.gx, g.gy, g.score, g.zone)
    for a, b in zip(off, on):  # smoothing touches only the pose / iris / gaze columns
        assert all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a if k not in _SMOOTHED), (a, b)
    assert any(a["d_yaw"] != b["d_yaw"] and a["gaze_x"] != b["gaze_x"] for a, b in zip(off, on))


def test_smoothing_cuts_jitter_on_a_still_head_3x():
    rng, cal = np.random.default_rng(1), _gcalib()
    faces = [_gface(0.5, 0.5, 1.0, rng) for _ in range(1500)]  # head and eyes still: pure sensor noise
    off, on = _rows(faces, cal, enabled=False), _rows(faces, cal)
    for c in ("d_yaw", "d_pitch", "head_x", "iris_lx", "gaze_x", "gaze_y"):  # measured 3.4x - 3.6x on 10 seeds
        a, b = _col(off, c)[20:], _col(on, c)[20:]
        assert a.std() >= 3 * b.std(), (c, a.std() / b.std())
        assert abs(a.mean() - b.mean()) < 0.3 * a.std()  # steadier, not shifted


def test_smoothing_step_response_latency():
    cal, cal0 = _gcalib(), _gcalib(0.0)

    def steps_to_90(rows, c) -> tuple[int, np.ndarray]:  # grid steps after the step until 90% of the way there
        v = _col(rows, c)
        return int(np.argmax(v[20:] - v[19] >= 0.9 * (v[-1] - v[19]))) + 1, v

    for mag, limit in ((10, 3), (20, 2), (40, 1)):  # measured 2 / 2 / 1 grid steps (the raw signal needs 1)
        faces = [_gface(.5, .5)] * 20 + [dataclasses.replace(_gface(.5, .5), yaw=mag)] * 40
        n, v = steps_to_90(_rows(faces, cal), "d_yaw")
        assert n <= limit, (mag, n)
        assert (np.diff(v[19:]) >= -1e-9).all() and abs(v[-1] - v[19] - mag) < 0.2  # no ringing, converges to the step
    faces = [_gface(.5, .5, 0.0)] * 20 + [_gface(.9, .5, 0.0)] * 40  # eyes only: the gaze point jumps 0.5 -> 0.9
    n, v = steps_to_90(_rows(faces, cal0), "gaze_x")
    assert n <= 4 and abs(v[-1] - 0.9) < 0.05, n  # measured 3


def test_smoothing_stops_off_screen_score_flicker():
    rng = np.random.default_rng(3)  # head pose only: yaw hovers at the 25 deg limit, where the score is 0.5
    faces = [dataclasses.replace(_face(), yaw=25.0 + 1.5 * math.sin(i / 40) + rng.normal(0, 1.0)) for i in range(600)]

    def crossings(rows) -> int:
        s = _col(rows, "off_screen_score")[20:] >= 0.5
        return int((s[1:] != s[:-1]).sum())

    raw, smooth = crossings(_rows(faces, enabled=False)), crossings(_rows(faces))
    assert raw >= 100 and smooth <= raw // 10, (raw, smooth)  # measured 142 -> 6


def _blink_faces(scale: float):
    """Still, on-screen face with a 200 ms blink every 3 s: lids shut => iris-y and eyeLookDown/Up are garbage."""
    rng, faces, shut = np.random.default_rng(5), [], []
    for i in range(300):
        f, b = _gface(0.5, 0.5, 1.0, rng), i % 30 in (10, 11)
        if b:
            f = dataclasses.replace(f, iris=(f.iris[0], 0.95, f.iris[2], 0.95), eye_open=(0.1 * scale,) * 2,
                                    blend={**f.blend, "eyeLookDownLeft": 0.9, "eyeLookDownRight": 0.9,
                                           "eyeLookUpLeft": 0.0, "eyeLookUpRight": 0.0})
        faces.append(f if b else dataclasses.replace(f, eye_open=(scale, scale)))
        shut.append(b)
    return faces, np.array(shut)


def test_blink_burst_does_not_dip_the_gaze():
    cal = _gcalib()
    for scale in (1.0, 0.5):  # eye_open's scale differs between versions; the hold keys off the relative lid drop
        faces, shut = _blink_faces(scale)
        out = {}
        for name, kw in (("off", {"enabled": False}), ("no_hold", {"blink_frac": 0.0}), ("hold", {})):
            rows = _rows(faces, cal, **kw)
            gy, sc = _col(rows, "gaze_y"), _col(rows, "off_screen_score")
            out[name] = (np.abs(gy[shut] - 0.5).max(), sc[shut].max(), sc[~shut][20:].max(), rows)
        assert out["off"][0] > 0.5 and out["off"][1] >= 0.5  # raw: a blink throws the gaze off the screen
        assert out["no_hold"][0] > 0.5 and out["no_hold"][1] >= 0.5  # the filter alone does not remove it
        assert out["hold"][0] < 0.05 and out["hold"][1] < 0.1 and out["hold"][2] < 0.1  # held: no dip, no tail
        assert all(r["zone"] == "on_screen" for r in out["hold"][3][20:])
        assert all(r["blink"] == 1.0 for r, b in zip(out["hold"][3], shut) if b)  # the blink column is unchanged


def test_smoothing_restarts_after_a_face_gap_or_a_different_face():
    def d_yaw_after(gap_steps: int, bbox=(0.3, 0.2, 0.7, 0.8)) -> float:
        ex = FeatureExtractor(Config(), None)
        for k in range(10 + gap_steps):
            faces = [_face()] if k < 10 else []
            ex.step(100 * k, _p(100 * k, faces), 0.0, 10.0)
        t = 100 * (10 + gap_steps)
        return ex.step(t, _p(t, [_face(yaw=20.0, bbox=bbox)]), 0.0, 10.0)["d_yaw"]

    assert d_yaw_after(0) < 19.5 and d_yaw_after(3) < 19.5  # still tracking: smoothed, lagging the 20 deg jump
    assert d_yaw_after(8) == 20.0  # unseen 0.9 s > reset_gap_s: no stale state, first sample passes through
    assert d_yaw_after(0, bbox=(0.0, 0.2, 0.25, 0.8)) == 20.0  # a different face (boxes do not overlap): restarted


def test_smoothing_does_not_delay_turned_away():
    ys = [8.0 * k for k in range(9)] + [None] * 12 + [0.0] * 5  # head swings to 64 deg, face lost, comes back
    faces = [None if y is None else _face(yaw=y) for y in ys]
    off, on = _rows(faces, enabled=False), _rows(faces)
    assert [r["turned_away"] for r in off] == [r["turned_away"] for r in on] and any(r["turned_away"] for r in on)
    for a, b in zip(off, on):
        if a["turned_away"]:  # TURNED_AWAY rows are decided on the raw yaw: identical, score exactly 1
            assert a["off_screen_score"] == b["off_screen_score"] == 1.0 and a["zone"] == b["zone"] == "right"
    first = lambda rows: next(i for i, r in enumerate(rows) if r["off_screen_score"] >= 0.5)
    assert first(on) - first(off) <= 1  # off-screen onset at most one grid step later
    assert on[-1]["d_yaw"] == 0.0 and on[-1]["time_since_face_ms"] == 0  # back to the raw value after the face gap
