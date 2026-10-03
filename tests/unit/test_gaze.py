import dataclasses
import itertools
import json
import math
import os
import tempfile

import numpy as np

from proctorlens.core.config import GazeCfg
from proctorlens.core.types import Face
from proctorlens.features.schema import COLUMNS
from proctorlens.perception.gaze import (EYELOOK, Baseline, Calibration, CalibSample, DriftMonitor, GazeModel,
                                         fit_calibration, gaze_features, off_screen)

GRID = list(itertools.product((0.1, 0.5, 0.9), repeat=2))
VAL = [(0.25, 0.3), (0.75, 0.3), (0.3, 0.75), (0.7, 0.7)]


def _face(x, y, head=1.0, rng=None, noise=1.0, flip=False):
    """Synthetic face whose head pose / iris / eye-look blendshapes linearly encode screen point (x, y).
    flip=True: unmirrored camera, screen-right appears as image-left."""
    n = (lambda s: rng.normal(0, s * noise)) if rng is not None else (lambda s: 0.0)
    dx, dy = (x - 0.5) * (-1 if flip else 1), y - 0.5
    bl = {"eyeLookInLeft": 0.3 - 0.4 * dx, "eyeLookOutLeft": 0.3 + 0.4 * dx,
          "eyeLookInRight": 0.3 + 0.4 * dx, "eyeLookOutRight": 0.3 - 0.4 * dx,
          "eyeLookUpLeft": 0.3 - 0.4 * dy, "eyeLookDownLeft": 0.3 + 0.4 * dy,
          "eyeLookUpRight": 0.3 - 0.4 * dy, "eyeLookDownRight": 0.3 + 0.4 * dy}
    bl = {k: v + n(0.01) for k, v in bl.items()}
    iris = (0.5 + 0.3 * dx + n(0.005), 0.5 + 0.3 * dy + n(0.005), 0.5 + 0.3 * dx + n(0.005),
            0.5 + 0.3 * dy + n(0.005))
    return Face(bbox=(0.3, 0.2, 0.7, 0.8), yaw=head * 40 * dx + n(0.5), pitch=-head * 30 * dy + n(0.5),
                roll=n(0.5), iris=iris, eye_open=(1.0, 1.0), blend=bl,
                center=(0.5 + 0.03 * head * dx, 0.5 + 0.02 * head * dy), scale=0.6)


def _samples(rng, head=1.0, val=VAL, garbage=False, flip=False):
    def mk(x, y):  # garbage: the face has nothing to do with the dot being shown
        if garbage:
            return _face(rng.uniform(), rng.uniform(), head, rng)
        return _face(x, y, head, rng, flip=flip)

    s = [CalibSample(mk(.5, .5), .5, .5, "neutral", 0) for _ in range(10)]
    for i, (x, y) in enumerate(GRID, 1):
        s += [CalibSample(mk(x, y), x, y, "grid", i) for _ in range(10)]
    for i, (x, y) in enumerate(val, 10):
        s += [CalibSample(mk(x, y), x, y, "validation", i) for _ in range(10)]
    return s


def test_calibration_recovers_known_mapping():
    cal = fit_calibration(_samples(np.random.default_rng(0)), GazeCfg())
    assert cal.accepted and cal.mode == "full" and cal.error < 0.05, cal.error
    px, py = cal.model.predict(gaze_features(_face(0.3, 0.6), cal.baseline))[0]
    assert abs(px - 0.3) < 0.06 and abs(py - 0.6) < 0.06
    assert abs(cal.baseline.yaw) < 0.5 and abs(cal.baseline.scale - 0.6) < 1e-9


def test_garbage_calibration_rejected_head_pose_only():
    cal = fit_calibration(_samples(np.random.default_rng(1), garbage=True), GazeCfg())
    assert not cal.accepted and cal.mode == "head_pose_only" and cal.error >= GazeCfg().accept_error
    g = off_screen(_face(0.5, 0.5), cal, GazeCfg())
    assert math.isnan(g.gx) and g.zone == "on_screen"  # gaze ignored in head_pose_only mode


def test_no_validation_is_not_accepted_and_json_roundtrip():
    rng = np.random.default_rng(2)
    full = fit_calibration(_samples(rng), GazeCfg())
    noval = fit_calibration(_samples(rng, val=[]), GazeCfg())
    assert math.isnan(noval.error) and not noval.accepted and noval.mode == "head_pose_only"
    with tempfile.TemporaryDirectory() as d:
        for cal in (full, noval):
            path = os.path.join(d, "calib.json")
            cal.save(path)
            json.load(open(path))  # plain JSON
            back = Calibration.load(path)
            assert back.mode == cal.mode and back.accepted == cal.accepted and back.baseline == cal.baseline
            assert back.error == cal.error or (math.isnan(back.error) and math.isnan(cal.error))
            x = gaze_features(_face(0.2, 0.7), cal.baseline)
            assert np.allclose(back.model.predict(x), cal.model.predict(x))


def test_off_screen_head_turn_soft_scores_and_zones():
    cfg, f = GazeCfg(), _face(0.5, 0.5)
    s = lambda **k: off_screen(dataclasses.replace(f, **k), None, cfg)  # noqa: E731
    assert s().score == 0.0 and s().zone == "on_screen"
    assert 0 < s(yaw=22).score < 0.5 and s(yaw=22).zone == "on_screen"  # soft ramp below the limit
    assert abs(s(yaw=25).score - 0.5) < 1e-9  # limit = 0.5
    assert s(yaw=40).score == 1.0 and s(yaw=40).zone == "right"  # yaw > 0 = image-right
    assert s(yaw=-40).zone == "left" and s(pitch=30).zone == "up" and s(pitch=-30).zone == "down"
    assert s(yaw=30, pitch=-10).zone == "right"  # dominant axis
    assert s(yaw=40).in_screen_prob == 0.0 and math.isnan(s(yaw=40).gx)
    calib = Calibration(Baseline(yaw=20.0), None, math.nan, "head_pose_only", False)  # deltas vs baseline
    assert off_screen(dataclasses.replace(f, yaw=20), calib, cfg).zone == "on_screen"
    assert off_screen(dataclasses.replace(f, yaw=50), calib, cfg).zone == "right"
    none = off_screen(None, calib, cfg)
    assert math.isnan(none.score) and none.zone == "none"


def test_off_screen_gaze_outside_with_eyes_only():
    rng = np.random.default_rng(3)
    cal = fit_calibration(_samples(rng, head=0.0), GazeCfg())  # head never turns: only eyes carry the signal
    assert cal.mode == "full"
    on = off_screen(_face(0.5, 0.5, head=0.0), cal, GazeCfg())
    assert on.zone == "on_screen" and on.score < 0.2 and 0.4 < on.gx < 0.6
    right = off_screen(_face(1.5, 0.5, head=0.0), cal, GazeCfg())
    assert right.zone == "right" and right.score > 0.5 and right.gx > 1.1
    assert off_screen(_face(0.5, -0.5, head=0.0), cal, GazeCfg()).zone == "up"  # y < 0 = above the screen


def test_zone_is_image_oriented_on_unmirrored_camera():
    cal = fit_calibration(_samples(np.random.default_rng(5), head=0.0, flip=True), GazeCfg())
    assert cal.mode == "full" and cal.model.x_sign == -1.0
    out = off_screen(_face(1.5, 0.5, head=0.0, flip=True), cal, GazeCfg())  # looks past screen-right
    assert out.gx > 1.1 and out.zone == "left"  # screen-right = image-left; matches the head-turn convention
    mirrored = fit_calibration(_samples(np.random.default_rng(5), head=0.0), GazeCfg())
    assert mirrored.model.x_sign == 1.0
    with tempfile.TemporaryDirectory() as d:  # x_sign survives save/load
        cal.save(os.path.join(d, "c.json"))
        assert Calibration.load(os.path.join(d, "c.json")).model.x_sign == -1.0


def test_gaze_features_layout_baseline_and_drift():
    names = [n for n, _ in EYELOOK]
    assert names == [c for c in COLUMNS if c.startswith("eyelook_")]
    blend = {f"eyeLook{d}{s}": 0.1 * (i + 1) for i, (d, s) in
             enumerate((d, s) for d in ("In", "Out", "Up", "Down") for s in ("Left", "Right"))}
    f = dataclasses.replace(_face(0.5, 0.5), blend=blend, yaw=5.0, center=(0.55, 0.5), scale=0.3)
    faces = [dataclasses.replace(f, yaw=y) for y in (1.0, 2.0, 30.0)]
    base = Baseline.from_faces(faces)
    assert base.yaw == 2.0 and base == Baseline.from_faces(faces[::-1])  # median: outlier and order ignored
    x = gaze_features(f, dataclasses.replace(base, cx=0.5, scale=0.6))
    assert x.shape == (18,) and x[0] == 3.0 and np.allclose(x[7:15], [0.1 * i for i in range(1, 9)])
    assert abs(x[15] - 0.05) < 1e-12 and abs(x[17] - 0.5) < 1e-12
    assert Baseline.from_faces([]) == Baseline()

    m = DriftMonitor(GazeCfg(drift_window_s=10.0, drift_bound=0.25), hz=10)  # window = 100 steps
    assert not any(m.update(0.5, 0.52, True) for _ in range(100))
    assert not m.update(0.9, 0.9, False)  # off-task samples are ignored
    assert any(m.update(0.95, 0.9, True) for _ in range(100))  # median moves away from the centre


def test_gaze_model_cv_picks_regularization():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 4))
    y = np.c_[0.5 + 0.1 * X[:, 0], 0.5 + 0.1 * X[:, 1]] + rng.normal(0, 0.01, (60, 2))
    m = GazeModel().fit(X, y, np.repeat(np.arange(6), 10), [0.01, 1.0, 1e6])
    assert m.alpha != 1e6 and np.abs(m.predict(X) - y).mean() < 0.03
    assert m.predict(np.full(4, 1e9)).max() <= 1.5  # clipped
