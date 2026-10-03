import dataclasses
import itertools
import math
from pathlib import Path

import numpy as np

from proctorlens.core.config import Config, SmoothCfg
from proctorlens.core.types import Detections, Face, Perceived
from proctorlens.features.extractor import FeatureExtractor
from proctorlens.features.smooth import FaceSmoother, OneEuro

ROOT = Path(__file__).resolve().parents[2]
BLEND = {"jawOpen": 0.1, "eyeLookInLeft": 0.3, "eyeLookDownLeft": 0.2}


def _face(**kw) -> Face:
    f = Face((0.3, 0.2, 0.7, 0.8), 0.0, 0.0, 0.0, (0.5, 0.5, 0.5, 0.5), (1.0, 1.0), dict(BLEND), (0.5, 0.5), 0.6)
    return dataclasses.replace(f, **kw)


def _oe() -> OneEuro:
    c = SmoothCfg()
    return OneEuro(c.min_cutoff, c.beta, c.d_cutoff, c.speed_floor)


def test_one_euro_first_sample_constant_and_nan():
    f = _oe()
    assert f(7.0, 1000) == 7.0  # first sample passes through
    assert all(f(7.0, 1000 + 100 * k) == 7.0 for k in range(1, 50))  # a constant stays exactly constant
    assert f(7.0, 1000) == 7.0 and f(9.0, 1000) == 7.0  # duplicate / out-of-order time: held, state untouched
    assert math.isnan(f(math.nan, 6000)) and f.x is None  # NaN passes through and does not poison the state
    assert f(3.0, 6100) == 3.0  # restart at the next finite sample


def test_one_euro_is_causal_and_dt_aware():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 40)
    y = x.copy()
    y[25:] += 50.0  # the future differs...
    a, b = _oe(), _oe()
    assert [a(v, 100 * i) for i, v in enumerate(x)][:25] == [b(v, 100 * i) for i, v in enumerate(y)][:25]  # ...the past does not

    def after_1s(hz: int) -> float:  # a pure low-pass (no speed boost) after a unit step, 1 s later
        f = OneEuro(0.25, 0.0, 0.5)
        f(0.0, 0)
        return [f(1.0, round(1000 * k / hz)) for k in range(1, hz + 1)][-1]

    assert abs(after_1s(10) - after_1s(100)) < 0.03  # same wall-clock response at 10 and 100 Hz


def test_one_euro_quiet_when_still_quick_when_moving():
    rng = np.random.default_rng(1)
    still = [rng.normal(0, 1.0) for _ in range(600)]
    f = _oe()
    out = [f(v, 100 * i) for i, v in enumerate(still)]
    assert np.std(still[20:]) > 3 * np.std(out[20:])  # >= 3x less jitter at rest (sigma 1 deg)
    f = _oe()
    for i in range(20):
        f(0.0, 100 * i)
    ys = [f(40.0, 2000 + 100 * k) for k in range(4)]
    assert ys[0] > 0.9 * 40  # a fast 40 deg jump is followed within one grid step (speed opens the cutoff)
    assert all(b >= a for a, b in itertools.pairwise(ys)) and ys[-1] <= 40.0 + 1e-9  # no overshoot


def test_face_smoother_touches_only_pose_and_eye_inputs():
    sm, f0 = FaceSmoother(SmoothCfg()), _face()
    sm(f0, 0)
    f1 = _face(yaw=3.0, iris=(0.6, 0.5, 0.5, 0.5), blend={**BLEND, "jawOpen": 0.9, "eyeLookInLeft": 0.8},
               eye_open=(0.4, 0.4), bbox=(0.31, 0.2, 0.7, 0.8), center=(0.51, 0.5), scale=0.61)
    g = sm(f1, 100)
    assert 0.0 < g.yaw < 3.0 and 0.5 < g.iris[0] < 0.6 and 0.3 < g.blend["eyeLookInLeft"] < 0.8  # smoothed
    assert g.blend["jawOpen"] == 0.9 and g.eye_open == (0.4, 0.4) and g.bbox == f1.bbox  # left alone
    assert "eyeLookOutLeft" not in g.blend  # a missing blendshape stays missing
    assert f1.yaw == 3.0 and f1.blend["eyeLookInLeft"] == 0.8  # the input Face is not modified


def test_face_smoother_hold_keeps_eye_values_not_pose_and_reset_restarts():
    sm = FaceSmoother(SmoothCfg())
    for k in range(20):
        sm(_face(), 100 * k)
    shut = _face(yaw=1.0, iris=(0.5, 0.95, 0.5, 0.95), blend={**BLEND, "eyeLookDownLeft": 0.9})
    held = sm(shut, 2000, hold_eyes=True)
    assert held.iris == (0.5, 0.5, 0.5, 0.5) and held.blend["eyeLookDownLeft"] == 0.2  # eyes: last good values
    assert held.yaw > 0.0  # head pose keeps following (a real head turn during a blink is not delayed)
    free = sm(shut, 2100)  # blink over: eye signals follow again
    assert free.iris[1] > 0.5
    sm.reset()
    assert sm(_face(yaw=9.0), 3000).yaw == 9.0  # after a reset the first sample passes through
    assert FaceSmoother(SmoothCfg())(shut, 0, hold_eyes=True).iris[1] == 0.95  # nothing to hold yet: raw


def test_blink_hold_is_bounded_by_hold_s():
    cfg = Config()
    ex = FeatureExtractor(cfg, None)
    p = lambda t, f: Perceived(t, [f], Detections(), None, 120.0, 80.0, 0.0, 0.9, [])
    for k in range(20):
        ex.step(100 * k, p(100 * k, _face()), 0.0, 10.0)
    shut = _face(iris=(0.5, 0.95, 0.5, 0.95), eye_open=(0.1, 0.1))  # eyes closed and staying closed
    ly = [ex.step(2000 + 100 * k, p(2000 + 100 * k, shut), 0.0, 10.0)["iris_ly"] for k in range(20)]
    n = math.ceil(cfg.smooth.hold_s * 10)
    assert all(v == 0.5 for v in ly[:n])  # held for hold_s (5 steps at 10 Hz) ...
    assert ly[-1] > 0.9  # ... then the values pass through again: shut eyes are not a blink


def test_real_landmarker_jitter_is_reduced():
    """Real MediaPipe on a real face crop with per-frame sensor noise: smoothed pose / iris jitter is >= 2.5x lower."""
    try:
        import cv2
        import ultralytics

        from proctorlens.perception.landmarks import Landmarker
        img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
        model = ROOT / "data" / "models" / "face_landmarker.task"
        if img is None or not model.exists():
            return
        lm = Landmarker(str(model))
    except (ImportError, OSError, RuntimeError, ValueError):  # libs / model / image missing: skip
        return
    crop, rng, faces = img[100:460, 650:1050], np.random.default_rng(0), []
    for i in range(80):
        noisy = np.clip(crop + rng.normal(0, 3.0, crop.shape), 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 90))])
        found = lm.process(cv2.imdecode(buf, cv2.IMREAD_COLOR), i * 100)
        faces.append(found[0] if found else None)
    assert sum(f is not None for f in faces) >= 60  # the crop must contain a detectable face

    def run(enabled: bool) -> list[dict]:
        cfg = Config()
        cfg.smooth.enabled = enabled
        ex = FeatureExtractor(cfg, None)
        return [ex.step(100 * i, Perceived(100 * i, [f] if f else [], Detections(), None, 120.0, 80.0, 0.0, 0.9, []),
                        0.0, 10.0) for i, f in enumerate(faces)]

    raw, smooth = run(False), run(True)
    for c in ("d_yaw", "d_pitch", "iris_lx", "iris_ry"):
        a, b = (np.array([r[c] for r in rows][10:], float) for rows in (raw, smooth))
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        assert a.std() > 2.5 * b.std() > 0, (c, a.std(), b.std())
        assert abs(a.mean() - b.mean()) < a.std()  # same operating point, just steadier
