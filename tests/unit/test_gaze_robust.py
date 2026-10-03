import dataclasses
import math
import os
import tempfile
import time

import numpy as np

from proctorlens.core.config import Config, GazeCfg
from proctorlens.core.types import Face
from proctorlens.perception.gaze import (
    Baseline,
    Calibration,
    CalibSample,
    GazeModel,
    _clean,
    fit_calibration,
    gaze_features,
)
from proctorlens.pipeline.calibration import collect_adaptive, dot_schedule

GRID = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)]
VAL = [(0.25, 0.3), (0.75, 0.3), (0.3, 0.75), (0.7, 0.7)]
SEEDS = range(8)
BLANK = np.zeros((4, 4, 3), np.uint8)


def _face(x, y, rng, wobble=0.0):
    """Synthetic face whose head pose / iris / blendshapes linearly encode screen point (x, y); wobble = head jitter."""
    dx, dy = x - 0.5, y - 0.5
    n = lambda s: rng.normal(0, s)
    bl = {"eyeLookInLeft": 0.3 - 0.4 * dx, "eyeLookOutLeft": 0.3 + 0.4 * dx, "eyeLookInRight": 0.3 + 0.4 * dx,
          "eyeLookOutRight": 0.3 - 0.4 * dx, "eyeLookUpLeft": 0.3 - 0.4 * dy, "eyeLookDownLeft": 0.3 + 0.4 * dy,
          "eyeLookUpRight": 0.3 - 0.4 * dy, "eyeLookDownRight": 0.3 + 0.4 * dy}
    return Face(bbox=(0.3, 0.2, 0.7, 0.8), yaw=40 * dx + n(0.5 + 2 * wobble), pitch=-30 * dy + n(0.5 + 2 * wobble),
                roll=n(0.5 + 1.5 * wobble), eye_open=(1.0, 1.0), blend={k: v + n(0.01 + 0.01 * wobble) for k, v in bl.items()},
                iris=tuple(0.5 + 0.3 * (dx if i % 2 == 0 else dy) + n(0.005 + 0.01 * wobble) for i in range(4)),
                center=(0.5 + 0.03 * dx + n(0.01 * wobble), 0.5 + 0.02 * dy + n(0.01 * wobble)),
                scale=0.6 + n(0.02 * wobble))


def _wild(x, y, rng):
    """A sample that should not be trusted: blink, tracking glitch (yaw / iris), or still on another dot."""
    f, k = _face(x, y, rng), rng.integers(4)
    if k == 0:  # blink: lids shut, iris and eye-look blendshapes unreliable
        return dataclasses.replace(f, eye_open=(0.1, 0.12), iris=(0.5, rng.uniform(0.6, 1), 0.5, rng.uniform(0.6, 1)),
                                   blend={b: v + rng.normal(0, 0.3) for b, v in f.blend.items()})
    if k == 1:
        return dataclasses.replace(f, yaw=f.yaw + rng.choice([-1, 1]) * rng.uniform(20, 40))
    if k == 2:
        return dataclasses.replace(f, iris=tuple(rng.uniform(0, 1, 4)))
    return _face(rng.uniform(), rng.uniform(), rng)


def _samples(rng, p_wild=0.0, wobble=0.0, n=10):
    mk = lambda x, y: _wild(x, y, rng) if rng.random() < p_wild else _face(x, y, rng, wobble)
    s = [CalibSample(mk(.5, .5), .5, .5, "neutral", 0) for _ in range(n)]
    for i, (x, y) in enumerate(GRID, 1):
        s += [CalibSample(mk(x, y), x, y, "grid", i) for _ in range(n)]
    for i, (x, y) in enumerate(VAL, 10):
        s += [CalibSample(mk(x, y), x, y, "validation", i) for _ in range(n)]
    return s


def _errors(**kw):
    """Median validation error over SEEDS: (robust off, robust on)."""
    out = []
    for robust in (False, True):
        out.append(float(np.median([fit_calibration(_samples(np.random.default_rng(s), **kw), GazeCfg(robust=robust)).error
                                    for s in SEEDS])))
    return out


def test_robust_off_is_the_original_fit():
    s = _samples(np.random.default_rng(0), p_wild=0.15)  # even with outliers: every raw row is used
    cal = fit_calibration(s, GazeCfg(robust=False))
    grid = [x for x in s if x.phase == "grid"]
    X, base = np.array([gaze_features(x.face, cal.baseline) for x in grid]), cal.baseline
    ref = GazeModel().fit(X, [(x.x, x.y) for x in grid], [x.dot_id for x in grid], GazeCfg().ridge_alphas)
    assert cal.n_dropped == 0 and all(np.array_equal(getattr(cal.model, k), getattr(ref, k)) for k in ("mean", "std", "w", "b"))
    errs = []
    for i in range(10, 14):
        v = [x for x in s if x.dot_id == i]
        px, py = np.median(ref.predict(np.array([gaze_features(x.face, base) for x in v])), axis=0)
        errs.append(math.hypot(px - v[0].x, py - v[0].y))
    assert cal.error == float(np.median(errs))


def test_outliers_robust_error_clearly_lower():
    off, on = _errors(p_wild=0.15)  # 15% blink / yaw glitch / iris glitch / wrong-dot samples
    assert on < off / 3 and on < 0.02, (off, on)
    cal = fit_calibration(_samples(np.random.default_rng(1), p_wild=0.15), GazeCfg())
    assert 10 < cal.n_dropped < 40, cal.n_dropped  # about the injected 15% of the 130 grid + validation rows
    assert len(cal.dot_errors) == 4 and cal.error == float(np.median(cal.dot_errors))


def test_clean_data_and_head_wobble_not_worse():
    off, on = _errors()
    assert on <= off + 0.005, (off, on)
    off, on = _errors(wobble=1.0)
    assert on <= off + 0.005, (off, on)
    assert fit_calibration(_samples(np.random.default_rng(2)), GazeCfg()).n_dropped <= 2  # clean rows are kept


def test_blinks_are_dropped_and_half_a_dot_is_kept():
    rng = np.random.default_rng(3)
    s = [x for x in _samples(rng) if x.phase == "grid"]
    blink = dataclasses.replace(s[0].face, eye_open=(0.05, 0.05))  # identical pose, lids shut
    s[:3] = [CalibSample(blink, x.x, x.y, x.phase, x.dot_id) for x in s[:3]]
    _, _, keep = _clean(s, Baseline(), 3.0)
    assert not keep[:3].any() and keep[3:].mean() > 0.97  # the three blinks are gone, clean rows stay
    # a dot whose 10 samples are all over the place: at most half of it is dropped, never the whole dot
    mess = [CalibSample(_face(rng.uniform(), rng.uniform(), rng), .1, .1, "grid", 1) for _ in range(10)]
    _, _, keep = _clean(mess + [x for x in s if x.dot_id != 1], Baseline(), 3.0)
    assert keep[:10].sum() == 5 and keep[10:].mean() > 0.97
    assert _clean([], Baseline(), 3.0)[2].size == 0  # no samples (e.g. no validation phase) is fine


def test_diagnostics_roundtrip_and_old_calibration_files_still_load():
    cal = fit_calibration(_samples(np.random.default_rng(4), p_wild=0.15), GazeCfg())
    assert cal.n_dropped > 0 and cal.dot_errors
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        cal.save(path)
        back = Calibration.load(path)
        assert back.n_dropped == cal.n_dropped and back.dot_errors == cal.dot_errors and back.error == cal.error
        old = cal.to_dict()  # a calib.json written before the diagnostics existed
        del old["n_dropped"], old["dot_errors"]
        back = Calibration.from_dict(old)
        assert back.n_dropped == 0 and back.dot_errors == [] and back.mode == cal.mode and back.error == cal.error


def _collect(robust, ramp=10, restless=False, seed=0):
    """Headless collect_adaptive; the subject needs `ramp` frames to turn to each new dot, dwell starts at once."""
    cfg = Config()
    cfg.gaze.settle_s, cfg.gaze.dot_timeout_s, cfg.gaze.robust = 0.0, 0.05 if restless else 0.5, robust
    rng, st = np.random.default_rng(seed), {"d": None, "from": (0.5, 0.5), "at": (0.5, 0.5), "k": 0}

    def show(d, n, total, seen, frame):
        if st["d"] is not d:
            st.update(d=d, k=0, **{"from": st["at"]})
        return -1

    def faces_fn(frame, t_ms):
        d = st["d"]
        if d is None:
            return []
        st["k"] += 1
        a = float(np.clip(st["k"] / ramp, 0, 1))
        st["at"] = (st["from"][0] + a * (d.x - st["from"][0]), st["from"][1] + a * (d.y - st["from"][1]))
        f = _face(*st["at"], rng)
        return [dataclasses.replace(f, yaw=f.yaw + 12 * (-1) ** st["k"]) if restless else f]  # restless: jumps every frame

    dots = dot_schedule(cfg.gaze, rng=seed)
    return cfg, dots, collect_adaptive(faces_fn, lambda: BLANK, show, cfg, dots)[0]


def test_stability_gate_skips_moving_samples():
    def moving(samples):  # samples whose yaw is still far from where this dot's pose settles
        return np.mean([abs(s.face.yaw - 40 * (s.x - 0.5)) > 3 for s in samples])

    _, _, raw = _collect(robust=False)
    cfg, dots, gated = _collect(robust=True)
    assert moving(raw) > 0.3 and moving(gated) < moving(raw) / 3, (moving(raw), moving(gated))
    assert len(gated) == len(dots) * cfg.gaze.dwell_samples
    assert fit_calibration(raw, GazeCfg(robust=False)).error > 0.1  # fails the accuracy bar without the gate
    assert fit_calibration(gated, cfg.gaze).error < 0.02


def test_stability_gate_never_stalls_on_a_restless_subject():
    t = time.monotonic()
    cfg, dots, s = _collect(robust=True, restless=True)  # never stable: dot_timeout_s ends every dot
    t_on = time.monotonic() - t
    assert t_on < 5 and len(s) == len(dots) * cfg.gaze.dwell_samples  # topped up from the unstable samples
    assert {x.dot_id for x in s} == {d.dot_id for d in dots}  # no dot comes out empty
    t = time.monotonic()
    _, _, off = _collect(robust=False, restless=True)  # robust=False: every face counts, the dot ends at 10 faces
    assert len(off) == len(s) and time.monotonic() - t < t_on
