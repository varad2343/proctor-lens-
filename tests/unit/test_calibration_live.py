import time

import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Face
from proctorlens.perception.gaze import fit_calibration
from proctorlens.pipeline.calibration import collect_adaptive, dot_schedule

BLANK = np.zeros((4, 4, 3), np.uint8)


def _face(x, y, rng):
    """Synthetic face whose head pose / iris / blendshapes linearly encode screen point (x, y)."""
    dx, dy = x - 0.5, y - 0.5
    n = lambda s: rng.normal(0, s)  # noqa: E731
    bl = {"eyeLookInLeft": 0.3 - 0.4 * dx, "eyeLookOutLeft": 0.3 + 0.4 * dx, "eyeLookUpLeft": 0.3 - 0.4 * dy}
    return Face(bbox=(0.3, 0.2, 0.7, 0.8), yaw=40 * dx + n(0.5), pitch=-30 * dy + n(0.5), roll=n(0.5),
                iris=(0.5 + 0.3 * dx + n(0.005), 0.5 + 0.3 * dy + n(0.005), 0.5 + 0.3 * dx, 0.5 + 0.3 * dy),
                eye_open=(1.0, 1.0), blend=bl, center=(0.5 + 0.03 * dx, 0.5 + 0.02 * dy), scale=0.6)


def _cfg(**gaze):
    cfg = Config()
    cfg.gaze.settle_s, cfg.gaze.dot_timeout_s = 0.0, 0.2  # dwell_samples stays at the default: 3 is too noisy to fit
    for k, v in gaze.items():
        setattr(cfg.gaze, k, v)
    return cfg


def _run(cfg, mode="full", face=True, key_at=None, key=-1, read=lambda: BLANK):
    """Drive collect_adaptive headlessly: the 'subject' looks exactly at whichever dot is shown."""
    dots, cur, rng, shown = dot_schedule(cfg.gaze, rng=0, mode=mode), {}, np.random.default_rng(0), [0]

    def show(d, n, total, seen, frame):
        cur["d"], shown[0] = d, shown[0] + 1
        return key if key_at is not None and shown[0] == key_at else -1

    def faces_fn(frame, t_ms):
        d = cur.get("d")
        return [_face(d.x, d.y, rng)] if face and d else []

    samples, action = collect_adaptive(faces_fn, read, show, cfg, dots)
    return dots, samples, action, shown[0]


def test_adaptive_moves_on_after_dwell_samples_and_fits():
    cfg = _cfg()
    t = time.monotonic()
    dots, samples, action, _ = _run(cfg)
    assert action == "done" and time.monotonic() - t < 2  # no fixed 1.5 s per dot
    assert len(samples) == len(dots) * cfg.gaze.dwell_samples
    cal = fit_calibration(samples, cfg.gaze)
    assert cal.accepted and cal.mode == "full" and cal.error < 0.05, cal.error


def test_quit_and_skip_keys_end_it_immediately():
    for key, expect in ((27, "quit"), (ord("q"), "quit"), (ord("s"), "skip")):
        _, samples, action, shown = _run(_cfg(), key_at=5, key=key)
        assert action == expect and shown == 5, (key, action, shown)


def test_no_face_times_out_instead_of_stalling():
    cfg = _cfg(dot_timeout_s=0.02)
    t = time.monotonic()
    dots, samples, action, _ = _run(cfg, face=False)
    assert action == "done" and samples == [] and time.monotonic() - t < 3


def test_dropped_reads_tolerated_but_dead_camera_raises():
    frames = iter([None, None, BLANK])  # then keeps returning BLANK
    _, samples, action, _ = _run(_cfg(), read=lambda: next(frames, BLANK))
    assert action == "done" and samples
    t0 = time.monotonic()
    try:
        # fake the last good frame being long ago by making every read fail
        collect_adaptive(lambda f, t: [], lambda: None, lambda *a: -1, _cfg(), dot_schedule(_cfg().gaze))
    except RuntimeError as e:
        assert "camera" in str(e) and time.monotonic() - t0 < 5
    else:
        raise AssertionError("dead camera should raise")


def test_quick_and_baseline_modes_are_shorter():
    g = Config().gaze
    full, quick, base = (dot_schedule(g, rng=1, mode=m) for m in ("full", "quick", "baseline"))
    assert [d.phase for d in base] == ["neutral"]
    assert sum(d.phase == "grid" for d in quick) == 5 and sum(d.phase == "validation" for d in quick) == 2
    assert len(base) < len(quick) < len(full)
    assert quick[-1].t_end_ms < 0.6 * full[-1].t_end_ms
    # baseline: a head-pose baseline but no gaze regression
    cfg = _cfg()
    _, samples, _, _ = _run(cfg, mode="baseline")
    cal = fit_calibration(samples, cfg.gaze)
    assert cal.model is None and cal.mode == "head_pose_only" and not cal.accepted
    # quick still gives a usable fit on clean data
    _, samples, _, _ = _run(cfg, mode="quick")
    cal = fit_calibration(samples, cfg.gaze)
    assert cal.model is not None and cal.error < 0.1, cal.error
