import math
import os
import tempfile

import numpy as np

from proctorlens.core.config import Config, GazeCfg
from proctorlens.core.types import Detections, Face, Perceived
from proctorlens.pipeline.calibration import (calibrate_from_frames, calibrate_from_video, dot_schedule,
                                              read_log, write_log)

GRID = {(x, y) for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)}


def _face(x, y, rng):
    """Synthetic face whose head pose / iris / blendshapes linearly encode screen point (x, y)."""
    dx, dy = x - 0.5, y - 0.5
    n = lambda s: rng.normal(0, s)  # noqa: E731
    bl = {"eyeLookInLeft": 0.3 - 0.4 * dx, "eyeLookOutLeft": 0.3 + 0.4 * dx, "eyeLookUpLeft": 0.3 - 0.4 * dy}
    return Face(bbox=(0.3, 0.2, 0.7, 0.8), yaw=40 * dx + n(0.5), pitch=-30 * dy + n(0.5), roll=n(0.5),
                iris=(0.5 + 0.3 * dx + n(0.005), 0.5 + 0.3 * dy + n(0.005), 0.5 + 0.3 * dx, 0.5 + 0.3 * dy),
                eye_open=(1.0, 1.0), blend=bl, center=(0.5 + 0.03 * dx, 0.5 + 0.02 * dy), scale=0.6)


class FakePerceiver:
    """Looks at the dot shown at t; during the first 0.5 s of each dot (saccade) it is still elsewhere."""

    def __init__(self, dots):
        self.dots, self.rng, self.calls = dots, np.random.default_rng(0), 0

    def perceive(self, frame, t_ms, idx):
        self.calls += 1
        # a frame exactly on a boundary still shows the old dot
        d = next((d for d in self.dots if d.t_start_ms < t_ms <= d.t_end_ms), self.dots[-1])
        x, y = (0.0, 1.0) if t_ms - d.t_start_ms < 500 else (d.x, d.y)
        return Perceived(t_ms, [_face(x, y, self.rng)], Detections(), None, 100.0, 100.0, 0.0, 1.0, [])


def test_dot_schedule_shape_and_log_roundtrip():
    cfg = GazeCfg()
    dots = dot_schedule(cfg, rng=7)
    assert [d.phase for d in dots] == ["neutral"] + ["grid"] * 9 + ["validation"] * cfg.n_validation
    assert [d.dot_id for d in dots] == list(range(len(dots)))
    assert (dots[0].x, dots[0].y, dots[0].t_start_ms, dots[0].t_end_ms) == (0.5, 0.5, 0, 2000)
    assert all(a.t_end_ms == b.t_start_ms for a, b in zip(dots, dots[1:]))  # contiguous
    assert all(d.t_end_ms - d.t_start_ms == 1500 for d in dots[1:])
    assert {(d.x, d.y) for d in dots if d.phase == "grid"} == GRID
    assert all(min(math.dist((d.x, d.y), g) for g in GRID) >= 0.15 for d in dots if d.phase == "validation")
    assert dots == dot_schedule(cfg, rng=7) and dots != dot_schedule(cfg, rng=8)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calib_log.csv")
        write_log(dots, path)
        assert open(path).readline().strip() == "dot_id,phase,x,y,t_start_ms,t_end_ms"
        assert read_log(path) == dots


def test_calibrate_from_frames_skips_settling_and_fits():
    cfg = Config()
    dots = dot_schedule(cfg.gaze, rng=1)
    end, blank = dots[-1].t_end_ms, np.zeros((4, 4, 3), np.uint8)
    per = FakePerceiver(dots)
    cal = calibrate_from_frames(((t, blank) for t in range(0, end + 3000, 100)), dots, per, cfg)
    assert cal.accepted and cal.mode == "full" and cal.error < 0.05, cal.error
    assert per.calls <= end // 100 + 2  # frames after the last dot are not processed


def test_calibrate_from_video_roundtrip():
    import cv2

    cfg, dots = Config(), dot_schedule(Config().gaze, rng=2)
    with tempfile.TemporaryDirectory() as tmp:
        vid, log = os.path.join(tmp, "calib.avi"), os.path.join(tmp, "calib.csv")
        w = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not w.isOpened():
            return  # no MJPG writer in this OpenCV build
        for _ in range(dots[-1].t_end_ms // 100 + 10):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
        write_log(dots, log)
        cal = calibrate_from_video(vid, log, cfg, FakePerceiver(dots))
    assert cal.accepted and cal.error < 0.05, cal.error
