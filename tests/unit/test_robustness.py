"""Perturbations are pure frame->frame functions; perturb_video round-trips; degraded_switch_rate on a hand-made case."""
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ml.evaluation.robustness import (PERTURBATIONS, blur, brightness, crop, degraded_switch_rate, downscale, flip,
                                      gamma, jpeg, noise, perturb_video)


def _frame(h=120, w=160):
    rng = np.random.default_rng(0)
    f = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    f[:, : w // 2] //= 2  # left half darker => not left/right symmetric
    return f


def _sharp(f):
    return cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def test_frame_functions():
    f = _frame()
    for name, fn in PERTURBATIONS.items():  # every registered perturbation: same dtype, valid, input untouched
        g, before = fn(f), f.copy()
        assert g.dtype == np.uint8 and g.ndim == 3 and g.shape[2] == 3, name
        assert (f == before).all(), name
    assert gamma(f, 2.0).mean() < f.mean() < gamma(f, 0.5).mean()
    assert (gamma(f, 1.0) == f).all()
    assert brightness(f, 0.5).mean() < f.mean() and (brightness(f, 1.0) == f).all()
    assert _sharp(blur(f, 3.0)) < _sharp(f)
    assert jpeg(f, 30).shape == f.shape and (jpeg(f, 30) != f).any()
    assert len(cv2.imencode(".jpg", jpeg(f, 30))[1]) < len(cv2.imencode(".jpg", jpeg(f, 90))[1])
    assert downscale(f, 60).shape == (60, 80, 3) and downscale(f, 500).shape == f.shape
    n = noise(f, 20.0)
    assert (n == noise(f, 20.0)).all() and (n != f).any() and abs(float(n.mean()) - float(f.mean())) < 3  # deterministic
    assert (flip(flip(f)) == f).all() and (flip(f) != f).any()
    c = crop(f, 0.5, dx=1.0)
    assert c.shape == f.shape and (c != f).any() and (crop(f, 1.0) == f).all()


def test_perturb_video_roundtrip():
    f = _frame()
    with tempfile.TemporaryDirectory() as d:
        src, dst = str(Path(d) / "a.avi"), str(Path(d) / "b.avi")
        w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
        if not w.isOpened():
            return  # no MJPG writer in this OpenCV build; nothing to check
        for _ in range(6):
            w.write(f)
        w.release()
        assert perturb_video(src, dst, lambda x: downscale(x, 60)) == 6
        cap = cv2.VideoCapture(dst)
        ok, g = cap.read()
        cap.release()
        assert ok and g.shape == (60, 80, 3)
        assert perturb_video(str(Path(d) / "missing.avi"), dst, flip) == 0  # unreadable source: nothing written


def test_degraded_switch_rate():
    E = lambda t, s, e: dict(type=t, start_ms=s * 1000, end_ms=e * 1000)
    A, D = "OFF_SCREEN_SUSTAINED", "MONITORING_DEGRADED"
    clean = [E(A, 10, 20), E("PROHIBITED_OBJECT", 50, 60), E(A, 100, 110)]
    # perturbed: A@10-20 kept; phone lost while degraded covers it; A@100-110 lost silently; new A@200-210 silent
    # and another new A@300-310 during a degraded interval
    pert = [E(A, 10, 20), E(D, 45, 65), E(A, 200, 210), E(A, 300, 310), E(D, 295, 320)]
    r = degraded_switch_rate(clean, pert)
    assert (r["n_changed"], r["n_covered"], r["rate"]) == (4, 2, 0.5)
    same = degraded_switch_rate(clean, clean + [E(D, 0, 5)])
    assert same["n_changed"] == 0 and math.isnan(same["rate"])  # nothing changed: rate undefined
