import math

import numpy as np

from proctorlens.core.types import Detections, Event, Face, Perceived
from proctorlens.explain.overlays import draw


def _perceived():
    f = Face(bbox=(0.3, 0.2, 0.6, 0.8), yaw=30.0, pitch=-10.0, roll=5.0, iris=(0.5,) * 4, eye_open=(0.6, 0.6), blend={},
             center=(0.45, 0.5), scale=0.6)
    det = Detections(0.9, 0.0, [], [("phone", 0.9, (0.65, 0.5, 0.85, 0.8))])
    return Perceived(0, [f, Face(**{**f.__dict__, "static": True})], det, None, 100.0, 80.0, 0.0, 0.8, [])


def test_draw_annotates_a_copy_and_tolerates_missing_inputs():
    frame = np.zeros((240, 320, 3), np.uint8)
    row = dict(zone="right", off_screen_score=0.9, gaze_x=1.2, gaze_y=0.4, mouth_energy_1s=0.7, quality=0.8,
               quality_reasons="", reliable=True)
    ongoing = [Event("OFF_SCREEN_SUSTAINED", 1000, 6200, 0.9, "rule@0.1.0", status="ongoing")]
    out = draw(frame, _perceived(), row, ongoing)
    assert out.shape == frame.shape and out.dtype == frame.dtype and out is not frame
    assert out.any() and not frame.any()  # drawn on a copy
    # degraded frame: no perceived, NaN gaze, unreliable row -> still draws (banner)
    bad = draw(frame, None, dict(quality=math.nan, reliable=False, quality_reasons="frame_gap"), [])
    assert bad.any()
    assert draw(frame, None, None, []).any()  # nothing known at all: banner only, no crash
    assert draw(np.zeros((48, 64, 3), np.uint8), _perceived(), row, ongoing).shape == (48, 64, 3)  # tiny frames


def test_track_labels_and_mesh():
    frame = np.zeros((240, 320, 3), np.uint8)
    row = dict(zone="right", off_screen_score=0.9, quality=0.8, quality_reasons="", reliable=True)
    plain = draw(frame, _perceived(), row, [])
    p = _perceived()
    p.faces[0].track, p.det.ids = 3, [7]
    labelled = draw(frame, p, row, [])
    assert not np.array_equal(labelled, plain)  # "face 3", "phone 7 0.90"
    p.det.ids = [7, 8, 9]  # more (or fewer) ids than boxes must not break drawing
    draw(frame, p, row, [])
    p.det.ids = []
    draw(frame, p, row, [])
    # mesh: only drawn when present (and the face is big enough), so untracked, mesh-less input keeps the old pixels
    m = _perceived()
    rng = np.random.default_rng(0)
    m.faces[0].mesh = np.c_[rng.uniform(0.32, 0.58, 478), rng.uniform(0.22, 0.78, 478)].astype(np.float32)
    meshed = draw(frame, m, row, [])
    assert not np.array_equal(meshed, plain)
    m.faces[0].mesh = m.faces[0].mesh[:10]  # fewer points than the contour edges reference: points only, no crash
    draw(frame, m, row, [])
    tiny = _perceived()
    tiny.faces[0].mesh = m.faces[0].mesh
    small = np.zeros((48, 64, 3), np.uint8)  # face box ~29 px tall: below the mesh minimum
    assert np.array_equal(draw(small, tiny, row, []), draw(small, _perceived(), row, []))


def test_draw_hud_kwargs_are_optional_and_backwards_compatible():
    frame = np.zeros((240, 320, 3), np.uint8)
    row = dict(zone="right", off_screen_score=0.9, quality=0.8, quality_reasons="", reliable=True)
    base = draw(frame, _perceived(), row, [])
    same = draw(frame, _perceived(), row, [], fps=None, zone=None, calib_mode=None)
    assert np.array_equal(base, same)  # no HUD kwargs (or NaN fps / empty text) = the old picture, pixel for pixel
    assert np.array_equal(base, draw(frame, _perceived(), row, [], fps=math.nan, zone="", calib_mode=""))
    bar = 4 + int(24 * max(0.35, 240 / 700))  # banner height at this size
    for kw in (dict(fps=12.5), dict(zone="left"), dict(calib_mode="full"),
               dict(fps=12.5, zone="left", calib_mode="head_pose_only")):
        hud = draw(frame, _perceived(), row, [], **kw)
        assert hud.shape == frame.shape and not np.array_equal(hud, base), kw
        assert not np.array_equal(hud[bar:2 * bar + 1], base[bar:2 * bar + 1]), kw  # drawn in the strip under the banner
        assert np.array_equal(hud[:bar], base[:bar]) and np.array_equal(hud[2 * bar + 1:], base[2 * bar + 1:]), kw
    assert draw(frame, None, None, [], fps=30.0, zone="none", calib_mode="none").any()  # frame gap: still draws
