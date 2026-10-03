"""Perception front-end accuracy aids: low-light enhancement, small-face recovery, eye openness, pose convention.

Pure-logic tests need no models. The real-model tests (MediaPipe landmarker, YOLO, the two sample images that ship
with ultralytics) return early when a library, model file or image is missing.
"""
import copy
import pathlib

import cv2
import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Detections
from proctorlens.perception import Perceiver
from proctorlens.perception.enhance import gamma_lut, lowlight_enhance, mean_luma
from proctorlens.perception.head_pose import euler_from_matrix
from proctorlens.perception.landmarks import (face_from_landmarks, from_crop, head_rect, recover_faces,
                                              uncovered)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _lm(eye_gap: float = 0.015) -> np.ndarray:
    """478 points at the face centre, then the ones that matter placed by hand (bbox 0.3..0.7 x 0.2..0.8).
    Eye width 0.07, so eye_gap/0.07 is the eye-aspect ratio."""
    lm = np.tile([0.5, 0.5, 0.0], (478, 1))
    g = eye_gap / 2
    pts = {10: (0.5, 0.2), 152: (0.5, 0.8), 234: (0.3, 0.5), 454: (0.7, 0.5),
           33: (0.40, 0.40), 133: (0.47, 0.40), 159: (0.435, 0.40 - g), 145: (0.435, 0.40 + g), 468: (0.4175, 0.40),
           362: (0.53, 0.40), 263: (0.60, 0.40), 386: (0.565, 0.40 - g), 374: (0.565, 0.40 + g), 473: (0.5825, 0.40),
           78: (0.44, 0.65), 308: (0.56, 0.65), 13: (0.50, 0.64), 14: (0.50, 0.66)}
    for i, (x, y) in pts.items():
        lm[i, :2] = x, y
    return lm


# ---------------------------------------------------------------- low-light enhancement (pure)

def test_gamma_lut_aims_at_target_and_never_darkens():
    lut = gamma_lut(30.0)
    assert lut.dtype == np.uint8 and len(lut) == 256 and np.all(np.diff(lut.astype(int)) >= 0)
    assert abs(int(lut[30]) - 110) <= 3  # a frame of mean luma 30 lands near the target
    assert np.array_equal(gamma_lut(200.0), np.arange(256))  # already bright: identity
    assert gamma_lut(0.0)[128] == gamma_lut(1.0)[128] and gamma_lut(0.0)[128] > 128  # bounded boost on a black frame


def test_lowlight_enhance_only_when_dark():
    rng = np.random.default_rng(0)
    dark = np.clip(rng.normal(30, 5, (96, 128, 3)), 0, 255).astype(np.uint8)
    out = lowlight_enhance(dark, 90.0)
    assert out.shape == dark.shape and out.dtype == np.uint8 and out is not dark
    assert mean_luma(dark) < 40 and mean_luma(out) > 80  # brightened
    assert mean_luma(dark) < 40  # the input is not modified in place
    bright = np.clip(rng.normal(130, 20, (96, 128, 3)), 0, 255).astype(np.uint8)
    assert lowlight_enhance(bright, 90.0) is bright  # not dark: the very same frame, untouched
    assert lowlight_enhance(dark, 20.0) is dark  # threshold is configurable
    assert mean_luma(lowlight_enhance(np.zeros((48, 64, 3), np.uint8), 90.0)) < 20  # black stays (near) black: bounded boost


# ---------------------------------------------------------------- recovery geometry (pure)

def test_head_rect_and_from_crop():
    assert head_rect((0.4, 0.1, 0.6, 0.9), 1000, 800, 0.0) == (400, 80, 600, 280)  # square: min(200 wide, .45*640 tall)
    assert head_rect((0.4, 0.1, 0.6, 0.9), 1000, 800, 0.15) == (370, 50, 630, 310)  # grown by pad x side
    assert head_rect((0.0, 0.0, 0.2, 1.0), 1000, 800, 0.15) == (0, 0, 230, 230)  # clipped to the frame
    x0, y0, x1, y1 = head_rect((0.1, 0.05, 0.9, 0.3), 640, 480, 0.15)  # short, wide box: height limits the square
    assert x1 - x0 == y1 - y0 == round(0.45 * 0.25 * 480 * 1.3) and (x0 + x1) / 2 == 320
    lm = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.1], [1.0, 1.0, 0.0]])
    got = from_crop(lm, (100, 50, 300, 250), 1000, 500)
    assert np.allclose(got, [[0.1, 0.1, 0.0], [0.2, 0.3, 0.02], [0.3, 0.5, 0.0]])
    rng = np.random.default_rng(1)  # round trip: frame point -> crop-normalized -> frame
    rect, (w, h) = (120, 40, 420, 340), (800, 600)
    p = np.c_[rng.uniform(120, 420, 20) / w, rng.uniform(40, 340, 20) / h, np.zeros(20)]
    crop = np.c_[(p[:, 0] * w - rect[0]) / (rect[2] - rect[0]), (p[:, 1] * h - rect[1]) / (rect[3] - rect[1]), p[:, 2]]
    assert np.allclose(from_crop(crop, rect, w, h), p)


class FakeRecoverer:
    """Image-mode Landmarker stand-in: always 'sees' one face centred in the crop (and logs what it was given)."""

    def __init__(self, dx: float = 0.0, log: list | None = None):
        self.dx, self.crops, self.log = dx, [], log

    def detect(self, crop, t_ms=0):
        self.crops.append(crop.shape[:2])
        if self.log is not None:
            self.log.append("rec")
        lm = _lm()
        lm[:, 0] += self.dx
        return [(lm, {}, np.eye(4))]


def test_recover_faces_maps_back_and_dedups():
    frame = np.full((800, 1000, 3), 128, np.uint8)
    box = (0.4, 0.1, 0.6, 0.9)
    rec = FakeRecoverer()
    out = recover_faces(rec, frame, [box], [], 0.15, 384)
    assert len(out) == 1 and out[0].recovered and rec.crops and min(rec.crops[0]) >= 383  # upscaled to the min side
    f = out[0]
    assert box[0] <= f.center[0] <= box[2] and box[1] <= f.center[1] <= box[3] and abs(f.center[0] - 0.5) < 1e-6
    assert abs(f.scale - 0.6 * 260 / 800) < 1e-6  # crop is 260 px; bbox height .6 of it, over the 800 px frame
    assert abs((f.bbox[2] - f.bbox[0]) - 0.4 * 260 / 1000) < 1e-6
    # a face the main pass already has inside that person box -> no crop at all
    main = face_from_landmarks(_lm(), {}, np.eye(4))
    main.center = (0.5, 0.3)
    rec = FakeRecoverer()
    assert recover_faces(rec, frame, [box], [main], 0.15, 384) == [] and rec.crops == []
    # a face that bleeds in from the neighbouring person (centre outside this person's box) is dropped
    assert recover_faces(FakeRecoverer(dx=-0.4), frame, [box], [], 0.15, 384) == []
    # tiny head crops (distant people) are not worth a run
    rec = FakeRecoverer()
    assert recover_faces(rec, np.zeros((100, 100, 3), np.uint8), [(0.45, 0.4, 0.55, 0.9)], [], 0.15, 384) == []
    assert rec.crops == []
    # uncovered(): only boxes without a face centre, largest first
    small, big, covered = (0.0, 0.0, 0.2, 0.2), (0.3, 0.5, 0.9, 0.95), (0.4, 0.2, 0.6, 0.4)  # main.center = (0.5, 0.3)
    assert uncovered([small, big, covered], [main]) == [big, small]


# ---------------------------------------------------------------- eye openness (pure)

def test_eye_open_from_blink_blendshapes_and_landmarks():
    m = np.eye(4)
    f = face_from_landmarks(_lm(), {"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.3}, m)  # left = subject's left eye
    assert np.allclose(f.eye_open, (0.9, 0.7), atol=1e-9)
    f = face_from_landmarks(_lm(), {"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.97}, m)
    assert max(f.eye_open) < 0.06  # closed
    f = face_from_landmarks(_lm(), {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0}, m)
    assert f.eye_open == (1.0, 1.0)
    # no (or only one) blink blendshape -> landmark eye-aspect ratio, scaled so a typical open eye (0.30-0.34) is ~0.8-0.95
    for bl in ({}, {"eyeBlinkLeft": 0.2}):
        assert np.allclose(face_from_landmarks(_lm(0.07 * 0.33), bl, m).eye_open, (0.33 / 0.36,) * 2, atol=1e-6)
    assert all(0.8 < v < 1.0 for v in face_from_landmarks(_lm(0.07 * 0.33), {}, m).eye_open)
    assert face_from_landmarks(_lm(0.0), {}, m).eye_open == (0.0, 0.0)  # closed lids, no blendshapes
    assert face_from_landmarks(_lm(0.07 * 0.6), {}, m).eye_open == (1.0, 1.0)  # wide open is clipped, not > 1


# ---------------------------------------------------------------- Perceiver wiring (fakes)

class FakeLM:
    def __init__(self, log: list, faces=()):
        self.log, self.faces, self.seen = log, list(faces), []

    def process(self, frame, t_ms):
        self.log.append("lm")
        self.seen.append(frame)
        return list(self.faces)


class FakeDet:
    def __init__(self, log: list, boxes=()):
        self.log, self.boxes = log, list(boxes)

    def process(self, frame):
        self.log.append("det")
        return Detections(person_boxes=list(self.boxes))


def _cfg(**flags) -> Config:
    cfg = Config()
    cfg.models.identity = "no/such/dir"  # never build a real identity component
    for k, v in flags.items():
        setattr(cfg.perception, k, v)
    return cfg


def test_perceiver_enhances_landmarker_input_only():
    rng = np.random.default_rng(0)
    frame = np.clip(rng.normal(25, 4, (48, 64, 3)), 0, 255).astype(np.uint8)
    lm = FakeLM([])
    p = Perceiver(_cfg(lowlight_enhance=True), lm, FakeDet([]), None).perceive(frame, 0, 0)
    assert mean_luma(lm.seen[0]) > 80 and lm.seen[0] is not frame  # landmarker saw the brightened frame
    assert p.luma < 40 and "dark" in p.quality_reasons and p.quality < 0.5  # quality kept the original: still 'dark'
    lm = FakeLM([])
    p = Perceiver(_cfg(lowlight_enhance=False), lm, FakeDet([]), None).perceive(frame, 0, 0)
    assert lm.seen[0] is frame  # flag off: exactly the old input
    lit = np.full((48, 64, 3), 130, np.uint8)
    lm = FakeLM([])
    Perceiver(_cfg(lowlight_enhance=True), lm, FakeDet([]), None).perceive(lit, 0, 0)
    assert lm.seen[0] is lit  # lit frame: untouched even with the flag on


def test_perceiver_recovers_after_detector_and_respects_flag():
    frame = np.full((200, 250, 3), 128, np.uint8)
    box = (0.4, 0.1, 0.6, 0.9)
    log: list = []
    on = Perceiver(_cfg(recover_small_faces=True), FakeLM(log), FakeDet(log, [box]), None, FakeRecoverer(log=log))
    p = on.perceive(frame, 0, 0)
    assert log == ["det", "lm", "rec"]  # detector first (recovery needs its person boxes), then landmarker, then recovery
    assert len(p.faces) == 1 and p.faces[0].recovered and "no_face" not in p.quality_reasons
    p = on.perceive(frame, 100, 1)  # detections carried over (not fresh): recovery still uses the carried box
    assert not p.det.fresh and len(p.faces) == 1 and p.faces[0].recovered
    log2: list = []
    rec = FakeRecoverer(log=log2)
    off = Perceiver(_cfg(recover_small_faces=False), FakeLM(log2), FakeDet(log2, [box]), None, rec)
    p = off.perceive(frame, 0, 0)
    assert p.faces == [] and rec.crops == [] and "no_face" in p.quality_reasons  # old behaviour exactly
    # a main-pass face inside the person box -> no recovery; a second, uncovered person -> merged, area-sorted
    main = face_from_landmarks(_lm(), {}, np.eye(4))
    main.center = (0.5, 0.3)
    rec = FakeRecoverer()
    p = Perceiver(_cfg(), FakeLM([], [main]), FakeDet([], [box]), None, rec).perceive(frame, 0, 0)
    assert p.faces == [main] and rec.crops == []
    other = (0.05, 0.1, 0.25, 0.9)
    p = Perceiver(_cfg(), FakeLM([], [main]), FakeDet([], [box, other]), None, FakeRecoverer()).perceive(frame, 0, 0)
    assert len(p.faces) == 2 and p.faces[0] is main and p.faces[1].recovered
    area = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in p.faces]
    assert area[0] >= area[1]
    # no model file and no injected recoverer (a fake landmarker run): recovery is skipped, not an error
    p = Perceiver(_cfg(), FakeLM([]), FakeDet([], [box]), None).perceive(frame, 0, 0)
    assert p.faces == []


# ---------------------------------------------------------------- real models (skip when missing)

def _real():
    """(cfg, zidane BGR, bus BGR) or None when mediapipe/ultralytics/models/sample images are not available."""
    try:
        import mediapipe  # noqa: F401
        import ultralytics
    except (ImportError, OSError):  # OSError: ultralytics imports torch, whose native DLLs can fail to load
        return None
    assets = pathlib.Path(ultralytics.__file__).parent / "assets"
    cfg = Config()
    cfg.models.landmarker = str(ROOT / "data/models/face_landmarker.task")
    cfg.models.detector = str(ROOT / "data/models/yolo11n.pt")
    cfg.models.identity = "no/such/dir"
    imgs = [cv2.imread(str(assets / n)) for n in ("zidane.jpg", "bus.jpg")]
    if any(i is None for i in imgs) or not (pathlib.Path(cfg.models.landmarker).is_file()
                                            and pathlib.Path(cfg.models.detector).is_file()):
        return None
    return cfg, *imgs


_DET: list = []  # the YOLO model is loaded once for the file


def _perceive(cfg, img, **flags):
    """One frame through a fresh Perceiver (real landmarker + detector) with the given perception flags."""
    from proctorlens.perception.landmarks import Landmarker
    from proctorlens.perception.objects import ObjectDetector

    cfg = copy.deepcopy(cfg)
    for k, v in flags.items():
        setattr(cfg.perception, k, v)
    if not _DET:
        _DET.append(ObjectDetector(cfg.models.detector, cfg.models.detector_conf))
    return Perceiver(cfg, Landmarker(cfg.models.landmarker), _DET[0]).perceive(img, 0, 0)


def test_real_recovery_finds_small_faces():
    r = _real()
    if r is None:
        return
    cfg, zidane, bus = r
    for img, n_min in ((zidane, 1), (bus, 2)):  # measured: 0 -> 1 and 0 -> 2
        off = _perceive(cfg, img, recover_small_faces=False, lowlight_enhance=False)
        on = _perceive(cfg, img, recover_small_faces=True, lowlight_enhance=False)
        assert off.faces == [] and "no_face" in off.quality_reasons  # the landmarker alone misses them in the full frame
        assert n_min <= len(on.faces) <= len(on.det.person_boxes) and all(f.recovered for f in on.faces)
        assert all(f.scale < 0.3 and any(b[0] <= f.center[0] <= b[2] and b[1] <= f.center[1] <= b[3]
                                         for b in on.det.person_boxes) for f in on.faces)
        assert "no_face" not in on.quality_reasons and all(np.isfinite(f.yaw) for f in on.faces)


def _clean_face():
    """Image-mode landmarker + the zidane crop the landmarker finds (the known single face)."""
    r = _real()
    if r is None:
        return None
    from proctorlens.perception.landmarks import Landmarker

    return Landmarker(r[0].models.landmarker, num_faces=1, image_mode=True), r[1][100:460, 650:1050].copy()


def _dark(img, kind, v, seed=0):
    if kind == "gamma":
        return (255 * (img / 255.0) ** v).astype(np.uint8)
    d = (img * v).astype(np.uint8)
    return d if kind == "mul" else np.clip(d + np.random.default_rng(seed).normal(0, 4, d.shape), 0, 255).astype(np.uint8)


def test_real_lowlight_enhancement_helps_dark_faces():
    c = _clean_face()
    if c is None:
        return
    lm, img = c
    clean = lm.process(img, 0)
    assert len(clean) == 1
    levels = [("gamma", 2), ("gamma", 3), ("gamma", 4), ("mul", 0.3), ("mul", 0.15)]  # darkened versions of the real crop
    hard = [("noisy", 0.15, s) for s in range(4)] + [("noisy", 0.08, s) for s in range(4)] + [("gamma", 6), ("mul", 0.05)]
    seen = {}
    for name, lv in (("required", levels), ("hard", hard)):
        res = {}
        for on in (False, True):
            hits, err = 0, []
            for kind, v, *s in lv:
                d = _dark(img, kind, v, *s)
                f = lm.process(lowlight_enhance(d, 90.0) if on else d, 0)
                hits += bool(f)
                err += [abs(f[0].yaw - clean[0].yaw)] if f else []
            res[on] = (hits, float(np.mean(err)) if err else 99.0)
        seen[name] = res
        assert res[True][0] >= res[False][0], (name, res)  # never loses detections
        assert res[True][1] < 6.0, (name, res)  # yaw stays within a few degrees of the clean yaw
    assert seen["required"][True][0] == len(levels)  # all five dark versions are found with enhancement on


def test_real_eye_open_not_saturated():
    c = _clean_face()
    if c is None:
        return
    lm, img = c
    f = lm.process(img, 0)[0]
    assert all(0.7 < v < 0.999 for v in f.eye_open), f.eye_open  # was exactly 1.0, 1.0
    raw, bl, m = lm.detect(img)[0]
    assert abs(f.eye_open[0] - (1 - bl["eyeBlinkLeft"])) < 1e-9 and abs(f.eye_open[1] - (1 - bl["eyeBlinkRight"])) < 1e-9
    nb = {k: v for k, v in bl.items() if not k.startswith("eyeBlink")}  # landmark-only path on the same real face
    assert all(0.6 < v <= 1.0 for v in face_from_landmarks(raw, nb, m, aspect=img.shape[1] / img.shape[0]).eye_open)
    shut = raw.copy()  # synthetic closed eyes: upper lid collapsed onto the lower lid
    for up, lo in ((159, 145), (386, 374)):
        shut[up] = shut[lo]
    assert max(face_from_landmarks(shut, nb, m, aspect=img.shape[1] / img.shape[0]).eye_open) < 0.05
    closed_bl = {**bl, "eyeBlinkLeft": 0.97, "eyeBlinkRight": 0.96}
    assert max(face_from_landmarks(raw, closed_bl, m).eye_open) < 0.06


def _plane_pose(lm: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Pose from the landmarks alone (independent of the matrix): face axes x = left->right cheek (234 -> 454),
    y = chin -> forehead (152 -> 10), z toward the camera = -landmark z; same angle convention as head_pose."""
    p = np.c_[lm[:, 0] * w, -lm[:, 1] * h, -lm[:, 2] * w]
    x = p[454] - p[234]
    x /= np.linalg.norm(x)
    y = p[10] - p[152]
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    return euler_from_matrix(np.c_[x, y, np.cross(x, y)])


def test_real_pose_convention_matches_geometry():
    c = _clean_face()
    if c is None:
        return
    lm, img = c
    h, w = img.shape[:2]
    raw, _, m = lm.detect(img)[0]
    yaw, pitch, roll = euler_from_matrix(np.array(m))
    # 1. matrix angles vs the independent landmark-geometry pose (a flipped axis would differ by 2 x |angle|)
    py, pp, pr = _plane_pose(raw, w, h)
    # ponytail: crude geometric estimator vs the matrix differ ~7 deg on a real asymmetric face; a flipped axis would differ ~19+
    assert abs(yaw - py) < 10 and abs(pitch - pp) < 8 and abs(roll - pr) < 5, ((yaw, pitch, roll), (py, pp, pr))
    # 2. yaw > 0 <=> nose tip right of the eye-corner midpoint (image coordinates)
    nose_off = raw[1, 0] - (raw[33, 0] + raw[263, 0]) / 2
    assert yaw > 3 and nose_off > 0, (yaw, nose_off)
    # 3. horizontal flip negates yaw and roll and keeps pitch
    fy, fp, fr = euler_from_matrix(np.array(lm.detect(img[:, ::-1].copy())[0][2]))
    assert abs(yaw + fy) < 2.5 and abs(roll + fr) < 3 and abs(pitch - fp) < 5, ((yaw, pitch, roll), (fy, fp, fr))
    # 4. image content rotated clockwise by theta -> roll + theta (cv2 angles are counter-clockwise)
    for theta in (-15, 15):
        rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), -theta, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)
        got = euler_from_matrix(np.array(lm.detect(rot)[0][2]))[2] - roll
        assert abs(got - theta) < 2.5, (theta, got)
