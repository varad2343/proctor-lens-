import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Detections, Face
from proctorlens.perception import Perceiver
from proctorlens.perception.identity import Enrollment, cosine

FRAME = np.random.default_rng(0).integers(60, 200, (48, 64, 3)).astype(np.uint8)


def _face(yaw=0.0):
    return Face(bbox=(0.3, 0.2, 0.7, 0.8), yaw=yaw, pitch=0.0, roll=0.0, iris=(.5, .5, .5, .5), eye_open=(1., 1.),
                blend={}, center=(.5, .5), scale=0.6)


class FakeLandmarker:
    def __init__(self, absent=(), yaw=0.0):
        self.absent, self.yaw = absent, yaw  # absent = (t0, t1) ms with no face

    def process(self, frame, t_ms):
        return [] if self.absent and self.absent[0] <= t_ms < self.absent[1] else [_face(self.yaw)]


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def process(self, frame):
        self.calls += 1
        return Detections(phone_conf=self.calls / 10)


class FakeIdentity:
    def __init__(self, emb=(1.0, 0.0)):
        self.emb, self.calls = emb, 0

    def embed(self, frame, face):
        self.calls += 1
        return None if self.emb is None else np.array(self.emb, np.float32)


def _cfg(every=2.0):
    cfg = Config()
    cfg.pipeline.identity_every_s = every
    cfg.models.identity = "no/such/dir"  # never build a real identity component
    return cfg


def _run(p, n=150):  # 10 fps
    return [p.perceive(FRAME, i * 100, i) for i in range(n)]


def test_detector_carry_over_and_quality():
    det = FakeDetector()
    out = _run(Perceiver(_cfg(), FakeLandmarker(), det, None), 9)
    assert [o.det.fresh for o in out] == [True, False, False] * 3 and det.calls == 3
    assert [o.det.phone_conf for o in out] == [0.1] * 3 + [0.2] * 3 + [0.3] * 3
    assert all(o.faces and o.quality >= 0.5 and o.quality_reasons == [] and o.t_ms == i * 100
               for i, o in enumerate(out))


def test_enrollment_then_periodic_checks():
    idc = FakeIdentity()
    p = Perceiver(_cfg(2.0), FakeLandmarker(), FakeDetector(), idc)
    out = _run(p)
    assert p.enrollment.ready and idc.calls <= 5 + 5  # <=5 enrollment embeddings + 5 checks
    assert all(o.id is None for o in out[:50])  # still enrolling
    times = [o.t_ms for o in out if o.id is not None]
    assert times == [5000, 7000, 9000, 11000, 13000]
    assert all(o.id.similarity > 0.999 and o.id.quality_ok for o in out if o.id)


def test_check_after_face_returns_and_not_frontal():
    p = Perceiver(_cfg(100.0), FakeLandmarker(absent=(6000, 8000)), FakeDetector(), FakeIdentity())
    assert [o.t_ms for o in _run(p, 120) if o.id is not None] == [5000, 8000]
    # non-frontal frames are never used for enrollment -> never ready -> id stays None
    p = Perceiver(_cfg(), FakeLandmarker(yaw=40.0), FakeDetector(), FakeIdentity())
    assert not any(o.id for o in _run(p, 120)) and not p.enrollment.ready


def test_no_identity_component_or_no_embedding():
    assert not any(o.id for o in _run(Perceiver(_cfg(), FakeLandmarker(), FakeDetector(), None)))
    assert not any(o.id for o in _run(Perceiver(_cfg(), FakeLandmarker(), FakeDetector(), FakeIdentity(None))))


def test_enrollment_and_cosine():
    e = Enrollment()
    for v in ((1, 0), (1, 0.1), (1, -0.1)):
        assert not e.ready
        e.add(np.array(v, np.float32))
    assert e.ready and abs(np.linalg.norm(e.mean_emb()) - 1) < 1e-6
    assert e.similarity(np.array([1.0, 0.0])) > 0.99 and e.similarity(np.array([0.0, 1.0])) < 0.1
    assert abs(cosine(np.array([2.0, 0]), np.array([5.0, 0])) - 1) < 1e-9
