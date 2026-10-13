"""End-of-exam detection summary: time seen, most at once, first/last, top confidence; stalls are capped."""
from proctorlens.core.types import Detections, Perceived
from proctorlens.explain.detections import DetectionTally
from tests.unit.test_perceiver import _face


def _p(t, faces=1, phone=None, hands=0, quality=0.9):
    boxes = [("phone", phone, (0.6, 0.5, 0.8, 0.8))] if phone else []
    return Perceived(t, [_face() for _ in range(faces)], Detections(boxes=boxes), None, 120.0, 200.0, 0.0, quality, [],
                     [], [[(0.5, 0.5)] * 21] * hands)


def test_tally():
    tl = DetectionTally()
    for i in range(50):  # 5 s at 10 fps; a phone from 1.0 s to 2.9 s, two faces at 3 s, a hand at the end
        t = i * 100
        tl.add(t, _p(t, faces=2 if i == 30 else 1, phone=0.6 + 0.01 * i if 10 <= i < 30 else None, hands=int(i >= 45)))
    tl.add(8000, _p(8000))  # 3 s stall: counts as at most 0.5 s
    s = tl.summary()
    d = {x["key"]: x for x in s["detectors"]}
    assert s["total_s"] == 5.4 and d["face"]["seen_s"] == 5.4 and d["face"]["pct"] == 100.0
    assert d["phone"]["seen_s"] == 2.0 and (d["phone"]["first_ms"], d["phone"]["last_ms"]) == (1000, 2900)
    assert d["phone"]["max_conf"] == 0.89 and d["faces_2plus"]["max_count"] == 2 and d["faces_2plus"]["seen_s"] == 0.1
    assert d["hand"]["seen_s"] == 0.5 and d["pose"]["seen_s"] == 0 and d["pose"]["first_ms"] is None
    assert [x["key"] for x in s["detectors"]][:2] == ["face", "faces_2plus"]
