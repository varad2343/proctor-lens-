import numpy as np

from proctorlens.perception.objects import IdTracker, _decode, _letterbox, detections_from_boxes

P = (0.1, 0.1, 0.5, 0.9)


def test_detections_from_boxes():
    items = [("cell phone", 0.6, (0.6, 0.6, 0.7, 0.7)), ("phone", 0.4, (0.1, 0.1, 0.2, 0.2)),
             ("book", 0.35, (0.3, 0.3, 0.5, 0.5)), ("notes", 0.2, (0.3, 0.3, 0.5, 0.5)),  # notes < thr
             ("person", 0.9, P), ("person", 0.7, (0.12, 0.1, 0.52, 0.9)),  # overlaps P -> suppressed
             ("person", 0.5, (0.6, 0.1, 0.95, 0.9)),
             ("dog", 0.99, (0, 0, 1, 1))]  # unknown class ignored
    d = detections_from_boxes(items, 0.3)
    assert d.phone_conf == 0.6 and d.notes_conf == 0.35 and d.fresh
    assert d.person_boxes == [P, (0.6, 0.1, 0.95, 0.9)]
    assert sorted(n for n, _, _ in d.boxes) == ["notes", "person", "person", "phone", "phone"]
    e = detections_from_boxes([], 0.3)
    assert (e.phone_conf, e.notes_conf, e.person_boxes, e.boxes) == (0.0, 0.0, [], [])
    # NMS is per class: duplicate phones collapse to the most confident, a phone inside a person box stays
    dup = detections_from_boxes([("cell phone", 0.5, (0.2, 0.2, 0.4, 0.4)), ("cell phone", 0.8, (0.21, 0.2, 0.41, 0.4)),
                                 ("person", 0.9, (0.0, 0.0, 1.0, 1.0))], 0.3)
    assert [(n, c) for n, c, _ in dup.boxes] == [("person", 0.9), ("phone", 0.8)] and dup.phone_conf == 0.8


def test_id_tracker():
    A, B = (0.1, 0.1, 0.4, 0.9), (0.6, 0.5, 0.8, 0.8)
    shift = lambda b, d: (b[0] + d, b[1], b[2] + d, b[3])  # noqa: E731

    def run():
        tr = IdTracker()
        return [tr([("person", A)], 0), tr([("phone", B), ("person", shift(A, 0.01))], 100),
                tr([("notes", B)], 200),  # same box, other class: never inherits the phone's number
                tr([("person", shift(A, 0.02))], 1000),  # 900 ms after it was last seen: keeps its number
                tr([("person", A)], 2100), tr([], 2200), tr([("face", A), ("face", B)], 2300)]  # > 1 s gap: new number

    got = run()
    assert got == [[1], [2, 1], [3], [1], [4], [], [5, 6]], got
    assert run() == got  # deterministic: same inputs, same numbers


def test_decode_yolo_outputs():
    names = {0: "person", 1: "cell phone"}
    frame = np.zeros((480, 640, 3), np.uint8)  # 640x480 -> letterbox scale 1.0, top pad 80
    x, lb = _letterbox(frame, 640)
    assert x.shape == (1, 3, 640, 640) and lb == (1.0, 0, 80)
    # standard layout (1, 4+nc, N): cx, cy, w, h in letterbox pixels; phone at original px (100,100)-(200,200)
    out = np.zeros((1, 6, 3), np.float32)
    out[0, :, 0] = (150, 230, 100, 100, 0.1, 0.8)  # class 1, conf 0.8
    out[0, :, 1] = (320, 320, 50, 50, 0.2, 0.25)  # below conf
    out[0, :, 2] = (400, 300, 80, 200, 0.9, 0.1)  # person
    got = _decode(out, names, lb, 640, 480, 0.3)
    assert [(n, round(c, 2)) for n, c, _ in got] == [("cell phone", 0.8), ("person", 0.9)]
    assert np.allclose(got[0][2], (100 / 640, 100 / 480, 200 / 640, 200 / 480))
    # NMS-free end2end layout (1, N, 6): x1y1x2y2, conf, cls
    e2e = np.array([[[100, 180, 200, 280, 0.8, 1], [0, 0, 1, 1, 0.1, 0]]], np.float32)
    got2 = _decode(e2e, names, lb, 640, 480, 0.3)
    assert len(got2) == 1 and got2[0][0] == "cell phone" and np.allclose(got2[0][2], got[0][2])
