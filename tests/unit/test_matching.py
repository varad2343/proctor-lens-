"""Event matching on hand-made cases with known answers (times in seconds * 1000)."""
from ml.evaluation.matching import match_events
from proctorlens.core.types import Event


def E(t, s, e):
    return dict(type=t, start_ms=int(s * 1000), end_ms=int(e * 1000))


A = "OFF_SCREEN_SUSTAINED"


def test_long_gt_iou_threshold():
    gt = [E(A, 10, 20)]  # 10 s: not padded
    assert match_events([E(A, 12, 18)], gt) == ([(0, 0)], [], [])  # IoU 0.6
    assert match_events([E(A, 17, 25)], gt) == ([], [0], [0])  # IoU 3/15 = 0.2
    assert match_events([E(A, 14, 22)], gt)[0] == [(0, 0)]  # IoU 6/12 = 0.5
    assert match_events([E(A, 10, 13)], gt)[0] == [(0, 0)]  # IoU exactly 0.3 counts


def test_short_gt_is_padded():
    gt = [E(A, 10, 12)]  # 2 s -> padded to [9, 13]
    assert match_events([E(A, 11, 13.5)], gt)[0] == [(0, 0)]  # 2/4.5 = 0.44
    assert match_events([E(A, 12.5, 14)], gt)[0] == []  # 0.5/5 = 0.1
    assert match_events([E(A, 12.5, 14)], gt, pad_ms=0)[0] == []  # no padding: 0 overlap
    assert match_events([E(A, 11.9, 14)], gt, pad_ms=0)[0] == []  # 0.1/4 < 0.3
    assert match_events([E(A, 13.5, 15)], gt)[0] == []  # beyond padded window


def test_type_must_match_and_one_to_one():
    assert match_events([E("FACE_ABSENT", 10, 20)], [E(A, 10, 20)]) == ([], [0], [0])
    # two preds on one GT: only the better one is TP, the other is a FP
    tp, fp, fn = match_events([E(A, 10, 14), E(A, 10, 20)], [E(A, 10, 20)])
    assert (tp, fp, fn) == ([(1, 0)], [0], [])
    # two GTs both above threshold for one pred (IoU .67 vs .5): only the higher-IoU GT is matched
    tp, fp, fn = match_events([E(A, 0, 30)], [E(A, 0, 20), E(A, 15, 30)])
    assert (tp, fp, fn) == ([(0, 0)], [], [1])


def test_greedy_by_iou_crossed():
    gt = [E(A, 0, 10), E(A, 6, 16)]
    pred = [E(A, 5, 15), E(A, -1, 9)]  # pred0 best fits gt1, pred1 best fits gt0
    tp, fp, fn = match_events(pred, gt)
    assert tp == [(0, 1), (1, 0)] and fp == [] and fn == []


def test_accepts_event_objects_and_empty():
    p = [Event(type=A, start_ms=10000, end_ms=20000, confidence=0.9, detector="rule@0.1")]
    assert match_events(p, [E(A, 10, 20)]) == ([(0, 0)], [], [])
    assert match_events([], []) == ([], [], [])
    assert match_events(p, []) == ([], [0], [])
    assert match_events([], [E(A, 1, 9)]) == ([], [], [0])
