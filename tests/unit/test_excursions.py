from proctorlens.core.config import GlanceCfg
from proctorlens.temporal.excursions import ExcursionTracker

STEP = 100


def glances(spec, tail_s=1.0):
    """spec = [(zone, dur_s, gap_s)]: off-screen for dur_s toward zone, then on-screen for gap_s."""
    tr, t, scores = ExcursionTracker(GlanceCfg(), STEP), 0, []
    for zone, dur, gap in spec:
        for _ in range(round(dur * 10)):
            scores.append(tr.update(t, 0.9, zone))
            t += STEP
        for _ in range(round(gap * 10)):
            scores.append(tr.update(t, 0.0, "on_screen"))
            t += STEP
    for _ in range(round(tail_s * 10)):
        scores.append(tr.update(t, 0.0, "on_screen"))
        t += STEP
    return tr, scores


def test_same_zone_glances_fire_within_window():
    tr, sc = glances([("right", 1.0, 5.0)] * 4)
    assert sc.index(1.0) == 3 * 60 + 10  # fires as the 4th excursion ends (first on-screen step)
    assert [x["zone"] for x in tr.excursions()] == ["right"] * 4
    assert tr.excursions()[0] == {"start_ms": 0, "end_ms": 1000, "zone": "right"}
    _, sc = glances([("right", 1.0, 5.0)] * 3 + [("left", 1.0, 5.0)])  # 3/4 = 75% >= 70%
    assert 1.0 in sc


def test_mixed_zones_short_long_and_spread_do_not_fire():
    assert 1.0 not in glances([("right", 1, 5), ("left", 1, 5), ("right", 1, 5), ("up", 1, 5)])[1]
    assert 1.0 not in glances([("right", 4.0, 5.0)] * 6)[1]  # > max_s: sustained, not a glance
    assert 1.0 not in glances([("right", 0.2, 5.0)] * 6)[1]  # < min_s
    assert 1.0 not in glances([("right", 1.0, 19.0)] * 5)[1]  # only 3 inside any 60 s window


def test_window_expiry_and_none():
    tr, sc = glances([("right", 1.0, 5.0)] * 4, tail_s=70)
    assert sc[-1] == 0.0 and tr.excursions() == []
    assert tr.update(10**6, None, "none") is None and tr.update(10**6 + 100, float("nan"), "none") is None
