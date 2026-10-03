"""GridResampler: jitter, gaps, duplicates, out-of-order, staleness, never-future. No fixtures."""
import numpy as np

from proctorlens.core.clock import GridResampler


def _feed(r: GridResampler, frames) -> list:
    """Push frames whose item is their own timestamp, so each step's provenance is visible."""
    out: list = []
    for t in frames:
        out += r.push(t, t)
    return out


def test_regular_and_jittered():
    assert _feed(GridResampler(), [0, 100, 200]) == [(0, 0, 0.0), (100, 100, 0.0), (200, 200, 0.0)]
    # first frame at 3 ms => grid starts at 100; a step takes the latest frame at/before it, never a later one
    assert _feed(GridResampler(), [3, 98, 205]) == [(100, 98, 2.0), (200, 98, 102.0)]


def test_gap_is_explicit_and_stale_steps_are_none():
    out = _feed(GridResampler(), [50, 1000])
    assert [T for T, _, _ in out] == list(range(100, 1100, 100))  # one step per missed step, none skipped
    item = {T: it for T, it, _ in out}
    assert item[100] == item[200] == item[300] == 50  # ages 50/150/250 <= max_age_ms (inclusive)
    assert all(item[T] is None for T in range(400, 1000, 100))  # stale => frame_valid=False, not interpolated
    assert item[1000] == 1000
    assert [a for _, _, a in out][:4] == [50.0, 150.0, 250.0, 350.0]  # age still reported when stale


def test_custom_rate_and_age():
    r = GridResampler(hz=5, max_age_ms=100)
    assert r.step == 200
    assert _feed(r, [0, 500]) == [(0, 0, 0.0), (200, None, 200.0), (400, None, 400.0)]


def test_duplicates_and_out_of_order_dropped():
    r = GridResampler()
    assert r.push(0, "a") == [(0, "a", 0.0)]
    assert r.push(0, "dup") == []  # same timestamp
    assert r.push(100, "b") == [(100, "b", 0.0)]
    assert r.push(50, "late") == []  # older than the last accepted frame
    assert r.push(100, "dup2") == []
    assert r.push(150, "c") == []  # accepted (no grid step due yet)
    assert r.push(120, "late2") == []  # now older than 150
    assert r.push(200, "d") == [(200, "d", 0.0)]  # rejected frames left no trace


def test_random_stream_invariants():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = GridResampler(hz=10, max_age_ms=250)
        ts = 1000 + np.cumsum(rng.integers(-30, 160, size=200))  # jitter + gaps + duplicates + reordering
        out: list = []
        for t in ts:
            out += r.push(int(t), int(t))
        Ts = [T for T, _, _ in out]
        assert all(b - a == 100 for a, b in zip(Ts, Ts[1:]))  # strictly increasing, no skipped steps
        for T, it, age in out:
            if it is None:
                assert age > 250
            else:
                assert it <= T and T - it == age <= 250  # never the future, age is exact
