import random

from proctorlens.core.config import EventCfg
from proctorlens.temporal.state_machine import ACTIVE, COOLDOWN, IDLE, PENDING, EventMachine

STEP = 100
BASE = dict(on_thr=0.5, off_thr=0.3, t_on_s=1.0, t_off_s=0.5, merge_gap_s=1.0, min_dur_s=0.0,
            cooldown_s=0.0, max_hold_s=5.0)


def mk(**kw):
    return EventMachine("k", EventCfg(**{**BASE, **kw}), "FACE_ABSENT", STEP, "rule@test")


def run(m, seq, t0=0):
    ev = []
    for i, s in enumerate(seq):
        ev += m.update(t0 + i * STEP, s)
    return ev


def test_onset_backdated_offset_and_details():
    m = mk()
    ev = []
    for i, s in enumerate([0.0] * 5 + [0.9] * 20):
        ev += m.update(i * STEP, s)
        assert m.state == (IDLE if i < 5 else PENDING if i < 14 else ACTIVE)
    o = m.ongoing()
    assert (o.status, o.start_ms, o.end_ms) == ("ongoing", 500, 2500) and not ev
    ev = run(m, [0.0] * 20, t0=2500)
    assert len(ev) == 1 and m.ongoing() is None
    e = ev[0]
    assert (e.type, e.start_ms, e.end_ms, e.status) == ("FACE_ABSENT", 500, 2500, "final")
    assert e.detector == "rule@test"
    assert abs(e.confidence - 0.9) < 1e-9
    assert e.details["n_steps"] == 20 and e.details["score_peak"] == 0.9 and e.details["key"] == "k"


def test_debounce_blip_ignored():
    assert run(mk(), [0.9] * 9 + [0.0] * 40) == []


def test_hysteresis_no_split():
    m = mk()
    # 0.4 is between off_thr and on_thr; a 3-step dip (< t_off = 5 steps) must not split either
    ev = run(m, [0.9] * 20 + [0.4] * 30 + [0.0] * 3 + [0.9] * 10 + [0.0] * 40)
    assert len(ev) == 1 and ev[0].start_ms == 0 and ev[0].end_ms == 6300


def test_merge_gap():
    seq = [0.9] * 20 + [0.0] * 12 + [0.9] * 20 + [0.0] * 40  # 1.2 s between the two runs
    two = run(mk(merge_gap_s=1.0), seq)
    assert [(e.start_ms, e.end_ms) for e in two] == [(0, 2000), (3200, 5200)]
    one = run(mk(merge_gap_s=2.0), seq)
    assert [(e.start_ms, e.end_ms) for e in one] == [(0, 5200)] and one[0].details["n_steps"] == 40


def test_min_duration_and_cooldown():
    m = mk(min_dur_s=3.0)
    ev = run(m, [0.9] * 20 + [0.0] * 30 + [0.9] * 40 + [0.0] * 30)  # 2 s dropped, 4 s kept
    assert [(e.start_ms, e.end_ms) for e in ev] == [(5000, 9000)]
    m = mk(merge_gap_s=0.0, cooldown_s=5.0)
    # the 2nd run (t=3000..4900) falls inside the 5 s cooldown
    seq = [0.9] * 20 + [0.0] * 10 + [0.9] * 20 + [0.0] * 30 + [0.9] * 20 + [0.0] * 30
    m_states = []
    ev = []
    for i, s in enumerate(seq):
        ev += m.update(i * STEP, s)
        m_states.append(m.state)
    assert [e.start_ms for e in ev] == [0, 8000] and COOLDOWN in m_states


def test_none_gating_and_max_hold():
    m = mk()
    assert run(m, [None] * 80) == [] and m.state == IDLE
    ev = run(mk(), [0.9] * 5 + [None] * 10 + [0.9] * 5 + [0.0] * 30)  # debounce timer paused over None
    assert len(ev) == 1 and ev[0].start_ms == 0
    ev = run(mk(), [0.9] * 20 + [None] * 30 + [0.9] * 10 + [0.0] * 30)  # ACTIVE held across short None run
    assert len(ev) == 1 and ev[0].end_ms == 6000
    m = mk()
    ev = run(m, [0.9] * 20 + [None] * 60)  # None longer than max_hold_s closes; None time is no evidence
    assert len(ev) == 1 and ev[0].end_ms == 2000 and m.ongoing() is None


def test_close_finalizes_open_and_held():
    m = mk()
    run(m, [0.9] * 20)
    ev = m.close(2000)
    assert len(ev) == 1 and ev[0].status == "final" and (ev[0].start_ms, ev[0].end_ms) == (0, 2000)
    assert m.ongoing() is None and m.state == IDLE and m.close(3000) == []
    m = mk()
    assert run(m, [0.9] * 20 + [0.0] * 6) == []  # ended, still inside merge_gap
    assert len(m.close(2600)) == 1
    m = mk(min_dur_s=3.0)
    run(m, [0.9] * 20)
    assert m.close(2000) == []  # shorter than min_dur_s


def test_randomized_invariants():
    for seed in range(300):
        r = random.Random(seed)
        kw = dict(t_on_s=r.choice([0, 0.3, 1.0]), t_off_s=r.choice([0, 0.3, 1.0]),
                  merge_gap_s=r.choice([0, 0.5, 2.0]), min_dur_s=r.choice([0, 0.5, 2.0]),
                  cooldown_s=r.choice([0, 1.0, 3.0]), max_hold_s=r.choice([0.5, 2.0]))
        seq = []
        for _ in range(r.randint(5, 40)):
            v = r.choice([0.9, 0.4, 0.0, None, "rand"])
            seq += [r.random() if v == "rand" else v] * r.randint(1, 30)

        def go():
            m = mk(**kw)
            ev = run(m, seq)
            return ev + m.close(len(seq) * STEP)

        ev = go()
        assert [e.to_dict() for e in ev] == [e.to_dict() for e in go()]  # deterministic
        gap = max(kw["merge_gap_s"], kw["cooldown_s"]) * 1000
        for i, e in enumerate(ev):
            assert e.end_ms >= e.start_ms and e.end_ms - e.start_ms >= kw["min_dur_s"] * 1000
            assert e.end_ms - e.start_ms >= max(kw["t_on_s"] * 1000, STEP)
            assert 0.0 <= e.confidence <= 1.0 and e.details["n_steps"] >= 1
            if i:
                assert e.start_ms - ev[i - 1].end_ms >= gap  # no overlap, merged if closer than merge_gap
