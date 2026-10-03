"""Event metrics on hand-made cases, plus participant bootstrap and the ablation config-override helper."""
import math

from ml.evaluation.ablations import make_cfg
from ml.evaluation.metrics import (bootstrap_ci, fa_counts, false_alarms_per_hour, flagged_time_fraction,
                                   onset_latency, per_type_prf, prf_stat, ratio_stat)
from proctorlens.core.config import Config

A, B = "OFF_SCREEN_SUSTAINED", "PROHIBITED_OBJECT"


def E(t, s, e, **details):
    return dict(type=t, start_ms=int(s * 1000), end_ms=int(e * 1000), details=details)


def test_prf_and_latency():
    gt = [E(A, 10, 20), E(A, 40, 50), E(B, 60, 70)]
    pred = [E(A, 12, 20, t_on_s=4.0), E(A, 100, 110), E(B, 60, 70)]
    r = per_type_prf(pred, gt)
    assert r[A]["tp"] == 1 and r[A]["fp"] == 1 and r[A]["fn"] == 1
    assert r[A]["precision"] == 0.5 and r[A]["recall"] == 0.5 and r[A]["f1"] == 0.5
    assert r[B]["f1"] == 1.0
    lat = onset_latency(pred, gt, A)
    assert lat["n"] == 1 and lat["start_median_ms"] == 2000 and lat["emit_median_ms"] == 6000
    assert math.isnan(onset_latency([], gt)["start_median_ms"])
    assert math.isnan(per_type_prf([], [E(A, 1, 9)])[A]["precision"])  # no predictions: precision undefined, not 0


def test_false_alarms_and_flagged_time():
    honest = [(0, 1_800_000), (1_000_000, 3_600_000)]  # overlapping -> merged: exactly 1 h
    pred = [E(A, 100, 110), E(A, 1500, 1520), E("MONITORING_DEGRADED", 200, 230), E(B, 4000, 4010)]
    assert fa_counts(pred, honest) == (2, 1.0)  # degraded time and out-of-honest events are not counted
    assert false_alarms_per_hour(pred, honest) == 2.0
    assert math.isnan(false_alarms_per_hour(pred, []))
    assert flagged_time_fraction([E(A, 0, 10), E(A, 5, 20)], 100_000) == 0.2  # union, not sum
    assert math.isnan(flagged_time_fraction([], 0))


def test_bootstrap_resamples_participants():
    same = {p: dict(tp=3, fp=1, fn=1) for p in "abcde"}
    est, lo, hi = bootstrap_ci(same, prf_stat("f1"), n=200)
    assert est == lo == hi == 6 / 8  # identical participants => degenerate CI
    mix = {"a": dict(tp=4, fp=0, fn=0), "b": dict(tp=0, fp=4, fn=4), "c": dict(tp=2, fp=1, fn=1)}
    e1 = bootstrap_ci(mix, prf_stat("recall"), n=300, seed=1)
    assert e1 == bootstrap_ci(mix, prf_stat("recall"), n=300, seed=1)  # deterministic
    assert e1[1] < e1[0] < e1[2] and 0 <= e1[1] and e1[2] <= 1
    r = bootstrap_ci({"a": dict(n=4, hours=2.0), "b": dict(n=0, hours=2.0)}, ratio_stat("n", "hours"), n=100)
    assert r[0] == 1.0 and r[1] <= r[0] <= r[2]
    assert math.isnan(bootstrap_ci({"a": dict(tp=1, fp=0, fn=0)}, prf_stat("f1"))[1])  # 1 participant: no CI


def test_make_cfg_overrides_do_not_leak():
    base = Config()
    c = make_cfg(base, {"events": {"off_screen": {"t_on_s": 2.0}}, "quality": {"min_quality": 0.0}})
    assert c.events["off_screen"].t_on_s == 2.0 and c.events["off_screen"].off_thr == 0.35
    assert c.quality.min_quality == 0.0 and base.events["off_screen"].t_on_s == 4.0
    try:
        make_cfg(base, {"events": {"off_screen": {"off_thr": 0.9}}})  # off_thr >= on_thr is rejected
        assert False
    except ValueError:
        pass
