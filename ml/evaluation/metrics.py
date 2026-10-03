"""Event-level metrics (spec 12.2): precision/recall/F1, onset latency, false alarms per honest hour,
flagged-time burden, participant-level bootstrap CIs. Events are Event objects or dict records."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ml.data.labels import merge_intervals
from ml.evaluation.matching import _get, iv, match_events
from proctorlens.core.types import MONITORING_DEGRADED

nan = math.nan


def counts(pred, gt, **kw) -> dict[str, dict[str, int]]:
    """{type: {tp, fp, fn}} from one match_events() call."""
    tp, fp, fn = match_events(pred, gt, **kw)
    out: dict[str, dict[str, int]] = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
    for pi, _ in tp:
        out[iv(pred[pi])[0]]["tp"] += 1
    for pi in fp:
        out[iv(pred[pi])[0]]["fp"] += 1
    for gi in fn:
        out[iv(gt[gi])[0]]["fn"] += 1
    return dict(out)


def prf(c: dict[str, int]) -> dict[str, float]:
    """precision/recall/F1 from counts; undefined (empty denominator) = NaN, never silently 0."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return dict(precision=tp / (tp + fp) if tp + fp else nan, recall=tp / (tp + fn) if tp + fn else nan,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan)


def per_type_prf(pred, gt, **kw) -> dict[str, dict]:
    return {t: {**c, **prf(c)} for t, c in counts(pred, gt, **kw).items()}


def onset_deltas(pred, gt, **kw) -> list[tuple[str, float, float]]:
    """Per TP: (type, pred.start - gt.start, emission - gt.start) in ms.
    # ponytail: emission time = back-dated start + details.t_on_s (Event carries no emission stamp)."""
    tp, _, _ = match_events(pred, gt, **kw)
    out = []
    for pi, gi in tp:
        d = iv(pred[pi])[1] - iv(gt[gi])[1]
        out.append((iv(pred[pi])[0], float(d), d + 1000.0 * float((_get(pred[pi], "details") or {}).get("t_on_s", 0.0))))
    return out


def latency_summary(deltas) -> dict[str, float]:
    s = np.array([d[1] for d in deltas], float)
    e = np.array([d[2] for d in deltas], float)
    q = (lambda a, p: float(np.percentile(a, p)) if len(a) else nan)
    return dict(n=len(s), start_median_ms=q(s, 50), start_q1_ms=q(s, 25), start_q3_ms=q(s, 75), emit_median_ms=q(e, 50))


def onset_latency(pred, gt, etype: str | None = None, **kw) -> dict[str, float]:
    return latency_summary([d for d in onset_deltas(pred, gt, **kw) if etype in (None, d[0])])


def fa_counts(pred, honest, types=None) -> tuple[int, float]:
    """(#events overlapping honest time, honest hours). Default types exclude MONITORING_DEGRADED
    (declaring blindness is not a false alarm). # ponytail: assumes honest intervals hold no GT events."""
    H = merge_intervals(honest)
    n = 0
    for t, s, e in map(iv, pred):
        if (t != MONITORING_DEGRADED if types is None else t in types) and any(s < he and e > hs for hs, he in H):
            n += 1
    return n, sum(e - s for s, e in H) / 3.6e6


def false_alarms_per_hour(pred, honest_intervals, types=None) -> float:
    n, hours = fa_counts(pred, honest_intervals, types)
    return n / hours if hours else nan


def flagged_time_fraction(pred, total_ms: float, types=None) -> float:
    """Fraction of the recording covered by the union of predicted events (capped at 1: an event closed at end of
    stream extends one grid step past the last feature row)."""
    U = merge_intervals((s, e) for t, s, e in map(iv, pred) if types is None or t in types)
    return min(1.0, sum(e - s for s, e in U) / total_ms) if total_ms > 0 else nan


def bootstrap_ci(per_participant_stats, stat_fn, n: int = 1000, seed: int = 0):
    """(estimate, lo, hi): resample PARTICIPANTS (not frames) with replacement; 95% percentile CI.
    per_participant_stats: dict or list of per-participant items; stat_fn(list of items) -> float.
    CI is NaN with < 2 participants (a bootstrap over one unit says nothing)."""
    items = list(per_participant_stats.values() if isinstance(per_participant_stats, dict) else per_participant_stats)
    est = stat_fn(items)
    if len(items) < 2:
        return est, nan, nan
    rng = np.random.default_rng(seed)
    v = np.array([stat_fn([items[j] for j in rng.integers(0, len(items), len(items))]) for _ in range(n)], float)
    v = v[~np.isnan(v)]
    return (est, *np.percentile(v, [2.5, 97.5])) if len(v) else (est, nan, nan)


def prf_stat(key: str):
    """stat_fn over per-participant {tp,fp,fn} items: key in precision|recall|f1."""
    return lambda items: prf({k: sum(i[k] for i in items) for k in ("tp", "fp", "fn")})[key]


def ratio_stat(num: str, den: str, scale: float = 1.0):
    """stat_fn over per-participant items: scale * sum(num) / sum(den) (FA/h, flagged fraction)."""
    return lambda items: scale * sum(i[num] for i in items) / d if (d := sum(i[den] for i in items)) else nan
