"""Causal window buffer + window statistics (GBM inputs / TCN inputs + mask)."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

_STATS = ("mean", "std", "min", "max", "slope", "last", "frac_above_med")
# (column, stat) appended after the per-column stats when the column is in cols
_EXTRA = (("mouth_energy_1s", "zcr"), ("off_screen_score", "cross05"))


class WindowBuffer:
    """Last `window` feature rows. array() is front zero-padded with NaN/inf replaced by 0 (always use
    mask(): True = frame_valid and every column finite; padded steps are False)."""

    def __init__(self, window: int, columns: list[str]):
        self.window, self.columns = window, list(columns)
        self._rows: deque = deque(maxlen=window)  # (values, valid)

    def push(self, row: dict) -> None:
        v = [math.nan if row.get(c) is None else float(row[c]) for c in self.columns]
        self._rows.append((v, bool(row.get("frame_valid", True)) and all(map(math.isfinite, v))))

    def __len__(self) -> int:
        return len(self._rows)

    def array(self) -> np.ndarray:
        out = np.zeros((self.window, len(self.columns)), np.float32)
        if self._rows:
            out[self.window - len(self._rows):] = np.nan_to_num(
                np.array([r[0] for r in self._rows]), nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def mask(self) -> np.ndarray:
        m = np.zeros(self.window, bool)
        m[self.window - len(self._rows):] = [r[1] for r in self._rows]
        return m


def _col_stats(v: np.ndarray, t: np.ndarray) -> list[float]:
    """v = valid values, t = their step indices. No valid step -> all NaN."""
    if len(v) == 0:
        return [math.nan] * len(_STATS)
    x = t - t.mean()
    slope = float(x @ (v - v.mean()) / (x @ x)) if len(v) > 1 else 0.0  # per step
    return [float(v.mean()), float(v.std()), float(v.min()), float(v.max()), slope, float(v[-1]),
            float((v > np.median(v)).mean())]


def _extra(v: np.ndarray, kind: str) -> float:
    if len(v) == 0:
        return math.nan
    if kind == "zcr":  # sign changes of the mean-removed series / steps (rhythmic mouth motion)
        c = v - v.mean()
        c = np.where(np.abs(c) < 1e-9, 0.0, c)
        return float((c[:-1] * c[1:] < 0).mean()) if len(v) > 1 else 0.0
    return float(((v[:-1] >= 0.5) != (v[1:] >= 0.5)).sum())  # cross05: count of 0.5-crossings


def window_stats(W: np.ndarray, mask: np.ndarray, cols: list[str]) -> np.ndarray:
    """Per column over valid & finite steps: mean,std,min,max,slope,last,frac above the window median;
    then zcr(mouth_energy_1s) and cross05(off_screen_score) when those columns are present."""
    W, mask = np.asarray(W, float), np.asarray(mask, bool)
    out, series = [], {}
    for j, c in enumerate(cols):
        ok = mask & np.isfinite(W[:, j])
        series[c] = W[ok, j]
        out += _col_stats(series[c], np.flatnonzero(ok))
    out += [_extra(series[c], k) for c, k in _EXTRA if c in series]
    return np.array(out, float)


def stat_names(cols: list[str]) -> list[str]:
    return [f"{c}__{s}" for c in cols for s in _STATS] + [f"{c}__{k}" for c, k in _EXTRA if c in cols]
