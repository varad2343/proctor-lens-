"""Structured event details (SPEC 8): the signal that fired, measured values vs thresholds, quality context.
Observations only: numbers, codes and flags, no free text and no verdicts."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import MACHINE_EVENT, Event

# score key -> (row column, reducer giving the "peak" = worst measured value of that column)
_SIGNAL = {
    "face_absent": ("time_since_face_ms", np.max), "multiple_people": ("n_faces", np.max),
    "phone": ("phone_conf", np.max), "notes": ("notes_conf", np.max),
    "off_screen": ("off_screen_score", np.max), "glancing": ("off_screen_score", np.max),
    "speaking": ("mouth_energy_1s", np.max), "id_mismatch": ("id_similarity", np.min),
    "degraded": ("quality", np.min),
}
_POLICY_FLAG = {"off_screen": "allow_looking_down", "glancing": "allow_looking_down",
                "notes": "allow_notes", "speaking": "allow_reading_aloud"}
_NO_ZONE = (None, "", "none", "on_screen")


def _col(rows: list[dict], name: str) -> np.ndarray:
    return np.array([r.get(name) for r in rows], dtype=float)  # None/NaN -> nan


def _red(fn, a: np.ndarray, nd: int = 3) -> float | None:
    """Reduce over finite values; None when there are none (JSON-safe)."""
    a = a[np.isfinite(a)]
    return round(float(fn(a)), nd) if a.size else None


def _modal_zone(rows: list[dict]) -> str | None:
    z = Counter(r.get("zone") for r in rows if r.get("zone") not in _NO_ZONE)
    return z.most_common(1)[0][0] if z else None


def _excursions(rows: list[dict], start_ms: int, cfg: Config) -> list[dict]:
    """Off-screen runs (off_screen_score >= glance.on_thr) lasting [min_s, max_s] that ended inside the window_s
    before the event fired (start_ms) or later: the glances the event is made of."""
    g, out, run = cfg.glance, [], []
    step = int(np.median(np.diff([r["t_ms"] for r in rows]))) if len(rows) > 1 else 100
    for r in [*rows, None]:  # None = sentinel closing the last run
        if r is not None and (r.get("off_screen_score") or 0) >= g.on_thr:  # NaN >= x is False
            run.append(r)
            continue
        if run:
            start, end = run[0]["t_ms"], run[-1]["t_ms"] + step
            if g.min_s <= (end - start) / 1000 <= g.max_s and end > start_ms - g.window_s * 1000:
                out.append({"start_ms": int(start), "end_ms": int(end), "zone": _modal_zone(run)})
            run = []
    return out


def _key(event: Event, rows: list[dict]) -> str:
    if k := event.details.get("key"):
        return k
    k = next(k for k, t in MACHINE_EVENT.items() if t == event.type)
    if k == "phone" and np.nansum(_col(rows, "notes_conf")) > np.nansum(_col(rows, "phone_conf")):
        k = "notes"
    return k


def build_details(event: Event, rows: list[dict], cfg: Config, calib: Any) -> dict:
    """rows = grid rows inside [start, end] (for glancing the caller widens by cfg.glance.window_s).
    calib = Calibration | None. Returns event.details plus the computed fields."""
    key, n = _key(event, rows), len(rows)
    col, worst = _SIGNAL[key]
    sig = _col(rows, col)
    n_persons = _col(rows, "n_persons")
    if key == "multiple_people":
        sig = np.fmax(sig, n_persons)
    quality = _col(rows, "quality")
    reasons = Counter(t for r in rows for t in re.split(r"[,;|\s]+", str(r.get("quality_reasons") or "")) if t)
    fin = np.flatnonzero(np.isfinite(sig))
    d: dict[str, Any] = {
        **event.details, "key": key, "signal": "n_faces_or_persons" if key == "multiple_people" else col,
        "peak": _red(worst, sig), "mean": _red(np.mean, sig),
        "peak_ms": int(rows[fin[(np.argmax if worst is np.max else np.argmin)(sig[fin])]]["t_ms"]) if fin.size else None,
        "duration_s": round((event.end_ms - event.start_ms) / 1000, 2),
        "calib_mode": calib.mode if calib is not None else "none",
        "calib_error": _red(np.max, np.array([calib.error], dtype=float), 4) if calib is not None else None,
        "quality_mean": _red(np.mean, quality),
    }
    if ec := cfg.events.get(key):
        d |= {"on_thr": ec.on_thr, "off_thr": ec.off_thr, "t_on_s": ec.t_on_s, "min_dur_s": ec.min_dur_s}
    flags = []
    if n and sum(1 for r in rows if not r.get("reliable", True)) >= 0.2 * n:
        flags.append("low_quality")
    if n and reasons["bright"] >= 0.2 * n:
        flags.append("glare")
    if calib is None or calib.mode != "full":
        flags.append("reduced_calibration")
    if getattr(cfg.policy, _POLICY_FLAG.get(key, ""), False):
        flags.append(_POLICY_FLAG[key])
    d["benign_flags"] = flags
    if key in ("off_screen", "glancing"):
        d["zone"] = _modal_zone(rows)
        for name, lim in (("yaw", cfg.gaze.yaw_limit_deg), ("pitch", cfg.gaze.pitch_limit_deg)):
            a = _col(rows, f"d_{name}")
            a = a[np.isfinite(a)]
            d[f"d_{name}_deg"] = round(float(a[np.argmax(np.abs(a))]), 1) if a.size else None  # signed, largest
            d[f"{name}_limit_deg"] = lim
    if key == "glancing":
        d |= {"excursions": _excursions(rows, event.start_ms, cfg), "n_required": cfg.glance.n,
              "window_s": cfg.glance.window_s}
    elif key in ("phone", "notes"):
        d["object"] = key
    elif key == "multiple_people":
        d |= {"n_faces_max": _red(np.max, _col(rows, "n_faces"), 0), "n_persons_max": _red(np.max, n_persons, 0),
              "static_face_flags_max": _red(np.max, _col(rows, "static_face_flags"), 0)}
    elif key == "degraded":
        d["reasons"] = [r for r, _ in reasons.most_common()]
    elif key == "speaking":
        d["jaw_open_mean"] = _red(np.mean, _col(rows, "jaw_open"))
    elif key == "id_mismatch":
        d |= {"tau": cfg.identity.tau, "consecutive": cfg.identity.consecutive}
    return d
