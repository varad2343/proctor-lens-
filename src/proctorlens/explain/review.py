"""Review layer (master spec 8.5, 9; docs/DECISIONS.md ADR-015): group events into review segments, order them by
review priority, and say in plain words what was measured. Priority only orders what a human looks at first; it is
never a judgement of a person. Text comes from fixed templates over `details` (no language model)."""
from __future__ import annotations

import math

from proctorlens.core.config import Config
from proctorlens.core.types import MONITORING_DEGRADED, Event


def segments(events: list[Event], cfg: Config) -> list[dict]:
    """Events overlapping or within review.merge_gap_s of each other form one segment. MONITORING_DEGRADED goes in its
    own "blind_spot" lane, never merged with observations. Sorted: review lane by priority (desc), then blind spots.
    priority = sum(w_type * confidence * (1 + ln(1 + duration_s))) * (1 + multi_bonus * (distinct types - 1))."""
    r, out = cfg.review, []
    for lane in ("review", "blind_spot"):
        cur = None
        for i in sorted((i for i, e in enumerate(events) if (e.type == MONITORING_DEGRADED) == (lane == "blind_spot")),
                        key=lambda i: events[i].start_ms):
            e = events[i]
            if cur and e.start_ms - cur["end_ms"] <= r.merge_gap_s * 1000:
                cur["end_ms"] = max(cur["end_ms"], e.end_ms)
                cur["events"].append(i)
            else:
                out.append(cur := {"lane": lane, "start_ms": e.start_ms, "end_ms": e.end_ms, "events": [i]})
    for s in out:
        evs = [events[i] for i in s["events"]]
        types = sorted({e.type for e in evs})
        p = sum(r.weights.get(e.type, 0.0) * e.confidence * (1 + math.log1p((e.end_ms - e.start_ms) / 1000))
                for e in evs) * (1 + r.multi_bonus * (len(types) - 1))
        s |= {"types": types, "review_priority": round(p, 3),
              "priority_label": "high" if p >= r.high else "medium" if p >= r.medium else "low"}
    out.sort(key=lambda s: (s["lane"] == "blind_spot", -s["review_priority"], s["start_ms"]))
    for j, s in enumerate(out):
        s["id"] = j
    return out


def ts(ms: float) -> str:
    """hh:mm:ss.s"""
    s = ms / 1000
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:04.1f}"


_ZONE = {"left": "to the left of the screen", "right": "to the right of the screen", "up": "above the screen",
         "down": "below the screen"}
_BROWSER = {"hidden": "exam tab hidden (another tab or window in front)", "blur": "exam window lost focus",
            "fullscreen_exit": "fullscreen left", "offline": "network connection lost", "paste": "text pasted",
            "copy": "text copied", "contextmenu": "context menu opened", "beforeunload": "page close or reload started"}
_FLAG = {"low_quality":"image quality was low for part of the interval",
         "glare": "glare or over-exposure was present",
         "reduced_calibration": "gaze came from head pose only (no accepted gaze calibration)",
         "allow_looking_down": "the policy allows looking down", "allow_notes": "the policy allows notes",
         "allow_reading_aloud": "the policy allows reading aloud"}


def explain(ev: Event) -> str:
    """One paragraph: when, what was measured vs the rule, how reliable the view was, benign context. Missing details
    read "n/a" instead of failing (older events.json files, empty intervals)."""
    d = ev.details

    def v(name: str, fmt: str = "{}") -> str:
        x = d.get(name)
        return "n/a" if x is None else fmt.format(x)

    zone = _ZONE.get(d.get("zone"), "away from the screen")
    what = {  # every value goes through v(), so building all of them is safe
        "face_absent": "no face and no person were visible",
        "multiple_people": f"up to {v('peak', '{:.0f}')} faces or persons were visible at once "
                           f"({v('static_face_flags_max', '{:.0f}')} static background face(s) excluded)",
        "phone": f"a phone was detected (peak detector confidence {v('peak', '{:.2f}')})",
        "notes": f"notes or a book were detected (peak detector confidence {v('peak', '{:.2f}')})",
        "off_screen": f"attention was directed {zone} (largest head turn {v('d_yaw_deg', '{:+.0f}')} deg yaw, "
                      f"{v('d_pitch_deg', '{:+.0f}')} deg pitch from the calibrated baseline; limits "
                      f"{v('yaw_limit_deg', '{:.0f}')} / {v('pitch_limit_deg', '{:.0f}')} deg)",
        "glancing": f"{len(d.get('excursions') or [])} short off-screen looks were observed, mostly {zone} "
                    f"(rule: {v('n_required')} within {v('window_s', '{:.0f}')} s)",
        "speaking": f"mouth movement consistent with speaking was observed (peak motion energy "
                    f"{v('peak', '{:.3f}')}, mean jaw opening {v('jaw_open_mean', '{:.2f}')})",
        "id_mismatch": f"the face differed from the enrolled reference on {v('consecutive')} or more consecutive "
                       f"checks (lowest similarity {v('peak', '{:.2f}')}, threshold {v('tau')})",
        "degraded": f"the camera could not observe reliably ({', '.join(d.get('reasons') or []) or 'reason unknown'}; "
                    f"lowest image quality {v('peak', '{:.2f}')})",
        "browser": f"the browser reported: {_BROWSER.get(d.get('kind'), d.get('kind') or 'n/a')}",
    }.get(d.get("key"), f"{ev.type} was observed")
    s = f"Between {ts(ev.start_ms)} and {ts(ev.end_ms)} ({(ev.end_ms - ev.start_ms) / 1000:.1f} s) {what}."
    if d.get("t_on_s"):
        s += f" Minimum duration to flag: {d['t_on_s']:g} s."
    err = "" if d.get("calib_error") is None else f", validation error {d['calib_error']:.2f}"
    s += (f" Detector confidence {ev.confidence:.2f}; mean image quality {v('quality_mean', '{:.2f}')}; "
          f"calibration {d.get('calib_mode', 'n/a')}{err}.")
    if flags := [_FLAG.get(f, f) for f in d.get("benign_flags") or []]:
        s += " Context: " + "; ".join(flags) + "."
    return s
