import json
import math
from types import SimpleNamespace

from proctorlens.core.config import Config
from proctorlens.core.types import Event
from proctorlens.explain.attribution import attribute_event
from proctorlens.explain.details import build_details


def _rows(n, t0=0, **over):
    base = dict(frame_valid=True, reliable=True, quality=0.8, quality_reasons="", zone="on_screen", off_screen_score=0.0,
                d_yaw=0.0, d_pitch=0.0, n_faces=1, n_persons=0, static_face_flags=0, phone_conf=0.0, notes_conf=0.0,
                mouth_energy_1s=0.0, jaw_open=0.0, time_since_face_ms=0.0, id_similarity=math.nan)
    return [{**base, "t_ms": t0 + 100 * i, **{k: (v(i) if callable(v) else v) for k, v in over.items()}} for i in range(n)]


def _ev(type_, key, n, start=0):
    return Event(type_, start, start + 100 * n, 0.9, "rule@0.1.0",
                 details={"key": key, "score_peak": 0.94, "score_mean": 0.9, "n_steps": n})


def test_details_per_type():
    cfg, calib = Config(), SimpleNamespace(mode="full", error=0.08)
    # off-screen right: modal zone, signed largest d_yaw vs limit, thresholds, calibration, quality
    rows = _rows(50, zone="right", off_screen_score=0.9, d_yaw=lambda i: 20 + i * 0.3, d_pitch=-4.0)
    d = build_details(_ev("OFF_SCREEN_SUSTAINED", "off_screen", 50), rows, cfg, calib)
    assert (d["signal"], d["zone"], d["peak"], d["calib_mode"], d["calib_error"]) == ("off_screen_score", "right", 0.9, "full", 0.08)
    assert d["d_yaw_deg"] == 34.7 and d["yaw_limit_deg"] == cfg.gaze.yaw_limit_deg and d["d_pitch_deg"] == -4.0
    assert d["peak_ms"] == 0  # first step at the peak value (off_screen_score is flat here)
    assert (d["t_on_s"], d["on_thr"], d["duration_s"], d["quality_mean"], d["benign_flags"]) == (4.0, 0.5, 5.0, 0.8, [])
    assert d["score_peak"] == 0.94  # machine fields survive
    json.dumps(d, allow_nan=False)
    d = build_details(_ev("MONITORING_DEGRADED", "degraded", 30), _rows(30, quality=lambda i: 0.2 if i == 7 else 0.8), cfg, calib)
    assert d["peak_ms"] == 700  # degraded's peak is the worst = lowest quality
    # uncalibrated + low quality + policy toggle => benign flags; no calibration => calib_mode none
    cfg.policy.allow_looking_down = True
    low = _rows(20, zone="down", off_screen_score=0.9, reliable=lambda i: i < 10, quality_reasons="bright")
    d = build_details(_ev("OFF_SCREEN_SUSTAINED", "off_screen", 20), low, cfg, None)
    assert d["calib_mode"] == "none" and d["calib_error"] is None
    assert d["benign_flags"] == ["low_quality", "glare", "reduced_calibration", "allow_looking_down"]
    # phone vs notes inferred from rows when the machine key is absent
    ev = Event("PROHIBITED_OBJECT", 0, 3000, 0.9, "rule@0.1.0")
    d = build_details(ev, _rows(30, notes_conf=0.8), cfg, calib)
    assert (d["key"], d["object"], d["peak"]) == ("notes", "notes", 0.8)
    # degraded: modal reasons; multiple people: max of faces/persons
    d = build_details(_ev("MONITORING_DEGRADED", "degraded", 30), _rows(30, quality=0.1, reliable=False, quality_reasons="dark,no_face"), cfg, calib)
    assert d["reasons"] == ["dark", "no_face"] and d["peak"] == 0.1
    d = build_details(_ev("MULTIPLE_PEOPLE", "multiple_people", 30), _rows(30, n_faces=1, n_persons=2), cfg, calib)
    assert d["peak"] == 2.0 and d["n_persons_max"] == 2.0
    # glancing: 5 short right-hand excursions (5 steps = 0.5 s each) are listed, a 5 s one is not
    off = lambda i: 0.9 if i % 20 < 5 else 0.0  # noqa: E731
    rows = _rows(100, zone=lambda i: "right" if i % 20 < 5 else "on_screen", off_screen_score=off)
    d = build_details(_ev("REPEATED_GLANCING", "glancing", 10, start=9000), rows, cfg, calib)
    assert len(d["excursions"]) == 5 and all(e["zone"] == "right" and e["end_ms"] - e["start_ms"] == 500 for e in d["excursions"])
    long = _rows(100, zone="right", off_screen_score=0.9)
    assert build_details(_ev("REPEATED_GLANCING", "glancing", 10), long, cfg, calib)["excursions"] == []
    # empty rows do not crash
    assert build_details(_ev("FACE_ABSENT", "face_absent", 5), [], cfg, None)["peak"] is None


def test_attribution_rule_provider_is_none():
    assert attribute_event(object(), "off_screen") is None
    assert attribute_event(SimpleNamespace(attribute=lambda t: {"eye": 0.3}), "off_screen") == {"eye": 0.3}
    assert attribute_event(SimpleNamespace(attribute=lambda t: {}), "off_screen") is None
