"""Review layer: segments + priority (hand-computed), explanation templates, evidence frames, offline report."""
import contextlib
import io
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from proctorlens import cli
from proctorlens.core.config import Config, validate
from proctorlens.core.types import MACHINE_EVENT, Event
from proctorlens.explain.evidence import clip_window, grab_frames
from proctorlens.explain.review import explain, segments, ts
from proctorlens.features.schema import COLUMNS
from proctorlens.io import save_events, save_features


def _ev(t, a, b, conf=1.0, **details):
    return Event(t, a, b, conf, "rule@0.1.0", details=details)


def test_segments_merge_lanes_and_priority():
    ev = [_ev("PROHIBITED_OBJECT", 10000, 13000, 0.9), _ev("OFF_SCREEN_SUSTAINED", 17000, 23000, 0.8),  # 4 s apart: merged
          _ev("MOUTH_ACTIVITY", 40000, 44000, 0.5), _ev("MONITORING_DEGRADED", 12000, 20000)]  # own lane, never merged
    s = segments(ev, Config())
    assert [x["lane"] for x in s] == ["review", "review", "blind_spot"] and [x["id"] for x in s] == [0, 1, 2]
    p = (1.0 * 0.9 * (1 + math.log(4)) + 0.5 * 0.8 * (1 + math.log(7))) * 1.25  # 2 distinct types: x (1 + 0.25)
    assert s[0]["events"] == [0, 1] and (s[0]["start_ms"], s[0]["end_ms"]) == (10000, 23000)
    assert s[0]["review_priority"] == round(p, 3) and s[0]["priority_label"] == "high"  # 4.16
    assert s[1]["events"] == [2] and s[1]["priority_label"] == "low"  # 0.4 * 0.5 * (1 + ln 5) = 0.52
    assert s[2]["events"] == [3] and s[2]["types"] == ["MONITORING_DEGRADED"]
    assert segments([], Config()) == []
    json.dumps(s, allow_nan=False)
    cfg = Config()
    cfg.review.weights["NOT_AN_EVENT"] = 1.0
    try:
        validate(cfg)
    except ValueError:
        return
    raise AssertionError("unknown review weight accepted")


def test_explain_is_plain_observational_text_for_every_event_key():
    bad = re.compile(r"\b(cheat\w*|fraud\w*|dishonest\w*|guilty|suspicious)\b", re.I)
    for key, typ in MACHINE_EVENT.items():
        s = explain(_ev(typ, 61000, 65500, 0.9, key=key))  # bare details (old files): missing values read n/a
        assert s.startswith("Between 00:01:01.0 and 00:01:05.5 (4.5 s) ") and not bad.search(s), s
    d = dict(key="off_screen", zone="right", d_yaw_deg=31.0, d_pitch_deg=-2.0, yaw_limit_deg=25.0, pitch_limit_deg=20.0,
             t_on_s=4.0, quality_mean=0.82, calib_mode="full", calib_error=0.08, benign_flags=["glare"])
    s = explain(_ev("OFF_SCREEN_SUSTAINED", 751000, 758100, 0.91, **d))
    assert "to the right of the screen" in s and "+31 deg yaw" in s and "limits 25 / 20 deg" in s, s
    assert "Minimum duration to flag: 4 s" in s and "validation error 0.08" in s and "glare" in s, s
    assert ts(751000) == "00:12:31.0" and ts(3_600_000) == "01:00:00.0"


def _avi(path, n=100):
    """n frames at 10 fps; frame i is flat gray 2*i, so a frame's brightness says which one it is."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 2 * i, np.uint8))
    w.release()


def test_grab_frames_and_clip_window():
    d = Path(tempfile.mkdtemp())
    _avi(d / "v.avi")
    f = grab_frames(d / "v.avi", [0, 2500, 99999])
    assert int(f[0].mean()) <= 3 and abs(f[2500].mean() - 50) <= 3  # first frame at/after 2.5 s = frame 25
    assert abs(f[99999].mean() - 198) <= 3  # past the end: the last frame
    assert clip_window(2000, 4000) == (0, 7000) and clip_window(60000, 120000) == (55000, 85000)  # clamp at 0, 30 s cap


def test_report_writes_html_segments_and_evidence():
    d = Path(tempfile.mkdtemp())
    _avi(d / "v.avi")
    out = d / "out"
    rows = [{**dict.fromkeys(COLUMNS, np.nan), "t_ms": t, "zone": "none", "quality_reasons": "",
             "phone_conf": 0.9 if 3000 <= t < 6000 else 0.0} for t in range(0, 10000, 100)]
    save_features(pd.DataFrame(rows, columns=COLUMNS), out / "features.parquet")
    ev = _ev("PROHIBITED_OBJECT", 3000, 6000, 0.9, key="phone", signal="phone_conf", peak=0.9, peak_ms=4000,
             on_thr=0.5, t_on_s=1.5)
    save_events([ev], out / "events.json", {"provider": "rule", "calibration": None})
    a = cli.build_parser().parse_args(["report", str(out), "--video", str(d / "v.avi")])
    with contextlib.redirect_stdout(io.StringIO()):
        assert a.fn is cli._report and cli._report(a) == 0
    page = (out / "report.html").read_text(encoding="utf-8")
    assert 'id="seg0"' in page and 'href="#ev0"' in page and "a phone was detected" in page
    assert "<polyline" in page and 'class="thr"' in page  # signal plot with its threshold line
    assert json.loads((out / "segments.json").read_text(encoding="utf-8"))[0]["events"] == [0]
    ev_dir = out / "evidence"
    assert sorted(p.name for p in ev_dir.glob("*.jpg")) == ["ev0_end.jpg", "ev0_onset.jpg", "ev0_peak.jpg"]
    assert abs(cv2.imread(str(ev_dir / "ev0_peak.jpg"))[:20].mean() - 80) <= 4  # frame 40 = t 4.0 s, above the caption
    assert (ev_dir / "ev0.mp4").exists() == (shutil.which("ffmpeg") is not None)
    if (ev_dir / "ev0.mp4").exists():  # window 0 .. 9 s at 10 fps
        assert 85 <= cv2.VideoCapture(str(ev_dir / "ev0.mp4")).get(cv2.CAP_PROP_FRAME_COUNT) <= 95
