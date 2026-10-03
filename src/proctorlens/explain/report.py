"""Offline review report (master spec 8, 9; docs/DECISIONS.md ADR-015): writes report.html, segments.json and
evidence/ (a clip and three keyframes per event) next to a replay's events.json + features. Opens from disk: no server,
no JavaScript. Every item is an observation for a human reviewer, never a determination about the person."""
from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from proctorlens.core.config import Config
from proctorlens.core.types import MONITORING_DEGRADED, REVIEW_TYPES, Event
from proctorlens.explain.evidence import caption, clip_window, grab_frames, keyframe_times, write_clip
from proctorlens.explain.review import explain, segments, ts
from proctorlens.io import load_events, load_features
from proctorlens.temporal.scores_rule import MOUTH_ENERGY_THR

_E = html.escape
NOTICE = ("Every item below is an observation flagged for human review. ProctorLens reports what the camera showed and "
          "how sure each detector was; it does not decide anything about the person. Review priority only orders what "
          "to look at first. A blind spot means the camera could not observe, not that anything happened.")
LIMITS = ("A webcam cannot see hands, laps or second screens; there is no liveness detection; gaze is coarse (zone level "
          "only); mouth activity is visual only; accuracy can vary with lighting, glasses, skin tone, head coverings and "
          "camera quality; thresholds are untuned starting values until evaluated on held-out data (docs/EVALUATION.md).")


def write_report(out_dir: str | Path, cfg: Config, video: str | Path | None = None,
                 media: dict[int, dict[str, str]] | None = None, reviews: dict[int, dict] | None = None) -> Path:
    """out_dir = a `replay --out` dir (or a web-app session dir). video = what to cut clips/keyframes from (None = no
    media), unless media ({event index: {onset|peak|end|clip: path relative to out_dir}}) already exists (web app).
    reviews = {event index: {decision, note}}: reviewer decisions shown on each event."""
    out = Path(out_dir)
    header, events = load_events(out / "events.json")
    df = load_features(out / "features.parquet")
    summary = json.loads(s.read_text(encoding="utf-8")) if (s := out / "summary.json").is_file() else None
    step = 1000 // cfg.pipeline.grid_hz
    segs = segments(events, cfg)
    (out / "segments.json").write_text(json.dumps(segs, indent=1), encoding="utf-8")
    if media is None:
        media = _evidence(out, Path(video), events, step) if video and events else {}
        if media and not any("clip" in m for m in media.values()):
            print("ffmpeg not on PATH: no clips written, keyframes only")
    dur = max([int(df["t_ms"].max()) + step if len(df) else 0, *(e.end_ms for e in events)])  # timeline: 0 .. dur
    body = [f"<h1>ProctorLens review report: {_E(out.resolve().name)}</h1>", f'<p class="notice">{_E(NOTICE)}</p>',
            _overview(header, summary, events, segs), _timeline(events, segs, dur)]
    for lane, title in (("review", "Review segments (highest priority first)"),
                        ("blind_spot", "Blind spots (monitoring degraded)")):
        cards = [_segment(s, events, media, reviews or {}, df, cfg) for s in segs if s["lane"] == lane]
        body += [f"<h2>{title}</h2>", "".join(cards) or "<p class=mut>None.</p>"]
    prov = _E(json.dumps(header, indent=1))
    body += [f'<h2>Limitations</h2><p class="mut">{_E(LIMITS)}</p>',
             f"<details><summary>Provenance (events.json header)</summary><pre>{prov}</pre></details>"]
    p = out / "report.html"
    p.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                 f'content="width=device-width,initial-scale=1"><title>ProctorLens review report</title>'
                 f'<style>{_CSS}</style></head><body>{"".join(body)}</body></html>', encoding="utf-8")
    return p


def _evidence(out: Path, video: Path, events: list[Event], step: int) -> dict[int, dict[str, str]]:
    """Keyframes (onset, peak, end) + clip per event under out/evidence; returns {event index: {name: relative path}}."""
    (out / "evidence").mkdir(exist_ok=True)
    shots = {i: keyframe_times(e, step) for i, e in enumerate(events)}
    frames = grab_frames(video, [t for s in shots.values() for t in s.values()])
    media: dict[int, dict[str, str]] = {}
    for i, e in enumerate(events):
        m = media[i] = {}
        for name, t in shots[i].items():
            if (f := frames.get(t)) is not None:  # imencode + write_bytes: cv2.imwrite fails silently on non-ASCII paths
                p = m[name] = f"evidence/ev{i}_{name}.jpg"
                (out / p).write_bytes(cv2.imencode(".jpg", caption(f, f"{e.type}  {name}  {ts(t)}"))[1].tobytes())
        if write_clip(video, out / f"evidence/ev{i}.mp4", *clip_window(e.start_ms, e.end_ms)):
            m["clip"] = f"evidence/ev{i}.mp4"
    return media


def _overview(header: dict, summary: dict | None, events: list[Event], segs: list[dict]) -> str:
    c = header.get("calibration") or {}
    rows = [("Scorer", header.get("provider")), ("Config hash", header.get("config_hash")),
            ("Calibration", c.get("mode", "none") + ("" if c.get("error") is None else f" (error {c['error']:.2f})"))]
    if summary:
        pct = lambda x: "n/a" if x is None else f"{x} %"  # noqa: E731
        rows += [("Duration", f"{summary['duration_s']:.1f} s"), ("Face seen", pct(summary["face_seen_pct"])),
                 ("Monitoring degraded", pct(summary["degraded_pct"]))]
    by = {lab: sum(s["lane"] == "review" and s["priority_label"] == lab for s in segs) for lab in ("high", "medium", "low")}
    rows.append(("Review segments", f"{by['high']} high, {by['medium']} medium, {by['low']} low"))
    rows += [(t, f"{n} event(s), {sum((e.end_ms - e.start_ms) for e in events if e.type == t) / 1000:.1f} s")
             for t in REVIEW_TYPES if (n := sum(e.type == t for e in events))]
    return "<table class=ov>" + "".join(f"<tr><th>{_E(k)}</th><td>{_E(str(v))}</td></tr>" for k, v in rows) + "</table>"


def _timeline(events: list[Event], segs: list[dict], dur_ms: int) -> str:
    """Lanes per event type (MONITORING_DEGRADED last = blind spots) under a row of segments; click jumps to the card."""
    left, W, lh = 230, 1000, 18
    H = lh * (len(REVIEW_TYPES) + 1) + 20
    X = lambda t: left + (W - left - 30) * t / max(dur_ms, 1)  # noqa: E731  30: room for the last tick label
    rect = lambda a, b, y, cls, href, tip: (  # noqa: E731
        f'<a href="#{href}"><rect class="{cls}" x="{X(a):.1f}" y="{y + 2}" width="{max(2.0, X(b) - X(a)):.1f}" '
        f'height="{lh - 4}"><title>{_E(tip)}</title></rect></a>')
    g = [f'<text x="{left - 6}" y="{lh - 5}" text-anchor="end">review segments</text>']
    g += [rect(s["start_ms"], s["end_ms"], 0, "blind" if s["lane"] == "blind_spot" else s["priority_label"],
               f"seg{s['id']}", f"segment {s['id']}: {ts(s['start_ms'])} - {ts(s['end_ms'])}, "
                                f"review priority {s['priority_label']}") for s in segs]
    for li, typ in enumerate(REVIEW_TYPES, 1):
        name = "blind spots (monitoring degraded)" if typ == MONITORING_DEGRADED else typ
        g.append(f'<text x="{left - 6}" y="{li * lh + lh - 5}" text-anchor="end">{name}</text>')
        g += [rect(e.start_ms, e.end_ms, li * lh, "blind" if typ == MONITORING_DEGRADED else "ev", f"ev{i}",
                   f"{e.type}: {ts(e.start_ms)} - {ts(e.end_ms)}") for i, e in enumerate(events) if e.type == typ]
    tick = next((s for s in (10, 30, 60, 120, 300, 600, 1800) if dur_ms / 1000 / s <= 10), 3600)
    g += [f'<text x="{X(k * 1000):.1f}" y="{H - 4}" text-anchor="middle">{ts(k * 1000)[:-2]}</text>'
          for k in range(0, int(dur_ms / 1000) + 1, tick)]
    return f'<svg class="tl" viewBox="0 0 {W} {H}" role="img" aria-label="event timeline">{"".join(g)}</svg>'


def _segment(s: dict, events: list[Event], media: dict, reviews: dict, df: pd.DataFrame, cfg: Config) -> str:
    lab = "blind" if s["lane"] == "blind_spot" else s["priority_label"]
    head = (f'<h3><span class="pill {lab}">{_E(s["priority_label"])} priority</span> Segment {s["id"]}: '
            f'{ts(s["start_ms"])} - {ts(s["end_ms"])} <span class=mut>({s["review_priority"]:.2f}; '
            f'{_E(", ".join(s["types"]))})</span></h3>')
    return f'<section class="seg" id="seg{s["id"]}">{head}' + "".join(
        _event(i, events[i], media.get(i, {}), reviews.get(i), df, cfg) for i in s["events"]) + "</section>"


def _event(i: int, e: Event, m: dict, rv: dict | None, df: pd.DataFrame, cfg: Config) -> str:
    vid = (f'<video controls preload="metadata" src="{m["clip"]}"></video>' if "clip" in m else
           '<p class="mut">No clip (run `proctorlens report --video ...` with ffmpeg on PATH).</p>')
    shots = "".join(f'<a href="{p}"><img src="{p}" alt="{n} keyframe" loading="lazy"></a>'
                    for n in ("onset", "peak", "end") if (p := m.get(n)))
    rows = "".join(f"<tr><th>{_E(k)}</th><td>{_E(v if isinstance(v, str) else json.dumps(v))}</td></tr>"
                   for k, v in e.details.items())
    plot = "" if (p := plot_svg(df, e, cfg)) is None else f'{p[0]}<div class="cap">{_E(p[1])}</div>'
    dec = "" if not rv else (f'<p><b>Reviewer decision: {_E(rv["decision"])}</b>'
                             f'{_E(" - " + rv["note"]) if rv.get("note") else ""}</p>')
    return (f'<article id="ev{i}"><h4>{_E(e.type)} <span class=mut>{_E(e.detector)}</span></h4>'
            f"<p>{_E(explain(e))}</p>{dec}<div class=media>{vid}<div class=shots>{shots}</div></div>"
            f"{plot}{_attr(e.attribution)}"
            f"<details><summary>Measured values</summary><table>{rows}</table></details></article>")


def _threshold(key: str | None, cfg: Config, d: dict) -> float | None:
    """Level of the plotted signal at which the rule fires; None where no single level applies (face_absent)."""
    return {"off_screen": d.get("on_thr"), "phone": d.get("on_thr"), "notes": d.get("on_thr"),
            "glancing": cfg.glance.on_thr, "multiple_people": 1.5,  # scores_rule: soft(max(faces, persons), 1.5, ...)
            "speaking": MOUTH_ENERGY_THR, "id_mismatch": cfg.identity.tau,
            "degraded": cfg.quality.min_quality}.get(key or "")


def plot_svg(df: pd.DataFrame, e: Event, cfg: Config) -> tuple[str, str] | None:
    """(inline SVG, caption) of the event's signal over its clip window: event span shaded, threshold dashed, NaN =
    gap. None when the event has no plottable signal (e.g. browser telemetry). Also served by the web app."""
    a, b = clip_window(e.start_ms, e.end_ms)
    d, col = e.details, e.details.get("signal")
    w = df[(df["t_ms"] >= a) & (df["t_ms"] < b)]
    if col == "n_faces_or_persons":
        y = np.fmax(pd.to_numeric(w["n_faces"]).to_numpy(float), pd.to_numeric(w["n_persons"]).to_numpy(float))
    elif col in w:
        y = pd.to_numeric(w[col], errors="coerce").to_numpy(float)
    else:
        return None
    thr = _threshold(d.get("key"), cfg, d)
    vals = np.append(y[np.isfinite(y)], [] if thr is None else [thr])
    if not np.isfinite(y).any():
        return None
    lo, hi = float(vals.min()), float(vals.max())
    hi = hi if hi > lo else lo + 1
    W, H = 360, 90
    X = lambda v: (v - a) / max(b - a, 1) * W  # noqa: E731
    Y = lambda v: H - 4 - (v - lo) / (hi - lo) * (H - 8)  # noqa: E731
    runs, cur = [], []
    for ti, yi in [*zip(w["t_ms"], y, strict=True), (0, np.nan)]:
        if np.isfinite(yi):
            cur.append(f"{X(ti):.1f},{Y(yi):.1f}")
        elif cur:
            runs, cur = [*runs, cur], []
    # presentation attributes = defaults wherever the SVG is shown; the report's CSS overrides them for dark mode
    svg = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="{_E(col)} around the event">',
           (f'<rect class="span" fill="rgba(47,111,159,.15)" x="{X(e.start_ms):.1f}" y="0" '
            f'width="{X(e.end_ms) - X(e.start_ms):.1f}" height="{H}"/>')]
    if thr is not None:
        svg.append(f'<line class="thr" stroke="#b03a2e" stroke-dasharray="4 3" x1="0" x2="{W}" y1="{Y(thr):.1f}" '
                   f'y2="{Y(thr):.1f}"/>')
    svg += [f'<polyline fill="none" stroke="#2f6f9f" stroke-width="1.5" points="{" ".join(r)}"/>' for r in runs]
    cap = f"{col}, {ts(a)} - {ts(b)}, range {lo:g} to {hi:g}; shaded = event" + (
        "" if thr is None else f"; dashed = threshold {thr:g}")
    return "".join(svg) + "</svg>", cap


def _attr(a: dict | None) -> str:
    if not a:
        return ""
    top = max(abs(v) for v in a.values()) or 1.0
    bars = "".join(f'<div class="ab"><span>{_E(k)}</span><i style="width:{60 * abs(v) / top:.0f}%"></i>'
                   f"<b>{v:+.3f}</b></div>" for k, v in sorted(a.items(), key=lambda kv: -abs(kv[1])))
    return f'<div class="attr"><div class="cap">Score drop when a feature group is neutralized</div>{bars}</div>'


_CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#deddd8;--card:#fff;--ev:#2f6f9f;--span:rgba(47,111,159,.14);
--high:#b03a2e;--medium:#b9770e;--low:#7b7d7d;--blind:#5d6d7e}
@media (prefers-color-scheme:dark){:root{--bg:#151515;--fg:#e6e6e3;--mut:#9b9b96;--line:#34342f;--card:#1e1e1c;
--ev:#6aa6d6;--span:rgba(106,166,214,.18);--high:#e0705f;--medium:#e3a43f;--low:#a3a5a5;--blind:#8fa2b5}}
*{box-sizing:border-box}
body{margin:0 auto;max-width:1100px;padding:16px;font:14px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:0 0 8px}h4{margin:12px 0 4px;font-size:14px}
.mut,.cap{color:var(--mut)}.cap{font-size:12px}
.notice{border-left:3px solid var(--ev);padding:6px 10px;background:var(--card)}
table{border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:2px 10px 2px 0;vertical-align:top}
th{font-weight:600;color:var(--mut)}td{overflow-wrap:anywhere}
svg{width:100%;height:auto;display:block}
.tl{margin:12px 0;font-size:11px}.tl text{fill:var(--mut)}
rect.ev{fill:var(--ev)}rect.blind{fill:var(--blind)}rect.high{fill:var(--high)}rect.medium{fill:var(--medium)}
rect.low{fill:var(--low)}a:hover rect{opacity:.7}
.seg{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;margin:10px 0}
.pill{display:inline-block;padding:0 8px;border-radius:10px;color:#fff;font-size:12px}
.pill.high{background:var(--high)}.pill.medium{background:var(--medium)}.pill.low{background:var(--low)}
.pill.blind{background:var(--blind)}
article{border-top:1px solid var(--line);margin-top:8px}
.media{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}video{width:360px;max-width:100%}
.shots{display:flex;gap:6px;flex-wrap:wrap}.shots img{width:170px;max-width:30vw;border:1px solid var(--line)}
.plot{max-width:360px;margin-top:6px;border:1px solid var(--line)}.plot polyline{fill:none;stroke:var(--ev);stroke-width:1.5}
.plot .span{fill:var(--span)}.plot .thr{stroke:var(--high);stroke-dasharray:4 3}
.attr{max-width:420px;margin-top:6px}.ab{display:flex;align-items:center;gap:6px;font-size:12px}
.ab span{width:80px}.ab i{display:block;height:8px;background:var(--ev)}
pre{white-space:pre-wrap;font-size:12px}
"""
