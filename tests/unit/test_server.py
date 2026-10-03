"""Web app (needs the `web` extra): auth, one session over /ws/stream (check -> calibration -> exam -> end) with fake
models, live events on /ws/proctor, the REST views, review, report, deletion, policies and privacy."""
import shutil
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tests.integration.test_pipeline_fake import FakePerceiver, _face

_JPEG = cv2.imencode(".jpg", np.full((48, 64, 3), 128, np.uint8))[1].tobytes()
SCRIPT = [(5, "normal"), (8, "phone"), (99, "normal")]  # exam time: a phone from 5 s to 8 s


class FakeLM:
    def process(self, frame, t_ms):
        return [_face()]


class ExamPerceiver(FakePerceiver):
    """FakePerceiver on exam time (the session clock also counts the setup steps)."""
    t0 = None

    def perceive(self, frame, t_ms, idx):
        self.t0 = t_ms if self.t0 is None else self.t0
        return super().perceive(frame, t_ms - self.t0, idx)


def _client(d, **kw):
    try:
        from fastapi.testclient import TestClient

        from proctorlens.server.app import create_app
    except ImportError:
        return None
    return TestClient(create_app((), d, password="pw", landmarker_factory=lambda cfg: FakeLM(),
                                 perceiver_factory=lambda cfg, lm: ExamPerceiver(SCRIPT), frontend=d / "none", **kw))


class Candidate:
    """Client side of /ws/stream: frames on a 10 fps client clock, each answered by an ack (sent last)."""

    def __init__(self, ws):
        self.ws, self.seq, self.t, self.seen = ws, 0, 1.7e12, []

    def frame(self, n=1):
        for _ in range(n):
            self.seq, self.t = self.seq + 1, self.t + 100
            self.ws.send_bytes(struct.pack("<Id", self.seq, self.t) + _JPEG)
            while (m := self.ws.receive_json())["type"] != "ack":
                self.seen.append(m)
            assert m["seq"] == self.seq

    def ask(self, msg):
        self.ws.send_json(msg)
        return self.ws.receive_json()


def _login(c):
    assert c.post("/api/auth/login", json={"username": "proctor", "password": "pw"}).status_code == 200


def test_session_lifecycle():
    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    from starlette.websockets import WebSocketDisconnect

    assert c.get("/api/sessions").status_code == 401 and c.get("/api/health").json()["ok"]
    assert c.post("/api/auth/login", json={"username": "proctor", "password": "nope"}).status_code == 401
    _login(c)
    s = c.post("/api/sessions", json={"candidate_label": "P01"}).json()
    sid, tok = s["session_id"], s["candidate_token"]
    assert s["join_url"] == f"/#/c/{sid}/{tok}"
    try:
        with c.websocket_connect(f"/ws/stream/{sid}?token=wrong") as ws:
            ws.receive_json()
        raise AssertionError("a wrong candidate token was accepted")
    except WebSocketDisconnect as e:
        assert e.code == 4401

    with c.websocket_connect("/ws/proctor") as pws, c.websocket_connect(f"/ws/stream/{sid}?token={tok}") as ws:
        cand = Candidate(ws)
        assert cand.ask({"type": "hello"})["phase"] == "check"
        cand.frame(6)
        st = [m for m in cand.seen if m["type"] == "status"]
        assert st and st[0]["phase"] == "check" and st[0]["quality"] is not None and "guidance" in st[0]
        dots = cand.ask({"type": "calib_start", "mode": "quick"})["dots"]
        for dot in dots:
            ws.send_json({"type": "calib_point", "dot_id": dot["dot_id"], "t": cand.t})
            cand.frame(dot["dur_ms"] // 100)
        r = cand.ask({"type": "calib_end"})
        assert r["type"] == "calib_result" and r["tries"] == 1 and r["n_samples"] >= 5 * len(dots)
        ws.send_json({"type": "bogus"})  # unknown types are ignored: no reply, nothing breaks
        assert cand.ask({"type": "exam_start"})["type"] == "exam_started"
        cand.frame(90)  # 9 s
        ws.send_json({"type": "browser_event", "kind": "hidden", "t": cand.t})
        cand.frame(10)
        ws.send_json({"type": "browser_event", "kind": "visible", "t": cand.t})
        cand.frame(20)
        assert c.get(f"/api/sessions/{sid}/frame.jpg").content == _JPEG  # live view while the exam runs
        ws.send_json({"type": "calib_start"})  # setup after the exam started: ignored (no calib_schedule reply)
        assert cand.ask({"type": "exam_end", "answers": {"q1": "b"}})["type"] == "ended"
        seen = []
        while not ((m := pws.receive_json())["type"] == "session_status" and m.get("phase") == "ended"):
            seen.append(m)
    kinds = {(m["type"], m.get("event", {}).get("type")) for m in seen}
    assert {("event_started", "PROHIBITED_OBJECT"), ("event_ended", "PROHIBITED_OBJECT"),
            ("event_ended", "BROWSER_INTEGRITY")} <= kinds
    assert any(m["type"] == "session_status" and m.get("phase") == "exam" for m in seen)

    evs = {e["type"]: e for e in c.get(f"/api/sessions/{sid}/events").json()}
    ph, br = evs["PROHIBITED_OBJECT"], evs["BROWSER_INTEGRITY"]
    assert 2500 <= ph["end_ms"] - ph["start_ms"] <= 3500 and "a phone was detected" in ph["explanation"]
    assert br["details"]["kind"] == "hidden" and br["end_ms"] - br["start_ms"] == 1000
    assert set(ph["thumbs"]) == {"onset", "peak", "end"}
    assert c.get(f"/api/sessions/{sid}/files/{ph['thumbs']['peak']}").headers["content-type"] == "image/jpeg"
    assert (ph["clip_path"] is not None) == (shutil.which("ffmpeg") is not None)
    if ph["clip_path"]:
        rng = c.get(f"/api/sessions/{sid}/files/{ph['clip_path']}", headers={"Range": "bytes=0-99"})
        assert rng.status_code == 206 and len(rng.content) == 100  # clips seek
    assert c.get(f"/api/sessions/{sid}/files/..%2F..%2Fproctorlens.db").status_code == 404  # no escaping the folder
    (seg,) = [s for s in c.get(f"/api/sessions/{sid}/segments").json() if s["lane"] == "review"]
    assert sorted(seg["events"]) == sorted([ph["id"], br["id"]])  # 1 s apart: one segment, by event id
    tl = c.get(f"/api/sessions/{sid}/timeline").json()
    assert 20 <= len(tl["t_ms"]) <= 26 and len(tl["series"]["phone_conf"]) == len(tl["t_ms"])  # 12 s at 2 Hz
    assert c.get(f"/api/events/{ph['id']}/plot").json()["svg"].startswith("<svg")
    assert c.post(f"/api/events/{ph['id']}/review", json={"decision": "maybe"}).status_code == 422
    assert c.post(f"/api/events/{ph['id']}/review", json={"decision": "confirm", "note": "held up"}).json()["decision"] == "confirm"
    row = c.get(f"/api/sessions/{sid}").json()
    assert row["status"] == "ended" and row["n_reviewed"] == 1 and row["counts"]["PROHIBITED_OBJECT"] == 1
    assert row["calibration"]["tries"] == 1 and sum(row["priority"].values()) == 1
    page = c.get(f"/api/sessions/{sid}/report").text  # redirect followed to report.html
    assert "a phone was detected" in page and "Reviewer decision: confirm" in page and "held up" in page
    assert c.get(f"/api/sessions/{sid}/frame.jpg").status_code == 404  # not live any more
    try:
        with c.websocket_connect(f"/ws/stream/{sid}?token={tok}") as ws:
            ws.receive_json()
        raise AssertionError("an ended session accepted a candidate")
    except WebSocketDisconnect as e:
        assert e.code == 4401
    assert c.delete(f"/api/sessions/{sid}").status_code == 200
    assert not (d / "sessions" / sid).exists() and c.get(f"/api/sessions/{sid}").status_code == 404


def test_no_events_no_media_and_retention_and_policies():
    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    _login(c)
    bad = c.post("/api/policies", json={"id": "loose", "yaml": "policy: {allow_everything: true}"})
    assert bad.status_code == 400 and "allow_everything" in bad.json()["detail"]
    assert c.post("/api/policies", json={"id": "../x", "yaml": "{}"}).status_code == 422
    assert c.post("/api/policies", json={"id": "notes-ok", "yaml": "policy: {allow_notes: true}"}).status_code == 200
    assert {p["id"] for p in c.get("/api/policies").json()} == {"default", "notes-ok"}
    s = c.post("/api/sessions", json={"candidate_label": "P02", "policy_id": "notes-ok"}).json()
    sid = s["session_id"]
    assert c.get(f"/api/sessions/{sid}").json()["policy"]["allow_notes"] is True  # snapshot taken at creation
    assert c.post(f"/api/sessions/{sid}/labels", json=[{"type": "FACE_ABSENT", "start_ms": 5, "end_ms": 1}]).status_code == 400
    assert c.post(f"/api/sessions/{sid}/labels", json=[{"type": "FACE_ABSENT", "start_ms": 1, "end_ms": 5}]).json() == {"added": 1}
    assert c.get(f"/api/sessions/{sid}/labels").json()[0]["source"] == "annotator"
    with c.websocket_connect(f"/ws/stream/{sid}?token={s['candidate_token']}") as ws:
        cand = Candidate(ws)
        cand.frame(3)
        cand.ask({"type": "exam_start"})
        cand.frame(30)  # 3 s, nothing happens
        cand.ask({"type": "exam_end"})
    assert sorted(p.name for p in (d / "sessions" / sid).iterdir()) == ["events.json", "features.parquet", "summary.json"]
    _client(d, retention_days=-1)  # restart with a retention window that has passed: ended sessions are removed
    assert c.get(f"/api/sessions/{sid}").status_code == 404 and not (d / "sessions" / sid).exists()
