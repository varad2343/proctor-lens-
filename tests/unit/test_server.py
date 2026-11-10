"""Web app (needs the `web` extra): auth, one session over /ws/stream (check -> calibration -> exam -> end) with fake
models, live events on /ws/proctor, the REST views, review, report, deletion, policies and privacy."""
import math
import shutil
import sys
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


def test_default_login_is_proctor_proctor():
    import json
    import os

    try:
        from fastapi.testclient import TestClient

        from proctorlens.server.app import create_app
    except ImportError:
        return
    d = Path(tempfile.mkdtemp())
    (d / "auth.json").write_text(json.dumps({"secret": "ab" * 32, "password": "old-plain", "pw_hash": "s$h"}))
    env = os.environ.pop("PROCTORLENS_PASSWORD", None)
    try:
        c = TestClient(create_app((), d, frontend=d / "none"))
    finally:
        if env is not None:
            os.environ["PROCTORLENS_PASSWORD"] = env
    assert json.loads((d / "auth.json").read_text()) == {"secret": "ab" * 32}  # older stored passwords are dropped
    ok = {"username": "proctor", "password": "proctor"}
    assert c.post("/api/auth/login", json=ok).status_code == 200
    for bad in ("old-plain", "wrong", ""):
        assert c.post("/api/auth/login", json={**ok, "password": bad}).status_code == 401
    c2 = TestClient(create_app((), d, password="chosen", frontend=d / "none"))  # a chosen password replaces it
    assert c2.post("/api/auth/login", json=ok).status_code == 401
    assert c2.post("/api/auth/login", json={**ok, "password": "chosen"}).status_code == 200
    shutil.rmtree(d, ignore_errors=True)


def test_join_link_uses_public_address():
    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    _login(c)
    assert c.post("/api/sessions", json={"candidate_label": "P1"}).json()["join_url"].startswith("/#/c/")  # local
    c.app.state.public_origin = "https://abc.trycloudflare.com"  # what `serve --tunnel` sets
    s = c.post("/api/sessions", json={"candidate_label": "P2"}).json()
    assert s["join_url"] == f"https://abc.trycloudflare.com/#/c/{s['session_id']}/{s['candidate_token']}"
    assert all(r["join_url"].startswith("https://") for r in c.get("/api/sessions").json())
    shutil.rmtree(d, ignore_errors=True)


def test_login_link_and_persistent_cookie():
    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    link, key = c.app.state.login_link, c.app.state.link_key
    assert link == f"/#/link/{key}"  # in the fragment: browsers never send it, so it never reaches access logs
    assert c.post("/api/auth/link", json={"key": key[:-1] + "x"}).status_code == 401
    r = c.post("/api/auth/link", json={"key": key})
    assert r.status_code == 200 and "max-age" in r.headers["set-cookie"].lower()
    tok = c.cookies.get("pl_token")
    assert c.get("/api/auth/me").status_code == 200
    c2 = _client(d)  # server restart, same data dir: the cookie still works, the old link does not
    c2.cookies.set("pl_token", tok)
    assert c2.get("/api/auth/me").status_code == 200
    assert c2.post("/api/auth/link", json={"key": key}).status_code == 401
    key2 = c2.app.state.link_key
    c2.cookies.set("pl_token", tok[:-1] + ("0" if tok[-1] != "0" else "1"))
    assert c2.get("/api/auth/me").status_code == 401  # tampered signature
    c2.cookies.set("pl_token", tok)
    c2.post("/api/auth/logout")
    c2.cookies.set("pl_token", tok)
    assert c2.get("/api/auth/me").status_code == 401  # logged out
    assert c2.post("/api/auth/link", json={"key": key2}).status_code == 401  # logout retires the link too
    assert c2.app.state.link_key != key2
    c2.cookies.clear()
    for bad in ("².x.y", "9" * 5000 + ".x.y", "", "a.b"):  # crafted cookies: 401, never a 500
        assert c2.get("/api/auth/me", headers={"cookie": f"pl_token={bad}".encode("latin-1")}).status_code == 401
    c3 = _client(d)  # two logins in the same second: logging one out leaves the other in
    t1, t2 = (c3.post("/api/auth/link", json={"key": c3.app.state.link_key}).cookies["pl_token"] for _ in "12")
    assert t1 != t2
    c3.cookies.set("pl_token", t1)
    c3.post("/api/auth/logout")
    c3.cookies.set("pl_token", t2)
    assert c3.get("/api/auth/me").status_code == 200
    shutil.rmtree(d, ignore_errors=True)


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
        ov = c.get(f"/api/sessions/{sid}/frame.jpg?overlay=1")  # the detectors' view, drawn server-side
        assert ov.headers["content-type"] == "image/jpeg" and ov.content != _JPEG
        assert cv2.imdecode(np.frombuffer(ov.content, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
        ws.send_json({"type": "calib_start"})  # setup after the exam started: ignored (no calib_schedule reply)
        assert cand.ask({"type": "exam_end", "answers": {"q1": "b"}})["type"] == "ended"
        seen = []
        while not ((m := pws.receive_json())["type"] == "session_status" and m.get("phase") == "ended"):
            seen.append(m)
    kinds = {(m["type"], m.get("event", {}).get("type")) for m in seen}
    assert {("event_started", "PROHIBITED_OBJECT"), ("event_ended", "PROHIBITED_OBJECT"),
            ("event_ended", "BROWSER_INTEGRITY")} <= kinds
    assert any(m["type"] == "session_status" and m.get("phase") == "exam" for m in seen)
    assert any(m["type"] == "session_status" and "proc_ms" in m and m.get("n_faces") == 1 for m in seen)
    # the detector readouts are proctor-only: the candidate's status messages carry setup guidance, nothing more
    assert all(set(m) <= {"type", "phase", "quality", "reasons", "guidance"} for m in cand.seen if m["type"] == "status")

    evs = {e["type"]: e for e in c.get(f"/api/sessions/{sid}/events").json()}
    ph, br = evs["PROHIBITED_OBJECT"], evs["BROWSER_INTEGRITY"]
    assert 2500 <= ph["end_ms"] - ph["start_ms"] <= 3500 and "a phone was detected" in ph["explanation"]
    assert br["details"]["kind"] == "hidden" and br["end_ms"] - br["start_ms"] == 1000
    assert set(ph["thumbs"]) == {"onset", "peak", "end"}
    peak = c.get(f"/api/sessions/{sid}/files/{ph['thumbs']['peak']}")
    assert peak.headers["content-type"] == "image/jpeg"
    assert set(ph["thumbs_overlay"]) == {"onset", "peak", "end"}  # detector-overlay twins of the keyframes
    peak_ov = c.get(f"/api/sessions/{sid}/files/{ph['thumbs_overlay']['peak']}")
    assert peak_ov.headers["content-type"] == "image/jpeg" and peak_ov.content != peak.content
    assert (ph["clip_path"] is not None) == (shutil.which("ffmpeg") is not None or sys.platform == "win32")
    assert ph["clip_overlay"] == (ph["clip_path"] and ph["clip_path"][:-4] + "_ov.mp4")  # tracking twin of the clip
    if ph["clip_path"]:
        rng = c.get(f"/api/sessions/{sid}/files/{ph['clip_path']}", headers={"Range": "bytes=0-99"})
        assert rng.status_code == 206 and len(rng.content) == 100  # clips seek
    assert c.get(f"/api/sessions/{sid}/files/..%2F..%2Fproctorlens.db").status_code == 404  # no escaping the folder
    (seg,) = [s for s in c.get(f"/api/sessions/{sid}/segments").json() if s["lane"] == "review"]
    assert sorted(seg["events"]) == sorted([ph["id"], br["id"]])  # 1 s apart: one segment, by event id
    tl = c.get(f"/api/sessions/{sid}/timeline").json()
    assert 20 <= len(tl["t_ms"]) <= 26 and len(tl["series"]["quality"]) == len(tl["t_ms"])  # 12 s at 2 Hz
    assert c.get(f"/api/events/{ph['id']}/plot").json()["svg"].startswith("<svg")
    assert c.post(f"/api/events/{ph['id']}/review", json={"decision": "maybe"}).status_code == 422
    assert c.post(f"/api/events/{ph['id']}/review", json={"decision": "confirm", "note": "held up"}).json()["decision"] == "confirm"
    row = c.get(f"/api/sessions/{sid}").json()
    assert row["status"] == "ended" and row["n_reviewed"] == 1 and row["counts"]["PROHIBITED_OBJECT"] == 1
    assert row["calibration"]["tries"] == 1 and sum(row["priority"].values()) == 1
    assert row["ongoing"] == [] and row["review_note"] == ""  # nothing in progress once ended; no note yet
    assert c.put(f"/api/sessions/{sid}/note", json={"note": "camera <b>was</b> dim"}).status_code == 200
    assert c.get(f"/api/sessions/{sid}").json()["review_note"] == "camera <b>was</b> dim"
    page = c.get(f"/api/sessions/{sid}/report").text  # redirect followed to report.html
    assert "a phone was detected" in page and "Reviewer decision: confirm" in page and "held up" in page
    assert "camera &lt;b&gt;was&lt;/b&gt; dim" in page  # the session note, escaped
    csv_rows = c.get(f"/api/sessions/{sid}/decisions.csv").text.lstrip("﻿").splitlines()
    assert csv_rows[0].startswith("roll,name,event_id,type,start,end") and any(",PROHIBITED_OBJECT," in r and ",confirm,held up," in r for r in csv_rows)
    assert c.get(f"/api/sessions/{sid}/frame.jpg").status_code == 204  # not live any more: no frame, no error
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


def test_face_mismatch_keeps_reference_picture():
    """The reference picture leaves memory only with a face-mismatch flag (the consent page says so)."""
    from proctorlens.core.config import Config
    from proctorlens.core.types import FACE_ABSENT, IDENTITY_MISMATCH, Event
    from proctorlens.server.db import DB
    from proctorlens.server.live import LiveSession

    d = Path(tempfile.mkdtemp())
    s = LiveSession("s1", Config(), d, DB(d / "t.db"), lambda cfg: FakeLM(), lambda cfg, lm: None)
    s.ref_jpeg, frames = b"REF", [(t, _JPEG, None, None) for t in (0, 100, 200)]
    s._evidence(1, Event(IDENTITY_MISMATCH, 0, 200, 0.9, "rule"), frames)
    s._evidence(2, Event(FACE_ABSENT, 0, 200, 0.9, "rule"), frames)
    assert (d / "evidence/1_reference.jpg").read_bytes() == b"REF" and not (d / "evidence/2_reference.jpg").exists()
    s.worker.shutdown()
    shutil.rmtree(d, ignore_errors=True)


def test_class_link_gives_each_student_a_session():
    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    _login(c)
    k = c.post("/api/classes", json={"name": "Physics 10B"}).json()
    cid, tok = k["id"], k["join_url"].split("/")[-1]
    assert k["join_url"] == f"/#/j/{cid}/{tok}" and k["open"] and k["n_students"] == 0
    c.cookies.clear()  # students are not logged in
    assert c.get(f"/api/join/{cid}", params={"token": tok}).json() == {"name": "Physics 10B", "open": True}
    assert c.get(f"/api/join/{cid}", params={"token": "wrong"}).status_code == 404
    b = c.post(f"/api/join/{cid}", json={"token": tok, "roll": "10", "name": "Ben"}).json()  # joins first, sorts second
    a = c.post(f"/api/join/{cid}", json={"token": tok, "roll": " 2 ", "name": " Asha  K "}).json()
    assert a["session_id"] != b["session_id"] and a["candidate_token"] != b["candidate_token"]
    assert c.post(f"/api/join/{cid}", json={"token": tok, "roll": "3", "name": "   "}).status_code == 422
    assert c.post(f"/api/join/{cid}", json={"token": tok, "roll": "2", "name": "Someone"}).status_code == 409  # roll taken
    _login(c)
    roster = c.get("/api/sessions", params={"class_id": cid}).json()
    assert [(s["roll"], s["candidate_label"]) for s in roster] == [("2", "Asha K"), ("10", "Ben")]  # by roll, 2 < 10
    assert all(s["exam_id"] == cid for s in roster)
    csv_rows = c.get(f"/api/sessions/{roster[0]['id']}/decisions.csv")
    assert csv_rows.text.lstrip("﻿").startswith("roll,name,event_id") and "proctorlens_2_Asha_K_" in csv_rows.headers["content-disposition"]
    assert c.get(f"/api/classes/{cid}").json()["n_students"] == 2
    assert c.post(f"/api/classes/{cid}/open", json={"open": False}).json()["open"] is False
    c.cookies.clear()
    assert c.post(f"/api/join/{cid}", json={"token": tok, "roll": "99", "name": "Late"}).status_code == 409  # closed
    shutil.rmtree(d, ignore_errors=True)


def test_class_custom_exam_and_bulk_export():
    """A class can set its own questions (the candidate gets them over `hello`, instead of the built-in bank), and a
    teacher can export every student's flags and evidence for the whole class in one request."""
    import io
    import zipfile

    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    _login(c)
    exam = {"minutes": 5, "questions": [{"id": "q1", "text": "2+2?", "options": ["3", "4"]}]}
    k = c.post("/api/classes", json={"name": "Physics", "exam": exam}).json()
    assert k["exam"] == exam
    assert c.post("/api/classes", json={"name": "No exam"}).json()["exam"] is None  # the default bank
    cid, tok = k["id"], k["join_url"].split("/")[-1]
    c.cookies.clear()
    s = c.post(f"/api/join/{cid}", json={"token": tok, "roll": "1", "name": "Ann"}).json()
    sid = s["session_id"]
    with c.websocket_connect(f"/ws/stream/{sid}?token={s['candidate_token']}") as ws:
        cand = Candidate(ws)
        assert cand.ask({"type": "hello"})["exam"] == exam  # the class's own questions reach the candidate
        cand.frame(6)
        assert cand.ask({"type": "exam_start"})["type"] == "exam_started"  # no calibration: falls back to mode "none"
        cand.frame(90)  # a phone, per SCRIPT -> one PROHIBITED_OBJECT with keyframes
        assert cand.ask({"type": "exam_end", "answers": {}})["type"] == "ended"
    _login(c)
    assert c.get(f"/api/sessions/{sid}/report").status_code == 200  # writes report.html into the session's files
    rows = c.get(f"/api/classes/{cid}/decisions.csv").text.lstrip("﻿").splitlines()
    assert rows[0].startswith("roll,name,event_id") and any(r.startswith("1,Ann,") and ",PROHIBITED_OBJECT," in r for r in rows)
    z = c.get(f"/api/classes/{cid}/evidence.zip")
    assert z.headers["content-type"] == "application/zip" and f"proctorlens_Physics_{cid}" in z.headers["content-disposition"]
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert any(n.endswith("/report.html") for n in names) and any(n.endswith(".jpg") for n in names)
    assert not any(n.endswith((".mp4", ".webm", "_ov.jpg")) for n in names)  # no clips, no overlay twins
    shutil.rmtree(d, ignore_errors=True)


def test_browser_mode_session_through_the_api():
    """Browser mode (ADR-019): a BrowserSession (models stubbed: nobody in view) runs the exam outside the server;
    its drained results go through the candidate API; the proctor side sees the same views as in server mode."""
    import base64
    import json

    try:
        from fastapi.testclient import TestClient

        from proctorlens.browser import BrowserSession
        from proctorlens.core.config import Config
        from proctorlens.server.app import create_app
    except ImportError:
        return

    class NoFace:
        def landmarks(self, rgba, w, h, t_ms, image_mode):
            return []

    d = Path(tempfile.mkdtemp())
    c = TestClient(create_app((), d, password="pw", frontend=d / "none", browser_mode=True))
    assert c.get("/api/health").json()["mode"] == "browser"
    _login(c)
    s = c.post("/api/sessions", json={"candidate_label": "P1"}).json()
    sid, tok = s["session_id"], s["candidate_token"]
    assert c.get(f"/api/c/{sid}", params={"token": "wrong"}).status_code == 401
    info = c.get(f"/api/c/{sid}", params={"token": tok}).json()
    cfg = __import__("proctorlens.core.config", fromlist=["from_dict"]).from_dict(Config, info["config"])

    b = BrowserSession(sid, cfg, NoFace(), {"names": {0: "person"}, "imgsz": [480, 640]}, identity=False)
    b.dir = d / "browser" / sid

    def push(status=None):
        out = json.loads(b.drain())
        assert c.post(f"/api/c/{sid}/sync", json={"token": tok, "ops": out["ops"], "feed": out["feed"],
                                                   "status": status}).status_code == 200
        for f in out["files"]:
            assert c.put(f"/api/c/{sid}/files/{f}", params={"token": tok}, content=(b.dir / f).read_bytes()).status_code == 200
        for clip in out["clips"]:  # the page encodes WebM; any bytes stand in for it here
            c.put(f"/api/c/{sid}/files/{clip['path']}", params={"token": tok}, content=b"\x1aE\xdf\xa3" + bytes(200))
            b.clip_done(clip["id"])
        return out

    since = c.get("/api/feed").json()["last"]  # a fresh database: position 0
    b.control_in(json.dumps({"type": "exam_start"}))
    room = cv2.GaussianBlur((np.random.default_rng(0).random((480, 640, 3)) * 120 + 60).astype(np.uint8), (5, 5), 0)
    jpeg = cv2.imencode(".jpg", room)[1].tobytes()  # lit, textured, nobody (tools/fake_cam.py's empty room)
    for i in range(60):  # 6 s, nobody in view -> FACE_ABSENT from 3 s
        t = 1.7e12 + i * 100
        b.prepare(jpeg)
        di = b.detection_input(t)
        b.frame_in(i, t, jpeg, *((np.zeros((1, 84, 6300), np.float32).tobytes(), [1, 84, 6300]) if di else ()))
        b.run_slices(20)
        if i % 20 == 19:
            push({"phase": b.phase, "ongoing": [{"type": e.type, "start_ms": e.start_ms} for e in b.pl.ongoing()]})
    ov, raw = b.snapshot()
    pic = {"overlay": base64.b64encode(bytes(ov)).decode(), "raw": base64.b64encode(bytes(raw)).decode()}
    assert c.post(f"/api/c/{sid}/sync", json={"token": tok, **pic}).status_code == 200  # the live picture rides the sync
    live = c.get(f"/api/sessions/{sid}").json()
    assert live["status"] == "live" and live["phase"] == "exam" and [o["type"] for o in live["ongoing"]] == ["FACE_ABSENT"], live
    assert c.get(f"/api/sessions/{sid}/frame.jpg", params={"overlay": 1}).content == bytes(ov)
    feed = c.get("/api/feed", params={"since": since}).json()
    assert any(m["type"] == "event_started" and m["event"]["type"] == "FACE_ABSENT" for m in feed["msgs"])
    assert c.get("/api/feed", params={"since": feed["last"]}).json()["msgs"] == []  # nothing new

    b.end_in()
    b.run_slices(math.inf)
    push()
    row = c.get(f"/api/sessions/{sid}").json()
    assert row["status"] == "ended" and row["ongoing"] == [] and row["counts"]["FACE_ABSENT"] == 1
    assert c.get(f"/api/sessions/{sid}/frame.jpg").status_code == 204  # ended: no live picture
    ev = next(e for e in c.get(f"/api/sessions/{sid}/events").json() if e["type"] == "FACE_ABSENT")
    assert ev["clip_path"].endswith(".webm") and ev["clip_overlay"].endswith("_ov.webm") and set(ev["thumbs_overlay"]) == {"onset", "peak", "end"}
    r = c.get(f"/api/sessions/{sid}/files/{ev['clip_path']}", headers={"Range": "bytes=0-3"})
    assert r.status_code == 206 and r.content == b"\x1aE\xdf\xa3"
    assert c.get(f"/api/sessions/{sid}/timeline").json()["t_ms"]  # features.csv from the browser
    page = c.get(f"/api/sessions/{sid}/report").text
    assert "No one in view" in page or "FACE_ABSENT" in page
    assert "FACE_ABSENT" in c.get(f"/api/sessions/{sid}/decisions.csv").text
    # what the candidate API refuses
    bad = [c.put(f"/api/c/{sid}/files/../escape.txt", params={"token": tok}, content=b"x"),
           c.put(f"/api/c/{sid}/files/evidence/1.exe", params={"token": tok}, content=b"x"),
           c.put(f"/api/c/{sid}/files/events.json", params={"token": "wrong"}, content=b"x"),
           c.post(f"/api/c/{sid}/sync", json={"token": tok, "ops": [{"op": "sql", "q": "DROP TABLE sessions"}]})]
    assert [x.status_code for x in bad][1:] == [400, 401, 200]  # an ended session ignores everything but evidence
    c2 = _client(d)  # server mode, same data: nothing of the above leaks into policies or breaks loading
    _login(c2)
    assert {p["id"] for p in c2.get("/api/policies").json()} == {"default"}
    shutil.rmtree(d, ignore_errors=True)


def test_live_video_handshake_rides_the_candidate_sync():
    """Browser mode: a proctor's page posts a WebRTC offer; the candidate's next sync returns it; the candidate's
    answer comes back to that page only. Ended sessions get no offers."""
    try:
        from fastapi.testclient import TestClient

        from proctorlens.server.app import create_app
    except ImportError:
        return
    d = Path(tempfile.mkdtemp())
    c = TestClient(create_app((), d, password="pw", frontend=d / "none", browser_mode=True))
    _login(c)
    s = c.post("/api/sessions", json={"candidate_label": "V"}).json()
    sid, tok = s["session_id"], s["candidate_token"]
    assert c.post(f"/api/sessions/{sid}/rtc", json={"viewer": "p1", "sdp": "v=0 offer"}).status_code == 200
    assert c.get(f"/api/sessions/{sid}/rtc/p1").json() == {"answer": None}
    assert c.post(f"/api/c/{sid}/sync", json={"token": tok}).json()["rtc"] == [{"viewer": "p1", "offer": "v=0 offer"}]
    assert c.post(f"/api/c/{sid}/rtc", json={"token": "wrong", "viewer": "p1", "sdp": "x"}).status_code == 401
    assert c.post(f"/api/c/{sid}/rtc", json={"token": tok, "viewer": "p1", "sdp": "v=0 answer"}).status_code == 200
    assert c.get(f"/api/sessions/{sid}/rtc/p1").json() == {"answer": "v=0 answer"}
    assert c.post(f"/api/c/{sid}/sync", json={"token": tok}).json()["rtc"] == []  # answered: not offered again
    assert [x["id"] for x in c.get("/api/sessions", params={"active": 1}).json()] == [sid]
    c.post(f"/api/c/{sid}/sync", json={"token": tok, "ops": [{"op": "end", "answers": None}]})
    c.post(f"/api/sessions/{sid}/rtc", json={"viewer": "p2", "sdp": "v=0 late"})
    assert c.post(f"/api/c/{sid}/sync", json={"token": tok}).json()["rtc"] == []
    assert c.get("/api/sessions", params={"active": 1}).json() == [] and len(c.get("/api/sessions").json()) == 1


def test_deleting_a_class_removes_its_student_sessions():
    try:
        from fastapi.testclient import TestClient

        from proctorlens.server.app import create_app
    except ImportError:
        return
    d = Path(tempfile.mkdtemp())
    c = TestClient(create_app((), d, password="pw", frontend=d / "none", browser_mode=True))
    _login(c)
    k = c.post("/api/classes", json={"name": "K"}).json()
    tok = k["join_url"].rsplit("/", 1)[1]
    c.post(f"/api/join/{k['id']}", json={"token": tok, "roll": "1", "name": "A"})
    other = c.post("/api/sessions", json={"candidate_label": "keep"}).json()["session_id"]
    assert len(c.get("/api/sessions").json()) == 2
    assert c.delete(f"/api/classes/{k['id']}").status_code == 200
    assert [x["id"] for x in c.get("/api/sessions").json()] == [other] and c.get("/api/classes").json() == []


def _browser_client(d):
    try:
        from fastapi.testclient import TestClient

        from proctorlens.server.app import create_app
    except ImportError:
        return None
    c = TestClient(create_app((), d, password="pw", frontend=d / "none", browser_mode=True))
    _login(c)
    return c


_EV = {"type": "FACE_ABSENT", "start_ms": 0, "end_ms": 4000, "confidence": 0.9, "detector": "rule"}


def test_session_list_batches_events_reviews_and_notes():
    """The list fetches events, reviewed counts and notes for all its sessions at once: each still gets its own."""
    d = Path(tempfile.mkdtemp())
    if (c := _browser_client(d)) is None:
        return
    a, b = (c.post("/api/sessions", json={"candidate_label": n}).json() for n in ("A", "B"))
    for s, n in ((a, 2), (b, 1)):
        ops = [{"op": "event", "client_eid": i + 1, "event": {**_EV, "start_ms": i * 5000, "end_ms": i * 5000 + 4000}}
               for i in range(n)]
        assert c.post(f"/api/c/{s['session_id']}/sync", json={"token": s["candidate_token"], "ops": ops}).status_code == 200
    eid = c.get(f"/api/sessions/{a['session_id']}/events").json()[1]["id"]
    for decision in ("dismiss", "confirm"):  # two reviews of one event: one reviewed event, the latest decision
        c.post(f"/api/events/{eid}/review", json={"decision": decision})
    c.put(f"/api/sessions/{b['session_id']}/note", json={"note": "late start"})
    rows = {r["candidate_label"]: r for r in c.get("/api/sessions").json()}
    assert (rows["A"]["n_events"], rows["A"]["n_reviewed"], rows["A"]["review_note"]) == (2, 1, "")
    assert (rows["B"]["n_events"], rows["B"]["n_reviewed"], rows["B"]["review_note"]) == (1, 0, "late start")
    assert rows["A"]["counts"] == {"FACE_ABSENT": 2} and rows["A"]["flagged_s"] > rows["B"]["flagged_s"] > 0
    one = c.get(f"/api/sessions/{a['session_id']}").json()
    keys = ("n_events", "n_reviewed", "counts", "priority", "review_note")
    assert {k: one[k] for k in keys} == {k: rows["A"][k] for k in keys}
    assert [e["review"] and e["review"]["decision"] for e in c.get(f"/api/sessions/{a['session_id']}/events").json()] \
        == [None, "confirm"]
    shutil.rmtree(d, ignore_errors=True)


def test_browser_sync_retries_reloads_and_uploads():
    """Browser mode: a retried sync stores its events (and their feed rows) once; the live picture may come without a
    raw twin; a reload mid-exam numbers new events (and their evidence files) after the stored ones and keeps the
    start time; evidence uploads only for events the session sent; a clip that never arrived is not offered."""
    import base64
    import json

    from proctorlens.browser import BrowserSession
    from proctorlens.core.config import Config, from_dict
    from proctorlens.server.db import open_db

    class NoFace:
        def landmarks(self, rgba, w, h, t_ms, image_mode):
            return []

    d = Path(tempfile.mkdtemp())
    if (c := _browser_client(d)) is None:
        return
    s = c.post("/api/sessions", json={"candidate_label": "R"}).json()
    sid, tok = s["session_id"], s["candidate_token"]
    assert c.get(f"/api/c/{sid}", params={"token": tok}).json()["next_eid"] == 1
    since = c.get("/api/feed").json()["last"]
    batch = {"token": tok, "ops": [{"op": "start", "mode": "none"}, {"op": "event", "client_eid": 1, "event": _EV}],
             "feed": [{"type": "event_ended", "event": {"id": 1, "type": "FACE_ABSENT", "start_ms": 0, "end_ms": 4000}}]}
    for _ in range(2):  # the first response was lost: the page sends the same batch again
        assert c.post(f"/api/c/{sid}/sync", json=batch).status_code == 200
    assert len(c.get(f"/api/sessions/{sid}/events").json()) == 1
    assert len(c.get("/api/feed", params={"since": since}).json()["msgs"]) == 1
    pic = base64.b64encode(_JPEG).decode()
    assert c.post(f"/api/c/{sid}/sync", json={"token": tok, "overlay": pic}).status_code == 200  # no raw: same picture
    assert c.get(f"/api/sessions/{sid}/frame.jpg").content == _JPEG == c.get(f"/api/sessions/{sid}/frame.jpg?overlay=1").content
    assert c.get("/api/sessions/nope/frame.jpg").status_code == 404

    db = open_db(d)  # the exam started a while ago; then the student reloads the page
    db.run("UPDATE sessions SET started_at='2020-01-01T00:00:00+00:00' WHERE id=?", sid)
    info = c.get(f"/api/c/{sid}", params={"token": tok}).json()
    assert info["next_eid"] == 2
    b = BrowserSession(sid, from_dict(Config, info["config"]), NoFace(), {"names": {0: "person"}, "imgsz": [480, 640]},
                       identity=False, first_eid=info["next_eid"])
    b.dir = d / "browser" / sid
    b.control_in(json.dumps({"type": "exam_start"}))
    room = cv2.GaussianBlur((np.random.default_rng(0).random((480, 640, 3)) * 120 + 60).astype(np.uint8), (5, 5), 0)
    jpeg = cv2.imencode(".jpg", room)[1].tobytes()
    for i in range(50):  # nobody in view -> FACE_ABSENT
        t = 1.7e12 + i * 100
        b.prepare(jpeg)
        di = b.detection_input(t)
        b.frame_in(i, t, jpeg, *((np.zeros((1, 84, 6300), np.float32).tobytes(), [1, 84, 6300]) if di else ()))
    b.end_in()
    b.run_slices(math.inf)
    out = json.loads(b.drain())
    assert [o["client_eid"] for o in out["ops"] if o["op"] == "event"] == [2]
    ev_files = [f for f in out["files"] if f.startswith("evidence/")]
    assert ev_files and all(f.startswith("evidence/2_") for f in ev_files)  # event 1's evidence is not overwritten
    assert c.post(f"/api/c/{sid}/sync", json={"token": tok, "ops": out["ops"], "feed": out["feed"]}).status_code == 200
    assert c.get(f"/api/sessions/{sid}").json()["started_at"] == "2020-01-01T00:00:00+00:00"  # the restart kept it
    for f in out["files"]:  # the clips are never uploaded (as when the browser fails to encode one)
        assert c.put(f"/api/c/{sid}/files/{f}", params={"token": tok}, content=(b.dir / f).read_bytes()).status_code == 200
    assert c.put(f"/api/c/{sid}/files/evidence/3_peak.jpg", params={"token": tok}, content=_JPEG).status_code == 409
    evs = {e["id"]: e for e in c.get(f"/api/sessions/{sid}/events").json()}
    new = evs[max(evs)]
    assert new["thumbs"] and set(new["thumbs_overlay"]) == {"onset", "peak", "end"}
    assert db.one("SELECT clip_path FROM events WHERE id=?", new["id"])["clip_path"] == "evidence/2.webm"
    assert new["clip_path"] is None and new["clip_overlay"] is None  # recorded, but no file arrived
    shutil.rmtree(d, ignore_errors=True)


def test_deleting_a_live_session_stops_it():
    """Server mode: deleting a session whose candidate is still connected ends it; the socket closes instead of the
    session writing for rows that are gone."""
    d = Path(tempfile.mkdtemp())
    if (c := _client(d)) is None:
        return
    from starlette.websockets import WebSocketDisconnect

    _login(c)
    s = c.post("/api/sessions", json={"candidate_label": "D"}).json()
    sid = s["session_id"]
    with c.websocket_connect(f"/ws/stream/{sid}?token={s['candidate_token']}") as ws:
        cand = Candidate(ws)
        cand.frame(3)
        cand.ask({"type": "exam_start"})
        cand.frame(10)
        assert c.delete(f"/api/sessions/{sid}").status_code == 200
        ws.send_json({"type": "browser_event", "kind": "hidden", "t": cand.t})
        try:
            while True:
                ws.receive_json()
        except WebSocketDisconnect:
            pass
    assert c.get(f"/api/sessions/{sid}").status_code == 404 and c.get("/api/health").json()["live_sessions"] == 0
    shutil.rmtree(d, ignore_errors=True)
