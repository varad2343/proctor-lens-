"""FastAPI app (master spec 10; docs/DECISIONS.md ADR-016): REST for the proctor dashboard, /ws/stream for candidate
frames + control, /ws/proctor for live events, and the built frontend (frontend/dist) at /. `proctorlens serve` binds
127.0.0.1 by default (local processing). Proctor auth: one account from env vars, HttpOnly SameSite=Strict cookie.
Candidates have no account: each session has a random token carried in its join link."""
from __future__ import annotations

import asyncio
import contextlib
import csv
import dataclasses
import io
import re
import datetime as dt
import hmac
import json
import os
import secrets
import shutil
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from proctorlens import __version__
from proctorlens.core.config import Config, from_dict, load_config
from proctorlens.core.types import MONITORING_DEGRADED
from proctorlens.explain.report import plot_svg, write_report
from proctorlens.explain.review import segments, ts
from proctorlens.features.schema import COLUMNS
from proctorlens.io import _clean, load_events, load_features, save_events
from proctorlens.server.db import DB, EVENT_ORDER, event_of
from proctorlens.server.live import LiveSession, now_iso

COOKIE = "pl_token"
NOTE_FILE = "review_note.txt"  # the reviewer's session note, in data/sessions/<id>/
COOKIE_DAYS = 30  # proctor stays logged in across browser and server restarts
SERIES = ("off_screen_score", "quality")  # timeline traces (the two Review.tsx plots)


class Login(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=200)


class LinkLogin(BaseModel):
    key: str = Field(max_length=200)


class NewSession(BaseModel):
    candidate_label: str = Field(min_length=1, max_length=100)
    exam_id: str = Field("mock", max_length=100)
    policy_id: str = "default"


class NewClass(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    policy_id: str = "default"


class JoinIn(BaseModel):
    token: str = Field(max_length=100)
    roll: str = Field(min_length=1, max_length=30)
    name: str = Field(min_length=1, max_length=100)


def _natural(s: str | None) -> list:
    """Sort key: roll "2" before "10"."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", s or "") if t]


class ClassOpen(BaseModel):
    open: bool


class NoteIn(BaseModel):
    note: str = Field("", max_length=5000)


class ReviewIn(BaseModel):
    decision: Literal["confirm", "dismiss", "needs_more_info"]
    note: str = Field("", max_length=2000)


class PolicyIn(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    yaml: str = Field(max_length=20000)


class LabelIn(BaseModel):
    type: str = Field(max_length=60)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    source: Literal["cue", "annotator"] = "annotator"
    annotator_id: str = Field("", max_length=60)


def _default_landmarker(cfg: Config):
    from proctorlens.perception.landmarks import Landmarker

    return Landmarker(cfg.models.landmarker)


def _default_perceiver(cfg: Config, lm):
    from proctorlens.perception import Perceiver

    return Perceiver(cfg, landmarker=lm)


def create_app(config_paths=(), data_dir: str | Path = "data", *, password: str | None = None,
               landmarker_factory=_default_landmarker, perceiver_factory=_default_perceiver,
               frontend: str | Path = "frontend/dist", retention_days: float | None = None) -> FastAPI:
    """config_paths = the server config (models, thresholds); a session's policy (DB) is merged on top of it.
    The factories are injectable so tests run without models."""
    data = Path(data_dir)
    (data / "sessions").mkdir(parents=True, exist_ok=True)
    db = DB(data / "proctorlens.db")
    load_config(*config_paths)  # a bad server config fails at start, not at the first new session
    if db.one("SELECT id FROM policies WHERE id='default'") is None:
        db.run("INSERT INTO policies VALUES ('default', ?)", "# overrides on top of the server config\n{}\n")
    user = os.environ.get("PROCTORLENS_USER", "proctor")
    pw = password or os.environ.get("PROCTORLENS_PASSWORD") or "proctor"  # the default only suits this machine (ADR-018)
    auth_file = data / "auth.json"  # only the cookie-signing secret; passwords stored by older versions are dropped
    old = json.loads(auth_file.read_text(encoding="utf-8")) if auth_file.is_file() else {}
    auth = {"secret": old.get("secret") or secrets.token_hex(32)}
    auth_file.write_text(json.dumps(auth), encoding="utf-8")

    def check(p: str) -> bool:
        return hmac.compare_digest(p.encode(), pw.encode())

    key = hmac.new(auth["secret"].encode(), f"{user}\0{pw}".encode(), "sha256").digest()  # new password: old cookies die
    revoked: set[str] = set()  # ponytail: logout is remembered in memory; after a restart a logged-out cookie works until it expires

    def sign(body: str) -> str:
        return hmac.new(key, body.encode(), "sha256").hexdigest()

    def new_token() -> str:  # "<expiry>.<nonce>.<sig>": the nonce keeps two logins in one second apart (logout)
        body = f"{int(time.time()) + COOKIE_DAYS * 86400}.{secrets.token_hex(8)}"
        return f"{body}.{sign(body)}"

    def valid(tok: str | None) -> bool:
        body, _, sig = (tok or "").rpartition(".")
        exp = body.partition(".")[0]
        return (exp.isascii() and exp.isdigit() and len(exp) <= 12 and int(exp) > time.time() and tok not in revoked
                and hmac.compare_digest(sig.encode(), sign(body).encode()))

    def set_login(response: Response) -> None:
        response.set_cookie(COOKIE, new_token(), httponly=True, samesite="strict", max_age=COOKIE_DAYS * 86400)
    live: dict[str, LiveSession] = {}
    proctors: set[WebSocket] = set()

    def join_url(sid: str, token: str) -> str:
        """Candidate link; absolute on the public (tunnel) address when there is one, so a link copied on this PC
        opens on a phone or another computer."""
        return f"{getattr(app.state, 'public_origin', '')}/#/c/{sid}/{token}"

    def sdir(sid: str) -> Path:
        return data / "sessions" / sid

    def delete_session(sid: str) -> None:
        db.run("DELETE FROM sessions WHERE id=?", sid)  # cascades to events, reviews, telemetry, labels
        shutil.rmtree(sdir(sid), ignore_errors=True)

    days = retention_days if retention_days is not None else float(os.environ.get("PROCTORLENS_RETENTION_DAYS", 30))
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat(timespec="seconds")
    for r in db.all("SELECT id FROM sessions WHERE status='ended' AND ended_at < ?", cutoff):
        delete_session(r["id"])  # retention (master spec 16): ended sessions older than `days` are removed at start

    async def run(s: LiveSession, fn, *a):
        return await asyncio.get_running_loop().run_in_executor(s.worker, fn, *a)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        yield
        for s in list(live.values()):  # graceful shutdown finalizes open events (master spec 10.5)
            if s.phase == "exam":
                await run(s, s.end)

    app = FastAPI(title="ProctorLens", version=__version__, lifespan=lifespan)

    def new_link() -> None:
        """One-click login, printed by `proctorlens serve`. A random key per server start and per logout, so a copied
        or logged-out link stops working. It rides in the URL fragment and the page POSTs it, so it never appears in a
        request line (access logs)."""
        app.state.link_key = secrets.token_hex(16)
        app.state.login_link = f"/#/link/{app.state.link_key}"

    new_link()

    def proctor(request: Request) -> None:
        if not valid(request.cookies.get(COOKIE)):
            raise HTTPException(401, "login required")

    api = APIRouter(prefix="/api", dependencies=[Depends(proctor)])

    def session_or_404(sid: str) -> dict:
        if (row := db.one("SELECT * FROM sessions WHERE id=?", sid)) is None:
            raise HTTPException(404, "no such session")
        return row

    def cfg_of(row: dict) -> Config:
        return from_dict(Config, json.loads(row["policy_snapshot_json"]))  # validated when the session was created

    def event_rows(sid: str) -> list[dict]:
        return db.all(f"SELECT * FROM events WHERE session_id=? {EVENT_ORDER}", sid)

    def last_review(eid: int) -> dict | None:
        return db.one("SELECT decision, note, reviewer, created_at FROM reviews WHERE event_id=? ORDER BY id DESC "
                      "LIMIT 1", eid)

    def features(sid: str) -> pd.DataFrame | None:
        if (s := live.get(sid)) is not None and s.pl is not None:
            return pd.DataFrame(list(s.pl.rows), columns=COLUMNS)
        return load_features(p) if (p := sdir(sid) / "features.parquet").exists() else None

    def session_json(row: dict) -> dict:
        sid, cfg = row["id"], cfg_of(row)
        evs = [event_of(r) for r in event_rows(sid)]
        segs = segments(evs, cfg)
        n_rev = db.one("SELECT COUNT(DISTINCT r.event_id) n FROM reviews r JOIN events e ON e.id = r.event_id "
                       "WHERE e.session_id=?", sid)["n"]
        start, end = (dt.datetime.fromisoformat(row[k]) if row[k] else None for k in ("started_at", "ended_at"))
        cal = json.loads(row["calibration_json"] or "null") or {}
        ls = live.get(sid)  # view = the snapshot request threads may read (LiveSession.frame)
        ongoing = ls.view[3] if ls and ls.phase == "exam" and ls.view else []
        return {**{k: row[k] for k in ("id", "exam_id", "candidate_label", "roll", "policy_id", "status", "mode",
                                       "created_at", "started_at", "ended_at")},
                "phase": live[sid].phase if sid in live else row["status"],
                "ongoing": [{"type": e.type, "start_ms": e.start_ms} for e in ongoing],  # flags in progress now
                "join_url": join_url(sid, row["token"]),
                "duration_s": None if start is None else
                round(((end or dt.datetime.now(dt.timezone.utc)) - start).total_seconds(), 1),
                "calibration": {k: cal.get(k) for k in ("mode", "error", "accepted", "tries")},
                "policy": dataclasses.asdict(cfg.policy),
                "counts": Counter(e.type for e in evs),
                "priority": Counter(s["priority_label"] for s in segs if s["lane"] == "review"),
                "blind_spot_s": round(sum(e.end_ms - e.start_ms for e in evs if e.type == MONITORING_DEGRADED) / 1000, 1),
                "flagged_s": round(sum(s["end_ms"] - s["start_ms"] for s in segs if s["lane"] == "review") / 1000, 1),
                "n_events": len(evs), "n_reviewed": n_rev,
                "review_note": (n.read_text(encoding="utf-8") if (n := sdir(sid) / NOTE_FILE).is_file() else "")}

    def event_json(r: dict) -> dict:
        th = json.loads(r["thumbs_json"] or "{}")
        ov = {k: q for k, v in th.items() if (sdir(r["session_id"]) / (q := v[:-4] + "_ov.jpg")).is_file()}
        c_ov = r["clip_path"] and r["clip_path"][:-4] + "_ov.mp4"  # detector-view twin of the clip, when written
        c_ov = c_ov if c_ov and (sdir(r["session_id"]) / c_ov).is_file() else None
        return {**{k: r[k] for k in ("id", "session_id", "type", "start_ms", "end_ms", "confidence", "detector",
                                     "explanation", "clip_path", "status")},
                "details": json.loads(r["details_json"]), "attribution": json.loads(r["attribution_json"] or "null"),
                "thumbs": th, "thumbs_overlay": ov, "clip_overlay": c_ov, "review": last_review(r["id"])}

    # ------------------------------------------------------------------------------------------------ auth + meta

    @app.post("/api/auth/login")
    def login(body: Login, response: Response):
        ok = hmac.compare_digest(body.username.encode(), user.encode()) & check(body.password)
        if not ok:
            time.sleep(0.5)  # slows guessing
            raise HTTPException(401, "wrong username or password")
        set_login(response)
        return {"user": user}

    @app.post("/api/auth/link")
    def login_link(body: LinkLogin, response: Response):
        if not hmac.compare_digest(body.key.encode(), app.state.link_key.encode()):
            time.sleep(0.5)
            raise HTTPException(401, "this login link is no longer valid: use the latest one the server printed")
        set_login(response)
        return {"user": user}

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "live_sessions": len(live)}

    @api.post("/auth/logout")
    def logout(request: Request, response: Response):
        revoked.add(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE)
        new_link()
        print(f"logged out: new one-click proctor login: {getattr(app.state, 'origin', '')}{app.state.login_link}")
        return {}

    @api.get("/auth/me")
    def me():
        return {"user": user}

    @api.get("/policies")
    def policies():
        return db.all("SELECT id, yaml FROM policies ORDER BY id")

    @api.post("/policies")
    def save_policy(body: PolicyIn):
        try:
            load_config(*config_paths, texts=(body.yaml,))
        except Exception as e:  # any YAML / unknown-key / validation failure is the caller's input
            raise HTTPException(400, f"invalid policy: {type(e).__name__}: {e}") from e
        db.run("INSERT OR REPLACE INTO policies VALUES (?, ?)", body.id, body.yaml)
        return {"id": body.id}

    # ------------------------------------------------------------------------------------------------ sessions

    def new_session(label: str, exam_id: str, policy_id: str, roll: str | None = None) -> dict:
        if (pol := db.one("SELECT yaml FROM policies WHERE id=?", policy_id)) is None:
            raise HTTPException(404, "no such policy")
        cfg = load_config(*config_paths, texts=(pol["yaml"],))
        sid, token = secrets.token_hex(6), secrets.token_urlsafe(16)
        db.run("INSERT INTO sessions (id, exam_id, candidate_label, roll, policy_id, policy_snapshot_json, token, "
               "status, created_at) VALUES (?,?,?,?,?,?,?,?,?)",  # model hashes go in each session's events.json header
               sid, exam_id, label, roll, policy_id, json.dumps(dataclasses.asdict(cfg)), token, "created", now_iso())
        return {"session_id": sid, "candidate_token": token, "join_url": join_url(sid, token)}

    @api.post("/sessions")
    def create_session(body: NewSession):
        return new_session(body.candidate_label, body.exam_id, body.policy_id)

    @api.get("/sessions")
    def list_sessions(class_id: str | None = None):
        rows = (sorted(db.all("SELECT * FROM sessions WHERE exam_id=?", class_id), key=lambda r: _natural(r["roll"]))
                if class_id else db.all("SELECT * FROM sessions ORDER BY created_at DESC"))
        return [session_json(r) for r in rows]

    # ------------------------------------------------------------------------------------------------ classes
    # One join link for a whole class: each student who opens it types a name and gets a session of their own
    # (sessions.exam_id = the class id), so monitoring, flags and review stay per student.

    def class_json(r: dict) -> dict:
        n = db.one("SELECT COUNT(*) n, SUM(status='live') live FROM sessions WHERE exam_id=?", r["id"])
        return {**{k: r[k] for k in ("id", "name", "policy_id", "created_at")}, "open": bool(r["open"]),
                "join_url": f"{getattr(app.state, 'public_origin', '')}/#/j/{r['id']}/{r['token']}",
                "n_students": n["n"], "n_live": n["live"] or 0}

    def class_or_404(cid: str) -> dict:
        if (r := db.one("SELECT * FROM classes WHERE id=?", cid)) is None:
            raise HTTPException(404, "no such class")
        return r

    @api.post("/classes")
    def create_class(body: NewClass):
        if db.one("SELECT id FROM policies WHERE id=?", body.policy_id) is None:
            raise HTTPException(404, "no such policy")
        cid = "k" + secrets.token_hex(5)  # never collides with a manual session's exam_id ("mock" or free text)
        db.run("INSERT INTO classes (id, name, policy_id, token, open, created_at) VALUES (?,?,?,?,1,?)",
               cid, body.name.strip(), body.policy_id, secrets.token_urlsafe(16), now_iso())
        return class_json(class_or_404(cid))

    @api.get("/classes")
    def list_classes():
        return [class_json(r) for r in db.all("SELECT * FROM classes ORDER BY created_at DESC")]

    @api.get("/classes/{cid}")
    def get_class(cid: str):
        return class_json(class_or_404(cid))

    @api.post("/classes/{cid}/open")
    def set_class_open(cid: str, body: ClassOpen):
        """Closing the link stops new students joining; students already in keep going."""
        class_or_404(cid)
        db.run("UPDATE classes SET open=? WHERE id=?", int(body.open), cid)
        return get_class(cid)

    def joinable(cid: str, token: str) -> dict:
        r = db.one("SELECT * FROM classes WHERE id=?", cid)
        if r is None or not hmac.compare_digest(r["token"].encode(), token.encode()):
            raise HTTPException(404, "this class link is not valid")  # same answer: no probing for class ids
        return r

    @app.get("/api/join/{cid}")  # public: the student side of a class link
    def join_info(cid: str, token: str = ""):
        r = joinable(cid, token)
        return {"name": r["name"], "open": bool(r["open"])}

    @app.post("/api/join/{cid}")
    def join_class(cid: str, body: JoinIn):
        """ponytail: anyone holding the class link can add students; close the link once everyone is in."""
        r = joinable(cid, body.token)
        if not r["open"]:
            raise HTTPException(409, "this class link is closed; ask your proctor")
        name, roll = " ".join(body.name.split()), " ".join(body.roll.split())
        if not name or not roll:
            raise HTTPException(422, "enter your roll number and name")
        if db.one("SELECT id FROM sessions WHERE exam_id=? AND lower(roll)=lower(?)", cid, roll):
            raise HTTPException(409, f"roll number {roll} has already joined this class; if that wasn't you, "
                                     "or you changed device, ask your proctor")
        s = new_session(name, cid, r["policy_id"], roll)
        return {"session_id": s["session_id"], "candidate_token": s["candidate_token"]}

    @api.get("/sessions/{sid}")
    def get_session(sid: str):
        return session_json(session_or_404(sid))

    @api.delete("/sessions/{sid}")
    async def remove_session(sid: str):
        session_or_404(sid)
        if (s := live.pop(sid, None)) is not None and s.phase == "exam":
            await run(s, s.end)  # finish the evidence threads before their folder goes
        delete_session(sid)
        return {}

    @api.get("/sessions/{sid}/events")
    def get_events(sid: str):
        session_or_404(sid)
        return [event_json(r) for r in event_rows(sid)]

    @api.get("/sessions/{sid}/segments")
    def get_segments(sid: str):
        row, rows = session_or_404(sid), event_rows(sid)
        segs = segments([event_of(r) for r in rows], cfg_of(row))
        return [{**s, "events": [rows[i]["id"] for i in s["events"]]} for s in segs]  # indices -> event ids

    @api.get("/sessions/{sid}/timeline")
    def timeline(sid: str, every: int = 5):
        """Feature traces, every `every`-th grid row (2 Hz by default), NaN = null."""
        session_or_404(sid)
        df = features(sid)
        if df is None or not len(df):
            return {"t_ms": [], "series": {}}
        d = df.iloc[::max(1, every)]
        return {"t_ms": d["t_ms"].astype(int).tolist(),
                "series": {c: _clean(pd.to_numeric(d[c], errors="coerce").tolist()) for c in SERIES}}

    @api.get("/sessions/{sid}/frame.jpg")
    def latest_frame(sid: str, overlay: bool = False):
        """Latest candidate frame: raw, or ?overlay=1 drawn with the detectors' view (exam phase; raw before that)."""
        session_or_404(sid)
        if (s := live.get(sid)) is None or s.latest_jpeg is None:
            return Response(status_code=204)  # no frame (yet / any more): not an error, the live view just shows none
        body = (overlay and s.view_jpeg()) or s.latest_jpeg
        return Response(body, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @api.get("/sessions/{sid}/files/{path:path}")
    def session_file(sid: str, path: str):
        """Clips, keyframes, report (FileResponse serves HTTP Range, so clips seek)."""
        session_or_404(sid)  # sid is now one of ours, so the root is a real session folder
        root = sdir(sid).resolve()
        if not (p := (root / path).resolve()).is_relative_to(root) or not p.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(p)

    @api.get("/sessions/{sid}/report")
    def report(sid: str):
        """Regenerate report.html from the DB (events, evidence paths, reviewer decisions) and redirect to it."""
        row, d = session_or_404(sid), sdir(sid)
        if row["status"] != "ended" or not (d / "events.json").is_file():
            raise HTTPException(409, "the report is available once the session has ended")
        rows = event_rows(sid)
        save_events([event_of(r) for r in rows], d / "events.json", load_events(d / "events.json")[0])
        media = {i: {**json.loads(r["thumbs_json"] or "{}"), **({"clip": r["clip_path"]} if r["clip_path"] else {})}
                 for i, r in enumerate(rows)}
        reviews = {i: rv for i, r in enumerate(rows) if (rv := last_review(r["id"]))}
        write_report(d, cfg_of(row), media=media, reviews=reviews)
        return RedirectResponse(f"/api/sessions/{sid}/files/report.html", status_code=303)

    @api.post("/sessions/{sid}/labels")
    def add_labels(sid: str, labels: list[LabelIn]):
        """Ground-truth import for evaluation (master spec 10.2)."""
        session_or_404(sid)
        if any(lb.end_ms < lb.start_ms for lb in labels):
            raise HTTPException(400, "end_ms < start_ms")
        for lb in labels:
            db.run("INSERT INTO ground_truth (session_id, type, start_ms, end_ms, source, annotator_id) "
                   "VALUES (?,?,?,?,?,?)", sid, lb.type, lb.start_ms, lb.end_ms, lb.source, lb.annotator_id)
        return {"added": len(labels)}

    @api.get("/sessions/{sid}/labels")
    def get_labels(sid: str):
        session_or_404(sid)
        return db.all("SELECT type, start_ms, end_ms, source, annotator_id FROM ground_truth WHERE session_id=? "
                      "ORDER BY start_ms", sid)

    # ------------------------------------------------------------------------------------------------ events

    def event_or_404(eid: int) -> dict:
        if (r := db.one("SELECT * FROM events WHERE id=?", eid)) is None:
            raise HTTPException(404, "no such event")
        return r

    @api.get("/events/{eid}/plot")
    def event_plot(eid: int):
        r = event_or_404(eid)
        df = features(r["session_id"])
        p = None if df is None else plot_svg(df, event_of(r), cfg_of(session_or_404(r["session_id"])))
        return {"svg": p and p[0], "caption": p and p[1]}

    @api.put("/sessions/{sid}/note")
    def set_note(sid: str, body: NoteIn):
        """Reviewer's note for the whole session (a file next to its evidence; it goes into the report)."""
        session_or_404(sid)
        sdir(sid).mkdir(parents=True, exist_ok=True)
        (sdir(sid) / NOTE_FILE).write_text(body.note, encoding="utf-8")
        return {"review_note": body.note}

    @api.get("/sessions/{sid}/decisions.csv")
    def decisions_csv(sid: str):
        """Every flag with its time, type, confidence and the latest reviewer decision, as a spreadsheet."""
        row = session_or_404(sid)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["roll", "name", "event_id", "type", "start", "end", "duration_s", "confidence", "decision", "note",
                    "reviewer", "reviewed_at"])
        for r in event_rows(sid):
            rv = last_review(r["id"]) or {}
            w.writerow([row["roll"] or "", row["candidate_label"], r["id"], r["type"], ts(r["start_ms"]), ts(r["end_ms"]), round((r["end_ms"] - r["start_ms"]) / 1000, 1),
                        round(r["confidence"], 3), rv.get("decision", ""), rv.get("note", ""), rv.get("reviewer", ""),
                        rv.get("created_at", "")])
        name = re.sub(r"[^A-Za-z0-9.-]+", "_", f"proctorlens_{row['roll'] or ''}_{row['candidate_label']}_{sid}.csv")
        return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",  # BOM: Excel reads it as UTF-8
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @api.post("/events/{eid}/review")
    def review(eid: int, body: ReviewIn):
        event_or_404(eid)
        db.run("INSERT INTO reviews (event_id, decision, note, reviewer, created_at) VALUES (?,?,?,?,?)",
               eid, body.decision, body.note, user, now_iso())
        return last_review(eid)

    app.include_router(api)

    # ------------------------------------------------------------------------------------------------ websockets

    async def broadcast(msgs: list[dict]) -> None:
        for m in msgs:
            for p in list(proctors):
                try:
                    await p.send_json(_clean(m))
                except Exception:  # a socket that died mid-send can raise anything: drop it
                    proctors.discard(p)

    @app.websocket("/ws/proctor")
    async def ws_proctor(ws: WebSocket):
        if not valid(ws.cookies.get(COOKIE)):
            await ws.close(code=4401)
            return
        await ws.accept()
        proctors.add(ws)
        try:
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass
        finally:
            proctors.discard(ws)

    @app.websocket("/ws/stream/{sid}")
    async def ws_stream(ws: WebSocket, sid: str, token: str = ""):
        """Binary = frame: seq uint32 LE, t_client_ms float64 LE, JPEG. Text = JSON control message. Latest frame
        wins: frames that arrive while one is processed are dropped (the ack tells the client what was handled)."""
        row = db.one("SELECT * FROM sessions WHERE id=?", sid)
        if row is None or row["status"] == "ended" or not hmac.compare_digest(row["token"].encode(), token.encode()):
            await ws.close(code=4401)
            return
        await ws.accept()
        if (s := live.get(sid)) is None:  # a reconnect resumes the same session
            s = live[sid] = LiveSession(sid, cfg_of(row), sdir(sid), db, landmarker_factory, perceiver_factory)
        slot: list[tuple] = []
        wake = asyncio.Event()

        async def send(res) -> None:
            cand, proc = res
            for m in cand:
                await ws.send_json(_clean(m))
            await broadcast(proc)

        async def pump() -> None:
            while True:
                try:
                    await asyncio.wait_for(wake.wait(), 0.5)
                except TimeoutError:
                    await send(await run(s, s.tick))
                    continue
                wake.clear()
                if slot:
                    await send(await run(s, s.frame, *slot.pop()))

        task = asyncio.create_task(pump())
        try:
            while (msg := await ws.receive())["type"] != "websocket.disconnect":
                if (b := msg.get("bytes")) is not None:
                    if len(b) > 12:
                        slot[:] = [(*struct.unpack_from("<Id", b), b[12:])]
                        wake.set()
                    continue
                try:
                    m = json.loads(msg.get("text") or "")
                    res = await run(s, s.control, m) if isinstance(m, dict) else ([], [])
                except (ValueError, TypeError, KeyError) as e:  # malformed control message: tell the client, go on
                    res = ([{"type": "error", "detail": f"{type(e).__name__}: {e}"}], [])
                await send(res)
                if s.phase == "ended":
                    live.pop(sid, None)
                    s.worker.shutdown(wait=False)
                    await ws.close()
                    break
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()

    if Path(frontend).is_dir():  # last: API and websocket routes win over the static files
        app.mount("/", StaticFiles(directory=frontend, html=True), name="ui")
    return app
