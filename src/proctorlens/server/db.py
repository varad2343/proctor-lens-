"""SQLite store for the web app (master spec 10.4) on stdlib sqlite3 (docs/DECISIONS.md ADR-016). Feature series,
clips and keyframes stay files under data/sessions/<id>/; rows hold their paths. Review segments are not stored:
they are recomputed from the events (explain/review.py), so they can never go stale. One short-lived connection per
call, because the API (event loop) and the session workers (threads) both write."""
from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path

from proctorlens.core.types import Event
from proctorlens.io import _clean

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, exam_id TEXT, candidate_label TEXT, policy_id TEXT,
  policy_snapshot_json TEXT, model_versions_json TEXT, calibration_json TEXT, token TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('created', 'live', 'ended')), mode TEXT, answers_json TEXT,
  created_at TEXT, started_at TEXT, ended_at TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, type TEXT, start_ms INTEGER, end_ms INTEGER,
  confidence REAL, detector TEXT, details_json TEXT, attribution_json TEXT, explanation TEXT, clip_path TEXT,
  thumbs_json TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  decision TEXT NOT NULL CHECK (decision IN ('confirm', 'dismiss', 'needs_more_info')), note TEXT, reviewer TEXT,
  created_at TEXT);
CREATE TABLE IF NOT EXISTS browser_events (id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, kind TEXT, t_ms INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS ground_truth (id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, type TEXT, start_ms INTEGER, end_ms INTEGER,
  source TEXT CHECK (source IN ('cue', 'annotator')), annotator_id TEXT);
CREATE TABLE IF NOT EXISTS policies (id TEXT PRIMARY KEY, yaml TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, start_ms);
"""
EVENT_ORDER = "ORDER BY start_ms, end_ms, type, id"  # = Pipeline.finish() order, so report indices line up


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self.tx() as c:
            c.execute("PRAGMA journal_mode = WAL")  # readers never wait on the session workers' writes
            c.executescript(SCHEMA)

    @contextlib.contextmanager
    def tx(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            with c:  # commit, or roll back on error
                yield c
        finally:
            c.close()

    def all(self, sql: str, *args) -> list[dict]:
        with self.tx() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def one(self, sql: str, *args) -> dict | None:
        return next(iter(self.all(sql, *args)), None)

    def run(self, sql: str, *args) -> int:
        with self.tx() as c:
            return c.execute(sql, args).lastrowid

    def add_event(self, sid: str, ev: Event, explanation: str) -> int:
        return self.run("INSERT INTO events (session_id, type, start_ms, end_ms, confidence, detector, details_json, "
                        "attribution_json, explanation, thumbs_json, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        sid, ev.type, ev.start_ms, ev.end_ms, ev.confidence, ev.detector, _json(ev.details),
                        _json(ev.attribution), explanation, "{}", ev.status)


def _json(o) -> str:
    return json.dumps(_clean(o), allow_nan=False)  # _clean: numpy -> python, NaN -> null


def event_of(row: dict) -> Event:
    return Event(row["type"], row["start_ms"], row["end_ms"], row["confidence"], row["detector"],
                 json.loads(row["details_json"]), json.loads(row["attribution_json"] or "null"), row["status"])
