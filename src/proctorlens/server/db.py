"""Store for the web app (master spec 10.4; docs/DECISIONS.md ADR-016, ADR-019). SQLite on stdlib sqlite3 locally;
Postgres (PgDB) where the server has no disk of its own (Vercel: Neon), with the same SQL (? placeholders) and the
same schema text. Feature series, clips and keyframes are files (data/sessions/<id>/ on disk, the `files` table in
Postgres; see server/store.py); rows hold their paths. Review segments are not stored: they are recomputed from the
events (explain/review.py), so they can never go stale. One short-lived connection per call, because the API (event
loop) and the session workers (threads) both write."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

from proctorlens.core.types import Event
from proctorlens.io import _clean

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, exam_id TEXT, candidate_label TEXT, policy_id TEXT,
  policy_snapshot_json TEXT, calibration_json TEXT, token TEXT NOT NULL,
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
CREATE TABLE IF NOT EXISTS classes (id TEXT PRIMARY KEY, name TEXT NOT NULL, policy_id TEXT NOT NULL,
  token TEXT NOT NULL, open INTEGER NOT NULL DEFAULT 1, created_at TEXT);
CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feed (id INTEGER PRIMARY KEY, session_id TEXT, msg TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS live_frames (session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
  overlay BLOB, raw BLOB, t REAL);
CREATE TABLE IF NOT EXISTS files (session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  path TEXT NOT NULL, data BLOB, PRIMARY KEY (session_id, path));
CREATE TABLE IF NOT EXISTS rtc (session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  viewer TEXT NOT NULL, offer TEXT NOT NULL, answer TEXT, t REAL, PRIMARY KEY (session_id, viewer));
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, start_ms);
"""
# columns added after the first release: (table, column, type); existing databases get them at start
COLUMNS = [("sessions", "roll", "TEXT"),  # class links
           ("sessions", "live_json", "TEXT"),  # browser mode: the latest status a candidate's browser sent
           ("events", "client_eid", "INTEGER")]  # browser mode: the candidate browser's own event number
SERIAL = ("events", "reviews", "browser_events", "ground_truth", "feed")  # tables whose INSERT returns the new id
EVENT_ORDER = "ORDER BY start_ms, end_ms, type, id"  # = Pipeline.finish() order, so report indices line up


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # a fresh --data folder
        with self.tx() as c:
            c.execute("PRAGMA journal_mode = WAL")  # readers never wait on the session workers' writes
            c.executescript(SCHEMA)
            for table, col, typ in COLUMNS:
                if col not in {r[1] for r in c.execute(f"PRAGMA table_info({table})")}:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

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

    def add_event(self, sid: str, ev: Event, explanation: str, client_eid: int | None = None) -> int:
        return self.run("INSERT INTO events (session_id, type, start_ms, end_ms, confidence, detector, details_json, "
                        "attribution_json, explanation, thumbs_json, status, client_eid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        sid, ev.type, ev.start_ms, ev.end_ms, ev.confidence, ev.detector, _json(ev.details),
                        _json(ev.attribution), explanation, "{}", ev.status, client_eid)


def pg_sql(sql: str) -> str:
    """SQLite-dialect statement (as written in this package) -> Postgres: ? -> %s (no literal ? or % in our SQL)."""
    return sql.replace("%", "%%").replace("?", "%s")


def pg_schema() -> list[str]:
    s = SCHEMA.replace("INTEGER PRIMARY KEY,", "BIGSERIAL PRIMARY KEY,").replace("BLOB", "BYTEA")
    return [x.strip() for x in s.split(";") if x.strip()]


class PgDB(DB):
    """Postgres: same calls, same SQL text (pg_sql translates placeholders), the SQLite schema with BIGSERIAL ids
    and BYTEA blobs. One connection per server instance, reused (a new one per query cost a TLS handshake each:
    a browser's end-of-exam sync took 20 s); checked before each transaction and reopened when the database dropped
    it (Neon suspends idle computes). ponytail: one lock per instance serializes its queries; a pool if that queues."""

    def __init__(self, url: str):
        import threading

        import psycopg  # only where Postgres is used (requirements of the Vercel function)

        self.url, self._psycopg, self._conn, self._lock = url, psycopg, None, threading.RLock()
        with self.tx() as c:
            # Vercel cold-starts several instances at once: one at a time through the DDL (else Postgres deadlocks)
            c.execute("SELECT pg_advisory_xact_lock(7243)")
            for stmt in pg_schema():
                c.execute(stmt)
            for table, col, typ in COLUMNS:
                c.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")

    @contextlib.contextmanager
    def tx(self):
        from psycopg.rows import dict_row

        with self._lock:
            try:
                if self._conn is None:
                    raise self._psycopg.OperationalError
                self._conn.execute("SELECT 1")
            except self._psycopg.Error:
                if self._conn is not None:
                    self._conn.close()
                self._conn = self._psycopg.connect(self.url, row_factory=dict_row, autocommit=True)
            with self._conn.transaction():  # commits on success, rolls back on error
                yield self._conn

    def all(self, sql: str, *args) -> list[dict]:
        with self.tx() as c:
            return [{k: bytes(v) if isinstance(v, memoryview) else v for k, v in r.items()}
                    for r in c.execute(pg_sql(sql), args).fetchall()]

    def run(self, sql: str, *args) -> int:
        q = pg_sql(sql)
        if (m := re.match(r"\s*INSERT INTO (\w+)", sql)) and m.group(1) in SERIAL:
            q += " RETURNING id"
        with self.tx() as c:
            cur = c.execute(q, args)
            return cur.fetchone()["id"] if q.endswith("RETURNING id") else 0


def open_db(data_dir: Path, env=None) -> DB:
    """Postgres when the environment names one (DATABASE_URL, or POSTGRES_URL from the Vercel/Neon integration),
    else SQLite at PROCTORLENS_DB or <data>/proctorlens.db."""
    import os

    env = os.environ if env is None else env
    if url := env.get("DATABASE_URL") or env.get("POSTGRES_URL"):
        return PgDB(url)
    return DB(env.get("PROCTORLENS_DB") or Path(data_dir) / "proctorlens.db")


def _json(o) -> str:
    return json.dumps(_clean(o), allow_nan=False)  # _clean: numpy -> python, NaN -> null


def event_of(row: dict) -> Event:
    return Event(row["type"], row["start_ms"], row["end_ms"], row["confidence"], row["detector"],
                 json.loads(row["details_json"]), json.loads(row["attribution_json"] or "null"), row["status"])
