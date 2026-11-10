"""Store for the web app (master spec 10.4; docs/DECISIONS.md ADR-016, ADR-019). SQLite on stdlib sqlite3 locally;
Postgres (PgDB) where the server has no disk of its own (Vercel: Neon), with the same SQL (? placeholders) and the
same schema text. Feature series, clips and keyframes are files (data/sessions/<id>/ on disk, the `files` table in
Postgres; see server/store.py); rows hold their paths. Review segments are not stored: they are recomputed from the
events (explain/review.py), so they can never go stale. One short-lived connection per call, because the API (event
loop) and the session workers (threads) both write."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import re
import sqlite3
import threading
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
           ("events", "client_eid", "INTEGER"),  # browser mode: the candidate browser's own event number
           ("classes", "exam_json", "TEXT")]  # custom {minutes, questions}; NULL = the built-in mock-exam bank
# after COLUMNS (client_eid is one): a retried browser sync must not store an event twice. NULLs stay distinct in a
# unique index (SQLite and Postgres), so server-mode events (no client_eid) are unaffected; duplicates stored before
# the index existed are dropped first (the oldest row, the one the evidence went to, stays)
MIGRATE = [("DELETE FROM events WHERE client_eid IS NOT NULL AND id NOT IN "
            "(SELECT MIN(id) FROM events WHERE client_eid IS NOT NULL GROUP BY session_id, client_eid)"),
           "CREATE UNIQUE INDEX IF NOT EXISTS events_client ON events(session_id, client_eid)"]
SERIAL = ("events", "reviews", "browser_events", "ground_truth", "feed")  # tables whose INSERT returns the new id
EVENT_ORDER = "ORDER BY start_ms, end_ms, type, id"  # = Pipeline.finish() order, so report indices line up


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: str | Path):
        self.path, self._local = str(path), threading.local()
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # a fresh --data folder
        with self.tx() as c:
            c.execute("PRAGMA journal_mode = WAL")  # readers never wait on the session workers' writes
            c.executescript(SCHEMA)
            for table, col, typ in COLUMNS:
                if col not in {r[1] for r in c.execute(f"PRAGMA table_info({table})")}:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
            for stmt in MIGRATE:
                c.execute(stmt)

    @contextlib.contextmanager
    def tx(self):
        """One transaction. Calls on this db from the same thread inside it join it (a whole request, one commit)."""
        if (c := getattr(self._local, "c", None)) is not None:
            yield c
            return
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            with c:  # commit, or roll back on error
                self._local.c = c
                yield c
        finally:
            self._local.c = None
            c.close()

    def all(self, sql: str, *args) -> list[dict]:
        with self.tx() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def one(self, sql: str, *args) -> dict | None:
        return next(iter(self.all(sql, *args)), None)

    def run(self, sql: str, *args) -> int | None:
        """The new row id of an INSERT; None when it inserted nothing (ON CONFLICT DO NOTHING)."""
        with self.tx() as c:
            cur = c.execute(sql, args)
            return cur.lastrowid if cur.rowcount else None  # lastrowid alone keeps the tx's previous insert

    def add_event(self, sid: str, ev: Event, explanation: str, client_eid: int | None = None) -> int | None:
        """None: a browser event this session already sent (a retried sync)."""
        return self.run("INSERT INTO events (session_id, type, start_ms, end_ms, confidence, detector, details_json, "
                        "attribution_json, explanation, thumbs_json, status, client_eid) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT DO NOTHING", sid, ev.type, ev.start_ms, ev.end_ms, ev.confidence, ev.detector,
                        _json(ev.details), _json(ev.attribution), explanation, "{}", ev.status, client_eid)


def pg_sql(sql: str) -> str:
    """SQLite-dialect statement (as written in this package) -> Postgres: ? -> %s (no literal ? or % in our SQL)."""
    return sql.replace("%", "%%").replace("?", "%s")


def pg_schema() -> list[str]:
    """Every DDL statement of the Postgres setup, in order. REAL is float4 there: epoch seconds would lose ~64 s."""
    s = (SCHEMA.replace("INTEGER PRIMARY KEY,", "BIGSERIAL PRIMARY KEY,").replace("BLOB", "BYTEA")
         .replace(" REAL", " DOUBLE PRECISION"))
    return ([x.strip() for x in s.split(";") if x.strip()]
            + [f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}" for table, col, typ in COLUMNS]
            + [f"ALTER TABLE {t} ALTER COLUMN {c} TYPE DOUBLE PRECISION"  # databases made while it was REAL
               for t, c in (("feed", "created_at"), ("live_frames", "t"), ("rtc", "t"))]
            + MIGRATE)


PG_VERSION = hashlib.sha1("\n".join(pg_schema()).encode()).hexdigest()[:16]  # changes whenever the DDL does


class PgDB(DB):
    """Postgres: same calls, same SQL text (pg_sql translates placeholders), the SQLite schema with BIGSERIAL ids
    and BYTEA blobs. One connection per server instance, reused (a new one per query cost a TLS handshake each:
    a browser's end-of-exam sync took 20 s). A single statement runs straight on the autocommit connection (one round
    trip); tx() is for several in one transaction. A connection the database dropped (Neon suspends idle computes) is
    reopened and the statement retried once. ponytail: one lock per instance serializes its queries; a pool if that
    queues."""

    def __init__(self, url: str):
        import psycopg  # only where Postgres is used (requirements of the Vercel function)

        self.url, self._psycopg, self._conn, self._lock, self._local = url, psycopg, None, threading.RLock(), threading.local()
        try:  # cold start: the schema is current -> one query, not ~25 DDL statements
            if (self.one("SELECT v FROM settings WHERE k='schema'") or {}).get("v") == PG_VERSION:
                return
        except psycopg.errors.UndefinedTable:  # a fresh database
            pass
        with self.tx() as c:
            # Vercel cold-starts several instances at once: one at a time through the DDL (else Postgres deadlocks)
            c.execute("SELECT pg_advisory_xact_lock(7243)")
            for stmt in pg_schema():
                c.execute(stmt)
            c.execute("INSERT INTO settings (k, v) VALUES ('schema', %s) ON CONFLICT (k) DO UPDATE SET v = excluded.v",
                      (PG_VERSION,))

    def _retry(self, fn):
        """fn(connection), on a fresh connection once more if the current one turns out dead. ponytail: a statement
        whose commit landed just before the drop runs twice; ours are upserts or harmless to repeat."""
        from psycopg.rows import dict_row

        with self._lock:
            for last in (False, True):
                try:
                    if self._conn is None or self._conn.closed:
                        self._conn = self._psycopg.connect(self.url, row_factory=dict_row, autocommit=True)
                    return fn(self._conn)
                except (self._psycopg.OperationalError, self._psycopg.InterfaceError):
                    if last:
                        raise
                    with contextlib.suppress(Exception):
                        self._conn.close()
                    self._conn = None

    @contextlib.contextmanager
    def tx(self):
        if (c := getattr(self._local, "c", None)) is not None:  # joins the transaction this thread has open
            yield c
            return
        with self._lock, contextlib.ExitStack() as stack:
            def begin(c):
                stack.enter_context(c.transaction())  # BEGIN (commit on success, roll back on error): safe to retry
                return c
            self._local.c = c = self._retry(begin)
            try:
                yield c
            finally:
                self._local.c = None

    def _exec(self, sql: str, args) -> list[dict]:
        q = pg_sql(sql)
        if (m := re.match(r"\s*INSERT INTO (\w+)", sql)) and m.group(1) in SERIAL:
            q += " RETURNING id"

        def run(c):
            cur = c.execute(q, args)
            return cur.fetchall() if cur.description else []
        c = getattr(self._local, "c", None)
        return run(c) if c is not None else self._retry(run)  # inside a transaction: no retry (it is broken)

    def all(self, sql: str, *args) -> list[dict]:
        return [{k: bytes(v) if isinstance(v, memoryview) else v for k, v in r.items()} for r in self._exec(sql, args)]

    def run(self, sql: str, *args) -> int | None:
        return r[0]["id"] if (r := self._exec(sql, args)) else None  # None: ON CONFLICT DO NOTHING did nothing


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
