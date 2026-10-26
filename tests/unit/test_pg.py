"""The web app on Postgres (the Vercel setup, ADR-019): the same flows as test_server.py, each against a fresh
database. Runs only when PROCTORLENS_TEST_PG names an admin connection URI (e.g. a local `pgserver`); skips otherwise."""
import os
import secrets

from tests.unit import test_server as server_tests


def _with_fresh_db(test) -> None:
    admin = os.environ.get("PROCTORLENS_TEST_PG")
    if not admin:
        return
    import psycopg

    name = f"pl_test_{secrets.token_hex(4)}"
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"CREATE DATABASE {name}")
    url = admin.rsplit("/", 1)[0] + f"/{name}"
    old = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        test()
    finally:
        if old is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def test_pg_browser_mode_session_through_the_api():
    _with_fresh_db(server_tests.test_browser_mode_session_through_the_api)


def test_pg_class_link_and_session_views():
    _with_fresh_db(server_tests.test_class_link_gives_each_student_a_session)


def test_pg_files_live_in_the_database():
    def check():
        import tempfile
        from pathlib import Path

        from proctorlens.server.db import PgDB, open_db
        from proctorlens.server.store import DBStore

        db = open_db(Path(tempfile.mkdtemp()))
        assert isinstance(db, PgDB)
        db.run("INSERT INTO sessions (id, token, status) VALUES ('s', 't', 'created')")
        st = DBStore(db)
        st.write("s", "evidence/1_peak.jpg", b"\xff\xd8jpg")
        st.write("s", "evidence/1_peak.jpg", b"\xff\xd8jpg2")  # upsert
        assert st.read("s", "evidence/1_peak.jpg") == b"\xff\xd8jpg2" and st.names("s") == {"evidence/1_peak.jpg"}
        assert (st.local("s") / "evidence/1_peak.jpg").read_bytes() == b"\xff\xd8jpg2"
        eid = db.run("INSERT INTO events (session_id, type, start_ms, end_ms) VALUES ('s', 'X', 0, 1)")
        assert eid >= 1 and db.one("SELECT type FROM events WHERE id=?", eid)["type"] == "X"
        db.run("DELETE FROM sessions WHERE id='s'")  # cascades to events and files
        assert st.names("s") == set() and db.one("SELECT 1 x FROM events WHERE id=?", eid) is None
    _with_fresh_db(check)


def test_pg_instances_starting_together():
    """Vercel cold-starts several instances at once: all set up the same fresh database without a deadlock."""
    def check():
        import tempfile
        from concurrent.futures import ThreadPoolExecutor

        from proctorlens.server.app import create_app

        with ThreadPoolExecutor(6) as ex:
            apps = list(ex.map(lambda _: create_app([], tempfile.mkdtemp(), browser_mode=True), range(6)))
        assert len(apps) == 6
    _with_fresh_db(check)


def test_pg_reconnects_after_the_database_drops_the_connection():
    """One connection per instance is reused; Neon closes it when an idle compute suspends: the next query reconnects."""
    def check():
        import tempfile
        from pathlib import Path

        import psycopg

        from proctorlens.server.db import open_db

        db = open_db(Path(tempfile.mkdtemp()))
        pid = db.one("SELECT pg_backend_pid() AS pid")["pid"]
        assert db.one("SELECT pg_backend_pid() AS pid")["pid"] == pid  # reused, not one per query
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        assert db.one("SELECT 1 AS x")["x"] == 1 and db.one("SELECT pg_backend_pid() AS pid")["pid"] != pid
    _with_fresh_db(check)
