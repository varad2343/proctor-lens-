"""Session files (clips, keyframes, features, events.json, report): a folder per session on disk, or rows of the
`files` table where the server has no lasting disk (Vercel). Paths are relative to the session folder."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class DiskStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, sid: str) -> Path:
        return self.root / sid

    def _p(self, sid: str, rel: str) -> Path | None:
        base = self.dir(sid).resolve()
        p = (base / rel).resolve()
        return p if p.is_relative_to(base) else None  # no way out of the session folder

    def read(self, sid: str, rel: str) -> bytes | None:
        p = self._p(sid, rel)
        return p.read_bytes() if p and p.is_file() else None

    def file(self, sid: str, rel: str) -> Path | None:
        """A disk path to serve directly (HTTP Range for seeking clips), when the store has one."""
        p = self._p(sid, rel)
        return p if p and p.is_file() else None

    def exists(self, sid: str, rel: str) -> bool:
        return self.file(sid, rel) is not None

    def names(self, sid: str) -> set[str]:
        d = self.dir(sid)
        return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} if d.is_dir() else set()

    def write(self, sid: str, rel: str, data: bytes) -> None:
        if (p := self._p(sid, rel)) is None:
            raise ValueError(f"bad path {rel!r}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def delete(self, sid: str) -> None:
        shutil.rmtree(self.dir(sid), ignore_errors=True)

    def local(self, sid: str) -> Path:
        """A folder holding the session's files, for code that reads and writes paths (the report)."""
        return self.dir(sid)

    def keep(self, sid: str, folder: Path, rels: list[str]) -> None:
        """After writing `rels` into local(sid): store them (on disk they are already there)."""


class DBStore(DiskStore):
    """Files in the database. local() copies a session's files into a temporary folder; keep() stores back what code
    wrote there. ponytail: whole files in rows (Vercel responses cap at 4.5 MB; clips are ~1-2 MB at 400 kbit/s)."""

    def __init__(self, db):
        self.db = db
        super().__init__(Path(tempfile.gettempdir()) / "proctorlens-files")

    def read(self, sid: str, rel: str) -> bytes | None:
        r = self.db.one("SELECT data FROM files WHERE session_id=? AND path=?", sid, rel)
        return None if r is None else bytes(r["data"])

    def file(self, sid: str, rel: str) -> Path | None:
        return None

    def exists(self, sid: str, rel: str) -> bool:
        return self.db.one("SELECT 1 x FROM files WHERE session_id=? AND path=?", sid, rel) is not None

    def names(self, sid: str) -> set[str]:
        return {r["path"] for r in self.db.all("SELECT path FROM files WHERE session_id=?", sid)}

    def write(self, sid: str, rel: str, data: bytes) -> None:
        if self._p(sid, rel) is None:
            raise ValueError(f"bad path {rel!r}")
        self.db.run("INSERT INTO files (session_id, path, data) VALUES (?, ?, ?) ON CONFLICT (session_id, path) "
                    "DO UPDATE SET data = excluded.data", sid, rel, data)

    def delete(self, sid: str) -> None:
        self.db.run("DELETE FROM files WHERE session_id=?", sid)
        shutil.rmtree(self.dir(sid), ignore_errors=True)

    def local(self, sid: str) -> Path:
        d = self.dir(sid)
        shutil.rmtree(d, ignore_errors=True)
        for r in self.db.all("SELECT path, data FROM files WHERE session_id=?", sid):
            (p := d / r["path"]).parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes(r["data"]))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def keep(self, sid: str, folder: Path, rels: list[str]) -> None:
        for rel in rels:
            if (p := folder / rel).is_file():
                self.write(sid, rel, p.read_bytes())
