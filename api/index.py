"""Vercel entry point (docs/DECISIONS.md ADR-019): the API in browser mode, everything stored in Postgres (Neon).
Candidates' browsers run the models; this function only stores what they send. The page is the static build
(vercel.json); frontend/ is left out of this function, so the app mounts no frontend here."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proctorlens.cli import _DEFAULT_CFGS  # noqa: E402
from proctorlens.server.app import create_app  # noqa: E402

if not (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")):
    raise RuntimeError("DATABASE_URL is not set: connect a Neon database (Vercel > Storage); /tmp is wiped between requests")
# proctor login: PROCTORLENS_USER / PROCTORLENS_PASSWORD, else proctor / proctor (the owner's choice; guessable on a
# public site: set PROCTORLENS_PASSWORD in Vercel to change it)
app =create_app([p for p in _DEFAULT_CFGS if Path(p).is_file()], "/tmp/proctorlens", browser_mode=True)
