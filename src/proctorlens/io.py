"""File I/O: feature table (parquet, transparent csv fallback) and events.json with a provenance header."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proctorlens.core.config import Config, config_hash
from proctorlens.core.types import Event
from proctorlens.features import schema

SCHEMA_VERSION = 1  # events.json layout


def save_features(df: pd.DataFrame, path: str | Path) -> Path:
    """Write df to path. A .csv path, or a missing pyarrow, gives CSV (suffix becomes .csv). Returns the path written."""
    p = Path(path)
    if p.suffix != ".csv" and importlib.util.find_spec("pyarrow") is None:
        p = p.with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return p


def load_features(path: str | Path) -> pd.DataFrame:
    """Inverse of save_features; a missing .parquet falls back to its .csv sibling."""
    p = Path(path)
    if not p.exists() and p.suffix != ".csv":
        p = p.with_suffix(".csv")
    if p.suffix != ".csv":
        return pd.read_parquet(p)
    # keep_default_na=False: zone "none" must stay a string; only empty cells are NaN
    df = pd.read_csv(p, dtype={c: str for c in schema.STRING_COLUMNS}, keep_default_na=False, na_values=[""])
    for c in schema.STRING_COLUMNS & set(df.columns):
        df[c] = df[c].fillna("")
    return df


def _clean(o: Any) -> Any:
    """JSON-safe: numpy -> python, non-finite float -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    return None if isinstance(o, float) and not math.isfinite(o) else o


def save_events(events: list[Event], path: str | Path, header: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"header": header, "events": [e.to_dict() for e in events]}
    p.write_text(json.dumps(_clean(doc), indent=1, allow_nan=False), encoding="utf-8")


def load_events(path: str | Path) -> tuple[dict, list[Event]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc["header"], [Event(**e) for e in doc["events"]]


def _sha(p: str | Path) -> str | None:
    """Short sha256 of a file, or of all files under a directory; None if missing."""
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    for f in [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(f.name.encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:12]


def make_header(cfg: Config, calib: Any = None) -> dict:
    """Provenance header for events.json (SPEC 8.3). calib = Calibration | None."""
    m = cfg.models
    paths = {"landmarker": m.landmarker, "detector": m.detector, "identity": m.identity,
             **{f"scorer.{k}": v for k, v in cfg.scorer.models.items()}}
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": schema.SCHEMA_VERSION,
        "config_hash": config_hash(cfg),
        "provider": cfg.scorer.provider,
        "model_hashes": {k: _sha(v) for k, v in paths.items()},
        "thresholds": {k: dataclasses.asdict(v) for k, v in cfg.events.items()},
        "policy": dataclasses.asdict(cfg.policy),
        "calibration": None if calib is None else
        {"mode": calib.mode, "error": calib.error, "accepted": calib.accepted, "tries": calib.tries},
    }
