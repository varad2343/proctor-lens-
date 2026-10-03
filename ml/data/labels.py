"""Loaders: label CSVs (type,start_ms,end_ms,source,annotator_id), manifest.csv, per-recording result files.

Benign behaviours are labelled as `benign_*` types; honest time = `natural`/`nuisance` blocks + benign_*.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from proctorlens.core.types import EVENT_TYPES

COLUMNS = ["type", "start_ms", "end_ms", "source", "annotator_id"]
HONEST_BLOCKS = ("natural", "nuisance")
MANIFEST_COLUMNS = ["recording", "participant", "conditions", "split", "consent_version"]


def load_labels(csv: str | Path) -> pd.DataFrame:
    """Read a label CSV -> DataFrame[COLUMNS]; extra columns (e.g. cue `prompt`) are dropped."""
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    if miss := {"type", "start_ms", "end_ms"} - set(df.columns):
        raise ValueError(f"{csv}: missing columns {sorted(miss)}")
    for c in ("source", "annotator_id"):
        if c not in df:
            df[c] = ""
    df = df[COLUMNS].copy()
    df["type"] = df["type"].str.strip()
    df[["start_ms", "end_ms"]] = df[["start_ms", "end_ms"]].astype("int64")
    if (df["end_ms"] < df["start_ms"]).any():
        raise ValueError(f"{csv}: end_ms < start_ms")
    return df


def frame_labels(labels: pd.DataFrame, t_ms: np.ndarray, types: list[str]) -> np.ndarray:
    """bool[len(t_ms)]: True where t falls in [start_ms, end_ms) of any label whose type is in `types`."""
    t = np.asarray(t_ms)
    out = np.zeros(t.shape, bool)
    for s, e in labels.loc[labels["type"].isin(types), ["start_ms", "end_ms"]].to_numpy():
        out |= (t >= s) & (t < e)
    return out


def gt_events(labels: pd.DataFrame) -> list[dict]:
    """Labels that are ground-truth events (core.types.EVENT_TYPES); drops natural/nuisance/benign_*/calibration."""
    return labels[labels["type"].isin(EVENT_TYPES)].to_dict("records")


def merge_intervals(iv) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted((int(s), int(e)) for s, e in iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def honest_intervals(labels: pd.DataFrame) -> list[tuple[int, int]]:
    """Merged [start,end] of natural + nuisance blocks and benign_* labels (false-alarm denominator)."""
    m = labels["type"].isin(HONEST_BLOCKS) | labels["type"].str.startswith("benign_")
    return merge_intervals(labels.loc[m, ["start_ms", "end_ms"]].to_numpy().tolist())


def load_manifest(csv: str | Path, split: str | list[str] | None = None) -> pd.DataFrame:
    """manifest.csv -> DataFrame[MANIFEST_COLUMNS], optionally filtered to one or more splits."""
    m = pd.read_csv(csv, dtype=str, keep_default_na=False)
    if miss := set(MANIFEST_COLUMNS) - set(m.columns):
        raise ValueError(f"{csv}: missing columns {sorted(miss)}")
    if split is not None:
        m = m[m["split"].isin([split] if isinstance(split, str) else split)]
    return m.reset_index(drop=True)


VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov")


def find_video(root: str | Path, rec: str) -> Path:
    for ext in VIDEO_EXT:
        if (p := Path(root) / f"{rec}{ext}").exists():
            return p
    raise FileNotFoundError(f"no video {rec}{{{','.join(VIDEO_EXT)}}} in {root}")


def find_file(root: str | Path, rec: str, name: str) -> Path:
    """Result file of one recording: <root>/<rec>/<name> or <root>/<rec>.<name>."""
    cand = [Path(root) / rec / name, Path(root) / f"{rec}.{name}"]
    for c in cand:
        if c.exists():
            return c
    raise FileNotFoundError(f"{name} for {rec!r}: tried {[str(c) for c in cand]}")
