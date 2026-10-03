"""Build / extend data/manifest.csv: recording, participant, conditions, split, consent_version (tracked in git; no media).

python -m ml.data.build_manifest --recordings data/recordings --manifest data/manifest.csv --consent-version v1

Per recording: participant / conditions / consent_version come from <id>.meta.json (tools/collect/record.py), else
participant = filename prefix before the first '_'. Splits are assigned per PARTICIPANT (~60/20/20, seeded), so every
split is participant-disjoint, and they are stable: participants already in the manifest keep their split, existing rows
are kept verbatim (also when the video is not on this machine), new recordings are appended.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from ml.data.labels import MANIFEST_COLUMNS, VIDEO_EXT


def assign_splits(participants, fixed: dict[str, str] | None = None, frac=(0.6, 0.2, 0.2), seed: int = 0) -> dict[str, str]:
    """participant -> train|val|test. Already-placed participants (`fixed`) stay; new ones (seeded shuffle) first fill the
    test then val quota (round(frac * N), >= 1 each once N >= 3), the rest go to train."""
    out, ps = dict(fixed or {}), set(participants)
    new = sorted(ps - set(out))
    random.Random(seed).shuffle(new)
    n = len(ps | set(out))
    quota = {"test": round(frac[2] * n), "val": round(frac[1] * n)}
    if n >= 3:
        quota = {k: max(1, v) for k, v in quota.items()}
    have = Counter(out.values())
    for p in new:
        s = next((k for k in ("test", "val") if have[k] < quota[k]), "train")
        out[p] = s
        have[s] += 1
    return out


def build(recordings: str, manifest: str, consent_version: str | None = None, seed: int = 0) -> pd.DataFrame:
    old = (pd.read_csv(manifest, dtype=str, keep_default_na=False) if Path(manifest).exists()
           else pd.DataFrame(columns=MANIFEST_COLUMNS))
    if (old.groupby("participant")["split"].nunique() > 1).any():
        raise ValueError(f"{manifest}: a participant appears in more than one split (leakage)")
    rows, have = [], set(old["recording"])
    for v in sorted(p for p in Path(recordings).iterdir() if p.suffix.lower() in VIDEO_EXT):
        if v.stem in have:
            continue
        mp = v.with_name(v.stem + ".meta.json")
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        if not (cv := meta.get("consent_version") or consent_version):
            raise ValueError(f"{v.name}: no consent_version (meta.json or --consent-version); record consent first")
        rows.append(dict(recording=v.stem, participant=meta.get("participant") or v.stem.split("_")[0],
                         conditions=meta.get("conditions", ""), split="", consent_version=cv))
    df = pd.concat([old, pd.DataFrame(rows, columns=MANIFEST_COLUMNS)], ignore_index=True)
    fixed = {p: s for p, s in zip(old["participant"], old["split"]) if s}
    df["split"] = df["participant"].map(assign_splits(df["participant"], fixed, seed=seed))
    return df.sort_values("recording").reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--consent-version", help="fallback for recordings without a meta.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    df = build(a.recordings, a.manifest, a.consent_version, a.seed)
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.manifest, index=False)
    per = df.drop_duplicates("participant")["split"].value_counts().to_dict()
    print(f"{len(df)} recordings, {df['participant'].nunique()} participants; participants per split: {per}")


if __name__ == "__main__":
    main()
