"""Identity threshold calibration (spec 5.5): genuine vs impostor cosine similarity -> EER, TAR@FAR=1%, tau.

python -m ml.training.calibrate_identity --videos data/recordings --out data/models/identity [--splits train val]

Embeddings stay in memory and are discarded; only similarity scores, metrics and tau are written. tau is the
midpoint below the smallest score whose impostor acceptance (FAR) is <= the target, i.e. the strictest-on-genuine
choice that still keeps FAR at 1%: sim < tau on `identity.consecutive` high-quality checks = IDENTITY_MISMATCH.
# ponytail: one pooled tau; no per-condition (glasses / lighting) breakdown yet - add when N allows.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _unit(a) -> np.ndarray:
    a = np.asarray(a, float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def similarity_sets(emb: dict[str, np.ndarray], n_enroll: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """{participant: (n, d) embeddings in time order} -> (genuine, impostor) cosine similarities.

    Mirrors runtime: each participant's enrollment = mean of their first n_enroll embeddings; genuine = their later
    embeddings vs own enrollment (other recordings/conditions included), impostor = everyone else's embeddings vs it."""
    U = {p: _unit(e) for p, e in emb.items() if len(e) > n_enroll}
    gen, imp = [], []
    for p, u in U.items():
        enroll = _unit(u[:n_enroll].mean(0))
        for q, v in U.items():
            (gen if q == p else imp).append((v[n_enroll:] if q == p else v) @ enroll)
    if not gen or not imp:
        raise ValueError(f"need >= 2 participants with > {n_enroll} embeddings each")
    return np.concatenate(gen), np.concatenate(imp)


def roc(genuine, impostor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds ascending incl. +inf, TAR = P(genuine >= t), FAR = P(impostor >= t))."""
    g, i = np.sort(genuine), np.sort(impostor)
    thr = np.unique(np.concatenate([g, i, [np.inf]]))
    return thr, 1 - np.searchsorted(g, thr, "left") / len(g), 1 - np.searchsorted(i, thr, "left") / len(i)


def calibrate(genuine, impostor, far: float = 0.01) -> dict:
    """EER, TAR at FAR<=far, and tau (see module docstring)."""
    thr, tar, fa = roc(genuine, impostor)
    k = int(np.argmin(np.abs(fa - (1 - tar))))  # FRR = 1 - TAR
    j = int(np.flatnonzero(fa <= far)[0])  # FAR is non-increasing in t; thr[-1] = inf guarantees a hit
    tau = float(thr[j - 1]) if np.isinf(thr[j]) else float((thr[max(j - 1, 0)] + thr[j]) / 2)
    return dict(eer=float((fa[k] + 1 - tar[k]) / 2), eer_threshold=float(thr[k]), far=far, tar_at_far=float(tar[j]),
                tau=tau, n_genuine=len(genuine), n_impostor=len(impostor))


def collect_embeddings(cfg, manifest: str, videos: str, labels: str, splits: list[str], every_s: float):
    """Embed quality-gated frontal primary faces every `every_s` of video time (labelled IDENTITY_MISMATCH frames skipped)."""
    import cv2

    from ml.data.labels import find_video, frame_labels, load_labels, load_manifest
    from proctorlens.core.types import IDENTITY_MISMATCH
    from proctorlens.perception.identity import IdentityChecker
    from proctorlens.perception.landmarks import Landmarker
    from proctorlens.perception.quality import assess

    idc, emb = IdentityChecker(cfg.models.identity), defaultdict(list)
    for r in load_manifest(manifest, splits).sort_values("recording").itertuples():
        lp = Path(labels) / f"{r.recording}.csv"
        lab = load_labels(lp) if lp.exists() else None
        lm = Landmarker(cfg.models.landmarker, num_faces=1)  # new per video: VIDEO mode needs monotonic timestamps
        cap = cv2.VideoCapture(str(find_video(videos, r.recording)))
        fps, nxt, i = cap.get(cv2.CAP_PROP_FPS) or 30.0, 0.0, 0
        while cap.grab():
            t, i = cap.get(cv2.CAP_PROP_POS_MSEC) or i * 1000 / fps, i + 1
            if t < nxt:
                continue
            nxt = t + every_s * 1000
            frame = cap.retrieve()[1]
            if lab is not None and frame_labels(lab, np.array([t]), [IDENTITY_MISMATCH])[0]:
                continue
            faces = lm.process(frame, int(t))
            if not faces or abs(faces[0].yaw) >= cfg.identity.max_abs_yaw:
                continue
            if assess(frame, faces[0].bbox, cfg.quality).quality < cfg.quality.min_quality:
                continue
            if (e := idc.embed(frame, faces[0])) is not None:
                emb[r.participant].append(e)
        cap.release()
    return {p: np.stack(v) for p, v in emb.items()}


def main(argv=None) -> None:
    import yaml

    from proctorlens.core.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="never `test`: tau is tuned here")
    ap.add_argument("--config", nargs="*", default=["configs/pipeline.yaml", "configs/policy.yaml"])
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--n-enroll", type=int, default=5)
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(*a.config)
    g, i = similarity_sets(collect_embeddings(cfg, a.manifest, a.videos, a.labels, a.splits, a.every_s), a.n_enroll)
    res = calibrate(g, i, a.far)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.savez(out / "similarities.npz", genuine=g, impostor=i)
    (out / "identity_tau.yaml").write_text(yaml.safe_dump({"identity": {"tau": round(res["tau"], 4)}}), encoding="utf-8")
    print(json.dumps(res, indent=2), f"\nadd {out / 'identity_tau.yaml'} as a --config overlay")


if __name__ == "__main__":
    main()
