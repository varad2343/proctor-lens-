"""Robustness stress tests (spec 12.4): synthetic perturbations applied to the video before the pipeline.
Perturbations are pure frame->frame functions (uint8 BGR in/out).

python -m ml.evaluation.robustness --videos data/recordings --results data/processed --split val --out reports/robustness
`--results` holds the clean replay outputs (the reference); each perturbed video is replayed with the same config.
"""
from __future__ import annotations

import argparse
import math
import tempfile
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from ml.evaluation.matching import iv, match_events
from proctorlens.core.types import MONITORING_DEGRADED


def gamma(frame: np.ndarray, g: float) -> np.ndarray:
    """g > 1 darkens (dim room), g < 1 brightens."""
    return cv2.LUT(frame, (255 * (np.arange(256) / 255) ** g).astype(np.uint8))


def brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1], cv2.IMREAD_COLOR)


def downscale(frame: np.ndarray, height: int) -> np.ndarray:
    """Resize to `height` rows keeping aspect (never upscales)."""
    h, w = frame.shape[:2]
    return frame.copy() if h <= height else cv2.resize(frame, (round(w * height / h), height), interpolation=cv2.INTER_AREA)


def noise(frame: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Additive Gaussian noise, deterministic: seed defaults to a hash of the frame content (varies per frame)."""
    seed = int(frame[::8, ::8].sum()) if seed is None else seed
    return np.clip(frame + np.random.default_rng(seed).normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)


def flip(frame: np.ndarray) -> np.ndarray:
    """Horizontal flip (sanity check: zones left/right swap, everything else unchanged)."""
    return cv2.flip(frame, 1)


def crop(frame: np.ndarray, frac: float = 0.8, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Keep a frac-size window, shifted by dx,dy in [-1,1] of the slack, resized back: zoomed / off-centre framing."""
    h, w = frame.shape[:2]
    ch, cw = round(h * frac), round(w * frac)
    y0, x0 = round((h - ch) / 2 * (1 + dy)), round((w - cw) / 2 * (1 + dx))
    return cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w, h))


PERTURBATIONS = {
    "reencode": lambda f: f.copy(),  # control: the cost of the re-encode itself
    **{f"gamma_{g}": partial(gamma, g=g) for g in (2.0, 3.0)},
    **{f"bright_{b}": partial(brightness, factor=b) for b in (0.4, 0.15)},
    **{f"blur_{s}": partial(blur, sigma=s) for s in (2.0, 5.0)},
    **{f"jpeg_{q}": partial(jpeg, quality=q) for q in (90, 60, 30)},
    **{f"down_{h}p": partial(downscale, height=h) for h in (480, 360)},
    **{f"noise_{s}": partial(noise, sigma=s) for s in (10.0, 25.0)},
    "flip": flip,
    "crop_0.8": partial(crop, frac=0.8),
    "crop_0.8_offcentre": partial(crop, frac=0.8, dx=1.0),
}


def perturb_video(src: str | Path, dst: str | Path, fn) -> int:
    """Write fn(frame) of every frame of `src` to `dst` (.avi -> MJPG, else mp4v); returns frames written.
    # ponytail: constant-fps rewrite at the source's nominal fps; variable-frame-rate timing is not preserved."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if str(dst).endswith(".avi") else "mp4v"))
    out, n = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            g = fn(frame)
            if out is None:
                out = cv2.VideoWriter(str(dst), fourcc, fps, (g.shape[1], g.shape[0]))
            out.write(g)
            n += 1
    finally:
        cap.release()
        if out is not None:
            out.release()
    return n


def _overlaps(e, deg: list) -> bool:
    _, s, t = iv(e)
    return any(s < de and t > ds for _, ds, de in map(iv, deg))


def degraded_switch_rate(clean, perturbed, **match_kw) -> dict:
    """Did the system say it was blind, or change silently? Compares a perturbed run with the clean run:
    `added` = non-degraded events only the perturbed run has, `removed` = clean events it lost. The rate is the
    fraction of these changes overlapping a MONITORING_DEGRADED event of the perturbed run (NaN if nothing changed)."""
    deg = [e for e in perturbed if iv(e)[0] == MONITORING_DEGRADED]
    c = [e for e in clean if iv(e)[0] != MONITORING_DEGRADED]
    p = [e for e in perturbed if iv(e)[0] != MONITORING_DEGRADED]
    _, fp, fn = match_events(p, c, **match_kw)
    changed = [p[i] for i in fp] + [c[i] for i in fn]
    k = sum(_overlaps(e, deg) for e in changed)
    return dict(n_changed=len(changed), n_covered=k, rate=k / len(changed) if changed else math.nan)


def main(argv=None) -> None:
    from ml.evaluation.ablations import find_video, load_calib, run_pipeline
    from ml.evaluation.run_eval import evaluate, load_split, md_table, stack
    from proctorlens.core.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", nargs="*", default=["configs/pipeline.yaml", "configs/policy.yaml"])
    ap.add_argument("--only", help="comma-separated perturbation names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    cfg = load_config(*a.config)
    recs = load_split(a.manifest, a.results, a.labels, a.split)
    perts = {k: PERTURBATIONS[k] for k in a.only.split(",")} if a.only else PERTURBATIONS
    results, sw = {"clean": evaluate(recs, a.boot)}, {"clean": (0, 0)}
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in perts.items():
            done, n, k = [], 0, 0
            for r in recs:
                dst = Path(tmp) / f"{r['recording']}.mp4"
                perturb_video(find_video(a.videos, r["recording"]), dst, fn)
                pred, total = run_pipeline(dst, cfg, load_calib(a.videos, r["recording"]))
                d = degraded_switch_rate(r["pred"], pred)
                n, k = n + d["n_changed"], k + d["n_covered"]
                done.append({**r, "pred": pred, "total_ms": total})
                dst.unlink()
            results[name], sw[name] = evaluate(done, a.boot), (n, k)
            print("done", name, flush=True)
    types, summ = stack(results)
    clean_f1 = types[types["variant"] == "clean"].set_index("type")["f1"]
    types["d_f1"] = types["f1"] - types["type"].map(clean_f1)
    summ["n_changed"] = [sw[v][0] for v in summ["variant"]]
    summ["switch_rate"] = [sw[v][1] / sw[v][0] if sw[v][0] else math.nan for v in summ["variant"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types.to_csv(out / "robustness_types.csv", index=False)
    summ.to_csv(out / "robustness_summary.csv", index=False)
    cols = ["variant", "fa_per_hour", "flagged_fraction", "n_changed", "switch_rate"]
    (out / "robustness.md").write_text(
        f"# Robustness ({a.split})\n\nswitch_rate = fraction of perturbation-induced event changes that coincide with "
        f"MONITORING_DEGRADED.\n\n{md_table(summ[cols])}\n\n"
        f"{md_table(types[['variant', 'type', 'tp', 'fp', 'fn', 'f1', 'd_f1']])}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
