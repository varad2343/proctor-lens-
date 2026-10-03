"""Ablations (spec 12.3 #2 calibration, #4 quality gating, #5 state-machine settings, #6 processing rate):
re-run the labelled recordings through the real Pipeline once per variant and score each with run_eval.evaluate.
Rules-vs-learned and COCO-vs-fine-tuned (#1, #3) are the same eval pointed at another --config (scorer.provider /
models.detector), i.e. `--variants my.yaml` or a separate replay + run_eval.

python -m ml.evaluation.ablations --videos data/recordings --split val --out reports/ablations [--only fps5,base]
Variants YAML: {name: {cfg: {<Config overrides>}, headpose_only: bool, fps: float}}; replaces the built-ins.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from pathlib import Path

import pandas as pd
import yaml

from ml.data.labels import find_video
from ml.evaluation.run_eval import evaluate, load_split, md_table, stack
from proctorlens.core.config import Config, _merge, from_dict, load_config, validate
from proctorlens.core.types import SCORE_KEYS

VARIANTS: dict[str, dict] = {
    "base": {},
    **{f"t_on_{s}s": {"cfg": {"events": {"off_screen": {"t_on_s": float(s), "min_dur_s": float(s)}}}} for s in (2, 3, 6)},
    **{f"off_thr_{v}": {"cfg": {"events": {"off_screen": {"off_thr": v}}}} for v in (0.2, 0.45)},
    "head_pose_only": {"headpose_only": True},  # calibration regression dropped, geometric head-turn term only
    "no_quality_gating": {"cfg": {"quality": {"min_quality": 0.0},  # everything 'reliable', no degraded machine
                                  "policy": {"active": [k for k in SCORE_KEYS if k != "degraded"]}}},
    "fps5": {"fps": 5},  # process ~5 frames/s instead of every frame
}


def make_cfg(base: Config, overrides: dict) -> Config:
    """Deep-merge `overrides` (nested dict) over `base`; validated, `base` untouched."""
    cfg = from_dict(Config, _merge(dataclasses.asdict(base), copy.deepcopy(overrides)))
    validate(cfg)
    return cfg


def load_calib(root: str | Path, rec: str):
    """<root>/<rec>.calib.json -> Calibration, or None when the recording has none."""
    from proctorlens.perception.gaze import Calibration

    p = Path(root) / f"{rec}.calib.json"
    return Calibration.load(p) if p.exists() else None


def run_pipeline(video: str | Path, cfg: Config, calib=None, fps: float | None = None) -> tuple[list, float]:
    """Replay one video through a fresh Pipeline -> (events, processed duration ms).
    fps = target processed frames/s (frames skipped, real timestamps kept)."""
    import cv2

    from proctorlens.perception import Perceiver
    from proctorlens.pipeline.runner import Pipeline

    cap = cv2.VideoCapture(str(video))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src / fps)) if fps else 1
    pipe, i, last, t0 = Pipeline(cfg, Perceiver(cfg), calib), 0, -1, None  # fresh Perceiver: VIDEO mode needs monotonic t
    try:
        while cap.grab():
            if i % step == 0:
                t = round(cap.get(cv2.CAP_PROP_POS_MSEC))
                if t <= last:  # container gave no usable timestamp (same rule as runner.run_video)
                    t = max(last + 1, round(i * 1000 / src))
                t0 = t if t0 is None else t0
                pipe.process(cap.retrieve()[1], t)
                last = t
            i += 1
    finally:
        cap.release()
    return pipe.finish(), float(max(last - (t0 or 0), 0))


def run_variant(recs: list[dict], videos: str, cfg: Config, headpose_only: bool = False, fps: float | None = None):
    """evaluate()-ready recs with `pred` / `total_ms` replaced by this variant's run."""
    out = []
    for r in recs:
        calib = load_calib(videos, r["recording"])
        if headpose_only and calib is not None:
            calib = dataclasses.replace(calib, model=None, mode="head_pose_only", accepted=False)
        pred, total = run_pipeline(find_video(videos, r["recording"]), cfg, calib, fps)
        out.append({**r, "pred": pred, "total_ms": total})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", nargs="*", default=["configs/pipeline.yaml", "configs/policy.yaml"])
    ap.add_argument("--variants", help="YAML replacing the built-in variants")
    ap.add_argument("--only", help="comma-separated variant names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    base = load_config(*a.config)
    variants = yaml.safe_load(Path(a.variants).read_text(encoding="utf-8")) if a.variants else VARIANTS
    if a.only:
        variants = {k: variants[k] for k in a.only.split(",")}
    recs = load_split(a.manifest, None, a.labels, a.split)
    results = {}
    for name, v in variants.items():
        done = run_variant(recs, a.videos, make_cfg(base, v.get("cfg", {})), v.get("headpose_only", False), v.get("fps"))
        results[name] = evaluate(done, a.boot)
        print("done", name, flush=True)
    types, summ = stack(results)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    types.to_csv(out / "ablations_types.csv", index=False)
    summ.to_csv(out / "ablations_summary.csv", index=False)
    cols = ["variant", "n_participants", "fa_per_hour", "fa_per_hour_lo", "fa_per_hour_hi", "flagged_fraction"]
    tcols = ["variant", "type", "tp", "fp", "fn", "precision", "recall", "f1", "f1_lo", "f1_hi"]
    (out / "ablations.md").write_text(f"# Ablations ({a.split})\n\n{md_table(summ[cols])}\n\n{md_table(types[tcols])}\n",
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
