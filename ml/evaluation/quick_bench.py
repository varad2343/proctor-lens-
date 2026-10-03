"""Quick A/B of the front-end accuracy aids on real still images. No video, no webcam, no windows.

python -m ml.evaluation.quick_bench [--image PATH ...] [--out report.md] [--config PATH ...]

Default images (shipped with ultralytics): the zidane crop (the one face MediaPipe finds in that frame), the full
zidane frame (two small faces) and bus.jpg. Each image is perturbed with the functions of ml/evaluation/robustness.py
and perceived under two configs: base = lowlight_enhance and recover_small_faces off, impr = both on.
The report is a table of observations per perturbation; it says nothing about people.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ml.evaluation.robustness import PERTURBATIONS, downscale, gamma
from ml.evaluation.run_eval import md_table

CONFIGS = {"base": dict(lowlight_enhance=False, recover_small_faces=False),
           "impr": dict(lowlight_enhance=True, recover_small_faces=True)}
_METRICS = {"face": "face rate", "dyaw": "|dyaw| deg", "eye": "eye_open", "persons": "persons"}  # record key -> header
_NO_IDENTITY = "__no_identity__"  # not a directory => the Perceiver skips identity (and its slow model load)
LEGEND = ("face rate = fraction of runs with >= 1 face. |dyaw| = mean |yaw - yaw of the unperturbed image| in degrees "
          "(same config; '-' when that image had no face). eye_open = mean of both eyes of the primary face. "
          "persons = mean person boxes from the detector. base = lowlight_enhance and recover_small_faces off; "
          "impr = both on.")


def perturbations() -> dict:
    """Name -> frame function, taken from ml/evaluation/robustness.py. Downscale is relative (x0.5, x0.25): the stock
    ones are absolute heights and do nothing to a 360 px crop. Gamma 4 and 5 are added: measured on the zidane crop,
    the landmarker still finds the face at gamma 4 (yaw 6 deg off) and loses it at gamma 5 (mean luma ~12)."""
    p = PERTURBATIONS
    rel = lambda r: lambda f: downscale(f, max(8, round(f.shape[0] * r)))  # noqa: E731
    return {"clean": p["reencode"], "gamma_2.0": p["gamma_2.0"], "gamma_3.0": p["gamma_3.0"],
            "gamma_4.0": partial(gamma, g=4.0), "gamma_5.0": partial(gamma, g=5.0),
            **{k: p[k] for k in ("blur_2.0", "blur_5.0", "jpeg_60", "jpeg_30", "noise_10.0", "noise_25.0")},
            "down_x0.5": rel(0.5), "down_x0.25": rel(0.25)}


def measure(p) -> dict:
    """One Perceived -> the numbers of the report (NaN where there is no face)."""
    f = p.faces[0] if p.faces else None
    return dict(face=float(f is not None), yaw=f.yaw if f else math.nan,
                eye=float(np.mean(f.eye_open)) if f else math.nan, persons=len(p.det.person_boxes))


def add_dyaw(recs: list[dict]) -> list[dict]:
    """|yaw - yaw of the same image, unperturbed, same config| into each record (NaN if either has no face)."""
    ref = {(r["image"], r["config"]): r["yaw"] for r in recs if r["pert"] == "clean"}
    for r in recs:
        r["dyaw"] = abs(r["yaw"] - ref.get((r["image"], r["config"]), math.nan))
    return recs


def tables(recs: list[dict], key: str = "pert") -> pd.DataFrame:
    """Mean of each metric per `key` value (first-seen order) and config, plus an ALL row over everything."""
    df = pd.DataFrame(recs + [{**r, key: "ALL"} for r in recs])
    g = df.groupby([key, "config"], sort=False)[list(_METRICS)].mean().unstack("config")
    t = pd.DataFrame({f"{label} {c}": g[(m, c)] for m, label in _METRICS.items() for c in CONFIGS})
    return t.reindex(df[key].unique()).rename_axis(key).reset_index()


def render(recs: list[dict]) -> str:
    """Markdown report: one table over all images, then one per image."""
    imgs = list(dict.fromkeys(r["image"] for r in recs))
    parts = ["# Quick perception bench", "", LEGEND, "", f"## All images ({', '.join(imgs)})", "",
             md_table(tables(recs))]
    for i in imgs:
        parts += ["", f"## {i}", "", md_table(tables([r for r in recs if r["image"] == i]))]
    return "\n".join(parts) + "\n"


def run_bench(cfg, images: dict, perts: dict) -> list[dict]:
    """Perceive every (image, perturbation) once per config. A fresh landmarker per run keeps VIDEO-mode tracking from
    carrying one image into the next; the detector is stateless and shared. Needs the real models."""
    from proctorlens.perception import Perceiver
    from proctorlens.perception.landmarks import Landmarker
    from proctorlens.perception.objects import ObjectDetector

    det = ObjectDetector(cfg.models.detector, cfg.models.detector_conf)
    cfgs = {}
    for name, flags in CONFIGS.items():
        cfgs[name] = c = copy.deepcopy(cfg)
        c.models.identity = _NO_IDENTITY
        for k, v in flags.items():
            setattr(c.perception, k, v)
    recs = []
    for iname, img in images.items():
        print(f"image {iname} {img.shape[1]}x{img.shape[0]}", file=sys.stderr, flush=True)
        for pname, fn in perts.items():
            x = fn(img)
            for cname, c in cfgs.items():
                p = Perceiver(c, landmarker=Landmarker(c.models.landmarker), detector=det).perceive(x, 0, 0)
                recs.append(dict(image=iname, pert=pname, config=cname, **measure(p)))
    return add_dyaw(recs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="extend", nargs="+", default=[], metavar="PATH",
                    help="image(s) to use instead of the ultralytics samples")
    ap.add_argument("--out", metavar="report.md", help="write the report here (default: print it)")
    ap.add_argument("--config", nargs="*", default=["configs/pipeline.yaml", "configs/policy.yaml"])
    a = ap.parse_args(argv)
    from proctorlens.cli import check_models, sample_images
    from proctorlens.core.config import load_config

    cfg = load_config(*a.config)
    imgs = {Path(p).stem: cv2.imread(p) for p in a.image} if a.image else sample_images()
    if not imgs or any(v is None for v in imgs.values()):
        print("no usable images: pass --image PATH (readable files), or install ultralytics for its samples", file=sys.stderr)
        return 1
    if missing := [d for s, _, d in check_models(cfg) if s == "FAIL"]:
        print("missing models:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    report = render(run_bench(cfg, imgs, perturbations()))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
