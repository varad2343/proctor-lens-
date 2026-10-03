"""Detector fine-tune + evaluation (spec 5.4). Thin wrapper over Ultralytics (imported lazily; AGPL-3.0 - see README).

python -m ml.training.train_detector train --config configs/training/detector.yaml
python -m ml.training.train_detector val --weights yolo11n.pt --data data/yolo_coco/dataset.yaml --split test
Pretrained-vs-fine-tuned study: `val` the fine-tuned weights on the normal dataset and the COCO weights on the one made
with `ml.data.to_yolo_format --coco-ids` (same images/splits, COCO class ids), then compare metrics.json files.
Verify the Ultralytics attribute names below against your installed version (spec 19).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from ml.training.train_temporal_gbm import _git_hash, sha256_file


def evaluate(weights: str, data: str, split: str = "val", conf: float = 0.30, imgsz: int = 640) -> dict:
    """mAP50 / mAP50-95 (Ultralytics default low conf) + per-class AP50; precision/recall read off the PR curves at the
    deployed threshold `conf` (None if this Ultralytics version lacks the curves). Classes without labels are skipped."""
    from ultralytics import YOLO

    m = YOLO(weights).val(data=data, split=split, imgsz=imgsz, verbose=False)
    b = m.box
    out = dict(weights=str(weights), data=data, split=split, conf=conf, map50=float(b.map50), map50_95=float(b.map),
               per_class_ap50={m.names[int(c)]: float(a) for c, a in zip(b.ap_class_index, b.ap50)},
               precision=None, recall=None)
    if all(hasattr(b, k) for k in ("px", "p_curve", "r_curve")):
        i = int(np.searchsorted(np.asarray(b.px), conf))
        out["precision"] = float(np.asarray(b.p_curve)[:, i].mean())
        out["recall"] = float(np.asarray(b.r_curve)[:, i].mean())
    return out


def onnx_latency_ms(path: str | Path, imgsz: int = 640, n: int = 30) -> float:
    """Mean CPU latency of one 1x3xHxW forward pass (onnxruntime)."""
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {s.get_inputs()[0].name: np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)}
    s.run(None, feed)  # warm-up
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, feed)
    return (time.perf_counter() - t) / n * 1000


def train(cfg: dict) -> dict:
    """Fine-tune from cfg['model'], then val-split metrics, ONNX export + CPU latency, all under out_dir/name/version."""
    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    model.train(data=cfg["data"], **cfg["train"])
    best, imgsz = Path(model.trainer.best), cfg["train"].get("imgsz", 640)
    out = Path(cfg["out_dir"]) / cfg["name"] / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out / "best.pt")
    metrics = evaluate(str(best), cfg["data"], "val", cfg["conf"], imgsz)  # test split: evaluated once, by hand, at the end
    shutil.copy2(YOLO(str(best)).export(format="onnx", imgsz=imgsz, simplify=True), out / "best.onnx")
    metrics |= dict(cpu_latency_ms=onnx_latency_ms(out / "best.onnx", imgsz), git=_git_hash(),
                    data_yaml_sha256=sha256_file(cfg["data"]), seed=cfg["train"].get("seed"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/training/detector.yaml")
    v = sub.add_parser("val")
    v.add_argument("--weights", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--split", default="val", choices=["val", "test"])
    v.add_argument("--conf", type=float, default=0.30)
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--out", help="write metrics JSON here")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        m = train(yaml.safe_load(Path(a.config).read_text(encoding="utf-8")))
    else:
        m = evaluate(a.weights, a.data, a.split, a.conf, a.imgsz)
        if a.out:
            Path(a.out).write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
