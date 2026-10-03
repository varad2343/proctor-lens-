"""Annotation export (Label Studio JSON / CVAT-for-images XML) -> YOLO txt + dataset.yaml, participant-disjoint splits.

python -m ml.data.to_yolo_format --ann export.json --images data/frames --manifest data/manifest.csv --out data/yolo
python -m ml.data.to_yolo_format ... --out data/yolo_coco --coco-ids   # same images/splits with COCO class ids, to
                                                                        # evaluate the pretrained COCO baseline

Image names must be <recording>_<t_ms>.jpg (ml.data.extract_frames); the split is that recording's manifest split, so a
participant never spans splits. Classes: phone, notes, person (synonyms below). Other labels (remote, wallet, ...) are
left as background = hard negatives. An image present in the export without boxes counts as annotated-empty, so export
finished tasks only. # ponytail: Label Studio uses the first annotation of a task (no consensus merge).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from proctorlens.core.types import Box

NAMES = ["phone", "notes", "person"]
SYN = {"phone": "phone", "cell phone": "phone", "mobile": "phone", "notes": "notes", "book": "notes",
       "notebook": "notes", "paper": "notes", "person": "person"}
COCO_ID = {"phone": 67, "notes": 73, "person": 0}  # cell phone, book, person
COCO_NAMES = {i: f"coco{i}" for i in range(80)} | {0: "person", 67: "cell phone", 73: "book"}

Ann = dict[str, list[tuple[str, Box]]]  # image file name -> [(label, normalised x0,y0,x1,y1)]


def parse_labelstudio(path: str | Path) -> Ann:
    out: Ann = {}
    for task in json.loads(Path(path).read_text(encoding="utf-8")):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        name = re.sub(r"^[0-9a-f]{8}-", "", Path(task["data"]["image"]).name)  # LS prefixes uploads with a hash
        out[name] = [(r["value"]["rectanglelabels"][0],
                      (r["value"]["x"] / 100, r["value"]["y"] / 100,
                       (r["value"]["x"] + r["value"]["width"]) / 100, (r["value"]["y"] + r["value"]["height"]) / 100))
                     for r in anns[0]["result"] if r.get("type") == "rectanglelabels"]
    return out


def parse_cvat(path: str | Path) -> Ann:
    out: Ann = {}
    for im in ET.parse(path).getroot().iter("image"):
        w, h = float(im.get("width")), float(im.get("height"))
        f = lambda b, k, s: float(b.get(k)) / s  # noqa: E731
        out[Path(im.get("name")).name] = [
            (b.get("label"), (f(b, "xtl", w), f(b, "ytl", h), f(b, "xbr", w), f(b, "ybr", h))) for b in im.iter("box")]
    return out


def yolo_lines(boxes: list[tuple[str, Box]], ids: dict[str, int]) -> list[str]:
    """`class cx cy w h` (normalised, clamped to the image); unknown labels and degenerate boxes are dropped."""
    out = []
    for label, box in boxes:
        c = SYN.get(label.strip().lower())
        x0, y0, x1, y1 = (min(max(v, 0.0), 1.0) for v in box)
        if c is not None and x1 > x0 and y1 > y0:
            out.append(f"{ids[c]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return out


def convert(ann: Ann, images: str | Path, manifest: pd.DataFrame, out: str | Path, coco_ids: bool = False) -> Counter:
    """Copy images + write labels into out/{images,labels}/<split>/ and out/dataset.yaml; returns images per split."""
    if (manifest.groupby("participant")["split"].nunique() > 1).any():
        raise ValueError("manifest: a participant appears in more than one split (leakage)")
    split_of, out = manifest.set_index("recording")["split"], Path(out)
    ids = COCO_ID if coco_ids else {n: i for i, n in enumerate(NAMES)}
    n: Counter = Counter()
    for name, boxes in ann.items():
        rec = Path(name).stem.rsplit("_", 1)[0]
        if rec not in split_of.index:
            raise KeyError(f"{name}: recording {rec!r} is not in the manifest")
        s = split_of[rec]
        for d in ("images", "labels"):
            (out / d / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(images) / name, out / "images" / s / name)
        lines = yolo_lines(boxes, ids)
        (out / "labels" / s / f"{Path(name).stem}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
        n[s] += 1
    spec = {"path": str(out.resolve()), **{k: f"images/{k}" for k in ("train", "val", "test") if n[k]},
            "names": COCO_NAMES if coco_ids else dict(enumerate(NAMES))}
    (out / "dataset.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    return n


def main(argv=None) -> None:
    from ml.data.labels import load_manifest

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="Label Studio .json or CVAT-for-images .xml export")
    ap.add_argument("--images", required=True)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coco-ids", action="store_true")
    a = ap.parse_args(argv)
    ann = parse_labelstudio(a.ann) if a.ann.endswith(".json") else parse_cvat(a.ann)
    print(dict(convert(ann, a.images, load_manifest(a.manifest), a.out, a.coco_ids)))


if __name__ == "__main__":
    main()
