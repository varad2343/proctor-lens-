"""Writes ml/notebooks/train_detector_colab.ipynb: fine-tune the phone / notes / person detector on free public data
(COCO 2017 + Open Images V7, via FiftyOne) on a free Colab GPU, compare it with the stock COCO yolo11n on the same
validation images, and export the ONNX that ProctorLens loads. Colab, because torch cannot load on this PC (ADR-018).

python ml/notebooks/make_detector_colab.py      # regenerate the .ipynb after editing CELLS
"""
import json
from pathlib import Path

INTRO = """# ProctorLens detector fine-tune (free, open data)

Runtime -> Change runtime type -> **T4 GPU**, then Runtime -> Run all (about 1-1.5 h with the defaults).
At the end your browser downloads `proctorlens_detector.onnx` and `metrics.json`.

Put the ONNX in `data/models/` and set `models.detector: data/models/proctorlens_detector.onnx` in
`configs/pipeline.yaml`. Keep it only if `metrics.json` shows it beating the stock model.

**Data:**
- COCO 2017 (`cell phone`, `book`, `person`, plus negatives such as `remote` / `laptop`).
- Open Images V7 (`Mobile phone`, `Book`, and the `Person` / `Man` / `Woman` / `Boy` / `Girl` boxes on those images).

Both datasets' annotations are CC BY 4.0. The images belong to their Flickr authors (COCO: Flickr terms; Open Images:
listed as CC BY 2.0). That is fine for coursework, but credit both datasets. Neither dataset shows webcam exam scenes,
so this only measures public-image accuracy: check the model on your own sessions before trusting it."""

SETUP = """!pip -q install ultralytics fiftyone
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
print(torch.cuda.get_device_name(0))"""

PARAMS = """N_COCO, N_OI, N_NEG, N_VAL = 3000, 1500, 600, 400  # images per source; smaller = faster, larger = better
EPOCHS = 30                                   # early stopping (patience 10) usually ends it sooner
OUT = "/content/pl_data"                      # YOLO-format dataset written here
SPLITS = {"train": ("train", "train"), "val": ("validation", "validation")}  # ours -> (COCO split, Open Images split)"""

BUILD = """import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

NAMES = ["phone", "notes", "person"]  # ProctorLens classes, in this id order (src/proctorlens/perception/objects.py)
MAP = {"cell phone": "phone", "Mobile phone": "phone", "book": "notes", "Book": "notes", "person": "person",
       "Person": "person", "Man": "person", "Woman": "person", "Boy": "person", "Girl": "person"}


def load(zoo, split, classes, n):
    return foz.load_zoo_dataset(zoo, split=split, label_types=["detections"], classes=classes, max_samples=n,
                                shuffle=True, seed=0, num_workers=16,  # downloads are IO-bound; Colab has 2 CPUs
                                dataset_name=f"{zoo}-{split}-{len(classes)}-{n}")


def export(ds, split):
    # every phone / book / person box on an image is kept (dropping the people next to a phone would teach the
    # model that people are background); any other box is dropped, so negative images end up with no labels
    v = ds.map_labels("ground_truth", MAP).filter_labels("ground_truth", F("label").is_in(NAMES), only_matches=False)
    v.export(export_dir=OUT, dataset_type=fo.types.YOLOv5Dataset, label_field="ground_truth", classes=NAMES,
             split=split)


for ours, (coco_split, oi_split) in SPLITS.items():
    val = ours == "val"
    export(load("coco-2017", coco_split, ["cell phone", "book"], N_VAL if val else N_COCO), ours)
    export(load("open-images-v7", oi_split, ["Mobile phone", "Book"], N_VAL if val else N_OI), ours)
    if not val:  # hard negatives: desk objects that are not phones or notes
        export(load("coco-2017", coco_split, ["remote", "laptop", "keyboard", "mouse"], N_NEG), ours)
print(open(f"{OUT}/dataset.yaml").read())"""

TRAIN = """from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # COCO-pretrained start, like configs/training/detector.yaml
model.train(data=f"{OUT}/dataset.yaml", epochs=EPOCHS, imgsz=640, batch=32, patience=10, seed=0, deterministic=True,
            hsv_s=0.6, hsv_v=0.5, close_mosaic=10, project="/content/runs", name="ft", exist_ok=True)
BEST = str(model.trainer.best)"""

COMPARE = """import json
import shutil
from pathlib import Path

# the stock model is scored on the same validation images: the same labels, renumbered to COCO class ids
coco_dir, coco_id = Path("/content/pl_data_coco"), {0: 67, 1: 73, 2: 0}  # phone -> cell phone, notes -> book
(coco_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
# a copy, not a symlink: Ultralytics resolves dataset paths, so a symlink would read OUT's labels (ids 0/1/2)
shutil.copytree(f"{OUT}/images/val", coco_dir / "images" / "val", dirs_exist_ok=True)
for f in Path(OUT, "labels", "val").glob("*.txt"):
    rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
    (coco_dir / "labels" / "val" / f.name).write_text("".join(f"{coco_id[int(r[0])]} {' '.join(r[1:])}\\n" for r in rows))
names = {i: f"coco{i}" for i in range(80)} | {0: "person", 67: "cell phone", 73: "book"}
(coco_dir / "dataset.yaml").write_text(json.dumps({"path": str(coco_dir), "train": "images/val", "val": "images/val",
                                                   "names": names}))


def ap50(weights, data, ids):
    b = YOLO(weights).val(data=data, split="val", imgsz=640, verbose=False).box
    got = dict(zip(map(int, b.ap_class_index), map(float, b.ap50)))
    return {n: round(got.get(i, 0.0), 4) for n, i in zip(NAMES, ids)}


metrics = {"fine_tuned_ap50": ap50(BEST, f"{OUT}/dataset.yaml", [0, 1, 2]),
           "stock_yolo11n_ap50": ap50("yolo11n.pt", str(coco_dir / "dataset.yaml"), [67, 73, 0]),
           "data": {"coco": N_COCO, "open_images": N_OI, "negatives": N_NEG, "val_per_source": N_VAL},
           "epochs": EPOCHS}
print(json.dumps(metrics, indent=1))"""

EXPORT = """import shutil

# 480x640 matches 4:3 webcams; ProctorLens reads the input size and the class names from the file
onnx = YOLO(BEST).export(format="onnx", imgsz=[480, 640], simplify=True, opset=12)
shutil.copy(onnx, "/content/proctorlens_detector.onnx")
Path("/content/metrics.json").write_text(json.dumps(metrics, indent=1))"""

# Training, scoring and export run in a fresh Python process: forked data-loader workers inherit FiftyOne's threads
# and database client from this one and can deadlock (seen: the run froze on epoch 1, batch 1).
SCRIPT = "\n\n".join([PARAMS, 'NAMES = ["phone", "notes", "person"]', TRAIN, COMPARE, EXPORT])
RUN = f"""import subprocess
import sys

from google.colab import files

open("/content/train_step.py", "w").write({SCRIPT!r})
p = subprocess.Popen([sys.executable, "-u", "/content/train_step.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True)
for line in p.stdout:
    print(line, end="")
assert p.wait() == 0, "training failed: see the output above"
files.download("/content/proctorlens_detector.onnx")
files.download("/content/metrics.json")"""

CELLS = [("markdown", INTRO), ("code", SETUP), ("code", PARAMS), ("code", BUILD), ("code", RUN)]


def notebook() -> dict:
    cell = lambda kind, src: {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),  # noqa: E731
                              **({"outputs": [], "execution_count": None} if kind == "code" else {})}
    return {"nbformat": 4, "nbformat_minor": 5, "cells": [cell(k, s) for k, s in CELLS],
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}}}


if __name__ == "__main__":
    out = Path(__file__).with_name("train_detector_colab.ipynb")
    out.write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    print(f"wrote {out}")
