"""Static files for the browser build (docs/DECISIONS.md ADR-019), into frontend/public/ (served at / by Vite and
Vercel): py/proctorlens.zip (this package's Python, unpacked into Pyodide) and models/ (the model files the page runs:
MediaPipe face landmarker, YOLO11n, buffalo_sc detector + recognizer; yolo11n.json = class names and input size).
`--zip-only` refreshes just the zip (a build machine without the model files).
Usage: python tools/web_assets.py [--zip-only]"""
import ast
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "frontend" / "public"
MODELS = {"face_landmarker.task": "data/models/face_landmarker.task", "yolo11n.onnx": "data/models/yolo11n.onnx",
          "det_500m.onnx": "data/models/buffalo_sc/det_500m.onnx", "w600k_mbf.onnx": "data/models/buffalo_sc/w600k_mbf.onnx"}


def build_zip() -> Path:
    out = PUBLIC / "py" / "proctorlens.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = ROOT / "src" / "proctorlens"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob("*.py")):
            z.write(p, Path("proctorlens") / p.relative_to(src))
    return out


def copy_models() -> list[Path]:
    import onnxruntime as ort

    d = PUBLIC / "models"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, src in MODELS.items():
        if not (ROOT / src).is_file():
            sys.exit(f"missing {src}: run `proctorlens fetch-models` first")
        shutil.copy2(ROOT / src, d / name)
        out.append(d / name)
    s = ort.InferenceSession(str(d / "yolo11n.onnx"), providers=["CPUExecutionProvider"])
    names = ast.literal_eval(s.get_modelmeta().custom_metadata_map["names"])
    (d / "yolo11n.json").write_text(json.dumps({"names": names, "imgsz": s.get_inputs()[0].shape[2:]}), encoding="utf-8")
    return out + [d / "yolo11n.json"]


if __name__ == "__main__":
    files = [build_zip()] + ([] if "--zip-only" in sys.argv else copy_models())
    for f in files:
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.1f} MB")
