"""Sample still frames from recordings for detector labelling (spec 5.4): <out>/<recording>_<t_ms:08d>.jpg.

python -m ml.data.extract_frames --videos data/recordings --out data/frames --every-s 3

Label them in CVAT / Label Studio (pre-label with a COCO model there), export, then ml.data.to_yolo_format. Frames show
faces: keep the output inside the restricted, gitignored data/ folder.
# ponytail: uniform time sampling; no near-duplicate filtering or hard-negative mining (raise --every-s / pick by hand).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ml.data.labels import VIDEO_EXT


def extract(video: str | Path, out: str | Path, every_s: float = 3.0, quality: int = 90) -> list[Path]:
    """Write a JPEG roughly every `every_s` seconds of video time; returns the paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps, nxt, i, paths = cap.get(cv2.CAP_PROP_FPS) or 30.0, 0.0, 0, []
    while cap.grab():
        t, i = cap.get(cv2.CAP_PROP_POS_MSEC) or i * 1000 / fps, i + 1
        if t < nxt:
            continue
        nxt = t + every_s * 1000
        paths.append(out / f"{Path(video).stem}_{int(t):08d}.jpg")
        cv2.imwrite(str(paths[-1]), cap.retrieve()[1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    cap.release()
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="video file or directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=3.0)
    ap.add_argument("--quality", type=int, default=90)
    a = ap.parse_args(argv)
    v = Path(a.videos)
    vids = sorted(p for p in v.iterdir() if p.suffix.lower() in VIDEO_EXT) if v.is_dir() else [v]
    for p in vids:
        print(p.name, len(extract(p, a.out, a.every_s, a.quality)), "frames")


if __name__ == "__main__":
    main()
