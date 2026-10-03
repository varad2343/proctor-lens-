"""Fake webcam for the web-app E2E run (frontend/e2e.mjs): a looping y4m for Chromium's
--use-file-for-fake-video-capture. 40 s: a face 0-20 s, covered camera 20-26 s, empty room 26-33 s, several people
33-40 s (images: ultralytics' bundled samples, so needs ultralytics). ~180 MB at 640x480.
Usage: python tools/fake_cam.py out.y4m"""
import sys
from pathlib import Path

import cv2
import numpy as np

from proctorlens.cli import sample_images

W, H, FPS = 640, 480, 10


def main(out: str) -> None:
    imgs = sample_images()
    if not imgs:
        sys.exit("ultralytics sample images not found (pip install ultralytics)")
    room = cv2.GaussianBlur((np.random.default_rng(0).random((H, W, 3)) * 120 + 60).astype(np.uint8), (5, 5), 0)
    script = [(20, cv2.resize(imgs["zidane_crop"], (W, H))),  # the face MediaPipe finds (doctor --selftest)
              (26, np.full((H, W, 3), 3, np.uint8)),  # covered: MONITORING_DEGRADED, not FACE_ABSENT
              (33, room),  # textured and lit, nobody: FACE_ABSENT
              (40, cv2.resize(imgs["bus"], (W, H)))]  # MULTIPLE_PEOPLE
    with Path(out).open("wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
        for i in range(40 * FPS):
            img = next(im for end, im in script if i / FPS < end)
            f.write(b"FRAME\n" + cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).tobytes())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_cam.y4m")
