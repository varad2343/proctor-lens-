"""Body pose (YOLO pose: 17 COCO keypoints per person) and hands (MediaPipe Hand Landmarker: 21 points per hand).
Display + detection summary only: nothing here feeds features, scores or events (docs/DECISIONS.md ADR-018).
Both adapters are thin and import their library lazily; outputs are normalized image coordinates."""
from __future__ import annotations

import cv2
import numpy as np

# COCO keypoint order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees,
# 15-16 ankles (left, right). Skeleton edges for drawing.
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


class PoseDetector:
    """YOLO pose model -> list of (17, 3) arrays: normalized x, y, keypoint confidence."""

    def __init__(self, model_path: str, conf: float = 0.3, imgsz: int = 640):
        from ultralytics import YOLO

        self._y, self.conf, self.imgsz = YOLO(str(model_path)), conf, imgsz

    def process(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        r = self._y.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.keypoints is None or r.keypoints.xyn is None:
            return []
        xy, c = r.keypoints.xyn.cpu().numpy(), r.keypoints.conf
        c = np.ones(xy.shape[:2], np.float32) if c is None else c.cpu().numpy()
        return [np.c_[p, k].astype(np.float32) for p, k in zip(xy, c)]


class HandDetector:
    """MediaPipe Hand Landmarker, IMAGE mode (stateless: it can run on any subset of frames) -> list of (21, 2)."""

    def __init__(self, model_path: str, num_hands: int = 4):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._h = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path)), num_hands=num_hands,
            running_mode=vision.RunningMode.IMAGE))

    def process(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return [np.array([(p.x, p.y) for p in h], np.float32) for h in self._h.detect(img).hand_landmarks]
