"""MediaPipe Face Landmarker adapter. All feature maths is in the pure face_from_landmarks().

Landmark ids (MediaPipe 478-point mesh, "left"/"right" = subject's, so left = image-right):
(image-left corner, image-right corner, upper lid, lower lid, iris centre) per eye.
Face.blend = the 52 MediaPipe blendshapes + one landmark-derived extra: "mar" (mouth aspect ratio).

Small-face recovery (head_rect / from_crop / recover_faces): the landmarker's face detector works on the whole frame
shrunk to ~128-192 px, so faces under ~100-150 px tall are missed. A person box from the object detector says where
to look: crop its head region, upscale, run a second (IMAGE-mode) landmarker on the crop, map the landmarks back.
"""
from __future__ import annotations

import cv2
import numpy as np

from proctorlens.core.types import Box, Face
from proctorlens.perception.head_pose import euler_from_matrix

_R_EYE = (33, 133, 159, 145, 468)
_L_EYE = (362, 263, 386, 374, 473)
_MOUTH = (78, 308, 13, 14)  # corners (image-left, image-right), inner-lip upper, inner-lip lower
# ponytail: lid-gap / eye-width of an open eye, fixed scale (per-user scale lives in the baseline). Measured on 11 real
# faces: open eyes 0.30-0.34 (-> 0.83-0.94 here), closed or lowered 0.0-0.11 (-> 0-0.3). It was 0.3, which saturated at 1.0.
_OPEN_REF = 0.36
_HEAD_FRAC = 0.45  # head + shoulders = the top 45% of a person box
_MIN_CROP = 24  # px: a head crop smaller than this (a distant person) is not worth a landmarker run
_RECOVER_MAX = 3  # person boxes re-examined per frame; ponytail: every frame while a person box has no face


def _eye(lm: np.ndarray, ids: tuple, aspect: float) -> tuple[float, float, float]:
    """(iris x ratio, iris y ratio, openness). x: 0 = image-left corner; y: 0 = upper lid. Ignores roll."""
    a, b, up, lo, ir = lm[list(ids)]
    ix = (ir[0] - a[0]) / (b[0] - a[0] + 1e-9)
    iy = (ir[1] - up[1]) / (lo[1] - up[1] + 1e-9)
    w = np.hypot((b[0] - a[0]) * aspect, b[1] - a[1])
    op = (lo[1] - up[1]) / (w + 1e-9) / _OPEN_REF
    return float(np.clip(ix, 0, 1)), float(np.clip(iy, 0, 1)), float(np.clip(op, 0, 1))


def face_from_landmarks(lm: np.ndarray, blend: dict[str, float], matrix: np.ndarray | None,
                        conf: float = 1.0, aspect: float = 1.0) -> Face:
    """lm: (478,3) normalized; matrix: 4x4 facial transformation matrix (None -> pose 0,0,0 ponytail: never
    happens with the Landmarker, which always requests matrices); aspect = frame width/height (for ratios).
    eye_open = 1 - eyeBlink{Left,Right} blendshape when both are present, else the landmark eye-aspect-ratio."""
    lm = np.asarray(lm, dtype=float)
    (x0, y0), (x1, y1) = lm[:, :2].min(0), lm[:, :2].max(0)
    lx, ly, lo_ = _eye(lm, _L_EYE, aspect)
    rx, ry, ro = _eye(lm, _R_EYE, aspect)
    if "eyeBlinkLeft" in blend and "eyeBlinkRight" in blend:
        lo_, ro = (float(np.clip(1.0 - blend[k], 0, 1)) for k in ("eyeBlinkLeft", "eyeBlinkRight"))
    c0, c1, u, d = lm[list(_MOUTH)]
    mar = abs(d[1] - u[1]) / (np.hypot((c1[0] - c0[0]) * aspect, c1[1] - c0[1]) + 1e-9)
    yaw, pitch, roll = euler_from_matrix(matrix) if matrix is not None else (0.0, 0.0, 0.0)
    return Face(bbox=(float(x0), float(y0), float(x1), float(y1)), yaw=yaw, pitch=pitch, roll=roll,
                iris=(lx, ly, rx, ry), eye_open=(lo_, ro), blend={**blend, "mar": float(mar)},
                center=(float((x0 + x1) / 2), float((y0 + y1) / 2)), scale=float(y1 - y0), conf=conf)


def sort_faces(faces: list[Face]) -> list[Face]:
    """Largest bbox first; faces[0] is the primary face."""
    return sorted(faces, key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _inside(p: tuple[float, float], b: Box) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def uncovered(boxes: list[Box], faces: list[Face]) -> list[Box]:
    """Person boxes with no face centre inside them, largest first."""
    return sorted((b for b in boxes if not any(_inside(f.center, b) for f in faces)),
                  key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))


def head_rect(box: Box, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    """Pixel rect (x0, y0, x1, y1) of a normalized person box's head region: a square centred on the box's width, side
    min(box width, 0.45 box height) from the box top, grown by `pad` x side on every edge, clipped to the frame.
    ponytail: assumes the head is horizontally centred (a wide box from an outstretched arm can miss it)."""
    s = min((box[2] - box[0]) * w, _HEAD_FRAC * (box[3] - box[1]) * h)
    cx, top = (box[0] + box[2]) * w / 2, box[1] * h
    x0, x1, y0, y1 = cx - s / 2 - pad * s, cx + s / 2 + pad * s, top - pad * s, top + s + pad * s
    return max(0, round(x0)), max(0, round(y0)), min(w, round(x1)), min(h, round(y1))


def from_crop(lm: np.ndarray, rect: tuple[int, int, int, int], w: int, h: int) -> np.ndarray:
    """(n,3) landmarks normalized to a crop at `rect` (any upscaling cancels) -> normalized to the full w x h frame;
    z is in units of the crop width, so it is rescaled to frame-width units."""
    x0, y0, x1, y1 = rect
    return np.c_[(x0 + lm[:, 0] * (x1 - x0)) / w, (y0 + lm[:, 1] * (y1 - y0)) / h, lm[:, 2] * (x1 - x0) / w]


class Landmarker:
    """MediaPipe Tasks FaceLandmarker, VIDEO mode (lazy import). image_mode=True: IMAGE running mode, stateless, for
    unrelated crops (no timestamp coupling)."""

    def __init__(self, model_path: str, num_faces: int = 3, image_mode: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self._mp, self._t, self._image = mp, -1, image_mode
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO, num_faces=num_faces,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))

    def detect(self, frame_bgr: np.ndarray, t_ms: int = 0) -> list[tuple[np.ndarray, dict[str, float], np.ndarray | None]]:
        """Raw per-face (landmarks (478,3) normalized to this image, blendshapes, 4x4 matrix)."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self._image:
            r = self._lm.detect(img)
        else:
            self._t = max(int(t_ms), self._t + 1)  # VIDEO mode needs strictly increasing timestamps
            r = self._lm.detect_for_video(img, self._t)
        out = []
        for i, pts in enumerate(r.face_landmarks):
            bl = {c.category_name: float(c.score) for c in r.face_blendshapes[i]} if r.face_blendshapes else {}
            m = r.facial_transformation_matrixes[i] if r.facial_transformation_matrixes else None
            out.append((np.array([(p.x, p.y, p.z) for p in pts]), bl, m))
        return out

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> list[Face]:
        """Faces sorted by bbox area, largest (primary) first."""
        h, w = frame_bgr.shape[:2]
        return sort_faces([face_from_landmarks(lm, bl, m, aspect=w / h) for lm, bl, m in self.detect(frame_bgr, t_ms)])


def recover_faces(lmk: Landmarker, frame_bgr: np.ndarray, boxes: list[Box], faces: list[Face],
                  pad: float, min_side: int) -> list[Face]:
    """Faces (recovered=True) the main pass missed: for each person box without a face, run `lmk` (an image-mode
    Landmarker) on its upscaled head crop. Landmarks are mapped back to full-frame coordinates; a face is kept only if
    its centre lies inside that person box and not inside an already known face (neighbours bleeding into the crop).
    Returns just the extra faces; the caller merges and re-sorts. Pose comes from the crop's own matrix.
    ponytail: the crop's camera model is its own, so an off-centre face's pose is relative to the crop centre."""
    h, w = frame_bgr.shape[:2]
    out: list[Face] = []
    for box in uncovered(boxes, faces)[:_RECOVER_MAX]:
        rect = head_rect(box, w, h, pad)
        crop = frame_bgr[rect[1]:rect[3], rect[0]:rect[2]]
        if min(crop.shape[:2]) < _MIN_CROP:
            continue
        if (s := min_side / min(crop.shape[:2])) > 1:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for lm, bl, m in lmk.detect(crop):
            f = face_from_landmarks(from_crop(lm, rect, w, h), bl, m, aspect=w / h)
            if _inside(f.center, box) and not any(_inside(f.center, g.bbox) for g in faces + out):
                f.recovered = True
                out.append(f)
    return out
