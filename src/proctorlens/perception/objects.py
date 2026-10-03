"""Phone / notes / person detection. detections_from_boxes() and _decode() are pure; ObjectDetector is thin.

Detections never become events by themselves: downstream turns them into presence scores.
"""
from __future__ import annotations

import ast

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from proctorlens.core.types import Box, Detections

# COCO names and fine-tuned names -> canonical classes; anything else is ignored.
_CLASS = {"cell phone": "phone", "phone": "phone", "book": "notes", "notes": "notes", "person": "person"}


def _iou(a: Box, b: Box) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detections_from_boxes(items: list[tuple[str, float, Box]], conf_thr: float) -> Detections:
    """Map class names, drop conf < conf_thr, NMS per class at IoU 0.5 (never across classes: a phone in a person box
    stays); phone/notes conf = max. boxes come out most confident first."""
    keep = [(_CLASS[n.lower()], c, b) for n, c, b in items if n.lower() in _CLASS and c >= conf_thr]
    kept: list[tuple[str, float, Box]] = []
    for k in sorted(keep, key=lambda k: -k[1]):  # greedy NMS: the ONNX raw layout (_decode) has none of its own
        if all(k[0] != p[0] or _iou(k[2], p[2]) <= 0.5 for p in kept):
            kept.append(k)

    def top(name: str) -> float:
        return max((c for n, c, _ in kept if n == name), default=0.0)

    return Detections(phone_conf=top("phone"), notes_conf=top("notes"),
                      person_boxes=[b for n, _, b in kept if n == "person"], boxes=kept)


class IdTracker:
    """Display-only track numbers (IoU + Hungarian matching): a box keeps its number while it overlaps its last
    position at IoU >= min_iou, never across classes. A track unseen for more than max_gap_ms ends; numbers are never
    reused, so a face that left and came back gets a new one. Numbers follow boxes: they do not identify anyone.
    ponytail: no motion model; add centre-distance gating if numbers churn on real recordings."""

    def __init__(self, min_iou: float = 0.3, max_gap_ms: int = 1000):
        self.min_iou, self.max_gap_ms, self.n = min_iou, max_gap_ms, 0
        self.tracks: dict[int, tuple[str, Box, int]] = {}  # number -> (class, last box, last seen t_ms)

    def __call__(self, items: list[tuple[str, Box]], t_ms: int) -> list[int]:
        self.tracks = {i: tr for i, tr in self.tracks.items() if t_ms - tr[2] <= self.max_gap_ms}
        ids = list(self.tracks)
        cost = np.array([[1 - _iou(b, self.tracks[i][1]) if c == self.tracks[i][0] else 1.0 for i in ids]
                         for c, b in items]).reshape(len(items), len(ids))
        out = [0] * len(items)
        for r, k in zip(*linear_sum_assignment(cost)):
            if cost[r, k] <= 1 - self.min_iou:  # the solver pairs every row it can, even at cost 1
                out[r] = ids[k]
        for r, (c, b) in enumerate(items):
            if not out[r]:
                self.n += 1
                out[r] = self.n
            self.tracks[out[r]] = (c, b, t_ms)
        return out


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, tuple[float, int, int]]:
    """Square letterbox (pad 114) -> float32 NCHW RGB in [0,1], plus (scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return x, (r, px, py)


def _decode(out: np.ndarray, names: dict, lb: tuple[float, int, int], w: int, h: int,
            conf: float) -> list[tuple[str, float, Box]]:
    """Ultralytics ONNX output -> (name, conf, normalized xyxy). Handles (1,4+nc,N) cxcywh and the
    NMS-free end2end (1,N,6) x1y1x2y2,conf,cls layouts. Caller's detections_from_boxes does the NMS."""
    r, px, py = lb
    o = np.asarray(out)[0]
    if o.shape[-1] == 6:
        xyxy, s, c = o[:, :4], o[:, 4], o[:, 5].astype(int)
    else:
        o = o.T
        s, c = o[:, 4:].max(1), o[:, 4:].argmax(1)
        xyxy = np.c_[o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2]
    k = s >= conf
    xyxy = (xyxy[k] - [px, py, px, py]) / r / [w, h, w, h]
    return [(names[int(ci)], float(si), tuple(float(v) for v in np.clip(b, 0, 1)))
            for b, si, ci in zip(xyxy, s[k], c[k])]


class ObjectDetector:
    """YOLO nano: ultralytics for .pt, onnxruntime for .onnx (both imported lazily)."""

    def __init__(self, model_path: str, conf: float = 0.3, imgsz: int = 640):
        self.conf, self.imgsz, self._y = conf, imgsz, None
        if str(model_path).endswith(".onnx"):
            import onnxruntime as ort

            self._s = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            side = self._s.get_inputs()[0].shape[2]  # fixed-size exports pin the input side
            self.imgsz = side if isinstance(side, int) else imgsz
            self._names = ast.literal_eval(self._s.get_modelmeta().custom_metadata_map["names"])
        else:
            from ultralytics import YOLO

            self._y = YOLO(str(model_path))

    def process(self, frame_bgr: np.ndarray) -> Detections:
        h, w = frame_bgr.shape[:2]
        if self._y is not None:
            r = self._y.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            items = [(r.names[int(c)], float(s), tuple(float(v) for v in b)) for b, s, c in zip(
                r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())]
        else:
            x, lb = _letterbox(frame_bgr, self.imgsz)
            out = self._s.run(None, {self._s.get_inputs()[0].name: x})[0]
            items = _decode(out, self._names, lb, w, h, self.conf)
        return detections_from_boxes(items, self.conf)
