"""Browser build of a candidate session (docs/DECISIONS.md ADR-019): the server's LiveSession runs in the candidate's
browser inside Pyodide, so a host without servers for AI (Vercel) can serve ProctorLens. The camera models run in
JavaScript (MediaPipe Tasks for the face landmarker, ONNX Runtime Web for YOLO and the identity models); every step
around them is this package's own Python, unchanged. The page sends BrowserSession the same messages it sends the
server over /ws/stream; results leave through an outbox (`drain`) that the page posts to the API.

Model handshake (ONNX Runtime Web has no blocking call, MediaPipe's is synchronous):
- landmarks: `js.landmarks(rgba, w, h, t_ms, image_mode)`, a synchronous callback.
- YOLO: before a frame, `detection_input(jpeg, t)` says whether the perceiver will run the detector on it (the
  perceiver's own rule) and returns the letterboxed tensor; the page runs YOLO and passes the raw output to `frame`.
  Detections therefore come from the same frame as on the server.
- identity: `embed` queues a model request and returns None; the page runs the models after the frame and feeds the
  outputs back (`model_result`). The perceiver retries on None already, so an embedding lands one frame later.
  Detection, alignment and embedding pre/post-processing follow insightface (SCRFD + ArcFace, buffalo_sc).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Detections, Event, Face
from proctorlens.io import _clean
from proctorlens.perception import Perceiver
from proctorlens.perception.landmarks import face_from_landmarks, sort_faces
from proctorlens.perception.objects import _decode, _letterbox, detections_from_boxes
from proctorlens.server.live import LiveSession

CLIP_EXT = ".webm"  # encoded in the browser (WebCodecs)
BROWSER_MAX_AGE_MS = 500


def _js(b: bytes) -> Any:
    """bytes -> a JavaScript Uint8Array inside Pyodide (natively, for tests: unchanged)."""
    try:
        from pyodide.ffi import to_js
    except ImportError:
        return b
    return to_js(b)


def _py(o: Any) -> Any:
    return o.to_py() if hasattr(o, "to_py") else o


def _array(data: Any, shape) -> np.ndarray:
    data = _py(data)
    return np.frombuffer(bytes(data), dtype=np.float32).reshape([int(d) for d in _py(shape)])


# ------------------------------------------------------------------------------------------------ face landmarks

class JsLandmarker:
    """Landmarker interface (process / detect) over MediaPipe Tasks in the page: same model, same outputs."""

    def __init__(self, js, image_mode: bool = False):
        self.js, self.image_mode = js, image_mode

    def detect(self, frame_bgr: np.ndarray, t_ms: int = 0) -> list[tuple[np.ndarray, dict[str, float], np.ndarray | None]]:
        h, w = frame_bgr.shape[:2]
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).tobytes()
        out = []
        for f in _py(self.js.landmarks(_js(rgba), w, h, int(t_ms), self.image_mode)) or []:
            lm = np.frombuffer(bytes(_py(f["lm"])), dtype=np.float32).reshape(-1, 3).astype(float)
            m = f.get("m")
            out.append((lm, {str(k): float(v) for k, v in dict(_py(f["blend"])).items()},
                        None if m is None else np.frombuffer(bytes(_py(m)), dtype=np.float32).reshape(4, 4).astype(float)))
        return out

    def process(self, frame_bgr: np.ndarray, t_ms: int) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        return sort_faces([face_from_landmarks(lm, bl, m, aspect=w / h) for lm, bl, m in self.detect(frame_bgr, t_ms)])


# ------------------------------------------------------------------------------------------------ YOLO

class JsDetector:
    """ObjectDetector interface; the raw YOLO output for this frame is set by BrowserSession before the frame."""

    def __init__(self, names: dict[int, str], imgsz, conf: float):
        self.names, self.imgsz, self.conf = names, tuple(imgsz), conf
        self.pending: tuple[np.ndarray, tuple[float, int, int]] | None = None

    def tensor(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
        return _letterbox(frame_bgr, self.imgsz)

    def process(self, frame_bgr: np.ndarray) -> Detections:
        if self.pending is None:  # the warm-up frame at exam start, or a frame the page did not run YOLO on
            return Detections()
        (out, lb), self.pending = self.pending, None
        h, w = frame_bgr.shape[:2]
        return detections_from_boxes(_decode(out, self.names, lb, w, h, self.conf), self.conf)


# ------------------------------------------------------------------------------------------------ identity

_ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                         [70.7299, 92.2041]], dtype=np.float32)
_SCRFD = {"size": 320, "strides": (8, 16, 32), "anchors": 2, "nms": 0.4, "thresh": 0.5}  # buffalo_sc det_500m


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity transform src -> dst (least squares, Umeyama 1991), as skimage's SimilarityTransform.estimate
    that insightface's norm_crop uses."""
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    A = d0.T @ s0 / n
    D = np.ones(d)
    if np.linalg.det(A) < 0:
        D[d - 1] = -1
    U, S, Vt = np.linalg.svd(A)
    T = np.eye(d + 1)
    r = np.linalg.matrix_rank(A)
    if r == d - 1 and np.linalg.det(U) * np.linalg.det(Vt) <= 0:
        T[:d, :d] = U @ Vt
    elif r == d - 1:
        s = D[d - 1]
        D[d - 1] = -1
        T[:d, :d] = U @ np.diag(D) @ Vt
        D[d - 1] = s
    else:
        T[:d, :d] = U @ np.diag(D) @ Vt
    scale = 1.0 / s0.var(0).sum() * (S @ D)
    T[:d, d] = mu_d - scale * (T[:d, :d] @ mu_s)
    T[:d, :d] *= scale
    return T[:2]


def scrfd_input(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """insightface SCRFD._detect_candidates preprocessing: keep aspect into a size x size canvas (top-left), then
    blobFromImage(1/128, mean 127.5, swapRB). Returns (NCHW float32, det_scale)."""
    size = _SCRFD["size"]
    h, w = crop_bgr.shape[:2]
    if h / w > 1.0:
        nh, nw = size, int(size / (h / w))
    else:
        nw, nh = size, int(size * (h / w))
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(crop_bgr, (nw, nh))
    blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True)
    return np.ascontiguousarray(blob, dtype=np.float32), nh / h


def scrfd_decode(outs: list[np.ndarray], det_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """insightface SCRFD.forward + detect (9 outputs: scores, boxes, keypoints per stride; not batched) -> (dets
    (n,5) x0,y0,x1,y1,score in crop pixels, kps (n,5,2)), after NMS, best first."""
    size, strides, na, fmc = _SCRFD["size"], _SCRFD["strides"], _SCRFD["anchors"], len(_SCRFD["strides"])
    scores_l, boxes_l, kps_l = [], [], []
    for i, stride in enumerate(strides):
        scores = outs[i].reshape(-1, 1)
        boxes, kps = outs[i + fmc].reshape(-1, 4) * stride, outs[i + 2 * fmc].reshape(-1, 10) * stride
        hh = ww = size // stride
        centers = (np.stack(np.mgrid[:hh, :ww][::-1], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
        centers = np.stack([centers] * na, axis=1).reshape(-1, 2)
        keep = np.where(scores >= _SCRFD["thresh"])[0]
        b = np.stack([centers[:, 0] - boxes[:, 0], centers[:, 1] - boxes[:, 1],
                      centers[:, 0] + boxes[:, 2], centers[:, 1] + boxes[:, 3]], axis=-1)
        k = np.stack([centers[:, j % 2] + kps[:, j] for j in range(10)], axis=-1).reshape(-1, 5, 2)
        scores_l.append(scores[keep]), boxes_l.append(b[keep]), kps_l.append(k[keep])
    scores = np.vstack(scores_l).ravel()
    if not scores.size:
        return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
    order = np.argsort(-scores, kind="stable")
    dets = np.hstack([np.vstack(boxes_l) / det_scale, scores[:, None]]).astype(np.float32)[order]
    kps = (np.vstack(kps_l) / det_scale)[order]
    x1, y1, x2, y2, sc = dets.T
    areas, idx, keep = (x2 - x1 + 1) * (y2 - y1 + 1), np.argsort(-sc, kind="stable"), []
    while idx.size:
        i = idx[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[idx[1:]]) - np.maximum(x1[i], x1[idx[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[idx[1:]]) - np.maximum(y1[i], y1[idx[1:]]) + 1)
        ovr = w * h / (areas[i] + areas[idx[1:]] - w * h)
        idx = idx[np.where(ovr <= _SCRFD["nms"])[0] + 1]
    return dets[keep], kps[keep]


def arcface_input(crop_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """insightface norm_crop (112, arcface template) + ArcFaceONNX.get_feat blob (mean 127.5, std 127.5, swapRB)."""
    aimg = cv2.warpAffine(crop_bgr, umeyama(kps.astype(np.float64), _ARCFACE_DST.astype(np.float64)), (112, 112),
                          borderValue=0.0)
    return np.ascontiguousarray(cv2.dnn.blobFromImages([aimg], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5),
                                                       swapRB=True), dtype=np.float32)


class JsIdentity:
    """IdentityChecker interface. embed() returns the latest finished embedding (from a request at most `max_age_ms`
    old) or None, and queues a new request; BrowserSession forwards requests to the page and results back."""

    def __init__(self, max_age_ms: int = 1000):
        self.max_age_ms = max_age_ms
        self.requests: list[dict] = []
        self._crop: np.ndarray | None = None
        self._ready: tuple[np.ndarray, int] | None = None
        self._busy, self._t, self._n = False, 0, 0

    def embed(self, frame_bgr: np.ndarray, face: Face, t_ms: int | None = None) -> np.ndarray | None:
        t = self._t if t_ms is None else t_ms
        if self._ready is not None:
            (emb, made), self._ready = self._ready, None
            if t - made <= self.max_age_ms:
                return emb
        if not self._busy:  # same crop as IdentityChecker.embed: landmarker bbox + 40 %
            h, w = frame_bgr.shape[:2]
            x0, y0, x1, y1 = face.bbox
            mx, my = 0.4 * (x1 - x0), 0.4 * (y1 - y0)
            a, b, c, d = (np.clip([x0 - mx, y0 - my, x1 + mx, y1 + my], 0, 1) * [w, h, w, h]).astype(int)
            crop = frame_bgr[b:d, a:c]
            if crop.size:
                blob, scale = scrfd_input(crop)
                self._crop, self._scale, self._made = crop, scale, t
                self._request("scrfd", blob)
        return None

    def _request(self, model: str, x: np.ndarray) -> None:
        self._busy, self._n = True, self._n + 1
        self.requests.append({"id": self._n, "model": model, "input": x})

    def result(self, model: str, outs: list[np.ndarray]) -> None:
        if model == "scrfd":
            dets, kps = scrfd_decode(outs, self._scale)
            if not len(dets):  # insightface finds no face in the crop: embed() gives None (the perceiver retries)
                self._busy = False
                return
            best = int(np.argmax((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])))
            self._request("arcface", arcface_input(self._crop, kps[best]))
        else:
            e = outs[0].reshape(-1).astype(np.float32)
            self._ready, self._busy = (e / (np.linalg.norm(e) + 1e-12), self._made), False


# ------------------------------------------------------------------------------------------------ session

class _Outbox:
    """DB interface for LiveSession; each of its writes becomes one named operation the page posts to the API (a
    fixed set: the browser never sends SQL). Event ids are the session's own counter (client_eid)."""

    def __init__(self, sid: str):
        self.sid, self.ops, self._n = sid, [], 0

    def add_event(self, sid: str, ev: Event, explanation: str) -> int:
        self._n += 1
        self.ops.append({"op": "event", "client_eid": self._n, "event": _clean(ev.to_dict()), "explanation": explanation})
        return self._n

    def run(self, sql: str, *a) -> int:
        q = " ".join(sql.split())
        if q.startswith("UPDATE sessions SET calibration_json=?"):
            self.ops.append({"op": "calibration", "calibration": json.loads(a[0]), "mode": a[1]})
        elif q.startswith("UPDATE sessions SET status='live'"):
            self.ops.append({"op": "start", "mode": a[1]})
        elif q.startswith("UPDATE events SET clip_path=?"):
            self.ops.append({"op": "evidence", "client_eid": a[2], "clip_path": a[0], "thumbs": json.loads(a[1])})
        elif q.startswith("INSERT INTO browser_events"):
            self.ops.append({"op": "browser_event", "kind": a[1], "t_ms": a[2], "detail": a[3]})
        elif q.startswith("UPDATE sessions SET status='ended'"):
            self.ops.append({"op": "end", "answers": json.loads(a[1] or "null")})
        else:
            raise ValueError(f"browser session: unexpected database write: {q[:60]}")
        return 0


class BrowserSession(LiveSession):
    clip_ext = CLIP_EXT
    features_file = "features.csv"  # the cloud API reads it without pyarrow

    def __init__(self, sid: str, cfg: Config, js, yolo: dict, identity: bool = True):
        # a browser runs the pipeline at ~4-7 fps (phones: less), not the server's 10: a frame stays current at a grid
        # step for up to 500 ms (valid down to 2 fps), else half the steps read as frame_gap
        cfg = dataclasses.replace(cfg, pipeline=dataclasses.replace(
            cfg.pipeline, max_age_ms=max(cfg.pipeline.max_age_ms, BROWSER_MAX_AGE_MS)))
        self.js, self._feed, self._files, self._slices = js, [], [], []
        self._clips: dict[int, dict] = {}  # id -> {path, fps, frames}: waiting for the page to encode and upload
        self._clip_n = 0
        self._det = JsDetector({int(k): v for k, v in yolo["names"].items()}, yolo["imgsz"], cfg.models.detector_conf)
        self._id = JsIdentity() if identity else None
        lm = JsLandmarker(js)
        super().__init__(sid, cfg, Path("/work") / sid, _Outbox(sid), lambda cfg: lm, self._perceiver, threads=False)

    def _perceiver(self, cfg: Config, lm: JsLandmarker) -> Perceiver:
        p = Perceiver(cfg, landmarker=lm, detector=self._det, identity=self._id, recoverer=JsLandmarker(self.js, True))
        if self._id is not None:
            embed = self._id.embed
            self._id.embed = lambda frame, face: embed(frame, face, self.last_t)  # requests carry the frame time
        return p

    # --- frames and control: the /ws/stream protocol, as JSON text in and out
    def detection_input(self, t_client: float) -> tuple[Any, list[int], list[float]] | None:
        """(tensor bytes, shape, letterbox) when the next frame at t_client goes through the detector, else None."""
        if self.phase != "exam" or self.pl is None or self.t0 is None:
            return None
        t, pl = round(float(t_client) - self.t0), self.pl
        if pl._last_t is not None and t <= pl._last_t:
            return None
        if pl.perceiver._det is not None and pl._idx % self.cfg.pipeline.yolo_every:
            return None
        return None if self._next_img is None else self._tensor()

    def prepare(self, jpeg) -> None:
        """Decode the next frame once for detection_input (frame() decodes again, as on the server)."""
        self._next_img = cv2.imdecode(np.frombuffer(bytes(_py(jpeg)), np.uint8), cv2.IMREAD_COLOR)

    _next_img: np.ndarray | None = None

    def _tensor(self):
        x, lb = self._det.tensor(self._next_img)
        self._lb = lb
        return _js(x.tobytes()), list(x.shape), list(lb)

    def frame_in(self, seq: int, t_client: float, jpeg, det_out=None, det_shape=None) -> str:
        if det_out is not None:
            self._det.pending = (_array(det_out, det_shape), self._lb)
        cand, proc = self.frame(int(seq), float(t_client), bytes(_py(jpeg)))
        self._feed += proc
        return json.dumps(_clean(cand))

    def live_status(self) -> str:
        """For the proctor's views (the server keeps it on the session row): phase, flags in progress and the latest
        readouts (quality, zone, faces, people, detector confidences, head turn, processing time)."""
        last = next((m for m in reversed(self._feed) if m.get("type") == "session_status"), None)
        if last is not None:
            self._status = {k: v for k, v in last.items() if k not in ("type", "session_id")}
        ongoing = [{"type": e.type, "start_ms": e.start_ms} for e in self.pl.ongoing()] if self.pl and self.phase == "exam" else []
        return json.dumps(_clean({**getattr(self, "_status", {}), "phase": self.phase, "ongoing": ongoing}))

    def control_in(self, msg: str) -> str:
        cand, proc = self.control(json.loads(msg))
        self._feed += proc
        return json.dumps(_clean(cand))

    def tick_in(self) -> None:
        self._feed += self.tick()[1]

    # --- evidence: spread over frames (no threads here)
    def write_clip(self, jpegs: list[bytes], fps: float, out: Path) -> bool:
        self._clip_n += 1
        cid = self._clip_n
        self._clips[cid] = {"path": out.relative_to(self.dir).as_posix(), "fps": fps, "frames": list(jpegs), "sent": False}
        return True

    def _evidence(self, eid, ev, frames) -> None:
        steps = self._evidence_steps(eid, ev, frames)

        def job():
            yield from steps
            self._files += [p.relative_to(self.dir).as_posix() for p in sorted((self.dir / "evidence").glob(f"{eid}_*.jpg"))]
        self._slices.append(job())

    def run_slices(self, budget_ms: float = 20.0) -> bool:
        """Advance evidence work for about budget_ms (math.inf = all of it). True while work is left."""
        import time

        t0 = time.perf_counter()
        while self._slices and (time.perf_counter() - t0) * 1000 < budget_ms:
            try:
                next(self._slices[0])
            except StopIteration:
                self._slices.pop(0)
        return bool(self._slices)

    def end_in(self, answers_json: str | None = None) -> str:
        """Ends the exam (answers, final events, files). The evidence still owed is run_slices' work: the page
        reports the exam submitted first, then finishes it."""
        cand = self.control({"type": "exam_end", "answers": json.loads(answers_json or "null")})
        self._files += [n for n in ("features.csv", "events.json", "summary.json") if (self.dir / n).is_file()]
        self._feed += cand[1]
        return json.dumps(cand[0])

    # --- identity requests
    def model_requests(self) -> list[dict]:
        if self._id is None:
            return []
        out, self._id.requests = self._id.requests, []
        return [{"id": r["id"], "model": r["model"], "input": _js(r["input"].tobytes()), "shape": list(r["input"].shape)}
                for r in out]

    def model_result(self, model: str, outputs) -> None:
        self._id.result(model, [_array(o["data"], o["shape"]) for o in _py(outputs)])

    # --- what leaves the browser
    def drain(self) -> str:
        """JSON of everything for the API since the last call: ops (DB writes, in order), feed (proctor messages),
        files (paths under the session folder: fetch with read_file), clips (id, path, fps: fetch the frames with
        clip_frames, encode, upload, then clip_done). The live picture is snapshot()."""
        new = [{"id": i, "path": c["path"], "fps": c["fps"]} for i, c in self._clips.items() if not c["sent"]]
        for c in new:
            self._clips[c["id"]]["sent"] = True
        out = {"ops": self.db.ops, "feed": self._feed, "files": self._files, "clips": new}
        self.db.ops, self._feed, self._files = [], [], []
        return json.dumps(_clean(out))

    def read_file(self, path: str) -> Any:
        return _js((self.dir / path).read_bytes())

    def clip_frames(self, cid: int) -> list:
        return [_js(b) for b in self._clips[int(cid)]["frames"]]

    def clip_done(self, cid: int) -> None:
        self._clips.pop(int(cid), None)  # the frames leave memory once the page has the encoded clip

    def snapshot(self) -> tuple[Any, Any] | None:
        """(detector-view jpeg, raw jpeg) of the latest exam frame for the proctor's live view."""
        if self.latest_jpeg is None:
            return None
        ov = self.view_jpeg() if self.phase == "exam" else None
        return _js(ov or self.latest_jpeg), _js(self.latest_jpeg)

