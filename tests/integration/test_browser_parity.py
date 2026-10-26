"""Browser build parity (ADR-019): the same frames through the server's LiveSession (models in Python) and through
BrowserSession (models behind the page handshake, answered here by the same Python models) give the same events.
Needs the model files, mediapipe, onnxruntime and ultralytics' sample images; skips otherwise."""
import ast
import tempfile
from pathlib import Path

import cv2
import numpy as np

from proctorlens.core.config import load_config

CFGS = ("configs/pipeline.yaml", "configs/policy.yaml")


def _frames():
    from proctorlens.cli import sample_images

    imgs = sample_images()
    if not imgs:
        return None
    W, H = 640, 480
    room = cv2.GaussianBlur((np.random.default_rng(0).random((H, W, 3)) * 120 + 60).astype(np.uint8), (5, 5), 0)
    script = [(20, cv2.resize(imgs["zidane_crop"], (W, H))), (26, np.full((H, W, 3), 3, np.uint8)), (33, room),
              (40, cv2.resize(imgs["bus"], (W, H)))]  # = tools/fake_cam.py: face, covered, empty room, several people
    jpg = {id(im): cv2.imencode(".jpg", im)[1].tobytes() for _, im in script}
    return [jpg[id(next(im for end, im in script if i / 10 < end))] for i in range(40 * 10)]


class FakeJS:
    """The page's model callbacks, answered by the Python models the server uses."""

    def __init__(self, cfg):
        from proctorlens.perception.landmarks import Landmarker

        self.video, self.image = Landmarker(cfg.models.landmarker), Landmarker(cfg.models.landmarker, 2, image_mode=True)

    def landmarks(self, rgba, w, h, t_ms, image_mode):
        bgr = cv2.cvtColor(np.frombuffer(rgba, np.uint8).reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
        lmk = self.image if image_mode else self.video
        return [{"lm": lm.astype(np.float32).tobytes(), "blend": bl, "m": None if m is None else np.asarray(m, np.float32).tobytes()}
                for lm, bl, m in lmk.detect(bgr, t_ms)]


def test_browser_session_matches_server_session():
    try:
        import onnxruntime as ort

        from proctorlens.browser import BrowserSession
        from proctorlens.perception import Perceiver
        from proctorlens.perception.landmarks import Landmarker
        from proctorlens.server.db import DB
        from proctorlens.server.live import LiveSession
    except ImportError:
        return
    cfg = load_config(*CFGS)
    m = cfg.models
    if not all(Path(p).exists() for p in (m.landmarker, m.detector, Path(m.identity) / "w600k_mbf.onnx")):
        return
    frames = _frames()
    if frames is None:
        return
    yolo = ort.InferenceSession(m.detector, providers=["CPUExecutionProvider"])
    names = ast.literal_eval(yolo.get_modelmeta().custom_metadata_map["names"])
    det = ort.InferenceSession(str(Path(m.identity) / "det_500m.onnx"), providers=["CPUExecutionProvider"])
    rec = ort.InferenceSession(str(Path(m.identity) / "w600k_mbf.onnx"), providers=["CPUExecutionProvider"])
    run = lambda s, x: [np.asarray(o, np.float32) for o in s.run(None, {s.get_inputs()[0].name: x})]  # noqa: E731

    d = Path(tempfile.mkdtemp())
    db = DB(d / "t.db")
    db.run("INSERT INTO sessions (id, token, status) VALUES ('s', 't', 'created')")
    server = LiveSession("s", cfg, d / "s", db, lambda c: Landmarker(c.models.landmarker), lambda c, lm: Perceiver(c, landmarker=lm))
    browser = BrowserSession("b", cfg, FakeJS(cfg), {"names": names, "imgsz": yolo.get_inputs()[0].shape[2:]})
    browser.dir = d / "b"

    t0, ops = 1.7e12, []
    for sess in (server, browser):
        sess.control({"type": "hello"})
        sess.control({"type": "exam_start"})
    for i, jpeg in enumerate(frames):
        t = t0 + i * 100
        server.frame(i, t, jpeg)
        browser.prepare(jpeg)
        if (di := browser.detection_input(t)) is not None:
            x = np.frombuffer(di[0], np.float32).reshape(di[1])
            out = run(yolo, x)[0]
            browser.frame_in(i, t, jpeg, out.tobytes(), list(out.shape))
        else:
            browser.frame_in(i, t, jpeg)
        for r in browser.model_requests():
            x = np.frombuffer(r["input"], np.float32).reshape(r["shape"])
            outs = run(det if r["model"] == "scrfd" else rec, x)
            browser.model_result(r["model"], [{"data": o.tobytes(), "shape": list(o.shape)} for o in outs])
        browser.run_slices(20)
        ops += __import__("json").loads(browser.drain())["ops"]
    server.control({"type": "exam_end"})
    browser.end_in()
    browser.run_slices(float("inf"))
    ops += __import__("json").loads(browser.drain())["ops"]

    srv = sorted((r["type"], r["start_ms"], r["end_ms"]) for r in db.all("SELECT * FROM events"))
    brw = sorted((o["event"]["type"], o["event"]["start_ms"], o["event"]["end_ms"]) for o in ops if o["op"] == "event")
    assert srv and brw == srv, f"server {srv}\nbrowser {brw}"
    ev = [o for o in ops if o["op"] == "evidence"]
    assert len(ev) == len(brw) and all(o["clip_path"].endswith(".webm") and o["thumbs"] for o in ev)
    server.worker.shutdown()
