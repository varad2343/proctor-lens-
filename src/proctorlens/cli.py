"""CLI: replay | live | calibrate | extract-features | report | serve (web app) | doctor | fetch-models.
Heavy imports live inside the commands so `--help` and argument parsing need nothing but the stdlib."""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_CANVAS = (720, 1280)  # ponytail: calibration canvas is fixed and stretched to the screen; real aspect if it matters
_DEFAULT_CFGS = ("configs/pipeline.yaml", "configs/policy.yaml")  # doctor / fetch-models when no --config is given


def _cfg(a: argparse.Namespace):
    from proctorlens.core.config import load_config

    return load_config(*a.config)


def _write(out: Path, prefix: str, df, events, cfg, calib) -> None:
    from proctorlens.io import make_header, save_events, save_features

    f = save_features(df, out / f"{prefix}features.parquet")
    save_events(events, out / f"{prefix}events.json", make_header(cfg, calib))
    print(f"wrote {f} ({len(df)} rows) and {out / f'{prefix}events.json'} ({len(events)} events)")


def _union_s(spans: list[tuple[int, int]]) -> float:
    """Total seconds covered by [start_ms, end_ms) spans; overlaps (phone + notes are one event type) count once."""
    tot, end = 0, None
    for s, e in sorted(spans):
        if end is None or s > end:
            tot, end = tot + (e - s), e
        elif e > end:
            tot, end = tot + (e - end), e
    return tot / 1000


def summarize(df, events, calib) -> dict:
    """Run summary: per-event-type count and seconds, % of time monitoring was degraded, % of time a face was seen,
    calibration, duration. calib = Calibration | None. JSON-safe (undefined = None). Counts observations only."""
    from proctorlens.core.types import EVENT_TYPES, MONITORING_DEGRADED

    ts = [int(t) for t in df["t_ms"]]
    step = statistics.median(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else 100  # grid step, ms
    dur = (ts[-1] - ts[0] + step) / 1000 if ts else 0.0
    by = {t: [(e.start_ms, e.end_ms) for e in events if e.type == t] for t in EVENT_TYPES}
    pct = lambda x: round(min(100.0, 100 * x / dur), 1) if dur else None  # noqa: E731
    seen = sum(v == 1 for v in df["primary_face_present"])  # NaN == 1 is False, so gaps count as not seen
    cal = dict(mode="none", error=None, accepted=None, tries=None) if calib is None else dict(
        mode=calib.mode, error=None if calib.error != calib.error else round(float(calib.error), 3),
        accepted=bool(calib.accepted), tries=int(calib.tries))
    return {"duration_s": round(dur, 1), "n_rows": len(ts), "face_seen_pct": pct(seen * step / 1000),
            "degraded_pct": pct(_union_s(by[MONITORING_DEGRADED])), "calibration": cal,
            "events": {t: {"count": len(v), "total_s": round(_union_s(v), 1)} for t, v in by.items()}}


def _summary(out: Path, df, events, calib) -> None:
    """Write summary.json next to events.json and print the same numbers."""
    s = summarize(df, events, calib)
    (out / "summary.json").write_text(json.dumps(s, indent=1, allow_nan=False), encoding="utf-8")
    p = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731
    c = s["calibration"]
    cal = c["mode"] + ("" if c["error"] is None else f" (error {c['error']:.2f})")
    print(f"summary: {s['duration_s']:.1f} s, face seen {p(s['face_seen_pct'])}, "
          f"monitoring degraded {p(s['degraded_pct'])}, calibration {cal}")
    for t, v in s["events"].items():
        if v["count"]:
            print(f"  {t}: {v['count']} event(s), {v['total_s']:.1f} s")


def _replay(a: argparse.Namespace, perceiver=None) -> int:
    from proctorlens.perception.gaze import Calibration
    from proctorlens.pipeline.runner import run_video

    cfg = _cfg(a)
    calib = Calibration.load(a.calib) if a.calib else None
    df, events = run_video(a.video, cfg, calib, perceiver, a.render, progress=True)
    _write(Path(a.out), "", df, events, cfg, calib)
    _summary(Path(a.out), df, events, calib)
    if a.report:
        from proctorlens.explain.report import write_report

        print(f"wrote {write_report(a.out, cfg, a.render or a.video)}")  # the overlay render, when there is one
    return 0


def _report(a: argparse.Namespace) -> int:
    from proctorlens.explain.report import write_report

    print(f"wrote {write_report(a.out, _cfg(a), a.video)}")
    return 0


def _serve(a: argparse.Namespace) -> int:
    import uvicorn

    from proctorlens.server.app import create_app

    paths = a.config or [p for p in _DEFAULT_CFGS if Path(p).is_file()]
    uvicorn.run(create_app(paths, a.data, frontend=a.frontend), host=a.host, port=a.port)
    return 0


def _extract(a: argparse.Namespace, perceiver=None) -> int:
    from proctorlens.perception.gaze import Calibration
    from proctorlens.pipeline.runner import run_video

    cfg, out = _cfg(a), Path(a.out)
    videos = sorted(p for p in Path(a.dir).iterdir() if p.suffix.lower() in _VIDEO_EXT)
    for v in videos:
        c = v.with_suffix(".calib.json")
        calib = Calibration.load(c) if c.exists() else None
        df, events = run_video(v, cfg, calib, perceiver)
        _write(out, f"{v.stem}.", df, events, cfg, calib)
    print(f"{len(videos)} recordings processed")
    return 0


def _open_camera(index: int):
    """Open a camera and wait for its first frame. DSHOW on Windows: OpenCV's default (MSMF) returned no frames."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    t = time.monotonic()
    while cap.isOpened() and time.monotonic() - t < 5:  # cameras need a few warm-up reads
        if cap.read()[0]:
            return cap
    cap.release()
    raise RuntimeError(f"cannot open camera {index}")


_WIN = "proctorlens calibration"
_QUIT = (27, ord("q"))


def _closed(win: str) -> bool:
    import cv2

    return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1  # the user closed it (-1/0 once gone)


def _text(canvas, s: str, y: int, scale: float = 0.8, color=(200, 200, 200)) -> None:
    import cv2

    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(canvas, s, ((_CANVAS[1] - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _show_dot(d, n: int, total: int, seen: bool, frame) -> int:
    """Draw one calibration dot with live feedback; returns the key pressed (27 if the window was closed)."""
    import cv2
    import numpy as np

    h, w = _CANVAS
    canvas = np.zeros((h, w, 3), np.uint8)
    c = (int(d.x * w), int(d.y * h))
    cv2.circle(canvas, c, 26, (0, 200, 0) if seen else (0, 0, 220), 3)  # ring: green = face seen, red = not
    cv2.circle(canvas, c, 14, (255, 255, 255), -1)
    # ponytail: fixed text rows can sit under a random validation dot; harmless, the dot is drawn on top
    _text(canvas, f"Dot {n + 1}/{total}: look at it" if seen else "No face detected: move into view / add light",
          int(0.3 * h), 0.9, (0, 220, 0) if seen else (0, 0, 255))
    _text(canvas, "Esc / Q: quit      S: skip (head-pose only)", int(0.7 * h), 0.7)
    cv2.imshow(_WIN, canvas)
    k = cv2.waitKey(1) & 0xFF
    return 27 if _closed(_WIN) else k


def _result(calib, mode: str) -> str:
    """Show the outcome; returns 'continue' | 'retry' | 'quit'. A good calibration continues by itself."""
    import cv2
    import numpy as np

    h, w = _CANVAS
    ok = calib is not None and (calib.accepted or mode == "baseline")
    if calib is None:
        msg = "No face seen during calibration"
    elif ok:
        msg = "Baseline captured (head-pose only)" if mode == "baseline" else f"Calibration OK (error {calib.error:.2f})"
    else:
        e = f" (error {calib.error:.2f})" if calib.error == calib.error else ""
        msg = f"Gaze fit rejected{e}: head-pose only"
    t0 = time.monotonic()
    while True:
        canvas = np.zeros((h, w, 3), np.uint8)
        _text(canvas, msg, h // 2 - 30, 1.0, (0, 220, 0) if ok else (0, 200, 255))
        _text(canvas, "Starting..." if ok else "R: retry      Enter: continue anyway      Esc: quit", h // 2 + 40)
        cv2.imshow(_WIN, canvas)
        k = cv2.waitKey(50) & 0xFF
        if _closed(_WIN) or k in _QUIT:
            return "quit"
        if k == ord("r") and not ok:
            return "retry"
        if k in (13, 32) or (ok and time.monotonic() - t0 > 1.2):
            return "continue"


def _calibrate_interactive(cap, lm, cfg, mode: str):
    """Fullscreen dots, adaptive dwell, live face feedback. Retry is the user's choice (R), not automatic.
    Returns (Calibration | None, quit)."""
    import cv2

    from proctorlens.perception.gaze import fit_calibration
    from proctorlens.pipeline.calibration import collect_adaptive, dot_schedule

    def read():
        ok, frame = cap.read()
        return frame if ok else None

    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    tries = 0
    try:
        while True:
            tries += 1
            samples, action = collect_adaptive(lm.process, read, _show_dot, cfg, dot_schedule(cfg.gaze, mode=mode))
            if action == "quit":
                return None, True
            calib = fit_calibration(samples, cfg.gaze) if any(s.phase in ("neutral", "grid") for s in samples) else None
            if calib:
                calib.tries = tries
            if action == "skip":
                return calib, False
            choice = _result(calib, mode)
            if choice == "quit":
                return None, True
            if choice == "continue":
                return calib, False
    finally:
        cv2.destroyWindow(_WIN)


def _calibrate_session(camera: int, cfg, mode: str):
    """Open camera + landmarker, calibrate, release the camera. Returns (Calibration | None, landmarker, quit)."""
    import cv2
    import numpy as np

    from proctorlens.perception.landmarks import Landmarker

    cap = _open_camera(camera)
    try:
        lm = Landmarker(cfg.models.landmarker)
        lm.process(np.zeros((480, 640, 3), np.uint8), 0)  # load models now, not during the first dot
        calib, quit_ = _calibrate_interactive(cap, lm, cfg, mode)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return calib, lm, quit_


def _calibrate(a: argparse.Namespace) -> int:
    calib, _, quit_ = _calibrate_session(a.camera, _cfg(a), a.calib_mode)
    if calib is None:
        print("calibration cancelled" if quit_ else "calibration failed: no face was seen")
        return 1
    calib.save(a.out)
    err = "n/a" if calib.error != calib.error else f"{calib.error:.3f}"
    print(f"calibration {calib.mode}: median validation error {err}, accepted={calib.accepted}, "
          f"tries={calib.tries} -> {a.out}")
    return 0


class _LatestFrame:
    """Latest-frame-wins camera reader: a thread keeps only the newest frame, so a slow pipeline drops frames, not time."""

    def __init__(self, cap):
        self.cap, self._slot, self._lock, self.stop = cap, None, threading.Lock(), False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok:  # unplugged: no frames arrive; the pipeline's tick() turns that into frame_gap
                time.sleep(0.05)
                continue
            with self._lock:
                self._slot = (time.monotonic(), frame)

    def take(self):
        with self._lock:
            s, self._slot = self._slot, None
        return s


def _fps(times) -> float:
    """Frames per second over a window of capture times (seconds); NaN until two distinct times."""
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else math.nan


def _live(a: argparse.Namespace) -> int:
    import cv2

    import numpy as np

    from proctorlens.explain.overlays import draw
    from proctorlens.perception import Perceiver
    from proctorlens.perception.gaze import Calibration
    from proctorlens.pipeline.runner import Pipeline

    cfg, lm = _cfg(a), None
    if a.calib:
        calib = Calibration.load(a.calib)
    else:
        calib, lm, quit_ = _calibrate_session(a.camera, cfg, a.calib_mode)
        if quit_:
            print("cancelled")
            return 1
        if calib:  # reuse next time and skip the calibration wait entirely
            calib.save("calib.json")
            print("saved calibration -> calib.json (next time: --calib calib.json)")
    cap = _open_camera(a.camera)
    win, loading = "proctorlens", np.zeros((360, 640, 3), np.uint8)
    cv2.putText(loading, "Loading models...", (180, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow(win, loading)  # the window exists (and can be closed) before the slow first-frame model loads
    cv2.waitKey(1)
    perc = Perceiver(cfg, landmarker=lm)  # reuses the calibration's landmarker
    perc.perceive(np.zeros((480, 640, 3), np.uint8), 0, 0)  # load YOLO/identity now
    pl, t0 = Pipeline(cfg, perc, calib), time.monotonic()
    cam, times = _LatestFrame(cap), collections.deque(maxlen=30)  # recent processed-frame times -> HUD fps
    try:
        while True:
            s = cam.take()
            if s is None:
                pl.tick(int((time.monotonic() - t0) * 1000))
                if cv2.waitKey(5) & 0xFF in _QUIT or _closed(win):
                    break
                continue
            pl.process(s[1], int((s[0] - t0) * 1000))
            times.append(s[0])
            img = draw(s[1], pl.last_perceived, pl.last_row, pl.ongoing(), fps=_fps(times),
                       zone=(pl.last_row or {}).get("zone"), calib_mode=calib.mode if calib else "none")
            cv2.putText(img, "q / Esc: quit", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF in _QUIT or _closed(win):
                break
    finally:
        cam.stop = True
        events = pl.finish()
        cap.release()
        cv2.destroyAllWindows()
    if a.out:  # live writes nothing unless asked
        df = pl.frame()
        _write(Path(a.out), "", df, events, cfg, calib)
        _summary(Path(a.out), df, events, calib)
    return 0


Check = tuple[str, str, str]  # (PASS | FAIL | WARN | SKIP, name, detail); only FAIL makes `doctor` exit non-zero
_DEPS = (  # (import name, required, what it is for when optional)
    ("numpy", True, ""), ("pandas", True, ""), ("cv2", True, ""), ("mediapipe", True, ""), ("ultralytics", True, ""),
    ("insightface", False, "identity checks"), ("torch", False, "tcn scorer / training"),
    ("lightgbm", False, "gbm scorer / training"), ("pyarrow", False, "parquet; csv is the fallback"))
_LM_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_LICENSES = ("MediaPipe Face Landmarker: Google model bundle, see its model card for terms",
             "Ultralytics YOLO: AGPL-3.0 (fine for coursework; check before redistributing)",
             "InsightFace pretrained weights: non-commercial research use only")


def check_deps(import_fn=importlib.import_module) -> list[Check]:
    """Import every library the pipeline can use and report its version. Optional ones only WARN."""
    out: list[Check] = []
    for name, required, why in _DEPS:
        try:
            out.append(("PASS", name, str(getattr(import_fn(name), "__version__", "?"))))
        except Exception as e:  # ImportError, or a native library that fails to load
            out.append(("FAIL" if required else "WARN", name, f"{type(e).__name__}: {e}" + ("" if required else f" (optional: {why})")))
    return out


def check_config(paths: list[str]) -> tuple[list[Check], object]:
    """Load the config; returns (checks, Config | None)."""
    from proctorlens.core.config import config_hash, load_config

    try:
        cfg = load_config(*paths)
    except Exception as e:
        return [("FAIL", "config", f"{type(e).__name__}: {e}")], None
    return [("PASS", "config", f"{len(paths)} file(s), hash {config_hash(cfg)}")], cfg


def _models(cfg) -> list[tuple[str, str, bool]]:
    """(name, configured path, present) for the three model files; identity is a folder holding .onnx files."""
    m, d = cfg.models, Path(cfg.models.identity)
    return [("landmarker", m.landmarker, Path(m.landmarker).is_file()), ("detector", m.detector, Path(m.detector).is_file()),
            ("identity", m.identity, d.is_dir() and any(d.glob("*.onnx")))]


def check_models(cfg) -> list[Check]:
    """Model files from cfg.models exist. Identity is optional (checks are skipped without it): WARN, not FAIL."""
    return [("PASS", f"model {n}", p) if ok else
            ("WARN" if n == "identity" else "FAIL", f"model {n}", f"{p} missing; run `proctorlens fetch-models`")
            for n, p, ok in _models(cfg)]


def assets_dir() -> Path | None:
    """Folder of the sample images that ship with ultralytics (None if it is not installed)."""
    try:
        import ultralytics
    except ImportError:
        return None
    d = Path(ultralytics.__file__).parent / "assets"
    return d if d.is_dir() else None


def sample_images() -> dict:
    """Ultralytics' bundled images: the zidane frame, the one face in it that MediaPipe finds (a crop; the two faces
    in the full frame are ~150 px tall and it finds none), and the bus. {} if the package or the files are missing."""
    import cv2

    d = assets_dir()
    z, b = (cv2.imread(str(d / n)) if d else None for n in ("zidane.jpg", "bus.jpg"))
    return {} if z is None or b is None else {"zidane_crop": z[100:460, 650:1050], "zidane": z, "bus": b}


def _try(name: str, fn) -> Check:
    """Run fn() -> (ok, detail); an exception is a FAIL with its message."""
    try:
        ok, detail = fn()
    except Exception as e:
        return ("FAIL", name, f"{type(e).__name__}: {e}")
    return ("PASS" if ok else "FAIL", name, detail)


def selftest(cfg, imgs: dict, lm=None, det=None, ident=None) -> list[Check]:
    """Run the models on sample images: landmarker finds >= 1 face in the zidane crop, detector >= 2 persons in the
    bus image, identity embeds that face (only if the identity model exists). lm/det/ident are injectable fakes."""
    if not imgs:
        return [("SKIP", "selftest", "ultralytics sample images not found")]
    crop, bus = imgs["zidane_crop"], imgs["bus"]
    faces: list = []

    def land():
        nonlocal lm
        from proctorlens.perception.landmarks import Landmarker

        lm = lm or Landmarker(cfg.models.landmarker)
        faces.extend(lm.process(crop, 0))
        return bool(faces), f"{len(faces)} face(s) in the zidane crop" + (f", yaw {faces[0].yaw:.1f}" if faces else "")

    def detect():
        from proctorlens.perception.objects import ObjectDetector

        n = len((det or ObjectDetector(cfg.models.detector, cfg.models.detector_conf)).process(bus).person_boxes)
        return n >= 2, f"{n} person(s) in bus.jpg (expect >= 2)"

    def ident_():
        from proctorlens.perception.identity import IdentityChecker

        e = (ident or IdentityChecker(cfg.models.identity)).embed(crop, faces[0])
        return e is not None, "no face embedded" if e is None else f"{len(e)}-d embedding"

    out = [_try("selftest landmarker", land), _try("selftest detector", detect)]
    if ident is None and not _models(cfg)[-1][2]:  # [-1] = identity, [2] = present
        out.append(("SKIP", "selftest identity", "no identity model (optional)"))
    elif not faces:
        out.append(("SKIP", "selftest identity", "needs a face from the landmarker"))
    else:
        out.append(_try("selftest identity", ident_))
    return out


def camera_report(cap, lm, cfg, seconds: float = 3.0, clock=time.monotonic) -> list[Check]:
    """Read ~`seconds` of frames from an open capture and report whether a face was seen, mean luma and blur, with
    advice. Never FAILs on image content (a setup hint, not a requirement); FAILs only when no frame arrives."""
    from proctorlens.perception.quality import assess

    t0, n, seen, luma, blur = clock(), 0, 0, 0.0, 0.0
    while clock() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        faces = lm.process(frame, int((clock() - t0) * 1000))
        q = assess(frame, faces[0].bbox if faces else None, cfg.quality)
        n, seen, luma, blur = n + 1, seen + bool(faces), luma + q.luma, blur + q.blur
    if not n:
        return [("FAIL", "camera", "no frames received: check that no other app is using it")]
    luma, blur, c = luma / n, blur / n, cfg.quality
    return [
        ("PASS", "camera face", f"seen in {seen}/{n} frames") if seen >= n / 2 else
        ("WARN", "camera face", f"seen in {seen}/{n} frames: face the camera, add light in front of you, move closer"),
        ("PASS", "camera luma", f"{luma:.0f}") if c.min_luma <= luma <= c.max_luma else
        ("WARN", "camera luma", f"{luma:.0f} (want {c.min_luma:.0f}-{c.max_luma:.0f}): " +
         ("too dark, add light" if luma < c.min_luma else "too bright, reduce backlight or exposure")),
        ("PASS", "camera blur", f"{blur:.0f}") if blur >= c.min_blur else
        ("WARN", "camera blur", f"{blur:.0f} (want >= {c.min_blur:.0f}): soft image, clean the lens, refocus, add light")]


def _camera_checks(index: int, cfg) -> list[Check]:
    """Opt-in (`doctor --camera N`): opens the real camera for ~3 s."""
    from proctorlens.perception.landmarks import Landmarker

    try:
        cap = _open_camera(index)
    except RuntimeError as e:
        return [("FAIL", "camera", str(e))]
    try:
        return camera_report(cap, Landmarker(cfg.models.landmarker), cfg)
    finally:
        cap.release()


def _doctor(a: argparse.Namespace) -> int:
    checks = check_deps()
    c, cfg = check_config(a.config or [p for p in _DEFAULT_CFGS if Path(p).is_file()])
    checks += c
    if cfg is None:
        checks.append(("SKIP", "models", "no config"))
    else:
        checks += check_models(cfg)
        if a.selftest:
            checks += selftest(cfg, sample_images())
        if a.camera is not None:
            checks += _camera_checks(a.camera, cfg)
    for s, name, detail in checks:
        print(f"{s} {name}: {detail}")
    bad = sum(s == "FAIL" for s, _, _ in checks)
    print(f"{bad} check(s) failed" if bad else "all required checks passed")
    return 1 if bad else 0


def plan_fetch(cfg) -> list[tuple[str, str, str | None]]:
    """Missing model files as (name, configured path, source); source None = no standard download (fetch by hand).
    A file that exists is never listed, so it is never overwritten."""
    out = []
    for name, path, present in _models(cfg):
        if present:
            continue
        base = Path(path).name
        src = {"landmarker": _LM_URL, "identity": f"insightface {base}",
               "detector": "ultralytics yolo11n.pt" if base == "yolo11n.pt" else None}[name]
        out.append((name, path, src))
    return out


def _fetch(name: str, path: str, src: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if name == "landmarker":
        import urllib.request

        tmp = p.with_name(p.name + ".part")  # a failed download must not leave a half file that looks present
        try:
            urllib.request.urlretrieve(src, tmp)
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
    elif name == "detector":
        from ultralytics import YOLO

        YOLO(str(p))  # a missing yolo11n.pt is downloaded to exactly this path
    else:
        from insightface.app import FaceAnalysis

        FaceAnalysis(name=p.name, root=str(p.parent.parent), providers=["CPUExecutionProvider"],
                     allowed_modules=["detection", "recognition"])  # downloads <root>/models/<name> when missing


def _fetch_models(a: argparse.Namespace) -> int:
    from proctorlens.core.config import load_config

    plan = plan_fetch(load_config(*(a.config or [p for p in _DEFAULT_CFGS if Path(p).is_file()])))
    print("licenses:\n  " + "\n  ".join(_LICENSES))
    if not plan:
        print("nothing to fetch: all model files are present")
        return 0
    rc = 0
    for name, path, src in plan:
        if src is None:
            print(f"manual: {name} -> {path} (not a standard download; export or place it yourself, see README)")
        elif a.dry_run:
            print(f"would fetch: {name} -> {path} [{src}]")
        else:
            print(f"fetching: {name} -> {path} [{src}]")
            try:
                _fetch(name, path, src)
            except Exception as e:
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                rc = 1
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="proctorlens", description="Webcam video -> observable, explainable events.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, help_: str, *, calib: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", action="append", default=[], metavar="PATH", help="YAML, repeatable; later wins")
        if calib:
            p.add_argument("--calib", metavar="calib.json", help="saved calibration (live: default = calibrate first)")
        p.set_defaults(fn=fn)
        return p

    p = add("replay", _replay, "run a video file through the pipeline", calib=True)
    p.add_argument("video")
    p.add_argument("--out", default="out", help="dir for features.parquet + events.json")
    p.add_argument("--render", metavar="out.mp4", help="also write an annotated video")
    p.add_argument("--report", action="store_true",
                   help="also write report.html, segments.json and per-event clips/keyframes (from --render if given)")
    mode = dict(choices=["full", "quick", "baseline"], default="full",
                help="full: 3x3 grid (~15 s); quick: 5 dots (~8 s); baseline: centre only (~3 s, head-pose only)")
    p = add("live", _live, "live camera window (calibrates first unless --calib)", calib=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--calib-mode", **mode)
    p.add_argument("--out", help="write features/events here on exit (default: write nothing)")
    p = add("calibrate", _calibrate, "run the gaze calibration and save it")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--calib-mode", **mode)
    p.add_argument("--out", default="calib.json")
    p = add("extract-features", _extract, "batch replay every video in a directory (+ <stem>.calib.json)")
    p.add_argument("dir")
    p.add_argument("--out", default="data/processed")
    p = add("report", _report, "review segments + offline report.html (+ clips/keyframes with --video) for a replay dir")
    p.add_argument("out", help="a `replay --out` dir (events.json, features.parquet)")
    p.add_argument("--video", help="video to cut clips/keyframes from: the source, or the --render overlay video")
    p = add("serve", _serve, "web app: candidate exam + proctor dashboard (pip extra 'web'; build frontend/ first)")
    p.add_argument("--host", default="127.0.0.1", help="default: this machine only (cameras need localhost or HTTPS)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data", default="data", help="SQLite DB + data/sessions/<id>/ (features, events, evidence)")
    p.add_argument("--frontend", default="frontend/dist", help="built frontend served at /")
    p = add("doctor", _doctor, "check libraries, model files and config (PASS/FAIL per line)")
    p.add_argument("--selftest", action="store_true", help="run the models on ultralytics' bundled sample images")
    p.add_argument("--camera", type=int, metavar="N", help="opt-in: open camera N for ~3 s; report face / light / focus")
    p = add("fetch-models", _fetch_models, "download only the model files that are missing (never overwrites)")
    p.add_argument("--dry-run", action="store_true", help="list what would be fetched, download nothing")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
