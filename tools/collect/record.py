"""Cue recorder for ProctorLens-Sessions (spec 10.2): webcam -> video (no audio), fullscreen calibration dots, on-screen
cue prompts, cue log. Only record a participant who has given written informed consent.

python tools/collect/record.py --participant P01 --session 1 --consent-version v1 --cues cues.csv [--conditions "dim|glasses"]

Outputs in --out (default data/recordings); id = <participant>_s<session>:
  <id>.mp4        video on a constant-fps wall-clock timeline (video time == dot/cue time)
  <id>.calib.csv  calibration dot log (dot_id,phase,x,y,t_start_ms,t_end_ms) for pipeline.calibration.calibrate_from_video
  <id>.cues.csv   type,start_ms,end_ms,source,annotator_id,prompt. source=cue: the PLANNED interval, not corrected for
                  reaction latency; annotators correct the boundaries, then it becomes data/labels/<id>.csv
  <id>.meta.json  participant, consent_version, conditions, fps, resolution (read by ml.data.build_manifest)
Cue script CSV, one row per cue, played in file order (rows with block=scripted are shuffled among themselves, --seed):
  block,type,prompt,duration_s
  scripted,OFF_SCREEN_SUSTAINED,Look LEFT and hold,5
  scripted,PROHIBITED_OBJECT,Hold your phone up to the camera,4
  natural,natural,Answer the quiz in your other window,300
  nuisance,benign_thinking,Look up and think for a moment,3
`type` = an event type, `natural`/`nuisance` (honest blocks) or `benign_*`. Natural/nuisance cues shrink this window to a
corner so the quiz stays visible. Keys: q / ESC stops and saves what was recorded.
# ponytail: fixed prep countdown; no pause/redo of a single cue - re-record the session instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # run from a checkout without installing
from proctorlens.core.config import load_config  # noqa: E402
from proctorlens.pipeline.calibration import dot_schedule, write_log  # noqa: E402

WIN = "ProctorLens recorder"


class Stop(Exception):
    """q / ESC pressed."""


class Rec:
    """Camera + video writer + window. Video frames are padded to the wall clock so video time == session time."""

    def __init__(self, cap, writer, fps: float):
        self.cap, self.writer, self.fps = cap, writer, fps
        self.t0 = self.last = None
        self.n = self.fails = 0

    def grab(self, write: bool = True) -> float:
        """Read a camera frame, write video frames up to now, return ms since the first written frame."""
        ok, frame = self.cap.read()
        self.fails = 0 if ok else self.fails + 1
        if self.fails > 60:
            raise RuntimeError("camera lost (60 consecutive failed reads)")
        self.last = frame if ok else self.last
        if not write or self.last is None:
            return 0.0
        now = time.monotonic()
        self.t0 = now if self.t0 is None else self.t0
        while self.n < int((now - self.t0) * self.fps) + 1:
            self.writer.write(self.last)
            self.n += 1
        return (now - self.t0) * 1000

    def show(self, canvas: np.ndarray, preview: bool = True) -> int:
        if preview and self.last is not None:  # small camera inset for positioning
            h, w = canvas.shape[:2]
            pw = max(w // 6, 1)
            ph = pw * self.last.shape[0] // self.last.shape[1]
            canvas[h - ph - 10:h - 10, w - pw - 10:w - 10] = cv2.resize(self.last, (pw, ph))
        cv2.imshow(WIN, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            raise Stop
        return k


def text(canvas: np.ndarray, s: str, y: float, scale: float = 1.0) -> None:
    """Centred, wrapped text; y = fraction of the canvas height of the first line."""
    h, w = canvas.shape[:2]
    for i, line in enumerate(textwrap.wrap(s, max(int(w / (34 * scale)), 12)) or [""]):
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(canvas, line, ((w - tw) // 2, int(h * y) + i * int(th * 1.8)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)


def layout(small: bool, screen: tuple[int, int]):
    """(blank-canvas factory, text scale); small = corner window for the natural/nuisance blocks."""
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if small else cv2.WINDOW_FULLSCREEN)
    if small:
        cv2.resizeWindow(WIN, 480, 270)
        cv2.moveWindow(WIN, 0, 0)
    size = (270, 480) if small else (screen[1], screen[0])
    return (lambda: np.zeros((*size, 3), np.uint8)), (0.6 if small else 1.6)


def read_cues(path: str, seed: int) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        cues = list(csv.DictReader(f))
    for c in cues:
        c["duration_s"] = float(c["duration_s"])
        c["block"] = c["block"].strip()
    idx = [i for i, c in enumerate(cues) if c["block"] == "scripted"]
    shuffled = [cues[i] for i in idx]
    random.Random(seed).shuffle(shuffled)
    for i, c in zip(idx, shuffled):
        cues[i] = c
    return cues


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="pseudonymous id, e.g. P01")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--consent-version", required=True, help="id of the signed consent form version")
    ap.add_argument("--conditions", default="", help="free text, e.g. 'dim|glasses|cam-low'")
    ap.add_argument("--cues", required=True, help="cue script CSV")
    ap.add_argument("--out", default="data/recordings")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--screen", default="1920x1080", help="canvas WxH; the window scales it, dot positions stay relative")
    ap.add_argument("--prep-s", type=float, default=3.0, help="'next cue' countdown before each cue (not logged)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", nargs="*", default=[], help="config YAMLs (only gaze.* calibration timing is used)")
    a = ap.parse_args(argv)
    cfg, cues = load_config(*a.config), read_cues(a.cues, a.seed)
    rid, out = f"{a.participant}_s{a.session}", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{rid}.mp4").exists():
        sys.exit(f"{rid}.mp4 already exists; pick another --session (never overwrite consented data)")
    cap = cv2.VideoCapture(a.camera)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"camera {a.camera} not available")
    size = (frame.shape[1], frame.shape[0])
    writer = cv2.VideoWriter(str(out / f"{rid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, size)
    screen = tuple(int(v) for v in a.screen.lower().split("x"))
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    rec, log, dots = Rec(cap, writer, a.fps), [], dot_schedule(cfg.gaze, a.seed)
    try:
        blank, sc = layout(False, screen)
        while True:  # 1. waiting screen (nothing is written yet)
            c = blank()
            text(c, f"{a.participant}: video only, no audio. Press SPACE to start calibration, Q to quit", 0.4, sc)
            rec.grab(write=False)
            if rec.show(c) == 32:
                break
        end = dots[-1].t_end_ms  # 2. calibration dots; dot times are session times
        while (t := rec.grab()) < end:
            d = next((d for d in dots if d.t_start_ms <= t < d.t_end_ms), dots[-1])
            c = blank()
            cv2.circle(c, (int(d.x * screen[0]), int(d.y * screen[1])), 18, (0, 255, 255), -1)
            cv2.circle(c, (int(d.x * screen[0]), int(d.y * screen[1])), 3, (0, 0, 0), -1)
            rec.show(c, preview=False)
        log.append(dict(type="calibration", start_ms=0, end_ms=int(end), prompt=""))
        small = False
        for cue in cues:  # 3. cues: prep countdown (not logged), then the cue (logged)
            if (now_small := cue["block"] in ("natural", "nuisance")) != small:  # window mode changed
                blank, sc = layout(now_small, screen)
                small = now_small
            for title, dur in (("NEXT", a.prep_s), ("NOW", cue["duration_s"])):
                start = rec.grab()
                while (t := rec.grab()) < start + dur * 1000:
                    c = blank()
                    text(c, title, 0.25, sc * 1.4)
                    text(c, cue["prompt"], 0.45, sc)
                    text(c, f"{(start + dur * 1000 - t) / 1000:.0f} s", 0.8, sc)
                    rec.show(c)
            log.append(dict(type=cue["type"], start_ms=int(start), end_ms=int(t), prompt=cue["prompt"]))
    except Stop:
        pass
    finally:  # also on a lost camera: keep what was recorded
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        if rec.n == 0:
            (out / f"{rid}.mp4").unlink(missing_ok=True)  # quit before recording started: leave nothing behind
        else:
            with open(out / f"{rid}.cues.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, ["type", "start_ms", "end_ms", "source", "annotator_id", "prompt"])
                w.writeheader()
                w.writerows({"source": "cue", "annotator_id": "", **r} for r in log)
            write_log(dots, out / f"{rid}.calib.csv")
            (out / f"{rid}.meta.json").write_text(json.dumps(dict(
                participant=a.participant, session=a.session, consent_version=a.consent_version,
                conditions=a.conditions, fps=a.fps, resolution=size, camera=a.camera, screen=a.screen, seed=a.seed,
                n_frames=rec.n, recorded_at=datetime.now().isoformat(timespec="seconds")), indent=1), encoding="utf-8")
            print(f"saved {rid}: {rec.n} frames, {len(log)} logged intervals in {out}")


if __name__ == "__main__":
    main()
