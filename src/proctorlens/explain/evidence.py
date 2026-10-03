"""Event evidence media (master spec 8.1, 8.3; docs/DECISIONS.md ADR-015, ADR-016): one browser-playable H.264 clip
per event, start - 5 s to end + 3 s (capped at 30 s), and three keyframes (onset, peak, end). Cut after replay from an
existing video file (the source recording, or the `--render` overlay video to get boxes and gaze arrows burned in), or
in the web app from the in-memory ring buffer of JPEG frames. Nothing else of the video is written."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from proctorlens.core.types import Event
from proctorlens.pipeline.runner import video_frames

PRE_MS, POST_MS, CAP_MS = 5000, 3000, 30000
# browser-playable H.264 (OpenCV's mp4v is not): even size, limited range, yuv420p, index up front
_ENCODE = ["-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=tv", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def clip_window(start_ms: int, end_ms: int) -> tuple[int, int]:
    a = max(0, start_ms - PRE_MS)
    return a, min(end_ms + POST_MS, a + CAP_MS)


def keyframe_times(ev: Event, step_ms: int) -> dict[str, int]:
    """onset / peak / end times of an event; end_ms is exclusive, so 'end' is its last grid step."""
    last = max(ev.start_ms, ev.end_ms - step_ms)
    pk = ev.details.get("peak_ms")
    return {"onset": ev.start_ms, "peak": ev.start_ms if pk is None else min(max(pk, ev.start_ms), last), "end": last}


def write_clip(video: str | Path, out: str | Path, a_ms: int, b_ms: int) -> bool:
    """Cut [a_ms, b_ms) of a video file via the ffmpeg binary. False when ffmpeg is not on PATH; a failure raises."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False
    subprocess.run([exe, "-y", "-loglevel", "error", "-ss", f"{a_ms / 1000:.3f}", "-i", str(video),
                    "-t", f"{(b_ms - a_ms) / 1000:.3f}", *_ENCODE, str(out)], check=True)
    return True


def write_clip_jpegs(jpegs: list[bytes], fps: float, out: str | Path) -> bool:
    """Same encoding from in-memory JPEG frames (the web app's ring buffer). False without ffmpeg or frames."""
    exe = shutil.which("ffmpeg")
    if exe is None or not jpegs:
        return False
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps:.3f}", "-c:v", "mjpeg",
                    "-i", "-", *_ENCODE, str(out)], input=b"".join(jpegs), check=True)
    return True


def grab_frames(video: str | Path, times_ms: list[int]) -> dict[int, np.ndarray]:
    """{t: first frame at or after t (the last frame for t past the end)} in one sequential read, with replay's own
    timestamps (seeking by CAP_PROP_POS_MSEC is unreliable on some containers)."""
    want, out, last = sorted(set(times_ms)), {}, None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    try:
        i = 0
        for t, frame in video_frames(cap):
            while i < len(want) and t >= want[i]:
                out[want[i]] = frame
                i += 1
            if i == len(want):
                break
            last = frame
    finally:
        cap.release()
    return out | ({t: last for t in want if t not in out} if last is not None else {})


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Copy of frame with text on a dark strip along the bottom."""
    img = frame.copy()
    h = img.shape[0]
    fs = max(0.4, h / 900)
    bar = int(30 * fs)
    img[h - bar:] = (40, 40, 40)
    cv2.putText(img, text, (6, h - int(9 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img
