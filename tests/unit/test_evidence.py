

def test_clip_without_ffmpeg_uses_media_foundation():
    """No ffmpeg on PATH: write_clip_jpegs falls back to OpenCV + Windows' H.264 encoder (False on other OSes)."""
    import sys
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np

    from proctorlens.explain import evidence

    jpegs = [cv2.imencode(".jpg", np.full((241, 321, 3), i * 10, np.uint8))[1].tobytes() for i in range(12)]  # odd size
    out = Path(tempfile.mkdtemp()) / "c.mp4"
    which, evidence.shutil.which = evidence.shutil.which, lambda _: None
    try:
        ok = evidence.write_clip_jpegs(jpegs, 10.0, out)
    finally:
        evidence.shutil.which = which
    assert ok or sys.platform != "win32"
    if ok:
        cap = cv2.VideoCapture(str(out))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 10 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
        cap.release()
