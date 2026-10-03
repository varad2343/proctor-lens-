import cv2
import numpy as np

from proctorlens.core.config import QualityCfg
from proctorlens.perception.quality import assess

CFG = QualityCfg()
FACE = (0.3, 0.2, 0.7, 0.8)


def _noise(lo=60, hi=200):
    return np.random.default_rng(0).integers(lo, hi, (240, 320, 3)).astype(np.uint8)


def _checker():
    yy, xx = np.mgrid[0:240, 0:320]
    g = np.where(((yy // 40) + (xx // 40)) % 2, 200, 50).astype(np.uint8)
    return cv2.merge([g, g, g])


def test_quality_reasons():
    ok = assess(_noise(), FACE, CFG)
    assert ok.reasons == [] and ok.quality >= CFG.min_quality and abs(ok.face_frac - 0.6) < 1e-9
    # no face: a code only, quality stays high so FACE_ABSENT is distinguishable from blindness
    nf = assess(_noise(), None, CFG)
    assert nf.reasons == ["no_face"] and nf.quality >= CFG.min_quality and nf.face_frac == 0.0
    dark = assess((_noise(0, 60) * 0.5).astype(np.uint8), FACE, CFG)
    assert "dark" in dark.reasons and dark.quality < CFG.min_quality
    bright = assess(_noise(235, 256), FACE, CFG)
    assert "bright" in bright.reasons and bright.quality < CFG.min_quality
    blur = assess(cv2.GaussianBlur(_checker(), (0, 0), 10), FACE, CFG)
    assert blur.reasons == ["blur"] and blur.quality < CFG.min_quality
    assert assess(_checker(), FACE, CFG).reasons == []  # sharp edges: not blurry
    blocked = assess(np.full((240, 320, 3), 100, np.uint8), FACE, CFG)
    assert "blocked" in blocked.reasons and blocked.quality == 0.0
    small = assess(_noise(), (0.45, 0.45, 0.5, 0.5), CFG)
    assert small.reasons == ["small_face"] and small.quality < CFG.min_quality
    cut = assess(_noise(), (-0.1, 0.2, 0.3, 0.8), CFG)
    assert cut.reasons == ["face_cut"] and cut.quality < CFG.min_quality
