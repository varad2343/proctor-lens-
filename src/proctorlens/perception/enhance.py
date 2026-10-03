"""Low-light enhancement of the landmarker input. quality.assess keeps seeing the ORIGINAL frame, so a dark
room still reports 'dark' and degrades monitoring; this only helps the landmarker find the face.

Chain (picked by measurement on darkened real face crops, see tests/unit/test_perception_accuracy.py):
light Gaussian blur (webcam sensor noise is amplified by the gamma step, without it detection collapses)
-> adaptive gamma towards a mean luma of _TARGET -> CLAHE on the L channel.
"""
from __future__ import annotations

import cv2
import numpy as np

_TARGET = 110.0  # mean luma the gamma stage aims for
_G_MIN = 0.2  # strongest gamma boost: a black frame is not amplified into noise
_SIGMA = 1.0  # denoise before boosting
_CLIP = 2.0  # CLAHE clip limit


def mean_luma(frame_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())


def gamma_lut(luma: float, target: float = _TARGET) -> np.ndarray:
    """256-entry uint8 LUT x -> 255*(x/255)^g; g in [_G_MIN, 1] is chosen so a frame of mean `luma` lands near `target`
    (g = 1, the identity, when the frame is already brighter than `target`)."""
    g = float(np.clip(np.log(target / 255) / np.log(max(luma, 1.0) / 255), _G_MIN, 1.0))
    return (255 * (np.arange(256) / 255) ** g).astype(np.uint8)


def lowlight_enhance(frame_bgr: np.ndarray, luma_below: float) -> np.ndarray:
    """The frame itself (same object) when its mean luma >= luma_below, else an enhanced copy."""
    luma = mean_luma(frame_bgr)
    if luma >= luma_below:
        return frame_bgr
    img = cv2.LUT(cv2.GaussianBlur(frame_bgr, (0, 0), _SIGMA), gamma_lut(luma))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=_CLIP, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
