"""Head pose <-> rotation matrix (degrees).

Frame = MediaPipe facial-transformation-matrix space (OpenGL style): x = image-right, y = image-up,
z = toward the camera; a face looking straight at the camera is the identity rotation.
  yaw   > 0: face turns toward image-right (nose -> +x)
  pitch > 0: face tilts up (nose -> +y)
  roll  > 0: clockwise in the image (top of head -> +x)
R = Ry(yaw) @ Rx(pitch) @ Rz(roll), signs baked into the matrices below. euler_from_matrix is the exact
inverse for |pitch| < 90.

VERIFIED on real MediaPipe Face Landmarker output (mediapipe 1.0.1, 11 faces from 5 photos: yaw -37..+19,
pitch -16..+3, roll -27..+15 deg); no axis is mirrored or flipped, so nothing was changed:
  - vs an independent 3D plane fit of the same landmarks (cheeks 234/454, forehead 10 / chin 152, depth toward the
    camera = -landmark z): same sign on every angle beyond 5 deg; corr 0.99 / 0.96 / 0.99 and mean |diff|
    2.4 / 3.8 / 1.1 deg for yaw / pitch / roll.
  - nose tip right of the eye-corner midpoint <=> yaw > 0 (corr 0.89); eye-line tilt tracks roll (corr 0.98).
  - horizontal flip: yaw and roll negate (mean |sum| 0.7 / 0.6 deg, max 1.8 / 2.4), pitch is kept (mean |diff| 1.1,
    max 4.0 deg).
  - image content rotated clockwise by theta (+-10, +-20 deg): roll changes by +theta, within 0.5 deg.
Re-checked by tests/unit/test_perception_accuracy.py whenever the model and sample images are present.
"""
from __future__ import annotations

import numpy as np


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """3x3 face->camera rotation for the convention above."""
    cy, cp, cr = np.cos(np.radians([yaw, pitch, roll]))
    sy, sp, sr = np.sin(np.radians([yaw, pitch, roll]))
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    p = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    r = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    return y @ p @ r


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a 3x3 or 4x4 matrix (uniform scale/translation are ignored)."""
    r = np.asarray(m, dtype=float)[:3, :3]
    r = r / np.linalg.norm(r, axis=0)
    yaw = np.arctan2(r[0, 2], r[2, 2])
    pitch = np.arcsin(np.clip(r[1, 2], -1.0, 1.0))
    roll = np.arctan2(-r[1, 0], r[1, 1])
    return tuple(float(v) for v in np.degrees([yaw, pitch, roll]))  # type: ignore[return-value]
