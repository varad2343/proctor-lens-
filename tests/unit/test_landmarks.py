import numpy as np

from proctorlens.perception.head_pose import rotation_matrix
from proctorlens.perception.landmarks import face_from_landmarks


def _lm() -> np.ndarray:
    """478 points at the face centre, then the ones that matter placed by hand (bbox 0.3..0.7 x 0.2..0.8)."""
    lm = np.tile([0.5, 0.5, 0.0], (478, 1))
    pts = {10: (0.5, 0.2), 152: (0.5, 0.8), 234: (0.3, 0.5), 454: (0.7, 0.5),
           33: (0.40, 0.40), 133: (0.47, 0.40), 159: (0.435, 0.3925), 145: (0.435, 0.4075),  # subject's right eye
           362: (0.53, 0.40), 263: (0.60, 0.40), 386: (0.565, 0.3925), 374: (0.565, 0.4075),  # subject's left eye
           468: (0.40 + 0.25 * 0.07, 0.40), 473: (0.53 + 0.75 * 0.07, 0.40),  # iris x ratios .25 (R) / .75 (L)
           78: (0.44, 0.65), 308: (0.56, 0.65), 13: (0.50, 0.64), 14: (0.50, 0.66)}  # mouth
    for i, (x, y) in pts.items():
        lm[i, :2] = x, y
    return lm


def test_face_from_landmarks():
    m = np.eye(4)
    m[:3, :3] = rotation_matrix(20, -10, 5)
    f = face_from_landmarks(_lm(), {"jawOpen": 0.3, "eyeLookInLeft": 0.1}, m, conf=0.9)
    assert np.allclose(f.bbox, (0.3, 0.2, 0.7, 0.8)) and np.allclose(f.center, (0.5, 0.5))
    assert abs(f.scale - 0.6) < 1e-9 and f.conf == 0.9 and not f.static
    assert np.allclose((f.yaw, f.pitch, f.roll), (20, -10, 5), atol=1e-6)
    assert np.allclose(f.iris, (0.75, 0.5, 0.25, 0.5), atol=1e-6)  # (lx, ly, rx, ry)
    assert np.allclose(f.eye_open, (0.0150 / 0.07 / 0.36,) * 2, atol=1e-6) and all(0 <= v <= 1 for v in f.eye_open)  # no eyeBlink -> landmark EAR
    assert abs(f.blend["mar"] - 0.02 / 0.12) < 1e-6 and f.blend["jawOpen"] == 0.3
    # pose falls back to zeros without a matrix; aspect only rescales the ratios that mix x and y
    g = face_from_landmarks(_lm(), {}, None, aspect=16 / 9)
    assert (g.yaw, g.pitch, g.roll) == (0.0, 0.0, 0.0) and g.eye_open[0] < f.eye_open[0]
