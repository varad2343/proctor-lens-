import itertools

import numpy as np

from proctorlens.perception.head_pose import euler_from_matrix, rotation_matrix


def test_round_trip_and_4x4():
    for y, p, r in itertools.product((-80, -30, 0, 45, 80), (-60, -10, 0, 25, 60), (-90, -20, 0, 35, 120)):
        got = euler_from_matrix(rotation_matrix(y, p, r))
        assert np.allclose(got, (y, p, r), atol=1e-6), ((y, p, r), got)
    m = np.eye(4)
    m[:3, :3] = 1.3 * rotation_matrix(20, -10, 5)  # uniform scale + translation, as in MediaPipe matrices
    m[:3, 3] = (1, 2, -30)
    assert np.allclose(euler_from_matrix(m), (20, -10, 5), atol=1e-6)


def test_sign_convention():
    """Face axes: forward = +z (toward camera), up = +y; image-right = +x."""
    fwd, up = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
    assert (rotation_matrix(30, 0, 0) @ fwd)[0] > 0.4  # yaw>0: nose toward image-right
    assert (rotation_matrix(-30, 0, 0) @ fwd)[0] < -0.4
    assert (rotation_matrix(0, 30, 0) @ fwd)[1] > 0.4  # pitch>0: nose up
    assert (rotation_matrix(0, 0, 30) @ up)[0] > 0.4  # roll>0: top of head toward image-right = clockwise
    assert np.allclose(rotation_matrix(0, 0, 0), np.eye(3))
