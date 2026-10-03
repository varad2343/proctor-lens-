import math

import numpy as np

from proctorlens.features.schema import MODEL_COLUMNS
from proctorlens.features.windows import WindowBuffer, stat_names, window_stats


def test_window_buffer_padding_mask_and_eviction():
    b = WindowBuffer(5, ["a", "b"])
    assert len(b) == 0 and b.array().shape == (5, 2) and not b.mask().any()
    b.push({"frame_valid": True, "a": 1, "b": 2.0})
    b.push({"frame_valid": False, "a": 3, "b": 4})  # invalid frame
    b.push({"frame_valid": True, "a": math.nan, "b": True})  # non-finite
    b.push({"a": 7})  # missing column -> NaN
    A, m = b.array(), b.mask()
    assert len(b) == 4 and A.dtype == np.float32 and m.dtype == bool
    assert A.tolist() == [[0, 0], [1, 2], [3, 4], [0, 1], [7, 0]]  # front zero-padded, NaN -> 0
    assert m.tolist() == [False, True, False, False, False]
    for i in range(6):
        b.push({"frame_valid": True, "a": 10 + i, "b": 0.0})
    assert len(b) == 5 and b.array()[-1, 0] == 15 and b.array()[0, 0] == 11 and b.mask().all()


def test_window_stats_values_mask_and_names():
    cols = ["mouth_energy_1s", "off_screen_score", "x"]
    W = np.c_[[0, 1, 0, 1, 0, 1], [0.2, 0.8, 0.8, 0.2, 0.9, 0.9], np.arange(6) * 2.0].astype(float)
    full = np.ones(6, bool)
    s = window_stats(W, full, cols)
    names = stat_names(cols)
    assert len(s) == len(names) == 3 * 7 + 2 and s.dtype == np.float64
    d = dict(zip(names, s))
    assert d["x__mean"] == 5.0 and d["x__min"] == 0 and d["x__max"] == 10 and d["x__last"] == 10
    assert abs(d["x__slope"] - 2.0) < 1e-12 and abs(d["x__std"] - np.std(W[:, 2])) < 1e-12
    assert d["x__frac_above_med"] == 0.5
    assert d["mouth_energy_1s__zcr"] == 1.0  # alternates around its mean at every step
    assert d["off_screen_score__cross05"] == 3
    # masked / non-finite steps are ignored (indices keep their place, so the slope is unchanged)
    W2, m2 = W.copy(), full.copy()
    m2[0], W2[3, 2] = False, math.nan
    d2 = dict(zip(names, window_stats(W2, m2, cols)))
    assert d2["x__mean"] == (2 + 4 + 8 + 10) / 4 and abs(d2["x__slope"] - 2.0) < 1e-12
    # nothing valid -> all NaN, no warnings/exceptions; deterministic
    none = window_stats(W, np.zeros(6, bool), cols)
    assert np.isnan(none).all() and len(none) == len(s)
    assert np.array_equal(window_stats(W2, m2, cols), window_stats(W2, m2, cols), equal_nan=True)
    # names/values stay aligned with and without the extra stats, and for the real model columns
    assert len(window_stats(W[:, :2], full, ["p", "q"])) == len(stat_names(["p", "q"])) == 14
    big = np.random.default_rng(0).random((60, len(MODEL_COLUMNS)))
    assert len(window_stats(big, np.ones(60, bool), MODEL_COLUMNS)) == len(stat_names(MODEL_COLUMNS))
