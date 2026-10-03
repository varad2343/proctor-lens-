import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from proctorlens.core.config import Config
from proctorlens.features.schema import ATTR_GROUPS
from proctorlens.temporal.scores_learned import LearnedScores, _Head, make_scores, write_model_dir
from proctorlens.temporal.scores_rule import RuleScores

NAN = math.nan
COLS = ["d_yaw", "iris_lx", "jaw_open", "quality"]  # one column from each attribution group
W = 5


class Buf:
    """Minimal stand-in for features.windows.WindowBuffer (push / array / mask / len)."""

    def __init__(self, window, cols):
        self.w, self.cols, self.rows, self.ok = window, cols, [], []

    def push(self, row):
        self.rows = (self.rows + [[float(row.get(c, NAN)) for c in self.cols]])[-self.w:]
        self.ok = (self.ok + [bool(row["frame_valid"])])[-self.w:]

    def array(self):
        a = np.zeros((self.w, len(self.cols)), np.float32)
        if self.rows:
            a[-len(self.rows):] = self.rows
        return a

    def mask(self):
        m = np.zeros(self.w, bool)
        if self.ok:
            m[-len(self.ok):] = self.ok
        return m

    def __len__(self):
        return len(self.rows)


def row(t, **kw):
    r = dict(t_ms=t, frame_valid=True, reliable=True, primary_face_present=True, turned_away=False, n_faces=1,
             n_persons=1, phone_conf=0.0, notes_conf=0.0, off_screen_score=0.0, zone="right",
             mouth_energy_1s=0.0, jaw_open=0.0, d_yaw=0.0, iris_lx=0.5, quality=0.9, id_similarity=NAN, id_quality_ok=False)
    r.update(kw)
    return r


def meta(target, **extra):
    return {"kind": "gbm", "target": target, "columns": COLS, "window": W, "calib": {"a": 1.0, "b": 0.0},
            "extra": extra}


def make(cfg=None, **extra):
    """off_screen logit depends only on the last step's d_yaw; speaking logit only on its jaw_open."""
    cfg = cfg or Config()
    heads = {
        "off_screen": _Head(meta("off_screen", **extra), lambda X, m: float(X[-1, 0]) * 0.2 - 2.0,
                            Buf(W, COLS)),
        "speaking": _Head(meta("speaking"), lambda X, m: float(X[-1, 2]) * 4.0 - 2.0, Buf(W, COLS)),
    }
    return LearnedScores(cfg, RuleScores(cfg), _heads=heads)


def test_learned_overrides_rule_after_window_fills_and_gates_none():
    ls = make()
    sc = [ls.update(row(i * 100, d_yaw=30.0))["off_screen"] for i in range(W + 1)]
    assert sc[:W - 1] == [0.0] * (W - 1)  # window not full: rule value (off_screen_score=0)
    assert abs(sc[W - 1] - 1 / (1 + math.exp(-4.0))) < 1e-6 and sc[W] == sc[W - 1]
    assert ls.update(row(600, d_yaw=30.0, reliable=False))["off_screen"] is None  # unreliable: rule None kept
    assert make().update(row(0, reliable=False))["off_screen"] is None
    ls = make(Config())
    ls.cfg.policy.allow_looking_down = True
    for i in range(W + 1):
        s = ls.update(row(i * 100, d_yaw=30.0, zone="down"))
    assert s["off_screen"] == 0.0


def test_undefined_model_input_keeps_rule_score():
    """No full calibration => gaze columns are NaN: the model must not run on an all-NaN window."""
    cfg, calls = Config(), []

    def predict(X, m):
        calls.append(1)
        return 4.0

    ls = LearnedScores(cfg, RuleScores(cfg), _heads={"off_screen": _Head(meta("off_screen"), predict)})  # real WindowBuffer
    for i in range(W + 2):
        s = ls.update(row(i * 100, d_yaw=NAN, off_screen_score=0.7))
    assert s["off_screen"] == 0.7 and not calls
    assert ls.update(row(900, d_yaw=10.0, off_screen_score=0.7))["off_screen"] > 0.9 and calls  # defined again


def test_learned_speaking_goes_through_window_rule():
    ls = make()
    sc = [ls.update(row(i * 100, jaw_open=1.0))["speaking"] for i in range(60)]
    assert sc[W - 2] == 0.0  # before the window is full: rule value
    p = 1 / (1 + math.exp(-2.0))
    assert abs(sc[W - 1] - p * 0.1 / 3) < 1e-9 and all(b >= a for a, b in zip(sc[W:], sc[W + 1:]))
    assert sc[-1] == 1.0
    ls = make()
    ls.cfg.policy.allow_reading_aloud = True
    assert [ls.update(row(i * 100, jaw_open=1.0))["speaking"] for i in range(10)][-1] == 0.0


def test_attribute_groups():
    ls = make()
    assert ls.attribute("off_screen") == {} and ls.attribute("nope") == {}  # window not full
    for i in range(W):
        ls.update(row(i * 100, d_yaw=30.0, jaw_open=1.0))
    a = ls.attribute("off_screen")
    assert set(a) == set(ATTR_GROUPS)
    p = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
    assert abs(a["head_pose"] - (p(4.0) - p(-2.0))) < 1e-6  # d_yaw zeroed -> logit -2
    assert a["eye"] == a["mouth"] == a["quality"] == 0.0
    assert ls.attribute("speaking")["mouth"] > 0.5 and ls.attribute("speaking")["head_pose"] == 0.0
    ls = make(feature_mean=[10.0, 0.5, 0.0, 0.9])  # neutral = training mean: logit 0 -> p 0.5
    for i in range(W):
        ls.update(row(i * 100, d_yaw=30.0))
    assert abs(ls.attribute("off_screen")["head_pose"] - (p(4.0) - 0.5)) < 1e-6


def test_make_scores_and_model_dir_meta():
    assert isinstance(make_scores(Config()), RuleScores)
    cfg = Config()
    cfg.scorer.provider = "gbm"
    try:
        make_scores(cfg)
        assert False, "gbm without models must raise"
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        p = write_model_dir(Path(d) / "v1", "gbm", "off_screen", COLS, 60, None, {"k": 1})
        m = json.loads((p / "meta.json").read_text())
        assert m == {"kind": "gbm", "target": "off_screen", "columns": COLS, "window": 60,
                     "calib": {"a": 1.0, "b": 0.0}, "extra": {"k": 1}, "version": "v1"}
        try:
            write_model_dir(Path(d) / "x", "svm", "off_screen", COLS, 60, None)
            assert False
        except ValueError:
            pass


def test_gbm_loader_and_real_window_buffer():
    try:  # needs the real features.windows; lightgbm is replaced by a stub Booster
        from proctorlens.features.windows import WindowBuffer, window_stats  # noqa: F401
    except ImportError:
        return
    seen = {}

    class Booster:
        def __init__(self, model_file):
            seen["file"] = model_file

        def predict(self, X, raw_score=False):
            seen["raw"], seen["n"] = raw_score, X.shape[1]
            return np.array([1.5])

    saved = sys.modules.get("lightgbm")
    sys.modules["lightgbm"] = types.SimpleNamespace(Booster=Booster)
    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.scorer.provider = "gbm"
            p = write_model_dir(Path(d) / "v1", "gbm", "off_screen", COLS, 4, {"a": 2.0, "b": -1.0})
            (p / "model.txt").write_text("stub")
            cfg.scorer.models = {"off_screen": str(p)}
            ls = make_scores(cfg)
            assert isinstance(ls, LearnedScores)
            s = [ls.update(row(i * 100, d_yaw=30.0))["off_screen"] for i in range(5)]
            assert s[2] == 0.0 and abs(s[3] - 1 / (1 + math.exp(-(2.0 * 1.5 - 1.0)))) < 1e-9
            buf = ls.heads["off_screen"].buf
            assert seen["raw"] is True and seen["n"] == len(window_stats(buf.array(), buf.mask(), COLS))
    finally:
        if saved is None:
            sys.modules.pop("lightgbm", None)
        else:
            sys.modules["lightgbm"] = saved


def test_tcn_shape_and_causality():
    try:
        import torch

        from proctorlens.temporal.tcn import build_tcn
    except (ImportError, OSError):  # OSError: torch installed but its native DLLs fail to load (Windows)
        return
    torch.manual_seed(0)
    net = build_tcn(6, channels=8, n_blocks=3, n_targets=2).eval()
    x = torch.randn(2, 40, 6)
    y = net(x)
    assert y.shape == (2, 40, 2)
    x2 = x.clone()
    x2[:, 25:] = torch.randn(2, 15, 6)  # change the future only
    y2 = net(x2)
    assert torch.allclose(y[:, :25], y2[:, :25], atol=1e-6) and not torch.allclose(y[:, 25:], y2[:, 25:])
    n = sum(p.numel() for p in build_tcn(35).parameters())
    assert 10_000 < n < 100_000
