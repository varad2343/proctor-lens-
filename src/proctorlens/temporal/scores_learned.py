"""Learned score provider: rule scores, with off_screen / speaking replaced by calibrated model probabilities.

Artifact dir (data/models/temporal/<name>/<version>/): meta.json + model.txt (lightgbm Booster) or model.pt
(torch state_dict). meta = {kind, target, columns, window, calib{a,b}, extra, version}.
  gbm: features = features.windows.window_stats(window, mask, columns); raw = Booster.predict(raw_score=True).
  tcn: extra = {"tcn": build_tcn kwargs, "feature_mean": [F], "feature_std": [F]}; input = (x - mean) / std
       with invalid/NaN steps set to 0 (the mean); raw = logit at the last step. ponytail: no mask channel.
  Platt: p = sigmoid(a * raw + b), a/b fitted on validation by the trainer.
Private hook for tests / custom predictors:
  LearnedScores(cfg, rule, _heads={target: _Head(meta, predict, buf)}),
  predict(X[window,F] float32, mask[window] bool) -> raw float.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.config import Config
from ..core.types import Scores
from ..features.schema import ATTR_GROUPS
from .scores_rule import RuleScores, SpeakingWindow

TARGETS = ("off_screen", "speaking")


def write_model_dir(path: str | Path, kind: str, target: str, columns: list[str], window: int,
                    calib: dict | None, extra: dict | None = None) -> Path:
    """Write meta.json; the caller saves the weights next to it (model.txt for gbm, model.pt for tcn)."""
    if kind not in ("gbm", "tcn") or target not in TARGETS:
        raise ValueError(f"bad kind/target: {kind!r}/{target!r}")
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "target": target, "columns": list(columns), "window": int(window),
            "calib": calib or {"a": 1.0, "b": 0.0}, "extra": extra or {}, "version": p.name}
    (p / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return p


class _Head:
    """One target's model + its causal window buffer."""

    def __init__(self, meta: dict, predict: Callable[[np.ndarray, np.ndarray], float], buf=None):
        self.meta, self.predict = meta, predict
        self.cols, self.window = list(meta["columns"]), int(meta["window"])
        c = meta.get("calib") or {}
        self.a, self.b = float(c.get("a", 1.0)), float(c.get("b", 0.0))
        mean = (meta.get("extra") or {}).get("feature_mean")
        # neutral value used by occlusion attribution; ponytail: 0 when the trainer stored no feature_mean
        self.neutral = np.asarray(mean if mean is not None else np.zeros(len(self.cols)), dtype=np.float32)
        if buf is None:
            from ..features.windows import WindowBuffer

            buf = WindowBuffer(self.window, self.cols)
        self.buf = buf

    def ready(self) -> bool:
        """Window full and its last step usable (frame valid, every model column finite): exactly the samples the
        trainers use. So a user without a full gaze calibration (gaze_x/gaze_y NaN) keeps the rule score."""
        return len(self.buf) >= self.window and bool(self.buf.mask()[-1])

    def prob(self, x: np.ndarray | None = None) -> float:
        z = self.a * self.predict(self.buf.array() if x is None else x, self.buf.mask()) + self.b
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0))))


def _load_head(path: str | Path) -> _Head:
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    cols = meta["columns"]
    if meta["kind"] == "gbm":
        import lightgbm as lgb

        from ..features.windows import window_stats

        bst = lgb.Booster(model_file=str(p / "model.txt"))

        def predict(x, m):
            return float(bst.predict(window_stats(x, m, cols)[None, :], raw_score=True)[0])
    else:
        import torch

        from .tcn import build_tcn

        ex = meta["extra"]
        net = build_tcn(**ex["tcn"])
        net.load_state_dict(torch.load(p / "model.pt", map_location="cpu", weights_only=True))
        net.eval()
        mu = np.asarray(ex["feature_mean"], np.float32)
        sd = np.maximum(np.asarray(ex["feature_std"], np.float32), 1e-6)

        def predict(x, m):
            z = np.nan_to_num((x - mu) / sd).astype(np.float32)
            z[~m] = 0.0
            with torch.no_grad():
                return float(net(torch.from_numpy(z[None]))[0, -1, 0])
    return _Head(meta, predict)


class LearnedScores:
    """update(row) -> Scores. Falls back to the rule value (None included) while a window is not full or its last
    step has an undefined model input (e.g. gaze columns without a full calibration)."""

    def __init__(self, cfg: Config, rule: RuleScores, _heads: dict[str, _Head] | None = None):
        self.cfg, self.rule = cfg, rule
        if _heads is None:
            if not cfg.scorer.models:
                raise ValueError(f"scorer.provider={cfg.scorer.provider!r} needs scorer.models")
            _heads = {t: _load_head(d) for t, d in cfg.scorer.models.items()}
            for t, h in _heads.items():
                if t not in TARGETS or h.meta["target"] != t or h.meta["kind"] != cfg.scorer.provider:
                    raise ValueError(f"scorer.models[{t!r}] is a {h.meta['kind']}/{h.meta['target']} model")
        self.heads = _heads
        self._speak = SpeakingWindow(cfg.pipeline.grid_hz)

    def update(self, row: dict) -> Scores:
        s, pol = self.rule.update(row), self.cfg.policy
        for h in self.heads.values():
            h.buf.push(row)
        h = self.heads.get("off_screen")
        if h and s["off_screen"] is not None and h.ready():
            s["off_screen"] = 0.0 if pol.allow_looking_down and row.get("zone") == "down" else h.prob()
        h = self.heads.get("speaking")
        if h and h.ready():  # per-step probability -> the same 10 s window rule as the rule provider
            v = self._speak.update(None if s["speaking"] is None else h.prob())
            s["speaking"] = 0.0 if v is not None and pol.allow_reading_aloud else v
        return s

    def attribute(self, target: str) -> dict[str, float]:
        """Feature-group occlusion over the CURRENT window: score drop when a group is set to its neutral
        value (positive = the group supports the score). {} when the window is not full."""
        h = self.heads.get(target)
        if h is None or not h.ready():
            return {}
        x, base, out = h.buf.array(), h.prob(), {}
        for g, cols in ATTR_GROUPS.items():
            if idx := [h.cols.index(c) for c in cols if c in h.cols]:
                x2 = x.copy()
                x2[:, idx] = h.neutral[idx]
                out[g] = float(base - h.prob(x2))
        return out


def make_scores(cfg: Config):
    """'rule' -> RuleScores; 'gbm' / 'tcn' -> LearnedScores loading cfg.scorer.models."""
    rule = RuleScores(cfg)
    return rule if cfg.scorer.provider == "rule" else LearnedScores(cfg, rule)
