"""Per-user gaze calibration (ridge regression) and off-screen scoring.

Conventions: zones are in IMAGE orientation: yaw>0 = face toward image-right -> "right"; pitch>0 = up -> "up".
Screen coordinates (gx, gy) are (0,0) top-left .. (1,1) bottom-right. For an unmirrored camera screen-right
is image-left, so calibration learns GazeModel.x_sign (iris-x vs dot-x correlation) to keep gaze- and
head-derived zones in the same orientation.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from proctorlens.core.config import GazeCfg
from proctorlens.core.types import Face

# (schema column, MediaPipe blendshape) in features.schema order: in_l, in_r, out_l, out_r, up_l, ...
EYELOOK = [(f"eyelook_{d.lower()}_{s[0].lower()}", f"eyeLook{d}{s}")
           for d in ("In", "Out", "Up", "Down") for s in ("Left", "Right")]
_CLIP = (-0.5, 1.5)


@dataclass
class Baseline:
    """Neutral (looking at screen centre) head pose and face position/scale. Defaults = 'no calibration'."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    cx: float = 0.5
    cy: float = 0.5
    scale: float = 1.0

    @classmethod
    def from_faces(cls, faces: list[Face]) -> Baseline:
        if not faces:
            return cls()
        return cls(*np.median([[f.yaw, f.pitch, f.roll, *f.center, f.scale] for f in faces], axis=0).tolist())


def gaze_features(face: Face, base: Baseline) -> np.ndarray:
    """[dyaw,dpitch,droll, iris x4, eyelook x8 (schema order), dcx, dcy, scale ratio] -> (18,)."""
    return np.array([face.yaw - base.yaw, face.pitch - base.pitch, face.roll - base.roll, *face.iris,
                     *(face.blend.get(k, 0.0) for _, k in EYELOOK),
                     face.center[0] - base.cx, face.center[1] - base.cy, face.scale / base.scale], float)


def _fit(X, y, a):
    m, s = X.mean(0), X.std(0)
    s = np.where(s < 1e-3, 1.0, s)  # near-constant features carry no signal; don't amplify their noise
    Z, b = (X - m) / s, y.mean(0)
    return m, s, np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - b)), b


class GazeModel:
    """Standardized ridge regression features -> screen (x, y). Closed form, JSON-serialisable.
    x_sign = -1 when screen-x runs against image-x (unmirrored camera); set by fit_calibration."""

    x_sign = 1.0

    def fit(self, X, y, groups, alphas) -> GazeModel:
        """alpha chosen by leave-one-point-out CV (group = dot id)."""
        X, y, g = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
        ids = np.unique(g)

        def cv(a: float) -> float:
            err = []
            for i in ids:
                te = g == i
                m, s, w, b = _fit(X[~te], y[~te], a)
                err.append(np.linalg.norm(((X[te] - m) / s) @ w + b - y[te], axis=1))
            return float(np.concatenate(err).mean())

        # ponytail: <2 dots => no CV possible, take the most regularized alpha
        self.alpha = float(min(alphas, key=cv) if len(ids) > 1 else max(alphas))
        self.mean, self.std, self.w, self.b = _fit(X, y, self.alpha)
        return self

    def predict(self, X) -> np.ndarray:
        """(n, 2) screen coordinates clipped to [-0.5, 1.5] (regression cannot extrapolate far)."""
        Z = (np.atleast_2d(np.asarray(X, float)) - self.mean) / self.std
        return np.clip(Z @ self.w + self.b, *_CLIP)

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "x_sign": self.x_sign,
                **{k: getattr(self, k).tolist() for k in ("mean", "std", "w", "b")}}

    @classmethod
    def from_dict(cls, d: dict) -> GazeModel:
        m = cls()
        m.alpha, m.x_sign = float(d["alpha"]), float(d.get("x_sign", 1.0))
        for k in ("mean", "std", "w", "b"):
            setattr(m, k, np.array(d[k], float))
        return m


@dataclass
class Calibration:
    baseline: Baseline
    model: GazeModel | None
    error: float  # median validation-dot error in normalized screen units; nan if no validation
    mode: str  # 'full' | 'head_pose_only'
    accepted: bool
    tries: int = 1
    n_dropped: int = 0  # robust fit: grid/validation samples rejected (blinks, outliers); diagnostics only
    dot_errors: list[float] = field(default_factory=list)  # per-validation-dot error (diagnostics only)

    def to_dict(self) -> dict:
        return {"baseline": asdict(self.baseline), "model": self.model.to_dict() if self.model else None,
                "error": None if math.isnan(self.error) else float(self.error), "mode": self.mode,
                "accepted": bool(self.accepted), "tries": int(self.tries), "n_dropped": int(self.n_dropped),
                "dot_errors": [float(e) for e in self.dot_errors]}

    @classmethod
    def from_dict(cls, d: dict) -> Calibration:
        return cls(Baseline(**d["baseline"]), GazeModel.from_dict(d["model"]) if d["model"] else None,
                   math.nan if d["error"] is None else float(d["error"]), d["mode"], bool(d["accepted"]),
                   int(d.get("tries", 1)), int(d.get("n_dropped", 0)), [float(e) for e in d.get("dot_errors", [])])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CalibSample:
    face: Face
    x: float
    y: float
    phase: str  # 'neutral' | 'grid' | 'validation'
    dot_id: int


_BLINK_FRAC = 0.6  # eye_open below this x the dot's median = blink (same rule as the extractor's blink flag)


def _clean(ss: list[CalibSample], base: Baseline, k: float | None):
    """-> (gaze features (n, 18), dot id per row, keep mask). With k (= cfg.mad_k) the mask drops, per dot, blink
    frames (mean eye_open < _BLINK_FRAC x the dot's median) and rows whose robust distance from the dot's median
    exceeds k MADs (saccades that outlived the settle window, tracking glitches), never more than half a dot.
    Distance = max(RMS over features, worst feature / 2) in robust sigmas: the RMS alone misses a lone-feature spike,
    the max alone drops ~25% of clean rows at k=3 (18 features). k=None keeps every row (the legacy fit).
    ponytail: sigma is the MAD pooled over all dots (10 rows per dot is too few for a per-dot MAD); a dot whose
    majority is bad keeps a bad median, which the validation error then shows."""
    g = np.array([s.dot_id for s in ss])
    F = np.array([gaze_features(s.face, base) for s in ss])
    keep = np.ones(len(ss), bool)
    if k is None or not ss:
        return F, g, keep
    eo, ids = np.array([np.mean(s.face.eye_open) for s in ss]), np.unique(g)
    for i in ids:
        keep[g == i] = ~(eo[g == i] < _BLINK_FRAC * np.median(eo[g == i]))  # nan compares False -> kept
    R = np.abs(F - np.array([np.median(F[(g == i) & keep], axis=0) for i in ids])[np.searchsorted(ids, g)])
    Z = R / np.maximum(1.4826 * np.median(R[keep], axis=0), 1e-3)  # floor: constant/noise-free features
    d = np.where(keep, np.maximum(np.sqrt((Z ** 2).mean(1)), Z.max(1) / 2), np.inf)
    for i in ids:
        ix = np.flatnonzero(g == i)
        keep[ix] = False
        keep[ix[np.argsort(d[ix], kind="stable")[:max((len(ix) + 1) // 2, int((d[ix] <= k).sum()))]]] = True
    return F, g, keep


def fit_calibration(samples: list[CalibSample], cfg: GazeCfg) -> Calibration:
    """neutral -> baseline, grid -> regression, validation -> error. Accepted iff error < cfg.accept_error.
    cfg.robust: blink/outlier rows are dropped (_clean) before the ridge fit and the per-dot validation medians;
    False = every raw row is used (the original fit, bit for bit)."""
    ph = {k: [s for s in samples if s.phase == k] for k in ("neutral", "grid", "validation")}
    base = Baseline.from_faces([s.face for s in (ph["neutral"] or ph["grid"])])
    model, err, dropped, errs = None, math.nan, 0, []
    grid, val = ph["grid"], ph["validation"]
    if len({s.dot_id for s in grid}) >= 3:  # ponytail: 3 dots is the floor for a meaningful fit/CV
        k = cfg.mad_k if cfg.robust else None
        X, g, keep = _clean(grid, base, k)
        xy = np.array([(s.x, s.y) for s in grid])
        model = GazeModel().fit(X[keep], xy[keep], g[keep], cfg.ridge_alphas)
        ix, dx = X[keep][:, [3, 5]].mean(1), xy[keep][:, 0]  # iris-x ratios vs dot x
        model.x_sign = -1.0 if (ix - ix.mean()) @ (dx - dx.mean()) < 0 else 1.0
        Xv, gv, kv = _clean(val, base, k)
        dropped = int((~keep).sum() + (~kv).sum())
        for i in np.unique(gv):
            px, py = np.median(model.predict(Xv[(gv == i) & kv]), axis=0)
            x, y = next((s.x, s.y) for s in val if s.dot_id == i)
            errs.append(math.hypot(px - x, py - y))
        if errs:
            err = float(np.median(errs))
    accepted = bool(err < cfg.accept_error)  # nan -> False
    return Calibration(base, model, err, "full" if accepted else "head_pose_only", accepted,
                       n_dropped=dropped, dot_errors=errs)


@dataclass
class GazeOut:
    score: float  # off-screen evidence in [0,1]; nan when there is no face
    zone: str  # one of core.types.ZONES
    gx: float
    gy: float
    in_screen_prob: float


def soft(x: float, center: float, width: float) -> float:
    """Linear soft threshold: 0 at center-width/2, 0.5 at center, 1 at center+width/2."""
    return float(np.clip(0.5 + (x - center) / width, 0.0, 1.0))


def off_screen(face: Face | None, calib: Calibration | None, cfg: GazeCfg) -> GazeOut:
    """score = max(gaze_outside, head_turn). Each source is an (ex, ey) deviation in units of its own limit
    (+x right, +y up; |e|=1 at the limit: yaw/pitch limit for the head, margin-expanded screen edge for
    gaze)."""
    if face is None:
        return GazeOut(math.nan, "none", math.nan, math.nan, math.nan)
    base = calib.baseline if calib else Baseline()
    es = [((face.yaw - base.yaw) / cfg.yaw_limit_deg, (face.pitch - base.pitch) / cfg.pitch_limit_deg)]
    gx = gy = math.nan
    if calib and calib.mode == "full" and calib.model is not None:
        px, py = calib.model.predict(gaze_features(face, base))[0]
        if math.isfinite(px) and math.isfinite(py):
            gx, gy = float(px), float(py)
            h = 0.5 + cfg.margin
            es.append(((gx - 0.5) * calib.model.x_sign / h, (0.5 - gy) / h))
    ex, ey = max(es, key=lambda e: max(abs(e[0]), abs(e[1])))  # dominant source
    score = soft(max(abs(ex), abs(ey)), 1.0, cfg.soft_width)
    if score < 0.5:
        zone = "on_screen"
    elif abs(ex) >= abs(ey):
        zone = "right" if ex > 0 else "left"
    else:
        zone = "up" if ey > 0 else "down"
    return GazeOut(score, zone, gx, gy, 1.0 - score)


class DriftMonitor:
    """Running median of on-task gaze predictions vs screen centre.
    ponytail: window counts on-task samples (not wall time); head-position drift not tracked."""

    def __init__(self, cfg: GazeCfg, hz: float):
        self.bound = cfg.drift_bound
        self.buf: deque = deque(maxlen=max(2, int(cfg.drift_window_s * hz)))

    def update(self, gx: float, gy: float, on_task: bool) -> bool:
        if on_task and math.isfinite(gx) and math.isfinite(gy):
            self.buf.append((gx, gy))
        if len(self.buf) < self.buf.maxlen // 2:  # wait for half a window before judging
            return False
        mx, my = np.median(self.buf, axis=0)
        return bool(math.hypot(mx - 0.5, my - 0.5) > self.bound)
