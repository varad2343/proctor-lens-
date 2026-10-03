"""Config: dataclasses + YAML (deviation from spec's Pydantic — see docs/DECISIONS.md). No threshold lives in logic."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import REVIEW_TYPES


@dataclass
class PipelineCfg:
    grid_hz: int = 10
    max_age_ms: int = 250  # a frame older than this at a grid step => frame_valid=False
    yolo_every: int = 3  # run the detector every Nth processed frame
    identity_every_s: float = 10.0
    enroll_seconds: float = 5.0  # first N seconds of a recording = identity enrollment
    stall_ms: int = 2000  # no valid frame for this long => degraded("frame_gap")
    static_window_s: float = 20.0  # face/person box unmoved this long => static_background
    static_iou: float = 0.92
    min_second_face_frac: float = 0.08  # ignore secondary faces smaller than this (frame-height fraction)
    turned_away_yaw: float = 50.0  # |yaw| above this when the face vanishes => TURNED_AWAY, not FACE_ABSENT
    turned_away_hold_ms: int = 10000  # ...for this long, and for as long as a person is still detected (> t_on of E4)


@dataclass
class QualityCfg:
    min_luma: float = 40.0
    max_luma: float = 220.0
    min_blur: float = 30.0  # variance of Laplacian
    min_face_frac: float = 0.10
    overexp_max: float = 0.25
    blocked_luma_std: float = 6.0  # frame luma std below this => camera blocked
    min_quality: float = 0.5  # quality below this => reliable=False


@dataclass
class GazeCfg:
    neutral_s: float = 2.0
    dot_s: float = 1.5
    settle_s: float = 0.5
    dwell_samples: int = 10  # interactive calibration moves on once this many face samples were seen at a dot
    dot_timeout_s: float = 4.0  # ...or after this long, so a missing face cannot stall it
    n_validation: int = 4
    accept_error: float = 0.15  # median validation-dot error, normalized screen units
    max_tries: int = 2
    ridge_alphas: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    margin: float = 0.12  # screen rect expanded by this before a gaze point counts as outside
    yaw_limit_deg: float = 25.0  # |d_yaw| beyond this => head_turn_score -> 1
    pitch_limit_deg: float = 20.0
    soft_width: float = 0.5  # soft-threshold width as fraction of the limit
    drift_window_s: float = 120.0
    drift_bound: float = 0.25  # running-median gaze distance from screen centre
    robust: bool = True  # calibration: reject blink/outlier samples before fitting + stability-gated dwell (False = old behaviour)
    mad_k: float = 3.0  # outlier threshold in median-absolute-deviations
    stable_deg: float = 2.0  # robust, interactive calibration: a face counts toward dwell_samples only if yaw and pitch moved less than this since the previous face


@dataclass
class SmoothCfg:
    """One Euro filter on pose/gaze signals (extractor). enabled=False = raw signals, the old behaviour."""
    enabled: bool = True
    min_cutoff: float = 0.25  # Hz; lower = steadier at rest
    beta: float = 0.3  # Hz per deg/s of speed above speed_floor; higher = less lag during fast head turns
    d_cutoff: float = 0.5  # Hz; low-pass on the speed estimate
    speed_floor: float = 12.0  # deg/s (50 deg ~ 1.0 for normalized signals): slower apparent motion is noise
    blink_frac: float = 0.75  # eyelid openness below this x its running open level = blink: eye signals are held
    hold_s: float = 0.5  # ...for at most this long (longer = eyes shut / squint, not a blink: values pass through)
    reset_gap_s: float = 0.5  # face unseen longer than this (or a different face) restarts the filters


@dataclass
class PerceptionCfg:
    """Front-end accuracy aids; each can be switched off for A/B comparison (False = old behaviour)."""
    lowlight_enhance: bool = True  # gamma/CLAHE the landmarker input when the frame is dim
    lowlight_luma: float = 90.0  # enhance frames whose mean luma is below this
    recover_small_faces: bool = True  # re-run the landmarker on upscaled person-box crops when no face was found
    recover_pad: float = 0.15  # fractional padding around the person-box crop
    recover_min_side: int = 384  # upscale the crop so its short side is at least this many pixels


@dataclass
class GlanceCfg:
    n: int = 4
    window_s: float = 60.0
    min_s: float = 0.3
    max_s: float = 3.0
    same_zone_frac: float = 0.7
    on_thr: float = 0.5  # off_screen_score above this = "off screen" for excursion extraction


@dataclass
class IdentityCfg:
    tau: float = 0.35  # cosine similarity threshold; calibrate with ml/training/calibrate_identity.py
    consecutive: int = 3
    max_abs_yaw: float = 20.0  # only frontal frames count as high quality


@dataclass
class ScorerCfg:
    provider: str = "rule"  # rule | gbm | tcn
    models: dict[str, str] = field(default_factory=dict)  # target ("off_screen"|"speaking") -> model dir
    window: int = 60  # steps at grid_hz (6 s at 10 Hz)


@dataclass
class PolicyCfg:
    allow_notes: bool = False
    allow_looking_down: bool = False
    allow_reading_aloud: bool = False
    active: list[str] = field(default_factory=lambda: list(_ALL_KEYS))


@dataclass
class ReviewCfg:
    """Review segments + priority (master spec 9; docs/DECISIONS.md ADR-015). Priority orders what a human looks at
    first; it is never a judgement of a person. Post-processing only: changes nothing in features or events."""
    merge_gap_s: float = 5.0  # events this close (or overlapping) form one review segment
    weights: dict[str, float] = field(default_factory=lambda: {
        "MULTIPLE_PEOPLE": 1.0, "PROHIBITED_OBJECT": 1.0, "IDENTITY_MISMATCH": 1.0, "FACE_ABSENT": 0.6,
        "REPEATED_GLANCING": 0.6, "OFF_SCREEN_SUSTAINED": 0.5, "MOUTH_ACTIVITY": 0.4, "BROWSER_INTEGRITY": 0.5,
        "MONITORING_DEGRADED": 0.2})
    multi_bonus: float = 0.25  # x (1 + multi_bonus * (distinct event types - 1)): co-occurring signals rank higher
    high: float = 2.0  # priority >= high => "high", >= medium => "medium", else "low" (ours; tune on validation)
    medium: float = 1.0


@dataclass
class ModelsCfg:
    landmarker: str = "models/face_landmarker.task"
    detector: str = "models/yolo11n.onnx"
    identity: str = "models/buffalo_sc"
    detector_conf: float = 0.30


@dataclass
class EventCfg:
    on_thr: float = 0.5
    off_thr: float = 0.3  # < on_thr (hysteresis)
    t_on_s: float = 0.0  # score must stay >= on_thr this long (debounce)
    t_off_s: float = 0.5  # score must stay < off_thr this long to end
    merge_gap_s: float = 1.0
    min_dur_s: float = 0.0
    cooldown_s: float = 0.0
    max_hold_s: float = 5.0  # ACTIVE event closes if score stays None (unreliable) this long


_ALL_KEYS = ["face_absent", "multiple_people", "phone", "notes", "off_screen",
             "glancing", "speaking", "id_mismatch", "degraded"]


def default_events() -> dict[str, EventCfg]:
    e = EventCfg
    return {
        "face_absent": e(t_on_s=3.0, t_off_s=0.5, merge_gap_s=1.0, min_dur_s=3.0),
        "multiple_people": e(t_on_s=2.0, t_off_s=1.0, merge_gap_s=2.0, min_dur_s=2.0),
        "phone": e(t_on_s=1.5, t_off_s=1.0, merge_gap_s=2.0, min_dur_s=1.5),
        "notes": e(t_on_s=3.0, t_off_s=1.0, merge_gap_s=2.0, min_dur_s=3.0),
        "off_screen": e(off_thr=0.35, t_on_s=4.0, t_off_s=1.0, merge_gap_s=2.0, min_dur_s=4.0),
        "glancing": e(t_on_s=0.0, t_off_s=5.0, merge_gap_s=5.0),  # window rule lives in ExcursionTracker
        "speaking": e(off_thr=0.35, t_on_s=0.0, t_off_s=1.5, merge_gap_s=2.0, min_dur_s=3.0),  # window rule in scorer
        "id_mismatch": e(t_on_s=0.0, t_off_s=0.0, merge_gap_s=0.0),  # consecutive-check rule in scorer
        "degraded": e(t_on_s=2.0, t_off_s=1.0, merge_gap_s=2.0, min_dur_s=2.0),
    }


@dataclass
class Config:
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    gaze: GazeCfg = field(default_factory=GazeCfg)
    smooth: SmoothCfg = field(default_factory=SmoothCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    glance: GlanceCfg = field(default_factory=GlanceCfg)
    identity: IdentityCfg = field(default_factory=IdentityCfg)
    scorer: ScorerCfg = field(default_factory=ScorerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    events: dict[str, EventCfg] = field(default_factory=default_events)
    review: ReviewCfg = field(default_factory=ReviewCfg)


def _build(tp, v):
    org = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, v or {})
    if org is dict:
        _, vt = typing.get_args(tp)
        return {k: _build(vt, x) for k, x in (v or {}).items()}
    if org is list:
        (t,) = typing.get_args(tp)
        return [_build(t, x) for x in (v or [])]
    if tp is float and isinstance(v, int) and not isinstance(v, bool):
        return float(v)  # YAML `t_on_s: 2` and `2.0` must give the same config_hash
    return v


def from_dict(cls, d: dict):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    if unknown := set(d) - names:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _build(hints[k], v) for k, v in d.items()})


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(*paths: str | Path, texts: tuple[str, ...] = ()) -> Config:
    """Defaults <- each YAML file in order, then each YAML text (e.g. a web-app policy) (deep merge). Unknown keys raise."""
    d = dataclasses.asdict(Config())
    for y in [Path(p).read_text(encoding="utf-8") for p in paths] + list(texts):
        _merge(d, yaml.safe_load(y) or {})
    cfg = from_dict(Config, d)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    for k, e in cfg.events.items():
        if k not in _ALL_KEYS:
            raise ValueError(f"unknown event key {k!r}")
        if not e.off_thr < e.on_thr:
            raise ValueError(f"events.{k}: off_thr must be < on_thr")
    if bad := set(cfg.policy.active) - set(_ALL_KEYS):
        raise ValueError(f"policy.active has unknown keys {sorted(bad)}")
    if cfg.scorer.provider not in ("rule", "gbm", "tcn"):
        raise ValueError("scorer.provider must be rule|gbm|tcn")
    if bad := set(cfg.review.weights) - set(REVIEW_TYPES):
        raise ValueError(f"review.weights has unknown event types {sorted(bad)}")
    if not cfg.review.medium <= cfg.review.high:
        raise ValueError("review: medium must be <= high")


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()).hexdigest()[:12]
