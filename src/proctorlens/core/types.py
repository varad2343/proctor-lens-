"""Shared data types — the contract between modules. Change here only, then update users."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

FACE_ABSENT = "FACE_ABSENT"
MULTIPLE_PEOPLE = "MULTIPLE_PEOPLE"
PROHIBITED_OBJECT = "PROHIBITED_OBJECT"
OFF_SCREEN_SUSTAINED = "OFF_SCREEN_SUSTAINED"
REPEATED_GLANCING = "REPEATED_GLANCING"
MOUTH_ACTIVITY = "MOUTH_ACTIVITY"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
MONITORING_DEGRADED = "MONITORING_DEGRADED"
EVENT_TYPES = (FACE_ABSENT, MULTIPLE_PEOPLE, PROHIBITED_OBJECT, OFF_SCREEN_SUSTAINED,
               REPEATED_GLANCING, MOUTH_ACTIVITY, IDENTITY_MISMATCH, MONITORING_DEGRADED)
# Web app only (docs/DECISIONS.md ADR-016): browser telemetry, not CV, so no state machine and not in EVENT_TYPES.
BROWSER_INTEGRITY = "BROWSER_INTEGRITY"
REVIEW_TYPES = (*EVENT_TYPES[:-1], BROWSER_INTEGRITY, MONITORING_DEGRADED)  # what a reviewer sees; degraded last

# Score keys (one state machine per key) -> event type they emit.
MACHINE_EVENT = {
    "face_absent": FACE_ABSENT,
    "multiple_people": MULTIPLE_PEOPLE,
    "phone": PROHIBITED_OBJECT,
    "notes": PROHIBITED_OBJECT,
    "off_screen": OFF_SCREEN_SUSTAINED,
    "glancing": REPEATED_GLANCING,
    "speaking": MOUTH_ACTIVITY,
    "id_mismatch": IDENTITY_MISMATCH,
    "degraded": MONITORING_DEGRADED,
}
SCORE_KEYS = tuple(MACHINE_EVENT)
# Face-derived machines: suppressed while MONITORING_DEGRADED is active and gated on `reliable`.
FACE_DERIVED = ("off_screen", "glancing", "speaking", "id_mismatch")

# Scores = {score_key: float in [0,1] | None}. None = unreliable/undefined (quality gating):
# state machines must not start an event on None and must not count None time as evidence.
Scores = dict[str, "float | None"]

# Normalized image coordinates: (x0, y0, x1, y1) in [0, 1], origin top-left.
Box = tuple[float, float, float, float]

ZONES = ("on_screen", "left", "right", "up", "down", "off", "none")  # "none" = no usable face


@dataclass
class Event:
    type: str
    start_ms: int
    end_ms: int
    confidence: float  # detector confidence that the observation is real (not a judgement of the person)
    detector: str  # e.g. "rule@0.1.0" / "gbm@0.1.0"
    details: dict[str, Any] = field(default_factory=dict)
    attribution: dict[str, float] | None = None  # learned scorer only: feature-group score drop
    status: str = "final"  # "ongoing" | "final"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Face:
    """One detected face, in camera frame (absolute, not baseline-relative)."""
    bbox: Box
    yaw: float  # degrees; >0 = face turned toward image-right (see perception/head_pose.py)
    pitch: float  # degrees; >0 = face tilted up
    roll: float  # degrees; >0 = clockwise in image
    iris: tuple[float, float, float, float]  # iris_lx, iris_ly, iris_rx, iris_ry: iris centre ratio inside eye, [0,1]
    eye_open: tuple[float, float]  # left, right eyelid openness, ~[0,1]
    blend: dict[str, float]  # MediaPipe blendshape name -> score
    center: tuple[float, float]  # face centre, normalized image coords
    scale: float  # face size = bbox height / frame height
    conf: float = 1.0  # landmark presence/confidence
    static: bool = False  # set by the extractor's static-face suppression
    recovered: bool = False  # found by the small-face recovery pass (upscaled person-box crop), not the main landmarker
    track: int = 0  # display-only track number from Pipeline (0 = untracked); follows the box, not an identity
    # (478, 2) float32 normalized landmark xy, for drawing only: never in rows, events or files (Any: no numpy here)
    mesh: Any = field(default=None, compare=False, repr=False)


@dataclass
class Detections:
    phone_conf: float = 0.0
    notes_conf: float = 0.0
    person_boxes: list[Box] = field(default_factory=list)  # after NMS
    boxes: list[tuple[str, float, Box]] = field(default_factory=list)  # (class, conf, box) for overlays
    fresh: bool = True  # False when carried over from an earlier frame (YOLO runs every ~3rd frame)
    ids: list[int] = field(default_factory=list)  # display-only track numbers, parallel to boxes (set by Pipeline)


@dataclass
class IdCheck:
    similarity: float  # cosine to enrollment embedding
    quality_ok: bool  # high-quality, frontal check


@dataclass
class Perceived:
    """Everything perception knows about one frame."""
    t_ms: int
    faces: list[Face]  # sorted by bbox area desc; faces[0] = primary
    det: Detections
    id: IdCheck | None  # None between identity checks
    luma: float
    blur: float
    overexp_frac: float
    quality: float  # [0,1]
    quality_reasons: list[str]  # codes: dark, bright, blur, small_face, face_cut, no_face, blocked


def nan() -> float:
    return math.nan
