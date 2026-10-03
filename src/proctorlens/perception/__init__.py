"""Perception: one frame -> Perceived (faces, detections, optional identity check, quality)."""
from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np

from proctorlens.core.config import Config
from proctorlens.core.types import Detections, Face, IdCheck, Perceived
from proctorlens.perception.enhance import lowlight_enhance
from proctorlens.perception.identity import Enrollment, IdentityChecker
from proctorlens.perception.landmarks import Landmarker, recover_faces, sort_faces, uncovered
from proctorlens.perception.objects import ObjectDetector
from proctorlens.perception.quality import QualityResult, assess

_N_ENROLL = 5  # ~5 embeddings spread over enroll_seconds (SPEC 5.5)
_RETURN_ABSENT_MS = 1000  # ponytail: a face missing < 1 s is tracker flicker, not a "return"


class Perceiver:
    """Components are injectable; None => built on first use from cfg.models (identity only if its dir exists).
    `recoverer` = the image-mode Landmarker of the small-face recovery pass; built on first need from
    cfg.models.landmarker, and recovery is simply skipped when that file is absent (e.g. a fake landmarker)."""

    def __init__(self, cfg: Config, landmarker=None, detector=None, identity=None, recoverer=None):
        self.cfg, self.landmarker, self.detector, self.identity = cfg, landmarker, detector, identity
        self.recoverer = recoverer
        self.enrollment = Enrollment()  # in memory only
        self._id_built = identity is not None
        self._rec_built = recoverer is not None
        self._det: Detections | None = None
        self._t0: int | None = None  # first frame time
        self._enr_t: int | None = None  # time of the last enrollment embedding
        self._next_id = 0  # time the next identity check is due
        self._lost: int | None = None  # start of the current no-face stretch

    def _build_identity(self) -> IdentityChecker | None:
        if not Path(self.cfg.models.identity).is_dir():
            return None
        try:
            return IdentityChecker(self.cfg.models.identity)
        except ImportError as e:
            warnings.warn(f"identity disabled: {e}")
            return None

    def _recover(self, frame: np.ndarray, faces: list[Face], det: Detections) -> list[Face]:
        """Add faces found in the head crops of person boxes that have none (cfg.perception.recover_*)."""
        pc = self.cfg.perception
        if not (pc.recover_small_faces and uncovered(det.person_boxes, faces)):
            return faces
        if not self._rec_built:
            self._rec_built = True
            if Path(self.cfg.models.landmarker).is_file():
                try:
                    self.recoverer = Landmarker(self.cfg.models.landmarker, num_faces=2, image_mode=True)
                except ImportError as e:
                    warnings.warn(f"small-face recovery disabled: {e}")
        if self.recoverer is None:
            return faces
        return sort_faces(faces + recover_faces(self.recoverer, frame, det.person_boxes, faces,
                                                pc.recover_pad, pc.recover_min_side))

    def perceive(self, frame_bgr: np.ndarray, t_ms: int, idx: int) -> Perceived:
        cfg, m, pc = self.cfg, self.cfg.models, self.cfg.perception
        if self.landmarker is None:
            self.landmarker = Landmarker(m.landmarker)
        if self.detector is None:
            self.detector = ObjectDetector(m.detector, m.detector_conf)
        if not self._id_built:
            self._id_built, self.identity = True, self._build_identity()
        if self._det is None or idx % cfg.pipeline.yolo_every == 0:  # detector first: recovery needs its person boxes
            self._det = det = self.detector.process(frame_bgr)
        else:
            det = dataclasses.replace(self._det, fresh=False)
        # the landmarker (and recovery) see the enhanced frame; quality below keeps the original, so dark stays 'dark'
        x = lowlight_enhance(frame_bgr, pc.lowlight_luma) if pc.lowlight_enhance else frame_bgr
        faces = self._recover(x, self.landmarker.process(x, t_ms), det)
        q = assess(frame_bgr, faces[0].bbox if faces else None, cfg.quality)
        return Perceived(t_ms, faces, det, self._identity(frame_bgr, t_ms, faces, q),
                         q.luma, q.blur, q.overexp_frac, q.quality, q.reasons)

    def _identity(self, frame: np.ndarray, t_ms: int, faces: list[Face], q: QualityResult) -> IdCheck | None:
        """Enroll from quality-ok frontal frames during the first enroll_seconds, then one IdCheck every
        identity_every_s and right after a face returns from absence."""
        p = self.cfg.pipeline
        enroll_ms = int(p.enroll_seconds * 1000)
        if self._t0 is None:
            self._t0, self._next_id = t_ms, t_ms + enroll_ms
        if not faces:
            self._lost = t_ms if self._lost is None else self._lost
            return None
        if self._lost is not None and t_ms - self._lost >= _RETURN_ABSENT_MS:
            self._next_id = t_ms
        self._lost = None
        if self.identity is None:
            return None
        f = faces[0]
        ok = q.quality >= self.cfg.quality.min_quality and abs(f.yaw) < self.cfg.identity.max_abs_yaw
        if t_ms - self._t0 < enroll_ms:
            if ok and (self._enr_t is None or t_ms - self._enr_t >= enroll_ms / _N_ENROLL):
                if (e := self.identity.embed(frame, f)) is not None:
                    self.enrollment.add(e)
                    self._enr_t = t_ms
        elif self.enrollment.ready and t_ms >= self._next_id:
            if (e := self.identity.embed(frame, f)) is not None:
                self._next_id = t_ms + int(p.identity_every_s * 1000)
                return IdCheck(self.enrollment.similarity(e), ok)
        return None
