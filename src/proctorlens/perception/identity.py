"""Identity: InsightFace embedding + in-memory enrollment. Embeddings are never written to disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from proctorlens.core.types import Face


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class IdentityChecker:
    """InsightFace detect+align+embed on the primary face crop (lazy import).

    model_dir must look like <root>/models/<name> (e.g. data/models/buffalo_sc); insightface fetches the
    pack itself when it is missing, so place it there beforehand for offline use."""

    def __init__(self, model_dir: str):
        from insightface.app import FaceAnalysis

        p = Path(model_dir)
        self._app = FaceAnalysis(name=p.name, root=str(p.parent.parent), providers=["CPUExecutionProvider"],
                                 allowed_modules=["detection", "recognition"])
        self._app.prepare(ctx_id=-1, det_size=(320, 320))

    def embed(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray | None:
        """L2-normalised embedding of the face (None if insightface finds no face in the padded crop).
        ponytail: crops the landmarker bbox +40% instead of reusing its landmarks for alignment."""
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = face.bbox
        mx, my = 0.4 * (x1 - x0), 0.4 * (y1 - y0)
        a, b, c, d = (np.clip([x0 - mx, y0 - my, x1 + mx, y1 + my], 0, 1) * [w, h, w, h]).astype(int)
        crop = frame_bgr[b:d, a:c]
        if crop.size == 0:
            return None
        found = self._app.get(crop)
        if not found:
            return None
        best = max(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return np.asarray(best.normed_embedding, dtype=np.float32)


class Enrollment:
    """Mean of the enrollment embeddings, in memory only."""

    def __init__(self) -> None:
        self._embs: list[np.ndarray] = []

    def add(self, emb: np.ndarray) -> None:
        self._embs.append(np.asarray(emb, dtype=np.float32))

    @property
    def ready(self) -> bool:
        return len(self._embs) >= 3

    def mean_emb(self) -> np.ndarray:
        m = np.mean(self._embs, axis=0)
        return m / (np.linalg.norm(m) + 1e-12)

    def similarity(self, emb: np.ndarray) -> float:
        return cosine(self.mean_emb(), emb)
