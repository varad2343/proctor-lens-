"""End-of-exam detection summary (ADR-018): for every detector, how long something was in view, the most seen at once,
first / last time and the highest confidence. Raw detector observations, not events: a phone seen for 0.3 s shows up
here even though it never became a PROHIBITED_OBJECT event (that needs 1.5 s). Never a judgement of a person."""
from __future__ import annotations

from proctorlens.core.types import Perceived

MAX_GAP_MS = 500  # a frame accounts for the time since the previous one, at most this (a stall counts as unseen)
# key -> label shown to the reviewer, in display order
LABELS = {"face": "Face", "faces_2plus": "More than one face", "person": "Person (body)",
          "persons_2plus": "More than one person", "phone": "Phone", "notes": "Book / notes", "hand": "Hand",
          "pose": "Body pose (skeleton)", "poor_image": "Poor image quality"}


def observe(p: Perceived, min_quality: float = 0.5) -> dict[str, tuple[int, float | None]]:
    """key -> (how many in this frame, highest detector confidence or None)."""
    boxes = p.det.boxes
    conf = lambda cls: max((c for n, c, _ in boxes if n == cls), default=None)  # noqa: E731
    n_person, nf = sum(n == "person" for n, _, _ in boxes), len(p.faces)
    return {"face": (nf, None), "faces_2plus": (nf if nf >= 2 else 0, None),
            "person": (n_person, conf("person")), "persons_2plus": (n_person if n_person >= 2 else 0, None),
            "phone": (sum(n == "phone" for n, _, _ in boxes), conf("phone")),
            "notes": (sum(n == "notes" for n, _, _ in boxes), conf("notes")),
            "hand": (len(p.hands), None), "pose": (len(p.poses), None),
            "poor_image": (int(p.quality < min_quality), None)}


class DetectionTally:
    def __init__(self, min_quality: float = 0.5):
        self.min_quality, self.last_t, self.total_ms = min_quality, None, 0
        self.stats = {k: {"seen_ms": 0, "max_count": 0, "first_ms": None, "last_ms": None, "max_conf": None}
                      for k in LABELS}

    def add(self, t_ms: int, p: Perceived | None) -> None:
        dt = 0 if self.last_t is None else min(max(t_ms - self.last_t, 0), MAX_GAP_MS)
        self.last_t, self.total_ms = t_ms, self.total_ms + dt
        if p is None:
            return
        for k, (n, c) in observe(p, self.min_quality).items():
            if not n:
                continue
            s = self.stats[k]
            s["seen_ms"] += dt
            s["max_count"] = max(s["max_count"], n)
            s["first_ms"] = t_ms if s["first_ms"] is None else s["first_ms"]
            s["last_ms"] = t_ms
            if c is not None:
                s["max_conf"] = round(max(c, s["max_conf"] or 0.0), 3)

    def summary(self) -> dict:
        tot = self.total_ms
        return {"total_s": round(tot / 1000, 1), "detectors": [
            {"key": k, "label": LABELS[k], "seen_s": round(s["seen_ms"] / 1000, 1),
             "pct": round(100 * s["seen_ms"] / tot, 1) if tot else None, **{x: s[x] for x in
                                                                             ("max_count", "first_ms", "last_ms", "max_conf")}}
            for k, s in self.stats.items()]}
