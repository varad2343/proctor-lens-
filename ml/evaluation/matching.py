"""Event-level matching (spec 12.2): same type, temporal IoU >= thr, one-to-one, greedy by IoU."""
from __future__ import annotations

from typing import Any


def _get(e: Any, k: str, default: Any = None) -> Any:
    return e.get(k, default) if isinstance(e, dict) else getattr(e, k, default)


def iv(e: Any) -> tuple[str, int, int]:
    """(type, start_ms, end_ms) of an Event or a dict record."""
    return _get(e, "type"), int(_get(e, "start_ms")), int(_get(e, "end_ms"))


def match_events(pred, gt, iou_thr: float = 0.3, pad_ms: int = 1000, short_ms: int = 5000):
    """-> (tp_pairs [(pred_idx, gt_idx)], fp_idx, fn_idx).

    A GT shorter than short_ms is padded by pad_ms on each side before IoU (onset/offset tolerance).
    Each GT matches at most one prediction and vice versa; candidates are taken by descending IoU
    (ties: lower pred, then lower GT index), so the result is deterministic.
    """
    P, G = [iv(e) for e in pred], [iv(e) for e in gt]
    cand = []
    for gi, (gt_type, gs, ge) in enumerate(G):
        if ge - gs < short_ms:
            gs, ge = gs - pad_ms, ge + pad_ms
        for pi, (pt, ps, pe) in enumerate(P):
            if pt != gt_type or min(pe, ge) - max(ps, gs) <= 0:
                continue
            iou = (min(pe, ge) - max(ps, gs)) / (max(pe, ge) - min(ps, gs))  # overlapping => union = span
            if iou >= iou_thr:
                cand.append((-iou, pi, gi))
    tp, used_p, used_g = [], set(), set()
    for _, pi, gi in sorted(cand):
        if pi not in used_p and gi not in used_g:
            tp.append((pi, gi))
            used_p.add(pi)
            used_g.add(gi)
    return (sorted(tp), [i for i in range(len(P)) if i not in used_p],
            [i for i in range(len(G)) if i not in used_g])
