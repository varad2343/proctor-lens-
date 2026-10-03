"""Feature-group occlusion attribution (SPEC 8.2). The learned provider does the work; rules have none."""
from __future__ import annotations

from typing import Any


def attribute_event(provider: Any, target: str) -> dict[str, float] | None:
    """{feature group: score drop when that group is neutralized} over the provider's current window.
    None for the rule provider (no `attribute`) or when the window is not ready."""
    fn = getattr(provider, "attribute", None)
    return (fn(target) or None) if fn else None
