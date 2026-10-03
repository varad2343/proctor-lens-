"""Feature schema + type-contract invariants, plus the no-accusatory-wording repo check."""
import re
from pathlib import Path

from proctorlens.core.types import EVENT_TYPES, FACE_DERIVED, MACHINE_EVENT, SCORE_KEYS, ZONES
from proctorlens.features.schema import (ATTR_GROUPS, COLUMNS, GROUPS, MODEL_COLUMNS, SCHEMA_VERSION,
                                         STRING_COLUMNS)

ROOT = Path(__file__).resolve().parents[2]


def test_columns():
    assert isinstance(SCHEMA_VERSION, int) and SCHEMA_VERSION >= 1
    assert len(COLUMNS) == len(set(COLUMNS)) == sum(len(g) for g in GROUPS.values())  # no duplicate column
    assert COLUMNS[0] == "t_ms" and STRING_COLUMNS <= set(COLUMNS)
    for c in ("frame_valid", "primary_face_present", "reliable", "zone", "quality_reasons", "id_similarity"):
        assert c in COLUMNS, c


def test_model_columns_and_attr_groups():
    assert len(MODEL_COLUMNS) == len(set(MODEL_COLUMNS)) and 30 <= len(MODEL_COLUMNS) <= 40  # "~35"
    assert set(MODEL_COLUMNS) <= set(COLUMNS)
    # numeric only; no identity/timing/string inputs (Section 7.4)
    banned = set(STRING_COLUMNS) | set(GROUPS["identity"]) | set(GROUPS["timing"])
    assert not banned & set(MODEL_COLUMNS)
    flat = [c for g in ATTR_GROUPS.values() for c in g]
    assert len(flat) == len(set(flat)) and set(flat) == set(MODEL_COLUMNS)  # groups partition the inputs exactly
    assert set(ATTR_GROUPS) == {"head_pose", "eye", "mouth", "quality"}


def test_type_contract():
    assert set(MACHINE_EVENT.values()) == set(EVENT_TYPES)  # every event type has a machine
    assert tuple(MACHINE_EVENT) == SCORE_KEYS and set(FACE_DERIVED) <= set(SCORE_KEYS)
    assert "degraded" not in FACE_DERIVED  # degraded must keep firing while everything face-derived is gated
    assert {"on_screen", "left", "right", "up", "down", "none"} <= set(ZONES)


def test_no_accusatory_wording():
    """Principle 1 (SPEC 1.2): events are observations, never verdicts. SPEC.md itself is exempt (it names the ban)."""
    bad = re.compile(r"\b(cheat\w*|fraud\w*|dishonest\w*|guilty|suspicious)\b", re.I)
    files = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md"), *ROOT.glob("configs/**/*.yaml")]
    files += [f for d in ("src", "ml", "tools") for f in (ROOT / d).rglob("*.py")]
    files += [f for p in ("src/**/*.ts", "src/**/*.tsx", "*.html", "*.mjs") for f in (ROOT / "frontend").glob(p)]  # UI copy
    hits = [f"{f.relative_to(ROOT)}:{i}: {m.group(0)}" for f in files if f.is_file() and f.name != "SPEC.md"
            for i, ln in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1)
            if (m := bad.search(ln))]
    assert not hits, hits
