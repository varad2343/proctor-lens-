"""Feature schema (10 Hz rows). Freeze early; bump SCHEMA_VERSION on any change."""
from __future__ import annotations

SCHEMA_VERSION = 1

_EYELOOK = [f"eyelook_{d}_{s}" for d in ("in", "out", "up", "down") for s in ("l", "r")]  # 8

GROUPS: dict[str, list[str]] = {
    "timing": ["t_ms", "frame_valid", "frame_age_ms", "effective_fps"],
    "face": ["n_faces", "primary_face_present", "face_x0", "face_y0", "face_x1", "face_y1",
             "face_size_frac", "landmark_conf", "time_since_face_ms", "turned_away"],
    "head_pose": ["d_yaw", "d_pitch", "d_roll", "head_x", "head_y", "head_scale", "head_speed"],
    "eye": ["iris_lx", "iris_ly", "iris_rx", "iris_ry", "eye_open_l", "eye_open_r", "blink", *_EYELOOK,
            "gaze_x", "gaze_y", "gaze_in_screen_prob", "off_screen_score", "zone"],
    "mouth": ["jaw_open", "mouth_close", "mar", "mouth_energy_1s"],
    "objects": ["phone_conf", "notes_conf", "n_persons", "static_face_flags"],
    "identity": ["id_similarity", "id_quality_ok"],
    "quality": ["luma", "blur", "overexp_frac", "quality", "quality_reasons", "reliable"],
}
COLUMNS: list[str] = [c for g in GROUPS.values() for c in g]
STRING_COLUMNS = {"zone", "quality_reasons"}  # everything else is numeric/bool (NaN = undefined)

# Numeric inputs to the learned temporal models (~35). Excludes identity, timing, strings.
MODEL_COLUMNS: list[str] = [
    "d_yaw", "d_pitch", "d_roll", "head_x", "head_y", "head_scale", "head_speed",
    "iris_lx", "iris_ly", "iris_rx", "iris_ry", "eye_open_l", "eye_open_r", "blink", *_EYELOOK,
    "gaze_x", "gaze_y", "gaze_in_screen_prob", "off_screen_score",
    "jaw_open", "mouth_close", "mar", "mouth_energy_1s",
    "n_faces", "primary_face_present", "face_size_frac", "landmark_conf",
    "quality",
]

# Feature groups for occlusion attribution (Section 8): group -> MODEL_COLUMNS members.
ATTR_GROUPS: dict[str, list[str]] = {
    "head_pose": ["d_yaw", "d_pitch", "d_roll", "head_x", "head_y", "head_scale", "head_speed"],
    "eye": ["iris_lx", "iris_ly", "iris_rx", "iris_ry", "eye_open_l", "eye_open_r", "blink", *_EYELOOK,
            "gaze_x", "gaze_y", "gaze_in_screen_prob", "off_screen_score"],
    "mouth": ["jaw_open", "mouth_close", "mar", "mouth_energy_1s"],
    "quality": ["n_faces", "primary_face_present", "face_size_frac", "landmark_conf", "quality"],
}

assert len(MODEL_COLUMNS) == len(set(MODEL_COLUMNS))
assert set(MODEL_COLUMNS) <= set(COLUMNS)
assert sorted(c for g in ATTR_GROUPS.values() for c in g) == sorted(MODEL_COLUMNS)
