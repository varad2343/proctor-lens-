import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from proctorlens.core.config import Config
from proctorlens.core.types import Event
from proctorlens.io import load_events, load_features, make_header, save_events, save_features


def _df():
    return pd.DataFrame({"t_ms": [0, 100, 200], "frame_valid": [True, False, True],
                         "d_yaw": [1.5, np.nan, -3.0], "zone": ["none", "", "right"], "quality_reasons": ["", "dark", ""]})


def test_features_roundtrip_csv_and_parquet_path():
    d = Path(tempfile.mkdtemp())
    df = _df()
    p = save_features(df, d / "f.csv")
    assert p.name == "f.csv"
    pd.testing.assert_frame_equal(load_features(p), df)
    # .parquet path: real parquet with pyarrow, else transparently csv; load_features(path) works either way
    p = save_features(df, d / "f.parquet")
    assert p.exists() and p.suffix in (".parquet", ".csv")
    pd.testing.assert_frame_equal(load_features(d / "f.parquet"), df)


def test_events_roundtrip_is_strict_json():
    d = Path(tempfile.mkdtemp())
    ev = Event("FACE_ABSENT", 1000, 5000, 0.9, "rule@0.1.0",
              details={"peak": np.float32(0.5), "calib_error": float("nan"), "n": np.int64(3), "zones": ("a", "b")},
              attribution={"eye": 0.25})
    hdr = make_header(Config(), None)
    save_events([ev], d / "events.json", hdr)
    json.loads((d / "events.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))  # no NaN
    h, evs = load_events(d / "events.json")
    assert h == hdr and h["calibration"] is None and h["config_hash"] == make_header(Config())["config_hash"]
    (e,) = evs
    assert (e.type, e.start_ms, e.end_ms, e.status, e.attribution) == ("FACE_ABSENT", 1000, 5000, "final", {"eye": 0.25})
    assert e.details == {"peak": 0.5, "calib_error": None, "n": 3, "zones": ["a", "b"]}
