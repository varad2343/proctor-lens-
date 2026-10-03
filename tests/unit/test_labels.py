"""Label loading, frame labels, honest intervals."""
import tempfile
from pathlib import Path

import numpy as np

from ml.data.labels import frame_labels, gt_events, honest_intervals, load_labels, load_manifest


def _labels(text: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        p.write_text(text, encoding="utf-8")
        return load_labels(p)


CSV = """type,start_ms,end_ms,source,annotator_id,prompt
OFF_SCREEN_SUSTAINED,1000,3000,cue,a1,look left
FACE_ABSENT,5000,8000,annotator,a2,
natural,10000,20000,cue,,
benign_thinking,12000,13000,annotator,a1,
nuisance,19000,25000,cue,,
"""


def test_load_and_frame_labels():
    df = _labels(CSV)
    assert list(df.columns) == ["type", "start_ms", "end_ms", "source", "annotator_id"]
    assert df["start_ms"].dtype == np.int64 and df.loc[2, "annotator_id"] == ""
    t = np.arange(0, 9000, 500)
    y = frame_labels(df, t, ["OFF_SCREEN_SUSTAINED"])
    assert list(t[y]) == [1000, 1500, 2000, 2500]  # [start, end) half-open
    both = frame_labels(df, t, ["OFF_SCREEN_SUSTAINED", "FACE_ABSENT"])
    assert both.sum() == 4 + 6 and not frame_labels(df, t, ["nope"]).any()
    assert frame_labels(df, np.array([]), ["FACE_ABSENT"]).shape == (0,)


def test_gt_and_honest_intervals():
    df = _labels(CSV)
    assert [g["type"] for g in gt_events(df)] == ["OFF_SCREEN_SUSTAINED", "FACE_ABSENT"]  # blocks/benign are not GT
    assert honest_intervals(df) == [(10000, 25000)]  # natural + benign + nuisance, merged


def test_bad_labels_rejected():
    for text in ("type,start_ms\nFACE_ABSENT,1\n", "type,start_ms,end_ms\nFACE_ABSENT,5,1\n",
                 "type,start_ms,end_ms\nFACE_ABSENT,,1\n"):
        try:
            _labels(text)
            assert False, text
        except ValueError:
            pass


def test_manifest_split_filter():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.csv"
        p.write_text("recording,participant,conditions,split,consent_version\n"
                     "P01_s1,P01,dim,train,v1\nP02_s1,P02,,test,v1\nP03_s1,P03,,val,v1\n", encoding="utf-8")
        assert list(load_manifest(p, ["train", "val"])["recording"]) == ["P01_s1", "P03_s1"]
        assert len(load_manifest(p)) == 3 and list(load_manifest(p, "test")["participant"]) == ["P02"]
