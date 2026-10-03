"""Training-data path (windowed datasets, participant-disjoint CV, Platt), TCN input prep, YOLO conversion,
manifest splits, frame extraction, shipped training configs. No torch / lightgbm needed."""
import inspect
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from ml.data.build_manifest import assign_splits, build
from ml.data.extract_frames import extract
from ml.data.labels import load_manifest
from ml.data.to_yolo_format import convert, parse_labelstudio
from ml.training.train_temporal_gbm import build_dataset, cv_oof, frame_metrics, load_cfg, platt_fit, windows
from ml.training.train_temporal_tcn import augment, prep
from proctorlens.core.config import Config
from proctorlens.core.types import EVENT_TYPES

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "training"
COLS = ["d_yaw", "jaw_open"]
A = "OFF_SCREEN_SUSTAINED"


def _rec(rec, participant, n, label=(3000, 5000), bad=(), seed=0):
    """n grid rows at 10 Hz; d_yaw carries the label signal; `bad` = {step: what} with 'invalid'|'unreliable'|'nan'."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 100
    lab = (t >= label[0]) & (t < label[1])
    df = pd.DataFrame(dict(t_ms=t, frame_valid=True, reliable=True, d_yaw=3.0 * lab + 0.3 * rng.normal(size=n),
                           jaw_open=rng.random(n)))
    for i, what in dict(bad).items():
        if what == "invalid":
            df.loc[i, "frame_valid"] = False
        elif what == "unreliable":
            df.loc[i, "reliable"] = False
        else:
            df.loc[i, "d_yaw"] = np.nan
    labels = pd.DataFrame([[A, label[0], label[1], "cue", ""]], columns=["type", "start_ms", "end_ms", "source", "annotator_id"])
    return dict(rec=rec, participant=participant, df=df, labels=labels)


def test_build_dataset_windows_and_parity():
    bad = {50: "invalid", 60: "nan", 70: "unreliable", 71: "unreliable", 72: "unreliable"}
    recs = [_rec("r1", "A", 100, bad=bad), _rec("r2", "B", 40, label=(0, 0))]
    ds = build_dataset(recs, COLS, window=10, stride=1, types=[A])
    n1 = (ds.rec == "r1").sum()
    assert n1 == 91 - 5 and (ds.rec == "r2").sum() == 40 - 9  # full window + valid & reliable last step only
    assert set(ds.groups) == {"A", "B"} and ds.X.shape == (140, 2) and ds.M.sum() == 140 - 2  # invalid + nan rows
    p1 = ds.pos[ds.rec == "r1"]
    assert p1.max() <= 99 and ds.pos[ds.rec == "r2"].min() - 9 >= 100  # a window never crosses recordings
    assert ds.y[ds.rec == "r1"].sum() == 20 and ds.y[ds.rec == "r2"].sum() == 0  # steps 30..49 (t in [3000, 5000))
    assert {50, 60, 70, 71, 72}.isdisjoint(set(p1))
    W, M = windows(ds, np.array([0, 5]))
    assert W.shape == (2, 10, 2) and (W[:, -1] == ds.X[ds.pos[[0, 5]]]).all()
    s3 = build_dataset(recs, COLS, window=10, stride=3, types=[A])
    assert set(s3.pos) <= set(ds.pos) and len(s3.pos) < len(ds.pos) / 2
    try:  # parity with the serving-side buffer: same window values where valid, same mask
        from proctorlens.features.windows import WindowBuffer
    except ImportError:
        return
    buf, rows = WindowBuffer(10, COLS), recs[0]["df"].to_dict("records")
    for i, r in enumerate(rows):
        buf.push(r)
        if i in set(p1):
            w, m = windows(ds, np.array([int(np.flatnonzero(ds.pos == i)[0])]))
            assert (buf.mask() == m[0]).all() and np.allclose(buf.array()[m[0]], w[0][m[0]]), i


def test_cv_is_participant_disjoint_and_platt():
    recs = [_rec(f"r{i}", f"P{i % 4}", 120, label=(2000, 6000), seed=i) for i in range(8)]  # 4 participants, 2 recs each
    ds = build_dataset(recs, COLS, window=10, stride=2, types=[A])
    seen = []

    def fit_predict(tr, te):
        assert not set(ds.groups[tr]) & set(ds.groups[te])  # no participant in both train and test
        seen.append(set(ds.groups[te]))
        from sklearn.linear_model import LogisticRegression

        X = ds.X[ds.pos]
        return LogisticRegression().fit(X[tr], ds.y[tr]).decision_function(X[te])

    oof, folds = cv_oof(ds, fit_predict, n_splits=5)  # capped at 4 participants
    assert len(folds) == 4 and np.isfinite(oof).all() and sorted(np.concatenate(folds)) == list(range(len(ds.y)))
    assert set().union(*seen) == {"P0", "P1", "P2", "P3"}
    a, b = platt_fit(oof, ds.y)
    m = frame_metrics(ds.y, 1 / (1 + np.exp(-(a * oof + b))))
    assert m["pr_auc"] > 0.9 and m["ece"] < 0.1 and 0 < m["pos_rate"] < 1
    try:
        cv_oof(build_dataset(recs[:1], COLS, 10, 2, [A]), fit_predict, 5)  # one participant: refuse
        assert False
    except ValueError:
        pass


def test_platt_recovers_known_calibration():
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 2, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-(0.5 * raw - 1.0)))
    a, b = platt_fit(raw, y)
    assert abs(a - 0.5) < 0.05 and abs(b + 1.0) < 0.1
    assert platt_fit(raw, np.zeros(20000, bool)) == (1.0, 0.0)  # single class: identity


def test_gbm_training_end_to_end_with_stub_lightgbm():
    """Whole train_temporal_gbm.train() on synthetic recordings; lightgbm (not installed here) is stubbed by a logistic model."""
    import sys
    import types

    from sklearn.linear_model import LogisticRegression

    from ml.training.train_temporal_gbm import train
    from proctorlens.features.schema import MODEL_COLUMNS

    class Booster:
        def __init__(self, m):
            self.m = m

        def predict(self, X, raw_score=False):
            assert raw_score  # serving applies Platt to raw scores
            return self.m.decision_function(np.nan_to_num(X))

        def save_model(self, path):
            Path(path).write_text("stub")

    stub = types.ModuleType("lightgbm")
    stub.Dataset = lambda X, label=None: (X, label)
    stub.train = lambda params, ds, rounds: Booster(LogisticRegression(max_iter=300).fit(np.nan_to_num(ds[0]), ds[1]))
    saved, rng = sys.modules.get("lightgbm"), np.random.default_rng(0)
    sys.modules["lightgbm"] = stub
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "lab").mkdir()
            rows = []
            for i in range(4):
                rec, t = f"P{i}_s1", np.arange(300) * 100
                rows.append([rec, f"P{i}", "", ["train", "val"][i % 2], "v1"])
                df = pd.DataFrame(rng.random((300, len(MODEL_COLUMNS))), columns=MODEL_COLUMNS)
                df["d_yaw"] = 3.0 * ((t >= 10000) & (t < 20000)) + 0.2 * rng.normal(size=300)
                df.insert(0, "t_ms", t)
                df["frame_valid"], df["reliable"] = True, True
                (d / "res" / rec).mkdir(parents=True)
                df.to_csv(d / "res" / rec / "features.csv", index=False)
                (d / "lab" / f"{rec}.csv").write_text(f"type,start_ms,end_ms\n{A},10000,20000\n")
            pd.DataFrame(rows, columns=["recording", "participant", "conditions", "split", "consent_version"]).to_csv(
                d / "m.csv", index=False)
            cfg = dict(target="off_screen", label_types=[A], name="off_gbm", version="v1", window=20, stride=2, n_splits=4,
                       seed=0, splits=["train", "val"], manifest=str(d / "m.csv"), features_dir=str(d / "res"),
                       labels_dir=str(d / "lab"), out_dir=str(d / "models"), gbm=dict(rounds=5, params={}))
            m = train(cfg)
            out = d / "models" / "off_gbm" / "v1"
            meta = json.loads((out / "meta.json").read_text())
            assert meta["kind"] == "gbm" and meta["target"] == "off_screen" and meta["columns"] == MODEL_COLUMNS
            assert meta["window"] == 20 and meta["version"] == "v1" and set(meta["calib"]) == {"a", "b"}
            assert meta["calib"]["a"] > 0  # higher raw score => higher probability
            assert (out / "model.txt").exists() and (out / "config.yaml").exists()
            saved_m = json.loads((out / "metrics.json").read_text())
            assert saved_m["n_participants"] == 4 and saved_m["oof"]["pr_auc"] > 0.9
            assert saved_m["oof"]["pr_auc"] == m["oof"]["pr_auc"]
            assert len(saved_m["fold_pr_auc"]) == 4 and saved_m["manifest_sha256"] and "git" in saved_m
            try:  # the test split must never be read by training
                train({**cfg, "splits": ["train", "test"]})
                assert False
            except ValueError:
                pass
    finally:
        sys.modules.pop("lightgbm", None)
        if saved is not None:
            sys.modules["lightgbm"] = saved


def test_tcn_input_prep_and_augment():
    recs = [_rec("r1", "A", 60, bad={30: "nan"})]
    ds = build_dataset(recs, COLS, window=10, stride=1, types=[A])
    mu, sd = np.array([1.0, 0.0], np.float32), np.array([2.0, 1.0], np.float32)
    x, m = prep(ds, np.arange(len(ds.y)), mu, sd)
    assert x.dtype == np.float32 and x.shape == (len(ds.y), 10, 2) and np.isfinite(x).all()
    W, M = windows(ds, np.arange(len(ds.y)))
    assert (x[~m] == 0).all() and (m == M).all() and np.allclose(x[m], ((W - mu) / sd)[m])  # (x - mean) / std, invalid -> 0
    assert (~m).any()  # the NaN row at step 30 is inside some windows
    rng = np.random.default_rng(0)
    xs, ms = np.ones((4, 10, 3), np.float32), np.ones((4, 10), bool)
    ms[:, 2] = False
    none = dict(noise=0.0, offset=0.0, feat_drop=0.0, step_drop=0.0)
    assert (augment(xs, ms, rng, none) == xs * ms[..., None]).all()  # invalid steps stay 0
    y = augment(xs, ms, rng, {**none, "step_drop": 1.0})
    assert (y[:, -1] == 1).all() and (y[:, :-1] == 0).all()  # the scored (last) step is never dropped
    assert (augment(xs, ms, rng, {**none, "feat_drop": 1.0}) == 0).all()


def _manifest(rows):
    return pd.DataFrame(rows, columns=["recording", "participant", "conditions", "split", "consent_version"])


def test_to_yolo_format_participant_disjoint():
    def rect(label, x, y, w, h):
        return {"type": "rectanglelabels", "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [label]}}

    ls = [{"data": {"image": "/data/upload/1/abcdef12-P01_s1_00000100.jpg"}, "annotations": [{"result": [
              rect("phone", 10, 20, 30, 40), rect("remote", 0, 0, 50, 50), rect("Person", 60, 60, 50, 50)]}]},
          {"data": {"image": "P02_s1_00000200.jpg"}, "annotations": [{"result": []}]},  # annotated, nothing in view
          {"data": {"image": "P03_s1_00000300.jpg"}, "annotations": [{"was_cancelled": True, "result": []}]}]  # skipped
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ls.json").write_text(json.dumps(ls), encoding="utf-8")
        (d / "img").mkdir()
        for n in ("P01_s1_00000100", "P02_s1_00000200"):
            cv2.imwrite(str(d / "img" / f"{n}.jpg"), np.zeros((20, 30, 3), np.uint8))
        ann = parse_labelstudio(d / "ls.json")
        assert sorted(ann) == ["P01_s1_00000100.jpg", "P02_s1_00000200.jpg"]  # hash prefix stripped, cancelled dropped
        man = _manifest([["P01_s1", "P01", "", "train", "v1"], ["P02_s1", "P02", "", "test", "v1"]])
        assert dict(convert(ann, d / "img", man, d / "yolo")) == {"train": 1, "test": 1}
        lines = (d / "yolo/labels/train/P01_s1_00000100.txt").read_text().split("\n")[:-1]
        assert lines == ["0 0.250000 0.400000 0.300000 0.400000",  # phone = 0; 'remote' dropped
                         "2 0.800000 0.800000 0.400000 0.400000"]  # person box clamped at the image border
        assert (d / "yolo/labels/test/P02_s1_00000200.txt").read_text() == ""  # hard negative image
        assert (d / "yolo/images/train/P01_s1_00000100.jpg").exists() and not (d / "yolo/images/test/P01_s1_00000100.jpg").exists()
        spec = yaml.safe_load((d / "yolo/dataset.yaml").read_text())
        assert spec["names"] == {0: "phone", 1: "notes", 2: "person"} and "val" not in spec and spec["train"] == "images/train"
        convert(ann, d / "img", man, d / "coco", coco_ids=True)
        assert (d / "coco/labels/train/P01_s1_00000100.txt").read_text().startswith("67 ")
        split_twice = _manifest([["P01_s1", "P01", "", "train", "v1"], ["P01_s2", "P01", "", "test", "v1"]])
        for bad_man, exc in ((split_twice, ValueError), (_manifest([["P09_s1", "P09", "", "train", "v1"]]), KeyError)):
            try:
                convert(ann, d / "img", bad_man, d / "x")
                assert False
            except exc:
                pass


def test_manifest_splits_are_stable_and_disjoint():
    ps = [f"P{i:02d}" for i in range(10)]
    s = assign_splits(ps)
    assert sorted(pd.Series(s).value_counts().to_dict().items()) == [("test", 2), ("train", 6), ("val", 2)]
    assert s == assign_splits(reversed(ps))  # deterministic, independent of input order
    s2 = assign_splits(ps + ["P10", "P11"], fixed=s)
    assert all(s2[p] == s[p] for p in ps) and len(s2) == 12  # earlier participants never move
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for n in ("P01_s1", "P01_s2", "P02_s1", "P03_s1", "P04_s1"):
            (d / f"{n}.mp4").write_bytes(b"")
        (d / "P02_s1.meta.json").write_text(json.dumps(dict(participant="P02", conditions="dim|glasses", consent_version="v2")))
        try:
            build(str(d), str(d / "m.csv"))  # no consent information anywhere
            assert False
        except ValueError:
            pass
        m = build(str(d), str(d / "m.csv"), "v1")
        assert len(m) == 5 and m.set_index("recording").loc["P02_s1", "consent_version"] == "v2"
        assert (m.groupby("participant")["split"].nunique() == 1).all()  # participant-disjoint
        m.loc[m["recording"] == "P03_s1", "conditions"] = "hand edited"
        m.to_csv(d / "m.csv", index=False)
        (d / "P05_s1.mp4").write_bytes(b"")
        m2 = build(str(d), str(d / "m.csv"), "v1")
        assert len(m2) == 6 and m2.set_index("recording").loc["P03_s1", "conditions"] == "hand edited"
        assert dict(zip(m["participant"], m["split"])).items() <= dict(zip(m2["participant"], m2["split"])).items()
        m2.to_csv(d / "m.csv", index=False)
        assert len(load_manifest(d / "m.csv", "train")) + len(load_manifest(d / "m.csv", ["val", "test"])) == 6


def test_extract_frames():
    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "P01_s1.avi")
        w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not w.isOpened():
            return  # no MJPG writer in this OpenCV build
        for i in range(25):
            w.write(np.full((48, 64, 3), i * 9, np.uint8))
        w.release()
        paths = extract(src, Path(d) / "frames", every_s=0.5)
        assert 4 <= len(paths) <= 6 and all(p.exists() and p.name.startswith("P01_s1_") for p in paths)
        ts = [int(p.stem.rsplit("_", 1)[1]) for p in paths]
        assert ts == sorted(ts) and all(b - a >= 490 for a, b in zip(ts, ts[1:]))  # about every 0.5 s of video time


def test_shipped_training_configs_match_their_consumers():
    from proctorlens.temporal.tcn import build_tcn  # torch is imported inside the factory, not here

    g, t = load_cfg(CONFIGS / "temporal_gbm.yaml", "off_screen"), load_cfg(CONFIGS / "temporal_tcn.yaml", "speaking")
    assert g["target"] == "off_screen" and g["name"] == "off_screen_gbm" and g["label_types"] == [A]
    assert t["target"] == "speaking" and t["name"] == "speaking_tcn" and t["label_types"] == ["MOUTH_ACTIVITY"]
    for c in (g, t):
        assert c["window"] == Config().scorer.window and "test" not in c["splits"]
        assert set(c["label_types"]) <= set(EVENT_TYPES)
        assert {"manifest", "features_dir", "labels_dir", "out_dir", "stride", "n_splits", "seed", "version"} <= set(c)
    assert {"rounds", "params"} <= set(g["gbm"])
    assert {"epochs", "batch", "lr", "weight_decay", "build", "aug"} <= set(t["tcn"])
    assert {"noise", "offset", "feat_drop", "step_drop"} == set(t["tcn"]["aug"])
    inspect.signature(build_tcn).bind(n_features=3, **t["tcn"]["build"])  # the YAML keys are real build_tcn kwargs
    d = yaml.safe_load((CONFIGS / "detector.yaml").read_text())
    assert {"model", "data", "out_dir", "name", "version", "conf", "train"} <= set(d) and d["train"]["imgsz"] == 640
    assert d["conf"] == Config().models.detector_conf
