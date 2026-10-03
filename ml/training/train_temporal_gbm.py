"""Temporal scorer training: shared data path + LightGBM on window statistics (Baseline B). The TCN trainer reuses it.

python -m ml.training.train_temporal_gbm --config configs/training/temporal_gbm.yaml --target off_screen

Sample = (recording, grid step) whose last row is reliable and whose causal window (cfg.window rows) is full, i.e. exactly
when LearnedScores would call the model. CV is participant-disjoint GroupKFold over the train+val splits (the `test`
split is never read). Platt (a, b) is fitted on the out-of-fold raw scores (p = sigmoid(a * raw + b)); the shipped model
is refit on all train+val participants. Event-level selection (F1, false alarms/h) happens afterwards: replay with the
model and run ml.evaluation.run_eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.data.labels import find_file, frame_labels, load_labels, load_manifest
from proctorlens.features.schema import MODEL_COLUMNS


@dataclass
class Dataset:
    X: np.ndarray  # (R, F) float32: rows of all recordings concatenated (NaN where undefined)
    M: np.ndarray  # (R,) bool: row valid = frame_valid and every column finite (== WindowBuffer.mask)
    pos: np.ndarray  # (n,) row index of each sample's LAST step; window = X[pos-window+1 : pos+1]
    y: np.ndarray  # (n,) bool frame label at the last step
    groups: np.ndarray  # (n,) participant
    rec: np.ndarray  # (n,) recording id
    t_ms: np.ndarray  # (n,)
    columns: list[str]
    window: int


def row_arrays(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[columns].to_numpy(np.float32)
    return X, df["frame_valid"].to_numpy(bool) & np.isfinite(X).all(1)


def build_dataset(recs: list[dict], columns: list[str], window: int, stride: int, types: list[str]) -> Dataset:
    """recs: [{rec, participant, df: features DataFrame, labels: DataFrame}] -> Dataset (pure numpy/pandas)."""
    Xs, Ms, pos, ys, gs, rs, ts, off = [], [], [], [], [], [], [], 0
    for r in recs:
        X, m = row_arrays(r["df"], columns)
        t = r["df"]["t_ms"].to_numpy()
        i = np.arange(window - 1, len(X), stride)
        i = i[m[i] & r["df"]["reliable"].to_numpy(bool)[i]]  # only where serving would call the model
        Xs.append(X), Ms.append(m), pos.append(off + i), ts.append(t[i])
        ys.append(frame_labels(r["labels"], t, types)[i])
        gs += [r["participant"]] * len(i)
        rs += [r["rec"]] * len(i)
        off += len(X)
    return Dataset(np.concatenate(Xs), np.concatenate(Ms), np.concatenate(pos), np.concatenate(ys), np.array(gs),
                   np.array(rs), np.concatenate(ts), list(columns), window)


def windows(ds: Dataset, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Samples idx -> (W (n, window, F), mask (n, window))."""
    g = ds.pos[idx, None] + np.arange(1 - ds.window, 1)
    return ds.X[g], ds.M[g]


def cv_oof(ds: Dataset, fit_predict, n_splits: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Participant-disjoint GroupKFold. fit_predict(train_idx, test_idx) -> raw scores of test_idx.
    Returns (out-of-fold raw scores, test index array per fold)."""
    from sklearn.model_selection import GroupKFold

    k = min(n_splits, len(set(ds.groups)))
    if k < 2:
        raise ValueError("participant-disjoint CV needs >= 2 participants")
    oof, folds = np.full(len(ds.y), np.nan), []
    for tr, te in GroupKFold(k).split(ds.y, ds.y, ds.groups):
        oof[te] = fit_predict(tr, te)
        folds.append(te)
    return oof, folds


def platt_fit(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * raw + b) on held-out scores; (1, 0) when only one class is present."""
    if y.min() == y.max():
        return 1.0, 0.0
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(raw, float).reshape(-1, 1), y)  # ~unregularised
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def frame_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Frame-level PR-AUC, F1@0.5, ECE + reliability bins (for the diagram) of calibrated probabilities p."""
    from sklearn.metrics import average_precision_score

    pred, tp = p >= 0.5, int(((p >= 0.5) & y).sum())
    bins, rel = np.minimum((p * 10).astype(int), 9), []
    for b in range(10):
        if (m := bins == b).any():
            rel.append(dict(bin_lo=b / 10, n=int(m.sum()), mean_p=float(p[m].mean()), frac_pos=float(y[m].mean())))
    return dict(pr_auc=float(average_precision_score(y, p)) if y.min() != y.max() else float("nan"),
                f1_at_0_5=2 * tp / (pred.sum() + y.sum()) if pred.sum() + y.sum() else float("nan"),
                pos_rate=float(y.mean()), ece=sum(r["n"] * abs(r["mean_p"] - r["frac_pos"]) for r in rel) / len(y),
                reliability=rel)


def _git_hash() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None if r.returncode == 0 else None


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def provenance(cfg: dict) -> dict:
    """git hash (None when git/repo unavailable), data-manifest hash, seed."""
    return dict(git=_git_hash(), manifest_sha256=sha256_file(cfg["manifest"]), seed=cfg["seed"])


def load_cfg(path: str, target: str) -> dict:
    """Training YAML: shared keys + targets.<target> overrides."""
    c = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {**{k: v for k, v in c.items() if k != "targets"}, **c["targets"][target], "target": target}


def load_recordings(cfg: dict) -> list[dict]:
    from proctorlens.io import load_features

    if "test" in cfg["splits"]:
        raise ValueError("training must not read the test split")
    recs = []
    for r in load_manifest(cfg["manifest"], cfg["splits"]).itertuples():
        for name in ("features.parquet", "features.csv"):
            try:
                df = load_features(find_file(cfg["features_dir"], r.recording, name))
                break
            except FileNotFoundError:
                pass
        else:
            raise FileNotFoundError(f"no features for {r.recording} in {cfg['features_dir']}")
        recs.append(dict(rec=r.recording, participant=r.participant, df=df,
                         labels=load_labels(Path(cfg["labels_dir"]) / f"{r.recording}.csv")))
    return recs


def make_dataset(cfg: dict) -> Dataset:
    recs = load_recordings(cfg)
    ds = build_dataset(recs, MODEL_COLUMNS, cfg["window"], cfg["stride"], cfg["label_types"])
    if skipped := sorted({r["rec"] for r in recs} - set(ds.rec)):  # e.g. head_pose_only users: gaze_x/gaze_y are NaN
        print(f"warning: no usable samples from {skipped} (too short, or no reliable row with every model column "
              "finite)", file=sys.stderr)
    if ds.y.min() == ds.y.max():
        raise ValueError(f"target {cfg['target']!r}: {len(ds.y)} samples, all {bool(ds.y[:1].any())} - check label_types")
    return ds


def finish(cfg: dict, ds: Dataset, oof: np.ndarray, folds: list, kind: str, save_weights, extra: dict | None) -> dict:
    """Platt on OOF scores, write the model dir (meta.json + weights), metrics.json, config.yaml."""
    from sklearn.metrics import average_precision_score

    from proctorlens.temporal.scores_learned import write_model_dir

    a, b = platt_fit(oof, ds.y)
    p = 1 / (1 + np.exp(-(a * oof + b)))
    out = Path(cfg["out_dir"]) / cfg["name"] / cfg["version"]
    write_model_dir(out, kind, cfg["target"], ds.columns, ds.window, {"a": a, "b": b}, extra)
    save_weights(out)
    fold_ap = [float(average_precision_score(ds.y[te], oof[te])) if ds.y[te].min() != ds.y[te].max() else None
               for te in folds]
    metrics = dict(target=cfg["target"], kind=kind, label_types=cfg["label_types"], n_samples=len(ds.y),
                   n_participants=len(set(ds.groups)), n_recordings=len(set(ds.rec)), platt=dict(a=a, b=b),
                   oof=frame_metrics(ds.y, p), fold_pr_auc=fold_ap,
                   fold_participants=[sorted(set(ds.groups[te])) for te in folds], **provenance(cfg))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return metrics


def cli(train, default_config: str, argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--target", required=True, choices=["off_screen", "speaking"])
    a = ap.parse_args(argv)
    m = train(load_cfg(a.config, a.target))
    print(json.dumps({k: m[k] for k in ("target", "kind", "n_samples", "n_participants", "platt")} |
                     {"oof_pr_auc": m["oof"]["pr_auc"]}, indent=1))


def train(cfg: dict) -> dict:
    import lightgbm as lgb

    from proctorlens.features.windows import window_stats

    ds = make_dataset(cfg)
    F = np.stack([window_stats(ds.X[p - ds.window + 1:p + 1], ds.M[p - ds.window + 1:p + 1], ds.columns) for p in ds.pos])
    params = {"objective": "binary", "seed": cfg["seed"], "verbose": -1, **cfg["gbm"]["params"]}

    def fit(tr):
        y = ds.y[tr]
        w = float((~y).sum() / max(y.sum(), 1))  # class weighting: neg/pos
        return lgb.train({**params, "scale_pos_weight": w}, lgb.Dataset(F[tr], label=y.astype(int)), cfg["gbm"]["rounds"])

    oof, folds = cv_oof(ds, lambda tr, te: fit(tr).predict(F[te], raw_score=True), cfg["n_splits"])
    final = fit(np.arange(len(F)))
    return finish(cfg, ds, oof, folds, "gbm", lambda out: final.save_model(str(out / "model.txt")), None)


def main(argv=None) -> None:
    cli(train, "configs/training/temporal_gbm.yaml", argv)


if __name__ == "__main__":
    main()
