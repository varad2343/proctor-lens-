"""Temporal scorer training: causal TCN (Model C). Data path, participant-disjoint CV, Platt, provenance are shared
with train_temporal_gbm.

python -m ml.training.train_temporal_tcn --config configs/training/temporal_tcn.yaml --target speaking

Input = window of standardised rows, invalid/NaN steps = 0 (exactly what LearnedScores feeds); the loss is on the LAST
step of each window, so training sees what serving sees. Class-weighted BCE. Augmentation: feature noise, per-window
offset (stand-in for a synthetic calibration shift), random feature dropout, random invalid steps.
# ponytail: no time-warp, no early stopping (fixed epochs), CPU/GPU single device.
"""
from __future__ import annotations

import numpy as np

from ml.training.train_temporal_gbm import cli, cv_oof, finish, make_dataset, windows


def prep(ds, idx: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Samples idx -> (standardised windows (n, window, F) float32 with invalid steps 0, mask (n, window))."""
    W, M = windows(ds, idx)
    return np.where(M[..., None], (W - mu) / sd, 0.0).astype(np.float32), M


def augment(x: np.ndarray, m: np.ndarray, rng: np.random.Generator, aug: dict) -> np.ndarray:
    B, T, F = x.shape
    x = x + rng.normal(0, aug["noise"], x.shape) + rng.normal(0, aug["offset"], (B, 1, F))
    x = x * (rng.random((B, 1, F)) >= aug["feat_drop"])
    drop = rng.random((B, T, 1)) < aug["step_drop"]
    drop[:, -1] = False  # the scored step is always present at serve time
    return (x * (m[..., None] & ~drop)).astype(np.float32)


def train(cfg: dict) -> dict:
    import torch

    from proctorlens.temporal.tcn import build_tcn

    ds, tc = make_dataset(cfg), cfg["tcn"]
    kw = dict(n_features=len(ds.columns), **tc["build"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(tr):
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        rows = ds.X[ds.pos[tr]]  # last-step rows: valid and finite by construction
        mu, sd = rows.mean(0), np.maximum(rows.std(0), 1e-6)
        net = build_tcn(**kw).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        y = ds.y[tr]
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float((~y).sum() / max(y.sum(), 1))], device=dev))
        for _ in range(tc["epochs"]):
            net.train()
            perm = rng.permutation(tr)
            for b in range(0, len(perm), tc["batch"]):
                idx = perm[b:b + tc["batch"]]
                x, m = prep(ds, idx, mu, sd)
                x = torch.from_numpy(augment(x, m, rng, tc["aug"])).to(dev)
                loss = lossf(net(x)[:, -1, 0], torch.from_numpy(ds.y[idx].astype(np.float32)).to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return net, mu, sd

    def predict(model, idx):
        net, mu, sd = model
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), 1024):
                x = torch.from_numpy(prep(ds, idx[b:b + 1024], mu, sd)[0]).to(dev)
                out.append(net(x)[:, -1, 0].cpu().numpy())
        return np.concatenate(out)

    oof, folds = cv_oof(ds, lambda tr, te: predict(fit(tr), te), cfg["n_splits"])
    net, mu, sd = fit(np.arange(len(ds.y)))
    extra = {"tcn": kw, "feature_mean": mu.tolist(), "feature_std": sd.tolist()}  # keys read by LearnedScores
    save = lambda out: torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out / "model.pt")
    return finish(cfg, ds, oof, folds, "tcn", save, extra)


def main(argv=None) -> None:
    cli(train, "configs/training/temporal_tcn.yaml", argv)


if __name__ == "__main__":
    main()
