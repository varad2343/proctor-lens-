"""Event-level evaluation (spec 12.2) of replay outputs against labels, with participant-bootstrap CIs.

python -m ml.evaluation.run_eval --manifest data/manifest.csv --results data/processed --labels data/labels \
    --split val --out reports/val

Expects per recording `<results>/<recording>/events.json` (+ `features.parquet|csv` for recording length) as
written by `proctorlens replay <video> --out <results>/<recording>`, and `<labels>/<recording>.csv`.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ml.data.labels import find_file, gt_events, honest_intervals, load_labels, load_manifest
from ml.evaluation.metrics import (bootstrap_ci, counts, fa_counts, flagged_time_fraction, latency_summary,
                                   onset_deltas, prf_stat, ratio_stat)
from proctorlens.core.types import EVENT_TYPES

ZERO = dict(tp=0, fp=0, fn=0)


def evaluate(recs: list[dict], n_boot: int = 1000, seed: int = 0, **match_kw) -> tuple[pd.DataFrame, dict]:
    """recs: [{participant, pred: [Event], labels: DataFrame, total_ms}] ->
    (per-event-type DataFrame with P/R/F1 + 95% CIs + onset latency, summary dict with FA/h + flagged fraction)."""
    parts: dict[str, dict] = {}
    lat: dict[str, list] = defaultdict(list)
    for r in recs:
        p = parts.setdefault(r["participant"], dict(cnt={}, fa=dict(n=0, hours=0.0), fl=dict(ms=0.0, total=0.0)))
        pred, gt = list(r["pred"]), gt_events(r["labels"])
        for t, c in counts(pred, gt, **match_kw).items():
            q = p["cnt"].setdefault(t, dict(ZERO))
            for k in q:
                q[k] += c[k]
        n, h = fa_counts(pred, honest_intervals(r["labels"]))
        p["fa"]["n"] += n
        p["fa"]["hours"] += h
        p["fl"]["ms"] += flagged_time_fraction(pred, r["total_ms"]) * r["total_ms"] if r["total_ms"] > 0 else 0.0
        p["fl"]["total"] += max(r["total_ms"], 0)
        for d in onset_deltas(pred, gt, **match_kw):
            lat[d[0]].append(d)
    rows = []
    for t in (t for t in EVENT_TYPES if any(t in p["cnt"] for p in parts.values())):
        items = [p["cnt"].get(t, ZERO) for p in parts.values()]
        row = dict(type=t, n_participants=sum(any(i.values()) for i in items),
                   **{k: sum(i[k] for i in items) for k in ZERO})
        for key in ("precision", "recall", "f1"):
            row[key], row[key + "_lo"], row[key + "_hi"] = bootstrap_ci(items, prf_stat(key), n_boot, seed)
        rows.append({**row, **latency_summary(lat[t])})
    fa = bootstrap_ci([p["fa"] for p in parts.values()], ratio_stat("n", "hours"), n_boot, seed)
    fl = bootstrap_ci([p["fl"] for p in parts.values()], ratio_stat("ms", "total"), n_boot, seed)
    summary = dict(n_participants=len(parts), n_recordings=len(recs),
                   honest_hours=sum(p["fa"]["hours"] for p in parts.values()),
                   fa_per_hour=fa[0], fa_per_hour_lo=fa[1], fa_per_hour_hi=fa[2],
                   flagged_fraction=fl[0], flagged_fraction_lo=fl[1], flagged_fraction_hi=fl[2])
    return pd.DataFrame(rows), summary


def load_split(manifest: str, results: str | None, labels: str, split: str | list[str]) -> list[dict]:
    """Manifest rows of `split` -> evaluate() inputs. A missing events/labels file is an error, never 'no events'.
    results=None -> pred=[] and total_ms from the labels (caller fills both in by running the pipeline)."""
    recs = []
    for r in load_manifest(manifest, split).itertuples():
        lab = load_labels(Path(labels) / f"{r.recording}.csv")
        pred, total = [], float(max(lab["end_ms"].tolist() + [0]))
        if results is not None:
            from proctorlens.io import load_events, load_features

            pred = load_events(find_file(results, r.recording, "events.json"))[1]
            total = max([total] + [float(e.end_ms) for e in pred])  # fallback when there is no features file
            for name in ("features.parquet", "features.csv"):
                try:
                    t = load_features(find_file(results, r.recording, name))["t_ms"]
                    total = float(t.max() - t.min())
                    break
                except FileNotFoundError:
                    pass
        recs.append(dict(recording=r.recording, participant=r.participant, pred=pred, labels=lab, total_ms=total))
    return recs


def _ci(row, k: str, nd: int = 2) -> str:
    return "-" if pd.isna(row[k]) else f"{row[k]:.{nd}f} [{row[k + '_lo']:.{nd}f}, {row[k + '_hi']:.{nd}f}]"


def md_table(df: pd.DataFrame) -> str:
    f = lambda v: "-" if isinstance(v, float) and v != v else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def report_md(title: str, df: pd.DataFrame, s: dict) -> str:
    t = df.apply(lambda r: dict(type=r["type"], participants=r["n_participants"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                                precision=_ci(r, "precision"), recall=_ci(r, "recall"), f1=_ci(r, "f1"),
                                onset_median_ms=r["start_median_ms"], onset_iqr_ms=(r["start_q3_ms"] - r["start_q1_ms"]),
                                emit_median_ms=r["emit_median_ms"]), axis=1, result_type="expand") if len(df) else df
    a = pd.Series(s)
    return (f"# {title}\n\nN = {s['n_participants']} participants, {s['n_recordings']} recordings "
            f"(CIs: 95% participant-level bootstrap; wide with small N).\n\n{md_table(t)}\n\n"
            f"- false alarms per honest hour: {_ci(a, 'fa_per_hour')} over {s['honest_hours']:.2f} h "
            f"(natural + nuisance + benign_* time; MONITORING_DEGRADED not counted)\n"
            f"- flagged-time fraction: {_ci(a, 'flagged_fraction', 3)}\n")


def write_report(out: str | Path, name: str, df: pd.DataFrame, s: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}_types.csv", index=False)
    (out / f"{name}_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    (out / f"{name}.md").write_text(report_md(name, df, s), encoding="utf-8")


def stack(results: dict[str, tuple[pd.DataFrame, dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """{variant: (types_df, summary)} -> (types_df with `variant` column, one summary row per variant)."""
    types = pd.concat([d.assign(variant=k) for k, (d, _) in results.items()], ignore_index=True)
    return types, pd.DataFrame([{"variant": k, **s} for k, (_, s) in results.items()])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--results", required=True)
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--split", default="val", help="use `test` exactly once, for the final report")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    df, s = evaluate(load_split(a.manifest, a.results, a.labels, a.split), a.boot)
    write_report(a.out, f"eval_{a.split}", df, s)
    print((Path(a.out) / f"eval_{a.split}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
