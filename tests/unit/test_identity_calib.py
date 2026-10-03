"""Identity calibration: EER / TAR@FAR / tau on hand-made score sets, and enrollment-style pair building."""
import numpy as np

from ml.training.calibrate_identity import calibrate, similarity_sets


def test_hand_made_scores():
    r = calibrate([0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4])  # perfectly separable
    assert r["eer"] == 0.0 and r["tar_at_far"] == 1.0
    assert abs(r["tau"] - 0.5) < 1e-12  # midpoint between best impostor (0.4) and worst genuine (0.6)
    r = calibrate([0.5, 0.6, 0.7, 0.8], [0.3, 0.4, 0.55, 0.65])  # overlapping: FAR=FRR=0.25 at t=0.6
    assert r["eer"] == 0.25 and r["eer_threshold"] == 0.6
    assert r["tar_at_far"] == 0.5 and abs(r["tau"] - 0.675) < 1e-12  # FAR<=1% needs t>=0.7 (above impostor 0.65)
    r = calibrate([0.1, 0.2], [0.8, 0.9])  # inverted: no finite threshold reaches FAR<=1% with any genuine accepted
    assert r["tar_at_far"] == 0.0 and np.isfinite(r["tau"])


def test_pairs_and_roundtrip():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 32))
    emb = {f"P{i}": c + 0.3 * rng.normal(size=(20, 32)) for i, c in enumerate(centers)}
    emb["short"] = rng.normal(size=(3, 32))  # too few embeddings to enroll: ignored, not an error
    g, i = similarity_sets(emb, n_enroll=5)
    assert len(g) == 4 * 15 and len(i) == 4 * 3 * 20  # later frames vs own enrollment; all frames of the 3 others
    assert g.mean() > 0.7 and i.mean() < 0.3 and -1 <= i.min() and g.max() <= 1
    r = calibrate(g, i)
    assert r["eer"] < 0.05 and r["tar_at_far"] > 0.9 and i.mean() < r["tau"] < g.mean()
    try:
        similarity_sets({"a": emb["P0"]})  # one participant: no impostors
        assert False
    except ValueError:
        pass
