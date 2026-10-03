"""Config: defaults, YAML deep-merge, strict validation, stable hash. No fixtures (pytest + tests/run.py)."""
import tempfile
from pathlib import Path

from proctorlens.core.config import Config, config_hash, load_config
from proctorlens.core.types import SCORE_KEYS

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _load(*yamls: str) -> Config:
    """Write each YAML string to a temp file and load_config() them in order."""
    with tempfile.TemporaryDirectory() as d:
        ps = []
        for i, y in enumerate(yamls):
            ps.append(Path(d) / f"{i}.yaml")
            ps[-1].write_text(y, encoding="utf-8")
        return load_config(*ps)


def _rejects(*yamls: str) -> bool:
    try:
        _load(*yamls)
    except ValueError:
        return True
    return False


def test_defaults():
    cfg = load_config()
    assert cfg == Config()
    assert set(cfg.events) == set(cfg.policy.active) == set(SCORE_KEYS)  # one machine per score key
    # spec-mandated defaults (Sections 3, 5.3, 7)
    e = cfg.events
    assert (e["face_absent"].t_on_s, e["phone"].t_on_s, e["notes"].t_on_s, e["off_screen"].t_on_s) == (3.0, 1.5, 3.0, 4.0)
    assert (cfg.glance.n, cfg.glance.window_s, cfg.glance.same_zone_frac) == (4, 60.0, 0.7)
    assert (cfg.glance.min_s, cfg.glance.max_s) == (0.3, 3.0)
    assert (cfg.identity.consecutive, cfg.gaze.accept_error, cfg.gaze.max_tries) == (3, 0.15, 2)
    assert cfg.pipeline.grid_hz == 10 and cfg.scorer.provider == "rule"
    assert all(x.off_thr < x.on_thr for x in e.values())
    load_config().events["phone"].t_on_s = 99.0  # no shared mutable defaults between loads
    assert load_config().events["phone"].t_on_s == 1.5


def test_yaml_merge():
    cfg = _load("events: {phone: {t_on_s: 2.5}}\npolicy: {allow_notes: true}\n",
                "events: {phone: {t_off_s: 0.25}}\npolicy: {active: [phone, degraded]}\n")
    p = cfg.events["phone"]
    assert (p.t_on_s, p.t_off_s) == (2.5, 0.25)  # both files applied, per key
    assert p.on_thr == 0.5 and p.min_dur_s == 1.5  # untouched fields keep defaults
    assert cfg.events["notes"] == Config().events["notes"]  # other events untouched
    assert cfg.policy.allow_notes and cfg.policy.allow_looking_down is False
    assert cfg.policy.active == ["phone", "degraded"]  # lists replace, never append
    assert _load("gaze: {margin: 0.2}", "gaze: {margin: 0.3}").gaze.margin == 0.3  # later file wins
    assert _load("").pipeline == Config().pipeline  # empty file = defaults


def test_shipped_configs_load():
    cfg = load_config(CONFIGS / "pipeline.yaml", CONFIGS / "policy.yaml")
    assert cfg.models.landmarker.endswith("face_landmarker.task") and cfg.models.detector_conf == 0.30
    assert cfg.events["off_screen"].off_thr == 0.35 and cfg.events["off_screen"].t_on_s == 4.0
    assert cfg.scorer.provider == "rule" and cfg.scorer.models == {}


def test_rejects_unknown_and_invalid():
    for y in ("bogus: 1", "gaze: {nope: 1}", "events: {phone: {nope: 1}}", "events: {bogus: {on_thr: 0.6}}",
              "policy: {active: [phone, bogus]}", "scorer: {provider: svm}"):
        assert _rejects(y), y


def test_off_thr_must_be_below_on_thr():
    assert _rejects("events: {phone: {off_thr: 0.5}}")  # equal
    assert _rejects("events: {phone: {off_thr: 0.8}}")  # above
    assert _rejects("events: {phone: {on_thr: 0.2}}")  # default off_thr 0.3 now >= on_thr
    assert not _rejects("events: {phone: {on_thr: 0.9, off_thr: 0.8}}")


def test_config_hash_stable():
    h = config_hash(Config())
    assert len(h) == 12 and int(h, 16) >= 0
    assert h == config_hash(load_config()) == config_hash(_load(""))
    a = _load("gaze: {margin: 0.2, yaw_limit_deg: 30.0}")
    b = _load("gaze: {yaw_limit_deg: 30.0, margin: 0.2}")  # key order must not matter
    assert config_hash(a) == config_hash(b) != h
    assert config_hash(_load("events: {phone: {t_on_s: 1.5}}")) == h  # restating a default = same hash
    assert config_hash(_load("events: {phone: {t_on_s: 2}}")) == config_hash(_load("events: {phone: {t_on_s: 2.0}}"))  # int -> float
    assert isinstance(_load("gaze: {margin: 1}").gaze.margin, float)
    for y in ("events: {phone: {t_on_s: 2.0}}", "policy: {allow_notes: true}", "scorer: {provider: gbm}"):
        assert config_hash(_load(y)) != h, y
