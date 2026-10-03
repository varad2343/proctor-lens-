import math

from proctorlens.core.config import Config
from proctorlens.core.types import SCORE_KEYS
from proctorlens.temporal.scores_rule import RuleScores, soft

NAN = math.nan


def row(t=0, **kw):
    r = dict(t_ms=t, frame_valid=True, reliable=True, primary_face_present=True, turned_away=False,
             n_faces=1, n_persons=1, phone_conf=0.0, notes_conf=0.0, off_screen_score=0.0, zone="on_screen",
             mouth_energy_1s=0.0, jaw_open=0.1, id_similarity=NAN, id_quality_ok=False)
    r.update(kw)
    return r


def cfg(**policy):
    c = Config()
    for k, v in policy.items():
        setattr(c.policy, k, v)
    return c


def test_soft_and_clean_row():
    assert soft(1.0, 1.5, 1.0) == 0.0 and soft(1.5, 1.5, 1.0) == 0.5 and soft(2.0, 1.5, 1.0) == 1.0
    s = RuleScores(Config()).update(row())
    assert tuple(s) == SCORE_KEYS and all(v == 0.0 for v in s.values())


def test_face_absent_turned_away_and_people():
    r = RuleScores(Config())
    assert r.update(row(primary_face_present=False, n_persons=0))["face_absent"] == 1.0
    assert r.update(row(primary_face_present=False, n_persons=0, turned_away=True))["face_absent"] == 0.0
    assert r.update(row(primary_face_present=False, n_persons=1))["face_absent"] == 0.0  # person in view
    assert r.update(row(n_faces=2))["multiple_people"] == 1.0
    assert r.update(row(n_persons=2))["multiple_people"] == 1.0
    assert r.update(row(n_faces=1, n_persons=1))["multiple_people"] == 0.0


def test_objects_and_policy_toggles():
    assert RuleScores(Config()).update(row(phone_conf=0.8, notes_conf=0.6))["phone"] == 0.8
    assert RuleScores(Config()).update(row(notes_conf=0.6))["notes"] == 0.6
    assert RuleScores(cfg(allow_notes=True)).update(row(notes_conf=0.6))["notes"] == 0.0
    down = row(off_screen_score=0.9, zone="down")
    assert RuleScores(Config()).update(down)["off_screen"] == 0.9
    assert RuleScores(cfg(allow_looking_down=True)).update(down)["off_screen"] == 0.0
    left = row(off_screen_score=0.9, zone="left")
    assert RuleScores(cfg(allow_looking_down=True)).update(left)["off_screen"] == 0.9
    assert RuleScores(Config()).update(row(off_screen_score=1.0, zone="right", primary_face_present=False,
                                           turned_away=True))["off_screen"] == 1.0
    noface = row(off_screen_score=NAN, zone="none")
    assert RuleScores(Config()).update(noface)["off_screen"] is None


def test_unreliable_and_invalid_rows_give_none_not_zero():
    s = RuleScores(Config()).update(row(reliable=False, phone_conf=0.7, off_screen_score=0.9))
    assert s["degraded"] == 1.0 and s["phone"] == 0.7
    assert all(s[k] is None for k in ("face_absent", "off_screen", "glancing", "speaking", "id_mismatch"))
    s = RuleScores(Config()).update(row(frame_valid=False, reliable=False, phone_conf=NAN, n_faces=NAN))
    assert s["degraded"] == 1.0 and s["phone"] is None and s["notes"] is None and s["multiple_people"] is None


def test_speaking_window_rule_and_policy():
    r = RuleScores(Config())  # 10 Hz: 15 active steps = 1.5 s of the 3 s needed
    sc = [r.update(row(i * 100, mouth_energy_1s=0.2))["speaking"] for i in range(30)]
    assert abs(sc[14] - 0.5) < 1e-9 and sc[29] == 1.0
    sc = [r.update(row(3000 + i * 100, mouth_energy_1s=0.0))["speaking"] for i in range(110)]
    assert sc[-1] == 0.0  # window (10 s) has emptied
    r = RuleScores(Config())
    yawn = [r.update(row(i * 100, mouth_energy_1s=0.2, jaw_open=0.9))["speaking"] for i in range(40)]
    assert max(yawn) == 0.0
    r = RuleScores(cfg(allow_reading_aloud=True))
    assert max(r.update(row(i * 100, mouth_energy_1s=0.2))["speaking"] for i in range(40)) == 0.0
    r = RuleScores(Config())
    for i in range(40):
        r.update(row(i * 100, mouth_energy_1s=0.2))
    assert r.update(row(4000, reliable=False))["speaking"] is None


def test_id_mismatch_consecutive_rule():
    r = RuleScores(Config())  # tau 0.35, consecutive 3
    t = iter(range(0, 10**6, 100))

    def chk(sim=NAN, q=True):
        return r.update(row(next(t), id_similarity=sim, id_quality_ok=q))["id_mismatch"]

    assert [chk(0.1), chk(), chk(0.12), chk(), chk(0.11)] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert chk() == 1.0  # step-held between checks
    assert chk(0.9) == 0.0 and chk(0.1) == 0.0 and chk(0.12) == 0.0  # good check clears; count restarts
    # same check held over several grid steps counts once
    r = RuleScores(Config())
    assert [chk(0.1), chk(0.1), chk(0.1), chk()] == [0.0] * 4
    # consecutive grid steps with distinct values are distinct checks
    r = RuleScores(Config())
    assert [chk(0.10), chk(0.11), chk(0.12)] == [0.0, 0.0, 1.0]
    # low-quality checks neither count nor reset
    r = RuleScores(Config())
    assert [chk(0.1), chk(0.5, False), chk(0.2), chk(0.3, False), chk(0.15)] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert chk(NAN) == 1.0 and r.update(row(next(t), reliable=False))["id_mismatch"] is None


def test_glancing_from_rows_and_down_policy():
    def glances(c, zone):
        r, t, out = RuleScores(c), 0, []
        for _ in range(4):
            for off in [0.9] * 10 + [0.0] * 50:
                z = zone if off else "on_screen"
                out.append(r.update(row(t, off_screen_score=off, zone=z))["glancing"])
                t += 100
        return out

    assert 1.0 in glances(Config(), "right") and 1.0 in glances(Config(), "down")
    assert 1.0 not in glances(cfg(allow_looking_down=True), "down")  # allowed looking down is not a glance
