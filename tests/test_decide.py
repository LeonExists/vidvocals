"""Decision stage (pure function of cached features) and the feature cache."""
import numpy as np
import pytest
import soundfile as sf

from voicefilter import activity, decide, extract, features, pipeline


def _fake_extracted(seed=0):
    """Two speakers: a 'female' (p_head 0.9) and a 'male' (p_head 0.1) event, plus a short
    unsure event next to the female one."""
    rng = np.random.default_rng(seed)
    evs = [activity.Event(0.0, 3.0, vad_ratio=0.9), activity.Event(4.0, 7.0, vad_ratio=0.9),
           activity.Event(7.3, 7.6, vad_ratio=0.0)]
    wins = features.make_windows(evs, 1.5, 0.75)
    for i, w in enumerate(wins):
        w.idx = i
    a, b = rng.standard_normal(192), rng.standard_normal(192)
    E = np.stack([(a if w.event == 0 else b) + 0.1 * rng.standard_normal(192) for w in wins]).astype(np.float32)
    ph = np.array([0.9 if w.event == 0 else (0.1 if w.event == 1 else 0.55) for w in wins])
    pitch = [{"f0_med": 220.0 if w.event == 0 else 110.0, "voiced": 0.6, "periodicity": 0.9, "f0_iqr": 0.1,
              "f0_p10": 0.0, "f0_p90": 0.0} for w in wins]
    tag = np.array([[0.8, 0.1, 0.6, 0.05] if w.event == 0 else [0.1, 0.8, 0.6, 0.05] if w.event == 1
                    else [0.2, 0.2, 0.1, 0.5] for w in wins], np.float32)
    return extract.Extracted(duration=8.0, events=evs, windows=wins, db=np.full(800, -30.0, np.float32), E=E,
                             p_head=ph, pitch=pitch, tag=tag)


def test_decide_is_deterministic_and_keeps_female():
    ex = _fake_extracted()
    P = {"window": 1.5, "hop": 0.75, "overlap": "off", "cluster_threshold": 0.65}
    r1 = decide.decide(ex, P, {})
    r2 = decide.decide(ex, P, {})
    assert r1["segments"] == r2["segments"]
    segs = r1["segments"]
    assert segs[0]["keep"] and segs[0]["label"] == "female"
    m = [s for s in segs if s["start"] >= 4.0 and s["end"] <= 7.0]
    assert m and not any(s["keep"] for s in m)
    assert set(segs[0]) == {"id", "start", "end", "speaker", "kind", "label", "p_female", "confidence",
                            "low_confidence", "keep", "override"}


def test_decision_threshold_and_unsure_policy():
    ex = _fake_extracted()
    P = {"overlap": "off"}
    base = decide.decide(ex, P, {"low_conf_threshold": 0.99})["segments"]
    assert all(s["low_confidence"] for s in base)
    rem = decide.decide(ex, P, {"low_conf_threshold": 0.99, "unsure_policy": "remove"})["segments"]
    assert not any(s["keep"] for s in rem)
    keep = decide.decide(ex, P, {"low_conf_threshold": 0.99, "unsure_policy": "keep"})["segments"]
    assert all(s["keep"] for s in keep)
    hi = decide.decide(ex, P, {"decision_threshold": 0.9999})["segments"]
    assert not any(s["keep"] for s in hi)


def test_tagger_gender_cue_and_calibration():
    ex = _fake_extracted()
    ex.p_head = np.full(len(ex.windows), 0.5)  # head undecided: the tagger decides
    P = {"overlap": "off"}
    off = decide.decide(ex, P, {"tagger_gender": False})["segments"]
    on = decide.decide(ex, P, {"tagger_gender": True, "w_tagger_speech": 1.0, "tagger_rel_floor": 1.0})["segments"]
    assert on[0]["p_female"] > off[0]["p_female"] + 0.1
    cal = decide.decide(ex, P, {"tagger_gender": True, "final_b": -5.0})["segments"]
    assert cal[0]["p_female"] < on[0]["p_female"]


def test_resolve_overlaps_split_spans():
    env_f = np.concatenate([np.full(100, -20.0), np.full(100, -80.0)])  # female source silent in 2nd half
    env_m = np.full(200, -20.0)
    pieces = {0: [{"start": 0.0, "end": 2.0, "ps": [0.9, 0.1], "rms": [0.1, 0.1], "wavs": ["a.wav", "b.wav"],
                   "env": np.stack([env_f, env_m])}]}
    segs = [{"start": 0.0, "end": 2.0, "overlap": True}]
    r = decide.resolve_overlaps(pieces, segs, {"ovl_split_db": 15.0, "ovl_split_min_s": 0.3})
    assert r[0]["replacement"] and r[0]["remove_spans"]
    a, b = r[0]["remove_spans"][0]
    assert a == pytest.approx(1.0, abs=0.02) and b == pytest.approx(2.0, abs=0.02)
    r2 = decide.resolve_overlaps(pieces, segs, {})
    assert "remove_spans" not in r2[0]


def test_window_store_and_content_stamp(tmp_path):
    x = (np.random.default_rng(0).standard_normal((44100, 2)) * 0.1).astype(np.float32)
    sf.write(str(tmp_path / "vocals.wav"), x, 44100, subtype="FLOAT")
    d = extract.open_cache(tmp_path, tmp_path / "vocals.wav")
    st = extract.WindowStore(d, "emb_test")
    keys = [(0, 1500), (750, 2250)]
    calls = []

    def comp(idx):
        calls.append(list(idx))
        return np.arange(len(idx) * 3, dtype=np.float32).reshape(len(idx), 3)

    v1 = extract.cached_window_values(st, keys, comp)
    v2 = extract.cached_window_values(extract.WindowStore(d, "emb_test"), keys + [(0, 1500)], comp)
    assert len(calls) == 1 and np.allclose(v1, v2[:2]) and np.allclose(v2[2], v1[0])
    # identical content rewritten -> cache kept; different content -> cache wiped
    sf.write(str(tmp_path / "vocals.wav"), x, 44100, subtype="FLOAT")
    extract.open_cache(tmp_path, tmp_path / "vocals.wav")
    assert (d / "win_emb_test.npz").exists()
    sf.write(str(tmp_path / "vocals.wav"), x * 0.5, 44100, subtype="FLOAT")
    extract.open_cache(tmp_path, tmp_path / "vocals.wav")
    assert not (d / "win_emb_test.npz").exists()


def test_decision_params_precedence(monkeypatch):
    from voicefilter import config

    P = {"decision": {"decision_threshold": 0.6}}
    assert pipeline.decision_params(P)["decision_threshold"] == 0.6
    monkeypatch.setitem(config.SETTINGS, "decision_threshold", 0.7)
    monkeypatch.setattr(config, "USER_SET", config.USER_SET | {"decision_threshold"})
    assert pipeline.decision_params(P)["decision_threshold"] == 0.7
