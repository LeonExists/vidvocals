"""Tests for the evaluation metric code (eval/metrics.py, eval/run_eval.py stubs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EVAL = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL))

import metrics as M  # noqa: E402


def _truth():
    # 10 s: female 1-3 s, male 4-6 s, female 7-9 s overlapping male 8-9.5 s
    return {"duration": 10.0, "sr": 44100, "events": [
        {"start": 1.0, "end": 3.0, "speaker": "S1", "gender": "F", "kind": "speech", "category": "read",
         "overlap": False, "active": [[1.0, 3.0]]},
        {"start": 4.0, "end": 6.0, "speaker": "S2", "gender": "M", "kind": "nonverbal", "category": "moaning",
         "overlap": False, "active": [[4.0, 6.0]]},
        {"start": 7.0, "end": 9.0, "speaker": "S1", "gender": "F", "kind": "nonverbal", "category": "sighing",
         "overlap": True, "active": [[7.0, 9.0]], "breathy": True},
        {"start": 8.0, "end": 9.5, "speaker": "S2", "gender": "M", "kind": "speech", "category": "read",
         "overlap": True, "active": [[8.0, 9.5]]},
    ]}


def _seg(a, b, keep, **kw):
    d = {"start": a, "end": b, "keep": keep, "speaker": kw.pop("speaker", None), "low_confidence": False,
         "override": None}
    d.update(kw)
    return d


def test_truth_frames_counts():
    f, m = M.truth_frames(_truth())
    assert len(f) == 1000
    assert f.sum() == 400 and m.sum() == 350
    assert (f & m).sum() == 100


def test_perfect_decisions():
    t = _truth()
    f, m = M.truth_frames(t)
    an = {"segments": [_seg(1, 3, True), _seg(4, 6, False), _seg(8, 9.5, False), _seg(7, 9, True)]}
    keep, low, cov = M.decision_frames(an, len(f))
    r = M.frame_metrics(f, m, keep, low, cov)
    assert r["accuracy"] == 1.0
    assert r["male_leak"] == 0.0 and r["female_loss"] == 0.0
    assert r["overlap_kept"] == 1.0  # keep wins where segments overlap
    assert r["coverage"] == 1.0
    ev = M.event_results(t, keep)
    # the male speech event is kept during 1 s of its 1.5 s (overlap) -> only 33 % right -> incorrect
    assert [e["correct"] for e in ev] == [True, True, True, False]


def test_keep_everything_and_nothing():
    f, m = M.truth_frames(_truth())
    allk = M.frame_metrics(f, m, np.ones(len(f), bool))
    assert allk["male_leak"] == 1.0 and allk["female_loss"] == 0.0
    assert allk["accuracy"] == pytest.approx(300 / (300 + 250))
    assert allk["balanced_accuracy"] == pytest.approx(0.5)
    none = M.frame_metrics(f, m, np.zeros(len(f), bool))
    assert none["male_leak"] == 0.0 and none["female_loss"] == 1.0


def test_outside_default_remove_and_coverage():
    f, m = M.truth_frames(_truth())
    keep, low, cov = M.decision_frames({"segments": [_seg(1, 2, True)]}, len(f))
    r = M.frame_metrics(f, m, keep, low, cov)
    assert r["female_loss"] == pytest.approx(1 - 100 / 300)
    assert r["coverage"] == pytest.approx(100 / 650)
    keep2, _, _ = M.decision_frames({"segments": []}, len(f), outside="keep")
    assert keep2.all()


def test_overrides_priority():
    segs = [_seg(0, 1, True, speaker="A", override="remove"), _seg(1, 2, False, speaker="A"),
            _seg(2, 3, False, speaker="B")]
    an = {"segments": segs, "speakers": [{"id": "A", "override": "keep"}, {"id": "B", "override": None}]}
    keep, _, _ = M.decision_frames(an, 300)
    assert not keep[:100].any()          # segment override beats speaker override
    assert keep[100:200].all()           # speaker override beats model decision
    assert not keep[200:300].any()       # model decision


def test_low_confidence_metrics():
    f, m = M.truth_frames(_truth())
    segs = [_seg(1, 3, True), _seg(4, 6, True, low_confidence=True)]  # wrong but flagged
    keep, low, cov = M.decision_frames({"segments": segs}, len(f))
    r = M.frame_metrics(f, m, keep, low, cov)
    assert r["low_conf_share"] == pytest.approx(200 / 650)
    assert r["accuracy_low_conf_only"] == 0.0
    # excluding the flagged male frames: female 1-3 kept (right), female-only 7-8 removed (wrong),
    # male-only 9-9.5 removed (right)
    assert r["accuracy_excl_low_conf"] == pytest.approx(250 / 350)


def test_event_summary_groups():
    f, m = M.truth_frames(_truth())
    keep = f.copy()
    s = M.summarize_events(M.event_results(_truth(), keep))
    assert s["all"]["n"] == 4
    assert s["kind=nonverbal"]["n"] == 2
    assert s["breathy"] == {"n": 1, "accuracy": 1.0}


def test_si_sdr_and_projection():
    rng = np.random.default_rng(0)
    sr = 1000
    n = 10 * sr
    F = rng.standard_normal((2, n))
    Mm = rng.standard_normal((2, n))
    B = 0.1 * rng.standard_normal((2, n))
    f_act = np.zeros(1000, bool)
    m_act = np.zeros(1000, bool)
    f_act[:500] = True
    m_act[500:] = True
    F[:, n // 2:] = 0
    Mm[:, : n // 2] = 0
    perfect = F + B
    r = M.audio_metrics(perfect, F, Mm, B, f_act, m_act, sr, mix=F + Mm + B)
    assert r["female_preserved_db"] == pytest.approx(0.0, abs=0.01)
    assert r["male_residual_db"] < -40
    assert r["si_sdr_vs_female_plus_bg"] > 60
    assert r["si_sdr_improvement"] > 50
    leaky = F + B + 0.1 * Mm  # -20 dB male residual
    r2 = M.audio_metrics(leaky, F, Mm, B, f_act, m_act, sr)
    assert r2["male_residual_db"] == pytest.approx(-20.0, abs=0.5)
    assert M.si_sdr(F, F) > 100


def test_gain_envelope_smooth_and_exact_length():
    keep = np.array([0, 0, 1, 1, 1, 0], bool)
    g = M.gain_envelope(keep, 6 * 441, 44100)
    assert len(g) == 6 * 441
    assert g.min() >= 0 and g.max() <= 1
    assert np.abs(np.diff(g)).max() < 0.01  # raised-cosine 15 ms ramps, no hard steps
    assert g[3 * 441] == pytest.approx(1.0)


def test_aggregate_micro_average():
    t = _truth()
    f, m = M.truth_frames(t)
    rows = []
    for keep in (np.ones(len(f), bool), np.zeros(len(f), bool)):
        rows.append({"frames": M.frame_metrics(f, m, keep), "event_list": M.event_results(t, keep),
                     "duration": 10.0, "processing_s": 5.0})
    a = M.aggregate(rows)
    assert a["frames"]["male_leak"] == pytest.approx(0.5)
    assert a["frames"]["female_loss"] == pytest.approx(0.5)
    assert a["rtf"] == pytest.approx(0.5)
    assert a["events"]["all"]["n"] == 8


def test_run_eval_oracle_and_random_on_tiny_mix(tmp_path):
    sf = pytest.importorskip("soundfile")
    import run_eval as R
    sr = 8000
    t = _truth()
    t["sr"] = sr
    n = 10 * sr
    rng = np.random.default_rng(1)
    f, m = M.truth_frames(t)
    F = np.repeat(np.repeat(f.astype(float), sr // 100)[None], 2, 0) * rng.standard_normal((2, n)) * 0.1
    Mm = np.repeat(np.repeat(m.astype(float), sr // 100)[None], 2, 0) * rng.standard_normal((2, n)) * 0.1
    B = 0.01 * rng.standard_normal((2, n))
    md = tmp_path / "mix_00"
    md.mkdir()
    for name, x in [("mix", F + Mm + B), ("vocals_female", F), ("vocals_male", Mm), ("background", B)]:
        sf.write(str(md / f"{name}.wav"), x.T.astype(np.float32), sr, subtype="FLOAT")
    (md / "truth.json").write_text(json.dumps(t))
    an = R.oracle_analysis(t)
    out = R.stub_render(md, an)
    r = R.score_mix(md, an, out, sr, 1.0)
    assert r["frames"]["accuracy"] == 1.0
    assert r["audio"]["female_preserved_db"] == pytest.approx(0.0, abs=0.2)
    assert r["audio"]["male_residual_db"] < -30
    assert r["audio"]["length_diff_samples"] == 0
    rr = R.random_analysis(t)
    assert all(0 <= s["p_female"] <= 1 for s in rr["segments"])
