"""separation streaming, roformer shapes, activity, clustering, fusion, presets, config."""
import os

import numpy as np
import pytest
import soundfile as sf

from voicefilter import activity, config, features, gender, presets, separation

SR = 44100


class Doubler(separation.Backend):
    """Scales by 0.5 -> exercises the overlap-add normalisation."""
    name = "half"
    chunk = 10000
    num_overlap = 4
    batch_size = 3

    def _infer(self, x):
        return 0.5 * x


@pytest.mark.parametrize("n", [1, 999, 10000, 123457])
def test_streaming_ola_exact(tmp_path, n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal((n, 2)).astype(np.float32) * 0.1
    sf.write(str(tmp_path / "m.wav"), x, SR, subtype="FLOAT")
    separation.separate_file(tmp_path / "m.wav", tmp_path / "v.wav", tmp_path / "b.wav", Doubler({}))
    v, _ = sf.read(str(tmp_path / "v.wav"), dtype="float32", always_2d=True)
    b, _ = sf.read(str(tmp_path / "b.wav"), dtype="float32", always_2d=True)
    assert v.shape == x.shape == b.shape
    assert np.allclose(v, 0.5 * x, atol=1e-5)
    assert np.abs(v + b - x).max() < 1e-6  # background = mix - vocals


def test_passthrough_inmemory():
    x = np.random.default_rng(0).standard_normal(SR // 2).astype(np.float32) * 0.1
    v, b = separation.separate(x, SR)
    assert v.shape == (len(x), 2) and np.allclose(v[:, 0], x, atol=1e-6) and np.abs(b).max() < 1e-6


@pytest.mark.parametrize("kind", ["mel_band_roformer", "bs_roformer"])
def test_roformer_small_forward(kind):
    import torch

    from voicefilter import roformer

    cfg = dict(dim=32, depth=1, stereo=True, num_stems=1, time_transformer_depth=1, freq_transformer_depth=1,
               dim_head=16, heads=2, stft_n_fft=2048, stft_hop_length=441, stft_win_length=2048,
               mask_estimator_depth=1)
    if kind == "mel_band_roformer":
        cfg.update(num_bands=16, sample_rate=44100)
    m = roformer.build_from_config(kind, cfg).eval()
    with torch.inference_mode():
        y = m(torch.randn(2, 2, 44100))
    assert y.shape == (2, 2, 44100) and torch.isfinite(y).all()


def _tone(f, dur, amp, sr=16000):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_activity_finds_quiet_events():
    sr = 16000
    x = np.random.default_rng(0).standard_normal(sr * 8).astype(np.float32) * 1e-4  # residual noise
    x[sr * 1:sr * 2] += _tone(220, 1.0, 0.3)
    x[int(sr * 3.0):int(sr * 3.5)] += _tone(400, 0.5, 0.01)  # quiet breathy sound (-40 dB)
    x[int(sr * 3.6):int(sr * 4.0)] += _tone(400, 0.4, 0.01)  # 100 ms gap -> merged
    x[int(sr * 6):int(sr * 6.03)] += _tone(300, 0.03, 0.0006)  # 30 ms weak click -> dropped
    ev, _ = activity.detect_events(x, sr, activity.ActivityParams(use_vad=False))
    spans = [(round(e.start, 2), round(e.end, 2)) for e in ev]
    assert len(ev) == 2, spans
    assert ev[0].start == pytest.approx(1.0, abs=0.08) and ev[0].end == pytest.approx(2.0, abs=0.08)
    assert ev[1].start == pytest.approx(3.0, abs=0.08) and ev[1].end == pytest.approx(4.0, abs=0.08)
    assert ev[1].quiet is False or ev[1].snr_db > 0


def test_yin_pitch():
    f0, per = features.yin_track(_tone(180, 1.0, 0.3))
    v = f0[f0 > 0]
    assert len(v) > 50 and np.median(v) == pytest.approx(180, rel=0.02)
    st = features.pitch_stats(f0, per, 0.01, 0, 1)
    p, r = features.pitch_p_female(st)
    assert 0.5 < p <= 0.9 and r > 0.5
    f0, _ = features.yin_track(_tone(110, 1.0, 0.3))
    assert np.median(f0[f0 > 0]) == pytest.approx(110, rel=0.02)


def test_windows():
    class E:
        def __init__(self, s, e):
            self.start, self.end = s, e
    w = features.make_windows([E(0, 0.4), E(1.0, 4.2)], 1.5, 0.5)
    assert w[0].start == 0 and w[0].end == 0.4
    rest = [x for x in w if x.event == 1]
    assert rest[0].start == 1.0 and rest[-1].end == pytest.approx(4.2)
    assert all(x.dur <= 1.5 + 1e-9 for x in rest)


def test_clustering_two_speakers_and_single():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(192), rng.standard_normal(192)
    E = np.concatenate([a + 0.3 * rng.standard_normal((20, 192)), b + 0.3 * rng.standard_normal((15, 192))])
    lab = gender.cluster_embeddings(E, np.full(35, 1.5), 0.65)
    assert len(set(lab[:20])) == 1 and len(set(lab[20:])) == 1 and lab[0] != lab[25]
    lab1 = gender.cluster_embeddings(E[:20], np.full(20, 1.5), 0.65)
    assert len(set(lab1)) == 1


def test_mixed_cluster_split():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(192)
    E = a + 0.2 * rng.standard_normal((20, 192))
    l = np.array([3.0] * 10 + [-3.0] * 10)
    lab = gender.cluster_embeddings(E, np.full(20, 1.5), 0.9, window_logits=l, window_conf=np.full(20, 0.9))
    assert len(set(lab)) == 2 and len(set(lab[:10])) == 1


def test_fusion_and_speaker_consistency():
    l, w = gender.fuse({"head": (0.9, 1.0), "pitch": (0.4, 1.0)}, "speech")
    assert 0.5 < gender.sigmoid(l) < 0.9
    l2, _ = gender.fuse({"pitch": (0.8, 1.0)}, "speech", head_available=False)
    assert gender.sigmoid(l2) == pytest.approx(0.8)
    spk = gender.speaker_posteriors(np.zeros(5, int), np.full(5, 2.0), np.full(5, 2.5), np.full(5, 0.85))[0]
    assert spk.p_female > 0.9 and spk.purity == 1.0
    # a short, uncertain segment leans on its speaker; a long confident one may disagree
    assert gender.combine_with_speaker(-0.2, 0.1, 0.3, spk) > 0.5
    assert gender.combine_with_speaker(-4.0, 0.96, 6.0, spk) < 0


def test_presets_data_driven(tmp_path, monkeypatch):
    for pid in ("fast", "balanced", "best"):
        p = presets.get_preset(pid)
        presets.model_spec(p["separation"])
    lst = presets.preset_list()
    assert [p["id"] for p in lst][:3] == ["fast", "balanced", "best"]
    # override via APP/presets.json
    (tmp_path / "presets.json").write_text('{"presets": {"fast": {"separation": "passthrough", "hop": 1.0}}}')
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    presets.reload()
    try:
        assert presets.get_preset("fast")["separation"] == "passthrough"
        assert presets.get_preset("fast")["window"] == 1.5
    finally:
        monkeypatch.undo()
        presets.reload()


def test_config_paths_inside_app():
    for p in (config.DATA_DIR, config.MODELS_DIR, config.TMP_DIR):
        assert str(p).startswith(str(config.APP_DIR)) or os.environ.get("VF_DATA_DIR") or os.environ.get("VF_MODELS_DIR")
    import tempfile

    assert tempfile.gettempdir() == str(config.TMP_DIR)
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_research_models_json_translation():
    """docs/models.json (research schema) drives the presets when present."""
    if not (config.DOCS_DIR / "models.json").exists():
        pytest.skip("no docs/models.json")
    presets.reload()
    ov = {p: presets.get_preset(p).get("separation_params", {}).get("num_overlap") for p in ("fast", "balanced", "best")}
    assert ov == {"fast": 1, "balanced": 2, "best": 4}
    spec = presets.model_spec(presets.get_preset("fast")["separation"])
    assert spec["type"] in ("bs_roformer", "mel_band_roformer", "ensemble")
    if spec["type"] != "ensemble":
        assert spec["ckpt"] and spec["config"]
        assert all("hf" in s or "url" in s or "path" in s for s in spec["ckpt"] + spec["config"])


def test_embedder_kinds():
    assert features.embedder_for_kind("ecapa_tao_vox2_192") == "ecapa_tao"
    assert features.embedder_for_kind("wavlm_base_plus_L4-5-6_mean") == "wavlm:4-5-6"
    assert features.embedder_for_kind("none") is None
    with pytest.raises(ValueError):
        features.embedder_for_kind("something_else")
    assert features.make_embedder("wavlm:3-6").layers == (3, 6)


def test_num_overlap_one_has_crossfade(tmp_path):
    x = np.random.default_rng(3).standard_normal((50000, 2)).astype(np.float32) * 0.1
    sf.write(str(tmp_path / "m.wav"), x, SR, subtype="FLOAT")

    class One(Doubler):
        num_overlap = 1

    separation.separate_file(tmp_path / "m.wav", tmp_path / "v.wav", tmp_path / "b.wav", One({}))
    v, _ = sf.read(str(tmp_path / "v.wav"), dtype="float32", always_2d=True)
    assert np.allclose(v, 0.5 * x, atol=1e-5)


def test_overlap_mask_estimates_split_fullband():
    from voicefilter import overlap

    sr = 44100
    t = np.arange(sr * 2) / sr
    lo = 0.3 * np.sin(2 * np.pi * 200 * t)
    hi = 0.3 * np.sin(2 * np.pi * 1500 * t)
    v44 = np.stack([lo + hi] * 2, 1).astype(np.float32)
    import soxr

    s_lo, s_hi = (soxr.resample(x.astype(np.float32), sr, 16000) for x in (lo, hi))
    ea, eb = overlap.mask_estimates(v44, s_hi, s_lo, sr)
    mid = slice(sr // 4, -sr // 4)
    assert ea.shape == v44.shape
    assert np.corrcoef(ea[mid, 0], hi[mid])[0, 1] > 0.99
    assert np.corrcoef(eb[mid, 0], lo[mid])[0, 1] > 0.99
    g = overlap.ls_gains(lo + hi, np.stack([lo * 2, hi * 0.5]))
    assert np.allclose(g, [0.5, 2.0], atol=1e-6)
