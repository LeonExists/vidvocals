"""render.py: click-free fades, exact length, background preserved, limiter."""
import numpy as np
import pytest
import soundfile as sf

from voicefilter import render

SR = 44100


def _write(path, x):
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


@pytest.fixture
def stems(tmp_path):
    n = SR * 3 + 123  # odd length on purpose
    t = np.arange(n) / SR
    voc = 0.5 * np.sin(2 * np.pi * 220 * t)
    bg = 0.2 * np.sin(2 * np.pi * 55 * t + 0.3)
    v = np.stack([voc, voc * 0.9], 1)
    b = np.stack([bg, bg], 1)
    _write(tmp_path / "v.wav", v)
    _write(tmp_path / "b.wav", b)
    return tmp_path, v.astype(np.float32), b.astype(np.float32)


def test_envelope_values_and_smoothness():
    env = render.GainEnvelope([(0.5, 1.0, 0.0), (1.0, 1.004, 1.0), (1.004, 2.0, 0.0)], SR, 0.015, 1.0)
    g = np.concatenate([env.block(s, min(1000, SR * 3 - s)) for s in range(0, SR * 3, 1000)])
    assert g[int(0.2 * SR)] == pytest.approx(1.0)
    assert g[int(0.75 * SR)] == pytest.approx(0.0, abs=1e-7)
    assert g[int(2.5 * SR)] == pytest.approx(1.0)
    assert g.min() >= -1e-6 and g.max() <= 1 + 1e-6
    # a 15 ms raised cosine has max slope pi/2 / (0.015*SR) per sample
    assert np.abs(np.diff(g)).max() <= np.pi / 2 / (0.015 * SR) * 1.01
    # block boundaries do not matter
    g2 = env.block(0, SR * 3)
    assert np.allclose(g, g2[: len(g)], atol=1e-6)


def test_render_click_free_exact_length(stems):
    d, v, b = stems
    out = render.render(d / "v.wav", d / "b.wav", d / "o.wav",
                        [(0.5, 1.2, 0.0), (1.2, 1.6, 1.0), (1.6, 1.605, 0.0), (2.0, 2.5, 0.0)],
                        fade_ms=15, keep_background=False)
    y, sr = sf.read(out, dtype="float32", always_2d=True)
    assert sr == SR and y.shape == v.shape
    voc_step = np.abs(np.diff(v, axis=0)).max()
    # gating may only add the envelope slope times the signal amplitude
    allowed = voc_step + 0.5 * np.pi / 2 / (0.015 * SR) + 1e-4
    assert np.abs(np.diff(y, axis=0)).max() <= allowed
    # removed region is silent, kept region untouched
    assert np.abs(y[int(0.7 * SR):int(1.0 * SR)]).max() < 1e-6
    assert np.allclose(y[int(0.1 * SR):int(0.4 * SR)], v[int(0.1 * SR):int(0.4 * SR)], atol=1e-6)


def test_background_preserved(stems):
    d, v, b = stems
    out = render.render(d / "v.wav", d / "b.wav", d / "o.wav", [(0.0, 10.0, 0.0)], keep_background=True)
    y, _ = sf.read(out, dtype="float32", always_2d=True)
    assert y.shape == b.shape
    mid = slice(int(0.1 * SR), int(2.9 * SR))
    assert np.allclose(y[mid], b[mid], atol=1e-6)
    # keep everything: output == vocals + background (the mix)
    out = render.render(d / "v.wav", d / "b.wav", d / "o2.wav", [], keep_background=True)
    y, _ = sf.read(out, dtype="float32", always_2d=True)
    assert np.allclose(y, v + b, atol=1e-6)


def test_gate_unlabeled_mute(stems):
    d, v, b = stems
    out = render.render(d / "v.wav", None, d / "o.wav", [(1.0, 2.0, 1.0)], gate_unlabeled="mute",
                        keep_background=False)
    y, _ = sf.read(out, dtype="float32", always_2d=True)
    assert np.abs(y[: int(0.9 * SR)]).max() < 1e-6
    assert np.allclose(y[int(1.1 * SR):int(1.9 * SR)], v[int(1.1 * SR):int(1.9 * SR)], atol=1e-6)


def test_limiter_no_clipping(tmp_path):
    n = SR * 2
    t = np.arange(n) / SR
    v = np.stack([0.9 * np.sin(2 * np.pi * 300 * t)] * 2, 1)
    b = np.stack([0.6 * np.sin(2 * np.pi * 300 * t)] * 2, 1)  # in phase: sum peaks at 1.5
    _write(tmp_path / "v.wav", v)
    _write(tmp_path / "b.wav", b)
    out = render.render(tmp_path / "v.wav", tmp_path / "b.wav", tmp_path / "o.wav", [])
    y, _ = sf.read(out, dtype="float32", always_2d=True)
    assert len(y) == n
    assert np.abs(y).max() <= 1.0
    # gentle: smooth gain, no hard clipping (waveform keeps its sine shape)
    ratio = y[:, 0] / (v[:, 0] + b[:, 0] + 1e-9)
    sel = np.abs(v[:, 0] + b[:, 0]) > 0.3
    assert ratio[sel].std() < 0.05


def test_effective_keep_priority():
    seg = {"id": 1, "speaker": "S1", "keep": True, "override": None, "low_confidence": False}
    assert render.effective_keep(seg, {"S1": "remove"}) is False
    seg["override"] = "keep"
    assert render.effective_keep(seg, {"S1": "remove"}) is True
    seg["override"] = None
    assert render.effective_keep(seg, {}) is True


def test_replacement_crossfade(stems):
    d, v, b = stems
    rep = np.zeros((int(1.0 * SR), 2), np.float32) + 0.1
    _write(d / "r.wav", rep)
    out = render.render(d / "v.wav", None, d / "o.wav", [], keep_background=False,
                        replacements=[{"start": 1.0, "end": 2.0, "wav": str(d / "r.wav")}])
    y, _ = sf.read(out, dtype="float32", always_2d=True)
    assert np.allclose(y[int(1.3 * SR):int(1.7 * SR)], 0.1, atol=1e-6)
    assert np.allclose(y[: int(0.9 * SR)], v[: int(0.9 * SR)], atol=1e-6)
    assert np.abs(np.diff(y, axis=0)).max() < 0.05
