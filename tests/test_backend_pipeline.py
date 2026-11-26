"""Pipeline smoke test with passthrough separation (no big models; needs the 66 MB ECAPA)."""
import copy
import subprocess

import numpy as np
import pytest
import soundfile as sf

from voicefilter import config, media, pipeline, presets
from voicefilter.features import EcapaTaoEmbedder

SEG_KEYS = {"id", "start", "end", "speaker", "kind", "label", "p_female", "confidence", "low_confidence",
            "keep", "override"}
SPK_KEYS = {"id", "p_female", "label", "total_dur", "n_segments", "override"}


@pytest.fixture
def test_preset(monkeypatch):
    models, ps, notes = copy.deepcopy(presets._get())
    ps["smoke"] = dict(ps["balanced"], separation="passthrough", hf_gender=None, tagger=None)
    monkeypatch.setattr(presets, "_CACHE", (models, ps, notes))
    yield "smoke"


@pytest.mark.skipif(not EcapaTaoEmbedder.present(), reason="ECAPA weights not downloaded")
def test_analyze_render_export(tmp_path, test_preset):
    sr = 44100
    t = np.arange(int(sr * 6)) / sr
    x = np.zeros_like(t)
    for a, b, f in ((0.5, 2.0, 120), (2.6, 4.0, 230), (4.5, 5.5, 125)):  # low / high / low "voices"
        m = (t >= a) & (t < b)
        ph = 2 * np.pi * f * t[m] * (1 + 0.02 * np.sin(2 * np.pi * 4 * t[m]))
        x[m] = 0.2 * (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph))
    sf.write(str(tmp_path / "a.wav"), np.stack([x, x], 1).astype(np.float32), sr)
    src = tmp_path / "in.mp4"
    subprocess.run([config.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=6",
                    "-i", str(tmp_path / "a.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(src)], check=True)
    stages = []
    an = pipeline.analyze_file(src, test_preset, tmp_path / "w", progress=lambda s, f, m: stages.append((s, f)))
    assert set(an) >= {"duration", "segments", "speakers", "low_conf_threshold"}
    assert an["segments"] and all(set(s) == SEG_KEYS for s in an["segments"])
    assert all(set(s) == SPK_KEYS for s in an["speakers"])
    assert all(isinstance(s["id"], int) for s in an["segments"])
    fr = [f for _, f in stages]
    assert fr == sorted(fr) and fr[-1] == 1.0
    for f in ("original.m4a", "vocals.m4a", "preview.m4a", "peaks.json", "analysis.json"):
        assert (tmp_path / "w" / f).exists()
    wav = pipeline.render_from_analysis(tmp_path / "w", an, keep_background=True)
    assert sf.info(wav).frames == sf.info(str(tmp_path / "w" / "audio.wav")).frames
    out = pipeline.export_video(src, wav, tmp_path / "out.mp4")
    assert media.probe(out)["has_video"]
