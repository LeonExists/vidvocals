"""download / selfcheck / cli entry points."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV = dict(os.environ, VF_APP_DIR=str(ROOT))


def run(*args, timeout=600):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=ENV, capture_output=True, text=True,
                          timeout=timeout)


def test_selfcheck_json():
    r = run("voicefilter.selfcheck", timeout=180)
    d = json.loads(r.stdout)
    assert set(d) >= {"python", "torch", "cuda_available", "gpu_name", "vram_gb", "ffmpeg", "ffprobe",
                      "ffmpeg_version", "models_present", "ok"}
    assert set(d["models_present"]) >= {"fast", "balanced", "best"}
    assert r.returncode == (0 if d["ok"] else 1)


def test_download_unknown_preset():
    r = run("voicefilter.download", "--presets", "nope", timeout=120)
    assert r.returncode == 2 and "unknown preset" in r.stderr


def test_download_presets():
    from voicefilter import download

    if not all(download.preset_present(p) for p in ("fast", "balanced", "best")) and not os.environ.get("VF_RUN_SLOW"):
        pytest.skip("models not downloaded (set VF_RUN_SLOW=1 to download ~2 GB)")
    r = run("voicefilter.download", "--presets", "fast,balanced,best", timeout=3600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "All required models are present" in r.stdout


def test_cli_help():
    r = run("voicefilter.cli", "process", "--help", timeout=60)
    assert r.returncode == 0 and "--analysis-out" in r.stdout and "--no-background" in r.stdout
