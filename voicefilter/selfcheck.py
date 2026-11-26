"""`python -m voicefilter.selfcheck` -> JSON environment report, exit 0 if ok."""
from __future__ import annotations

import json
import platform
import sys

from . import config


def run() -> dict:
    out: dict = {"python": platform.python_version(), "torch": None, "cuda_available": False,
                 "gpu_name": None, "vram_gb": None, "ffmpeg": config.FFMPEG, "ffprobe": config.FFPROBE,
                 "ffmpeg_version": None, "models_present": {}, "ok": False}
    problems = []
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            out["gpu_name"] = p.name
            out["vram_gb"] = round(p.total_memory / 1024**3, 2)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch: {exc}")
    from . import media

    out["ffmpeg_version"] = media.tool_version(config.FFMPEG)
    ffprobe_v = media.tool_version(config.FFPROBE)
    if not out["ffmpeg_version"]:
        problems.append("ffmpeg not working")
    if not ffprobe_v:
        problems.append("ffprobe not working")
    for mod in ("numpy", "scipy", "soundfile", "soxr", "fastapi", "uvicorn", "sklearn", "librosa"):
        try:
            __import__(mod)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{mod}: {exc}")
    try:
        from . import download, presets

        for pid in presets.presets():
            out["models_present"][pid] = download.preset_present(pid)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"models: {exc}")
    out["device"] = config.get_device() if out["torch"] else "cpu"
    out["app_dir"] = str(config.APP_DIR)
    out["models_dir"] = str(config.MODELS_DIR)
    out["problems"] = problems
    if out["models_present"] and not any(out["models_present"].values()):
        problems.append("no preset has all its models; run python -m voicefilter.download")
    out["ok"] = not problems and out["torch"] is not None
    return out


def main() -> int:
    r = run()
    print(json.dumps(r, indent=2))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
