"""`python -m voicefilter.download --presets fast,balanced,best`

Downloads every model the presets need into the models folder (MODELS_DIR).  Each model
has an ordered list of sources (Hugging Face, then GitHub / torchaudio fallbacks).
Exit code 0 when every required model is present afterwards.
"""
from __future__ import annotations

import argparse
import sys
import time

from . import config, modelstore, presets


def model_present(mid: str) -> bool:
    spec = presets.model_spec(mid)
    t = spec.get("type")
    try:
        if spec.get("kind") == "separation" or t in ("ensemble", "mel_band_roformer", "bs_roformer",
                                                     "demucs", "torchaudio_hdemucs", "passthrough"):
            from . import separation

            return separation.spec_present(spec)
        if t == "ecapa_tao":
            from .features import EcapaTaoEmbedder

            return EcapaTaoEmbedder.present()
        if t == "torchaudio":
            return modelstore.torchaudio_present(spec["bundle"])
        if t == "hf_audio_classification":
            from .features import HFGenderModel

            return HFGenderModel(spec).present()
        if t in ("tagger",) or spec.get("kind") == "tagger":
            from . import tagger

            return tagger.make_tagger(spec, "cpu").present()
    except Exception:
        return False
    return False


def model_download(mid: str) -> None:
    spec = presets.model_spec(mid)
    t = spec.get("type")
    if spec.get("kind") == "separation" or t in ("ensemble", "mel_band_roformer", "bs_roformer",
                                                 "demucs", "torchaudio_hdemucs", "passthrough"):
        from . import separation

        separation.spec_download(spec)
    elif t == "ecapa_tao":
        from .features import EcapaTaoEmbedder

        EcapaTaoEmbedder.download()
    elif t == "torchaudio":
        modelstore.torchaudio_bundle(spec["bundle"]).get_model()
    elif t == "hf_audio_classification":
        from .features import HFGenderModel

        HFGenderModel(spec).download()
    elif spec.get("kind") == "tagger":
        from . import tagger

        tagger.make_tagger(spec, "cpu").download()
    else:
        raise ValueError(f"don't know how to download model type {t!r}")


def preset_models(pid: str) -> tuple[list[str], list[str]]:
    """(required, optional) model ids of a preset."""
    p = presets.get_preset(pid)
    req = [p["separation"], p.get("embedding", "ecapa_tao")]
    if p.get("overlap") == "separate" and p.get("overlap_model"):
        req.append(p["overlap_model"])
    opt = []
    if p.get("tagger") and "tagger" in p.get("gender_models", []):
        # GitHub-hosted taggers are required, HF-only ones optional
        (req if presets.model_spec(p["tagger"]).get("type") == "efficientat" else opt).append(p["tagger"])
    if p.get("hf_gender") and "hf" in p.get("gender_models", []):
        opt.append(p["hf_gender"])
    return list(dict.fromkeys(req)), [m for m in dict.fromkeys(opt) if m not in req]


def preset_present(pid: str) -> bool:
    req, _ = preset_models(pid)
    return all(model_present(m) for m in req)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicefilter.download", description=__doc__.split("\n\n")[0])
    ap.add_argument("--presets", default="fast,balanced,best", help="comma separated preset ids")
    ap.add_argument("--skip-optional", action="store_true", help="only the required models")
    a = ap.parse_args(argv)
    config.ensure_dirs()
    pids = [p.strip() for p in a.presets.split(",") if p.strip()]
    for p in pids:
        if p not in presets.presets():
            print(f"unknown preset {p!r}; available: {', '.join(presets.presets())}", file=sys.stderr)
            return 2
    req: list[str] = []
    opt: list[str] = []
    for p in pids:
        r, o = preset_models(p)
        req += [m for m in r if m not in req]
        opt += [m for m in o if m not in opt and m not in req]
    print(f"Models folder: {config.MODELS_DIR}", flush=True)
    failed_req, failed_opt = [], []
    todo = [(m, True) for m in req] + ([] if a.skip_optional else [(m, False) for m in opt])
    for i, (m, required) in enumerate(todo, 1):
        spec = presets.model_spec(m)
        size = spec.get("size_mb")
        tag = "" if required else " (optional)"
        if model_present(m):
            print(f"[{i}/{len(todo)}] {m}{tag}: already present", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {m}{tag}: downloading" + (f" (~{size} MB)" if size else ""), flush=True)
        t0 = time.time()
        try:
            model_download(m)
            ok = model_present(m)
        except Exception as exc:  # noqa: BLE001
            print(f"    failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            ok = False
        if ok:
            print(f"    done in {time.time() - t0:.0f} s", flush=True)
        else:
            (failed_req if required else failed_opt).append(m)
    if failed_opt:
        print("Optional models not available (the presets still work without them): "
              + ", ".join(failed_opt), flush=True)
    if failed_req:
        print("ERROR: required models missing: " + ", ".join(failed_req), file=sys.stderr, flush=True)
        return 1
    print("All required models are present.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
