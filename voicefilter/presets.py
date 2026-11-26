"""Presets -> model choices and parameters (data-driven).

``MODELS`` lists every model with its type and an ordered list of download sources
(Hugging Face first where it exists, then GitHub release / torchaudio fallbacks).
``PRESETS`` reference models by id.  Both can be overridden or extended without code
changes by ``docs/models.json`` (written by the model research) and by ``APP/presets.json``
(user tweaks)::

    {"models": {"<id>": {...partial or full spec...}},
     "presets": {"balanced": {"separation": "<model id>", ...}}}

Entries are merged key by key (dicts recursively, lists replaced).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import config

GH_UVR = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
RAW_MSST = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/"
RAW_UVR_CFG = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"

MODELS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- separation
    "sep_hdemucs_mmi": {
        "kind": "separation", "type": "torchaudio_hdemucs", "bundle": "HDEMUCS_HIGH_MUSDB_PLUS",
        "license": "MIT (torchaudio Hybrid Demucs, MUSDB-HQ + extra data)",
        "size_mb": 320, "chunk_seconds": 10.0, "num_overlap": 2,
    },
    "melband_kim_ep3005": {
        "kind": "separation", "type": "mel_band_roformer",
        "license": "MIT (Kimberley Jensen Mel-Band RoFormer vocals, SDR 11.4)",
        "size_mb": 913, "num_overlap": 2,
        "ckpt": [{"hf": "KimberleyJSN/melbandroformer", "file": "MelBandRoformer.ckpt"},
                 {"url": GH_UVR + "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"}],
        "config": [{"url": RAW_UVR_CFG + "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml"}],
    },
    "bs_roformer_viperx_1297": {
        "kind": "separation", "type": "bs_roformer",
        "license": "see docs/MODELS.md (ViperX BS-RoFormer 317, SDR 12.97)",
        "size_mb": 610, "num_overlap": 2,
        "ckpt": [{"url": GH_UVR + "model_bs_roformer_ep_317_sdr_12.9755.ckpt"}],
        "config": [{"url": RAW_MSST + "viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"}],
    },
    "passthrough": {"kind": "separation", "type": "passthrough", "size_mb": 0},
    # ---------------------------------------------------------------- embeddings
    "ecapa_tao": {
        "kind": "embedding", "type": "ecapa_tao", "size_mb": 66,
        "license": "MIT (TaoRuijie/ECAPA-TDNN, VoxCeleb2)",
    },
    # ---------------------------------------------------------------- optional extras (HF only)
    "ast_audioset": {
        "kind": "tagger", "type": "hf_audio_tagger", "repo": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "allow_patterns": ["*.json", "*.safetensors"], "size_mb": 350,
        "license": "BSD-3-Clause (Audio Spectrogram Transformer, AudioSet)",
    },
    # ---------------------------------------------------------------- overlap separation
    "convtasnet_libri2mix": {
        "kind": "speech_separation", "type": "torchaudio", "bundle": "CONVTASNET_BASE_LIBRI2MIX",
        "sample_rate": 8000, "size_mb": 20, "license": "torchaudio pipeline (Libri2Mix, CC BY 4.0 data)",
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "name": "Fast",
        "description": "Hybrid Demucs separation, gender head on speaker embeddings + pitch. "
                       "Quickest; good for clean dialogue.",
        "separation": "sep_hdemucs_mmi",
        "embedding": "ecapa_tao",
        "window": 1.5, "hop": 0.75,
        "gender_models": ["head", "pitch"],
        "overlap": "off",
        "cluster_threshold": 0.65,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Mel-Band RoFormer vocal separation, fused gender models, speaker "
                       "clustering and overlap flagging.",
        "separation": "melband_kim_ep3005",
        "embedding": "ecapa_tao",
        "window": 1.5, "hop": 0.5,
        "gender_models": ["head", "hf", "pitch"],
        "overlap": "flag",
        "cluster_threshold": 0.65,
    },
    "best": {
        "name": "Best",
        "description": "Strongest RoFormer vocal separation, all gender models, overlapping "
                       "male/female speech is separated and only the female voice kept. Slowest.",
        "separation": "bs_roformer_viperx_1297",
        "embedding": "ecapa_tao",
        "window": 1.5, "hop": 0.5,
        "gender_models": ["head", "hf", "tagger", "pitch"],
        "overlap": "separate",
        "overlap_model": "convtasnet_libri2mix",
        "tagger": "ast_audioset",
        "cluster_threshold": 0.65,
    },
}

PRESET_ORDER = ["fast", "balanced", "best"]


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_overrides() -> tuple[dict, dict, list[str]]:
    models, presets = copy.deepcopy(MODELS), copy.deepcopy(PRESETS)
    notes: list[str] = []
    for path in (config.DOCS_DIR / "models.json", config.APP_DIR / "presets.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        except Exception as exc:
            notes.append(f"{path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        backend = data.get("backend", data)  # allow a {"backend": {...}} section
        for mid, spec in (backend.get("models") or {}).items():
            if isinstance(spec, dict) and _usable_model(spec, models.get(mid)):
                models[mid] = _merge(models.get(mid, {}), spec)
        for pid, spec in (backend.get("presets") or {}).items():
            if isinstance(spec, dict):
                presets[pid] = _merge(presets.get(pid, {}), spec)
    return models, presets, notes


def _usable_model(spec: dict, existing: dict | None) -> bool:
    merged = _merge(existing or {}, spec)
    return "type" in merged


_CACHE: tuple[dict, dict, list[str]] | None = None


def reload() -> None:
    global _CACHE
    _CACHE = None


def _get() -> tuple[dict, dict, list[str]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_overrides()
    return _CACHE


def models() -> dict[str, dict]:
    return _get()[0]


def presets() -> dict[str, dict]:
    return _get()[1]


def get_preset(pid: str) -> dict:
    p = presets().get(pid)
    if p is None:
        raise KeyError(f"unknown preset {pid!r} (choose {', '.join(PRESET_ORDER)})")
    out = dict(p)
    out["id"] = pid
    return out


def model_spec(mid: str) -> dict:
    m = models().get(mid)
    if m is None:
        raise KeyError(f"unknown model {mid!r}")
    out = copy.deepcopy(m)
    out.setdefault("id", mid)
    if out.get("type") == "ensemble":
        out["members"] = [model_spec(x) if isinstance(x, str) else x for x in out.get("members", [])]
    return out


def preset_list() -> list[dict]:
    ps = presets()
    order = [p for p in PRESET_ORDER if p in ps] + [p for p in ps if p not in PRESET_ORDER]
    return [{"id": p, "name": ps[p].get("name", p.title()), "description": ps[p].get("description", "")}
            for p in order if not ps[p].get("hidden")]
