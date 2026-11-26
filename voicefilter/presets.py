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
GH_NOMAD = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
RAW_UVR_CFG = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"

MODELS: dict[str, dict[str, Any]] = {
    # ---------------------------------------------------------------- separation
    "sep_hdemucs_mmi": {
        "kind": "separation", "type": "torchaudio_hdemucs", "bundle": "HDEMUCS_HIGH_MUSDB_PLUS",
        "license": "MIT (torchaudio Hybrid Demucs, MUSDB-HQ + extra data)",
        "size_mb": 320, "chunk_seconds": 10.0, "num_overlap": 2,
    },
    "sep_resurrection_bs": {
        "kind": "separation", "type": "bs_roformer",
        "license": "not stated (unwa BS-RoFormer Resurrection vocals; free download, personal use)",
        "size_mb": 204, "num_overlap": 2,
        "ckpt": [{"path": "models/separation/bs_roformer_vocals_resurrection_unwa.ckpt"},
                 {"hf": "pcunwa/BS-Roformer-Resurrection", "file": "BS-Roformer-Resurrection.ckpt"},
                 {"url": GH_NOMAD + "bs_roformer_vocals_resurrection_unwa.ckpt"}],
        "config": [{"path": "models/separation/config_bs_roformer_vocals_resurrection_unwa.yaml"},
                   {"url": GH_NOMAD + "config_bs_roformer_vocals_resurrection_unwa.yaml"}],
    },
    "sep_kim_melband": {
        "kind": "separation", "type": "mel_band_roformer",
        "license": "MIT (Kimberley Jensen Mel-Band RoFormer vocals)",
        "size_mb": 913, "num_overlap": 2,
        "ckpt": [{"path": "models/separation/vocals_mel_band_roformer.ckpt"},
                 {"hf": "KimberleyJSN/melbandroformer", "file": "MelBandRoformer.ckpt"},
                 {"url": GH_NOMAD + "vocals_mel_band_roformer.ckpt"}],
        "config": [{"path": "models/separation/vocals_mel_band_roformer.yaml"},
                   {"url": GH_NOMAD + "vocals_mel_band_roformer.yaml"}],
    },
    "sep_viperx_bs_317": {
        "kind": "separation", "type": "bs_roformer",
        "license": "not stated (viperx BS-RoFormer ep_317, UVR public release; personal use)",
        "size_mb": 610, "num_overlap": 2,
        "ckpt": [{"path": "models/separation/model_bs_roformer_ep_317_sdr_12.9755.ckpt"},
                 {"url": GH_UVR + "model_bs_roformer_ep_317_sdr_12.9755.ckpt"}],
        "config": [{"path": "models/separation/model_bs_roformer_ep_317_sdr_12.9755.yaml"},
                   {"url": RAW_MSST + "viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"}],
    },
    "sep_ens_kim_viperx": {
        "kind": "separation", "type": "ensemble", "members": ["sep_kim_melband", "sep_viperx_bs_317"],
        "size_mb": 1523, "license": "see members",
    },
    "passthrough": {"kind": "separation", "type": "passthrough", "size_mb": 0},
    # ---------------------------------------------------------------- embeddings
    "ecapa_tao": {
        "kind": "embedding", "type": "ecapa_tao", "size_mb": 66,
        "license": "MIT (TaoRuijie/ECAPA-TDNN, VoxCeleb2)",
    },
    # ---------------------------------------------------------------- optional extras (HF only)
    "tagger_dymn10": {
        "kind": "tagger", "type": "efficientat", "width_mult": 1.0, "size_mb": 43,
        "license": "MIT (EfficientAT DyMN, AudioSet)",
        "weights": [{"url": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn10_as.pt",
                     "sha256": "f363baafb6f9dd807c55a60254565ff94373ea0964f08730ea8cf23d1865725d"}],
    },
    "tagger_dymn20": {
        "kind": "tagger", "type": "efficientat", "width_mult": 2.0, "size_mb": 160,
        "license": "MIT (EfficientAT DyMN, AudioSet)",
        "weights": [{"url": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn20_as_mAP_493.pt"}],
    },
    "ast_audioset": {
        "kind": "tagger", "type": "hf_audio_tagger", "repo": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "allow_patterns": ["*.json", "*.safetensors"], "size_mb": 350,
        "license": "BSD-3-Clause (Audio Spectrogram Transformer, AudioSet)",
    },
    # ---------------------------------------------------------------- overlap separation
    "ovl_convtasnet_libri2mix": {
        "kind": "speech_separation", "type": "torchaudio", "bundle": "CONVTASNET_BASE_LIBRI2MIX",
        "sample_rate": 8000, "size_mb": 20, "license": "torchaudio pipeline (Libri2Mix, CC BY 4.0 data)",
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "name": "Fast",
        "description": "Small BS-RoFormer vocal separation, gender head on speaker embeddings + "
                       "pitch, speaker clustering. Quickest.",
        "separation": "sep_resurrection_bs",
        "embedding": "ecapa_tao",
        "window": 1.5, "hop": 0.75,
        "gender_models": ["head", "tagger", "pitch"],
        "tagger": "tagger_dymn10",
        "overlap": "off",
        "cluster_threshold": 0.65,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Mel-Band RoFormer vocal separation, fused gender models, speaker "
                       "clustering and overlap flagging.",
        "separation": "sep_kim_melband",
        "embedding": "ecapa_tao",
        "window": 1.5, "hop": 0.5,
        "gender_models": ["head", "hf", "tagger", "pitch"],
        "tagger": "tagger_dymn10",
        "overlap": "flag",
        "cluster_threshold": 0.65,
    },
    "best": {
        "name": "Best",
        "description": "Ensemble of two RoFormer vocal models, all gender models, overlapping "
                       "male/female speech is separated and only the female voice kept. Slowest.",
        "separation": "sep_ens_kim_viperx",
        "embedding": "ecapa_tao",
        "window": 1.5, "hop": 0.5,
        "gender_models": ["head", "hf", "tagger", "pitch"],
        "overlap": "separate",
        "overlap_model": "ovl_convtasnet_libri2mix",
        "tagger": "tagger_dymn20",
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


def _src(src: dict, sha: str | None = None) -> dict | None:
    """docs/models.json source entry -> modelstore source."""
    k = src.get("kind")
    if k == "hf" and src.get("repo_id") and src.get("filename"):
        out = {"hf": src["repo_id"], "file": src["filename"]}
    elif k == "url" and src.get("url"):
        out = {"url": src["url"]}
        if src.get("sha256") or sha:
            out["sha256"] = src.get("sha256") or sha
    else:
        return None
    return out


def _translate_research(data: dict) -> dict:
    """Translate the research schema of docs/models.json ({_doc, models, presets} with
    ``sources: [{kind: hf|url, ...}]``) into this module's schema.  Unknown model types are
    skipped (they stay documentation), so the file can carry optional upgrades."""
    models_out: dict[str, dict] = {}
    for mid, m in (data.get("models") or {}).items():
        if not isinstance(m, dict):
            continue
        t = m.get("type")
        local = f"models/{m['local_dir']}/" if m.get("local_dir") else None
        inf = m.get("inference") or {}
        if t in ("bs_roformer", "mel_band_roformer") and m.get("filename") and m.get("config"):
            ckpt = ([{"path": local + m["filename"]}] if local else []) + \
                [x for x in (_src(s_, None) for s_ in m.get("sources", [])) if x]
            for x in ckpt:  # one sha for every mirror (byte-identical)
                if "url" in x and m.get("sha256"):
                    x.setdefault("sha256", m["sha256"])
            cfg = m["config"]
            conf = ([{"path": local + cfg["filename"]}] if local and cfg.get("filename") else []) + \
                [x for x in (_src(s_) for s_ in cfg.get("sources", [])) if x]
            spec = {"kind": "separation", "type": t, "ckpt": ckpt, "config": conf,
                    "license": m.get("license", ""), "size_mb": round((m.get("size_bytes") or 0) / 2**20)}
            for k in ("chunk_size", "num_overlap", "fp16"):
                if k in inf:
                    spec[k] = inf[k]
            if inf.get("target"):
                spec["target"] = str(inf["target"]).lower()
            if m.get("verified_here") is not True:
                spec["unverified"] = True
            models_out[mid] = spec
        elif t == "efficientat_dymn" and m.get("filename"):
            name = str(m.get("pretrained_name", mid))
            wm = 2.0 if "20" in name else (0.4 if "04" in name else 1.0)
            weights = ([{"path": local + m["filename"]}] if local else []) + \
                [x for x in (_src(s_, m.get("sha256")) for s_ in m.get("sources", [])) if x]
            models_out[mid] = {"kind": "tagger", "type": "efficientat", "width_mult": wm, "weights": weights,
                               "license": m.get("license", "MIT")}
    known = set(MODELS) | set(models_out)
    alias = {"emb_ecapa_taoruijie": "ecapa_tao", "ovl_convtasnet_libri2mix": "ovl_convtasnet_libri2mix"}
    presets_out: dict[str, dict] = {}
    for pid, p in (data.get("presets") or {}).items():
        if not isinstance(p, dict):
            continue
        out: dict[str, Any] = {}
        sep = p.get("separation")
        if isinstance(sep, dict):
            mm = sep.get("model")
            if isinstance(mm, list) and mm and all(x in known for x in mm):
                eid = next((k for k, v in MODELS.items() if v.get("type") == "ensemble"
                            and list(v.get("members", [])) == list(mm)), "sep_ens_" + "_".join(mm))
                models_out[eid] = {"kind": "separation", "type": "ensemble", "members": list(mm)}
                out["separation"] = eid
            elif isinstance(mm, str) and mm in known:
                out["separation"] = mm
        tg = p.get("tagger")
        if isinstance(tg, dict):
            for cand in (tg.get("model"), tg.get("fallback")):
                if isinstance(cand, str) and cand in known:
                    out["tagger"] = cand
                    break
        emb = p.get("embedding")
        if isinstance(emb, dict) and alias.get(emb.get("model", "")):
            out["embedding"] = alias[emb["model"]]
        ov = p.get("overlap_separation")
        if isinstance(ov, dict):
            for cand in (ov.get("model"), ov.get("fallback")):
                c = alias.get(cand, cand) if isinstance(cand, str) else None
                if c and c in known:
                    out["overlap_model"] = c
                    break
        if out:
            presets_out[pid] = out
    return {"models": models_out, "presets": presets_out}


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
        if any(isinstance(m, dict) and "sources" in m for m in (data.get("models") or {}).values()):
            data = _translate_research(data)
        for mid, spec in (data.get("models") or {}).items():
            if isinstance(spec, dict) and _usable_model(spec, models.get(mid)):
                models[mid] = _merge(models.get(mid, {}), spec) if not spec.get("ckpt") else \
                    _merge({k: v for k, v in models.get(mid, {}).items() if k not in ("ckpt", "config")}, spec)
        for pid, spec in (data.get("presets") or {}).items():
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
