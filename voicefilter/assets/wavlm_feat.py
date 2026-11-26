"""WavLM Base+ pooled hidden-state embedding, exactly as used to train the gender head.

Model: torchaudio.pipelines.WAVLM_BASE_PLUS (weights from download.pytorch.org, ~360 MB,
MIT-licensed code / weights released by Microsoft). Embedding kind strings look like
"wavlm_base_plus_L4-5-6_mean": the time-mean of the outputs of transformer layers 4, 5 and 6
(1-based, as returned by `model.extract_features`), averaged over those layers -> 768-d.
The gender head L2-normalises it itself (do not normalise here).

    model = load_wavlm(device)                 # cached in TORCH_HOME
    emb = embed(model, wav_16k_mono, layers=(4, 5, 6))
    # or: emb = embed_for_head(model, wav, head.embedding_kind)
"""
from __future__ import annotations

import re

import numpy as np
import torch

MIN_SAMPLES = 8000  # clips shorter than 0.5 s are tiled to 0.5 s (as in training)


def load_wavlm(device: str = "cpu"):
    import torchaudio
    return torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval().to(device)


def layers_from_kind(kind: str) -> tuple[int, ...]:
    m = re.match(r"wavlm_base_plus_L([\d-]+)_mean$", kind)
    if not m:
        raise ValueError(f"not a WavLM embedding kind: {kind}")
    return tuple(int(x) for x in m.group(1).split("-"))


@torch.inference_mode()
def all_layer_means(model, wav_16k_mono: np.ndarray) -> np.ndarray:
    """(12, 768) time-mean of every transformer layer output."""
    y = np.asarray(wav_16k_mono, dtype=np.float32)
    if len(y) == 0:
        y = np.zeros(MIN_SAMPLES, np.float32)
    if len(y) < MIN_SAMPLES:
        y = np.tile(y, int(np.ceil(MIN_SAMPLES / len(y))))[:MIN_SAMPLES]
    dev = next(model.parameters()).device
    hs, _ = model.extract_features(torch.from_numpy(y)[None].to(dev))
    return np.stack([h[0].float().mean(0).cpu().numpy() for h in hs])


def embed(model, wav_16k_mono: np.ndarray, layers=(4, 5, 6)) -> np.ndarray:
    # training stored the per-layer means as float16, then averaged the chosen layers
    m = all_layer_means(model, wav_16k_mono).astype(np.float16).astype(np.float64)
    return m[[l - 1 for l in layers]].mean(0).astype(np.float32)


def embed_for_head(model, wav_16k_mono: np.ndarray, embedding_kind: str) -> np.ndarray:
    return embed(model, wav_16k_mono, layers_from_kind(embedding_kind))
