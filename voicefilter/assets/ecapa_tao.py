"""ECAPA-TDNN speaker encoder (192-d) from TaoRuijie/ECAPA-TDNN (MIT licence).

Model definition vendored from https://github.com/TaoRuijie/ECAPA-TDNN (model.py, MIT,
Copyright (c) 2021 Ruijie Tao); pretrained on VoxCeleb2 (Vox1-O EER 0.96 %).
This is the same backbone the JaesungHuh/voice-gender-classifier was fine-tuned from.
It is used here because Hugging Face file downloads were blocked on the build machine;
the weights come from GitHub:

    https://github.com/TaoRuijie/ECAPA-TDNN/raw/main/exps/pretrain.model  (66 MB)

Usage:
    enc = load_ecapa(device="cpu")          # downloads weights to models/ext/ if missing
    emb = embed(enc, wav_16k_mono)          # np.float32 (192,), NOT normalised
"""
from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_URL = "https://github.com/TaoRuijie/ECAPA-TDNN/raw/main/exps/pretrain.model"
EMBEDDING_KIND = "ecapa_tao_vox2_192"


def default_weights_path() -> Path:
    env = os.environ.get("VOICEFILTER_ECAPA_TAO")
    if env:
        return Path(env)
    models = os.environ.get("VOICEFILTER_MODELS_DIR")
    base = Path(models) if models else Path(__file__).resolve().parents[2] / "models"
    return base / "ext" / "ECAPA-TDNN" / "exps" / "pretrain.model"


class _SE(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, bottleneck, 1), nn.ReLU(),
                                nn.Conv1d(bottleneck, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class _Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        pad = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=pad)
                                    for _ in range(self.nums)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = _SE(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.se(self.bn3(self.relu(self.conv3(out))))
        return out + residual


class _PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super().__init__()
        self.register_buffer("flipped_filter", torch.FloatTensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (1, 0), "reflect")
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, C=1024):
        super().__init__()
        import torchaudio
        self.torchfbank = nn.Sequential(
            _PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80))
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = _Bottle2neck(C, C, kernel_size=3, dilation=2)
        self.layer2 = _Bottle2neck(C, C, kernel_size=3, dilation=3)
        self.layer3 = _Bottle2neck(C, C, kernel_size=3, dilation=4)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(nn.Conv1d(4608, 256, 1), nn.ReLU(), nn.BatchNorm1d(256), nn.Tanh(),
                                       nn.Conv1d(256, 1536, 1), nn.Softmax(dim=2))
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        with torch.no_grad():
            x = (self.torchfbank(x) + 1e-6).log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
        x = self.bn1(self.relu(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
        t = x.size()[-1]
        gx = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                        torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(gx)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = self.bn5(torch.cat((mu, sg), 1))
        return self.bn6(self.fc6(x))


def load_ecapa(weights: str | Path | None = None, device: str = "cpu", download: bool = True) -> ECAPA_TDNN:
    path = Path(weights) if weights else default_weights_path()
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.replace(path)
    model = ECAPA_TDNN(C=1024)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    own = model.state_dict()
    new = {}
    for k, v in state.items():
        k2 = k.replace("speaker_encoder.", "").replace("module.", "")
        k2 = k2.replace("torchfbank.0.", "torchfbank.0.")
        if k2 in own and own[k2].shape == v.shape:
            new[k2] = v
    missing = [k for k in own if k not in new and not k.startswith("torchfbank")]
    if missing:
        raise RuntimeError(f"ECAPA weights incomplete, missing {missing[:5]}")
    model.load_state_dict(new, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def embed(model: ECAPA_TDNN, wav_16k_mono: np.ndarray, min_seconds: float = 0.5) -> np.ndarray:
    """192-d embedding of a 16 kHz mono clip (short clips are tiled to `min_seconds`)."""
    x = np.asarray(wav_16k_mono, dtype=np.float32)
    n_min = int(min_seconds * 16000)
    if len(x) == 0:
        x = np.zeros(n_min, np.float32)
    if len(x) < n_min:
        x = np.tile(x, int(np.ceil(n_min / len(x))))[:n_min]
    dev = next(model.parameters()).device
    t = torch.from_numpy(x).unsqueeze(0).to(dev)
    return model(t)[0].float().cpu().numpy()
