"""EfficientAT DyMN AudioSet taggers, vendored from github.com/fschmid56/EfficientAT @a425fdc (MIT, see LICENSE).

Changes: relative imports, torchvision ConvNormActivation vendored (convnorm.py), prints removed,
weights loaded from a local file (``load_dymn``) instead of torch.hub, complex STFT in preprocess.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def labels() -> list[str]:
    with open(HERE / "class_labels_indices.csv", "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [r[2] for r in rows[1:]]


def load_dymn(weights: str, width_mult: float = 1.0, device: str = "cpu"):
    from .dymn_model import get_model
    from .preprocess import AugmentMelSTFT

    model = get_model(width_mult=width_mult, pretrained_name=None, T_max=1.0)
    sd = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
    return model.eval().to(device), mel.eval().to(device)
