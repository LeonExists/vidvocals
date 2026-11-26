"""Optional AudioSet sound-event tagger (event kind + a weak gender opinion).

Default implementation: any Hugging Face ``AutoModelForAudioClassification`` trained on
AudioSet (e.g. the AST checkpoint ``MIT/ast-finetuned-audioset-10-10-0.4593``, BSD-3).
Outputs per clip (sigmoid scores)::

    [p_female_speech, p_male_speech, p_speech, p_nonverbal]

p_nonverbal is the max over AudioSet's non-verbal human vocalisation classes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import features

FEMALE = ("Female speech, woman speaking", "Female singing")
MALE = ("Male speech, man speaking", "Male singing")
SPEECH = ("Speech", "Conversation", "Narration, monologue", "Male speech, man speaking",
          "Female speech, woman speaking", "Child speech, kid speaking")
NONVERBAL = ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing",
             "Whimper", "Groan", "Grunt", "Sigh", "Gasp", "Breathing", "Pant", "Wheeze", "Moan", "Wail, moan",
             "Screaming", "Shout", "Yell", "Battle cry", "Whoop", "Humming", "Cough", "Sneeze",
             "Throat clearing", "Sniff", "Snort", "Whistling", "Baby cry, infant cry")


class HFTagger:
    def __init__(self, spec: dict, device: str = "cpu"):
        self.spec, self.device = spec, device
        self.model = None

    def present(self) -> bool:
        try:
            from huggingface_hub import snapshot_download

            d = snapshot_download(self.spec["repo"], revision=self.spec.get("revision"), local_files_only=True)
            from pathlib import Path as _P

            return any(_P(d).glob("*.safetensors")) or any(_P(d).glob("*.bin"))
        except Exception:
            return False

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.spec["repo"], revision=self.spec.get("revision"),
                          allow_patterns=self.spec.get("allow_patterns") or ["*.json", "*.safetensors", "*.txt"])

    def load(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        repo = self.spec["repo"]
        self.fe = AutoFeatureExtractor.from_pretrained(repo, revision=self.spec.get("revision"))
        self.model = AutoModelForAudioClassification.from_pretrained(repo, revision=self.spec.get("revision"))
        self.model.eval().to(self.device)
        lab = {str(v): int(k) for k, v in self.model.config.id2label.items()}
        pick = lambda names: [lab[n] for n in names if n in lab]  # noqa: E731
        self.idx = [pick(FEMALE), pick(MALE), pick(SPEECH), pick(NONVERBAL)]

    def predict_many(self, clips: list[np.ndarray], bs: int = 8) -> np.ndarray:
        import torch

        out = np.zeros((len(clips), 4), np.float32)
        with torch.inference_mode():
            for k in range(0, len(clips), bs):
                batch = [np.asarray(c, np.float32) for c in clips[k:k + bs]]
                inp = self.fe(batch, sampling_rate=features.SR16, return_tensors="pt")
                inp = {kk: v.to(self.device) for kk, v in inp.items()}
                p = torch.sigmoid(self.model(**inp).logits.float()).cpu().numpy()
                for j, ids in enumerate(self.idx):
                    out[k:k + len(batch), j] = p[:, ids].max(axis=1) if ids else 0.0
        return out

    def unload(self) -> None:
        self.model = None
        features.free_torch()


class EfficientATTagger:
    """EfficientAT DyMN (dymn10_as / dymn20_as, MIT, weights from GitHub releases)."""

    def __init__(self, spec: dict, device: str = "cpu"):
        self.spec, self.device = spec, device
        self.model = None

    def present(self) -> bool:
        from . import modelstore

        return modelstore.present(self.spec["id"], self.spec["weights"])

    def download(self) -> None:
        from . import modelstore

        modelstore.resolve(self.spec["id"], self.spec["weights"], download=True, verbose=True)

    def load(self) -> None:
        from . import modelstore
        from .third_party import efficientat

        path = modelstore.resolve(self.spec["id"], self.spec["weights"], download=True)
        self.model, self.mel = efficientat.load_dymn(path, float(self.spec.get("width_mult", 1.0)), self.device)
        lab = {n: i for i, n in enumerate(efficientat.labels())}
        pick = lambda names: [lab[n] for n in names if n in lab]  # noqa: E731
        self.idx = [pick(FEMALE[:1]), pick(MALE[:1]), pick(SPEECH), pick(NONVERBAL)]

    def predict_many(self, clips: list[np.ndarray], bs: int = 16) -> np.ndarray:
        import soxr
        import torch

        out = np.zeros((len(clips), 4), np.float32)
        x32 = [soxr.resample(np.asarray(c, np.float32), features.SR16, 32000) if len(c) else np.zeros(3200, np.float32)
               for c in clips]
        by_len: dict[int, list[int]] = {}
        for i, x in enumerate(x32):
            by_len.setdefault(len(x), []).append(i)
        with torch.inference_mode():
            for _, ids in by_len.items():
                for k in range(0, len(ids), bs):
                    sel = ids[k:k + bs]
                    t = torch.from_numpy(np.stack([x32[i] for i in sel])).to(self.device)
                    logits, _ = self.model(self.mel(t).unsqueeze(1))
                    p = torch.sigmoid(logits.float()).cpu().numpy()
                    for j, cls in enumerate(self.idx):
                        out[sel, j] = p[:, cls].max(axis=1) if cls else 0.0
        return out

    def unload(self) -> None:
        self.model = None
        features.free_torch()


def make_tagger(spec: dict, device: Optional[str] = "cpu"):
    t = spec.get("type", "hf_audio_tagger")
    if t == "efficientat":
        return EfficientATTagger(spec, device or "cpu")
    if t in ("hf_audio_tagger", "tagger"):
        return HFTagger(spec, device or "cpu")
    raise ValueError(f"unknown tagger type {t!r}")
