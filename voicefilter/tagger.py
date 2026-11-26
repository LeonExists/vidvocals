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
             "Whimper", "Groan", "Grunt", "Sigh", "Gasp", "Breathing", "Pant", "Wheeze", "Moan",
             "Screaming", "Shout", "Yell", "Battle cry", "Whoop", "Humming", "Cough", "Sneeze",
             "Throat clearing", "Sniff", "Snort", "Whistling", "Baby cry, infant cry")


class HFTagger:
    def __init__(self, spec: dict, device: str = "cpu"):
        self.spec, self.device = spec, device
        self.model = None

    def present(self) -> bool:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(self.spec["repo"], revision=self.spec.get("revision"), local_files_only=True)
            return True
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


def make_tagger(spec: dict, device: Optional[str] = "cpu") -> HFTagger:
    t = spec.get("type", "hf_audio_tagger")
    if t in ("hf_audio_tagger", "tagger"):
        return HFTagger(spec, device or "cpu")
    raise ValueError(f"unknown tagger type {t!r}")
