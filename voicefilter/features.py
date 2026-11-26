"""Feature extraction on the vocal stem (16 kHz mono).

* resampling 44.1 kHz stereo -> 16 kHz mono (soxr, streamed from disk)
* analysis windows inside vocal events (default 1.5 s window, 0.5 s hop; short events get
  one window covering the event)
* pitch (F0) track with a vectorised YIN (cumulative mean normalised difference) giving F0
  and periodicity per 10 ms frame, and per-window statistics
* model wrappers, all lazy and following one GPU-memory policy: a model is loaded for its
  stage, used with batched inference, then released (``del`` + ``torch.cuda.empty_cache``):

  - speaker embeddings (``ecapa_tao`` 192-d ECAPA-TDNN, the embedding the gender head is
    trained on; optional ``speechbrain`` ECAPA)
  - optional Hugging Face audio-classification gender model (wav2vec2 style)
  - optional AudioSet tagger for the event kind (speech vs non-verbal)
"""
from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import config, modelstore

SR16 = 16000
Progress = Optional[Callable[[float, str], None]]


def free_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def load_mono_16k(wav: str | Path, block: int = 1 << 20) -> np.ndarray:
    """Read a wav (any rate/channels) as 16 kHz mono float32, streamed through soxr."""
    import soundfile as sf
    import soxr

    with sf.SoundFile(str(wav)) as f:
        sr = f.samplerate
        n_out = int(np.ceil(f.frames * SR16 / sr)) + 16
        out = np.zeros(n_out, np.float32)
        pos = 0
        rs = soxr.ResampleStream(sr, SR16, 1, dtype="float32", quality="HQ") if sr != SR16 else None
        while True:
            x = f.read(block, dtype="float32", always_2d=True)
            last = len(x) < block
            m = x.mean(axis=1) if len(x) else np.zeros(0, np.float32)
            y = rs.resample_chunk(m, last=last) if rs is not None else m
            if len(y):
                k = min(len(y), n_out - pos)
                out[pos:pos + k] = y[:k]
                pos += k
            if last:
                break
    expected = int(round(sf.info(str(wav)).frames * SR16 / sr))
    return out[:expected]


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
@dataclass
class Window:
    start: float
    end: float
    event: int
    idx: int = 0
    feats: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.end - self.start


def make_windows(events: Sequence[Any], win: float = 1.5, hop: float = 0.5,
                 min_win: float = 0.25) -> list[Window]:
    """Sliding windows inside events.  Events shorter than ``win`` get a single window."""
    out: list[Window] = []
    for ei, ev in enumerate(events):
        s, e = float(ev.start), float(ev.end)
        d = e - s
        if d <= win + 1e-6:
            out.append(Window(s, e, ei))
            continue
        n = int(np.floor((d - win) / hop)) + 1
        starts = [s + k * hop for k in range(n)]
        if starts[-1] + win < e - 0.05:  # cover the tail
            starts.append(e - win)
        for st in starts:
            out.append(Window(st, min(e, st + win), ei))
    for i, w in enumerate(out):
        w.idx = i
    return [w for w in out if w.dur >= min(min_win, 0.05)]


def cut(x16: np.ndarray, s: float, e: float) -> np.ndarray:
    a, b = max(0, int(s * SR16)), min(len(x16), int(np.ceil(e * SR16)))
    return x16[a:b]


# ---------------------------------------------------------------------------
# pitch (vectorised YIN)
# ---------------------------------------------------------------------------
def yin_track(x: np.ndarray, sr: int = SR16, fmin: float = 60.0, fmax: float = 1000.0,
              frame: int = 1024, hop: int = 160, threshold: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0 Hz, periodicity 0..1) per hop; f0 = 0 for unvoiced frames.

    Uses the YIN cumulative-mean-normalised difference computed with FFTs, in batches.
    periodicity = 1 - d'(tau*) (high for clearly periodic voiced frames).
    """
    x = np.asarray(x, np.float32)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n_frames = 1 + (len(x) - frame) // hop
    tau_min = max(2, int(sr / fmax))
    tau_max = min(frame // 2, int(sr / fmin))
    W = frame // 2  # integration window
    f0 = np.zeros(n_frames, np.float32)
    per = np.zeros(n_frames, np.float32)
    nfft = 1
    while nfft < frame * 2:
        nfft *= 2
    B = 2048
    idx = np.arange(frame)[None, :]
    for b0 in range(0, n_frames, B):
        nb = min(B, n_frames - b0)
        fr = x[(np.arange(b0, b0 + nb) * hop)[:, None] + idx].astype(np.float64)
        fr -= fr.mean(axis=1, keepdims=True)
        energy = (fr[:, :W] ** 2).sum(axis=1)
        # difference function d(tau) = r0 + r_tau(shifted energy) - 2 * acf(tau)
        F = np.fft.rfft(fr, nfft)
        Fw = np.fft.rfft(fr[:, :W], nfft)
        acf = np.fft.irfft(np.conj(Fw) * F, nfft)[:, :tau_max + 1]
        sq = np.concatenate([np.zeros((nb, 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
        taus = np.arange(tau_max + 1)
        e_tau = sq[:, taus + W] - sq[:, taus]
        d = energy[:, None] + e_tau - 2 * acf
        d[:, 0] = 0
        cs = np.cumsum(d[:, 1:], axis=1)
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] * np.arange(1, tau_max + 1)[None, :] / np.maximum(cs, 1e-12)
        sub = dn[:, tau_min:tau_max + 1]
        below = sub < threshold
        has = below.any(axis=1)
        first = np.where(has, below.argmax(axis=1), sub.argmin(axis=1))
        # walk to the local minimum after the first dip below threshold
        rows = np.arange(nb)
        t = first.copy()
        for _ in range(64):
            nxt = np.minimum(t + 1, sub.shape[1] - 1)
            better = sub[rows, nxt] < sub[rows, t]
            if not better.any():
                break
            t = np.where(better, nxt, t)
        val = sub[rows, t]
        # parabolic interpolation
        tl = np.maximum(t - 1, 0)
        tr = np.minimum(t + 1, sub.shape[1] - 1)
        a, bb, c = sub[rows, tl], val, sub[rows, tr]
        den = a - 2 * bb + c
        shift = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
        shift = np.clip(shift, -1, 1)
        tau = t + tau_min + shift
        p = np.clip(1.0 - val, 0.0, 1.0)
        voiced = has & (energy > 1e-7 * W)
        f0[b0:b0 + nb] = np.where(voiced, sr / np.maximum(tau, 1), 0.0)
        per[b0:b0 + nb] = np.where(energy > 1e-9 * W, p, 0.0)
    return f0, per


def pitch_stats(f0: np.ndarray, per: np.ndarray, hop_s: float, s: float, e: float) -> dict:
    a, b = int(s / hop_s), max(int(s / hop_s) + 1, int(e / hop_s))
    ff, pp = f0[a:b], per[a:b]
    v = ff > 0
    if v.sum() < 3:
        return {"f0_med": 0.0, "f0_iqr": 0.0, "voiced": float(v.mean()) if len(v) else 0.0,
                "periodicity": float(pp.mean()) if len(pp) else 0.0, "f0_p10": 0.0, "f0_p90": 0.0}
    lf = np.log2(ff[v])
    q10, q25, q50, q75, q90 = np.percentile(lf, [10, 25, 50, 75, 90])
    return {"f0_med": float(2 ** q50), "f0_iqr": float(q75 - q25), "voiced": float(v.mean()),
            "periodicity": float(pp[v].mean()), "f0_p10": float(2 ** q10), "f0_p90": float(2 ** q90)}


def pitch_p_female(st: dict) -> tuple[float, float]:
    """Weak prior from F0 alone: (p_female, reliability 0..1).

    Logistic on log2 F0 centred at ~165 Hz (male speech ~85-155 Hz, female ~165-255 Hz).
    Reliability grows with the voiced ratio and shrinks for very high F0 (screams/laughs
    of both sexes go above 300 Hz).
    """
    f = st.get("f0_med", 0.0)
    if f <= 0 or st.get("voiced", 0) < 0.15:
        return 0.5, 0.0
    z = (np.log2(f) - np.log2(165.0)) / 0.3
    p = float(np.clip(1 / (1 + np.exp(-z)), 0.1, 0.9))  # F0 alone is never decisive
    rel = min(1.0, st["voiced"] / 0.5) * (0.4 if f > 320 else 1.0)
    return p, float(rel)


# ---------------------------------------------------------------------------
# speaker embeddings
# ---------------------------------------------------------------------------
ECAPA_TAO_SOURCES = [
    {"path": "models/ext/ECAPA-TDNN/exps/pretrain.model"},  # shared with eval/
    {"url": "https://raw.githubusercontent.com/TaoRuijie/ECAPA-TDNN/main/exps/pretrain.model",
     "filename": "pretrain.model"},
]


class Embedder:
    kind = "none"
    dim = 0

    def load(self, device: str) -> None: ...

    def embed_batch(self, clips: list[np.ndarray]) -> np.ndarray: ...

    def unload(self) -> None:
        free_torch()


class EcapaTaoEmbedder(Embedder):
    """TaoRuijie ECAPA-TDNN (VoxCeleb2, 192-d, MIT).  Kind ``ecapa_tao_vox2_192``."""

    kind = "ecapa_tao_vox2_192"
    dim = 192
    sources = ECAPA_TAO_SOURCES

    def __init__(self, spec: Optional[dict] = None):
        self.spec = spec or {}
        self.model = None

    @classmethod
    def present(cls) -> bool:
        return modelstore.present("ecapa_tao", cls.sources)

    @classmethod
    def download(cls) -> str:
        return modelstore.resolve("ecapa_tao", cls.sources, download=True, verbose=True)

    def load(self, device: str) -> None:
        from .assets import ecapa_tao

        path = modelstore.resolve("ecapa_tao", self.sources, download=True)
        self.model = ecapa_tao.load_ecapa(path, device=device, download=False)
        self.device = device
        self.kind = getattr(ecapa_tao, "EMBEDDING_KIND", self.kind)

    def embed_batch(self, clips: list[np.ndarray], min_s: float = 0.5) -> np.ndarray:
        import torch

        n_min = int(min_s * SR16)
        prepared = []
        for x in clips:
            x = np.asarray(x, np.float32)
            if len(x) == 0:
                x = np.zeros(n_min, np.float32)
            if len(x) < n_min:
                x = np.tile(x, int(np.ceil(n_min / len(x))))[:n_min]
            prepared.append(x)
        out = np.zeros((len(prepared), self.dim), np.float32)
        # group equal lengths -> batched; others one by one
        by_len: dict[int, list[int]] = {}
        for i, x in enumerate(prepared):
            by_len.setdefault(len(x), []).append(i)
        bs = 64 if self.device == "cuda" else 16
        with torch.inference_mode():
            for L, ids in by_len.items():
                for k in range(0, len(ids), bs):
                    sel = ids[k:k + bs]
                    t = torch.from_numpy(np.stack([prepared[i] for i in sel])).to(self.device)
                    out[sel] = self.model(t).float().cpu().numpy()
        return out

    def unload(self) -> None:
        self.model = None
        free_torch()


EMBEDDERS = {"ecapa_tao": EcapaTaoEmbedder, "ecapa_tao_vox2_192": EcapaTaoEmbedder}


def make_embedder(name: str) -> Embedder:
    return EMBEDDERS[name]()


# ---------------------------------------------------------------------------
# gender head (trained by eval/, lives in voicefilter/assets/gender_head.py)
# ---------------------------------------------------------------------------
SUPPORTED_EMBEDDINGS = {"ecapa_tao_vox2_192": "ecapa_tao", "none": None}


class GenderHead:
    """Wrapper around assets/gender_head.py (``load_head`` -> object with
    ``embedding_kind``, ``compute_features(wav16, sr, emb)``, ``predict_proba(X)``).

    Prefers ``gender_head.npz``; if its embedding kind is not supported here, the first
    ``gender_head*.npz`` with a supported kind is used.  ``embedder`` names the embedder
    the head needs (None for the pitch/spectral-only head).
    """

    def __init__(self, path: Optional[str] = None):
        self.mod = None
        self.head = None
        self.embedding_kind: Optional[str] = None
        self.embedder: Optional[str] = None
        self.error: Optional[str] = None
        self.path: Optional[str] = None
        try:
            self.mod = importlib.import_module(".assets.gender_head", __package__)
            cands = [Path(path)] if path else []
            default = Path(getattr(self.mod, "DEFAULT_HEAD", config.ASSETS_DIR / "gender_head.npz"))
            cands += [default] + sorted(p for p in config.ASSETS_DIR.glob("gender_head*.npz") if p != default)
            for c in cands:
                if not c.exists():
                    continue
                h = self.mod.load_head(c)
                kind = str(getattr(h, "embedding_kind", "none") or "none")
                if kind in SUPPORTED_EMBEDDINGS:
                    self.head, self.embedding_kind, self.path = h, kind, str(c)
                    self.embedder = SUPPORTED_EMBEDDINGS[kind]
                    break
            if self.head is None:
                raise FileNotFoundError("no usable gender_head*.npz in assets/")
        except Exception as exc:  # missing or broken head -> pitch fallback
            self.error = f"{type(exc).__name__}: {exc}"
            self.head = None

    @property
    def available(self) -> bool:
        return self.head is not None

    def features(self, wav16: np.ndarray, embedding: Optional[np.ndarray]) -> np.ndarray:
        return self.head.compute_features(wav16, SR16, embedding if self.embedder else None)

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.asarray(self.head.predict_proba(X), np.float64).reshape(-1)

    def predict_many(self, clips: list[np.ndarray], embs: Optional[np.ndarray] = None,
                     progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
        if not clips:
            return np.zeros(0)
        feats = []
        for i, c in enumerate(clips):
            feats.append(self.features(c, None if embs is None else embs[i]))
            if progress and i % 20 == 0:
                progress(i / len(clips))
        return self.predict_features(np.stack(feats))


# ---------------------------------------------------------------------------
# Hugging Face audio-classification gender model (optional 2nd opinion)
# ---------------------------------------------------------------------------
class HFGenderModel:
    """Any HF ``AutoModelForAudioClassification`` whose labels contain female/male."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.model = None

    def present(self) -> bool:
        try:
            from huggingface_hub import snapshot_download

            d = snapshot_download(self.spec["repo"], revision=self.spec.get("revision"),
                                  local_files_only=True)
            return any(Path(d).glob("*.safetensors")) or any(Path(d).glob("*.bin"))
        except Exception:
            return False

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(self.spec["repo"], revision=self.spec.get("revision"),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.bin"]
                          if not self.spec.get("allow_patterns") else self.spec["allow_patterns"])

    def load(self, device: str) -> None:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        repo = self.spec["repo"]
        self.fe = AutoFeatureExtractor.from_pretrained(repo, revision=self.spec.get("revision"))
        self.model = AutoModelForAudioClassification.from_pretrained(repo, revision=self.spec.get("revision"))
        self.model.eval().to(device)
        self.device = device
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.i_f = next(i for i, l in labels.items() if l.startswith("f") or "female" in l or "woman" in l)
        self.i_m = next(i for i, l in labels.items() if (l.startswith("m") and "female" not in l)
                        or l == "male" or "man" == l)

    def predict_many(self, clips: list[np.ndarray], bs: int = 16) -> np.ndarray:
        import torch

        out = np.zeros(len(clips))
        with torch.inference_mode():
            for k in range(0, len(clips), bs):
                batch = [np.asarray(c, np.float32) if len(c) >= 4000 else
                         np.pad(np.asarray(c, np.float32), (0, 4000 - len(c))) for c in clips[k:k + bs]]
                inp = self.fe(batch, sampling_rate=SR16, return_tensors="pt", padding=True)
                inp = {kk: v.to(self.device) for kk, v in inp.items()}
                logits = self.model(**inp).logits.float()
                lf, lm = logits[:, self.i_f], logits[:, self.i_m]
                out[k:k + len(batch)] = torch.sigmoid(lf - lm).cpu().numpy()
        return out

    def unload(self) -> None:
        self.model = None
        free_torch()
