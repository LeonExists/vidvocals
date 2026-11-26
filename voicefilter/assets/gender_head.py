"""Small trained gender head: p(female) from [speaker embedding | pitch/spectral features].

Trained by eval/train_gender_head.py on speech AND non-verbal vocalizations
(speaker-disjoint from the evaluation speakers). Weights live next to this file as
pure numpy arrays (gender_head*.npz, loaded with allow_pickle=False).

Typical backend use:

    from voicefilter.assets.gender_head import load_head
    head = load_head()                       # default: gender_head.npz
    emb = <embedding of the segment, kind == head.embedding_kind>   (None if embedding_dim == 0)
    x = head.compute_features(wav_16k_mono, 16000, emb)
    p_female = float(head.predict_proba(x))

`compute_features` computes the non-embedding features exactly as in training, so the
backend must not re-implement them. Embedding kinds:
    "ecapa_tao_vox2_192" -> voicefilter.assets.ecapa_tao.embed(load_ecapa(), wav)
    "none"               -> pass embedding=None (pitch/spectral-only fallback head)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_HEAD = ASSET_DIR / "gender_head.npz"

SR = 16000
F0_MIN, F0_MAX = 60.0, 1000.0
FRAME, HOP = 1024, 160            # pyin frame (64 ms) and hop (10 ms) at 16 kHz
N_FFT = 512
ACTIVE_DB = 40.0                  # frames within 40 dB of the loudest frame are "active"
NEUTRAL_LOG_F0 = float(np.log(165.0))
N_MFCC = 13

HANDCRAFTED_NAMES = [
    "log_f0_median", "log_f0_p10", "log_f0_p90", "f0_iqr_semitones", "voiced_ratio", "voiced_prob_mean",
    "log_centroid_median", "log_rolloff85_median", "flatness_median", "zcr_median",
] + [f"mfcc{i}_mean" for i in range(1, N_MFCC)]


def _to_16k_mono(wav: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(wav, dtype=np.float32)
    if y.ndim == 2:  # (channels, n) or (n, channels)
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


HIGHPASS_HZ = 60.0


def pitch_track(y16: np.ndarray):
    """pyin F0 track (Hz, nan when unvoiced), voiced flag, voiced prob; 10 ms hop.

    The signal is high-passed at 60 Hz (zero-phase) first: low-frequency rumble/hum otherwise
    makes pyin lock onto sub-harmonics at the 60 Hz search bound (seen on phone/crowd recordings).
    Frames whose F0 sits on the lower bound (<= 1.03 * F0_MIN) are treated as unvoiced.
    """
    import librosa
    from scipy.signal import butter, sosfiltfilt
    if len(y16) < FRAME:
        y16 = np.pad(y16, (0, FRAME - len(y16)))
    if len(y16) > 64:
        sos = butter(4, HIGHPASS_HZ, "highpass", fs=SR, output="sos")
        y16 = sosfiltfilt(sos, y16).astype(np.float32)
    f0, vflag, vprob = librosa.pyin(y16, fmin=F0_MIN, fmax=F0_MAX, sr=SR, frame_length=FRAME,
                                    hop_length=HOP, center=True)
    vflag = vflag.astype(bool) & np.isfinite(f0) & (np.nan_to_num(f0) > 1.03 * F0_MIN)
    return f0, vflag, np.nan_to_num(vprob)


def handcrafted_features(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """The 22 non-embedding features (order = HANDCRAFTED_NAMES)."""
    import librosa
    y = _to_16k_mono(wav, sr)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)) + 1e-10
    rms = np.sqrt(np.mean(S ** 2, axis=0))
    db = 20 * np.log10(rms + 1e-12)
    active = db > (db.max() - ACTIVE_DB)
    if active.sum() < 1:
        active[:] = True

    f0, vflag, vprob = pitch_track(y)
    n = min(len(f0), len(active))
    f0, vflag, vprob, act_p = f0[:n], vflag[:n], vprob[:n], active[:n]
    voiced = vflag & act_p & np.isfinite(f0)
    if voiced.sum() >= 3:
        lf = np.log(f0[voiced])
        med, p10, p90 = np.median(lf), np.percentile(lf, 10), np.percentile(lf, 90)
        iqr = 12.0 / np.log(2) * (np.percentile(lf, 75) - np.percentile(lf, 25))
    else:
        med = p10 = p90 = NEUTRAL_LOG_F0
        iqr = 0.0
    voiced_ratio = float(voiced.sum()) / max(1, int(act_p.sum()))
    vprob_mean = float(vprob[act_p].mean()) if act_p.any() else 0.0

    Sa = S[:, active]
    cent = librosa.feature.spectral_centroid(S=Sa, sr=SR, n_fft=N_FFT)[0]
    roll = librosa.feature.spectral_rolloff(S=Sa, sr=SR, n_fft=N_FFT, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(S=Sa)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    zcr = zcr[: len(active)][active[: len(zcr)]] if len(zcr) else np.zeros(1)
    mel = librosa.feature.melspectrogram(S=Sa ** 2, sr=SR, n_fft=N_FFT, n_mels=40, fmin=20, fmax=7600)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)
    feats = [med, p10, p90, iqr, voiced_ratio, vprob_mean,
             np.log(np.median(cent) + 1.0), np.log(np.median(roll) + 1.0), np.median(flat),
             np.median(zcr) if len(zcr) else 0.0] + list(mfcc[1:].mean(axis=1))
    return np.asarray(feats, dtype=np.float32)


def prepare_embedding(embedding: np.ndarray | None, dim: int) -> np.ndarray:
    if dim == 0:
        return np.zeros(0, np.float32)
    if embedding is None:
        raise ValueError(f"this head needs a {dim}-d embedding")
    e = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if e.shape[0] != dim:
        raise ValueError(f"embedding has dim {e.shape[0]}, head expects {dim}")
    return e / (np.linalg.norm(e) + 1e-8)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


@dataclass
class GenderHead:
    feature_names: list[str]
    embedding_kind: str
    embedding_dim: int
    model_type: str                     # "logreg" | "mlp"
    mean: np.ndarray
    std: np.ndarray
    params: dict
    calib_a: float = 1.0                # Platt: p = sigmoid(a * logit + b)
    calib_b: float = 0.0
    info: dict = field(default_factory=dict)

    def compute_features(self, wav_16k_mono: np.ndarray, sr: int = SR,
                         embedding: np.ndarray | None = None) -> np.ndarray:
        e = prepare_embedding(embedding, self.embedding_dim)
        return np.concatenate([e, handcrafted_features(wav_16k_mono, sr)]).astype(np.float32)

    def logit(self, features: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if X.shape[1] != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} features, got {X.shape[1]}")
        X = (X - self.mean) / self.std
        p = self.params
        if self.model_type == "logreg":
            z = X @ p["w"] + p["b"]
        elif self.model_type == "mlp":
            h = np.maximum(0.0, X @ p["W1"] + p["b1"])
            z = h @ p["w2"] + p["b2"]
        else:
            raise ValueError(self.model_type)
        return np.asarray(z).reshape(-1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray | float:
        """Calibrated p(female); scalar for a single feature vector, array for a batch."""
        single = np.asarray(features).ndim == 1
        p = _sigmoid(self.calib_a * self.logit(features) + self.calib_b)
        return float(p[0]) if single else p


def save_head(path: str | Path, head: GenderHead) -> None:
    arrays = dict(
        feature_names=np.array(head.feature_names, dtype=np.str_),
        embedding_kind=np.array(head.embedding_kind, dtype=np.str_),
        embedding_dim=np.array(head.embedding_dim, dtype=np.int64),
        model_type=np.array(head.model_type, dtype=np.str_),
        mean=head.mean.astype(np.float64), std=head.std.astype(np.float64),
        calib_a=np.array(head.calib_a, dtype=np.float64), calib_b=np.array(head.calib_b, dtype=np.float64),
        info_json=np.array(json.dumps(head.info, sort_keys=True), dtype=np.str_),
    )
    for k, v in head.params.items():
        arrays["param_" + k] = np.asarray(v, dtype=np.float64)
    np.savez(str(path), **arrays)


def load_head(path: str | Path | None = None) -> GenderHead:
    path = Path(path) if path else DEFAULT_HEAD
    with np.load(str(path), allow_pickle=False) as z:
        params = {k[len("param_"):]: z[k] for k in z.files if k.startswith("param_")}
        params = {k: (float(v) if v.ndim == 0 else v) for k, v in params.items()}
        return GenderHead(
            feature_names=[str(s) for s in z["feature_names"]],
            embedding_kind=str(z["embedding_kind"]),
            embedding_dim=int(z["embedding_dim"]),
            model_type=str(z["model_type"]),
            mean=z["mean"], std=z["std"], params=params,
            calib_a=float(z["calib_a"]), calib_b=float(z["calib_b"]),
            info=json.loads(str(z["info_json"])),
        )


_DEFAULT: GenderHead | None = None


def _default() -> GenderHead:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_head()
    return _DEFAULT


def predict_proba(features: np.ndarray):
    """p(female) with the default head (gender_head.npz)."""
    return _default().predict_proba(features)


def compute_features(wav_16k_mono: np.ndarray, sr: int = SR, embedding: np.ndarray | None = None) -> np.ndarray:
    """Feature vector for the default head: [L2-normalised embedding | handcrafted features]."""
    return _default().compute_features(wav_16k_mono, sr, embedding)
