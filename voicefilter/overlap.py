"""Overlapping male + female speech.

Detection (all presets with overlap != "off")
    A window is an overlap candidate when its embedding is similar to the centroids of two
    speakers of *opposite* gender (top-2 cosine similarities both >= ``sim_min`` and within
    ``margin`` of each other), or when the AudioSet tagger (if present) reports both
    "Male speech" and "Female speech".  Runs of >= ``min_windows`` candidate windows
    (>= ``min_dur``) become overlap regions; segments are split there and labelled
    "overlap".

Resolution
    * without separation (``overlap="flag"``): decision per ``overlap_policy``
      (``p_female``: keep if the fused p_female >= 0.5; ``keep``; ``remove``).
    * Best (``overlap="separate"``): a 2-speaker speech separation model splits the
      region's vocal stem into two sources; each is classified with the gender head.  If one
      is female and the other male, a *replacement* vocal signal containing only the female
      part is built: the separated sources (8/16 kHz) are turned into a soft time-frequency
      mask (female power share; above the separator's band the per-frame share of the
      1-3.8 kHz band is used) that is applied to the full-band 44.1 kHz vocal stem, so the
      kept voice keeps its full bandwidth.  render.py crossfades it in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import features, modelstore


@dataclass
class OverlapParams:
    sim_min: float = 0.30
    margin: float = 0.12
    min_windows: int = 2
    min_dur: float = 0.8
    tagger_min: float = 0.35


def candidate_windows(E: np.ndarray, labels: np.ndarray, spk_logit: dict[int, float],
                      p: OverlapParams, tagger: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean per window."""
    n = len(E)
    cand = np.zeros(n, bool)
    ks = sorted(spk_logit)
    if len(ks) >= 2 and n:
        X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        C = np.stack([X[labels == k].mean(0) for k in ks])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        S = X @ C.T
        order = np.argsort(-S, axis=1)
        s1 = S[np.arange(n), order[:, 0]]
        s2 = S[np.arange(n), order[:, 1]]
        g1 = np.array([spk_logit[ks[i]] for i in order[:, 0]])
        g2 = np.array([spk_logit[ks[i]] for i in order[:, 1]])
        opposite = (np.sign(g1) != np.sign(g2)) & (np.abs(g1) > 0.5) & (np.abs(g2) > 0.5)
        cand = opposite & (s2 >= p.sim_min) & (s1 - s2 <= p.margin)
    if tagger is not None and len(tagger) == n:
        cand |= (tagger[:, 0] >= p.tagger_min) & (tagger[:, 1] >= p.tagger_min)
    return cand


def regions_from_windows(windows: Sequence[Any], cand: np.ndarray, p: OverlapParams) -> list[tuple[float, float]]:
    """Merge runs of candidate windows (consecutive in time, same event) to regions."""
    regs: list[list[float]] = []
    run: list[Any] = []

    def close():
        if len(run) >= p.min_windows:
            s, e = run[0].start, run[-1].end
            if e - s >= p.min_dur:
                # trim to the part covered by >= 2 windows when possible
                if len(run) >= 3:
                    s, e = run[1].start, run[-2].end
                regs.append([s, e])

    prev = None
    for w, c in zip(windows, cand):
        if c and prev is not None and prev.event == w.event and run:
            run.append(w)
        elif c:
            close()
            run = [w]
        else:
            close()
            run = []
        prev = w
    close()
    merged: list[list[float]] = []
    for s, e in sorted(regs):
        if merged and s <= merged[-1][1] + 0.1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# 2-speaker separation (Best preset)
# ---------------------------------------------------------------------------
class SpeechSeparator:
    """torchaudio ConvTasNet (Libri2Mix, 8 kHz) by default; spec from presets.MODELS."""

    def __init__(self, spec: dict, device: str = "cpu"):
        self.spec, self.device = spec, device
        self.model = None
        self.sr = int(spec.get("sample_rate", 8000))

    def present(self) -> bool:
        if self.spec.get("type") == "torchaudio":
            return modelstore.torchaudio_present(self.spec["bundle"])
        return False

    def download(self) -> None:
        if self.spec.get("type") == "torchaudio":
            modelstore.torchaudio_bundle(self.spec["bundle"]).get_model()

    def load(self) -> None:
        if self.spec.get("type") != "torchaudio":
            raise ValueError(f"unsupported speech separation model {self.spec}")
        b = modelstore.torchaudio_bundle(self.spec["bundle"])
        self.sr = int(getattr(b, "sample_rate", self.sr))
        self.model = b.get_model().eval().to(self.device)

    def separate(self, x16: np.ndarray) -> np.ndarray:
        """x16 mono 16 kHz -> (2, n) sources at 16 kHz."""
        import soxr
        import torch

        x = soxr.resample(x16.astype(np.float32), features.SR16, self.sr) if self.sr != features.SR16 else x16
        sd = float(np.std(x)) + 1e-8
        with torch.inference_mode():
            t = torch.from_numpy((x / sd).astype(np.float32))[None, None].to(self.device)
            y = self.model(t)[0].float().cpu().numpy()  # (2, n)
        y = y * sd
        if self.sr != features.SR16:
            y = np.stack([soxr.resample(s, self.sr, features.SR16) for s in y])
        n = len(x16)
        y = y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))
        return y

    def unload(self) -> None:
        self.model = None
        features.free_torch()


def ls_gains(mix: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Least-squares scale of each source so that sum(g_i s_i) ~ mix."""
    A = sources.T.astype(np.float64)
    g, *_ = np.linalg.lstsq(A, mix.astype(np.float64), rcond=None)
    return g


def female_mask_replacement(vocals44: np.ndarray, src_f16: np.ndarray, src_m16: np.ndarray,
                            sr44: int = 44100, band_hz: float = 3800.0) -> np.ndarray:
    """Apply a soft female mask (from 16 kHz separated sources) to the full-band stem.

    vocals44: (n, ch) float32 -> (n, ch) female-only estimate.
    """
    import soxr
    from scipy.signal import istft, stft

    n = len(vocals44)
    up = [soxr.resample(s.astype(np.float32), features.SR16, sr44) for s in (src_f16, src_m16)]
    up = [u[:n] if len(u) >= n else np.pad(u, (0, n - len(u))) for u in up]
    nper, hop = 2048, 512
    _, _, Sf = stft(up[0], sr44, nperseg=nper, noverlap=nper - hop)
    _, _, Sm = stft(up[1], sr44, nperseg=nper, noverlap=nper - hop)
    pf, pm = np.abs(Sf) ** 2, np.abs(Sm) ** 2
    M = pf / (pf + pm + 1e-10)
    freqs = np.fft.rfftfreq(nper, 1 / sr44)
    band = (freqs >= 1000) & (freqs <= band_hz)
    hi = freqs > band_hz
    w = (pf + pm)[band]
    frame_share = (M[band] * w).sum(0) / (w.sum(0) + 1e-10)
    M[hi] = frame_share[None, :]
    # light temporal smoothing against musical noise
    from scipy.ndimage import uniform_filter1d

    M = uniform_filter1d(M, size=3, axis=1)
    out = np.zeros_like(vocals44, dtype=np.float32)
    for c in range(vocals44.shape[1]):
        _, _, V = stft(vocals44[:, c], sr44, nperseg=nper, noverlap=nper - hop)
        _, y = istft(V * M[:, :V.shape[1]], sr44, nperseg=nper, noverlap=nper - hop)
        out[:, c] = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return out
