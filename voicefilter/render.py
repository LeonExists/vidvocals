"""Render the filtered audio: gain envelope on the vocal stem + background.

The gain envelope is built analytically from the list of gain changes, so it is exact at
sample level and cheap for multi-hour files::

    g(t) = g0 + sum_i  delta_i * S((t - t_i) / F + 1/2)

where ``t_i`` are the boundaries, ``delta_i`` the gain change there, ``F`` the fade length and
``S(u) = (1 - cos(pi * clip(u, 0, 1))) / 2`` a raised-cosine step (C1-continuous).  Ramps of
neighbouring boundaries simply add up, so segments shorter than the fade are handled
gracefully (the envelope never jumps).

Everything is streamed block-wise from/to disk (soundfile), and the output has exactly the
length of the vocal stem.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from . import config

BLOCK = 1 << 18  # samples per processing block (~6 s at 44.1 kHz)


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------
def effective_keep(seg: dict, speaker_overrides: dict[str, Optional[str]] | None = None,
                   low_conf_default: str = "model") -> bool:
    """segment override > speaker override > model decision."""
    ov = seg.get("override")
    if ov in ("keep", "remove"):
        return ov == "keep"
    if speaker_overrides:
        so = speaker_overrides.get(str(seg.get("speaker")))
        if so in ("keep", "remove"):
            return so == "keep"
    if seg.get("low_confidence") and low_conf_default in ("keep", "remove"):
        return low_conf_default == "keep"
    return bool(seg.get("keep", True))


def decisions_from_analysis(analysis: dict) -> list[tuple[float, float, float]]:
    """[(start, end, gain)] for all segments, overrides applied."""
    spk_ov = {str(s["id"]): s.get("override") for s in analysis.get("speakers", [])}
    lcd = analysis.get("low_conf_default") or config.setting("low_conf_default")
    out = []
    for seg in analysis.get("segments", []):
        keep = effective_keep(seg, spk_ov, lcd)
        out.append((float(seg["start"]), float(seg["end"]), 1.0 if keep else 0.0))
    return out


# ---------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------
def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


class GainEnvelope:
    """Piecewise-constant gain with raised-cosine transitions, evaluated block-wise."""

    def __init__(self, intervals: Sequence[tuple[float, float, float]], sr: int,
                 fade_s: float, default_gain: float = 1.0):
        self.sr = sr
        self.fade = max(1.0, fade_s * sr)  # in samples
        # sorted, non-overlapping step function (an interval starting inside the previous
        # one is clipped to start where the previous ends)
        iv: list[tuple[int, int, float]] = []
        for s_, e_, g_ in intervals:
            a, b = int(round(float(s_) * sr)), int(round(float(e_) * sr))
            if b > a:
                iv.append((a, b, float(g_)))
        iv.sort()
        changes: list[tuple[float, float]] = []
        cur = default_gain
        last_end = -1
        for a, b, g in iv:
            a = max(a, last_end)
            if b <= a:
                continue
            if a > last_end and last_end >= 0 and cur != default_gain:
                changes.append((float(last_end), default_gain - cur))  # gap back to default
                cur = default_gain
            if g != cur:
                changes.append((float(a), g - cur))
                cur = g
            last_end = b
        if last_end >= 0 and cur != default_gain:
            changes.append((float(last_end), default_gain - cur))
        self.g0 = default_gain
        self.pos = np.array([c[0] for c in changes], dtype=np.float64)
        self.delta = np.array([c[1] for c in changes], dtype=np.float64)
        self.cum = np.concatenate([[0.0], np.cumsum(self.delta)])

    def block(self, start: int, n: int) -> np.ndarray:
        half = self.fade / 2
        t = np.arange(start, start + n, dtype=np.float64)
        # boundaries fully before the block contribute their whole delta
        i0 = int(np.searchsorted(self.pos, start - half, side="right"))
        i1 = int(np.searchsorted(self.pos, start + n + half, side="left"))
        g = np.full(n, self.g0 + self.cum[i0], dtype=np.float64)
        for i in range(i0, i1):
            u = (t - self.pos[i]) / self.fade + 0.5
            g += self.delta[i] * _smoothstep(u)
        return g.astype(np.float32)


# ---------------------------------------------------------------------------
# limiter
# ---------------------------------------------------------------------------
LIM_FRAME = 64


def _limiter_gains(frame_peaks: np.ndarray, ceiling: float = 0.98, look: int = 48,
                   avg: int = 32) -> np.ndarray:
    """Per-frame gains so that frame_peak*gain <= ceiling, smooth (min-filter + moving average).

    Moving average width (2*avg+1) <= min-filter width (2*look+1) guarantees the averaged gain
    never exceeds the required gain inside the window.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    req = np.minimum(1.0, ceiling / np.maximum(frame_peaks, 1e-9))
    m = minimum_filter1d(req, size=2 * look + 1, mode="nearest")
    return uniform_filter1d(m, size=2 * avg + 1, mode="nearest")


# ---------------------------------------------------------------------------
# main render
# ---------------------------------------------------------------------------
def render(vocals_wav: str | Path, background_wav: Optional[str | Path], out_wav: str | Path,
           intervals: Sequence[tuple[float, float, float]], *,
           fade_ms: Optional[float] = None, gate_unlabeled: Optional[str] = None,
           keep_background: bool = True,
           replacements: Iterable[dict] = (),
           progress: Optional[Callable[[float], None]] = None) -> str:
    """Write ``gain * vocals(+replacements) + background`` to ``out_wav`` (float32).

    intervals: [(start_s, end_s, gain)] for every labelled segment.
    replacements: [{start, end, wav}] regions where the vocal stem is replaced (crossfaded)
      by another signal (e.g. the female source of an overlap separation).  ``wav`` is a
      file at the stem sample rate whose first sample corresponds to ``start``.
    """
    import soundfile as sf

    fade_s = (config.setting("fade_ms") if fade_ms is None else fade_ms) / 1000.0
    gate = gate_unlabeled or config.setting("gate_unlabeled")
    default_gain = 0.0 if gate == "mute" else 1.0
    out_wav = Path(out_wav)
    tmp = out_wav.with_name(out_wav.stem + ".part.wav")

    with sf.SoundFile(str(vocals_wav)) as fv:
        sr, ch, n_total = fv.samplerate, fv.channels, fv.frames
        env = GainEnvelope(intervals, sr, fade_s, default_gain)
        reps = []
        for r in replacements:
            try:
                rf = sf.SoundFile(str(r["wav"]))
            except Exception:
                continue
            a = int(round(float(r["start"]) * sr))
            b = min(n_total, a + rf.frames, int(round(float(r["end"]) * sr)))
            if b <= a:
                rf.close()
                continue
            xf = max(fade_s, 0.03) * sr  # crossfade fully inside the replaced region
            xf = min(xf, (b - a) / 2)
            renv = GainEnvelope([((a + xf / 2) / sr, (b - xf / 2) / sr, 1.0)], sr,
                                max(1.0, xf) / sr, 0.0)
            reps.append((a, b, rf, renv))
        fb = sf.SoundFile(str(background_wav)) if (background_wav and keep_background) else None
        if fb is not None and (fb.samplerate != sr or fb.frames != n_total):
            fb.close()
            raise ValueError("background and vocal stems differ in rate or length")
        peaks: list[np.ndarray] = []
        try:
            with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=ch, subtype="FLOAT",
                              format="RF64" if n_total * ch * 4 > 2**31 else "WAV") as fo:
                pos = 0
                while pos < n_total:
                    n = min(BLOCK, n_total - pos)
                    v = fv.read(n, dtype="float32", always_2d=True)
                    n = len(v)
                    if n == 0:
                        break
                    for a, b, rf, renv in reps:
                        lo, hi = max(pos, a), min(pos + n, b)
                        if hi <= lo:
                            continue
                        w = renv.block(lo, hi - lo)[:, None]
                        rf.seek(lo - a)
                        rr = np.zeros((hi - lo, ch), np.float32)
                        got = rf.read(hi - lo, dtype="float32", always_2d=True)
                        if got.shape[1] != ch:
                            got = np.repeat(got[:, :1], ch, axis=1)
                        rr[:len(got)] = got
                        seg = v[lo - pos:hi - pos]
                        v[lo - pos:hi - pos] = seg * (1 - w) + rr * w
                    g = env.block(pos, n)
                    y = v * g[:, None]
                    if fb is not None:
                        y = y + fb.read(n, dtype="float32", always_2d=True)
                    fo.write(y)
                    m = np.abs(y).max(axis=1)
                    k = len(m) // LIM_FRAME * LIM_FRAME
                    fp = m[:k].reshape(-1, LIM_FRAME).max(axis=1) if k else np.zeros(0, np.float32)
                    if k < len(m):
                        fp = np.concatenate([fp, [m[k:].max()]])
                    peaks.append(fp)
                    pos += n
                    if progress:
                        progress(0.85 * pos / max(1, n_total))
        finally:
            if fb is not None:
                fb.close()
            for _, _, rf, _ in reps:
                rf.close()

    fpk = np.concatenate(peaks) if peaks else np.zeros(0)
    if len(fpk) and fpk.max() > 1.0:
        _apply_limiter(tmp, out_wav, fpk, progress)
        try:
            os.remove(tmp)
        except OSError:
            pass
    else:
        os.replace(tmp, out_wav)
    if progress:
        progress(1.0)
    return str(out_wav)


def _apply_limiter(src: Path, dst: Path, frame_peaks: np.ndarray,
                   progress: Optional[Callable[[float], None]] = None) -> None:
    """Gentle look-ahead peak limiter (only used when the sum exceeds 0 dBFS)."""
    import soundfile as sf

    g_frames = _limiter_gains(frame_peaks.astype(np.float64))
    centers = (np.arange(len(g_frames)) + 0.5) * LIM_FRAME
    with sf.SoundFile(str(src)) as fi:
        sr, ch, n_total = fi.samplerate, fi.channels, fi.frames
        part = dst.with_name(dst.stem + ".lim.wav")
        with sf.SoundFile(str(part), "w", samplerate=sr, channels=ch, subtype="FLOAT",
                          format="RF64" if n_total * ch * 4 > 2**31 else "WAV") as fo:
            pos = 0
            while pos < n_total:
                x = fi.read(BLOCK, dtype="float32", always_2d=True)
                n = len(x)
                if n == 0:
                    break
                t = np.arange(pos, pos + n) + 0.5
                g = np.interp(t, centers, g_frames).astype(np.float32)
                y = x * g[:, None]
                np.clip(y, -1.0, 1.0, out=y)  # safety net, should be a no-op
                fo.write(y)
                pos += n
                if progress:
                    progress(0.85 + 0.15 * pos / max(1, n_total))
    os.replace(part, dst)
