"""Vocal activity: find vocal events on the (separated) vocal stem.

Two detectors are combined (union):

1. **Energy** - frame RMS (25 ms window, 10 ms hop) in dB compared with an adaptive noise
   floor (running low percentile over ~8 s, clamped by a global floor).  Hysteresis: an event
   starts above ``floor + on_db`` and continues while above ``floor + off_db``.  This catches
   non-verbal sounds (moans, breaths, grunts) that speech VADs ignore.
2. **Silero VAD** (MIT, bundled weights) - speech probability per 32 ms.  It is blind to
   screams and most non-verbal sounds (docs/MODELS.md), so by default it only *labels*
   events (``vad_ratio`` -> speech vs non-verbal kind) and never decides activity;
   ``vad_gates=True`` adds VAD-only regions as events.

Post-processing: gaps < ``merge_gap`` are merged, events shorter than ``min_event`` are only
kept if they are clearly loud (else dropped as clicks), every event is padded by ``pad``.
Quiet events are never dropped for being quiet: they are flagged ``quiet`` so the
classifier can mark them low-confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

HOP_S = 0.010
WIN_S = 0.025


@dataclass
class ActivityParams:
    on_db: float = 9.0           # above adaptive floor to start an event
    off_db: float = 5.0          # above adaptive floor to continue an event
    abs_min_db: float = -62.0    # never trigger below this level (dBFS, RMS)
    rel_max_db: float = -45.0    # ...or more than 45 dB below the loudest frames
    floor_window_s: float = 8.0
    floor_percentile: float = 10.0
    vad_threshold: float = 0.5
    vad_neg_threshold: float = 0.35
    min_event: float = 0.08
    merge_gap: float = 0.15
    pad: float = 0.045
    short_keep_db: float = 18.0  # short events kept if this far above the floor
    use_vad: bool = True
    vad_gates: bool = False      # silero misses screams/coughs: it only labels speech by default


@dataclass
class Event:
    start: float
    end: float
    peak_db: float = -100.0
    mean_db: float = -100.0
    snr_db: float = 0.0
    vad_ratio: float = 0.0
    quiet: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return self.end - self.start


def frame_db(x: np.ndarray, sr: int, hop_s: float = HOP_S, win_s: float = WIN_S,
             block_frames: int = 200000) -> np.ndarray:
    """RMS level per frame in dBFS (frames centred at k*hop), computed block-wise so that
    multi-hour stems need little extra memory."""
    hop = int(round(hop_s * sr))
    win = int(round(win_s * sr))
    n = max(1, int(np.ceil(len(x) / hop)))
    pad = win // 2
    out = np.empty(n, np.float32)
    for f0 in range(0, n, block_frames):
        f1 = min(n, f0 + block_frames)
        a = f0 * hop - pad
        b = (f1 - 1) * hop - pad + win
        seg = x[max(0, a):max(0, min(len(x), b))].astype(np.float64)
        seg = np.pad(seg, (max(0, -a), max(0, b - max(a, len(x)) if b > len(x) else 0)))
        if len(seg) < b - a:
            seg = np.pad(seg, (0, b - a - len(seg)))
        c = np.concatenate([[0.0], np.cumsum(seg ** 2)])
        idx = (np.arange(f0, f1) - f0) * hop
        e = (c[idx + win] - c[idx]) / win
        out[f0:f1] = 10.0 * np.log10(np.maximum(e, 1e-12))
    return out


def adaptive_floor(db: np.ndarray, p: ActivityParams) -> np.ndarray:
    """Running low percentile of the frame level (computed on a decimated grid)."""
    from scipy.ndimage import percentile_filter

    dec = 10  # 100 ms grid
    n = len(db)
    m = int(np.ceil(n / dec))
    pad = m * dec - n
    d = np.pad(db, (0, pad), mode="edge").reshape(m, dec).min(axis=1)
    w = max(3, int(p.floor_window_s / (HOP_S * dec)) | 1)
    fl = percentile_filter(d, p.floor_percentile, size=w, mode="nearest")
    # smooth and upsample
    fl = np.repeat(fl, dec)[:n]
    glob = np.percentile(db, 5) if n else -100.0
    # the floor never drops far below the global floor and never rises above a sane level
    fl = np.clip(fl, glob - 3.0, max(glob + 25.0, -30.0))
    return fl.astype(np.float32)


def silero_probs(x16: np.ndarray, progress: Optional[Callable[[float], None]] = None,
                 device: str = "cpu") -> Optional[np.ndarray]:
    """Speech probability per 512-sample (32 ms) frame at 16 kHz, or None if unavailable."""
    try:
        import torch
        from silero_vad import load_silero_vad
    except Exception:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_silero_vad()
    model.eval()
    n = len(x16) // 512
    out = np.zeros(n, dtype=np.float32)
    chunk = 512 * 1875  # 60 s
    with torch.inference_mode():
        model.reset_states()
        done = 0
        for s in range(0, n * 512, chunk):
            seg = torch.from_numpy(np.ascontiguousarray(x16[s:min(s + chunk, n * 512)], dtype=np.float32))
            k = len(seg) // 512
            if k == 0:
                break
            try:
                pr = model.audio_forward(seg[: k * 512], 16000)  # (1, k) keeps state across chunks? no
                pr = pr.reshape(-1).cpu().numpy()[:k]
            except Exception:
                pr = np.array([float(model(seg[i * 512:(i + 1) * 512], 16000)) for i in range(k)])
            out[done:done + len(pr)] = pr
            done += len(pr)
            if progress:
                progress(done / max(1, n))
    del model
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not len(mask):
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _hysteresis(level: np.ndarray, on: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Frames active by hysteresis: runs above `off` that contain a frame above `on`."""
    above_off = level > off
    above_on = level > on
    active = np.zeros(len(level), dtype=bool)
    for a, b in _runs(above_off):
        if above_on[a:b].any():
            active[a:b] = True
    return active


def detect_events(x16: np.ndarray, sr: int = 16000, params: Optional[ActivityParams] = None,
                  vad: Optional[np.ndarray] = None,
                  progress: Optional[Callable[[float], None]] = None) -> tuple[list[Event], dict]:
    """Return (events, frame_info) for a mono vocal stem.

    ``vad`` (optional) are silero probabilities per 32 ms frame; computed here if None and
    params.use_vad.
    """
    p = params or ActivityParams()
    db = frame_db(x16, sr)
    n = len(db)
    if n == 0:
        return [], {"db": db}
    floor = adaptive_floor(db, p)
    loud_ref = float(np.percentile(db, 99.5))
    trig_min = max(p.abs_min_db, loud_ref + p.rel_max_db)
    on = np.maximum(floor + p.on_db, trig_min)
    off = np.maximum(floor + p.off_db, trig_min - 4.0)
    act = _hysteresis(db, on, off)

    vad_frames = None
    if vad is None and p.use_vad:
        vad = silero_probs(x16, progress)
    if vad is not None and len(vad):
        # map 32 ms VAD frames onto the 10 ms grid
        t = np.arange(n) * HOP_S
        vi = np.minimum((t / 0.032).astype(int), len(vad) - 1)
        vad_frames = vad[vi]
        vact = _hysteresis(vad_frames, np.full(n, p.vad_threshold), np.full(n, p.vad_neg_threshold))
        # VAD alone must still have *some* energy (avoid hallucinated speech in silence)
        vact &= db > (trig_min - 6.0)
        if p.vad_gates:
            act |= vact

    runs = _runs(act)
    # merge small gaps
    gap = int(round(p.merge_gap / HOP_S))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    events: list[Event] = []
    min_len = int(round(p.min_event / HOP_S))
    padf = p.pad
    dur = len(x16) / sr
    for a, b in merged:
        seg_db = db[a:b]
        peak = float(seg_db.max())
        fl = float(np.median(floor[a:b]))
        snr = peak - fl
        if b - a < min_len and snr < p.short_keep_db:
            continue
        lin = 10 ** (seg_db / 10)
        mean_db = float(10 * np.log10(lin.mean() + 1e-12))
        vr = float((vad_frames[a:b] > p.vad_threshold).mean()) if vad_frames is not None else 0.0
        s = max(0.0, a * HOP_S - padf)
        e = min(dur, b * HOP_S + padf)
        if (e - s) < p.min_event:  # loud click: extend to min length
            c = (s + e) / 2
            s, e = max(0.0, c - p.min_event / 2), min(dur, c + p.min_event / 2)
        events.append(Event(s, e, peak, mean_db, snr, vr, quiet=snr < p.on_db + 6))
    # padding can create overlaps -> merge
    out: list[Event] = []
    for ev in events:
        if out and ev.start <= out[-1].end:
            o = out[-1]
            w1, w2 = o.dur, ev.dur
            o.vad_ratio = (o.vad_ratio * w1 + ev.vad_ratio * w2) / max(1e-9, w1 + w2)
            o.end = max(o.end, ev.end)
            o.peak_db = max(o.peak_db, ev.peak_db)
            o.mean_db = max(o.mean_db, ev.mean_db)
            o.snr_db = max(o.snr_db, ev.snr_db)
            o.quiet = o.quiet and ev.quiet
        else:
            out.append(ev)
    info = {"db": db, "floor": floor, "vad": vad_frames, "trig_min": trig_min}
    return out, info
