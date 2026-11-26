"""Metric code for the Voice Filter evaluation (pure numpy; tested in tests/test_eval_metrics.py).

Conventions
- 10 ms frames. Truth vocal activity comes from the `active` sub-intervals of each truth
  event (falls back to [start, end]).
- Frame classes: female-only, male-only, overlap (both), none. Keep/remove accuracy is
  computed on single-gender frames; overlap frames are reported separately.
- The pipeline decision for a frame is the effective `keep` of the analysis segment
  covering it (segment override > speaker override > model keep). Frames not covered by
  any segment get `outside` (default "remove": the vocal stem is gated off there). Where
  segments overlap, a frame counts as kept if any covering segment keeps it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

HOP = 0.01
EVENT_CORRECT_FRACTION = 0.8


def n_frames(duration: float, hop: float = HOP) -> int:
    return int(np.ceil(round(duration / hop, 6)))


def _fill(mask: np.ndarray, a: float, b: float, hop: float = HOP) -> None:
    i, j = int(round(a / hop)), int(round(b / hop))
    mask[max(0, i):max(0, min(len(mask), j))] = True


def event_frames(ev: dict, nf: int, hop: float = HOP) -> np.ndarray:
    m = np.zeros(nf, bool)
    for a, b in ev.get("active") or [[ev["start"], ev["end"]]]:
        _fill(m, a, b, hop)
    return m


def truth_frames(truth: dict, hop: float = HOP) -> tuple[np.ndarray, np.ndarray]:
    """Boolean female / male activity per frame."""
    nf = n_frames(truth["duration"], hop)
    f = np.zeros(nf, bool)
    m = np.zeros(nf, bool)
    for ev in truth["events"]:
        (f if ev["gender"] == "F" else m)[event_frames(ev, nf, hop)] = True
    return f, m


def effective_keep(seg: dict, speakers: dict | None = None) -> bool:
    ov = seg.get("override")
    if ov in ("keep", "remove"):
        return ov == "keep"
    spk = (speakers or {}).get(seg.get("speaker"))
    if spk is not None and spk.get("override") in ("keep", "remove"):
        return spk["override"] == "keep"
    return bool(seg.get("keep"))


def decision_frames(analysis: dict, nf: int, outside: str = "remove", hop: float = HOP):
    """Per-frame keep, low-confidence and covered masks from an analysis dict."""
    kept = np.zeros(nf, bool)
    low = np.zeros(nf, bool)
    cov = np.zeros(nf, bool)
    speakers = {s.get("id"): s for s in analysis.get("speakers", []) or []}
    for seg in analysis.get("segments", []):
        m = np.zeros(nf, bool)
        _fill(m, float(seg["start"]), float(seg["end"]), hop)
        if effective_keep(seg, speakers):
            kept |= m
        if seg.get("low_confidence", False):
            low |= m
        cov |= m
    # Where segments overlap, "keep" wins (a frame is audible if any covering segment keeps it).
    keep = kept | (~cov & (outside == "keep"))
    return keep, low, cov


def _ratio(num, den):
    return float(num) / float(den) if den else None


def frame_metrics(f: np.ndarray, m: np.ndarray, keep: np.ndarray, low: np.ndarray | None = None,
                  cov: np.ndarray | None = None) -> dict:
    low = np.zeros_like(keep) if low is None else low
    cov = np.ones_like(keep) if cov is None else cov
    fo, mo, ov = f & ~m, m & ~f, f & m
    single = fo | mo
    correct = (fo & keep) | (mo & ~keep)
    vocal = f | m
    out = {
        "frames_female_only": int(fo.sum()), "frames_male_only": int(mo.sum()), "frames_overlap": int(ov.sum()),
        "accuracy": _ratio((correct & single).sum(), single.sum()),
        "male_leak": _ratio((mo & keep).sum(), mo.sum()),
        "female_loss": _ratio((fo & ~keep).sum(), fo.sum()),
        "overlap_kept": _ratio((ov & keep).sum(), ov.sum()),
        "low_conf_share": _ratio((vocal & low).sum(), vocal.sum()),
        "accuracy_excl_low_conf": _ratio((correct & single & ~low).sum(), (single & ~low).sum()),
        "accuracy_low_conf_only": _ratio((correct & single & low).sum(), (single & low).sum()),
        "coverage": _ratio((vocal & cov).sum(), vocal.sum()),
    }
    bal = [x for x in (None if out["male_leak"] is None else 1 - out["male_leak"],
                       None if out["female_loss"] is None else 1 - out["female_loss"]) if x is not None]
    out["balanced_accuracy"] = float(np.mean(bal)) if bal else None
    return out


def event_results(truth: dict, keep: np.ndarray, hop: float = HOP) -> list[dict]:
    """Per truth event: share of its frames with the right decision; correct if >= 80 %."""
    nf = len(keep)
    res = []
    for ev in truth["events"]:
        fr = event_frames(ev, nf, hop)
        if not fr.any():
            continue
        want = ev["gender"] == "F"
        share = float((keep[fr] == want).mean())
        res.append({"gender": ev["gender"], "kind": ev["kind"], "category": ev["category"],
                    "overlap": bool(ev.get("overlap")), "short": bool(ev.get("short")),
                    "breathy": bool(ev.get("breathy")), "share_right": share,
                    "correct": share >= EVENT_CORRECT_FRACTION, "duration": float(fr.sum() * hop),
                    "wrong_s": float((keep[fr] != want).sum() * hop)})
    return res


def summarize_events(evres: list[dict]) -> dict:
    groups = defaultdict(list)
    for e in evres:
        keys = ["all", f"kind={e['kind']}", f"gender={e['gender']}", f"kind={e['kind']},gender={e['gender']}",
                f"category={e['category']}", f"overlap={e['overlap']}"]
        if e["short"]:
            keys.append("short(<0.4s)")
        if e["breathy"]:
            keys.append("breathy")
        for k in keys:
            groups[k].append(e["correct"])
    return {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(groups.items())}


# ----------------------------------------------------------------------------- audio level
def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[None, :] if x.ndim == 1 else (x if x.shape[0] <= 8 else x.T)


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = _as2d(est).ravel(), _as2d(ref).ravel()
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    tgt = alpha * ref
    return float(10 * np.log10((tgt ** 2).sum() / (((est - tgt) ** 2).sum() + 1e-12) + 1e-12))


def _runs(mask: np.ndarray, min_len: int = 20):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return [(a, b) for a, b in zip(idx[::2], idx[1::2]) if b - a >= min_len]


def projected_energy_db(out: np.ndarray, src: np.ndarray, mask: np.ndarray, sr: int, hop: float = HOP):
    """Energy of `src` present in `out` inside `mask` regions, relative to src's own energy (dB).

    Per contiguous region (>= 0.2 s) the least-squares gain g = <out,src>/<src,src> is
    estimated; 0 dB = fully present, very negative = removed. Independent background in
    `out` does not bias g (zero correlation in expectation).
    """
    out, src = _as2d(out), _as2d(src)
    n = min(out.shape[1], src.shape[1])
    if out.shape[0] != src.shape[0]:
        out = np.repeat(out.mean(0, keepdims=True), src.shape[0], 0)
    num = den = 0.0
    step = int(round(hop * sr))
    for a, b in _runs(mask):
        s0, s1 = a * step, min(n, b * step)
        if s1 <= s0:
            continue
        o, s = out[:, s0:s1], src[:, s0:s1]
        e = float((s ** 2).sum())
        if e <= 1e-12:
            continue
        g = float((o * s).sum()) / e
        num += g * g * e
        den += e
    return float(10 * np.log10(num / den + 1e-12)) if den > 0 else None


def audio_metrics(out: np.ndarray, female: np.ndarray, male: np.ndarray, background: np.ndarray,
                  f_act: np.ndarray, m_act: np.ndarray, sr: int, mix: np.ndarray | None = None) -> dict:
    out, female, male, background = map(_as2d, (out, female, male, background))
    n = min(out.shape[1], female.shape[1])
    out, female, male, background = out[:, :n], female[:, :n], male[:, :n], background[:, :n]
    if out.shape[0] != female.shape[0]:
        out = np.repeat(out.mean(0, keepdims=True), female.shape[0], 0)
    fo, mo = f_act & ~m_act, m_act & ~f_act
    target = female + background
    res = {
        "male_residual_db": projected_energy_db(out, male, mo, sr),
        "female_preserved_db": projected_energy_db(out, female, fo, sr),
        "male_residual_db_overlap": projected_energy_db(out, male, f_act & m_act, sr),
        "female_preserved_db_overlap": projected_energy_db(out, female, f_act & m_act, sr),
        "si_sdr_vs_female_plus_bg": si_sdr(out, target),
        "length_diff_samples": None,
    }
    if mix is not None:
        res["si_sdr_mix_vs_female_plus_bg"] = si_sdr(_as2d(mix)[:, :n], target)
        res["si_sdr_improvement"] = res["si_sdr_vs_female_plus_bg"] - res["si_sdr_mix_vs_female_plus_bg"]
    return res


# ----------------------------------------------------------------------------- stub renderer
def gain_envelope(keep: np.ndarray, n_samples: int, sr: int, hop: float = HOP, fade: float = 0.015) -> np.ndarray:
    """Sample-level 0/1 gain from frame decisions with raised-cosine transitions."""
    step = int(round(hop * sr))
    g = np.repeat(keep.astype(np.float64), step)[:n_samples]
    if len(g) < n_samples:
        g = np.pad(g, (0, n_samples - len(g)), constant_values=float(keep[-1]) if len(keep) else 0.0)
    L = max(1, int(fade * sr))
    k = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, L))
    k /= k.sum()
    return np.clip(np.convolve(g, k, mode="same"), 0.0, 1.0)


def aggregate(per_mix: list[dict]) -> dict:
    """Micro-average frame metrics (frame-count weighted) and event accuracies across mixes."""
    def w_avg(key, wkey):
        num = den = 0.0
        for r in per_mix:
            v, w = r["frames"].get(key), r["frames"].get(wkey, 0)
            if v is not None and w:
                num += v * w
                den += w
        return num / den if den else None

    tot = defaultdict(int)
    for r in per_mix:
        for k in ("frames_female_only", "frames_male_only", "frames_overlap"):
            tot[k] += r["frames"][k]
    fr = dict(tot)
    fr["male_leak"] = w_avg("male_leak", "frames_male_only")
    fr["female_loss"] = w_avg("female_loss", "frames_female_only")
    fr["overlap_kept"] = w_avg("overlap_kept", "frames_overlap")
    single = tot["frames_female_only"] + tot["frames_male_only"]
    if single:
        fr["accuracy"] = ((1 - (fr["female_loss"] or 0)) * tot["frames_female_only"]
                          + (1 - (fr["male_leak"] or 0)) * tot["frames_male_only"]) / single
    bal = [1 - x for x in (fr["male_leak"], fr["female_loss"]) if x is not None]
    fr["balanced_accuracy"] = float(np.mean(bal)) if bal else None
    for key in ("low_conf_share", "accuracy_excl_low_conf", "coverage"):
        vals = [r["frames"][key] for r in per_mix if r["frames"].get(key) is not None]
        fr[key + "_mean_over_mixes"] = float(np.mean(vals)) if vals else None
    ev = defaultdict(list)
    for r in per_mix:
        for e in r["event_list"]:
            ev["all"].append(e["correct"])
            ev[f"kind={e['kind']}"].append(e["correct"])
            ev[f"kind={e['kind']},gender={e['gender']}"].append(e["correct"])
            ev[f"category={e['category']}"].append(e["correct"])
            ev[f"overlap={e['overlap']}"].append(e["correct"])
            if e["short"]:
                ev["short(<0.4s)"].append(e["correct"])
            if e["breathy"]:
                ev["breathy"].append(e["correct"])
    events = {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(ev.items())}
    # error breakdown: female time lost / male time leaked, by event attribute
    err = defaultdict(lambda: {"n": 0, "n_wrong": 0, "time_s": 0.0, "wrong_s": 0.0})
    for r in per_mix:
        for e in r["event_list"]:
            keys = ["all", f"kind={e['kind']}", f"category={e['category']}",
                    "overlap" if e["overlap"] else "no-overlap"]
            if e["short"]:
                keys.append("short(<0.4s)")
            if e["breathy"]:
                keys.append("breathy/whisper")
            for k in keys:
                c = err[(e["gender"], k)]
                c["n"] += 1
                c["n_wrong"] += int(not e["correct"])
                c["time_s"] += e["duration"]
                c["wrong_s"] += e.get("wrong_s", 0.0)
    errors = {f"{g}|{k}": v for (g, k), v in sorted(err.items())}
    audio = {}
    for key in ("male_residual_db", "female_preserved_db", "si_sdr_vs_female_plus_bg", "si_sdr_improvement",
                "male_residual_db_overlap"):
        vals = [r["audio"][key] for r in per_mix if r.get("audio") and r["audio"].get(key) is not None]
        audio[key + "_mean"] = float(np.mean(vals)) if vals else None
    dur = sum(r["duration"] for r in per_mix)
    t = sum(r.get("processing_s", 0.0) for r in per_mix)
    return {"n_mixes": len(per_mix), "total_duration_s": dur, "frames": fr, "events": events, "errors": errors,
            "audio": audio,
            "processing_s": t, "rtf": t / dur if dur else None}
