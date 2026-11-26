"""Orchestration: analyze_file -> analysis dict, render_from_analysis -> filtered wav.

Work-dir layout (``work_dir``)::

    audio.wav          44.1 kHz stereo float mix (extracted)
    vocals.wav         vocal stem            background.wav  mix - vocals
    analysis.json      contract of PLAN section 5 (+ a few extra informational keys)
    debug.json         per-window evidence (not used by the UI)
    peaks.json         {rate, peaks} of the vocal stem
    replacements.json  overlap replacements [{start, end, wav, segment}]
    original.m4a / vocals.m4a / preview.m4a   review previews
    filtered.wav       rendered result
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import activity, config, features, gender, media, overlap, presets, render, separation

ProgressCB = Optional[Callable[[str, float, str], None]]

# share of the total analysis progress per stage (sums to 1)
STAGE_WEIGHTS = {"extract": 0.04, "separate": 0.55, "activity": 0.05, "features": 0.25,
                 "speakers": 0.03, "overlap": 0.04, "preview": 0.04}


class Progress:
    def __init__(self, cb: ProgressCB, weights: dict[str, float]):
        self.cb, self.w = cb, weights
        self.done = 0.0
        self.stage = ""
        self.t_stage = time.time()
        self.timings: dict[str, float] = {}

    def start(self, stage: str, message: str = "") -> None:
        if self.stage:
            self.finish()
        self.stage = stage
        self.t_stage = time.time()
        self.update(0.0, message or stage)

    def update(self, frac: float, message: str = "") -> None:
        if self.cb:
            f = self.done + self.w.get(self.stage, 0.0) * max(0.0, min(1.0, frac))
            try:
                self.cb(self.stage, min(0.999, f), message)
            except Exception:
                pass

    def finish(self) -> None:
        if self.stage:
            self.timings[self.stage] = round(self.timings.get(self.stage, 0.0) + time.time() - self.t_stage, 2)
            self.done += self.w.get(self.stage, 0.0)
            self.stage = ""

    def sub(self) -> Callable[[float, str], None]:
        return lambda f, m="": self.update(f, m)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------------------
# gender head inference in worker processes (pitch/spectral features are CPU heavy)
# ---------------------------------------------------------------------------
_W_HEAD: Optional[features.GenderHead] = None


def _worker_init(head_path: str) -> None:
    global _W_HEAD
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    _W_HEAD = features.GenderHead(head_path)


def _worker_feats(args) -> np.ndarray:
    clip, emb = args
    assert _W_HEAD is not None
    return _W_HEAD.features(clip, emb)


def head_features_parallel(head: features.GenderHead, clips: list[np.ndarray],
                           embs: Optional[np.ndarray], progress: Callable[[float], None],
                           workers: Optional[int] = None) -> np.ndarray:
    n = len(clips)
    if n == 0:
        return np.zeros((0, 0))
    cpu = os.cpu_count() or 2
    workers = workers or int(config.setting("feature_workers") or 0) or max(1, min(8, cpu // 2))
    if n < 64 or workers <= 1:
        out = []
        for i, c in enumerate(clips):
            out.append(head.features(c, None if embs is None else embs[i]))
            if i % 16 == 0:
                progress(i / n)
        return np.stack(out)
    args = [(c, None if embs is None else embs[i]) for i, c in enumerate(clips)]
    out = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(head.path,)) as ex:
        for i, f in enumerate(ex.map(_worker_feats, args, chunksize=16)):
            out.append(f)
            if i % 32 == 0:
                progress(i / n)
    return np.stack(out)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def _kind_from_vad(vr: float) -> str:
    if vr >= 0.5:
        return "speech"
    if vr <= 0.2:
        return "nonverbal"
    return "mixed"


def _split_points(ev_windows: list[features.Window], labels: np.ndarray, db: np.ndarray) -> list[tuple[float, int]]:
    """[(time, new_label)] at speaker changes inside one event (lowest-energy frame)."""
    pts = []
    for a, b in zip(ev_windows[:-1], ev_windows[1:]):
        la, lb = labels[a.idx], labels[b.idx]
        if la == lb:
            continue
        lo, hi = b.start, a.end
        if hi <= lo:
            t = (a.end + b.start) / 2
        else:
            i0, i1 = int(lo / activity.HOP_S), max(int(lo / activity.HOP_S) + 1, int(hi / activity.HOP_S))
            seg = db[i0:i1]
            t = (i0 + int(np.argmin(seg))) * activity.HOP_S if len(seg) else (lo + hi) / 2
        pts.append((t, lb))
    return pts


def analyze_file(path: str | Path, preset: str = "balanced", work_dir: str | Path | None = None,
                 progress: ProgressCB = None, keep_intermediate: bool = True) -> dict:
    """Run the full analysis on a video or audio file and return the analysis dict."""
    config.apply_torch_threads()
    path = Path(path)
    work = Path(work_dir) if work_dir else config.DATA_DIR / "work" / path.stem
    work.mkdir(parents=True, exist_ok=True)
    P = presets.get_preset(preset)
    dev = config.get_device()
    pr = Progress(progress, STAGE_WEIGHTS)
    notes: list[str] = []
    models_used: dict[str, Any] = {}

    # ---------------------------------------------------------------- extract
    pr.start("extract", "reading media")
    info = media.probe(path)
    media.extract_audio(path, work / "audio.wav", info, progress=lambda f: pr.update(f, "extracting audio"))
    import soundfile as sf

    n44 = sf.info(str(work / "audio.wav")).frames
    duration = n44 / media.SR

    # ---------------------------------------------------------------- separate
    pr.start("separate", "separating vocals")
    sep_spec = presets.model_spec(P["separation"])
    st = path.stat()
    stamp = {"input": path.name, "size": st.st_size, "mtime": int(st.st_mtime), "frames": n44,
             "model": P["separation"], "spec": json.dumps(sep_spec, sort_keys=True, default=str)}
    old = _read_json(work / "separation_stamp.json")
    if (old == stamp and (work / "vocals.wav").exists() and (work / "background.wav").exists()
            and sf.info(str(work / "vocals.wav")).frames == n44):
        sep_info = {"backend": P["separation"], "cached": True}
        notes.append("vocal separation reused from a previous run of the same file and model")
    else:
        sep_info = separation.run_separation(work / "audio.wav", work / "vocals.wav", work / "background.wav",
                                             sep_spec, progress=pr.sub(), device=dev)
        _write_json(work / "separation_stamp.json", stamp)
    models_used["separation"] = P["separation"]

    # ---------------------------------------------------------------- activity
    pr.start("activity", "finding vocal events")
    x16 = features.load_mono_16k(work / "vocals.wav")
    events, ainfo = activity.detect_events(x16, features.SR16, activity.ActivityParams(**P.get("activity", {})),
                                           progress=lambda f: pr.update(f, "voice activity"))
    db = ainfo["db"]
    if ainfo.get("vad") is None:
        notes.append("silero-vad unavailable: energy-only activity detection")

    # ---------------------------------------------------------------- features
    pr.start("features", "speaker embeddings")
    windows = features.make_windows(events, float(P.get("window", 1.5)), float(P.get("hop", 0.5)))
    clips = [features.cut(x16, w.start, w.end) for w in windows]
    nW = len(windows)
    head = features.GenderHead()
    if not head.available:
        notes.append(f"gender head unavailable ({head.error}); using pitch prior only")
    else:
        models_used["gender_head"] = Path(head.path).name
    emb_name = head.embedder or P.get("embedding", "ecapa_tao")
    E = np.zeros((nW, 0), np.float32)
    if nW:
        try:
            embedder = features.make_embedder(emb_name)
            embedder.load(dev)
            E = embedder.embed_batch(clips)
            models_used["embedding"] = embedder.kind
            embedder.unload()
            del embedder
        except Exception as exc:
            notes.append(f"speaker embedding failed ({type(exc).__name__}: {exc}); clustering disabled")
            E = np.zeros((nW, 0), np.float32)
        features.free_torch()
    pr.update(0.25, "pitch")
    # pitch track over the events only
    f0 = np.zeros(int(len(x16) / 160) + 1, np.float32)
    per = np.zeros_like(f0)
    for ev in events:
        a, b = int(ev.start * features.SR16), int(ev.end * features.SR16)
        ff, pp = features.yin_track(x16[a:b])
        i0 = a // 160
        k = min(len(ff), len(f0) - i0)
        f0[i0:i0 + k], per[i0:i0 + k] = ff[:k], pp[:k]
    pitch = [features.pitch_stats(f0, per, 0.01, w.start, w.end) for w in windows]

    p_head = None
    if head.available and nW:
        pr.update(0.35, "gender head")
        try:
            embs = E if (head.embedder and E.shape[1]) else None
            if head.embedder and embs is None:
                raise RuntimeError("head needs embeddings that are unavailable")
            X = head_features_parallel(head, clips, embs, lambda f: pr.update(0.35 + 0.45 * f, "gender head"))
            p_head = head.predict_features(X)
        except Exception as exc:
            notes.append(f"gender head failed: {type(exc).__name__}: {exc}")
            p_head = None

    p_hf = None
    if "hf" in P.get("gender_models", []) and P.get("hf_gender") and nW:
        pr.update(0.8, "speech gender model")
        try:
            hf = features.HFGenderModel(presets.model_spec(P["hf_gender"]))
            if hf.present():
                hf.load(dev)
                p_hf = hf.predict_many(clips)
                hf.unload()
                models_used["hf_gender"] = P["hf_gender"]
            else:
                notes.append(f"{P['hf_gender']} not downloaded; skipped")
        except Exception as exc:
            notes.append(f"speech gender model failed: {type(exc).__name__}: {exc}")
            p_hf = None
        features.free_torch()

    tag = None  # (nW, 4): p_female_speech, p_male_speech, p_speech, p_nonverbal
    if "tagger" in P.get("gender_models", []) and P.get("tagger") and nW:
        pr.update(0.9, "sound event tagger")
        try:
            from . import tagger as tagger_mod

            tg = tagger_mod.make_tagger(presets.model_spec(P["tagger"]), dev)
            if tg.present():
                tg.load()
                # taggers are trained on 10 s clips: give them >= 3 s of context around each window
                tclips = [features.cut(x16, (w.start + w.end) / 2 - 1.5, (w.start + w.end) / 2 + 1.5)
                          for w in windows]
                tag = tg.predict_many(tclips)
                tg.unload()
                models_used["tagger"] = P["tagger"]
            else:
                notes.append(f"{P['tagger']} not downloaded; skipped")
        except Exception as exc:
            notes.append(f"tagger failed: {type(exc).__name__}: {exc}")
            tag = None
        features.free_torch()

    # fuse per window
    weights = P.get("fusion_weights") or gender.DEFAULT_WEIGHTS
    wl = np.zeros(nW)
    wc = np.zeros(nW)
    wkind = []
    for i, w in enumerate(windows):
        ev = events[w.event]
        kind = _kind_from_vad(ev.vad_ratio)
        if tag is not None:
            ps, pn = float(tag[i, 2]), float(tag[i, 3])
            if ps >= 0.2 and ps > 2 * pn:
                kind = "speech"
            elif pn > ps and ev.vad_ratio < 0.5:
                kind = "nonverbal"
        wkind.append(kind)
        evd: dict[str, tuple[float, float]] = {}
        if p_head is not None:
            evd["head"] = (float(p_head[i]), 1.0)
        pp, rel = features.pitch_p_female(pitch[i])
        evd["pitch"] = (pp, rel)
        if p_hf is not None:
            evd["hf"] = (float(p_hf[i]), 1.0 if kind == "speech" else 0.6)
        if tag is not None:
            fs, ms = float(tag[i, 0]), float(tag[i, 1])
            if fs + ms > 1e-3:
                pt = float(gender.sigmoid(gender.logit(fs, 1e-4) - gender.logit(ms, 1e-4)))
                evd["tagger"] = (pt, float(np.clip(tag[i, 2] * 1.5, 0.0, 1.0)))
        l, _ = gender.fuse(evd, kind, weights, head_available=p_head is not None)
        if ev.quiet:
            l *= 0.8  # quiet events: a bit less certain
        wl[i] = l
        wc[i] = abs(2 * gender.sigmoid(l) - 1)
        w.feats = {"kind": kind, "evidence": {k: [round(v[0], 3), round(v[1], 2)] for k, v in evd.items()},
                   "f0": round(pitch[i]["f0_med"], 1), "voiced": round(pitch[i]["voiced"], 2)}

    # ---------------------------------------------------------------- speakers
    pr.start("speakers", "clustering speakers")
    durs = np.array([w.dur for w in windows])
    if nW and E.shape[1]:
        labels = gender.cluster_embeddings(E, durs, float(P.get("cluster_threshold", 0.65)),
                                           window_logits=wl, window_conf=wc)
    else:
        labels = np.zeros(nW, int)
    # smooth labels within events
    for ei in range(len(events)):
        idx = [w.idx for w in windows if w.event == ei]
        if len(idx) >= 3:
            labels[idx] = gender.smooth_labels(labels[idx], 3)

    # segments: split events at speaker changes
    raw_segs: list[dict] = []
    by_event: dict[int, list[features.Window]] = {}
    for w in windows:
        by_event.setdefault(w.event, []).append(w)
    for ei, ev in enumerate(events):
        ws = by_event.get(ei, [])
        cuts = _split_points(ws, labels, db) if len(ws) > 1 else []
        bounds = [ev.start] + [t for t, _ in cuts] + [ev.end]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a > 0.02:
                raw_segs.append({"start": a, "end": b, "event": ei})

    # ---------------------------------------------------------------- overlap detection
    ov_regions: list[tuple[float, float]] = []
    spk_stats = gender.speaker_posteriors(labels, durs, wl, wc) if nW else {}
    if P.get("overlap", "off") != "off" and nW and E.shape[1] and len(spk_stats) >= 2:
        op = overlap.OverlapParams(**P.get("overlap_params", {}))
        cand = overlap.candidate_windows(E, labels, {k: s.logit for k, s in spk_stats.items()}, op,
                                         tagger=tag)
        ov_regions = overlap.regions_from_windows(windows, cand, op)
    # split segments at overlap regions
    segs: list[dict] = []
    for s in raw_segs:
        pieces = [(s["start"], s["end"], False)]
        for (os_, oe) in ov_regions:
            nxt = []
            for a, b, ov in pieces:
                if ov or oe <= a or os_ >= b:
                    nxt.append((a, b, ov))
                    continue
                if os_ > a + 0.05:
                    nxt.append((a, os_, False))
                nxt.append((max(a, os_), min(b, oe), True))
                if oe < b - 0.05:
                    nxt.append((oe, b, False))
            pieces = nxt
        for a, b, ov in pieces:
            segs.append({"start": a, "end": b, "event": s["event"], "overlap": ov})

    # per-segment evidence from its windows
    centers = np.array([(w.start + w.end) / 2 for w in windows]) if nW else np.zeros(0)
    starts_w = np.array([w.start for w in windows]) if nW else np.zeros(0)
    ends_w = np.array([w.end for w in windows]) if nW else np.zeros(0)
    seg_l, seg_c, seg_spk, seg_kind = [], [], [], []
    for s in segs:
        ovl = np.clip(np.minimum(ends_w, s["end"]) - np.maximum(starts_w, s["start"]), 0, None) if nW else np.zeros(0)
        sel = np.flatnonzero(ovl > 0)
        if len(sel) == 0:
            seg_l.append(0.0); seg_c.append(0.0); seg_spk.append(0); seg_kind.append("mixed")
            continue
        wts = ovl[sel] * (0.05 + wc[sel])
        l = float((wts * wl[sel]).sum() / wts.sum())
        seg_l.append(l)
        seg_c.append(abs(2 * gender.sigmoid(l) - 1))
        lab_w: dict[int, float] = {}
        for j in sel:
            lab_w[int(labels[j])] = lab_w.get(int(labels[j]), 0.0) + ovl[j]
        seg_spk.append(max(lab_w, key=lab_w.get))
        kinds: dict[str, float] = {}
        for j in sel:
            kinds[wkind[j]] = kinds.get(wkind[j], 0.0) + ovl[j]
        tot = sum(kinds.values())
        top = max(kinds, key=kinds.get)
        seg_kind.append(top if kinds[top] / tot >= 0.7 else "mixed")

    # speaker stats from segments (duration-weighted, overlap segments excluded)
    seg_l_a, seg_c_a = np.array(seg_l), np.array(seg_c)
    seg_d = np.array([s["end"] - s["start"] for s in segs])
    seg_lab = np.array(seg_spk, int)
    non_ov = np.array([not s["overlap"] for s in segs], bool)
    spk_final: dict[int, gender.SpeakerStats] = {}
    if len(segs):
        m = non_ov if non_ov.any() else np.ones(len(segs), bool)
        spk_final = gender.speaker_posteriors(seg_lab[m], seg_d[m], seg_l_a[m], seg_c_a[m])
        for k in set(seg_lab.tolist()) - set(spk_final):
            sel = seg_lab == k
            spk_final.update({kk: v for kk, v in gender.speaker_posteriors(
                seg_lab[sel], seg_d[sel], seg_l_a[sel], seg_c_a[sel]).items()})

    # ---------------------------------------------------------------- overlap resolution
    pr.start("overlap", "overlapping voices")
    thr = float(config.setting("low_conf_threshold"))
    policy = P.get("overlap_policy") or config.setting("overlap_policy")
    replacements: list[dict] = []
    ov_results: dict[int, dict] = {}
    if P.get("overlap") == "separate":
        # Best: besides the flagged regions, every speech-like segment is checked with the
        # 2-speaker separator (separated-source energies + gender of each source)
        smin = float(P.get("overlap_scan_min_dur", 0.8))
        for i, s in enumerate(segs):
            s["scan"] = (not s["overlap"]) and seg_kind[i] in ("speech", "mixed") and \
                s["end"] - s["start"] >= smin
    if P.get("overlap") == "separate" and any(s["overlap"] or s.get("scan") for s in segs):
        try:
            ov_results = _separate_overlaps(work, segs, x16, P, head, emb_name, dev, pr, notes)
            models_used["overlap_separation"] = P.get("overlap_model")
        except Exception as exc:
            notes.append(f"overlap separation failed: {type(exc).__name__}: {exc}")

    out_segs = []
    for i, s in enumerate(segs):
        spk = spk_final.get(seg_spk[i])
        dur = s["end"] - s["start"]
        if ov_results.get(i, {}).get("replacement"):
            s["overlap"] = True  # confirmed by the separator
        l_fin = gender.combine_with_speaker(seg_l[i], seg_c[i], dur, spk) if not s["overlap"] else seg_l[i]
        p = float(gender.sigmoid(l_fin))
        conf = abs(2 * p - 1)
        label = "female" if p >= 0.5 else "male"
        keep = p >= 0.5
        if s["overlap"]:
            label = "overlap"
            r = ov_results.get(i)
            if r and r.get("replacement"):
                keep = True
                p = r["p_female"]
                conf = r["confidence"]
                for piece in r["pieces"]:
                    if piece.get("wav"):
                        replacements.append({"start": piece["start"], "end": piece["end"], "wav": piece["wav"],
                                             "segment": i + 1})
            elif r and r.get("decision") in ("keep", "remove"):
                keep = r["decision"] == "keep"
                p, conf = r["p_female"], r["confidence"]
            else:
                keep = {"keep": True, "remove": False}.get(policy, p >= 0.5)
                conf = min(conf, 0.5)  # an unresolved overlap is never "confident"
        out_segs.append({
            "id": i + 1, "start": round(s["start"], 3), "end": round(s["end"], 3),
            "speaker": spk.id if spk else "S1", "kind": seg_kind[i], "label": label,
            "p_female": round(p, 4), "confidence": round(conf, 4), "low_confidence": bool(conf < thr),
            "keep": bool(keep), "override": None,
        })
    for s in out_segs:
        k = next((kk for kk, v in spk_final.items() if v.id == s["speaker"]), None)
        if k is not None:
            spk_final[k].n_segments += 1
    speakers = [{"id": v.id, "p_female": round(v.p_female, 4),
                 "label": "female" if v.p_female >= 0.5 else "male",
                 "total_dur": round(v.total_dur, 3), "n_segments": v.n_segments, "override": None}
                for k, v in sorted(spk_final.items())]
    pr.finish()

    analysis = {
        "duration": round(duration, 3),
        "segments": out_segs,
        "speakers": speakers,
        "low_conf_threshold": thr,
        # informational extras
        "version": config.VERSION,
        "preset": preset,
        "device": dev,
        "audio_offset": round(media.audio_offset(info), 4),
        "low_conf_default": config.setting("low_conf_default"),
        "models": models_used,
        "notes": notes,
        "timings": {},
    }
    _write_json(work / "replacements.json", replacements)
    _write_json(work / "debug.json", {
        "events": [{"start": e.start, "end": e.end, "vad": round(e.vad_ratio, 3), "snr": round(e.snr_db, 1),
                    "quiet": e.quiet} for e in events],
        "windows": [{"start": round(w.start, 3), "end": round(w.end, 3), "speaker": int(labels[w.idx]),
                     "logit": round(float(wl[w.idx]), 3), **w.feats} for w in windows],
        "overlap_regions": ov_regions, "overlap_results": {str(k): v for k, v in ov_results.items()},
        "separation": sep_info,
    })

    # ---------------------------------------------------------------- previews
    pr.start("preview", "preparing review audio")
    _write_json(work / "peaks.json", media.compute_peaks(work / "vocals.wav"))
    media.encode_preview(work / "audio.wav", work / "original.m4a")
    pr.update(0.3)
    media.encode_preview(work / "vocals.wav", work / "vocals.m4a")
    pr.update(0.6)
    render_preview(work, analysis)
    pr.finish()
    analysis["timings"] = pr.timings
    _write_json(work / "analysis.json", analysis)
    if progress:
        progress("done", 1.0, "analysis complete")
    return analysis


def _separate_overlaps(work: Path, segs: list[dict], x16: np.ndarray, P: dict,
                       head: features.GenderHead, emb_name: str, dev: str, pr: Progress,
                       notes: list[str]) -> dict[int, dict]:
    import soundfile as sf

    spec = presets.model_spec(P["overlap_model"])
    sep = overlap.SpeechSeparator(spec, dev)
    sep.load()
    results: dict[int, dict] = {}
    pad = 0.25
    todo = [i for i, s in enumerate(segs) if s["overlap"] or s.get("scan")]
    piece_len = float(P.get("overlap_piece_s", 15.0))
    srcs: dict[int, list] = {}
    for n_done, i in enumerate(todo):
        s = segs[i]
        n_p = max(1, int(np.ceil((s["end"] - s["start"]) / piece_len)))
        edges = np.linspace(s["start"], s["end"], n_p + 1)
        for a0, b0 in zip(edges[:-1], edges[1:]):
            a, b = max(0.0, a0 - pad), b0 + pad
            x = features.cut(x16, a, b)
            if len(x) < 1600:
                continue
            y = sep.separate(x)
            g = overlap.ls_gains(x, y)
            srcs.setdefault(i, []).append((a, b, y * g[:, None]))
        pr.update(0.5 * (n_done + 1) / max(1, len(todo)), "separating overlapping voices")
    sep.unload()
    if not srcs:
        return results
    emb = None
    if head.available and head.embedder:
        emb = features.make_embedder(head.embedder)
        emb.load(dev)
    rdir = work / "replacements"
    rdir.mkdir(exist_ok=True)
    k16 = int(pad * features.SR16)
    with sf.SoundFile(str(work / "vocals.wav")) as fv:
        for n_done, (i, pieces) in enumerate(srcs.items()):
            flagged = segs[i]["overlap"]
            pf_min, pm_max, db_min = (0.5, 0.5, -20.0) if flagged else (0.65, 0.35, -12.0)
            out_pieces, confirmed = [], False
            for k, (a, b, y) in enumerate(pieces):
                core = [yy[k16:len(yy) - k16] if len(yy) > 2 * k16 + 160 else yy for yy in y]
                if head.available:
                    E2 = emb.embed_batch(core) if emb is not None else None
                    ps = head.predict_many(core, E2)
                else:
                    ps = np.array([features.pitch_p_female(features.pitch_stats(*features.yin_track(c), 0.01, 0, 1e9))[0]
                                   for c in core])
                rms = np.array([np.sqrt(np.mean(c ** 2) + 1e-12) for c in core])
                rel_db = 20 * np.log10(rms / rms.max())
                fi, mi = int(np.argmax(ps)), int(np.argmin(ps))
                pc = {"start": round(a, 3), "end": round(b, 3), "p_sources": [round(float(q), 3) for q in ps],
                      "rel_db": [round(float(r), 1) for r in rel_db]}
                if fi != mi and ps[fi] >= pf_min and ps[mi] < pm_max and rel_db.min() > db_min:
                    # one female, one male: full-band female-only replacement for this piece
                    fv.seek(int(a * media.SR))
                    v44 = fv.read(int((b - a) * media.SR), dtype="float32", always_2d=True)
                    rep = overlap.female_mask_replacement(v44, y[fi], y[mi], media.SR)
                    rp = rdir / f"seg{i + 1}_{k}.wav"
                    sf.write(str(rp), rep, media.SR, subtype="FLOAT")
                    pc.update(wav=rp.name, p_female=float(ps[fi]),
                              confidence=float(min(abs(2 * ps[fi] - 1), abs(2 * ps[mi] - 1))))
                    confirmed = True
                else:
                    dom = int(np.argmax(rms))
                    q = float(ps[dom]) if rel_db.min() < -10 else float(np.max(ps))
                    pc.update(p_female=q, confidence=float(abs(2 * q - 1)))
                out_pieces.append(pc)
            if confirmed:
                good = [pc for pc in out_pieces if pc.get("wav")]
                results[i] = {"replacement": True, "pieces": out_pieces,
                              "p_female": float(np.mean([pc["p_female"] for pc in good])),
                              "confidence": float(np.min([pc["confidence"] for pc in good]))}
            elif flagged:
                # not a male/female overlap after all: decide on the dominant sources
                q = float(np.mean([pc["p_female"] for pc in out_pieces]))
                results[i] = {"decision": "keep" if q >= 0.5 else "remove", "p_female": q,
                              "confidence": float(abs(2 * q - 1)), "pieces": out_pieces}
            else:
                results[i] = {"scan_only": True, "pieces": out_pieces}
            pr.update(0.5 + 0.5 * (n_done + 1) / len(srcs), "classifying separated voices")
    if emb is not None:
        emb.unload()
    return results


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _replacements(work: Path, keep_ids: set[str]) -> list[dict]:
    reps = _read_json(work / "replacements.json", []) or []
    out = []
    for r in reps:
        if str(r.get("segment")) not in {str(k) for k in keep_ids}:
            continue
        p = Path(r["wav"])
        if not p.is_absolute():
            p = work / "replacements" / p.name
        if p.exists():
            out.append({"start": r["start"], "end": r["end"], "wav": str(p)})
    return out


def render_from_analysis(work_dir: str | Path, analysis: Optional[dict] = None,
                         keep_background: bool = True, out_name: str = "filtered.wav",
                         progress: Optional[Callable[[float], None]] = None) -> str:
    """Render the filtered audio (overrides in ``analysis`` applied); returns the wav path."""
    work = Path(work_dir)
    analysis = analysis or _read_json(work / "analysis.json")
    if not analysis:
        raise FileNotFoundError("analysis.json missing")
    intervals = render.decisions_from_analysis(analysis)
    # replacements only where the overlap segment is kept
    kept = {s["id"] for s in analysis["segments"]
            if render.effective_keep(s, {str(x["id"]): x.get("override") for x in analysis.get("speakers", [])},
                                     analysis.get("low_conf_default") or config.setting("low_conf_default"))}
    reps = _replacements(work, kept)
    return render.render(work / "vocals.wav", work / "background.wav", work / out_name, intervals,
                         keep_background=keep_background, replacements=reps, progress=progress)


def render_preview(work_dir: str | Path, analysis: Optional[dict] = None, keep_background: bool = True) -> str:
    work = Path(work_dir)
    wav = render_from_analysis(work, analysis, keep_background, out_name="preview.wav")
    media.encode_preview(wav, work / "preview.m4a")
    try:
        os.remove(wav)
    except OSError:
        pass
    return str(work / "preview.m4a")


def export_video(src_video: str | Path, filtered_wav: str | Path, out_path: str | Path,
                 progress: Optional[Callable[[float], None]] = None) -> str:
    return media.export_video(src_video, filtered_wav, out_path, progress=progress)


def process_file(src: str | Path, out: str | Path, preset: str = "balanced", keep_background: bool = True,
                 work_dir: str | Path | None = None, progress: ProgressCB = None) -> dict:
    """analyze + render + export in one go (CLI / auto_export)."""
    src, out = Path(src), Path(out)
    work = Path(work_dir) if work_dir else config.DATA_DIR / "work" / src.stem
    t0 = time.time()
    analysis = analyze_file(src, preset, work, progress)
    t1 = time.time()
    if progress:
        progress("rendering", 0.0, "rendering audio")
    wav = render_from_analysis(work, analysis, keep_background,
                               progress=(lambda f: progress("rendering", 0.5 * f, "rendering audio")) if progress else None)
    t2 = time.time()
    if progress:
        progress("export", 0.5, "writing video")
    export_video(src, wav, out, progress=(lambda f: progress("export", 0.5 + 0.5 * f, "writing video")) if progress else None)
    t3 = time.time()
    analysis["timings"].update({"render": round(t2 - t1, 2), "mux": round(t3 - t2, 2),
                                "total": round(t3 - t0, 2)})
    _write_json(work / "analysis.json", analysis)
    return analysis
