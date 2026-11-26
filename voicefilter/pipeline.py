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
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import config, decide, extract, features, media, overlap, presets, render, separation

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
# analysis
# ---------------------------------------------------------------------------
def decision_params(P: dict) -> dict:
    """Decision parameters: decide.DEFAULT_DECISION < preset "decision" < user settings."""
    prm = {**decide.DEFAULT_DECISION, **(P.get("decision") or {})}
    if "low_conf_threshold" not in (P.get("decision") or {}):
        prm["low_conf_threshold"] = float(config.setting("low_conf_threshold"))
    for k in ("decision_threshold", "unsure_policy", "low_conf_threshold"):
        v = config.user_setting(k)
        if v is not None:
            prm[k] = v
    return prm


def analyze_file(path: str | Path, preset: str = "balanced", work_dir: str | Path | None = None,
                 progress: ProgressCB = None, keep_intermediate: bool = True,
                 decision_overrides: Optional[dict] = None) -> dict:
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
    sep_spec.update(P.get("separation_params") or {})
    if sep_spec.get("type") == "ensemble":
        for m in sep_spec.get("members", []):
            m.update(P.get("separation_params") or {})
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

    # ---------------------------------------------------------------- activity + features (cached)
    pr.start("activity", "finding vocal events")
    x16 = features.load_mono_16k(work / "vocals.wav")
    pr.start("features", "speaker embeddings")
    ex = extract.extract(work, x16, P, dev, lambda f, m="": pr.update(f, m), notes, models_used)

    # ---------------------------------------------------------------- decisions
    pr.start("speakers", "clustering speakers")
    prm = {**decision_params(P), **(decision_overrides or {})}
    resolver = None
    if P.get("overlap") == "separate":
        resolver = OverlapSeparator(work, x16, P, dev, notes, models_used,
                                    progress=lambda f, m="": pr.update(f, m))
    pr.start("overlap", "overlapping voices")
    res = decide.decide(ex, P, prm, resolver=resolver, notes=notes)
    pr.finish()

    analysis = {
        "duration": round(duration, 3),
        "segments": res["segments"],
        "speakers": res["speakers"],
        "low_conf_threshold": res["low_conf_threshold"],
        # informational extras
        "version": config.VERSION,
        "preset": preset,
        "device": dev,
        "audio_offset": round(media.audio_offset(info), 4),
        "low_conf_default": config.setting("low_conf_default"),
        "decision_threshold": prm["decision_threshold"],
        "unsure_policy": prm["unsure_policy"],
        "models": models_used,
        "notes": notes,
        "timings": {},
    }
    _write_json(work / "replacements.json", res["replacements"])
    _write_json(work / "debug.json", {
        "events": [{"start": e.start, "end": e.end, "vad": round(e.vad_ratio, 3), "snr": round(e.snr_db, 1),
                    "quiet": e.quiet} for e in ex.events],
        **res["debug"],
        "separation": sep_info,
        "params": {k: v for k, v in prm.items() if isinstance(v, (int, float, str, bool, type(None)))},
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


class OverlapSeparator:
    """2-speaker separation of overlap / scan segments for the Best preset, cached per piece.

    ``__call__(segs, todo)`` -> {seg index: [{start, end, ps, rms, wavs}]}: for every piece
    (<= ``overlap_piece_s``) the two separated sources are turned into full-band masked
    estimates of the 44.1 kHz vocal stem (written to ``replacements/``), and each estimate
    is classified with the gender head.  Results are cached in ``cache/ovl_<model>.json``
    keyed by piece times, so re-deciding with other thresholds needs no model run.
    """

    def __init__(self, work: Path, x16: np.ndarray, P: dict, dev: str, notes: list[str],
                 models_used: dict, progress: Optional[Callable[[float, str], None]] = None,
                 allow_compute: bool = True):
        self.work, self.x16, self.P, self.dev = Path(work), x16, P, dev
        self.notes, self.models_used = notes, models_used
        self.progress = progress or (lambda f, m="": None)
        self.allow_compute = allow_compute
        self.pad = 0.25
        cdir = extract.open_cache(self.work, self.work / "vocals.wav")
        mids = P.get("overlap_models") or [P["overlap_model"]]
        self.mids = mids
        self.cache_path = cdir / f"ovl_{'+'.join(mids)}.json"
        self.cache: dict[str, dict] = _read_json(self.cache_path, {}) or {}
        self.computed = 0
        self._envs: dict[tuple, Optional[np.ndarray]] = {}

    def _pieces_of(self, s: dict) -> list[tuple[float, float]]:
        piece_len = float(self.P.get("overlap_piece_s", 15.0))
        n_p = max(1, int(np.ceil((s["end"] - s["start"]) / piece_len)))
        edges = np.linspace(s["start"], s["end"], n_p + 1)
        out = []
        for a0, b0 in zip(edges[:-1], edges[1:]):
            a, b = max(0.0, a0 - self.pad), b0 + self.pad
            if int(b * features.SR16) - int(a * features.SR16) < 1600:
                continue
            out.append((round(a, 3), round(b, 3)))
        return out

    def __call__(self, segs: list[dict], todo: list[int]) -> dict[int, list[dict]]:
        want = {i: self._pieces_of(segs[i]) for i in todo}
        missing = sorted({pc for v in want.values() for pc in v if f"{pc[0]:.3f}-{pc[1]:.3f}" not in self.cache})
        if missing and self.allow_compute:
            self._compute(missing)
        out: dict[int, list[dict]] = {}
        for i, pcs in want.items():
            lst = []
            for a, b in pcs:
                c = self.cache.get(f"{a:.3f}-{b:.3f}")
                if c and all((self.work / "replacements" / w).exists() for w in c["wavs"]):
                    lst.append({"start": a, "end": b, "ps": c["ps"], "rms": c["rms"], "wavs": c["wavs"],
                                "env": self._env(c["wavs"])})
            if lst:
                out[i] = lst
        return out

    def _env(self, wavs: list[str]) -> Optional[np.ndarray]:
        """10 ms frame level (dB) of each masked estimate (for splitting replaced segments)."""
        key = tuple(wavs)
        if key not in self._envs:
            import soundfile as sf

            from . import activity

            try:
                xs = [sf.read(str(self.work / "replacements" / w), dtype="float32", always_2d=True)[0].mean(axis=1)
                      for w in wavs]
                self._envs[key] = np.stack([activity.frame_db(x, media.SR) for x in xs])
            except Exception:
                self._envs[key] = None
        return self._envs[key]

    def _compute(self, pieces: list[tuple[float, float]]) -> None:
        import soundfile as sf
        import soxr

        sep = None
        for mid in self.mids:
            cand = overlap.SpeechSeparator(presets.model_spec(mid), self.dev)
            if not cand.present():
                continue
            try:
                cand.load()
                sep = cand
                self.models_used["overlap_separation"] = mid
                break
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"overlap model {mid} failed to load ({type(exc).__name__}: {exc})")
        if sep is None:
            raise RuntimeError("no 2-speaker separation model available")
        ys = []
        for n_done, (a, b) in enumerate(pieces):
            x = features.cut(self.x16, a, b)
            y = sep.separate(x)
            ys.append(y * overlap.ls_gains(x, y)[:, None])
            self.progress(0.5 * (n_done + 1) / len(pieces), "separating overlapping voices")
        band = 0.95 * min(sep.sr, features.SR16) / 2
        sep.unload()
        head = features.GenderHead()
        emb = None
        if head.available and head.embedder:
            emb = features.make_embedder(head.embedder)
            emb.load(self.dev)
        rdir = self.work / "replacements"
        rdir.mkdir(exist_ok=True)
        k16 = int(self.pad * features.SR16)
        try:
            with sf.SoundFile(str(self.work / "vocals.wav")) as fv:
                for n_done, ((a, b), y) in enumerate(zip(pieces, ys)):
                    # classify full-band masked estimates (the separator output is band-limited,
                    # which biases pitch/spectral features and embeddings)
                    fv.seek(int(a * media.SR))
                    v44 = fv.read(int((b - a) * media.SR), dtype="float32", always_2d=True)
                    est = overlap.mask_estimates(v44, y[0], y[1], media.SR, band)
                    e16 = [soxr.resample(e.mean(axis=1), media.SR, features.SR16) for e in est]
                    core = [yy[k16:len(yy) - k16] if len(yy) > 2 * k16 + 160 else yy for yy in e16]
                    if head.available:
                        E2 = emb.embed_batch(core) if emb is not None else None
                        ps = head.predict_many(core, E2)
                    else:
                        ps = np.array([features.pitch_p_female(features.pitch_stats(
                            *features.yin_track(c), 0.01, 0, 1e9))[0] for c in core])
                    rms = [float(np.sqrt(np.mean(c ** 2) + 1e-12)) for c in core]
                    wavs = []
                    for k, e in enumerate(est):
                        name = f"ov_{int(round(a * 1000))}_{int(round(b * 1000))}_{k}.wav"
                        sf.write(str(rdir / name), e, media.SR, subtype="FLOAT")
                        wavs.append(name)
                    self.cache[f"{a:.3f}-{b:.3f}"] = {"ps": [float(q) for q in ps], "rms": rms, "wavs": wavs}
                    self.computed += 1
                    self.progress(0.5 + 0.5 * (n_done + 1) / len(pieces), "classifying separated voices")
        finally:
            if emb is not None:
                emb.unload()
            _write_json(self.cache_path, self.cache)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _replacements(work: Path, keep_ids: set[str]) -> list[dict]:
    reps = _read_json(work / "replacements.json", []) or []
    out = []
    kept = {str(k) for k in keep_ids}
    for r in reps:
        ids = r.get("segments") or [r.get("segment")]
        if not any(str(x) in kept for x in ids):
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
