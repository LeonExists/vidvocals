"""Feature extraction stage of the analysis (cached in the work dir).

``extract(work, x16, P, ...)`` runs activity detection and computes every per-window
model output the decision stage needs.  Everything expensive is cached under
``<work>/cache/`` so that re-running the analysis with changed *decision* logic or
parameters (``voicefilter.decide``) takes seconds:

``cache/stamp.json``        vocal stem identity (size, mtime, frames) + cache version; any
                            mismatch wipes the cache
``cache/vad.npy``           silero speech probability per 32 ms frame (whole stem)
``cache/f0.npy, per.npy``   YIN pitch + periodicity per 10 ms frame (whole stem)
``cache/win_<name>.npz``    per-window outputs keyed by window (start ms, end ms) or, for the
                            tagger, by window centre: ``emb_<embedder>`` speaker / head
                            embeddings, ``hand_<hash>`` the gender head's handcrafted
                            features, ``tag_<tagger id>`` tagger outputs

Window stores are keyed by time, not by index, so changed activity parameters only
compute the windows that are new.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import activity, config, features, presets

CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------
def _stem_stamp(vocals: Path, n_chunks: int = 64, chunk: int = 8192) -> dict:
    """Identity of the vocal stem: frames, rate and a hash of evenly spaced chunks of its audio
    data (content-based: rewriting identical stems keeps the cache; header timestamps such as
    the wav PEAK chunk are ignored)."""
    import soundfile as sf

    h = hashlib.sha1()
    with sf.SoundFile(str(vocals)) as f:
        n, sr = f.frames, f.samplerate
        starts = [0] if n <= n_chunks * chunk else [int(k * (n - chunk) / (n_chunks - 1)) for k in range(n_chunks)]
        for st in starts:
            f.seek(st)
            h.update(np.ascontiguousarray(f.read(chunk if len(starts) > 1 else -1, dtype="float32")).tobytes())
    return {"v": CACHE_VERSION, "frames": int(n), "sr": int(sr), "sha1_sampled": h.hexdigest()}


def open_cache(work: Path, vocals: Path) -> Path:
    d = work / "cache"
    stamp = _stem_stamp(vocals)
    try:
        old = json.loads((d / "stamp.json").read_text(encoding="utf-8"))
    except Exception:
        old = None
    if old != stamp:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stamp.json").write_text(json.dumps(stamp), encoding="utf-8")
    return d


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


class WindowStore:
    """Per-window vectors keyed by integer (a, b) millisecond pairs, stored as one npz."""

    def __init__(self, cache_dir: Optional[Path], name: str):
        self.path = cache_dir / f"win_{_safe(name)}.npz" if cache_dir else None
        self.map: dict[tuple[int, int], np.ndarray] = {}
        if self.path and self.path.exists():
            try:
                with np.load(str(self.path), allow_pickle=False) as z:
                    for k, v in zip(z["keys"], z["vals"]):
                        self.map[(int(k[0]), int(k[1]))] = v
            except Exception:
                self.map = {}
        self.dirty = False

    def missing(self, keys: list[tuple[int, int]]) -> list[int]:
        return [i for i, k in enumerate(keys) if k not in self.map]

    def put(self, keys: list[tuple[int, int]], vals: np.ndarray) -> None:
        for k, v in zip(keys, vals):
            self.map[k] = np.asarray(v)
        self.dirty = True

    def get(self, keys: list[tuple[int, int]]) -> np.ndarray:
        return np.stack([self.map[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)

    def save(self) -> None:
        if not (self.path and self.dirty and self.map):
            return
        ks = np.array(list(self.map.keys()), np.int64)
        vs = np.stack(list(self.map.values()))
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(str(tmp), keys=ks, vals=vs)
        os.replace(tmp, self.path)
        self.dirty = False


def win_key(w: features.Window) -> tuple[int, int]:
    return int(round(w.start * 1000)), int(round(w.end * 1000))


def centre_key(w: features.Window) -> tuple[int, int]:
    c = int(round((w.start + w.end) * 500))
    return c, c


def cached_window_values(store: WindowStore, keys: list[tuple[int, int]],
                         compute: Callable[[list[int]], np.ndarray]) -> np.ndarray:
    """Values for all keys; ``compute(indices)`` is called for the missing ones (unique keys)."""
    miss = store.missing(keys)
    if miss:
        uniq: dict[tuple[int, int], int] = {}
        for i in miss:
            uniq.setdefault(keys[i], i)
        idx = list(uniq.values())
        vals = compute(idx)
        store.put([keys[i] for i in idx], vals)
        store.save()
    return store.get(keys)


# ---------------------------------------------------------------------------
# handcrafted head features in worker processes (pyin is CPU heavy)
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _worker_hand(clip: np.ndarray) -> np.ndarray:
    from .assets import gender_head as gh

    return gh.handcrafted_features(clip, features.SR16)


def hand_features_parallel(clips: list[np.ndarray], progress: Callable[[float], None],
                           workers: Optional[int] = None) -> np.ndarray:
    from .assets import gender_head as gh

    n = len(clips)
    if n == 0:
        return np.zeros((0, len(gh.HANDCRAFTED_NAMES)), np.float32)
    cpu = os.cpu_count() or 2
    workers = workers or int(config.setting("feature_workers") or 0) or max(1, min(8, cpu // 2))
    out = []
    if n < 64 or workers <= 1:
        for i, c in enumerate(clips):
            out.append(gh.handcrafted_features(c, features.SR16))
            if i % 16 == 0:
                progress(i / n)
        return np.stack(out)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for i, f in enumerate(ex.map(_worker_hand, clips, chunksize=16)):
            out.append(f)
            if i % 32 == 0:
                progress(i / n)
    return np.stack(out)


def _hand_tag() -> str:
    """Cache tag of the handcrafted features: hash of the gender_head module source."""
    from .assets import gender_head as gh

    try:
        h = hashlib.sha1(Path(gh.__file__).read_bytes()).hexdigest()[:10]
    except Exception:
        h = "x"
    return f"hand_{h}"


# ---------------------------------------------------------------------------
# extracted features
# ---------------------------------------------------------------------------
@dataclass
class Extracted:
    duration: float
    events: list
    windows: list
    db: np.ndarray
    E: np.ndarray                         # (nW, d) clustering embedding (d = 0: unavailable)
    p_head: Optional[np.ndarray] = None   # (nW,) calibrated head p(female)
    pitch: list = field(default_factory=list)
    tag: Optional[np.ndarray] = None      # (nW, 4) p_female_speech, p_male_speech, p_speech, p_nonverbal
    p_hf: Optional[np.ndarray] = None
    p_head_alt: dict = field(default_factory=dict)   # other heads (diagnostics / fusion experiments)
    extra: dict = field(default_factory=dict)


def _embed(store: WindowStore, keys, clips, name: str, dev: str, models_used: dict, notes: list,
           label: str) -> Optional[np.ndarray]:
    def compute(idx):
        emb = features.make_embedder(name)
        emb.load(dev)
        try:
            out = emb.embed_batch([clips[i] for i in idx])
            models_used[label] = emb.kind
        finally:
            emb.unload()
            features.free_torch()
        return out

    try:
        out = cached_window_values(store, keys, compute)
        if label not in models_used:
            models_used[label] = name
        return out
    except Exception as exc:
        notes.append(f"{label} {name} failed ({type(exc).__name__}: {exc})")
        return None


def extract(work: Path, x16: np.ndarray, P: dict, dev: str, progress: Callable[[float, str], None],
            notes: list[str], models_used: dict[str, Any], activity_overrides: Optional[dict] = None,
            use_cache: bool = True) -> Extracted:
    """Activity detection + per-window features (cached).  ``progress(frac, message)``."""
    cdir = open_cache(work, work / "vocals.wav") if use_cache else None

    # ------------------------------------------------------------ frame-level (whole stem)
    vad = None
    if cdir is not None and (cdir / "vad.npy").exists():
        vad = np.load(str(cdir / "vad.npy"))
    ap = activity.ActivityParams(**{**(P.get("activity") or {}), **(activity_overrides or {})})
    if vad is None and ap.use_vad:
        vad = activity.silero_probs(x16, lambda f: progress(0.05 * f, "voice activity"))
        if vad is not None and cdir is not None:
            np.save(str(cdir / "vad.npy"), vad)
    events, ainfo = activity.detect_events(x16, features.SR16, ap, vad=vad if vad is not None else np.zeros(0))
    db = ainfo["db"]
    if vad is None and ap.use_vad:
        notes.append("silero-vad unavailable: energy-only activity detection")

    if cdir is not None and (cdir / "f0.npy").exists() and (cdir / "per.npy").exists():
        f0, per = np.load(str(cdir / "f0.npy")), np.load(str(cdir / "per.npy"))
    else:
        progress(0.06, "pitch")
        f0, per = features.yin_track(x16)
        if cdir is not None:
            np.save(str(cdir / "f0.npy"), f0)
            np.save(str(cdir / "per.npy"), per)

    windows = features.make_windows(events, float(P.get("window", 1.5)), float(P.get("hop", 0.5)))
    for i, w in enumerate(windows):
        w.idx = i
    nW = len(windows)
    clips = [features.cut(x16, w.start, w.end) for w in windows]
    keys = [win_key(w) for w in windows]
    pitch = [features.pitch_stats(f0, per, 0.01, w.start, w.end) for w in windows]
    ex = Extracted(duration=len(x16) / features.SR16, events=events, windows=windows, db=db,
                   E=np.zeros((nW, 0), np.float32), pitch=pitch)
    ex.extra["activity_params"] = ap.__dict__.copy()
    if not nW:
        return ex

    # ------------------------------------------------------------ embeddings
    progress(0.1, "speaker embeddings")
    emb_name = P.get("embedding", "ecapa_tao")
    E = _embed(WindowStore(cdir, f"emb_{emb_name}"), keys, clips, emb_name, dev, models_used, notes, "embedding")
    if E is None:
        notes.append("clustering disabled")
    else:
        ex.E = E.astype(np.float32)

    head = features.GenderHead()
    if not head.available:
        notes.append(f"gender head unavailable ({head.error}); using pitch prior only")
    else:
        models_used["gender_head"] = Path(head.path).name
    E_head = None
    if head.available and head.embedder:
        if head.embedder == emb_name:
            E_head = ex.E if ex.E.shape[1] else None
        else:
            progress(0.25, "gender features")
            E_head = _embed(WindowStore(cdir, f"emb_{head.embedder}"), keys, clips, head.embedder, dev,
                            models_used, notes, "head_embedding")

    hand = None
    if head.available:
        progress(0.45, "gender head")
        try:
            hand = cached_window_values(
                WindowStore(cdir, _hand_tag()), keys,
                lambda idx: hand_features_parallel([clips[i] for i in idx],
                                                   lambda f: progress(0.45 + 0.35 * f, "gender head")))
        except Exception as exc:
            notes.append(f"gender head features failed: {type(exc).__name__}: {exc}")
    if head.available and hand is not None:
        try:
            if head.embedder and E_head is None:
                raise RuntimeError("head needs embeddings that are unavailable")
            ex.p_head = head_probs(head, E_head, hand)
        except Exception as exc:
            notes.append(f"gender head failed: {type(exc).__name__}: {exc}")
        # alternative heads on cached inputs (diagnostics, no extra model runs)
        for alt in sorted(config.ASSETS_DIR.glob("gender_head*.npz")):
            if head.path and Path(head.path) == alt:
                continue
            try:
                h2 = features.GenderHead(str(alt))
                if not h2.available or Path(h2.path) != alt:
                    continue
                e2 = None
                if h2.embedder == emb_name:
                    e2 = ex.E if ex.E.shape[1] else None
                elif h2.embedder == head.embedder:
                    e2 = E_head
                if h2.embedder and e2 is None:
                    continue
                ex.p_head_alt[alt.stem] = head_probs(h2, e2, hand)
            except Exception:
                pass

    # ------------------------------------------------------------ optional models
    if "hf" in P.get("gender_models", []) and P.get("hf_gender"):
        progress(0.8, "speech gender model")
        try:
            hf = features.HFGenderModel(presets.model_spec(P["hf_gender"]))
            if hf.present():
                def comp_hf(idx):
                    hf.load(dev)
                    try:
                        return hf.predict_many([clips[i] for i in idx])[:, None]
                    finally:
                        hf.unload()
                ex.p_hf = cached_window_values(WindowStore(cdir, f"hf_{P['hf_gender']}"), keys, comp_hf)[:, 0]
                models_used["hf_gender"] = P["hf_gender"]
            else:
                notes.append(f"{P['hf_gender']} not downloaded; skipped")
        except Exception as exc:
            notes.append(f"speech gender model failed: {type(exc).__name__}: {exc}")
        features.free_torch()

    if P.get("tagger"):
        progress(0.9, "sound event tagger")
        try:
            from . import tagger as tagger_mod

            tg = tagger_mod.make_tagger(presets.model_spec(P["tagger"]), dev)
            if tg.present():
                def comp_tag(idx):
                    tg.load()
                    try:
                        # taggers are trained on 10 s clips: >= 3 s of context around each window
                        tcl = [features.cut(x16, (windows[i].start + windows[i].end) / 2 - 1.5,
                                            (windows[i].start + windows[i].end) / 2 + 1.5) for i in idx]
                        return tg.predict_many(tcl)
                    finally:
                        tg.unload()
                ex.tag = cached_window_values(WindowStore(cdir, f"tag_{P['tagger']}"),
                                              [centre_key(w) for w in windows], comp_tag)
                models_used["tagger"] = P["tagger"]
            else:
                notes.append(f"{P['tagger']} not downloaded; skipped")
        except Exception as exc:
            notes.append(f"tagger failed: {type(exc).__name__}: {exc}")
            ex.tag = None
        features.free_torch()
    progress(1.0, "features done")
    return ex


def head_probs(head: features.GenderHead, E_head: Optional[np.ndarray], hand: np.ndarray) -> np.ndarray:
    from .assets import gender_head as gh

    if head.embedder:
        X = np.concatenate([np.stack([gh.prepare_embedding(e, head.head.embedding_dim) for e in E_head]), hand],
                           axis=1)
    else:
        X = hand
    return head.predict_features(X)
