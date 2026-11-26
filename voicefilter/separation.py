"""Vocal separation backends with streaming chunked overlap-add inference.

Every backend turns a stereo 44.1 kHz mix into a vocal estimate; the background is always
``mix - vocals`` so ``vocals + background`` reconstructs the mix exactly (up to float
rounding).  Inference streams from/to disk: only a few chunks are in memory, so multi-hour
files fit into RAM and 10 GB of VRAM.

Backends (selected by the ``type`` of a separation spec, see presets.py):

* ``passthrough``           vocals = mix (tests / debugging only)
* ``mel_band_roformer``     Mel-Band RoFormer checkpoint (own implementation, roformer.py)
* ``bs_roformer``           BS-RoFormer checkpoint (own implementation, roformer.py)
* ``demucs``                Demucs/HTDemucs via the ``demucs`` package (weights from HF)
* ``ensemble``              average of the vocal estimates of several specs
"""
from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import config

Progress = Optional[Callable[[float, str], None]]
SR = 44100


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def free_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _yaml_load(path: str) -> dict:
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple)
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda l, s, n: l.construct_sequence(n)
                                  if isinstance(n, yaml.SequenceNode) else None)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader) or {}


def hf_file(repo: str, filename: str, revision: Optional[str] = None,
            local_only: bool = False) -> str:
    """Path of a file from the Hugging Face hub (cached in MODELS/hf)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                           local_files_only=local_only)


def _window(chunk: int, fade: int) -> np.ndarray:
    w = np.ones(chunk, dtype=np.float32)
    if fade > 0:
        r = (0.5 - 0.5 * np.cos(np.pi * (np.arange(fade) + 0.5) / fade)).astype(np.float32)
        w[:fade] = r
        w[-fade:] = r[::-1]
    return np.maximum(w, 1e-3)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
class Backend:
    """Base class: chunked model inference.  Subclasses implement load/_infer."""

    name = "base"
    sr = SR
    chunk = SR * 8
    num_overlap = 2
    batch_size = 1
    outputs_vocals = True  # False: model output is the instrumental, vocals = mix - output

    def __init__(self, spec: dict, device: str = "cpu"):
        self.spec = spec
        self.device = device

    def load(self) -> None:
        pass

    def unload(self) -> None:
        free_torch()

    def _infer(self, x: np.ndarray) -> np.ndarray:  # (b, 2, chunk) -> (b, 2, chunk) target
        raise NotImplementedError

    def infer_vocals(self, x: np.ndarray) -> np.ndarray:
        y = self._infer(x)
        return y if self.outputs_vocals else x - y


class PassthroughBackend(Backend):
    name = "passthrough"
    chunk = SR * 4

    def _infer(self, x):
        return x.copy()


class TorchBackend(Backend):
    model: Any = None

    def unload(self) -> None:
        self.model = None
        free_torch()

    def _infer(self, x):
        import torch

        with torch.inference_mode():
            t = torch.from_numpy(x).to(self.device)
            use_amp = self.device == "cuda" and self.spec.get("fp16", True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                y = self._forward(t)
            return y.float().cpu().numpy()

    def _forward(self, t):
        return self.model(t)


class RoformerBackend(TorchBackend):
    """Mel-Band / BS RoFormer checkpoints in the ZFTurbo MSST format (yaml config + ckpt)."""

    def __init__(self, spec, device="cpu"):
        super().__init__(spec, device)
        self.name = spec.get("id", spec["type"])
        self.target_index = 0

    def _paths(self, local_only=False):
        repo = self.spec["repo"]
        ckpt = hf_file(repo, self.spec["ckpt"], self.spec.get("revision"), local_only)
        cfg = self.spec.get("model_config")
        cfg_path = None
        if not cfg:
            cfg_path = hf_file(self.spec.get("config_repo", repo), self.spec["config"],
                               self.spec.get("revision"), local_only)
        return ckpt, cfg_path

    def download(self):
        self._paths(local_only=False)

    def is_present(self):
        try:
            self._paths(local_only=True)
            return True
        except Exception:
            return False

    def load(self):
        from . import roformer

        ckpt, cfg_path = self._paths(local_only=False)
        full = {"model": dict(self.spec["model_config"])} if cfg_path is None else _yaml_load(cfg_path)
        mcfg = dict(full.get("model", {}))
        audio = full.get("audio", {}) or {}
        tr = full.get("training", {}) or {}
        inf = full.get("inference", {}) or {}
        mtype = self.spec["type"]
        self.model = roformer.build_and_load(mtype, mcfg, ckpt).to(self.device)
        self.chunk = int(self.spec.get("chunk_size") or audio.get("chunk_size") or 352800)
        self.num_overlap = int(self.spec.get("num_overlap") or inf.get("num_overlap") or 2)
        self.num_overlap = max(2, min(self.num_overlap, 4))
        self.batch_size = int(self.spec.get("batch_size") or (2 if self.device == "cuda" else 1))
        instruments = list(tr.get("instruments") or [])
        target = (self.spec.get("target") or tr.get("target_instrument") or
                  (instruments[0] if instruments else "vocals"))
        self.num_stems = int(mcfg.get("num_stems", 1))
        if self.num_stems > 1 and instruments:
            want = "vocals" if "vocals" in instruments else target
            self.target_index = instruments.index(want)
            target = want
        self.outputs_vocals = str(target).lower() in ("vocals", "vocal", "voice")

    def _forward(self, t):
        y = self.model(t)
        if self.num_stems > 1:
            y = y[:, self.target_index]
        return y


class DemucsBackend(TorchBackend):
    """HTDemucs (``demucs`` package); vocals stem."""

    def load(self):
        import torch
        from demucs.pretrained import get_model

        name = self.spec.get("model", "htdemucs")
        repo = self.spec.get("repo")
        if repo:
            from pathlib import Path as _P
            from huggingface_hub import snapshot_download

            d = snapshot_download(repo_id=repo, allow_patterns=self.spec.get("allow_patterns"))
            model = get_model(name, repo=_P(d))
        else:
            model = get_model(name)
        if hasattr(model, "models"):  # BagOfModels
            self.bag = model
        model.eval()
        self.model = model.to(self.device)
        self.sources = list(model.sources)
        seg = float(getattr(model, "segment", 7.8) or 7.8)
        if hasattr(model, "models"):
            seg = min(float(getattr(m, "segment", seg)) for m in model.models)
        self.chunk = int(seg * SR) - 1
        self.num_overlap = 4 if self.spec.get("high_overlap") else 2
        self.batch_size = 1

    def _forward(self, t):
        from demucs.apply import apply_model

        out = apply_model(self.model, t, shifts=0, split=False, device=self.device, progress=False)
        return out[:, self.sources.index("vocals")]


BACKENDS = {
    "passthrough": PassthroughBackend,
    "mel_band_roformer": RoformerBackend,
    "bs_roformer": RoformerBackend,
    "demucs": DemucsBackend,
}


def make_backend(spec: dict, device: Optional[str] = None) -> Backend:
    t = spec["type"]
    if t not in BACKENDS:
        raise ValueError(f"unknown separation backend {t!r}")
    return BACKENDS[t](spec, device or config.get_device())


# ---------------------------------------------------------------------------
# streaming overlap-add
# ---------------------------------------------------------------------------
class _PaddedReader:
    """Random-forward access to [pad zeros | file | zeros] without loading the whole file."""

    def __init__(self, sf_file, pad: int, n: int):
        self.f, self.pad, self.n = sf_file, pad, n
        self.buf = np.zeros((2, 0), np.float32)
        self.buf_start = 0  # padded coordinate of buf[:,0]
        self.file_pos = 0

    def get(self, start: int, length: int) -> np.ndarray:
        end = start + length
        # drop data before start
        if start > self.buf_start:
            drop = min(start - self.buf_start, self.buf.shape[1])
            self.buf = self.buf[:, drop:]
            self.buf_start += drop
        while self.buf_start + self.buf.shape[1] < end:
            need_pos = self.buf_start + self.buf.shape[1]  # padded coordinate to append
            real = need_pos - self.pad
            if real < 0:
                k = min(-real, end - need_pos)
                add = np.zeros((2, k), np.float32)
            elif self.file_pos < self.n:
                k = max(end - need_pos, 1 << 16)
                x = self.f.read(min(k, self.n - self.file_pos), dtype="float32", always_2d=True)
                if len(x) == 0:
                    self.file_pos = self.n
                    continue
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                add = x[:, :2].T.copy()
                self.file_pos += len(x)
            else:
                add = np.zeros((2, end - need_pos), np.float32)
            self.buf = np.concatenate([self.buf, add], axis=1)
        o = start - self.buf_start
        return self.buf[:, o:o + length]


def separate_file(mix_wav: str | Path, vocals_wav: str | Path, background_wav: str | Path,
                  backend: Backend, progress: Progress = None) -> dict:
    """Stream ``mix_wav`` through ``backend``; write vocals and background (= mix - vocals)."""
    import soundfile as sf

    t0 = time.time()
    with sf.SoundFile(str(mix_wav)) as fin:
        if fin.samplerate != backend.sr:
            raise ValueError(f"mix must be {backend.sr} Hz")
        n = fin.frames
        C = int(backend.chunk)
        step = max(1, C // int(backend.num_overlap))
        fade = C - step
        pad = fade
        total = pad + n + pad
        n_chunks = max(1, int(math.ceil(max(0, total - C) / step)) + 1)
        win = _window(C, fade)
        reader = _PaddedReader(fin, pad, n)
        fmt = "RF64" if n * 8 > 2**31 else "WAV"
        acc = np.zeros((2, 0), np.float64)
        wsum = np.zeros(0, np.float64)
        acc_start = 0  # padded coordinate
        written = 0     # real samples written
        with sf.SoundFile(str(vocals_wav), "w", samplerate=backend.sr, channels=2, subtype="FLOAT",
                          format=fmt) as fv, \
                sf.SoundFile(str(background_wav), "w", samplerate=backend.sr, channels=2,
                             subtype="FLOAT", format=fmt) as fb:
            def flush(upto: int) -> None:
                """Write padded range [acc_start, upto) as final."""
                nonlocal acc, wsum, acc_start, written
                k = upto - acc_start
                if k <= 0:
                    return
                v = (acc[:, :k] / np.maximum(wsum[:k], 1e-8)).astype(np.float32)
                # map to real coordinates
                r0 = acc_start - pad
                lo = max(0, -r0)
                hi = min(k, n - r0)
                if hi > lo:
                    vv = v[:, lo:hi]
                    mix = mix_reader_buf_get(r0 + lo, hi - lo)
                    fv.write(vv.T)
                    fb.write((mix - vv).T)
                    written += hi - lo
                acc = acc[:, k:]
                wsum = wsum[k:]
                acc_start = upto

            # second sequential reader for the mix (exact background = mix - vocals)
            mix_f = sf.SoundFile(str(mix_wav))
            mix_rd = _PaddedReader(mix_f, 0, n)

            def mix_reader_buf_get(start: int, length: int) -> np.ndarray:
                return mix_rd.get(start, length)

            try:
                k = 0
                while k < n_chunks:
                    ks = list(range(k, min(n_chunks, k + backend.batch_size)))
                    batch = np.stack([reader.get(i * step, C) for i in ks]).astype(np.float32)
                    y = backend.infer_vocals(batch)
                    for j, i in enumerate(ks):
                        p0 = i * step
                        need = p0 + C - acc_start
                        if need > acc.shape[1]:
                            acc = np.concatenate([acc, np.zeros((2, need - acc.shape[1]))], axis=1)
                            wsum = np.concatenate([wsum, np.zeros(need - wsum.shape[0])])
                        o = p0 - acc_start
                        acc[:, o:o + C] += y[j] * win
                        wsum[o:o + C] += win
                    k = ks[-1] + 1
                    nxt = k * step if k < n_chunks else acc_start + acc.shape[1]
                    flush(nxt)
                    if progress:
                        progress(min(1.0, k / n_chunks), f"separating {k}/{n_chunks}")
                if written < n:  # numerical safety: pad with silence vocals
                    rest = n - written
                    mix = mix_rd.get(written, rest)
                    fv.write(np.zeros((rest, 2), np.float32))
                    fb.write(mix.T)
                    written = n
            finally:
                mix_f.close()
    return {"backend": backend.name, "seconds": time.time() - t0, "samples": n}


def run_separation(mix_wav: str | Path, vocals_wav: str | Path, background_wav: str | Path,
                   spec: dict, progress: Progress = None, device: Optional[str] = None) -> dict:
    """Load the backend(s) of ``spec``, separate, unload.  Handles ``ensemble`` specs."""
    import soundfile as sf

    if spec["type"] == "ensemble":
        members = spec["members"]
        tmp_v = []
        info = {"backend": "ensemble", "members": []}
        for i, m in enumerate(members):
            vi = Path(vocals_wav).with_name(f"_ens{i}_vocals.wav")
            bi = Path(vocals_wav).with_name(f"_ens{i}_bg.wav")

            def pr(f, msg, i=i):
                if progress:
                    progress((i + f) / len(members), f"model {i + 1}/{len(members)}: {msg}")

            info["members"].append(run_separation(mix_wav, vi, bi, m, pr, device))
            os.remove(bi)
            tmp_v.append(vi)
        weights = spec.get("weights") or [1.0] * len(members)
        wsum = float(sum(weights))
        with sf.SoundFile(str(mix_wav)) as fm:
            n = fm.frames
            files = [sf.SoundFile(str(p)) for p in tmp_v]
            fmt = "RF64" if n * 8 > 2**31 else "WAV"
            with sf.SoundFile(str(vocals_wav), "w", samplerate=fm.samplerate, channels=2,
                              subtype="FLOAT", format=fmt) as fv, \
                    sf.SoundFile(str(background_wav), "w", samplerate=fm.samplerate, channels=2,
                                 subtype="FLOAT", format=fmt) as fb:
                while True:
                    x = fm.read(1 << 18, dtype="float32", always_2d=True)
                    if len(x) == 0:
                        break
                    if x.shape[1] == 1:
                        x = np.repeat(x, 2, axis=1)
                    v = sum(w * f.read(len(x), dtype="float32", always_2d=True)
                            for w, f in zip(weights, files)) / wsum
                    fv.write(v.astype(np.float32))
                    fb.write((x[:, :2] - v).astype(np.float32))
            for f in files:
                f.close()
        for p in tmp_v:
            os.remove(p)
        return info

    backend = make_backend(spec, device)
    t0 = time.time()
    if progress:
        progress(0.0, f"loading {backend.name}")
    backend.load()
    load_s = time.time() - t0
    try:
        info = separate_file(mix_wav, vocals_wav, background_wav, backend, progress)
    finally:
        backend.unload()
    info["load_seconds"] = load_s
    return info


def separate(audio: np.ndarray, sr: int, spec: Optional[dict] = None,
             device: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """In-memory convenience wrapper: ``audio`` (n, 2) or (n,) at 44.1 kHz -> (vocals, background)."""
    import tempfile

    import soundfile as sf

    if sr != SR:
        raise ValueError("separate() expects 44.1 kHz audio")
    a = audio if audio.ndim == 2 else audio[:, None]
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    spec = spec or {"type": "passthrough"}
    with tempfile.TemporaryDirectory() as d:
        mix = Path(d) / "mix.wav"
        sf.write(str(mix), a.astype(np.float32), sr, subtype="FLOAT")
        run_separation(mix, Path(d) / "v.wav", Path(d) / "b.wav", spec, device=device)
        v, _ = sf.read(str(Path(d) / "v.wav"), dtype="float32", always_2d=True)
        b, _ = sf.read(str(Path(d) / "b.wav"), dtype="float32", always_2d=True)
    return v, b


def spec_present(spec: dict) -> bool:
    """Are the weights of this separation spec in the local cache?"""
    t = spec["type"]
    if t == "passthrough":
        return True
    if t == "ensemble":
        return all(spec_present(m) for m in spec["members"])
    if t in ("mel_band_roformer", "bs_roformer"):
        return RoformerBackend(spec).is_present()
    if t == "demucs":
        if spec.get("repo"):
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=spec["repo"], allow_patterns=spec.get("allow_patterns"),
                                  local_files_only=True)
                return True
            except Exception:
                return False
        return False
    return False


def spec_download(spec: dict) -> None:
    t = spec["type"]
    if t == "ensemble":
        for m in spec["members"]:
            spec_download(m)
    elif t in ("mel_band_roformer", "bs_roformer"):
        RoformerBackend(spec).download()
    elif t == "demucs" and spec.get("repo"):
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=spec["repo"], allow_patterns=spec.get("allow_patterns"))
