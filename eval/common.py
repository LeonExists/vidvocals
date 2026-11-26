"""Shared helpers for the evaluation tooling (paths, manifest, audio IO, splits).

Everything here is CPU-friendly and has no dependency on the voicefilter backend.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"            # git-ignored raw data
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = EVAL_DIR / "cache"          # git-ignored features / predictions
MANIFEST = DATA_DIR / "clips.csv"       # one row per labelled clip
SPLITS = EVAL_DIR / "splits.json"       # committed: speaker -> train/test
MIXES_DIR = DATA_DIR / "mixes"
RESULTS_DIR = EVAL_DIR / "results"

MANIFEST_FIELDS = ["clip_id", "path", "source", "speaker", "gender", "kind", "category",
                   "lang", "duration", "license"]


def setup_env() -> None:
    """Shared model caches inside the repo (see docs/AGENT_BRIEF.md)."""
    os.environ.setdefault("HF_HOME", str(REPO_DIR / "models" / "hf"))
    os.environ.setdefault("TORCH_HOME", str(REPO_DIR / "models" / "torch"))


def read_manifest(path: Path = MANIFEST) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["duration"] = float(r["duration"])
    return rows


def write_manifest(rows: list[dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def stable_unit(key: str, salt: str = "voicefilter-split-v1") -> float:
    """Deterministic pseudo-random number in [0,1) from a string."""
    h = hashlib.sha1(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def load_splits(path: Path = SPLITS) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def speaker_split(speaker: str, splits: dict | None = None) -> str:
    splits = splits or load_splits()
    return splits["speakers"].get(speaker, "unassigned")


def load_audio(path: str | Path, sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Load any audio file as float32 at `sr` (mono -> 1-D, else (channels, n))."""
    import librosa
    import soundfile as sf

    path = Path(path)
    if not path.is_absolute():
        path = DATA_DIR / path
    try:
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.T
    except Exception:  # e.g. ogg/mp3 that libsndfile cannot read
        y, file_sr = librosa.load(str(path), sr=None, mono=False)
        y = np.atleast_2d(y).astype(np.float32)
    if mono:
        y = y.mean(axis=0)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
    return np.ascontiguousarray(y, dtype=np.float32)


def duration_bucket(d: float) -> str:
    if d < 0.5:
        return "<0.5s"
    if d <= 1.5:
        return "0.5-1.5s"
    return ">1.5s"
