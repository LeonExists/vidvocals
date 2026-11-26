"""Paths and settings.

Everything the app writes lives inside the app folder (``VF_APP_DIR``):

* ``data/``   jobs, uploads, temp files   (``VF_DATA_DIR``)
* ``models/`` Hugging Face / torch caches (``VF_MODELS_DIR``)

Importing this module sets ``HF_HOME``, ``TORCH_HOME`` and the temp directory so that no
library writes to the user profile.  Settings can come from environment variables or from
``APP/config.json`` (written by the installer; keys: port, host, ffmpeg, ffprobe, device and
any key of :data:`DEFAULT_SETTINGS`).  Environment variables win over the config file.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

VERSION = "0.1.0"

PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(os.environ.get("VF_APP_DIR") or PACKAGE_DIR.parent).resolve()
CONFIG_FILE = APP_DIR / "config.json"


def _load_config_file() -> dict[str, Any]:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # broken config must not prevent start-up
        print(f"[voicefilter] warning: could not read {CONFIG_FILE}: {exc}", file=sys.stderr)
        return {}


FILE_CONFIG: dict[str, Any] = _load_config_file()


def _setting(env: str, key: str, default: Any) -> Any:
    v = os.environ.get(env)
    if v not in (None, ""):
        return v
    v = FILE_CONFIG.get(key)
    if v not in (None, ""):
        return v
    return default


DATA_DIR = Path(_setting("VF_DATA_DIR", "data_dir", APP_DIR / "data")).resolve()
MODELS_DIR = Path(_setting("VF_MODELS_DIR", "models_dir", APP_DIR / "models")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
TMP_DIR = DATA_DIR / "tmp"
WEB_DIR = PACKAGE_DIR / "web"
ASSETS_DIR = PACKAGE_DIR / "assets"
DOCS_DIR = APP_DIR / "docs"

HOST = str(_setting("VF_HOST", "host", "0.0.0.0"))
PORT = int(_setting("VF_PORT", "port", 8765))
DEVICE_PREF = str(_setting("VF_DEVICE", "device", "auto")).lower()


def _exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def _find_tool(env: str, key: str, name: str) -> str:
    v = _setting(env, key, None)
    if v:
        return str(v)
    bundled = APP_DIR / "runtime" / "ffmpeg" / "bin" / _exe(name)
    if bundled.exists():
        return str(bundled)
    found = shutil.which(name)
    return found or name


FFMPEG = _find_tool("VF_FFMPEG", "ffmpeg", "ffmpeg")
FFPROBE = _find_tool("VF_FFPROBE", "ffprobe", "ffprobe")

# Tunables used by the pipeline.  Any of these can be overridden in config.json.
DEFAULT_SETTINGS: dict[str, Any] = {
    "low_conf_threshold": 0.6,     # confidence = |2p-1| below this -> low_confidence
    "fade_ms": 15.0,               # raised-cosine fade length at every gain boundary
    "gate_unlabeled": "keep",      # vocal-stem regions without any event: keep|mute
    "low_conf_default": "model",   # model|keep|remove : render-time override for low-confidence segments
    # decision defaults are tuned per preset (presets.py "decision"); setting these in
    # config.json / VF_* env vars overrides every preset
    "decision_threshold": 0.5,     # keep a segment if p_female >= this
    "unsure_policy": "model",      # model|keep|remove : model decision for low-confidence segments
    "overlap_policy": "p_female",  # overlap without separation: p_female|keep|remove
    "max_upload_bytes": 0,         # 0 = unlimited
    "preview_debounce_s": 1.0,
    "cpu_threads": 0,              # 0 = torch default
    "feature_workers": 0,          # processes for pitch/spectral features, 0 = auto
    "keep_job_files": True,
}
SETTINGS: dict[str, Any] = dict(DEFAULT_SETTINGS)
for _k in DEFAULT_SETTINGS:
    if _k in FILE_CONFIG:
        SETTINGS[_k] = FILE_CONFIG[_k]
    _env = os.environ.get("VF_" + _k.upper())
    if _env not in (None, ""):
        _t = type(DEFAULT_SETTINGS[_k])
        try:
            SETTINGS[_k] = _t(_env) if _t is not bool else _env.lower() in ("1", "true", "yes")
        except ValueError:
            pass


def setting(key: str) -> Any:
    return SETTINGS.get(key, DEFAULT_SETTINGS.get(key))


USER_SET: set[str] = {k for k in DEFAULT_SETTINGS
                      if k in FILE_CONFIG or os.environ.get("VF_" + k.upper()) not in (None, "")}


def user_setting(key: str) -> Any:
    """The setting if the user set it (config.json or VF_* env var), else None."""
    return SETTINGS.get(key) if key in USER_SET else None


# ---------------------------------------------------------------------------
# environment for libraries: caches + temp inside the app folder
# ---------------------------------------------------------------------------
def _init_env() -> None:
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # numba (librosa) cache, matplotlib etc. should not touch the profile either
    os.environ.setdefault("NUMBA_CACHE_DIR", str(DATA_DIR / "cache" / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "cache" / "mpl"))
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(TMP_DIR)
        for var in ("TMP", "TEMP", "TMPDIR"):
            os.environ[var] = str(TMP_DIR)
    except OSError:
        pass


_init_env()

HF_HOME = Path(os.environ["HF_HOME"])
TORCH_HOME = Path(os.environ["TORCH_HOME"])


def ensure_dirs() -> None:
    for d in (DATA_DIR, JOBS_DIR, TMP_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


_device_cache: str | None = None


def get_device() -> str:
    """Resolve VF_DEVICE (auto|cuda|cpu) to 'cuda' or 'cpu'."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    pref = DEVICE_PREF
    dev = "cpu"
    if pref in ("auto", "cuda", "gpu"):
        try:
            import torch

            if torch.cuda.is_available():
                dev = "cuda"
            elif pref in ("cuda", "gpu"):
                print("[voicefilter] warning: VF_DEVICE=cuda but CUDA is not available, using CPU",
                      file=sys.stderr)
        except Exception:
            dev = "cpu"
    _device_cache = dev
    return dev


def device_info() -> dict[str, Any]:
    """{'device': 'cuda'|'cpu', 'gpu_name': str|None, 'vram_gb': float|None}"""
    dev = get_device()
    info: dict[str, Any] = {"device": dev, "gpu_name": None, "vram_gb": None}
    if dev == "cuda":
        try:
            import torch

            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["vram_gb"] = round(p.total_memory / 1024**3, 2)
        except Exception:
            pass
    return info


def apply_torch_threads() -> None:
    n = int(setting("cpu_threads") or 0)
    if n > 0:
        try:
            import torch

            torch.set_num_threads(n)
        except Exception:
            pass
