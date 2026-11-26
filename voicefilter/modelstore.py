"""Model file resolution with ordered fallback sources.

Every model file is described by a list of sources, tried in order::

    {"hf": "owner/repo", "file": "path/in/repo", "revision": "main"}   Hugging Face hub
    {"url": "https://...", "sha256": "..." (optional)}                  plain HTTPS download
    {"torchaudio": "BUNDLE_NAME"}                                       torchaudio pipeline

URL downloads land in ``MODELS/files/<key>/<basename>``; HF downloads in ``MODELS/hf``.
:func:`resolve` returns a local path, downloading only when ``download=True``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from . import config

FILES_DIR = config.MODELS_DIR / "files"


class ModelUnavailable(RuntimeError):
    pass


def _url_target(key: str, src: dict) -> Path:
    name = src.get("filename") or src["url"].split("?")[0].rstrip("/").split("/")[-1]
    return FILES_DIR / key / name


def _local(key: str, src: dict) -> Optional[str]:
    if "url" in src:
        p = _url_target(key, src)
        return str(p) if p.exists() and p.stat().st_size > 0 else None
    if "hf" in src:
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(src["hf"], src["file"], revision=src.get("revision"),
                                   local_files_only=True)
        except Exception:
            return None
    if "path" in src:
        p = Path(src["path"])
        if not p.is_absolute():
            p = config.APP_DIR / p
        return str(p) if p.exists() else None
    return None


def _print_progress(label: str) -> Callable[[int, Optional[int]], None]:
    state = {"t": 0.0}

    def cb(done: int, total: Optional[int]) -> None:
        now = time.time()
        if now - state["t"] < 1.0 and (total is None or done < total):
            return
        state["t"] = now
        if total:
            print(f"  {label}: {done / 2**20:7.1f} / {total / 2**20:.1f} MB ({100 * done / total:5.1f}%)",
                  flush=True)
        else:
            print(f"  {label}: {done / 2**20:7.1f} MB", flush=True)

    return cb


def _ssl_context():
    """CA bundle from SSL_CERT_FILE/REQUESTS_CA_BUNDLE, else the OS store, else certifi."""
    import ssl

    extra = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if extra and os.path.exists(extra):
        return ssl.create_default_context(cafile=extra)
    try:
        import truststore  # type: ignore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_url(url: str, dest: Path, sha256: Optional[str] = None,
                 progress: Optional[Callable[[int, Optional[int]], None]] = None,
                 timeout: float = 60.0, retries: int = 8) -> Path:
    """Download with resume (HTTP Range) and retries; atomic rename at the end."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    ctx = _ssl_context()
    total_i: Optional[int] = None
    attempt = 0
    while True:
        done = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "voicefilter/" + config.VERSION}
        if done:
            headers["Range"] = f"bytes={done}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                status = getattr(r, "status", 200)
                if done and status != 206:  # server ignored the range: restart
                    done = 0
                length = r.headers.get("Content-Length")
                if length and length.isdigit():
                    total_i = int(length) + (done if status == 206 else 0)
                with open(part, "ab" if done else "wb") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if progress:
                            progress(done, total_i)
            if total_i is not None and done < total_i:
                raise IOError(f"connection closed at {done} of {total_i} bytes")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and total_i and done >= total_i:
                break
            if exc.code in (401, 403, 404) or attempt >= retries:
                raise
        except (OSError, IOError):
            if attempt >= retries:
                raise
        attempt += 1
        time.sleep(min(10, 1.5 * attempt))
    if sha256:
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for buf in iter(lambda: f.read(1 << 22), b""):
                h.update(buf)
        if h.hexdigest().lower() != sha256.lower():
            os.remove(part)
            raise IOError(f"checksum mismatch for {url}")
    os.replace(part, dest)
    return dest


def _fetch(key: str, src: dict, verbose: bool) -> str:
    if "url" in src:
        dest = _url_target(key, src)
        download_url(src["url"], dest, src.get("sha256"),
                     _print_progress(dest.name) if verbose else None)
        return str(dest)
    if "hf" in src:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(src["hf"], src["file"], revision=src.get("revision"))
    raise ModelUnavailable(f"source {src} cannot be downloaded")


def resolve(key: str, sources: list[dict], download: bool = True, verbose: bool = False) -> str:
    """Local path of the model file ``key``, trying ``sources`` in order."""
    for src in sources:
        p = _local(key, src)
        if p:
            return p
    if not download:
        raise ModelUnavailable(f"{key}: not downloaded")
    errors = []
    for src in sources:
        if "path" in src or "torchaudio" in src:
            continue
        try:
            if verbose:
                where = src.get("hf") or src.get("url")
                print(f"  downloading {key} from {where}", flush=True)
            return _fetch(key, src, verbose)
        except Exception as exc:  # try next source
            errors.append(f"{src.get('hf') or src.get('url')}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  failed: {errors[-1]}", file=sys.stderr, flush=True)
    raise ModelUnavailable(f"{key}: all sources failed:\n  " + "\n  ".join(errors))


def present(key: str, sources: list[dict]) -> bool:
    return any(_local(key, s) for s in sources)


# ---------------------------------------------------------------------------
# torchaudio pipelines (weights from download.pytorch.org, cached in TORCH_HOME)
# ---------------------------------------------------------------------------
def torchaudio_bundle(name: str) -> Any:
    import torchaudio.pipelines as tp

    return getattr(tp, name)


def torchaudio_present(name: str) -> bool:
    try:
        b = torchaudio_bundle(name)
        path = getattr(b, "_model_path", None) or getattr(b, "_path", None)
        if not path:
            return False
        fname = os.path.basename(path)
        return (config.TORCH_HOME / "hub" / "checkpoints" / fname).exists()
    except Exception:
        return False
