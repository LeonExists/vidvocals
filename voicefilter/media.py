"""ffprobe / ffmpeg helpers: probe, extract, mux, preview encodes, peaks."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import config

SR = 44100  # internal sample rate for the full-band stems

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".ts", ".mts",
              ".m2ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob", ".mxf", ".asf"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif",
              ".mka", ".ac3", ".eac3", ".dts", ".amr", ".alac", ".caf"}


class MediaError(RuntimeError):
    """User-facing media problem (message is shown in the UI)."""


_NOWINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW on Windows


def _run(cmd: list[str], *, capture: bool = True, check: bool = True,
         timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=timeout, creationflags=_NOWINDOW)
    except FileNotFoundError as exc:
        raise MediaError(f"{Path(cmd[0]).name} not found ({cmd[0]}). Install ffmpeg or set VF_FFMPEG.") from exc
    if check and p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = "\n".join(err[-8:])
        raise MediaError(f"{Path(cmd[0]).name} failed (exit {p.returncode}): {tail}")
    return p


def tool_version(exe: str) -> Optional[str]:
    try:
        p = subprocess.run([exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=20, creationflags=_NOWINDOW)
        if p.returncode != 0:
            return None
        first = p.stdout.decode("utf-8", "replace").splitlines()[0]
        m = re.search(r"version\s+(\S+)", first)
        return m.group(1) if m else first.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def _f(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def probe(path: str | Path) -> dict[str, Any]:
    """Return a compact description of the media file.

    keys: format_name, duration, start_time, has_video, video_streams, audio (first audio stream
    dict or None: index, codec, channels, sample_rate, start_time, duration, bit_rate),
    n_audio, subtitles, raw (ffprobe json).
    """
    try:
        p = _run([config.FFPROBE, "-v", "error", "-print_format", "json", "-show_format",
                  "-show_streams", str(path)])
    except MediaError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):  # ffprobe itself missing
            raise
        last = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
        last = last.replace(str(path), Path(path).name)
        raise MediaError(f"Could not read this file as video or audio ({last.strip()})") from exc
    try:
        data = json.loads(p.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid output") from exc
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [s for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition", {}) or {}).get("attached_pic")]
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    fstart = _f(fmt.get("start_time"), 0.0) or 0.0
    a = None
    if audio:
        s = audio[0]
        a = {
            "index": int(s.get("index", 0)),
            "codec": s.get("codec_name"),
            "channels": int(s.get("channels") or 2),
            "sample_rate": int(s.get("sample_rate") or SR),
            "start_time": _f(s.get("start_time"), fstart),
            "duration": _f(s.get("duration")),
            "bit_rate": int(s["bit_rate"]) if str(s.get("bit_rate", "")).isdigit() else None,
        }
    duration = _f(fmt.get("duration"))
    if duration is None:
        ds = [_f(s.get("duration")) for s in streams]
        ds = [d for d in ds if d]
        duration = max(ds) if ds else None
    return {
        "format_name": fmt.get("format_name", ""),
        "duration": duration,
        "start_time": fstart,
        "has_video": bool(video),
        "video_streams": [{"index": int(s.get("index", 0)), "codec": s.get("codec_name"),
                           "width": s.get("width"), "height": s.get("height"),
                           "start_time": _f(s.get("start_time"), fstart)} for s in video],
        "audio": a,
        "n_audio": len(audio),
        "subtitles": [{"index": int(s.get("index", 0)), "codec": s.get("codec_name")} for s in subs],
        "raw": data,
    }


def audio_offset(info: dict[str, Any]) -> float:
    """Start of the first audio stream relative to the start of the file (seconds, >= 0)."""
    a = info.get("audio") or {}
    off = (a.get("start_time") or 0.0) - (info.get("start_time") or 0.0)
    return max(0.0, float(off))


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(r"out_time_(?:us|ms)=(\d+)")


def _run_progress(cmd: list[str], total_s: Optional[float],
                  progress: Optional[Callable[[float], None]] = None) -> None:
    """Run ffmpeg with -progress pipe:1 and report fraction done."""
    cmd = cmd[:1] + ["-nostdin", "-progress", "pipe:1", "-nostats"] + cmd[1:]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=_NOWINDOW)
    except FileNotFoundError as exc:
        raise MediaError(f"ffmpeg not found ({cmd[0]}). Install ffmpeg or set VF_FFMPEG.") from exc
    err_lines: list[str] = []

    import threading

    def _drain_err() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_lines.append(line.decode("utf-8", "replace").rstrip())
            if len(err_lines) > 200:
                del err_lines[:100]

    t = threading.Thread(target=_drain_err, daemon=True)
    t.start()
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").strip()
        m = _TIME_RE.match(line)
        if m and total_s and progress:
            us = int(m.group(1))
            progress(min(1.0, us / 1e6 / total_s))
    rc = proc.wait()
    t.join(timeout=5)
    if rc != 0:
        raise MediaError(f"ffmpeg failed (exit {rc}): " + "\n".join(err_lines[-8:]))


def extract_audio(src: str | Path, out_wav: str | Path, info: Optional[dict] = None,
                  sr: int = SR, progress: Optional[Callable[[float], None]] = None) -> dict:
    """Decode the first audio stream to a stereo float32 WAV at ``sr``.

    Mono sources are duplicated to both channels, multichannel sources are downmixed.
    Returns the probe info.
    """
    info = info or probe(src)
    if not info.get("audio"):
        raise MediaError("This file has no audio track, there is nothing to filter.")
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [config.FFMPEG, "-v", "error", "-y", "-i", str(src), "-map", f"0:{info['audio']['index']}",
           "-vn", "-sn", "-dn", "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le",
           "-rf64", "auto", str(out_wav)]
    dur = info.get("duration") or (info["audio"].get("duration"))
    _run_progress(cmd, dur, progress)
    if not out_wav.exists() or out_wav.stat().st_size < 100:
        raise MediaError("Audio extraction produced no data (empty or broken audio track).")
    import soundfile as sf

    if sf.info(str(out_wav)).frames == 0:
        raise MediaError("The audio track is empty.")
    return info


# ---------------------------------------------------------------------------
# mux / export
# ---------------------------------------------------------------------------
def _audio_codec_for(ext: str, src_info: dict) -> list[str]:
    """ffmpeg audio codec args for the output container."""
    ext = ext.lower()
    a = src_info.get("audio") or {}
    ch = 1 if a.get("channels") == 1 else 2
    sr = a.get("sample_rate") or 48000
    if ext in (".webm",) or (ext in (".ogg", ".opus", ".ogv")):
        sr_o = 48000  # opus only supports 48 kHz family
        return ["-c:a", "libopus", "-b:a", "192k" if ch == 2 else "128k", "-ar", str(sr_o), "-ac", str(ch)]
    if sr not in (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000):
        sr = 48000
    if ext in (".wav",):
        return ["-c:a", "pcm_s24le", "-ar", str(sr), "-ac", str(ch)]
    if ext in (".flac",):
        return ["-c:a", "flac", "-ar", str(sr), "-ac", str(ch)]
    if ext in (".mp3", ".avi", ".wmv", ".asf", ".mpg", ".mpeg", ".vob", ".flv"):
        if ext in (".mpg", ".mpeg", ".vob"):
            return ["-c:a", "mp2", "-b:a", "256k", "-ar", str(min(sr, 48000)), "-ac", str(ch)]
        if ext == ".flv":
            return ["-c:a", "aac", "-b:a", "256k", "-ar", str(min(sr, 48000)), "-ac", str(ch)]
        return ["-c:a", "libmp3lame", "-b:a", "320k", "-ar", str(min(sr, 48000)), "-ac", str(ch)]
    # mp4, m4v, mov, mkv, ts, m4a, 3gp, ...
    return ["-c:a", "aac", "-b:a", "256k" if ch == 2 else "160k", "-ar", str(min(sr, 96000)), "-ac", str(ch)]


def output_extension(src: str | Path) -> str:
    ext = Path(src).suffix.lower()
    if ext in (".aac",):
        return ".m4a"
    if ext in (".wma", ".aiff", ".aif", ".amr", ".ac3", ".eac3", ".dts", ".alac", ".caf"):
        return ".m4a"
    if ext in (".mka",):
        return ".mka"
    if ext == "" or ext not in VIDEO_EXTS | AUDIO_EXTS:
        return ".mkv"
    if ext in (".mxf",):
        return ".mkv"
    return ext


def export_video(src_video: str | Path, filtered_wav: str | Path, out_path: str | Path,
                 info: Optional[dict] = None,
                 progress: Optional[Callable[[float], None]] = None) -> str:
    """Replace the audio of ``src_video`` with ``filtered_wav``.

    The video stream(s) and subtitles are copied bit-for-bit, metadata and chapters are kept,
    and the new audio starts at the same time offset as the original first audio stream so
    A/V sync is identical.  Works for audio-only sources too.
    """
    src_video, out_path = Path(src_video), Path(out_path)
    info = info or probe(src_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    offset = audio_offset(info)
    tmp_out = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
    cmd = [config.FFMPEG, "-v", "error", "-y", "-i", str(src_video)]
    if offset > 0.0005:
        cmd += ["-itsoffset", f"{offset:.6f}"]
    cmd += ["-i", str(filtered_wav)]
    is_video = info.get("has_video") and ext not in AUDIO_EXTS
    if is_video:
        cmd += ["-map", "0:v?"]
    cmd += ["-map", "1:a:0"]
    sub_ok = ext in (".mkv", ".mp4", ".m4v", ".mov", ".webm", ".mka")
    if is_video and sub_ok and info.get("subtitles"):
        sub_codec = "copy"
        if ext in (".mp4", ".m4v", ".mov") and _has_subs_needing_conversion(info):
            sub_codec = "mov_text"  # text subtitles from mkv sources cannot be copied into mp4
        cmd += ["-map", "0:s?", "-c:s", sub_codec]
    if ext in (".mkv", ".mka"):
        cmd += ["-map", "0:t?"]  # attachments (fonts)
    cmd += ["-c:v", "copy"]
    cmd += _audio_codec_for(ext, info)
    cmd += ["-map_metadata", "0", "-map_chapters", "0"]
    a = info.get("audio") or {}
    if a.get("index") is not None:
        # carry over language / title of the original audio stream
        cmd += ["-map_metadata:s:a:0", f"0:s:{a['index']}"]
    if ext in (".mp4", ".m4v", ".mov", ".m4a", ".3gp"):
        cmd += ["-movflags", "+faststart"]
    cmd += [str(tmp_out)]
    try:
        _run_progress(cmd, info.get("duration"), progress)
    except MediaError:
        # retry without subtitles/attachments: some subtitle codecs cannot be copied
        cmd2 = [c for c in cmd]
        for flag in ("0:s?", "0:t?"):
            if flag in cmd2:
                i = cmd2.index(flag)
                del cmd2[i - 1:i + 1]
        if "-c:s" in cmd2:
            while "-c:s" in cmd2:
                i = cmd2.index("-c:s")
                del cmd2[i:i + 2]
        if cmd2 == cmd:
            raise
        _run_progress(cmd2, info.get("duration"), progress)
    os.replace(tmp_out, out_path)
    return str(out_path)


def _has_subs_needing_conversion(info: dict) -> bool:
    subs = info.get("subtitles") or []
    return any(s.get("codec") in ("subrip", "ass", "ssa", "webvtt", "text") for s in subs)


# ---------------------------------------------------------------------------
# previews and peaks
# ---------------------------------------------------------------------------
def encode_preview(wav: str | Path, out_m4a: str | Path, bitrate: str = "128k",
                   offset: float = 0.0) -> str:
    """AAC/m4a preview for the review UI (faststart for range streaming).

    ``offset`` (seconds) pads silence at the start so the preview lines up with the source
    video's timeline when the audio stream starts late.
    """
    out_m4a = Path(out_m4a)
    tmp = out_m4a.with_name(out_m4a.stem + ".part.m4a")
    cmd = [config.FFMPEG, "-v", "error", "-y", "-nostdin", "-i", str(wav)]
    if offset > 0.0005:
        cmd += ["-af", f"adelay={int(round(offset * 1000))}:all=1"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ac", "2", "-movflags", "+faststart", str(tmp)]
    _run(cmd, capture=False)
    os.replace(tmp, out_m4a)
    return str(out_m4a)


def compute_peaks(wav: str | Path, rate: int = 50, block_s: float = 60.0) -> dict:
    """Peak envelope of a wav for drawing: ``{rate, peaks}`` with values in 0..1.

    Streams the file block-wise so multi-hour stems do not need to be in RAM.
    """
    import soundfile as sf

    with sf.SoundFile(str(wav)) as f:
        sr = f.samplerate
        hop = max(1, int(round(sr / rate)))
        block = hop * max(1, int(block_s * rate))
        vals: list[np.ndarray] = []
        carry = np.zeros((0,), dtype=np.float32)
        while True:
            x = f.read(block, dtype="float32", always_2d=True)
            if len(x) == 0:
                break
            m = np.abs(x).max(axis=1)
            m = np.concatenate([carry, m])
            n = len(m) // hop
            if n:
                vals.append(m[: n * hop].reshape(n, hop).max(axis=1))
            carry = m[n * hop:]
        if len(carry):
            vals.append(np.array([carry.max()], dtype=np.float32))
    peaks = np.concatenate(vals) if vals else np.zeros(0, np.float32)
    ref = float(np.percentile(peaks, 99.5)) if len(peaks) else 0.0
    ref = max(ref, 1e-4)
    peaks = np.clip(peaks / ref, 0.0, 1.0)
    return {"rate": rate, "peaks": [round(float(v), 3) for v in peaks]}


def ffmpeg_ok() -> bool:
    return tool_version(config.FFMPEG) is not None and tool_version(config.FFPROBE) is not None


if __name__ == "__main__":  # tiny debugging aid
    print(json.dumps({k: v for k, v in probe(sys.argv[1]).items() if k != "raw"}, indent=2))
