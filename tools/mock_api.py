"""Mock Voice Filter backend for UI development and UI tests.

Implements the HTTP API contract from PLAN.md section 5 with in-memory jobs,
fake progress and a synthetic (but self-consistent) analysis, and serves
``voicefilter/web`` at "/".

    python tools/mock_api.py --port 8799 [--speed 2] [--device cpu]

Upload a file whose name contains "fail" to get a job that ends in the error
state (useful to look at error rendering).
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
import shutil
import socket
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "voicefilter" / "web"

SR = 8000  # low sample rate keeps the synthetic WAVs small
DURATION = 300.0
PEAK_RATE = 50
N_SEGMENTS = 120
LOW_CONF_THRESHOLD = 0.4

PRESETS = [
    {"id": "fast", "name": "Fast", "description": "Light separation and pitch + embedding model. Quick check, a few minutes per hour of video."},
    {"id": "balanced", "name": "Balanced", "description": "RoFormer vocal separation and fused gender models. Recommended."},
    {"id": "best", "name": "Best", "description": "Strongest separation and overlap speech separation. Slowest, most accurate."},
]
PRESET_IDS = {p["id"] for p in PRESETS}

ANALYZE_STAGES = [
    ("extract", "Extracting audio", 0.8),
    ("separate", "Separating vocals", 2.8),
    ("activity", "Detecting vocal events", 1.2),
    ("features", "Analyzing voices", 1.4),
    ("speakers", "Grouping speakers", 0.9),
    ("classify", "Classifying", 0.9),
]
RENDER_STAGES = [
    ("render", "Rendering filtered audio", 1.6),
    ("mux", "Muxing video", 1.4),
]

state: dict = {"speed": 1.0, "device": "cuda", "port": 8799}
JOBS: dict[str, dict] = {}
TMP = Path(tempfile.mkdtemp(prefix="vf-mock-"))
gpu_lock = asyncio.Lock()


# ---------------------------------------------------------------- synthetic data
def make_analysis(seed: int = 7) -> dict:
    rng = random.Random(seed)
    speakers = [
        {"id": "S1", "p": 0.94},
        {"id": "S2", "p": 0.06},
        {"id": "S3", "p": 0.72},
        {"id": "S4", "p": 0.30},
    ]
    weights = [0.34, 0.30, 0.20, 0.16]
    raw = []
    t = 0.0
    cur = 0
    for _ in range(N_SEGMENTS):
        t += rng.uniform(0.15, 1.6)
        if rng.random() < 0.45:  # turn change
            cur = rng.choices(range(4), weights)[0]
        r = rng.random()
        kind = "speech" if r < 0.62 else ("nonverbal" if r < 0.88 else "mixed")
        dur = rng.uniform(0.6, 3.2) if kind != "nonverbal" else rng.uniform(0.25, 1.4)
        raw.append([t, t + dur, cur, kind])
        t += dur
    scale = (DURATION - 1.0) / t
    segments = []
    for i, (a, b, s, kind) in enumerate(raw):
        spk = speakers[s]
        overlap = rng.random() < 0.07
        noise = rng.gauss(0, 1.6 if kind == "nonverbal" else 1.0)
        logit = math.log(spk["p"] / (1 - spk["p"])) + noise
        p = 1 / (1 + math.exp(-logit))
        if overlap:
            p = rng.uniform(0.35, 0.65)
            kind = "mixed" if rng.random() < 0.5 else "speech"
        conf = abs(p - 0.5) * 2
        conf = conf * rng.uniform(0.75, 1.0) + rng.uniform(0.0, 0.1)
        if overlap:
            conf = min(conf, rng.uniform(0.1, 0.45))
        conf = max(0.02, min(0.99, conf))
        label = "overlap" if overlap else ("female" if p >= 0.5 else "male")
        segments.append({
            "id": i,
            "start": round(a * scale + 0.5, 3),
            "end": round(b * scale + 0.5, 3),
            "speaker": spk["id"],
            "kind": kind,
            "label": label,
            "p_female": round(p, 3),
            "confidence": round(conf, 3),
            "low_confidence": conf < LOW_CONF_THRESHOLD,
            "keep": label in ("female", "overlap"),
            "override": None,
        })
    spk_out = []
    for spk in speakers:
        segs = [x for x in segments if x["speaker"] == spk["id"]]
        tot = sum(x["end"] - x["start"] for x in segs)
        pf = sum(x["p_female"] * (x["end"] - x["start"]) for x in segs) / max(tot, 1e-6)
        spk_out.append({
            "id": spk["id"],
            "p_female": round(pf, 3),
            "label": "female" if pf >= 0.5 else "male",
            "total_dur": round(tot, 2),
            "n_segments": len(segs),
            "override": None,
        })
    return {"duration": DURATION, "segments": segments, "speakers": spk_out,
            "low_conf_threshold": LOW_CONF_THRESHOLD}


def effective_keep(seg: dict, speakers: dict) -> bool:
    if seg["override"]:
        return seg["override"] == "keep"
    spk = speakers.get(seg["speaker"])
    if spk and spk["override"]:
        return spk["override"] == "keep"
    return seg["keep"]


BASE_ANALYSIS = make_analysis()
N = int(SR * DURATION)
_tt = np.arange(N) / SR


def synth_audio():
    rng = np.random.default_rng(1)
    # background: slow chord + soft noise
    bg = 0.06 * (np.sin(2 * np.pi * 110 * _tt) + 0.7 * np.sin(2 * np.pi * 164.8 * _tt)
                 + 0.5 * np.sin(2 * np.pi * 220 * _tt)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * _tt))
    bg += 0.01 * rng.standard_normal(N)
    voc = np.zeros(N)
    for seg in BASE_ANALYSIS["segments"]:
        a, b = int(seg["start"] * SR), int(seg["end"] * SR)
        tt = _tt[a:b] - seg["start"]
        f0 = 240 if seg["label"] == "female" else (125 if seg["label"] == "male" else 180)
        f = f0 * (1 + 0.04 * np.sin(2 * np.pi * 5 * tt))
        ph = 2 * np.pi * np.cumsum(f) / SR
        tone = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
        env = np.minimum(1, np.minimum(tt, (b - a) / SR - tt) / 0.03)
        env *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(1.5, 4) * tt))
        voc[a:b] = 0.25 * tone * env
    return bg, voc


BG, VOC = synth_audio()


def write_wav(path: Path, x: np.ndarray):
    sf.write(str(path), np.clip(x, -1, 1).astype(np.float32), SR, subtype="PCM_16")


write_wav(TMP / "original.wav", BG + VOC)
write_wav(TMP / "vocals.wav", VOC)


def make_peaks() -> list[float]:
    hop = SR // PEAK_RATE
    frames = VOC[: (N // hop) * hop].reshape(-1, hop)
    pk = np.abs(frames).max(axis=1)
    pk = pk / max(pk.max(), 1e-9)
    return [round(float(v), 3) for v in pk]


PEAKS = make_peaks()


def render_preview(job: dict):
    an = job["analysis"]
    spk = {s["id"]: s for s in an["speakers"]}
    gain = np.zeros(N)
    for seg in an["segments"]:
        if effective_keep(seg, spk):
            gain[int(seg["start"] * SR): int(seg["end"] * SR)] = 1.0
    out = VOC * gain + (BG if job["keep_background"] else 0)
    path = job["dir"] / f"preview-{job['preview_version']}.wav"
    write_wav(path, out)
    old = job.get("preview_path")
    job["preview_path"] = path
    if old and old != path:
        old.unlink(missing_ok=True)


# ---------------------------------------------------------------- job simulation
def public(job: dict) -> dict:
    return {k: job[k] for k in ("id", "filename", "preset", "keep_background", "state", "stage",
                                "progress", "message", "error", "duration", "created", "has_output")}


async def run_stages(job: dict, stages, base: float, span: float):
    total = sum(s[2] for s in stages)
    done = 0.0
    for key, label, secs in stages:
        job["stage"] = key
        steps = max(1, int(secs * 10 / state["speed"]))
        for i in range(steps):
            frac = (i + 1) / steps
            job["message"] = f"{label} ({int(frac * 100)}%)" if secs > 1 else label
            job["progress"] = round(base + span * (done + secs * frac) / total, 4)
            await asyncio.sleep(0.1)
            if job.get("fail") and key == "separate" and frac > 0.4:
                raise RuntimeError("CUDA out of memory while separating vocals (needed 3.1 GB, 1.2 GB free). "
                                   "Close other GPU apps or try the Fast preset.")
        done += secs


async def analyze(job: dict):
    try:
        async with gpu_lock:
            job.update(state="analyzing", progress=0.0, message="Starting")
            await run_stages(job, ANALYZE_STAGES, 0.0, 1.0)
            job["analysis"] = make_analysis()
            job["duration"] = DURATION
            await asyncio.to_thread(render_preview, job)
            job.update(state="review", stage="review", progress=1.0, message="Ready for review")
        if job["auto_export"]:
            await export(job)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        job.update(state="error", stage="error", error=str(e), message="Failed")


async def export(job: dict):
    async with gpu_lock:
        job.update(state="rendering", stage="render", progress=0.0, message="Rendering", has_output=False)
        await run_stages(job, RENDER_STAGES, 0.0, 1.0)
        out = job["dir"] / "output.mp4"
        out.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"mock voicefilter output\n" * 40)
        job["output"] = out
        job.update(state="done", stage="done", progress=1.0, message="Export finished", has_output=True)


# ---------------------------------------------------------------- API
app = FastAPI(title="Voice Filter mock API")


def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def need_analysis(job: dict) -> dict:
    if not job.get("analysis"):
        raise HTTPException(409, "Analysis is not ready yet")
    return job["analysis"]


def lan_urls(port: int) -> list[str]:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return [f"http://{ip}:{port}" for ip in ips] or [f"http://localhost:{port}"]


@app.get("/api/info")
def info():
    gpu = state["device"] != "cpu"
    return {"version": "0.0.0-mock", "device": "cuda" if gpu else "cpu",
            "gpu_name": "NVIDIA GeForce RTX 3080 (mock)" if gpu else None,
            "lan_urls": lan_urls(state["port"]), "presets": PRESETS}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), preset: str = Form("balanced"),
                     keep_background: str = Form("true"), auto_export: str = Form("false")):
    if preset not in PRESET_IDS:
        raise HTTPException(400, f"Unknown preset '{preset}'")
    job_id = uuid.uuid4().hex[:12]
    d = TMP / job_id
    d.mkdir()
    # stream the upload to a temp file, then throw it away (mock)
    with tempfile.NamedTemporaryFile(dir=d, delete=True) as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    name = file.filename or "video.mp4"
    job = {
        "id": job_id, "filename": name, "preset": preset,
        "keep_background": keep_background.lower() != "false",
        "auto_export": auto_export.lower() == "true",
        "state": "queued", "stage": "queued", "progress": 0.0, "message": "Waiting for the GPU",
        "error": None, "duration": None, "created": time.time(), "has_output": False,
        "dir": d, "fail": "fail" in name.lower(), "preview_version": 0,
    }
    JOBS[job_id] = job
    job["task"] = asyncio.create_task(analyze(job))
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return [public(j) for j in sorted(JOBS.values(), key=lambda j: -j["created"])]


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    return public(get_job(job_id))


@app.get("/api/jobs/{job_id}/analysis")
def analysis(job_id: str):
    return need_analysis(get_job(job_id))


@app.get("/api/jobs/{job_id}/peaks")
def peaks(job_id: str):
    need_analysis(get_job(job_id))
    return {"rate": PEAK_RATE, "peaks": PEAKS}


@app.get("/api/jobs/{job_id}/audio/{kind}")
def audio(job_id: str, kind: str):
    job = get_job(job_id)
    need_analysis(job)
    if kind == "original":
        path = TMP / "original.wav"
    elif kind == "vocals":
        path = TMP / "vocals.wav"
    elif kind == "preview":
        path = job["preview_path"]
    else:
        raise HTTPException(404, "Unknown audio kind")
    return FileResponse(path, media_type="audio/wav")


class Overrides(BaseModel):
    segments: dict[str, str | None] = {}
    speakers: dict[str, str | None] = {}


@app.post("/api/jobs/{job_id}/overrides")
async def overrides(job_id: str, body: Overrides):
    job = get_job(job_id)
    an = need_analysis(job)
    segs = {str(s["id"]): s for s in an["segments"]}
    spks = {str(s["id"]): s for s in an["speakers"]}
    for table, src, what in ((segs, body.segments, "segment"), (spks, body.speakers, "speaker")):
        for k, v in src.items():
            if k not in table:
                raise HTTPException(400, f"Unknown {what} '{k}'")
            if v not in (None, "keep", "remove"):
                raise HTTPException(400, f"Bad override value '{v}'")
            table[k]["override"] = v
    job["preview_version"] += 1
    asyncio.get_running_loop().run_in_executor(None, render_preview, job)
    return an


@app.post("/api/jobs/{job_id}/export")
async def start_export(job_id: str):
    job = get_job(job_id)
    if job["state"] not in ("review", "done"):
        raise HTTPException(409, f"Cannot export while the job is {job['state']}")
    job.update(state="rendering", stage="queued", progress=0.0, message="Waiting for the GPU")
    job["task"] = asyncio.create_task(export(job))
    return public(job)


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = get_job(job_id)
    if not job.get("has_output"):
        raise HTTPException(409, "No output yet; export first")
    p = Path(job["filename"])
    return FileResponse(job["output"], media_type="video/mp4",
                        filename=f"{p.stem}.voicefilter{p.suffix or '.mp4'}")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = get_job(job_id)
    if job.get("task"):
        job["task"].cancel()
    JOBS.pop(job_id, None)
    shutil.rmtree(job["dir"], ignore_errors=True)
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed multiplier")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args()
    state.update(speed=a.speed, device=a.device, port=a.port)
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
