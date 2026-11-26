"""FastAPI app: HTTP API of PLAN.md section 5 + static web UI at "/"."""
from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, presets
from .jobs import JobStore, sanitize_filename


# ---------------------------------------------------------------------------
# LAN addresses
# ---------------------------------------------------------------------------
def lan_ipv4() -> list[str]:
    ips: list[str] = []

    def add(ip: str) -> None:
        if ip and not ip.startswith("127.") and not ip.startswith("169.254.") and ip != "0.0.0.0" \
                and ip not in ips:
            ips.append(ip)

    try:  # UDP connect trick: no packet is sent, gives the default-route interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.254.254.254", 1))
            add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return ips


def lan_urls(port: int) -> list[str]:
    return [f"http://{ip}:{port}" for ip in lan_ipv4()]


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
class Overrides(BaseModel):
    segments: dict[str, Optional[str]] = {}
    speakers: dict[str, Optional[str]] = {}


def _bool(v: Any, default: bool) -> bool:
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def create_app(store: Optional[JobStore] = None, pipeline: Any = None, start_worker: bool = True) -> FastAPI:
    state: dict[str, Any] = {"store": store}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.ensure_dirs()
        if state["store"] is None:
            state["store"] = JobStore(pipeline=pipeline, start_worker=start_worker)
        yield
        st: JobStore = state["store"]
        st.stop()
        st.cleanup_tmp()

    app = FastAPI(title="Voice Filter", version=config.VERSION, lifespan=lifespan)
    app.state.vf = state

    def S() -> JobStore:
        if state["store"] is None:
            state["store"] = JobStore(pipeline=pipeline, start_worker=start_worker)
        return state["store"]

    def job_or_404(jid: str) -> dict:
        job = S().get(jid)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    def analysis_or_409(jid: str) -> dict:
        an = S().analysis(jid)
        if an is None:
            raise HTTPException(409, "Analysis is not ready yet")
        return an

    @app.get("/api/info")
    def info():
        d = config.device_info()
        return {"version": config.VERSION, "device": d["device"], "gpu_name": d["gpu_name"],
                "lan_urls": lan_urls(config.PORT), "presets": presets.preset_list()}

    @app.post("/api/jobs")
    async def create_job(request: Request, file: UploadFile = File(...), preset: str = Form("balanced"),
                         keep_background: str = Form("true"), auto_export: str = Form("false")):
        if preset not in presets.presets():
            raise HTTPException(400, f"Unknown preset '{preset}'")
        name = sanitize_filename(file.filename or "video.mp4")
        limit = int(config.setting("max_upload_bytes") or 0)
        jid, d = S().new_job_dir()
        ext = Path(name).suffix.lower() or ".bin"
        dest = d / ("input" + ext)
        size = 0
        try:
            with open(dest, "wb") as f:
                while True:
                    chunk = await file.read(1 << 22)
                    if not chunk:
                        break
                    size += len(chunk)
                    if limit and size > limit:
                        raise HTTPException(413, f"File too large (limit {limit // 2**20} MB)")
                    f.write(chunk)
        except BaseException:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            raise
        finally:
            await file.close()
        if size == 0:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, "Empty upload")
        S().create(jid, name, dest, preset, _bool(keep_background, True), _bool(auto_export, False))
        return {"id": jid}

    @app.get("/api/jobs")
    def list_jobs():
        return S().list()

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return S().public(job_or_404(jid))

    @app.get("/api/jobs/{jid}/analysis")
    def get_analysis(jid: str):
        job_or_404(jid)
        return analysis_or_409(jid)

    @app.get("/api/jobs/{jid}/peaks")
    def get_peaks(jid: str):
        job_or_404(jid)
        p = S().job_dir(jid) / "peaks.json"
        if not p.exists():
            raise HTTPException(409, "Analysis is not ready yet")
        return FileResponse(p, media_type="application/json")

    @app.get("/api/jobs/{jid}/audio/{kind}")
    def get_audio(jid: str, kind: str):
        job_or_404(jid)
        if kind not in ("original", "vocals", "preview"):
            raise HTTPException(404, "Unknown audio kind")
        p = S().job_dir(jid) / f"{kind}.m4a"
        if not p.exists():
            raise HTTPException(409, "Audio is not ready yet")
        return FileResponse(p, media_type="audio/mp4", headers={"Cache-Control": "no-cache"})

    @app.post("/api/jobs/{jid}/overrides")
    def post_overrides(jid: str, body: Overrides):
        job_or_404(jid)
        try:
            return S().apply_overrides(jid, body.segments, body.speakers)
        except LookupError as exc:
            if isinstance(exc, KeyError):
                raise HTTPException(400, str(exc).strip("'\""))
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/jobs/{jid}/export")
    def post_export(jid: str):
        job_or_404(jid)
        try:
            return S().public(S().request_export(jid))
        except LookupError:
            raise HTTPException(404, "Job not found")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/jobs/{jid}/download")
    def download(jid: str):
        job = job_or_404(jid)
        if not job.get("has_output") or not job.get("output"):
            raise HTTPException(409, "No output yet; export first")
        p = S().job_dir(jid) / job["output"]
        if not p.exists():
            raise HTTPException(410, "Output file is missing; export again")
        stem = Path(job["filename"]).stem or "video"
        name = f"{stem}.voicefilter{p.suffix}"
        ascii_name = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
        cd = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"
        import mimetypes

        mt = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return FileResponse(p, media_type=mt, headers={"Content-Disposition": cd})

    @app.delete("/api/jobs/{jid}")
    def delete_job(jid: str):
        job_or_404(jid)
        S().delete(jid)
        return {"ok": True}

    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):  # simple error JSON
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    if config.WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
    else:
        @app.get("/")
        def no_ui():
            return Response("Voice Filter API is running (web UI not found).", media_type="text/plain")

    return app


def banner(host: str, port: int) -> str:
    d = config.device_info()
    dev = f"CUDA GPU: {d['gpu_name']} ({d['vram_gb']} GB)" if d["device"] == "cuda" else "CPU (no CUDA GPU found)"
    lines = ["", "=" * 64, f"  Voice Filter {config.VERSION}", "=" * 64,
             f"  Open on this PC:      http://localhost:{port}"]
    if host in ("0.0.0.0", ""):
        urls = lan_urls(port)
        if urls:
            lines.append("  Open on your phone (same WiFi):")
            lines += [f"                        {u}" for u in urls]
        else:
            lines.append("  (no LAN address found; phone access needs a network connection)")
    elif not host.startswith("127."):
        lines.append(f"  Network address:     http://{host}:{port}")
    lines += [f"  Device:               {dev}", f"  Data folder:          {config.DATA_DIR}",
              "  Stop with Ctrl+C", "=" * 64, ""]
    return "\n".join(lines)


def main() -> None:
    import uvicorn

    config.ensure_dirs()
    host, port = config.HOST, config.PORT
    print(banner(host, port), flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="warning", timeout_keep_alive=30)
