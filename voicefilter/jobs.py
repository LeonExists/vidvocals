"""Job store (DATA/jobs/<id>/job.json) and the single FIFO worker thread.

States: queued -> analyzing -> review -> rendering -> done | error.
``auto_export`` jobs go straight from analysis to rendering.

The GPU worker runs one job step at a time (analysis or export).  Preview renders after
overrides run in a separate light thread (CPU only, debounced) and replace preview.m4a
atomically.  On start-up, jobs found in ``queued`` are re-queued, jobs found in
``analyzing``/``rendering`` become ``error`` ("interrupted, please retry").
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import config

PUBLIC_FIELDS = ("id", "filename", "preset", "keep_background", "state", "stage", "progress",
                 "message", "error", "duration", "created", "has_output")

STAGE_TEXT = {
    "extract": "Extracting audio", "separate": "Separating vocals", "activity": "Detecting vocal events",
    "features": "Analyzing voices", "speakers": "Grouping speakers", "overlap": "Checking overlapping voices",
    "preview": "Preparing review", "rendering": "Rendering filtered audio", "export": "Writing video",
    "done": "Done",
}


def sanitize_filename(name: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", name)
    name = name.strip(" .") or "upload"
    stem, ext = os.path.splitext(name)
    if len(stem) > 120:
        stem = stem[:120]
    if stem.upper() in {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}:
        stem = "_" + stem
    return stem + ext[:12]


class JobStore:
    def __init__(self, root: Path | None = None, pipeline: Any = None, start_worker: bool = True):
        self.root = Path(root or config.JOBS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        if pipeline is None:
            from . import pipeline as pipeline_mod

            pipeline = pipeline_mod
        self.pipeline = pipeline
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._preview_lock = threading.Lock()
        self._preview_pending: dict[str, float] = {}
        self._preview_cv = threading.Condition()
        self._stop = threading.Event()
        self._load_existing()
        self.worker: Optional[threading.Thread] = None
        self.preview_thread: Optional[threading.Thread] = None
        if start_worker:
            self.start()

    # ------------------------------------------------------------ persistence
    def _dir(self, jid: str) -> Path:
        return self.root / jid

    def _save(self, job: dict) -> None:
        d = self._dir(job["id"])
        if not d.exists():
            return
        tmp = d / "job.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
        os.replace(tmp, d / "job.json")

    def _load_existing(self) -> None:
        requeue = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            p = d / "job.json"
            if not p.is_file():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    job = json.load(f)
            except Exception:
                continue
            if job.get("state") == "queued":
                requeue.append(job)
            elif job.get("state") in ("analyzing", "rendering"):
                was = job["state"]
                job.update(state="error", stage="error", error="interrupted, please retry",
                           message="Interrupted (server restarted)")
                if was == "rendering" and (d / "analysis.json").exists():
                    # analysis is still valid: back to review so the user can export again
                    job.update(state="review", stage="review", error=None,
                               message="Export interrupted (server restarted); export again")
            self.jobs[job["id"]] = job
            self._save(job)
        for job in sorted(requeue, key=lambda j: j.get("created", 0)):
            self.q.put((job["id"], job.get("next", "analyze")))

    # ------------------------------------------------------------ public api
    def public(self, job: dict) -> dict:
        return {k: job.get(k) for k in PUBLIC_FIELDS}

    def list(self) -> list[dict]:
        with self.lock:
            return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: -j.get("created", 0))]

    def get(self, jid: str) -> Optional[dict]:
        with self.lock:
            return self.jobs.get(jid)

    def job_dir(self, jid: str) -> Path:
        return self._dir(jid)

    def new_job_dir(self) -> tuple[str, Path]:
        jid = uuid.uuid4().hex[:12]
        d = self._dir(jid)
        d.mkdir(parents=True)
        return jid, d

    def create(self, jid: str, filename: str, input_path: Path, preset: str,
               keep_background: bool = True, auto_export: bool = False) -> dict:
        job = {
            "id": jid, "filename": filename, "preset": preset, "keep_background": bool(keep_background),
            "auto_export": bool(auto_export), "state": "queued", "stage": "queued", "progress": 0.0,
            "message": "Waiting in queue", "error": None, "duration": None, "created": time.time(),
            "has_output": False, "input": input_path.name, "output": None, "next": "analyze",
            "preview_version": 0,
        }
        with self.lock:
            self.jobs[jid] = job
            self._save(job)
        self.q.put((jid, "analyze"))
        return job

    def update(self, jid: str, **kw) -> Optional[dict]:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None:
                return None
            job.update(kw)
            self._save(job)
            return job

    def delete(self, jid: str) -> bool:
        with self.lock:
            job = self.jobs.pop(jid, None)
        if job is None:
            return False
        job["deleted"] = True
        shutil.rmtree(self._dir(jid), ignore_errors=True)
        return True

    def analysis(self, jid: str) -> Optional[dict]:
        p = self._dir(jid) / "analysis.json"
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save_analysis(self, jid: str, analysis: dict) -> None:
        p = self._dir(jid) / "analysis.json"
        tmp = p.with_name("analysis.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, p)

    def apply_overrides(self, jid: str, seg_ov: dict, spk_ov: dict) -> dict:
        """Validate + apply overrides; schedules a preview render.  Raises KeyError/ValueError."""
        with self.lock:
            an = self.analysis(jid)
            if an is None:
                raise LookupError("analysis is not ready yet")
            segs = {str(s["id"]): s for s in an["segments"]}
            spks = {str(s["id"]): s for s in an["speakers"]}
            for table, src, what in ((segs, seg_ov or {}, "segment"), (spks, spk_ov or {}, "speaker")):
                for k, v in src.items():
                    if str(k) not in table:
                        raise KeyError(f"unknown {what} {k!r}")
                    if v not in (None, "keep", "remove"):
                        raise ValueError(f"bad override value {v!r} (use keep, remove or null)")
            for table, src in ((segs, seg_ov or {}), (spks, spk_ov or {})):
                for k, v in src.items():
                    table[str(k)]["override"] = v
            self.save_analysis(jid, an)
            job = self.jobs.get(jid)
            if job is not None:
                job["preview_version"] = int(job.get("preview_version", 0)) + 1
                self._save(job)
        self.schedule_preview(jid)
        return an

    def request_export(self, jid: str) -> dict:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None:
                raise LookupError("job not found")
            if job["state"] not in ("review", "done") and not (job["state"] == "error" and self.analysis(jid)):
                raise RuntimeError(f"cannot export while the job is {job['state']}")
            job.update(state="rendering", stage="queued", progress=0.0, message="Waiting in queue",
                       error=None, has_output=False, next="export")
            self._save(job)
        self.q.put((jid, "export"))
        return job

    # ------------------------------------------------------------ worker
    def start(self) -> None:
        if self.worker is None:
            self.worker = threading.Thread(target=self._run, name="vf-worker", daemon=True)
            self.worker.start()
        if self.preview_thread is None:
            self.preview_thread = threading.Thread(target=self._preview_loop, name="vf-preview", daemon=True)
            self.preview_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.q.put(("", "stop"))
        with self._preview_cv:
            self._preview_cv.notify_all()

    def _progress_cb(self, jid: str, base: float = 0.0, span: float = 1.0) -> Callable[[str, float, str], None]:
        last = {"t": 0.0}

        def cb(stage: str, frac: float, message: str) -> None:
            job = self.jobs.get(jid)
            if job is None or job.get("deleted"):
                raise InterruptedError("job deleted")
            now = time.time()
            job["stage"] = stage
            job["progress"] = round(base + span * float(frac), 4)
            label = STAGE_TEXT.get(stage, stage)
            if message and message != stage:
                job["message"] = message[0].upper() + message[1:]
            else:
                job["message"] = label
            if now - last["t"] > 1.0:
                last["t"] = now
                with self.lock:
                    if jid in self.jobs:
                        self._save(job)

        return cb

    def _run(self) -> None:
        while not self._stop.is_set():
            jid, what = self.q.get()
            if what == "stop":
                break
            job = self.get(jid)
            if job is None or job.get("deleted"):
                continue
            try:
                if what == "analyze":
                    self._analyze(job)
                    if job.get("auto_export") and not job.get("deleted"):
                        self.update(jid, state="rendering", stage="queued", next="export")
                        self._export(job)
                elif what == "export":
                    self._export(job)
            except InterruptedError:
                pass
            except Exception as exc:  # noqa: BLE001
                if not job.get("deleted"):
                    tb = traceback.format_exc()
                    try:
                        (self._dir(jid) / "error.log").write_text(tb, encoding="utf-8")
                    except OSError:
                        pass
                    msg = str(exc) or type(exc).__name__
                    self.update(jid, state="error", stage="error", error=msg, message="Failed")
            finally:
                if job.get("deleted"):  # deleted while running: files may have been in use
                    shutil.rmtree(self._dir(jid), ignore_errors=True)

    def _analyze(self, job: dict) -> None:
        jid = job["id"]
        d = self._dir(jid)
        self.update(jid, state="analyzing", stage="extract", progress=0.0, message="Starting analysis",
                    error=None)
        analysis = self.pipeline.analyze_file(d / job["input"], job["preset"], d,
                                              progress=self._progress_cb(jid))
        if job.get("deleted"):
            raise InterruptedError
        self.update(jid, state="review", stage="review", progress=1.0, message="Ready for review",
                    duration=analysis.get("duration"), next=None)

    def _export(self, job: dict) -> None:
        jid = job["id"]
        d = self._dir(jid)
        an = self.analysis(jid)
        if an is None:
            raise RuntimeError("analysis missing, please re-run")
        cb = self._progress_cb(jid)
        self.update(jid, state="rendering", stage="rendering", progress=0.0, message="Rendering filtered audio")
        wav = self.pipeline.render_from_analysis(d, an, job.get("keep_background", True),
                                                 progress=lambda f: cb("rendering", 0.6 * f, "Rendering filtered audio"))
        src = d / job["input"]
        from .media import output_extension

        ext = output_extension(src)
        out = d / ("output" + ext)
        self.pipeline.export_video(src, wav, out, progress=lambda f: cb("export", 0.6 + 0.4 * f, "Writing video"))
        try:
            os.remove(wav)
        except OSError:
            pass
        self.update(jid, state="done", stage="done", progress=1.0, message="Export finished", has_output=True,
                    output=out.name, next=None)

    # ------------------------------------------------------------ previews
    def schedule_preview(self, jid: str) -> None:
        with self._preview_cv:
            self._preview_pending[jid] = time.time() + float(config.setting("preview_debounce_s"))
            self._preview_cv.notify_all()

    def _preview_loop(self) -> None:
        while not self._stop.is_set():
            with self._preview_cv:
                if not self._preview_pending:
                    self._preview_cv.wait(timeout=1.0)
                    continue
                jid, due = min(self._preview_pending.items(), key=lambda kv: kv[1])
                wait = due - time.time()
                if wait > 0:
                    self._preview_cv.wait(timeout=wait)
                    continue
                del self._preview_pending[jid]
            job = self.get(jid)
            an = self.analysis(jid)
            if job is None or an is None:
                continue
            try:
                with self._preview_lock:
                    self.pipeline.render_preview(self._dir(jid), an, job.get("keep_background", True))
            except Exception:
                try:
                    (self._dir(jid) / "preview_error.log").write_text(traceback.format_exc(), encoding="utf-8")
                except OSError:
                    pass

    def cleanup_tmp(self) -> None:
        for d in self.root.iterdir() if self.root.exists() else []:
            for p in d.glob("*.part*"):
                try:
                    p.unlink()
                except OSError:
                    pass
        # stale temp entries of our own CLI / pipeline runs (never other processes' fresh files)
        cutoff = time.time() - 6 * 3600
        try:
            for p in config.TMP_DIR.iterdir():
                if not p.name.startswith("vf-"):
                    continue
                if p.stat().st_mtime > cutoff:
                    continue
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
        except OSError:
            pass
