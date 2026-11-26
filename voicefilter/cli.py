"""Headless processing.

    python -m voicefilter.cli process IN.mp4 -o OUT.mp4 --preset balanced
        [--no-background] [--analysis-out a.json] [--work-dir D] [--keep-work]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import config


def _progress():
    state = {"last": "", "t": 0.0}

    def cb(stage: str, frac: float, msg: str) -> None:
        now = time.time()
        line = f"[{frac * 100:5.1f}%] {stage:<10} {msg}"
        if stage != state["last"] or now - state["t"] > 2.0:
            print(line, flush=True)
            state["last"], state["t"] = stage, now

    return cb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicefilter.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process", help="analyze + render + export one file")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--preset", default="balanced")
    p.add_argument("--no-background", action="store_true", help="drop the non-vocal background")
    p.add_argument("--analysis-out", help="write the analysis JSON here")
    p.add_argument("--work-dir", help="keep intermediate files here (default: temp, deleted)")
    a = ap.parse_args(argv)

    from . import pipeline

    config.ensure_dirs()
    src = Path(a.input)
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2
    tmp = None
    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix="vf-cli-", dir=str(config.TMP_DIR)))
    if not a.work_dir:
        tmp = work
    try:
        t0 = time.time()
        an = pipeline.process_file(src, a.output, a.preset, not a.no_background, work, _progress())
        if a.analysis_out:
            Path(a.analysis_out).write_text(json.dumps(an, indent=1), encoding="utf-8")
        n_keep = sum(1 for s in an["segments"] if s["keep"])
        print(f"done in {time.time() - t0:.1f} s: {len(an['segments'])} segments, {n_keep} kept, "
              f"{len(an['speakers'])} speakers -> {a.output}", flush=True)
        print("timings: " + json.dumps(an.get("timings", {})), flush=True)
        for n in an.get("notes", []):
            print("note: " + n, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
