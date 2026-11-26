"""Run the Voice Filter pipeline (or a stub) on the synthetic mixes and score it.

  python eval/run_eval.py --preset fast|balanced|best [--mixes mix_00 mix_01 | smoke] [--input mp4|wav]
  python eval/run_eval.py --oracle      # truth segments, keep = female   (upper bound for gating)
  python eval/run_eval.py --random      # truth segments, random keep     (sanity floor)
  python eval/run_eval.py --report-only # rebuild eval/REPORT.md from eval/results/*.json

Pipeline contract (voicefilter/pipeline.py, PLAN.md section 5):
  analyze_file(path, preset, work_dir, progress=None) -> {duration, segments:[...], speakers:[...]}
  render_from_analysis(work_dir, analysis, keep_background=True) -> path to filtered wav

Stub modes gate the TRUE vocal stem (female+male) with the decisions, exactly like the
real renderer gates the separated vocal stem, and add the true background.

Results: eval/results/<preset|oracle|random>.json and the combined eval/REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL_DIR, MIXES_DIR, RESULTS_DIR  # noqa: E402
import metrics as M  # noqa: E402

REPORT = EVAL_DIR / "REPORT.md"


def list_mixes(names: list[str] | None, root: Path = MIXES_DIR) -> list[Path]:
    if names:
        return [root / n for n in names]
    return sorted(p for p in root.glob("mix_*") if (p / "truth.json").exists())


def read_wav(path: Path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return x.T, sr


# ----------------------------------------------------------------------------- stub analyses
def truth_segments(truth: dict) -> list[dict]:
    segs = []
    for i, e in enumerate(truth["events"]):
        segs.append({"id": f"t{i}", "start": e["start"], "end": e["end"], "speaker": e["speaker"],
                     "kind": e["kind"], "label": "female" if e["gender"] == "F" else "male",
                     "p_female": 1.0 if e["gender"] == "F" else 0.0, "confidence": 1.0, "low_confidence": False,
                     "keep": e["gender"] == "F", "override": None})
    return segs


def oracle_analysis(truth: dict) -> dict:
    # Overlapping events: female wins (keep) -> sort so female segments are applied last.
    segs = sorted(truth_segments(truth), key=lambda s: (s["keep"], s["start"]))
    return {"duration": truth["duration"], "segments": segs, "speakers": []}


def oracle_active_analysis(truth: dict) -> dict:
    """Keep exactly the female-active sub-intervals (perfect frame-level gate)."""
    segs = []
    for i, e in enumerate(truth["events"]):
        if e["gender"] != "F":
            continue
        for j, (a, b) in enumerate(e.get("active") or [[e["start"], e["end"]]]):
            segs.append({"id": f"a{i}_{j}", "start": a, "end": b, "speaker": e["speaker"], "kind": e["kind"],
                         "label": "female", "p_female": 1.0, "confidence": 1.0, "low_confidence": False,
                         "keep": True, "override": None})
    return {"duration": truth["duration"], "segments": segs, "speakers": []}


def random_analysis(truth: dict, seed: int = 0) -> dict:
    rng = random.Random(f"random-{truth.get('name')}-{seed}")
    segs = truth_segments(truth)
    for s in segs:
        p = rng.random()
        s.update(p_female=p, keep=p >= 0.5, confidence=abs(p - 0.5) * 2, low_confidence=abs(p - 0.5) < 0.1,
                 label="female" if p >= 0.5 else "male")
    return {"duration": truth["duration"], "segments": segs, "speakers": []}


def stub_render(mix_dir: Path, analysis: dict, keep_background: bool = True) -> np.ndarray:
    F, sr = read_wav(mix_dir / "vocals_female.wav")
    Mm, _ = read_wav(mix_dir / "vocals_male.wav")
    B, _ = read_wav(mix_dir / "background.wav")
    nf = M.n_frames(F.shape[1] / sr)
    keep, _, _ = M.decision_frames(analysis, nf)
    g = M.gain_envelope(keep, F.shape[1], sr)
    return (F + Mm) * g + (B if keep_background else 0.0)


# ----------------------------------------------------------------------------- scoring
def score_mix(mix_dir: Path, analysis: dict, out_audio: np.ndarray | None, out_sr: int | None,
              processing_s: float, outside: str = "remove") -> dict:
    truth = json.loads((mix_dir / "truth.json").read_text())
    f, m = M.truth_frames(truth)
    keep, low, cov = M.decision_frames(analysis, len(f), outside)
    frames = M.frame_metrics(f, m, keep, low, cov)
    evres = M.event_results(truth, keep)
    res = {"name": mix_dir.name, "duration": truth["duration"], "frames": frames, "event_list": evres,
           "events": M.summarize_events(evres), "processing_s": processing_s,
           "rtf": processing_s / truth["duration"] if truth["duration"] else None,
           "n_segments": len(analysis.get("segments", [])), "meta": truth.get("meta", {})}
    if out_audio is not None:
        F, sr = read_wav(mix_dir / "vocals_female.wav")
        Mm, _ = read_wav(mix_dir / "vocals_male.wav")
        B, _ = read_wav(mix_dir / "background.wav")
        mix, _ = read_wav(mix_dir / "mix.wav")
        out = M._as2d(out_audio)
        if out_sr and out_sr != sr:
            import librosa
            out = np.stack([librosa.resample(c, orig_sr=out_sr, target_sr=sr) for c in out])
        res["audio"] = M.audio_metrics(out, F, Mm, B, f, m, sr, mix=mix)
        res["audio"]["length_diff_samples"] = int(out.shape[1] - F.shape[1])
    return res


def run(mode: str, preset: str | None, mixes: list[Path], work_root: Path, input_kind: str,
        keep_background: bool) -> dict:
    per_mix = []
    for md in mixes:
        t0 = time.time()
        out_audio, out_sr = None, None
        if mode in ("oracle", "oracle_active"):
            truth = json.loads((md / "truth.json").read_text())
            analysis = oracle_analysis(truth) if mode == "oracle" else oracle_active_analysis(truth)
            out_audio = stub_render(md, analysis, keep_background)
        elif mode == "random":
            truth = json.loads((md / "truth.json").read_text())
            analysis = random_analysis(truth)
            out_audio = stub_render(md, analysis, keep_background)
        else:
            from voicefilter.pipeline import analyze_file, render_from_analysis
            wd = work_root / preset / md.name
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir(parents=True)
            src = md / ("mix.mp4" if input_kind == "mp4" else "mix.wav")
            analysis = analyze_file(str(src), preset, str(wd), progress=None)
            out_path = render_from_analysis(str(wd), analysis, keep_background=keep_background)
            out_audio, out_sr = read_wav(Path(out_path))
            (wd / "analysis.json").write_text(json.dumps(analysis, indent=1, default=float))
        dt = time.time() - t0
        r = score_mix(md, analysis, out_audio, out_sr or 44100, dt)
        per_mix.append(r)
        fr = r["frames"]
        print(f"{md.name}: acc {fmtp(fr['accuracy'])} male_leak {fmtp(fr['male_leak'])} "
              f"female_loss {fmtp(fr['female_loss'])} overlap_kept {fmtp(fr['overlap_kept'])} "
              f"events {fmtp(r['events'].get('all', {}).get('accuracy'))} rtf {r['rtf']:.2f}", flush=True)
    agg = M.aggregate(per_mix)
    name = preset if mode == "pipeline" else mode
    out = {"name": name, "mode": mode, "preset": preset, "input": input_kind, "keep_background": keep_background,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mixes": [p.name for p in mixes], "aggregate": agg,
           "per_mix": [{k: v for k, v in r.items() if k != "event_list"} for r in per_mix]}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in mixes]
    suffix = "" if set(names) == set(p.name for p in list_mixes(None)) else (
        "_smoke" if names == ["smoke"] else "_subset")
    (RESULTS_DIR / f"{name}{suffix}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def fmtp(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def fmtd(x):
    return "-" if x is None else f"{x:.1f}"


ORDER = ["fast", "balanced", "best", "oracle_active", "oracle", "random"]


def write_report() -> None:
    files = sorted(p for p in RESULTS_DIR.glob("*.json") if p.stem != "gender_models")
    runs = [json.loads(p.read_text()) | {"_file": p.name} for p in files]
    runs.sort(key=lambda r: (r["_file"].endswith(("_smoke.json", "_subset.json")),
                             ORDER.index(r["name"]) if r["name"] in ORDER else 99, r["_file"]))
    L = ["# Voice Filter accuracy report", "",
         "Generated by `eval/run_eval.py` from `eval/results/*.json`. Synthetic mixes built by "
         "`eval/build_dataset.py` from **test speakers only** (see `eval/DATA.md`). 10 ms frames on truth vocal "
         "activity. `oracle` = perfect decisions on truth *event* boundaries with gating of the true vocal stem "
         "(overlap is kept so male voice leaks there, and an event that overlaps the other gender for >20 % of "
         "its frames counts as wrong); `oracle_active` = perfect frame-level gate on the truth active "
         "intervals (upper bound for any gate-only approach); `random` = coin flips on truth events (floor). "
         "Pipeline runs use the real separated stems; `*_smoke` rows are the 10 s smoke mix only.", "",
         "## Frame-level keep/remove", "",
         "| run | mixes | audio min | acc | bal. acc | male leak | female loss | overlap kept | low-conf share | "
         "acc excl. low-conf | coverage | RTF |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        a = r["aggregate"]
        f = a["frames"]
        L.append(f"| {r['_file'][:-5]} | {a['n_mixes']} | {a['total_duration_s'] / 60:.1f} | {fmtp(f.get('accuracy'))} | "
                 f"{fmtp(f.get('balanced_accuracy'))} | {fmtp(f.get('male_leak'))} | {fmtp(f.get('female_loss'))} | "
                 f"{fmtp(f.get('overlap_kept'))} | {fmtp(f.get('low_conf_share_mean_over_mixes'))} | "
                 f"{fmtp(f.get('accuracy_excl_low_conf_mean_over_mixes'))} | "
                 f"{fmtp(f.get('coverage_mean_over_mixes'))} | {fmtd(a.get('rtf'))} |")
    keys = ["all", "kind=speech", "kind=nonverbal", "kind=speech,gender=F", "kind=speech,gender=M",
            "kind=nonverbal,gender=F", "kind=nonverbal,gender=M", "overlap=True", "overlap=False",
            "short(<0.4s)", "breathy"]
    L += ["", "## Event-level accuracy (event correct if >= 80 % of its frames get the right decision)", "",
          "| run | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for r in runs:
        ev = r["aggregate"]["events"]
        L.append(f"| {r['_file'][:-5]} | " + " | ".join(
            f"{fmtp(ev[k]['accuracy'])} (n={ev[k]['n']})" if k in ev else "-" for k in keys) + " |")
    cats = sorted({k for r in runs for k in r["aggregate"]["events"] if k.startswith("category=")})
    if cats:
        L += ["", "## Event accuracy by category", "", "| category | " + " | ".join(r["_file"][:-5] for r in runs) + " |",
              "|---|" + "---|" * len(runs)]
        for c in cats:
            L.append(f"| {c[9:]} | " + " | ".join(
                f"{fmtp(r['aggregate']['events'][c]['accuracy'])} (n={r['aggregate']['events'][c]['n']})"
                if c in r["aggregate"]["events"] else "-" for r in runs) + " |")
    L += ["", "## Audio level (rendered output vs. stems)", "",
          "male residual = energy of the male stem still present in male-only regions (dB re. original, lower "
          "is better); female preserved = same for the female stem in female-only regions (0 dB = intact); "
          "SI-SDR against female+background.", "",
          "| run | male residual dB | male residual in overlap dB | female preserved dB | SI-SDR dB | SI-SDR improvement dB |",
          "|---|---|---|---|---|---|"]
    for r in runs:
        au = r["aggregate"]["audio"]
        L.append(f"| {r['_file'][:-5]} | {fmtd(au.get('male_residual_db_mean'))} | "
                 f"{fmtd(au.get('male_residual_db_overlap_mean'))} | {fmtd(au.get('female_preserved_db_mean'))} | "
                 f"{fmtd(au.get('si_sdr_vs_female_plus_bg_mean'))} | {fmtd(au.get('si_sdr_improvement_mean'))} |")
    L += ["", "## Per mix", ""]
    for r in runs:
        L += [f"### {r['_file'][:-5]}", "", "| mix | dur s | comp | bg | SNR dB | reverb | acc | male leak | "
              "female loss | overlap kept | events | RTF |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for pm in r["per_mix"]:
            mt, fr = pm.get("meta", {}), pm["frames"]
            L.append(f"| {pm['name']} | {pm['duration']:.0f} | {mt.get('composition', '-')} | "
                     f"{mt.get('background', {}).get('type', '-')} | {mt.get('snr_db', '-')} | {mt.get('reverb', '-')} | "
                     f"{fmtp(fr['accuracy'])} | {fmtp(fr['male_leak'])} | {fmtp(fr['female_loss'])} | "
                     f"{fmtp(fr['overlap_kept'])} | {fmtp(pm['events'].get('all', {}).get('accuracy'))} | "
                     f"{fmtd(pm.get('rtf'))} |")
        L.append("")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["fast", "balanced", "best"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--oracle-active", action="store_true",
                    help="perfect frame-level gate: keep exactly the female-active truth intervals")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--mixes", nargs="*", default=None, help="mix directory names (e.g. mix_00 smoke)")
    ap.add_argument("--mixes-root", default=str(MIXES_DIR))
    ap.add_argument("--input", choices=["mp4", "wav"], default="mp4")
    ap.add_argument("--no-background", action="store_true")
    ap.add_argument("--work-dir", default=str(EVAL_DIR / "cache" / "work"))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        mode = ("oracle" if a.oracle else "oracle_active" if a.oracle_active else "random" if a.random
                else "pipeline")
        if mode == "pipeline" and not a.preset:
            ap.error("--preset is required unless --oracle/--oracle-active/--random")
        mixes = list_mixes(a.mixes, Path(a.mixes_root))
        if not mixes:
            ap.error(f"no mixes found in {a.mixes_root}; run eval/build_dataset.py first")
        run(mode, a.preset, mixes, Path(a.work_dir), a.input, not a.no_background)
    write_report()
    print("report:", REPORT)


if __name__ == "__main__":
    main()
