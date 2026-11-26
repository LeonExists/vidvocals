"""Fast re-scoring of decision parameters on cached features (no model runs).

Needs one normal pipeline run per mix first (``eval/run_eval.py --oracle-separation --preset P``
or a real-separation run), which leaves the per-window features in
``eval/cache/work/<oracle_sep|real_sep>/<preset>/<mix>/cache/``.  Then:

  python eval/tune.py --preset fast                                  # dev split, default params
  python eval/tune.py --preset fast --set speaker_k=0.5 --set decision_threshold=0.55
  python eval/tune.py --preset fast --grid decision_threshold=0.4,0.5,0.6 --grid unsure_policy=model,remove
  python eval/tune.py --preset fast --split test                     # only for the final numbers
  python eval/tune.py --preset best --compute-missing                # run ConvTasNet on new pieces
  python eval/tune.py --preset fast --act merge_gap=0.1              # activity change (new windows computed)

Scores with eval/metrics.py exactly like run_eval.py (frame decisions from the segments).
``obj`` = (w * male frames kept + female frames removed) / single-gender frames, w = --leak-weight.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL_DIR, MIXES_DIR  # noqa: E402
import metrics as M  # noqa: E402

WORK = EVAL_DIR / "cache" / "work"
SPLITS = json.loads((EVAL_DIR / "splits.json").read_text())["tuning"]


def split_mixes(name: str) -> list[str]:
    if name == "all":
        return sorted(SPLITS["dev"] + SPLITS["test"])
    if name in SPLITS:
        return list(SPLITS[name])
    return name.split(",")


class Context:
    """Cached features of one mix, loaded once and re-decided many times."""

    def __init__(self, mix: str, preset: str, sep: str = "oracle", compute_missing: bool = False,
                 activity_overrides: dict | None = None):
        from voicefilter import extract, features, pipeline, presets

        self.mix, self.preset = mix, preset
        self.work = WORK / ("oracle_sep" if sep == "oracle" else "real_sep") / preset / mix
        if not (self.work / "cache" / "stamp.json").exists():
            raise FileNotFoundError(f"no cached features in {self.work}; run eval/run_eval.py first")
        self.truth = json.loads((MIXES_DIR / mix / "truth.json").read_text())
        self.P = presets.get_preset(preset)
        self.x16 = features.load_mono_16k(self.work / "vocals.wav")
        self.notes: list[str] = []
        self.compute_missing = compute_missing
        self.f, self.m = M.truth_frames(self.truth)
        self.load(activity_overrides)
        self.resolver = None
        if self.P.get("overlap") == "separate":
            self.resolver = pipeline.OverlapSeparator(self.work, self.x16, self.P, "cpu", self.notes, {},
                                                      allow_compute=compute_missing)

    def load(self, activity_overrides: dict | None = None) -> None:
        from voicefilter import extract

        self.activity_overrides = activity_overrides
        self.ex = extract.extract(self.work, self.x16, self.P, "cpu", lambda f, m="": None, self.notes, {},
                                  activity_overrides=activity_overrides)

    def analysis(self, params: dict) -> dict:
        from voicefilter import decide, pipeline

        prm = {**pipeline.decision_params(self.P), **params}
        res = decide.decide(self.ex, self.P, prm, resolver=self.resolver, notes=self.notes)
        return {"duration": self.truth["duration"], "segments": res["segments"], "speakers": res["speakers"],
                "_res": res}

    def score(self, params: dict) -> dict:
        an = self.analysis(params)
        keep, low, cov = M.decision_frames(an, len(self.f))
        fr = M.frame_metrics(self.f, self.m, keep, low, cov)
        ev = M.event_results(self.truth, keep)
        return {"name": self.mix, "duration": self.truth["duration"], "frames": fr, "event_list": ev,
                "events": M.summarize_events(ev), "processing_s": 0.0, "meta": self.truth.get("meta", {}),
                "_analysis": an}


def objective(agg: dict, w: float = 1.5) -> float:
    fr = agg["frames"]
    fo, mo = fr["frames_female_only"], fr["frames_male_only"]
    return (w * (fr["male_leak"] or 0) * mo + (fr["female_loss"] or 0) * fo) / max(1, fo + mo)


def summarize(per_mix: list[dict], w: float = 1.5) -> dict:
    agg = M.aggregate(per_mix)
    fr, er = agg["frames"], agg["errors"]
    frac = lambda k: (er[k]["wrong_s"] / er[k]["time_s"]) if k in er and er[k]["time_s"] else float("nan")  # noqa: E731
    lowsh = np.mean([r["frames"]["low_conf_share"] or 0 for r in per_mix])
    return {"acc": fr["accuracy"], "leak": fr["male_leak"], "loss": fr["female_loss"], "obj": objective(agg, w),
            "ev": agg["events"]["all"]["accuracy"], "ovl_kept": fr["overlap_kept"], "low": lowsh,
            "nv_F": frac("F|kind=nonverbal"), "nv_M": frac("M|kind=nonverbal"),
            "sp_F": frac("F|kind=speech"), "sp_M": frac("M|kind=speech"),
            "br_F": frac("F|breathy/whisper"), "br_M": frac("M|breathy/whisper"),
            "sh_F": frac("F|short(<0.4s)"), "sh_M": frac("M|short(<0.4s)"),
            "ov_F": frac("F|overlap"), "ov_M": frac("M|overlap"), "agg": agg}


COLS = ["acc", "leak", "loss", "obj", "ev", "low", "nv_F", "nv_M", "sp_F", "sp_M", "br_F", "br_M", "sh_F", "ov_M"]


def fmt_row(label: str, s: dict) -> str:
    return f"{label:<48} " + " ".join(f"{100 * s[c]:5.1f}" if c in s and s[c] is not None and s[c] == s[c] else "    -" for c in COLS)


def header() -> str:
    return f"{'params':<48} " + " ".join(f"{c:>5}" for c in COLS)


def parse_val(v: str):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=["fast", "balanced", "best"])
    ap.add_argument("--split", default="dev", help="dev | test | all | comma-separated mix names")
    ap.add_argument("--sep", default="oracle", choices=["oracle", "real"])
    ap.add_argument("--set", action="append", default=[], help="key=value decision parameter")
    ap.add_argument("--grid", action="append", default=[], help="key=v1,v2,... (cartesian product)")
    ap.add_argument("--act", action="append", default=[], help="key=value activity parameter (computes "
                    "features of new windows)")
    ap.add_argument("--leak-weight", type=float, default=1.5)
    ap.add_argument("--compute-missing", action="store_true", help="run overlap separation for new pieces")
    ap.add_argument("--per-mix", action="store_true")
    ap.add_argument("--json", help="write the rows to this file")
    a = ap.parse_args()
    try:
        import torch

        torch.set_num_threads(2)
    except Exception:
        pass
    base = {k: parse_val(v) for k, v in (s.split("=", 1) for s in a.set)}
    grid = [(k, [parse_val(x) for x in v.split(",")]) for k, v in (g.split("=", 1) for g in a.grid)]
    t0 = time.time()
    act = {k: parse_val(v) for k, v in (s.split("=", 1) for s in a.act)} or None
    ctxs = [Context(m, a.preset, a.sep, a.compute_missing, act) for m in split_mixes(a.split)]
    print(f"loaded {len(ctxs)} mixes in {time.time() - t0:.1f} s", file=sys.stderr)
    print(header())
    rows = []
    combos = list(itertools.product(*[v for _, v in grid])) if grid else [()]
    for combo in combos:
        prm = dict(base)
        prm.update({k: v for (k, _), v in zip(grid, combo)})
        per = [c.score(prm) for c in ctxs]
        s = summarize(per, a.leak_weight)
        label = ",".join(f"{k}={v}" for k, v in prm.items()) or "(defaults)"
        print(fmt_row(label, s), flush=True)
        if a.per_mix:
            for r in per:
                print(fmt_row("  " + r["name"], summarize([r], a.leak_weight)))
        rows.append({"params": prm, **{k: v for k, v in s.items() if k != "agg"}})
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
