"""Compare gender classifiers on TEST speakers: speech vs non-verbal, per category and duration.

Writes eval/GENDER_MODELS.md and eval/results/gender_models.json.

Candidates (all trained only on TRAIN speakers; see eval/train_gender_head.py):
  pitch_165        median F0 > 165 Hz (no training; p = sigmoid(8*(log f0 - log 165)))
  pitch_tuned      logistic regression on log median F0 (+ voiced ratio)
  hand             head on the 22 handcrafted pitch/spectral features (no embedding)
  ecapa            head on ECAPA (TaoRuijie, VoxCeleb2) embedding only
  ecapa+hand       head on ECAPA + handcrafted (logreg / MLP)
  ecapa+hand[speech-only]   same, trained without non-verbal clips (ablation)
  wavlm+hand       head on WavLM Base+ mean hidden states + handcrafted (if extracted)
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL_DIR, RESULTS_DIR, duration_bucket  # noqa: E402
from train_gender_head import (Data, balanced_weights, fit, logits, oof_logits, platt,  # noqa: E402
                               WAVLM_LAYERS_DEFAULT)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    n = len(y)
    if n == 0:
        return {"n": 0}
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    recs = [float((pred[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    bacc = float(np.mean(recs))
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    # ECE, 10 equal-width bins on confidence of the predicted class
    conf = np.where(pred == 1, p, 1 - p)
    bins = np.linspace(0.5, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return {"n": int(n), "n_f": int(y.sum()), "n_m": int(n - y.sum()), "acc": acc, "bacc": bacc,
            "auc": auc, "ece": float(ece)}


def candidates(d: Data) -> "OrderedDict[str, dict]":
    out = OrderedDict()
    hand = d.arrays["hand"].astype(float)
    logf0 = hand[:, 0]
    voiced = hand[:, 4]
    has_pitch = voiced > 0

    # (1) pitch-only, fixed threshold
    p = sigmoid(8 * (logf0 - np.log(165.0)))
    p[~has_pitch] = 0.5
    out["pitch_165"] = {"p": p, "desc": "median F0 (pyin) > 165 Hz, no training"}

    # (1b) pitch tuned on train speakers
    def train_and_predict(features, model="logreg", C=1.0, include_nv=True, layers=WAVLM_LAYERS_DEFAULT,
                          cv_only=False, **kw):
        X, *_ = d.X(features, layers)
        m = d.train_mask(include_nv)
        w = balanced_weights(d, m)
        z_oof = oof_logits(model, X[m], d.y[m], w[m], d.speaker[m], C, **kw)
        a, b = platt(z_oof, d.y[m], w[m])
        p_oof = sigmoid(a * z_oof + b)
        cv = metrics(p_oof, d.y[m])
        # speaker-grouped CV on the TRAIN non-verbal clips (also when the head was trained without them)
        nvm = d.train_mask(True) & (d.group == "nonverbal")
        if include_nv:
            idx = np.flatnonzero(m)
            sel = d.group[idx] == "nonverbal"
            cv["nonverbal"] = metrics(p_oof[sel], d.y[idx][sel])
        if cv_only:
            return None, cv
        mu, sd, params = fit(model, X[m], d.y[m], w[m], C, **kw)
        z = logits(model, params, (X - mu) / sd)
        if not include_nv:
            cv["nonverbal"] = metrics(sigmoid(a * z[nvm] + b), d.y[nvm])
        return sigmoid(a * z + b), cv

    feats_avail = set(d.arrays)
    plan = [("pitch_tuned", "pitch", "logreg", 1.0, True, "logreg on 6 pitch stats"),
            ("hand", "hand", "logreg", 1.0, True, "logreg on 22 pitch/spectral/MFCC features"),
            ("hand_mlp", "hand", "mlp", 1.0, True, "MLP(64) on 22 handcrafted features [shipped as gender_head_nofeat.npz]")]
    if "ecapa" in feats_avail:
        plan += [("ecapa", "ecapa", "logreg", 0.3, True, "logreg on ECAPA-TDNN (Tao, Vox2) 192-d"),
                 ("ecapa+hand", "ecapa+hand", "logreg", 0.3, True, "logreg on ECAPA + handcrafted"),
                 ("ecapa+hand_mlp", "ecapa+hand", "mlp", 1.0, True, "MLP(64) on ECAPA + handcrafted [shipped as gender_head_ecapa.npz]"),
                 ("ecapa+hand[speech-only]", "ecapa+hand", "logreg", 0.3, False,
                  "ablation: logreg on ECAPA + handcrafted, trained WITHOUT non-verbal clips")]
    layers = {}
    if "wavlm" in feats_avail:
        # choose the WavLM layer set by speaker-grouped CV on TRAIN speakers (non-verbal bacc)
        scan = {}
        for ls in [(2,), (4,), (6,), (8,), (10,), (12,), (3, 4, 5), (4, 5, 6), (6, 7, 8), (8, 9, 10),
                   tuple(range(1, 13))]:
            _, cv = train_and_predict("wavlm+hand", "logreg", 0.3, True, layers=ls, cv_only=True)
            scan["-".join(map(str, ls))] = {"cv_bacc": cv["bacc"], "cv_nv_bacc": cv["nonverbal"]["bacc"]}
            print("wavlm layers", ls, scan["-".join(map(str, ls))], flush=True)
        best = max(scan, key=lambda k: (scan[k]["cv_nv_bacc"], scan[k]["cv_bacc"]))
        LAYER_SCAN.update(scan)
        layers = {"layers": tuple(int(x) for x in best.split("-"))}
        plan += [("wavlm", "wavlm", "logreg", 0.3, True, f"logreg on WavLM Base+ layers {best} mean only"),
                 ("wavlm+hand", "wavlm+hand", "logreg", 0.3, True,
                  f"logreg on WavLM Base+ layers {best} mean + handcrafted"),
                 ("wavlm+hand_mlp", "wavlm+hand", "mlp", 1.0, True, f"MLP(64) on WavLM {best} + handcrafted [shipped as gender_head.npz]"),
                 ("wavlm+hand[speech-only]", "wavlm+hand", "logreg", 0.3, False,
                  "ablation: WavLM + handcrafted trained WITHOUT non-verbal clips"),
                 ("ecapa+wavlm+hand", "ecapa+wavlm+hand", "logreg", 0.3, True,
                  "logreg on ECAPA + WavLM + handcrafted"),
                 ("ecapa+wavlm+hand_mlp", "ecapa+wavlm+hand", "mlp", 1.0, True,
                  "MLP(64) on ECAPA + WavLM + handcrafted (needs two embeddings; not shipped)")]
    for name, feats, model, C, inv, desc in plan:
        kw = layers if "wavlm" in feats else {}
        p, cv = train_and_predict(feats, model, C, inv, **kw)
        out[name] = {"p": p, "desc": desc, "train_cv": cv}
        print(f"trained {name}: train-CV bacc {cv['bacc']:.3f} NV {cv.get('nonverbal', {}).get('bacc')}", flush=True)
    return out


LAYER_SCAN: dict = {}


def evaluate(d: Data, cands) -> dict:
    test = d.split == "test"
    slices = OrderedDict()
    slices["speech (all)"] = test & (d.group == "speech")
    slices["non-verbal phonated (all)"] = test & (d.group == "nonverbal")
    slices["non-verbal non-phonated (clicks etc.)"] = test & (d.group == "nonverbal_nonphonated")
    for v in ("full", "crop"):
        slices[f"speech {v}"] = test & (d.group == "speech") & (d.variant == v)
        slices[f"non-verbal {v}"] = test & (d.group == "nonverbal") & (d.variant == v)
    for src in sorted(set(d.source[test & (d.group == "speech")])):
        slices[f"speech src={src}"] = test & (d.group == "speech") & (d.source == src)
    slices["speech whispered (EARS)"] = test & (d.category == "ears_whisper")
    for src in sorted(set(d.source[test & (d.group == "nonverbal")])):
        slices[f"non-verbal src={src}"] = test & (d.group == "nonverbal") & (d.source == src)
        for g, gl in ((1, "F"), (0, "M")):
            slices[f"non-verbal src={src} gender={gl}"] = test & (d.group == "nonverbal") & (d.source == src) & (d.y == g)
    for cat in sorted(set(d.category[test & np.isin(d.group, ["nonverbal", "nonverbal_nonphonated"])])):
        slices[f"nv cat={cat}"] = test & (d.category == cat)
    buckets = np.array([duration_bucket(x) for x in d.dur])
    for grp in ("speech", "nonverbal"):
        for b in ("<0.5s", "0.5-1.5s", ">1.5s"):
            slices[f"{grp} dur {b}"] = test & (d.group == grp) & (buckets == b)
    res = OrderedDict()
    for name, c in cands.items():
        res[name] = {"desc": c["desc"], "train_cv": c.get("train_cv"),
                     "slices": OrderedDict((s, metrics(c["p"][m], d.y[m])) for s, m in slices.items())}
    return res


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def write_report(res: dict, d: Data) -> None:
    names = list(res)
    L = ["# Gender classifier comparison (speech vs non-verbal)", "",
         "Generated by `eval/compare_gender_models.py` from features extracted by `eval/extract_features.py`.",
         "All numbers are on **test speakers only** (speaker-disjoint from every trained head, see "
         "`eval/splits.json`). Items: each test clip appears twice, as the (≤3 s) full clip and as one random "
         "0.25-1.5 s crop. Threshold 0.5 on calibrated p(female). bacc = balanced accuracy, ECE = expected "
         "calibration error (10 bins, %). See `eval/DATA.md` for the data and its gaps.", "",
         "## Models that could not be evaluated here", "",
         "Hugging Face file downloads are blocked on this build machine (the CDN host `us.aws.cdn.hf.co` is "
         "refused by the network proxy; adding it was declined), so these requested candidates were **not** "
         "evaluated: `JaesungHuh/voice-gender-classifier` (weights only on HF / Google Drive), "
         "`audeering/wav2vec2-large-robust-24-ft-age-gender` (+6-layer), "
         "`alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech`, and "
         "`speechbrain/spkrec-ecapa-voxceleb` embeddings. Substitutes: the ECAPA-TDNN of "
         "TaoRuijie/ECAPA-TDNN (GitHub, MIT; VoxCeleb2-trained, 192-d, the same backbone JaesungHuh fine-tuned) "
         "and WavLM Base+ from torchaudio (download.pytorch.org).", ""]
    main_slices = ["speech (all)", "non-verbal phonated (all)", "non-verbal non-phonated (clicks etc.)"]
    L += ["## Headline: speech vs non-verbal", "",
          "| model | " + " | ".join(f"{s} acc / bacc / AUC / ECE" for s in main_slices)
          + " | train-CV bacc (all / non-verbal) |",
          "|---|" + "---|" * (len(main_slices) + 1)]
    for n in names:
        cells = []
        for s in main_slices:
            m = res[n]["slices"][s]
            cells.append(f"{fmt(m.get('acc'))} / {fmt(m.get('bacc'))} / {fmt(m.get('auc'), False)} / "
                         f"{fmt(m.get('ece'))}")
        cv = res[n].get("train_cv")
        cvn = (cv or {}).get("nonverbal", {}).get("bacc")
        L.append(f"| {n} | " + " | ".join(cells) + f" | {fmt(cv['bacc']) if cv else '-'} / {fmt(cvn)} |")
    L += ["", "Model descriptions:", ""] + [f"- **{n}**: {res[n]['desc']}" for n in names] + [""]
    if LAYER_SCAN:
        L += ["## WavLM layer selection (speaker-grouped 5-fold CV on TRAIN speakers, WavLM + handcrafted, logreg)",
              "", "| layers (1-based, time-mean, averaged) | CV bacc all | CV bacc non-verbal |", "|---|---|---|"]
        L += [f"| {k} | {fmt(v['cv_bacc'])} | {fmt(v['cv_nv_bacc'])} |" for k, v in LAYER_SCAN.items()] + [""]

    def table(title, keys):
        T = [f"## {title}", "", "| slice | n (F/M) | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names)]
        for s in keys:
            m0 = res[names[0]]["slices"][s]
            if not m0.get("n"):
                continue
            row = [f"{fmt(res[n]['slices'][s].get('bacc'))}" for n in names]
            T.append(f"| {s} | {m0['n']} ({m0['n_f']}/{m0['n_m']}) | " + " | ".join(row) + " |")
        return T + [""]

    allk = list(res[names[0]]["slices"])
    L += table("Balanced accuracy (%) by slice", allk)
    L += ["## AUC by slice", "", "| slice | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in allk:
        if res[names[0]]["slices"][s].get("n"):
            L.append(f"| {s} | " + " | ".join(fmt(res[n]['slices'][s].get('auc'), False) for n in names) + " |")
    L.append("")
    (EVAL_DIR / "GENDER_MODELS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    d = Data()
    cands = candidates(d)
    res = evaluate(d, cands)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gender_models.json").write_text(json.dumps({"models": res, "wavlm_layer_scan": LAYER_SCAN},
                                                               indent=1))
    write_report(res, d)
    for n, r in res.items():
        s, nv = r["slices"]["speech (all)"], r["slices"]["non-verbal phonated (all)"]
        print(f"{n:28s} speech bacc {fmt(s.get('bacc'))} auc {fmt(s.get('auc'), False)} | "
              f"NV bacc {fmt(nv.get('bacc'))} auc {fmt(nv.get('auc'), False)} ece {fmt(nv.get('ece'))}")


if __name__ == "__main__":
    main()
