"""Train the gender head (voicefilter/assets/gender_head*.npz) on frozen features.

Inputs: eval/cache/features/all.npz + eval/cache/items.csv (eval/extract_features.py).
Training data: TRAIN speakers only (eval/splits.json), speech + phonated non-verbal clips,
full clips and short crops. Sample weights balance F/M within every (source, kind) cell
and give non-verbal clips the same total weight as speech, so the head cannot use
"which corpus / which recording channel" as a proxy for gender.

Calibration: Platt scaling fitted on speaker-grouped out-of-fold logits of the train set.
Seeds are fixed; the run is deterministic on a given machine.

Usage:
  python eval/train_gender_head.py                    # trains + saves the default heads
  python eval/train_gender_head.py --features ecapa+hand --model logreg --out PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE_DIR  # noqa: E402

from voicefilter.assets.gender_head import HANDCRAFTED_NAMES, GenderHead, save_head  # noqa: E402

FEAT = CACHE_DIR / "features" / "all.npz"
ITEMS = CACHE_DIR / "items.csv"
ASSETS = Path(__file__).resolve().parents[1] / "voicefilter" / "assets"
SEED = 1234
WAVLM_LAYERS_DEFAULT = (3, 4, 5)   # 1-based transformer layers averaged (picked by train CV)
MLP_EPOCHS = 40                    # picked by train CV (less over-confident than 200)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, feat_path: Path = FEAT, items_path: Path = ITEMS):
        z = np.load(feat_path, allow_pickle=False)
        self.arrays = {k: z[k] for k in z.files}
        meta = {r["item_id"]: r for r in csv.DictReader(open(items_path, encoding="utf-8"))}
        ids = [str(s) for s in self.arrays["item_id"]]
        self.meta = [meta[i] for i in ids]
        self.y = np.array([1 if m["gender"] == "F" else 0 for m in self.meta])
        self.split = np.array([m["split"] for m in self.meta])
        self.group = np.array([m["group"] for m in self.meta])
        self.source = np.array([m["source"] for m in self.meta])
        self.speaker = np.array([m["speaker"] for m in self.meta])
        self.category = np.array([m["category"] for m in self.meta])
        self.variant = np.array([m["variant"] for m in self.meta])
        self.dur = self.arrays["dur"].astype(float)

    def X(self, features: str, wavlm_layers=WAVLM_LAYERS_DEFAULT) -> tuple[np.ndarray, list[str], str, int]:
        """Feature matrix, names, embedding kind and embedding dim for a feature-set name."""
        parts = features.split("+")
        cols, names, kind, edim = [], [], "none", 0
        for p in parts:
            if p == "ecapa":
                e = self.arrays["ecapa"].astype(np.float64)
                e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
                cols.append(e)
                names += [f"ecapa_{i}" for i in range(e.shape[1])]
                kind, edim = "ecapa_tao_vox2_192", e.shape[1]
            elif p == "wavlm":
                w = self.arrays["wavlm"].astype(np.float64)[:, [l - 1 for l in wavlm_layers]].mean(1)
                w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)
                cols.append(w)
                names += [f"wavlm_{i}" for i in range(w.shape[1])]
                kind = "wavlm_base_plus_L" + "-".join(map(str, wavlm_layers)) + "_mean"
                edim = w.shape[1]
            elif p == "hand":
                cols.append(self.arrays["hand"].astype(np.float64))
                names += list(HANDCRAFTED_NAMES)
            elif p == "pitch":
                h = self.arrays["hand"].astype(np.float64)
                cols.append(h[:, :6])
                names += list(HANDCRAFTED_NAMES[:6])
            else:
                raise ValueError(p)
        return np.concatenate(cols, axis=1), names, kind, edim

    def train_mask(self, include_nonverbal: bool = True, variants=("full", "crop")) -> np.ndarray:
        m = (self.split == "train") & np.isin(self.variant, variants)
        allowed = ["speech"] + (["nonverbal"] if include_nonverbal else [])
        return m & np.isin(self.group, allowed)


def balanced_weights(d: Data, mask: np.ndarray, nonverbal_share: float = 0.5) -> np.ndarray:
    """F/M balanced inside each (source, group) cell; NV total = nonverbal_share of all weight."""
    w = np.zeros(len(d.y))
    idx = np.flatnonzero(mask)
    cells = defaultdict(list)
    for i in idx:
        cells[(d.source[i], d.group[i], d.y[i])].append(i)
    for (src, grp, g), ii in cells.items():
        w[ii] = 1.0 / len(ii)
    for grp, share in (("speech", 1 - nonverbal_share), ("nonverbal", nonverbal_share)):
        gi = idx[d.group[idx] == grp]
        if len(gi):
            w[gi] *= share / w[gi].sum()
    w[idx] *= len(idx) / w[idx].sum()
    return w


# ----------------------------------------------------------------------------- models
def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def fit_logreg(X, y, w, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
    clf.fit(X, y, sample_weight=w)
    return {"w": clf.coef_[0].astype(np.float64), "b": float(clf.intercept_[0])}


def fit_mlp(X, y, w, hidden=64, weight_decay=1e-3, epochs=MLP_EPOCHS, lr=3e-3):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    wt = torch.tensor(w / w.mean(), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(SEED)
    n = len(yt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, 256):
            b = perm[s:s + 256]
            # light input dropout as regulariser
            xb = Xt[b] * (torch.rand(Xt[b].shape, generator=g) > 0.1).float() / 0.9
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(xb)[:, 0], yt[b], reduction="none")
                    * wt[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    W1 = net[0].weight.detach().numpy().T.astype(np.float64)
    return {"W1": W1, "b1": net[0].bias.detach().numpy().astype(np.float64),
            "w2": net[2].weight.detach().numpy()[0].astype(np.float64), "b2": float(net[2].bias.item())}


def logits(model_type, params, Xs):
    if model_type == "logreg":
        return Xs @ params["w"] + params["b"]
    h = np.maximum(0, Xs @ params["W1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def fit(model_type, X, y, w, C=1.0, **kw):
    mu, sd = standardize_fit(X)
    Xs = (X - mu) / sd
    params = fit_logreg(Xs, y, w, C) if model_type == "logreg" else fit_mlp(Xs, y, w, **kw)
    return mu, sd, params


def group_folds(groups: np.ndarray, k: int = 5) -> np.ndarray:
    from common import stable_unit
    uniq = sorted(set(groups))
    f = {g: int(stable_unit(g, "fold") * k) for g in uniq}
    return np.array([f[g] for g in groups])


def oof_logits(model_type, X, y, w, groups, C=1.0, k=5, **kw):
    folds = group_folds(groups, k)
    out = np.zeros(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        mu, sd, p = fit(model_type, X[tr], y[tr], w[tr], C, **kw)
        out[te] = logits(model_type, p, (X[te] - mu) / sd)
    return out


def platt(z, y, w):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1e4, max_iter=1000)
    clf.fit(z.reshape(-1, 1), y, sample_weight=w)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def train_head(d: Data, features: str, model_type: str = "logreg", C: float = 1.0,
               include_nonverbal: bool = True, variants=("full", "crop"), wavlm_layers=WAVLM_LAYERS_DEFAULT,
               **kw) -> GenderHead:
    X, names, kind, edim = d.X(features, wavlm_layers)
    m = d.train_mask(include_nonverbal, variants)
    w = balanced_weights(d, m)
    Xtr, ytr, wtr, gtr = X[m], d.y[m], w[m], d.speaker[m]
    z = oof_logits(model_type, Xtr, ytr, wtr, gtr, C, **kw)
    a, b = platt(z, ytr, wtr)
    mu, sd, params = fit(model_type, Xtr, ytr, wtr, C, **kw)
    if "hand" not in features.split("+") and "pitch" not in features.split("+"):
        raise ValueError("deployable heads always include the handcrafted features")
    info = {"features": features, "model_type": model_type, "C": C, "include_nonverbal": include_nonverbal,
            "variants": list(variants), "n_train": int(m.sum()), "n_train_speakers": int(len(set(gtr))),
            "seed": SEED, "trainer": "eval/train_gender_head.py", **{k: v for k, v in kw.items()}}
    if model_type == "mlp":
        info.setdefault("epochs", MLP_EPOCHS)
    if "wavlm" in features:
        info["wavlm_layers"] = list(wavlm_layers)
    if features.endswith("pitch"):
        raise ValueError("use 'hand' (all 22 handcrafted features) for deployable heads")
    return GenderHead(feature_names=names, embedding_kind=kind, embedding_dim=edim, model_type=model_type,
                      mean=mu, std=sd, params=params, calib_a=a, calib_b=b, info=info)


# Chosen with eval/compare_gender_models.py (see eval/GENDER_MODELS.md): the default head is the
# candidate with the best non-verbal balanced accuracy; ECAPA and no-embedding heads are fallbacks.
DEFAULT_HEADS = {
    # file name                 features             model     C    WavLM layers
    "gender_head.npz": ("wavlm+hand", "mlp", 1.0, WAVLM_LAYERS_DEFAULT),
    "gender_head_ecapa.npz": ("ecapa+hand", "mlp", 1.0, None),
    "gender_head_nofeat.npz": ("hand", "mlp", 1.0, None),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--model", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Data()
    jobs = {a.out or "custom.npz": (a.features, a.model, a.C, None)} if a.features else DEFAULT_HEADS
    for fname, (features, model, C, layers) in jobs.items():
        kw = {"wavlm_layers": layers} if layers else {}
        head = train_head(d, features, model, C, **kw)
        out = Path(fname) if a.out else ASSETS / fname
        save_head(out, head)
        print("saved", out, head.embedding_kind, head.model_type, json.dumps(head.info),
              f"platt a={head.calib_a:.3f} b={head.calib_b:.3f}")


if __name__ == "__main__":
    main()
