"""Speaker clustering, gender fusion and speaker-level consistency.

Fusion (per analysis window)
----------------------------
Every available model m gives p_m(female) and a reliability r_m in [0, 1] (e.g. the pitch
prior is unreliable for unvoiced sounds).  With a configured weight w_m per model and
event kind (speech / nonverbal), the fused logit is the reliability-weighted *average* of
the model logits::

    l = sum_m w_m r_m logit(p_m) / sum_m w_m r_m

An average (not a sum) keeps the calibration of the individual (Platt-calibrated) models:
the models are strongly correlated, so adding their logits would be overconfident.
Default weights: trained gender head 1.0 (primary signal, the only one trained on
non-verbal sounds), HF speech gender model 0.6 for speech / 0.2 for non-verbal, AudioSet
tagger gender classes 0.3 for speech, F0 prior 0.15 (1.0 if no head is available).

Speaker consistency
-------------------
Windows are clustered by speaker embedding (agglomerative, cosine, average linkage,
distance threshold; tiny clusters are merged into their nearest neighbour; a cluster
with confident evidence for both genders is split in two).  The speaker logit is the
confidence- and duration-weighted mean of its window logits::

    L_s = sum_i d_i (eps + c_i) l_i / sum_i d_i (eps + c_i),     c_i = |2 p_i - 1|

and its *purity* is the weighted share of evidence agreeing with sign(L_s).  The final
segment logit mixes its own evidence with its speaker's::

    l_final = (a l_seg + b L_s) / (a + b)
    a = min(1, dur / 2 s) (0.25 + c_seg)
    b = k purity min(1, total_dur_s / 5 s) (0.25 + c_s)

so short or uncertain segments lean on the speaker, while long confident segments can
disagree with it (e.g. when clustering merged a man and a woman).  Confidence of the
final posterior is |2p - 1|; below ``low_conf_threshold`` the segment is flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

EPS = 1e-4

DEFAULT_WEIGHTS = {
    "speech": {"head": 1.0, "hf": 0.6, "tagger": 0.3, "pitch": 0.15},
    "nonverbal": {"head": 1.0, "hf": 0.2, "tagger": 0.15, "pitch": 0.15},
    "mixed": {"head": 1.0, "hf": 0.4, "tagger": 0.2, "pitch": 0.15},
}


def logit(p: np.ndarray | float, clip: float = 1e-3) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), clip, 1 - clip)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, np.float64), -40, 40)))


def fuse(evidence: dict[str, tuple[float, float]], kind: str,
         weights: Optional[dict] = None, head_available: bool = True) -> tuple[float, float]:
    """evidence: {model: (p_female, reliability)} -> (fused logit, total weight)."""
    w = dict((weights or DEFAULT_WEIGHTS).get(kind, DEFAULT_WEIGHTS["mixed"]))
    if not head_available or "head" not in evidence:
        w["pitch"] = 1.0
    num = den = 0.0
    for m, (p, r) in evidence.items():
        wm = w.get(m, 0.0) * float(r)
        if wm <= 0:
            continue
        num += wm * float(logit(p))
        den += wm
    if den <= 0:
        return 0.0, 0.0
    return num / den, den


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def _norm(E: np.ndarray) -> np.ndarray:
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def cluster_embeddings(E: np.ndarray, durations: np.ndarray, threshold: float = 0.65,
                       max_points: int = 2500, min_cluster_s: float = 2.0,
                       window_logits: Optional[np.ndarray] = None,
                       window_conf: Optional[np.ndarray] = None,
                       seed: int = 0) -> np.ndarray:
    """Cluster window embeddings; returns labels 0..K-1 (ordered by first appearance)."""
    n = len(E)
    if n == 0:
        return np.zeros(0, int)
    if n == 1:
        return np.zeros(1, int)
    X = _norm(np.asarray(E, np.float64))
    rng = np.random.default_rng(seed)
    if n > max_points:
        sub = np.sort(rng.choice(n, max_points, replace=False))
    else:
        sub = np.arange(n)
    from sklearn.cluster import AgglomerativeClustering

    ac = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                 distance_threshold=threshold)
    lab_sub = ac.fit_predict(X[sub])
    K = lab_sub.max() + 1
    cent = np.stack([_norm(X[sub][lab_sub == k].mean(0, keepdims=True))[0] for k in range(K)])
    labels = np.argmax(X @ cent.T, axis=1)
    labels = _merge_small(X, labels, durations, min_cluster_s)
    if window_logits is not None and window_conf is not None:
        labels = _split_mixed_gender(X, labels, durations, window_logits, window_conf)
    # relabel by first appearance
    order: dict[int, int] = {}
    for l in labels:
        if l not in order:
            order[l] = len(order)
    return np.array([order[l] for l in labels], int)


def _centroids(X, labels):
    ks = sorted(set(labels.tolist()))
    return ks, np.stack([_norm(X[labels == k].mean(0, keepdims=True))[0] for k in ks])


def _merge_small(X, labels, durations, min_s):
    labels = labels.copy()
    for _ in range(50):
        ks, C = _centroids(X, labels)
        if len(ks) <= 1:
            break
        tot = {k: float(durations[labels == k].sum()) for k in ks}
        small = [k for k in ks if tot[k] < min_s]
        if not small:
            break
        k = min(small, key=lambda q: tot[q])
        i = ks.index(k)
        sims = C @ C[i]
        sims[i] = -9
        j = int(np.argmax(sims))
        labels[labels == k] = ks[j]
    return labels


def _split_mixed_gender(X, labels, durations, l, c, min_share=0.25, conf=0.5):
    """Split clusters holding confident evidence for both genders (merged man + woman)."""
    labels = labels.copy()
    nxt = labels.max() + 1
    for k in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == k)
        if len(idx) < 4:
            continue
        w = durations[idx] * (c[idx] > conf)
        tot = durations[idx].sum()
        f_share = w[l[idx] > 0].sum() / max(tot, 1e-9)
        m_share = w[l[idx] < 0].sum() / max(tot, 1e-9)
        if f_share >= min_share and m_share >= min_share:
            # 2-means in [embedding | scaled gender logit] space, seeded by gender sign
            Z = np.concatenate([X[idx], 0.15 * np.clip(l[idx], -4, 4)[:, None]], axis=1)
            a = (l[idx] > 0).astype(int)
            for _ in range(10):
                if a.min() == a.max():
                    break
                c0, c1 = Z[a == 0].mean(0), Z[a == 1].mean(0)
                na = (np.linalg.norm(Z - c1, axis=1) < np.linalg.norm(Z - c0, axis=1)).astype(int)
                if (na == a).all():
                    break
                a = na
            if 0 < a.sum() < len(a):
                labels[idx[a == 1]] = nxt
                nxt += 1
    return labels


# ---------------------------------------------------------------------------
# speaker consistency
# ---------------------------------------------------------------------------
@dataclass
class SpeakerStats:
    id: str
    logit: float
    p_female: float
    purity: float
    total_dur: float
    n_segments: int = 0
    conf: float = 0.0
    extra: dict = field(default_factory=dict)


def speaker_posteriors(labels: np.ndarray, durations: np.ndarray, l: np.ndarray,
                       c: np.ndarray, prefix: str = "S") -> dict[int, SpeakerStats]:
    out: dict[int, SpeakerStats] = {}
    for k in sorted(set(labels.tolist())):
        m = labels == k
        w = durations[m] * (0.05 + c[m])
        L = float((w * l[m]).sum() / max(w.sum(), 1e-9))
        sign = 1.0 if L >= 0 else -1.0
        agree = float((w * (np.sign(l[m]) == sign)).sum() / max(w.sum(), 1e-9))
        p = float(sigmoid(L))
        out[k] = SpeakerStats(id=f"{prefix}{k + 1}", logit=L, p_female=p, purity=agree,
                              total_dur=float(durations[m].sum()), conf=abs(2 * p - 1))
    return out


def combine_with_speaker(l_seg: float, c_seg: float, dur: float, spk: Optional[SpeakerStats],
                         k: float = 1.0) -> float:
    if spk is None:
        return l_seg
    a = min(1.0, dur / 2.0) * (0.25 + c_seg)
    b = k * spk.purity * min(1.0, spk.total_dur / 5.0) * (0.25 + spk.conf)
    if a + b <= 0:
        return l_seg
    return (a * l_seg + b * spk.logit) / (a + b)


def smooth_labels(labels: np.ndarray, k: int = 3) -> np.ndarray:
    """Majority filter of window speaker labels (odd window k)."""
    if len(labels) < k:
        return labels
    out = labels.copy()
    h = k // 2
    for i in range(len(labels)):
        seg = labels[max(0, i - h):i + h + 1]
        vals, cnt = np.unique(seg, return_counts=True)
        if cnt.max() > len(seg) // 2:
            out[i] = vals[np.argmax(cnt)]
    return out
