"""Extract per-item features for the gender-classifier study (CPU, resumable, sharded).

Items = labelled clips from eval/data/clips.csv, capped per speaker, each in two variants:
  full : the clip (centre-cropped to at most MAX_FULL seconds)
  crop : one random 0.25-1.5 s excerpt centred on an active frame (short-event robustness)
For every item we store:
  hand   : 22 handcrafted pitch/spectral features (voicefilter.assets.gender_head)
  ecapa  : 192-d TaoRuijie ECAPA-TDNN embedding (raw, un-normalised)
  wavlm  : WavLM Base+ hidden states, time-mean per layer (12 x 768, float16) [--wavlm]

Usage:
  python eval/extract_features.py --shard 0 --nshards 2 [--wavlm]
  python eval/extract_features.py --merge
Outputs eval/cache/features/shard_<k>of<n>.npz and eval/cache/features/all.npz
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE_DIR, DATA_DIR, load_audio, load_splits, read_manifest, setup_env  # noqa: E402

FEAT_DIR = CACHE_DIR / "features"
ITEMS_CSV = CACHE_DIR / "items.csv"
MAX_FULL = 3.0
CROP_MIN, CROP_MAX = 0.25, 1.5
PER_SPEAKER = {  # (train, test) clips per speaker, by source
    "librispeech": (3, 3), "crowd": (2, 3), "crema": (3, 3), "audiomnist": (2, 3), "deeply": (99, 99), "ears": (99, 99),
}
NONPHONATED = {"teeth-chattering", "teeth-grinding", "tongue-clicking", "lip-popping", "lip-smacking",
               "nose-blowing"}


def select_items() -> list[dict]:
    rows = read_manifest()
    splits = load_splits()["speakers"]
    by_spk = defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    items = []
    for spk in sorted(by_spk):
        rs = sorted(by_spk[spk], key=lambda r: r["clip_id"])
        split = splits.get(spk, "unassigned")
        k_train, k_test = PER_SPEAKER[rs[0]["source"]]
        k = k_test if split == "test" else k_train
        rng = random.Random(f"items-{spk}")
        if rs[0]["source"] == "crema":  # spread over emotions
            rng.shuffle(rs)
            rs = sorted(rs, key=lambda r: ("neu" in r["category"], r["category"]))
            seen, pick = set(), []
            for r in rs:
                if r["category"] not in seen:
                    pick.append(r)
                    seen.add(r["category"])
            pick += [r for r in rs if r not in pick]
            rs = pick
        else:
            rng.shuffle(rs)
        for r in rs[:k]:
            group = "nonverbal_nonphonated" if r["category"] in NONPHONATED else r["kind"]
            for variant in ("full", "crop"):
                items.append(dict(r, item_id=f"{r['clip_id']}#{variant}", variant=variant, split=split,
                                  group=group))
    return items


def write_items(items: list[dict]) -> None:
    ITEMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "clip_id", "variant", "split", "group", "source", "speaker", "gender", "kind",
              "category", "lang", "duration", "path"]
    with open(ITEMS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)


def cut(y: np.ndarray, item: dict, sr: int = 16000) -> tuple[np.ndarray, float]:
    n = len(y)
    if item["variant"] == "full":
        m = int(MAX_FULL * sr)
        if n <= m:
            return y, 0.0
        s = (n - m) // 2
        return y[s:s + m], s / sr
    rng = random.Random(f"crop-{item['clip_id']}")
    dur = rng.uniform(CROP_MIN, CROP_MAX)
    m = min(n, int(dur * sr))
    hop = 160
    env = np.array([np.sqrt(np.mean(y[i:i + 400] ** 2) + 1e-12) for i in range(0, max(1, n - 400), hop)])
    db = 20 * np.log10(env + 1e-12)
    act = np.flatnonzero(db > db.max() - 25)
    c = int(act[rng.randrange(len(act))]) * hop + 200 if len(act) else n // 2
    s = int(np.clip(c - m // 2, 0, n - m))
    return y[s:s + m], s / sr


def _load_cache() -> dict:
    """Previously merged features (all.npz), used so a later pass only computes what is missing."""
    p = FEAT_DIR / "all.npz"
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    ids = [str(i) for i in z["item_id"]]
    arrays = {k: z[k] for k in z.files}
    return {i: {k: arrays[k][j] for k in arrays if k != "item_id"} for j, i in enumerate(ids)}


def run_shard(shard: int, nshards: int, use_wavlm: bool, use_ecapa: bool = True, prefix: str = "shard") -> None:
    import torch
    setup_env()
    torch.set_num_threads(2)
    from voicefilter.assets.gender_head import handcrafted_features
    items = select_items()
    if shard == 0:
        write_items(items)
    mine = items[shard::nshards]
    out = FEAT_DIR / f"{prefix}_{shard}of{nshards}.npz"
    cache = _load_cache() if prefix != "shard" else {}
    done = {}
    if out.exists():
        z = np.load(out, allow_pickle=False)
        done = {k: i for i, k in enumerate(z["item_id"])}
        prev = {k: z[k] for k in z.files}
    ecapa = None
    if use_ecapa:
        from voicefilter.assets.ecapa_tao import embed, load_ecapa
        ecapa = load_ecapa()
    wavlm = None
    if use_wavlm:
        import torchaudio
        wavlm = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval()
    res = defaultdict(list)
    produce = {"item_id", "start", "dur", "hand"} | ({"ecapa"} if use_ecapa else set()) | (
        {"wavlm"} if use_wavlm else set())
    if done and not produce <= set(prev):
        done = {}
    t0 = time.time()
    audio_cache: dict[str, np.ndarray] = {}
    for i, it in enumerate(mine):
        if it["item_id"] in done:
            j = done[it["item_id"]]
            for k in produce:
                res[k].append(prev[k][j])
            continue
        p = it["path"]
        if p not in audio_cache:
            audio_cache.clear()
            audio_cache[p] = load_audio(DATA_DIR / p)
        y, start = cut(audio_cache[p], it)
        c = cache.get(it["item_id"], {})
        res["item_id"].append(it["item_id"])
        res["start"].append(start)
        res["dur"].append(len(y) / 16000)
        res["hand"].append(c["hand"] if "hand" in c else handcrafted_features(y, 16000))
        if ecapa is not None:
            res["ecapa"].append(c["ecapa"] if "ecapa" in c else embed(ecapa, y))
        if wavlm is not None and "wavlm" in c:
            res["wavlm"].append(c["wavlm"])
        elif wavlm is not None:
            yy = y if len(y) >= 8000 else np.tile(y, int(np.ceil(8000 / max(1, len(y)))))[:8000]
            with torch.inference_mode():
                hs, _ = wavlm.extract_features(torch.from_numpy(yy)[None])
            res["wavlm"].append(np.stack([h[0].mean(0).numpy() for h in hs]).astype(np.float16))
        if (i + 1) % 50 == 0:
            print(f"shard {shard}: {i + 1}/{len(mine)} items, {time.time() - t0:.0f}s", flush=True)
            _save(out, res)
    _save(out, res)
    print(f"shard {shard}: done {len(mine)} items in {time.time() - t0:.0f}s")


def _save(out: Path, res: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in res.items()}
    n = {len(v) for v in arrays.values()}
    assert len(n) == 1, {k: len(v) for k, v in arrays.items()}
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(str(tmp), **arrays)
    tmp.replace(out)


def merge() -> Path:
    parts = sorted(FEAT_DIR.glob("v2_*of*.npz")) or sorted(FEAT_DIR.glob("shard_*of*.npz"))
    keys = None
    acc = defaultdict(list)
    for p in parts:
        z = np.load(p, allow_pickle=False)
        ks = set(z.files)
        keys = ks if keys is None else keys & ks
        for k in z.files:
            acc[k].append(z[k])
    arrays = {k: np.concatenate(acc[k]) for k in keys}
    _, first = np.unique(arrays["item_id"], return_index=True)   # drop duplicates across shards
    first = np.sort(first)
    arrays = {k: v[first] for k, v in arrays.items()}
    out = FEAT_DIR / "all.npz"
    np.savez(str(out), **arrays)
    print("merged", {k: v.shape for k, v in arrays.items()}, "->", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--wavlm", action="store_true")
    ap.add_argument("--no-ecapa", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pass2", action="store_true",
                    help="second pass: reuse eval/cache/features/all.npz, compute only missing features")
    ap.add_argument("--list", action="store_true", help="only write items.csv and print counts")
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    if a.list:
        items = select_items()
        write_items(items)
        from collections import Counter
        c = Counter((i["split"], i["group"], i["gender"], i["variant"]) for i in items)
        for k in sorted(c):
            print(k, c[k])
        print(len(items))
        return
    run_shard(a.shard, a.nshards, a.wavlm, not a.no_ecapa, prefix="v2" if a.pass2 else "shard")


if __name__ == "__main__":
    main()
