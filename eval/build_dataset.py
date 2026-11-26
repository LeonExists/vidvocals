"""Build synthetic evaluation scenes ("mixes") with ground-truth vocal timelines.

Only TEST speakers (eval/splits.json) are used, so no trained head has seen these voices.

Each mix (44.1 kHz stereo, 60-120 s by default) is a small scene with 2-4 consistent
speakers (sometimes only one gender): alternating speech turns and non-verbal bursts with
realistic gaps, ~10-20 % of vocal time with male+female overlapping, music / environmental
noise / synthetic noise beds at -5..+15 dB vocal-to-background ratio, level variation and
light reverb on some mixes. Hard cases: very short events (<0.4 s), the lowest-pitched
female and the highest-pitched male test speakers, breathy categories (sigh/pant/yawn),
and one background with sung vocals.

A scene "speaker" pairs one test speech speaker with a pool of non-verbal clips from
test speakers of the same gender in the Deeply corpus (no corpus has both speech and
non-verbal sounds of the same person); `source_speaker` in truth.json keeps the corpus id.

Per mix directory (eval/data/mixes/<name>/):
  mix.wav, vocals_female.wav, vocals_male.wav, background.wav   float32 WAV, mix = sum of stems
  truth.json  {duration, sr, events:[{start,end,speaker,gender,kind,category,overlap,active,...}], meta}
  mix.mp4     ffmpeg testsrc2 video + mix audio (h264 + aac)

Usage:
  python eval/build_dataset.py                  # 12 mixes + smoke
  python eval/build_dataset.py --n 4 --seed 7 --no-video
  python eval/build_dataset.py --smoke-only
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, DATA_DIR, MIXES_DIR, RAW_DIR, load_audio, load_splits, read_manifest  # noqa: E402

SR = 44100
NV_WEIGHTS = {"moaning": 3.0, "panting": 2.0, "sighing": 2.0, "screaming": 1.5, "laughing": 1.5,
              "crying": 1.5, "yawning": 1.0, "coughing": 0.5, "throat-clearing": 0.5, "sneezing": 0.3}
BREATHY = {"sighing", "panting", "yawning"}
ESC_NOISE = ["rain", "sea_waves", "crackling_fire", "wind", "water_drops", "pouring_water", "vacuum_cleaner",
             "washing_machine", "clock_tick", "keyboard_typing", "door_wood_knock", "engine", "train",
             "airplane", "church_bells", "chirping_birds", "thunderstorm", "toilet_flush", "footsteps", "dog"]
VOCAL_MUSIC = "Karissa_Hobbs_-_Lets_Go_Fishin"


# ----------------------------------------------------------------------------- pools
class Pools:
    def __init__(self):
        rows = read_manifest()
        sp = load_splits()["speakers"]
        test = [r for r in rows if sp.get(r["speaker"]) == "test"]
        self.speech = defaultdict(lambda: defaultdict(list))   # gender -> speaker -> rows
        self.digits = defaultdict(list)                        # gender -> rows (AudioMNIST)
        self.nv = defaultdict(lambda: defaultdict(list))       # gender -> category -> rows
        for r in test:
            if r["kind"] == "speech" and r["source"] == "audiomnist":
                self.digits[r["gender"]].append(r)
            elif r["kind"] == "speech":
                self.speech[r["gender"]][r["speaker"]].append(r)
            elif r["category"] in NV_WEIGHTS:
                self.nv[r["gender"]][r["category"]].append(r)
        self._cache: dict[str, np.ndarray] = {}

    def audio(self, row: dict) -> np.ndarray:
        p = row["path"]
        if p not in self._cache:
            if len(self._cache) > 400:
                self._cache.clear()
            y = load_audio(DATA_DIR / p, sr=SR)
            self._cache[p] = y
        return self._cache[p]

    def speaker_f0(self) -> dict:
        """Median F0 per test speech speaker (cached) for the low-F / high-M hard cases."""
        path = CACHE_DIR / "speaker_f0.json"
        if path.exists():
            return json.loads(path.read_text())
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from voicefilter.assets.gender_head import pitch_track
        out = {}
        for g in ("F", "M"):
            for spk, rows in sorted(self.speech[g].items()):
                f0s = []
                for r in sorted(rows, key=lambda r: r["clip_id"])[:3]:
                    y = load_audio(DATA_DIR / r["path"], sr=16000)[: 16000 * 5]
                    f0, vf, _ = pitch_track(y)
                    f0s += list(f0[vf])
                out[spk] = float(np.median(f0s)) if f0s else float("nan")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        return out


# ----------------------------------------------------------------------------- dsp
def trim(y: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    import librosa
    if len(y) < 2048:
        return y
    yt, _ = librosa.effects.trim(y, top_db=top_db, frame_length=1024, hop_length=256)
    return yt if len(yt) > 0 else y


def active_intervals(y: np.ndarray, rel_db: float = 30.0, min_gap: float = 0.15, min_len: float = 0.05):
    """Active sub-intervals (s) of a dry clip: 10 ms frames within rel_db of the peak frame."""
    hop = SR // 100
    n = max(1, len(y) // hop)
    e = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(e)
    on = db > db.max() - rel_db
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, n])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * 0.01 < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [[a * 0.01, b * 0.01] for a, b in merged if (b - a) * 0.01 >= min_len] or [[0.0, len(y) / SR]]


def rms_norm(y: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    hop = 441
    frames = y[: len(y) // hop * hop].reshape(-1, hop) if len(y) >= hop else y.reshape(1, -1)
    e = np.sqrt((frames ** 2).mean(1) + 1e-12)
    act = e[20 * np.log10(e) > 20 * np.log10(e.max()) - 30]
    r = float(np.sqrt((act ** 2).mean())) if len(act) else 1e-3
    return y * (10 ** (target_db / 20) / max(r, 1e-6))


def synth_rir(rng: np.random.Generator, rt60: float, sr: int = SR) -> np.ndarray:
    """Stereo exponentially decaying noise RIR with a direct path (2, n)."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    decay = np.exp(-6.9 * t / rt60)
    out = []
    for _ in range(2):
        h = rng.standard_normal(n) * decay * 0.25
        h[: int(0.003 * sr)] *= np.linspace(0, 1, int(0.003 * sr))
        pre = int(rng.uniform(0.005, 0.02) * sr)
        h = np.concatenate([np.zeros(pre), h])
        h[0] = 1.0
        out.append(h / np.sqrt((h ** 2).sum()))
    m = max(len(h) for h in out)
    return np.stack([np.pad(h, (0, m - len(h))) for h in out])


def pink_noise(rng: np.random.Generator, n: int, beta: float = 1.0) -> np.ndarray:
    X = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    y = np.fft.irfft(X / f ** (beta / 2), n)
    return y / (np.std(y) + 1e-9)


def crossfade_concat(pieces: list[np.ndarray], fade: int) -> np.ndarray:
    out = pieces[0]
    for p in pieces[1:]:
        f = min(fade, out.shape[-1] // 2, p.shape[-1] // 2)
        w = np.linspace(0, 1, f)
        mid = out[..., -f:] * (1 - w) + p[..., :f] * w
        out = np.concatenate([out[..., :-f], mid, p[..., f:]], axis=-1)
    return out


def background(kind: str, n: int, rng: np.random.Generator, prng: random.Random) -> tuple[np.ndarray, dict]:
    if kind in ("music", "music_vocals"):
        tracks = sorted(p for p in (RAW_DIR / "music").glob("*.ogg")
                        if (VOCAL_MUSIC in p.stem) == (kind == "music_vocals") and "trumpet" not in p.stem)
        pieces, used, total = [], [], 0
        while total < n + 2 * SR:
            p = prng.choice(tracks)
            y = load_audio(p, sr=SR, mono=False)
            if y.shape[0] == 1:
                y = np.repeat(y, 2, 0)
            s = prng.randrange(0, max(1, y.shape[1] - 20 * SR)) if not pieces else 0
            y = y[:, s:]
            pieces.append(y)
            used.append(p.stem)
            total += y.shape[1] - SR
        bed = crossfade_concat(pieces, SR)[:, :n]
        info = {"type": kind, "tracks": used}
    elif kind == "esc_noise":
        import csv as _csv
        meta = list(_csv.DictReader(open(RAW_DIR / "esc50" / "esc50.csv", encoding="utf-8")))
        have = {p.name for p in (RAW_DIR / "esc50" / "audio").glob("*.wav")}
        cats = prng.sample(ESC_NOISE, 3)
        files = sorted(r["filename"] for r in meta if r["category"] in cats and r["filename"] in have)
        pieces, total = [], 0
        while total < n + 2 * SR:
            y = load_audio(RAW_DIR / "esc50" / "audio" / prng.choice(files), sr=SR)
            y = rms_norm(y, -30)
            pieces.append(np.stack([y, np.roll(y, 300)]))
            total += len(y) - SR // 2
        bed = crossfade_concat(pieces, SR // 2)[:, :n]
        floor = np.stack([pink_noise(rng, n), pink_noise(rng, n)]) * 10 ** (-45 / 20)
        bed = bed + floor
        info = {"type": kind, "categories": cats}
    else:  # synthetic: pink/brown noise + mains hum + slow modulation (room tone / AC)
        beta = float(rng.uniform(1.0, 2.0))
        bed = np.stack([pink_noise(rng, n, beta), pink_noise(rng, n, beta)])
        t = np.arange(n) / SR
        mains = float(prng.choice([50.0, 60.0]))
        hum = sum((0.3 / k) * np.sin(2 * np.pi * mains * k * t + rng.uniform(0, 6.28)) for k in range(1, 5))
        mod = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t)
        bed = bed * mod + hum
        info = {"type": "synthetic_noise", "beta": beta, "mains_hz": mains}
    return bed.astype(np.float64), info


# ----------------------------------------------------------------------------- scenes
def pick_nv(pools: Pools, gender: str, prng: random.Random, used: set, breathy: bool = False) -> dict:
    cats = [c for c in NV_WEIGHTS if pools.nv[gender][c]]
    if breathy:
        cats = [c for c in cats if c in BREATHY] or cats
    cat = prng.choices(cats, weights=[NV_WEIGHTS[c] for c in cats])[0]
    rows = pools.nv[gender][cat]
    fresh = [r for r in rows if r["clip_id"] not in used] or rows
    r = prng.choice(sorted(fresh, key=lambda r: r["clip_id"]))
    used.add(r["clip_id"])
    return r


def build_scene(idx: int, seed: int, pools: Pools, f0: dict, out_dir: Path, duration: float | None = None,
                n_speakers: int | None = None, composition: str | None = None, video: bool = True,
                script: list | None = None, bg_kind: str | None = None) -> dict:
    """Random scene; `script` = [(speaker_index, "speech"|"nv"|"short", start_s, max_len_s), ...]
    replaces the random timeline (used for the smoke mix)."""
    prng = random.Random(seed * 1000 + idx)
    rng = np.random.default_rng(seed * 1000 + idx)
    T = duration or prng.uniform(60, 120)
    n = int(T * SR)
    comp = composition or {4: "F", 5: "M"}.get(idx % 6, "mixed")
    ns = n_speakers or prng.choice([2, 2, 3, 3, 4])
    if comp == "mixed":
        n_f = prng.randint(1, ns - 1)
        genders = ["F"] * n_f + ["M"] * (ns - n_f)
    else:
        genders = [comp] * ns
    hard = []
    # speakers
    speakers = []
    used_spk, used_nv = set(), set()
    for k, g in enumerate(genders):
        cands = sorted(s for s in pools.speech[g] if s not in used_spk)
        spk = prng.choice(cands)
        if k == 0 and g == "F" and idx % 4 == 2:
            spk = min((s for s in cands if np.isfinite(f0.get(s, np.nan))), key=lambda s: f0[s])
            hard.append(f"low_pitched_female:{spk}:{f0[spk]:.0f}Hz")
        if g == "M" and idx % 4 == 3 and not any(h.startswith("high_pitched_male") for h in hard):
            spk = max((s for s in cands if np.isfinite(f0.get(s, np.nan))), key=lambda s: f0[s])
            hard.append(f"high_pitched_male:{spk}:{f0[spk]:.0f}Hz")
        used_spk.add(spk)
        speakers.append({"id": f"S{k + 1}", "gender": g, "source_speaker": spk, "f0_median": f0.get(spk),
                         "gain_db": prng.uniform(-4, 4), "pan": prng.uniform(-0.5, 0.5),
                         "clips": sorted(pools.speech[g][spk], key=lambda r: r["clip_id"])})
    reverb = idx % 3 == 1 and not script
    rirs = {s["id"]: synth_rir(rng, prng.uniform(0.3, 0.9)) for s in speakers} if reverb else {}
    wet = prng.uniform(0.25, 0.5) if reverb else 0.0

    stems = {"F": np.zeros((2, n)), "M": np.zeros((2, n))}
    events = []
    t = prng.uniform(0.5, 2.0)
    last = None
    speech_turn = defaultdict(int)
    p_ovl = 0.55
    forced = None
    script = list(script or [])
    scripted = bool(script)
    max_len = None
    while t < T - 1.0:
        if scripted:
            if not script:
                break
            si, stype, t, max_len = script.pop(0)
            spk = speakers[si]
        elif forced is not None:
            spk, forced = forced, None
        elif last is not None and prng.random() < 0.75 and len(speakers) > 1:
            spk = prng.choice([s for s in speakers if s["id"] != last["id"]])
        else:
            spk = prng.choice(speakers)
        g = spk["gender"]
        r = prng.random()
        if scripted:
            r = {"short": 0.0, "nv": 0.3, "speech": 0.9}[stype]
        if r < 0.1:  # very short event (<0.4 s)
            if not scripted and prng.random() < 0.5 and pools.digits[g]:
                row = prng.choice(pools.digits[g])
                y = trim(pools.audio(row))
                kind, cat = "speech", "short_digit"
            else:
                row = pick_nv(pools, g, prng, used_nv)
                y = trim(pools.audio(row))
                kind, cat = "nonverbal", f"short_{row['category']}"
            L = int(prng.uniform(0.15, 0.4) * SR)
            if len(y) > L:
                env = np.convolve(y ** 2, np.ones(441), "same")
                c = int(np.argmax(env))
                s0 = int(np.clip(c - L // 2, 0, len(y) - L))
                y = y[s0:s0 + L]
            y = y.copy()
            fade = max(1, min(len(y) // 4, int(0.01 * SR)))
            y[:fade] *= np.linspace(0, 1, fade)
            y[-fade:] *= np.linspace(1, 0, fade)
            clips = [(y, kind, cat, row)]
        elif r < 0.45:  # non-verbal burst: 1-3 sounds from the same speaker
            clips = []
            for _ in range(1 if scripted else prng.choice([1, 1, 2, 3])):
                row = pick_nv(pools, g, prng, used_nv, breathy=prng.random() < 0.25)
                y = trim(pools.audio(row))[: int(6 * SR)]
                clips.append((y, "nonverbal", row["category"], row))
        else:
            row = spk["clips"][speech_turn[spk["id"]] % len(spk["clips"])]
            speech_turn[spk["id"]] += 1
            y = trim(pools.audio(row))[: int(8 * SR)]
            clips = [(y, "speech", row["category"], row)]
        start = t
        n_before = len(events)
        for y, kind, cat, src in clips:
            if max_len:
                y = y[: int(max_len * SR)]
            if start + len(y) / SR > T - 0.2:
                break
            y = rms_norm(y) * 10 ** ((spk["gain_db"] + prng.uniform(-6, 3)) / 20)
            s0 = int(start * SR)
            pan = spk["pan"]
            st = np.stack([y * np.sqrt(1 - pan), y * np.sqrt(1 + pan)])
            if reverb:
                from scipy.signal import fftconvolve
                h = rirs[spk["id"]]
                wetsig = np.stack([fftconvolve(st[c], h[c]) for c in range(2)])
                st = np.pad(st, ((0, 0), (0, wetsig.shape[1] - st.shape[1]))) * (1 - wet) + wetsig * wet * 1.5
            e = min(n, s0 + st.shape[1])
            stems[g][:, s0:e] += st[:, : e - s0]
            act = [[start + a0, start + b0] for a0, b0 in active_intervals(y)]
            events.append({"start": round(start, 3), "end": round(start + len(y) / SR, 3), "speaker": spk["id"],
                           "gender": g, "kind": kind, "category": cat, "overlap": False,
                           "active": [[round(a0, 3), round(b0, 3)] for a0, b0 in act],
                           "source_clip": src["clip_id"], "source_speaker": src["speaker"],
                           "short": len(y) / SR < 0.4, "breathy": cat in BREATHY})
            start = start + len(y) / SR + prng.uniform(0.1, 0.6)
        if len(events) == n_before and not scripted:  # nothing fitted before the end
            break
        if scripted:
            continue
        last = spk
        cur = events[-1]
        others = [s for s in speakers if s["gender"] != g]
        if others and prng.random() < p_ovl:
            # next event comes from the other gender and starts before this one ends
            ln = cur["end"] - cur["start"]
            t = max(cur["start"] + 0.2, cur["end"] - prng.uniform(0.6, min(2.5, max(0.7, ln))))
            forced = prng.choice(others)
        else:
            gap = prng.uniform(0.15, 1.2) if prng.random() > 0.1 else prng.uniform(2.0, 4.0)
            t = max(e["end"] for e in events) + gap
        # never start inside the stretch already used by the same speaker
        t = max(t, max((e["end"] for e in events if forced is not None and e["speaker"] == forced["id"]),
                       default=0.0))

    # overlap flags (male+female simultaneously)
    for e in events:
        for o in events:
            if o is e or o["gender"] == e["gender"]:
                continue
            if min(e["end"], o["end"]) - max(e["start"], o["start"]) > 0.05:
                e["overlap"] = True
                break
    bg_kind = bg_kind or {0: "music", 1: "esc_noise", 2: "synthetic"}[idx % 3]
    if idx % 12 == 11 and not scripted:
        bg_kind = "music_vocals"
        hard.append("background_with_sung_vocals")
    bed, bg_info = background(bg_kind, n, rng, prng)
    vocals = stems["F"] + stems["M"]
    act_mask = np.zeros(n, bool)
    for e in events:
        for a, b in e["active"]:
            act_mask[int(a * SR):int(b * SR)] = True
    v_rms = np.sqrt((vocals[:, act_mask] ** 2).mean()) if act_mask.any() else 1e-3
    b_rms = np.sqrt((bed ** 2).mean())
    snr = prng.uniform(-5, 15)
    bed *= v_rms / (b_rms * 10 ** (snr / 20))
    mix = vocals + bed
    g = min(1.0, 0.89 / (np.abs(mix).max() + 1e-9))
    stems = {k: v * g for k, v in stems.items()}
    bed *= g
    mix = stems["F"] + stems["M"] + bed
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    for name, x in [("mix", mix), ("vocals_female", stems["F"]), ("vocals_male", stems["M"]), ("background", bed)]:
        sf.write(str(out_dir / f"{name}.wav"), x.T.astype(np.float32), SR, subtype="FLOAT")
    dur_f = sum(b - a for e in events if e["gender"] == "F" for a, b in e["active"])
    dur_m = sum(b - a for e in events if e["gender"] == "M" for a, b in e["active"])
    fm = _frame_classes(events, T)
    vocal_frames = (fm > 0).sum()
    truth = {"name": out_dir.name, "duration": round(n / SR, 6), "sr": SR, "events": events,
             "speakers": [{k: v for k, v in s.items() if k != "clips"} for s in speakers],
             "meta": {"seed": seed, "index": idx, "composition": comp, "n_speakers": ns, "background": bg_info,
                      "snr_db": round(snr, 2), "reverb": reverb, "wet": round(wet, 3), "hard_cases": hard,
                      "female_active_s": round(dur_f, 2), "male_active_s": round(dur_m, 2),
                      "overlap_share_of_vocal_time": round(float((fm == 3).sum() / max(1, vocal_frames)), 4),
                      "n_events": len(events), "n_short": sum(e["short"] for e in events),
                      "n_nonverbal": sum(e["kind"] == "nonverbal" for e in events),
                      "stems_sum_to_mix": True}}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    if video:
        make_video(out_dir / "mix.wav", out_dir / "mix.mp4", n / SR)
    return truth


def _frame_classes(events, T, hop=0.01):
    """0 none, 1 female only, 2 male only, 3 both (10 ms frames, active sub-intervals)."""
    nf = int(np.ceil(T / hop))
    f = np.zeros(nf, bool)
    m = np.zeros(nf, bool)
    for e in events:
        tgt = f if e["gender"] == "F" else m
        for a, b in e["active"]:
            tgt[int(round(a / hop)):int(round(b / hop))] = True
    return f * 1 + m * 2


def make_video(wav: Path, mp4: Path, dur: float) -> None:
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found, skipping video")
        return
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
           f"testsrc2=size=640x360:rate=25:duration={dur:.3f}", "-i", str(wav), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-b:a", "192k", "-shortest", str(mp4)]
    subprocess.run(cmd, check=True)


def build_smoke(pools: Pools, out_dir: Path, video: bool = True) -> dict:
    """10 s mix: female speech, overlapping male non-verbal, short female NV, male speech."""
    script = [(0, "speech", 0.5, 3.2), (1, "nv", 3.0, 2.0), (0, "short", 5.6, None), (1, "speech", 6.5, 3.0)]
    return build_scene(0, 99, pools, {}, out_dir, duration=10.0, n_speakers=2, composition="mixed", video=video,
                       script=script, bg_kind="esc_noise")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=str(MIXES_DIR))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    pools = Pools()
    out = Path(a.out)
    index = []
    t = build_smoke(pools, out / "smoke", video=not a.no_video)
    index.append(_summary(t))
    if not a.smoke_only:
        f0 = pools.speaker_f0()
        for i in range(a.n):
            t = build_scene(i, a.seed, pools, f0, out / f"mix_{i:02d}", video=not a.no_video)
            index.append(_summary(t))
            print(json.dumps(index[-1]))
    (out / "index.json").write_text(json.dumps(index, indent=1))


def _summary(t: dict) -> dict:
    m = t["meta"]
    return {"name": t["name"], "duration": t["duration"], "composition": m["composition"],
            "n_events": m["n_events"], "n_nonverbal": m["n_nonverbal"], "n_short": m["n_short"],
            "overlap_share": m["overlap_share_of_vocal_time"], "snr_db": m["snr_db"], "reverb": m["reverb"],
            "background": m["background"]["type"], "hard_cases": m["hard_cases"]}


if __name__ == "__main__":
    main()
