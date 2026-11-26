"""Download the labelled evaluation / training data into eval/data/ (git-ignored).

Reproducible: every subset choice is a deterministic function of file names
(seeded RNG / sha1), so re-running gives the same clips. Re-running is cheap:
files that already exist are skipped.

Sources (see eval/DATA.md for licenses and coverage):
  ears       EARS (GitHub release zips, CC BY-NC 4.0): 103 speakers with gender; screams, crying,
             laughter, yelling, yawns, coughs, pain / ecstasy expressions + regular / whispered
             sentence; first 14 s per file via HTTP range, loudest 3 s kept.  ~1.4 GB download
  slr99      Deeply Nonverbal Vocalization Dataset (OpenSLR 99): 16 non-verbal classes,
             ~410 speakers, sex labels.                                    ~45 MB
  libri      LibriSpeech dev-clean + test-clean (OpenSLR 12), SPEAKERS.TXT gender.
             Only the first 15 utterances per speaker are kept on disk.   ~680 MB download
  crowd      Google crowdsourced multi-speaker sets (OpenSLR 61/63/66/69/70/76/78/83/86):
             12 languages / dialects, gender-separated zips. Only selected members are
             fetched with HTTP range requests (no full zip download).      ~250 MB
  crema      CREMA-D (GitHub LFS): 91 actors, acted emotional speech
             (anger/fear/... incl. shouting), demographics CSV.            ~75 MB
  audiomnist AudioMNIST (GitHub): 60 speakers, isolated spoken digits (~0.5 s),
             gender metadata -> very short events.                         ~30 MB
  music      librosa example audio (GitHub librosa/data, CC-BY / CC0 / PD).~25 MB
  esc50      ESC-50 non-human environmental sounds (GitHub, CC BY-NC).    ~35 MB

Usage:
  python eval/fetch_data.py                 # everything
  python eval/fetch_data.py --only slr99 crema
  python eval/fetch_data.py --manifest-only # rebuild clips.csv + splits.json from disk

Hugging Face file downloads are blocked on the build machine (CDN host refused by the
proxy), so all sources here are on openslr.org / GitHub.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import struct
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, MANIFEST, RAW_DIR, SPLITS, stable_unit, write_manifest  # noqa: E402

UA = {"User-Agent": "voicefilter-eval-fetch/1.0"}
SLR = "https://www.openslr.org/resources"
GH_RAW = "https://raw.githubusercontent.com"
GH_MEDIA = "https://media.githubusercontent.com/media"


# --------------------------------------------------------------------------- http
def http_get(url: str, headers: dict | None = None, retries: int = 4, timeout: int = 120) -> bytes:
    h = dict(UA)
    h.update(headers or {})
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = http_get(url)
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url = url
        self.pos = 0
        self.block = block
        self.cache: dict[int, bytes] = {}
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range")
            if not cr:
                raise RuntimeError(f"server does not support ranges: {url}")
            self.size = int(cr.split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            start = i * self.block
            end = min(self.size, start + self.block) - 1
            self.cache[i] = http_get(self.url, {"Range": f"bytes={start}-{end}"})
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            bi, off = divmod(self.pos, self.block)
            chunk = self._block(bi)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def zip_member_direct(url: str, info: zipfile.ZipInfo, size_total: int) -> bytes:
    """Fetch one stored/deflated member with a single range request."""
    import zlib
    start = info.header_offset
    hdr = http_get(url, {"Range": f"bytes={start}-{start + 29}"})
    n_name, n_extra = struct.unpack("<HH", hdr[26:30])
    data_start = start + 30 + n_name + n_extra
    raw = http_get(url, {"Range": f"bytes={data_start}-{data_start + info.compress_size - 1}"})
    if info.compress_type == zipfile.ZIP_STORED:
        return raw
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    raise RuntimeError(f"unsupported compression {info.compress_type}")


# --------------------------------------------------------------------------- sources
def fetch_slr99() -> None:
    out = RAW_DIR / "slr99"
    if (out / "NonverbalVocalization" / "Nonverbal_Vocalization.json").exists():
        print("slr99: present")
        return
    tgz = download(f"{SLR}/99/NonverbalVocalization.tgz", out / "NonverbalVocalization.tgz")
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if "/._" not in m.name and not m.name.startswith("._")]
        tf.extractall(out, members=members, filter="data")
    tgz.unlink()
    print("slr99: done")


LIBRI_SETS = ["dev-clean", "test-clean"]
LIBRI_PER_SPEAKER = 15


def fetch_libri() -> None:
    out = RAW_DIR / "librispeech"
    for name in LIBRI_SETS:
        marker = out / f".{name}.done"
        if marker.exists():
            print(f"libri {name}: present")
            continue
        url = f"{SLR}/12/{name}.tar.gz"
        print(f"libri {name}: streaming {url}")
        req = urllib.request.Request(url, headers=UA)
        kept = defaultdict(int)
        with urllib.request.urlopen(req, timeout=600) as r:
            with tarfile.open(fileobj=r, mode="r|gz") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    base = m.name.split("/")[-1]
                    keep = False
                    if base in ("SPEAKERS.TXT", "LICENSE.TXT", "README.TXT"):
                        keep = True
                    elif base.endswith(".flac"):
                        spk = base.split("-")[0]
                        if kept[spk] < LIBRI_PER_SPEAKER:
                            kept[spk] += 1
                            keep = True
                    if keep:
                        dest = out / m.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        fh = tf.extractfile(m)
                        dest.write_bytes(fh.read())
        marker.write_text("ok")
        print(f"libri {name}: kept {sum(kept.values())} files from {len(kept)} speakers")


# (slr id, zip name, lang, gender, line index file, filename prefix or None)
CROWD = [
    (83, "midlands_english_female.zip", "en-GB-midlands", "F", "line_index_all.csv", "mif"),
    (83, "midlands_english_male.zip", "en-GB-midlands", "M", "line_index_all.csv", "mim"),
    (83, "irish_english_male.zip", "en-IE", "M", "line_index_all.csv", "irm"),
    (83, "scottish_english_female.zip", "en-GB-scottish", "F", "line_index_all.csv", "scf"),
    (83, "scottish_english_male.zip", "en-GB-scottish", "M", "line_index_all.csv", "scm"),
    (83, "welsh_english_female.zip", "en-GB-welsh", "F", "line_index_all.csv", "wef"),
    (83, "welsh_english_male.zip", "en-GB-welsh", "M", "line_index_all.csv", "wem"),
    (61, "es_ar_female.zip", "es-AR", "F", "line_index_female.tsv", None),
    (61, "es_ar_male.zip", "es-AR", "M", "line_index_male.tsv", None),
    (70, "en_ng_female.zip", "en-NG", "F", "line_index_female.tsv", None),
    (70, "en_ng_male.zip", "en-NG", "M", "line_index_male.tsv", None),
    (86, "yo_ng_female.zip", "yo-NG", "F", "line_index_female.tsv", None),
    (86, "yo_ng_male.zip", "yo-NG", "M", "line_index_male.tsv", None),
    (63, "ml_in_female.zip", "ml-IN", "F", "line_index_female.tsv", None),
    (63, "ml_in_male.zip", "ml-IN", "M", "line_index_male.tsv", None),
    (78, "gu_in_female.zip", "gu-IN", "F", "line_index_female.tsv", None),
    (78, "gu_in_male.zip", "gu-IN", "M", "line_index_male.tsv", None),
    (69, "ca_es_female.zip", "ca-ES", "F", "line_index_female.tsv", None),
    (69, "ca_es_male.zip", "ca-ES", "M", "line_index_male.tsv", None),
    (76, "eu_es_female.zip", "eu-ES", "F", "line_index_female.tsv", None),
    (76, "eu_es_male.zip", "eu-ES", "M", "line_index_male.tsv", None),
    (66, "te_in_female.zip", "te-IN", "F", "line_index_female.tsv", None),
    (66, "te_in_male.zip", "te-IN", "M", "line_index_male.tsv", None),
]
CROWD_SPEAKERS = 10
CROWD_CLIPS = 4


def _crowd_ids(slr: int, index_name: str, prefix: str | None) -> list[str]:
    cache = RAW_DIR / "crowd" / f"slr{slr}_{index_name}"
    txt = download(f"{SLR}/{slr}/{index_name}", cache).read_text(encoding="utf-8", errors="replace")
    ids = []
    for line in txt.splitlines():
        parts = [p.strip() for p in re.split(r"[\t,]", line) if p.strip()]
        for p in parts[:2]:
            if re.fullmatch(r"[a-z]{3}_\d{5}_\d+", p):
                ids.append(p)
                break
    if prefix:
        ids = [i for i in ids if i.startswith(prefix + "_")]
    return ids


def fetch_crowd() -> None:
    for slr, zname, lang, gender, index_name, prefix in CROWD:
        out = RAW_DIR / "crowd" / zname.replace(".zip", "")
        marker = out / ".done"
        if marker.exists():
            continue
        ids = _crowd_ids(slr, index_name, prefix)
        by_spk = defaultdict(list)
        for i in ids:
            by_spk[i.split("_")[1]].append(i)
        rng = random.Random(f"crowd-{zname}")
        spks = sorted(by_spk)
        rng.shuffle(spks)
        chosen = {}
        for s in spks[:CROWD_SPEAKERS]:
            files = sorted(by_spk[s])
            rng.shuffle(files)
            for f in files[:CROWD_CLIPS]:
                chosen[f + ".wav"] = s
        url = f"{SLR}/{slr}/{zname}"
        rf = HttpRangeFile(url)
        zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
        infos = {Path(i.filename).name: i for i in zf.infolist() if i.filename.endswith(".wav")}
        out.mkdir(parents=True, exist_ok=True)
        todo = [(n, infos[n]) for n in chosen if n in infos and not (out / n).exists()]

        def job(item):
            n, info = item
            (out / n).write_bytes(zip_member_direct(url, info, rf.size))

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(job, todo))
        (out / "speakers.json").write_text(json.dumps({"lang": lang, "gender": gender, "files": chosen}, indent=1))
        marker.write_text("ok")
        print(f"crowd {zname}: {len(chosen)} clips / {len(set(chosen.values()))} speakers")
    for f in ["LICENSE"]:
        download(f"{SLR}/83/{f}", RAW_DIR / "crowd" / "LICENSE_CC-BY-SA-4.0.txt")


CREMA_PER_ACTOR = 8


def fetch_crema() -> None:
    out = RAW_DIR / "crema"
    base_raw = f"{GH_RAW}/CheyneyComputerScience/CREMA-D/master"
    download(f"{base_raw}/VideoDemographics.csv", out / "VideoDemographics.csv")
    download(f"{base_raw}/SentenceFilenames.csv", out / "SentenceFilenames.csv")
    names = []
    with open(out / "SentenceFilenames.csv", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            for c in row:
                if re.fullmatch(r"\d{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}", c.strip()):
                    names.append(c.strip())
    by_actor = defaultdict(list)
    for n in sorted(set(names)):
        by_actor[n[:4]].append(n)
    chosen = []
    for actor, files in sorted(by_actor.items()):
        rng = random.Random(f"crema-{actor}")
        by_emo = defaultdict(list)
        for f in files:
            by_emo[f.split("_")[2]].append(f)
        pick = []
        for emo in sorted(by_emo):
            pick.append(rng.choice(sorted(by_emo[emo])))
        rest = sorted(set(files) - set(pick))
        rng.shuffle(rest)
        pick += rest[: max(0, CREMA_PER_ACTOR - len(pick))]
        chosen += pick[:CREMA_PER_ACTOR]
    (out / "wav").mkdir(parents=True, exist_ok=True)

    def job(n):
        dest = out / "wav" / f"{n}.wav"
        if not dest.exists():
            try:
                download(f"{GH_MEDIA}/CheyneyComputerScience/CREMA-D/master/AudioWAV/{n}.wav", dest)
            except Exception as e:  # noqa: BLE001
                print("crema miss", n, e)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(job, chosen))
    print(f"crema: {len(chosen)} clips from {len(by_actor)} actors")


AUDIOMNIST_PER_SPEAKER = 6


def fetch_audiomnist() -> None:
    out = RAW_DIR / "audiomnist"
    base = f"{GH_RAW}/soerenab/AudioMNIST/master"
    meta = json.loads(download(f"{base}/data/audioMNIST_meta.txt", out / "audioMNIST_meta.txt").read_text())
    download(f"{base}/LICENSE", out / "LICENSE")
    jobs = []
    for spk in sorted(meta):
        rng = random.Random(f"amnist-{spk}")
        for _ in range(AUDIOMNIST_PER_SPEAKER):
            d, i = rng.randrange(10), rng.randrange(50)
            jobs.append((spk, f"{d}_{spk}_{i}.wav"))

    def job(item):
        spk, n = item
        try:
            download(f"{base}/data/{spk}/{n}", out / spk / n)
        except Exception as e:  # noqa: BLE001
            print("audiomnist miss", n, e)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(job, jobs))
    print(f"audiomnist: {len(jobs)} clips")


MUSIC = {  # librosa example data: name -> has_vocals
    "Kevin_MacLeod_-_Vibe_Ace": False,
    "admiralbob77_-_Choice_-_Drum-bass": False,
    "Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy": False,
    "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra)": False,
    "147793__setuniman__sweet-waltz-0i-22mi": False,
    "442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime": False,
    "sorohanro_-_solo-trumpet-06": False,
    "Karissa_Hobbs_-_Lets_Go_Fishin": True,
}


def fetch_music() -> None:
    out = RAW_DIR / "music"
    base = f"{GH_RAW}/librosa/data/main/audio"
    info = {}
    for name, vocals in MUSIC.items():
        q = urllib.request.quote(name, safe="")
        ogg = download(f"{base}/{q}.hq.ogg", out / f"{name}.ogg")
        if ogg.read_bytes()[:4] != b"OggS":
            ogg.unlink()
            raise RuntimeError(f"music download failed: {name}")
        lic = download(f"{base}/{q}.txt", out / f"{name}.txt").read_text(errors="replace")
        info[name] = {"has_vocals": vocals, "license_text": lic.strip()}
    (out / "music.json").write_text(json.dumps(info, indent=1))
    print(f"music: {len(info)} tracks")


ESC_CATEGORIES = ["rain", "sea_waves", "crackling_fire", "wind", "water_drops", "pouring_water",
                  "vacuum_cleaner", "washing_machine", "clock_tick", "keyboard_typing",
                  "door_wood_knock", "engine", "train", "airplane", "church_bells",
                  "chirping_birds", "thunderstorm", "toilet_flush", "footsteps", "dog"]
ESC_PER_CAT = 4


def fetch_esc50() -> None:
    out = RAW_DIR / "esc50"
    base = f"{GH_RAW}/karolpiczak/ESC-50/master"
    meta = download(f"{base}/meta/esc50.csv", out / "esc50.csv")
    download(f"{base}/LICENSE", out / "LICENSE")
    rows = list(csv.DictReader(open(meta, encoding="utf-8")))
    jobs = []
    for cat in ESC_CATEGORIES:
        files = sorted(r["filename"] for r in rows if r["category"] == cat)
        rng = random.Random(f"esc-{cat}")
        rng.shuffle(files)
        jobs += [(cat, f) for f in files[:ESC_PER_CAT]]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: download(f"{base}/audio/{j[1]}", out / "audio" / j[1]), jobs))
    (out / "chosen.json").write_text(json.dumps(jobs, indent=0))
    print(f"esc50: {len(jobs)} clips")


# EARS (facebookresearch/ears_dataset, GitHub release zips, CC BY-NC 4.0): 107 speakers with
# self-reported gender, 48 kHz studio recordings incl. screams, crying, laughter, yelling, coughs,
# yawns and freeform pain / ecstasy / desire expressions. The zip members are STORED, so we read only
# the first EARS_SECONDS of each wanted file with HTTP range requests and keep the loudest
# EARS_CHUNK-second window (FLAC, 48 kHz mono).
EARS_URL = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/{spk}.zip"
EARS_STATS = f"{GH_RAW}/facebookresearch/ears_dataset/main/speaker_statistics.json"
EARS_FILES = {  # file stem -> (kind, category)
    "nonverbal_screaming": ("nonverbal", "ears_screaming"),
    "nonverbal_crying": ("nonverbal", "ears_crying"),
    "nonverbal_laughter_open": ("nonverbal", "ears_laughing"),
    "nonverbal_yelling": ("nonverbal", "ears_yelling"),
    "vegetative_yawning": ("nonverbal", "ears_yawning"),
    "vegetative_coughing": ("nonverbal", "ears_coughing"),
    "emo_pain_freeform": ("nonverbal", "ears_pain"),
    "emo_extasy_freeform": ("nonverbal", "ears_ecstasy"),
    "sentences_01_regular": ("speech", "ears_regular"),
    "sentences_02_whisper": ("speech", "ears_whisper"),
}
EARS_SECONDS = 14.0
EARS_CHUNK = 3.0


def _parse_wav_prefix(b: bytes):
    """Return (sr, channels, bits, data_offset) from the first bytes of a RIFF/WAVE file."""
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", "not a wav"
    pos, fmt = 12, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", b[pos + 8:pos + 24])
            fmt = (sr, ch, bits, tag)
        if cid == b"data":
            return fmt[0], fmt[1], fmt[2], fmt[3], pos + 8
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk in prefix")


def fetch_ears() -> None:
    import numpy as np
    import soundfile as sf
    out = RAW_DIR / "ears"
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads(download(EARS_STATS, out / "speaker_statistics.json").read_text())
    download(f"{GH_RAW}/facebookresearch/ears_dataset/main/LICENSE", out / "LICENSE")
    for spk in sorted(stats):
        g = stats[spk].get("gender")
        if g not in ("male", "female"):
            continue
        marker = out / spk / ".done"
        if marker.exists():
            continue
        url = EARS_URL.format(spk=spk)
        rf = HttpRangeFile(url)
        zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
        infos = {Path(i.filename).stem: i for i in zf.infolist()}
        (out / spk).mkdir(exist_ok=True)

        def job(stem):
            info = infos.get(stem)
            dest = out / spk / f"{stem}.flac"
            if info is None or dest.exists() or info.compress_type != zipfile.ZIP_STORED:
                return
            start = info.header_offset
            hdr = http_get(rf.url, {"Range": f"bytes={start}-{start + 29}"})
            n_name, n_extra = struct.unpack("<HH", hdr[26:30])
            d0 = start + 30 + n_name + n_extra
            pre = http_get(rf.url, {"Range": f"bytes={d0}-{d0 + 4095}"})
            sr, ch, bits, tag, off = _parse_wav_prefix(pre)
            bps = bits // 8 * ch
            nbytes = min(info.file_size - off, int(EARS_SECONDS * sr) * bps)
            raw = http_get(rf.url, {"Range": f"bytes={d0 + off}-{d0 + off + nbytes - 1}"})
            raw = raw[: len(raw) // bps * bps]
            if tag == 3 and bits == 32:
                x = np.frombuffer(raw, "<f4").reshape(-1, ch).mean(1)
            elif bits == 16:
                x = np.frombuffer(raw, "<i2").reshape(-1, ch).mean(1) / 32768.0
            elif bits == 24:
                a = np.frombuffer(raw, np.uint8).reshape(-1, 3)
                v = (a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8) | (a[:, 2].astype(np.int32) << 16))
                v = np.where(v >= 1 << 23, v - (1 << 24), v)
                x = v.reshape(-1, ch).mean(1) / float(1 << 23)
            else:
                raise ValueError(f"unsupported wav format {tag}/{bits}")
            L = int(EARS_CHUNK * sr)
            if len(x) > L:
                hop = sr // 20
                e = np.array([np.mean(x[i:i + hop] ** 2) for i in range(0, len(x) - hop, hop)])
                w = int(EARS_CHUNK * 20)
                k = int(np.argmax(np.convolve(e, np.ones(w), "valid"))) if len(e) > w else 0
                x = x[k * hop:k * hop + L]
            sf.write(str(dest), x.astype(np.float32), sr, subtype="PCM_16")

        with ThreadPoolExecutor(4) as ex:
            list(ex.map(job, list(EARS_FILES)))
        marker.write_text("ok")
        print(f"ears {spk} ({g}) done", flush=True)


# --------------------------------------------------------------------------- manifest
DEEPLY_SEX = {0: "F", 1: "M"}  # from the dataset description PDF: {0: Female, 1: Male}


def _dur(p: Path) -> float:
    import soundfile as sf
    try:
        i = sf.info(str(p))
        return i.frames / float(i.samplerate)
    except Exception:  # noqa: BLE001
        return -1.0


def build_manifest() -> list[dict]:
    rows: list[dict] = []
    rel = lambda p: p.relative_to(DATA_DIR).as_posix()  # noqa: E731

    # Deeply non-verbal
    root = RAW_DIR / "slr99" / "NonverbalVocalization"
    if root.exists():
        meta = json.load(open(root / "Nonverbal_Vocalization.json", encoding="utf-8"))
        sexes = defaultdict(set)
        for cat, files in meta.items():
            for fn, m in files.items():
                sexes[m["speakerID"]].add(m["sex"])
        for cat, files in meta.items():
            for fn, m in sorted(files.items()):
                p = root / cat / fn
                if not p.exists() or len(sexes[m["speakerID"]]) != 1:
                    continue  # skip missing files and the 2 speaker ids with conflicting sex
                rows.append(dict(clip_id=f"dpl_{Path(fn).stem}", path=rel(p), source="deeply",
                                 speaker=f"dpl:{m['speakerID']}", gender=DEEPLY_SEX[m["sex"]],
                                 kind="nonverbal", category=cat, lang="ko", duration=_dur(p),
                                 license="CC-BY-NC-ND-4.0"))

    # LibriSpeech
    lroot = RAW_DIR / "librispeech" / "LibriSpeech"
    if lroot.exists():
        sex = {}
        for line in (lroot / "SPEAKERS.TXT").read_text().splitlines():
            if line.startswith(";"):
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) >= 2:
                sex[parts[0]] = parts[1]
        for p in sorted(lroot.rglob("*.flac")):
            spk = p.name.split("-")[0]
            if spk not in sex:
                continue
            rows.append(dict(clip_id=f"ls_{p.stem}", path=rel(p), source="librispeech",
                             speaker=f"ls:{spk}", gender=sex[spk], kind="speech", category="read",
                             lang="en-US", duration=_dur(p), license="CC-BY-4.0"))

    # crowdsourced OpenSLR
    croot = RAW_DIR / "crowd"
    for d in sorted(croot.glob("*")) if croot.exists() else []:
        sj = d / "speakers.json"
        if not sj.exists():
            continue
        info = json.loads(sj.read_text())
        for fn, spk in sorted(info["files"].items()):
            p = d / fn
            if p.exists():
                rows.append(dict(clip_id=f"gc_{p.stem}", path=rel(p), source="crowd",
                                 speaker=f"gc:{info['lang']}:{spk}", gender=info["gender"],
                                 kind="speech", category="read", lang=info["lang"],
                                 duration=_dur(p), license="CC-BY-SA-4.0"))

    # CREMA-D
    croot = RAW_DIR / "crema"
    if (croot / "VideoDemographics.csv").exists():
        demo = {r["ActorID"]: r for r in csv.DictReader(open(croot / "VideoDemographics.csv", encoding="utf-8"))}
        for p in sorted((croot / "wav").glob("*.wav")):
            actor = p.stem[:4]
            g = demo[actor]["Sex"][0].upper()
            emo = p.stem.split("_")[2].lower()
            rows.append(dict(clip_id=f"cr_{p.stem}", path=rel(p), source="crema",
                             speaker=f"cr:{actor}", gender=g, kind="speech",
                             category=f"acted_{emo}", lang="en-US", duration=_dur(p),
                             license="ODbL-1.0/DbCL-1.0"))

    # AudioMNIST
    aroot = RAW_DIR / "audiomnist"
    if (aroot / "audioMNIST_meta.txt").exists():
        meta = json.loads((aroot / "audioMNIST_meta.txt").read_text())
        for p in sorted(aroot.glob("*/*.wav")):
            spk = p.parent.name
            g = meta[spk]["gender"].strip().lower()[0].upper()
            rows.append(dict(clip_id=f"am_{p.stem}", path=rel(p), source="audiomnist",
                             speaker=f"am:{spk}", gender=g, kind="speech", category="digit",
                             lang="en/de", duration=_dur(p), license="MIT"))
    # EARS
    eroot = RAW_DIR / "ears"
    if (eroot / "speaker_statistics.json").exists():
        stats = json.loads((eroot / "speaker_statistics.json").read_text())
        for p in sorted(eroot.glob("p*/*.flac")):
            spk = p.parent.name
            g = {"male": "M", "female": "F"}.get(stats.get(spk, {}).get("gender"))
            if g is None or p.stem not in EARS_FILES:
                continue
            kind, cat = EARS_FILES[p.stem]
            rows.append(dict(clip_id=f"ears_{spk}_{p.stem}", path=rel(p), source="ears", speaker=f"ears:{spk}",
                             gender=g, kind=kind, category=cat, lang="en", duration=_dur(p),
                             license="CC-BY-NC-4.0"))
    rows = [r for r in rows if r["duration"] > 0 and r["gender"] in ("F", "M")]
    write_manifest(rows)
    return rows


TEST_FRACTION = {"deeply": 0.5, "ears": 0.5, "librispeech": 0.4, "crowd": 0.4, "crema": 0.4, "audiomnist": 0.4}


def build_splits(rows: list[dict]) -> dict:
    spk_gender = {}
    spk_source = {}
    for r in rows:
        spk_gender[r["speaker"]] = r["gender"]
        spk_source[r["speaker"]] = r["source"]
    speakers = {}
    for s in sorted(spk_gender):
        speakers[s] = "test" if stable_unit(s) < TEST_FRACTION.get(spk_source[s], 0.4) else "train"
    out = {"version": 1, "rule": "sha1('voicefilter-split-v1:'+speaker) < test_fraction[source] -> test",
           "test_fraction": TEST_FRACTION, "speakers": speakers}
    SPLITS.write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


def summarize(rows: list[dict], splits: dict) -> None:
    from collections import Counter
    c = Counter()
    for r in rows:
        c[(r["source"], r["kind"], splits["speakers"][r["speaker"]], r["gender"])] += 1
    for k in sorted(c):
        print("  ", k, c[k])
    print("total clips", len(rows), "speakers", len(splits["speakers"]))


FETCHERS = {"ears": fetch_ears, "slr99": fetch_slr99, "libri": fetch_libri, "crowd": fetch_crowd, "crema": fetch_crema,
            "audiomnist": fetch_audiomnist, "music": fetch_music, "esc50": fetch_esc50}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(FETCHERS), default=None)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        for name in args.only or list(FETCHERS):
            print(f"== {name}")
            FETCHERS[name]()
    rows = build_manifest()
    splits = build_splits(rows)
    summarize(rows, splits)
    print("manifest:", MANIFEST)


if __name__ == "__main__":
    main()
