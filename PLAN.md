# Voice Filter: Architecture and Implementation Plan

Goal: take a video and produce a copy whose audio keeps every female vocal sound
(speech, moans, grunts, other vocalizations) and the non-vocal background, while
male vocal sounds are removed. Runs locally on Windows 11 (RTX 3080 10 GB, i9-12900K),
usable from a phone on the same WiFi.

Model choices below are the starting point; `docs/MODELS.md` holds the research,
licenses and measured numbers, and wins where the two disagree.

## 1. Stack

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.12 managed by `uv`, installed **inside the app folder** | No system-wide Python needed; uninstall leaves no trace |
| ML | PyTorch (CUDA 12.x wheels on Windows), Hugging Face `transformers`, SpeechBrain | All models run on CUDA; CPU fallback for testing |
| Media | ffmpeg / ffprobe (existing on PATH, else portable build in `runtime/ffmpeg`) | Extract, mux with `-c:v copy` |
| Server | FastAPI + uvicorn, bound to `0.0.0.0:8765` | Simple, streaming uploads, range requests for audio |
| UI | Single page, vanilla HTML/CSS/JS (no build step), served by FastAPI | Mobile-first, nothing to compile on Windows |
| Jobs | One background worker thread (GPU is the bottleneck), jobs persisted as JSON in `data/jobs/<id>/` | Survives restarts; one GPU job at a time avoids VRAM fights |

## 2. Pipeline

```
video ──ffmpeg──▶ audio.wav (44.1 kHz stereo, float)
          │
          ▼  [separation]  vocals.wav + background.wav   (background = mix − vocals)
          ▼  [activity]    vocal events on the vocal stem (energy + VAD, merges gaps,
          │                splits at speaker changes) → candidate segments
          ▼  [features]    per segment & per 1.5 s window:
          │                  • speaker embedding (ECAPA-TDNN)
          │                  • gender posterior: trained head on embeddings + F0 stats
          │                  • speech gender model (wav2vec2 based) as 2nd opinion
          │                  • F0 (pitch) statistics, voiced ratio
          │                  • event type (speech / non-verbal) from an AudioSet tagger
          ▼  [speakers]    cluster windows by embedding → speakers S1..Sn
          │                speaker-level gender = confidence-weighted vote;
          │                segment posterior shrunk toward its speaker's gender
          ▼  [overlap]     windows where 2 speakers are active (embedding/tagger
          │                evidence); Best preset: 2-speaker speech separation in
          │                overlap regions, keep only the female source
          ▼  [decide]      keep = female; low confidence flagged (|p−0.5| small)
          ▼  [review]      UI timeline: flip segments or whole speakers
          ▼  [render]      gain envelope on vocal stem (raised-cosine 15 ms fades,
          │                sample-exact, same length) + background → filtered.wav
          ▼  [mux]         ffmpeg: -map 0:v -c:v copy, new audio (AAC 256k or
                           original codec family), copy subtitles → output video
```

Key design points:
- **Only the vocal stem is gated.** Music/background is untouched, so removing a male
  voice does not punch holes into the soundtrack. Option: drop the background entirely.
- **Sync:** the gated vocals have exactly the input length; muting never shifts time.
  Video stream is copied bit-for-bit; audio starts at the original stream's start time.
- **Non-speech vocalizations:** speech-only gender classifiers are trained on read
  speech. We explicitly evaluate them on non-verbal clips, and combine them with
  (a) pitch, which carries gender information for voiced moans/grunts, (b) a small
  gender head trained on frozen speaker embeddings using speech **plus** non-verbal
  data with gender labels, and (c) speaker-level consistency across the whole video.
- **Uncertain = flagged,** never silently guessed: the UI highlights segments with
  low confidence and a configurable default (keep or remove) for them.

## 3. Presets (initial choice, confirmed in `docs/MODELS.md`)

| Preset | Separation | Gender | Speakers / overlap |
|---|---|---|---|
| Fast | MDX/Demucs-class vocal model (light) | trained head on ECAPA embeddings + F0 | embedding clustering |
| Balanced | Mel-Band / BS-RoFormer vocal model | head + wav2vec2 gender model fusion | clustering + overlap flagging |
| Best | best RoFormer vocal model (+ ensemble if it helps) | full fusion incl. tagger | clustering + overlap speech separation, keep female source |

## 4. File structure

```
install.bat / install.ps1      installer (safe to re-run)
uninstall.bat / uninstall.ps1  full removal (safe to re-run)
start.bat / start.ps1          start server (created/refreshed by installer)
requirements.txt               python deps except torch
voicefilter/
  __main__.py                  `python -m voicefilter` → server
  config.py                    paths (APP_DIR, DATA_DIR, MODELS_DIR, HF_HOME), settings
  presets.py                   preset → model mapping
  media.py                     ffprobe/ffmpeg helpers (extract, mux, preview encode)
  separation.py                vocal separation backends
  activity.py                  vocal activity / event segmentation
  features.py                  embeddings, pitch, model wrappers (lazy, cached, GPU)
  gender.py                    fusion + trained head + speaker consistency
  overlap.py                   overlap detection + speech separation
  render.py                    gain envelope, crossfades, mixing
  pipeline.py                  orchestration, progress callbacks, analysis.json
  jobs.py                      job store + worker thread
  server.py                    FastAPI app, LAN URL banner
  assets/gender_head.*         small trained classifier weights (committed)
  web/index.html, app.js, style.css
eval/
  build_dataset.py             synthetic mixes with ground-truth timelines
  run_eval.py                  per-preset accuracy → eval/results/*.json + REPORT.md
  train_gender_head.py         trains assets/gender_head from labelled clips
tests/                         pytest: render, media, API (with fake pipeline)
docs/MODELS.md                 research, licenses, VRAM, speed, decisions
```

Runtime folders (git-ignored, all inside the app folder): `.venv/`, `runtime/`
(uv, Python, portable ffmpeg), `models/` (`HF_HOME=models/hf`, `TORCH_HOME=models/torch`),
`data/` (jobs, uploads, temp; `TMP`/`TEMP` point here while the app runs).

## 5. HTTP API (contract between UI and backend)

All JSON. Times in seconds (float).

- `GET /api/info` → `{version, device, gpu_name, lan_urls:[...], presets:[{id,name,description}]}`
- `POST /api/jobs` multipart: `file`, `preset` (`fast|balanced|best`), `keep_background`
  (`true|false`, default true), `auto_export` (default false) → `{id}`
- `GET /api/jobs` → `[job]`; `GET /api/jobs/{id}` → `job`
  - `job = {id, filename, preset, keep_background, state, stage, progress (0..1), message,
    error, duration, created, has_output}`
  - `state ∈ queued | analyzing | review | rendering | done | error`
- `GET /api/jobs/{id}/analysis` →
  `{duration, segments:[seg], speakers:[spk], low_conf_threshold}`
  - `seg = {id, start, end, speaker, kind ("speech"|"nonverbal"|"mixed"), label
    ("female"|"male"|"overlap"), p_female (0..1), confidence (0..1), low_confidence,
    keep (bool, model decision), override (null|"keep"|"remove")}`
  - `spk = {id, p_female, label, total_dur, n_segments, override}`
- `GET /api/jobs/{id}/peaks` → `{rate, peaks:[0..1]}` (vocal stem envelope for drawing)
- `GET /api/jobs/{id}/audio/{kind}` kind ∈ `original|vocals|preview` → m4a, range requests
- `POST /api/jobs/{id}/overrides` body `{segments:{seg_id: "keep"|"remove"|null},
  speakers:{spk_id: "keep"|"remove"|null}}` → updated analysis (preview re-render is async)
- `POST /api/jobs/{id}/export` → starts render + mux; state → rendering → done
- `GET /api/jobs/{id}/download` → output video (`<name>.voicefilter.<ext>`)
- `DELETE /api/jobs/{id}`

Effective decision per segment: segment override › speaker override › model `keep`.

## 6. Install / uninstall (Windows)

- `install.ps1` (wrapped by `install.bat`, runs with `-ExecutionPolicy Bypass`, no admin
  except for the firewall rule, requested via a single elevated sub-process):
  1. Download `uv` into `runtime\uv\`; `UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR` inside the app.
  2. `uv venv --python 3.12 .venv`; install CUDA PyTorch (`cu126` index; falls back to
     `cu124`, then CPU with a warning if no NVIDIA driver), then `requirements.txt`.
  3. ffmpeg: use existing one on PATH, else download a portable build to `runtime\ffmpeg`.
  4. Pre-download the models of the chosen presets into `models\`.
  5. Firewall rule `VoiceFilter (TCP 8765)`, inbound, **Profile Private** only.
  6. Write `start.bat` and a Desktop + Start-menu shortcut.
  7. Write `install-manifest.json` recording what was created/installed by us.
- `uninstall.ps1`: stops a running server, removes `.venv`, `runtime`, `models`, `data`,
  caches, shortcuts, firewall rule, manifest; asks before touching anything that was
  present before install (system Python/ffmpeg are never installed system-wide by us,
  so they are only removed if the manifest says we installed them and the user agrees).
- Both are idempotent.

## 7. Test and evaluation plan

- **Unit tests:** crossfade has no discontinuities (max sample step bounded), render keeps
  exact length, mux keeps video stream identical (packet hash) and duration within one
  audio frame, API flow with a fake pipeline.
- **End-to-end:** generated test video (ffmpeg testsrc + synthetic mix) through all three
  presets on CPU here.
- **Accuracy evaluation:** synthetic mixes built from labelled corpora: male/female speech
  (speaker-disjoint from any training data), male/female non-verbal vocalizations, music/noise
  beds, plus overlap cases. Ground-truth timeline per mix. Metrics per preset:
  - frame-level accuracy of the keep/remove decision on vocal frames (10 ms frames),
  - male-leak rate (male vocal time kept) and female-loss rate (female time removed),
  - segment-level accuracy split by type (speech / non-verbal / overlap),
  - share of time flagged low-confidence and accuracy on the rest,
  - speed (real-time factor) on the CPU here; GPU numbers on the 3080 are estimates.
- Report goes to `eval/REPORT.md`; README states the honest numbers.

## 8. Work split (subagents)

1. Model research → `docs/MODELS.md`, verified downloads, quick CPU sanity runs.
2. Web UI → `voicefilter/web/` against the API contract above.
3. Install/uninstall/start scripts → `*.ps1`, `*.bat`.
4. Evaluation data + gender head training → `eval/`.
5. Backend pipeline + server → `voicefilter/` (after 1).
6. Integration, end-to-end runs, accuracy tuning, README, final report (me).

## 9. Known limits up front

- The build machine is Linux without a GPU: Windows scripts and CUDA speed can only be
  checked statically and must be confirmed on the target PC.
- Accuracy is measured on synthetic mixes; real videos with heavy reverb, whispering,
  pitched-up male voices or very low female voices will be harder.
