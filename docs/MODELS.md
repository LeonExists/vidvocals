# Voice Filter: model research, licenses, verified loading code

Research agent output, 2026-11-26. Target PC: Windows 11, RTX 3080 10 GB, CUDA.
Build box: Linux, no GPU, 4 shared CPU cores. Every number below says whether it was
**measured here**, **read** (model card / leaderboard / paper) or **estimated** (with the reasoning).

> **Owner constraint (applies to every choice below):** everything must be free to build and run:
> free-to-download weights that run locally, no paid APIs, no paid licenses, no cloud inference.
> Models that are gated behind a free Hugging Face account/token (pyannote) are **optional upgrades
> only, never required defaults**. Several free weights are **non-commercial** (CC-BY-NC*). That is fine
> for this personal-use app, and the licenses table marks each one.

> **Network note for this build box:** Hugging Face *file* downloads (LFS/xet) go to
> `us.aws.cdn.hf.co`, which is blocked here and which the owner chose not to allow. HF API/model-card
> calls work. Reachable: github.com (public repos, release assets, raw/media), dl.fbaipublicfiles.com,
> download.pytorch.org, www.openslr.org, PyPI. So every stage has a default or fallback whose weights
> come from a reachable host, and **verified here** means downloaded from such a host and run on CPU.
> Where the GitHub copy is a mirror of an HF original, I compared sha256 against the HF API's LFS
> hash. They are byte-identical (table in section 2.1), so the app can use HF first and fall back to GitHub.

---

## 0. Summary: defaults per preset

| Stage | Fast | Balanced | Best | Verified here? |
|---|---|---|---|---|
| **Vocal separation** (44.1 kHz stereo) | BS-RoFormer **Resurrection** (unwa), 51 M params, 204 MB | **Mel-Band RoFormer Kim** (KimberleyJSN), 228 M params, 913 MB | **Ensemble: Kim Mel-Band + BS-RoFormer viperx ep_317** (waveform average) | yes, all three: downloaded from GitHub releases, sha256 equals the HF original, run on CPU (section 6.1) |
| **Activity / VAD** | energy gate on the vocal stem (+ silero-vad v6 as a speech label only) | same | same | yes, silero (pip-bundled weights) |
| **Speaker embedding / clustering** | **ECAPA-TDNN (TaoRuijie, VoxCeleb2)** 192-d, own agglomerative clustering | same | same + optional WeSpeaker ResNet293-LM (HF) for clustering | ECAPA: yes (raw GitHub). ResNet293: not verifiable here (HF CDN) |
| **Gender** | fused heads: **WavLM-Base+** (torchaudio) features + ECAPA embedding + F0 stats (torchfcpe), speaker-level vote | same | + tagger male/female-speech logits, longer context windows | yes: WavLM-Base+, ECAPA, torchfcpe all run here. Heads are trained by the eval agent |
| **Event tagger** (speech vs non-verbal; scream/laugh/moan...) | **EfficientAT dymn10_as** (10.6 M, mAP 47.7) | dymn10_as | **dymn20_as** (40 M, mAP 49.1) | dymn10_as: yes (GitHub release). dymn20_as: same source, not run |
| **Overlap detection** | none (flag only from embedding/tagger evidence) | heuristic flagging | heuristic + optional pyannote segmentation-3.0 (gated, user token) | heuristic: n/a. pyannote: not verifiable here |
| **Overlap separation** (2 speakers) | none | none | **SepFormer-WHAMR 16 kHz** (SpeechBrain), fallback **torchaudio ConvTasNet Libri2Mix (8 kHz)** | SepFormer: not verifiable here (HF CDN). ConvTasNet: yes |
| Optional upgrades (user's PC, free) | – | audeering wav2vec2 age-gender (CC-BY-NC-SA) as an extra gender opinion | pyannote community-1 diarization with a free HF token; unwa BS-RoFormer Leap Xe or becruily deux as the separator | not verifiable here |

**Estimated total time for a 10-minute video on the RTX 3080:** Fast **~1-1.5 min**, Balanced **~1.5-2 min**,
Best **~3.5-5 min** (section 7). Peak VRAM per stage stays **under ~4 GB**, and stages are loaded one at a time and freed.

Exact identifiers, file names, URLs and sha256 are in `docs/models.json`.

---

## 1. Key findings (read this first)

1. **Speech-trained detectors miss non-verbal sounds.** Measured on EARS clips (section 5): silero-vad
   finds **0 %** of screams and only 10-18 % of coughs, sneezes, throat-clears and yawns, but 73-79 % of
   speech frames. **Vocal activity must come from energy on the separated vocal stem**, with silero used only to
   label speech vs non-speech.
2. **Pitch alone does not work for male non-verbal sounds.** F0 median > 165 Hz gives 95 % on speech
   but only **11 % on male screams/laughs/yells/cheers** (males go high) and 62 % on male emotional
   vocalizations. F0 is still a useful *feature* for the head. It must not be the decision rule.
3. **Speech-only heads drop 15-40 points on non-verbal sounds** (section 5). Of the cues we can run here, **WavLM-Base+ features**
   held up best: 98 % on speech and 76-83 % on non-verbal sounds with a head trained only on LibriSpeech. ECAPA got 81 % / 60-64 % and F0 95 % / 54-61 %.
   So WavLM-Base+ goes into **every** preset (it costs only a few GPU seconds), and the heads must be trained on speech **plus**
   non-verbal data from **many** speakers (a 7-speaker head was at chance). The EARS dataset (github release, CC-BY-NC-4.0, 107
   speakers with gender metadata, screams/laughs/crying/coughs/moan-like emotional freeform) is **reachable here**
   and is a very good training/eval source for the eval agent.
4. **Separation quality matters more than anything else downstream**: residual male voice that the separator
   leaves in the background stem cannot be removed by gating. On our video-like test mix, RoFormers beat
   HTDemucs by a wide margin (section 2.3).
5. **All three separation defaults are free and reachable from GitHub releases** (TRvlvr/model_repo and
   nomadkaraoke/python-audio-separator `model-configs`), byte-identical to the HF uploads. We run them with
   **vendored ZFTurbo MSST model code (MIT)**, not audio-separator, to keep dependencies small (section 2.4).

---

## 2. Vocal / source separation

### 2.1 Candidates

SDR (vocals) from public leaderboards (**read**). "Multisong" = MVSEP Multisong (100 songs, the community
standard). "MUSDB med." = audio-separator's bundled `models-scores.json` (median vocals SDR on MUSDB18 tracks;
note some models saw MUSDB in training, so use it only for relative order).

| Model | Arch / size | Multisong voc. SDR | MUSDB med. | License (weights) | Sources (sha256 verified = same file) | Verified here |
|---|---|---|---|---|---|---|
| **Kim Mel-Band RoFormer** (`vocals_mel_band_roformer.ckpt`) | MelBand, dim 384, depth 6, 60 bands, 228 M params, 913 MB | 11.01 | 12.60 (top of that list) | **MIT** (HF card) | HF `KimberleyJSN/melbandroformer/MelBandRoformer.ckpt`; GitHub `nomadkaraoke/python-audio-separator` release `model-configs` (sha256 87201f4d… identical) | yes |
| **BS-RoFormer viperx ep_317** (`model_bs_roformer_ep_317_sdr_12.9755.ckpt`) | BS, dim 512, depth 12, 160 M params, 639 MB | 10.87 (the "12.97" in the name is viperx's private validation set) | 11.77 | not stated (UVR community release; free) | GitHub `TRvlvr/model_repo` release `all_public_uvr_models`; HF `Politrees/UVR_resources` (sha256 5b84f37e… identical) | yes |
| **BS-RoFormer Resurrection (unwa)** vocals | BS, dim 256, depth 12, 51 M params, 204 MB, chunk 17.8 s | not found in search results | – | not stated (free on HF) | HF `pcunwa/BS-Roformer-Resurrection/BS-Roformer-Resurrection.ckpt`; GitHub nomadkaraoke `bs_roformer_vocals_resurrection_unwa.ckpt` (sha256 9dbfe5cb… identical) | yes |
| HTDemucs-ft, vocals model `f7e0c4bc` | Hybrid transformer demucs, 42 M params, 84 MB | ~9-10 (htdemucs_ft bag) | 10.79 (4-model bag) | MIT | dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th; HF `adefossez/HTDemucs-ft` (safetensors re-export) | yes |
| UVR-MDX-NET-Voc_FT / Kim_Vocal_2 | MDX-Net ONNX, 67 MB | ~9.5 | 10.15 / 10.18 | MIT-style UVR community | TRvlvr release; HF `Politrees/UVR_resources` (sha256 identical) | downloaded only (needs onnxruntime + MDX STFT code; not run) |
| unwa BS-RoFormer **Leap Xe** voc | BS, 268 MB | **11.76** | – | not stated | HF `pcunwa/BS-Roformer-Leap/Xe/bs_leap_xe_voc.ckpt` + `Xe/leap_xe_config_voc.yaml` | not verifiable here (HF CDN); not on GitHub mirrors |
| unwa BS-RoFormer **HyperACE v2** voc | BS + custom SegmModel (`bs_roformer.py` shipped in repo), 289 MB | 11.40 (card) | – | not stated | HF `pcunwa/BS-Roformer-HyperACE/v2_voc/` | not verifiable here; needs the repo's own model code |
| becruily **deux** (dual voc/inst) | MelBand, 435 MB | 11.37 (card) | – | **CC-BY-NC-4.0** | HF `becruily/mel-band-roformer-deux` | not verifiable here |
| unwa Kim-FT / FT2 / FT3, gabox FV1-FV7, becruily vocals, Revive 1-3e | MelBand / BS fine-tunes, 639-913 MB | ~10.9-11.2 | ~12.4 (Kim FT) | not stated | GitHub nomadkaraoke mirrors + HF | not run (same code path as Kim / viperx) |
| MVSEP BS-RoFormer 124-band (2026.07/10), 12.3 multisong | – | 12.33 | – | **not downloadable** (MVSEP site only) | – | n/a, excluded (cloud only) |
| HiDolen Mini-BS-RoFormer V2.5 (4-stem, 46 M) | transformers remote code | MUSDB vocal 10.97 | – | CC-BY-NC-4.0 | HF only | not verifiable here |
| SCNet / MDX23C | – | ≤ 10.5 | ≤ 10.6 | MIT | HF/GitHub | not pursued: below the RoFormers |

Cinematic (dialogue/music/effects) separators such as Bandit / Bandit v2 / BS-RoFormer-DnR exist (weights on
Zenodo or MVSEP-only mirrors). They target the *speech* stem and are not trained to put screams or moans in it, so
they are not recommended as defaults. This is worth an experiment later.

### 2.2 Which one per preset, and why

- **Fast: Resurrection BS-RoFormer (unwa).** It is the smallest RoFormer here (204 MB, 51 M params) and scored the
  **best SDR on our video-like mix** (below). Its 17.8 s chunks make it the most FLOP-heavy per second of audio, but on a
  3080 that is still about 1 minute per 10 minutes of audio at overlap 2 (section 7). Even faster:
  run it with `num_overlap=1` (half the compute) or swap in HTDemucs-ft vocals (~10 s per 10 min, clearly worse).
- **Balanced: Kim Mel-Band RoFormer.** It is the best-documented free vocal model (MIT, 11.01 Multisong), the base of most
  community fine-tunes, and the cheapest RoFormer per second of audio (388 GFLOP/s). It is a "full" vocal model, so it keeps
  breaths and other non-verbal sounds well (community consensus, **read**).
- **Best: ensemble of Kim Mel-Band + viperx BS ep_317.** Averaging the waveforms of two different architectures is the
  standard MVSEP trick (+0.2-0.4 dB **read**). Both files are free and verified here.
  **Optional upgrade on the user's PC:** unwa Leap Xe voc (11.76 Multisong, HF only) as a third ensemble member
  or as a replacement for viperx. It loads with the same `BSRoformer` class and the yaml from its repo.

### 2.3 Measured here (CPU, 2 threads, heavily shared box: CPU times are inflated, use them only relative to each other)

Test mix: 30 s, 44.1 kHz stereo. Vocals = EARS clips (female speech, male laugh, female pain vocalization,
male speech, female scream, male pain vocalization) over synthetic music (square-wave chords, bass, kick, hats) +
noise. `SDR` = 10·log10(‖ref‖²/‖ref−est‖²) of the vocal stem.

RESULTS_SEP_TABLE

GFLOP/audio-s = FLOPs of one forward pass over one chunk (torch `FlopCounterMode`, including SDPA attention) divided by
chunk length. Multiply by `num_overlap` for the real cost.

### 2.4 Inference wrapper: vendor MSST code (recommended) vs audio-separator

- **Recommended: vendor three files** from `ZFTurbo/Music-Source-Separation-Training` (MIT, commit `1523206`, 2026-11-21):
  `models/bs_roformer/bs_roformer.py`, `models/bs_roformer/mel_band_roformer.py`, `models/bs_roformer/attend.py`.
  Change the `from models.bs_roformer.attend import Attend` lines to a relative import. Runtime deps: `einops`,
  `rotary-embedding-torch`, `beartype`, `librosa` (mel filters), `pyyaml`. All are on PyPI and all were installed and tested here.
  `torch.load(ckpt, weights_only=True)` + `load_state_dict(strict=True)` worked for all three checkpoints.
  The chunked overlap-add loop is ~25 lines (section 6.1).
  - Note: `attend.py` uses `torch.backends.cuda.sdp_kernel` (deprecated in torch 2.15, it still works but warns).
    If a future torch removes it, replace it with `torch.nn.attention.sdpa_kernel`, or construct with `flash_attn=False`
    (the plain-SDPA path, which still uses fused kernels).
- **audio-separator 0.48.0** (MIT, 2026-10-29) supports all of these models and downloads them from GitHub
  (`nomadkaraoke/python-audio-separator` model-configs release first, TRvlvr second). Those URLs work here and on Windows.
  It is not recommended as the core dependency because it pins `beartype<0.19`, `rotary-embedding-torch<0.7` and
  `samplerate==0.1.0`, and pulls `onnx-weekly`, `onnx2torch`, `diffq`/`diffq-fixed` and `pydub`. That is fragile on a
  Windows uv install, and it also fetches `download_checks.json` etc. from raw.githubusercontent at runtime.
  It is a fine fallback if the vendored code breaks.
- `demucs` 4.1.0 (PyPI, 2026-07) now also loads from HF (`adefossez/HTDemucs-ft`). `demucs.states.load_model(path)` on
  the `.th` from dl.fbaipublicfiles works and was tested here.
- `BS-RoFormer` 1.4.1 (lucidrains) and `bs-roformer-infer` / `melband-roformer-infer` (openmirlab) exist on PyPI. They
  were not tested: the lucidrains package has drifted from the community checkpoint layout (extra hyper-connection and
  PoPE modules), so key mismatches are likely.

---

## 3. Speaker embeddings / diarization

| Model | EER Vox1-O (read) | Size | License | Gated | Source | Verified |
|---|---|---|---|---|---|---|
| **ECAPA-TDNN, TaoRuijie** (`exps/pretrain.model`, C=1024, 192-d) | 0.96 % (repo README) | 66 MB, ~14.7 M params | **MIT** | no | github.com/TaoRuijie/ECAPA-TDNN (file in git, raw.githubusercontent) | **yes** |
| SpeechBrain ECAPA `speechbrain/spkrec-ecapa-voxceleb` | 0.80 % | 83 MB | Apache-2.0 | no | HF only | not verifiable here |
| WeSpeaker ResNet34-LM (`Wespeaker/wespeaker-voxceleb-resnet34-LM`, also `pyannote/wespeaker-voxceleb-resnet34-LM`, ungated) | 0.72 % | 27 MB onnx / 45 MB pt | CC-BY-4.0 | no | HF only | not verifiable here |
| WeSpeaker ResNet293-LM | 0.447 % | 114 MB onnx | CC-BY-4.0 | no | HF only | not verifiable here |
| WeSpeaker CAM++ | 0.654 % | 29 MB onnx | Apache-2.0 | no | HF only | not verifiable here |
| NVIDIA TitaNet-L | ~0.66 % | 102 MB | CC-BY-4.0 | no | HF (`nvidia/speakerverification_en_titanet_large`), needs NeMo | no (NeMo too heavy) |
| Resemblyzer GE2E | weak (~4 %) | 17 MB | Apache-2.0 | no | **PyPI-bundled** | usable last-resort fallback |

**Choice: TaoRuijie ECAPA for all presets** (both the gender head input and clustering). It is verified here, it is MIT,
the eval agent can train the gender head on exactly these embeddings, and its quality is close to SpeechBrain ECAPA
(0.96 vs 0.80 % EER). The `ECAPA_TDNN` class is 150 lines of plain torch + torchaudio `MelSpectrogram`, so vendor `model.py`
(MIT). If the head is later retrained on another embedding, SpeechBrain ECAPA or WeSpeaker ResNet293 are drop-in
upgrades for the user's PC. **Do not mix embedding models between training and runtime.**

**Diarization pipelines**
- `pyannote/speaker-diarization-community-1` (pyannote.audio 4.x): CC-BY-4.0, **gated (free HF account + accept terms + token)**.
  This is the best free diarization pipeline, so it is an **optional upgrade** in Best/Balanced when the user pastes an HF token. Same for
  `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` (MIT, gated). pyannote.audio 4 decodes files via
  torchcodec. Pass an in-memory `{"waveform": tensor, "sample_rate": 16000}` dict to avoid the FFmpeg-DLL dependency on Windows.
- NVIDIA Sortformer: `diar_sortformer_4spk-v1` is **CC-BY-NC-4.0**, and the streaming v2 is CC-BY-4.0 (v2.1 is under the "NVIDIA Open Model License").
  All of them need NeMo (heavy, painful on Windows) and are limited to 4 speakers. **Not recommended.**
- **Default:** our own clustering: ECAPA on 1.5 s windows (hop 0.75 s) over vocal-active regions, cosine
  agglomerative clustering (threshold ~0.5-0.6 on L2-normalised embeddings, tune on eval), plus re-segmentation at speaker changes.

---

## 4. Gender classification

### 4.1 Off-the-shelf classifiers (all speech-trained)

| Model | Trained on | Size | License | Source | Verified here |
|---|---|---|---|---|---|
| JaesungHuh/voice-gender-classifier | ECAPA (TaoRuijie arch) + linear, VoxCeleb2 (speech), 98.7 % Vox1 | 62 MB | MIT | HF weights + GitHub `model.py` | **no** (weights HF/Google-Drive only) |
| audeering/wav2vec2-large-robust-24-ft-age-gender (and -6- variant, 727 MB) | aGender, Mozilla CV, Timit, Voxceleb2 (speech); outputs female/male/**child** | 2.5 GB / 0.73 GB | **CC-BY-NC-SA-4.0** (personal use OK, document it) | HF | no |
| alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech | LibriSpeech (read speech) | 1.26 GB | Apache-2.0 | HF | no |
| prithivMLmods/Common-Voice-Gender-Detection | wav2vec2-base, Common Voice | 378 MB | Apache-2.0 | HF | no |
| inaSpeechSegmenter (keras CNN male/female) | French broadcast speech | small | MIT | GitHub release (reachable) | no: needs TensorFlow (no native-Windows GPU) |
| EfficientAT tagger "Male speech"/"Female speech" logits | AudioSet (weak labels) | – | MIT | GitHub release | yes (as a gender cue, section 5) |
| alkiskoudounas/voc2vec (wav2vec2 pre-trained on non-verbal vocalizations) | 125 h NVV | 380 MB | Apache-2.0 | HF | no. **Promising backbone for a non-verbal gender head** |

None of these was trained on non-verbal vocalizations with gender labels. That is why the plan's own head matters.

### 4.2 Recommended gender stack (all free, verifiable here)

- **Features (all presets):** **WavLM-Base+** (torchaudio pipeline from download.pytorch.org, MIT) mean-pooled hidden states of
  layers 3+6 (1536-d; the strongest cue measured here, and the eval agent may tune which layers), ECAPA (TaoRuijie) 192-d embedding,
  and F0 stats from **torchfcpe** (MIT, weights bundled in the pip wheel, CPU RTF 0.08 here) such as log-median F0, IQR and voiced ratio.
- **Heads:** logistic regression / small MLP trained by the eval agent on speech **and** non-verbal clips with
  gender labels (EARS, plus whatever else the eval agent can reach). Fuse by averaging logits, with weights tuned on held-out speakers.
- **Speaker-level vote** (all presets): confidence-weighted mean of segment posteriors per cluster, then shrink each
  segment toward its speaker's posterior. This fixes most single-segment errors on screams and laughs.
- Optional extra opinion on the user's PC: audeering 6-layer age-gender (CC-BY-NC-SA). It is also useful to flag
  **child** voices.

F0 extractors: torchfcpe (fast, bundled weights, verified), torchcrepe 0.0.24 (MIT, bundled `full`/`tiny` weights,
installed), librosa.pyin (slow), pyworld (MIT), parselmouth (**GPL-3**, avoid bundling), RMVPE (HF/`lj1995` only), PENN (MIT).

---

## 5. Measured sanity checks (gender, VAD, tagger) on EARS non-verbal clips

Data: 8 EARS speakers (p001, p005, p006, p008 male; p002, p003, p004, p011 female), downloaded per-file from
the GitHub release zips via HTTP range requests (`scratchpad/models/httpzip.py`). Up to 3 × 3 s energy-picked chunks per file,
resampled to 16 kHz, 334 clips in total. Speech-only training data for the heads: LibriSpeech dev-clean (openslr; 40 speakers,
20 F / 20 M, 6 utterances each, cropped to 4 s), speaker-disjoint from EARS.

Accuracy per clip type (overall [male-correct, female-correct]); the heads were trained on **LibriSpeech speech only** (logistic regression):

| Classifier (all verified here) | speech (n=43) | whisper (24) | emotional freeform: pain/ecstasy/desire/relief (94) | non-verbal: laugh/scream/cry/yell/cheer (109) | vegetative: cough/throat/sneeze/yawn (64) | all (334) | CPU RTF |
|---|---|---|---|---|---|---|---|
| F0 median > 165 Hz (torchfcpe) | 95 % [100, 91] | 62 % [92, 33] | 76 % [62, 89] | **54 % [11, 98]** | 61 % [42, 79] | 67 % | 0.08 |
| ECAPA (TaoRuijie) + LogReg | 81 % [86, 77] | 83 % [67, 100] | 72 % [64, 81] | 60 % [47, 72] | 64 % [45, 82] | 69 % | 1.05* |
| **WavLM-Base+ L3+L6 mean + LogReg** | **98 %** [95, 100] | 54 % [8, 100] | **82 %** [64, 100] | **76 %** [76, 76] | **83 %** [94, 73] | **80 %** | 0.51* |
| EfficientAT dymn10_as, P(female speech) > P(male speech) | 93 % | 62 % | 85 % | 68 % | 61 % | – | 0.15* |

\* CPU RTF with 2 threads on a box loaded at 12-18 by other agents; it would be several times faster unloaded.

What this says (sanity level only, 8 test speakers, 3 s clips; the eval agent does the real evaluation):
- Every speech-trained classifier loses 15-40 points on non-verbal sounds. **Male screams, laughs, yells and cheers are the main
  failure** (high-pitched). Female whispers fail for F0 and ECAPA, and male whispers fail for WavLM.
- **WavLM-Base+ features were the most robust** single cue on non-verbal sounds (76-83 %), even with a speech-only head.
  That is why Balanced/Best fuse it in. It is the plan's "wav2vec2 second opinion", but it is verifiable here and MIT.
- Fusion and training heads on non-verbal data are needed. A quick leave-one-speaker-out head on ECAPA+F0 trained on the
  other 7 EARS speakers was at **chance (51 %)**: with that few speakers the head learns identities, not gender.
  **Train heads on many speakers** (EARS has 107 speakers with gender metadata, all reachable here; add LibriSpeech/others).
- Speaker-level consistency (cluster vote) is essential for screams and laughs inside a conversation.

**silero-vad v6.3** (pip-bundled weights, 16 kHz, threshold 0.5, CPU RTF 0.008 with 1 thread): share of each clip flagged as speech

| EARS clip type | silero speech coverage | energy-active share | clips with **no** silero detection |
|---|---|---|---|
| speech (sentences regular / neutral) | 73-79 % | 98 % | 0 % |
| whisper | 55 % | 100 % | 8 % |
| emotional freeform (pain, ecstasy, desire, relief) | 63-87 % | 94-99 % | 0 % |
| laughter (open / closed) | 59 / 50 % | 84-96 % | 11-14 % |
| crying, yelling, cheering | 36-43 % | 81-92 % | 18-29 % |
| **screaming** | **0 %** | 89 % | **100 %** |
| coughing, sneezing, throat clearing, yawning | 10-18 % | 76-92 % | 35-56 % |

**EfficientAT dymn10_as** on the same 3 s clips (top-1 class counts): sneezes are found as `Sneeze` (10/13), coughs as
`Throat clearing`/`Cough`, whispers as `Whispering` (19/24), crying as `Gasp`/`Wail, moan`. Screams are weak (`Screaming` top-1 in only 3/15;
the max vocal-class probability is only ~0.1-0.45 on 3 s clips, because AudioSet models are trained on 10 s clips). **Use the tagger for the
speech vs non-verbal `kind` label on ≥ 3 s windows with relative scores, not for detection.** As a gender cue, female−male
speech logit: speech 93 %, emotional freeform 85 %, nonverbal 68 %, vegetative 61 %.
- `mn10_as` (MobileNetV3) gave garbage (`Sawing`/`Sanding` at p≈1.0 on every vocal clip) in this torch 2.15 env, while
  `dymn10_as` was fine. The cause was not found (possibly my `torchvision.ops.misc.ConvNormActivation` stand-in; torchvision is not installed
  here). Use the **dymn** models.

**Overlap separation fallback, torchaudio `CONVTASNET_BASE_LIBRI2MIX`** (8 kHz): EARS female + male speech mixed at 0 dB:
SI-SDR 0.0 → **12.2 dB** (+12.2 dB), load 1.1 s.

---

## 6. Verified loading code (copy-paste for the backend)

All snippets were run here (CPU) with: Python 3.12, torch 2.15, torchaudio 2.11, `einops 0.8.2`, `rotary-embedding-torch 0.9.2`,
`beartype 0.23.1`, `silero-vad 6.3`, `torchfcpe 0.0.4`, `demucs 4.1.0`, `soxr`, `soundfile`, `pyyaml`, `librosa 1.0.1`.
Set `HF_HOME=models/hf` and `TORCH_HOME=models/torch`. Use `device="cuda"` + `torch.autocast("cuda", torch.float16)` on the 3080.

### 6.1 RoFormer separation (Resurrection / Kim / viperx), vendored MSST code

```python
# files: voicefilter/third_party/msst/{bs_roformer.py, mel_band_roformer.py, attend.py}
#   copied from github.com/ZFTurbo/Music-Source-Separation-Training @1523206 (MIT), imports made relative.
import torch, yaml, numpy as np
from voicefilter.third_party.msst.bs_roformer import BSRoformer
from voicefilter.third_party.msst.mel_band_roformer import MelBandRoformer

def load_roformer(kind, ckpt_path, cfg_path, device):
    cfg = yaml.load(open(cfg_path), Loader=yaml.FullLoader)        # FullLoader needed: configs contain !!python/tuple
    cls = BSRoformer if kind == "bs_roformer" else MelBandRoformer
    model = cls(**cfg["model"])
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=True)                        # strict load OK for all 3 checkpoints
    return model.eval().to(device), cfg

def _window(C, fade):
    w = torch.ones(C); w[:fade] = torch.linspace(0, 1, fade); w[-fade:] = torch.linspace(1, 0, fade); return w

@torch.inference_mode()
def separate_vocals(model, cfg, mix, device, num_overlap=2, batch=1):
    """mix: float32 np.array (T, 2) at 44100 Hz. Returns vocals (T, 2). Background = mix - vocals."""
    C = cfg["audio"]["chunk_size"]; step = C // num_overlap; pad = C - step
    x = torch.from_numpy(mix.T.copy())
    T = x.shape[-1]
    xp = torch.nn.functional.pad(x[None], (pad, pad + C), mode="reflect" if T > pad + C else "constant")[0]
    w = _window(C, C // 10).to(device)
    out = torch.zeros(2, xp.shape[-1], device=device); cnt = torch.zeros(xp.shape[-1], device=device)
    for s in range(0, T + pad, step):
        chunk = xp[:, s:s + C]
        if chunk.shape[-1] < C:
            chunk = torch.nn.functional.pad(chunk, (0, C - chunk.shape[-1]))
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
            y = model(chunk[None].to(device))[0]          # (2, C) for num_stems=1  (or (stems, 2, C))
        if y.dim() == 3: y = y[0]                         # target stem = vocals for all 3 configs
        out[:, s:s + C] += y.float() * w; cnt[s:s + C] += w
    voc = (out / cnt.clamp_min(1e-8))[:, pad:pad + T]
    return voc.cpu().numpy().T
# Best preset: vocals = 0.5 * (sep(kim) + sep(viperx)); load/free one model at a time (del model; torch.cuda.empty_cache()).
```
Notes: all three configs have `target_instrument: vocals/Vocals`, so `num_stems=1` → output = vocals. Use `num_overlap=2`
(Kim's yaml says 1). On GPU a batch of 2-4 chunks is safe on 10 GB. Normalising the input is not needed (min_mean_abs only skips silence).

### 6.2 HTDemucs-ft vocals (fallback / ultra-fast)
```python
from demucs.states import load_model; from demucs.apply import apply_model
m = load_model("models/separation/f7e0c4bc-ba3fe64a.th").eval()      # from dl.fbaipublicfiles.com
x = torch.from_numpy(mix.T.copy()); ref = x.mean(0); xn = (x - ref.mean()) / ref.std()
y = apply_model(m, xn[None], shifts=0, split=True, overlap=0.25, device=device)[0] * ref.std() + ref.mean()
vocals = y[m.sources.index("vocals")]
```

### 6.3 ECAPA-TDNN speaker embedding (TaoRuijie, MIT)
```python
# vendor model.py from github.com/TaoRuijie/ECAPA-TDNN @4904fda (MIT); weights: exps/pretrain.model (sha256 a435faa1...)
from voicefilter.third_party.ecapa_tdnn import ECAPA_TDNN
ec = ECAPA_TDNN(C=1024)
sd = torch.load("models/ecapa/pretrain.model", map_location="cpu", weights_only=True)
ec.load_state_dict({k[len("speaker_encoder."):]: v for k, v in sd.items() if k.startswith("speaker_encoder.")})
ec.eval().to(device)
with torch.inference_mode():
    emb = ec(wav16k_batch.to(device), aug=False)            # (B, 192); input float32 16 kHz mono, (B, T)
    emb = torch.nn.functional.normalize(emb, dim=1)
```
Weights URL: `https://raw.githubusercontent.com/TaoRuijie/ECAPA-TDNN/main/exps/pretrain.model` (66.7 MB).

### 6.4 F0 (torchfcpe, weights inside the wheel)
```python
from torchfcpe import spawn_bundled_infer_model
fcpe = spawn_bundled_infer_model(device=device)
f0 = fcpe.infer(torch.from_numpy(wav16k)[None, :, None].to(device), sr=16000,
                decoder_mode="local_argmax", threshold=0.006)[0, :, 0].cpu().numpy()   # Hz per 10 ms, 0 = unvoiced
```

### 6.5 WavLM-Base+ features (torchaudio pipeline, download.pytorch.org)
```python
import torchaudio
wl = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model().eval().to(device)   # -> TORCH_HOME/hub/checkpoints/wavlm_base_plus.pth (378 MB)
with torch.inference_mode():
    feats, _ = wl.extract_features(wav16k_batch.to(device))               # list of 12 layers, each (B, frames, 768)
    vec = torch.cat([feats[3].mean(1), feats[6].mean(1)], dim=1)          # (B, 1536)
```

### 6.6 silero-vad (pip wheel bundles the model)
```python
from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad()                                    # JIT, CPU is fine (RTF ~0.008)
ts = get_speech_timestamps(torch.from_numpy(wav16k), vad, sampling_rate=16000, threshold=0.5)  # [{'start','end'} in samples]
```

### 6.7 EfficientAT tagger (dymn10_as / dymn20_as)
```python
# vendor github.com/fschmid56/EfficientAT @a425fdc (MIT): models/, helpers/utils.py, metadata/class_labels_indices.csv
#  - helpers/utils.py opens 'metadata/class_labels_indices.csv' relative to CWD: patch it to a path relative to __file__
#  - models/dymn/model.py and models/mn/* import torchvision.ops.misc.ConvNormActivation: install torchvision from the
#    same CUDA index as torch, or vendor that 20-line class (torchvision, BSD-3)
#  - models/preprocess.py: torch.stft(..., return_complex=False) is deprecated; use return_complex=True and x.abs()**2
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.utils import labels                           # 527 AudioSet names
tagger = get_dymn(width_mult=1.0, pretrained_name="dymn10_as").eval().to(device)   # downloads from GitHub release v0.0.1
mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320).eval().to(device)
with torch.inference_mode():
    logits, _ = tagger(mel(wav32k_batch.to(device)).unsqueeze(1))   # wav at 32 kHz mono, (B, T); 10 s windows are best
    probs = torch.sigmoid(logits)                               # (B, 527)
```
Weights go to the module-level `model_dir = "resources"` (relative to CWD!) in `models/dymn/model.py`; patch it to `models/torch/efficientat` or pre-download:
`https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn10_as.pt` (42.5 MB, sha256 f363baaf…), `.../v0.0.1/dymn20_as_mAP_493.pt` (Best).

### 6.8 Overlap: 2-speaker separation
```python
# fallback, verified (8 kHz, clean Libri2Mix):
import torchaudio
B = torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX
sep = B.get_model().eval().to(device)                      # B.sample_rate == 8000
srcs = sep(wav8k[None, None].to(device))[0]                # (2, T)
# Best preset on the user's PC (NOT verified here, HF CDN blocked): SpeechBrain SepFormer WHAMR! 16 kHz (Apache-2.0)
from speechbrain.inference.separation import SepformerSeparation
sepf = SepformerSeparation.from_hparams("speechbrain/sepformer-whamr16k", savedir="models/hf/sepformer-whamr16k",
                                        run_opts={"device": device})
est = sepf.separate_batch(wav16k[None])                    # (1, T, 2)
```
Separated sources are band-limited (8/16 kHz). Use them to pick **which** speaker is female (ECAPA + gender head on each
source). Then either (a) replace the overlap region of the vocal stem with the female source, upsampled and lightly
crossfaded, or (b) build a soft T-F mask `|S_f|/(|S_f|+|S_m|)` and apply it to the full-band vocal stem below 8 kHz, keeping the band above it
gated as a whole. (b) sounds better; let the eval agent decide.

---

## 7. VRAM and time budget on the RTX 3080 (estimates)

Method: FLOPs per audio second measured here × `num_overlap`, divided by an effective 3080 throughput of **12-18 TFLOP/s**
for fp16-autocast transformer inference (29.8 TFLOPS fp16-with-fp32-accumulate tensor-core peak on GeForce Ampere, 40-60 %
utilisation, plus STFT/iSTFT and Python overhead). Small models are dominated by overhead, so a floor of a few seconds applies.

TIME_TABLE

VRAM: RoFormers with batch 1-2 and an 8-18 s chunk in fp16 need ~1-2 GB of weights plus activations, **~2-4 GB peak (estimate)**.
WavLM-Base+, ECAPA and the dymn models each need < 1.5 GB, SepFormer < 2 GB. With load → run → `del` + `torch.cuda.empty_cache()` per stage, the
peak stays **< 4-5 GB**, well inside 10 GB. A 1-2 GB margin for the desktop is left even with batch 4 on the RoFormers.

---

## 8. Licenses

| Component | License | Commercial? | Notes |
|---|---|---|---|
| Kim Mel-Band RoFormer weights | MIT | yes | HF card |
| viperx BS-RoFormer ep_317 / ep_368 weights | not stated (UVR public release) | unclear | free to download; personal use fine |
| unwa Resurrection / Leap / HyperACE / Kim-FT, gabox FV*, becruily vocals | not stated | unclear | free; personal use fine |
| becruily deux, HiDolen Mini-BS | CC-BY-NC-4.0 | no | optional only |
| HTDemucs / demucs code | MIT | yes | |
| MSST model code (ZFTurbo) | MIT | yes | vendored |
| audio-separator | MIT | yes | not used by default |
| TaoRuijie ECAPA-TDNN code + weights | MIT | yes | trained on VoxCeleb2 (dataset terms: research) |
| SpeechBrain ECAPA / SepFormer | Apache-2.0 | yes | optional |
| WeSpeaker ResNet34/293-LM | CC-BY-4.0 | yes (attribution) | optional |
| pyannote community-1 | CC-BY-4.0, **gated** (free account) | yes | optional, user token |
| pyannote 3.1 / segmentation-3.0 | MIT, **gated** | yes | optional |
| NVIDIA Sortformer v1 | CC-BY-NC-4.0 | no | not used |
| JaesungHuh voice-gender-classifier | MIT | yes | optional |
| audeering age-gender (6/24 layer) | **CC-BY-NC-SA-4.0** | no | optional; document in README if offered |
| alefiury / prithivMLmods gender | Apache-2.0 | yes | optional |
| WavLM-Base+ (torchaudio) | MIT (microsoft/unilm) | yes | |
| torchaudio ConvTasNet Libri2Mix | BSD-2 (torchaudio) / Libri2Mix CC-BY-4.0 data | yes | |
| silero-vad | MIT | yes | |
| torchfcpe / torchcrepe / PENN / pyworld | MIT | yes | |
| praat-parselmouth | **GPL-3.0** | copyleft | avoid bundling |
| EfficientAT (code + AudioSet weights) | MIT | yes | AudioSet labels CC-BY-4.0 |
| MIT AST AudioSet | BSD-3-Clause | yes | optional |
| CED (mispeech) | Apache-2.0 | yes | optional |
| laion vocalburst-locator | Apache-2.0 | yes | optional research |
| EARS dataset (test/training data) | CC-BY-NC-4.0 | no | eval/training only, not shipped |
| LibriSpeech | CC-BY-4.0 | yes | |

---

## 9. Other categories (research notes)

### 9.1 Audio event taggers and AudioSet classes
- **EfficientAT** (MIT, weights on GitHub releases): dymn04/10/20_as (mAP 45.0/47.7/49.1), mn10/mn40_as(_ext) (47.1/48.7).
  Frame-level output is possible with `windowed_inference.py`.
- **MIT/ast-finetuned-audioset-10-10-0.4593** (BSD-3, HF, transformers-native, 87 M params, mAP 45.9): fine on the user's PC, not verifiable here.
- **CED** base/small/tiny (`mispeech/ced-*`, Apache-2.0, HF, transformers remote code, base mAP ~50.0): strongest HF tagger, not verifiable here.
- BEATs (MIT, weights on an Azure blob, not reachable here), PANNs CNN14 (MIT, Zenodo, not reachable here), CLAP (`laion/clap-htsat-unfused`,
  `laion/larger_clap_general`, Apache-2.0, HF) for zero-shot prompts such as "a woman moaning".
- **laion/vocalburst-locator** (Apache-2.0, Whisper-small based, Aug 2026): detects and timestamps vocal bursts (laughs, sighs,
  gasps, screams, cries). Event F1 0.61 on real audio (card). 972 MB, HF only. It is a strong candidate for a *non-verbal detector* in
  Best once verifiable. `laion/vocalburst-classifier-multilabel` (CC-BY-4.0, 57 MB) labels the type.
- AudioSet ontology labels relevant to us (index: name, from the AST/EfficientAT 527-class list):
  0 Speech, **1 Male speech, man speaking**, **2 Female speech, woman speaking**, 3 Child speech, kid speaking, 4 Conversation,
  5 Narration, monologue, 6 Babbling, 8 Shout, 9 Bellow, 10 Whoop, 11 Yell, 13 Children shouting, 14 Screaming, 15 Whispering,
  16 Laughter, 17 Baby laughter, 18 Giggle, 19 Snicker, 20 Belly laugh, 21 Chuckle, chortle, 22 Crying, sobbing, 23 Baby cry, infant cry,
  24 Whimper, **25 Wail, moan**, 26 Sigh, 27 Singing, 32 Male singing, 33 Female singing, 36 Rapping, 37 Humming, **38 Groan**, **39 Grunt**,
  41 Breathing, 43 Snoring, 44 Gasp, 45 Pant, 46 Snort, 47 Cough, 48 Throat clearing, 49 Sneeze, 50 Sniff, 58 Burping, eructation, 59 Hiccup,
  66 Cheering, 68 Chatter, 69 Crowd, 70 Hubbub, speech noise, speech babble. There is no separate "Moan" class: it is merged into "Wail, moan".
  Gendered classes exist only for speech and singing.

### 9.2 Overlapping speech
- Detection: pyannote `segmentation-3.0` (MIT, gated) is the standard overlapped-speech detector, an optional upgrade with a token.
  Default: flag windows where (a) ECAPA similarity to both neighbouring speaker centroids is low and in between, (b) the tagger shows
  both male and female speech logits high, or (c) the separated 2-source energies are both high (Best preset).
- Separation: SpeechBrain `sepformer-whamr16k` (Apache-2.0, trained on noisy + reverberant 16 kHz mixtures, the best fit for
  movie audio and already in our SpeechBrain dependency), `alibabasglab/MossFormer2_SS_16K` (Apache-2.0, 670 MB, stronger on
  clean data, needs ClearerVoice code), pyannote PixIT / `separation-ami-1.0` (MIT, gated), MERL **TF-Locoformer**
  (Apache-2.0, weights in the GitHub repo via git-LFS, reachable, but only 8 kHz WHAMR!/Libri2Mix and 16 kHz DNS enhancement)
  and torchaudio ConvTasNet Libri2Mix (8 kHz, verified).

### 9.3 VAD
- silero-vad v6.3 (MIT, pip wheel contains the model, 16 kHz, very fast) is good for speech and **blind to screams and many non-verbal sounds**.
- pyannote segmentation (gated) has the same speech bias.
- **Energy on the separated vocal stem** (e.g. RMS in 20 ms frames, threshold relative to the stem's 95th percentile, hysteresis
  plus 150 ms hang-over) is the primary activity detector. The separator already removed music and SFX.

---

## 10. Open questions / not verified

- **Not verifiable here (HF CDN blocked):** SpeechBrain ECAPA, WeSpeaker, pyannote, all wav2vec2 gender models, AST, CED, voc2vec,
  laion vocalburst models, SepFormer, MossFormer2, Leap Xe / HyperACE / deux separators. Their file lists, sizes, licenses
  and gating were checked through the HF API; the loading code for them in this file is from the model cards and has not been run.
- GPU speed and VRAM are estimates (section 7). The backend should log per-stage times on the first real run on the 3080.
- `mn10_as` misbehaved here (see section 5). Re-test it once torchvision is installed; dymn models are fine.
- Separation quality on real film/TV audio (dialogue + SFX + music) is not measured. Our test mix uses synthetic music. The eval
  agent should test whether moans, breaths and screams survive separation into the vocal stem, comparing Kim (full) vs Resurrection.
- Licences of community RoFormer fine-tunes (viperx, unwa, gabox) are unstated. They are free downloads; we treat them as personal-use.
- The JaesungHuh and audeering classifiers could not be sanity-checked here. The eval agent's head on ECAPA + F0 (+ WavLM) is the default instead.
