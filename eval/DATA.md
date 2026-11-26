# Evaluation data

Everything here is fetched by `python eval/fetch_data.py` into `eval/data/` (git-ignored, ~0.8 GB on
disk, ~1.2 GB downloaded). The subset choice is deterministic (seeded by file names), so a re-run gives
the same clips. `eval/data/clips.csv` is the manifest (one row per labelled clip) and `eval/splits.json`
(committed) maps every speaker to `train` or `test`.

**Network note.** On the build machine Hugging Face *file* downloads are blocked (models and datasets are
served from `us.aws.cdn.hf.co`, which the proxy refuses; adding the host was declined), so none of the
Hugging Face candidates (google/fleurs, VCTK, LibriTTS-R parquet, VocalSound, NonverbalTTS, VIVAE mirror,
laion vocal-burst sets, Nonspeech7k, ...) could be downloaded. Everything below comes from openslr.org
and GitHub, which work. If HF becomes reachable, the most valuable additions would be VocalSound
(3,365 speakers with gender: laughter, sighs, coughs, throat clearing, sneezes, sniffs) and
NonverbalTTS (VoxCeleb/Expresso clips with groans, grunts, breathing, laughter and speaker gender).

## Sources

| id | corpus | what we use | labels used | speakers (F/M) | license |
|---|---|---|---|---|---|
| `deeply` | Deeply Nonverbal Vocalization Dataset, [OpenSLR 99](https://www.openslr.org/99/) (public ~1 % sample) | all 727 clips, 16 classes, phone recordings, 16 kHz | class, speaker id, sex (`0=female, 1=male` per the dataset PDF), noise | 408 usable (2 ids with conflicting sex dropped) | **CC BY-NC-ND 4.0** (non-commercial, no derivatives) |
| `ears` | [EARS](https://github.com/facebookresearch/ears_dataset) (Meta, GitHub release zips, 48 kHz studio) | 10 files per speaker: screaming, crying, open laughter, yelling, yawning, coughing, freeform **pain** and **ecstasy** expressions (non-verbal), one regular and one **whispered** sentence (speech); only the first 14 s of each file are range-read, the loudest 3 s are kept | self-reported gender (`speaker_statistics.json`; 4 speakers "prefer not to answer"/non-binary skipped) | 103 (60/43) | **CC BY-NC 4.0** |
| `librispeech` | LibriSpeech dev-clean + test-clean, [OpenSLR 12](https://www.openslr.org/12/) | first 15 utterances per speaker | SPEAKERS.TXT sex | 80 (40/40) | CC BY 4.0 |
| `crowd` | Google crowdsourced multi-speaker sets, OpenSLR [83](https://www.openslr.org/83/) (UK/Irish English dialects), [61](https://www.openslr.org/61/) (es-AR), [70](https://www.openslr.org/70/) (en-NG), [86](https://www.openslr.org/86/) (Yoruba), [63](https://www.openslr.org/63/) (Malayalam), [78](https://www.openslr.org/78/) (Gujarati), [69](https://www.openslr.org/69/) (Catalan), [76](https://www.openslr.org/76/) (Basque), [66](https://www.openslr.org/66/) (Telugu) | up to 10 speakers x 4 clips per gender-zip, fetched with HTTP range requests (no full zips) | gender from the zip, speaker id from the file name | ~225, 12 languages/dialects, 48 kHz | CC BY-SA 4.0 |
| `crema` | [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (GitHub LFS) | 8 clips per actor covering all 6 emotions (anger, disgust, fear, happy, neutral, sad; incl. shouted/high-intensity takes) | VideoDemographics.csv sex | 91 (43/48) | ODbL 1.0 / DbCL 1.0 |
| `audiomnist` | [AudioMNIST](https://github.com/soerenab/AudioMNIST) | 6 isolated spoken digits per speaker (~0.4-0.7 s) | audioMNIST_meta.txt gender | 60 (12/48) | MIT |
| music | librosa example audio ([librosa/data](https://github.com/librosa/data)) | 7 instrumental tracks (~6 min) + 1 song with female vocals (2.2 min, hard case, kept separate) | - | - | CC BY 3.0 (Kevin MacLeod x2, Karissa Hobbs, Mihai Sorohan), CC BY-NC 3.0 (Setuniman, Lena Orsa, Admiral Bob), Public Domain (Brahms / US Army Strings) |
| noise | [ESC-50](https://github.com/karolpiczak/ESC-50) | 4 clips each of 20 non-human classes (rain, wind, engine, vacuum, keyboard, birds, dog, ...) | - | - | CC BY-NC 3.0 (dataset); per-clip CC0/CC BY |
| synthetic | generated in `build_dataset.py` | pink/brown noise, 50/60 Hz hum, slow modulation (room tone / AC), synthetic exponential-decay RIRs | - | - | - |

Licensing consequences: EARS, Deeply, ESC-50 and some music clips are **non-commercial**; Deeply is
additionally **no-derivatives**. The gender head is trained on EARS + Deeply, so **the head is for
personal, non-commercial use only**. We only use them locally for evaluation and to fit a small classifier
head; the mixes built from them must not be redistributed, and the trained head weights
(`voicefilter/assets/gender_head*.npz`) are a derived artifact of NC-ND data — fine for this private,
non-commercial tool, but do not ship them commercially. A commercially clean head can be trained with
`eval/train_gender_head.py` after removing `deeply` (speech-only head, see the ablation in
`eval/GENDER_MODELS.md`).

## Splits (speaker-disjoint)

`speaker -> test` iff `sha1("voicefilter-split-v1:" + speaker) < test_fraction[source]` (0.5 for Deeply,
0.5 for EARS, 0.4 otherwise). No speaker is in both splits; the trained heads use only `train` speakers; the
synthetic mixes and every number in `GENDER_MODELS.md` / `REPORT.md` use only `test` speakers.

Clips in the manifest per split (feature extraction then caps speech at 2-3 clips per speaker, see `eval/extract_features.py`; non-phonated = mouth/teeth/nose noises):

| source | kind | train F | train M | test F | test M | speakers train F/M, test F/M |
|---|---|---|---|---|---|---|
| audiomnist | speech | 36 | 173 | 36 | 114 | 6/29, 6/19 |
| crema | speech | 168 | 240 | 176 | 144 | 21/30, 22/18 |
| crowd | speech | 232 | 256 | 152 | 168 | 58/64, 38/42 |
| deeply | nonphonated | 98 | 46 | 106 | 30 | 131/74, 154/49 |
| deeply | nonverbal | 158 | 73 | 165 | 45 | 131/74, 154/49 |
| ears | nonverbal | 216 | 192 | 264 | 152 | 27/24, 33/19 |
| ears | speech | 54 | 48 | 66 | 38 | 27/24, 33/19 |
| librispeech | speech | 405 | 390 | 195 | 210 | 27/26, 13/14 |

Phonated categories: moaning (45 clips total), screaming (51), panting (44), sighing (51), crying (24),
laughing (25), yawning (48), coughing (50), throat clearing (54), sneezing (52). Non-phonated:
teeth-chattering, teeth-grinding, tongue-clicking, lip-popping, lip-smacking, nose-blowing.

## Coverage gaps (honest list)

- **Grunts and effort sounds are not a labelled category anywhere we could reach.** The closest are EARS
  freeform *pain* (groans, moans, cries of pain) and *ecstasy* (pleasure moans, sighs, exclamations; may
  contain words) with ~100 speakers, and Deeply "moaning" (45 clips, Korean phone recordings, mostly
  pain/discomfort). Grunts during physical effort, whimpers and gasps have no dedicated labels.
- Male non-verbal clips: Deeply has few (45 phonated in test); EARS adds ~43 male speakers x 8 sounds.
- Deeply speakers are Korean adults on Android phones; EARS speakers are mostly US English studio
  recordings. Speech is read or acted. No spontaneous conversational speech, no children.
- In the synthetic scenes about half of the speakers are EARS speakers whose speech and non-verbal sounds
  come from the same person; the others pair a speech speaker with non-verbal clips of other people of the
  same gender, so speaker-level consistency is weaker than in real videos for those.
- EARS speech is only two 3 s sentences per speaker (one whispered), so EARS speakers' speech turns repeat.
- Whispered speech: only the EARS whispered sentence; breathy sounds: sighing / panting / yawning.
- Reverb is synthetic (exponential decay noise, RT60 0.3-0.9 s); real room responses (SLR 28) were not
  downloaded to save space.

## Synthetic evaluation mixes (`eval/build_dataset.py`, seed 2024)

44.1 kHz stereo, test speakers only; per mix `mix.wav`, `mix.mp4` (testsrc2 + AAC), `vocals_female.wav`, `vocals_male.wav`, `background.wav` (float32, mix = exact sum) and `truth.json`. Overlap = share of vocal time with male and female active at once.

| mix | s | speakers | events (non-verbal / short) | overlap | background | SNR dB | reverb | hard cases |
|---|---|---|---|---|---|---|---|---|
| smoke | 10 | mixed (F M) | 4 (2 / 1) | 7 % | esc_noise | -4.34 | False | - |
| mix_00 | 64 | mixed (F M M M) | 18 (8 / 2) | 7 % | music | 10.73 | False | - |
| mix_01 | 114 | mixed (F M M) | 37 (15 / 2) | 17 % | esc_noise | 0.8 | True | - |
| mix_02 | 95 | mixed (F M) | 24 (6 / 3) | 10 % | synthetic_noise | 1.42 | False | low_pitched_female:gc:en-GB-welsh:06136:96Hz |
| mix_03 | 99 | mixed (F F M) | 27 (13 / 3) | 13 % | music | 7.96 | False | high_pitched_male:cr:1005:259Hz |
| mix_04 | 64 | F (F F F F) | 12 (5 / 0) | 0 % | esc_noise | 8.36 | True | - |
| mix_05 | 97 | M (M M) | 26 (9 / 2) | 0 % | synthetic_noise | -1.32 | False | - |
| mix_06 | 89 | mixed (F M) | 27 (11 / 2) | 21 % | music | 4.37 | False | low_pitched_female:gc:en-GB-welsh:06136:96Hz |
| mix_07 | 72 | mixed (F M) | 19 (11 / 2) | 10 % | esc_noise | 9.63 | True | high_pitched_male:cr:1005:259Hz |
| mix_08 | 69 | mixed (F M) | 26 (9 / 3) | 16 % | synthetic_noise | 8.93 | False | - |
| mix_09 | 67 | mixed (F M M) | 27 (15 / 1) | 18 % | music | -2.93 | False | - |
| mix_10 | 111 | F (F F) | 26 (17 / 3) | 0 % | esc_noise | 9.3 | True | low_pitched_female:gc:en-GB-welsh:06136:96Hz |
| mix_11 | 79 | M (M M) | 23 (16 / 2) | 0 % | music_vocals | 2.85 | False | high_pitched_male:cr:1005:259Hz, background_with_sung_vocals |

Note: the "low-pitched female" (gc:en-GB-welsh:06136, median F0 96 Hz by pyin) may be creaky voice or a pitch-tracking artefact; the "high-pitched male" is CREMA-D actor 1005 (259 Hz, acted emotional speech).
