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
| `librispeech` | LibriSpeech dev-clean + test-clean, [OpenSLR 12](https://www.openslr.org/12/) | first 15 utterances per speaker | SPEAKERS.TXT sex | 80 (40/40) | CC BY 4.0 |
| `crowd` | Google crowdsourced multi-speaker sets, OpenSLR [83](https://www.openslr.org/83/) (UK/Irish English dialects), [61](https://www.openslr.org/61/) (es-AR), [70](https://www.openslr.org/70/) (en-NG), [86](https://www.openslr.org/86/) (Yoruba), [63](https://www.openslr.org/63/) (Malayalam), [78](https://www.openslr.org/78/) (Gujarati), [69](https://www.openslr.org/69/) (Catalan), [76](https://www.openslr.org/76/) (Basque), [66](https://www.openslr.org/66/) (Telugu) | up to 10 speakers x 4 clips per gender-zip, fetched with HTTP range requests (no full zips) | gender from the zip, speaker id from the file name | ~225, 12 languages/dialects, 48 kHz | CC BY-SA 4.0 |
| `crema` | [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (GitHub LFS) | 8 clips per actor covering all 6 emotions (anger, disgust, fear, happy, neutral, sad; incl. shouted/high-intensity takes) | VideoDemographics.csv sex | 91 (43/48) | ODbL 1.0 / DbCL 1.0 |
| `audiomnist` | [AudioMNIST](https://github.com/soerenab/AudioMNIST) | 6 isolated spoken digits per speaker (~0.4-0.7 s) | audioMNIST_meta.txt gender | 60 (12/48) | MIT |
| music | librosa example audio ([librosa/data](https://github.com/librosa/data)) | 7 instrumental tracks (~6 min) + 1 song with female vocals (2.2 min, hard case, kept separate) | - | - | CC BY 3.0 (Kevin MacLeod x2, Karissa Hobbs, Mihai Sorohan), CC BY-NC 3.0 (Setuniman, Lena Orsa, Admiral Bob), Public Domain (Brahms / US Army Strings) |
| noise | [ESC-50](https://github.com/karolpiczak/ESC-50) | 4 clips each of 20 non-human classes (rain, wind, engine, vacuum, keyboard, birds, dog, ...) | - | - | CC BY-NC 3.0 (dataset); per-clip CC0/CC BY |
| synthetic | generated in `build_dataset.py` | pink/brown noise, 50/60 Hz hum, slow modulation (room tone / AC), synthetic exponential-decay RIRs | - | - | - |

Licensing consequences: the Deeply, ESC-50 and some music clips are **non-commercial**; Deeply is
additionally **no-derivatives**. We only use them locally for evaluation and to fit a small classifier
head; the mixes built from them must not be redistributed, and the trained head weights
(`voicefilter/assets/gender_head*.npz`) are a derived artifact of NC-ND data — fine for this private,
non-commercial tool, but do not ship them commercially. A commercially clean head can be trained with
`eval/train_gender_head.py` after removing `deeply` (speech-only head, see the ablation in
`eval/GENDER_MODELS.md`).

## Splits (speaker-disjoint)

`speaker -> test` iff `sha1("voicefilter-split-v1:" + speaker) < test_fraction[source]` (0.5 for Deeply,
0.4 otherwise). No speaker is in both splits; the trained heads use only `train` speakers; the
synthetic mixes and every number in `GENDER_MODELS.md` / `REPORT.md` use only `test` speakers.

Non-verbal clips per split (phonated = vocal-fold sounds; non-phonated = mouth/teeth/nose noises):

| split | phonated F | phonated M | non-phonated F | non-phonated M |
|---|---|---|---|---|
| train | 158 | 73 | 98 | 46 |
| test | 165 | 45 | 106 | 30 |

Phonated categories: moaning (45 clips total), screaming (51), panting (44), sighing (51), crying (24),
laughing (25), yawning (48), coughing (50), throat clearing (54), sneezing (52). Non-phonated:
teeth-chattering, teeth-grinding, tongue-clicking, lip-popping, lip-smacking, nose-blowing.

## Coverage gaps (honest list)

- **Grunts, groans and effort sounds are missing as a labelled category.** Deeply "moaning" (Korean
  crowd-sourced, mostly pain/discomfort moans) is the closest; there are only 45 moaning clips, 27 F / 18 M,
  i.e. about 20 per split. Sexual moans, grunts during physical effort, whimpers and gasps are not covered
  by any reachable labelled corpus. Numbers for "non-verbal" are therefore dominated by
  screaming / sighing / panting / yawning / coughing.
- **Male non-verbal clips are scarce** (45 phonated in test). Male-side accuracies have wide confidence
  intervals (±7-10 % at 95 %).
- **Non-verbal speakers are Korean adults recorded on Android phones**; speech is read or acted speech.
  No spontaneous conversational speech, no children.
- In the synthetic scenes a "speaker" pairs a speech speaker with non-verbal clips of other people of the
  same gender (no corpus has both from one person), so speaker-level consistency between a person's speech
  and their moans is weaker than in real videos (this makes speaker clustering harder, not easier).
- Whispered speech is not covered; breathy sounds come only from sighing / panting / yawning.
- Reverb is synthetic (exponential decay noise, RT60 0.3-0.9 s); real room responses (SLR 28) were not
  downloaded to save space.
